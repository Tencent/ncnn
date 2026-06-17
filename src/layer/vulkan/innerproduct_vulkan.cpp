// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "innerproduct_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"
#include "modelbin.h"

#include <string.h>

namespace ncnn {

InnerProduct_vulkan::InnerProduct_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    flatten = 0;

    pipeline_innerproduct = 0;
    pipeline_innerproduct_sum8 = 0;
    pipeline_innerproduct_reduce_sum8 = 0;
    pipeline_innerproduct_gemm = 0;

#if NCNN_INT8
    quantize = 0;

    pipeline_innerproduct_int8 = 0;
    pipeline_innerproduct_sum8_int8 = 0;
    pipeline_innerproduct_reduce_sum8_int8 = 0;
    pipeline_innerproduct_gemm_int8 = 0;
#endif
}

int InnerProduct_vulkan::create_pipeline(const Option& opt)
{
#if NCNN_INT8
    if (int8_scale_term && weight_data.elemsize == (size_t)1u)
    {
        return create_pipeline_int8(opt);
    }
#endif

    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    const int num_input = weight_data_size / num_output;

    int in_elempack = num_input % 4 == 0 ? 4 : 1;
    int out_elempack = num_output % 4 == 0 ? 4 : 1;

    // src = inch-outch
    // dst = pa-pb-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_packed.create(num_input / in_elempack, num_output / out_elempack, (size_t)4 * in_elempack * out_elempack, in_elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            float* g00 = weight_data_packed.row(q / out_elempack);

            for (int p = 0; p + (in_elempack - 1) < num_input; p += in_elempack)
            {
                for (int i = 0; i < out_elempack; i++)
                {
                    const float* k0 = weight_data_r2.row(q + i);
                    k0 += p;

                    for (int j = 0; j < in_elempack; j++)
                    {
                        g00[0] = k0[j];

                        g00++;
                    }
                }
            }
        }
    }

    if (shape.dims == 2 && shape.w == num_input)
    {
        // gemm
        Mat shape_unpacked(shape.w, shape.h * shape.elempack, (void*)0);
        Mat out_shape_unpacked(out_shape.w, out_shape.h * out_shape.elempack, (void*)0);

        std::vector<vk_specialization_type> specializations(4 + 10);
        specializations[0].i = bias_term;
        specializations[1].i = activation_type;
        specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
        specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
        specializations[4 + 0].i = shape_unpacked.dims;
        specializations[4 + 1].i = shape_unpacked.w;
        specializations[4 + 2].i = shape_unpacked.h;
        specializations[4 + 3].i = shape_unpacked.c;
        specializations[4 + 4].i = shape_unpacked.cstep;
        specializations[4 + 5].i = out_shape_unpacked.dims;
        specializations[4 + 6].i = out_shape_unpacked.w;
        specializations[4 + 7].i = out_shape_unpacked.h;
        specializations[4 + 8].i = out_shape_unpacked.c;
        specializations[4 + 9].i = out_shape_unpacked.cstep;

        Mat local_size_xyz(std::min(16, num_output / out_elempack), 4, 1, (void*)0);
        if (out_shape_unpacked.dims != 0)
        {
            local_size_xyz.w = std::min(16, out_shape_unpacked.w);
            local_size_xyz.h = std::min(4, out_shape_unpacked.h);
            local_size_xyz.c = 1;
        }

        int shader_type_index = -1;
        if (in_elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::innerproduct_gemm;
        if (in_elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_gemm_wp4;
        if (in_elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_gemm_wp1to4;
        if (in_elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::innerproduct_gemm_wp4to1;

        pipeline_innerproduct_gemm = new Pipeline(vkdev);
        pipeline_innerproduct_gemm->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_gemm->create(shader_type_index, opt, specializations);

        if (opt.lightmode)
        {
            weight_data.release();
        }

        return 0;
    }

    size_t elemsize;
    size_t out_elemsize;
    if (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed)
    {
        elemsize = in_elempack * 2u;
        out_elemsize = out_elempack * 2u;
    }
    else
    {
        elemsize = in_elempack * 4u;
        out_elemsize = out_elempack * 4u;
    }

    Mat shape_flatten;
    if (shape.dims != 0)
    {
        shape_flatten = Mat(shape.w * shape.h * shape.d * shape.c * shape.elempack / in_elempack, (void*)0, elemsize, in_elempack);
    }

    {
        flatten = ncnn::create_layer_vulkan(ncnn::LayerType::Flatten);
        flatten->vkdev = vkdev;

        flatten->bottom_shapes.resize(1);
        flatten->bottom_shapes[0] = shape;
        flatten->top_shapes.resize(1);
        flatten->top_shapes[0] = shape_flatten;

        ncnn::ParamDict pd;

        flatten->load_param(pd);

        flatten->create_pipeline(opt);
    }

    if (num_input / in_elempack >= 32)
    {
        Mat out_sum8_shape((num_input / in_elempack + 7) / 8, num_output / out_elempack, (void*)0, out_elemsize, out_elempack);

        // sum8
        {
            std::vector<vk_specialization_type> specializations(0 + 3);
            specializations[0 + 0].i = shape_flatten.w;
            specializations[0 + 1].i = out_sum8_shape.w;
            specializations[0 + 2].i = out_sum8_shape.h;

            int shader_type_index = -1;
            if (in_elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::innerproduct_sum8;
            if (in_elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_sum8_pack4;
            if (in_elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_sum8_pack1to4;
            if (in_elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::innerproduct_sum8_pack4to1;

            pipeline_innerproduct_sum8 = new Pipeline(vkdev);
            pipeline_innerproduct_sum8->set_local_size_xyz(8, std::min(8, num_output / out_elempack), 1);
            pipeline_innerproduct_sum8->create(shader_type_index, opt, specializations);
        }

        // reduce sum8
        {
            std::vector<vk_specialization_type> specializations(4 + 3);
            specializations[0].i = bias_term;
            specializations[1].i = activation_type;
            specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[4 + 0].i = out_sum8_shape.w;
            specializations[4 + 1].i = out_sum8_shape.h;
            specializations[4 + 2].i = out_shape.w;

            int shader_type_index = -1;
            if (out_elempack == 1) shader_type_index = LayerShaderType::innerproduct_reduce_sum8;
            if (out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_reduce_sum8_pack4;

            pipeline_innerproduct_reduce_sum8 = new Pipeline(vkdev);
            pipeline_innerproduct_reduce_sum8->set_local_size_xyz(std::min(64, num_output / out_elempack), 1, 1);
            pipeline_innerproduct_reduce_sum8->create(shader_type_index, opt, specializations);
        }
    }
    else
    {
        std::vector<vk_specialization_type> specializations(4 + 10);
        specializations[0].i = bias_term;
        specializations[1].i = activation_type;
        specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
        specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
        specializations[4 + 0].i = shape_flatten.dims;
        specializations[4 + 1].i = shape_flatten.w;
        specializations[4 + 2].i = shape_flatten.h;
        specializations[4 + 3].i = shape_flatten.c;
        specializations[4 + 4].i = shape_flatten.cstep;
        specializations[4 + 5].i = out_shape.dims;
        specializations[4 + 6].i = out_shape.w;
        specializations[4 + 7].i = out_shape.h;
        specializations[4 + 8].i = out_shape.c;
        specializations[4 + 9].i = out_shape.cstep;

        Mat local_size_xyz(std::min(64, num_output / out_elempack), 1, 1, (void*)0);
        if (out_shape.dims != 0)
        {
            local_size_xyz.w = std::min(64, out_shape.w);
            local_size_xyz.h = 1;
            local_size_xyz.c = 1;
        }

        int shader_type_index = -1;
        if (in_elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::innerproduct;
        if (in_elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_pack4;
        if (in_elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_pack1to4;
        if (in_elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::innerproduct_pack4to1;

        pipeline_innerproduct = new Pipeline(vkdev);
        pipeline_innerproduct->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct->create(shader_type_index, opt, specializations);
    }

    // gemm for no shape hint
    if (shape.dims == 0)
    {
        std::vector<vk_specialization_type> specializations(4 + 10);
        specializations[0].i = bias_term;
        specializations[1].i = activation_type;
        specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
        specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
        specializations[4 + 0].i = 0;
        specializations[4 + 1].i = 0;
        specializations[4 + 2].i = 0;
        specializations[4 + 3].i = 0;
        specializations[4 + 4].i = 0;
        specializations[4 + 5].i = 0;
        specializations[4 + 6].i = 0;
        specializations[4 + 7].i = 0;
        specializations[4 + 8].i = 0;
        specializations[4 + 9].i = 0;

        Mat local_size_xyz(std::min(16, num_output / out_elempack), 4, 1, (void*)0);

        int shader_type_index = -1;
        if (in_elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::innerproduct_gemm;
        if (in_elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_gemm_wp4;
        if (in_elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_gemm_wp1to4;
        if (in_elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::innerproduct_gemm_wp4to1;

        pipeline_innerproduct_gemm = new Pipeline(vkdev);
        pipeline_innerproduct_gemm->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_gemm->create(shader_type_index, opt, specializations);

        if (opt.lightmode)
        {
            weight_data.release();
        }

        return 0;
    }

    if (opt.lightmode)
    {
        weight_data.release();
    }

    return 0;
}

int InnerProduct_vulkan::destroy_pipeline(const Option& opt)
{
    if (flatten)
    {
        flatten->destroy_pipeline(opt);
        delete flatten;
        flatten = 0;
    }
    delete pipeline_innerproduct;
    pipeline_innerproduct = 0;

    delete pipeline_innerproduct_sum8;
    delete pipeline_innerproduct_reduce_sum8;
    pipeline_innerproduct_sum8 = 0;
    pipeline_innerproduct_reduce_sum8 = 0;

    delete pipeline_innerproduct_gemm;
    pipeline_innerproduct_gemm = 0;

#if NCNN_INT8
    if (quantize)
    {
        quantize->destroy_pipeline(opt);
        delete quantize;
        quantize = 0;
    }

    delete pipeline_innerproduct_int8;
    pipeline_innerproduct_int8 = 0;

    delete pipeline_innerproduct_sum8_int8;
    delete pipeline_innerproduct_reduce_sum8_int8;
    pipeline_innerproduct_sum8_int8 = 0;
    pipeline_innerproduct_reduce_sum8_int8 = 0;

    delete pipeline_innerproduct_gemm_int8;
    pipeline_innerproduct_gemm_int8 = 0;
#endif

    return 0;
}

int InnerProduct_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
#if NCNN_INT8
    if (int8_scale_term && weight_data_int8_packed.elembits() == 8)
    {
        return upload_model_int8(cmd, opt);
    }
#endif

    cmd.record_upload(weight_data_packed, weight_data_gpu, opt);

    weight_data_packed.release();

    if (bias_term)
    {
        cmd.record_upload(bias_data, bias_data_gpu, opt);

        bias_data.release();
    }

    return 0;
}

int InnerProduct_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        return forward_int8(bottom_blob, top_blob, cmd, opt);
    }
#endif

    const int num_input = weight_data_size / num_output;

    int in_elempack = num_input % 4 == 0 ? 4 : 1;
    int out_elempack = num_output % 4 == 0 ? 4 : 1;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input)
    {
        // gemm
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        int elempack = bottom_blob.elempack;

        // unpacking
        VkMat bottom_blob_unpacked = bottom_blob;
        if (elempack > 1)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

            vkdev->convert_packing(bottom_blob, bottom_blob_unpacked, 1, cmd, opt_pack1);
        }

        top_blob.create(num_output, h, elemsize, elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        VkMat top_blob_unpacked = top_blob;
        if (elempack > 1)
        {
            top_blob_unpacked.create(num_output, h * elempack, bottom_blob_unpacked.elemsize, 1, opt.workspace_vkallocator);
            if (top_blob_unpacked.empty())
                return -100;
        }

        std::vector<VkMat> bindings(4);
        bindings[0] = bottom_blob_unpacked;
        bindings[1] = top_blob_unpacked;
        bindings[2] = weight_data_gpu;
        bindings[3] = bias_data_gpu;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob_unpacked.dims;
        constants[1].i = bottom_blob_unpacked.w;
        constants[2].i = bottom_blob_unpacked.h;
        constants[3].i = bottom_blob_unpacked.c;
        constants[4].i = bottom_blob_unpacked.cstep;
        constants[5].i = top_blob_unpacked.dims;
        constants[6].i = top_blob_unpacked.w;
        constants[7].i = top_blob_unpacked.h;
        constants[8].i = top_blob_unpacked.c;
        constants[9].i = top_blob_unpacked.cstep;

        VkMat dispatcher;
        dispatcher.w = top_blob_unpacked.w / out_elempack;
        dispatcher.h = top_blob_unpacked.h;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_innerproduct_gemm, bindings, constants, dispatcher);

        // packing
        if (elempack > 1)
        {
            vkdev->convert_packing(top_blob_unpacked, top_blob, elempack, cmd, opt);
        }

        return 0;
    }

    // flatten
    VkMat bottom_blob_flattened = bottom_blob;
    {
        Option opt_flatten = opt;
        opt_flatten.blob_vkallocator = opt.workspace_vkallocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, cmd, opt_flatten);
    }

    size_t elemsize = bottom_blob_flattened.elemsize;
    size_t out_elemsize = elemsize / in_elempack * out_elempack;

    if (num_input / in_elempack >= 32)
    {
        // sum8
        VkMat top_blob_sum8;
        {
            top_blob_sum8.create((num_input / in_elempack + 7) / 8, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
            if (top_blob_sum8.empty())
                return -100;

            std::vector<VkMat> bindings(3);
            bindings[0] = bottom_blob_flattened;
            bindings[1] = top_blob_sum8;
            bindings[2] = weight_data_gpu;

            std::vector<vk_constant_type> constants(3);
            constants[0].i = bottom_blob_flattened.w;
            constants[1].i = top_blob_sum8.w;
            constants[2].i = top_blob_sum8.h;

            cmd.record_pipeline(pipeline_innerproduct_sum8, bindings, constants, top_blob_sum8);
        }

        // reduce sum8
        {
            top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
            if (top_blob.empty())
                return -100;

            std::vector<VkMat> bindings(3);
            bindings[0] = top_blob_sum8;
            bindings[1] = top_blob;
            bindings[2] = bias_data_gpu;

            std::vector<vk_constant_type> constants(3);
            constants[0].i = top_blob_sum8.w;
            constants[1].i = top_blob_sum8.h;
            constants[2].i = top_blob.w;

            cmd.record_pipeline(pipeline_innerproduct_reduce_sum8, bindings, constants, top_blob);
        }
    }
    else
    {
        top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        std::vector<VkMat> bindings(4);
        bindings[0] = bottom_blob_flattened;
        bindings[1] = top_blob;
        bindings[2] = weight_data_gpu;
        bindings[3] = bias_data_gpu;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob_flattened.dims;
        constants[1].i = bottom_blob_flattened.w;
        constants[2].i = bottom_blob_flattened.h;
        constants[3].i = bottom_blob_flattened.c;
        constants[4].i = bottom_blob_flattened.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = top_blob.cstep;

        cmd.record_pipeline(pipeline_innerproduct, bindings, constants, top_blob);
    }

    return 0;
}

#if NCNN_INT8
int InnerProduct_vulkan::create_pipeline_int8(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    const int num_input = weight_data_size / num_output;

    const int num_input_packed = (num_input + 7) / 8 * 8;
    const int num_output_packed = (num_output + 3) / 4 * 4;

    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_int8_packed.create(num_input_packed / 4, num_output_packed / 4, (size_t)16u, 16);
        if (weight_data_int8_packed.empty())
            return -100;
        memset(weight_data_int8_packed.data, 0, weight_data_int8_packed.total() * weight_data_int8_packed.elemsize);

        for (int q = 0; q < num_output_packed; q += 4)
        {
            signed char* g00 = weight_data_int8_packed.row<signed char>(q / 4);

            for (int p = 0; p < num_input_packed; p += 4)
            {
                for (int i = 0; i < 4; i++)
                {
                    const signed char* k0 = q + i < num_output && p < num_input ? weight_data_r2.row<const signed char>(q + i) + p : 0;

                    for (int j = 0; j < 4; j++)
                    {
                        g00[0] = k0 && p + j < num_input ? k0[j] : 0;
                        g00++;
                    }
                }
            }
        }
    }

    {
        const float bottom_blob_int8_scale = bottom_blob_int8_scales.empty() ? 1.f : bottom_blob_int8_scales[0];
        const float bottom_blob_int8_descale = bottom_blob_int8_scale == 0.f ? 0.f : 1.f / bottom_blob_int8_scale;

        weight_data_int8_descales.create(num_output_packed / 4, (size_t)4u * 4, 4);
        if (weight_data_int8_descales.empty())
            return -100;
        memset(weight_data_int8_descales.data, 0, weight_data_int8_descales.total() * weight_data_int8_descales.elemsize);

        float* outptr = weight_data_int8_descales;
        for (int q = 0; q < num_output; q++)
        {
            float scale = weight_data_int8_scales[q];
            outptr[q] = scale == 0.f ? 0.f : bottom_blob_int8_descale / scale;
        }
    }

    if (bias_term)
    {
        bias_data_int8_packed.create(num_output_packed / 4, (size_t)4u * 4, 4);
        if (bias_data_int8_packed.empty())
            return -100;
        memset(bias_data_int8_packed.data, 0, bias_data_int8_packed.total() * bias_data_int8_packed.elemsize);

        float* outptr = bias_data_int8_packed;
        for (int q = 0; q < num_output; q++)
        {
            outptr[q] = bias_data[q];
        }
    }

    Option opt_int8 = opt;
    opt_int8.use_fp16_arithmetic = false;
    opt_int8.use_int16_packed = false;
    opt_int8.use_int16_storage = false;

    {
        quantize = ncnn::create_layer_vulkan(ncnn::LayerType::Quantize);
        quantize->vkdev = vkdev;

        Mat shape_quantize;
        Mat out_shape_quantize;
        if (shape.dims == 2 && shape.w == num_input)
        {
            const size_t elemsize = shape.elemsize / shape.elempack;
            shape_quantize = Mat(shape.w, shape.h, (void*)0, elemsize * shape.elempack, shape.elempack);
            out_shape_quantize = Mat(shape.w, shape.h, (void*)0, (size_t)shape.elempack, shape.elempack);
        }
        else if (shape.dims != 0)
        {
            const int total = shape.w * shape.h * shape.d * shape.c * shape.elempack;
            const int flatten_elempack = total % 4 == 0 ? 4 : 1;
            const size_t elemsize = shape.elemsize / shape.elempack;
            shape_quantize = Mat(total / flatten_elempack, (void*)0, elemsize * flatten_elempack, flatten_elempack);
            out_shape_quantize = Mat(total / flatten_elempack, (void*)0, (size_t)flatten_elempack, flatten_elempack);
        }

        quantize->bottom_shapes.resize(1);
        quantize->bottom_shapes[0] = shape_quantize;
        quantize->top_shapes.resize(1);
        quantize->top_shapes[0] = out_shape_quantize;

        ncnn::ParamDict pd;
        pd.set(0, 1);
        quantize->load_param(pd);

        Mat weights[1];
        weights[0] = bottom_blob_int8_scales;
        quantize->load_model(ModelBinFromMatArray(weights));

        Option opt_quantize = opt;
        opt_quantize.use_fp16_arithmetic = false;

        quantize->create_pipeline(opt_quantize);
    }

    if (shape.dims == 2 && shape.w == num_input)
    {
        // gemm
        Mat shape_unpacked(num_input, shape.h * shape.elempack, (void*)0);
        Mat out_shape_unpacked(num_output, out_shape.dims == 0 ? 0 : out_shape.h * out_shape.elempack, (void*)0);

        std::vector<vk_specialization_type> specializations(5 + 5);
        specializations[0].i = bias_term;
        specializations[1].i = activation_type;
        specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
        specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
        specializations[4].i = num_input_packed / 4;
        specializations[5 + 0].i = shape_unpacked.w;
        specializations[5 + 1].i = shape.elempack;
        specializations[5 + 2].i = shape.w;
        specializations[5 + 3].i = out_shape_unpacked.w;
        specializations[5 + 4].i = out_shape_unpacked.h;

        Mat local_size_xyz(std::min(16, (num_output + 3) / 4), 4, 1, (void*)0);
        if (out_shape_unpacked.dims != 0)
        {
            local_size_xyz.w = std::min(16, (out_shape_unpacked.w + 3) / 4);
            local_size_xyz.h = std::min(4, out_shape_unpacked.h);
            local_size_xyz.c = 1;
        }

        pipeline_innerproduct_gemm_int8 = new Pipeline(vkdev);
        if (opt.use_shader_local_memory)
        {
            pipeline_innerproduct_gemm_int8->set_local_size_xyz(8, 8, 1);
        }
        else
        {
            pipeline_innerproduct_gemm_int8->set_optimal_local_size_xyz(local_size_xyz);
        }
        pipeline_innerproduct_gemm_int8->create(LayerShaderType::innerproduct_gemm_int8, opt_int8, specializations);

        if (opt.lightmode)
        {
            weight_data.release();
        }

        return 0;
    }

    size_t elemsize;
    if (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed)
    {
        elemsize = 2u;
    }
    else
    {
        elemsize = 4u;
    }

    Mat shape_flatten;
    if (shape.dims != 0)
    {
        const int total = shape.w * shape.h * shape.d * shape.c * shape.elempack;
        const int flatten_elempack = total % 4 == 0 ? 4 : 1;
        shape_flatten = Mat(total / flatten_elempack, (void*)0, elemsize * flatten_elempack, flatten_elempack);
    }

    {
        flatten = ncnn::create_layer_vulkan(ncnn::LayerType::Flatten);
        flatten->vkdev = vkdev;

        flatten->bottom_shapes.resize(1);
        flatten->bottom_shapes[0] = shape;
        flatten->top_shapes.resize(1);
        flatten->top_shapes[0] = shape_flatten;

        ncnn::ParamDict pd;

        flatten->load_param(pd);

        flatten->create_pipeline(opt);
    }

    if (num_input_packed / 4 >= 32)
    {
        const int outw_sum8 = (num_input_packed / 4 + 7) / 8;
        const int outh_sum8 = num_output_packed / 4;

        // sum8
        {
            std::vector<vk_specialization_type> specializations(1 + 3);
            specializations[0].i = num_input_packed / 4;
            specializations[1 + 0].i = shape_flatten.w * shape_flatten.elempack;
            specializations[1 + 1].i = outw_sum8;
            specializations[1 + 2].i = outh_sum8;

            pipeline_innerproduct_sum8_int8 = new Pipeline(vkdev);
            pipeline_innerproduct_sum8_int8->set_local_size_xyz(8, std::min(8, outh_sum8), 1);
            pipeline_innerproduct_sum8_int8->create(LayerShaderType::innerproduct_sum8_int8, opt_int8, specializations);
        }

        // reduce sum8
        {
            std::vector<vk_specialization_type> specializations(4 + 3);
            specializations[0].i = bias_term;
            specializations[1].i = activation_type;
            specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[4 + 0].i = outw_sum8;
            specializations[4 + 1].i = outh_sum8;
            specializations[4 + 2].i = (num_output + 3) / 4;

            pipeline_innerproduct_reduce_sum8_int8 = new Pipeline(vkdev);
            pipeline_innerproduct_reduce_sum8_int8->set_local_size_xyz(std::min(64, (num_output + 3) / 4), 1, 1);
            pipeline_innerproduct_reduce_sum8_int8->create(LayerShaderType::innerproduct_reduce_sum8_int8, opt_int8, specializations);
        }
    }
    else
    {
        std::vector<vk_specialization_type> specializations(5 + 2);
        specializations[0].i = bias_term;
        specializations[1].i = activation_type;
        specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
        specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
        specializations[4].i = num_input_packed / 4;
        specializations[5 + 0].i = shape_flatten.w * shape_flatten.elempack;
        specializations[5 + 1].i = num_output;

        Mat local_size_xyz(std::min(64, (num_output + 3) / 4), 1, 1, (void*)0);
        if (out_shape.dims != 0)
        {
            local_size_xyz.w = std::min(64, (num_output + 3) / 4);
            local_size_xyz.h = 1;
            local_size_xyz.c = 1;
        }

        pipeline_innerproduct_int8 = new Pipeline(vkdev);
        pipeline_innerproduct_int8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_int8->create(LayerShaderType::innerproduct_int8, opt_int8, specializations);
    }

    // gemm for no shape hint
    if (shape.dims == 0)
    {
        std::vector<vk_specialization_type> specializations(5 + 5);
        specializations[0].i = bias_term;
        specializations[1].i = activation_type;
        specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
        specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
        specializations[4].i = num_input_packed / 4;
        specializations[5 + 0].i = 0;
        specializations[5 + 1].i = 0;
        specializations[5 + 2].i = 0;
        specializations[5 + 3].i = 0;
        specializations[5 + 4].i = 0;

        Mat local_size_xyz(std::min(16, (num_output + 3) / 4), 4, 1, (void*)0);

        pipeline_innerproduct_gemm_int8 = new Pipeline(vkdev);
        if (opt.use_shader_local_memory)
        {
            pipeline_innerproduct_gemm_int8->set_local_size_xyz(8, 8, 1);
        }
        else
        {
            pipeline_innerproduct_gemm_int8->set_optimal_local_size_xyz(local_size_xyz);
        }
        pipeline_innerproduct_gemm_int8->create(LayerShaderType::innerproduct_gemm_int8, opt_int8, specializations);
    }

    if (opt.lightmode)
    {
        weight_data.release();
    }

    return 0;
}

int InnerProduct_vulkan::upload_model_int8(VkTransfer& cmd, const Option& opt)
{
    cmd.record_upload(weight_data_int8_packed, weight_data_gpu, opt);

    weight_data_int8_packed.release();

    cmd.record_upload(weight_data_int8_descales, weight_data_int8_descales_gpu, opt);

    weight_data_int8_descales.release();
    weight_data_int8_scales.release();

    if (bias_term)
    {
        cmd.record_upload(bias_data_int8_packed, bias_data_gpu, opt);

        bias_data_int8_packed.release();
        bias_data.release();
    }

    quantize->upload_model(cmd, opt);

    return 0;
}

int InnerProduct_vulkan::forward_int8(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    const int num_input = weight_data_size / num_output;
    const int out_elempack = num_output % 4 == 0 ? 4 : 1;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input)
    {
        // gemm
        VkMat bottom_blob_quantized = bottom_blob;

        if (bottom_blob_quantized.elembits() != 8)
        {
            Option opt_quantize = opt;
            opt_quantize.blob_vkallocator = opt.workspace_vkallocator;
            opt_quantize.use_fp16_arithmetic = false;

            VkMat bottom_blob_int8;
            quantize->forward(bottom_blob_quantized, bottom_blob_int8, cmd, opt_quantize);

            bottom_blob_quantized = bottom_blob_int8;
        }

        const int h = bottom_blob_quantized.h;
        const int elempack = bottom_blob_quantized.elempack;
        const int outh = h * elempack;
        size_t out_elemsize;
        if (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed)
        {
            out_elemsize = elempack * 2u;
        }
        else
        {
            out_elemsize = elempack * 4u;
        }

        top_blob.create(num_output, h, out_elemsize, elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        std::vector<VkMat> bindings(7);
        bindings[0] = bottom_blob_quantized;
        bindings[1] = top_blob;
        bindings[2] = bottom_blob_quantized;
        bindings[3] = top_blob;
        bindings[4] = weight_data_gpu;
        bindings[5] = weight_data_int8_descales_gpu;
        bindings[6] = bias_data_gpu;

        std::vector<vk_constant_type> constants(5);
        constants[0].i = num_input;
        constants[1].i = elempack;
        constants[2].i = bottom_blob_quantized.w;
        constants[3].i = num_output;
        constants[4].i = outh;

        VkMat dispatcher;
        dispatcher.w = (num_output + 3) / 4;
        dispatcher.h = (outh + 3) / 4;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_innerproduct_gemm_int8, bindings, constants, dispatcher);

        return 0;
    }

    // flatten
    VkMat bottom_blob_flattened = bottom_blob;
    {
        Option opt_flatten = opt;
        opt_flatten.blob_vkallocator = opt.workspace_vkallocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, cmd, opt_flatten);
    }

    if (bottom_blob_flattened.elembits() != 8)
    {
        Option opt_quantize = opt;
        opt_quantize.blob_vkallocator = opt.workspace_vkallocator;
        opt_quantize.use_fp16_arithmetic = false;

        VkMat bottom_blob_int8;
        quantize->forward(bottom_blob_flattened, bottom_blob_int8, cmd, opt_quantize);

        bottom_blob_flattened = bottom_blob_int8;
    }

    size_t out_elemsize;
    if (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed)
    {
        out_elemsize = out_elempack * 2u;
    }
    else
    {
        out_elemsize = out_elempack * 4u;
    }

    const int num_input_packed = (num_input + 7) / 8 * 8;
    const int num_output_packed = (num_output + 3) / 4 * 4;

    if (num_input_packed / 4 >= 32)
    {
        // sum8
        VkMat top_blob_sum8;
        {
            top_blob_sum8.create((num_input_packed / 4 + 7) / 8, num_output_packed / 4, (size_t)4u * 4, 4, opt.blob_vkallocator);
            if (top_blob_sum8.empty())
                return -100;

            std::vector<VkMat> bindings(3);
            bindings[0] = bottom_blob_flattened;
            bindings[1] = top_blob_sum8;
            bindings[2] = weight_data_gpu;

            std::vector<vk_constant_type> constants(3);
            constants[0].i = bottom_blob_flattened.w * bottom_blob_flattened.elempack;
            constants[1].i = top_blob_sum8.w;
            constants[2].i = top_blob_sum8.h;

            cmd.record_pipeline(pipeline_innerproduct_sum8_int8, bindings, constants, top_blob_sum8);
        }

        // reduce sum8
        {
            top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
            if (top_blob.empty())
                return -100;

            std::vector<VkMat> bindings(4);
            bindings[0] = top_blob_sum8;
            bindings[1] = top_blob;
            bindings[2] = weight_data_int8_descales_gpu;
            bindings[3] = bias_data_gpu;

            std::vector<vk_constant_type> constants(3);
            constants[0].i = top_blob_sum8.w;
            constants[1].i = top_blob_sum8.h;
            constants[2].i = (num_output + 3) / 4;

            cmd.record_pipeline(pipeline_innerproduct_reduce_sum8_int8, bindings, constants, top_blob);
        }

        return 0;
    }

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(5);
    bindings[0] = bottom_blob_flattened;
    bindings[1] = top_blob;
    bindings[2] = weight_data_gpu;
    bindings[3] = weight_data_int8_descales_gpu;
    bindings[4] = bias_data_gpu;

    std::vector<vk_constant_type> constants(2);
    constants[0].i = bottom_blob_flattened.w * bottom_blob_flattened.elempack;
    constants[1].i = num_output;

    VkMat dispatcher;
    dispatcher.w = (num_output + 3) / 4;
    dispatcher.h = 1;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline_innerproduct_int8, bindings, constants, dispatcher);

    return 0;
}
#endif // NCNN_INT8

} // namespace ncnn

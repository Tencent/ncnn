// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "innerproduct_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

InnerProduct_vulkan::InnerProduct_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    flatten = 0;

    pipeline_innerproduct = 0;

    pipeline_innerproduct_sum8 = 0;
    pipeline_innerproduct_reduce_sum8 = 0;

    pipeline_innerproduct_gemm = 0;
}

int InnerProduct_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    const int num_input = weight_data_size / num_output;

    int in_elempack = opt.use_shader_pack8 && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;

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

    if (bias_term)
    {
        convert_packing(bias_data, bias_data_packed, out_elempack, opt);
    }

    if (shape.dims == 2 && shape.w == num_input)
    {
        // gemm
        int elempack = opt.use_shader_pack8 && shape.h % 8 == 0 ? 8 : shape.h % 4 == 0 ? 4 : 1;

        size_t elemsize;
        if (opt.use_fp16_storage)
        {
            elemsize = elempack * 2u;
        }
        else if (opt.use_fp16_packed)
        {
            elemsize = elempack == 1 ? 4u : elempack * 2u;
        }
        else
        {
            elemsize = elempack * 4u;
        }

        Mat shape_packed = Mat(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);
        Mat out_shape_packed = Mat(out_shape.w, out_shape.h / elempack, (void*)0, elemsize, elempack);

        // check blob shape
        if (!vkdev->shape_support_image_storage(shape) || !vkdev->shape_support_image_storage(out_shape))
        {
            support_image_storage = false;
            opt.use_image_storage = false;
        }

        // check blob shape
        if (!vkdev->shape_support_image_storage(shape_packed) || !vkdev->shape_support_image_storage(out_shape_packed))
        {
            support_image_storage = false;
            opt.use_image_storage = false;
        }

        std::vector<vk_specialization_type> specializations(4 + 10);
        specializations[0].i = bias_term;
        specializations[1].i = activation_type;
        specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
        specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
        specializations[4 + 0].i = shape.dims;
        specializations[4 + 1].i = shape.w;
        specializations[4 + 2].i = shape.h;
        specializations[4 + 3].i = shape.c;
        specializations[4 + 4].i = shape.cstep;
        specializations[4 + 5].i = out_shape.dims;
        specializations[4 + 6].i = out_shape.w;
        specializations[4 + 7].i = out_shape.h;
        specializations[4 + 8].i = out_shape.c;
        specializations[4 + 9].i = out_shape.cstep;

        Mat local_size_xyz(std::min(16, num_output / out_elempack), 4, 1, (void*)0);
        if (out_shape.dims != 0)
        {
            local_size_xyz.w = std::min(16, out_shape.w / out_elempack);
            local_size_xyz.h = std::min(4, out_shape.h);
            local_size_xyz.c = 1;
        }

        int shader_type_index = -1;
        if (in_elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::innerproduct_gemm;
        if (in_elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_gemm_wp4;
        if (in_elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_gemm_wp1to4;
        if (in_elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::innerproduct_gemm_wp4to1;
        if (in_elempack == 8 && out_elempack == 8) shader_type_index = LayerShaderType::innerproduct_gemm_wp8;
        if (in_elempack == 1 && out_elempack == 8) shader_type_index = LayerShaderType::innerproduct_gemm_wp1to8;
        if (in_elempack == 8 && out_elempack == 1) shader_type_index = LayerShaderType::innerproduct_gemm_wp8to1;
        if (in_elempack == 4 && out_elempack == 8) shader_type_index = LayerShaderType::innerproduct_gemm_wp4to8;
        if (in_elempack == 8 && out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_gemm_wp8to4;

        pipeline_innerproduct_gemm = new Pipeline(vkdev);
        pipeline_innerproduct_gemm->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_gemm->create(shader_type_index, opt, specializations);

        weight_data.release();
        bias_data.release();

        return 0;
    }

    Mat shape_flatten;
    if (shape.dims != 0)
    {
        shape_flatten = Mat(shape.w * shape.h * shape.c, (void*)0);
    }

    size_t elemsize;
    size_t out_elemsize;
    if (opt.use_fp16_storage)
    {
        elemsize = in_elempack * 2u;
        out_elemsize = out_elempack * 2u;
    }
    else if (opt.use_fp16_packed)
    {
        elemsize = in_elempack == 1 ? 4u : in_elempack * 2u;
        out_elemsize = out_elempack == 1 ? 4u : out_elempack * 2u;
    }
    else
    {
        elemsize = in_elempack * 4u;
        out_elemsize = out_elempack * 4u;
    }

    Mat shape_flatten_packed;
    if (shape_flatten.dims == 1) shape_flatten_packed = Mat(shape_flatten.w / in_elempack, (void*)0, elemsize, in_elempack);

    Mat out_shape_packed;
    if (out_shape.dims == 1) out_shape_packed = Mat(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);

    // check blob shape
    if (!vkdev->shape_support_image_storage(shape_flatten_packed) || !vkdev->shape_support_image_storage(out_shape_packed))
    {
        support_image_storage = false;
        opt.use_image_storage = false;
    }

    // check weight shape
    Mat weight_data_packed(num_input / in_elempack, num_output / out_elempack, (void*)0, (size_t)4 * in_elempack * out_elempack, in_elempack * out_elempack);
    if (!vkdev->shape_support_image_storage(weight_data_packed))
    {
        support_image_storage = false;
        opt.use_image_storage = false;
    }

    if (shape.dims == 0)
    {
        // check weight shape
        Mat weight_data_packed(num_input, num_output, (void*)0, (size_t)4u, 1);
        if (!vkdev->shape_support_image_storage(weight_data_packed))
        {
            support_image_storage = false;
            opt.use_image_storage = false;
        }
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
        Mat out_sum8_shape((num_input / in_elempack + 7) / 8, num_output, (void*)0);
        Mat out_sum8_shape_packed = Mat(out_sum8_shape.w, out_sum8_shape.h / out_elempack, (void*)0, out_elemsize, out_elempack);
        if (!vkdev->shape_support_image_storage(out_sum8_shape_packed))
        {
            support_image_storage = false;
            opt.use_image_storage = false;
        }

        // sum8
        {
            std::vector<vk_specialization_type> specializations(0 + 3);
            specializations[0 + 0].i = shape_flatten_packed.w;
            specializations[0 + 1].i = out_sum8_shape_packed.w;
            specializations[0 + 2].i = out_sum8_shape_packed.h;

            int shader_type_index = -1;
            if (in_elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::innerproduct_sum8;
            if (in_elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_sum8_pack4;
            if (in_elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_sum8_pack1to4;
            if (in_elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::innerproduct_sum8_pack4to1;
            if (in_elempack == 8 && out_elempack == 8) shader_type_index = LayerShaderType::innerproduct_sum8_pack8;
            if (in_elempack == 1 && out_elempack == 8) shader_type_index = LayerShaderType::innerproduct_sum8_pack1to8;
            if (in_elempack == 8 && out_elempack == 1) shader_type_index = LayerShaderType::innerproduct_sum8_pack8to1;
            if (in_elempack == 4 && out_elempack == 8) shader_type_index = LayerShaderType::innerproduct_sum8_pack4to8;
            if (in_elempack == 8 && out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_sum8_pack8to4;

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
            specializations[4 + 0].i = out_sum8_shape_packed.w;
            specializations[4 + 1].i = out_sum8_shape_packed.h;
            specializations[4 + 2].i = out_shape_packed.w;

            int shader_type_index = -1;
            if (out_elempack == 1) shader_type_index = LayerShaderType::innerproduct_reduce_sum8;
            if (out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_reduce_sum8_pack4;
            if (out_elempack == 8) shader_type_index = LayerShaderType::innerproduct_reduce_sum8_pack8;

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
        specializations[4 + 0].i = shape_flatten_packed.dims;
        specializations[4 + 1].i = shape_flatten_packed.w;
        specializations[4 + 2].i = shape_flatten_packed.h;
        specializations[4 + 3].i = shape_flatten_packed.c;
        specializations[4 + 4].i = shape_flatten_packed.cstep;
        specializations[4 + 5].i = out_shape_packed.dims;
        specializations[4 + 6].i = out_shape_packed.w;
        specializations[4 + 7].i = out_shape_packed.h;
        specializations[4 + 8].i = out_shape_packed.c;
        specializations[4 + 9].i = out_shape_packed.cstep;

        Mat local_size_xyz(std::min(64, num_output / out_elempack), 1, 1, (void*)0);
        if (out_shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(64, out_shape_packed.w);
            local_size_xyz.h = 1;
            local_size_xyz.c = 1;
        }

        int shader_type_index = -1;
        if (in_elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::innerproduct;
        if (in_elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_pack4;
        if (in_elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_pack1to4;
        if (in_elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::innerproduct_pack4to1;
        if (in_elempack == 8 && out_elempack == 8) shader_type_index = LayerShaderType::innerproduct_pack8;
        if (in_elempack == 1 && out_elempack == 8) shader_type_index = LayerShaderType::innerproduct_pack1to8;
        if (in_elempack == 8 && out_elempack == 1) shader_type_index = LayerShaderType::innerproduct_pack8to1;
        if (in_elempack == 4 && out_elempack == 8) shader_type_index = LayerShaderType::innerproduct_pack4to8;
        if (in_elempack == 8 && out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_pack8to4;

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
        if (in_elempack == 8 && out_elempack == 8) shader_type_index = LayerShaderType::innerproduct_gemm_wp8;
        if (in_elempack == 1 && out_elempack == 8) shader_type_index = LayerShaderType::innerproduct_gemm_wp1to8;
        if (in_elempack == 8 && out_elempack == 1) shader_type_index = LayerShaderType::innerproduct_gemm_wp8to1;
        if (in_elempack == 4 && out_elempack == 8) shader_type_index = LayerShaderType::innerproduct_gemm_wp4to8;
        if (in_elempack == 8 && out_elempack == 4) shader_type_index = LayerShaderType::innerproduct_gemm_wp8to4;

        pipeline_innerproduct_gemm = new Pipeline(vkdev);
        pipeline_innerproduct_gemm->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_gemm->create(shader_type_index, opt, specializations);

        weight_data.release();
        bias_data.release();

        return 0;
    }

    weight_data.release();
    bias_data.release();

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

    return 0;
}

int InnerProduct_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (support_image_storage && opt.use_image_storage)
    {
        cmd.record_upload(weight_data_packed, weight_data_gpu_image, opt);
    }
    else
    {
        cmd.record_upload(weight_data_packed, weight_data_gpu, opt);
    }

    weight_data_packed.release();

    if (bias_term)
    {
        if (support_image_storage && opt.use_image_storage)
        {
            cmd.record_upload(bias_data_packed, bias_data_gpu_image, opt);
        }
        else
        {
            cmd.record_upload(bias_data_packed, bias_data_gpu, opt);
        }

        bias_data_packed.release();
    }

    return 0;
}

int InnerProduct_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    const int num_input = weight_data_size / num_output;

    int in_elempack = opt.use_shader_pack8 && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;

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

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

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

int InnerProduct_vulkan::forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    const int num_input = weight_data_size / num_output;

    int in_elempack = opt.use_shader_pack8 && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input)
    {
        // gemm
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        int elempack = bottom_blob.elempack;

        // unpacking
        VkImageMat bottom_blob_unpacked = bottom_blob;
        if (elempack > 1)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

            vkdev->convert_packing(bottom_blob, bottom_blob_unpacked, 1, cmd, opt_pack1);
        }

        top_blob.create(num_output, h, elemsize, elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        VkImageMat top_blob_unpacked = top_blob;
        if (elempack > 1)
        {
            top_blob_unpacked.create(num_output, h * elempack, bottom_blob_unpacked.elemsize, 1, opt.workspace_vkallocator);
            if (top_blob_unpacked.empty())
                return -100;
        }

        std::vector<VkImageMat> bindings(4);
        bindings[0] = bottom_blob_unpacked;
        bindings[1] = top_blob_unpacked;
        bindings[2] = weight_data_gpu_image;
        bindings[3] = bias_data_gpu_image;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob_unpacked.dims;
        constants[1].i = bottom_blob_unpacked.w;
        constants[2].i = bottom_blob_unpacked.h;
        constants[3].i = bottom_blob_unpacked.c;
        constants[4].i = 0; //bottom_blob_unpacked.cstep;
        constants[5].i = top_blob_unpacked.dims;
        constants[6].i = top_blob_unpacked.w;
        constants[7].i = top_blob_unpacked.h;
        constants[8].i = top_blob_unpacked.c;
        constants[9].i = 0; //top_blob_unpacked.cstep;

        VkImageMat dispatcher;
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
    VkImageMat bottom_blob_flattened = bottom_blob;
    {
        Option opt_flatten = opt;
        opt_flatten.blob_vkallocator = opt.workspace_vkallocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, cmd, opt_flatten);
    }

    size_t elemsize = bottom_blob_flattened.elemsize;
    size_t out_elemsize = elemsize / in_elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    if (num_input / in_elempack >= 32)
    {
        // sum8
        VkImageMat top_blob_sum8;
        {
            top_blob_sum8.create((num_input / in_elempack + 7) / 8, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
            if (top_blob_sum8.empty())
                return -100;

            std::vector<VkImageMat> bindings(3);
            bindings[0] = bottom_blob_flattened;
            bindings[1] = top_blob_sum8;
            bindings[2] = weight_data_gpu_image;

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

            std::vector<VkImageMat> bindings(3);
            bindings[0] = top_blob_sum8;
            bindings[1] = top_blob;
            bindings[2] = bias_data_gpu_image;

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

        std::vector<VkImageMat> bindings(4);
        bindings[0] = bottom_blob_flattened;
        bindings[1] = top_blob;
        bindings[2] = weight_data_gpu_image;
        bindings[3] = bias_data_gpu_image;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob_flattened.dims;
        constants[1].i = bottom_blob_flattened.w;
        constants[2].i = bottom_blob_flattened.h;
        constants[3].i = bottom_blob_flattened.c;
        constants[4].i = 0; //bottom_blob_flattened.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = 0; //top_blob.cstep;

        cmd.record_pipeline(pipeline_innerproduct, bindings, constants, top_blob);
    }

    return 0;
}

} // namespace ncnn

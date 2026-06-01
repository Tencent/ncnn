// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolutiondepthwise_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

ConvolutionDepthWise_vulkan::ConvolutionDepthWise_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    padding = 0;

    pipeline_convolutiondepthwise = 0;
    pipeline_convolutiondepthwise_pack4 = 0;

    pipeline_convolutiondepthwise_group = 0;
    pipeline_convolutiondepthwise_group_pack4 = 0;
    pipeline_convolutiondepthwise_group_pack1to4 = 0;
    pipeline_convolutiondepthwise_group_pack4to1 = 0;
}

int ConvolutionDepthWise_vulkan::load_param(const ParamDict& pd)
{
    int ret = ConvolutionDepthWise::load_param(pd);

    if (dynamic_weight)
    {
        support_vulkan = false;
    }

#if NCNN_INT8
    if (int8_scale_term)
    {
        const int maxk = kernel_w * kernel_h;
        if (pad_value != 0.f || group == 0 || num_output % group != 0)
        {
            NCNN_LOGE("ConvolutionDepthWise_vulkan int8 nonzero pad value or invalid group is not supported");
            support_vulkan = false;
        }
        else
        {
            const int num_output_g = num_output / group;
            const int weight_data_size_g = group * maxk * num_output_g;
            if (weight_data_size_g == 0 || weight_data_size % weight_data_size_g != 0)
            {
                NCNN_LOGE("ConvolutionDepthWise_vulkan int8 weight shape mismatch");
                support_vulkan = false;
            }
        }
    }
#else
    if (int8_scale_term)
    {
        support_vulkan = false;
    }
#endif

    return ret;
}

int ConvolutionDepthWise_vulkan::create_pipeline(const Option& opt)
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        return create_pipeline_int8(opt);
    }
#endif

    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    // the shape after padding
    Mat shape_bordered;
    if (shape.dims != 0)
    {
        if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
        {
            shape_bordered = Mat(shape.w + pad_left + pad_right, shape.h + pad_top + pad_bottom, shape.c, (void*)0, shape.elemsize, shape.elempack);
        }
        else if ((pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
                 || (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234))
        {
            const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
            const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

            int wpad = kernel_extent_w + (shape.w - 1) / stride_w * stride_w - shape.w;
            int hpad = kernel_extent_h + (shape.h - 1) / stride_h * stride_h - shape.h;
            if (wpad > 0 || hpad > 0)
            {
                shape_bordered = Mat(shape.w + wpad, shape.h + hpad, shape.c, (void*)0, shape.elemsize, shape.elempack);
            }
        }
        else
        {
            shape_bordered = shape;
        }
    }

    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    int elempack = channels % 4 == 0 ? 4 : 1;
    int out_elempack = num_output % 4 == 0 ? 4 : 1;

    // group convolution
    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

    int elempack_g = channels_g % 4 == 0 ? 4 : 1;
    int out_elempack_g = num_output_g % 4 == 0 ? 4 : 1;

    size_t elemsize_g;
    size_t out_elemsize_g;
    if (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed)
    {
        elemsize_g = elempack_g * 2u;
        out_elemsize_g = out_elempack_g * 2u;
    }
    else
    {
        elemsize_g = elempack_g * 4u;
        out_elemsize_g = out_elempack_g * 4u;
    }

    Mat shape_bordered_g;
    if (shape_bordered.dims == 3) shape_bordered_g = Mat(shape_bordered.w, shape_bordered.h, shape_bordered.c * elempack / elempack_g, (void*)0, elemsize_g, elempack_g);

    Mat out_shape_g;
    if (out_shape.dims == 3) out_shape_g = Mat(out_shape.w, out_shape.h, out_shape.c * out_elempack / out_elempack_g, (void*)0, out_elemsize_g, out_elempack_g);

    {
        padding = ncnn::create_layer_vulkan(ncnn::LayerType::Padding);
        padding->vkdev = vkdev;

        padding->bottom_shapes.resize(1);
        padding->bottom_shapes[0] = shape;
        padding->top_shapes.resize(1);
        padding->top_shapes[0] = shape_bordered;

        ncnn::ParamDict pd;
        pd.set(0, pad_top);
        pd.set(1, pad_bottom);
        pd.set(2, pad_left);
        pd.set(3, pad_right);
        pd.set(4, 0);
        pd.set(5, pad_value);

        padding->load_param(pd);

        padding->create_pipeline(opt);
    }

    std::vector<vk_specialization_type> specializations(11 + 10);
    specializations[0].i = kernel_w;
    specializations[1].i = kernel_h;
    specializations[2].i = dilation_w;
    specializations[3].i = dilation_h;
    specializations[4].i = stride_w;
    specializations[5].i = stride_h;
    specializations[6].i = bias_term;
    specializations[7].i = group;
    specializations[8].i = activation_type;
    specializations[9].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
    specializations[10].f = activation_params.w == 2 ? activation_params[1] : 0.f;

    // depth-wise
    if (channels == group && group == num_output)
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, group);
        convert_packing(weight_data_r2, weight_data_packed, elempack, opt);

        specializations[11 + 0].i = shape_bordered.dims;
        specializations[11 + 1].i = shape_bordered.w;
        specializations[11 + 2].i = shape_bordered.h;
        specializations[11 + 3].i = shape_bordered.c;
        specializations[11 + 4].i = shape_bordered.cstep;
        specializations[11 + 5].i = out_shape.dims;
        specializations[11 + 6].i = out_shape.w;
        specializations[11 + 7].i = out_shape.h;
        specializations[11 + 8].i = out_shape.c;
        specializations[11 + 9].i = out_shape.cstep;

        Mat local_size_xyz(8, 8, std::min(4, num_output / out_elempack), (void*)0);
        if (out_shape.dims != 0)
        {
            local_size_xyz.w = std::min(8, out_shape.w);
            local_size_xyz.h = std::min(8, out_shape.h);
            local_size_xyz.c = std::min(4, out_shape.c);
        }

        // pack1
        if (elempack == 1)
        {
            pipeline_convolutiondepthwise = new Pipeline(vkdev);
            pipeline_convolutiondepthwise->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_convolutiondepthwise->create(LayerShaderType::convolutiondepthwise, opt, specializations);
        }

        // pack4
        if (elempack == 4)
        {
            pipeline_convolutiondepthwise_pack4 = new Pipeline(vkdev);
            pipeline_convolutiondepthwise_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_convolutiondepthwise_pack4->create(LayerShaderType::convolutiondepthwise_pack4, opt, specializations);
        }

        if (opt.lightmode)
        {
            weight_data.release();
        }

        return 0;
    }

    // src = kw-kh-inch-outch
    // dst = pa-pb-kw-kh-inch/pa-outch/pb
    {
        Mat weight_data_r2_groups = weight_data.reshape(maxk, channels_g, num_output_g * group);

        weight_data_packed_groups.create(maxk, channels_g / elempack_g, num_output_g / out_elempack_g * group, (size_t)4 * elempack_g * out_elempack_g, elempack_g * out_elempack_g);

        for (int g = 0; g < group; g++)
        {
            const Mat weight_data_r2 = weight_data_r2_groups.channel_range(num_output_g * g, num_output_g);

            Mat weight_data_packed = weight_data_packed_groups.channel_range(num_output_g / out_elempack_g * g, num_output_g / out_elempack_g);

            for (int q = 0; q + (out_elempack_g - 1) < num_output_g; q += out_elempack_g)
            {
                float* g00 = weight_data_packed.channel(q / out_elempack_g);

                for (int p = 0; p + (elempack_g - 1) < channels_g; p += elempack_g)
                {
                    for (int k = 0; k < maxk; k++)
                    {
                        for (int i = 0; i < out_elempack_g; i++)
                        {
                            const Mat k0 = weight_data_r2.channel(q + i);

                            for (int j = 0; j < elempack_g; j++)
                            {
                                const float* k00 = k0.row(p + j);

                                g00[0] = k00[k];

                                g00++;
                            }
                        }
                    }
                }
            }
        }
    }

    specializations[11 + 0].i = shape_bordered_g.dims;
    specializations[11 + 1].i = shape_bordered_g.w;
    specializations[11 + 2].i = shape_bordered_g.h;
    specializations[11 + 3].i = shape_bordered_g.c;
    specializations[11 + 4].i = shape_bordered_g.cstep;
    specializations[11 + 5].i = out_shape_g.dims;
    specializations[11 + 6].i = out_shape_g.w;
    specializations[11 + 7].i = out_shape_g.h;
    specializations[11 + 8].i = out_shape_g.c;
    specializations[11 + 9].i = out_shape_g.cstep;

    Mat local_size_xyz(8, 8, std::min(4, num_output / out_elempack_g), (void*)0);
    if (out_shape_g.dims != 0)
    {
        local_size_xyz.w = std::min(8, out_shape_g.w);
        local_size_xyz.h = std::min(8, out_shape_g.h);
        local_size_xyz.c = std::min(4, out_shape_g.c);
    }

    // pack1
    if (elempack_g == 1 && out_elempack_g == 1)
    {
        pipeline_convolutiondepthwise_group = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise_group->create(LayerShaderType::convolutiondepthwise_group, opt, specializations);
    }

    // pack4
    if (elempack_g == 4 && out_elempack_g == 4)
    {
        pipeline_convolutiondepthwise_group_pack4 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise_group_pack4->create(LayerShaderType::convolutiondepthwise_group_pack4, opt, specializations);
    }

    // pack1to4
    if (elempack_g == 1 && out_elempack_g == 4)
    {
        pipeline_convolutiondepthwise_group_pack1to4 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise_group_pack1to4->create(LayerShaderType::convolutiondepthwise_group_pack1to4, opt, specializations);
    }

    // pack4to1
    if (elempack_g == 4 && out_elempack_g == 1)
    {
        pipeline_convolutiondepthwise_group_pack4to1 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack4to1->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise_group_pack4to1->create(LayerShaderType::convolutiondepthwise_group_pack4to1, opt, specializations);
    }

    if (opt.lightmode)
    {
        weight_data.release();
    }

    return 0;
}


int ConvolutionDepthWise_vulkan::destroy_pipeline(const Option& opt)
{
    if (padding)
    {
        padding->destroy_pipeline(opt);
        delete padding;
        padding = 0;
    }

    delete pipeline_convolutiondepthwise;
    pipeline_convolutiondepthwise = 0;

    delete pipeline_convolutiondepthwise_pack4;
    pipeline_convolutiondepthwise_pack4 = 0;

    delete pipeline_convolutiondepthwise_group;
    pipeline_convolutiondepthwise_group = 0;

    delete pipeline_convolutiondepthwise_group_pack4;
    pipeline_convolutiondepthwise_group_pack4 = 0;

    delete pipeline_convolutiondepthwise_group_pack1to4;
    pipeline_convolutiondepthwise_group_pack1to4 = 0;

    delete pipeline_convolutiondepthwise_group_pack4to1;
    pipeline_convolutiondepthwise_group_pack4to1 = 0;

    return 0;
}

int ConvolutionDepthWise_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        return upload_model_int8(cmd, opt);
    }
#endif

    if (padding)
    {
        padding->upload_model(cmd, opt);
    }

    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        cmd.record_upload(weight_data_packed, weight_data_gpu, opt);

        weight_data_packed.release();
    }
    else
    {
        cmd.record_upload(weight_data_packed_groups, weight_data_gpu, opt);

        weight_data_packed_groups.release();
    }

    if (bias_term)
    {
        cmd.record_upload(bias_data, bias_data_gpu, opt);

        bias_data.release();
    }

    return 0;
}


int ConvolutionDepthWise_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        return forward_int8(bottom_blob, top_blob, cmd, opt);
    }
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    VkMat bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->forward(bottom_blob, bottom_blob_bordered, cmd, opt_pad);
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad / 2;
            padding_params[1] = hpad - hpad / 2;
            padding_params[2] = wpad / 2;
            padding_params[3] = wpad - wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;

            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad - hpad / 2;
            padding_params[1] = hpad / 2;
            padding_params[2] = wpad - wpad / 2;
            padding_params[3] = wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;

            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    // depth-wise
    if (channels == group / elempack && group / elempack == num_output / elempack)
    {
        std::vector<VkMat> bindings(4);
        bindings[0] = bottom_blob_bordered;
        bindings[1] = top_blob;
        bindings[2] = weight_data_gpu;
        bindings[3] = bias_data_gpu;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob_bordered.dims;
        constants[1].i = bottom_blob_bordered.w;
        constants[2].i = bottom_blob_bordered.h;
        constants[3].i = bottom_blob_bordered.c;
        constants[4].i = bottom_blob_bordered.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = top_blob.cstep;

        const Pipeline* pipeline = elempack == 4 ? pipeline_convolutiondepthwise_pack4 : pipeline_convolutiondepthwise;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    const int channels_g = channels * elempack / group;
    const int num_output_g = num_output / group;

    int elempack_g = channels_g % 4 == 0 ? 4 : 1;
    int out_elempack_g = num_output_g % 4 == 0 ? 4 : 1;
    size_t out_elemsize_g = elemsize / elempack * out_elempack_g;

    // unpacking
    VkMat bottom_blob_bordered_unpacked = bottom_blob_bordered;
    if (elempack > elempack_g)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

        vkdev->convert_packing(bottom_blob_bordered, bottom_blob_bordered_unpacked, elempack_g, cmd, opt_pack1);
    }

    VkMat top_blob_unpacked = top_blob;
    if (out_elempack_g < out_elempack)
    {
        top_blob_unpacked.create(outw, outh, num_output / out_elempack_g, out_elemsize_g, out_elempack_g, opt.workspace_vkallocator);
        if (top_blob_unpacked.empty())
            return -100;
    }

    std::vector<VkMat> bindings(4);
    bindings[0] = bottom_blob_bordered_unpacked;
    bindings[1] = top_blob_unpacked;
    bindings[2] = weight_data_gpu;
    bindings[3] = bias_data_gpu;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob_bordered_unpacked.dims;
    constants[1].i = bottom_blob_bordered_unpacked.w;
    constants[2].i = bottom_blob_bordered_unpacked.h;
    constants[3].i = bottom_blob_bordered_unpacked.c;
    constants[4].i = bottom_blob_bordered_unpacked.cstep;
    constants[5].i = top_blob_unpacked.dims;
    constants[6].i = top_blob_unpacked.w;
    constants[7].i = top_blob_unpacked.h;
    constants[8].i = top_blob_unpacked.c;
    constants[9].i = top_blob_unpacked.cstep;

    const Pipeline* pipeline = 0;
    if (elempack_g == 1 && out_elempack_g == 1)
    {
        pipeline = pipeline_convolutiondepthwise_group;
    }
    else if (elempack_g == 4 && out_elempack_g == 4)
    {
        pipeline = pipeline_convolutiondepthwise_group_pack4;
    }
    else if (elempack_g == 1 && out_elempack_g == 4)
    {
        pipeline = pipeline_convolutiondepthwise_group_pack1to4;
    }
    else if (elempack_g == 4 && out_elempack_g == 1)
    {
        pipeline = pipeline_convolutiondepthwise_group_pack4to1;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob_unpacked);

    // packing
    if (out_elempack_g < out_elempack)
    {
        vkdev->convert_packing(top_blob_unpacked, top_blob, out_elempack, cmd, opt);
    }
    else
    {
        top_blob = top_blob_unpacked;
    }

    return 0;
}

#if NCNN_INT8
int ConvolutionDepthWise_vulkan::create_pipeline_int8(const Option& opt)
{
    Mat shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    Mat out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    if (shape.dims != 3) shape = Mat();
    if (out_shape.dims != 3) out_shape = Mat();

    const int maxk = kernel_w * kernel_h;
    if (group == 0 || num_output % group != 0)
    {
        NCNN_LOGE("ConvolutionDepthWise_vulkan int8 invalid group");
        return -1;
    }

    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;
    const bool is_depthwise = channels == group && group == num_output;
    const int channels_g = channels / group;
    const int num_output_g = num_output / group;
    const int elempack = is_depthwise && opt.use_packing_layout && group % 4 == 0 ? 4 : 1;
    const int out_elempack = opt.use_packing_layout && num_output % 4 == 0 ? 4 : 1;
    const int elempack_g = !is_depthwise && opt.use_packing_layout && channels_g % 4 == 0 ? 4 : 1;
    const int out_elempack_g = !is_depthwise && opt.use_packing_layout && num_output_g % 4 == 0 ? 4 : 1;

    if (weight_data.elemsize != (size_t)1u)
    {
        NCNN_LOGE("ConvolutionDepthWise_vulkan int8 weight data is not int8");
        return -1;
    }

    Option opt_int8 = opt;
    opt_int8.use_fp16_packed = false;
    opt_int8.use_fp16_storage = false;
    opt_int8.use_fp16_arithmetic = false;
    opt_int8.use_bf16_packed = false;
    opt_int8.use_bf16_storage = false;
    opt_int8.use_int16_packed = false;
    opt_int8.use_int16_storage = false;
    opt_int8.use_int8_arithmetic = opt_int8.use_int8_storage && vkdev->info.support_int8_arithmetic();

    Option opt_int8_pack4 = opt;
    opt_int8_pack4.use_fp16_arithmetic = false;
    opt_int8_pack4.use_int16_packed = false;
    opt_int8_pack4.use_int16_storage = false;
    opt_int8_pack4.use_int8_arithmetic = opt_int8_pack4.use_int8_storage && vkdev->info.support_int8_arithmetic();

    if (is_depthwise)
    {
        if (elempack == 4)
        {
            const Mat weight_data_r2 = weight_data.reshape(maxk, group);

            weight_data_int8_packed.create(maxk, group / 4, (size_t)4u, 4);

            for (int q = 0; q + 3 < group; q += 4)
            {
                signed char* g00 = weight_data_int8_packed.row<signed char>(q / 4);

                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        const signed char* k0 = weight_data_r2.row<const signed char>(q + i);
                        g00[i] = k0[k];
                    }

                    g00 += 4;
                }
            }
        }
        else
        {
            weight_data_int8_packed = weight_data.reshape(maxk, group);
        }
    }
    else
    {
        if (elempack_g == 1 && out_elempack_g == 1)
        {
            weight_data_int8_packed = weight_data.reshape(weight_data_size);
        }
        else
        {
            const Mat weight_data_r2_groups = weight_data.reshape(maxk, channels_g, num_output_g * group);

            weight_data_int8_packed.create(maxk, channels_g / elempack_g, num_output_g / out_elempack_g * group, (size_t)elempack_g * out_elempack_g, elempack_g * out_elempack_g);

            for (int g = 0; g < group; g++)
            {
                const Mat weight_data_r2 = weight_data_r2_groups.channel_range(num_output_g * g, num_output_g);

                Mat weight_data_packed = weight_data_int8_packed.channel_range(num_output_g / out_elempack_g * g, num_output_g / out_elempack_g);

                for (int q = 0; q + (out_elempack_g - 1) < num_output_g; q += out_elempack_g)
                {
                    signed char* g00 = weight_data_packed.channel(q / out_elempack_g);

                    for (int p = 0; p + (elempack_g - 1) < channels_g; p += elempack_g)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            for (int i = 0; i < out_elempack_g; i++)
                            {
                                const Mat k0 = weight_data_r2.channel(q + i);

                                for (int j = 0; j < elempack_g; j++)
                                {
                                    const signed char* k00 = k0.row<const signed char>(p + j);

                                    g00[0] = k00[k];

                                    g00++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<vk_specialization_type> specializations(12 + 10);
    const bool use_int8_requantize = int8_scale_term > 100;
    specializations[0].i = kernel_w;
    specializations[1].i = kernel_h;
    specializations[2].i = dilation_w;
    specializations[3].i = dilation_h;
    specializations[4].i = stride_w;
    specializations[5].i = stride_h;
    specializations[6].i = bias_term;
    specializations[7].i = group;
    specializations[8].i = activation_type;
    specializations[9].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
    specializations[10].f = activation_params.w == 2 ? activation_params[1] : 0.f;
    specializations[11].i = use_int8_requantize ? 1 : 0;
    specializations[12 + 0].i = shape.dims;
    specializations[12 + 1].i = shape.w;
    specializations[12 + 2].i = shape.h;
    specializations[12 + 3].i = shape.dims != 0 ? channels / (is_depthwise ? elempack : elempack_g) : 0;
    specializations[12 + 4].i = 0;
    specializations[12 + 5].i = out_shape.dims;
    specializations[12 + 6].i = out_shape.w;
    specializations[12 + 7].i = out_shape.h;
    specializations[12 + 8].i = out_shape.dims != 0 ? num_output / (is_depthwise ? out_elempack : out_elempack_g) : 0;
    specializations[12 + 9].i = 0;

    if (is_depthwise)
    {
        Mat local_size_xyz(8, 8, std::min(4, num_output), (void*)0);
        if (out_shape.dims != 0)
        {
            local_size_xyz.w = std::min(8, out_shape.w);
            local_size_xyz.h = std::min(8, out_shape.h);
            local_size_xyz.c = std::min(4, num_output);
        }

        if (opt.use_packing_layout && group % 4 == 0)
        {
            pipeline_convolutiondepthwise_pack4 = new Pipeline(vkdev);
            pipeline_convolutiondepthwise_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_convolutiondepthwise_pack4->create(LayerShaderType::convolutiondepthwise_pack4_int8, opt_int8_pack4, specializations);
        }
        else
        {
            pipeline_convolutiondepthwise = new Pipeline(vkdev);
            pipeline_convolutiondepthwise->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_convolutiondepthwise->create(LayerShaderType::convolutiondepthwise_int8, opt_int8, specializations);
        }
    }
    else
    {
        Mat local_size_xyz(8, 8, 1, (void*)0);
        if (out_elempack_g == 1 && elempack_g == 1)
        {
            if (out_shape.dims != 0)
            {
                local_size_xyz.w = std::min(8, std::max(4, (out_shape.w * out_shape.h + 3) / 4));
                local_size_xyz.h = std::min(8, std::max(4, (num_output_g + 3) / 4));
                local_size_xyz.c = 1;
            }

            pipeline_convolutiondepthwise_group = new Pipeline(vkdev);
            if (opt_int8.use_shader_local_memory)
            {
                pipeline_convolutiondepthwise_group->set_local_size_xyz(8, 8, 1);
            }
            else
            {
                pipeline_convolutiondepthwise_group->set_optimal_local_size_xyz(local_size_xyz);
            }
            pipeline_convolutiondepthwise_group->create(LayerShaderType::convolutiondepthwise_group_int8, opt_int8, specializations);
        }
        else
        {
            if (out_shape.dims != 0)
            {
                local_size_xyz.w = std::min(8, out_shape.w);
                local_size_xyz.h = std::min(8, out_shape.h);
                local_size_xyz.c = std::min(4, num_output / out_elempack_g);
            }

            if (elempack_g == 4 && out_elempack_g == 4)
            {
                pipeline_convolutiondepthwise_group_pack4 = new Pipeline(vkdev);
                pipeline_convolutiondepthwise_group_pack4->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_convolutiondepthwise_group_pack4->create(LayerShaderType::convolutiondepthwise_group_pack4_int8, opt_int8_pack4, specializations);
            }

            if (elempack_g == 1 && out_elempack_g == 4)
            {
                pipeline_convolutiondepthwise_group_pack1to4 = new Pipeline(vkdev);
                pipeline_convolutiondepthwise_group_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_convolutiondepthwise_group_pack1to4->create(LayerShaderType::convolutiondepthwise_group_pack1to4_int8, opt_int8_pack4, specializations);
            }

            if (elempack_g == 4 && out_elempack_g == 1)
            {
                pipeline_convolutiondepthwise_group_pack4to1 = new Pipeline(vkdev);
                pipeline_convolutiondepthwise_group_pack4to1->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_convolutiondepthwise_group_pack4to1->create(LayerShaderType::convolutiondepthwise_group_pack4to1_int8, opt_int8_pack4, specializations);
            }
        }
    }

    if (opt.lightmode)
    {
        weight_data.release();
    }

    return 0;
}

int ConvolutionDepthWise_vulkan::upload_model_int8(VkTransfer& cmd, const Option& opt)
{
    Option opt_float = opt;
    opt_float.use_fp16_packed = false;
    opt_float.use_fp16_storage = false;
    opt_float.use_fp16_arithmetic = false;
    opt_float.use_bf16_packed = false;
    opt_float.use_bf16_storage = false;
    opt_float.use_int16_packed = false;
    opt_float.use_int16_storage = false;
    opt_float.use_int8_packed = false;
    opt_float.use_int8_storage = false;
    opt_float.use_int8_arithmetic = false;

    Option opt_int8 = opt;
    opt_int8.use_fp16_packed = false;
    opt_int8.use_fp16_storage = false;
    opt_int8.use_fp16_arithmetic = false;
    opt_int8.use_bf16_packed = false;
    opt_int8.use_bf16_storage = false;
    opt_int8.use_int16_packed = false;
    opt_int8.use_int16_storage = false;

    cmd.record_upload(weight_data_int8_packed, weight_data_gpu, opt_int8);

    weight_data_int8_packed.release();

    cmd.record_upload(weight_data_int8_scales, weight_data_int8_scales_gpu, opt_float);
    cmd.record_upload(bottom_blob_int8_scales, bottom_blob_int8_scales_gpu, opt_float);

    const bool use_int8_requantize = int8_scale_term > 100;
    if (use_int8_requantize)
    {
        cmd.record_upload(top_blob_int8_scales, top_blob_int8_scales_gpu, opt_float);
    }

    if (bias_term)
    {
        cmd.record_upload(bias_data, bias_data_gpu, opt_float);

        bias_data.release();
    }

    return 0;
}

int ConvolutionDepthWise_vulkan::forward_int8(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    VkMat bottom = bottom_blob;
    const bool bottom_is_int8 = bottom.elembits() == 8;

    int channels = bottom.c * bottom.elempack;
    const bool is_depthwise = channels == group && group == num_output;
    const int channels_g = channels / group;
    const int num_output_g = num_output / group;
    const int elempack = is_depthwise && opt.use_packing_layout && channels % 4 == 0 ? 4 : 1;
    const int elempack_g = !is_depthwise && opt.use_packing_layout && channels_g % 4 == 0 ? 4 : 1;
    const int out_elempack_g = !is_depthwise && opt.use_packing_layout && num_output_g % 4 == 0 ? 4 : 1;
    const bool use_depthwise_pack4 = is_depthwise && elempack == 4;
    const bool use_group_packing = !is_depthwise && (elempack_g != 1 || out_elempack_g != 1);
    const int bottom_elempack = is_depthwise ? elempack : elempack_g;

    if (bottom.elempack != bottom_elempack || (!bottom_is_int8 && bottom.elembits() == 16 && !use_depthwise_pack4 && !use_group_packing))
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

        VkMat bottom_unpacked;
        vkdev->convert_packing(bottom, bottom_unpacked, bottom_elempack, bottom_is_int8 || use_depthwise_pack4 || use_group_packing ? 0 : 1, cmd, opt_pack1);
        bottom = bottom_unpacked;
    }

    const int w = bottom.w;
    const int h = bottom.h;
    channels = bottom.c * bottom.elempack;

    if (channels % group != 0)
    {
        NCNN_LOGE("ConvolutionDepthWise_vulkan int8 input channels mismatch");
        return -1;
    }

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int wpad = 0;
    int hpad = 0;
    int pad_left_real = 0;
    int pad_top_real = 0;

    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        wpad = pad_left + pad_right;
        hpad = pad_top + pad_bottom;
        pad_left_real = pad_left;
        pad_top_real = pad_top;
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        pad_left_real = wpad / 2;
        pad_top_real = hpad / 2;
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        pad_left_real = wpad - wpad / 2;
        pad_top_real = hpad - hpad / 2;
    }

    const int outw = (w + wpad - kernel_extent_w) / stride_w + 1;
    const int outh = (h + hpad - kernel_extent_h) / stride_h + 1;

    const bool use_int8_requantize = int8_scale_term > 100;
    const int out_elempack = opt.use_packing_layout && num_output % 4 == 0 ? 4 : 1;
    const bool use_sfp_output = !use_int8_requantize && (use_depthwise_pack4 || use_group_packing) && (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed);
    size_t out_elemsize;
    if (use_int8_requantize)
    {
        out_elemsize = out_elempack;
    }
    else if (use_sfp_output)
    {
        out_elemsize = (size_t)2u * out_elempack;
    }
    else
    {
        out_elemsize = (size_t)4u * out_elempack;
    }

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    VkMat top_blob_unpacked = top_blob;
    if (!is_depthwise && out_elempack_g != out_elempack)
    {
        size_t out_elemsize_g;
        if (use_int8_requantize)
        {
            out_elemsize_g = out_elempack_g;
        }
        else if (use_sfp_output)
        {
            out_elemsize_g = (size_t)2u * out_elempack_g;
        }
        else
        {
            out_elemsize_g = (size_t)4u * out_elempack_g;
        }

        top_blob_unpacked.create(outw, outh, num_output / out_elempack_g, out_elemsize_g, out_elempack_g, opt.workspace_vkallocator);
        if (top_blob_unpacked.empty())
            return -100;
    }

    std::vector<VkMat> bindings(9);
    bindings[0] = bottom;
    bindings[1] = top_blob_unpacked;
    bindings[2] = weight_data_gpu;
    bindings[3] = bias_data_gpu;
    bindings[4] = weight_data_int8_scales_gpu;
    bindings[5] = bottom_blob_int8_scales_gpu;
    bindings[6] = top_blob_int8_scales_gpu;
    // bindings 7/8 alias top/bottom with int8 SSBO element types
    bindings[7] = top_blob_unpacked;
    bindings[8] = bottom;

    std::vector<vk_constant_type> constants(13);
    constants[0].i = bottom.dims;
    constants[1].i = bottom.w;
    constants[2].i = bottom.h;
    constants[3].i = bottom.c;
    constants[4].i = bottom.cstep;
    constants[5].i = top_blob_unpacked.dims;
    constants[6].i = top_blob_unpacked.w;
    constants[7].i = top_blob_unpacked.h;
    constants[8].i = top_blob_unpacked.c;
    constants[9].i = top_blob_unpacked.cstep;
    constants[10].i = pad_left_real;
    constants[11].i = pad_top_real;
    constants[12].i = bottom_is_int8 ? 1 : 0;

    if (is_depthwise)
    {
        const Pipeline* pipeline = bottom.elempack == 4 ? pipeline_convolutiondepthwise_pack4 : pipeline_convolutiondepthwise;
        cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    }
    else
    {
        const Pipeline* pipeline = 0;
        if (elempack_g == 1 && out_elempack_g == 1)
        {
            pipeline = pipeline_convolutiondepthwise_group;

            VkMat dispatcher;
            dispatcher.w = (top_blob_unpacked.w * top_blob_unpacked.h + 3) / 4;
            dispatcher.h = (num_output_g + 3) / 4;
            dispatcher.c = group;

            cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
        }
        else
        {
            if (elempack_g == 4 && out_elempack_g == 4) pipeline = pipeline_convolutiondepthwise_group_pack4;
            if (elempack_g == 1 && out_elempack_g == 4) pipeline = pipeline_convolutiondepthwise_group_pack1to4;
            if (elempack_g == 4 && out_elempack_g == 1) pipeline = pipeline_convolutiondepthwise_group_pack4to1;

            cmd.record_pipeline(pipeline, bindings, constants, top_blob_unpacked);
        }

        if (out_elempack_g != out_elempack)
        {
            vkdev->convert_packing(top_blob_unpacked, top_blob, out_elempack, cmd, opt);
        }
    }

    return 0;
}
#endif // NCNN_INT8

} // namespace ncnn

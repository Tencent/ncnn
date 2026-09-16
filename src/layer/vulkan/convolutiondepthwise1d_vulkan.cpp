// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "convolutiondepthwise1d_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

ConvolutionDepthWise1D_vulkan::ConvolutionDepthWise1D_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    padding = 0;

    pipeline_convolutiondepthwise1d = 0;
    pipeline_convolutiondepthwise1d_pack4 = 0;

    pipeline_convolutiondepthwise1d_group = 0;
    pipeline_convolutiondepthwise1d_group_pack4 = 0;
    pipeline_convolutiondepthwise1d_group_pack1to4 = 0;
    pipeline_convolutiondepthwise1d_group_pack4to1 = 0;
}

int ConvolutionDepthWise1D_vulkan::load_param(const ParamDict& pd)
{
    int ret = ConvolutionDepthWise1D::load_param(pd);

    if (dynamic_weight)
    {
        support_vulkan = false;
    }

    return ret;
}

int ConvolutionDepthWise1D_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;

    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    // the shape after padding
    Mat shape_bordered;
    if (shape.dims != 0)
    {
        const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

        if (pad_left > 0 || pad_right > 0)
        {
            shape_bordered = Mat(shape.w + pad_left + pad_right, shape.h, (void*)0, shape.elemsize, shape.elempack);
        }
        else if ((pad_left == -233 && pad_right == -233) || (pad_left == -234 && pad_right == -234))
        {
            int wpad = kernel_extent_w + (shape.w - 1) / stride_w * stride_w - shape.w;
            if (wpad > 0)
            {
                shape_bordered = Mat(shape.w + wpad, shape.h, (void*)0, shape.elemsize, shape.elempack);
            }
        }
        else
        {
            shape_bordered = shape;
        }
    }

    const int maxk = kernel_w;
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
    if (shape_bordered.dims == 2) shape_bordered_g = Mat(shape_bordered.w, shape_bordered.h * elempack / elempack_g, (void*)0, elemsize_g, elempack_g);

    Mat out_shape_g;
    if (out_shape.dims == 2) out_shape_g = Mat(out_shape.w, out_shape.h * out_elempack / out_elempack_g, (void*)0, out_elemsize_g, out_elempack_g);

    {
        padding = ncnn::create_layer_vulkan(ncnn::LayerType::Padding);
        padding->vkdev = vkdev;

        padding->bottom_shapes.resize(1);
        padding->bottom_shapes[0] = shape;
        padding->top_shapes.resize(1);
        padding->top_shapes[0] = shape_bordered;

        ncnn::ParamDict pd;
        pd.set(0, 0);
        pd.set(1, 0);
        pd.set(2, pad_left);
        pd.set(3, pad_right);
        pd.set(4, 0);
        pd.set(5, pad_value);

        padding->load_param(pd);

        padding->create_pipeline(opt);
    }

    std::vector<vk_specialization_type> specializations(8 + 10);
    specializations[0].i = kernel_w;
    specializations[1].i = dilation_w;
    specializations[2].i = stride_w;
    specializations[3].i = bias_term;
    specializations[4].i = group;
    specializations[5].i = activation_type;
    specializations[6].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
    specializations[7].f = activation_params.w == 2 ? activation_params[1] : 0.f;

    // depth-wise
    if (channels == group && group == num_output)
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, group);
        convert_packing(weight_data_r2, weight_data_packed, elempack, opt);

        specializations[8 + 0].i = shape_bordered.dims;
        specializations[8 + 1].i = shape_bordered.w;
        specializations[8 + 2].i = shape_bordered.h;
        specializations[8 + 3].i = shape_bordered.c;
        specializations[8 + 4].i = shape_bordered.cstep;
        specializations[8 + 5].i = out_shape.dims;
        specializations[8 + 6].i = out_shape.w;
        specializations[8 + 7].i = out_shape.h;
        specializations[8 + 8].i = out_shape.c;
        specializations[8 + 9].i = out_shape.cstep;

        Mat local_size_xyz(8, 8, 1, (void*)0);
        if (out_shape.dims != 0)
        {
            local_size_xyz.w = std::min(8, out_shape.w);
            local_size_xyz.h = std::min(8, out_shape.h);
            local_size_xyz.c = 1;
        }

        // pack1
        if (elempack == 1)
        {
            pipeline_convolutiondepthwise1d = new Pipeline(vkdev);
            pipeline_convolutiondepthwise1d->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_convolutiondepthwise1d->create(LayerShaderType::convolutiondepthwise1d, opt, specializations);
        }

        // pack4
        if (elempack == 4)
        {
            pipeline_convolutiondepthwise1d_pack4 = new Pipeline(vkdev);
            pipeline_convolutiondepthwise1d_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_convolutiondepthwise1d_pack4->create(LayerShaderType::convolutiondepthwise1d_pack4, opt, specializations);
        }

        if (opt.lightmode)
        {
            weight_data.release();
        }

        return 0;
    }

    // src = kw-inch/pa-outch/pb
    // dst = pa-pb-kw-inch/pa-outch/pb
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

    specializations[8 + 0].i = shape_bordered_g.dims;
    specializations[8 + 1].i = shape_bordered_g.w;
    specializations[8 + 2].i = shape_bordered_g.h;
    specializations[8 + 3].i = shape_bordered_g.c;
    specializations[8 + 4].i = shape_bordered_g.cstep;
    specializations[8 + 5].i = out_shape_g.dims;
    specializations[8 + 6].i = out_shape_g.w;
    specializations[8 + 7].i = out_shape_g.h;
    specializations[8 + 8].i = out_shape_g.c;
    specializations[8 + 9].i = out_shape_g.cstep;

    Mat local_size_xyz(8, 8, 1, (void*)0);
    if (out_shape_g.dims != 0)
    {
        local_size_xyz.w = std::min(8, out_shape_g.w);
        local_size_xyz.h = std::min(8, out_shape_g.h);
        local_size_xyz.c = 1;
    }

    // pack1
    if (elempack_g == 1 && out_elempack_g == 1)
    {
        pipeline_convolutiondepthwise1d_group = new Pipeline(vkdev);
        pipeline_convolutiondepthwise1d_group->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise1d_group->create(LayerShaderType::convolutiondepthwise1d_group, opt, specializations);
    }

    // pack4
    if (elempack_g == 4 && out_elempack_g == 4)
    {
        pipeline_convolutiondepthwise1d_group_pack4 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise1d_group_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise1d_group_pack4->create(LayerShaderType::convolutiondepthwise1d_group_pack4, opt, specializations);
    }

    // pack1to4
    if (elempack_g == 1 && out_elempack_g == 4)
    {
        pipeline_convolutiondepthwise1d_group_pack1to4 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise1d_group_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise1d_group_pack1to4->create(LayerShaderType::convolutiondepthwise1d_group_pack1to4, opt, specializations);
    }

    // pack4to1
    if (elempack_g == 4 && out_elempack_g == 1)
    {
        pipeline_convolutiondepthwise1d_group_pack4to1 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise1d_group_pack4to1->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise1d_group_pack4to1->create(LayerShaderType::convolutiondepthwise1d_group_pack4to1, opt, specializations);
    }

    if (opt.lightmode)
    {
        weight_data.release();
    }

    return 0;
}

int ConvolutionDepthWise1D_vulkan::destroy_pipeline(const Option& opt)
{
    if (padding)
    {
        padding->destroy_pipeline(opt);
        delete padding;
        padding = 0;
    }

    delete pipeline_convolutiondepthwise1d;
    pipeline_convolutiondepthwise1d = 0;

    delete pipeline_convolutiondepthwise1d_pack4;
    pipeline_convolutiondepthwise1d_pack4 = 0;

    delete pipeline_convolutiondepthwise1d_group;
    pipeline_convolutiondepthwise1d_group = 0;

    delete pipeline_convolutiondepthwise1d_group_pack4;
    pipeline_convolutiondepthwise1d_group_pack4 = 0;

    delete pipeline_convolutiondepthwise1d_group_pack1to4;
    pipeline_convolutiondepthwise1d_group_pack1to4 = 0;

    delete pipeline_convolutiondepthwise1d_group_pack4to1;
    pipeline_convolutiondepthwise1d_group_pack4to1 = 0;

    return 0;
}

int ConvolutionDepthWise1D_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (padding)
    {
        padding->upload_model(cmd, opt);
    }

    const int maxk = kernel_w;
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

int ConvolutionDepthWise1D_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int channels = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    VkMat bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0)
    {
        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->forward(bottom_blob, bottom_blob_bordered, cmd, opt_pad);
    }
    else if (pad_left == -233 && pad_right == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        if (wpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = 0;
            padding_params[1] = 0;
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
    else if (pad_left == -234 && pad_right == -234)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        if (wpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = 0;
            padding_params[1] = 0;
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

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int out_elempack = num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
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

        const Pipeline* pipeline = elempack == 4 ? pipeline_convolutiondepthwise1d_pack4 : pipeline_convolutiondepthwise1d;

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
        top_blob_unpacked.create(outw, num_output / out_elempack_g, out_elemsize_g, out_elempack_g, opt.workspace_vkallocator);
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
        pipeline = pipeline_convolutiondepthwise1d_group;
    }
    else if (elempack_g == 4 && out_elempack_g == 4)
    {
        pipeline = pipeline_convolutiondepthwise1d_group_pack4;
    }
    else if (elempack_g == 1 && out_elempack_g == 4)
    {
        pipeline = pipeline_convolutiondepthwise1d_group_pack1to4;
    }
    else if (elempack_g == 4 && out_elempack_g == 1)
    {
        pipeline = pipeline_convolutiondepthwise1d_group_pack4to1;
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

} // namespace ncnn

// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deformableconv2d_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

DeformableConv2D_vulkan::DeformableConv2D_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    padding = 0;

    pipeline_deformableconv2d_packed = 0;
    pipeline_deformableconv2d_packed_mask = 0;
}

int DeformableConv2D_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

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

    int elempack = num_input % 4 == 0 ? 4 : 1;
    int out_elempack = num_output % 4 == 0 ? 4 : 1;

    size_t elemsize;
    size_t out_elemsize;
    if (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed)
    {
        elemsize = elempack * 2u;
        out_elemsize = out_elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
        out_elemsize = out_elempack * 4u;
    }

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
        pd.set(5, 0);

        padding->load_param(pd);

        padding->create_pipeline(opt);
    }

    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        const int num_output_packed = (num_output + 3) / 4 * 4;

        weight_data_packed.create(maxk, num_input / elempack, num_output_packed / 4, (size_t)4 * 4 * elempack, 4 * elempack);

        for (int q = 0; q < num_output_packed; q += 4)
        {
            float* g00 = weight_data_packed.channel(q / 4);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        for (int j = 0; j < elempack; j++)
                        {
                            if (q + i < num_output)
                            {
                                const float* k00 = weight_data_r2.channel(q + i).row(p + j);
                                g00[0] = k00[k];
                            }
                            else
                            {
                                g00[0] = 0.f;
                            }
                            g00++;
                        }
                    }
                }
            }
        }
    }

    std::vector<vk_specialization_type> specializations(13 + 13);
    specializations[0].i = kernel_w;
    specializations[1].i = kernel_h;
    specializations[2].i = dilation_w;
    specializations[3].i = dilation_h;
    specializations[4].i = stride_w;
    specializations[5].i = stride_h;
    specializations[6].i = bias_term;
    specializations[7].i = 0;
    specializations[8].i = activation_type;
    specializations[9].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
    specializations[10].f = activation_params.w == 2 ? activation_params[1] : 0.f;
    specializations[11].i = elempack;
    specializations[12].i = out_elempack;

    const int num_output_packed = (num_output + 3) / 4 * 4;
    const int c_iterations = num_input / elempack;
    const int cstep_scalar = (shape_bordered.dims != 0) ? shape_bordered.cstep : 0;
    const int outcstep_scalar = (out_shape.dims != 0) ? ((out_elempack == 4) ? out_shape.cstep : (out_shape.cstep * 4)) : 0;

    specializations[13].i = shape_bordered.dims;
    specializations[14].i = shape_bordered.w;
    specializations[15].i = shape_bordered.h;
    specializations[16].i = c_iterations;
    specializations[17].i = cstep_scalar;
    specializations[18].i = out_shape.dims;
    specializations[19].i = out_shape.w;
    specializations[20].i = out_shape.h;
    specializations[21].i = out_shape.dims != 0 ? num_output_packed / 4 : 0;
    specializations[22].i = outcstep_scalar;
    specializations[23].i = num_output;
    specializations[24].i = 0;
    specializations[25].i = 0;

    Mat local_size_xyz;
    if (out_shape.dims == 3)
    {
        local_size_xyz.w = std::min(4, out_shape.w);
        local_size_xyz.h = std::min(4, out_shape.h);
        local_size_xyz.c = std::min(4, out_shape.c);
    }
    else
    {
        local_size_xyz.w = 4;
        local_size_xyz.h = 4;
        local_size_xyz.c = 4;
    }

    pipeline_deformableconv2d_packed = new Pipeline(vkdev);
    pipeline_deformableconv2d_packed->set_optimal_local_size_xyz(local_size_xyz);
    pipeline_deformableconv2d_packed->create(LayerShaderType::deformableconv2d_packed, opt, specializations);

    specializations[7].i = 1;
    pipeline_deformableconv2d_packed_mask = new Pipeline(vkdev);
    pipeline_deformableconv2d_packed_mask->set_optimal_local_size_xyz(local_size_xyz);
    pipeline_deformableconv2d_packed_mask->create(LayerShaderType::deformableconv2d_packed, opt, specializations);

    return 0;
}

int DeformableConv2D_vulkan::destroy_pipeline(const Option& opt)
{
    if (padding)
    {
        padding->destroy_pipeline(opt);
        delete padding;
        padding = 0;
    }

    delete pipeline_deformableconv2d_packed;
    pipeline_deformableconv2d_packed = 0;

    delete pipeline_deformableconv2d_packed_mask;
    pipeline_deformableconv2d_packed_mask = 0;

    return 0;
}

int DeformableConv2D_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (padding)
    {
        padding->upload_model(cmd, opt);
    }

    cmd.record_upload(weight_data_packed, weight_data_gpu, opt);

    weight_data_packed.release();

    if (bias_term)
    {
        cmd.record_upload(bias_data, bias_data_gpu, opt);

        bias_data.release();
    }

    return 0;
}

int DeformableConv2D_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    const VkMat& offset_packed = bottom_blobs[1];

    const bool has_mask = (bottom_blobs.size() == 3);

    VkMat offset;
    if (offset_packed.elempack != 1)
    {
        vkdev->convert_packing(offset_packed, offset, 1, cmd, opt);
    }
    else
    {
        offset = offset_packed;
    }

    VkMat mask;
    VkMat mask_packed;
    if (has_mask)
    {
        mask_packed = bottom_blobs[2];
        if (mask_packed.elempack != 1)
        {
            vkdev->convert_packing(mask_packed, mask, 1, cmd, opt);
        }
        else
        {
            mask = mask_packed;
        }
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
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
    int channels = bottom_blob_bordered.c;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = (h - kernel_extent_h) / stride_h + 1;

    const int out_elempack = num_output % 4 == 0 ? 4 : 1;
    const size_t out_elemsize = elemsize / elempack * out_elempack;

    VkMat& top_blob = top_blobs[0];
    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    const int num_output_packed = (num_output + 3) / 4 * 4;
    const int c_iterations = channels / elempack;
    const int cstep_scalar = bottom_blob_bordered.cstep;
    const int outc_pack4 = num_output_packed / 4;
    const int outcstep_scalar = (out_elempack == 4) ? top_blob.cstep : (top_blob.cstep * 4);

    std::vector<VkMat> bindings(8);
    bindings[0] = bottom_blob_bordered;
    bindings[1] = top_blob;
    bindings[2] = bottom_blob_bordered;
    bindings[3] = top_blob;
    bindings[4] = offset;
    bindings[5] = has_mask ? mask : VkMat();
    bindings[6] = weight_data_gpu;
    bindings[7] = bias_data_gpu;

    std::vector<vk_constant_type> constants(13);
    constants[0].i = bottom_blob_bordered.dims;
    constants[1].i = bottom_blob_bordered.w;
    constants[2].i = bottom_blob_bordered.h;
    constants[3].i = c_iterations;
    constants[4].i = cstep_scalar;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = outc_pack4;
    constants[9].i = outcstep_scalar;
    constants[10].i = num_output;
    constants[11].i = offset.cstep;
    constants[12].i = has_mask ? mask.cstep : 0;

    VkMat dispatcher;
    dispatcher.w = top_blob.w;
    dispatcher.h = top_blob.h;
    dispatcher.c = outc_pack4;

    Pipeline* pipeline = has_mask ? pipeline_deformableconv2d_packed_mask : pipeline_deformableconv2d_packed;
    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn

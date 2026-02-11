// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "unfold_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

Unfold_vulkan::Unfold_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;
    support_vulkan_any_packing = true;

    padding = 0;

    pipeline_unfold_im2col = 0;
    pipeline_unfold_im2col_pack4 = 0;
    pipeline_unfold_im2col_pack1to4 = 0;
    pipeline_unfold_im2col_pack4to1 = 0;
}

int Unfold_vulkan::load_param(const ParamDict& pd)
{
    return Unfold::load_param(pd);
}

int Unfold_vulkan::create_pipeline(const Option& opt)
{
    padding = ncnn::create_layer_vulkan(ncnn::LayerType::Padding);
    padding->vkdev = vkdev;
    {
        ncnn::ParamDict pd;

        pd.set(0, 0);
        pd.set(1, 0);
        pd.set(2, 0);
        pd.set(3, 0);
        pd.set(4, 0);
        pd.set(5, pad_value);
        pd.set(6, 0);
        pd.set(7, 0);
        pd.set(8, 0);

        padding->load_param(pd);
        padding->load_model(ModelBinFromMatArray(0));
        padding->create_pipeline(opt);
    }

    {
        pipeline_unfold_im2col = new Pipeline(vkdev);
        pipeline_unfold_im2col->set_local_size_xyz(8, 8, 1);
        std::vector<vk_specialization_type> specializations;
        pipeline_unfold_im2col->create(LayerShaderType::unfold_im2col, opt, specializations);
    }

    {
        pipeline_unfold_im2col_pack4 = new Pipeline(vkdev);
        pipeline_unfold_im2col_pack4->set_local_size_xyz(8, 8, 1);
        std::vector<vk_specialization_type> specializations;
        pipeline_unfold_im2col_pack4->create(LayerShaderType::unfold_im2col_pack4, opt, specializations);
    }

    {
        pipeline_unfold_im2col_pack1to4 = new Pipeline(vkdev);
        pipeline_unfold_im2col_pack1to4->set_local_size_xyz(8, 8, 1);
        std::vector<vk_specialization_type> specializations;
        pipeline_unfold_im2col_pack1to4->create(LayerShaderType::unfold_im2col_pack1to4, opt, specializations);
    }

    {
        pipeline_unfold_im2col_pack4to1 = new Pipeline(vkdev);
        pipeline_unfold_im2col_pack4to1->set_local_size_xyz(8, 8, 1);
        std::vector<vk_specialization_type> specializations;
        pipeline_unfold_im2col_pack4to1->create(LayerShaderType::unfold_im2col_pack4to1, opt, specializations);
    }

    return 0;
}

int Unfold_vulkan::destroy_pipeline(const Option& opt)
{
    if (padding)
    {
        padding->destroy_pipeline(opt);
        delete padding;
        padding = 0;
    }

    delete pipeline_unfold_im2col;
    pipeline_unfold_im2col = 0;

    delete pipeline_unfold_im2col_pack4;
    pipeline_unfold_im2col_pack4 = 0;

    delete pipeline_unfold_im2col_pack1to4;
    pipeline_unfold_im2col_pack1to4 = 0;

    delete pipeline_unfold_im2col_pack4to1;
    pipeline_unfold_im2col_pack4to1 = 0;

    return 0;
}

int Unfold_vulkan::make_padding(const VkMat& bottom_blob, VkMat& bottom_blob_bordered, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int pl = pad_left;
    int pr = pad_right;
    int pt = pad_top;
    int pb = pad_bottom;

    if (pl > 0 || pr > 0 || pt > 0 || pb > 0)
    {
        // Explicit positive padding specified; no adjustment needed here.
    }
    else if (pl == -233 && pr == -233 && pt == -233 && pb == -233)
    {
        // tensorflow padding=SAME or onnx padding=SAME_UPPER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;

        if (wpad > 0 || hpad > 0)
        {
            pl = wpad / 2;
            pr = wpad - pl;
            pt = hpad / 2;
            pb = hpad - pt;
        }
        else
        {
            pl = pr = pt = pb = 0;
        }
    }
    else if (pl == -234 && pr == -234 && pt == -234 && pb == -234)
    {
        // onnx padding=SAME_LOWER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;

        if (wpad > 0 || hpad > 0)
        {
            pr = wpad / 2;
            pl = wpad - pr;
            pb = hpad / 2;
            pt = hpad - pb;
        }
        else
        {
            pl = pr = pt = pb = 0;
        }
    }
    else
    {
        pl = pr = pt = pb = 0;
    }

    if (pl == 0 && pr == 0 && pt == 0 && pb == 0)
    {
        bottom_blob_bordered = bottom_blob;
        return 0;
    }

    if (!padding)
        return -1;

    VkMat reference_blob;
    reference_blob.create(6, (size_t)4u, 1, opt.staging_vkallocator);
    if (reference_blob.empty())
        return -100;

    int* param_data = reference_blob.mapped();
    param_data[0] = pt;
    param_data[1] = pb;
    param_data[2] = pl;
    param_data[3] = pr;
    param_data[4] = 0;
    param_data[5] = 0;

    std::vector<VkMat> inputs(2);
    inputs[0] = bottom_blob;
    inputs[1] = reference_blob;

    std::vector<VkMat> outputs(1);
    outputs[0] = VkMat();

    Option opt_pad = opt;
    opt_pad.blob_vkallocator = opt.workspace_vkallocator;

    int ret = padding->forward(inputs, outputs, cmd, opt_pad);
    if (ret != 0)
        return ret;

    bottom_blob_bordered = outputs[0];
    return 0;
}

int Unfold_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    const int in_elempack = bottom_blob.elempack;
    if (in_elempack != 1 && in_elempack != 4)
        return -1;

    VkMat bottom_blob_bordered;
    int ret = make_padding(bottom_blob, bottom_blob_bordered, cmd, opt);
    if (ret != 0)
        return ret;

    const int elempack = bottom_blob_bordered.elempack;
    if (elempack != 1 && elempack != 4)
        return -1;

    const int w = bottom_blob_bordered.w;
    const int h = bottom_blob_bordered.h;
    const int channels_packed = bottom_blob_bordered.c;
    const int channels = channels_packed * elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = (h - kernel_extent_h) / stride_h + 1;

    const int size = outw * outh;
    const int maxk = kernel_w * kernel_h;

    const int out_h = maxk * channels;

    int out_elempack = 1;
    if (opt.use_packing_layout)
        out_elempack = out_h % 4 == 0 ? 4 : 1;

    const size_t out_elemsize = bottom_blob_bordered.elemsize / elempack * out_elempack;

    top_blob.create(size, out_h / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob_bordered;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(12);
    constants[0].i = w;
    constants[1].i = h;
    constants[2].i = channels;
    constants[3].i = outw;
    constants[4].i = outh;
    constants[5].i = kernel_w;
    constants[6].i = kernel_h;
    constants[7].i = dilation_w;
    constants[8].i = dilation_h;
    constants[9].i = stride_w;
    constants[10].i = stride_h;
    constants[11].i = (int)bottom_blob_bordered.cstep;

    VkMat dispatcher;
    dispatcher.w = size;
    dispatcher.h = top_blob.h;
    dispatcher.c = 1;

    Pipeline* pipeline = 0;
    if (elempack == 1 && out_elempack == 1)
        pipeline = pipeline_unfold_im2col;
    else if (elempack == 4 && out_elempack == 4)
        pipeline = pipeline_unfold_im2col_pack4;
    else if (elempack == 1 && out_elempack == 4)
        pipeline = pipeline_unfold_im2col_pack1to4;
    else if (elempack == 4 && out_elempack == 1)
        pipeline = pipeline_unfold_im2col_pack4to1;

    if (!pipeline)
        return -1;
    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn

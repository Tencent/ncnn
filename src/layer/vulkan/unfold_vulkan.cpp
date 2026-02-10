// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "unfold_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Unfold_vulkan::Unfold_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = false;
    support_any_packing = false;

    pipeline_unfold_im2col = 0;
    pipeline_unfold_padding = 0;
}

int Unfold_vulkan::load_param(const ParamDict& pd)
{
    return Unfold::load_param(pd);
}

int Unfold_vulkan::create_pipeline(const Option& opt)
{
    {
        pipeline_unfold_padding = new Pipeline(vkdev);
        pipeline_unfold_padding->set_local_size_xyz(8, 8, 1);
        std::vector<vk_specialization_type> specializations;
        pipeline_unfold_padding->create(LayerShaderType::unfold_padding, opt, specializations);
    }

    {
        pipeline_unfold_im2col = new Pipeline(vkdev);
        pipeline_unfold_im2col->set_local_size_xyz(8, 8, 1);
        std::vector<vk_specialization_type> specializations;
        pipeline_unfold_im2col->create(LayerShaderType::unfold_im2col, opt, specializations);
    }

    return 0;
}

int Unfold_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_unfold_padding;
    pipeline_unfold_padding = 0;

    delete pipeline_unfold_im2col;
    pipeline_unfold_im2col = 0;

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
    }
    else if (pl == -233 && pr == -233 && pt == -233 && pb == -233)
    {
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

    const int outw = w + pl + pr;
    const int outh = h + pt + pb;
    const int channels = bottom_blob.c;

    bottom_blob_bordered.create(outw, outh, channels, bottom_blob.elemsize, 1, opt.workspace_vkallocator);
    if (bottom_blob_bordered.empty())
        return -100;

    const int src_cstep = (int)bottom_blob.cstep;
    const int dst_cstep = (int)bottom_blob_bordered.cstep;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = bottom_blob_bordered;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = w;
    constants[1].i = h;
    constants[2].i = outw;
    constants[3].i = outh;
    constants[4].i = channels;
    constants[5].i = pl;
    constants[6].i = pt;
    constants[7].f = pad_value;
    constants[8].i = src_cstep;
    constants[9].i = dst_cstep;

    VkMat dispatcher;
    dispatcher.w = outw;
    dispatcher.h = outh;
    dispatcher.c = channels;

    cmd.record_pipeline(pipeline_unfold_padding, bindings, constants, dispatcher);

    return 0;
}

int Unfold_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    if (bottom_blob.elempack != 1)
        return -1;

    VkMat bottom_blob_bordered;
    int ret = make_padding(bottom_blob, bottom_blob_bordered, cmd, opt);
    if (ret != 0)
        return ret;

    const int w = bottom_blob_bordered.w;
    const int h = bottom_blob_bordered.h;
    const int channels = bottom_blob_bordered.c;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = (h - kernel_extent_h) / stride_h + 1;

    const int size = outw * outh;
    const int maxk = kernel_w * kernel_h;

    top_blob.create(size, maxk * channels, bottom_blob_bordered.elemsize, 1, opt.blob_vkallocator);
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
    constants[11].i = bottom_blob_bordered.cstep;

    VkMat dispatcher;
    dispatcher.w = size;
    dispatcher.h = maxk * channels;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline_unfold_im2col, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn

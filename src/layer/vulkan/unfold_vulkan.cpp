// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "unfold_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Unfold_vulkan::Unfold_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;
    support_vulkan_any_packing = true;

    pipeline_unfold_im2col = 0;
    pipeline_unfold_im2col_pack4 = 0;
    pipeline_unfold_im2col_pack1to4 = 0;
    pipeline_unfold_im2col_pack4to1 = 0;
}

int Unfold_vulkan::create_pipeline(const Option& opt)
{
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

int Unfold_vulkan::destroy_pipeline(const Option& /*opt*/)
{
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

int Unfold_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    const int in_elempack = bottom_blob.elempack;
    if (in_elempack != 1 && in_elempack != 4)
        return -1;

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = in_elempack;

    const int channels_packed = bottom_blob.c;
    const int channels = channels_packed * elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int pl = pad_left;
    int pr = pad_right;
    int pt = pad_top;
    int pb = pad_bottom;

    if (pl > 0 || pr > 0 || pt > 0 || pb > 0)
    {
        // explicit pad
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

    const int wpaded = w + pl + pr;
    const int hpaded = h + pt + pb;

    const int outw = (wpaded - kernel_extent_w) / stride_w + 1;
    const int outh = (hpaded - kernel_extent_h) / stride_h + 1;

    const int size = outw * outh;
    const int maxk = kernel_w * kernel_h;
    const int out_h = maxk * channels;

    int out_elempack = 1;
    if (opt.use_packing_layout)
        out_elempack = out_h % 4 == 0 ? 4 : 1;

    const size_t out_elemsize = bottom_blob.elemsize / elempack * out_elempack;

    top_blob.create(size, out_h / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(15);
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
    constants[11].i = (int)bottom_blob.cstep;
    constants[12].i = pl;
    constants[13].i = pt;
    constants[14].f = pad_value;

    VkMat dispatcher;
    dispatcher.w = size;
    dispatcher.c = 1;

    Pipeline* pipeline = 0;

    if (elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_unfold_im2col;
        dispatcher.h = top_blob.h;
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        pipeline = pipeline_unfold_im2col_pack4;
        dispatcher.h = top_blob.h;
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_unfold_im2col_pack1to4;
        dispatcher.h = top_blob.h;
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        pipeline = pipeline_unfold_im2col_pack4to1;
        dispatcher.h = (maxk * channels) / 4;
    }

    if (!pipeline)
        return -1;

    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn

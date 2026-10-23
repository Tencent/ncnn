// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fold_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Fold_vulkan::Fold_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;
    support_vulkan_any_packing = true;

    pipeline_fold_col2im = 0;
    pipeline_fold_col2im_pack4 = 0;
    pipeline_fold_col2im_pack1to4 = 0;
    pipeline_fold_col2im_pack4to1 = 0;
}

int Fold_vulkan::create_pipeline(const Option& opt)
{
    {
        pipeline_fold_col2im = new Pipeline(vkdev);
        pipeline_fold_col2im->set_local_size_xyz(8, 8, 1);
        std::vector<vk_specialization_type> specializations;
        pipeline_fold_col2im->create(LayerShaderType::fold_col2im, opt, specializations);
    }

    {
        pipeline_fold_col2im_pack4 = new Pipeline(vkdev);
        pipeline_fold_col2im_pack4->set_local_size_xyz(8, 8, 1);
        std::vector<vk_specialization_type> specializations;
        pipeline_fold_col2im_pack4->create(LayerShaderType::fold_col2im_pack4, opt, specializations);
    }

    {
        pipeline_fold_col2im_pack1to4 = new Pipeline(vkdev);
        pipeline_fold_col2im_pack1to4->set_local_size_xyz(8, 8, 1);
        std::vector<vk_specialization_type> specializations;
        pipeline_fold_col2im_pack1to4->create(LayerShaderType::fold_col2im_pack1to4, opt, specializations);
    }

    {
        pipeline_fold_col2im_pack4to1 = new Pipeline(vkdev);
        pipeline_fold_col2im_pack4to1->set_local_size_xyz(8, 8, 1);
        std::vector<vk_specialization_type> specializations;
        pipeline_fold_col2im_pack4to1->create(LayerShaderType::fold_col2im_pack4to1, opt, specializations);
    }

    return 0;
}

int Fold_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_fold_col2im;
    pipeline_fold_col2im = 0;

    delete pipeline_fold_col2im_pack4;
    pipeline_fold_col2im_pack4 = 0;

    delete pipeline_fold_col2im_pack1to4;
    pipeline_fold_col2im_pack1to4 = 0;

    delete pipeline_fold_col2im_pack4to1;
    pipeline_fold_col2im_pack4to1 = 0;

    return 0;
}

int Fold_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    const int in_elempack = bottom_blob.elempack;
    if (in_elempack != 1 && in_elempack != 4)
        return -1;

    // input is a 2d blob (size, maxk * channels)
    const int size = bottom_blob.w;
    const int maxk = kernel_w * kernel_h;
    const int channels = bottom_blob.h * in_elempack / maxk;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int outw = output_w + pad_left + pad_right;
    const int outh = output_h + pad_top + pad_bottom;

    const int inw = (outw - kernel_extent_w) / stride_w + 1;
    const int inh = (outh - kernel_extent_h) / stride_h + 1;

    if (inw <= 0 || inh <= 0 || inw * inh != size)
        return -1;

    const int out_elempack = opt.use_packing_layout && channels % 4 == 0 ? 4 : 1;
    const size_t out_elemsize = bottom_blob.elemsize / in_elempack * out_elempack;

    // the shader emits the cut output directly; pad_left/pad_top fold into
    // the bordered-space gather offsets
    top_blob.create(output_w, output_h, channels / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(15);
    constants[0].i = size;
    constants[1].i = maxk;
    constants[2].i = inw;
    constants[3].i = inh;
    constants[4].i = kernel_w;
    constants[5].i = kernel_h;
    constants[6].i = dilation_w;
    constants[7].i = dilation_h;
    constants[8].i = stride_w;
    constants[9].i = stride_h;
    constants[10].i = pad_left;
    constants[11].i = pad_top;
    constants[12].i = output_w;
    constants[13].i = output_h;
    constants[14].i = (int)top_blob.cstep;

    Pipeline* pipeline = 0;
    if (in_elempack == 1 && out_elempack == 1)
        pipeline = pipeline_fold_col2im;
    else if (in_elempack == 4 && out_elempack == 4)
        pipeline = pipeline_fold_col2im_pack4;
    else if (in_elempack == 1 && out_elempack == 4)
        pipeline = pipeline_fold_col2im_pack1to4;
    else if (in_elempack == 4 && out_elempack == 1)
        pipeline = pipeline_fold_col2im_pack4to1;

    if (!pipeline)
        return -1;

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn

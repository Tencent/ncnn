// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "shufflechannel_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

ShuffleChannel_vulkan::ShuffleChannel_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_shufflechannel = 0;
    pipeline_shufflechannel_pack4 = 0;
}

int ShuffleChannel_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    std::vector<vk_specialization_type> specializations(2 + 10);
    specializations[0].i = reverse ? shape.c * shape.elempack / group : group;
    specializations[1].i = vkdev->info.bug_implicit_fp16_arithmetic();
    specializations[2 + 0].i = shape.dims;
    specializations[2 + 1].i = shape.w;
    specializations[2 + 2].i = shape.h;
    specializations[2 + 3].i = shape.c;
    specializations[2 + 4].i = shape.cstep;
    specializations[2 + 5].i = out_shape.dims;
    specializations[2 + 6].i = out_shape.w;
    specializations[2 + 7].i = out_shape.h;
    specializations[2 + 8].i = out_shape.c;
    specializations[2 + 9].i = out_shape.cstep;

    Mat local_size_xyz;
    if (out_shape.dims != 0)
    {
        local_size_xyz.w = std::min(4, out_shape.w);
        local_size_xyz.h = std::min(4, out_shape.h);
        local_size_xyz.c = std::min(4, out_shape.c);
    }

    // pack1
    if (shape.dims == 0 || shape.elempack == 1)
    {
        pipeline_shufflechannel = new Pipeline(vkdev);
        pipeline_shufflechannel->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_shufflechannel->create(LayerShaderType::shufflechannel, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || shape.elempack == 4)
    {
        pipeline_shufflechannel_pack4 = new Pipeline(vkdev);
        pipeline_shufflechannel_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_shufflechannel_pack4->create(LayerShaderType::shufflechannel_pack4, opt, specializations);
    }

    return 0;
}

int ShuffleChannel_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_shufflechannel;
    pipeline_shufflechannel = 0;

    delete pipeline_shufflechannel_pack4;
    pipeline_shufflechannel_pack4 = 0;

    return 0;
}

int ShuffleChannel_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    top_blob.create(w, h, channels, elemsize, elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(11);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.c;
    constants[4].i = bottom_blob.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;
    constants[10].i = reverse ? channels * elempack / group : group;

    const Pipeline* pipeline = elempack == 4 ? pipeline_shufflechannel_pack4 : pipeline_shufflechannel;

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn

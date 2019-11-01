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

#include "lrn_vulkan.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(LRN_vulkan)

LRN_vulkan::LRN_vulkan()
{
    support_vulkan = true;

    pipeline_lrn_square_pad = 0;
    pipeline_lrn_norm = 0;
    pipeline_lrn_square_pad_across_channel_pack4 = 0;
    pipeline_lrn_norm_across_channel_pack4 = 0;
    pipeline_lrn_square_pad_within_channel_pack4 = 0;
    pipeline_lrn_norm_within_channel_pack4 = 0;
}

int LRN_vulkan::create_pipeline(const Option& opt)
{
    {
        pipeline_lrn_square_pad = new Pipeline(vkdev);
        pipeline_lrn_square_pad->set_optimal_local_size_xyz();

        std::vector<vk_specialization_type> specializations(3);
        specializations[0].i = region_type;

        int pad = local_size / 2;
        if (pad == 0)
        {
            specializations[1].i = 0;
            specializations[2].i = 0;
        }
        else
        {
            specializations[1].i = pad;
            specializations[2].i = local_size - pad - 1;
        }

        pipeline_lrn_square_pad->create("lrn_square_pad", opt, specializations, 2, 10);

        // pack4
        if (region_type == 0)
        {
            pipeline_lrn_square_pad_across_channel_pack4 = new Pipeline(vkdev);
            pipeline_lrn_square_pad_across_channel_pack4->set_optimal_local_size_xyz();
            pipeline_lrn_square_pad_across_channel_pack4->create("lrn_square_pad_across_channel_pack4", opt, specializations, 2, 10);
        }
        if (region_type == 1)
        {
            pipeline_lrn_square_pad_within_channel_pack4 = new Pipeline(vkdev);
            pipeline_lrn_square_pad_within_channel_pack4->set_optimal_local_size_xyz();
            pipeline_lrn_square_pad_within_channel_pack4->create("lrn_square_pad_within_channel_pack4", opt, specializations, 2, 10);
        }
    }

    {
        pipeline_lrn_norm = new Pipeline(vkdev);
        pipeline_lrn_norm->set_optimal_local_size_xyz();

        std::vector<vk_specialization_type> specializations(5);
        specializations[0].i = region_type;
        specializations[1].i = local_size;
        specializations[2].f = alpha;
        specializations[3].f = beta;
        specializations[4].f = bias;

        pipeline_lrn_norm->create("lrn_norm", opt, specializations, 2, 10);

        // pack4
        if (region_type == 0)
        {
            pipeline_lrn_norm_across_channel_pack4 = new Pipeline(vkdev);
            pipeline_lrn_norm_across_channel_pack4->set_optimal_local_size_xyz();
            pipeline_lrn_norm_across_channel_pack4->create("lrn_norm_across_channel_pack4", opt, specializations, 2, 10);
        }
        if (region_type == 1)
        {
            pipeline_lrn_norm_within_channel_pack4 = new Pipeline(vkdev);
            pipeline_lrn_norm_within_channel_pack4->set_optimal_local_size_xyz();
            pipeline_lrn_norm_within_channel_pack4->create("lrn_norm_within_channel_pack4", opt, specializations, 2, 10);
        }
    }

    return 0;
}

int LRN_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_lrn_square_pad;
    pipeline_lrn_square_pad = 0;

    delete pipeline_lrn_norm;
    pipeline_lrn_norm = 0;

    delete pipeline_lrn_square_pad_across_channel_pack4;
    pipeline_lrn_square_pad_across_channel_pack4 = 0;

    delete pipeline_lrn_norm_across_channel_pack4;
    pipeline_lrn_norm_across_channel_pack4 = 0;

    delete pipeline_lrn_square_pad_within_channel_pack4;
    pipeline_lrn_square_pad_within_channel_pack4 = 0;

    delete pipeline_lrn_norm_within_channel_pack4;
    pipeline_lrn_norm_within_channel_pack4 = 0;

    return 0;
}

int LRN_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    size_t elemsize = bottom_top_blob.elemsize;
    int elempack = bottom_top_blob.elempack;

    VkMat square_workspace;

    int pad = local_size / 2;
    if (pad == 0)
    {
        square_workspace.create(w, h, channels, elemsize, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
    }
    else if (region_type == NormRegion_ACROSS_CHANNELS)
    {
        // always create scalar square workspace blob for norm across channel
        square_workspace.create(w, h, channels * elempack + local_size - 1, 4u, 1, opt.workspace_vkallocator, opt.staging_vkallocator);
    }
    else if (region_type == NormRegion_WITHIN_CHANNEL)
    {
        square_workspace.create(w + local_size - 1, h + local_size - 1, channels, elemsize, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
    }

    // square pad
    {
    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_top_blob;
    bindings[1] = square_workspace;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;
    constants[5].i = square_workspace.dims;
    constants[6].i = square_workspace.w;
    constants[7].i = square_workspace.h;
    constants[8].i = square_workspace.c;
    constants[9].i = square_workspace.cstep;

    const Pipeline* pipeline = 0;
    if (elempack == 4)
    {
        if (region_type == 0) pipeline = pipeline_lrn_square_pad_across_channel_pack4;
        if (region_type == 1) pipeline = pipeline_lrn_square_pad_within_channel_pack4;
    }
    else
    {
        pipeline = pipeline_lrn_square_pad;
    }

    cmd.record_pipeline(pipeline, bindings, constants, square_workspace);
    }

    // norm
    {
    std::vector<VkMat> bindings(2);
    bindings[0] = square_workspace;
    bindings[1] = bottom_top_blob;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = square_workspace.dims;
    constants[1].i = square_workspace.w;
    constants[2].i = square_workspace.h;
    constants[3].i = square_workspace.c;
    constants[4].i = square_workspace.cstep;
    constants[5].i = bottom_top_blob.dims;
    constants[6].i = bottom_top_blob.w;
    constants[7].i = bottom_top_blob.h;
    constants[8].i = bottom_top_blob.c;
    constants[9].i = bottom_top_blob.cstep;

    const Pipeline* pipeline = 0;
    if (elempack == 4)
    {
        if (region_type == 0) pipeline = pipeline_lrn_norm_across_channel_pack4;
        if (region_type == 1) pipeline = pipeline_lrn_norm_within_channel_pack4;
    }
    else
    {
        pipeline = pipeline_lrn_norm;
    }

    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    return 0;
}

} // namespace ncnn

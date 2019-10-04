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

#include "permute_vulkan.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Permute_vulkan)

Permute_vulkan::Permute_vulkan()
{
    support_vulkan = true;

    pipeline_permute = 0;
    pipeline_permute_pack4to1 = 0;
}

int Permute_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations(1);
    specializations[0].i = order_type;

    // pack1
    {
        pipeline_permute = new Pipeline(vkdev);
        pipeline_permute->set_optimal_local_size_xyz();
        pipeline_permute->create("permute", opt, specializations, 2, 10);
    }

    // pack4
    {
        pipeline_permute_pack4to1 = new Pipeline(vkdev);
        pipeline_permute_pack4to1->set_optimal_local_size_xyz();
        pipeline_permute_pack4to1->create("permute_pack4to1", opt, specializations, 2, 10);
    }

    return 0;
}

int Permute_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_permute;
    pipeline_permute = 0;

    delete pipeline_permute_pack4to1;
    pipeline_permute_pack4to1 = 0;

    return 0;
}

int Permute_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int dims = bottom_blob.dims;

    int out_elempack = 1;
    size_t out_elemsize = elemsize / elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 4) out_elemsize = 4*2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    if (dims == 2)
    {
        // order_type
        // 0 = w h
        // 1 = h w

        h = h * elempack;

        if (order_type == 0)
        {
            top_blob.create(w, h, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;
        }
        else if (order_type == 1)
        {
            top_blob.create(h, w, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;
        }
    }
    else if (dims == 3)
    {
        // order_type
        // 0 = w h c
        // 1 = h w c
        // 2 = w c h
        // 3 = c w h
        // 4 = h c w
        // 5 = c h w

        channels = channels * elempack;

        if (order_type == 0)
        {
            top_blob.create(w, h, channels, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;
        }
        else if (order_type == 1)
        {
            top_blob.create(h, w, channels, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;
        }
        else if (order_type == 2)
        {
            top_blob.create(w, channels, h, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;
        }
        else if (order_type == 3)
        {
            top_blob.create(channels, w, h, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;
        }
        else if (order_type == 4)
        {
            top_blob.create(h, channels, w, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;
        }
        else if (order_type == 5)
        {
            top_blob.create(channels, h, w, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;
        }
    }

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(10);
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

    if (elempack == 1)
    {
        cmd.record_pipeline(pipeline_permute, bindings, constants, top_blob);
    }

    if (elempack == 4)
    {
        cmd.record_pipeline(pipeline_permute_pack4to1, bindings, constants, bottom_blob);
    }

    return 0;
}

} // namespace ncnn

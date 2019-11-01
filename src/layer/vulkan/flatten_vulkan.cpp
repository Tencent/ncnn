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

#include "flatten_vulkan.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Flatten_vulkan)

Flatten_vulkan::Flatten_vulkan()
{
    support_vulkan = true;

    pipeline_flatten = 0;
    pipeline_flatten_pack4 = 0;
    pipeline_flatten_pack1to4 = 0;
}

int Flatten_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations;

    // pack1
    {
        pipeline_flatten = new Pipeline(vkdev);
        pipeline_flatten->set_optimal_local_size_xyz();
        pipeline_flatten->create("flatten", opt, specializations, 2, 10);
    }

    // pack4
    {
        pipeline_flatten_pack4 = new Pipeline(vkdev);
        pipeline_flatten_pack4->set_optimal_local_size_xyz();
        pipeline_flatten_pack4->create("flatten_pack4", opt, specializations, 2, 10);
    }

    // pack1to4
    {
        pipeline_flatten_pack1to4 = new Pipeline(vkdev);
        pipeline_flatten_pack1to4->set_optimal_local_size_xyz();
        pipeline_flatten_pack1to4->create("flatten_pack1to4", opt, specializations, 2, 10);
    }

    return 0;
}

int Flatten_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_flatten;
    pipeline_flatten = 0;

    delete pipeline_flatten_pack4;
    pipeline_flatten_pack4 = 0;

    delete pipeline_flatten_pack1to4;
    pipeline_flatten_pack1to4 = 0;

    return 0;
}

int Flatten_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int total = w * h * channels * elempack;

    int out_elempack = total % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 4) out_elemsize = 4*2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    if (dims == 2 && elempack == 1)
    {
        top_blob = bottom_blob;
        top_blob.dims = 1;
        top_blob.w = total / out_elempack;
        top_blob.h = 1;
        top_blob.cstep = top_blob.w;
        top_blob.elemsize = out_elemsize;
        top_blob.elempack = out_elempack;
        return 0;
    }

    top_blob.create(total / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

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

    const Pipeline* pipeline = 0;
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_flatten;
    }
    else if (elempack == 4 /*&& out_elempack == 4*/)
    {
        pipeline = pipeline_flatten_pack4;
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_flatten_pack1to4;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn

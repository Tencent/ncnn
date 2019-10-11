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

#include "eltwise_vulkan.h"
#include <algorithm>

namespace ncnn {

DEFINE_LAYER_CREATOR(Eltwise_vulkan)

Eltwise_vulkan::Eltwise_vulkan()
{
    support_vulkan = true;

    pipeline_eltwise[0] = 0;
    pipeline_eltwise[1] = 0;
    pipeline_eltwise_pack4[0] = 0;
    pipeline_eltwise_pack4[1] = 0;
}

int Eltwise_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations(2);
    specializations[0].i = op_type;
    specializations[1].i = coeffs.w == 0 ? 0 : 1;

    // pack1
    {
        pipeline_eltwise[0] = new Pipeline(vkdev);
        pipeline_eltwise[0]->set_optimal_local_size_xyz();
        pipeline_eltwise[0]->create("eltwise", opt, specializations, 3, 5+2);
        pipeline_eltwise[1] = new Pipeline(vkdev);
        pipeline_eltwise[1]->set_optimal_local_size_xyz();
        pipeline_eltwise[1]->create("eltwise", opt, specializations, 3, 5+2);
    }

    // pack4
    {
        pipeline_eltwise_pack4[0] = new Pipeline(vkdev);
        pipeline_eltwise_pack4[0]->set_optimal_local_size_xyz();
        pipeline_eltwise_pack4[0]->create("eltwise_pack4", opt, specializations, 3, 5+2);
        pipeline_eltwise_pack4[1] = new Pipeline(vkdev);
        pipeline_eltwise_pack4[1]->set_optimal_local_size_xyz();
        pipeline_eltwise_pack4[1]->create("eltwise_pack4", opt, specializations, 3, 5+2);
    }

    return 0;
}

int Eltwise_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_eltwise[0];
    delete pipeline_eltwise[1];
    pipeline_eltwise[0] = 0;
    pipeline_eltwise[1] = 0;

    delete pipeline_eltwise_pack4[0];
    delete pipeline_eltwise_pack4[1];
    pipeline_eltwise_pack4[0] = 0;
    pipeline_eltwise_pack4[1] = 0;

    return 0;
}

int Eltwise_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    const VkMat& bottom_blob1 = bottom_blobs[1];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    VkMat& top_blob = top_blobs[0];
    top_blob.create(w, h, channels, elemsize, elempack, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(3);
    bindings[0] = bottom_blob;
    bindings[1] = bottom_blob1;
    bindings[2] = top_blob;

    std::vector<vk_constant_type> constants(5 + 2);
    constants[0].i = top_blob.dims;
    constants[1].i = top_blob.w;
    constants[2].i = top_blob.h;
    constants[3].i = top_blob.c;
    constants[4].i = top_blob.cstep;
    constants[5].f = coeffs.w == 0 ? 1.f : coeffs[0];
    constants[6].f = coeffs.w == 0 ? 1.f : coeffs[1];

    const Pipeline* pipeline = elempack == 4 ? pipeline_eltwise_pack4[1] : pipeline_eltwise[1];

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    for (size_t b=2; b<bottom_blobs.size(); b++)
    {
        std::vector<VkMat> bindings(3);
        bindings[0] = top_blob;
        bindings[1] = bottom_blobs[b];
        bindings[2] = top_blob;// TODO use separated pipeline ?

        std::vector<vk_constant_type> constants(5 + 2);
        constants[0].i = top_blob.dims;
        constants[1].i = top_blob.w;
        constants[2].i = top_blob.h;
        constants[3].i = top_blob.c;
        constants[4].i = top_blob.cstep;
        constants[5].f = 1.f;
        constants[6].f = coeffs.w == 0 ? 1 : coeffs[b];

        const Pipeline* pipeline = elempack == 4 ? pipeline_eltwise_pack4[b%2] : pipeline_eltwise[b%2];

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    }

    return 0;
}

} // namespace ncnn

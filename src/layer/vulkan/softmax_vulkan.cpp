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

#include "softmax_vulkan.h"
#include <float.h>
#include <math.h>
#include <algorithm>

namespace ncnn {

DEFINE_LAYER_CREATOR(Softmax_vulkan)

Softmax_vulkan::Softmax_vulkan()
{
    support_vulkan = true;

    pipeline_softmax_reduce_max = 0;
    pipeline_softmax_exp_sub_max = 0;
    pipeline_softmax_reduce_sum = 0;
    pipeline_softmax_div_sum = 0;

    pipeline_softmax_reduce_max_pack4 = 0;
    pipeline_softmax_exp_sub_max_pack4 = 0;
    pipeline_softmax_reduce_sum_pack4 = 0;
    pipeline_softmax_div_sum_pack4 = 0;
}

int Softmax_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations(1);
    specializations[0].i = axis;

    // pack1
    {
        pipeline_softmax_reduce_max = new Pipeline(vkdev);
        pipeline_softmax_exp_sub_max = new Pipeline(vkdev);
        pipeline_softmax_reduce_sum = new Pipeline(vkdev);
        pipeline_softmax_div_sum = new Pipeline(vkdev);

        pipeline_softmax_reduce_max->set_optimal_local_size_xyz();
        pipeline_softmax_exp_sub_max->set_optimal_local_size_xyz();
        pipeline_softmax_reduce_sum->set_optimal_local_size_xyz();
        pipeline_softmax_div_sum->set_optimal_local_size_xyz();

        pipeline_softmax_reduce_max->create("softmax_reduce_max", opt, specializations, 2, 10);
        pipeline_softmax_exp_sub_max->create("softmax_exp_sub_max", opt, specializations, 2, 10);
        pipeline_softmax_reduce_sum->create("softmax_reduce_sum", opt, specializations, 2, 10);
        pipeline_softmax_div_sum->create("softmax_div_sum", opt, specializations, 2, 10);
    }

    // pack4
    {
        pipeline_softmax_reduce_max_pack4 = new Pipeline(vkdev);
        pipeline_softmax_exp_sub_max_pack4 = new Pipeline(vkdev);
        pipeline_softmax_reduce_sum_pack4 = new Pipeline(vkdev);
        pipeline_softmax_div_sum_pack4 = new Pipeline(vkdev);

        pipeline_softmax_reduce_max_pack4->set_optimal_local_size_xyz();
        pipeline_softmax_exp_sub_max_pack4->set_optimal_local_size_xyz();
        pipeline_softmax_reduce_sum_pack4->set_optimal_local_size_xyz();
        pipeline_softmax_div_sum_pack4->set_optimal_local_size_xyz();

        pipeline_softmax_reduce_max_pack4->create("softmax_reduce_max_pack4", opt, specializations, 2, 10);
        pipeline_softmax_exp_sub_max_pack4->create("softmax_exp_sub_max_pack4", opt, specializations, 2, 10);
        pipeline_softmax_reduce_sum_pack4->create("softmax_reduce_sum_pack4", opt, specializations, 2, 10);
        pipeline_softmax_div_sum_pack4->create("softmax_div_sum_pack4", opt, specializations, 2, 10);
    }

    return 0;
}

int Softmax_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_softmax_reduce_max;
    pipeline_softmax_reduce_max = 0;

    delete pipeline_softmax_exp_sub_max;
    pipeline_softmax_exp_sub_max = 0;

    delete pipeline_softmax_reduce_sum;
    pipeline_softmax_reduce_sum = 0;

    delete pipeline_softmax_div_sum;
    pipeline_softmax_div_sum = 0;

    delete pipeline_softmax_reduce_max_pack4;
    pipeline_softmax_reduce_max_pack4 = 0;

    delete pipeline_softmax_exp_sub_max_pack4;
    pipeline_softmax_exp_sub_max_pack4 = 0;

    delete pipeline_softmax_reduce_sum_pack4;
    pipeline_softmax_reduce_sum_pack4 = 0;

    delete pipeline_softmax_div_sum_pack4;
    pipeline_softmax_div_sum_pack4 = 0;

    return 0;
}

int Softmax_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    size_t elemsize = bottom_top_blob.elemsize;
    int elempack = bottom_top_blob.elempack;

    VkMat max_workspace;
    VkMat sum_workspace;

    if (dims == 1) // axis == 0
    {
        max_workspace.create(1, elemsize, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
        sum_workspace.create(1, elemsize, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
    }
    else if (dims == 2 && axis == 0)
    {
        max_workspace.create(w, elemsize, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
        sum_workspace.create(w, elemsize, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
    }
    else if (dims == 2 && axis == 1)
    {
        max_workspace.create(h, elemsize, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
        sum_workspace.create(h, elemsize, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
    }
    else if (dims == 3 && axis == 0)
    {
        max_workspace.create(w, h, elemsize, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
        sum_workspace.create(w, h, elemsize, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
    }
    else if (dims == 3 && axis == 1)
    {
        max_workspace.create(w, channels, elemsize, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
        sum_workspace.create(w, channels, elemsize, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
    }
    else if (dims == 3 && axis == 2)
    {
        max_workspace.create(h, channels, elemsize, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
        sum_workspace.create(h, channels, elemsize, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
    }

    // reduce max
    {
    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_top_blob;
    bindings[1] = max_workspace;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;
    constants[5].i = max_workspace.dims;
    constants[6].i = max_workspace.w;
    constants[7].i = max_workspace.h;
    constants[8].i = max_workspace.c;
    constants[9].i = max_workspace.cstep;

    const Pipeline* pipeline = elempack == 4 ? pipeline_softmax_reduce_max_pack4 : pipeline_softmax_reduce_max;

    cmd.record_pipeline(pipeline, bindings, constants, max_workspace);
    }

    // exp( v - max )
    {
    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_top_blob;
    bindings[1] = max_workspace;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;
    constants[5].i = max_workspace.dims;
    constants[6].i = max_workspace.w;
    constants[7].i = max_workspace.h;
    constants[8].i = max_workspace.c;
    constants[9].i = max_workspace.cstep;

    const Pipeline* pipeline = elempack == 4 ? pipeline_softmax_exp_sub_max_pack4 : pipeline_softmax_exp_sub_max;

    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    // reduce sum
    {
    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_top_blob;
    bindings[1] = sum_workspace;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;
    constants[5].i = sum_workspace.dims;
    constants[6].i = sum_workspace.w;
    constants[7].i = sum_workspace.h;
    constants[8].i = sum_workspace.c;
    constants[9].i = sum_workspace.cstep;

    const Pipeline* pipeline = elempack == 4 ? pipeline_softmax_reduce_sum_pack4 : pipeline_softmax_reduce_sum;

    cmd.record_pipeline(pipeline, bindings, constants, sum_workspace);
    }

    // div sum
    {
    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_top_blob;
    bindings[1] = sum_workspace;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;
    constants[5].i = sum_workspace.dims;
    constants[6].i = sum_workspace.w;
    constants[7].i = sum_workspace.h;
    constants[8].i = sum_workspace.c;
    constants[9].i = sum_workspace.cstep;

    const Pipeline* pipeline = elempack == 4 ? pipeline_softmax_div_sum_pack4 : pipeline_softmax_div_sum;

    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    return 0;
}

} // namespace ncnn

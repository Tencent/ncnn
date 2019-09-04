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

#include "pooling_vulkan.h"
#include <float.h>
#include <algorithm>
#include "layer_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Pooling_vulkan)

Pooling_vulkan::Pooling_vulkan()
{
    support_vulkan = true;

    padding = 0;
    pipeline_pooling = 0;
    pipeline_pooling_global = 0;
    pipeline_pooling_pack4 = 0;
    pipeline_pooling_global_pack4 = 0;
}

int Pooling_vulkan::create_pipeline(const Option& opt)
{
    {
        padding = ncnn::create_layer(ncnn::LayerType::Padding);
        padding->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, pad_top);
        pd.set(1, pad_bottom);
        pd.set(2, pad_left);
        pd.set(3, pad_right);
        pd.set(4, 0);

        if (pooling_type == PoolMethod_MAX)
        {
            pd.set(5, -FLT_MAX);
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            pd.set(5, 0.f);
        }

        padding->load_param(pd);

        padding->create_pipeline(opt);
    }

    std::vector<vk_specialization_type> specializations(8);
    specializations[0].i = pooling_type;
    specializations[1].i = kernel_w;
    specializations[2].i = kernel_h;
    specializations[3].i = stride_w;
    specializations[4].i = stride_h;
    specializations[5].i = global_pooling;
    specializations[6].i = pad_mode;
    specializations[7].i = avgpool_count_include_pad;

    // pack1
    {
        pipeline_pooling = new Pipeline(vkdev);
        pipeline_pooling->set_optimal_local_size_xyz();
        pipeline_pooling->create("pooling", opt, specializations, 2, 10);
    }

    // pack4
    {
        pipeline_pooling_pack4 = new Pipeline(vkdev);
        pipeline_pooling_pack4->set_optimal_local_size_xyz();
        pipeline_pooling_pack4->create("pooling_pack4", opt, specializations, 2, 10);
    }

    if (global_pooling)
    {
        std::vector<vk_specialization_type> specializations(1);
        specializations[0].i = pooling_type;

        // pack1
        {
            pipeline_pooling_global = new Pipeline(vkdev);
            pipeline_pooling_global->set_optimal_local_size_xyz(256, 1, 1);
            pipeline_pooling_global->create("pooling_global", opt, specializations, 2, 10);
        }

        // pack4
        {
            pipeline_pooling_global_pack4 = new Pipeline(vkdev);
            pipeline_pooling_global_pack4->set_optimal_local_size_xyz(256, 1, 1);
            pipeline_pooling_global_pack4->create("pooling_global_pack4", opt, specializations, 2, 10);
        }
    }

    return 0;
}

int Pooling_vulkan::destroy_pipeline(const Option& opt)
{
    if (padding)
    {
        padding->destroy_pipeline(opt);
        delete padding;
        padding = 0;
    }

    delete pipeline_pooling;
    pipeline_pooling = 0;

    delete pipeline_pooling_pack4;
    pipeline_pooling_pack4 = 0;

    delete pipeline_pooling_global;
    pipeline_pooling_global = 0;

    delete pipeline_pooling_global_pack4;
    pipeline_pooling_global_pack4 = 0;

    return 0;
}

int Pooling_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (global_pooling)
    {
        top_blob.create(channels, elemsize, elempack, opt.blob_vkallocator, opt.staging_vkallocator);
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

        const Pipeline* pipeline = elempack == 4 ? pipeline_pooling_global_pack4 : pipeline_pooling_global;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    VkMat bottom_blob_bordered = bottom_blob;
    {
        ncnn::Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->forward(bottom_blob, bottom_blob_bordered, cmd, opt_pad);

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    if (pad_mode == 0) // full padding
    {
        int wtail = (w + pad_left + pad_right - kernel_w) % stride_w;
        int htail = (h + pad_top + pad_bottom - kernel_h) % stride_h;

        if (wtail != 0)
            w += stride_w - wtail;
        if (htail != 0)
            h += stride_h - htail;
    }

    // FIXME full pad and valid pad only
    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;

    top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob_bordered;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob_bordered.dims;
    constants[1].i = bottom_blob_bordered.w;
    constants[2].i = bottom_blob_bordered.h;
    constants[3].i = bottom_blob_bordered.c;
    constants[4].i = bottom_blob_bordered.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;

    const Pipeline* pipeline = elempack == 4 ? pipeline_pooling_pack4 : pipeline_pooling;

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn

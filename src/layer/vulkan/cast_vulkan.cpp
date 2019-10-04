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

#include "cast_vulkan.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Cast_vulkan)

Cast_vulkan::Cast_vulkan()
{
    support_vulkan = true;

    pipeline_cast_fp32_to_fp16 = 0;
    pipeline_cast_fp32_to_fp16_pack4 = 0;
    pipeline_cast_fp16_to_fp32 = 0;
    pipeline_cast_fp16_to_fp32_pack4 = 0;
}

int Cast_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations;

    if (type_from == 1 && type_to == 2)
    {
        // pack1
        {
            pipeline_cast_fp32_to_fp16 = new Pipeline(vkdev);
            pipeline_cast_fp32_to_fp16->set_optimal_local_size_xyz();
            pipeline_cast_fp32_to_fp16->create("cast_fp32_to_fp16", opt, specializations, 2, 10);
        }

        // pack4
        {
            pipeline_cast_fp32_to_fp16_pack4 = new Pipeline(vkdev);
            pipeline_cast_fp32_to_fp16_pack4->set_optimal_local_size_xyz();
            pipeline_cast_fp32_to_fp16_pack4->create("cast_fp32_to_fp16_pack4", opt, specializations, 2, 10);
        }
    }

    if (type_from == 2 && type_to == 1)
    {
        // pack1
        {
            pipeline_cast_fp16_to_fp32 = new Pipeline(vkdev);
            pipeline_cast_fp16_to_fp32->set_optimal_local_size_xyz();
            pipeline_cast_fp16_to_fp32->create("cast_fp16_to_fp32", opt, specializations, 2, 10);
        }

        // pack4
        {
            pipeline_cast_fp16_to_fp32_pack4 = new Pipeline(vkdev);
            pipeline_cast_fp16_to_fp32_pack4->set_optimal_local_size_xyz();
            pipeline_cast_fp16_to_fp32_pack4->create("cast_fp16_to_fp32_pack4", opt, specializations, 2, 10);
        }
    }

    return 0;
}

int Cast_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_cast_fp32_to_fp16;
    pipeline_cast_fp32_to_fp16 = 0;

    delete pipeline_cast_fp32_to_fp16_pack4;
    pipeline_cast_fp32_to_fp16_pack4 = 0;

    delete pipeline_cast_fp16_to_fp32;
    pipeline_cast_fp16_to_fp32 = 0;

    delete pipeline_cast_fp16_to_fp32_pack4;
    pipeline_cast_fp16_to_fp32_pack4 = 0;

    return 0;
}

int Cast_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    if (type_from == type_to)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    size_t out_elemsize = elemsize;
    if (type_to == 1)
    {
        // float32
        out_elemsize = 4 * elempack;
    }
    else if (type_to == 2)
    {
        // float16
        out_elemsize = 2 * elempack;
    }
    else if (type_to == 3)
    {
        // int8
        out_elemsize = elempack;
    }

    if (dims == 1)
    {
        top_blob.create(w, out_elemsize, elempack, opt.blob_vkallocator, opt.staging_vkallocator);
    }
    else if (dims == 2)
    {
        top_blob.create(w, h, out_elemsize, elempack, opt.blob_vkallocator, opt.staging_vkallocator);
    }
    else if (dims == 3)
    {
        top_blob.create(w, h, channels, out_elemsize, elempack, opt.blob_vkallocator, opt.staging_vkallocator);
    }
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

    if (type_from == 1 && type_to == 2)
    {
        pipeline = elempack == 4 ? pipeline_cast_fp32_to_fp16_pack4 : pipeline_cast_fp32_to_fp16;
    }
    if (type_from == 2 && type_to == 1)
    {
        pipeline = elempack == 4 ? pipeline_cast_fp16_to_fp32_pack4 : pipeline_cast_fp16_to_fp32;
    }

    // TODO more cast type

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn

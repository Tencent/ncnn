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

#include "padding_vulkan.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Padding_vulkan)

Padding_vulkan::Padding_vulkan()
{
    support_vulkan = true;

    pipeline_padding = 0;
    pipeline_padding_pack4 = 0;
}

int Padding_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations(2);
    specializations[0].i = type;
    specializations[1].f = value;

    // pack1
    {
        pipeline_padding = new Pipeline(vkdev);
        pipeline_padding->set_optimal_local_size_xyz();
        pipeline_padding->create("padding", opt, specializations, 2, 12);
    }

    // pack4
    {
        pipeline_padding_pack4 = new Pipeline(vkdev);
        pipeline_padding_pack4->set_optimal_local_size_xyz();
        pipeline_padding_pack4->create("padding_pack4", opt, specializations, 2, 12);
    }

    return 0;
}

int Padding_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_padding;
    pipeline_padding = 0;

    delete pipeline_padding_pack4;
    pipeline_padding_pack4 = 0;

    return 0;
}

int Padding_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    if (top == 0 && bottom == 0 && left == 0 && right == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    // TODO vec and image padding

    int outw = w + left + right;
    int outh = h + top + bottom;

    top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(12);
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
    constants[10].i = left;
    constants[11].i = top;

    const Pipeline* pipeline = elempack == 4 ? pipeline_padding_pack4 : pipeline_padding;

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

int Padding_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    const VkMat& reference_blob = bottom_blobs[1];

    VkMat& top_blob = top_blobs[0];

    int _top;
    int _bottom;
    int _left;
    int _right;
    {
        const int* param_data = reference_blob.mapped();

        _top = param_data[0];
        _bottom = param_data[1];
        _left = param_data[2];
        _right = param_data[3];
    }

    if (_top == 0 && _bottom == 0 && _left == 0 && _right == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    // TODO vec and image padding

    int outw = w + _left + _right;
    int outh = h + _top + _bottom;

    top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(12);
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
    constants[10].i = _left;
    constants[11].i = _top;

    const Pipeline* pipeline = elempack == 4 ? pipeline_padding_pack4 : pipeline_padding;

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn

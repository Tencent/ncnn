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

#include "reshape_vulkan.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Reshape_vulkan)

Reshape_vulkan::Reshape_vulkan()
{
    support_vulkan = true;

    pipeline_reshape = 0;
    pipeline_reshape_pack4 = 0;
    pipeline_reshape_pack1to4 = 0;
    pipeline_reshape_pack4to1 = 0;
}

int Reshape_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations(1);
    specializations[0].i = ndim;

    // pack1
    {
        pipeline_reshape = new Pipeline(vkdev);
        pipeline_reshape->set_optimal_local_size_xyz();
        pipeline_reshape->create("reshape", opt, specializations, 2, 10);
    }

    // pack4
    {
        pipeline_reshape_pack4 = new Pipeline(vkdev);
        pipeline_reshape_pack4->set_optimal_local_size_xyz();
        pipeline_reshape_pack4->create("reshape_pack4", opt, specializations, 2, 10);
    }

    // pack1to4
    {
        pipeline_reshape_pack1to4 = new Pipeline(vkdev);
        pipeline_reshape_pack1to4->set_optimal_local_size_xyz();
        pipeline_reshape_pack1to4->create("reshape_pack1to4", opt, specializations, 2, 10);
    }

    // pack4to1
    {
        pipeline_reshape_pack4to1 = new Pipeline(vkdev);
        pipeline_reshape_pack4to1->set_optimal_local_size_xyz();
        pipeline_reshape_pack4to1->create("reshape_pack4to1", opt, specializations, 2, 10);
    }

    return 0;
}

int Reshape_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_reshape;
    pipeline_reshape = 0;

    delete pipeline_reshape_pack4;
    pipeline_reshape_pack4 = 0;

    delete pipeline_reshape_pack1to4;
    pipeline_reshape_pack1to4 = 0;

    delete pipeline_reshape_pack4to1;
    pipeline_reshape_pack4to1 = 0;

    return 0;
}

int Reshape_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int out_elempack;

    int total = bottom_blob.w * bottom_blob.h * bottom_blob.c * elempack;

    if (ndim == 1)
    {
        int _w = w;

        if (_w == 0)
            _w = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;

        if (_w == -1)
            _w = total;

        // TODO permute support

        out_elempack = _w % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 4) out_elemsize = 4*2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        if (dims == 1 && bottom_blob.w == _w && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(_w / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
    }
    else if (ndim == 2)
    {
        int _w = w;
        int _h = h;

        if (_w == 0)
            _w = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (_h == 0)
            _h = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;

        if (_w == -1)
            _w = total / _h;
        if (_h == -1)
            _h = total / _w;

        out_elempack = _h % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 4) out_elemsize = 4*2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        if (dims == 2 && bottom_blob.h == _h && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(_w, _h / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
    }
    else // if (ndim == 3)
    {
        int _w = w;
        int _h = h;
        int _c = c;

        if (_w == 0)
            _w = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (_h == 0)
            _h = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (_c == 0)
            _c = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;

        if (_w == -1)
            _w = total / _c / _h;
        if (_h == -1)
            _h = total / _c / _w;
        if (_c == -1)
            _c = total / _h / _w;

        out_elempack = _c % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 4) out_elemsize = 4*2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        if (dims == 3 && bottom_blob.c == _c && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            top_blob.w = _w;
            top_blob.h = _h;
            return 0;
        }

        top_blob.create(_w, _h, _c / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
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
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_reshape;
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        pipeline = pipeline_reshape_pack4;
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_reshape_pack1to4;
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        pipeline = pipeline_reshape_pack4to1;
    }

    if (elempack == 4 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline, bindings, constants, bottom_blob);
    }
    else
    {
        cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    }

    return 0;
}

} // namespace ncnn

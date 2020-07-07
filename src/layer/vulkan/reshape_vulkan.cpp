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

#include "layer_type.h"
#include "layer_shader_type.h"

#include <algorithm>

namespace ncnn {

Reshape_vulkan::Reshape_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    permute_hwc = 0;

    pipeline_reshape = 0;
    pipeline_reshape_pack4 = 0;
    pipeline_reshape_pack1to4 = 0;
    pipeline_reshape_pack4to1 = 0;
    pipeline_reshape_pack8 = 0;
    pipeline_reshape_pack1to8 = 0;
    pipeline_reshape_pack4to8 = 0;
    pipeline_reshape_pack8to4 = 0;
    pipeline_reshape_pack8to1 = 0;
}

int Reshape_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    Mat shape_permuted = shape;
    if (shape.dims == 3 && ndim == 1 && permute == 1)
    {
        shape_permuted = Mat(shape.c, shape.w, shape.h, (void*)0);
    }

    int elempack = 1;
    if (shape_permuted.dims == 1) elempack = opt.use_shader_pack8 && shape_permuted.w % 8 == 0 ? 8 : shape_permuted.w % 4 == 0 ? 4 : 1;
    if (shape_permuted.dims == 2) elempack = opt.use_shader_pack8 && shape_permuted.h % 8 == 0 ? 8 : shape_permuted.h % 4 == 0 ? 4 : 1;
    if (shape_permuted.dims == 3) elempack = opt.use_shader_pack8 && shape_permuted.c % 8 == 0 ? 8 : shape_permuted.c % 4 == 0 ? 4 : 1;

    int out_elempack = 1;
    if (out_shape.dims == 1) out_elempack = opt.use_shader_pack8 && out_shape.w % 8 == 0 ? 8 : out_shape.w % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 2) out_elempack = opt.use_shader_pack8 && out_shape.h % 8 == 0 ? 8 : out_shape.h % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 3) out_elempack = opt.use_shader_pack8 && out_shape.c % 8 == 0 ? 8 : out_shape.c % 4 == 0 ? 4 : 1;

    size_t elemsize;
    size_t out_elemsize;
    if (opt.use_fp16_storage)
    {
        elemsize = elempack * 2u;
        out_elemsize = out_elempack * 2u;
    }
    else if (opt.use_fp16_packed)
    {
        elemsize = elempack == 1 ? 4u : elempack * 2u;
        out_elemsize = out_elempack == 1 ? 4u : out_elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
        out_elemsize = out_elempack * 4u;
    }

    Mat shape_packed;
    if (shape_permuted.dims == 1) shape_packed = Mat(shape_permuted.w / elempack, (void*)0, elemsize, elempack);
    if (shape_permuted.dims == 2) shape_packed = Mat(shape_permuted.w, shape_permuted.h / elempack, (void*)0, elemsize, elempack);
    if (shape_permuted.dims == 3) shape_packed = Mat(shape_permuted.w, shape_permuted.h, shape_permuted.c / elempack, (void*)0, elemsize, elempack);

    Mat out_shape_packed;
    if (out_shape.dims == 1) out_shape_packed = Mat(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 2) out_shape_packed = Mat(out_shape.w, out_shape.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    // check blob shape
    if (!vkdev->shape_support_image_storage(shape_packed) || !vkdev->shape_support_image_storage(out_shape_packed))
    {
        support_image_storage = false;
        opt.use_image_storage = false;
    }

    if (ndim == 1 && permute == 1)
    {
        permute_hwc = ncnn::create_layer(ncnn::LayerType::Permute);
        permute_hwc->vkdev = vkdev;

        permute_hwc->bottom_shapes.resize(1);
        permute_hwc->bottom_shapes[0] = shape;
        permute_hwc->top_shapes.resize(1);
        permute_hwc->top_shapes[0] = shape_permuted;

        ncnn::ParamDict pd;
        pd.set(0, 3); // chw -> hwc

        permute_hwc->load_param(pd);

        permute_hwc->create_pipeline(opt);
    }

    std::vector<vk_specialization_type> specializations(1 + 10);
    specializations[0].i = ndim;
    specializations[1 + 0].i = shape_packed.dims;
    specializations[1 + 1].i = shape_packed.w;
    specializations[1 + 2].i = shape_packed.h;
    specializations[1 + 3].i = shape_packed.c;
    specializations[1 + 4].i = shape_packed.cstep;
    specializations[1 + 5].i = out_shape_packed.dims;
    specializations[1 + 6].i = out_shape_packed.w;
    specializations[1 + 7].i = out_shape_packed.h;
    specializations[1 + 8].i = out_shape_packed.c;
    specializations[1 + 9].i = out_shape_packed.cstep;

    Mat local_size_xyz_bottom; // pack4to1 and pack8to1
    if (shape_packed.dims == 1)
    {
        local_size_xyz_bottom.w = std::min(64, shape_packed.w);
        local_size_xyz_bottom.h = 1;
        local_size_xyz_bottom.c = 1;
    }
    if (shape_packed.dims == 2)
    {
        local_size_xyz_bottom.w = std::min(8, shape_packed.w);
        local_size_xyz_bottom.h = std::min(8, shape_packed.h);
        local_size_xyz_bottom.c = 1;
    }
    if (shape_packed.dims == 3)
    {
        local_size_xyz_bottom.w = std::min(4, shape_packed.w);
        local_size_xyz_bottom.h = std::min(4, shape_packed.h);
        local_size_xyz_bottom.c = std::min(4, shape_packed.c);
    }

    Mat local_size_xyz;
    if (out_shape_packed.dims == 1)
    {
        local_size_xyz.w = std::min(64, out_shape_packed.w);
        local_size_xyz.h = 1;
        local_size_xyz.c = 1;
    }
    if (out_shape_packed.dims == 2)
    {
        local_size_xyz.w = std::min(8, out_shape_packed.w);
        local_size_xyz.h = std::min(8, out_shape_packed.h);
        local_size_xyz.c = 1;
    }
    if (out_shape_packed.dims == 3)
    {
        local_size_xyz.w = std::min(4, out_shape_packed.w);
        local_size_xyz.h = std::min(4, out_shape_packed.h);
        local_size_xyz.c = std::min(4, out_shape_packed.c);
    }

    // pack1
    if (shape_permuted.dims == 0 || (elempack == 1 && out_elempack == 1))
    {
        pipeline_reshape = new Pipeline(vkdev);
        pipeline_reshape->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reshape->create(LayerShaderType::reshape, opt, specializations);
    }

    // pack4
    if (shape_permuted.dims == 0 || (elempack == 4 && out_elempack == 4))
    {
        pipeline_reshape_pack4 = new Pipeline(vkdev);
        pipeline_reshape_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reshape_pack4->create(LayerShaderType::reshape_pack4, opt, specializations);
    }

    // pack1to4
    if (shape_permuted.dims == 0 || (elempack == 1 && out_elempack == 4))
    {
        pipeline_reshape_pack1to4 = new Pipeline(vkdev);
        pipeline_reshape_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reshape_pack1to4->create(LayerShaderType::reshape_pack1to4, opt, specializations);
    }

    // pack4to1
    if (shape_permuted.dims == 0 || (elempack == 4 && out_elempack == 1))
    {
        pipeline_reshape_pack4to1 = new Pipeline(vkdev);
        pipeline_reshape_pack4to1->set_optimal_local_size_xyz(local_size_xyz_bottom);
        pipeline_reshape_pack4to1->create(LayerShaderType::reshape_pack4to1, opt, specializations);
    }

    // pack8
    if ((opt.use_shader_pack8 && shape_permuted.dims == 0) || (elempack == 8 && out_elempack == 8))
    {
        pipeline_reshape_pack8 = new Pipeline(vkdev);
        pipeline_reshape_pack8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reshape_pack8->create(LayerShaderType::reshape_pack8, opt, specializations);
    }

    // pack1to8
    if ((opt.use_shader_pack8 && shape_permuted.dims == 0) || (elempack == 1 && out_elempack == 8))
    {
        pipeline_reshape_pack1to8 = new Pipeline(vkdev);
        pipeline_reshape_pack1to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reshape_pack1to8->create(LayerShaderType::reshape_pack1to8, opt, specializations);
    }

    // pack4to8
    if ((opt.use_shader_pack8 && shape_permuted.dims == 0) || (elempack == 4 && out_elempack == 8))
    {
        pipeline_reshape_pack4to8 = new Pipeline(vkdev);
        pipeline_reshape_pack4to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reshape_pack4to8->create(LayerShaderType::reshape_pack4to8, opt, specializations);
    }

    // pack8to4
    if ((opt.use_shader_pack8 && shape_permuted.dims == 0) || (elempack == 8 && out_elempack == 4))
    {
        pipeline_reshape_pack8to4 = new Pipeline(vkdev);
        pipeline_reshape_pack8to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reshape_pack8to4->create(LayerShaderType::reshape_pack8to4, opt, specializations);
    }

    // pack8to1
    if ((opt.use_shader_pack8 && shape_permuted.dims == 0) || (elempack == 8 && out_elempack == 1))
    {
        pipeline_reshape_pack8to1 = new Pipeline(vkdev);
        pipeline_reshape_pack8to1->set_optimal_local_size_xyz(local_size_xyz_bottom);
        pipeline_reshape_pack8to1->create(LayerShaderType::reshape_pack8to1, opt, specializations);
    }

    return 0;
}

int Reshape_vulkan::destroy_pipeline(const Option& opt)
{
    if (permute_hwc)
    {
        permute_hwc->destroy_pipeline(opt);
        delete permute_hwc;
        permute_hwc = 0;
    }

    delete pipeline_reshape;
    pipeline_reshape = 0;

    delete pipeline_reshape_pack4;
    pipeline_reshape_pack4 = 0;

    delete pipeline_reshape_pack1to4;
    pipeline_reshape_pack1to4 = 0;

    delete pipeline_reshape_pack4to1;
    pipeline_reshape_pack4to1 = 0;

    delete pipeline_reshape_pack8;
    pipeline_reshape_pack8 = 0;

    delete pipeline_reshape_pack1to8;
    pipeline_reshape_pack1to8 = 0;

    delete pipeline_reshape_pack4to8;
    pipeline_reshape_pack4to8 = 0;

    delete pipeline_reshape_pack8to4;
    pipeline_reshape_pack8to4 = 0;

    delete pipeline_reshape_pack8to1;
    pipeline_reshape_pack8to1 = 0;

    return 0;
}

int Reshape_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    // permute
    VkMat bottom_blob_permuted = bottom_blob;

    if (bottom_blob.dims == 3 && ndim == 1 && permute == 1)
    {
        Option opt_permute = opt;
        opt_permute.blob_vkallocator = opt.workspace_vkallocator;

        permute_hwc->forward(bottom_blob, bottom_blob_permuted, cmd, opt_permute);
    }

    int dims = bottom_blob_permuted.dims;
    size_t elemsize = bottom_blob_permuted.elemsize;
    int elempack = bottom_blob_permuted.elempack;
    int out_elempack;

    int total = bottom_blob_permuted.w * bottom_blob_permuted.h * bottom_blob_permuted.c * elempack;

    if (ndim == 1)
    {
        int _w = w;

        if (_w == 0)
            _w = dims == 1 ? bottom_blob_permuted.w * elempack : bottom_blob_permuted.w;

        if (_w == -1)
            _w = total;

        // TODO permute support

        out_elempack = opt.use_shader_pack8 && _w % 8 == 0 ? 8 : _w % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8 * 2u;
            if (out_elempack == 4) out_elemsize = 4 * 2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        if (dims == 1 && bottom_blob_permuted.w == _w && elempack == out_elempack)
        {
            top_blob = bottom_blob_permuted;
            return 0;
        }

        top_blob.create(_w / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else if (ndim == 2)
    {
        int _w = w;
        int _h = h;

        if (_w == 0)
            _w = dims == 1 ? bottom_blob_permuted.w * elempack : bottom_blob_permuted.w;
        if (_h == 0)
            _h = dims == 2 ? bottom_blob_permuted.h * elempack : bottom_blob_permuted.h;

        if (_w == -1)
            _w = total / _h;
        if (_h == -1)
            _h = total / _w;

        out_elempack = opt.use_shader_pack8 && _h % 8 == 0 ? 8 : _h % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8 * 2u;
            if (out_elempack == 4) out_elemsize = 4 * 2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        if (dims == 2 && bottom_blob_permuted.h == _h && elempack == out_elempack)
        {
            top_blob = bottom_blob_permuted;
            return 0;
        }

        top_blob.create(_w, _h / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else // if (ndim == 3)
    {
        int _w = w;
        int _h = h;
        int _c = c;

        if (_w == 0)
            _w = dims == 1 ? bottom_blob_permuted.w * elempack : bottom_blob_permuted.w;
        if (_h == 0)
            _h = dims == 2 ? bottom_blob_permuted.h * elempack : bottom_blob_permuted.h;
        if (_c == 0)
            _c = dims == 3 ? bottom_blob_permuted.c * elempack : bottom_blob_permuted.c;

        if (_w == -1)
            _w = total / _c / _h;
        if (_h == -1)
            _h = total / _c / _w;
        if (_c == -1)
            _c = total / _h / _w;

        out_elempack = opt.use_shader_pack8 && _c % 8 == 0 ? 8 : _c % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8 * 2u;
            if (out_elempack == 4) out_elemsize = 4 * 2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        if (dims == 3 && bottom_blob_permuted.c == _c && elempack == out_elempack)
        {
            top_blob = bottom_blob_permuted;
            top_blob.w = _w;
            top_blob.h = _h;
            return 0;
        }

        top_blob.create(_w, _h, _c / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }

    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob_permuted;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob_permuted.dims;
    constants[1].i = bottom_blob_permuted.w;
    constants[2].i = bottom_blob_permuted.h;
    constants[3].i = bottom_blob_permuted.c;
    constants[4].i = bottom_blob_permuted.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;

    if (elempack == 1 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_reshape, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_reshape_pack4, bindings, constants, top_blob);
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_reshape_pack1to4, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_reshape_pack4to1, bindings, constants, bottom_blob_permuted);
    }
    else if (elempack == 8 && out_elempack == 8)
    {
        cmd.record_pipeline(pipeline_reshape_pack8, bindings, constants, top_blob);
    }
    else if (elempack == 1 && out_elempack == 8)
    {
        cmd.record_pipeline(pipeline_reshape_pack1to8, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 8)
    {
        cmd.record_pipeline(pipeline_reshape_pack4to8, bindings, constants, top_blob);
    }
    else if (elempack == 8 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_reshape_pack8to4, bindings, constants, top_blob);
    }
    else if (elempack == 8 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_reshape_pack8to1, bindings, constants, bottom_blob_permuted);
    }

    return 0;
}

int Reshape_vulkan::forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    // permute
    VkImageMat bottom_blob_permuted = bottom_blob;

    if (bottom_blob.dims == 3 && ndim == 1 && permute == 1)
    {
        Option opt_permute = opt;
        opt_permute.blob_vkallocator = opt.workspace_vkallocator;

        permute_hwc->forward(bottom_blob, bottom_blob_permuted, cmd, opt_permute);
    }

    int dims = bottom_blob_permuted.dims;
    size_t elemsize = bottom_blob_permuted.elemsize;
    int elempack = bottom_blob_permuted.elempack;
    int out_elempack;

    int total = bottom_blob_permuted.w * bottom_blob_permuted.h * bottom_blob_permuted.c * elempack;

    if (ndim == 1)
    {
        int _w = w;

        if (_w == 0)
            _w = dims == 1 ? bottom_blob_permuted.w * elempack : bottom_blob_permuted.w;

        if (_w == -1)
            _w = total;

        // TODO permute support

        out_elempack = opt.use_shader_pack8 && _w % 8 == 0 ? 8 : _w % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8 * 2u;
            if (out_elempack == 4) out_elemsize = 4 * 2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        if (dims == 1 && bottom_blob_permuted.w == _w && elempack == out_elempack)
        {
            top_blob = bottom_blob_permuted;
            return 0;
        }

        top_blob.create(_w / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else if (ndim == 2)
    {
        int _w = w;
        int _h = h;

        if (_w == 0)
            _w = dims == 1 ? bottom_blob_permuted.w * elempack : bottom_blob_permuted.w;
        if (_h == 0)
            _h = dims == 2 ? bottom_blob_permuted.h * elempack : bottom_blob_permuted.h;

        if (_w == -1)
            _w = total / _h;
        if (_h == -1)
            _h = total / _w;

        out_elempack = opt.use_shader_pack8 && _h % 8 == 0 ? 8 : _h % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8 * 2u;
            if (out_elempack == 4) out_elemsize = 4 * 2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        if (dims == 2 && bottom_blob_permuted.w == _w && bottom_blob_permuted.h == _h && elempack == out_elempack)
        {
            top_blob = bottom_blob_permuted;
            return 0;
        }

        top_blob.create(_w, _h / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else // if (ndim == 3)
    {
        int _w = w;
        int _h = h;
        int _c = c;

        if (_w == 0)
            _w = dims == 1 ? bottom_blob_permuted.w * elempack : bottom_blob_permuted.w;
        if (_h == 0)
            _h = dims == 2 ? bottom_blob_permuted.h * elempack : bottom_blob_permuted.h;
        if (_c == 0)
            _c = dims == 3 ? bottom_blob_permuted.c * elempack : bottom_blob_permuted.c;

        if (_w == -1)
            _w = total / _c / _h;
        if (_h == -1)
            _h = total / _c / _w;
        if (_c == -1)
            _c = total / _h / _w;

        out_elempack = opt.use_shader_pack8 && _c % 8 == 0 ? 8 : _c % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8 * 2u;
            if (out_elempack == 4) out_elemsize = 4 * 2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        if (dims == 3 && bottom_blob_permuted.w == _w && bottom_blob_permuted.h == _h && bottom_blob_permuted.c == _c && elempack == out_elempack)
        {
            top_blob = bottom_blob_permuted;
            top_blob.w = _w;
            top_blob.h = _h;
            return 0;
        }

        top_blob.create(_w, _h, _c / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }

    if (top_blob.empty())
        return -100;

    std::vector<VkImageMat> bindings(2);
    bindings[0] = bottom_blob_permuted;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob_permuted.dims;
    constants[1].i = bottom_blob_permuted.w;
    constants[2].i = bottom_blob_permuted.h;
    constants[3].i = bottom_blob_permuted.c;
    constants[4].i = 0; //bottom_blob_permuted.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = 0; //top_blob.cstep;

    if (elempack == 1 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_reshape, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_reshape_pack4, bindings, constants, top_blob);
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_reshape_pack1to4, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_reshape_pack4to1, bindings, constants, bottom_blob_permuted);
    }
    else if (elempack == 8 && out_elempack == 8)
    {
        cmd.record_pipeline(pipeline_reshape_pack8, bindings, constants, top_blob);
    }
    else if (elempack == 1 && out_elempack == 8)
    {
        cmd.record_pipeline(pipeline_reshape_pack1to8, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 8)
    {
        cmd.record_pipeline(pipeline_reshape_pack4to8, bindings, constants, top_blob);
    }
    else if (elempack == 8 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_reshape_pack8to4, bindings, constants, top_blob);
    }
    else if (elempack == 8 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_reshape_pack8to1, bindings, constants, bottom_blob_permuted);
    }

    return 0;
}

} // namespace ncnn

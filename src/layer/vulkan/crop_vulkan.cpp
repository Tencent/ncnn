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

#include "crop_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

Crop_vulkan::Crop_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    pipeline_crop = 0;
    pipeline_crop_pack4 = 0;
    pipeline_crop_pack1to4 = 0;
    pipeline_crop_pack4to1 = 0;
    pipeline_crop_pack8 = 0;
    pipeline_crop_pack1to8 = 0;
    pipeline_crop_pack4to8 = 0;
    pipeline_crop_pack8to4 = 0;
    pipeline_crop_pack8to1 = 0;
}

int Crop_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = 1;
    if (shape.dims == 1) elempack = opt.use_shader_pack8 && shape.w % 8 == 0 ? 8 : shape.w % 4 == 0 ? 4 : 1;
    if (shape.dims == 2) elempack = opt.use_shader_pack8 && shape.h % 8 == 0 ? 8 : shape.h % 4 == 0 ? 4 : 1;
    if (shape.dims == 3 || shape.dims == 4) elempack = opt.use_shader_pack8 && shape.c % 8 == 0 ? 8 : shape.c % 4 == 0 ? 4 : 1;

    int out_elempack = 1;
    if (out_shape.dims == 1) out_elempack = opt.use_shader_pack8 && out_shape.w % 8 == 0 ? 8 : out_shape.w % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 2) out_elempack = opt.use_shader_pack8 && out_shape.h % 8 == 0 ? 8 : out_shape.h % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 3 || out_shape.dims == 4) out_elempack = opt.use_shader_pack8 && out_shape.c % 8 == 0 ? 8 : out_shape.c % 4 == 0 ? 4 : 1;

    int offset_elempack = 1;
    bool numpy_style_slice = !starts.empty() && !ends.empty();
    if (numpy_style_slice)
    {
        offset_elempack = elempack;

        const int* starts_ptr = starts;
        const int* axes_ptr = axes;

        int _axes[4] = {0, 1, 2, 3};
        int num_axis = axes.w;
        if (num_axis == 0)
        {
            num_axis = shape.dims;
        }
        else
        {
            for (int i = 0; i < num_axis; i++)
            {
                int axis = axes_ptr[i];
                if (axis < 0)
                    axis = shape.dims + axis;
                _axes[i] = axis;
            }
        }

        for (int i = 0; i < num_axis; i++)
        {
            int start = starts_ptr[i];
            int axis = _axes[i];

            if (shape.dims == 1 && axis == 0)
            {
                int _woffset = start >= 0 ? start : shape.w + start;
                offset_elempack = opt.use_shader_pack8 && _woffset % 8 == 0 ? 8 : _woffset % 4 == 0 ? 4 : 1;
            }
            if (shape.dims == 2 && axis == 0)
            {
                int _hoffset = start >= 0 ? start : shape.h + start;
                offset_elempack = opt.use_shader_pack8 && _hoffset % 8 == 0 ? 8 : _hoffset % 4 == 0 ? 4 : 1;
            }
            if ((shape.dims == 3 || shape.dims == 4) && axis == 0)
            {
                int _coffset = start >= 0 ? start : shape.c + start;
                offset_elempack = opt.use_shader_pack8 && _coffset % 8 == 0 ? 8 : _coffset % 4 == 0 ? 4 : 1;
            }
        }
    }
    else
    {
        if (shape.dims == 1)
        {
            if (woffset == 0)
                offset_elempack = elempack;
            else
                offset_elempack = opt.use_shader_pack8 && woffset % 8 == 0 ? 8 : woffset % 4 == 0 ? 4 : 1;
        }
        else if (shape.dims == 2)
        {
            if (hoffset == 0)
                offset_elempack = elempack;
            else
                offset_elempack = opt.use_shader_pack8 && hoffset % 8 == 0 ? 8 : hoffset % 4 == 0 ? 4 : 1;
        }
        else // if (shape.dims == 3 || shape.dims == 4)
        {
            if (coffset == 0)
                offset_elempack = elempack;
            else
                offset_elempack = opt.use_shader_pack8 && coffset % 8 == 0 ? 8 : coffset % 4 == 0 ? 4 : 1;
        }
    }

    offset_elempack = std::min(offset_elempack, elempack);

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
    if (shape.dims == 1) shape_packed = Mat(shape.w / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 2) shape_packed = Mat(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 4) shape_packed = Mat(shape.w, shape.h, shape.d, shape.c / elempack, (void*)0, elemsize, elempack);

    Mat out_shape_packed;
    if (out_shape.dims == 1) out_shape_packed = Mat(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 2) out_shape_packed = Mat(out_shape.w, out_shape.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 4) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.d, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    Mat shape_unpacked = shape_packed;
    if (one_blob_only && shape.dims != 0 && elempack == out_elempack && elempack > offset_elempack)
    {
        size_t offset_elemsize;
        if (opt.use_fp16_storage)
        {
            offset_elemsize = offset_elempack * 2u;
        }
        else if (opt.use_fp16_packed)
        {
            offset_elemsize = offset_elempack == 1 ? 4u : offset_elempack * 2u;
        }
        else
        {
            offset_elemsize = offset_elempack * 4u;
        }

        if (shape.dims == 1) shape_unpacked = Mat(shape.w / offset_elempack, (void*)0, offset_elemsize, offset_elempack);
        if (shape.dims == 2) shape_unpacked = Mat(shape.w, shape.h / offset_elempack, (void*)0, offset_elemsize, offset_elempack);
        if (shape.dims == 3) shape_unpacked = Mat(shape.w, shape.h, shape.c / offset_elempack, (void*)0, offset_elemsize, offset_elempack);
        if (shape.dims == 4) shape_unpacked = Mat(shape.w, shape.h, shape.d, shape.c / offset_elempack, (void*)0, offset_elemsize, offset_elempack);
    }

    std::vector<vk_specialization_type> specializations(1 + 12);
    specializations[0].i = vkdev->info.bug_implicit_fp16_arithmetic();
    specializations[1 + 0].i = shape_unpacked.dims;
    specializations[1 + 1].i = shape_unpacked.w;
    specializations[1 + 2].i = shape_unpacked.h;
    specializations[1 + 3].i = shape_unpacked.d;
    specializations[1 + 4].i = shape_unpacked.c;
    specializations[1 + 5].i = shape_unpacked.cstep;
    specializations[1 + 6].i = out_shape_packed.dims;
    specializations[1 + 7].i = out_shape_packed.w;
    specializations[1 + 8].i = out_shape_packed.h;
    specializations[1 + 9].i = out_shape_packed.d;
    specializations[1 + 10].i = out_shape_packed.c;
    specializations[1 + 11].i = out_shape_packed.cstep;

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
    if (out_shape_packed.dims == 4)
    {
        local_size_xyz.w = std::min(4, out_shape_packed.w);
        local_size_xyz.h = std::min(4, out_shape_packed.h * out_shape_packed.d);
        local_size_xyz.c = std::min(4, out_shape_packed.c);
    }

    // pack1
    if (out_shape.dims == 0 || out_elempack == 1)
    {
        pipeline_crop = new Pipeline(vkdev);
        pipeline_crop->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_crop->create(LayerShaderType::crop, opt, specializations);
    }

    // pack4
    if (out_shape.dims == 0 || out_elempack == 4)
    {
        pipeline_crop_pack4 = new Pipeline(vkdev);
        pipeline_crop_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_crop_pack4->create(LayerShaderType::crop_pack4, opt, specializations);
    }

    // pack1to4
    if (out_shape.dims == 0 || out_elempack == 4)
    {
        pipeline_crop_pack1to4 = new Pipeline(vkdev);
        pipeline_crop_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_crop_pack1to4->create(LayerShaderType::crop_pack1to4, opt, specializations);
    }

    // pack4to1
    if (out_shape.dims == 0 || out_elempack == 1)
    {
        pipeline_crop_pack4to1 = new Pipeline(vkdev);
        pipeline_crop_pack4to1->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_crop_pack4to1->create(LayerShaderType::crop_pack4to1, opt, specializations);
    }

    // pack8
    if ((opt.use_shader_pack8 && out_shape.dims == 0) || (elempack == 8 && out_elempack == 8))
    {
        pipeline_crop_pack8 = new Pipeline(vkdev);
        pipeline_crop_pack8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_crop_pack8->create(LayerShaderType::crop_pack8, opt, specializations);
    }

    // pack1to8
    if ((opt.use_shader_pack8 && out_shape.dims == 0) || out_elempack == 8)
    {
        pipeline_crop_pack1to8 = new Pipeline(vkdev);
        pipeline_crop_pack1to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_crop_pack1to8->create(LayerShaderType::crop_pack1to8, opt, specializations);
    }

    // pack4to8
    if ((opt.use_shader_pack8 && out_shape.dims == 0) || out_elempack == 8)
    {
        pipeline_crop_pack4to8 = new Pipeline(vkdev);
        pipeline_crop_pack4to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_crop_pack4to8->create(LayerShaderType::crop_pack4to8, opt, specializations);
    }

    // pack8to4
    if ((opt.use_shader_pack8 && out_shape.dims == 0) || (elempack == 8 && out_elempack == 4))
    {
        pipeline_crop_pack8to4 = new Pipeline(vkdev);
        pipeline_crop_pack8to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_crop_pack8to4->create(LayerShaderType::crop_pack8to4, opt, specializations);
    }

    // pack8to1
    if ((opt.use_shader_pack8 && out_shape.dims == 0) || (elempack == 8 && out_elempack == 1))
    {
        pipeline_crop_pack8to1 = new Pipeline(vkdev);
        pipeline_crop_pack8to1->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_crop_pack8to1->create(LayerShaderType::crop_pack8to1, opt, specializations);
    }

    return 0;
}

int Crop_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_crop;
    pipeline_crop = 0;

    delete pipeline_crop_pack4;
    pipeline_crop_pack4 = 0;

    delete pipeline_crop_pack1to4;
    pipeline_crop_pack1to4 = 0;

    delete pipeline_crop_pack4to1;
    pipeline_crop_pack4to1 = 0;

    delete pipeline_crop_pack8;
    pipeline_crop_pack8 = 0;

    delete pipeline_crop_pack1to8;
    pipeline_crop_pack1to8 = 0;

    delete pipeline_crop_pack4to8;
    pipeline_crop_pack4to8 = 0;

    delete pipeline_crop_pack8to4;
    pipeline_crop_pack8to4 = 0;

    delete pipeline_crop_pack8to1;
    pipeline_crop_pack8to1 = 0;

    return 0;
}

int Crop_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int _woffset, _hoffset, _doffset, _coffset;
    int _outw, _outh, _outd, _outc;
    resolve_crop_roi(bottom_blob.shape(), _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);

    int offset_elempack;
    int out_elempack;

    if (dims == 1)
    {
        if (_woffset == 0 && _outw == bottom_blob.w * elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        offset_elempack = _woffset == 0 ? elempack : opt.use_shader_pack8 && _woffset % 8 == 0 ? 8 : _woffset % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_shader_pack8 && _outw % 8 == 0 ? 8 : _outw % 4 == 0 ? 4 : 1;
    }
    else if (dims == 2)
    {
        if (_woffset == 0 && _hoffset == 0 && _outw == bottom_blob.w && _outh == bottom_blob.h * elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        offset_elempack = _hoffset == 0 ? elempack : opt.use_shader_pack8 && _hoffset % 8 == 0 ? 8 : _hoffset % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_shader_pack8 && _outh % 8 == 0 ? 8 : _outh % 4 == 0 ? 4 : 1;
    }
    else if (dims == 3)
    {
        if (_woffset == 0 && _hoffset == 0 && _coffset == 0 && _outw == bottom_blob.w && _outh == bottom_blob.h && _outc == bottom_blob.c * elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        offset_elempack = _coffset == 0 ? elempack : opt.use_shader_pack8 && _coffset % 8 == 0 ? 8 : _coffset % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_shader_pack8 && _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
    }
    else // if (dims == 4)
    {
        if (_woffset == 0 && _hoffset == 0 && _doffset == 0 && _coffset == 0 && _outw == bottom_blob.w && _outh == bottom_blob.h && _outd == bottom_blob.d && _outc == bottom_blob.c * elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        offset_elempack = _coffset == 0 ? elempack : opt.use_shader_pack8 && _coffset % 8 == 0 ? 8 : _coffset % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_shader_pack8 && _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
    }

    offset_elempack = std::min(offset_elempack, elempack);

    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    // unpacking
    VkMat bottom_blob_unpacked = bottom_blob;
    if (elempack == out_elempack && elempack > offset_elempack)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

        vkdev->convert_packing(bottom_blob, bottom_blob_unpacked, offset_elempack, cmd, opt_pack1);
    }

    if (dims == 1)
    {
        top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else if (dims == 2)
    {
        top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else if (dims == 3)
    {
        top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else // if (dims == 4)
    {
        top_blob.create(_outw, _outh, _outd, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob_unpacked;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(16);
    constants[0].i = bottom_blob_unpacked.dims;
    constants[1].i = bottom_blob_unpacked.w;
    constants[2].i = bottom_blob_unpacked.h;
    constants[3].i = bottom_blob_unpacked.d;
    constants[4].i = bottom_blob_unpacked.c;
    constants[5].i = bottom_blob_unpacked.cstep;
    constants[6].i = top_blob.dims;
    constants[7].i = top_blob.w;
    constants[8].i = top_blob.h;
    constants[9].i = top_blob.d;
    constants[10].i = top_blob.c;
    constants[11].i = top_blob.cstep;
    constants[12].i = _woffset;
    constants[13].i = _hoffset;
    constants[14].i = _doffset;
    constants[15].i = _coffset;

    const Pipeline* pipeline = 0;
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_crop;
    }
    else if (elempack == 4 && offset_elempack == 4 && out_elempack == 4)
    {
        pipeline = pipeline_crop_pack4;
    }
    else if (elempack == 4 && offset_elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_crop_pack1to4;
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_crop_pack1to4;
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        pipeline = pipeline_crop_pack4to1;
    }
    else if (elempack == 8 && offset_elempack == 8 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack8;
    }
    else if (elempack == 8 && offset_elempack == 4 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack4to8;
    }
    else if (elempack == 8 && offset_elempack == 1 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack1to8;
    }
    else if (elempack == 1 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack1to8;
    }
    else if (elempack == 4 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack4to8;
    }
    else if (elempack == 8 && out_elempack == 4)
    {
        pipeline = pipeline_crop_pack8to4;
    }
    else if (elempack == 8 && out_elempack == 1)
    {
        pipeline = pipeline_crop_pack8to1;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

int Crop_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    const VkMat& reference_blob = bottom_blobs[1];
    VkMat& top_blob = top_blobs[0];

    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int _woffset, _hoffset, _doffset, _coffset;
    int _outw, _outh, _outd, _outc;
    if (woffset == -233)
    {
        resolve_crop_roi(bottom_blob.shape(), (const int*)reference_blob.mapped(), _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
    }
    else
    {
        resolve_crop_roi(bottom_blob.shape(), reference_blob.shape(), _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
    }

    int offset_elempack;
    int out_elempack;

    if (dims == 1)
    {
        if (_woffset == 0 && _outw == bottom_blob.w * elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        offset_elempack = _woffset == 0 ? elempack : opt.use_shader_pack8 && _woffset % 8 == 0 ? 8 : _woffset % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_shader_pack8 && _outw % 8 == 0 ? 8 : _outw % 4 == 0 ? 4 : 1;
    }
    else if (dims == 2)
    {
        if (_woffset == 0 && _hoffset == 0 && _outw == bottom_blob.w && _outh == bottom_blob.h * elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        offset_elempack = _hoffset == 0 ? elempack : opt.use_shader_pack8 && _hoffset % 8 == 0 ? 8 : _hoffset % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_shader_pack8 && _outh % 8 == 0 ? 8 : _outh % 4 == 0 ? 4 : 1;
    }
    else if (dims == 3)
    {
        if (_woffset == 0 && _hoffset == 0 && _coffset == 0 && _outw == bottom_blob.w && _outh == bottom_blob.h && _outc == bottom_blob.c * elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        offset_elempack = _coffset == 0 ? elempack : opt.use_shader_pack8 && _coffset % 8 == 0 ? 8 : _coffset % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_shader_pack8 && _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
    }
    else // if (dims == 4)
    {
        if (_woffset == 0 && _hoffset == 0 && _doffset == 0 && _coffset == 0 && _outw == bottom_blob.w && _outh == bottom_blob.h && _outd == bottom_blob.d && _outc == bottom_blob.c * elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        offset_elempack = _coffset == 0 ? elempack : opt.use_shader_pack8 && _coffset % 8 == 0 ? 8 : _coffset % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_shader_pack8 && _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
    }

    offset_elempack = std::min(offset_elempack, elempack);

    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    // unpacking
    VkMat bottom_blob_unpacked = bottom_blob;
    if (elempack == out_elempack && elempack > offset_elempack)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

        vkdev->convert_packing(bottom_blob, bottom_blob_unpacked, offset_elempack, cmd, opt_pack1);
    }

    if (dims == 1)
    {
        top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else if (dims == 2)
    {
        top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else if (dims == 3)
    {
        top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else // if (dims == 4)
    {
        top_blob.create(_outw, _outh, _outd, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob_unpacked;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(16);
    constants[0].i = bottom_blob_unpacked.dims;
    constants[1].i = bottom_blob_unpacked.w;
    constants[2].i = bottom_blob_unpacked.h;
    constants[3].i = bottom_blob_unpacked.d;
    constants[4].i = bottom_blob_unpacked.c;
    constants[5].i = bottom_blob_unpacked.cstep;
    constants[6].i = top_blob.dims;
    constants[7].i = top_blob.w;
    constants[8].i = top_blob.h;
    constants[9].i = top_blob.d;
    constants[10].i = top_blob.c;
    constants[11].i = top_blob.cstep;
    constants[12].i = _woffset;
    constants[13].i = _hoffset;
    constants[14].i = _doffset;
    constants[15].i = _coffset;

    const Pipeline* pipeline = 0;
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_crop;
    }
    else if (elempack == 4 && offset_elempack == 4 && out_elempack == 4)
    {
        pipeline = pipeline_crop_pack4;
    }
    else if (elempack == 4 && offset_elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_crop_pack1to4;
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_crop_pack1to4;
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        pipeline = pipeline_crop_pack4to1;
    }
    else if (elempack == 8 && offset_elempack == 8 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack8;
    }
    else if (elempack == 8 && offset_elempack == 4 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack4to8;
    }
    else if (elempack == 8 && offset_elempack == 1 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack1to8;
    }
    else if (elempack == 1 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack1to8;
    }
    else if (elempack == 4 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack4to8;
    }
    else if (elempack == 8 && out_elempack == 4)
    {
        pipeline = pipeline_crop_pack8to4;
    }
    else if (elempack == 8 && out_elempack == 1)
    {
        pipeline = pipeline_crop_pack8to1;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

int Crop_vulkan::forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int _woffset, _hoffset, _doffset, _coffset;
    int _outw, _outh, _outd, _outc;
    resolve_crop_roi(bottom_blob.shape(), _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);

    int offset_elempack;
    int out_elempack;

    if (dims == 1)
    {
        if (_woffset == 0 && _outw == bottom_blob.w * elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        offset_elempack = _woffset == 0 ? elempack : opt.use_shader_pack8 && _woffset % 8 == 0 ? 8 : _woffset % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_shader_pack8 && _outw % 8 == 0 ? 8 : _outw % 4 == 0 ? 4 : 1;
    }
    else if (dims == 2)
    {
        if (_woffset == 0 && _hoffset == 0 && _outw == bottom_blob.w && _outh == bottom_blob.h * elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        offset_elempack = _hoffset == 0 ? elempack : opt.use_shader_pack8 && _hoffset % 8 == 0 ? 8 : _hoffset % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_shader_pack8 && _outh % 8 == 0 ? 8 : _outh % 4 == 0 ? 4 : 1;
    }
    else if (dims == 3)
    {
        if (_woffset == 0 && _hoffset == 0 && _coffset == 0 && _outw == bottom_blob.w && _outh == bottom_blob.h && _outc == bottom_blob.c * elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        offset_elempack = _coffset == 0 ? elempack : opt.use_shader_pack8 && _coffset % 8 == 0 ? 8 : _coffset % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_shader_pack8 && _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
    }
    else // if (dims == 4)
    {
        if (_woffset == 0 && _hoffset == 0 && _doffset == 0 && _coffset == 0 && _outw == bottom_blob.w && _outh == bottom_blob.h && _outd == bottom_blob.d && _outc == bottom_blob.c * elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        offset_elempack = _coffset == 0 ? elempack : opt.use_shader_pack8 && _coffset % 8 == 0 ? 8 : _coffset % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_shader_pack8 && _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
    }

    offset_elempack = std::min(offset_elempack, elempack);

    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    // unpacking
    VkImageMat bottom_blob_unpacked = bottom_blob;
    if (elempack == out_elempack && elempack > offset_elempack)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

        vkdev->convert_packing(bottom_blob, bottom_blob_unpacked, offset_elempack, cmd, opt_pack1);
    }

    if (dims == 1)
    {
        top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else if (dims == 2)
    {
        top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else if (dims == 3)
    {
        top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else // if (dims == 4)
    {
        top_blob.create(_outw, _outh, _outd, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    if (top_blob.empty())
        return -100;

    std::vector<VkImageMat> bindings(2);
    bindings[0] = bottom_blob_unpacked;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(16);
    constants[0].i = bottom_blob_unpacked.dims;
    constants[1].i = bottom_blob_unpacked.w;
    constants[2].i = bottom_blob_unpacked.h;
    constants[3].i = bottom_blob_unpacked.d;
    constants[4].i = bottom_blob_unpacked.c;
    constants[5].i = 0; //bottom_blob_unpacked.cstep;
    constants[6].i = top_blob.dims;
    constants[7].i = top_blob.w;
    constants[8].i = top_blob.h;
    constants[9].i = top_blob.d;
    constants[10].i = top_blob.c;
    constants[11].i = 0; //top_blob.cstep;
    constants[12].i = _woffset;
    constants[13].i = _hoffset;
    constants[14].i = _doffset;
    constants[15].i = _coffset;

    const Pipeline* pipeline = 0;
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_crop;
    }
    else if (elempack == 4 && offset_elempack == 4 && out_elempack == 4)
    {
        pipeline = pipeline_crop_pack4;
    }
    else if (elempack == 4 && offset_elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_crop_pack1to4;
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_crop_pack1to4;
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        pipeline = pipeline_crop_pack4to1;
    }
    else if (elempack == 8 && offset_elempack == 8 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack8;
    }
    else if (elempack == 8 && offset_elempack == 4 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack4to8;
    }
    else if (elempack == 8 && offset_elempack == 1 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack1to8;
    }
    else if (elempack == 1 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack1to8;
    }
    else if (elempack == 4 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack4to8;
    }
    else if (elempack == 8 && out_elempack == 4)
    {
        pipeline = pipeline_crop_pack8to4;
    }
    else if (elempack == 8 && out_elempack == 1)
    {
        pipeline = pipeline_crop_pack8to1;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

int Crop_vulkan::forward(const std::vector<VkImageMat>& bottom_blobs, std::vector<VkImageMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkImageMat& bottom_blob = bottom_blobs[0];
    const VkImageMat& reference_blob = bottom_blobs[1];
    VkImageMat& top_blob = top_blobs[0];

    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int _woffset, _hoffset, _doffset, _coffset;
    int _outw, _outh, _outd, _outc;
    if (woffset == -233)
    {
        resolve_crop_roi(bottom_blob.shape(), (const int*)reference_blob.mapped(), _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
    }
    else
    {
        resolve_crop_roi(bottom_blob.shape(), reference_blob.shape(), _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
    }

    int offset_elempack;
    int out_elempack;

    if (dims == 1)
    {
        if (_woffset == 0 && _outw == bottom_blob.w * elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        offset_elempack = _woffset == 0 ? elempack : opt.use_shader_pack8 && _woffset % 8 == 0 ? 8 : _woffset % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_shader_pack8 && _outw % 8 == 0 ? 8 : _outw % 4 == 0 ? 4 : 1;
    }
    else if (dims == 2)
    {
        if (_woffset == 0 && _hoffset == 0 && _outw == bottom_blob.w && _outh == bottom_blob.h * elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        offset_elempack = _hoffset == 0 ? elempack : opt.use_shader_pack8 && _hoffset % 8 == 0 ? 8 : _hoffset % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_shader_pack8 && _outh % 8 == 0 ? 8 : _outh % 4 == 0 ? 4 : 1;
    }
    else if (dims == 3)
    {
        if (_woffset == 0 && _hoffset == 0 && _coffset == 0 && _outw == bottom_blob.w && _outh == bottom_blob.h && _outc == bottom_blob.c * elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        offset_elempack = _coffset == 0 ? elempack : opt.use_shader_pack8 && _coffset % 8 == 0 ? 8 : _coffset % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_shader_pack8 && _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
    }
    else // if (dims == 4)
    {
        if (_woffset == 0 && _hoffset == 0 && _doffset == 0 && _coffset == 0 && _outw == bottom_blob.w && _outh == bottom_blob.h && _outd == bottom_blob.d && _outc == bottom_blob.c * elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        offset_elempack = _coffset == 0 ? elempack : opt.use_shader_pack8 && _coffset % 8 == 0 ? 8 : _coffset % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_shader_pack8 && _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
    }

    offset_elempack = std::min(offset_elempack, elempack);

    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    // unpacking
    VkImageMat bottom_blob_unpacked = bottom_blob;
    if (elempack == out_elempack && elempack > offset_elempack)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

        vkdev->convert_packing(bottom_blob, bottom_blob_unpacked, offset_elempack, cmd, opt_pack1);
    }

    if (dims == 1)
    {
        top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else if (dims == 2)
    {
        top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else if (dims == 3)
    {
        top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else // if (dims == 4)
    {
        top_blob.create(_outw, _outh, _outd, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    if (top_blob.empty())
        return -100;

    std::vector<VkImageMat> bindings(2);
    bindings[0] = bottom_blob_unpacked;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(16);
    constants[0].i = bottom_blob_unpacked.dims;
    constants[1].i = bottom_blob_unpacked.w;
    constants[2].i = bottom_blob_unpacked.h;
    constants[3].i = bottom_blob_unpacked.d;
    constants[4].i = bottom_blob_unpacked.c;
    constants[5].i = 0; //bottom_blob_unpacked.cstep;
    constants[6].i = top_blob.dims;
    constants[7].i = top_blob.w;
    constants[8].i = top_blob.h;
    constants[9].i = top_blob.d;
    constants[10].i = top_blob.c;
    constants[11].i = 0; //top_blob.cstep;
    constants[12].i = _woffset;
    constants[13].i = _hoffset;
    constants[14].i = _doffset;
    constants[15].i = _coffset;

    const Pipeline* pipeline = 0;
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_crop;
    }
    else if (elempack == 4 && offset_elempack == 4 && out_elempack == 4)
    {
        pipeline = pipeline_crop_pack4;
    }
    else if (elempack == 4 && offset_elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_crop_pack1to4;
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_crop_pack1to4;
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        pipeline = pipeline_crop_pack4to1;
    }
    else if (elempack == 8 && offset_elempack == 8 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack8;
    }
    else if (elempack == 8 && offset_elempack == 4 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack4to8;
    }
    else if (elempack == 8 && offset_elempack == 1 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack1to8;
    }
    else if (elempack == 1 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack1to8;
    }
    else if (elempack == 4 && out_elempack == 8)
    {
        pipeline = pipeline_crop_pack4to8;
    }
    else if (elempack == 8 && out_elempack == 4)
    {
        pipeline = pipeline_crop_pack8to4;
    }
    else if (elempack == 8 && out_elempack == 1)
    {
        pipeline = pipeline_crop_pack8to1;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn

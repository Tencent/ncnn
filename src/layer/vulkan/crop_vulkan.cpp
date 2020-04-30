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
#include <algorithm>
#include "layer_type.h"
#include "layer_shader_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Crop_vulkan)

Crop_vulkan::Crop_vulkan()
{
    support_vulkan = true;

    packing_pack1 = 0;
    packing_pack4 = 0;

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
    if (shape.dims == 3) elempack = opt.use_shader_pack8 && shape.c % 8 == 0 ? 8 : shape.c % 4 == 0 ? 4 : 1;

    int out_elempack = 1;
    if (out_shape.dims == 1) out_elempack = opt.use_shader_pack8 && out_shape.w % 8 == 0 ? 8 : out_shape.w % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 2) out_elempack = opt.use_shader_pack8 && out_shape.h % 8 == 0 ? 8 : out_shape.h % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 3) out_elempack = opt.use_shader_pack8 && out_shape.c % 8 == 0 ? 8 : out_shape.c % 4 == 0 ? 4 : 1;

    int offset_elempack = 1;
    bool numpy_style_slice = !starts.empty() && !ends.empty();
    if (numpy_style_slice)
    {
        // TODO vec and image crop
        offset_elempack = elempack;

        const int* starts_ptr = starts;
        const int* axes_ptr = axes;

        int _axes[3] = {0,1,2};
        int num_axis = axes.w;
        if (num_axis == 0)
        {
            num_axis = shape.dims;
        }
        else
        {
            for (int i=0; i<num_axis; i++)
            {
                int axis = axes_ptr[i];
                if (axis < 0)
                    axis = shape.dims + axis;
                _axes[i] = axis;
            }
        }

        for (int i=0; i<num_axis; i++)
        {
            int axis = _axes[i];

            if (axis == 0)
            {
                int start = starts_ptr[i];
                int _coffset = start >= 0 ? start : shape.c + start;
                offset_elempack = opt.use_shader_pack8 && _coffset % 8 == 0 ? 8 : _coffset % 4 == 0 ? 4 : 1;
            }
        }
    }
    else
    {
        // TODO vec and image crop
        if (coffset == 0)
            offset_elempack = elempack;
        else
            offset_elempack = opt.use_shader_pack8 && coffset % 8 == 0 ? 8 : coffset % 4 == 0 ? 4 : 1;
    }

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

    Mat out_shape_packed;
    if (out_shape.dims == 1) out_shape_packed = Mat(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 2) out_shape_packed = Mat(out_shape.w, out_shape.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    Mat shape_unpacked = shape_packed;
    if (bottom_shapes.size() == 1 && shape.dims != 0 && elempack == out_elempack && elempack > offset_elempack)
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
    }

    if (shape.dims == 0 || (elempack > 1 && offset_elempack == 1) || bottom_shapes.size() == 2)
    {
        packing_pack1 = ncnn::create_layer(ncnn::LayerType::Packing);
        packing_pack1->vkdev = vkdev;

        packing_pack1->bottom_shapes.resize(1);
        packing_pack1->bottom_shapes[0] = shape_packed;
        packing_pack1->top_shapes.resize(1);
        packing_pack1->top_shapes[0] = shape_unpacked;

        ncnn::ParamDict pd;
        pd.set(0, 1);

        packing_pack1->load_param(pd);

        packing_pack1->create_pipeline(opt);
    }

    if (shape.dims == 0 || (elempack > 4 && offset_elempack == 4) || bottom_shapes.size() == 2)
    {
        packing_pack4 = ncnn::create_layer(ncnn::LayerType::Packing);
        packing_pack4->vkdev = vkdev;

        packing_pack4->bottom_shapes.resize(1);
        packing_pack4->bottom_shapes[0] = shape_packed;
        packing_pack4->top_shapes.resize(1);
        packing_pack4->top_shapes[0] = shape_unpacked;

        ncnn::ParamDict pd;
        pd.set(0, 4);

        packing_pack4->load_param(pd);

        packing_pack4->create_pipeline(opt);
    }

    std::vector<vk_specialization_type> specializations(0 + 10);
    specializations[0 + 0].i = shape_unpacked.dims;
    specializations[0 + 1].i = shape_unpacked.w;
    specializations[0 + 2].i = shape_unpacked.h;
    specializations[0 + 3].i = shape_unpacked.c;
    specializations[0 + 4].i = shape_unpacked.cstep;
    specializations[0 + 5].i = out_shape_packed.dims;
    specializations[0 + 6].i = out_shape_packed.w;
    specializations[0 + 7].i = out_shape_packed.h;
    specializations[0 + 8].i = out_shape_packed.c;
    specializations[0 + 9].i = out_shape_packed.cstep;

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

int Crop_vulkan::destroy_pipeline(const Option& opt)
{
    if (packing_pack1)
    {
        packing_pack1->destroy_pipeline(opt);
        delete packing_pack1;
        packing_pack1 = 0;
    }

    if (packing_pack4)
    {
        packing_pack4->destroy_pipeline(opt);
        delete packing_pack4;
        packing_pack4 = 0;
    }

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
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int _woffset, _hoffset, _coffset;
    int _outw, _outh, _outc;
    resolve_crop_roi(bottom_blob.shape(), _woffset, _hoffset, _coffset, _outw, _outh, _outc);

    // TODO vec and image crop

    if (dims == 3)
    {
        int offset_elempack = _coffset == 0 ? elempack : opt.use_shader_pack8 && _coffset % 8 == 0 ? 8 : _coffset % 4 == 0 ? 4 : 1;

        int out_elempack = opt.use_shader_pack8 && _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8*2u;
            if (out_elempack == 4) out_elemsize = 4*2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        // unpacking
        VkMat bottom_blob_unpacked = bottom_blob;
        if (elempack == out_elempack && elempack > offset_elempack)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

            const Layer* packing = offset_elempack == 4 ? packing_pack4 : packing_pack1;
            packing->forward(bottom_blob, bottom_blob_unpacked, cmd, opt_pack1);
        }

        top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_blob_unpacked;
        bindings[1] = top_blob;

        std::vector<vk_constant_type> constants(13);
        constants[0].i = bottom_blob_unpacked.dims;
        constants[1].i = bottom_blob_unpacked.w;
        constants[2].i = bottom_blob_unpacked.h;
        constants[3].i = bottom_blob_unpacked.c;
        constants[4].i = bottom_blob_unpacked.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = top_blob.cstep;
        constants[10].i = _woffset;
        constants[11].i = _hoffset;
        constants[12].i = _coffset;

        const Pipeline* pipeline = 0;
        if (elempack == 1 && out_elempack == 1)
        {
            pipeline = pipeline_crop;
        }
        else if (elempack == 4 && offset_elempack == 4 && out_elempack == 4)
        {
            constants[12].i = _coffset / 4;

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
            constants[12].i = _coffset / 8;

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
    }

    return 0;
}

int Crop_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    const VkMat& reference_blob = bottom_blobs[1];

    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int _woffset, _hoffset, _coffset;
    int _outw, _outh, _outc;
    if (woffset == -233)
    {
        resolve_crop_roi(bottom_blob.shape(), (const int*)reference_blob.mapped(), _woffset, _hoffset, _coffset, _outw, _outh, _outc);
    }
    else
    {
        resolve_crop_roi(bottom_blob.shape(), reference_blob.shape(), _woffset, _hoffset, _coffset, _outw, _outh, _outc);
    }

    // TODO vec and image crop

    if (dims == 3)
    {
        int offset_elempack = _coffset == 0 ? elempack : opt.use_shader_pack8 && _coffset % 8 == 0 ? 8 : _coffset % 4 == 0 ? 4 : 1;

        int out_elempack = opt.use_shader_pack8 && _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8*2u;
            if (out_elempack == 4) out_elemsize = 4*2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        // unpacking
        VkMat bottom_blob_unpacked = bottom_blob;
        if (elempack == out_elempack && elempack > offset_elempack)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

            const Layer* packing = offset_elempack == 4 ? packing_pack4 : packing_pack1;
            packing->forward(bottom_blob, bottom_blob_unpacked, cmd, opt_pack1);
        }

        VkMat& top_blob = top_blobs[0];

        top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_blob_unpacked;
        bindings[1] = top_blob;

        std::vector<vk_constant_type> constants(13);
        constants[0].i = bottom_blob_unpacked.dims;
        constants[1].i = bottom_blob_unpacked.w;
        constants[2].i = bottom_blob_unpacked.h;
        constants[3].i = bottom_blob_unpacked.c;
        constants[4].i = bottom_blob_unpacked.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = top_blob.cstep;
        constants[10].i = _woffset;
        constants[11].i = _hoffset;
        constants[12].i = _coffset;

        const Pipeline* pipeline = 0;
        if (elempack == 1 && out_elempack == 1)
        {
            pipeline = pipeline_crop;
        }
        else if (elempack == 4 && offset_elempack == 4 && out_elempack == 4)
        {
            constants[12].i = _coffset / 4;

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
            constants[12].i = _coffset / 8;

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
    }

    return 0;
}

} // namespace ncnn

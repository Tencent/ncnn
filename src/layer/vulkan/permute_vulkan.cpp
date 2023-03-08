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

#include "permute_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Permute_vulkan::Permute_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    pipeline_permute = 0;
    pipeline_permute_pack4 = 0;
    pipeline_permute_pack1to4 = 0;
    pipeline_permute_pack4to1 = 0;
    pipeline_permute_pack8 = 0;
    pipeline_permute_pack1to8 = 0;
    pipeline_permute_pack4to8 = 0;
    pipeline_permute_pack8to4 = 0;
    pipeline_permute_pack8to1 = 0;
}

int Permute_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
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

    // check blob shape
    if (!vkdev->shape_support_image_storage(shape_packed) || !vkdev->shape_support_image_storage(out_shape_packed))
    {
        support_image_storage = false;
        opt.use_image_storage = false;
    }

    std::vector<vk_specialization_type> specializations(2 + 12);
    specializations[0].i = order_type;
    specializations[1].i = vkdev->info.bug_implicit_fp16_arithmetic();
    specializations[2 + 0].i = shape_packed.dims;
    specializations[2 + 1].i = shape_packed.w;
    specializations[2 + 2].i = shape_packed.h;
    specializations[2 + 3].i = shape_packed.d;
    specializations[2 + 4].i = shape_packed.c;
    specializations[2 + 5].i = shape_packed.cstep;
    specializations[2 + 6].i = out_shape_packed.dims;
    specializations[2 + 7].i = out_shape_packed.w;
    specializations[2 + 8].i = out_shape_packed.h;
    specializations[2 + 9].i = out_shape_packed.d;
    specializations[2 + 10].i = out_shape_packed.c;
    specializations[2 + 11].i = out_shape_packed.cstep;

    Mat local_size_xyz_bottom; // pack4to1 and pack8to1
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
    if (shape_packed.dims == 4)
    {
        local_size_xyz_bottom.w = std::min(4, shape_packed.w);
        local_size_xyz_bottom.h = std::min(4, shape_packed.h * shape_packed.d);
        local_size_xyz_bottom.c = std::min(4, shape_packed.c);
    }

    Mat local_size_xyz;
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
    if (shape.dims == 0 || (elempack == 1 && out_elempack == 1))
    {
        pipeline_permute = new Pipeline(vkdev);
        pipeline_permute->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_permute->create(LayerShaderType::permute, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || (elempack == 4 && out_elempack == 4))
    {
        pipeline_permute_pack4 = new Pipeline(vkdev);
        pipeline_permute_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_permute_pack4->create(LayerShaderType::permute_pack4, opt, specializations);
    }

    // pack1to4
    if (shape.dims == 0 || (elempack == 1 && out_elempack == 4))
    {
        pipeline_permute_pack1to4 = new Pipeline(vkdev);
        pipeline_permute_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_permute_pack1to4->create(LayerShaderType::permute_pack1to4, opt, specializations);
    }

    // pack4to1
    if (shape.dims == 0 || (elempack == 4 && out_elempack == 1))
    {
        pipeline_permute_pack4to1 = new Pipeline(vkdev);
        pipeline_permute_pack4to1->set_optimal_local_size_xyz(local_size_xyz_bottom);
        pipeline_permute_pack4to1->create(LayerShaderType::permute_pack4to1, opt, specializations);
    }

    // pack8
    if ((opt.use_shader_pack8 && shape.dims == 0) || (elempack == 8 && out_elempack == 8))
    {
        pipeline_permute_pack8 = new Pipeline(vkdev);
        pipeline_permute_pack8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_permute_pack8->create(LayerShaderType::permute_pack8, opt, specializations);
    }

    // pack1to8
    if ((opt.use_shader_pack8 && shape.dims == 0) || (elempack == 1 && out_elempack == 8))
    {
        pipeline_permute_pack1to8 = new Pipeline(vkdev);
        pipeline_permute_pack1to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_permute_pack1to8->create(LayerShaderType::permute_pack1to8, opt, specializations);
    }

    // pack4to8
    if ((opt.use_shader_pack8 && shape.dims == 0) || (elempack == 4 && out_elempack == 8))
    {
        pipeline_permute_pack4to8 = new Pipeline(vkdev);
        pipeline_permute_pack4to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_permute_pack4to8->create(LayerShaderType::permute_pack4to8, opt, specializations);
    }

    // pack8to4
    if ((opt.use_shader_pack8 && shape.dims == 0) || (elempack == 8 && out_elempack == 4))
    {
        pipeline_permute_pack8to4 = new Pipeline(vkdev);
        pipeline_permute_pack8to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_permute_pack8to4->create(LayerShaderType::permute_pack8to4, opt, specializations);
    }

    // pack8to1
    if ((opt.use_shader_pack8 && shape.dims == 0) || (elempack == 8 && out_elempack == 1))
    {
        pipeline_permute_pack8to1 = new Pipeline(vkdev);
        pipeline_permute_pack8to1->set_optimal_local_size_xyz(local_size_xyz_bottom);
        pipeline_permute_pack8to1->create(LayerShaderType::permute_pack8to1, opt, specializations);
    }

    return 0;
}

int Permute_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_permute;
    pipeline_permute = 0;

    delete pipeline_permute_pack4;
    pipeline_permute_pack4 = 0;

    delete pipeline_permute_pack1to4;
    pipeline_permute_pack1to4 = 0;

    delete pipeline_permute_pack4to1;
    pipeline_permute_pack4to1 = 0;

    delete pipeline_permute_pack8;
    pipeline_permute_pack8 = 0;

    delete pipeline_permute_pack1to8;
    pipeline_permute_pack1to8 = 0;

    delete pipeline_permute_pack4to8;
    pipeline_permute_pack4to8 = 0;

    delete pipeline_permute_pack8to4;
    pipeline_permute_pack8to4 = 0;

    delete pipeline_permute_pack8to1;
    pipeline_permute_pack8to1 = 0;

    return 0;
}

int Permute_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int dims = bottom_blob.dims;

    if (dims == 1 || order_type == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int out_elempack;
    size_t out_elemsize;

    if (dims == 2)
    {
        // order_type
        // 0 = w h
        // 1 = h w

        int outw;
        int outh;

        // if (order_type == 1)
        {
            outw = h * elempack;
            outh = w;
        }

        out_elempack = opt.use_shader_pack8 && outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
        out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8 * 2u;
            if (out_elempack == 4) out_elemsize = 4 * 2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }
    else if (dims == 3)
    {
        // order_type
        // 0 = w h c
        // 1 = h w c
        // 2 = w c h
        // 3 = c w h
        // 4 = h c w
        // 5 = c h w

        const int c = channels * elempack;

        int outw;
        int outh;
        int outc;

        if (order_type == 1)
        {
            outw = h;
            outh = w;
            outc = c;
        }
        else if (order_type == 2)
        {
            outw = w;
            outh = c;
            outc = h;
        }
        else if (order_type == 3)
        {
            outw = c;
            outh = w;
            outc = h;
        }
        else if (order_type == 4)
        {
            outw = h;
            outh = c;
            outc = w;
        }
        else // if (order_type == 5)
        {
            outw = c;
            outh = h;
            outc = w;
        }

        out_elempack = opt.use_shader_pack8 && outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
        out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8 * 2u;
            if (out_elempack == 4) out_elemsize = 4 * 2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }
    else // if (dims == 4)
    {
        // order_type
        // 0 = w h d c
        // 1 = h w d c
        // 2 = w d h c
        // 3 = d w h c
        // 4 = h d w c
        // 5 = d h w c
        // 6 = w h c d
        // 7 = h w c d
        // 8 = w c h d
        // 9 = c w h d
        //10 = h c w d
        //11 = c h w d
        //12 = w d c h
        //13 = d w c h
        //14 = w c d h
        //15 = c w d h
        //16 = d c w h
        //17 = c d w h
        //18 = h d c w
        //19 = d h c w
        //20 = h c d w
        //21 = c h d w
        //22 = d c h w
        //23 = c d h w

        const int c = channels * elempack;

        int outw;
        int outh;
        int outd;
        int outc;

        if (order_type == 1)
        {
            outw = h;
            outh = w;
            outd = d;
            outc = c;
        }
        else if (order_type == 2)
        {
            outw = w;
            outh = d;
            outd = h;
            outc = c;
        }
        else if (order_type == 3)
        {
            outw = d;
            outh = w;
            outd = h;
            outc = c;
        }
        else if (order_type == 4)
        {
            outw = h;
            outh = d;
            outd = w;
            outc = c;
        }
        else if (order_type == 5)
        {
            outw = d;
            outh = h;
            outd = w;
            outc = c;
        }
        else if (order_type == 6)
        {
            outw = w;
            outh = h;
            outd = c;
            outc = d;
        }
        else if (order_type == 7)
        {
            outw = h;
            outh = w;
            outd = c;
            outc = d;
        }
        else if (order_type == 8)
        {
            outw = w;
            outh = c;
            outd = h;
            outc = d;
        }
        else if (order_type == 9)
        {
            outw = c;
            outh = w;
            outd = h;
            outc = d;
        }
        else if (order_type == 10)
        {
            outw = h;
            outh = c;
            outd = w;
            outc = d;
        }
        else if (order_type == 11)
        {
            outw = c;
            outh = h;
            outd = w;
            outc = d;
        }
        else if (order_type == 12)
        {
            outw = w;
            outh = d;
            outd = c;
            outc = h;
        }
        else if (order_type == 13)
        {
            outw = d;
            outh = w;
            outd = c;
            outc = h;
        }
        else if (order_type == 14)
        {
            outw = w;
            outh = c;
            outd = d;
            outc = h;
        }
        else if (order_type == 15)
        {
            outw = c;
            outh = w;
            outd = d;
            outc = h;
        }
        else if (order_type == 16)
        {
            outw = d;
            outh = c;
            outd = w;
            outc = h;
        }
        else if (order_type == 17)
        {
            outw = c;
            outh = d;
            outd = w;
            outc = h;
        }
        else if (order_type == 18)
        {
            outw = h;
            outh = d;
            outd = c;
            outc = w;
        }
        else if (order_type == 19)
        {
            outw = d;
            outh = h;
            outd = c;
            outc = w;
        }
        else if (order_type == 20)
        {
            outw = h;
            outh = c;
            outd = d;
            outc = w;
        }
        else if (order_type == 21)
        {
            outw = c;
            outh = h;
            outd = d;
            outc = w;
        }
        else if (order_type == 22)
        {
            outw = d;
            outh = c;
            outd = h;
            outc = w;
        }
        else // if (order_type == 23)
        {
            outw = c;
            outh = d;
            outd = h;
            outc = w;
        }

        out_elempack = opt.use_shader_pack8 && outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
        out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8 * 2u;
            if (out_elempack == 4) out_elemsize = 4 * 2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(12);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.d;
    constants[4].i = bottom_blob.c;
    constants[5].i = bottom_blob.cstep;
    constants[6].i = top_blob.dims;
    constants[7].i = top_blob.w;
    constants[8].i = top_blob.h;
    constants[9].i = top_blob.d;
    constants[10].i = top_blob.c;
    constants[11].i = top_blob.cstep;

    if (elempack == 1 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_permute, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_permute_pack4, bindings, constants, top_blob);
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_permute_pack1to4, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_permute_pack4to1, bindings, constants, bottom_blob);
    }
    else if (elempack == 8 && out_elempack == 8)
    {
        cmd.record_pipeline(pipeline_permute_pack8, bindings, constants, top_blob);
    }
    else if (elempack == 1 && out_elempack == 8)
    {
        cmd.record_pipeline(pipeline_permute_pack1to8, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 8)
    {
        cmd.record_pipeline(pipeline_permute_pack4to8, bindings, constants, top_blob);
    }
    else if (elempack == 8 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_permute_pack8to4, bindings, constants, top_blob);
    }
    else if (elempack == 8 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_permute_pack8to1, bindings, constants, bottom_blob);
    }

    return 0;
}

int Permute_vulkan::forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int dims = bottom_blob.dims;

    if (dims == 1 || order_type == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int out_elempack;
    size_t out_elemsize;

    if (dims == 2)
    {
        // order_type
        // 0 = w h
        // 1 = h w

        int outw;
        int outh;

        // if (order_type == 1)
        {
            outw = h * elempack;
            outh = w;
        }

        out_elempack = opt.use_shader_pack8 && outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
        out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8 * 2u;
            if (out_elempack == 4) out_elemsize = 4 * 2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }
    else if (dims == 3)
    {
        // order_type
        // 0 = w h c
        // 1 = h w c
        // 2 = w c h
        // 3 = c w h
        // 4 = h c w
        // 5 = c h w

        const int c = channels * elempack;

        int outw;
        int outh;
        int outc;

        if (order_type == 1)
        {
            outw = h;
            outh = w;
            outc = c;
        }
        else if (order_type == 2)
        {
            outw = w;
            outh = c;
            outc = h;
        }
        else if (order_type == 3)
        {
            outw = c;
            outh = w;
            outc = h;
        }
        else if (order_type == 4)
        {
            outw = h;
            outh = c;
            outc = w;
        }
        else // if (order_type == 5)
        {
            outw = c;
            outh = h;
            outc = w;
        }

        out_elempack = opt.use_shader_pack8 && outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
        out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8 * 2u;
            if (out_elempack == 4) out_elemsize = 4 * 2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }
    else // if (dims == 4)
    {
        // order_type
        // 0 = w h d c
        // 1 = h w d c
        // 2 = w d h c
        // 3 = d w h c
        // 4 = h d w c
        // 5 = d h w c
        // 6 = w h c d
        // 7 = h w c d
        // 8 = w c h d
        // 9 = c w h d
        //10 = h c w d
        //11 = c h w d
        //12 = w d c h
        //13 = d w c h
        //14 = w c d h
        //15 = c w d h
        //16 = d c w h
        //17 = c d w h
        //18 = h d c w
        //19 = d h c w
        //20 = h c d w
        //21 = c h d w
        //22 = d c h w
        //23 = c d h w

        const int c = channels * elempack;

        int outw;
        int outh;
        int outd;
        int outc;

        if (order_type == 1)
        {
            outw = h;
            outh = w;
            outd = d;
            outc = c;
        }
        else if (order_type == 2)
        {
            outw = w;
            outh = d;
            outd = h;
            outc = c;
        }
        else if (order_type == 3)
        {
            outw = d;
            outh = w;
            outd = h;
            outc = c;
        }
        else if (order_type == 4)
        {
            outw = h;
            outh = d;
            outd = w;
            outc = c;
        }
        else if (order_type == 5)
        {
            outw = d;
            outh = h;
            outd = w;
            outc = c;
        }
        else if (order_type == 6)
        {
            outw = w;
            outh = h;
            outd = c;
            outc = d;
        }
        else if (order_type == 7)
        {
            outw = h;
            outh = w;
            outd = c;
            outc = d;
        }
        else if (order_type == 8)
        {
            outw = w;
            outh = c;
            outd = h;
            outc = d;
        }
        else if (order_type == 9)
        {
            outw = c;
            outh = w;
            outd = h;
            outc = d;
        }
        else if (order_type == 10)
        {
            outw = h;
            outh = c;
            outd = w;
            outc = d;
        }
        else if (order_type == 11)
        {
            outw = c;
            outh = h;
            outd = w;
            outc = d;
        }
        else if (order_type == 12)
        {
            outw = w;
            outh = d;
            outd = c;
            outc = h;
        }
        else if (order_type == 13)
        {
            outw = d;
            outh = w;
            outd = c;
            outc = h;
        }
        else if (order_type == 14)
        {
            outw = w;
            outh = c;
            outd = d;
            outc = h;
        }
        else if (order_type == 15)
        {
            outw = c;
            outh = w;
            outd = d;
            outc = h;
        }
        else if (order_type == 16)
        {
            outw = d;
            outh = c;
            outd = w;
            outc = h;
        }
        else if (order_type == 17)
        {
            outw = c;
            outh = d;
            outd = w;
            outc = h;
        }
        else if (order_type == 18)
        {
            outw = h;
            outh = d;
            outd = c;
            outc = w;
        }
        else if (order_type == 19)
        {
            outw = d;
            outh = h;
            outd = c;
            outc = w;
        }
        else if (order_type == 20)
        {
            outw = h;
            outh = c;
            outd = d;
            outc = w;
        }
        else if (order_type == 21)
        {
            outw = c;
            outh = h;
            outd = d;
            outc = w;
        }
        else if (order_type == 22)
        {
            outw = d;
            outh = c;
            outd = h;
            outc = w;
        }
        else // if (order_type == 23)
        {
            outw = c;
            outh = d;
            outd = h;
            outc = w;
        }

        out_elempack = opt.use_shader_pack8 && outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
        out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8 * 2u;
            if (out_elempack == 4) out_elemsize = 4 * 2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }

    std::vector<VkImageMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(12);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.d;
    constants[4].i = bottom_blob.c;
    constants[5].i = 0; //bottom_blob.cstep;
    constants[6].i = top_blob.dims;
    constants[7].i = top_blob.w;
    constants[8].i = top_blob.h;
    constants[9].i = top_blob.d;
    constants[10].i = top_blob.c;
    constants[11].i = 0; //top_blob.cstep;

    if (elempack == 1 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_permute, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_permute_pack4, bindings, constants, top_blob);
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_permute_pack1to4, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_permute_pack4to1, bindings, constants, bottom_blob);
    }
    else if (elempack == 8 && out_elempack == 8)
    {
        cmd.record_pipeline(pipeline_permute_pack8, bindings, constants, top_blob);
    }
    else if (elempack == 1 && out_elempack == 8)
    {
        cmd.record_pipeline(pipeline_permute_pack1to8, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 8)
    {
        cmd.record_pipeline(pipeline_permute_pack4to8, bindings, constants, top_blob);
    }
    else if (elempack == 8 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_permute_pack8to4, bindings, constants, top_blob);
    }
    else if (elempack == 8 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_permute_pack8to1, bindings, constants, bottom_blob);
    }

    return 0;
}

} // namespace ncnn

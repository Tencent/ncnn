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

#include "binaryop_vulkan.h"

#include "layer_shader_type.h"

#include <math.h>

namespace ncnn {

BinaryOp_vulkan::BinaryOp_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    pipeline_binaryop = 0;
    pipeline_binaryop_pack4 = 0;
    pipeline_binaryop_pack8 = 0;

    pipeline_binaryop_broadcast = 0;
    pipeline_binaryop_broadcast_pack4 = 0;
    pipeline_binaryop_broadcast_a1_pack4 = 0;
    pipeline_binaryop_broadcast_b1_pack4 = 0;
    pipeline_binaryop_broadcast_pack8 = 0;
    pipeline_binaryop_broadcast_a1_pack8 = 0;
    pipeline_binaryop_broadcast_b1_pack8 = 0;
}

int BinaryOp_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& shape1 = with_scalar ? shape : bottom_shapes.empty() ? Mat() : bottom_shapes[1];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = 1;
    if (shape.dims == 1) elempack = opt.use_shader_pack8 && shape.w % 8 == 0 ? 8 : shape.w % 4 == 0 ? 4 : 1;
    if (shape.dims == 2) elempack = opt.use_shader_pack8 && shape.h % 8 == 0 ? 8 : shape.h % 4 == 0 ? 4 : 1;
    if (shape.dims == 3) elempack = opt.use_shader_pack8 && shape.c % 8 == 0 ? 8 : shape.c % 4 == 0 ? 4 : 1;

    int elempack1 = 1;
    if (shape1.dims == 1) elempack1 = opt.use_shader_pack8 && shape1.w % 8 == 0 ? 8 : shape1.w % 4 == 0 ? 4 : 1;
    if (shape1.dims == 2) elempack1 = opt.use_shader_pack8 && shape1.h % 8 == 0 ? 8 : shape1.h % 4 == 0 ? 4 : 1;
    if (shape1.dims == 3) elempack1 = opt.use_shader_pack8 && shape1.c % 8 == 0 ? 8 : shape1.c % 4 == 0 ? 4 : 1;

    int out_elempack = 1;
    if (out_shape.dims == 1) out_elempack = opt.use_shader_pack8 && out_shape.w % 8 == 0 ? 8 : out_shape.w % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 2) out_elempack = opt.use_shader_pack8 && out_shape.h % 8 == 0 ? 8 : out_shape.h % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 3) out_elempack = opt.use_shader_pack8 && out_shape.c % 8 == 0 ? 8 : out_shape.c % 4 == 0 ? 4 : 1;

    size_t elemsize;
    size_t elemsize1;
    size_t out_elemsize;
    if (opt.use_fp16_storage)
    {
        elemsize = elempack * 2u;
        elemsize1 = elempack1 * 2u;
        out_elemsize = out_elempack * 2u;
    }
    else if (opt.use_fp16_packed)
    {
        elemsize = elempack == 1 ? 4u : elempack * 2u;
        elemsize1 = elempack1 == 1 ? 4u : elempack1 * 2u;
        out_elemsize = out_elempack == 1 ? 4u : out_elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
        elemsize1 = elempack1 * 4u;
        out_elemsize = out_elempack * 4u;
    }

    Mat shape_packed;
    if (shape.dims == 1) shape_packed = Mat(shape.w / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 2) shape_packed = Mat(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);

    Mat shape1_packed;
    if (shape1.dims == 1) shape1_packed = Mat(shape1.w / elempack1, (void*)0, elemsize1, elempack1);
    if (shape1.dims == 2) shape1_packed = Mat(shape1.w, shape1.h / elempack1, (void*)0, elemsize1, elempack1);
    if (shape1.dims == 3) shape1_packed = Mat(shape1.w, shape1.h, shape1.c / elempack1, (void*)0, elemsize1, elempack1);

    Mat out_shape_packed;
    if (out_shape.dims == 1) out_shape_packed = Mat(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 2) out_shape_packed = Mat(out_shape.w, out_shape.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    bool broadcast = true;
    if (shape.dims == shape1.dims && shape.w == shape1.w && shape.h == shape1.h && shape.c == shape1.c)
    {
        broadcast = false;
    }

    // no broadcast
    if (shape.dims == 0 || !broadcast)
    {
        std::vector<vk_specialization_type> specializations(3 + 15);
        specializations[0].i = op_type;
        specializations[1].i = with_scalar;
        specializations[2].f = b;
        specializations[3 + 0].i = shape_packed.dims;
        specializations[3 + 1].i = shape_packed.w;
        specializations[3 + 2].i = shape_packed.h;
        specializations[3 + 3].i = shape_packed.c;
        specializations[3 + 4].i = shape_packed.cstep;
        specializations[3 + 5].i = shape1_packed.dims;
        specializations[3 + 6].i = shape1_packed.w;
        specializations[3 + 7].i = shape1_packed.h;
        specializations[3 + 8].i = shape1_packed.c;
        specializations[3 + 9].i = shape1_packed.cstep;
        specializations[3 + 10].i = out_shape_packed.dims;
        specializations[3 + 11].i = out_shape_packed.w;
        specializations[3 + 12].i = out_shape_packed.h;
        specializations[3 + 13].i = out_shape_packed.c;
        specializations[3 + 14].i = out_shape_packed.cstep;

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
        if (shape.dims == 0 || elempack == 1)
        {
            pipeline_binaryop = new Pipeline(vkdev);
            pipeline_binaryop->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop->create(LayerShaderType::binaryop, opt, specializations);
        }

        // pack4
        if (shape.dims == 0 || elempack == 4)
        {
            pipeline_binaryop_pack4 = new Pipeline(vkdev);
            pipeline_binaryop_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_pack4->create(LayerShaderType::binaryop_pack4, opt, specializations);
        }

        // pack8
        if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
        {
            pipeline_binaryop_pack8 = new Pipeline(vkdev);
            pipeline_binaryop_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_pack8->create(LayerShaderType::binaryop_pack8, opt, specializations);
        }
    }

    // broadcast
    if (shape.dims == 0 || broadcast)
    {
        std::vector<vk_specialization_type> specializations(1 + 15);
        specializations[0].i = op_type;
        specializations[1 + 0].i = shape_packed.dims;
        specializations[1 + 1].i = shape_packed.w;
        specializations[1 + 2].i = shape_packed.h;
        specializations[1 + 3].i = shape_packed.c;
        specializations[1 + 4].i = shape_packed.cstep;
        specializations[1 + 5].i = shape1_packed.dims;
        specializations[1 + 6].i = shape1_packed.w;
        specializations[1 + 7].i = shape1_packed.h;
        specializations[1 + 8].i = shape1_packed.c;
        specializations[1 + 9].i = shape1_packed.cstep;
        specializations[1 + 10].i = out_shape_packed.dims;
        specializations[1 + 11].i = out_shape_packed.w;
        specializations[1 + 12].i = out_shape_packed.h;
        specializations[1 + 13].i = out_shape_packed.c;
        specializations[1 + 14].i = out_shape_packed.cstep;

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
        if (shape.dims == 0 || (elempack == 1 && elempack1 == 1))
        {
            pipeline_binaryop_broadcast = new Pipeline(vkdev);
            pipeline_binaryop_broadcast->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_broadcast->create(LayerShaderType::binaryop_broadcast, opt, specializations);
        }

        // pack4
        if (shape.dims == 0 || (elempack == 4 && elempack1 == 4))
        {
            pipeline_binaryop_broadcast_pack4 = new Pipeline(vkdev);
            pipeline_binaryop_broadcast_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_broadcast_pack4->create(LayerShaderType::binaryop_broadcast_pack4, opt, specializations);
        }

        if (shape.dims == 0 || (shape.dims == 1 && shape.w == 1 && elempack == 1 && elempack1 == 4)
                || (shape.dims == 3 && shape1.dims == 3 && shape1.w == shape.w && shape1.h == shape.h && shape.c == 1 && elempack == 1 && elempack1 == 4))
        {
            pipeline_binaryop_broadcast_a1_pack4 = new Pipeline(vkdev);
            pipeline_binaryop_broadcast_a1_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_broadcast_a1_pack4->create(LayerShaderType::binaryop_broadcast_a1_pack4, opt, specializations);
        }

        if (shape.dims == 0 || (shape1.dims == 1 && shape1.w == 1 && elempack1 == 1 && elempack == 4)
                || (shape.dims == 3 && shape1.dims == 3 && shape1.w == shape.w && shape1.h == shape.h && shape1.c == 1 && elempack1 == 1 && elempack == 4))
        {
            pipeline_binaryop_broadcast_b1_pack4 = new Pipeline(vkdev);
            pipeline_binaryop_broadcast_b1_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_broadcast_b1_pack4->create(LayerShaderType::binaryop_broadcast_b1_pack4, opt, specializations);
        }

        // pack8
        if ((opt.use_shader_pack8 && shape.dims == 0) || (elempack == 8 && elempack1 == 8))
        {
            pipeline_binaryop_broadcast_pack8 = new Pipeline(vkdev);
            pipeline_binaryop_broadcast_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_broadcast_pack8->create(LayerShaderType::binaryop_broadcast_pack8, opt, specializations);
        }

        if ((opt.use_shader_pack8 && shape.dims == 0) || (shape.dims == 1 && shape.w == 1 && elempack == 1 && elempack1 == 8)
                || (shape.dims == 3 && shape1.dims == 3 && shape1.w == shape.w && shape1.h == shape.h && shape.c == 1 && elempack == 1 && elempack1 == 8))
        {
            pipeline_binaryop_broadcast_a1_pack8 = new Pipeline(vkdev);
            pipeline_binaryop_broadcast_a1_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_broadcast_a1_pack8->create(LayerShaderType::binaryop_broadcast_a1_pack8, opt, specializations);
        }

        if ((opt.use_shader_pack8 && shape.dims == 0) || (shape1.dims == 1 && shape1.w == 1 && elempack1 == 1 && elempack == 8)
                || (shape.dims == 3 && shape1.dims == 3 && shape1.w == shape.w && shape1.h == shape.h && shape1.c == 1 && elempack1 == 1 && elempack == 8))
        {
            pipeline_binaryop_broadcast_b1_pack8 = new Pipeline(vkdev);
            pipeline_binaryop_broadcast_b1_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_broadcast_b1_pack8->create(LayerShaderType::binaryop_broadcast_b1_pack8, opt, specializations);
        }
    }

    return 0;
}

int BinaryOp_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_binaryop;
    pipeline_binaryop = 0;

    delete pipeline_binaryop_pack4;
    pipeline_binaryop_pack4 = 0;

    delete pipeline_binaryop_pack8;
    pipeline_binaryop_pack8 = 0;

    delete pipeline_binaryop_broadcast;
    pipeline_binaryop_broadcast = 0;

    delete pipeline_binaryop_broadcast_pack4;
    pipeline_binaryop_broadcast_pack4 = 0;

    delete pipeline_binaryop_broadcast_a1_pack4;
    pipeline_binaryop_broadcast_a1_pack4 = 0;

    delete pipeline_binaryop_broadcast_b1_pack4;
    pipeline_binaryop_broadcast_b1_pack4 = 0;

    delete pipeline_binaryop_broadcast_pack8;
    pipeline_binaryop_broadcast_pack8 = 0;

    delete pipeline_binaryop_broadcast_a1_pack8;
    pipeline_binaryop_broadcast_a1_pack8 = 0;

    delete pipeline_binaryop_broadcast_b1_pack8;
    pipeline_binaryop_broadcast_b1_pack8 = 0;

    return 0;
}

int BinaryOp_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    const VkMat& bottom_blob1 = bottom_blobs[1];

    VkMat& top_blob = top_blobs[0];

    // broadcast
    if (bottom_blob.dims > bottom_blob1.dims)
    {
        top_blob.create_like(bottom_blob, opt.blob_vkallocator);
    }
    else if (bottom_blob.dims < bottom_blob1.dims)
    {
        top_blob.create_like(bottom_blob1, opt.blob_vkallocator);
    }
    else // if (bottom_blob.dims == bottom_blob1.dims)
    {
        if (bottom_blob.w * bottom_blob.h * bottom_blob.c * bottom_blob.elempack >= bottom_blob1.w * bottom_blob1.h * bottom_blob1.c * bottom_blob1.elempack)
        {
            top_blob.create_like(bottom_blob, opt.blob_vkallocator);
        }
        else
        {
            top_blob.create_like(bottom_blob1, opt.blob_vkallocator);
        }
    }
    if (top_blob.empty())
        return -100;

    int out_elempack = top_blob.elempack;

    std::vector<VkMat> bindings(3);
    bindings[0] = bottom_blob;
    bindings[1] = bottom_blob1;
    bindings[2] = top_blob;

    std::vector<vk_constant_type> constants(15);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.c;
    constants[4].i = bottom_blob.cstep;
    constants[5].i = bottom_blob1.dims;
    constants[6].i = bottom_blob1.w;
    constants[7].i = bottom_blob1.h;
    constants[8].i = bottom_blob1.c;
    constants[9].i = bottom_blob1.cstep;
    constants[10].i = top_blob.dims;
    constants[11].i = top_blob.w;
    constants[12].i = top_blob.h;
    constants[13].i = top_blob.c;
    constants[14].i = top_blob.cstep;

    bool broadcast = true;
    if (bottom_blob.dims == bottom_blob1.dims
            && bottom_blob.w == bottom_blob1.w
            && bottom_blob.h == bottom_blob1.h
            && bottom_blob.c == bottom_blob1.c
            && bottom_blob.elempack == bottom_blob1.elempack)
    {
        broadcast = false;
    }

    const Pipeline* pipeline = 0;
    if (broadcast)
    {
        if (bottom_blob.elempack == 1 && bottom_blob1.elempack == 1)
        {
            pipeline = pipeline_binaryop_broadcast;
        }
        else
        {
            if (bottom_blob.dims == 1 && bottom_blob.w == 1 && bottom_blob.elempack == 1)
            {
                pipeline = out_elempack == 8 ? pipeline_binaryop_broadcast_a1_pack8 : pipeline_binaryop_broadcast_a1_pack4;
            }
            else if (bottom_blob1.dims == 1 && bottom_blob1.w == 1 && bottom_blob1.elempack == 1)
            {
                pipeline = out_elempack == 8 ? pipeline_binaryop_broadcast_b1_pack8 : pipeline_binaryop_broadcast_b1_pack4;
            }
            else if (bottom_blob.dims == 3 && bottom_blob1.dims == 3 && bottom_blob1.w == bottom_blob.w && bottom_blob1.h == bottom_blob.h && bottom_blob1.c == 1 && bottom_blob1.elempack == 1)
            {
                // special type 2
                pipeline = out_elempack == 8 ? pipeline_binaryop_broadcast_b1_pack8 : pipeline_binaryop_broadcast_b1_pack4;
            }
            else if (bottom_blob.dims == 3 && bottom_blob1.dims == 3 && bottom_blob1.w == bottom_blob.w && bottom_blob1.h == bottom_blob.h && bottom_blob.c == 1 && bottom_blob.elempack == 1)
            {
                // special type 4
                pipeline = out_elempack == 8 ? pipeline_binaryop_broadcast_a1_pack8 : pipeline_binaryop_broadcast_a1_pack4;
            }
            else
            {
                pipeline = out_elempack == 8 ? pipeline_binaryop_broadcast_pack8 : pipeline_binaryop_broadcast_pack4;
            }
        }
    }
    else
    {
        pipeline = out_elempack == 8 ? pipeline_binaryop_pack8
                   : out_elempack == 4 ? pipeline_binaryop_pack4
                   : pipeline_binaryop;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

int BinaryOp_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& /*opt*/) const
{
    int elempack = bottom_top_blob.elempack;

    std::vector<VkMat> bindings(3);
    bindings[0] = bottom_top_blob;
    bindings[1] = bottom_top_blob; // TODO use dummy buffer
    bindings[2] = bottom_top_blob; // TODO use dummy buffer

    std::vector<vk_constant_type> constants(15);
    constants[10].i = bottom_top_blob.dims;
    constants[11].i = bottom_top_blob.w;
    constants[12].i = bottom_top_blob.h;
    constants[13].i = bottom_top_blob.c;
    constants[14].i = bottom_top_blob.cstep;

    const Pipeline* pipeline = elempack == 8 ? pipeline_binaryop_pack8
                               : elempack == 4 ? pipeline_binaryop_pack4
                               : pipeline_binaryop;

    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

    return 0;
}

int BinaryOp_vulkan::forward(const std::vector<VkImageMat>& bottom_blobs, std::vector<VkImageMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkImageMat& bottom_blob = bottom_blobs[0];
    const VkImageMat& bottom_blob1 = bottom_blobs[1];

    VkImageMat& top_blob = top_blobs[0];

    // broadcast
    if (bottom_blob.dims > bottom_blob1.dims)
    {
        top_blob.create_like(bottom_blob, opt.blob_vkallocator);
    }
    else if (bottom_blob.dims < bottom_blob1.dims)
    {
        top_blob.create_like(bottom_blob1, opt.blob_vkallocator);
    }
    else // if (bottom_blob.dims == bottom_blob1.dims)
    {
        if (bottom_blob.w * bottom_blob.h * bottom_blob.c * bottom_blob.elempack >= bottom_blob1.w * bottom_blob1.h * bottom_blob1.c * bottom_blob1.elempack)
        {
            top_blob.create_like(bottom_blob, opt.blob_vkallocator);
        }
        else
        {
            top_blob.create_like(bottom_blob1, opt.blob_vkallocator);
        }
    }
    if (top_blob.empty())
        return -100;

    int out_elempack = top_blob.elempack;

    std::vector<VkImageMat> bindings(3);
    bindings[0] = bottom_blob;
    bindings[1] = bottom_blob1;
    bindings[2] = top_blob;

    std::vector<vk_constant_type> constants(15);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.c;
    constants[4].i = 0; //bottom_blob.cstep;
    constants[5].i = bottom_blob1.dims;
    constants[6].i = bottom_blob1.w;
    constants[7].i = bottom_blob1.h;
    constants[8].i = bottom_blob1.c;
    constants[9].i = 0; //bottom_blob1.cstep;
    constants[10].i = top_blob.dims;
    constants[11].i = top_blob.w;
    constants[12].i = top_blob.h;
    constants[13].i = top_blob.c;
    constants[14].i = 0; //top_blob.cstep;

    bool broadcast = true;
    if (bottom_blob.dims == bottom_blob1.dims
            && bottom_blob.w == bottom_blob1.w
            && bottom_blob.h == bottom_blob1.h
            && bottom_blob.c == bottom_blob1.c
            && bottom_blob.elempack == bottom_blob1.elempack)
    {
        broadcast = false;
    }

    const Pipeline* pipeline = 0;
    if (broadcast)
    {
        if (bottom_blob.elempack == 1 && bottom_blob1.elempack == 1)
        {
            pipeline = pipeline_binaryop_broadcast;
        }
        else
        {
            if (bottom_blob.dims == 1 && bottom_blob.w == 1 && bottom_blob.elempack == 1)
            {
                pipeline = out_elempack == 8 ? pipeline_binaryop_broadcast_a1_pack8 : pipeline_binaryop_broadcast_a1_pack4;
            }
            else if (bottom_blob1.dims == 1 && bottom_blob1.w == 1 && bottom_blob1.elempack == 1)
            {
                pipeline = out_elempack == 8 ? pipeline_binaryop_broadcast_b1_pack8 : pipeline_binaryop_broadcast_b1_pack4;
            }
            else if (bottom_blob.dims == 3 && bottom_blob1.dims == 3 && bottom_blob1.w == bottom_blob.w && bottom_blob1.h == bottom_blob.h && bottom_blob1.c == 1 && bottom_blob1.elempack == 1)
            {
                // special type 2
                pipeline = out_elempack == 8 ? pipeline_binaryop_broadcast_b1_pack8 : pipeline_binaryop_broadcast_b1_pack4;
            }
            else if (bottom_blob.dims == 3 && bottom_blob1.dims == 3 && bottom_blob1.w == bottom_blob.w && bottom_blob1.h == bottom_blob.h && bottom_blob.c == 1 && bottom_blob.elempack == 1)
            {
                // special type 4
                pipeline = out_elempack == 8 ? pipeline_binaryop_broadcast_a1_pack8 : pipeline_binaryop_broadcast_a1_pack4;
            }
            else
            {
                pipeline = out_elempack == 8 ? pipeline_binaryop_broadcast_pack8 : pipeline_binaryop_broadcast_pack4;
            }
        }
    }
    else
    {
        pipeline = out_elempack == 8 ? pipeline_binaryop_pack8
                   : out_elempack == 4 ? pipeline_binaryop_pack4
                   : pipeline_binaryop;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

int BinaryOp_vulkan::forward_inplace(VkImageMat& bottom_top_blob, VkCompute& cmd, const Option& /*opt*/) const
{
    int elempack = bottom_top_blob.elempack;

    std::vector<VkImageMat> bindings(3);
    bindings[0] = bottom_top_blob;
    bindings[1] = bottom_top_blob; // TODO use dummy buffer
    bindings[2] = bottom_top_blob;

    std::vector<vk_constant_type> constants(15);
    constants[10].i = bottom_top_blob.dims;
    constants[11].i = bottom_top_blob.w;
    constants[12].i = bottom_top_blob.h;
    constants[13].i = bottom_top_blob.c;
    constants[14].i = 0; //bottom_top_blob.cstep;

    const Pipeline* pipeline = elempack == 8 ? pipeline_binaryop_pack8
                               : elempack == 4 ? pipeline_binaryop_pack4
                               : pipeline_binaryop;

    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

    return 0;
}

} // namespace ncnn

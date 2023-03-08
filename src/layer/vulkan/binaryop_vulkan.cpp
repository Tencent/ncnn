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

    pipeline_binaryop_broadcast_inner[0] = 0;
    pipeline_binaryop_broadcast_inner[1] = 0;
    pipeline_binaryop_broadcast_inner_pack4[0] = 0;
    pipeline_binaryop_broadcast_inner_pack4[1] = 0;
    pipeline_binaryop_broadcast_inner_pack8[0] = 0;
    pipeline_binaryop_broadcast_inner_pack8[1] = 0;
    pipeline_binaryop_broadcast_outer[0] = 0;
    pipeline_binaryop_broadcast_outer[1] = 0;
    pipeline_binaryop_broadcast_outer_pack4[0] = 0;
    pipeline_binaryop_broadcast_outer_pack4[1] = 0;
    pipeline_binaryop_broadcast_outer_pack8[0] = 0;
    pipeline_binaryop_broadcast_outer_pack8[1] = 0;
}

static int get_reverse_op_type(int op_type)
{
    if (op_type == BinaryOp::Operation_SUB) return BinaryOp::Operation_RSUB;
    if (op_type == BinaryOp::Operation_DIV) return BinaryOp::Operation_RDIV;
    if (op_type == BinaryOp::Operation_POW) return BinaryOp::Operation_RPOW;
    if (op_type == BinaryOp::Operation_ATAN2) return BinaryOp::Operation_RATAN2;
    if (op_type == BinaryOp::Operation_RSUB) return BinaryOp::Operation_SUB;
    if (op_type == BinaryOp::Operation_RDIV) return BinaryOp::Operation_DIV;
    if (op_type == BinaryOp::Operation_RPOW) return BinaryOp::Operation_POW;
    if (op_type == BinaryOp::Operation_RATAN2) return BinaryOp::Operation_ATAN2;
    return op_type;
}

int BinaryOp_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& shape1 = with_scalar ? shape : bottom_shapes.empty() ? Mat() : bottom_shapes[1];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = 1;
    if (shape.dims == 1) elempack = opt.use_shader_pack8 && shape.w % 8 == 0 ? 8 : shape.w % 4 == 0 ? 4 : 1;
    if (shape.dims == 2) elempack = opt.use_shader_pack8 && shape.h % 8 == 0 ? 8 : shape.h % 4 == 0 ? 4 : 1;
    if (shape.dims == 3 || shape.dims == 4) elempack = opt.use_shader_pack8 && shape.c % 8 == 0 ? 8 : shape.c % 4 == 0 ? 4 : 1;

    int elempack1 = 1;
    if (shape1.dims == 1) elempack1 = opt.use_shader_pack8 && shape1.w % 8 == 0 ? 8 : shape1.w % 4 == 0 ? 4 : 1;
    if (shape1.dims == 2) elempack1 = opt.use_shader_pack8 && shape1.h % 8 == 0 ? 8 : shape1.h % 4 == 0 ? 4 : 1;
    if (shape1.dims == 3 || shape1.dims == 4) elempack1 = opt.use_shader_pack8 && shape1.c % 8 == 0 ? 8 : shape1.c % 4 == 0 ? 4 : 1;

    int out_elempack = 1;
    if (out_shape.dims == 1) out_elempack = opt.use_shader_pack8 && out_shape.w % 8 == 0 ? 8 : out_shape.w % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 2) out_elempack = opt.use_shader_pack8 && out_shape.h % 8 == 0 ? 8 : out_shape.h % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 3 || out_shape.dims == 4) out_elempack = opt.use_shader_pack8 && out_shape.c % 8 == 0 ? 8 : out_shape.c % 4 == 0 ? 4 : 1;

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
    if (shape.dims == 4) shape_packed = Mat(shape.w, shape.h, shape.d, shape.c / elempack, (void*)0, elemsize, elempack);

    Mat shape1_packed;
    if (shape1.dims == 1) shape1_packed = Mat(shape1.w / elempack1, (void*)0, elemsize1, elempack1);
    if (shape1.dims == 2) shape1_packed = Mat(shape1.w, shape1.h / elempack1, (void*)0, elemsize1, elempack1);
    if (shape1.dims == 3) shape1_packed = Mat(shape1.w, shape1.h, shape1.c / elempack1, (void*)0, elemsize1, elempack1);
    if (shape1.dims == 4) shape1_packed = Mat(shape1.w, shape1.h, shape1.d, shape1.c / elempack1, (void*)0, elemsize1, elempack1);

    Mat out_shape_packed;
    if (out_shape.dims == 1) out_shape_packed = Mat(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 2) out_shape_packed = Mat(out_shape.w, out_shape.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 4) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.d, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    bool broadcast = true;
    if (shape.dims == shape1.dims && shape.w == shape1.w && shape.h == shape1.h && shape.d == shape1.d && shape.c == shape1.c)
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
        specializations[3 + 2].i = shape_packed.h * shape_packed.d;
        specializations[3 + 3].i = shape_packed.c;
        specializations[3 + 4].i = shape_packed.cstep;
        specializations[3 + 5].i = shape1_packed.dims;
        specializations[3 + 6].i = shape1_packed.w;
        specializations[3 + 7].i = shape1_packed.h * shape1_packed.d;
        specializations[3 + 8].i = shape1_packed.c;
        specializations[3 + 9].i = shape1_packed.cstep;
        specializations[3 + 10].i = out_shape_packed.dims;
        specializations[3 + 11].i = out_shape_packed.w;
        specializations[3 + 12].i = out_shape_packed.h * out_shape_packed.d;
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
        if (out_shape_packed.dims == 4)
        {
            local_size_xyz.w = std::min(4, out_shape_packed.w);
            local_size_xyz.h = std::min(4, out_shape_packed.h * out_shape_packed.d);
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
        bool a_is_lower = false;
        if (shape.dims != 0 && shape1.dims != 0)
        {
            const bool b_is_scalar = shape1_packed.w * shape1_packed.h * shape1_packed.d * shape1_packed.c * shape1_packed.elempack == 1;
            const bool a_rank_is_lower = shape_packed.dims < shape1_packed.dims && !b_is_scalar;
            const bool a_size_is_lower = shape_packed.w * shape_packed.h * shape_packed.d * shape_packed.c * shape_packed.elempack < shape1_packed.w * shape1_packed.h * shape1_packed.d * shape1_packed.c * shape1_packed.elempack;
            a_is_lower = a_rank_is_lower || (!a_rank_is_lower && a_size_is_lower);
        }
        const Mat& A_shape_packed = a_is_lower ? shape1_packed : shape_packed;
        const Mat& B_shape_packed = a_is_lower ? shape_packed : shape1_packed;

        const int op_type_r = get_reverse_op_type(op_type);

        std::vector<vk_specialization_type> specializations(1 + 18);
        specializations[0].i = op_type;
        specializations[1 + 0].i = A_shape_packed.dims;
        specializations[1 + 1].i = A_shape_packed.w;
        specializations[1 + 2].i = A_shape_packed.h;
        specializations[1 + 3].i = A_shape_packed.d;
        specializations[1 + 4].i = A_shape_packed.c;
        specializations[1 + 5].i = A_shape_packed.cstep;
        specializations[1 + 6].i = B_shape_packed.dims;
        specializations[1 + 7].i = B_shape_packed.w;
        specializations[1 + 8].i = B_shape_packed.h;
        specializations[1 + 9].i = B_shape_packed.d;
        specializations[1 + 10].i = B_shape_packed.c;
        specializations[1 + 11].i = B_shape_packed.cstep;
        specializations[1 + 12].i = out_shape_packed.dims;
        specializations[1 + 13].i = out_shape_packed.w;
        specializations[1 + 14].i = out_shape_packed.h;
        specializations[1 + 15].i = out_shape_packed.d;
        specializations[1 + 16].i = out_shape_packed.c;
        specializations[1 + 17].i = out_shape_packed.cstep;

        std::vector<vk_specialization_type> specializations_r(1 + 18);
        specializations_r[0].i = op_type_r;
        specializations_r[1 + 0].i = A_shape_packed.dims;
        specializations_r[1 + 1].i = A_shape_packed.w;
        specializations_r[1 + 2].i = A_shape_packed.h;
        specializations_r[1 + 3].i = A_shape_packed.d;
        specializations_r[1 + 4].i = A_shape_packed.c;
        specializations_r[1 + 5].i = A_shape_packed.cstep;
        specializations_r[1 + 6].i = B_shape_packed.dims;
        specializations_r[1 + 7].i = B_shape_packed.w;
        specializations_r[1 + 8].i = B_shape_packed.h;
        specializations_r[1 + 9].i = B_shape_packed.d;
        specializations_r[1 + 10].i = B_shape_packed.c;
        specializations_r[1 + 11].i = B_shape_packed.cstep;
        specializations_r[1 + 12].i = out_shape_packed.dims;
        specializations_r[1 + 13].i = out_shape_packed.w;
        specializations_r[1 + 14].i = out_shape_packed.h;
        specializations_r[1 + 15].i = out_shape_packed.d;
        specializations_r[1 + 16].i = out_shape_packed.c;
        specializations_r[1 + 17].i = out_shape_packed.cstep;

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
        if (shape.dims == 0 || (out_elempack == 1))
        {
            pipeline_binaryop_broadcast_inner[0] = new Pipeline(vkdev);
            pipeline_binaryop_broadcast_inner[0]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_broadcast_inner[0]->create(LayerShaderType::binaryop_broadcast_inner, opt, specializations);

            pipeline_binaryop_broadcast_outer[0] = new Pipeline(vkdev);
            pipeline_binaryop_broadcast_outer[0]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_broadcast_outer[0]->create(LayerShaderType::binaryop_broadcast_outer, opt, specializations);

            if (op_type_r != op_type)
            {
                // sub div pow ...
                pipeline_binaryop_broadcast_inner[1] = new Pipeline(vkdev);
                pipeline_binaryop_broadcast_inner[1]->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_binaryop_broadcast_inner[1]->create(LayerShaderType::binaryop_broadcast_inner, opt, specializations_r);

                pipeline_binaryop_broadcast_outer[1] = new Pipeline(vkdev);
                pipeline_binaryop_broadcast_outer[1]->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_binaryop_broadcast_outer[1]->create(LayerShaderType::binaryop_broadcast_outer, opt, specializations_r);
            }
        }

        // pack4
        if (shape.dims == 0 || (out_elempack == 4))
        {
            pipeline_binaryop_broadcast_inner_pack4[0] = new Pipeline(vkdev);
            pipeline_binaryop_broadcast_inner_pack4[0]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_broadcast_inner_pack4[0]->create(LayerShaderType::binaryop_broadcast_inner_pack4, opt, specializations);

            pipeline_binaryop_broadcast_outer_pack4[0] = new Pipeline(vkdev);
            pipeline_binaryop_broadcast_outer_pack4[0]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_broadcast_outer_pack4[0]->create(LayerShaderType::binaryop_broadcast_outer_pack4, opt, specializations);

            if (op_type_r != op_type)
            {
                // sub div pow ...
                pipeline_binaryop_broadcast_inner_pack4[1] = new Pipeline(vkdev);
                pipeline_binaryop_broadcast_inner_pack4[1]->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_binaryop_broadcast_inner_pack4[1]->create(LayerShaderType::binaryop_broadcast_inner_pack4, opt, specializations_r);

                pipeline_binaryop_broadcast_outer_pack4[1] = new Pipeline(vkdev);
                pipeline_binaryop_broadcast_outer_pack4[1]->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_binaryop_broadcast_outer_pack4[1]->create(LayerShaderType::binaryop_broadcast_outer_pack4, opt, specializations_r);
            }
        }

        // pack8
        if ((opt.use_shader_pack8 && shape.dims == 0) || (out_elempack == 8))
        {
            pipeline_binaryop_broadcast_inner_pack8[0] = new Pipeline(vkdev);
            pipeline_binaryop_broadcast_inner_pack8[0]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_broadcast_inner_pack8[0]->create(LayerShaderType::binaryop_broadcast_inner_pack8, opt, specializations);

            pipeline_binaryop_broadcast_outer_pack8[0] = new Pipeline(vkdev);
            pipeline_binaryop_broadcast_outer_pack8[0]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_broadcast_outer_pack8[0]->create(LayerShaderType::binaryop_broadcast_outer_pack8, opt, specializations);

            if (op_type_r != op_type)
            {
                // sub div pow ...
                pipeline_binaryop_broadcast_inner_pack8[1] = new Pipeline(vkdev);
                pipeline_binaryop_broadcast_inner_pack8[1]->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_binaryop_broadcast_inner_pack8[1]->create(LayerShaderType::binaryop_broadcast_inner_pack8, opt, specializations_r);

                pipeline_binaryop_broadcast_outer_pack8[1] = new Pipeline(vkdev);
                pipeline_binaryop_broadcast_outer_pack8[1]->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_binaryop_broadcast_outer_pack8[1]->create(LayerShaderType::binaryop_broadcast_outer_pack8, opt, specializations_r);
            }
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

    delete pipeline_binaryop_broadcast_inner[0];
    delete pipeline_binaryop_broadcast_inner[1];
    pipeline_binaryop_broadcast_inner[0] = 0;
    pipeline_binaryop_broadcast_inner[1] = 0;

    delete pipeline_binaryop_broadcast_inner_pack4[0];
    delete pipeline_binaryop_broadcast_inner_pack4[1];
    pipeline_binaryop_broadcast_inner_pack4[0] = 0;
    pipeline_binaryop_broadcast_inner_pack4[1] = 0;

    delete pipeline_binaryop_broadcast_inner_pack8[0];
    delete pipeline_binaryop_broadcast_inner_pack8[1];
    pipeline_binaryop_broadcast_inner_pack8[0] = 0;
    pipeline_binaryop_broadcast_inner_pack8[1] = 0;

    delete pipeline_binaryop_broadcast_outer[0];
    delete pipeline_binaryop_broadcast_outer[1];
    pipeline_binaryop_broadcast_outer[0] = 0;
    pipeline_binaryop_broadcast_outer[1] = 0;

    delete pipeline_binaryop_broadcast_outer_pack4[0];
    delete pipeline_binaryop_broadcast_outer_pack4[1];
    pipeline_binaryop_broadcast_outer_pack4[0] = 0;
    pipeline_binaryop_broadcast_outer_pack4[1] = 0;

    delete pipeline_binaryop_broadcast_outer_pack8[0];
    delete pipeline_binaryop_broadcast_outer_pack8[1];
    pipeline_binaryop_broadcast_outer_pack8[0] = 0;
    pipeline_binaryop_broadcast_outer_pack8[1] = 0;

    return 0;
}

int BinaryOp_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const bool b_is_scalar = bottom_blobs[1].w * bottom_blobs[1].h * bottom_blobs[1].d * bottom_blobs[1].c * bottom_blobs[1].elempack == 1;
    const bool a_rank_is_lower = bottom_blobs[0].dims < bottom_blobs[1].dims && !b_is_scalar;
    const bool a_size_is_lower = bottom_blobs[0].w * bottom_blobs[0].h * bottom_blobs[0].d * bottom_blobs[0].c * bottom_blobs[0].elempack < bottom_blobs[1].w * bottom_blobs[1].h * bottom_blobs[1].d * bottom_blobs[1].c * bottom_blobs[1].elempack;
    const bool a_is_lower = a_rank_is_lower || (!a_rank_is_lower && a_size_is_lower);
    const VkMat& A = a_is_lower ? bottom_blobs[1] : bottom_blobs[0];
    const VkMat& B = a_is_lower ? bottom_blobs[0] : bottom_blobs[1];
    const int op_type_r = a_is_lower ? get_reverse_op_type(op_type) : op_type;

    VkMat& top_blob = top_blobs[0];
    top_blob.create_like(A, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    int out_elempack = top_blob.elempack;

    std::vector<VkMat> bindings(3);
    bindings[0] = A;
    bindings[1] = B;
    bindings[2] = top_blob;

    // no broadcast
    if (A.dims == B.dims && A.w == B.w && A.h == B.h && A.d == B.d && A.c == B.c && A.elempack == B.elempack)
    {
        std::vector<vk_constant_type> constants(15);
        constants[0].i = A.dims;
        constants[1].i = A.w;
        constants[2].i = A.h * A.d;
        constants[3].i = A.c;
        constants[4].i = A.cstep;
        constants[5].i = B.dims;
        constants[6].i = B.w;
        constants[7].i = B.h * B.d;
        constants[8].i = B.c;
        constants[9].i = B.cstep;
        constants[10].i = top_blob.dims;
        constants[11].i = top_blob.w;
        constants[12].i = top_blob.h * top_blob.d;
        constants[13].i = top_blob.c;
        constants[14].i = top_blob.cstep;

        const Pipeline* pipeline = out_elempack == 8 ? pipeline_binaryop_pack8
                                   : out_elempack == 4 ? pipeline_binaryop_pack4
                                   : pipeline_binaryop;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    std::vector<vk_constant_type> constants(18);
    constants[0].i = A.dims;
    constants[1].i = A.w;
    constants[2].i = A.h;
    constants[3].i = A.d;
    constants[4].i = A.c;
    constants[5].i = A.cstep;
    constants[6].i = B.dims;
    constants[7].i = B.w;
    constants[8].i = B.h;
    constants[9].i = B.d;
    constants[10].i = B.c;
    constants[11].i = B.cstep;
    constants[12].i = top_blob.dims;
    constants[13].i = top_blob.w;
    constants[14].i = top_blob.h;
    constants[15].i = top_blob.d;
    constants[16].i = top_blob.c;
    constants[17].i = top_blob.cstep;

    const int ri = op_type_r == op_type ? 0 : 1;

    if (B.w * B.h * B.d * B.c * B.elempack == 1)
    {
        const Pipeline* pipeline = out_elempack == 8 ? pipeline_binaryop_broadcast_outer_pack8[ri]
                                   : out_elempack == 4 ? pipeline_binaryop_broadcast_outer_pack4[ri]
                                   : pipeline_binaryop_broadcast_outer[ri];

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    // broadcast B for inner axis
    if ((B.dims < A.dims)
            || (A.dims == 2 && B.w == 1 && B.h == A.h)
            || (A.dims == 3 && B.w == 1 && B.h == 1 && B.c == A.c)
            || (A.dims == 3 && B.w == 1 && B.h == A.h && B.c == A.c)
            || (A.dims == 4 && B.w == 1 && B.h == 1 && B.d == 1 && B.c == A.c)
            || (A.dims == 4 && B.w == 1 && B.h == 1 && B.d == A.d && B.c == A.c)
            || (A.dims == 4 && B.w == 1 && B.h == A.h && B.d == A.d && B.c == A.c))
    {
        const Pipeline* pipeline = out_elempack == 8 ? pipeline_binaryop_broadcast_inner_pack8[ri]
                                   : out_elempack == 4 ? pipeline_binaryop_broadcast_inner_pack4[ri]
                                   : pipeline_binaryop_broadcast_inner[ri];

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    // broadcast B for outer axis
    if (B.elempack == 1 && ((A.dims == 2 && B.w == A.w && B.h == 1) || (A.dims == 3 && B.w == A.w && B.h == 1 && B.c == 1) || (A.dims == 3 && B.w == A.w && B.h == A.h && B.c == 1) || (A.dims == 4 && B.w == A.w && B.h == 1 && B.d == 1 && B.c == 1) || (A.dims == 4 && B.w == A.w && B.h == A.h && B.d == 1 && B.c == 1) || (A.dims == 4 && B.w == A.w && B.h == A.h && B.d == A.d && B.c == 1)))
    {
        const Pipeline* pipeline = out_elempack == 8 ? pipeline_binaryop_broadcast_outer_pack8[ri]
                                   : out_elempack == 4 ? pipeline_binaryop_broadcast_outer_pack4[ri]
                                   : pipeline_binaryop_broadcast_outer[ri];

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    // some special broadcast rule here
    if (A.dims == 3 && B.dims == 3 && A.w == B.w && B.h == 1 && A.c == B.c)
    {
        const Pipeline* pipeline = out_elempack == 8 ? pipeline_binaryop_broadcast_inner_pack8[ri]
                                   : out_elempack == 4 ? pipeline_binaryop_broadcast_inner_pack4[ri]
                                   : pipeline_binaryop_broadcast_inner[ri];

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    // should never reach here
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
    constants[12].i = bottom_top_blob.h * bottom_top_blob.d;
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
    const bool b_is_scalar = bottom_blobs[1].w * bottom_blobs[1].h * bottom_blobs[1].d * bottom_blobs[1].c * bottom_blobs[1].elempack == 1;
    const bool a_rank_is_lower = bottom_blobs[0].dims < bottom_blobs[1].dims && !b_is_scalar;
    const bool a_size_is_lower = bottom_blobs[0].w * bottom_blobs[0].h * bottom_blobs[0].d * bottom_blobs[0].c * bottom_blobs[0].elempack < bottom_blobs[1].w * bottom_blobs[1].h * bottom_blobs[1].d * bottom_blobs[1].c * bottom_blobs[1].elempack;
    const bool a_is_lower = a_rank_is_lower || a_size_is_lower;
    const VkImageMat& A = a_is_lower ? bottom_blobs[1] : bottom_blobs[0];
    const VkImageMat& B = a_is_lower ? bottom_blobs[0] : bottom_blobs[1];
    const int op_type_r = a_is_lower ? get_reverse_op_type(op_type) : op_type;

    VkImageMat& top_blob = top_blobs[0];
    top_blob.create_like(A, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    int out_elempack = top_blob.elempack;

    std::vector<VkImageMat> bindings(3);
    bindings[0] = A;
    bindings[1] = B;
    bindings[2] = top_blob;

    // no broadcast
    if (A.dims == B.dims && A.w == B.w && A.h == B.h && A.d == B.d && A.c == B.c && A.elempack == B.elempack)
    {
        std::vector<vk_constant_type> constants(15);
        constants[0].i = A.dims;
        constants[1].i = A.w;
        constants[2].i = A.h * A.d;
        constants[3].i = A.c;
        constants[4].i = 0; //A.cstep;
        constants[5].i = B.dims;
        constants[6].i = B.w;
        constants[7].i = B.h * B.d;
        constants[8].i = B.c;
        constants[9].i = 0; //B.cstep;
        constants[10].i = top_blob.dims;
        constants[11].i = top_blob.w;
        constants[12].i = top_blob.h * top_blob.d;
        constants[13].i = top_blob.c;
        constants[14].i = 0; //top_blob.cstep;

        const Pipeline* pipeline = out_elempack == 8 ? pipeline_binaryop_pack8
                                   : out_elempack == 4 ? pipeline_binaryop_pack4
                                   : pipeline_binaryop;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    std::vector<vk_constant_type> constants(18);
    constants[0].i = A.dims;
    constants[1].i = A.w;
    constants[2].i = A.h;
    constants[3].i = A.d;
    constants[4].i = A.c;
    constants[5].i = 0; //A.cstep;
    constants[6].i = B.dims;
    constants[7].i = B.w;
    constants[8].i = B.h;
    constants[9].i = B.d;
    constants[10].i = B.c;
    constants[11].i = 0; //B.cstep;
    constants[12].i = top_blob.dims;
    constants[13].i = top_blob.w;
    constants[14].i = top_blob.h;
    constants[15].i = top_blob.d;
    constants[16].i = top_blob.c;
    constants[17].i = 0; //top_blob.cstep;

    const int ri = op_type_r == op_type ? 0 : 1;

    if (B.w * B.h * B.d * B.c * B.elempack == 1)
    {
        const Pipeline* pipeline = out_elempack == 8 ? pipeline_binaryop_broadcast_outer_pack8[ri]
                                   : out_elempack == 4 ? pipeline_binaryop_broadcast_outer_pack4[ri]
                                   : pipeline_binaryop_broadcast_outer[ri];

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    // broadcast B for inner axis
    if ((B.dims < A.dims)
            || (A.dims == 2 && B.w == 1 && B.h == A.h)
            || (A.dims == 3 && B.w == 1 && B.h == 1 && B.c == A.c)
            || (A.dims == 3 && B.w == 1 && B.h == A.h && B.c == A.c)
            || (A.dims == 4 && B.w == 1 && B.h == 1 && B.d == 1 && B.c == A.c)
            || (A.dims == 4 && B.w == 1 && B.h == 1 && B.d == A.d && B.c == A.c)
            || (A.dims == 4 && B.w == 1 && B.h == A.h && B.d == A.d && B.c == A.c))
    {
        const Pipeline* pipeline = out_elempack == 8 ? pipeline_binaryop_broadcast_inner_pack8[ri]
                                   : out_elempack == 4 ? pipeline_binaryop_broadcast_inner_pack4[ri]
                                   : pipeline_binaryop_broadcast_inner[ri];

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    // broadcast B for outer axis
    if (B.elempack == 1 && ((A.dims == 2 && B.w == A.w && B.h == 1) || (A.dims == 3 && B.w == A.w && B.h == 1 && B.c == 1) || (A.dims == 3 && B.w == A.w && B.h == A.h && B.c == 1) || (A.dims == 4 && B.w == A.w && B.h == 1 && B.d == 1 && B.c == 1) || (A.dims == 4 && B.w == A.w && B.h == A.h && B.d == 1 && B.c == 1) || (A.dims == 4 && B.w == A.w && B.h == A.h && B.d == A.d && B.c == 1)))
    {
        const Pipeline* pipeline = out_elempack == 8 ? pipeline_binaryop_broadcast_outer_pack8[ri]
                                   : out_elempack == 4 ? pipeline_binaryop_broadcast_outer_pack4[ri]
                                   : pipeline_binaryop_broadcast_outer[ri];

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    // some special broadcast rule here
    if (A.dims == 3 && B.dims == 3 && A.w == B.w && B.h == 1 && A.c == B.c)
    {
        const Pipeline* pipeline = out_elempack == 8 ? pipeline_binaryop_broadcast_inner_pack8[ri]
                                   : out_elempack == 4 ? pipeline_binaryop_broadcast_inner_pack4[ri]
                                   : pipeline_binaryop_broadcast_inner[ri];

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    // should never reach here
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
    constants[12].i = bottom_top_blob.h * bottom_top_blob.d;
    constants[13].i = bottom_top_blob.c;
    constants[14].i = 0; //bottom_top_blob.cstep;

    const Pipeline* pipeline = elempack == 8 ? pipeline_binaryop_pack8
                               : elempack == 4 ? pipeline_binaryop_pack4
                               : pipeline_binaryop;

    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

    return 0;
}

} // namespace ncnn

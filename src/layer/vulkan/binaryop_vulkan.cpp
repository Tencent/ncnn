// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "binaryop_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

BinaryOp_vulkan::BinaryOp_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_binaryop = 0;
    pipeline_binaryop_pack4 = 0;

    pipeline_binaryop_broadcast[0] = 0;
    pipeline_binaryop_broadcast[1] = 0;
    pipeline_binaryop_broadcast_pack4[0] = 0;
    pipeline_binaryop_broadcast_pack4[1] = 0;
    pipeline_binaryop_broadcast_pack1to4[0] = 0;
    pipeline_binaryop_broadcast_pack1to4[1] = 0;
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
    Mat A_shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    Mat B_shape = with_scalar ? A_shape : bottom_shapes.empty() ? Mat() : bottom_shapes[1];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    bool broadcast = true;
    if (A_shape.dims == B_shape.dims && A_shape.w == B_shape.w && A_shape.h == B_shape.h && A_shape.d == B_shape.d && A_shape.c == B_shape.c)
    {
        broadcast = false;
    }

    // no broadcast
    if (out_shape.dims == 0 || !broadcast)
    {
        std::vector<vk_specialization_type> specializations(3 + 15);
        specializations[0].i = op_type;
        specializations[1].i = with_scalar;
        specializations[2].f = b;
        specializations[3 + 0].i = A_shape.dims;
        specializations[3 + 1].i = A_shape.w;
        specializations[3 + 2].i = A_shape.h * A_shape.d;
        specializations[3 + 3].i = A_shape.c;
        specializations[3 + 4].i = A_shape.cstep;
        specializations[3 + 5].i = B_shape.dims;
        specializations[3 + 6].i = B_shape.w;
        specializations[3 + 7].i = B_shape.h * B_shape.d;
        specializations[3 + 8].i = B_shape.c;
        specializations[3 + 9].i = B_shape.cstep;
        specializations[3 + 10].i = out_shape.dims;
        specializations[3 + 11].i = out_shape.w;
        specializations[3 + 12].i = out_shape.h * out_shape.d;
        specializations[3 + 13].i = out_shape.c;
        specializations[3 + 14].i = out_shape.cstep;

        Mat local_size_xyz;
        if (out_shape.dims == 1)
        {
            local_size_xyz.w = std::min(64, out_shape.w);
            local_size_xyz.h = 1;
            local_size_xyz.c = 1;
        }
        if (out_shape.dims == 2)
        {
            local_size_xyz.w = std::min(8, out_shape.w);
            local_size_xyz.h = std::min(8, out_shape.h);
            local_size_xyz.c = 1;
        }
        if (out_shape.dims == 3)
        {
            local_size_xyz.w = std::min(4, out_shape.w);
            local_size_xyz.h = std::min(4, out_shape.h);
            local_size_xyz.c = std::min(4, out_shape.c);
        }
        if (out_shape.dims == 4)
        {
            local_size_xyz.w = std::min(4, out_shape.w);
            local_size_xyz.h = std::min(4, out_shape.h * out_shape.d);
            local_size_xyz.c = std::min(4, out_shape.c);
        }

        // pack1
        if (out_shape.dims == 0 || out_shape.elempack == 1)
        {
            pipeline_binaryop = new Pipeline(vkdev);
            pipeline_binaryop->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop->create(LayerShaderType::binaryop, opt, specializations);
        }

        // pack4
        if (out_shape.dims == 0 || out_shape.elempack == 4)
        {
            pipeline_binaryop_pack4 = new Pipeline(vkdev);
            pipeline_binaryop_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_pack4->create(LayerShaderType::binaryop_pack4, opt, specializations);
        }
    }

    // broadcast
    if (out_shape.dims == 0 || broadcast)
    {
        if (A_shape.dims > 0 && B_shape.dims > 0)
        {
            const bool a_rank_is_lower = A_shape.dims < B_shape.dims;
            const bool a_rank_is_equal = A_shape.dims == B_shape.dims;
            const bool a_pack_is_lower = A_shape.elempack < B_shape.elempack;
            const bool a_pack_is_equal = A_shape.elempack == B_shape.elempack;
            const bool a_size_is_lower = A_shape.w * A_shape.h * A_shape.d * A_shape.c < B_shape.w * B_shape.h * B_shape.d * B_shape.c;
            if (a_rank_is_lower || (a_rank_is_equal && a_pack_is_lower) || (a_pack_is_equal && a_size_is_lower))
            {
                // swap AB
                std::swap(A_shape, B_shape);
            }

            if (B_shape.dims == 1 && ((A_shape.dims == 2 && B_shape.w * B_shape.elempack != A_shape.h * A_shape.elempack) || ((A_shape.dims == 3 || A_shape.dims == 4) && B_shape.w * B_shape.elempack != A_shape.c * A_shape.elempack)))
            {
                B_shape.dims = out_shape.dims;
                B_shape.w = B_shape.w * B_shape.elempack;
                B_shape.elempack = 1;
            }
        }

        const int op_type_r = get_reverse_op_type(op_type);

        std::vector<vk_specialization_type> specializations(1 + 18);
        specializations[0].i = op_type;
        specializations[1 + 0].i = A_shape.dims;
        specializations[1 + 1].i = A_shape.w;
        specializations[1 + 2].i = A_shape.h;
        specializations[1 + 3].i = A_shape.d;
        specializations[1 + 4].i = A_shape.c;
        specializations[1 + 5].i = A_shape.cstep;
        specializations[1 + 6].i = B_shape.dims;
        specializations[1 + 7].i = B_shape.w;
        specializations[1 + 8].i = B_shape.h;
        specializations[1 + 9].i = B_shape.d;
        specializations[1 + 10].i = B_shape.c;
        specializations[1 + 11].i = B_shape.cstep;
        specializations[1 + 12].i = out_shape.dims;
        specializations[1 + 13].i = out_shape.w;
        specializations[1 + 14].i = out_shape.h;
        specializations[1 + 15].i = out_shape.d;
        specializations[1 + 16].i = out_shape.c;
        specializations[1 + 17].i = out_shape.cstep;

        std::vector<vk_specialization_type> specializations_r(1 + 18);
        specializations_r[0].i = op_type_r;
        specializations_r[1 + 0].i = A_shape.dims;
        specializations_r[1 + 1].i = A_shape.w;
        specializations_r[1 + 2].i = A_shape.h;
        specializations_r[1 + 3].i = A_shape.d;
        specializations_r[1 + 4].i = A_shape.c;
        specializations_r[1 + 5].i = A_shape.cstep;
        specializations_r[1 + 6].i = B_shape.dims;
        specializations_r[1 + 7].i = B_shape.w;
        specializations_r[1 + 8].i = B_shape.h;
        specializations_r[1 + 9].i = B_shape.d;
        specializations_r[1 + 10].i = B_shape.c;
        specializations_r[1 + 11].i = B_shape.cstep;
        specializations_r[1 + 12].i = out_shape.dims;
        specializations_r[1 + 13].i = out_shape.w;
        specializations_r[1 + 14].i = out_shape.h;
        specializations_r[1 + 15].i = out_shape.d;
        specializations_r[1 + 16].i = out_shape.c;
        specializations_r[1 + 17].i = out_shape.cstep;

        Mat local_size_xyz;
        if (out_shape.dims == 1)
        {
            local_size_xyz.w = std::min(64, out_shape.w);
            local_size_xyz.h = 1;
            local_size_xyz.c = 1;
        }
        if (out_shape.dims == 2)
        {
            local_size_xyz.w = std::min(8, out_shape.w);
            local_size_xyz.h = std::min(8, out_shape.h);
            local_size_xyz.c = 1;
        }
        if (out_shape.dims == 3)
        {
            local_size_xyz.w = std::min(4, out_shape.w);
            local_size_xyz.h = std::min(4, out_shape.h);
            local_size_xyz.c = std::min(4, out_shape.c);
        }
        if (out_shape.dims == 4)
        {
            local_size_xyz.w = std::min(4, out_shape.w);
            local_size_xyz.h = std::min(4, out_shape.h * out_shape.d);
            local_size_xyz.c = std::min(4, out_shape.c);
        }

        // pack1
        if (out_shape.dims == 0 || (out_shape.elempack == 1))
        {
            pipeline_binaryop_broadcast[0] = new Pipeline(vkdev);
            pipeline_binaryop_broadcast[0]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_broadcast[0]->create(LayerShaderType::binaryop_broadcast, opt, specializations);

            if (op_type_r != op_type)
            {
                // sub div pow ...
                pipeline_binaryop_broadcast[1] = new Pipeline(vkdev);
                pipeline_binaryop_broadcast[1]->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_binaryop_broadcast[1]->create(LayerShaderType::binaryop_broadcast, opt, specializations_r);
            }
        }

        // pack4
        if (out_shape.dims == 0 || (A_shape.elempack == 4 && B_shape.elempack == 4 && out_shape.elempack == 4))
        {
            pipeline_binaryop_broadcast_pack4[0] = new Pipeline(vkdev);
            pipeline_binaryop_broadcast_pack4[0]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_broadcast_pack4[0]->create(LayerShaderType::binaryop_broadcast_pack4, opt, specializations);

            if (op_type_r != op_type)
            {
                // sub div pow ...
                pipeline_binaryop_broadcast_pack4[1] = new Pipeline(vkdev);
                pipeline_binaryop_broadcast_pack4[1]->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_binaryop_broadcast_pack4[1]->create(LayerShaderType::binaryop_broadcast_pack4, opt, specializations_r);
            }
        }

        // pack1to4
        if (out_shape.dims == 0 || ((A_shape.elempack == 1 || B_shape.elempack == 1) && out_shape.elempack == 4))
        {
            pipeline_binaryop_broadcast_pack1to4[0] = new Pipeline(vkdev);
            pipeline_binaryop_broadcast_pack1to4[0]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_binaryop_broadcast_pack1to4[0]->create(LayerShaderType::binaryop_broadcast_pack1to4, opt, specializations);

            if (op_type_r != op_type)
            {
                // sub div pow ...
                pipeline_binaryop_broadcast_pack1to4[1] = new Pipeline(vkdev);
                pipeline_binaryop_broadcast_pack1to4[1]->set_optimal_local_size_xyz(local_size_xyz);
                pipeline_binaryop_broadcast_pack1to4[1]->create(LayerShaderType::binaryop_broadcast_pack1to4, opt, specializations_r);
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

    delete pipeline_binaryop_broadcast[0];
    delete pipeline_binaryop_broadcast[1];
    pipeline_binaryop_broadcast[0] = 0;
    pipeline_binaryop_broadcast[1] = 0;

    delete pipeline_binaryop_broadcast_pack4[0];
    delete pipeline_binaryop_broadcast_pack4[1];
    pipeline_binaryop_broadcast_pack4[0] = 0;
    pipeline_binaryop_broadcast_pack4[1] = 0;

    delete pipeline_binaryop_broadcast_pack1to4[0];
    delete pipeline_binaryop_broadcast_pack1to4[1];
    pipeline_binaryop_broadcast_pack1to4[0] = 0;
    pipeline_binaryop_broadcast_pack1to4[1] = 0;

    return 0;
}

int BinaryOp_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& A = bottom_blobs[0];
    const VkMat& B = bottom_blobs[1];
    const int outdims = std::max(A.dims, B.dims);

    const bool a_rank_is_lower = A.dims < B.dims;
    const bool b_rank_is_lower = B.dims < A.dims;
    const bool a_rank_is_equal = A.dims == B.dims;

    VkMat& top_blob = top_blobs[0];
    if (a_rank_is_lower)
    {
        top_blob.create_like(B, opt.blob_vkallocator);
    }
    else if (b_rank_is_lower)
    {
        top_blob.create_like(A, opt.blob_vkallocator);
    }
    else
    {
        const int outw = std::max(A.w, B.w);
        const int outh = std::max(A.h, B.h);
        const int outd = std::max(A.d, B.d);
        const int outc = std::max(A.c, B.c);
        const int out_elempack = std::max(A.elempack, B.elempack);
        const size_t out_elemsize = std::max(A.elemsize, B.elemsize);

        if (outdims == 1)
        {
            top_blob.create(outw, out_elemsize, out_elempack, opt.blob_vkallocator);
        }
        if (outdims == 2)
        {
            top_blob.create(outw, outh, out_elemsize, out_elempack, opt.blob_vkallocator);
        }
        if (outdims == 3)
        {
            top_blob.create(outw, outh, outc, out_elemsize, out_elempack, opt.blob_vkallocator);
        }
        if (outdims == 4)
        {
            top_blob.create(outw, outh, outd, outc, out_elemsize, out_elempack, opt.blob_vkallocator);
        }
    }
    if (top_blob.empty())
        return -100;

    // no broadcast
    if (A.dims == B.dims && A.w == B.w && A.h == B.h && A.d == B.d && A.c == B.c && A.elempack == B.elempack)
    {
        std::vector<VkMat> bindings(3);
        bindings[0] = A;
        bindings[1] = B;
        bindings[2] = top_blob;

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

        const Pipeline* pipeline = top_blob.elempack == 4 ? pipeline_binaryop_pack4 : pipeline_binaryop;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    const bool a_pack_is_lower = A.elempack < B.elempack;
    const bool a_pack_is_equal = A.elempack == B.elempack;
    const bool a_size_is_lower = A.w * A.h * A.d * A.c * A.elempack < B.w * B.h * B.d * B.c * B.elempack;
    if (a_rank_is_lower || (a_rank_is_equal && a_pack_is_lower) || (a_pack_is_equal && a_size_is_lower))
    {
        VkMat A2;
        if (A.dims == 1 && ((B.dims == 2 && A.w * A.elempack != B.h * B.elempack) || ((B.dims == 3 || B.dims == 4) && A.w * A.elempack != B.c * B.elempack)))
        {
            vkdev->convert_packing(A, A2, 1, cmd, opt);
            A2.dims = top_blob.dims;
        }
        else
        {
            A2 = A;
        }

        std::vector<VkMat> bindings(3);
        bindings[0] = B;
        bindings[1] = A2;
        bindings[2] = top_blob;

        std::vector<vk_constant_type> constants(18);
        constants[0].i = B.dims;
        constants[1].i = B.w;
        constants[2].i = B.h;
        constants[3].i = B.d;
        constants[4].i = B.c;
        constants[5].i = B.cstep;
        constants[6].i = A2.dims;
        constants[7].i = A2.w;
        constants[8].i = A2.h;
        constants[9].i = A2.d;
        constants[10].i = A2.c;
        constants[11].i = A2.cstep;
        constants[12].i = top_blob.dims;
        constants[13].i = top_blob.w;
        constants[14].i = top_blob.h;
        constants[15].i = top_blob.d;
        constants[16].i = top_blob.c;
        constants[17].i = top_blob.cstep;

        const int ri = get_reverse_op_type(op_type) == op_type ? 0 : 1;

        const Pipeline* pipeline = 0;
        if (A2.elempack == 1 && top_blob.elempack == 1)
        {
            pipeline = pipeline_binaryop_broadcast[ri];
        }
        if (A2.elempack == 4 && top_blob.elempack == 4)
        {
            pipeline = pipeline_binaryop_broadcast_pack4[ri];
        }
        if (A2.elempack == 1 && top_blob.elempack == 4)
        {
            pipeline = pipeline_binaryop_broadcast_pack1to4[ri];
        }

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    }
    else
    {
        VkMat B2;
        if (B.dims == 1 && ((A.dims == 2 && B.w * B.elempack != A.h * A.elempack) || ((A.dims == 3 || A.dims == 4) && B.w * B.elempack != A.c * A.elempack)))
        {
            vkdev->convert_packing(B, B2, 1, cmd, opt);
            B2.dims = top_blob.dims;
        }
        else
        {
            B2 = B;
        }

        std::vector<VkMat> bindings(3);
        bindings[0] = A;
        bindings[1] = B2;
        bindings[2] = top_blob;

        std::vector<vk_constant_type> constants(18);
        constants[0].i = A.dims;
        constants[1].i = A.w;
        constants[2].i = A.h;
        constants[3].i = A.d;
        constants[4].i = A.c;
        constants[5].i = A.cstep;
        constants[6].i = B2.dims;
        constants[7].i = B2.w;
        constants[8].i = B2.h;
        constants[9].i = B2.d;
        constants[10].i = B2.c;
        constants[11].i = B2.cstep;
        constants[12].i = top_blob.dims;
        constants[13].i = top_blob.w;
        constants[14].i = top_blob.h;
        constants[15].i = top_blob.d;
        constants[16].i = top_blob.c;
        constants[17].i = top_blob.cstep;

        const Pipeline* pipeline = 0;
        if (B2.elempack == 1 && top_blob.elempack == 1)
        {
            pipeline = pipeline_binaryop_broadcast[0];
        }
        if (B2.elempack == 4 && top_blob.elempack == 4)
        {
            pipeline = pipeline_binaryop_broadcast_pack4[0];
        }
        if (B2.elempack == 1 && top_blob.elempack == 4)
        {
            pipeline = pipeline_binaryop_broadcast_pack1to4[0];
        }

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    }

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

    const Pipeline* pipeline = elempack == 4 ? pipeline_binaryop_pack4 : pipeline_binaryop;

    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

    return 0;
}

} // namespace ncnn

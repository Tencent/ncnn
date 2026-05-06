// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "binaryop_x86.h"

#include "cpu.h"
#include "layer.h"
#include "layer_type.h"
#include "mat.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

namespace BinaryOp_x86_functor {

#include "binaryop_functor.h"

} // namespace BinaryOp_x86_functor

#include "binaryop_bf16s.h"

void binary_op_vector_bf16s_avx512bf16(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int aw, int bw, int ap, int bp, int op_type)
{
    using namespace BinaryOp_x86_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_vector_bf16s<binary_op_add>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_vector_bf16s<binary_op_sub>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_vector_bf16s<binary_op_mul>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_vector_bf16s<binary_op_div>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_vector_bf16s<binary_op_max>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_vector_bf16s<binary_op_min>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_POW) return binary_op_vector_bf16s<binary_op_pow>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_vector_bf16s<binary_op_rsub>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_vector_bf16s<binary_op_rdiv>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_vector_bf16s<binary_op_rpow>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_vector_bf16s<binary_op_atan2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_vector_bf16s<binary_op_ratan2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_FMOD) return binary_op_vector_bf16s<binary_op_fmod>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RFMOD) return binary_op_vector_bf16s<binary_op_rfmod>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_LOGADDEXP) return binary_op_vector_bf16s<binary_op_logaddexp>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_FLOOR_DIVIDE) return binary_op_vector_bf16s<binary_op_floor_divide>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RFLOOR_DIVIDE) return binary_op_vector_bf16s<binary_op_rfloor_divide>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_REMAINDER) return binary_op_vector_bf16s<binary_op_remainder>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RREMAINDER) return binary_op_vector_bf16s<binary_op_rremainder>(ptr, ptr1, outptr, aw, bw, ap, bp);

    // should never reach here
}

} // namespace ncnn

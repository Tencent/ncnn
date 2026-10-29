// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "unaryop_riscv.h"

#include <float.h>
#include <fenv.h>

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

#include "cpu.h"
#include "riscv_usability.h"

namespace ncnn {

#if __riscv_vector
#include "unaryop_functor.h"
#endif

#if NCNN_BF16
#include "unaryop_bf16s.h"
#endif

UnaryOp_riscv::UnaryOp_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
#if NCNN_ZFH
#if __riscv_vector
    support_fp16_storage = cpu_support_riscv_zvfh();
#else
    support_fp16_storage = cpu_support_riscv_zfh();
#endif
#endif
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

#if __riscv_vector
template<typename Op>
static int unary_op_inplace(Mat& a, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int size = w * h * d;
    int elempack = a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

        int n = size * elempack;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);

            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            _p = op(_p, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);

            ptr += vl;
            n -= vl;
        }
    }

    return 0;
}

#endif // __riscv_vector

int UnaryOp_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_ZFH || NCNN_BF16
    int elembits = bottom_top_blob.elembits();
#endif

#if NCNN_ZFH
    if (opt.use_fp16_storage && elembits == 16)
        return forward_inplace_fp16s(bottom_top_blob, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

#if __riscv_vector
    using namespace unaryop_functor;

    if (op_type == Operation_ABS)
        return unary_op_inplace<unary_op_abs>(bottom_top_blob, opt);

    if (op_type == Operation_NEG)
        return unary_op_inplace<unary_op_neg>(bottom_top_blob, opt);

    if (op_type == Operation_FLOOR)
        return unary_op_inplace<unary_op_floor>(bottom_top_blob, opt);

    if (op_type == Operation_CEIL)
        return unary_op_inplace<unary_op_ceil>(bottom_top_blob, opt);

    if (op_type == Operation_SQUARE)
        return unary_op_inplace<unary_op_square>(bottom_top_blob, opt);

    if (op_type == Operation_SQRT)
        return unary_op_inplace<unary_op_sqrt>(bottom_top_blob, opt);

    if (op_type == Operation_RSQRT)
        return unary_op_inplace<unary_op_rsqrt>(bottom_top_blob, opt);

    if (op_type == Operation_EXP)
        return unary_op_inplace<unary_op_exp>(bottom_top_blob, opt);

    if (op_type == Operation_LOG)
        return unary_op_inplace<unary_op_log>(bottom_top_blob, opt);

    if (op_type == Operation_SIN)
        return unary_op_inplace<unary_op_sin>(bottom_top_blob, opt);

    if (op_type == Operation_COS)
        return unary_op_inplace<unary_op_cos>(bottom_top_blob, opt);

    if (op_type == Operation_TAN)
        return unary_op_inplace<unary_op_tan>(bottom_top_blob, opt);

    if (op_type == Operation_ASIN)
        return unary_op_inplace<unary_op_asin>(bottom_top_blob, opt);

    if (op_type == Operation_ACOS)
        return unary_op_inplace<unary_op_acos>(bottom_top_blob, opt);

    if (op_type == Operation_ATAN)
        return unary_op_inplace<unary_op_atan>(bottom_top_blob, opt);

    if (op_type == Operation_RECIPROCAL)
        return unary_op_inplace<unary_op_reciprocal>(bottom_top_blob, opt);

    if (op_type == Operation_TANH)
        return unary_op_inplace<unary_op_tanh>(bottom_top_blob, opt);

    if (op_type == Operation_LOG10)
        return unary_op_inplace<unary_op_log10>(bottom_top_blob, opt);

    if (op_type == Operation_ROUND)
    {
        // round to nearest even
#ifdef FE_TONEAREST
        int old_rm = fegetround();
        fesetround(FE_TONEAREST);
#endif
        int ret = unary_op_inplace<unary_op_round>(bottom_top_blob, opt);
#ifdef FE_TONEAREST
        fesetround(old_rm);
#endif
        return ret;
    }

    if (op_type == Operation_TRUNC)
        return unary_op_inplace<unary_op_trunc>(bottom_top_blob, opt);

    if (op_type == Operation_SIGN)
        return unary_op_inplace<unary_op_sign>(bottom_top_blob, opt);

    if (op_type == Operation_EXPM1)
        return unary_op_inplace<unary_op_expm1>(bottom_top_blob, opt);

    if (op_type == Operation_SINH)
        return unary_op_inplace<unary_op_sinh>(bottom_top_blob, opt);

    if (op_type == Operation_ASINH)
        return unary_op_inplace<unary_op_asinh>(bottom_top_blob, opt);

    if (op_type == Operation_COSH)
        return unary_op_inplace<unary_op_cosh>(bottom_top_blob, opt);

    if (op_type == Operation_ACOSH)
        return unary_op_inplace<unary_op_acosh>(bottom_top_blob, opt);

    if (op_type == Operation_ATANH)
        return unary_op_inplace<unary_op_atanh>(bottom_top_blob, opt);

    if (op_type == Operation_LOG1P)
        return unary_op_inplace<unary_op_log1p>(bottom_top_blob, opt);

    return 0;
#else  // __riscv_vector
    return UnaryOp::forward_inplace(bottom_top_blob, opt);
#endif // __riscv_vector
}

#if NCNN_BF16
int UnaryOp_riscv::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    return unaryop_bf16s(bottom_top_blob, op_type, opt);
}
#endif // NCNN_BF16

} // namespace ncnn

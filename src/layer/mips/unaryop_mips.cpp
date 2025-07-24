// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "unaryop_mips.h"

// #include <fenv.h>
#include <float.h>

#if __mips_msa
#include <msa.h>
#include "msa_mathfun.h"
#endif // __mips_msa

namespace ncnn {

UnaryOp_mips::UnaryOp_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
}

template<typename Op>
static int unary_op_inplace(Mat& a, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

        int i = 0;
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            _p = op.func_pack4(_p);
            __msa_st_w((v4i32)_p, ptr, 0);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *ptr = op.func(*ptr);
            ptr++;
        }
    }

    return 0;
}

namespace UnaryOp_mips_functor {

struct unary_op_abs
{
    float func(const float& x) const
    {
        return (float)fabsf(x);
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        return (v4f32)__msa_bclri_w((v4u32)x, 31);
    }
#endif // __mips_msa
};

struct unary_op_neg
{
    float func(const float& x) const
    {
        return -x;
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        return (v4f32)__msa_bnegi_w((v4u32)x, 31);
    }
#endif // __mips_msa
};

struct unary_op_floor
{
    float func(const float& x) const
    {
        return (float)floorf(x);
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        v4i32 _xi = __msa_ftrunc_s_w(x);
        v4i32 _mask = __msa_fclt_w(x, __msa_ffint_s_w(_xi));
        return __msa_ffint_s_w(__msa_addv_w(_xi, _mask));
        // int old_msacsr = __msa_cfcmsa_msacsr();
        // __msa_ctcmsa_msacsr(old_msacsr | 3); // round towards -inf
        // v4f32 y = __msa_frint_w(x);
        // __msa_ctcmsa_msacsr(old_msacsr);
        // return y;
    }
#endif // __mips_msa
};

struct unary_op_ceil
{
    float func(const float& x) const
    {
        return (float)ceilf(x);
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        v4i32 _xi = __msa_ftrunc_s_w(x);
        v4i32 _mask = __msa_fclt_w(__msa_ffint_s_w(_xi), x);
        return __msa_ffint_s_w(__msa_subv_w(_xi, _mask));
        // int old_msacsr = __msa_cfcmsa_msacsr();
        // __msa_ctcmsa_msacsr((old_msacsr | 3) ^ 1); // round towards +inf
        // v4f32 y = __msa_frint_w(x);
        // __msa_ctcmsa_msacsr(old_msacsr);
        // return y;
    }
#endif // __mips_msa
};

struct unary_op_square
{
    float func(const float& x) const
    {
        return x * x;
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        return __msa_fmul_w(x, x);
    }
#endif // __mips_msa
};

struct unary_op_sqrt
{
    float func(const float& x) const
    {
        return (float)sqrtf(x);
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        return __msa_fsqrt_w(x);
    }
#endif // __mips_msa
};

struct unary_op_rsqrt
{
    float func(const float& x) const
    {
        return (float)(1.f / sqrtf(x));
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        return __msa_frsqrt_w(x);
    }
#endif // __mips_msa
};

struct unary_op_exp
{
    float func(const float& x) const
    {
        return (float)expf(x);
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        return exp_ps(x);
    }
#endif // __mips_msa
};

struct unary_op_log
{
    float func(const float& x) const
    {
        return (float)logf(x);
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        return log_ps(x);
    }
#endif // __mips_msa
};

struct unary_op_sin
{
    float func(const float& x) const
    {
        return (float)sinf(x);
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __msa_st_w((v4i32)x, tmp, 0);
        tmp[0] = sinf(tmp[0]);
        tmp[1] = sinf(tmp[1]);
        tmp[2] = sinf(tmp[2]);
        tmp[3] = sinf(tmp[3]);
        return (v4f32)__msa_ld_w(tmp, 0);
    }
#endif // __mips_msa
};

struct unary_op_cos
{
    float func(const float& x) const
    {
        return (float)cosf(x);
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __msa_st_w((v4i32)x, tmp, 0);
        tmp[0] = cosf(tmp[0]);
        tmp[1] = cosf(tmp[1]);
        tmp[2] = cosf(tmp[2]);
        tmp[3] = cosf(tmp[3]);
        return (v4f32)__msa_ld_w(tmp, 0);
    }
#endif // __mips_msa
};

struct unary_op_tan
{
    float func(const float& x) const
    {
        return (float)tanf(x);
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __msa_st_w((v4i32)x, tmp, 0);
        tmp[0] = tanf(tmp[0]);
        tmp[1] = tanf(tmp[1]);
        tmp[2] = tanf(tmp[2]);
        tmp[3] = tanf(tmp[3]);
        return (v4f32)__msa_ld_w(tmp, 0);
    }
#endif // __mips_msa
};

struct unary_op_asin
{
    float func(const float& x) const
    {
        return (float)asinf(x);
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __msa_st_w((v4i32)x, tmp, 0);
        tmp[0] = asinf(tmp[0]);
        tmp[1] = asinf(tmp[1]);
        tmp[2] = asinf(tmp[2]);
        tmp[3] = asinf(tmp[3]);
        return (v4f32)__msa_ld_w(tmp, 0);
    }
#endif // __mips_msa
};

struct unary_op_acos
{
    float func(const float& x) const
    {
        return (float)acosf(x);
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __msa_st_w((v4i32)x, tmp, 0);
        tmp[0] = acosf(tmp[0]);
        tmp[1] = acosf(tmp[1]);
        tmp[2] = acosf(tmp[2]);
        tmp[3] = acosf(tmp[3]);
        return (v4f32)__msa_ld_w(tmp, 0);
    }
#endif // __mips_msa
};

struct unary_op_atan
{
    float func(const float& x) const
    {
        return (float)atanf(x);
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __msa_st_w((v4i32)x, tmp, 0);
        tmp[0] = atanf(tmp[0]);
        tmp[1] = atanf(tmp[1]);
        tmp[2] = atanf(tmp[2]);
        tmp[3] = atanf(tmp[3]);
        return (v4f32)__msa_ld_w(tmp, 0);
    }
#endif // __mips_msa
};

struct unary_op_reciprocal
{
    float func(const float& x) const
    {
        return 1.f / x;
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        return __msa_frcp_w(x);
    }
#endif // __mips_msa
};

struct unary_op_tanh
{
    float func(const float& x) const
    {
        return (float)tanhf(x);
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        return tanh_ps(x);
    }
#endif // __mips_msa
};

struct unary_op_log10
{
    float func(const float& x) const
    {
        return (float)log10f(x);
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        return __msa_fmul_w(log_ps(x), __msa_fill_w_f32(0.434294481903));
    }
#endif // __mips_msa
};

struct unary_op_round
{
    float func(const float& x) const
    {
        // round to nearest even
#if NCNN_GNU_INLINE_ASM
        // return (x + 12582912.f) - 12582912.f;
        float y;
        const float magic = 12582912.f;
        asm volatile(
            "add.s   %0, %1, %2  \n"
            "sub.s   %0, %0, %2  \n"
            : "=f"(y)
            : "f"(x), "f"(magic)
            :);
        return y;
#else
#ifdef FE_TONEAREST
        int old_rm = fegetround();
        fesetround(FE_TONEAREST);
#endif
        float y = nearbyintf(x);
#ifdef FE_TONEAREST
        fesetround(old_rm);
#endif
        return y;
#endif
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        // round towards nearest even by default
        return __msa_frint_w(x);
    }
#endif // __mips_msa
};

struct unary_op_trunc
{
    float func(const float& x) const
    {
        return (float)truncf(x);
    }
#if __mips_msa
    v4f32 func_pack4(const v4f32& x) const
    {
        return __msa_ffint_s_w(__msa_ftrunc_s_w(x));
        // int old_msacsr = __msa_cfcmsa_msacsr();
        // __msa_ctcmsa_msacsr((old_msacsr | 3) ^ 2); // round towards zero
        // v4f32 y = __msa_frint_w(x);
        // __msa_ctcmsa_msacsr(old_msacsr);
        // return y;
    }
#endif // __mips_msa
};

} // namespace UnaryOp_mips_functor

int UnaryOp_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    using namespace UnaryOp_mips_functor;

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
        return unary_op_inplace<unary_op_round>(bottom_top_blob, opt);

    if (op_type == Operation_TRUNC)
        return unary_op_inplace<unary_op_trunc>(bottom_top_blob, opt);

    return 0;
}

} // namespace ncnn

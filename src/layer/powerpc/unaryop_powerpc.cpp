// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "unaryop_powerpc.h"

#include <math.h>

#include "platform.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#include "powerpc_mathfun.h"
#endif // __ALTIVEC__

namespace ncnn {

UnaryOp_powerpc::UnaryOp_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
}

#if __ALTIVEC__

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
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = __ncnn_altivec_load(ptr);
            _p = op.func_pack4(_p);
            __ncnn_altivec_store(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            *ptr = op.func(*ptr);
            ptr++;
        }
    }

    return 0;
}

namespace UnaryOp_powerpc_functor {

struct unary_op_abs
{
    float func(const float& x) const
    {
        return (float)fabsf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return abs_ps(x);
    }
};

struct unary_op_neg
{
    float func(const float& x) const
    {
        return -x;
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return __ncnn_altivec_neg(x);
    }
};

struct unary_op_floor
{
    float func(const float& x) const
    {
        return (float)floorf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return __ncnn_altivec_floor(x);
    }
};

struct unary_op_ceil
{
    float func(const float& x) const
    {
        return (float)ceilf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return __ncnn_altivec_ceil(x);
    }
};

struct unary_op_square
{
    float func(const float& x) const
    {
        return x * x;
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return __ncnn_altivec_mul(x, x);
    }
};

struct unary_op_sqrt
{
    float func(const float& x) const
    {
        return (float)sqrtf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return __ncnn_altivec_sqrt2(x);
    }
};

struct unary_op_rsqrt
{
    float func(const float& x) const
    {
        return (float)(1.f / sqrtf(x));
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return __ncnn_altivec_rsqrt(x);
    }
};

struct unary_op_exp
{
    float func(const float& x) const
    {
        return (float)expf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return exp_ps(x);
    }
};

struct unary_op_log
{
    float func(const float& x) const
    {
        return (float)logf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return log_ps(x);
    }
};

struct unary_op_sin
{
    float func(const float& x) const
    {
        return (float)sinf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return sin_ps(x);
    }
};

struct unary_op_cos
{
    float func(const float& x) const
    {
        return (float)cosf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return cos_ps(x);
    }
};

struct unary_op_tan
{
    float func(const float& x) const
    {
        return (float)tanf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return tan_ps(x);
    }
};

struct unary_op_asin
{
    float func(const float& x) const
    {
        return (float)asinf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return asin_ps(x);
    }
};

struct unary_op_acos
{
    float func(const float& x) const
    {
        return (float)acosf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return acos_ps(x);
    }
};

struct unary_op_atan
{
    float func(const float& x) const
    {
        return (float)atanf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return atan_ps(x);
    }
};

struct unary_op_reciprocal
{
    float func(const float& x) const
    {
        return 1.f / x;
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return __ncnn_altivec_recip2(x);
    }
};

struct unary_op_tanh
{
    float func(const float& x) const
    {
        return (float)tanhf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return tanh_ps(x);
    }
};

struct unary_op_log10
{
    float func(const float& x) const
    {
        return (float)log10f(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return __ncnn_altivec_mul(log_ps(x), __ncnn_altivec_fill_f32(0.434294481903f));
    }
};

struct unary_op_round
{
    float func(const float& x) const
    {
        // ties to even, matching nearbyintf in the generic implementation
        return nearbyintf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return __ncnn_altivec_round(x);
    }
};

struct unary_op_trunc
{
    float func(const float& x) const
    {
        return (float)truncf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return __ncnn_altivec_trunc(x);
    }
};

struct unary_op_sign
{
    float func(const float& x) const
    {
        return x > 0.f ? 1.f : x < 0.f ? -1.f : 0.f;
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        v4f32 _zero = vec_splats(0.f);
        v4f32 _one = vec_splats(1.f);
        v4f32 _negone = vec_splats(-1.f);
        v4i32 _posmask = __ncnn_altivec_cmpgt(x, _zero);
        v4i32 _negmask = __ncnn_altivec_cmpgt(_zero, x);
        v4f32 _sign = __ncnn_altivec_sel(_posmask, _one, _zero);
        return __ncnn_altivec_sel(_negmask, _negone, _sign);
    }
};

struct unary_op_expm1
{
    float func(const float& x) const
    {
        return (float)expm1f(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return expm1_ps(x);
    }
};

struct unary_op_sinh
{
    float func(const float& x) const
    {
        return (float)sinhf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return sinh_ps(x);
    }
};

struct unary_op_asinh
{
    float func(const float& x) const
    {
        return (float)asinhf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return asinh_ps(x);
    }
};

struct unary_op_cosh
{
    float func(const float& x) const
    {
        return (float)coshf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return cosh_ps(x);
    }
};

struct unary_op_acosh
{
    float func(const float& x) const
    {
        return (float)acoshf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return acosh_ps(x);
    }
};

struct unary_op_atanh
{
    float func(const float& x) const
    {
        return (float)atanhf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return atanh_ps(x);
    }
};

struct unary_op_log1p
{
    float func(const float& x) const
    {
        return (float)log1pf(x);
    }
    v4f32 func_pack4(const v4f32& x) const
    {
        return log1p_ps(x);
    }
};

} // namespace UnaryOp_powerpc_functor

#endif // __ALTIVEC__

int UnaryOp_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if __ALTIVEC__
    if (bottom_top_blob.elembits() == 32)
    {
        using namespace UnaryOp_powerpc_functor;

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
    }
#endif // __ALTIVEC__

    // unsupported path, delegate to the generic implementation
    if (bottom_top_blob.elempack != 1)
    {
        const int elempack = bottom_top_blob.elempack;

        Mat bottom_top_blob_unpacked;
        convert_packing(bottom_top_blob, bottom_top_blob_unpacked, 1, opt);
        if (bottom_top_blob_unpacked.empty())
            return -100;

        int ret = UnaryOp::forward_inplace(bottom_top_blob_unpacked, opt);
        if (ret != 0)
            return ret;

        Mat bottom_top_blob_packed;
        convert_packing(bottom_top_blob_unpacked, bottom_top_blob_packed, elempack, opt);
        if (bottom_top_blob_packed.empty())
            return -100;

        bottom_top_blob = bottom_top_blob_packed;

        return ret;
    }

    return UnaryOp::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn

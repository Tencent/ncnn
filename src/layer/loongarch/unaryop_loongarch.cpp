// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "unaryop_loongarch.h"

// #include <fenv.h>
#include <float.h>

#if __loongarch_sx
#include <lsxintrin.h>
#include "lsx_mathfun.h"
#if __loongarch_asx
#include <lasxintrin.h>
#include "lasx_mathfun.h"
#endif // __loongarch_asx
#endif // __loongarch_sx

namespace ncnn {

UnaryOp_loongarch::UnaryOp_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
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
#if __loongarch_sx
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __builtin_prefetch(ptr + 32);
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            _p = op.func_pack8(_p);
            __lasx_xvst((__m256i)_p, ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            _p = op.func_pack4(_p);
            __lsx_vst(_p, ptr, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            *ptr = op.func(*ptr);
            ptr++;
        }
    }

    return 0;
}

namespace UnaryOp_loongarch_functor {

struct unary_op_abs
{
    float func(const float& x) const
    {
        return (float)fabsf(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return (__m128)__lsx_vbitclri_w((__m128i)x, 31);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        return (__m256)__lasx_xvbitclri_w((__m256i)x, 31);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

struct unary_op_neg
{
    float func(const float& x) const
    {
        return -x;
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return (__m128)__lsx_vbitrevi_w((__m128i)x, 31);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        return (__m256)__lasx_xvbitrevi_w((__m256i)x, 31);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

struct unary_op_floor
{
    float func(const float& x) const
    {
        return (float)floorf(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return (__m128)__lsx_vfrintrm_s(x);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        return (__m256)__lasx_xvfrintrm_s(x);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

struct unary_op_ceil
{
    float func(const float& x) const
    {
        return (float)ceilf(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return (__m128)__lsx_vfrintrp_s(x);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        return (__m256)__lasx_xvfrintrp_s(x);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

struct unary_op_square
{
    float func(const float& x) const
    {
        return x * x;
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return __lsx_vfmul_s(x, x);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        return __lasx_xvfmul_s(x, x);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

struct unary_op_sqrt
{
    float func(const float& x) const
    {
        return (float)sqrtf(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return __lsx_vfsqrt_s(x);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        return __lasx_xvfsqrt_s(x);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

struct unary_op_rsqrt
{
    float func(const float& x) const
    {
        return (float)(1.f / sqrtf(x));
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return __lsx_vfrsqrt_s(x);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        return __lasx_xvfrsqrt_s(x);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

struct unary_op_exp
{
    float func(const float& x) const
    {
        return (float)expf(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return exp_ps(x);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        return exp256_ps(x);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

struct unary_op_log
{
    float func(const float& x) const
    {
        return (float)logf(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return log_ps(x);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        return log256_ps(x);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

struct unary_op_sin
{
    float func(const float& x) const
    {
        return (float)sinf(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        float tmp[4];
        __lsx_vst(x, tmp, 0);
        tmp[0] = sinf(tmp[0]);
        tmp[1] = sinf(tmp[1]);
        tmp[2] = sinf(tmp[2]);
        tmp[3] = sinf(tmp[3]);
        return (__m128)__lsx_vld(tmp, 0);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        float tmp[8];
        __lasx_xvst(x, tmp, 0);
        tmp[0] = sinf(tmp[0]);
        tmp[1] = sinf(tmp[1]);
        tmp[2] = sinf(tmp[2]);
        tmp[3] = sinf(tmp[3]);
        tmp[4] = sinf(tmp[4]);
        tmp[5] = sinf(tmp[5]);
        tmp[6] = sinf(tmp[6]);
        tmp[7] = sinf(tmp[7]);
        return (__m256)__lasx_xvld(tmp, 0);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

struct unary_op_cos
{
    float func(const float& x) const
    {
        return (float)cosf(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        float tmp[4];
        __lsx_vst(x, tmp, 0);
        tmp[0] = cosf(tmp[0]);
        tmp[1] = cosf(tmp[1]);
        tmp[2] = cosf(tmp[2]);
        tmp[3] = cosf(tmp[3]);
        return (__m128)__lsx_vld(tmp, 0);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        float tmp[8];
        __lasx_xvst(x, tmp, 0);
        tmp[0] = cosf(tmp[0]);
        tmp[1] = cosf(tmp[1]);
        tmp[2] = cosf(tmp[2]);
        tmp[3] = cosf(tmp[3]);
        tmp[4] = cosf(tmp[4]);
        tmp[5] = cosf(tmp[5]);
        tmp[6] = cosf(tmp[6]);
        tmp[7] = cosf(tmp[7]);
        return (__m256)__lasx_xvld(tmp, 0);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

struct unary_op_tan
{
    float func(const float& x) const
    {
        return (float)tanf(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        float tmp[4];
        __lsx_vst(x, tmp, 0);
        tmp[0] = tanf(tmp[0]);
        tmp[1] = tanf(tmp[1]);
        tmp[2] = tanf(tmp[2]);
        tmp[3] = tanf(tmp[3]);
        return (__m128)__lsx_vld(tmp, 0);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        float tmp[8];
        __lasx_xvst(x, tmp, 0);
        tmp[0] = tanf(tmp[0]);
        tmp[1] = tanf(tmp[1]);
        tmp[2] = tanf(tmp[2]);
        tmp[3] = tanf(tmp[3]);
        tmp[4] = tanf(tmp[4]);
        tmp[5] = tanf(tmp[5]);
        tmp[6] = tanf(tmp[6]);
        tmp[7] = tanf(tmp[7]);
        return (__m256)__lasx_xvld(tmp, 0);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

struct unary_op_asin
{
    float func(const float& x) const
    {
        return (float)asinf(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        float tmp[4];
        __lsx_vst(x, tmp, 0);
        tmp[0] = asinf(tmp[0]);
        tmp[1] = asinf(tmp[1]);
        tmp[2] = asinf(tmp[2]);
        tmp[3] = asinf(tmp[3]);
        return (__m128)__lsx_vld(tmp, 0);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        float tmp[8];
        __lasx_xvst(x, tmp, 0);
        tmp[0] = asinf(tmp[0]);
        tmp[1] = asinf(tmp[1]);
        tmp[2] = asinf(tmp[2]);
        tmp[3] = asinf(tmp[3]);
        tmp[4] = asinf(tmp[4]);
        tmp[5] = asinf(tmp[5]);
        tmp[6] = asinf(tmp[6]);
        tmp[7] = asinf(tmp[7]);
        return (__m256)__lasx_xvld(tmp, 0);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

struct unary_op_acos
{
    float func(const float& x) const
    {
        return (float)acosf(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        float tmp[4];
        __lsx_vst(x, tmp, 0);
        tmp[0] = acosf(tmp[0]);
        tmp[1] = acosf(tmp[1]);
        tmp[2] = acosf(tmp[2]);
        tmp[3] = acosf(tmp[3]);
        return (__m128)__lsx_vld(tmp, 0);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        float tmp[8];
        __lasx_xvst(x, tmp, 0);
        tmp[0] = acosf(tmp[0]);
        tmp[1] = acosf(tmp[1]);
        tmp[2] = acosf(tmp[2]);
        tmp[3] = acosf(tmp[3]);
        tmp[4] = acosf(tmp[4]);
        tmp[5] = acosf(tmp[5]);
        tmp[6] = acosf(tmp[6]);
        tmp[7] = acosf(tmp[7]);
        return (__m256)__lasx_xvld(tmp, 0);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

struct unary_op_atan
{
    float func(const float& x) const
    {
        return (float)atanf(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        float tmp[4];
        __lsx_vst(x, tmp, 0);
        tmp[0] = atanf(tmp[0]);
        tmp[1] = atanf(tmp[1]);
        tmp[2] = atanf(tmp[2]);
        tmp[3] = atanf(tmp[3]);
        return (__m128)__lsx_vld(tmp, 0);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        float tmp[8];
        __lasx_xvst(x, tmp, 0);
        tmp[0] = atanf(tmp[0]);
        tmp[1] = atanf(tmp[1]);
        tmp[2] = atanf(tmp[2]);
        tmp[3] = atanf(tmp[3]);
        tmp[4] = atanf(tmp[4]);
        tmp[5] = atanf(tmp[5]);
        tmp[6] = atanf(tmp[6]);
        tmp[7] = atanf(tmp[7]);
        return (__m256)__lasx_xvld(tmp, 0);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

struct unary_op_reciprocal
{
    float func(const float& x) const
    {
        return 1.f / x;
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return __lsx_vfrecip_s(x);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        return __lasx_xvfrecip_s(x);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

struct unary_op_tanh
{
    float func(const float& x) const
    {
        return (float)tanhf(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return tanh_ps(x);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        return tanh256_ps(x);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

struct unary_op_log10
{
    float func(const float& x) const
    {
        return (float)log10f(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return __lsx_vfmul_s(log_ps(x), __lsx_vreplfr2vr_s(0.434294481903));
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        return __lasx_xvfmul_s(log256_ps(x), __lasx_xvreplfr2vr_s(0.434294481903));
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
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
            "fadd.s     %0, %1, %2  \n"
            "fsub.s     %0, %0, %2  \n"
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
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return (__m128)__lsx_vfrintrne_s(x);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        return (__m256)__lasx_xvfrintrne_s(x);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

struct unary_op_trunc
{
    float func(const float& x) const
    {
        return (float)truncf(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return (__m128)__lsx_vfrintrz_s(x);
    }
#if __loongarch_asx
    __m256 func_pack8(const __m256& x) const
    {
        return (__m256)__lasx_xvfrintrz_s(x);
    }
#endif // __loongarch_asx
#endif // __loongarch_sx
};

} // namespace UnaryOp_loongarch_functor

#if NCNN_BF16
#include "loongarch_usability.h"

template<typename Op>
static int unary_op_inplace_bf16s(Mat& a, const Option& opt)
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
        unsigned short* ptr = a.channel(q);

        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
            _p = op.func_pack8(_p);
            __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx(ptr);
            _p = op.func_pack4(_p);
            __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            *ptr = float32_to_bfloat16(op.func(bfloat16_to_float32(*ptr)));
            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

int UnaryOp_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    using namespace UnaryOp_loongarch_functor;

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

#if NCNN_BF16
int UnaryOp_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    using namespace UnaryOp_loongarch_functor;

    if (op_type == Operation_ABS)
        return unary_op_inplace_bf16s<unary_op_abs>(bottom_top_blob, opt);

    if (op_type == Operation_NEG)
        return unary_op_inplace_bf16s<unary_op_neg>(bottom_top_blob, opt);

    if (op_type == Operation_FLOOR)
        return unary_op_inplace_bf16s<unary_op_floor>(bottom_top_blob, opt);

    if (op_type == Operation_CEIL)
        return unary_op_inplace_bf16s<unary_op_ceil>(bottom_top_blob, opt);

    if (op_type == Operation_SQUARE)
        return unary_op_inplace_bf16s<unary_op_square>(bottom_top_blob, opt);

    if (op_type == Operation_SQRT)
        return unary_op_inplace_bf16s<unary_op_sqrt>(bottom_top_blob, opt);

    if (op_type == Operation_RSQRT)
        return unary_op_inplace_bf16s<unary_op_rsqrt>(bottom_top_blob, opt);

    if (op_type == Operation_EXP)
        return unary_op_inplace_bf16s<unary_op_exp>(bottom_top_blob, opt);

    if (op_type == Operation_LOG)
        return unary_op_inplace_bf16s<unary_op_log>(bottom_top_blob, opt);

    if (op_type == Operation_SIN)
        return unary_op_inplace_bf16s<unary_op_sin>(bottom_top_blob, opt);

    if (op_type == Operation_COS)
        return unary_op_inplace_bf16s<unary_op_cos>(bottom_top_blob, opt);

    if (op_type == Operation_TAN)
        return unary_op_inplace_bf16s<unary_op_tan>(bottom_top_blob, opt);

    if (op_type == Operation_ASIN)
        return unary_op_inplace_bf16s<unary_op_asin>(bottom_top_blob, opt);

    if (op_type == Operation_ACOS)
        return unary_op_inplace_bf16s<unary_op_acos>(bottom_top_blob, opt);

    if (op_type == Operation_ATAN)
        return unary_op_inplace_bf16s<unary_op_atan>(bottom_top_blob, opt);

    if (op_type == Operation_RECIPROCAL)
        return unary_op_inplace_bf16s<unary_op_reciprocal>(bottom_top_blob, opt);

    if (op_type == Operation_TANH)
        return unary_op_inplace_bf16s<unary_op_tanh>(bottom_top_blob, opt);

    if (op_type == Operation_LOG10)
        return unary_op_inplace_bf16s<unary_op_log10>(bottom_top_blob, opt);

    if (op_type == Operation_ROUND)
        return unary_op_inplace_bf16s<unary_op_round>(bottom_top_blob, opt);

    if (op_type == Operation_TRUNC)
        return unary_op_inplace_bf16s<unary_op_trunc>(bottom_top_blob, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn

// yala is pleased to support the open source community by making ncnn available.
//
//
// Copyright (C) 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "unaryop_loongarch.h"

// #include <fenv.h>
#include <float.h>

#if __loongarch_sx
#include <lsxintrin.h>
#include "lsx_mathfun.h"
#endif // __loongarch_sx

namespace ncnn {

UnaryOp_loongarch::UnaryOp_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
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
        return (float)fabs(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return (__m128)__lsx_vbitclri_w((__m128i)x, 31);
    }
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
#endif // __loongarch_sx
};

struct unary_op_floor
{
    float func(const float& x) const
    {
        return (float)floor(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return (__m128)__lsx_vfrintrm_s(x);
    }
#endif // __loongarch_sx
};

struct unary_op_ceil
{
    float func(const float& x) const
    {
        return (float)ceil(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return (__m128)__lsx_vfrintrp_s(x);
    }
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
#endif // __loongarch_sx
};

struct unary_op_sqrt
{
    float func(const float& x) const
    {
        return (float)sqrt(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return __lsx_vfsqrt_s(x);
    }
#endif // __loongarch_sx
};

struct unary_op_rsqrt
{
    float func(const float& x) const
    {
        return (float)(1.f / sqrt(x));
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return __lsx_vfrsqrt_s(x);
    }
#endif // __loongarch_sx
};

struct unary_op_exp
{
    float func(const float& x) const
    {
        return (float)exp(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return exp_ps(x);
    }
#endif // __loongarch_sx
};

struct unary_op_log
{
    float func(const float& x) const
    {
        return (float)log(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return log_ps(x);
    }
#endif // __loongarch_sx
};

struct unary_op_sin
{
    float func(const float& x) const
    {
        return (float)sin(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __lsx_vst(x, tmp, 0);
        tmp[0] = sin(tmp[0]);
        tmp[1] = sin(tmp[1]);
        tmp[2] = sin(tmp[2]);
        tmp[3] = sin(tmp[3]);
        return (__m128)__lsx_vld(tmp, 0);
    }
#endif // __loongarch_sx
};

struct unary_op_cos
{
    float func(const float& x) const
    {
        return (float)cos(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __lsx_vst(x, tmp, 0);
        tmp[0] = cos(tmp[0]);
        tmp[1] = cos(tmp[1]);
        tmp[2] = cos(tmp[2]);
        tmp[3] = cos(tmp[3]);
        return (__m128)__lsx_vld(tmp, 0);
    }
#endif // __loongarch_sx
};

struct unary_op_tan
{
    float func(const float& x) const
    {
        return (float)tan(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __lsx_vst(x, tmp, 0);
        tmp[0] = tan(tmp[0]);
        tmp[1] = tan(tmp[1]);
        tmp[2] = tan(tmp[2]);
        tmp[3] = tan(tmp[3]);
        return (__m128)__lsx_vld(tmp, 0);
    }
#endif // __loongarch_sx
};

struct unary_op_asin
{
    float func(const float& x) const
    {
        return (float)asin(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __lsx_vst(x, tmp, 0);
        tmp[0] = asin(tmp[0]);
        tmp[1] = asin(tmp[1]);
        tmp[2] = asin(tmp[2]);
        tmp[3] = asin(tmp[3]);
        return (__m128)__lsx_vld(tmp, 0);
    }
#endif // __loongarch_sx
};

struct unary_op_acos
{
    float func(const float& x) const
    {
        return (float)acos(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __lsx_vst(x, tmp, 0);
        tmp[0] = acos(tmp[0]);
        tmp[1] = acos(tmp[1]);
        tmp[2] = acos(tmp[2]);
        tmp[3] = acos(tmp[3]);
        return (__m128)__lsx_vld(tmp, 0);
    }
#endif // __loongarch_sx
};

struct unary_op_atan
{
    float func(const float& x) const
    {
        return (float)atan(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __lsx_vst(x, tmp, 0);
        tmp[0] = atan(tmp[0]);
        tmp[1] = atan(tmp[1]);
        tmp[2] = atan(tmp[2]);
        tmp[3] = atan(tmp[3]);
        return (__m128)__lsx_vld(tmp, 0);
    }
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
#endif // __loongarch_sx
};

struct unary_op_tanh
{
    float func(const float& x) const
    {
        return (float)tanh(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return tanh_ps(x);
    }
#endif // __loongarch_sx
};

struct unary_op_log10
{
    float func(const float& x) const
    {
        return (float)log10(x);
    }
#if __loongarch_sx
    __m128 func_pack4(const __m128& x) const
    {
        return __lsx_vfmul_s(log_ps(x), __lsx_vreplfr2vr_s(0.434294481903));
    }
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
#endif // __loongarch_sx
};

} // namespace UnaryOp_loongarch_functor

int UnaryOp_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
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

} // namespace ncnn

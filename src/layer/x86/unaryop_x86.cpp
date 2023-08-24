// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "unaryop_x86.h"

#include <fenv.h>
#include <float.h>
#include <math.h>

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __SSE4_1__
#include <smmintrin.h>
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE4_1__
#endif // __SSE2__
#include "x86_usability.h"
#include "x86_activation.h"

namespace ncnn {

UnaryOp_x86::UnaryOp_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
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
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _p = op.func_pack16(_p);
            _mm512_storeu_ps(ptr, _p);
            ptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _p = op.func_pack8(_p);
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_load_ps(ptr);
            _p = op.func_pack4(_p);
            _mm_store_ps(ptr, _p);
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = op.func(*ptr);
            ptr++;
        }
    }

    return 0;
}

namespace UnaryOp_x86_functor {
struct unary_op_abs
{
    float func(const float& x) const
    {
        return (float)fabsf(x);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
        return abs_ps(x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return abs256_ps(x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return abs512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_neg
{
    float func(const float& x) const
    {
        return -x;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
        return _mm_sub_ps(_mm_setzero_ps(), x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return _mm256_sub_ps(_mm256_setzero_ps(), x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return _mm512_sub_ps(_mm512_setzero_ps(), x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_floor
{
    float func(const float& x) const
    {
        return (float)floorf(x);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
        return floor_ps(x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return _mm256_floor_ps(x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return _mm512_roundscale_ps(x, _MM_FROUND_TO_NEG_INF);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_ceil
{
    float func(const float& x) const
    {
        return (float)ceilf(x);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
        return ceil_ps(x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return _mm256_ceil_ps(x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return _mm512_roundscale_ps(x, _MM_FROUND_TO_POS_INF);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_square
{
    float func(const float& x) const
    {
        return x * x;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
        return _mm_mul_ps(x, x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return _mm256_mul_ps(x, x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return _mm512_mul_ps(x, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_sqrt
{
    float func(const float& x) const
    {
        return (float)sqrtf(x);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
        return _mm_sqrt_ps(x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return _mm256_sqrt_ps(x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return _mm512_sqrt_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_rsqrt
{
    float func(const float& x) const
    {
        return 1.f / sqrtf(x);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
        return _mm_rsqrt_ps(x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return _mm256_rsqrt_ps(x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        __m256 _x0 = _mm512_extractf32x8_ps(x, 0);
        __m256 _x1 = _mm512_extractf32x8_ps(x, 1);
        _x0 = _mm256_rsqrt_ps(_x0);
        _x1 = _mm256_rsqrt_ps(_x1);
        return _mm512_insertf32x8(_mm512_castps256_ps512(_x0), _x1, 1);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_exp
{
    float func(const float& x) const
    {
        return (float)expf(x);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
        return exp_ps(x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return exp256_ps(x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return exp512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_log
{
    float func(const float& x) const
    {
        return (float)logf(x);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
        return log_ps(x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return log256_ps(x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return log512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_sin
{
    float func(const float& x) const
    {
        return (float)sinf(x);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
        return sin_ps(x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return sin256_ps(x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return sin512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_cos
{
    float func(const float& x) const
    {
        return (float)cosf(x);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
        return cos_ps(x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return cos256_ps(x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return cos512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_tan
{
    float func(const float& x) const
    {
        return (float)tanf(x);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
        return tan_ps(x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return tan256_ps(x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return tan512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_asin
{
    float func(const float& x) const
    {
        return (float)asinf(x);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
        return asin_ps(x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return asin256_ps(x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return asin512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_acos
{
    float func(const float& x) const
    {
        return (float)acosf(x);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
        return acos_ps(x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return acos256_ps(x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return acos512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_atan
{
    float func(const float& x) const
    {
        return (float)atanf(x);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
        return atan_ps(x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return atan256_ps(x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return atan512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_reciprocal
{
    float func(const float& x) const
    {
        return 1.f / x;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
        return _mm_div_ps(*(__m128*)_ps_1, x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return _mm256_div_ps(*(__m256*)_ps256_1, x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return _mm512_div_ps(*(__m512*)_ps512_1, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_tanh
{
    float func(const float& x) const
    {
        return (float)tanhf(x);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
        return tanh_sse(x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return tanh_avx(x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return tanh_avx512(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_log10
{
    float func(const float& x) const
    {
        return (float)log10f(x);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
        return _mm_mul_ps(log_ps(x), _mm_set1_ps(0.434294481903));
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return _mm256_mul_ps(log256_ps(x), _mm256_set1_ps(0.434294481903));
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return _mm512_mul_ps(log512_ps(x), _mm512_set1_ps(0.434294481903));
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_round
{
    float func(const float& x) const
    {
        // round to nearest even
        // return (x + 12582912.f) - 12582912.f;
#ifdef FE_TONEAREST
        int old_rm = fegetround();
        fesetround(FE_TONEAREST);
#endif
        float y = nearbyintf(x);
#ifdef FE_TONEAREST
        fesetround(old_rm);
#endif
        return y;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
#if __SSE4_1__
        return _mm_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
        return _mm_cvtepi32_ps(_mm_cvtps_epi32(x));
#endif
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_trunc
{
    float func(const float& x) const
    {
        return (float)truncf(x);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x) const
    {
#if __SSE4_1__
        return _mm_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
#else
        return _mm_cvtepi32_ps(_mm_cvttps_epi32(x));
#endif
    }
#if __AVX__
    __m256 func_pack8(const __m256& x) const
    {
        return _mm256_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x) const
    {
        return _mm512_roundscale_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

} // namespace UnaryOp_x86_functor

int UnaryOp_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    using namespace UnaryOp_x86_functor;
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

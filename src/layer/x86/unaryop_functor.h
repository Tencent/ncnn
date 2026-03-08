// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_UNARYOP_FUNCTOR_H
#define LAYER_UNARYOP_FUNCTOR_H

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

namespace UnaryOp_x86_functor {

struct unary_op_abs
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return (float)fabsf(x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
        return abs_ps(x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return abs256_ps(x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return abs512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_neg
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return -x;
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
        return _mm_sub_ps(_mm_setzero_ps(), x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return _mm256_sub_ps(_mm256_setzero_ps(), x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return _mm512_sub_ps(_mm512_setzero_ps(), x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_floor
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return (float)floorf(x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
        return floor_ps(x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return _mm256_floor_ps(x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return _mm512_roundscale_ps(x, _MM_FROUND_TO_NEG_INF);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_ceil
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return (float)ceilf(x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
        return ceil_ps(x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return _mm256_ceil_ps(x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return _mm512_roundscale_ps(x, _MM_FROUND_TO_POS_INF);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_square
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return x * x;
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
        return _mm_mul_ps(x, x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return _mm256_mul_ps(x, x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return _mm512_mul_ps(x, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_sqrt
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return (float)sqrtf(x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
        return _mm_sqrt_ps(x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return _mm256_sqrt_ps(x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return _mm512_sqrt_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_rsqrt
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return 1.f / sqrtf(x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
        return _mm_rsqrt_ps(x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return _mm256_rsqrt_ps(x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        __m256 _x0 = _mm512_extractf32x8_ps(x, 0);
        __m256 _x1 = _mm512_extractf32x8_ps(x, 1);
        _x0 = _mm256_rsqrt_ps(_x0);
        _x1 = _mm256_rsqrt_ps(_x1);
        return combine8x2_ps(_x0, _x1);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_exp
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return (float)expf(x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
        return exp_ps(x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return exp256_ps(x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return exp512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_log
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return (float)logf(x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
        return log_ps(x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return log256_ps(x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return log512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_sin
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return (float)sinf(x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
        return sin_ps(x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return sin256_ps(x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return sin512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_cos
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return (float)cosf(x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
        return cos_ps(x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return cos256_ps(x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return cos512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_tan
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return (float)tanf(x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
        return tan_ps(x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return tan256_ps(x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return tan512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_asin
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return (float)asinf(x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
        return asin_ps(x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return asin256_ps(x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return asin512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_acos
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return (float)acosf(x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
        return acos_ps(x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return acos256_ps(x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return acos512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_atan
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return (float)atanf(x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
        return atan_ps(x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return atan256_ps(x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return atan512_ps(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_reciprocal
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return 1.f / x;
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
        return _mm_div_ps(*(__m128*)_ps_1, x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return _mm256_div_ps(*(__m256*)_ps256_1, x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return _mm512_div_ps(*(__m512*)_ps512_1, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_tanh
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return (float)tanhf(x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
        return tanh_sse(x);
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return tanh_avx(x);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return tanh_avx512(x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_log10
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return (float)log10f(x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
        return _mm_mul_ps(log_ps(x), _mm_set1_ps(0.434294481903));
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return _mm256_mul_ps(log256_ps(x), _mm256_set1_ps(0.434294481903));
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return _mm512_mul_ps(log512_ps(x), _mm512_set1_ps(0.434294481903));
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_round
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return nearbyintf(x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
#if __SSE4_1__
        return _mm_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
        return _mm_cvtepi32_ps(_mm_cvtps_epi32(x));
#endif
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct unary_op_trunc
{
    NCNN_FORCEINLINE float func(const float& x) const
    {
        return (float)truncf(x);
    }
#if __SSE2__
    NCNN_FORCEINLINE __m128 func_pack4(const __m128& x) const
    {
#if __SSE4_1__
        return _mm_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
#else
        return _mm_cvtepi32_ps(_mm_cvttps_epi32(x));
#endif
    }
#if __AVX__
    NCNN_FORCEINLINE __m256 func_pack8(const __m256& x) const
    {
        return _mm256_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
#if __AVX512F__
    NCNN_FORCEINLINE __m512 func_pack16(const __m512& x) const
    {
        return _mm512_roundscale_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

} // namespace UnaryOp_x86_functor

namespace UnaryOp_x86_avx_functor = UnaryOp_x86_functor;
namespace UnaryOp_x86_fma_functor = UnaryOp_x86_functor;
namespace UnaryOp_x86_avx512_functor = UnaryOp_x86_functor;

#endif // LAYER_UNARYOP_FUNCTOR_H

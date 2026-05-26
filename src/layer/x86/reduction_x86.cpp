// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reduction_x86.h"

#include <float.h>
#include <math.h>

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

namespace ncnn {

#if __SSE2__
static NCNN_FORCEINLINE float _mm_reduce_min_ps(const __m128& x128)
{
    const __m128 x64 = _mm_min_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_min_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

#if __AVX__
static NCNN_FORCEINLINE float _mm256_reduce_min_ps(const __m256& x)
{
    const __m128 x128 = _mm_min_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    const __m128 x64 = _mm_min_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_min_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

#if __AVX512F__
static NCNN_FORCEINLINE float _mm512_comp_reduce_min_ps(const __m512& x)
{
    const __m256 x256 = _mm256_min_ps(_mm512_castps512_ps256(x), _mm512_extractf32x8_ps(x, 1));
    const __m128 x128 = _mm_min_ps(_mm256_castps256_ps128(x256), _mm256_extractf128_ps(x256, 1));
    const __m128 x64 = _mm_min_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_min_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

static bool reduction_x86_operation_supported(int operation)
{
    return operation == Reduction::ReductionOp_SUM
           || operation == Reduction::ReductionOp_ASUM
           || operation == Reduction::ReductionOp_SUMSQ
           || operation == Reduction::ReductionOp_MEAN
           || operation == Reduction::ReductionOp_MAX
           || operation == Reduction::ReductionOp_MIN
           || operation == Reduction::ReductionOp_PROD
           || operation == Reduction::ReductionOp_L1
           || operation == Reduction::ReductionOp_L2
           || operation == Reduction::ReductionOp_LogSum
           || operation == Reduction::ReductionOp_LogSumExp;
}

static float reduction_x86_initial_value(int operation)
{
    if (operation == Reduction::ReductionOp_MAX)
        return -FLT_MAX;
    if (operation == Reduction::ReductionOp_MIN)
        return FLT_MAX;
    if (operation == Reduction::ReductionOp_PROD)
        return 1.f;

    return 0.f;
}

static float reduction_x86_combine(float x, float y, int operation)
{
    if (operation == Reduction::ReductionOp_MAX)
        return x > y ? x : y;
    if (operation == Reduction::ReductionOp_MIN)
        return x < y ? x : y;
    if (operation == Reduction::ReductionOp_PROD)
        return x * y;

    return x + y;
}

static float reduction_x86_finalize(float v, int operation, float coeff, int scale)
{
    if (operation == Reduction::ReductionOp_MEAN)
    {
        return v * (coeff / scale);
    }

    if (operation == Reduction::ReductionOp_L2)
    {
        v = sqrtf(v < FLT_MIN ? 0.f : v);
    }

    if (operation == Reduction::ReductionOp_LogSum || operation == Reduction::ReductionOp_LogSumExp)
    {
        v = logf(v);
    }

    return coeff == 1.f ? v : v * coeff;
}

static float reduction_x86_sum(const float* ptr, int size)
{
    float sum = 0.f;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _sum_avx512 = _mm512_setzero_ps();
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        _sum_avx512 = _mm512_add_ps(_sum_avx512, _p);
        ptr += 16;
    }
    sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
    __m256 _sum_avx = _mm256_setzero_ps();
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        _sum_avx = _mm256_add_ps(_sum_avx, _p);
        ptr += 8;
    }
    sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
    __m128 _sum = _mm_setzero_ps();
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        _sum = _mm_add_ps(_sum, _p);
        ptr += 4;
    }
    sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__
    for (; i < size; i++)
    {
        sum += ptr[0];
        ptr++;
    }

    return sum;
}

static float reduction_x86_asum(const float* ptr, int size)
{
    float sum = 0.f;

    int i = 0;
#if __SSE2__
    const __m128 _mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
#if __AVX__
    const __m256 _mask_avx = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
#if __AVX512F__
    const __m512 _mask_avx512 = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    __m512 _sum_avx512 = _mm512_setzero_ps();
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        _p = _mm512_and_ps(_p, _mask_avx512);
        _sum_avx512 = _mm512_add_ps(_sum_avx512, _p);
        ptr += 16;
    }
    sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
    __m256 _sum_avx = _mm256_setzero_ps();
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        _p = _mm256_and_ps(_p, _mask_avx);
        _sum_avx = _mm256_add_ps(_sum_avx, _p);
        ptr += 8;
    }
    sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
    __m128 _sum = _mm_setzero_ps();
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        _p = _mm_and_ps(_p, _mask);
        _sum = _mm_add_ps(_sum, _p);
        ptr += 4;
    }
    sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__
    for (; i < size; i++)
    {
        sum += fabsf(ptr[0]);
        ptr++;
    }

    return sum;
}

static float reduction_x86_sumsq(const float* ptr, int size)
{
    float sum = 0.f;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _sum_avx512 = _mm512_setzero_ps();
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        _sum_avx512 = _mm512_fmadd_ps(_p, _p, _sum_avx512);
        ptr += 16;
    }
    sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
    __m256 _sum_avx = _mm256_setzero_ps();
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        _sum_avx = _mm256_comp_fmadd_ps(_p, _p, _sum_avx);
        ptr += 8;
    }
    sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
    __m128 _sum = _mm_setzero_ps();
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        _sum = _mm_comp_fmadd_ps(_p, _p, _sum);
        ptr += 4;
    }
    sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__
    for (; i < size; i++)
    {
        sum += ptr[0] * ptr[0];
        ptr++;
    }

    return sum;
}

static float reduction_x86_max(const float* ptr, int size)
{
    float v = -FLT_MAX;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _v_avx512 = _mm512_set1_ps(-FLT_MAX);
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        _v_avx512 = _mm512_max_ps(_v_avx512, _p);
        ptr += 16;
    }
    const float v_avx512 = _mm512_comp_reduce_max_ps(_v_avx512);
    v = v > v_avx512 ? v : v_avx512;
#endif // __AVX512F__
    __m256 _v_avx = _mm256_set1_ps(-FLT_MAX);
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        _v_avx = _mm256_max_ps(_v_avx, _p);
        ptr += 8;
    }
    const float v_avx = _mm256_reduce_max_ps(_v_avx);
    v = v > v_avx ? v : v_avx;
#endif // __AVX__
    __m128 _v = _mm_set1_ps(-FLT_MAX);
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        _v = _mm_max_ps(_v, _p);
        ptr += 4;
    }
    const float v_sse = _mm_reduce_max_ps(_v);
    v = v > v_sse ? v : v_sse;
#endif // __SSE2__
    for (; i < size; i++)
    {
        v = v > ptr[0] ? v : ptr[0];
        ptr++;
    }

    return v;
}

static float reduction_x86_min(const float* ptr, int size)
{
    float v = FLT_MAX;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _v_avx512 = _mm512_set1_ps(FLT_MAX);
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        _v_avx512 = _mm512_min_ps(_v_avx512, _p);
        ptr += 16;
    }
    const float v_avx512 = _mm512_comp_reduce_min_ps(_v_avx512);
    v = v < v_avx512 ? v : v_avx512;
#endif // __AVX512F__
    __m256 _v_avx = _mm256_set1_ps(FLT_MAX);
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        _v_avx = _mm256_min_ps(_v_avx, _p);
        ptr += 8;
    }
    const float v_avx = _mm256_reduce_min_ps(_v_avx);
    v = v < v_avx ? v : v_avx;
#endif // __AVX__
    __m128 _v = _mm_set1_ps(FLT_MAX);
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        _v = _mm_min_ps(_v, _p);
        ptr += 4;
    }
    const float v_sse = _mm_reduce_min_ps(_v);
    v = v < v_sse ? v : v_sse;
#endif // __SSE2__
    for (; i < size; i++)
    {
        v = v < ptr[0] ? v : ptr[0];
        ptr++;
    }

    return v;
}

static float reduction_x86_prod(const float* ptr, int size)
{
    float v = 1.f;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _v_avx512 = _mm512_set1_ps(1.f);
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        _v_avx512 = _mm512_mul_ps(_v_avx512, _p);
        ptr += 16;
    }
    float prod_avx512 = 1.f;
    {
        float tmp[16];
        _mm512_storeu_ps(tmp, _v_avx512);
        for (int k = 0; k < 16; k++) prod_avx512 *= tmp[k];
    }
    v *= prod_avx512;
#endif // __AVX512F__
    __m256 _v_avx = _mm256_set1_ps(1.f);
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        _v_avx = _mm256_mul_ps(_v_avx, _p);
        ptr += 8;
    }
    {
        float tmp[8];
        _mm256_storeu_ps(tmp, _v_avx);
        for (int k = 0; k < 8; k++) v *= tmp[k];
    }
#endif // __AVX__
    __m128 _v = _mm_set1_ps(1.f);
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        _v = _mm_mul_ps(_v, _p);
        ptr += 4;
    }
    {
        float tmp[4];
        _mm_storeu_ps(tmp, _v);
        for (int k = 0; k < 4; k++) v *= tmp[k];
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        v *= ptr[0];
        ptr++;
    }

    return v;
}

static float reduction_x86_sumexp(const float* ptr, int size)
{
    float sum = 0.f;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _sum_avx512 = _mm512_setzero_ps();
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        _p = exp512_ps(_p);
        _sum_avx512 = _mm512_add_ps(_sum_avx512, _p);
        ptr += 16;
    }
    sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
    __m256 _sum_avx = _mm256_setzero_ps();
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        _p = exp256_ps(_p);
        _sum_avx = _mm256_add_ps(_sum_avx, _p);
        ptr += 8;
    }
    sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
    __m128 _sum = _mm_setzero_ps();
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        _p = exp_ps(_p);
        _sum = _mm_add_ps(_sum, _p);
        ptr += 4;
    }
    sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__
    for (; i < size; i++)
    {
        sum += expf(ptr[0]);
        ptr++;
    }

    return sum;
}

static float reduction_x86_contiguous(const float* ptr, int size, int operation)
{
    if (operation == Reduction::ReductionOp_ASUM || operation == Reduction::ReductionOp_L1)
        return reduction_x86_asum(ptr, size);
    if (operation == Reduction::ReductionOp_SUMSQ || operation == Reduction::ReductionOp_L2)
        return reduction_x86_sumsq(ptr, size);
    if (operation == Reduction::ReductionOp_MAX)
        return reduction_x86_max(ptr, size);
    if (operation == Reduction::ReductionOp_MIN)
        return reduction_x86_min(ptr, size);
    if (operation == Reduction::ReductionOp_PROD)
        return reduction_x86_prod(ptr, size);
    if (operation == Reduction::ReductionOp_LogSumExp)
        return reduction_x86_sumexp(ptr, size);

    return reduction_x86_sum(ptr, size);
}

static void reduction_x86_vector_fill(float* ptr, int size, int operation)
{
    const float v0 = reduction_x86_initial_value(operation);

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _v0_avx512 = _mm512_set1_ps(v0);
    for (; i + 15 < size; i += 16)
    {
        _mm512_storeu_ps(ptr, _v0_avx512);
        ptr += 16;
    }
#endif // __AVX512F__
    __m256 _v0_avx = _mm256_set1_ps(v0);
    for (; i + 7 < size; i += 8)
    {
        _mm256_storeu_ps(ptr, _v0_avx);
        ptr += 8;
    }
#endif // __AVX__
    __m128 _v0 = _mm_set1_ps(v0);
    for (; i + 3 < size; i += 4)
    {
        _mm_storeu_ps(ptr, _v0);
        ptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        ptr[0] = v0;
        ptr++;
    }
}

static void reduction_x86_vector_accumulate(const float* ptr, float* outptr, int size, int operation)
{
    if (operation == Reduction::ReductionOp_MAX)
    {
        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m512 _outp = _mm512_loadu_ps(outptr);
            _outp = _mm512_max_ps(_outp, _p);
            _mm512_storeu_ps(outptr, _outp);
            ptr += 16;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _outp = _mm256_loadu_ps(outptr);
            _outp = _mm256_max_ps(_outp, _p);
            _mm256_storeu_ps(outptr, _outp);
            ptr += 8;
            outptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            __m128 _outp = _mm_loadu_ps(outptr);
            _outp = _mm_max_ps(_outp, _p);
            _mm_storeu_ps(outptr, _outp);
            ptr += 4;
            outptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            outptr[0] = outptr[0] > ptr[0] ? outptr[0] : ptr[0];
            ptr++;
            outptr++;
        }
        return;
    }

    if (operation == Reduction::ReductionOp_MIN)
    {
        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m512 _outp = _mm512_loadu_ps(outptr);
            _outp = _mm512_min_ps(_outp, _p);
            _mm512_storeu_ps(outptr, _outp);
            ptr += 16;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _outp = _mm256_loadu_ps(outptr);
            _outp = _mm256_min_ps(_outp, _p);
            _mm256_storeu_ps(outptr, _outp);
            ptr += 8;
            outptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            __m128 _outp = _mm_loadu_ps(outptr);
            _outp = _mm_min_ps(_outp, _p);
            _mm_storeu_ps(outptr, _outp);
            ptr += 4;
            outptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            outptr[0] = outptr[0] < ptr[0] ? outptr[0] : ptr[0];
            ptr++;
            outptr++;
        }
        return;
    }

    if (operation == Reduction::ReductionOp_ASUM || operation == Reduction::ReductionOp_L1)
    {
        int i = 0;
#if __SSE2__
        const __m128 _mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
#if __AVX__
        const __m256 _mask_avx = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
#if __AVX512F__
        const __m512 _mask_avx512 = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m512 _outp = _mm512_loadu_ps(outptr);
            _p = _mm512_and_ps(_p, _mask_avx512);
            _outp = _mm512_add_ps(_outp, _p);
            _mm512_storeu_ps(outptr, _outp);
            ptr += 16;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _outp = _mm256_loadu_ps(outptr);
            _p = _mm256_and_ps(_p, _mask_avx);
            _outp = _mm256_add_ps(_outp, _p);
            _mm256_storeu_ps(outptr, _outp);
            ptr += 8;
            outptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            __m128 _outp = _mm_loadu_ps(outptr);
            _p = _mm_and_ps(_p, _mask);
            _outp = _mm_add_ps(_outp, _p);
            _mm_storeu_ps(outptr, _outp);
            ptr += 4;
            outptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            outptr[0] += fabsf(ptr[0]);
            ptr++;
            outptr++;
        }
        return;
    }

    if (operation == Reduction::ReductionOp_SUMSQ || operation == Reduction::ReductionOp_L2)
    {
        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m512 _outp = _mm512_loadu_ps(outptr);
            _outp = _mm512_fmadd_ps(_p, _p, _outp);
            _mm512_storeu_ps(outptr, _outp);
            ptr += 16;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _outp = _mm256_loadu_ps(outptr);
            _outp = _mm256_comp_fmadd_ps(_p, _p, _outp);
            _mm256_storeu_ps(outptr, _outp);
            ptr += 8;
            outptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            __m128 _outp = _mm_loadu_ps(outptr);
            _outp = _mm_comp_fmadd_ps(_p, _p, _outp);
            _mm_storeu_ps(outptr, _outp);
            ptr += 4;
            outptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            outptr[0] += ptr[0] * ptr[0];
            ptr++;
            outptr++;
        }
        return;
    }

    if (operation == Reduction::ReductionOp_PROD)
    {
        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m512 _outp = _mm512_loadu_ps(outptr);
            _outp = _mm512_mul_ps(_outp, _p);
            _mm512_storeu_ps(outptr, _outp);
            ptr += 16;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _outp = _mm256_loadu_ps(outptr);
            _outp = _mm256_mul_ps(_outp, _p);
            _mm256_storeu_ps(outptr, _outp);
            ptr += 8;
            outptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            __m128 _outp = _mm_loadu_ps(outptr);
            _outp = _mm_mul_ps(_outp, _p);
            _mm_storeu_ps(outptr, _outp);
            ptr += 4;
            outptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            outptr[0] *= ptr[0];
            ptr++;
            outptr++;
        }
        return;
    }

    if (operation == Reduction::ReductionOp_LogSumExp)
    {
        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m512 _outp = _mm512_loadu_ps(outptr);
            _p = exp512_ps(_p);
            _outp = _mm512_add_ps(_outp, _p);
            _mm512_storeu_ps(outptr, _outp);
            ptr += 16;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _outp = _mm256_loadu_ps(outptr);
            _p = exp256_ps(_p);
            _outp = _mm256_add_ps(_outp, _p);
            _mm256_storeu_ps(outptr, _outp);
            ptr += 8;
            outptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            __m128 _outp = _mm_loadu_ps(outptr);
            _p = exp_ps(_p);
            _outp = _mm_add_ps(_outp, _p);
            _mm_storeu_ps(outptr, _outp);
            ptr += 4;
            outptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            outptr[0] += expf(ptr[0]);
            ptr++;
            outptr++;
        }
        return;
    }

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        __m512 _outp = _mm512_loadu_ps(outptr);
        _outp = _mm512_add_ps(_outp, _p);
        _mm512_storeu_ps(outptr, _outp);
        ptr += 16;
        outptr += 16;
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        __m256 _outp = _mm256_loadu_ps(outptr);
        _outp = _mm256_add_ps(_outp, _p);
        _mm256_storeu_ps(outptr, _outp);
        ptr += 8;
        outptr += 8;
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        __m128 _outp = _mm_loadu_ps(outptr);
        _outp = _mm_add_ps(_outp, _p);
        _mm_storeu_ps(outptr, _outp);
        ptr += 4;
        outptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        outptr[0] += ptr[0];
        ptr++;
        outptr++;
    }
}

static void reduction_x86_finalize_vector(float* ptr, int size, int operation, float coeff, int scale)
{
    if (operation == Reduction::ReductionOp_MEAN)
    {
        coeff = coeff / scale;
    }

    if (operation == Reduction::ReductionOp_L2 || operation == Reduction::ReductionOp_LogSum || operation == Reduction::ReductionOp_LogSumExp)
    {
        for (int i = 0; i < size; i++)
        {
            ptr[i] = reduction_x86_finalize(ptr[i], operation, coeff, scale);
        }
        return;
    }

    if (coeff == 1.f)
        return;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _coeff_avx512 = _mm512_set1_ps(coeff);
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        _p = _mm512_mul_ps(_p, _coeff_avx512);
        _mm512_storeu_ps(ptr, _p);
        ptr += 16;
    }
#endif // __AVX512F__
    __m256 _coeff_avx = _mm256_set1_ps(coeff);
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        _p = _mm256_mul_ps(_p, _coeff_avx);
        _mm256_storeu_ps(ptr, _p);
        ptr += 8;
    }
#endif // __AVX__
    __m128 _coeff = _mm_set1_ps(coeff);
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        _p = _mm_mul_ps(_p, _coeff);
        _mm_storeu_ps(ptr, _p);
        ptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        ptr[0] = ptr[0] * coeff;
        ptr++;
    }
}

static int reduction_x86_create_top_blob(Mat& top_blob, int outdims, int outw, int outh, int outd, int outc, size_t elemsize, Allocator* allocator)
{
    if (outdims == 0)
        top_blob.create(1, elemsize, allocator);
    if (outdims == 1)
        top_blob.create(outw, elemsize, allocator);
    if (outdims == 2)
        top_blob.create(outw, outh, elemsize, allocator);
    if (outdims == 3)
        top_blob.create(outw, outh, outc, elemsize, allocator);
    if (outdims == 4)
        top_blob.create(outw, outh, outd, outc, elemsize, allocator);

    if (top_blob.empty())
        return -100;

    return 0;
}

int Reduction_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    bool reduce_w, reduce_h, reduce_d, reduce_c;
    int outdims, outw, outh, outd, outc;
    resolve_reduce_flags_and_output_shape(bottom_blob, reduce_w, reduce_h, reduce_d, reduce_c, outdims, outw, outh, outd, outc);

    const int dims = bottom_blob.dims;

    int ret = reduction_x86_create_top_blob(top_blob, outdims, outw, outh, outd, outc, bottom_blob.elemsize, opt.blob_allocator);
    if (ret != 0)
        return ret;

    if (dims == 1)
    {
        const int w = bottom_blob.w;
        float v = reduction_x86_contiguous(bottom_blob, w, operation);
        top_blob[0] = reduction_x86_finalize(v, operation, coeff, w);
        return 0;
    }

    if (dims == 2)
    {
        const int w = bottom_blob.w;
        const int h = bottom_blob.h;

        if (reduce_w && reduce_h)
        {
            float v = reduction_x86_contiguous(bottom_blob, w * h, operation);
            top_blob[0] = reduction_x86_finalize(v, operation, coeff, w * h);
            return 0;
        }

        if (reduce_w && !reduce_h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                float v = reduction_x86_contiguous(ptr, w, operation);
                top_blob[i] = reduction_x86_finalize(v, operation, coeff, w);
            }
            return 0;
        }

        if (!reduce_w && reduce_h)
        {
            float* outptr = top_blob;
            reduction_x86_vector_fill(outptr, w, operation);

            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                reduction_x86_vector_accumulate(ptr, outptr, w, operation);
            }

            reduction_x86_finalize_vector(outptr, w, operation, coeff, h);
            return 0;
        }
    }

    if (dims == 3)
    {
        const int w = bottom_blob.w;
        const int h = bottom_blob.h;
        const int channels = bottom_blob.c;
        const int size = w * h;

        if (reduce_w && reduce_h && !reduce_c)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = keepdims ? top_blob.channel(q) : (float*)top_blob + q;

                float v = reduction_x86_contiguous(ptr, size, operation);
                outptr[0] = reduction_x86_finalize(v, operation, coeff, size);
            }
            return 0;
        }

        if (reduce_w && reduce_h && reduce_c)
        {
            Mat sums(channels, bottom_blob.elemsize, opt.workspace_allocator);
            if (sums.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                sums[q] = reduction_x86_contiguous(ptr, size, operation);
            }

            float v = reduction_x86_initial_value(operation);
            for (int q = 0; q < channels; q++)
            {
                v = reduction_x86_combine(v, sums[q], operation);
            }

            top_blob[0] = reduction_x86_finalize(v, operation, coeff, size * channels);
            return 0;
        }

        if (reduce_w && !reduce_h && !reduce_c)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = keepdims ? top_blob.channel(q) : top_blob.row(q);

                for (int i = 0; i < h; i++)
                {
                    float v = reduction_x86_contiguous(ptr, w, operation);
                    outptr[i] = reduction_x86_finalize(v, operation, coeff, w);
                    ptr += w;
                }
            }
            return 0;
        }

        if (reduce_w && !reduce_h && reduce_c)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float v = reduction_x86_initial_value(operation);
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q).row(i);
                    float vq = reduction_x86_contiguous(ptr, w, operation);
                    v = reduction_x86_combine(v, vq, operation);
                }

                top_blob[i] = reduction_x86_finalize(v, operation, coeff, w * channels);
            }
            return 0;
        }

        if (!reduce_w && reduce_h && reduce_c)
        {
            float* outptr = top_blob;
            reduction_x86_vector_fill(outptr, w, operation);

            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                for (int i = 0; i < h; i++)
                {
                    const float* ptr = m.row(i);
                    reduction_x86_vector_accumulate(ptr, outptr, w, operation);
                }
            }

            reduction_x86_finalize_vector(outptr, w, operation, coeff, h * channels);
            return 0;
        }

        if (!reduce_w && !reduce_h && reduce_c)
        {
            float* outptr = top_blob;
            reduction_x86_vector_fill(outptr, size, operation);

            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                reduction_x86_vector_accumulate(ptr, outptr, size, operation);
            }

            reduction_x86_finalize_vector(outptr, size, operation, coeff, channels);
            return 0;
        }

        if (!reduce_w && reduce_h && !reduce_c)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                float* outptr = keepdims ? top_blob.channel(q) : top_blob.row(q);

                reduction_x86_vector_fill(outptr, w, operation);

                for (int i = 0; i < h; i++)
                {
                    const float* ptr = m.row(i);
                    reduction_x86_vector_accumulate(ptr, outptr, w, operation);
                }

                reduction_x86_finalize_vector(outptr, w, operation, coeff, h);
            }
            return 0;
        }
    }

    if (dims == 4)
    {
        const int w = bottom_blob.w;
        const int h = bottom_blob.h;
        const int d = bottom_blob.d;
        const int channels = bottom_blob.c;
        const int size = w * h * d;

        if (reduce_w && reduce_h && reduce_d && !reduce_c)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = keepdims ? top_blob.channel(q) : (float*)top_blob + q;

                float v = reduction_x86_contiguous(ptr, size, operation);
                outptr[0] = reduction_x86_finalize(v, operation, coeff, size);
            }
            return 0;
        }

        if (reduce_w && reduce_h && reduce_d && reduce_c)
        {
            Mat sums(channels, bottom_blob.elemsize, opt.workspace_allocator);
            if (sums.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                sums[q] = reduction_x86_contiguous(ptr, size, operation);
            }

            float v = reduction_x86_initial_value(operation);
            for (int q = 0; q < channels; q++)
            {
                v = reduction_x86_combine(v, sums[q], operation);
            }

            top_blob[0] = reduction_x86_finalize(v, operation, coeff, size * channels);
            return 0;
        }

        if (reduce_w && reduce_h && !reduce_d && reduce_c)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                float v = reduction_x86_initial_value(operation);
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q).depth(z);
                    float vq = reduction_x86_contiguous(ptr, w * h, operation);
                    v = reduction_x86_combine(v, vq, operation);
                }

                top_blob[z] = reduction_x86_finalize(v, operation, coeff, w * h * channels);
            }
            return 0;
        }

        if (reduce_w && !reduce_h && reduce_d && reduce_c)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                float v = reduction_x86_initial_value(operation);
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    for (int z = 0; z < d; z++)
                    {
                        const float* ptr = m.depth(z).row(y);
                        float vz = reduction_x86_contiguous(ptr, w, operation);
                        v = reduction_x86_combine(v, vz, operation);
                    }
                }

                top_blob[y] = reduction_x86_finalize(v, operation, coeff, w * d * channels);
            }
            return 0;
        }

        if (!reduce_w && reduce_h && reduce_d && reduce_c)
        {
            float* outptr = top_blob;
            reduction_x86_vector_fill(outptr, w, operation);

            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                for (int z = 0; z < d; z++)
                {
                    const Mat mz = m.depth(z);
                    for (int y = 0; y < h; y++)
                    {
                        const float* ptr = mz.row(y);
                        reduction_x86_vector_accumulate(ptr, outptr, w, operation);
                    }
                }
            }

            reduction_x86_finalize_vector(outptr, w, operation, coeff, h * d * channels);
            return 0;
        }

        if (reduce_w && reduce_h && !reduce_d && !reduce_c)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                float* outptr = keepdims ? top_blob.channel(q) : top_blob.row(q);

                for (int z = 0; z < d; z++)
                {
                    const float* ptr = m.depth(z);
                    float v = reduction_x86_contiguous(ptr, w * h, operation);
                    outptr[z] = reduction_x86_finalize(v, operation, coeff, w * h);
                }
            }
            return 0;
        }

        if (reduce_w && !reduce_h && !reduce_d && reduce_c)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                float* outptr = keepdims ? top_blob.depth(z) : top_blob.row(z);
                for (int y = 0; y < h; y++)
                {
                    float v = reduction_x86_initial_value(operation);
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr = bottom_blob.channel(q).depth(z).row(y);
                        float vq = reduction_x86_contiguous(ptr, w, operation);
                        v = reduction_x86_combine(v, vq, operation);
                    }

                    outptr[y] = reduction_x86_finalize(v, operation, coeff, w * channels);
                }
            }
            return 0;
        }

        if (!reduce_w && !reduce_h && reduce_d && reduce_c)
        {
            float* outptr = top_blob;
            reduction_x86_vector_fill(outptr, w * h, operation);

            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                for (int z = 0; z < d; z++)
                {
                    const float* ptr = m.depth(z);
                    reduction_x86_vector_accumulate(ptr, outptr, w * h, operation);
                }
            }

            reduction_x86_finalize_vector(outptr, w * h, operation, coeff, d * channels);
            return 0;
        }

        if (reduce_w && !reduce_h && reduce_d && !reduce_c)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                float* outptr = keepdims ? top_blob.channel(q) : top_blob.row(q);

                for (int y = 0; y < h; y++)
                {
                    float v = reduction_x86_initial_value(operation);
                    for (int z = 0; z < d; z++)
                    {
                        const float* ptr = m.depth(z).row(y);
                        float vz = reduction_x86_contiguous(ptr, w, operation);
                        v = reduction_x86_combine(v, vz, operation);
                    }

                    outptr[y] = reduction_x86_finalize(v, operation, coeff, w * d);
                }
            }
            return 0;
        }

        if (!reduce_w && reduce_h && !reduce_d && reduce_c)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                float* outptr = top_blob.row(z);
                reduction_x86_vector_fill(outptr, w, operation);

                for (int q = 0; q < channels; q++)
                {
                    const Mat mz = bottom_blob.channel(q).depth(z);
                    for (int y = 0; y < h; y++)
                    {
                        const float* ptr = mz.row(y);
                        reduction_x86_vector_accumulate(ptr, outptr, w, operation);
                    }
                }

                reduction_x86_finalize_vector(outptr, w, operation, coeff, h * channels);
            }
            return 0;
        }

        if (!reduce_w && reduce_h && reduce_d && !reduce_c)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                float* outptr = keepdims ? top_blob.channel(q) : top_blob.row(q);
                reduction_x86_vector_fill(outptr, w, operation);

                for (int z = 0; z < d; z++)
                {
                    const Mat mz = m.depth(z);
                    for (int y = 0; y < h; y++)
                    {
                        const float* ptr = mz.row(y);
                        reduction_x86_vector_accumulate(ptr, outptr, w, operation);
                    }
                }

                reduction_x86_finalize_vector(outptr, w, operation, coeff, h * d);
            }
            return 0;
        }

        if (!reduce_w && !reduce_h && !reduce_d && reduce_c)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                float* outptr = keepdims ? top_blob.depth(z) : top_blob.channel(z);
                reduction_x86_vector_fill(outptr, w * h, operation);

                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q).depth(z);
                    reduction_x86_vector_accumulate(ptr, outptr, w * h, operation);
                }

                reduction_x86_finalize_vector(outptr, w * h, operation, coeff, channels);
            }
            return 0;
        }

        if (!reduce_w && reduce_h && !reduce_d && !reduce_c)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                Mat outm = top_blob.channel(q);

                for (int z = 0; z < d; z++)
                {
                    float* outptr = outm.row(z);
                    reduction_x86_vector_fill(outptr, w, operation);

                    const Mat mz = m.depth(z);
                    for (int y = 0; y < h; y++)
                    {
                        const float* ptr = mz.row(y);
                        reduction_x86_vector_accumulate(ptr, outptr, w, operation);
                    }

                    reduction_x86_finalize_vector(outptr, w, operation, coeff, h);
                }
            }
            return 0;
        }

        if (!reduce_w && !reduce_h && reduce_d && !reduce_c)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);
                reduction_x86_vector_fill(outptr, w * h, operation);

                for (int z = 0; z < d; z++)
                {
                    const float* ptr = m.depth(z);
                    reduction_x86_vector_accumulate(ptr, outptr, w * h, operation);
                }

                reduction_x86_finalize_vector(outptr, w * h, operation, coeff, d);
            }
            return 0;
        }

        if (reduce_w && !reduce_h && !reduce_d && !reduce_c)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < d * h; i++)
                {
                    float v = reduction_x86_contiguous(ptr, w, operation);
                    outptr[i] = reduction_x86_finalize(v, operation, coeff, w);
                    ptr += w;
                }
            }
            return 0;
        }
    }

    return 0;
}

} // namespace ncnn

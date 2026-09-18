// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

namespace reduction_x86_functor {

#include "reduction_functor.h"

} // namespace reduction_x86_functor

#if __SSE2__
template<typename Op>
static NCNN_FORCEINLINE float reduction_horizontal(const __m128& v)
{
    const Op op;
    __m128 _v = op.func_pack4(v, _mm_movehl_ps(v, v));
    _v = op.func_pack4(_v, _mm_shuffle_ps(_v, _v, 0x55));
    return _mm_cvtss_f32(_v);
}
#endif // __SSE2__

template<typename Op, typename Op2>
static float reduction(float v0, const float* ptr, int size)
{
    const Op op;
    float v = v0;

    int i = 0;
#if __SSE2__
#if __AVX__
    const Op2 op2;
#if __AVX512F__
    __m512 _sum16 = _mm512_set1_ps(v0);
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        _sum16 = op.func_pack16(_sum16, _p);
        ptr += 16;
    }
    __m256 _sum8 = op2.func_pack8(_mm512_castps512_ps256(_sum16), _mm512_extractf32x8_ps(_sum16, 1));
#else
    __m256 _sum8 = _mm256_set1_ps(v0);
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        _sum8 = op.func_pack8(_sum8, _p);
        ptr += 8;
    }
    __m128 _sum4 = op2.func_pack4(_mm256_castps256_ps128(_sum8), _mm256_extractf128_ps(_sum8, 1));
#else
    __m128 _sum4 = _mm_set1_ps(v0);
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        _sum4 = op.func_pack4(_sum4, _p);
        ptr += 4;
    }
    v = reduction_horizontal<Op2>(_sum4);
#endif // __SSE2__
    for (; i < size; i++)
    {
        v = op.func(v, *ptr);
        ptr++;
    }

    return v;
}

template<typename Op>
static void reduction_vector(const float* ptr, float* outptr, int size)
{
    const Op op;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        __m512 _outp = _mm512_loadu_ps(outptr);
        _outp = op.func_pack16(_outp, _p);
        _mm512_storeu_ps(outptr, _outp);
        ptr += 16;
        outptr += 16;
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        __m256 _outp = _mm256_loadu_ps(outptr);
        _outp = op.func_pack8(_outp, _p);
        _mm256_storeu_ps(outptr, _outp);
        ptr += 8;
        outptr += 8;
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        __m128 _outp = _mm_loadu_ps(outptr);
        _outp = op.func_pack4(_outp, _p);
        _mm_storeu_ps(outptr, _outp);
        ptr += 4;
        outptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        *outptr = op.func(*outptr, *ptr);
        ptr++;
        outptr++;
    }
}

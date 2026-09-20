// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Full register tiles have no size or packing branches.

// Contiguous tiles take only pointers; stride variants use scalar-element strides.

#if __SSE2__
static NCNN_FORCEINLINE void permute_transpose4x4(const float* ptr, float* outptr)
{
#if __AVX512F__
    const __m512i _index = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    __m512i _v = _mm512_loadu_si512(ptr);
    _v = _mm512_permutexvar_epi32(_index, _v);
    _mm512_storeu_si512(outptr, _v);
#else
    __m128 _r0 = _mm_loadu_ps(ptr);
    __m128 _r1 = _mm_loadu_ps(ptr + 4);
    __m128 _r2 = _mm_loadu_ps(ptr + 8);
    __m128 _r3 = _mm_loadu_ps(ptr + 12);
    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
    _mm_storeu_ps(outptr, _r0);
    _mm_storeu_ps(outptr + 4, _r1);
    _mm_storeu_ps(outptr + 8, _r2);
    _mm_storeu_ps(outptr + 12, _r3);
#endif // __AVX512F__
}

static NCNN_FORCEINLINE void permute_transpose4x4_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    __m128 _r0 = _mm_loadu_ps(ptr);
    __m128 _r1 = _mm_loadu_ps(ptr + stride);
    __m128 _r2 = _mm_loadu_ps(ptr + 2 * stride);
    __m128 _r3 = _mm_loadu_ps(ptr + 3 * stride);
    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
    _mm_storeu_ps(outptr, _r0);
    _mm_storeu_ps(outptr + outstride, _r1);
    _mm_storeu_ps(outptr + 2 * outstride, _r2);
    _mm_storeu_ps(outptr + 3 * outstride, _r3);
}
#endif // __SSE2__

#if __AVX__
static NCNN_FORCEINLINE void permute_transpose4x8(const float* ptr, float* outptr)
{
#if __AVX512F__
    __m512i _r0 = _mm512_loadu_si512(ptr);
    __m512i _r1 = _mm512_loadu_si512(ptr + 16);
    const __m512i _index0 = _mm512_setr_epi32(0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27);
    const __m512i _index1 = _mm512_setr_epi32(4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31);
    _mm512_storeu_si512(outptr, _mm512_permutex2var_epi32(_r0, _index0, _r1));
    _mm512_storeu_si512(outptr + 16, _mm512_permutex2var_epi32(_r0, _index1, _r1));
#else
    __m256 _r0 = _mm256_loadu_ps(ptr);
    __m256 _r1 = _mm256_loadu_ps(ptr + 8);
    __m256 _r2 = _mm256_loadu_ps(ptr + 16);
    __m256 _r3 = _mm256_loadu_ps(ptr + 24);
    __m256 _t0 = _mm256_unpacklo_ps(_r0, _r1);
    __m256 _t1 = _mm256_unpackhi_ps(_r0, _r1);
    __m256 _t2 = _mm256_unpacklo_ps(_r2, _r3);
    __m256 _t3 = _mm256_unpackhi_ps(_r2, _r3);
    _r0 = _mm256_shuffle_ps(_t0, _t2, _MM_SHUFFLE(1, 0, 1, 0));
    _r1 = _mm256_shuffle_ps(_t0, _t2, _MM_SHUFFLE(3, 2, 3, 2));
    _r2 = _mm256_shuffle_ps(_t1, _t3, _MM_SHUFFLE(1, 0, 1, 0));
    _r3 = _mm256_shuffle_ps(_t1, _t3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _a = _mm256_permute2f128_ps(_r0, _r1, 0x20);
    __m256 _b = _mm256_permute2f128_ps(_r2, _r3, 0x20);
    __m256 _c = _mm256_permute2f128_ps(_r0, _r1, 0x31);
    __m256 _d = _mm256_permute2f128_ps(_r2, _r3, 0x31);
    _r0 = _a;
    _r1 = _b;
    _r2 = _c;
    _r3 = _d;
    _mm256_storeu_ps(outptr, _r0);
    _mm256_storeu_ps(outptr + 8, _r1);
    _mm256_storeu_ps(outptr + 16, _r2);
    _mm256_storeu_ps(outptr + 24, _r3);
#endif // __AVX512F__
}

static NCNN_FORCEINLINE void permute_transpose4x8_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    permute_transpose4x4_stride(ptr, stride, outptr, outstride);
    permute_transpose4x4_stride(ptr + 4, stride, outptr + 4 * outstride, outstride);
}
#endif // __AVX__

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose4x16(const float* ptr, float* outptr)
{
    __m512 _r0 = _mm512_loadu_ps(ptr);
    __m512 _r1 = _mm512_loadu_ps(ptr + 16);
    __m512 _r2 = _mm512_loadu_ps(ptr + 32);
    __m512 _r3 = _mm512_loadu_ps(ptr + 48);
    transpose16x4_ps(_r0, _r1, _r2, _r3);
    _mm512_storeu_ps(outptr, _r0);
    _mm512_storeu_ps(outptr + 16, _r1);
    _mm512_storeu_ps(outptr + 32, _r2);
    _mm512_storeu_ps(outptr + 48, _r3);
}

static NCNN_FORCEINLINE void permute_transpose4x16_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    permute_transpose4x4_stride(ptr, stride, outptr, outstride);
    permute_transpose4x4_stride(ptr + 4, stride, outptr + 4 * outstride, outstride);
    permute_transpose4x4_stride(ptr + 8, stride, outptr + 8 * outstride, outstride);
    permute_transpose4x4_stride(ptr + 12, stride, outptr + 12 * outstride, outstride);
}
#endif // __AVX512F__

#if __AVX__
static NCNN_FORCEINLINE void permute_transpose8x4(const float* ptr, float* outptr)
{
#if __AVX512F__
    __m512i _r0 = _mm512_loadu_si512(ptr);
    __m512i _r1 = _mm512_loadu_si512(ptr + 16);
    const __m512i _index0 = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29);
    const __m512i _index1 = _mm512_setr_epi32(2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31);
    _mm512_storeu_si512(outptr, _mm512_permutex2var_epi32(_r0, _index0, _r1));
    _mm512_storeu_si512(outptr + 16, _mm512_permutex2var_epi32(_r0, _index1, _r1));
#else
    __m256 _r0 = _mm256_loadu_ps(ptr);
    __m256 _r1 = _mm256_loadu_ps(ptr + 8);
    __m256 _r2 = _mm256_loadu_ps(ptr + 16);
    __m256 _r3 = _mm256_loadu_ps(ptr + 24);
    __m256 _a = _mm256_permute2f128_ps(_r0, _r2, 0x20);
    __m256 _b = _mm256_permute2f128_ps(_r0, _r2, 0x31);
    __m256 _c = _mm256_permute2f128_ps(_r1, _r3, 0x20);
    __m256 _d = _mm256_permute2f128_ps(_r1, _r3, 0x31);
    _r0 = _a;
    _r1 = _b;
    _r2 = _c;
    _r3 = _d;
    __m256 _t0 = _mm256_unpacklo_ps(_r0, _r1);
    __m256 _t1 = _mm256_unpackhi_ps(_r0, _r1);
    __m256 _t2 = _mm256_unpacklo_ps(_r2, _r3);
    __m256 _t3 = _mm256_unpackhi_ps(_r2, _r3);
    _r0 = _mm256_shuffle_ps(_t0, _t2, _MM_SHUFFLE(1, 0, 1, 0));
    _r1 = _mm256_shuffle_ps(_t0, _t2, _MM_SHUFFLE(3, 2, 3, 2));
    _r2 = _mm256_shuffle_ps(_t1, _t3, _MM_SHUFFLE(1, 0, 1, 0));
    _r3 = _mm256_shuffle_ps(_t1, _t3, _MM_SHUFFLE(3, 2, 3, 2));
    _mm256_storeu_ps(outptr, _r0);
    _mm256_storeu_ps(outptr + 8, _r1);
    _mm256_storeu_ps(outptr + 16, _r2);
    _mm256_storeu_ps(outptr + 24, _r3);
#endif // __AVX512F__
}

static NCNN_FORCEINLINE void permute_transpose8x4_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    permute_transpose4x4_stride(ptr, stride, outptr, outstride);
    permute_transpose4x4_stride(ptr + 4 * stride, stride, outptr + 4, outstride);
}
#endif // __AVX__

#if __AVX__
static NCNN_FORCEINLINE void permute_transpose8x8_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    __m256 _r0 = _mm256_loadu_ps(ptr);
    __m256 _r1 = _mm256_loadu_ps(ptr + stride);
    __m256 _r2 = _mm256_loadu_ps(ptr + 2 * stride);
    __m256 _r3 = _mm256_loadu_ps(ptr + 3 * stride);
    __m256 _r4 = _mm256_loadu_ps(ptr + 4 * stride);
    __m256 _r5 = _mm256_loadu_ps(ptr + 5 * stride);
    __m256 _r6 = _mm256_loadu_ps(ptr + 6 * stride);
    __m256 _r7 = _mm256_loadu_ps(ptr + 7 * stride);
    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
    _mm256_storeu_ps(outptr, _r0);
    _mm256_storeu_ps(outptr + outstride, _r1);
    _mm256_storeu_ps(outptr + 2 * outstride, _r2);
    _mm256_storeu_ps(outptr + 3 * outstride, _r3);
    _mm256_storeu_ps(outptr + 4 * outstride, _r4);
    _mm256_storeu_ps(outptr + 5 * outstride, _r5);
    _mm256_storeu_ps(outptr + 6 * outstride, _r6);
    _mm256_storeu_ps(outptr + 7 * outstride, _r7);
}
#endif // __AVX__

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose8x16_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    permute_transpose8x8_stride(ptr, stride, outptr, outstride);
    permute_transpose8x8_stride(ptr + 8, stride, outptr + 8 * outstride, outstride);
}
#endif // __AVX512F__

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose16x4_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    permute_transpose4x4_stride(ptr, stride, outptr, outstride);
    permute_transpose4x4_stride(ptr + 4 * stride, stride, outptr + 4, outstride);
    permute_transpose4x4_stride(ptr + 8 * stride, stride, outptr + 8, outstride);
    permute_transpose4x4_stride(ptr + 12 * stride, stride, outptr + 12, outstride);
}
#endif // __AVX512F__

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose16x8_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    permute_transpose8x8_stride(ptr, stride, outptr, outstride);
    permute_transpose8x8_stride(ptr + 8 * stride, stride, outptr + 8, outstride);
}
#endif // __AVX512F__

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose16x16_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    __m512 _r0 = _mm512_loadu_ps(ptr);
    __m512 _r1 = _mm512_loadu_ps(ptr + stride);
    __m512 _r2 = _mm512_loadu_ps(ptr + 2 * stride);
    __m512 _r3 = _mm512_loadu_ps(ptr + 3 * stride);
    __m512 _r4 = _mm512_loadu_ps(ptr + 4 * stride);
    __m512 _r5 = _mm512_loadu_ps(ptr + 5 * stride);
    __m512 _r6 = _mm512_loadu_ps(ptr + 6 * stride);
    __m512 _r7 = _mm512_loadu_ps(ptr + 7 * stride);
    __m512 _r8 = _mm512_loadu_ps(ptr + 8 * stride);
    __m512 _r9 = _mm512_loadu_ps(ptr + 9 * stride);
    __m512 _ra = _mm512_loadu_ps(ptr + 10 * stride);
    __m512 _rb = _mm512_loadu_ps(ptr + 11 * stride);
    __m512 _rc = _mm512_loadu_ps(ptr + 12 * stride);
    __m512 _rd = _mm512_loadu_ps(ptr + 13 * stride);
    __m512 _re = _mm512_loadu_ps(ptr + 14 * stride);
    __m512 _rf = _mm512_loadu_ps(ptr + 15 * stride);
    transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
    _mm512_storeu_ps(outptr, _r0);
    _mm512_storeu_ps(outptr + outstride, _r1);
    _mm512_storeu_ps(outptr + 2 * outstride, _r2);
    _mm512_storeu_ps(outptr + 3 * outstride, _r3);
    _mm512_storeu_ps(outptr + 4 * outstride, _r4);
    _mm512_storeu_ps(outptr + 5 * outstride, _r5);
    _mm512_storeu_ps(outptr + 6 * outstride, _r6);
    _mm512_storeu_ps(outptr + 7 * outstride, _r7);
    _mm512_storeu_ps(outptr + 8 * outstride, _r8);
    _mm512_storeu_ps(outptr + 9 * outstride, _r9);
    _mm512_storeu_ps(outptr + 10 * outstride, _ra);
    _mm512_storeu_ps(outptr + 11 * outstride, _rb);
    _mm512_storeu_ps(outptr + 12 * outstride, _rc);
    _mm512_storeu_ps(outptr + 13 * outstride, _rd);
    _mm512_storeu_ps(outptr + 14 * outstride, _re);
    _mm512_storeu_ps(outptr + 15 * outstride, _rf);
}
#endif // __AVX512F__

// Final two or one rows: load complete vectors and write only the valid output lanes.
#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose2x16_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    __m512 _r0 = _mm512_loadu_ps(ptr);
    __m512 _r1 = _mm512_loadu_ps(ptr + stride);
    __m512 _t0 = _mm512_unpacklo_ps(_r0, _r1);
    __m512 _t1 = _mm512_unpackhi_ps(_r0, _r1);
    __m128 _v0 = _mm512_extractf32x4_ps(_t0, 0);
    _mm_storel_epi64((__m128i*)(outptr), _mm_castps_si128(_v0));
    _mm_storel_epi64((__m128i*)(outptr + outstride), _mm_srli_si128(_mm_castps_si128(_v0), 8));
    __m128 _v1 = _mm512_extractf32x4_ps(_t1, 0);
    _mm_storel_epi64((__m128i*)(outptr + 2 * outstride), _mm_castps_si128(_v1));
    _mm_storel_epi64((__m128i*)(outptr + 3 * outstride), _mm_srli_si128(_mm_castps_si128(_v1), 8));
    __m128 _v2 = _mm512_extractf32x4_ps(_t0, 1);
    _mm_storel_epi64((__m128i*)(outptr + 4 * outstride), _mm_castps_si128(_v2));
    _mm_storel_epi64((__m128i*)(outptr + 5 * outstride), _mm_srli_si128(_mm_castps_si128(_v2), 8));
    __m128 _v3 = _mm512_extractf32x4_ps(_t1, 1);
    _mm_storel_epi64((__m128i*)(outptr + 6 * outstride), _mm_castps_si128(_v3));
    _mm_storel_epi64((__m128i*)(outptr + 7 * outstride), _mm_srli_si128(_mm_castps_si128(_v3), 8));
    __m128 _v4 = _mm512_extractf32x4_ps(_t0, 2);
    _mm_storel_epi64((__m128i*)(outptr + 8 * outstride), _mm_castps_si128(_v4));
    _mm_storel_epi64((__m128i*)(outptr + 9 * outstride), _mm_srli_si128(_mm_castps_si128(_v4), 8));
    __m128 _v5 = _mm512_extractf32x4_ps(_t1, 2);
    _mm_storel_epi64((__m128i*)(outptr + 10 * outstride), _mm_castps_si128(_v5));
    _mm_storel_epi64((__m128i*)(outptr + 11 * outstride), _mm_srli_si128(_mm_castps_si128(_v5), 8));
    __m128 _v6 = _mm512_extractf32x4_ps(_t0, 3);
    _mm_storel_epi64((__m128i*)(outptr + 12 * outstride), _mm_castps_si128(_v6));
    _mm_storel_epi64((__m128i*)(outptr + 13 * outstride), _mm_srli_si128(_mm_castps_si128(_v6), 8));
    __m128 _v7 = _mm512_extractf32x4_ps(_t1, 3);
    _mm_storel_epi64((__m128i*)(outptr + 14 * outstride), _mm_castps_si128(_v7));
    _mm_storel_epi64((__m128i*)(outptr + 15 * outstride), _mm_srli_si128(_mm_castps_si128(_v7), 8));
}
#endif // __AVX512F__

#if __AVX__
static NCNN_FORCEINLINE void permute_transpose2x8_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    __m256 _r0 = _mm256_loadu_ps(ptr);
    __m256 _r1 = _mm256_loadu_ps(ptr + stride);
    __m256 _t0 = _mm256_unpacklo_ps(_r0, _r1);
    __m256 _t1 = _mm256_unpackhi_ps(_r0, _r1);
    __m128 _v0 = _mm256_castps256_ps128(_t0);
    _mm_storel_epi64((__m128i*)(outptr), _mm_castps_si128(_v0));
    _mm_storel_epi64((__m128i*)(outptr + outstride), _mm_srli_si128(_mm_castps_si128(_v0), 8));
    __m128 _v1 = _mm256_castps256_ps128(_t1);
    _mm_storel_epi64((__m128i*)(outptr + 2 * outstride), _mm_castps_si128(_v1));
    _mm_storel_epi64((__m128i*)(outptr + 3 * outstride), _mm_srli_si128(_mm_castps_si128(_v1), 8));
    __m128 _v2 = _mm256_extractf128_ps(_t0, 1);
    _mm_storel_epi64((__m128i*)(outptr + 4 * outstride), _mm_castps_si128(_v2));
    _mm_storel_epi64((__m128i*)(outptr + 5 * outstride), _mm_srli_si128(_mm_castps_si128(_v2), 8));
    __m128 _v3 = _mm256_extractf128_ps(_t1, 1);
    _mm_storel_epi64((__m128i*)(outptr + 6 * outstride), _mm_castps_si128(_v3));
    _mm_storel_epi64((__m128i*)(outptr + 7 * outstride), _mm_srli_si128(_mm_castps_si128(_v3), 8));
}
#endif // __AVX__

#if __SSE2__
static NCNN_FORCEINLINE void permute_transpose2x4_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    __m128 _r0 = _mm_loadu_ps(ptr);
    __m128 _r1 = _mm_loadu_ps(ptr + stride);
    __m128 _t0 = _mm_unpacklo_ps(_r0, _r1);
    __m128 _t1 = _mm_unpackhi_ps(_r0, _r1);
    __m128 _v0 = _t0;
    _mm_storel_epi64((__m128i*)(outptr), _mm_castps_si128(_v0));
    _mm_storel_epi64((__m128i*)(outptr + outstride), _mm_srli_si128(_mm_castps_si128(_v0), 8));
    __m128 _v1 = _t1;
    _mm_storel_epi64((__m128i*)(outptr + 2 * outstride), _mm_castps_si128(_v1));
    _mm_storel_epi64((__m128i*)(outptr + 3 * outstride), _mm_srli_si128(_mm_castps_si128(_v1), 8));
}
#endif // __SSE2__

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose1x16_stride(const float* ptr, float* outptr, size_t outstride)
{
    __m512 _r0 = _mm512_loadu_ps(ptr);
    __m128 _v0 = _mm512_extractf32x4_ps(_r0, 0);
    _mm_store_ss(outptr, _v0);
    _mm_store_ss(outptr + outstride, _mm_shuffle_ps(_v0, _v0, _MM_SHUFFLE(1, 1, 1, 1)));
    _mm_store_ss(outptr + 2 * outstride, _mm_shuffle_ps(_v0, _v0, _MM_SHUFFLE(2, 2, 2, 2)));
    _mm_store_ss(outptr + 3 * outstride, _mm_shuffle_ps(_v0, _v0, _MM_SHUFFLE(3, 3, 3, 3)));
    __m128 _v1 = _mm512_extractf32x4_ps(_r0, 1);
    _mm_store_ss(outptr + 4 * outstride, _v1);
    _mm_store_ss(outptr + 5 * outstride, _mm_shuffle_ps(_v1, _v1, _MM_SHUFFLE(1, 1, 1, 1)));
    _mm_store_ss(outptr + 6 * outstride, _mm_shuffle_ps(_v1, _v1, _MM_SHUFFLE(2, 2, 2, 2)));
    _mm_store_ss(outptr + 7 * outstride, _mm_shuffle_ps(_v1, _v1, _MM_SHUFFLE(3, 3, 3, 3)));
    __m128 _v2 = _mm512_extractf32x4_ps(_r0, 2);
    _mm_store_ss(outptr + 8 * outstride, _v2);
    _mm_store_ss(outptr + 9 * outstride, _mm_shuffle_ps(_v2, _v2, _MM_SHUFFLE(1, 1, 1, 1)));
    _mm_store_ss(outptr + 10 * outstride, _mm_shuffle_ps(_v2, _v2, _MM_SHUFFLE(2, 2, 2, 2)));
    _mm_store_ss(outptr + 11 * outstride, _mm_shuffle_ps(_v2, _v2, _MM_SHUFFLE(3, 3, 3, 3)));
    __m128 _v3 = _mm512_extractf32x4_ps(_r0, 3);
    _mm_store_ss(outptr + 12 * outstride, _v3);
    _mm_store_ss(outptr + 13 * outstride, _mm_shuffle_ps(_v3, _v3, _MM_SHUFFLE(1, 1, 1, 1)));
    _mm_store_ss(outptr + 14 * outstride, _mm_shuffle_ps(_v3, _v3, _MM_SHUFFLE(2, 2, 2, 2)));
    _mm_store_ss(outptr + 15 * outstride, _mm_shuffle_ps(_v3, _v3, _MM_SHUFFLE(3, 3, 3, 3)));
}
#endif // __AVX512F__

#if __AVX__
static NCNN_FORCEINLINE void permute_transpose1x8_stride(const float* ptr, float* outptr, size_t outstride)
{
    __m256 _r0 = _mm256_loadu_ps(ptr);
    __m128 _v0 = _mm256_castps256_ps128(_r0);
    _mm_store_ss(outptr, _v0);
    _mm_store_ss(outptr + outstride, _mm_shuffle_ps(_v0, _v0, _MM_SHUFFLE(1, 1, 1, 1)));
    _mm_store_ss(outptr + 2 * outstride, _mm_shuffle_ps(_v0, _v0, _MM_SHUFFLE(2, 2, 2, 2)));
    _mm_store_ss(outptr + 3 * outstride, _mm_shuffle_ps(_v0, _v0, _MM_SHUFFLE(3, 3, 3, 3)));
    __m128 _v1 = _mm256_extractf128_ps(_r0, 1);
    _mm_store_ss(outptr + 4 * outstride, _v1);
    _mm_store_ss(outptr + 5 * outstride, _mm_shuffle_ps(_v1, _v1, _MM_SHUFFLE(1, 1, 1, 1)));
    _mm_store_ss(outptr + 6 * outstride, _mm_shuffle_ps(_v1, _v1, _MM_SHUFFLE(2, 2, 2, 2)));
    _mm_store_ss(outptr + 7 * outstride, _mm_shuffle_ps(_v1, _v1, _MM_SHUFFLE(3, 3, 3, 3)));
}
#endif // __AVX__

#if __SSE2__
static NCNN_FORCEINLINE void permute_transpose1x4_stride(const float* ptr, float* outptr, size_t outstride)
{
    __m128 _r0 = _mm_loadu_ps(ptr);
    __m128 _v0 = _r0;
    _mm_store_ss(outptr, _v0);
    _mm_store_ss(outptr + outstride, _mm_shuffle_ps(_v0, _v0, _MM_SHUFFLE(1, 1, 1, 1)));
    _mm_store_ss(outptr + 2 * outstride, _mm_shuffle_ps(_v0, _v0, _MM_SHUFFLE(2, 2, 2, 2)));
    _mm_store_ss(outptr + 3 * outstride, _mm_shuffle_ps(_v0, _v0, _MM_SHUFFLE(3, 3, 3, 3)));
}
#endif // __SSE2__

// Unpacked matrix transpose, shared by 2d and channel/spatial permutations.
static void permute_transpose_pack1(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    int i = 0;
#if __AVX512F__
    for (; i + 15 < rows; i += 16)
    {
        int j = 0;
        for (; j + 15 < cols; j += 16)
        {
            permute_transpose16x16_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 7 < cols; j += 8)
        {
            permute_transpose16x8_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose16x4_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j < cols; j++)
        {
            for (int k = 0; k < 16; k++)
            {
                outptr[j * outstride + i + k] = ptr[(i + k) * stride + j];
            }
        }
    }
#endif // __AVX512F__
#if __AVX__
    for (; i + 7 < rows; i += 8)
    {
        int j = 0;
#if __AVX512F__
        for (; j + 15 < cols; j += 16)
        {
            permute_transpose8x16_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
#endif // __AVX512F__
        for (; j + 7 < cols; j += 8)
        {
            permute_transpose8x8_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose8x4_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j < cols; j++)
        {
            for (int k = 0; k < 8; k++)
            {
                outptr[j * outstride + i + k] = ptr[(i + k) * stride + j];
            }
        }
    }
#endif // __AVX__
#if __SSE2__
    for (; i + 3 < rows; i += 4)
    {
        int j = 0;
#if __AVX512F__
        for (; j + 15 < cols; j += 16)
        {
            permute_transpose4x16_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
#endif // __AVX512F__
#if __AVX__
        for (; j + 7 < cols; j += 8)
        {
            permute_transpose4x8_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
#endif // __AVX__
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose4x4_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j < cols; j++)
        {
            for (int k = 0; k < 4; k++)
            {
                outptr[j * outstride + i + k] = ptr[(i + k) * stride + j];
            }
        }
    }
#endif // __SSE2__
#if __SSE2__
    for (; i + 1 < rows; i += 2)
    {
        int j = 0;
#if __AVX512F__
        for (; j + 15 < cols; j += 16)
        {
            permute_transpose2x16_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
#endif // __AVX512F__
#if __AVX__
        for (; j + 7 < cols; j += 8)
        {
            permute_transpose2x8_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
#endif // __AVX__
#if __SSE2__
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose2x4_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
#endif // __SSE2__
        for (; j < cols; j++)
        {
            outptr[j * outstride + i] = ptr[i * stride + j];
            outptr[j * outstride + i + 1] = ptr[(i + 1) * stride + j];
        }
    }
#endif // __SSE2__
    for (; i < rows; i++)
    {
        int j = 0;
#if __AVX512F__
        for (; j + 15 < cols; j += 16)
        {
            permute_transpose1x16_stride(ptr + i * stride + j, outptr + j * outstride + i, outstride);
        }
#endif // __AVX512F__
#if __AVX__
        for (; j + 7 < cols; j += 8)
        {
            permute_transpose1x8_stride(ptr + i * stride + j, outptr + j * outstride + i, outstride);
        }
#endif // __AVX__
#if __SSE2__
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose1x4_stride(ptr + i * stride + j, outptr + j * outstride + i, outstride);
        }
#endif // __SSE2__
        for (; j < cols; j++)
        {
            outptr[j * outstride + i] = ptr[i * stride + j];
        }
    }
}

// 2d: packed rows become packed output rows after transposing w and h.
#if __SSE2__
static void permute_transpose_pack1to4(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int x = 0; x < cols; x += 4)
    {
        for (int y = 0; y < rows; y++)
        {
            const float* p = ptr + y * stride + x;
            float* out = outptr + (x / 4) * outstride + y * 4;
            __m128i _v = _mm_loadu_si128((const __m128i*)p);
            _mm_storeu_si128((__m128i*)out, _v);
        }
    }
}
#endif // __SSE2__

#if __AVX__
static void permute_transpose_pack1to8(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int x = 0; x < cols; x += 8)
    {
        for (int y = 0; y < rows; y++)
        {
            const float* p = ptr + y * stride + x;
            float* out = outptr + (x / 8) * outstride + y * 8;
            __m256 _v = _mm256_loadu_ps((const float*)p);
            _mm256_storeu_ps((float*)out, _v);
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute_transpose_pack1to16(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int x = 0; x < cols; x += 16)
    {
        for (int y = 0; y < rows; y++)
        {
            const float* p = ptr + y * stride + x;
            float* out = outptr + (x / 16) * outstride + y * 16;
            __m512 _v = _mm512_loadu_ps(p);
            _mm512_storeu_ps(out, _v);
        }
    }
}
#endif // __AVX512F__

#if __SSE2__
static void permute_transpose_pack4to1(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        for (int x = 0; x < cols; x++)
        {
            const float* p = ptr + (y / 4) * stride + x * 4;
            float* out = outptr + x * outstride + y;
            __m128i _v = _mm_loadu_si128((const __m128i*)p);
            _mm_storeu_si128((__m128i*)out, _v);
        }
    }
}
#endif // __SSE2__

#if __SSE2__
static void permute_transpose_pack4to4(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        for (int x = 0; x < cols; x += 4)
        {
            const float* p = ptr + (y / 4) * stride + x * 4;
            float* out = outptr + (x / 4) * outstride + y * 4;
            permute_transpose4x4(p, out);
        }
    }
}
#endif // __SSE2__

#if __AVX__
static void permute_transpose_pack4to8(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        for (int x = 0; x < cols; x += 8)
        {
            const float* p = ptr + (y / 4) * stride + x * 4;
            float* out = outptr + (x / 8) * outstride + y * 8;
            permute_transpose8x4(p, out);
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute_transpose_pack4to16(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        for (int x = 0; x < cols; x += 16)
        {
            const float* p = ptr + (y / 4) * stride + x * 4;
            float* out = outptr + (x / 16) * outstride + y * 16;
            permute_transpose16x4_stride(p, 4, out, 16);
        }
    }
}
#endif // __AVX512F__

#if __AVX__
static void permute_transpose_pack8to1(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        for (int x = 0; x < cols; x++)
        {
            const float* p = ptr + (y / 8) * stride + x * 8;
            float* out = outptr + x * outstride + y;
            __m256 _v = _mm256_loadu_ps((const float*)p);
            _mm256_storeu_ps((float*)out, _v);
        }
    }
}
#endif // __AVX__

#if __AVX__
static void permute_transpose_pack8to4(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        for (int x = 0; x < cols; x += 4)
        {
            const float* p = ptr + (y / 8) * stride + x * 8;
            float* out = outptr + (x / 4) * outstride + y * 4;
            permute_transpose4x8(p, out);
        }
    }
}
#endif // __AVX__

#if __AVX__
static void permute_transpose_pack8to8(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        for (int x = 0; x < cols; x += 8)
        {
            const float* p = ptr + (y / 8) * stride + x * 8;
            float* out = outptr + (x / 8) * outstride + y * 8;
            permute_transpose8x8_stride(p, 8, out, 8);
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute_transpose_pack8to16(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        for (int x = 0; x < cols; x += 16)
        {
            const float* p = ptr + (y / 8) * stride + x * 8;
            float* out = outptr + (x / 16) * outstride + y * 16;
            permute_transpose16x8_stride(p, 8, out, 16);
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute_transpose_pack16to1(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 16)
    {
        for (int x = 0; x < cols; x++)
        {
            const float* p = ptr + (y / 16) * stride + x * 16;
            float* out = outptr + x * outstride + y;
            __m512 _v = _mm512_loadu_ps(p);
            _mm512_storeu_ps(out, _v);
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute_transpose_pack16to4(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 16)
    {
        for (int x = 0; x < cols; x += 4)
        {
            const float* p = ptr + (y / 16) * stride + x * 16;
            float* out = outptr + (x / 4) * outstride + y * 4;
            permute_transpose4x16(p, out);
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute_transpose_pack16to8(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 16)
    {
        for (int x = 0; x < cols; x += 8)
        {
            const float* p = ptr + (y / 16) * stride + x * 16;
            float* out = outptr + (x / 8) * outstride + y * 8;
            permute_transpose8x16_stride(p, 16, out, 8);
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute_transpose_pack16to16(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 16)
    {
        for (int x = 0; x < cols; x += 16)
        {
            const float* p = ptr + (y / 16) * stride + x * 16;
            float* out = outptr + (x / 16) * outstride + y * 16;
            permute_transpose16x16_stride(p, 16, out, 16);
        }
    }
}
#endif // __AVX512F__

static void permute_transpose2d(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols, int elempack, int out_elempack)
{
    if (elempack == 1 && out_elempack == 1)
    {
        permute_transpose_pack1(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

#if __SSE2__
    if (elempack == 1 && out_elempack == 4)
    {
        permute_transpose_pack1to4(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __SSE2__

#if __AVX__
    if (elempack == 1 && out_elempack == 8)
    {
        permute_transpose_pack1to8(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 1 && out_elempack == 16)
    {
        permute_transpose_pack1to16(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__

#if __SSE2__
    if (elempack == 4 && out_elempack == 1)
    {
        permute_transpose_pack4to1(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __SSE2__

#if __SSE2__
    if (elempack == 4 && out_elempack == 4)
    {
        permute_transpose_pack4to4(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __SSE2__

#if __AVX__
    if (elempack == 4 && out_elempack == 8)
    {
        permute_transpose_pack4to8(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 4 && out_elempack == 16)
    {
        permute_transpose_pack4to16(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__

#if __AVX__
    if (elempack == 8 && out_elempack == 1)
    {
        permute_transpose_pack8to1(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX__

#if __AVX__
    if (elempack == 8 && out_elempack == 4)
    {
        permute_transpose_pack8to4(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX__

#if __AVX__
    if (elempack == 8 && out_elempack == 8)
    {
        permute_transpose_pack8to8(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 8 && out_elempack == 16)
    {
        permute_transpose_pack8to16(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 1)
    {
        permute_transpose_pack16to1(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 4)
    {
        permute_transpose_pack16to4(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 8)
    {
        permute_transpose_pack16to8(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 16)
    {
        permute_transpose_pack16to16(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__
}

// Spatial transpose within one input channel group. outcstep is used when unpacking.
#if __SSE2__
static void permute_spatial_pack4(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        const int ymax = std::min(y + 8, rows);
        for (int x = 0; x < cols; x += 8)
        {
            const int xmax = std::min(x + 8, cols);
            for (int i = y; i < ymax; i++)
            {
                const float* p = ptr + i * stride + x * 4;
                float* out = outptr + x * outstride + i * 4;
                for (int j = x; j < xmax; j++)
                {
                    __m128i _v = _mm_loadu_si128((const __m128i*)p);
                    _mm_storeu_si128((__m128i*)out, _v);
                    p += 4;
                    out += outstride;
                }
            }
        }
    }
}
#endif // __SSE2__

#if __AVX__
static void permute_spatial_pack8(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        const int ymax = std::min(y + 8, rows);
        for (int x = 0; x < cols; x += 8)
        {
            const int xmax = std::min(x + 8, cols);
            for (int i = y; i < ymax; i++)
            {
                const float* p = ptr + i * stride + x * 8;
                float* out = outptr + x * outstride + i * 8;
                for (int j = x; j < xmax; j++)
                {
                    __m256 _v = _mm256_loadu_ps((const float*)p);
                    _mm256_storeu_ps((float*)out, _v);
                    p += 8;
                    out += outstride;
                }
            }
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute_spatial_pack16(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        const int ymax = std::min(y + 8, rows);
        for (int x = 0; x < cols; x += 8)
        {
            const int xmax = std::min(x + 8, cols);
            for (int i = y; i < ymax; i++)
            {
                const float* p = ptr + i * stride + x * 16;
                float* out = outptr + x * outstride + i * 16;
                for (int j = x; j < xmax; j++)
                {
                    __m512 _v = _mm512_loadu_ps(p);
                    _mm512_storeu_ps(out, _v);
                    p += 16;
                    out += outstride;
                }
            }
        }
    }
}
#endif // __AVX512F__

static void permute_transpose_spatial(const float* ptr, size_t stride, float* outptr, size_t outstride, size_t outcstep, int rows, int cols, int elempack, int out_elempack)
{
    if (elempack == 1)
    {
        permute_transpose_pack1(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#if __SSE2__
    if (elempack == 4 && out_elempack == 4)
    {
        permute_spatial_pack4(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
    if (elempack == 4 && out_elempack == 1)
    {
        for (int x = 0; x < cols; x++)
            permute_transpose_pack1(ptr + x * 4, stride, outptr + x * outstride, outcstep, rows, 4);
        return;
    }
#endif // __SSE2__

#if __AVX__
    if (elempack == 8 && out_elempack == 8)
    {
        permute_spatial_pack8(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
    if (elempack == 8 && out_elempack == 1)
    {
        for (int x = 0; x < cols; x++)
            permute_transpose_pack1(ptr + x * 8, stride, outptr + x * outstride, outcstep, rows, 8);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 16)
    {
        permute_spatial_pack16(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
    if (elempack == 16 && out_elempack == 1)
    {
        for (int x = 0; x < cols; x++)
            permute_transpose_pack1(ptr + x * 16, stride, outptr + x * outstride, outcstep, rows, 16);
        return;
    }
#endif // __AVX512F__
}

static void permute_copy_spatial(const float* ptr, float* outptr, size_t outcstep, int size, int elempack, int out_elempack)
{
    if (elempack == out_elempack)
    {
        memcpy(outptr, ptr, (size_t)size * elempack * sizeof(float));
        return;
    }
#if __SSE2__
    if (elempack == 4)
    {
        permute_transpose_pack1(ptr, 4, outptr, outcstep, size, 4);
        return;
    }
#endif // __SSE2__

#if __AVX__
    if (elempack == 8)
    {
        permute_transpose_pack1(ptr, 8, outptr, outcstep, size, 8);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 16)
    {
        permute_transpose_pack1(ptr, 16, outptr, outcstep, size, 16);
        return;
    }
#endif // __AVX512F__
}

// Exchange the input channel axis with h. w is the remaining spatial axis.
// Strides are in scalar elements. cstep and outhstep include channel padding.
// Pack1 input uses contiguous w or h; pack1 output uses contiguous w or c.
#if __SSE2__
static void permute3d_pack1to4(const float* ptr, float* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (hstep != 1)
    {
        for (int q = 0; q < h / 4; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const float* p = ptr + c * cstep + q * 4 * hstep;
                float* out = outptr + q * outhstep + c * outcstep;
                permute_transpose_pack1(p, hstep, out, outwstep, 4, w);
            }
        }
        return;
    }
    for (int q = 0; q < h / 4; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const float* p = ptr + c * cstep + q * 4 * hstep;
            float* out = outptr + q * outhstep + c * outcstep;
            for (int x = 0; x < w; x++)
            {
                __m128i _v = _mm_loadu_si128((const __m128i*)p);
                _mm_storeu_si128((__m128i*)out, _v);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __SSE2__

#if __AVX__
static void permute3d_pack1to8(const float* ptr, float* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (hstep != 1)
    {
        for (int q = 0; q < h / 8; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const float* p = ptr + c * cstep + q * 8 * hstep;
                float* out = outptr + q * outhstep + c * outcstep;
                permute_transpose_pack1(p, hstep, out, outwstep, 8, w);
            }
        }
        return;
    }
    for (int q = 0; q < h / 8; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const float* p = ptr + c * cstep + q * 8 * hstep;
            float* out = outptr + q * outhstep + c * outcstep;
            for (int x = 0; x < w; x++)
            {
                __m256 _v = _mm256_loadu_ps((const float*)p);
                _mm256_storeu_ps((float*)out, _v);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute3d_pack1to16(const float* ptr, float* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (hstep != 1)
    {
        for (int q = 0; q < h / 16; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const float* p = ptr + c * cstep + q * 16 * hstep;
                float* out = outptr + q * outhstep + c * outcstep;
                permute_transpose_pack1(p, hstep, out, outwstep, 16, w);
            }
        }
        return;
    }
    for (int q = 0; q < h / 16; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const float* p = ptr + c * cstep + q * 16 * hstep;
            float* out = outptr + q * outhstep + c * outcstep;
            for (int x = 0; x < w; x++)
            {
                __m512 _v = _mm512_loadu_ps(p);
                _mm512_storeu_ps(out, _v);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX512F__

#if __SSE2__
static void permute3d_pack4to1(const float* ptr, float* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (outcstep != 1)
    {
        for (int q = 0; q < h; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const float* p = ptr + c * cstep + q * hstep;
                float* out = outptr + q * outhstep + c * 4 * outcstep;
                permute_transpose_pack1(p, wstep, out, outcstep, w, 4);
            }
        }
        return;
    }
    for (int q = 0; q < h; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const float* p = ptr + c * cstep + q * hstep;
            float* out = outptr + q * outhstep + c * 4 * outcstep;
            for (int x = 0; x < w; x++)
            {
                __m128i _v = _mm_loadu_si128((const __m128i*)p);
                _mm_storeu_si128((__m128i*)out, _v);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __SSE2__

#if __SSE2__
static void permute3d_pack4to4(const float* ptr, float* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (hstep == 4 && outcstep == 4)
    {
        for (int q = 0; q < h / 4; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const float* p = ptr + c * cstep + q * 4 * hstep;
                float* out = outptr + q * outhstep + c * 4 * outcstep;
                for (int x = 0; x < w; x++)
                {
                    permute_transpose4x4(p, out);
                    p += wstep;
                    out += outwstep;
                }
            }
        }
        return;
    }

    for (int q = 0; q < h / 4; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const float* p = ptr + c * cstep + q * 4 * hstep;
            float* out = outptr + q * outhstep + c * 4 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose4x4_stride(p, hstep, out, outcstep);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __SSE2__

#if __AVX__
static void permute3d_pack4to8(const float* ptr, float* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (hstep == 4 && outcstep == 8)
    {
        for (int q = 0; q < h / 8; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const float* p = ptr + c * cstep + q * 8 * hstep;
                float* out = outptr + q * outhstep + c * 4 * outcstep;
                for (int x = 0; x < w; x++)
                {
                    permute_transpose8x4(p, out);
                    p += wstep;
                    out += outwstep;
                }
            }
        }
        return;
    }

    for (int q = 0; q < h / 8; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const float* p = ptr + c * cstep + q * 8 * hstep;
            float* out = outptr + q * outhstep + c * 4 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose8x4_stride(p, hstep, out, outcstep);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute3d_pack4to16(const float* ptr, float* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    for (int q = 0; q < h / 16; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const float* p = ptr + c * cstep + q * 16 * hstep;
            float* out = outptr + q * outhstep + c * 4 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose16x4_stride(p, hstep, out, outcstep);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX512F__

#if __AVX__
static void permute3d_pack8to1(const float* ptr, float* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (outcstep != 1)
    {
        for (int q = 0; q < h; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const float* p = ptr + c * cstep + q * hstep;
                float* out = outptr + q * outhstep + c * 8 * outcstep;
                permute_transpose_pack1(p, wstep, out, outcstep, w, 8);
            }
        }
        return;
    }
    for (int q = 0; q < h; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const float* p = ptr + c * cstep + q * hstep;
            float* out = outptr + q * outhstep + c * 8 * outcstep;
            for (int x = 0; x < w; x++)
            {
                __m256 _v = _mm256_loadu_ps((const float*)p);
                _mm256_storeu_ps((float*)out, _v);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX__

#if __AVX__
static void permute3d_pack8to4(const float* ptr, float* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (hstep == 8 && outcstep == 4)
    {
        for (int q = 0; q < h / 4; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const float* p = ptr + c * cstep + q * 4 * hstep;
                float* out = outptr + q * outhstep + c * 8 * outcstep;
                for (int x = 0; x < w; x++)
                {
                    permute_transpose4x8(p, out);
                    p += wstep;
                    out += outwstep;
                }
            }
        }
        return;
    }

    for (int q = 0; q < h / 4; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const float* p = ptr + c * cstep + q * 4 * hstep;
            float* out = outptr + q * outhstep + c * 8 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose4x8_stride(p, hstep, out, outcstep);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX__

#if __AVX__
static void permute3d_pack8to8(const float* ptr, float* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    for (int q = 0; q < h / 8; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const float* p = ptr + c * cstep + q * 8 * hstep;
            float* out = outptr + q * outhstep + c * 8 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose8x8_stride(p, hstep, out, outcstep);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute3d_pack8to16(const float* ptr, float* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    for (int q = 0; q < h / 16; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const float* p = ptr + c * cstep + q * 16 * hstep;
            float* out = outptr + q * outhstep + c * 8 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose16x8_stride(p, hstep, out, outcstep);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute3d_pack16to1(const float* ptr, float* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (outcstep != 1)
    {
        for (int q = 0; q < h; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const float* p = ptr + c * cstep + q * hstep;
                float* out = outptr + q * outhstep + c * 16 * outcstep;
                permute_transpose_pack1(p, wstep, out, outcstep, w, 16);
            }
        }
        return;
    }
    for (int q = 0; q < h; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const float* p = ptr + c * cstep + q * hstep;
            float* out = outptr + q * outhstep + c * 16 * outcstep;
            for (int x = 0; x < w; x++)
            {
                __m512 _v = _mm512_loadu_ps(p);
                _mm512_storeu_ps(out, _v);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute3d_pack16to4(const float* ptr, float* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (hstep == 16 && outcstep == 4)
    {
        for (int q = 0; q < h / 4; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const float* p = ptr + c * cstep + q * 4 * hstep;
                float* out = outptr + q * outhstep + c * 16 * outcstep;
                for (int x = 0; x < w; x++)
                {
                    permute_transpose4x16(p, out);
                    p += wstep;
                    out += outwstep;
                }
            }
        }
        return;
    }

    for (int q = 0; q < h / 4; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const float* p = ptr + c * cstep + q * 4 * hstep;
            float* out = outptr + q * outhstep + c * 16 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose4x16_stride(p, hstep, out, outcstep);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute3d_pack16to8(const float* ptr, float* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    for (int q = 0; q < h / 8; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const float* p = ptr + c * cstep + q * 8 * hstep;
            float* out = outptr + q * outhstep + c * 16 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose8x16_stride(p, hstep, out, outcstep);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute3d_pack16to16(const float* ptr, float* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    for (int q = 0; q < h / 16; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const float* p = ptr + c * cstep + q * 16 * hstep;
            float* out = outptr + q * outhstep + c * 16 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose16x16_stride(p, hstep, out, outcstep);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX512F__

static void permute3d(const float* ptr, float* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep, int elempack, int out_elempack)
{
#if __SSE2__
    if (elempack == 1 && out_elempack == 4)
    {
        permute3d_pack1to4(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __SSE2__

#if __AVX__
    if (elempack == 1 && out_elempack == 8)
    {
        permute3d_pack1to8(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 1 && out_elempack == 16)
    {
        permute3d_pack1to16(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX512F__

#if __SSE2__
    if (elempack == 4 && out_elempack == 1)
    {
        permute3d_pack4to1(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __SSE2__

#if __SSE2__
    if (elempack == 4 && out_elempack == 4)
    {
        permute3d_pack4to4(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __SSE2__

#if __AVX__
    if (elempack == 4 && out_elempack == 8)
    {
        permute3d_pack4to8(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 4 && out_elempack == 16)
    {
        permute3d_pack4to16(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX512F__

#if __AVX__
    if (elempack == 8 && out_elempack == 1)
    {
        permute3d_pack8to1(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX__

#if __AVX__
    if (elempack == 8 && out_elempack == 4)
    {
        permute3d_pack8to4(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX__

#if __AVX__
    if (elempack == 8 && out_elempack == 8)
    {
        permute3d_pack8to8(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 8 && out_elempack == 16)
    {
        permute3d_pack8to16(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 1)
    {
        permute3d_pack16to1(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 4)
    {
        permute3d_pack16to4(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 8)
    {
        permute3d_pack16to8(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 16)
    {
        permute3d_pack16to16(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX512F__
}

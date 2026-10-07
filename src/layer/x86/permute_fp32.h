// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// full register tiles have no size or packing branches

// tile names use input columns x rows
// contiguous tiles take only pointers; stride variants use scalar-element strides

#if __SSE2__
static NCNN_FORCEINLINE void permute_transpose2x4_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    __m128 _r0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr)));
    __m128 _r1 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + stride)));
    __m128 _r2 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 2 * stride)));
    __m128 _r3 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 3 * stride)));
    __m128 _t0 = _mm_movelh_ps(_r0, _r1);
    __m128 _t1 = _mm_movelh_ps(_r2, _r3);
    __m128 _a0 = _mm_shuffle_ps(_t0, _t1, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 _b0 = _mm_shuffle_ps(_t0, _t1, _MM_SHUFFLE(3, 1, 3, 1));
    _mm_storeu_ps(outptr, _a0);
    _mm_storeu_ps(outptr + outstride, _b0);
}

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

#if __AVX__
static NCNN_FORCEINLINE void permute_transpose8x4(const float* ptr, float* outptr)
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
    transpose8x4_ps(_r0, _r1, _r2, _r3);
    _mm256_storeu_ps(outptr, _r0);
    _mm256_storeu_ps(outptr + 8, _r1);
    _mm256_storeu_ps(outptr + 16, _r2);
    _mm256_storeu_ps(outptr + 24, _r3);
#endif // __AVX512F__
}

static NCNN_FORCEINLINE void permute_transpose8x4_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    permute_transpose4x4_stride(ptr, stride, outptr, outstride);
    permute_transpose4x4_stride(ptr + 4, stride, outptr + 4 * outstride, outstride);
}

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose16x4(const float* ptr, float* outptr)
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

static NCNN_FORCEINLINE void permute_transpose16x4_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    permute_transpose4x4_stride(ptr, stride, outptr, outstride);
    permute_transpose4x4_stride(ptr + 4, stride, outptr + 4 * outstride, outstride);
    permute_transpose4x4_stride(ptr + 8, stride, outptr + 8 * outstride, outstride);
    permute_transpose4x4_stride(ptr + 12, stride, outptr + 12 * outstride, outstride);
}
#endif // __AVX512F__

static NCNN_FORCEINLINE void permute_transpose2x8_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    __m128 _r0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr)));
    __m128 _r1 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + stride)));
    __m128 _r2 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 2 * stride)));
    __m128 _r3 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 3 * stride)));
    __m128 _t0 = _mm_movelh_ps(_r0, _r1);
    __m128 _t1 = _mm_movelh_ps(_r2, _r3);
    __m128 _a0 = _mm_shuffle_ps(_t0, _t1, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 _b0 = _mm_shuffle_ps(_t0, _t1, _MM_SHUFFLE(3, 1, 3, 1));
    __m128 _r4 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 4 * stride)));
    __m128 _r5 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 5 * stride)));
    __m128 _r6 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 6 * stride)));
    __m128 _r7 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 7 * stride)));
    __m128 _t2 = _mm_movelh_ps(_r4, _r5);
    __m128 _t3 = _mm_movelh_ps(_r6, _r7);
    __m128 _a1 = _mm_shuffle_ps(_t2, _t3, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 _b1 = _mm_shuffle_ps(_t2, _t3, _MM_SHUFFLE(3, 1, 3, 1));
    __m256 _a = combine4x2_ps(_a0, _a1);
    __m256 _b = combine4x2_ps(_b0, _b1);
    _mm256_storeu_ps(outptr, _a);
    _mm256_storeu_ps(outptr + outstride, _b);
}

static NCNN_FORCEINLINE void permute_transpose4x8(const float* ptr, float* outptr)
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

static NCNN_FORCEINLINE void permute_transpose4x8_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    permute_transpose4x4_stride(ptr, stride, outptr, outstride);
    permute_transpose4x4_stride(ptr + 4 * stride, stride, outptr + 4, outstride);
}

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

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose16x8_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    permute_transpose8x8_stride(ptr, stride, outptr, outstride);
    permute_transpose8x8_stride(ptr + 8, stride, outptr + 8 * outstride, outstride);
}

static NCNN_FORCEINLINE void permute_transpose2x16_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    __m128 _r0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr)));
    __m128 _r1 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + stride)));
    __m128 _r2 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 2 * stride)));
    __m128 _r3 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 3 * stride)));
    __m128 _t0 = _mm_movelh_ps(_r0, _r1);
    __m128 _t1 = _mm_movelh_ps(_r2, _r3);
    __m128 _a0 = _mm_shuffle_ps(_t0, _t1, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 _b0 = _mm_shuffle_ps(_t0, _t1, _MM_SHUFFLE(3, 1, 3, 1));
    __m128 _r4 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 4 * stride)));
    __m128 _r5 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 5 * stride)));
    __m128 _r6 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 6 * stride)));
    __m128 _r7 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 7 * stride)));
    __m128 _t2 = _mm_movelh_ps(_r4, _r5);
    __m128 _t3 = _mm_movelh_ps(_r6, _r7);
    __m128 _a1 = _mm_shuffle_ps(_t2, _t3, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 _b1 = _mm_shuffle_ps(_t2, _t3, _MM_SHUFFLE(3, 1, 3, 1));
    __m128 _r8 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 8 * stride)));
    __m128 _r9 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 9 * stride)));
    __m128 _r10 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 10 * stride)));
    __m128 _r11 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 11 * stride)));
    __m128 _t4 = _mm_movelh_ps(_r8, _r9);
    __m128 _t5 = _mm_movelh_ps(_r10, _r11);
    __m128 _a2 = _mm_shuffle_ps(_t4, _t5, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 _b2 = _mm_shuffle_ps(_t4, _t5, _MM_SHUFFLE(3, 1, 3, 1));
    __m128 _r12 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 12 * stride)));
    __m128 _r13 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 13 * stride)));
    __m128 _r14 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 14 * stride)));
    __m128 _r15 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(ptr + 15 * stride)));
    __m128 _t6 = _mm_movelh_ps(_r12, _r13);
    __m128 _t7 = _mm_movelh_ps(_r14, _r15);
    __m128 _a3 = _mm_shuffle_ps(_t6, _t7, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 _b3 = _mm_shuffle_ps(_t6, _t7, _MM_SHUFFLE(3, 1, 3, 1));
    __m512 _a = combine4x4_ps(_a0, _a1, _a2, _a3);
    __m512 _b = combine4x4_ps(_b0, _b1, _b2, _b3);
    _mm512_storeu_ps(outptr, _a);
    _mm512_storeu_ps(outptr + outstride, _b);
}

static NCNN_FORCEINLINE void permute_transpose4x16_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    permute_transpose4x4_stride(ptr, stride, outptr, outstride);
    permute_transpose4x4_stride(ptr + 4 * stride, stride, outptr + 4, outstride);
    permute_transpose4x4_stride(ptr + 8 * stride, stride, outptr + 8, outstride);
    permute_transpose4x4_stride(ptr + 12 * stride, stride, outptr + 12, outstride);
}

static NCNN_FORCEINLINE void permute_transpose8x16_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    permute_transpose8x8_stride(ptr, stride, outptr, outstride);
    permute_transpose8x8_stride(ptr + 8 * stride, stride, outptr + 8, outstride);
}

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
#endif // __AVX__
#endif // __SSE2__

// final two or one rows: load complete vectors and write only the valid output lanes
#if __SSE2__
#if __AVX__
#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose16x2_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
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

static NCNN_FORCEINLINE void permute_transpose8x2_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
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

static NCNN_FORCEINLINE void permute_transpose4x2_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
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

#if __AVX__
#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose16x1_stride(const float* ptr, float* outptr, size_t outstride)
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

static NCNN_FORCEINLINE void permute_transpose8x1_stride(const float* ptr, float* outptr, size_t outstride)
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

static NCNN_FORCEINLINE void permute_transpose4x1_stride(const float* ptr, float* outptr, size_t outstride)
{
    memcpy(outptr, ptr, sizeof(float));
    memcpy(outptr + outstride, ptr + 1, sizeof(float));
    memcpy(outptr + 2 * outstride, ptr + 2, sizeof(float));
    memcpy(outptr + 3 * outstride, ptr + 3, sizeof(float));
}

static void permute_pack4_stride(const float* ptr, size_t stride, float* outptr, size_t outstride, int cols)
{
    int j = 0;
#if __AVX__
#if __AVX512F__
    for (; j + 15 < cols; j += 16)
    {
        permute_transpose16x4_stride(ptr + j, stride, outptr + j * outstride, outstride);
    }
#endif // __AVX512F__
    for (; j + 7 < cols; j += 8)
    {
        permute_transpose8x4_stride(ptr + j, stride, outptr + j * outstride, outstride);
    }
#endif // __AVX__
    for (; j + 3 < cols; j += 4)
    {
        permute_transpose4x4_stride(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j + 1 < cols; j += 2)
    {
        permute_transpose2x4_stride(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j < cols; j++)
    {
        for (int k = 0; k < 4; k++)
            outptr[j * outstride + k] = ptr[k * stride + j];
    }
}

#if __AVX__
static void permute_pack8_stride(const float* ptr, size_t stride, float* outptr, size_t outstride, int cols)
{
    int j = 0;
#if __AVX512F__
    for (; j + 15 < cols; j += 16)
    {
        permute_transpose16x8_stride(ptr + j, stride, outptr + j * outstride, outstride);
    }
#endif // __AVX512F__
    for (; j + 7 < cols; j += 8)
    {
        permute_transpose8x8_stride(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j + 3 < cols; j += 4)
    {
        permute_transpose4x8_stride(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j + 1 < cols; j += 2)
    {
        permute_transpose2x8_stride(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j < cols; j++)
    {
        for (int k = 0; k < 8; k++)
            outptr[j * outstride + k] = ptr[k * stride + j];
    }
}

#if __AVX512F__
static void permute_pack16_stride(const float* ptr, size_t stride, float* outptr, size_t outstride, int cols)
{
    int j = 0;
    for (; j + 15 < cols; j += 16)
    {
        permute_transpose16x16_stride(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j + 7 < cols; j += 8)
    {
        permute_transpose8x16_stride(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j + 3 < cols; j += 4)
    {
        permute_transpose4x16_stride(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j + 1 < cols; j += 2)
    {
        permute_transpose2x16_stride(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j < cols; j++)
    {
        for (int k = 0; k < 16; k++)
            outptr[j * outstride + k] = ptr[k * stride + j];
    }
}
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

// unpacked matrix transpose, shared by 2d and channel/spatial permutations
#if __SSE2__
// fixed input width; callers select the packing before traversing the rows
static void permute_unpack4_stride(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows)
{
    int i = 0;
#if __AVX__
#if __AVX512F__
    for (; i + 15 < rows; i += 16)
    {
        permute_transpose4x16_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
#endif // __AVX512F__
    for (; i + 7 < rows; i += 8)
    {
        permute_transpose4x8_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
#endif // __AVX__
    for (; i + 3 < rows; i += 4)
    {
        permute_transpose4x4_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 1 < rows; i += 2)
    {
        permute_transpose4x2_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i < rows; i++)
    {
        permute_transpose4x1_stride(ptr + i * stride, outptr + i, outstride);
    }
}

#if __AVX__
// fixed input width; callers select the packing before traversing the rows
static void permute_unpack8_stride(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows)
{
    int i = 0;
#if __AVX512F__
    for (; i + 15 < rows; i += 16)
    {
        permute_transpose8x16_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
#endif // __AVX512F__
    for (; i + 7 < rows; i += 8)
    {
        permute_transpose8x8_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 3 < rows; i += 4)
    {
        permute_transpose8x4_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 1 < rows; i += 2)
    {
        permute_transpose8x2_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i < rows; i++)
    {
        permute_transpose8x1_stride(ptr + i * stride, outptr + i, outstride);
    }
}

#if __AVX512F__
// fixed input width; callers select the packing before traversing the rows
static void permute_unpack16_stride(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows)
{
    int i = 0;
    for (; i + 15 < rows; i += 16)
    {
        permute_transpose16x16_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 7 < rows; i += 8)
    {
        permute_transpose16x8_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 3 < rows; i += 4)
    {
        permute_transpose16x4_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 1 < rows; i += 2)
    {
        permute_transpose16x2_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i < rows; i++)
    {
        permute_transpose16x1_stride(ptr + i * stride, outptr + i, outstride);
    }
}
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

static void permute_transpose_pack1(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    if (cols == 1)
    {
        if (stride == 1)
            memcpy(outptr, ptr, (size_t)rows * sizeof(float));
        else
            for (int i = 0; i < rows; i++)
                outptr[i] = ptr[i * stride];
        return;
    }
    if (rows == 1 && outstride == 1)
    {
        memcpy(outptr, ptr, (size_t)cols * sizeof(float));
        return;
    }

    int i = 0;
#if __SSE2__
#if __AVX__
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
            permute_transpose8x16_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose4x16_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 1 < cols; j += 2)
        {
            permute_transpose2x16_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
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
    for (; i + 7 < rows; i += 8)
    {
        int j = 0;
#if __AVX512F__
        for (; j + 15 < cols; j += 16)
        {
            permute_transpose16x8_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
#endif // __AVX512F__
        for (; j + 7 < cols; j += 8)
        {
            permute_transpose8x8_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose4x8_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 1 < cols; j += 2)
        {
            permute_transpose2x8_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
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
    for (; i + 3 < rows; i += 4)
    {
        int j = 0;
#if __AVX__
#if __AVX512F__
        for (; j + 15 < cols; j += 16)
        {
            permute_transpose16x4_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
#endif // __AVX512F__
        for (; j + 7 < cols; j += 8)
        {
            permute_transpose8x4_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
#endif // __AVX__
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose4x4_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 1 < cols; j += 2)
        {
            permute_transpose2x4_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j < cols; j++)
        {
            for (int k = 0; k < 4; k++)
            {
                outptr[j * outstride + i + k] = ptr[(i + k) * stride + j];
            }
        }
    }
    for (; i + 1 < rows; i += 2)
    {
        int j = 0;
#if __AVX__
#if __AVX512F__
        for (; j + 15 < cols; j += 16)
        {
            permute_transpose16x2_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
#endif // __AVX512F__
        for (; j + 7 < cols; j += 8)
        {
            permute_transpose8x2_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
#endif // __AVX__
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose4x2_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
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
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; j + 15 < cols; j += 16)
        {
            permute_transpose16x1_stride(ptr + i * stride + j, outptr + j * outstride + i, outstride);
        }
#endif // __AVX512F__
        for (; j + 7 < cols; j += 8)
        {
            permute_transpose8x1_stride(ptr + i * stride + j, outptr + j * outstride + i, outstride);
        }
#endif // __AVX__
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose4x1_stride(ptr + i * stride + j, outptr + j * outstride + i, outstride);
        }
#endif // __SSE2__
        for (; j < cols; j++)
        {
            outptr[j * outstride + i] = ptr[i * stride + j];
        }
    }
}

// 2d: packed rows become packed output rows after transposing w and h
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
#endif // __AVX__

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

#if __AVX__
static void permute_transpose_pack4to8(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        for (int x = 0; x < cols; x += 8)
        {
            const float* p = ptr + (y / 4) * stride + x * 4;
            float* out = outptr + (x / 8) * outstride + y * 8;
            permute_transpose4x8(p, out);
        }
    }
}

#if __AVX512F__
static void permute_transpose_pack4to16(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        for (int x = 0; x < cols; x += 16)
        {
            const float* p = ptr + (y / 4) * stride + x * 4;
            float* out = outptr + (x / 16) * outstride + y * 16;
            permute_transpose4x16_stride(p, 4, out, 16);
        }
    }
}
#endif // __AVX512F__

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

static void permute_transpose_pack8to4(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        for (int x = 0; x < cols; x += 4)
        {
            const float* p = ptr + (y / 8) * stride + x * 8;
            float* out = outptr + (x / 4) * outstride + y * 4;
            permute_transpose8x4(p, out);
        }
    }
}

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

#if __AVX512F__
static void permute_transpose_pack8to16(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        for (int x = 0; x < cols; x += 16)
        {
            const float* p = ptr + (y / 8) * stride + x * 8;
            float* out = outptr + (x / 16) * outstride + y * 16;
            permute_transpose8x16_stride(p, 8, out, 16);
        }
    }
}

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

static void permute_transpose_pack16to4(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 16)
    {
        for (int x = 0; x < cols; x += 4)
        {
            const float* p = ptr + (y / 16) * stride + x * 16;
            float* out = outptr + (x / 4) * outstride + y * 4;
            permute_transpose16x4(p, out);
        }
    }
}

static void permute_transpose_pack16to8(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 16)
    {
        for (int x = 0; x < cols; x += 8)
        {
            const float* p = ptr + (y / 16) * stride + x * 16;
            float* out = outptr + (x / 8) * outstride + y * 8;
            permute_transpose16x8_stride(p, 16, out, 8);
        }
    }
}

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
#endif // __AVX__
#endif // __SSE2__

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

#if __AVX__
    if (elempack == 1 && out_elempack == 8)
    {
        permute_transpose_pack1to8(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

#if __AVX512F__
    if (elempack == 1 && out_elempack == 16)
    {
        permute_transpose_pack1to16(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__
#endif // __AVX__

    if (elempack == 4 && out_elempack == 1)
    {
        permute_transpose_pack4to1(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

    if (elempack == 4 && out_elempack == 4)
    {
        permute_transpose_pack4to4(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

#if __AVX__
    if (elempack == 4 && out_elempack == 8)
    {
        permute_transpose_pack4to8(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

#if __AVX512F__
    if (elempack == 4 && out_elempack == 16)
    {
        permute_transpose_pack4to16(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__

    if (elempack == 8 && out_elempack == 1)
    {
        permute_transpose_pack8to1(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

    if (elempack == 8 && out_elempack == 4)
    {
        permute_transpose_pack8to4(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

    if (elempack == 8 && out_elempack == 8)
    {
        permute_transpose_pack8to8(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

#if __AVX512F__
    if (elempack == 8 && out_elempack == 16)
    {
        permute_transpose_pack8to16(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

    if (elempack == 16 && out_elempack == 1)
    {
        permute_transpose_pack16to1(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

    if (elempack == 16 && out_elempack == 4)
    {
        permute_transpose_pack16to4(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

    if (elempack == 16 && out_elempack == 8)
    {
        permute_transpose_pack16to8(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

    if (elempack == 16 && out_elempack == 16)
    {
        permute_transpose_pack16to16(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
}

// spatial transpose within one input channel group
// outcstep is used when unpacking
#if __SSE2__
static NCNN_FORCEINLINE void permute_spatial2x2_pack4_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
#if __AVX__
    __m256 _a = _mm256_loadu_ps(ptr);
    __m256 _b = _mm256_loadu_ps(ptr + stride);
    _mm256_storeu_ps(outptr, _mm256_permute2f128_ps(_a, _b, 0x20));
    _mm256_storeu_ps(outptr + outstride, _mm256_permute2f128_ps(_a, _b, 0x31));
#else
    __m128i _v0 = _mm_loadu_si128((const __m128i*)ptr);
    _mm_storeu_si128((__m128i*)outptr, _v0);
    __m128i _v1 = _mm_loadu_si128((const __m128i*)(ptr + stride));
    _mm_storeu_si128((__m128i*)(outptr + 4), _v1);
    __m128i _v2 = _mm_loadu_si128((const __m128i*)(ptr + 4));
    _mm_storeu_si128((__m128i*)(outptr + outstride), _v2);
    __m128i _v3 = _mm_loadu_si128((const __m128i*)(ptr + stride + 4));
    _mm_storeu_si128((__m128i*)(outptr + outstride + 4), _v3);
#endif
}

static void permute_spatial_pack4(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    int i = 0;
    for (; i + 1 < rows; i += 2)
    {
        int j = 0;
        for (; j + 1 < cols; j += 2)
        {
            permute_spatial2x2_pack4_stride(ptr + i * stride + j * 4, stride, outptr + j * outstride + i * 4, outstride);
        }
        for (; j < cols; j++)
        {
            __m128i _v0 = _mm_loadu_si128((const __m128i*)(ptr + i * stride + j * 4));
            _mm_storeu_si128((__m128i*)(outptr + j * outstride + i * 4), _v0);
            __m128i _v1 = _mm_loadu_si128((const __m128i*)(ptr + (i + 1) * stride + j * 4));
            _mm_storeu_si128((__m128i*)(outptr + j * outstride + (i + 1) * 4), _v1);
        }
    }
    for (; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            __m128i _v0 = _mm_loadu_si128((const __m128i*)(ptr + i * stride + j * 4));
            _mm_storeu_si128((__m128i*)(outptr + j * outstride + i * 4), _v0);
        }
    }
}

static void permute_spatial_pack4to1(const float* ptr, size_t stride, float* outptr, size_t outstride, size_t outcstep, int rows, int cols)
{
#if __AVX512F__
    int i = 0;
    for (; i + 15 < rows; i += 16)
    {
        for (int j = 0; j < cols; j++)
        {
            permute_transpose4x16_stride(ptr + i * stride + j * 4, stride, outptr + j * outstride + i, outcstep);
        }
    }
    for (; i + 7 < rows; i += 8)
    {
        for (int j = 0; j < cols; j++)
        {
            permute_transpose4x8_stride(ptr + i * stride + j * 4, stride, outptr + j * outstride + i, outcstep);
        }
    }
    for (; i + 3 < rows; i += 4)
    {
        for (int j = 0; j < cols; j++)
        {
            permute_transpose4x4_stride(ptr + i * stride + j * 4, stride, outptr + j * outstride + i, outcstep);
        }
    }
    for (; i + 1 < rows; i += 2)
    {
        for (int j = 0; j < cols; j++)
        {
            permute_transpose4x2_stride(ptr + i * stride + j * 4, stride, outptr + j * outstride + i, outcstep);
        }
    }
    for (; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            permute_transpose4x1_stride(ptr + i * stride + j * 4, outptr + j * outstride + i, outcstep);
        }
    }
#else
    for (int x = 0; x < cols; x++)
        permute_unpack4_stride(ptr + x * 4, stride, outptr + x * outstride, outcstep, rows);
#endif // __AVX512F__
}

#if __AVX__
static NCNN_FORCEINLINE void permute_spatial2x2_pack8_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
#if __AVX512F__
    __m512 _a = _mm512_loadu_ps(ptr);
    __m512 _b = _mm512_loadu_ps(ptr + stride);
    _mm512_storeu_ps(outptr, _mm512_shuffle_f32x4(_a, _b, 0x44));
    _mm512_storeu_ps(outptr + outstride, _mm512_shuffle_f32x4(_a, _b, 0xee));
#else
    __m256 _v0 = _mm256_loadu_ps(ptr);
    _mm256_storeu_ps(outptr, _v0);
    __m256 _v1 = _mm256_loadu_ps(ptr + stride);
    _mm256_storeu_ps(outptr + 8, _v1);
    __m256 _v2 = _mm256_loadu_ps(ptr + 8);
    _mm256_storeu_ps(outptr + outstride, _v2);
    __m256 _v3 = _mm256_loadu_ps(ptr + stride + 8);
    _mm256_storeu_ps(outptr + outstride + 8, _v3);
#endif
}

static void permute_spatial_pack8(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    int i = 0;
    for (; i + 1 < rows; i += 2)
    {
        int j = 0;
        for (; j + 1 < cols; j += 2)
        {
            permute_spatial2x2_pack8_stride(ptr + i * stride + j * 8, stride, outptr + j * outstride + i * 8, outstride);
        }
        for (; j < cols; j++)
        {
            __m256 _v0 = _mm256_loadu_ps(ptr + i * stride + j * 8);
            _mm256_storeu_ps(outptr + j * outstride + i * 8, _v0);
            __m256 _v1 = _mm256_loadu_ps(ptr + (i + 1) * stride + j * 8);
            _mm256_storeu_ps(outptr + j * outstride + (i + 1) * 8, _v1);
        }
    }
    for (; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            __m256 _v0 = _mm256_loadu_ps(ptr + i * stride + j * 8);
            _mm256_storeu_ps(outptr + j * outstride + i * 8, _v0);
        }
    }
}

static void permute_spatial_pack8to1(const float* ptr, size_t stride, float* outptr, size_t outstride, size_t outcstep, int rows, int cols)
{
#if __AVX512F__
    // limit the number of output streams for large channel planes
    if (rows >= 128 && stride >= 8 * 512 && outstride >= 512)
    {
        for (int x = 0; x < cols; x++)
            permute_unpack8_stride(ptr + x * 8, stride, outptr + x * outstride, outcstep, rows);
        return;
    }

    int i = 0;
    for (; i + 15 < rows; i += 16)
    {
        for (int j = 0; j < cols; j++)
        {
            permute_transpose8x16_stride(ptr + i * stride + j * 8, stride, outptr + j * outstride + i, outcstep);
        }
    }
    for (; i + 7 < rows; i += 8)
    {
        for (int j = 0; j < cols; j++)
        {
            permute_transpose8x8_stride(ptr + i * stride + j * 8, stride, outptr + j * outstride + i, outcstep);
        }
    }
    for (; i + 3 < rows; i += 4)
    {
        for (int j = 0; j < cols; j++)
        {
            permute_transpose8x4_stride(ptr + i * stride + j * 8, stride, outptr + j * outstride + i, outcstep);
        }
    }
    for (; i + 1 < rows; i += 2)
    {
        for (int j = 0; j < cols; j++)
        {
            permute_transpose8x2_stride(ptr + i * stride + j * 8, stride, outptr + j * outstride + i, outcstep);
        }
    }
    for (; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            permute_transpose8x1_stride(ptr + i * stride + j * 8, outptr + j * outstride + i, outcstep);
        }
    }
#else
    for (int x = 0; x < cols; x++)
        permute_unpack8_stride(ptr + x * 8, stride, outptr + x * outstride, outcstep, rows);
#endif // __AVX512F__
}

#if __AVX512F__
static NCNN_FORCEINLINE void permute_spatial2x2_pack16_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    __m512 _v0 = _mm512_loadu_ps(ptr);
    _mm512_storeu_ps(outptr, _v0);
    __m512 _v1 = _mm512_loadu_ps(ptr + stride);
    _mm512_storeu_ps(outptr + 16, _v1);
    __m512 _v2 = _mm512_loadu_ps(ptr + 16);
    _mm512_storeu_ps(outptr + outstride, _v2);
    __m512 _v3 = _mm512_loadu_ps(ptr + stride + 16);
    _mm512_storeu_ps(outptr + outstride + 16, _v3);
}

static void permute_spatial_pack16(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    int i = 0;
    for (; i + 1 < rows; i += 2)
    {
        int j = 0;
        for (; j + 1 < cols; j += 2)
        {
            permute_spatial2x2_pack16_stride(ptr + i * stride + j * 16, stride, outptr + j * outstride + i * 16, outstride);
        }
        for (; j < cols; j++)
        {
            __m512 _v0 = _mm512_loadu_ps(ptr + i * stride + j * 16);
            _mm512_storeu_ps(outptr + j * outstride + i * 16, _v0);
            __m512 _v1 = _mm512_loadu_ps(ptr + (i + 1) * stride + j * 16);
            _mm512_storeu_ps(outptr + j * outstride + (i + 1) * 16, _v1);
        }
    }
    for (; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            __m512 _v0 = _mm512_loadu_ps(ptr + i * stride + j * 16);
            _mm512_storeu_ps(outptr + j * outstride + i * 16, _v0);
        }
    }
}

static void permute_spatial_pack16to1(const float* ptr, size_t stride, float* outptr, size_t outstride, size_t outcstep, int rows, int cols)
{
    int i = 0;
    for (; i + 15 < rows; i += 16)
    {
        for (int j = 0; j < cols; j++)
        {
            permute_transpose16x16_stride(ptr + i * stride + j * 16, stride, outptr + j * outstride + i, outcstep);
        }
    }
    for (; i + 7 < rows; i += 8)
    {
        for (int j = 0; j < cols; j++)
        {
            permute_transpose16x8_stride(ptr + i * stride + j * 16, stride, outptr + j * outstride + i, outcstep);
        }
    }
    for (; i + 3 < rows; i += 4)
    {
        for (int j = 0; j < cols; j++)
        {
            permute_transpose16x4_stride(ptr + i * stride + j * 16, stride, outptr + j * outstride + i, outcstep);
        }
    }
    for (; i + 1 < rows; i += 2)
    {
        for (int j = 0; j < cols; j++)
        {
            permute_transpose16x2_stride(ptr + i * stride + j * 16, stride, outptr + j * outstride + i, outcstep);
        }
    }
    for (; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            permute_transpose16x1_stride(ptr + i * stride + j * 16, outptr + j * outstride + i, outcstep);
        }
    }
}
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

static void permute_transpose_spatial(const float* ptr, size_t stride, float* outptr, size_t outstride, size_t outcstep, int rows, int cols, int elempack, int out_elempack)
{
    if (elempack == out_elempack)
    {
        if (rows == 1 && outstride == (size_t)elempack)
        {
            memcpy(outptr, ptr, (size_t)cols * elempack * sizeof(float));
            return;
        }
        if (cols == 1 && stride == (size_t)elempack)
        {
            memcpy(outptr, ptr, (size_t)rows * elempack * sizeof(float));
            return;
        }
    }

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
        permute_spatial_pack4to1(ptr, stride, outptr, outstride, outcstep, rows, cols);
        return;
    }

#if __AVX__
    if (elempack == 8 && out_elempack == 8)
    {
        permute_spatial_pack8(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
    if (elempack == 8 && out_elempack == 1)
    {
        permute_spatial_pack8to1(ptr, stride, outptr, outstride, outcstep, rows, cols);
        return;
    }

#if __AVX512F__
    if (elempack == 16 && out_elempack == 16)
    {
        permute_spatial_pack16(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
    if (elempack == 16 && out_elempack == 1)
    {
        permute_spatial_pack16to1(ptr, stride, outptr, outstride, outcstep, rows, cols);
        return;
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
}

#if __SSE2__
static void permute_transpose_blocks2(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    int j = 0;
    for (; j + 1 < cols; j += 2)
    {
        const float* p = ptr + (size_t)j * 2;
        float* out0 = outptr + j * outstride;
        float* out1 = out0 + outstride;
        int i = 0;
        for (; i + 1 < rows; i += 2)
        {
            __m128 _a = _mm_loadu_ps(p);
            __m128 _b = _mm_loadu_ps(p + stride);
            _mm_storeu_ps(out0, _mm_movelh_ps(_a, _b));
            _mm_storeu_ps(out1, _mm_movehl_ps(_b, _a));
            p += stride * 2;
            out0 += 4;
            out1 += 4;
        }
        for (; i < rows; i++)
        {
            memcpy(out0, p, 2 * sizeof(float));
            memcpy(out1, p + 2, 2 * sizeof(float));
            p += stride;
            out0 += 2;
            out1 += 2;
        }
    }
    for (; j < cols; j++)
    {
        const float* p = ptr + (size_t)j * 2;
        float* out = outptr + j * outstride;
        for (int i = 0; i < rows; i++)
        {
            memcpy(out, p, 2 * sizeof(float));
            p += stride;
            out += 2;
        }
    }
}
#endif // __SSE2__

// transpose rows of contiguous blocks
// size is independent of elempack
// strides include padding; the block contents keep their original order
static void permute_transpose_blocks(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols, int size)
{
    if (rows == 1 && outstride == (size_t)size)
    {
        memcpy(outptr, ptr, (size_t)cols * size * sizeof(float));
        return;
    }
    if (cols == 1 && stride == (size_t)size)
    {
        memcpy(outptr, ptr, (size_t)rows * size * sizeof(float));
        return;
    }

    if (size == 1)
    {
        permute_transpose_pack1(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#if __SSE2__
    if (size == 2)
    {
        permute_transpose_blocks2(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
    if (size == 4)
    {
        permute_spatial_pack4(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#if __AVX__
    if (size == 8)
    {
        permute_spatial_pack8(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#if __AVX512F__
    if (size == 16)
    {
        permute_spatial_pack16(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__
#endif // __AVX__
    if (size >= 2 && size < 4)
    {
        for (int j = 0; j < cols; j++)
        {
            const float* p = ptr + (size_t)j * size;
            float* out = outptr + j * outstride;
            for (int i = 0; i < rows; i++)
            {
                __m128i _a = _mm_loadl_epi64((const __m128i*)p);
                __m128i _b = _mm_loadl_epi64((const __m128i*)(p + size - 2));
                _mm_storel_epi64((__m128i*)out, _a);
                _mm_storel_epi64((__m128i*)(out + size - 2), _b);
                p += stride;
                out += size;
            }
        }
        return;
    }
    if (size >= 4 && size < 8)
    {
        for (int j = 0; j < cols; j++)
        {
            const float* p = ptr + (size_t)j * size;
            float* out = outptr + j * outstride;
            for (int i = 0; i < rows; i++)
            {
                __m128i _a = _mm_loadu_si128((const __m128i*)p);
                __m128i _b = _mm_loadu_si128((const __m128i*)(p + size - 4));
                _mm_storeu_si128((__m128i*)out, _a);
                _mm_storeu_si128((__m128i*)(out + size - 4), _b);
                p += stride;
                out += size;
            }
        }
        return;
    }
#if __AVX__
    if (size >= 8 && size < 16)
    {
        for (int j = 0; j < cols; j++)
        {
            const float* p = ptr + (size_t)j * size;
            float* out = outptr + j * outstride;
            for (int i = 0; i < rows; i++)
            {
                __m256i _a = _mm256_loadu_si256((const __m256i*)p);
                __m256i _b = _mm256_loadu_si256((const __m256i*)(p + size - 8));
                _mm256_storeu_si256((__m256i*)out, _a);
                _mm256_storeu_si256((__m256i*)(out + size - 8), _b);
                p += stride;
                out += size;
            }
        }
        return;
    }
#if __AVX512F__
    if (size >= 16 && size < 32)
    {
        for (int j = 0; j < cols; j++)
        {
            const float* p = ptr + (size_t)j * size;
            float* out = outptr + j * outstride;
            for (int i = 0; i < rows; i++)
            {
                __m512i _a = _mm512_loadu_si512((const void*)p);
                __m512i _b = _mm512_loadu_si512((const void*)(p + size - 16));
                _mm512_storeu_si512((void*)out, _a);
                _mm512_storeu_si512((void*)(out + size - 16), _b);
                p += stride;
                out += size;
            }
        }
        return;
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
    for (int j = 0; j < cols; j++)
    {
        const float* p = ptr + (size_t)j * size;
        float* out = outptr + j * outstride;
        for (int i = 0; i < rows; i++)
        {
            memcpy(out, p, (size_t)size * sizeof(float));
            p += stride;
            out += size;
        }
    }
}

static void permute_unpack_spatial(const float* ptr, float* outptr, size_t outcstep, int size, int elempack)
{
#if __SSE2__
    if (elempack == 4)
    {
        permute_unpack4_stride(ptr, 4, outptr, outcstep, size);
        return;
    }

#if __AVX__
    if (elempack == 8)
    {
        permute_unpack8_stride(ptr, 8, outptr, outcstep, size);
        return;
    }

#if __AVX512F__
    if (elempack == 16)
    {
        permute_unpack16_stride(ptr, 16, outptr, outcstep, size);
        return;
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
}

// exchange the input channel axis with one output channel group
// w is the remaining spatial extent; cstep includes input channel padding
#if __SSE2__
// the exchanged axis is contiguous; the stride variant has contiguous spatial input
static void permute3d_pack1to4(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * outcstep;
        for (int x = 0; x < w; x++)
        {
            __m128i _v = _mm_loadu_si128((const __m128i*)p);
            _mm_storeu_si128((__m128i*)out, _v);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack1to4_stride(const float* ptr, float* outptr, int w, int channels, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * outcstep;
        permute_pack4_stride(p, hstep, out, outwstep, w);
    }
}

#if __AVX__
// the exchanged axis is contiguous; the stride variant has contiguous spatial input
static void permute3d_pack1to8(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * outcstep;
        for (int x = 0; x < w; x++)
        {
            __m256 _v = _mm256_loadu_ps((const float*)p);
            _mm256_storeu_ps((float*)out, _v);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack1to8_stride(const float* ptr, float* outptr, int w, int channels, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * outcstep;
        permute_pack8_stride(p, hstep, out, outwstep, w);
    }
}

#if __AVX512F__
// the exchanged axis is contiguous; the stride variant has contiguous spatial input
static void permute3d_pack1to16(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * outcstep;
        for (int x = 0; x < w; x++)
        {
            __m512 _v = _mm512_loadu_ps(p);
            _mm512_storeu_ps(out, _v);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack1to16_stride(const float* ptr, float* outptr, int w, int channels, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * outcstep;
        permute_pack16_stride(p, hstep, out, outwstep, w);
    }
}
#endif // __AVX512F__
#endif // __AVX__

// output channels are contiguous; the stride variant has contiguous spatial output
static void permute3d_pack4to1(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 4;
        for (int x = 0; x < w; x++)
        {
            __m128i _v = _mm_loadu_si128((const __m128i*)p);
            _mm_storeu_si128((__m128i*)out, _v);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack4to1_stride(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 4 * outcstep;
        permute_unpack4_stride(p, wstep, out, outcstep, w);
    }
}

static void permute3d_pack4to4(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 4 * 4;
        for (int x = 0; x < w; x++)
        {
            permute_transpose4x4(p, out);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack4to4_stride(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 4 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose4x4_stride(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}

#if __AVX__
static void permute3d_pack4to8(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 4 * 8;
        for (int x = 0; x < w; x++)
        {
            permute_transpose4x8(p, out);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack4to8_stride(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 4 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose4x8_stride(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}

#if __AVX512F__
static void permute3d_pack4to16_stride(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 4 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose4x16_stride(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}
#endif // __AVX512F__

// output channels are contiguous; the stride variant has contiguous spatial output
static void permute3d_pack8to1(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 8;
        for (int x = 0; x < w; x++)
        {
            __m256 _v = _mm256_loadu_ps((const float*)p);
            _mm256_storeu_ps((float*)out, _v);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack8to1_stride(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 8 * outcstep;
        permute_unpack8_stride(p, wstep, out, outcstep, w);
    }
}

static void permute3d_pack8to4(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 8 * 4;
        for (int x = 0; x < w; x++)
        {
            permute_transpose8x4(p, out);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack8to4_stride(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 8 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose8x4_stride(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack8to8_stride(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 8 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose8x8_stride(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}

#if __AVX512F__
static void permute3d_pack8to16_stride(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 8 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose8x16_stride(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}

// output channels are contiguous; the stride variant has contiguous spatial output
static void permute3d_pack16to1(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 16;
        for (int x = 0; x < w; x++)
        {
            __m512 _v = _mm512_loadu_ps(p);
            _mm512_storeu_ps(out, _v);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack16to1_stride(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 16 * outcstep;
        permute_unpack16_stride(p, wstep, out, outcstep, w);
    }
}

static void permute3d_pack16to4(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 16 * 4;
        for (int x = 0; x < w; x++)
        {
            permute_transpose16x4(p, out);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack16to4_stride(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 16 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose16x4_stride(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack16to8_stride(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 16 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose16x8_stride(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack16to16_stride(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 16 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose16x16_stride(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

// the exchanged input axis is contiguous
static void permute_pack_channels(const Mat& bottom_blob, const Mat& top_blob, const float* ptr, float* outptr, int size, int channels, size_t wstep, size_t outwstep, size_t outcstep)
{
    const size_t cstep = bottom_blob.cstep * bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

#if __SSE2__
    if (out_elempack == 4)
    {
        permute3d_pack1to4(ptr, outptr, size, channels, wstep, cstep, outwstep, outcstep);
        return;
    }
#if __AVX__
    if (out_elempack == 8)
    {
        permute3d_pack1to8(ptr, outptr, size, channels, wstep, cstep, outwstep, outcstep);
        return;
    }
#if __AVX512F__
    if (out_elempack == 16)
    {
        permute3d_pack1to16(ptr, outptr, size, channels, wstep, cstep, outwstep, outcstep);
        return;
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
}

// the remaining input spatial axis is contiguous
static void permute_pack_channels_stride(const Mat& bottom_blob, const Mat& top_blob, const float* ptr, float* outptr, int size, int channels, size_t hstep, size_t outwstep, size_t outcstep)
{
    const size_t cstep = bottom_blob.cstep * bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

#if __SSE2__
    if (out_elempack == 4)
    {
        permute3d_pack1to4_stride(ptr, outptr, size, channels, hstep, cstep, outwstep, outcstep);
        return;
    }
#if __AVX__
    if (out_elempack == 8)
    {
        permute3d_pack1to8_stride(ptr, outptr, size, channels, hstep, cstep, outwstep, outcstep);
        return;
    }
#if __AVX512F__
    if (out_elempack == 16)
    {
        permute3d_pack1to16_stride(ptr, outptr, size, channels, hstep, cstep, outwstep, outcstep);
        return;
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
}

// the output channel axis is contiguous
static void permute_unpack_channels(const Mat& bottom_blob, const float* ptr, float* outptr, int size, int channels, size_t wstep, size_t outwstep)
{
    const size_t cstep = bottom_blob.cstep * bottom_blob.elempack;
    const int elempack = bottom_blob.elempack;

#if __SSE2__
    if (elempack == 4)
    {
        permute3d_pack4to1(ptr, outptr, size, channels, wstep, cstep, outwstep);
        return;
    }
#if __AVX__
    if (elempack == 8)
    {
        permute3d_pack8to1(ptr, outptr, size, channels, wstep, cstep, outwstep);
        return;
    }
#if __AVX512F__
    if (elempack == 16)
    {
        permute3d_pack16to1(ptr, outptr, size, channels, wstep, cstep, outwstep);
        return;
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
}

// the remaining output spatial axis is contiguous
static void permute_unpack_channels_stride(const Mat& bottom_blob, const float* ptr, float* outptr, int size, int channels, size_t wstep, size_t outcstep)
{
    const size_t cstep = bottom_blob.cstep * bottom_blob.elempack;
    const int elempack = bottom_blob.elempack;

#if __SSE2__
    if (elempack == 4)
    {
        permute3d_pack4to1_stride(ptr, outptr, size, channels, wstep, cstep, outcstep);
        return;
    }
#if __AVX__
    if (elempack == 8)
    {
        permute3d_pack8to1_stride(ptr, outptr, size, channels, wstep, cstep, outcstep);
        return;
    }
#if __AVX512F__
    if (elempack == 16)
    {
        permute3d_pack16to1_stride(ptr, outptr, size, channels, wstep, cstep, outcstep);
        return;
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
}

// both channel axes are packed
static void permute_channels_spatial_input_stride(const Mat& bottom_blob, const Mat& top_blob, const float* ptr, float* outptr, int size, int channels, size_t hstep, size_t outwstep, size_t outcstep)
{
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t wstep = elempack;

#if __SSE2__
    if (elempack == 4 && out_elempack == 4)
    {
        if (hstep == 4 && outcstep == 4)
        {
            permute3d_pack4to4(ptr, outptr, size, channels, wstep, cstep, outwstep);
        }
        else
        {
            permute3d_pack4to4_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#if __AVX__
    if (elempack == 4 && out_elempack == 8)
    {
        if (hstep == 4 && outcstep == 8)
        {
            permute3d_pack4to8(ptr, outptr, size, channels, wstep, cstep, outwstep);
        }
        else
        {
            permute3d_pack4to8_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#if __AVX512F__
    if (elempack == 4 && out_elempack == 16)
    {
        permute3d_pack4to16_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
#endif // __AVX512F__
    if (elempack == 8 && out_elempack == 4)
    {
        if (hstep == 8 && outcstep == 4)
        {
            permute3d_pack8to4(ptr, outptr, size, channels, wstep, cstep, outwstep);
        }
        else
        {
            permute3d_pack8to4_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
    if (elempack == 8 && out_elempack == 8)
    {
        permute3d_pack8to8_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
#if __AVX512F__
    if (elempack == 8 && out_elempack == 16)
    {
        permute3d_pack8to16_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
    if (elempack == 16 && out_elempack == 4)
    {
        if (hstep == 16 && outcstep == 4)
        {
            permute3d_pack16to4(ptr, outptr, size, channels, wstep, cstep, outwstep);
        }
        else
        {
            permute3d_pack16to4_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
    if (elempack == 16 && out_elempack == 8)
    {
        permute3d_pack16to8_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
    if (elempack == 16 && out_elempack == 16)
    {
        permute3d_pack16to16_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
}

static void permute_channels_axis_input_stride(const Mat& bottom_blob, const Mat& top_blob, const float* ptr, float* outptr, int size, int channels, size_t wstep, size_t outwstep, size_t outcstep)
{
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t hstep = elempack;

#if __SSE2__
    if (elempack == 4 && out_elempack == 4)
    {
        if (hstep == 4 && outcstep == 4)
        {
            permute3d_pack4to4(ptr, outptr, size, channels, wstep, cstep, outwstep);
        }
        else
        {
            permute3d_pack4to4_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#if __AVX__
    if (elempack == 4 && out_elempack == 8)
    {
        if (hstep == 4 && outcstep == 8)
        {
            permute3d_pack4to8(ptr, outptr, size, channels, wstep, cstep, outwstep);
        }
        else
        {
            permute3d_pack4to8_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#if __AVX512F__
    if (elempack == 4 && out_elempack == 16)
    {
        permute3d_pack4to16_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
#endif // __AVX512F__
    if (elempack == 8 && out_elempack == 4)
    {
        if (hstep == 8 && outcstep == 4)
        {
            permute3d_pack8to4(ptr, outptr, size, channels, wstep, cstep, outwstep);
        }
        else
        {
            permute3d_pack8to4_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
    if (elempack == 8 && out_elempack == 8)
    {
        permute3d_pack8to8_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
#if __AVX512F__
    if (elempack == 8 && out_elempack == 16)
    {
        permute3d_pack8to16_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
    if (elempack == 16 && out_elempack == 4)
    {
        if (hstep == 16 && outcstep == 4)
        {
            permute3d_pack16to4(ptr, outptr, size, channels, wstep, cstep, outwstep);
        }
        else
        {
            permute3d_pack16to4_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
    if (elempack == 16 && out_elempack == 8)
    {
        permute3d_pack16to8_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
    if (elempack == 16 && out_elempack == 16)
    {
        permute3d_pack16to16_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
}

static void permute_transpose_matrix(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols, int size, int nT)
{
    (void)nT;

    // scalar matrices keep the full register transpose width
#if __AVX512F__
    const int step = size == 1 ? 16 : 1;
#elif __AVX__
    const int step = size == 1 ? 8 : 1;
#elif __SSE2__
    const int step = size == 1 ? 4 : 1;
#else
    const int step = 1;
#endif
    #pragma omp parallel for num_threads(nT)
    for (int j = 0; j < cols; j += step)
    {
        permute_transpose_blocks(ptr + (size_t)j * size, stride, outptr + j * outstride, outstride, rows, std::min(step, cols - j), size);
    }
}

// transpose independent matrices
static void permute_transpose_matrices_stride(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols, int size, int planes, size_t step, size_t outstep, int nT)
{
    if (planes == 1)
    {
        permute_transpose_matrix(ptr, stride, outptr, outstride, rows, cols, size, nT);
        return;
    }

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < planes; q++)
    {
        permute_transpose_blocks(ptr + q * step, stride, outptr + q * outstep, outstride, rows, cols, size);
    }
}

// transpose spatial planes within each input channel group
static void permute_transpose_spatial_planes_stride(const Mat& bottom_blob, Mat& top_blob, int rows, int cols, size_t stride, size_t outstride, int planes, size_t step, size_t outstep, int nT)
{
    (void)nT;

    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q * elempack / out_elempack);
        for (int z = 0; z < planes; z++)
        {
            permute_transpose_spatial(ptr + z * step, stride, outptr + z * outstep, outstride, top_blob.cstep, rows, cols, elempack, out_elempack);
        }
    }
}

// transpose h and w while preserving the other two slice axes
static void permute_transpose_hw_pack1_stride(const Mat& bottom_blob, Mat& top_blob, size_t outstride, size_t outcstep, size_t outdstep, int nT)
{
    (void)nT;

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const size_t stride = (size_t)w;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = (float*)top_blob + q * outcstep;
        for (int z = 0; z < d; z++)
        {
            permute_transpose_pack1(ptr + z * (size_t)w * h, stride, outptr + z * outdstep, outstride, h, w);
        }
    }
}

// transpose d and w while preserving the other two slice axes
static void permute_transpose_dw_pack1_stride(const Mat& bottom_blob, Mat& top_blob, size_t outstride, size_t outcstep, size_t outhstep, int nT)
{
    (void)nT;

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const size_t stride = (size_t)w * h;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = (float*)top_blob + q * outcstep;
        for (int y = 0; y < h; y++)
        {
            permute_transpose_pack1(ptr + y * (size_t)w, stride, outptr + y * outhstep, outstride, d, w);
        }
    }
}

// transpose c and w while preserving the other two slice axes
static void permute_transpose_cw_pack1_stride(const Mat& bottom_blob, Mat& top_blob, size_t outstride, size_t outdstep, size_t outhstep, int nT)
{
    (void)nT;

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const size_t stride = bottom_blob.cstep;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        const float* ptr = (const float*)bottom_blob + z * (size_t)w * h;
        float* outptr = (float*)top_blob + z * outdstep;
        for (int y = 0; y < h; y++)
        {
            permute_transpose_pack1(ptr + y * (size_t)w, stride, outptr + y * outhstep, outstride, channels, w);
        }
    }
}

static void permute2d(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        permute_transpose_matrix(bottom_blob, w, top_blob, h, h, w, 1, nT);
        return;
    }

    #pragma omp parallel for num_threads(nT)
    for (int x = 0; x < w; x += out_elempack)
    {
        const float* ptr = (const float*)bottom_blob + (size_t)x * elempack;
        float* outptr = top_blob.row<float>(x / out_elempack);
        permute_transpose2d(ptr, (size_t)w * elempack, outptr, (size_t)top_blob.w * out_elempack, h * elempack, out_elempack, elempack, out_elempack);
    }
}

static void permute3d_hwc(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t stride = (size_t)w * elempack;
    const size_t outstride = (size_t)h * out_elempack;
    const size_t step = 0;
    const size_t outstep = 0;
    permute_transpose_spatial_planes_stride(bottom_blob, top_blob, h, w, stride, outstride, 1, step, outstep, nT);
}

static void permute3d_wch(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        if (w < 32)
        {
            // short blocks include the w == 1 matrix-transpose case
            permute_transpose_matrix(bottom_blob, bottom_blob.cstep, top_blob, top_blob.cstep, channels, h, w, nT);
            return;
        }

        #pragma omp parallel for num_threads(nT)
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr = (const float*)bottom_blob + q * (size_t)w;
            float* outptr = top_blob.channel(q);
            for (int y = 0; y < top_blob.h; y++)
            {
                memcpy(outptr + (size_t)y * w, ptr + y * bottom_blob.cstep, (size_t)w * sizeof(float));
            }
        }
        return;
    }

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < top_blob.c; q++)
    {
        const float* ptr = (const float*)bottom_blob + (size_t)q * out_elempack * hstep;
        float* outptr = (float*)top_blob.channel(q);
        // exchange c and h, keeping w as the inner spatial axis
        if (elempack == 1)
            permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        else if (out_elempack == 1)
            permute_unpack_channels_stride(bottom_blob, ptr, outptr, w, channels, wstep, outcstep);
        else
            permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
    }
}

static void permute3d_cwh(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        // w and h stay adjacent in the output; retain input channel padding
        if (top_blob.cstep == (size_t)w * channels && (size_t)w * h <= INT_MAX)
        {
            permute_transpose_matrix(bottom_blob, bottom_blob.cstep, top_blob, channels, channels, w * h, 1, nT);
            return;
        }

        const size_t stride = bottom_blob.cstep;
        const size_t outstride = (size_t)top_blob.w;
        const size_t step = (size_t)w;
        const size_t outstep = top_blob.cstep;
        permute_transpose_matrices_stride(bottom_blob, stride, top_blob, outstride, channels, w, 1, h, step, outstep, nT);
        return;
    }

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t outcstep = out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < top_blob.c; q++)
    {
        const float* ptr = (const float*)bottom_blob + (size_t)q * out_elempack * hstep;
        float* outptr = (float*)top_blob.channel(q);
        // exchange c and h, keeping w as the inner spatial axis
        if (elempack == 1)
            permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        else if (out_elempack == 1)
            permute_unpack_channels(bottom_blob, ptr, outptr, w, channels, wstep, outwstep);
        else
            permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
    }
}

static void permute3d_hcw(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        // c and h stay adjacent in the input; retain output channel padding
        if (bottom_blob.cstep == (size_t)w * h && (size_t)channels * h <= INT_MAX)
        {
            permute_transpose_matrix(bottom_blob, w, top_blob, top_blob.cstep, channels * h, w, 1, nT);
            return;
        }

        const size_t stride = (size_t)w;
        const size_t outstride = top_blob.cstep;
        const size_t step = bottom_blob.cstep;
        const size_t outstep = (size_t)top_blob.w;
        permute_transpose_matrices_stride(bottom_blob, stride, top_blob, outstride, h, w, 1, channels, step, outstep, nT);
        return;
    }

    const size_t wstep = (size_t)w * elempack;
    const size_t hstep = elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < top_blob.c; q++)
    {
        const float* ptr = (const float*)bottom_blob + (size_t)q * out_elempack * hstep;
        float* outptr = (float*)top_blob.channel(q);
        // exchange c and w, keeping h as the inner spatial axis
        if (elempack == 1)
            permute_pack_channels(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
        else if (out_elempack == 1)
            permute_unpack_channels_stride(bottom_blob, ptr, outptr, h, channels, wstep, outcstep);
        else
            permute_channels_axis_input_stride(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
    }
}

static void permute3d_chw(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        // with w or h removed, the remaining operation is one c/spatial transpose
        if (w == 1 || h == 1)
        {
            permute_transpose_matrix(bottom_blob, bottom_blob.cstep, top_blob, w == 1 ? (size_t)channels : top_blob.cstep, channels, w * h, 1, nT);
            return;
        }

        if (channels == 1)
        {
            permute_transpose_matrix(bottom_blob, w, top_blob, top_blob.cstep, h, w, 1, nT);
            return;
        }

        const size_t stride = bottom_blob.cstep;
        const size_t outstride = top_blob.cstep;
        const size_t step = (size_t)w;
        const size_t outstep = (size_t)top_blob.w;
        permute_transpose_matrices_stride(bottom_blob, stride, top_blob, outstride, channels, w, 1, h, step, outstep, nT);
        return;
    }

    const size_t wstep = (size_t)w * elempack;
    const size_t hstep = elempack;
    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t outcstep = out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < top_blob.c; q++)
    {
        const float* ptr = (const float*)bottom_blob + (size_t)q * out_elempack * hstep;
        float* outptr = (float*)top_blob.channel(q);
        // exchange c and w, keeping h as the inner spatial axis
        if (elempack == 1)
            permute_pack_channels(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
        else if (out_elempack == 1)
            permute_unpack_channels(bottom_blob, ptr, outptr, h, channels, wstep, outwstep);
        else
            permute_channels_axis_input_stride(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
    }
}

static void permute4d_hwdc(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t stride = (size_t)w * elempack;
    const size_t outstride = (size_t)h * out_elempack;
    const size_t step = (size_t)w * h * elempack;
    const size_t outstep = (size_t)w * h * out_elempack;
    permute_transpose_spatial_planes_stride(bottom_blob, top_blob, h, w, stride, outstride, d, step, outstep, nT);
}

static void permute4d_wdhc(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == out_elempack)
    {
        const size_t stride = (size_t)w * h * elempack;
        const size_t outstride = (size_t)w * d * elempack;
        const size_t step = bottom_blob.cstep * elempack;
        const size_t outstep = top_blob.cstep * elempack;
        permute_transpose_matrices_stride(bottom_blob, stride, top_blob, outstride, d, h, w * elempack, channels, step, outstep, nT);
        return;
    }

    const size_t hstep = (size_t)w * elempack;
    const size_t dstep = (size_t)w * h * elempack;
    const size_t outhstep = (size_t)w * d * out_elempack;
    const size_t outdstep = (size_t)w * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q * elempack / out_elempack);
        for (int y = 0; y < h; y++)
        {
            const float* p = ptr + y * hstep;
            float* out = outptr + y * outhstep;
            for (int z = 0; z < d; z++)
            {
                permute_unpack_spatial(p + z * dstep, out + z * outdstep, top_blob.cstep, w, elempack);
            }
        }
    }
}

static void permute4d_dwhc(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t stride = (size_t)w * h * elempack;
    const size_t outstride = (size_t)d * out_elempack;
    const size_t step = (size_t)w * elempack;
    const size_t outstep = (size_t)w * d * out_elempack;
    permute_transpose_spatial_planes_stride(bottom_blob, top_blob, d, w, stride, outstride, h, step, outstep, nT);
}

static void permute4d_hdwc(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t stride = (size_t)w * elempack;
    const size_t outstride = (size_t)h * d * out_elempack;
    const size_t step = (size_t)w * h * elempack;
    const size_t outstep = (size_t)h * out_elempack;
    permute_transpose_spatial_planes_stride(bottom_blob, top_blob, h, w, stride, outstride, d, step, outstep, nT);
}

static void permute4d_dhwc(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t stride = (size_t)w * h * elempack;
    const size_t outstride = (size_t)h * d * out_elempack;
    const size_t step = (size_t)w * elempack;
    const size_t outstep = (size_t)d * out_elempack;
    permute_transpose_spatial_planes_stride(bottom_blob, top_blob, d, w, stride, outstride, h, step, outstep, nT);
}

static void permute4d_whcd(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const size_t out_elemsize = top_blob.elemsize;

    // w and h stay adjacent on both sides
    Mat bottom_blob_3d(w * h, d, channels, bottom_blob.data, elemsize, elempack);
    bottom_blob_3d.cstep = bottom_blob.cstep;
    Mat top_blob_3d(w * h, channels * elempack, top_blob.c, top_blob.data, out_elemsize, out_elempack);
    top_blob_3d.cstep = top_blob.cstep;
    permute3d_wch(bottom_blob_3d, top_blob_3d, nT);
}

static void permute4d_hwcd(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w;
        const size_t outcstep = (size_t)top_blob.w * top_blob.h;
        const size_t outdstep = top_blob.cstep;
        permute_transpose_hw_pack1_stride(bottom_blob, top_blob, outstride, outcstep, outdstep, nT);
        return;
    }

    const int channels = bottom_blob.c;
    const size_t hstep = (size_t)w * h * elempack;
    const size_t outcstep = (size_t)top_blob.w * top_blob.h * out_elempack;

    if (out_elempack == 1)
    {
        const size_t wstep = (size_t)w * elempack;
        const size_t xstep = (size_t)elempack;
        const size_t outxstep = (size_t)top_blob.w * out_elempack;

        #pragma omp parallel for num_threads(nT)
        for (int x = 0; x < w; x++)
        {
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob + x * xstep + (size_t)q * out_elempack * hstep;
                float* outptr = (float*)top_blob.channel(q) + x * outxstep;
                // exchange c and d, keeping h as the inner spatial axis
                permute_unpack_channels_stride(bottom_blob, ptr, outptr, h, channels, wstep, outcstep);
            }
        }
        return;
    }

    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t ystep = (size_t)w * elempack;
    const size_t outystep = (size_t)out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int y = 0; y < h; y++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr = (const float*)bottom_blob + y * ystep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + y * outystep;
            // exchange c and d, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_wchd(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t stride = bottom_blob.cstep;
        const size_t outstride = (size_t)channels * w;
        const size_t step = (size_t)w * h;
        const size_t outstep = top_blob.cstep;
        permute_transpose_matrices_stride(bottom_blob, stride, top_blob, outstride, channels, h, w, d, step, outstep, nT);
        return;
    }

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * h * elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;
    const size_t ystep = (size_t)w * elempack;
    const size_t outystep = (size_t)top_blob.w * top_blob.h * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int y = 0; y < h; y++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr = (const float*)bottom_blob + y * ystep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + y * outystep;
            // exchange c and d, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels_stride(bottom_blob, ptr, outptr, w, channels, wstep, outcstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_cwhd(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const size_t out_elemsize = top_blob.elemsize;

    // w and h stay adjacent on both sides
    Mat bottom_blob_3d(w * h, d, channels, bottom_blob.data, elemsize, elempack);
    bottom_blob_3d.cstep = bottom_blob.cstep;
    Mat top_blob_3d(channels * elempack, w * h, top_blob.c, top_blob.data, out_elemsize, out_elempack);
    top_blob_3d.cstep = top_blob.cstep;
    permute3d_cwh(bottom_blob_3d, top_blob_3d, nT);
}

static void permute4d_hcwd(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w * top_blob.h;
        const size_t outcstep = (size_t)top_blob.w;
        const size_t outdstep = top_blob.cstep;
        permute_transpose_hw_pack1_stride(bottom_blob, top_blob, outstride, outcstep, outdstep, nT);
        return;
    }

    const int channels = bottom_blob.c;
    const size_t hstep = (size_t)w * h * elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;

    if (out_elempack == 1)
    {
        const size_t wstep = (size_t)w * elempack;
        const size_t xstep = (size_t)elempack;
        const size_t outxstep = (size_t)top_blob.w * top_blob.h * out_elempack;

        #pragma omp parallel for num_threads(nT)
        for (int x = 0; x < w; x++)
        {
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob + x * xstep + (size_t)q * out_elempack * hstep;
                float* outptr = (float*)top_blob.channel(q) + x * outxstep;
                // exchange c and d, keeping h as the inner spatial axis
                permute_unpack_channels_stride(bottom_blob, ptr, outptr, h, channels, wstep, outcstep);
            }
        }
        return;
    }

    const size_t outwstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t ystep = (size_t)w * elempack;
    const size_t outystep = (size_t)out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int y = 0; y < h; y++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr = (const float*)bottom_blob + y * ystep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + y * outystep;
            // exchange c and d, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_chwd(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w * top_blob.h;
        const size_t outdstep = top_blob.cstep;
        const size_t outhstep = (size_t)top_blob.w;
        permute_transpose_cw_pack1_stride(bottom_blob, top_blob, outstride, outdstep, outhstep, nT);
        return;
    }

    const int w = bottom_blob.w;
    const int channels = bottom_blob.c;
    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * h * elempack;
    const size_t outwstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t outcstep = out_elempack;
    const size_t ystep = (size_t)w * elempack;
    const size_t outystep = (size_t)top_blob.w * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int y = 0; y < h; y++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr = (const float*)bottom_blob + y * ystep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + y * outystep;
            // exchange c and d, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels(bottom_blob, ptr, outptr, w, channels, wstep, outwstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_wdch(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t stride = (size_t)w * h;
        const size_t outstride = top_blob.cstep;
        const size_t step = bottom_blob.cstep;
        const size_t outstep = (size_t)d * w;
        permute_transpose_matrices_stride(bottom_blob, stride, top_blob, outstride, d, h, w, channels, step, outstep, nT);
        return;
    }

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)top_blob.w * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and h, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels_stride(bottom_blob, ptr, outptr, w, channels, wstep, outcstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_dwch(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w;
        const size_t outcstep = (size_t)top_blob.w * top_blob.h;
        const size_t outhstep = top_blob.cstep;
        permute_transpose_dw_pack1_stride(bottom_blob, top_blob, outstride, outcstep, outhstep, nT);
        return;
    }

    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const size_t hstep = (size_t)w * elempack;
    const size_t outcstep = (size_t)top_blob.w * top_blob.h * out_elempack;

    if (out_elempack == 1)
    {
        const size_t wstep = (size_t)w * h * elempack;
        const size_t xstep = (size_t)elempack;
        const size_t outxstep = (size_t)top_blob.w * out_elempack;

        #pragma omp parallel for num_threads(nT)
        for (int x = 0; x < w; x++)
        {
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob + x * xstep + (size_t)q * out_elempack * hstep;
                float* outptr = (float*)top_blob.channel(q) + x * outxstep;
                // exchange c and h, keeping d as the inner spatial axis
                permute_unpack_channels_stride(bottom_blob, ptr, outptr, d, channels, wstep, outcstep);
            }
        }
        return;
    }

    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and h, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_wcdh(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t stride = bottom_blob.cstep;
        const size_t outstride = top_blob.cstep;
        const size_t step = (size_t)w * h;
        const size_t outstep = (size_t)channels * w;
        permute_transpose_matrices_stride(bottom_blob, stride, top_blob, outstride, channels, h, w, d, step, outstep, nT);
        return;
    }

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)top_blob.w * top_blob.h * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and h, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels_stride(bottom_blob, ptr, outptr, w, channels, wstep, outcstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_cwdh(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w;
        const size_t outdstep = (size_t)top_blob.w * top_blob.h;
        const size_t outhstep = top_blob.cstep;
        permute_transpose_cw_pack1_stride(bottom_blob, top_blob, outstride, outdstep, outhstep, nT);
        return;
    }

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t outcstep = out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)top_blob.w * top_blob.h * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and h, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels(bottom_blob, ptr, outptr, w, channels, wstep, outwstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_dcwh(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w * top_blob.h;
        const size_t outcstep = (size_t)top_blob.w;
        const size_t outhstep = top_blob.cstep;
        permute_transpose_dw_pack1_stride(bottom_blob, top_blob, outstride, outcstep, outhstep, nT);
        return;
    }

    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const size_t hstep = (size_t)w * elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;

    if (out_elempack == 1)
    {
        const size_t wstep = (size_t)w * h * elempack;
        const size_t xstep = (size_t)elempack;
        const size_t outxstep = (size_t)top_blob.w * top_blob.h * out_elempack;

        #pragma omp parallel for num_threads(nT)
        for (int x = 0; x < w; x++)
        {
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob + x * xstep + (size_t)q * out_elempack * hstep;
                float* outptr = (float*)top_blob.channel(q) + x * outxstep;
                // exchange c and h, keeping d as the inner spatial axis
                permute_unpack_channels_stride(bottom_blob, ptr, outptr, d, channels, wstep, outcstep);
            }
        }
        return;
    }

    const size_t outwstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and h, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_cdwh(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w * top_blob.h;
        const size_t outdstep = (size_t)top_blob.w;
        const size_t outhstep = top_blob.cstep;
        permute_transpose_cw_pack1_stride(bottom_blob, top_blob, outstride, outdstep, outhstep, nT);
        return;
    }

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t outcstep = out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)top_blob.w * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and h, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels(bottom_blob, ptr, outptr, w, channels, wstep, outwstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_hdcw(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const size_t out_elemsize = top_blob.elemsize;

    // h and d stay adjacent on both sides
    Mat bottom_blob_3d(w, h * d, channels, bottom_blob.data, elemsize, elempack);
    bottom_blob_3d.cstep = bottom_blob.cstep;
    Mat top_blob_3d(h * d, channels * elempack, top_blob.c, top_blob.data, out_elemsize, out_elempack);
    top_blob_3d.cstep = top_blob.cstep;
    permute3d_hcw(bottom_blob_3d, top_blob_3d, nT);
}

static void permute4d_dhcw(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = top_blob.cstep;
        const size_t outcstep = (size_t)top_blob.w * top_blob.h;
        const size_t outhstep = (size_t)top_blob.w;
        permute_transpose_dw_pack1_stride(bottom_blob, top_blob, outstride, outcstep, outhstep, nT);
        return;
    }

    const int w = bottom_blob.w;
    const int channels = bottom_blob.c;
    const size_t hstep = elempack;
    const size_t outcstep = (size_t)top_blob.w * top_blob.h * out_elempack;

    if (out_elempack == 1)
    {
        const size_t wstep = (size_t)w * h * elempack;
        const size_t outwstep = out_elempack;
        const size_t ystep = (size_t)w * elempack;
        const size_t outystep = (size_t)top_blob.w * out_elempack;

        #pragma omp parallel for num_threads(nT)
        for (int y = 0; y < h; y++)
        {
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob + y * ystep + (size_t)q * out_elempack * hstep;
                float* outptr = (float*)top_blob.channel(q) + y * outystep;
                // exchange c and w, keeping d as the inner spatial axis
                if (elempack == 1)
                    permute_pack_channels(bottom_blob, top_blob, ptr, outptr, d, channels, wstep, outwstep, outcstep);
                else
                    permute_unpack_channels_stride(bottom_blob, ptr, outptr, d, channels, wstep, outcstep);
            }
        }
        return;
    }

    const size_t wstep = (size_t)w * elempack;
    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and w, keeping h as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
            else
                permute_channels_axis_input_stride(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
        }
    }
}

static void permute4d_hcdw(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = top_blob.cstep;
        const size_t outcstep = (size_t)top_blob.w;
        const size_t outdstep = (size_t)top_blob.w * top_blob.h;
        permute_transpose_hw_pack1_stride(bottom_blob, top_blob, outstride, outcstep, outdstep, nT);
        return;
    }

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const size_t wstep = (size_t)w * elempack;
    const size_t hstep = elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)top_blob.w * top_blob.h * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and w, keeping h as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels_stride(bottom_blob, ptr, outptr, h, channels, wstep, outcstep);
            else
                permute_channels_axis_input_stride(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
        }
    }
}

static void permute4d_chdw(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const size_t out_elemsize = top_blob.elemsize;

    // h and d stay adjacent on both sides
    Mat bottom_blob_3d(w, h * d, channels, bottom_blob.data, elemsize, elempack);
    bottom_blob_3d.cstep = bottom_blob.cstep;
    Mat top_blob_3d(channels * elempack, h * d, top_blob.c, top_blob.data, out_elemsize, out_elempack);
    top_blob_3d.cstep = top_blob.cstep;
    permute3d_chw(bottom_blob_3d, top_blob_3d, nT);
}

static void permute4d_dchw(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = top_blob.cstep;
        const size_t outcstep = (size_t)top_blob.w;
        const size_t outhstep = (size_t)top_blob.w * top_blob.h;
        permute_transpose_dw_pack1_stride(bottom_blob, top_blob, outstride, outcstep, outhstep, nT);
        return;
    }

    const int w = bottom_blob.w;
    const int channels = bottom_blob.c;
    const size_t hstep = elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;

    if (out_elempack == 1)
    {
        const size_t wstep = (size_t)w * h * elempack;
        const size_t outwstep = out_elempack;
        const size_t ystep = (size_t)w * elempack;
        const size_t outystep = (size_t)top_blob.w * top_blob.h * out_elempack;

        #pragma omp parallel for num_threads(nT)
        for (int y = 0; y < h; y++)
        {
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob + y * ystep + (size_t)q * out_elempack * hstep;
                float* outptr = (float*)top_blob.channel(q) + y * outystep;
                // exchange c and w, keeping d as the inner spatial axis
                if (elempack == 1)
                    permute_pack_channels(bottom_blob, top_blob, ptr, outptr, d, channels, wstep, outwstep, outcstep);
                else
                    permute_unpack_channels_stride(bottom_blob, ptr, outptr, d, channels, wstep, outcstep);
            }
        }
        return;
    }

    const size_t wstep = (size_t)w * elempack;
    const size_t outwstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and w, keeping h as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
            else
                permute_channels_axis_input_stride(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
        }
    }
}

static void permute4d_cdhw(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = top_blob.cstep;
        const size_t outdstep = (size_t)top_blob.w;
        const size_t outhstep = (size_t)top_blob.w * top_blob.h;
        permute_transpose_cw_pack1_stride(bottom_blob, top_blob, outstride, outdstep, outhstep, nT);
        return;
    }

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const size_t wstep = (size_t)w * elempack;
    const size_t hstep = elempack;
    const size_t outwstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t outcstep = out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)top_blob.w * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and w, keeping h as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels(bottom_blob, ptr, outptr, h, channels, wstep, outwstep);
            else
                permute_channels_axis_input_stride(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
        }
    }
}

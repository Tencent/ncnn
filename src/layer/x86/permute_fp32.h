// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// full register tiles have no size or packing branches

// contiguous tiles take only pointers; stride variants use scalar-element strides

#if __SSE2__
static NCNN_FORCEINLINE void permute_transpose4x2_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
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
#endif // __SSE2__

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
static NCNN_FORCEINLINE void permute_transpose8x2_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
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
    __m256 _a = _mm256_insertf128_ps(_mm256_castps128_ps256(_a0), _a1, 1);
    __m256 _b = _mm256_insertf128_ps(_mm256_castps128_ps256(_b0), _b1, 1);
    _mm256_storeu_ps(outptr, _a);
    _mm256_storeu_ps(outptr + outstride, _b);
}
#endif // __AVX__

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
static NCNN_FORCEINLINE void permute_transpose16x2_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
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
    __m512 _a = _mm512_castps128_ps512(_a0);
    _a = _mm512_insertf32x4(_a, _a1, 1);
    _a = _mm512_insertf32x4(_a, _a2, 2);
    _a = _mm512_insertf32x4(_a, _a3, 3);
    __m512 _b = _mm512_castps128_ps512(_b0);
    _b = _mm512_insertf32x4(_b, _b1, 1);
    _b = _mm512_insertf32x4(_b, _b2, 2);
    _b = _mm512_insertf32x4(_b, _b3, 3);
    _mm512_storeu_ps(outptr, _a);
    _mm512_storeu_ps(outptr + outstride, _b);
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

// final two or one rows: load complete vectors and write only the valid output lanes
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

// unpacked matrix transpose, shared by 2d and channel/spatial permutations
#if __SSE2__
// fixed input width; callers select the packing before traversing the rows
static void permute_unpack4_stride(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows)
{
    int i = 0;
#if __AVX512F__
    for (; i + 15 < rows; i += 16)
    {
        permute_transpose16x4_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
#endif // __AVX512F__
#if __AVX__
    for (; i + 7 < rows; i += 8)
    {
        permute_transpose8x4_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
#endif // __AVX__
    for (; i + 3 < rows; i += 4)
    {
        permute_transpose4x4_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 1 < rows; i += 2)
    {
        permute_transpose2x4_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i < rows; i++)
    {
        permute_transpose1x4_stride(ptr + i * stride, outptr + i, outstride);
    }
}
#endif // __SSE2__

#if __AVX__
// fixed input width; callers select the packing before traversing the rows
static void permute_unpack8_stride(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows)
{
    int i = 0;
#if __AVX512F__
    for (; i + 15 < rows; i += 16)
    {
        permute_transpose16x8_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
#endif // __AVX512F__
    for (; i + 7 < rows; i += 8)
    {
        permute_transpose8x8_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 3 < rows; i += 4)
    {
        permute_transpose4x8_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 1 < rows; i += 2)
    {
        permute_transpose2x8_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i < rows; i++)
    {
        permute_transpose1x8_stride(ptr + i * stride, outptr + i, outstride);
    }
}
#endif // __AVX__

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
        permute_transpose8x16_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 3 < rows; i += 4)
    {
        permute_transpose4x16_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 1 < rows; i += 2)
    {
        permute_transpose2x16_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i < rows; i++)
    {
        permute_transpose1x16_stride(ptr + i * stride, outptr + i, outstride);
    }
}
#endif // __AVX512F__

static void permute_transpose_pack1_block(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
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
        for (; j + 1 < cols; j += 2)
        {
            permute_transpose16x2_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
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
        for (; j + 1 < cols; j += 2)
        {
            permute_transpose8x2_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
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
        for (; j + 1 < cols; j += 2)
        {
            permute_transpose4x2_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
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

// cache blocking is useful for medium planes with regularly spaced rows
// small task blocks and narrow pack/unpack matrices use the direct kernel
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

    // large planes keep an output stripe resident while scanning the input
    if (rows >= 512 && cols >= 16 && stride >= 512 && outstride >= 512)
    {
        for (int j = 0; j < cols; j += 32)
            permute_transpose_pack1_block(ptr + j, stride, outptr + j * outstride, outstride, rows, std::min(32, cols - j));
        return;
    }
    if (rows >= 16 && cols >= 16 && (rows > 512 || cols > 512))
    {
        // coalesced axes can form long rectangles
        // bound the tile payload to 16 KiB on each side, allowing wider tiles for fewer input rows
        const int row_block = std::min(rows, 64);
        const int col_block = std::min(256, 16384 / (row_block * (int)sizeof(float)));
        for (int j = 0; j < cols; j += col_block)
        {
            for (int i = 0; i < rows; i += row_block)
                permute_transpose_pack1_block(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride, std::min(row_block, rows - i), std::min(col_block, cols - j));
        }
        return;
    }

#if __SSE2__
    if (rows >= 256 && rows <= 512 && cols >= 256 && cols <= 512 && stride <= 1024 && outstride <= 1024
        && stride % 256 == 0 && outstride % 256 == 0)
    {
        const int block = 32;
        for (int i = 0; i < rows; i += block)
        {
            for (int j = 0; j < cols; j += block)
            {
                permute_transpose_pack1_block(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride, std::min(block, rows - i), std::min(block, cols - j));
            }
        }
        return;
    }
#endif // __SSE2__
    permute_transpose_pack1_block(ptr, stride, outptr, outstride, rows, cols);
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
        permute_transpose_pack1_block(ptr, stride, outptr, outstride, rows, cols);
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
    // keep adjacent input records and output rows in the same cache block
    for (int x = 0; x < cols; x += 8)
    {
        const int xmax = std::min(x + 8, cols);
        int i = 0;
        for (; i + 1 < rows; i += 2)
        {
            int j = x;
            for (; j + 1 < xmax; j += 2)
            {
                permute_spatial2x2_pack4_stride(ptr + i * stride + j * 4, stride, outptr + j * outstride + i * 4, outstride);
            }
            for (; j < xmax; j++)
            {
                __m128i _v0 = _mm_loadu_si128((const __m128i*)(ptr + i * stride + j * 4));
                _mm_storeu_si128((__m128i*)(outptr + j * outstride + i * 4), _v0);
                __m128i _v1 = _mm_loadu_si128((const __m128i*)(ptr + (i + 1) * stride + j * 4));
                _mm_storeu_si128((__m128i*)(outptr + j * outstride + (i + 1) * 4), _v1);
            }
        }
        for (; i < rows; i++)
        {
            for (int j = x; j < xmax; j++)
            {
                __m128i _v0 = _mm_loadu_si128((const __m128i*)(ptr + i * stride + j * 4));
                _mm_storeu_si128((__m128i*)(outptr + j * outstride + i * 4), _v0);
            }
        }
    }
}

static void permute_spatial_pack4to1(const float* ptr, size_t stride, float* outptr, size_t outstride, size_t outcstep, int rows, int cols)
{
#if __AVX512F__
    // keep adjacent input records and output rows in the same cache block
    for (int x = 0; x < cols; x += 8)
    {
        const int xmax = std::min(x + 8, cols);

        int i = 0;
        for (; i + 15 < rows; i += 16)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose16x4_stride(ptr + i * stride + j * 4, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i + 7 < rows; i += 8)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose8x4_stride(ptr + i * stride + j * 4, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i + 3 < rows; i += 4)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose4x4_stride(ptr + i * stride + j * 4, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i + 1 < rows; i += 2)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose2x4_stride(ptr + i * stride + j * 4, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i < rows; i++)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose1x4_stride(ptr + i * stride + j * 4, outptr + j * outstride + i, outcstep);
            }
        }
    }
#else
    for (int x = 0; x < cols; x++)
        permute_unpack4_stride(ptr + x * 4, stride, outptr + x * outstride, outcstep, rows);
#endif // __AVX512F__
}
#endif // __SSE2__

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
    // keep adjacent input records and output rows in the same cache block
    for (int x = 0; x < cols; x += 8)
    {
        const int xmax = std::min(x + 8, cols);
        int i = 0;
        for (; i + 1 < rows; i += 2)
        {
            int j = x;
            for (; j + 1 < xmax; j += 2)
            {
                permute_spatial2x2_pack8_stride(ptr + i * stride + j * 8, stride, outptr + j * outstride + i * 8, outstride);
            }
            for (; j < xmax; j++)
            {
                __m256 _v0 = _mm256_loadu_ps(ptr + i * stride + j * 8);
                _mm256_storeu_ps(outptr + j * outstride + i * 8, _v0);
                __m256 _v1 = _mm256_loadu_ps(ptr + (i + 1) * stride + j * 8);
                _mm256_storeu_ps(outptr + j * outstride + (i + 1) * 8, _v1);
            }
        }
        for (; i < rows; i++)
        {
            for (int j = x; j < xmax; j++)
            {
                __m256 _v0 = _mm256_loadu_ps(ptr + i * stride + j * 8);
                _mm256_storeu_ps(outptr + j * outstride + i * 8, _v0);
            }
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

    // keep adjacent input records and output rows in the same cache block
    for (int x = 0; x < cols; x += 8)
    {
        const int xmax = std::min(x + 8, cols);

        int i = 0;
        for (; i + 15 < rows; i += 16)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose16x8_stride(ptr + i * stride + j * 8, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i + 7 < rows; i += 8)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose8x8_stride(ptr + i * stride + j * 8, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i + 3 < rows; i += 4)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose4x8_stride(ptr + i * stride + j * 8, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i + 1 < rows; i += 2)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose2x8_stride(ptr + i * stride + j * 8, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i < rows; i++)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose1x8_stride(ptr + i * stride + j * 8, outptr + j * outstride + i, outcstep);
            }
        }
    }
#else
    for (int x = 0; x < cols; x++)
        permute_unpack8_stride(ptr + x * 8, stride, outptr + x * outstride, outcstep, rows);
#endif // __AVX512F__
}
#endif // __AVX__

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
    // keep adjacent input records and output rows in the same cache block
    for (int x = 0; x < cols; x += 8)
    {
        const int xmax = std::min(x + 8, cols);
        int i = 0;
        for (; i + 1 < rows; i += 2)
        {
            int j = x;
            for (; j + 1 < xmax; j += 2)
            {
                permute_spatial2x2_pack16_stride(ptr + i * stride + j * 16, stride, outptr + j * outstride + i * 16, outstride);
            }
            for (; j < xmax; j++)
            {
                __m512 _v0 = _mm512_loadu_ps(ptr + i * stride + j * 16);
                _mm512_storeu_ps(outptr + j * outstride + i * 16, _v0);
                __m512 _v1 = _mm512_loadu_ps(ptr + (i + 1) * stride + j * 16);
                _mm512_storeu_ps(outptr + j * outstride + (i + 1) * 16, _v1);
            }
        }
        for (; i < rows; i++)
        {
            for (int j = x; j < xmax; j++)
            {
                __m512 _v0 = _mm512_loadu_ps(ptr + i * stride + j * 16);
                _mm512_storeu_ps(outptr + j * outstride + i * 16, _v0);
            }
        }
    }
}

static void permute_spatial_pack16to1(const float* ptr, size_t stride, float* outptr, size_t outstride, size_t outcstep, int rows, int cols)
{
    // keep adjacent input records and output rows in the same cache block
    for (int x = 0; x < cols; x += 8)
    {
        const int xmax = std::min(x + 8, cols);

        int i = 0;
        for (; i + 15 < rows; i += 16)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose16x16_stride(ptr + i * stride + j * 16, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i + 7 < rows; i += 8)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose8x16_stride(ptr + i * stride + j * 16, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i + 3 < rows; i += 4)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose4x16_stride(ptr + i * stride + j * 16, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i + 1 < rows; i += 2)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose2x16_stride(ptr + i * stride + j * 16, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i < rows; i++)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose1x16_stride(ptr + i * stride + j * 16, outptr + j * outstride + i, outcstep);
            }
        }
    }
}
#endif // __AVX512F__

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
#endif // __SSE2__

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
#endif // __AVX__

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
#endif // __SSE2__
#if __AVX__
    if (size == 8)
    {
        permute_spatial_pack8(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX__
#if __AVX512F__
    if (size == 16)
    {
        permute_spatial_pack16(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__
#if __SSE2__
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
#endif // __SSE2__
#if __SSE2__
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
#endif // __SSE2__
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
#endif // __AVX__
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
#endif // __SSE2__

#if __AVX__
    if (elempack == 8)
    {
        permute_unpack8_stride(ptr, 8, outptr, outcstep, size);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 16)
    {
        permute_unpack16_stride(ptr, 16, outptr, outcstep, size);
        return;
    }
#endif // __AVX512F__
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
        permute_transpose_pack1_block(p, hstep, out, outwstep, 4, w);
    }
}
#endif // __SSE2__

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
        permute_transpose_pack1_block(p, hstep, out, outwstep, 8, w);
    }
}
#endif // __AVX__

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
        permute_transpose_pack1_block(p, hstep, out, outwstep, 16, w);
    }
}
#endif // __AVX512F__

#if __SSE2__
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
#endif // __SSE2__

#if __SSE2__
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
#endif // __SSE2__

#if __AVX__
static void permute3d_pack4to8(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 4 * 8;
        for (int x = 0; x < w; x++)
        {
            permute_transpose8x4(p, out);
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
            permute_transpose8x4_stride(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute3d_pack4to16_stride(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 4 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose16x4_stride(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}
#endif // __AVX512F__

#if __AVX__
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
#endif // __AVX__

#if __AVX__
static void permute3d_pack8to4(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 8 * 4;
        for (int x = 0; x < w; x++)
        {
            permute_transpose4x8(p, out);
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
            permute_transpose4x8_stride(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}
#endif // __AVX__

#if __AVX__
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
#endif // __AVX__

#if __AVX512F__
static void permute3d_pack8to16_stride(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 8 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose16x8_stride(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
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
#endif // __AVX512F__

#if __AVX512F__
static void permute3d_pack16to4(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 16 * 4;
        for (int x = 0; x < w; x++)
        {
            permute_transpose4x16(p, out);
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
            permute_transpose4x16_stride(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute3d_pack16to8_stride(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 16 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose8x16_stride(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
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

// process one output channel group
// all strides are in scalar elements
static void permute3d(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, int elempack, int out_elempack)
{
#if __SSE2__
    if (elempack == 1 && out_elempack == 4)
    {
        if (hstep == 1)
            permute3d_pack1to4(ptr, outptr, w, channels, wstep, cstep, outwstep, outcstep);
        else
            permute3d_pack1to4_stride(ptr, outptr, w, channels, hstep, cstep, outwstep, outcstep);
        return;
    }
#endif // __SSE2__

#if __AVX__
    if (elempack == 1 && out_elempack == 8)
    {
        if (hstep == 1)
            permute3d_pack1to8(ptr, outptr, w, channels, wstep, cstep, outwstep, outcstep);
        else
            permute3d_pack1to8_stride(ptr, outptr, w, channels, hstep, cstep, outwstep, outcstep);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 1 && out_elempack == 16)
    {
        if (hstep == 1)
            permute3d_pack1to16(ptr, outptr, w, channels, wstep, cstep, outwstep, outcstep);
        else
            permute3d_pack1to16_stride(ptr, outptr, w, channels, hstep, cstep, outwstep, outcstep);
        return;
    }
#endif // __AVX512F__

#if __SSE2__
    if (elempack == 4 && out_elempack == 1)
    {
        if (outcstep == 1)
            permute3d_pack4to1(ptr, outptr, w, channels, wstep, cstep, outwstep);
        else
            permute3d_pack4to1_stride(ptr, outptr, w, channels, wstep, cstep, outcstep);
        return;
    }
#endif // __SSE2__

#if __SSE2__
    if (elempack == 4 && out_elempack == 4)
    {
        if (hstep == 4 && outcstep == 4)
            permute3d_pack4to4(ptr, outptr, w, channels, wstep, cstep, outwstep);
        else
            permute3d_pack4to4_stride(ptr, outptr, w, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
#endif // __SSE2__

#if __AVX__
    if (elempack == 4 && out_elempack == 8)
    {
        if (hstep == 4 && outcstep == 8)
            permute3d_pack4to8(ptr, outptr, w, channels, wstep, cstep, outwstep);
        else
            permute3d_pack4to8_stride(ptr, outptr, w, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 4 && out_elempack == 16)
    {
        permute3d_pack4to16_stride(ptr, outptr, w, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
#endif // __AVX512F__

#if __AVX__
    if (elempack == 8 && out_elempack == 1)
    {
        if (outcstep == 1)
            permute3d_pack8to1(ptr, outptr, w, channels, wstep, cstep, outwstep);
        else
            permute3d_pack8to1_stride(ptr, outptr, w, channels, wstep, cstep, outcstep);
        return;
    }
#endif // __AVX__

#if __AVX__
    if (elempack == 8 && out_elempack == 4)
    {
        if (hstep == 8 && outcstep == 4)
            permute3d_pack8to4(ptr, outptr, w, channels, wstep, cstep, outwstep);
        else
            permute3d_pack8to4_stride(ptr, outptr, w, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
#endif // __AVX__

#if __AVX__
    if (elempack == 8 && out_elempack == 8)
    {
        permute3d_pack8to8_stride(ptr, outptr, w, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 8 && out_elempack == 16)
    {
        permute3d_pack8to16_stride(ptr, outptr, w, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 1)
    {
        if (outcstep == 1)
            permute3d_pack16to1(ptr, outptr, w, channels, wstep, cstep, outwstep);
        else
            permute3d_pack16to1_stride(ptr, outptr, w, channels, wstep, cstep, outcstep);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 4)
    {
        if (hstep == 16 && outcstep == 4)
            permute3d_pack16to4(ptr, outptr, w, channels, wstep, cstep, outwstep);
        else
            permute3d_pack16to4_stride(ptr, outptr, w, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 8)
    {
        permute3d_pack16to8_stride(ptr, outptr, w, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 16)
    {
        permute3d_pack16to16_stride(ptr, outptr, w, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
#endif // __AVX512F__
}

// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Full register tiles have no size or packing branches.

// Contiguous tiles take only pointers; stride variants use scalar-element strides.

#if __SSE2__
static NCNN_FORCEINLINE void permute_transpose4x2_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    int _p0;
    memcpy(&_p0, ptr, 4);
    __m128i _r0 = _mm_cvtsi32_si128(_p0);
    int _p1;
    memcpy(&_p1, ptr + stride, 4);
    __m128i _r1 = _mm_cvtsi32_si128(_p1);
    int _p2;
    memcpy(&_p2, ptr + 2 * stride, 4);
    __m128i _r2 = _mm_cvtsi32_si128(_p2);
    int _p3;
    memcpy(&_p3, ptr + 3 * stride, 4);
    __m128i _r3 = _mm_cvtsi32_si128(_p3);
    __m128i _t0 = _mm_unpacklo_epi32(_r0, _r1);
    __m128i _t1 = _mm_unpacklo_epi32(_r2, _r3);
    __m128i _v0 = _mm_unpacklo_epi64(_t0, _t1);
    _v0 = _mm_shufflelo_epi16(_v0, _MM_SHUFFLE(3, 1, 2, 0));
    _v0 = _mm_shufflehi_epi16(_v0, _MM_SHUFFLE(3, 1, 2, 0));
    _v0 = _mm_shuffle_epi32(_v0, _MM_SHUFFLE(3, 1, 2, 0));
    _mm_storel_epi64((__m128i*)outptr, _v0);
    _mm_storel_epi64((__m128i*)(outptr + outstride), _mm_srli_si128(_v0, 8));
}
#endif // __SSE2__

#if __SSE2__
static NCNN_FORCEINLINE void permute_transpose4x4_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr)
{
    __m128i _r0 = _mm_loadu_si128((const __m128i*)ptr);
    __m128i _r1 = _mm_loadu_si128((const __m128i*)(ptr + 8));
    __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
    __m128i _t1 = _mm_unpackhi_epi16(_r0, _r1);
    _r0 = _mm_unpacklo_epi16(_t0, _t1);
    _r1 = _mm_unpackhi_epi16(_t0, _t1);
    _mm_storeu_si128((__m128i*)outptr, _r0);
    _mm_storeu_si128((__m128i*)(outptr + 8), _r1);
}

static NCNN_FORCEINLINE void permute_transpose4x4_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    __m128i _r0 = _mm_loadl_epi64((const __m128i*)(ptr));
    __m128i _r1 = _mm_loadl_epi64((const __m128i*)(ptr + stride));
    __m128i _r2 = _mm_loadl_epi64((const __m128i*)(ptr + 2 * stride));
    __m128i _r3 = _mm_loadl_epi64((const __m128i*)(ptr + 3 * stride));
    __m128i _r4 = _mm_setzero_si128();
    __m128i _r5 = _mm_setzero_si128();
    __m128i _r6 = _mm_setzero_si128();
    __m128i _r7 = _mm_setzero_si128();
    transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
    _mm_storel_epi64((__m128i*)(outptr), _r0);
    _mm_storel_epi64((__m128i*)(outptr + outstride), _r1);
    _mm_storel_epi64((__m128i*)(outptr + 2 * outstride), _r2);
    _mm_storel_epi64((__m128i*)(outptr + 3 * outstride), _r3);
}
#endif // __SSE2__

#if __AVX__
static NCNN_FORCEINLINE void permute_transpose4x8_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr)
{
#if __AVX512BW__
    const __m512i _index = _mm512_set_epi16(31, 23, 15, 7, 30, 22, 14, 6, 29, 21, 13, 5, 28, 20, 12, 4, 27, 19, 11, 3, 26, 18, 10, 2, 25, 17, 9, 1, 24, 16, 8, 0);
    __m512i _v = _mm512_loadu_si512(ptr);
    _v = _mm512_permutexvar_epi16(_index, _v);
    _mm512_storeu_si512(outptr, _v);
#else
    __m128i _r0 = _mm_loadu_si128((const __m128i*)(ptr));
    __m128i _r1 = _mm_loadu_si128((const __m128i*)(ptr + 8));
    __m128i _r2 = _mm_loadu_si128((const __m128i*)(ptr + 16));
    __m128i _r3 = _mm_loadu_si128((const __m128i*)(ptr + 24));
    transpose8x4_epi16(_r0, _r1, _r2, _r3);
    _mm_storeu_si128((__m128i*)(outptr), _r0);
    _mm_storeu_si128((__m128i*)(outptr + 8), _r1);
    _mm_storeu_si128((__m128i*)(outptr + 16), _r2);
    _mm_storeu_si128((__m128i*)(outptr + 24), _r3);
#endif // __AVX512BW__
}

#endif // __AVX__

#if __SSE2__
static NCNN_FORCEINLINE void permute_transpose4x8_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    __m128i _r0 = _mm_loadu_si128((const __m128i*)(ptr));
    __m128i _r1 = _mm_loadu_si128((const __m128i*)(ptr + stride));
    __m128i _r2 = _mm_loadu_si128((const __m128i*)(ptr + 2 * stride));
    __m128i _r3 = _mm_loadu_si128((const __m128i*)(ptr + 3 * stride));
    __m128i _r4 = _mm_setzero_si128();
    __m128i _r5 = _mm_setzero_si128();
    __m128i _r6 = _mm_setzero_si128();
    __m128i _r7 = _mm_setzero_si128();
    transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
    _mm_storel_epi64((__m128i*)(outptr), _r0);
    _mm_storel_epi64((__m128i*)(outptr + outstride), _r1);
    _mm_storel_epi64((__m128i*)(outptr + 2 * outstride), _r2);
    _mm_storel_epi64((__m128i*)(outptr + 3 * outstride), _r3);
    _mm_storel_epi64((__m128i*)(outptr + 4 * outstride), _r4);
    _mm_storel_epi64((__m128i*)(outptr + 5 * outstride), _r5);
    _mm_storel_epi64((__m128i*)(outptr + 6 * outstride), _r6);
    _mm_storel_epi64((__m128i*)(outptr + 7 * outstride), _r7);
}
#endif // __SSE2__

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose4x16_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    permute_transpose4x8_stride_bf16s_fp16s(ptr, stride, outptr, outstride);
    permute_transpose4x8_stride_bf16s_fp16s(ptr + 8, stride, outptr + 8 * outstride, outstride);
}
#endif // __AVX512F__

#if __SSE2__
static NCNN_FORCEINLINE void permute_transpose8x2_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    int _p0;
    memcpy(&_p0, ptr, 4);
    __m128i _r0 = _mm_cvtsi32_si128(_p0);
    int _p1;
    memcpy(&_p1, ptr + stride, 4);
    __m128i _r1 = _mm_cvtsi32_si128(_p1);
    int _p2;
    memcpy(&_p2, ptr + 2 * stride, 4);
    __m128i _r2 = _mm_cvtsi32_si128(_p2);
    int _p3;
    memcpy(&_p3, ptr + 3 * stride, 4);
    __m128i _r3 = _mm_cvtsi32_si128(_p3);
    __m128i _t0 = _mm_unpacklo_epi32(_r0, _r1);
    __m128i _t1 = _mm_unpacklo_epi32(_r2, _r3);
    __m128i _v0 = _mm_unpacklo_epi64(_t0, _t1);
    _v0 = _mm_shufflelo_epi16(_v0, _MM_SHUFFLE(3, 1, 2, 0));
    _v0 = _mm_shufflehi_epi16(_v0, _MM_SHUFFLE(3, 1, 2, 0));
    _v0 = _mm_shuffle_epi32(_v0, _MM_SHUFFLE(3, 1, 2, 0));
    int _p4;
    memcpy(&_p4, ptr + 4 * stride, 4);
    __m128i _r4 = _mm_cvtsi32_si128(_p4);
    int _p5;
    memcpy(&_p5, ptr + 5 * stride, 4);
    __m128i _r5 = _mm_cvtsi32_si128(_p5);
    int _p6;
    memcpy(&_p6, ptr + 6 * stride, 4);
    __m128i _r6 = _mm_cvtsi32_si128(_p6);
    int _p7;
    memcpy(&_p7, ptr + 7 * stride, 4);
    __m128i _r7 = _mm_cvtsi32_si128(_p7);
    __m128i _t2 = _mm_unpacklo_epi32(_r4, _r5);
    __m128i _t3 = _mm_unpacklo_epi32(_r6, _r7);
    __m128i _v1 = _mm_unpacklo_epi64(_t2, _t3);
    _v1 = _mm_shufflelo_epi16(_v1, _MM_SHUFFLE(3, 1, 2, 0));
    _v1 = _mm_shufflehi_epi16(_v1, _MM_SHUFFLE(3, 1, 2, 0));
    _v1 = _mm_shuffle_epi32(_v1, _MM_SHUFFLE(3, 1, 2, 0));
    _mm_storeu_si128((__m128i*)outptr, _mm_unpacklo_epi64(_v0, _v1));
    _mm_storeu_si128((__m128i*)(outptr + outstride), _mm_unpackhi_epi64(_v0, _v1));
}
#endif // __SSE2__

#if __AVX__
static NCNN_FORCEINLINE void permute_transpose8x4_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr)
{
#if __AVX512BW__
    const __m512i _index = _mm512_set_epi16(31, 27, 23, 19, 15, 11, 7, 3, 30, 26, 22, 18, 14, 10, 6, 2, 29, 25, 21, 17, 13, 9, 5, 1, 28, 24, 20, 16, 12, 8, 4, 0);
    __m512i _v = _mm512_loadu_si512(ptr);
    _v = _mm512_permutexvar_epi16(_index, _v);
    _mm512_storeu_si512(outptr, _v);
#else
    __m128i _r0 = _mm_loadu_si128((const __m128i*)(ptr));
    __m128i _r1 = _mm_loadu_si128((const __m128i*)(ptr + 8));
    __m128i _r2 = _mm_loadu_si128((const __m128i*)(ptr + 16));
    __m128i _r3 = _mm_loadu_si128((const __m128i*)(ptr + 24));
    __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
    __m128i _t1 = _mm_unpackhi_epi16(_r0, _r1);
    __m128i _t2 = _mm_unpacklo_epi16(_r2, _r3);
    __m128i _t3 = _mm_unpackhi_epi16(_r2, _r3);
    __m128i _a = _mm_unpacklo_epi16(_t0, _t1);
    __m128i _b = _mm_unpackhi_epi16(_t0, _t1);
    __m128i _c = _mm_unpacklo_epi16(_t2, _t3);
    __m128i _d = _mm_unpackhi_epi16(_t2, _t3);
    _r0 = _mm_unpacklo_epi64(_a, _c);
    _r1 = _mm_unpackhi_epi64(_a, _c);
    _r2 = _mm_unpacklo_epi64(_b, _d);
    _r3 = _mm_unpackhi_epi64(_b, _d);
    _mm_storeu_si128((__m128i*)(outptr), _r0);
    _mm_storeu_si128((__m128i*)(outptr + 8), _r1);
    _mm_storeu_si128((__m128i*)(outptr + 16), _r2);
    _mm_storeu_si128((__m128i*)(outptr + 24), _r3);
#endif // __AVX512BW__
}

#endif // __AVX__

#if __SSE2__
static NCNN_FORCEINLINE void permute_transpose8x4_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    __m128i _r0 = _mm_loadl_epi64((const __m128i*)(ptr));
    __m128i _r1 = _mm_loadl_epi64((const __m128i*)(ptr + stride));
    __m128i _r2 = _mm_loadl_epi64((const __m128i*)(ptr + 2 * stride));
    __m128i _r3 = _mm_loadl_epi64((const __m128i*)(ptr + 3 * stride));
    __m128i _r4 = _mm_loadl_epi64((const __m128i*)(ptr + 4 * stride));
    __m128i _r5 = _mm_loadl_epi64((const __m128i*)(ptr + 5 * stride));
    __m128i _r6 = _mm_loadl_epi64((const __m128i*)(ptr + 6 * stride));
    __m128i _r7 = _mm_loadl_epi64((const __m128i*)(ptr + 7 * stride));
    transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
    _mm_storeu_si128((__m128i*)(outptr), _r0);
    _mm_storeu_si128((__m128i*)(outptr + outstride), _r1);
    _mm_storeu_si128((__m128i*)(outptr + 2 * outstride), _r2);
    _mm_storeu_si128((__m128i*)(outptr + 3 * outstride), _r3);
}
#endif // __SSE2__

#if __SSE2__
static NCNN_FORCEINLINE void permute_transpose8x8_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    __m128i _r0 = _mm_loadu_si128((const __m128i*)(ptr));
    __m128i _r1 = _mm_loadu_si128((const __m128i*)(ptr + stride));
    __m128i _r2 = _mm_loadu_si128((const __m128i*)(ptr + 2 * stride));
    __m128i _r3 = _mm_loadu_si128((const __m128i*)(ptr + 3 * stride));
    __m128i _r4 = _mm_loadu_si128((const __m128i*)(ptr + 4 * stride));
    __m128i _r5 = _mm_loadu_si128((const __m128i*)(ptr + 5 * stride));
    __m128i _r6 = _mm_loadu_si128((const __m128i*)(ptr + 6 * stride));
    __m128i _r7 = _mm_loadu_si128((const __m128i*)(ptr + 7 * stride));
    transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
    _mm_storeu_si128((__m128i*)(outptr), _r0);
    _mm_storeu_si128((__m128i*)(outptr + outstride), _r1);
    _mm_storeu_si128((__m128i*)(outptr + 2 * outstride), _r2);
    _mm_storeu_si128((__m128i*)(outptr + 3 * outstride), _r3);
    _mm_storeu_si128((__m128i*)(outptr + 4 * outstride), _r4);
    _mm_storeu_si128((__m128i*)(outptr + 5 * outstride), _r5);
    _mm_storeu_si128((__m128i*)(outptr + 6 * outstride), _r6);
    _mm_storeu_si128((__m128i*)(outptr + 7 * outstride), _r7);
}
#endif // __SSE2__

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose8x16_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    permute_transpose8x8_stride_bf16s_fp16s(ptr, stride, outptr, outstride);
    permute_transpose8x8_stride_bf16s_fp16s(ptr + 8, stride, outptr + 8 * outstride, outstride);
}
#endif // __AVX512F__

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose16x2_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    int _p0;
    memcpy(&_p0, ptr, 4);
    __m128i _r0 = _mm_cvtsi32_si128(_p0);
    int _p1;
    memcpy(&_p1, ptr + stride, 4);
    __m128i _r1 = _mm_cvtsi32_si128(_p1);
    int _p2;
    memcpy(&_p2, ptr + 2 * stride, 4);
    __m128i _r2 = _mm_cvtsi32_si128(_p2);
    int _p3;
    memcpy(&_p3, ptr + 3 * stride, 4);
    __m128i _r3 = _mm_cvtsi32_si128(_p3);
    __m128i _t0 = _mm_unpacklo_epi32(_r0, _r1);
    __m128i _t1 = _mm_unpacklo_epi32(_r2, _r3);
    __m128i _v0 = _mm_unpacklo_epi64(_t0, _t1);
    _v0 = _mm_shufflelo_epi16(_v0, _MM_SHUFFLE(3, 1, 2, 0));
    _v0 = _mm_shufflehi_epi16(_v0, _MM_SHUFFLE(3, 1, 2, 0));
    _v0 = _mm_shuffle_epi32(_v0, _MM_SHUFFLE(3, 1, 2, 0));
    int _p4;
    memcpy(&_p4, ptr + 4 * stride, 4);
    __m128i _r4 = _mm_cvtsi32_si128(_p4);
    int _p5;
    memcpy(&_p5, ptr + 5 * stride, 4);
    __m128i _r5 = _mm_cvtsi32_si128(_p5);
    int _p6;
    memcpy(&_p6, ptr + 6 * stride, 4);
    __m128i _r6 = _mm_cvtsi32_si128(_p6);
    int _p7;
    memcpy(&_p7, ptr + 7 * stride, 4);
    __m128i _r7 = _mm_cvtsi32_si128(_p7);
    __m128i _t2 = _mm_unpacklo_epi32(_r4, _r5);
    __m128i _t3 = _mm_unpacklo_epi32(_r6, _r7);
    __m128i _v1 = _mm_unpacklo_epi64(_t2, _t3);
    _v1 = _mm_shufflelo_epi16(_v1, _MM_SHUFFLE(3, 1, 2, 0));
    _v1 = _mm_shufflehi_epi16(_v1, _MM_SHUFFLE(3, 1, 2, 0));
    _v1 = _mm_shuffle_epi32(_v1, _MM_SHUFFLE(3, 1, 2, 0));
    int _p8;
    memcpy(&_p8, ptr + 8 * stride, 4);
    __m128i _r8 = _mm_cvtsi32_si128(_p8);
    int _p9;
    memcpy(&_p9, ptr + 9 * stride, 4);
    __m128i _r9 = _mm_cvtsi32_si128(_p9);
    int _p10;
    memcpy(&_p10, ptr + 10 * stride, 4);
    __m128i _r10 = _mm_cvtsi32_si128(_p10);
    int _p11;
    memcpy(&_p11, ptr + 11 * stride, 4);
    __m128i _r11 = _mm_cvtsi32_si128(_p11);
    __m128i _t4 = _mm_unpacklo_epi32(_r8, _r9);
    __m128i _t5 = _mm_unpacklo_epi32(_r10, _r11);
    __m128i _v2 = _mm_unpacklo_epi64(_t4, _t5);
    _v2 = _mm_shufflelo_epi16(_v2, _MM_SHUFFLE(3, 1, 2, 0));
    _v2 = _mm_shufflehi_epi16(_v2, _MM_SHUFFLE(3, 1, 2, 0));
    _v2 = _mm_shuffle_epi32(_v2, _MM_SHUFFLE(3, 1, 2, 0));
    int _p12;
    memcpy(&_p12, ptr + 12 * stride, 4);
    __m128i _r12 = _mm_cvtsi32_si128(_p12);
    int _p13;
    memcpy(&_p13, ptr + 13 * stride, 4);
    __m128i _r13 = _mm_cvtsi32_si128(_p13);
    int _p14;
    memcpy(&_p14, ptr + 14 * stride, 4);
    __m128i _r14 = _mm_cvtsi32_si128(_p14);
    int _p15;
    memcpy(&_p15, ptr + 15 * stride, 4);
    __m128i _r15 = _mm_cvtsi32_si128(_p15);
    __m128i _t6 = _mm_unpacklo_epi32(_r12, _r13);
    __m128i _t7 = _mm_unpacklo_epi32(_r14, _r15);
    __m128i _v3 = _mm_unpacklo_epi64(_t6, _t7);
    _v3 = _mm_shufflelo_epi16(_v3, _MM_SHUFFLE(3, 1, 2, 0));
    _v3 = _mm_shufflehi_epi16(_v3, _MM_SHUFFLE(3, 1, 2, 0));
    _v3 = _mm_shuffle_epi32(_v3, _MM_SHUFFLE(3, 1, 2, 0));
    __m128i _a0 = _mm_unpacklo_epi64(_v0, _v1);
    __m128i _a1 = _mm_unpacklo_epi64(_v2, _v3);
    __m128i _b0 = _mm_unpackhi_epi64(_v0, _v1);
    __m128i _b1 = _mm_unpackhi_epi64(_v2, _v3);
    __m256i _a = _mm256_insertf128_si256(_mm256_castsi128_si256(_a0), _a1, 1);
    __m256i _b = _mm256_insertf128_si256(_mm256_castsi128_si256(_b0), _b1, 1);
    _mm256_storeu_si256((__m256i*)outptr, _a);
    _mm256_storeu_si256((__m256i*)(outptr + outstride), _b);
}
#endif // __AVX512F__

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose16x4_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    permute_transpose8x4_stride_bf16s_fp16s(ptr, stride, outptr, outstride);
    permute_transpose8x4_stride_bf16s_fp16s(ptr + 8 * stride, stride, outptr + 8, outstride);
}
#endif // __AVX512F__

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose16x8_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    permute_transpose8x8_stride_bf16s_fp16s(ptr, stride, outptr, outstride);
    permute_transpose8x8_stride_bf16s_fp16s(ptr + 8 * stride, stride, outptr + 8, outstride);
}
#endif // __AVX512F__

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose16x16_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    __m256i _r0 = _mm256_loadu_si256((const __m256i*)(ptr));
    __m256i _r1 = _mm256_loadu_si256((const __m256i*)(ptr + stride));
    __m256i _r2 = _mm256_loadu_si256((const __m256i*)(ptr + 2 * stride));
    __m256i _r3 = _mm256_loadu_si256((const __m256i*)(ptr + 3 * stride));
    __m256i _r4 = _mm256_loadu_si256((const __m256i*)(ptr + 4 * stride));
    __m256i _r5 = _mm256_loadu_si256((const __m256i*)(ptr + 5 * stride));
    __m256i _r6 = _mm256_loadu_si256((const __m256i*)(ptr + 6 * stride));
    __m256i _r7 = _mm256_loadu_si256((const __m256i*)(ptr + 7 * stride));
    __m256i _r8 = _mm256_loadu_si256((const __m256i*)(ptr + 8 * stride));
    __m256i _r9 = _mm256_loadu_si256((const __m256i*)(ptr + 9 * stride));
    __m256i _ra = _mm256_loadu_si256((const __m256i*)(ptr + 10 * stride));
    __m256i _rb = _mm256_loadu_si256((const __m256i*)(ptr + 11 * stride));
    __m256i _rc = _mm256_loadu_si256((const __m256i*)(ptr + 12 * stride));
    __m256i _rd = _mm256_loadu_si256((const __m256i*)(ptr + 13 * stride));
    __m256i _re = _mm256_loadu_si256((const __m256i*)(ptr + 14 * stride));
    __m256i _rf = _mm256_loadu_si256((const __m256i*)(ptr + 15 * stride));
    transpose16x16_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
    _mm256_storeu_si256((__m256i*)(outptr), _r0);
    _mm256_storeu_si256((__m256i*)(outptr + outstride), _r1);
    _mm256_storeu_si256((__m256i*)(outptr + 2 * outstride), _r2);
    _mm256_storeu_si256((__m256i*)(outptr + 3 * outstride), _r3);
    _mm256_storeu_si256((__m256i*)(outptr + 4 * outstride), _r4);
    _mm256_storeu_si256((__m256i*)(outptr + 5 * outstride), _r5);
    _mm256_storeu_si256((__m256i*)(outptr + 6 * outstride), _r6);
    _mm256_storeu_si256((__m256i*)(outptr + 7 * outstride), _r7);
    _mm256_storeu_si256((__m256i*)(outptr + 8 * outstride), _r8);
    _mm256_storeu_si256((__m256i*)(outptr + 9 * outstride), _r9);
    _mm256_storeu_si256((__m256i*)(outptr + 10 * outstride), _ra);
    _mm256_storeu_si256((__m256i*)(outptr + 11 * outstride), _rb);
    _mm256_storeu_si256((__m256i*)(outptr + 12 * outstride), _rc);
    _mm256_storeu_si256((__m256i*)(outptr + 13 * outstride), _rd);
    _mm256_storeu_si256((__m256i*)(outptr + 14 * outstride), _re);
    _mm256_storeu_si256((__m256i*)(outptr + 15 * outstride), _rf);
}
#endif // __AVX512F__

// Final two or one rows: load complete vectors and write only the valid output lanes.
#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose2x16_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    __m256i _r0 = _mm256_loadu_si256((const __m256i*)ptr);
    __m256i _r1 = _mm256_loadu_si256((const __m256i*)(ptr + stride));
    __m256i _t0 = _mm256_unpacklo_epi16(_r0, _r1);
    __m256i _t1 = _mm256_unpackhi_epi16(_r0, _r1);
    __m128i _v0 = _mm256_castsi256_si128(_t0);
    int _p0 = _mm_cvtsi128_si32(_v0);
    memcpy(outptr, &_p0, 4);
    int _p1 = _mm_cvtsi128_si32(_mm_srli_si128(_v0, 4));
    memcpy(outptr + outstride, &_p1, 4);
    int _p2 = _mm_cvtsi128_si32(_mm_srli_si128(_v0, 8));
    memcpy(outptr + 2 * outstride, &_p2, 4);
    int _p3 = _mm_cvtsi128_si32(_mm_srli_si128(_v0, 12));
    memcpy(outptr + 3 * outstride, &_p3, 4);
    __m128i _v1 = _mm256_castsi256_si128(_t1);
    int _p4 = _mm_cvtsi128_si32(_v1);
    memcpy(outptr + 4 * outstride, &_p4, 4);
    int _p5 = _mm_cvtsi128_si32(_mm_srli_si128(_v1, 4));
    memcpy(outptr + 5 * outstride, &_p5, 4);
    int _p6 = _mm_cvtsi128_si32(_mm_srli_si128(_v1, 8));
    memcpy(outptr + 6 * outstride, &_p6, 4);
    int _p7 = _mm_cvtsi128_si32(_mm_srli_si128(_v1, 12));
    memcpy(outptr + 7 * outstride, &_p7, 4);
    __m128i _v2 = _mm256_extractf128_si256(_t0, 1);
    int _p8 = _mm_cvtsi128_si32(_v2);
    memcpy(outptr + 8 * outstride, &_p8, 4);
    int _p9 = _mm_cvtsi128_si32(_mm_srli_si128(_v2, 4));
    memcpy(outptr + 9 * outstride, &_p9, 4);
    int _p10 = _mm_cvtsi128_si32(_mm_srli_si128(_v2, 8));
    memcpy(outptr + 10 * outstride, &_p10, 4);
    int _p11 = _mm_cvtsi128_si32(_mm_srli_si128(_v2, 12));
    memcpy(outptr + 11 * outstride, &_p11, 4);
    __m128i _v3 = _mm256_extractf128_si256(_t1, 1);
    int _p12 = _mm_cvtsi128_si32(_v3);
    memcpy(outptr + 12 * outstride, &_p12, 4);
    int _p13 = _mm_cvtsi128_si32(_mm_srli_si128(_v3, 4));
    memcpy(outptr + 13 * outstride, &_p13, 4);
    int _p14 = _mm_cvtsi128_si32(_mm_srli_si128(_v3, 8));
    memcpy(outptr + 14 * outstride, &_p14, 4);
    int _p15 = _mm_cvtsi128_si32(_mm_srli_si128(_v3, 12));
    memcpy(outptr + 15 * outstride, &_p15, 4);
}
#endif // __AVX512F__

#if __SSE2__
static NCNN_FORCEINLINE void permute_transpose2x8_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    __m128i _r0 = _mm_loadu_si128((const __m128i*)ptr);
    __m128i _r1 = _mm_loadu_si128((const __m128i*)(ptr + stride));
    __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
    __m128i _t1 = _mm_unpackhi_epi16(_r0, _r1);
    __m128i _v0 = _t0;
    int _p0 = _mm_cvtsi128_si32(_v0);
    memcpy(outptr, &_p0, 4);
    int _p1 = _mm_cvtsi128_si32(_mm_srli_si128(_v0, 4));
    memcpy(outptr + outstride, &_p1, 4);
    int _p2 = _mm_cvtsi128_si32(_mm_srli_si128(_v0, 8));
    memcpy(outptr + 2 * outstride, &_p2, 4);
    int _p3 = _mm_cvtsi128_si32(_mm_srli_si128(_v0, 12));
    memcpy(outptr + 3 * outstride, &_p3, 4);
    __m128i _v1 = _t1;
    int _p4 = _mm_cvtsi128_si32(_v1);
    memcpy(outptr + 4 * outstride, &_p4, 4);
    int _p5 = _mm_cvtsi128_si32(_mm_srli_si128(_v1, 4));
    memcpy(outptr + 5 * outstride, &_p5, 4);
    int _p6 = _mm_cvtsi128_si32(_mm_srli_si128(_v1, 8));
    memcpy(outptr + 6 * outstride, &_p6, 4);
    int _p7 = _mm_cvtsi128_si32(_mm_srli_si128(_v1, 12));
    memcpy(outptr + 7 * outstride, &_p7, 4);
}
#endif // __SSE2__

#if __SSE2__
static NCNN_FORCEINLINE void permute_transpose2x4_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    __m128i _r0 = _mm_loadl_epi64((const __m128i*)ptr);
    __m128i _r1 = _mm_loadl_epi64((const __m128i*)(ptr + stride));
    __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
    __m128i _v0 = _t0;
    int _p0 = _mm_cvtsi128_si32(_v0);
    memcpy(outptr, &_p0, 4);
    int _p1 = _mm_cvtsi128_si32(_mm_srli_si128(_v0, 4));
    memcpy(outptr + outstride, &_p1, 4);
    int _p2 = _mm_cvtsi128_si32(_mm_srli_si128(_v0, 8));
    memcpy(outptr + 2 * outstride, &_p2, 4);
    int _p3 = _mm_cvtsi128_si32(_mm_srli_si128(_v0, 12));
    memcpy(outptr + 3 * outstride, &_p3, 4);
}
#endif // __SSE2__

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose1x16_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, size_t outstride)
{
    __m256i _r0 = _mm256_loadu_si256((const __m256i*)ptr);
    __m128i _v0 = _mm256_castsi256_si128(_r0);
    outptr[0] = (unsigned short)_mm_extract_epi16(_v0, 0);
    outptr[1 * outstride] = (unsigned short)_mm_extract_epi16(_v0, 1);
    outptr[2 * outstride] = (unsigned short)_mm_extract_epi16(_v0, 2);
    outptr[3 * outstride] = (unsigned short)_mm_extract_epi16(_v0, 3);
    outptr[4 * outstride] = (unsigned short)_mm_extract_epi16(_v0, 4);
    outptr[5 * outstride] = (unsigned short)_mm_extract_epi16(_v0, 5);
    outptr[6 * outstride] = (unsigned short)_mm_extract_epi16(_v0, 6);
    outptr[7 * outstride] = (unsigned short)_mm_extract_epi16(_v0, 7);
    __m128i _v1 = _mm256_extractf128_si256(_r0, 1);
    outptr[8 * outstride] = (unsigned short)_mm_extract_epi16(_v1, 0);
    outptr[9 * outstride] = (unsigned short)_mm_extract_epi16(_v1, 1);
    outptr[10 * outstride] = (unsigned short)_mm_extract_epi16(_v1, 2);
    outptr[11 * outstride] = (unsigned short)_mm_extract_epi16(_v1, 3);
    outptr[12 * outstride] = (unsigned short)_mm_extract_epi16(_v1, 4);
    outptr[13 * outstride] = (unsigned short)_mm_extract_epi16(_v1, 5);
    outptr[14 * outstride] = (unsigned short)_mm_extract_epi16(_v1, 6);
    outptr[15 * outstride] = (unsigned short)_mm_extract_epi16(_v1, 7);
}
#endif // __AVX512F__

#if __SSE2__
static NCNN_FORCEINLINE void permute_transpose1x8_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, size_t outstride)
{
    __m128i _r0 = _mm_loadu_si128((const __m128i*)ptr);
    __m128i _v0 = _r0;
    outptr[0] = (unsigned short)_mm_extract_epi16(_v0, 0);
    outptr[1 * outstride] = (unsigned short)_mm_extract_epi16(_v0, 1);
    outptr[2 * outstride] = (unsigned short)_mm_extract_epi16(_v0, 2);
    outptr[3 * outstride] = (unsigned short)_mm_extract_epi16(_v0, 3);
    outptr[4 * outstride] = (unsigned short)_mm_extract_epi16(_v0, 4);
    outptr[5 * outstride] = (unsigned short)_mm_extract_epi16(_v0, 5);
    outptr[6 * outstride] = (unsigned short)_mm_extract_epi16(_v0, 6);
    outptr[7 * outstride] = (unsigned short)_mm_extract_epi16(_v0, 7);
}
#endif // __SSE2__

#if __SSE2__
static NCNN_FORCEINLINE void permute_transpose1x4_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, size_t outstride)
{
    __m128i _r0 = _mm_loadl_epi64((const __m128i*)ptr);
    __m128i _v0 = _r0;
    outptr[0] = (unsigned short)_mm_extract_epi16(_v0, 0);
    outptr[1 * outstride] = (unsigned short)_mm_extract_epi16(_v0, 1);
    outptr[2 * outstride] = (unsigned short)_mm_extract_epi16(_v0, 2);
    outptr[3 * outstride] = (unsigned short)_mm_extract_epi16(_v0, 3);
}
#endif // __SSE2__

// Unpacked matrix transpose, shared by 2d and channel/spatial permutations.
static void permute_transpose_pack1_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    int i = 0;
#if __AVX512F__
    for (; i + 15 < rows; i += 16)
    {
        int j = 0;
        for (; j + 15 < cols; j += 16)
        {
            permute_transpose16x16_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 7 < cols; j += 8)
        {
            permute_transpose16x8_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose16x4_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 1 < cols; j += 2)
        {
            permute_transpose16x2_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
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
#if __SSE2__
    for (; i + 7 < rows; i += 8)
    {
        int j = 0;
#if __AVX512F__
        for (; j + 15 < cols; j += 16)
        {
            permute_transpose8x16_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
#endif // __AVX512F__
        for (; j + 7 < cols; j += 8)
        {
            permute_transpose8x8_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose8x4_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 1 < cols; j += 2)
        {
            permute_transpose8x2_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j < cols; j++)
        {
            for (int k = 0; k < 8; k++)
            {
                outptr[j * outstride + i + k] = ptr[(i + k) * stride + j];
            }
        }
    }
#endif // __SSE2__
#if __SSE2__
    for (; i + 3 < rows; i += 4)
    {
        int j = 0;
#if __AVX512F__
        for (; j + 15 < cols; j += 16)
        {
            permute_transpose4x16_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
#endif // __AVX512F__
        for (; j + 7 < cols; j += 8)
        {
            permute_transpose4x8_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose4x4_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 1 < cols; j += 2)
        {
            permute_transpose4x2_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
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
            permute_transpose2x16_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
#endif // __AVX512F__
#if __SSE2__
        for (; j + 7 < cols; j += 8)
        {
            permute_transpose2x8_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
#endif // __SSE2__
#if __SSE2__
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose2x4_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
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
            permute_transpose1x16_stride_bf16s_fp16s(ptr + i * stride + j, outptr + j * outstride + i, outstride);
        }
#endif // __AVX512F__
#if __SSE2__
        for (; j + 7 < cols; j += 8)
        {
            permute_transpose1x8_stride_bf16s_fp16s(ptr + i * stride + j, outptr + j * outstride + i, outstride);
        }
#endif // __SSE2__
#if __SSE2__
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose1x4_stride_bf16s_fp16s(ptr + i * stride + j, outptr + j * outstride + i, outstride);
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
static void permute_transpose_pack1to4_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int x = 0; x < cols; x += 4)
    {
        for (int y = 0; y < rows; y++)
        {
            const unsigned short* p = ptr + y * stride + x;
            unsigned short* out = outptr + (x / 4) * outstride + y * 4;
            __m128i _v = _mm_loadl_epi64((const __m128i*)p);
            _mm_storel_epi64((__m128i*)out, _v);
        }
    }
}
#endif // __SSE2__

#if __AVX__
static void permute_transpose_pack1to8_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int x = 0; x < cols; x += 8)
    {
        for (int y = 0; y < rows; y++)
        {
            const unsigned short* p = ptr + y * stride + x;
            unsigned short* out = outptr + (x / 8) * outstride + y * 8;
            __m128i _v = _mm_loadu_si128((const __m128i*)p);
            _mm_storeu_si128((__m128i*)out, _v);
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute_transpose_pack1to16_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int x = 0; x < cols; x += 16)
    {
        for (int y = 0; y < rows; y++)
        {
            const unsigned short* p = ptr + y * stride + x;
            unsigned short* out = outptr + (x / 16) * outstride + y * 16;
            __m256 _v = _mm256_loadu_ps((const float*)p);
            _mm256_storeu_ps((float*)out, _v);
        }
    }
}
#endif // __AVX512F__

#if __SSE2__
static void permute_transpose_pack4to1_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        for (int x = 0; x < cols; x++)
        {
            const unsigned short* p = ptr + (y / 4) * stride + x * 4;
            unsigned short* out = outptr + x * outstride + y;
            __m128i _v = _mm_loadl_epi64((const __m128i*)p);
            _mm_storel_epi64((__m128i*)out, _v);
        }
    }
}
#endif // __SSE2__

#if __SSE2__
static void permute_transpose_pack4to4_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        for (int x = 0; x < cols; x += 4)
        {
            const unsigned short* p = ptr + (y / 4) * stride + x * 4;
            unsigned short* out = outptr + (x / 4) * outstride + y * 4;
            permute_transpose4x4_bf16s_fp16s(p, out);
        }
    }
}
#endif // __SSE2__

#if __AVX__
static void permute_transpose_pack4to8_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        for (int x = 0; x < cols; x += 8)
        {
            const unsigned short* p = ptr + (y / 4) * stride + x * 4;
            unsigned short* out = outptr + (x / 8) * outstride + y * 8;
            permute_transpose8x4_bf16s_fp16s(p, out);
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute_transpose_pack4to16_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        for (int x = 0; x < cols; x += 16)
        {
            const unsigned short* p = ptr + (y / 4) * stride + x * 4;
            unsigned short* out = outptr + (x / 16) * outstride + y * 16;
            permute_transpose16x4_stride_bf16s_fp16s(p, 4, out, 16);
        }
    }
}
#endif // __AVX512F__

#if __AVX__
static void permute_transpose_pack8to1_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        for (int x = 0; x < cols; x++)
        {
            const unsigned short* p = ptr + (y / 8) * stride + x * 8;
            unsigned short* out = outptr + x * outstride + y;
            __m128i _v = _mm_loadu_si128((const __m128i*)p);
            _mm_storeu_si128((__m128i*)out, _v);
        }
    }
}
#endif // __AVX__

#if __AVX__
static void permute_transpose_pack8to4_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        for (int x = 0; x < cols; x += 4)
        {
            const unsigned short* p = ptr + (y / 8) * stride + x * 8;
            unsigned short* out = outptr + (x / 4) * outstride + y * 4;
            permute_transpose4x8_bf16s_fp16s(p, out);
        }
    }
}
#endif // __AVX__

#if __AVX__
static void permute_transpose_pack8to8_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        for (int x = 0; x < cols; x += 8)
        {
            const unsigned short* p = ptr + (y / 8) * stride + x * 8;
            unsigned short* out = outptr + (x / 8) * outstride + y * 8;
            permute_transpose8x8_stride_bf16s_fp16s(p, 8, out, 8);
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute_transpose_pack8to16_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        for (int x = 0; x < cols; x += 16)
        {
            const unsigned short* p = ptr + (y / 8) * stride + x * 8;
            unsigned short* out = outptr + (x / 16) * outstride + y * 16;
            permute_transpose16x8_stride_bf16s_fp16s(p, 8, out, 16);
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute_transpose_pack16to1_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 16)
    {
        for (int x = 0; x < cols; x++)
        {
            const unsigned short* p = ptr + (y / 16) * stride + x * 16;
            unsigned short* out = outptr + x * outstride + y;
            __m256 _v = _mm256_loadu_ps((const float*)p);
            _mm256_storeu_ps((float*)out, _v);
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute_transpose_pack16to4_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 16)
    {
        for (int x = 0; x < cols; x += 4)
        {
            const unsigned short* p = ptr + (y / 16) * stride + x * 16;
            unsigned short* out = outptr + (x / 4) * outstride + y * 4;
            permute_transpose4x16_stride_bf16s_fp16s(p, 16, out, 4);
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute_transpose_pack16to8_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 16)
    {
        for (int x = 0; x < cols; x += 8)
        {
            const unsigned short* p = ptr + (y / 16) * stride + x * 16;
            unsigned short* out = outptr + (x / 8) * outstride + y * 8;
            permute_transpose8x16_stride_bf16s_fp16s(p, 16, out, 8);
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute_transpose_pack16to16_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 16)
    {
        for (int x = 0; x < cols; x += 16)
        {
            const unsigned short* p = ptr + (y / 16) * stride + x * 16;
            unsigned short* out = outptr + (x / 16) * outstride + y * 16;
            permute_transpose16x16_stride_bf16s_fp16s(p, 16, out, 16);
        }
    }
}
#endif // __AVX512F__

static void permute_transpose2d_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols, int elempack, int out_elempack)
{
    if (elempack == 1 && out_elempack == 1)
    {
        permute_transpose_pack1_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

#if __SSE2__
    if (elempack == 1 && out_elempack == 4)
    {
        permute_transpose_pack1to4_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __SSE2__

#if __AVX__
    if (elempack == 1 && out_elempack == 8)
    {
        permute_transpose_pack1to8_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 1 && out_elempack == 16)
    {
        permute_transpose_pack1to16_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__

#if __SSE2__
    if (elempack == 4 && out_elempack == 1)
    {
        permute_transpose_pack4to1_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __SSE2__

#if __SSE2__
    if (elempack == 4 && out_elempack == 4)
    {
        permute_transpose_pack4to4_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __SSE2__

#if __AVX__
    if (elempack == 4 && out_elempack == 8)
    {
        permute_transpose_pack4to8_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 4 && out_elempack == 16)
    {
        permute_transpose_pack4to16_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__

#if __AVX__
    if (elempack == 8 && out_elempack == 1)
    {
        permute_transpose_pack8to1_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX__

#if __AVX__
    if (elempack == 8 && out_elempack == 4)
    {
        permute_transpose_pack8to4_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX__

#if __AVX__
    if (elempack == 8 && out_elempack == 8)
    {
        permute_transpose_pack8to8_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 8 && out_elempack == 16)
    {
        permute_transpose_pack8to16_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 1)
    {
        permute_transpose_pack16to1_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 4)
    {
        permute_transpose_pack16to4_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 8)
    {
        permute_transpose_pack16to8_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 16)
    {
        permute_transpose_pack16to16_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__
}

// Spatial transpose within one input channel group. outcstep is used when unpacking.
#if __SSE2__
static void permute_spatial_pack4_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        const int ymax = std::min(y + 8, rows);
        for (int x = 0; x < cols; x += 8)
        {
            const int xmax = std::min(x + 8, cols);
            for (int i = y; i < ymax; i++)
            {
                const unsigned short* p = ptr + i * stride + x * 4;
                unsigned short* out = outptr + x * outstride + i * 4;
                for (int j = x; j < xmax; j++)
                {
                    __m128i _v = _mm_loadl_epi64((const __m128i*)p);
                    _mm_storel_epi64((__m128i*)out, _v);
                    p += 4;
                    out += outstride;
                }
            }
        }
    }
}
#endif // __SSE2__

#if __AVX__
static void permute_spatial_pack8_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        const int ymax = std::min(y + 8, rows);
        for (int x = 0; x < cols; x += 8)
        {
            const int xmax = std::min(x + 8, cols);
            for (int i = y; i < ymax; i++)
            {
                const unsigned short* p = ptr + i * stride + x * 8;
                unsigned short* out = outptr + x * outstride + i * 8;
                for (int j = x; j < xmax; j++)
                {
                    __m128i _v = _mm_loadu_si128((const __m128i*)p);
                    _mm_storeu_si128((__m128i*)out, _v);
                    p += 8;
                    out += outstride;
                }
            }
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute_spatial_pack16_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        const int ymax = std::min(y + 8, rows);
        for (int x = 0; x < cols; x += 8)
        {
            const int xmax = std::min(x + 8, cols);
            for (int i = y; i < ymax; i++)
            {
                const unsigned short* p = ptr + i * stride + x * 16;
                unsigned short* out = outptr + x * outstride + i * 16;
                for (int j = x; j < xmax; j++)
                {
                    __m256 _v = _mm256_loadu_ps((const float*)p);
                    _mm256_storeu_ps((float*)out, _v);
                    p += 16;
                    out += outstride;
                }
            }
        }
    }
}
#endif // __AVX512F__

static void permute_transpose_spatial_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, size_t outcstep, int rows, int cols, int elempack, int out_elempack)
{
    if (elempack == 1)
    {
        permute_transpose_pack1_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#if __SSE2__
    if (elempack == 4 && out_elempack == 4)
    {
        permute_spatial_pack4_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
    if (elempack == 4 && out_elempack == 1)
    {
        for (int x = 0; x < cols; x++)
            permute_transpose_pack1_bf16s_fp16s(ptr + x * 4, stride, outptr + x * outstride, outcstep, rows, 4);
        return;
    }
#endif // __SSE2__

#if __AVX__
    if (elempack == 8 && out_elempack == 8)
    {
        permute_spatial_pack8_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
    if (elempack == 8 && out_elempack == 1)
    {
        for (int x = 0; x < cols; x++)
            permute_transpose_pack1_bf16s_fp16s(ptr + x * 8, stride, outptr + x * outstride, outcstep, rows, 8);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 16)
    {
        permute_spatial_pack16_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
    if (elempack == 16 && out_elempack == 1)
    {
        for (int x = 0; x < cols; x++)
            permute_transpose_pack1_bf16s_fp16s(ptr + x * 16, stride, outptr + x * outstride, outcstep, rows, 16);
        return;
    }
#endif // __AVX512F__
}

static void permute_copy_spatial_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, size_t outcstep, int size, int elempack, int out_elempack)
{
    if (elempack == out_elempack)
    {
        memcpy(outptr, ptr, (size_t)size * elempack * sizeof(unsigned short));
        return;
    }
#if __SSE2__
    if (elempack == 4)
    {
        permute_transpose_pack1_bf16s_fp16s(ptr, 4, outptr, outcstep, size, 4);
        return;
    }
#endif // __SSE2__

#if __AVX__
    if (elempack == 8)
    {
        permute_transpose_pack1_bf16s_fp16s(ptr, 8, outptr, outcstep, size, 8);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 16)
    {
        permute_transpose_pack1_bf16s_fp16s(ptr, 16, outptr, outcstep, size, 16);
        return;
    }
#endif // __AVX512F__
}

// Exchange the input channel axis with h. w is the remaining spatial axis.
// Strides are in scalar elements. cstep and outhstep include channel padding.
// Pack1 input uses contiguous w or h; pack1 output uses contiguous w or c.
#if __SSE2__
static void permute3d_pack1to4_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (hstep != 1)
    {
        for (int q = 0; q < h / 4; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const unsigned short* p = ptr + c * cstep + q * 4 * hstep;
                unsigned short* out = outptr + q * outhstep + c * outcstep;
                permute_transpose_pack1_bf16s_fp16s(p, hstep, out, outwstep, 4, w);
            }
        }
        return;
    }
    for (int q = 0; q < h / 4; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned short* p = ptr + c * cstep + q * 4 * hstep;
            unsigned short* out = outptr + q * outhstep + c * outcstep;
            for (int x = 0; x < w; x++)
            {
                __m128i _v = _mm_loadl_epi64((const __m128i*)p);
                _mm_storel_epi64((__m128i*)out, _v);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __SSE2__

#if __AVX__
static void permute3d_pack1to8_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (hstep != 1)
    {
        for (int q = 0; q < h / 8; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const unsigned short* p = ptr + c * cstep + q * 8 * hstep;
                unsigned short* out = outptr + q * outhstep + c * outcstep;
                permute_transpose_pack1_bf16s_fp16s(p, hstep, out, outwstep, 8, w);
            }
        }
        return;
    }
    for (int q = 0; q < h / 8; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned short* p = ptr + c * cstep + q * 8 * hstep;
            unsigned short* out = outptr + q * outhstep + c * outcstep;
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
#endif // __AVX__

#if __AVX512F__
static void permute3d_pack1to16_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (hstep != 1)
    {
        for (int q = 0; q < h / 16; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const unsigned short* p = ptr + c * cstep + q * 16 * hstep;
                unsigned short* out = outptr + q * outhstep + c * outcstep;
                permute_transpose_pack1_bf16s_fp16s(p, hstep, out, outwstep, 16, w);
            }
        }
        return;
    }
    for (int q = 0; q < h / 16; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned short* p = ptr + c * cstep + q * 16 * hstep;
            unsigned short* out = outptr + q * outhstep + c * outcstep;
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
#endif // __AVX512F__

#if __SSE2__
static void permute3d_pack4to1_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (outcstep != 1)
    {
        for (int q = 0; q < h; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const unsigned short* p = ptr + c * cstep + q * hstep;
                unsigned short* out = outptr + q * outhstep + c * 4 * outcstep;
                permute_transpose_pack1_bf16s_fp16s(p, wstep, out, outcstep, w, 4);
            }
        }
        return;
    }
    for (int q = 0; q < h; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned short* p = ptr + c * cstep + q * hstep;
            unsigned short* out = outptr + q * outhstep + c * 4 * outcstep;
            for (int x = 0; x < w; x++)
            {
                __m128i _v = _mm_loadl_epi64((const __m128i*)p);
                _mm_storel_epi64((__m128i*)out, _v);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __SSE2__

#if __SSE2__
static void permute3d_pack4to4_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (hstep == 4 && outcstep == 4)
    {
        for (int q = 0; q < h / 4; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const unsigned short* p = ptr + c * cstep + q * 4 * hstep;
                unsigned short* out = outptr + q * outhstep + c * 4 * outcstep;
                for (int x = 0; x < w; x++)
                {
                    permute_transpose4x4_bf16s_fp16s(p, out);
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
            const unsigned short* p = ptr + c * cstep + q * 4 * hstep;
            unsigned short* out = outptr + q * outhstep + c * 4 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose4x4_stride_bf16s_fp16s(p, hstep, out, outcstep);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __SSE2__

#if __AVX__
static void permute3d_pack4to8_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (hstep == 4 && outcstep == 8)
    {
        for (int q = 0; q < h / 8; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const unsigned short* p = ptr + c * cstep + q * 8 * hstep;
                unsigned short* out = outptr + q * outhstep + c * 4 * outcstep;
                for (int x = 0; x < w; x++)
                {
                    permute_transpose8x4_bf16s_fp16s(p, out);
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
            const unsigned short* p = ptr + c * cstep + q * 8 * hstep;
            unsigned short* out = outptr + q * outhstep + c * 4 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose8x4_stride_bf16s_fp16s(p, hstep, out, outcstep);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute3d_pack4to16_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    for (int q = 0; q < h / 16; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned short* p = ptr + c * cstep + q * 16 * hstep;
            unsigned short* out = outptr + q * outhstep + c * 4 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose16x4_stride_bf16s_fp16s(p, hstep, out, outcstep);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX512F__

#if __AVX__
static void permute3d_pack8to1_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (outcstep != 1)
    {
        for (int q = 0; q < h; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const unsigned short* p = ptr + c * cstep + q * hstep;
                unsigned short* out = outptr + q * outhstep + c * 8 * outcstep;
                permute_transpose_pack1_bf16s_fp16s(p, wstep, out, outcstep, w, 8);
            }
        }
        return;
    }
    for (int q = 0; q < h; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned short* p = ptr + c * cstep + q * hstep;
            unsigned short* out = outptr + q * outhstep + c * 8 * outcstep;
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
#endif // __AVX__

#if __AVX__
static void permute3d_pack8to4_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (hstep == 8 && outcstep == 4)
    {
        for (int q = 0; q < h / 4; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const unsigned short* p = ptr + c * cstep + q * 4 * hstep;
                unsigned short* out = outptr + q * outhstep + c * 8 * outcstep;
                for (int x = 0; x < w; x++)
                {
                    permute_transpose4x8_bf16s_fp16s(p, out);
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
            const unsigned short* p = ptr + c * cstep + q * 4 * hstep;
            unsigned short* out = outptr + q * outhstep + c * 8 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose4x8_stride_bf16s_fp16s(p, hstep, out, outcstep);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX__

#if __AVX__
static void permute3d_pack8to8_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    for (int q = 0; q < h / 8; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned short* p = ptr + c * cstep + q * 8 * hstep;
            unsigned short* out = outptr + q * outhstep + c * 8 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose8x8_stride_bf16s_fp16s(p, hstep, out, outcstep);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute3d_pack8to16_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    for (int q = 0; q < h / 16; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned short* p = ptr + c * cstep + q * 16 * hstep;
            unsigned short* out = outptr + q * outhstep + c * 8 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose16x8_stride_bf16s_fp16s(p, hstep, out, outcstep);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute3d_pack16to1_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    if (outcstep != 1)
    {
        for (int q = 0; q < h; q++)
        {
            for (int c = 0; c < channels; c++)
            {
                const unsigned short* p = ptr + c * cstep + q * hstep;
                unsigned short* out = outptr + q * outhstep + c * 16 * outcstep;
                permute_transpose_pack1_bf16s_fp16s(p, wstep, out, outcstep, w, 16);
            }
        }
        return;
    }
    for (int q = 0; q < h; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned short* p = ptr + c * cstep + q * hstep;
            unsigned short* out = outptr + q * outhstep + c * 16 * outcstep;
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
#endif // __AVX512F__

#if __AVX512F__
static void permute3d_pack16to4_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    for (int q = 0; q < h / 4; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned short* p = ptr + c * cstep + q * 4 * hstep;
            unsigned short* out = outptr + q * outhstep + c * 16 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose4x16_stride_bf16s_fp16s(p, hstep, out, outcstep);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute3d_pack16to8_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    for (int q = 0; q < h / 8; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned short* p = ptr + c * cstep + q * 8 * hstep;
            unsigned short* out = outptr + q * outhstep + c * 16 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose8x16_stride_bf16s_fp16s(p, hstep, out, outcstep);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute3d_pack16to16_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep)
{
    for (int q = 0; q < h / 16; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned short* p = ptr + c * cstep + q * 16 * hstep;
            unsigned short* out = outptr + q * outhstep + c * 16 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose16x16_stride_bf16s_fp16s(p, hstep, out, outcstep);
                p += wstep;
                out += outwstep;
            }
        }
    }
}
#endif // __AVX512F__

static void permute3d_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int h, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep, size_t outhstep, int elempack, int out_elempack)
{
#if __SSE2__
    if (elempack == 1 && out_elempack == 4)
    {
        permute3d_pack1to4_bf16s_fp16s(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __SSE2__

#if __AVX__
    if (elempack == 1 && out_elempack == 8)
    {
        permute3d_pack1to8_bf16s_fp16s(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 1 && out_elempack == 16)
    {
        permute3d_pack1to16_bf16s_fp16s(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX512F__

#if __SSE2__
    if (elempack == 4 && out_elempack == 1)
    {
        permute3d_pack4to1_bf16s_fp16s(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __SSE2__

#if __SSE2__
    if (elempack == 4 && out_elempack == 4)
    {
        permute3d_pack4to4_bf16s_fp16s(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __SSE2__

#if __AVX__
    if (elempack == 4 && out_elempack == 8)
    {
        permute3d_pack4to8_bf16s_fp16s(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 4 && out_elempack == 16)
    {
        permute3d_pack4to16_bf16s_fp16s(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX512F__

#if __AVX__
    if (elempack == 8 && out_elempack == 1)
    {
        permute3d_pack8to1_bf16s_fp16s(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX__

#if __AVX__
    if (elempack == 8 && out_elempack == 4)
    {
        permute3d_pack8to4_bf16s_fp16s(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX__

#if __AVX__
    if (elempack == 8 && out_elempack == 8)
    {
        permute3d_pack8to8_bf16s_fp16s(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 8 && out_elempack == 16)
    {
        permute3d_pack8to16_bf16s_fp16s(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 1)
    {
        permute3d_pack16to1_bf16s_fp16s(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 4)
    {
        permute3d_pack16to4_bf16s_fp16s(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 8)
    {
        permute3d_pack16to8_bf16s_fp16s(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 16)
    {
        permute3d_pack16to16_bf16s_fp16s(ptr, outptr, w, h, channels, wstep, hstep, cstep, outwstep, outcstep, outhstep);
        return;
    }
#endif // __AVX512F__
}

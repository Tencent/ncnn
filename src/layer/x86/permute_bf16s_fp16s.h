// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// full register tiles have no size or packing branches

// contiguous tiles take only pointers; stride variants use scalar-element strides

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

// final two or one rows: load complete vectors and write only the valid output lanes
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

#if __SSE2__
static void permute_pack4_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int cols)
{
    int j = 0;
#if __AVX512F__
    for (; j + 15 < cols; j += 16)
    {
        permute_transpose4x16_stride_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride);
    }
#endif // __AVX512F__
    for (; j + 7 < cols; j += 8)
    {
        permute_transpose4x8_stride_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j + 3 < cols; j += 4)
    {
        permute_transpose4x4_stride_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j + 1 < cols; j += 2)
    {
        permute_transpose4x2_stride_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j < cols; j++)
    {
        for (int k = 0; k < 4; k++)
            outptr[j * outstride + k] = ptr[k * stride + j];
    }
}
#endif // __SSE2__

#if __AVX__
static void permute_pack8_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int cols)
{
    int j = 0;
#if __AVX512F__
    for (; j + 15 < cols; j += 16)
    {
        permute_transpose8x16_stride_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride);
    }
#endif // __AVX512F__
    for (; j + 7 < cols; j += 8)
    {
        permute_transpose8x8_stride_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j + 3 < cols; j += 4)
    {
        permute_transpose8x4_stride_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j + 1 < cols; j += 2)
    {
        permute_transpose8x2_stride_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j < cols; j++)
    {
        for (int k = 0; k < 8; k++)
            outptr[j * outstride + k] = ptr[k * stride + j];
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute_pack16_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int cols)
{
    int j = 0;
    for (; j + 15 < cols; j += 16)
    {
        permute_transpose16x16_stride_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j + 7 < cols; j += 8)
    {
        permute_transpose16x8_stride_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j + 3 < cols; j += 4)
    {
        permute_transpose16x4_stride_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j + 1 < cols; j += 2)
    {
        permute_transpose16x2_stride_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j < cols; j++)
    {
        for (int k = 0; k < 16; k++)
            outptr[j * outstride + k] = ptr[k * stride + j];
    }
}
#endif // __AVX512F__

// unpacked matrix transpose, shared by 2d and channel/spatial permutations
#if __SSE2__
// fixed input width; callers select the packing before traversing the rows
static void permute_unpack4_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows)
{
    int i = 0;
#if __AVX512F__
    for (; i + 15 < rows; i += 16)
    {
        permute_transpose16x4_stride_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride);
    }
#endif // __AVX512F__
#if __AVX__
    for (; i + 7 < rows; i += 8)
    {
        permute_transpose8x4_stride_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride);
    }
#endif // __AVX__
    for (; i + 3 < rows; i += 4)
    {
        permute_transpose4x4_stride_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 1 < rows; i += 2)
    {
        permute_transpose2x4_stride_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i < rows; i++)
    {
        permute_transpose1x4_stride_bf16s_fp16s(ptr + i * stride, outptr + i, outstride);
    }
}
#endif // __SSE2__

#if __AVX__
// fixed input width; callers select the packing before traversing the rows
static void permute_unpack8_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows)
{
    int i = 0;
#if __AVX512F__
    for (; i + 15 < rows; i += 16)
    {
        permute_transpose16x8_stride_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride);
    }
#endif // __AVX512F__
    for (; i + 7 < rows; i += 8)
    {
        permute_transpose8x8_stride_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 3 < rows; i += 4)
    {
        permute_transpose4x8_stride_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 1 < rows; i += 2)
    {
        permute_transpose2x8_stride_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i < rows; i++)
    {
        permute_transpose1x8_stride_bf16s_fp16s(ptr + i * stride, outptr + i, outstride);
    }
}
#endif // __AVX__

#if __AVX512F__
// fixed input width; callers select the packing before traversing the rows
static void permute_unpack16_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows)
{
    int i = 0;
    for (; i + 15 < rows; i += 16)
    {
        permute_transpose16x16_stride_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 7 < rows; i += 8)
    {
        permute_transpose8x16_stride_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 3 < rows; i += 4)
    {
        permute_transpose4x16_stride_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 1 < rows; i += 2)
    {
        permute_transpose2x16_stride_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i < rows; i++)
    {
        permute_transpose1x16_stride_bf16s_fp16s(ptr + i * stride, outptr + i, outstride);
    }
}
#endif // __AVX512F__

static void permute_transpose_pack1_block_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
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

// cache blocking is useful for medium planes with regularly spaced rows
// small task blocks and narrow pack/unpack matrices use the direct kernel
static void permute_transpose_pack1_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    if (cols == 1)
    {
        if (stride == 1)
            memcpy(outptr, ptr, (size_t)rows * sizeof(unsigned short));
        else
            for (int i = 0; i < rows; i++)
                outptr[i] = ptr[i * stride];
        return;
    }
    if (rows == 1 && outstride == 1)
    {
        memcpy(outptr, ptr, (size_t)cols * sizeof(unsigned short));
        return;
    }

    // large planes keep an output stripe resident while scanning the input
    if (rows >= 512 && cols >= 16 && stride >= 512 && outstride >= 512)
    {
        for (int j = 0; j < cols; j += 32)
            permute_transpose_pack1_block_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride, rows, std::min(32, cols - j));
        return;
    }
    // narrow input rows already fit in a register-row sweep
    if (rows >= 16 && (size_t)cols * sizeof(unsigned short) >= 1024 && (rows > 512 || cols > 512))
    {
        // coalesced axes can form long rectangles
        // bound the tile payload to 16 KiB on each side, allowing wider tiles for fewer input rows
        const int row_block = std::min(rows, 64);
        const int col_block = std::min(256, 16384 / (row_block * (int)sizeof(unsigned short)));
        for (int j = 0; j < cols; j += col_block)
        {
            for (int i = 0; i < rows; i += row_block)
                permute_transpose_pack1_block_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride, std::min(row_block, rows - i), std::min(col_block, cols - j));
        }
        return;
    }

#if __SSE2__
    if (rows >= 256 && rows <= 512 && cols >= 256 && cols <= 512 && stride <= 1024 && outstride <= 1024
        && stride % 256 == 0 && outstride % 256 == 0)
    {
#if __AVX512F__
        const int block = rows * cols < 131072 ? 64 : 32;
#else
        const int block = 32;
#endif
        for (int i = 0; i < rows; i += block)
        {
            for (int j = 0; j < cols; j += block)
            {
                permute_transpose_pack1_block_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride, std::min(block, rows - i), std::min(block, cols - j));
            }
        }
        return;
    }
#endif // __SSE2__
    permute_transpose_pack1_block_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
}

// 2d: packed rows become packed output rows after transposing w and h
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
        for (int j = 0; j < cols; j += 32)
            permute_transpose_pack1to4_bf16s_fp16s(ptr + (size_t)j * 1, stride, outptr + (j / 4) * outstride, outstride, rows, std::min(32, cols - j));
        return;
    }
#endif // __SSE2__

#if __AVX__
    if (elempack == 1 && out_elempack == 8)
    {
        for (int j = 0; j < cols; j += 32)
            permute_transpose_pack1to8_bf16s_fp16s(ptr + (size_t)j * 1, stride, outptr + (j / 8) * outstride, outstride, rows, std::min(32, cols - j));
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 1 && out_elempack == 16)
    {
        for (int j = 0; j < cols; j += 32)
            permute_transpose_pack1to16_bf16s_fp16s(ptr + (size_t)j * 1, stride, outptr + (j / 16) * outstride, outstride, rows, std::min(32, cols - j));
        return;
    }
#endif // __AVX512F__

#if __SSE2__
    if (elempack == 4 && out_elempack == 1)
    {
        for (int j = 0; j < cols; j += 32)
            permute_transpose_pack4to1_bf16s_fp16s(ptr + (size_t)j * 4, stride, outptr + (j / 1) * outstride, outstride, rows, std::min(32, cols - j));
        return;
    }
#endif // __SSE2__

#if __SSE2__
    if (elempack == 4 && out_elempack == 4)
    {
        for (int j = 0; j < cols; j += 32)
            permute_transpose_pack4to4_bf16s_fp16s(ptr + (size_t)j * 4, stride, outptr + (j / 4) * outstride, outstride, rows, std::min(32, cols - j));
        return;
    }
#endif // __SSE2__

#if __AVX__
    if (elempack == 4 && out_elempack == 8)
    {
        for (int j = 0; j < cols; j += 32)
            permute_transpose_pack4to8_bf16s_fp16s(ptr + (size_t)j * 4, stride, outptr + (j / 8) * outstride, outstride, rows, std::min(32, cols - j));
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 4 && out_elempack == 16)
    {
        for (int j = 0; j < cols; j += 32)
            permute_transpose_pack4to16_bf16s_fp16s(ptr + (size_t)j * 4, stride, outptr + (j / 16) * outstride, outstride, rows, std::min(32, cols - j));
        return;
    }
#endif // __AVX512F__

#if __AVX__
    if (elempack == 8 && out_elempack == 1)
    {
        for (int j = 0; j < cols; j += 32)
            permute_transpose_pack8to1_bf16s_fp16s(ptr + (size_t)j * 8, stride, outptr + (j / 1) * outstride, outstride, rows, std::min(32, cols - j));
        return;
    }
#endif // __AVX__

#if __AVX__
    if (elempack == 8 && out_elempack == 4)
    {
        for (int j = 0; j < cols; j += 32)
            permute_transpose_pack8to4_bf16s_fp16s(ptr + (size_t)j * 8, stride, outptr + (j / 4) * outstride, outstride, rows, std::min(32, cols - j));
        return;
    }
#endif // __AVX__

#if __AVX__
    if (elempack == 8 && out_elempack == 8)
    {
        for (int j = 0; j < cols; j += 32)
            permute_transpose_pack8to8_bf16s_fp16s(ptr + (size_t)j * 8, stride, outptr + (j / 8) * outstride, outstride, rows, std::min(32, cols - j));
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 8 && out_elempack == 16)
    {
        for (int j = 0; j < cols; j += 32)
            permute_transpose_pack8to16_bf16s_fp16s(ptr + (size_t)j * 8, stride, outptr + (j / 16) * outstride, outstride, rows, std::min(32, cols - j));
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 1)
    {
        for (int j = 0; j < cols; j += 32)
            permute_transpose_pack16to1_bf16s_fp16s(ptr + (size_t)j * 16, stride, outptr + (j / 1) * outstride, outstride, rows, std::min(32, cols - j));
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 4)
    {
        for (int j = 0; j < cols; j += 32)
            permute_transpose_pack16to4_bf16s_fp16s(ptr + (size_t)j * 16, stride, outptr + (j / 4) * outstride, outstride, rows, std::min(32, cols - j));
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 8)
    {
        for (int j = 0; j < cols; j += 32)
            permute_transpose_pack16to8_bf16s_fp16s(ptr + (size_t)j * 16, stride, outptr + (j / 8) * outstride, outstride, rows, std::min(32, cols - j));
        return;
    }
#endif // __AVX512F__

#if __AVX512F__
    if (elempack == 16 && out_elempack == 16)
    {
        for (int j = 0; j < cols; j += 32)
            permute_transpose_pack16to16_bf16s_fp16s(ptr + (size_t)j * 16, stride, outptr + (j / 16) * outstride, outstride, rows, std::min(32, cols - j));
        return;
    }
#endif // __AVX512F__
}

// spatial transpose within one input channel group
// outcstep is used when unpacking
#if __SSE2__
static NCNN_FORCEINLINE void permute_spatial2x2_pack4_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    __m128i _a = _mm_loadu_si128((const __m128i*)ptr);
    __m128i _b = _mm_loadu_si128((const __m128i*)(ptr + stride));
    _mm_storeu_si128((__m128i*)outptr, _mm_unpacklo_epi64(_a, _b));
    _mm_storeu_si128((__m128i*)(outptr + outstride), _mm_unpackhi_epi64(_a, _b));
}

static void permute_spatial_pack4_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
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
                permute_spatial2x2_pack4_stride_bf16s_fp16s(ptr + i * stride + j * 4, stride, outptr + j * outstride + i * 4, outstride);
            }
            for (; j < xmax; j++)
            {
                __m128i _v0 = _mm_loadl_epi64((const __m128i*)(ptr + i * stride + j * 4));
                _mm_storel_epi64((__m128i*)(outptr + j * outstride + i * 4), _v0);
                __m128i _v1 = _mm_loadl_epi64((const __m128i*)(ptr + (i + 1) * stride + j * 4));
                _mm_storel_epi64((__m128i*)(outptr + j * outstride + (i + 1) * 4), _v1);
            }
        }
        for (; i < rows; i++)
        {
            for (int j = x; j < xmax; j++)
            {
                __m128i _v0 = _mm_loadl_epi64((const __m128i*)(ptr + i * stride + j * 4));
                _mm_storel_epi64((__m128i*)(outptr + j * outstride + i * 4), _v0);
            }
        }
    }
}

static void permute_spatial_pack4to1_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, size_t outcstep, int rows, int cols)
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
                permute_transpose16x4_stride_bf16s_fp16s(ptr + i * stride + j * 4, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i + 7 < rows; i += 8)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose8x4_stride_bf16s_fp16s(ptr + i * stride + j * 4, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i + 3 < rows; i += 4)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose4x4_stride_bf16s_fp16s(ptr + i * stride + j * 4, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i + 1 < rows; i += 2)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose2x4_stride_bf16s_fp16s(ptr + i * stride + j * 4, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i < rows; i++)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose1x4_stride_bf16s_fp16s(ptr + i * stride + j * 4, outptr + j * outstride + i, outcstep);
            }
        }
    }
#else
    for (int x = 0; x < cols; x++)
        permute_unpack4_stride_bf16s_fp16s(ptr + x * 4, stride, outptr + x * outstride, outcstep, rows);
#endif // __AVX512F__
}
#endif // __SSE2__

#if __AVX__
static NCNN_FORCEINLINE void permute_spatial2x2_pack8_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    __m256 _a = _mm256_loadu_ps((const float*)ptr);
    __m256 _b = _mm256_loadu_ps((const float*)(ptr + stride));
    _mm256_storeu_ps((float*)(outptr), _mm256_permute2f128_ps(_a, _b, 0x20));
    _mm256_storeu_ps((float*)(outptr + outstride), _mm256_permute2f128_ps(_a, _b, 0x31));
}

static void permute_spatial_pack8_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
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
                permute_spatial2x2_pack8_stride_bf16s_fp16s(ptr + i * stride + j * 8, stride, outptr + j * outstride + i * 8, outstride);
            }
            for (; j < xmax; j++)
            {
                __m128i _v0 = _mm_loadu_si128((const __m128i*)(ptr + i * stride + j * 8));
                _mm_storeu_si128((__m128i*)(outptr + j * outstride + i * 8), _v0);
                __m128i _v1 = _mm_loadu_si128((const __m128i*)(ptr + (i + 1) * stride + j * 8));
                _mm_storeu_si128((__m128i*)(outptr + j * outstride + (i + 1) * 8), _v1);
            }
        }
        for (; i < rows; i++)
        {
            for (int j = x; j < xmax; j++)
            {
                __m128i _v0 = _mm_loadu_si128((const __m128i*)(ptr + i * stride + j * 8));
                _mm_storeu_si128((__m128i*)(outptr + j * outstride + i * 8), _v0);
            }
        }
    }
}

static void permute_spatial_pack8to1_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, size_t outcstep, int rows, int cols)
{
#if __AVX512F__
    // limit the number of output streams for large channel planes
    if (rows >= 128 && stride >= 8 * 512 && outstride >= 512)
    {
        for (int x = 0; x < cols; x++)
            permute_unpack8_stride_bf16s_fp16s(ptr + x * 8, stride, outptr + x * outstride, outcstep, rows);
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
                permute_transpose16x8_stride_bf16s_fp16s(ptr + i * stride + j * 8, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i + 7 < rows; i += 8)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose8x8_stride_bf16s_fp16s(ptr + i * stride + j * 8, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i + 3 < rows; i += 4)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose4x8_stride_bf16s_fp16s(ptr + i * stride + j * 8, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i + 1 < rows; i += 2)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose2x8_stride_bf16s_fp16s(ptr + i * stride + j * 8, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i < rows; i++)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose1x8_stride_bf16s_fp16s(ptr + i * stride + j * 8, outptr + j * outstride + i, outcstep);
            }
        }
    }
#else
    for (int x = 0; x < cols; x++)
        permute_unpack8_stride_bf16s_fp16s(ptr + x * 8, stride, outptr + x * outstride, outcstep, rows);
#endif // __AVX512F__
}
#endif // __AVX__

#if __AVX512F__
static NCNN_FORCEINLINE void permute_spatial2x2_pack16_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    __m512 _a = _mm512_loadu_ps((const float*)ptr);
    __m512 _b = _mm512_loadu_ps((const float*)(ptr + stride));
    _mm512_storeu_ps((float*)(outptr), _mm512_shuffle_f32x4(_a, _b, 0x44));
    _mm512_storeu_ps((float*)(outptr + outstride), _mm512_shuffle_f32x4(_a, _b, 0xee));
}

static void permute_spatial_pack16_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
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
                permute_spatial2x2_pack16_stride_bf16s_fp16s(ptr + i * stride + j * 16, stride, outptr + j * outstride + i * 16, outstride);
            }
            for (; j < xmax; j++)
            {
                __m256 _v0 = _mm256_loadu_ps((const float*)(ptr + i * stride + j * 16));
                _mm256_storeu_ps((float*)(outptr + j * outstride + i * 16), _v0);
                __m256 _v1 = _mm256_loadu_ps((const float*)(ptr + (i + 1) * stride + j * 16));
                _mm256_storeu_ps((float*)(outptr + j * outstride + (i + 1) * 16), _v1);
            }
        }
        for (; i < rows; i++)
        {
            for (int j = x; j < xmax; j++)
            {
                __m256 _v0 = _mm256_loadu_ps((const float*)(ptr + i * stride + j * 16));
                _mm256_storeu_ps((float*)(outptr + j * outstride + i * 16), _v0);
            }
        }
    }
}

static void permute_spatial_pack16to1_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, size_t outcstep, int rows, int cols)
{
    // limit the number of output streams for large channel planes
    if (rows >= 128 && stride >= 16 * 512 && outstride >= 512)
    {
        for (int x = 0; x < cols; x++)
            permute_unpack16_stride_bf16s_fp16s(ptr + x * 16, stride, outptr + x * outstride, outcstep, rows);
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
                permute_transpose16x16_stride_bf16s_fp16s(ptr + i * stride + j * 16, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i + 7 < rows; i += 8)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose8x16_stride_bf16s_fp16s(ptr + i * stride + j * 16, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i + 3 < rows; i += 4)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose4x16_stride_bf16s_fp16s(ptr + i * stride + j * 16, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i + 1 < rows; i += 2)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose2x16_stride_bf16s_fp16s(ptr + i * stride + j * 16, stride, outptr + j * outstride + i, outcstep);
            }
        }
        for (; i < rows; i++)
        {
            for (int j = x; j < xmax; j++)
            {
                permute_transpose1x16_stride_bf16s_fp16s(ptr + i * stride + j * 16, outptr + j * outstride + i, outcstep);
            }
        }
    }
}
#endif // __AVX512F__

static void permute_transpose_spatial_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, size_t outcstep, int rows, int cols, int elempack, int out_elempack)
{
    if (elempack == out_elempack)
    {
        if (rows == 1 && outstride == (size_t)elempack)
        {
            memcpy(outptr, ptr, (size_t)cols * elempack * sizeof(unsigned short));
            return;
        }
        if (cols == 1 && stride == (size_t)elempack)
        {
            memcpy(outptr, ptr, (size_t)rows * elempack * sizeof(unsigned short));
            return;
        }
    }

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
        permute_spatial_pack4to1_bf16s_fp16s(ptr, stride, outptr, outstride, outcstep, rows, cols);
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
        permute_spatial_pack8to1_bf16s_fp16s(ptr, stride, outptr, outstride, outcstep, rows, cols);
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
        permute_spatial_pack16to1_bf16s_fp16s(ptr, stride, outptr, outstride, outcstep, rows, cols);
        return;
    }
#endif // __AVX512F__
}

#if __SSE2__
static void permute_transpose_blocks2_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    int j = 0;
    for (; j + 1 < cols; j += 2)
    {
        const unsigned short* p = ptr + (size_t)j * 2;
        unsigned short* out0 = outptr + j * outstride;
        unsigned short* out1 = out0 + outstride;
        int i = 0;
        for (; i + 1 < rows; i += 2)
        {
            __m128i _a = _mm_loadl_epi64((const __m128i*)p);
            __m128i _b = _mm_loadl_epi64((const __m128i*)(p + stride));
            __m128i _v = _mm_unpacklo_epi32(_a, _b);
            _mm_storel_epi64((__m128i*)out0, _v);
            _mm_storel_epi64((__m128i*)out1, _mm_srli_si128(_v, 8));
            p += stride * 2;
            out0 += 4;
            out1 += 4;
        }
        for (; i < rows; i++)
        {
            memcpy(out0, p, 2 * sizeof(unsigned short));
            memcpy(out1, p + 2, 2 * sizeof(unsigned short));
            p += stride;
            out0 += 2;
            out1 += 2;
        }
    }
    for (; j < cols; j++)
    {
        const unsigned short* p = ptr + (size_t)j * 2;
        unsigned short* out = outptr + j * outstride;
        for (int i = 0; i < rows; i++)
        {
            memcpy(out, p, 2 * sizeof(unsigned short));
            p += stride;
            out += 2;
        }
    }
}
#endif // __SSE2__

// transpose rows of contiguous blocks
// size is independent of elempack
// strides include padding; the block contents keep their original order
static void permute_transpose_blocks_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols, int size)
{
    if (rows == 1 && outstride == (size_t)size)
    {
        memcpy(outptr, ptr, (size_t)cols * size * sizeof(unsigned short));
        return;
    }
    if (cols == 1 && stride == (size_t)size)
    {
        memcpy(outptr, ptr, (size_t)rows * size * sizeof(unsigned short));
        return;
    }

    if (size == 1)
    {
        permute_transpose_pack1_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#if __SSE2__
    if (size == 2)
    {
        permute_transpose_blocks2_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
    if (size == 4)
    {
        permute_spatial_pack4_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __SSE2__
#if __AVX__
    if (size == 8)
    {
        permute_spatial_pack8_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX__
#if __AVX512F__
    if (size == 16)
    {
        permute_spatial_pack16_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__
#if __SSE2__
    if (size >= 2 && size < 4)
    {
        for (int j = 0; j < cols; j++)
        {
            const unsigned short* p = ptr + (size_t)j * size;
            unsigned short* out = outptr + j * outstride;
            for (int i = 0; i < rows; i++)
            {
                int a;
                int b;
                memcpy(&a, p, 4);
                memcpy(&b, p + size - 2, 4);
                memcpy(out, &a, 4);
                memcpy(out + size - 2, &b, 4);
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
            const unsigned short* p = ptr + (size_t)j * size;
            unsigned short* out = outptr + j * outstride;
            for (int i = 0; i < rows; i++)
            {
                __m128i _a = _mm_loadl_epi64((const __m128i*)p);
                __m128i _b = _mm_loadl_epi64((const __m128i*)(p + size - 4));
                _mm_storel_epi64((__m128i*)out, _a);
                _mm_storel_epi64((__m128i*)(out + size - 4), _b);
                p += stride;
                out += size;
            }
        }
        return;
    }
#endif // __SSE2__
#if __SSE2__
    if (size >= 8 && size < 16)
    {
        for (int j = 0; j < cols; j++)
        {
            const unsigned short* p = ptr + (size_t)j * size;
            unsigned short* out = outptr + j * outstride;
            for (int i = 0; i < rows; i++)
            {
                __m128i _a = _mm_loadu_si128((const __m128i*)p);
                __m128i _b = _mm_loadu_si128((const __m128i*)(p + size - 8));
                _mm_storeu_si128((__m128i*)out, _a);
                _mm_storeu_si128((__m128i*)(out + size - 8), _b);
                p += stride;
                out += size;
            }
        }
        return;
    }
#endif // __SSE2__
#if __AVX__
    if (size >= 16 && size < 32)
    {
        for (int j = 0; j < cols; j++)
        {
            const unsigned short* p = ptr + (size_t)j * size;
            unsigned short* out = outptr + j * outstride;
            for (int i = 0; i < rows; i++)
            {
                __m256i _a = _mm256_loadu_si256((const __m256i*)p);
                __m256i _b = _mm256_loadu_si256((const __m256i*)(p + size - 16));
                _mm256_storeu_si256((__m256i*)out, _a);
                _mm256_storeu_si256((__m256i*)(out + size - 16), _b);
                p += stride;
                out += size;
            }
        }
        return;
    }
#endif // __AVX__
#if __AVX512F__
    if (size >= 32 && size < 64)
    {
        for (int j = 0; j < cols; j++)
        {
            const unsigned short* p = ptr + (size_t)j * size;
            unsigned short* out = outptr + j * outstride;
            for (int i = 0; i < rows; i++)
            {
                __m512i _a = _mm512_loadu_si512((const void*)p);
                __m512i _b = _mm512_loadu_si512((const void*)(p + size - 32));
                _mm512_storeu_si512((void*)out, _a);
                _mm512_storeu_si512((void*)(out + size - 32), _b);
                p += stride;
                out += size;
            }
        }
        return;
    }
#endif // __AVX512F__
    for (int j = 0; j < cols; j++)
    {
        const unsigned short* p = ptr + (size_t)j * size;
        unsigned short* out = outptr + j * outstride;
        for (int i = 0; i < rows; i++)
        {
            memcpy(out, p, (size_t)size * sizeof(unsigned short));
            p += stride;
            out += size;
        }
    }
}

static void permute_unpack_spatial_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, size_t outcstep, int size, int elempack)
{
#if __SSE2__
    if (elempack == 4)
    {
        permute_unpack4_stride_bf16s_fp16s(ptr, 4, outptr, outcstep, size);
        return;
    }
#endif // __SSE2__

#if __AVX__
    if (elempack == 8)
    {
        permute_unpack8_stride_bf16s_fp16s(ptr, 8, outptr, outcstep, size);
        return;
    }
#endif // __AVX__

#if __AVX512F__
    if (elempack == 16)
    {
        permute_unpack16_stride_bf16s_fp16s(ptr, 16, outptr, outcstep, size);
        return;
    }
#endif // __AVX512F__
}

// exchange the input channel axis with one output channel group
// w is the remaining spatial extent; cstep includes input channel padding
#if __SSE2__
// the exchanged axis is contiguous; the stride variant has contiguous spatial input
static void permute3d_pack1to4_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * outcstep;
        for (int x = 0; x < w; x++)
        {
            __m128i _v = _mm_loadl_epi64((const __m128i*)p);
            _mm_storel_epi64((__m128i*)out, _v);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack1to4_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * outcstep;
        permute_pack4_stride_bf16s_fp16s(p, hstep, out, outwstep, w);
    }
}
#endif // __SSE2__

#if __AVX__
// the exchanged axis is contiguous; the stride variant has contiguous spatial input
static void permute3d_pack1to8_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * outcstep;
        for (int x = 0; x < w; x++)
        {
            __m128i _v = _mm_loadu_si128((const __m128i*)p);
            _mm_storeu_si128((__m128i*)out, _v);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack1to8_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * outcstep;
        permute_pack8_stride_bf16s_fp16s(p, hstep, out, outwstep, w);
    }
}
#endif // __AVX__

#if __AVX512F__
// the exchanged axis is contiguous; the stride variant has contiguous spatial input
static void permute3d_pack1to16_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * outcstep;
        for (int x = 0; x < w; x++)
        {
            __m256 _v = _mm256_loadu_ps((const float*)p);
            _mm256_storeu_ps((float*)out, _v);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack1to16_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * outcstep;
        permute_pack16_stride_bf16s_fp16s(p, hstep, out, outwstep, w);
    }
}
#endif // __AVX512F__

#if __SSE2__
// output channels are contiguous; the stride variant has contiguous spatial output
static void permute3d_pack4to1_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 4;
        for (int x = 0; x < w; x++)
        {
            __m128i _v = _mm_loadl_epi64((const __m128i*)p);
            _mm_storel_epi64((__m128i*)out, _v);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack4to1_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 4 * outcstep;
        permute_unpack4_stride_bf16s_fp16s(p, wstep, out, outcstep, w);
    }
}
#endif // __SSE2__

#if __SSE2__
static void permute3d_pack4to4_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 4 * 4;
        for (int x = 0; x < w; x++)
        {
            permute_transpose4x4_bf16s_fp16s(p, out);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack4to4_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 4 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose4x4_stride_bf16s_fp16s(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}
#endif // __SSE2__

#if __AVX__
static void permute3d_pack4to8_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 4 * 8;
        for (int x = 0; x < w; x++)
        {
            permute_transpose8x4_bf16s_fp16s(p, out);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack4to8_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 4 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose8x4_stride_bf16s_fp16s(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute3d_pack4to16_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 4 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose16x4_stride_bf16s_fp16s(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}
#endif // __AVX512F__

#if __AVX__
// output channels are contiguous; the stride variant has contiguous spatial output
static void permute3d_pack8to1_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 8;
        for (int x = 0; x < w; x++)
        {
            __m128i _v = _mm_loadu_si128((const __m128i*)p);
            _mm_storeu_si128((__m128i*)out, _v);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack8to1_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 8 * outcstep;
        permute_unpack8_stride_bf16s_fp16s(p, wstep, out, outcstep, w);
    }
}
#endif // __AVX__

#if __AVX__
static void permute3d_pack8to4_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 8 * 4;
        for (int x = 0; x < w; x++)
        {
            permute_transpose4x8_bf16s_fp16s(p, out);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack8to4_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 8 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose4x8_stride_bf16s_fp16s(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}
#endif // __AVX__

#if __AVX__
static void permute3d_pack8to8_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 8 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose8x8_stride_bf16s_fp16s(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute3d_pack8to16_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 8 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose16x8_stride_bf16s_fp16s(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
// output channels are contiguous; the stride variant has contiguous spatial output
static void permute3d_pack16to1_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 16;
        for (int x = 0; x < w; x++)
        {
            __m256 _v = _mm256_loadu_ps((const float*)p);
            _mm256_storeu_ps((float*)out, _v);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack16to1_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 16 * outcstep;
        permute_unpack16_stride_bf16s_fp16s(p, wstep, out, outcstep, w);
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute3d_pack16to4_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 16 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose4x16_stride_bf16s_fp16s(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute3d_pack16to8_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 16 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose8x16_stride_bf16s_fp16s(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute3d_pack16to16_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 16 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose16x16_stride_bf16s_fp16s(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}
#endif // __AVX512F__

// the exchanged input axis is contiguous
static void permute_pack_channels_bf16s_fp16s(const Mat& bottom_blob, const Mat& top_blob, const unsigned short* ptr, unsigned short* outptr, int size, int channels, int outchannels, size_t wstep, size_t outwstep, size_t outcstep)
{
    const size_t cstep = bottom_blob.cstep * bottom_blob.elempack;
    const size_t outstep = top_blob.cstep * top_blob.elempack;
    const int out_elempack = top_blob.elempack;

#if __SSE2__
    if (out_elempack == 4)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * 4;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack1to4_bf16s_fp16s(p, out, size, channels, wstep, cstep, outwstep, outcstep);
        }
        return;
    }
#endif // __SSE2__
#if __AVX__
    if (out_elempack == 8)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * 8;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack1to8_bf16s_fp16s(p, out, size, channels, wstep, cstep, outwstep, outcstep);
        }
        return;
    }
#endif // __AVX__
#if __AVX512F__
    if (out_elempack == 16)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * 16;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack1to16_bf16s_fp16s(p, out, size, channels, wstep, cstep, outwstep, outcstep);
        }
        return;
    }
#endif // __AVX512F__
}

// the remaining input spatial axis is contiguous
static void permute_pack_channels_stride_bf16s_fp16s(const Mat& bottom_blob, const Mat& top_blob, const unsigned short* ptr, unsigned short* outptr, int size, int channels, int outchannels, size_t hstep, size_t outwstep, size_t outcstep)
{
    const size_t cstep = bottom_blob.cstep * bottom_blob.elempack;
    const size_t outstep = top_blob.cstep * top_blob.elempack;
    const int out_elempack = top_blob.elempack;

#if __SSE2__
    if (out_elempack == 4)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep * 4;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack1to4_stride_bf16s_fp16s(p, out, size, channels, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#endif // __SSE2__
#if __AVX__
    if (out_elempack == 8)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep * 8;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack1to8_stride_bf16s_fp16s(p, out, size, channels, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#endif // __AVX__
#if __AVX512F__
    if (out_elempack == 16)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep * 16;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack1to16_stride_bf16s_fp16s(p, out, size, channels, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#endif // __AVX512F__
}

// the output channel axis is contiguous
static void permute_unpack_channels_bf16s_fp16s(const Mat& bottom_blob, const Mat& top_blob, const unsigned short* ptr, unsigned short* outptr, int size, int channels, int outchannels, size_t wstep, size_t hstep, size_t outwstep)
{
    const size_t cstep = bottom_blob.cstep * bottom_blob.elempack;
    const size_t outstep = top_blob.cstep * top_blob.elempack;
    const int elempack = bottom_blob.elempack;

#if __SSE2__
    if (elempack == 4)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack4to1_bf16s_fp16s(p, out, size, channels, wstep, cstep, outwstep);
        }
        return;
    }
#endif // __SSE2__
#if __AVX__
    if (elempack == 8)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack8to1_bf16s_fp16s(p, out, size, channels, wstep, cstep, outwstep);
        }
        return;
    }
#endif // __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack16to1_bf16s_fp16s(p, out, size, channels, wstep, cstep, outwstep);
        }
        return;
    }
#endif // __AVX512F__
}

// the remaining output spatial axis is contiguous
static void permute_unpack_channels_stride_bf16s_fp16s(const Mat& bottom_blob, const Mat& top_blob, const unsigned short* ptr, unsigned short* outptr, int size, int channels, int outchannels, size_t wstep, size_t hstep, size_t outcstep)
{
    const size_t cstep = bottom_blob.cstep * bottom_blob.elempack;
    const size_t outstep = top_blob.cstep * top_blob.elempack;
    const int elempack = bottom_blob.elempack;

#if __SSE2__
    if (elempack == 4)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack4to1_stride_bf16s_fp16s(p, out, size, channels, wstep, cstep, outcstep);
        }
        return;
    }
#endif // __SSE2__
#if __AVX__
    if (elempack == 8)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack8to1_stride_bf16s_fp16s(p, out, size, channels, wstep, cstep, outcstep);
        }
        return;
    }
#endif // __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack16to1_stride_bf16s_fp16s(p, out, size, channels, wstep, cstep, outcstep);
        }
        return;
    }
#endif // __AVX512F__
}

// both channel axes are packed
static void permute_channels_spatial_input_stride_bf16s_fp16s(const Mat& bottom_blob, const Mat& top_blob, const unsigned short* ptr, unsigned short* outptr, int size, int channels, int outchannels, size_t hstep, size_t outwstep, size_t outcstep)
{
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t wstep = elempack;

#if __SSE2__
    if (elempack == 4 && out_elempack == 4)
    {
        if (hstep == 4 && outcstep == 4)
        {
            for (int q = 0; q < outchannels; q++)
            {
                const unsigned short* p = ptr + q * hstep * 4;
                unsigned short* out = outptr + q * outstep;
                permute3d_pack4to4_bf16s_fp16s(p, out, size, channels, wstep, cstep, outwstep);
            }
        }
        else
        {
            for (int q = 0; q < outchannels; q++)
            {
                const unsigned short* p = ptr + q * hstep * 4;
                unsigned short* out = outptr + q * outstep;
                permute3d_pack4to4_stride_bf16s_fp16s(p, out, size, channels, wstep, hstep, cstep, outwstep, outcstep);
            }
        }
        return;
    }
#endif // __SSE2__
#if __AVX__
    if (elempack == 4 && out_elempack == 8)
    {
        if (hstep == 4 && outcstep == 8)
        {
            for (int q = 0; q < outchannels; q++)
            {
                const unsigned short* p = ptr + q * hstep * 8;
                unsigned short* out = outptr + q * outstep;
                permute3d_pack4to8_bf16s_fp16s(p, out, size, channels, wstep, cstep, outwstep);
            }
        }
        else
        {
            for (int q = 0; q < outchannels; q++)
            {
                const unsigned short* p = ptr + q * hstep * 8;
                unsigned short* out = outptr + q * outstep;
                permute3d_pack4to8_stride_bf16s_fp16s(p, out, size, channels, wstep, hstep, cstep, outwstep, outcstep);
            }
        }
        return;
    }
#endif // __AVX__
#if __AVX512F__
    if (elempack == 4 && out_elempack == 16)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep * 16;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack4to16_stride_bf16s_fp16s(p, out, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#endif // __AVX512F__
#if __AVX__
    if (elempack == 8 && out_elempack == 4)
    {
        if (hstep == 8 && outcstep == 4)
        {
            for (int q = 0; q < outchannels; q++)
            {
                const unsigned short* p = ptr + q * hstep * 4;
                unsigned short* out = outptr + q * outstep;
                permute3d_pack8to4_bf16s_fp16s(p, out, size, channels, wstep, cstep, outwstep);
            }
        }
        else
        {
            for (int q = 0; q < outchannels; q++)
            {
                const unsigned short* p = ptr + q * hstep * 4;
                unsigned short* out = outptr + q * outstep;
                permute3d_pack8to4_stride_bf16s_fp16s(p, out, size, channels, wstep, hstep, cstep, outwstep, outcstep);
            }
        }
        return;
    }
#endif // __AVX__
#if __AVX__
    if (elempack == 8 && out_elempack == 8)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep * 8;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack8to8_stride_bf16s_fp16s(p, out, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#endif // __AVX__
#if __AVX512F__
    if (elempack == 8 && out_elempack == 16)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep * 16;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack8to16_stride_bf16s_fp16s(p, out, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#endif // __AVX512F__
#if __AVX512F__
    if (elempack == 16 && out_elempack == 4)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep * 4;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack16to4_stride_bf16s_fp16s(p, out, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#endif // __AVX512F__
#if __AVX512F__
    if (elempack == 16 && out_elempack == 8)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep * 8;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack16to8_stride_bf16s_fp16s(p, out, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#endif // __AVX512F__
#if __AVX512F__
    if (elempack == 16 && out_elempack == 16)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep * 16;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack16to16_stride_bf16s_fp16s(p, out, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#endif // __AVX512F__
}

static void permute_channels_axis_input_stride_bf16s_fp16s(const Mat& bottom_blob, const Mat& top_blob, const unsigned short* ptr, unsigned short* outptr, int size, int channels, int outchannels, size_t wstep, size_t outwstep, size_t outcstep)
{
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t hstep = elempack;

#if __SSE2__
    if (elempack == 4 && out_elempack == 4)
    {
        if (hstep == 4 && outcstep == 4)
        {
            for (int q = 0; q < outchannels; q++)
            {
                const unsigned short* p = ptr + q * hstep * 4;
                unsigned short* out = outptr + q * outstep;
                permute3d_pack4to4_bf16s_fp16s(p, out, size, channels, wstep, cstep, outwstep);
            }
        }
        else
        {
            for (int q = 0; q < outchannels; q++)
            {
                const unsigned short* p = ptr + q * hstep * 4;
                unsigned short* out = outptr + q * outstep;
                permute3d_pack4to4_stride_bf16s_fp16s(p, out, size, channels, wstep, hstep, cstep, outwstep, outcstep);
            }
        }
        return;
    }
#endif // __SSE2__
#if __AVX__
    if (elempack == 4 && out_elempack == 8)
    {
        if (hstep == 4 && outcstep == 8)
        {
            for (int q = 0; q < outchannels; q++)
            {
                const unsigned short* p = ptr + q * hstep * 8;
                unsigned short* out = outptr + q * outstep;
                permute3d_pack4to8_bf16s_fp16s(p, out, size, channels, wstep, cstep, outwstep);
            }
        }
        else
        {
            for (int q = 0; q < outchannels; q++)
            {
                const unsigned short* p = ptr + q * hstep * 8;
                unsigned short* out = outptr + q * outstep;
                permute3d_pack4to8_stride_bf16s_fp16s(p, out, size, channels, wstep, hstep, cstep, outwstep, outcstep);
            }
        }
        return;
    }
#endif // __AVX__
#if __AVX512F__
    if (elempack == 4 && out_elempack == 16)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep * 16;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack4to16_stride_bf16s_fp16s(p, out, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#endif // __AVX512F__
#if __AVX__
    if (elempack == 8 && out_elempack == 4)
    {
        if (hstep == 8 && outcstep == 4)
        {
            for (int q = 0; q < outchannels; q++)
            {
                const unsigned short* p = ptr + q * hstep * 4;
                unsigned short* out = outptr + q * outstep;
                permute3d_pack8to4_bf16s_fp16s(p, out, size, channels, wstep, cstep, outwstep);
            }
        }
        else
        {
            for (int q = 0; q < outchannels; q++)
            {
                const unsigned short* p = ptr + q * hstep * 4;
                unsigned short* out = outptr + q * outstep;
                permute3d_pack8to4_stride_bf16s_fp16s(p, out, size, channels, wstep, hstep, cstep, outwstep, outcstep);
            }
        }
        return;
    }
#endif // __AVX__
#if __AVX__
    if (elempack == 8 && out_elempack == 8)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep * 8;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack8to8_stride_bf16s_fp16s(p, out, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#endif // __AVX__
#if __AVX512F__
    if (elempack == 8 && out_elempack == 16)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep * 16;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack8to16_stride_bf16s_fp16s(p, out, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#endif // __AVX512F__
#if __AVX512F__
    if (elempack == 16 && out_elempack == 4)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep * 4;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack16to4_stride_bf16s_fp16s(p, out, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#endif // __AVX512F__
#if __AVX512F__
    if (elempack == 16 && out_elempack == 8)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep * 8;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack16to8_stride_bf16s_fp16s(p, out, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#endif // __AVX512F__
#if __AVX512F__
    if (elempack == 16 && out_elempack == 16)
    {
        for (int q = 0; q < outchannels; q++)
        {
            const unsigned short* p = ptr + q * hstep * 16;
            unsigned short* out = outptr + q * outstep;
            permute3d_pack16to16_stride_bf16s_fp16s(p, out, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
#endif // __AVX512F__
}

static void permute_transpose_matrix_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols, int size, int num_threads)
{
    num_threads = permute_record_threads((size_t)rows * cols * size * sizeof(unsigned short), (size_t)size * sizeof(unsigned short), num_threads);

    if (num_threads == 1)
    {
        permute_transpose_blocks_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols, size);
        return;
    }

    // a few long contiguous blocks can be split without changing their order
    if ((size_t)rows * cols <= (size_t)num_threads && size * sizeof(unsigned short) >= 16384)
    {
        const int block = permute_block_size(size, sizeof(unsigned short), rows * cols, num_threads, 64 / sizeof(unsigned short));
        #pragma omp parallel for collapse(3) num_threads(num_threads)
        for (int j = 0; j < cols; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                for (int x = 0; x < size; x += block)
                {
                    const unsigned short* p = ptr + i * stride + (size_t)j * size + x;
                    unsigned short* out = outptr + j * outstride + (size_t)i * size + x;
                    memcpy(out, p, (size_t)std::min(block, size - x) * sizeof(unsigned short));
                }
            }
        }
        return;
    }

    const int col_block = permute_block_size(cols, (size_t)rows * size * sizeof(unsigned short), 1, num_threads, size == 1 ? 4 : permute_record_alignment(outstride, sizeof(unsigned short)));
    const int row_block = permute_block_size(rows, (size_t)col_block * size * sizeof(unsigned short), (cols + col_block - 1) / col_block, num_threads, size == 1 ? 4 : permute_record_alignment(size, sizeof(unsigned short)));
    #pragma omp parallel for collapse(2) num_threads(num_threads)
    for (int j = 0; j < cols; j += col_block)
    {
        for (int i = 0; i < rows; i += row_block)
        {
            permute_transpose_blocks_bf16s_fp16s(ptr + i * stride + (size_t)j * size, stride, outptr + j * outstride + (size_t)i * size, outstride, std::min(row_block, rows - i), std::min(col_block, cols - j), size);
        }
    }
}

// transpose independent matrices, splitting records only when the planes provide too few tasks
static void permute_transpose_matrices_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols, int size, int planes, size_t step, size_t outstep, int num_threads)
{
    if (planes == 1)
    {
        permute_transpose_matrix_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols, size, num_threads);
        return;
    }

    num_threads = permute_record_threads((size_t)planes * rows * cols * size * sizeof(unsigned short), (size_t)size * sizeof(unsigned short), num_threads);

    if (num_threads == 1)
    {
        for (int q = 0; q < planes; q++)
            permute_transpose_blocks_bf16s_fp16s(ptr + q * step, stride, outptr + q * outstep, outstride, rows, cols, size);
        return;
    }
    if (planes >= num_threads)
    {
        #pragma omp parallel for num_threads(num_threads)
        for (int q = 0; q < planes; q++)
            permute_transpose_blocks_bf16s_fp16s(ptr + q * step, stride, outptr + q * outstep, outstride, rows, cols, size);
        return;
    }

    if ((size_t)planes * rows * cols < (size_t)num_threads && size * sizeof(unsigned short) >= 16384)
    {
        const int block = permute_block_size(size, sizeof(unsigned short), planes * rows * cols, num_threads, 64 / sizeof(unsigned short));
        #pragma omp parallel for collapse(4) num_threads(num_threads)
        for (int q = 0; q < planes; q++)
        {
            for (int j = 0; j < cols; j++)
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int x = 0; x < size; x += block)
                    {
                        const unsigned short* p = ptr + q * step + i * stride + (size_t)j * size + x;
                        unsigned short* out = outptr + q * outstep + j * outstride + (size_t)i * size + x;
                        memcpy(out, p, (size_t)std::min(block, size - x) * sizeof(unsigned short));
                    }
                }
            }
        }
        return;
    }

    const int col_block = permute_block_size(cols, (size_t)rows * size * sizeof(unsigned short), planes, num_threads, size == 1 ? 32 : permute_record_alignment(outstride, sizeof(unsigned short)));
    const int row_block = permute_block_size(rows, (size_t)col_block * size * sizeof(unsigned short), planes * ((cols + col_block - 1) / col_block), num_threads, size == 1 ? 32 : permute_record_alignment(size, sizeof(unsigned short)));
    #pragma omp parallel for collapse(3) num_threads(num_threads)
    for (int q = 0; q < planes; q++)
    {
        for (int j = 0; j < cols; j += col_block)
        {
            for (int i = 0; i < rows; i += row_block)
            {
                const unsigned short* p = ptr + q * step + i * stride + (size_t)j * size;
                unsigned short* out = outptr + q * outstep + j * outstride + (size_t)i * size;
                permute_transpose_blocks_bf16s_fp16s(p, stride, out, outstride, std::min(row_block, rows - i), std::min(col_block, cols - j), size);
            }
        }
    }
}

// transpose spatial planes within each input channel group
static void permute_transpose_spatial_planes_stride_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int rows, int cols, size_t stride, size_t outstride, int planes, size_t step, size_t outstep, int num_threads)
{
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;

    if (num_threads == 1)
    {
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            unsigned short* outptr = top_blob.channel(q * elempack / out_elempack);
            for (int z = 0; z < planes; z++)
                permute_transpose_spatial_bf16s_fp16s(ptr + z * step, stride, outptr + z * outstep, outstride, top_blob.cstep, rows, cols, elempack, out_elempack);
        }
        return;
    }
    if (channels * planes >= num_threads)
    {
        #pragma omp parallel for collapse(2) num_threads(num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int z = 0; z < planes; z++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + z * step;
                unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack) + z * outstep;
                permute_transpose_spatial_bf16s_fp16s(ptr, stride, outptr, outstride, top_blob.cstep, rows, cols, elempack, out_elempack);
            }
        }
        return;
    }

    const int row_block = permute_block_size(rows, (size_t)cols * elemsize, channels * planes, num_threads, 32);
    const int col_block = permute_block_size(cols, (size_t)rows * elemsize, channels * planes * ((rows + row_block - 1) / row_block), num_threads, 32);
    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int q = 0; q < channels; q++)
    {
        for (int z = 0; z < planes; z++)
        {
            for (int i = 0; i < rows; i += row_block)
            {
                for (int j = 0; j < cols; j += col_block)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + z * step + i * stride + j * elempack;
                    unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack) + z * outstep + j * outstride + i * out_elempack;
                    permute_transpose_spatial_bf16s_fp16s(ptr, stride, outptr, outstride, top_blob.cstep, std::min(row_block, rows - i), std::min(col_block, cols - j), elempack, out_elempack);
                }
            }
        }
    }
}

// transpose h and w while preserving the other two slice axes
static void permute_transpose_hw_pack1_stride_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, size_t outstride, size_t outcstep, size_t outdstep, int num_threads)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const size_t stride = (size_t)w;

    if (num_threads == 1)
    {
        for (int q = 0; q < channels; q++)
        {
            for (int z = 0; z < d; z++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + z * (size_t)w * h;
                unsigned short* outptr = (unsigned short*)top_blob + q * outcstep + z * outdstep;
                permute_transpose_pack1_bf16s_fp16s(ptr, stride, outptr, outstride, h, w);
            }
        }
        return;
    }

    const int col_block = permute_block_size(w, (size_t)h * sizeof(unsigned short), channels * d, num_threads, 32);
    const int row_block = permute_block_size(h, (size_t)w * sizeof(unsigned short), channels * d * ((w + col_block - 1) / col_block), num_threads, 32);
    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int i = 0; i < h; i += row_block)
    {
        for (int j = 0; j < w; j += col_block)
        {
            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + z * (size_t)w * h + i * stride + j;
                    unsigned short* outptr = (unsigned short*)top_blob + q * outcstep + z * outdstep + j * outstride + i;
                    permute_transpose_pack1_bf16s_fp16s(ptr, stride, outptr, outstride, std::min(row_block, h - i), std::min(col_block, w - j));
                }
            }
        }
    }
}

// transpose d and w while preserving the other two slice axes
static void permute_transpose_dw_pack1_stride_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, size_t outstride, size_t outcstep, size_t outhstep, int num_threads)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const size_t stride = (size_t)w * h;

    if (num_threads == 1)
    {
        for (int q = 0; q < channels; q++)
        {
            for (int y = 0; y < h; y++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + y * (size_t)w;
                unsigned short* outptr = (unsigned short*)top_blob + q * outcstep + y * outhstep;
                permute_transpose_pack1_bf16s_fp16s(ptr, stride, outptr, outstride, d, w);
            }
        }
        return;
    }

    const int col_block = permute_block_size(w, (size_t)d * sizeof(unsigned short), channels * h, num_threads, 32);
    const int row_block = permute_block_size(d, (size_t)w * sizeof(unsigned short), channels * h * ((w + col_block - 1) / col_block), num_threads, 32);
    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int i = 0; i < d; i += row_block)
    {
        for (int j = 0; j < w; j += col_block)
        {
            for (int q = 0; q < channels; q++)
            {
                for (int y = 0; y < h; y++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + y * (size_t)w + i * stride + j;
                    unsigned short* outptr = (unsigned short*)top_blob + q * outcstep + y * outhstep + j * outstride + i;
                    permute_transpose_pack1_bf16s_fp16s(ptr, stride, outptr, outstride, std::min(row_block, d - i), std::min(col_block, w - j));
                }
            }
        }
    }
}

// transpose c and w while preserving the other two slice axes
static void permute_transpose_cw_pack1_stride_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, size_t outstride, size_t outdstep, size_t outhstep, int num_threads)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const size_t stride = bottom_blob.cstep;

    if (num_threads == 1)
    {
        for (int z = 0; z < d; z++)
        {
            for (int y = 0; y < h; y++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h + y * (size_t)w;
                unsigned short* outptr = (unsigned short*)top_blob + z * outdstep + y * outhstep;
                permute_transpose_pack1_bf16s_fp16s(ptr, stride, outptr, outstride, channels, w);
            }
        }
        return;
    }

    const int col_block = permute_block_size(w, (size_t)channels * sizeof(unsigned short), d * h, num_threads, 32);
    const int row_block = permute_block_size(channels, (size_t)w * sizeof(unsigned short), d * h * ((w + col_block - 1) / col_block), num_threads, 32);
    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int i = 0; i < channels; i += row_block)
    {
        for (int j = 0; j < w; j += col_block)
        {
            for (int z = 0; z < d; z++)
            {
                for (int y = 0; y < h; y++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h + y * (size_t)w + i * stride + j;
                    unsigned short* outptr = (unsigned short*)top_blob + z * outdstep + y * outhstep + j * outstride + i;
                    permute_transpose_pack1_bf16s_fp16s(ptr, stride, outptr, outstride, std::min(row_block, channels - i), std::min(col_block, w - j));
                }
            }
        }
    }
}

static void permute2d_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        permute_transpose_matrix_bf16s_fp16s(bottom_blob, w, top_blob, h, h, w, 1, num_threads);
        return;
    }

    if (num_threads == 1)
    {
        permute_transpose2d_bf16s_fp16s(bottom_blob, (size_t)w * elempack, top_blob, (size_t)top_blob.w * out_elempack, h * elempack, w, elempack, out_elempack);
        return;
    }

    const int row_block = permute_block_size(h, (size_t)w * elemsize, (w + 31) / 32, num_threads, 32);

    #pragma omp parallel for collapse(2) num_threads(num_threads)
    for (int i = 0; i < h; i += row_block)
    {
        for (int x = 0; x < w; x += 32)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)i * w * elempack + x * elempack;
            unsigned short* outptr = top_blob.row<unsigned short>(x / out_elempack) + (size_t)i * elempack * out_elempack;
            permute_transpose2d_bf16s_fp16s(ptr, (size_t)w * elempack, outptr, (size_t)top_blob.w * out_elempack, std::min(row_block, h - i) * elempack, std::min(32, w - x), elempack, out_elempack);
        }
    }
}

static void permute3d_hwc_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t stride = (size_t)w * elempack;
    const size_t outstride = (size_t)h * out_elempack;
    const size_t step = 0;
    const size_t outstep = 0;
    permute_transpose_spatial_planes_stride_bf16s_fp16s(bottom_blob, top_blob, h, w, stride, outstride, 1, step, outstep, num_threads);
}

static NCNN_FORCEINLINE void permute3d_wch_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels)
{
    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and h, keeping w as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
    else if (out_elempack == 1)
        permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, hstep, outcstep);
    else
        permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
}

static void permute3d_wch_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const size_t out_elemsize = top_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        num_threads = permute_record_threads(bottom_blob.total() * elemsize, (size_t)w * sizeof(unsigned short), num_threads);

        if (w < 64)
        {
            // short blocks include the w == 1 matrix-transpose case
            permute_transpose_matrix_bf16s_fp16s(bottom_blob, bottom_blob.cstep, top_blob, top_blob.cstep, channels, h, w, num_threads);
            return;
        }

        if (num_threads == 1)
        {
            const size_t cstep = bottom_blob.cstep;
            for (int q = 0; q < h; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)q * w;
                unsigned short* outptr = top_blob.channel(q);
                for (int c = 0; c < channels; c++)
                {
                    memcpy(outptr, ptr, (size_t)w * sizeof(unsigned short));
                    ptr += cstep;
                    outptr += w;
                }
            }
            return;
        }

        const int block = permute_block_size(w, sizeof(unsigned short), top_blob.c * top_blob.h, num_threads, 32);

        #pragma omp parallel for collapse(3) num_threads(num_threads)
        for (int i = 0; i < w; i += block)
        {
            for (int q = 0; q < top_blob.c; q++)
            {
                for (int y = 0; y < top_blob.h; y++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + q * (size_t)w + y * bottom_blob.cstep;
                    unsigned short* outptr = (unsigned short*)top_blob.channel(q) + (size_t)y * w;
                    memcpy(outptr + i, ptr + i, (size_t)std::min(block, w - i) * sizeof(unsigned short));
                }
            }
        }
        return;
    }

    if (num_threads == 1)
    {
        permute3d_wch_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, w, channels, top_blob.c);
        return;
    }

    const int out_channel_block = permute_block_size(top_blob.c, (size_t)w * channels * elempack * out_elemsize, 1, num_threads, 1);
    const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
    const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize * out_channel_block, out_channel_blocks, num_threads, 32);
    const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize * out_channel_block, out_channel_blocks * ((w + block - 1) / block), num_threads, 16);

    #pragma omp parallel for collapse(3) num_threads(num_threads)
    for (int i = 0; i < w; i += block)
    {
        for (int c = 0; c < channels; c += channel_block)
        {
            for (int q = 0; q < top_blob.c; q += out_channel_block)
            {
                permute3d_wch_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, w - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q));
            }
        }
    }
}

static NCNN_FORCEINLINE void permute3d_cwh_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels)
{
    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t outcstep = out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and h, keeping w as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
    else if (out_elempack == 1)
        permute_unpack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, hstep, outwstep);
    else
        permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
}

static void permute3d_cwh_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t out_elemsize = top_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        // w and h stay adjacent in the output; retain input channel padding
        if (top_blob.cstep == (size_t)w * channels && (size_t)w * h <= INT_MAX)
        {
            permute_transpose_matrix_bf16s_fp16s(bottom_blob, bottom_blob.cstep, top_blob, channels, channels, w * h, 1, num_threads);
            return;
        }

        const size_t stride = bottom_blob.cstep;
        const size_t outstride = (size_t)top_blob.w;
        const size_t step = (size_t)w;
        const size_t outstep = top_blob.cstep;
        permute_transpose_matrices_stride_bf16s_fp16s(bottom_blob, stride, top_blob, outstride, channels, w, 1, h, step, outstep, num_threads);
        return;
    }

    if (num_threads == 1)
    {
        permute3d_cwh_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, w, channels, top_blob.c);
        return;
    }

    const int out_channel_block = permute_block_size(top_blob.c, (size_t)w * channels * elempack * out_elemsize, 1, num_threads, 1);
    const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
    const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize * out_channel_block, out_channel_blocks, num_threads, 32);
    const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize * out_channel_block, out_channel_blocks * ((w + block - 1) / block), num_threads, 16);

    #pragma omp parallel for collapse(3) num_threads(num_threads)
    for (int i = 0; i < w; i += block)
    {
        for (int c = 0; c < channels; c += channel_block)
        {
            for (int q = 0; q < top_blob.c; q += out_channel_block)
            {
                permute3d_cwh_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, w - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q));
            }
        }
    }
}

static NCNN_FORCEINLINE void permute3d_hcw_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels)
{
    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = (size_t)w * elempack;
    const size_t hstep = elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and w, keeping h as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, outwstep, outcstep);
    else if (out_elempack == 1)
        permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, hstep, outcstep);
    else
        permute_channels_axis_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, outwstep, outcstep);
}

static void permute3d_hcw_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t out_elemsize = top_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        // c and h stay adjacent in the input; retain output channel padding
        if (bottom_blob.cstep == (size_t)w * h && (size_t)channels * h <= INT_MAX)
        {
            permute_transpose_matrix_bf16s_fp16s(bottom_blob, w, top_blob, top_blob.cstep, channels * h, w, 1, num_threads);
            return;
        }

        const size_t stride = (size_t)w;
        const size_t outstride = top_blob.cstep;
        const size_t step = bottom_blob.cstep;
        const size_t outstep = (size_t)top_blob.w;
        permute_transpose_matrices_stride_bf16s_fp16s(bottom_blob, stride, top_blob, outstride, h, w, 1, channels, step, outstep, num_threads);
        return;
    }

    if (num_threads == 1)
    {
        permute3d_hcw_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, h, channels, top_blob.c);
        return;
    }

    const int out_channel_block = permute_block_size(top_blob.c, (size_t)h * channels * elempack * out_elemsize, 1, num_threads, 1);
    const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
    const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize * out_channel_block, out_channel_blocks, num_threads, 32);
    const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize * out_channel_block, out_channel_blocks * ((h + block - 1) / block), num_threads, 16);

    #pragma omp parallel for collapse(3) num_threads(num_threads)
    for (int i = 0; i < h; i += block)
    {
        for (int c = 0; c < channels; c += channel_block)
        {
            for (int q = 0; q < top_blob.c; q += out_channel_block)
            {
                permute3d_hcw_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, h - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q));
            }
        }
    }
}

static NCNN_FORCEINLINE void permute3d_chw_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels)
{
    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = (size_t)w * elempack;
    const size_t hstep = elempack;
    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t outcstep = out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and w, keeping h as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, outwstep, outcstep);
    else if (out_elempack == 1)
        permute_unpack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, hstep, outwstep);
    else
        permute_channels_axis_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, outwstep, outcstep);
}

static void permute3d_chw_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t out_elemsize = top_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        // with w or h removed, the remaining operation is one c/spatial transpose
        if (w == 1 || h == 1)
        {
            permute_transpose_matrix_bf16s_fp16s(bottom_blob, bottom_blob.cstep, top_blob, w == 1 ? (size_t)channels : top_blob.cstep, channels, w * h, 1, num_threads);
            return;
        }

        if (channels == 1)
        {
            permute_transpose_matrix_bf16s_fp16s(bottom_blob, w, top_blob, top_blob.cstep, h, w, 1, num_threads);
            return;
        }

        const size_t stride = bottom_blob.cstep;
        const size_t outstride = top_blob.cstep;
        const size_t step = (size_t)w;
        const size_t outstep = (size_t)top_blob.w;
        permute_transpose_matrices_stride_bf16s_fp16s(bottom_blob, stride, top_blob, outstride, channels, w, 1, h, step, outstep, num_threads);
        return;
    }

    if (num_threads == 1)
    {
        permute3d_chw_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, h, channels, top_blob.c);
        return;
    }

    const int out_channel_block = permute_block_size(top_blob.c, (size_t)h * channels * elempack * out_elemsize, 1, num_threads, 1);
    const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
    const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize * out_channel_block, out_channel_blocks, num_threads, 32);
    const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize * out_channel_block, out_channel_blocks * ((h + block - 1) / block), num_threads, 16);

    #pragma omp parallel for collapse(3) num_threads(num_threads)
    for (int i = 0; i < h; i += block)
    {
        for (int c = 0; c < channels; c += channel_block)
        {
            for (int q = 0; q < top_blob.c; q += out_channel_block)
            {
                permute3d_chw_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, h - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q));
            }
        }
    }
}

static void permute4d_hwdc_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
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
    permute_transpose_spatial_planes_stride_bf16s_fp16s(bottom_blob, top_blob, h, w, stride, outstride, d, step, outstep, num_threads);
}

static void permute4d_wdhc_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;

    if (elempack == out_elempack)
    {
        num_threads = permute_record_threads(bottom_blob.total() * elemsize, (size_t)(w * elempack) * sizeof(unsigned short), num_threads);
        const size_t stride = (size_t)w * h * elempack;
        const size_t outstride = (size_t)w * d * elempack;
        const size_t step = bottom_blob.cstep * elempack;
        const size_t outstep = top_blob.cstep * elempack;
        permute_transpose_matrices_stride_bf16s_fp16s(bottom_blob, stride, top_blob, outstride, d, h, w * elempack, channels, step, outstep, num_threads);
        return;
    }

    if (num_threads == 1)
    {
        for (int q = 0; q < channels; q++)
        {
            for (int y = 0; y < h; y++)
            {
                for (int z = 0; z < d; z++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + ((size_t)z * h + y) * w * elempack;
                    unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack) + (((size_t)y * d + z) * w) * out_elempack;
                    permute_unpack_spatial_bf16s_fp16s(ptr, outptr, top_blob.cstep, w, elempack);
                }
            }
        }
        return;
    }

    const int block = permute_block_size(w, elemsize, channels * h * d, num_threads, 32);

    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int q = 0; q < channels; q++)
    {
        for (int y = 0; y < h; y++)
        {
            for (int z = 0; z < d; z++)
            {
                for (int i = 0; i < w; i += block)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + ((size_t)z * h + y) * w * elempack + i * elempack;
                    unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack) + (((size_t)y * d + z) * w + i) * out_elempack;
                    permute_unpack_spatial_bf16s_fp16s(ptr, outptr, top_blob.cstep, std::min(block, w - i), elempack);
                }
            }
        }
    }
}

static void permute4d_dwhc_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
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
    permute_transpose_spatial_planes_stride_bf16s_fp16s(bottom_blob, top_blob, d, w, stride, outstride, h, step, outstep, num_threads);
}

static void permute4d_hdwc_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
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
    permute_transpose_spatial_planes_stride_bf16s_fp16s(bottom_blob, top_blob, h, w, stride, outstride, d, step, outstep, num_threads);
}

static void permute4d_dhwc_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
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
    permute_transpose_spatial_planes_stride_bf16s_fp16s(bottom_blob, top_blob, d, w, stride, outstride, h, step, outstep, num_threads);
}

static void permute4d_whcd_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
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
    permute3d_wch_bf16s_fp16s(bottom_blob_3d, top_blob_3d, num_threads);
}

static NCNN_FORCEINLINE void permute4d_hwcd_unpack_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int x)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = (size_t)w * elempack;
    const size_t hstep = (size_t)w * h * elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t xstep = (size_t)elempack;
    const size_t outxstep = (size_t)top_blob.w * out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + x * xstep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + x * outxstep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and d, keeping h as the inner spatial axis
    permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, hstep, outcstep);
}

static NCNN_FORCEINLINE void permute4d_hwcd_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int y)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * h * elempack;
    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t outcstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t ystep = (size_t)w * elempack;
    const size_t outystep = (size_t)out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + y * ystep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + y * outystep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and d, keeping w as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
    else
        permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
}

static void permute4d_hwcd_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t out_elemsize = top_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w;
        const size_t outcstep = (size_t)top_blob.w * top_blob.h;
        const size_t outdstep = top_blob.cstep;
        permute_transpose_hw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outcstep, outdstep, num_threads);
        return;
    }

    if (out_elempack == 1)
    {
        if (num_threads == 1)
        {
            for (int x = 0; x < w; x++)
            {
                permute4d_hwcd_unpack_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, h, channels, top_blob.c, x);
            }
            return;
        }

        const int out_channel_block = permute_block_size(top_blob.c, (size_t)h * channels * elempack * out_elemsize, w, num_threads, 1);
        const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
        const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize * out_channel_block, w * out_channel_blocks, num_threads, 32);
        const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize * out_channel_block, w * out_channel_blocks * ((h + block - 1) / block), num_threads, 16);

        #pragma omp parallel for collapse(4) num_threads(num_threads)
        for (int i = 0; i < h; i += block)
        {
            for (int c = 0; c < channels; c += channel_block)
            {
                for (int x = 0; x < w; x++)
                {
                    for (int q = 0; q < top_blob.c; q += out_channel_block)
                    {
                        permute4d_hwcd_unpack_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, h - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), x);
                    }
                }
            }
        }
        return;
    }

    if (num_threads == 1)
    {
        for (int y = 0; y < h; y++)
        {
            permute4d_hwcd_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, w, channels, top_blob.c, y);
        }
        return;
    }

    const int out_channel_block = permute_block_size(top_blob.c, (size_t)w * channels * elempack * out_elemsize, h, num_threads, 1);
    const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
    const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize * out_channel_block, h * out_channel_blocks, num_threads, 32);
    const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize * out_channel_block, h * out_channel_blocks * ((w + block - 1) / block), num_threads, 16);

    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int i = 0; i < w; i += block)
    {
        for (int c = 0; c < channels; c += channel_block)
        {
            for (int y = 0; y < h; y++)
            {
                for (int q = 0; q < top_blob.c; q += out_channel_block)
                {
                    permute4d_hwcd_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, w - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), y);
                }
            }
        }
    }
}

static NCNN_FORCEINLINE void permute4d_wchd_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int y)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * h * elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t ystep = (size_t)w * elempack;
    const size_t outystep = (size_t)top_blob.w * top_blob.h * out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + y * ystep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + y * outystep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and d, keeping w as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
    else if (out_elempack == 1)
        permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, hstep, outcstep);
    else
        permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
}

static void permute4d_wchd_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const size_t out_elemsize = top_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        num_threads = permute_record_threads(bottom_blob.total() * elemsize, (size_t)w * sizeof(unsigned short), num_threads);
        const size_t stride = bottom_blob.cstep;
        const size_t outstride = (size_t)channels * w;
        const size_t step = (size_t)w * h;
        const size_t outstep = top_blob.cstep;
        permute_transpose_matrices_stride_bf16s_fp16s(bottom_blob, stride, top_blob, outstride, channels, h, w, d, step, outstep, num_threads);
        return;
    }

    if (num_threads == 1)
    {
        for (int y = 0; y < h; y++)
        {
            permute4d_wchd_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, w, channels, top_blob.c, y);
        }
        return;
    }

    const int out_channel_block = permute_block_size(top_blob.c, (size_t)w * channels * elempack * out_elemsize, h, num_threads, 1);
    const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
    const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize * out_channel_block, h * out_channel_blocks, num_threads, 32);
    const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize * out_channel_block, h * out_channel_blocks * ((w + block - 1) / block), num_threads, 16);

    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int i = 0; i < w; i += block)
    {
        for (int c = 0; c < channels; c += channel_block)
        {
            for (int y = 0; y < h; y++)
            {
                for (int q = 0; q < top_blob.c; q += out_channel_block)
                {
                    permute4d_wchd_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, w - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), y);
                }
            }
        }
    }
}

static void permute4d_cwhd_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
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
    permute3d_cwh_bf16s_fp16s(bottom_blob_3d, top_blob_3d, num_threads);
}

static NCNN_FORCEINLINE void permute4d_hcwd_unpack_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int x)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = (size_t)w * elempack;
    const size_t hstep = (size_t)w * h * elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t xstep = (size_t)elempack;
    const size_t outxstep = (size_t)top_blob.w * top_blob.h * out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + x * xstep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + x * outxstep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and d, keeping h as the inner spatial axis
    permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, hstep, outcstep);
}

static NCNN_FORCEINLINE void permute4d_hcwd_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int y)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * h * elempack;
    const size_t outwstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t ystep = (size_t)w * elempack;
    const size_t outystep = (size_t)out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + y * ystep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + y * outystep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and d, keeping w as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
    else
        permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
}

static void permute4d_hcwd_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t out_elemsize = top_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w * top_blob.h;
        const size_t outcstep = (size_t)top_blob.w;
        const size_t outdstep = top_blob.cstep;
        permute_transpose_hw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outcstep, outdstep, num_threads);
        return;
    }

    if (out_elempack == 1)
    {
        if (num_threads == 1)
        {
            for (int x = 0; x < w; x++)
            {
                permute4d_hcwd_unpack_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, h, channels, top_blob.c, x);
            }
            return;
        }

        const int out_channel_block = permute_block_size(top_blob.c, (size_t)h * channels * elempack * out_elemsize, w, num_threads, 1);
        const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
        const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize * out_channel_block, w * out_channel_blocks, num_threads, 32);
        const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize * out_channel_block, w * out_channel_blocks * ((h + block - 1) / block), num_threads, 16);

        #pragma omp parallel for collapse(4) num_threads(num_threads)
        for (int i = 0; i < h; i += block)
        {
            for (int c = 0; c < channels; c += channel_block)
            {
                for (int x = 0; x < w; x++)
                {
                    for (int q = 0; q < top_blob.c; q += out_channel_block)
                    {
                        permute4d_hcwd_unpack_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, h - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), x);
                    }
                }
            }
        }
        return;
    }

    if (num_threads == 1)
    {
        for (int y = 0; y < h; y++)
        {
            permute4d_hcwd_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, w, channels, top_blob.c, y);
        }
        return;
    }

    const int out_channel_block = permute_block_size(top_blob.c, (size_t)w * channels * elempack * out_elemsize, h, num_threads, 1);
    const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
    const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize * out_channel_block, h * out_channel_blocks, num_threads, 32);
    const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize * out_channel_block, h * out_channel_blocks * ((w + block - 1) / block), num_threads, 16);

    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int i = 0; i < w; i += block)
    {
        for (int c = 0; c < channels; c += channel_block)
        {
            for (int y = 0; y < h; y++)
            {
                for (int q = 0; q < top_blob.c; q += out_channel_block)
                {
                    permute4d_hcwd_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, w - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), y);
                }
            }
        }
    }
}

static NCNN_FORCEINLINE void permute4d_chwd_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int y)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * h * elempack;
    const size_t outwstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t outcstep = out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t ystep = (size_t)w * elempack;
    const size_t outystep = (size_t)top_blob.w * out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + y * ystep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + y * outystep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and d, keeping w as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
    else if (out_elempack == 1)
        permute_unpack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, hstep, outwstep);
    else
        permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
}

static void permute4d_chwd_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t out_elemsize = top_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w * top_blob.h;
        const size_t outdstep = top_blob.cstep;
        const size_t outhstep = (size_t)top_blob.w;
        permute_transpose_cw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outdstep, outhstep, num_threads);
        return;
    }

    if (num_threads == 1)
    {
        for (int y = 0; y < h; y++)
        {
            permute4d_chwd_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, w, channels, top_blob.c, y);
        }
        return;
    }

    const int out_channel_block = permute_block_size(top_blob.c, (size_t)w * channels * elempack * out_elemsize, h, num_threads, 1);
    const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
    const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize * out_channel_block, h * out_channel_blocks, num_threads, 32);
    const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize * out_channel_block, h * out_channel_blocks * ((w + block - 1) / block), num_threads, 16);

    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int i = 0; i < w; i += block)
    {
        for (int c = 0; c < channels; c += channel_block)
        {
            for (int y = 0; y < h; y++)
            {
                for (int q = 0; q < top_blob.c; q += out_channel_block)
                {
                    permute4d_chwd_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, w - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), y);
                }
            }
        }
    }
}

static NCNN_FORCEINLINE void permute4d_wdch_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int z)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)top_blob.w * out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + z * outzstep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and h, keeping w as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
    else if (out_elempack == 1)
        permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, hstep, outcstep);
    else
        permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
}

static void permute4d_wdch_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const size_t out_elemsize = top_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        num_threads = permute_record_threads(bottom_blob.total() * elemsize, (size_t)w * sizeof(unsigned short), num_threads);
        const size_t stride = (size_t)w * h;
        const size_t outstride = top_blob.cstep;
        const size_t step = bottom_blob.cstep;
        const size_t outstep = (size_t)d * w;
        permute_transpose_matrices_stride_bf16s_fp16s(bottom_blob, stride, top_blob, outstride, d, h, w, channels, step, outstep, num_threads);
        return;
    }

    if (num_threads == 1)
    {
        for (int z = 0; z < d; z++)
        {
            permute4d_wdch_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, w, channels, top_blob.c, z);
        }
        return;
    }

    const int out_channel_block = permute_block_size(top_blob.c, (size_t)w * channels * elempack * out_elemsize, d, num_threads, 1);
    const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
    const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize * out_channel_block, d * out_channel_blocks, num_threads, 32);
    const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize * out_channel_block, d * out_channel_blocks * ((w + block - 1) / block), num_threads, 16);

    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int i = 0; i < w; i += block)
    {
        for (int c = 0; c < channels; c += channel_block)
        {
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q += out_channel_block)
                {
                    permute4d_wdch_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, w - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), z);
                }
            }
        }
    }
}

static NCNN_FORCEINLINE void permute4d_dwch_unpack_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int x)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = (size_t)w * h * elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t xstep = (size_t)elempack;
    const size_t outxstep = (size_t)top_blob.w * out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + x * xstep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + x * outxstep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and h, keeping d as the inner spatial axis
    permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, hstep, outcstep);
}

static NCNN_FORCEINLINE void permute4d_dwch_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int z)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t outcstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + z * outzstep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and h, keeping w as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
    else
        permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
}

static void permute4d_dwch_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int w = bottom_blob.w;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t out_elemsize = top_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w;
        const size_t outcstep = (size_t)top_blob.w * top_blob.h;
        const size_t outhstep = top_blob.cstep;
        permute_transpose_dw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outcstep, outhstep, num_threads);
        return;
    }

    if (out_elempack == 1)
    {
        if (num_threads == 1)
        {
            for (int x = 0; x < w; x++)
            {
                permute4d_dwch_unpack_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, d, channels, top_blob.c, x);
            }
            return;
        }

        const int out_channel_block = permute_block_size(top_blob.c, (size_t)d * channels * elempack * out_elemsize, w, num_threads, 1);
        const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
        const int block = permute_block_size(d, (size_t)channels * elempack * out_elemsize * out_channel_block, w * out_channel_blocks, num_threads, 32);
        const int channel_block = permute_block_size(channels, (size_t)d * elempack * out_elemsize * out_channel_block, w * out_channel_blocks * ((d + block - 1) / block), num_threads, 16);

        #pragma omp parallel for collapse(4) num_threads(num_threads)
        for (int i = 0; i < d; i += block)
        {
            for (int c = 0; c < channels; c += channel_block)
            {
                for (int x = 0; x < w; x++)
                {
                    for (int q = 0; q < top_blob.c; q += out_channel_block)
                    {
                        permute4d_dwch_unpack_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, d - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), x);
                    }
                }
            }
        }
        return;
    }

    if (num_threads == 1)
    {
        for (int z = 0; z < d; z++)
        {
            permute4d_dwch_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, w, channels, top_blob.c, z);
        }
        return;
    }

    const int out_channel_block = permute_block_size(top_blob.c, (size_t)w * channels * elempack * out_elemsize, d, num_threads, 1);
    const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
    const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize * out_channel_block, d * out_channel_blocks, num_threads, 32);
    const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize * out_channel_block, d * out_channel_blocks * ((w + block - 1) / block), num_threads, 16);

    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int i = 0; i < w; i += block)
    {
        for (int c = 0; c < channels; c += channel_block)
        {
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q += out_channel_block)
                {
                    permute4d_dwch_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, w - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), z);
                }
            }
        }
    }
}

static NCNN_FORCEINLINE void permute4d_wcdh_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int z)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)top_blob.w * top_blob.h * out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + z * outzstep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and h, keeping w as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
    else if (out_elempack == 1)
        permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, hstep, outcstep);
    else
        permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
}

static void permute4d_wcdh_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const size_t out_elemsize = top_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        num_threads = permute_record_threads(bottom_blob.total() * elemsize, (size_t)w * sizeof(unsigned short), num_threads);
        const size_t stride = bottom_blob.cstep;
        const size_t outstride = top_blob.cstep;
        const size_t step = (size_t)w * h;
        const size_t outstep = (size_t)channels * w;
        permute_transpose_matrices_stride_bf16s_fp16s(bottom_blob, stride, top_blob, outstride, channels, h, w, d, step, outstep, num_threads);
        return;
    }

    if (num_threads == 1)
    {
        for (int z = 0; z < d; z++)
        {
            permute4d_wcdh_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, w, channels, top_blob.c, z);
        }
        return;
    }

    const int out_channel_block = permute_block_size(top_blob.c, (size_t)w * channels * elempack * out_elemsize, d, num_threads, 1);
    const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
    const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize * out_channel_block, d * out_channel_blocks, num_threads, 32);
    const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize * out_channel_block, d * out_channel_blocks * ((w + block - 1) / block), num_threads, 16);

    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int i = 0; i < w; i += block)
    {
        for (int c = 0; c < channels; c += channel_block)
        {
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q += out_channel_block)
                {
                    permute4d_wcdh_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, w - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), z);
                }
            }
        }
    }
}

static NCNN_FORCEINLINE void permute4d_cwdh_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int z)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t outcstep = out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)top_blob.w * top_blob.h * out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + z * outzstep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and h, keeping w as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
    else if (out_elempack == 1)
        permute_unpack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, hstep, outwstep);
    else
        permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
}

static void permute4d_cwdh_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int w = bottom_blob.w;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t out_elemsize = top_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w;
        const size_t outdstep = (size_t)top_blob.w * top_blob.h;
        const size_t outhstep = top_blob.cstep;
        permute_transpose_cw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outdstep, outhstep, num_threads);
        return;
    }

    if (num_threads == 1)
    {
        for (int z = 0; z < d; z++)
        {
            permute4d_cwdh_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, w, channels, top_blob.c, z);
        }
        return;
    }

    const int out_channel_block = permute_block_size(top_blob.c, (size_t)w * channels * elempack * out_elemsize, d, num_threads, 1);
    const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
    const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize * out_channel_block, d * out_channel_blocks, num_threads, 32);
    const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize * out_channel_block, d * out_channel_blocks * ((w + block - 1) / block), num_threads, 16);

    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int i = 0; i < w; i += block)
    {
        for (int c = 0; c < channels; c += channel_block)
        {
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q += out_channel_block)
                {
                    permute4d_cwdh_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, w - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), z);
                }
            }
        }
    }
}

static NCNN_FORCEINLINE void permute4d_dcwh_unpack_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int x)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = (size_t)w * h * elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t xstep = (size_t)elempack;
    const size_t outxstep = (size_t)top_blob.w * top_blob.h * out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + x * xstep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + x * outxstep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and h, keeping d as the inner spatial axis
    permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, hstep, outcstep);
}

static NCNN_FORCEINLINE void permute4d_dcwh_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int z)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + z * outzstep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and h, keeping w as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
    else
        permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
}

static void permute4d_dcwh_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int w = bottom_blob.w;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t out_elemsize = top_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w * top_blob.h;
        const size_t outcstep = (size_t)top_blob.w;
        const size_t outhstep = top_blob.cstep;
        permute_transpose_dw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outcstep, outhstep, num_threads);
        return;
    }

    if (out_elempack == 1)
    {
        if (num_threads == 1)
        {
            for (int x = 0; x < w; x++)
            {
                permute4d_dcwh_unpack_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, d, channels, top_blob.c, x);
            }
            return;
        }

        const int out_channel_block = permute_block_size(top_blob.c, (size_t)d * channels * elempack * out_elemsize, w, num_threads, 1);
        const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
        const int block = permute_block_size(d, (size_t)channels * elempack * out_elemsize * out_channel_block, w * out_channel_blocks, num_threads, 32);
        const int channel_block = permute_block_size(channels, (size_t)d * elempack * out_elemsize * out_channel_block, w * out_channel_blocks * ((d + block - 1) / block), num_threads, 16);

        #pragma omp parallel for collapse(4) num_threads(num_threads)
        for (int i = 0; i < d; i += block)
        {
            for (int c = 0; c < channels; c += channel_block)
            {
                for (int x = 0; x < w; x++)
                {
                    for (int q = 0; q < top_blob.c; q += out_channel_block)
                    {
                        permute4d_dcwh_unpack_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, d - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), x);
                    }
                }
            }
        }
        return;
    }

    if (num_threads == 1)
    {
        for (int z = 0; z < d; z++)
        {
            permute4d_dcwh_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, w, channels, top_blob.c, z);
        }
        return;
    }

    const int out_channel_block = permute_block_size(top_blob.c, (size_t)w * channels * elempack * out_elemsize, d, num_threads, 1);
    const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
    const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize * out_channel_block, d * out_channel_blocks, num_threads, 32);
    const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize * out_channel_block, d * out_channel_blocks * ((w + block - 1) / block), num_threads, 16);

    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int i = 0; i < w; i += block)
    {
        for (int c = 0; c < channels; c += channel_block)
        {
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q += out_channel_block)
                {
                    permute4d_dcwh_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, w - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), z);
                }
            }
        }
    }
}

static NCNN_FORCEINLINE void permute4d_cdwh_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int z)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t outcstep = out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)top_blob.w * out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + z * outzstep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and h, keeping w as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
    else if (out_elempack == 1)
        permute_unpack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, hstep, outwstep);
    else
        permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, hstep, outwstep, outcstep);
}

static void permute4d_cdwh_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int w = bottom_blob.w;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t out_elemsize = top_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w * top_blob.h;
        const size_t outdstep = (size_t)top_blob.w;
        const size_t outhstep = top_blob.cstep;
        permute_transpose_cw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outdstep, outhstep, num_threads);
        return;
    }

    if (num_threads == 1)
    {
        for (int z = 0; z < d; z++)
        {
            permute4d_cdwh_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, w, channels, top_blob.c, z);
        }
        return;
    }

    const int out_channel_block = permute_block_size(top_blob.c, (size_t)w * channels * elempack * out_elemsize, d, num_threads, 1);
    const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
    const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize * out_channel_block, d * out_channel_blocks, num_threads, 32);
    const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize * out_channel_block, d * out_channel_blocks * ((w + block - 1) / block), num_threads, 16);

    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int i = 0; i < w; i += block)
    {
        for (int c = 0; c < channels; c += channel_block)
        {
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q += out_channel_block)
                {
                    permute4d_cdwh_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, w - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), z);
                }
            }
        }
    }
}

static void permute4d_hdcw_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
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
    permute3d_hcw_bf16s_fp16s(bottom_blob_3d, top_blob_3d, num_threads);
}

static NCNN_FORCEINLINE void permute4d_dhcw_unpack_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int y)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = (size_t)w * h * elempack;
    const size_t hstep = elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t ystep = (size_t)w * elempack;
    const size_t outystep = (size_t)top_blob.w * out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + y * ystep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + y * outystep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and w, keeping d as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, outwstep, outcstep);
    else
        permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, hstep, outcstep);
}

static NCNN_FORCEINLINE void permute4d_dhcw_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int z)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = (size_t)w * elempack;
    const size_t hstep = elempack;
    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t outcstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + z * outzstep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and w, keeping h as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, outwstep, outcstep);
    else
        permute_channels_axis_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, outwstep, outcstep);
}

static void permute4d_dhcw_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t out_elemsize = top_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = top_blob.cstep;
        const size_t outcstep = (size_t)top_blob.w * top_blob.h;
        const size_t outhstep = (size_t)top_blob.w;
        permute_transpose_dw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outcstep, outhstep, num_threads);
        return;
    }

    if (out_elempack == 1)
    {
        if (num_threads == 1)
        {
            for (int y = 0; y < h; y++)
            {
                permute4d_dhcw_unpack_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, d, channels, top_blob.c, y);
            }
            return;
        }

        const int out_channel_block = permute_block_size(top_blob.c, (size_t)d * channels * elempack * out_elemsize, h, num_threads, 1);
        const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
        const int block = permute_block_size(d, (size_t)channels * elempack * out_elemsize * out_channel_block, h * out_channel_blocks, num_threads, 32);
        const int channel_block = permute_block_size(channels, (size_t)d * elempack * out_elemsize * out_channel_block, h * out_channel_blocks * ((d + block - 1) / block), num_threads, 16);

        #pragma omp parallel for collapse(4) num_threads(num_threads)
        for (int i = 0; i < d; i += block)
        {
            for (int c = 0; c < channels; c += channel_block)
            {
                for (int y = 0; y < h; y++)
                {
                    for (int q = 0; q < top_blob.c; q += out_channel_block)
                    {
                        permute4d_dhcw_unpack_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, d - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), y);
                    }
                }
            }
        }
        return;
    }

    if (num_threads == 1)
    {
        for (int z = 0; z < d; z++)
        {
            permute4d_dhcw_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, h, channels, top_blob.c, z);
        }
        return;
    }

    const int out_channel_block = permute_block_size(top_blob.c, (size_t)h * channels * elempack * out_elemsize, d, num_threads, 1);
    const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
    const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize * out_channel_block, d * out_channel_blocks, num_threads, 32);
    const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize * out_channel_block, d * out_channel_blocks * ((h + block - 1) / block), num_threads, 16);

    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int i = 0; i < h; i += block)
    {
        for (int c = 0; c < channels; c += channel_block)
        {
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q += out_channel_block)
                {
                    permute4d_dhcw_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, h - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), z);
                }
            }
        }
    }
}

static NCNN_FORCEINLINE void permute4d_hcdw_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int z)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = (size_t)w * elempack;
    const size_t hstep = elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)top_blob.w * top_blob.h * out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + z * outzstep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and w, keeping h as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, outwstep, outcstep);
    else if (out_elempack == 1)
        permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, hstep, outcstep);
    else
        permute_channels_axis_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, outwstep, outcstep);
}

static void permute4d_hcdw_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t out_elemsize = top_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = top_blob.cstep;
        const size_t outcstep = (size_t)top_blob.w;
        const size_t outdstep = (size_t)top_blob.w * top_blob.h;
        permute_transpose_hw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outcstep, outdstep, num_threads);
        return;
    }

    if (num_threads == 1)
    {
        for (int z = 0; z < d; z++)
        {
            permute4d_hcdw_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, h, channels, top_blob.c, z);
        }
        return;
    }

    const int out_channel_block = permute_block_size(top_blob.c, (size_t)h * channels * elempack * out_elemsize, d, num_threads, 1);
    const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
    const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize * out_channel_block, d * out_channel_blocks, num_threads, 32);
    const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize * out_channel_block, d * out_channel_blocks * ((h + block - 1) / block), num_threads, 16);

    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int i = 0; i < h; i += block)
    {
        for (int c = 0; c < channels; c += channel_block)
        {
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q += out_channel_block)
                {
                    permute4d_hcdw_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, h - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), z);
                }
            }
        }
    }
}

static void permute4d_chdw_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
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
    permute3d_chw_bf16s_fp16s(bottom_blob_3d, top_blob_3d, num_threads);
}

static NCNN_FORCEINLINE void permute4d_dchw_unpack_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int y)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = (size_t)w * h * elempack;
    const size_t hstep = elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t ystep = (size_t)w * elempack;
    const size_t outystep = (size_t)top_blob.w * top_blob.h * out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + y * ystep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + y * outystep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and w, keeping d as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, outwstep, outcstep);
    else
        permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, hstep, outcstep);
}

static NCNN_FORCEINLINE void permute4d_dchw_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int z)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = (size_t)w * elempack;
    const size_t hstep = elempack;
    const size_t outwstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + z * outzstep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and w, keeping h as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, outwstep, outcstep);
    else
        permute_channels_axis_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, outwstep, outcstep);
}

static void permute4d_dchw_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t out_elemsize = top_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = top_blob.cstep;
        const size_t outcstep = (size_t)top_blob.w;
        const size_t outhstep = (size_t)top_blob.w * top_blob.h;
        permute_transpose_dw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outcstep, outhstep, num_threads);
        return;
    }

    if (out_elempack == 1)
    {
        if (num_threads == 1)
        {
            for (int y = 0; y < h; y++)
            {
                permute4d_dchw_unpack_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, d, channels, top_blob.c, y);
            }
            return;
        }

        const int out_channel_block = permute_block_size(top_blob.c, (size_t)d * channels * elempack * out_elemsize, h, num_threads, 1);
        const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
        const int block = permute_block_size(d, (size_t)channels * elempack * out_elemsize * out_channel_block, h * out_channel_blocks, num_threads, 32);
        const int channel_block = permute_block_size(channels, (size_t)d * elempack * out_elemsize * out_channel_block, h * out_channel_blocks * ((d + block - 1) / block), num_threads, 16);

        #pragma omp parallel for collapse(4) num_threads(num_threads)
        for (int i = 0; i < d; i += block)
        {
            for (int c = 0; c < channels; c += channel_block)
            {
                for (int y = 0; y < h; y++)
                {
                    for (int q = 0; q < top_blob.c; q += out_channel_block)
                    {
                        permute4d_dchw_unpack_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, d - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), y);
                    }
                }
            }
        }
        return;
    }

    if (num_threads == 1)
    {
        for (int z = 0; z < d; z++)
        {
            permute4d_dchw_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, h, channels, top_blob.c, z);
        }
        return;
    }

    const int out_channel_block = permute_block_size(top_blob.c, (size_t)h * channels * elempack * out_elemsize, d, num_threads, 1);
    const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
    const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize * out_channel_block, d * out_channel_blocks, num_threads, 32);
    const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize * out_channel_block, d * out_channel_blocks * ((h + block - 1) / block), num_threads, 16);

    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int i = 0; i < h; i += block)
    {
        for (int c = 0; c < channels; c += channel_block)
        {
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q += out_channel_block)
                {
                    permute4d_dchw_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, h - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), z);
                }
            }
        }
    }
}

static NCNN_FORCEINLINE void permute4d_cdhw_block_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int i, int c, int q, int size, int inchannels, int outchannels, int z)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t wstep = (size_t)w * elempack;
    const size_t hstep = elempack;
    const size_t outwstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t outcstep = out_elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t outstep = top_blob.cstep * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)top_blob.w * out_elempack;

    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + q * out_elempack * hstep + c * cstep + i * wstep;
    unsigned short* outptr = (unsigned short*)top_blob + z * outzstep + q * outstep + c * elempack * outcstep + i * outwstep;
    // exchange c and w, keeping h as the inner spatial axis
    if (elempack == 1)
        permute_pack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, outwstep, outcstep);
    else if (out_elempack == 1)
        permute_unpack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, hstep, outwstep);
    else
        permute_channels_axis_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, size, inchannels, outchannels, wstep, outwstep, outcstep);
}

static void permute4d_cdhw_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int num_threads)
{
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t out_elemsize = top_blob.elemsize;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = top_blob.cstep;
        const size_t outdstep = (size_t)top_blob.w;
        const size_t outhstep = (size_t)top_blob.w * top_blob.h;
        permute_transpose_cw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outdstep, outhstep, num_threads);
        return;
    }

    if (num_threads == 1)
    {
        for (int z = 0; z < d; z++)
        {
            permute4d_cdhw_block_bf16s_fp16s(bottom_blob, top_blob, 0, 0, 0, h, channels, top_blob.c, z);
        }
        return;
    }

    const int out_channel_block = permute_block_size(top_blob.c, (size_t)h * channels * elempack * out_elemsize, d, num_threads, 1);
    const int out_channel_blocks = (top_blob.c + out_channel_block - 1) / out_channel_block;
    const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize * out_channel_block, d * out_channel_blocks, num_threads, 32);
    const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize * out_channel_block, d * out_channel_blocks * ((h + block - 1) / block), num_threads, 16);

    #pragma omp parallel for collapse(4) num_threads(num_threads)
    for (int i = 0; i < h; i += block)
    {
        for (int c = 0; c < channels; c += channel_block)
        {
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q += out_channel_block)
                {
                    permute4d_cdhw_block_bf16s_fp16s(bottom_blob, top_blob, i, c, q, std::min(block, h - i), std::min(channel_block, channels - c), std::min(out_channel_block, top_blob.c - q), z);
                }
            }
        }
    }
}

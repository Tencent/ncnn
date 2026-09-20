// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Full register tiles have no size or packing branches. Bounds checks are
// confined to permute_transpose_tail_bf16s_fp16s.

#if __SSE2__
static NCNN_FORCEINLINE void permute_transpose4x4_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
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
static NCNN_FORCEINLINE void permute_transpose4x8_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
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
#endif // __AVX__

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose4x16_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    permute_transpose4x8_bf16s_fp16s(ptr, stride, outptr, outstride);
    permute_transpose4x8_bf16s_fp16s(ptr + 8, stride, outptr + 8 * outstride, outstride);
}
#endif // __AVX512F__

#if __AVX__
static NCNN_FORCEINLINE void permute_transpose8x4_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
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
#endif // __AVX__

#if __SSE2__
static NCNN_FORCEINLINE void permute_transpose8x8_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
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
static NCNN_FORCEINLINE void permute_transpose8x16_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    permute_transpose8x8_bf16s_fp16s(ptr, stride, outptr, outstride);
    permute_transpose8x8_bf16s_fp16s(ptr + 8, stride, outptr + 8 * outstride, outstride);
}
#endif // __AVX512F__

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose16x4_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    permute_transpose8x4_bf16s_fp16s(ptr, stride, outptr, outstride);
    permute_transpose8x4_bf16s_fp16s(ptr + 8 * stride, stride, outptr + 8, outstride);
}
#endif // __AVX512F__

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose16x8_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    permute_transpose8x8_bf16s_fp16s(ptr, stride, outptr, outstride);
    permute_transpose8x8_bf16s_fp16s(ptr + 8 * stride, stride, outptr + 8, outstride);
}
#endif // __AVX512F__

#if __AVX512F__
static NCNN_FORCEINLINE void permute_transpose16x16_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
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

static void permute_transpose_tail_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int i = 0; i < rows; i += 8)
    {
        const int nr = std::min(8, rows - i);
        for (int j = 0; j < cols; j += 8)
        {
            const int nc = std::min(8, cols - j);
#if __SSE2__
            __m128i _r0 = nr > 0 ? permute_load(ptr + (i)*stride + j, nc * 2) : _mm_setzero_si128();
            __m128i _r1 = nr > 1 ? permute_load(ptr + (i + 1) * stride + j, nc * 2) : _mm_setzero_si128();
            __m128i _r2 = nr > 2 ? permute_load(ptr + (i + 2) * stride + j, nc * 2) : _mm_setzero_si128();
            __m128i _r3 = nr > 3 ? permute_load(ptr + (i + 3) * stride + j, nc * 2) : _mm_setzero_si128();
            __m128i _r4 = nr > 4 ? permute_load(ptr + (i + 4) * stride + j, nc * 2) : _mm_setzero_si128();
            __m128i _r5 = nr > 5 ? permute_load(ptr + (i + 5) * stride + j, nc * 2) : _mm_setzero_si128();
            __m128i _r6 = nr > 6 ? permute_load(ptr + (i + 6) * stride + j, nc * 2) : _mm_setzero_si128();
            __m128i _r7 = nr > 7 ? permute_load(ptr + (i + 7) * stride + j, nc * 2) : _mm_setzero_si128();
            transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            if (nc > 0) permute_store(outptr + (j)*outstride + i, _r0, nr * 2);
            if (nc > 1) permute_store(outptr + (j + 1) * outstride + i, _r1, nr * 2);
            if (nc > 2) permute_store(outptr + (j + 2) * outstride + i, _r2, nr * 2);
            if (nc > 3) permute_store(outptr + (j + 3) * outstride + i, _r3, nr * 2);
            if (nc > 4) permute_store(outptr + (j + 4) * outstride + i, _r4, nr * 2);
            if (nc > 5) permute_store(outptr + (j + 5) * outstride + i, _r5, nr * 2);
            if (nc > 6) permute_store(outptr + (j + 6) * outstride + i, _r6, nr * 2);
            if (nc > 7) permute_store(outptr + (j + 7) * outstride + i, _r7, nr * 2);
#else
            for (int y = 0; y < nr; y++)
                for (int x = 0; x < nc; x++)
                    memcpy(outptr + (j + x) * outstride + (i + y), ptr + (i + y) * stride + (j + x), 2);
#endif // __SSE2__
        }
    }
}

// Unpacked matrix transpose, shared by 2d and channel/spatial permutations.
static void permute_transpose_pack1_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    int i = 0;
#if __AVX512F__
    if (cols >= 16)
    {
        for (; i + 15 < rows; i += 16)
        {
            int j = 0;
            for (; j + 15 < cols; j += 16)
            {
                permute_transpose16x16_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
            }
            if (j < cols)
                permute_transpose_tail_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride, 16, cols - j);
        }
    }
#endif // __AVX512F__
#if __SSE2__
    if (cols >= 8)
    {
        for (; i + 7 < rows; i += 8)
        {
            int j = 0;
            for (; j + 7 < cols; j += 8)
            {
                permute_transpose8x8_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
            }
            if (j < cols)
                permute_transpose_tail_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride, 8, cols - j);
        }
    }
#endif // __SSE2__
#if __SSE2__
    if (cols >= 4)
    {
        for (; i + 3 < rows; i += 4)
        {
            int j = 0;
            for (; j + 3 < cols; j += 4)
            {
                permute_transpose4x4_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
            }
            if (j < cols)
                permute_transpose_tail_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride, 4, cols - j);
        }
    }
#endif // __SSE2__
    if (i < rows)
        permute_transpose_tail_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride, rows - i, cols);
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
            permute_transpose4x4_bf16s_fp16s(p, 4, out, 4);
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
            permute_transpose8x4_bf16s_fp16s(p, 4, out, 8);
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
            permute_transpose16x4_bf16s_fp16s(p, 4, out, 16);
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
            permute_transpose4x8_bf16s_fp16s(p, 8, out, 4);
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
            permute_transpose8x8_bf16s_fp16s(p, 8, out, 8);
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
            permute_transpose16x8_bf16s_fp16s(p, 8, out, 16);
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
            permute_transpose4x16_bf16s_fp16s(p, 16, out, 4);
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
            permute_transpose8x16_bf16s_fp16s(p, 16, out, 8);
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
            permute_transpose16x16_bf16s_fp16s(p, 16, out, 16);
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
    for (int q = 0; q < h / 4; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned short* p = ptr + c * cstep + q * 4 * hstep;
            unsigned short* out = outptr + q * outhstep + c * 4 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose4x4_bf16s_fp16s(p, hstep, out, outcstep);
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
    for (int q = 0; q < h / 8; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned short* p = ptr + c * cstep + q * 8 * hstep;
            unsigned short* out = outptr + q * outhstep + c * 4 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose8x4_bf16s_fp16s(p, hstep, out, outcstep);
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
                permute_transpose16x4_bf16s_fp16s(p, hstep, out, outcstep);
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
    for (int q = 0; q < h / 4; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned short* p = ptr + c * cstep + q * 4 * hstep;
            unsigned short* out = outptr + q * outhstep + c * 8 * outcstep;
            for (int x = 0; x < w; x++)
            {
                permute_transpose4x8_bf16s_fp16s(p, hstep, out, outcstep);
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
                permute_transpose8x8_bf16s_fp16s(p, hstep, out, outcstep);
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
                permute_transpose16x8_bf16s_fp16s(p, hstep, out, outcstep);
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
                permute_transpose4x16_bf16s_fp16s(p, hstep, out, outcstep);
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
                permute_transpose8x16_bf16s_fp16s(p, hstep, out, outcstep);
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
                permute_transpose16x16_bf16s_fp16s(p, hstep, out, outcstep);
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

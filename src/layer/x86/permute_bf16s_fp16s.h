// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Full register tiles have no size or packing branches. Bounds checks are
// confined to permute_transpose_tail_bf16s_fp16s.

static NCNN_FORCEINLINE void permute_transpose4x4_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride)
{
#if __SSE2__
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
#else
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            memcpy(outptr + j * outstride + i * 2, ptr + i * stride + j * 2, 2);
#endif // __SSE2__
}

static NCNN_FORCEINLINE void permute_transpose4x8_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride)
{
#if __SSE2__
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
#else
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++)
            memcpy(outptr + j * outstride + i * 2, ptr + i * stride + j * 2, 2);
#endif // __SSE2__
}

static NCNN_FORCEINLINE void permute_transpose4x16_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride)
{
#if __SSE2__
    permute_transpose4x8_bf16s_fp16s(ptr, stride, outptr, outstride);
    permute_transpose4x8_bf16s_fp16s(ptr + 16, stride, outptr + 8 * outstride, outstride);
#else
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 16; j++)
            memcpy(outptr + j * outstride + i * 2, ptr + i * stride + j * 2, 2);
#endif // __SSE2__
}

static NCNN_FORCEINLINE void permute_transpose8x4_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride)
{
#if __SSE2__
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
#else
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            memcpy(outptr + j * outstride + i * 2, ptr + i * stride + j * 2, 2);
#endif // __SSE2__
}

static NCNN_FORCEINLINE void permute_transpose8x8_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride)
{
#if __SSE2__
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
#else
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            memcpy(outptr + j * outstride + i * 2, ptr + i * stride + j * 2, 2);
#endif // __SSE2__
}

static NCNN_FORCEINLINE void permute_transpose8x16_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride)
{
#if __SSE2__
    permute_transpose8x8_bf16s_fp16s(ptr, stride, outptr, outstride);
    permute_transpose8x8_bf16s_fp16s(ptr + 16, stride, outptr + 8 * outstride, outstride);
#else
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 16; j++)
            memcpy(outptr + j * outstride + i * 2, ptr + i * stride + j * 2, 2);
#endif // __SSE2__
}

static NCNN_FORCEINLINE void permute_transpose16x4_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride)
{
#if __SSE2__
    permute_transpose8x4_bf16s_fp16s(ptr, stride, outptr, outstride);
    permute_transpose8x4_bf16s_fp16s(ptr + 8 * stride, stride, outptr + 16, outstride);
#else
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 4; j++)
            memcpy(outptr + j * outstride + i * 2, ptr + i * stride + j * 2, 2);
#endif // __SSE2__
}

static NCNN_FORCEINLINE void permute_transpose16x8_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride)
{
#if __SSE2__
    permute_transpose8x8_bf16s_fp16s(ptr, stride, outptr, outstride);
    permute_transpose8x8_bf16s_fp16s(ptr + 8 * stride, stride, outptr + 16, outstride);
#else
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 8; j++)
            memcpy(outptr + j * outstride + i * 2, ptr + i * stride + j * 2, 2);
#endif // __SSE2__
}

static NCNN_FORCEINLINE void permute_transpose16x16_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride)
{
#if __AVX512F__
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
#else
#if __SSE2__
    permute_transpose8x8_bf16s_fp16s(ptr, stride, outptr, outstride);
    permute_transpose8x8_bf16s_fp16s(ptr + 16, stride, outptr + 8 * outstride, outstride);
    permute_transpose8x8_bf16s_fp16s(ptr + 8 * stride, stride, outptr + 16, outstride);
    permute_transpose8x8_bf16s_fp16s(ptr + 8 * stride + 16, stride, outptr + 8 * outstride + 16, outstride);
#else
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
            memcpy(outptr + j * outstride + i * 2, ptr + i * stride + j * 2, 2);
#endif // __SSE2__
#endif // __AVX512F__
}

static void permute_transpose_tail_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int i = 0; i < rows; i += 8)
    {
        const int nr = std::min(8, rows - i);
        for (int j = 0; j < cols; j += 8)
        {
            const int nc = std::min(8, cols - j);
#if __SSE2__
            __m128i _r0 = nr > 0 ? permute_load(ptr + (i) * stride + j * 2, nc * 2) : _mm_setzero_si128();
            __m128i _r1 = nr > 1 ? permute_load(ptr + (i + 1) * stride + j * 2, nc * 2) : _mm_setzero_si128();
            __m128i _r2 = nr > 2 ? permute_load(ptr + (i + 2) * stride + j * 2, nc * 2) : _mm_setzero_si128();
            __m128i _r3 = nr > 3 ? permute_load(ptr + (i + 3) * stride + j * 2, nc * 2) : _mm_setzero_si128();
            __m128i _r4 = nr > 4 ? permute_load(ptr + (i + 4) * stride + j * 2, nc * 2) : _mm_setzero_si128();
            __m128i _r5 = nr > 5 ? permute_load(ptr + (i + 5) * stride + j * 2, nc * 2) : _mm_setzero_si128();
            __m128i _r6 = nr > 6 ? permute_load(ptr + (i + 6) * stride + j * 2, nc * 2) : _mm_setzero_si128();
            __m128i _r7 = nr > 7 ? permute_load(ptr + (i + 7) * stride + j * 2, nc * 2) : _mm_setzero_si128();
            transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            if (nc > 0) permute_store(outptr + (j) * outstride + i * 2, _r0, nr * 2);
            if (nc > 1) permute_store(outptr + (j + 1) * outstride + i * 2, _r1, nr * 2);
            if (nc > 2) permute_store(outptr + (j + 2) * outstride + i * 2, _r2, nr * 2);
            if (nc > 3) permute_store(outptr + (j + 3) * outstride + i * 2, _r3, nr * 2);
            if (nc > 4) permute_store(outptr + (j + 4) * outstride + i * 2, _r4, nr * 2);
            if (nc > 5) permute_store(outptr + (j + 5) * outstride + i * 2, _r5, nr * 2);
            if (nc > 6) permute_store(outptr + (j + 6) * outstride + i * 2, _r6, nr * 2);
            if (nc > 7) permute_store(outptr + (j + 7) * outstride + i * 2, _r7, nr * 2);
#else
            for (int y = 0; y < nr; y++)
                for (int x = 0; x < nc; x++)
                    memcpy(outptr + (j + x) * outstride + (i + y) * 2, ptr + (i + y) * stride + (j + x) * 2, 2);
#endif // __SSE2__
        }
    }
}

// Unpacked matrix transpose, shared by 2d and channel/spatial permutations.
static void permute_transpose_pack1_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
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
                permute_transpose16x16_bf16s_fp16s(ptr + i * stride + j * 2, stride, outptr + j * outstride + i * 2, outstride);
            }
            if (j < cols)
                permute_transpose_tail_bf16s_fp16s(ptr + i * stride + j * 2, stride, outptr + j * outstride + i * 2, outstride, 16, cols - j);
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
                permute_transpose8x8_bf16s_fp16s(ptr + i * stride + j * 2, stride, outptr + j * outstride + i * 2, outstride);
            }
            if (j < cols)
                permute_transpose_tail_bf16s_fp16s(ptr + i * stride + j * 2, stride, outptr + j * outstride + i * 2, outstride, 8, cols - j);
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
                permute_transpose4x4_bf16s_fp16s(ptr + i * stride + j * 2, stride, outptr + j * outstride + i * 2, outstride);
            }
            if (j < cols)
                permute_transpose_tail_bf16s_fp16s(ptr + i * stride + j * 2, stride, outptr + j * outstride + i * 2, outstride, 4, cols - j);
        }
    }
#endif // __SSE2__
    if (i < rows)
        permute_transpose_tail_bf16s_fp16s(ptr + i * stride, stride, outptr + i * 2, outstride, rows - i, cols);
}

#if __SSE2__
static void permute_transpose_pack1to4_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int x = 0; x < cols; x += 4)
    {
        const unsigned char* p = ptr + x * 2;
        unsigned char* out = outptr + (x / 4) * outstride;
        for (int y = 0; y < rows; y++)
        {
            __m128i _r0 = _mm_loadl_epi64((const __m128i*)(p));
            _mm_storel_epi64((__m128i*)(out), _r0);
            p += stride;
            out += 8;
        }
    }
}
#endif // __SSE2__

#if __AVX__
static void permute_transpose_pack1to8_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int x = 0; x < cols; x += 8)
    {
        const unsigned char* p = ptr + x * 2;
        unsigned char* out = outptr + (x / 8) * outstride;
        for (int y = 0; y < rows; y++)
        {
            __m128i _r0 = _mm_loadu_si128((const __m128i*)(p));
            _mm_storeu_si128((__m128i*)(out), _r0);
            p += stride;
            out += 16;
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute_transpose_pack1to16_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int x = 0; x < cols; x += 16)
    {
        const unsigned char* p = ptr + x * 2;
        unsigned char* out = outptr + (x / 16) * outstride;
        for (int y = 0; y < rows; y++)
        {
            __m256 _v0 = _mm256_loadu_ps((const float*)(p));
            _mm256_storeu_ps((float*)(out), _v0);
            p += stride;
            out += 32;
        }
    }
}
#endif // __AVX512F__

#if __SSE2__
static void permute_transpose_pack4to1_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        const unsigned char* p = ptr + (y / 4) * stride;
        unsigned char* out = outptr + y * 2;
        for (int x = 0; x < cols; x++)
        {
            __m128i _r0 = _mm_loadl_epi64((const __m128i*)(p));
            _mm_storel_epi64((__m128i*)(out), _r0);
            p += 8;
            out += outstride;
        }
    }
}
#endif // __SSE2__

#if __SSE2__
static void permute_transpose_pack4to4_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        const unsigned char* p = ptr + (y / 4) * stride;
        unsigned char* out = outptr + y * 8;
        for (int x = 0; x < cols; x += 4)
        {
            permute_transpose4x4_bf16s_fp16s(p, 8, out, 8);
            p += 32;
            out += outstride;
        }
    }
}
#endif // __SSE2__

#if __AVX__
static void permute_transpose_pack4to8_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        const unsigned char* p = ptr + (y / 4) * stride;
        unsigned char* out = outptr + y * 16;
        for (int x = 0; x < cols; x += 8)
        {
            permute_transpose8x4_bf16s_fp16s(p, 8, out, 16);
            p += 64;
            out += outstride;
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute_transpose_pack4to16_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        const unsigned char* p = ptr + (y / 4) * stride;
        unsigned char* out = outptr + y * 32;
        for (int x = 0; x < cols; x += 16)
        {
            permute_transpose16x4_bf16s_fp16s(p, 8, out, 32);
            p += 128;
            out += outstride;
        }
    }
}
#endif // __AVX512F__

#if __AVX__
static void permute_transpose_pack8to1_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        const unsigned char* p = ptr + (y / 8) * stride;
        unsigned char* out = outptr + y * 2;
        for (int x = 0; x < cols; x++)
        {
            __m128i _r0 = _mm_loadu_si128((const __m128i*)(p));
            _mm_storeu_si128((__m128i*)(out), _r0);
            p += 16;
            out += outstride;
        }
    }
}
#endif // __AVX__

#if __AVX__
static void permute_transpose_pack8to4_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        const unsigned char* p = ptr + (y / 8) * stride;
        unsigned char* out = outptr + y * 8;
        for (int x = 0; x < cols; x += 4)
        {
            permute_transpose4x8_bf16s_fp16s(p, 16, out, 8);
            p += 64;
            out += outstride;
        }
    }
}
#endif // __AVX__

#if __AVX__
static void permute_transpose_pack8to8_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        const unsigned char* p = ptr + (y / 8) * stride;
        unsigned char* out = outptr + y * 16;
        for (int x = 0; x < cols; x += 8)
        {
            permute_transpose8x8_bf16s_fp16s(p, 16, out, 16);
            p += 128;
            out += outstride;
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute_transpose_pack8to16_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        const unsigned char* p = ptr + (y / 8) * stride;
        unsigned char* out = outptr + y * 32;
        for (int x = 0; x < cols; x += 16)
        {
            permute_transpose16x8_bf16s_fp16s(p, 16, out, 32);
            p += 256;
            out += outstride;
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute_transpose_pack16to1_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 16)
    {
        const unsigned char* p = ptr + (y / 16) * stride;
        unsigned char* out = outptr + y * 2;
        for (int x = 0; x < cols; x++)
        {
            __m256 _v0 = _mm256_loadu_ps((const float*)(p));
            _mm256_storeu_ps((float*)(out), _v0);
            p += 32;
            out += outstride;
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute_transpose_pack16to4_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 16)
    {
        const unsigned char* p = ptr + (y / 16) * stride;
        unsigned char* out = outptr + y * 8;
        for (int x = 0; x < cols; x += 4)
        {
            permute_transpose4x16_bf16s_fp16s(p, 32, out, 8);
            p += 128;
            out += outstride;
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute_transpose_pack16to8_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 16)
    {
        const unsigned char* p = ptr + (y / 16) * stride;
        unsigned char* out = outptr + y * 16;
        for (int x = 0; x < cols; x += 8)
        {
            permute_transpose8x16_bf16s_fp16s(p, 32, out, 16);
            p += 256;
            out += outstride;
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute_transpose_pack16to16_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 16)
    {
        const unsigned char* p = ptr + (y / 16) * stride;
        unsigned char* out = outptr + y * 32;
        for (int x = 0; x < cols; x += 16)
        {
            permute_transpose16x16_bf16s_fp16s(p, 32, out, 32);
            p += 512;
            out += outstride;
        }
    }
}
#endif // __AVX512F__

static void permute_transpose2d_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols, int elempack, int out_elempack)
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

#if __SSE2__
static void permute3d_pack1to4_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const int size = order_type <= 3 ? w : h;
    const size_t src_rowstep = (order_type <= 3 ? (size_t)w * elempack : elempack) * 2;
    const size_t src_step = (order_type <= 3 ? elempack : (size_t)w * elempack) * 2;
    const size_t dst_rowstep = (order_type == 2 ? (size_t)w : order_type == 4 ? h : 1) * out_elempack * 2;
    const size_t dst_step = (order_type == 2 || order_type == 4 ? (size_t)1 : channels * elempack) * out_elempack * 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < top_blob.c; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned char* p = (const unsigned char*)bottom_blob.channel(c) + q * 4 * src_rowstep;
            unsigned char* out = (unsigned char*)top_blob.channel(q) + c * 1 * dst_rowstep;
            if (order_type <= 3)
            {
                permute_transpose_pack1_bf16s_fp16s(p, src_rowstep, out, dst_step, 4, size);
                continue;
            }
            for (int i = 0; i < size; i++)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)(p));
                _mm_storel_epi64((__m128i*)(out), _r0);
                p += src_step;
                out += dst_step;
            }
        }
    }
}
#endif // __SSE2__

#if __AVX__
static void permute3d_pack1to8_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const int size = order_type <= 3 ? w : h;
    const size_t src_rowstep = (order_type <= 3 ? (size_t)w * elempack : elempack) * 2;
    const size_t src_step = (order_type <= 3 ? elempack : (size_t)w * elempack) * 2;
    const size_t dst_rowstep = (order_type == 2 ? (size_t)w : order_type == 4 ? h : 1) * out_elempack * 2;
    const size_t dst_step = (order_type == 2 || order_type == 4 ? (size_t)1 : channels * elempack) * out_elempack * 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < top_blob.c; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned char* p = (const unsigned char*)bottom_blob.channel(c) + q * 8 * src_rowstep;
            unsigned char* out = (unsigned char*)top_blob.channel(q) + c * 1 * dst_rowstep;
            if (order_type <= 3)
            {
                permute_transpose_pack1_bf16s_fp16s(p, src_rowstep, out, dst_step, 8, size);
                continue;
            }
            for (int i = 0; i < size; i++)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)(p));
                _mm_storeu_si128((__m128i*)(out), _r0);
                p += src_step;
                out += dst_step;
            }
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute3d_pack1to16_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const int size = order_type <= 3 ? w : h;
    const size_t src_rowstep = (order_type <= 3 ? (size_t)w * elempack : elempack) * 2;
    const size_t src_step = (order_type <= 3 ? elempack : (size_t)w * elempack) * 2;
    const size_t dst_rowstep = (order_type == 2 ? (size_t)w : order_type == 4 ? h : 1) * out_elempack * 2;
    const size_t dst_step = (order_type == 2 || order_type == 4 ? (size_t)1 : channels * elempack) * out_elempack * 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < top_blob.c; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned char* p = (const unsigned char*)bottom_blob.channel(c) + q * 16 * src_rowstep;
            unsigned char* out = (unsigned char*)top_blob.channel(q) + c * 1 * dst_rowstep;
            if (order_type <= 3)
            {
                permute_transpose_pack1_bf16s_fp16s(p, src_rowstep, out, dst_step, 16, size);
                continue;
            }
            for (int i = 0; i < size; i++)
            {
                __m256 _v0 = _mm256_loadu_ps((const float*)(p));
                _mm256_storeu_ps((float*)(out), _v0);
                p += src_step;
                out += dst_step;
            }
        }
    }
}
#endif // __AVX512F__

#if __SSE2__
static void permute3d_pack4to1_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const int size = order_type <= 3 ? w : h;
    const size_t src_rowstep = (order_type <= 3 ? (size_t)w * elempack : elempack) * 2;
    const size_t src_step = (order_type <= 3 ? elempack : (size_t)w * elempack) * 2;
    const size_t dst_rowstep = (order_type == 2 ? (size_t)w : order_type == 4 ? h : 1) * out_elempack * 2;
    const size_t dst_step = (order_type == 2 || order_type == 4 ? (size_t)1 : channels * elempack) * out_elempack * 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < top_blob.c; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned char* p = (const unsigned char*)bottom_blob.channel(c) + q * 1 * src_rowstep;
            unsigned char* out = (unsigned char*)top_blob.channel(q) + c * 4 * dst_rowstep;
            if (order_type == 2 || order_type == 4)
            {
                permute_transpose_pack1_bf16s_fp16s(p, src_step, out, dst_rowstep, size, 4);
                continue;
            }
            for (int i = 0; i < size; i++)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)(p));
                _mm_storel_epi64((__m128i*)(out), _r0);
                p += src_step;
                out += dst_step;
            }
        }
    }
}
#endif // __SSE2__

#if __SSE2__
static void permute3d_pack4to4_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const int size = order_type <= 3 ? w : h;
    const size_t src_rowstep = (order_type <= 3 ? (size_t)w * elempack : elempack) * 2;
    const size_t src_step = (order_type <= 3 ? elempack : (size_t)w * elempack) * 2;
    const size_t dst_rowstep = (order_type == 2 ? (size_t)w : order_type == 4 ? h : 1) * out_elempack * 2;
    const size_t dst_step = (order_type == 2 || order_type == 4 ? (size_t)1 : channels * elempack) * out_elempack * 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < top_blob.c; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned char* p = (const unsigned char*)bottom_blob.channel(c) + q * 4 * src_rowstep;
            unsigned char* out = (unsigned char*)top_blob.channel(q) + c * 4 * dst_rowstep;
            for (int i = 0; i < size; i++)
            {
                permute_transpose4x4_bf16s_fp16s(p, src_rowstep, out, dst_rowstep);
                p += src_step;
                out += dst_step;
            }
        }
    }
}
#endif // __SSE2__

#if __AVX__
static void permute3d_pack4to8_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const int size = order_type <= 3 ? w : h;
    const size_t src_rowstep = (order_type <= 3 ? (size_t)w * elempack : elempack) * 2;
    const size_t src_step = (order_type <= 3 ? elempack : (size_t)w * elempack) * 2;
    const size_t dst_rowstep = (order_type == 2 ? (size_t)w : order_type == 4 ? h : 1) * out_elempack * 2;
    const size_t dst_step = (order_type == 2 || order_type == 4 ? (size_t)1 : channels * elempack) * out_elempack * 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < top_blob.c; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned char* p = (const unsigned char*)bottom_blob.channel(c) + q * 8 * src_rowstep;
            unsigned char* out = (unsigned char*)top_blob.channel(q) + c * 4 * dst_rowstep;
            for (int i = 0; i < size; i++)
            {
                permute_transpose8x4_bf16s_fp16s(p, src_rowstep, out, dst_rowstep);
                p += src_step;
                out += dst_step;
            }
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute3d_pack4to16_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const int size = order_type <= 3 ? w : h;
    const size_t src_rowstep = (order_type <= 3 ? (size_t)w * elempack : elempack) * 2;
    const size_t src_step = (order_type <= 3 ? elempack : (size_t)w * elempack) * 2;
    const size_t dst_rowstep = (order_type == 2 ? (size_t)w : order_type == 4 ? h : 1) * out_elempack * 2;
    const size_t dst_step = (order_type == 2 || order_type == 4 ? (size_t)1 : channels * elempack) * out_elempack * 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < top_blob.c; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned char* p = (const unsigned char*)bottom_blob.channel(c) + q * 16 * src_rowstep;
            unsigned char* out = (unsigned char*)top_blob.channel(q) + c * 4 * dst_rowstep;
            for (int i = 0; i < size; i++)
            {
                permute_transpose16x4_bf16s_fp16s(p, src_rowstep, out, dst_rowstep);
                p += src_step;
                out += dst_step;
            }
        }
    }
}
#endif // __AVX512F__

#if __AVX__
static void permute3d_pack8to1_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const int size = order_type <= 3 ? w : h;
    const size_t src_rowstep = (order_type <= 3 ? (size_t)w * elempack : elempack) * 2;
    const size_t src_step = (order_type <= 3 ? elempack : (size_t)w * elempack) * 2;
    const size_t dst_rowstep = (order_type == 2 ? (size_t)w : order_type == 4 ? h : 1) * out_elempack * 2;
    const size_t dst_step = (order_type == 2 || order_type == 4 ? (size_t)1 : channels * elempack) * out_elempack * 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < top_blob.c; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned char* p = (const unsigned char*)bottom_blob.channel(c) + q * 1 * src_rowstep;
            unsigned char* out = (unsigned char*)top_blob.channel(q) + c * 8 * dst_rowstep;
            if (order_type == 2 || order_type == 4)
            {
                permute_transpose_pack1_bf16s_fp16s(p, src_step, out, dst_rowstep, size, 8);
                continue;
            }
            for (int i = 0; i < size; i++)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)(p));
                _mm_storeu_si128((__m128i*)(out), _r0);
                p += src_step;
                out += dst_step;
            }
        }
    }
}
#endif // __AVX__

#if __AVX__
static void permute3d_pack8to4_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const int size = order_type <= 3 ? w : h;
    const size_t src_rowstep = (order_type <= 3 ? (size_t)w * elempack : elempack) * 2;
    const size_t src_step = (order_type <= 3 ? elempack : (size_t)w * elempack) * 2;
    const size_t dst_rowstep = (order_type == 2 ? (size_t)w : order_type == 4 ? h : 1) * out_elempack * 2;
    const size_t dst_step = (order_type == 2 || order_type == 4 ? (size_t)1 : channels * elempack) * out_elempack * 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < top_blob.c; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned char* p = (const unsigned char*)bottom_blob.channel(c) + q * 4 * src_rowstep;
            unsigned char* out = (unsigned char*)top_blob.channel(q) + c * 8 * dst_rowstep;
            for (int i = 0; i < size; i++)
            {
                permute_transpose4x8_bf16s_fp16s(p, src_rowstep, out, dst_rowstep);
                p += src_step;
                out += dst_step;
            }
        }
    }
}
#endif // __AVX__

#if __AVX__
static void permute3d_pack8to8_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const int size = order_type <= 3 ? w : h;
    const size_t src_rowstep = (order_type <= 3 ? (size_t)w * elempack : elempack) * 2;
    const size_t src_step = (order_type <= 3 ? elempack : (size_t)w * elempack) * 2;
    const size_t dst_rowstep = (order_type == 2 ? (size_t)w : order_type == 4 ? h : 1) * out_elempack * 2;
    const size_t dst_step = (order_type == 2 || order_type == 4 ? (size_t)1 : channels * elempack) * out_elempack * 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < top_blob.c; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned char* p = (const unsigned char*)bottom_blob.channel(c) + q * 8 * src_rowstep;
            unsigned char* out = (unsigned char*)top_blob.channel(q) + c * 8 * dst_rowstep;
            for (int i = 0; i < size; i++)
            {
                permute_transpose8x8_bf16s_fp16s(p, src_rowstep, out, dst_rowstep);
                p += src_step;
                out += dst_step;
            }
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute3d_pack8to16_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const int size = order_type <= 3 ? w : h;
    const size_t src_rowstep = (order_type <= 3 ? (size_t)w * elempack : elempack) * 2;
    const size_t src_step = (order_type <= 3 ? elempack : (size_t)w * elempack) * 2;
    const size_t dst_rowstep = (order_type == 2 ? (size_t)w : order_type == 4 ? h : 1) * out_elempack * 2;
    const size_t dst_step = (order_type == 2 || order_type == 4 ? (size_t)1 : channels * elempack) * out_elempack * 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < top_blob.c; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned char* p = (const unsigned char*)bottom_blob.channel(c) + q * 16 * src_rowstep;
            unsigned char* out = (unsigned char*)top_blob.channel(q) + c * 8 * dst_rowstep;
            for (int i = 0; i < size; i++)
            {
                permute_transpose16x8_bf16s_fp16s(p, src_rowstep, out, dst_rowstep);
                p += src_step;
                out += dst_step;
            }
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute3d_pack16to1_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const int size = order_type <= 3 ? w : h;
    const size_t src_rowstep = (order_type <= 3 ? (size_t)w * elempack : elempack) * 2;
    const size_t src_step = (order_type <= 3 ? elempack : (size_t)w * elempack) * 2;
    const size_t dst_rowstep = (order_type == 2 ? (size_t)w : order_type == 4 ? h : 1) * out_elempack * 2;
    const size_t dst_step = (order_type == 2 || order_type == 4 ? (size_t)1 : channels * elempack) * out_elempack * 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < top_blob.c; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned char* p = (const unsigned char*)bottom_blob.channel(c) + q * 1 * src_rowstep;
            unsigned char* out = (unsigned char*)top_blob.channel(q) + c * 16 * dst_rowstep;
            if (order_type == 2 || order_type == 4)
            {
                permute_transpose_pack1_bf16s_fp16s(p, src_step, out, dst_rowstep, size, 16);
                continue;
            }
            for (int i = 0; i < size; i++)
            {
                __m256 _v0 = _mm256_loadu_ps((const float*)(p));
                _mm256_storeu_ps((float*)(out), _v0);
                p += src_step;
                out += dst_step;
            }
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute3d_pack16to4_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const int size = order_type <= 3 ? w : h;
    const size_t src_rowstep = (order_type <= 3 ? (size_t)w * elempack : elempack) * 2;
    const size_t src_step = (order_type <= 3 ? elempack : (size_t)w * elempack) * 2;
    const size_t dst_rowstep = (order_type == 2 ? (size_t)w : order_type == 4 ? h : 1) * out_elempack * 2;
    const size_t dst_step = (order_type == 2 || order_type == 4 ? (size_t)1 : channels * elempack) * out_elempack * 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < top_blob.c; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned char* p = (const unsigned char*)bottom_blob.channel(c) + q * 4 * src_rowstep;
            unsigned char* out = (unsigned char*)top_blob.channel(q) + c * 16 * dst_rowstep;
            for (int i = 0; i < size; i++)
            {
                permute_transpose4x16_bf16s_fp16s(p, src_rowstep, out, dst_rowstep);
                p += src_step;
                out += dst_step;
            }
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute3d_pack16to8_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const int size = order_type <= 3 ? w : h;
    const size_t src_rowstep = (order_type <= 3 ? (size_t)w * elempack : elempack) * 2;
    const size_t src_step = (order_type <= 3 ? elempack : (size_t)w * elempack) * 2;
    const size_t dst_rowstep = (order_type == 2 ? (size_t)w : order_type == 4 ? h : 1) * out_elempack * 2;
    const size_t dst_step = (order_type == 2 || order_type == 4 ? (size_t)1 : channels * elempack) * out_elempack * 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < top_blob.c; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned char* p = (const unsigned char*)bottom_blob.channel(c) + q * 8 * src_rowstep;
            unsigned char* out = (unsigned char*)top_blob.channel(q) + c * 16 * dst_rowstep;
            for (int i = 0; i < size; i++)
            {
                permute_transpose8x16_bf16s_fp16s(p, src_rowstep, out, dst_rowstep);
                p += src_step;
                out += dst_step;
            }
        }
    }
}
#endif // __AVX512F__

#if __AVX512F__
static void permute3d_pack16to16_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const int size = order_type <= 3 ? w : h;
    const size_t src_rowstep = (order_type <= 3 ? (size_t)w * elempack : elempack) * 2;
    const size_t src_step = (order_type <= 3 ? elempack : (size_t)w * elempack) * 2;
    const size_t dst_rowstep = (order_type == 2 ? (size_t)w : order_type == 4 ? h : 1) * out_elempack * 2;
    const size_t dst_step = (order_type == 2 || order_type == 4 ? (size_t)1 : channels * elempack) * out_elempack * 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < top_blob.c; q++)
    {
        for (int c = 0; c < channels; c++)
        {
            const unsigned char* p = (const unsigned char*)bottom_blob.channel(c) + q * 16 * src_rowstep;
            unsigned char* out = (unsigned char*)top_blob.channel(q) + c * 16 * dst_rowstep;
            for (int i = 0; i < size; i++)
            {
                permute_transpose16x16_bf16s_fp16s(p, src_rowstep, out, dst_rowstep);
                p += src_step;
                out += dst_step;
            }
        }
    }
}
#endif // __AVX512F__

static void permute3d_cross_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
#if __SSE2__
    if (elempack == 1 && out_elempack == 4)
    {
        permute3d_pack1to4_bf16s_fp16s(bottom_blob, top_blob, order_type, opt);
        return;
    }
#endif // __SSE2__
#if __AVX__
    if (elempack == 1 && out_elempack == 8)
    {
        permute3d_pack1to8_bf16s_fp16s(bottom_blob, top_blob, order_type, opt);
        return;
    }
#endif // __AVX__
#if __AVX512F__
    if (elempack == 1 && out_elempack == 16)
    {
        permute3d_pack1to16_bf16s_fp16s(bottom_blob, top_blob, order_type, opt);
        return;
    }
#endif // __AVX512F__
#if __SSE2__
    if (elempack == 4 && out_elempack == 1)
    {
        permute3d_pack4to1_bf16s_fp16s(bottom_blob, top_blob, order_type, opt);
        return;
    }
#endif // __SSE2__
#if __SSE2__
    if (elempack == 4 && out_elempack == 4)
    {
        permute3d_pack4to4_bf16s_fp16s(bottom_blob, top_blob, order_type, opt);
        return;
    }
#endif // __SSE2__
#if __AVX__
    if (elempack == 4 && out_elempack == 8)
    {
        permute3d_pack4to8_bf16s_fp16s(bottom_blob, top_blob, order_type, opt);
        return;
    }
#endif // __AVX__
#if __AVX512F__
    if (elempack == 4 && out_elempack == 16)
    {
        permute3d_pack4to16_bf16s_fp16s(bottom_blob, top_blob, order_type, opt);
        return;
    }
#endif // __AVX512F__
#if __AVX__
    if (elempack == 8 && out_elempack == 1)
    {
        permute3d_pack8to1_bf16s_fp16s(bottom_blob, top_blob, order_type, opt);
        return;
    }
#endif // __AVX__
#if __AVX__
    if (elempack == 8 && out_elempack == 4)
    {
        permute3d_pack8to4_bf16s_fp16s(bottom_blob, top_blob, order_type, opt);
        return;
    }
#endif // __AVX__
#if __AVX__
    if (elempack == 8 && out_elempack == 8)
    {
        permute3d_pack8to8_bf16s_fp16s(bottom_blob, top_blob, order_type, opt);
        return;
    }
#endif // __AVX__
#if __AVX512F__
    if (elempack == 8 && out_elempack == 16)
    {
        permute3d_pack8to16_bf16s_fp16s(bottom_blob, top_blob, order_type, opt);
        return;
    }
#endif // __AVX512F__
#if __AVX512F__
    if (elempack == 16 && out_elempack == 1)
    {
        permute3d_pack16to1_bf16s_fp16s(bottom_blob, top_blob, order_type, opt);
        return;
    }
#endif // __AVX512F__
#if __AVX512F__
    if (elempack == 16 && out_elempack == 4)
    {
        permute3d_pack16to4_bf16s_fp16s(bottom_blob, top_blob, order_type, opt);
        return;
    }
#endif // __AVX512F__
#if __AVX512F__
    if (elempack == 16 && out_elempack == 8)
    {
        permute3d_pack16to8_bf16s_fp16s(bottom_blob, top_blob, order_type, opt);
        return;
    }
#endif // __AVX512F__
#if __AVX512F__
    if (elempack == 16 && out_elempack == 16)
    {
        permute3d_pack16to16_bf16s_fp16s(bottom_blob, top_blob, order_type, opt);
        return;
    }
#endif // __AVX512F__
}

#if __SSE2__
static void permute_spatial_pack4_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        const int ymax = std::min(y + 8, rows);
        for (int x = 0; x < cols; x += 8)
        {
            const int xmax = std::min(x + 8, cols);
            for (int i = y; i < ymax; i++)
            {
                const unsigned char* p = ptr + i * stride + (x * 4) * 2;
                unsigned char* out = outptr + x * outstride + (i * 4) * 2;
                for (int j = x; j < xmax; j++)
                {
                    __m128i _v0 = _mm_loadl_epi64((const __m128i*)(p));
                    _mm_storel_epi64((__m128i*)(out), _v0);
                    p += 4 * 2;
                    out += outstride;
                }
            }
        }
    }
}
#endif // __SSE2__

#if __AVX__
static void permute_spatial_pack8_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        const int ymax = std::min(y + 8, rows);
        for (int x = 0; x < cols; x += 8)
        {
            const int xmax = std::min(x + 8, cols);
            for (int i = y; i < ymax; i++)
            {
                const unsigned char* p = ptr + i * stride + (x * 8) * 2;
                unsigned char* out = outptr + x * outstride + (i * 8) * 2;
                for (int j = x; j < xmax; j++)
                {
                    __m128i _v0 = _mm_loadu_si128((const __m128i*)(p));
                    _mm_storeu_si128((__m128i*)(out), _v0);
                    p += 8 * 2;
                    out += outstride;
                }
            }
        }
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute_spatial_pack16_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        const int ymax = std::min(y + 8, rows);
        for (int x = 0; x < cols; x += 8)
        {
            const int xmax = std::min(x + 8, cols);
            for (int i = y; i < ymax; i++)
            {
                const unsigned char* p = ptr + i * stride + (x * 16) * 2;
                unsigned char* out = outptr + x * outstride + (i * 16) * 2;
                for (int j = x; j < xmax; j++)
                {
                    __m256 _v0 = _mm256_loadu_ps((const float*)(p));
                    _mm256_storeu_ps((float*)(out), _v0);
                    p += 16 * 2;
                    out += outstride;
                }
            }
        }
    }
}
#endif // __AVX512F__

static void permute_transpose_spatial_bf16s_fp16s(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int rows, int cols, int elempack)
{
    if (elempack == 1)
    {
        permute_transpose_pack1_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#if __SSE2__
    if (elempack == 4)
    {
        permute_spatial_pack4_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __SSE2__
#if __AVX__
    if (elempack == 8)
    {
        permute_spatial_pack8_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        permute_spatial_pack16_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
#endif // __AVX512F__
}

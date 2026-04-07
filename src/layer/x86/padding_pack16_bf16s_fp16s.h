// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void padding_constant_pack16_bf16s_fp16s_avx512(const Mat& src, Mat& dst, int top, int bottom, int left, int right, const __m256i& v)
{
    const unsigned short* ptr = src;
    unsigned short* outptr = dst;
    int top_size = top * dst.w;
    int bottom_size = bottom * dst.w;

    // fill top
    for (int y = 0; y < top_size; y++)
    {
        _mm256_storeu_si256((__m256i*)outptr, v);
        outptr += 16;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            _mm256_storeu_si256((__m256i*)outptr, v);
            outptr += 16;
        }
        for (int x = 0; x < src.w; x++)
        {
            _mm256_storeu_si256((__m256i*)outptr, _mm256_loadu_si256((const __m256i*)ptr));
            ptr += 16;
            outptr += 16;
        }
        for (int x = 0; x < right; x++)
        {
            _mm256_storeu_si256((__m256i*)outptr, v);
            outptr += 16;
        }
    }
    // fill bottom
    for (int y = 0; y < bottom_size; y++)
    {
        _mm256_storeu_si256((__m256i*)outptr, v);
        outptr += 16;
    }
}

static void padding_replicate_pack16_bf16s_fp16s_avx512(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const unsigned short* ptr = src;
    unsigned short* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        const unsigned short* ptr0 = ptr;
        __m256i _p = _mm256_loadu_si256((const __m256i*)ptr0);
        for (int x = 0; x < left; x++)
        {
            _mm256_storeu_si256((__m256i*)outptr, _p);
            outptr += 16;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = _mm256_loadu_si256((const __m256i*)ptr0);
            _mm256_storeu_si256((__m256i*)outptr, _p);
            ptr0 += 16;
            outptr += 16;
        }
        for (int x = 0; x < right; x++)
        {
            _mm256_storeu_si256((__m256i*)outptr, _p);
            outptr += 16;
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        __m256i _p = _mm256_loadu_si256((const __m256i*)ptr);
        for (int x = 0; x < left; x++)
        {
            _mm256_storeu_si256((__m256i*)outptr, _p);
            outptr += 16;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = _mm256_loadu_si256((const __m256i*)ptr);
            _mm256_storeu_si256((__m256i*)outptr, _p);
            ptr += 16;
            outptr += 16;
        }
        for (int x = 0; x < right; x++)
        {
            _mm256_storeu_si256((__m256i*)outptr, _p);
            outptr += 16;
        }
    }
    // fill bottom
    ptr -= src.w * 16;
    for (int y = 0; y < bottom; y++)
    {
        const unsigned short* ptr0 = ptr;
        __m256i _p = _mm256_loadu_si256((const __m256i*)ptr0);
        for (int x = 0; x < left; x++)
        {
            _mm256_storeu_si256((__m256i*)outptr, _p);
            outptr += 16;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = _mm256_loadu_si256((const __m256i*)ptr0);
            _mm256_storeu_si256((__m256i*)outptr, _p);
            ptr0 += 16;
            outptr += 16;
        }
        for (int x = 0; x < right; x++)
        {
            _mm256_storeu_si256((__m256i*)outptr, _p);
            outptr += 16;
        }
    }
}

static void padding_reflect_pack16_bf16s_fp16s_avx512(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const unsigned short* ptr = src;
    unsigned short* outptr = dst;

    // fill top
    ptr += top * src.w * 16;
    for (int y = 0; y < top; y++)
    {
        const unsigned short* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            __m256i _p = _mm256_loadu_si256((const __m256i*)(ptr0 + (left - x) * 16));
            _mm256_storeu_si256((__m256i*)outptr, _p);
            outptr += 16;
        }
        for (int x = 0; x < src.w; x++)
        {
            __m256i _p = _mm256_loadu_si256((const __m256i*)ptr0);
            _mm256_storeu_si256((__m256i*)outptr, _p);
            ptr0 += 16;
            outptr += 16;
        }
        for (int x = 0; x < right; x++)
        {
            __m256i _p = _mm256_loadu_si256((const __m256i*)(ptr0 - 32 - x * 16));
            _mm256_storeu_si256((__m256i*)outptr, _p);
            outptr += 16;
        }
        ptr -= src.w * 16;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            __m256i _p = _mm256_loadu_si256((const __m256i*)(ptr + (left - x) * 16));
            _mm256_storeu_si256((__m256i*)outptr, _p);
            outptr += 16;
        }
        for (int x = 0; x < src.w; x++)
        {
            __m256i _p = _mm256_loadu_si256((const __m256i*)ptr);
            _mm256_storeu_si256((__m256i*)outptr, _p);
            ptr += 16;
            outptr += 16;
        }
        for (int x = 0; x < right; x++)
        {
            __m256i _p = _mm256_loadu_si256((const __m256i*)(ptr - 32 - x * 16));
            _mm256_storeu_si256((__m256i*)outptr, _p);
            outptr += 16;
        }
    }
    // fill bottom
    ptr -= 2 * src.w * 16;
    for (int y = 0; y < bottom; y++)
    {
        const unsigned short* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            __m256i _p = _mm256_loadu_si256((const __m256i*)(ptr0 + (left - x) * 16));
            _mm256_storeu_si256((__m256i*)outptr, _p);
            outptr += 16;
        }
        for (int x = 0; x < src.w; x++)
        {
            __m256i _p = _mm256_loadu_si256((const __m256i*)ptr0);
            _mm256_storeu_si256((__m256i*)outptr, _p);
            ptr0 += 16;
            outptr += 16;
        }
        for (int x = 0; x < right; x++)
        {
            __m256i _p = _mm256_loadu_si256((const __m256i*)(ptr0 - 32 - x * 16));
            _mm256_storeu_si256((__m256i*)outptr, _p);
            outptr += 16;
        }
        ptr -= src.w * 16;
    }
}

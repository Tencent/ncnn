// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void padding_constant_pack8_bf16s_fp16s_avx(const Mat& src, Mat& dst, int top, int bottom, int left, int right, const __m128i& v)
{
    const unsigned short* ptr = src;
    unsigned short* outptr = dst;
    int top_size = top * dst.w;
    int bottom_size = bottom * dst.w;

    // fill top
    for (int y = 0; y < top_size; y++)
    {
        _mm_storeu_si128((__m128i*)outptr, v);
        outptr += 8;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            _mm_storeu_si128((__m128i*)outptr, v);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _mm_storeu_si128((__m128i*)outptr, _mm_loadu_si128((const __m128i*)ptr));
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            _mm_storeu_si128((__m128i*)outptr, v);
            outptr += 8;
        }
    }
    // fill bottom
    for (int y = 0; y < bottom_size; y++)
    {
        _mm_storeu_si128((__m128i*)outptr, v);
        outptr += 8;
    }
}

static void padding_replicate_pack8_bf16s_fp16s_avx(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const unsigned short* ptr = src;
    unsigned short* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        const unsigned short* ptr0 = ptr;
        __m128i _p = _mm_loadu_si128((const __m128i*)ptr0);
        for (int x = 0; x < left; x++)
        {
            _mm_storeu_si128((__m128i*)outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = _mm_loadu_si128((const __m128i*)ptr0);
            _mm_storeu_si128((__m128i*)outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            _mm_storeu_si128((__m128i*)outptr, _p);
            outptr += 8;
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        __m128i _p = _mm_loadu_si128((const __m128i*)ptr);
        for (int x = 0; x < left; x++)
        {
            _mm_storeu_si128((__m128i*)outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = _mm_loadu_si128((const __m128i*)ptr);
            _mm_storeu_si128((__m128i*)outptr, _p);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            _mm_storeu_si128((__m128i*)outptr, _p);
            outptr += 8;
        }
    }
    // fill bottom
    ptr -= src.w * 8;
    for (int y = 0; y < bottom; y++)
    {
        const unsigned short* ptr0 = ptr;
        __m128i _p = _mm_loadu_si128((const __m128i*)ptr0);
        for (int x = 0; x < left; x++)
        {
            _mm_storeu_si128((__m128i*)outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = _mm_loadu_si128((const __m128i*)ptr0);
            _mm_storeu_si128((__m128i*)outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            _mm_storeu_si128((__m128i*)outptr, _p);
            outptr += 8;
        }
    }
}

static void padding_reflect_pack8_bf16s_fp16s_avx(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const unsigned short* ptr = src;
    unsigned short* outptr = dst;

    // fill top
    ptr += top * src.w * 8;
    for (int y = 0; y < top; y++)
    {
        const unsigned short* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            __m128i _p = _mm_loadu_si128((const __m128i*)(ptr0 + (left - x) * 8));
            _mm_storeu_si128((__m128i*)outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            __m128i _p = _mm_loadu_si128((const __m128i*)ptr0);
            _mm_storeu_si128((__m128i*)outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __m128i _p = _mm_loadu_si128((const __m128i*)(ptr0 - 16 - x * 8));
            _mm_storeu_si128((__m128i*)outptr, _p);
            outptr += 8;
        }
        ptr -= src.w * 8;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            __m128i _p = _mm_loadu_si128((const __m128i*)(ptr + (left - x) * 8));
            _mm_storeu_si128((__m128i*)outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            __m128i _p = _mm_loadu_si128((const __m128i*)ptr);
            _mm_storeu_si128((__m128i*)outptr, _p);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __m128i _p = _mm_loadu_si128((const __m128i*)(ptr - 16 - x * 8));
            _mm_storeu_si128((__m128i*)outptr, _p);
            outptr += 8;
        }
    }
    // fill bottom
    ptr -= 2 * src.w * 8;
    for (int y = 0; y < bottom; y++)
    {
        const unsigned short* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            __m128i _p = _mm_loadu_si128((const __m128i*)(ptr0 + (left - x) * 8));
            _mm_storeu_si128((__m128i*)outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            __m128i _p = _mm_loadu_si128((const __m128i*)ptr0);
            _mm_storeu_si128((__m128i*)outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __m128i _p = _mm_loadu_si128((const __m128i*)(ptr0 - 16 - x * 8));
            _mm_storeu_si128((__m128i*)outptr, _p);
            outptr += 8;
        }
        ptr -= src.w * 8;
    }
}

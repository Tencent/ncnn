// Copyright 2026 nihui. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

static void padding_constant_pack8_lasx(const Mat& src, Mat& dst, int top, int bottom, int left, int right, const __m256& v)
{
    const float* ptr = src;
    float* outptr = dst;
    int top_size = top * dst.w;
    int bottom_size = bottom * dst.w;

    // fill top
    for (int y = 0; y < top_size; y++)
    {
        __lasx_xvst((__m256i)v, outptr, 0);
        outptr += 8;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            __lasx_xvst((__m256i)v, outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            __lasx_xvst(__lasx_xvld(ptr, 0), outptr, 0);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __lasx_xvst((__m256i)v, outptr, 0);
            outptr += 8;
        }
    }
    // fill bottom
    for (int y = 0; y < bottom_size; y++)
    {
        __lasx_xvst((__m256i)v, outptr, 0);
        outptr += 8;
    }
}

static void padding_replicate_pack8_lasx(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const float* ptr = src;
    float* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        const float* ptr0 = ptr;
        __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
        for (int x = 0; x < left; x++)
        {
            __lasx_xvst((__m256i)_p, outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = (__m256)__lasx_xvld(ptr0, 0);
            __lasx_xvst((__m256i)_p, outptr, 0);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __lasx_xvst((__m256i)_p, outptr, 0);
            outptr += 8;
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        __m256 _p = (__m256)__lasx_xvld(ptr, 0);
        for (int x = 0; x < left; x++)
        {
            __lasx_xvst((__m256i)_p, outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = (__m256)__lasx_xvld(ptr, 0);
            __lasx_xvst((__m256i)_p, outptr, 0);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __lasx_xvst((__m256i)_p, outptr, 0);
            outptr += 8;
        }
    }
    // fill bottom
    ptr -= src.w * 8;
    for (int y = 0; y < bottom; y++)
    {
        const float* ptr0 = ptr;
        __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
        for (int x = 0; x < left; x++)
        {
            __lasx_xvst((__m256i)_p, outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = (__m256)__lasx_xvld(ptr0, 0);
            __lasx_xvst((__m256i)_p, outptr, 0);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __lasx_xvst((__m256i)_p, outptr, 0);
            outptr += 8;
        }
    }
}

static void padding_reflect_pack8_lasx(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const float* ptr = src;
    float* outptr = dst;

    // fill top
    ptr += top * src.w * 8;
    for (int y = 0; y < top; y++)
    {
        const float* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr0 + (left - x) * 8, 0);
            __lasx_xvst((__m256i)_p, outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
            __lasx_xvst((__m256i)_p, outptr, 0);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr0 - 16 - x * 8, 0);
            __lasx_xvst((__m256i)_p, outptr, 0);
            outptr += 8;
        }
        ptr -= src.w * 8;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr + (left - x) * 8, 0);
            __lasx_xvst((__m256i)_p, outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            __lasx_xvst((__m256i)_p, outptr, 0);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr - 16 - x * 8, 0);
            __lasx_xvst((__m256i)_p, outptr, 0);
            outptr += 8;
        }
    }
    // fill bottom
    ptr -= 2 * src.w * 8;
    for (int y = 0; y < bottom; y++)
    {
        const float* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr0 + (left - x) * 8, 0);
            __lasx_xvst((__m256i)_p, outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
            __lasx_xvst((__m256i)_p, outptr, 0);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr0 - 16 - x * 8, 0);
            __lasx_xvst((__m256i)_p, outptr, 0);
            outptr += 8;
        }
        ptr -= src.w * 8;
    }
}

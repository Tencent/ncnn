// Tencent is pleased to support the open source community by making ncnn available.
//
//                    https://opensource.org/licenses/BSD-3-Clause
//
// Copyright (C) 2026 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

static void padding_constant_pack8_bf16s_lasx(const Mat& src, Mat& dst, int top, int bottom, int left, int right, const __m128i& v)
{
    const unsigned short* ptr = src;
    unsigned short* outptr = dst;
    int top_size = top * dst.w;
    int bottom_size = bottom * dst.w;

    // fill top
    for (int y = 0; y < top_size; y++)
    {
        __lsx_vst(v, outptr, 0);
        outptr += 8;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            __lsx_vst(v, outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            __lsx_vst(__lsx_vld(ptr, 0), outptr, 0);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __lsx_vst(v, outptr, 0);
            outptr += 8;
        }
    }
    // fill bottom
    for (int y = 0; y < bottom_size; y++)
    {
        __lsx_vst(v, outptr, 0);
        outptr += 8;
    }
}

static void padding_replicate_pack8_bf16s_lasx(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const unsigned short* ptr = src;
    unsigned short* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        const unsigned short* ptr0 = ptr;
        __m128i _p = __lsx_vld(ptr0, 0);
        for (int x = 0; x < left; x++)
        {
            __lsx_vst(_p, outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = __lsx_vld(ptr0, 0);
            __lsx_vst(_p, outptr, 0);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __lsx_vst(_p, outptr, 0);
            outptr += 8;
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        __m128i _p = __lsx_vld(ptr, 0);
        for (int x = 0; x < left; x++)
        {
            __lsx_vst(_p, outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = __lsx_vld(ptr, 0);
            __lsx_vst(_p, outptr, 0);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __lsx_vst(_p, outptr, 0);
            outptr += 8;
        }
    }
    // fill bottom
    ptr -= src.w * 8;
    for (int y = 0; y < bottom; y++)
    {
        const unsigned short* ptr0 = ptr;
        __m128i _p = __lsx_vld(ptr0, 0);
        for (int x = 0; x < left; x++)
        {
            __lsx_vst(_p, outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = __lsx_vld(ptr0, 0);
            __lsx_vst(_p, outptr, 0);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __lsx_vst(_p, outptr, 0);
            outptr += 8;
        }
    }
}

static void padding_reflect_pack8_bf16s_lasx(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
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
            __lsx_vst(__lsx_vld(ptr0 + (left - x) * 8, 0), outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            __lsx_vst(__lsx_vld(ptr0, 0), outptr, 0);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __lsx_vst(__lsx_vld(ptr0 - 16 - x * 8, 0), outptr, 0);
            outptr += 8;
        }
        ptr -= src.w * 8;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            __lsx_vst(__lsx_vld(ptr + (left - x) * 8, 0), outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            __lsx_vst(__lsx_vld(ptr, 0), outptr, 0);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __lsx_vst(__lsx_vld(ptr - 16 - x * 8, 0), outptr, 0);
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
            __lsx_vst(__lsx_vld(ptr0 + (left - x) * 8, 0), outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            __lsx_vst(__lsx_vld(ptr0, 0), outptr, 0);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __lsx_vst(__lsx_vld(ptr0 - 16 - x * 8, 0), outptr, 0);
            outptr += 8;
        }
        ptr -= src.w * 8;
    }
}

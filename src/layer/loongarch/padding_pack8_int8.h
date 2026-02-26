// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

static void padding_constant_pack8_int8_lsx(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int64_t _v)
{
    const int64_t* ptr = src;
    int64_t* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            *outptr++ = _v;
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            *outptr++ = _v;
        }
        for (int x = 0; x < src.w; x++)
        {
            *outptr++ = *ptr++;
        }
        for (int x = 0; x < right; x++)
        {
            *outptr++ = _v;
        }
    }
    // fill bottom
    for (int y = 0; y < bottom; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            *outptr++ = _v;
        }
    }
}

static void padding_replicate_pack8_int8_lsx(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const int64_t* ptr = src;
    int64_t* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        const int64_t* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            *outptr++ = *ptr0;
        }
        for (int x = 0; x < src.w; x++)
        {
            *outptr++ = *ptr0++;
        }
        for (int x = 0; x < right; x++)
        {
            *outptr++ = ptr0[-1];
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            *outptr++ = *ptr;
        }
        for (int x = 0; x < src.w; x++)
        {
            *outptr++ = *ptr++;
        }
        for (int x = 0; x < right; x++)
        {
            *outptr++ = ptr[-1];
        }
    }
    // fill bottom
    ptr -= src.w;
    for (int y = 0; y < bottom; y++)
    {
        const int64_t* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            *outptr++ = *ptr0;
        }
        for (int x = 0; x < src.w; x++)
        {
            *outptr++ = *ptr0++;
        }
        for (int x = 0; x < right; x++)
        {
            *outptr++ = ptr0[-1];
        }
    }
}

static void padding_reflect_pack8_int8_lsx(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const int64_t* ptr = src;
    int64_t* outptr = dst;

    // fill top
    ptr += top * src.w;
    for (int y = 0; y < top; y++)
    {
        const int64_t* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            *outptr++ = ptr0[left - x];
        }
        for (int x = 0; x < src.w; x++)
        {
            *outptr++ = *ptr0++;
        }
        for (int x = 0; x < right; x++)
        {
            *outptr++ = ptr0[-2 - x];
        }
        ptr -= src.w;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            *outptr++ = ptr[left - x];
        }
        for (int x = 0; x < src.w; x++)
        {
            *outptr++ = *ptr++;
        }
        for (int x = 0; x < right; x++)
        {
            *outptr++ = ptr[-2 - x];
        }
    }
    // fill bottom
    ptr -= 2 * src.w;
    for (int y = 0; y < bottom; y++)
    {
        const int64_t* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            *outptr++ = ptr0[left - x];
        }
        for (int x = 0; x < src.w; x++)
        {
            *outptr++ = *ptr0++;
        }
        for (int x = 0; x < right; x++)
        {
            *outptr++ = ptr0[-2 - x];
        }
        ptr -= src.w;
    }
}

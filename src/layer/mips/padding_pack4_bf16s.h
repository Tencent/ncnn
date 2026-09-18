// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void padding_constant_pack4_bf16s_msa(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int64_t v)
{
    const unsigned short* ptr = src;
    unsigned short* outptr = dst;
    int top_size = top * dst.w;
    int bottom_size = bottom * dst.w;

    // fill top
    for (int y = 0; y < top_size; y++)
    {
        *(int64_t*)outptr = v;
        outptr += 4;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            *(int64_t*)outptr = v;
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            *(int64_t*)outptr = *(const int64_t*)ptr;
            ptr += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            *(int64_t*)outptr = v;
            outptr += 4;
        }
    }
    // fill bottom
    for (int y = 0; y < bottom_size; y++)
    {
        *(int64_t*)outptr = v;
        outptr += 4;
    }
}

static void padding_replicate_pack4_bf16s_msa(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const unsigned short* ptr = src;
    unsigned short* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        const unsigned short* ptr0 = ptr;
        int64_t _p = *(const int64_t*)ptr0;
        for (int x = 0; x < left; x++)
        {
            *(int64_t*)outptr = _p;
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = *(const int64_t*)ptr0;
            *(int64_t*)outptr = _p;
            ptr0 += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            *(int64_t*)outptr = _p;
            outptr += 4;
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        int64_t _p = *(const int64_t*)ptr;
        for (int x = 0; x < left; x++)
        {
            *(int64_t*)outptr = _p;
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = *(const int64_t*)ptr;
            *(int64_t*)outptr = _p;
            ptr += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            *(int64_t*)outptr = _p;
            outptr += 4;
        }
    }
    // fill bottom
    ptr -= src.w * 4;
    for (int y = 0; y < bottom; y++)
    {
        const unsigned short* ptr0 = ptr;
        int64_t _p = *(const int64_t*)ptr0;
        for (int x = 0; x < left; x++)
        {
            *(int64_t*)outptr = _p;
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = *(const int64_t*)ptr0;
            *(int64_t*)outptr = _p;
            ptr0 += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            *(int64_t*)outptr = _p;
            outptr += 4;
        }
    }
}

static void padding_reflect_pack4_bf16s_msa(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const unsigned short* ptr = src;
    unsigned short* outptr = dst;

    // fill top
    ptr += top * src.w * 4;
    for (int y = 0; y < top; y++)
    {
        const unsigned short* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            *(int64_t*)outptr = *(const int64_t*)(ptr0 + (left - x) * 4);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            *(int64_t*)outptr = *(const int64_t*)ptr0;
            ptr0 += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            *(int64_t*)outptr = *(const int64_t*)(ptr0 - 8 - x * 4);
            outptr += 4;
        }
        ptr -= src.w * 4;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            *(int64_t*)outptr = *(const int64_t*)(ptr + (left - x) * 4);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            *(int64_t*)outptr = *(const int64_t*)ptr;
            ptr += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            *(int64_t*)outptr = *(const int64_t*)(ptr - 8 - x * 4);
            outptr += 4;
        }
    }
    // fill bottom
    ptr -= 2 * src.w * 4;
    for (int y = 0; y < bottom; y++)
    {
        const unsigned short* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            *(int64_t*)outptr = *(const int64_t*)(ptr0 + (left - x) * 4);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            *(int64_t*)outptr = *(const int64_t*)ptr0;
            ptr0 += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            *(int64_t*)outptr = *(const int64_t*)(ptr0 - 8 - x * 4);
            outptr += 4;
        }
        ptr -= src.w * 4;
    }
}

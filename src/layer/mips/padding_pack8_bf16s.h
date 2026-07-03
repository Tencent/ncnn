// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void padding_constant_pack8_bf16s_msa(const Mat& src, Mat& dst, int top, int bottom, int left, int right, const v8i16& v)
{
    const unsigned short* ptr = src;
    unsigned short* outptr = dst;
    int top_size = top * dst.w;
    int bottom_size = bottom * dst.w;

    // fill top
    for (int y = 0; y < top_size; y++)
    {
        __msa_st_h(v, outptr, 0);
        outptr += 8;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            __msa_st_h(v, outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            __msa_st_h((v8i16)__msa_ld_h(ptr, 0), outptr, 0);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __msa_st_h(v, outptr, 0);
            outptr += 8;
        }
    }
    // fill bottom
    for (int y = 0; y < bottom_size; y++)
    {
        __msa_st_h(v, outptr, 0);
        outptr += 8;
    }
}

static void padding_replicate_pack8_bf16s_msa(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const unsigned short* ptr = src;
    unsigned short* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        const unsigned short* ptr0 = ptr;
        v8i16 _p = (v8i16)__msa_ld_h(ptr0, 0);
        for (int x = 0; x < left; x++)
        {
            __msa_st_h(_p, outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = (v8i16)__msa_ld_h(ptr0, 0);
            __msa_st_h(_p, outptr, 0);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __msa_st_h(_p, outptr, 0);
            outptr += 8;
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        v8i16 _p = (v8i16)__msa_ld_h(ptr, 0);
        for (int x = 0; x < left; x++)
        {
            __msa_st_h(_p, outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = (v8i16)__msa_ld_h(ptr, 0);
            __msa_st_h(_p, outptr, 0);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __msa_st_h(_p, outptr, 0);
            outptr += 8;
        }
    }
    // fill bottom
    ptr -= src.w * 8;
    for (int y = 0; y < bottom; y++)
    {
        const unsigned short* ptr0 = ptr;
        v8i16 _p = (v8i16)__msa_ld_h(ptr0, 0);
        for (int x = 0; x < left; x++)
        {
            __msa_st_h(_p, outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = (v8i16)__msa_ld_h(ptr0, 0);
            __msa_st_h(_p, outptr, 0);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __msa_st_h(_p, outptr, 0);
            outptr += 8;
        }
    }
}

static void padding_reflect_pack8_bf16s_msa(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
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
            __msa_st_h((v8i16)__msa_ld_h(ptr0 + (left - x) * 8, 0), outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            __msa_st_h((v8i16)__msa_ld_h(ptr0, 0), outptr, 0);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __msa_st_h((v8i16)__msa_ld_h(ptr0 - 16 - x * 8, 0), outptr, 0);
            outptr += 8;
        }
        ptr -= src.w * 8;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            __msa_st_h((v8i16)__msa_ld_h(ptr + (left - x) * 8, 0), outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            __msa_st_h((v8i16)__msa_ld_h(ptr, 0), outptr, 0);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __msa_st_h((v8i16)__msa_ld_h(ptr - 16 - x * 8, 0), outptr, 0);
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
            __msa_st_h((v8i16)__msa_ld_h(ptr0 + (left - x) * 8, 0), outptr, 0);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            __msa_st_h((v8i16)__msa_ld_h(ptr0, 0), outptr, 0);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __msa_st_h((v8i16)__msa_ld_h(ptr0 - 16 - x * 8, 0), outptr, 0);
            outptr += 8;
        }
        ptr -= src.w * 8;
    }
}

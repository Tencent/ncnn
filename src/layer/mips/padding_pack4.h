// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void padding_constant_pack4_msa(const Mat& src, Mat& dst, int top, int bottom, int left, int right, v4f32 v)
{
    const float* ptr = src;
    float* outptr = dst;
    int top_size = top * dst.w;
    int bottom_size = bottom * dst.w;

    // fill top
    for (int y = 0; y < top_size; y++)
    {
        __msa_st_w((v4i32)v, outptr, 0);
        outptr += 4;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            __msa_st_w((v4i32)v, outptr, 0);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            __builtin_prefetch(ptr + 32);
            __msa_st_w(__msa_ld_w(ptr, 0), outptr, 0);
            ptr += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            __msa_st_w((v4i32)v, outptr, 0);
            outptr += 4;
        }
    }
    // fill top
    for (int y = 0; y < bottom_size; y++)
    {
        __msa_st_w((v4i32)v, outptr, 0);
        outptr += 4;
    }
}

static void padding_replicate_pack4_msa(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const float* ptr = src;
    float* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        const float* ptr0 = ptr;
        v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
        for (int x = 0; x < left; x++)
        {
            __msa_st_w((v4i32)_p, outptr, 0);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = (v4f32)__msa_ld_w(ptr0, 0);
            __msa_st_w((v4i32)_p, outptr, 0);
            ptr0 += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            __msa_st_w((v4i32)_p, outptr, 0);
            outptr += 4;
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
        for (int x = 0; x < left; x++)
        {
            __msa_st_w((v4i32)_p, outptr, 0);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = (v4f32)__msa_ld_w(ptr, 0);
            __msa_st_w((v4i32)_p, outptr, 0);
            ptr += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            __msa_st_w((v4i32)_p, outptr, 0);
            outptr += 4;
        }
    }
    // fill bottom
    ptr -= src.w * 4;
    for (int y = 0; y < bottom; y++)
    {
        const float* ptr0 = ptr;
        v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
        for (int x = 0; x < left; x++)
        {
            __msa_st_w((v4i32)_p, outptr, 0);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = (v4f32)__msa_ld_w(ptr0, 0);
            __msa_st_w((v4i32)_p, outptr, 0);
            ptr0 += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            __msa_st_w((v4i32)_p, outptr, 0);
            outptr += 4;
        }
    }
}

static void padding_reflect_pack4_msa(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const float* ptr = src;
    float* outptr = dst;

    // fill top
    ptr += top * src.w * 4;
    for (int y = 0; y < top; y++)
    {
        const float* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr0 + (left - x) * 4, 0);
            __msa_st_w((v4i32)_p, outptr, 0);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
            __msa_st_w((v4i32)_p, outptr, 0);
            ptr0 += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr0 - 8 - x * 4, 0);
            __msa_st_w((v4i32)_p, outptr, 0);
            outptr += 4;
        }
        ptr -= src.w * 4;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr + (left - x) * 4, 0);
            __msa_st_w((v4i32)_p, outptr, 0);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            __msa_st_w((v4i32)_p, outptr, 0);
            ptr += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr - 8 - x * 4, 0);
            __msa_st_w((v4i32)_p, outptr, 0);
            outptr += 4;
        }
    }
    // fill bottom
    ptr -= 2 * src.w * 4;
    for (int y = 0; y < bottom; y++)
    {
        const float* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr0 + (left - x) * 4, 0);
            __msa_st_w((v4i32)_p, outptr, 0);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
            __msa_st_w((v4i32)_p, outptr, 0);
            ptr0 += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr0 - 8 - x * 4, 0);
            __msa_st_w((v4i32)_p, outptr, 0);
            outptr += 4;
        }
        ptr -= src.w * 4;
    }
}

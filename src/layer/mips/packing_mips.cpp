// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "packing_mips.h"

#include <string.h>
#include <stdint.h>

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

namespace ncnn {

Packing_mips::Packing_mips()
{
    support_packing = true;
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Packing_mips::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

    if (elembits == 8)
        return forward_int8(bottom_blob, top_blob, opt);

#if NCNN_BF16
    if (elembits == 16)
        return forward_bf16s(bottom_blob, top_blob, opt);
#endif

    if (use_padding)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (elempack == out_elempack)
    {
        top_blob = bottom_blob;
        return 0;
    }

    bool pack1to4 = elempack == 1 && out_elempack == 4;
    bool pack4to1 = elempack == 4 && out_elempack == 1;

    if (!pack1to4 && !pack4to1)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;

    if (!use_padding)
    {
        // identity if use_padding not allowed
        if (dims == 1 && w * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 2 && h * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if ((dims == 3 || dims == 4) && channels * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    if (dims == 1)
    {
        top_blob = bottom_blob;
        top_blob.w = w * elempack / out_elempack;
        top_blob.cstep = bottom_blob.cstep * elempack / out_elempack;
        top_blob.elemsize = elemsize / elempack * out_elempack;
        top_blob.elempack = out_elempack;
        return 0;
    }

    if (dims == 2)
    {
        int outh = h * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* r0 = bottom_blob.row(i * 4);
                const float* r1 = bottom_blob.row(i * 4 + 1);
                const float* r2 = bottom_blob.row(i * 4 + 2);
                const float* r3 = bottom_blob.row(i * 4 + 3);

                float* outptr = top_blob.row(i);

                int j = 0;
#if __mips_msa
                for (; j + 3 < w; j += 4)
                {
                    // transpose 4x4
                    v4f32 _r0 = (v4f32)__msa_ld_w(r0, 0);
                    v4f32 _r1 = (v4f32)__msa_ld_w(r1, 0);
                    v4f32 _r2 = (v4f32)__msa_ld_w(r2, 0);
                    v4f32 _r3 = (v4f32)__msa_ld_w(r3, 0);

                    v4i32 _r01r = __msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
                    v4i32 _r01l = __msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
                    v4i32 _r23r = __msa_ilvr_w((v4i32)_r3, (v4i32)_r2);
                    v4i32 _r23l = __msa_ilvl_w((v4i32)_r3, (v4i32)_r2);
                    v2i64 _r0123_0 = __msa_ilvr_d((v2i64)_r23r, (v2i64)_r01r);
                    v2i64 _r0123_1 = __msa_ilvl_d((v2i64)_r23r, (v2i64)_r01r);
                    v2i64 _r0123_2 = __msa_ilvr_d((v2i64)_r23l, (v2i64)_r01l);
                    v2i64 _r0123_3 = __msa_ilvl_d((v2i64)_r23l, (v2i64)_r01l);

                    __msa_st_w((v4i32)_r0123_0, outptr, 0);
                    __msa_st_w((v4i32)_r0123_1, outptr + 4, 0);
                    __msa_st_w((v4i32)_r0123_2, outptr + 4 * 2, 0);
                    __msa_st_w((v4i32)_r0123_3, outptr + 4 * 3, 0);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
#endif // __mips_msa
                for (; j < w; j++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
            }
        }
        if (pack4to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* r0 = bottom_blob.row(i);

                float* outptr0 = top_blob.row(i * 4);
                float* outptr1 = top_blob.row(i * 4 + 1);
                float* outptr2 = top_blob.row(i * 4 + 2);
                float* outptr3 = top_blob.row(i * 4 + 3);

                int j = 0;
#if __mips_msa
                for (; j + 3 < w; j += 4)
                {
                    // transpose 4x4
                    v4f32 _r0 = (v4f32)__msa_ld_w(r0, 0);
                    v4f32 _r1 = (v4f32)__msa_ld_w(r0 + 4, 0);
                    v4f32 _r2 = (v4f32)__msa_ld_w(r0 + 4 * 2, 0);
                    v4f32 _r3 = (v4f32)__msa_ld_w(r0 + 4 * 3, 0);

                    v4i32 _r01r = __msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
                    v4i32 _r01l = __msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
                    v4i32 _r23r = __msa_ilvr_w((v4i32)_r3, (v4i32)_r2);
                    v4i32 _r23l = __msa_ilvl_w((v4i32)_r3, (v4i32)_r2);
                    v2i64 _r0123_0 = __msa_ilvr_d((v2i64)_r23r, (v2i64)_r01r);
                    v2i64 _r0123_1 = __msa_ilvl_d((v2i64)_r23r, (v2i64)_r01r);
                    v2i64 _r0123_2 = __msa_ilvr_d((v2i64)_r23l, (v2i64)_r01l);
                    v2i64 _r0123_3 = __msa_ilvl_d((v2i64)_r23l, (v2i64)_r01l);

                    __msa_st_w((v4i32)_r0123_0, outptr0, 0);
                    __msa_st_w((v4i32)_r0123_1, outptr1, 0);
                    __msa_st_w((v4i32)_r0123_2, outptr2, 0);
                    __msa_st_w((v4i32)_r0123_3, outptr3, 0);

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
#endif // __mips_msa
                for (; j < w; j++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
            }
        }

        return 0;
    }

    if (dims == 3 || dims == 4)
    {
        int size = w * h * d;
        int outc = channels * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 3)
            top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        else // if (dims == 4)
            top_blob.create(w, h, d, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* r0 = bottom_blob.channel(q * 4);
                const float* r1 = bottom_blob.channel(q * 4 + 1);
                const float* r2 = bottom_blob.channel(q * 4 + 2);
                const float* r3 = bottom_blob.channel(q * 4 + 3);

                float* outptr = top_blob.channel(q);

                int i = 0;
#if __mips_msa
                for (; i + 3 < size; i += 4)
                {
                    // transpose 4x4
                    v4f32 _r0 = (v4f32)__msa_ld_w(r0, 0);
                    v4f32 _r1 = (v4f32)__msa_ld_w(r1, 0);
                    v4f32 _r2 = (v4f32)__msa_ld_w(r2, 0);
                    v4f32 _r3 = (v4f32)__msa_ld_w(r3, 0);

                    v4i32 _r01r = __msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
                    v4i32 _r01l = __msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
                    v4i32 _r23r = __msa_ilvr_w((v4i32)_r3, (v4i32)_r2);
                    v4i32 _r23l = __msa_ilvl_w((v4i32)_r3, (v4i32)_r2);
                    v2i64 _r0123_0 = __msa_ilvr_d((v2i64)_r23r, (v2i64)_r01r);
                    v2i64 _r0123_1 = __msa_ilvl_d((v2i64)_r23r, (v2i64)_r01r);
                    v2i64 _r0123_2 = __msa_ilvr_d((v2i64)_r23l, (v2i64)_r01l);
                    v2i64 _r0123_3 = __msa_ilvl_d((v2i64)_r23l, (v2i64)_r01l);

                    __msa_st_w((v4i32)_r0123_0, outptr, 0);
                    __msa_st_w((v4i32)_r0123_1, outptr + 4, 0);
                    __msa_st_w((v4i32)_r0123_2, outptr + 4 * 2, 0);
                    __msa_st_w((v4i32)_r0123_3, outptr + 4 * 3, 0);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
#endif // __mips_msa
                for (; i < size; i++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
            }
        }
        if (pack4to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q * 4);
                float* outptr1 = top_blob.channel(q * 4 + 1);
                float* outptr2 = top_blob.channel(q * 4 + 2);
                float* outptr3 = top_blob.channel(q * 4 + 3);

                int i = 0;
#if __mips_msa
                for (; i + 3 < size; i += 4)
                {
                    // transpose 4x4
                    v4f32 _r0 = (v4f32)__msa_ld_w(r0, 0);
                    v4f32 _r1 = (v4f32)__msa_ld_w(r0 + 4, 0);
                    v4f32 _r2 = (v4f32)__msa_ld_w(r0 + 4 * 2, 0);
                    v4f32 _r3 = (v4f32)__msa_ld_w(r0 + 4 * 3, 0);

                    v4i32 _r01r = __msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
                    v4i32 _r01l = __msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
                    v4i32 _r23r = __msa_ilvr_w((v4i32)_r3, (v4i32)_r2);
                    v4i32 _r23l = __msa_ilvl_w((v4i32)_r3, (v4i32)_r2);
                    v2i64 _r0123_0 = __msa_ilvr_d((v2i64)_r23r, (v2i64)_r01r);
                    v2i64 _r0123_1 = __msa_ilvl_d((v2i64)_r23r, (v2i64)_r01r);
                    v2i64 _r0123_2 = __msa_ilvr_d((v2i64)_r23l, (v2i64)_r01l);
                    v2i64 _r0123_3 = __msa_ilvl_d((v2i64)_r23l, (v2i64)_r01l);

                    __msa_st_w((v4i32)_r0123_0, outptr0, 0);
                    __msa_st_w((v4i32)_r0123_1, outptr1, 0);
                    __msa_st_w((v4i32)_r0123_2, outptr2, 0);
                    __msa_st_w((v4i32)_r0123_3, outptr3, 0);

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
#endif // __mips_msa
                for (; i < size; i++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
            }
        }

        return 0;
    }

    return 0;
}

int Packing_mips::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (use_padding)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (elempack == out_elempack)
    {
        top_blob = bottom_blob;
        return 0;
    }

    bool pack1to8 = elempack == 1 && out_elempack == 8;
    bool pack8to1 = elempack == 8 && out_elempack == 1;

    if (!pack1to8 && !pack8to1)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;

    if (!use_padding)
    {
        // identity if use_padding not allowed
        if (dims == 1 && w * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 2 && h * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if ((dims == 3 || dims == 4) && channels * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    if (dims == 1)
    {
        top_blob = bottom_blob;
        top_blob.w = w * elempack / out_elempack;
        top_blob.cstep = bottom_blob.cstep * elempack / out_elempack;
        top_blob.elemsize = elemsize / elempack * out_elempack;
        top_blob.elempack = out_elempack;
        return 0;
    }

    if (dims == 2)
    {
        int outh = h * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const signed char* r0 = bottom_blob.row<const signed char>(i * 8);
                const signed char* r1 = bottom_blob.row<const signed char>(i * 8 + 1);
                const signed char* r2 = bottom_blob.row<const signed char>(i * 8 + 2);
                const signed char* r3 = bottom_blob.row<const signed char>(i * 8 + 3);
                const signed char* r4 = bottom_blob.row<const signed char>(i * 8 + 4);
                const signed char* r5 = bottom_blob.row<const signed char>(i * 8 + 5);
                const signed char* r6 = bottom_blob.row<const signed char>(i * 8 + 6);
                const signed char* r7 = bottom_blob.row<const signed char>(i * 8 + 7);

                signed char* outptr = top_blob.row<signed char>(i);

                int j = 0;
                for (; j < w; j++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;
                    outptr[4] = *r4++;
                    outptr[5] = *r5++;
                    outptr[6] = *r6++;
                    outptr[7] = *r7++;

                    outptr += 8;
                }
            }
        }
        if (pack8to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const signed char* r0 = bottom_blob.row<const signed char>(i);

                signed char* outptr0 = top_blob.row<signed char>(i * 8);
                signed char* outptr1 = top_blob.row<signed char>(i * 8 + 1);
                signed char* outptr2 = top_blob.row<signed char>(i * 8 + 2);
                signed char* outptr3 = top_blob.row<signed char>(i * 8 + 3);
                signed char* outptr4 = top_blob.row<signed char>(i * 8 + 4);
                signed char* outptr5 = top_blob.row<signed char>(i * 8 + 5);
                signed char* outptr6 = top_blob.row<signed char>(i * 8 + 6);
                signed char* outptr7 = top_blob.row<signed char>(i * 8 + 7);

                int j = 0;
                for (; j < w; j++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];
                    *outptr4++ = r0[4];
                    *outptr5++ = r0[5];
                    *outptr6++ = r0[6];
                    *outptr7++ = r0[7];

                    r0 += 8;
                }
            }
        }

        return 0;
    }

    if (dims == 3 || dims == 4)
    {
        int size = w * h * d;
        int outc = channels * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 3)
            top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        else // if (dims == 4)
            top_blob.create(w, h, d, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const signed char* r0 = bottom_blob.channel(q * 8);
                const signed char* r1 = bottom_blob.channel(q * 8 + 1);
                const signed char* r2 = bottom_blob.channel(q * 8 + 2);
                const signed char* r3 = bottom_blob.channel(q * 8 + 3);
                const signed char* r4 = bottom_blob.channel(q * 8 + 4);
                const signed char* r5 = bottom_blob.channel(q * 8 + 5);
                const signed char* r6 = bottom_blob.channel(q * 8 + 6);
                const signed char* r7 = bottom_blob.channel(q * 8 + 7);

                signed char* outptr = top_blob.channel(q);

                int i = 0;
                for (; i < size; i++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;
                    outptr[4] = *r4++;
                    outptr[5] = *r5++;
                    outptr[6] = *r6++;
                    outptr[7] = *r7++;

                    outptr += 8;
                }
            }
        }
        if (pack8to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const signed char* r0 = bottom_blob.channel(q);

                signed char* outptr0 = top_blob.channel(q * 8);
                signed char* outptr1 = top_blob.channel(q * 8 + 1);
                signed char* outptr2 = top_blob.channel(q * 8 + 2);
                signed char* outptr3 = top_blob.channel(q * 8 + 3);
                signed char* outptr4 = top_blob.channel(q * 8 + 4);
                signed char* outptr5 = top_blob.channel(q * 8 + 5);
                signed char* outptr6 = top_blob.channel(q * 8 + 6);
                signed char* outptr7 = top_blob.channel(q * 8 + 7);

                int i = 0;
                for (; i < size; i++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];
                    *outptr4++ = r0[4];
                    *outptr5++ = r0[5];
                    *outptr6++ = r0[6];
                    *outptr7++ = r0[7];

                    r0 += 8;
                }
            }
        }

        return 0;
    }

    return 0;
}

#if NCNN_BF16
int Packing_mips::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (use_padding)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (elempack == out_elempack)
    {
        top_blob = bottom_blob;
        return 0;
    }

    bool pack1to4 = elempack == 1 && out_elempack == 4;
    bool pack4to1 = elempack == 4 && out_elempack == 1;

    if (!pack1to4 && !pack4to1)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;

    if (!use_padding)
    {
        // identity if use_padding not allowed
        if (dims == 1 && w * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 2 && h * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if ((dims == 3 || dims == 4) && channels * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    if (dims == 1)
    {
        top_blob = bottom_blob;
        top_blob.w = w * elempack / out_elempack;
        top_blob.cstep = bottom_blob.cstep * elempack / out_elempack;
        top_blob.elemsize = elemsize / elempack * out_elempack;
        top_blob.elempack = out_elempack;
        return 0;
    }

    if (dims == 2)
    {
        int outh = h * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i * 4);
                const unsigned short* r1 = bottom_blob.row<const unsigned short>(i * 4 + 1);
                const unsigned short* r2 = bottom_blob.row<const unsigned short>(i * 4 + 2);
                const unsigned short* r3 = bottom_blob.row<const unsigned short>(i * 4 + 3);

                unsigned short* outptr = top_blob.row<unsigned short>(i);

                int j = 0;
#if __mips_msa
                for (; j + 3 < w; j += 4)
                {
                    // transpose 4x4 unsigned short
                    v8i16 _r0 = (v8i16)__msa_ld_h(r0, 0);
                    v8i16 _r1 = (v8i16)__msa_ld_h(r1, 0);
                    v8i16 _r2 = (v8i16)__msa_ld_h(r2, 0);
                    v8i16 _r3 = (v8i16)__msa_ld_h(r3, 0);

                    v8i16 _r01 = __msa_ilvr_h(_r1, _r0);
                    v8i16 _r23 = __msa_ilvr_h(_r3, _r2);

                    v4i32 _r0123l = (v4i32)__msa_ilvr_w((v4i32)_r23, (v4i32)_r01);
                    v4i32 _r0123h = (v4i32)__msa_ilvl_w((v4i32)_r23, (v4i32)_r01);

                    __msa_st_w(_r0123l, outptr, 0);
                    __msa_st_w(_r0123h, outptr + 8, 0);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
#endif // __mips_msa
                for (; j < w; j++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
            }
        }
        if (pack4to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i);

                unsigned short* outptr0 = top_blob.row<unsigned short>(i * 4);
                unsigned short* outptr1 = top_blob.row<unsigned short>(i * 4 + 1);
                unsigned short* outptr2 = top_blob.row<unsigned short>(i * 4 + 2);
                unsigned short* outptr3 = top_blob.row<unsigned short>(i * 4 + 3);

                int j = 0;
#if __mips_msa
                for (; j + 3 < w; j += 4)
                {
                    // transpose 4x4 unsigned short
                    v8i16 _r0 = (v8i16)__msa_ld_h(r0, 0);
                    v8i16 _r1 = (v8i16)__msa_ld_h(r0 + 8, 0);

                    v8i16 _r01l = __msa_ilvr_h(_r1, _r0);
                    v8i16 _r01h = __msa_ilvl_h(_r1, _r0);

                    v8i16 _r0123ll = (v8i16)__msa_ilvr_h(_r01h, _r01l);
                    v8i16 _r0123lh = (v8i16)__msa_ilvl_h(_r01h, _r01l);

                    uint64_t v0 = __msa_copy_s_d((v2i64)_r0123ll, 0);
                    uint64_t v1 = __msa_copy_s_d((v2i64)_r0123ll, 1);
                    uint64_t v2 = __msa_copy_s_d((v2i64)_r0123lh, 0);
                    uint64_t v3 = __msa_copy_s_d((v2i64)_r0123lh, 1);

                    memcpy(outptr0, &v0, 8);
                    memcpy(outptr1, &v1, 8);
                    memcpy(outptr2, &v2, 8);
                    memcpy(outptr3, &v3, 8);

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
#endif // __mips_msa
                for (; j < w; j++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
            }
        }

        return 0;
    }

    if (dims == 3 || dims == 4)
    {
        int size = w * h * d;
        int outc = channels * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 3)
            top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        else // if (dims == 4)
            top_blob.create(w, h, d, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const unsigned short* r0 = bottom_blob.channel(q * 4);
                const unsigned short* r1 = bottom_blob.channel(q * 4 + 1);
                const unsigned short* r2 = bottom_blob.channel(q * 4 + 2);
                const unsigned short* r3 = bottom_blob.channel(q * 4 + 3);

                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __mips_msa
                for (; i + 3 < size; i += 4)
                {
                    v8i16 _r0 = (v8i16)__msa_ld_h(r0, 0);
                    v8i16 _r1 = (v8i16)__msa_ld_h(r1, 0);
                    v8i16 _r2 = (v8i16)__msa_ld_h(r2, 0);
                    v8i16 _r3 = (v8i16)__msa_ld_h(r3, 0);

                    v8i16 _r01 = __msa_ilvr_h(_r1, _r0);
                    v8i16 _r23 = __msa_ilvr_h(_r3, _r2);

                    v4i32 _r0123l = (v4i32)__msa_ilvr_w((v4i32)_r23, (v4i32)_r01);
                    v4i32 _r0123h = (v4i32)__msa_ilvl_w((v4i32)_r23, (v4i32)_r01);

                    __msa_st_w(_r0123l, outptr, 0);
                    __msa_st_w(_r0123h, outptr + 8, 0);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
#endif // __mips_msa
                for (; i < size; i++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
            }
        }
        if (pack4to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* r0 = bottom_blob.channel(q);

                unsigned short* outptr0 = top_blob.channel(q * 4);
                unsigned short* outptr1 = top_blob.channel(q * 4 + 1);
                unsigned short* outptr2 = top_blob.channel(q * 4 + 2);
                unsigned short* outptr3 = top_blob.channel(q * 4 + 3);

                int i = 0;
#if __mips_msa
                for (; i + 3 < size; i += 4)
                {
                    v8i16 _r0 = (v8i16)__msa_ld_h(r0, 0);
                    v8i16 _r1 = (v8i16)__msa_ld_h(r0 + 8, 0);

                    v8i16 _r01l = __msa_ilvr_h(_r1, _r0);
                    v8i16 _r01h = __msa_ilvl_h(_r1, _r0);

                    v8i16 _r0123ll = (v8i16)__msa_ilvr_h(_r01h, _r01l);
                    v8i16 _r0123lh = (v8i16)__msa_ilvl_h(_r01h, _r01l);

                    uint64_t v0 = __msa_copy_s_d((v2i64)_r0123ll, 0);
                    uint64_t v1 = __msa_copy_s_d((v2i64)_r0123ll, 1);
                    uint64_t v2 = __msa_copy_s_d((v2i64)_r0123lh, 0);
                    uint64_t v3 = __msa_copy_s_d((v2i64)_r0123lh, 1);

                    memcpy(outptr0, &v0, 8);
                    memcpy(outptr1, &v1, 8);
                    memcpy(outptr2, &v2, 8);
                    memcpy(outptr3, &v3, 8);

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
#endif // __mips_msa
                for (; i < size; i++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
            }
        }

        return 0;
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn

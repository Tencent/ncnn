// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "concat_mips.h"

#include <stdint.h>
#include <string.h>

#if __mips_msa
#include <msa.h>
#include "mips_usability.h"
#endif // __mips_msa

namespace ncnn {

Concat_mips::Concat_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Concat_mips::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int elembits = bottom_blobs[0].elembits();
    if (elembits == 16)
        return forward_bf16s_fp16s(bottom_blobs, top_blobs, opt);

    int dims = bottom_blobs[0].dims;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        // concat vector
        // total length
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w * bottom_blob.elempack;
        }

        int out_elempack = opt.use_packing_layout && top_w % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        float* outptr = top_blob;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            const float* ptr = bottom_blob;
            memcpy(outptr, ptr, bottom_blob.w * bottom_blob.elemsize);

            outptr += bottom_blob.w * bottom_blob.elempack;
        }
    }

    if (dims == 2 && positive_axis == 0)
    {
        // concat image
        int w = bottom_blobs[0].w;

        // total height
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            elemsize = std::min(elemsize, bottom_blob.elemsize);
            elempack = std::min(elempack, bottom_blob.elempack);
            top_h += bottom_blob.h * bottom_blob.elempack;
        }

        int out_elempack = opt.use_packing_layout && top_h % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        Mat top_blob_unpacked = top_blob;
        if (elempack < out_elempack)
        {
            top_blob_unpacked.create(w, top_h / elempack, elemsize, elempack, opt.workspace_allocator);
            if (top_blob_unpacked.empty())
                return -100;
        }

        float* outptr = top_blob_unpacked;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            if (bottom_blob.elempack == 4 && elempack == 1)
            {
                for (int i = 0; i < bottom_blob.h; i++)
                {
                    const float* r0 = bottom_blob.row(i);

                    float* outptr0 = outptr;
                    float* outptr1 = outptr + w;
                    float* outptr2 = outptr + w * 2;
                    float* outptr3 = outptr + w * 3;

                    int j = 0;
#if __mips_msa
                    for (; j + 3 < w; j += 4)
                    {
                        __builtin_prefetch(r0 + 32);

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
                        __builtin_prefetch(r0 + 32);

                        *outptr0++ = r0[0];
                        *outptr1++ = r0[1];
                        *outptr2++ = r0[2];
                        *outptr3++ = r0[3];

                        r0 += 4;
                    }

                    outptr += w * 4;
                }
            }
            else // if (bottom_blob.elempack == 1 && elempack == 1) if (bottom_blob.elempack == 4 && elempack == 4)
            {
                int size = w * bottom_blob.h;

                const float* ptr = bottom_blob;
                memcpy(outptr, ptr, size * bottom_blob.elemsize);

                outptr += size * bottom_blob.elempack;
            }
        }

        // packing
        if (elempack < out_elempack)
        {
            convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        // interleave image row
        int h = bottom_blobs[0].h;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total width
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* outptr = top_blob.row(i);
            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                const float* ptr = bottom_blob.row(i);
                memcpy(outptr, ptr, bottom_blob.w * elemsize);

                outptr += bottom_blob.w * elempack;
            }
        }
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        // concat dim
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;
        int d = bottom_blobs[0].d;

        // total channels
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_channels = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            elemsize = std::min(elemsize, bottom_blob.elemsize);
            elempack = std::min(elempack, bottom_blob.elempack);
            top_channels += bottom_blob.c * bottom_blob.elempack;
        }

        int out_elempack = opt.use_packing_layout && top_channels % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, d, top_channels / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        Mat top_blob_unpacked = top_blob;
        if (elempack < out_elempack)
        {
            top_blob_unpacked.create(w, h, d, top_channels / elempack, elemsize, elempack, opt.workspace_allocator);
            if (top_blob_unpacked.empty())
                return -100;

            top_blob_unpacked.dims = dims;
        }

        int p = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            if (bottom_blob.elempack == 4 && elempack == 1)
            {
                int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

                for (int q = 0; q < bottom_blob.c; q++)
                {
                    const float* r0 = bottom_blob.channel(q);

                    float* outptr0 = top_blob_unpacked.channel(p);
                    float* outptr1 = top_blob_unpacked.channel(p + 1);
                    float* outptr2 = top_blob_unpacked.channel(p + 2);
                    float* outptr3 = top_blob_unpacked.channel(p + 3);

                    int i = 0;
#if __mips_msa
                    for (; i + 3 < size; i += 4)
                    {
                        __builtin_prefetch(r0 + 32);

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
                        __builtin_prefetch(r0 + 32);

                        *outptr0++ = r0[0];
                        *outptr1++ = r0[1];
                        *outptr2++ = r0[2];
                        *outptr3++ = r0[3];

                        r0 += 4;
                    }

                    p += 4;
                }
            }
            else // if (bottom_blob.elempack == 1 && elempack == 1) if (bottom_blob.elempack == 4 && elempack == 4)
            {
                int size = bottom_blob.total();

                const float* ptr = bottom_blob;
                float* outptr = top_blob_unpacked.channel(p);
                memcpy(outptr, ptr, size * bottom_blob.elemsize);

                p += bottom_blob.c;
            }
        }

        // packing
        if (elempack < out_elempack)
        {
            convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        // interleave dim height
        int w = bottom_blobs[0].w;
        int d = bottom_blobs[0].d;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, d, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (size_t b = 0; b < bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob = bottom_blobs[b];

                    int size = bottom_blob.w * bottom_blob.h;

                    const float* ptr = bottom_blob.channel(q).depth(i);
                    memcpy(outptr, ptr, size * elemsize);

                    outptr += size * elempack;
                }
            }
        }
    }

    if ((dims == 3 && positive_axis == 2) || (dims == 4 && positive_axis == 3))
    {
        // interleave dim width
        int h = bottom_blobs[0].h;
        int d = bottom_blobs[0].d;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total height
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, d, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    for (size_t b = 0; b < bottom_blobs.size(); b++)
                    {
                        const Mat& bottom_blob = bottom_blobs[b];

                        const float* ptr = bottom_blob.channel(q).depth(i).row(j);
                        memcpy(outptr, ptr, bottom_blob.w * elemsize);

                        outptr += bottom_blob.w * elempack;
                    }
                }
            }
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        // interleave dim depth
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total depth
        int top_d = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_d += bottom_blob.d;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, top_d, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = top_blob.channel(q);

            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

                const float* ptr = bottom_blob.channel(q);
                memcpy(outptr, ptr, size * elemsize);

                outptr += size * elempack;
            }
        }
    }

    return 0;
}

int Concat_mips::forward_bf16s_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int dims = bottom_blobs[0].dims;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        // concat vector
        // total length
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w * bottom_blob.elempack;
        }

        int out_elempack = 1;
#if __mips_msa
        if (opt.use_packing_layout)
            out_elempack = top_w % 8 == 0 ? 8 : top_w % 4 == 0 ? 4 : 1;
#endif
        size_t out_elemsize = elemsize / elempack * out_elempack;

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        unsigned short* outptr = top_blob;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            const unsigned short* ptr = bottom_blob;
            memcpy(outptr, ptr, bottom_blob.w * bottom_blob.elemsize);

            outptr += bottom_blob.w * bottom_blob.elempack;
        }
    }

    if (dims == 2 && positive_axis == 0)
    {
        // concat image
        int w = bottom_blobs[0].w;

        // total height
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            elemsize = std::min(elemsize, bottom_blob.elemsize);
            elempack = std::min(elempack, bottom_blob.elempack);
            top_h += bottom_blob.h * bottom_blob.elempack;
        }

        int out_elempack = 1;
#if __mips_msa
        if (opt.use_packing_layout)
            out_elempack = top_h % 8 == 0 ? 8 : top_h % 4 == 0 ? 4 : 1;
#endif
        size_t out_elemsize = elemsize / elempack * out_elempack;

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        Mat top_blob_unpacked = top_blob;
        if (elempack < out_elempack)
        {
            top_blob_unpacked.create(w, top_h / elempack, elemsize, elempack, opt.workspace_allocator);
            if (top_blob_unpacked.empty())
                return -100;
        }

        unsigned short* outptr = top_blob_unpacked;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

#if __mips_msa
            if (bottom_blob.elempack == 8 && elempack == 4)
            {
                for (int i = 0; i < bottom_blob.h; i++)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(i);

                    unsigned short* outptr0 = outptr;
                    unsigned short* outptr1 = outptr + w * 4;

                    int j = 0;
                    for (; j + 1 < w; j += 2)
                    {
                        __builtin_prefetch(r0 + 32);

                        v8i16 _r0 = __msa_ld_h(r0, 0);
                        v8i16 _r1 = __msa_ld_h(r0 + 8, 0);
                        __msa_st_h((v8i16)__msa_ilvr_d((v2i64)_r1, (v2i64)_r0), outptr0, 0);
                        __msa_st_h((v8i16)__msa_ilvl_d((v2i64)_r1, (v2i64)_r0), outptr1, 0);

                        outptr0 += 8;
                        outptr1 += 8;
                        r0 += 16;
                    }
                    for (; j < w; j++)
                    {
                        __builtin_prefetch(r0 + 16);

                        v8i16 _p = __msa_ld_h(r0, 0);
                        *(int64_t*)outptr0 = __msa_copy_s_d((v2i64)_p, 0);
                        *(int64_t*)outptr1 = __msa_copy_s_d((v2i64)_p, 1);

                        outptr0 += 4;
                        outptr1 += 4;
                        r0 += 8;
                    }

                    outptr += w * 8;
                }
            }
            if (bottom_blob.elempack == 8 && elempack == 1)
            {
                for (int i = 0; i < bottom_blob.h; i++)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(i);

                    unsigned short* outptr0 = outptr;
                    unsigned short* outptr1 = outptr + w;
                    unsigned short* outptr2 = outptr + w * 2;
                    unsigned short* outptr3 = outptr + w * 3;
                    unsigned short* outptr4 = outptr + w * 4;
                    unsigned short* outptr5 = outptr + w * 5;
                    unsigned short* outptr6 = outptr + w * 6;
                    unsigned short* outptr7 = outptr + w * 7;

                    int j = 0;
#if __mips_msa
                    for (; j + 7 < w; j += 8)
                    {
                        __builtin_prefetch(r0 + 64);

                        v8i16 _r0 = (v8i16)__msa_ld_h(r0, 0);
                        v8i16 _r1 = (v8i16)__msa_ld_h(r0 + 8, 0);
                        v8i16 _r2 = (v8i16)__msa_ld_h(r0 + 16, 0);
                        v8i16 _r3 = (v8i16)__msa_ld_h(r0 + 24, 0);
                        v8i16 _r4 = (v8i16)__msa_ld_h(r0 + 32, 0);
                        v8i16 _r5 = (v8i16)__msa_ld_h(r0 + 40, 0);
                        v8i16 _r6 = (v8i16)__msa_ld_h(r0 + 48, 0);
                        v8i16 _r7 = (v8i16)__msa_ld_h(r0 + 56, 0);

                        transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                        __msa_st_h(_r0, outptr0, 0);
                        __msa_st_h(_r1, outptr1, 0);
                        __msa_st_h(_r2, outptr2, 0);
                        __msa_st_h(_r3, outptr3, 0);
                        __msa_st_h(_r4, outptr4, 0);
                        __msa_st_h(_r5, outptr5, 0);
                        __msa_st_h(_r6, outptr6, 0);
                        __msa_st_h(_r7, outptr7, 0);

                        r0 += 64;
                        outptr0 += 8;
                        outptr1 += 8;
                        outptr2 += 8;
                        outptr3 += 8;
                        outptr4 += 8;
                        outptr5 += 8;
                        outptr6 += 8;
                        outptr7 += 8;
                    }
#endif // __mips_msa
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

                    outptr += w * 8;
                }
            }
#endif // __mips_msa
            if (bottom_blob.elempack == 4 && elempack == 1)
            {
                for (int i = 0; i < bottom_blob.h; i++)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(i);

                    unsigned short* outptr0 = outptr;
                    unsigned short* outptr1 = outptr + w;
                    unsigned short* outptr2 = outptr + w * 2;
                    unsigned short* outptr3 = outptr + w * 3;

                    int j = 0;
#if __mips_msa
                    for (; j + 7 < w; j += 8)
                    {
                        __builtin_prefetch(r0 + 32);

                        v8i16 _r0 = (v8i16)__msa_ld_h(r0, 0);
                        v8i16 _r1 = (v8i16)__msa_ld_h(r0 + 8, 0);
                        v8i16 _r2 = (v8i16)__msa_ld_h(r0 + 16, 0);
                        v8i16 _r3 = (v8i16)__msa_ld_h(r0 + 24, 0);

                        v8i16 _r01l = __msa_ilvr_h(_r1, _r0);
                        v8i16 _r01h = __msa_ilvl_h(_r1, _r0);
                        v8i16 _r0123ll = (v8i16)__msa_ilvr_h(_r01h, _r01l);
                        v8i16 _r0123lh = (v8i16)__msa_ilvl_h(_r01h, _r01l);

                        v8i16 _r23l = __msa_ilvr_h(_r3, _r2);
                        v8i16 _r23h = __msa_ilvl_h(_r3, _r2);
                        v8i16 _r4567ll = (v8i16)__msa_ilvr_h(_r23h, _r23l);
                        v8i16 _r4567lh = (v8i16)__msa_ilvl_h(_r23h, _r23l);

                        v8i16 _out0 = (v8i16)__msa_ilvr_d((v2i64)_r4567ll, (v2i64)_r0123ll);
                        v8i16 _out1 = (v8i16)__msa_ilvl_d((v2i64)_r4567ll, (v2i64)_r0123ll);
                        v8i16 _out2 = (v8i16)__msa_ilvr_d((v2i64)_r4567lh, (v2i64)_r0123lh);
                        v8i16 _out3 = (v8i16)__msa_ilvl_d((v2i64)_r4567lh, (v2i64)_r0123lh);

                        __msa_st_h(_out0, outptr0, 0);
                        __msa_st_h(_out1, outptr1, 0);
                        __msa_st_h(_out2, outptr2, 0);
                        __msa_st_h(_out3, outptr3, 0);

                        r0 += 32;
                        outptr0 += 8;
                        outptr1 += 8;
                        outptr2 += 8;
                        outptr3 += 8;
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

                    outptr += w * 4;
                }
            }
            if (bottom_blob.elempack == elempack) // 1-1 4-4 8-8
            {
                int size = w * bottom_blob.h;

                const unsigned short* ptr = bottom_blob;
                memcpy(outptr, ptr, size * bottom_blob.elemsize);

                outptr += size * bottom_blob.elempack;
            }
        }

        // packing
        if (elempack < out_elempack)
        {
            convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        // interleave image row
        int h = bottom_blobs[0].h;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total width
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(i);
            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                const unsigned short* ptr = bottom_blob.row<const unsigned short>(i);
                memcpy(outptr, ptr, bottom_blob.w * elemsize);

                outptr += bottom_blob.w * elempack;
            }
        }
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        // concat dim
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;
        int d = bottom_blobs[0].d;

        // total channels
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_channels = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            elemsize = std::min(elemsize, bottom_blob.elemsize);
            elempack = std::min(elempack, bottom_blob.elempack);
            top_channels += bottom_blob.c * bottom_blob.elempack;
        }

        int out_elempack = 1;
#if __mips_msa
        if (opt.use_packing_layout)
            out_elempack = top_channels % 8 == 0 ? 8 : top_channels % 4 == 0 ? 4 : 1;
#endif
        size_t out_elemsize = elemsize / elempack * out_elempack;

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, d, top_channels / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        Mat top_blob_unpacked = top_blob;
        if (elempack < out_elempack)
        {
            top_blob_unpacked.create(w, h, d, top_channels / elempack, elemsize, elempack, opt.workspace_allocator);
            if (top_blob_unpacked.empty())
                return -100;

            top_blob_unpacked.dims = dims;
        }

        int p = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

#if __mips_msa
            if (bottom_blob.elempack == 8 && elempack == 4)
            {
                int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

                for (int q = 0; q < bottom_blob.c; q++)
                {
                    const unsigned short* r0 = bottom_blob.channel(q);

                    unsigned short* outptr0 = top_blob_unpacked.channel(p);
                    unsigned short* outptr1 = top_blob_unpacked.channel(p + 1);

                    int i = 0;
                    for (; i + 1 < size; i += 2)
                    {
                        __builtin_prefetch(r0 + 32);

                        v8i16 _r0 = __msa_ld_h(r0, 0);
                        v8i16 _r1 = __msa_ld_h(r0 + 8, 0);
                        __msa_st_h((v8i16)__msa_ilvr_d((v2i64)_r1, (v2i64)_r0), outptr0, 0);
                        __msa_st_h((v8i16)__msa_ilvl_d((v2i64)_r1, (v2i64)_r0), outptr1, 0);

                        outptr0 += 8;
                        outptr1 += 8;
                        r0 += 16;
                    }
                    for (; i < size; i++)
                    {
                        __builtin_prefetch(r0 + 16);

                        v8i16 _p = __msa_ld_h(r0, 0);
                        *(int64_t*)outptr0 = __msa_copy_s_d((v2i64)_p, 0);
                        *(int64_t*)outptr1 = __msa_copy_s_d((v2i64)_p, 1);

                        outptr0 += 4;
                        outptr1 += 4;
                        r0 += 8;
                    }

                    p += 2;
                }
            }
            if (bottom_blob.elempack == 8 && elempack == 1)
            {
                int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

                for (int q = 0; q < bottom_blob.c; q++)
                {
                    const unsigned short* r0 = bottom_blob.channel(q);

                    unsigned short* outptr0 = top_blob_unpacked.channel(p);
                    unsigned short* outptr1 = top_blob_unpacked.channel(p + 1);
                    unsigned short* outptr2 = top_blob_unpacked.channel(p + 2);
                    unsigned short* outptr3 = top_blob_unpacked.channel(p + 3);
                    unsigned short* outptr4 = top_blob_unpacked.channel(p + 4);
                    unsigned short* outptr5 = top_blob_unpacked.channel(p + 5);
                    unsigned short* outptr6 = top_blob_unpacked.channel(p + 6);
                    unsigned short* outptr7 = top_blob_unpacked.channel(p + 7);

                    int i = 0;
#if __mips_msa
                    for (; i + 7 < size; i += 8)
                    {
                        __builtin_prefetch(r0 + 64);

                        v8i16 _r0 = (v8i16)__msa_ld_h(r0, 0);
                        v8i16 _r1 = (v8i16)__msa_ld_h(r0 + 8, 0);
                        v8i16 _r2 = (v8i16)__msa_ld_h(r0 + 16, 0);
                        v8i16 _r3 = (v8i16)__msa_ld_h(r0 + 24, 0);
                        v8i16 _r4 = (v8i16)__msa_ld_h(r0 + 32, 0);
                        v8i16 _r5 = (v8i16)__msa_ld_h(r0 + 40, 0);
                        v8i16 _r6 = (v8i16)__msa_ld_h(r0 + 48, 0);
                        v8i16 _r7 = (v8i16)__msa_ld_h(r0 + 56, 0);

                        transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                        __msa_st_h(_r0, outptr0, 0);
                        __msa_st_h(_r1, outptr1, 0);
                        __msa_st_h(_r2, outptr2, 0);
                        __msa_st_h(_r3, outptr3, 0);
                        __msa_st_h(_r4, outptr4, 0);
                        __msa_st_h(_r5, outptr5, 0);
                        __msa_st_h(_r6, outptr6, 0);
                        __msa_st_h(_r7, outptr7, 0);

                        r0 += 64;
                        outptr0 += 8;
                        outptr1 += 8;
                        outptr2 += 8;
                        outptr3 += 8;
                        outptr4 += 8;
                        outptr5 += 8;
                        outptr6 += 8;
                        outptr7 += 8;
                    }
#endif // __mips_msa
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

                    p += 8;
                }
            }
#endif // __mips_msa
            if (bottom_blob.elempack == 4 && elempack == 1)
            {
                int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

                for (int q = 0; q < bottom_blob.c; q++)
                {
                    const unsigned short* r0 = bottom_blob.channel(q);

                    unsigned short* outptr0 = top_blob_unpacked.channel(p);
                    unsigned short* outptr1 = top_blob_unpacked.channel(p + 1);
                    unsigned short* outptr2 = top_blob_unpacked.channel(p + 2);
                    unsigned short* outptr3 = top_blob_unpacked.channel(p + 3);

                    int i = 0;
#if __mips_msa
                    for (; i + 7 < size; i += 8)
                    {
                        __builtin_prefetch(r0 + 32);

                        v8i16 _r0 = (v8i16)__msa_ld_h(r0, 0);
                        v8i16 _r1 = (v8i16)__msa_ld_h(r0 + 8, 0);
                        v8i16 _r2 = (v8i16)__msa_ld_h(r0 + 16, 0);
                        v8i16 _r3 = (v8i16)__msa_ld_h(r0 + 24, 0);

                        v8i16 _r01l = __msa_ilvr_h(_r1, _r0);
                        v8i16 _r01h = __msa_ilvl_h(_r1, _r0);
                        v8i16 _r0123ll = (v8i16)__msa_ilvr_h(_r01h, _r01l);
                        v8i16 _r0123lh = (v8i16)__msa_ilvl_h(_r01h, _r01l);

                        v8i16 _r23l = __msa_ilvr_h(_r3, _r2);
                        v8i16 _r23h = __msa_ilvl_h(_r3, _r2);
                        v8i16 _r4567ll = (v8i16)__msa_ilvr_h(_r23h, _r23l);
                        v8i16 _r4567lh = (v8i16)__msa_ilvl_h(_r23h, _r23l);

                        v8i16 _out0 = (v8i16)__msa_ilvr_d((v2i64)_r4567ll, (v2i64)_r0123ll);
                        v8i16 _out1 = (v8i16)__msa_ilvl_d((v2i64)_r4567ll, (v2i64)_r0123ll);
                        v8i16 _out2 = (v8i16)__msa_ilvr_d((v2i64)_r4567lh, (v2i64)_r0123lh);
                        v8i16 _out3 = (v8i16)__msa_ilvl_d((v2i64)_r4567lh, (v2i64)_r0123lh);

                        __msa_st_h(_out0, outptr0, 0);
                        __msa_st_h(_out1, outptr1, 0);
                        __msa_st_h(_out2, outptr2, 0);
                        __msa_st_h(_out3, outptr3, 0);

                        r0 += 32;
                        outptr0 += 8;
                        outptr1 += 8;
                        outptr2 += 8;
                        outptr3 += 8;
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

                    p += 4;
                }
            }
            if (bottom_blob.elempack == elempack) // 1-1 4-4 8-8
            {
                int size = bottom_blob.total();

                const unsigned short* ptr = bottom_blob;
                unsigned short* outptr = top_blob_unpacked.channel(p);
                memcpy(outptr, ptr, size * bottom_blob.elemsize);

                p += bottom_blob.c;
            }
        }

        // packing
        if (elempack < out_elempack)
        {
            convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        // interleave dim height
        int w = bottom_blobs[0].w;
        int d = bottom_blobs[0].d;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, d, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* outptr = top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (size_t b = 0; b < bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob = bottom_blobs[b];

                    int size = bottom_blob.w * bottom_blob.h;

                    const unsigned short* ptr = bottom_blob.channel(q).depth(i);
                    memcpy(outptr, ptr, size * elemsize);

                    outptr += size * elempack;
                }
            }
        }
    }

    if ((dims == 3 && positive_axis == 2) || (dims == 4 && positive_axis == 3))
    {
        // interleave dim width
        int h = bottom_blobs[0].h;
        int d = bottom_blobs[0].d;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total height
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, d, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* outptr = top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    for (size_t b = 0; b < bottom_blobs.size(); b++)
                    {
                        const Mat& bottom_blob = bottom_blobs[b];

                        const unsigned short* ptr = bottom_blob.channel(q).depth(i).row<const unsigned short>(j);
                        memcpy(outptr, ptr, bottom_blob.w * elemsize);

                        outptr += bottom_blob.w * elempack;
                    }
                }
            }
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        // interleave dim depth
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total depth
        int top_d = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_d += bottom_blob.d;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, top_d, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* outptr = top_blob.channel(q);

            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

                const unsigned short* ptr = bottom_blob.channel(q);
                memcpy(outptr, ptr, size * elemsize);

                outptr += size * elempack;
            }
        }
    }

    return 0;
}

} // namespace ncnn

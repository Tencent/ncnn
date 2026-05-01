// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "concat_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "loongarch_usability.h"
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

namespace ncnn {

Concat_loongarch::Concat_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Concat_loongarch::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_BF16
    int elembits = bottom_blobs[0].elembits();
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blobs, top_blobs, opt);
#endif

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
#if __loongarch_sx
        if (opt.use_packing_layout)
        {
#if __loongarch_asx
            out_elempack = top_w % 8 == 0 ? 8 : top_w % 4 == 0 ? 4 : 1;
#else
            out_elempack = top_w % 4 == 0 ? 4 : 1;
#endif
        }
#endif // __loongarch_sx
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

        int out_elempack = 1;
#if __loongarch_sx
        if (opt.use_packing_layout)
        {
#if __loongarch_asx
            out_elempack = top_h % 8 == 0 ? 8 : top_h % 4 == 0 ? 4 : 1;
#else
            out_elempack = top_h % 4 == 0 ? 4 : 1;
#endif
        }
#endif // __loongarch_sx
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

#if __loongarch_asx
            if (bottom_blob.elempack == 8 && elempack == 4)
            {
                for (int i = 0; i < bottom_blob.h; i++)
                {
                    const float* r0 = bottom_blob.row(i);

                    float* outptr0 = outptr;
                    float* outptr1 = outptr + w * 4;

                    for (int j = 0; j < w; j++)
                    {
                        __m256 _p = (__m256)__lasx_xvld(r0, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_p), outptr0, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_p), outptr1, 0);

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
                    const float* r0 = bottom_blob.row(i);

                    float* outptr0 = outptr;
                    float* outptr1 = outptr + w;
                    float* outptr2 = outptr + w * 2;
                    float* outptr3 = outptr + w * 3;
                    float* outptr4 = outptr + w * 4;
                    float* outptr5 = outptr + w * 5;
                    float* outptr6 = outptr + w * 6;
                    float* outptr7 = outptr + w * 7;

                    int j = 0;
                    for (; j + 7 < w; j += 8)
                    {
                        __m256 _r0 = (__m256)__lasx_xvld(r0, 0);
                        __m256 _r1 = (__m256)__lasx_xvld(r0 + 8, 0);
                        __m256 _r2 = (__m256)__lasx_xvld(r0 + 16, 0);
                        __m256 _r3 = (__m256)__lasx_xvld(r0 + 24, 0);
                        __m256 _r4 = (__m256)__lasx_xvld(r0 + 32, 0);
                        __m256 _r5 = (__m256)__lasx_xvld(r0 + 40, 0);
                        __m256 _r6 = (__m256)__lasx_xvld(r0 + 48, 0);
                        __m256 _r7 = (__m256)__lasx_xvld(r0 + 56, 0);

                        transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                        __lasx_xvst((__m256i)_r0, outptr0, 0);
                        __lasx_xvst((__m256i)_r1, outptr1, 0);
                        __lasx_xvst((__m256i)_r2, outptr2, 0);
                        __lasx_xvst((__m256i)_r3, outptr3, 0);
                        __lasx_xvst((__m256i)_r4, outptr4, 0);
                        __lasx_xvst((__m256i)_r5, outptr5, 0);
                        __lasx_xvst((__m256i)_r6, outptr6, 0);
                        __lasx_xvst((__m256i)_r7, outptr7, 0);

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
#endif // __loongarch_asx
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
#if __loongarch_sx
                    for (; j + 3 < w; j += 4)
                    {
                        __m128 _r0 = (__m128)__lsx_vld(r0, 0);
                        __m128 _r1 = (__m128)__lsx_vld(r0 + 4, 0);
                        __m128 _r2 = (__m128)__lsx_vld(r0 + 8, 0);
                        __m128 _r3 = (__m128)__lsx_vld(r0 + 12, 0);

                        transpose4x4_ps(_r0, _r1, _r2, _r3);

                        __lsx_vst((__m128i)_r0, outptr0, 0);
                        __lsx_vst((__m128i)_r1, outptr1, 0);
                        __lsx_vst((__m128i)_r2, outptr2, 0);
                        __lsx_vst((__m128i)_r3, outptr3, 0);

                        r0 += 16;
                        outptr0 += 4;
                        outptr1 += 4;
                        outptr2 += 4;
                        outptr3 += 4;
                    }
#endif // __loongarch_sx
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

        int out_elempack = 1;
#if __loongarch_sx
        if (opt.use_packing_layout)
        {
#if __loongarch_asx
            out_elempack = top_channels % 8 == 0 ? 8 : top_channels % 4 == 0 ? 4 : 1;
#else
            out_elempack = top_channels % 4 == 0 ? 4 : 1;
#endif
        }
#endif // __loongarch_sx
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

#if __loongarch_asx
            if (bottom_blob.elempack == 8 && elempack == 4)
            {
                int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

                for (int q = 0; q < bottom_blob.c; q++)
                {
                    const float* r0 = bottom_blob.channel(q);

                    float* outptr0 = top_blob_unpacked.channel(p);
                    float* outptr1 = top_blob_unpacked.channel(p + 1);

                    for (int i = 0; i < size; i++)
                    {
                        __m256 _p = (__m256)__lasx_xvld(r0, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_p), outptr0, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_p), outptr1, 0);

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
                    const float* r0 = bottom_blob.channel(q);

                    float* outptr0 = top_blob_unpacked.channel(p);
                    float* outptr1 = top_blob_unpacked.channel(p + 1);
                    float* outptr2 = top_blob_unpacked.channel(p + 2);
                    float* outptr3 = top_blob_unpacked.channel(p + 3);
                    float* outptr4 = top_blob_unpacked.channel(p + 4);
                    float* outptr5 = top_blob_unpacked.channel(p + 5);
                    float* outptr6 = top_blob_unpacked.channel(p + 6);
                    float* outptr7 = top_blob_unpacked.channel(p + 7);

                    int i = 0;
                    for (; i + 7 < size; i += 8)
                    {
                        __m256 _r0 = (__m256)__lasx_xvld(r0, 0);
                        __m256 _r1 = (__m256)__lasx_xvld(r0 + 8, 0);
                        __m256 _r2 = (__m256)__lasx_xvld(r0 + 16, 0);
                        __m256 _r3 = (__m256)__lasx_xvld(r0 + 24, 0);
                        __m256 _r4 = (__m256)__lasx_xvld(r0 + 32, 0);
                        __m256 _r5 = (__m256)__lasx_xvld(r0 + 40, 0);
                        __m256 _r6 = (__m256)__lasx_xvld(r0 + 48, 0);
                        __m256 _r7 = (__m256)__lasx_xvld(r0 + 56, 0);

                        transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                        __lasx_xvst((__m256i)_r0, outptr0, 0);
                        __lasx_xvst((__m256i)_r1, outptr1, 0);
                        __lasx_xvst((__m256i)_r2, outptr2, 0);
                        __lasx_xvst((__m256i)_r3, outptr3, 0);
                        __lasx_xvst((__m256i)_r4, outptr4, 0);
                        __lasx_xvst((__m256i)_r5, outptr5, 0);
                        __lasx_xvst((__m256i)_r6, outptr6, 0);
                        __lasx_xvst((__m256i)_r7, outptr7, 0);

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
#endif // __loongarch_asx
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
#if __loongarch_sx
                    for (; i + 3 < size; i += 4)
                    {
                        __m128 _r0 = (__m128)__lsx_vld(r0, 0);
                        __m128 _r1 = (__m128)__lsx_vld(r0 + 4, 0);
                        __m128 _r2 = (__m128)__lsx_vld(r0 + 8, 0);
                        __m128 _r3 = (__m128)__lsx_vld(r0 + 12, 0);

                        transpose4x4_ps(_r0, _r1, _r2, _r3);

                        __lsx_vst((__m128i)_r0, outptr0, 0);
                        __lsx_vst((__m128i)_r1, outptr1, 0);
                        __lsx_vst((__m128i)_r2, outptr2, 0);
                        __lsx_vst((__m128i)_r3, outptr3, 0);

                        r0 += 16;
                        outptr0 += 4;
                        outptr1 += 4;
                        outptr2 += 4;
                        outptr3 += 4;
                    }
#endif // __loongarch_sx
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

int Concat_loongarch::forward_bf16s_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
#if __loongarch_sx
        if (opt.use_packing_layout)
        {
#if __loongarch_asx
            out_elempack = top_w % 8 == 0 ? 8 : top_w % 4 == 0 ? 4 : 1;
#else
            out_elempack = top_w % 4 == 0 ? 4 : 1;
#endif
        }
#endif // __loongarch_sx
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
#if __loongarch_sx
        if (opt.use_packing_layout)
        {
#if __loongarch_asx
            out_elempack = top_h % 8 == 0 ? 8 : top_h % 4 == 0 ? 4 : 1;
#else
            out_elempack = top_h % 4 == 0 ? 4 : 1;
#endif
        }
#endif // __loongarch_sx
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

#if __loongarch_asx
            if (bottom_blob.elempack == 8 && elempack == 4)
            {
                for (int i = 0; i < bottom_blob.h; i++)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(i);

                    unsigned short* outptr0 = outptr;
                    unsigned short* outptr1 = outptr + w * 4;

                    for (int j = 0; j < w; j++)
                    {
                        __m128i _p = __lsx_vld(r0, 0);
                        __lsx_vstelm_d(_p, outptr0, 0, 0);
                        __lsx_vstelm_d(_p, outptr1, 0, 1);

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
                    for (; j + 7 < w; j += 8)
                    {
                        __m128i _r0 = __lsx_vld(r0, 0);
                        __m128i _r1 = __lsx_vld(r0 + 8, 0);
                        __m128i _r2 = __lsx_vld(r0 + 16, 0);
                        __m128i _r3 = __lsx_vld(r0 + 24, 0);
                        __m128i _r4 = __lsx_vld(r0 + 32, 0);
                        __m128i _r5 = __lsx_vld(r0 + 40, 0);
                        __m128i _r6 = __lsx_vld(r0 + 48, 0);
                        __m128i _r7 = __lsx_vld(r0 + 56, 0);

                        transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                        __lsx_vst(_r0, outptr0, 0);
                        __lsx_vst(_r1, outptr1, 0);
                        __lsx_vst(_r2, outptr2, 0);
                        __lsx_vst(_r3, outptr3, 0);
                        __lsx_vst(_r4, outptr4, 0);
                        __lsx_vst(_r5, outptr5, 0);
                        __lsx_vst(_r6, outptr6, 0);
                        __lsx_vst(_r7, outptr7, 0);

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
#endif // __loongarch_asx
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
#if __loongarch_sx
                    for (; j + 3 < w; j += 4)
                    {
                        __m128i _r0 = __lsx_vld(r0, 0);
                        __m128i _r1 = __lsx_vld(r0 + 8, 0);

                        __m128i _r01l = __lsx_vilvl_h(_r1, _r0);
                        __m128i _r01h = __lsx_vilvh_h(_r1, _r0);
                        __m128i _r0123ll = __lsx_vilvl_h(_r01h, _r01l);
                        __m128i _r0123lh = __lsx_vilvh_h(_r01h, _r01l);

                        __lsx_vstelm_d(_r0123ll, outptr0, 0, 0);
                        __lsx_vstelm_d(_r0123ll, outptr1, 0, 1);
                        __lsx_vstelm_d(_r0123lh, outptr2, 0, 0);
                        __lsx_vstelm_d(_r0123lh, outptr3, 0, 1);

                        r0 += 16;
                        outptr0 += 4;
                        outptr1 += 4;
                        outptr2 += 4;
                        outptr3 += 4;
                    }
#endif // __loongarch_sx
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
#if __loongarch_sx
        if (opt.use_packing_layout)
        {
#if __loongarch_asx
            out_elempack = top_channels % 8 == 0 ? 8 : top_channels % 4 == 0 ? 4 : 1;
#else
            out_elempack = top_channels % 4 == 0 ? 4 : 1;
#endif
        }
#endif // __loongarch_sx
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

#if __loongarch_asx
            if (bottom_blob.elempack == 8 && elempack == 4)
            {
                int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

                for (int q = 0; q < bottom_blob.c; q++)
                {
                    const unsigned short* r0 = bottom_blob.channel(q);

                    unsigned short* outptr0 = top_blob_unpacked.channel(p);
                    unsigned short* outptr1 = top_blob_unpacked.channel(p + 1);

                    for (int i = 0; i < size; i++)
                    {
                        __m128i _p = __lsx_vld(r0, 0);
                        __lsx_vstelm_d(_p, outptr0, 0, 0);
                        __lsx_vstelm_d(_p, outptr1, 0, 1);

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
                    for (; i + 7 < size; i += 8)
                    {
                        __m128i _r0 = __lsx_vld(r0, 0);
                        __m128i _r1 = __lsx_vld(r0 + 8, 0);
                        __m128i _r2 = __lsx_vld(r0 + 16, 0);
                        __m128i _r3 = __lsx_vld(r0 + 24, 0);
                        __m128i _r4 = __lsx_vld(r0 + 32, 0);
                        __m128i _r5 = __lsx_vld(r0 + 40, 0);
                        __m128i _r6 = __lsx_vld(r0 + 48, 0);
                        __m128i _r7 = __lsx_vld(r0 + 56, 0);

                        transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                        __lsx_vst(_r0, outptr0, 0);
                        __lsx_vst(_r1, outptr1, 0);
                        __lsx_vst(_r2, outptr2, 0);
                        __lsx_vst(_r3, outptr3, 0);
                        __lsx_vst(_r4, outptr4, 0);
                        __lsx_vst(_r5, outptr5, 0);
                        __lsx_vst(_r6, outptr6, 0);
                        __lsx_vst(_r7, outptr7, 0);

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
#endif // __loongarch_asx
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
#if __loongarch_sx
                    for (; i + 3 < size; i += 4)
                    {
                        __m128i _r0 = __lsx_vld(r0, 0);
                        __m128i _r1 = __lsx_vld(r0 + 8, 0);

                        __m128i _r01l = __lsx_vilvl_h(_r1, _r0);
                        __m128i _r01h = __lsx_vilvh_h(_r1, _r0);
                        __m128i _r0123ll = __lsx_vilvl_h(_r01h, _r01l);
                        __m128i _r0123lh = __lsx_vilvh_h(_r01h, _r01l);

                        __lsx_vstelm_d(_r0123ll, outptr0, 0, 0);
                        __lsx_vstelm_d(_r0123ll, outptr1, 0, 1);
                        __lsx_vstelm_d(_r0123lh, outptr2, 0, 0);
                        __lsx_vstelm_d(_r0123lh, outptr3, 0, 1);

                        r0 += 16;
                        outptr0 += 4;
                        outptr1 += 4;
                        outptr2 += 4;
                        outptr3 += 4;
                    }
#endif // __loongarch_sx
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

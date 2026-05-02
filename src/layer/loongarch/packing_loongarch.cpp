// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "packing_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "loongarch_usability.h"
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

namespace ncnn {

Packing_loongarch::Packing_loongarch()
{
    support_packing = true;
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Packing_loongarch::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
#if __loongarch_sx
#if __loongarch_asx
    bool pack1to8 = elempack == 1 && out_elempack == 8;
    bool pack8to1 = elempack == 8 && out_elempack == 1;
    bool pack4to8 = elempack == 4 && out_elempack == 8;
    bool pack8to4 = elempack == 8 && out_elempack == 4;
#endif // __loongarch_asx
#endif // __loongarch_sx

    if (!pack1to4 && !pack4to1
#if __loongarch_sx
#if __loongarch_asx
            && !pack1to8 && !pack8to1 && !pack4to8 && !pack8to4
#endif // __loongarch_asx
#endif // __loongarch_sx
       )
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
#if __loongarch_sx
                for (; j + 3 < w; j += 4)
                {
                    // transpose 4x4
                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r1 = __lsx_vld(r1, 0);
                    __m128i _r2 = __lsx_vld(r2, 0);
                    __m128i _r3 = __lsx_vld(r3, 0);

                    __m128i _r01r = __lsx_vilvl_w(_r1, _r0);
                    __m128i _r01l = __lsx_vilvh_w(_r1, _r0);
                    __m128i _r23r = __lsx_vilvl_w(_r3, _r2);
                    __m128i _r23l = __lsx_vilvh_w(_r3, _r2);
                    __m128i _r0123_0 = __lsx_vilvl_d(_r23r, _r01r);
                    __m128i _r0123_1 = __lsx_vilvh_d(_r23r, _r01r);
                    __m128i _r0123_2 = __lsx_vilvl_d(_r23l, _r01l);
                    __m128i _r0123_3 = __lsx_vilvh_d(_r23l, _r01l);

                    __lsx_vst(_r0123_0, outptr, 0);
                    __lsx_vst(_r0123_1, outptr + 4, 0);
                    __lsx_vst(_r0123_2, outptr + 4 * 2, 0);
                    __lsx_vst(_r0123_3, outptr + 4 * 3, 0);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
#endif // __loongarch_sx
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
#if __loongarch_sx
                for (; j + 3 < w; j += 4)
                {
                    // transpose 4x4
                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r1 = __lsx_vld(r0 + 4, 0);
                    __m128i _r2 = __lsx_vld(r0 + 4 * 2, 0);
                    __m128i _r3 = __lsx_vld(r0 + 4 * 3, 0);

                    __m128i _r01r = __lsx_vilvl_w(_r1, _r0);
                    __m128i _r01l = __lsx_vilvh_w(_r1, _r0);
                    __m128i _r23r = __lsx_vilvl_w(_r3, _r2);
                    __m128i _r23l = __lsx_vilvh_w(_r3, _r2);
                    __m128i _r0123_0 = __lsx_vilvl_d(_r23r, _r01r);
                    __m128i _r0123_1 = __lsx_vilvh_d(_r23r, _r01r);
                    __m128i _r0123_2 = __lsx_vilvl_d(_r23l, _r01l);
                    __m128i _r0123_3 = __lsx_vilvh_d(_r23l, _r01l);

                    __lsx_vst(_r0123_0, outptr0, 0);
                    __lsx_vst(_r0123_1, outptr1, 0);
                    __lsx_vst(_r0123_2, outptr2, 0);
                    __lsx_vst(_r0123_3, outptr3, 0);

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
            }
        }
#if __loongarch_sx
#if __loongarch_asx
        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* r0 = bottom_blob.row(i * 8);
                const float* r1 = bottom_blob.row(i * 8 + 1);
                const float* r2 = bottom_blob.row(i * 8 + 2);
                const float* r3 = bottom_blob.row(i * 8 + 3);
                const float* r4 = bottom_blob.row(i * 8 + 4);
                const float* r5 = bottom_blob.row(i * 8 + 5);
                const float* r6 = bottom_blob.row(i * 8 + 6);
                const float* r7 = bottom_blob.row(i * 8 + 7);

                float* outptr = top_blob.row(i);

                int j = 0;
                for (; j + 7 < w; j += 8)
                {
                    __m256 _r0 = (__m256)__lasx_xvld(r0, 0);
                    __m256 _r1 = (__m256)__lasx_xvld(r1, 0);
                    __m256 _r2 = (__m256)__lasx_xvld(r2, 0);
                    __m256 _r3 = (__m256)__lasx_xvld(r3, 0);
                    __m256 _r4 = (__m256)__lasx_xvld(r4, 0);
                    __m256 _r5 = (__m256)__lasx_xvld(r5, 0);
                    __m256 _r6 = (__m256)__lasx_xvld(r6, 0);
                    __m256 _r7 = (__m256)__lasx_xvld(r7, 0);

                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                    __lasx_xvst((__m256i)_r0, outptr, 0);
                    __lasx_xvst((__m256i)_r1, outptr + 8, 0);
                    __lasx_xvst((__m256i)_r2, outptr + 16, 0);
                    __lasx_xvst((__m256i)_r3, outptr + 24, 0);
                    __lasx_xvst((__m256i)_r4, outptr + 32, 0);
                    __lasx_xvst((__m256i)_r5, outptr + 40, 0);
                    __lasx_xvst((__m256i)_r6, outptr + 48, 0);
                    __lasx_xvst((__m256i)_r7, outptr + 56, 0);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    r4 += 8;
                    r5 += 8;
                    r6 += 8;
                    r7 += 8;
                    outptr += 64;
                }
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
                const float* r0 = bottom_blob.row(i);

                float* outptr0 = top_blob.row(i * 8);
                float* outptr1 = top_blob.row(i * 8 + 1);
                float* outptr2 = top_blob.row(i * 8 + 2);
                float* outptr3 = top_blob.row(i * 8 + 3);
                float* outptr4 = top_blob.row(i * 8 + 4);
                float* outptr5 = top_blob.row(i * 8 + 5);
                float* outptr6 = top_blob.row(i * 8 + 6);
                float* outptr7 = top_blob.row(i * 8 + 7);

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
            }
        }
        if (pack4to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* r0 = bottom_blob.row(i * 2);
                const float* r1 = bottom_blob.row(i * 2 + 1);

                float* outptr = top_blob.row(i);

                for (int j = 0; j < w; j++)
                {
                    __m128 _r0 = (__m128)__lsx_vld(r0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(r1, 0);
                    __m256 _p = __lasx_concat_128_s(_r0, _r1);
                    __lasx_xvst((__m256i)_p, outptr, 0);

                    r0 += 4;
                    r1 += 4;
                    outptr += 8;
                }
            }
        }
        if (pack8to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* r0 = bottom_blob.row(i);

                float* outptr0 = top_blob.row(i * 2);
                float* outptr1 = top_blob.row(i * 2 + 1);

                for (int j = 0; j < w; j++)
                {
                    __m256 _p = (__m256)__lasx_xvld(r0, 0);
                    __lsx_vst(__lasx_extract_128_lo((__m256i)_p), outptr0, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_p), outptr1, 0);

                    r0 += 8;
                    outptr0 += 4;
                    outptr1 += 4;
                }
            }
        }
#endif // __loongarch_asx
#endif // __loongarch_sx

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
#if __loongarch_sx
                for (; i + 3 < size; i += 4)
                {
                    // transpose 4x4
                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r1 = __lsx_vld(r1, 0);
                    __m128i _r2 = __lsx_vld(r2, 0);
                    __m128i _r3 = __lsx_vld(r3, 0);

                    __m128i _r01r = __lsx_vilvl_w(_r1, _r0);
                    __m128i _r01l = __lsx_vilvh_w(_r1, _r0);
                    __m128i _r23r = __lsx_vilvl_w(_r3, _r2);
                    __m128i _r23l = __lsx_vilvh_w(_r3, _r2);
                    __m128i _r0123_0 = __lsx_vilvl_d(_r23r, _r01r);
                    __m128i _r0123_1 = __lsx_vilvh_d(_r23r, _r01r);
                    __m128i _r0123_2 = __lsx_vilvl_d(_r23l, _r01l);
                    __m128i _r0123_3 = __lsx_vilvh_d(_r23l, _r01l);

                    __lsx_vst(_r0123_0, outptr, 0);
                    __lsx_vst(_r0123_1, outptr + 4, 0);
                    __lsx_vst(_r0123_2, outptr + 4 * 2, 0);
                    __lsx_vst(_r0123_3, outptr + 4 * 3, 0);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
#endif // __loongarch_sx
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
#if __loongarch_sx
                for (; i + 3 < size; i += 4)
                {
                    // transpose 4x4
                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r1 = __lsx_vld(r0 + 4, 0);
                    __m128i _r2 = __lsx_vld(r0 + 4 * 2, 0);
                    __m128i _r3 = __lsx_vld(r0 + 4 * 3, 0);

                    __m128i _r01r = __lsx_vilvl_w(_r1, _r0);
                    __m128i _r01l = __lsx_vilvh_w(_r1, _r0);
                    __m128i _r23r = __lsx_vilvl_w(_r3, _r2);
                    __m128i _r23l = __lsx_vilvh_w(_r3, _r2);
                    __m128i _r0123_0 = __lsx_vilvl_d(_r23r, _r01r);
                    __m128i _r0123_1 = __lsx_vilvh_d(_r23r, _r01r);
                    __m128i _r0123_2 = __lsx_vilvl_d(_r23l, _r01l);
                    __m128i _r0123_3 = __lsx_vilvh_d(_r23l, _r01l);

                    __lsx_vst(_r0123_0, outptr0, 0);
                    __lsx_vst(_r0123_1, outptr1, 0);
                    __lsx_vst(_r0123_2, outptr2, 0);
                    __lsx_vst(_r0123_3, outptr3, 0);

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
            }
        }
#if __loongarch_sx
#if __loongarch_asx
        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* r0 = bottom_blob.channel(q * 8);
                const float* r1 = bottom_blob.channel(q * 8 + 1);
                const float* r2 = bottom_blob.channel(q * 8 + 2);
                const float* r3 = bottom_blob.channel(q * 8 + 3);
                const float* r4 = bottom_blob.channel(q * 8 + 4);
                const float* r5 = bottom_blob.channel(q * 8 + 5);
                const float* r6 = bottom_blob.channel(q * 8 + 6);
                const float* r7 = bottom_blob.channel(q * 8 + 7);

                float* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    __m256 _r0 = (__m256)__lasx_xvld(r0, 0);
                    __m256 _r1 = (__m256)__lasx_xvld(r1, 0);
                    __m256 _r2 = (__m256)__lasx_xvld(r2, 0);
                    __m256 _r3 = (__m256)__lasx_xvld(r3, 0);
                    __m256 _r4 = (__m256)__lasx_xvld(r4, 0);
                    __m256 _r5 = (__m256)__lasx_xvld(r5, 0);
                    __m256 _r6 = (__m256)__lasx_xvld(r6, 0);
                    __m256 _r7 = (__m256)__lasx_xvld(r7, 0);

                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                    __lasx_xvst((__m256i)_r0, outptr, 0);
                    __lasx_xvst((__m256i)_r1, outptr + 8, 0);
                    __lasx_xvst((__m256i)_r2, outptr + 16, 0);
                    __lasx_xvst((__m256i)_r3, outptr + 24, 0);
                    __lasx_xvst((__m256i)_r4, outptr + 32, 0);
                    __lasx_xvst((__m256i)_r5, outptr + 40, 0);
                    __lasx_xvst((__m256i)_r6, outptr + 48, 0);
                    __lasx_xvst((__m256i)_r7, outptr + 56, 0);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    r4 += 8;
                    r5 += 8;
                    r6 += 8;
                    r7 += 8;
                    outptr += 64;
                }
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
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q * 8);
                float* outptr1 = top_blob.channel(q * 8 + 1);
                float* outptr2 = top_blob.channel(q * 8 + 2);
                float* outptr3 = top_blob.channel(q * 8 + 3);
                float* outptr4 = top_blob.channel(q * 8 + 4);
                float* outptr5 = top_blob.channel(q * 8 + 5);
                float* outptr6 = top_blob.channel(q * 8 + 6);
                float* outptr7 = top_blob.channel(q * 8 + 7);

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
            }
        }
        if (pack4to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* r0 = bottom_blob.channel(q * 2);
                const float* r1 = bottom_blob.channel(q * 2 + 1);

                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    __m128 _r0 = (__m128)__lsx_vld(r0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(r1, 0);
                    __m256 _p = __lasx_concat_128_s(_r0, _r1);
                    __lasx_xvst((__m256i)_p, outptr, 0);

                    r0 += 4;
                    r1 += 4;
                    outptr += 8;
                }
            }
        }
        if (pack8to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q * 2);
                float* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    __m256 _p = (__m256)__lasx_xvld(r0, 0);
                    __lsx_vst(__lasx_extract_128_lo((__m256i)_p), outptr0, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_p), outptr1, 0);

                    r0 += 8;
                    outptr0 += 4;
                    outptr1 += 4;
                }
            }
        }
#endif // __loongarch_asx
#endif // __loongarch_sx

        return 0;
    }

    return 0;
}

int Packing_loongarch::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
int Packing_loongarch::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
#if __loongarch_sx
#if __loongarch_asx
    bool pack1to8 = elempack == 1 && out_elempack == 8;
    bool pack8to1 = elempack == 8 && out_elempack == 1;
    bool pack4to8 = elempack == 4 && out_elempack == 8;
    bool pack8to4 = elempack == 8 && out_elempack == 4;
#endif // __loongarch_asx
#endif // __loongarch_sx

    if (!pack1to4 && !pack4to1
#if __loongarch_sx
#if __loongarch_asx
            && !pack1to8 && !pack8to1 && !pack4to8 && !pack8to4
#endif // __loongarch_asx
#endif // __loongarch_sx
       )
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
#if __loongarch_sx
                for (; j + 7 < w; j += 8)
                {
                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r1 = __lsx_vld(r1, 0);
                    __m128i _r2 = __lsx_vld(r2, 0);
                    __m128i _r3 = __lsx_vld(r3, 0);

                    transpose8x4_epi16(_r0, _r1, _r2, _r3);

                    __lsx_vst(_r0, outptr, 0);
                    __lsx_vst(_r1, outptr + 8, 0);
                    __lsx_vst(_r2, outptr + 16, 0);
                    __lsx_vst(_r3, outptr + 24, 0);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    outptr += 32;
                }
#endif // __loongarch_sx
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
#if __loongarch_sx
                for (; j + 7 < w; j += 8)
                {
                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r1 = __lsx_vld(r0 + 8, 0);
                    __m128i _r2 = __lsx_vld(r0 + 16, 0);
                    __m128i _r3 = __lsx_vld(r0 + 24, 0);

                    __m128i _r01l = __lsx_vilvl_h(_r1, _r0);
                    __m128i _r01h = __lsx_vilvh_h(_r1, _r0);
                    __m128i _r0123ll = __lsx_vilvl_h(_r01h, _r01l);
                    __m128i _r0123lh = __lsx_vilvh_h(_r01h, _r01l);

                    __m128i _r23l = __lsx_vilvl_h(_r3, _r2);
                    __m128i _r23h = __lsx_vilvh_h(_r3, _r2);
                    __m128i _r4567ll = __lsx_vilvl_h(_r23h, _r23l);
                    __m128i _r4567lh = __lsx_vilvh_h(_r23h, _r23l);

                    __m128i _out0 = __lsx_vilvl_d(_r4567ll, _r0123ll);
                    __m128i _out1 = __lsx_vilvh_d(_r4567ll, _r0123ll);
                    __m128i _out2 = __lsx_vilvl_d(_r4567lh, _r0123lh);
                    __m128i _out3 = __lsx_vilvh_d(_r4567lh, _r0123lh);

                    __lsx_vst(_out0, outptr0, 0);
                    __lsx_vst(_out1, outptr1, 0);
                    __lsx_vst(_out2, outptr2, 0);
                    __lsx_vst(_out3, outptr3, 0);

                    r0 += 32;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                    outptr3 += 8;
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
            }
        }
#if __loongarch_sx
#if __loongarch_asx
        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i * 8);
                const unsigned short* r1 = bottom_blob.row<const unsigned short>(i * 8 + 1);
                const unsigned short* r2 = bottom_blob.row<const unsigned short>(i * 8 + 2);
                const unsigned short* r3 = bottom_blob.row<const unsigned short>(i * 8 + 3);
                const unsigned short* r4 = bottom_blob.row<const unsigned short>(i * 8 + 4);
                const unsigned short* r5 = bottom_blob.row<const unsigned short>(i * 8 + 5);
                const unsigned short* r6 = bottom_blob.row<const unsigned short>(i * 8 + 6);
                const unsigned short* r7 = bottom_blob.row<const unsigned short>(i * 8 + 7);

                unsigned short* outptr = top_blob.row<unsigned short>(i);

                int j = 0;
                for (; j + 7 < w; j += 8)
                {
                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r1 = __lsx_vld(r1, 0);
                    __m128i _r2 = __lsx_vld(r2, 0);
                    __m128i _r3 = __lsx_vld(r3, 0);
                    __m128i _r4 = __lsx_vld(r4, 0);
                    __m128i _r5 = __lsx_vld(r5, 0);
                    __m128i _r6 = __lsx_vld(r6, 0);
                    __m128i _r7 = __lsx_vld(r7, 0);

                    transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                    __lsx_vst(_r0, outptr, 0);
                    __lsx_vst(_r1, outptr + 8, 0);
                    __lsx_vst(_r2, outptr + 16, 0);
                    __lsx_vst(_r3, outptr + 24, 0);
                    __lsx_vst(_r4, outptr + 32, 0);
                    __lsx_vst(_r5, outptr + 40, 0);
                    __lsx_vst(_r6, outptr + 48, 0);
                    __lsx_vst(_r7, outptr + 56, 0);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    r4 += 8;
                    r5 += 8;
                    r6 += 8;
                    r7 += 8;
                    outptr += 64;
                }
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
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i);

                unsigned short* outptr0 = top_blob.row<unsigned short>(i * 8);
                unsigned short* outptr1 = top_blob.row<unsigned short>(i * 8 + 1);
                unsigned short* outptr2 = top_blob.row<unsigned short>(i * 8 + 2);
                unsigned short* outptr3 = top_blob.row<unsigned short>(i * 8 + 3);
                unsigned short* outptr4 = top_blob.row<unsigned short>(i * 8 + 4);
                unsigned short* outptr5 = top_blob.row<unsigned short>(i * 8 + 5);
                unsigned short* outptr6 = top_blob.row<unsigned short>(i * 8 + 6);
                unsigned short* outptr7 = top_blob.row<unsigned short>(i * 8 + 7);

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
            }
        }
        if (pack4to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i * 2);
                const unsigned short* r1 = bottom_blob.row<const unsigned short>(i * 2 + 1);

                unsigned short* outptr = top_blob.row<unsigned short>(i);

                int j = 0;
                for (; j + 1 < w; j += 2)
                {
                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r1 = __lsx_vld(r1, 0);
                    __lsx_vst(__lsx_vilvl_d(_r1, _r0), outptr, 0);
                    __lsx_vst(__lsx_vilvh_d(_r1, _r0), outptr + 8, 0);

                    r0 += 8;
                    r1 += 8;
                    outptr += 16;
                }
                for (; j < w; j++)
                {
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r1[0];
                    outptr[5] = r1[1];
                    outptr[6] = r1[2];
                    outptr[7] = r1[3];

                    r0 += 4;
                    r1 += 4;
                    outptr += 8;
                }
            }
        }
        if (pack8to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i);

                unsigned short* outptr0 = top_blob.row<unsigned short>(i * 2);
                unsigned short* outptr1 = top_blob.row<unsigned short>(i * 2 + 1);

                for (int j = 0; j < w; j++)
                {
                    __m128i _p = __lsx_vld(r0, 0);
                    __lsx_vstelm_d(_p, outptr0, 0, 0);
                    __lsx_vstelm_d(_p, outptr1, 0, 1);

                    r0 += 8;
                    outptr0 += 4;
                    outptr1 += 4;
                }
            }
        }
#endif // __loongarch_asx
#endif // __loongarch_sx

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
#if __loongarch_sx
                for (; i + 7 < size; i += 8)
                {
                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r1 = __lsx_vld(r1, 0);
                    __m128i _r2 = __lsx_vld(r2, 0);
                    __m128i _r3 = __lsx_vld(r3, 0);

                    transpose8x4_epi16(_r0, _r1, _r2, _r3);

                    __lsx_vst(_r0, outptr, 0);
                    __lsx_vst(_r1, outptr + 8, 0);
                    __lsx_vst(_r2, outptr + 16, 0);
                    __lsx_vst(_r3, outptr + 24, 0);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    outptr += 32;
                }
#endif // __loongarch_sx
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
#if __loongarch_sx
                for (; i + 7 < size; i += 8)
                {
                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r1 = __lsx_vld(r0 + 8, 0);
                    __m128i _r2 = __lsx_vld(r0 + 16, 0);
                    __m128i _r3 = __lsx_vld(r0 + 24, 0);

                    __m128i _r01l = __lsx_vilvl_h(_r1, _r0);
                    __m128i _r01h = __lsx_vilvh_h(_r1, _r0);
                    __m128i _r0123ll = __lsx_vilvl_h(_r01h, _r01l);
                    __m128i _r0123lh = __lsx_vilvh_h(_r01h, _r01l);

                    __m128i _r23l = __lsx_vilvl_h(_r3, _r2);
                    __m128i _r23h = __lsx_vilvh_h(_r3, _r2);
                    __m128i _r4567ll = __lsx_vilvl_h(_r23h, _r23l);
                    __m128i _r4567lh = __lsx_vilvh_h(_r23h, _r23l);

                    __m128i _out0 = __lsx_vilvl_d(_r4567ll, _r0123ll);
                    __m128i _out1 = __lsx_vilvh_d(_r4567ll, _r0123ll);
                    __m128i _out2 = __lsx_vilvl_d(_r4567lh, _r0123lh);
                    __m128i _out3 = __lsx_vilvh_d(_r4567lh, _r0123lh);

                    __lsx_vst(_out0, outptr0, 0);
                    __lsx_vst(_out1, outptr1, 0);
                    __lsx_vst(_out2, outptr2, 0);
                    __lsx_vst(_out3, outptr3, 0);

                    r0 += 32;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                    outptr3 += 8;
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
            }
        }
#if __loongarch_sx
#if __loongarch_asx
        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const unsigned short* r0 = bottom_blob.channel(q * 8);
                const unsigned short* r1 = bottom_blob.channel(q * 8 + 1);
                const unsigned short* r2 = bottom_blob.channel(q * 8 + 2);
                const unsigned short* r3 = bottom_blob.channel(q * 8 + 3);
                const unsigned short* r4 = bottom_blob.channel(q * 8 + 4);
                const unsigned short* r5 = bottom_blob.channel(q * 8 + 5);
                const unsigned short* r6 = bottom_blob.channel(q * 8 + 6);
                const unsigned short* r7 = bottom_blob.channel(q * 8 + 7);

                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r1 = __lsx_vld(r1, 0);
                    __m128i _r2 = __lsx_vld(r2, 0);
                    __m128i _r3 = __lsx_vld(r3, 0);
                    __m128i _r4 = __lsx_vld(r4, 0);
                    __m128i _r5 = __lsx_vld(r5, 0);
                    __m128i _r6 = __lsx_vld(r6, 0);
                    __m128i _r7 = __lsx_vld(r7, 0);

                    transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                    __lsx_vst(_r0, outptr, 0);
                    __lsx_vst(_r1, outptr + 8, 0);
                    __lsx_vst(_r2, outptr + 16, 0);
                    __lsx_vst(_r3, outptr + 24, 0);
                    __lsx_vst(_r4, outptr + 32, 0);
                    __lsx_vst(_r5, outptr + 40, 0);
                    __lsx_vst(_r6, outptr + 48, 0);
                    __lsx_vst(_r7, outptr + 56, 0);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    r4 += 8;
                    r5 += 8;
                    r6 += 8;
                    r7 += 8;
                    outptr += 64;
                }
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
                const unsigned short* r0 = bottom_blob.channel(q);

                unsigned short* outptr0 = top_blob.channel(q * 8);
                unsigned short* outptr1 = top_blob.channel(q * 8 + 1);
                unsigned short* outptr2 = top_blob.channel(q * 8 + 2);
                unsigned short* outptr3 = top_blob.channel(q * 8 + 3);
                unsigned short* outptr4 = top_blob.channel(q * 8 + 4);
                unsigned short* outptr5 = top_blob.channel(q * 8 + 5);
                unsigned short* outptr6 = top_blob.channel(q * 8 + 6);
                unsigned short* outptr7 = top_blob.channel(q * 8 + 7);

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
            }
        }
        if (pack4to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const unsigned short* r0 = bottom_blob.channel(q * 2);
                const unsigned short* r1 = bottom_blob.channel(q * 2 + 1);

                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 1 < size; i += 2)
                {
                    __m128i _r0 = __lsx_vld(r0, 0);
                    __m128i _r1 = __lsx_vld(r1, 0);
                    __lsx_vst(__lsx_vilvl_d(_r1, _r0), outptr, 0);
                    __lsx_vst(__lsx_vilvh_d(_r1, _r0), outptr + 8, 0);

                    r0 += 8;
                    r1 += 8;
                    outptr += 16;
                }
                for (; i < size; i++)
                {
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r1[0];
                    outptr[5] = r1[1];
                    outptr[6] = r1[2];
                    outptr[7] = r1[3];

                    r0 += 4;
                    r1 += 4;
                    outptr += 8;
                }
            }
        }
        if (pack8to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* r0 = bottom_blob.channel(q);

                unsigned short* outptr0 = top_blob.channel(q * 2);
                unsigned short* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    __m128i _p = __lsx_vld(r0, 0);
                    __lsx_vstelm_d(_p, outptr0, 0, 0);
                    __lsx_vstelm_d(_p, outptr1, 0, 1);

                    r0 += 8;
                    outptr0 += 4;
                    outptr1 += 4;
                }
            }
        }
#endif // __loongarch_asx
#endif // __loongarch_sx

        return 0;
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn

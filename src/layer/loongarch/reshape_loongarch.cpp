// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape_loongarch.h"

#if __loongarch_sx
#include "loongarch_usability.h"
#endif // __loongarch_sx

#include <string.h>

namespace ncnn {

Reshape_loongarch::Reshape_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Reshape_loongarch::forward_batch(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    int outw = w;
    int outh = h;
    int outd = d;
    int outc = c;

    if (!shape_expr.empty())
    {
        int er = eval_shape_expr(bottom_blobs, outw, outh, outd, outc);
        if (er != 0)
            return -1;
    }

    if (batch_mode == 2 && (outw == -1 || outh == -1 || outd == -1 || outc == -1))
        return -1;

    const int dims = bottom_blob.dims;
    const int elempack = bottom_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const size_t scalar_elemsize = elemsize / elempack;

    int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;
    if (batch_mode == 1)
        total *= bottom_blob.n;

    if (ndim == 0)
        return -1;

    if (ndim == 1)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outw == -1)
            outw = total;
    }
    if (ndim == 2)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (outw == -1)
            outw = total / outh;
        if (outh == -1)
            outh = total / outw;
    }
    if (ndim == 3)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (outc == 0)
            outc = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;
        if (outw == -1)
            outw = total / outc / outh;
        if (outh == -1)
            outh = total / outc / outw;
        if (outc == -1)
            outc = total / outh / outw;
    }
    if (ndim == 4)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (outc == 0)
            outc = (dims == 3 || dims == 4) ? bottom_blob.c * elempack : bottom_blob.c;
        if (outd == 0)
            outd = bottom_blob.d;
        if (outw == -1)
            outw = total / outc / outd / outh;
        if (outh == -1)
            outh = total / outc / outd / outw;
        if (outd == -1)
            outd = total / outc / outh / outw;
        if (outc == -1)
            outc = total / outd / outh / outw;
    }

    int out_elempack = 1;
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
        if (ndim == 1)
        {
#if __loongarch_asx
            out_elempack = outw % 8 == 0 ? 8 : outw % 4 == 0 ? 4 : 1;
#else
            out_elempack = outw % 4 == 0 ? 4 : 1;
#endif // __loongarch_asx
        }
        if (ndim == 2)
        {
#if __loongarch_asx
            out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
            out_elempack = outh % 4 == 0 ? 4 : 1;
#endif // __loongarch_asx
        }
        if (ndim == 3 || ndim == 4)
        {
#if __loongarch_asx
            out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#else
            out_elempack = outc % 4 == 0 ? 4 : 1;
#endif // __loongarch_asx
        }
    }
#endif // __loongarch_sx
    const size_t out_elemsize = scalar_elemsize * out_elempack;

    int shape[4] = {0, 0, 0, 0};
    if (ndim == 1)
        shape[0] = outw;
    if (ndim == 2)
    {
        shape[0] = outh;
        shape[1] = outw;
    }
    if (ndim == 3)
    {
        shape[0] = outc;
        shape[1] = outh;
        shape[2] = outw;
    }
    if (ndim == 4)
    {
        shape[0] = outc;
        shape[1] = outd;
        shape[2] = outh;
        shape[3] = outw;
    }

    if (batch_mode == 1)
    {
        if (ndim == 1)
            top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (ndim == 2)
            top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (ndim == 3)
            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (ndim == 4)
            top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);

        if (top_blob.empty())
            return -100;

        if (batch_axis == 0 && elempack == out_elempack && dims == 1 && ndim == 1 && top_blob.w == bottom_blob.w * bottom_blob.n)
        {
            const size_t size = (size_t)bottom_blob.w * elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int b = 0; b < bottom_blob.n; b++)
            {
                const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * bottom_blob.nstep * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * bottom_blob.w * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == out_elempack && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && top_blob.h == bottom_blob.h * bottom_blob.n)
        {
            const size_t size = (size_t)bottom_blob.w * bottom_blob.h * elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int b = 0; b < bottom_blob.n; b++)
            {
                const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * bottom_blob.nstep * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * bottom_blob.w * bottom_blob.h * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

#if __loongarch_sx
        if (batch_axis == 0 && elempack == 1 && out_elempack == 4 && scalar_elemsize == 4 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && top_blob.h * 4 == bottom_blob.h * bottom_blob.n)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const int y0 = i * 4;
                const int b0 = y0 / bottom_blob.h;
                const int b1 = (y0 + 1) / bottom_blob.h;
                const int b2 = (y0 + 2) / bottom_blob.h;
                const int b3 = (y0 + 3) / bottom_blob.h;
                const int r0 = y0 - b0 * bottom_blob.h;
                const int r1 = y0 + 1 - b1 * bottom_blob.h;
                const int r2 = y0 + 2 - b2 * bottom_blob.h;
                const int r3 = y0 + 3 - b3 * bottom_blob.h;

                const float* ptr0 = (const float*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)r0 * bottom_blob.w;
                const float* ptr1 = (const float*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)r1 * bottom_blob.w;
                const float* ptr2 = (const float*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)r2 * bottom_blob.w;
                const float* ptr3 = (const float*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)r3 * bottom_blob.w;
                float* outptr = top_blob.row(i);

                int j = 0;
                for (; j + 3 < bottom_blob.w; j += 4)
                {
                    __builtin_prefetch(ptr0 + 16);
                    __builtin_prefetch(ptr1 + 16);
                    __builtin_prefetch(ptr2 + 16);
                    __builtin_prefetch(ptr3 + 16);

                    __m128i _r0 = __lsx_vld(ptr0, 0);
                    __m128i _r1 = __lsx_vld(ptr1, 0);
                    __m128i _r2 = __lsx_vld(ptr2, 0);
                    __m128i _r3 = __lsx_vld(ptr3, 0);

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

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; j < bottom_blob.w; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }

#if __loongarch_asx
        if (batch_axis == 0 && elempack == 1 && out_elempack == 8 && scalar_elemsize == 4 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && top_blob.h * 8 == bottom_blob.h * bottom_blob.n)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const int y0 = i * 8;
                const int b0 = y0 / bottom_blob.h;
                const int b1 = (y0 + 1) / bottom_blob.h;
                const int b2 = (y0 + 2) / bottom_blob.h;
                const int b3 = (y0 + 3) / bottom_blob.h;
                const int b4 = (y0 + 4) / bottom_blob.h;
                const int b5 = (y0 + 5) / bottom_blob.h;
                const int b6 = (y0 + 6) / bottom_blob.h;
                const int b7 = (y0 + 7) / bottom_blob.h;
                const int r0 = y0 - b0 * bottom_blob.h;
                const int r1 = y0 + 1 - b1 * bottom_blob.h;
                const int r2 = y0 + 2 - b2 * bottom_blob.h;
                const int r3 = y0 + 3 - b3 * bottom_blob.h;
                const int r4 = y0 + 4 - b4 * bottom_blob.h;
                const int r5 = y0 + 5 - b5 * bottom_blob.h;
                const int r6 = y0 + 6 - b6 * bottom_blob.h;
                const int r7 = y0 + 7 - b7 * bottom_blob.h;

                const float* ptr0 = (const float*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)r0 * bottom_blob.w;
                const float* ptr1 = (const float*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)r1 * bottom_blob.w;
                const float* ptr2 = (const float*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)r2 * bottom_blob.w;
                const float* ptr3 = (const float*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)r3 * bottom_blob.w;
                const float* ptr4 = (const float*)bottom_blob + (size_t)b4 * bottom_blob.nstep + (size_t)r4 * bottom_blob.w;
                const float* ptr5 = (const float*)bottom_blob + (size_t)b5 * bottom_blob.nstep + (size_t)r5 * bottom_blob.w;
                const float* ptr6 = (const float*)bottom_blob + (size_t)b6 * bottom_blob.nstep + (size_t)r6 * bottom_blob.w;
                const float* ptr7 = (const float*)bottom_blob + (size_t)b7 * bottom_blob.nstep + (size_t)r7 * bottom_blob.w;
                float* outptr = top_blob.row(i);

                int j = 0;
                for (; j + 7 < bottom_blob.w; j += 8)
                {
                    __m256 _r0 = (__m256)__lasx_xvld(ptr0, 0);
                    __m256 _r1 = (__m256)__lasx_xvld(ptr1, 0);
                    __m256 _r2 = (__m256)__lasx_xvld(ptr2, 0);
                    __m256 _r3 = (__m256)__lasx_xvld(ptr3, 0);
                    __m256 _r4 = (__m256)__lasx_xvld(ptr4, 0);
                    __m256 _r5 = (__m256)__lasx_xvld(ptr5, 0);
                    __m256 _r6 = (__m256)__lasx_xvld(ptr6, 0);
                    __m256 _r7 = (__m256)__lasx_xvld(ptr7, 0);

                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                    __lasx_xvst((__m256i)_r0, outptr, 0);
                    __lasx_xvst((__m256i)_r1, outptr + 8, 0);
                    __lasx_xvst((__m256i)_r2, outptr + 16, 0);
                    __lasx_xvst((__m256i)_r3, outptr + 24, 0);
                    __lasx_xvst((__m256i)_r4, outptr + 32, 0);
                    __lasx_xvst((__m256i)_r5, outptr + 40, 0);
                    __lasx_xvst((__m256i)_r6, outptr + 48, 0);
                    __lasx_xvst((__m256i)_r7, outptr + 56, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    ptr4 += 8;
                    ptr5 += 8;
                    ptr6 += 8;
                    ptr7 += 8;
                    outptr += 64;
                }
                for (; j < bottom_blob.w; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }

            return 0;
        }
#endif // __loongarch_asx

        if (batch_axis == 0 && elempack == 1 && out_elempack == 8 && scalar_elemsize == 2 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && top_blob.h * 8 == bottom_blob.h * bottom_blob.n)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const int y0 = i * 8;
                const int b0 = y0 / bottom_blob.h;
                const int b1 = (y0 + 1) / bottom_blob.h;
                const int b2 = (y0 + 2) / bottom_blob.h;
                const int b3 = (y0 + 3) / bottom_blob.h;
                const int b4 = (y0 + 4) / bottom_blob.h;
                const int b5 = (y0 + 5) / bottom_blob.h;
                const int b6 = (y0 + 6) / bottom_blob.h;
                const int b7 = (y0 + 7) / bottom_blob.h;
                const int r0 = y0 - b0 * bottom_blob.h;
                const int r1 = y0 + 1 - b1 * bottom_blob.h;
                const int r2 = y0 + 2 - b2 * bottom_blob.h;
                const int r3 = y0 + 3 - b3 * bottom_blob.h;
                const int r4 = y0 + 4 - b4 * bottom_blob.h;
                const int r5 = y0 + 5 - b5 * bottom_blob.h;
                const int r6 = y0 + 6 - b6 * bottom_blob.h;
                const int r7 = y0 + 7 - b7 * bottom_blob.h;

                const unsigned short* ptr0 = (const unsigned short*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)r0 * bottom_blob.w;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)r1 * bottom_blob.w;
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)r2 * bottom_blob.w;
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)r3 * bottom_blob.w;
                const unsigned short* ptr4 = (const unsigned short*)bottom_blob + (size_t)b4 * bottom_blob.nstep + (size_t)r4 * bottom_blob.w;
                const unsigned short* ptr5 = (const unsigned short*)bottom_blob + (size_t)b5 * bottom_blob.nstep + (size_t)r5 * bottom_blob.w;
                const unsigned short* ptr6 = (const unsigned short*)bottom_blob + (size_t)b6 * bottom_blob.nstep + (size_t)r6 * bottom_blob.w;
                const unsigned short* ptr7 = (const unsigned short*)bottom_blob + (size_t)b7 * bottom_blob.nstep + (size_t)r7 * bottom_blob.w;
                unsigned short* outptr = top_blob.row<unsigned short>(i);

                int j = 0;
                for (; j + 7 < bottom_blob.w; j += 8)
                {
                    __m128i _r0 = __lsx_vld(ptr0, 0);
                    __m128i _r1 = __lsx_vld(ptr1, 0);
                    __m128i _r2 = __lsx_vld(ptr2, 0);
                    __m128i _r3 = __lsx_vld(ptr3, 0);
                    __m128i _r4 = __lsx_vld(ptr4, 0);
                    __m128i _r5 = __lsx_vld(ptr5, 0);
                    __m128i _r6 = __lsx_vld(ptr6, 0);
                    __m128i _r7 = __lsx_vld(ptr7, 0);

                    transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                    __lsx_vst(_r0, outptr, 0);
                    __lsx_vst(_r1, outptr + 8, 0);
                    __lsx_vst(_r2, outptr + 16, 0);
                    __lsx_vst(_r3, outptr + 24, 0);
                    __lsx_vst(_r4, outptr + 32, 0);
                    __lsx_vst(_r5, outptr + 40, 0);
                    __lsx_vst(_r6, outptr + 48, 0);
                    __lsx_vst(_r7, outptr + 56, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    ptr4 += 8;
                    ptr5 += 8;
                    ptr6 += 8;
                    ptr7 += 8;
                    outptr += 64;
                }
                for (; j < bottom_blob.w; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == 1 && out_elempack == 4 && scalar_elemsize == 2 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && top_blob.h * 4 == bottom_blob.h * bottom_blob.n)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const int y0 = i * 4;
                const int b0 = y0 / bottom_blob.h;
                const int b1 = (y0 + 1) / bottom_blob.h;
                const int b2 = (y0 + 2) / bottom_blob.h;
                const int b3 = (y0 + 3) / bottom_blob.h;
                const int r0 = y0 - b0 * bottom_blob.h;
                const int r1 = y0 + 1 - b1 * bottom_blob.h;
                const int r2 = y0 + 2 - b2 * bottom_blob.h;
                const int r3 = y0 + 3 - b3 * bottom_blob.h;

                const unsigned short* ptr0 = (const unsigned short*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)r0 * bottom_blob.w;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)r1 * bottom_blob.w;
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)r2 * bottom_blob.w;
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)r3 * bottom_blob.w;
                unsigned short* outptr = top_blob.row<unsigned short>(i);

                for (int j = 0; j < bottom_blob.w; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }
#endif // __loongarch_sx

        if (batch_axis == 0 && elempack == out_elempack && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && top_blob.c == bottom_blob.c * bottom_blob.n)
        {
            const size_t size = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d * elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < bottom_blob.n * bottom_blob.c; bq++)
            {
                const int b = bq / bottom_blob.c;
                const int q = bq - b * bottom_blob.c;

                const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)b * bottom_blob.nstep + (size_t)q * bottom_blob.cstep) * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + (size_t)bq * top_blob.cstep * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

#if __loongarch_sx
        if (batch_axis == 0 && elempack == 1 && out_elempack == 4 && scalar_elemsize == 4 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && top_blob.c * 4 == bottom_blob.c * bottom_blob.n)
        {
            const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const int p0 = q * 4;
                const int b0 = p0 / bottom_blob.c;
                const int b1 = (p0 + 1) / bottom_blob.c;
                const int b2 = (p0 + 2) / bottom_blob.c;
                const int b3 = (p0 + 3) / bottom_blob.c;
                const int q0 = p0 - b0 * bottom_blob.c;
                const int q1 = p0 + 1 - b1 * bottom_blob.c;
                const int q2 = p0 + 2 - b2 * bottom_blob.c;
                const int q3 = p0 + 3 - b3 * bottom_blob.c;

                const float* ptr0 = (const float*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)q0 * bottom_blob.cstep;
                const float* ptr1 = (const float*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)q1 * bottom_blob.cstep;
                const float* ptr2 = (const float*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)q2 * bottom_blob.cstep;
                const float* ptr3 = (const float*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)q3 * bottom_blob.cstep;
                float* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    __builtin_prefetch(ptr0 + 16);
                    __builtin_prefetch(ptr1 + 16);
                    __builtin_prefetch(ptr2 + 16);
                    __builtin_prefetch(ptr3 + 16);

                    __m128i _r0 = __lsx_vld(ptr0, 0);
                    __m128i _r1 = __lsx_vld(ptr1, 0);
                    __m128i _r2 = __lsx_vld(ptr2, 0);
                    __m128i _r3 = __lsx_vld(ptr3, 0);

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

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }

#if __loongarch_asx
        if (batch_axis == 0 && elempack == 1 && out_elempack == 8 && scalar_elemsize == 4 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && top_blob.c * 8 == bottom_blob.c * bottom_blob.n)
        {
            const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const int p0 = q * 8;
                const int b0 = p0 / bottom_blob.c;
                const int b1 = (p0 + 1) / bottom_blob.c;
                const int b2 = (p0 + 2) / bottom_blob.c;
                const int b3 = (p0 + 3) / bottom_blob.c;
                const int b4 = (p0 + 4) / bottom_blob.c;
                const int b5 = (p0 + 5) / bottom_blob.c;
                const int b6 = (p0 + 6) / bottom_blob.c;
                const int b7 = (p0 + 7) / bottom_blob.c;
                const int q0 = p0 - b0 * bottom_blob.c;
                const int q1 = p0 + 1 - b1 * bottom_blob.c;
                const int q2 = p0 + 2 - b2 * bottom_blob.c;
                const int q3 = p0 + 3 - b3 * bottom_blob.c;
                const int q4 = p0 + 4 - b4 * bottom_blob.c;
                const int q5 = p0 + 5 - b5 * bottom_blob.c;
                const int q6 = p0 + 6 - b6 * bottom_blob.c;
                const int q7 = p0 + 7 - b7 * bottom_blob.c;

                const float* ptr0 = (const float*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)q0 * bottom_blob.cstep;
                const float* ptr1 = (const float*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)q1 * bottom_blob.cstep;
                const float* ptr2 = (const float*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)q2 * bottom_blob.cstep;
                const float* ptr3 = (const float*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)q3 * bottom_blob.cstep;
                const float* ptr4 = (const float*)bottom_blob + (size_t)b4 * bottom_blob.nstep + (size_t)q4 * bottom_blob.cstep;
                const float* ptr5 = (const float*)bottom_blob + (size_t)b5 * bottom_blob.nstep + (size_t)q5 * bottom_blob.cstep;
                const float* ptr6 = (const float*)bottom_blob + (size_t)b6 * bottom_blob.nstep + (size_t)q6 * bottom_blob.cstep;
                const float* ptr7 = (const float*)bottom_blob + (size_t)b7 * bottom_blob.nstep + (size_t)q7 * bottom_blob.cstep;
                float* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    __m256 _r0 = (__m256)__lasx_xvld(ptr0, 0);
                    __m256 _r1 = (__m256)__lasx_xvld(ptr1, 0);
                    __m256 _r2 = (__m256)__lasx_xvld(ptr2, 0);
                    __m256 _r3 = (__m256)__lasx_xvld(ptr3, 0);
                    __m256 _r4 = (__m256)__lasx_xvld(ptr4, 0);
                    __m256 _r5 = (__m256)__lasx_xvld(ptr5, 0);
                    __m256 _r6 = (__m256)__lasx_xvld(ptr6, 0);
                    __m256 _r7 = (__m256)__lasx_xvld(ptr7, 0);

                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                    __lasx_xvst((__m256i)_r0, outptr, 0);
                    __lasx_xvst((__m256i)_r1, outptr + 8, 0);
                    __lasx_xvst((__m256i)_r2, outptr + 16, 0);
                    __lasx_xvst((__m256i)_r3, outptr + 24, 0);
                    __lasx_xvst((__m256i)_r4, outptr + 32, 0);
                    __lasx_xvst((__m256i)_r5, outptr + 40, 0);
                    __lasx_xvst((__m256i)_r6, outptr + 48, 0);
                    __lasx_xvst((__m256i)_r7, outptr + 56, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    ptr4 += 8;
                    ptr5 += 8;
                    ptr6 += 8;
                    ptr7 += 8;
                    outptr += 64;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }

            return 0;
        }
#endif // __loongarch_asx

        if (batch_axis == 0 && elempack == 1 && out_elempack == 8 && scalar_elemsize == 2 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && top_blob.c * 8 == bottom_blob.c * bottom_blob.n)
        {
            const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const int p0 = q * 8;
                const int b0 = p0 / bottom_blob.c;
                const int b1 = (p0 + 1) / bottom_blob.c;
                const int b2 = (p0 + 2) / bottom_blob.c;
                const int b3 = (p0 + 3) / bottom_blob.c;
                const int b4 = (p0 + 4) / bottom_blob.c;
                const int b5 = (p0 + 5) / bottom_blob.c;
                const int b6 = (p0 + 6) / bottom_blob.c;
                const int b7 = (p0 + 7) / bottom_blob.c;
                const int q0 = p0 - b0 * bottom_blob.c;
                const int q1 = p0 + 1 - b1 * bottom_blob.c;
                const int q2 = p0 + 2 - b2 * bottom_blob.c;
                const int q3 = p0 + 3 - b3 * bottom_blob.c;
                const int q4 = p0 + 4 - b4 * bottom_blob.c;
                const int q5 = p0 + 5 - b5 * bottom_blob.c;
                const int q6 = p0 + 6 - b6 * bottom_blob.c;
                const int q7 = p0 + 7 - b7 * bottom_blob.c;

                const unsigned short* ptr0 = (const unsigned short*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)q0 * bottom_blob.cstep;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)q1 * bottom_blob.cstep;
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)q2 * bottom_blob.cstep;
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)q3 * bottom_blob.cstep;
                const unsigned short* ptr4 = (const unsigned short*)bottom_blob + (size_t)b4 * bottom_blob.nstep + (size_t)q4 * bottom_blob.cstep;
                const unsigned short* ptr5 = (const unsigned short*)bottom_blob + (size_t)b5 * bottom_blob.nstep + (size_t)q5 * bottom_blob.cstep;
                const unsigned short* ptr6 = (const unsigned short*)bottom_blob + (size_t)b6 * bottom_blob.nstep + (size_t)q6 * bottom_blob.cstep;
                const unsigned short* ptr7 = (const unsigned short*)bottom_blob + (size_t)b7 * bottom_blob.nstep + (size_t)q7 * bottom_blob.cstep;
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    __m128i _r0 = __lsx_vld(ptr0, 0);
                    __m128i _r1 = __lsx_vld(ptr1, 0);
                    __m128i _r2 = __lsx_vld(ptr2, 0);
                    __m128i _r3 = __lsx_vld(ptr3, 0);
                    __m128i _r4 = __lsx_vld(ptr4, 0);
                    __m128i _r5 = __lsx_vld(ptr5, 0);
                    __m128i _r6 = __lsx_vld(ptr6, 0);
                    __m128i _r7 = __lsx_vld(ptr7, 0);

                    transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                    __lsx_vst(_r0, outptr, 0);
                    __lsx_vst(_r1, outptr + 8, 0);
                    __lsx_vst(_r2, outptr + 16, 0);
                    __lsx_vst(_r3, outptr + 24, 0);
                    __lsx_vst(_r4, outptr + 32, 0);
                    __lsx_vst(_r5, outptr + 40, 0);
                    __lsx_vst(_r6, outptr + 48, 0);
                    __lsx_vst(_r7, outptr + 56, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    ptr4 += 8;
                    ptr5 += 8;
                    ptr6 += 8;
                    ptr7 += 8;
                    outptr += 64;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == 1 && out_elempack == 4 && scalar_elemsize == 2 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && top_blob.c * 4 == bottom_blob.c * bottom_blob.n)
        {
            const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const int p0 = q * 4;
                const int b0 = p0 / bottom_blob.c;
                const int b1 = (p0 + 1) / bottom_blob.c;
                const int b2 = (p0 + 2) / bottom_blob.c;
                const int b3 = (p0 + 3) / bottom_blob.c;
                const int q0 = p0 - b0 * bottom_blob.c;
                const int q1 = p0 + 1 - b1 * bottom_blob.c;
                const int q2 = p0 + 2 - b2 * bottom_blob.c;
                const int q3 = p0 + 3 - b3 * bottom_blob.c;

                const unsigned short* ptr0 = (const unsigned short*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)q0 * bottom_blob.cstep;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)q1 * bottom_blob.cstep;
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)q2 * bottom_blob.cstep;
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)q3 * bottom_blob.cstep;
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }
#endif // __loongarch_sx

        if (batch_axis == 1 && elempack == out_elempack && dims == 2 && ndim == 3 && bottom_blob.w == top_blob.w && top_blob.h == bottom_blob.n && top_blob.c == bottom_blob.h)
        {
            const size_t size = (size_t)bottom_blob.w * elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < bottom_blob.n * bottom_blob.h; bq++)
            {
                const int b = bq / bottom_blob.h;
                const int q = bq - b * bottom_blob.h;

                const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)b * bottom_blob.nstep + (size_t)q * bottom_blob.w) * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + ((size_t)q * top_blob.cstep + (size_t)b * top_blob.w) * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 1 && elempack == out_elempack && dims == 3 && ndim == 4 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && top_blob.d == bottom_blob.n && top_blob.c == bottom_blob.c)
        {
            const size_t size = (size_t)bottom_blob.w * bottom_blob.h * elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < bottom_blob.n * bottom_blob.c; bq++)
            {
                const int b = bq / bottom_blob.c;
                const int q = bq - b * bottom_blob.c;

                const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)b * bottom_blob.nstep + (size_t)q * bottom_blob.cstep) * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + ((size_t)q * top_blob.cstep + (size_t)b * top_blob.w * top_blob.h) * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

#if __loongarch_sx
        if (batch_axis == 1 && elempack == 1 && out_elempack == 4 && scalar_elemsize == 4 && dims == 3 && ndim == 4 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && top_blob.d == bottom_blob.n && top_blob.c * 4 == bottom_blob.c)
        {
            const int size = bottom_blob.w * bottom_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < bottom_blob.n * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;

                const float* ptr0 = (const float*)bottom_blob + (size_t)b * bottom_blob.nstep + (size_t)(q * 4) * bottom_blob.cstep;
                const float* ptr1 = ptr0 + bottom_blob.cstep;
                const float* ptr2 = ptr0 + bottom_blob.cstep * 2;
                const float* ptr3 = ptr0 + bottom_blob.cstep * 3;
                float* outptr = (float*)top_blob + ((size_t)q * top_blob.cstep + (size_t)b * size) * 4;

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    __builtin_prefetch(ptr0 + 16);
                    __builtin_prefetch(ptr1 + 16);
                    __builtin_prefetch(ptr2 + 16);
                    __builtin_prefetch(ptr3 + 16);

                    __m128i _r0 = __lsx_vld(ptr0, 0);
                    __m128i _r1 = __lsx_vld(ptr1, 0);
                    __m128i _r2 = __lsx_vld(ptr2, 0);
                    __m128i _r3 = __lsx_vld(ptr3, 0);

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

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }
#endif // __loongarch_sx

        size_t prefix = 1;
        for (int i = 0; i < batch_axis; i++)
            prefix *= shape[i];

        size_t suffix = 1;
        if (batch_axis == 0)
            suffix = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;
        else
        {
            for (int i = batch_axis + 1; i < ndim; i++)
                suffix *= shape[i];
        }

        const size_t bottom_channel_size = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d;
        const size_t top_channel_size = (size_t)top_blob.w * top_blob.h * top_blob.d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < (int)prefix; pp++)
        {
            for (int b = 0; b < bottom_blob.n; b++)
            {
                for (size_t s = 0; s < suffix; s++)
                {
                    const size_t srci = batch_axis == 0 ? s : (size_t)pp * suffix + s;
                    const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * bottom_blob.nstep * elemsize;
                    if (dims == 1)
                    {
                        const int x = srci / elempack;
                        const int k = srci % elempack;
                        ptr += (size_t)x * elemsize + k * scalar_elemsize;
                    }
                    else if (dims == 2)
                    {
                        const int x = srci % bottom_blob.w;
                        const int y = srci / bottom_blob.w;
                        const int y0 = y / elempack;
                        const int k = y % elempack;
                        ptr += ((size_t)y0 * bottom_blob.w + x) * elemsize + k * scalar_elemsize;
                    }
                    else
                    {
                        const int q = srci / bottom_channel_size;
                        const size_t r = srci - (size_t)q * bottom_channel_size;
                        const int q0 = q / elempack;
                        const int k = q % elempack;
                        ptr += ((size_t)q0 * bottom_blob.cstep + r) * elemsize + k * scalar_elemsize;
                    }

                    const size_t dsti = batch_axis == 0 ? (size_t)b * suffix + s : ((size_t)pp * bottom_blob.n + b) * suffix + s;
                    unsigned char* outptr = (unsigned char*)top_blob;
                    if (top_blob.dims == 1)
                    {
                        const int x = dsti / out_elempack;
                        const int k = dsti % out_elempack;
                        outptr += (size_t)x * out_elemsize + k * scalar_elemsize;
                    }
                    else if (top_blob.dims == 2)
                    {
                        const int x = dsti % top_blob.w;
                        const int y = dsti / top_blob.w;
                        const int y0 = y / out_elempack;
                        const int k = y % out_elempack;
                        outptr += ((size_t)y0 * top_blob.w + x) * out_elemsize + k * scalar_elemsize;
                    }
                    else
                    {
                        const int q = dsti / top_channel_size;
                        const size_t r = dsti - (size_t)q * top_channel_size;
                        const int q0 = q / out_elempack;
                        const int k = q % out_elempack;
                        outptr += ((size_t)q0 * top_blob.cstep + r) * out_elemsize + k * scalar_elemsize;
                    }

                    memcpy(outptr, ptr, scalar_elemsize);
                }
            }
        }

        return 0;
    }

    if (batch_mode == 2)
    {
        if (bottom_blob.n != 1)
            return -1;

        size_t out_total = outw;
        if (ndim == 2)
            out_total *= outh;
        if (ndim == 3)
            out_total *= (size_t)outh * outc;
        if (ndim == 4)
            out_total *= (size_t)outh * outd * outc;

        if (out_total == 0)
            return -1;

        const size_t bottom_total = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;
        const int batch = bottom_total / out_total;
        if ((size_t)batch * out_total != bottom_total)
            return -1;

        if (ndim == 1)
            top_blob.create_batch(outw / out_elempack, batch, out_elemsize, out_elempack, opt.blob_allocator);
        if (ndim == 2)
            top_blob.create_batch(outw, outh / out_elempack, batch, out_elemsize, out_elempack, opt.blob_allocator);
        if (ndim == 3)
            top_blob.create_batch(outw, outh, outc / out_elempack, batch, out_elemsize, out_elempack, opt.blob_allocator);
        if (ndim == 4)
            top_blob.create_batch(outw, outh, outd, outc / out_elempack, batch, out_elemsize, out_elempack, opt.blob_allocator);

        if (top_blob.empty())
            return -100;

        if (batch_axis == 0 && elempack == out_elempack && dims == 1 && ndim == 1 && bottom_blob.w == top_blob.w * batch)
        {
            const size_t size = (size_t)top_blob.w * out_elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int b = 0; b < batch; b++)
            {
                const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * top_blob.w * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * top_blob.nstep * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == out_elempack && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h * batch)
        {
            const size_t size = (size_t)top_blob.w * top_blob.h * out_elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int b = 0; b < batch; b++)
            {
                const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * top_blob.w * top_blob.h * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * top_blob.nstep * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

#if __loongarch_sx
        if (batch_axis == 0 && elempack == 1 && out_elempack == 4 && scalar_elemsize == 4 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h * batch * 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bi = 0; bi < batch * top_blob.h; bi++)
            {
                const int b = bi / top_blob.h;
                const int i = bi - b * top_blob.h;
                const int y = bi * 4;

                const float* ptr0 = bottom_blob.row(y);
                const float* ptr1 = bottom_blob.row(y + 1);
                const float* ptr2 = bottom_blob.row(y + 2);
                const float* ptr3 = bottom_blob.row(y + 3);
                float* outptr = (float*)top_blob + ((size_t)b * top_blob.nstep + (size_t)i * top_blob.w) * 4;

                int j = 0;
                for (; j + 3 < bottom_blob.w; j += 4)
                {
                    __m128i _r0 = __lsx_vld(ptr0, 0);
                    __m128i _r1 = __lsx_vld(ptr1, 0);
                    __m128i _r2 = __lsx_vld(ptr2, 0);
                    __m128i _r3 = __lsx_vld(ptr3, 0);

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
                    __lsx_vst(_r0123_2, outptr + 8, 0);
                    __lsx_vst(_r0123_3, outptr + 12, 0);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; j < bottom_blob.w; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }

#if __loongarch_asx
        if (batch_axis == 0 && elempack == 1 && out_elempack == 8 && scalar_elemsize == 4 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h * batch * 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bi = 0; bi < batch * top_blob.h; bi++)
            {
                const int b = bi / top_blob.h;
                const int i = bi - b * top_blob.h;
                const int y = bi * 8;

                const float* ptr0 = bottom_blob.row(y);
                const float* ptr1 = bottom_blob.row(y + 1);
                const float* ptr2 = bottom_blob.row(y + 2);
                const float* ptr3 = bottom_blob.row(y + 3);
                const float* ptr4 = bottom_blob.row(y + 4);
                const float* ptr5 = bottom_blob.row(y + 5);
                const float* ptr6 = bottom_blob.row(y + 6);
                const float* ptr7 = bottom_blob.row(y + 7);
                float* outptr = (float*)top_blob + ((size_t)b * top_blob.nstep + (size_t)i * top_blob.w) * 8;

                int j = 0;
                for (; j + 7 < bottom_blob.w; j += 8)
                {
                    __m256 _r0 = (__m256)__lasx_xvld(ptr0, 0);
                    __m256 _r1 = (__m256)__lasx_xvld(ptr1, 0);
                    __m256 _r2 = (__m256)__lasx_xvld(ptr2, 0);
                    __m256 _r3 = (__m256)__lasx_xvld(ptr3, 0);
                    __m256 _r4 = (__m256)__lasx_xvld(ptr4, 0);
                    __m256 _r5 = (__m256)__lasx_xvld(ptr5, 0);
                    __m256 _r6 = (__m256)__lasx_xvld(ptr6, 0);
                    __m256 _r7 = (__m256)__lasx_xvld(ptr7, 0);

                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                    __lasx_xvst((__m256i)_r0, outptr, 0);
                    __lasx_xvst((__m256i)_r1, outptr + 8, 0);
                    __lasx_xvst((__m256i)_r2, outptr + 16, 0);
                    __lasx_xvst((__m256i)_r3, outptr + 24, 0);
                    __lasx_xvst((__m256i)_r4, outptr + 32, 0);
                    __lasx_xvst((__m256i)_r5, outptr + 40, 0);
                    __lasx_xvst((__m256i)_r6, outptr + 48, 0);
                    __lasx_xvst((__m256i)_r7, outptr + 56, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    ptr4 += 8;
                    ptr5 += 8;
                    ptr6 += 8;
                    ptr7 += 8;
                    outptr += 64;
                }
                for (; j < bottom_blob.w; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == 8 && out_elempack == 1 && scalar_elemsize == 4 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h * 8 == top_blob.h * batch)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < bottom_blob.h; i++)
            {
                const int y0 = i * 8;
                const int b0 = y0 / top_blob.h;
                const int b1 = (y0 + 1) / top_blob.h;
                const int b2 = (y0 + 2) / top_blob.h;
                const int b3 = (y0 + 3) / top_blob.h;
                const int b4 = (y0 + 4) / top_blob.h;
                const int b5 = (y0 + 5) / top_blob.h;
                const int b6 = (y0 + 6) / top_blob.h;
                const int b7 = (y0 + 7) / top_blob.h;
                const int r0 = y0 - b0 * top_blob.h;
                const int r1 = y0 + 1 - b1 * top_blob.h;
                const int r2 = y0 + 2 - b2 * top_blob.h;
                const int r3 = y0 + 3 - b3 * top_blob.h;
                const int r4 = y0 + 4 - b4 * top_blob.h;
                const int r5 = y0 + 5 - b5 * top_blob.h;
                const int r6 = y0 + 6 - b6 * top_blob.h;
                const int r7 = y0 + 7 - b7 * top_blob.h;

                const float* ptr = bottom_blob.row(i);
                float* outptr0 = (float*)top_blob + (size_t)b0 * top_blob.nstep + (size_t)r0 * top_blob.w;
                float* outptr1 = (float*)top_blob + (size_t)b1 * top_blob.nstep + (size_t)r1 * top_blob.w;
                float* outptr2 = (float*)top_blob + (size_t)b2 * top_blob.nstep + (size_t)r2 * top_blob.w;
                float* outptr3 = (float*)top_blob + (size_t)b3 * top_blob.nstep + (size_t)r3 * top_blob.w;
                float* outptr4 = (float*)top_blob + (size_t)b4 * top_blob.nstep + (size_t)r4 * top_blob.w;
                float* outptr5 = (float*)top_blob + (size_t)b5 * top_blob.nstep + (size_t)r5 * top_blob.w;
                float* outptr6 = (float*)top_blob + (size_t)b6 * top_blob.nstep + (size_t)r6 * top_blob.w;
                float* outptr7 = (float*)top_blob + (size_t)b7 * top_blob.nstep + (size_t)r7 * top_blob.w;

                int j = 0;
                for (; j + 7 < bottom_blob.w; j += 8)
                {
                    __m256 _r0 = (__m256)__lasx_xvld(ptr, 0);
                    __m256 _r1 = (__m256)__lasx_xvld(ptr + 8, 0);
                    __m256 _r2 = (__m256)__lasx_xvld(ptr + 16, 0);
                    __m256 _r3 = (__m256)__lasx_xvld(ptr + 24, 0);
                    __m256 _r4 = (__m256)__lasx_xvld(ptr + 32, 0);
                    __m256 _r5 = (__m256)__lasx_xvld(ptr + 40, 0);
                    __m256 _r6 = (__m256)__lasx_xvld(ptr + 48, 0);
                    __m256 _r7 = (__m256)__lasx_xvld(ptr + 56, 0);

                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                    __lasx_xvst((__m256i)_r0, outptr0, 0);
                    __lasx_xvst((__m256i)_r1, outptr1, 0);
                    __lasx_xvst((__m256i)_r2, outptr2, 0);
                    __lasx_xvst((__m256i)_r3, outptr3, 0);
                    __lasx_xvst((__m256i)_r4, outptr4, 0);
                    __lasx_xvst((__m256i)_r5, outptr5, 0);
                    __lasx_xvst((__m256i)_r6, outptr6, 0);
                    __lasx_xvst((__m256i)_r7, outptr7, 0);

                    ptr += 64;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                    outptr3 += 8;
                    outptr4 += 8;
                    outptr5 += 8;
                    outptr6 += 8;
                    outptr7 += 8;
                }
                for (; j < bottom_blob.w; j++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];
                    *outptr4++ = ptr[4];
                    *outptr5++ = ptr[5];
                    *outptr6++ = ptr[6];
                    *outptr7++ = ptr[7];

                    ptr += 8;
                }
            }

            return 0;
        }
#endif // __loongarch_asx

        if (batch_axis == 0 && elempack == 4 && out_elempack == 1 && scalar_elemsize == 4 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h * 4 == top_blob.h * batch)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < bottom_blob.h; i++)
            {
                const int y0 = i * 4;
                const int b0 = y0 / top_blob.h;
                const int b1 = (y0 + 1) / top_blob.h;
                const int b2 = (y0 + 2) / top_blob.h;
                const int b3 = (y0 + 3) / top_blob.h;
                const int r0 = y0 - b0 * top_blob.h;
                const int r1 = y0 + 1 - b1 * top_blob.h;
                const int r2 = y0 + 2 - b2 * top_blob.h;
                const int r3 = y0 + 3 - b3 * top_blob.h;

                const float* ptr = bottom_blob.row(i);
                float* outptr0 = (float*)top_blob + (size_t)b0 * top_blob.nstep + (size_t)r0 * top_blob.w;
                float* outptr1 = (float*)top_blob + (size_t)b1 * top_blob.nstep + (size_t)r1 * top_blob.w;
                float* outptr2 = (float*)top_blob + (size_t)b2 * top_blob.nstep + (size_t)r2 * top_blob.w;
                float* outptr3 = (float*)top_blob + (size_t)b3 * top_blob.nstep + (size_t)r3 * top_blob.w;

                int j = 0;
                for (; j + 3 < bottom_blob.w; j += 4)
                {
                    __builtin_prefetch(ptr + 32);

                    __m128i _r0 = __lsx_vld(ptr, 0);
                    __m128i _r1 = __lsx_vld(ptr + 4, 0);
                    __m128i _r2 = __lsx_vld(ptr + 4 * 2, 0);
                    __m128i _r3 = __lsx_vld(ptr + 4 * 3, 0);

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

                    ptr += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
                for (; j < bottom_blob.w; j++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }

            return 0;
        }
#endif // __loongarch_sx

        if (batch_axis == 0 && elempack == out_elempack && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c == top_blob.c * batch)
        {
            const size_t size = (size_t)top_blob.w * top_blob.h * top_blob.d * out_elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;

                const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)bq * bottom_blob.cstep * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

#if __loongarch_sx
        if (batch_axis == 0 && elempack == 1 && out_elempack == 4 && scalar_elemsize == 4 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c == top_blob.c * batch * 4)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;
                const int sq = b * top_blob.c * 4 + q * 4;

                const float* ptr0 = bottom_blob.channel(sq);
                const float* ptr1 = bottom_blob.channel(sq + 1);
                const float* ptr2 = bottom_blob.channel(sq + 2);
                const float* ptr3 = bottom_blob.channel(sq + 3);
                float* outptr = (float*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * 4;

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    __m128i _r0 = __lsx_vld(ptr0, 0);
                    __m128i _r1 = __lsx_vld(ptr1, 0);
                    __m128i _r2 = __lsx_vld(ptr2, 0);
                    __m128i _r3 = __lsx_vld(ptr3, 0);

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
                    __lsx_vst(_r0123_2, outptr + 8, 0);
                    __lsx_vst(_r0123_3, outptr + 12, 0);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }

#if __loongarch_asx
        if (batch_axis == 0 && elempack == 1 && out_elempack == 8 && scalar_elemsize == 4 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c == top_blob.c * batch * 8)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;
                const int sq = b * top_blob.c * 8 + q * 8;

                const float* ptr0 = bottom_blob.channel(sq);
                const float* ptr1 = bottom_blob.channel(sq + 1);
                const float* ptr2 = bottom_blob.channel(sq + 2);
                const float* ptr3 = bottom_blob.channel(sq + 3);
                const float* ptr4 = bottom_blob.channel(sq + 4);
                const float* ptr5 = bottom_blob.channel(sq + 5);
                const float* ptr6 = bottom_blob.channel(sq + 6);
                const float* ptr7 = bottom_blob.channel(sq + 7);
                float* outptr = (float*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * 8;

                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    __m256 _r0 = (__m256)__lasx_xvld(ptr0, 0);
                    __m256 _r1 = (__m256)__lasx_xvld(ptr1, 0);
                    __m256 _r2 = (__m256)__lasx_xvld(ptr2, 0);
                    __m256 _r3 = (__m256)__lasx_xvld(ptr3, 0);
                    __m256 _r4 = (__m256)__lasx_xvld(ptr4, 0);
                    __m256 _r5 = (__m256)__lasx_xvld(ptr5, 0);
                    __m256 _r6 = (__m256)__lasx_xvld(ptr6, 0);
                    __m256 _r7 = (__m256)__lasx_xvld(ptr7, 0);

                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                    __lasx_xvst((__m256i)_r0, outptr, 0);
                    __lasx_xvst((__m256i)_r1, outptr + 8, 0);
                    __lasx_xvst((__m256i)_r2, outptr + 16, 0);
                    __lasx_xvst((__m256i)_r3, outptr + 24, 0);
                    __lasx_xvst((__m256i)_r4, outptr + 32, 0);
                    __lasx_xvst((__m256i)_r5, outptr + 40, 0);
                    __lasx_xvst((__m256i)_r6, outptr + 48, 0);
                    __lasx_xvst((__m256i)_r7, outptr + 56, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    ptr4 += 8;
                    ptr5 += 8;
                    ptr6 += 8;
                    ptr7 += 8;
                    outptr += 64;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == 8 && out_elempack == 1 && scalar_elemsize == 4 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c * 8 == top_blob.c * batch)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < bottom_blob.c; q++)
            {
                const int p0 = q * 8;
                const int b0 = p0 / top_blob.c;
                const int b1 = (p0 + 1) / top_blob.c;
                const int b2 = (p0 + 2) / top_blob.c;
                const int b3 = (p0 + 3) / top_blob.c;
                const int b4 = (p0 + 4) / top_blob.c;
                const int b5 = (p0 + 5) / top_blob.c;
                const int b6 = (p0 + 6) / top_blob.c;
                const int b7 = (p0 + 7) / top_blob.c;
                const int q0 = p0 - b0 * top_blob.c;
                const int q1 = p0 + 1 - b1 * top_blob.c;
                const int q2 = p0 + 2 - b2 * top_blob.c;
                const int q3 = p0 + 3 - b3 * top_blob.c;
                const int q4 = p0 + 4 - b4 * top_blob.c;
                const int q5 = p0 + 5 - b5 * top_blob.c;
                const int q6 = p0 + 6 - b6 * top_blob.c;
                const int q7 = p0 + 7 - b7 * top_blob.c;

                const float* ptr = bottom_blob.channel(q);
                float* outptr0 = (float*)top_blob + (size_t)b0 * top_blob.nstep + (size_t)q0 * top_blob.cstep;
                float* outptr1 = (float*)top_blob + (size_t)b1 * top_blob.nstep + (size_t)q1 * top_blob.cstep;
                float* outptr2 = (float*)top_blob + (size_t)b2 * top_blob.nstep + (size_t)q2 * top_blob.cstep;
                float* outptr3 = (float*)top_blob + (size_t)b3 * top_blob.nstep + (size_t)q3 * top_blob.cstep;
                float* outptr4 = (float*)top_blob + (size_t)b4 * top_blob.nstep + (size_t)q4 * top_blob.cstep;
                float* outptr5 = (float*)top_blob + (size_t)b5 * top_blob.nstep + (size_t)q5 * top_blob.cstep;
                float* outptr6 = (float*)top_blob + (size_t)b6 * top_blob.nstep + (size_t)q6 * top_blob.cstep;
                float* outptr7 = (float*)top_blob + (size_t)b7 * top_blob.nstep + (size_t)q7 * top_blob.cstep;

                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    __m256 _r0 = (__m256)__lasx_xvld(ptr, 0);
                    __m256 _r1 = (__m256)__lasx_xvld(ptr + 8, 0);
                    __m256 _r2 = (__m256)__lasx_xvld(ptr + 16, 0);
                    __m256 _r3 = (__m256)__lasx_xvld(ptr + 24, 0);
                    __m256 _r4 = (__m256)__lasx_xvld(ptr + 32, 0);
                    __m256 _r5 = (__m256)__lasx_xvld(ptr + 40, 0);
                    __m256 _r6 = (__m256)__lasx_xvld(ptr + 48, 0);
                    __m256 _r7 = (__m256)__lasx_xvld(ptr + 56, 0);

                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                    __lasx_xvst((__m256i)_r0, outptr0, 0);
                    __lasx_xvst((__m256i)_r1, outptr1, 0);
                    __lasx_xvst((__m256i)_r2, outptr2, 0);
                    __lasx_xvst((__m256i)_r3, outptr3, 0);
                    __lasx_xvst((__m256i)_r4, outptr4, 0);
                    __lasx_xvst((__m256i)_r5, outptr5, 0);
                    __lasx_xvst((__m256i)_r6, outptr6, 0);
                    __lasx_xvst((__m256i)_r7, outptr7, 0);

                    ptr += 64;
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
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];
                    *outptr4++ = ptr[4];
                    *outptr5++ = ptr[5];
                    *outptr6++ = ptr[6];
                    *outptr7++ = ptr[7];

                    ptr += 8;
                }
            }

            return 0;
        }
#endif // __loongarch_asx

        if (batch_axis == 0 && elempack == 1 && out_elempack == 4 && scalar_elemsize == 2 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c == top_blob.c * batch * 4)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;
                const int sq = b * top_blob.c * 4 + q * 4;

                const unsigned short* ptr0 = (const unsigned short*)bottom_blob + (size_t)sq * bottom_blob.cstep;
                const unsigned short* ptr1 = ptr0 + bottom_blob.cstep;
                const unsigned short* ptr2 = ptr1 + bottom_blob.cstep;
                const unsigned short* ptr3 = ptr2 + bottom_blob.cstep;
                unsigned short* outptr = (unsigned short*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * 4;

                for (int i = 0; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == 4 && out_elempack == 1 && scalar_elemsize == 2 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c * 4 == top_blob.c * batch)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < bottom_blob.c; q++)
            {
                const int p0 = q * 4;
                const int b0 = p0 / top_blob.c;
                const int b1 = (p0 + 1) / top_blob.c;
                const int b2 = (p0 + 2) / top_blob.c;
                const int b3 = (p0 + 3) / top_blob.c;
                const int q0 = p0 - b0 * top_blob.c;
                const int q1 = p0 + 1 - b1 * top_blob.c;
                const int q2 = p0 + 2 - b2 * top_blob.c;
                const int q3 = p0 + 3 - b3 * top_blob.c;

                const unsigned short* ptr = bottom_blob.channel(q);
                unsigned short* outptr0 = (unsigned short*)top_blob + (size_t)b0 * top_blob.nstep + (size_t)q0 * top_blob.cstep;
                unsigned short* outptr1 = (unsigned short*)top_blob + (size_t)b1 * top_blob.nstep + (size_t)q1 * top_blob.cstep;
                unsigned short* outptr2 = (unsigned short*)top_blob + (size_t)b2 * top_blob.nstep + (size_t)q2 * top_blob.cstep;
                unsigned short* outptr3 = (unsigned short*)top_blob + (size_t)b3 * top_blob.nstep + (size_t)q3 * top_blob.cstep;

                for (int i = 0; i < size; i++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == 4 && out_elempack == 1 && scalar_elemsize == 4 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c * 4 == top_blob.c * batch)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < bottom_blob.c; q++)
            {
                const int p0 = q * 4;
                const int b0 = p0 / top_blob.c;
                const int b1 = (p0 + 1) / top_blob.c;
                const int b2 = (p0 + 2) / top_blob.c;
                const int b3 = (p0 + 3) / top_blob.c;
                const int q0 = p0 - b0 * top_blob.c;
                const int q1 = p0 + 1 - b1 * top_blob.c;
                const int q2 = p0 + 2 - b2 * top_blob.c;
                const int q3 = p0 + 3 - b3 * top_blob.c;

                const float* ptr = bottom_blob.channel(q);
                float* outptr0 = (float*)top_blob + (size_t)b0 * top_blob.nstep + (size_t)q0 * top_blob.cstep;
                float* outptr1 = (float*)top_blob + (size_t)b1 * top_blob.nstep + (size_t)q1 * top_blob.cstep;
                float* outptr2 = (float*)top_blob + (size_t)b2 * top_blob.nstep + (size_t)q2 * top_blob.cstep;
                float* outptr3 = (float*)top_blob + (size_t)b3 * top_blob.nstep + (size_t)q3 * top_blob.cstep;

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    __builtin_prefetch(ptr + 32);

                    __m128i _r0 = __lsx_vld(ptr, 0);
                    __m128i _r1 = __lsx_vld(ptr + 4, 0);
                    __m128i _r2 = __lsx_vld(ptr + 4 * 2, 0);
                    __m128i _r3 = __lsx_vld(ptr + 4 * 3, 0);

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

                    ptr += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
                for (; i < size; i++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }

            return 0;
        }
#endif // __loongarch_sx

        if (batch_axis == 1 && elempack == out_elempack && dims == 3 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == batch && bottom_blob.c == top_blob.h)
        {
            const size_t size = (size_t)top_blob.w * out_elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.h; bq++)
            {
                const int b = bq / top_blob.h;
                const int q = bq - b * top_blob.h;

                const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)q * bottom_blob.cstep + (size_t)b * bottom_blob.w) * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.w) * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 1 && elempack == out_elempack && dims == 4 && ndim == 3 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == batch && bottom_blob.c == top_blob.c)
        {
            const size_t size = (size_t)top_blob.w * top_blob.h * out_elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;

                const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)q * bottom_blob.cstep + (size_t)b * bottom_blob.w * bottom_blob.h) * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

#if __loongarch_sx
        if (batch_axis == 1 && elempack == 4 && out_elempack == 1 && scalar_elemsize == 4 && dims == 4 && ndim == 3 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == batch && bottom_blob.c * 4 == top_blob.c)
        {
            const int size = top_blob.w * top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * bottom_blob.c; bq++)
            {
                const int b = bq / bottom_blob.c;
                const int q = bq - b * bottom_blob.c;

                const float* ptr = (const float*)bottom_blob + ((size_t)q * bottom_blob.cstep + (size_t)b * size) * 4;
                float* outptr0 = (float*)top_blob + (size_t)b * top_blob.nstep + (size_t)(q * 4) * top_blob.cstep;
                float* outptr1 = outptr0 + top_blob.cstep;
                float* outptr2 = outptr0 + top_blob.cstep * 2;
                float* outptr3 = outptr0 + top_blob.cstep * 3;

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    __builtin_prefetch(ptr + 32);

                    __m128i _r0 = __lsx_vld(ptr, 0);
                    __m128i _r1 = __lsx_vld(ptr + 4, 0);
                    __m128i _r2 = __lsx_vld(ptr + 4 * 2, 0);
                    __m128i _r3 = __lsx_vld(ptr + 4 * 3, 0);

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

                    ptr += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
                for (; i < size; i++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }

            return 0;
        }
#endif // __loongarch_sx

        size_t prefix = 1;
        for (int i = 0; i < batch_axis; i++)
            prefix *= shape[i];

        size_t suffix = 1;
        if (batch_axis == 0)
            suffix = out_total;
        else
        {
            for (int i = batch_axis; i < ndim; i++)
                suffix *= shape[i];
        }

        const size_t bottom_channel_size = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d;
        const size_t top_channel_size = (size_t)top_blob.w * top_blob.h * top_blob.d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < (int)prefix; pp++)
        {
            for (int b = 0; b < batch; b++)
            {
                for (size_t s = 0; s < suffix; s++)
                {
                    const size_t srci = batch_axis == 0 ? (size_t)b * suffix + s : ((size_t)pp * batch + b) * suffix + s;
                    const unsigned char* ptr = (const unsigned char*)bottom_blob;
                    if (dims == 1)
                    {
                        const int x = srci / elempack;
                        const int k = srci % elempack;
                        ptr += (size_t)x * elemsize + k * scalar_elemsize;
                    }
                    else if (dims == 2)
                    {
                        const int x = srci % bottom_blob.w;
                        const int y = srci / bottom_blob.w;
                        const int y0 = y / elempack;
                        const int k = y % elempack;
                        ptr += ((size_t)y0 * bottom_blob.w + x) * elemsize + k * scalar_elemsize;
                    }
                    else
                    {
                        const int q = srci / bottom_channel_size;
                        const size_t r = srci - (size_t)q * bottom_channel_size;
                        const int q0 = q / elempack;
                        const int k = q % elempack;
                        ptr += ((size_t)q0 * bottom_blob.cstep + r) * elemsize + k * scalar_elemsize;
                    }

                    const size_t dsti = batch_axis == 0 ? s : (size_t)pp * suffix + s;
                    unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * top_blob.nstep * out_elemsize;
                    if (top_blob.dims == 1)
                    {
                        const int x = dsti / out_elempack;
                        const int k = dsti % out_elempack;
                        outptr += (size_t)x * out_elemsize + k * scalar_elemsize;
                    }
                    else if (top_blob.dims == 2)
                    {
                        const int x = dsti % top_blob.w;
                        const int y = dsti / top_blob.w;
                        const int y0 = y / out_elempack;
                        const int k = y % out_elempack;
                        outptr += ((size_t)y0 * top_blob.w + x) * out_elemsize + k * scalar_elemsize;
                    }
                    else
                    {
                        const int q = dsti / top_channel_size;
                        const size_t r = dsti - (size_t)q * top_channel_size;
                        const int q0 = q / out_elempack;
                        const int k = q % out_elempack;
                        outptr += ((size_t)q0 * top_blob.cstep + r) * out_elemsize + k * scalar_elemsize;
                    }

                    memcpy(outptr, ptr, scalar_elemsize);
                }
            }
        }

        return 0;
    }

    return -1;
}

int Reshape_loongarch::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    if (batch_mode != 0)
        return forward_batch(bottom_blobs, top_blobs, opt);

    const int elembits = bottom_blob.elembits();
    if (elembits == 16)
        return forward_bf16s(bottom_blobs, top_blobs, opt);

    int outw = w;
    int outh = h;
    int outd = d;
    int outc = c;

    if (!shape_expr.empty())
    {
        int er = eval_shape_expr(bottom_blobs, outw, outh, outd, outc);
        if (er != 0)
            return -1;
    }

    if (ndim == 1)
    {
        flatten(bottom_blob, top_blob, opt);
        if (top_blob.empty())
            return -100;

        return 0;
    }

    const int dims = bottom_blob.dims;
    const int elempack = bottom_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;

    if (ndim == 2)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;

        if (outw == -1)
            outw = total / outh;
        if (outh == -1)
            outh = total / outw;

        int out_elempack = 1;
#if __loongarch_sx
        if (opt.use_packing_layout)
        {
#if __loongarch_asx
            out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
            out_elempack = outh % 4 == 0 ? 4 : 1;
#endif // __loongarch_asx
        }
#endif // __loongarch_sx
        const size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 2 && bottom_blob.h * elempack == outh && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        if (out_elempack == 1)
        {
            flatten(bottom_blob, top_blob, opt);
            if (top_blob.empty())
                return -100;

            top_blob.dims = 2;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.cstep = top_blob.cstep * top_blob.elempack;
            top_blob.elemsize = out_elemsize;
            top_blob.elempack = out_elempack;

            return 0;
        }

        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __loongarch_asx
        if (out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + outw * i * 8;
                const float* ptr1 = (const float*)bottom_blob_flattened + outw * (i * 8 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + outw * (i * 8 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + outw * (i * 8 + 3);
                const float* ptr4 = (const float*)bottom_blob_flattened + outw * (i * 8 + 4);
                const float* ptr5 = (const float*)bottom_blob_flattened + outw * (i * 8 + 5);
                const float* ptr6 = (const float*)bottom_blob_flattened + outw * (i * 8 + 6);
                const float* ptr7 = (const float*)bottom_blob_flattened + outw * (i * 8 + 7);
                float* outptr = top_blob.row(i);

                int j = 0;
                for (; j + 7 < outw; j += 8)
                {
                    __m256 _row0 = (__m256)__lasx_xvld(ptr0, 0);
                    __m256 _row1 = (__m256)__lasx_xvld(ptr1, 0);
                    __m256 _row2 = (__m256)__lasx_xvld(ptr2, 0);
                    __m256 _row3 = (__m256)__lasx_xvld(ptr3, 0);
                    __m256 _row4 = (__m256)__lasx_xvld(ptr4, 0);
                    __m256 _row5 = (__m256)__lasx_xvld(ptr5, 0);
                    __m256 _row6 = (__m256)__lasx_xvld(ptr6, 0);
                    __m256 _row7 = (__m256)__lasx_xvld(ptr7, 0);

                    transpose8x8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);

                    __lasx_xvst((__m256i)_row0, outptr, 0);
                    __lasx_xvst((__m256i)_row1, outptr + 8, 0);
                    __lasx_xvst((__m256i)_row2, outptr + 8 * 2, 0);
                    __lasx_xvst((__m256i)_row3, outptr + 8 * 3, 0);
                    __lasx_xvst((__m256i)_row4, outptr + 8 * 4, 0);
                    __lasx_xvst((__m256i)_row5, outptr + 8 * 5, 0);
                    __lasx_xvst((__m256i)_row6, outptr + 8 * 6, 0);
                    __lasx_xvst((__m256i)_row7, outptr + 8 * 7, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    ptr4 += 8;
                    ptr5 += 8;
                    ptr6 += 8;
                    ptr7 += 8;
                    outptr += 64;
                }
                for (; j < outw; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }

            return 0;
        }
#endif // __loongarch_asx

#if __loongarch_sx
        if (out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + outw * i * 4;
                const float* ptr1 = (const float*)bottom_blob_flattened + outw * (i * 4 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + outw * (i * 4 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + outw * (i * 4 + 3);
                float* outptr = top_blob.row(i);

                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
                    __m128 _row0 = (__m128)__lsx_vld(ptr0, 0);
                    __m128 _row1 = (__m128)__lsx_vld(ptr1, 0);
                    __m128 _row2 = (__m128)__lsx_vld(ptr2, 0);
                    __m128 _row3 = (__m128)__lsx_vld(ptr3, 0);

                    __m128i _row01r = __lsx_vilvl_w((__m128i)_row1, (__m128i)_row0);
                    __m128i _row01l = __lsx_vilvh_w((__m128i)_row1, (__m128i)_row0);
                    __m128i _row23r = __lsx_vilvl_w((__m128i)_row3, (__m128i)_row2);
                    __m128i _row23l = __lsx_vilvh_w((__m128i)_row3, (__m128i)_row2);
                    __m128i _row0123_0 = __lsx_vilvl_d(_row23r, _row01r);
                    __m128i _row0123_1 = __lsx_vilvh_d(_row23r, _row01r);
                    __m128i _row0123_2 = __lsx_vilvl_d(_row23l, _row01l);
                    __m128i _row0123_3 = __lsx_vilvh_d(_row23l, _row01l);

                    __lsx_vst(_row0123_0, outptr, 0);
                    __lsx_vst(_row0123_1, outptr + 4, 0);
                    __lsx_vst(_row0123_2, outptr + 4 * 2, 0);
                    __lsx_vst(_row0123_3, outptr + 4 * 3, 0);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; j < outw; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }
#endif // __loongarch_sx
    }

    if (ndim == 3 || ndim == 4)
    {
        if (ndim == 3)
        {
            if (outw == 0)
                outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
            if (outh == 0)
                outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
            if (outc == 0)
                outc = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;

            if (outw == -1)
                outw = total / outc / outh;
            if (outh == -1)
                outh = total / outc / outw;
            if (outc == -1)
                outc = total / outh / outw;

            outd = 1;
        }
        else // if (ndim == 4)
        {
            if (outw == 0)
                outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
            if (outh == 0)
                outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
            if (outd == 0)
                outd = bottom_blob.d;
            if (outc == 0)
                outc = (dims == 3 || dims == 4) ? bottom_blob.c * elempack : bottom_blob.c;

            if (outw == -1)
                outw = total / outc / outd / outh;
            if (outh == -1)
                outh = total / outc / outd / outw;
            if (outd == -1)
                outd = total / outc / outh / outw;
            if (outc == -1)
                outc = total / outd / outh / outw;
        }

        int out_elempack = 1;
#if __loongarch_sx
        if (opt.use_packing_layout)
        {
#if __loongarch_asx
            out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#else
            out_elempack = outc % 4 == 0 ? 4 : 1;
#endif // __loongarch_asx
        }
#endif // __loongarch_sx
        const size_t out_elemsize = elemsize / elempack * out_elempack;

        if ((dims == 3 || dims == 4) && bottom_blob.c * elempack == outc && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            top_blob.dims = ndim;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.d = outd;
            return 0;
        }

        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        if (ndim == 3)
            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        else // if (ndim == 4)
            top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int size = top_blob.w * top_blob.h * top_blob.d;

#if __loongarch_asx
        if (out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + size * q * 8;
                const float* ptr1 = (const float*)bottom_blob_flattened + size * (q * 8 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + size * (q * 8 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + size * (q * 8 + 3);
                const float* ptr4 = (const float*)bottom_blob_flattened + size * (q * 8 + 4);
                const float* ptr5 = (const float*)bottom_blob_flattened + size * (q * 8 + 5);
                const float* ptr6 = (const float*)bottom_blob_flattened + size * (q * 8 + 6);
                const float* ptr7 = (const float*)bottom_blob_flattened + size * (q * 8 + 7);
                float* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    __m256 _row0 = (__m256)__lasx_xvld(ptr0, 0);
                    __m256 _row1 = (__m256)__lasx_xvld(ptr1, 0);
                    __m256 _row2 = (__m256)__lasx_xvld(ptr2, 0);
                    __m256 _row3 = (__m256)__lasx_xvld(ptr3, 0);
                    __m256 _row4 = (__m256)__lasx_xvld(ptr4, 0);
                    __m256 _row5 = (__m256)__lasx_xvld(ptr5, 0);
                    __m256 _row6 = (__m256)__lasx_xvld(ptr6, 0);
                    __m256 _row7 = (__m256)__lasx_xvld(ptr7, 0);

                    transpose8x8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);

                    __lasx_xvst((__m256i)_row0, outptr, 0);
                    __lasx_xvst((__m256i)_row1, outptr + 8, 0);
                    __lasx_xvst((__m256i)_row2, outptr + 8 * 2, 0);
                    __lasx_xvst((__m256i)_row3, outptr + 8 * 3, 0);
                    __lasx_xvst((__m256i)_row4, outptr + 8 * 4, 0);
                    __lasx_xvst((__m256i)_row5, outptr + 8 * 5, 0);
                    __lasx_xvst((__m256i)_row6, outptr + 8 * 6, 0);
                    __lasx_xvst((__m256i)_row7, outptr + 8 * 7, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    ptr4 += 8;
                    ptr5 += 8;
                    ptr6 += 8;
                    ptr7 += 8;
                    outptr += 64;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }

            return 0;
        }
#endif // __loongarch_asx

#if __loongarch_sx
        if (out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + size * q * 4;
                const float* ptr1 = (const float*)bottom_blob_flattened + size * (q * 4 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + size * (q * 4 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + size * (q * 4 + 3);
                float* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    __m128 _row0 = (__m128)__lsx_vld(ptr0, 0);
                    __m128 _row1 = (__m128)__lsx_vld(ptr1, 0);
                    __m128 _row2 = (__m128)__lsx_vld(ptr2, 0);
                    __m128 _row3 = (__m128)__lsx_vld(ptr3, 0);

                    __m128i _row01r = __lsx_vilvl_w((__m128i)_row1, (__m128i)_row0);
                    __m128i _row01l = __lsx_vilvh_w((__m128i)_row1, (__m128i)_row0);
                    __m128i _row23r = __lsx_vilvl_w((__m128i)_row3, (__m128i)_row2);
                    __m128i _row23l = __lsx_vilvh_w((__m128i)_row3, (__m128i)_row2);
                    __m128i _row0123_0 = __lsx_vilvl_d(_row23r, _row01r);
                    __m128i _row0123_1 = __lsx_vilvh_d(_row23r, _row01r);
                    __m128i _row0123_2 = __lsx_vilvl_d(_row23l, _row01l);
                    __m128i _row0123_3 = __lsx_vilvh_d(_row23l, _row01l);

                    __lsx_vst(_row0123_0, outptr, 0);
                    __lsx_vst(_row0123_1, outptr + 4, 0);
                    __lsx_vst(_row0123_2, outptr + 4 * 2, 0);
                    __lsx_vst(_row0123_3, outptr + 4 * 3, 0);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }
#endif // __loongarch_sx

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr = (const float*)bottom_blob_flattened + size * q;
            float* outptr = top_blob.channel(q);

            int i = 0;
#if __loongarch_asx
            for (; i + 7 < size; i += 8)
            {
                __lasx_xvst(__lasx_xvld(ptr, 0), outptr, 0);
                ptr += 8;
                outptr += 8;
            }
#endif // __loongarch_asx
#if __loongarch_sx
            for (; i + 3 < size; i += 4)
            {
                __lsx_vst(__lsx_vld(ptr, 0), outptr, 0);
                ptr += 4;
                outptr += 4;
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                *outptr++ = *ptr++;
            }
        }
    }

    return 0;
}

int Reshape_loongarch::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (batch_mode != 0)
        return forward_batch(bottom_blobs, top_blobs, opt);

    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    int outw = w;
    int outh = h;
    int outd = d;
    int outc = c;

    if (!shape_expr.empty())
    {
        int er = eval_shape_expr(bottom_blobs, outw, outh, outd, outc);
        if (er != 0)
            return -1;
    }

    if (ndim == 1)
    {
        flatten(bottom_blob, top_blob, opt);
        if (top_blob.empty())
            return -100;

        return 0;
    }

    const int dims = bottom_blob.dims;
    const int elempack = bottom_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;

    if (ndim == 2)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;

        if (outw == -1)
            outw = total / outh;
        if (outh == -1)
            outh = total / outw;

        int out_elempack = 1;
#if __loongarch_sx
        if (opt.use_packing_layout)
        {
            out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
        }
#endif // __loongarch_sx
        const size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 2 && bottom_blob.h * elempack == outh && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        if (out_elempack == 1)
        {
            flatten(bottom_blob, top_blob, opt);
            if (top_blob.empty())
                return -100;

            top_blob.dims = 2;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.cstep = top_blob.cstep * top_blob.elempack;
            top_blob.elemsize = out_elemsize;
            top_blob.elempack = out_elempack;

            return 0;
        }

        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __loongarch_sx
        if (out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const unsigned short* ptr0 = (const unsigned short*)bottom_blob_flattened + outw * i * 8;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 1);
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 2);
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 3);
                const unsigned short* ptr4 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 4);
                const unsigned short* ptr5 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 5);
                const unsigned short* ptr6 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 6);
                const unsigned short* ptr7 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 7);
                unsigned short* outptr = top_blob.row<unsigned short>(i);

                int j = 0;
                for (; j + 7 < outw; j += 8)
                {
                    __m128i _r0 = __lsx_vld(ptr0, 0);
                    __m128i _r1 = __lsx_vld(ptr1, 0);
                    __m128i _r2 = __lsx_vld(ptr2, 0);
                    __m128i _r3 = __lsx_vld(ptr3, 0);
                    __m128i _r4 = __lsx_vld(ptr4, 0);
                    __m128i _r5 = __lsx_vld(ptr5, 0);
                    __m128i _r6 = __lsx_vld(ptr6, 0);
                    __m128i _r7 = __lsx_vld(ptr7, 0);

                    transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                    __lsx_vst(_r0, outptr, 0);
                    __lsx_vst(_r1, outptr + 8, 0);
                    __lsx_vst(_r2, outptr + 16, 0);
                    __lsx_vst(_r3, outptr + 24, 0);
                    __lsx_vst(_r4, outptr + 32, 0);
                    __lsx_vst(_r5, outptr + 40, 0);
                    __lsx_vst(_r6, outptr + 48, 0);
                    __lsx_vst(_r7, outptr + 56, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    ptr4 += 8;
                    ptr5 += 8;
                    ptr6 += 8;
                    ptr7 += 8;
                    outptr += 64;
                }
                for (; j < outw; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }

            return 0;
        }
#endif // __loongarch_sx

#if __loongarch_sx
        if (out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const unsigned short* ptr0 = (const unsigned short*)bottom_blob_flattened + outw * i * 4;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob_flattened + outw * (i * 4 + 1);
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob_flattened + outw * (i * 4 + 2);
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob_flattened + outw * (i * 4 + 3);
                unsigned short* outptr = top_blob.row<unsigned short>(i);

                for (int j = 0; j < outw; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }
#endif // __loongarch_sx
    }

    if (ndim == 3 || ndim == 4)
    {
        if (ndim == 3)
        {
            if (outw == 0)
                outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
            if (outh == 0)
                outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
            if (outc == 0)
                outc = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;

            if (outw == -1)
                outw = total / outc / outh;
            if (outh == -1)
                outh = total / outc / outw;
            if (outc == -1)
                outc = total / outh / outw;

            outd = 1;
        }
        else // if (ndim == 4)
        {
            if (outw == 0)
                outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
            if (outh == 0)
                outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
            if (outd == 0)
                outd = bottom_blob.d;
            if (outc == 0)
                outc = (dims == 3 || dims == 4) ? bottom_blob.c * elempack : bottom_blob.c;

            if (outw == -1)
                outw = total / outc / outd / outh;
            if (outh == -1)
                outh = total / outc / outd / outw;
            if (outd == -1)
                outd = total / outc / outh / outw;
            if (outc == -1)
                outc = total / outd / outh / outw;
        }

        int out_elempack = 1;
#if __loongarch_sx
        if (opt.use_packing_layout)
        {
            out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
        }
#endif // __loongarch_sx
        const size_t out_elemsize = elemsize / elempack * out_elempack;

        if ((dims == 3 || dims == 4) && bottom_blob.c * elempack == outc && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            top_blob.dims = ndim;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.d = outd;
            return 0;
        }

        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        if (ndim == 3)
            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        else // if (ndim == 4)
            top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int size = top_blob.w * top_blob.h * top_blob.d;

#if __loongarch_sx
        if (out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr0 = (const unsigned short*)bottom_blob_flattened + size * q * 8;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 1);
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 2);
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 3);
                const unsigned short* ptr4 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 4);
                const unsigned short* ptr5 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 5);
                const unsigned short* ptr6 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 6);
                const unsigned short* ptr7 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 7);
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    __m128i _r0 = __lsx_vld(ptr0, 0);
                    __m128i _r1 = __lsx_vld(ptr1, 0);
                    __m128i _r2 = __lsx_vld(ptr2, 0);
                    __m128i _r3 = __lsx_vld(ptr3, 0);
                    __m128i _r4 = __lsx_vld(ptr4, 0);
                    __m128i _r5 = __lsx_vld(ptr5, 0);
                    __m128i _r6 = __lsx_vld(ptr6, 0);
                    __m128i _r7 = __lsx_vld(ptr7, 0);

                    transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                    __lsx_vst(_r0, outptr, 0);
                    __lsx_vst(_r1, outptr + 8, 0);
                    __lsx_vst(_r2, outptr + 16, 0);
                    __lsx_vst(_r3, outptr + 24, 0);
                    __lsx_vst(_r4, outptr + 32, 0);
                    __lsx_vst(_r5, outptr + 40, 0);
                    __lsx_vst(_r6, outptr + 48, 0);
                    __lsx_vst(_r7, outptr + 56, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    ptr4 += 8;
                    ptr5 += 8;
                    ptr6 += 8;
                    ptr7 += 8;
                    outptr += 64;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }

            return 0;
        }
#endif // __loongarch_sx

#if __loongarch_sx
        if (out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr0 = (const unsigned short*)bottom_blob_flattened + size * q * 4;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob_flattened + size * (q * 4 + 1);
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob_flattened + size * (q * 4 + 2);
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob_flattened + size * (q * 4 + 3);
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }
#endif // __loongarch_sx

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob_flattened + size * q;
            unsigned short* outptr = top_blob.channel(q);

            memcpy(outptr, ptr, (size_t)size * sizeof(unsigned short));
        }
    }

    return 0;
}

} // namespace ncnn

// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif // __SSE2__

#include "cpu.h"
#include "x86_usability.h"

#include <string.h>

namespace ncnn {

Reshape_x86::Reshape_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
    support_fp16_storage = cpu_support_x86_f16c();
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

#if NCNN_BATCH
int Reshape_x86::forward_batch(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    // resolve out shape
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
#if __SSE2__
    if (opt.use_packing_layout)
    {
        if (ndim == 1)
        {
#if __AVX512F__
            out_elempack = outw % 16 == 0 ? 16 : outw % 8 == 0 ? 8 : outw % 4 == 0 ? 4 : 1;
#elif __AVX__
            out_elempack = outw % 8 == 0 ? 8 : outw % 4 == 0 ? 4 : 1;
#else
            out_elempack = outw % 4 == 0 ? 4 : 1;
#endif
        }
        if (ndim == 2)
        {
#if __AVX512F__
            out_elempack = outh % 16 == 0 ? 16 : outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#elif __AVX__
            out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
            out_elempack = outh % 4 == 0 ? 4 : 1;
#endif
        }
        if (ndim == 3 || ndim == 4)
        {
#if __AVX512F__
            out_elempack = outc % 16 == 0 ? 16 : outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#elif __AVX__
            out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#else
            out_elempack = outc % 4 == 0 ? 4 : 1;
#endif
        }
    }
#endif // __SSE2__
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

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (scalar_elemsize == 4 && batch_axis == 0 && elempack == 1 && out_elempack == 16 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && top_blob.h * 16 == bottom_blob.h * bottom_blob.n)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const int y = i * 16;
                const int b0 = y / bottom_blob.h;
                const int b1 = (y + 1) / bottom_blob.h;
                const int b2 = (y + 2) / bottom_blob.h;
                const int b3 = (y + 3) / bottom_blob.h;
                const int b4 = (y + 4) / bottom_blob.h;
                const int b5 = (y + 5) / bottom_blob.h;
                const int b6 = (y + 6) / bottom_blob.h;
                const int b7 = (y + 7) / bottom_blob.h;
                const int b8 = (y + 8) / bottom_blob.h;
                const int b9 = (y + 9) / bottom_blob.h;
                const int ba = (y + 10) / bottom_blob.h;
                const int bb = (y + 11) / bottom_blob.h;
                const int bc = (y + 12) / bottom_blob.h;
                const int bd = (y + 13) / bottom_blob.h;
                const int be = (y + 14) / bottom_blob.h;
                const int bf = (y + 15) / bottom_blob.h;
                const int y0 = y - b0 * bottom_blob.h;
                const int y1 = y + 1 - b1 * bottom_blob.h;
                const int y2 = y + 2 - b2 * bottom_blob.h;
                const int y3 = y + 3 - b3 * bottom_blob.h;
                const int y4 = y + 4 - b4 * bottom_blob.h;
                const int y5 = y + 5 - b5 * bottom_blob.h;
                const int y6 = y + 6 - b6 * bottom_blob.h;
                const int y7 = y + 7 - b7 * bottom_blob.h;
                const int y8 = y + 8 - b8 * bottom_blob.h;
                const int y9 = y + 9 - b9 * bottom_blob.h;
                const int ya = y + 10 - ba * bottom_blob.h;
                const int yb = y + 11 - bb * bottom_blob.h;
                const int yc = y + 12 - bc * bottom_blob.h;
                const int yd = y + 13 - bd * bottom_blob.h;
                const int ye = y + 14 - be * bottom_blob.h;
                const int yf = y + 15 - bf * bottom_blob.h;

                const float* ptr0 = (const float*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)y0 * bottom_blob.w;
                const float* ptr1 = (const float*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)y1 * bottom_blob.w;
                const float* ptr2 = (const float*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)y2 * bottom_blob.w;
                const float* ptr3 = (const float*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)y3 * bottom_blob.w;
                const float* ptr4 = (const float*)bottom_blob + (size_t)b4 * bottom_blob.nstep + (size_t)y4 * bottom_blob.w;
                const float* ptr5 = (const float*)bottom_blob + (size_t)b5 * bottom_blob.nstep + (size_t)y5 * bottom_blob.w;
                const float* ptr6 = (const float*)bottom_blob + (size_t)b6 * bottom_blob.nstep + (size_t)y6 * bottom_blob.w;
                const float* ptr7 = (const float*)bottom_blob + (size_t)b7 * bottom_blob.nstep + (size_t)y7 * bottom_blob.w;
                const float* ptr8 = (const float*)bottom_blob + (size_t)b8 * bottom_blob.nstep + (size_t)y8 * bottom_blob.w;
                const float* ptr9 = (const float*)bottom_blob + (size_t)b9 * bottom_blob.nstep + (size_t)y9 * bottom_blob.w;
                const float* ptra = (const float*)bottom_blob + (size_t)ba * bottom_blob.nstep + (size_t)ya * bottom_blob.w;
                const float* ptrb = (const float*)bottom_blob + (size_t)bb * bottom_blob.nstep + (size_t)yb * bottom_blob.w;
                const float* ptrc = (const float*)bottom_blob + (size_t)bc * bottom_blob.nstep + (size_t)yc * bottom_blob.w;
                const float* ptrd = (const float*)bottom_blob + (size_t)bd * bottom_blob.nstep + (size_t)yd * bottom_blob.w;
                const float* ptre = (const float*)bottom_blob + (size_t)be * bottom_blob.nstep + (size_t)ye * bottom_blob.w;
                const float* ptrf = (const float*)bottom_blob + (size_t)bf * bottom_blob.nstep + (size_t)yf * bottom_blob.w;
                float* outptr = top_blob.row(i);

                int j = 0;
                for (; j + 15 < bottom_blob.w; j += 16)
                {
                    __m512 _row0 = _mm512_loadu_ps(ptr0);
                    __m512 _row1 = _mm512_loadu_ps(ptr1);
                    __m512 _row2 = _mm512_loadu_ps(ptr2);
                    __m512 _row3 = _mm512_loadu_ps(ptr3);
                    __m512 _row4 = _mm512_loadu_ps(ptr4);
                    __m512 _row5 = _mm512_loadu_ps(ptr5);
                    __m512 _row6 = _mm512_loadu_ps(ptr6);
                    __m512 _row7 = _mm512_loadu_ps(ptr7);
                    __m512 _row8 = _mm512_loadu_ps(ptr8);
                    __m512 _row9 = _mm512_loadu_ps(ptr9);
                    __m512 _rowa = _mm512_loadu_ps(ptra);
                    __m512 _rowb = _mm512_loadu_ps(ptrb);
                    __m512 _rowc = _mm512_loadu_ps(ptrc);
                    __m512 _rowd = _mm512_loadu_ps(ptrd);
                    __m512 _rowe = _mm512_loadu_ps(ptre);
                    __m512 _rowf = _mm512_loadu_ps(ptrf);

                    transpose16x16_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7, _row8, _row9, _rowa, _rowb, _rowc, _rowd, _rowe, _rowf);

                    _mm512_storeu_ps(outptr, _row0);
                    _mm512_storeu_ps(outptr + 16, _row1);
                    _mm512_storeu_ps(outptr + 16 * 2, _row2);
                    _mm512_storeu_ps(outptr + 16 * 3, _row3);
                    _mm512_storeu_ps(outptr + 16 * 4, _row4);
                    _mm512_storeu_ps(outptr + 16 * 5, _row5);
                    _mm512_storeu_ps(outptr + 16 * 6, _row6);
                    _mm512_storeu_ps(outptr + 16 * 7, _row7);
                    _mm512_storeu_ps(outptr + 16 * 8, _row8);
                    _mm512_storeu_ps(outptr + 16 * 9, _row9);
                    _mm512_storeu_ps(outptr + 16 * 10, _rowa);
                    _mm512_storeu_ps(outptr + 16 * 11, _rowb);
                    _mm512_storeu_ps(outptr + 16 * 12, _rowc);
                    _mm512_storeu_ps(outptr + 16 * 13, _rowd);
                    _mm512_storeu_ps(outptr + 16 * 14, _rowe);
                    _mm512_storeu_ps(outptr + 16 * 15, _rowf);

                    ptr0 += 16;
                    ptr1 += 16;
                    ptr2 += 16;
                    ptr3 += 16;
                    ptr4 += 16;
                    ptr5 += 16;
                    ptr6 += 16;
                    ptr7 += 16;
                    ptr8 += 16;
                    ptr9 += 16;
                    ptra += 16;
                    ptrb += 16;
                    ptrc += 16;
                    ptrd += 16;
                    ptre += 16;
                    ptrf += 16;
                    outptr += 256;
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
                    outptr[8] = *ptr8++;
                    outptr[9] = *ptr9++;
                    outptr[10] = *ptra++;
                    outptr[11] = *ptrb++;
                    outptr[12] = *ptrc++;
                    outptr[13] = *ptrd++;
                    outptr[14] = *ptre++;
                    outptr[15] = *ptrf++;

                    outptr += 16;
                }
            }

            return 0;
        }
#endif // __AVX512F__

        if (scalar_elemsize == 4 && batch_axis == 0 && elempack == 1 && out_elempack == 8 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && top_blob.h * 8 == bottom_blob.h * bottom_blob.n)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const int y = i * 8;
                const int b0 = y / bottom_blob.h;
                const int b1 = (y + 1) / bottom_blob.h;
                const int b2 = (y + 2) / bottom_blob.h;
                const int b3 = (y + 3) / bottom_blob.h;
                const int b4 = (y + 4) / bottom_blob.h;
                const int b5 = (y + 5) / bottom_blob.h;
                const int b6 = (y + 6) / bottom_blob.h;
                const int b7 = (y + 7) / bottom_blob.h;
                const int y0 = y - b0 * bottom_blob.h;
                const int y1 = y + 1 - b1 * bottom_blob.h;
                const int y2 = y + 2 - b2 * bottom_blob.h;
                const int y3 = y + 3 - b3 * bottom_blob.h;
                const int y4 = y + 4 - b4 * bottom_blob.h;
                const int y5 = y + 5 - b5 * bottom_blob.h;
                const int y6 = y + 6 - b6 * bottom_blob.h;
                const int y7 = y + 7 - b7 * bottom_blob.h;

                const float* ptr0 = (const float*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)y0 * bottom_blob.w;
                const float* ptr1 = (const float*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)y1 * bottom_blob.w;
                const float* ptr2 = (const float*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)y2 * bottom_blob.w;
                const float* ptr3 = (const float*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)y3 * bottom_blob.w;
                const float* ptr4 = (const float*)bottom_blob + (size_t)b4 * bottom_blob.nstep + (size_t)y4 * bottom_blob.w;
                const float* ptr5 = (const float*)bottom_blob + (size_t)b5 * bottom_blob.nstep + (size_t)y5 * bottom_blob.w;
                const float* ptr6 = (const float*)bottom_blob + (size_t)b6 * bottom_blob.nstep + (size_t)y6 * bottom_blob.w;
                const float* ptr7 = (const float*)bottom_blob + (size_t)b7 * bottom_blob.nstep + (size_t)y7 * bottom_blob.w;
                float* outptr = top_blob.row(i);

                int j = 0;
                for (; j + 7 < bottom_blob.w; j += 8)
                {
                    __m256 _row0 = _mm256_loadu_ps(ptr0);
                    __m256 _row1 = _mm256_loadu_ps(ptr1);
                    __m256 _row2 = _mm256_loadu_ps(ptr2);
                    __m256 _row3 = _mm256_loadu_ps(ptr3);
                    __m256 _row4 = _mm256_loadu_ps(ptr4);
                    __m256 _row5 = _mm256_loadu_ps(ptr5);
                    __m256 _row6 = _mm256_loadu_ps(ptr6);
                    __m256 _row7 = _mm256_loadu_ps(ptr7);

                    transpose8x8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);

                    _mm256_storeu_ps(outptr, _row0);
                    _mm256_storeu_ps(outptr + 8, _row1);
                    _mm256_storeu_ps(outptr + 16, _row2);
                    _mm256_storeu_ps(outptr + 24, _row3);
                    _mm256_storeu_ps(outptr + 32, _row4);
                    _mm256_storeu_ps(outptr + 40, _row5);
                    _mm256_storeu_ps(outptr + 48, _row6);
                    _mm256_storeu_ps(outptr + 56, _row7);

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
#endif // __AVX__

        if (scalar_elemsize == 4 && batch_axis == 0 && elempack == 1 && out_elempack == 4 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && top_blob.h * 4 == bottom_blob.h * bottom_blob.n)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const int y = i * 4;
                const int b0 = y / bottom_blob.h;
                const int b1 = (y + 1) / bottom_blob.h;
                const int b2 = (y + 2) / bottom_blob.h;
                const int b3 = (y + 3) / bottom_blob.h;
                const int y0 = y - b0 * bottom_blob.h;
                const int y1 = y + 1 - b1 * bottom_blob.h;
                const int y2 = y + 2 - b2 * bottom_blob.h;
                const int y3 = y + 3 - b3 * bottom_blob.h;

                const float* ptr0 = (const float*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)y0 * bottom_blob.w;
                const float* ptr1 = (const float*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)y1 * bottom_blob.w;
                const float* ptr2 = (const float*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)y2 * bottom_blob.w;
                const float* ptr3 = (const float*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)y3 * bottom_blob.w;
                float* outptr = top_blob.row(i);

                int j = 0;
                for (; j + 3 < bottom_blob.w; j += 4)
                {
                    __m128 _row0 = _mm_loadu_ps(ptr0);
                    __m128 _row1 = _mm_loadu_ps(ptr1);
                    __m128 _row2 = _mm_loadu_ps(ptr2);
                    __m128 _row3 = _mm_loadu_ps(ptr3);

                    _MM_TRANSPOSE4_PS(_row0, _row1, _row2, _row3);

                    _mm_storeu_ps(outptr, _row0);
                    _mm_storeu_ps(outptr + 4, _row1);
                    _mm_storeu_ps(outptr + 8, _row2);
                    _mm_storeu_ps(outptr + 12, _row3);

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
#endif // __SSE2__

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

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (scalar_elemsize == 4 && batch_axis == 0 && elempack == 1 && out_elempack == 16 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && top_blob.c * 16 == bottom_blob.c * bottom_blob.n)
        {
            const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const int bq0 = q * 16;
                const int b0 = bq0 / bottom_blob.c;
                const int b1 = (bq0 + 1) / bottom_blob.c;
                const int b2 = (bq0 + 2) / bottom_blob.c;
                const int b3 = (bq0 + 3) / bottom_blob.c;
                const int b4 = (bq0 + 4) / bottom_blob.c;
                const int b5 = (bq0 + 5) / bottom_blob.c;
                const int b6 = (bq0 + 6) / bottom_blob.c;
                const int b7 = (bq0 + 7) / bottom_blob.c;
                const int b8 = (bq0 + 8) / bottom_blob.c;
                const int b9 = (bq0 + 9) / bottom_blob.c;
                const int ba = (bq0 + 10) / bottom_blob.c;
                const int bb = (bq0 + 11) / bottom_blob.c;
                const int bc = (bq0 + 12) / bottom_blob.c;
                const int bd = (bq0 + 13) / bottom_blob.c;
                const int be = (bq0 + 14) / bottom_blob.c;
                const int bf = (bq0 + 15) / bottom_blob.c;
                const int q0 = bq0 - b0 * bottom_blob.c;
                const int q1 = bq0 + 1 - b1 * bottom_blob.c;
                const int q2 = bq0 + 2 - b2 * bottom_blob.c;
                const int q3 = bq0 + 3 - b3 * bottom_blob.c;
                const int q4 = bq0 + 4 - b4 * bottom_blob.c;
                const int q5 = bq0 + 5 - b5 * bottom_blob.c;
                const int q6 = bq0 + 6 - b6 * bottom_blob.c;
                const int q7 = bq0 + 7 - b7 * bottom_blob.c;
                const int q8 = bq0 + 8 - b8 * bottom_blob.c;
                const int q9 = bq0 + 9 - b9 * bottom_blob.c;
                const int qa = bq0 + 10 - ba * bottom_blob.c;
                const int qb = bq0 + 11 - bb * bottom_blob.c;
                const int qc = bq0 + 12 - bc * bottom_blob.c;
                const int qd = bq0 + 13 - bd * bottom_blob.c;
                const int qe = bq0 + 14 - be * bottom_blob.c;
                const int qf = bq0 + 15 - bf * bottom_blob.c;

                const float* ptr0 = (const float*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)q0 * bottom_blob.cstep;
                const float* ptr1 = (const float*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)q1 * bottom_blob.cstep;
                const float* ptr2 = (const float*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)q2 * bottom_blob.cstep;
                const float* ptr3 = (const float*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)q3 * bottom_blob.cstep;
                const float* ptr4 = (const float*)bottom_blob + (size_t)b4 * bottom_blob.nstep + (size_t)q4 * bottom_blob.cstep;
                const float* ptr5 = (const float*)bottom_blob + (size_t)b5 * bottom_blob.nstep + (size_t)q5 * bottom_blob.cstep;
                const float* ptr6 = (const float*)bottom_blob + (size_t)b6 * bottom_blob.nstep + (size_t)q6 * bottom_blob.cstep;
                const float* ptr7 = (const float*)bottom_blob + (size_t)b7 * bottom_blob.nstep + (size_t)q7 * bottom_blob.cstep;
                const float* ptr8 = (const float*)bottom_blob + (size_t)b8 * bottom_blob.nstep + (size_t)q8 * bottom_blob.cstep;
                const float* ptr9 = (const float*)bottom_blob + (size_t)b9 * bottom_blob.nstep + (size_t)q9 * bottom_blob.cstep;
                const float* ptra = (const float*)bottom_blob + (size_t)ba * bottom_blob.nstep + (size_t)qa * bottom_blob.cstep;
                const float* ptrb = (const float*)bottom_blob + (size_t)bb * bottom_blob.nstep + (size_t)qb * bottom_blob.cstep;
                const float* ptrc = (const float*)bottom_blob + (size_t)bc * bottom_blob.nstep + (size_t)qc * bottom_blob.cstep;
                const float* ptrd = (const float*)bottom_blob + (size_t)bd * bottom_blob.nstep + (size_t)qd * bottom_blob.cstep;
                const float* ptre = (const float*)bottom_blob + (size_t)be * bottom_blob.nstep + (size_t)qe * bottom_blob.cstep;
                const float* ptrf = (const float*)bottom_blob + (size_t)bf * bottom_blob.nstep + (size_t)qf * bottom_blob.cstep;
                float* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 15 < size; i += 16)
                {
                    __m512 _row0 = _mm512_loadu_ps(ptr0);
                    __m512 _row1 = _mm512_loadu_ps(ptr1);
                    __m512 _row2 = _mm512_loadu_ps(ptr2);
                    __m512 _row3 = _mm512_loadu_ps(ptr3);
                    __m512 _row4 = _mm512_loadu_ps(ptr4);
                    __m512 _row5 = _mm512_loadu_ps(ptr5);
                    __m512 _row6 = _mm512_loadu_ps(ptr6);
                    __m512 _row7 = _mm512_loadu_ps(ptr7);
                    __m512 _row8 = _mm512_loadu_ps(ptr8);
                    __m512 _row9 = _mm512_loadu_ps(ptr9);
                    __m512 _rowa = _mm512_loadu_ps(ptra);
                    __m512 _rowb = _mm512_loadu_ps(ptrb);
                    __m512 _rowc = _mm512_loadu_ps(ptrc);
                    __m512 _rowd = _mm512_loadu_ps(ptrd);
                    __m512 _rowe = _mm512_loadu_ps(ptre);
                    __m512 _rowf = _mm512_loadu_ps(ptrf);

                    transpose16x16_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7, _row8, _row9, _rowa, _rowb, _rowc, _rowd, _rowe, _rowf);

                    _mm512_storeu_ps(outptr, _row0);
                    _mm512_storeu_ps(outptr + 16, _row1);
                    _mm512_storeu_ps(outptr + 16 * 2, _row2);
                    _mm512_storeu_ps(outptr + 16 * 3, _row3);
                    _mm512_storeu_ps(outptr + 16 * 4, _row4);
                    _mm512_storeu_ps(outptr + 16 * 5, _row5);
                    _mm512_storeu_ps(outptr + 16 * 6, _row6);
                    _mm512_storeu_ps(outptr + 16 * 7, _row7);
                    _mm512_storeu_ps(outptr + 16 * 8, _row8);
                    _mm512_storeu_ps(outptr + 16 * 9, _row9);
                    _mm512_storeu_ps(outptr + 16 * 10, _rowa);
                    _mm512_storeu_ps(outptr + 16 * 11, _rowb);
                    _mm512_storeu_ps(outptr + 16 * 12, _rowc);
                    _mm512_storeu_ps(outptr + 16 * 13, _rowd);
                    _mm512_storeu_ps(outptr + 16 * 14, _rowe);
                    _mm512_storeu_ps(outptr + 16 * 15, _rowf);

                    ptr0 += 16;
                    ptr1 += 16;
                    ptr2 += 16;
                    ptr3 += 16;
                    ptr4 += 16;
                    ptr5 += 16;
                    ptr6 += 16;
                    ptr7 += 16;
                    ptr8 += 16;
                    ptr9 += 16;
                    ptra += 16;
                    ptrb += 16;
                    ptrc += 16;
                    ptrd += 16;
                    ptre += 16;
                    ptrf += 16;
                    outptr += 256;
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
                    outptr[8] = *ptr8++;
                    outptr[9] = *ptr9++;
                    outptr[10] = *ptra++;
                    outptr[11] = *ptrb++;
                    outptr[12] = *ptrc++;
                    outptr[13] = *ptrd++;
                    outptr[14] = *ptre++;
                    outptr[15] = *ptrf++;

                    outptr += 16;
                }
            }

            return 0;
        }
#endif // __AVX512F__

        if (scalar_elemsize == 4 && batch_axis == 0 && elempack == 1 && out_elempack == 8 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && top_blob.c * 8 == bottom_blob.c * bottom_blob.n)
        {
            const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const int bq0 = q * 8;
                const int b0 = bq0 / bottom_blob.c;
                const int b1 = (bq0 + 1) / bottom_blob.c;
                const int b2 = (bq0 + 2) / bottom_blob.c;
                const int b3 = (bq0 + 3) / bottom_blob.c;
                const int b4 = (bq0 + 4) / bottom_blob.c;
                const int b5 = (bq0 + 5) / bottom_blob.c;
                const int b6 = (bq0 + 6) / bottom_blob.c;
                const int b7 = (bq0 + 7) / bottom_blob.c;
                const int q0 = bq0 - b0 * bottom_blob.c;
                const int q1 = bq0 + 1 - b1 * bottom_blob.c;
                const int q2 = bq0 + 2 - b2 * bottom_blob.c;
                const int q3 = bq0 + 3 - b3 * bottom_blob.c;
                const int q4 = bq0 + 4 - b4 * bottom_blob.c;
                const int q5 = bq0 + 5 - b5 * bottom_blob.c;
                const int q6 = bq0 + 6 - b6 * bottom_blob.c;
                const int q7 = bq0 + 7 - b7 * bottom_blob.c;

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
                    __m256 _row0 = _mm256_loadu_ps(ptr0);
                    __m256 _row1 = _mm256_loadu_ps(ptr1);
                    __m256 _row2 = _mm256_loadu_ps(ptr2);
                    __m256 _row3 = _mm256_loadu_ps(ptr3);
                    __m256 _row4 = _mm256_loadu_ps(ptr4);
                    __m256 _row5 = _mm256_loadu_ps(ptr5);
                    __m256 _row6 = _mm256_loadu_ps(ptr6);
                    __m256 _row7 = _mm256_loadu_ps(ptr7);

                    transpose8x8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);

                    _mm256_storeu_ps(outptr, _row0);
                    _mm256_storeu_ps(outptr + 8, _row1);
                    _mm256_storeu_ps(outptr + 16, _row2);
                    _mm256_storeu_ps(outptr + 24, _row3);
                    _mm256_storeu_ps(outptr + 32, _row4);
                    _mm256_storeu_ps(outptr + 40, _row5);
                    _mm256_storeu_ps(outptr + 48, _row6);
                    _mm256_storeu_ps(outptr + 56, _row7);

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
#endif // __AVX__

        if (scalar_elemsize == 2 && batch_axis == 0 && elempack == 1 && out_elempack == 4 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && top_blob.c * 4 == bottom_blob.c * bottom_blob.n)
        {
            const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const int bq0 = q * 4;
                const int b0 = bq0 / bottom_blob.c;
                const int b1 = (bq0 + 1) / bottom_blob.c;
                const int b2 = (bq0 + 2) / bottom_blob.c;
                const int b3 = (bq0 + 3) / bottom_blob.c;
                const int q0 = bq0 - b0 * bottom_blob.c;
                const int q1 = bq0 + 1 - b1 * bottom_blob.c;
                const int q2 = bq0 + 2 - b2 * bottom_blob.c;
                const int q3 = bq0 + 3 - b3 * bottom_blob.c;

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

        if (scalar_elemsize == 4 && batch_axis == 0 && elempack == 1 && out_elempack == 4 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && top_blob.c * 4 == bottom_blob.c * bottom_blob.n)
        {
            const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const int bq0 = q * 4;
                const int b0 = bq0 / bottom_blob.c;
                const int b1 = (bq0 + 1) / bottom_blob.c;
                const int b2 = (bq0 + 2) / bottom_blob.c;
                const int b3 = (bq0 + 3) / bottom_blob.c;
                const int q0 = bq0 - b0 * bottom_blob.c;
                const int q1 = bq0 + 1 - b1 * bottom_blob.c;
                const int q2 = bq0 + 2 - b2 * bottom_blob.c;
                const int q3 = bq0 + 3 - b3 * bottom_blob.c;

                const float* ptr0 = (const float*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)q0 * bottom_blob.cstep;
                const float* ptr1 = (const float*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)q1 * bottom_blob.cstep;
                const float* ptr2 = (const float*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)q2 * bottom_blob.cstep;
                const float* ptr3 = (const float*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)q3 * bottom_blob.cstep;
                float* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    __m128 _row0 = _mm_loadu_ps(ptr0);
                    __m128 _row1 = _mm_loadu_ps(ptr1);
                    __m128 _row2 = _mm_loadu_ps(ptr2);
                    __m128 _row3 = _mm_loadu_ps(ptr3);

                    _MM_TRANSPOSE4_PS(_row0, _row1, _row2, _row3);

                    _mm_storeu_ps(outptr, _row0);
                    _mm_storeu_ps(outptr + 4, _row1);
                    _mm_storeu_ps(outptr + 8, _row2);
                    _mm_storeu_ps(outptr + 12, _row3);

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

        if (scalar_elemsize == 2 && batch_axis == 1 && elempack == 1 && out_elempack == 4 && dims == 3 && ndim == 4 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && top_blob.d == bottom_blob.n && top_blob.c * 4 == bottom_blob.c)
        {
            const int size = bottom_blob.w * bottom_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < bottom_blob.n * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;

                const unsigned short* ptr0 = (const unsigned short*)bottom_blob + (size_t)b * bottom_blob.nstep + (size_t)(q * 4) * bottom_blob.cstep;
                const unsigned short* ptr1 = ptr0 + bottom_blob.cstep;
                const unsigned short* ptr2 = ptr1 + bottom_blob.cstep;
                const unsigned short* ptr3 = ptr2 + bottom_blob.cstep;
                unsigned short* outptr = (unsigned short*)top_blob + ((size_t)q * top_blob.cstep + (size_t)b * top_blob.w * top_blob.h) * 4;

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

        if (scalar_elemsize == 4 && batch_axis == 1 && elempack == 1 && out_elempack == 4 && dims == 3 && ndim == 4 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && top_blob.d == bottom_blob.n && top_blob.c * 4 == bottom_blob.c)
        {
            const int size = bottom_blob.w * bottom_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < bottom_blob.n * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;

                const float* ptr0 = (const float*)bottom_blob + (size_t)b * bottom_blob.nstep + (size_t)(q * 4) * bottom_blob.cstep;
                const float* ptr1 = ptr0 + bottom_blob.cstep;
                const float* ptr2 = ptr1 + bottom_blob.cstep;
                const float* ptr3 = ptr2 + bottom_blob.cstep;
                float* outptr = (float*)top_blob + ((size_t)q * top_blob.cstep + (size_t)b * top_blob.w * top_blob.h) * 4;

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    __m128 _row0 = _mm_loadu_ps(ptr0);
                    __m128 _row1 = _mm_loadu_ps(ptr1);
                    __m128 _row2 = _mm_loadu_ps(ptr2);
                    __m128 _row3 = _mm_loadu_ps(ptr3);

                    _MM_TRANSPOSE4_PS(_row0, _row1, _row2, _row3);

                    _mm_storeu_ps(outptr, _row0);
                    _mm_storeu_ps(outptr + 4, _row1);
                    _mm_storeu_ps(outptr + 8, _row2);
                    _mm_storeu_ps(outptr + 12, _row3);

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
#endif // __SSE2__

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
            top_blob.create(outw / out_elempack, out_elemsize, out_elempack, batch, opt.blob_allocator);
        if (ndim == 2)
            top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, batch, opt.blob_allocator);
        if (ndim == 3)
            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, batch, opt.blob_allocator);
        if (ndim == 4)
            top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, batch, opt.blob_allocator);

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

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (scalar_elemsize == 4 && batch_axis == 0 && elempack == 1 && out_elempack == 16 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h * batch * 16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bi = 0; bi < batch * top_blob.h; bi++)
            {
                const int b = bi / top_blob.h;
                const int i = bi - b * top_blob.h;
                const int y = bi * 16;

                const float* ptr0 = bottom_blob.row(y);
                const float* ptr1 = bottom_blob.row(y + 1);
                const float* ptr2 = bottom_blob.row(y + 2);
                const float* ptr3 = bottom_blob.row(y + 3);
                const float* ptr4 = bottom_blob.row(y + 4);
                const float* ptr5 = bottom_blob.row(y + 5);
                const float* ptr6 = bottom_blob.row(y + 6);
                const float* ptr7 = bottom_blob.row(y + 7);
                const float* ptr8 = bottom_blob.row(y + 8);
                const float* ptr9 = bottom_blob.row(y + 9);
                const float* ptra = bottom_blob.row(y + 10);
                const float* ptrb = bottom_blob.row(y + 11);
                const float* ptrc = bottom_blob.row(y + 12);
                const float* ptrd = bottom_blob.row(y + 13);
                const float* ptre = bottom_blob.row(y + 14);
                const float* ptrf = bottom_blob.row(y + 15);
                float* outptr = (float*)top_blob + ((size_t)b * top_blob.nstep + (size_t)i * top_blob.w) * 16;

                int j = 0;
                for (; j + 15 < bottom_blob.w; j += 16)
                {
                    __m512 _row0 = _mm512_loadu_ps(ptr0);
                    __m512 _row1 = _mm512_loadu_ps(ptr1);
                    __m512 _row2 = _mm512_loadu_ps(ptr2);
                    __m512 _row3 = _mm512_loadu_ps(ptr3);
                    __m512 _row4 = _mm512_loadu_ps(ptr4);
                    __m512 _row5 = _mm512_loadu_ps(ptr5);
                    __m512 _row6 = _mm512_loadu_ps(ptr6);
                    __m512 _row7 = _mm512_loadu_ps(ptr7);
                    __m512 _row8 = _mm512_loadu_ps(ptr8);
                    __m512 _row9 = _mm512_loadu_ps(ptr9);
                    __m512 _rowa = _mm512_loadu_ps(ptra);
                    __m512 _rowb = _mm512_loadu_ps(ptrb);
                    __m512 _rowc = _mm512_loadu_ps(ptrc);
                    __m512 _rowd = _mm512_loadu_ps(ptrd);
                    __m512 _rowe = _mm512_loadu_ps(ptre);
                    __m512 _rowf = _mm512_loadu_ps(ptrf);

                    transpose16x16_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7, _row8, _row9, _rowa, _rowb, _rowc, _rowd, _rowe, _rowf);

                    _mm512_storeu_ps(outptr, _row0);
                    _mm512_storeu_ps(outptr + 16, _row1);
                    _mm512_storeu_ps(outptr + 16 * 2, _row2);
                    _mm512_storeu_ps(outptr + 16 * 3, _row3);
                    _mm512_storeu_ps(outptr + 16 * 4, _row4);
                    _mm512_storeu_ps(outptr + 16 * 5, _row5);
                    _mm512_storeu_ps(outptr + 16 * 6, _row6);
                    _mm512_storeu_ps(outptr + 16 * 7, _row7);
                    _mm512_storeu_ps(outptr + 16 * 8, _row8);
                    _mm512_storeu_ps(outptr + 16 * 9, _row9);
                    _mm512_storeu_ps(outptr + 16 * 10, _rowa);
                    _mm512_storeu_ps(outptr + 16 * 11, _rowb);
                    _mm512_storeu_ps(outptr + 16 * 12, _rowc);
                    _mm512_storeu_ps(outptr + 16 * 13, _rowd);
                    _mm512_storeu_ps(outptr + 16 * 14, _rowe);
                    _mm512_storeu_ps(outptr + 16 * 15, _rowf);

                    ptr0 += 16;
                    ptr1 += 16;
                    ptr2 += 16;
                    ptr3 += 16;
                    ptr4 += 16;
                    ptr5 += 16;
                    ptr6 += 16;
                    ptr7 += 16;
                    ptr8 += 16;
                    ptr9 += 16;
                    ptra += 16;
                    ptrb += 16;
                    ptrc += 16;
                    ptrd += 16;
                    ptre += 16;
                    ptrf += 16;
                    outptr += 256;
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
                    outptr[8] = *ptr8++;
                    outptr[9] = *ptr9++;
                    outptr[10] = *ptra++;
                    outptr[11] = *ptrb++;
                    outptr[12] = *ptrc++;
                    outptr[13] = *ptrd++;
                    outptr[14] = *ptre++;
                    outptr[15] = *ptrf++;

                    outptr += 16;
                }
            }

            return 0;
        }

        if (scalar_elemsize == 4 && batch_axis == 0 && elempack == 16 && out_elempack == 1 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h * 16 == top_blob.h * batch)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < bottom_blob.h; i++)
            {
                const int y = i * 16;
                const int b0 = y / top_blob.h;
                const int b1 = (y + 1) / top_blob.h;
                const int b2 = (y + 2) / top_blob.h;
                const int b3 = (y + 3) / top_blob.h;
                const int b4 = (y + 4) / top_blob.h;
                const int b5 = (y + 5) / top_blob.h;
                const int b6 = (y + 6) / top_blob.h;
                const int b7 = (y + 7) / top_blob.h;
                const int b8 = (y + 8) / top_blob.h;
                const int b9 = (y + 9) / top_blob.h;
                const int ba = (y + 10) / top_blob.h;
                const int bb = (y + 11) / top_blob.h;
                const int bc = (y + 12) / top_blob.h;
                const int bd = (y + 13) / top_blob.h;
                const int be = (y + 14) / top_blob.h;
                const int bf = (y + 15) / top_blob.h;
                const int y0 = y - b0 * top_blob.h;
                const int y1 = y + 1 - b1 * top_blob.h;
                const int y2 = y + 2 - b2 * top_blob.h;
                const int y3 = y + 3 - b3 * top_blob.h;
                const int y4 = y + 4 - b4 * top_blob.h;
                const int y5 = y + 5 - b5 * top_blob.h;
                const int y6 = y + 6 - b6 * top_blob.h;
                const int y7 = y + 7 - b7 * top_blob.h;
                const int y8 = y + 8 - b8 * top_blob.h;
                const int y9 = y + 9 - b9 * top_blob.h;
                const int ya = y + 10 - ba * top_blob.h;
                const int yb = y + 11 - bb * top_blob.h;
                const int yc = y + 12 - bc * top_blob.h;
                const int yd = y + 13 - bd * top_blob.h;
                const int ye = y + 14 - be * top_blob.h;
                const int yf = y + 15 - bf * top_blob.h;

                const float* ptr = bottom_blob.row(i);
                float* outptr0 = (float*)top_blob + (size_t)b0 * top_blob.nstep + (size_t)y0 * top_blob.w;
                float* outptr1 = (float*)top_blob + (size_t)b1 * top_blob.nstep + (size_t)y1 * top_blob.w;
                float* outptr2 = (float*)top_blob + (size_t)b2 * top_blob.nstep + (size_t)y2 * top_blob.w;
                float* outptr3 = (float*)top_blob + (size_t)b3 * top_blob.nstep + (size_t)y3 * top_blob.w;
                float* outptr4 = (float*)top_blob + (size_t)b4 * top_blob.nstep + (size_t)y4 * top_blob.w;
                float* outptr5 = (float*)top_blob + (size_t)b5 * top_blob.nstep + (size_t)y5 * top_blob.w;
                float* outptr6 = (float*)top_blob + (size_t)b6 * top_blob.nstep + (size_t)y6 * top_blob.w;
                float* outptr7 = (float*)top_blob + (size_t)b7 * top_blob.nstep + (size_t)y7 * top_blob.w;
                float* outptr8 = (float*)top_blob + (size_t)b8 * top_blob.nstep + (size_t)y8 * top_blob.w;
                float* outptr9 = (float*)top_blob + (size_t)b9 * top_blob.nstep + (size_t)y9 * top_blob.w;
                float* outptra = (float*)top_blob + (size_t)ba * top_blob.nstep + (size_t)ya * top_blob.w;
                float* outptrb = (float*)top_blob + (size_t)bb * top_blob.nstep + (size_t)yb * top_blob.w;
                float* outptrc = (float*)top_blob + (size_t)bc * top_blob.nstep + (size_t)yc * top_blob.w;
                float* outptrd = (float*)top_blob + (size_t)bd * top_blob.nstep + (size_t)yd * top_blob.w;
                float* outptre = (float*)top_blob + (size_t)be * top_blob.nstep + (size_t)ye * top_blob.w;
                float* outptrf = (float*)top_blob + (size_t)bf * top_blob.nstep + (size_t)yf * top_blob.w;

                int j = 0;
                for (; j + 15 < bottom_blob.w; j += 16)
                {
                    __m512 _row0 = _mm512_loadu_ps(ptr);
                    __m512 _row1 = _mm512_loadu_ps(ptr + 16);
                    __m512 _row2 = _mm512_loadu_ps(ptr + 16 * 2);
                    __m512 _row3 = _mm512_loadu_ps(ptr + 16 * 3);
                    __m512 _row4 = _mm512_loadu_ps(ptr + 16 * 4);
                    __m512 _row5 = _mm512_loadu_ps(ptr + 16 * 5);
                    __m512 _row6 = _mm512_loadu_ps(ptr + 16 * 6);
                    __m512 _row7 = _mm512_loadu_ps(ptr + 16 * 7);
                    __m512 _row8 = _mm512_loadu_ps(ptr + 16 * 8);
                    __m512 _row9 = _mm512_loadu_ps(ptr + 16 * 9);
                    __m512 _rowa = _mm512_loadu_ps(ptr + 16 * 10);
                    __m512 _rowb = _mm512_loadu_ps(ptr + 16 * 11);
                    __m512 _rowc = _mm512_loadu_ps(ptr + 16 * 12);
                    __m512 _rowd = _mm512_loadu_ps(ptr + 16 * 13);
                    __m512 _rowe = _mm512_loadu_ps(ptr + 16 * 14);
                    __m512 _rowf = _mm512_loadu_ps(ptr + 16 * 15);

                    transpose16x16_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7, _row8, _row9, _rowa, _rowb, _rowc, _rowd, _rowe, _rowf);

                    _mm512_storeu_ps(outptr0, _row0);
                    _mm512_storeu_ps(outptr1, _row1);
                    _mm512_storeu_ps(outptr2, _row2);
                    _mm512_storeu_ps(outptr3, _row3);
                    _mm512_storeu_ps(outptr4, _row4);
                    _mm512_storeu_ps(outptr5, _row5);
                    _mm512_storeu_ps(outptr6, _row6);
                    _mm512_storeu_ps(outptr7, _row7);
                    _mm512_storeu_ps(outptr8, _row8);
                    _mm512_storeu_ps(outptr9, _row9);
                    _mm512_storeu_ps(outptra, _rowa);
                    _mm512_storeu_ps(outptrb, _rowb);
                    _mm512_storeu_ps(outptrc, _rowc);
                    _mm512_storeu_ps(outptrd, _rowd);
                    _mm512_storeu_ps(outptre, _rowe);
                    _mm512_storeu_ps(outptrf, _rowf);

                    ptr += 256;
                    outptr0 += 16;
                    outptr1 += 16;
                    outptr2 += 16;
                    outptr3 += 16;
                    outptr4 += 16;
                    outptr5 += 16;
                    outptr6 += 16;
                    outptr7 += 16;
                    outptr8 += 16;
                    outptr9 += 16;
                    outptra += 16;
                    outptrb += 16;
                    outptrc += 16;
                    outptrd += 16;
                    outptre += 16;
                    outptrf += 16;
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
                    *outptr8++ = ptr[8];
                    *outptr9++ = ptr[9];
                    *outptra++ = ptr[10];
                    *outptrb++ = ptr[11];
                    *outptrc++ = ptr[12];
                    *outptrd++ = ptr[13];
                    *outptre++ = ptr[14];
                    *outptrf++ = ptr[15];

                    ptr += 16;
                }
            }

            return 0;
        }
#endif // __AVX512F__

        if (scalar_elemsize == 4 && batch_axis == 0 && elempack == 1 && out_elempack == 8 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h * batch * 8)
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
                    __m256 _row0 = _mm256_loadu_ps(ptr0);
                    __m256 _row1 = _mm256_loadu_ps(ptr1);
                    __m256 _row2 = _mm256_loadu_ps(ptr2);
                    __m256 _row3 = _mm256_loadu_ps(ptr3);
                    __m256 _row4 = _mm256_loadu_ps(ptr4);
                    __m256 _row5 = _mm256_loadu_ps(ptr5);
                    __m256 _row6 = _mm256_loadu_ps(ptr6);
                    __m256 _row7 = _mm256_loadu_ps(ptr7);

                    transpose8x8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);

                    _mm256_storeu_ps(outptr, _row0);
                    _mm256_storeu_ps(outptr + 8, _row1);
                    _mm256_storeu_ps(outptr + 16, _row2);
                    _mm256_storeu_ps(outptr + 24, _row3);
                    _mm256_storeu_ps(outptr + 32, _row4);
                    _mm256_storeu_ps(outptr + 40, _row5);
                    _mm256_storeu_ps(outptr + 48, _row6);
                    _mm256_storeu_ps(outptr + 56, _row7);

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

        if (scalar_elemsize == 4 && batch_axis == 0 && elempack == 8 && out_elempack == 1 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h * 8 == top_blob.h * batch)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < bottom_blob.h; i++)
            {
                const int y = i * 8;
                const int b0 = y / top_blob.h;
                const int b1 = (y + 1) / top_blob.h;
                const int b2 = (y + 2) / top_blob.h;
                const int b3 = (y + 3) / top_blob.h;
                const int b4 = (y + 4) / top_blob.h;
                const int b5 = (y + 5) / top_blob.h;
                const int b6 = (y + 6) / top_blob.h;
                const int b7 = (y + 7) / top_blob.h;
                const int y0 = y - b0 * top_blob.h;
                const int y1 = y + 1 - b1 * top_blob.h;
                const int y2 = y + 2 - b2 * top_blob.h;
                const int y3 = y + 3 - b3 * top_blob.h;
                const int y4 = y + 4 - b4 * top_blob.h;
                const int y5 = y + 5 - b5 * top_blob.h;
                const int y6 = y + 6 - b6 * top_blob.h;
                const int y7 = y + 7 - b7 * top_blob.h;

                const float* ptr = bottom_blob.row(i);
                float* outptr0 = (float*)top_blob + (size_t)b0 * top_blob.nstep + (size_t)y0 * top_blob.w;
                float* outptr1 = (float*)top_blob + (size_t)b1 * top_blob.nstep + (size_t)y1 * top_blob.w;
                float* outptr2 = (float*)top_blob + (size_t)b2 * top_blob.nstep + (size_t)y2 * top_blob.w;
                float* outptr3 = (float*)top_blob + (size_t)b3 * top_blob.nstep + (size_t)y3 * top_blob.w;
                float* outptr4 = (float*)top_blob + (size_t)b4 * top_blob.nstep + (size_t)y4 * top_blob.w;
                float* outptr5 = (float*)top_blob + (size_t)b5 * top_blob.nstep + (size_t)y5 * top_blob.w;
                float* outptr6 = (float*)top_blob + (size_t)b6 * top_blob.nstep + (size_t)y6 * top_blob.w;
                float* outptr7 = (float*)top_blob + (size_t)b7 * top_blob.nstep + (size_t)y7 * top_blob.w;

                int j = 0;
                for (; j + 7 < bottom_blob.w; j += 8)
                {
                    __m256 _row0 = _mm256_loadu_ps(ptr);
                    __m256 _row1 = _mm256_loadu_ps(ptr + 8);
                    __m256 _row2 = _mm256_loadu_ps(ptr + 16);
                    __m256 _row3 = _mm256_loadu_ps(ptr + 24);
                    __m256 _row4 = _mm256_loadu_ps(ptr + 32);
                    __m256 _row5 = _mm256_loadu_ps(ptr + 40);
                    __m256 _row6 = _mm256_loadu_ps(ptr + 48);
                    __m256 _row7 = _mm256_loadu_ps(ptr + 56);

                    transpose8x8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);

                    _mm256_storeu_ps(outptr0, _row0);
                    _mm256_storeu_ps(outptr1, _row1);
                    _mm256_storeu_ps(outptr2, _row2);
                    _mm256_storeu_ps(outptr3, _row3);
                    _mm256_storeu_ps(outptr4, _row4);
                    _mm256_storeu_ps(outptr5, _row5);
                    _mm256_storeu_ps(outptr6, _row6);
                    _mm256_storeu_ps(outptr7, _row7);

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
#endif // __AVX__

        if (scalar_elemsize == 4 && batch_axis == 0 && elempack == 1 && out_elempack == 4 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h * batch * 4)
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
                    __m128 _row0 = _mm_loadu_ps(ptr0);
                    __m128 _row1 = _mm_loadu_ps(ptr1);
                    __m128 _row2 = _mm_loadu_ps(ptr2);
                    __m128 _row3 = _mm_loadu_ps(ptr3);

                    _MM_TRANSPOSE4_PS(_row0, _row1, _row2, _row3);

                    _mm_storeu_ps(outptr, _row0);
                    _mm_storeu_ps(outptr + 4, _row1);
                    _mm_storeu_ps(outptr + 8, _row2);
                    _mm_storeu_ps(outptr + 12, _row3);

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

        if (scalar_elemsize == 4 && batch_axis == 0 && elempack == 4 && out_elempack == 1 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h * 4 == top_blob.h * batch)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < bottom_blob.h; i++)
            {
                const int y = i * 4;
                const int b0 = y / top_blob.h;
                const int b1 = (y + 1) / top_blob.h;
                const int b2 = (y + 2) / top_blob.h;
                const int b3 = (y + 3) / top_blob.h;
                const int y0 = y - b0 * top_blob.h;
                const int y1 = y + 1 - b1 * top_blob.h;
                const int y2 = y + 2 - b2 * top_blob.h;
                const int y3 = y + 3 - b3 * top_blob.h;

                const float* ptr = bottom_blob.row(i);
                float* outptr0 = (float*)top_blob + (size_t)b0 * top_blob.nstep + (size_t)y0 * top_blob.w;
                float* outptr1 = (float*)top_blob + (size_t)b1 * top_blob.nstep + (size_t)y1 * top_blob.w;
                float* outptr2 = (float*)top_blob + (size_t)b2 * top_blob.nstep + (size_t)y2 * top_blob.w;
                float* outptr3 = (float*)top_blob + (size_t)b3 * top_blob.nstep + (size_t)y3 * top_blob.w;

                int j = 0;
                for (; j + 3 < bottom_blob.w; j += 4)
                {
                    __m128 _row0 = _mm_loadu_ps(ptr);
                    __m128 _row1 = _mm_loadu_ps(ptr + 4);
                    __m128 _row2 = _mm_loadu_ps(ptr + 8);
                    __m128 _row3 = _mm_loadu_ps(ptr + 12);

                    _MM_TRANSPOSE4_PS(_row0, _row1, _row2, _row3);

                    _mm_storeu_ps(outptr0, _row0);
                    _mm_storeu_ps(outptr1, _row1);
                    _mm_storeu_ps(outptr2, _row2);
                    _mm_storeu_ps(outptr3, _row3);

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
#endif // __SSE2__

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

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (scalar_elemsize == 4 && batch_axis == 0 && elempack == 1 && out_elempack == 16 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c == top_blob.c * batch * 16)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;
                const int sq = b * top_blob.c * 16 + q * 16;

                const float* ptr0 = bottom_blob.channel(sq);
                const float* ptr1 = bottom_blob.channel(sq + 1);
                const float* ptr2 = bottom_blob.channel(sq + 2);
                const float* ptr3 = bottom_blob.channel(sq + 3);
                const float* ptr4 = bottom_blob.channel(sq + 4);
                const float* ptr5 = bottom_blob.channel(sq + 5);
                const float* ptr6 = bottom_blob.channel(sq + 6);
                const float* ptr7 = bottom_blob.channel(sq + 7);
                const float* ptr8 = bottom_blob.channel(sq + 8);
                const float* ptr9 = bottom_blob.channel(sq + 9);
                const float* ptra = bottom_blob.channel(sq + 10);
                const float* ptrb = bottom_blob.channel(sq + 11);
                const float* ptrc = bottom_blob.channel(sq + 12);
                const float* ptrd = bottom_blob.channel(sq + 13);
                const float* ptre = bottom_blob.channel(sq + 14);
                const float* ptrf = bottom_blob.channel(sq + 15);
                float* outptr = (float*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * 16;

                int i = 0;
                for (; i + 15 < size; i += 16)
                {
                    __m512 _row0 = _mm512_loadu_ps(ptr0);
                    __m512 _row1 = _mm512_loadu_ps(ptr1);
                    __m512 _row2 = _mm512_loadu_ps(ptr2);
                    __m512 _row3 = _mm512_loadu_ps(ptr3);
                    __m512 _row4 = _mm512_loadu_ps(ptr4);
                    __m512 _row5 = _mm512_loadu_ps(ptr5);
                    __m512 _row6 = _mm512_loadu_ps(ptr6);
                    __m512 _row7 = _mm512_loadu_ps(ptr7);
                    __m512 _row8 = _mm512_loadu_ps(ptr8);
                    __m512 _row9 = _mm512_loadu_ps(ptr9);
                    __m512 _rowa = _mm512_loadu_ps(ptra);
                    __m512 _rowb = _mm512_loadu_ps(ptrb);
                    __m512 _rowc = _mm512_loadu_ps(ptrc);
                    __m512 _rowd = _mm512_loadu_ps(ptrd);
                    __m512 _rowe = _mm512_loadu_ps(ptre);
                    __m512 _rowf = _mm512_loadu_ps(ptrf);

                    transpose16x16_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7, _row8, _row9, _rowa, _rowb, _rowc, _rowd, _rowe, _rowf);

                    _mm512_storeu_ps(outptr, _row0);
                    _mm512_storeu_ps(outptr + 16, _row1);
                    _mm512_storeu_ps(outptr + 16 * 2, _row2);
                    _mm512_storeu_ps(outptr + 16 * 3, _row3);
                    _mm512_storeu_ps(outptr + 16 * 4, _row4);
                    _mm512_storeu_ps(outptr + 16 * 5, _row5);
                    _mm512_storeu_ps(outptr + 16 * 6, _row6);
                    _mm512_storeu_ps(outptr + 16 * 7, _row7);
                    _mm512_storeu_ps(outptr + 16 * 8, _row8);
                    _mm512_storeu_ps(outptr + 16 * 9, _row9);
                    _mm512_storeu_ps(outptr + 16 * 10, _rowa);
                    _mm512_storeu_ps(outptr + 16 * 11, _rowb);
                    _mm512_storeu_ps(outptr + 16 * 12, _rowc);
                    _mm512_storeu_ps(outptr + 16 * 13, _rowd);
                    _mm512_storeu_ps(outptr + 16 * 14, _rowe);
                    _mm512_storeu_ps(outptr + 16 * 15, _rowf);

                    ptr0 += 16;
                    ptr1 += 16;
                    ptr2 += 16;
                    ptr3 += 16;
                    ptr4 += 16;
                    ptr5 += 16;
                    ptr6 += 16;
                    ptr7 += 16;
                    ptr8 += 16;
                    ptr9 += 16;
                    ptra += 16;
                    ptrb += 16;
                    ptrc += 16;
                    ptrd += 16;
                    ptre += 16;
                    ptrf += 16;
                    outptr += 256;
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
                    outptr[8] = *ptr8++;
                    outptr[9] = *ptr9++;
                    outptr[10] = *ptra++;
                    outptr[11] = *ptrb++;
                    outptr[12] = *ptrc++;
                    outptr[13] = *ptrd++;
                    outptr[14] = *ptre++;
                    outptr[15] = *ptrf++;

                    outptr += 16;
                }
            }

            return 0;
        }
#endif // __AVX512F__

        if (scalar_elemsize == 4 && batch_axis == 0 && elempack == 1 && out_elempack == 8 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c == top_blob.c * batch * 8)
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
                    __m256 _row0 = _mm256_loadu_ps(ptr0);
                    __m256 _row1 = _mm256_loadu_ps(ptr1);
                    __m256 _row2 = _mm256_loadu_ps(ptr2);
                    __m256 _row3 = _mm256_loadu_ps(ptr3);
                    __m256 _row4 = _mm256_loadu_ps(ptr4);
                    __m256 _row5 = _mm256_loadu_ps(ptr5);
                    __m256 _row6 = _mm256_loadu_ps(ptr6);
                    __m256 _row7 = _mm256_loadu_ps(ptr7);

                    transpose8x8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);

                    _mm256_storeu_ps(outptr, _row0);
                    _mm256_storeu_ps(outptr + 8, _row1);
                    _mm256_storeu_ps(outptr + 16, _row2);
                    _mm256_storeu_ps(outptr + 24, _row3);
                    _mm256_storeu_ps(outptr + 32, _row4);
                    _mm256_storeu_ps(outptr + 40, _row5);
                    _mm256_storeu_ps(outptr + 48, _row6);
                    _mm256_storeu_ps(outptr + 56, _row7);

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
#endif // __AVX__

        if (scalar_elemsize == 2 && batch_axis == 0 && elempack == 1 && out_elempack == 4 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c == top_blob.c * batch * 4)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;
                const int sq = b * top_blob.c * 4 + q * 4;

                const unsigned short* ptr0 = bottom_blob.channel(sq);
                const unsigned short* ptr1 = bottom_blob.channel(sq + 1);
                const unsigned short* ptr2 = bottom_blob.channel(sq + 2);
                const unsigned short* ptr3 = bottom_blob.channel(sq + 3);
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

        if (scalar_elemsize == 4 && batch_axis == 0 && elempack == 1 && out_elempack == 4 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c == top_blob.c * batch * 4)
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
                    __m128 _row0 = _mm_loadu_ps(ptr0);
                    __m128 _row1 = _mm_loadu_ps(ptr1);
                    __m128 _row2 = _mm_loadu_ps(ptr2);
                    __m128 _row3 = _mm_loadu_ps(ptr3);

                    _MM_TRANSPOSE4_PS(_row0, _row1, _row2, _row3);

                    _mm_storeu_ps(outptr, _row0);
                    _mm_storeu_ps(outptr + 4, _row1);
                    _mm_storeu_ps(outptr + 8, _row2);
                    _mm_storeu_ps(outptr + 12, _row3);

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

        if (scalar_elemsize == 2 && batch_axis == 1 && elempack == 1 && out_elempack == 4 && dims == 4 && ndim == 3 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == batch && bottom_blob.c == top_blob.c * 4)
        {
            const int size = top_blob.w * top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;

                const unsigned short* ptr0 = (const unsigned short*)bottom_blob + ((size_t)(q * 4) * bottom_blob.cstep + (size_t)b * bottom_blob.w * bottom_blob.h);
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

        if (scalar_elemsize == 4 && batch_axis == 1 && elempack == 1 && out_elempack == 4 && dims == 4 && ndim == 3 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == batch && bottom_blob.c == top_blob.c * 4)
        {
            const int size = top_blob.w * top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;

                const float* ptr0 = (const float*)bottom_blob + ((size_t)(q * 4) * bottom_blob.cstep + (size_t)b * bottom_blob.w * bottom_blob.h);
                const float* ptr1 = ptr0 + bottom_blob.cstep;
                const float* ptr2 = ptr1 + bottom_blob.cstep;
                const float* ptr3 = ptr2 + bottom_blob.cstep;
                float* outptr = (float*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * 4;

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    __m128 _row0 = _mm_loadu_ps(ptr0);
                    __m128 _row1 = _mm_loadu_ps(ptr1);
                    __m128 _row2 = _mm_loadu_ps(ptr2);
                    __m128 _row3 = _mm_loadu_ps(ptr3);

                    _MM_TRANSPOSE4_PS(_row0, _row1, _row2, _row3);

                    _mm_storeu_ps(outptr, _row0);
                    _mm_storeu_ps(outptr + 4, _row1);
                    _mm_storeu_ps(outptr + 8, _row2);
                    _mm_storeu_ps(outptr + 12, _row3);

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

#if __AVX__
#if __AVX512F__
        if (scalar_elemsize == 4 && batch_axis == 0 && elempack == 16 && out_elempack == 1 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c * 16 == top_blob.c * batch)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < bottom_blob.c; q++)
            {
                const int bq0 = q * 16;
                const int b0 = bq0 / top_blob.c;
                const int b1 = (bq0 + 1) / top_blob.c;
                const int b2 = (bq0 + 2) / top_blob.c;
                const int b3 = (bq0 + 3) / top_blob.c;
                const int b4 = (bq0 + 4) / top_blob.c;
                const int b5 = (bq0 + 5) / top_blob.c;
                const int b6 = (bq0 + 6) / top_blob.c;
                const int b7 = (bq0 + 7) / top_blob.c;
                const int b8 = (bq0 + 8) / top_blob.c;
                const int b9 = (bq0 + 9) / top_blob.c;
                const int ba = (bq0 + 10) / top_blob.c;
                const int bb = (bq0 + 11) / top_blob.c;
                const int bc = (bq0 + 12) / top_blob.c;
                const int bd = (bq0 + 13) / top_blob.c;
                const int be = (bq0 + 14) / top_blob.c;
                const int bf = (bq0 + 15) / top_blob.c;
                const int q0 = bq0 - b0 * top_blob.c;
                const int q1 = bq0 + 1 - b1 * top_blob.c;
                const int q2 = bq0 + 2 - b2 * top_blob.c;
                const int q3 = bq0 + 3 - b3 * top_blob.c;
                const int q4 = bq0 + 4 - b4 * top_blob.c;
                const int q5 = bq0 + 5 - b5 * top_blob.c;
                const int q6 = bq0 + 6 - b6 * top_blob.c;
                const int q7 = bq0 + 7 - b7 * top_blob.c;
                const int q8 = bq0 + 8 - b8 * top_blob.c;
                const int q9 = bq0 + 9 - b9 * top_blob.c;
                const int qa = bq0 + 10 - ba * top_blob.c;
                const int qb = bq0 + 11 - bb * top_blob.c;
                const int qc = bq0 + 12 - bc * top_blob.c;
                const int qd = bq0 + 13 - bd * top_blob.c;
                const int qe = bq0 + 14 - be * top_blob.c;
                const int qf = bq0 + 15 - bf * top_blob.c;

                const float* ptr = bottom_blob.channel(q);
                float* outptr0 = (float*)top_blob + (size_t)b0 * top_blob.nstep + (size_t)q0 * top_blob.cstep;
                float* outptr1 = (float*)top_blob + (size_t)b1 * top_blob.nstep + (size_t)q1 * top_blob.cstep;
                float* outptr2 = (float*)top_blob + (size_t)b2 * top_blob.nstep + (size_t)q2 * top_blob.cstep;
                float* outptr3 = (float*)top_blob + (size_t)b3 * top_blob.nstep + (size_t)q3 * top_blob.cstep;
                float* outptr4 = (float*)top_blob + (size_t)b4 * top_blob.nstep + (size_t)q4 * top_blob.cstep;
                float* outptr5 = (float*)top_blob + (size_t)b5 * top_blob.nstep + (size_t)q5 * top_blob.cstep;
                float* outptr6 = (float*)top_blob + (size_t)b6 * top_blob.nstep + (size_t)q6 * top_blob.cstep;
                float* outptr7 = (float*)top_blob + (size_t)b7 * top_blob.nstep + (size_t)q7 * top_blob.cstep;
                float* outptr8 = (float*)top_blob + (size_t)b8 * top_blob.nstep + (size_t)q8 * top_blob.cstep;
                float* outptr9 = (float*)top_blob + (size_t)b9 * top_blob.nstep + (size_t)q9 * top_blob.cstep;
                float* outptra = (float*)top_blob + (size_t)ba * top_blob.nstep + (size_t)qa * top_blob.cstep;
                float* outptrb = (float*)top_blob + (size_t)bb * top_blob.nstep + (size_t)qb * top_blob.cstep;
                float* outptrc = (float*)top_blob + (size_t)bc * top_blob.nstep + (size_t)qc * top_blob.cstep;
                float* outptrd = (float*)top_blob + (size_t)bd * top_blob.nstep + (size_t)qd * top_blob.cstep;
                float* outptre = (float*)top_blob + (size_t)be * top_blob.nstep + (size_t)qe * top_blob.cstep;
                float* outptrf = (float*)top_blob + (size_t)bf * top_blob.nstep + (size_t)qf * top_blob.cstep;

                int i = 0;
                for (; i + 15 < size; i += 16)
                {
                    __m512 _row0 = _mm512_loadu_ps(ptr);
                    __m512 _row1 = _mm512_loadu_ps(ptr + 16);
                    __m512 _row2 = _mm512_loadu_ps(ptr + 16 * 2);
                    __m512 _row3 = _mm512_loadu_ps(ptr + 16 * 3);
                    __m512 _row4 = _mm512_loadu_ps(ptr + 16 * 4);
                    __m512 _row5 = _mm512_loadu_ps(ptr + 16 * 5);
                    __m512 _row6 = _mm512_loadu_ps(ptr + 16 * 6);
                    __m512 _row7 = _mm512_loadu_ps(ptr + 16 * 7);
                    __m512 _row8 = _mm512_loadu_ps(ptr + 16 * 8);
                    __m512 _row9 = _mm512_loadu_ps(ptr + 16 * 9);
                    __m512 _rowa = _mm512_loadu_ps(ptr + 16 * 10);
                    __m512 _rowb = _mm512_loadu_ps(ptr + 16 * 11);
                    __m512 _rowc = _mm512_loadu_ps(ptr + 16 * 12);
                    __m512 _rowd = _mm512_loadu_ps(ptr + 16 * 13);
                    __m512 _rowe = _mm512_loadu_ps(ptr + 16 * 14);
                    __m512 _rowf = _mm512_loadu_ps(ptr + 16 * 15);

                    transpose16x16_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7, _row8, _row9, _rowa, _rowb, _rowc, _rowd, _rowe, _rowf);

                    _mm512_storeu_ps(outptr0, _row0);
                    _mm512_storeu_ps(outptr1, _row1);
                    _mm512_storeu_ps(outptr2, _row2);
                    _mm512_storeu_ps(outptr3, _row3);
                    _mm512_storeu_ps(outptr4, _row4);
                    _mm512_storeu_ps(outptr5, _row5);
                    _mm512_storeu_ps(outptr6, _row6);
                    _mm512_storeu_ps(outptr7, _row7);
                    _mm512_storeu_ps(outptr8, _row8);
                    _mm512_storeu_ps(outptr9, _row9);
                    _mm512_storeu_ps(outptra, _rowa);
                    _mm512_storeu_ps(outptrb, _rowb);
                    _mm512_storeu_ps(outptrc, _rowc);
                    _mm512_storeu_ps(outptrd, _rowd);
                    _mm512_storeu_ps(outptre, _rowe);
                    _mm512_storeu_ps(outptrf, _rowf);

                    ptr += 256;
                    outptr0 += 16;
                    outptr1 += 16;
                    outptr2 += 16;
                    outptr3 += 16;
                    outptr4 += 16;
                    outptr5 += 16;
                    outptr6 += 16;
                    outptr7 += 16;
                    outptr8 += 16;
                    outptr9 += 16;
                    outptra += 16;
                    outptrb += 16;
                    outptrc += 16;
                    outptrd += 16;
                    outptre += 16;
                    outptrf += 16;
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
                    *outptr8++ = ptr[8];
                    *outptr9++ = ptr[9];
                    *outptra++ = ptr[10];
                    *outptrb++ = ptr[11];
                    *outptrc++ = ptr[12];
                    *outptrd++ = ptr[13];
                    *outptre++ = ptr[14];
                    *outptrf++ = ptr[15];

                    ptr += 16;
                }
            }

            return 0;
        }
#endif // __AVX512F__

        if (scalar_elemsize == 4 && batch_axis == 0 && elempack == 8 && out_elempack == 1 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c * 8 == top_blob.c * batch)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < bottom_blob.c; q++)
            {
                const int bq0 = q * 8;
                const int b0 = bq0 / top_blob.c;
                const int b1 = (bq0 + 1) / top_blob.c;
                const int b2 = (bq0 + 2) / top_blob.c;
                const int b3 = (bq0 + 3) / top_blob.c;
                const int b4 = (bq0 + 4) / top_blob.c;
                const int b5 = (bq0 + 5) / top_blob.c;
                const int b6 = (bq0 + 6) / top_blob.c;
                const int b7 = (bq0 + 7) / top_blob.c;
                const int q0 = bq0 - b0 * top_blob.c;
                const int q1 = bq0 + 1 - b1 * top_blob.c;
                const int q2 = bq0 + 2 - b2 * top_blob.c;
                const int q3 = bq0 + 3 - b3 * top_blob.c;
                const int q4 = bq0 + 4 - b4 * top_blob.c;
                const int q5 = bq0 + 5 - b5 * top_blob.c;
                const int q6 = bq0 + 6 - b6 * top_blob.c;
                const int q7 = bq0 + 7 - b7 * top_blob.c;

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
                    __m256 _row0 = _mm256_loadu_ps(ptr);
                    __m256 _row1 = _mm256_loadu_ps(ptr + 8);
                    __m256 _row2 = _mm256_loadu_ps(ptr + 16);
                    __m256 _row3 = _mm256_loadu_ps(ptr + 24);
                    __m256 _row4 = _mm256_loadu_ps(ptr + 32);
                    __m256 _row5 = _mm256_loadu_ps(ptr + 40);
                    __m256 _row6 = _mm256_loadu_ps(ptr + 48);
                    __m256 _row7 = _mm256_loadu_ps(ptr + 56);

                    transpose8x8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);

                    _mm256_storeu_ps(outptr0, _row0);
                    _mm256_storeu_ps(outptr1, _row1);
                    _mm256_storeu_ps(outptr2, _row2);
                    _mm256_storeu_ps(outptr3, _row3);
                    _mm256_storeu_ps(outptr4, _row4);
                    _mm256_storeu_ps(outptr5, _row5);
                    _mm256_storeu_ps(outptr6, _row6);
                    _mm256_storeu_ps(outptr7, _row7);

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
#endif // __AVX__

        if (scalar_elemsize == 2 && batch_axis == 0 && elempack == 4 && out_elempack == 1 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c * 4 == top_blob.c * batch)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < bottom_blob.c; q++)
            {
                const int bq0 = q * 4;
                const int b0 = bq0 / top_blob.c;
                const int b1 = (bq0 + 1) / top_blob.c;
                const int b2 = (bq0 + 2) / top_blob.c;
                const int b3 = (bq0 + 3) / top_blob.c;
                const int q0 = bq0 - b0 * top_blob.c;
                const int q1 = bq0 + 1 - b1 * top_blob.c;
                const int q2 = bq0 + 2 - b2 * top_blob.c;
                const int q3 = bq0 + 3 - b3 * top_blob.c;

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

        if (scalar_elemsize == 4 && batch_axis == 0 && elempack == 4 && out_elempack == 1 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c * 4 == top_blob.c * batch)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < bottom_blob.c; q++)
            {
                const int bq0 = q * 4;
                const int b0 = bq0 / top_blob.c;
                const int b1 = (bq0 + 1) / top_blob.c;
                const int b2 = (bq0 + 2) / top_blob.c;
                const int b3 = (bq0 + 3) / top_blob.c;
                const int q0 = bq0 - b0 * top_blob.c;
                const int q1 = bq0 + 1 - b1 * top_blob.c;
                const int q2 = bq0 + 2 - b2 * top_blob.c;
                const int q3 = bq0 + 3 - b3 * top_blob.c;

                const float* ptr = bottom_blob.channel(q);
                float* outptr0 = (float*)top_blob + (size_t)b0 * top_blob.nstep + (size_t)q0 * top_blob.cstep;
                float* outptr1 = (float*)top_blob + (size_t)b1 * top_blob.nstep + (size_t)q1 * top_blob.cstep;
                float* outptr2 = (float*)top_blob + (size_t)b2 * top_blob.nstep + (size_t)q2 * top_blob.cstep;
                float* outptr3 = (float*)top_blob + (size_t)b3 * top_blob.nstep + (size_t)q3 * top_blob.cstep;

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    __m128 _row0 = _mm_loadu_ps(ptr);
                    __m128 _row1 = _mm_loadu_ps(ptr + 4);
                    __m128 _row2 = _mm_loadu_ps(ptr + 8);
                    __m128 _row3 = _mm_loadu_ps(ptr + 12);

                    _MM_TRANSPOSE4_PS(_row0, _row1, _row2, _row3);

                    _mm_storeu_ps(outptr0, _row0);
                    _mm_storeu_ps(outptr1, _row1);
                    _mm_storeu_ps(outptr2, _row2);
                    _mm_storeu_ps(outptr3, _row3);

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

        if (scalar_elemsize == 2 && batch_axis == 1 && elempack == 4 && out_elempack == 1 && dims == 4 && ndim == 3 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == batch && bottom_blob.c * 4 == top_blob.c)
        {
            const int size = top_blob.w * top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * bottom_blob.c; bq++)
            {
                const int b = bq / bottom_blob.c;
                const int q = bq - b * bottom_blob.c;

                const unsigned short* ptr = (const unsigned short*)bottom_blob + ((size_t)q * bottom_blob.cstep + (size_t)b * bottom_blob.w * bottom_blob.h) * 4;
                unsigned short* outptr0 = (unsigned short*)top_blob + (size_t)b * top_blob.nstep + (size_t)(q * 4) * top_blob.cstep;
                unsigned short* outptr1 = outptr0 + top_blob.cstep;
                unsigned short* outptr2 = outptr1 + top_blob.cstep;
                unsigned short* outptr3 = outptr2 + top_blob.cstep;

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

        if (scalar_elemsize == 4 && batch_axis == 1 && elempack == 4 && out_elempack == 1 && dims == 4 && ndim == 3 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == batch && bottom_blob.c * 4 == top_blob.c)
        {
            const int size = top_blob.w * top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * bottom_blob.c; bq++)
            {
                const int b = bq / bottom_blob.c;
                const int q = bq - b * bottom_blob.c;

                const float* ptr = (const float*)bottom_blob + ((size_t)q * bottom_blob.cstep + (size_t)b * bottom_blob.w * bottom_blob.h) * 4;
                float* outptr0 = (float*)top_blob + (size_t)b * top_blob.nstep + (size_t)(q * 4) * top_blob.cstep;
                float* outptr1 = outptr0 + top_blob.cstep;
                float* outptr2 = outptr1 + top_blob.cstep;
                float* outptr3 = outptr2 + top_blob.cstep;

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    __m128 _row0 = _mm_loadu_ps(ptr);
                    __m128 _row1 = _mm_loadu_ps(ptr + 4);
                    __m128 _row2 = _mm_loadu_ps(ptr + 8);
                    __m128 _row3 = _mm_loadu_ps(ptr + 12);

                    _MM_TRANSPOSE4_PS(_row0, _row1, _row2, _row3);

                    _mm_storeu_ps(outptr0, _row0);
                    _mm_storeu_ps(outptr1, _row1);
                    _mm_storeu_ps(outptr2, _row2);
                    _mm_storeu_ps(outptr3, _row3);

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
#endif // __SSE2__

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
#endif // NCNN_BATCH

int Reshape_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

#if NCNN_BATCH
    if (batch_mode != 0)
        return forward_batch(bottom_blobs, top_blobs, opt);
#endif

    int elembits = bottom_blob.elembits();

    if (elembits == 16)
        return forward_bf16s_fp16s(bottom_blobs, top_blobs, opt);

    // resolve out shape
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
        // flatten
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
#if __SSE2__
        if (opt.use_packing_layout)
        {
#if __AVX512F__
            out_elempack = outh % 16 == 0 ? 16 : outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#elif __AVX__
            out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
            out_elempack = outh % 4 == 0 ? 4 : 1;
#endif
        }
#endif // __SSE2__
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 2 && bottom_blob.h * elempack == outh && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        if (out_elempack == 1)
        {
            // flatten
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

        // flatten
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

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (out_elempack == 16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + outw * i * 16;
                const float* ptr1 = (const float*)bottom_blob_flattened + outw * (i * 16 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + outw * (i * 16 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + outw * (i * 16 + 3);
                const float* ptr4 = (const float*)bottom_blob_flattened + outw * (i * 16 + 4);
                const float* ptr5 = (const float*)bottom_blob_flattened + outw * (i * 16 + 5);
                const float* ptr6 = (const float*)bottom_blob_flattened + outw * (i * 16 + 6);
                const float* ptr7 = (const float*)bottom_blob_flattened + outw * (i * 16 + 7);
                const float* ptr8 = (const float*)bottom_blob_flattened + outw * (i * 16 + 8);
                const float* ptr9 = (const float*)bottom_blob_flattened + outw * (i * 16 + 9);
                const float* ptra = (const float*)bottom_blob_flattened + outw * (i * 16 + 10);
                const float* ptrb = (const float*)bottom_blob_flattened + outw * (i * 16 + 11);
                const float* ptrc = (const float*)bottom_blob_flattened + outw * (i * 16 + 12);
                const float* ptrd = (const float*)bottom_blob_flattened + outw * (i * 16 + 13);
                const float* ptre = (const float*)bottom_blob_flattened + outw * (i * 16 + 14);
                const float* ptrf = (const float*)bottom_blob_flattened + outw * (i * 16 + 15);
                float* outptr = top_blob.row(i);

                int j = 0;
                for (; j + 15 < outw; j += 16)
                {
                    __m512 _row0 = _mm512_loadu_ps(ptr0);
                    __m512 _row1 = _mm512_loadu_ps(ptr1);
                    __m512 _row2 = _mm512_loadu_ps(ptr2);
                    __m512 _row3 = _mm512_loadu_ps(ptr3);
                    __m512 _row4 = _mm512_loadu_ps(ptr4);
                    __m512 _row5 = _mm512_loadu_ps(ptr5);
                    __m512 _row6 = _mm512_loadu_ps(ptr6);
                    __m512 _row7 = _mm512_loadu_ps(ptr7);
                    __m512 _row8 = _mm512_loadu_ps(ptr8);
                    __m512 _row9 = _mm512_loadu_ps(ptr9);
                    __m512 _rowa = _mm512_loadu_ps(ptra);
                    __m512 _rowb = _mm512_loadu_ps(ptrb);
                    __m512 _rowc = _mm512_loadu_ps(ptrc);
                    __m512 _rowd = _mm512_loadu_ps(ptrd);
                    __m512 _rowe = _mm512_loadu_ps(ptre);
                    __m512 _rowf = _mm512_loadu_ps(ptrf);

                    transpose16x16_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7, _row8, _row9, _rowa, _rowb, _rowc, _rowd, _rowe, _rowf);

                    _mm512_storeu_ps(outptr, _row0);
                    _mm512_storeu_ps(outptr + 16, _row1);
                    _mm512_storeu_ps(outptr + 16 * 2, _row2);
                    _mm512_storeu_ps(outptr + 16 * 3, _row3);
                    _mm512_storeu_ps(outptr + 16 * 4, _row4);
                    _mm512_storeu_ps(outptr + 16 * 5, _row5);
                    _mm512_storeu_ps(outptr + 16 * 6, _row6);
                    _mm512_storeu_ps(outptr + 16 * 7, _row7);
                    _mm512_storeu_ps(outptr + 16 * 8, _row8);
                    _mm512_storeu_ps(outptr + 16 * 9, _row9);
                    _mm512_storeu_ps(outptr + 16 * 10, _rowa);
                    _mm512_storeu_ps(outptr + 16 * 11, _rowb);
                    _mm512_storeu_ps(outptr + 16 * 12, _rowc);
                    _mm512_storeu_ps(outptr + 16 * 13, _rowd);
                    _mm512_storeu_ps(outptr + 16 * 14, _rowe);
                    _mm512_storeu_ps(outptr + 16 * 15, _rowf);

                    ptr0 += 16;
                    ptr1 += 16;
                    ptr2 += 16;
                    ptr3 += 16;
                    ptr4 += 16;
                    ptr5 += 16;
                    ptr6 += 16;
                    ptr7 += 16;
                    ptr8 += 16;
                    ptr9 += 16;
                    ptra += 16;
                    ptrb += 16;
                    ptrc += 16;
                    ptrd += 16;
                    ptre += 16;
                    ptrf += 16;
                    outptr += 256;
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
                    outptr[8] = *ptr8++;
                    outptr[9] = *ptr9++;
                    outptr[10] = *ptra++;
                    outptr[11] = *ptrb++;
                    outptr[12] = *ptrc++;
                    outptr[13] = *ptrd++;
                    outptr[14] = *ptre++;
                    outptr[15] = *ptrf++;

                    outptr += 16;
                }
            }
        }
#endif // __AVX512F__

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
                    __m256 _row0 = _mm256_loadu_ps(ptr0);
                    __m256 _row1 = _mm256_loadu_ps(ptr1);
                    __m256 _row2 = _mm256_loadu_ps(ptr2);
                    __m256 _row3 = _mm256_loadu_ps(ptr3);
                    __m256 _row4 = _mm256_loadu_ps(ptr4);
                    __m256 _row5 = _mm256_loadu_ps(ptr5);
                    __m256 _row6 = _mm256_loadu_ps(ptr6);
                    __m256 _row7 = _mm256_loadu_ps(ptr7);

                    transpose8x8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);

                    _mm256_storeu_ps(outptr, _row0);
                    _mm256_storeu_ps(outptr + 8, _row1);
                    _mm256_storeu_ps(outptr + 16, _row2);
                    _mm256_storeu_ps(outptr + 24, _row3);
                    _mm256_storeu_ps(outptr + 32, _row4);
                    _mm256_storeu_ps(outptr + 40, _row5);
                    _mm256_storeu_ps(outptr + 48, _row6);
                    _mm256_storeu_ps(outptr + 56, _row7);

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
        }
#endif // __AVX__

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
                    __m128 _row0 = _mm_loadu_ps(ptr0);
                    __m128 _row1 = _mm_loadu_ps(ptr1);
                    __m128 _row2 = _mm_loadu_ps(ptr2);
                    __m128 _row3 = _mm_loadu_ps(ptr3);

                    _MM_TRANSPOSE4_PS(_row0, _row1, _row2, _row3);

                    _mm_storeu_ps(outptr, _row0);
                    _mm_storeu_ps(outptr + 4, _row1);
                    _mm_storeu_ps(outptr + 8, _row2);
                    _mm_storeu_ps(outptr + 12, _row3);

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
        }
#endif // __SSE2__
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
#if __SSE2__
        if (opt.use_packing_layout)
        {
#if __AVX512F__
            out_elempack = outc % 16 == 0 ? 16 : outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#elif __AVX__
            out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#else
            out_elempack = outc % 4 == 0 ? 4 : 1;
#endif
        }
#endif // __SSE2__
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if ((dims == 3 || dims == 4) && bottom_blob.c * elempack == outc && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            top_blob.dims = ndim;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.d = outd;
            return 0;
        }

        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        if (ndim == 3)
        {
            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        }
        else // if (ndim == 4)
        {
            top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        }
        if (top_blob.empty())
            return -100;

        int size = top_blob.w * top_blob.h * top_blob.d;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (out_elempack == 16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + size * q * 16;
                const float* ptr1 = (const float*)bottom_blob_flattened + size * (q * 16 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + size * (q * 16 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + size * (q * 16 + 3);
                const float* ptr4 = (const float*)bottom_blob_flattened + size * (q * 16 + 4);
                const float* ptr5 = (const float*)bottom_blob_flattened + size * (q * 16 + 5);
                const float* ptr6 = (const float*)bottom_blob_flattened + size * (q * 16 + 6);
                const float* ptr7 = (const float*)bottom_blob_flattened + size * (q * 16 + 7);
                const float* ptr8 = (const float*)bottom_blob_flattened + size * (q * 16 + 8);
                const float* ptr9 = (const float*)bottom_blob_flattened + size * (q * 16 + 9);
                const float* ptra = (const float*)bottom_blob_flattened + size * (q * 16 + 10);
                const float* ptrb = (const float*)bottom_blob_flattened + size * (q * 16 + 11);
                const float* ptrc = (const float*)bottom_blob_flattened + size * (q * 16 + 12);
                const float* ptrd = (const float*)bottom_blob_flattened + size * (q * 16 + 13);
                const float* ptre = (const float*)bottom_blob_flattened + size * (q * 16 + 14);
                const float* ptrf = (const float*)bottom_blob_flattened + size * (q * 16 + 15);
                float* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 15 < size; i += 16)
                {
                    __m512 _row0 = _mm512_loadu_ps(ptr0);
                    __m512 _row1 = _mm512_loadu_ps(ptr1);
                    __m512 _row2 = _mm512_loadu_ps(ptr2);
                    __m512 _row3 = _mm512_loadu_ps(ptr3);
                    __m512 _row4 = _mm512_loadu_ps(ptr4);
                    __m512 _row5 = _mm512_loadu_ps(ptr5);
                    __m512 _row6 = _mm512_loadu_ps(ptr6);
                    __m512 _row7 = _mm512_loadu_ps(ptr7);
                    __m512 _row8 = _mm512_loadu_ps(ptr8);
                    __m512 _row9 = _mm512_loadu_ps(ptr9);
                    __m512 _rowa = _mm512_loadu_ps(ptra);
                    __m512 _rowb = _mm512_loadu_ps(ptrb);
                    __m512 _rowc = _mm512_loadu_ps(ptrc);
                    __m512 _rowd = _mm512_loadu_ps(ptrd);
                    __m512 _rowe = _mm512_loadu_ps(ptre);
                    __m512 _rowf = _mm512_loadu_ps(ptrf);

                    transpose16x16_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7, _row8, _row9, _rowa, _rowb, _rowc, _rowd, _rowe, _rowf);

                    _mm512_storeu_ps(outptr, _row0);
                    _mm512_storeu_ps(outptr + 16, _row1);
                    _mm512_storeu_ps(outptr + 16 * 2, _row2);
                    _mm512_storeu_ps(outptr + 16 * 3, _row3);
                    _mm512_storeu_ps(outptr + 16 * 4, _row4);
                    _mm512_storeu_ps(outptr + 16 * 5, _row5);
                    _mm512_storeu_ps(outptr + 16 * 6, _row6);
                    _mm512_storeu_ps(outptr + 16 * 7, _row7);
                    _mm512_storeu_ps(outptr + 16 * 8, _row8);
                    _mm512_storeu_ps(outptr + 16 * 9, _row9);
                    _mm512_storeu_ps(outptr + 16 * 10, _rowa);
                    _mm512_storeu_ps(outptr + 16 * 11, _rowb);
                    _mm512_storeu_ps(outptr + 16 * 12, _rowc);
                    _mm512_storeu_ps(outptr + 16 * 13, _rowd);
                    _mm512_storeu_ps(outptr + 16 * 14, _rowe);
                    _mm512_storeu_ps(outptr + 16 * 15, _rowf);

                    ptr0 += 16;
                    ptr1 += 16;
                    ptr2 += 16;
                    ptr3 += 16;
                    ptr4 += 16;
                    ptr5 += 16;
                    ptr6 += 16;
                    ptr7 += 16;
                    ptr8 += 16;
                    ptr9 += 16;
                    ptra += 16;
                    ptrb += 16;
                    ptrc += 16;
                    ptrd += 16;
                    ptre += 16;
                    ptrf += 16;
                    outptr += 256;
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
                    outptr[8] = *ptr8++;
                    outptr[9] = *ptr9++;
                    outptr[10] = *ptra++;
                    outptr[11] = *ptrb++;
                    outptr[12] = *ptrc++;
                    outptr[13] = *ptrd++;
                    outptr[14] = *ptre++;
                    outptr[15] = *ptrf++;

                    outptr += 16;
                }
            }
        }
#endif // __AVX512F__

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
                    __m256 _row0 = _mm256_loadu_ps(ptr0);
                    __m256 _row1 = _mm256_loadu_ps(ptr1);
                    __m256 _row2 = _mm256_loadu_ps(ptr2);
                    __m256 _row3 = _mm256_loadu_ps(ptr3);
                    __m256 _row4 = _mm256_loadu_ps(ptr4);
                    __m256 _row5 = _mm256_loadu_ps(ptr5);
                    __m256 _row6 = _mm256_loadu_ps(ptr6);
                    __m256 _row7 = _mm256_loadu_ps(ptr7);

                    transpose8x8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);

                    _mm256_storeu_ps(outptr, _row0);
                    _mm256_storeu_ps(outptr + 8, _row1);
                    _mm256_storeu_ps(outptr + 16, _row2);
                    _mm256_storeu_ps(outptr + 24, _row3);
                    _mm256_storeu_ps(outptr + 32, _row4);
                    _mm256_storeu_ps(outptr + 40, _row5);
                    _mm256_storeu_ps(outptr + 48, _row6);
                    _mm256_storeu_ps(outptr + 56, _row7);

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
        }
#endif // __AVX__

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
                    __m128 _row0 = _mm_loadu_ps(ptr0);
                    __m128 _row1 = _mm_loadu_ps(ptr1);
                    __m128 _row2 = _mm_loadu_ps(ptr2);
                    __m128 _row3 = _mm_loadu_ps(ptr3);

                    _MM_TRANSPOSE4_PS(_row0, _row1, _row2, _row3);

                    _mm_storeu_ps(outptr, _row0);
                    _mm_storeu_ps(outptr + 4, _row1);
                    _mm_storeu_ps(outptr + 8, _row2);
                    _mm_storeu_ps(outptr + 12, _row3);

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
        }
#endif // __SSE2__

        if (out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob_flattened + size * q;
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __SSE2__
#if __AVX__
                for (; i + 7 < size; i += 8)
                {
                    __m256 _v = _mm256_loadu_ps(ptr);
                    _mm256_storeu_ps(outptr, _v);
                    ptr += 8;
                    outptr += 8;
                }
#endif
                for (; i + 3 < size; i += 4)
                {
                    __m128 _v = _mm_loadu_ps(ptr);
                    _mm_storeu_ps(outptr, _v);
                    ptr += 4;
                    outptr += 4;
                }
#endif // __SSE2__
                for (; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
            }
        }
    }

    return 0;
}

int Reshape_x86::forward_bf16s_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_BATCH
    if (batch_mode != 0)
        return forward_batch(bottom_blobs, top_blobs, opt);
#endif

    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    // resolve out shape
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
        // flatten
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
#if __SSE2__
        if (opt.use_packing_layout)
        {
#if __AVX512F__
            out_elempack = outh % 16 == 0 ? 16 : outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#elif __AVX__
            out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
            out_elempack = outh % 4 == 0 ? 4 : 1;
#endif
        }
#endif // __SSE2__
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 2 && bottom_blob.h * elempack == outh && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        if (out_elempack == 1)
        {
            // flatten
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

        // flatten
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

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (out_elempack == 16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const unsigned short* ptr0 = (const unsigned short*)bottom_blob_flattened + outw * i * 16;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob_flattened + outw * (i * 16 + 1);
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob_flattened + outw * (i * 16 + 2);
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob_flattened + outw * (i * 16 + 3);
                const unsigned short* ptr4 = (const unsigned short*)bottom_blob_flattened + outw * (i * 16 + 4);
                const unsigned short* ptr5 = (const unsigned short*)bottom_blob_flattened + outw * (i * 16 + 5);
                const unsigned short* ptr6 = (const unsigned short*)bottom_blob_flattened + outw * (i * 16 + 6);
                const unsigned short* ptr7 = (const unsigned short*)bottom_blob_flattened + outw * (i * 16 + 7);
                const unsigned short* ptr8 = (const unsigned short*)bottom_blob_flattened + outw * (i * 16 + 8);
                const unsigned short* ptr9 = (const unsigned short*)bottom_blob_flattened + outw * (i * 16 + 9);
                const unsigned short* ptra = (const unsigned short*)bottom_blob_flattened + outw * (i * 16 + 10);
                const unsigned short* ptrb = (const unsigned short*)bottom_blob_flattened + outw * (i * 16 + 11);
                const unsigned short* ptrc = (const unsigned short*)bottom_blob_flattened + outw * (i * 16 + 12);
                const unsigned short* ptrd = (const unsigned short*)bottom_blob_flattened + outw * (i * 16 + 13);
                const unsigned short* ptre = (const unsigned short*)bottom_blob_flattened + outw * (i * 16 + 14);
                const unsigned short* ptrf = (const unsigned short*)bottom_blob_flattened + outw * (i * 16 + 15);
                unsigned short* outptr = top_blob.row<unsigned short>(i);

                for (int j = 0; j < outw; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;
                    outptr[8] = *ptr8++;
                    outptr[9] = *ptr9++;
                    outptr[10] = *ptra++;
                    outptr[11] = *ptrb++;
                    outptr[12] = *ptrc++;
                    outptr[13] = *ptrd++;
                    outptr[14] = *ptre++;
                    outptr[15] = *ptrf++;

                    outptr += 16;
                }
            }
        }
#endif // __AVX512F__

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

                for (int j = 0; j < outw; j++)
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
        }
#endif // __AVX__

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
        }
#endif // __SSE2__
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
#if __SSE2__
        if (opt.use_packing_layout)
        {
#if __AVX512F__
            out_elempack = outc % 16 == 0 ? 16 : outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#elif __AVX__
            out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#else
            out_elempack = outc % 4 == 0 ? 4 : 1;
#endif
        }
#endif // __SSE2__
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if ((dims == 3 || dims == 4) && bottom_blob.c * elempack == outc && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            top_blob.dims = ndim;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.d = outd;
            return 0;
        }

        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        if (ndim == 3)
        {
            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        }
        else // if (ndim == 4)
        {
            top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        }
        if (top_blob.empty())
            return -100;

        int size = top_blob.w * top_blob.h * top_blob.d;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (out_elempack == 16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr0 = (const unsigned short*)bottom_blob_flattened + size * q * 16;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob_flattened + size * (q * 16 + 1);
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob_flattened + size * (q * 16 + 2);
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob_flattened + size * (q * 16 + 3);
                const unsigned short* ptr4 = (const unsigned short*)bottom_blob_flattened + size * (q * 16 + 4);
                const unsigned short* ptr5 = (const unsigned short*)bottom_blob_flattened + size * (q * 16 + 5);
                const unsigned short* ptr6 = (const unsigned short*)bottom_blob_flattened + size * (q * 16 + 6);
                const unsigned short* ptr7 = (const unsigned short*)bottom_blob_flattened + size * (q * 16 + 7);
                const unsigned short* ptr8 = (const unsigned short*)bottom_blob_flattened + size * (q * 16 + 8);
                const unsigned short* ptr9 = (const unsigned short*)bottom_blob_flattened + size * (q * 16 + 9);
                const unsigned short* ptra = (const unsigned short*)bottom_blob_flattened + size * (q * 16 + 10);
                const unsigned short* ptrb = (const unsigned short*)bottom_blob_flattened + size * (q * 16 + 11);
                const unsigned short* ptrc = (const unsigned short*)bottom_blob_flattened + size * (q * 16 + 12);
                const unsigned short* ptrd = (const unsigned short*)bottom_blob_flattened + size * (q * 16 + 13);
                const unsigned short* ptre = (const unsigned short*)bottom_blob_flattened + size * (q * 16 + 14);
                const unsigned short* ptrf = (const unsigned short*)bottom_blob_flattened + size * (q * 16 + 15);
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;
                    outptr[8] = *ptr8++;
                    outptr[9] = *ptr9++;
                    outptr[10] = *ptra++;
                    outptr[11] = *ptrb++;
                    outptr[12] = *ptrc++;
                    outptr[13] = *ptrd++;
                    outptr[14] = *ptre++;
                    outptr[15] = *ptrf++;

                    outptr += 16;
                }
            }
        }
#endif // __AVX512F__

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

                for (int i = 0; i < size; i++)
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
        }
#endif // __AVX__

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
        }
#endif // __SSE2__

        if (out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob_flattened + size * q;
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __SSE2__
#if __AVX__
                for (; i + 15 < size; i += 16)
                {
                    __m256i _v = _mm256_loadu_si256((const __m256i*)ptr);
                    _mm256_storeu_si256((__m256i*)outptr, _v);
                    ptr += 16;
                    outptr += 16;
                }
#endif
                for (; i + 7 < size; i += 8)
                {
                    __m128i _v = _mm_loadu_si128((const __m128i*)ptr);
                    _mm_storeu_si128((__m128i*)outptr, _v);
                    ptr += 8;
                    outptr += 8;
                }
#endif // __SSE2__
                for (; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn

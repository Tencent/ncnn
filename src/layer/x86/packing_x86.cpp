// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "packing_x86.h"

#include "x86_usability.h"

namespace ncnn {

Packing_x86::Packing_x86()
{
    support_packing = true;
}

int Packing_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

    if (elembits == 8)
        return forward_int8(bottom_blob, top_blob, opt);

    if (use_padding)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    if (elembits != 32)
    {
        // non-fp32 type
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
    bool pack1to8 = elempack == 1 && out_elempack == 8;
    bool pack8to1 = elempack == 8 && out_elempack == 1;
    bool pack4to8 = elempack == 4 && out_elempack == 8;
    bool pack8to4 = elempack == 8 && out_elempack == 4;
    bool pack1to16 = elempack == 1 && out_elempack == 16;
    bool pack16to1 = elempack == 16 && out_elempack == 1;
    bool pack4to16 = elempack == 4 && out_elempack == 16;
    bool pack16to4 = elempack == 16 && out_elempack == 4;
    bool pack8to16 = elempack == 8 && out_elempack == 16;
    bool pack16to8 = elempack == 16 && out_elempack == 8;

    if (!pack1to4 && !pack4to1 && !pack1to8 && !pack8to1 && !pack4to8 && !pack8to4 && !pack1to16 && !pack16to1 && !pack4to16 && !pack16to4 && !pack8to16 && !pack16to8)
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
        top_blob.cstep = w * elempack / out_elempack;
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
#if __SSE2__
                for (; j + 3 < w; j += 4)
                {
                    // transpose 4x4
                    __m128 _r0 = _mm_loadu_ps(r0);
                    __m128 _r1 = _mm_loadu_ps(r1);
                    __m128 _r2 = _mm_loadu_ps(r2);
                    __m128 _r3 = _mm_loadu_ps(r3);

                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

                    _mm_store_ps(outptr, _r0);
                    _mm_store_ps(outptr + 4, _r1);
                    _mm_store_ps(outptr + 4 * 2, _r2);
                    _mm_store_ps(outptr + 4 * 3, _r3);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
#endif // __SSE2__
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
#if __SSE2__
                for (; j + 3 < w; j += 4)
                {
                    // transpose 4x4
                    __m128 _r0 = _mm_load_ps(r0);
                    __m128 _r1 = _mm_load_ps(r0 + 4);
                    __m128 _r2 = _mm_load_ps(r0 + 4 * 2);
                    __m128 _r3 = _mm_load_ps(r0 + 4 * 3);

                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

                    _mm_storeu_ps(outptr0, _r0);
                    _mm_storeu_ps(outptr1, _r1);
                    _mm_storeu_ps(outptr2, _r2);
                    _mm_storeu_ps(outptr3, _r3);

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
#endif // __SSE2__
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
#if __AVX__
                for (; j + 7 < w; j += 8)
                {
                    __m256 _row0 = _mm256_loadu_ps(r0);
                    __m256 _row1 = _mm256_loadu_ps(r1);
                    __m256 _row2 = _mm256_loadu_ps(r2);
                    __m256 _row3 = _mm256_loadu_ps(r3);
                    __m256 _row4 = _mm256_loadu_ps(r4);
                    __m256 _row5 = _mm256_loadu_ps(r5);
                    __m256 _row6 = _mm256_loadu_ps(r6);
                    __m256 _row7 = _mm256_loadu_ps(r7);
                    transpose8x8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
                    _mm256_storeu_ps(outptr, _row0);
                    _mm256_storeu_ps(outptr + 8, _row1);
                    _mm256_storeu_ps(outptr + 16, _row2);
                    _mm256_storeu_ps(outptr + 24, _row3);
                    _mm256_storeu_ps(outptr + 32, _row4);
                    _mm256_storeu_ps(outptr + 40, _row5);
                    _mm256_storeu_ps(outptr + 48, _row6);
                    _mm256_storeu_ps(outptr + 56, _row7);
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
#endif // __AVX__
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
#if __AVX__
                for (; j + 7 < w; j += 8)
                {
                    __m256 _row0 = _mm256_loadu_ps(r0);
                    __m256 _row1 = _mm256_loadu_ps(r0 + 8);
                    __m256 _row2 = _mm256_loadu_ps(r0 + 16);
                    __m256 _row3 = _mm256_loadu_ps(r0 + 24);
                    __m256 _row4 = _mm256_loadu_ps(r0 + 32);
                    __m256 _row5 = _mm256_loadu_ps(r0 + 40);
                    __m256 _row6 = _mm256_loadu_ps(r0 + 48);
                    __m256 _row7 = _mm256_loadu_ps(r0 + 56);
                    transpose8x8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
                    _mm256_storeu_ps(outptr0, _row0);
                    _mm256_storeu_ps(outptr1, _row1);
                    _mm256_storeu_ps(outptr2, _row2);
                    _mm256_storeu_ps(outptr3, _row3);
                    _mm256_storeu_ps(outptr4, _row4);
                    _mm256_storeu_ps(outptr5, _row5);
                    _mm256_storeu_ps(outptr6, _row6);
                    _mm256_storeu_ps(outptr7, _row7);

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
#endif // __AVX__
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
                const float* r0 = bottom_blob.row(i);

                float* outptr0 = top_blob.row(i * 2);
                float* outptr1 = top_blob.row(i * 2 + 1);

                for (int j = 0; j < w; j++)
                {
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr1[0] = r0[4];
                    outptr1[1] = r0[5];
                    outptr1[2] = r0[6];
                    outptr1[3] = r0[7];

                    r0 += 8;
                    outptr0 += 4;
                    outptr1 += 4;
                }
            }
        }
        if (pack1to16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* r0 = bottom_blob.row(i * 16);
                const float* r1 = bottom_blob.row(i * 16 + 1);
                const float* r2 = bottom_blob.row(i * 16 + 2);
                const float* r3 = bottom_blob.row(i * 16 + 3);
                const float* r4 = bottom_blob.row(i * 16 + 4);
                const float* r5 = bottom_blob.row(i * 16 + 5);
                const float* r6 = bottom_blob.row(i * 16 + 6);
                const float* r7 = bottom_blob.row(i * 16 + 7);
                const float* r8 = bottom_blob.row(i * 16 + 8);
                const float* r9 = bottom_blob.row(i * 16 + 9);
                const float* ra = bottom_blob.row(i * 16 + 10);
                const float* rb = bottom_blob.row(i * 16 + 11);
                const float* rc = bottom_blob.row(i * 16 + 12);
                const float* rd = bottom_blob.row(i * 16 + 13);
                const float* re = bottom_blob.row(i * 16 + 14);
                const float* rf = bottom_blob.row(i * 16 + 15);

                float* outptr = top_blob.row(i);

                int j = 0;
#if __AVX512F__
                for (; j + 15 < w; j += 16)
                {
                    __m512 _r0 = _mm512_loadu_ps(r0);
                    __m512 _r1 = _mm512_loadu_ps(r1);
                    __m512 _r2 = _mm512_loadu_ps(r2);
                    __m512 _r3 = _mm512_loadu_ps(r3);
                    __m512 _r4 = _mm512_loadu_ps(r4);
                    __m512 _r5 = _mm512_loadu_ps(r5);
                    __m512 _r6 = _mm512_loadu_ps(r6);
                    __m512 _r7 = _mm512_loadu_ps(r7);
                    __m512 _r8 = _mm512_loadu_ps(r8);
                    __m512 _r9 = _mm512_loadu_ps(r9);
                    __m512 _ra = _mm512_loadu_ps(ra);
                    __m512 _rb = _mm512_loadu_ps(rb);
                    __m512 _rc = _mm512_loadu_ps(rc);
                    __m512 _rd = _mm512_loadu_ps(rd);
                    __m512 _re = _mm512_loadu_ps(re);
                    __m512 _rf = _mm512_loadu_ps(rf);
                    transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                    _mm512_storeu_ps(outptr, _r0);
                    _mm512_storeu_ps(outptr + 16, _r1);
                    _mm512_storeu_ps(outptr + 16 * 2, _r2);
                    _mm512_storeu_ps(outptr + 16 * 3, _r3);
                    _mm512_storeu_ps(outptr + 16 * 4, _r4);
                    _mm512_storeu_ps(outptr + 16 * 5, _r5);
                    _mm512_storeu_ps(outptr + 16 * 6, _r6);
                    _mm512_storeu_ps(outptr + 16 * 7, _r7);
                    _mm512_storeu_ps(outptr + 16 * 8, _r8);
                    _mm512_storeu_ps(outptr + 16 * 9, _r9);
                    _mm512_storeu_ps(outptr + 16 * 10, _ra);
                    _mm512_storeu_ps(outptr + 16 * 11, _rb);
                    _mm512_storeu_ps(outptr + 16 * 12, _rc);
                    _mm512_storeu_ps(outptr + 16 * 13, _rd);
                    _mm512_storeu_ps(outptr + 16 * 14, _re);
                    _mm512_storeu_ps(outptr + 16 * 15, _rf);
                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                    r3 += 16;
                    r4 += 16;
                    r5 += 16;
                    r6 += 16;
                    r7 += 16;
                    r8 += 16;
                    r9 += 16;
                    ra += 16;
                    rb += 16;
                    rc += 16;
                    rd += 16;
                    re += 16;
                    rf += 16;
                    outptr += 256;
                }
#endif // __AVX512F__
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
                    outptr[8] = *r8++;
                    outptr[9] = *r9++;
                    outptr[10] = *ra++;
                    outptr[11] = *rb++;
                    outptr[12] = *rc++;
                    outptr[13] = *rd++;
                    outptr[14] = *re++;
                    outptr[15] = *rf++;

                    outptr += 16;
                }
            }
        }
        if (pack16to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* r0 = bottom_blob.row(i);

                float* outptr0 = top_blob.row(i * 16);
                float* outptr1 = top_blob.row(i * 16 + 1);
                float* outptr2 = top_blob.row(i * 16 + 2);
                float* outptr3 = top_blob.row(i * 16 + 3);
                float* outptr4 = top_blob.row(i * 16 + 4);
                float* outptr5 = top_blob.row(i * 16 + 5);
                float* outptr6 = top_blob.row(i * 16 + 6);
                float* outptr7 = top_blob.row(i * 16 + 7);
                float* outptr8 = top_blob.row(i * 16 + 8);
                float* outptr9 = top_blob.row(i * 16 + 9);
                float* outptra = top_blob.row(i * 16 + 10);
                float* outptrb = top_blob.row(i * 16 + 11);
                float* outptrc = top_blob.row(i * 16 + 12);
                float* outptrd = top_blob.row(i * 16 + 13);
                float* outptre = top_blob.row(i * 16 + 14);
                float* outptrf = top_blob.row(i * 16 + 15);

                int j = 0;
#if __AVX512F__
                for (; j + 15 < w; j += 16)
                {
                    __m512 _r0 = _mm512_loadu_ps(r0);
                    __m512 _r1 = _mm512_loadu_ps(r0 + 16);
                    __m512 _r2 = _mm512_loadu_ps(r0 + 16 * 2);
                    __m512 _r3 = _mm512_loadu_ps(r0 + 16 * 3);
                    __m512 _r4 = _mm512_loadu_ps(r0 + 16 * 4);
                    __m512 _r5 = _mm512_loadu_ps(r0 + 16 * 5);
                    __m512 _r6 = _mm512_loadu_ps(r0 + 16 * 6);
                    __m512 _r7 = _mm512_loadu_ps(r0 + 16 * 7);
                    __m512 _r8 = _mm512_loadu_ps(r0 + 16 * 8);
                    __m512 _r9 = _mm512_loadu_ps(r0 + 16 * 9);
                    __m512 _ra = _mm512_loadu_ps(r0 + 16 * 10);
                    __m512 _rb = _mm512_loadu_ps(r0 + 16 * 11);
                    __m512 _rc = _mm512_loadu_ps(r0 + 16 * 12);
                    __m512 _rd = _mm512_loadu_ps(r0 + 16 * 13);
                    __m512 _re = _mm512_loadu_ps(r0 + 16 * 14);
                    __m512 _rf = _mm512_loadu_ps(r0 + 16 * 15);
                    transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                    _mm512_storeu_ps(outptr0, _r0);
                    _mm512_storeu_ps(outptr1, _r1);
                    _mm512_storeu_ps(outptr2, _r2);
                    _mm512_storeu_ps(outptr3, _r3);
                    _mm512_storeu_ps(outptr4, _r4);
                    _mm512_storeu_ps(outptr5, _r5);
                    _mm512_storeu_ps(outptr6, _r6);
                    _mm512_storeu_ps(outptr7, _r7);
                    _mm512_storeu_ps(outptr8, _r8);
                    _mm512_storeu_ps(outptr9, _r9);
                    _mm512_storeu_ps(outptra, _ra);
                    _mm512_storeu_ps(outptrb, _rb);
                    _mm512_storeu_ps(outptrc, _rc);
                    _mm512_storeu_ps(outptrd, _rd);
                    _mm512_storeu_ps(outptre, _re);
                    _mm512_storeu_ps(outptrf, _rf);

                    r0 += 256;
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
#endif // __AVX512F__
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
                    *outptr8++ = r0[8];
                    *outptr9++ = r0[9];
                    *outptra++ = r0[10];
                    *outptrb++ = r0[11];
                    *outptrc++ = r0[12];
                    *outptrd++ = r0[13];
                    *outptre++ = r0[14];
                    *outptrf++ = r0[15];

                    r0 += 16;
                }
            }
        }
        if (pack4to16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* r0 = bottom_blob.row(i * 4);
                const float* r1 = bottom_blob.row(i * 4 + 1);
                const float* r2 = bottom_blob.row(i * 4 + 2);
                const float* r3 = bottom_blob.row(i * 4 + 3);

                float* outptr = top_blob.row(i);

                for (int j = 0; j < w; j++)
                {
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r1[0];
                    outptr[5] = r1[1];
                    outptr[6] = r1[2];
                    outptr[7] = r1[3];
                    outptr[8] = r2[0];
                    outptr[9] = r2[1];
                    outptr[10] = r2[2];
                    outptr[11] = r2[3];
                    outptr[12] = r3[0];
                    outptr[13] = r3[1];
                    outptr[14] = r3[2];
                    outptr[15] = r3[3];

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
            }
        }
        if (pack16to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* r0 = bottom_blob.row(i);

                float* outptr0 = top_blob.row(i * 4);
                float* outptr1 = top_blob.row(i * 4 + 1);
                float* outptr2 = top_blob.row(i * 4 + 2);
                float* outptr3 = top_blob.row(i * 4 + 3);

                for (int j = 0; j < w; j++)
                {
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr1[0] = r0[4];
                    outptr1[1] = r0[5];
                    outptr1[2] = r0[6];
                    outptr1[3] = r0[7];
                    outptr2[0] = r0[8];
                    outptr2[1] = r0[9];
                    outptr2[2] = r0[10];
                    outptr2[3] = r0[11];
                    outptr3[0] = r0[12];
                    outptr3[1] = r0[13];
                    outptr3[2] = r0[14];
                    outptr3[3] = r0[15];

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
            }
        }
        if (pack8to16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* r0 = bottom_blob.row(i * 2);
                const float* r1 = bottom_blob.row(i * 2 + 1);

                float* outptr = top_blob.row(i);

                for (int j = 0; j < w; j++)
                {
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r0[4];
                    outptr[5] = r0[5];
                    outptr[6] = r0[6];
                    outptr[7] = r0[7];
                    outptr[8] = r1[0];
                    outptr[9] = r1[1];
                    outptr[10] = r1[2];
                    outptr[11] = r1[3];
                    outptr[12] = r1[4];
                    outptr[13] = r1[5];
                    outptr[14] = r1[6];
                    outptr[15] = r1[7];

                    r0 += 8;
                    r1 += 8;
                    outptr += 16;
                }
            }
        }
        if (pack16to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* r0 = bottom_blob.row(i);

                float* outptr0 = top_blob.row(i * 2);
                float* outptr1 = top_blob.row(i * 2 + 1);

                for (int j = 0; j < w; j++)
                {
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr0[4] = r0[4];
                    outptr0[5] = r0[5];
                    outptr0[6] = r0[6];
                    outptr0[7] = r0[7];
                    outptr1[0] = r0[8];
                    outptr1[1] = r0[9];
                    outptr1[2] = r0[10];
                    outptr1[3] = r0[11];
                    outptr1[4] = r0[12];
                    outptr1[5] = r0[13];
                    outptr1[6] = r0[14];
                    outptr1[7] = r0[15];

                    r0 += 16;
                    outptr0 += 8;
                    outptr1 += 8;
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
#if __SSE2__
                for (; i + 3 < size; i += 4)
                {
                    // transpose 4x4
                    __m128 _r0 = _mm_loadu_ps(r0);
                    __m128 _r1 = _mm_loadu_ps(r1);
                    __m128 _r2 = _mm_loadu_ps(r2);
                    __m128 _r3 = _mm_loadu_ps(r3);

                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

                    _mm_store_ps(outptr, _r0);
                    _mm_store_ps(outptr + 4, _r1);
                    _mm_store_ps(outptr + 4 * 2, _r2);
                    _mm_store_ps(outptr + 4 * 3, _r3);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
#endif // __SSE2__
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
#if __SSE2__
                for (; i + 3 < size; i += 4)
                {
                    // transpose 4x4
                    __m128 _r0 = _mm_load_ps(r0);
                    __m128 _r1 = _mm_load_ps(r0 + 4);
                    __m128 _r2 = _mm_load_ps(r0 + 4 * 2);
                    __m128 _r3 = _mm_load_ps(r0 + 4 * 3);

                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

                    _mm_storeu_ps(outptr0, _r0);
                    _mm_storeu_ps(outptr1, _r1);
                    _mm_storeu_ps(outptr2, _r2);
                    _mm_storeu_ps(outptr3, _r3);

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
#endif // __SSE2__
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
#if __AVX__
                for (; i + 7 < size; i += 8)
                {
                    __m256 _row0 = _mm256_loadu_ps(r0);
                    __m256 _row1 = _mm256_loadu_ps(r1);
                    __m256 _row2 = _mm256_loadu_ps(r2);
                    __m256 _row3 = _mm256_loadu_ps(r3);
                    __m256 _row4 = _mm256_loadu_ps(r4);
                    __m256 _row5 = _mm256_loadu_ps(r5);
                    __m256 _row6 = _mm256_loadu_ps(r6);
                    __m256 _row7 = _mm256_loadu_ps(r7);
                    transpose8x8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
                    _mm256_storeu_ps(outptr, _row0);
                    _mm256_storeu_ps(outptr + 8, _row1);
                    _mm256_storeu_ps(outptr + 16, _row2);
                    _mm256_storeu_ps(outptr + 24, _row3);
                    _mm256_storeu_ps(outptr + 32, _row4);
                    _mm256_storeu_ps(outptr + 40, _row5);
                    _mm256_storeu_ps(outptr + 48, _row6);
                    _mm256_storeu_ps(outptr + 56, _row7);
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
#endif // __AVX__
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
#if __AVX__
                for (; i + 7 < size; i += 8)
                {
                    __m256 _row0 = _mm256_loadu_ps(r0);
                    __m256 _row1 = _mm256_loadu_ps(r0 + 8);
                    __m256 _row2 = _mm256_loadu_ps(r0 + 16);
                    __m256 _row3 = _mm256_loadu_ps(r0 + 24);
                    __m256 _row4 = _mm256_loadu_ps(r0 + 32);
                    __m256 _row5 = _mm256_loadu_ps(r0 + 40);
                    __m256 _row6 = _mm256_loadu_ps(r0 + 48);
                    __m256 _row7 = _mm256_loadu_ps(r0 + 56);
                    transpose8x8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
                    _mm256_storeu_ps(outptr0, _row0);
                    _mm256_storeu_ps(outptr1, _row1);
                    _mm256_storeu_ps(outptr2, _row2);
                    _mm256_storeu_ps(outptr3, _row3);
                    _mm256_storeu_ps(outptr4, _row4);
                    _mm256_storeu_ps(outptr5, _row5);
                    _mm256_storeu_ps(outptr6, _row6);
                    _mm256_storeu_ps(outptr7, _row7);

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
#endif // __AVX__
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
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q * 2);
                float* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr1[0] = r0[4];
                    outptr1[1] = r0[5];
                    outptr1[2] = r0[6];
                    outptr1[3] = r0[7];

                    r0 += 8;
                    outptr0 += 4;
                    outptr1 += 4;
                }
            }
        }
        if (pack1to16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* r0 = bottom_blob.channel(q * 16);
                const float* r1 = bottom_blob.channel(q * 16 + 1);
                const float* r2 = bottom_blob.channel(q * 16 + 2);
                const float* r3 = bottom_blob.channel(q * 16 + 3);
                const float* r4 = bottom_blob.channel(q * 16 + 4);
                const float* r5 = bottom_blob.channel(q * 16 + 5);
                const float* r6 = bottom_blob.channel(q * 16 + 6);
                const float* r7 = bottom_blob.channel(q * 16 + 7);
                const float* r8 = bottom_blob.channel(q * 16 + 8);
                const float* r9 = bottom_blob.channel(q * 16 + 9);
                const float* ra = bottom_blob.channel(q * 16 + 10);
                const float* rb = bottom_blob.channel(q * 16 + 11);
                const float* rc = bottom_blob.channel(q * 16 + 12);
                const float* rd = bottom_blob.channel(q * 16 + 13);
                const float* re = bottom_blob.channel(q * 16 + 14);
                const float* rf = bottom_blob.channel(q * 16 + 15);

                float* outptr = top_blob.channel(q);

                int i = 0;
#if __AVX512F__
                for (; i + 15 < size; i += 16)
                {
                    __m512 _r0 = _mm512_loadu_ps(r0);
                    __m512 _r1 = _mm512_loadu_ps(r1);
                    __m512 _r2 = _mm512_loadu_ps(r2);
                    __m512 _r3 = _mm512_loadu_ps(r3);
                    __m512 _r4 = _mm512_loadu_ps(r4);
                    __m512 _r5 = _mm512_loadu_ps(r5);
                    __m512 _r6 = _mm512_loadu_ps(r6);
                    __m512 _r7 = _mm512_loadu_ps(r7);
                    __m512 _r8 = _mm512_loadu_ps(r8);
                    __m512 _r9 = _mm512_loadu_ps(r9);
                    __m512 _ra = _mm512_loadu_ps(ra);
                    __m512 _rb = _mm512_loadu_ps(rb);
                    __m512 _rc = _mm512_loadu_ps(rc);
                    __m512 _rd = _mm512_loadu_ps(rd);
                    __m512 _re = _mm512_loadu_ps(re);
                    __m512 _rf = _mm512_loadu_ps(rf);
                    transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                    _mm512_storeu_ps(outptr, _r0);
                    _mm512_storeu_ps(outptr + 16, _r1);
                    _mm512_storeu_ps(outptr + 16 * 2, _r2);
                    _mm512_storeu_ps(outptr + 16 * 3, _r3);
                    _mm512_storeu_ps(outptr + 16 * 4, _r4);
                    _mm512_storeu_ps(outptr + 16 * 5, _r5);
                    _mm512_storeu_ps(outptr + 16 * 6, _r6);
                    _mm512_storeu_ps(outptr + 16 * 7, _r7);
                    _mm512_storeu_ps(outptr + 16 * 8, _r8);
                    _mm512_storeu_ps(outptr + 16 * 9, _r9);
                    _mm512_storeu_ps(outptr + 16 * 10, _ra);
                    _mm512_storeu_ps(outptr + 16 * 11, _rb);
                    _mm512_storeu_ps(outptr + 16 * 12, _rc);
                    _mm512_storeu_ps(outptr + 16 * 13, _rd);
                    _mm512_storeu_ps(outptr + 16 * 14, _re);
                    _mm512_storeu_ps(outptr + 16 * 15, _rf);
                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                    r3 += 16;
                    r4 += 16;
                    r5 += 16;
                    r6 += 16;
                    r7 += 16;
                    r8 += 16;
                    r9 += 16;
                    ra += 16;
                    rb += 16;
                    rc += 16;
                    rd += 16;
                    re += 16;
                    rf += 16;
                    outptr += 256;
                }
#endif // __AVX512F__
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
                    outptr[8] = *r8++;
                    outptr[9] = *r9++;
                    outptr[10] = *ra++;
                    outptr[11] = *rb++;
                    outptr[12] = *rc++;
                    outptr[13] = *rd++;
                    outptr[14] = *re++;
                    outptr[15] = *rf++;

                    outptr += 16;
                }
            }
        }
        if (pack16to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q * 16);
                float* outptr1 = top_blob.channel(q * 16 + 1);
                float* outptr2 = top_blob.channel(q * 16 + 2);
                float* outptr3 = top_blob.channel(q * 16 + 3);
                float* outptr4 = top_blob.channel(q * 16 + 4);
                float* outptr5 = top_blob.channel(q * 16 + 5);
                float* outptr6 = top_blob.channel(q * 16 + 6);
                float* outptr7 = top_blob.channel(q * 16 + 7);
                float* outptr8 = top_blob.channel(q * 16 + 8);
                float* outptr9 = top_blob.channel(q * 16 + 9);
                float* outptra = top_blob.channel(q * 16 + 10);
                float* outptrb = top_blob.channel(q * 16 + 11);
                float* outptrc = top_blob.channel(q * 16 + 12);
                float* outptrd = top_blob.channel(q * 16 + 13);
                float* outptre = top_blob.channel(q * 16 + 14);
                float* outptrf = top_blob.channel(q * 16 + 15);

                int i = 0;
#if __AVX512F__
                for (; i + 15 < size; i += 16)
                {
                    __m512 _r0 = _mm512_loadu_ps(r0);
                    __m512 _r1 = _mm512_loadu_ps(r0 + 16);
                    __m512 _r2 = _mm512_loadu_ps(r0 + 16 * 2);
                    __m512 _r3 = _mm512_loadu_ps(r0 + 16 * 3);
                    __m512 _r4 = _mm512_loadu_ps(r0 + 16 * 4);
                    __m512 _r5 = _mm512_loadu_ps(r0 + 16 * 5);
                    __m512 _r6 = _mm512_loadu_ps(r0 + 16 * 6);
                    __m512 _r7 = _mm512_loadu_ps(r0 + 16 * 7);
                    __m512 _r8 = _mm512_loadu_ps(r0 + 16 * 8);
                    __m512 _r9 = _mm512_loadu_ps(r0 + 16 * 9);
                    __m512 _ra = _mm512_loadu_ps(r0 + 16 * 10);
                    __m512 _rb = _mm512_loadu_ps(r0 + 16 * 11);
                    __m512 _rc = _mm512_loadu_ps(r0 + 16 * 12);
                    __m512 _rd = _mm512_loadu_ps(r0 + 16 * 13);
                    __m512 _re = _mm512_loadu_ps(r0 + 16 * 14);
                    __m512 _rf = _mm512_loadu_ps(r0 + 16 * 15);
                    transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                    _mm512_storeu_ps(outptr0, _r0);
                    _mm512_storeu_ps(outptr1, _r1);
                    _mm512_storeu_ps(outptr2, _r2);
                    _mm512_storeu_ps(outptr3, _r3);
                    _mm512_storeu_ps(outptr4, _r4);
                    _mm512_storeu_ps(outptr5, _r5);
                    _mm512_storeu_ps(outptr6, _r6);
                    _mm512_storeu_ps(outptr7, _r7);
                    _mm512_storeu_ps(outptr8, _r8);
                    _mm512_storeu_ps(outptr9, _r9);
                    _mm512_storeu_ps(outptra, _ra);
                    _mm512_storeu_ps(outptrb, _rb);
                    _mm512_storeu_ps(outptrc, _rc);
                    _mm512_storeu_ps(outptrd, _rd);
                    _mm512_storeu_ps(outptre, _re);
                    _mm512_storeu_ps(outptrf, _rf);

                    r0 += 256;
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
#endif // __AVX512F__
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
                    *outptr8++ = r0[8];
                    *outptr9++ = r0[9];
                    *outptra++ = r0[10];
                    *outptrb++ = r0[11];
                    *outptrc++ = r0[12];
                    *outptrd++ = r0[13];
                    *outptre++ = r0[14];
                    *outptrf++ = r0[15];

                    r0 += 16;
                }
            }
        }
        if (pack4to16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* r0 = bottom_blob.channel(q * 4);
                const float* r1 = bottom_blob.channel(q * 4 + 1);
                const float* r2 = bottom_blob.channel(q * 4 + 2);
                const float* r3 = bottom_blob.channel(q * 4 + 3);

                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r1[0];
                    outptr[5] = r1[1];
                    outptr[6] = r1[2];
                    outptr[7] = r1[3];
                    outptr[8] = r2[0];
                    outptr[9] = r2[1];
                    outptr[10] = r2[2];
                    outptr[11] = r2[3];
                    outptr[12] = r3[0];
                    outptr[13] = r3[1];
                    outptr[14] = r3[2];
                    outptr[15] = r3[3];

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
            }
        }
        if (pack16to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q * 4);
                float* outptr1 = top_blob.channel(q * 4 + 1);
                float* outptr2 = top_blob.channel(q * 4 + 2);
                float* outptr3 = top_blob.channel(q * 4 + 3);

                for (int i = 0; i < size; i++)
                {
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr1[0] = r0[4];
                    outptr1[1] = r0[5];
                    outptr1[2] = r0[6];
                    outptr1[3] = r0[7];
                    outptr2[0] = r0[8];
                    outptr2[1] = r0[9];
                    outptr2[2] = r0[10];
                    outptr2[3] = r0[11];
                    outptr3[0] = r0[12];
                    outptr3[1] = r0[13];
                    outptr3[2] = r0[14];
                    outptr3[3] = r0[15];

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
            }
        }
        if (pack8to16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* r0 = bottom_blob.channel(q * 2);
                const float* r1 = bottom_blob.channel(q * 2 + 1);

                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r0[4];
                    outptr[5] = r0[5];
                    outptr[6] = r0[6];
                    outptr[7] = r0[7];
                    outptr[8] = r1[0];
                    outptr[9] = r1[1];
                    outptr[10] = r1[2];
                    outptr[11] = r1[3];
                    outptr[12] = r1[4];
                    outptr[13] = r1[5];
                    outptr[14] = r1[6];
                    outptr[15] = r1[7];

                    r0 += 8;
                    r1 += 8;
                    outptr += 16;
                }
            }
        }
        if (pack16to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q * 2);
                float* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr0[4] = r0[4];
                    outptr0[5] = r0[5];
                    outptr0[6] = r0[6];
                    outptr0[7] = r0[7];
                    outptr1[0] = r0[8];
                    outptr1[1] = r0[9];
                    outptr1[2] = r0[10];
                    outptr1[3] = r0[11];
                    outptr1[4] = r0[12];
                    outptr1[5] = r0[13];
                    outptr1[6] = r0[14];
                    outptr1[7] = r0[15];

                    r0 += 16;
                    outptr0 += 8;
                    outptr1 += 8;
                }
            }
        }

        return 0;
    }

    return 0;
}

int Packing_x86::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
        top_blob.cstep = w * elempack / out_elempack;
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

} // namespace ncnn

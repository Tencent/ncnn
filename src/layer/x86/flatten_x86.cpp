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

#include "flatten_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif // __SSE2__

#include "x86_usability.h"

namespace ncnn {

Flatten_x86::Flatten_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int Flatten_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

    if (elembits == 8)
        return forward_int8(bottom_blob, top_blob, opt);

    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h * d;

    int total = size * channels * elempack;

    int out_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX512F__
        out_elempack = total % 16 == 0 ? 16 : total % 8 == 0 ? 8 : total % 4 == 0 ? 4 : 1;
#elif __AVX__
        out_elempack = total % 8 == 0 ? 8 : total % 4 == 0 ? 4 : 1;
#else
        out_elempack = total % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (out_elempack == 1)
    {
        return Flatten::forward(bottom_blob, top_blob, opt);
    }

    if (dims == 2 && elempack == 1) // out_elempack == 4 || out_elempack == 8 || out_elempack == 16
    {
        top_blob = bottom_blob;
        top_blob.dims = 1;
        top_blob.w = total / out_elempack;
        top_blob.h = 1;
        top_blob.cstep = top_blob.w;
        top_blob.elemsize = out_elemsize;
        top_blob.elempack = out_elempack;
        return 0;
    }

    top_blob.create(total / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (dims == 2)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16) // out_elempack == 16
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                float* outptr0 = (float*)top_blob + w * i * 16;
                float* outptr1 = (float*)top_blob + w * (i * 16 + 1);
                float* outptr2 = (float*)top_blob + w * (i * 16 + 2);
                float* outptr3 = (float*)top_blob + w * (i * 16 + 3);
                float* outptr4 = (float*)top_blob + w * (i * 16 + 4);
                float* outptr5 = (float*)top_blob + w * (i * 16 + 5);
                float* outptr6 = (float*)top_blob + w * (i * 16 + 6);
                float* outptr7 = (float*)top_blob + w * (i * 16 + 7);
                float* outptr8 = (float*)top_blob + w * (i * 16 + 8);
                float* outptr9 = (float*)top_blob + w * (i * 16 + 9);
                float* outptra = (float*)top_blob + w * (i * 16 + 10);
                float* outptrb = (float*)top_blob + w * (i * 16 + 11);
                float* outptrc = (float*)top_blob + w * (i * 16 + 12);
                float* outptrd = (float*)top_blob + w * (i * 16 + 13);
                float* outptre = (float*)top_blob + w * (i * 16 + 14);
                float* outptrf = (float*)top_blob + w * (i * 16 + 15);

                int j = 0;
                for (; j + 15 < w; j += 16)
                {
                    __m512 _r0 = _mm512_loadu_ps(ptr);
                    __m512 _r1 = _mm512_loadu_ps(ptr + 16);
                    __m512 _r2 = _mm512_loadu_ps(ptr + 16 * 2);
                    __m512 _r3 = _mm512_loadu_ps(ptr + 16 * 3);
                    __m512 _r4 = _mm512_loadu_ps(ptr + 16 * 4);
                    __m512 _r5 = _mm512_loadu_ps(ptr + 16 * 5);
                    __m512 _r6 = _mm512_loadu_ps(ptr + 16 * 6);
                    __m512 _r7 = _mm512_loadu_ps(ptr + 16 * 7);
                    __m512 _r8 = _mm512_loadu_ps(ptr + 16 * 8);
                    __m512 _r9 = _mm512_loadu_ps(ptr + 16 * 9);
                    __m512 _ra = _mm512_loadu_ps(ptr + 16 * 10);
                    __m512 _rb = _mm512_loadu_ps(ptr + 16 * 11);
                    __m512 _rc = _mm512_loadu_ps(ptr + 16 * 12);
                    __m512 _rd = _mm512_loadu_ps(ptr + 16 * 13);
                    __m512 _re = _mm512_loadu_ps(ptr + 16 * 14);
                    __m512 _rf = _mm512_loadu_ps(ptr + 16 * 15);

                    transpose16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);

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
                    ptr += 256;
                }
                for (; j < w; j++)
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
        }
#endif // __AVX512F__

        if (elempack == 8) // out_elempack == 8 || out_elempack == 16
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                float* outptr0 = (float*)top_blob + w * i * 8;
                float* outptr1 = (float*)top_blob + w * (i * 8 + 1);
                float* outptr2 = (float*)top_blob + w * (i * 8 + 2);
                float* outptr3 = (float*)top_blob + w * (i * 8 + 3);
                float* outptr4 = (float*)top_blob + w * (i * 8 + 4);
                float* outptr5 = (float*)top_blob + w * (i * 8 + 5);
                float* outptr6 = (float*)top_blob + w * (i * 8 + 6);
                float* outptr7 = (float*)top_blob + w * (i * 8 + 7);

                int j = 0;
                for (; j + 7 < w; j += 8)
                {
                    __m256 _row0 = _mm256_loadu_ps(ptr);
                    __m256 _row1 = _mm256_loadu_ps(ptr + 8);
                    __m256 _row2 = _mm256_loadu_ps(ptr + 16);
                    __m256 _row3 = _mm256_loadu_ps(ptr + 24);
                    __m256 _row4 = _mm256_loadu_ps(ptr + 32);
                    __m256 _row5 = _mm256_loadu_ps(ptr + 40);
                    __m256 _row6 = _mm256_loadu_ps(ptr + 48);
                    __m256 _row7 = _mm256_loadu_ps(ptr + 56);

                    transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);

                    _mm256_storeu_ps(outptr0, _row0);
                    _mm256_storeu_ps(outptr1, _row1);
                    _mm256_storeu_ps(outptr2, _row2);
                    _mm256_storeu_ps(outptr3, _row3);
                    _mm256_storeu_ps(outptr4, _row4);
                    _mm256_storeu_ps(outptr5, _row5);
                    _mm256_storeu_ps(outptr6, _row6);
                    _mm256_storeu_ps(outptr7, _row7);

                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                    outptr3 += 8;
                    outptr4 += 8;
                    outptr5 += 8;
                    outptr6 += 8;
                    outptr7 += 8;
                    ptr += 64;
                }
                for (; j < w; j++)
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
        }
#endif // __AVX__

        if (elempack == 4) // out_elempack == 4 || out_elempack == 8 || out_elempack == 16
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                float* outptr0 = (float*)top_blob + w * i * 4;
                float* outptr1 = (float*)top_blob + w * (i * 4 + 1);
                float* outptr2 = (float*)top_blob + w * (i * 4 + 2);
                float* outptr3 = (float*)top_blob + w * (i * 4 + 3);

                int j = 0;
                for (; j + 3 < w; j += 4)
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
                for (; j < w; j++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }
        }
#endif // __SSE2__
    }

    if (dims == 3 || dims == 4)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16) // out_elempack == 16
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr0 = (float*)top_blob + size * q * 16;
                float* outptr1 = (float*)top_blob + size * (q * 16 + 1);
                float* outptr2 = (float*)top_blob + size * (q * 16 + 2);
                float* outptr3 = (float*)top_blob + size * (q * 16 + 3);
                float* outptr4 = (float*)top_blob + size * (q * 16 + 4);
                float* outptr5 = (float*)top_blob + size * (q * 16 + 5);
                float* outptr6 = (float*)top_blob + size * (q * 16 + 6);
                float* outptr7 = (float*)top_blob + size * (q * 16 + 7);
                float* outptr8 = (float*)top_blob + size * (q * 16 + 8);
                float* outptr9 = (float*)top_blob + size * (q * 16 + 9);
                float* outptra = (float*)top_blob + size * (q * 16 + 10);
                float* outptrb = (float*)top_blob + size * (q * 16 + 11);
                float* outptrc = (float*)top_blob + size * (q * 16 + 12);
                float* outptrd = (float*)top_blob + size * (q * 16 + 13);
                float* outptre = (float*)top_blob + size * (q * 16 + 14);
                float* outptrf = (float*)top_blob + size * (q * 16 + 15);

                int i = 0;
                for (; i + 15 < size; i += 16)
                {
                    __m512 _r0 = _mm512_loadu_ps(ptr);
                    __m512 _r1 = _mm512_loadu_ps(ptr + 16);
                    __m512 _r2 = _mm512_loadu_ps(ptr + 16 * 2);
                    __m512 _r3 = _mm512_loadu_ps(ptr + 16 * 3);
                    __m512 _r4 = _mm512_loadu_ps(ptr + 16 * 4);
                    __m512 _r5 = _mm512_loadu_ps(ptr + 16 * 5);
                    __m512 _r6 = _mm512_loadu_ps(ptr + 16 * 6);
                    __m512 _r7 = _mm512_loadu_ps(ptr + 16 * 7);
                    __m512 _r8 = _mm512_loadu_ps(ptr + 16 * 8);
                    __m512 _r9 = _mm512_loadu_ps(ptr + 16 * 9);
                    __m512 _ra = _mm512_loadu_ps(ptr + 16 * 10);
                    __m512 _rb = _mm512_loadu_ps(ptr + 16 * 11);
                    __m512 _rc = _mm512_loadu_ps(ptr + 16 * 12);
                    __m512 _rd = _mm512_loadu_ps(ptr + 16 * 13);
                    __m512 _re = _mm512_loadu_ps(ptr + 16 * 14);
                    __m512 _rf = _mm512_loadu_ps(ptr + 16 * 15);

                    transpose16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);

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
                    ptr += 256;
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
        }
#endif // __AVX512F__

        if (elempack == 8) // out_elempack == 8 || out_elempack == 16
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr0 = (float*)top_blob + size * q * 8;
                float* outptr1 = (float*)top_blob + size * (q * 8 + 1);
                float* outptr2 = (float*)top_blob + size * (q * 8 + 2);
                float* outptr3 = (float*)top_blob + size * (q * 8 + 3);
                float* outptr4 = (float*)top_blob + size * (q * 8 + 4);
                float* outptr5 = (float*)top_blob + size * (q * 8 + 5);
                float* outptr6 = (float*)top_blob + size * (q * 8 + 6);
                float* outptr7 = (float*)top_blob + size * (q * 8 + 7);

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

                    transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);

                    _mm256_storeu_ps(outptr0, _row0);
                    _mm256_storeu_ps(outptr1, _row1);
                    _mm256_storeu_ps(outptr2, _row2);
                    _mm256_storeu_ps(outptr3, _row3);
                    _mm256_storeu_ps(outptr4, _row4);
                    _mm256_storeu_ps(outptr5, _row5);
                    _mm256_storeu_ps(outptr6, _row6);
                    _mm256_storeu_ps(outptr7, _row7);

                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                    outptr3 += 8;
                    outptr4 += 8;
                    outptr5 += 8;
                    outptr6 += 8;
                    outptr7 += 8;
                    ptr += 64;
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
        }
#endif // __AVX__

        if (elempack == 4) // out_elempack == 4 || out_elempack == 8 || out_elempack == 16
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr0 = (float*)top_blob + size * q * 4;
                float* outptr1 = (float*)top_blob + size * (q * 4 + 1);
                float* outptr2 = (float*)top_blob + size * (q * 4 + 2);
                float* outptr3 = (float*)top_blob + size * (q * 4 + 3);

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
        }
#endif // __SSE2__

        if (elempack == 1) // out_elempack == 4 || out_elempack == 8 || out_elempack == 16
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = (float*)top_blob + size * q;

                int i = 0;
#if __AVX__
                for (; i + 7 < size; i += 8)
                {
                    __m256 _v = _mm256_loadu_ps(ptr);
                    _mm256_storeu_ps(outptr, _v);
                    ptr += 8;
                    outptr += 8;
                }
#endif
                for (; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
            }
        }
    }

    return 0;
}

int Flatten_x86::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h * d;

    int total = size * channels * elempack;

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        out_elempack = total % 8 == 0 ? 8 : 1;
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (out_elempack == 1)
    {
        return Flatten::forward(bottom_blob, top_blob, opt);
    }

    if (dims == 2 && elempack == 1) // out_elempack == 8
    {
        top_blob = bottom_blob;
        top_blob.dims = 1;
        top_blob.w = total / out_elempack;
        top_blob.h = 1;
        top_blob.cstep = top_blob.w;
        top_blob.elemsize = out_elemsize;
        top_blob.elempack = out_elempack;
        return 0;
    }

    top_blob.create(total / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (dims == 2)
    {
        if (elempack == 8) // out_elempack == 8
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const signed char* ptr = bottom_blob.row<const signed char>(i);
                signed char* outptr0 = (signed char*)top_blob + w * i * 8;
                signed char* outptr1 = (signed char*)top_blob + w * (i * 8 + 1);
                signed char* outptr2 = (signed char*)top_blob + w * (i * 8 + 2);
                signed char* outptr3 = (signed char*)top_blob + w * (i * 8 + 3);
                signed char* outptr4 = (signed char*)top_blob + w * (i * 8 + 4);
                signed char* outptr5 = (signed char*)top_blob + w * (i * 8 + 5);
                signed char* outptr6 = (signed char*)top_blob + w * (i * 8 + 6);
                signed char* outptr7 = (signed char*)top_blob + w * (i * 8 + 7);

                int j = 0;
                for (; j < w; j++)
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
        }
    }

    if (dims == 3 || dims == 4)
    {
        if (elempack == 8) // out_elempack == 8
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const signed char* ptr = bottom_blob.channel(q);
                signed char* outptr0 = (signed char*)top_blob + size * q * 8;
                signed char* outptr1 = (signed char*)top_blob + size * (q * 8 + 1);
                signed char* outptr2 = (signed char*)top_blob + size * (q * 8 + 2);
                signed char* outptr3 = (signed char*)top_blob + size * (q * 8 + 3);
                signed char* outptr4 = (signed char*)top_blob + size * (q * 8 + 4);
                signed char* outptr5 = (signed char*)top_blob + size * (q * 8 + 5);
                signed char* outptr6 = (signed char*)top_blob + size * (q * 8 + 6);
                signed char* outptr7 = (signed char*)top_blob + size * (q * 8 + 7);

                int i = 0;
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
        }

        if (elempack == 1) // out_elempack == 8
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const signed char* ptr = bottom_blob.channel(q);
                signed char* outptr = (signed char*)top_blob + size * q;

                int i = 0;
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

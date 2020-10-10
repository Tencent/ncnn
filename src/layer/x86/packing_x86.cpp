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
#if __AVX__
#include "avx_usability.h"
#endif // __AVX__

#include "packing_x86.h"

namespace ncnn {

Packing_x86::Packing_x86()
{
    support_packing = true;
}

int Packing_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    bool elemtype_is_fp32 = (elemsize == 4u && elempack == 1) || (elemsize == 32u && elempack == 8);
    if (use_padding)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    if (!elemtype_is_fp32)
    {
        // non-fp32 type
        return Packing::forward(bottom_blob, top_blob, opt);
    }

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
        if (dims == 3 && channels * elempack % out_elempack != 0)
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
                const float* r0 = bottom_blob.row(i * 8);
                const float* r1 = bottom_blob.row(i * 8 + 1);
                const float* r2 = bottom_blob.row(i * 8 + 2);
                const float* r3 = bottom_blob.row(i * 8 + 3);
                const float* r4 = bottom_blob.row(i * 8 + 4);
                const float* r5 = bottom_blob.row(i * 8 + 5);
                const float* r6 = bottom_blob.row(i * 8 + 6);
                const float* r7 = bottom_blob.row(i * 8 + 7);

                float* outptr = top_blob.row(i);

#if __AVX__
                int nn = w >> 3;
                int remain = w & 7;
                for (; nn > 0; nn--)
                {
                    __m256 _row0 = _mm256_loadu_ps(r0);
                    __m256 _row1 = _mm256_loadu_ps(r1);
                    __m256 _row2 = _mm256_loadu_ps(r2);
                    __m256 _row3 = _mm256_loadu_ps(r3);
                    __m256 _row4 = _mm256_loadu_ps(r4);
                    __m256 _row5 = _mm256_loadu_ps(r5);
                    __m256 _row6 = _mm256_loadu_ps(r6);
                    __m256 _row7 = _mm256_loadu_ps(r7);
                    transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
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
#else
                int remain = w;
#endif

                for (; remain > 0; remain--)
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
#if __AVX__
                int nn = w >> 3;
                int remain = w & 7;
#else
                int remain = w;
#endif

#if __AVX__
                for (; nn > 0; nn--)
                {
                    __m256 _row0 = _mm256_loadu_ps(r0);
                    __m256 _row1 = _mm256_loadu_ps(r0 + 8);
                    __m256 _row2 = _mm256_loadu_ps(r0 + 16);
                    __m256 _row3 = _mm256_loadu_ps(r0 + 24);
                    __m256 _row4 = _mm256_loadu_ps(r0 + 32);
                    __m256 _row5 = _mm256_loadu_ps(r0 + 40);
                    __m256 _row6 = _mm256_loadu_ps(r0 + 48);
                    __m256 _row7 = _mm256_loadu_ps(r0 + 56);
                    transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
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
#endif
                for (; remain > 0; remain--)
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

    if (dims == 3)
    {
        int size = w * h;
        int outc = channels * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

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

#if __AVX__
                int nn = size >> 3;
                int remain = size & 7;
#else
                int remain = size;
#endif

#if __AVX__
                for (; nn > 0; nn--)
                {
                    __m256 _row0 = _mm256_loadu_ps(r0);
                    __m256 _row1 = _mm256_loadu_ps(r1);
                    __m256 _row2 = _mm256_loadu_ps(r2);
                    __m256 _row3 = _mm256_loadu_ps(r3);
                    __m256 _row4 = _mm256_loadu_ps(r4);
                    __m256 _row5 = _mm256_loadu_ps(r5);
                    __m256 _row6 = _mm256_loadu_ps(r6);
                    __m256 _row7 = _mm256_loadu_ps(r7);
                    transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
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
#endif
                for (; remain > 0; remain--)
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
#if __AVX__
                int nn = size >> 3;
                int remain = size & 7;
#else
                int remain = size;
#endif

#if __AVX__
                for (; nn > 0; nn--)
                {
                    __m256 _row0 = _mm256_loadu_ps(r0);
                    __m256 _row1 = _mm256_loadu_ps(r0 + 8);
                    __m256 _row2 = _mm256_loadu_ps(r0 + 16);
                    __m256 _row3 = _mm256_loadu_ps(r0 + 24);
                    __m256 _row4 = _mm256_loadu_ps(r0 + 32);
                    __m256 _row5 = _mm256_loadu_ps(r0 + 40);
                    __m256 _row6 = _mm256_loadu_ps(r0 + 48);
                    __m256 _row7 = _mm256_loadu_ps(r0 + 56);
                    transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
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
#endif

                for (; remain > 0; remain--)
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

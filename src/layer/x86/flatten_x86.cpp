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

#include "flatten_x86.h"

namespace ncnn {

Flatten_x86::Flatten_x86()
{
#if __AVX__
    support_packing = true;
#endif // __AVX__
}

int Flatten_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

#if __AVX__
    if (opt.use_packing_layout)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        size_t elemsize = bottom_blob.elemsize;
        int elempack = bottom_blob.elempack;
        int size = w * h;

        int total = size * channels * elempack;
        int out_elempack = total % 8 == 0 ? 8 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 2 && elempack == 1)
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

        if (dims == 2 && elempack == 8)
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

            return 0;
        }

        if (dims == 3 && elempack == 8)
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

            return 0;
        }

        if (dims == 3 && elempack == 1 && out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = (float*)top_blob + size * q;

                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    __m256 _v = _mm256_loadu_ps(ptr);
                    _mm256_storeu_ps(outptr, _v);
                    ptr += 8;
                    outptr += 8;
                }
                for (; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
            }

            return 0;
        }

    }  // opt.use_packing_layout
#endif // __AVX__

    return Flatten::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn

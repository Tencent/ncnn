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

#include "reshape_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include "avx_usability.h"
#endif // __AVX__
#endif // __SSE2__

namespace ncnn {

Reshape_x86::Reshape_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int Reshape_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elempack = bottom_blob.elempack;

    if (permute == 1)
    {
        // TODO implement permute on-the-fly
        Option opt_pack = opt;
        opt_pack.blob_allocator = opt.workspace_allocator;

        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);

        Mat top_blob_unpacked;
        int ret = Reshape::forward(bottom_blob_unpacked, top_blob_unpacked, opt_pack);
        if (ret != 0)
            return ret;

        int out_elempack = 1;
#if __SSE2__
        if (opt.use_packing_layout)
        {
            // resolve dst_elempack
            int dims = top_blob_unpacked.dims;
#if __AVX__
            if (dims == 1) out_elempack = top_blob_unpacked.w % 8 == 0 ? 8 : top_blob_unpacked.w % 4 == 0 ? 4 : 1;
            if (dims == 2) out_elempack = top_blob_unpacked.h % 8 == 0 ? 8 : top_blob_unpacked.h % 4 == 0 ? 4 : 1;
            if (dims == 3) out_elempack = top_blob_unpacked.c % 8 == 0 ? 8 : top_blob_unpacked.c % 4 == 0 ? 4 : 1;
#else
            if (dims == 1) out_elempack = top_blob_unpacked.w % 4 == 0 ? 4 : 1;
            if (dims == 2) out_elempack = top_blob_unpacked.h % 4 == 0 ? 4 : 1;
            if (dims == 3) out_elempack = top_blob_unpacked.c % 4 == 0 ? 4 : 1;
#endif
        }
#endif // __SSE2__
        convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);

        return 0;
    }

    if (ndim == 1)
    {
        // flatten
        flatten(bottom_blob, top_blob, opt);
        if (top_blob.empty())
            return -100;

        return 0;
    }

    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    int total = bottom_blob.w * bottom_blob.h * bottom_blob.c * elempack;

    if (ndim == 2)
    {
        int _w = w;
        int _h = h;

        if (_w == 0)
            _w = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (_h == 0)
            _h = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;

        if (_w == -1)
            _w = total / _h;
        if (_h == -1)
            _h = total / _w;

        int out_elempack = 1;
#if __SSE2__
        if (opt.use_packing_layout)
        {
#if __AVX__
            out_elempack = _h % 8 == 0 ? 8 : _h % 4 == 0 ? 4 : 1;
#else
            out_elempack = _h % 4 == 0 ? 4 : 1;
#endif
        }
#endif // __SSE2__
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 2 && bottom_blob.h == _h && elempack == out_elempack)
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
            top_blob.w = _w;
            top_blob.h = _h;
            top_blob.cstep = _w * _h;
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

        top_blob.create(_w, _h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int outw = top_blob.w;
        int outh = top_blob.h;

#if __SSE2__
#if __AVX__
        if (out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
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

                    transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);

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
            for (int i = 0; i < outh; i++)
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

    if (ndim == 3)
    {
        int _w = w;
        int _h = h;
        int _c = c;

        if (_w == 0)
            _w = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (_h == 0)
            _h = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (_c == 0)
            _c = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;

        if (_w == -1)
            _w = total / _c / _h;
        if (_h == -1)
            _h = total / _c / _w;
        if (_c == -1)
            _c = total / _h / _w;

        int out_elempack = 1;
#if __SSE2__
        if (opt.use_packing_layout)
        {
#if __AVX__
            out_elempack = _c % 8 == 0 ? 8 : _c % 4 == 0 ? 4 : 1;
#else
            out_elempack = _c % 4 == 0 ? 4 : 1;
#endif
        }
#endif // __SSE2__
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 3 && bottom_blob.c == _c && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            top_blob.w = _w;
            top_blob.h = _h;
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

        top_blob.create(_w, _h, _c / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int size = top_blob.w * top_blob.h;

#if __SSE2__
#if __AVX__
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

                    transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);

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

} // namespace ncnn

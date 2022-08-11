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

#include "binaryop_x86.h"

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

#include <math.h>

namespace ncnn {

BinaryOp_x86::BinaryOp_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

template<typename Op>
static int binary_op_2_3_4_20(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = b.w;
    int h = b.h;
    int d = b.d;
    int channels = b.c;
    int elempack = b.elempack;
    int size = w * h * d * elempack;

    // type 2 3 4 20
    c.create_like(b, opt.blob_allocator);
    if (c.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float a0 = a[0];
        const float* ptr = b.channel(q);
        float* outptr = c.channel(q);

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _a0_avx512 = _mm512_set1_ps(a0);
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m512 _outp = op.func_pack16(_a0_avx512, _p);
            _mm512_storeu_ps(outptr, _outp);
            ptr += 16;
            outptr += 16;
        }
#endif // __AVX512F__
        __m256 _a0_avx = _mm256_set1_ps(a0);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _outp = op.func_pack8(_a0_avx, _p);
            _mm256_storeu_ps(outptr, _outp);
            ptr += 8;
            outptr += 8;
        }
#endif // __AVX__
        __m128 _a0 = _mm_set1_ps(a0);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_load_ps(ptr);
            __m128 _outp = op.func_pack4(_a0, _p);
            _mm_store_ps(outptr, _outp);
            ptr += 4;
            outptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *outptr = op.func(a0, *ptr);
            ptr += 1;
            outptr += 1;
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_6_11_16_25(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;
    int size = w * h * d * elempack;

    // type 6 11 16 25
    c.create_like(a, opt.blob_allocator);
    if (c.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = a.channel(q);
        const float b0 = b[0];
        float* outptr = c.channel(q);

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _b0_avx512 = _mm512_set1_ps(b0);
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m512 _outp = op.func_pack16(_p, _b0_avx512);
            _mm512_storeu_ps(outptr, _outp);
            ptr += 16;
            outptr += 16;
        }
#endif // __AVX512F__
        __m256 _b0_avx = _mm256_set1_ps(b0);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _outp = op.func_pack8(_p, _b0_avx);
            _mm256_storeu_ps(outptr, _outp);
            ptr += 8;
            outptr += 8;
        }
#endif // __AVX__
        __m128 _b0 = _mm_set1_ps(b0);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_load_ps(ptr);
            __m128 _outp = op.func_pack4(_p, _b0);
            _mm_store_ps(outptr, _outp);
            ptr += 4;
            outptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *outptr = op.func(*ptr, b0);
            ptr += 1;
            outptr += 1;
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_7_13_19_29(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;
    int size = w * h * d * elempack;

    // type 7 13 19 29
    c.create_like(a, opt.blob_allocator);
    if (c.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = a.channel(q);
        const float* ptr1 = b.channel(q);
        float* outptr = c.channel(q);

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m512 _p1 = _mm512_loadu_ps(ptr1);
            __m512 _outp = op.func_pack16(_p, _p1);
            _mm512_storeu_ps(outptr, _outp);
            ptr += 16;
            ptr1 += 16;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _p1 = _mm256_loadu_ps(ptr1);
            __m256 _outp = op.func_pack8(_p, _p1);
            _mm256_storeu_ps(outptr, _outp);
            ptr += 8;
            ptr1 += 8;
            outptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_load_ps(ptr);
            __m128 _p1 = _mm_load_ps(ptr1);
            __m128 _outp = op.func_pack4(_p, _p1);
            _mm_store_ps(outptr, _outp);
            ptr += 4;
            ptr1 += 4;
            outptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *outptr = op.func(*ptr, *ptr1);
            ptr += 1;
            ptr1 += 1;
            outptr += 1;
        }
    }

    return 0;
}

#if __SSE2__
#if __AVX__
#if __AVX512F__
// broadcasting rule
// https://github.com/Tencent/ncnn/wiki/binaryop-broadcasting

template<typename Op>
static int binary_op_pack16(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int size = w * h * d;
    size_t elemsize = a.elemsize;
    int elempack = a.elempack;

    int w1 = b.w;
    int h1 = b.h;
    int d1 = b.d;
    int channels1 = b.c;
    int size1 = w1 * h1 * d1;
    size_t elemsize1 = b.elemsize;
    int elempack1 = b.elempack;

    if (a.dims == 4)
    {
        if (b.dims == 4)
        {
            // type 29
            return binary_op_7_13_19_29<Op>(a, b, c, opt);
        }

        c.create(w, h, d, channels, elemsize, elempack, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 3)
        {
            // type 28
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int z = 0; z < d; z++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        __m512 _b0 = _mm512_loadu_ps(ptr1);
                        for (int x = 0; x < w; x++)
                        {
                            __m512 _p = _mm512_loadu_ps(ptr);
                            __m512 _outp = op.func_pack16(_p, _b0);
                            _mm512_storeu_ps(outptr, _outp);
                            ptr += 16;
                            outptr += 16;
                        }

                        ptr1 += 16;
                    }
                }
            }

            return 0;
        }

        if (b.dims == 2)
        {
            // type 27
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.row(q);
                float* outptr = c.channel(q);

                for (int z = 0; z < d; z++)
                {
                    __m512 _b0 = _mm512_loadu_ps(ptr1);
                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            __m512 _p = _mm512_loadu_ps(ptr);
                            __m512 _outp = op.func_pack16(_p, _b0);
                            _mm512_storeu_ps(outptr, _outp);
                            ptr += 16;
                            outptr += 16;
                        }
                    }

                    ptr1 += 16;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1 && elempack1 == 1)
            {
                // type 25
                return binary_op_6_11_16_25<Op>(a, b, c, opt);
            }

            // type 26
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                __m512 _b0 = _mm512_loadu_ps((const float*)b + q * 16);
                float* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    __m512 _p = _mm512_loadu_ps(ptr);
                    __m512 _outp = op.func_pack16(_p, _b0);
                    _mm512_storeu_ps(outptr, _outp);
                    ptr += 16;
                    outptr += 16;
                }
            }

            return 0;
        }
    }
    else if (a.dims == 3)
    {
        if (b.dims == 4)
        {
            // type 23
            c.create(w1, h1, d1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int z = 0; z < d1; z++)
                {
                    for (int y = 0; y < h1; y++)
                    {
                        __m512 _a0 = _mm512_loadu_ps(ptr);
                        for (int x = 0; x < w1; x++)
                        {
                            __m512 _p = _mm512_loadu_ps(ptr1);
                            __m512 _outp = op.func_pack16(_a0, _p);
                            _mm512_storeu_ps(outptr, _outp);
                            ptr1 += 16;
                            outptr += 16;
                        }

                        ptr += 16;
                    }
                }
            }

            return 0;
        }

        if (b.dims == 3)
        {
            if (w1 == 1 && h1 == 1 && channels1 == channels)
            {
                // special type 1
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* b0 = b.channel(q);
                    float* outptr = c.channel(q);
                    __m512 _b0 = _mm512_loadu_ps(b0);
                    for (int i = 0; i < size; i++)
                    {
                        __m512 _p = _mm512_loadu_ps(ptr);
                        __m512 _outp = op.func_pack16(_p, _b0);
                        _mm512_storeu_ps(outptr, _outp);
                        ptr += 16;
                        outptr += 16;
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels1 == 1 && elempack1 == 1)
            {
                // special type 2
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b;
                    float* outptr = c.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        __m512 _p = _mm512_loadu_ps(ptr);
                        __m512 _p1 = _mm512_set1_ps(*ptr1);
                        __m512 _outp = op.func_pack16(_p, _p1);
                        _mm512_storeu_ps(outptr, _outp);
                        ptr += 16;
                        ptr1 += 1;
                        outptr += 16;
                    }
                }

                return 0;
            }

            if (w == 1 && h == 1 && channels1 == channels)
            {
                // special type 3
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* a0 = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);
                    __m512 _a0 = _mm512_loadu_ps(a0);
                    for (int i = 0; i < size1; i++)
                    {
                        __m512 _p1 = _mm512_loadu_ps(ptr1);
                        __m512 _outp = op.func_pack16(_a0, _p1);
                        _mm512_storeu_ps(outptr, _outp);
                        ptr1 += 16;
                        outptr += 16;
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels == 1 && elempack == 1)
            {
                // special type 4
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a;
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        __m512 _p = _mm512_set1_ps(*ptr);
                        __m512 _p1 = _mm512_loadu_ps(ptr1);
                        __m512 _outp = op.func_pack16(_p, _p1);
                        _mm512_storeu_ps(outptr, _outp);
                        ptr += 1;
                        ptr1 += 16;
                        outptr += 16;
                    }
                }

                return 0;
            }

            if (w != 1 && w1 == 1 && h1 == h && channels1 == channels)
            {
                // special type 5
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int y = 0; y < h; y++)
                    {
                        __m512 _p1 = _mm512_loadu_ps(ptr1 + y * 16);
                        for (int x = 0; x < w; x++)
                        {
                            __m512 _p = _mm512_loadu_ps(ptr);
                            __m512 _outp = op.func_pack16(_p, _p1);
                            _mm512_storeu_ps(outptr, _outp);

                            ptr += 16;
                            outptr += 16;
                        }
                    }
                }

                return 0;
            }

            if (w1 == w && h != 1 && h1 == 1 && channels1 == channels)
            {
                // special type 6
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            __m512 _p = _mm512_loadu_ps(ptr);
                            __m512 _p1 = _mm512_loadu_ps(ptr1 + x * 16);
                            __m512 _outp = op.func_pack16(_p, _p1);
                            _mm512_storeu_ps(outptr, _outp);

                            ptr += 16;
                            outptr += 16;
                        }
                    }
                }

                return 0;
            }

            if (w1 != 1 && w == 1 && h1 == h && channels1 == channels)
            {
                // special type 7
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int y = 0; y < h1; y++)
                    {
                        __m512 _p = _mm512_loadu_ps(ptr + y * 16);
                        for (int x = 0; x < w1; x++)
                        {
                            __m512 _p1 = _mm512_loadu_ps(ptr1);
                            __m512 _outp = op.func_pack16(_p, _p1);
                            _mm512_storeu_ps(outptr, _outp);

                            ptr1 += 16;
                            outptr += 16;
                        }
                    }
                }

                return 0;
            }

            if (w1 == w && h1 != 1 && h == 1 && channels1 == channels)
            {
                // special type 8
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int y = 0; y < h1; y++)
                    {
                        for (int x = 0; x < w1; x++)
                        {
                            __m512 _p = _mm512_loadu_ps(ptr + x * 16);
                            __m512 _p1 = _mm512_loadu_ps(ptr1);
                            __m512 _outp = op.func_pack16(_p, _p1);
                            _mm512_storeu_ps(outptr, _outp);

                            ptr1 += 16;
                            outptr += 16;
                        }
                    }
                }

                return 0;
            }

            // type 19
            return binary_op_7_13_19_29<Op>(a, b, c, opt);
        }

        c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 18
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.row(q);
                float* outptr = c.channel(q);

                for (int y = 0; y < h; y++)
                {
                    __m512 _b0 = _mm512_loadu_ps(ptr1);
                    for (int x = 0; x < w; x++)
                    {
                        __m512 _p = _mm512_loadu_ps(ptr);
                        __m512 _outp = op.func_pack16(_p, _b0);
                        _mm512_storeu_ps(outptr, _outp);
                        ptr += 16;
                        outptr += 16;
                    }

                    ptr1 += 16;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1 && elempack1 == 1)
            {
                // type 16
                return binary_op_6_11_16_25<Op>(a, b, c, opt);
            }

            // type 17
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                __m512 _b0 = _mm512_loadu_ps((const float*)b + q * 16);
                float* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    __m512 _p = _mm512_loadu_ps(ptr);
                    __m512 _outp = op.func_pack16(_p, _b0);
                    _mm512_storeu_ps(outptr, _outp);
                    ptr += 16;
                    outptr += 16;
                }
            }

            return 0;
        }
    }
    else if (a.dims == 2)
    {
        if (b.dims == 4)
        {
            // type 22
            c.create(w1, h1, d1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const float* ptr = a.row(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int z = 0; z < d1; z++)
                {
                    __m512 _a0 = _mm512_loadu_ps(ptr);
                    for (int y = 0; y < h1; y++)
                    {
                        for (int x = 0; x < w1; x++)
                        {
                            __m512 _p = _mm512_loadu_ps(ptr1);
                            __m512 _outp = op.func_pack16(_a0, _p);
                            _mm512_storeu_ps(outptr, _outp);
                            ptr1 += 16;
                            outptr += 16;
                        }
                    }

                    ptr += 16;
                }
            }

            return 0;
        }

        if (b.dims == 3)
        {
            // type 14
            c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const float* ptr = a.row(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int y = 0; y < h1; y++)
                {
                    __m512 _a0 = _mm512_loadu_ps(ptr);
                    for (int x = 0; x < w1; x++)
                    {
                        __m512 _p1 = _mm512_loadu_ps(ptr1);
                        __m512 _outp = op.func_pack16(_a0, _p1);
                        _mm512_storeu_ps(outptr, _outp);
                        ptr1 += 16;
                        outptr += 16;
                    }

                    ptr += 16;
                }
            }

            return 0;
        }

        c.create(w, h, elemsize, elempack, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 13
            return binary_op_7_13_19_29<Op>(a, b, c, opt);
        }

        if (b.dims == 1)
        {
            c.create(w, h, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1 && elempack1 == 1)
            {
                // type 11
                return binary_op_6_11_16_25<Op>(a, b, c, opt);
            }

            // type 12
            const float* ptr = a;
            const float* ptr1 = b;
            float* outptr = c;

            for (int y = 0; y < h; y++)
            {
                __m512 _b0 = _mm512_loadu_ps(ptr1);
                for (int x = 0; x < w; x++)
                {
                    __m512 _p = _mm512_loadu_ps(ptr);
                    __m512 _outp = op.func_pack16(_p, _b0);
                    _mm512_storeu_ps(outptr, _outp);
                    ptr += 16;
                    outptr += 16;
                }

                ptr1 += 16;
            }

            return 0;
        }
    }
    else if (a.dims == 1)
    {
        if (a.w == 1 && elempack == 1)
        {
            // type 2 3 4 20
            return binary_op_2_3_4_20<Op>(a, b, c, opt);
        }

        if (b.dims == 4)
        {
            // type 21
            c.create(w1, h1, d1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                __m512 _a0 = _mm512_loadu_ps((const float*)a + q * 16);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    __m512 _p1 = _mm512_loadu_ps(ptr1);
                    __m512 _outp = op.func_pack16(_a0, _p1);
                    _mm512_storeu_ps(outptr, _outp);
                    ptr1 += 16;
                    outptr += 16;
                }
            }

            return 0;
        }

        if (b.dims == 3)
        {
            // type 9
            c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                __m512 _a0 = _mm512_loadu_ps((const float*)a + q * 16);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    __m512 _p1 = _mm512_loadu_ps(ptr1);
                    __m512 _outp = op.func_pack16(_a0, _p1);
                    _mm512_storeu_ps(outptr, _outp);
                    ptr1 += 16;
                    outptr += 16;
                }
            }

            return 0;
        }

        if (b.dims == 2)
        {
            // type 8
            c.create(w1, h1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            const float* ptr = a;
            const float* ptr1 = b;
            float* outptr = c;

            for (int y = 0; y < h1; y++)
            {
                __m512 _a0 = _mm512_loadu_ps(ptr);
                for (int x = 0; x < w1; x++)
                {
                    __m512 _p1 = _mm512_loadu_ps(ptr1);
                    __m512 _outp = op.func_pack16(_a0, _p1);
                    _mm512_storeu_ps(outptr, _outp);
                    ptr1 += 16;
                    outptr += 16;
                }

                ptr += 16;
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1 && elempack1 == 1)
            {
                // type 6
                return binary_op_6_11_16_25<Op>(a, b, c, opt);
            }

            // type 7
            binary_op_7_13_19_29<Op>(a, b, c, opt);
        }
    }

    return 0;
}
#endif // __AVX512F__

template<typename Op>
static int binary_op_pack8(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int size = w * h * d;
    size_t elemsize = a.elemsize;
    int elempack = a.elempack;

    int w1 = b.w;
    int h1 = b.h;
    int d1 = b.d;
    int channels1 = b.c;
    int size1 = w1 * h1 * d1;
    size_t elemsize1 = b.elemsize;
    int elempack1 = b.elempack;

    if (a.dims == 4)
    {
        if (b.dims == 4)
        {
            // type 29
            return binary_op_7_13_19_29<Op>(a, b, c, opt);
        }

        c.create(w, h, d, channels, elemsize, elempack, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 3)
        {
            // type 28
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int z = 0; z < d; z++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        __m256 _b0 = _mm256_loadu_ps(ptr1);
                        for (int x = 0; x < w; x++)
                        {
                            __m256 _p = _mm256_loadu_ps(ptr);
                            __m256 _outp = op.func_pack8(_p, _b0);
                            _mm256_storeu_ps(outptr, _outp);
                            ptr += 8;
                            outptr += 8;
                        }

                        ptr1 += 8;
                    }
                }
            }

            return 0;
        }

        if (b.dims == 2)
        {
            // type 27
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.row(q);
                float* outptr = c.channel(q);

                for (int z = 0; z < d; z++)
                {
                    __m256 _b0 = _mm256_loadu_ps(ptr1);
                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            __m256 _p = _mm256_loadu_ps(ptr);
                            __m256 _outp = op.func_pack8(_p, _b0);
                            _mm256_storeu_ps(outptr, _outp);
                            ptr += 8;
                            outptr += 8;
                        }
                    }

                    ptr1 += 8;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1 && elempack1 == 1)
            {
                // type 25
                return binary_op_6_11_16_25<Op>(a, b, c, opt);
            }

            // type 26
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                __m256 _b0 = _mm256_loadu_ps((const float*)b + q * 8);
                float* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    __m256 _outp = op.func_pack8(_p, _b0);
                    _mm256_storeu_ps(outptr, _outp);
                    ptr += 8;
                    outptr += 8;
                }
            }

            return 0;
        }
    }
    else if (a.dims == 3)
    {
        if (b.dims == 4)
        {
            // type 23
            c.create(w1, h1, d1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int z = 0; z < d1; z++)
                {
                    for (int y = 0; y < h1; y++)
                    {
                        __m256 _a0 = _mm256_loadu_ps(ptr);
                        for (int x = 0; x < w1; x++)
                        {
                            __m256 _p = _mm256_loadu_ps(ptr1);
                            __m256 _outp = op.func_pack8(_a0, _p);
                            _mm256_storeu_ps(outptr, _outp);
                            ptr1 += 8;
                            outptr += 8;
                        }

                        ptr += 8;
                    }
                }
            }

            return 0;
        }

        if (b.dims == 3)
        {
            if (w1 == 1 && h1 == 1 && channels1 == channels)
            {
                // special type 1
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* b0 = b.channel(q);
                    float* outptr = c.channel(q);
                    __m256 _b0 = _mm256_loadu_ps(b0);
                    for (int i = 0; i < size; i++)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        __m256 _outp = op.func_pack8(_p, _b0);
                        _mm256_storeu_ps(outptr, _outp);
                        ptr += 8;
                        outptr += 8;
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels1 == 1 && elempack1 == 1)
            {
                // special type 2
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b;
                    float* outptr = c.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        __m256 _p1 = _mm256_broadcast_ss(ptr1);
                        __m256 _outp = op.func_pack8(_p, _p1);
                        _mm256_storeu_ps(outptr, _outp);
                        ptr += 8;
                        ptr1 += 1;
                        outptr += 8;
                    }
                }

                return 0;
            }

            if (w == 1 && h == 1 && channels1 == channels)
            {
                // special type 3
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* a0 = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);
                    __m256 _a0 = _mm256_loadu_ps(a0);
                    for (int i = 0; i < size1; i++)
                    {
                        __m256 _p1 = _mm256_loadu_ps(ptr1);
                        __m256 _outp = op.func_pack8(_a0, _p1);
                        _mm256_storeu_ps(outptr, _outp);
                        ptr1 += 8;
                        outptr += 8;
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels == 1 && elempack == 1)
            {
                // special type 4
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a;
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        __m256 _p = _mm256_broadcast_ss(ptr);
                        __m256 _p1 = _mm256_loadu_ps(ptr1);
                        __m256 _outp = op.func_pack8(_p, _p1);
                        _mm256_storeu_ps(outptr, _outp);
                        ptr += 1;
                        ptr1 += 8;
                        outptr += 8;
                    }
                }

                return 0;
            }

            if (w != 1 && w1 == 1 && h1 == h && channels1 == channels)
            {
                // special type 5
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int y = 0; y < h; y++)
                    {
                        __m256 _p1 = _mm256_loadu_ps(ptr1 + y * 8);
                        for (int x = 0; x < w; x++)
                        {
                            __m256 _p = _mm256_loadu_ps(ptr);
                            __m256 _outp = op.func_pack8(_p, _p1);
                            _mm256_storeu_ps(outptr, _outp);

                            ptr += 8;
                            outptr += 8;
                        }
                    }
                }

                return 0;
            }

            if (w1 == w && h != 1 && h1 == 1 && channels1 == channels)
            {
                // special type 6
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            __m256 _p = _mm256_loadu_ps(ptr);
                            __m256 _p1 = _mm256_loadu_ps(ptr1 + x * 8);
                            __m256 _outp = op.func_pack8(_p, _p1);
                            _mm256_storeu_ps(outptr, _outp);

                            ptr += 8;
                            outptr += 8;
                        }
                    }
                }

                return 0;
            }

            if (w1 != 1 && w == 1 && h1 == h && channels1 == channels)
            {
                // special type 7
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int y = 0; y < h1; y++)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr + y * 8);
                        for (int x = 0; x < w1; x++)
                        {
                            __m256 _p1 = _mm256_loadu_ps(ptr1);
                            __m256 _outp = op.func_pack8(_p, _p1);
                            _mm256_storeu_ps(outptr, _outp);

                            ptr1 += 8;
                            outptr += 8;
                        }
                    }
                }

                return 0;
            }

            if (w1 == w && h1 != 1 && h == 1 && channels1 == channels)
            {
                // special type 8
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int y = 0; y < h1; y++)
                    {
                        for (int x = 0; x < w1; x++)
                        {
                            __m256 _p = _mm256_loadu_ps(ptr + x * 8);
                            __m256 _p1 = _mm256_loadu_ps(ptr1);
                            __m256 _outp = op.func_pack8(_p, _p1);
                            _mm256_storeu_ps(outptr, _outp);

                            ptr1 += 8;
                            outptr += 8;
                        }
                    }
                }

                return 0;
            }

            // type 19
            return binary_op_7_13_19_29<Op>(a, b, c, opt);
        }

        c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 18
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.row(q);
                float* outptr = c.channel(q);

                for (int y = 0; y < h; y++)
                {
                    __m256 _b0 = _mm256_loadu_ps(ptr1);
                    for (int x = 0; x < w; x++)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        __m256 _outp = op.func_pack8(_p, _b0);
                        _mm256_storeu_ps(outptr, _outp);
                        ptr += 8;
                        outptr += 8;
                    }

                    ptr1 += 8;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1 && elempack1 == 1)
            {
                // type 16
                return binary_op_6_11_16_25<Op>(a, b, c, opt);
            }

            // type 17
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                __m256 _b0 = _mm256_loadu_ps((const float*)b + q * 8);
                float* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    __m256 _outp = op.func_pack8(_p, _b0);
                    _mm256_storeu_ps(outptr, _outp);
                    ptr += 8;
                    outptr += 8;
                }
            }

            return 0;
        }
    }
    else if (a.dims == 2)
    {
        if (b.dims == 4)
        {
            // type 22
            c.create(w1, h1, d1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const float* ptr = a.row(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int z = 0; z < d1; z++)
                {
                    __m256 _a0 = _mm256_loadu_ps(ptr);
                    for (int y = 0; y < h1; y++)
                    {
                        for (int x = 0; x < w1; x++)
                        {
                            __m256 _p = _mm256_loadu_ps(ptr1);
                            __m256 _outp = op.func_pack8(_a0, _p);
                            _mm256_storeu_ps(outptr, _outp);
                            ptr1 += 8;
                            outptr += 8;
                        }
                    }

                    ptr += 8;
                }
            }

            return 0;
        }

        if (b.dims == 3)
        {
            // type 14
            c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const float* ptr = a.row(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int y = 0; y < h1; y++)
                {
                    __m256 _a0 = _mm256_loadu_ps(ptr);
                    for (int x = 0; x < w1; x++)
                    {
                        __m256 _p1 = _mm256_loadu_ps(ptr1);
                        __m256 _outp = op.func_pack8(_a0, _p1);
                        _mm256_storeu_ps(outptr, _outp);
                        ptr1 += 8;
                        outptr += 8;
                    }

                    ptr += 8;
                }
            }

            return 0;
        }

        c.create(w, h, elemsize, elempack, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 13
            return binary_op_7_13_19_29<Op>(a, b, c, opt);
        }

        if (b.dims == 1)
        {
            c.create(w, h, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1 && elempack1 == 1)
            {
                // type 11
                return binary_op_6_11_16_25<Op>(a, b, c, opt);
            }

            // type 12
            const float* ptr = a;
            const float* ptr1 = b;
            float* outptr = c;

            for (int y = 0; y < h; y++)
            {
                __m256 _b0 = _mm256_loadu_ps(ptr1);
                for (int x = 0; x < w; x++)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    __m256 _outp = op.func_pack8(_p, _b0);
                    _mm256_storeu_ps(outptr, _outp);
                    ptr += 8;
                    outptr += 8;
                }

                ptr1 += 8;
            }

            return 0;
        }
    }
    else if (a.dims == 1)
    {
        if (a.w == 1 && elempack == 1)
        {
            // type 2 3 4 20
            return binary_op_2_3_4_20<Op>(a, b, c, opt);
        }

        if (b.dims == 4)
        {
            // type 21
            c.create(w1, h1, d1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                __m256 _a0 = _mm256_loadu_ps((const float*)a + q * 8);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    __m256 _p1 = _mm256_loadu_ps(ptr1);
                    __m256 _outp = op.func_pack8(_a0, _p1);
                    _mm256_storeu_ps(outptr, _outp);
                    ptr1 += 8;
                    outptr += 8;
                }
            }

            return 0;
        }

        if (b.dims == 3)
        {
            // type 9
            c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                __m256 _a0 = _mm256_loadu_ps((const float*)a + q * 8);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    __m256 _p1 = _mm256_loadu_ps(ptr1);
                    __m256 _outp = op.func_pack8(_a0, _p1);
                    _mm256_storeu_ps(outptr, _outp);
                    ptr1 += 8;
                    outptr += 8;
                }
            }

            return 0;
        }

        if (b.dims == 2)
        {
            // type 8
            c.create(w1, h1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            const float* ptr = a;
            const float* ptr1 = b;
            float* outptr = c;

            for (int y = 0; y < h1; y++)
            {
                __m256 _a0 = _mm256_loadu_ps(ptr);
                for (int x = 0; x < w1; x++)
                {
                    __m256 _p1 = _mm256_loadu_ps(ptr1);
                    __m256 _outp = op.func_pack8(_a0, _p1);
                    _mm256_storeu_ps(outptr, _outp);
                    ptr1 += 8;
                    outptr += 8;
                }

                ptr += 8;
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1 && elempack1 == 1)
            {
                // type 6
                return binary_op_6_11_16_25<Op>(a, b, c, opt);
            }

            // type 7
            binary_op_7_13_19_29<Op>(a, b, c, opt);
        }
    }

    return 0;
}
#endif // __AVX__

template<typename Op>
static int binary_op_pack4(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int size = w * h * d;
    size_t elemsize = a.elemsize;
    int elempack = a.elempack;

    int w1 = b.w;
    int h1 = b.h;
    int d1 = b.d;
    int channels1 = b.c;
    int size1 = w1 * h1 * d1;
    size_t elemsize1 = b.elemsize;
    int elempack1 = b.elempack;

    if (a.dims == 4)
    {
        if (b.dims == 4)
        {
            // type 29
            return binary_op_7_13_19_29<Op>(a, b, c, opt);
        }

        c.create(w, h, d, channels, elemsize, elempack, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 3)
        {
            // type 28
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int z = 0; z < d; z++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        __m128 _b0 = _mm_loadu_ps(ptr1);
                        for (int x = 0; x < w; x++)
                        {
                            __m128 _p = _mm_loadu_ps(ptr);
                            __m128 _outp = op.func_pack4(_p, _b0);
                            _mm_storeu_ps(outptr, _outp);
                            ptr += 4;
                            outptr += 4;
                        }

                        ptr1 += 4;
                    }
                }
            }

            return 0;
        }

        if (b.dims == 2)
        {
            // type 27
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.row(q);
                float* outptr = c.channel(q);

                for (int z = 0; z < d; z++)
                {
                    __m128 _b0 = _mm_loadu_ps(ptr1);
                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            __m128 _p = _mm_loadu_ps(ptr);
                            __m128 _outp = op.func_pack4(_p, _b0);
                            _mm_storeu_ps(outptr, _outp);
                            ptr += 4;
                            outptr += 4;
                        }
                    }

                    ptr1 += 4;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1 && elempack1 == 1)
            {
                // type 25
                return binary_op_6_11_16_25<Op>(a, b, c, opt);
            }

            // type 26
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                __m128 _b0 = _mm_loadu_ps((const float*)b + q * 4);
                float* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    __m128 _outp = op.func_pack4(_p, _b0);
                    _mm_storeu_ps(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }
            }

            return 0;
        }
    }
    else if (a.dims == 3)
    {
        if (b.dims == 4)
        {
            // type 23
            c.create(w1, h1, d1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int z = 0; z < d1; z++)
                {
                    for (int y = 0; y < h1; y++)
                    {
                        __m128 _a0 = _mm_loadu_ps(ptr);
                        for (int x = 0; x < w1; x++)
                        {
                            __m128 _p = _mm_loadu_ps(ptr1);
                            __m128 _outp = op.func_pack4(_a0, _p);
                            _mm_storeu_ps(outptr, _outp);
                            ptr1 += 4;
                            outptr += 4;
                        }

                        ptr += 4;
                    }
                }
            }

            return 0;
        }

        if (b.dims == 3)
        {
            if (w1 == 1 && h1 == 1 && channels1 == channels)
            {
                // special type 1
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = a.channel(q);
                    float* outptr = c.channel(q);
                    const float* b0 = b.channel(q);
                    __m128 _b0 = _mm_loadu_ps(b0);
                    for (int i = 0; i < size; i++)
                    {
                        __m128 _p = _mm_loadu_ps(ptr);
                        __m128 _outp = op.func_pack4(_p, _b0);
                        _mm_storeu_ps(outptr, _outp);
                        ptr += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels1 == 1 && elempack1 == 1)
            {
                // special type 2
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b;
                    float* outptr = c.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        __m128 _p = _mm_loadu_ps(ptr);
                        __m128 _p1 = _mm_set1_ps(*ptr1);
                        __m128 _outp = op.func_pack4(_p, _p1);
                        _mm_storeu_ps(outptr, _outp);
                        ptr += 4;
                        ptr1 += 1;
                        outptr += 4;
                    }
                }

                return 0;
            }

            if (w == 1 && h == 1 && channels1 == channels)
            {
                // special type 3
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* a0 = a.channel(q);
                    float* outptr = c.channel(q);
                    const float* ptr1 = b.channel(q);
                    __m128 _a0 = _mm_loadu_ps(a0);
                    for (int i = 0; i < size1; i++)
                    {
                        __m128 _p1 = _mm_loadu_ps(ptr1);
                        __m128 _outp = op.func_pack4(_a0, _p1);
                        _mm_storeu_ps(outptr, _outp);
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels == 1 && elempack == 1)
            {
                // special type 4
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a;
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        __m128 _p = _mm_set1_ps(*ptr);
                        __m128 _p1 = _mm_loadu_ps(ptr1);
                        __m128 _outp = op.func_pack4(_p, _p1);
                        _mm_storeu_ps(outptr, _outp);
                        ptr += 1;
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            if (w != 1 && w1 == 1 && h1 == h && channels1 == channels)
            {
                // special type 5
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int y = 0; y < h; y++)
                    {
                        __m128 _p1 = _mm_loadu_ps(ptr1 + y * 4);
                        for (int x = 0; x < w; x++)
                        {
                            __m128 _p = _mm_loadu_ps(ptr);
                            __m128 _outp = op.func_pack4(_p, _p1);
                            _mm_storeu_ps(outptr, _outp);

                            ptr += 4;
                            outptr += 4;
                        }
                    }
                }

                return 0;
            }

            if (w1 == w && h != 1 && h1 == 1 && channels1 == channels)
            {
                // special type 6
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            __m128 _p = _mm_loadu_ps(ptr);
                            __m128 _p1 = _mm_loadu_ps(ptr1 + x * 4);
                            __m128 _outp = op.func_pack4(_p, _p1);
                            _mm_storeu_ps(outptr, _outp);

                            ptr += 4;
                            outptr += 4;
                        }
                    }
                }

                return 0;
            }

            if (w1 != 1 && w == 1 && h1 == h && channels1 == channels)
            {
                // special type 7
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int y = 0; y < h1; y++)
                    {
                        __m128 _p = _mm_loadu_ps(ptr + y * 4);
                        for (int x = 0; x < w1; x++)
                        {
                            __m128 _p1 = _mm_loadu_ps(ptr1);
                            __m128 _outp = op.func_pack4(_p, _p1);
                            _mm_storeu_ps(outptr, _outp);

                            ptr1 += 4;
                            outptr += 4;
                        }
                    }
                }

                return 0;
            }

            if (w1 == w && h1 != 1 && h == 1 && channels1 == channels)
            {
                // special type 8
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int y = 0; y < h1; y++)
                    {
                        for (int x = 0; x < w1; x++)
                        {
                            __m128 _p = _mm_loadu_ps(ptr + x * 4);
                            __m128 _p1 = _mm_loadu_ps(ptr1);
                            __m128 _outp = op.func_pack4(_p, _p1);
                            _mm_storeu_ps(outptr, _outp);

                            ptr1 += 4;
                            outptr += 4;
                        }
                    }
                }

                return 0;
            }

            // type 19
            return binary_op_7_13_19_29<Op>(a, b, c, opt);
        }

        c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 18
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.row<const float>(q);
                float* outptr = c.channel(q);

                for (int y = 0; y < h; y++)
                {
                    __m128 _b0 = _mm_loadu_ps(ptr1);
                    for (int x = 0; x < w; x++)
                    {
                        __m128 _p = _mm_loadu_ps(ptr);
                        __m128 _outp = op.func_pack4(_p, _b0);
                        _mm_storeu_ps(outptr, _outp);
                        ptr += 4;
                        outptr += 4;
                    }

                    ptr1 += 4;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1 && elempack1 == 1)
            {
                // type 16
                return binary_op_6_11_16_25<Op>(a, b, c, opt);
            }

            // type 17
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                __m128 _b0 = _mm_loadu_ps((const float*)b + q * 4);
                float* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    __m128 _outp = op.func_pack4(_p, _b0);
                    _mm_storeu_ps(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }
            }

            return 0;
        }
    }
    else if (a.dims == 2)
    {
        if (b.dims == 4)
        {
            // type 22
            c.create(w1, h1, d1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const float* ptr = a.row(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int z = 0; z < d1; z++)
                {
                    __m128 _a0 = _mm_loadu_ps(ptr);
                    for (int y = 0; y < h1; y++)
                    {
                        for (int x = 0; x < w1; x++)
                        {
                            __m128 _p = _mm_loadu_ps(ptr1);
                            __m128 _outp = op.func_pack4(_a0, _p);
                            _mm_storeu_ps(outptr, _outp);
                            ptr1 += 4;
                            outptr += 4;
                        }
                    }

                    ptr += 4;
                }
            }

            return 0;
        }

        if (b.dims == 3)
        {
            // type 14
            c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const float* ptr = a.row<const float>(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int y = 0; y < h1; y++)
                {
                    __m128 _a0 = _mm_loadu_ps(ptr);
                    for (int x = 0; x < w1; x++)
                    {
                        __m128 _p1 = _mm_loadu_ps(ptr1);
                        __m128 _outp = op.func_pack4(_a0, _p1);
                        _mm_storeu_ps(outptr, _outp);
                        ptr1 += 4;
                        outptr += 4;
                    }

                    ptr += 4;
                }
            }

            return 0;
        }

        c.create(w, h, elemsize, elempack, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 13
            return binary_op_7_13_19_29<Op>(a, b, c, opt);
        }

        if (b.dims == 1)
        {
            c.create(w, h, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1 && elempack1 == 1)
            {
                // type 11
                return binary_op_6_11_16_25<Op>(a, b, c, opt);
            }

            // type 12
            const float* ptr = a;
            const float* ptr1 = b;
            float* outptr = c;

            for (int y = 0; y < h; y++)
            {
                __m128 _b0 = _mm_loadu_ps(ptr1);
                for (int x = 0; x < w; x++)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    __m128 _outp = op.func_pack4(_p, _b0);
                    _mm_storeu_ps(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }

                ptr1 += 4;
            }

            return 0;
        }
    }
    else if (a.dims == 1)
    {
        if (a.w == 1 && elempack == 1)
        {
            // type 2 3 4 20
            return binary_op_2_3_4_20<Op>(a, b, c, opt);
        }

        if (b.dims == 4)
        {
            // type 21
            c.create(w1, h1, d1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                __m128 _a0 = _mm_loadu_ps((const float*)a + q * 4);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    __m128 _p1 = _mm_loadu_ps(ptr1);
                    __m128 _outp = op.func_pack4(_a0, _p1);
                    _mm_storeu_ps(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            return 0;
        }

        if (b.dims == 3)
        {
            // type 9
            c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                __m128 _a0 = _mm_loadu_ps((const float*)a + q * 4);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    __m128 _p1 = _mm_loadu_ps(ptr1);
                    __m128 _outp = op.func_pack4(_a0, _p1);
                    _mm_storeu_ps(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            return 0;
        }

        if (b.dims == 2)
        {
            // type 8
            c.create(w1, h1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            const float* ptr = a;
            const float* ptr1 = b;
            float* outptr = c;

            for (int y = 0; y < h1; y++)
            {
                __m128 _a0 = _mm_loadu_ps(ptr);
                for (int x = 0; x < w1; x++)
                {
                    __m128 _p1 = _mm_loadu_ps(ptr1);
                    __m128 _outp = op.func_pack4(_a0, _p1);
                    _mm_storeu_ps(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }

                ptr += 4;
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1 && elempack1 == 1)
            {
                // type 6
                return binary_op_6_11_16_25<Op>(a, b, c, opt);
            }

            // type 7
            binary_op_7_13_19_29<Op>(a, b, c, opt);
        }
    }

    return 0;
}
#endif // __SSE2__

template<typename Op>
static int binary_op_scalar_inplace(Mat& a, float b, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _b_avx512 = _mm512_set1_ps(b);
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _p = op.func_pack16(_p, _b_avx512);
            _mm512_storeu_ps(ptr, _p);
            ptr += 16;
        }
#endif // __AVX512F__
        __m256 _b_avx = _mm256_set1_ps(b);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _p = op.func_pack8(_p, _b_avx);
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
        }
#endif // __AVX__
        __m128 _b = _mm_set1_ps((float)b);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_load_ps(ptr);
            _p = op.func_pack4(_p, _b);
            _mm_store_ps(ptr, _p);
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = op.func(*ptr, b);
            ptr++;
        }
    }

    return 0;
}

namespace BinaryOp_x86_functor {

struct binary_op_add
{
    float func(const float& x, const float& y) const
    {
        return x + y;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_add_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_add_ps(x, y);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_add_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_sub
{
    float func(const float& x, const float& y) const
    {
        return x - y;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_sub_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_sub_ps(x, y);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_sub_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_mul
{
    float func(const float& x, const float& y) const
    {
        return x * y;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_mul_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_mul_ps(x, y);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_mul_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_div
{
    float func(const float& x, const float& y) const
    {
        return x / y;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_div_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_div_ps(x, y);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_div_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_max
{
    float func(const float& x, const float& y) const
    {
        return std::max(x, y);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_max_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_max_ps(x, y);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_max_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_min
{
    float func(const float& x, const float& y) const
    {
        return std::min(x, y);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_min_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_min_ps(x, y);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_min_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_pow
{
    float func(const float& x, const float& y) const
    {
        return (float)pow(x, y);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return pow_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return pow256_ps(x, y);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return pow512_ps(x, y);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_rsub
{
    float func(const float& x, const float& y) const
    {
        return y - x;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_sub_ps(y, x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_sub_ps(y, x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_sub_ps(y, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_rdiv
{
    float func(const float& x, const float& y) const
    {
        return y / x;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_div_ps(y, x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_div_ps(y, x);
    }
#if __AVX512F__
    __m512 func_pack16(const __m512& x, const __m512& y) const
    {
        return _mm512_div_ps(y, x);
    }
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
};

} // namespace BinaryOp_x86_functor

int BinaryOp_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if __SSE2__
    using namespace BinaryOp_x86_functor;

    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    int elempack = bottom_blob.elempack;
    int elempack1 = bottom_blob1.elempack;

#if __AVX__
#if __AVX512F__
    if (elempack == 16 || elempack1 == 16)
    {
        if (op_type == Operation_ADD)
            return binary_op_pack16<binary_op_add>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op_pack16<binary_op_sub>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op_pack16<binary_op_mul>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op_pack16<binary_op_div>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op_pack16<binary_op_max>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op_pack16<binary_op_min>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op_pack16<binary_op_pow>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op_pack16<binary_op_sub>(bottom_blob1, bottom_blob, top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op_pack16<binary_op_div>(bottom_blob1, bottom_blob, top_blob, opt);
    }
#endif // __AVX512F__

    if (elempack == 8 || elempack1 == 8)
    {
        if (op_type == Operation_ADD)
            return binary_op_pack8<binary_op_add>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op_pack8<binary_op_sub>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op_pack8<binary_op_mul>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op_pack8<binary_op_div>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op_pack8<binary_op_max>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op_pack8<binary_op_min>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op_pack8<binary_op_pow>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op_pack8<binary_op_sub>(bottom_blob1, bottom_blob, top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op_pack8<binary_op_div>(bottom_blob1, bottom_blob, top_blob, opt);
    }
#endif // __AVX__

    if (elempack == 4 || elempack1 == 4)
    {
        if (op_type == Operation_ADD)
            return binary_op_pack4<binary_op_add>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op_pack4<binary_op_sub>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op_pack4<binary_op_mul>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op_pack4<binary_op_div>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op_pack4<binary_op_max>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op_pack4<binary_op_min>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op_pack4<binary_op_pow>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op_pack4<binary_op_sub>(bottom_blob1, bottom_blob, top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op_pack4<binary_op_div>(bottom_blob1, bottom_blob, top_blob, opt);
    }
#endif // __SSE2__

    return BinaryOp::forward(bottom_blobs, top_blobs, opt);
}

int BinaryOp_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    using namespace BinaryOp_x86_functor;

    if (op_type == Operation_ADD)
        return binary_op_scalar_inplace<binary_op_add>(bottom_top_blob, b, opt);

    if (op_type == Operation_SUB)
        return binary_op_scalar_inplace<binary_op_sub>(bottom_top_blob, b, opt);

    if (op_type == Operation_MUL)
        return binary_op_scalar_inplace<binary_op_mul>(bottom_top_blob, b, opt);

    if (op_type == Operation_DIV)
        return binary_op_scalar_inplace<binary_op_div>(bottom_top_blob, b, opt);

    if (op_type == Operation_MAX)
        return binary_op_scalar_inplace<binary_op_max>(bottom_top_blob, b, opt);

    if (op_type == Operation_MIN)
        return binary_op_scalar_inplace<binary_op_min>(bottom_top_blob, b, opt);

    if (op_type == Operation_POW)
        return binary_op_scalar_inplace<binary_op_pow>(bottom_top_blob, b, opt);

    if (op_type == Operation_RSUB)
        return binary_op_scalar_inplace<binary_op_rsub>(bottom_top_blob, b, opt);

    if (op_type == Operation_RDIV)
        return binary_op_scalar_inplace<binary_op_rdiv>(bottom_top_blob, b, opt);

    return 0;
}

} // namespace ncnn

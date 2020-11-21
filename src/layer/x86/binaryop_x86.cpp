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

#include "sse_mathfun.h"
#if __AVX__
#include "avx_mathfun.h"
#endif // __AVX__

#include <math.h>

namespace ncnn {

BinaryOp_x86::BinaryOp_x86()
{
    support_packing = true;
}

#if __AVX__
// broadcasting rule
// https://github.com/Tencent/ncnn/wiki/binaryop-broadcasting

template<typename Op>
static int binary_op_pack8(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;
    size_t elemsize = a.elemsize;
    int elempack = a.elempack;

    int w1 = b.w;
    int h1 = b.h;
    int channels1 = b.c;
    int size1 = w1 * h1;
    size_t elemsize1 = b.elemsize;
    int elempack1 = b.elempack;

    if (a.dims == 3)
    {
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
                        __m256 _outp = op(_p, _b0);
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
                        __m256 _outp = op(_p, _p1);
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
                        __m256 _outp = op(_a0, _p1);
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
                        __m256 _outp = op(_p, _p1);
                        _mm256_storeu_ps(outptr, _outp);
                        ptr += 1;
                        ptr1 += 8;
                        outptr += 8;
                    }
                }

                return 0;
            }

            // type 19
            c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    __m256 _p1 = _mm256_loadu_ps(ptr1);
                    __m256 _outp = op(_p, _p1);
                    _mm256_storeu_ps(outptr, _outp);
                    ptr += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
            }

            return 0;
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
                        __m256 _outp = op(_p, _b0);
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
                __m256 _b0 = _mm256_set1_ps(b[0]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = a.channel(q);
                    float* outptr = c.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        __m256 _outp = op(_p, _b0);
                        _mm256_storeu_ps(outptr, _outp);
                        ptr += 8;
                        outptr += 8;
                    }
                }

                return 0;
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
                    __m256 _outp = op(_p, _b0);
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
                        __m256 _outp = op(_a0, _p1);
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
            const float* ptr = a;
            const float* ptr1 = b;
            float* outptr = c;
            for (int i = 0; i < size; i++)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _p1 = _mm256_loadu_ps(ptr1);
                __m256 _outp = op(_p, _p1);
                _mm256_storeu_ps(outptr, _outp);
                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, h, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1 && elempack1 == 1)
            {
                // type 11
                __m256 _b0 = _mm256_set1_ps(b[0]);
                const float* ptr = a;
                float* outptr = c;
                for (int i = 0; i < size; i++)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    __m256 _outp = op(_p, _b0);
                    _mm256_storeu_ps(outptr, _outp);
                    ptr += 8;
                    outptr += 8;
                }

                return 0;
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
                    __m256 _outp = op(_p, _b0);
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
            if (b.dims == 3)
            {
                // type 4
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                __m256 _a0 = _mm256_set1_ps(a[0]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int i = 0; i < size1; i++)
                    {
                        __m256 _p1 = _mm256_loadu_ps(ptr1);
                        __m256 _outp = op(_a0, _p1);
                        _mm256_storeu_ps(outptr, _outp);
                        ptr1 += 8;
                        outptr += 8;
                    }
                }

                return 0;
            }

            if (b.dims == 2)
            {
                // type 3
                c.create(w1, h1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                __m256 _a0 = _mm256_set1_ps(a[0]);
                const float* ptr1 = b;
                float* outptr = c;
                for (int i = 0; i < size1; i++)
                {
                    __m256 _p1 = _mm256_loadu_ps(ptr1);
                    __m256 _outp = op(_a0, _p1);
                    _mm256_storeu_ps(outptr, _outp);
                    ptr1 += 8;
                    outptr += 8;
                }

                return 0;
            }

            if (b.dims == 1)
            {
                // type 2
                c.create(w1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                __m256 _a0 = _mm256_set1_ps(a[0]);
                const float* ptr1 = b;
                float* outptr = c;
                for (int i = 0; i < w1; i++)
                {
                    __m256 _p1 = _mm256_loadu_ps(ptr1);
                    __m256 _outp = op(_a0, _p1);
                    _mm256_storeu_ps(outptr, _outp);
                    ptr1 += 8;
                    outptr += 8;
                }

                return 0;
            }
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
                    __m256 _outp = op(_a0, _p1);
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
                    __m256 _outp = op(_a0, _p1);
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
                __m256 _b0 = _mm256_set1_ps(b[0]);
                const float* ptr = a;
                float* outptr = c;
                for (int i = 0; i < w; i++)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    __m256 _outp = op(_p, _b0);
                    _mm256_storeu_ps(outptr, _outp);
                    ptr += 8;
                    outptr += 8;
                }

                return 0;
            }

            // type 7
            const float* ptr = a;
            const float* ptr1 = b;
            float* outptr = c;
            for (int i = 0; i < w; i++)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _p1 = _mm256_loadu_ps(ptr1);
                __m256 _outp = op(_p, _p1);
                _mm256_storeu_ps(outptr, _outp);
                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_scalar_inplace_pack8(Mat& a, float b, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;

    __m256 _b = _mm256_set1_ps(b);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _p = op(_p, _b);
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
        }
    }

    return 0;
}

struct binary_op_add_pack8
{
    __m256 operator()(const __m256& x, const __m256& y) const
    {
        return _mm256_add_ps(x, y);
    }
};

struct binary_op_sub_pack8
{
    __m256 operator()(const __m256& x, const __m256& y) const
    {
        return _mm256_sub_ps(x, y);
    }
};

struct binary_op_mul_pack8
{
    __m256 operator()(const __m256& x, const __m256& y) const
    {
        return _mm256_mul_ps(x, y);
    }
};

struct binary_op_div_pack8
{
    __m256 operator()(const __m256& x, const __m256& y) const
    {
        return _mm256_div_ps(x, y);
    }
};

struct binary_op_max_pack8
{
    __m256 operator()(const __m256& x, const __m256& y) const
    {
        return _mm256_max_ps(x, y);
    }
};

struct binary_op_min_pack8
{
    __m256 operator()(const __m256& x, const __m256& y) const
    {
        return _mm256_min_ps(x, y);
    }
};

struct binary_op_pow_pack8
{
    __m256 operator()(const __m256& x, const __m256& y) const
    {
        return exp256_ps(_mm256_mul_ps(y, log256_ps(x)));
    }
};

struct binary_op_rsub_pack8
{
    __m256 operator()(const __m256& x, const __m256& y) const
    {
        return _mm256_sub_ps(y, x);
    }
};

struct binary_op_rdiv_pack8
{
    __m256 operator()(const __m256& x, const __m256& y) const
    {
        return _mm256_div_ps(y, x);
    }
};
#endif // __AVX__

template<typename Op>
static int binary_op_pack4(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;
    size_t elemsize = a.elemsize;
    int elempack = a.elempack;

    int w1 = b.w;
    int h1 = b.h;
    int channels1 = b.c;
    int size1 = w1 * h1;
    size_t elemsize1 = b.elemsize;
    int elempack1 = b.elempack;

    if (a.dims == 3)
    {
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
                        __m128 _outp = op(_p, _b0);
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
                        __m128 _outp = op(_p, _p1);
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
                        __m128 _outp = op(_a0, _p1);
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
                        __m128 _outp = op(_p, _p1);
                        _mm_storeu_ps(outptr, _outp);
                        ptr += 1;
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            // type 19
            c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    __m128 _p1 = _mm_loadu_ps(ptr1);
                    __m128 _outp = op(_p, _p1);
                    _mm_storeu_ps(outptr, _outp);
                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            return 0;
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
                        __m128 _outp = op(_p, _b0);
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
                __m128 _b0 = _mm_set1_ps(((const float*)b)[0]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = a.channel(q);
                    float* outptr = c.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        __m128 _p = _mm_loadu_ps(ptr);
                        __m128 _outp = op(_p, _b0);
                        _mm_storeu_ps(outptr, _outp);
                        ptr += 4;
                        outptr += 4;
                    }
                }

                return 0;
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
                    __m128 _outp = op(_p, _b0);
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
                        __m128 _outp = op(_a0, _p1);
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
            const float* ptr = a;
            const float* ptr1 = b;
            float* outptr = c;
            for (int i = 0; i < size; i++)
            {
                __m128 _p = _mm_loadu_ps(ptr);
                __m128 _p1 = _mm_loadu_ps(ptr1);
                __m128 _outp = op(_p, _p1);
                _mm_storeu_ps(outptr, _outp);
                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, h, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1 && elempack1 == 1)
            {
                // type 11
                __m128 _b0 = _mm_set1_ps(((const float*)b)[0]);
                const float* ptr = a;
                float* outptr = c;
                for (int i = 0; i < size; i++)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    __m128 _outp = op(_p, _b0);
                    _mm_storeu_ps(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }

                return 0;
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
                    __m128 _outp = op(_p, _b0);
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
            if (b.dims == 3)
            {
                // type 4
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                __m128 _a0 = _mm_set1_ps(((const float*)a)[0]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int i = 0; i < size1; i++)
                    {
                        __m128 _p1 = _mm_loadu_ps(ptr1);
                        __m128 _outp = op(_a0, _p1);
                        _mm_storeu_ps(outptr, _outp);
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            if (b.dims == 2)
            {
                // type 3
                c.create(w1, h1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                __m128 _a0 = _mm_set1_ps(((const float*)a)[0]);
                const float* ptr1 = b;
                float* outptr = c;
                for (int i = 0; i < size1; i++)
                {
                    __m128 _p1 = _mm_loadu_ps(ptr1);
                    __m128 _outp = op(_a0, _p1);
                    _mm_storeu_ps(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }

                return 0;
            }

            if (b.dims == 1)
            {
                // type 2
                c.create(w1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                __m128 _a0 = _mm_set1_ps(((const float*)a)[0]);
                const float* ptr1 = b;
                float* outptr = c;
                for (int i = 0; i < w1; i++)
                {
                    __m128 _p1 = _mm_loadu_ps(ptr1);
                    __m128 _outp = op(_a0, _p1);
                    _mm_storeu_ps(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }

                return 0;
            }
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
                    __m128 _outp = op(_a0, _p1);
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
                    __m128 _outp = op(_a0, _p1);
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
                __m128 _b0 = _mm_set1_ps(((const float*)b)[0]);
                const float* ptr = a;
                float* outptr = c;
                for (int i = 0; i < w; i++)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    __m128 _outp = op(_p, _b0);
                    _mm_storeu_ps(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }

                return 0;
            }

            // type 7
            const float* ptr = a;
            const float* ptr1 = b;
            float* outptr = c;
            for (int i = 0; i < w; i++)
            {
                __m128 _p = _mm_loadu_ps(ptr);
                __m128 _p1 = _mm_loadu_ps(ptr1);
                __m128 _outp = op(_p, _p1);
                _mm_storeu_ps(outptr, _outp);
                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_scalar_inplace_pack4(Mat& a, float b, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;

    __m128 _b = _mm_set1_ps((float)b);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            _p = op(_p, _b);
            _mm_storeu_ps(ptr, _p);
            ptr += 4;
        }
    }

    return 0;
}

struct binary_op_add_pack4
{
    __m128 operator()(const __m128& x, const __m128& y) const
    {
        return _mm_add_ps(x, y);
    }
};

struct binary_op_sub_pack4
{
    __m128 operator()(const __m128& x, const __m128& y) const
    {
        return _mm_sub_ps(x, y);
    }
};

struct binary_op_mul_pack4
{
    __m128 operator()(const __m128& x, const __m128& y) const
    {
        return _mm_mul_ps(x, y);
    }
};

struct binary_op_div_pack4
{
    __m128 operator()(const __m128& x, const __m128& y) const
    {
        return _mm_div_ps(x, y);
    }
};

struct binary_op_max_pack4
{
    __m128 operator()(const __m128& x, const __m128& y) const
    {
        return _mm_max_ps(x, y);
    }
};

struct binary_op_min_pack4
{
    __m128 operator()(const __m128& x, const __m128& y) const
    {
        return _mm_min_ps(x, y);
    }
};

struct binary_op_pow_pack4
{
    __m128 operator()(const __m128& x, const __m128& y) const
    {
        return exp_ps(_mm_mul_ps(y, log_ps(x)));
    }
};

struct binary_op_rsub_pack4
{
    __m128 operator()(const __m128& x, const __m128& y) const
    {
        return _mm_sub_ps(y, x);
    }
};

struct binary_op_rdiv_pack4
{
    __m128 operator()(const __m128& x, const __m128& y) const
    {
        return _mm_div_ps(y, x);
    }
};

int BinaryOp_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    int elempack = bottom_blob.elempack;
    int elempack1 = bottom_blob1.elempack;

#if __AVX__
    if (elempack == 8 || elempack1 == 8)
    {
        if (op_type == Operation_ADD)
            return binary_op_pack8<binary_op_add_pack8>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op_pack8<binary_op_sub_pack8>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op_pack8<binary_op_mul_pack8>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op_pack8<binary_op_div_pack8>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op_pack8<binary_op_max_pack8>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op_pack8<binary_op_min_pack8>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op_pack8<binary_op_pow_pack8>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op_pack8<binary_op_rsub_pack8>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op_pack8<binary_op_rdiv_pack8>(bottom_blob, bottom_blob1, top_blob, opt);
    }
#endif // __AVX__

    if (elempack == 4 || elempack1 == 4)
    {
        if (op_type == Operation_ADD)
            return binary_op_pack4<binary_op_add_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op_pack4<binary_op_sub_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op_pack4<binary_op_mul_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op_pack4<binary_op_div_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op_pack4<binary_op_max_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op_pack4<binary_op_min_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op_pack4<binary_op_pow_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op_pack4<binary_op_rsub_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op_pack4<binary_op_rdiv_pack4>(bottom_blob, bottom_blob1, top_blob, opt);
    }

    return BinaryOp::forward(bottom_blobs, top_blobs, opt);
}

int BinaryOp_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elempack = bottom_top_blob.elempack;

#if __AVX__
    if (elempack == 8)
    {
        if (op_type == Operation_ADD)
            return binary_op_scalar_inplace_pack8<binary_op_add_pack8>(bottom_top_blob, b, opt);

        if (op_type == Operation_SUB)
            return binary_op_scalar_inplace_pack8<binary_op_sub_pack8>(bottom_top_blob, b, opt);

        if (op_type == Operation_MUL)
            return binary_op_scalar_inplace_pack8<binary_op_mul_pack8>(bottom_top_blob, b, opt);

        if (op_type == Operation_DIV)
            return binary_op_scalar_inplace_pack8<binary_op_div_pack8>(bottom_top_blob, b, opt);

        if (op_type == Operation_MAX)
            return binary_op_scalar_inplace_pack8<binary_op_max_pack8>(bottom_top_blob, b, opt);

        if (op_type == Operation_MIN)
            return binary_op_scalar_inplace_pack8<binary_op_min_pack8>(bottom_top_blob, b, opt);

        if (op_type == Operation_POW)
            return binary_op_scalar_inplace_pack8<binary_op_pow_pack8>(bottom_top_blob, b, opt);

        if (op_type == Operation_RSUB)
            return binary_op_scalar_inplace_pack8<binary_op_rsub_pack8>(bottom_top_blob, b, opt);

        if (op_type == Operation_RDIV)
            return binary_op_scalar_inplace_pack8<binary_op_rdiv_pack8>(bottom_top_blob, b, opt);
    }
#endif // __AVX__

    if (elempack == 4)
    {
        if (op_type == Operation_ADD)
            return binary_op_scalar_inplace_pack4<binary_op_add_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_SUB)
            return binary_op_scalar_inplace_pack4<binary_op_sub_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_MUL)
            return binary_op_scalar_inplace_pack4<binary_op_mul_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_DIV)
            return binary_op_scalar_inplace_pack4<binary_op_div_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_MAX)
            return binary_op_scalar_inplace_pack4<binary_op_max_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_MIN)
            return binary_op_scalar_inplace_pack4<binary_op_min_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_POW)
            return binary_op_scalar_inplace_pack4<binary_op_pow_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_RSUB)
            return binary_op_scalar_inplace_pack4<binary_op_rsub_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_RDIV)
            return binary_op_scalar_inplace_pack4<binary_op_rdiv_pack4>(bottom_top_blob, b, opt);
    }

    return BinaryOp::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn

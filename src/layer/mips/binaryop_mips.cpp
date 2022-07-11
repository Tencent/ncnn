// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "binaryop_mips.h"

#include <math.h>

#if __mips_msa
#include <msa.h>
#include "msa_mathfun.h"
#endif // __mips_msa

namespace ncnn {

BinaryOp_mips::BinaryOp_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
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
#if __mips_msa
        v4f32 _a0 = __msa_fill_w_f32(a0);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            v4f32 _outp = op(_a0, _p);
            __msa_st_w((v4i32)_outp, outptr, 0);
            ptr += 4;
            outptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *outptr = op(a0, *ptr);
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
#if __mips_msa
        v4f32 _b0 = __msa_fill_w_f32(b0);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            v4f32 _outp = op(_p, _b0);
            __msa_st_w((v4i32)_outp, outptr, 0);
            ptr += 4;
            outptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *outptr = op(*ptr, b0);
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
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            __builtin_prefetch(ptr1 + 16);
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
            v4f32 _outp = op(_p, _p1);
            __msa_st_w((v4i32)_outp, outptr, 0);
            ptr += 4;
            ptr1 += 4;
            outptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *outptr = op(*ptr, *ptr1);
            ptr += 1;
            ptr1 += 1;
            outptr += 1;
        }
    }

    return 0;
}

#if __mips_msa
// broadcasting rule
// https://github.com/Tencent/ncnn/wiki/binaryop-broadcasting

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
                        v4f32 _b0 = (v4f32)__msa_ld_w(ptr1, 0);
                        for (int x = 0; x < w; x++)
                        {
                            __builtin_prefetch(ptr + 16);
                            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                            v4f32 _outp = op(_p, _b0);
                            __msa_st_w((v4i32)_outp, outptr, 0);
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
                    v4f32 _b0 = (v4f32)__msa_ld_w(ptr1, 0);
                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            __builtin_prefetch(ptr + 16);
                            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                            v4f32 _outp = op(_p, _b0);
                            __msa_st_w((v4i32)_outp, outptr, 0);
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
                v4f32 _b0 = (v4f32)__msa_ld_w((const float*)b + q * 4, 0);
                float* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    __builtin_prefetch(ptr + 16);
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    v4f32 _outp = op(_p, _b0);
                    __msa_st_w((v4i32)_outp, outptr, 0);
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
                        v4f32 _a0 = (v4f32)__msa_ld_w(ptr, 0);
                        for (int x = 0; x < w1; x++)
                        {
                            __builtin_prefetch(ptr1 + 16);
                            v4f32 _p = (v4f32)__msa_ld_w(ptr1, 0);
                            v4f32 _outp = op(_a0, _p);
                            __msa_st_w((v4i32)_outp, outptr, 0);
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
                    const float* b0 = b.channel(q);
                    float* outptr = c.channel(q);
                    v4f32 _b0 = (v4f32)__msa_ld_w(b0, 0);
                    for (int i = 0; i < size; i++)
                    {
                        __builtin_prefetch(ptr + 16);
                        v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                        v4f32 _outp = op(_p, _b0);
                        __msa_st_w((v4i32)_outp, outptr, 0);
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
                        __builtin_prefetch(ptr + 16);
                        v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                        v4f32 _p1 = __msa_fill_w_f32(ptr1[0]);
                        v4f32 _outp = op(_p, _p1);
                        __msa_st_w((v4i32)_outp, outptr, 0);
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
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);
                    v4f32 _a0 = (v4f32)__msa_ld_w(a0, 0);
                    for (int i = 0; i < size1; i++)
                    {
                        __builtin_prefetch(ptr1 + 16);
                        v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                        v4f32 _outp = op(_a0, _p1);
                        __msa_st_w((v4i32)_outp, outptr, 0);
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
                        __builtin_prefetch(ptr + 16);
                        __builtin_prefetch(ptr1 + 16);
                        v4f32 _p = __msa_fill_w_f32(ptr[0]);
                        v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                        v4f32 _outp = op(_p, _p1);
                        __msa_st_w((v4i32)_outp, outptr, 0);
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
                        v4f32 _p1 = (v4f32)__msa_ld_w(ptr1 + y * 4, 0);
                        for (int x = 0; x < w; x++)
                        {
                            __builtin_prefetch(ptr + 16);
                            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                            v4f32 _outp = op(_p, _p1);
                            __msa_st_w((v4i32)_outp, outptr, 0);

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
                            __builtin_prefetch(ptr + 16);
                            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                            v4f32 _p1 = (v4f32)__msa_ld_w(ptr1 + x * 4, 0);
                            v4f32 _outp = op(_p, _p1);
                            __msa_st_w((v4i32)_outp, outptr, 0);

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
                        v4f32 _p = (v4f32)__msa_ld_w(ptr + y * 4, 0);
                        for (int x = 0; x < w1; x++)
                        {
                            __builtin_prefetch(ptr1 + 16);
                            v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                            v4f32 _outp = op(_p, _p1);
                            __msa_st_w((v4i32)_outp, outptr, 0);

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
                            __builtin_prefetch(ptr1 + 16);
                            v4f32 _p = (v4f32)__msa_ld_w(ptr + x * 4, 0);
                            v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                            v4f32 _outp = op(_p, _p1);
                            __msa_st_w((v4i32)_outp, outptr, 0);

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
                const float* ptr1 = b.row(q);
                float* outptr = c.channel(q);

                for (int y = 0; y < h; y++)
                {
                    v4f32 _b0 = (v4f32)__msa_ld_w(ptr1, 0);
                    for (int x = 0; x < w; x++)
                    {
                        __builtin_prefetch(ptr + 16);
                        v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                        v4f32 _outp = op(_p, _b0);
                        __msa_st_w((v4i32)_outp, outptr, 0);
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
                v4f32 _b0 = (v4f32)__msa_ld_w((const float*)b + q * 4, 0);
                float* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    __builtin_prefetch(ptr + 16);
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    v4f32 _outp = op(_p, _b0);
                    __msa_st_w((v4i32)_outp, outptr, 0);
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
                    v4f32 _a0 = (v4f32)__msa_ld_w(ptr, 0);
                    for (int y = 0; y < h1; y++)
                    {
                        for (int x = 0; x < w1; x++)
                        {
                            __builtin_prefetch(ptr1 + 16);
                            v4f32 _p = (v4f32)__msa_ld_w(ptr1, 0);
                            v4f32 _outp = op(_a0, _p);
                            __msa_st_w((v4i32)_outp, outptr, 0);
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
                const float* ptr = a.row(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int y = 0; y < h1; y++)
                {
                    v4f32 _a0 = (v4f32)__msa_ld_w(ptr, 0);
                    for (int x = 0; x < w1; x++)
                    {
                        __builtin_prefetch(ptr1 + 16);
                        v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                        v4f32 _outp = op(_a0, _p1);
                        __msa_st_w((v4i32)_outp, outptr, 0);
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
                v4f32 _b0 = (v4f32)__msa_ld_w(ptr1, 0);
                for (int x = 0; x < w; x++)
                {
                    __builtin_prefetch(ptr + 16);
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    v4f32 _outp = op(_p, _b0);
                    __msa_st_w((v4i32)_outp, outptr, 0);
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
                v4f32 _a0 = (v4f32)__msa_ld_w((const float*)a + q * 4, 0);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    __builtin_prefetch(ptr1 + 16);
                    v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                    v4f32 _outp = op(_a0, _p1);
                    __msa_st_w((v4i32)_outp, outptr, 0);
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
                v4f32 _a0 = (v4f32)__msa_ld_w((const float*)a + q * 4, 0);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    __builtin_prefetch(ptr1 + 16);
                    v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                    v4f32 _outp = op(_a0, _p1);
                    __msa_st_w((v4i32)_outp, outptr, 0);
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
                v4f32 _a0 = (v4f32)__msa_ld_w(ptr, 0);
                for (int x = 0; x < w1; x++)
                {
                    __builtin_prefetch(ptr1 + 16);
                    v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                    v4f32 _outp = op(_a0, _p1);
                    __msa_st_w((v4i32)_outp, outptr, 0);
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
#endif // __mips_msa

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
#if __mips_msa
        v4f32 _b = __msa_fill_w_f32(b);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            _p = op(_p, _b);
            __msa_st_w((v4i32)_p, ptr, 0);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *ptr = op(*ptr, b);
            ptr++;
        }
    }

    return 0;
}

namespace BinaryOp_mips_functor {

#if __mips_msa
#define MAKE_FUNCTION(NAME, IMPL, IMPL4)                       \
    struct NAME                                                \
    {                                                          \
        float operator()(const float& x, const float& y) const \
        {                                                      \
            return IMPL;                                       \
        }                                                      \
        v4f32 operator()(const v4f32& x, const v4f32& y) const \
        {                                                      \
            return IMPL4;                                      \
        }                                                      \
    };
#else
#define MAKE_FUNCTION(NAME, IMPL, IMPL4)                       \
    struct NAME                                                \
    {                                                          \
        float operator()(const float& x, const float& y) const \
        {                                                      \
            return IMPL;                                       \
        }                                                      \
    };
#endif // __mips_msa

// clang-format off
// *INDENT-OFF*
MAKE_FUNCTION(binary_op_add, x + y, __msa_fadd_w(x, y))
MAKE_FUNCTION(binary_op_sub, x - y, __msa_fsub_w(x, y))
MAKE_FUNCTION(binary_op_mul, x * y, __msa_fmul_w(x, y))
MAKE_FUNCTION(binary_op_div, x / y, __msa_fdiv_w(x, y))
MAKE_FUNCTION(binary_op_max, std::max(x, y), __msa_fmax_w(x, y))
MAKE_FUNCTION(binary_op_min, std::min(x, y), __msa_fmin_w(x, y))
MAKE_FUNCTION(binary_op_pow, (float)pow(x, y), pow_ps(x, y))
MAKE_FUNCTION(binary_op_rsub, y - x, __msa_fsub_w(y, x))
MAKE_FUNCTION(binary_op_rdiv, y / x, __msa_fdiv_w(y, x))
// *INDENT-ON*
// clang-format on

#undef MAKE_FUNCTION

} // namespace BinaryOp_mips_functor

int BinaryOp_mips::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if __mips_msa
    using namespace BinaryOp_mips_functor;

    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    int elempack = bottom_blob.elempack;
    int elempack1 = bottom_blob1.elempack;

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
#endif // __mips_msa

    return BinaryOp::forward(bottom_blobs, top_blobs, opt);
}

int BinaryOp_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    using namespace BinaryOp_mips_functor;

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

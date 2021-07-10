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

#if __mips_msa
// broadcasting rule
// https://github.com/Tencent/ncnn/wiki/binaryop-broadcasting

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
                    const float* b0 = b.channel(q);
                    float* outptr = c.channel(q);
                    v4f32 _b0 = (v4f32)__msa_ld_w(b0, 0);
                    for (int i = 0; i < size; i++)
                    {
                        __builtin_prefetch(ptr + 32);
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
                        __builtin_prefetch(ptr + 32);
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
                        __builtin_prefetch(ptr1 + 32);
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
                        __builtin_prefetch(ptr + 8);
                        __builtin_prefetch(ptr1 + 32);
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
                            __builtin_prefetch(ptr + 32);
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
                            __builtin_prefetch(ptr + 32);
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
                            __builtin_prefetch(ptr1 + 32);
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
                            __builtin_prefetch(ptr1 + 32);
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
                    __builtin_prefetch(ptr + 32);
                    __builtin_prefetch(ptr1 + 32);
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                    v4f32 _outp = op(_p, _p1);
                    __msa_st_w((v4i32)_outp, outptr, 0);
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
                const float* ptr1 = b.row(q);
                float* outptr = c.channel(q);

                for (int y = 0; y < h; y++)
                {
                    v4f32 _b0 = (v4f32)__msa_ld_w(ptr1, 0);
                    for (int x = 0; x < w; x++)
                    {
                        __builtin_prefetch(ptr + 32);
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
                v4f32 _b0 = __msa_fill_w_f32(b[0]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = a.channel(q);
                    float* outptr = c.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        __builtin_prefetch(ptr + 32);
                        v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                        v4f32 _outp = op(_p, _b0);
                        __msa_st_w((v4i32)_outp, outptr, 0);
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
                v4f32 _b0 = (v4f32)__msa_ld_w((const float*)b + q * 4, 0);
                float* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    __builtin_prefetch(ptr + 32);
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
                        __builtin_prefetch(ptr1 + 32);
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
            const float* ptr = a;
            const float* ptr1 = b;
            float* outptr = c;
            for (int i = 0; i < size; i++)
            {
                __builtin_prefetch(ptr + 32);
                __builtin_prefetch(ptr1 + 32);
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                v4f32 _outp = op(_p, _p1);
                __msa_st_w((v4i32)_outp, outptr, 0);
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
                v4f32 _b0 = __msa_fill_w_f32(b[0]);
                const float* ptr = a;
                float* outptr = c;
                for (int i = 0; i < size; i++)
                {
                    __builtin_prefetch(ptr + 32);
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    v4f32 _outp = op(_p, _b0);
                    __msa_st_w((v4i32)_outp, outptr, 0);
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
                v4f32 _b0 = (v4f32)__msa_ld_w(ptr1, 0);
                for (int x = 0; x < w; x++)
                {
                    __builtin_prefetch(ptr + 32);
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
            if (b.dims == 3)
            {
                // type 4
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                v4f32 _a0 = __msa_fill_w_f32(a[0]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int i = 0; i < size1; i++)
                    {
                        __builtin_prefetch(ptr1 + 32);
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
                // type 3
                c.create(w1, h1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                v4f32 _a0 = __msa_fill_w_f32(a[0]);
                const float* ptr1 = b;
                float* outptr = c;
                for (int i = 0; i < size1; i++)
                {
                    __builtin_prefetch(ptr1 + 32);
                    v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                    v4f32 _outp = op(_a0, _p1);
                    __msa_st_w((v4i32)_outp, outptr, 0);
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

                v4f32 _a0 = __msa_fill_w_f32(a[0]);
                const float* ptr1 = b;
                float* outptr = c;
                for (int i = 0; i < w1; i++)
                {
                    __builtin_prefetch(ptr1 + 32);
                    v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                    v4f32 _outp = op(_a0, _p1);
                    __msa_st_w((v4i32)_outp, outptr, 0);
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
                v4f32 _a0 = (v4f32)__msa_ld_w((const float*)a + q * 4, 0);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    __builtin_prefetch(ptr1 + 32);
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
                    __builtin_prefetch(ptr1 + 32);
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
                v4f32 _b0 = __msa_fill_w_f32(b[0]);
                const float* ptr = a;
                float* outptr = c;
                for (int i = 0; i < w; i++)
                {
                    __builtin_prefetch(ptr + 32);
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    v4f32 _outp = op(_p, _b0);
                    __msa_st_w((v4i32)_outp, outptr, 0);
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
                __builtin_prefetch(ptr + 32);
                __builtin_prefetch(ptr1 + 32);
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                v4f32 _outp = op(_p, _p1);
                __msa_st_w((v4i32)_outp, outptr, 0);
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

    v4f32 _b = __msa_fill_w_f32(b);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            __builtin_prefetch(ptr + 32);
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            _p = op(_p, _b);
            __msa_st_w((v4i32)_p, ptr, 0);
            ptr += 4;
        }
    }

    return 0;
}

struct binary_op_add_pack4
{
    v4f32 operator()(const v4f32& x, const v4f32& y) const
    {
        return __msa_fadd_w(x, y);
    }
};

struct binary_op_sub_pack4
{
    v4f32 operator()(const v4f32& x, const v4f32& y) const
    {
        return __msa_fsub_w(x, y);
    }
};

struct binary_op_mul_pack4
{
    v4f32 operator()(const v4f32& x, const v4f32& y) const
    {
        return __msa_fmul_w(x, y);
    }
};

struct binary_op_div_pack4
{
    v4f32 operator()(const v4f32& x, const v4f32& y) const
    {
        return __msa_fdiv_w(x, y);
    }
};

struct binary_op_max_pack4
{
    v4f32 operator()(const v4f32& x, const v4f32& y) const
    {
        return __msa_fmax_w(x, y);
    }
};

struct binary_op_min_pack4
{
    v4f32 operator()(const v4f32& x, const v4f32& y) const
    {
        return __msa_fmin_w(x, y);
    }
};

struct binary_op_pow_pack4
{
    v4f32 operator()(const v4f32& x, const v4f32& y) const
    {
        return pow_ps(x, y);
    }
};

struct binary_op_rsub_pack4
{
    v4f32 operator()(const v4f32& x, const v4f32& y) const
    {
        return __msa_fsub_w(y, x);
    }
};

struct binary_op_rdiv_pack4
{
    v4f32 operator()(const v4f32& x, const v4f32& y) const
    {
        return __msa_fdiv_w(y, x);
    }
};
#endif // __mips_msa

int BinaryOp_mips::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

#if __mips_msa
    int elempack = bottom_blob.elempack;
    int elempack1 = bottom_blob1.elempack;

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
#endif // __mips_msa

    return BinaryOp::forward(bottom_blobs, top_blobs, opt);
}

int BinaryOp_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if __mips_msa
    int elempack = bottom_top_blob.elempack;

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
#endif // __mips_msa

    return BinaryOp::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn

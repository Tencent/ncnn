// Xavier Hsinyuan is pleased to support the open source community by making
// ncnn available.
//
// Copyright (C) 2021 Xavier Hsinyuan <me@lstlx.com> All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include "binaryop_riscv.h"

#include <math.h>

#if __riscv_vector
#ifdef RVV_SPEC_0_7
#include "riscv_v_071_fix.h"
#else
#include <riscv_vector.h>
#endif
#include "rvv_mathfun.h"
#include "rvv_mathfun_fp16s.h"
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

BinaryOp_riscv::BinaryOp_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif
}

#if __riscv_vector
template<typename Op>
static int binary_op_rvv(const Mat& a, const Mat& b, Mat& c, const Option& opt)
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
            c.create(w, h, d, channels, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                int n = size * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                    vfloat32m8_t _outp = op(_p, _p1, vl);
                    vse32_v_f32m8(outptr, _outp, vl);

                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
            }

            return 0;
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
                        vfloat32m8_t _b0x = vle32_v_f32m8_f32m1(ptr1);

                        int n = w * elempack;
                        while (n > 0)
                        {
                            word_type vl = vsetvl_e32m8(n);
                            vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                            vfloat32m8_t _outp = op(_p, _b0x, vl);
                            vse32_v_f32m8(outptr, _outp, vl);

                            ptr += vl;
                            outptr += vl;
                            n -= vl;
                        }

                        ptr1 += elempack1;
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
                    vfloat32m8_t _b0x = vle32_v_f32m8_f32m1(ptr1);

                    int n = w * h * elempack;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e32m8(n);
                        vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _outp = op(_p, _b0x, vl);
                        vse32_v_f32m8(outptr, _outp, vl);

                        ptr += vl;
                        outptr += vl;
                        n -= vl;
                    }

                    ptr1 += elempack1;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1 && elempack1 == 1)
            {
                // type 25
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = a.channel(q);
                    const float b0 = b[0];
                    float* outptr = c.channel(q);

                    int n = size * elempack;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e32m8(n);
                        vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _outp = op(_p, b0, vl);
                        vse32_v_f32m8(outptr, _outp, vl);

                        ptr += vl;
                        outptr += vl;
                        n -= vl;
                    }
                }

                return 0;
            }

            // type 26
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* outptr = c.channel(q);

                vfloat32m8_t _b0x = vle32_v_f32m8_f32m1((const float*)b + q * elempack);

                int n = size * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _outp = op(_p, _b0x, vl);
                    vse32_v_f32m8(outptr, _outp, vl);

                    outptr += vl;
                    ptr += vl;
                    n -= vl;
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
                        vfloat32m8_t _a0x = vle32_v_f32m8_f32m1(ptr);

                        int n = w1 * elempack1;
                        while (n > 0)
                        {
                            word_type vl = vsetvl_e32m8(n);
                            vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                            vfloat32m8_t _outp = op(_a0x, _p1, vl);
                            vse32_v_f32m8(outptr, _outp, vl);

                            ptr1 += vl;
                            outptr += vl;
                            n -= vl;
                        }

                        ptr += elempack;
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
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    vfloat32m8_t _b0x = vle32_v_f32m8_f32m1(ptr1);

                    int n = size * elempack;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e32m8(n);
                        vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _outp = op(_p, _b0x, vl);
                        vse32_v_f32m8(outptr, _outp, vl);

                        ptr += vl;
                        outptr += vl;
                        n -= vl;
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
                        int n = elempack;
                        while (n > 0)
                        {
                            word_type vl = vsetvl_e32m8(n);
                            vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                            vfloat32m8_t _outp = op(_p, *ptr1, vl);
                            vse32_v_f32m8(outptr, _outp, vl);

                            n -= vl;
                            ptr += vl;
                            outptr += vl;
                        }

                        ptr1 += 1;
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
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    vfloat32m8_t _a0x = vle32_v_f32m8_f32m1(ptr);

                    int n1 = size1 * elempack1;
                    while (n1 > 0)
                    {
                        word_type vl = vsetvl_e32m8(n1);
                        vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                        vfloat32m8_t _outp = op(_a0x, _p1, vl);
                        vse32_v_f32m8(outptr, _outp, vl);

                        ptr1 += vl;
                        outptr += vl;
                        n1 -= vl;
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
                        int n1 = elempack1;
                        while (n1 > 0)
                        {
                            word_type vl = vsetvl_e32m8(n1);
                            vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                            vfloat32m8_t _p = vfmv_v_f_f32m8(*ptr, vl);
                            vfloat32m8_t _outp = op(_p, _p1, vl);
                            vse32_v_f32m8(outptr, _outp, vl);

                            n1 -= vl;
                            ptr1 += vl;
                            outptr += vl;
                        }

                        ptr += 1;
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
                        vfloat32m8_t _b0x = vle32_v_f32m8_f32m1(ptr1);

                        int n = w * elempack;
                        while (n > 0)
                        {
                            word_type vl = vsetvl_e32m8(n);
                            vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                            vfloat32m8_t _outp = op(_p, _b0x, vl);
                            vse32_v_f32m8(outptr, _outp, vl);

                            ptr += vl;
                            outptr += vl;
                            n -= vl;
                        }

                        ptr1 += elempack1;
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
                        int n = w * elempack;
                        const float* ptr1_vol = ptr1;
                        while (n > 0)
                        {
                            word_type vl = vsetvl_e32m8(n);
                            vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                            vfloat32m8_t _p1 = vle32_v_f32m8(ptr1_vol, vl);
                            vfloat32m8_t _outp = op(_p, _p1, vl);
                            vse32_v_f32m8(outptr, _outp, vl);

                            outptr += vl;
                            ptr += vl;
                            n -= vl;
                            ptr1_vol += vl;
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
                        vfloat32m8_t _a0x = vle32_v_f32m8_f32m1(ptr);

                        int n = w1 * elempack1;
                        while (n > 0)
                        {
                            word_type vl = vsetvl_e32m8(n);
                            vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                            vfloat32m8_t _outp = op(_a0x, _p1, vl);
                            vse32_v_f32m8(outptr, _outp, vl);

                            ptr1 += vl;
                            outptr += vl;
                            n -= vl;
                        }

                        ptr += elempack;
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
                        int n = w1 * elempack1;
                        const float* ptr_vol = ptr;
                        while (n > 0)
                        {
                            word_type vl = vsetvl_e32m8(n);
                            vfloat32m8_t _p = vle32_v_f32m8(ptr_vol, vl);
                            vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                            vfloat32m8_t _outp = op(_p, _p1, vl);
                            vse32_v_f32m8(outptr, _outp, vl);

                            ptr1 += vl;
                            outptr += vl;
                            ptr_vol += vl;
                            n -= vl;
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

                int n = size * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                    vfloat32m8_t _outp = op(_p, _p1, vl);
                    vse32_v_f32m8(outptr, _outp, vl);

                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
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
                    vfloat32m8_t _b0x = vle32_v_f32m8_f32m1(ptr1);

                    int n = w * elempack;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e32m8(n);
                        vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _outp = op(_p, _b0x, vl);
                        vse32_v_f32m8(outptr, _outp, vl);

                        ptr += vl;
                        outptr += vl;
                        n -= vl;
                    }

                    ptr1 += elempack1;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1 && elempack1 == 1)
            {
                // type 16
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = a.channel(q);
                    const float b0 = b[0];
                    float* outptr = c.channel(q);

                    int n = size * elempack;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e32m8(n);
                        vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _outp = op(_p, b0, vl);
                        vse32_v_f32m8(outptr, _outp, vl);

                        ptr += vl;
                        outptr += vl;
                        n -= vl;
                    }
                }

                return 0;
            }

            // type 17
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* outptr = c.channel(q);

                vfloat32m8_t _b0x = vle32_v_f32m8_f32m1((const float*)b + q * elempack);

                int n = size * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _outp = op(_p, _b0x, vl);
                    vse32_v_f32m8(outptr, _outp, vl);

                    outptr += vl;
                    ptr += vl;
                    n -= vl;
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
                    vfloat32m8_t _a0x = vle32_v_f32m8_f32m1(ptr);

                    int n = w1 * h1 * elempack1;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e32m8(n);
                        vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                        vfloat32m8_t _outp = op(_a0x, _p1, vl);
                        vse32_v_f32m8(outptr, _outp, vl);

                        ptr1 += vl;
                        outptr += vl;
                        n -= vl;
                    }

                    ptr += elempack;
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
                    vfloat32m8_t _a0x = vle32_v_f32m8_f32m1(ptr);

                    int n = w1 * elempack1;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e32m8(n);
                        vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                        vfloat32m8_t _outp = op(_a0x, _p1, vl);
                        vse32_v_f32m8(outptr, _outp, vl);

                        ptr1 += vl;
                        outptr += vl;
                        n -= vl;
                    }

                    ptr += elempack;
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

            int n = size * elempack;
            while (n > 0)
            {
                word_type vl = vsetvl_e32m8(n);
                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                vfloat32m8_t _outp = op(_p, _p1, vl);
                vse32_v_f32m8(outptr, _outp, vl);

                ptr += vl;
                ptr1 += vl;
                outptr += vl;
                n -= vl;
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
                const float* ptr = a;
                const float b0 = b[0];
                float* outptr = c;

                int n = size * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _outp = op(_p, b0, vl);
                    vse32_v_f32m8(outptr, _outp, vl);

                    ptr += vl;
                    outptr += vl;
                    n -= vl;
                }

                return 0;
            }

            // type 12
            const float* ptr = a;
            const float* ptr1 = b;
            float* outptr = c;

            for (int y = 0; y < h; y++)
            {
                vfloat32m8_t _b0x = vle32_v_f32m8_f32m1(ptr1);

                int n = w * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _outp = op(_p, _b0x, vl);
                    vse32_v_f32m8(outptr, _outp, vl);

                    ptr += vl;
                    outptr += vl;
                    n -= vl;
                }

                ptr1 += elempack;
            }

            return 0;
        }
    }
    else if (a.dims == 1)
    {
        if (a.w == 1 && elempack == 1)
        {
            if (b.dims == 4)
            {
                // type 20
                c.create(w1, h1, d1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float a0 = a[0];
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    int n1 = size1 * elempack1;
                    while (n1 > 0)
                    {
                        word_type vl = vsetvl_e32m8(n1);
                        vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                        vfloat32m8_t _outp = op(a0, _p1, vl);
                        vse32_v_f32m8(outptr, _outp, vl);

                        ptr1 += vl;
                        outptr += vl;
                        n1 -= vl;
                    }
                }

                return 0;
            }

            if (b.dims == 3)
            {
                // type 4
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float a0 = a[0];
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    int n1 = size1 * elempack1;
                    while (n1 > 0)
                    {
                        word_type vl = vsetvl_e32m8(n1);
                        vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                        vfloat32m8_t _outp = op(a0, _p1, vl);
                        vse32_v_f32m8(outptr, _outp, vl);

                        ptr1 += vl;
                        outptr += vl;
                        n1 -= vl;
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

                const float a0 = a[0];
                const float* ptr1 = b;
                float* outptr = c;

                int n1 = size1 * elempack1;
                while (n1 > 0)
                {
                    word_type vl = vsetvl_e32m8(n1);
                    vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                    vfloat32m8_t _outp = op(a0, _p1, vl);
                    vse32_v_f32m8(outptr, _outp, vl);

                    ptr1 += vl;
                    outptr += vl;
                    n1 -= vl;
                }

                return 0;
            }

            if (b.dims == 1)
            {
                // type 2
                c.create(w1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const float a0 = a[0];
                const float* ptr1 = b;
                float* outptr = c;

                int n1 = w1 * elempack1;
                while (n1 > 0)
                {
                    word_type vl = vsetvl_e32m8(n1);
                    vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                    vfloat32m8_t _outp = op(a0, _p1, vl);
                    vse32_v_f32m8(outptr, _outp, vl);

                    ptr1 += vl;
                    outptr += vl;
                    n1 -= vl;
                }

                return 0;
            }
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
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                vfloat32m8_t _a0x = vle32_v_f32m8_f32m1((const float*)a + q * elempack);

                int n1 = size1 * elempack1;
                while (n1 > 0)
                {
                    word_type vl = vsetvl_e32m8(n1);
                    vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                    vfloat32m8_t _outp = op(_a0x, _p1, vl);
                    vse32_v_f32m8(outptr, _outp, vl);

                    ptr1 += vl;
                    outptr += vl;
                    n1 -= vl;
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
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                vfloat32m8_t _a0x = vle32_v_f32m8_f32m1((const float*)a + q * elempack);

                int n1 = size1 * elempack1;
                while (n1 > 0)
                {
                    word_type vl = vsetvl_e32m8(n1);
                    vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                    vfloat32m8_t _outp = op(_a0x, _p1, vl);
                    vse32_v_f32m8(outptr, _outp, vl);

                    ptr1 += vl;
                    outptr += vl;
                    n1 -= vl;
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
                vfloat32m8_t _a0x = vle32_v_f32m8_f32m1(ptr);

                int n = w1 * elempack1;
                while (n > 0)
                {
                    word_type vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                    vfloat32m8_t _outp = op(_a0x, _p1, vl);
                    vse32_v_f32m8(outptr, _outp, vl);

                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }

                ptr += elempack;
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
                const float* ptr = a;
                const float b0 = b[0];
                float* outptr = c;

                int n = w * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _outp = op(_p, b0, vl);
                    vse32_v_f32m8(outptr, _outp, vl);

                    ptr += vl;
                    outptr += vl;
                    n -= vl;
                }

                return 0;
            }

            // type 7
            const float* ptr = a;
            const float* ptr1 = b;
            float* outptr = c;

            int n = size * elempack;
            while (n > 0)
            {
                word_type vl = vsetvl_e32m8(n);
                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                vfloat32m8_t _outp = op(_p, _p1, vl);
                vse32_v_f32m8(outptr, _outp, vl);

                ptr += vl;
                ptr1 += vl;
                outptr += vl;
                n -= vl;
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_scalar_rvv(Mat& a, float b, const Option& opt)
{
    Op op;
    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int size = w * h * d;
    int elempack = a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);
        int n = size * elempack;
        while (n > 0)
        {
            word_type vl = vsetvl_e32m8(n);
            vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
            _p = op(_p, b, vl);
            vse32_v_f32m8(ptr, _p, vl);

            n -= vl;
            ptr += vl;
        }
    }

    return 0;
}

namespace BinaryOp_riscv_functor {

#define MAKE_FUNCTION(NAME, IMPLVV, IMPLVS, IMPLSV)                                                     \
    struct NAME                                                                                         \
    {                                                                                                   \
        vfloat32m8_t operator()(const vfloat32m8_t& x, const vfloat32m8_t& y, const word_type vl) const \
        {                                                                                               \
            return IMPLVV;                                                                              \
        }                                                                                               \
        vfloat32m8_t operator()(const vfloat32m8_t& x, const float y, const word_type vl) const         \
        {                                                                                               \
            return IMPLVS;                                                                              \
        }                                                                                               \
        vfloat32m8_t operator()(const float x, const vfloat32m8_t& y, const word_type vl) const         \
        {                                                                                               \
            return IMPLSV;                                                                              \
        }                                                                                               \
    };

MAKE_FUNCTION(binary_op_add_rvv, vfadd_vv_f32m8(x, y, vl), vfadd_vf_f32m8(x, y, vl), vfadd_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_sub_rvv, vfsub_vv_f32m8(x, y, vl), vfsub_vf_f32m8(x, y, vl), vfrsub_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_mul_rvv, vfmul_vv_f32m8(x, y, vl), vfmul_vf_f32m8(x, y, vl), vfmul_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_div_rvv, vfdiv_vv_f32m8(x, y, vl), vfdiv_vf_f32m8(x, y, vl), vfrdiv_vf_f32m8(y, x, vl))

MAKE_FUNCTION(binary_op_max_rvv, vfmax_vv_f32m8(x, y, vl), vfmax_vf_f32m8(x, y, vl), vfmax_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_min_rvv, vfmin_vv_f32m8(x, y, vl), vfmin_vf_f32m8(x, y, vl), vfmin_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_pow_rvv, pow_ps(x, y, vl), pow_ps(x, vfmv_v_f_f32m8(y, vl), vl), pow_ps(vfmv_v_f_f32m8(x, vl), y, vl))
MAKE_FUNCTION(binary_op_rsub_rvv, vfsub_vv_f32m8(y, x, vl), vfrsub_vf_f32m8(x, y, vl), vfsub_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_rdiv_rvv, vfdiv_vv_f32m8(y, x, vl), vfrdiv_vf_f32m8(x, y, vl), vfdiv_vf_f32m8(y, x, vl))

#undef MAKE_FUNCTION

} // namespace BinaryOp_riscv_functor
#endif

int BinaryOp_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int elembits = std::max(bottom_blobs[0].elembits(), bottom_blobs[1].elembits());
#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
    {
        return forward_fp16s(bottom_blobs, top_blobs, opt);
    }
#endif
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

#if __riscv_vector
    using namespace BinaryOp_riscv_functor;

    int elempack = bottom_blob.elempack;
    int elempack1 = bottom_blob1.elempack;
    if (elempack != 1 || elempack1 != 1)
    {
        if (op_type == Operation_ADD)
            return binary_op_rvv<binary_op_add_rvv>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op_rvv<binary_op_sub_rvv>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op_rvv<binary_op_mul_rvv>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op_rvv<binary_op_div_rvv>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op_rvv<binary_op_max_rvv>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op_rvv<binary_op_min_rvv>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op_rvv<binary_op_pow_rvv>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op_rvv<binary_op_rsub_rvv>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op_rvv<binary_op_rdiv_rvv>(bottom_blob, bottom_blob1, top_blob, opt);
    }
#endif

    return BinaryOp::forward(bottom_blobs, top_blobs, opt);
}

int BinaryOp_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if __riscv_vector
    int elembits = bottom_top_blob.elembits();

#if __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
    {
        return forward_inplace_fp16s(bottom_top_blob, opt);
    }
#endif

    using namespace BinaryOp_riscv_functor;

    if (op_type == Operation_ADD)
        return binary_op_scalar_rvv<binary_op_add_rvv>(bottom_top_blob, b, opt);

    if (op_type == Operation_SUB)
        return binary_op_scalar_rvv<binary_op_sub_rvv>(bottom_top_blob, b, opt);

    if (op_type == Operation_MUL)
        return binary_op_scalar_rvv<binary_op_mul_rvv>(bottom_top_blob, b, opt);

    if (op_type == Operation_DIV)
        return binary_op_scalar_rvv<binary_op_div_rvv>(bottom_top_blob, b, opt);

    if (op_type == Operation_MAX)
        return binary_op_scalar_rvv<binary_op_max_rvv>(bottom_top_blob, b, opt);

    if (op_type == Operation_MIN)
        return binary_op_scalar_rvv<binary_op_min_rvv>(bottom_top_blob, b, opt);

    if (op_type == Operation_POW)
        return binary_op_scalar_rvv<binary_op_pow_rvv>(bottom_top_blob, b, opt);

    if (op_type == Operation_RSUB)
        return binary_op_scalar_rvv<binary_op_rsub_rvv>(bottom_top_blob, b, opt);

    if (op_type == Operation_RDIV)
        return binary_op_scalar_rvv<binary_op_rdiv_rvv>(bottom_top_blob, b, opt);

#endif
    return BinaryOp::forward_inplace(bottom_top_blob, opt);
}

// fp16sa
#if __riscv_vector && __riscv_zfh
template<typename Op>
static int binary_op_rvv_fp16s(const Mat& a, const Mat& b, Mat& c, const Option& opt)
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
            c.create(w, h, d, channels, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                int n = size * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                    vfloat16m8_t _outp = op(_p, _p1, vl);
                    vse16_v_f16m8(outptr, _outp, vl);

                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
            }

            return 0;
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
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int z = 0; z < d; z++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        vfloat16m8_t _b0x = vle16_v_f16m8_f16m1(ptr1);

                        int n = w * elempack;
                        while (n > 0)
                        {
                            word_type vl = vsetvl_e16m8(n);
                            vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                            vfloat16m8_t _outp = op(_p, _b0x, vl);
                            vse16_v_f16m8(outptr, _outp, vl);

                            ptr += vl;
                            outptr += vl;
                            n -= vl;
                        }

                        ptr1 += elempack1;
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
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.row<const __fp16>(q);
                __fp16* outptr = c.channel(q);

                for (int z = 0; z < d; z++)
                {
                    vfloat16m8_t _b0x = vle16_v_f16m8_f16m1(ptr1);

                    int n = w * h * elempack;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e16m8(n);
                        vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                        vfloat16m8_t _outp = op(_p, _b0x, vl);
                        vse16_v_f16m8(outptr, _outp, vl);

                        ptr += vl;
                        outptr += vl;
                        n -= vl;
                    }

                    ptr1 += elempack1;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1 && elempack1 == 1)
            {
                // type 25
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    const __fp16 b0 = ((const __fp16*)b)[0];
                    __fp16* outptr = c.channel(q);

                    int n = size * elempack;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e16m8(n);
                        vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                        vfloat16m8_t _outp = op(_p, b0, vl);
                        vse16_v_f16m8(outptr, _outp, vl);

                        ptr += vl;
                        outptr += vl;
                        n -= vl;
                    }
                }

                return 0;
            }

            // type 26
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                __fp16* outptr = c.channel(q);

                vfloat16m8_t _b0x = vle16_v_f16m8_f16m1((const __fp16*)b + q * elempack);

                int n = size * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _outp = op(_p, _b0x, vl);
                    vse16_v_f16m8(outptr, _outp, vl);

                    outptr += vl;
                    ptr += vl;
                    n -= vl;
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
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int z = 0; z < d1; z++)
                {
                    for (int y = 0; y < h1; y++)
                    {
                        vfloat16m8_t _a0x = vle16_v_f16m8_f16m1(ptr);

                        int n = w1 * elempack1;
                        while (n > 0)
                        {
                            word_type vl = vsetvl_e16m8(n);
                            vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                            vfloat16m8_t _outp = op(_a0x, _p1, vl);
                            vse16_v_f16m8(outptr, _outp, vl);

                            ptr1 += vl;
                            outptr += vl;
                            n -= vl;
                        }

                        ptr += elempack;
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
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    vfloat16m8_t _b0x = vle16_v_f16m8_f16m1(ptr1);

                    int n = size * elempack;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e16m8(n);
                        vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                        vfloat16m8_t _outp = op(_p, _b0x, vl);
                        vse16_v_f16m8(outptr, _outp, vl);

                        ptr += vl;
                        outptr += vl;
                        n -= vl;
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
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b;
                    __fp16* outptr = c.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        int n = elempack;
                        while (n > 0)
                        {
                            word_type vl = vsetvl_e16m8(n);
                            vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                            vfloat16m8_t _outp = op(_p, *ptr1, vl);

                            vse16_v_f16m8(outptr, _outp, vl);
                            n -= vl;
                            ptr += vl;
                            outptr += vl;
                        }
                        ptr1 += 1;
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
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    vfloat16m8_t _a0x = vle16_v_f16m8_f16m1(ptr);

                    int n1 = size1 * elempack1;
                    while (n1 > 0)
                    {
                        word_type vl = vsetvl_e16m8(n1);
                        vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                        vfloat16m8_t _outp = op(_a0x, _p1, vl);
                        vse16_v_f16m8(outptr, _outp, vl);

                        ptr1 += vl;
                        outptr += vl;
                        n1 -= vl;
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
                    const __fp16* ptr = a;
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int i = 0; i < size1; i++)
                    {
                        int n1 = elempack1;
                        while (n1 > 0)
                        {
                            word_type vl = vsetvl_e16m8(n1);
                            vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                            vfloat16m8_t _p = vfmv_v_f_f16m8(*ptr, vl);
                            vfloat16m8_t _outp = op(_p, _p1, vl);
                            vse16_v_f16m8(outptr, _outp, vl);

                            n1 -= vl;
                            ptr1 += vl;
                            outptr += vl;
                        }
                        ptr += 1;
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
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int y = 0; y < h; y++)
                    {
                        vfloat16m8_t _b0x = vle16_v_f16m8_f16m1(ptr1);

                        int n = w * elempack;
                        while (n > 0)
                        {
                            word_type vl = vsetvl_e16m8(n);
                            vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                            vfloat16m8_t _outp = op(_p, _b0x, vl);
                            vse16_v_f16m8(outptr, _outp, vl);

                            ptr += vl;
                            outptr += vl;
                            n -= vl;
                        }

                        ptr1 += elempack1;
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
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            int n = elempack;
                            const __fp16* ptr1_vol = ptr1 + x * elempack;
                            while (n > 0)
                            {
                                word_type vl = vsetvl_e16m8(n);
                                vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                                vfloat16m8_t _p1 = vle16_v_f16m8(ptr1_vol, vl);
                                vfloat16m8_t _outp = op(_p, _p1, vl);
                                vse16_v_f16m8(outptr, _outp, vl);

                                outptr += vl;
                                ptr += vl;
                                n -= vl;
                                ptr1_vol += vl;
                            }
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
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int y = 0; y < h1; y++)
                    {
                        vfloat16m8_t _a0x = vle16_v_f16m8_f16m1(ptr);

                        int n = w1 * elempack1;
                        while (n > 0)
                        {
                            word_type vl = vsetvl_e16m8(n);
                            vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                            vfloat16m8_t _outp = op(_a0x, _p1, vl);
                            vse16_v_f16m8(outptr, _outp, vl);

                            ptr1 += vl;
                            outptr += vl;
                            n -= vl;
                        }

                        ptr += elempack;
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
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int y = 0; y < h1; y++)
                    {
                        for (int x = 0; x < w1; x++)
                        {
                            int n = elempack;
                            const __fp16* ptr_vol = ptr + x * elempack;
                            while (n > 0)
                            {
                                word_type vl = vsetvl_e16m8(n);
                                vfloat16m8_t _p = vle16_v_f16m8(ptr_vol, vl);
                                vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                                vfloat16m8_t _outp = op(_p, _p1, vl);
                                vse16_v_f16m8(outptr, _outp, vl);

                                ptr1 += vl;
                                outptr += vl;
                                ptr_vol += vl;
                                n -= vl;
                            }
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
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                int n = size * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                    vfloat16m8_t _outp = op(_p, _p1, vl);
                    vse16_v_f16m8(outptr, _outp, vl);

                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
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
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.row<const __fp16>(q);
                __fp16* outptr = c.channel(q);

                for (int y = 0; y < h; y++)
                {
                    vfloat16m8_t _b0x = vle16_v_f16m8_f16m1(ptr1);

                    int n = w * elempack;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e16m8(n);
                        vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                        vfloat16m8_t _outp = op(_p, _b0x, vl);
                        vse16_v_f16m8(outptr, _outp, vl);

                        ptr += vl;
                        outptr += vl;
                        n -= vl;
                    }

                    ptr1 += elempack1;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1 && elempack1 == 1)
            {
                // type 16
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    const __fp16 b0 = *(const __fp16*)b;
                    __fp16* outptr = c.channel(q);

                    int n = size * elempack;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e16m8(n);
                        vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                        vfloat16m8_t _outp = op(_p, b0, vl);
                        vse16_v_f16m8(outptr, _outp, vl);

                        ptr += vl;
                        outptr += vl;
                        n -= vl;
                    }
                }

                return 0;
            }

            // type 17
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                __fp16* outptr = c.channel(q);

                vfloat16m8_t _b0x = vle16_v_f16m8_f16m1((const __fp16*)b + q * elempack);

                int n = size * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _outp = op(_p, _b0x, vl);
                    vse16_v_f16m8(outptr, _outp, vl);

                    outptr += vl;
                    ptr += vl;
                    n -= vl;
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
                const __fp16* ptr = a.row<const __fp16>(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int z = 0; z < d1; z++)
                {
                    vfloat16m8_t _a0x = vle16_v_f16m8_f16m1(ptr);

                    int n = w1 * h1 * elempack1;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e16m8(n);
                        vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                        vfloat16m8_t _outp = op(_a0x, _p1, vl);
                        vse16_v_f16m8(outptr, _outp, vl);

                        ptr1 += vl;
                        outptr += vl;
                        n -= vl;
                    }

                    ptr += elempack;
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
                const __fp16* ptr = a.row<const __fp16>(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int y = 0; y < h1; y++)
                {
                    vfloat16m8_t _a0x = vle16_v_f16m8_f16m1(ptr);

                    int n = w1 * elempack1;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e16m8(n);
                        vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                        vfloat16m8_t _outp = op(_a0x, _p1, vl);
                        vse16_v_f16m8(outptr, _outp, vl);

                        ptr1 += vl;
                        outptr += vl;
                        n -= vl;
                    }

                    ptr += elempack;
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
            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;

            int n = size * elempack;
            while (n > 0)
            {
                word_type vl = vsetvl_e16m8(n);
                vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                vfloat16m8_t _outp = op(_p, _p1, vl);
                vse16_v_f16m8(outptr, _outp, vl);

                ptr += vl;
                ptr1 += vl;
                outptr += vl;
                n -= vl;
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
                const __fp16* ptr = a;
                const __fp16 b0 = ((const __fp16*)b)[0];
                __fp16* outptr = c;

                int n = size * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _outp = op(_p, b0, vl);
                    vse16_v_f16m8(outptr, _outp, vl);

                    ptr += vl;
                    outptr += vl;
                    n -= vl;
                }

                return 0;
            }

            // type 12
            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;

            for (int y = 0; y < h; y++)
            {
                vfloat16m8_t _b0x = vle16_v_f16m8_f16m1(ptr1);

                int n = w * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _outp = op(_p, _b0x, vl);
                    vse16_v_f16m8(outptr, _outp, vl);

                    ptr += vl;
                    outptr += vl;
                    n -= vl;
                }

                ptr1 += elempack;
            }

            return 0;
        }
    }
    else if (a.dims == 1)
    {
        if (a.w == 1 && elempack == 1)
        {
            if (b.dims == 4)
            {
                // type 20
                c.create(w1, h1, d1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16 a0 = ((const __fp16*)a)[0];
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    int n1 = size1 * elempack1;
                    while (n1 > 0)
                    {
                        word_type vl = vsetvl_e16m8(n1);
                        vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                        vfloat16m8_t _outp = op(a0, _p1, vl);
                        vse16_v_f16m8(outptr, _outp, vl);

                        ptr1 += vl;
                        outptr += vl;
                        n1 -= vl;
                    }
                }

                return 0;
            }

            if (b.dims == 3)
            {
                // type 4
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16 a0 = ((const __fp16*)a)[0];
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    int n1 = size1 * elempack1;
                    while (n1 > 0)
                    {
                        word_type vl = vsetvl_e16m8(n1);
                        vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                        vfloat16m8_t _outp = op(a0, _p1, vl);
                        vse16_v_f16m8(outptr, _outp, vl);

                        ptr1 += vl;
                        outptr += vl;
                        n1 -= vl;
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

                const __fp16 a0 = ((const __fp16*)a)[0];
                const __fp16* ptr1 = b;
                __fp16* outptr = c;

                int n1 = size1 * elempack1;
                while (n1 > 0)
                {
                    word_type vl = vsetvl_e16m8(n1);
                    vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                    vfloat16m8_t _outp = op(a0, _p1, vl);
                    vse16_v_f16m8(outptr, _outp, vl);

                    ptr1 += vl;
                    outptr += vl;
                    n1 -= vl;
                }

                return 0;
            }

            if (b.dims == 1)
            {
                // type 2

                c.create(w1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const __fp16 a0 = ((const __fp16*)a)[0];
                const __fp16* ptr1 = b;
                __fp16* outptr = c;

                int n1 = w1 * elempack1;
                while (n1 > 0)
                {
                    word_type vl = vsetvl_e16m8(n1);
                    vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                    vfloat16m8_t _outp = op(a0, _p1, vl);
                    vse16_v_f16m8(outptr, _outp, vl);

                    ptr1 += vl;
                    outptr += vl;
                    n1 -= vl;
                }

                return 0;
            }
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
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                vfloat16m8_t _a0x = vle16_v_f16m8_f16m1((const __fp16*)a + q * elempack);

                int n1 = size1 * elempack1;
                while (n1 > 0)
                {
                    word_type vl = vsetvl_e16m8(n1);
                    vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                    vfloat16m8_t _outp = op(_a0x, _p1, vl);
                    vse16_v_f16m8(outptr, _outp, vl);

                    ptr1 += vl;
                    outptr += vl;
                    n1 -= vl;
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
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                vfloat16m8_t _a0x = vle16_v_f16m8_f16m1((const __fp16*)a + q * elempack);

                int n1 = size1 * elempack1;
                while (n1 > 0)
                {
                    word_type vl = vsetvl_e16m8(n1);
                    vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                    vfloat16m8_t _outp = op(_a0x, _p1, vl);
                    vse16_v_f16m8(outptr, _outp, vl);

                    ptr1 += vl;
                    outptr += vl;
                    n1 -= vl;
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

            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;

            for (int y = 0; y < h1; y++)
            {
                vfloat16m8_t _a0x = vle16_v_f16m8_f16m1(ptr);

                int n = w1 * elempack1;
                while (n > 0)
                {
                    word_type vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                    vfloat16m8_t _outp = op(_a0x, _p1, vl);
                    vse16_v_f16m8(outptr, _outp, vl);

                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }

                ptr += elempack;
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
                const __fp16* ptr = a;
                const __fp16 b0 = ((const __fp16*)b)[0];
                __fp16* outptr = c;

                int n = w * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _outp = op(_p, b0, vl);
                    vse16_v_f16m8(outptr, _outp, vl);

                    ptr += vl;
                    outptr += vl;
                    n -= vl;
                }

                return 0;
            }

            // type 7
            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;

            int n = size * elempack;
            while (n > 0)
            {
                word_type vl = vsetvl_e16m8(n);
                vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                vfloat16m8_t _outp = op(_p, _p1, vl);
                vse16_v_f16m8(outptr, _outp, vl);

                ptr += vl;
                ptr1 += vl;
                outptr += vl;
                n -= vl;
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_scalar_rvv_fp16s(Mat& a, float b, const Option& opt)
{
    Op op;
    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int size = w * h * d;
    int elempack = a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = a.channel(q);
        int n = size * elempack;
        while (n > 0)
        {
            word_type vl = vsetvl_e16m8(n);
            vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
            _p = op(_p, b, vl);
            vse16_v_f16m8(ptr, _p, vl);

            n -= vl;
            ptr += vl;
        }
    }

    return 0;
}

namespace BinaryOp_riscv_functor {

#define MAKE_FUNCTION(NAME, IMPLVV, IMPLVS, IMPLSV)                                                     \
    struct NAME                                                                                         \
    {                                                                                                   \
        vfloat16m8_t operator()(const vfloat16m8_t& x, const vfloat16m8_t& y, const word_type vl) const \
        {                                                                                               \
            return IMPLVV;                                                                              \
        }                                                                                               \
        vfloat16m8_t operator()(const vfloat16m8_t& x, const float y, const word_type vl) const         \
        {                                                                                               \
            return IMPLVS;                                                                              \
        }                                                                                               \
        vfloat16m8_t operator()(const float x, const vfloat16m8_t& y, const word_type vl) const         \
        {                                                                                               \
            return IMPLSV;                                                                              \
        }                                                                                               \
    };

MAKE_FUNCTION(binary_op_add_rvv_fp16, vfadd_vv_f16m8(x, y, vl), vfadd_vf_f16m8(x, y, vl), vfadd_vf_f16m8(y, x, vl))
MAKE_FUNCTION(binary_op_sub_rvv_fp16, vfsub_vv_f16m8(x, y, vl), vfsub_vf_f16m8(x, y, vl), vfrsub_vf_f16m8(y, x, vl))
MAKE_FUNCTION(binary_op_mul_rvv_fp16, vfmul_vv_f16m8(x, y, vl), vfmul_vf_f16m8(x, y, vl), vfmul_vf_f16m8(y, x, vl))
MAKE_FUNCTION(binary_op_div_rvv_fp16, vfdiv_vv_f16m8(x, y, vl), vfdiv_vf_f16m8(x, y, vl), vfrdiv_vf_f16m8(y, x, vl))
MAKE_FUNCTION(binary_op_max_rvv_fp16, vfmax_vv_f16m8(x, y, vl), vfmax_vf_f16m8(x, y, vl), vfmax_vf_f16m8(y, x, vl))
MAKE_FUNCTION(binary_op_min_rvv_fp16, vfmin_vv_f16m8(x, y, vl), vfmin_vf_f16m8(x, y, vl), vfmin_vf_f16m8(y, x, vl))
MAKE_FUNCTION(binary_op_pow_rvv_fp16, pow_ps(x, y, vl), pow_ps(x, vfmv_v_f_f16m8(y, vl), vl), pow_ps(vfmv_v_f_f16m8(x, vl), y, vl))
MAKE_FUNCTION(binary_op_rsub_rvv_fp16, vfsub_vv_f16m8(y, x, vl), vfrsub_vf_f16m8(x, y, vl), vfsub_vf_f16m8(y, x, vl))
MAKE_FUNCTION(binary_op_rdiv_rvv_fp16, vfdiv_vv_f16m8(y, x, vl), vfrdiv_vf_f16m8(x, y, vl), vfdiv_vf_f16m8(y, x, vl))

#undef MAKE_FUNCTION

} // namespace BinaryOp_riscv_functor

template<typename Op>
static int binary_op_fp16s(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int size = w * h * d;
    size_t elemsize = a.elemsize;

    int w1 = b.w;
    int h1 = b.h;
    int d1 = b.d;
    int channels1 = b.c;
    int size1 = w1 * h1 * d1;

    if (a.dims == 4)
    {
        if (b.dims == 4)
        {
            // type 29
            c.create(w, h, d, channels, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = op(ptr[i], ptr1[i]);
                }
            }

            return 0;
        }

        c.create(w, h, d, channels, elemsize, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 3)
        {
            // type 28
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int z = 0; z < d; z++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const __fp16 b0 = ptr1[y];
                        for (int x = 0; x < w; x++)
                        {
                            outptr[x] = op(ptr[x], b0);
                        }

                        ptr += w;
                        outptr += w;
                    }

                    ptr1 += h;
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
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.row<const __fp16>(q);
                __fp16* outptr = c.channel(q);

                for (int z = 0; z < d; z++)
                {
                    const __fp16 b0 = ptr1[z];
                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            outptr[x] = op(ptr[x], b0);
                        }

                        ptr += w;
                        outptr += w;
                    }
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1)
            {
                // type 25
                const __fp16 b0 = ((const __fp16*)b)[0];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] = op(ptr[i], b0);
                    }
                }

                return 0;
            }

            // type 26
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                const __fp16 b0 = ((const __fp16*)b)[q];
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = op(ptr[i], b0);
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
            c.create(w1, h1, d1, channels1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int z = 0; z < d1; z++)
                {
                    for (int y = 0; y < h1; y++)
                    {
                        const __fp16 a0 = ptr[y];
                        for (int x = 0; x < w1; x++)
                        {
                            outptr[x] = op(a0, ptr1[x]);
                        }

                        ptr1 += w1;
                        outptr += w1;
                    }

                    ptr += h1;
                }
            }

            return 0;
        }

        if (b.dims == 3)
        {
            if (w1 == 1 && h1 == 1 && channels1 == channels)
            {
                // special type 1
                c.create(w, h, channels, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    const __fp16* b0 = b.channel(q);
                    __fp16* outptr = c.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] = op(ptr[i], b0[0]);
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels1 == 1)
            {
                // special type 2
                c.create(w, h, channels, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b;
                    __fp16* outptr = c.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] = op(ptr[i], ptr1[i]);
                    }
                }

                return 0;
            }

            if (w == 1 && h == 1 && channels1 == channels)
            {
                // special type 3
                c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16* a0 = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        outptr[i] = op(a0[0], ptr1[i]);
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels == 1)
            {
                // special type 4
                c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16* ptr = a;
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        outptr[i] = op(ptr[i], ptr1[i]);
                    }
                }

                return 0;
            }

            if (w != 1 && w1 == 1 && h1 == h && channels1 == channels)
            {
                // special type 5
                c.create(w, h, channels, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int y = 0; y < h; y++)
                    {
                        const __fp16 b0 = ptr1[y];
                        for (int x = 0; x < w; x++)
                        {
                            outptr[x] = op(ptr[x], b0);
                        }

                        ptr += w;
                        outptr += w;
                    }
                }

                return 0;
            }

            if (w1 == w && h != 1 && h1 == 1 && channels1 == channels)
            {
                // special type 6
                c.create(w, h, channels, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            outptr[x] = op(ptr[x], ptr1[x]);
                        }

                        ptr += w;
                        outptr += w;
                    }
                }

                return 0;
            }

            if (w1 != 1 && w == 1 && h1 == h && channels1 == channels)
            {
                // special type 7
                c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int y = 0; y < h1; y++)
                    {
                        const __fp16 a0 = ptr[y];
                        for (int x = 0; x < w1; x++)
                        {
                            outptr[x] = op(a0, ptr1[x]);
                        }

                        ptr1 += w1;
                        outptr += w1;
                    }
                }

                return 0;
            }

            if (w1 == w && h1 != 1 && h == 1 && channels1 == channels)
            {
                // special type 8
                c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int y = 0; y < h1; y++)
                    {
                        for (int x = 0; x < w1; x++)
                        {
                            outptr[x] = op(ptr[x], ptr1[x]);
                        }

                        ptr1 += w1;
                        outptr += w1;
                    }
                }

                return 0;
            }

            // type 19
            c.create(w, h, channels, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = op(ptr[i], ptr1[i]);
                }
            }

            return 0;
        }

        c.create(w, h, channels, elemsize, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 18
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.row<const __fp16>(q);
                __fp16* outptr = c.channel(q);

                for (int y = 0; y < h; y++)
                {
                    const __fp16 b0 = ptr1[y];
                    for (int x = 0; x < w; x++)
                    {
                        outptr[x] = op(ptr[x], b0);
                    }

                    ptr += w;
                    outptr += w;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1)
            {
                // type 16
                const __fp16 b0 = ((const __fp16*)b)[0];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] = op(ptr[i], b0);
                    }
                }

                return 0;
            }

            // type 17
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                const __fp16 b0 = ((const __fp16*)b)[q];
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = op(ptr[i], b0);
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
            c.create(w1, h1, d1, channels1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const __fp16* ptr = a.row<const __fp16>(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int z = 0; z < d1; z++)
                {
                    const __fp16 a0 = ptr[z];
                    for (int y = 0; y < h1; y++)
                    {
                        for (int x = 0; x < w1; x++)
                        {
                            outptr[x] = op(a0, ptr1[x]);
                        }

                        ptr1 += w1;
                        outptr += w1;
                    }
                }
            }

            return 0;
        }

        if (b.dims == 3)
        {
            // type 14
            c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const __fp16* ptr = a.row<const __fp16>(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int y = 0; y < h1; y++)
                {
                    const __fp16 a0 = ptr[y];
                    for (int x = 0; x < w1; x++)
                    {
                        outptr[x] = op(a0, ptr1[x]);
                    }

                    ptr1 += w1;
                    outptr += w1;
                }
            }

            return 0;
        }

        c.create(w, h, elemsize, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 13
            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;
            for (int i = 0; i < size; i++)
            {
                outptr[i] = op(ptr[i], ptr1[i]);
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, h, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1)
            {
                // type 11
                const __fp16 b0 = ((const __fp16*)b)[0];
                const __fp16* ptr = a;
                __fp16* outptr = c;
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = op(ptr[i], b0);
                }

                return 0;
            }

            // type 12
            const __fp16* ptr = a;
            __fp16* outptr = c;

            for (int y = 0; y < h; y++)
            {
                const __fp16 b0 = ((const __fp16*)b)[y];
                for (int x = 0; x < w; x++)
                {
                    outptr[x] = op(ptr[x], b0);
                }

                ptr += w;
                outptr += w;
            }

            return 0;
        }
    }
    else if (a.dims == 1)
    {
        if (a.w == 1)
        {
            if (b.dims == 4)
            {
                // type 20
                c.create(w1, h1, d1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const __fp16 a0 = ((const __fp16*)a)[0];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int i = 0; i < size1; i++)
                    {
                        outptr[i] = op(a0, ptr1[i]);
                    }
                }

                return 0;
            }

            if (b.dims == 3)
            {
                // type 4
                c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const __fp16 a0 = ((const __fp16*)a)[0];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int i = 0; i < size1; i++)
                    {
                        outptr[i] = op(a0, ptr1[i]);
                    }
                }

                return 0;
            }

            if (b.dims == 2)
            {
                // type 3
                c.create(w1, h1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const __fp16 a0 = ((const __fp16*)a)[0];
                const __fp16* ptr1 = b;
                __fp16* outptr = c;
                for (int i = 0; i < size1; i++)
                {
                    outptr[i] = op(a0, ptr1[i]);
                }

                return 0;
            }

            if (b.dims == 1)
            {
                // type 2
                c.create(w1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const __fp16 a0 = ((const __fp16*)a)[0];
                const __fp16* ptr1 = b;
                __fp16* outptr = c;
                for (int i = 0; i < w1; i++)
                {
                    outptr[i] = op(a0, ptr1[i]);
                }

                return 0;
            }
        }

        if (b.dims == 4)
        {
            // type 21
            c.create(w1, h1, d1, channels1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const __fp16 a0 = ((const __fp16*)a)[q];
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    outptr[i] = op(a0, ptr1[i]);
                }
            }

            return 0;
        }

        if (b.dims == 3)
        {
            // type 9
            c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const __fp16 a0 = ((const __fp16*)a)[q];
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    outptr[i] = op(a0, ptr1[i]);
                }
            }

            return 0;
        }

        if (b.dims == 2)
        {
            // type 8
            c.create(w1, h1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            const __fp16* ptr1 = b;
            __fp16* outptr = c;

            for (int y = 0; y < h1; y++)
            {
                const __fp16 a0 = ((const __fp16*)a)[y];
                for (int x = 0; x < w1; x++)
                {
                    outptr[x] = op(a0, ptr1[x]);
                }

                ptr1 += w1;
                outptr += w1;
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1)
            {
                // type 6
                const __fp16 b0 = ((const __fp16*)b)[0];
                const __fp16* ptr = a;
                __fp16* outptr = c;
                for (int i = 0; i < w; i++)
                {
                    outptr[i] = op(ptr[i], b0);
                }

                return 0;
            }

            // type 7
            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;
            for (int i = 0; i < w; i++)
            {
                outptr[i] = op(ptr[i], ptr1[i]);
            }
        }
    }

    return 0;
}

namespace BinaryOp_riscv_functor {

#define MAKE_FUNCTION(NAME, IMPL)                                 \
    struct NAME                                                   \
    {                                                             \
        __fp16 operator()(const __fp16& x, const __fp16& y) const \
        {                                                         \
            return IMPL;                                          \
        }                                                         \
    };

MAKE_FUNCTION(binary_op_add_fp16s, x + y)
MAKE_FUNCTION(binary_op_sub_fp16s, x - y)
// clang-format off
// *INDENT-OFF*
MAKE_FUNCTION(binary_op_mul_fp16s, x * y)
// *INDENT-ON*
// clang-format on
MAKE_FUNCTION(binary_op_div_fp16s, x / y)
MAKE_FUNCTION(binary_op_max_fp16s, std::max(x, y))
MAKE_FUNCTION(binary_op_min_fp16s, std::min(x, y))
MAKE_FUNCTION(binary_op_pow_fp16s, (__fp16)pow((float)x, (float)y))
MAKE_FUNCTION(binary_op_rsub_fp16s, y - x)
MAKE_FUNCTION(binary_op_rdiv_fp16s, y / x)

#undef MAKE_FUNCTION

} // namespace BinaryOp_riscv_functor

int BinaryOp_riscv::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    using namespace BinaryOp_riscv_functor;

    int elempack = bottom_blob.elempack;
    int elempack1 = bottom_blob1.elempack;
    if (elempack != 1 || elempack1 != 1)
    {
        if (op_type == Operation_ADD)
            return binary_op_rvv_fp16s<binary_op_add_rvv_fp16>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op_rvv_fp16s<binary_op_sub_rvv_fp16>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op_rvv_fp16s<binary_op_mul_rvv_fp16>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op_rvv_fp16s<binary_op_div_rvv_fp16>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op_rvv_fp16s<binary_op_max_rvv_fp16>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op_rvv_fp16s<binary_op_min_rvv_fp16>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op_rvv_fp16s<binary_op_pow_rvv_fp16>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op_rvv_fp16s<binary_op_rsub_rvv_fp16>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op_rvv_fp16s<binary_op_rdiv_rvv_fp16>(bottom_blob, bottom_blob1, top_blob, opt);
    }

    if (elempack == 1 && elempack1 == 1)
    {
        if (op_type == Operation_ADD)
            return binary_op_fp16s<binary_op_add_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op_fp16s<binary_op_sub_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op_fp16s<binary_op_mul_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op_fp16s<binary_op_div_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op_fp16s<binary_op_max_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op_fp16s<binary_op_min_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op_fp16s<binary_op_pow_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op_fp16s<binary_op_sub_fp16s>(bottom_blob1, bottom_blob, top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op_fp16s<binary_op_div_fp16s>(bottom_blob1, bottom_blob, top_blob, opt);
    }

    return 0;
}

int BinaryOp_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    using namespace BinaryOp_riscv_functor;

    if (op_type == Operation_ADD)
        return binary_op_scalar_rvv_fp16s<binary_op_add_rvv_fp16>(bottom_top_blob, b, opt);

    if (op_type == Operation_SUB)
        return binary_op_scalar_rvv_fp16s<binary_op_sub_rvv_fp16>(bottom_top_blob, b, opt);

    if (op_type == Operation_MUL)
        return binary_op_scalar_rvv_fp16s<binary_op_mul_rvv_fp16>(bottom_top_blob, b, opt);

    if (op_type == Operation_DIV)
        return binary_op_scalar_rvv_fp16s<binary_op_div_rvv_fp16>(bottom_top_blob, b, opt);

    if (op_type == Operation_MAX)
        return binary_op_scalar_rvv_fp16s<binary_op_max_rvv_fp16>(bottom_top_blob, b, opt);

    if (op_type == Operation_MIN)
        return binary_op_scalar_rvv_fp16s<binary_op_min_rvv_fp16>(bottom_top_blob, b, opt);

    if (op_type == Operation_POW)
        return binary_op_scalar_rvv_fp16s<binary_op_pow_rvv_fp16>(bottom_top_blob, b, opt);

    if (op_type == Operation_RSUB)
        return binary_op_scalar_rvv_fp16s<binary_op_rsub_rvv_fp16>(bottom_top_blob, b, opt);

    if (op_type == Operation_RDIV)
        return binary_op_scalar_rvv_fp16s<binary_op_rdiv_rvv_fp16>(bottom_top_blob, b, opt);

    return 0;
}
#endif

} // namespace ncnn

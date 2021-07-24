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
static int binary_op_rvv(const Mat& a, const Mat& b, Mat& c,
                         const Option& opt)
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

                    int n = size * elempack;
                    while (n > 0)
                    {
                        const float* b_vol = b0;
                        int n1 = size1 * elempack1;
                        while (n1 > 0)
                        {
                            word_type vl = vsetvl_e32m8(std::min(n1, n));

                            vfloat32m8_t _b = vle32_v_f32m8(b_vol, vl);
                            vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                            vfloat32m8_t _outp = op(_p, _b, vl);
                            vse32_v_f32m8(outptr, _outp, vl);

                            ptr += vl;
                            b_vol += vl;
                            outptr += vl;

                            n1 -= vl;
                            n -= vl;
                        }
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
                    const float* a0 = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    int n1 = size1 * elempack1;
                    while (n1 > 0)
                    {
                        const float* a_vol = a0;
                        int n = size * elempack;
                        while (n > 0)
                        {
                            word_type vl = vsetvl_e32m8(std::min(n1, n));

                            vfloat32m8_t _a0 = vle32_v_f32m8(a_vol, vl);
                            vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                            vfloat32m8_t _outp = op(_a0, _p1, vl);
                            vse32_v_f32m8(outptr, _outp, vl);

                            ptr1 += vl;
                            a_vol += vl;
                            outptr += vl;

                            n1 -= vl;
                            n -= vl;
                        }
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
                        for (int x = 0; x < w; x++)
                        {
                            const float* ptr1_vol = ptr1 + y * elempack;
                            int n = elempack;
                            while (n > 0)
                            {
                                word_type vl = vsetvl_e32m8(n);
                                vfloat32m8_t _p1 = vle32_v_f32m8(ptr1_vol, vl);
                                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                                vfloat32m8_t _outp = op(_p, _p1, vl);
                                vse32_v_f32m8(outptr, _outp, vl);
                                ptr += vl;
                                outptr += vl;
                                n -= vl;
                            }
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
                            int n = elempack;
                            const float* ptr1_vol = ptr1 + x * elempack;
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
                        for (int x = 0; x < w1; x++)
                        {
                            int n = elempack;
                            const float* ptr_vol = ptr + y * elempack;
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
                            int n = elempack;
                            const float* ptr_vol = ptr + x * elempack;
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
                    for (int x = 0; x < w; x++)
                    {
                        const float* ptr1_vol = ptr1;
                        int n = elempack1;
                        while (n > 0)
                        {
                            word_type vl = vsetvl_e32m8(n);
                            vfloat32m8_t _b0 = vle32_v_f32m8(ptr1_vol, vl);
                            vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                            vfloat32m8_t _outp = op(_p, _b0, vl);
                            vse32_v_f32m8(outptr, _outp, vl);

                            ptr += vl;
                            outptr += vl;
                            ptr1_vol += vl;
                            n -= vl;
                        }
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
                    float* outptr = c.channel(q);

                    int n = size * elempack;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e32m8(n);
                        vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _outp = op(_p, b[0], vl);
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

                int n = size * elempack;

                while (n > 0)
                {
                    int n1 = elempack1;
                    const float* ptr1_vol = (const float*)b + q * elempack1;
                    while (n1 > 0)
                    {
                        word_type vl = vsetvl_e32m8(n1);

                        vfloat32m8_t _b0 = vle32_v_f32m8(ptr1_vol, vl);
                        vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _outp = op(_p, _b0, vl);
                        vse32_v_f32m8(outptr, _outp, vl);

                        ptr1_vol += vl;
                        outptr += vl;
                        ptr += vl;
                        n1 -= vl;
                        n -= vl;
                    }
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
                    for (int x = 0; x < w1; x++)
                    {
                        const float* ptr_vol = ptr;
                        int n = elempack1;
                        while (n > 0)
                        {
                            word_type vl = vsetvl_e32m8(n);
                            vfloat32m8_t _a0 = vle32_v_f32m8(ptr_vol, vl);
                            vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                            vfloat32m8_t _outp = op(_a0, _p1, vl);
                            vse32_v_f32m8(outptr, _outp, vl);
                            ptr1 += vl;
                            outptr += vl;
                            ptr_vol += vl;
                            n -= vl;
                        }
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
                float* outptr = c;
                int n = size * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _outp = op(_p, b[0], vl);
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
                for (int x = 0; x < w; x++)
                {
                    int n = elempack;
                    const float* ptr1_vol = ptr1;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e32m8(n);
                        vfloat32m8_t _b0 = vle32_v_f32m8(ptr1_vol, vl);
                        vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _outp = op(_p, _b0, vl);
                        vse32_v_f32m8(outptr, _outp, vl);

                        ptr += vl;
                        ptr1_vol += vl;
                        outptr += vl;
                        n -= vl;
                    }
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
            if (b.dims == 3)
            {
                // type 4
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    int n1 = size1 * elempack1;
                    while (n1 > 0)
                    {
                        word_type vl = vsetvl_e32m8(n1);
                        vfloat32m8_t _a0 = vfmv_v_f_f32m8(a[0], vl);
                        vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                        vfloat32m8_t _outp = op(_a0, _p1, vl);
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

                const float* ptr1 = b;
                float* outptr = c;

                int n1 = size1 * elempack1;
                while (n1 > 0)
                {
                    word_type vl = vsetvl_e32m8(n1);
                    vfloat32m8_t _a0 = vfmv_v_f_f32m8(a[0], vl);
                    vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                    vfloat32m8_t _outp = op(_a0, _p1, vl);
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

                const float* ptr1 = b;
                float* outptr = c;
                int n1 = w1 * elempack1;
                while (n1 > 0)
                {
                    word_type vl = vsetvl_e32m8(n1);

                    vfloat32m8_t _a0 = vfmv_v_f_f32m8(a[0], vl);
                    vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                    vfloat32m8_t _outp = op(_a0, _p1, vl);
                    vse32_v_f32m8(outptr, _outp, vl);

                    ptr1 += vl;
                    outptr += vl;
                    n1 -= vl;
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
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);
                int n1 = size1 * elempack1;
                while (n1 > 0)
                {
                    int n = elempack;
                    const float* ptr_vol = (const float*)a + q * elempack;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e32m8(n);
                        vfloat32m8_t _a0 = vle32_v_f32m8(ptr_vol, vl);
                        vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                        vfloat32m8_t _outp = op(_a0, _p1, vl);
                        vse32_v_f32m8(outptr, _outp, vl);
                        ptr1 += vl;
                        outptr += vl;
                        n1 -= vl;
                        n -= vl;
                    }
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
                for (int x = 0; x < w1; x++)
                {
                    const float* ptr_vol = ptr;
                    int n = elempack;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e32m8(n);

                        vfloat32m8_t _a0 = vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                        vfloat32m8_t _outp = op(_a0, _p1, vl);
                        vse32_v_f32m8(outptr, _outp, vl);

                        ptr1 += vl;
                        outptr += vl;
                        ptr_vol += vl;
                        n -= vl;
                    }
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
                float* outptr = c;
                int n = w * elempack;
                while (n > 0)
                {
                    word_type vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _outp = op(_p, b[0], vl);
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

struct binary_op_add_rvv
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const vfloat32m8_t& y,
                            const word_type& vl) const
    {
        return vfadd_vv_f32m8(x, y, vl);
    }
    vfloat32m8_t operator()(const vfloat32m8_t& x, const float& y,
                            const word_type& vl) const
    {
        return vfadd_vf_f32m8(x, y, vl);
    }
};

struct binary_op_sub_rvv
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const vfloat32m8_t& y,
                            const word_type& vl) const
    {
        return vfsub_vv_f32m8(x, y, vl);
    }
    vfloat32m8_t operator()(const vfloat32m8_t& x, float y,
                            const word_type& vl) const
    {
        return vfsub_vf_f32m8(x, y, vl);
    }
};

struct binary_op_mul_rvv
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const vfloat32m8_t& y,
                            const word_type& vl) const
    {
        return vfmul_vv_f32m8(x, y, vl);
    }
    vfloat32m8_t operator()(const vfloat32m8_t& x, float y,
                            const word_type& vl) const
    {
        return vfmul_vf_f32m8(x, y, vl);
    }
};

struct binary_op_div_rvv
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const vfloat32m8_t& y,
                            const word_type& vl) const
    {
        return vfdiv_vv_f32m8(x, y, vl);
    }
    vfloat32m8_t operator()(const vfloat32m8_t& x, float y,
                            const word_type& vl) const
    {
        return vfdiv_vf_f32m8(x, y, vl);
    }
};

struct binary_op_max_rvv
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const vfloat32m8_t& y,
                            const word_type& vl) const
    {
        return vfmax_vv_f32m8(x, y, vl);
    }
    vfloat32m8_t operator()(const vfloat32m8_t& x, float y,
                            const word_type& vl) const
    {
        return vfmax_vf_f32m8(x, y, vl);
    }
};

struct binary_op_min_rvv
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const vfloat32m8_t& y,
                            const word_type& vl) const
    {
        return vfmin_vv_f32m8(x, y, vl);
    }
    vfloat32m8_t operator()(const vfloat32m8_t& x, float y,
                            const word_type& vl) const
    {
        return vfmin_vf_f32m8(x, y, vl);
    }
};

struct binary_op_pow_rvv
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const vfloat32m8_t& y,
                            const word_type& vl) const
    {
        return pow_ps(x, y, vl); // rvv_mathfun.h
    }
    vfloat32m8_t operator()(const vfloat32m8_t& x, float y,
                            const word_type& vl) const
    {
        return pow_ps(x, vfmv_v_f_f32m8(y, vl), vl);
    }
};

struct binary_op_rsub_rvv
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const vfloat32m8_t& y,
                            const word_type& vl) const
    {
        return vfsub_vv_f32m8(y, x, vl);
    }
    vfloat32m8_t operator()(const vfloat32m8_t& x, const float& y,
                            const word_type& vl) const
    {
        return vfrsub_vf_f32m8(x, y, vl);
    }
};

struct binary_op_rdiv_rvv
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const vfloat32m8_t& y,
                            const word_type& vl) const
    {
        return vfdiv_vv_f32m8(y, x, vl);
    }
    vfloat32m8_t operator()(const vfloat32m8_t& x, float y,
                            const word_type& vl) const
    {
        return vfrdiv_vf_f32m8(x, y, vl);
    }
};
#endif

int BinaryOp_riscv::forward(const std::vector<Mat>& bottom_blobs,
                            std::vector<Mat>& top_blobs,
                            const Option& opt) const
{
    int elembits = std::max(bottom_blobs[0].elembits(), bottom_blobs[1].elembits());
#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
    {
        return forward_fp16sa(bottom_blobs, top_blobs, opt);
    }
#endif
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

#if __riscv_vector
    int elempack = bottom_blob.elempack;
    int elempack1 = bottom_blob1.elempack;
    if (elempack != 1 || elempack1 != 1)
    {
        if (op_type == Operation_ADD)
            return binary_op_rvv<binary_op_add_rvv>(bottom_blob, bottom_blob1,
                                                    top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op_rvv<binary_op_sub_rvv>(bottom_blob, bottom_blob1,
                                                    top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op_rvv<binary_op_mul_rvv>(bottom_blob, bottom_blob1,
                                                    top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op_rvv<binary_op_div_rvv>(bottom_blob, bottom_blob1,
                                                    top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op_rvv<binary_op_max_rvv>(bottom_blob, bottom_blob1,
                                                    top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op_rvv<binary_op_min_rvv>(bottom_blob, bottom_blob1,
                                                    top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op_rvv<binary_op_pow_rvv>(bottom_blob, bottom_blob1,
                                                    top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op_rvv<binary_op_rsub_rvv>(bottom_blob, bottom_blob1,
                    top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op_rvv<binary_op_rdiv_rvv>(bottom_blob, bottom_blob1,
                    top_blob, opt);
    }
#endif

    return BinaryOp::forward(bottom_blobs, top_blobs, opt);
}

#if __riscv_vector
template<typename Op>
static int binary_op_scalar_rvv(Mat& a, float b, const Option& opt)
{
    Op op;
    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;
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
#endif

int BinaryOp_riscv::forward_inplace(Mat& bottom_top_blob,
                                    const Option& opt) const
{
#if __riscv_vector
    int elembits = bottom_top_blob.elembits();

#if __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
    {
        return forward_inplace_fp16sa(bottom_top_blob, opt);
    }
#endif
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
static int binary_op_rvv_fp16sa(const Mat& a, const Mat& b, Mat& c,
                                const Option& opt)
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
                    const __fp16* ptr = a.channel(q);
                    const __fp16* b0 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    int n = size * elempack;
                    while (n > 0)
                    {
                        const __fp16* b_vol = b0;
                        int n1 = size1 * elempack1;
                        while (n1 > 0)
                        {
                            word_type vl = vsetvl_e16m8(std::min(n1, n));

                            vfloat16m8_t _b = vle16_v_f16m8(b_vol, vl);
                            vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                            vfloat16m8_t _outp = op(_p, _b, vl);
                            vse16_v_f16m8(outptr, _outp, vl);

                            ptr += vl;
                            b_vol += vl;
                            outptr += vl;

                            n1 -= vl;
                            n -= vl;
                        }
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
                    const __fp16* a0 = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    int n1 = size1 * elempack1;
                    while (n1 > 0)
                    {
                        const __fp16* a_vol = a0;
                        int n = size * elempack;
                        while (n > 0)
                        {
                            word_type vl = vsetvl_e16m8(std::min(n1, n));

                            vfloat16m8_t _a0 = vle16_v_f16m8(a_vol, vl);
                            vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                            vfloat16m8_t _outp = op(_a0, _p1, vl);
                            vse16_v_f16m8(outptr, _outp, vl);

                            ptr1 += vl;
                            a_vol += vl;
                            outptr += vl;

                            n1 -= vl;
                            n -= vl;
                        }
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
                        for (int x = 0; x < w; x++)
                        {
                            const __fp16* ptr1_vol = ptr1 + y * elempack;
                            int n = elempack;
                            while (n > 0)
                            {
                                word_type vl = vsetvl_e16m8(n);
                                vfloat16m8_t _p1 = vle16_v_f16m8(ptr1_vol, vl);
                                vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                                vfloat16m8_t _outp = op(_p, _p1, vl);
                                vse16_v_f16m8(outptr, _outp, vl);
                                ptr += vl;
                                outptr += vl;
                                n -= vl;
                            }
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
                        for (int x = 0; x < w1; x++)
                        {
                            int n = elempack;
                            const __fp16* ptr_vol = ptr + y * elempack;
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
                    for (int x = 0; x < w; x++)
                    {
                        const __fp16* ptr1_vol = ptr1;
                        int n = elempack1;
                        while (n > 0)
                        {
                            word_type vl = vsetvl_e16m8(n);
                            vfloat16m8_t _b0 = vle16_v_f16m8(ptr1_vol, vl);
                            vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                            vfloat16m8_t _outp = op(_p, _b0, vl);
                            vse16_v_f16m8(outptr, _outp, vl);

                            ptr += vl;
                            outptr += vl;
                            ptr1_vol += vl;
                            n -= vl;
                        }
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

                int n = size * elempack;

                while (n > 0)
                {
                    int n1 = elempack1;
                    const __fp16* ptr1_vol = (const __fp16*)b + q * elempack1;
                    while (n1 > 0)
                    {
                        word_type vl = vsetvl_e16m8(n1);

                        vfloat16m8_t _b0 = vle16_v_f16m8(ptr1_vol, vl);
                        vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                        vfloat16m8_t _outp = op(_p, _b0, vl);
                        vse16_v_f16m8(outptr, _outp, vl);

                        ptr1_vol += vl;
                        outptr += vl;
                        ptr += vl;
                        n1 -= vl;
                        n -= vl;
                    }
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
                const __fp16* ptr = a.row<const __fp16>(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int y = 0; y < h1; y++)
                {
                    for (int x = 0; x < w1; x++)
                    {
                        const __fp16* ptr_vol = ptr;
                        int n = elempack1;
                        while (n > 0)
                        {
                            word_type vl = vsetvl_e16m8(n);
                            vfloat16m8_t _a0 = vle16_v_f16m8(ptr_vol, vl);
                            vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                            vfloat16m8_t _outp = op(_a0, _p1, vl);
                            vse16_v_f16m8(outptr, _outp, vl);
                            ptr1 += vl;
                            outptr += vl;
                            ptr_vol += vl;
                            n -= vl;
                        }
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
                const __fp16 b0 = *(const __fp16*)b;
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
                for (int x = 0; x < w; x++)
                {
                    int n = elempack;
                    const __fp16* ptr1_vol = ptr1;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e16m8(n);
                        vfloat16m8_t _b0 = vle16_v_f16m8(ptr1_vol, vl);
                        vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                        vfloat16m8_t _outp = op(_p, _b0, vl);
                        vse16_v_f16m8(outptr, _outp, vl);

                        ptr += vl;
                        ptr1_vol += vl;
                        outptr += vl;
                        n -= vl;
                    }
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
            if (b.dims == 3)
            {
                // type 4
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16 a0 = *(const __fp16*)a;
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    int n1 = size1 * elempack1;
                    while (n1 > 0)
                    {
                        word_type vl = vsetvl_e16m8(n1);
                        vfloat16m8_t _a0 = vfmv_v_f_f16m8(a0, vl);
                        vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                        vfloat16m8_t _outp = op(_a0, _p1, vl);
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

                const __fp16 a0 = *(const __fp16*)a;
                const __fp16* ptr1 = b;
                __fp16* outptr = c;

                int n1 = size1 * elempack1;
                while (n1 > 0)
                {
                    word_type vl = vsetvl_e16m8(n1);
                    vfloat16m8_t _a0 = vfmv_v_f_f16m8(a0, vl);
                    vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                    vfloat16m8_t _outp = op(_a0, _p1, vl);
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

                const __fp16 a0 = *(const __fp16*)a;
                const __fp16* ptr1 = b;
                __fp16* outptr = c;
                int n1 = w1 * elempack1;
                while (n1 > 0)
                {
                    word_type vl = vsetvl_e16m8(n1);

                    vfloat16m8_t _a0 = vfmv_v_f_f16m8(a0, vl);
                    vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                    vfloat16m8_t _outp = op(_a0, _p1, vl);
                    vse16_v_f16m8(outptr, _outp, vl);

                    ptr1 += vl;
                    outptr += vl;
                    n1 -= vl;
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
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);
                int n1 = size1 * elempack1;
                while (n1 > 0)
                {
                    int n = elempack;
                    const __fp16* ptr_vol = (const __fp16*)a + q * elempack;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e16m8(n);
                        vfloat16m8_t _a0 = vle16_v_f16m8(ptr_vol, vl);
                        vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                        vfloat16m8_t _outp = op(_a0, _p1, vl);
                        vse16_v_f16m8(outptr, _outp, vl);
                        ptr1 += vl;
                        outptr += vl;
                        n1 -= vl;
                        n -= vl;
                    }
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
                for (int x = 0; x < w1; x++)
                {
                    const __fp16* ptr_vol = ptr;
                    int n = elempack;
                    while (n > 0)
                    {
                        word_type vl = vsetvl_e16m8(n);

                        vfloat16m8_t _a0 = vle16_v_f16m8(ptr, vl);
                        vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                        vfloat16m8_t _outp = op(_a0, _p1, vl);
                        vse16_v_f16m8(outptr, _outp, vl);

                        ptr1 += vl;
                        outptr += vl;
                        ptr_vol += vl;
                        n -= vl;
                    }
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
                const __fp16 b0 = *(const __fp16*)b;
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

struct binary_op_add_rvv_fp16
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const vfloat16m8_t& y,
                            const word_type& vl) const
    {
        return vfadd_vv_f16m8(x, y, vl);
    }
    vfloat16m8_t operator()(const vfloat16m8_t& x, const float& y,
                            const word_type& vl) const
    {
        return vfadd_vf_f16m8(x, y, vl);
    }
};

struct binary_op_sub_rvv_fp16
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const vfloat16m8_t& y,
                            const word_type& vl) const
    {
        return vfsub_vv_f16m8(x, y, vl);
    }
    vfloat16m8_t operator()(const vfloat16m8_t& x, float y,
                            const word_type& vl) const
    {
        return vfsub_vf_f16m8(x, y, vl);
    }
};

struct binary_op_mul_rvv_fp16
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const vfloat16m8_t& y,
                            const word_type& vl) const
    {
        return vfmul_vv_f16m8(x, y, vl);
    }
    vfloat16m8_t operator()(const vfloat16m8_t& x, float y,
                            const word_type& vl) const
    {
        return vfmul_vf_f16m8(x, y, vl);
    }
};

struct binary_op_div_rvv_fp16
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const vfloat16m8_t& y,
                            const word_type& vl) const
    {
        return vfdiv_vv_f16m8(x, y, vl);
    }
    vfloat16m8_t operator()(const vfloat16m8_t& x, float y,
                            const word_type& vl) const
    {
        return vfdiv_vf_f16m8(x, y, vl);
    }
};

struct binary_op_max_rvv_fp16
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const vfloat16m8_t& y,
                            const word_type& vl) const
    {
        return vfmax_vv_f16m8(x, y, vl);
    }
    vfloat16m8_t operator()(const vfloat16m8_t& x, float y,
                            const word_type& vl) const
    {
        return vfmax_vf_f16m8(x, y, vl);
    }
};

struct binary_op_min_rvv_fp16
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const vfloat16m8_t& y,
                            const word_type& vl) const
    {
        return vfmin_vv_f16m8(x, y, vl);
    }
    vfloat16m8_t operator()(const vfloat16m8_t& x, float y,
                            const word_type& vl) const
    {
        return vfmin_vf_f16m8(x, y, vl);
    }
};

struct binary_op_pow_rvv_fp16
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const vfloat16m8_t& y,
                            const word_type& vl) const
    {
        return pow_ps(x, y, vl); // rvv_mathfun_fp16s.h
    }
    vfloat16m8_t operator()(const vfloat16m8_t& x, const __fp16& y,
                            const word_type& vl) const
    {
        vfloat16m8_t _op2 = vfmv_v_f_f16m8(y, vl);
        vfloat16m8_t retval = pow_ps(x, _op2, vl); // rvv_mathfun_fp16s.h
        return retval;
    }
};

struct binary_op_rsub_rvv_fp16
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const vfloat16m8_t& y,
                            const word_type& vl) const
    {
        return vfsub_vv_f16m8(y, x, vl);
    }
    vfloat16m8_t operator()(const vfloat16m8_t& x, const float& y,
                            const word_type& vl) const
    {
        return vfrsub_vf_f16m8(x, y, vl);
    }
};

struct binary_op_rdiv_rvv_fp16
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const vfloat16m8_t& y,
                            const word_type& vl) const
    {
        return vfdiv_vv_f16m8(y, x, vl);
    }
    vfloat16m8_t operator()(const vfloat16m8_t& x, float y,
                            const word_type& vl) const
    {
        return vfrdiv_vf_f16m8(x, y, vl);
    }
};
#endif

#if __riscv_vector && __riscv_zfh
int BinaryOp_riscv::forward_fp16sa(const std::vector<Mat>& bottom_blobs,
                                   std::vector<Mat>& top_blobs,
                                   const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    if (op_type == Operation_ADD)
        return binary_op_rvv_fp16sa<binary_op_add_rvv_fp16>(bottom_blob, bottom_blob1,
                top_blob, opt);

    if (op_type == Operation_SUB)
        return binary_op_rvv_fp16sa<binary_op_sub_rvv_fp16>(bottom_blob, bottom_blob1,
                top_blob, opt);

    if (op_type == Operation_MUL)
        return binary_op_rvv_fp16sa<binary_op_mul_rvv_fp16>(bottom_blob, bottom_blob1,
                top_blob, opt);

    if (op_type == Operation_DIV)
        return binary_op_rvv_fp16sa<binary_op_div_rvv_fp16>(bottom_blob, bottom_blob1,
                top_blob, opt);

    if (op_type == Operation_MAX)
        return binary_op_rvv_fp16sa<binary_op_max_rvv_fp16>(bottom_blob, bottom_blob1,
                top_blob, opt);

    if (op_type == Operation_MIN)
        return binary_op_rvv_fp16sa<binary_op_min_rvv_fp16>(bottom_blob, bottom_blob1,
                top_blob, opt);

    if (op_type == Operation_POW)
        return binary_op_rvv_fp16sa<binary_op_pow_rvv_fp16>(bottom_blob, bottom_blob1,
                top_blob, opt);

    if (op_type == Operation_RSUB)
        return binary_op_rvv_fp16sa<binary_op_rsub_rvv_fp16>(bottom_blob, bottom_blob1,
                top_blob, opt);

    if (op_type == Operation_RDIV)
        return binary_op_rvv_fp16sa<binary_op_rdiv_rvv_fp16>(bottom_blob, bottom_blob1,
                top_blob, opt);

    return 0;
}

#if __riscv_vector && __riscv_zfh
template<typename Op>
static int binary_op_scalar_rvv_fp16sa(Mat& a, float b, const Option& opt)
{
    Op op;
    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;
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
#endif
int BinaryOp_riscv::forward_inplace_fp16sa(Mat& bottom_top_blob,
        const Option& opt) const
{
    if (op_type == Operation_ADD)
        return binary_op_scalar_rvv_fp16sa<binary_op_add_rvv_fp16>(bottom_top_blob, b,
                opt);

    if (op_type == Operation_SUB)
        return binary_op_scalar_rvv_fp16sa<binary_op_sub_rvv_fp16>(bottom_top_blob, b,
                opt);

    if (op_type == Operation_MUL)
        return binary_op_scalar_rvv_fp16sa<binary_op_mul_rvv_fp16>(bottom_top_blob, b,
                opt);

    if (op_type == Operation_DIV)
        return binary_op_scalar_rvv_fp16sa<binary_op_div_rvv_fp16>(bottom_top_blob, b,
                opt);

    if (op_type == Operation_MAX)
        return binary_op_scalar_rvv_fp16sa<binary_op_max_rvv_fp16>(bottom_top_blob, b,
                opt);

    if (op_type == Operation_MIN)
        return binary_op_scalar_rvv_fp16sa<binary_op_min_rvv_fp16>(bottom_top_blob, b,
                opt);

    if (op_type == Operation_POW)
        return binary_op_scalar_rvv_fp16sa<binary_op_pow_rvv_fp16>(bottom_top_blob, b,
                opt);

    if (op_type == Operation_RSUB)
        return binary_op_scalar_rvv_fp16sa<binary_op_rsub_rvv_fp16>(bottom_top_blob, b,
                opt);

    if (op_type == Operation_RDIV)
        return binary_op_scalar_rvv_fp16sa<binary_op_rdiv_rvv_fp16>(bottom_top_blob, b,
                opt);
    return 0;
}

#endif

} // namespace ncnn
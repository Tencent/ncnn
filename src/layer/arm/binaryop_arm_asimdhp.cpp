// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "binaryop_arm.h"

#include <math.h>

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template<typename Op>
static int binary_op_2_3_4_20_fp16s(const Mat& a, const Mat& b, Mat& c, const Option& opt)
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
        const __fp16 a0 = ((const __fp16*)a)[0];
        const __fp16* ptr = b.channel(q);
        __fp16* outptr = c.channel(q);

        int i = 0;
        float16x8_t _a0 = vdupq_n_f16(a0);
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float16x8_t _outp = op(_a0, _p);
            vst1q_f16(outptr, _outp);
            ptr += 8;
            outptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            float16x4_t _outp = op(vget_low_f16(_a0), _p);
            vst1_f16(outptr, _outp);
            ptr += 4;
            outptr += 4;
        }
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
static int binary_op_6_11_16_25_fp16s(const Mat& a, const Mat& b, Mat& c, const Option& opt)
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
        const __fp16* ptr = a.channel(q);
        const __fp16 b0 = ((const __fp16*)b)[0];
        __fp16* outptr = c.channel(q);

        int i = 0;
        float16x8_t _b0 = vdupq_n_f16(b0);
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float16x8_t _outp = op(_p, _b0);
            vst1q_f16(outptr, _outp);
            ptr += 8;
            outptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            float16x4_t _outp = op(_p, vget_low_f16(_b0));
            vst1_f16(outptr, _outp);
            ptr += 4;
            outptr += 4;
        }
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
static int binary_op_7_13_19_29_fp16s(const Mat& a, const Mat& b, Mat& c, const Option& opt)
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
        const __fp16* ptr = a.channel(q);
        const __fp16* ptr1 = b.channel(q);
        __fp16* outptr = c.channel(q);

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr1);
            float16x8_t _outp = op(_p, _p1);
            vst1q_f16(outptr, _outp);
            ptr += 8;
            ptr1 += 8;
            outptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            float16x4_t _p1 = vld1_f16(ptr1);
            float16x4_t _outp = op(_p, _p1);
            vst1_f16(outptr, _outp);
            ptr += 4;
            ptr1 += 4;
            outptr += 4;
        }
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

template<typename Op>
static int binary_op_pack8_fp16s(const Mat& a, const Mat& b, Mat& c, const Option& opt)
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
            return binary_op_7_13_19_29_fp16s<Op>(a, b, c, opt);
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
                        float16x8_t _b0 = vld1q_f16(ptr1);
                        for (int x = 0; x < w; x++)
                        {
                            float16x8_t _p = vld1q_f16(ptr);
                            float16x8_t _outp = op(_p, _b0);
                            vst1q_f16(outptr, _outp);
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
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.row<const __fp16>(q);
                __fp16* outptr = c.channel(q);

                for (int z = 0; z < d; z++)
                {
                    float16x8_t _b0 = vld1q_f16(ptr1);
                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            float16x8_t _p = vld1q_f16(ptr);
                            float16x8_t _outp = op(_p, _b0);
                            vst1q_f16(outptr, _outp);
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
                return binary_op_6_11_16_25_fp16s<Op>(a, b, c, opt);
            }

            // type 26
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                float16x8_t _b0 = vld1q_f16((const __fp16*)b + q * 8);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _outp = op(_p, _b0);
                    vst1q_f16(outptr, _outp);
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
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int z = 0; z < d1; z++)
                {
                    for (int y = 0; y < h1; y++)
                    {
                        float16x8_t _a0 = vld1q_f16(ptr);
                        for (int x = 0; x < w1; x++)
                        {
                            float16x8_t _p = vld1q_f16(ptr1);
                            float16x8_t _outp = op(_a0, _p);
                            vst1q_f16(outptr, _outp);
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
                    const __fp16* ptr = a.channel(q);
                    __fp16* outptr = c.channel(q);
                    const __fp16* b0 = b.channel(q);
                    float16x8_t _b0 = vld1q_f16(b0);
                    for (int i = 0; i < size; i++)
                    {
                        float16x8_t _p = vld1q_f16(ptr);
                        float16x8_t _outp = op(_p, _b0);
                        vst1q_f16(outptr, _outp);
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
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b;
                    __fp16* outptr = c.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        float16x8_t _p = vld1q_f16(ptr);
                        float16x8_t _p1 = vdupq_n_f16(*ptr1);
                        float16x8_t _outp = op(_p, _p1);
                        vst1q_f16(outptr, _outp);
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
                    const __fp16* a0 = a.channel(q);
                    __fp16* outptr = c.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    float16x8_t _a0 = vld1q_f16(a0);
                    for (int i = 0; i < size1; i++)
                    {
                        float16x8_t _p1 = vld1q_f16(ptr1);
                        float16x8_t _outp = op(_a0, _p1);
                        vst1q_f16(outptr, _outp);
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
                    const __fp16* ptr = a;
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        float16x8_t _p = vdupq_n_f16(*ptr);
                        float16x8_t _p1 = vld1q_f16(ptr1);
                        float16x8_t _outp = op(_p, _p1);
                        vst1q_f16(outptr, _outp);
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
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int y = 0; y < h; y++)
                    {
                        float16x8_t _p1 = vld1q_f16(ptr1 + y * 8);
                        for (int x = 0; x < w; x++)
                        {
                            float16x8_t _p = vld1q_f16(ptr);
                            float16x8_t _outp = op(_p, _p1);
                            vst1q_f16(outptr, _outp);

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
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            float16x8_t _p = vld1q_f16(ptr);
                            float16x8_t _p1 = vld1q_f16(ptr1 + x * 8);
                            float16x8_t _outp = op(_p, _p1);
                            vst1q_f16(outptr, _outp);

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
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int y = 0; y < h1; y++)
                    {
                        float16x8_t _p = vld1q_f16(ptr + y * 8);
                        for (int x = 0; x < w1; x++)
                        {
                            float16x8_t _p1 = vld1q_f16(ptr1);
                            float16x8_t _outp = op(_p, _p1);
                            vst1q_f16(outptr, _outp);

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
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int y = 0; y < h1; y++)
                    {
                        for (int x = 0; x < w1; x++)
                        {
                            float16x8_t _p = vld1q_f16(ptr + x * 8);
                            float16x8_t _p1 = vld1q_f16(ptr1);
                            float16x8_t _outp = op(_p, _p1);
                            vst1q_f16(outptr, _outp);

                            ptr1 += 8;
                            outptr += 8;
                        }
                    }
                }

                return 0;
            }

            // type 19
            return binary_op_7_13_19_29_fp16s<Op>(a, b, c, opt);
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
                    float16x8_t _b0 = vld1q_f16(ptr1);
                    for (int x = 0; x < w; x++)
                    {
                        float16x8_t _p = vld1q_f16(ptr);
                        float16x8_t _outp = op(_p, _b0);
                        vst1q_f16(outptr, _outp);
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
                return binary_op_6_11_16_25_fp16s<Op>(a, b, c, opt);
            }

            // type 17
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                float16x8_t _b0 = vld1q_f16((const __fp16*)b + q * 8);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _outp = op(_p, _b0);
                    vst1q_f16(outptr, _outp);
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
                const __fp16* ptr = a.row<const __fp16>(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int z = 0; z < d1; z++)
                {
                    float16x8_t _a0 = vld1q_f16(ptr);
                    for (int y = 0; y < h1; y++)
                    {
                        for (int x = 0; x < w1; x++)
                        {
                            float16x8_t _p = vld1q_f16(ptr1);
                            float16x8_t _outp = op(_a0, _p);
                            vst1q_f16(outptr, _outp);
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
                const __fp16* ptr = a.row<const __fp16>(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int y = 0; y < h1; y++)
                {
                    float16x8_t _a0 = vld1q_f16(ptr);
                    for (int x = 0; x < w1; x++)
                    {
                        float16x8_t _p1 = vld1q_f16(ptr1);
                        float16x8_t _outp = op(_a0, _p1);
                        vst1q_f16(outptr, _outp);
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
            return binary_op_7_13_19_29_fp16s<Op>(a, b, c, opt);
        }

        if (b.dims == 1)
        {
            c.create(w, h, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1 && elempack1 == 1)
            {
                // type 11
                return binary_op_6_11_16_25_fp16s<Op>(a, b, c, opt);
            }

            // type 12
            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;

            for (int y = 0; y < h; y++)
            {
                float16x8_t _b0 = vld1q_f16(ptr1);
                for (int x = 0; x < w; x++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _outp = op(_p, _b0);
                    vst1q_f16(outptr, _outp);
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
            return binary_op_2_3_4_20_fp16s<Op>(a, b, c, opt);
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
                float16x8_t _a0 = vld1q_f16((const __fp16*)a + q * 8);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    float16x8_t _p1 = vld1q_f16(ptr1);
                    float16x8_t _outp = op(_a0, _p1);
                    vst1q_f16(outptr, _outp);
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
                float16x8_t _a0 = vld1q_f16((const __fp16*)a + q * 8);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    float16x8_t _p1 = vld1q_f16(ptr1);
                    float16x8_t _outp = op(_a0, _p1);
                    vst1q_f16(outptr, _outp);
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

            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;

            for (int y = 0; y < h1; y++)
            {
                float16x8_t _a0 = vld1q_f16(ptr);
                for (int x = 0; x < w1; x++)
                {
                    float16x8_t _p1 = vld1q_f16(ptr1);
                    float16x8_t _outp = op(_a0, _p1);
                    vst1q_f16(outptr, _outp);
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
                return binary_op_6_11_16_25_fp16s<Op>(a, b, c, opt);
            }

            // type 7
            binary_op_7_13_19_29_fp16s<Op>(a, b, c, opt);
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_pack4_fp16s(const Mat& a, const Mat& b, Mat& c, const Option& opt)
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
            return binary_op_7_13_19_29_fp16s<Op>(a, b, c, opt);
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
                        float16x4_t _b0 = vld1_f16(ptr1);
                        for (int x = 0; x < w; x++)
                        {
                            float16x4_t _p = vld1_f16(ptr);
                            float16x4_t _outp = op(_p, _b0);
                            vst1_f16(outptr, _outp);
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
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.row<const __fp16>(q);
                __fp16* outptr = c.channel(q);

                for (int z = 0; z < d; z++)
                {
                    float16x4_t _b0 = vld1_f16(ptr1);
                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            float16x4_t _p = vld1_f16(ptr);
                            float16x4_t _outp = op(_p, _b0);
                            vst1_f16(outptr, _outp);
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
                return binary_op_6_11_16_25_fp16s<Op>(a, b, c, opt);
            }

            // type 26
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                float16x4_t _b0 = vld1_f16((const __fp16*)b + q * 4);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _outp = op(_p, _b0);
                    vst1_f16(outptr, _outp);
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
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int z = 0; z < d1; z++)
                {
                    for (int y = 0; y < h1; y++)
                    {
                        float16x4_t _a0 = vld1_f16(ptr);
                        for (int x = 0; x < w1; x++)
                        {
                            float16x4_t _p = vld1_f16(ptr1);
                            float16x4_t _outp = op(_a0, _p);
                            vst1_f16(outptr, _outp);
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
                    const __fp16* ptr = a.channel(q);
                    __fp16* outptr = c.channel(q);
                    const __fp16* b0 = b.channel(q);
                    float16x4_t _b0 = vld1_f16(b0);
                    for (int i = 0; i < size; i++)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        float16x4_t _outp = op(_p, _b0);
                        vst1_f16(outptr, _outp);
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
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b;
                    __fp16* outptr = c.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        float16x4_t _p1 = vdup_n_f16(*ptr1);
                        float16x4_t _outp = op(_p, _p1);
                        vst1_f16(outptr, _outp);
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
                    const __fp16* a0 = a.channel(q);
                    __fp16* outptr = c.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    float16x4_t _a0 = vld1_f16(a0);
                    for (int i = 0; i < size1; i++)
                    {
                        float16x4_t _p1 = vld1_f16(ptr1);
                        float16x4_t _outp = op(_a0, _p1);
                        vst1_f16(outptr, _outp);
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
                    const __fp16* ptr = a;
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        float16x4_t _p = vdup_n_f16(*ptr);
                        float16x4_t _p1 = vld1_f16(ptr1);
                        float16x4_t _outp = op(_p, _p1);
                        vst1_f16(outptr, _outp);
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
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int y = 0; y < h; y++)
                    {
                        float16x4_t _p1 = vld1_f16(ptr1 + y * 4);
                        for (int x = 0; x < w; x++)
                        {
                            float16x4_t _p = vld1_f16(ptr);
                            float16x4_t _outp = op(_p, _p1);
                            vst1_f16(outptr, _outp);

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
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            float16x4_t _p = vld1_f16(ptr);
                            float16x4_t _p1 = vld1_f16(ptr1 + x * 4);
                            float16x4_t _outp = op(_p, _p1);
                            vst1_f16(outptr, _outp);

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
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int y = 0; y < h1; y++)
                    {
                        float16x4_t _p = vld1_f16(ptr + y * 4);
                        for (int x = 0; x < w1; x++)
                        {
                            float16x4_t _p1 = vld1_f16(ptr1);
                            float16x4_t _outp = op(_p, _p1);
                            vst1_f16(outptr, _outp);

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
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int y = 0; y < h1; y++)
                    {
                        for (int x = 0; x < w1; x++)
                        {
                            float16x4_t _p = vld1_f16(ptr + x * 4);
                            float16x4_t _p1 = vld1_f16(ptr1);
                            float16x4_t _outp = op(_p, _p1);
                            vst1_f16(outptr, _outp);

                            ptr1 += 4;
                            outptr += 4;
                        }
                    }
                }

                return 0;
            }

            // type 19
            return binary_op_7_13_19_29_fp16s<Op>(a, b, c, opt);
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
                    float16x4_t _b0 = vld1_f16(ptr1);
                    for (int x = 0; x < w; x++)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        float16x4_t _outp = op(_p, _b0);
                        vst1_f16(outptr, _outp);
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
                return binary_op_6_11_16_25_fp16s<Op>(a, b, c, opt);
            }

            // type 17
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                float16x4_t _b0 = vld1_f16((const __fp16*)b + q * 4);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _outp = op(_p, _b0);
                    vst1_f16(outptr, _outp);
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
                const __fp16* ptr = a.row<const __fp16>(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int z = 0; z < d1; z++)
                {
                    float16x4_t _a0 = vld1_f16(ptr);
                    for (int y = 0; y < h1; y++)
                    {
                        for (int x = 0; x < w1; x++)
                        {
                            float16x4_t _p = vld1_f16(ptr1);
                            float16x4_t _outp = op(_a0, _p);
                            vst1_f16(outptr, _outp);
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
                const __fp16* ptr = a.row<const __fp16>(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int y = 0; y < h1; y++)
                {
                    float16x4_t _a0 = vld1_f16(ptr);
                    for (int x = 0; x < w1; x++)
                    {
                        float16x4_t _p1 = vld1_f16(ptr1);
                        float16x4_t _outp = op(_a0, _p1);
                        vst1_f16(outptr, _outp);
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
            return binary_op_7_13_19_29_fp16s<Op>(a, b, c, opt);
        }

        if (b.dims == 1)
        {
            c.create(w, h, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1 && elempack1 == 1)
            {
                // type 11
                return binary_op_6_11_16_25_fp16s<Op>(a, b, c, opt);
            }

            // type 12
            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;

            for (int y = 0; y < h; y++)
            {
                float16x4_t _b0 = vld1_f16(ptr1);
                for (int x = 0; x < w; x++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _outp = op(_p, _b0);
                    vst1_f16(outptr, _outp);
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
            return binary_op_2_3_4_20_fp16s<Op>(a, b, c, opt);
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
                float16x4_t _a0 = vld1_f16((const __fp16*)a + q * 4);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    float16x4_t _p1 = vld1_f16(ptr1);
                    float16x4_t _outp = op(_a0, _p1);
                    vst1_f16(outptr, _outp);
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
                float16x4_t _a0 = vld1_f16((const __fp16*)a + q * 4);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    float16x4_t _p1 = vld1_f16(ptr1);
                    float16x4_t _outp = op(_a0, _p1);
                    vst1_f16(outptr, _outp);
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

            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;

            for (int y = 0; y < h1; y++)
            {
                float16x4_t _a0 = vld1_f16(ptr);
                for (int x = 0; x < w1; x++)
                {
                    float16x4_t _p1 = vld1_f16(ptr1);
                    float16x4_t _outp = op(_a0, _p1);
                    vst1_f16(outptr, _outp);
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
                return binary_op_6_11_16_25_fp16s<Op>(a, b, c, opt);
            }

            // type 7
            binary_op_7_13_19_29_fp16s<Op>(a, b, c, opt);
        }
    }

    return 0;
}

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
            return binary_op_7_13_19_29_fp16s<Op>(a, b, c, opt);
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
                return binary_op_6_11_16_25_fp16s<Op>(a, b, c, opt);
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
            return binary_op_7_13_19_29_fp16s<Op>(a, b, c, opt);
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
                return binary_op_6_11_16_25_fp16s<Op>(a, b, c, opt);
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
            return binary_op_7_13_19_29_fp16s<Op>(a, b, c, opt);
        }

        if (b.dims == 1)
        {
            c.create(w, h, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1)
            {
                // type 11
                return binary_op_6_11_16_25_fp16s<Op>(a, b, c, opt);
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
            // type 2 3 4 20
            return binary_op_2_3_4_20_fp16s<Op>(a, b, c, opt);
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
                return binary_op_6_11_16_25_fp16s<Op>(a, b, c, opt);
            }

            // type 7
            binary_op_7_13_19_29_fp16s<Op>(a, b, c, opt);
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_scalar_inplace_fp16s(Mat& a, float b, const Option& opt)
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
        __fp16* ptr = a.channel(q);

        int i = 0;
        float16x8_t _b = vdupq_n_f16((__fp16)b);
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _p = op(_p, _b);
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = op(_p, vget_low_f16(_b));
            vst1_f16(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            *ptr = op(*ptr, (__fp16)b);
            ptr += 1;
        }
    }

    return 0;
}

namespace BinaryOp_arm_functor {

#define MAKE_FUNCTION(NAME, IMPL, IMPL4, IMPL8)                                  \
    struct NAME                                                                  \
    {                                                                            \
        __fp16 operator()(const __fp16& x, const __fp16& y) const                \
        {                                                                        \
            return IMPL;                                                         \
        }                                                                        \
        float16x4_t operator()(const float16x4_t& x, const float16x4_t& y) const \
        {                                                                        \
            return IMPL4;                                                        \
        }                                                                        \
        float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const \
        {                                                                        \
            return IMPL8;                                                        \
        }                                                                        \
    };

// clang-format off
// *INDENT-OFF*
MAKE_FUNCTION(binary_op_add_fp16s, x + y, vadd_f16(x, y), vaddq_f16(x, y))
MAKE_FUNCTION(binary_op_sub_fp16s, x - y, vsub_f16(x, y), vsubq_f16(x, y))
MAKE_FUNCTION(binary_op_mul_fp16s, x * y, vmul_f16(x, y), vmulq_f16(x, y))
MAKE_FUNCTION(binary_op_div_fp16s, x / y, vdiv_f16(x, y), vdivq_f16(x, y))
MAKE_FUNCTION(binary_op_max_fp16s, std::max(x, y), vmax_f16(x, y), vmaxq_f16(x, y))
MAKE_FUNCTION(binary_op_min_fp16s, std::min(x, y), vmin_f16(x, y), vminq_f16(x, y))
MAKE_FUNCTION(binary_op_pow_fp16s, (__fp16)pow(x, y), vcvt_f16_f32(pow_ps(vcvt_f32_f16(x), vcvt_f32_f16(y))), vcombine_f16(vcvt_f16_f32(pow_ps(vcvt_f32_f16(vget_low_f16(x)), vcvt_f32_f16(vget_low_f16(y)))), vcvt_f16_f32(pow_ps(vcvt_f32_f16(vget_high_f16(x)), vcvt_f32_f16(vget_high_f16(y))))))
MAKE_FUNCTION(binary_op_rsub_fp16s, y - x, vsub_f16(y, x), vsubq_f16(y, x))
MAKE_FUNCTION(binary_op_rdiv_fp16s, y / x, vdiv_f16(y, x), vdivq_f16(y, x))
// *INDENT-ON*
// clang-format on

#undef MAKE_FUNCTION

} // namespace BinaryOp_arm_functor

int BinaryOp_arm::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];

    Mat& top_blob = top_blobs[0];

    using namespace BinaryOp_arm_functor;

    int elempack = bottom_blob.elempack;
    int elempack1 = bottom_blob1.elempack;

    if (elempack == 8 || elempack1 == 8)
    {
        if (op_type == Operation_ADD)
            return binary_op_pack8_fp16s<binary_op_add_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op_pack8_fp16s<binary_op_sub_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op_pack8_fp16s<binary_op_mul_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op_pack8_fp16s<binary_op_div_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op_pack8_fp16s<binary_op_max_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op_pack8_fp16s<binary_op_min_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op_pack8_fp16s<binary_op_pow_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op_pack8_fp16s<binary_op_sub_fp16s>(bottom_blob1, bottom_blob, top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op_pack8_fp16s<binary_op_div_fp16s>(bottom_blob1, bottom_blob, top_blob, opt);
    }

    if (elempack == 4 || elempack1 == 4)
    {
        if (op_type == Operation_ADD)
            return binary_op_pack4_fp16s<binary_op_add_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op_pack4_fp16s<binary_op_sub_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op_pack4_fp16s<binary_op_mul_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op_pack4_fp16s<binary_op_div_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op_pack4_fp16s<binary_op_max_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op_pack4_fp16s<binary_op_min_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op_pack4_fp16s<binary_op_pow_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op_pack4_fp16s<binary_op_sub_fp16s>(bottom_blob1, bottom_blob, top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op_pack4_fp16s<binary_op_div_fp16s>(bottom_blob1, bottom_blob, top_blob, opt);
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

int BinaryOp_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    using namespace BinaryOp_arm_functor;

    if (op_type == Operation_ADD)
        return binary_op_scalar_inplace_fp16s<binary_op_add_fp16s>(bottom_top_blob, b, opt);

    if (op_type == Operation_SUB)
        return binary_op_scalar_inplace_fp16s<binary_op_sub_fp16s>(bottom_top_blob, b, opt);

    if (op_type == Operation_MUL)
        return binary_op_scalar_inplace_fp16s<binary_op_mul_fp16s>(bottom_top_blob, b, opt);

    if (op_type == Operation_DIV)
        return binary_op_scalar_inplace_fp16s<binary_op_div_fp16s>(bottom_top_blob, b, opt);

    if (op_type == Operation_MAX)
        return binary_op_scalar_inplace_fp16s<binary_op_max_fp16s>(bottom_top_blob, b, opt);

    if (op_type == Operation_MIN)
        return binary_op_scalar_inplace_fp16s<binary_op_min_fp16s>(bottom_top_blob, b, opt);

    if (op_type == Operation_POW)
        return binary_op_scalar_inplace_fp16s<binary_op_pow_fp16s>(bottom_top_blob, b, opt);

    if (op_type == Operation_RSUB)
        return binary_op_scalar_inplace_fp16s<binary_op_rsub_fp16s>(bottom_top_blob, b, opt);

    if (op_type == Operation_RDIV)
        return binary_op_scalar_inplace_fp16s<binary_op_rdiv_fp16s>(bottom_top_blob, b, opt);

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn

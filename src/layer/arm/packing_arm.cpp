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

#include "packing_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

Packing_arm::Packing_arm()
{
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif

    support_bf16_storage = true;
}

int Packing_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

    if (elembits == 8)
        return forward_int8(bottom_blob, top_blob, opt);

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);
#endif

    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);

    if (use_padding)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    if (elembits != 32)
    {
        // non-fp32 type
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (elempack == out_elempack)
    {
        top_blob = bottom_blob;
        return 0;
    }

    bool pack1to4 = elempack == 1 && out_elempack == 4;
    bool pack4to1 = elempack == 4 && out_elempack == 1;

    if (!pack1to4 && !pack4to1)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
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
        if ((dims == 3 || dims == 4) && channels * elempack % out_elempack != 0)
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

        if (pack1to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* r0 = bottom_blob.row(i * 4);
                const float* r1 = bottom_blob.row(i * 4 + 1);
                const float* r2 = bottom_blob.row(i * 4 + 2);
                const float* r3 = bottom_blob.row(i * 4 + 3);

                float* outptr = top_blob.row(i);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < w; j += 4)
                {
                    float32x4x4_t _p;
                    _p.val[0] = vld1q_f32(r0);
                    _p.val[1] = vld1q_f32(r1);
                    _p.val[2] = vld1q_f32(r2);
                    _p.val[3] = vld1q_f32(r3);
                    vst4q_f32(outptr, _p);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
#endif
                for (; j < w; j++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
            }
        }
        if (pack4to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* r0 = bottom_blob.row(i);

                float* outptr0 = top_blob.row(i * 4);
                float* outptr1 = top_blob.row(i * 4 + 1);
                float* outptr2 = top_blob.row(i * 4 + 2);
                float* outptr3 = top_blob.row(i * 4 + 3);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < w; j += 4)
                {
                    float32x4x4_t _p = vld4q_f32(r0);
                    vst1q_f32(outptr0, _p.val[0]);
                    vst1q_f32(outptr1, _p.val[1]);
                    vst1q_f32(outptr2, _p.val[2]);
                    vst1q_f32(outptr3, _p.val[3]);

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
#endif
                for (; j < w; j++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
            }
        }

        return 0;
    }

    if (dims == 3 || dims == 4)
    {
        int size = w * h * d;
        int outc = channels * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 3)
            top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        else // if (dims == 4)
            top_blob.create(w, h, d, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* r0 = bottom_blob.channel(q * 4);
                const float* r1 = bottom_blob.channel(q * 4 + 1);
                const float* r2 = bottom_blob.channel(q * 4 + 2);
                const float* r3 = bottom_blob.channel(q * 4 + 3);

                float* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4x4_t _p;
                    _p.val[0] = vld1q_f32(r0);
                    _p.val[1] = vld1q_f32(r1);
                    _p.val[2] = vld1q_f32(r2);
                    _p.val[3] = vld1q_f32(r3);
                    vst4q_f32(outptr, _p);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
#endif
                for (; i < size; i++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
            }
        }
        if (pack4to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q * 4);
                float* outptr1 = top_blob.channel(q * 4 + 1);
                float* outptr2 = top_blob.channel(q * 4 + 2);
                float* outptr3 = top_blob.channel(q * 4 + 3);

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4x4_t _p = vld4q_f32(r0);
                    vst1q_f32(outptr0, _p.val[0]);
                    vst1q_f32(outptr1, _p.val[1]);
                    vst1q_f32(outptr2, _p.val[2]);
                    vst1q_f32(outptr3, _p.val[3]);

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
#endif
                for (; i < size; i++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
            }
        }

        return 0;
    }

    return 0;
}

int Packing_arm::forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (use_padding)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (elempack == out_elempack)
    {
        top_blob = bottom_blob;
        return 0;
    }

    bool pack1to4 = elempack == 1 && out_elempack == 4;
    bool pack4to1 = elempack == 4 && out_elempack == 1;
    bool pack1to8 = elempack == 1 && out_elempack == 8;
    bool pack8to1 = elempack == 8 && out_elempack == 1;
    bool pack4to8 = elempack == 4 && out_elempack == 8;
    bool pack8to4 = elempack == 8 && out_elempack == 4;

    if (!pack1to4 && !pack4to1 && !pack1to8 && !pack8to1 && !pack4to8 && !pack8to4)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
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
        if ((dims == 3 || dims == 4) && channels * elempack % out_elempack != 0)
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

        if (pack1to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i * 4);
                const unsigned short* r1 = bottom_blob.row<const unsigned short>(i * 4 + 1);
                const unsigned short* r2 = bottom_blob.row<const unsigned short>(i * 4 + 2);
                const unsigned short* r3 = bottom_blob.row<const unsigned short>(i * 4 + 3);

                unsigned short* outptr = top_blob.row<unsigned short>(i);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < w; j += 4)
                {
                    uint16x4x4_t _p;
                    _p.val[0] = vld1_u16(r0);
                    _p.val[1] = vld1_u16(r1);
                    _p.val[2] = vld1_u16(r2);
                    _p.val[3] = vld1_u16(r3);
                    vst4_u16(outptr, _p);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
#endif
                for (; j < w; j++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
            }
        }
        if (pack4to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i);

                unsigned short* outptr0 = top_blob.row<unsigned short>(i * 4);
                unsigned short* outptr1 = top_blob.row<unsigned short>(i * 4 + 1);
                unsigned short* outptr2 = top_blob.row<unsigned short>(i * 4 + 2);
                unsigned short* outptr3 = top_blob.row<unsigned short>(i * 4 + 3);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < w; j += 4)
                {
                    uint16x4x4_t _p = vld4_u16(r0);
                    vst1_u16(outptr0, _p.val[0]);
                    vst1_u16(outptr1, _p.val[1]);
                    vst1_u16(outptr2, _p.val[2]);
                    vst1_u16(outptr3, _p.val[3]);

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
#endif
                for (; j < w; j++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
            }
        }
        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i * 8);
                const unsigned short* r1 = bottom_blob.row<const unsigned short>(i * 8 + 1);
                const unsigned short* r2 = bottom_blob.row<const unsigned short>(i * 8 + 2);
                const unsigned short* r3 = bottom_blob.row<const unsigned short>(i * 8 + 3);
                const unsigned short* r4 = bottom_blob.row<const unsigned short>(i * 8 + 4);
                const unsigned short* r5 = bottom_blob.row<const unsigned short>(i * 8 + 5);
                const unsigned short* r6 = bottom_blob.row<const unsigned short>(i * 8 + 6);
                const unsigned short* r7 = bottom_blob.row<const unsigned short>(i * 8 + 7);

                unsigned short* outptr = top_blob.row<unsigned short>(i);

                int j = 0;
#if __ARM_NEON
                for (; j + 7 < w; j += 8)
                {
                    // transpose 8x8
#if __aarch64__
                    asm volatile(
                        "ld1    {v0.8h}, [%0], #16      \n"
                        "ld1    {v1.8h}, [%1], #16      \n"
                        "ld1    {v2.8h}, [%2], #16      \n"
                        "ld1    {v3.8h}, [%3], #16      \n"
                        "ld1    {v4.8h}, [%4], #16      \n"
                        "ld1    {v5.8h}, [%5], #16      \n"
                        "ld1    {v6.8h}, [%6], #16      \n"
                        "ld1    {v7.8h}, [%7], #16      \n"

                        "zip1   v16.8h, v0.8h, v4.8h    \n"
                        "zip2   v20.8h, v0.8h, v4.8h    \n"
                        "zip1   v17.8h, v1.8h, v5.8h    \n"
                        "zip2   v21.8h, v1.8h, v5.8h    \n"
                        "zip1   v18.8h, v2.8h, v6.8h    \n"
                        "zip2   v22.8h, v2.8h, v6.8h    \n"
                        "zip1   v19.8h, v3.8h, v7.8h    \n"
                        "zip2   v23.8h, v3.8h, v7.8h    \n"

                        "st4    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"
                        "st4    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"
                        : "=r"(r0),    // %0
                        "=r"(r1),    // %1
                        "=r"(r2),    // %2
                        "=r"(r3),    // %3
                        "=r"(r4),    // %4
                        "=r"(r5),    // %5
                        "=r"(r6),    // %6
                        "=r"(r7),    // %7
                        "=r"(outptr) // %8
                        : "0"(r0),
                        "1"(r1),
                        "2"(r2),
                        "3"(r3),
                        "4"(r4),
                        "5"(r5),
                        "6"(r6),
                        "7"(r7),
                        "8"(outptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else
                    asm volatile(
                        "vld1.u16   {d16-d17}, [%0 : 128]! \n"
                        "vld1.u16   {d18-d19}, [%1 : 128]! \n"
                        "vld1.u16   {d20-d21}, [%2 : 128]! \n"
                        "vld1.u16   {d22-d23}, [%3 : 128]! \n"
                        "vld1.u16   {d24-d25}, [%4 : 128]! \n"
                        "vld1.u16   {d26-d27}, [%5 : 128]! \n"
                        "vld1.u16   {d28-d29}, [%6 : 128]! \n"
                        "vld1.u16   {d30-d31}, [%7 : 128]! \n"

                        "vtrn.u16   q8, q9              \n"
                        "vtrn.u16   q10, q11            \n"
                        "vtrn.u16   q12, q13            \n"
                        "vtrn.u16   q14, q15            \n"

                        "vtrn.u32   q8, q10             \n"
                        "vtrn.u32   q9, q11             \n"
                        "vtrn.u32   q12, q14            \n"
                        "vtrn.u32   q13, q15            \n"

                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"
                        "vswp       d23, d30            \n"

                        "vstm       %8!, {d16-d23}      \n"
                        "vstm       %8!, {d24-d31}      \n"
                        : "=r"(r0),    // %0
                        "=r"(r1),    // %1
                        "=r"(r2),    // %2
                        "=r"(r3),    // %3
                        "=r"(r4),    // %4
                        "=r"(r5),    // %5
                        "=r"(r6),    // %6
                        "=r"(r7),    // %7
                        "=r"(outptr) // %8
                        : "0"(r0),
                        "1"(r1),
                        "2"(r2),
                        "3"(r3),
                        "4"(r4),
                        "5"(r5),
                        "6"(r6),
                        "7"(r7),
                        "8"(outptr)
                        : "memory", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
                }
#endif
                for (; j < w; j++)
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
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i);

                unsigned short* outptr0 = top_blob.row<unsigned short>(i * 8);
                unsigned short* outptr1 = top_blob.row<unsigned short>(i * 8 + 1);
                unsigned short* outptr2 = top_blob.row<unsigned short>(i * 8 + 2);
                unsigned short* outptr3 = top_blob.row<unsigned short>(i * 8 + 3);
                unsigned short* outptr4 = top_blob.row<unsigned short>(i * 8 + 4);
                unsigned short* outptr5 = top_blob.row<unsigned short>(i * 8 + 5);
                unsigned short* outptr6 = top_blob.row<unsigned short>(i * 8 + 6);
                unsigned short* outptr7 = top_blob.row<unsigned short>(i * 8 + 7);

                int j = 0;
#if __ARM_NEON
                for (; j + 7 < w; j += 8)
                {
                    // transpose 8x8
#if __aarch64__
                    asm volatile(
                        "ld4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
                        "ld4    {v4.8h, v5.8h, v6.8h, v7.8h}, [%0], #64 \n"

                        "uzp1   v16.8h, v0.8h, v4.8h    \n"
                        "uzp2   v20.8h, v0.8h, v4.8h    \n"
                        "uzp1   v17.8h, v1.8h, v5.8h    \n"
                        "uzp2   v21.8h, v1.8h, v5.8h    \n"
                        "uzp1   v18.8h, v2.8h, v6.8h    \n"
                        "uzp2   v22.8h, v2.8h, v6.8h    \n"
                        "uzp1   v19.8h, v3.8h, v7.8h    \n"
                        "uzp2   v23.8h, v3.8h, v7.8h    \n"

                        "st1    {v16.8h}, [%1], #16      \n"
                        "st1    {v17.8h}, [%2], #16      \n"
                        "st1    {v18.8h}, [%3], #16      \n"
                        "st1    {v19.8h}, [%4], #16      \n"
                        "st1    {v20.8h}, [%5], #16      \n"
                        "st1    {v21.8h}, [%6], #16      \n"
                        "st1    {v22.8h}, [%7], #16      \n"
                        "st1    {v23.8h}, [%8], #16      \n"
                        : "=r"(r0),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(outptr4), // %5
                        "=r"(outptr5), // %6
                        "=r"(outptr6), // %7
                        "=r"(outptr7)  // %8
                        : "0"(r0),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(outptr4),
                        "6"(outptr5),
                        "7"(outptr6),
                        "8"(outptr7)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else
                    asm volatile(
                        "vldm       %0!, {d16-d23}      \n"
                        "vldm       %0!, {d24-d31}      \n"

                        "vtrn.u16   q8, q9              \n"
                        "vtrn.u16   q10, q11            \n"
                        "vtrn.u16   q12, q13            \n"
                        "vtrn.u16   q14, q15            \n"

                        "vtrn.u32   q8, q10             \n"
                        "vtrn.u32   q9, q11             \n"
                        "vtrn.u32   q12, q14            \n"
                        "vtrn.u32   q13, q15            \n"

                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"
                        "vswp       d23, d30            \n"

                        "vst1.u16   {d16-d17}, [%1 : 128]! \n"
                        "vst1.u16   {d18-d19}, [%2 : 128]! \n"
                        "vst1.u16   {d20-d21}, [%3 : 128]! \n"
                        "vst1.u16   {d22-d23}, [%4 : 128]! \n"
                        "vst1.u16   {d24-d25}, [%5 : 128]! \n"
                        "vst1.u16   {d26-d27}, [%6 : 128]! \n"
                        "vst1.u16   {d28-d29}, [%7 : 128]! \n"
                        "vst1.u16   {d30-d31}, [%8 : 128]! \n"
                        : "=r"(r0),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(outptr4), // %5
                        "=r"(outptr5), // %6
                        "=r"(outptr6), // %7
                        "=r"(outptr7)  // %8
                        : "0"(r0),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(outptr4),
                        "6"(outptr5),
                        "7"(outptr6),
                        "8"(outptr7)
                        : "memory", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
                }
#endif
                for (; j < w; j++)
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
        if (pack4to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i * 2);
                const unsigned short* r1 = bottom_blob.row<const unsigned short>(i * 2 + 1);

                unsigned short* outptr = top_blob.row<unsigned short>(i);

                int j = 0;
#if __ARM_NEON
                for (; j + 1 < w; j += 2)
                {
#if __aarch64__
                    asm volatile(
                        "ld1    {v0.8h}, [%0], #16      \n"
                        "ld1    {v1.8h}, [%1], #16      \n"

                        "zip1   v2.2d, v0.2d, v1.2d     \n"
                        "zip2   v3.2d, v0.2d, v1.2d     \n"

                        "st1    {v2.8h, v3.8h}, [%2], #32\n"
                        : "=r"(r0),    // %0
                        "=r"(r1),    // %1
                        "=r"(outptr) // %2
                        : "0"(r0),
                        "1"(r1),
                        "2"(outptr)
                        : "memory", "v0", "v1", "v2", "v3");
#else
                    asm volatile(
                        "vld1.u16   {d0-d1}, [%0 :128]! \n"
                        "vld1.u16   {d2-d3}, [%1 :128]! \n"

                        "vswp       d1, d2              \n"

                        "vst1.u16   {d0-d3}, [%2 :128]! \n"
                        : "=r"(r0),    // %0
                        "=r"(r1),    // %1
                        "=r"(outptr) // %2
                        : "0"(r0),
                        "1"(r1),
                        "2"(outptr)
                        : "memory", "q0", "q1");
#endif
                }
#endif
                for (; j < w; j++)
                {
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r1[0];
                    outptr[5] = r1[1];
                    outptr[6] = r1[2];
                    outptr[7] = r1[3];

                    r0 += 4;
                    r1 += 4;
                    outptr += 8;
                }
            }
        }
        if (pack8to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i);

                unsigned short* outptr0 = top_blob.row<unsigned short>(i * 2);
                unsigned short* outptr1 = top_blob.row<unsigned short>(i * 2 + 1);

                int j = 0;
#if __ARM_NEON
                for (; j + 1 < w; j += 2)
                {
#if __aarch64__
                    asm volatile(
                        "ld1    {v0.8h, v1.8h}, [%0], #32 \n"

                        "uzp1   v2.2d, v0.2d, v1.2d     \n"
                        "uzp2   v3.2d, v0.2d, v1.2d     \n"

                        "st1    {v2.8h}, [%1], #16      \n"
                        "st1    {v3.8h}, [%2], #16      \n"
                        : "=r"(r0),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1)  // %2
                        : "0"(r0),
                        "1"(outptr0),
                        "2"(outptr1)
                        : "memory", "v0", "v1", "v2", "v3");
#else
                    asm volatile(
                        "vld1.u16   {d0-d3}, [%0 :128]! \n"

                        "vswp       d1, d2              \n"

                        "vst1.u16   {d0-d1}, [%1 :128]! \n"
                        "vst1.u16   {d2-d3}, [%2 :128]! \n"
                        : "=r"(r0),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1)  // %2
                        : "0"(r0),
                        "1"(outptr0),
                        "2"(outptr1)
                        : "memory", "q0", "q1");
#endif
                }
#endif
                for (; j < w; j++)
                {
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr1[0] = r0[4];
                    outptr1[1] = r0[5];
                    outptr1[2] = r0[6];
                    outptr1[3] = r0[7];

                    r0 += 8;
                    outptr0 += 4;
                    outptr1 += 4;
                }
            }
        }

        return 0;
    }

    if (dims == 3 || dims == 4)
    {
        int size = w * h * d;
        int outc = channels * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 3)
            top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        else // if (dims == 4)
            top_blob.create(w, h, d, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const unsigned short* r0 = bottom_blob.channel(q * 4);
                const unsigned short* r1 = bottom_blob.channel(q * 4 + 1);
                const unsigned short* r2 = bottom_blob.channel(q * 4 + 2);
                const unsigned short* r3 = bottom_blob.channel(q * 4 + 3);

                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    uint16x4x4_t _p;
                    _p.val[0] = vld1_u16(r0);
                    _p.val[1] = vld1_u16(r1);
                    _p.val[2] = vld1_u16(r2);
                    _p.val[3] = vld1_u16(r3);
                    vst4_u16(outptr, _p);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
#endif
                for (; i < size; i++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
            }
        }
        if (pack4to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* r0 = bottom_blob.channel(q);

                unsigned short* outptr0 = top_blob.channel(q * 4);
                unsigned short* outptr1 = top_blob.channel(q * 4 + 1);
                unsigned short* outptr2 = top_blob.channel(q * 4 + 2);
                unsigned short* outptr3 = top_blob.channel(q * 4 + 3);

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    uint16x4x4_t _p = vld4_u16(r0);
                    vst1_u16(outptr0, _p.val[0]);
                    vst1_u16(outptr1, _p.val[1]);
                    vst1_u16(outptr2, _p.val[2]);
                    vst1_u16(outptr3, _p.val[3]);

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
#endif
                for (; i < size; i++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
            }
        }
        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const unsigned short* r0 = bottom_blob.channel(q * 8);
                const unsigned short* r1 = bottom_blob.channel(q * 8 + 1);
                const unsigned short* r2 = bottom_blob.channel(q * 8 + 2);
                const unsigned short* r3 = bottom_blob.channel(q * 8 + 3);
                const unsigned short* r4 = bottom_blob.channel(q * 8 + 4);
                const unsigned short* r5 = bottom_blob.channel(q * 8 + 5);
                const unsigned short* r6 = bottom_blob.channel(q * 8 + 6);
                const unsigned short* r7 = bottom_blob.channel(q * 8 + 7);

                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 7 < size; i += 8)
                {
                    // transpose 8x8
#if __aarch64__
                    asm volatile(
                        "ld1    {v0.8h}, [%0], #16      \n"
                        "ld1    {v1.8h}, [%1], #16      \n"
                        "ld1    {v2.8h}, [%2], #16      \n"
                        "ld1    {v3.8h}, [%3], #16      \n"
                        "ld1    {v4.8h}, [%4], #16      \n"
                        "ld1    {v5.8h}, [%5], #16      \n"
                        "ld1    {v6.8h}, [%6], #16      \n"
                        "ld1    {v7.8h}, [%7], #16      \n"

                        "zip1   v16.8h, v0.8h, v4.8h    \n"
                        "zip2   v20.8h, v0.8h, v4.8h    \n"
                        "zip1   v17.8h, v1.8h, v5.8h    \n"
                        "zip2   v21.8h, v1.8h, v5.8h    \n"
                        "zip1   v18.8h, v2.8h, v6.8h    \n"
                        "zip2   v22.8h, v2.8h, v6.8h    \n"
                        "zip1   v19.8h, v3.8h, v7.8h    \n"
                        "zip2   v23.8h, v3.8h, v7.8h    \n"

                        "st4    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"
                        "st4    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"
                        : "=r"(r0),    // %0
                        "=r"(r1),    // %1
                        "=r"(r2),    // %2
                        "=r"(r3),    // %3
                        "=r"(r4),    // %4
                        "=r"(r5),    // %5
                        "=r"(r6),    // %6
                        "=r"(r7),    // %7
                        "=r"(outptr) // %8
                        : "0"(r0),
                        "1"(r1),
                        "2"(r2),
                        "3"(r3),
                        "4"(r4),
                        "5"(r5),
                        "6"(r6),
                        "7"(r7),
                        "8"(outptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else
                    asm volatile(
                        "vld1.u16   {d16-d17}, [%0 : 128]! \n"
                        "vld1.u16   {d18-d19}, [%1 : 128]! \n"
                        "vld1.u16   {d20-d21}, [%2 : 128]! \n"
                        "vld1.u16   {d22-d23}, [%3 : 128]! \n"
                        "vld1.u16   {d24-d25}, [%4 : 128]! \n"
                        "vld1.u16   {d26-d27}, [%5 : 128]! \n"
                        "vld1.u16   {d28-d29}, [%6 : 128]! \n"
                        "vld1.u16   {d30-d31}, [%7 : 128]! \n"

                        "vtrn.u16   q8, q9              \n"
                        "vtrn.u16   q10, q11            \n"
                        "vtrn.u16   q12, q13            \n"
                        "vtrn.u16   q14, q15            \n"

                        "vtrn.u32   q8, q10             \n"
                        "vtrn.u32   q9, q11             \n"
                        "vtrn.u32   q12, q14            \n"
                        "vtrn.u32   q13, q15            \n"

                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"
                        "vswp       d23, d30            \n"

                        "vstm       %8!, {d16-d23}      \n"
                        "vstm       %8!, {d24-d31}      \n"
                        : "=r"(r0),    // %0
                        "=r"(r1),    // %1
                        "=r"(r2),    // %2
                        "=r"(r3),    // %3
                        "=r"(r4),    // %4
                        "=r"(r5),    // %5
                        "=r"(r6),    // %6
                        "=r"(r7),    // %7
                        "=r"(outptr) // %8
                        : "0"(r0),
                        "1"(r1),
                        "2"(r2),
                        "3"(r3),
                        "4"(r4),
                        "5"(r5),
                        "6"(r6),
                        "7"(r7),
                        "8"(outptr)
                        : "memory", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
                }
#endif
                for (; i < size; i++)
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
                const unsigned short* r0 = bottom_blob.channel(q);

                unsigned short* outptr0 = top_blob.channel(q * 8);
                unsigned short* outptr1 = top_blob.channel(q * 8 + 1);
                unsigned short* outptr2 = top_blob.channel(q * 8 + 2);
                unsigned short* outptr3 = top_blob.channel(q * 8 + 3);
                unsigned short* outptr4 = top_blob.channel(q * 8 + 4);
                unsigned short* outptr5 = top_blob.channel(q * 8 + 5);
                unsigned short* outptr6 = top_blob.channel(q * 8 + 6);
                unsigned short* outptr7 = top_blob.channel(q * 8 + 7);

                int i = 0;
#if __ARM_NEON
                for (; i + 7 < size; i += 8)
                {
                    // transpose 8x8
#if __aarch64__
                    asm volatile(
                        "ld4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
                        "ld4    {v4.8h, v5.8h, v6.8h, v7.8h}, [%0], #64 \n"

                        "uzp1   v16.8h, v0.8h, v4.8h    \n"
                        "uzp2   v20.8h, v0.8h, v4.8h    \n"
                        "uzp1   v17.8h, v1.8h, v5.8h    \n"
                        "uzp2   v21.8h, v1.8h, v5.8h    \n"
                        "uzp1   v18.8h, v2.8h, v6.8h    \n"
                        "uzp2   v22.8h, v2.8h, v6.8h    \n"
                        "uzp1   v19.8h, v3.8h, v7.8h    \n"
                        "uzp2   v23.8h, v3.8h, v7.8h    \n"

                        "st1    {v16.8h}, [%1], #16      \n"
                        "st1    {v17.8h}, [%2], #16      \n"
                        "st1    {v18.8h}, [%3], #16      \n"
                        "st1    {v19.8h}, [%4], #16      \n"
                        "st1    {v20.8h}, [%5], #16      \n"
                        "st1    {v21.8h}, [%6], #16      \n"
                        "st1    {v22.8h}, [%7], #16      \n"
                        "st1    {v23.8h}, [%8], #16      \n"
                        : "=r"(r0),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(outptr4), // %5
                        "=r"(outptr5), // %6
                        "=r"(outptr6), // %7
                        "=r"(outptr7)  // %8
                        : "0"(r0),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(outptr4),
                        "6"(outptr5),
                        "7"(outptr6),
                        "8"(outptr7)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else
                    asm volatile(
                        "vldm       %0!, {d16-d23}      \n"
                        "vldm       %0!, {d24-d31}      \n"

                        "vtrn.u16   q8, q9              \n"
                        "vtrn.u16   q10, q11            \n"
                        "vtrn.u16   q12, q13            \n"
                        "vtrn.u16   q14, q15            \n"

                        "vtrn.u32   q8, q10             \n"
                        "vtrn.u32   q9, q11             \n"
                        "vtrn.u32   q12, q14            \n"
                        "vtrn.u32   q13, q15            \n"

                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"
                        "vswp       d23, d30            \n"

                        "vst1.u16   {d16-d17}, [%1 : 128]! \n"
                        "vst1.u16   {d18-d19}, [%2 : 128]! \n"
                        "vst1.u16   {d20-d21}, [%3 : 128]! \n"
                        "vst1.u16   {d22-d23}, [%4 : 128]! \n"
                        "vst1.u16   {d24-d25}, [%5 : 128]! \n"
                        "vst1.u16   {d26-d27}, [%6 : 128]! \n"
                        "vst1.u16   {d28-d29}, [%7 : 128]! \n"
                        "vst1.u16   {d30-d31}, [%8 : 128]! \n"
                        : "=r"(r0),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(outptr4), // %5
                        "=r"(outptr5), // %6
                        "=r"(outptr6), // %7
                        "=r"(outptr7)  // %8
                        : "0"(r0),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(outptr4),
                        "6"(outptr5),
                        "7"(outptr6),
                        "8"(outptr7)
                        : "memory", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
                }
#endif
                for (; i < size; i++)
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
        if (pack4to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const unsigned short* r0 = bottom_blob.channel(q * 2);
                const unsigned short* r1 = bottom_blob.channel(q * 2 + 1);

                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 1 < size; i += 2)
                {
#if __aarch64__
                    asm volatile(
                        "ld1    {v0.8h}, [%0], #16      \n"
                        "ld1    {v1.8h}, [%1], #16      \n"

                        "zip1   v2.2d, v0.2d, v1.2d     \n"
                        "zip2   v3.2d, v0.2d, v1.2d     \n"

                        "st1    {v2.8h, v3.8h}, [%2], #32\n"
                        : "=r"(r0),    // %0
                        "=r"(r1),    // %1
                        "=r"(outptr) // %2
                        : "0"(r0),
                        "1"(r1),
                        "2"(outptr)
                        : "memory", "v0", "v1", "v2", "v3");
#else
                    asm volatile(
                        "vld1.u16   {d0-d1}, [%0 :128]! \n"
                        "vld1.u16   {d2-d3}, [%1 :128]! \n"

                        "vswp       d1, d2              \n"

                        "vst1.u16   {d0-d3}, [%2 :128]! \n"
                        : "=r"(r0),    // %0
                        "=r"(r1),    // %1
                        "=r"(outptr) // %2
                        : "0"(r0),
                        "1"(r1),
                        "2"(outptr)
                        : "memory", "q0", "q1");
#endif
                }
#endif
                for (; i < size; i++)
                {
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r1[0];
                    outptr[5] = r1[1];
                    outptr[6] = r1[2];
                    outptr[7] = r1[3];

                    r0 += 4;
                    r1 += 4;
                    outptr += 8;
                }
            }
        }
        if (pack8to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* r0 = bottom_blob.channel(q);

                unsigned short* outptr0 = top_blob.channel(q * 2);
                unsigned short* outptr1 = top_blob.channel(q * 2 + 1);

                int i = 0;
#if __ARM_NEON
                for (; i + 1 < size; i += 2)
                {
#if __aarch64__
                    asm volatile(
                        "ld1    {v0.8h, v1.8h}, [%0], #32 \n"

                        "uzp1   v2.2d, v0.2d, v1.2d     \n"
                        "uzp2   v3.2d, v0.2d, v1.2d     \n"

                        "st1    {v2.8h}, [%1], #16      \n"
                        "st1    {v3.8h}, [%2], #16      \n"
                        : "=r"(r0),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1)  // %2
                        : "0"(r0),
                        "1"(outptr0),
                        "2"(outptr1)
                        : "memory", "v0", "v1", "v2", "v3");
#else
                    asm volatile(
                        "vld1.u16   {d0-d3}, [%0 :128]! \n"

                        "vswp       d1, d2              \n"

                        "vst1.u16   {d0-d1}, [%1 :128]! \n"
                        "vst1.u16   {d2-d3}, [%2 :128]! \n"
                        : "=r"(r0),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1)  // %2
                        : "0"(r0),
                        "1"(outptr0),
                        "2"(outptr1)
                        : "memory", "q0", "q1");
#endif
                }
#endif
                for (; i < size; i++)
                {
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr1[0] = r0[4];
                    outptr1[1] = r0[5];
                    outptr1[2] = r0[6];
                    outptr1[3] = r0[7];

                    r0 += 8;
                    outptr0 += 4;
                    outptr1 += 4;
                }
            }
        }

        return 0;
    }

    return 0;
}

int Packing_arm::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (use_padding)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

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
    int d = bottom_blob.d;
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
        if ((dims == 3 || dims == 4) && channels * elempack % out_elempack != 0)
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
                const signed char* r0 = bottom_blob.row<const signed char>(i * 8);
                const signed char* r1 = bottom_blob.row<const signed char>(i * 8 + 1);
                const signed char* r2 = bottom_blob.row<const signed char>(i * 8 + 2);
                const signed char* r3 = bottom_blob.row<const signed char>(i * 8 + 3);
                const signed char* r4 = bottom_blob.row<const signed char>(i * 8 + 4);
                const signed char* r5 = bottom_blob.row<const signed char>(i * 8 + 5);
                const signed char* r6 = bottom_blob.row<const signed char>(i * 8 + 6);
                const signed char* r7 = bottom_blob.row<const signed char>(i * 8 + 7);

                signed char* outptr = top_blob.row<signed char>(i);

                int j = 0;
                for (; j < w; j++)
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
                const signed char* r0 = bottom_blob.row<const signed char>(i);

                signed char* outptr0 = top_blob.row<signed char>(i * 8);
                signed char* outptr1 = top_blob.row<signed char>(i * 8 + 1);
                signed char* outptr2 = top_blob.row<signed char>(i * 8 + 2);
                signed char* outptr3 = top_blob.row<signed char>(i * 8 + 3);
                signed char* outptr4 = top_blob.row<signed char>(i * 8 + 4);
                signed char* outptr5 = top_blob.row<signed char>(i * 8 + 5);
                signed char* outptr6 = top_blob.row<signed char>(i * 8 + 6);
                signed char* outptr7 = top_blob.row<signed char>(i * 8 + 7);

                int j = 0;
                for (; j < w; j++)
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

    if (dims == 3 || dims == 4)
    {
        int size = w * h * d;
        int outc = channels * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 3)
            top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        else // if (dims == 4)
            top_blob.create(w, h, d, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const signed char* r0 = bottom_blob.channel(q * 8);
                const signed char* r1 = bottom_blob.channel(q * 8 + 1);
                const signed char* r2 = bottom_blob.channel(q * 8 + 2);
                const signed char* r3 = bottom_blob.channel(q * 8 + 3);
                const signed char* r4 = bottom_blob.channel(q * 8 + 4);
                const signed char* r5 = bottom_blob.channel(q * 8 + 5);
                const signed char* r6 = bottom_blob.channel(q * 8 + 6);
                const signed char* r7 = bottom_blob.channel(q * 8 + 7);

                signed char* outptr = top_blob.channel(q);

                int i = 0;
                for (; i < size; i++)
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
                const signed char* r0 = bottom_blob.channel(q);

                signed char* outptr0 = top_blob.channel(q * 8);
                signed char* outptr1 = top_blob.channel(q * 8 + 1);
                signed char* outptr2 = top_blob.channel(q * 8 + 2);
                signed char* outptr3 = top_blob.channel(q * 8 + 3);
                signed char* outptr4 = top_blob.channel(q * 8 + 4);
                signed char* outptr5 = top_blob.channel(q * 8 + 5);
                signed char* outptr6 = top_blob.channel(q * 8 + 6);
                signed char* outptr7 = top_blob.channel(q * 8 + 7);

                int i = 0;
                for (; i < size; i++)
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

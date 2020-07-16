// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "quantize_arm.h"

#include <math.h>

namespace ncnn {

static inline signed char float2int8(float v)
{
    int int32 = round(v);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

int Quantize_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        int w = bottom_blob.w;

        top_blob.create(w, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* ptr = bottom_blob;
        signed char* outptr = top_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < w; i++)
        {
            outptr[i] = float2int8(ptr[i] * scale);
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int size = w * h;

        top_blob.create(w, h, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* ptr = bottom_blob;
        signed char* outptr = top_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < size; i++)
        {
            outptr[i] = float2int8(ptr[i] * scale);
        }
    }

    if (dims == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        top_blob.create(w, h, channels, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            signed char* outptr = top_blob.channel(q);

#if __ARM_NEON
            int nn = size >> 3;
            int remain = size & 7;
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
                asm volatile(
                    "dup    v2.4s, %w6                   \n" //scale
                    "0:                                  \n"
                    "prfm   pldl1keep, [%1, #128]        \n"
                    "ld1    {v0.4s, v1.4s}, [%1], #32    \n" //data
                    // bottom_f32 = bottom_f32 * scale
                    "fmul   v3.4s, v0.4s, v2.4s          \n"
                    "fmul   v4.4s, v1.4s, v2.4s          \n"
                    // top_f32 -> top_s32
                    "fcvtas v5.4s, v3.4s                 \n"
                    "fcvtas v6.4s, v4.4s                 \n"
                    // top_s32 -> top_s16
                    "sqxtn  v7.4h, v5.4s                 \n"
                    "sqxtn2 v7.8h, v6.4s                 \n"
                    // top_s16 -> top_s8
                    "sqxtn  v8.8b, v7.8h                 \n"
                    // save top_s8
                    "st1    {v8.8b}, [%2], #8            \n"
                    "subs   %w0, %w0, #1                 \n"
                    "bne    0b                           \n"
                    : "=r"(nn),    // %0
                    "=r"(ptr),   // %1
                    "=r"(outptr) // %2
                    : "0"(nn),
                    "1"(ptr),
                    "2"(outptr),
                    "r"(scale) // %6
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8");
            }
#else
            if (nn > 0)
            {
                asm volatile(
                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d0-d3}, [%1]!      \n"
                    "vdup.32    q10, %6             \n"

                    "0:                             \n"
                    "vmul.f32   q0,q0,q10           \n"
                    "vmul.f32   q1,q1,q10           \n"

                    "vcvtr.s32.f32 s0,s0            \n"
                    "vcvtr.s32.f32 s1,s1            \n"
                    "vcvtr.s32.f32 s2,s2            \n"
                    "vcvtr.s32.f32 s3,s3            \n"
                    "vcvtr.s32.f32 s4,s4            \n"
                    "vcvtr.s32.f32 s5,s5            \n"
                    "vcvtr.s32.f32 s6,s6            \n"
                    "vcvtr.s32.f32 s7,s7            \n"

                    "vqmovn.s32 d4,q0               \n"
                    "vqmovn.s32 d5,q1               \n"

                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d0-d3}, [%1]!      \n"

                    "vqmovn.s16 d4, q2              \n"
                    "vst1.8     {d4}, [%2]!         \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    "sub        %1, #32             \n"
                    : "=r"(nn),    // %0
                    "=r"(ptr),   // %1
                    "=r"(outptr) // %2
                    : "0"(nn),
                    "1"(ptr),
                    "2"(outptr),
                    "r"(scale) // %6
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q10", "q11");
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain > 0; remain--)
            {
                *outptr = float2int8(*ptr * scale);

                ptr++;
                outptr++;
            }
        }
    }

    return 0;
}

} // namespace ncnn

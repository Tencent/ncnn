// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 SenseNets Technology Ltd. All rights reserved.
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

#include "dequantize_arm.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Dequantize_arm)

int Dequantize_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        int* intptr = bottom_top_blob;
        float* ptr = bottom_top_blob;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<w; i++)
            {
                ptr[i] = intptr[i] * scale + bias_data[i];
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<w; i++)
            {
                ptr[i] = intptr[i] * scale;
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<h; i++)
            {
                const int* intptr = bottom_top_blob.row<const int>(i);
                float* ptr = bottom_top_blob.row(i);

                float bias = bias_data_size > 1 ? bias_data[i] : bias_data[0];

                for (int j=0; j<w; j++)
                {
                    ptr[j] = intptr[j] * scale + bias;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<h; i++)
            {
                const int* intptr = bottom_top_blob.row<const int>(i);
                float* ptr = bottom_top_blob.row(i);

                for (int j=0; j<w; j++)
                {
                    ptr[j] = intptr[j] * scale;
                }
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                int* intptr = bottom_top_blob.channel(q);
                float* ptr = bottom_top_blob.channel(q);

                float bias = bias_data[q];

#if __ARM_NEON
                int nn = size >> 3;
                int remain = size & 7;
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                float32x4_t _bias = vdupq_n_f32(bias);
                float32x4_t _scale = vdupq_n_f32(scale);

                if (nn > 0)
                {
                asm volatile(
                    "dup    v2.4s, %w6                   \n" // scale
                    "dup    v3.4s, %w7                   \n" // bias
                    "0:                                  \n"
                    "prfm   pldl1keep, [%1, #128]        \n"
                    "ld1    {v0.4s, v1.4s}, [%1], #32    \n" // data
                    // top_s32 -> top_f32
                    "scvtf  v5.4s, v0.4s                 \n"
                    "scvtf  v6.4s, v1.4s                 \n"
                    // top_f32 = top_f32 * scale_out
                    "fmul   v5.4s, v5.4s, v2.4s          \n"
                    "fmul   v6.4s, v6.4s, v2.4s          \n"
                    // top_f32 = top_f32 + bias_tm
                    "fadd   v5.4s, v5.4s, v3.4s          \n"
                    "fadd   v6.4s, v6.4s, v3.4s          \n"
                    // save top_f32
                    "st1    {v5.4s, v6.4s}, [%2], #32    \n"
                    "subs   %w0, %w0, #1                 \n"
                    "bne    0b                           \n"
                    : "=r"(nn),         // %0
                      "=r"(intptr),     // %1
                      "=r"(ptr)         // %2
                    : "0"(nn),
                      "1"(intptr),
                      "2"(ptr),
                      "r"(_scale),      // %6
                      "r"(_bias)        // %7
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%1, #256]          \n"
                    "vld1.s32   {d0-d3}, [%1]!      \n" //q0-q1 data
                    "vdup.f32   q10, %6             \n" //q10 scale
                    "vdup.f32   q12, %7             \n" //q12 bias

                    "0:                             \n"
                    "vcvt.f32.s32 q0, q0            \n"
                    "vcvt.f32.s32 q1, q1            \n"

                    "vmul.f32   q0,q0,q10           \n"
                    "vmul.f32   q1,q1,q10           \n"

                    "vadd.f32   q2,q0,q12           \n"
                    "vadd.f32   q3,q1,q12           \n"

                    "pld        [%1, #256]          \n"
                    "vld1.s32   {d0-d3}, [%1]!      \n"
                    "vst1.f32   {d4-d7}, [%2]!      \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    "sub        %1, #32             \n"
                    : "=r"(nn),         // %0
                      "=r"(intptr),     // %1
                      "=r"(ptr)         // %2
                    : "0"(nn),
                      "1"(intptr),
                      "2"(ptr),
                      "r"(scale),       // %6
                      "r"(bias)         // %7
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q10", "q12"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    *ptr = *intptr * scale + bias;

                    intptr++;
                    ptr++;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                int* intptr = bottom_top_blob.channel(q);
                float* ptr = bottom_top_blob.channel(q);

#if __ARM_NEON
                int nn = size >> 3;
                int remain = size & 7;
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                float32x4_t _scale = vdupq_n_f32(scale);

                if (nn > 0)
                {
                asm volatile(
                    "dup    v2.4s, %w6                   \n" // scale
                    "0:                                  \n"
                    "prfm   pldl1keep, [%1, #128]      \n"
                    "ld1    {v0.4s, v1.4s}, [%1], #32    \n" // data
                    // top_s32 -> top_f32
                    "scvtf  v5.4s, v0.4s                 \n"
                    "scvtf  v6.4s, v1.4s                 \n"
                    // top_f32 = top_f32 * scale_out
                    "fmul   v5.4s, v5.4s, v2.4s          \n"
                    "fmul   v6.4s, v6.4s, v2.4s          \n"
                    // save top_f32
                    "st1    {v5.4s, v6.4s}, [%2], #32    \n"
                    "subs   %w0, %w0, #1                 \n"
                    "bne    0b                           \n"
                    : "=r"(nn),         // %0
                      "=r"(intptr),     // %1
                      "=r"(ptr)         // %2
                    : "0"(nn),
                      "1"(intptr),
                      "2"(ptr),
                      "r"(_scale)       // %6
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6"
                );
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%1, #256]          \n"
                    "vld1.s32   {d0-d3}, [%1]!      \n" //q0-q1 data
                    "vdup.f32   q10, %6             \n" //q10 scale

                    "0:                             \n"
                    "vcvt.f32.s32 q0, q0            \n"
                    "vcvt.f32.s32 q1, q1            \n"

                    "vmul.f32   q2,q0,q10           \n"
                    "vmul.f32   q3,q1,q10           \n"

                    "pld        [%1, #256]          \n"
                    "vld1.s32   {d0-d3}, [%1]!      \n"
                    "vst1.f32   {d4-d7}, [%2]!      \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    "sub        %1, #32             \n"
                    : "=r"(nn),         // %0
                      "=r"(intptr),     // %1
                      "=r"(ptr)         // %2
                    : "0"(nn),
                      "1"(intptr),
                      "2"(ptr),
                      "r"(scale)        // %6
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q10", "q12"
                );              
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    *ptr = *intptr * scale;

                    intptr++;
                    ptr++;
                }
            }
        }   
    }

    return 0;
}

} // namespace ncnn

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

#include "dequantize_arm.h"

namespace ncnn {

Dequantize_arm::Dequantize_arm()
{
#if __ARM_NEON
    support_packing = true;
// #if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
//     support_fp16_storage = true;
// #endif
#endif // __ARM_NEON

    //     support_bf16_storage = true;
}

int Dequantize_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    //     int elembits = bottom_blob.elembits();

    // #if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    //     if (opt.use_fp16_storage && elembits == 16)
    //     {
    //         if (opt.use_fp16_arithmetic)
    //             return forward_fp16sa(bottom_blob, top_blob, opt);
    //         else
    //             return forward_fp16s(bottom_blob, top_blob, opt);
    //     }
    // #endif

    //     if (opt.use_bf16_storage && elembits == 16)
    //         return forward_bf16s(bottom_blob, top_blob, opt);

    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;

            top_blob.create(w, (size_t)16u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_term)
            {
                if (bias_data_size > 1)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        ptr[0] = intptr[0] * scale + bias_data[0];
                        ptr[1] = intptr[1] * scale + bias_data[1];
                        ptr[2] = intptr[2] * scale + bias_data[2];
                        ptr[3] = intptr[3] * scale + bias_data[3];
                    }
                }
                else
                {
                    float bias = bias_data[0];

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        ptr[0] = intptr[0] * scale + bias;
                        ptr[1] = intptr[1] * scale + bias;
                        ptr[2] = intptr[2] * scale + bias;
                        ptr[3] = intptr[3] * scale + bias;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const int* intptr = (const int*)bottom_blob + i * 4;
                    float* ptr = (float*)top_blob + i * 4;

                    ptr[0] = intptr[0] * scale;
                    ptr[1] = intptr[1] * scale;
                    ptr[2] = intptr[2] * scale;
                    ptr[3] = intptr[3] * scale;
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;

            top_blob.create(w, h, (size_t)16u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr = top_blob.row(i);

                    float bias = bias_data_size > 1 ? bias_data[i] : bias_data[0];

                    for (int j = 0; j < w; j++)
                    {
                        ptr[0] = intptr[0] * scale + bias;
                        ptr[1] = intptr[1] * scale + bias;
                        ptr[2] = intptr[2] * scale + bias;
                        ptr[3] = intptr[3] * scale + bias;

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr = top_blob.row(i);

                    for (int j = 0; j < w; j++)
                    {
                        ptr[0] = intptr[0] * scale;
                        ptr[1] = intptr[1] * scale;
                        ptr[2] = intptr[2] * scale;
                        ptr[3] = intptr[3] * scale;

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            int size = w * h;

            top_blob.create(w, h, channels, (size_t)16u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr = top_blob.channel(q);

                    float bias = bias_data[q];

                    for (int i = 0; i < size; i++)
                    {
                        ptr[0] = intptr[0] * scale + bias;
                        ptr[1] = intptr[1] * scale + bias;
                        ptr[2] = intptr[2] * scale + bias;
                        ptr[3] = intptr[3] * scale + bias;

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        ptr[0] = intptr[0] * scale;
                        ptr[1] = intptr[1] * scale;
                        ptr[2] = intptr[2] * scale;
                        ptr[3] = intptr[3] * scale;

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    if (dims == 1)
    {
        int w = bottom_blob.w;

        top_blob.create(w, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int* intptr = bottom_blob;
        float* ptr = top_blob;

        if (bias_term)
        {
            if (bias_data_size > 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale + bias_data[i];
                }
            }
            else
            {
                float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale + bias;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                ptr[i] = intptr[i] * scale;
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                float* ptr = top_blob.row(i);

                float bias = bias_data_size > 1 ? bias_data[i] : bias_data[0];

                for (int j = 0; j < w; j++)
                {
                    ptr[j] = intptr[j] * scale + bias;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                float* ptr = top_blob.row(i);

                for (int j = 0; j < w; j++)
                {
                    ptr[j] = intptr[j] * scale;
                }
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        top_blob.create(w, h, channels, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                float* ptr = top_blob.channel(q);

                float bias = bias_data[q];

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
                        : "=r"(nn),     // %0
                        "=r"(intptr), // %1
                        "=r"(ptr)     // %2
                        : "0"(nn),
                        "1"(intptr),
                        "2"(ptr),
                        "r"(scale), // %6
                        "r"(bias)   // %7
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6");
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
                        : "=r"(nn),     // %0
                        "=r"(intptr), // %1
                        "=r"(ptr)     // %2
                        : "0"(nn),
                        "1"(intptr),
                        "2"(ptr),
                        "r"(scale), // %6
                        "r"(bias)   // %7
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q10", "q12");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
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
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                float* ptr = top_blob.channel(q);

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
                        : "=r"(nn),     // %0
                        "=r"(intptr), // %1
                        "=r"(ptr)     // %2
                        : "0"(nn),
                        "1"(intptr),
                        "2"(ptr),
                        "r"(scale) // %6
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6");
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
                        : "=r"(nn),     // %0
                        "=r"(intptr), // %1
                        "=r"(ptr)     // %2
                        : "0"(nn),
                        "1"(intptr),
                        "2"(ptr),
                        "r"(scale) // %6
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q10", "q12");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
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

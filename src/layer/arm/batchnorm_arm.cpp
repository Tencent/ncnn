// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "batchnorm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

BatchNorm_arm::BatchNorm_arm()
{
#if __ARM_NEON
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int BatchNorm_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_inplace_fp16sa(bottom_top_blob, opt);
        else
            return forward_inplace_fp16s(bottom_top_blob, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                float* ptr = (float*)bottom_top_blob + i * 4;

                float32x4_t _a = vld1q_f32((const float*)a_data + i * 4);
                float32x4_t _b = vld1q_f32((const float*)b_data + i * 4);

                float32x4_t _p = vld1q_f32(ptr);
                _p = vmlaq_f32(_a, _p, _b);
                vst1q_f32(ptr, _p);
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float32x4_t _a = vld1q_f32((const float*)a_data + i * 4);
                float32x4_t _b = vld1q_f32((const float*)b_data + i * 4);

                float* ptr = bottom_top_blob.row(i);

                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    _p = vmlaq_f32(_a, _p, _b);
                    vst1q_f32(ptr, _p);

                    ptr += 4;
                }
            }
        }

        if (dims == 3 || dims == 4)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int d = bottom_top_blob.d;
            int c = bottom_top_blob.c;
            int size = w * h * d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < c; q++)
            {
                float32x4_t _a = vld1q_f32((const float*)a_data + q * 4);
                float32x4_t _b = vld1q_f32((const float*)b_data + q * 4);

                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    _p = vmlaq_f32(_a, _p, _b);
                    vst1q_f32(ptr, _p);

                    ptr += 4;
                }
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < w; i++)
        {
            ptr[i] = b_data[i] * ptr[i] + a_data[i];
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            float a = a_data[i];
            float b = b_data[i];

            int j = 0;
#if __ARM_NEON
            float32x4_t _a = vdupq_n_f32(a);
            float32x4_t _b = vdupq_n_f32(b);

            for (; j + 3 < w; j += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                _p = vmlaq_f32(_a, _p, _b);
                vst1q_f32(ptr, _p);

                ptr += 4;
            }
#endif // __ARM_NEON
            for (; j < w; j++)
            {
                *ptr = b * *ptr + a;

                ptr++;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int c = bottom_top_blob.c;
        int size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float a = a_data[q];
            float b = b_data[q];

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
                asm volatile(
                    "dup        v1.4s, %w4             \n"
                    "dup        v2.4s, %w5             \n"
                    "0:                                \n"
                    "prfm       pldl1keep, [%1, #128]  \n"
                    "ld1        {v0.4s}, [%1]          \n"
                    "orr        v3.16b, v1.16b, v1.16b \n"
                    "fmla       v3.4s, v0.4s, v2.4s    \n"
                    "subs       %w0, %w0, #1           \n"
                    "st1        {v3.4s}, [%1], #16     \n"
                    "bne        0b                     \n"
                    : "=r"(nn), // %0
                    "=r"(ptr) // %1
                    : "0"(nn),
                    "1"(ptr),
                    "r"(a), // %4
                    "r"(b)  // %5
                    : "cc", "memory", "v0", "v1", "v2", "v3");
            }
#else
            if (nn > 0)
            {
                asm volatile(
                    "vdup.f32   q1, %4              \n"
                    "vdup.f32   q2, %5              \n"
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]  \n"
                    "vorr.32    q3, q1, q1          \n"
                    "vmla.f32   q3, q0, q2          \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d6-d7}, [%1 :128]! \n"
                    "bne        0b                  \n"
                    : "=r"(nn), // %0
                    "=r"(ptr) // %1
                    : "0"(nn),
                    "1"(ptr),
                    "r"(a), // %4
                    "r"(b)  // %5
                    : "cc", "memory", "q0", "q1", "q2", "q3");
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain > 0; remain--)
            {
                *ptr = b * *ptr + a;

                ptr++;
            }
        }
    }

    return 0;
}

#if NCNN_BF16
int BatchNorm_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                unsigned short* ptr = (unsigned short*)bottom_top_blob + i * 4;

                float32x4_t _a = vld1q_f32((const float*)a_data + i * 4);
                float32x4_t _b = vld1q_f32((const float*)b_data + i * 4);

                float32x4_t _p = float2bfloat(vld1_u16(ptr));
                _p = vmlaq_f32(_a, _p, _b);
                vst1_u16(ptr, bfloat2float(_p));
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float32x4_t _a = vld1q_f32((const float*)a_data + i * 4);
                float32x4_t _b = vld1q_f32((const float*)b_data + i * 4);

                unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);

                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = float2bfloat(vld1_u16(ptr));
                    _p = vmlaq_f32(_a, _p, _b);
                    vst1_u16(ptr, bfloat2float(_p));

                    ptr += 4;
                }
            }
        }

        if (dims == 3 || dims == 4)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int d = bottom_top_blob.d;
            int c = bottom_top_blob.c;
            int size = w * h * d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < c; q++)
            {
                float32x4_t _a = vld1q_f32((const float*)a_data + q * 4);
                float32x4_t _b = vld1q_f32((const float*)b_data + q * 4);

                unsigned short* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = float2bfloat(vld1_u16(ptr));
                    _p = vmlaq_f32(_a, _p, _b);
                    vst1_u16(ptr, bfloat2float(_p));

                    ptr += 4;
                }
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        unsigned short* ptr = bottom_top_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < w; i++)
        {
            ptr[i] = float32_to_bfloat16(b_data[i] * bfloat16_to_float32(ptr[i]) + a_data[i]);
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);

            float a = a_data[i];
            float b = b_data[i];

            int j = 0;
#if __ARM_NEON
            float32x4_t _a = vdupq_n_f32(a);
            float32x4_t _b = vdupq_n_f32(b);

            for (; j + 3 < w; j += 4)
            {
                float32x4_t _p = float2bfloat(vld1_u16(ptr));
                _p = vmlaq_f32(_a, _p, _b);
                vst1_u16(ptr, bfloat2float(_p));

                ptr += 4;
            }
#endif // __ARM_NEON
            for (; j < w; j++)
            {
                *ptr = float32_to_bfloat16(b * bfloat16_to_float32(*ptr) + a);

                ptr++;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int c = bottom_top_blob.c;
        int size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            float a = a_data[q];
            float b = b_data[q];

            int j = 0;
#if __ARM_NEON
            float32x4_t _a = vdupq_n_f32(a);
            float32x4_t _b = vdupq_n_f32(b);

            for (; j + 3 < size; j += 4)
            {
                float32x4_t _p = float2bfloat(vld1_u16(ptr));
                _p = vmlaq_f32(_a, _p, _b);
                vst1_u16(ptr, bfloat2float(_p));

                ptr += 4;
            }
#endif // __ARM_NEON
            for (; j < size; j++)
            {
                *ptr = float32_to_bfloat16(b * bfloat16_to_float32(*ptr) + a);

                ptr++;
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn

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

#include "clip_arm.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

Clip_arm::Clip_arm()
{
#if __ARM_NEON
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

    support_bf16_storage = true;
}

int Clip_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
        return forward_inplace_fp16s(bottom_top_blob, opt);
#endif

    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int elempack = bottom_top_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float32x4_t _max = vdupq_n_f32(max);
            float32x4_t _min = vdupq_n_f32(min);

            for (int i = 0; i < size; i++)
            {
                float32x4_t _ptr = vld1q_f32(ptr);
                _ptr = vmaxq_f32(_ptr, _min);
                _ptr = vminq_f32(_ptr, _max);
                vst1q_f32(ptr, _ptr);

                ptr += 4;
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

#if __ARM_NEON
        int nn = size >> 2;
        int remain = size & 3;
#else
        int remain = size;
#endif

#if __ARM_NEON
        float32x4_t _max = vdupq_n_f32(max);
        float32x4_t _min = vdupq_n_f32(min);
#if __aarch64__
        for (; nn > 0; nn--)
        {
            float32x4_t _ptr = vld1q_f32(ptr);
            _ptr = vmaxq_f32(_ptr, _min);
            _ptr = vminq_f32(_ptr, _max);
            vst1q_f32(ptr, _ptr);
            ptr += 4;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.f32   {d0-d1}, [%1: 128]  \n"

                "vmax.f32   q0, q0, %q4         \n"
                "vmin.f32   q0, q0, %q5         \n"

                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%1: 128]! \n"

                "bne        0b                  \n"

                : "=r"(nn), // %0
                "=r"(ptr) // %1
                : "0"(nn),
                "1"(ptr),
                "w"(_min), // %q4
                "w"(_max)  // %q5
                : "cc", "memory", "q0");
        }
#endif // __aarch64__
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            if (*ptr < min)
                *ptr = min;

            if (*ptr > max)
                *ptr = max;

            ptr++;
        }
    }

    return 0;
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int Clip_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int elempack = bottom_top_blob.elempack;

    if (elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            float16x8_t _max = vdupq_n_f16(max);
            float16x8_t _min = vdupq_n_f16(min);

            for (int i = 0; i < size; i++)
            {
                float16x8_t _ptr = vld1q_f16(ptr);
                _ptr = vmaxq_f16(_ptr, _min);
                _ptr = vminq_f16(_ptr, _max);
                vst1q_f16(ptr, _ptr);

                ptr += 8;
            }
        }

        return 0;
    }

    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            float16x8_t _max = vdupq_n_f16(max);
            float16x8_t _min = vdupq_n_f16(min);

            int i = 0;
            for (; i + 1 < size; i += 2)
            {
                float16x8_t _ptr = vld1q_f16(ptr);
                _ptr = vmaxq_f16(_ptr, _min);
                _ptr = vminq_f16(_ptr, _max);
                vst1q_f16(ptr, _ptr);

                ptr += 8;
            }
            for (; i < size; i++)
            {
                float16x4_t _ptr = vld1_f16(ptr);
                _ptr = vmax_f16(_ptr, vget_low_f16(_min));
                _ptr = vmin_f16(_ptr, vget_low_f16(_max));
                vst1_f16(ptr, _ptr);

                ptr += 4;
            }
        }

        return 0;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

        int i = 0;

        float16x8_t _max = vdupq_n_f16(max);
        float16x8_t _min = vdupq_n_f16(min);

        for (; i + 7 < size; i += 8)
        {
            float16x8_t _ptr = vld1q_f16(ptr);
            _ptr = vmaxq_f16(_ptr, _min);
            _ptr = vminq_f16(_ptr, _max);
            vst1q_f16(ptr, _ptr);

            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _ptr = vld1_f16(ptr);
            _ptr = vmax_f16(_ptr, vget_low_f16(_min));
            _ptr = vmin_f16(_ptr, vget_low_f16(_max));
            vst1_f16(ptr, _ptr);

            ptr += 4;
        }

        __fp16 min_fp16 = min;
        __fp16 max_fp16 = max;

        for (; i < size; i++)
        {
            __fp16 v = *ptr;
            if (v < min_fp16)
                v = min_fp16;

            if (v > max_fp16)
                v = max_fp16;

            *ptr = v;
            ptr++;
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

int Clip_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int elempack = bottom_top_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            float32x4_t _max = vdupq_n_f32(max);
            float32x4_t _min = vdupq_n_f32(min);

            for (int i = 0; i < size; i++)
            {
                float32x4_t _ptr = vcvt_f32_bf16(vld1_u16(ptr));
                _ptr = vmaxq_f32(_ptr, _min);
                _ptr = vminq_f32(_ptr, _max);
                vst1_u16(ptr, vcvt_bf16_f32(_ptr));

                ptr += 4;
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

#if __ARM_NEON
        int nn = size >> 2;
        int remain = size & 3;
#else
        int remain = size;
#endif

#if __ARM_NEON
        float32x4_t _max = vdupq_n_f32(max);
        float32x4_t _min = vdupq_n_f32(min);
        for (; nn > 0; nn--)
        {
            float32x4_t _ptr = vcvt_f32_bf16(vld1_u16(ptr));
            _ptr = vmaxq_f32(_ptr, _min);
            _ptr = vminq_f32(_ptr, _max);
            vst1_u16(ptr, vcvt_bf16_f32(_ptr));
            ptr += 4;
        }
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v < min)
                v = min;

            if (v > max)
                v = max;

            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn

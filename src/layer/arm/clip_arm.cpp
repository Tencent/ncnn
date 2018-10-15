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

DEFINE_LAYER_CREATOR(Clip_arm)

int Clip_arm::forward_inplace(Mat &bottom_top_blob, const Option &opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

#if __ARM_NEON
    int nn = size >> 2;
    int remain = size - (nn << 2);
#else
    int remian = size;
#endif

#if __ARM_NEON
    float32x4_t maxf32 = vmovq_n_f32(max);
    float32x4_t minf32 = vmovq_n_f32(min);
#endif

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < channels; ++i)
    {
        float *channel_ptr = bottom_top_blob.channel(i);
#if __ARM_NEON
#if __aarch64__
        for (; nn > 0; --nn)
        {
            float32x4_t clip_f32 = vld1q_f32(channel_ptr);
            float32x4_t clip_min_f32 = vmaxq_f32(minf32, clip_f32);
            float32x4_t clip_max_f32 = vminq_f32(maxf32, clip_min_f32);
            vst1q_f32(channel_ptr, clip_max_f32);
            channel_ptr += 4;
        }
#else
        if (nn > 0)
        {
            asm volatile(
            "0:"
            "pld        [%1,    #128]           \n"
            "vld1.f32   {d0-d1},    [%1:128]    \n"

            "vmax.f32   q1, %q4,    q0          \n"
            "vmin.f32   q2, %q5,    q1          \n"

            "subs       %0,          #1          \n"
            "vst1.f32   {d4-d5},    [%1:128]!   \n"

            "bne        0b                      \n"

            :"=r"(nn),              //%0
            "=r"(channel_ptr)       //%1
            :"0"(nn),
            "1"(channel_ptr),
            "w"(minf32),            //%q4
            "w"(maxf32)             //%q5
            :"cc", "memory", "q0", "q1", "q2"
            );

        }
#endif // __aarch64__
#endif // __ARM_NEON

        for (; remain > 0; --remain)
        {

            if (*channel_ptr < min)
            {
                *channel_ptr = min;
            }

            if (*channel_ptr > max)
            {
                *channel_ptr = max;
            }

            ++channel_ptr;
        }

    }

    return 0;
}

} // namespace ncnn

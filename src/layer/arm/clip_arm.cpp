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

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
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
        for (; nn>0; nn--)
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

            : "=r"(nn),     // %0
              "=r"(ptr)     // %1
            : "0"(nn),
              "1"(ptr),
              "w"(_min),    // %q4
              "w"(_max)     // %q5
            : "cc", "memory", "q0"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON

        for (; remain>0; remain--)
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

} // namespace ncnn

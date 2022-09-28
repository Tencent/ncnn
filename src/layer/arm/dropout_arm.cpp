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

#include "dropout_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

Dropout_arm::Dropout_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

int Dropout_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (scale == 1.f)
    {
        return 0;
    }

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __ARM_NEON
        float32x4_t _scale = vdupq_n_f32(scale);
        for (; i + 15 < size; i += 16)
        {
#if __aarch64__
            asm volatile(
                "prfm   pldl1keep, [%0, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"
                "fmul   v0.4s, v0.4s, %2.4s     \n"
                "fmul   v1.4s, v1.4s, %2.4s     \n"
                "fmul   v2.4s, v2.4s, %2.4s     \n"
                "fmul   v3.4s, v3.4s, %2.4s     \n"
                "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                : "=r"(ptr) // %0
                : "0"(ptr),
                "w"(_scale) // %2
                : "memory", "v0", "v1", "v2", "v3");
#else  // __aarch64__
            asm volatile(
                "pld        [%0, #512]      \n"
                "vldm       %0, {d0-d7}     \n"
                "vmul.f32   q0, q0, %q2     \n"
                "vmul.f32   q1, q1, %q2     \n"
                "vmul.f32   q2, q2, %q2     \n"
                "vmul.f32   q3, q3, %q2     \n"
                "vstm       %0!, {d0-d7}    \n"
                : "=r"(ptr) // %0
                : "0"(ptr),
                "w"(_scale) // %2
                : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
        }
        for (; i + 7 < size; i += 8)
        {
            float32x4_t _p0 = vld1q_f32(ptr);
            float32x4_t _p1 = vld1q_f32(ptr + 4);
            _p0 = vmulq_f32(_p0, _scale);
            _p1 = vmulq_f32(_p1, _scale);
            vst1q_f32(ptr, _p0);
            vst1q_f32(ptr + 4, _p1);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = vmulq_f32(_p, _scale);
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            *ptr = *ptr * scale;

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn

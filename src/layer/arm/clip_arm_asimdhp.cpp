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

#include "clip_arm.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int Clip_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

        __fp16 min_fp16 = min;
        __fp16 max_fp16 = max;

        float16x8_t _min = vdupq_n_f16(min_fp16);
        float16x8_t _max = vdupq_n_f16(max_fp16);

        int i = 0;
        for (; i + 31 < size; i += 32)
        {
#if NCNN_GNU_INLINE_ASM
            asm volatile(
                "prfm   pldl1keep, [%0, #512]   \n"
                "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0] \n"
                "fmax   v0.8h, v0.8h, %2.8h     \n"
                "fmax   v1.8h, v1.8h, %2.8h     \n"
                "fmax   v2.8h, v2.8h, %2.8h     \n"
                "fmax   v3.8h, v3.8h, %2.8h     \n"
                "fmin   v0.8h, v0.8h, %3.8h     \n"
                "fmin   v1.8h, v1.8h, %3.8h     \n"
                "fmin   v2.8h, v2.8h, %3.8h     \n"
                "fmin   v3.8h, v3.8h, %3.8h     \n"
                "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
                : "=r"(ptr) // %0
                : "0"(ptr),
                "w"(_min), // %2
                "w"(_max)  // %3
                : "memory", "v0", "v1", "v2", "v3");
#else  // NCNN_GNU_INLINE_ASM
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x8_t _p2 = vld1q_f16(ptr + 16);
            float16x8_t _p3 = vld1q_f16(ptr + 24);
            _p0 = vmaxq_f16(_p0, _min);
            _p1 = vmaxq_f16(_p1, _min);
            _p2 = vmaxq_f16(_p2, _min);
            _p3 = vmaxq_f16(_p3, _min);
            _p0 = vminq_f16(_p0, _max);
            _p1 = vminq_f16(_p1, _max);
            _p2 = vminq_f16(_p2, _max);
            _p3 = vminq_f16(_p3, _max);
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            vst1q_f16(ptr + 16, _p2);
            vst1q_f16(ptr + 24, _p3);
            ptr += 32;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; i + 15 < size; i += 16)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            _p0 = vmaxq_f16(_p0, _min);
            _p1 = vmaxq_f16(_p1, _min);
            _p0 = vminq_f16(_p0, _max);
            _p1 = vminq_f16(_p1, _max);
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _p = vmaxq_f16(_p, _min);
            _p = vminq_f16(_p, _max);
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = vmax_f16(_p, vget_low_f16(_min));
            _p = vmin_f16(_p, vget_low_f16(_max));
            vst1_f16(ptr, _p);
            ptr += 4;
        }
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

} // namespace ncnn

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

#include "relu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int ReLU_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            float16x8_t _zero = vdupq_n_f16((__fp16)0.f);

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
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
                    : "=r"(ptr) // %0
                    : "0"(ptr),
                    "w"(_zero) // %2
                    : "memory", "v0", "v1", "v2", "v3");
#else  // NCNN_GNU_INLINE_ASM
                float16x8_t _p0 = vld1q_f16(ptr);
                float16x8_t _p1 = vld1q_f16(ptr + 8);
                float16x8_t _p2 = vld1q_f16(ptr + 16);
                float16x8_t _p3 = vld1q_f16(ptr + 24);
                _p0 = vmaxq_f16(_p0, _zero);
                _p1 = vmaxq_f16(_p1, _zero);
                _p2 = vmaxq_f16(_p2, _zero);
                _p3 = vmaxq_f16(_p3, _zero);
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
                _p0 = vmaxq_f16(_p0, _zero);
                _p1 = vmaxq_f16(_p1, _zero);
                vst1q_f16(ptr, _p0);
                vst1q_f16(ptr + 8, _p1);
                ptr += 16;
            }
            for (; i + 7 < size; i += 8)
            {
                float16x8_t _p = vld1q_f16(ptr);
                _p = vmaxq_f16(_p, _zero);
                vst1q_f16(ptr, _p);
                ptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float16x4_t _p = vld1_f16(ptr);
                _p = vmax_f16(_p, vget_low_f16(_zero));
                vst1_f16(ptr, _p);
                ptr += 4;
            }
            for (; i < size; i++)
            {
                __fp16 v = ptr[0];
                if (v < (__fp16)0.f)
                    ptr[0] = (__fp16)0.f;

                ptr += 1;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            float16x8_t _zero = vdupq_n_f16((__fp16)0.f);
            float16x8_t _slope = vdupq_n_f16((__fp16)slope);

            int i = 0;
            for (; i + 31 < size; i += 32)
            {
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0] \n"
                    "fcmle  v4.8h, v0.8h, #0        \n"
                    "fcmle  v5.8h, v1.8h, #0        \n"
                    "fcmle  v6.8h, v2.8h, #0        \n"
                    "fcmle  v7.8h, v3.8h, #0        \n"
                    "fmul   v8.8h, v0.8h, %2.8h     \n"
                    "fmul   v9.8h, v1.8h, %2.8h     \n"
                    "fmul   v10.8h, v2.8h, %2.8h    \n"
                    "fmul   v11.8h, v3.8h, %2.8h    \n"
                    "bit    v0.16b, v8.16b, v4.16b  \n"
                    "bit    v1.16b, v9.16b, v5.16b  \n"
                    "bit    v2.16b, v10.16b, v6.16b \n"
                    "bit    v3.16b, v11.16b, v7.16b \n"
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
                    : "=r"(ptr) // %0
                    : "0"(ptr),
                    "w"(_slope) // %2
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else  // NCNN_GNU_INLINE_ASM
                float16x8_t _p0 = vld1q_f16(ptr);
                float16x8_t _p1 = vld1q_f16(ptr + 8);
                float16x8_t _p2 = vld1q_f16(ptr + 16);
                float16x8_t _p3 = vld1q_f16(ptr + 24);
                uint16x8_t _lemask0 = vcleq_f16(_p0, _zero);
                uint16x8_t _lemask1 = vcleq_f16(_p1, _zero);
                uint16x8_t _lemask2 = vcleq_f16(_p2, _zero);
                uint16x8_t _lemask3 = vcleq_f16(_p3, _zero);
                float16x8_t _ps0 = vmulq_f16(_p0, _slope);
                float16x8_t _ps1 = vmulq_f16(_p1, _slope);
                float16x8_t _ps2 = vmulq_f16(_p2, _slope);
                float16x8_t _ps3 = vmulq_f16(_p3, _slope);
                _p0 = vbslq_f16(_lemask0, _ps0, _p0);
                _p1 = vbslq_f16(_lemask1, _ps1, _p1);
                _p2 = vbslq_f16(_lemask2, _ps2, _p2);
                _p3 = vbslq_f16(_lemask3, _ps3, _p3);
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
                uint16x8_t _lemask0 = vcleq_f16(_p0, _zero);
                uint16x8_t _lemask1 = vcleq_f16(_p1, _zero);
                float16x8_t _ps0 = vmulq_f16(_p0, _slope);
                float16x8_t _ps1 = vmulq_f16(_p1, _slope);
                _p0 = vbslq_f16(_lemask0, _ps0, _p0);
                _p1 = vbslq_f16(_lemask1, _ps1, _p1);
                vst1q_f16(ptr, _p0);
                vst1q_f16(ptr + 8, _p1);
                ptr += 16;
            }
            for (; i + 7 < size; i += 8)
            {
                float16x8_t _p = vld1q_f16(ptr);
                uint16x8_t _lemask = vcleq_f16(_p, _zero);
                float16x8_t _ps = vmulq_f16(_p, _slope);
                _p = vbslq_f16(_lemask, _ps, _p);
                vst1q_f16(ptr, _p);
                ptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float16x4_t _p = vld1_f16(ptr);
                uint16x4_t _lemask = vcle_f16(_p, vget_low_f16(_zero));
                float16x4_t _ps = vmul_f16(_p, vget_low_f16(_slope));
                _p = vbsl_f16(_lemask, _ps, _p);
                vst1_f16(ptr, _p);
                ptr += 4;
            }
            for (; i < size; i++)
            {
                __fp16 v = ptr[0];
                if (v < (__fp16)0.f)
                    ptr[0] = v * (__fp16)slope;

                ptr += 1;
            }
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn

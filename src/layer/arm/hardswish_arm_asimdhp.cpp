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

#include "hardswish_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int HardSwish_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
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

        float32x4_t _zero = vdupq_n_f32(0.f);
        float32x4_t _one = vdupq_n_f32(1.f);
        float32x4_t _alpha = vdupq_n_f32(alpha);
        float32x4_t _beta = vdupq_n_f32(beta);

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
            float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
            float32x4_t _ans0 = vfmaq_f32(_beta, _p0, _alpha);
            float32x4_t _ans1 = vfmaq_f32(_beta, _p1, _alpha);
            _ans0 = vmaxq_f32(_ans0, _zero);
            _ans1 = vmaxq_f32(_ans1, _zero);
            _ans0 = vminq_f32(_ans0, _one);
            _ans1 = vminq_f32(_ans1, _one);
            _p0 = vmulq_f32(_ans0, _p0);
            _p1 = vmulq_f32(_ans1, _p1);
            _p = vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1));
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
            float32x4_t _ans = vfmaq_f32(_beta, _p, _alpha);
            _ans = vmaxq_f32(_ans, _zero);
            _ans = vminq_f32(_ans, _one);
            _p = vmulq_f32(_ans, _p);
            vst1_f16(ptr, vcvt_f16_f32(_p));
            ptr += 4;
        }
        for (; i < size; i++)
        {
            float v = (float)*ptr;
            if (v < lower)
                v = 0.f;
            else if (v > upper)
                ;
            else
                v = v * (v * alpha + beta);
            *ptr = (__fp16)v;

            ptr++;
        }
    }

    return 0;
}

int HardSwish_arm::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
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

        __fp16 alpha_fp16 = (__fp16)alpha;
        __fp16 beta_fp16 = (__fp16)beta;

        float16x8_t _zero = vdupq_n_f16((__fp16)0.f);
        float16x8_t _one = vdupq_n_f16((__fp16)1.f);
        float16x8_t _alpha = vdupq_n_f16(alpha_fp16);
        float16x8_t _beta = vdupq_n_f16(beta_fp16);

        int i = 0;
        for (; i + 31 < size; i += 32)
        {
#if NCNN_GNU_INLINE_ASM
            asm volatile(
                "prfm   pldl1keep, [%0, #512]   \n"
                "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0] \n"
                "mov    v4.16b, %5.16b          \n"
                "mov    v5.16b, %5.16b          \n"
                "mov    v6.16b, %5.16b          \n"
                "mov    v7.16b, %5.16b          \n"
                "fmla   v4.8h, v0.8h, %4.8h     \n"
                "fmla   v5.8h, v1.8h, %4.8h     \n"
                "fmla   v6.8h, v2.8h, %4.8h     \n"
                "fmla   v7.8h, v3.8h, %4.8h     \n"
                "fmax   v4.8h, v4.8h, %2.8h     \n"
                "fmax   v5.8h, v5.8h, %2.8h     \n"
                "fmax   v6.8h, v6.8h, %2.8h     \n"
                "fmax   v7.8h, v7.8h, %2.8h     \n"
                "fmin   v4.8h, v4.8h, %3.8h     \n"
                "fmin   v5.8h, v5.8h, %3.8h     \n"
                "fmin   v6.8h, v6.8h, %3.8h     \n"
                "fmin   v7.8h, v7.8h, %3.8h     \n"
                "fmul   v0.8h, v4.8h, v0.8h     \n"
                "fmul   v1.8h, v5.8h, v1.8h     \n"
                "fmul   v2.8h, v6.8h, v2.8h     \n"
                "fmul   v3.8h, v7.8h, v3.8h     \n"
                "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
                : "=r"(ptr) // %0
                : "0"(ptr),
                "w"(_zero),  // %2
                "w"(_one),   // %3
                "w"(_alpha), // %4
                "w"(_beta)   // %5
                : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else  // NCNN_GNU_INLINE_ASM
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x8_t _p2 = vld1q_f16(ptr + 16);
            float16x8_t _p3 = vld1q_f16(ptr + 24);
            float16x8_t _ans0 = vfmaq_f16(_beta, _p0, _alpha);
            float16x8_t _ans1 = vfmaq_f16(_beta, _p1, _alpha);
            float16x8_t _ans2 = vfmaq_f16(_beta, _p2, _alpha);
            float16x8_t _ans3 = vfmaq_f16(_beta, _p3, _alpha);
            _ans0 = vmaxq_f16(_ans0, _zero);
            _ans1 = vmaxq_f16(_ans1, _zero);
            _ans2 = vmaxq_f16(_ans2, _zero);
            _ans3 = vmaxq_f16(_ans3, _zero);
            _ans0 = vminq_f16(_ans0, _one);
            _ans1 = vminq_f16(_ans1, _one);
            _ans2 = vminq_f16(_ans2, _one);
            _ans3 = vminq_f16(_ans3, _one);
            _p0 = vmulq_f16(_ans0, _p0);
            _p1 = vmulq_f16(_ans1, _p1);
            _p2 = vmulq_f16(_ans2, _p2);
            _p3 = vmulq_f16(_ans3, _p3);
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
            float16x8_t _ans0 = vfmaq_f16(_beta, _p0, _alpha);
            float16x8_t _ans1 = vfmaq_f16(_beta, _p1, _alpha);
            _ans0 = vmaxq_f16(_ans0, _zero);
            _ans1 = vmaxq_f16(_ans1, _zero);
            _ans0 = vminq_f16(_ans0, _one);
            _ans1 = vminq_f16(_ans1, _one);
            _p0 = vmulq_f16(_ans0, _p0);
            _p1 = vmulq_f16(_ans1, _p1);
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float16x8_t _ans = vfmaq_f16(_beta, _p, _alpha);
            _ans = vmaxq_f16(_ans, _zero);
            _ans = vminq_f16(_ans, _one);
            _p = vmulq_f16(_ans, _p);
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            float16x4_t _ans = vfma_f16(vget_low_f16(_beta), _p, vget_low_f16(_alpha));
            _ans = vmax_f16(_ans, vget_low_f16(_zero));
            _ans = vmin_f16(_ans, vget_low_f16(_one));
            _p = vmul_f16(_ans, _p);
            vst1_f16(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            __fp16 v = *ptr;
            if (v < (__fp16)lower)
                v = (__fp16)0.f;
            else if (v > (__fp16)upper)
                ;
            else
                v = v * (v * alpha_fp16 + beta_fp16);
            *ptr = v;

            ptr++;
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn

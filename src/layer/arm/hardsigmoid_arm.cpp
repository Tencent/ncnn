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

#include "hardsigmoid_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

HardSigmoid_arm::HardSigmoid_arm()
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

int HardSigmoid_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
        float32x4_t _zero = vdupq_n_f32(0.f);
        float32x4_t _one = vdupq_n_f32(1.f);
        float32x4_t _alpha = vdupq_n_f32(alpha);
        float32x4_t _beta = vdupq_n_f32(beta);
        for (; i + 15 < size; i += 16)
        {
#if __aarch64__
            asm volatile(
                "prfm   pldl1keep, [%0, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"
                "mov    v4.16b, %5.16b          \n"
                "mov    v5.16b, %5.16b          \n"
                "mov    v6.16b, %5.16b          \n"
                "mov    v7.16b, %5.16b          \n"
                "fmla   v4.4s, v0.4s, %4.4s     \n"
                "fmla   v5.4s, v1.4s, %4.4s     \n"
                "fmla   v6.4s, v2.4s, %4.4s     \n"
                "fmla   v7.4s, v3.4s, %4.4s     \n"
                "fmax   v0.4s, v4.4s, %2.4s     \n"
                "fmax   v1.4s, v5.4s, %2.4s     \n"
                "fmax   v2.4s, v6.4s, %2.4s     \n"
                "fmax   v3.4s, v7.4s, %2.4s     \n"
                "fmin   v0.4s, v0.4s, %3.4s     \n"
                "fmin   v1.4s, v1.4s, %3.4s     \n"
                "fmin   v2.4s, v2.4s, %3.4s     \n"
                "fmin   v3.4s, v3.4s, %3.4s     \n"
                "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                : "=r"(ptr) // %0
                : "0"(ptr),
                "w"(_zero),  // %2
                "w"(_one),   // %3
                "w"(_alpha), // %4
                "w"(_beta)   // %5
                : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else  // __aarch64__
            asm volatile(
                "pld        [%0, #512]      \n"
                "vldm       %0, {d0-d7}     \n"
                "vmov       q4, %q5         \n"
                "vmov       q5, %q5         \n"
                "vmov       q6, %q5         \n"
                "vmov       q7, %q5         \n"
                "vmla.f32   q4, q0, %q4     \n"
                "vmla.f32   q5, q1, %q4     \n"
                "vmla.f32   q6, q2, %q4     \n"
                "vmla.f32   q7, q3, %q4     \n"
                "vmax.f32   q0, q4, %q2     \n"
                "vmax.f32   q1, q5, %q2     \n"
                "vmax.f32   q2, q6, %q2     \n"
                "vmax.f32   q3, q7, %q2     \n"
                "vmin.f32   q0, q0, %q3     \n"
                "vmin.f32   q1, q1, %q3     \n"
                "vmin.f32   q2, q2, %q3     \n"
                "vmin.f32   q3, q3, %q3     \n"
                "vstm       %0!, {d0-d7}    \n"
                : "=r"(ptr) // %0
                : "0"(ptr),
                "w"(_zero),  // %2
                "w"(_one),   // %3
                "w"(_alpha), // %4
                "w"(_beta)   // %5
                : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
#endif // __aarch64__
        }
        for (; i + 7 < size; i += 8)
        {
            float32x4_t _p0 = vld1q_f32(ptr);
            float32x4_t _p1 = vld1q_f32(ptr + 4);
            _p0 = vmlaq_f32(_beta, _p0, _alpha);
            _p1 = vmlaq_f32(_beta, _p1, _alpha);
            _p0 = vmaxq_f32(_p0, _zero);
            _p1 = vmaxq_f32(_p1, _zero);
            _p0 = vminq_f32(_p0, _one);
            _p1 = vminq_f32(_p1, _one);
            vst1q_f32(ptr, _p0);
            vst1q_f32(ptr + 4, _p1);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = vmlaq_f32(_beta, _p, _alpha);
            _p = vmaxq_f32(_p, _zero);
            _p = vminq_f32(_p, _one);
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            if (*ptr < lower)
                *ptr = 0.f;
            else if (*ptr > upper)
                *ptr = 1.f;
            else
                *ptr = *ptr * alpha + beta;

            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int HardSigmoid_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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
        unsigned short* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __ARM_NEON
        float32x4_t _zero = vdupq_n_f32(0.f);
        float32x4_t _one = vdupq_n_f32(1.f);
        float32x4_t _alpha = vdupq_n_f32(alpha);
        float32x4_t _beta = vdupq_n_f32(beta);
        for (; i + 15 < size; i += 16)
        {
#if __aarch64__
            asm volatile(
                "prfm   pldl1keep, [%0, #256]   \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%0] \n"
                "shll   v0.4s, v0.4h, #16       \n"
                "shll   v1.4s, v1.4h, #16       \n"
                "shll   v2.4s, v2.4h, #16       \n"
                "shll   v3.4s, v3.4h, #16       \n"
                "mov    v4.16b, %5.16b          \n"
                "mov    v5.16b, %5.16b          \n"
                "mov    v6.16b, %5.16b          \n"
                "mov    v7.16b, %5.16b          \n"
                "fmla   v4.4s, v0.4s, %4.4s     \n"
                "fmla   v5.4s, v1.4s, %4.4s     \n"
                "fmla   v6.4s, v2.4s, %4.4s     \n"
                "fmla   v7.4s, v3.4s, %4.4s     \n"
                "fmax   v0.4s, v4.4s, %2.4s     \n"
                "fmax   v1.4s, v5.4s, %2.4s     \n"
                "fmax   v2.4s, v6.4s, %2.4s     \n"
                "fmax   v3.4s, v7.4s, %2.4s     \n"
                "fmin   v0.4s, v0.4s, %3.4s     \n"
                "fmin   v1.4s, v1.4s, %3.4s     \n"
                "fmin   v2.4s, v2.4s, %3.4s     \n"
                "fmin   v3.4s, v3.4s, %3.4s     \n"
                "shrn   v0.4h, v0.4s, #16       \n"
                "shrn   v1.4h, v1.4s, #16       \n"
                "shrn   v2.4h, v2.4s, #16       \n"
                "shrn   v3.4h, v3.4s, #16       \n"
                "st1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%0], #32 \n"
                : "=r"(ptr) // %0
                : "0"(ptr),
                "w"(_zero),  // %2
                "w"(_one),   // %3
                "w"(_alpha), // %4
                "w"(_beta)   // %5
                : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else  // __aarch64__
            asm volatile(
                "pld        [%0, #256]      \n"
                "vld1.u16   {d4-d7}, [%0]   \n"
                "vshll.u16  q0, d4, #16     \n"
                "vshll.u16  q1, d5, #16     \n"
                "vshll.u16  q2, d6, #16     \n"
                "vshll.u16  q3, d7, #16     \n"
                "vmov       q4, %q5         \n"
                "vmov       q5, %q5         \n"
                "vmov       q6, %q5         \n"
                "vmov       q7, %q5         \n"
                "vmla.f32   q4, q0, %q4     \n"
                "vmla.f32   q5, q1, %q4     \n"
                "vmla.f32   q6, q2, %q4     \n"
                "vmla.f32   q7, q3, %q4     \n"
                "vmax.f32   q0, q4, %q2     \n"
                "vmax.f32   q1, q5, %q2     \n"
                "vmax.f32   q2, q6, %q2     \n"
                "vmax.f32   q3, q7, %q2     \n"
                "vmin.f32   q0, q0, %q3     \n"
                "vmin.f32   q1, q1, %q3     \n"
                "vmin.f32   q2, q2, %q3     \n"
                "vmin.f32   q3, q3, %q3     \n"
                "vshrn.u32  d0, q0, #16     \n"
                "vshrn.u32  d1, q1, #16     \n"
                "vshrn.u32  d2, q2, #16     \n"
                "vshrn.u32  d3, q3, #16     \n"
                "vst1.u16   {d0-d3}, [%0]!  \n"
                : "=r"(ptr) // %0
                : "0"(ptr),
                "w"(_zero),  // %2
                "w"(_one),   // %3
                "w"(_alpha), // %4
                "w"(_beta)   // %5
                : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
#endif // __aarch64__
        }
        for (; i + 7 < size; i += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            float32x4_t _p0 = float2bfloat(vget_low_u16(_p));
            float32x4_t _p1 = float2bfloat(vget_high_u16(_p));
            _p0 = vmlaq_f32(_beta, _p0, _alpha);
            _p1 = vmlaq_f32(_beta, _p1, _alpha);
            _p0 = vmaxq_f32(_p0, _zero);
            _p1 = vmaxq_f32(_p1, _zero);
            _p0 = vminq_f32(_p0, _one);
            _p1 = vminq_f32(_p1, _one);
            _p = vcombine_u16(bfloat2float(_p0), bfloat2float(_p1));
            vst1q_u16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = float2bfloat(vld1_u16(ptr));
            _p = vmlaq_f32(_beta, _p, _alpha);
            _p = vmaxq_f32(_p, _zero);
            _p = vminq_f32(_p, _one);
            vst1_u16(ptr, bfloat2float(_p));
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v < lower)
                v = 0.f;
            else if (v > upper)
                v = 1.f;
            else
                v = v * alpha + beta;
            *ptr = float32_to_bfloat16(v);

            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn

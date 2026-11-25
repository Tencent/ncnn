// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "dropout_arm.h"

#include "arm_16bit.h"
#include "cpu.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

#include "dropout_arm_16bit.h"

Dropout_arm::Dropout_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Dropout_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (scale == 1.f)
    {
        return 0;
    }

    const int elembits = bottom_top_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_inplace_fp16s(bottom_top_blob, opt);
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
        float32x4_t _scale = vdupq_n_f32(scale);
        for (; i + 15 < size; i += 16)
        {
#if NCNN_GNU_INLINE_ASM
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
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _p0 = vld1q_f32(ptr);
            float32x4_t _p1 = vld1q_f32(ptr + 4);
            float32x4_t _p2 = vld1q_f32(ptr + 8);
            float32x4_t _p3 = vld1q_f32(ptr + 12);
            _p0 = vmulq_f32(_p0, _scale);
            _p1 = vmulq_f32(_p1, _scale);
            _p2 = vmulq_f32(_p2, _scale);
            _p3 = vmulq_f32(_p3, _scale);
            vst1q_f32(ptr, _p0);
            vst1q_f32(ptr + 4, _p1);
            vst1q_f32(ptr + 8, _p2);
            vst1q_f32(ptr + 12, _p3);
            ptr += 16;
#endif // NCNN_GNU_INLINE_ASM
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

#if NCNN_BF16
int Dropout_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    return dropout_forward_inplace_16bit<ArmBF16Converter>(bottom_top_blob, scale, opt);
}
#endif // NCNN_BF16

} // namespace ncnn

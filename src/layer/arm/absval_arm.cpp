// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "absval_arm.h"

#include "cpu.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

AbsVal_arm::AbsVal_arm()
{
#if __ARM_NEON
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#if NCNN_VFPV4
    support_fp16_storage = support_fp16_storage || cpu_support_arm_vfpv4();
#endif
#endif // __ARM_NEON
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int AbsVal_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s_fp16s(bottom_top_blob, opt);

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
        for (; i + 15 < size; i += 16)
        {
#if NCNN_GNU_INLINE_ASM
#if __aarch64__
            asm volatile(
                "prfm   pldl1keep, [%0, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"
                "fabs   v0.4s, v0.4s            \n"
                "fabs   v1.4s, v1.4s            \n"
                "fabs   v2.4s, v2.4s            \n"
                "fabs   v3.4s, v3.4s            \n"
                "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                : "=r"(ptr) // %0
                : "0"(ptr)
                : "memory", "v0", "v1", "v2", "v3");
#else  // __aarch64__
            asm volatile(
                "pld        [%0, #512]      \n"
                "vldm       %0, {d0-d7}     \n"
                "vabs.f32   q0, q0          \n"
                "vabs.f32   q1, q1          \n"
                "vabs.f32   q2, q2          \n"
                "vabs.f32   q3, q3          \n"
                "vstm       %0!, {d0-d7}    \n"
                : "=r"(ptr) // %0
                : "0"(ptr)
                : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _p0 = vld1q_f32(ptr);
            float32x4_t _p1 = vld1q_f32(ptr + 4);
            float32x4_t _p2 = vld1q_f32(ptr + 8);
            float32x4_t _p3 = vld1q_f32(ptr + 12);
            _p0 = vabsq_f32(_p0);
            _p1 = vabsq_f32(_p1);
            _p2 = vabsq_f32(_p2);
            _p3 = vabsq_f32(_p3);
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
            _p0 = vabsq_f32(_p0);
            _p1 = vabsq_f32(_p1);
            vst1q_f32(ptr, _p0);
            vst1q_f32(ptr + 4, _p1);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = vabsq_f32(_p);
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            *ptr = *ptr > 0 ? *ptr : -*ptr;

            ptr++;
        }
    }

    return 0;
}

int AbsVal_arm::forward_inplace_bf16s_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int size = w * h * d * elempack;

    // fp16 and bf16 both store the sign bit at bit 15
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __ARM_NEON
        uint16x8_t _sign_mask = vdupq_n_u16(0x7fff);
        for (; i + 15 < size; i += 16)
        {
            uint16x8_t _p0 = vld1q_u16(ptr);
            uint16x8_t _p1 = vld1q_u16(ptr + 8);
            vst1q_u16(ptr, vandq_u16(_p0, _sign_mask));
            vst1q_u16(ptr + 8, vandq_u16(_p1, _sign_mask));
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            vst1q_u16(ptr, vandq_u16(_p, _sign_mask));
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            uint16x4_t _p = vld1_u16(ptr);
            vst1_u16(ptr, vand_u16(_p, vdup_n_u16(0x7fff)));
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            *ptr = *ptr & 0x7fffu;
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn

// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "bias_arm.h"

#include "cpu.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

Bias_arm::Bias_arm()
{
#if __ARM_NEON
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Bias_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
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
    int size = w * h * d;

    const float* bias_ptr = bias_data;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        float bias = bias_ptr[q];

#if __ARM_NEON
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
        float32x4_t _bias = vdupq_n_f32(bias);
        for (; nn > 0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _outp = vaddq_f32(_p, _bias);
            vst1q_f32(ptr, _outp);

            ptr += 4;
        }
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *ptr = *ptr + bias;

            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int Bias_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int size = w * h * d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

        const float bias = bias_data[q];

        int i = 0;
#if __ARM_NEON
        float32x4_t _bias = vdupq_n_f32(bias);
        for (; i + 7 < size; i += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
            float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
            vst1q_u16(ptr, vcombine_u16(float2bfloat(vaddq_f32(_p0, _bias)), float2bfloat(vaddq_f32(_p1, _bias))));
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            vst1_u16(ptr, float2bfloat(vaddq_f32(_p, _bias)));
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) + bias);
            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn

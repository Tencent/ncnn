// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gelu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

GELU_arm::GELU_arm()
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

int GELU_arm::create_pipeline(const Option& /*opt*/)
{
    return 0;
}

int GELU_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
    int elempack = bottom_top_blob.elempack;
    int channels = bottom_top_blob.c;
    int size = w * h * d * elempack;

    if (fast_gelu)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;

#if __ARM_NEON
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _pLoad = vld1q_f32(ptr);
                float32x4_t _blob = fast_gelu_ps(_pLoad);
                vst1q_f32(ptr, _blob);
                ptr += 4;
            }
#endif
            for (; i < size; i++)
            {
                *ptr = 0.5f * *ptr * (1.0f + tanhf(0.79788452f * (*ptr + 0.044715f * *ptr * *ptr * *ptr)));
                ptr++;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;

#if __ARM_NEON
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _pLoad = vld1q_f32(ptr);
                float32x4_t _blob = gelu_ps(_pLoad);
                vst1q_f32(ptr, _blob);
                ptr += 4;
            }
#endif
            for (; i < size; i++)
            {
                *ptr = 0.5f * *ptr * erfcf(-0.70710678f * *ptr);
                ptr++;
            }
        }
    }

    return 0;
}

#if NCNN_BF16
int GELU_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int elempack = bottom_top_blob.elempack;
    int channels = bottom_top_blob.c;
    int size = w * h * d * elempack;

    if (fast_gelu)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            int i = 0;

#if __ARM_NEON
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _pLoad = bfloat2float(vld1_u16(ptr));
                float32x4_t _blob = fast_gelu_ps(_pLoad);
                vst1_u16(ptr, float2bfloat(_blob));
                ptr += 4;
            }
#endif // __ARM_NEON

            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                v = 0.5f * v * (1.0f + tanhf(0.79788452f * (v + 0.044715f * v * v * v)));
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            int i = 0;

#if __ARM_NEON
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _pLoad = bfloat2float(vld1_u16(ptr));
                float32x4_t _blob = gelu_ps(_pLoad);
                vst1_u16(ptr, float2bfloat(_blob));
                ptr += 4;
            }
#endif // __ARM_NEON

            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                v = 0.5f * v * erfcf(-0.70710678f * v);
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn

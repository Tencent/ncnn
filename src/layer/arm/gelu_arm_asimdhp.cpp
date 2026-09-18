// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gelu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#include "neon_mathfun.h"
#if NCNN_ARM82
#include "neon_mathfun_fp16s.h"
#endif
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int GELU_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
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
            __fp16* ptr = (__fp16*)bottom_top_blob.channel(q);

            int i = 0;

            for (; i + 3 < size; i += 4)
            {
                float32x4_t _pLoad = vcvt_f32_f16(vld1_f16(ptr));
                float32x4_t _blob = fast_gelu_ps(_pLoad);
                vst1_f16(ptr, vcvt_f16_f32(_blob));
                ptr += 4;
            }

            for (; i < size; i++)
            {
                float v = (float)*ptr;
                v = 0.5f * v * (1.0f + tanhf(0.79788452f * (v + 0.044715f * v * v * v)));
                *ptr = (__fp16)v;
                ptr++;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = (__fp16*)bottom_top_blob.channel(q);

            int i = 0;

            for (; i + 3 < size; i += 4)
            {
                float32x4_t _pLoad = vcvt_f32_f16(vld1_f16(ptr));
                float32x4_t _blob = gelu_ps(_pLoad);
                vst1_f16(ptr, vcvt_f16_f32(_blob));
                ptr += 4;
            }

            for (; i < size; i++)
            {
                float v = (float)*ptr;
                v = 0.5f * v * erfcf(-0.70710678f * v);
                *ptr = (__fp16)v;
                ptr++;
            }
        }
    }

    return 0;
}

int GELU_arm::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
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
            __fp16* ptr = (__fp16*)bottom_top_blob.channel(q);

            int i = 0;

            for (; i + 7 < size; i += 8)
            {
                float16x8_t _pLoad = vld1q_f16(ptr);
                float16x8_t _blob = fast_gelu_ps_f16(_pLoad);
                vst1q_f16(ptr, _blob);
                ptr += 8;
            }

            for (; i < size; i++)
            {
                *ptr = (__fp16)0.5f * *ptr * (__fp16)(1.0f + tanhf((__fp16)0.79788452f * (*ptr + (__fp16)0.044715f * *ptr * *ptr * *ptr)));
                ptr++;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = (__fp16*)bottom_top_blob.channel(q);

            int i = 0;

            for (; i + 7 < size; i += 8)
            {
                float16x8_t _pLoad = vld1q_f16(ptr);
                float16x8_t _blob = gelu_ps_f16(_pLoad);
                vst1q_f16(ptr, _blob);
                ptr += 8;
            }

            for (; i < size; i++)
            {
                float v = (float)*ptr;
                v = 0.5f * v * erfcf(-0.70710678f * v);
                *ptr = (__fp16)v;
                ptr++;
            }
        }
    }

    return 0;
}
#endif

} // namespace ncnn

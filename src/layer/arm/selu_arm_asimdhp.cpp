// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "selu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#include "neon_mathfun.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "neon_mathfun_fp16s.h"
#endif
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int SELU_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    float alphaxlambda = alpha * lambda;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

        int i = 0;

        float32x4_t _alphaxlambda_f32 = vdupq_n_f32(alphaxlambda);
        float32x4_t _lambda_f32 = vdupq_n_f32(lambda);

        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);

            float32x4_t _p_low = selu_ps(vcvt_f32_f16(vget_low_f16(_p)), _alphaxlambda_f32, _lambda_f32);
            float32x4_t _p_high = selu_ps(vcvt_f32_f16(vget_high_f16(_p)), _alphaxlambda_f32, _lambda_f32);

            vst1q_f16(ptr, vcombine_f16(vcvt_f16_f32(_p_low), vcvt_f16_f32(_p_high)));
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);

            float32x4_t _p_f32 = selu_ps(vcvt_f32_f16(_p), _alphaxlambda_f32, _lambda_f32);

            vst1_f16(ptr, vcvt_f16_f32(_p_f32));
            ptr += 4;
        }
        for (; i < size; i++)
        {
            __fp16 v = ptr[0];
            if (v < (__fp16)0.f)
                ptr[0] = (__fp16)((expf((float)v) - 1.f) * alphaxlambda);
            else
                ptr[0] = (__fp16)((float)v * lambda);

            ptr += 1;
        }
    }

    return 0;
}

int SELU_arm::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    float alphaxlambda = alpha * lambda;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

        int i = 0;

        float16x8_t _alphaxlambda = vdupq_n_f16((__fp16)alphaxlambda);
        float16x8_t _lambda = vdupq_n_f16((__fp16)lambda);

        for (; i + 31 < size; i += 32)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x8_t _p2 = vld1q_f16(ptr + 16);
            float16x8_t _p3 = vld1q_f16(ptr + 24);

            _p0 = selu_ps_f16(_p0, _alphaxlambda, _lambda);
            _p1 = selu_ps_f16(_p1, _alphaxlambda, _lambda);
            _p2 = selu_ps_f16(_p2, _alphaxlambda, _lambda);
            _p3 = selu_ps_f16(_p3, _alphaxlambda, _lambda);

            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            vst1q_f16(ptr + 16, _p2);
            vst1q_f16(ptr + 24, _p3);
            ptr += 32;
        }
        for (; i + 15 < size; i += 16)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);

            _p0 = selu_ps_f16(_p0, _alphaxlambda, _lambda);
            _p1 = selu_ps_f16(_p1, _alphaxlambda, _lambda);

            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);

            _p = selu_ps_f16(_p, _alphaxlambda, _lambda);

            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);

            _p = selu_ps_f16(_p, vget_low_f16(_alphaxlambda), vget_low_f16(_lambda));

            vst1_f16(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            __fp16 v = ptr[0];
            if (v < (__fp16)0.f)
                ptr[0] = (__fp16)((expf((float)v) - 1.f) * alphaxlambda);
            else
                ptr[0] = v * (__fp16)lambda;

            ptr += 1;
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn

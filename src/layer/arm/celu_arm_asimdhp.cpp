// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "celu_arm.h"

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
int CELU_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    const float inv_alpha = 1.f / alpha;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

        int i = 0;

        float32x4_t _alpha_f32 = vdupq_n_f32(alpha);
        float32x4_t _inv_alpha_f32 = vdupq_n_f32(inv_alpha);

        for (; i + 15 < size; i += 16)
        {
            float16x8_t _p01 = vld1q_f16(ptr);
            float16x8_t _p23 = vld1q_f16(ptr + 8);
            float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p01));
            float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p01));
            float32x4_t _p2 = vcvt_f32_f16(vget_low_f16(_p23));
            float32x4_t _p3 = vcvt_f32_f16(vget_high_f16(_p23));
            _p0 = celu_ps(_p0, _alpha_f32, _inv_alpha_f32);
            _p1 = celu_ps(_p1, _alpha_f32, _inv_alpha_f32);
            _p2 = celu_ps(_p2, _alpha_f32, _inv_alpha_f32);
            _p3 = celu_ps(_p3, _alpha_f32, _inv_alpha_f32);
            _p01 = vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1));
            _p23 = vcombine_f16(vcvt_f16_f32(_p2), vcvt_f16_f32(_p3));
            vst1q_f16(ptr, _p01);
            vst1q_f16(ptr + 8, _p23);
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
            float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
            _p0 = celu_ps(_p0, _alpha_f32, _inv_alpha_f32);
            _p1 = celu_ps(_p1, _alpha_f32, _inv_alpha_f32);
            _p = vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1));
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
            _p = celu_ps(_p, _alpha_f32, _inv_alpha_f32);
            vst1_f16(ptr, vcvt_f16_f32(_p));
            ptr += 4;
        }
        for (; i < size; i++)
        {
            __fp16 v = ptr[0];
            if (v < (__fp16)0.f)
                ptr[0] = (__fp16)((expf((float)v * inv_alpha) - 1.f) * alpha);

            ptr += 1;
        }
    }

    return 0;
}

int CELU_arm::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    // fallback to fp32 compute when alpha or 1/alpha risks fp16 overflow or cancellation
    if (fabsf(alpha) < 1e-4f || fabsf(alpha) > 32.f)
        return forward_inplace_fp16s(bottom_top_blob, opt);

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    const float inv_alpha = 1.f / alpha;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

        int i = 0;

        float16x8_t _alpha = vdupq_n_f16((__fp16)alpha);
        float16x8_t _inv_alpha = vdupq_n_f16((__fp16)inv_alpha);

        for (; i + 31 < size; i += 32)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x8_t _p2 = vld1q_f16(ptr + 16);
            float16x8_t _p3 = vld1q_f16(ptr + 24);
            _p0 = celu_ps_f16(_p0, _alpha, _inv_alpha);
            _p1 = celu_ps_f16(_p1, _alpha, _inv_alpha);
            _p2 = celu_ps_f16(_p2, _alpha, _inv_alpha);
            _p3 = celu_ps_f16(_p3, _alpha, _inv_alpha);
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
            _p0 = celu_ps_f16(_p0, _alpha, _inv_alpha);
            _p1 = celu_ps_f16(_p1, _alpha, _inv_alpha);
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _p = celu_ps_f16(_p, _alpha, _inv_alpha);
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = celu_ps_f16(_p, vget_low_f16(_alpha), vget_low_f16(_inv_alpha));
            vst1_f16(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            __fp16 v = ptr[0];
            if (v < (__fp16)0.f)
                ptr[0] = (__fp16)((expf((float)v * inv_alpha) - 1.f) * alpha);

            ptr += 1;
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn

// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "erf_arm.h"

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
int Erf_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d * bottom_top_blob.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

        int i = 0;
        for (; i + 15 < size; i += 16)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float16x8_t _q = vld1q_f16(ptr + 8);
            float32x4_t _p0 = erf_ps(vcvt_f32_f16(vget_low_f16(_p)));
            float32x4_t _p1 = erf_ps(vcvt_f32_f16(vget_high_f16(_p)));
            float32x4_t _p2 = erf_ps(vcvt_f32_f16(vget_low_f16(_q)));
            float32x4_t _p3 = erf_ps(vcvt_f32_f16(vget_high_f16(_q)));
            _p = vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1));
            _q = vcombine_f16(vcvt_f16_f32(_p2), vcvt_f16_f32(_p3));
            vst1q_f16(ptr, _p);
            vst1q_f16(ptr + 8, _q);
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float32x4_t _p0 = erf_ps(vcvt_f32_f16(vget_low_f16(_p)));
            float32x4_t _p1 = erf_ps(vcvt_f32_f16(vget_high_f16(_p)));
            _p = vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1));
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
            _p = erf_ps(_p);
            vst1_f16(ptr, vcvt_f16_f32(_p));

            ptr += 4;
        }
        for (; i < size; i++)
        {
            float v = (float)*ptr;
            v = erff(v);
            *ptr = (__fp16)v;
            ptr++;
        }
    }

    return 0;
}

int Erf_arm::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d * bottom_top_blob.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

        int i = 0;
        for (; i + 15 < size; i += 16)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            _p0 = erf_ps_f16(_p0);
            _p1 = erf_ps_f16(_p1);
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _p = erf_ps_f16(_p);
            vst1q_f16(ptr, _p);

            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = erf_ps_f16(_p);
            vst1_f16(ptr, _p);

            ptr += 4;
        }
        for (; i < size; i++)
        {
            __fp16 v = *ptr;
            v = erff(v);
            *ptr = v;
            ptr++;
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn

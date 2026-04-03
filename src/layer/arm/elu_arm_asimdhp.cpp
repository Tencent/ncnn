// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "elu_arm.h"

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
int ELU_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
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

        int i = 0;

        float16x8_t _alpha = vdupq_n_f16((__fp16)alpha);
        float16x8_t _one = vdupq_n_f16((__fp16)1.f);
        float16x8_t _zero = vdupq_n_f16((__fp16)0.f);

        for (; i + 31 < size; i += 32)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x8_t _p2 = vld1q_f16(ptr + 16);
            float16x8_t _p3 = vld1q_f16(ptr + 24);

            uint16x8_t _lemask0 = vcleq_f16(_p0, _zero);
            uint16x8_t _lemask1 = vcleq_f16(_p1, _zero);
            uint16x8_t _lemask2 = vcleq_f16(_p2, _zero);
            uint16x8_t _lemask3 = vcleq_f16(_p3, _zero);

            // Convert to float32 for exp calculation
            float32x4_t _p0_low = vcvt_f32_f16(vget_low_f16(_p0));
            float32x4_t _p0_high = vcvt_f32_f16(vget_high_f16(_p0));
            float32x4_t _p1_low = vcvt_f32_f16(vget_low_f16(_p1));
            float32x4_t _p1_high = vcvt_f32_f16(vget_high_f16(_p1));
            float32x4_t _p2_low = vcvt_f32_f16(vget_low_f16(_p2));
            float32x4_t _p2_high = vcvt_f32_f16(vget_high_f16(_p2));
            float32x4_t _p3_low = vcvt_f32_f16(vget_low_f16(_p3));
            float32x4_t _p3_high = vcvt_f32_f16(vget_high_f16(_p3));

            _p0_low = exp_ps(_p0_low);
            _p0_high = exp_ps(_p0_high);
            _p1_low = exp_ps(_p1_low);
            _p1_high = exp_ps(_p1_high);
            _p2_low = exp_ps(_p2_low);
            _p2_high = exp_ps(_p2_high);
            _p3_low = exp_ps(_p3_low);
            _p3_high = exp_ps(_p3_high);

            float32x4_t _one_f32 = vdupq_n_f32(1.f);
            float32x4_t _alpha_f32 = vdupq_n_f32(alpha);

            _p0_low = vsubq_f32(_p0_low, _one_f32);
            _p0_high = vsubq_f32(_p0_high, _one_f32);
            _p1_low = vsubq_f32(_p1_low, _one_f32);
            _p1_high = vsubq_f32(_p1_high, _one_f32);
            _p2_low = vsubq_f32(_p2_low, _one_f32);
            _p2_high = vsubq_f32(_p2_high, _one_f32);
            _p3_low = vsubq_f32(_p3_low, _one_f32);
            _p3_high = vsubq_f32(_p3_high, _one_f32);

            _p0_low = vmulq_f32(_p0_low, _alpha_f32);
            _p0_high = vmulq_f32(_p0_high, _alpha_f32);
            _p1_low = vmulq_f32(_p1_low, _alpha_f32);
            _p1_high = vmulq_f32(_p1_high, _alpha_f32);
            _p2_low = vmulq_f32(_p2_low, _alpha_f32);
            _p2_high = vmulq_f32(_p2_high, _alpha_f32);
            _p3_low = vmulq_f32(_p3_low, _alpha_f32);
            _p3_high = vmulq_f32(_p3_high, _alpha_f32);

            float16x8_t _nps0 = vcombine_f16(vcvt_f16_f32(_p0_low), vcvt_f16_f32(_p0_high));
            float16x8_t _nps1 = vcombine_f16(vcvt_f16_f32(_p1_low), vcvt_f16_f32(_p1_high));
            float16x8_t _nps2 = vcombine_f16(vcvt_f16_f32(_p2_low), vcvt_f16_f32(_p2_high));
            float16x8_t _nps3 = vcombine_f16(vcvt_f16_f32(_p3_low), vcvt_f16_f32(_p3_high));

            _p0 = vbslq_f16(_lemask0, _nps0, _p0);
            _p1 = vbslq_f16(_lemask1, _nps1, _p1);
            _p2 = vbslq_f16(_lemask2, _nps2, _p2);
            _p3 = vbslq_f16(_lemask3, _nps3, _p3);

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

            uint16x8_t _lemask0 = vcleq_f16(_p0, _zero);
            uint16x8_t _lemask1 = vcleq_f16(_p1, _zero);

            float32x4_t _p0_low = vcvt_f32_f16(vget_low_f16(_p0));
            float32x4_t _p0_high = vcvt_f32_f16(vget_high_f16(_p0));
            float32x4_t _p1_low = vcvt_f32_f16(vget_low_f16(_p1));
            float32x4_t _p1_high = vcvt_f32_f16(vget_high_f16(_p1));

            _p0_low = exp_ps(_p0_low);
            _p0_high = exp_ps(_p0_high);
            _p1_low = exp_ps(_p1_low);
            _p1_high = exp_ps(_p1_high);

            float32x4_t _one_f32 = vdupq_n_f32(1.f);
            float32x4_t _alpha_f32 = vdupq_n_f32(alpha);

            _p0_low = vsubq_f32(_p0_low, _one_f32);
            _p0_high = vsubq_f32(_p0_high, _one_f32);
            _p1_low = vsubq_f32(_p1_low, _one_f32);
            _p1_high = vsubq_f32(_p1_high, _one_f32);

            _p0_low = vmulq_f32(_p0_low, _alpha_f32);
            _p0_high = vmulq_f32(_p0_high, _alpha_f32);
            _p1_low = vmulq_f32(_p1_low, _alpha_f32);
            _p1_high = vmulq_f32(_p1_high, _alpha_f32);

            float16x8_t _nps0 = vcombine_f16(vcvt_f16_f32(_p0_low), vcvt_f16_f32(_p0_high));
            float16x8_t _nps1 = vcombine_f16(vcvt_f16_f32(_p1_low), vcvt_f16_f32(_p1_high));

            _p0 = vbslq_f16(_lemask0, _nps0, _p0);
            _p1 = vbslq_f16(_lemask1, _nps1, _p1);

            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            uint16x8_t _lemask = vcleq_f16(_p, _zero);

            float32x4_t _p_low = vcvt_f32_f16(vget_low_f16(_p));
            float32x4_t _p_high = vcvt_f32_f16(vget_high_f16(_p));

            _p_low = exp_ps(_p_low);
            _p_high = exp_ps(_p_high);

            float32x4_t _one_f32 = vdupq_n_f32(1.f);
            float32x4_t _alpha_f32 = vdupq_n_f32(alpha);

            _p_low = vsubq_f32(_p_low, _one_f32);
            _p_high = vsubq_f32(_p_high, _one_f32);
            _p_low = vmulq_f32(_p_low, _alpha_f32);
            _p_high = vmulq_f32(_p_high, _alpha_f32);

            float16x8_t _nps = vcombine_f16(vcvt_f16_f32(_p_low), vcvt_f16_f32(_p_high));
            _p = vbslq_f16(_lemask, _nps, _p);

            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            uint16x4_t _lemask = vcle_f16(_p, vget_low_f16(_zero));

            float32x4_t _p_f32 = vcvt_f32_f16(_p);
            _p_f32 = exp_ps(_p_f32);

            float32x4_t _one_f32 = vdupq_n_f32(1.f);
            float32x4_t _alpha_f32 = vdupq_n_f32(alpha);

            _p_f32 = vsubq_f32(_p_f32, _one_f32);
            _p_f32 = vmulq_f32(_p_f32, _alpha_f32);

            float16x4_t _nps = vcvt_f16_f32(_p_f32);
            _p = vbsl_f16(_lemask, _nps, _p);

            vst1_f16(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            __fp16 v = ptr[0];
            if (v < (__fp16)0.f)
                ptr[0] = (__fp16)(alpha * (expf((float)v) - 1.f));

            ptr += 1;
        }
    }

    return 0;
}

int ELU_arm::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
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

        int i = 0;

        float16x8_t _alpha = vdupq_n_f16((__fp16)alpha);
        float16x8_t _one = vdupq_n_f16((__fp16)1.f);
        float16x8_t _zero = vdupq_n_f16((__fp16)0.f);

        for (; i + 31 < size; i += 32)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x8_t _p2 = vld1q_f16(ptr + 16);
            float16x8_t _p3 = vld1q_f16(ptr + 24);

            uint16x8_t _lemask0 = vcleq_f16(_p0, _zero);
            uint16x8_t _lemask1 = vcleq_f16(_p1, _zero);
            uint16x8_t _lemask2 = vcleq_f16(_p2, _zero);
            uint16x8_t _lemask3 = vcleq_f16(_p3, _zero);

            float16x8_t _nps0 = exp_ps_f16(_p0);
            float16x8_t _nps1 = exp_ps_f16(_p1);
            float16x8_t _nps2 = exp_ps_f16(_p2);
            float16x8_t _nps3 = exp_ps_f16(_p3);

            _nps0 = vsubq_f16(_nps0, _one);
            _nps1 = vsubq_f16(_nps1, _one);
            _nps2 = vsubq_f16(_nps2, _one);
            _nps3 = vsubq_f16(_nps3, _one);

            _nps0 = vmulq_f16(_nps0, _alpha);
            _nps1 = vmulq_f16(_nps1, _alpha);
            _nps2 = vmulq_f16(_nps2, _alpha);
            _nps3 = vmulq_f16(_nps3, _alpha);

            _p0 = vbslq_f16(_lemask0, _nps0, _p0);
            _p1 = vbslq_f16(_lemask1, _nps1, _p1);
            _p2 = vbslq_f16(_lemask2, _nps2, _p2);
            _p3 = vbslq_f16(_lemask3, _nps3, _p3);

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

            uint16x8_t _lemask0 = vcleq_f16(_p0, _zero);
            uint16x8_t _lemask1 = vcleq_f16(_p1, _zero);

            float16x8_t _nps0 = exp_ps_f16(_p0);
            float16x8_t _nps1 = exp_ps_f16(_p1);

            _nps0 = vsubq_f16(_nps0, _one);
            _nps1 = vsubq_f16(_nps1, _one);

            _nps0 = vmulq_f16(_nps0, _alpha);
            _nps1 = vmulq_f16(_nps1, _alpha);

            _p0 = vbslq_f16(_lemask0, _nps0, _p0);
            _p1 = vbslq_f16(_lemask1, _nps1, _p1);

            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            uint16x8_t _lemask = vcleq_f16(_p, _zero);

            float16x8_t _nps = exp_ps_f16(_p);
            _nps = vsubq_f16(_nps, _one);
            _nps = vmulq_f16(_nps, _alpha);
            _p = vbslq_f16(_lemask, _nps, _p);

            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            uint16x4_t _lemask = vcle_f16(_p, vget_low_f16(_zero));

            float16x4_t _nps = exp_ps_f16(_p);
            _nps = vsub_f16(_nps, vget_low_f16(_one));
            _nps = vmul_f16(_nps, vget_low_f16(_alpha));
            _p = vbsl_f16(_lemask, _nps, _p);

            vst1_f16(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            __fp16 v = ptr[0];
            if (v < (__fp16)0.f)
                ptr[0] = (__fp16)(alpha * (expf((float)v) - 1.f));

            ptr += 1;
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn

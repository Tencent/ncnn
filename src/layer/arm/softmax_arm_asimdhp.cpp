// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "softmax_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#include "neon_mathfun_fp16s.h"
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static void softmax_fp16s(__fp16* _ptr, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // reduce max
    float16x8_t _max8 = vdupq_n_f16(-65504.f);
    float16x4_t _max4 = vdup_n_f16(-65504.f);
    __fp16 max = -65504.f;
    {
        const __fp16* ptr = _ptr;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _max8 = vmaxq_f16(_max8, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _max4 = vmax_f16(_max4, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            max = std::max(max, *ptr++);
        }
    }

    if (elempack == 4)
    {
        _max4 = vmax_f16(_max4, vget_low_f16(_max8));
        _max4 = vmax_f16(_max4, vget_high_f16(_max8));

        _max8 = vcombine_f16(_max4, _max4);
    }
    if (elempack == 1)
    {
        max = std::max(max, vmaxvq_f16(_max8));
        max = std::max(max, vmaxv_f16(_max4));

        _max4 = vdup_n_f16(max);
        _max8 = vdupq_n_f16(max);
    }

    // reduce exp(x - max)
    float16x8_t _sum8 = vdupq_n_f16(0.f);
    float16x4_t _sum4 = vdup_n_f16(0.f);
    __fp16 sum = 0.f;
    {
        __fp16* ptr = _ptr;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _p = exp_ps_f16(vsubq_f16(_p, _max8));
            vst1q_f16(ptr, _p);
            _sum8 = vaddq_f16(_sum8, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = exp_ps_f16(vsub_f16(_p, _max4));
            vst1_f16(ptr, _p);
            _sum4 = vadd_f16(_sum4, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            __fp16 v = (__fp16)expf(*ptr - max);
            *ptr = v;
            sum += v;
            ptr++;
        }
    }

    if (elempack == 4)
    {
        _sum4 = vadd_f16(_sum4, vget_low_f16(_sum8));
        _sum4 = vadd_f16(_sum4, vget_high_f16(_sum8));

        _sum8 = vcombine_f16(_sum4, _sum4);
    }
    if (elempack == 1)
    {
        _sum4 = vadd_f16(_sum4, vget_low_f16(_sum8));
        _sum4 = vadd_f16(_sum4, vget_high_f16(_sum8));
        _sum4 = vpadd_f16(_sum4, _sum4);
        _sum4 = vpadd_f16(_sum4, _sum4);
        sum += vget_lane_f16(_sum4, 0);

        _sum4 = vdup_n_f16(sum);
        _sum8 = vdupq_n_f16(sum);
    }

    _sum8 = vdivq_f16(vdupq_n_f16(1.f), _sum8);
    _sum4 = vdiv_f16(vdup_n_f16(1.f), _sum4);
    sum = (__fp16)1.f / sum;

    // div sum
    {
        __fp16* ptr = _ptr;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _p = vmulq_f16(_p, _sum8);
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = vmul_f16(_p, _sum4);
            vst1_f16(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            *ptr++ *= sum;
        }
    }
}

static void softmax_fp16s_pack8(__fp16* _ptr, int elemcount, size_t stride, int size1, __fp16* _maxptr, __fp16* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const __fp16* ptr = _ptr + i * stride;
        __fp16* maxptr = _maxptr;

        int j = 0;
        for (; j + 7 < size1; j += 8)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x8_t _p2 = vld1q_f16(ptr + 16);
            float16x8_t _p3 = vld1q_f16(ptr + 24);
            float16x8_t _p4 = vld1q_f16(ptr + 32);
            float16x8_t _p5 = vld1q_f16(ptr + 40);
            float16x8_t _p6 = vld1q_f16(ptr + 48);
            float16x8_t _p7 = vld1q_f16(ptr + 56);
            float16x8_t _max = vld1q_f16(maxptr);
            float16x8_t _max01 = vpmaxq_f16(_p0, _p1);
            float16x8_t _max23 = vpmaxq_f16(_p2, _p3);
            float16x8_t _max45 = vpmaxq_f16(_p4, _p5);
            float16x8_t _max67 = vpmaxq_f16(_p6, _p7);
            float16x8_t _max2 = vpmaxq_f16(_max01, _max23);
            float16x8_t _max4 = vpmaxq_f16(_max45, _max67);
            _max = vmaxq_f16(_max, vpmaxq_f16(_max2, _max4));
            vst1q_f16(maxptr, _max);
            ptr += 64;
            maxptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x8_t _p2 = vld1q_f16(ptr + 16);
            float16x8_t _p3 = vld1q_f16(ptr + 24);
            float16x4_t _max = vld1_f16(maxptr);
            float16x8_t _max01 = vpmaxq_f16(_p0, _p1);
            float16x8_t _max23 = vpmaxq_f16(_p2, _p3);
            float16x8_t _max2 = vpmaxq_f16(_max01, _max23);
            _max = vmax_f16(_max, vpmax_f16(vget_low_f16(_max2), vget_high_f16(_max2)));
            vst1_f16(maxptr, _max);
            ptr += 32;
            maxptr += 4;
        }
        for (; j < size1; j++)
        {
            float16x8_t _p = vld1q_f16(ptr);
            __fp16 max0 = vmaxvq_f16(_p);
            *maxptr = std::max(*maxptr, max0);
            ptr += 8;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        __fp16* ptr = _ptr + i * stride;
        const __fp16* maxptr = _maxptr;
        __fp16* sumptr = _sumptr;

        int j = 0;
        for (; j + 7 < size1; j += 8)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x8_t _p2 = vld1q_f16(ptr + 16);
            float16x8_t _p3 = vld1q_f16(ptr + 24);
            float16x8_t _p4 = vld1q_f16(ptr + 32);
            float16x8_t _p5 = vld1q_f16(ptr + 40);
            float16x8_t _p6 = vld1q_f16(ptr + 48);
            float16x8_t _p7 = vld1q_f16(ptr + 56);
            float16x8_t _max = vld1q_f16(maxptr);
            _p0 = exp_ps_f16(vsubq_f16(_p0, vdupq_laneq_f16(_max, 0)));
            _p1 = exp_ps_f16(vsubq_f16(_p1, vdupq_laneq_f16(_max, 1)));
            _p2 = exp_ps_f16(vsubq_f16(_p2, vdupq_laneq_f16(_max, 2)));
            _p3 = exp_ps_f16(vsubq_f16(_p3, vdupq_laneq_f16(_max, 3)));
            _p4 = exp_ps_f16(vsubq_f16(_p4, vdupq_laneq_f16(_max, 4)));
            _p5 = exp_ps_f16(vsubq_f16(_p5, vdupq_laneq_f16(_max, 5)));
            _p6 = exp_ps_f16(vsubq_f16(_p6, vdupq_laneq_f16(_max, 6)));
            _p7 = exp_ps_f16(vsubq_f16(_p7, vdupq_laneq_f16(_max, 7)));
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            vst1q_f16(ptr + 16, _p2);
            vst1q_f16(ptr + 24, _p3);
            vst1q_f16(ptr + 32, _p4);
            vst1q_f16(ptr + 40, _p5);
            vst1q_f16(ptr + 48, _p6);
            vst1q_f16(ptr + 56, _p7);
            float16x8_t _sum = vld1q_f16(sumptr);
            float16x8_t _ss01 = vpaddq_f16(_p0, _p1);
            float16x8_t _ss23 = vpaddq_f16(_p2, _p3);
            float16x8_t _ss45 = vpaddq_f16(_p4, _p5);
            float16x8_t _ss67 = vpaddq_f16(_p6, _p7);
            float16x8_t _ss2 = vpaddq_f16(_ss01, _ss23);
            float16x8_t _ss4 = vpaddq_f16(_ss45, _ss67);
            _sum = vaddq_f16(_sum, vpaddq_f16(_ss2, _ss4));
            vst1q_f16(sumptr, _sum);
            ptr += 64;
            maxptr += 8;
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x8_t _p2 = vld1q_f16(ptr + 16);
            float16x8_t _p3 = vld1q_f16(ptr + 24);
            float16x4_t _max = vld1_f16(maxptr);
            _p0 = exp_ps_f16(vsubq_f16(_p0, vdupq_lane_f16(_max, 0)));
            _p1 = exp_ps_f16(vsubq_f16(_p1, vdupq_lane_f16(_max, 1)));
            _p2 = exp_ps_f16(vsubq_f16(_p2, vdupq_lane_f16(_max, 2)));
            _p3 = exp_ps_f16(vsubq_f16(_p3, vdupq_lane_f16(_max, 3)));
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            vst1q_f16(ptr + 16, _p2);
            vst1q_f16(ptr + 24, _p3);
            float16x4_t _sum = vld1_f16(sumptr);
            float16x8_t _ss01 = vpaddq_f16(_p0, _p1);
            float16x8_t _ss23 = vpaddq_f16(_p2, _p3);
            float16x8_t _ss2 = vpaddq_f16(_ss01, _ss23);
            _sum = vadd_f16(_sum, vpadd_f16(vget_low_f16(_ss2), vget_high_f16(_ss2)));
            vst1_f16(sumptr, _sum);
            ptr += 32;
            maxptr += 4;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float16x8_t _max = vdupq_n_f16(*maxptr);
            _p = exp_ps_f16(vsubq_f16(_p, _max));
            vst1q_f16(ptr, _p);
            float16x4_t _sum2 = vadd_f16(vget_low_f16(_p), vget_high_f16(_p));
            float16x4_t _ss2 = vpadd_f16(_sum2, _sum2);
            __fp16 sum0 = vget_lane_f16(_ss2, 0) + vget_lane_f16(_ss2, 1);
            *sumptr += sum0;
            ptr += 8;
            maxptr++;
            sumptr++;
        }
    }

    {
        float16x8_t _one = vdupq_n_f16(1.f);
        __fp16* sumptr = _sumptr;
        int j = 0;
        for (; j + 7 < size1; j += 8)
        {
            float16x8_t _sum = vld1q_f16(sumptr);
            _sum = vdivq_f16(_one, _sum);
            vst1q_f16(sumptr, _sum);
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            float16x4_t _sum = vld1_f16(sumptr);
            _sum = vdiv_f16(vget_low_f16(_one), _sum);
            vst1_f16(sumptr, _sum);
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            *sumptr = (__fp16)1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        __fp16* ptr = _ptr + i * stride;
        const __fp16* sumptr = _sumptr;

        int j = 0;
        for (; j + 7 < size1; j += 8)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x8_t _p2 = vld1q_f16(ptr + 16);
            float16x8_t _p3 = vld1q_f16(ptr + 24);
            float16x8_t _p4 = vld1q_f16(ptr + 32);
            float16x8_t _p5 = vld1q_f16(ptr + 40);
            float16x8_t _p6 = vld1q_f16(ptr + 48);
            float16x8_t _p7 = vld1q_f16(ptr + 56);
            float16x8_t _sum = vld1q_f16(sumptr);
            _p0 = vmulq_laneq_f16(_p0, _sum, 0);
            _p1 = vmulq_laneq_f16(_p1, _sum, 1);
            _p2 = vmulq_laneq_f16(_p2, _sum, 2);
            _p3 = vmulq_laneq_f16(_p3, _sum, 3);
            _p4 = vmulq_laneq_f16(_p4, _sum, 4);
            _p5 = vmulq_laneq_f16(_p5, _sum, 5);
            _p6 = vmulq_laneq_f16(_p6, _sum, 6);
            _p7 = vmulq_laneq_f16(_p7, _sum, 7);
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            vst1q_f16(ptr + 16, _p2);
            vst1q_f16(ptr + 24, _p3);
            vst1q_f16(ptr + 32, _p4);
            vst1q_f16(ptr + 40, _p5);
            vst1q_f16(ptr + 48, _p6);
            vst1q_f16(ptr + 56, _p7);
            ptr += 64;
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x8_t _p2 = vld1q_f16(ptr + 16);
            float16x8_t _p3 = vld1q_f16(ptr + 24);
            float16x4_t _sum = vld1_f16(sumptr);
            _p0 = vmulq_lane_f16(_p0, _sum, 0);
            _p1 = vmulq_lane_f16(_p1, _sum, 1);
            _p2 = vmulq_lane_f16(_p2, _sum, 2);
            _p3 = vmulq_lane_f16(_p3, _sum, 3);
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            vst1q_f16(ptr + 16, _p2);
            vst1q_f16(ptr + 24, _p3);
            ptr += 32;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float16x8_t _sum = vld1q_dup_f16(sumptr);
            _p = vmulq_f16(_p, _sum);
            vst1q_f16(ptr, _p);
            ptr += 8;
            sumptr++;
        }
    }
}

static void softmax_fp16s_pack4(__fp16* _ptr, int elemcount, size_t stride, int size1, __fp16* _maxptr, __fp16* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const __fp16* ptr = _ptr + i * stride;
        __fp16* maxptr = _maxptr;

        int j = 0;
        for (; j + 7 < size1; j += 8)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x8_t _p2 = vld1q_f16(ptr + 16);
            float16x8_t _p3 = vld1q_f16(ptr + 24);
            float16x8_t _max = vld1q_f16(maxptr);
            float16x8_t _max2 = vpmaxq_f16(_p0, _p1);
            float16x8_t _max4 = vpmaxq_f16(_p2, _p3);
            _max = vmaxq_f16(_max, vpmaxq_f16(_max2, _max4));
            vst1q_f16(maxptr, _max);
            ptr += 32;
            maxptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x4_t _max = vld1_f16(maxptr);
            float16x8_t _max2 = vpmaxq_f16(_p0, _p1);
            _max = vmax_f16(_max, vpmax_f16(vget_low_f16(_max2), vget_high_f16(_max2)));
            vst1_f16(maxptr, _max);
            ptr += 16;
            maxptr += 4;
        }
        for (; j < size1; j++)
        {
            float16x4_t _p = vld1_f16(ptr);
            __fp16 max0 = vmaxv_f16(_p);
            *maxptr = std::max(*maxptr, max0);
            ptr += 4;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        __fp16* ptr = _ptr + i * stride;
        const __fp16* maxptr = _maxptr;
        __fp16* sumptr = _sumptr;

        int j = 0;
        for (; j + 7 < size1; j += 8)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x8_t _p2 = vld1q_f16(ptr + 16);
            float16x8_t _p3 = vld1q_f16(ptr + 24);
            float16x8_t _max = vld1q_f16(maxptr);
            float16x8_t _max0 = vcombine_f16(vdup_laneq_f16(_max, 0), vdup_laneq_f16(_max, 1));
            float16x8_t _max1 = vcombine_f16(vdup_laneq_f16(_max, 2), vdup_laneq_f16(_max, 3));
            float16x8_t _max2 = vcombine_f16(vdup_laneq_f16(_max, 4), vdup_laneq_f16(_max, 5));
            float16x8_t _max3 = vcombine_f16(vdup_laneq_f16(_max, 6), vdup_laneq_f16(_max, 7));
            _p0 = exp_ps_f16(vsubq_f16(_p0, _max0));
            _p1 = exp_ps_f16(vsubq_f16(_p1, _max1));
            _p2 = exp_ps_f16(vsubq_f16(_p2, _max2));
            _p3 = exp_ps_f16(vsubq_f16(_p3, _max3));
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            vst1q_f16(ptr + 16, _p2);
            vst1q_f16(ptr + 24, _p3);
            float16x8_t _sum = vld1q_f16(sumptr);
            float16x8_t _ss2 = vpaddq_f16(_p0, _p1);
            float16x8_t _ss4 = vpaddq_f16(_p2, _p3);
            _sum = vaddq_f16(_sum, vpaddq_f16(_ss2, _ss4));
            vst1q_f16(sumptr, _sum);
            ptr += 32;
            maxptr += 8;
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x4_t _max = vld1_f16(maxptr);
            float16x8_t _max0 = vcombine_f16(vdup_lane_f16(_max, 0), vdup_lane_f16(_max, 1));
            float16x8_t _max1 = vcombine_f16(vdup_lane_f16(_max, 2), vdup_lane_f16(_max, 3));
            _p0 = exp_ps_f16(vsubq_f16(_p0, _max0));
            _p1 = exp_ps_f16(vsubq_f16(_p1, _max1));
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            float16x4_t _sum = vld1_f16(sumptr);
            float16x8_t _ss2 = vpaddq_f16(_p0, _p1);
            _sum = vadd_f16(_sum, vpadd_f16(vget_low_f16(_ss2), vget_high_f16(_ss2)));
            vst1_f16(sumptr, _sum);
            ptr += 16;
            maxptr += 4;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            float16x4_t _p = vld1_f16(ptr);
            float16x4_t _max = vdup_n_f16(*maxptr);
            _p = exp_ps_f16(vsub_f16(_p, _max));
            vst1_f16(ptr, _p);
            float16x4_t _ss2 = vpadd_f16(_p, _p);
            __fp16 sum0 = vget_lane_f16(_ss2, 0) + vget_lane_f16(_ss2, 1);
            *sumptr += sum0;
            ptr += 4;
            maxptr++;
            sumptr++;
        }
    }

    {
        float16x8_t _one = vdupq_n_f16(1.f);
        __fp16* sumptr = _sumptr;
        int j = 0;
        for (; j + 7 < size1; j += 8)
        {
            float16x8_t _sum = vld1q_f16(sumptr);
            _sum = vdivq_f16(_one, _sum);
            vst1q_f16(sumptr, _sum);
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            float16x4_t _sum = vld1_f16(sumptr);
            _sum = vdiv_f16(vget_low_f16(_one), _sum);
            vst1_f16(sumptr, _sum);
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            *sumptr = (__fp16)1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        __fp16* ptr = _ptr + i * stride;
        const __fp16* sumptr = _sumptr;

        int j = 0;
        for (; j + 7 < size1; j += 8)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x8_t _p2 = vld1q_f16(ptr + 16);
            float16x8_t _p3 = vld1q_f16(ptr + 24);
            float16x8_t _sum = vld1q_f16(sumptr);
            float16x8_t _sum0 = vcombine_f16(vdup_laneq_f16(_sum, 0), vdup_laneq_f16(_sum, 1));
            float16x8_t _sum1 = vcombine_f16(vdup_laneq_f16(_sum, 2), vdup_laneq_f16(_sum, 3));
            float16x8_t _sum2 = vcombine_f16(vdup_laneq_f16(_sum, 4), vdup_laneq_f16(_sum, 5));
            float16x8_t _sum3 = vcombine_f16(vdup_laneq_f16(_sum, 6), vdup_laneq_f16(_sum, 7));
            _p0 = vmulq_f16(_p0, _sum0);
            _p1 = vmulq_f16(_p1, _sum1);
            _p2 = vmulq_f16(_p2, _sum2);
            _p3 = vmulq_f16(_p3, _sum3);
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            vst1q_f16(ptr + 16, _p2);
            vst1q_f16(ptr + 24, _p3);
            ptr += 32;
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x4_t _sum = vld1_f16(sumptr);
            float16x8_t _sum0 = vcombine_f16(vdup_lane_f16(_sum, 0), vdup_lane_f16(_sum, 1));
            float16x8_t _sum1 = vcombine_f16(vdup_lane_f16(_sum, 2), vdup_lane_f16(_sum, 3));
            _p0 = vmulq_f16(_p0, _sum0);
            _p1 = vmulq_f16(_p1, _sum1);
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            ptr += 16;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            float16x4_t _p = vld1_f16(ptr);
            float16x4_t _sum = vld1_dup_f16(sumptr);
            _p = vmul_f16(_p, _sum);
            vst1_f16(ptr, _p);
            ptr += 4;
            sumptr++;
        }
    }
}

static void softmax_fp16s_pack1(__fp16* _ptr, int elemcount, size_t stride, int size1, __fp16* _maxptr, __fp16* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const __fp16* ptr = _ptr + i * stride;
        __fp16* maxptr = _maxptr;

        int j = 0;
        for (; j + 7 < size1; j += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float16x8_t _max = vld1q_f16(maxptr);
            _max = vmaxq_f16(_max, _p);
            vst1q_f16(maxptr, _max);
            ptr += 8;
            maxptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            float16x4_t _max = vld1_f16(maxptr);
            _max = vmax_f16(_max, _p);
            vst1_f16(maxptr, _max);
            ptr += 4;
            maxptr += 4;
        }
        for (; j < size1; j++)
        {
            *maxptr = std::max(*maxptr, *ptr);
            ptr++;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        __fp16* ptr = _ptr + i * stride;
        const __fp16* maxptr = _maxptr;
        __fp16* sumptr = _sumptr;

        int j = 0;
        for (; j + 7 < size1; j += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float16x8_t _max = vld1q_f16(maxptr);
            float16x8_t _sum = vld1q_f16(sumptr);
            _p = vsubq_f16(_p, _max);
            _p = exp_ps_f16(_p);
            vst1q_f16(ptr, _p);
            _sum = vaddq_f16(_sum, _p);
            vst1q_f16(sumptr, _sum);
            ptr += 8;
            maxptr += 8;
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            float16x4_t _max = vld1_f16(maxptr);
            float16x4_t _sum = vld1_f16(sumptr);
            _p = vsub_f16(_p, _max);
            _p = exp_ps_f16(_p);
            vst1_f16(ptr, _p);
            _sum = vadd_f16(_sum, _p);
            vst1_f16(sumptr, _sum);
            ptr += 4;
            maxptr += 4;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            __fp16 v = expf(*ptr - *maxptr);
            *ptr = v;
            *sumptr += v;
            ptr++;
            maxptr++;
            sumptr++;
        }
    }

    {
        float16x8_t _one = vdupq_n_f16(1.f);
        __fp16* sumptr = _sumptr;
        int j = 0;
        for (; j + 7 < size1; j += 8)
        {
            float16x8_t _sum = vld1q_f16(sumptr);
            _sum = vdivq_f16(_one, _sum);
            vst1q_f16(sumptr, _sum);
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            float16x4_t _sum = vld1_f16(sumptr);
            _sum = vdiv_f16(vget_low_f16(_one), _sum);
            vst1_f16(sumptr, _sum);
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            *sumptr = (__fp16)1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        __fp16* ptr = _ptr + i * stride;
        const __fp16* sumptr = _sumptr;

        int j = 0;
        for (; j + 7 < size1; j += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float16x8_t _sum = vld1q_f16(sumptr);
            _p = vmulq_f16(_p, _sum);
            vst1q_f16(ptr, _p);
            ptr += 8;
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            float16x4_t _sum = vld1_f16(sumptr);
            _p = vmul_f16(_p, _sum);
            vst1_f16(ptr, _p);
            ptr += 4;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            *ptr *= *sumptr;
            ptr++;
            sumptr++;
        }
    }
}

static void softmax_fp16s(__fp16* _ptr, int elemcount, int elempack, size_t stride, int size1, __fp16* _maxptr, __fp16* _sumptr)
{
    // reduce max
    {
        float16x8_t _negmax = vdupq_n_f16(-65504.f);

        __fp16* maxptr = _maxptr;

        int j = 0;
        for (; j + 7 < size1; j += 8)
        {
            vst1q_f16(maxptr, _negmax);
            maxptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            vst1_f16(maxptr, vget_low_f16(_negmax));
            maxptr += 4;
        }
        for (; j < size1; j++)
        {
            *maxptr++ = -65504.f;
        }
    }

    // reduce exp(x - max)
    {
        float16x8_t _zero = vdupq_n_f16(0.f);

        __fp16* sumptr = _sumptr;

        int j = 0;
        for (; j + 7 < size1; j += 8)
        {
            vst1q_f16(sumptr, _zero);
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            vst1_f16(sumptr, vget_low_f16(_zero));
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            *sumptr++ = 0.f;
        }
    }

    if (elempack == 8)
    {
        softmax_fp16s_pack8(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
    if (elempack == 4)
    {
        softmax_fp16s_pack4(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
    if (elempack == 1)
    {
        softmax_fp16s_pack1(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
}

int Softmax_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        __fp16* ptr = bottom_top_blob;

        const int size = w * elempack;

        softmax_fp16s(ptr, size, 1);
    }

    if (dims == 2 && positive_axis == 0)
    {
        const int size = w;
        const int sizen = (size + (opt.num_threads - 1)) / opt.num_threads;
        const size_t stride = (size_t)w * elempack;

        Mat maxsum(sizen, 2, opt.num_threads, 2u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        const int nn_size = (size + sizen - 1) / sizen;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            __fp16* maxsumptr = maxsum.channel(get_omp_thread_num());
            __fp16* maxptr = maxsumptr;
            __fp16* sumptr = maxptr + sizen;

            __fp16* ptr = (__fp16*)bottom_top_blob + i * elempack;

            softmax_fp16s(ptr, h, elempack, stride, size1, maxptr, sumptr);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            __fp16* ptr = bottom_top_blob.row<__fp16>(i);

            softmax_fp16s(ptr, w, elempack);
        }
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        const int size = w * h * d;
        const int sizen = (size + (opt.num_threads - 1)) / opt.num_threads;
        const size_t stride = bottom_top_blob.cstep * elempack;

        Mat maxsum(sizen, 2, opt.num_threads, 2u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        const int nn_size = (size + sizen - 1) / sizen;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            __fp16* maxsumptr = maxsum.channel(get_omp_thread_num());
            __fp16* maxptr = maxsumptr;
            __fp16* sumptr = maxptr + sizen;

            __fp16* ptr = (__fp16*)bottom_top_blob + i * elempack;

            softmax_fp16s(ptr, channels, elempack, stride, size1, maxptr, sumptr);
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        const int size = w * elempack;

        Mat maxsum(size, 2, opt.num_threads, 2u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int i = 0; i < d; i++)
            {
                __fp16* ptr = bottom_top_blob.channel(q).depth(i);

                __fp16* maxsumptr = maxsum.channel(get_omp_thread_num());
                __fp16* maxptr = maxsumptr;
                __fp16* sumptr = maxptr + size;

                softmax_fp16s(ptr, h, 1, size, size, maxptr, sumptr);
            }
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                softmax_fp16s(ptr, w, elempack);
                ptr += w * elempack;
            }
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        const int size = w * h * elempack;

        Mat maxsum(size, 2, opt.num_threads, 2u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            __fp16* maxsumptr = maxsum.channel(get_omp_thread_num());
            __fp16* maxptr = maxsumptr;
            __fp16* sumptr = maxptr + size;

            softmax_fp16s(ptr, d, 1, size, size, maxptr, sumptr);
        }
    }

    if (dims == 4 && positive_axis == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    softmax_fp16s(ptr, w, elempack);
                    ptr += w * elempack;
                }
            }
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn

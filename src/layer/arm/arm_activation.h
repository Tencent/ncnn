// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef ARM_ACTIVATION_H
#define ARM_ACTIVATION_H

#include "fused_activation.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"

static inline float32x4_t activation_ps(float32x4_t _v, int activation_type, const ncnn::Mat& activation_params)
{
    if (activation_type == 1)
    {
        const float32x4_t _zero = vdupq_n_f32(0.f);
        _v = vmaxq_f32(_v, _zero);
    }
    else if (activation_type == 2)
    {
        const float32x4_t _zero = vdupq_n_f32(0.f);
        const float32x4_t _slope = vdupq_n_f32(activation_params[0]);
        const uint32x4_t _lemask = vcleq_f32(_v, _zero);
        float32x4_t _ps = vmulq_f32(_v, _slope);
        _v = vbslq_f32(_lemask, _ps, _v);
    }
    else if (activation_type == 3)
    {
        const float32x4_t _min = vdupq_n_f32(activation_params[0]);
        const float32x4_t _max = vdupq_n_f32(activation_params[1]);
        _v = vmaxq_f32(_v, _min);
        _v = vminq_f32(_v, _max);
    }
    else if (activation_type == 4)
    {
        _v = sigmoid_ps(_v);
    }
    else if (activation_type == 5)
    {
        _v = vmulq_f32(_v, tanh_ps(log_ps(vaddq_f32(exp_ps(_v), vdupq_n_f32(1.f)))));
    }
    else if (activation_type == 6)
    {
        const float alpha = activation_params[0];
        const float beta = activation_params[1];
        const float32x4_t _zero = vdupq_n_f32(0.f);
        const float32x4_t _one = vdupq_n_f32(1.f);
        float32x4_t _ans = vdupq_n_f32(beta);
        _ans = vmlaq_n_f32(_ans, _v, alpha);
        _ans = vmaxq_f32(_ans, _zero);
        _ans = vminq_f32(_ans, _one);
        _v = vmulq_f32(_ans, _v);
    }

    return _v;
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "arm_usability.h"
#include "neon_mathfun_fp16s.h"

static inline __fp16 activation_ss_f16(__fp16 v, int activation_type, const ncnn::Mat& activation_params)
{
    if (activation_type == 1)
    {
        v = std::max(v, (__fp16)0.f);
    }
    else if (activation_type == 2)
    {
        const __fp16 slope = (__fp16)(activation_params[0]);
        v = v > 0.f ? v : v * slope;
    }
    else if (activation_type == 3)
    {
        const __fp16 min = (__fp16)(activation_params[0]);
        const __fp16 max = (__fp16)(activation_params[1]);
        if (v < min)
            v = min;
        if (v > max)
            v = max;
    }
    else if (activation_type == 4)
    {
        v = (__fp16)1.f / ((__fp16)1.f + (__fp16)expf(-v));
    }
    else if (activation_type == 5)
    {
        v = v * (__fp16)tanhf(logf(expf((float)v) + 1.f));
    }
    else if (activation_type == 6)
    {
        const __fp16 alpha = (__fp16)(activation_params[0]);
        const __fp16 beta = (__fp16)(activation_params[1]);
        const __fp16 lower = -beta / alpha;
        const __fp16 upper = ((__fp16)1.f / alpha) + lower;
        if (v < lower)
            v = (__fp16)0.f;
        else if (v > upper)
            ;
        else
            v = v * (v * alpha + beta);
    }

    return v;
}

static inline float16x4_t activation_ps_f16(float16x4_t _v, int activation_type, const ncnn::Mat& activation_params)
{
    if (activation_type == 1)
    {
        const float16x4_t _zero = vdup_n_f16(0.f);
        _v = vmax_f16(_v, _zero);
    }
    else if (activation_type == 2)
    {
        const float16x4_t _zero = vdup_n_f16(0.f);
#if _MSC_VER
        const float16x4_t _slope = vcvt_f16_f32(vdupq_n_f32(activation_params[0]));
#else
        const float16x4_t _slope = vdup_n_f16((__fp16)activation_params[0]);
#endif
        const uint16x4_t _lemask = vcle_f16(_v, _zero);
        float16x4_t _ps = vmul_f16(_v, _slope);
        _v = vbsl_f16(_lemask, _ps, _v);
    }
    else if (activation_type == 3)
    {
        const float16x4_t _min = vdup_n_f16((__fp16)activation_params[0]);
        const float16x4_t _max = vdup_n_f16((__fp16)activation_params[1]);
        _v = vmax_f16(_v, _min);
        _v = vmin_f16(_v, _max);
    }
    else if (activation_type == 4)
    {
        _v = sigmoid_ps_f16(_v);
    }
    else if (activation_type == 5)
    {
        _v = vmul_f16(_v, tanh_ps_f16(log_ps_f16(vadd_f16(exp_ps_f16(_v), vdup_n_f16(1.f)))));
    }
    else if (activation_type == 6)
    {
        const __fp16 alpha = (__fp16)activation_params[0];
        const __fp16 beta = (__fp16)activation_params[1];
        const float16x4_t _zero = vdup_n_f16(0.f);
        const float16x4_t _one = vdup_n_f16(1.f);
        float16x4_t _ans = vdup_n_f16(beta);
        _ans = vfma_n_f16(_ans, _v, alpha);
        _ans = vmax_f16(_ans, _zero);
        _ans = vmin_f16(_ans, _one);
        _v = vmul_f16(_ans, _v);
    }

    return _v;
}

static inline float16x8_t activation_ps_f16(float16x8_t _v, int activation_type, const ncnn::Mat& activation_params)
{
    if (activation_type == 1)
    {
        const float16x8_t _zero = vdupq_n_f16(0.f);
        _v = vmaxq_f16(_v, _zero);
    }
    else if (activation_type == 2)
    {
        const float16x8_t _zero = vdupq_n_f16(0.f);
#if _MSC_VER
        const float16x4_t _slope0 = vcvt_f16_f32(vdupq_n_f32(activation_params[0]));
        const float16x8_t _slope = vcombine_f16(_slope0, _slope0);
#else
        const float16x8_t _slope = vdupq_n_f16((__fp16)activation_params[0]);
#endif
        const uint16x8_t _lemask = vcleq_f16(_v, _zero);
        float16x8_t _ps = vmulq_f16(_v, _slope);
        _v = vbslq_f16(_lemask, _ps, _v);
    }
    else if (activation_type == 3)
    {
        const float16x8_t _min = vdupq_n_f16((__fp16)activation_params[0]);
        const float16x8_t _max = vdupq_n_f16((__fp16)activation_params[1]);
        _v = vmaxq_f16(_v, _min);
        _v = vminq_f16(_v, _max);
    }
    else if (activation_type == 4)
    {
        _v = sigmoid_ps_f16(_v);
    }
    else if (activation_type == 5)
    {
        _v = vmulq_f16(_v, tanh_ps_f16(log_ps_f16(vaddq_f16(exp_ps_f16(_v), vdupq_n_f16(1.f)))));
    }
    else if (activation_type == 6)
    {
        const __fp16 alpha_fp16 = (__fp16)activation_params[0];
        const __fp16 beta_fp16 = (__fp16)activation_params[1];
        const float16x8_t _zero = vdupq_n_f16(0.f);
        const float16x8_t _one = vdupq_n_f16(1.f);
        float16x8_t _ans = vdupq_n_f16(beta_fp16);
        _ans = vfmaq_n_f16(_ans, _v, alpha_fp16);
        _ans = vmaxq_f16(_ans, _zero);
        _ans = vminq_f16(_ans, _one);
        _v = vmulq_f16(_ans, _v);
    }
    return _v;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#endif // __ARM_NEON

#endif // ARM_ACTIVATION_H

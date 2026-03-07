// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
    else if (activation_type == 7)
    {
        const int fast_gelu = activation_params.row<int>(0)[0];
        if (fast_gelu == 0)
        {
            float32x4_t _erf = erf_ps(vmulq_n_f32(_v, 0.70710678f));
            _erf = vaddq_f32(vdupq_n_f32(1.f), _erf);
            _v = vmulq_n_f32(vmulq_f32(_erf, _v), 0.5f);
        }
        else
        {
            float32x4_t _blob = vmulq_f32(_v, _v);
            _blob = vmulq_f32(_v, _blob);
            _blob = vmulq_f32(vdupq_n_f32(0.044715f * 0.79788452f), _blob);
            _blob = vmlaq_f32(_blob, vdupq_n_f32(0.79788452f), _v);
            _blob = tanh_ps(_blob);
            _blob = vaddq_f32(vdupq_n_f32(1.f), _blob);
            _v = vmulq_f32(vdupq_n_f32(0.5f), vmulq_f32(_blob, _v));
        }
    }
    else if (activation_type == 8)
    {
        const float32x4_t _one = vdupq_n_f32(1.f);
        _v = div_ps(_v, vaddq_f32(_one, exp_ps(vnegq_f32(_v))));
    }
    else if (activation_type == 9)
    {
        const float alpha = activation_params[0];
        const float32x4_t _zero = vdupq_n_f32(0.f);
        const float32x4_t _one = vdupq_n_f32(1.f);
        const uint32x4_t _lemask = vcleq_f32(_v, _zero);
        float32x4_t _nps = exp_ps(_v);
        _nps = vsubq_f32(_nps, _one);
        _nps = vmulq_n_f32(_nps, alpha);
        _v = vbslq_f32(_lemask, _nps, _v);
    }
    else if (activation_type == 10)
    {
        const float alpha = 1.67326324f;
        const float lambda = 1.050700987f;
        const float alphaxlambda = alpha * lambda;
        const float32x4_t _zero = vdupq_n_f32(0.f);
        const float32x4_t _one = vdupq_n_f32(1.f);
        const float32x4_t _alphaxlambda = vdupq_n_f32(alphaxlambda);
        const float32x4_t _lambda = vdupq_n_f32(lambda);
        const uint32x4_t _lemask = vcleq_f32(_v, _zero);
        float32x4_t _nps = exp_ps(_v);
        _nps = vsubq_f32(_nps, _one);
        _nps = vmulq_f32(_nps, _alphaxlambda);
        float32x4_t _pps = vmulq_f32(_v, _lambda);
        _v = vbslq_f32(_lemask, _nps, _pps);
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
    else if (activation_type == 7)
    {
        const int fast_gelu = activation_params.row<int>(0)[0];
        if (fast_gelu == 0)
        {
            v = 0.5f * v * (1.0f + erff((float)v * 0.70710678f));
        }
        else
        {
            v = 0.5f * v * (1.0f + tanhf(0.79788452f * (v + 0.044715f * v * v * v)));
        }
    }
    else if (activation_type == 8)
    {
        v = v / ((__fp16)1.f + (__fp16)expf(-v));
    }
    else if (activation_type == 9)
    {
        const __fp16 alpha = (__fp16)(activation_params[0]);
        if (v < (__fp16)0.f)
            v = (expf(v) - 1.f) * alpha;
    }
    else if (activation_type == 10)
    {
        const __fp16 alpha = 1.67326324f;
        const __fp16 lambda = 1.050700987f;
        const __fp16 alphaxlambda = alpha * lambda;
        if (v < (__fp16)0.f)
            v = (expf(v) - 1.f) * alphaxlambda;
        else
            v = v * lambda;
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
#if defined(_MSC_VER) && !defined(__clang__)
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
    else if (activation_type == 7)
    {
        const int fast_gelu = activation_params.row<int>(0)[0];
        if (fast_gelu == 0)
        {
            float32x4_t _v_f32 = vcvt_f32_f16(_v);
            float32x4_t _erf = erf_ps(vmulq_n_f32(_v_f32, 0.70710678f));
            _erf = vaddq_f32(vdupq_n_f32(1.f), _erf);
            _v_f32 = vmulq_n_f32(vmulq_f32(_erf, _v_f32), 0.5f);
            _v = vcvt_f16_f32(_v_f32);
        }
        else
        {
            float16x4_t _blob = vmul_f16(_v, _v);
            _blob = vmul_f16(_v, _blob);
            _blob = vmul_f16(vdup_n_f16(0.044715f * 0.79788452f), _blob);
            _blob = vfma_f16(_blob, vdup_n_f16(0.79788452f), _v);
            _blob = tanh_ps_f16(_blob);
            _blob = vadd_f16(vdup_n_f16(1.f), _blob);
            _v = vmul_f16(vdup_n_f16(0.5f), vmul_f16(_blob, _v));
        }
    }
    else if (activation_type == 8)
    {
        const float16x4_t _one = vdup_n_f16(1.f);
        _v = vdiv_f16(_v, vadd_f16(_one, exp_ps_f16(vneg_f16(_v))));
    }
    else if (activation_type == 9)
    {
        const __fp16 alpha = (__fp16)activation_params[0];
        const float16x4_t _zero = vdup_n_f16(0.f);
        const float16x4_t _one = vdup_n_f16(1.f);
        const uint16x4_t _lemask = vcle_f16(_v, _zero);
        float16x4_t _nps = exp_ps_f16(_v);
        _nps = vsub_f16(_nps, _one);
        _nps = vmul_n_f16(_nps, alpha);
        _v = vbsl_f16(_lemask, _nps, _v);
    }
    else if (activation_type == 10)
    {
        const __fp16 alpha = 1.67326324f;
        const __fp16 lambda = 1.050700987f;
        const __fp16 alphaxlambda = alpha * lambda;
        const float16x4_t _zero = vdup_n_f16(0.f);
        const float16x4_t _one = vdup_n_f16(1.f);
        const float16x4_t _alphaxlambda = vdup_n_f16(alphaxlambda);
        const float16x4_t _lambda = vdup_n_f16(lambda);
        const uint16x4_t _lemask = vcle_f16(_v, _zero);
        float16x4_t _nps = exp_ps_f16(_v);
        _nps = vsub_f16(_nps, _one);
        _nps = vmul_f16(_nps, _alphaxlambda);
        float16x4_t _pps = vmul_f16(_v, _lambda);
        _v = vbsl_f16(_lemask, _nps, _pps);
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
#if defined(_MSC_VER) && !defined(__clang__)
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
    else if (activation_type == 7)
    {
        const int fast_gelu = activation_params.row<int>(0)[0];
        if (fast_gelu == 0)
        {
            float32x4_t _v0_f32 = vcvt_f32_f16(vget_low_f16(_v));
            float32x4_t _v1_f32 = vcvt_f32_f16(vget_high_f16(_v));
            float32x4_t _erf0 = erf_ps(vmulq_n_f32(_v0_f32, 0.70710678f));
            float32x4_t _erf1 = erf_ps(vmulq_n_f32(_v1_f32, 0.70710678f));
            _erf0 = vaddq_f32(vdupq_n_f32(1.f), _erf0);
            _erf1 = vaddq_f32(vdupq_n_f32(1.f), _erf1);
            _v0_f32 = vmulq_n_f32(vmulq_f32(_erf0, _v0_f32), 0.5f);
            _v1_f32 = vmulq_n_f32(vmulq_f32(_erf1, _v1_f32), 0.5f);
            _v = vcombine_f16(vcvt_f16_f32(_v0_f32), vcvt_f16_f32(_v1_f32));
        }
        else
        {
            float16x8_t _blob = vmulq_f16(_v, _v);
            _blob = vmulq_f16(_v, _blob);
            _blob = vmulq_f16(vdupq_n_f16(0.044715f * 0.79788452f), _blob);
            _blob = vfmaq_f16(_blob, vdupq_n_f16(0.79788452f), _v);
            _blob = tanh_ps_f16(_blob);
            _blob = vaddq_f16(vdupq_n_f16(1.f), _blob);
            _v = vmulq_f16(vdupq_n_f16(0.5f), vmulq_f16(_blob, _v));
        }
    }
    else if (activation_type == 8)
    {
        const float16x8_t _one = vdupq_n_f16(1.f);
        _v = vdivq_f16(_v, vaddq_f16(_one, exp_ps_f16(vnegq_f16(_v))));
    }
    else if (activation_type == 9)
    {
        const __fp16 alpha = (__fp16)activation_params[0];
        const float16x8_t _zero = vdupq_n_f16(0.f);
        const float16x8_t _one = vdupq_n_f16(1.f);
        const uint16x8_t _lemask = vcleq_f16(_v, _zero);
        float16x8_t _nps = exp_ps_f16(_v);
        _nps = vsubq_f16(_nps, _one);
        _nps = vmulq_n_f16(_nps, alpha);
        _v = vbslq_f16(_lemask, _nps, _v);
    }
    else if (activation_type == 10)
    {
        const __fp16 alpha = 1.67326324f;
        const __fp16 lambda = 1.050700987f;
        const __fp16 alphaxlambda = alpha * lambda;
        const float16x8_t _zero = vdupq_n_f16(0.f);
        const float16x8_t _one = vdupq_n_f16(1.f);
        const float16x8_t _alphaxlambda = vdupq_n_f16(alphaxlambda);
        const float16x8_t _lambda = vdupq_n_f16(lambda);
        const uint16x8_t _lemask = vcleq_f16(_v, _zero);
        float16x8_t _nps = exp_ps_f16(_v);
        _nps = vsubq_f16(_nps, _one);
        _nps = vmulq_f16(_nps, _alphaxlambda);
        float16x8_t _pps = vmulq_f16(_v, _lambda);
        _v = vbslq_f16(_lemask, _nps, _pps);
    }
    return _v;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#endif // __ARM_NEON

#endif // ARM_ACTIVATION_H

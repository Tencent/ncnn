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

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static inline float activation_ss(float v, int activation_type, const ncnn::Mat& activation_params)
{
    if (activation_type == 1)
    {
        v = std::max(v, 0.f);
    }
    else if (activation_type == 2)
    {
        float slope = activation_params[0];
        v = v > 0.f ? v : v * slope;
    }
    else if (activation_type == 3)
    {
        float min = activation_params[0];
        float max = activation_params[1];
        if (v < min)
            v = min;
        if (v > max)
            v = max;
    }
    else if (activation_type == 4)
    {
        v = 1.f / (1.f + exp(-v));
    }
    else if (activation_type == 5)
    {
        v = v * tanh(log(exp(v) + 1.f));
    }

    return v;
}

#if __ARM_NEON
static inline float32x4_t activation_ps(float32x4_t _v, int activation_type, const ncnn::Mat& activation_params)
{
    if (activation_type == 1)
    {
        float32x4_t _zero = vdupq_n_f32(0.f);
        _v = vmaxq_f32(_v, _zero);
    }
    else if (activation_type == 2)
    {
        float32x4_t _zero = vdupq_n_f32(0.f);
        float32x4_t _slope = vdupq_n_f32(activation_params[0]);
        uint32x4_t _lemask = vcleq_f32(_v, _zero);
        float32x4_t _ps = vmulq_f32(_v, _slope);
        _v = vbslq_f32(_lemask, _ps, _v);
    }
    else if (activation_type == 3)
    {
        float32x4_t _min = vdupq_n_f32(activation_params[0]);
        float32x4_t _max = vdupq_n_f32(activation_params[1]);
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

    return _v;
}
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static inline __fp16 activation_ss(__fp16 v, int activation_type, const ncnn::Mat& activation_params)
{
    float v32 = v;
    v32 = activation_ss(v32, activation_type, activation_params);
    return (__fp16)v32;
}

static inline float16x4_t activation_ps(float16x4_t _v, int activation_type, const ncnn::Mat& activation_params)
{
    float32x4_t _v32 = vcvt_f32_f16(_v);
    _v32 = activation_ps(_v32, activation_type, activation_params);
    return vcvt_f16_f32(_v32);
}

static inline float16x8_t activation_ps(float16x8_t _v, int activation_type, const ncnn::Mat& activation_params)
{
    float32x4_t _v32_low = vcvt_f32_f16(vget_low_f16(_v));
    float32x4_t _v32_high = vcvt_f32_f16(vget_high_f16(_v));
    _v32_low = activation_ps(_v32_low, activation_type, activation_params);
    _v32_high = activation_ps(_v32_high, activation_type, activation_params);
    return vcombine_f16(vcvt_f16_f32(_v32_low), vcvt_f16_f32(_v32_high));
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#endif // __ARM_NEON

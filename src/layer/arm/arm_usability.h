// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef ARM_USABILITY_H
#define ARM_USABILITY_H

static inline signed char float2int8(float v)
{
    int int32 = round(v);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

#if __ARM_NEON
#include <arm_neon.h>

static inline int8x8_t float2int8(float32x4_t _vlow, float32x4_t _vhigh)
{
#if __aarch64__
    int32x4_t _vlow32 = vcvtaq_s32_f32(_vlow);
    int32x4_t _vhigh32 = vcvtaq_s32_f32(_vhigh);
#else
    // vcvtq_s32_f32 is round to zero
    // simulate round to nearest via +/-0.5
    float32x4_t _p5 = vdupq_n_f32(0.5f);
    int32x4_t _signmask = vdupq_n_s32(1 << 31);
    int32x4_t _signlow = vandq_s32(vreinterpretq_s32_f32(_vlow), _signmask);
    int32x4_t _signhigh = vandq_s32(vreinterpretq_s32_f32(_vhigh), _signmask);
    float32x4_t _p5low = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(_p5), _signlow));
    float32x4_t _p5high = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(_p5), _signhigh));
    float32x4_t _vlow5 = vaddq_f32(_vlow, _p5low);
    float32x4_t _vhigh5 = vaddq_f32(_vhigh, _p5high);
    int32x4_t _vlow32 = vcvtq_s32_f32(_vlow5);
    int32x4_t _vhigh32 = vcvtq_s32_f32(_vhigh5);
#endif
    int16x8_t _v16 = vcombine_s16(vqmovn_s32(_vlow32), vqmovn_s32(_vhigh32));
    int8x8_t _v8 = vqmovn_s16(_v16);
    return vmax_s8(_v8, vdup_n_s8(-127));
}

static inline int8x8_t float2int8relu(float32x4_t _vlow, float32x4_t _vhigh)
{
#if __aarch64__
    int32x4_t _vlow32 = vcvtaq_s32_f32(_vlow);
    int32x4_t _vhigh32 = vcvtaq_s32_f32(_vhigh);
#else
    // vcvtq_s32_f32 is round to zero
    // simulate round to nearest via +/-0.5
    float32x4_t _p5 = vdupq_n_f32(0.5f);
    int32x4_t _signmask = vdupq_n_s32(1 << 31);
    int32x4_t _signlow = vandq_s32(vreinterpretq_s32_f32(_vlow), _signmask);
    int32x4_t _signhigh = vandq_s32(vreinterpretq_s32_f32(_vhigh), _signmask);
    float32x4_t _p5low = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(_p5), _signlow));
    float32x4_t _p5high = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(_p5), _signhigh));
    float32x4_t _vlow5 = vaddq_f32(_vlow, _p5low);
    float32x4_t _vhigh5 = vaddq_f32(_vhigh, _p5high);
    int32x4_t _vlow32 = vcvtq_s32_f32(_vlow5);
    int32x4_t _vhigh32 = vcvtq_s32_f32(_vhigh5);
#endif
    int16x8_t _v16 = vcombine_s16(vqmovn_s32(_vlow32), vqmovn_s32(_vhigh32));
    int8x8_t _v8 = vqmovn_s16(_v16);
    return vmax_s8(_v8, vdup_n_s8(0));
}

static inline int8x8_t float2int8leakyrelu(float32x4_t _vlow, float32x4_t _vhigh, float32x4_t _slope)
{
    float32x4_t _vlow_leaky = vmulq_f32(_vlow, _slope);
    float32x4_t _vhigh_leaky = vmulq_f32(_vhigh, _slope);
#if __aarch64__
    int32x4_t _vlow32 = vcvtaq_s32_f32(_vlow);
    int32x4_t _vhigh32 = vcvtaq_s32_f32(_vhigh);
    int32x4_t _vlow32_leaky = vcvtaq_s32_f32(_vlow_leaky);
    int32x4_t _vhigh32_leaky = vcvtaq_s32_f32(_vhigh_leaky);
#else
    // vcvtq_s32_f32 is round to zero
    // simulate round to nearest via +/-0.5
    float32x4_t _p5 = vdupq_n_f32(0.5f);
    int32x4_t _signmask = vdupq_n_s32(1 << 31);
    int32x4_t _signlow = vandq_s32(vreinterpretq_s32_f32(_vlow), _signmask);
    int32x4_t _signhigh = vandq_s32(vreinterpretq_s32_f32(_vhigh), _signmask);
    float32x4_t _p5low = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(_p5), _signlow));
    float32x4_t _p5high = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(_p5), _signhigh));
    float32x4_t _vlow5 = vaddq_f32(_vlow, _p5low);
    float32x4_t _vhigh5 = vaddq_f32(_vhigh, _p5high);
    int32x4_t _vlow32 = vcvtq_s32_f32(_vlow5);
    int32x4_t _vhigh32 = vcvtq_s32_f32(_vhigh5);

    int32x4_t _signlow_leaky = vandq_s32(vreinterpretq_s32_f32(_vlow_leaky), _signmask);
    int32x4_t _signhigh_leaky = vandq_s32(vreinterpretq_s32_f32(_vhigh_leaky), _signmask);
    float32x4_t _p5low_leaky = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(_p5), _signlow_leaky));
    float32x4_t _p5high_leaky = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(_p5), _signhigh_leaky));
    float32x4_t _vlow5_leaky = vaddq_f32(_vlow_leaky, _p5low_leaky);
    float32x4_t _vhigh5_leaky = vaddq_f32(_vhigh_leaky, _p5high_leaky);
    int32x4_t _vlow32_leaky = vcvtq_s32_f32(_vlow5_leaky);
    int32x4_t _vhigh32_leaky = vcvtq_s32_f32(_vhigh5_leaky);
#endif
    int16x8_t _v16 = vcombine_s16(vqmovn_s32(_vlow32), vqmovn_s32(_vhigh32));
    int16x8_t _v16_leaky = vcombine_s16(vqmovn_s32(_vlow32_leaky), vqmovn_s32(_vhigh32_leaky));
    int8x8_t _v8 = vqmovn_s16(_v16);
    int8x8_t _v8_leaky = vqmovn_s16(_v16_leaky);
    return vmax_s8(_v8, _v8_leaky);
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static inline signed char float2int8(__fp16 v)
{
    int int32 = round(v);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static inline int8x8_t float2int8(float16x8_t _v)
{
    int16x8_t _v16 = vcvtaq_s16_f16(_v);
    int8x8_t _v8 = vqmovn_s16(_v16);
    return vmax_s8(_v8, vdup_n_s8(-127));
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#endif // __ARM_NEON

#endif // ARM_USABILITY_H

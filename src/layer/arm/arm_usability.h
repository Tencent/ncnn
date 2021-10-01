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

class FPSCRGuard
{
#if !__aarch64__
public:
    FPSCRGuard()
    {
        // set round mode to round to nearest for armv7
        // this impacts all vcvtr instructions in guarded scope
        asm volatile(
            "vmrs   %0, FPSCR               \n"
            "bic    r10, %0, #0x00c00000    \n"
            "vmsr   FPSCR, r10              \n"
            : "=r"(FPSCR_value)
            :
            : "memory", "r10");
    }

    ~FPSCRGuard()
    {
        asm volatile(
            "vmsr   FPSCR, %0           \n"
            :
            : "r"(FPSCR_value)
            : "memory");
    }

private:
    int FPSCR_value;
#endif // !__aarch64__
};

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
    // use vcvtr.s32.f32
    int32x4_t _vlow32 = int32x4_t();
    int32x4_t _vhigh32 = int32x4_t();
    _vlow32 = vsetq_lane_s32(round(vgetq_lane_f32(_vlow, 0)), _vlow32, 0);
    _vlow32 = vsetq_lane_s32(round(vgetq_lane_f32(_vlow, 1)), _vlow32, 1);
    _vlow32 = vsetq_lane_s32(round(vgetq_lane_f32(_vlow, 2)), _vlow32, 2);
    _vlow32 = vsetq_lane_s32(round(vgetq_lane_f32(_vlow, 3)), _vlow32, 3);
    _vhigh32 = vsetq_lane_s32(round(vgetq_lane_f32(_vhigh, 0)), _vhigh32, 0);
    _vhigh32 = vsetq_lane_s32(round(vgetq_lane_f32(_vhigh, 1)), _vhigh32, 1);
    _vhigh32 = vsetq_lane_s32(round(vgetq_lane_f32(_vhigh, 2)), _vhigh32, 2);
    _vhigh32 = vsetq_lane_s32(round(vgetq_lane_f32(_vhigh, 3)), _vhigh32, 3);
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
    // use vcvtr.s32.f32
    int32x4_t _vlow32 = int32x4_t();
    int32x4_t _vhigh32 = int32x4_t();
    _vlow32 = vsetq_lane_s32(round(vgetq_lane_f32(_vlow, 0)), _vlow32, 0);
    _vlow32 = vsetq_lane_s32(round(vgetq_lane_f32(_vlow, 1)), _vlow32, 1);
    _vlow32 = vsetq_lane_s32(round(vgetq_lane_f32(_vlow, 2)), _vlow32, 2);
    _vlow32 = vsetq_lane_s32(round(vgetq_lane_f32(_vlow, 3)), _vlow32, 3);
    _vhigh32 = vsetq_lane_s32(round(vgetq_lane_f32(_vhigh, 0)), _vhigh32, 0);
    _vhigh32 = vsetq_lane_s32(round(vgetq_lane_f32(_vhigh, 1)), _vhigh32, 1);
    _vhigh32 = vsetq_lane_s32(round(vgetq_lane_f32(_vhigh, 2)), _vhigh32, 2);
    _vhigh32 = vsetq_lane_s32(round(vgetq_lane_f32(_vhigh, 3)), _vhigh32, 3);
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
    // use vcvtr.s32.f32
    int32x4_t _vlow32 = int32x4_t();
    int32x4_t _vhigh32 = int32x4_t();
    _vlow32 = vsetq_lane_s32(round(vgetq_lane_f32(_vlow, 0)), _vlow32, 0);
    _vlow32 = vsetq_lane_s32(round(vgetq_lane_f32(_vlow, 1)), _vlow32, 1);
    _vlow32 = vsetq_lane_s32(round(vgetq_lane_f32(_vlow, 2)), _vlow32, 2);
    _vlow32 = vsetq_lane_s32(round(vgetq_lane_f32(_vlow, 3)), _vlow32, 3);
    _vhigh32 = vsetq_lane_s32(round(vgetq_lane_f32(_vhigh, 0)), _vhigh32, 0);
    _vhigh32 = vsetq_lane_s32(round(vgetq_lane_f32(_vhigh, 1)), _vhigh32, 1);
    _vhigh32 = vsetq_lane_s32(round(vgetq_lane_f32(_vhigh, 2)), _vhigh32, 2);
    _vhigh32 = vsetq_lane_s32(round(vgetq_lane_f32(_vhigh, 3)), _vhigh32, 3);
    int32x4_t _vlow32_leaky = int32x4_t();
    int32x4_t _vhigh32_leaky = int32x4_t();
    _vlow32_leaky = vsetq_lane_s32(round(vgetq_lane_f32(_vlow_leaky, 0)), _vlow32_leaky, 0);
    _vlow32_leaky = vsetq_lane_s32(round(vgetq_lane_f32(_vlow_leaky, 1)), _vlow32_leaky, 1);
    _vlow32_leaky = vsetq_lane_s32(round(vgetq_lane_f32(_vlow_leaky, 2)), _vlow32_leaky, 2);
    _vlow32_leaky = vsetq_lane_s32(round(vgetq_lane_f32(_vlow_leaky, 3)), _vlow32_leaky, 3);
    _vhigh32_leaky = vsetq_lane_s32(round(vgetq_lane_f32(_vhigh_leaky, 0)), _vhigh32_leaky, 0);
    _vhigh32_leaky = vsetq_lane_s32(round(vgetq_lane_f32(_vhigh_leaky, 1)), _vhigh32_leaky, 1);
    _vhigh32_leaky = vsetq_lane_s32(round(vgetq_lane_f32(_vhigh_leaky, 2)), _vhigh32_leaky, 2);
    _vhigh32_leaky = vsetq_lane_s32(round(vgetq_lane_f32(_vhigh_leaky, 3)), _vhigh32_leaky, 3);
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

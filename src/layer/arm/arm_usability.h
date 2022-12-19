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

static inline uint16x4_t bfloat2float(float32x4_t _v)
{
    return vshrn_n_u32(vreinterpretq_u32_f32(_v), 16);
}
static inline float32x4_t float2bfloat(uint16x4_t _v)
{
    return vreinterpretq_f32_u32(vshll_n_u16(_v, 16));
}

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

static inline void transpose4x4_ps(float32x4_t& _r0, float32x4_t& _r1, float32x4_t& _r2, float32x4_t& _r3)
{
    float32x4x2_t _r01t = vtrnq_f32(_r0, _r1);
    float32x4x2_t _r23t = vtrnq_f32(_r2, _r3);
    _r0 = vcombine_f32(vget_low_f32(_r01t.val[0]), vget_low_f32(_r23t.val[0]));
    _r1 = vcombine_f32(vget_low_f32(_r01t.val[1]), vget_low_f32(_r23t.val[1]));
    _r2 = vcombine_f32(vget_high_f32(_r01t.val[0]), vget_high_f32(_r23t.val[0]));
    _r3 = vcombine_f32(vget_high_f32(_r01t.val[1]), vget_high_f32(_r23t.val[1]));
}

static inline void transpose4x8_ps(float32x4_t& _r0, float32x4_t& _r1, float32x4_t& _r2, float32x4_t& _r3, float32x4_t& _r4, float32x4_t& _r5, float32x4_t& _r6, float32x4_t& _r7)
{
    float32x4x2_t _r01t = vtrnq_f32(_r0, _r1);
    float32x4x2_t _r23t = vtrnq_f32(_r2, _r3);
    float32x4x2_t _r45t = vtrnq_f32(_r4, _r5);
    float32x4x2_t _r67t = vtrnq_f32(_r6, _r7);
    _r0 = vcombine_f32(vget_low_f32(_r01t.val[0]), vget_low_f32(_r23t.val[0]));
    _r2 = vcombine_f32(vget_low_f32(_r01t.val[1]), vget_low_f32(_r23t.val[1]));
    _r4 = vcombine_f32(vget_high_f32(_r01t.val[0]), vget_high_f32(_r23t.val[0]));
    _r6 = vcombine_f32(vget_high_f32(_r01t.val[1]), vget_high_f32(_r23t.val[1]));
    _r1 = vcombine_f32(vget_low_f32(_r45t.val[0]), vget_low_f32(_r67t.val[0]));
    _r3 = vcombine_f32(vget_low_f32(_r45t.val[1]), vget_low_f32(_r67t.val[1]));
    _r5 = vcombine_f32(vget_high_f32(_r45t.val[0]), vget_high_f32(_r67t.val[0]));
    _r7 = vcombine_f32(vget_high_f32(_r45t.val[1]), vget_high_f32(_r67t.val[1]));
}

static inline void transpose4x12_ps(float32x4_t& _r0, float32x4_t& _r1, float32x4_t& _r2, float32x4_t& _r3, float32x4_t& _r4, float32x4_t& _r5, float32x4_t& _r6, float32x4_t& _r7, float32x4_t& _r8, float32x4_t& _r9, float32x4_t& _ra, float32x4_t& _rb)
{
    float32x4x2_t _r01t = vtrnq_f32(_r0, _r1);
    float32x4x2_t _r23t = vtrnq_f32(_r2, _r3);
    float32x4x2_t _r45t = vtrnq_f32(_r4, _r5);
    float32x4x2_t _r67t = vtrnq_f32(_r6, _r7);
    float32x4x2_t _r89t = vtrnq_f32(_r8, _r9);
    float32x4x2_t _rabt = vtrnq_f32(_ra, _rb);
    _r0 = vcombine_f32(vget_low_f32(_r01t.val[0]), vget_low_f32(_r23t.val[0]));
    _r3 = vcombine_f32(vget_low_f32(_r01t.val[1]), vget_low_f32(_r23t.val[1]));
    _r6 = vcombine_f32(vget_high_f32(_r01t.val[0]), vget_high_f32(_r23t.val[0]));
    _r9 = vcombine_f32(vget_high_f32(_r01t.val[1]), vget_high_f32(_r23t.val[1]));
    _r1 = vcombine_f32(vget_low_f32(_r45t.val[0]), vget_low_f32(_r67t.val[0]));
    _r4 = vcombine_f32(vget_low_f32(_r45t.val[1]), vget_low_f32(_r67t.val[1]));
    _r7 = vcombine_f32(vget_high_f32(_r45t.val[0]), vget_high_f32(_r67t.val[0]));
    _ra = vcombine_f32(vget_high_f32(_r45t.val[1]), vget_high_f32(_r67t.val[1]));
    _r2 = vcombine_f32(vget_low_f32(_r89t.val[0]), vget_low_f32(_rabt.val[0]));
    _r5 = vcombine_f32(vget_low_f32(_r89t.val[1]), vget_low_f32(_rabt.val[1]));
    _r8 = vcombine_f32(vget_high_f32(_r89t.val[0]), vget_high_f32(_rabt.val[0]));
    _rb = vcombine_f32(vget_high_f32(_r89t.val[1]), vget_high_f32(_rabt.val[1]));
}

static inline void transpose8x4_ps(float32x4_t& _r0l, float32x4_t& _r0h,
                                   float32x4_t& _r1l, float32x4_t& _r1h,
                                   float32x4_t& _r2l, float32x4_t& _r2h,
                                   float32x4_t& _r3l, float32x4_t& _r3h)
{
    float32x4x2_t _r01lt = vtrnq_f32(_r0l, _r1l);
    float32x4x2_t _r23lt = vtrnq_f32(_r2l, _r3l);
    float32x4x2_t _r01ht = vtrnq_f32(_r0h, _r1h);
    float32x4x2_t _r23ht = vtrnq_f32(_r2h, _r3h);
    _r0l = vcombine_f32(vget_low_f32(_r01lt.val[0]), vget_low_f32(_r23lt.val[0]));
    _r0h = vcombine_f32(vget_low_f32(_r01lt.val[1]), vget_low_f32(_r23lt.val[1]));
    _r1l = vcombine_f32(vget_high_f32(_r01lt.val[0]), vget_high_f32(_r23lt.val[0]));
    _r1h = vcombine_f32(vget_high_f32(_r01lt.val[1]), vget_high_f32(_r23lt.val[1]));
    _r2l = vcombine_f32(vget_low_f32(_r01ht.val[0]), vget_low_f32(_r23ht.val[0]));
    _r2h = vcombine_f32(vget_low_f32(_r01ht.val[1]), vget_low_f32(_r23ht.val[1]));
    _r3l = vcombine_f32(vget_high_f32(_r01ht.val[0]), vget_high_f32(_r23ht.val[0]));
    _r3h = vcombine_f32(vget_high_f32(_r01ht.val[1]), vget_high_f32(_r23ht.val[1]));
}

static inline void transpose12x4_ps(float32x4_t& _r0l, float32x4_t& _r0m, float32x4_t& _r0h,
                                    float32x4_t& _r1l, float32x4_t& _r1m, float32x4_t& _r1h,
                                    float32x4_t& _r2l, float32x4_t& _r2m, float32x4_t& _r2h,
                                    float32x4_t& _r3l, float32x4_t& _r3m, float32x4_t& _r3h)
{
    float32x4x2_t _r01lt = vtrnq_f32(_r0l, _r1l);
    float32x4x2_t _r23lt = vtrnq_f32(_r2l, _r3l);
    float32x4x2_t _r01mt = vtrnq_f32(_r0m, _r1m);
    float32x4x2_t _r23mt = vtrnq_f32(_r2m, _r3m);
    float32x4x2_t _r01ht = vtrnq_f32(_r0h, _r1h);
    float32x4x2_t _r23ht = vtrnq_f32(_r2h, _r3h);
    _r0l = vcombine_f32(vget_low_f32(_r01lt.val[0]), vget_low_f32(_r23lt.val[0]));
    _r0m = vcombine_f32(vget_low_f32(_r01lt.val[1]), vget_low_f32(_r23lt.val[1]));
    _r0h = vcombine_f32(vget_high_f32(_r01lt.val[0]), vget_high_f32(_r23lt.val[0]));
    _r1l = vcombine_f32(vget_high_f32(_r01lt.val[1]), vget_high_f32(_r23lt.val[1]));
    _r1m = vcombine_f32(vget_low_f32(_r01mt.val[0]), vget_low_f32(_r23mt.val[0]));
    _r1h = vcombine_f32(vget_low_f32(_r01mt.val[1]), vget_low_f32(_r23mt.val[1]));
    _r2l = vcombine_f32(vget_high_f32(_r01mt.val[0]), vget_high_f32(_r23mt.val[0]));
    _r2m = vcombine_f32(vget_high_f32(_r01mt.val[1]), vget_high_f32(_r23mt.val[1]));
    _r2h = vcombine_f32(vget_low_f32(_r01ht.val[0]), vget_low_f32(_r23ht.val[0]));
    _r3l = vcombine_f32(vget_low_f32(_r01ht.val[1]), vget_low_f32(_r23ht.val[1]));
    _r3m = vcombine_f32(vget_high_f32(_r01ht.val[0]), vget_high_f32(_r23ht.val[0]));
    _r3h = vcombine_f32(vget_high_f32(_r01ht.val[1]), vget_high_f32(_r23ht.val[1]));
}

#if __aarch64__
static inline void transpose8x8_ps(float32x4_t& _r0l, float32x4_t& _r0h,
                                   float32x4_t& _r1l, float32x4_t& _r1h,
                                   float32x4_t& _r2l, float32x4_t& _r2h,
                                   float32x4_t& _r3l, float32x4_t& _r3h,
                                   float32x4_t& _r4l, float32x4_t& _r4h,
                                   float32x4_t& _r5l, float32x4_t& _r5h,
                                   float32x4_t& _r6l, float32x4_t& _r6h,
                                   float32x4_t& _r7l, float32x4_t& _r7h)
{
    float32x4x2_t _r01lt = vtrnq_f32(_r0l, _r1l);
    float32x4x2_t _r23lt = vtrnq_f32(_r2l, _r3l);
    float32x4x2_t _r01ht = vtrnq_f32(_r0h, _r1h);
    float32x4x2_t _r23ht = vtrnq_f32(_r2h, _r3h);
    float32x4x2_t _r45lt = vtrnq_f32(_r4l, _r5l);
    float32x4x2_t _r67lt = vtrnq_f32(_r6l, _r7l);
    float32x4x2_t _r45ht = vtrnq_f32(_r4h, _r5h);
    float32x4x2_t _r67ht = vtrnq_f32(_r6h, _r7h);
    _r0l = vcombine_f32(vget_low_f32(_r01lt.val[0]), vget_low_f32(_r23lt.val[0]));
    _r1l = vcombine_f32(vget_low_f32(_r01lt.val[1]), vget_low_f32(_r23lt.val[1]));
    _r2l = vcombine_f32(vget_high_f32(_r01lt.val[0]), vget_high_f32(_r23lt.val[0]));
    _r3l = vcombine_f32(vget_high_f32(_r01lt.val[1]), vget_high_f32(_r23lt.val[1]));
    _r0h = vcombine_f32(vget_low_f32(_r45lt.val[0]), vget_low_f32(_r67lt.val[0]));
    _r1h = vcombine_f32(vget_low_f32(_r45lt.val[1]), vget_low_f32(_r67lt.val[1]));
    _r2h = vcombine_f32(vget_high_f32(_r45lt.val[0]), vget_high_f32(_r67lt.val[0]));
    _r3h = vcombine_f32(vget_high_f32(_r45lt.val[1]), vget_high_f32(_r67lt.val[1]));
    _r4l = vcombine_f32(vget_low_f32(_r01ht.val[0]), vget_low_f32(_r23ht.val[0]));
    _r5l = vcombine_f32(vget_low_f32(_r01ht.val[1]), vget_low_f32(_r23ht.val[1]));
    _r6l = vcombine_f32(vget_high_f32(_r01ht.val[0]), vget_high_f32(_r23ht.val[0]));
    _r7l = vcombine_f32(vget_high_f32(_r01ht.val[1]), vget_high_f32(_r23ht.val[1]));
    _r4h = vcombine_f32(vget_low_f32(_r45ht.val[0]), vget_low_f32(_r67ht.val[0]));
    _r5h = vcombine_f32(vget_low_f32(_r45ht.val[1]), vget_low_f32(_r67ht.val[1]));
    _r6h = vcombine_f32(vget_high_f32(_r45ht.val[0]), vget_high_f32(_r67ht.val[0]));
    _r7h = vcombine_f32(vget_high_f32(_r45ht.val[1]), vget_high_f32(_r67ht.val[1]));
}

static inline void transpose8x12_ps(float32x4_t& _r0l, float32x4_t& _r0h,
                                    float32x4_t& _r1l, float32x4_t& _r1h,
                                    float32x4_t& _r2l, float32x4_t& _r2h,
                                    float32x4_t& _r3l, float32x4_t& _r3h,
                                    float32x4_t& _r4l, float32x4_t& _r4h,
                                    float32x4_t& _r5l, float32x4_t& _r5h,
                                    float32x4_t& _r6l, float32x4_t& _r6h,
                                    float32x4_t& _r7l, float32x4_t& _r7h,
                                    float32x4_t& _r8l, float32x4_t& _r8h,
                                    float32x4_t& _r9l, float32x4_t& _r9h,
                                    float32x4_t& _ral, float32x4_t& _rah,
                                    float32x4_t& _rbl, float32x4_t& _rbh)
{
    float32x4x2_t _r01lt = vtrnq_f32(_r0l, _r1l);
    float32x4x2_t _r23lt = vtrnq_f32(_r2l, _r3l);
    float32x4x2_t _r01ht = vtrnq_f32(_r0h, _r1h);
    float32x4x2_t _r23ht = vtrnq_f32(_r2h, _r3h);
    float32x4x2_t _r45lt = vtrnq_f32(_r4l, _r5l);
    float32x4x2_t _r67lt = vtrnq_f32(_r6l, _r7l);
    float32x4x2_t _r45ht = vtrnq_f32(_r4h, _r5h);
    float32x4x2_t _r67ht = vtrnq_f32(_r6h, _r7h);
    float32x4x2_t _r89lt = vtrnq_f32(_r8l, _r9l);
    float32x4x2_t _rablt = vtrnq_f32(_ral, _rbl);
    float32x4x2_t _r89ht = vtrnq_f32(_r8h, _r9h);
    float32x4x2_t _rabht = vtrnq_f32(_rah, _rbh);
    _r0l = vcombine_f32(vget_low_f32(_r01lt.val[0]), vget_low_f32(_r23lt.val[0]));
    _r1h = vcombine_f32(vget_low_f32(_r01lt.val[1]), vget_low_f32(_r23lt.val[1]));
    _r3l = vcombine_f32(vget_high_f32(_r01lt.val[0]), vget_high_f32(_r23lt.val[0]));
    _r4h = vcombine_f32(vget_high_f32(_r01lt.val[1]), vget_high_f32(_r23lt.val[1]));
    _r0h = vcombine_f32(vget_low_f32(_r45lt.val[0]), vget_low_f32(_r67lt.val[0]));
    _r2l = vcombine_f32(vget_low_f32(_r45lt.val[1]), vget_low_f32(_r67lt.val[1]));
    _r3h = vcombine_f32(vget_high_f32(_r45lt.val[0]), vget_high_f32(_r67lt.val[0]));
    _r5l = vcombine_f32(vget_high_f32(_r45lt.val[1]), vget_high_f32(_r67lt.val[1]));
    _r1l = vcombine_f32(vget_low_f32(_r89lt.val[0]), vget_low_f32(_rablt.val[0]));
    _r2h = vcombine_f32(vget_low_f32(_r89lt.val[1]), vget_low_f32(_rablt.val[1]));
    _r4l = vcombine_f32(vget_high_f32(_r89lt.val[0]), vget_high_f32(_rablt.val[0]));
    _r5h = vcombine_f32(vget_high_f32(_r89lt.val[1]), vget_high_f32(_rablt.val[1]));
    _r6l = vcombine_f32(vget_low_f32(_r01ht.val[0]), vget_low_f32(_r23ht.val[0]));
    _r7h = vcombine_f32(vget_low_f32(_r01ht.val[1]), vget_low_f32(_r23ht.val[1]));
    _r9l = vcombine_f32(vget_high_f32(_r01ht.val[0]), vget_high_f32(_r23ht.val[0]));
    _rah = vcombine_f32(vget_high_f32(_r01ht.val[1]), vget_high_f32(_r23ht.val[1]));
    _r6h = vcombine_f32(vget_low_f32(_r45ht.val[0]), vget_low_f32(_r67ht.val[0]));
    _r8l = vcombine_f32(vget_low_f32(_r45ht.val[1]), vget_low_f32(_r67ht.val[1]));
    _r9h = vcombine_f32(vget_high_f32(_r45ht.val[0]), vget_high_f32(_r67ht.val[0]));
    _rbl = vcombine_f32(vget_high_f32(_r45ht.val[1]), vget_high_f32(_r67ht.val[1]));
    _r7l = vcombine_f32(vget_low_f32(_r89ht.val[0]), vget_low_f32(_rabht.val[0]));
    _r8h = vcombine_f32(vget_low_f32(_r89ht.val[1]), vget_low_f32(_rabht.val[1]));
    _ral = vcombine_f32(vget_high_f32(_r89ht.val[0]), vget_high_f32(_rabht.val[0]));
    _rbh = vcombine_f32(vget_high_f32(_r89ht.val[1]), vget_high_f32(_rabht.val[1]));
}

static inline void transpose12x8_ps(float32x4_t& _r0l, float32x4_t& _r0m, float32x4_t& _r0h,
                                    float32x4_t& _r1l, float32x4_t& _r1m, float32x4_t& _r1h,
                                    float32x4_t& _r2l, float32x4_t& _r2m, float32x4_t& _r2h,
                                    float32x4_t& _r3l, float32x4_t& _r3m, float32x4_t& _r3h,
                                    float32x4_t& _r4l, float32x4_t& _r4m, float32x4_t& _r4h,
                                    float32x4_t& _r5l, float32x4_t& _r5m, float32x4_t& _r5h,
                                    float32x4_t& _r6l, float32x4_t& _r6m, float32x4_t& _r6h,
                                    float32x4_t& _r7l, float32x4_t& _r7m, float32x4_t& _r7h)
{
    float32x4x2_t _r01lt = vtrnq_f32(_r0l, _r1l);
    float32x4x2_t _r23lt = vtrnq_f32(_r2l, _r3l);
    float32x4x2_t _r01mt = vtrnq_f32(_r0m, _r1m);
    float32x4x2_t _r23mt = vtrnq_f32(_r2m, _r3m);
    float32x4x2_t _r01ht = vtrnq_f32(_r0h, _r1h);
    float32x4x2_t _r23ht = vtrnq_f32(_r2h, _r3h);
    float32x4x2_t _r45lt = vtrnq_f32(_r4l, _r5l);
    float32x4x2_t _r67lt = vtrnq_f32(_r6l, _r7l);
    float32x4x2_t _r45mt = vtrnq_f32(_r4m, _r5m);
    float32x4x2_t _r67mt = vtrnq_f32(_r6m, _r7m);
    float32x4x2_t _r45ht = vtrnq_f32(_r4h, _r5h);
    float32x4x2_t _r67ht = vtrnq_f32(_r6h, _r7h);
    _r0l = vcombine_f32(vget_low_f32(_r01lt.val[0]), vget_low_f32(_r23lt.val[0]));
    _r0h = vcombine_f32(vget_low_f32(_r01lt.val[1]), vget_low_f32(_r23lt.val[1]));
    _r1m = vcombine_f32(vget_high_f32(_r01lt.val[0]), vget_high_f32(_r23lt.val[0]));
    _r2l = vcombine_f32(vget_high_f32(_r01lt.val[1]), vget_high_f32(_r23lt.val[1]));
    _r0m = vcombine_f32(vget_low_f32(_r45lt.val[0]), vget_low_f32(_r67lt.val[0]));
    _r1l = vcombine_f32(vget_low_f32(_r45lt.val[1]), vget_low_f32(_r67lt.val[1]));
    _r1h = vcombine_f32(vget_high_f32(_r45lt.val[0]), vget_high_f32(_r67lt.val[0]));
    _r2m = vcombine_f32(vget_high_f32(_r45lt.val[1]), vget_high_f32(_r67lt.val[1]));
    _r2h = vcombine_f32(vget_low_f32(_r01mt.val[0]), vget_low_f32(_r23mt.val[0]));
    _r3m = vcombine_f32(vget_low_f32(_r01mt.val[1]), vget_low_f32(_r23mt.val[1]));
    _r4l = vcombine_f32(vget_high_f32(_r01mt.val[0]), vget_high_f32(_r23mt.val[0]));
    _r4h = vcombine_f32(vget_high_f32(_r01mt.val[1]), vget_high_f32(_r23mt.val[1]));
    _r3l = vcombine_f32(vget_low_f32(_r45mt.val[0]), vget_low_f32(_r67mt.val[0]));
    _r3h = vcombine_f32(vget_low_f32(_r45mt.val[1]), vget_low_f32(_r67mt.val[1]));
    _r4m = vcombine_f32(vget_high_f32(_r45mt.val[0]), vget_high_f32(_r67mt.val[0]));
    _r5l = vcombine_f32(vget_high_f32(_r45mt.val[1]), vget_high_f32(_r67mt.val[1]));
    _r5m = vcombine_f32(vget_low_f32(_r01ht.val[0]), vget_low_f32(_r23ht.val[0]));
    _r6l = vcombine_f32(vget_low_f32(_r01ht.val[1]), vget_low_f32(_r23ht.val[1]));
    _r6h = vcombine_f32(vget_high_f32(_r01ht.val[0]), vget_high_f32(_r23ht.val[0]));
    _r7m = vcombine_f32(vget_high_f32(_r01ht.val[1]), vget_high_f32(_r23ht.val[1]));
    _r5h = vcombine_f32(vget_low_f32(_r45ht.val[0]), vget_low_f32(_r67ht.val[0]));
    _r6m = vcombine_f32(vget_low_f32(_r45ht.val[1]), vget_low_f32(_r67ht.val[1]));
    _r7l = vcombine_f32(vget_high_f32(_r45ht.val[0]), vget_high_f32(_r67ht.val[0]));
    _r7h = vcombine_f32(vget_high_f32(_r45ht.val[1]), vget_high_f32(_r67ht.val[1]));
}
#endif // __aarch64__
#endif // __ARM_NEON

#endif // ARM_USABILITY_H

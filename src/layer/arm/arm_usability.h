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
    int int32 = (int)roundf(v);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

#if __ARM_NEON
#include <arm_neon.h>

static inline uint16x4_t float2bfloat(float32x4_t _v)
{
    return vshrn_n_u32(vreinterpretq_u32_f32(_v), 16);
}
static inline float32x4_t bfloat2float(uint16x4_t _v)
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
#ifdef _MSC_VER
struct __fp16
{
    __fp16()
    {
        _u16 = 0;
    }

    __fp16(float f32)
    {
        _u16 = vget_lane_u16(vreinterpretq_u16_f16(vcvt_f16_f32(vdupq_n_f32(f32))), 0);
    }

    __fp16(__n16 n16)
    {
        _u16 = n16.n16_u16[0];
    }

    operator const float() const
    {
        return vgetq_lane_f32(vcvt_f32_f16(vreinterpretq_f16_u16(vdup_n_u16(_u16))), 0);
    }

    __fp16& operator+=(const __fp16& b)
    {
        float a = (float)*this;
        float f32 = (a + (float)b);
        _u16 = vget_lane_u16(vreinterpretq_u16_f16(vcvt_f16_f32(vdupq_n_f32(f32))), 0);
        return *this;
    }

    __fp16& operator-=(const __fp16& b)
    {
        float a = (float)*this;
        float f32 = (a - (float)b);
        _u16 = vget_lane_u16(vreinterpretq_u16_f16(vcvt_f16_f32(vdupq_n_f32(f32))), 0);
        return *this;
    }

    __fp16& operator*=(const __fp16& b)
    {
        float a = (float)*this;
        float f32 = (a * (float)b);
        _u16 = vget_lane_u16(vreinterpretq_u16_f16(vcvt_f16_f32(vdupq_n_f32(f32))), 0);
        return *this;
    }

    __fp16& operator/=(const __fp16& b)
    {
        float a = (float)*this;
        float f32 = (a / (float)b);
        _u16 = vget_lane_u16(vreinterpretq_u16_f16(vcvt_f16_f32(vdupq_n_f32(f32))), 0);
        return *this;
    }

    unsigned short _u16;
};

static inline __fp16 operator-(const __fp16& a)
{
    return __fp16(-(float)a);
}
static inline __fp16 operator+(const __fp16& a, const __fp16& b)
{
    return __fp16((float)a + (float)b);
}
static inline __fp16 operator-(const __fp16& a, const __fp16& b)
{
    return __fp16((float)a - (float)b);
}
static inline __fp16 operator*(const __fp16& a, const __fp16& b)
{
    return __fp16((float)a * (float)b);
}
static inline __fp16 operator/(const __fp16& a, const __fp16& b)
{
    return __fp16((float)a / (float)b);
}

static inline float16x4_t vdup_n_f16(const __fp16& f16)
{
    return vreinterpret_f16_u16(vdup_n_u16(f16._u16));
}

static inline float16x8_t vdupq_n_f16(const __fp16& f16)
{
    return vreinterpretq_f16_u16(vdupq_n_u16(f16._u16));
}

static inline __fp16 vmaxv_f16(float16x4_t a)
{
    return __fp16(vmaxvq_f32(vcvt_f32_f16(a)));
}

static inline __fp16 vmaxvq_f16(float16x8_t a)
{
    float x = vmaxvq_f32(vcvt_f32_f16(vget_low_f16(a)));
    float y = vmaxvq_f32(vcvt_f32_f16(vget_high_f16(a)));
    return __fp16(x > y ? x : y);
}

#define vld1q_f16 vld1q_u16
#define vst1q_f16 vst1q_u16

#define vld2_f16 vld2_u16
#define vst2_f16 vst2_u16

#define vld2q_f16 vld2q_u16
#define vst2q_f16 vst2q_u16

#define vld4_f16 vld4_u16
#define vst4_f16 vst4_u16

#define vld4q_f16 vld4q_u16
#define vst4q_f16 vst4q_u16

#define vld1q_dup_f16 vld1q_dup_u16

#define vset_lane_f16(x, v, i)  vset_lane_u16(x._u16, (uint16x4_t)v, i)
#define vsetq_lane_f16(x, v, i) vsetq_lane_u16(x._u16, (uint16x8_t)v, i)

#define vfma_n_f16(va, vb, x)  vfma_f16(va, vb, vdup_n_f16(x))
#define vfmaq_n_f16(va, vb, x) vfmaq_f16(va, vb, vdupq_n_f16(x))

#endif

static inline signed char float2int8(__fp16 v)
{
    int int32 = (int)roundf(v);
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

static inline void transpose4x4_u16(uint16x4_t& _r0, uint16x4_t& _r1, uint16x4_t& _r2, uint16x4_t& _r3)
{
    uint16x4x2_t _r01z = vzip_u16(_r0, _r1);
    uint16x4x2_t _r23z = vzip_u16(_r2, _r3);
    uint32x2x2_t _r01 = vzip_u32(vreinterpret_u32_u16(_r01z.val[0]), vreinterpret_u32_u16(_r23z.val[0]));
    uint32x2x2_t _r23 = vzip_u32(vreinterpret_u32_u16(_r01z.val[1]), vreinterpret_u32_u16(_r23z.val[1]));
    _r0 = vreinterpret_u16_u32(_r01.val[0]);
    _r1 = vreinterpret_u16_u32(_r01.val[1]);
    _r2 = vreinterpret_u16_u32(_r23.val[0]);
    _r3 = vreinterpret_u16_u32(_r23.val[1]);
}

static inline void transpose4x8_u16(uint16x4_t& _r0, uint16x4_t& _r1, uint16x4_t& _r2, uint16x4_t& _r3, uint16x4_t& _r4, uint16x4_t& _r5, uint16x4_t& _r6, uint16x4_t& _r7)
{
    uint16x4x2_t _r01z = vzip_u16(_r0, _r1);
    uint16x4x2_t _r23z = vzip_u16(_r2, _r3);
    uint16x4x2_t _r45z = vzip_u16(_r4, _r5);
    uint16x4x2_t _r67z = vzip_u16(_r6, _r7);
    uint32x2x2_t _r01_0 = vzip_u32(vreinterpret_u32_u16(_r01z.val[0]), vreinterpret_u32_u16(_r23z.val[0]));
    uint32x2x2_t _r23_0 = vzip_u32(vreinterpret_u32_u16(_r01z.val[1]), vreinterpret_u32_u16(_r23z.val[1]));
    uint32x2x2_t _r01_1 = vzip_u32(vreinterpret_u32_u16(_r45z.val[0]), vreinterpret_u32_u16(_r67z.val[0]));
    uint32x2x2_t _r23_1 = vzip_u32(vreinterpret_u32_u16(_r45z.val[1]), vreinterpret_u32_u16(_r67z.val[1]));
    _r0 = vreinterpret_u16_u32(_r01_0.val[0]);
    _r1 = vreinterpret_u16_u32(_r01_1.val[0]);
    _r2 = vreinterpret_u16_u32(_r01_0.val[1]);
    _r3 = vreinterpret_u16_u32(_r01_1.val[1]);
    _r4 = vreinterpret_u16_u32(_r23_0.val[0]);
    _r5 = vreinterpret_u16_u32(_r23_1.val[0]);
    _r6 = vreinterpret_u16_u32(_r23_0.val[1]);
    _r7 = vreinterpret_u16_u32(_r23_1.val[1]);
}

static inline void transpose4x12_u16(uint16x4_t& _r0, uint16x4_t& _r1, uint16x4_t& _r2, uint16x4_t& _r3, uint16x4_t& _r4, uint16x4_t& _r5, uint16x4_t& _r6, uint16x4_t& _r7, uint16x4_t& _r8, uint16x4_t& _r9, uint16x4_t& _ra, uint16x4_t& _rb)
{
    uint16x4x2_t _r01z = vzip_u16(_r0, _r1);
    uint16x4x2_t _r23z = vzip_u16(_r2, _r3);
    uint16x4x2_t _r45z = vzip_u16(_r4, _r5);
    uint16x4x2_t _r67z = vzip_u16(_r6, _r7);
    uint16x4x2_t _r89z = vzip_u16(_r8, _r9);
    uint16x4x2_t _rabz = vzip_u16(_ra, _rb);
    uint32x2x2_t _r01_0 = vzip_u32(vreinterpret_u32_u16(_r01z.val[0]), vreinterpret_u32_u16(_r23z.val[0]));
    uint32x2x2_t _r23_0 = vzip_u32(vreinterpret_u32_u16(_r01z.val[1]), vreinterpret_u32_u16(_r23z.val[1]));
    uint32x2x2_t _r01_1 = vzip_u32(vreinterpret_u32_u16(_r45z.val[0]), vreinterpret_u32_u16(_r67z.val[0]));
    uint32x2x2_t _r23_1 = vzip_u32(vreinterpret_u32_u16(_r45z.val[1]), vreinterpret_u32_u16(_r67z.val[1]));
    uint32x2x2_t _r01_2 = vzip_u32(vreinterpret_u32_u16(_r89z.val[0]), vreinterpret_u32_u16(_rabz.val[0]));
    uint32x2x2_t _r23_2 = vzip_u32(vreinterpret_u32_u16(_r89z.val[1]), vreinterpret_u32_u16(_rabz.val[1]));
    _r0 = vreinterpret_u16_u32(_r01_0.val[0]);
    _r1 = vreinterpret_u16_u32(_r01_1.val[0]);
    _r2 = vreinterpret_u16_u32(_r01_2.val[0]);
    _r3 = vreinterpret_u16_u32(_r01_0.val[1]);
    _r4 = vreinterpret_u16_u32(_r01_1.val[1]);
    _r5 = vreinterpret_u16_u32(_r01_2.val[1]);
    _r6 = vreinterpret_u16_u32(_r23_0.val[0]);
    _r7 = vreinterpret_u16_u32(_r23_1.val[0]);
    _r8 = vreinterpret_u16_u32(_r23_2.val[0]);
    _r9 = vreinterpret_u16_u32(_r23_0.val[1]);
    _ra = vreinterpret_u16_u32(_r23_1.val[1]);
    _rb = vreinterpret_u16_u32(_r23_2.val[1]);
}

static inline void transpose8x4_u16(uint16x8_t& _r0, uint16x8_t& _r1, uint16x8_t& _r2, uint16x8_t& _r3)
{
    uint16x8x2_t _r01t = vzipq_u16(_r0, _r1);
    uint16x8x2_t _r23t = vzipq_u16(_r2, _r3);
    uint32x4x2_t _r01_0 = vzipq_u32(vreinterpretq_u32_u16(_r01t.val[0]), vreinterpretq_u32_u16(_r23t.val[0]));
    uint32x4x2_t _r23_0 = vzipq_u32(vreinterpretq_u32_u16(_r01t.val[1]), vreinterpretq_u32_u16(_r23t.val[1]));
    _r0 = vreinterpretq_u16_u32(_r01_0.val[0]);
    _r1 = vreinterpretq_u16_u32(_r01_0.val[1]);
    _r2 = vreinterpretq_u16_u32(_r23_0.val[0]);
    _r3 = vreinterpretq_u16_u32(_r23_0.val[1]);
}

static inline void transpose8x8_u16(uint16x8_t& _r0, uint16x8_t& _r1, uint16x8_t& _r2, uint16x8_t& _r3, uint16x8_t& _r4, uint16x8_t& _r5, uint16x8_t& _r6, uint16x8_t& _r7)
{
    uint16x8x2_t _r01t = vzipq_u16(_r0, _r1);
    uint16x8x2_t _r23t = vzipq_u16(_r2, _r3);
    uint16x8x2_t _r45t = vzipq_u16(_r4, _r5);
    uint16x8x2_t _r67t = vzipq_u16(_r6, _r7);
    uint32x4x2_t _r01_0 = vzipq_u32(vreinterpretq_u32_u16(_r01t.val[0]), vreinterpretq_u32_u16(_r23t.val[0]));
    uint32x4x2_t _r23_0 = vzipq_u32(vreinterpretq_u32_u16(_r01t.val[1]), vreinterpretq_u32_u16(_r23t.val[1]));
    uint32x4x2_t _r01_1 = vzipq_u32(vreinterpretq_u32_u16(_r45t.val[0]), vreinterpretq_u32_u16(_r67t.val[0]));
    uint32x4x2_t _r23_1 = vzipq_u32(vreinterpretq_u32_u16(_r45t.val[1]), vreinterpretq_u32_u16(_r67t.val[1]));
    _r0 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_r01_0.val[0]), vget_low_u32(_r01_1.val[0])));
    _r1 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_r01_0.val[0]), vget_high_u32(_r01_1.val[0])));
    _r2 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_r01_0.val[1]), vget_low_u32(_r01_1.val[1])));
    _r3 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_r01_0.val[1]), vget_high_u32(_r01_1.val[1])));
    _r4 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_r23_0.val[0]), vget_low_u32(_r23_1.val[0])));
    _r5 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_r23_0.val[0]), vget_high_u32(_r23_1.val[0])));
    _r6 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_r23_0.val[1]), vget_low_u32(_r23_1.val[1])));
    _r7 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_r23_0.val[1]), vget_high_u32(_r23_1.val[1])));
}

static inline void transpose8x12_u16(uint16x8_t& _r0, uint16x8_t& _r1, uint16x8_t& _r2, uint16x8_t& _r3, uint16x8_t& _r4, uint16x8_t& _r5, uint16x8_t& _r6, uint16x8_t& _r7, uint16x8_t& _r8, uint16x8_t& _r9, uint16x8_t& _ra, uint16x8_t& _rb)
{
    uint16x8x2_t _r01t = vzipq_u16(_r0, _r1);
    uint16x8x2_t _r23t = vzipq_u16(_r2, _r3);
    uint16x8x2_t _r45t = vzipq_u16(_r4, _r5);
    uint16x8x2_t _r67t = vzipq_u16(_r6, _r7);
    uint16x8x2_t _r89t = vzipq_u16(_r8, _r9);
    uint16x8x2_t _rabt = vzipq_u16(_ra, _rb);
    uint32x4x2_t _r01_0 = vzipq_u32(vreinterpretq_u32_u16(_r01t.val[0]), vreinterpretq_u32_u16(_r23t.val[0]));
    uint32x4x2_t _r23_0 = vzipq_u32(vreinterpretq_u32_u16(_r01t.val[1]), vreinterpretq_u32_u16(_r23t.val[1]));
    uint32x4x2_t _r01_1 = vzipq_u32(vreinterpretq_u32_u16(_r45t.val[0]), vreinterpretq_u32_u16(_r67t.val[0]));
    uint32x4x2_t _r23_1 = vzipq_u32(vreinterpretq_u32_u16(_r45t.val[1]), vreinterpretq_u32_u16(_r67t.val[1]));
    uint32x4x2_t _r01_2 = vzipq_u32(vreinterpretq_u32_u16(_r89t.val[0]), vreinterpretq_u32_u16(_rabt.val[0]));
    uint32x4x2_t _r23_2 = vzipq_u32(vreinterpretq_u32_u16(_r89t.val[1]), vreinterpretq_u32_u16(_rabt.val[1]));
    _r0 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_r01_0.val[0]), vget_low_u32(_r01_1.val[0])));
    _r1 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_r01_2.val[0]), vget_high_u32(_r01_0.val[0])));
    _r2 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_r01_1.val[0]), vget_high_u32(_r01_2.val[0])));
    _r3 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_r01_0.val[1]), vget_low_u32(_r01_1.val[1])));
    _r4 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_r01_2.val[1]), vget_high_u32(_r01_0.val[1])));
    _r5 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_r01_1.val[1]), vget_high_u32(_r01_2.val[1])));
    _r6 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_r23_0.val[0]), vget_low_u32(_r23_1.val[0])));
    _r7 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_r23_2.val[0]), vget_high_u32(_r23_0.val[0])));
    _r8 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_r23_1.val[0]), vget_high_u32(_r23_2.val[0])));
    _r9 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_r23_0.val[1]), vget_low_u32(_r23_1.val[1])));
    _ra = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_r23_2.val[1]), vget_high_u32(_r23_0.val[1])));
    _rb = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_r23_1.val[1]), vget_high_u32(_r23_2.val[1])));
}

static inline void transpose4x4_ps(float32x4_t& _r0, float32x4_t& _r1, float32x4_t& _r2, float32x4_t& _r3)
{
    float32x4x2_t _r01z = vzipq_f32(_r0, _r1);
    float32x4x2_t _r23z = vzipq_f32(_r2, _r3);
    _r0 = vcombine_f32(vget_low_f32(_r01z.val[0]), vget_low_f32(_r23z.val[0]));
    _r1 = vcombine_f32(vget_high_f32(_r01z.val[0]), vget_high_f32(_r23z.val[0]));
    _r2 = vcombine_f32(vget_low_f32(_r01z.val[1]), vget_low_f32(_r23z.val[1]));
    _r3 = vcombine_f32(vget_high_f32(_r01z.val[1]), vget_high_f32(_r23z.val[1]));
}

static inline void transpose4x8_ps(float32x4_t& _r0, float32x4_t& _r1, float32x4_t& _r2, float32x4_t& _r3, float32x4_t& _r4, float32x4_t& _r5, float32x4_t& _r6, float32x4_t& _r7)
{
    float32x4x2_t _r01z = vzipq_f32(_r0, _r1);
    float32x4x2_t _r23z = vzipq_f32(_r2, _r3);
    float32x4x2_t _r45z = vzipq_f32(_r4, _r5);
    float32x4x2_t _r67z = vzipq_f32(_r6, _r7);
    _r0 = vcombine_f32(vget_low_f32(_r01z.val[0]), vget_low_f32(_r23z.val[0]));
    _r1 = vcombine_f32(vget_low_f32(_r45z.val[0]), vget_low_f32(_r67z.val[0]));
    _r2 = vcombine_f32(vget_high_f32(_r01z.val[0]), vget_high_f32(_r23z.val[0]));
    _r3 = vcombine_f32(vget_high_f32(_r45z.val[0]), vget_high_f32(_r67z.val[0]));
    _r4 = vcombine_f32(vget_low_f32(_r01z.val[1]), vget_low_f32(_r23z.val[1]));
    _r5 = vcombine_f32(vget_low_f32(_r45z.val[1]), vget_low_f32(_r67z.val[1]));
    _r6 = vcombine_f32(vget_high_f32(_r01z.val[1]), vget_high_f32(_r23z.val[1]));
    _r7 = vcombine_f32(vget_high_f32(_r45z.val[1]), vget_high_f32(_r67z.val[1]));
}

static inline void transpose4x12_ps(float32x4_t& _r0, float32x4_t& _r1, float32x4_t& _r2, float32x4_t& _r3, float32x4_t& _r4, float32x4_t& _r5, float32x4_t& _r6, float32x4_t& _r7, float32x4_t& _r8, float32x4_t& _r9, float32x4_t& _ra, float32x4_t& _rb)
{
    float32x4x2_t _r01z = vzipq_f32(_r0, _r1);
    float32x4x2_t _r23z = vzipq_f32(_r2, _r3);
    float32x4x2_t _r45z = vzipq_f32(_r4, _r5);
    float32x4x2_t _r67z = vzipq_f32(_r6, _r7);
    float32x4x2_t _r89z = vzipq_f32(_r8, _r9);
    float32x4x2_t _rabz = vzipq_f32(_ra, _rb);
    _r0 = vcombine_f32(vget_low_f32(_r01z.val[0]), vget_low_f32(_r23z.val[0]));
    _r1 = vcombine_f32(vget_low_f32(_r45z.val[0]), vget_low_f32(_r67z.val[0]));
    _r2 = vcombine_f32(vget_low_f32(_r89z.val[0]), vget_low_f32(_rabz.val[0]));
    _r3 = vcombine_f32(vget_high_f32(_r01z.val[0]), vget_high_f32(_r23z.val[0]));
    _r4 = vcombine_f32(vget_high_f32(_r45z.val[0]), vget_high_f32(_r67z.val[0]));
    _r5 = vcombine_f32(vget_high_f32(_r89z.val[0]), vget_high_f32(_rabz.val[0]));
    _r6 = vcombine_f32(vget_low_f32(_r01z.val[1]), vget_low_f32(_r23z.val[1]));
    _r7 = vcombine_f32(vget_low_f32(_r45z.val[1]), vget_low_f32(_r67z.val[1]));
    _r8 = vcombine_f32(vget_low_f32(_r89z.val[1]), vget_low_f32(_rabz.val[1]));
    _r9 = vcombine_f32(vget_high_f32(_r01z.val[1]), vget_high_f32(_r23z.val[1]));
    _ra = vcombine_f32(vget_high_f32(_r45z.val[1]), vget_high_f32(_r67z.val[1]));
    _rb = vcombine_f32(vget_high_f32(_r89z.val[1]), vget_high_f32(_rabz.val[1]));
}

static inline void transpose8x4_ps(float32x4_t& _r0l, float32x4_t& _r0h,
                                   float32x4_t& _r1l, float32x4_t& _r1h,
                                   float32x4_t& _r2l, float32x4_t& _r2h,
                                   float32x4_t& _r3l, float32x4_t& _r3h)
{
    float32x4x2_t _r01lz = vzipq_f32(_r0l, _r1l);
    float32x4x2_t _r23lz = vzipq_f32(_r2l, _r3l);
    float32x4x2_t _r01hz = vzipq_f32(_r0h, _r1h);
    float32x4x2_t _r23hz = vzipq_f32(_r2h, _r3h);
    _r0l = vcombine_f32(vget_low_f32(_r01lz.val[0]), vget_low_f32(_r23lz.val[0]));
    _r0h = vcombine_f32(vget_high_f32(_r01lz.val[0]), vget_high_f32(_r23lz.val[0]));
    _r1l = vcombine_f32(vget_low_f32(_r01lz.val[1]), vget_low_f32(_r23lz.val[1]));
    _r1h = vcombine_f32(vget_high_f32(_r01lz.val[1]), vget_high_f32(_r23lz.val[1]));
    _r2l = vcombine_f32(vget_low_f32(_r01hz.val[0]), vget_low_f32(_r23hz.val[0]));
    _r2h = vcombine_f32(vget_high_f32(_r01hz.val[0]), vget_high_f32(_r23hz.val[0]));
    _r3l = vcombine_f32(vget_low_f32(_r01hz.val[1]), vget_low_f32(_r23hz.val[1]));
    _r3h = vcombine_f32(vget_high_f32(_r01hz.val[1]), vget_high_f32(_r23hz.val[1]));
}

static inline void transpose12x4_ps(float32x4_t& _r0l, float32x4_t& _r0m, float32x4_t& _r0h,
                                    float32x4_t& _r1l, float32x4_t& _r1m, float32x4_t& _r1h,
                                    float32x4_t& _r2l, float32x4_t& _r2m, float32x4_t& _r2h,
                                    float32x4_t& _r3l, float32x4_t& _r3m, float32x4_t& _r3h)
{
    float32x4x2_t _r01lz = vzipq_f32(_r0l, _r1l);
    float32x4x2_t _r23lz = vzipq_f32(_r2l, _r3l);
    float32x4x2_t _r01mz = vzipq_f32(_r0m, _r1m);
    float32x4x2_t _r23mz = vzipq_f32(_r2m, _r3m);
    float32x4x2_t _r01hz = vzipq_f32(_r0h, _r1h);
    float32x4x2_t _r23hz = vzipq_f32(_r2h, _r3h);
    _r0l = vcombine_f32(vget_low_f32(_r01lz.val[0]), vget_low_f32(_r23lz.val[0]));
    _r0m = vcombine_f32(vget_high_f32(_r01lz.val[0]), vget_high_f32(_r23lz.val[0]));
    _r0h = vcombine_f32(vget_low_f32(_r01lz.val[1]), vget_low_f32(_r23lz.val[1]));
    _r1l = vcombine_f32(vget_high_f32(_r01lz.val[1]), vget_high_f32(_r23lz.val[1]));
    _r1m = vcombine_f32(vget_low_f32(_r01mz.val[0]), vget_low_f32(_r23mz.val[0]));
    _r1h = vcombine_f32(vget_high_f32(_r01mz.val[0]), vget_high_f32(_r23mz.val[0]));
    _r2l = vcombine_f32(vget_low_f32(_r01mz.val[1]), vget_low_f32(_r23mz.val[1]));
    _r2m = vcombine_f32(vget_high_f32(_r01mz.val[1]), vget_high_f32(_r23mz.val[1]));
    _r2h = vcombine_f32(vget_low_f32(_r01hz.val[0]), vget_low_f32(_r23hz.val[0]));
    _r3l = vcombine_f32(vget_high_f32(_r01hz.val[0]), vget_high_f32(_r23hz.val[0]));
    _r3m = vcombine_f32(vget_low_f32(_r01hz.val[1]), vget_low_f32(_r23hz.val[1]));
    _r3h = vcombine_f32(vget_high_f32(_r01hz.val[1]), vget_high_f32(_r23hz.val[1]));
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
    float32x4x2_t _r01lz = vzipq_f32(_r0l, _r1l);
    float32x4x2_t _r23lz = vzipq_f32(_r2l, _r3l);
    float32x4x2_t _r01hz = vzipq_f32(_r0h, _r1h);
    float32x4x2_t _r23hz = vzipq_f32(_r2h, _r3h);
    float32x4x2_t _r45lz = vzipq_f32(_r4l, _r5l);
    float32x4x2_t _r67lz = vzipq_f32(_r6l, _r7l);
    float32x4x2_t _r45hz = vzipq_f32(_r4h, _r5h);
    float32x4x2_t _r67hz = vzipq_f32(_r6h, _r7h);
    _r0l = vcombine_f32(vget_low_f32(_r01lz.val[0]), vget_low_f32(_r23lz.val[0]));
    _r0h = vcombine_f32(vget_low_f32(_r45lz.val[0]), vget_low_f32(_r67lz.val[0]));
    _r1l = vcombine_f32(vget_high_f32(_r01lz.val[0]), vget_high_f32(_r23lz.val[0]));
    _r1h = vcombine_f32(vget_high_f32(_r45lz.val[0]), vget_high_f32(_r67lz.val[0]));
    _r2l = vcombine_f32(vget_low_f32(_r01lz.val[1]), vget_low_f32(_r23lz.val[1]));
    _r2h = vcombine_f32(vget_low_f32(_r45lz.val[1]), vget_low_f32(_r67lz.val[1]));
    _r3l = vcombine_f32(vget_high_f32(_r01lz.val[1]), vget_high_f32(_r23lz.val[1]));
    _r3h = vcombine_f32(vget_high_f32(_r45lz.val[1]), vget_high_f32(_r67lz.val[1]));
    _r4l = vcombine_f32(vget_low_f32(_r01hz.val[0]), vget_low_f32(_r23hz.val[0]));
    _r4h = vcombine_f32(vget_low_f32(_r45hz.val[0]), vget_low_f32(_r67hz.val[0]));
    _r5l = vcombine_f32(vget_high_f32(_r01hz.val[0]), vget_high_f32(_r23hz.val[0]));
    _r5h = vcombine_f32(vget_high_f32(_r45hz.val[0]), vget_high_f32(_r67hz.val[0]));
    _r6l = vcombine_f32(vget_low_f32(_r01hz.val[1]), vget_low_f32(_r23hz.val[1]));
    _r6h = vcombine_f32(vget_low_f32(_r45hz.val[1]), vget_low_f32(_r67hz.val[1]));
    _r7l = vcombine_f32(vget_high_f32(_r01hz.val[1]), vget_high_f32(_r23hz.val[1]));
    _r7h = vcombine_f32(vget_high_f32(_r45hz.val[1]), vget_high_f32(_r67hz.val[1]));
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
    float32x4x2_t _r01lz = vzipq_f32(_r0l, _r1l);
    float32x4x2_t _r23lz = vzipq_f32(_r2l, _r3l);
    float32x4x2_t _r01hz = vzipq_f32(_r0h, _r1h);
    float32x4x2_t _r23hz = vzipq_f32(_r2h, _r3h);
    float32x4x2_t _r45lz = vzipq_f32(_r4l, _r5l);
    float32x4x2_t _r67lz = vzipq_f32(_r6l, _r7l);
    float32x4x2_t _r45hz = vzipq_f32(_r4h, _r5h);
    float32x4x2_t _r67hz = vzipq_f32(_r6h, _r7h);
    float32x4x2_t _r89lz = vzipq_f32(_r8l, _r9l);
    float32x4x2_t _rablz = vzipq_f32(_ral, _rbl);
    float32x4x2_t _r89hz = vzipq_f32(_r8h, _r9h);
    float32x4x2_t _rabhz = vzipq_f32(_rah, _rbh);
    _r0l = vcombine_f32(vget_low_f32(_r01lz.val[0]), vget_low_f32(_r23lz.val[0]));
    _r0h = vcombine_f32(vget_low_f32(_r45lz.val[0]), vget_low_f32(_r67lz.val[0]));
    _r1l = vcombine_f32(vget_low_f32(_r89lz.val[0]), vget_low_f32(_rablz.val[0]));
    _r1h = vcombine_f32(vget_high_f32(_r01lz.val[0]), vget_high_f32(_r23lz.val[0]));
    _r2l = vcombine_f32(vget_high_f32(_r45lz.val[0]), vget_high_f32(_r67lz.val[0]));
    _r2h = vcombine_f32(vget_high_f32(_r89lz.val[0]), vget_high_f32(_rablz.val[0]));
    _r3l = vcombine_f32(vget_low_f32(_r01lz.val[1]), vget_low_f32(_r23lz.val[1]));
    _r3h = vcombine_f32(vget_low_f32(_r45lz.val[1]), vget_low_f32(_r67lz.val[1]));
    _r4l = vcombine_f32(vget_low_f32(_r89lz.val[1]), vget_low_f32(_rablz.val[1]));
    _r4h = vcombine_f32(vget_high_f32(_r01lz.val[1]), vget_high_f32(_r23lz.val[1]));
    _r5l = vcombine_f32(vget_high_f32(_r45lz.val[1]), vget_high_f32(_r67lz.val[1]));
    _r5h = vcombine_f32(vget_high_f32(_r89lz.val[1]), vget_high_f32(_rablz.val[1]));
    _r6l = vcombine_f32(vget_low_f32(_r01hz.val[0]), vget_low_f32(_r23hz.val[0]));
    _r6h = vcombine_f32(vget_low_f32(_r45hz.val[0]), vget_low_f32(_r67hz.val[0]));
    _r7l = vcombine_f32(vget_low_f32(_r89hz.val[0]), vget_low_f32(_rabhz.val[0]));
    _r7h = vcombine_f32(vget_high_f32(_r01hz.val[0]), vget_high_f32(_r23hz.val[0]));
    _r8l = vcombine_f32(vget_high_f32(_r45hz.val[0]), vget_high_f32(_r67hz.val[0]));
    _r8h = vcombine_f32(vget_high_f32(_r89hz.val[0]), vget_high_f32(_rabhz.val[0]));
    _r9l = vcombine_f32(vget_low_f32(_r01hz.val[1]), vget_low_f32(_r23hz.val[1]));
    _r9h = vcombine_f32(vget_low_f32(_r45hz.val[1]), vget_low_f32(_r67hz.val[1]));
    _ral = vcombine_f32(vget_low_f32(_r89hz.val[1]), vget_low_f32(_rabhz.val[1]));
    _rah = vcombine_f32(vget_high_f32(_r01hz.val[1]), vget_high_f32(_r23hz.val[1]));
    _rbl = vcombine_f32(vget_high_f32(_r45hz.val[1]), vget_high_f32(_r67hz.val[1]));
    _rbh = vcombine_f32(vget_high_f32(_r89hz.val[1]), vget_high_f32(_rabhz.val[1]));
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
    float32x4x2_t _r01lz = vzipq_f32(_r0l, _r1l);
    float32x4x2_t _r23lz = vzipq_f32(_r2l, _r3l);
    float32x4x2_t _r01mz = vzipq_f32(_r0m, _r1m);
    float32x4x2_t _r23mz = vzipq_f32(_r2m, _r3m);
    float32x4x2_t _r01hz = vzipq_f32(_r0h, _r1h);
    float32x4x2_t _r23hz = vzipq_f32(_r2h, _r3h);
    float32x4x2_t _r45lz = vzipq_f32(_r4l, _r5l);
    float32x4x2_t _r67lz = vzipq_f32(_r6l, _r7l);
    float32x4x2_t _r45mz = vzipq_f32(_r4m, _r5m);
    float32x4x2_t _r67mz = vzipq_f32(_r6m, _r7m);
    float32x4x2_t _r45hz = vzipq_f32(_r4h, _r5h);
    float32x4x2_t _r67hz = vzipq_f32(_r6h, _r7h);
    _r0l = vcombine_f32(vget_low_f32(_r01lz.val[0]), vget_low_f32(_r23lz.val[0]));
    _r0m = vcombine_f32(vget_low_f32(_r45lz.val[0]), vget_low_f32(_r67lz.val[0]));
    _r0h = vcombine_f32(vget_high_f32(_r01lz.val[0]), vget_high_f32(_r23lz.val[0]));
    _r1l = vcombine_f32(vget_high_f32(_r45lz.val[0]), vget_high_f32(_r67lz.val[0]));
    _r1m = vcombine_f32(vget_low_f32(_r01lz.val[1]), vget_low_f32(_r23lz.val[1]));
    _r1h = vcombine_f32(vget_low_f32(_r45lz.val[1]), vget_low_f32(_r67lz.val[1]));
    _r2l = vcombine_f32(vget_high_f32(_r01lz.val[1]), vget_high_f32(_r23lz.val[1]));
    _r2m = vcombine_f32(vget_high_f32(_r45lz.val[1]), vget_high_f32(_r67lz.val[1]));
    _r2h = vcombine_f32(vget_low_f32(_r01mz.val[0]), vget_low_f32(_r23mz.val[0]));
    _r3l = vcombine_f32(vget_low_f32(_r45mz.val[0]), vget_low_f32(_r67mz.val[0]));
    _r3m = vcombine_f32(vget_high_f32(_r01mz.val[0]), vget_high_f32(_r23mz.val[0]));
    _r3h = vcombine_f32(vget_high_f32(_r45mz.val[0]), vget_high_f32(_r67mz.val[0]));
    _r4l = vcombine_f32(vget_low_f32(_r01mz.val[1]), vget_low_f32(_r23mz.val[1]));
    _r4m = vcombine_f32(vget_low_f32(_r45mz.val[1]), vget_low_f32(_r67mz.val[1]));
    _r4h = vcombine_f32(vget_high_f32(_r01mz.val[1]), vget_high_f32(_r23mz.val[1]));
    _r5l = vcombine_f32(vget_high_f32(_r45mz.val[1]), vget_high_f32(_r67mz.val[1]));
    _r5m = vcombine_f32(vget_low_f32(_r01hz.val[0]), vget_low_f32(_r23hz.val[0]));
    _r5h = vcombine_f32(vget_low_f32(_r45hz.val[0]), vget_low_f32(_r67hz.val[0]));
    _r6l = vcombine_f32(vget_high_f32(_r01hz.val[0]), vget_high_f32(_r23hz.val[0]));
    _r6m = vcombine_f32(vget_high_f32(_r45hz.val[0]), vget_high_f32(_r67hz.val[0]));
    _r6h = vcombine_f32(vget_low_f32(_r01hz.val[1]), vget_low_f32(_r23hz.val[1]));
    _r7l = vcombine_f32(vget_low_f32(_r45hz.val[1]), vget_low_f32(_r67hz.val[1]));
    _r7m = vcombine_f32(vget_high_f32(_r01hz.val[1]), vget_high_f32(_r23hz.val[1]));
    _r7h = vcombine_f32(vget_high_f32(_r45hz.val[1]), vget_high_f32(_r67hz.val[1]));
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static inline void transpose4x4_ph(float16x4_t& _r0, float16x4_t& _r1, float16x4_t& _r2, float16x4_t& _r3)
{
    uint16x4x2_t _r01z = vzip_u16(vreinterpret_u16_f16(_r0), vreinterpret_u16_f16(_r1));
    uint16x4x2_t _r23z = vzip_u16(vreinterpret_u16_f16(_r2), vreinterpret_u16_f16(_r3));
    uint32x2x2_t _r01 = vzip_u32(vreinterpret_u32_u16(_r01z.val[0]), vreinterpret_u32_u16(_r23z.val[0]));
    uint32x2x2_t _r23 = vzip_u32(vreinterpret_u32_u16(_r01z.val[1]), vreinterpret_u32_u16(_r23z.val[1]));
    _r0 = vreinterpret_f16_u32(_r01.val[0]);
    _r1 = vreinterpret_f16_u32(_r01.val[1]);
    _r2 = vreinterpret_f16_u32(_r23.val[0]);
    _r3 = vreinterpret_f16_u32(_r23.val[1]);
}

static inline void transpose4x8_ph(float16x4_t& _r0, float16x4_t& _r1, float16x4_t& _r2, float16x4_t& _r3, float16x4_t& _r4, float16x4_t& _r5, float16x4_t& _r6, float16x4_t& _r7)
{
    uint16x4x2_t _r01z = vzip_u16(vreinterpret_u16_f16(_r0), vreinterpret_u16_f16(_r1));
    uint16x4x2_t _r23z = vzip_u16(vreinterpret_u16_f16(_r2), vreinterpret_u16_f16(_r3));
    uint16x4x2_t _r45z = vzip_u16(vreinterpret_u16_f16(_r4), vreinterpret_u16_f16(_r5));
    uint16x4x2_t _r67z = vzip_u16(vreinterpret_u16_f16(_r6), vreinterpret_u16_f16(_r7));
    uint32x2x2_t _r01_0 = vzip_u32(vreinterpret_u32_u16(_r01z.val[0]), vreinterpret_u32_u16(_r23z.val[0]));
    uint32x2x2_t _r23_0 = vzip_u32(vreinterpret_u32_u16(_r01z.val[1]), vreinterpret_u32_u16(_r23z.val[1]));
    uint32x2x2_t _r01_1 = vzip_u32(vreinterpret_u32_u16(_r45z.val[0]), vreinterpret_u32_u16(_r67z.val[0]));
    uint32x2x2_t _r23_1 = vzip_u32(vreinterpret_u32_u16(_r45z.val[1]), vreinterpret_u32_u16(_r67z.val[1]));
    _r0 = vreinterpret_f16_u32(_r01_0.val[0]);
    _r1 = vreinterpret_f16_u32(_r01_1.val[0]);
    _r2 = vreinterpret_f16_u32(_r01_0.val[1]);
    _r3 = vreinterpret_f16_u32(_r01_1.val[1]);
    _r4 = vreinterpret_f16_u32(_r23_0.val[0]);
    _r5 = vreinterpret_f16_u32(_r23_1.val[0]);
    _r6 = vreinterpret_f16_u32(_r23_0.val[1]);
    _r7 = vreinterpret_f16_u32(_r23_1.val[1]);
}

static inline void transpose4x12_ph(float16x4_t& _r0, float16x4_t& _r1, float16x4_t& _r2, float16x4_t& _r3, float16x4_t& _r4, float16x4_t& _r5, float16x4_t& _r6, float16x4_t& _r7, float16x4_t& _r8, float16x4_t& _r9, float16x4_t& _ra, float16x4_t& _rb)
{
    uint16x4x2_t _r01z = vzip_u16(vreinterpret_u16_f16(_r0), vreinterpret_u16_f16(_r1));
    uint16x4x2_t _r23z = vzip_u16(vreinterpret_u16_f16(_r2), vreinterpret_u16_f16(_r3));
    uint16x4x2_t _r45z = vzip_u16(vreinterpret_u16_f16(_r4), vreinterpret_u16_f16(_r5));
    uint16x4x2_t _r67z = vzip_u16(vreinterpret_u16_f16(_r6), vreinterpret_u16_f16(_r7));
    uint16x4x2_t _r89z = vzip_u16(vreinterpret_u16_f16(_r8), vreinterpret_u16_f16(_r9));
    uint16x4x2_t _rabz = vzip_u16(vreinterpret_u16_f16(_ra), vreinterpret_u16_f16(_rb));
    uint32x2x2_t _r01_0 = vzip_u32(vreinterpret_u32_u16(_r01z.val[0]), vreinterpret_u32_u16(_r23z.val[0]));
    uint32x2x2_t _r23_0 = vzip_u32(vreinterpret_u32_u16(_r01z.val[1]), vreinterpret_u32_u16(_r23z.val[1]));
    uint32x2x2_t _r01_1 = vzip_u32(vreinterpret_u32_u16(_r45z.val[0]), vreinterpret_u32_u16(_r67z.val[0]));
    uint32x2x2_t _r23_1 = vzip_u32(vreinterpret_u32_u16(_r45z.val[1]), vreinterpret_u32_u16(_r67z.val[1]));
    uint32x2x2_t _r01_2 = vzip_u32(vreinterpret_u32_u16(_r89z.val[0]), vreinterpret_u32_u16(_rabz.val[0]));
    uint32x2x2_t _r23_2 = vzip_u32(vreinterpret_u32_u16(_r89z.val[1]), vreinterpret_u32_u16(_rabz.val[1]));
    _r0 = vreinterpret_f16_u32(_r01_0.val[0]);
    _r1 = vreinterpret_f16_u32(_r01_1.val[0]);
    _r2 = vreinterpret_f16_u32(_r01_2.val[0]);
    _r3 = vreinterpret_f16_u32(_r01_0.val[1]);
    _r4 = vreinterpret_f16_u32(_r01_1.val[1]);
    _r5 = vreinterpret_f16_u32(_r01_2.val[1]);
    _r6 = vreinterpret_f16_u32(_r23_0.val[0]);
    _r7 = vreinterpret_f16_u32(_r23_1.val[0]);
    _r8 = vreinterpret_f16_u32(_r23_2.val[0]);
    _r9 = vreinterpret_f16_u32(_r23_0.val[1]);
    _ra = vreinterpret_f16_u32(_r23_1.val[1]);
    _rb = vreinterpret_f16_u32(_r23_2.val[1]);
}

static inline void transpose8x4_ph(float16x8_t& _r0, float16x8_t& _r1, float16x8_t& _r2, float16x8_t& _r3)
{
    uint16x8x2_t _r01t = vzipq_u16(vreinterpretq_u16_f16(_r0), vreinterpretq_u16_f16(_r1));
    uint16x8x2_t _r23t = vzipq_u16(vreinterpretq_u16_f16(_r2), vreinterpretq_u16_f16(_r3));
    uint32x4x2_t _r01 = vzipq_u32(vreinterpretq_u32_u16(_r01t.val[0]), vreinterpretq_u32_u16(_r23t.val[0]));
    uint32x4x2_t _r23 = vzipq_u32(vreinterpretq_u32_u16(_r01t.val[1]), vreinterpretq_u32_u16(_r23t.val[1]));
    _r0 = vreinterpretq_f16_u32(_r01.val[0]);
    _r1 = vreinterpretq_f16_u32(_r01.val[1]);
    _r2 = vreinterpretq_f16_u32(_r23.val[0]);
    _r3 = vreinterpretq_f16_u32(_r23.val[1]);
}

static inline void transpose8x8_ph(float16x8_t& _r0, float16x8_t& _r1, float16x8_t& _r2, float16x8_t& _r3, float16x8_t& _r4, float16x8_t& _r5, float16x8_t& _r6, float16x8_t& _r7)
{
    uint16x8x2_t _r01t = vzipq_u16(vreinterpretq_u16_f16(_r0), vreinterpretq_u16_f16(_r1));
    uint16x8x2_t _r23t = vzipq_u16(vreinterpretq_u16_f16(_r2), vreinterpretq_u16_f16(_r3));
    uint16x8x2_t _r45t = vzipq_u16(vreinterpretq_u16_f16(_r4), vreinterpretq_u16_f16(_r5));
    uint16x8x2_t _r67t = vzipq_u16(vreinterpretq_u16_f16(_r6), vreinterpretq_u16_f16(_r7));
    uint32x4x2_t _r01_0 = vzipq_u32(vreinterpretq_u32_u16(_r01t.val[0]), vreinterpretq_u32_u16(_r23t.val[0]));
    uint32x4x2_t _r23_0 = vzipq_u32(vreinterpretq_u32_u16(_r01t.val[1]), vreinterpretq_u32_u16(_r23t.val[1]));
    uint32x4x2_t _r01_1 = vzipq_u32(vreinterpretq_u32_u16(_r45t.val[0]), vreinterpretq_u32_u16(_r67t.val[0]));
    uint32x4x2_t _r23_1 = vzipq_u32(vreinterpretq_u32_u16(_r45t.val[1]), vreinterpretq_u32_u16(_r67t.val[1]));
    _r0 = vreinterpretq_f16_u32(vcombine_u32(vget_low_u32(_r01_0.val[0]), vget_low_u32(_r01_1.val[0])));
    _r1 = vreinterpretq_f16_u32(vcombine_u32(vget_high_u32(_r01_0.val[0]), vget_high_u32(_r01_1.val[0])));
    _r2 = vreinterpretq_f16_u32(vcombine_u32(vget_low_u32(_r01_0.val[1]), vget_low_u32(_r01_1.val[1])));
    _r3 = vreinterpretq_f16_u32(vcombine_u32(vget_high_u32(_r01_0.val[1]), vget_high_u32(_r01_1.val[1])));
    _r4 = vreinterpretq_f16_u32(vcombine_u32(vget_low_u32(_r23_0.val[0]), vget_low_u32(_r23_1.val[0])));
    _r5 = vreinterpretq_f16_u32(vcombine_u32(vget_high_u32(_r23_0.val[0]), vget_high_u32(_r23_1.val[0])));
    _r6 = vreinterpretq_f16_u32(vcombine_u32(vget_low_u32(_r23_0.val[1]), vget_low_u32(_r23_1.val[1])));
    _r7 = vreinterpretq_f16_u32(vcombine_u32(vget_high_u32(_r23_0.val[1]), vget_high_u32(_r23_1.val[1])));
}

static inline void transpose8x12_ph(float16x8_t& _r0, float16x8_t& _r1, float16x8_t& _r2, float16x8_t& _r3, float16x8_t& _r4, float16x8_t& _r5, float16x8_t& _r6, float16x8_t& _r7, float16x8_t& _r8, float16x8_t& _r9, float16x8_t& _ra, float16x8_t& _rb)
{
    uint16x8x2_t _r01t = vzipq_u16(vreinterpretq_u16_f16(_r0), vreinterpretq_u16_f16(_r1));
    uint16x8x2_t _r23t = vzipq_u16(vreinterpretq_u16_f16(_r2), vreinterpretq_u16_f16(_r3));
    uint16x8x2_t _r45t = vzipq_u16(vreinterpretq_u16_f16(_r4), vreinterpretq_u16_f16(_r5));
    uint16x8x2_t _r67t = vzipq_u16(vreinterpretq_u16_f16(_r6), vreinterpretq_u16_f16(_r7));
    uint16x8x2_t _r89t = vzipq_u16(vreinterpretq_u16_f16(_r8), vreinterpretq_u16_f16(_r9));
    uint16x8x2_t _rabt = vzipq_u16(vreinterpretq_u16_f16(_ra), vreinterpretq_u16_f16(_rb));
    uint32x4x2_t _r01_0 = vzipq_u32(vreinterpretq_u32_u16(_r01t.val[0]), vreinterpretq_u32_u16(_r23t.val[0]));
    uint32x4x2_t _r23_0 = vzipq_u32(vreinterpretq_u32_u16(_r01t.val[1]), vreinterpretq_u32_u16(_r23t.val[1]));
    uint32x4x2_t _r01_1 = vzipq_u32(vreinterpretq_u32_u16(_r45t.val[0]), vreinterpretq_u32_u16(_r67t.val[0]));
    uint32x4x2_t _r23_1 = vzipq_u32(vreinterpretq_u32_u16(_r45t.val[1]), vreinterpretq_u32_u16(_r67t.val[1]));
    uint32x4x2_t _r01_2 = vzipq_u32(vreinterpretq_u32_u16(_r89t.val[0]), vreinterpretq_u32_u16(_rabt.val[0]));
    uint32x4x2_t _r23_2 = vzipq_u32(vreinterpretq_u32_u16(_r89t.val[1]), vreinterpretq_u32_u16(_rabt.val[1]));
    _r0 = vreinterpretq_f16_u32(vcombine_u32(vget_low_u32(_r01_0.val[0]), vget_low_u32(_r01_1.val[0])));
    _r1 = vreinterpretq_f16_u32(vcombine_u32(vget_low_u32(_r01_2.val[0]), vget_high_u32(_r01_0.val[0])));
    _r2 = vreinterpretq_f16_u32(vcombine_u32(vget_high_u32(_r01_1.val[0]), vget_high_u32(_r01_2.val[0])));
    _r3 = vreinterpretq_f16_u32(vcombine_u32(vget_low_u32(_r01_0.val[1]), vget_low_u32(_r01_1.val[1])));
    _r4 = vreinterpretq_f16_u32(vcombine_u32(vget_low_u32(_r01_2.val[1]), vget_high_u32(_r01_0.val[1])));
    _r5 = vreinterpretq_f16_u32(vcombine_u32(vget_high_u32(_r01_1.val[1]), vget_high_u32(_r01_2.val[1])));
    _r6 = vreinterpretq_f16_u32(vcombine_u32(vget_low_u32(_r23_0.val[0]), vget_low_u32(_r23_1.val[0])));
    _r7 = vreinterpretq_f16_u32(vcombine_u32(vget_low_u32(_r23_2.val[0]), vget_high_u32(_r23_0.val[0])));
    _r8 = vreinterpretq_f16_u32(vcombine_u32(vget_high_u32(_r23_1.val[0]), vget_high_u32(_r23_2.val[0])));
    _r9 = vreinterpretq_f16_u32(vcombine_u32(vget_low_u32(_r23_0.val[1]), vget_low_u32(_r23_1.val[1])));
    _ra = vreinterpretq_f16_u32(vcombine_u32(vget_low_u32(_r23_2.val[1]), vget_high_u32(_r23_0.val[1])));
    _rb = vreinterpretq_f16_u32(vcombine_u32(vget_high_u32(_r23_1.val[1]), vget_high_u32(_r23_2.val[1])));
}

static inline void transpose12x4_ph(float16x4_t& _r0l, float16x4_t& _r0m, float16x4_t& _r0h,
                                    float16x4_t& _r1l, float16x4_t& _r1m, float16x4_t& _r1h,
                                    float16x4_t& _r2l, float16x4_t& _r2m, float16x4_t& _r2h,
                                    float16x4_t& _r3l, float16x4_t& _r3m, float16x4_t& _r3h)
{
    uint16x4x2_t _r01lz = vzip_u16(vreinterpret_u16_f16(_r0l), vreinterpret_u16_f16(_r1l));
    uint16x4x2_t _r23lz = vzip_u16(vreinterpret_u16_f16(_r2l), vreinterpret_u16_f16(_r3l));
    uint16x4x2_t _r01mz = vzip_u16(vreinterpret_u16_f16(_r0m), vreinterpret_u16_f16(_r1m));
    uint16x4x2_t _r23mz = vzip_u16(vreinterpret_u16_f16(_r2m), vreinterpret_u16_f16(_r3m));
    uint16x4x2_t _r01hz = vzip_u16(vreinterpret_u16_f16(_r0h), vreinterpret_u16_f16(_r1h));
    uint16x4x2_t _r23hz = vzip_u16(vreinterpret_u16_f16(_r2h), vreinterpret_u16_f16(_r3h));
    uint32x2x2_t _r01 = vzip_u32(vreinterpret_u32_u16(_r01lz.val[0]), vreinterpret_u32_u16(_r23lz.val[0]));
    uint32x2x2_t _r23 = vzip_u32(vreinterpret_u32_u16(_r01lz.val[1]), vreinterpret_u32_u16(_r23lz.val[1]));
    uint32x2x2_t _r45 = vzip_u32(vreinterpret_u32_u16(_r01mz.val[0]), vreinterpret_u32_u16(_r23mz.val[0]));
    uint32x2x2_t _r67 = vzip_u32(vreinterpret_u32_u16(_r01mz.val[1]), vreinterpret_u32_u16(_r23mz.val[1]));
    uint32x2x2_t _r89 = vzip_u32(vreinterpret_u32_u16(_r01hz.val[0]), vreinterpret_u32_u16(_r23hz.val[0]));
    uint32x2x2_t _rab = vzip_u32(vreinterpret_u32_u16(_r01hz.val[1]), vreinterpret_u32_u16(_r23hz.val[1]));
    _r0l = vreinterpret_f16_u32(_r01.val[0]);
    _r0m = vreinterpret_f16_u32(_r01.val[1]);
    _r0h = vreinterpret_f16_u32(_r23.val[0]);
    _r1l = vreinterpret_f16_u32(_r23.val[1]);
    _r1m = vreinterpret_f16_u32(_r45.val[0]);
    _r1h = vreinterpret_f16_u32(_r45.val[1]);
    _r2l = vreinterpret_f16_u32(_r67.val[0]);
    _r2m = vreinterpret_f16_u32(_r67.val[1]);
    _r2h = vreinterpret_f16_u32(_r89.val[0]);
    _r3l = vreinterpret_f16_u32(_r89.val[1]);
    _r3m = vreinterpret_f16_u32(_rab.val[0]);
    _r3h = vreinterpret_f16_u32(_rab.val[1]);
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#endif // __aarch64__
#endif // __ARM_NEON

#endif // ARM_USABILITY_H

// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_PERMUTE_ARM_NEON_H
#define NCNN_PERMUTE_ARM_NEON_H

#include <arm_neon.h>

struct permute_u16x16
{
    uint16x8_t lo;
    uint16x8_t hi;
};

struct permute_f32x8
{
    float32x4_t lo;
    float32x4_t hi;
};

static NCNN_FORCEINLINE float32x4_t permute_movelh_f32(float32x4_t a, float32x4_t b)
{
    return vcombine_f32(vget_low_f32(a), vget_low_f32(b));
}

static NCNN_FORCEINLINE float32x4_t permute_movehl_f32(float32x4_t a, float32x4_t b)
{
    return vcombine_f32(vget_high_f32(b), vget_high_f32(a));
}

static NCNN_FORCEINLINE float32x4_t permute_load_f32x2(const float* ptr)
{
    return vcombine_f32(vld1_f32(ptr), vdup_n_f32(0.f));
}

static NCNN_FORCEINLINE int permute_extract_u32(uint16x8_t v)
{
    return (int)vgetq_lane_u32(vreinterpretq_u32_u16(v), 0);
}

static NCNN_FORCEINLINE uint16x8_t permute_loadl_u16x8(const unsigned short* ptr)
{
    return vcombine_u16(vld1_u16(ptr), vdup_n_u16(0));
}

static NCNN_FORCEINLINE void permute_storel_u16x8(unsigned short* ptr, uint16x8_t v)
{
    vst1_u16(ptr, vget_low_u16(v));
}

static NCNN_FORCEINLINE uint16x8_t permute_set_u32(int v)
{
    return vreinterpretq_u16_u32(vcombine_u32(vdup_n_u32((unsigned int)v), vdup_n_u32(0)));
}

static NCNN_FORCEINLINE uint16x8_t permute_shift_right_u16x8(uint16x8_t v, int imm)
{
    if (imm == 4)
        return vextq_u16(v, vdupq_n_u16(0), 2);
    if (imm == 8)
        return vcombine_u16(vget_high_u16(v), vdup_n_u16(0));
    if (imm == 12)
        return vextq_u16(v, vdupq_n_u16(0), 6);

    unsigned char idx[16];
    for (int i = 0; i < 16; i++)
        idx[i] = (unsigned char)(i + imm < 16 ? i + imm : 255);
    return vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(v), vld1q_u8(idx)));
}

static NCNN_FORCEINLINE uint16x8_t permute_zip1_u32(uint16x8_t a, uint16x8_t b)
{
    return vreinterpretq_u16_u32(vzip1q_u32(vreinterpretq_u32_u16(a), vreinterpretq_u32_u16(b)));
}

static NCNN_FORCEINLINE uint16x8_t permute_zip1_u64(uint16x8_t a, uint16x8_t b)
{
    return vreinterpretq_u16_u64(vzip1q_u64(vreinterpretq_u64_u16(a), vreinterpretq_u64_u16(b)));
}

static NCNN_FORCEINLINE uint16x8_t permute_zip2_u64(uint16x8_t a, uint16x8_t b)
{
    return vreinterpretq_u16_u64(vzip2q_u64(vreinterpretq_u64_u16(a), vreinterpretq_u64_u16(b)));
}

static NCNN_FORCEINLINE permute_u16x16 permute_load_u16x16(const unsigned short* ptr)
{
    permute_u16x16 v;
    v.lo = vld1q_u16(ptr);
    v.hi = vld1q_u16(ptr + 8);
    return v;
}

static NCNN_FORCEINLINE void permute_store_u16x16(unsigned short* ptr, permute_u16x16 v)
{
    vst1q_u16(ptr, v.lo);
    vst1q_u16(ptr + 8, v.hi);
}

static NCNN_FORCEINLINE permute_f32x8 permute_load_f32x8(const float* ptr)
{
    permute_f32x8 v;
    v.lo = vreinterpretq_f32_u16(vld1q_u16((const unsigned short*)ptr));
    v.hi = vreinterpretq_f32_u16(vld1q_u16((const unsigned short*)ptr + 8));
    return v;
}

static NCNN_FORCEINLINE void permute_store_f32x8(float* ptr, permute_f32x8 v)
{
    vst1q_u16((unsigned short*)ptr, vreinterpretq_u16_f32(v.lo));
    vst1q_u16((unsigned short*)ptr + 8, vreinterpretq_u16_f32(v.hi));
}

static NCNN_FORCEINLINE permute_f32x8 permute_shuffle_f32x8_halves(permute_f32x8 a, permute_f32x8 b, const int imm)
{
    permute_f32x8 v;
    const int l0 = (imm >> 0) & 3;
    const int l1 = (imm >> 4) & 3;
    v.lo = (l0 & 2) ? (l0 & 1 ? b.hi : b.lo) : (l0 & 1 ? a.hi : a.lo);
    v.hi = (l1 & 2) ? (l1 & 1 ? b.hi : b.lo) : (l1 & 1 ? a.hi : a.lo);
    return v;
}

static NCNN_FORCEINLINE uint16x8_t permute_deinterleave_u16(uint16x8_t v)
{
    uint16x8x2_t _v = vuzpq_u16(v, v);
    return vcombine_u16(vget_low_u16(_v.val[0]), vget_low_u16(_v.val[1]));
}

#endif // NCNN_PERMUTE_ARM_NEON_H

// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_PERMUTE_ARM_SSE2_H
#define NCNN_PERMUTE_ARM_SSE2_H

#include <arm_neon.h>
#include "arm_usability.h"

typedef float32x4_t __m128;
typedef uint32x4_t __m128i;

#define _MM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))

static NCNN_FORCEINLINE __m128 _mm_loadu_ps(const float* ptr)
{
    return vld1q_f32(ptr);
}

static NCNN_FORCEINLINE void _mm_storeu_ps(float* ptr, __m128 v)
{
    vst1q_f32(ptr, v);
}

static NCNN_FORCEINLINE __m128i _mm_loadu_si128(const __m128i* ptr)
{
    return vld1q_u32((const unsigned int*)ptr);
}

static NCNN_FORCEINLINE void _mm_storeu_si128(__m128i* ptr, __m128i v)
{
    vst1q_u32((unsigned int*)ptr, v);
}

static NCNN_FORCEINLINE __m128i _mm_loadl_epi64(const __m128i* ptr)
{
    return vcombine_u32(vld1_u32((const unsigned int*)ptr), vdup_n_u32(0));
}

static NCNN_FORCEINLINE void _mm_storel_epi64(__m128i* ptr, __m128i v)
{
    vst1_u32((unsigned int*)ptr, vget_low_u32(v));
}

static NCNN_FORCEINLINE __m128i _mm_castps_si128(__m128 v)
{
    return vreinterpretq_u32_f32(v);
}

static NCNN_FORCEINLINE __m128 _mm_castsi128_ps(__m128i v)
{
    return vreinterpretq_f32_u32(v);
}

static NCNN_FORCEINLINE __m128 _mm_movelh_ps(__m128 a, __m128 b)
{
    return vcombine_f32(vget_low_f32(a), vget_low_f32(b));
}

static NCNN_FORCEINLINE __m128 _mm_movehl_ps(__m128 a, __m128 b)
{
    return vcombine_f32(vget_high_f32(b), vget_high_f32(a));
}

static NCNN_FORCEINLINE __m128 _mm_shuffle_ps(__m128 a, __m128 b, const int imm)
{
#if __aarch64__
    uint8x16x2_t tab = {{vreinterpretq_u8_f32(a), vreinterpretq_u8_f32(b)}};
    const int i0 = (imm >> 0) & 3;
    const int i1 = (imm >> 2) & 3;
    const int i2 = 4 + ((imm >> 4) & 3);
    const int i3 = 4 + ((imm >> 6) & 3);
    unsigned char idx[16] =
    {
        (unsigned char)(i0 * 4 + 0), (unsigned char)(i0 * 4 + 1), (unsigned char)(i0 * 4 + 2), (unsigned char)(i0 * 4 + 3),
        (unsigned char)(i1 * 4 + 0), (unsigned char)(i1 * 4 + 1), (unsigned char)(i1 * 4 + 2), (unsigned char)(i1 * 4 + 3),
        (unsigned char)(i2 * 4 + 0), (unsigned char)(i2 * 4 + 1), (unsigned char)(i2 * 4 + 2), (unsigned char)(i2 * 4 + 3),
        (unsigned char)(i3 * 4 + 0), (unsigned char)(i3 * 4 + 1), (unsigned char)(i3 * 4 + 2), (unsigned char)(i3 * 4 + 3)
    };
    return vreinterpretq_f32_u8(vqtbl2q_u8(tab, vld1q_u8(idx)));
#else
    float aa[4];
    float bb[4];
    vst1q_f32(aa, a);
    vst1q_f32(bb, b);
    float cc[4] = {aa[(imm >> 0) & 3], aa[(imm >> 2) & 3], bb[(imm >> 4) & 3], bb[(imm >> 6) & 3]};
    return vld1q_f32(cc);
#endif
}

static NCNN_FORCEINLINE __m128 _mm_unpacklo_ps(__m128 a, __m128 b)
{
#if __aarch64__
    return vzip1q_f32(a, b);
#else
    float32x2_t lo = vzip_f32(vget_low_f32(a), vget_low_f32(b));
    return vcombine_f32(lo, lo);
#endif
}

static NCNN_FORCEINLINE __m128 _mm_unpackhi_ps(__m128 a, __m128 b)
{
#if __aarch64__
    return vzip2q_f32(a, b);
#else
    float32x2_t hi = vzip_f32(vget_high_f32(a), vget_high_f32(b));
    return vcombine_f32(hi, hi);
#endif
}

static NCNN_FORCEINLINE __m128i _mm_srli_si128(__m128i v, int imm)
{
    (void)imm;
    return vcombine_u32(vget_high_u32(v), vdup_n_u32(0));
}

static NCNN_FORCEINLINE void _mm_store_ss(float* ptr, __m128 v)
{
    *ptr = vgetq_lane_f32(v, 0);
}

#define _MM_TRANSPOSE4_PS(r0, r1, r2, r3) transpose4x4_ps(r0, r1, r2, r3)

#endif // NCNN_PERMUTE_ARM_SSE2_H

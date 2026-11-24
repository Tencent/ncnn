// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_PERMUTE_ARM_SSE2_U16_H
#define NCNN_PERMUTE_ARM_SSE2_U16_H

#include <arm_neon.h>

typedef uint16x8_t permute_u16_m128i;

#define _MM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))

static NCNN_FORCEINLINE permute_u16_m128i _mm_loadu_si128(const permute_u16_m128i* ptr)
{
    return vld1q_u16((const unsigned short*)ptr);
}

static NCNN_FORCEINLINE void _mm_storeu_si128(permute_u16_m128i* ptr, permute_u16_m128i v)
{
    vst1q_u16((unsigned short*)ptr, v);
}

static NCNN_FORCEINLINE permute_u16_m128i _mm_loadl_epi64(const permute_u16_m128i* ptr)
{
    return vcombine_u16(vld1_u16((const unsigned short*)ptr), vdup_n_u16(0));
}

static NCNN_FORCEINLINE void _mm_storel_epi64(permute_u16_m128i* ptr, permute_u16_m128i v)
{
    vst1_u16((unsigned short*)ptr, vget_low_u16(v));
}

static NCNN_FORCEINLINE int _mm_cvtsi128_si32(permute_u16_m128i v)
{
    return (int)vgetq_lane_u32(vreinterpretq_u32_u16(v), 0);
}

static NCNN_FORCEINLINE permute_u16_m128i _mm_cvtsi32_si128(int v)
{
    return vreinterpretq_u16_u32(vcombine_u32(vdup_n_u32((unsigned int)v), vdup_n_u32(0)));
}

static NCNN_FORCEINLINE int _mm_extract_epi16(permute_u16_m128i v, const int imm)
{
    return (int)vgetq_lane_u16(v, imm);
}

static NCNN_FORCEINLINE permute_u16_m128i _mm_srli_si128(permute_u16_m128i v, int imm)
{
    unsigned char idx[16];
    for (int i = 0; i < 16; i++)
        idx[i] = (unsigned char)(i + imm < 16 ? i + imm : 255);
    return vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(v), vld1q_u8(idx)));
}

static NCNN_FORCEINLINE permute_u16_m128i _mm_shuffle_epi32(permute_u16_m128i v, const int imm)
{
    unsigned char idx[16];
    for (int i = 0; i < 4; i++)
    {
        const int src = (imm >> (i * 2)) & 3;
        for (int j = 0; j < 4; j++)
            idx[i * 4 + j] = (unsigned char)(src * 4 + j);
    }
    return vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(v), vld1q_u8(idx)));
}

static NCNN_FORCEINLINE permute_u16_m128i _mm_shufflelo_epi16(permute_u16_m128i v, const int imm)
{
    unsigned short vals[8];
    vst1q_u16(vals, v);
    for (int i = 0; i < 4; i++)
        vals[i] = vgetq_lane_u16(v, (imm >> (i * 2)) & 3);
    return vld1q_u16(vals);
}

static NCNN_FORCEINLINE permute_u16_m128i _mm_shufflehi_epi16(permute_u16_m128i v, const int imm)
{
    unsigned short vals[8];
    vst1q_u16(vals, v);
    for (int i = 0; i < 4; i++)
        vals[4 + i] = vgetq_lane_u16(v, 4 + ((imm >> (i * 2)) & 3));
    return vld1q_u16(vals);
}

static NCNN_FORCEINLINE permute_u16_m128i _mm_unpacklo_epi16(permute_u16_m128i a, permute_u16_m128i b)
{
#if __aarch64__
    return vzip1q_u16(a, b);
#else
    uint16x4_t lo = vzip_u16(vget_low_u16(a), vget_low_u16(b));
    return vcombine_u16(lo, lo);
#endif
}

static NCNN_FORCEINLINE permute_u16_m128i _mm_unpackhi_epi16(permute_u16_m128i a, permute_u16_m128i b)
{
#if __aarch64__
    return vzip2q_u16(a, b);
#else
    uint16x4_t hi = vzip_u16(vget_high_u16(a), vget_high_u16(b));
    return vcombine_u16(hi, hi);
#endif
}

static NCNN_FORCEINLINE permute_u16_m128i _mm_unpacklo_epi32(permute_u16_m128i a, permute_u16_m128i b)
{
#if __aarch64__
    return vreinterpretq_u16_u32(vzip1q_u32(vreinterpretq_u32_u16(a), vreinterpretq_u32_u16(b)));
#else
    uint32x2_t lo = vzip_u32(vget_low_u32(vreinterpretq_u32_u16(a)), vget_low_u32(vreinterpretq_u32_u16(b)));
    return vreinterpretq_u16_u32(vcombine_u32(lo, lo));
#endif
}

static NCNN_FORCEINLINE permute_u16_m128i _mm_unpackhi_epi32(permute_u16_m128i a, permute_u16_m128i b)
{
#if __aarch64__
    return vreinterpretq_u16_u32(vzip2q_u32(vreinterpretq_u32_u16(a), vreinterpretq_u32_u16(b)));
#else
    uint32x2_t hi = vzip_u32(vget_high_u32(vreinterpretq_u32_u16(a)), vget_high_u32(vreinterpretq_u32_u16(b)));
    return vreinterpretq_u16_u32(vcombine_u32(hi, hi));
#endif
}

static NCNN_FORCEINLINE permute_u16_m128i _mm_unpacklo_epi64(permute_u16_m128i a, permute_u16_m128i b)
{
#if __aarch64__
    return vreinterpretq_u16_u64(vzip1q_u64(vreinterpretq_u64_u16(a), vreinterpretq_u64_u16(b)));
#else
    return vreinterpretq_u16_u64(vcombine_u64(vget_low_u64(vreinterpretq_u64_u16(a)), vget_low_u64(vreinterpretq_u64_u16(b))));
#endif
}

static NCNN_FORCEINLINE permute_u16_m128i _mm_unpackhi_epi64(permute_u16_m128i a, permute_u16_m128i b)
{
#if __aarch64__
    return vreinterpretq_u16_u64(vzip2q_u64(vreinterpretq_u64_u16(a), vreinterpretq_u64_u16(b)));
#else
    return vreinterpretq_u16_u64(vcombine_u64(vget_high_u64(vreinterpretq_u64_u16(a)), vget_high_u64(vreinterpretq_u64_u16(b))));
#endif
}

static NCNN_FORCEINLINE void transpose8x4_epi16(permute_u16_m128i& _r0, permute_u16_m128i& _r1, permute_u16_m128i& _r2, permute_u16_m128i& _r3)
{
    permute_u16_m128i _tmp0 = _mm_unpacklo_epi16(_r0, _r1);
    permute_u16_m128i _tmp1 = _mm_unpackhi_epi16(_r0, _r1);
    permute_u16_m128i _tmp2 = _mm_unpacklo_epi16(_r2, _r3);
    permute_u16_m128i _tmp3 = _mm_unpackhi_epi16(_r2, _r3);

    _r0 = _mm_unpacklo_epi32(_tmp0, _tmp2);
    _r1 = _mm_unpackhi_epi32(_tmp0, _tmp2);
    _r2 = _mm_unpacklo_epi32(_tmp1, _tmp3);
    _r3 = _mm_unpackhi_epi32(_tmp1, _tmp3);
}

static NCNN_FORCEINLINE void transpose8x8_epi16(permute_u16_m128i& _r0, permute_u16_m128i& _r1, permute_u16_m128i& _r2, permute_u16_m128i& _r3, permute_u16_m128i& _r4, permute_u16_m128i& _r5,
                                                permute_u16_m128i& _r6, permute_u16_m128i& _r7)
{
    permute_u16_m128i _tmp0 = _mm_unpacklo_epi16(_r0, _r1);
    permute_u16_m128i _tmp1 = _mm_unpackhi_epi16(_r0, _r1);
    permute_u16_m128i _tmp2 = _mm_unpacklo_epi16(_r2, _r3);
    permute_u16_m128i _tmp3 = _mm_unpackhi_epi16(_r2, _r3);
    permute_u16_m128i _tmp4 = _mm_unpacklo_epi16(_r4, _r5);
    permute_u16_m128i _tmp5 = _mm_unpackhi_epi16(_r4, _r5);
    permute_u16_m128i _tmp6 = _mm_unpacklo_epi16(_r6, _r7);
    permute_u16_m128i _tmp7 = _mm_unpackhi_epi16(_r6, _r7);

    permute_u16_m128i _tmp8 = _mm_unpacklo_epi32(_tmp0, _tmp2);
    permute_u16_m128i _tmp9 = _mm_unpackhi_epi32(_tmp0, _tmp2);
    permute_u16_m128i _tmpa = _mm_unpacklo_epi32(_tmp1, _tmp3);
    permute_u16_m128i _tmpb = _mm_unpackhi_epi32(_tmp1, _tmp3);
    permute_u16_m128i _tmpc = _mm_unpacklo_epi32(_tmp4, _tmp6);
    permute_u16_m128i _tmpd = _mm_unpackhi_epi32(_tmp4, _tmp6);
    permute_u16_m128i _tmpe = _mm_unpacklo_epi32(_tmp5, _tmp7);
    permute_u16_m128i _tmpf = _mm_unpackhi_epi32(_tmp5, _tmp7);

    _r0 = _mm_unpacklo_epi64(_tmp8, _tmpc);
    _r1 = _mm_unpackhi_epi64(_tmp8, _tmpc);
    _r2 = _mm_unpacklo_epi64(_tmp9, _tmpd);
    _r3 = _mm_unpackhi_epi64(_tmp9, _tmpd);
    _r4 = _mm_unpacklo_epi64(_tmpa, _tmpe);
    _r5 = _mm_unpackhi_epi64(_tmpa, _tmpe);
    _r6 = _mm_unpacklo_epi64(_tmpb, _tmpf);
    _r7 = _mm_unpackhi_epi64(_tmpb, _tmpf);
}

#endif // NCNN_PERMUTE_ARM_SSE2_U16_H

// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void cumulative_sum(float* ptr, int w)
{
    int i = 0;
    float sum = 0.f;
#if __ARM_NEON
    const float32x4_t _zero = vdupq_n_f32(0.f);
    float32x4_t _sum_neon = vdupq_n_f32(sum);
    for (; i + 3 < w; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);

        float32x4_t _t = vextq_f32(_zero, _p, 3);
        _p = vaddq_f32(_p, _t);
        _t = vextq_f32(_zero, _p, 2);
        _p = vaddq_f32(_p, _t);

        _p = vaddq_f32(_p, _sum_neon);
        vst1q_f32(ptr, _p);

        _sum_neon = vdupq_lane_f32(vget_high_f32(_p), 1);

        ptr += 4;
    }

    if (i > 0)
        sum = ptr[-1];
#endif // __ARM_NEON
    for (; i < w; i++)
    {
        sum += *ptr;
        *ptr++ = sum;
    }
}

static void cumulative_sum_add(const float* ptr, float* outptr, int size)
{
    int i = 0;
#if __ARM_NEON
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        float32x4_t _outp = vld1q_f32(outptr);
        vst1q_f32(outptr, vaddq_f32(_outp, _p));

        ptr += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *outptr++ += *ptr++;
    }
}

// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// sdpa decode, one query token per head
// key and value are plain [seqlen][dim] rows of one kv head
// query heads sharing the same kv head are processed together, 4 at a time
// scores are computed block by block and merged with online softmax
//
// the kernels are templates over the storage type T of query, key, value and mask,
// and the output type TO, they compute in fp32. the element access helpers below
// are for fp32, other types declare their own overloads before including this file

static inline float sdpa_cvt(float v)
{
    return v;
}

static inline void sdpa_store1(float* p, float v)
{
    *p = v;
}

#if __ARM_NEON
static inline float32x4_t sdpa_load4(const float* p)
{
    return vld1q_f32(p);
}

static inline float32x4_t sdpa_load4_dup(const float* p)
{
    return vld1q_dup_f32(p);
}

static inline void sdpa_store4(float* p, float32x4_t _v)
{
    vst1q_f32(p, _v);
}

static inline float32x4_t sdpa_fmaq(float32x4_t _c, float32x4_t _a, float32x4_t _b)
{
#if __aarch64__
    return vfmaq_f32(_c, _a, _b);
#else
    return vmlaq_f32(_c, _a, _b);
#endif
}

// [sum(_a0), sum(_a1), sum(_a2), sum(_a3)]
static inline float32x4_t sdpa_hadd4(float32x4_t _a0, float32x4_t _a1, float32x4_t _a2, float32x4_t _a3)
{
#if __aarch64__
    return vpaddq_f32(vpaddq_f32(_a0, _a1), vpaddq_f32(_a2, _a3));
#else
    transpose4x4_ps(_a0, _a1, _a2, _a3);
    return vaddq_f32(vaddq_f32(_a0, _a1), vaddq_f32(_a2, _a3));
#endif
}

static inline float sdpa_reduce_add(float32x4_t _v)
{
#if __aarch64__
    return vaddvq_f32(_v);
#else
    float32x2_t _s = vadd_f32(vget_low_f32(_v), vget_high_f32(_v));
    _s = vpadd_f32(_s, _s);
    return vget_lane_f32(_s, 0);
#endif
}

static inline float sdpa_reduce_max(float32x4_t _v)
{
#if __aarch64__
    return vmaxvq_f32(_v);
#else
    float32x2_t _m = vmax_f32(vget_low_f32(_v), vget_high_f32(_v));
    _m = vpmax_f32(_m, _m);
    return vget_lane_f32(_m, 0);
#endif
}
#endif // __ARM_NEON

static int sdpa_decode_get_optimal_tile_n(int value_dim, int key_seqlen)
{
    // the pv loop walks over one block of values several times,
    // keep the block small enough to stay in l1 cache
    const int value_block_size = 32 * 1024;
    int TILE_N = value_block_size / (value_dim * (int)sizeof(float));
    TILE_N = std::max(8, std::min(TILE_N, 512)) / 8 * 8;

    if (key_seqlen > 0)
    {
        // spread keys evenly over the blocks
        const int num_blocks = (key_seqlen + TILE_N - 1) / TILE_N;
        TILE_N = (key_seqlen + num_blocks - 1) / num_blocks;
        TILE_N = (TILE_N + 7) / 8 * 8;
    }

    return TILE_N;
}

template<typename T>
static const T* sdpa_decode_get_mask_ptr(const Mat& attn_mask_blob, int q)
{
    if (attn_mask_blob.empty())
        return 0;

    if (attn_mask_blob.dims == 3)
        return attn_mask_blob.channel(attn_mask_blob.c > 1 ? q : 0);

    return attn_mask_blob;
}

// one query head
// qptr[head_dim] is already multiplied by scale
// workspace: score[block_n] out[value_dim]
template<typename T, typename TO>
static void sdpa_decode_attention_1(const float* qptr, const Mat& key_head, const Mat& value_head, const T* maskptr, TO* outptr, int block_n, float* score, float* out)
{
    const int head_dim = key_head.w;
    const int value_dim = value_head.w;
    const int key_seqlen = key_head.h;

    float m = -FLT_MAX;
    float l = 0.f;
    memset(out, 0, value_dim * sizeof(float));

    for (int n = 0; n < key_seqlen; n += block_n)
    {
        const int max_jj = std::min(key_seqlen - n, block_n);

        // qk
        float block_max = -FLT_MAX;
        {
            const T* kptr = key_head.row<const T>(n);
            const T* mptr = maskptr ? maskptr + n : 0;

            int j = 0;
#if __ARM_NEON
            float32x4_t _max = vdupq_n_f32(-FLT_MAX);
#if __aarch64__
            for (; j + 7 < max_jj; j += 8)
            {
                const T* k0 = kptr + (size_t)j * head_dim;
                const T* k1 = k0 + head_dim;
                const T* k2 = k1 + head_dim;
                const T* k3 = k2 + head_dim;
                const T* k4 = k3 + head_dim;
                const T* k5 = k4 + head_dim;
                const T* k6 = k5 + head_dim;
                const T* k7 = k6 + head_dim;

                float32x4_t _s0 = vdupq_n_f32(0.f);
                float32x4_t _s1 = vdupq_n_f32(0.f);
                float32x4_t _s2 = vdupq_n_f32(0.f);
                float32x4_t _s3 = vdupq_n_f32(0.f);
                float32x4_t _s4 = vdupq_n_f32(0.f);
                float32x4_t _s5 = vdupq_n_f32(0.f);
                float32x4_t _s6 = vdupq_n_f32(0.f);
                float32x4_t _s7 = vdupq_n_f32(0.f);

                int d = 0;
                for (; d + 3 < head_dim; d += 4)
                {
                    float32x4_t _q = vld1q_f32(qptr + d);
                    _s0 = vfmaq_f32(_s0, _q, sdpa_load4(k0 + d));
                    _s1 = vfmaq_f32(_s1, _q, sdpa_load4(k1 + d));
                    _s2 = vfmaq_f32(_s2, _q, sdpa_load4(k2 + d));
                    _s3 = vfmaq_f32(_s3, _q, sdpa_load4(k3 + d));
                    _s4 = vfmaq_f32(_s4, _q, sdpa_load4(k4 + d));
                    _s5 = vfmaq_f32(_s5, _q, sdpa_load4(k5 + d));
                    _s6 = vfmaq_f32(_s6, _q, sdpa_load4(k6 + d));
                    _s7 = vfmaq_f32(_s7, _q, sdpa_load4(k7 + d));
                }

                float32x4_t _r0 = sdpa_hadd4(_s0, _s1, _s2, _s3);
                float32x4_t _r1 = sdpa_hadd4(_s4, _s5, _s6, _s7);

                if (d < head_dim)
                {
                    float tail[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                    for (; d < head_dim; d++)
                    {
                        const float q = qptr[d];
                        tail[0] += q * sdpa_cvt(k0[d]);
                        tail[1] += q * sdpa_cvt(k1[d]);
                        tail[2] += q * sdpa_cvt(k2[d]);
                        tail[3] += q * sdpa_cvt(k3[d]);
                        tail[4] += q * sdpa_cvt(k4[d]);
                        tail[5] += q * sdpa_cvt(k5[d]);
                        tail[6] += q * sdpa_cvt(k6[d]);
                        tail[7] += q * sdpa_cvt(k7[d]);
                    }
                    _r0 = vaddq_f32(_r0, vld1q_f32(tail));
                    _r1 = vaddq_f32(_r1, vld1q_f32(tail + 4));
                }

                if (mptr)
                {
                    _r0 = vaddq_f32(_r0, sdpa_load4(mptr + j));
                    _r1 = vaddq_f32(_r1, sdpa_load4(mptr + j + 4));
                }

                vst1q_f32(score + j, _r0);
                vst1q_f32(score + j + 4, _r1);
                _max = vmaxq_f32(_max, vmaxq_f32(_r0, _r1));
            }
#endif // __aarch64__
            for (; j + 3 < max_jj; j += 4)
            {
                const T* k0 = kptr + (size_t)j * head_dim;
                const T* k1 = k0 + head_dim;
                const T* k2 = k1 + head_dim;
                const T* k3 = k2 + head_dim;

                float32x4_t _s0 = vdupq_n_f32(0.f);
                float32x4_t _s1 = vdupq_n_f32(0.f);
                float32x4_t _s2 = vdupq_n_f32(0.f);
                float32x4_t _s3 = vdupq_n_f32(0.f);

                int d = 0;
                for (; d + 3 < head_dim; d += 4)
                {
                    float32x4_t _q = vld1q_f32(qptr + d);
                    _s0 = sdpa_fmaq(_s0, _q, sdpa_load4(k0 + d));
                    _s1 = sdpa_fmaq(_s1, _q, sdpa_load4(k1 + d));
                    _s2 = sdpa_fmaq(_s2, _q, sdpa_load4(k2 + d));
                    _s3 = sdpa_fmaq(_s3, _q, sdpa_load4(k3 + d));
                }

                float32x4_t _r = sdpa_hadd4(_s0, _s1, _s2, _s3);

                if (d < head_dim)
                {
                    float tail[4] = {0.f, 0.f, 0.f, 0.f};
                    for (; d < head_dim; d++)
                    {
                        const float q = qptr[d];
                        tail[0] += q * sdpa_cvt(k0[d]);
                        tail[1] += q * sdpa_cvt(k1[d]);
                        tail[2] += q * sdpa_cvt(k2[d]);
                        tail[3] += q * sdpa_cvt(k3[d]);
                    }
                    _r = vaddq_f32(_r, vld1q_f32(tail));
                }

                if (mptr)
                    _r = vaddq_f32(_r, sdpa_load4(mptr + j));

                vst1q_f32(score + j, _r);
                _max = vmaxq_f32(_max, _r);
            }
            block_max = sdpa_reduce_max(_max);
#endif // __ARM_NEON
            for (; j < max_jj; j++)
            {
                const T* k0 = kptr + (size_t)j * head_dim;

                float sum = 0.f;
                int d = 0;
#if __ARM_NEON
                float32x4_t _s = vdupq_n_f32(0.f);
                for (; d + 3 < head_dim; d += 4)
                {
                    _s = sdpa_fmaq(_s, vld1q_f32(qptr + d), sdpa_load4(k0 + d));
                }
                sum = sdpa_reduce_add(_s);
#endif // __ARM_NEON
                for (; d < head_dim; d++)
                {
                    sum += qptr[d] * sdpa_cvt(k0[d]);
                }

                if (mptr)
                    sum += sdpa_cvt(mptr[j]);

                score[j] = sum;
                block_max = std::max(block_max, sum);
            }
        }

        // online softmax
        float alpha;
        {
            const float m_new = std::max(m, block_max);
            alpha = l == 0.f ? 0.f : expf(m - m_new);

            float sum = 0.f;
            int j = 0;
#if __ARM_NEON
            float32x4_t _m_new = vdupq_n_f32(m_new);
            float32x4_t _sum = vdupq_n_f32(0.f);
            for (; j + 3 < max_jj; j += 4)
            {
                float32x4_t _p = exp_ps(vsubq_f32(vld1q_f32(score + j), _m_new));
                vst1q_f32(score + j, _p);
                _sum = vaddq_f32(_sum, _p);
            }
            sum = sdpa_reduce_add(_sum);
#endif // __ARM_NEON
            for (; j < max_jj; j++)
            {
                score[j] = expf(score[j] - m_new);
                sum += score[j];
            }

            l = l * alpha + sum;
            m = m_new;
        }

        // pv
        {
            const T* vptr = value_head.row<const T>(n);

            int d = 0;
#if __ARM_NEON
            float32x4_t _alpha = vdupq_n_f32(alpha);
#if __aarch64__
            for (; d + 63 < value_dim; d += 64)
            {
                float32x4_t _o0 = vmulq_f32(vld1q_f32(out + d), _alpha);
                float32x4_t _o1 = vmulq_f32(vld1q_f32(out + d + 4), _alpha);
                float32x4_t _o2 = vmulq_f32(vld1q_f32(out + d + 8), _alpha);
                float32x4_t _o3 = vmulq_f32(vld1q_f32(out + d + 12), _alpha);
                float32x4_t _o4 = vmulq_f32(vld1q_f32(out + d + 16), _alpha);
                float32x4_t _o5 = vmulq_f32(vld1q_f32(out + d + 20), _alpha);
                float32x4_t _o6 = vmulq_f32(vld1q_f32(out + d + 24), _alpha);
                float32x4_t _o7 = vmulq_f32(vld1q_f32(out + d + 28), _alpha);
                float32x4_t _o8 = vmulq_f32(vld1q_f32(out + d + 32), _alpha);
                float32x4_t _o9 = vmulq_f32(vld1q_f32(out + d + 36), _alpha);
                float32x4_t _oa = vmulq_f32(vld1q_f32(out + d + 40), _alpha);
                float32x4_t _ob = vmulq_f32(vld1q_f32(out + d + 44), _alpha);
                float32x4_t _oc = vmulq_f32(vld1q_f32(out + d + 48), _alpha);
                float32x4_t _od = vmulq_f32(vld1q_f32(out + d + 52), _alpha);
                float32x4_t _oe = vmulq_f32(vld1q_f32(out + d + 56), _alpha);
                float32x4_t _of = vmulq_f32(vld1q_f32(out + d + 60), _alpha);

                const T* pV = vptr + d;
                for (int j = 0; j < max_jj; j++)
                {
                    float32x4_t _p = vld1q_dup_f32(score + j);
                    _o0 = vfmaq_f32(_o0, sdpa_load4(pV), _p);
                    _o1 = vfmaq_f32(_o1, sdpa_load4(pV + 4), _p);
                    _o2 = vfmaq_f32(_o2, sdpa_load4(pV + 8), _p);
                    _o3 = vfmaq_f32(_o3, sdpa_load4(pV + 12), _p);
                    _o4 = vfmaq_f32(_o4, sdpa_load4(pV + 16), _p);
                    _o5 = vfmaq_f32(_o5, sdpa_load4(pV + 20), _p);
                    _o6 = vfmaq_f32(_o6, sdpa_load4(pV + 24), _p);
                    _o7 = vfmaq_f32(_o7, sdpa_load4(pV + 28), _p);
                    _o8 = vfmaq_f32(_o8, sdpa_load4(pV + 32), _p);
                    _o9 = vfmaq_f32(_o9, sdpa_load4(pV + 36), _p);
                    _oa = vfmaq_f32(_oa, sdpa_load4(pV + 40), _p);
                    _ob = vfmaq_f32(_ob, sdpa_load4(pV + 44), _p);
                    _oc = vfmaq_f32(_oc, sdpa_load4(pV + 48), _p);
                    _od = vfmaq_f32(_od, sdpa_load4(pV + 52), _p);
                    _oe = vfmaq_f32(_oe, sdpa_load4(pV + 56), _p);
                    _of = vfmaq_f32(_of, sdpa_load4(pV + 60), _p);
                    pV += value_dim;
                }

                vst1q_f32(out + d, _o0);
                vst1q_f32(out + d + 4, _o1);
                vst1q_f32(out + d + 8, _o2);
                vst1q_f32(out + d + 12, _o3);
                vst1q_f32(out + d + 16, _o4);
                vst1q_f32(out + d + 20, _o5);
                vst1q_f32(out + d + 24, _o6);
                vst1q_f32(out + d + 28, _o7);
                vst1q_f32(out + d + 32, _o8);
                vst1q_f32(out + d + 36, _o9);
                vst1q_f32(out + d + 40, _oa);
                vst1q_f32(out + d + 44, _ob);
                vst1q_f32(out + d + 48, _oc);
                vst1q_f32(out + d + 52, _od);
                vst1q_f32(out + d + 56, _oe);
                vst1q_f32(out + d + 60, _of);
            }
#endif // __aarch64__
            for (; d + 15 < value_dim; d += 16)
            {
                float32x4_t _o0 = vmulq_f32(vld1q_f32(out + d), _alpha);
                float32x4_t _o1 = vmulq_f32(vld1q_f32(out + d + 4), _alpha);
                float32x4_t _o2 = vmulq_f32(vld1q_f32(out + d + 8), _alpha);
                float32x4_t _o3 = vmulq_f32(vld1q_f32(out + d + 12), _alpha);

                const T* pV = vptr + d;
                for (int j = 0; j < max_jj; j++)
                {
                    float32x4_t _p = vld1q_dup_f32(score + j);
                    _o0 = sdpa_fmaq(_o0, sdpa_load4(pV), _p);
                    _o1 = sdpa_fmaq(_o1, sdpa_load4(pV + 4), _p);
                    _o2 = sdpa_fmaq(_o2, sdpa_load4(pV + 8), _p);
                    _o3 = sdpa_fmaq(_o3, sdpa_load4(pV + 12), _p);
                    pV += value_dim;
                }

                vst1q_f32(out + d, _o0);
                vst1q_f32(out + d + 4, _o1);
                vst1q_f32(out + d + 8, _o2);
                vst1q_f32(out + d + 12, _o3);
            }
            for (; d + 3 < value_dim; d += 4)
            {
                float32x4_t _o = vmulq_f32(vld1q_f32(out + d), _alpha);

                const T* pV = vptr + d;
                for (int j = 0; j < max_jj; j++)
                {
                    _o = sdpa_fmaq(_o, sdpa_load4(pV), vld1q_dup_f32(score + j));
                    pV += value_dim;
                }

                vst1q_f32(out + d, _o);
            }
#endif // __ARM_NEON
            for (; d < value_dim; d++)
            {
                float o = out[d] * alpha;

                const T* pV = vptr + d;
                for (int j = 0; j < max_jj; j++)
                {
                    o += score[j] * sdpa_cvt(*pV);
                    pV += value_dim;
                }

                out[d] = o;
            }
        }
    }

    const float inv_l = l == 0.f ? 0.f : 1.f / l;

    int d = 0;
#if __ARM_NEON
    float32x4_t _inv_l = vdupq_n_f32(inv_l);
    for (; d + 3 < value_dim; d += 4)
    {
        sdpa_store4(outptr + d, vmulq_f32(vld1q_f32(out + d), _inv_l));
    }
#endif // __ARM_NEON
    for (; d < value_dim; d++)
    {
        sdpa_store1(outptr + d, out[d] * inv_l);
    }
}

// four query heads sharing one kv head
// qT[head_dim][4] is already multiplied by scale
// maskptrs[4] are per head mask rows, or all null
// workspace: scoreT[block_n][4] outT[value_dim][4]
template<typename T, typename TO>
static void sdpa_decode_attention_4(const float* qT, const Mat& key_head, const Mat& value_head, const T* const* maskptrs, TO* const* outptrs, int block_n, float* scoreT, float* outT)
{
    const int head_dim = key_head.w;
    const int value_dim = value_head.w;
    const int key_seqlen = key_head.h;
    const bool mask_shared = maskptrs[0] == maskptrs[1] && maskptrs[0] == maskptrs[2] && maskptrs[0] == maskptrs[3];

#if __ARM_NEON
    float32x4_t _m = vdupq_n_f32(-FLT_MAX);
    float32x4_t _l = vdupq_n_f32(0.f);
#else
    float m[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    float l[4] = {0.f, 0.f, 0.f, 0.f};
#endif
    memset(outT, 0, value_dim * 4 * sizeof(float));

    for (int n = 0; n < key_seqlen; n += block_n)
    {
        const int max_jj = std::min(key_seqlen - n, block_n);

        // qk, scoreT[j][4]
#if __ARM_NEON
        float32x4_t _block_max = vdupq_n_f32(-FLT_MAX);
#else
        float block_max[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
#endif
        {
            const T* kptr = key_head.row<const T>(n);

            int j = 0;
#if __ARM_NEON
#if __aarch64__
            for (; j + 3 < max_jj; j += 4)
            {
                const T* k0 = kptr + (size_t)j * head_dim;
                const T* k1 = k0 + head_dim;
                const T* k2 = k1 + head_dim;
                const T* k3 = k2 + head_dim;

                // two accumulators per key for more independent fma chains
                float32x4_t _s00 = vdupq_n_f32(0.f);
                float32x4_t _s01 = vdupq_n_f32(0.f);
                float32x4_t _s10 = vdupq_n_f32(0.f);
                float32x4_t _s11 = vdupq_n_f32(0.f);
                float32x4_t _s20 = vdupq_n_f32(0.f);
                float32x4_t _s21 = vdupq_n_f32(0.f);
                float32x4_t _s30 = vdupq_n_f32(0.f);
                float32x4_t _s31 = vdupq_n_f32(0.f);

                const float* pQ = qT;
                int d = 0;
                for (; d + 3 < head_dim; d += 4)
                {
                    float32x4_t _q0 = vld1q_f32(pQ);
                    float32x4_t _q1 = vld1q_f32(pQ + 4);
                    float32x4_t _q2 = vld1q_f32(pQ + 8);
                    float32x4_t _q3 = vld1q_f32(pQ + 12);
                    float32x4_t _k0 = sdpa_load4(k0 + d);
                    float32x4_t _k1 = sdpa_load4(k1 + d);
                    float32x4_t _k2 = sdpa_load4(k2 + d);
                    float32x4_t _k3 = sdpa_load4(k3 + d);
                    _s00 = vfmaq_laneq_f32(_s00, _q0, _k0, 0);
                    _s10 = vfmaq_laneq_f32(_s10, _q0, _k1, 0);
                    _s20 = vfmaq_laneq_f32(_s20, _q0, _k2, 0);
                    _s30 = vfmaq_laneq_f32(_s30, _q0, _k3, 0);
                    _s01 = vfmaq_laneq_f32(_s01, _q1, _k0, 1);
                    _s11 = vfmaq_laneq_f32(_s11, _q1, _k1, 1);
                    _s21 = vfmaq_laneq_f32(_s21, _q1, _k2, 1);
                    _s31 = vfmaq_laneq_f32(_s31, _q1, _k3, 1);
                    _s00 = vfmaq_laneq_f32(_s00, _q2, _k0, 2);
                    _s10 = vfmaq_laneq_f32(_s10, _q2, _k1, 2);
                    _s20 = vfmaq_laneq_f32(_s20, _q2, _k2, 2);
                    _s30 = vfmaq_laneq_f32(_s30, _q2, _k3, 2);
                    _s01 = vfmaq_laneq_f32(_s01, _q3, _k0, 3);
                    _s11 = vfmaq_laneq_f32(_s11, _q3, _k1, 3);
                    _s21 = vfmaq_laneq_f32(_s21, _q3, _k2, 3);
                    _s31 = vfmaq_laneq_f32(_s31, _q3, _k3, 3);
                    pQ += 16;
                }
                for (; d < head_dim; d++)
                {
                    float32x4_t _q = vld1q_f32(pQ);
                    _s00 = vfmaq_n_f32(_s00, _q, sdpa_cvt(k0[d]));
                    _s10 = vfmaq_n_f32(_s10, _q, sdpa_cvt(k1[d]));
                    _s20 = vfmaq_n_f32(_s20, _q, sdpa_cvt(k2[d]));
                    _s30 = vfmaq_n_f32(_s30, _q, sdpa_cvt(k3[d]));
                    pQ += 4;
                }

                float32x4_t _r0 = vaddq_f32(_s00, _s01);
                float32x4_t _r1 = vaddq_f32(_s10, _s11);
                float32x4_t _r2 = vaddq_f32(_s20, _s21);
                float32x4_t _r3 = vaddq_f32(_s30, _s31);

                if (maskptrs[0])
                {
                    const int jn = n + j;
                    if (mask_shared)
                    {
                        _r0 = vaddq_f32(_r0, sdpa_load4_dup(maskptrs[0] + jn));
                        _r1 = vaddq_f32(_r1, sdpa_load4_dup(maskptrs[0] + jn + 1));
                        _r2 = vaddq_f32(_r2, sdpa_load4_dup(maskptrs[0] + jn + 2));
                        _r3 = vaddq_f32(_r3, sdpa_load4_dup(maskptrs[0] + jn + 3));
                    }
                    else
                    {
                        // mask rows are per head, transpose 4 keys x 4 heads
                        float32x4_t _m0 = sdpa_load4(maskptrs[0] + jn);
                        float32x4_t _m1 = sdpa_load4(maskptrs[1] + jn);
                        float32x4_t _m2 = sdpa_load4(maskptrs[2] + jn);
                        float32x4_t _m3 = sdpa_load4(maskptrs[3] + jn);
                        transpose4x4_ps(_m0, _m1, _m2, _m3);
                        _r0 = vaddq_f32(_r0, _m0);
                        _r1 = vaddq_f32(_r1, _m1);
                        _r2 = vaddq_f32(_r2, _m2);
                        _r3 = vaddq_f32(_r3, _m3);
                    }
                }

                vst1q_f32(scoreT + j * 4, _r0);
                vst1q_f32(scoreT + j * 4 + 4, _r1);
                vst1q_f32(scoreT + j * 4 + 8, _r2);
                vst1q_f32(scoreT + j * 4 + 12, _r3);
                _block_max = vmaxq_f32(_block_max, vmaxq_f32(vmaxq_f32(_r0, _r1), vmaxq_f32(_r2, _r3)));
            }
#endif // __aarch64__
            for (; j < max_jj; j++)
            {
                const T* k0 = kptr + (size_t)j * head_dim;

                float32x4_t _s0 = vdupq_n_f32(0.f);
                float32x4_t _s1 = vdupq_n_f32(0.f);

                const float* pQ = qT;
                int d = 0;
                for (; d + 1 < head_dim; d += 2)
                {
                    _s0 = sdpa_fmaq(_s0, vld1q_f32(pQ), sdpa_load4_dup(k0 + d));
                    _s1 = sdpa_fmaq(_s1, vld1q_f32(pQ + 4), sdpa_load4_dup(k0 + d + 1));
                    pQ += 8;
                }
                for (; d < head_dim; d++)
                {
                    _s0 = sdpa_fmaq(_s0, vld1q_f32(pQ), sdpa_load4_dup(k0 + d));
                    pQ += 4;
                }

                float32x4_t _r = vaddq_f32(_s0, _s1);

                if (maskptrs[0])
                {
                    const int jn = n + j;
                    if (mask_shared)
                    {
                        _r = vaddq_f32(_r, sdpa_load4_dup(maskptrs[0] + jn));
                    }
                    else
                    {
                        float mask[4] = {sdpa_cvt(maskptrs[0][jn]), sdpa_cvt(maskptrs[1][jn]), sdpa_cvt(maskptrs[2][jn]), sdpa_cvt(maskptrs[3][jn])};
                        _r = vaddq_f32(_r, vld1q_f32(mask));
                    }
                }

                vst1q_f32(scoreT + j * 4, _r);
                _block_max = vmaxq_f32(_block_max, _r);
            }
#else  // __ARM_NEON
            for (; j < max_jj; j++)
            {
                const T* k0 = kptr + (size_t)j * head_dim;
                for (int h = 0; h < 4; h++)
                {
                    float sum = 0.f;
                    for (int d = 0; d < head_dim; d++)
                    {
                        sum += qT[d * 4 + h] * sdpa_cvt(k0[d]);
                    }
                    if (maskptrs[h])
                        sum += sdpa_cvt(maskptrs[h][n + j]);
                    scoreT[j * 4 + h] = sum;
                    block_max[h] = std::max(block_max[h], sum);
                }
            }
#endif // __ARM_NEON
        }

        // online softmax, all four heads at once
#if __ARM_NEON
        float32x4_t _alpha;
        {
            float32x4_t _m_new = vmaxq_f32(_m, _block_max);

            // alpha is 0 for heads that have no history yet
            uint32x4_t _active = vmvnq_u32(vceqq_f32(_l, vdupq_n_f32(0.f)));
            _alpha = exp_ps(vreinterpretq_f32_u32(vandq_u32(_active, vreinterpretq_u32_f32(vsubq_f32(_m, _m_new)))));
            _alpha = vreinterpretq_f32_u32(vandq_u32(_active, vreinterpretq_u32_f32(_alpha)));

            float32x4_t _sum = vdupq_n_f32(0.f);
            for (int j = 0; j < max_jj; j++)
            {
                float32x4_t _p = exp_ps(vsubq_f32(vld1q_f32(scoreT + j * 4), _m_new));
                vst1q_f32(scoreT + j * 4, _p);
                _sum = vaddq_f32(_sum, _p);
            }

            _l = sdpa_fmaq(_sum, _l, _alpha);
            _m = _m_new;
        }
#else  // __ARM_NEON
        float alpha[4];
        for (int h = 0; h < 4; h++)
        {
            const float m_new = std::max(m[h], block_max[h]);
            alpha[h] = l[h] == 0.f ? 0.f : expf(m[h] - m_new);

            float sum = 0.f;
            for (int j = 0; j < max_jj; j++)
            {
                scoreT[j * 4 + h] = expf(scoreT[j * 4 + h] - m_new);
                sum += scoreT[j * 4 + h];
            }

            l[h] = l[h] * alpha[h] + sum;
            m[h] = m_new;
        }
#endif // __ARM_NEON

        // pv, outT[d][4]
        {
            const T* vptr = value_head.row<const T>(n);

            int d = 0;
#if __ARM_NEON
#if __aarch64__
            for (; d + 15 < value_dim; d += 16)
            {
                float* pO = outT + d * 4;
                float32x4_t _o0 = vmulq_f32(vld1q_f32(pO), _alpha);
                float32x4_t _o1 = vmulq_f32(vld1q_f32(pO + 4), _alpha);
                float32x4_t _o2 = vmulq_f32(vld1q_f32(pO + 8), _alpha);
                float32x4_t _o3 = vmulq_f32(vld1q_f32(pO + 12), _alpha);
                float32x4_t _o4 = vmulq_f32(vld1q_f32(pO + 16), _alpha);
                float32x4_t _o5 = vmulq_f32(vld1q_f32(pO + 20), _alpha);
                float32x4_t _o6 = vmulq_f32(vld1q_f32(pO + 24), _alpha);
                float32x4_t _o7 = vmulq_f32(vld1q_f32(pO + 28), _alpha);
                float32x4_t _o8 = vmulq_f32(vld1q_f32(pO + 32), _alpha);
                float32x4_t _o9 = vmulq_f32(vld1q_f32(pO + 36), _alpha);
                float32x4_t _oa = vmulq_f32(vld1q_f32(pO + 40), _alpha);
                float32x4_t _ob = vmulq_f32(vld1q_f32(pO + 44), _alpha);
                float32x4_t _oc = vmulq_f32(vld1q_f32(pO + 48), _alpha);
                float32x4_t _od = vmulq_f32(vld1q_f32(pO + 52), _alpha);
                float32x4_t _oe = vmulq_f32(vld1q_f32(pO + 56), _alpha);
                float32x4_t _of = vmulq_f32(vld1q_f32(pO + 60), _alpha);

                const T* pV = vptr + d;
                for (int j = 0; j < max_jj; j++)
                {
                    float32x4_t _p = vld1q_f32(scoreT + j * 4);
                    float32x4_t _v0 = sdpa_load4(pV);
                    float32x4_t _v1 = sdpa_load4(pV + 4);
                    float32x4_t _v2 = sdpa_load4(pV + 8);
                    float32x4_t _v3 = sdpa_load4(pV + 12);
                    _o0 = vfmaq_laneq_f32(_o0, _p, _v0, 0);
                    _o1 = vfmaq_laneq_f32(_o1, _p, _v0, 1);
                    _o2 = vfmaq_laneq_f32(_o2, _p, _v0, 2);
                    _o3 = vfmaq_laneq_f32(_o3, _p, _v0, 3);
                    _o4 = vfmaq_laneq_f32(_o4, _p, _v1, 0);
                    _o5 = vfmaq_laneq_f32(_o5, _p, _v1, 1);
                    _o6 = vfmaq_laneq_f32(_o6, _p, _v1, 2);
                    _o7 = vfmaq_laneq_f32(_o7, _p, _v1, 3);
                    _o8 = vfmaq_laneq_f32(_o8, _p, _v2, 0);
                    _o9 = vfmaq_laneq_f32(_o9, _p, _v2, 1);
                    _oa = vfmaq_laneq_f32(_oa, _p, _v2, 2);
                    _ob = vfmaq_laneq_f32(_ob, _p, _v2, 3);
                    _oc = vfmaq_laneq_f32(_oc, _p, _v3, 0);
                    _od = vfmaq_laneq_f32(_od, _p, _v3, 1);
                    _oe = vfmaq_laneq_f32(_oe, _p, _v3, 2);
                    _of = vfmaq_laneq_f32(_of, _p, _v3, 3);
                    pV += value_dim;
                }

                vst1q_f32(pO, _o0);
                vst1q_f32(pO + 4, _o1);
                vst1q_f32(pO + 8, _o2);
                vst1q_f32(pO + 12, _o3);
                vst1q_f32(pO + 16, _o4);
                vst1q_f32(pO + 20, _o5);
                vst1q_f32(pO + 24, _o6);
                vst1q_f32(pO + 28, _o7);
                vst1q_f32(pO + 32, _o8);
                vst1q_f32(pO + 36, _o9);
                vst1q_f32(pO + 40, _oa);
                vst1q_f32(pO + 44, _ob);
                vst1q_f32(pO + 48, _oc);
                vst1q_f32(pO + 52, _od);
                vst1q_f32(pO + 56, _oe);
                vst1q_f32(pO + 60, _of);
            }
#endif // __aarch64__
            for (; d + 3 < value_dim; d += 4)
            {
                float* pO = outT + d * 4;
                float32x4_t _o0 = vmulq_f32(vld1q_f32(pO), _alpha);
                float32x4_t _o1 = vmulq_f32(vld1q_f32(pO + 4), _alpha);
                float32x4_t _o2 = vmulq_f32(vld1q_f32(pO + 8), _alpha);
                float32x4_t _o3 = vmulq_f32(vld1q_f32(pO + 12), _alpha);

                const T* pV = vptr + d;
                for (int j = 0; j < max_jj; j++)
                {
                    float32x4_t _p = vld1q_f32(scoreT + j * 4);
                    _o0 = sdpa_fmaq(_o0, _p, sdpa_load4_dup(pV));
                    _o1 = sdpa_fmaq(_o1, _p, sdpa_load4_dup(pV + 1));
                    _o2 = sdpa_fmaq(_o2, _p, sdpa_load4_dup(pV + 2));
                    _o3 = sdpa_fmaq(_o3, _p, sdpa_load4_dup(pV + 3));
                    pV += value_dim;
                }

                vst1q_f32(pO, _o0);
                vst1q_f32(pO + 4, _o1);
                vst1q_f32(pO + 8, _o2);
                vst1q_f32(pO + 12, _o3);
            }
            for (; d < value_dim; d++)
            {
                float* pO = outT + d * 4;
                float32x4_t _o = vmulq_f32(vld1q_f32(pO), _alpha);

                const T* pV = vptr + d;
                for (int j = 0; j < max_jj; j++)
                {
                    _o = sdpa_fmaq(_o, vld1q_f32(scoreT + j * 4), sdpa_load4_dup(pV));
                    pV += value_dim;
                }

                vst1q_f32(pO, _o);
            }
#else  // __ARM_NEON
            for (; d < value_dim; d++)
            {
                for (int h = 0; h < 4; h++)
                {
                    float o = outT[d * 4 + h] * alpha[h];

                    const T* pV = vptr + d;
                    for (int j = 0; j < max_jj; j++)
                    {
                        o += scoreT[j * 4 + h] * sdpa_cvt(*pV);
                        pV += value_dim;
                    }

                    outT[d * 4 + h] = o;
                }
            }
#endif // __ARM_NEON
        }
    }

    // outT[d][4] / l -> out[4][value_dim]
    float inv_l[4];
#if __ARM_NEON
    float l[4];
    vst1q_f32(l, _l);
#endif
    for (int h = 0; h < 4; h++)
    {
        inv_l[h] = l[h] == 0.f ? 0.f : 1.f / l[h];
    }

    int d = 0;
#if __ARM_NEON
    for (; d + 3 < value_dim; d += 4)
    {
        float32x4_t _r0 = vld1q_f32(outT + d * 4);
        float32x4_t _r1 = vld1q_f32(outT + d * 4 + 4);
        float32x4_t _r2 = vld1q_f32(outT + d * 4 + 8);
        float32x4_t _r3 = vld1q_f32(outT + d * 4 + 12);
        transpose4x4_ps(_r0, _r1, _r2, _r3);
        sdpa_store4(outptrs[0] + d, vmulq_n_f32(_r0, inv_l[0]));
        sdpa_store4(outptrs[1] + d, vmulq_n_f32(_r1, inv_l[1]));
        sdpa_store4(outptrs[2] + d, vmulq_n_f32(_r2, inv_l[2]));
        sdpa_store4(outptrs[3] + d, vmulq_n_f32(_r3, inv_l[3]));
    }
#endif // __ARM_NEON
    for (; d < value_dim; d++)
    {
        for (int h = 0; h < 4; h++)
        {
            sdpa_store1(outptrs[h] + d, outT[d * 4 + h] * inv_l[h]);
        }
    }
}

template<typename T, typename TO>
static int sdpa_decode(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
{
    const int head_dim = query.w;
    const int value_dim = value.w;
    const int key_seqlen = key.h;
    const int num_heads = query.c;
    const int num_kv_heads = key.c;
    const int num_heads_per_kv_head = num_heads / num_kv_heads;
    const int num_qblocks = (num_heads_per_kv_head + 3) / 4;
    const int num_tasks = num_kv_heads * num_qblocks;
    const int nT = std::min(std::max(opt.num_threads, 1), num_tasks);
    const int block_n = sdpa_decode_get_optimal_tile_n(value_dim, key_seqlen);

    // per thread: packed query, scores and output accumulator for up to 4 heads
    const int workspace_size = 4 * head_dim + 4 * block_n + 4 * value_dim;
    Mat workspace(workspace_size, 1, nT, 4u, opt.workspace_allocator);
    if (workspace.empty())
        return -100;

    #pragma omp parallel for num_threads(nT)
    for (int task_id = 0; task_id < num_tasks; task_id++)
    {
        const int g = task_id / num_qblocks;
        const int qblock_id = task_id % num_qblocks;
        const int q0 = g * num_heads_per_kv_head + qblock_id * 4;
        const int max_qq = std::min(num_heads_per_kv_head - qblock_id * 4, 4);

        float* qbuf = workspace.channel(get_omp_thread_num());
        float* score = qbuf + 4 * head_dim;
        float* out = score + 4 * block_n;

        const Mat key_head = key.channel(g);
        const Mat value_head = value.channel(g);

        if (max_qq == 4)
        {
            // qbuf[head_dim][4]
            for (int h = 0; h < 4; h++)
            {
                const T* qptr = query.channel(q0 + h);
                for (int d = 0; d < head_dim; d++)
                {
                    qbuf[d * 4 + h] = sdpa_cvt(qptr[d]) * scale;
                }
            }

            const T* maskptrs[4];
            TO* outptrs[4];
            for (int h = 0; h < 4; h++)
            {
                maskptrs[h] = sdpa_decode_get_mask_ptr<T>(attn_mask_blob, q0 + h);
                outptrs[h] = top_blob.channel(q0 + h);
            }

            sdpa_decode_attention_4<T, TO>(qbuf, key_head, value_head, maskptrs, outptrs, block_n, score, out);
        }
        else
        {
            for (int qq = 0; qq < max_qq; qq++)
            {
                const int q = q0 + qq;
                const T* qptr = query.channel(q);
                for (int d = 0; d < head_dim; d++)
                {
                    qbuf[d] = sdpa_cvt(qptr[d]) * scale;
                }

                TO* outptr = top_blob.channel(q);
                sdpa_decode_attention_1<T, TO>(qbuf, key_head, value_head, sdpa_decode_get_mask_ptr<T>(attn_mask_blob, q), outptr, block_n, score, out);
            }
        }
    }

    return 0;
}

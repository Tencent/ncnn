// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <math.h>

static int pack_B_wq_int8(const Mat& B, const Mat& B_scales, Mat& BT, Mat& BT_descales, int N, int K, int block_size, int num_threads)
{
    const int block_count = (K + block_size - 1) / block_size;
    Mat BT_packed(N * K, (size_t)1u);
    Mat BT_packed_descales(N * block_count, (size_t)4u);
    if (BT_packed.empty() || BT_packed_descales.empty())
        return -100;
    BT_packed.cstep = (size_t)N * K;
    BT_packed_descales.cstep = (size_t)N * block_count;

    int panel_start = 0;
    int panel_count = 0;
#if __loongarch_sx
#if __loongarch_asx
    const int nn8 = (N - panel_start) / 8;
    const int panel_start8 = panel_start;
    panel_start += nn8 * 8;
    panel_count += nn8;
#endif
    const int nn4 = (N - panel_start) / 4;
    const int panel_start4 = panel_start;
    panel_start += nn4 * 4;
    panel_count += nn4;
#endif
    const int nn2 = (N - panel_start) / 2;
    const int panel_start2 = panel_start;
    panel_start += nn2 * 2;
    panel_count += nn2;
    const int nn1 = N - panel_start;
    const int panel_start1 = panel_start;
    panel_count += nn1;

    #pragma omp parallel for num_threads(num_threads)
    for (int p = 0; p < panel_count; p++)
    {
        int q = p;
        int j = 0;
        int nr = 1;
#if __loongarch_sx
#if __loongarch_asx
        if (q < nn8)
        {
            j = panel_start8 + q * 8;
            nr = 8;
        }
        else
        {
            q -= nn8;
#endif
            if (q < nn4)
            {
                j = panel_start4 + q * 4;
                nr = 4;
            }
            else
            {
                q -= nn4;
#endif
                if (q < nn2)
                {
                    j = panel_start2 + q * 2;
                    nr = 2;
                }
                else
                {
                    q -= nn2;
                    j = panel_start1 + q;
                    nr = 1;
                }
#if __loongarch_sx
            }
#if __loongarch_asx
        }
#endif
#endif

        signed char* pp = (signed char*)BT_packed + j * K;
        float* pd = (float*)BT_packed_descales + j * block_count;

#if __loongarch_sx
#if __loongarch_asx
        if (nr == 8)
        {
            const signed char* p0 = B.row<const signed char>(j);
            const signed char* p1 = B.row<const signed char>(j + 1);
            const signed char* p2 = B.row<const signed char>(j + 2);
            const signed char* p3 = B.row<const signed char>(j + 3);
            const signed char* p4 = B.row<const signed char>(j + 4);
            const signed char* p5 = B.row<const signed char>(j + 5);
            const signed char* p6 = B.row<const signed char>(j + 6);
            const signed char* p7 = B.row<const signed char>(j + 7);
            const float* s0 = B_scales.row(j);
            const float* s1 = B_scales.row(j + 1);
            const float* s2 = B_scales.row(j + 2);
            const float* s3 = B_scales.row(j + 3);
            const float* s4 = B_scales.row(j + 4);
            const float* s5 = B_scales.row(j + 5);
            const float* s6 = B_scales.row(j + 6);
            const float* s7 = B_scales.row(j + 7);

            for (int g = 0; g < block_count; g++)
            {
                const int k0 = g * block_size;
                const int max_kk = std::min(K - k0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _p0 = __lsx_vldrepl_w(p0, 0);
                    __m128i _p1 = __lsx_vldrepl_w(p1, 0);
                    __m128i _p2 = __lsx_vldrepl_w(p2, 0);
                    __m128i _p3 = __lsx_vldrepl_w(p3, 0);
                    __m128i _p4 = __lsx_vldrepl_w(p4, 0);
                    __m128i _p5 = __lsx_vldrepl_w(p5, 0);
                    __m128i _p6 = __lsx_vldrepl_w(p6, 0);
                    __m128i _p7 = __lsx_vldrepl_w(p7, 0);
                    __m128i _p01 = __lsx_vilvl_w(_p1, _p0);
                    __m128i _p23 = __lsx_vilvl_w(_p3, _p2);
                    __m128i _p45 = __lsx_vilvl_w(_p5, _p4);
                    __m128i _p67 = __lsx_vilvl_w(_p7, _p6);
                    __m128i _p0123 = __lsx_vilvl_d(_p23, _p01);
                    __m128i _p4567 = __lsx_vilvl_d(_p67, _p45);
                    __lasx_xvst(__lasx_concat_128(_p0123, _p4567), pp, 0);
                    pp += 32;
                    p0 += 4;
                    p1 += 4;
                    p2 += 4;
                    p3 += 4;
                    p4 += 4;
                    p5 += 4;
                    p6 += 4;
                    p7 += 4;
                }
                if (kk + 1 < max_kk)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p1[0];
                    pp[3] = p1[1];
                    pp[4] = p2[0];
                    pp[5] = p2[1];
                    pp[6] = p3[0];
                    pp[7] = p3[1];
                    pp[8] = p4[0];
                    pp[9] = p4[1];
                    pp[10] = p5[0];
                    pp[11] = p5[1];
                    pp[12] = p6[0];
                    pp[13] = p6[1];
                    pp[14] = p7[0];
                    pp[15] = p7[1];
                    pp += 16;
                    p0 += 2;
                    p1 += 2;
                    p2 += 2;
                    p3 += 2;
                    p4 += 2;
                    p5 += 2;
                    p6 += 2;
                    p7 += 2;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    pp[0] = *p0++;
                    pp[1] = *p1++;
                    pp[2] = *p2++;
                    pp[3] = *p3++;
                    pp[4] = *p4++;
                    pp[5] = *p5++;
                    pp[6] = *p6++;
                    pp[7] = *p7++;
                    pp += 8;
                }

                *pd++ = 1.f / *s0++;
                *pd++ = 1.f / *s1++;
                *pd++ = 1.f / *s2++;
                *pd++ = 1.f / *s3++;
                *pd++ = 1.f / *s4++;
                *pd++ = 1.f / *s5++;
                *pd++ = 1.f / *s6++;
                *pd++ = 1.f / *s7++;
            }

            continue;
        }
#endif
        if (nr == 4)
        {
            const signed char* p0 = B.row<const signed char>(j);
            const signed char* p1 = B.row<const signed char>(j + 1);
            const signed char* p2 = B.row<const signed char>(j + 2);
            const signed char* p3 = B.row<const signed char>(j + 3);
            const float* s0 = B_scales.row(j);
            const float* s1 = B_scales.row(j + 1);
            const float* s2 = B_scales.row(j + 2);
            const float* s3 = B_scales.row(j + 3);

            for (int g = 0; g < block_count; g++)
            {
                const int k0 = g * block_size;
                const int max_kk = std::min(K - k0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _p0 = __lsx_vldrepl_w(p0, 0);
                    __m128i _p1 = __lsx_vldrepl_w(p1, 0);
                    __m128i _p2 = __lsx_vldrepl_w(p2, 0);
                    __m128i _p3 = __lsx_vldrepl_w(p3, 0);
                    __m128i _p01 = __lsx_vilvl_w(_p1, _p0);
                    __m128i _p23 = __lsx_vilvl_w(_p3, _p2);
                    __lsx_vst(__lsx_vilvl_d(_p23, _p01), pp, 0);
                    pp += 16;
                    p0 += 4;
                    p1 += 4;
                    p2 += 4;
                    p3 += 4;
                }
                if (kk + 1 < max_kk)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p1[0];
                    pp[3] = p1[1];
                    pp[4] = p2[0];
                    pp[5] = p2[1];
                    pp[6] = p3[0];
                    pp[7] = p3[1];
                    pp += 8;
                    p0 += 2;
                    p1 += 2;
                    p2 += 2;
                    p3 += 2;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    pp[0] = *p0++;
                    pp[1] = *p1++;
                    pp[2] = *p2++;
                    pp[3] = *p3++;
                    pp += 4;
                }

                *pd++ = 1.f / *s0++;
                *pd++ = 1.f / *s1++;
                *pd++ = 1.f / *s2++;
                *pd++ = 1.f / *s3++;
            }

            continue;
        }
#endif
        if (nr == 2)
        {
            const signed char* p0 = B.row<const signed char>(j);
            const signed char* p1 = B.row<const signed char>(j + 1);
            const float* s0 = B_scales.row(j);
            const float* s1 = B_scales.row(j + 1);

            for (int g = 0; g < block_count; g++)
            {
                const int k0 = g * block_size;
                const int max_kk = std::min(K - k0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p0[2];
                    pp[3] = p0[3];
                    pp[4] = p1[0];
                    pp[5] = p1[1];
                    pp[6] = p1[2];
                    pp[7] = p1[3];
                    pp += 8;
                    p0 += 4;
                    p1 += 4;
                }
                if (kk + 1 < max_kk)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p1[0];
                    pp[3] = p1[1];
                    pp += 4;
                    p0 += 2;
                    p1 += 2;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    pp[0] = *p0++;
                    pp[1] = *p1++;
                    pp += 2;
                }

                *pd++ = 1.f / *s0++;
                *pd++ = 1.f / *s1++;
            }

            continue;
        }

        const signed char* p0 = B.row<const signed char>(j);
        const float* s0 = B_scales.row(j);
        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += 4;
            }
            if (kk + 1 < max_kk)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += 2;
                kk += 2;
            }
            if (kk < max_kk)
                *pp++ = *p0++;

            *pd++ = 1.f / *s0++;
        }
    }

    BT = BT_packed;
    BT_descales = BT_packed_descales;
    return 0;
}

static void quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const float* input_scale_ptr)
{
    signed char* outptr = AT_tile;
    const int out_hstep = AT_tile.w;
    float* descales = AT_descales_tile;
    const int descales_hstep = AT_descales_tile.w;
    const int K = max_kk;
    const int block_count = (K + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
    const float* A_data = (const float*)A + k;
    input_scale_ptr = input_scale_ptr ? input_scale_ptr + k : 0;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = A_data + (i + ii) * A_hstep;
        const float* p1 = p0 + A_hstep;
        const float* p2 = p1 + A_hstep;
        const float* p3 = p2 + A_hstep;
        const float* p4 = p3 + A_hstep;
        const float* p5 = p4 + A_hstep;
        const float* p6 = p5 + A_hstep;
        const float* p7 = p6 + A_hstep;

        signed char* pp = outptr + ii * out_hstep;
        float* pd = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            const float* p0g = p0 + k0;
            const float* p1g = p1 + k0;
            const float* p2g = p2 + k0;
            const float* p3g = p3 + k0;
            const float* p4g = p4 + k0;
            const float* p5g = p5 + k0;
            const float* p6g = p6 + k0;
            const float* p7g = p7 + k0;
            const float* sg = input_scale_ptr ? input_scale_ptr + k0 : 0;
            const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
            __m128 _absmax0 = (__m128)__lsx_vldi(0);
            __m128 _absmax1 = (__m128)__lsx_vldi(0);
            __m128 _absmax2 = (__m128)__lsx_vldi(0);
            __m128 _absmax3 = (__m128)__lsx_vldi(0);
            __m128 _absmax4 = (__m128)__lsx_vldi(0);
            __m128 _absmax5 = (__m128)__lsx_vldi(0);
            __m128 _absmax6 = (__m128)__lsx_vldi(0);
            __m128 _absmax7 = (__m128)__lsx_vldi(0);
            const float* p0a = p0g;
            const float* p1a = p1g;
            const float* p2a = p2g;
            const float* p3a = p3g;
            const float* p4a = p4g;
            const float* p5a = p5g;
            const float* p6a = p6g;
            const float* p7a = p7g;
            const float* psa = sg;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _v0 = (__m128)__lsx_vld(p0a, 0);
                __m128 _v1 = (__m128)__lsx_vld(p1a, 0);
                __m128 _v2 = (__m128)__lsx_vld(p2a, 0);
                __m128 _v3 = (__m128)__lsx_vld(p3a, 0);
                __m128 _v4 = (__m128)__lsx_vld(p4a, 0);
                __m128 _v5 = (__m128)__lsx_vld(p5a, 0);
                __m128 _v6 = (__m128)__lsx_vld(p6a, 0);
                __m128 _v7 = (__m128)__lsx_vld(p7a, 0);
                if (psa)
                {
                    __m128 _s = (__m128)__lsx_vld(psa, 0);
                    _v0 = __lsx_vfmul_s(_v0, _s);
                    _v1 = __lsx_vfmul_s(_v1, _s);
                    _v2 = __lsx_vfmul_s(_v2, _s);
                    _v3 = __lsx_vfmul_s(_v3, _s);
                    _v4 = __lsx_vfmul_s(_v4, _s);
                    _v5 = __lsx_vfmul_s(_v5, _s);
                    _v6 = __lsx_vfmul_s(_v6, _s);
                    _v7 = __lsx_vfmul_s(_v7, _s);
                }
                _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_v0, _abs_mask));
                _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_v1, _abs_mask));
                _absmax2 = __lsx_vfmax_s(_absmax2, (__m128)__lsx_vand_v((__m128i)_v2, _abs_mask));
                _absmax3 = __lsx_vfmax_s(_absmax3, (__m128)__lsx_vand_v((__m128i)_v3, _abs_mask));
                _absmax4 = __lsx_vfmax_s(_absmax4, (__m128)__lsx_vand_v((__m128i)_v4, _abs_mask));
                _absmax5 = __lsx_vfmax_s(_absmax5, (__m128)__lsx_vand_v((__m128i)_v5, _abs_mask));
                _absmax6 = __lsx_vfmax_s(_absmax6, (__m128)__lsx_vand_v((__m128i)_v6, _abs_mask));
                _absmax7 = __lsx_vfmax_s(_absmax7, (__m128)__lsx_vand_v((__m128i)_v7, _abs_mask));
                p0a += 4;
                p1a += 4;
                p2a += 4;
                p3a += 4;
                p4a += 4;
                p5a += 4;
                p6a += 4;
                p7a += 4;
                if (psa)
                    psa += 4;
            }
            float absmax0 = __lsx_reduce_fmax_s(_absmax0);
            float absmax1 = __lsx_reduce_fmax_s(_absmax1);
            float absmax2 = __lsx_reduce_fmax_s(_absmax2);
            float absmax3 = __lsx_reduce_fmax_s(_absmax3);
            float absmax4 = __lsx_reduce_fmax_s(_absmax4);
            float absmax5 = __lsx_reduce_fmax_s(_absmax5);
            float absmax6 = __lsx_reduce_fmax_s(_absmax6);
            float absmax7 = __lsx_reduce_fmax_s(_absmax7);
            for (; kk < max_kk; kk++)
            {
                const float s = psa ? *psa++ : 1.f;
                absmax0 = std::max(absmax0, fabsf(*p0a++ * s));
                absmax1 = std::max(absmax1, fabsf(*p1a++ * s));
                absmax2 = std::max(absmax2, fabsf(*p2a++ * s));
                absmax3 = std::max(absmax3, fabsf(*p3a++ * s));
                absmax4 = std::max(absmax4, fabsf(*p4a++ * s));
                absmax5 = std::max(absmax5, fabsf(*p5a++ * s));
                absmax6 = std::max(absmax6, fabsf(*p6a++ * s));
                absmax7 = std::max(absmax7, fabsf(*p7a++ * s));
            }

            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd[2] = absmax2 / 127.f;
            pd[3] = absmax3 / 127.f;
            pd[4] = absmax4 / 127.f;
            pd[5] = absmax5 / 127.f;
            pd[6] = absmax6 / 127.f;
            pd[7] = absmax7 / 127.f;
            pd += 8;

            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
            const float scale2 = absmax2 == 0.f ? 0.f : 127.f / absmax2;
            const float scale3 = absmax3 == 0.f ? 0.f : 127.f / absmax3;
            const float scale4 = absmax4 == 0.f ? 0.f : 127.f / absmax4;
            const float scale5 = absmax5 == 0.f ? 0.f : 127.f / absmax5;
            const float scale6 = absmax6 == 0.f ? 0.f : 127.f / absmax6;
            const float scale7 = absmax7 == 0.f ? 0.f : 127.f / absmax7;
            __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
            __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
            __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
            __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
            __m128 _scale4 = __lsx_vreplfr2vr_s(scale4);
            __m128 _scale5 = __lsx_vreplfr2vr_s(scale5);
            __m128 _scale6 = __lsx_vreplfr2vr_s(scale6);
            __m128 _scale7 = __lsx_vreplfr2vr_s(scale7);
            const float* p0q = p0g;
            const float* p1q = p1g;
            const float* p2q = p2g;
            const float* p3q = p3g;
            const float* p4q = p4g;
            const float* p5q = p5g;
            const float* p6q = p6g;
            const float* p7q = p7g;
            const float* psq = sg;
            kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _v0 = (__m128)__lsx_vld(p0q, 0);
                __m128 _v1 = (__m128)__lsx_vld(p1q, 0);
                __m128 _v2 = (__m128)__lsx_vld(p2q, 0);
                __m128 _v3 = (__m128)__lsx_vld(p3q, 0);
                __m128 _v4 = (__m128)__lsx_vld(p4q, 0);
                __m128 _v5 = (__m128)__lsx_vld(p5q, 0);
                __m128 _v6 = (__m128)__lsx_vld(p6q, 0);
                __m128 _v7 = (__m128)__lsx_vld(p7q, 0);
                if (psq)
                {
                    __m128 _s = (__m128)__lsx_vld(psq, 0);
                    _v0 = __lsx_vfmul_s(_v0, _s);
                    _v1 = __lsx_vfmul_s(_v1, _s);
                    _v2 = __lsx_vfmul_s(_v2, _s);
                    _v3 = __lsx_vfmul_s(_v3, _s);
                    _v4 = __lsx_vfmul_s(_v4, _s);
                    _v5 = __lsx_vfmul_s(_v5, _s);
                    _v6 = __lsx_vfmul_s(_v6, _s);
                    _v7 = __lsx_vfmul_s(_v7, _s);
                }
                *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_v0, _scale0), __lsx_vfmul_s(_v1, _scale1));
                *((int64_t*)(pp + 8)) = float2int8(__lsx_vfmul_s(_v2, _scale2), __lsx_vfmul_s(_v3, _scale3));
                *((int64_t*)(pp + 16)) = float2int8(__lsx_vfmul_s(_v4, _scale4), __lsx_vfmul_s(_v5, _scale5));
                *((int64_t*)(pp + 24)) = float2int8(__lsx_vfmul_s(_v6, _scale6), __lsx_vfmul_s(_v7, _scale7));
                pp += 32;
                p0q += 4;
                p1q += 4;
                p2q += 4;
                p3q += 4;
                p4q += 4;
                p5q += 4;
                p6q += 4;
                p7q += 4;
                if (psq)
                    psq += 4;
            }
            for (; kk < max_kk; kk++)
            {
                const float s = psq ? *psq++ : 1.f;
                pp[0] = float2int8(*p0q++ * s * scale0);
                pp[1] = float2int8(*p1q++ * s * scale1);
                pp[2] = float2int8(*p2q++ * s * scale2);
                pp[3] = float2int8(*p3q++ * s * scale3);
                pp[4] = float2int8(*p4q++ * s * scale4);
                pp[5] = float2int8(*p5q++ * s * scale5);
                pp[6] = float2int8(*p6q++ * s * scale6);
                pp[7] = float2int8(*p7q++ * s * scale7);
                pp += 8;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = A_data + (i + ii) * A_hstep;
        const float* p1 = p0 + A_hstep;
        const float* p2 = p1 + A_hstep;
        const float* p3 = p2 + A_hstep;
        signed char* pp = outptr + ii * out_hstep;
        float* pd = descales + ii * descales_hstep;
        const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            const float* p0g = p0 + k0;
            const float* p1g = p1 + k0;
            const float* p2g = p2 + k0;
            const float* p3g = p3 + k0;
            const float* sg = input_scale_ptr ? input_scale_ptr + k0 : 0;
            __m128 _absmax0 = (__m128)__lsx_vldi(0);
            __m128 _absmax1 = (__m128)__lsx_vldi(0);
            __m128 _absmax2 = (__m128)__lsx_vldi(0);
            __m128 _absmax3 = (__m128)__lsx_vldi(0);
            const float* p0a = p0g;
            const float* p1a = p1g;
            const float* p2a = p2g;
            const float* p3a = p3g;
            const float* psa = sg;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _v0 = (__m128)__lsx_vld(p0a, 0);
                __m128 _v1 = (__m128)__lsx_vld(p1a, 0);
                __m128 _v2 = (__m128)__lsx_vld(p2a, 0);
                __m128 _v3 = (__m128)__lsx_vld(p3a, 0);
                if (psa)
                {
                    __m128 _s = (__m128)__lsx_vld(psa, 0);
                    _v0 = __lsx_vfmul_s(_v0, _s);
                    _v1 = __lsx_vfmul_s(_v1, _s);
                    _v2 = __lsx_vfmul_s(_v2, _s);
                    _v3 = __lsx_vfmul_s(_v3, _s);
                }
                _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_v0, _abs_mask));
                _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_v1, _abs_mask));
                _absmax2 = __lsx_vfmax_s(_absmax2, (__m128)__lsx_vand_v((__m128i)_v2, _abs_mask));
                _absmax3 = __lsx_vfmax_s(_absmax3, (__m128)__lsx_vand_v((__m128i)_v3, _abs_mask));
                p0a += 4;
                p1a += 4;
                p2a += 4;
                p3a += 4;
                if (psa)
                    psa += 4;
            }
            float absmax0 = __lsx_reduce_fmax_s(_absmax0);
            float absmax1 = __lsx_reduce_fmax_s(_absmax1);
            float absmax2 = __lsx_reduce_fmax_s(_absmax2);
            float absmax3 = __lsx_reduce_fmax_s(_absmax3);
            for (; kk < max_kk; kk++)
            {
                const float s = psa ? *psa++ : 1.f;
                absmax0 = std::max(absmax0, fabsf(*p0a++ * s));
                absmax1 = std::max(absmax1, fabsf(*p1a++ * s));
                absmax2 = std::max(absmax2, fabsf(*p2a++ * s));
                absmax3 = std::max(absmax3, fabsf(*p3a++ * s));
            }
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd[2] = absmax2 / 127.f;
            pd[3] = absmax3 / 127.f;
            pd += 4;
            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
            const float scale2 = absmax2 == 0.f ? 0.f : 127.f / absmax2;
            const float scale3 = absmax3 == 0.f ? 0.f : 127.f / absmax3;
            __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
            __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
            __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
            __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
            const float* p0q = p0g;
            const float* p1q = p1g;
            const float* p2q = p2g;
            const float* p3q = p3g;
            const float* psq = sg;
            kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _v0 = (__m128)__lsx_vld(p0q, 0);
                __m128 _v1 = (__m128)__lsx_vld(p1q, 0);
                __m128 _v2 = (__m128)__lsx_vld(p2q, 0);
                __m128 _v3 = (__m128)__lsx_vld(p3q, 0);
                if (psq)
                {
                    __m128 _s = (__m128)__lsx_vld(psq, 0);
                    _v0 = __lsx_vfmul_s(_v0, _s);
                    _v1 = __lsx_vfmul_s(_v1, _s);
                    _v2 = __lsx_vfmul_s(_v2, _s);
                    _v3 = __lsx_vfmul_s(_v3, _s);
                }
                *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_v0, _scale0), __lsx_vfmul_s(_v1, _scale1));
                *((int64_t*)(pp + 8)) = float2int8(__lsx_vfmul_s(_v2, _scale2), __lsx_vfmul_s(_v3, _scale3));
                pp += 16;
                p0q += 4;
                p1q += 4;
                p2q += 4;
                p3q += 4;
                if (psq)
                    psq += 4;
            }
            if (kk + 1 < max_kk)
            {
                const float s0 = psq ? psq[0] : 1.f;
                const float s1 = psq ? psq[1] : 1.f;
                pp[0] = float2int8(p0q[0] * s0 * scale0);
                pp[1] = float2int8(p0q[1] * s1 * scale0);
                pp[2] = float2int8(p1q[0] * s0 * scale1);
                pp[3] = float2int8(p1q[1] * s1 * scale1);
                pp[4] = float2int8(p2q[0] * s0 * scale2);
                pp[5] = float2int8(p2q[1] * s1 * scale2);
                pp[6] = float2int8(p3q[0] * s0 * scale3);
                pp[7] = float2int8(p3q[1] * s1 * scale3);
                pp += 8;
                p0q += 2;
                p1q += 2;
                p2q += 2;
                p3q += 2;
                if (psq)
                    psq += 2;
                kk += 2;
            }
            if (kk < max_kk)
            {
                const float s = psq ? *psq : 1.f;
                pp[0] = float2int8(*p0q * s * scale0);
                pp[1] = float2int8(*p1q * s * scale1);
                pp[2] = float2int8(*p2q * s * scale2);
                pp[3] = float2int8(*p3q * s * scale3);
                pp += 4;
            }
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = A_data + (i + ii) * A_hstep;
        const float* p1 = p0 + A_hstep;
        signed char* pp = outptr + ii * out_hstep;
        float* pd = descales + ii * descales_hstep;
        const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            const float* p0g = p0 + k0;
            const float* p1g = p1 + k0;
            const float* sg = input_scale_ptr ? input_scale_ptr + k0 : 0;
            __m128 _absmax0 = (__m128)__lsx_vldi(0);
            __m128 _absmax1 = (__m128)__lsx_vldi(0);
            const float* p0a = p0g;
            const float* p1a = p1g;
            const float* psa = sg;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _v0 = (__m128)__lsx_vld(p0a, 0);
                __m128 _v1 = (__m128)__lsx_vld(p1a, 0);
                if (psa)
                {
                    __m128 _s = (__m128)__lsx_vld(psa, 0);
                    _v0 = __lsx_vfmul_s(_v0, _s);
                    _v1 = __lsx_vfmul_s(_v1, _s);
                }
                _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_v0, _abs_mask));
                _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_v1, _abs_mask));
                p0a += 4;
                p1a += 4;
                if (psa)
                    psa += 4;
            }
            float absmax0 = __lsx_reduce_fmax_s(_absmax0);
            float absmax1 = __lsx_reduce_fmax_s(_absmax1);
            for (; kk < max_kk; kk++)
            {
                const float s = psa ? *psa++ : 1.f;
                absmax0 = std::max(absmax0, fabsf(*p0a++ * s));
                absmax1 = std::max(absmax1, fabsf(*p1a++ * s));
            }
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;
            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
            __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
            __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
            const float* p0q = p0g;
            const float* p1q = p1g;
            const float* psq = sg;
            kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _v0 = (__m128)__lsx_vld(p0q, 0);
                __m128 _v1 = (__m128)__lsx_vld(p1q, 0);
                if (psq)
                {
                    __m128 _s = (__m128)__lsx_vld(psq, 0);
                    _v0 = __lsx_vfmul_s(_v0, _s);
                    _v1 = __lsx_vfmul_s(_v1, _s);
                }
                *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_v0, _scale0), __lsx_vfmul_s(_v1, _scale1));
                pp += 8;
                p0q += 4;
                p1q += 4;
                if (psq)
                    psq += 4;
            }
            if (kk + 1 < max_kk)
            {
                const float s0 = psq ? psq[0] : 1.f;
                const float s1 = psq ? psq[1] : 1.f;
                pp[0] = float2int8(p0q[0] * s0 * scale0);
                pp[1] = float2int8(p0q[1] * s1 * scale0);
                pp[2] = float2int8(p1q[0] * s0 * scale1);
                pp[3] = float2int8(p1q[1] * s1 * scale1);
                pp += 4;
                p0q += 2;
                p1q += 2;
                if (psq)
                    psq += 2;
                kk += 2;
            }
            if (kk < max_kk)
            {
                const float s = psq ? *psq : 1.f;
                pp[0] = float2int8(*p0q * s * scale0);
                pp[1] = float2int8(*p1q * s * scale1);
                pp += 2;
            }
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = A_data + (i + ii) * A_hstep;
        const float* p1 = p0 + A_hstep;
        signed char* pp = outptr + ii * out_hstep;
        float* pd = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            const float* p0g = p0 + k0;
            const float* p1g = p1 + k0;
            const float* sg = input_scale_ptr ? input_scale_ptr + k0 : 0;
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            const float* p0a = p0g;
            const float* p1a = p1g;
            const float* psa = sg;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const float s = psa ? *psa++ : 1.f;
                absmax0 = std::max(absmax0, fabsf(*p0a++ * s));
                absmax1 = std::max(absmax1, fabsf(*p1a++ * s));
            }
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;
            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
            const float* p0q = p0g;
            const float* p1q = p1g;
            const float* psq = sg;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                const float s0 = psq ? psq[0] : 1.f;
                const float s1 = psq ? psq[1] : 1.f;
                const float s2 = psq ? psq[2] : 1.f;
                const float s3 = psq ? psq[3] : 1.f;
                pp[0] = float2int8(p0q[0] * s0 * scale0);
                pp[1] = float2int8(p0q[1] * s1 * scale0);
                pp[2] = float2int8(p0q[2] * s2 * scale0);
                pp[3] = float2int8(p0q[3] * s3 * scale0);
                pp[4] = float2int8(p1q[0] * s0 * scale1);
                pp[5] = float2int8(p1q[1] * s1 * scale1);
                pp[6] = float2int8(p1q[2] * s2 * scale1);
                pp[7] = float2int8(p1q[3] * s3 * scale1);
                pp += 8;
                p0q += 4;
                p1q += 4;
                if (psq)
                    psq += 4;
            }
            if (kk + 1 < max_kk)
            {
                const float s0 = psq ? psq[0] : 1.f;
                const float s1 = psq ? psq[1] : 1.f;
                pp[0] = float2int8(p0q[0] * s0 * scale0);
                pp[1] = float2int8(p0q[1] * s1 * scale0);
                pp[2] = float2int8(p1q[0] * s0 * scale1);
                pp[3] = float2int8(p1q[1] * s1 * scale1);
                pp += 4;
                p0q += 2;
                p1q += 2;
                if (psq)
                    psq += 2;
                kk += 2;
            }
            if (kk < max_kk)
            {
                const float s = psq ? *psq : 1.f;
                pp[0] = float2int8(*p0q * s * scale0);
                pp[1] = float2int8(*p1q * s * scale1);
                pp += 2;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        const float* p0 = A_data + (i + ii) * A_hstep;
        signed char* pp = outptr + ii * out_hstep;
        float* pd = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            const float* p0g = p0 + k0;
            const float* sg = input_scale_ptr ? input_scale_ptr + k0 : 0;
            float absmax = 0.f;
            const float* p0a = p0g;
            const float* psa = sg;
            int kk = 0;
#if __loongarch_sx
            const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
#if __loongarch_asx
            const __m256i _abs_mask256 = __lasx_xvreplgr2vr_w(0x7fffffff);
            __m256 _absmax256 = (__m256)__lasx_xvldi(0);
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _v = (__m256)__lasx_xvld(p0a, 0);
                if (psa)
                    _v = __lasx_xvfmul_s(_v, (__m256)__lasx_xvld(psa, 0));
                _v = (__m256)__lasx_xvand_v((__m256i)_v, _abs_mask256);
                _absmax256 = __lasx_xvfmax_s(_absmax256, _v);
                p0a += 8;
                if (psa)
                    psa += 8;
            }
            absmax = __lasx_reduce_fmax_s(_absmax256);
#endif
            __m128 _absmax128 = __lsx_vreplfr2vr_s(absmax);
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _v = (__m128)__lsx_vld(p0a, 0);
                if (psa)
                    _v = __lsx_vfmul_s(_v, (__m128)__lsx_vld(psa, 0));
                _v = (__m128)__lsx_vand_v((__m128i)_v, _abs_mask);
                _absmax128 = __lsx_vfmax_s(_absmax128, _v);
                p0a += 4;
                if (psa)
                    psa += 4;
            }
            absmax = __lsx_reduce_fmax_s(_absmax128);
#endif
            for (; kk < max_kk; kk++)
            {
                float v = *p0a++;
                if (psa)
                    v *= *psa++;
                absmax = std::max(absmax, fabsf(v));
            }

            if (absmax == 0.f)
            {
                *pd++ = 0.f;
                for (int kk = 0; kk < max_kk; kk++)
                    *pp++ = 0;
                continue;
            }

            const float scale = 127.f / absmax;
            *pd++ = absmax / 127.f;
            const float* p0q = p0g;
            const float* psq = sg;
            kk = 0;
#if __loongarch_sx
#if __loongarch_asx
            __m256 _scale256 = (__m256)__lasx_xvreplfr2vr_s(scale);
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _v = (__m256)__lasx_xvld(p0q, 0);
                if (psq)
                    _v = __lasx_xvfmul_s(_v, (__m256)__lasx_xvld(psq, 0));
                _v = __lasx_xvfmul_s(_v, _scale256);
                __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(_v)), pp, 0, 0);
                pp += 8;
                p0q += 8;
                if (psq)
                    psq += 8;
            }
#endif
            __m128 _scale128 = __lsx_vreplfr2vr_s(scale);
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _v = (__m128)__lsx_vld(p0q, 0);
                if (psq)
                    _v = __lsx_vfmul_s(_v, (__m128)__lsx_vld(psq, 0));
                _v = __lsx_vfmul_s(_v, _scale128);
                __lsx_vstelm_w(float2int8(_v), pp, 0, 0);
                pp += 4;
                p0q += 4;
                if (psq)
                    psq += 4;
            }
#endif
            for (; kk < max_kk; kk++)
            {
                float v = *p0q++;
                if (psq)
                    v *= *psq++;
                *pp++ = float2int8(v * scale);
            }
        }
    }
}

static void transpose_quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const float* input_scale_ptr)
{
    signed char* outptr = AT_tile;
    const int out_hstep = AT_tile.w;
    float* descales = AT_descales_tile;
    const int descales_hstep = AT_descales_tile.w;
    const int K = max_kk;
    const int block_count = (K + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
    const float* A_data = (const float*)A + (size_t)k * A_hstep;
    input_scale_ptr = input_scale_ptr ? input_scale_ptr + k : 0;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* ptrA = A_data + i + ii;
        signed char* pp = outptr + ii * out_hstep;
        float* pd = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            const float* p0g = ptrA + (size_t)k0 * A_hstep;
            const float* sg = input_scale_ptr ? input_scale_ptr + k0 : 0;
            const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
            __m128 _absmax0 = (__m128)__lsx_vldi(0);
            __m128 _absmax1 = (__m128)__lsx_vldi(0);
            const float* p0a = p0g;
            const float* psa = sg;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _v0 = (__m128)__lsx_vld(p0a, 0);
                __m128 _v1 = (__m128)__lsx_vld(p0a + 4, 0);
                if (psa)
                {
                    __m128 _s = __lsx_vreplfr2vr_s(*psa++);
                    _v0 = __lsx_vfmul_s(_v0, _s);
                    _v1 = __lsx_vfmul_s(_v1, _s);
                }
                _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_v0, _abs_mask));
                _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_v1, _abs_mask));
                p0a += A_hstep;
            }

            float absmax[8];
            __lsx_vst(_absmax0, absmax, 0);
            __lsx_vst(_absmax1, absmax + 4, 0);
            const float absmax0 = absmax[0];
            const float absmax1 = absmax[1];
            const float absmax2 = absmax[2];
            const float absmax3 = absmax[3];
            const float absmax4 = absmax[4];
            const float absmax5 = absmax[5];
            const float absmax6 = absmax[6];
            const float absmax7 = absmax[7];

            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd[2] = absmax2 / 127.f;
            pd[3] = absmax3 / 127.f;
            pd[4] = absmax4 / 127.f;
            pd[5] = absmax5 / 127.f;
            pd[6] = absmax6 / 127.f;
            pd[7] = absmax7 / 127.f;
            pd += 8;

            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
            const float scale2 = absmax2 == 0.f ? 0.f : 127.f / absmax2;
            const float scale3 = absmax3 == 0.f ? 0.f : 127.f / absmax3;
            const float scale4 = absmax4 == 0.f ? 0.f : 127.f / absmax4;
            const float scale5 = absmax5 == 0.f ? 0.f : 127.f / absmax5;
            const float scale6 = absmax6 == 0.f ? 0.f : 127.f / absmax6;
            const float scale7 = absmax7 == 0.f ? 0.f : 127.f / absmax7;
            __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
            __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
            __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
            __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
            __m128 _scale4 = __lsx_vreplfr2vr_s(scale4);
            __m128 _scale5 = __lsx_vreplfr2vr_s(scale5);
            __m128 _scale6 = __lsx_vreplfr2vr_s(scale6);
            __m128 _scale7 = __lsx_vreplfr2vr_s(scale7);
            const float scales0[4] = {scale0, scale1, scale2, scale3};
            const float scales1[4] = {scale4, scale5, scale6, scale7};
            __m128 _scales0 = (__m128)__lsx_vld(scales0, 0);
            __m128 _scales1 = (__m128)__lsx_vld(scales1, 0);

            const float* p0q = p0g;
            const float* psq = sg;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                const float* p0 = p0q;
                const float* p1 = p0 + A_hstep;
                const float* p2 = p1 + A_hstep;
                const float* p3 = p2 + A_hstep;
                __m128 _p0 = (__m128)__lsx_vld(p0, 0);
                __m128 _p1 = (__m128)__lsx_vld(p1, 0);
                __m128 _p2 = (__m128)__lsx_vld(p2, 0);
                __m128 _p3 = (__m128)__lsx_vld(p3, 0);
                __m128 _p4 = (__m128)__lsx_vld(p0 + 4, 0);
                __m128 _p5 = (__m128)__lsx_vld(p1 + 4, 0);
                __m128 _p6 = (__m128)__lsx_vld(p2 + 4, 0);
                __m128 _p7 = (__m128)__lsx_vld(p3 + 4, 0);
                if (psq)
                {
                    __m128 _s0 = __lsx_vreplfr2vr_s(psq[0]);
                    __m128 _s1 = __lsx_vreplfr2vr_s(psq[1]);
                    __m128 _s2 = __lsx_vreplfr2vr_s(psq[2]);
                    __m128 _s3 = __lsx_vreplfr2vr_s(psq[3]);
                    _p0 = __lsx_vfmul_s(_p0, _s0);
                    _p1 = __lsx_vfmul_s(_p1, _s1);
                    _p2 = __lsx_vfmul_s(_p2, _s2);
                    _p3 = __lsx_vfmul_s(_p3, _s3);
                    _p4 = __lsx_vfmul_s(_p4, _s0);
                    _p5 = __lsx_vfmul_s(_p5, _s1);
                    _p6 = __lsx_vfmul_s(_p6, _s2);
                    _p7 = __lsx_vfmul_s(_p7, _s3);
                }
                transpose4x4_ps(_p0, _p1, _p2, _p3);
                transpose4x4_ps(_p4, _p5, _p6, _p7);
                *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_p0, _scale0), __lsx_vfmul_s(_p1, _scale1));
                *((int64_t*)(pp + 8)) = float2int8(__lsx_vfmul_s(_p2, _scale2), __lsx_vfmul_s(_p3, _scale3));
                *((int64_t*)(pp + 16)) = float2int8(__lsx_vfmul_s(_p4, _scale4), __lsx_vfmul_s(_p5, _scale5));
                *((int64_t*)(pp + 24)) = float2int8(__lsx_vfmul_s(_p6, _scale6), __lsx_vfmul_s(_p7, _scale7));
                pp += 32;
                p0q += (size_t)4 * A_hstep;
                if (psq)
                    psq += 4;
            }
            for (; kk < max_kk; kk++)
            {
                __m128 _p0 = (__m128)__lsx_vld(p0q, 0);
                __m128 _p1 = (__m128)__lsx_vld(p0q + 4, 0);
                if (psq)
                {
                    __m128 _s = __lsx_vreplfr2vr_s(*psq++);
                    _p0 = __lsx_vfmul_s(_p0, _s);
                    _p1 = __lsx_vfmul_s(_p1, _s);
                }
                _p0 = __lsx_vfmul_s(_p0, _scales0);
                _p1 = __lsx_vfmul_s(_p1, _scales1);
                *((int64_t*)pp) = float2int8(_p0, _p1);
                pp += 8;
                p0q += A_hstep;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* ptrA = A_data + i + ii;
        signed char* pp = outptr + ii * out_hstep;
        float* pd = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            const float* p0g = ptrA + (size_t)k0 * A_hstep;
            const float* sg = input_scale_ptr ? input_scale_ptr + k0 : 0;

            const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
            __m128 _absmax = (__m128)__lsx_vldi(0);
            const float* p0a = p0g;
            const float* psa = sg;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _v = (__m128)__lsx_vld(p0a, 0);
                if (psa)
                    _v = __lsx_vfmul_s(_v, __lsx_vreplfr2vr_s(*psa++));
                _v = (__m128)__lsx_vand_v((__m128i)_v, _abs_mask);
                _absmax = __lsx_vfmax_s(_absmax, _v);
                p0a += A_hstep;
            }

            float absmax[4];
            __lsx_vst(_absmax, absmax, 0);
            pd[0] = absmax[0] / 127.f;
            pd[1] = absmax[1] / 127.f;
            pd[2] = absmax[2] / 127.f;
            pd[3] = absmax[3] / 127.f;
            pd += 4;

            const float scales[4] = {
                absmax[0] == 0.f ? 0.f : 127.f / absmax[0],
                absmax[1] == 0.f ? 0.f : 127.f / absmax[1],
                absmax[2] == 0.f ? 0.f : 127.f / absmax[2],
                absmax[3] == 0.f ? 0.f : 127.f / absmax[3]
            };
            __m128 _scale = (__m128)__lsx_vld(scales, 0);

            const float* p0q = p0g;
            const float* psq = sg;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                const float* p0 = p0q;
                const float* p1 = p0 + A_hstep;
                const float* p2 = p1 + A_hstep;
                const float* p3 = p2 + A_hstep;
                __m128 _v0 = (__m128)__lsx_vld(p0, 0);
                __m128 _v1 = (__m128)__lsx_vld(p1, 0);
                __m128 _v2 = (__m128)__lsx_vld(p2, 0);
                __m128 _v3 = (__m128)__lsx_vld(p3, 0);
                if (psq)
                {
                    _v0 = __lsx_vfmul_s(_v0, __lsx_vreplfr2vr_s(psq[0]));
                    _v1 = __lsx_vfmul_s(_v1, __lsx_vreplfr2vr_s(psq[1]));
                    _v2 = __lsx_vfmul_s(_v2, __lsx_vreplfr2vr_s(psq[2]));
                    _v3 = __lsx_vfmul_s(_v3, __lsx_vreplfr2vr_s(psq[3]));
                }
                transpose4x4_ps(_v0, _v1, _v2, _v3);
                *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_v0, __lsx_vreplfr2vr_s(scales[0])), __lsx_vfmul_s(_v1, __lsx_vreplfr2vr_s(scales[1])));
                *((int64_t*)(pp + 8)) = float2int8(__lsx_vfmul_s(_v2, __lsx_vreplfr2vr_s(scales[2])), __lsx_vfmul_s(_v3, __lsx_vreplfr2vr_s(scales[3])));
                pp += 16;
                p0q += (size_t)4 * A_hstep;
                if (psq)
                    psq += 4;
            }
            if (kk + 1 < max_kk)
            {
                __m128 _v0 = (__m128)__lsx_vld(p0q, 0);
                __m128 _v1 = (__m128)__lsx_vld(p0q + A_hstep, 0);
                if (psq)
                {
                    _v0 = __lsx_vfmul_s(_v0, __lsx_vreplfr2vr_s(psq[0]));
                    _v1 = __lsx_vfmul_s(_v1, __lsx_vreplfr2vr_s(psq[1]));
                }
                const int q0 = __lsx_vpickve2gr_w(float2int8(__lsx_vfmul_s(_v0, _scale)), 0);
                const int q1 = __lsx_vpickve2gr_w(float2int8(__lsx_vfmul_s(_v1, _scale)), 0);
                pp[0] = (signed char)q0;
                pp[1] = (signed char)q1;
                pp[2] = (signed char)(q0 >> 8);
                pp[3] = (signed char)(q1 >> 8);
                pp[4] = (signed char)(q0 >> 16);
                pp[5] = (signed char)(q1 >> 16);
                pp[6] = (signed char)(q0 >> 24);
                pp[7] = (signed char)(q1 >> 24);
                pp += 8;
                p0q += (size_t)2 * A_hstep;
                if (psq)
                    psq += 2;
                kk += 2;
            }
            if (kk < max_kk)
            {
                __m128 _v = (__m128)__lsx_vld(p0q, 0);
                if (psq)
                    _v = __lsx_vfmul_s(_v, __lsx_vreplfr2vr_s(*psq));
                const int q = __lsx_vpickve2gr_w(float2int8(__lsx_vfmul_s(_v, _scale)), 0);
                pp[0] = (signed char)q;
                pp[1] = (signed char)(q >> 8);
                pp[2] = (signed char)(q >> 16);
                pp[3] = (signed char)(q >> 24);
                pp += 4;
            }
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* ptrA = A_data + i + ii;
        signed char* pp = outptr + ii * out_hstep;
        float* pd = descales + ii * descales_hstep;
        const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            const float* p0g = ptrA + (size_t)k0 * A_hstep;
            const float* sg = input_scale_ptr ? input_scale_ptr + k0 : 0;
            __m128 _absmax = (__m128)__lsx_vldi(0);
            const float* p0a = p0g;
            const float* psa = sg;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _v = (__m128)__lsx_vldrepl_d(p0a, 0);
                if (psa)
                    _v = __lsx_vfmul_s(_v, __lsx_vreplfr2vr_s(*psa++));
                _absmax = __lsx_vfmax_s(_absmax, (__m128)__lsx_vand_v((__m128i)_v, _abs_mask));
                p0a += A_hstep;
            }
            float absmax[2];
            __lsx_vstelm_d((__m128i)_absmax, absmax, 0, 0);
            pd[0] = absmax[0] / 127.f;
            pd[1] = absmax[1] / 127.f;
            pd += 2;
            const float scale0 = absmax[0] == 0.f ? 0.f : 127.f / absmax[0];
            const float scale1 = absmax[1] == 0.f ? 0.f : 127.f / absmax[1];
            const float* p0q = p0g;
            const float* psq = sg;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                const float* p0 = p0q;
                const float* p1 = p0 + A_hstep;
                const float* p2 = p1 + A_hstep;
                const float* p3 = p2 + A_hstep;
                const float s0 = psq ? psq[0] : 1.f;
                const float s1 = psq ? psq[1] : 1.f;
                const float s2 = psq ? psq[2] : 1.f;
                const float s3 = psq ? psq[3] : 1.f;
                pp[0] = float2int8(p0[0] * s0 * scale0);
                pp[1] = float2int8(p1[0] * s1 * scale0);
                pp[2] = float2int8(p2[0] * s2 * scale0);
                pp[3] = float2int8(p3[0] * s3 * scale0);
                pp[4] = float2int8(p0[1] * s0 * scale1);
                pp[5] = float2int8(p1[1] * s1 * scale1);
                pp[6] = float2int8(p2[1] * s2 * scale1);
                pp[7] = float2int8(p3[1] * s3 * scale1);
                pp += 8;
                p0q += (size_t)4 * A_hstep;
                if (psq)
                    psq += 4;
            }
            if (kk + 1 < max_kk)
            {
                const float* p0 = p0q;
                const float* p1 = p0 + A_hstep;
                const float s0 = psq ? psq[0] : 1.f;
                const float s1 = psq ? psq[1] : 1.f;
                pp[0] = float2int8(p0[0] * s0 * scale0);
                pp[1] = float2int8(p1[0] * s1 * scale0);
                pp[2] = float2int8(p0[1] * s0 * scale1);
                pp[3] = float2int8(p1[1] * s1 * scale1);
                pp += 4;
                p0q += (size_t)2 * A_hstep;
                if (psq)
                    psq += 2;
                kk += 2;
            }
            if (kk < max_kk)
            {
                const float s = psq ? *psq : 1.f;
                pp[0] = float2int8(p0q[0] * s * scale0);
                pp[1] = float2int8(p0q[1] * s * scale1);
                pp += 2;
            }
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* ptrA = A_data + i + ii;
        signed char* pp = outptr + ii * out_hstep;
        float* pd = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            const float* p0g = ptrA + (size_t)k0 * A_hstep;
            const float* sg = input_scale_ptr ? input_scale_ptr + k0 : 0;
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            const float* p0a = p0g;
            const float* psa = sg;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const float s = psa ? *psa++ : 1.f;
                absmax0 = std::max(absmax0, fabsf(p0a[0] * s));
                absmax1 = std::max(absmax1, fabsf(p0a[1] * s));
                p0a += A_hstep;
            }
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;
            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
            const float* p0q = p0g;
            const float* psq = sg;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                const float* p0 = p0q;
                const float* p1 = p0 + A_hstep;
                const float* p2 = p1 + A_hstep;
                const float* p3 = p2 + A_hstep;
                const float s0 = psq ? psq[0] : 1.f;
                const float s1 = psq ? psq[1] : 1.f;
                const float s2 = psq ? psq[2] : 1.f;
                const float s3 = psq ? psq[3] : 1.f;
                pp[0] = float2int8(p0[0] * s0 * scale0);
                pp[1] = float2int8(p1[0] * s1 * scale0);
                pp[2] = float2int8(p2[0] * s2 * scale0);
                pp[3] = float2int8(p3[0] * s3 * scale0);
                pp[4] = float2int8(p0[1] * s0 * scale1);
                pp[5] = float2int8(p1[1] * s1 * scale1);
                pp[6] = float2int8(p2[1] * s2 * scale1);
                pp[7] = float2int8(p3[1] * s3 * scale1);
                pp += 8;
                p0q += (size_t)4 * A_hstep;
                if (psq)
                    psq += 4;
            }
            if (kk + 1 < max_kk)
            {
                const float* p0 = p0q;
                const float* p1 = p0 + A_hstep;
                const float s0 = psq ? psq[0] : 1.f;
                const float s1 = psq ? psq[1] : 1.f;
                pp[0] = float2int8(p0[0] * s0 * scale0);
                pp[1] = float2int8(p1[0] * s1 * scale0);
                pp[2] = float2int8(p0[1] * s0 * scale1);
                pp[3] = float2int8(p1[1] * s1 * scale1);
                pp += 4;
                p0q += (size_t)2 * A_hstep;
                if (psq)
                    psq += 2;
                kk += 2;
            }
            if (kk < max_kk)
            {
                const float s = psq ? *psq : 1.f;
                pp[0] = float2int8(p0q[0] * s * scale0);
                pp[1] = float2int8(p0q[1] * s * scale1);
                pp += 2;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        const float* p0 = A_data + i + ii;
        signed char* pp = outptr + ii * out_hstep;
        float* pd = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            const float* p0g = p0 + (size_t)k0 * A_hstep;
            const float* sg = input_scale_ptr ? input_scale_ptr + k0 : 0;

            float absmax = 0.f;
            const float* p0a = p0g;
            const float* psa = sg;
            for (int kk = 0; kk < max_kk; kk++)
            {
                float v = *p0a;
                if (psa)
                    v *= *psa++;
                absmax = std::max(absmax, fabsf(v));
                p0a += A_hstep;
            }

            if (absmax == 0.f)
            {
                *pd++ = 0.f;
                for (int kk = 0; kk < max_kk; kk++)
                    *pp++ = 0;
                continue;
            }

            const float scale = 127.f / absmax;
            *pd++ = absmax / 127.f;

            const float* p0q = p0g;
            const float* psq = sg;
            for (int kk = 0; kk < max_kk; kk++)
            {
                float v = *p0q;
                if (psq)
                    v *= *psq++;
                *pp++ = float2int8(v * scale);
                p0q += A_hstep;
            }
        }
    }
}

static void gemm_transB_packed_tile_wq_int8(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int full_K, int k0, int max_kk0, int block_size)
{
    const signed char* pAT = AT_tile;
    const int A_hstep = max_kk0;
    const float* pAT_descales = AT_descales_tile;
    const int A_descales_hstep = (max_kk0 + block_size - 1) / block_size;
    const signed char* pBT = BT_tile;
    const float* pBT_descales = BT_descales_tile;
    float* outptr = topT_tile;
    const int K = max_kk0;
    const int block_count = (full_K + block_size - 1) / block_size;
    const int block_start = k0 / block_size;
    const int tile_blocks = (max_kk0 + block_size - 1) / block_size;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;

        int jj = 0;
#if __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            pB += (size_t)8 * k0;
            pB_descales += (size_t)8 * block_start;
            __m256 _out0 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr, 0);
            __m256 _out1 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 8, 0);
            __m256 _out2 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 16, 0);
            __m256 _out3 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 24, 0);
            __m256 _out4 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 32, 0);
            __m256 _out5 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 40, 0);
            __m256 _out6 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 48, 0);
            __m256 _out7 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 56, 0);
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                __m256i _sum0 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum1 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum2 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum3 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum4 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum5 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum6 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum7 = __lasx_xvreplgr2vr_w(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m256i _pA = __lasx_xvld(pA, 0);
                    __m256i _pA1 = __lasx_xvshuf4i_w(_pA, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m256i _pA2 = __lasx_xvpermi_q(_pA, _pA, _LSX_SHUFFLE(0, 0, 0, 1));
                    __m256i _pA3 = __lasx_xvshuf4i_w(_pA2, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m256i _pB0 = __lasx_xvld(pB, 0);
                    __m256i _pB1 = __lasx_xvshuf4i_w(_pB0, _LSX_SHUFFLE(1, 0, 3, 2));

                    __m256i _s0 = __lasx_xvmulwev_h_b(_pA, _pB0);
                    __m256i _s1 = __lasx_xvmulwev_h_b(_pA1, _pB0);
                    _s0 = __lasx_xvmaddwod_h_b(_s0, _pA, _pB0);
                    _s1 = __lasx_xvmaddwod_h_b(_s1, _pA1, _pB0);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s0, _s0));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvhaddw_w_h(_s1, _s1));
                    _s0 = __lasx_xvmulwev_h_b(_pA, _pB1);
                    _s1 = __lasx_xvmulwev_h_b(_pA1, _pB1);
                    _s0 = __lasx_xvmaddwod_h_b(_s0, _pA, _pB1);
                    _s1 = __lasx_xvmaddwod_h_b(_s1, _pA1, _pB1);
                    _sum2 = __lasx_xvadd_w(_sum2, __lasx_xvhaddw_w_h(_s0, _s0));
                    _sum3 = __lasx_xvadd_w(_sum3, __lasx_xvhaddw_w_h(_s1, _s1));
                    _s0 = __lasx_xvmulwev_h_b(_pA2, _pB0);
                    _s1 = __lasx_xvmulwev_h_b(_pA3, _pB0);
                    _s0 = __lasx_xvmaddwod_h_b(_s0, _pA2, _pB0);
                    _s1 = __lasx_xvmaddwod_h_b(_s1, _pA3, _pB0);
                    _sum4 = __lasx_xvadd_w(_sum4, __lasx_xvhaddw_w_h(_s0, _s0));
                    _sum5 = __lasx_xvadd_w(_sum5, __lasx_xvhaddw_w_h(_s1, _s1));
                    _s0 = __lasx_xvmulwev_h_b(_pA2, _pB1);
                    _s1 = __lasx_xvmulwev_h_b(_pA3, _pB1);
                    _s0 = __lasx_xvmaddwod_h_b(_s0, _pA2, _pB1);
                    _s1 = __lasx_xvmaddwod_h_b(_s1, _pA3, _pB1);
                    _sum6 = __lasx_xvadd_w(_sum6, __lasx_xvhaddw_w_h(_s0, _s0));
                    _sum7 = __lasx_xvadd_w(_sum7, __lasx_xvhaddw_w_h(_s1, _s1));
                    pB += 32;
                    pA += 32;
                }
                if (kk + 1 < max_kk)
                {
                    __m128i _pAs = __lsx_vld(pA, 0);
                    __m128i _pA8 = __lsx_vbsrl_v(_pAs, 8);
                    __m256i _pA = __lasx_concat_128(__lsx_vilvl_b(_pA8, _pAs), __lsx_vilvl_b(_pA8, _pAs));
                    __m256i _pA1 = __lasx_xvshuf4i_h(_pA, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m256i _pA2 = __lasx_xvshuf4i_w(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m256i _pA3 = __lasx_xvshuf4i_h(_pA2, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m128i _pBs = __lsx_vld(pB, 0);
                    __m256i _pB0 = __lasx_concat_128(_pBs, _pBs);
                    __m256i _pB1 = __lasx_xvshuf4i_h(_pB0, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m256i _s0 = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA, _pB0), _pA, _pB0);
                    __m256i _s1 = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA1, _pB0), _pA1, _pB0);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(_s0));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_vext2xv_w_h(_s1));
                    _s0 = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA, _pB1), _pA, _pB1);
                    _s1 = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    _sum2 = __lasx_xvadd_w(_sum2, __lasx_vext2xv_w_h(_s0));
                    _sum3 = __lasx_xvadd_w(_sum3, __lasx_vext2xv_w_h(_s1));
                    _s0 = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA2, _pB0), _pA2, _pB0);
                    _s1 = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA3, _pB0), _pA3, _pB0);
                    _sum4 = __lasx_xvadd_w(_sum4, __lasx_vext2xv_w_h(_s0));
                    _sum5 = __lasx_xvadd_w(_sum5, __lasx_vext2xv_w_h(_s1));
                    _s0 = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA2, _pB1), _pA2, _pB1);
                    _s1 = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA3, _pB1), _pA3, _pB1);
                    _sum6 = __lasx_xvadd_w(_sum6, __lasx_vext2xv_w_h(_s0));
                    _sum7 = __lasx_xvadd_w(_sum7, __lasx_vext2xv_w_h(_s1));
                    pB += 16;
                    pA += 16;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    __m256i _pA = __lasx_xvldrepl_d(pA, 0);
                    _pA = __lasx_xvilvl_b(__lasx_xvslti_b(_pA, 0), _pA);
                    __m256i _pA1 = __lasx_xvshuf4i_h(_pA, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m256i _pA2 = __lasx_xvshuf4i_w(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m256i _pA3 = __lasx_xvshuf4i_h(_pA2, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m256i _pB0 = __lasx_xvldrepl_d(pB, 0);
                    _pB0 = __lasx_xvilvl_b(__lasx_xvslti_b(_pB0, 0), _pB0);
                    __m256i _pB1 = __lasx_xvshuf4i_h(_pB0, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m256i _s0 = __lasx_xvmul_h(_pA, _pB0);
                    __m256i _s1 = __lasx_xvmul_h(_pA1, _pB0);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(_s0));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_vext2xv_w_h(_s1));
                    _s0 = __lasx_xvmul_h(_pA, _pB1);
                    _s1 = __lasx_xvmul_h(_pA1, _pB1);
                    _sum2 = __lasx_xvadd_w(_sum2, __lasx_vext2xv_w_h(_s0));
                    _sum3 = __lasx_xvadd_w(_sum3, __lasx_vext2xv_w_h(_s1));
                    _s0 = __lasx_xvmul_h(_pA2, _pB0);
                    _s1 = __lasx_xvmul_h(_pA3, _pB0);
                    _sum4 = __lasx_xvadd_w(_sum4, __lasx_vext2xv_w_h(_s0));
                    _sum5 = __lasx_xvadd_w(_sum5, __lasx_vext2xv_w_h(_s1));
                    _s0 = __lasx_xvmul_h(_pA2, _pB1);
                    _s1 = __lasx_xvmul_h(_pA3, _pB1);
                    _sum6 = __lasx_xvadd_w(_sum6, __lasx_vext2xv_w_h(_s0));
                    _sum7 = __lasx_xvadd_w(_sum7, __lasx_vext2xv_w_h(_s1));
                    pB += 8;
                    pA += 8;
                }

                __m256 _bscale = (__m256)__lasx_xvld(pB_descales, 0);
                __m256 _bscale1 = (__m256)__lasx_xvshuf4i_w((__m256i)_bscale, _LSX_SHUFFLE(1, 0, 3, 2));
                __m256 _ascale = (__m256)__lasx_xvld(pA_descales, 0);
                __m256 _ascale1 = (__m256)__lasx_xvshuf4i_w((__m256i)_ascale, _LSX_SHUFFLE(0, 3, 2, 1));
                __m256 _ascale2 = (__m256)__lasx_xvpermi_q((__m256i)_ascale, (__m256i)_ascale, _LSX_SHUFFLE(0, 0, 0, 1));
                __m256 _ascale3 = (__m256)__lasx_xvshuf4i_w((__m256i)_ascale2, _LSX_SHUFFLE(0, 3, 2, 1));
                _out0 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum0), __lasx_xvfmul_s(_ascale, _bscale), _out0);
                _out1 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum1), __lasx_xvfmul_s(_ascale1, _bscale), _out1);
                _out2 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum2), __lasx_xvfmul_s(_ascale, _bscale1), _out2);
                _out3 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum3), __lasx_xvfmul_s(_ascale1, _bscale1), _out3);
                _out4 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum4), __lasx_xvfmul_s(_ascale2, _bscale), _out4);
                _out5 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum5), __lasx_xvfmul_s(_ascale3, _bscale), _out5);
                _out6 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum6), __lasx_xvfmul_s(_ascale2, _bscale1), _out6);
                _out7 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum7), __lasx_xvfmul_s(_ascale3, _bscale1), _out7);
                pA_descales += 8;
                pB_descales += 8;
            }
            __lasx_xvst(_out0, outptr, 0);
            __lasx_xvst(_out1, outptr + 8, 0);
            __lasx_xvst(_out2, outptr + 16, 0);
            __lasx_xvst(_out3, outptr + 24, 0);
            __lasx_xvst(_out4, outptr + 32, 0);
            __lasx_xvst(_out5, outptr + 40, 0);
            __lasx_xvst(_out6, outptr + 48, 0);
            __lasx_xvst(_out7, outptr + 56, 0);
            outptr += 64;
            pB += (size_t)8 * (full_K - k0 - max_kk0);
            pB_descales += (size_t)8 * (block_count - block_start - tile_blocks);
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
            pB += (size_t)4 * k0;
            pB_descales += (size_t)4 * block_start;
            __m128 _out00 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr, 0);
            __m128 _out01 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 4, 0);
            __m128 _out10 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 8, 0);
            __m128 _out11 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 12, 0);
            __m128 _out20 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 16, 0);
            __m128 _out21 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 20, 0);
            __m128 _out30 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 24, 0);
            __m128 _out31 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 28, 0);
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                __m128i _sum00 = __lsx_vreplgr2vr_w(0);
                __m128i _sum01 = __lsx_vreplgr2vr_w(0);
                __m128i _sum10 = __lsx_vreplgr2vr_w(0);
                __m128i _sum11 = __lsx_vreplgr2vr_w(0);
                __m128i _sum20 = __lsx_vreplgr2vr_w(0);
                __m128i _sum21 = __lsx_vreplgr2vr_w(0);
                __m128i _sum30 = __lsx_vreplgr2vr_w(0);
                __m128i _sum31 = __lsx_vreplgr2vr_w(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _pA0 = __lsx_vld(pA, 0);
                    __m128i _pA1 = __lsx_vld(pA + 16, 0);
                    __m128i _pA0r = __lsx_vshuf4i_w(_pA0, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m128i _pA1r = __lsx_vshuf4i_w(_pA1, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m128i _pB0 = __lsx_vld(pB, 0);
                    __m128i _pB0r = __lsx_vshuf4i_w(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));

                    __m128i _s0 = __lsx_vmulwev_h_b(_pA0, _pB0);
                    __m128i _s1 = __lsx_vmulwev_h_b(_pA1, _pB0);
                    _s0 = __lsx_vmaddwod_h_b(_s0, _pA0, _pB0);
                    _s1 = __lsx_vmaddwod_h_b(_s1, _pA1, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vhaddw_w_h(_s0, _s0));
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vhaddw_w_h(_s1, _s1));

                    _s0 = __lsx_vmulwev_h_b(_pA0, _pB0r);
                    _s1 = __lsx_vmulwev_h_b(_pA1, _pB0r);
                    _s0 = __lsx_vmaddwod_h_b(_s0, _pA0, _pB0r);
                    _s1 = __lsx_vmaddwod_h_b(_s1, _pA1, _pB0r);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vhaddw_w_h(_s0, _s0));
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vhaddw_w_h(_s1, _s1));

                    _s0 = __lsx_vmulwev_h_b(_pA0r, _pB0);
                    _s1 = __lsx_vmulwev_h_b(_pA1r, _pB0);
                    _s0 = __lsx_vmaddwod_h_b(_s0, _pA0r, _pB0);
                    _s1 = __lsx_vmaddwod_h_b(_s1, _pA1r, _pB0);
                    _sum20 = __lsx_vadd_w(_sum20, __lsx_vhaddw_w_h(_s0, _s0));
                    _sum21 = __lsx_vadd_w(_sum21, __lsx_vhaddw_w_h(_s1, _s1));

                    _s0 = __lsx_vmulwev_h_b(_pA0r, _pB0r);
                    _s1 = __lsx_vmulwev_h_b(_pA1r, _pB0r);
                    _s0 = __lsx_vmaddwod_h_b(_s0, _pA0r, _pB0r);
                    _s1 = __lsx_vmaddwod_h_b(_s1, _pA1r, _pB0r);
                    _sum30 = __lsx_vadd_w(_sum30, __lsx_vhaddw_w_h(_s0, _s0));
                    _sum31 = __lsx_vadd_w(_sum31, __lsx_vhaddw_w_h(_s1, _s1));
                    pB += 16;
                    pA += 32;
                }
                if (kk + 1 < max_kk)
                {
                    __m128i _pAs = __lsx_vld(pA, 0);
                    __m128i _pA8 = __lsx_vbsrl_v(_pAs, 8);
                    __m128i _pA = __lsx_vilvl_b(_pA8, _pAs);
                    __m128i _pA0 = __lsx_vreplvei_d(_pA, 0);
                    __m128i _pA1 = __lsx_vreplvei_d(_pA, 1);
                    __m128i _pA0r = __lsx_vshuf4i_h(_pA0, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m128i _pA1r = __lsx_vshuf4i_h(_pA1, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m128i _pBs = __lsx_vldrepl_d(pB, 0);
                    __m128i _pB0r = __lsx_vshuf4i_h(_pBs, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pBs), _pA0, _pBs);
                    __m128i _s1 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pBs), _pA1, _pBs);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB0r), _pA0, _pB0r);
                    _s1 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB0r), _pA1, _pB0r);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0r, _pBs), _pA0r, _pBs);
                    _s1 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1r, _pBs), _pA1r, _pBs);
                    _sum20 = __lsx_vadd_w(_sum20, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum21 = __lsx_vadd_w(_sum21, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0r, _pB0r), _pA0r, _pB0r);
                    _s1 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1r, _pB0r), _pA1r, _pB0r);
                    _sum30 = __lsx_vadd_w(_sum30, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum31 = __lsx_vadd_w(_sum31, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    pB += 8;
                    pA += 16;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    __m128i _pA0 = __lsx_vldrepl_w(pA, 0);
                    _pA0 = __lsx_vilvl_b(__lsx_vslti_b(_pA0, 0), _pA0);
                    __m128i _pA1 = __lsx_vldrepl_w(pA + 4, 0);
                    _pA1 = __lsx_vilvl_b(__lsx_vslti_b(_pA1, 0), _pA1);
                    __m128i _pA0r = __lsx_vshuf4i_h(_pA0, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m128i _pA1r = __lsx_vshuf4i_h(_pA1, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m128i _pB0 = __lsx_vldrepl_w(pB, 0);
                    _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                    __m128i _pB0r = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m128i _s0 = __lsx_vmul_h(_pA0, _pB0);
                    __m128i _s1 = __lsx_vmul_h(_pA1, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    _s0 = __lsx_vmul_h(_pA0, _pB0r);
                    _s1 = __lsx_vmul_h(_pA1, _pB0r);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    _s0 = __lsx_vmul_h(_pA0r, _pB0);
                    _s1 = __lsx_vmul_h(_pA1r, _pB0);
                    _sum20 = __lsx_vadd_w(_sum20, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum21 = __lsx_vadd_w(_sum21, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    _s0 = __lsx_vmul_h(_pA0r, _pB0r);
                    _s1 = __lsx_vmul_h(_pA1r, _pB0r);
                    _sum30 = __lsx_vadd_w(_sum30, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum31 = __lsx_vadd_w(_sum31, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    pB += 4;
                    pA += 8;
                }

                __m128 _bscale = (__m128)__lsx_vld(pB_descales, 0);
                __m128 _bscaler = (__m128)__lsx_vshuf4i_w((__m128i)_bscale, _LSX_SHUFFLE(0, 3, 2, 1));
                __m128 _ascale0 = (__m128)__lsx_vld(pA_descales, 0);
                __m128 _ascale1 = (__m128)__lsx_vld(pA_descales + 4, 0);
                __m128 _ascale0r = (__m128)__lsx_vshuf4i_w((__m128i)_ascale0, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _ascale1r = (__m128)__lsx_vshuf4i_w((__m128i)_ascale1, _LSX_SHUFFLE(1, 0, 3, 2));
                _out00 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum00), __lsx_vfmul_s(_ascale0, _bscale), _out00);
                _out01 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum01), __lsx_vfmul_s(_ascale1, _bscale), _out01);
                _out10 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum10), __lsx_vfmul_s(_ascale0, _bscaler), _out10);
                _out11 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum11), __lsx_vfmul_s(_ascale1, _bscaler), _out11);
                _out20 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum20), __lsx_vfmul_s(_ascale0r, _bscale), _out20);
                _out21 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum21), __lsx_vfmul_s(_ascale1r, _bscale), _out21);
                _out30 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum30), __lsx_vfmul_s(_ascale0r, _bscaler), _out30);
                _out31 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum31), __lsx_vfmul_s(_ascale1r, _bscaler), _out31);
                pA_descales += 8;
                pB_descales += 4;
            }
            __lsx_vst((__m128i)_out00, outptr, 0);
            __lsx_vst((__m128i)_out01, outptr + 4, 0);
            __lsx_vst((__m128i)_out10, outptr + 8, 0);
            __lsx_vst((__m128i)_out11, outptr + 12, 0);
            __lsx_vst((__m128i)_out20, outptr + 16, 0);
            __lsx_vst((__m128i)_out21, outptr + 20, 0);
            __lsx_vst((__m128i)_out30, outptr + 24, 0);
            __lsx_vst((__m128i)_out31, outptr + 28, 0);
            outptr += 32;
            pB += (size_t)4 * (full_K - k0 - max_kk0);
            pB_descales += (size_t)4 * (block_count - block_start - tile_blocks);
        }

        for (; jj + 1 < max_jj; jj += 2)
        {
            pB += (size_t)2 * k0;
            pB_descales += (size_t)2 * block_start;
            __m128 _out00 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr, 0);
            __m128 _out01 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 4, 0);
            __m128 _out10 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 8, 0);
            __m128 _out11 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 12, 0);
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                __m128i _sum00 = __lsx_vreplgr2vr_w(0);
                __m128i _sum01 = __lsx_vreplgr2vr_w(0);
                __m128i _sum10 = __lsx_vreplgr2vr_w(0);
                __m128i _sum11 = __lsx_vreplgr2vr_w(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _pA0 = __lsx_vld(pA, 0);
                    __m128i _pA1 = __lsx_vld(pA + 16, 0);
                    __m128i _pB = __lsx_vldrepl_d(pB, 0);
                    __m128i _pB0 = __lsx_vreplvei_w(_pB, 0);
                    __m128i _pB1 = __lsx_vreplvei_w(_pB, 1);
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB0), _pA0, _pB0);
                    __m128i _s1 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB0), _pA1, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vhaddw_w_h(_s0, _s0));
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vhaddw_w_h(_s1, _s1));
                    _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB1), _pA0, _pB1);
                    _s1 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vhaddw_w_h(_s0, _s0));
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vhaddw_w_h(_s1, _s1));
                    pA += 32;
                    pB += 8;
                }
                if (kk + 1 < max_kk)
                {
                    __m128i _pAs = __lsx_vld(pA, 0);
                    __m128i _pA8 = __lsx_vbsrl_v(_pAs, 8);
                    __m128i _pA = __lsx_vilvl_b(_pA8, _pAs);
                    __m128i _pA0 = __lsx_vreplvei_d(_pA, 0);
                    __m128i _pA1 = __lsx_vreplvei_d(_pA, 1);
                    __m128i _pB = __lsx_vldrepl_w(pB, 0);
                    __m128i _pB0 = __lsx_vreplvei_h(_pB, 0);
                    __m128i _pB1 = __lsx_vreplvei_h(_pB, 1);
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB0), _pA0, _pB0);
                    __m128i _s1 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB0), _pA1, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB1), _pA0, _pB1);
                    _s1 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    pA += 16;
                    pB += 4;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    __m128i _pA = __lsx_vldrepl_d(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _pA0 = __lsx_vreplvei_d(_pA, 0);
                    __m128i _pA1 = __lsx_vreplvei_d(_pA, 1);
                    __m128i _pB = __lsx_vldrepl_h(pB, 0);
                    _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);
                    __m128i _pB0 = __lsx_vreplvei_h(_pB, 0);
                    __m128i _pB1 = __lsx_vreplvei_h(_pB, 1);
                    __m128i _s0 = __lsx_vmul_h(_pA0, _pB0);
                    __m128i _s1 = __lsx_vmul_h(_pA1, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    _s0 = __lsx_vmul_h(_pA0, _pB1);
                    _s1 = __lsx_vmul_h(_pA1, _pB1);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    pA += 8;
                    pB += 2;
                }
                __m128 _ascale0 = (__m128)__lsx_vld(pA_descales, 0);
                __m128 _ascale1 = (__m128)__lsx_vld(pA_descales + 4, 0);
                __m128 _bscale = __lsx_vreplfr2vr_s(pB_descales[0]);
                _out00 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum00), __lsx_vfmul_s(_ascale0, _bscale), _out00);
                _out01 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum01), __lsx_vfmul_s(_ascale1, _bscale), _out01);
                _bscale = __lsx_vreplfr2vr_s(pB_descales[1]);
                _out10 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum10), __lsx_vfmul_s(_ascale0, _bscale), _out10);
                _out11 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum11), __lsx_vfmul_s(_ascale1, _bscale), _out11);
                pA_descales += 8;
                pB_descales += 2;
            }
            __lsx_vst((__m128i)_out00, outptr, 0);
            __lsx_vst((__m128i)_out01, outptr + 4, 0);
            __lsx_vst((__m128i)_out10, outptr + 8, 0);
            __lsx_vst((__m128i)_out11, outptr + 12, 0);
            outptr += 16;
            pB += (size_t)2 * (full_K - k0 - max_kk0);
            pB_descales += (size_t)2 * (block_count - block_start - tile_blocks);
        }
        for (; jj < max_jj; jj++)
        {
            pB += k0;
            pB_descales += block_start;
            __m128 _out0 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr, 0);
            __m128 _out1 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 4, 0);
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                __m128i _sum1 = __lsx_vreplgr2vr_w(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _pA0 = __lsx_vld(pA, 0);
                    __m128i _pA1 = __lsx_vld(pA + 16, 0);
                    __m128i _pB = __lsx_vldrepl_w(pB, 0);
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB), _pA0, _pB);
                    __m128i _s1 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB), _pA1, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s1, _s1));
                    pA += 32;
                    pB += 4;
                }
                if (kk + 1 < max_kk)
                {
                    __m128i _pAs = __lsx_vld(pA, 0);
                    __m128i _pA8 = __lsx_vbsrl_v(_pAs, 8);
                    __m128i _pA = __lsx_vilvl_b(_pA8, _pAs);
                    __m128i _pA0 = __lsx_vreplvei_d(_pA, 0);
                    __m128i _pA1 = __lsx_vreplvei_d(_pA, 1);
                    __m128i _pB = __lsx_vldrepl_h(pB, 0);
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB), _pA0, _pB);
                    __m128i _s1 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB), _pA1, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    pA += 16;
                    pB += 2;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    __m128i _pA = __lsx_vldrepl_d(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _pA0 = __lsx_vreplvei_d(_pA, 0);
                    __m128i _pA1 = __lsx_vreplvei_d(_pA, 1);
                    __m128i _pB = __lsx_vreplgr2vr_h((signed char)pB[0]);
                    __m128i _s0 = __lsx_vmul_h(_pA0, _pB);
                    __m128i _s1 = __lsx_vmul_h(_pA1, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    pA += 8;
                    pB++;
                }
                __m128 _bscale = __lsx_vreplfr2vr_s(*pB_descales++);
                _out0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum0), __lsx_vfmul_s((__m128)__lsx_vld(pA_descales, 0), _bscale), _out0);
                _out1 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum1), __lsx_vfmul_s((__m128)__lsx_vld(pA_descales + 4, 0), _bscale), _out1);
                pA_descales += 8;
            }
            __lsx_vst((__m128i)_out0, outptr, 0);
            __lsx_vst((__m128i)_out1, outptr + 4, 0);
            outptr += 8;
            pB += full_K - k0 - max_kk0;
            pB_descales += block_count - block_start - tile_blocks;
        }

        pAT += K * 8;
        pAT_descales += (K + block_size - 1) / block_size * 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;

        int jj = 0;
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            const signed char* pB0 = pB + 8 * k0;
            const signed char* pB1 = pB + 8 * full_K + 8 * k0;
            const float* pB_descales0 = pB_descales + 8 * block_start;
            const float* pB_descales1 = pB_descales + 8 * block_count + 8 * block_start;
            __m256 _out00 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr, 0);
            __m256 _out01 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 8, 0);
            __m256 _out10 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 16, 0);
            __m256 _out11 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 24, 0);
            __m256 _out20 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 32, 0);
            __m256 _out21 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 40, 0);
            __m256 _out30 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 48, 0);
            __m256 _out31 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 56, 0);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                __m256i _sum00 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum01 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum10 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum11 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum20 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum21 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum30 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum31 = __lasx_xvreplgr2vr_w(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _pA4 = __lsx_vld(pA, 0);
                    __m256i _pA = __lasx_concat_128(_pA4, _pA4);
                    __m256i _pA1 = __lasx_xvshuf4i_w(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m256i _pB0 = __lasx_xvld(pB0, 0);
                    __m256i _pB1 = __lasx_xvld(pB1, 0);
                    __m256i _pB0r = __lasx_xvshuf4i_w(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m256i _pB1r = __lasx_xvshuf4i_w(_pB1, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m256i _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA, _pB0), _pA, _pB0);
                    _sum00 = __lasx_xvadd_w(_sum00, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA, _pB0r), _pA, _pB0r);
                    _sum10 = __lasx_xvadd_w(_sum10, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA1, _pB0), _pA1, _pB0);
                    _sum20 = __lasx_xvadd_w(_sum20, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA1, _pB0r), _pA1, _pB0r);
                    _sum30 = __lasx_xvadd_w(_sum30, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA, _pB1), _pA, _pB1);
                    _sum01 = __lasx_xvadd_w(_sum01, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA, _pB1r), _pA, _pB1r);
                    _sum11 = __lasx_xvadd_w(_sum11, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    _sum21 = __lasx_xvadd_w(_sum21, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA1, _pB1r), _pA1, _pB1r);
                    _sum31 = __lasx_xvadd_w(_sum31, __lasx_xvhaddw_w_h(_s, _s));
                    pB0 += 32;
                    pB1 += 32;
                    pA += 16;
                }
                _sum20 = __lasx_xvshuf4i_w(_sum20, _LSX_SHUFFLE(1, 0, 3, 2));
                _sum30 = __lasx_xvshuf4i_w(_sum30, _LSX_SHUFFLE(1, 0, 3, 2));
                {
                    __m256i _tmp0 = __lasx_xvilvl_w(_sum10, _sum00);
                    __m256i _tmp1 = __lasx_xvilvh_w(_sum10, _sum00);
                    __m256i _tmp2 = __lasx_xvilvl_w(_sum30, _sum20);
                    __m256i _tmp3 = __lasx_xvilvh_w(_sum30, _sum20);
                    _sum00 = __lasx_xvilvl_d(_tmp2, _tmp0);
                    _sum10 = __lasx_xvilvh_d(_tmp2, _tmp0);
                    _sum20 = __lasx_xvilvl_d(_tmp3, _tmp1);
                    _sum30 = __lasx_xvilvh_d(_tmp3, _tmp1);
                }
                _sum10 = __lasx_xvshuf4i_w(_sum10, _LSX_SHUFFLE(2, 1, 0, 3));
                _sum20 = __lasx_xvshuf4i_w(_sum20, _LSX_SHUFFLE(1, 0, 3, 2));
                _sum30 = __lasx_xvshuf4i_w(_sum30, _LSX_SHUFFLE(0, 3, 2, 1));

                _sum21 = __lasx_xvshuf4i_w(_sum21, _LSX_SHUFFLE(1, 0, 3, 2));
                _sum31 = __lasx_xvshuf4i_w(_sum31, _LSX_SHUFFLE(1, 0, 3, 2));
                {
                    __m256i _tmp0 = __lasx_xvilvl_w(_sum11, _sum01);
                    __m256i _tmp1 = __lasx_xvilvh_w(_sum11, _sum01);
                    __m256i _tmp2 = __lasx_xvilvl_w(_sum31, _sum21);
                    __m256i _tmp3 = __lasx_xvilvh_w(_sum31, _sum21);
                    _sum01 = __lasx_xvilvl_d(_tmp2, _tmp0);
                    _sum11 = __lasx_xvilvh_d(_tmp2, _tmp0);
                    _sum21 = __lasx_xvilvl_d(_tmp3, _tmp1);
                    _sum31 = __lasx_xvilvh_d(_tmp3, _tmp1);
                }
                _sum11 = __lasx_xvshuf4i_w(_sum11, _LSX_SHUFFLE(2, 1, 0, 3));
                _sum21 = __lasx_xvshuf4i_w(_sum21, _LSX_SHUFFLE(1, 0, 3, 2));
                _sum31 = __lasx_xvshuf4i_w(_sum31, _LSX_SHUFFLE(0, 3, 2, 1));
                if (kk + 1 < max_kk)
                {
                    __m128i _pB0 = __lsx_vld(pB0, 0);
                    __m128i _pB1 = __lsx_vld(pB1, 0);
                    __m128i _pA = __lsx_vldrepl_d(pA, 0);
                    __m128i _pA0 = __lsx_vreplvei_h(_pA, 0);
                    __m128i _pA1 = __lsx_vreplvei_h(_pA, 1);
                    __m128i _pA2 = __lsx_vreplvei_h(_pA, 2);
                    __m128i _pA3 = __lsx_vreplvei_h(_pA, 3);
                    __m128i _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB0), _pA0, _pB0);
                    _sum00 = __lasx_xvadd_w(_sum00, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB1), _pA0, _pB1);
                    _sum01 = __lasx_xvadd_w(_sum01, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB0), _pA1, _pB0);
                    _sum10 = __lasx_xvadd_w(_sum10, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    _sum11 = __lasx_xvadd_w(_sum11, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA2, _pB0), _pA2, _pB0);
                    _sum20 = __lasx_xvadd_w(_sum20, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA2, _pB1), _pA2, _pB1);
                    _sum21 = __lasx_xvadd_w(_sum21, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA3, _pB0), _pA3, _pB0);
                    _sum30 = __lasx_xvadd_w(_sum30, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA3, _pB1), _pA3, _pB1);
                    _sum31 = __lasx_xvadd_w(_sum31, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    pB0 += 16;
                    pB1 += 16;
                    pA += 8;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    __m128i _pB0 = __lsx_vldrepl_d(pB0, 0);
                    __m128i _pB1 = __lsx_vldrepl_d(pB1, 0);
                    _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                    _pB1 = __lsx_vilvl_b(__lsx_vslti_b(_pB1, 0), _pB1);
                    __m128i _pA = __lsx_vldrepl_w(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _pA0 = __lsx_vreplvei_h(_pA, 0);
                    __m128i _pA1 = __lsx_vreplvei_h(_pA, 1);
                    __m128i _pA2 = __lsx_vreplvei_h(_pA, 2);
                    __m128i _pA3 = __lsx_vreplvei_h(_pA, 3);
                    __m128i _s = __lsx_vmul_h(_pA0, _pB0);
                    _sum00 = __lasx_xvadd_w(_sum00, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(_pA0, _pB1);
                    _sum01 = __lasx_xvadd_w(_sum01, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(_pA1, _pB0);
                    _sum10 = __lasx_xvadd_w(_sum10, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(_pA1, _pB1);
                    _sum11 = __lasx_xvadd_w(_sum11, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(_pA2, _pB0);
                    _sum20 = __lasx_xvadd_w(_sum20, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(_pA2, _pB1);
                    _sum21 = __lasx_xvadd_w(_sum21, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(_pA3, _pB0);
                    _sum30 = __lasx_xvadd_w(_sum30, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(_pA3, _pB1);
                    _sum31 = __lasx_xvadd_w(_sum31, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    pB0 += 8;
                    pB1 += 8;
                    pA += 4;
                }

                __m256 _bscale0 = (__m256)__lasx_xvld(pB_descales0, 0);
                __m256 _bscale1 = (__m256)__lasx_xvld(pB_descales1, 0);
                __m256 _ascale = (__m256)__lasx_xvreplfr2vr_s(pA_descales[0]);
                _out00 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum00), __lasx_xvfmul_s(_bscale0, _ascale), _out00);
                _out01 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum01), __lasx_xvfmul_s(_bscale1, _ascale), _out01);
                _ascale = (__m256)__lasx_xvreplfr2vr_s(pA_descales[1]);
                _out10 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum10), __lasx_xvfmul_s(_bscale0, _ascale), _out10);
                _out11 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum11), __lasx_xvfmul_s(_bscale1, _ascale), _out11);
                _ascale = (__m256)__lasx_xvreplfr2vr_s(pA_descales[2]);
                _out20 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum20), __lasx_xvfmul_s(_bscale0, _ascale), _out20);
                _out21 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum21), __lasx_xvfmul_s(_bscale1, _ascale), _out21);
                _ascale = (__m256)__lasx_xvreplfr2vr_s(pA_descales[3]);
                _out30 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum30), __lasx_xvfmul_s(_bscale0, _ascale), _out30);
                _out31 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum31), __lasx_xvfmul_s(_bscale1, _ascale), _out31);
                pA_descales += 4;
                pB_descales0 += 8;
                pB_descales1 += 8;
            }

            pB = pB1 + 8 * (full_K - k0 - max_kk0);
            pB_descales = pB_descales1 + 8 * (block_count - block_start - tile_blocks);

            __lasx_xvst(_out00, outptr, 0);
            __lasx_xvst(_out01, outptr + 8, 0);
            __lasx_xvst(_out10, outptr + 16, 0);
            __lasx_xvst(_out11, outptr + 24, 0);
            __lasx_xvst(_out20, outptr + 32, 0);
            __lasx_xvst(_out21, outptr + 40, 0);
            __lasx_xvst(_out30, outptr + 48, 0);
            __lasx_xvst(_out31, outptr + 56, 0);
            outptr += 64;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            pB += (size_t)8 * k0;
            pB_descales += (size_t)8 * block_start;
            __m256 _out0 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr, 0);
            __m256 _out1 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 8, 0);
            __m256 _out2 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 16, 0);
            __m256 _out3 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 24, 0);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                __m256i _sum0 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum1 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum2 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum3 = __lasx_xvreplgr2vr_w(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _pA4 = __lsx_vld(pA, 0);
                    __m256i _pA = __lasx_concat_128(_pA4, _pA4);
                    __m256i _pA1 = __lasx_xvshuf4i_w(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m256i _pB = __lasx_xvld(pB, 0);
                    __m256i _pB1 = __lasx_xvshuf4i_w(_pB, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m256i _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA, _pB), _pA, _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA, _pB1), _pA, _pB1);
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA1, _pB), _pA1, _pB);
                    _sum2 = __lasx_xvadd_w(_sum2, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    _sum3 = __lasx_xvadd_w(_sum3, __lasx_xvhaddw_w_h(_s, _s));
                    pB += 32;
                    pA += 16;
                }
                _sum2 = __lasx_xvshuf4i_w(_sum2, _LSX_SHUFFLE(1, 0, 3, 2));
                _sum3 = __lasx_xvshuf4i_w(_sum3, _LSX_SHUFFLE(1, 0, 3, 2));
                {
                    __m256i _tmp0 = __lasx_xvilvl_w(_sum1, _sum0);
                    __m256i _tmp1 = __lasx_xvilvh_w(_sum1, _sum0);
                    __m256i _tmp2 = __lasx_xvilvl_w(_sum3, _sum2);
                    __m256i _tmp3 = __lasx_xvilvh_w(_sum3, _sum2);
                    _sum0 = __lasx_xvilvl_d(_tmp2, _tmp0);
                    _sum1 = __lasx_xvilvh_d(_tmp2, _tmp0);
                    _sum2 = __lasx_xvilvl_d(_tmp3, _tmp1);
                    _sum3 = __lasx_xvilvh_d(_tmp3, _tmp1);
                }
                _sum1 = __lasx_xvshuf4i_w(_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
                _sum2 = __lasx_xvshuf4i_w(_sum2, _LSX_SHUFFLE(1, 0, 3, 2));
                _sum3 = __lasx_xvshuf4i_w(_sum3, _LSX_SHUFFLE(0, 3, 2, 1));
                if (kk + 1 < max_kk)
                {
                    __m128i _pB = __lsx_vld(pB, 0);
                    __m128i _pA = __lsx_vldrepl_d(pA, 0);
                    __m128i _pA0 = __lsx_vreplvei_h(_pA, 0);
                    __m128i _pA1 = __lsx_vreplvei_h(_pA, 1);
                    __m128i _pA2 = __lsx_vreplvei_h(_pA, 2);
                    __m128i _pA3 = __lsx_vreplvei_h(_pA, 3);
                    __m128i _s0 = __lsx_vmulwev_h_b(_pA0, _pB);
                    __m128i _s1 = __lsx_vmulwev_h_b(_pA1, _pB);
                    __m128i _s2 = __lsx_vmulwev_h_b(_pA2, _pB);
                    __m128i _s3 = __lsx_vmulwev_h_b(_pA3, _pB);
                    _s0 = __lsx_vmaddwod_h_b(_s0, _pA0, _pB);
                    _s1 = __lsx_vmaddwod_h_b(_s1, _pA1, _pB);
                    _s2 = __lsx_vmaddwod_h_b(_s2, _pA2, _pB);
                    _s3 = __lsx_vmaddwod_h_b(_s3, _pA3, _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(__lasx_cast_128(_s0)));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_vext2xv_w_h(__lasx_cast_128(_s1)));
                    _sum2 = __lasx_xvadd_w(_sum2, __lasx_vext2xv_w_h(__lasx_cast_128(_s2)));
                    _sum3 = __lasx_xvadd_w(_sum3, __lasx_vext2xv_w_h(__lasx_cast_128(_s3)));
                    pB += 16;
                    pA += 8;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    __m128i _pB = __lsx_vldrepl_d(pB, 0);
                    _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);
                    __m128i _pA = __lsx_vldrepl_w(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _s0 = __lsx_vmul_h(__lsx_vreplvei_h(_pA, 0), _pB);
                    __m128i _s1 = __lsx_vmul_h(__lsx_vreplvei_h(_pA, 1), _pB);
                    __m128i _s2 = __lsx_vmul_h(__lsx_vreplvei_h(_pA, 2), _pB);
                    __m128i _s3 = __lsx_vmul_h(__lsx_vreplvei_h(_pA, 3), _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(__lasx_cast_128(_s0)));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_vext2xv_w_h(__lasx_cast_128(_s1)));
                    _sum2 = __lasx_xvadd_w(_sum2, __lasx_vext2xv_w_h(__lasx_cast_128(_s2)));
                    _sum3 = __lasx_xvadd_w(_sum3, __lasx_vext2xv_w_h(__lasx_cast_128(_s3)));
                    pB += 8;
                    pA += 4;
                }

                __m256 _bscale = (__m256)__lasx_xvld(pB_descales, 0);
                __m256 _scale0 = __lasx_xvfmul_s(_bscale, (__m256)__lasx_xvreplfr2vr_s(pA_descales[0]));
                __m256 _scale1 = __lasx_xvfmul_s(_bscale, (__m256)__lasx_xvreplfr2vr_s(pA_descales[1]));
                __m256 _scale2 = __lasx_xvfmul_s(_bscale, (__m256)__lasx_xvreplfr2vr_s(pA_descales[2]));
                __m256 _scale3 = __lasx_xvfmul_s(_bscale, (__m256)__lasx_xvreplfr2vr_s(pA_descales[3]));
                _out0 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum0), _scale0, _out0);
                _out1 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum1), _scale1, _out1);
                _out2 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum2), _scale2, _out2);
                _out3 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum3), _scale3, _out3);
                pA_descales += 4;
                pB_descales += 8;
            }

            __lasx_xvst(_out0, outptr, 0);
            __lasx_xvst(_out1, outptr + 8, 0);
            __lasx_xvst(_out2, outptr + 16, 0);
            __lasx_xvst(_out3, outptr + 24, 0);
            outptr += 32;
            pB += (size_t)8 * (full_K - k0 - max_kk0);
            pB_descales += (size_t)8 * (block_count - block_start - tile_blocks);
        }
#endif // __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB0 = pB + 4 * k0;
            const signed char* pB1 = pB + 4 * full_K + 4 * k0;
            const float* pB_descales0 = pB_descales + 4 * block_start;
            const float* pB_descales1 = pB_descales + 4 * block_count + 4 * block_start;
            __m128 _out00 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr, 0);
            __m128 _out01 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 4, 0);
            __m128 _out10 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 8, 0);
            __m128 _out11 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 12, 0);
            __m128 _out20 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 16, 0);
            __m128 _out21 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 20, 0);
            __m128 _out30 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 24, 0);
            __m128 _out31 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 28, 0);
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                __m128i _sum00 = __lsx_vreplgr2vr_w(0);
                __m128i _sum01 = __lsx_vreplgr2vr_w(0);
                __m128i _sum10 = __lsx_vreplgr2vr_w(0);
                __m128i _sum11 = __lsx_vreplgr2vr_w(0);
                __m128i _sum20 = __lsx_vreplgr2vr_w(0);
                __m128i _sum21 = __lsx_vreplgr2vr_w(0);
                __m128i _sum30 = __lsx_vreplgr2vr_w(0);
                __m128i _sum31 = __lsx_vreplgr2vr_w(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _pA = __lsx_vld(pA, 0);
                    __m128i _pA1 = __lsx_vshuf4i_w(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m128i _pB0 = __lsx_vld(pB0, 0);
                    __m128i _pB1 = __lsx_vld(pB1, 0);
                    __m128i _pB0r = __lsx_vshuf4i_w(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m128i _pB1r = __lsx_vshuf4i_w(_pB1, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m128i _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA, _pB0), _pA, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA, _pB0r), _pA, _pB0r);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB0), _pA1, _pB0);
                    _sum20 = __lsx_vadd_w(_sum20, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB0r), _pA1, _pB0r);
                    _sum30 = __lsx_vadd_w(_sum30, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA, _pB1), _pA, _pB1);
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA, _pB1r), _pA, _pB1r);
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    _sum21 = __lsx_vadd_w(_sum21, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB1r), _pA1, _pB1r);
                    _sum31 = __lsx_vadd_w(_sum31, __lsx_vhaddw_w_h(_s, _s));
                    pB0 += 16;
                    pB1 += 16;
                    pA += 16;
                }
                _sum20 = __lsx_vshuf4i_w(_sum20, _LSX_SHUFFLE(1, 0, 3, 2));
                _sum30 = __lsx_vshuf4i_w(_sum30, _LSX_SHUFFLE(1, 0, 3, 2));
                transpose4x4_epi32(_sum00, _sum10, _sum20, _sum30);
                _sum10 = __lsx_vshuf4i_w(_sum10, _LSX_SHUFFLE(2, 1, 0, 3));
                _sum20 = __lsx_vshuf4i_w(_sum20, _LSX_SHUFFLE(1, 0, 3, 2));
                _sum30 = __lsx_vshuf4i_w(_sum30, _LSX_SHUFFLE(0, 3, 2, 1));

                _sum21 = __lsx_vshuf4i_w(_sum21, _LSX_SHUFFLE(1, 0, 3, 2));
                _sum31 = __lsx_vshuf4i_w(_sum31, _LSX_SHUFFLE(1, 0, 3, 2));
                transpose4x4_epi32(_sum01, _sum11, _sum21, _sum31);
                _sum11 = __lsx_vshuf4i_w(_sum11, _LSX_SHUFFLE(2, 1, 0, 3));
                _sum21 = __lsx_vshuf4i_w(_sum21, _LSX_SHUFFLE(1, 0, 3, 2));
                _sum31 = __lsx_vshuf4i_w(_sum31, _LSX_SHUFFLE(0, 3, 2, 1));
                if (kk + 1 < max_kk)
                {
                    __m128i _pB0 = __lsx_vldrepl_d(pB0, 0);
                    __m128i _pB1 = __lsx_vldrepl_d(pB1, 0);
                    __m128i _pA = __lsx_vldrepl_d(pA, 0);
                    __m128i _pA0 = __lsx_vreplvei_h(_pA, 0);
                    __m128i _pA1 = __lsx_vreplvei_h(_pA, 1);
                    __m128i _pA2 = __lsx_vreplvei_h(_pA, 2);
                    __m128i _pA3 = __lsx_vreplvei_h(_pA, 3);
                    __m128i _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB0), _pA0, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB1), _pA0, _pB1);
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB0), _pA1, _pB0);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA2, _pB0), _pA2, _pB0);
                    _sum20 = __lsx_vadd_w(_sum20, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA2, _pB1), _pA2, _pB1);
                    _sum21 = __lsx_vadd_w(_sum21, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA3, _pB0), _pA3, _pB0);
                    _sum30 = __lsx_vadd_w(_sum30, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA3, _pB1), _pA3, _pB1);
                    _sum31 = __lsx_vadd_w(_sum31, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    pB0 += 8;
                    pB1 += 8;
                    pA += 8;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    __m128i _pB0 = __lsx_vldrepl_w(pB0, 0);
                    __m128i _pB1 = __lsx_vldrepl_w(pB1, 0);
                    _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                    _pB1 = __lsx_vilvl_b(__lsx_vslti_b(_pB1, 0), _pB1);
                    __m128i _pA = __lsx_vldrepl_w(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _pA0 = __lsx_vreplvei_h(_pA, 0);
                    __m128i _pA1 = __lsx_vreplvei_h(_pA, 1);
                    __m128i _pA2 = __lsx_vreplvei_h(_pA, 2);
                    __m128i _pA3 = __lsx_vreplvei_h(_pA, 3);
                    __m128i _s = __lsx_vmul_h(_pA0, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA0, _pB1);
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA1, _pB0);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA1, _pB1);
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA2, _pB0);
                    _sum20 = __lsx_vadd_w(_sum20, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA2, _pB1);
                    _sum21 = __lsx_vadd_w(_sum21, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA3, _pB0);
                    _sum30 = __lsx_vadd_w(_sum30, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA3, _pB1);
                    _sum31 = __lsx_vadd_w(_sum31, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    pB0 += 4;
                    pB1 += 4;
                    pA += 4;
                }
                __m128 _bscale0 = (__m128)__lsx_vld(pB_descales0, 0);
                __m128 _bscale1 = (__m128)__lsx_vld(pB_descales1, 0);
                __m128 _ascale = __lsx_vreplfr2vr_s(pA_descales[0]);
                _out00 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum00), __lsx_vfmul_s(_bscale0, _ascale), _out00);
                _out01 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum01), __lsx_vfmul_s(_bscale1, _ascale), _out01);
                _ascale = __lsx_vreplfr2vr_s(pA_descales[1]);
                _out10 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum10), __lsx_vfmul_s(_bscale0, _ascale), _out10);
                _out11 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum11), __lsx_vfmul_s(_bscale1, _ascale), _out11);
                _ascale = __lsx_vreplfr2vr_s(pA_descales[2]);
                _out20 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum20), __lsx_vfmul_s(_bscale0, _ascale), _out20);
                _out21 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum21), __lsx_vfmul_s(_bscale1, _ascale), _out21);
                _ascale = __lsx_vreplfr2vr_s(pA_descales[3]);
                _out30 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum30), __lsx_vfmul_s(_bscale0, _ascale), _out30);
                _out31 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum31), __lsx_vfmul_s(_bscale1, _ascale), _out31);
                pA_descales += 4;
                pB_descales0 += 4;
                pB_descales1 += 4;
            }
            pB = pB1 + 4 * (full_K - k0 - max_kk0);
            pB_descales = pB_descales1 + 4 * (block_count - block_start - tile_blocks);
            __lsx_vst((__m128i)_out00, outptr, 0);
            __lsx_vst((__m128i)_out01, outptr + 4, 0);
            __lsx_vst((__m128i)_out10, outptr + 8, 0);
            __lsx_vst((__m128i)_out11, outptr + 12, 0);
            __lsx_vst((__m128i)_out20, outptr + 16, 0);
            __lsx_vst((__m128i)_out21, outptr + 20, 0);
            __lsx_vst((__m128i)_out30, outptr + 24, 0);
            __lsx_vst((__m128i)_out31, outptr + 28, 0);
            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            pB += (size_t)4 * k0;
            pB_descales += (size_t)4 * block_start;
            __m128 _out0 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr, 0);
            __m128 _out1 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 4, 0);
            __m128 _out2 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 8, 0);
            __m128 _out3 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 12, 0);
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                __m128i _sum1 = __lsx_vreplgr2vr_w(0);
                __m128i _sum2 = __lsx_vreplgr2vr_w(0);
                __m128i _sum3 = __lsx_vreplgr2vr_w(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _pA = __lsx_vld(pA, 0);
                    __m128i _pA1 = __lsx_vshuf4i_w(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m128i _pB = __lsx_vld(pB, 0);
                    __m128i _pB1 = __lsx_vshuf4i_w(_pB, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m128i _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA, _pB), _pA, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA, _pB1), _pA, _pB1);
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB), _pA1, _pB);
                    _sum2 = __lsx_vadd_w(_sum2, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    _sum3 = __lsx_vadd_w(_sum3, __lsx_vhaddw_w_h(_s, _s));
                    pB += 16;
                    pA += 16;
                }
                _sum2 = __lsx_vshuf4i_w(_sum2, _LSX_SHUFFLE(1, 0, 3, 2));
                _sum3 = __lsx_vshuf4i_w(_sum3, _LSX_SHUFFLE(1, 0, 3, 2));
                transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
                _sum1 = __lsx_vshuf4i_w(_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
                _sum2 = __lsx_vshuf4i_w(_sum2, _LSX_SHUFFLE(1, 0, 3, 2));
                _sum3 = __lsx_vshuf4i_w(_sum3, _LSX_SHUFFLE(0, 3, 2, 1));
                if (kk + 1 < max_kk)
                {
                    __m128i _pB = __lsx_vldrepl_d(pB, 0);
                    __m128i _pA = __lsx_vldrepl_d(pA, 0);
                    __m128i _pA0 = __lsx_vreplvei_h(_pA, 0);
                    __m128i _pA1 = __lsx_vreplvei_h(_pA, 1);
                    __m128i _pA2 = __lsx_vreplvei_h(_pA, 2);
                    __m128i _pA3 = __lsx_vreplvei_h(_pA, 3);
                    __m128i _s0 = __lsx_vmulwev_h_b(_pA0, _pB);
                    __m128i _s1 = __lsx_vmulwev_h_b(_pA1, _pB);
                    __m128i _s2 = __lsx_vmulwev_h_b(_pA2, _pB);
                    __m128i _s3 = __lsx_vmulwev_h_b(_pA3, _pB);
                    _s0 = __lsx_vmaddwod_h_b(_s0, _pA0, _pB);
                    _s1 = __lsx_vmaddwod_h_b(_s1, _pA1, _pB);
                    _s2 = __lsx_vmaddwod_h_b(_s2, _pA2, _pB);
                    _s3 = __lsx_vmaddwod_h_b(_s3, _pA3, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    _sum2 = __lsx_vadd_w(_sum2, __lsx_vilvl_h(__lsx_vslti_h(_s2, 0), _s2));
                    _sum3 = __lsx_vadd_w(_sum3, __lsx_vilvl_h(__lsx_vslti_h(_s3, 0), _s3));
                    pB += 8;
                    pA += 8;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    __m128i _pB = __lsx_vldrepl_w(pB, 0);
                    _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);
                    __m128i _pA = __lsx_vldrepl_w(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _s0 = __lsx_vmul_h(__lsx_vreplvei_h(_pA, 0), _pB);
                    __m128i _s1 = __lsx_vmul_h(__lsx_vreplvei_h(_pA, 1), _pB);
                    __m128i _s2 = __lsx_vmul_h(__lsx_vreplvei_h(_pA, 2), _pB);
                    __m128i _s3 = __lsx_vmul_h(__lsx_vreplvei_h(_pA, 3), _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    _sum2 = __lsx_vadd_w(_sum2, __lsx_vilvl_h(__lsx_vslti_h(_s2, 0), _s2));
                    _sum3 = __lsx_vadd_w(_sum3, __lsx_vilvl_h(__lsx_vslti_h(_s3, 0), _s3));
                    pB += 4;
                    pA += 4;
                }
                __m128 _bscale = (__m128)__lsx_vld(pB_descales, 0);
                _out0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum0), __lsx_vfmul_s(_bscale, __lsx_vreplfr2vr_s(pA_descales[0])), _out0);
                _out1 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum1), __lsx_vfmul_s(_bscale, __lsx_vreplfr2vr_s(pA_descales[1])), _out1);
                _out2 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum2), __lsx_vfmul_s(_bscale, __lsx_vreplfr2vr_s(pA_descales[2])), _out2);
                _out3 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum3), __lsx_vfmul_s(_bscale, __lsx_vreplfr2vr_s(pA_descales[3])), _out3);
                pA_descales += 4;
                pB_descales += 4;
            }
            __lsx_vst((__m128i)_out0, outptr, 0);
            __lsx_vst((__m128i)_out1, outptr + 4, 0);
            __lsx_vst((__m128i)_out2, outptr + 8, 0);
            __lsx_vst((__m128i)_out3, outptr + 12, 0);
            outptr += 16;
            pB += (size_t)4 * (full_K - k0 - max_kk0);
            pB_descales += (size_t)4 * (block_count - block_start - tile_blocks);
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            pB += (size_t)2 * k0;
            pB_descales += (size_t)2 * block_start;
            __m128 _out0 = (__m128)__lsx_vldi(0);
            __m128 _out1 = (__m128)__lsx_vldi(0);
            if (k0 != 0)
            {
                __m128i _out01 = __lsx_vld(outptr, 0);
                __m128i _out23 = __lsx_vld(outptr + 4, 0);
                _out0 = (__m128)__lsx_vpickev_w(_out23, _out01);
                _out1 = (__m128)__lsx_vpickod_w(_out23, _out01);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                __m128i _sum1 = __lsx_vreplgr2vr_w(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _pA = __lsx_vld(pA, 0);
                    __m128i _pB0 = __lsx_vldrepl_d(pB, 0);
                    __m128i _pB1 = __lsx_vshuf4i_w(_pB0, _LSX_SHUFFLE(2, 3, 0, 1));
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA, _pB0), _pA, _pB0);
                    __m128i _s1 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA, _pB1), _pA, _pB1);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s1, _s1));
                    pA += 16;
                    pB += 8;
                }
                if (kk + 1 < max_kk)
                {
                    __m128i _pA = __lsx_vldrepl_d(pA, 0);
                    __m128i _pA0 = __lsx_vreplvei_w(__lsx_vpickev_b(_pA, _pA), 0);
                    __m128i _pA1 = __lsx_vreplvei_w(__lsx_vpickod_b(_pA, _pA), 0);
                    _pA0 = __lsx_vilvl_b(__lsx_vslti_b(_pA0, 0), _pA0);
                    _pA1 = __lsx_vilvl_b(__lsx_vslti_b(_pA1, 0), _pA1);
                    __m128i _pBs = __lsx_vldrepl_w(pB, 0);
                    __m128i _pB0 = __lsx_vreplvei_w(__lsx_vpickev_b(_pBs, _pBs), 0);
                    _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                    _pB0 = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(1, 0, 1, 0));
                    __m128i _pB1 = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m128i _s0 = __lsx_vmul_h(_pA0, _pB0);
                    __m128i _s1 = __lsx_vmul_h(_pA0, _pB1);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    _pB0 = __lsx_vreplvei_w(__lsx_vpickod_b(_pBs, _pBs), 0);
                    _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                    _pB0 = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(1, 0, 1, 0));
                    _pB1 = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                    _s0 = __lsx_vmul_h(_pA1, _pB0);
                    _s1 = __lsx_vmul_h(_pA1, _pB1);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    pA += 8;
                    pB += 4;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    __m128i _pA = __lsx_vldrepl_w(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _pB0 = __lsx_vldrepl_h(pB, 0);
                    _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                    __m128i _pB1 = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m128i _s0 = __lsx_vmul_h(_pA, _pB0);
                    __m128i _s1 = __lsx_vmul_h(_pA, _pB1);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    pA += 4;
                    pB += 2;
                }
                __m128i _sum0e = __lsx_vshuf4i_w(_sum0, _LSX_SHUFFLE(3, 1, 2, 0));
                __m128i _sum0o = __lsx_vshuf4i_w(_sum0, _LSX_SHUFFLE(2, 0, 3, 1));
                __m128i _sum1e = __lsx_vshuf4i_w(_sum1, _LSX_SHUFFLE(3, 1, 2, 0));
                __m128i _sum1o = __lsx_vshuf4i_w(_sum1, _LSX_SHUFFLE(2, 0, 3, 1));
                __m128i _sumc0 = __lsx_vilvl_w(_sum1o, _sum0e);
                __m128i _sumc1 = __lsx_vilvl_w(_sum0o, _sum1e);
                __m128 _ascale = (__m128)__lsx_vld(pA_descales, 0);
                _out0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sumc0), __lsx_vfmul_s(_ascale, __lsx_vreplfr2vr_s(pB_descales[0])), _out0);
                _out1 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sumc1), __lsx_vfmul_s(_ascale, __lsx_vreplfr2vr_s(pB_descales[1])), _out1);
                pA_descales += 4;
                pB_descales += 2;
            }
            __lsx_vstelm_w((__m128i)_out0, outptr, 0, 0);
            __lsx_vstelm_w((__m128i)_out1, outptr + 1, 0, 0);
            __lsx_vstelm_w((__m128i)_out0, outptr + 2, 0, 1);
            __lsx_vstelm_w((__m128i)_out1, outptr + 3, 0, 1);
            __lsx_vstelm_w((__m128i)_out0, outptr + 4, 0, 2);
            __lsx_vstelm_w((__m128i)_out1, outptr + 5, 0, 2);
            __lsx_vstelm_w((__m128i)_out0, outptr + 6, 0, 3);
            __lsx_vstelm_w((__m128i)_out1, outptr + 7, 0, 3);
            outptr += 8;
            pB += (size_t)2 * (full_K - k0 - max_kk0);
            pB_descales += (size_t)2 * (block_count - block_start - tile_blocks);
        }
        for (; jj < max_jj; jj++)
        {
            pB += k0;
            pB_descales += block_start;
            __m128 _out0 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr, 0);
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _pA = __lsx_vld(pA, 0);
                    __m128i _pB = __lsx_vldrepl_w(pB, 0);
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA, _pB), _pA, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                    pA += 16;
                    pB += 4;
                }
                if (kk + 1 < max_kk)
                {
                    __m128i _pA = __lsx_vldrepl_d(pA, 0);
                    __m128i _pB = __lsx_vldrepl_h(pB, 0);
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA, _pB), _pA, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    pA += 8;
                    pB += 2;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    __m128i _pA = __lsx_vldrepl_w(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _s0 = __lsx_vmul_h(_pA, __lsx_vreplgr2vr_h(pB[0]));
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    pA += 4;
                    pB++;
                }
                __m128 _scale = __lsx_vfmul_s((__m128)__lsx_vld(pA_descales, 0), __lsx_vreplfr2vr_s(*pB_descales++));
                _out0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum0), _scale, _out0);
                pA_descales += 4;
            }
            __lsx_vst((__m128i)_out0, outptr, 0);
            outptr += 4;
            pB += full_K - k0 - max_kk0;
            pB_descales += block_count - block_start - tile_blocks;
        }
        pAT += A_hstep * 4;
        pAT_descales += A_descales_hstep * 4;
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;

        int jj = 0;
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            const signed char* pB0 = pB + 8 * k0;
            const signed char* pB1 = pB + 8 * full_K + 8 * k0;
            const float* pB_descales0 = pB_descales + 8 * block_start;
            const float* pB_descales1 = pB_descales + 8 * block_count + 8 * block_start;
            __m256 _out00 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr, 0);
            __m256 _out01 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 8, 0);
            __m256 _out10 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 16, 0);
            __m256 _out11 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 24, 0);
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                __m256i _sum00 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum01 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum10 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum11 = __lasx_xvreplgr2vr_w(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m256i _pB0 = __lasx_xvld(pB0, 0);
                    __m256i _pB1 = __lasx_xvld(pB1, 0);
                    __m256i _pA0 = __lasx_xvldrepl_w(pA, 0);
                    __m256i _pA1 = __lasx_xvldrepl_w(pA + 4, 0);
                    __m256i _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA0, _pB0), _pA0, _pB0);
                    _sum00 = __lasx_xvadd_w(_sum00, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA0, _pB1), _pA0, _pB1);
                    _sum01 = __lasx_xvadd_w(_sum01, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA1, _pB0), _pA1, _pB0);
                    _sum10 = __lasx_xvadd_w(_sum10, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    _sum11 = __lasx_xvadd_w(_sum11, __lasx_xvhaddw_w_h(_s, _s));
                    pB0 += 32;
                    pB1 += 32;
                    pA += 8;
                }
                if (kk + 1 < max_kk)
                {
                    __m128i _pB0 = __lsx_vld(pB0, 0);
                    __m128i _pB1 = __lsx_vld(pB1, 0);
                    __m128i _pAs = __lsx_vldrepl_w(pA, 0);
                    __m128i _pA0 = __lsx_vreplvei_h(_pAs, 0);
                    __m128i _pA1 = __lsx_vreplvei_h(_pAs, 1);
                    __m128i _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB0), _pA0, _pB0);
                    _sum00 = __lasx_xvadd_w(_sum00, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB1), _pA0, _pB1);
                    _sum01 = __lasx_xvadd_w(_sum01, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB0), _pA1, _pB0);
                    _sum10 = __lasx_xvadd_w(_sum10, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    _sum11 = __lasx_xvadd_w(_sum11, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    pB0 += 16;
                    pB1 += 16;
                    pA += 4;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    __m128i _pB0 = __lsx_vldrepl_d(pB0, 0);
                    __m128i _pB1 = __lsx_vldrepl_d(pB1, 0);
                    _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                    _pB1 = __lsx_vilvl_b(__lsx_vslti_b(_pB1, 0), _pB1);
                    __m128i _pA = __lsx_vldrepl_h(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _pA0 = __lsx_vreplvei_h(_pA, 0);
                    __m128i _pA1 = __lsx_vreplvei_h(_pA, 1);
                    __m128i _s = __lsx_vmul_h(_pA0, _pB0);
                    _sum00 = __lasx_xvadd_w(_sum00, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(_pA0, _pB1);
                    _sum01 = __lasx_xvadd_w(_sum01, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(_pA1, _pB0);
                    _sum10 = __lasx_xvadd_w(_sum10, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(_pA1, _pB1);
                    _sum11 = __lasx_xvadd_w(_sum11, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    pB0 += 8;
                    pB1 += 8;
                    pA += 2;
                }
                __m256 _bscale0 = (__m256)__lasx_xvld(pB_descales0, 0);
                __m256 _bscale1 = (__m256)__lasx_xvld(pB_descales1, 0);
                __m256 _ascale = (__m256)__lasx_xvreplfr2vr_s(pA_descales[0]);
                _out00 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum00), __lasx_xvfmul_s(_bscale0, _ascale), _out00);
                _out01 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum01), __lasx_xvfmul_s(_bscale1, _ascale), _out01);
                _ascale = (__m256)__lasx_xvreplfr2vr_s(pA_descales[1]);
                _out10 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum10), __lasx_xvfmul_s(_bscale0, _ascale), _out10);
                _out11 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum11), __lasx_xvfmul_s(_bscale1, _ascale), _out11);
                pA_descales += 2;
                pB_descales0 += 8;
                pB_descales1 += 8;
            }
            pB = pB1 + 8 * (full_K - k0 - max_kk0);
            pB_descales = pB_descales1 + 8 * (block_count - block_start - tile_blocks);
            __lasx_xvst(_out00, outptr, 0);
            __lasx_xvst(_out01, outptr + 8, 0);
            __lasx_xvst(_out10, outptr + 16, 0);
            __lasx_xvst(_out11, outptr + 24, 0);
            outptr += 32;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            pB += (size_t)8 * k0;
            pB_descales += (size_t)8 * block_start;
            __m256 _out0 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr, 0);
            __m256 _out1 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 8, 0);
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                __m256i _sum0 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum1 = __lasx_xvreplgr2vr_w(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m256i _pB = __lasx_xvld(pB, 0);
                    __m256i _pA0 = __lasx_xvldrepl_w(pA, 0);
                    __m256i _pA1 = __lasx_xvldrepl_w(pA + 4, 0);
                    __m256i _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA0, _pB), _pA0, _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA1, _pB), _pA1, _pB);
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvhaddw_w_h(_s, _s));
                    pB += 32;
                    pA += 8;
                }
                if (kk + 1 < max_kk)
                {
                    __m128i _pB = __lsx_vld(pB, 0);
                    __m128i _pAs = __lsx_vldrepl_w(pA, 0);
                    __m128i _pA0 = __lsx_vreplvei_h(_pAs, 0);
                    __m128i _pA1 = __lsx_vreplvei_h(_pAs, 1);
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB), _pA0, _pB);
                    __m128i _s1 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB), _pA1, _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(__lasx_cast_128(_s0)));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_vext2xv_w_h(__lasx_cast_128(_s1)));
                    pB += 16;
                    pA += 4;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    __m128i _pB = __lsx_vldrepl_d(pB, 0);
                    _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);
                    __m128i _pA = __lsx_vldrepl_h(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _s0 = __lsx_vmul_h(__lsx_vreplvei_h(_pA, 0), _pB);
                    __m128i _s1 = __lsx_vmul_h(__lsx_vreplvei_h(_pA, 1), _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(__lasx_cast_128(_s0)));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_vext2xv_w_h(__lasx_cast_128(_s1)));
                    pB += 8;
                    pA += 2;
                }
                __m256 _bscale = (__m256)__lasx_xvld(pB_descales, 0);
                _out0 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum0), __lasx_xvfmul_s(_bscale, (__m256)__lasx_xvreplfr2vr_s(pA_descales[0])), _out0);
                _out1 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum1), __lasx_xvfmul_s(_bscale, (__m256)__lasx_xvreplfr2vr_s(pA_descales[1])), _out1);
                pA_descales += 2;
                pB_descales += 8;
            }
            __lasx_xvst(_out0, outptr, 0);
            __lasx_xvst(_out1, outptr + 8, 0);
            outptr += 16;
            pB += (size_t)8 * (full_K - k0 - max_kk0);
            pB_descales += (size_t)8 * (block_count - block_start - tile_blocks);
        }
#endif // __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB0 = pB + 4 * k0;
            const signed char* pB1 = pB + 4 * full_K + 4 * k0;
            const float* pB_descales0 = pB_descales + 4 * block_start;
            const float* pB_descales1 = pB_descales + 4 * block_count + 4 * block_start;
            __m128 _out00 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr, 0);
            __m128 _out01 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 4, 0);
            __m128 _out10 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 8, 0);
            __m128 _out11 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 12, 0);
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                __m128i _sum00 = __lsx_vreplgr2vr_w(0);
                __m128i _sum01 = __lsx_vreplgr2vr_w(0);
                __m128i _sum10 = __lsx_vreplgr2vr_w(0);
                __m128i _sum11 = __lsx_vreplgr2vr_w(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _pB0 = __lsx_vld(pB0, 0);
                    __m128i _pB1 = __lsx_vld(pB1, 0);
                    __m128i _pA0 = __lsx_vldrepl_w(pA, 0);
                    __m128i _pA1 = __lsx_vldrepl_w(pA + 4, 0);
                    __m128i _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB0), _pA0, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB1), _pA0, _pB1);
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB0), _pA1, _pB0);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vhaddw_w_h(_s, _s));
                    pB0 += 16;
                    pB1 += 16;
                    pA += 8;
                }
                if (kk + 1 < max_kk)
                {
                    __m128i _pB0 = __lsx_vldrepl_d(pB0, 0);
                    __m128i _pB1 = __lsx_vldrepl_d(pB1, 0);
                    __m128i _pAs = __lsx_vldrepl_w(pA, 0);
                    __m128i _pA0 = __lsx_vreplvei_h(_pAs, 0);
                    __m128i _pA1 = __lsx_vreplvei_h(_pAs, 1);
                    __m128i _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB0), _pA0, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB1), _pA0, _pB1);
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB0), _pA1, _pB0);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    pB0 += 8;
                    pB1 += 8;
                    pA += 4;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    __m128i _pB0 = __lsx_vldrepl_w(pB0, 0);
                    __m128i _pB1 = __lsx_vldrepl_w(pB1, 0);
                    _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                    _pB1 = __lsx_vilvl_b(__lsx_vslti_b(_pB1, 0), _pB1);
                    __m128i _pA = __lsx_vldrepl_h(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _pA0 = __lsx_vreplvei_h(_pA, 0);
                    __m128i _pA1 = __lsx_vreplvei_h(_pA, 1);
                    __m128i _s = __lsx_vmul_h(_pA0, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA0, _pB1);
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA1, _pB0);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA1, _pB1);
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    pB0 += 4;
                    pB1 += 4;
                    pA += 2;
                }
                __m128 _bscale0 = (__m128)__lsx_vld(pB_descales0, 0);
                __m128 _bscale1 = (__m128)__lsx_vld(pB_descales1, 0);
                __m128 _ascale = __lsx_vreplfr2vr_s(pA_descales[0]);
                _out00 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum00), __lsx_vfmul_s(_bscale0, _ascale), _out00);
                _out01 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum01), __lsx_vfmul_s(_bscale1, _ascale), _out01);
                _ascale = __lsx_vreplfr2vr_s(pA_descales[1]);
                _out10 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum10), __lsx_vfmul_s(_bscale0, _ascale), _out10);
                _out11 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum11), __lsx_vfmul_s(_bscale1, _ascale), _out11);
                pA_descales += 2;
                pB_descales0 += 4;
                pB_descales1 += 4;
            }
            pB = pB1 + 4 * (full_K - k0 - max_kk0);
            pB_descales = pB_descales1 + 4 * (block_count - block_start - tile_blocks);
            __lsx_vst((__m128i)_out00, outptr, 0);
            __lsx_vst((__m128i)_out01, outptr + 4, 0);
            __lsx_vst((__m128i)_out10, outptr + 8, 0);
            __lsx_vst((__m128i)_out11, outptr + 12, 0);
            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            pB += (size_t)4 * k0;
            pB_descales += (size_t)4 * block_start;
            __m128 _out0 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr, 0);
            __m128 _out1 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 4, 0);
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                __m128i _sum1 = __lsx_vreplgr2vr_w(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _pB = __lsx_vld(pB, 0);
                    __m128i _pA0 = __lsx_vldrepl_w(pA, 0);
                    __m128i _pA1 = __lsx_vldrepl_w(pA + 4, 0);
                    __m128i _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB), _pA0, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB), _pA1, _pB);
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s, _s));
                    pB += 16;
                    pA += 8;
                }
                if (kk + 1 < max_kk)
                {
                    __m128i _pB = __lsx_vldrepl_d(pB, 0);
                    __m128i _pAs = __lsx_vldrepl_w(pA, 0);
                    __m128i _pA0 = __lsx_vreplvei_h(_pAs, 0);
                    __m128i _pA1 = __lsx_vreplvei_h(_pAs, 1);
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB), _pA0, _pB);
                    __m128i _s1 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB), _pA1, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    pB += 8;
                    pA += 4;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    __m128i _pB = __lsx_vldrepl_w(pB, 0);
                    _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);
                    __m128i _pA = __lsx_vldrepl_h(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _s0 = __lsx_vmul_h(__lsx_vreplvei_h(_pA, 0), _pB);
                    __m128i _s1 = __lsx_vmul_h(__lsx_vreplvei_h(_pA, 1), _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    pB += 4;
                    pA += 2;
                }
                __m128 _bscale = (__m128)__lsx_vld(pB_descales, 0);
                _out0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum0), __lsx_vfmul_s(_bscale, __lsx_vreplfr2vr_s(pA_descales[0])), _out0);
                _out1 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum1), __lsx_vfmul_s(_bscale, __lsx_vreplfr2vr_s(pA_descales[1])), _out1);
                pA_descales += 2;
                pB_descales += 4;
            }
            __lsx_vst((__m128i)_out0, outptr, 0);
            __lsx_vst((__m128i)_out1, outptr + 4, 0);
            outptr += 8;
            pB += (size_t)4 * (full_K - k0 - max_kk0);
            pB_descales += (size_t)4 * (block_count - block_start - tile_blocks);
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            pB += (size_t)2 * k0;
            pB_descales += (size_t)2 * block_start;
            float _out00 = k0 == 0 ? 0.f : outptr[0];
            float _out01 = k0 == 0 ? 0.f : outptr[1];
            float _out10 = k0 == 0 ? 0.f : outptr[2];
            float _out11 = k0 == 0 ? 0.f : outptr[3];
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                int _sum00 = 0;
                int _sum01 = 0;
                int _sum10 = 0;
                int _sum11 = 0;
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    asm volatile(""
                                 :
                                 :
                                 : "memory");

                    _sum00 += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    _sum01 += pA[0] * pB[4] + pA[1] * pB[5] + pA[2] * pB[6] + pA[3] * pB[7];
                    _sum10 += pA[4] * pB[0] + pA[5] * pB[1] + pA[6] * pB[2] + pA[7] * pB[3];
                    _sum11 += pA[4] * pB[4] + pA[5] * pB[5] + pA[6] * pB[6] + pA[7] * pB[7];
                    pB += 8;
                    pA += 8;
                }
                if (kk + 1 < max_kk)
                {
                    _sum00 += pA[0] * pB[0] + pA[1] * pB[1];
                    _sum01 += pA[0] * pB[2] + pA[1] * pB[3];
                    _sum10 += pA[2] * pB[0] + pA[3] * pB[1];
                    _sum11 += pA[2] * pB[2] + pA[3] * pB[3];
                    pB += 4;
                    pA += 4;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    _sum00 += pA[0] * pB[0];
                    _sum01 += pA[0] * pB[1];
                    _sum10 += pA[1] * pB[0];
                    _sum11 += pA[1] * pB[1];
                    pB += 2;
                    pA += 2;
                }
                const float bscale0 = pB_descales[0];
                const float bscale1 = pB_descales[1];
                const float ascale0 = pA_descales[0];
                const float ascale1 = pA_descales[1];
                _out00 += _sum00 * ascale0 * bscale0;
                _out01 += _sum01 * ascale0 * bscale1;
                _out10 += _sum10 * ascale1 * bscale0;
                _out11 += _sum11 * ascale1 * bscale1;
                pA_descales += 2;
                pB_descales += 2;
            }
            outptr[0] = _out00;
            outptr[1] = _out01;
            outptr[2] = _out10;
            outptr[3] = _out11;
            outptr += 4;
            pB += (size_t)2 * (full_K - k0 - max_kk0);
            pB_descales += (size_t)2 * (block_count - block_start - tile_blocks);
        }
        for (; jj < max_jj; jj++)
        {
            pB += k0;
            pB_descales += block_start;
            float _out0 = k0 == 0 ? 0.f : outptr[0];
            float _out1 = k0 == 0 ? 0.f : outptr[1];
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                int _sum0 = 0;
                int _sum1 = 0;
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    asm volatile(""
                                 :
                                 :
                                 : "memory");

                    _sum0 += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    _sum1 += pA[4] * pB[0] + pA[5] * pB[1] + pA[6] * pB[2] + pA[7] * pB[3];
                    pB += 4;
                    pA += 8;
                }
                if (kk + 1 < max_kk)
                {
                    _sum0 += pA[0] * pB[0] + pA[1] * pB[1];
                    _sum1 += pA[2] * pB[0] + pA[3] * pB[1];
                    pB += 2;
                    pA += 4;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    _sum0 += pA[0] * pB[0];
                    _sum1 += pA[1] * pB[0];
                    pB++;
                    pA += 2;
                }
                const float bscale = *pB_descales++;
                _out0 += _sum0 * pA_descales[0] * bscale;
                _out1 += _sum1 * pA_descales[1] * bscale;
                pA_descales += 2;
            }
            outptr[0] = _out0;
            outptr[1] = _out1;
            outptr += 2;
            pB += full_K - k0 - max_kk0;
            pB_descales += block_count - block_start - tile_blocks;
        }
        pAT += A_hstep * 2;
        pAT_descales += A_descales_hstep * 2;
    }
#endif // __loongarch_sx
#if !__loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;

        int jj = 0;
        for (; jj + 1 < max_jj; jj += 2)
        {
            pB += (size_t)2 * k0;
            pB_descales += (size_t)2 * block_start;
            float _out00 = k0 == 0 ? 0.f : outptr[0];
            float _out01 = k0 == 0 ? 0.f : outptr[1];
            float _out10 = k0 == 0 ? 0.f : outptr[2];
            float _out11 = k0 == 0 ? 0.f : outptr[3];
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                int _sum00 = 0;
                int _sum01 = 0;
                int _sum10 = 0;
                int _sum11 = 0;
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    asm volatile(""
                                 :
                                 :
                                 : "memory");

                    _sum00 += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    _sum01 += pA[0] * pB[4] + pA[1] * pB[5] + pA[2] * pB[6] + pA[3] * pB[7];
                    _sum10 += pA[4] * pB[0] + pA[5] * pB[1] + pA[6] * pB[2] + pA[7] * pB[3];
                    _sum11 += pA[4] * pB[4] + pA[5] * pB[5] + pA[6] * pB[6] + pA[7] * pB[7];
                    pB += 8;
                    pA += 8;
                }
                if (kk + 1 < max_kk)
                {
                    _sum00 += pA[0] * pB[0] + pA[1] * pB[1];
                    _sum01 += pA[0] * pB[2] + pA[1] * pB[3];
                    _sum10 += pA[2] * pB[0] + pA[3] * pB[1];
                    _sum11 += pA[2] * pB[2] + pA[3] * pB[3];
                    pB += 4;
                    pA += 4;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    _sum00 += pA[0] * pB[0];
                    _sum01 += pA[0] * pB[1];
                    _sum10 += pA[1] * pB[0];
                    _sum11 += pA[1] * pB[1];
                    pB += 2;
                    pA += 2;
                }
                const float bscale0 = pB_descales[0];
                const float bscale1 = pB_descales[1];
                const float ascale0 = pA_descales[0];
                const float ascale1 = pA_descales[1];
                _out00 += _sum00 * ascale0 * bscale0;
                _out01 += _sum01 * ascale0 * bscale1;
                _out10 += _sum10 * ascale1 * bscale0;
                _out11 += _sum11 * ascale1 * bscale1;
                pA_descales += 2;
                pB_descales += 2;
            }
            outptr[0] = _out00;
            outptr[1] = _out01;
            outptr[2] = _out10;
            outptr[3] = _out11;
            outptr += 4;
            pB += (size_t)2 * (full_K - k0 - max_kk0);
            pB_descales += (size_t)2 * (block_count - block_start - tile_blocks);
        }
        for (; jj < max_jj; jj++)
        {
            pB += k0;
            pB_descales += block_start;
            float _out0 = k0 == 0 ? 0.f : outptr[0];
            float _out1 = k0 == 0 ? 0.f : outptr[1];
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                int _sum0 = 0;
                int _sum1 = 0;
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    asm volatile(""
                                 :
                                 :
                                 : "memory");

                    _sum0 += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    _sum1 += pA[4] * pB[0] + pA[5] * pB[1] + pA[6] * pB[2] + pA[7] * pB[3];
                    pB += 4;
                    pA += 8;
                }
                if (kk + 1 < max_kk)
                {
                    _sum0 += pA[0] * pB[0] + pA[1] * pB[1];
                    _sum1 += pA[2] * pB[0] + pA[3] * pB[1];
                    pB += 2;
                    pA += 4;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    _sum0 += pA[0] * pB[0];
                    _sum1 += pA[1] * pB[0];
                    pB++;
                    pA += 2;
                }
                const float bscale = *pB_descales++;
                _out0 += _sum0 * pA_descales[0] * bscale;
                _out1 += _sum1 * pA_descales[1] * bscale;
                pA_descales += 2;
            }
            outptr[0] = _out0;
            outptr[1] = _out1;
            outptr += 2;
            pB += full_K - k0 - max_kk0;
            pB_descales += block_count - block_start - tile_blocks;
        }
        pAT += A_hstep * 2;
        pAT_descales += A_descales_hstep * 2;
    }
#endif // !__loongarch_sx
    for (; ii < max_ii; ii++)
    {
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;

        int jj = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            const signed char* pB0 = pB + 8 * k0;
            const signed char* pB1 = pB + 8 * full_K + 8 * k0;
            const float* pB_descales0 = pB_descales + 8 * block_start;
            const float* pB_descales1 = pB_descales + 8 * block_count + 8 * block_start;
            __m256 _out00 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr, 0);
            __m256 _out01 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 8, 0);
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                __m256i _sum00 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum01 = __lasx_xvreplgr2vr_w(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m256i _pB0 = __lasx_xvld(pB0, 0);
                    __m256i _pB1 = __lasx_xvld(pB1, 0);
                    __m256i _pA0 = __lasx_xvldrepl_w(pA, 0);
                    __m256i _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA0, _pB0), _pA0, _pB0);
                    _sum00 = __lasx_xvadd_w(_sum00, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA0, _pB1), _pA0, _pB1);
                    _sum01 = __lasx_xvadd_w(_sum01, __lasx_xvhaddw_w_h(_s, _s));
                    pB0 += 32;
                    pB1 += 32;
                    pA += 4;
                }
                if (kk + 1 < max_kk)
                {
                    __m128i _pB0 = __lsx_vld(pB0, 0);
                    __m128i _pB1 = __lsx_vld(pB1, 0);
                    __m128i _pA0 = __lsx_vreplgr2vr_h((unsigned char)pA[0] | ((unsigned char)pA[1] << 8));
                    __m128i _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB0), _pA0, _pB0);
                    _sum00 = __lasx_xvadd_w(_sum00, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB1), _pA0, _pB1);
                    _sum01 = __lasx_xvadd_w(_sum01, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    pB0 += 16;
                    pB1 += 16;
                    pA += 2;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    __m128i _pB0 = __lsx_vldrepl_d(pB0, 0);
                    __m128i _pB1 = __lsx_vldrepl_d(pB1, 0);
                    _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                    _pB1 = __lsx_vilvl_b(__lsx_vslti_b(_pB1, 0), _pB1);
                    __m128i _pA0 = __lsx_vreplgr2vr_h(pA[0]);
                    __m128i _s = __lsx_vmul_h(_pA0, _pB0);
                    _sum00 = __lasx_xvadd_w(_sum00, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(_pA0, _pB1);
                    _sum01 = __lasx_xvadd_w(_sum01, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    pB0 += 8;
                    pB1 += 8;
                    pA++;
                }
                __m256 _bscale0 = (__m256)__lasx_xvld(pB_descales0, 0);
                __m256 _bscale1 = (__m256)__lasx_xvld(pB_descales1, 0);
                __m256 _ascale = (__m256)__lasx_xvreplfr2vr_s(*pA_descales++);
                _out00 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum00), __lasx_xvfmul_s(_bscale0, _ascale), _out00);
                _out01 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum01), __lasx_xvfmul_s(_bscale1, _ascale), _out01);
                pB_descales0 += 8;
                pB_descales1 += 8;
            }
            pB = pB1 + 8 * (full_K - k0 - max_kk0);
            pB_descales = pB_descales1 + 8 * (block_count - block_start - tile_blocks);
            __lasx_xvst(_out00, outptr, 0);
            __lasx_xvst(_out01, outptr + 8, 0);
            outptr += 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            pB += (size_t)8 * k0;
            pB_descales += (size_t)8 * block_start;
            __m256 _out0 = k0 == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr, 0);
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                __m256i _sum0 = __lasx_xvreplgr2vr_w(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m256i _pB = __lasx_xvld(pB, 0);
                    __m256i _pA0 = __lasx_xvldrepl_w(pA, 0);
                    __m256i _s0 = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA0, _pB), _pA0, _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s0, _s0));
                    pB += 32;
                    pA += 4;
                }
                if (kk + 1 < max_kk)
                {
                    __m128i _pB = __lsx_vld(pB, 0);
                    __m128i _pA0 = __lsx_vreplgr2vr_h((unsigned char)pA[0] | ((unsigned char)pA[1] << 8));
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB), _pA0, _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(__lasx_cast_128(_s0)));
                    pB += 16;
                    pA += 2;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    __m128i _pB = __lsx_vldrepl_d(pB, 0);
                    _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);
                    __m128i _s0 = __lsx_vmul_h(__lsx_vreplgr2vr_h(pA[0]), _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(__lasx_cast_128(_s0)));
                    pB += 8;
                    pA++;
                }
                __m256 _bscale = (__m256)__lasx_xvld(pB_descales, 0);
                _out0 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum0), __lasx_xvfmul_s(_bscale, (__m256)__lasx_xvreplfr2vr_s(*pA_descales++)), _out0);
                pB_descales += 8;
            }
            __lasx_xvst(_out0, outptr, 0);
            outptr += 8;
            pB += (size_t)8 * (full_K - k0 - max_kk0);
            pB_descales += (size_t)8 * (block_count - block_start - tile_blocks);
        }
#endif
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB0 = pB + 4 * k0;
            const signed char* pB1 = pB + 4 * full_K + 4 * k0;
            const float* pB_descales0 = pB_descales + 4 * block_start;
            const float* pB_descales1 = pB_descales + 4 * block_count + 4 * block_start;
            __m128 _out00 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr, 0);
            __m128 _out01 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 4, 0);
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                __m128i _sum00 = __lsx_vreplgr2vr_w(0);
                __m128i _sum01 = __lsx_vreplgr2vr_w(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _pB0 = __lsx_vld(pB0, 0);
                    __m128i _pB1 = __lsx_vld(pB1, 0);
                    __m128i _pA0 = __lsx_vldrepl_w(pA, 0);
                    __m128i _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB0), _pA0, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB1), _pA0, _pB1);
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vhaddw_w_h(_s, _s));
                    pB0 += 16;
                    pB1 += 16;
                    pA += 4;
                }
                if (kk + 1 < max_kk)
                {
                    __m128i _pB0 = __lsx_vldrepl_d(pB0, 0);
                    __m128i _pB1 = __lsx_vldrepl_d(pB1, 0);
                    __m128i _pA0 = __lsx_vreplgr2vr_h((unsigned char)pA[0] | ((unsigned char)pA[1] << 8));
                    __m128i _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB0), _pA0, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB1), _pA0, _pB1);
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    pB0 += 8;
                    pB1 += 8;
                    pA += 2;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    __m128i _pB0 = __lsx_vldrepl_w(pB0, 0);
                    __m128i _pB1 = __lsx_vldrepl_w(pB1, 0);
                    _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                    _pB1 = __lsx_vilvl_b(__lsx_vslti_b(_pB1, 0), _pB1);
                    __m128i _pA0 = __lsx_vreplgr2vr_h(pA[0]);
                    __m128i _s = __lsx_vmul_h(_pA0, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA0, _pB1);
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    pB0 += 4;
                    pB1 += 4;
                    pA++;
                }
                __m128 _bscale0 = (__m128)__lsx_vld(pB_descales0, 0);
                __m128 _bscale1 = (__m128)__lsx_vld(pB_descales1, 0);
                __m128 _ascale = __lsx_vreplfr2vr_s(*pA_descales++);
                _out00 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum00), __lsx_vfmul_s(_bscale0, _ascale), _out00);
                _out01 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum01), __lsx_vfmul_s(_bscale1, _ascale), _out01);
                pB_descales0 += 4;
                pB_descales1 += 4;
            }
            pB = pB1 + 4 * (full_K - k0 - max_kk0);
            pB_descales = pB_descales1 + 4 * (block_count - block_start - tile_blocks);
            __lsx_vst((__m128i)_out00, outptr, 0);
            __lsx_vst((__m128i)_out01, outptr + 4, 0);
            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            pB += (size_t)4 * k0;
            pB_descales += (size_t)4 * block_start;
            __m128 _out0 = k0 == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr, 0);
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128i _pB = __lsx_vld(pB, 0);
                    __m128i _pA0 = __lsx_vldrepl_w(pA, 0);
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB), _pA0, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                    pB += 16;
                    pA += 4;
                }
                if (kk + 1 < max_kk)
                {
                    __m128i _pB = __lsx_vldrepl_d(pB, 0);
                    __m128i _pA0 = __lsx_vreplgr2vr_h((unsigned char)pA[0] | ((unsigned char)pA[1] << 8));
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB), _pA0, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    pB += 8;
                    pA += 2;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    __m128i _pB = __lsx_vldrepl_w(pB, 0);
                    _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);
                    __m128i _s0 = __lsx_vmul_h(__lsx_vreplgr2vr_h(pA[0]), _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    pB += 4;
                    pA++;
                }
                __m128 _bscale = (__m128)__lsx_vld(pB_descales, 0);
                _out0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum0), __lsx_vfmul_s(_bscale, __lsx_vreplfr2vr_s(*pA_descales++)), _out0);
                pB_descales += 4;
            }
            __lsx_vst((__m128i)_out0, outptr, 0);
            outptr += 4;
            pB += (size_t)4 * (full_K - k0 - max_kk0);
            pB_descales += (size_t)4 * (block_count - block_start - tile_blocks);
        }
#endif
        for (; jj + 1 < max_jj; jj += 2)
        {
            pB += (size_t)2 * k0;
            pB_descales += (size_t)2 * block_start;
            float _out0 = k0 == 0 ? 0.f : outptr[0];
            float _out1 = k0 == 0 ? 0.f : outptr[1];
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                int _sum0 = 0;
                int _sum1 = 0;
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    asm volatile(""
                                 :
                                 :
                                 : "memory");

                    _sum0 += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    _sum1 += pA[0] * pB[4] + pA[1] * pB[5] + pA[2] * pB[6] + pA[3] * pB[7];
                    pB += 8;
                    pA += 4;
                }
                if (kk + 1 < max_kk)
                {
                    _sum0 += pA[0] * pB[0] + pA[1] * pB[1];
                    _sum1 += pA[0] * pB[2] + pA[1] * pB[3];
                    pB += 4;
                    pA += 2;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    _sum0 += pA[0] * pB[0];
                    _sum1 += pA[0] * pB[1];
                    pB += 2;
                    pA++;
                }
                const float ascale = *pA_descales++;
                _out0 += _sum0 * ascale * pB_descales[0];
                _out1 += _sum1 * ascale * pB_descales[1];
                pB_descales += 2;
            }
            outptr[0] = _out0;
            outptr[1] = _out1;
            outptr += 2;
            pB += (size_t)2 * (full_K - k0 - max_kk0);
            pB_descales += (size_t)2 * (block_count - block_start - tile_blocks);
        }
        for (; jj < max_jj; jj++)
        {
            pB += k0;
            pB_descales += block_start;
            float _out0 = k0 == 0 ? 0.f : outptr[0];
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int k = 0; k < K; k += block_size)
            {
                int _sum0 = 0;
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    _sum0 += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    pB += 4;
                    pA += 4;
                }
                if (kk + 1 < max_kk)
                {
                    _sum0 += pA[0] * pB[0] + pA[1] * pB[1];
                    pB += 2;
                    pA += 2;
                    kk += 2;
                }
                if (kk < max_kk)
                {
                    _sum0 += pA[0] * pB[0];
                    pB++;
                    pA++;
                }
                _out0 += _sum0 * *pA_descales++ * *pB_descales++;
            }
            *outptr++ = _out0;
            pB += full_K - k0 - max_kk0;
            pB_descales += block_count - block_start - tile_blocks;
        }
        pAT += A_hstep;
        pAT_descales += A_descales_hstep;
    }
}

static void unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int N, float alpha, float beta)
{
    const float* pp = topT;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const float* pC_base = C;
    float* outptr = top_blob;
    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* p0 = outptr + (size_t)(i + ii) * N + j;
        float* p1 = p0 + N;
        float* p2 = p1 + N;
        float* p3 = p2 + N;
        float* p4 = p3 + N;
        float* p5 = p4 + N;
        float* p6 = p5 + N;
        float* p7 = p6 + N;

        float c0 = 0.f;
        float c1 = 0.f;
        float c2 = 0.f;
        float c3 = 0.f;
        float c4 = 0.f;
        float c5 = 0.f;
        float c6 = 0.f;
        float c7 = 0.f;
        const float* pC = pC_base;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
                c1 = c0;
                c2 = c0;
                c3 = c0;
                c4 = c0;
                c5 = c0;
                c6 = c0;
                c7 = c0;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[i + ii] * beta;
                c1 = pC[i + ii + 1] * beta;
                c2 = pC[i + ii + 2] * beta;
                c3 = pC[i + ii + 3] * beta;
                c4 = pC[i + ii + 4] * beta;
                c5 = pC[i + ii + 5] * beta;
                c6 = pC[i + ii + 6] * beta;
                c7 = pC[i + ii + 7] * beta;
            }
            if (broadcast_type_C == 3)
            {
                pC += (size_t)(i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
                pC += j;
        }

        int jj = 0;
#if __loongarch_asx
        __m256 _alpha256 = (__m256)__lasx_xvreplfr2vr_s(alpha);
        __m256 _beta256 = (__m256)__lasx_xvreplfr2vr_s(beta);
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256i _sum0 = __lasx_xvld(pp, 0);
            __m256i _sum1 = __lasx_xvld(pp + 8, 0);
            __m256i _sum2 = __lasx_xvld(pp + 16, 0);
            __m256i _sum3 = __lasx_xvld(pp + 24, 0);
            __m256i _sum4 = __lasx_xvld(pp + 32, 0);
            __m256i _sum5 = __lasx_xvld(pp + 40, 0);
            __m256i _sum6 = __lasx_xvld(pp + 48, 0);
            __m256i _sum7 = __lasx_xvld(pp + 56, 0);
            __m256i _tmp0 = _sum0;
            __m256i _tmp1 = __lasx_xvshuf4i_w(_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
            __m256i _tmp2 = _sum2;
            __m256i _tmp3 = __lasx_xvshuf4i_w(_sum3, _LSX_SHUFFLE(2, 1, 0, 3));
            __m256i _tmp4 = _sum4;
            __m256i _tmp5 = __lasx_xvshuf4i_w(_sum5, _LSX_SHUFFLE(2, 1, 0, 3));
            __m256i _tmp6 = _sum6;
            __m256i _tmp7 = __lasx_xvshuf4i_w(_sum7, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum0 = __lasx_xvilvl_w(_tmp3, _tmp0);
            _sum1 = __lasx_xvilvh_w(_tmp3, _tmp0);
            _sum2 = __lasx_xvilvl_w(_tmp1, _tmp2);
            _sum3 = __lasx_xvilvh_w(_tmp1, _tmp2);
            _sum4 = __lasx_xvilvl_w(_tmp7, _tmp4);
            _sum5 = __lasx_xvilvh_w(_tmp7, _tmp4);
            _sum6 = __lasx_xvilvl_w(_tmp5, _tmp6);
            _sum7 = __lasx_xvilvh_w(_tmp5, _tmp6);
            _tmp0 = __lasx_xvilvl_d(_sum2, _sum0);
            _tmp1 = __lasx_xvilvh_d(_sum2, _sum0);
            _tmp2 = __lasx_xvilvl_d(_sum1, _sum3);
            _tmp3 = __lasx_xvilvh_d(_sum1, _sum3);
            _tmp4 = __lasx_xvilvl_d(_sum6, _sum4);
            _tmp5 = __lasx_xvilvh_d(_sum6, _sum4);
            _tmp6 = __lasx_xvilvl_d(_sum5, _sum7);
            _tmp7 = __lasx_xvilvh_d(_sum5, _sum7);
            _tmp1 = __lasx_xvshuf4i_w(_tmp1, _LSX_SHUFFLE(2, 1, 0, 3));
            _tmp3 = __lasx_xvshuf4i_w(_tmp3, _LSX_SHUFFLE(2, 1, 0, 3));
            _tmp5 = __lasx_xvshuf4i_w(_tmp5, _LSX_SHUFFLE(2, 1, 0, 3));
            _tmp7 = __lasx_xvshuf4i_w(_tmp7, _LSX_SHUFFLE(2, 1, 0, 3));
            __m256 _f0 = (__m256)__lasx_xvpermi_q(_tmp4, _tmp0, _LSX_SHUFFLE(0, 3, 0, 0));
            __m256 _f1 = (__m256)__lasx_xvpermi_q(_tmp5, _tmp1, _LSX_SHUFFLE(0, 3, 0, 0));
            __m256 _f2 = (__m256)__lasx_xvpermi_q(_tmp6, _tmp2, _LSX_SHUFFLE(0, 3, 0, 0));
            __m256 _f3 = (__m256)__lasx_xvpermi_q(_tmp7, _tmp3, _LSX_SHUFFLE(0, 3, 0, 0));
            __m256 _f4 = (__m256)__lasx_xvpermi_q(_tmp0, _tmp4, _LSX_SHUFFLE(0, 3, 0, 0));
            __m256 _f5 = (__m256)__lasx_xvpermi_q(_tmp1, _tmp5, _LSX_SHUFFLE(0, 3, 0, 0));
            __m256 _f6 = (__m256)__lasx_xvpermi_q(_tmp2, _tmp6, _LSX_SHUFFLE(0, 3, 0, 0));
            __m256 _f7 = (__m256)__lasx_xvpermi_q(_tmp3, _tmp7, _LSX_SHUFFLE(0, 3, 0, 0));
            pp += 64;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m256 _c = (__m256)__lasx_xvreplfr2vr_s(c0);
                    _f0 = __lasx_xvfadd_s(_f0, _c);
                    _f1 = __lasx_xvfadd_s(_f1, _c);
                    _f2 = __lasx_xvfadd_s(_f2, _c);
                    _f3 = __lasx_xvfadd_s(_f3, _c);
                    _f4 = __lasx_xvfadd_s(_f4, _c);
                    _f5 = __lasx_xvfadd_s(_f5, _c);
                    _f6 = __lasx_xvfadd_s(_f6, _c);
                    _f7 = __lasx_xvfadd_s(_f7, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lasx_xvfadd_s(_f0, (__m256)__lasx_xvreplfr2vr_s(c0));
                    _f1 = __lasx_xvfadd_s(_f1, (__m256)__lasx_xvreplfr2vr_s(c1));
                    _f2 = __lasx_xvfadd_s(_f2, (__m256)__lasx_xvreplfr2vr_s(c2));
                    _f3 = __lasx_xvfadd_s(_f3, (__m256)__lasx_xvreplfr2vr_s(c3));
                    _f4 = __lasx_xvfadd_s(_f4, (__m256)__lasx_xvreplfr2vr_s(c4));
                    _f5 = __lasx_xvfadd_s(_f5, (__m256)__lasx_xvreplfr2vr_s(c5));
                    _f6 = __lasx_xvfadd_s(_f6, (__m256)__lasx_xvreplfr2vr_s(c6));
                    _f7 = __lasx_xvfadd_s(_f7, (__m256)__lasx_xvreplfr2vr_s(c7));
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c0 = (__m256)__lasx_xvld(pC, 0);
                    __m256 _c1 = (__m256)__lasx_xvld(pC + c_hstep, 0);
                    __m256 _c2 = (__m256)__lasx_xvld(pC + c_hstep * 2, 0);
                    __m256 _c3 = (__m256)__lasx_xvld(pC + c_hstep * 3, 0);
                    __m256 _c4 = (__m256)__lasx_xvld(pC + c_hstep * 4, 0);
                    __m256 _c5 = (__m256)__lasx_xvld(pC + c_hstep * 5, 0);
                    __m256 _c6 = (__m256)__lasx_xvld(pC + c_hstep * 6, 0);
                    __m256 _c7 = (__m256)__lasx_xvld(pC + c_hstep * 7, 0);
                    if (beta == 1.f)
                    {
                        _f0 = __lasx_xvfadd_s(_f0, _c0);
                        _f1 = __lasx_xvfadd_s(_f1, _c1);
                        _f2 = __lasx_xvfadd_s(_f2, _c2);
                        _f3 = __lasx_xvfadd_s(_f3, _c3);
                        _f4 = __lasx_xvfadd_s(_f4, _c4);
                        _f5 = __lasx_xvfadd_s(_f5, _c5);
                        _f6 = __lasx_xvfadd_s(_f6, _c6);
                        _f7 = __lasx_xvfadd_s(_f7, _c7);
                    }
                    else
                    {
                        _f0 = __lasx_xvfmadd_s(_c0, _beta256, _f0);
                        _f1 = __lasx_xvfmadd_s(_c1, _beta256, _f1);
                        _f2 = __lasx_xvfmadd_s(_c2, _beta256, _f2);
                        _f3 = __lasx_xvfmadd_s(_c3, _beta256, _f3);
                        _f4 = __lasx_xvfmadd_s(_c4, _beta256, _f4);
                        _f5 = __lasx_xvfmadd_s(_c5, _beta256, _f5);
                        _f6 = __lasx_xvfmadd_s(_c6, _beta256, _f6);
                        _f7 = __lasx_xvfmadd_s(_c7, _beta256, _f7);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c = (__m256)__lasx_xvld(pC, 0);
                    if (beta != 1.f)
                        _c = __lasx_xvfmul_s(_c, _beta256);
                    _f0 = __lasx_xvfadd_s(_f0, _c);
                    _f1 = __lasx_xvfadd_s(_f1, _c);
                    _f2 = __lasx_xvfadd_s(_f2, _c);
                    _f3 = __lasx_xvfadd_s(_f3, _c);
                    _f4 = __lasx_xvfadd_s(_f4, _c);
                    _f5 = __lasx_xvfadd_s(_f5, _c);
                    _f6 = __lasx_xvfadd_s(_f6, _c);
                    _f7 = __lasx_xvfadd_s(_f7, _c);
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lasx_xvfmul_s(_f0, _alpha256);
                _f1 = __lasx_xvfmul_s(_f1, _alpha256);
                _f2 = __lasx_xvfmul_s(_f2, _alpha256);
                _f3 = __lasx_xvfmul_s(_f3, _alpha256);
                _f4 = __lasx_xvfmul_s(_f4, _alpha256);
                _f5 = __lasx_xvfmul_s(_f5, _alpha256);
                _f6 = __lasx_xvfmul_s(_f6, _alpha256);
                _f7 = __lasx_xvfmul_s(_f7, _alpha256);
            }
            __lasx_xvst(_f0, p0, 0);
            __lasx_xvst(_f1, p1, 0);
            __lasx_xvst(_f2, p2, 0);
            __lasx_xvst(_f3, p3, 0);
            __lasx_xvst(_f4, p4, 0);
            __lasx_xvst(_f5, p5, 0);
            __lasx_xvst(_f6, p6, 0);
            __lasx_xvst(_f7, p7, 0);
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
            p4 += 8;
            p5 += 8;
            p6 += 8;
            p7 += 8;
        }
#endif // __loongarch_asx
        __m128 _alpha128 = __lsx_vreplfr2vr_s(alpha);
        __m128 _beta128 = __lsx_vreplfr2vr_s(beta);
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _sum0 = __lsx_vld(pp, 0);
            __m128i _sum1 = __lsx_vld(pp + 8, 0);
            __m128i _sum2 = __lsx_vld(pp + 16, 0);
            __m128i _sum3 = __lsx_vld(pp + 24, 0);
            _sum2 = __lsx_vshuf4i_w(_sum2, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum3 = __lsx_vshuf4i_w(_sum3, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
            _sum1 = __lsx_vshuf4i_w(_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum2 = __lsx_vshuf4i_w(_sum2, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum3 = __lsx_vshuf4i_w(_sum3, _LSX_SHUFFLE(0, 3, 2, 1));
            __m128i _sum4 = __lsx_vld(pp + 4, 0);
            __m128i _sum5 = __lsx_vld(pp + 12, 0);
            __m128i _sum6 = __lsx_vld(pp + 20, 0);
            __m128i _sum7 = __lsx_vld(pp + 28, 0);
            _sum6 = __lsx_vshuf4i_w(_sum6, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum7 = __lsx_vshuf4i_w(_sum7, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum4, _sum5, _sum6, _sum7);
            _sum5 = __lsx_vshuf4i_w(_sum5, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum6 = __lsx_vshuf4i_w(_sum6, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum7 = __lsx_vshuf4i_w(_sum7, _LSX_SHUFFLE(0, 3, 2, 1));
            __m128 _f0 = (__m128)_sum0;
            __m128 _f1 = (__m128)_sum1;
            __m128 _f2 = (__m128)_sum2;
            __m128 _f3 = (__m128)_sum3;
            __m128 _f4 = (__m128)_sum4;
            __m128 _f5 = (__m128)_sum5;
            __m128 _f6 = (__m128)_sum6;
            __m128 _f7 = (__m128)_sum7;
            pp += 32;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = __lsx_vreplfr2vr_s(c0);
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                    _f2 = __lsx_vfadd_s(_f2, _c);
                    _f3 = __lsx_vfadd_s(_f3, _c);
                    _f4 = __lsx_vfadd_s(_f4, _c);
                    _f5 = __lsx_vfadd_s(_f5, _c);
                    _f6 = __lsx_vfadd_s(_f6, _c);
                    _f7 = __lsx_vfadd_s(_f7, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(c0));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vreplfr2vr_s(c1));
                    _f2 = __lsx_vfadd_s(_f2, __lsx_vreplfr2vr_s(c2));
                    _f3 = __lsx_vfadd_s(_f3, __lsx_vreplfr2vr_s(c3));
                    _f4 = __lsx_vfadd_s(_f4, __lsx_vreplfr2vr_s(c4));
                    _f5 = __lsx_vfadd_s(_f5, __lsx_vreplfr2vr_s(c5));
                    _f6 = __lsx_vfadd_s(_f6, __lsx_vreplfr2vr_s(c6));
                    _f7 = __lsx_vfadd_s(_f7, __lsx_vreplfr2vr_s(c7));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = (__m128)__lsx_vld(pC, 0);
                    __m128 _c1 = (__m128)__lsx_vld(pC + c_hstep, 0);
                    __m128 _c2 = (__m128)__lsx_vld(pC + c_hstep * 2, 0);
                    __m128 _c3 = (__m128)__lsx_vld(pC + c_hstep * 3, 0);
                    __m128 _c4 = (__m128)__lsx_vld(pC + c_hstep * 4, 0);
                    __m128 _c5 = (__m128)__lsx_vld(pC + c_hstep * 5, 0);
                    __m128 _c6 = (__m128)__lsx_vld(pC + c_hstep * 6, 0);
                    __m128 _c7 = (__m128)__lsx_vld(pC + c_hstep * 7, 0);
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                        _f1 = __lsx_vfadd_s(_f1, _c1);
                        _f2 = __lsx_vfadd_s(_f2, _c2);
                        _f3 = __lsx_vfadd_s(_f3, _c3);
                        _f4 = __lsx_vfadd_s(_f4, _c4);
                        _f5 = __lsx_vfadd_s(_f5, _c5);
                        _f6 = __lsx_vfadd_s(_f6, _c6);
                        _f7 = __lsx_vfadd_s(_f7, _c7);
                    }
                    else
                    {
                        _f0 = __lsx_vfmadd_s(_c0, _beta128, _f0);
                        _f1 = __lsx_vfmadd_s(_c1, _beta128, _f1);
                        _f2 = __lsx_vfmadd_s(_c2, _beta128, _f2);
                        _f3 = __lsx_vfmadd_s(_c3, _beta128, _f3);
                        _f4 = __lsx_vfmadd_s(_c4, _beta128, _f4);
                        _f5 = __lsx_vfmadd_s(_c5, _beta128, _f5);
                        _f6 = __lsx_vfmadd_s(_c6, _beta128, _f6);
                        _f7 = __lsx_vfmadd_s(_c7, _beta128, _f7);
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = (__m128)__lsx_vld(pC, 0);
                    if (beta != 1.f)
                        _c = __lsx_vfmul_s(_c, _beta128);
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                    _f2 = __lsx_vfadd_s(_f2, _c);
                    _f3 = __lsx_vfadd_s(_f3, _c);
                    _f4 = __lsx_vfadd_s(_f4, _c);
                    _f5 = __lsx_vfadd_s(_f5, _c);
                    _f6 = __lsx_vfadd_s(_f6, _c);
                    _f7 = __lsx_vfadd_s(_f7, _c);
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lsx_vfmul_s(_f0, _alpha128);
                _f1 = __lsx_vfmul_s(_f1, _alpha128);
                _f2 = __lsx_vfmul_s(_f2, _alpha128);
                _f3 = __lsx_vfmul_s(_f3, _alpha128);
                _f4 = __lsx_vfmul_s(_f4, _alpha128);
                _f5 = __lsx_vfmul_s(_f5, _alpha128);
                _f6 = __lsx_vfmul_s(_f6, _alpha128);
                _f7 = __lsx_vfmul_s(_f7, _alpha128);
            }
            __lsx_vst((__m128i)_f0, p0, 0);
            __lsx_vst((__m128i)_f1, p1, 0);
            __lsx_vst((__m128i)_f2, p2, 0);
            __lsx_vst((__m128i)_f3, p3, 0);
            __lsx_vst((__m128i)_f4, p4, 0);
            __lsx_vst((__m128i)_f5, p5, 0);
            __lsx_vst((__m128i)_f6, p6, 0);
            __lsx_vst((__m128i)_f7, p7, 0);
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
            p4 += 4;
            p5 += 4;
            p6 += 4;
            p7 += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128i _fi0 = __lsx_vldrepl_w(pp, 0);
            __m128i _fi1 = __lsx_vldrepl_w(pp + 1, 0);
            __m128i _fi2 = __lsx_vldrepl_w(pp + 2, 0);
            __m128i _fi3 = __lsx_vldrepl_w(pp + 3, 0);
            __m128i _fi4 = __lsx_vldrepl_w(pp + 4, 0);
            __m128i _fi5 = __lsx_vldrepl_w(pp + 5, 0);
            __m128i _fi6 = __lsx_vldrepl_w(pp + 6, 0);
            __m128i _fi7 = __lsx_vldrepl_w(pp + 7, 0);
            _fi0 = __lsx_vinsgr2vr_w(_fi0, ((const int*)pp)[8], 1);
            _fi1 = __lsx_vinsgr2vr_w(_fi1, ((const int*)pp)[9], 1);
            _fi2 = __lsx_vinsgr2vr_w(_fi2, ((const int*)pp)[10], 1);
            _fi3 = __lsx_vinsgr2vr_w(_fi3, ((const int*)pp)[11], 1);
            _fi4 = __lsx_vinsgr2vr_w(_fi4, ((const int*)pp)[12], 1);
            _fi5 = __lsx_vinsgr2vr_w(_fi5, ((const int*)pp)[13], 1);
            _fi6 = __lsx_vinsgr2vr_w(_fi6, ((const int*)pp)[14], 1);
            _fi7 = __lsx_vinsgr2vr_w(_fi7, ((const int*)pp)[15], 1);
            __m128 _f0 = (__m128)_fi0;
            __m128 _f1 = (__m128)_fi1;
            __m128 _f2 = (__m128)_fi2;
            __m128 _f3 = (__m128)_fi3;
            __m128 _f4 = (__m128)_fi4;
            __m128 _f5 = (__m128)_fi5;
            __m128 _f6 = (__m128)_fi6;
            __m128 _f7 = (__m128)_fi7;
            pp += 16;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = __lsx_vreplfr2vr_s(c0);
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                    _f2 = __lsx_vfadd_s(_f2, _c);
                    _f3 = __lsx_vfadd_s(_f3, _c);
                    _f4 = __lsx_vfadd_s(_f4, _c);
                    _f5 = __lsx_vfadd_s(_f5, _c);
                    _f6 = __lsx_vfadd_s(_f6, _c);
                    _f7 = __lsx_vfadd_s(_f7, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(c0));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vreplfr2vr_s(c1));
                    _f2 = __lsx_vfadd_s(_f2, __lsx_vreplfr2vr_s(c2));
                    _f3 = __lsx_vfadd_s(_f3, __lsx_vreplfr2vr_s(c3));
                    _f4 = __lsx_vfadd_s(_f4, __lsx_vreplfr2vr_s(c4));
                    _f5 = __lsx_vfadd_s(_f5, __lsx_vreplfr2vr_s(c5));
                    _f6 = __lsx_vfadd_s(_f6, __lsx_vreplfr2vr_s(c6));
                    _f7 = __lsx_vfadd_s(_f7, __lsx_vreplfr2vr_s(c7));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = (__m128)__lsx_vldrepl_d(pC, 0);
                    __m128 _c1 = (__m128)__lsx_vldrepl_d(pC + c_hstep, 0);
                    __m128 _c2 = (__m128)__lsx_vldrepl_d(pC + c_hstep * 2, 0);
                    __m128 _c3 = (__m128)__lsx_vldrepl_d(pC + c_hstep * 3, 0);
                    __m128 _c4 = (__m128)__lsx_vldrepl_d(pC + c_hstep * 4, 0);
                    __m128 _c5 = (__m128)__lsx_vldrepl_d(pC + c_hstep * 5, 0);
                    __m128 _c6 = (__m128)__lsx_vldrepl_d(pC + c_hstep * 6, 0);
                    __m128 _c7 = (__m128)__lsx_vldrepl_d(pC + c_hstep * 7, 0);
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                        _f1 = __lsx_vfadd_s(_f1, _c1);
                        _f2 = __lsx_vfadd_s(_f2, _c2);
                        _f3 = __lsx_vfadd_s(_f3, _c3);
                        _f4 = __lsx_vfadd_s(_f4, _c4);
                        _f5 = __lsx_vfadd_s(_f5, _c5);
                        _f6 = __lsx_vfadd_s(_f6, _c6);
                        _f7 = __lsx_vfadd_s(_f7, _c7);
                    }
                    else
                    {
                        _f0 = __lsx_vfmadd_s(_c0, _beta128, _f0);
                        _f1 = __lsx_vfmadd_s(_c1, _beta128, _f1);
                        _f2 = __lsx_vfmadd_s(_c2, _beta128, _f2);
                        _f3 = __lsx_vfmadd_s(_c3, _beta128, _f3);
                        _f4 = __lsx_vfmadd_s(_c4, _beta128, _f4);
                        _f5 = __lsx_vfmadd_s(_c5, _beta128, _f5);
                        _f6 = __lsx_vfmadd_s(_c6, _beta128, _f6);
                        _f7 = __lsx_vfmadd_s(_c7, _beta128, _f7);
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = (__m128)__lsx_vldrepl_d(pC, 0);
                    if (beta != 1.f)
                        _c = __lsx_vfmul_s(_c, _beta128);
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                    _f2 = __lsx_vfadd_s(_f2, _c);
                    _f3 = __lsx_vfadd_s(_f3, _c);
                    _f4 = __lsx_vfadd_s(_f4, _c);
                    _f5 = __lsx_vfadd_s(_f5, _c);
                    _f6 = __lsx_vfadd_s(_f6, _c);
                    _f7 = __lsx_vfadd_s(_f7, _c);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lsx_vfmul_s(_f0, _alpha128);
                _f1 = __lsx_vfmul_s(_f1, _alpha128);
                _f2 = __lsx_vfmul_s(_f2, _alpha128);
                _f3 = __lsx_vfmul_s(_f3, _alpha128);
                _f4 = __lsx_vfmul_s(_f4, _alpha128);
                _f5 = __lsx_vfmul_s(_f5, _alpha128);
                _f6 = __lsx_vfmul_s(_f6, _alpha128);
                _f7 = __lsx_vfmul_s(_f7, _alpha128);
            }
            __lsx_vstelm_d((__m128i)_f0, p0, 0, 0);
            __lsx_vstelm_d((__m128i)_f1, p1, 0, 0);
            __lsx_vstelm_d((__m128i)_f2, p2, 0, 0);
            __lsx_vstelm_d((__m128i)_f3, p3, 0, 0);
            __lsx_vstelm_d((__m128i)_f4, p4, 0, 0);
            __lsx_vstelm_d((__m128i)_f5, p5, 0, 0);
            __lsx_vstelm_d((__m128i)_f6, p6, 0, 0);
            __lsx_vstelm_d((__m128i)_f7, p7, 0, 0);
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
            p4 += 2;
            p5 += 2;
            p6 += 2;
            p7 += 2;
        }
        for (; jj < max_jj; jj++)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp, 0);
            __m128 _f4 = (__m128)__lsx_vld(pp + 4, 0);
            pp += 8;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = __lsx_vreplfr2vr_s(c0);
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f4 = __lsx_vfadd_s(_f4, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vfmul_s((__m128)__lsx_vld(pC_base + i + ii, 0), _beta128));
                    _f4 = __lsx_vfadd_s(_f4, __lsx_vfmul_s((__m128)__lsx_vld(pC_base + i + ii + 4, 0), _beta128));
                }
                if (broadcast_type_C == 3)
                {
                    __m128i _c0 = __lsx_vreplgr2vr_w(((const int*)pC)[0]);
                    _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep))[0], 1);
                    _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep * 2))[0], 2);
                    _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep * 3))[0], 3);
                    __m128i _c4 = __lsx_vreplgr2vr_w(((const int*)(pC + c_hstep * 4))[0]);
                    _c4 = __lsx_vinsgr2vr_w(_c4, ((const int*)(pC + c_hstep * 5))[0], 1);
                    _c4 = __lsx_vinsgr2vr_w(_c4, ((const int*)(pC + c_hstep * 6))[0], 2);
                    _c4 = __lsx_vinsgr2vr_w(_c4, ((const int*)(pC + c_hstep * 7))[0], 3);
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, (__m128)_c0);
                        _f4 = __lsx_vfadd_s(_f4, (__m128)_c4);
                    }
                    else
                    {
                        _f0 = __lsx_vfmadd_s((__m128)_c0, _beta128, _f0);
                        _f4 = __lsx_vfmadd_s((__m128)_c4, _beta128, _f4);
                    }
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = __lsx_vreplfr2vr_s(beta == 1.f ? pC[0] : pC[0] * beta);
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f4 = __lsx_vfadd_s(_f4, _c);
                    pC++;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lsx_vfmul_s(_f0, _alpha128);
                _f4 = __lsx_vfmul_s(_f4, _alpha128);
            }
            __lsx_vstelm_w((__m128i)_f0, p0, 0, 0);
            __lsx_vstelm_w((__m128i)_f0, p1, 0, 1);
            __lsx_vstelm_w((__m128i)_f0, p2, 0, 2);
            __lsx_vstelm_w((__m128i)_f0, p3, 0, 3);
            __lsx_vstelm_w((__m128i)_f4, p4, 0, 0);
            __lsx_vstelm_w((__m128i)_f4, p5, 0, 1);
            __lsx_vstelm_w((__m128i)_f4, p6, 0, 2);
            __lsx_vstelm_w((__m128i)_f4, p7, 0, 3);
            p0++;
            p1++;
            p2++;
            p3++;
            p4++;
            p5++;
            p6++;
            p7++;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* p0 = outptr + (size_t)(i + ii) * N + j;
        float* p1 = p0 + N;
        float* p2 = p1 + N;
        float* p3 = p2 + N;

        float c0 = 0.f;
        float c1 = 0.f;
        float c2 = 0.f;
        float c3 = 0.f;
        const float* pC = pC_base;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
                c1 = c0;
                c2 = c0;
                c3 = c0;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[i + ii] * beta;
                c1 = pC[i + ii + 1] * beta;
                c2 = pC[i + ii + 2] * beta;
                c3 = pC[i + ii + 3] * beta;
            }
            if (broadcast_type_C == 3)
            {
                pC += (size_t)(i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC += j;
            }
        }

        int jj = 0;
#if __loongarch_asx
        __m256 _alpha256 = (__m256)__lasx_xvreplfr2vr_s(alpha);
        __m256 _beta256 = (__m256)__lasx_xvreplfr2vr_s(beta);
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _f00 = (__m256)__lasx_xvld(pp, 0);
            __m256 _f01 = (__m256)__lasx_xvld(pp + 8, 0);
            __m256 _f10 = (__m256)__lasx_xvld(pp + 16, 0);
            __m256 _f11 = (__m256)__lasx_xvld(pp + 24, 0);
            __m256 _f20 = (__m256)__lasx_xvld(pp + 32, 0);
            __m256 _f21 = (__m256)__lasx_xvld(pp + 40, 0);
            __m256 _f30 = (__m256)__lasx_xvld(pp + 48, 0);
            __m256 _f31 = (__m256)__lasx_xvld(pp + 56, 0);
            pp += 64;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m256 _c = (__m256)__lasx_xvreplfr2vr_s(c0);
                    _f00 = __lasx_xvfadd_s(_f00, _c);
                    _f01 = __lasx_xvfadd_s(_f01, _c);
                    _f10 = __lasx_xvfadd_s(_f10, _c);
                    _f11 = __lasx_xvfadd_s(_f11, _c);
                    _f20 = __lasx_xvfadd_s(_f20, _c);
                    _f21 = __lasx_xvfadd_s(_f21, _c);
                    _f30 = __lasx_xvfadd_s(_f30, _c);
                    _f31 = __lasx_xvfadd_s(_f31, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m256 _c0 = (__m256)__lasx_xvreplfr2vr_s(c0);
                    __m256 _c1 = (__m256)__lasx_xvreplfr2vr_s(c1);
                    __m256 _c2 = (__m256)__lasx_xvreplfr2vr_s(c2);
                    __m256 _c3 = (__m256)__lasx_xvreplfr2vr_s(c3);
                    _f00 = __lasx_xvfadd_s(_f00, _c0);
                    _f01 = __lasx_xvfadd_s(_f01, _c0);
                    _f10 = __lasx_xvfadd_s(_f10, _c1);
                    _f11 = __lasx_xvfadd_s(_f11, _c1);
                    _f20 = __lasx_xvfadd_s(_f20, _c2);
                    _f21 = __lasx_xvfadd_s(_f21, _c2);
                    _f30 = __lasx_xvfadd_s(_f30, _c3);
                    _f31 = __lasx_xvfadd_s(_f31, _c3);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c00 = (__m256)__lasx_xvld(pC, 0);
                    __m256 _c01 = (__m256)__lasx_xvld(pC + 8, 0);
                    __m256 _c10 = (__m256)__lasx_xvld(pC + c_hstep, 0);
                    __m256 _c11 = (__m256)__lasx_xvld(pC + c_hstep + 8, 0);
                    __m256 _c20 = (__m256)__lasx_xvld(pC + c_hstep * 2, 0);
                    __m256 _c21 = (__m256)__lasx_xvld(pC + c_hstep * 2 + 8, 0);
                    __m256 _c30 = (__m256)__lasx_xvld(pC + c_hstep * 3, 0);
                    __m256 _c31 = (__m256)__lasx_xvld(pC + c_hstep * 3 + 8, 0);
                    if (beta == 1.f)
                    {
                        _f00 = __lasx_xvfadd_s(_f00, _c00);
                        _f01 = __lasx_xvfadd_s(_f01, _c01);
                        _f10 = __lasx_xvfadd_s(_f10, _c10);
                        _f11 = __lasx_xvfadd_s(_f11, _c11);
                        _f20 = __lasx_xvfadd_s(_f20, _c20);
                        _f21 = __lasx_xvfadd_s(_f21, _c21);
                        _f30 = __lasx_xvfadd_s(_f30, _c30);
                        _f31 = __lasx_xvfadd_s(_f31, _c31);
                    }
                    else
                    {
                        _f00 = __lasx_xvfmadd_s(_c00, _beta256, _f00);
                        _f01 = __lasx_xvfmadd_s(_c01, _beta256, _f01);
                        _f10 = __lasx_xvfmadd_s(_c10, _beta256, _f10);
                        _f11 = __lasx_xvfmadd_s(_c11, _beta256, _f11);
                        _f20 = __lasx_xvfmadd_s(_c20, _beta256, _f20);
                        _f21 = __lasx_xvfmadd_s(_c21, _beta256, _f21);
                        _f30 = __lasx_xvfmadd_s(_c30, _beta256, _f30);
                        _f31 = __lasx_xvfmadd_s(_c31, _beta256, _f31);
                    }
                    pC += 16;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c0 = (__m256)__lasx_xvld(pC, 0);
                    __m256 _c1 = (__m256)__lasx_xvld(pC + 8, 0);
                    if (beta != 1.f)
                    {
                        _c0 = __lasx_xvfmul_s(_c0, _beta256);
                        _c1 = __lasx_xvfmul_s(_c1, _beta256);
                    }
                    _f00 = __lasx_xvfadd_s(_f00, _c0);
                    _f01 = __lasx_xvfadd_s(_f01, _c1);
                    _f10 = __lasx_xvfadd_s(_f10, _c0);
                    _f11 = __lasx_xvfadd_s(_f11, _c1);
                    _f20 = __lasx_xvfadd_s(_f20, _c0);
                    _f21 = __lasx_xvfadd_s(_f21, _c1);
                    _f30 = __lasx_xvfadd_s(_f30, _c0);
                    _f31 = __lasx_xvfadd_s(_f31, _c1);
                    pC += 16;
                }
            }
            if (alpha != 1.f)
            {
                _f00 = __lasx_xvfmul_s(_f00, _alpha256);
                _f01 = __lasx_xvfmul_s(_f01, _alpha256);
                _f10 = __lasx_xvfmul_s(_f10, _alpha256);
                _f11 = __lasx_xvfmul_s(_f11, _alpha256);
                _f20 = __lasx_xvfmul_s(_f20, _alpha256);
                _f21 = __lasx_xvfmul_s(_f21, _alpha256);
                _f30 = __lasx_xvfmul_s(_f30, _alpha256);
                _f31 = __lasx_xvfmul_s(_f31, _alpha256);
            }
            __lasx_xvst(_f00, p0, 0);
            __lasx_xvst(_f01, p0 + 8, 0);
            __lasx_xvst(_f10, p1, 0);
            __lasx_xvst(_f11, p1 + 8, 0);
            __lasx_xvst(_f20, p2, 0);
            __lasx_xvst(_f21, p2 + 8, 0);
            __lasx_xvst(_f30, p3, 0);
            __lasx_xvst(_f31, p3 + 8, 0);
            p0 += 16;
            p1 += 16;
            p2 += 16;
            p3 += 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = (__m256)__lasx_xvld(pp, 0);
            __m256 _f1 = (__m256)__lasx_xvld(pp + 8, 0);
            __m256 _f2 = (__m256)__lasx_xvld(pp + 16, 0);
            __m256 _f3 = (__m256)__lasx_xvld(pp + 24, 0);
            pp += 32;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = __lasx_xvfadd_s(_f0, (__m256)__lasx_xvreplfr2vr_s(c0));
                    _f1 = __lasx_xvfadd_s(_f1, (__m256)__lasx_xvreplfr2vr_s(c0));
                    _f2 = __lasx_xvfadd_s(_f2, (__m256)__lasx_xvreplfr2vr_s(c0));
                    _f3 = __lasx_xvfadd_s(_f3, (__m256)__lasx_xvreplfr2vr_s(c0));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lasx_xvfadd_s(_f0, (__m256)__lasx_xvreplfr2vr_s(c0));
                    _f1 = __lasx_xvfadd_s(_f1, (__m256)__lasx_xvreplfr2vr_s(c1));
                    _f2 = __lasx_xvfadd_s(_f2, (__m256)__lasx_xvreplfr2vr_s(c2));
                    _f3 = __lasx_xvfadd_s(_f3, (__m256)__lasx_xvreplfr2vr_s(c3));
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c0 = (__m256)__lasx_xvld(pC, 0);
                    __m256 _c1 = (__m256)__lasx_xvld(pC + c_hstep, 0);
                    __m256 _c2 = (__m256)__lasx_xvld(pC + c_hstep * 2, 0);
                    __m256 _c3 = (__m256)__lasx_xvld(pC + c_hstep * 3, 0);
                    if (beta == 1.f)
                    {
                        _f0 = __lasx_xvfadd_s(_f0, _c0);
                        _f1 = __lasx_xvfadd_s(_f1, _c1);
                        _f2 = __lasx_xvfadd_s(_f2, _c2);
                        _f3 = __lasx_xvfadd_s(_f3, _c3);
                    }
                    else
                    {
                        _f0 = __lasx_xvfmadd_s(_c0, _beta256, _f0);
                        _f1 = __lasx_xvfmadd_s(_c1, _beta256, _f1);
                        _f2 = __lasx_xvfmadd_s(_c2, _beta256, _f2);
                        _f3 = __lasx_xvfmadd_s(_c3, _beta256, _f3);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c = (__m256)__lasx_xvld(pC, 0);
                    if (beta != 1.f)
                        _c = __lasx_xvfmul_s(_c, _beta256);
                    _f0 = __lasx_xvfadd_s(_f0, _c);
                    _f1 = __lasx_xvfadd_s(_f1, _c);
                    _f2 = __lasx_xvfadd_s(_f2, _c);
                    _f3 = __lasx_xvfadd_s(_f3, _c);
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lasx_xvfmul_s(_f0, _alpha256);
                _f1 = __lasx_xvfmul_s(_f1, _alpha256);
                _f2 = __lasx_xvfmul_s(_f2, _alpha256);
                _f3 = __lasx_xvfmul_s(_f3, _alpha256);
            }
            __lasx_xvst(_f0, p0, 0);
            __lasx_xvst(_f1, p1, 0);
            __lasx_xvst(_f2, p2, 0);
            __lasx_xvst(_f3, p3, 0);
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
        }
#endif // __loongarch_asx
        __m128 _alpha128 = __lsx_vreplfr2vr_s(alpha);
        __m128 _beta128 = __lsx_vreplfr2vr_s(beta);
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f00 = (__m128)__lsx_vld(pp, 0);
            __m128 _f01 = (__m128)__lsx_vld(pp + 4, 0);
            __m128 _f10 = (__m128)__lsx_vld(pp + 8, 0);
            __m128 _f11 = (__m128)__lsx_vld(pp + 12, 0);
            __m128 _f20 = (__m128)__lsx_vld(pp + 16, 0);
            __m128 _f21 = (__m128)__lsx_vld(pp + 20, 0);
            __m128 _f30 = (__m128)__lsx_vld(pp + 24, 0);
            __m128 _f31 = (__m128)__lsx_vld(pp + 28, 0);
            pp += 32;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = __lsx_vreplfr2vr_s(c0);
                    _f00 = __lsx_vfadd_s(_f00, _c);
                    _f01 = __lsx_vfadd_s(_f01, _c);
                    _f10 = __lsx_vfadd_s(_f10, _c);
                    _f11 = __lsx_vfadd_s(_f11, _c);
                    _f20 = __lsx_vfadd_s(_f20, _c);
                    _f21 = __lsx_vfadd_s(_f21, _c);
                    _f30 = __lsx_vfadd_s(_f30, _c);
                    _f31 = __lsx_vfadd_s(_f31, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(c0);
                    __m128 _c1 = __lsx_vreplfr2vr_s(c1);
                    __m128 _c2 = __lsx_vreplfr2vr_s(c2);
                    __m128 _c3 = __lsx_vreplfr2vr_s(c3);
                    _f00 = __lsx_vfadd_s(_f00, _c0);
                    _f01 = __lsx_vfadd_s(_f01, _c0);
                    _f10 = __lsx_vfadd_s(_f10, _c1);
                    _f11 = __lsx_vfadd_s(_f11, _c1);
                    _f20 = __lsx_vfadd_s(_f20, _c2);
                    _f21 = __lsx_vfadd_s(_f21, _c2);
                    _f30 = __lsx_vfadd_s(_f30, _c3);
                    _f31 = __lsx_vfadd_s(_f31, _c3);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c00 = (__m128)__lsx_vld(pC, 0);
                    __m128 _c01 = (__m128)__lsx_vld(pC + 4, 0);
                    __m128 _c10 = (__m128)__lsx_vld(pC + c_hstep, 0);
                    __m128 _c11 = (__m128)__lsx_vld(pC + c_hstep + 4, 0);
                    __m128 _c20 = (__m128)__lsx_vld(pC + c_hstep * 2, 0);
                    __m128 _c21 = (__m128)__lsx_vld(pC + c_hstep * 2 + 4, 0);
                    __m128 _c30 = (__m128)__lsx_vld(pC + c_hstep * 3, 0);
                    __m128 _c31 = (__m128)__lsx_vld(pC + c_hstep * 3 + 4, 0);
                    if (beta == 1.f)
                    {
                        _f00 = __lsx_vfadd_s(_f00, _c00);
                        _f01 = __lsx_vfadd_s(_f01, _c01);
                        _f10 = __lsx_vfadd_s(_f10, _c10);
                        _f11 = __lsx_vfadd_s(_f11, _c11);
                        _f20 = __lsx_vfadd_s(_f20, _c20);
                        _f21 = __lsx_vfadd_s(_f21, _c21);
                        _f30 = __lsx_vfadd_s(_f30, _c30);
                        _f31 = __lsx_vfadd_s(_f31, _c31);
                    }
                    else
                    {
                        _f00 = __lsx_vfmadd_s(_c00, _beta128, _f00);
                        _f01 = __lsx_vfmadd_s(_c01, _beta128, _f01);
                        _f10 = __lsx_vfmadd_s(_c10, _beta128, _f10);
                        _f11 = __lsx_vfmadd_s(_c11, _beta128, _f11);
                        _f20 = __lsx_vfmadd_s(_c20, _beta128, _f20);
                        _f21 = __lsx_vfmadd_s(_c21, _beta128, _f21);
                        _f30 = __lsx_vfmadd_s(_c30, _beta128, _f30);
                        _f31 = __lsx_vfmadd_s(_c31, _beta128, _f31);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c0 = (__m128)__lsx_vld(pC, 0);
                    __m128 _c1 = (__m128)__lsx_vld(pC + 4, 0);
                    if (beta != 1.f)
                    {
                        _c0 = __lsx_vfmul_s(_c0, _beta128);
                        _c1 = __lsx_vfmul_s(_c1, _beta128);
                    }
                    _f00 = __lsx_vfadd_s(_f00, _c0);
                    _f01 = __lsx_vfadd_s(_f01, _c1);
                    _f10 = __lsx_vfadd_s(_f10, _c0);
                    _f11 = __lsx_vfadd_s(_f11, _c1);
                    _f20 = __lsx_vfadd_s(_f20, _c0);
                    _f21 = __lsx_vfadd_s(_f21, _c1);
                    _f30 = __lsx_vfadd_s(_f30, _c0);
                    _f31 = __lsx_vfadd_s(_f31, _c1);
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                _f00 = __lsx_vfmul_s(_f00, _alpha128);
                _f01 = __lsx_vfmul_s(_f01, _alpha128);
                _f10 = __lsx_vfmul_s(_f10, _alpha128);
                _f11 = __lsx_vfmul_s(_f11, _alpha128);
                _f20 = __lsx_vfmul_s(_f20, _alpha128);
                _f21 = __lsx_vfmul_s(_f21, _alpha128);
                _f30 = __lsx_vfmul_s(_f30, _alpha128);
                _f31 = __lsx_vfmul_s(_f31, _alpha128);
            }
            __lsx_vst((__m128i)_f00, p0, 0);
            __lsx_vst((__m128i)_f01, p0 + 4, 0);
            __lsx_vst((__m128i)_f10, p1, 0);
            __lsx_vst((__m128i)_f11, p1 + 4, 0);
            __lsx_vst((__m128i)_f20, p2, 0);
            __lsx_vst((__m128i)_f21, p2 + 4, 0);
            __lsx_vst((__m128i)_f30, p3, 0);
            __lsx_vst((__m128i)_f31, p3 + 4, 0);
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp, 0);
            __m128 _f1 = (__m128)__lsx_vld(pp + 4, 0);
            __m128 _f2 = (__m128)__lsx_vld(pp + 8, 0);
            __m128 _f3 = (__m128)__lsx_vld(pp + 12, 0);
            pp += 16;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(c0));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vreplfr2vr_s(c0));
                    _f2 = __lsx_vfadd_s(_f2, __lsx_vreplfr2vr_s(c0));
                    _f3 = __lsx_vfadd_s(_f3, __lsx_vreplfr2vr_s(c0));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(c0));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vreplfr2vr_s(c1));
                    _f2 = __lsx_vfadd_s(_f2, __lsx_vreplfr2vr_s(c2));
                    _f3 = __lsx_vfadd_s(_f3, __lsx_vreplfr2vr_s(c3));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = (__m128)__lsx_vld(pC, 0);
                    __m128 _c1 = (__m128)__lsx_vld(pC + c_hstep, 0);
                    __m128 _c2 = (__m128)__lsx_vld(pC + c_hstep * 2, 0);
                    __m128 _c3 = (__m128)__lsx_vld(pC + c_hstep * 3, 0);
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                        _f1 = __lsx_vfadd_s(_f1, _c1);
                        _f2 = __lsx_vfadd_s(_f2, _c2);
                        _f3 = __lsx_vfadd_s(_f3, _c3);
                    }
                    else
                    {
                        _f0 = __lsx_vfmadd_s(_c0, _beta128, _f0);
                        _f1 = __lsx_vfmadd_s(_c1, _beta128, _f1);
                        _f2 = __lsx_vfmadd_s(_c2, _beta128, _f2);
                        _f3 = __lsx_vfmadd_s(_c3, _beta128, _f3);
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = (__m128)__lsx_vld(pC, 0);
                    if (beta != 1.f)
                        _c = __lsx_vfmul_s(_c, _beta128);
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                    _f2 = __lsx_vfadd_s(_f2, _c);
                    _f3 = __lsx_vfadd_s(_f3, _c);
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lsx_vfmul_s(_f0, _alpha128);
                _f1 = __lsx_vfmul_s(_f1, _alpha128);
                _f2 = __lsx_vfmul_s(_f2, _alpha128);
                _f3 = __lsx_vfmul_s(_f3, _alpha128);
            }
            __lsx_vst((__m128i)_f0, p0, 0);
            __lsx_vst((__m128i)_f1, p1, 0);
            __lsx_vst((__m128i)_f2, p2, 0);
            __lsx_vst((__m128i)_f3, p3, 0);
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _f0 = (__m128)__lsx_vldrepl_d(pp, 0);
            __m128 _f1 = (__m128)__lsx_vldrepl_d(pp + 2, 0);
            __m128 _f2 = (__m128)__lsx_vldrepl_d(pp + 4, 0);
            __m128 _f3 = (__m128)__lsx_vldrepl_d(pp + 6, 0);
            pp += 8;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = __lsx_vreplfr2vr_s(c0);
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                    _f2 = __lsx_vfadd_s(_f2, _c);
                    _f3 = __lsx_vfadd_s(_f3, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(c0));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vreplfr2vr_s(c1));
                    _f2 = __lsx_vfadd_s(_f2, __lsx_vreplfr2vr_s(c2));
                    _f3 = __lsx_vfadd_s(_f3, __lsx_vreplfr2vr_s(c3));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = (__m128)__lsx_vldrepl_d(pC, 0);
                    __m128 _c1 = (__m128)__lsx_vldrepl_d(pC + c_hstep, 0);
                    __m128 _c2 = (__m128)__lsx_vldrepl_d(pC + c_hstep * 2, 0);
                    __m128 _c3 = (__m128)__lsx_vldrepl_d(pC + c_hstep * 3, 0);
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                        _f1 = __lsx_vfadd_s(_f1, _c1);
                        _f2 = __lsx_vfadd_s(_f2, _c2);
                        _f3 = __lsx_vfadd_s(_f3, _c3);
                    }
                    else
                    {
                        _f0 = __lsx_vfmadd_s(_c0, _beta128, _f0);
                        _f1 = __lsx_vfmadd_s(_c1, _beta128, _f1);
                        _f2 = __lsx_vfmadd_s(_c2, _beta128, _f2);
                        _f3 = __lsx_vfmadd_s(_c3, _beta128, _f3);
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = (__m128)__lsx_vldrepl_d(pC, 0);
                    if (beta != 1.f)
                        _c = __lsx_vfmul_s(_c, _beta128);
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                    _f2 = __lsx_vfadd_s(_f2, _c);
                    _f3 = __lsx_vfadd_s(_f3, _c);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lsx_vfmul_s(_f0, _alpha128);
                _f1 = __lsx_vfmul_s(_f1, _alpha128);
                _f2 = __lsx_vfmul_s(_f2, _alpha128);
                _f3 = __lsx_vfmul_s(_f3, _alpha128);
            }
            __lsx_vstelm_d((__m128i)_f0, p0, 0, 0);
            __lsx_vstelm_d((__m128i)_f1, p1, 0, 0);
            __lsx_vstelm_d((__m128i)_f2, p2, 0, 0);
            __lsx_vstelm_d((__m128i)_f3, p3, 0, 0);
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
        }
        for (; jj < max_jj; jj++)
        {
            __m128i _fi = __lsx_vld(pp, 0);
            pp += 4;
            __m128 _f0 = (__m128)_fi;
            if (pC)
            {
                if (broadcast_type_C == 0)
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(c0));
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vfmul_s((__m128)__lsx_vld(pC_base + i + ii, 0), _beta128));
                if (broadcast_type_C == 3)
                {
                    __m128i _c0 = __lsx_vreplgr2vr_w(((const int*)pC)[0]);
                    _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep))[0], 1);
                    _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep * 2))[0], 2);
                    _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep * 3))[0], 3);
                    if (beta == 1.f)
                        _f0 = __lsx_vfadd_s(_f0, (__m128)_c0);
                    else
                        _f0 = __lsx_vfmadd_s((__m128)_c0, _beta128, _f0);
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(beta == 1.f ? pC[0] : pC[0] * beta));
                    pC++;
                }
            }
            if (alpha != 1.f)
                _f0 = __lsx_vfmul_s(_f0, _alpha128);
            __lsx_vstelm_w((__m128i)_f0, p0, 0, 0);
            __lsx_vstelm_w((__m128i)_f0, p1, 0, 1);
            __lsx_vstelm_w((__m128i)_f0, p2, 0, 2);
            __lsx_vstelm_w((__m128i)_f0, p3, 0, 3);
            p0++;
            p1++;
            p2++;
            p3++;
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* p0 = outptr + (size_t)(i + ii) * N + j;
        float* p1 = p0 + N;

        float c0 = 0.f;
        float c1 = 0.f;
        const float* pC = pC_base;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
                c1 = c0;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[i + ii] * beta;
                c1 = pC[i + ii + 1] * beta;
            }
            if (broadcast_type_C == 3)
            {
                pC += (size_t)(i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC += j;
            }
        }

        int jj = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _alpha256 = (__m256)__lasx_xvreplfr2vr_s(alpha);
        __m256 _beta256 = (__m256)__lasx_xvreplfr2vr_s(beta);
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _f00 = (__m256)__lasx_xvld(pp, 0);
            __m256 _f01 = (__m256)__lasx_xvld(pp + 8, 0);
            __m256 _f10 = (__m256)__lasx_xvld(pp + 16, 0);
            __m256 _f11 = (__m256)__lasx_xvld(pp + 24, 0);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m256 _c = (__m256)__lasx_xvreplfr2vr_s(c0);
                    _f00 = __lasx_xvfadd_s(_f00, _c);
                    _f01 = __lasx_xvfadd_s(_f01, _c);
                    _f10 = __lasx_xvfadd_s(_f10, _c);
                    _f11 = __lasx_xvfadd_s(_f11, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m256 _c0 = (__m256)__lasx_xvreplfr2vr_s(c0);
                    __m256 _c1 = (__m256)__lasx_xvreplfr2vr_s(c1);
                    _f00 = __lasx_xvfadd_s(_f00, _c0);
                    _f01 = __lasx_xvfadd_s(_f01, _c0);
                    _f10 = __lasx_xvfadd_s(_f10, _c1);
                    _f11 = __lasx_xvfadd_s(_f11, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c00 = (__m256)__lasx_xvld(pC, 0);
                    __m256 _c01 = (__m256)__lasx_xvld(pC + 8, 0);
                    __m256 _c10 = (__m256)__lasx_xvld(pC + c_hstep, 0);
                    __m256 _c11 = (__m256)__lasx_xvld(pC + c_hstep + 8, 0);
                    if (beta == 1.f)
                    {
                        _f00 = __lasx_xvfadd_s(_f00, _c00);
                        _f01 = __lasx_xvfadd_s(_f01, _c01);
                        _f10 = __lasx_xvfadd_s(_f10, _c10);
                        _f11 = __lasx_xvfadd_s(_f11, _c11);
                    }
                    else
                    {
                        _f00 = __lasx_xvfmadd_s(_c00, _beta256, _f00);
                        _f01 = __lasx_xvfmadd_s(_c01, _beta256, _f01);
                        _f10 = __lasx_xvfmadd_s(_c10, _beta256, _f10);
                        _f11 = __lasx_xvfmadd_s(_c11, _beta256, _f11);
                    }
                    pC += 16;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c0 = (__m256)__lasx_xvld(pC, 0);
                    __m256 _c1 = (__m256)__lasx_xvld(pC + 8, 0);
                    if (beta != 1.f)
                    {
                        _c0 = __lasx_xvfmul_s(_c0, _beta256);
                        _c1 = __lasx_xvfmul_s(_c1, _beta256);
                    }
                    _f00 = __lasx_xvfadd_s(_f00, _c0);
                    _f01 = __lasx_xvfadd_s(_f01, _c1);
                    _f10 = __lasx_xvfadd_s(_f10, _c0);
                    _f11 = __lasx_xvfadd_s(_f11, _c1);
                    pC += 16;
                }
            }
            if (alpha != 1.f)
            {
                _f00 = __lasx_xvfmul_s(_f00, _alpha256);
                _f01 = __lasx_xvfmul_s(_f01, _alpha256);
                _f10 = __lasx_xvfmul_s(_f10, _alpha256);
                _f11 = __lasx_xvfmul_s(_f11, _alpha256);
            }
            __lasx_xvst(_f00, p0, 0);
            __lasx_xvst(_f01, p0 + 8, 0);
            __lasx_xvst(_f10, p1, 0);
            __lasx_xvst(_f11, p1 + 8, 0);
            p0 += 16;
            p1 += 16;
            pp += 32;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = (__m256)__lasx_xvld(pp, 0);
            __m256 _f1 = (__m256)__lasx_xvld(pp + 8, 0);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = __lasx_xvfadd_s(_f0, (__m256)__lasx_xvreplfr2vr_s(c0));
                    _f1 = __lasx_xvfadd_s(_f1, (__m256)__lasx_xvreplfr2vr_s(c0));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lasx_xvfadd_s(_f0, (__m256)__lasx_xvreplfr2vr_s(c0));
                    _f1 = __lasx_xvfadd_s(_f1, (__m256)__lasx_xvreplfr2vr_s(c1));
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c0 = (__m256)__lasx_xvld(pC, 0);
                    __m256 _c1 = (__m256)__lasx_xvld(pC + c_hstep, 0);
                    if (beta == 1.f)
                    {
                        _f0 = __lasx_xvfadd_s(_f0, _c0);
                        _f1 = __lasx_xvfadd_s(_f1, _c1);
                    }
                    else
                    {
                        _f0 = __lasx_xvfmadd_s(_c0, _beta256, _f0);
                        _f1 = __lasx_xvfmadd_s(_c1, _beta256, _f1);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c = (__m256)__lasx_xvld(pC, 0);
                    if (beta != 1.f)
                        _c = __lasx_xvfmul_s(_c, _beta256);
                    _f0 = __lasx_xvfadd_s(_f0, _c);
                    _f1 = __lasx_xvfadd_s(_f1, _c);
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lasx_xvfmul_s(_f0, _alpha256);
                _f1 = __lasx_xvfmul_s(_f1, _alpha256);
            }
            __lasx_xvst(_f0, p0, 0);
            __lasx_xvst(_f1, p1, 0);
            p0 += 8;
            p1 += 8;
            pp += 16;
        }
#endif // __loongarch_asx
        __m128 _alpha128 = __lsx_vreplfr2vr_s(alpha);
        __m128 _beta128 = __lsx_vreplfr2vr_s(beta);
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f00 = (__m128)__lsx_vld(pp, 0);
            __m128 _f01 = (__m128)__lsx_vld(pp + 4, 0);
            __m128 _f10 = (__m128)__lsx_vld(pp + 8, 0);
            __m128 _f11 = (__m128)__lsx_vld(pp + 12, 0);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = __lsx_vreplfr2vr_s(c0);
                    _f00 = __lsx_vfadd_s(_f00, _c);
                    _f01 = __lsx_vfadd_s(_f01, _c);
                    _f10 = __lsx_vfadd_s(_f10, _c);
                    _f11 = __lsx_vfadd_s(_f11, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(c0);
                    __m128 _c1 = __lsx_vreplfr2vr_s(c1);
                    _f00 = __lsx_vfadd_s(_f00, _c0);
                    _f01 = __lsx_vfadd_s(_f01, _c0);
                    _f10 = __lsx_vfadd_s(_f10, _c1);
                    _f11 = __lsx_vfadd_s(_f11, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c00 = (__m128)__lsx_vld(pC, 0);
                    __m128 _c01 = (__m128)__lsx_vld(pC + 4, 0);
                    __m128 _c10 = (__m128)__lsx_vld(pC + c_hstep, 0);
                    __m128 _c11 = (__m128)__lsx_vld(pC + c_hstep + 4, 0);
                    if (beta == 1.f)
                    {
                        _f00 = __lsx_vfadd_s(_f00, _c00);
                        _f01 = __lsx_vfadd_s(_f01, _c01);
                        _f10 = __lsx_vfadd_s(_f10, _c10);
                        _f11 = __lsx_vfadd_s(_f11, _c11);
                    }
                    else
                    {
                        _f00 = __lsx_vfmadd_s(_c00, _beta128, _f00);
                        _f01 = __lsx_vfmadd_s(_c01, _beta128, _f01);
                        _f10 = __lsx_vfmadd_s(_c10, _beta128, _f10);
                        _f11 = __lsx_vfmadd_s(_c11, _beta128, _f11);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c0 = (__m128)__lsx_vld(pC, 0);
                    __m128 _c1 = (__m128)__lsx_vld(pC + 4, 0);
                    if (beta != 1.f)
                    {
                        _c0 = __lsx_vfmul_s(_c0, _beta128);
                        _c1 = __lsx_vfmul_s(_c1, _beta128);
                    }
                    _f00 = __lsx_vfadd_s(_f00, _c0);
                    _f01 = __lsx_vfadd_s(_f01, _c1);
                    _f10 = __lsx_vfadd_s(_f10, _c0);
                    _f11 = __lsx_vfadd_s(_f11, _c1);
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                _f00 = __lsx_vfmul_s(_f00, _alpha128);
                _f01 = __lsx_vfmul_s(_f01, _alpha128);
                _f10 = __lsx_vfmul_s(_f10, _alpha128);
                _f11 = __lsx_vfmul_s(_f11, _alpha128);
            }
            __lsx_vst((__m128i)_f00, p0, 0);
            __lsx_vst((__m128i)_f01, p0 + 4, 0);
            __lsx_vst((__m128i)_f10, p1, 0);
            __lsx_vst((__m128i)_f11, p1 + 4, 0);
            p0 += 8;
            p1 += 8;
            pp += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp, 0);
            __m128 _f1 = (__m128)__lsx_vld(pp + 4, 0);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(c0));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vreplfr2vr_s(c0));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(c0));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vreplfr2vr_s(c1));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = (__m128)__lsx_vld(pC, 0);
                    __m128 _c1 = (__m128)__lsx_vld(pC + c_hstep, 0);
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                        _f1 = __lsx_vfadd_s(_f1, _c1);
                    }
                    else
                    {
                        _f0 = __lsx_vfmadd_s(_c0, _beta128, _f0);
                        _f1 = __lsx_vfmadd_s(_c1, _beta128, _f1);
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = (__m128)__lsx_vld(pC, 0);
                    if (beta != 1.f)
                        _c = __lsx_vfmul_s(_c, _beta128);
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lsx_vfmul_s(_f0, _alpha128);
                _f1 = __lsx_vfmul_s(_f1, _alpha128);
            }
            __lsx_vst((__m128i)_f0, p0, 0);
            __lsx_vst((__m128i)_f1, p1, 0);
            p0 += 4;
            p1 += 4;
            pp += 8;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _f0 = (__m128)__lsx_vldrepl_d(pp, 0);
            __m128 _f1 = (__m128)__lsx_vldrepl_d(pp + 2, 0);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c = __lsx_vreplfr2vr_s(c0);
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(c0));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vreplfr2vr_s(c1));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = (__m128)__lsx_vldrepl_d(pC, 0);
                    __m128 _c1 = (__m128)__lsx_vldrepl_d(pC + c_hstep, 0);
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                        _f1 = __lsx_vfadd_s(_f1, _c1);
                    }
                    else
                    {
                        _f0 = __lsx_vfmadd_s(_c0, _beta128, _f0);
                        _f1 = __lsx_vfmadd_s(_c1, _beta128, _f1);
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = (__m128)__lsx_vldrepl_d(pC, 0);
                    if (beta != 1.f)
                        _c = __lsx_vfmul_s(_c, _beta128);
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lsx_vfmul_s(_f0, _alpha128);
                _f1 = __lsx_vfmul_s(_f1, _alpha128);
            }
            __lsx_vstelm_d((__m128i)_f0, p0, 0, 0);
            __lsx_vstelm_d((__m128i)_f1, p1, 0, 0);
            p0 += 2;
            p1 += 2;
            pp += 4;
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            float f00 = pp[0];
            float f01 = pp[1];
            float f10 = pp[2];
            float f11 = pp[3];
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    f00 += c0;
                    f01 += c0;
                    f10 += c0;
                    f11 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f00 += c0;
                    f01 += c0;
                    f10 += c1;
                    f11 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    if (beta == 1.f)
                    {
                        f00 += pC[0];
                        f01 += pC[1];
                        f10 += pC[c_hstep];
                        f11 += pC[c_hstep + 1];
                    }
                    else
                    {
                        f00 += pC[0] * beta;
                        f01 += pC[1] * beta;
                        f10 += pC[c_hstep] * beta;
                        f11 += pC[c_hstep + 1] * beta;
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    const float cc0 = beta == 1.f ? pC[0] : pC[0] * beta;
                    const float cc1 = beta == 1.f ? pC[1] : pC[1] * beta;
                    f00 += cc0;
                    f01 += cc1;
                    f10 += cc0;
                    f11 += cc1;
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                f00 *= alpha;
                f01 *= alpha;
                f10 *= alpha;
                f11 *= alpha;
            }
            p0[0] = f00;
            p0[1] = f01;
            p1[0] = f10;
            p1[1] = f11;
            p0 += 2;
            p1 += 2;
            pp += 4;
        }
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[0];
            float f1 = pp[1];
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    f0 += c0;
                    f1 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                    f1 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    if (beta == 1.f)
                    {
                        f0 += pC[0];
                        f1 += pC[c_hstep];
                    }
                    else
                    {
                        f0 += pC[0] * beta;
                        f1 += pC[c_hstep] * beta;
                    }
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    float c = beta == 1.f ? pC[0] : pC[0] * beta;
                    f0 += c;
                    f1 += c;
                    pC++;
                }
            }
            if (alpha != 1.f)
            {
                f0 *= alpha;
                f1 *= alpha;
            }
            p0[0] = f0;
            p1[0] = f1;
            p0++;
            p1++;
            pp += 2;
        }
    }
    for (; ii < max_ii; ii++)
    {
        float* p0 = outptr + (size_t)(i + ii) * N + j;

        float c0 = 0.f;
        const float* pC = pC_base;
        if (pC)
        {
            if (broadcast_type_C == 0)
                c0 = pC[0] * beta;
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                c0 = pC[i + ii] * beta;
            if (broadcast_type_C == 3)
                pC += (size_t)(i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC += j;
        }

        int jj = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _alpha256 = (__m256)__lasx_xvreplfr2vr_s(alpha);
        __m256 _beta256 = (__m256)__lasx_xvreplfr2vr_s(beta);
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _f0 = (__m256)__lasx_xvld(pp, 0);
            __m256 _f1 = (__m256)__lasx_xvld(pp + 8, 0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m256 _c = (__m256)__lasx_xvreplfr2vr_s(c0);
                    _f0 = __lasx_xvfadd_s(_f0, _c);
                    _f1 = __lasx_xvfadd_s(_f1, _c);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m256 _c0 = (__m256)__lasx_xvld(pC, 0);
                    __m256 _c1 = (__m256)__lasx_xvld(pC + 8, 0);
                    if (beta == 1.f)
                    {
                        _f0 = __lasx_xvfadd_s(_f0, _c0);
                        _f1 = __lasx_xvfadd_s(_f1, _c1);
                    }
                    else
                    {
                        _f0 = __lasx_xvfmadd_s(_c0, _beta256, _f0);
                        _f1 = __lasx_xvfmadd_s(_c1, _beta256, _f1);
                    }
                    pC += 16;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lasx_xvfmul_s(_f0, _alpha256);
                _f1 = __lasx_xvfmul_s(_f1, _alpha256);
            }
            __lasx_xvst(_f0, p0, 0);
            __lasx_xvst(_f1, p0 + 8, 0);
            p0 += 16;
            pp += 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = (__m256)__lasx_xvld(pp, 0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f0 = __lasx_xvfadd_s(_f0, (__m256)__lasx_xvreplfr2vr_s(c0));
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m256 _c0 = (__m256)__lasx_xvld(pC, 0);
                    if (beta == 1.f)
                        _f0 = __lasx_xvfadd_s(_f0, _c0);
                    else
                        _f0 = __lasx_xvfmadd_s(_c0, _beta256, _f0);
                    pC += 8;
                }
            }
            if (alpha != 1.f) _f0 = __lasx_xvfmul_s(_f0, _alpha256);
            __lasx_xvst(_f0, p0, 0);
            p0 += 8;
            pp += 8;
        }
#endif // __loongarch_asx
        __m128 _alpha128 = __lsx_vreplfr2vr_s(alpha);
        __m128 _beta128 = __lsx_vreplfr2vr_s(beta);
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp, 0);
            __m128 _f1 = (__m128)__lsx_vld(pp + 4, 0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c = __lsx_vreplfr2vr_s(c0);
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m128 _c0 = (__m128)__lsx_vld(pC, 0);
                    __m128 _c1 = (__m128)__lsx_vld(pC + 4, 0);
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                        _f1 = __lsx_vfadd_s(_f1, _c1);
                    }
                    else
                    {
                        _f0 = __lsx_vfmadd_s(_c0, _beta128, _f0);
                        _f1 = __lsx_vfmadd_s(_c1, _beta128, _f1);
                    }
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lsx_vfmul_s(_f0, _alpha128);
                _f1 = __lsx_vfmul_s(_f1, _alpha128);
            }
            __lsx_vst((__m128i)_f0, p0, 0);
            __lsx_vst((__m128i)_f1, p0 + 4, 0);
            p0 += 8;
            pp += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp, 0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(c0));
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m128 _c0 = (__m128)__lsx_vld(pC, 0);
                    if (beta == 1.f)
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                    else
                        _f0 = __lsx_vfmadd_s(_c0, _beta128, _f0);
                    pC += 4;
                }
            }
            if (alpha != 1.f) _f0 = __lsx_vfmul_s(_f0, _alpha128);
            __lsx_vst((__m128i)_f0, p0, 0);
            p0 += 4;
            pp += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _f0 = (__m128)__lsx_vldrepl_d(pp, 0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(c0));
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m128 _c0 = (__m128)__lsx_vldrepl_d(pC, 0);
                    if (beta == 1.f)
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                    else
                        _f0 = __lsx_vfmadd_s(_c0, _beta128, _f0);
                    pC += 2;
                }
            }
            if (alpha != 1.f) _f0 = __lsx_vfmul_s(_f0, _alpha128);
            __lsx_vstelm_d((__m128i)_f0, p0, 0, 0);
            p0 += 2;
            pp += 2;
        }
#endif // __loongarch_sx
        for (; jj < max_jj; jj++)
        {
            float f0 = *pp++;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    f0 += c0;
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    f0 += beta == 1.f ? pC[0] : pC[0] * beta;
                    pC++;
                }
            }
            if (alpha != 1.f)
                f0 *= alpha;
            p0[0] = f0;
            p0++;
        }
    }
}

static void transpose_unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int M, float alpha, float beta)
{
    const float* pp = topT;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const float* pC_base = C;
    float* outptr = top_blob;
    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* p0 = outptr + (size_t)j * M + i + ii;
        const float* pC = pC_base;

        __m128 _c0 = __lsx_vreplfr2vr_s(0.f);
        __m128 _c1 = __lsx_vreplfr2vr_s(0.f);
        if (pC && broadcast_type_C == 0)
        {
            _c0 = __lsx_vreplfr2vr_s(pC[0] * beta);
            _c1 = _c0;
        }
        if (pC && (broadcast_type_C == 1 || broadcast_type_C == 2))
        {
            _c0 = (__m128)__lsx_vld(pC + i + ii, 0);
            _c1 = (__m128)__lsx_vld(pC + i + ii + 4, 0);
            if (beta != 1.f)
            {
                __m128 _beta = __lsx_vreplfr2vr_s(beta);
                _c0 = __lsx_vfmul_s(_c0, _beta);
                _c1 = __lsx_vfmul_s(_c1, _beta);
            }
        }

        if (pC && broadcast_type_C == 3)
            pC += (size_t)(i + ii) * c_hstep + j;
        if (pC && broadcast_type_C == 4)
            pC += j;

        __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
        __m128 _beta = __lsx_vreplfr2vr_s(beta);
        int jj = 0;
#if __loongarch_asx
        __m256 _alpha256 = (__m256)__lasx_xvreplfr2vr_s(alpha);
        __m256 _beta256 = (__m256)__lasx_xvreplfr2vr_s(beta);
        __m256 _c256 = __lasx_concat_128_s(_c0, _c1);
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256i _sum0 = __lasx_xvld(pp, 0);
            __m256i _sum1 = __lasx_xvld(pp + 8, 0);
            __m256i _sum2 = __lasx_xvld(pp + 16, 0);
            __m256i _sum3 = __lasx_xvld(pp + 24, 0);
            __m256i _sum4 = __lasx_xvld(pp + 32, 0);
            __m256i _sum5 = __lasx_xvld(pp + 40, 0);
            __m256i _sum6 = __lasx_xvld(pp + 48, 0);
            __m256i _sum7 = __lasx_xvld(pp + 56, 0);
            __m256i _tmp0 = _sum0;
            __m256i _tmp1 = __lasx_xvshuf4i_w(_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
            __m256i _tmp2 = _sum2;
            __m256i _tmp3 = __lasx_xvshuf4i_w(_sum3, _LSX_SHUFFLE(2, 1, 0, 3));
            __m256i _tmp4 = _sum4;
            __m256i _tmp5 = __lasx_xvshuf4i_w(_sum5, _LSX_SHUFFLE(2, 1, 0, 3));
            __m256i _tmp6 = _sum6;
            __m256i _tmp7 = __lasx_xvshuf4i_w(_sum7, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum0 = __lasx_xvilvl_w(_tmp3, _tmp0);
            _sum1 = __lasx_xvilvh_w(_tmp3, _tmp0);
            _sum2 = __lasx_xvilvl_w(_tmp1, _tmp2);
            _sum3 = __lasx_xvilvh_w(_tmp1, _tmp2);
            _sum4 = __lasx_xvilvl_w(_tmp7, _tmp4);
            _sum5 = __lasx_xvilvh_w(_tmp7, _tmp4);
            _sum6 = __lasx_xvilvl_w(_tmp5, _tmp6);
            _sum7 = __lasx_xvilvh_w(_tmp5, _tmp6);
            _tmp0 = __lasx_xvilvl_d(_sum2, _sum0);
            _tmp1 = __lasx_xvilvh_d(_sum2, _sum0);
            _tmp2 = __lasx_xvilvl_d(_sum1, _sum3);
            _tmp3 = __lasx_xvilvh_d(_sum1, _sum3);
            _tmp4 = __lasx_xvilvl_d(_sum6, _sum4);
            _tmp5 = __lasx_xvilvh_d(_sum6, _sum4);
            _tmp6 = __lasx_xvilvl_d(_sum5, _sum7);
            _tmp7 = __lasx_xvilvh_d(_sum5, _sum7);
            _tmp1 = __lasx_xvshuf4i_w(_tmp1, _LSX_SHUFFLE(2, 1, 0, 3));
            _tmp3 = __lasx_xvshuf4i_w(_tmp3, _LSX_SHUFFLE(2, 1, 0, 3));
            _tmp5 = __lasx_xvshuf4i_w(_tmp5, _LSX_SHUFFLE(2, 1, 0, 3));
            _tmp7 = __lasx_xvshuf4i_w(_tmp7, _LSX_SHUFFLE(2, 1, 0, 3));
            __m256 _f0 = (__m256)__lasx_xvpermi_q(_tmp4, _tmp0, _LSX_SHUFFLE(0, 3, 0, 0));
            __m256 _f1 = (__m256)__lasx_xvpermi_q(_tmp5, _tmp1, _LSX_SHUFFLE(0, 3, 0, 0));
            __m256 _f2 = (__m256)__lasx_xvpermi_q(_tmp6, _tmp2, _LSX_SHUFFLE(0, 3, 0, 0));
            __m256 _f3 = (__m256)__lasx_xvpermi_q(_tmp7, _tmp3, _LSX_SHUFFLE(0, 3, 0, 0));
            __m256 _f4 = (__m256)__lasx_xvpermi_q(_tmp0, _tmp4, _LSX_SHUFFLE(0, 3, 0, 0));
            __m256 _f5 = (__m256)__lasx_xvpermi_q(_tmp1, _tmp5, _LSX_SHUFFLE(0, 3, 0, 0));
            __m256 _f6 = (__m256)__lasx_xvpermi_q(_tmp2, _tmp6, _LSX_SHUFFLE(0, 3, 0, 0));
            __m256 _f7 = (__m256)__lasx_xvpermi_q(_tmp3, _tmp7, _LSX_SHUFFLE(0, 3, 0, 0));
            pp += 64;
            transpose8x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lasx_xvfadd_s(_f0, _c256);
                    _f1 = __lasx_xvfadd_s(_f1, _c256);
                    _f2 = __lasx_xvfadd_s(_f2, _c256);
                    _f3 = __lasx_xvfadd_s(_f3, _c256);
                    _f4 = __lasx_xvfadd_s(_f4, _c256);
                    _f5 = __lasx_xvfadd_s(_f5, _c256);
                    _f6 = __lasx_xvfadd_s(_f6, _c256);
                    _f7 = __lasx_xvfadd_s(_f7, _c256);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _cc0 = (__m256)__lasx_xvld(pC, 0);
                    __m256 _cc1 = (__m256)__lasx_xvld(pC + c_hstep, 0);
                    __m256 _cc2 = (__m256)__lasx_xvld(pC + c_hstep * 2, 0);
                    __m256 _cc3 = (__m256)__lasx_xvld(pC + c_hstep * 3, 0);
                    __m256 _cc4 = (__m256)__lasx_xvld(pC + c_hstep * 4, 0);
                    __m256 _cc5 = (__m256)__lasx_xvld(pC + c_hstep * 5, 0);
                    __m256 _cc6 = (__m256)__lasx_xvld(pC + c_hstep * 6, 0);
                    __m256 _cc7 = (__m256)__lasx_xvld(pC + c_hstep * 7, 0);
                    transpose8x8_ps(_cc0, _cc1, _cc2, _cc3, _cc4, _cc5, _cc6, _cc7);
                    if (beta == 1.f)
                    {
                        _f0 = __lasx_xvfadd_s(_f0, _cc0);
                        _f1 = __lasx_xvfadd_s(_f1, _cc1);
                        _f2 = __lasx_xvfadd_s(_f2, _cc2);
                        _f3 = __lasx_xvfadd_s(_f3, _cc3);
                        _f4 = __lasx_xvfadd_s(_f4, _cc4);
                        _f5 = __lasx_xvfadd_s(_f5, _cc5);
                        _f6 = __lasx_xvfadd_s(_f6, _cc6);
                        _f7 = __lasx_xvfadd_s(_f7, _cc7);
                    }
                    else
                    {
                        _f0 = __lasx_xvfmadd_s(_cc0, _beta256, _f0);
                        _f1 = __lasx_xvfmadd_s(_cc1, _beta256, _f1);
                        _f2 = __lasx_xvfmadd_s(_cc2, _beta256, _f2);
                        _f3 = __lasx_xvfmadd_s(_cc3, _beta256, _f3);
                        _f4 = __lasx_xvfmadd_s(_cc4, _beta256, _f4);
                        _f5 = __lasx_xvfmadd_s(_cc5, _beta256, _f5);
                        _f6 = __lasx_xvfmadd_s(_cc6, _beta256, _f6);
                        _f7 = __lasx_xvfmadd_s(_cc7, _beta256, _f7);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    _f0 = __lasx_xvfadd_s(_f0, (__m256)__lasx_xvreplfr2vr_s(pC[0] * beta));
                    _f1 = __lasx_xvfadd_s(_f1, (__m256)__lasx_xvreplfr2vr_s(pC[1] * beta));
                    _f2 = __lasx_xvfadd_s(_f2, (__m256)__lasx_xvreplfr2vr_s(pC[2] * beta));
                    _f3 = __lasx_xvfadd_s(_f3, (__m256)__lasx_xvreplfr2vr_s(pC[3] * beta));
                    _f4 = __lasx_xvfadd_s(_f4, (__m256)__lasx_xvreplfr2vr_s(pC[4] * beta));
                    _f5 = __lasx_xvfadd_s(_f5, (__m256)__lasx_xvreplfr2vr_s(pC[5] * beta));
                    _f6 = __lasx_xvfadd_s(_f6, (__m256)__lasx_xvreplfr2vr_s(pC[6] * beta));
                    _f7 = __lasx_xvfadd_s(_f7, (__m256)__lasx_xvreplfr2vr_s(pC[7] * beta));
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lasx_xvfmul_s(_f0, _alpha256);
                _f1 = __lasx_xvfmul_s(_f1, _alpha256);
                _f2 = __lasx_xvfmul_s(_f2, _alpha256);
                _f3 = __lasx_xvfmul_s(_f3, _alpha256);
                _f4 = __lasx_xvfmul_s(_f4, _alpha256);
                _f5 = __lasx_xvfmul_s(_f5, _alpha256);
                _f6 = __lasx_xvfmul_s(_f6, _alpha256);
                _f7 = __lasx_xvfmul_s(_f7, _alpha256);
            }
            __lasx_xvst(_f0, p0, 0);
            __lasx_xvst(_f1, p0 + M, 0);
            __lasx_xvst(_f2, p0 + M * 2, 0);
            __lasx_xvst(_f3, p0 + M * 3, 0);
            __lasx_xvst(_f4, p0 + M * 4, 0);
            __lasx_xvst(_f5, p0 + M * 5, 0);
            __lasx_xvst(_f6, p0 + M * 6, 0);
            __lasx_xvst(_f7, p0 + M * 7, 0);
            p0 += M * 8;
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _sum0 = __lsx_vld(pp, 0);
            __m128i _sum1 = __lsx_vld(pp + 8, 0);
            __m128i _sum2 = __lsx_vld(pp + 16, 0);
            __m128i _sum3 = __lsx_vld(pp + 24, 0);
            _sum2 = __lsx_vshuf4i_w(_sum2, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum3 = __lsx_vshuf4i_w(_sum3, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
            _sum1 = __lsx_vshuf4i_w(_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum2 = __lsx_vshuf4i_w(_sum2, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum3 = __lsx_vshuf4i_w(_sum3, _LSX_SHUFFLE(0, 3, 2, 1));
            __m128i _sum4 = __lsx_vld(pp + 4, 0);
            __m128i _sum5 = __lsx_vld(pp + 12, 0);
            __m128i _sum6 = __lsx_vld(pp + 20, 0);
            __m128i _sum7 = __lsx_vld(pp + 28, 0);
            _sum6 = __lsx_vshuf4i_w(_sum6, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum7 = __lsx_vshuf4i_w(_sum7, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum4, _sum5, _sum6, _sum7);
            _sum5 = __lsx_vshuf4i_w(_sum5, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum6 = __lsx_vshuf4i_w(_sum6, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum7 = __lsx_vshuf4i_w(_sum7, _LSX_SHUFFLE(0, 3, 2, 1));
            __m128 _f0 = (__m128)_sum0;
            __m128 _f1 = (__m128)_sum1;
            __m128 _f2 = (__m128)_sum2;
            __m128 _f3 = (__m128)_sum3;
            __m128 _f4 = (__m128)_sum4;
            __m128 _f5 = (__m128)_sum5;
            __m128 _f6 = (__m128)_sum6;
            __m128 _f7 = (__m128)_sum7;
            pp += 32;
            transpose4x4_ps(_f0, _f1, _f2, _f3);
            transpose4x4_ps(_f4, _f5, _f6, _f7);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                    _f2 = __lsx_vfadd_s(_f2, _c0);
                    _f3 = __lsx_vfadd_s(_f3, _c0);
                    _f4 = __lsx_vfadd_s(_f4, _c1);
                    _f5 = __lsx_vfadd_s(_f5, _c1);
                    _f6 = __lsx_vfadd_s(_f6, _c1);
                    _f7 = __lsx_vfadd_s(_f7, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _cc0 = (__m128)__lsx_vld(pC, 0);
                    __m128 _cc1 = (__m128)__lsx_vld(pC + c_hstep, 0);
                    __m128 _cc2 = (__m128)__lsx_vld(pC + c_hstep * 2, 0);
                    __m128 _cc3 = (__m128)__lsx_vld(pC + c_hstep * 3, 0);
                    __m128 _cc4 = (__m128)__lsx_vld(pC + c_hstep * 4, 0);
                    __m128 _cc5 = (__m128)__lsx_vld(pC + c_hstep * 5, 0);
                    __m128 _cc6 = (__m128)__lsx_vld(pC + c_hstep * 6, 0);
                    __m128 _cc7 = (__m128)__lsx_vld(pC + c_hstep * 7, 0);
                    transpose4x4_ps(_cc0, _cc1, _cc2, _cc3);
                    transpose4x4_ps(_cc4, _cc5, _cc6, _cc7);
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, _cc0);
                        _f1 = __lsx_vfadd_s(_f1, _cc1);
                        _f2 = __lsx_vfadd_s(_f2, _cc2);
                        _f3 = __lsx_vfadd_s(_f3, _cc3);
                        _f4 = __lsx_vfadd_s(_f4, _cc4);
                        _f5 = __lsx_vfadd_s(_f5, _cc5);
                        _f6 = __lsx_vfadd_s(_f6, _cc6);
                        _f7 = __lsx_vfadd_s(_f7, _cc7);
                    }
                    else
                    {
                        _f0 = __lsx_vfmadd_s(_cc0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s(_cc1, _beta, _f1);
                        _f2 = __lsx_vfmadd_s(_cc2, _beta, _f2);
                        _f3 = __lsx_vfmadd_s(_cc3, _beta, _f3);
                        _f4 = __lsx_vfmadd_s(_cc4, _beta, _f4);
                        _f5 = __lsx_vfmadd_s(_cc5, _beta, _f5);
                        _f6 = __lsx_vfmadd_s(_cc6, _beta, _f6);
                        _f7 = __lsx_vfmadd_s(_cc7, _beta, _f7);
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _cc = (__m128)__lsx_vld(pC, 0);
                    if (beta != 1.f)
                        _cc = __lsx_vfmul_s(_cc, _beta);
                    _f0 = __lsx_vfadd_s(_f0, (__m128)__lsx_vreplvei_w((__m128i)_cc, 0));
                    _f1 = __lsx_vfadd_s(_f1, (__m128)__lsx_vreplvei_w((__m128i)_cc, 1));
                    _f2 = __lsx_vfadd_s(_f2, (__m128)__lsx_vreplvei_w((__m128i)_cc, 2));
                    _f3 = __lsx_vfadd_s(_f3, (__m128)__lsx_vreplvei_w((__m128i)_cc, 3));
                    _f4 = __lsx_vfadd_s(_f4, (__m128)__lsx_vreplvei_w((__m128i)_cc, 0));
                    _f5 = __lsx_vfadd_s(_f5, (__m128)__lsx_vreplvei_w((__m128i)_cc, 1));
                    _f6 = __lsx_vfadd_s(_f6, (__m128)__lsx_vreplvei_w((__m128i)_cc, 2));
                    _f7 = __lsx_vfadd_s(_f7, (__m128)__lsx_vreplvei_w((__m128i)_cc, 3));
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
                _f2 = __lsx_vfmul_s(_f2, _alpha);
                _f3 = __lsx_vfmul_s(_f3, _alpha);
                _f4 = __lsx_vfmul_s(_f4, _alpha);
                _f5 = __lsx_vfmul_s(_f5, _alpha);
                _f6 = __lsx_vfmul_s(_f6, _alpha);
                _f7 = __lsx_vfmul_s(_f7, _alpha);
            }
            __lsx_vst((__m128i)_f0, p0, 0);
            __lsx_vst((__m128i)_f4, p0 + 4, 0);
            __lsx_vst((__m128i)_f1, p0 + M, 0);
            __lsx_vst((__m128i)_f5, p0 + M + 4, 0);
            __lsx_vst((__m128i)_f2, p0 + M * 2, 0);
            __lsx_vst((__m128i)_f6, p0 + M * 2 + 4, 0);
            __lsx_vst((__m128i)_f3, p0 + M * 3, 0);
            __lsx_vst((__m128i)_f7, p0 + M * 3 + 4, 0);
            p0 += M * 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp, 0);
            __m128 _f1 = (__m128)__lsx_vld(pp + 4, 0);
            __m128 _f2 = (__m128)__lsx_vld(pp + 8, 0);
            __m128 _f3 = (__m128)__lsx_vld(pp + 12, 0);
            pp += 16;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c1);
                    _f2 = __lsx_vfadd_s(_f2, _c0);
                    _f3 = __lsx_vfadd_s(_f3, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    __m128i _ci0 = __lsx_vreplgr2vr_w(((const int*)pC)[0]);
                    _ci0 = __lsx_vinsgr2vr_w(_ci0, ((const int*)(pC + c_hstep))[0], 1);
                    _ci0 = __lsx_vinsgr2vr_w(_ci0, ((const int*)(pC + c_hstep * 2))[0], 2);
                    _ci0 = __lsx_vinsgr2vr_w(_ci0, ((const int*)(pC + c_hstep * 3))[0], 3);
                    __m128i _ci1 = __lsx_vreplgr2vr_w(((const int*)(pC + c_hstep * 4))[0]);
                    _ci1 = __lsx_vinsgr2vr_w(_ci1, ((const int*)(pC + c_hstep * 5))[0], 1);
                    _ci1 = __lsx_vinsgr2vr_w(_ci1, ((const int*)(pC + c_hstep * 6))[0], 2);
                    _ci1 = __lsx_vinsgr2vr_w(_ci1, ((const int*)(pC + c_hstep * 7))[0], 3);
                    __m128i _ci2 = __lsx_vreplgr2vr_w(((const int*)pC)[1]);
                    _ci2 = __lsx_vinsgr2vr_w(_ci2, ((const int*)(pC + c_hstep))[1], 1);
                    _ci2 = __lsx_vinsgr2vr_w(_ci2, ((const int*)(pC + c_hstep * 2))[1], 2);
                    _ci2 = __lsx_vinsgr2vr_w(_ci2, ((const int*)(pC + c_hstep * 3))[1], 3);
                    __m128i _ci3 = __lsx_vreplgr2vr_w(((const int*)(pC + c_hstep * 4))[1]);
                    _ci3 = __lsx_vinsgr2vr_w(_ci3, ((const int*)(pC + c_hstep * 5))[1], 1);
                    _ci3 = __lsx_vinsgr2vr_w(_ci3, ((const int*)(pC + c_hstep * 6))[1], 2);
                    _ci3 = __lsx_vinsgr2vr_w(_ci3, ((const int*)(pC + c_hstep * 7))[1], 3);
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, (__m128)_ci0);
                        _f1 = __lsx_vfadd_s(_f1, (__m128)_ci1);
                        _f2 = __lsx_vfadd_s(_f2, (__m128)_ci2);
                        _f3 = __lsx_vfadd_s(_f3, (__m128)_ci3);
                    }
                    else
                    {
                        _f0 = __lsx_vfmadd_s((__m128)_ci0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s((__m128)_ci1, _beta, _f1);
                        _f2 = __lsx_vfmadd_s((__m128)_ci2, _beta, _f2);
                        _f3 = __lsx_vfmadd_s((__m128)_ci3, _beta, _f3);
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _cc0 = __lsx_vreplfr2vr_s(pC[0] * beta);
                    __m128 _cc1 = __lsx_vreplfr2vr_s(pC[1] * beta);
                    _f0 = __lsx_vfadd_s(_f0, _cc0);
                    _f1 = __lsx_vfadd_s(_f1, _cc0);
                    _f2 = __lsx_vfadd_s(_f2, _cc1);
                    _f3 = __lsx_vfadd_s(_f3, _cc1);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
                _f2 = __lsx_vfmul_s(_f2, _alpha);
                _f3 = __lsx_vfmul_s(_f3, _alpha);
            }
            __lsx_vst((__m128i)_f0, p0, 0);
            __lsx_vst((__m128i)_f1, p0 + 4, 0);
            __lsx_vst((__m128i)_f2, p0 + M, 0);
            __lsx_vst((__m128i)_f3, p0 + M + 4, 0);
            p0 += M * 2;
        }
        for (; jj < max_jj; jj++)
        {
            __m128i _fi0 = __lsx_vld(pp, 0);
            __m128i _fi1 = __lsx_vld(pp + 4, 0);
            pp += 8;
            __m128 _f0 = (__m128)_fi0;
            __m128 _f1 = (__m128)_fi1;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    __m128i _ci0 = __lsx_vldrepl_w(pC, 0);
                    _ci0 = __lsx_vinsgr2vr_w(_ci0, ((const int*)(pC + c_hstep))[0], 1);
                    _ci0 = __lsx_vinsgr2vr_w(_ci0, ((const int*)(pC + c_hstep * 2))[0], 2);
                    _ci0 = __lsx_vinsgr2vr_w(_ci0, ((const int*)(pC + c_hstep * 3))[0], 3);
                    __m128i _ci1 = __lsx_vldrepl_w(pC + c_hstep * 4, 0);
                    _ci1 = __lsx_vinsgr2vr_w(_ci1, ((const int*)(pC + c_hstep * 5))[0], 1);
                    _ci1 = __lsx_vinsgr2vr_w(_ci1, ((const int*)(pC + c_hstep * 6))[0], 2);
                    _ci1 = __lsx_vinsgr2vr_w(_ci1, ((const int*)(pC + c_hstep * 7))[0], 3);
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, (__m128)_ci0);
                        _f1 = __lsx_vfadd_s(_f1, (__m128)_ci1);
                    }
                    else
                    {
                        _f0 = __lsx_vfmadd_s((__m128)_ci0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s((__m128)_ci1, _beta, _f1);
                    }
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _cc = __lsx_vreplfr2vr_s(pC[0] * beta);
                    _f0 = __lsx_vfadd_s(_f0, _cc);
                    _f1 = __lsx_vfadd_s(_f1, _cc);
                    pC++;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
            }
            __lsx_vst((__m128i)_f0, p0, 0);
            __lsx_vst((__m128i)_f1, p0 + 4, 0);
            p0 += M;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* p0 = outptr + (size_t)j * M + i + ii;
        const float* pC = pC_base;

        __m128 _c = __lsx_vreplfr2vr_s(0.f);
        if (pC && broadcast_type_C == 0)
            _c = __lsx_vreplfr2vr_s(pC[0] * beta);
        if (pC && (broadcast_type_C == 1 || broadcast_type_C == 2))
        {
            _c = (__m128)__lsx_vld(pC + i + ii, 0);
            if (beta != 1.f)
                _c = __lsx_vfmul_s(_c, __lsx_vreplfr2vr_s(beta));
        }

        if (pC && broadcast_type_C == 3)
            pC += (size_t)(i + ii) * c_hstep + j;
        if (pC && broadcast_type_C == 4)
            pC += j;

        __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
        __m128 _beta = __lsx_vreplfr2vr_s(beta);
        int jj = 0;
#if __loongarch_asx
        __m256 _alpha256 = (__m256)__lasx_xvreplfr2vr_s(alpha);
        __m256 _beta256 = (__m256)__lasx_xvreplfr2vr_s(beta);
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _f00 = (__m256)__lasx_xvld(pp, 0);
            __m256 _f01 = (__m256)__lasx_xvld(pp + 8, 0);
            __m256 _f10 = (__m256)__lasx_xvld(pp + 16, 0);
            __m256 _f11 = (__m256)__lasx_xvld(pp + 24, 0);
            __m256 _f20 = (__m256)__lasx_xvld(pp + 32, 0);
            __m256 _f21 = (__m256)__lasx_xvld(pp + 40, 0);
            __m256 _f30 = (__m256)__lasx_xvld(pp + 48, 0);
            __m256 _f31 = (__m256)__lasx_xvld(pp + 56, 0);
            pp += 64;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m256 _cc = (__m256)__lasx_xvreplfr2vr_s(pC[0] * beta);
                    _f00 = __lasx_xvfadd_s(_f00, _cc);
                    _f01 = __lasx_xvfadd_s(_f01, _cc);
                    _f10 = __lasx_xvfadd_s(_f10, _cc);
                    _f11 = __lasx_xvfadd_s(_f11, _cc);
                    _f20 = __lasx_xvfadd_s(_f20, _cc);
                    _f21 = __lasx_xvfadd_s(_f21, _cc);
                    _f30 = __lasx_xvfadd_s(_f30, _cc);
                    _f31 = __lasx_xvfadd_s(_f31, _cc);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m256 _c0 = (__m256)__lasx_xvreplfr2vr_s(pC[i + ii] * beta);
                    __m256 _c1 = (__m256)__lasx_xvreplfr2vr_s(pC[i + ii + 1] * beta);
                    __m256 _c2 = (__m256)__lasx_xvreplfr2vr_s(pC[i + ii + 2] * beta);
                    __m256 _c3 = (__m256)__lasx_xvreplfr2vr_s(pC[i + ii + 3] * beta);
                    _f00 = __lasx_xvfadd_s(_f00, _c0);
                    _f01 = __lasx_xvfadd_s(_f01, _c0);
                    _f10 = __lasx_xvfadd_s(_f10, _c1);
                    _f11 = __lasx_xvfadd_s(_f11, _c1);
                    _f20 = __lasx_xvfadd_s(_f20, _c2);
                    _f21 = __lasx_xvfadd_s(_f21, _c2);
                    _f30 = __lasx_xvfadd_s(_f30, _c3);
                    _f31 = __lasx_xvfadd_s(_f31, _c3);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c00 = (__m256)__lasx_xvld(pC, 0);
                    __m256 _c01 = (__m256)__lasx_xvld(pC + 8, 0);
                    __m256 _c10 = (__m256)__lasx_xvld(pC + c_hstep, 0);
                    __m256 _c11 = (__m256)__lasx_xvld(pC + c_hstep + 8, 0);
                    __m256 _c20 = (__m256)__lasx_xvld(pC + c_hstep * 2, 0);
                    __m256 _c21 = (__m256)__lasx_xvld(pC + c_hstep * 2 + 8, 0);
                    __m256 _c30 = (__m256)__lasx_xvld(pC + c_hstep * 3, 0);
                    __m256 _c31 = (__m256)__lasx_xvld(pC + c_hstep * 3 + 8, 0);
                    if (beta == 1.f)
                    {
                        _f00 = __lasx_xvfadd_s(_f00, _c00);
                        _f01 = __lasx_xvfadd_s(_f01, _c01);
                        _f10 = __lasx_xvfadd_s(_f10, _c10);
                        _f11 = __lasx_xvfadd_s(_f11, _c11);
                        _f20 = __lasx_xvfadd_s(_f20, _c20);
                        _f21 = __lasx_xvfadd_s(_f21, _c21);
                        _f30 = __lasx_xvfadd_s(_f30, _c30);
                        _f31 = __lasx_xvfadd_s(_f31, _c31);
                    }
                    else
                    {
                        _f00 = __lasx_xvfmadd_s(_c00, _beta256, _f00);
                        _f01 = __lasx_xvfmadd_s(_c01, _beta256, _f01);
                        _f10 = __lasx_xvfmadd_s(_c10, _beta256, _f10);
                        _f11 = __lasx_xvfmadd_s(_c11, _beta256, _f11);
                        _f20 = __lasx_xvfmadd_s(_c20, _beta256, _f20);
                        _f21 = __lasx_xvfmadd_s(_c21, _beta256, _f21);
                        _f30 = __lasx_xvfmadd_s(_c30, _beta256, _f30);
                        _f31 = __lasx_xvfmadd_s(_c31, _beta256, _f31);
                    }
                    pC += 16;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c0 = (__m256)__lasx_xvld(pC, 0);
                    __m256 _c1 = (__m256)__lasx_xvld(pC + 8, 0);
                    if (beta != 1.f)
                    {
                        _c0 = __lasx_xvfmul_s(_c0, _beta256);
                        _c1 = __lasx_xvfmul_s(_c1, _beta256);
                    }
                    _f00 = __lasx_xvfadd_s(_f00, _c0);
                    _f01 = __lasx_xvfadd_s(_f01, _c1);
                    _f10 = __lasx_xvfadd_s(_f10, _c0);
                    _f11 = __lasx_xvfadd_s(_f11, _c1);
                    _f20 = __lasx_xvfadd_s(_f20, _c0);
                    _f21 = __lasx_xvfadd_s(_f21, _c1);
                    _f30 = __lasx_xvfadd_s(_f30, _c0);
                    _f31 = __lasx_xvfadd_s(_f31, _c1);
                    pC += 16;
                }
            }
            if (alpha != 1.f)
            {
                _f00 = __lasx_xvfmul_s(_f00, _alpha256);
                _f01 = __lasx_xvfmul_s(_f01, _alpha256);
                _f10 = __lasx_xvfmul_s(_f10, _alpha256);
                _f11 = __lasx_xvfmul_s(_f11, _alpha256);
                _f20 = __lasx_xvfmul_s(_f20, _alpha256);
                _f21 = __lasx_xvfmul_s(_f21, _alpha256);
                _f30 = __lasx_xvfmul_s(_f30, _alpha256);
                _f31 = __lasx_xvfmul_s(_f31, _alpha256);
            }

            __m256i _tmp0 = __lasx_xvilvl_w((__m256i)_f10, (__m256i)_f00);
            __m256i _tmp1 = __lasx_xvilvh_w((__m256i)_f10, (__m256i)_f00);
            __m256i _tmp2 = __lasx_xvilvl_w((__m256i)_f30, (__m256i)_f20);
            __m256i _tmp3 = __lasx_xvilvh_w((__m256i)_f30, (__m256i)_f20);
            __m256i _r0 = __lasx_xvilvl_d(_tmp2, _tmp0);
            __m256i _r1 = __lasx_xvilvh_d(_tmp2, _tmp0);
            __m256i _r2 = __lasx_xvilvl_d(_tmp3, _tmp1);
            __m256i _r3 = __lasx_xvilvh_d(_tmp3, _tmp1);
            __lsx_vst(__lasx_extract_128_lo(_r0), p0, 0);
            __lsx_vst(__lasx_extract_128_lo(_r1), p0 + M, 0);
            __lsx_vst(__lasx_extract_128_lo(_r2), p0 + M * 2, 0);
            __lsx_vst(__lasx_extract_128_lo(_r3), p0 + M * 3, 0);
            __lsx_vst(__lasx_extract_128_hi(_r0), p0 + M * 4, 0);
            __lsx_vst(__lasx_extract_128_hi(_r1), p0 + M * 5, 0);
            __lsx_vst(__lasx_extract_128_hi(_r2), p0 + M * 6, 0);
            __lsx_vst(__lasx_extract_128_hi(_r3), p0 + M * 7, 0);

            _tmp0 = __lasx_xvilvl_w((__m256i)_f11, (__m256i)_f01);
            _tmp1 = __lasx_xvilvh_w((__m256i)_f11, (__m256i)_f01);
            _tmp2 = __lasx_xvilvl_w((__m256i)_f31, (__m256i)_f21);
            _tmp3 = __lasx_xvilvh_w((__m256i)_f31, (__m256i)_f21);
            _r0 = __lasx_xvilvl_d(_tmp2, _tmp0);
            _r1 = __lasx_xvilvh_d(_tmp2, _tmp0);
            _r2 = __lasx_xvilvl_d(_tmp3, _tmp1);
            _r3 = __lasx_xvilvh_d(_tmp3, _tmp1);
            __lsx_vst(__lasx_extract_128_lo(_r0), p0 + M * 8, 0);
            __lsx_vst(__lasx_extract_128_lo(_r1), p0 + M * 9, 0);
            __lsx_vst(__lasx_extract_128_lo(_r2), p0 + M * 10, 0);
            __lsx_vst(__lasx_extract_128_lo(_r3), p0 + M * 11, 0);
            __lsx_vst(__lasx_extract_128_hi(_r0), p0 + M * 12, 0);
            __lsx_vst(__lasx_extract_128_hi(_r1), p0 + M * 13, 0);
            __lsx_vst(__lasx_extract_128_hi(_r2), p0 + M * 14, 0);
            __lsx_vst(__lasx_extract_128_hi(_r3), p0 + M * 15, 0);
            p0 += M * 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = (__m256)__lasx_xvld(pp, 0);
            __m256 _f1 = (__m256)__lasx_xvld(pp + 8, 0);
            __m256 _f2 = (__m256)__lasx_xvld(pp + 16, 0);
            __m256 _f3 = (__m256)__lasx_xvld(pp + 24, 0);
            pp += 32;
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m256 _c0 = (__m256)__lasx_xvreplfr2vr_s(pC[0] * beta);
                    _f0 = __lasx_xvfadd_s(_f0, _c0);
                    _f1 = __lasx_xvfadd_s(_f1, _c0);
                    _f2 = __lasx_xvfadd_s(_f2, _c0);
                    _f3 = __lasx_xvfadd_s(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lasx_xvfadd_s(_f0, (__m256)__lasx_xvreplfr2vr_s(pC[i + ii] * beta));
                    _f1 = __lasx_xvfadd_s(_f1, (__m256)__lasx_xvreplfr2vr_s(pC[i + ii + 1] * beta));
                    _f2 = __lasx_xvfadd_s(_f2, (__m256)__lasx_xvreplfr2vr_s(pC[i + ii + 2] * beta));
                    _f3 = __lasx_xvfadd_s(_f3, (__m256)__lasx_xvreplfr2vr_s(pC[i + ii + 3] * beta));
                }
                if (broadcast_type_C == 3)
                {
                    if (beta == 1.f)
                    {
                        _f0 = __lasx_xvfadd_s(_f0, (__m256)__lasx_xvld(pC, 0));
                        _f1 = __lasx_xvfadd_s(_f1, (__m256)__lasx_xvld(pC + c_hstep, 0));
                        _f2 = __lasx_xvfadd_s(_f2, (__m256)__lasx_xvld(pC + c_hstep * 2, 0));
                        _f3 = __lasx_xvfadd_s(_f3, (__m256)__lasx_xvld(pC + c_hstep * 3, 0));
                    }
                    else
                    {
                        _f0 = __lasx_xvfmadd_s((__m256)__lasx_xvld(pC, 0), _beta256, _f0);
                        _f1 = __lasx_xvfmadd_s((__m256)__lasx_xvld(pC + c_hstep, 0), _beta256, _f1);
                        _f2 = __lasx_xvfmadd_s((__m256)__lasx_xvld(pC + c_hstep * 2, 0), _beta256, _f2);
                        _f3 = __lasx_xvfmadd_s((__m256)__lasx_xvld(pC + c_hstep * 3, 0), _beta256, _f3);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c4 = (__m256)__lasx_xvld(pC, 0);
                    if (beta != 1.f)
                        _c4 = __lasx_xvfmul_s(_c4, _beta256);
                    _f0 = __lasx_xvfadd_s(_f0, _c4);
                    _f1 = __lasx_xvfadd_s(_f1, _c4);
                    _f2 = __lasx_xvfadd_s(_f2, _c4);
                    _f3 = __lasx_xvfadd_s(_f3, _c4);
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lasx_xvfmul_s(_f0, _alpha256);
                _f1 = __lasx_xvfmul_s(_f1, _alpha256);
                _f2 = __lasx_xvfmul_s(_f2, _alpha256);
                _f3 = __lasx_xvfmul_s(_f3, _alpha256);
            }

            __m256i _tmp0 = __lasx_xvilvl_w((__m256i)_f1, (__m256i)_f0);
            __m256i _tmp1 = __lasx_xvilvh_w((__m256i)_f1, (__m256i)_f0);
            __m256i _tmp2 = __lasx_xvilvl_w((__m256i)_f3, (__m256i)_f2);
            __m256i _tmp3 = __lasx_xvilvh_w((__m256i)_f3, (__m256i)_f2);
            __m256i _r0 = __lasx_xvilvl_d(_tmp2, _tmp0);
            __m256i _r1 = __lasx_xvilvh_d(_tmp2, _tmp0);
            __m256i _r2 = __lasx_xvilvl_d(_tmp3, _tmp1);
            __m256i _r3 = __lasx_xvilvh_d(_tmp3, _tmp1);

            __lsx_vst(__lasx_extract_128_lo(_r0), p0, 0);
            __lsx_vst(__lasx_extract_128_lo(_r1), p0 + M, 0);
            __lsx_vst(__lasx_extract_128_lo(_r2), p0 + M * 2, 0);
            __lsx_vst(__lasx_extract_128_lo(_r3), p0 + M * 3, 0);
            __lsx_vst(__lasx_extract_128_hi(_r0), p0 + M * 4, 0);
            __lsx_vst(__lasx_extract_128_hi(_r1), p0 + M * 5, 0);
            __lsx_vst(__lasx_extract_128_hi(_r2), p0 + M * 6, 0);
            __lsx_vst(__lasx_extract_128_hi(_r3), p0 + M * 7, 0);
            p0 += M * 8;
        }
#endif // __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f00 = (__m128)__lsx_vld(pp, 0);
            __m128 _f01 = (__m128)__lsx_vld(pp + 4, 0);
            __m128 _f10 = (__m128)__lsx_vld(pp + 8, 0);
            __m128 _f11 = (__m128)__lsx_vld(pp + 12, 0);
            __m128 _f20 = (__m128)__lsx_vld(pp + 16, 0);
            __m128 _f21 = (__m128)__lsx_vld(pp + 20, 0);
            __m128 _f30 = (__m128)__lsx_vld(pp + 24, 0);
            __m128 _f31 = (__m128)__lsx_vld(pp + 28, 0);
            pp += 32;
            transpose4x4_ps(_f00, _f10, _f20, _f30);
            transpose4x4_ps(_f01, _f11, _f21, _f31);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f00 = __lsx_vfadd_s(_f00, _c);
                    _f10 = __lsx_vfadd_s(_f10, _c);
                    _f20 = __lsx_vfadd_s(_f20, _c);
                    _f30 = __lsx_vfadd_s(_f30, _c);
                    _f01 = __lsx_vfadd_s(_f01, _c);
                    _f11 = __lsx_vfadd_s(_f11, _c);
                    _f21 = __lsx_vfadd_s(_f21, _c);
                    _f31 = __lsx_vfadd_s(_f31, _c);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c00 = (__m128)__lsx_vld(pC, 0);
                    __m128 _c01 = (__m128)__lsx_vld(pC + 4, 0);
                    __m128 _c10 = (__m128)__lsx_vld(pC + c_hstep, 0);
                    __m128 _c11 = (__m128)__lsx_vld(pC + c_hstep + 4, 0);
                    __m128 _c20 = (__m128)__lsx_vld(pC + c_hstep * 2, 0);
                    __m128 _c21 = (__m128)__lsx_vld(pC + c_hstep * 2 + 4, 0);
                    __m128 _c30 = (__m128)__lsx_vld(pC + c_hstep * 3, 0);
                    __m128 _c31 = (__m128)__lsx_vld(pC + c_hstep * 3 + 4, 0);
                    transpose4x4_ps(_c00, _c10, _c20, _c30);
                    transpose4x4_ps(_c01, _c11, _c21, _c31);
                    if (beta == 1.f)
                    {
                        _f00 = __lsx_vfadd_s(_f00, _c00);
                        _f10 = __lsx_vfadd_s(_f10, _c10);
                        _f20 = __lsx_vfadd_s(_f20, _c20);
                        _f30 = __lsx_vfadd_s(_f30, _c30);
                        _f01 = __lsx_vfadd_s(_f01, _c01);
                        _f11 = __lsx_vfadd_s(_f11, _c11);
                        _f21 = __lsx_vfadd_s(_f21, _c21);
                        _f31 = __lsx_vfadd_s(_f31, _c31);
                    }
                    else
                    {
                        _f00 = __lsx_vfmadd_s(_c00, _beta, _f00);
                        _f10 = __lsx_vfmadd_s(_c10, _beta, _f10);
                        _f20 = __lsx_vfmadd_s(_c20, _beta, _f20);
                        _f30 = __lsx_vfmadd_s(_c30, _beta, _f30);
                        _f01 = __lsx_vfmadd_s(_c01, _beta, _f01);
                        _f11 = __lsx_vfmadd_s(_c11, _beta, _f11);
                        _f21 = __lsx_vfmadd_s(_c21, _beta, _f21);
                        _f31 = __lsx_vfmadd_s(_c31, _beta, _f31);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _cc0 = (__m128)__lsx_vld(pC, 0);
                    __m128 _cc1 = (__m128)__lsx_vld(pC + 4, 0);
                    if (beta != 1.f)
                    {
                        _cc0 = __lsx_vfmul_s(_cc0, _beta);
                        _cc1 = __lsx_vfmul_s(_cc1, _beta);
                    }
                    _f00 = __lsx_vfadd_s(_f00, (__m128)__lsx_vreplvei_w((__m128i)_cc0, 0));
                    _f10 = __lsx_vfadd_s(_f10, (__m128)__lsx_vreplvei_w((__m128i)_cc0, 1));
                    _f20 = __lsx_vfadd_s(_f20, (__m128)__lsx_vreplvei_w((__m128i)_cc0, 2));
                    _f30 = __lsx_vfadd_s(_f30, (__m128)__lsx_vreplvei_w((__m128i)_cc0, 3));
                    _f01 = __lsx_vfadd_s(_f01, (__m128)__lsx_vreplvei_w((__m128i)_cc1, 0));
                    _f11 = __lsx_vfadd_s(_f11, (__m128)__lsx_vreplvei_w((__m128i)_cc1, 1));
                    _f21 = __lsx_vfadd_s(_f21, (__m128)__lsx_vreplvei_w((__m128i)_cc1, 2));
                    _f31 = __lsx_vfadd_s(_f31, (__m128)__lsx_vreplvei_w((__m128i)_cc1, 3));
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                _f00 = __lsx_vfmul_s(_f00, _alpha);
                _f10 = __lsx_vfmul_s(_f10, _alpha);
                _f20 = __lsx_vfmul_s(_f20, _alpha);
                _f30 = __lsx_vfmul_s(_f30, _alpha);
                _f01 = __lsx_vfmul_s(_f01, _alpha);
                _f11 = __lsx_vfmul_s(_f11, _alpha);
                _f21 = __lsx_vfmul_s(_f21, _alpha);
                _f31 = __lsx_vfmul_s(_f31, _alpha);
            }
            __lsx_vst((__m128i)_f00, p0, 0);
            __lsx_vst((__m128i)_f10, p0 + M, 0);
            __lsx_vst((__m128i)_f20, p0 + M * 2, 0);
            __lsx_vst((__m128i)_f30, p0 + M * 3, 0);
            __lsx_vst((__m128i)_f01, p0 + M * 4, 0);
            __lsx_vst((__m128i)_f11, p0 + M * 5, 0);
            __lsx_vst((__m128i)_f21, p0 + M * 6, 0);
            __lsx_vst((__m128i)_f31, p0 + M * 7, 0);
            p0 += M * 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp, 0);
            __m128 _f1 = (__m128)__lsx_vld(pp + 4, 0);
            __m128 _f2 = (__m128)__lsx_vld(pp + 8, 0);
            __m128 _f3 = (__m128)__lsx_vld(pp + 12, 0);
            pp += 16;
            transpose4x4_ps(_f0, _f1, _f2, _f3);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                    _f2 = __lsx_vfadd_s(_f2, _c);
                    _f3 = __lsx_vfadd_s(_f3, _c);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = (__m128)__lsx_vld(pC, 0);
                    __m128 _c1 = (__m128)__lsx_vld(pC + c_hstep, 0);
                    __m128 _c2 = (__m128)__lsx_vld(pC + c_hstep * 2, 0);
                    __m128 _c3 = (__m128)__lsx_vld(pC + c_hstep * 3, 0);
                    transpose4x4_ps(_c0, _c1, _c2, _c3);
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                        _f1 = __lsx_vfadd_s(_f1, _c1);
                        _f2 = __lsx_vfadd_s(_f2, _c2);
                        _f3 = __lsx_vfadd_s(_f3, _c3);
                    }
                    else
                    {
                        _f0 = __lsx_vfmadd_s(_c0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s(_c1, _beta, _f1);
                        _f2 = __lsx_vfmadd_s(_c2, _beta, _f2);
                        _f3 = __lsx_vfmadd_s(_c3, _beta, _f3);
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c4 = (__m128)__lsx_vld(pC, 0);
                    if (beta != 1.f)
                        _c4 = __lsx_vfmul_s(_c4, _beta);
                    _f0 = __lsx_vfadd_s(_f0, (__m128)__lsx_vreplvei_w((__m128i)_c4, 0));
                    _f1 = __lsx_vfadd_s(_f1, (__m128)__lsx_vreplvei_w((__m128i)_c4, 1));
                    _f2 = __lsx_vfadd_s(_f2, (__m128)__lsx_vreplvei_w((__m128i)_c4, 2));
                    _f3 = __lsx_vfadd_s(_f3, (__m128)__lsx_vreplvei_w((__m128i)_c4, 3));
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
                _f2 = __lsx_vfmul_s(_f2, _alpha);
                _f3 = __lsx_vfmul_s(_f3, _alpha);
            }
            __lsx_vst((__m128i)_f0, p0, 0);
            __lsx_vst((__m128i)_f1, p0 + M, 0);
            __lsx_vst((__m128i)_f2, p0 + M * 2, 0);
            __lsx_vst((__m128i)_f3, p0 + M * 3, 0);
            p0 += M * 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128i _r0 = __lsx_vldrepl_d(pp, 0);
            __m128i _r1 = __lsx_vldrepl_d(pp + 2, 0);
            __m128i _r2 = __lsx_vldrepl_d(pp + 4, 0);
            __m128i _r3 = __lsx_vldrepl_d(pp + 6, 0);
            pp += 8;
            __m128i _t0 = __lsx_vilvl_w(_r1, _r0);
            __m128i _t1 = __lsx_vilvl_w(_r3, _r2);
            __m128 _f0 = (__m128)__lsx_vilvl_d(_t1, _t0);
            __m128 _f1 = (__m128)__lsx_vilvh_d(_t1, _t0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                }
                if (broadcast_type_C == 3)
                {
                    _r0 = __lsx_vldrepl_d(pC, 0);
                    _r1 = __lsx_vldrepl_d(pC + c_hstep, 0);
                    _r2 = __lsx_vldrepl_d(pC + c_hstep * 2, 0);
                    _r3 = __lsx_vldrepl_d(pC + c_hstep * 3, 0);
                    _t0 = __lsx_vilvl_w(_r1, _r0);
                    _t1 = __lsx_vilvl_w(_r3, _r2);
                    __m128 _cc0 = (__m128)__lsx_vilvl_d(_t1, _t0);
                    __m128 _cc1 = (__m128)__lsx_vilvh_d(_t1, _t0);
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, _cc0);
                        _f1 = __lsx_vfadd_s(_f1, _cc1);
                    }
                    else
                    {
                        _f0 = __lsx_vfmadd_s(_cc0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s(_cc1, _beta, _f1);
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(pC[0] * beta));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vreplfr2vr_s(pC[1] * beta));
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
            }
            __lsx_vst((__m128i)_f0, p0, 0);
            __lsx_vst((__m128i)_f1, p0 + M, 0);
            p0 += M * 2;
        }
        for (; jj < max_jj; jj++)
        {
            __m128i _fi = __lsx_vld(pp, 0);
            pp += 4;
            __m128 _f = (__m128)_fi;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f = __lsx_vfadd_s(_f, _c);
                if (broadcast_type_C == 3)
                {
                    __m128i _ci = __lsx_vldrepl_w(pC, 0);
                    _ci = __lsx_vinsgr2vr_w(_ci, ((const int*)(pC + c_hstep))[0], 1);
                    _ci = __lsx_vinsgr2vr_w(_ci, ((const int*)(pC + c_hstep * 2))[0], 2);
                    _ci = __lsx_vinsgr2vr_w(_ci, ((const int*)(pC + c_hstep * 3))[0], 3);
                    if (beta == 1.f)
                        _f = __lsx_vfadd_s(_f, (__m128)_ci);
                    else
                        _f = __lsx_vfmadd_s((__m128)_ci, _beta, _f);
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c4 = __lsx_vreplfr2vr_s(pC[0]);
                    if (beta == 1.f)
                        _f = __lsx_vfadd_s(_f, _c4);
                    else
                        _f = __lsx_vfmadd_s(_c4, _beta, _f);
                    pC++;
                }
            }
            if (alpha != 1.f)
                _f = __lsx_vfmul_s(_f, _alpha);
            __lsx_vst((__m128i)_f, p0, 0);
            p0 += M;
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* p0 = outptr + (size_t)j * M + i + ii;
        const float* pC = pC_base;

        float c0 = 0.f;
        float c1 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
                c1 = c0;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[i + ii] * beta;
                c1 = pC[i + ii + 1] * beta;
            }
            if (broadcast_type_C == 3)
            {
                pC += (size_t)(i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
                pC += j;
        }

        int jj = 0;
#if __loongarch_sx
        __m128 _c = __lsx_vreplfr2vr_s(0.f);
        if (pC && broadcast_type_C == 0)
            _c = __lsx_vreplfr2vr_s(c0);
        if (pC && (broadcast_type_C == 1 || broadcast_type_C == 2))
        {
            _c = (__m128)__lsx_vldrepl_d(pC + i + ii, 0);
            if (beta != 1.f)
                _c = __lsx_vfmul_s(_c, __lsx_vreplfr2vr_s(beta));
        }
        __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
        __m128 _beta = __lsx_vreplfr2vr_s(beta);
#if __loongarch_asx
        __m256 _alpha256 = (__m256)__lasx_xvreplfr2vr_s(alpha);
        __m256 _beta256 = (__m256)__lasx_xvreplfr2vr_s(beta);
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _f00 = (__m256)__lasx_xvld(pp, 0);
            __m256 _f01 = (__m256)__lasx_xvld(pp + 8, 0);
            __m256 _f10 = (__m256)__lasx_xvld(pp + 16, 0);
            __m256 _f11 = (__m256)__lasx_xvld(pp + 24, 0);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m256 _cc = (__m256)__lasx_xvreplfr2vr_s(c0);
                    _f00 = __lasx_xvfadd_s(_f00, _cc);
                    _f01 = __lasx_xvfadd_s(_f01, _cc);
                    _f10 = __lasx_xvfadd_s(_f10, _cc);
                    _f11 = __lasx_xvfadd_s(_f11, _cc);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m256 _c0 = (__m256)__lasx_xvreplfr2vr_s(c0);
                    __m256 _c1 = (__m256)__lasx_xvreplfr2vr_s(c1);
                    _f00 = __lasx_xvfadd_s(_f00, _c0);
                    _f01 = __lasx_xvfadd_s(_f01, _c0);
                    _f10 = __lasx_xvfadd_s(_f10, _c1);
                    _f11 = __lasx_xvfadd_s(_f11, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c00 = (__m256)__lasx_xvld(pC, 0);
                    __m256 _c01 = (__m256)__lasx_xvld(pC + 8, 0);
                    __m256 _c10 = (__m256)__lasx_xvld(pC + c_hstep, 0);
                    __m256 _c11 = (__m256)__lasx_xvld(pC + c_hstep + 8, 0);
                    if (beta == 1.f)
                    {
                        _f00 = __lasx_xvfadd_s(_f00, _c00);
                        _f01 = __lasx_xvfadd_s(_f01, _c01);
                        _f10 = __lasx_xvfadd_s(_f10, _c10);
                        _f11 = __lasx_xvfadd_s(_f11, _c11);
                    }
                    else
                    {
                        _f00 = __lasx_xvfmadd_s(_c00, _beta256, _f00);
                        _f01 = __lasx_xvfmadd_s(_c01, _beta256, _f01);
                        _f10 = __lasx_xvfmadd_s(_c10, _beta256, _f10);
                        _f11 = __lasx_xvfmadd_s(_c11, _beta256, _f11);
                    }
                    pC += 16;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c0 = (__m256)__lasx_xvld(pC, 0);
                    __m256 _c1 = (__m256)__lasx_xvld(pC + 8, 0);
                    if (beta != 1.f)
                    {
                        _c0 = __lasx_xvfmul_s(_c0, _beta256);
                        _c1 = __lasx_xvfmul_s(_c1, _beta256);
                    }
                    _f00 = __lasx_xvfadd_s(_f00, _c0);
                    _f01 = __lasx_xvfadd_s(_f01, _c1);
                    _f10 = __lasx_xvfadd_s(_f10, _c0);
                    _f11 = __lasx_xvfadd_s(_f11, _c1);
                    pC += 16;
                }
            }
            if (alpha != 1.f)
            {
                _f00 = __lasx_xvfmul_s(_f00, _alpha256);
                _f01 = __lasx_xvfmul_s(_f01, _alpha256);
                _f10 = __lasx_xvfmul_s(_f10, _alpha256);
                _f11 = __lasx_xvfmul_s(_f11, _alpha256);
            }
            __m256i _tmp0 = __lasx_xvilvl_w((__m256i)_f10, (__m256i)_f00);
            __m256i _tmp1 = __lasx_xvilvh_w((__m256i)_f10, (__m256i)_f00);
            __lasx_xvstelm_d(_tmp0, p0, 0, 0);
            __lasx_xvstelm_d(_tmp0, p0 + M, 0, 1);
            __lasx_xvstelm_d(_tmp1, p0 + M * 2, 0, 0);
            __lasx_xvstelm_d(_tmp1, p0 + M * 3, 0, 1);
            __lasx_xvstelm_d(_tmp0, p0 + M * 4, 0, 2);
            __lasx_xvstelm_d(_tmp0, p0 + M * 5, 0, 3);
            __lasx_xvstelm_d(_tmp1, p0 + M * 6, 0, 2);
            __lasx_xvstelm_d(_tmp1, p0 + M * 7, 0, 3);
            _tmp0 = __lasx_xvilvl_w((__m256i)_f11, (__m256i)_f01);
            _tmp1 = __lasx_xvilvh_w((__m256i)_f11, (__m256i)_f01);
            __lasx_xvstelm_d(_tmp0, p0 + M * 8, 0, 0);
            __lasx_xvstelm_d(_tmp0, p0 + M * 9, 0, 1);
            __lasx_xvstelm_d(_tmp1, p0 + M * 10, 0, 0);
            __lasx_xvstelm_d(_tmp1, p0 + M * 11, 0, 1);
            __lasx_xvstelm_d(_tmp0, p0 + M * 12, 0, 2);
            __lasx_xvstelm_d(_tmp0, p0 + M * 13, 0, 3);
            __lasx_xvstelm_d(_tmp1, p0 + M * 14, 0, 2);
            __lasx_xvstelm_d(_tmp1, p0 + M * 15, 0, 3);
            p0 += M * 16;
            pp += 32;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = (__m256)__lasx_xvld(pp, 0);
            __m256 _f1 = (__m256)__lasx_xvld(pp + 8, 0);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m256 _c0 = (__m256)__lasx_xvreplfr2vr_s(c0);
                    _f0 = __lasx_xvfadd_s(_f0, _c0);
                    _f1 = __lasx_xvfadd_s(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lasx_xvfadd_s(_f0, (__m256)__lasx_xvreplfr2vr_s(c0));
                    _f1 = __lasx_xvfadd_s(_f1, (__m256)__lasx_xvreplfr2vr_s(c1));
                }
                if (broadcast_type_C == 3)
                {
                    if (beta == 1.f)
                    {
                        _f0 = __lasx_xvfadd_s(_f0, (__m256)__lasx_xvld(pC, 0));
                        _f1 = __lasx_xvfadd_s(_f1, (__m256)__lasx_xvld(pC + c_hstep, 0));
                    }
                    else
                    {
                        _f0 = __lasx_xvfmadd_s((__m256)__lasx_xvld(pC, 0), _beta256, _f0);
                        _f1 = __lasx_xvfmadd_s((__m256)__lasx_xvld(pC + c_hstep, 0), _beta256, _f1);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c4 = (__m256)__lasx_xvld(pC, 0);
                    if (beta != 1.f)
                        _c4 = __lasx_xvfmul_s(_c4, _beta256);
                    _f0 = __lasx_xvfadd_s(_f0, _c4);
                    _f1 = __lasx_xvfadd_s(_f1, _c4);
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lasx_xvfmul_s(_f0, _alpha256);
                _f1 = __lasx_xvfmul_s(_f1, _alpha256);
            }

            __m256i _tmp0 = __lasx_xvilvl_w((__m256i)_f1, (__m256i)_f0);
            __m256i _tmp1 = __lasx_xvilvh_w((__m256i)_f1, (__m256i)_f0);
            __lasx_xvstelm_d(_tmp0, p0, 0, 0);
            __lasx_xvstelm_d(_tmp0, p0 + M, 0, 1);
            __lasx_xvstelm_d(_tmp1, p0 + M * 2, 0, 0);
            __lasx_xvstelm_d(_tmp1, p0 + M * 3, 0, 1);
            __lasx_xvstelm_d(_tmp0, p0 + M * 4, 0, 2);
            __lasx_xvstelm_d(_tmp0, p0 + M * 5, 0, 3);
            __lasx_xvstelm_d(_tmp1, p0 + M * 6, 0, 2);
            __lasx_xvstelm_d(_tmp1, p0 + M * 7, 0, 3);
            p0 += M * 8;
            pp += 16;
        }
#endif // __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f00 = (__m128)__lsx_vld(pp, 0);
            __m128 _f01 = (__m128)__lsx_vld(pp + 4, 0);
            __m128 _f10 = (__m128)__lsx_vld(pp + 8, 0);
            __m128 _f11 = (__m128)__lsx_vld(pp + 12, 0);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _cc = __lsx_vreplfr2vr_s(c0);
                    _f00 = __lsx_vfadd_s(_f00, _cc);
                    _f01 = __lsx_vfadd_s(_f01, _cc);
                    _f10 = __lsx_vfadd_s(_f10, _cc);
                    _f11 = __lsx_vfadd_s(_f11, _cc);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(c0);
                    __m128 _c1 = __lsx_vreplfr2vr_s(c1);
                    _f00 = __lsx_vfadd_s(_f00, _c0);
                    _f01 = __lsx_vfadd_s(_f01, _c0);
                    _f10 = __lsx_vfadd_s(_f10, _c1);
                    _f11 = __lsx_vfadd_s(_f11, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c00 = (__m128)__lsx_vld(pC, 0);
                    __m128 _c01 = (__m128)__lsx_vld(pC + 4, 0);
                    __m128 _c10 = (__m128)__lsx_vld(pC + c_hstep, 0);
                    __m128 _c11 = (__m128)__lsx_vld(pC + c_hstep + 4, 0);
                    if (beta == 1.f)
                    {
                        _f00 = __lsx_vfadd_s(_f00, _c00);
                        _f01 = __lsx_vfadd_s(_f01, _c01);
                        _f10 = __lsx_vfadd_s(_f10, _c10);
                        _f11 = __lsx_vfadd_s(_f11, _c11);
                    }
                    else
                    {
                        _f00 = __lsx_vfmadd_s(_c00, _beta, _f00);
                        _f01 = __lsx_vfmadd_s(_c01, _beta, _f01);
                        _f10 = __lsx_vfmadd_s(_c10, _beta, _f10);
                        _f11 = __lsx_vfmadd_s(_c11, _beta, _f11);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c0 = (__m128)__lsx_vld(pC, 0);
                    __m128 _c1 = (__m128)__lsx_vld(pC + 4, 0);
                    if (beta != 1.f)
                    {
                        _c0 = __lsx_vfmul_s(_c0, _beta);
                        _c1 = __lsx_vfmul_s(_c1, _beta);
                    }
                    _f00 = __lsx_vfadd_s(_f00, _c0);
                    _f01 = __lsx_vfadd_s(_f01, _c1);
                    _f10 = __lsx_vfadd_s(_f10, _c0);
                    _f11 = __lsx_vfadd_s(_f11, _c1);
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                _f00 = __lsx_vfmul_s(_f00, _alpha);
                _f01 = __lsx_vfmul_s(_f01, _alpha);
                _f10 = __lsx_vfmul_s(_f10, _alpha);
                _f11 = __lsx_vfmul_s(_f11, _alpha);
            }
            __m128i _tmp0 = __lsx_vilvl_w((__m128i)_f10, (__m128i)_f00);
            __m128i _tmp1 = __lsx_vilvh_w((__m128i)_f10, (__m128i)_f00);
            __lsx_vstelm_d(_tmp0, p0, 0, 0);
            __lsx_vstelm_d(_tmp0, p0 + M, 0, 1);
            __lsx_vstelm_d(_tmp1, p0 + M * 2, 0, 0);
            __lsx_vstelm_d(_tmp1, p0 + M * 3, 0, 1);
            _tmp0 = __lsx_vilvl_w((__m128i)_f11, (__m128i)_f01);
            _tmp1 = __lsx_vilvh_w((__m128i)_f11, (__m128i)_f01);
            __lsx_vstelm_d(_tmp0, p0 + M * 4, 0, 0);
            __lsx_vstelm_d(_tmp0, p0 + M * 5, 0, 1);
            __lsx_vstelm_d(_tmp1, p0 + M * 6, 0, 0);
            __lsx_vstelm_d(_tmp1, p0 + M * 7, 0, 1);
            p0 += M * 8;
            pp += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp, 0);
            __m128 _f1 = (__m128)__lsx_vld(pp + 4, 0);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _cc = __lsx_vreplfr2vr_s(c0);
                    _f0 = __lsx_vfadd_s(_f0, _cc);
                    _f1 = __lsx_vfadd_s(_f1, _cc);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(c0));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vreplfr2vr_s(c1));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c0 = (__m128)__lsx_vld(pC, 0);
                    __m128 _c1 = (__m128)__lsx_vld(pC + c_hstep, 0);
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                        _f1 = __lsx_vfadd_s(_f1, _c1);
                    }
                    else
                    {
                        _f0 = __lsx_vfmadd_s(_c0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s(_c1, _beta, _f1);
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c4 = (__m128)__lsx_vld(pC, 0);
                    if (beta != 1.f)
                        _c4 = __lsx_vfmul_s(_c4, _beta);
                    _f0 = __lsx_vfadd_s(_f0, _c4);
                    _f1 = __lsx_vfadd_s(_f1, _c4);
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
            }
            __m128i _tmp0 = __lsx_vilvl_w((__m128i)_f1, (__m128i)_f0);
            __m128i _tmp1 = __lsx_vilvh_w((__m128i)_f1, (__m128i)_f0);
            __lsx_vstelm_d(_tmp0, p0, 0, 0);
            __lsx_vstelm_d(_tmp0, p0 + M, 0, 1);
            __lsx_vstelm_d(_tmp1, p0 + M * 2, 0, 0);
            __lsx_vstelm_d(_tmp1, p0 + M * 3, 0, 1);
            p0 += M * 4;
            pp += 8;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _f = (__m128)__lsx_vshuf4i_w(__lsx_vld(pp, 0), _LSX_SHUFFLE(3, 1, 2, 0));
            __m128i _r0;
            __m128i _r1;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f = __lsx_vfadd_s(_f, _c);
                if (broadcast_type_C == 3)
                {
                    _r0 = __lsx_vldrepl_d(pC, 0);
                    _r1 = __lsx_vldrepl_d(pC + c_hstep, 0);
                    __m128 _cc = (__m128)__lsx_vilvl_w(_r1, _r0);
                    if (beta == 1.f)
                        _f = __lsx_vfadd_s(_f, _cc);
                    else
                        _f = __lsx_vfmadd_s(_cc, _beta, _f);
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    __m128i _cc = __lsx_vldrepl_d(pC, 0);
                    _cc = __lsx_vilvl_w(_cc, _cc);
                    if (beta == 1.f)
                        _f = __lsx_vfadd_s(_f, (__m128)_cc);
                    else
                        _f = __lsx_vfmadd_s((__m128)_cc, _beta, _f);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
                _f = __lsx_vfmul_s(_f, _alpha);
            __lsx_vstelm_d((__m128i)_f, p0, 0, 0);
            __lsx_vstelm_d((__m128i)_f, p0 + M, 0, 1);
            p0 += M * 2;
            pp += 4;
        }
        for (; jj < max_jj; jj++)
        {
            __m128i _fi = __lsx_vldrepl_d(pp, 0);
            __m128 _f = (__m128)_fi;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f = __lsx_vfadd_s(_f, _c);
                if (broadcast_type_C == 3)
                {
                    __m128i _ci = __lsx_vldrepl_w(pC, 0);
                    _ci = __lsx_vinsgr2vr_w(_ci, ((const int*)(pC + c_hstep))[0], 1);
                    if (beta == 1.f)
                        _f = __lsx_vfadd_s(_f, (__m128)_ci);
                    else
                        _f = __lsx_vfmadd_s((__m128)_ci, _beta, _f);
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c4 = __lsx_vreplfr2vr_s(pC[0]);
                    if (beta == 1.f)
                        _f = __lsx_vfadd_s(_f, _c4);
                    else
                        _f = __lsx_vfmadd_s(_c4, _beta, _f);
                    pC++;
                }
            }
            if (alpha != 1.f)
                _f = __lsx_vfmul_s(_f, _alpha);
            __lsx_vstelm_d((__m128i)_f, p0, 0, 0);
            p0 += M;
            pp += 2;
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            float f00 = pp[0];
            float f01 = pp[1];
            float f10 = pp[2];
            float f11 = pp[3];
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    f00 += c0;
                    f01 += c0;
                    f10 += c0;
                    f11 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f00 += c0;
                    f01 += c0;
                    f10 += c1;
                    f11 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    if (beta == 1.f)
                    {
                        f00 += pC[0];
                        f01 += pC[1];
                        f10 += pC[c_hstep];
                        f11 += pC[c_hstep + 1];
                    }
                    else
                    {
                        f00 += pC[0] * beta;
                        f01 += pC[1] * beta;
                        f10 += pC[c_hstep] * beta;
                        f11 += pC[c_hstep + 1] * beta;
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    const float cc0 = beta == 1.f ? pC[0] : pC[0] * beta;
                    const float cc1 = beta == 1.f ? pC[1] : pC[1] * beta;
                    f00 += cc0;
                    f01 += cc1;
                    f10 += cc0;
                    f11 += cc1;
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                f00 *= alpha;
                f01 *= alpha;
                f10 *= alpha;
                f11 *= alpha;
            }
            p0[0] = f00;
            p0[1] = f10;
            p0[M] = f01;
            p0[M + 1] = f11;
            p0 += M * 2;
            pp += 4;
        }
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[0];
            float f1 = pp[1];
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                    f1 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    if (beta == 1.f)
                    {
                        f0 += pC[0];
                        f1 += pC[c_hstep];
                    }
                    else
                    {
                        f0 += pC[0] * beta;
                        f1 += pC[c_hstep] * beta;
                    }
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    float c = beta == 1.f ? pC[0] : pC[0] * beta;
                    f0 += c;
                    f1 += c;
                    pC++;
                }
            }
            if (alpha != 1.f)
            {
                f0 *= alpha;
                f1 *= alpha;
            }
            p0[0] = f0;
            p0[1] = f1;
            p0 += M;
            pp += 2;
        }
    }
    for (; ii < max_ii; ii++)
    {
        float* p0 = outptr + (size_t)j * M + i + ii;
        const float* pC = pC_base;

        float c0 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0)
                c0 = pC[0] * beta;
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                c0 = pC[i + ii] * beta;
            if (broadcast_type_C == 3)
                pC += (size_t)(i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC += j;
        }

        int jj = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _alpha256 = (__m256)__lasx_xvreplfr2vr_s(alpha);
        __m256 _beta256 = (__m256)__lasx_xvreplfr2vr_s(beta);
        __m256 _c256 = (__m256)__lasx_xvreplfr2vr_s(c0);
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _f0 = (__m256)__lasx_xvld(pp, 0);
            __m256 _f1 = (__m256)__lasx_xvld(pp + 8, 0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lasx_xvfadd_s(_f0, _c256);
                    _f1 = __lasx_xvfadd_s(_f1, _c256);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m256 _c0 = (__m256)__lasx_xvld(pC, 0);
                    __m256 _c1 = (__m256)__lasx_xvld(pC + 8, 0);
                    if (beta == 1.f)
                    {
                        _f0 = __lasx_xvfadd_s(_f0, _c0);
                        _f1 = __lasx_xvfadd_s(_f1, _c1);
                    }
                    else
                    {
                        _f0 = __lasx_xvfmadd_s(_c0, _beta256, _f0);
                        _f1 = __lasx_xvfmadd_s(_c1, _beta256, _f1);
                    }
                    pC += 16;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lasx_xvfmul_s(_f0, _alpha256);
                _f1 = __lasx_xvfmul_s(_f1, _alpha256);
            }
            if (M == 1)
            {
                __lasx_xvst(_f0, p0, 0);
                __lasx_xvst(_f1, p0 + 8, 0);
            }
            else
            {
                __lasx_xvstelm_w((__m256i)_f0, p0, 0, 0);
                __lasx_xvstelm_w((__m256i)_f0, p0 + M, 0, 1);
                __lasx_xvstelm_w((__m256i)_f0, p0 + M * 2, 0, 2);
                __lasx_xvstelm_w((__m256i)_f0, p0 + M * 3, 0, 3);
                __lasx_xvstelm_w((__m256i)_f0, p0 + M * 4, 0, 4);
                __lasx_xvstelm_w((__m256i)_f0, p0 + M * 5, 0, 5);
                __lasx_xvstelm_w((__m256i)_f0, p0 + M * 6, 0, 6);
                __lasx_xvstelm_w((__m256i)_f0, p0 + M * 7, 0, 7);
                __lasx_xvstelm_w((__m256i)_f1, p0 + M * 8, 0, 0);
                __lasx_xvstelm_w((__m256i)_f1, p0 + M * 9, 0, 1);
                __lasx_xvstelm_w((__m256i)_f1, p0 + M * 10, 0, 2);
                __lasx_xvstelm_w((__m256i)_f1, p0 + M * 11, 0, 3);
                __lasx_xvstelm_w((__m256i)_f1, p0 + M * 12, 0, 4);
                __lasx_xvstelm_w((__m256i)_f1, p0 + M * 13, 0, 5);
                __lasx_xvstelm_w((__m256i)_f1, p0 + M * 14, 0, 6);
                __lasx_xvstelm_w((__m256i)_f1, p0 + M * 15, 0, 7);
            }
            p0 += M * 16;
            pp += 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = (__m256)__lasx_xvld(pp, 0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f0 = __lasx_xvfadd_s(_f0, _c256);
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m256 _c0 = (__m256)__lasx_xvld(pC, 0);
                    if (beta == 1.f)
                        _f0 = __lasx_xvfadd_s(_f0, _c0);
                    else
                        _f0 = __lasx_xvfmadd_s(_c0, _beta256, _f0);
                    pC += 8;
                }
            }
            if (alpha != 1.f) _f0 = __lasx_xvfmul_s(_f0, _alpha256);
            if (M == 1)
                __lasx_xvst(_f0, p0, 0);
            else
            {
                __lasx_xvstelm_w((__m256i)_f0, p0, 0, 0);
                __lasx_xvstelm_w((__m256i)_f0, p0 + M, 0, 1);
                __lasx_xvstelm_w((__m256i)_f0, p0 + M * 2, 0, 2);
                __lasx_xvstelm_w((__m256i)_f0, p0 + M * 3, 0, 3);
                __lasx_xvstelm_w((__m256i)_f0, p0 + M * 4, 0, 4);
                __lasx_xvstelm_w((__m256i)_f0, p0 + M * 5, 0, 5);
                __lasx_xvstelm_w((__m256i)_f0, p0 + M * 6, 0, 6);
                __lasx_xvstelm_w((__m256i)_f0, p0 + M * 7, 0, 7);
            }
            p0 += M * 8;
            pp += 8;
        }
#endif // __loongarch_asx
        __m128 _alpha128 = __lsx_vreplfr2vr_s(alpha);
        __m128 _beta128 = __lsx_vreplfr2vr_s(beta);
        __m128 _c128 = __lsx_vreplfr2vr_s(c0);
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp, 0);
            __m128 _f1 = (__m128)__lsx_vld(pp + 4, 0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, _c128);
                    _f1 = __lsx_vfadd_s(_f1, _c128);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m128 _c0 = (__m128)__lsx_vld(pC, 0);
                    __m128 _c1 = (__m128)__lsx_vld(pC + 4, 0);
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                        _f1 = __lsx_vfadd_s(_f1, _c1);
                    }
                    else
                    {
                        _f0 = __lsx_vfmadd_s(_c0, _beta128, _f0);
                        _f1 = __lsx_vfmadd_s(_c1, _beta128, _f1);
                    }
                    pC += 8;
                }
            }
            if (alpha != 1.f)
            {
                _f0 = __lsx_vfmul_s(_f0, _alpha128);
                _f1 = __lsx_vfmul_s(_f1, _alpha128);
            }
            if (M == 1)
            {
                __lsx_vst((__m128i)_f0, p0, 0);
                __lsx_vst((__m128i)_f1, p0 + 4, 0);
            }
            else
            {
                __lsx_vstelm_w((__m128i)_f0, p0, 0, 0);
                __lsx_vstelm_w((__m128i)_f0, p0 + M, 0, 1);
                __lsx_vstelm_w((__m128i)_f0, p0 + M * 2, 0, 2);
                __lsx_vstelm_w((__m128i)_f0, p0 + M * 3, 0, 3);
                __lsx_vstelm_w((__m128i)_f1, p0 + M * 4, 0, 0);
                __lsx_vstelm_w((__m128i)_f1, p0 + M * 5, 0, 1);
                __lsx_vstelm_w((__m128i)_f1, p0 + M * 6, 0, 2);
                __lsx_vstelm_w((__m128i)_f1, p0 + M * 7, 0, 3);
            }
            p0 += M * 8;
            pp += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp, 0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f0 = __lsx_vfadd_s(_f0, _c128);
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m128 _c0 = (__m128)__lsx_vld(pC, 0);
                    if (beta == 1.f)
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                    else
                        _f0 = __lsx_vfmadd_s(_c0, _beta128, _f0);
                    pC += 4;
                }
            }
            if (alpha != 1.f) _f0 = __lsx_vfmul_s(_f0, _alpha128);
            if (M == 1)
                __lsx_vst((__m128i)_f0, p0, 0);
            else
            {
                __lsx_vstelm_w((__m128i)_f0, p0, 0, 0);
                __lsx_vstelm_w((__m128i)_f0, p0 + M, 0, 1);
                __lsx_vstelm_w((__m128i)_f0, p0 + M * 2, 0, 2);
                __lsx_vstelm_w((__m128i)_f0, p0 + M * 3, 0, 3);
            }
            p0 += M * 4;
            pp += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _f0 = (__m128)__lsx_vldrepl_d(pp, 0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f0 = __lsx_vfadd_s(_f0, _c128);
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m128 _cc = (__m128)__lsx_vldrepl_d(pC, 0);
                    if (beta == 1.f)
                        _f0 = __lsx_vfadd_s(_f0, _cc);
                    else
                        _f0 = __lsx_vfmadd_s(_cc, _beta128, _f0);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
                _f0 = __lsx_vfmul_s(_f0, _alpha128);
            if (M == 1)
                __lsx_vstelm_d((__m128i)_f0, p0, 0, 0);
            else
            {
                __lsx_vstelm_w((__m128i)_f0, p0, 0, 0);
                __lsx_vstelm_w((__m128i)_f0, p0 + M, 0, 1);
            }
            p0 += M * 2;
            pp += 2;
        }
#endif // __loongarch_sx
        for (; jj < max_jj; jj++)
        {
            float f0 = *pp++;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    f0 += c0;
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    f0 += beta == 1.f ? pC[0] : pC[0] * beta;
                    pC++;
                }
            }
            if (alpha != 1.f)
                f0 *= alpha;
            p0[0] = f0;
            p0 += M;
        }
    }
}

static void get_optimal_tile_mnk_wq_int8(int M, int N, int K, int block_size, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrtf((float)l2_cache_size / (2 * sizeof(signed char) + sizeof(float)));

    TILE_K = std::max(block_size, tile_size / block_size * block_size);
    if (K > 0)
    {
        if (TILE_K >= K)
        {
            TILE_K = K;
        }
        else
        {
            const int nn_K = (K + TILE_K - 1) / TILE_K;
            const int tile_k = (K + nn_K - 1) / nn_K;
            TILE_K = std::max(block_size, tile_k / block_size * block_size);
        }
    }

    tile_size = std::max(1, (int)((float)l2_cache_size / 2 / sizeof(signed char) / std::max(1, TILE_K)));

#if __loongarch_sx
    const int tile_m_align = M >= nT * 8 ? 8 : M >= nT * 4 ? 4 : M >= nT * 2 ? 2 : 1;
#if __loongarch_asx
    const int tile_n_align = tile_m_align == 8 ? 8 : 16;
#else
    const int tile_n_align = tile_m_align == 8 ? 4 : 8;
#endif
#else
    const int tile_m_align = M >= nT * 2 ? 2 : 1;
    const int tile_n_align = 2;
#endif
    TILE_M = tile_m_align;
    TILE_N = std::max(tile_n_align, tile_size / tile_n_align * tile_n_align);

    if (N > 0)
    {
        const int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + tile_n_align - 1) / tile_n_align * tile_n_align);
    }

    // always take constant TILE_N value when provided
    if (constant_TILE_N > 0)
        TILE_N = (constant_TILE_N + tile_n_align - 1) / tile_n_align * tile_n_align;

    if (constant_TILE_K > 0)
    {
        TILE_K = std::max(block_size, constant_TILE_K / block_size * block_size);
        if (K > 0)
            TILE_K = std::min(TILE_K, K);
    }

    (void)constant_TILE_M;
}

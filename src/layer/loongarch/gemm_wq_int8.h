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
#if __loongarch_asx
    const int nn8 = (N - panel_start) / 8;
    const int panel_start8 = panel_start;
    panel_start += nn8 * 8;
    panel_count += nn8;
#endif
#if __loongarch_sx
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
#if __loongarch_sx
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
#endif
#if __loongarch_asx
        }
#endif

        signed char* pp = (signed char*)BT_packed + j * K;
        float* pd = (float*)BT_packed_descales + j * block_count;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                for (int jj = 0; jj < nr; jj++)
                {
                    const signed char* pB = B.row<const signed char>(j + jj) + k0 + kk;
                    pp[0] = pB[0];
                    pp[1] = pB[1];
                    pp[2] = pB[2];
                    pp[3] = pB[3];
                    pp += 4;
                }
            }
            if (kk + 1 < max_kk)
            {
                for (int jj = 0; jj < nr; jj++)
                {
                    const signed char* pB = B.row<const signed char>(j + jj) + k0 + kk;
                    pp[0] = pB[0];
                    pp[1] = pB[1];
                    pp += 2;
                }
                kk += 2;
            }
            if (kk < max_kk)
            {
                for (int jj = 0; jj < nr; jj++)
                    *pp++ = B.row<const signed char>(j + jj)[k0 + kk];
            }

            for (int jj = 0; jj < nr; jj++)
                pd[g * nr + jj] = 1.f / B_scales.row(j + jj)[g];
        }
    }

    BT = BT_packed;
    BT_descales = BT_packed_descales;
    return 0;
}

static void quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int K, int block_size, const float* input_scale_ptr)
{
    signed char* outptr = AT_tile;
    const int out_hstep = AT_tile.w;
    float* descales = AT_descales_tile;
    const int descales_hstep = AT_descales_tile.w;
    const int block_count = (K + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep;
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
            const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
            __m128 _absmax0 = (__m128)__lsx_vldi(0);
            __m128 _absmax1 = (__m128)__lsx_vldi(0);
            __m128 _absmax2 = (__m128)__lsx_vldi(0);
            __m128 _absmax3 = (__m128)__lsx_vldi(0);
            __m128 _absmax4 = (__m128)__lsx_vldi(0);
            __m128 _absmax5 = (__m128)__lsx_vldi(0);
            __m128 _absmax6 = (__m128)__lsx_vldi(0);
            __m128 _absmax7 = (__m128)__lsx_vldi(0);
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _v0 = (__m128)__lsx_vld(p0 + k0 + kk, 0);
                __m128 _v1 = (__m128)__lsx_vld(p1 + k0 + kk, 0);
                __m128 _v2 = (__m128)__lsx_vld(p2 + k0 + kk, 0);
                __m128 _v3 = (__m128)__lsx_vld(p3 + k0 + kk, 0);
                __m128 _v4 = (__m128)__lsx_vld(p4 + k0 + kk, 0);
                __m128 _v5 = (__m128)__lsx_vld(p5 + k0 + kk, 0);
                __m128 _v6 = (__m128)__lsx_vld(p6 + k0 + kk, 0);
                __m128 _v7 = (__m128)__lsx_vld(p7 + k0 + kk, 0);
                if (input_scale_ptr)
                {
                    const __m128 _s = (__m128)__lsx_vld(input_scale_ptr + k0 + kk, 0);
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
                const int k = k0 + kk;
                const float s = input_scale_ptr ? input_scale_ptr[k] : 1.f;
                absmax0 = std::max(absmax0, fabsf(p0[k] * s));
                absmax1 = std::max(absmax1, fabsf(p1[k] * s));
                absmax2 = std::max(absmax2, fabsf(p2[k] * s));
                absmax3 = std::max(absmax3, fabsf(p3[k] * s));
                absmax4 = std::max(absmax4, fabsf(p4[k] * s));
                absmax5 = std::max(absmax5, fabsf(p5[k] * s));
                absmax6 = std::max(absmax6, fabsf(p6[k] * s));
                absmax7 = std::max(absmax7, fabsf(p7[k] * s));
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

            volatile double scale0_fp64 = absmax0 == 0.f ? 0.0 : 127.0 / (double)absmax0;
            volatile double scale1_fp64 = absmax1 == 0.f ? 0.0 : 127.0 / (double)absmax1;
            volatile double scale2_fp64 = absmax2 == 0.f ? 0.0 : 127.0 / (double)absmax2;
            volatile double scale3_fp64 = absmax3 == 0.f ? 0.0 : 127.0 / (double)absmax3;
            volatile double scale4_fp64 = absmax4 == 0.f ? 0.0 : 127.0 / (double)absmax4;
            volatile double scale5_fp64 = absmax5 == 0.f ? 0.0 : 127.0 / (double)absmax5;
            volatile double scale6_fp64 = absmax6 == 0.f ? 0.0 : 127.0 / (double)absmax6;
            volatile double scale7_fp64 = absmax7 == 0.f ? 0.0 : 127.0 / (double)absmax7;
            const float scale0 = (float)scale0_fp64;
            const float scale1 = (float)scale1_fp64;
            const float scale2 = (float)scale2_fp64;
            const float scale3 = (float)scale3_fp64;
            const float scale4 = (float)scale4_fp64;
            const float scale5 = (float)scale5_fp64;
            const float scale6 = (float)scale6_fp64;
            const float scale7 = (float)scale7_fp64;
            const __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
            const __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
            const __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
            const __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
            const __m128 _scale4 = __lsx_vreplfr2vr_s(scale4);
            const __m128 _scale5 = __lsx_vreplfr2vr_s(scale5);
            const __m128 _scale6 = __lsx_vreplfr2vr_s(scale6);
            const __m128 _scale7 = __lsx_vreplfr2vr_s(scale7);
            kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                const int k = k0 + kk;
                __m128 _v0 = (__m128)__lsx_vld(p0 + k, 0);
                __m128 _v1 = (__m128)__lsx_vld(p1 + k, 0);
                __m128 _v2 = (__m128)__lsx_vld(p2 + k, 0);
                __m128 _v3 = (__m128)__lsx_vld(p3 + k, 0);
                __m128 _v4 = (__m128)__lsx_vld(p4 + k, 0);
                __m128 _v5 = (__m128)__lsx_vld(p5 + k, 0);
                __m128 _v6 = (__m128)__lsx_vld(p6 + k, 0);
                __m128 _v7 = (__m128)__lsx_vld(p7 + k, 0);
                if (input_scale_ptr)
                {
                    const __m128 _s = (__m128)__lsx_vld(input_scale_ptr + k, 0);
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
            }
            for (; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                const float s = input_scale_ptr ? input_scale_ptr[k] : 1.f;
                pp[0] = float2int8(p0[k] * s * scale0);
                pp[1] = float2int8(p1[k] * s * scale1);
                pp[2] = float2int8(p2[k] * s * scale2);
                pp[3] = float2int8(p3[k] * s * scale3);
                pp[4] = float2int8(p4[k] * s * scale4);
                pp[5] = float2int8(p5[k] * s * scale5);
                pp[6] = float2int8(p6[k] * s * scale6);
                pp[7] = float2int8(p7[k] * s * scale7);
                pp += 8;
            }
        }
    }
#endif // __loongarch_sx
#if __loongarch_sx
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep;
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
            __m128 _absmax0 = (__m128)__lsx_vldi(0);
            __m128 _absmax1 = (__m128)__lsx_vldi(0);
            __m128 _absmax2 = (__m128)__lsx_vldi(0);
            __m128 _absmax3 = (__m128)__lsx_vldi(0);
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _v0 = (__m128)__lsx_vld(p0 + k0 + kk, 0);
                __m128 _v1 = (__m128)__lsx_vld(p1 + k0 + kk, 0);
                __m128 _v2 = (__m128)__lsx_vld(p2 + k0 + kk, 0);
                __m128 _v3 = (__m128)__lsx_vld(p3 + k0 + kk, 0);
                if (input_scale_ptr)
                {
                    const __m128 _s = (__m128)__lsx_vld(input_scale_ptr + k0 + kk, 0);
                    _v0 = __lsx_vfmul_s(_v0, _s);
                    _v1 = __lsx_vfmul_s(_v1, _s);
                    _v2 = __lsx_vfmul_s(_v2, _s);
                    _v3 = __lsx_vfmul_s(_v3, _s);
                }
                _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_v0, _abs_mask));
                _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_v1, _abs_mask));
                _absmax2 = __lsx_vfmax_s(_absmax2, (__m128)__lsx_vand_v((__m128i)_v2, _abs_mask));
                _absmax3 = __lsx_vfmax_s(_absmax3, (__m128)__lsx_vand_v((__m128i)_v3, _abs_mask));
            }
            float absmax0 = __lsx_reduce_fmax_s(_absmax0);
            float absmax1 = __lsx_reduce_fmax_s(_absmax1);
            float absmax2 = __lsx_reduce_fmax_s(_absmax2);
            float absmax3 = __lsx_reduce_fmax_s(_absmax3);
            for (; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                const float s = input_scale_ptr ? input_scale_ptr[k] : 1.f;
                absmax0 = std::max(absmax0, fabsf(p0[k] * s));
                absmax1 = std::max(absmax1, fabsf(p1[k] * s));
                absmax2 = std::max(absmax2, fabsf(p2[k] * s));
                absmax3 = std::max(absmax3, fabsf(p3[k] * s));
            }
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd[2] = absmax2 / 127.f;
            pd[3] = absmax3 / 127.f;
            pd += 4;
            volatile double scale0_fp64 = absmax0 == 0.f ? 0.0 : 127.0 / (double)absmax0;
            volatile double scale1_fp64 = absmax1 == 0.f ? 0.0 : 127.0 / (double)absmax1;
            volatile double scale2_fp64 = absmax2 == 0.f ? 0.0 : 127.0 / (double)absmax2;
            volatile double scale3_fp64 = absmax3 == 0.f ? 0.0 : 127.0 / (double)absmax3;
            const float scale0 = (float)scale0_fp64;
            const float scale1 = (float)scale1_fp64;
            const float scale2 = (float)scale2_fp64;
            const float scale3 = (float)scale3_fp64;
            const __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
            const __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
            const __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
            const __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
            kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _v0 = (__m128)__lsx_vld(p0 + k0 + kk, 0);
                __m128 _v1 = (__m128)__lsx_vld(p1 + k0 + kk, 0);
                __m128 _v2 = (__m128)__lsx_vld(p2 + k0 + kk, 0);
                __m128 _v3 = (__m128)__lsx_vld(p3 + k0 + kk, 0);
                if (input_scale_ptr)
                {
                    const __m128 _s = (__m128)__lsx_vld(input_scale_ptr + k0 + kk, 0);
                    _v0 = __lsx_vfmul_s(_v0, _s);
                    _v1 = __lsx_vfmul_s(_v1, _s);
                    _v2 = __lsx_vfmul_s(_v2, _s);
                    _v3 = __lsx_vfmul_s(_v3, _s);
                }
                *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_v0, _scale0), __lsx_vfmul_s(_v1, _scale1));
                *((int64_t*)(pp + 8)) = float2int8(__lsx_vfmul_s(_v2, _scale2), __lsx_vfmul_s(_v3, _scale3));
                pp += 16;
            }
            if (kk + 1 < max_kk)
            {
                const int k = k0 + kk;
                const float s0 = input_scale_ptr ? input_scale_ptr[k] : 1.f;
                const float s1 = input_scale_ptr ? input_scale_ptr[k + 1] : 1.f;
                pp[0] = float2int8(p0[k] * s0 * scale0);
                pp[1] = float2int8(p0[k + 1] * s1 * scale0);
                pp[2] = float2int8(p1[k] * s0 * scale1);
                pp[3] = float2int8(p1[k + 1] * s1 * scale1);
                pp[4] = float2int8(p2[k] * s0 * scale2);
                pp[5] = float2int8(p2[k + 1] * s1 * scale2);
                pp[6] = float2int8(p3[k] * s0 * scale3);
                pp[7] = float2int8(p3[k + 1] * s1 * scale3);
                pp += 8;
                kk += 2;
            }
            if (kk < max_kk)
            {
                const int k = k0 + kk;
                const float s = input_scale_ptr ? input_scale_ptr[k] : 1.f;
                pp[0] = float2int8(p0[k] * s * scale0);
                pp[1] = float2int8(p1[k] * s * scale1);
                pp[2] = float2int8(p2[k] * s * scale2);
                pp[3] = float2int8(p3[k] * s * scale3);
                pp += 4;
            }
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep;
        const float* p1 = p0 + A_hstep;
        signed char* pp = outptr + ii * out_hstep;
        float* pd = descales + ii * descales_hstep;
        const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            __m128 _absmax0 = (__m128)__lsx_vldi(0);
            __m128 _absmax1 = (__m128)__lsx_vldi(0);
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _v0 = (__m128)__lsx_vld(p0 + k0 + kk, 0);
                __m128 _v1 = (__m128)__lsx_vld(p1 + k0 + kk, 0);
                if (input_scale_ptr)
                {
                    const __m128 _s = (__m128)__lsx_vld(input_scale_ptr + k0 + kk, 0);
                    _v0 = __lsx_vfmul_s(_v0, _s);
                    _v1 = __lsx_vfmul_s(_v1, _s);
                }
                _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_v0, _abs_mask));
                _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_v1, _abs_mask));
            }
            float absmax0 = __lsx_reduce_fmax_s(_absmax0);
            float absmax1 = __lsx_reduce_fmax_s(_absmax1);
            for (; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                const float s = input_scale_ptr ? input_scale_ptr[k] : 1.f;
                absmax0 = std::max(absmax0, fabsf(p0[k] * s));
                absmax1 = std::max(absmax1, fabsf(p1[k] * s));
            }
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;
            volatile double scale0_fp64 = absmax0 == 0.f ? 0.0 : 127.0 / (double)absmax0;
            volatile double scale1_fp64 = absmax1 == 0.f ? 0.0 : 127.0 / (double)absmax1;
            const float scale0 = (float)scale0_fp64;
            const float scale1 = (float)scale1_fp64;
            const __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
            const __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
            kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _v0 = (__m128)__lsx_vld(p0 + k0 + kk, 0);
                __m128 _v1 = (__m128)__lsx_vld(p1 + k0 + kk, 0);
                if (input_scale_ptr)
                {
                    const __m128 _s = (__m128)__lsx_vld(input_scale_ptr + k0 + kk, 0);
                    _v0 = __lsx_vfmul_s(_v0, _s);
                    _v1 = __lsx_vfmul_s(_v1, _s);
                }
                *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_v0, _scale0), __lsx_vfmul_s(_v1, _scale1));
                pp += 8;
            }
            if (kk + 1 < max_kk)
            {
                const int k = k0 + kk;
                const float s0 = input_scale_ptr ? input_scale_ptr[k] : 1.f;
                const float s1 = input_scale_ptr ? input_scale_ptr[k + 1] : 1.f;
                pp[0] = float2int8(p0[k] * s0 * scale0);
                pp[1] = float2int8(p0[k + 1] * s1 * scale0);
                pp[2] = float2int8(p1[k] * s0 * scale1);
                pp[3] = float2int8(p1[k + 1] * s1 * scale1);
                pp += 4;
                kk += 2;
            }
            if (kk < max_kk)
            {
                const int k = k0 + kk;
                const float s = input_scale_ptr ? input_scale_ptr[k] : 1.f;
                pp[0] = float2int8(p0[k] * s * scale0);
                pp[1] = float2int8(p1[k] * s * scale1);
                pp += 2;
            }
        }
    }
#endif // __loongarch_sx
    for (; ii < max_ii; ii++)
    {
        const float* ptrA = (const float*)A + (i + ii) * A_hstep;
        signed char* outptr0 = outptr + ii * out_hstep;
        float* descale_ptr = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            float absmax = 0.f;
            int kk = 0;
#if __loongarch_sx
            const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
#if __loongarch_asx
            const __m256i _abs_mask256 = __lasx_xvreplgr2vr_w(0x7fffffff);
            __m256 _absmax256 = (__m256)__lasx_xvldi(0);
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _v = (__m256)__lasx_xvld(ptrA + k0 + kk, 0);
                if (input_scale_ptr)
                    _v = __lasx_xvfmul_s(_v, (__m256)__lasx_xvld(input_scale_ptr + k0 + kk, 0));
                _v = (__m256)__lasx_xvand_v((__m256i)_v, _abs_mask256);
                _absmax256 = __lasx_xvfmax_s(_absmax256, _v);
            }
            absmax = __lasx_reduce_fmax_s(_absmax256);
#endif
            __m128 _absmax128 = __lsx_vreplfr2vr_s(absmax);
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _v = (__m128)__lsx_vld(ptrA + k0 + kk, 0);
                if (input_scale_ptr)
                    _v = __lsx_vfmul_s(_v, (__m128)__lsx_vld(input_scale_ptr + k0 + kk, 0));
                _v = (__m128)__lsx_vand_v((__m128i)_v, _abs_mask);
                _absmax128 = __lsx_vfmax_s(_absmax128, _v);
            }
            absmax = __lsx_reduce_fmax_s(_absmax128);
#endif
            for (; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v = ptrA[k];
                if (input_scale_ptr)
                    v *= input_scale_ptr[k];
                absmax = std::max(absmax, fabsf(v));
            }

            if (absmax == 0.f)
            {
                descale_ptr[g] = 0.f;
                for (int k = 0; k < max_kk; k++)
                    outptr0[k0 + k] = 0;
                continue;
            }

            volatile double scale_fp64 = 127.0 / (double)absmax;
            const float scale = (float)scale_fp64;
            descale_ptr[g] = absmax / 127.f;
            kk = 0;
#if __loongarch_sx
#if __loongarch_asx
            const __m256 _scale256 = (__m256)__lasx_xvreplfr2vr_s(scale);
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _v = (__m256)__lasx_xvld(ptrA + k0 + kk, 0);
                if (input_scale_ptr)
                    _v = __lasx_xvfmul_s(_v, (__m256)__lasx_xvld(input_scale_ptr + k0 + kk, 0));
                _v = __lasx_xvfmul_s(_v, _scale256);
                __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(_v)), outptr0 + k0 + kk, 0, 0);
            }
#endif
            const __m128 _scale128 = __lsx_vreplfr2vr_s(scale);
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _v = (__m128)__lsx_vld(ptrA + k0 + kk, 0);
                if (input_scale_ptr)
                    _v = __lsx_vfmul_s(_v, (__m128)__lsx_vld(input_scale_ptr + k0 + kk, 0));
                _v = __lsx_vfmul_s(_v, _scale128);
                __lsx_vstelm_w(float2int8(_v), outptr0 + k0 + kk, 0, 0);
            }
#endif
            for (; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v = ptrA[k];
                if (input_scale_ptr)
                {
                    v *= input_scale_ptr[k];
                    // preserve multiplication order for consistent rounding
                    asm volatile(""
                                 : "+f"(v));
                }
                outptr0[k] = float2int8(v * scale);
            }
        }
    }
}

static void transpose_quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int K, int block_size, const float* input_scale_ptr)
{
    signed char* outptr = AT_tile;
    const int out_hstep = AT_tile.w;
    float* descales = AT_descales_tile;
    const int descales_hstep = AT_descales_tile.w;
    const int block_count = (K + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* ptrA = (const float*)A + i + ii;
        signed char* pp = outptr + ii * out_hstep;
        float* pd = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
            __m128 _absmax0 = (__m128)__lsx_vldi(0);
            __m128 _absmax1 = (__m128)__lsx_vldi(0);
            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                const float* p = ptrA + (size_t)k * A_hstep;
                __m128 _v0 = (__m128)__lsx_vld(p, 0);
                __m128 _v1 = (__m128)__lsx_vld(p + 4, 0);
                if (input_scale_ptr)
                {
                    const __m128 _s = __lsx_vreplfr2vr_s(input_scale_ptr[k]);
                    _v0 = __lsx_vfmul_s(_v0, _s);
                    _v1 = __lsx_vfmul_s(_v1, _s);
                }
                _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_v0, _abs_mask));
                _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_v1, _abs_mask));
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

            volatile double scale0_fp64 = absmax0 == 0.f ? 0.0 : 127.0 / (double)absmax0;
            volatile double scale1_fp64 = absmax1 == 0.f ? 0.0 : 127.0 / (double)absmax1;
            volatile double scale2_fp64 = absmax2 == 0.f ? 0.0 : 127.0 / (double)absmax2;
            volatile double scale3_fp64 = absmax3 == 0.f ? 0.0 : 127.0 / (double)absmax3;
            volatile double scale4_fp64 = absmax4 == 0.f ? 0.0 : 127.0 / (double)absmax4;
            volatile double scale5_fp64 = absmax5 == 0.f ? 0.0 : 127.0 / (double)absmax5;
            volatile double scale6_fp64 = absmax6 == 0.f ? 0.0 : 127.0 / (double)absmax6;
            volatile double scale7_fp64 = absmax7 == 0.f ? 0.0 : 127.0 / (double)absmax7;
            const float scale0 = (float)scale0_fp64;
            const float scale1 = (float)scale1_fp64;
            const float scale2 = (float)scale2_fp64;
            const float scale3 = (float)scale3_fp64;
            const float scale4 = (float)scale4_fp64;
            const float scale5 = (float)scale5_fp64;
            const float scale6 = (float)scale6_fp64;
            const float scale7 = (float)scale7_fp64;
            const __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
            const __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
            const __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
            const __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
            const __m128 _scale4 = __lsx_vreplfr2vr_s(scale4);
            const __m128 _scale5 = __lsx_vreplfr2vr_s(scale5);
            const __m128 _scale6 = __lsx_vreplfr2vr_s(scale6);
            const __m128 _scale7 = __lsx_vreplfr2vr_s(scale7);
            const float scales0[4] = {scale0, scale1, scale2, scale3};
            const float scales1[4] = {scale4, scale5, scale6, scale7};
            const __m128 _scales0 = (__m128)__lsx_vld(scales0, 0);
            const __m128 _scales1 = (__m128)__lsx_vld(scales1, 0);

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                const int k = k0 + kk;
                const float* p0 = ptrA + (size_t)k * A_hstep;
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
                if (input_scale_ptr)
                {
                    _p0 = __lsx_vfmul_s(_p0, __lsx_vreplfr2vr_s(input_scale_ptr[k]));
                    _p1 = __lsx_vfmul_s(_p1, __lsx_vreplfr2vr_s(input_scale_ptr[k + 1]));
                    _p2 = __lsx_vfmul_s(_p2, __lsx_vreplfr2vr_s(input_scale_ptr[k + 2]));
                    _p3 = __lsx_vfmul_s(_p3, __lsx_vreplfr2vr_s(input_scale_ptr[k + 3]));
                    _p4 = __lsx_vfmul_s(_p4, __lsx_vreplfr2vr_s(input_scale_ptr[k]));
                    _p5 = __lsx_vfmul_s(_p5, __lsx_vreplfr2vr_s(input_scale_ptr[k + 1]));
                    _p6 = __lsx_vfmul_s(_p6, __lsx_vreplfr2vr_s(input_scale_ptr[k + 2]));
                    _p7 = __lsx_vfmul_s(_p7, __lsx_vreplfr2vr_s(input_scale_ptr[k + 3]));
                }
                transpose4x4_ps(_p0, _p1, _p2, _p3);
                transpose4x4_ps(_p4, _p5, _p6, _p7);
                *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_p0, _scale0), __lsx_vfmul_s(_p1, _scale1));
                *((int64_t*)(pp + 8)) = float2int8(__lsx_vfmul_s(_p2, _scale2), __lsx_vfmul_s(_p3, _scale3));
                *((int64_t*)(pp + 16)) = float2int8(__lsx_vfmul_s(_p4, _scale4), __lsx_vfmul_s(_p5, _scale5));
                *((int64_t*)(pp + 24)) = float2int8(__lsx_vfmul_s(_p6, _scale6), __lsx_vfmul_s(_p7, _scale7));
                pp += 32;
            }
            for (; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                const float* p = ptrA + (size_t)k * A_hstep;
                __m128 _p0 = (__m128)__lsx_vld(p, 0);
                __m128 _p1 = (__m128)__lsx_vld(p + 4, 0);
                if (input_scale_ptr)
                {
                    const __m128 _s = __lsx_vreplfr2vr_s(input_scale_ptr[k]);
                    _p0 = __lsx_vfmul_s(_p0, _s);
                    _p1 = __lsx_vfmul_s(_p1, _s);
                }
                _p0 = __lsx_vfmul_s(_p0, _scales0);
                _p1 = __lsx_vfmul_s(_p1, _scales1);
                *((int64_t*)pp) = float2int8(_p0, _p1);
                pp += 8;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* ptrA = (const float*)A + i + ii;
        signed char* pp = outptr + ii * out_hstep;
        float* pd = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);

            const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
            __m128 _absmax = (__m128)__lsx_vldi(0);
            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                __m128 _v = (__m128)__lsx_vld(ptrA + (size_t)k * A_hstep, 0);
                if (input_scale_ptr)
                    _v = __lsx_vfmul_s(_v, __lsx_vreplfr2vr_s(input_scale_ptr[k]));
                _v = (__m128)__lsx_vand_v((__m128i)_v, _abs_mask);
                _absmax = __lsx_vfmax_s(_absmax, _v);
            }

            float absmax[4];
            __lsx_vst(_absmax, absmax, 0);
            pd[0] = absmax[0] / 127.f;
            pd[1] = absmax[1] / 127.f;
            pd[2] = absmax[2] / 127.f;
            pd[3] = absmax[3] / 127.f;
            pd += 4;

            volatile double scale0_fp64 = absmax[0] == 0.f ? 0.0 : 127.0 / (double)absmax[0];
            volatile double scale1_fp64 = absmax[1] == 0.f ? 0.0 : 127.0 / (double)absmax[1];
            volatile double scale2_fp64 = absmax[2] == 0.f ? 0.0 : 127.0 / (double)absmax[2];
            volatile double scale3_fp64 = absmax[3] == 0.f ? 0.0 : 127.0 / (double)absmax[3];
            const float scales[4] = {
                (float)scale0_fp64,
                (float)scale1_fp64,
                (float)scale2_fp64,
                (float)scale3_fp64
            };
            const __m128 _scale = (__m128)__lsx_vld(scales, 0);

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                const int k = k0 + kk;
                const float* p0 = ptrA + (size_t)k * A_hstep;
                const float* p1 = p0 + A_hstep;
                const float* p2 = p1 + A_hstep;
                const float* p3 = p2 + A_hstep;
                __m128 _v0 = (__m128)__lsx_vld(p0, 0);
                __m128 _v1 = (__m128)__lsx_vld(p1, 0);
                __m128 _v2 = (__m128)__lsx_vld(p2, 0);
                __m128 _v3 = (__m128)__lsx_vld(p3, 0);
                if (input_scale_ptr)
                {
                    _v0 = __lsx_vfmul_s(_v0, __lsx_vreplfr2vr_s(input_scale_ptr[k]));
                    _v1 = __lsx_vfmul_s(_v1, __lsx_vreplfr2vr_s(input_scale_ptr[k + 1]));
                    _v2 = __lsx_vfmul_s(_v2, __lsx_vreplfr2vr_s(input_scale_ptr[k + 2]));
                    _v3 = __lsx_vfmul_s(_v3, __lsx_vreplfr2vr_s(input_scale_ptr[k + 3]));
                }
                transpose4x4_ps(_v0, _v1, _v2, _v3);
                *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_v0, __lsx_vreplfr2vr_s(scales[0])), __lsx_vfmul_s(_v1, __lsx_vreplfr2vr_s(scales[1])));
                *((int64_t*)(pp + 8)) = float2int8(__lsx_vfmul_s(_v2, __lsx_vreplfr2vr_s(scales[2])), __lsx_vfmul_s(_v3, __lsx_vreplfr2vr_s(scales[3])));
                pp += 16;
            }
            if (kk + 1 < max_kk)
            {
                const int k = k0 + kk;
                __m128 _v0 = (__m128)__lsx_vld(ptrA + (size_t)k * A_hstep, 0);
                __m128 _v1 = (__m128)__lsx_vld(ptrA + (size_t)(k + 1) * A_hstep, 0);
                if (input_scale_ptr)
                {
                    _v0 = __lsx_vfmul_s(_v0, __lsx_vreplfr2vr_s(input_scale_ptr[k]));
                    _v1 = __lsx_vfmul_s(_v1, __lsx_vreplfr2vr_s(input_scale_ptr[k + 1]));
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
                kk += 2;
            }
            if (kk < max_kk)
            {
                const int k = k0 + kk;
                __m128 _v = (__m128)__lsx_vld(ptrA + (size_t)k * A_hstep, 0);
                if (input_scale_ptr)
                    _v = __lsx_vfmul_s(_v, __lsx_vreplfr2vr_s(input_scale_ptr[k]));
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
        const float* ptrA = (const float*)A + i + ii;
        signed char* pp = outptr + ii * out_hstep;
        float* pd = descales + ii * descales_hstep;
        const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            __m128 _absmax = (__m128)__lsx_vldi(0);
            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                __m128 _v = (__m128)__lsx_vldrepl_d(ptrA + (size_t)k * A_hstep, 0);
                if (input_scale_ptr)
                    _v = __lsx_vfmul_s(_v, __lsx_vreplfr2vr_s(input_scale_ptr[k]));
                _absmax = __lsx_vfmax_s(_absmax, (__m128)__lsx_vand_v((__m128i)_v, _abs_mask));
            }
            float absmax[2];
            __lsx_vstelm_d((__m128i)_absmax, absmax, 0, 0);
            pd[0] = absmax[0] / 127.f;
            pd[1] = absmax[1] / 127.f;
            pd += 2;
            volatile double scale0_fp64 = absmax[0] == 0.f ? 0.0 : 127.0 / (double)absmax[0];
            volatile double scale1_fp64 = absmax[1] == 0.f ? 0.0 : 127.0 / (double)absmax[1];
            const float scale0 = (float)scale0_fp64;
            const float scale1 = (float)scale1_fp64;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                const int k = k0 + kk;
                const float* p0 = ptrA + (size_t)k * A_hstep;
                const float* p1 = p0 + A_hstep;
                const float* p2 = p1 + A_hstep;
                const float* p3 = p2 + A_hstep;
                const float s0 = input_scale_ptr ? input_scale_ptr[k] : 1.f;
                const float s1 = input_scale_ptr ? input_scale_ptr[k + 1] : 1.f;
                const float s2 = input_scale_ptr ? input_scale_ptr[k + 2] : 1.f;
                const float s3 = input_scale_ptr ? input_scale_ptr[k + 3] : 1.f;
                pp[0] = float2int8(p0[0] * s0 * scale0);
                pp[1] = float2int8(p1[0] * s1 * scale0);
                pp[2] = float2int8(p2[0] * s2 * scale0);
                pp[3] = float2int8(p3[0] * s3 * scale0);
                pp[4] = float2int8(p0[1] * s0 * scale1);
                pp[5] = float2int8(p1[1] * s1 * scale1);
                pp[6] = float2int8(p2[1] * s2 * scale1);
                pp[7] = float2int8(p3[1] * s3 * scale1);
                pp += 8;
            }
            if (kk + 1 < max_kk)
            {
                const int k = k0 + kk;
                const float* p0 = ptrA + (size_t)k * A_hstep;
                const float* p1 = p0 + A_hstep;
                const float s0 = input_scale_ptr ? input_scale_ptr[k] : 1.f;
                const float s1 = input_scale_ptr ? input_scale_ptr[k + 1] : 1.f;
                pp[0] = float2int8(p0[0] * s0 * scale0);
                pp[1] = float2int8(p1[0] * s1 * scale0);
                pp[2] = float2int8(p0[1] * s0 * scale1);
                pp[3] = float2int8(p1[1] * s1 * scale1);
                pp += 4;
                kk += 2;
            }
            if (kk < max_kk)
            {
                const int k = k0 + kk;
                const float* p = ptrA + (size_t)k * A_hstep;
                const float s = input_scale_ptr ? input_scale_ptr[k] : 1.f;
                pp[0] = float2int8(p[0] * s * scale0);
                pp[1] = float2int8(p[1] * s * scale1);
                pp += 2;
            }
        }
    }
#endif // __loongarch_sx
    for (; ii < max_ii; ii++)
    {
        const float* ptrA = (const float*)A + i + ii;
        signed char* outptr0 = outptr + ii * out_hstep;
        float* descale_ptr = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);

            float absmax = 0.f;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v = ptrA[(size_t)k * A_hstep];
                if (input_scale_ptr)
                    v *= input_scale_ptr[k];
                absmax = std::max(absmax, fabsf(v));
            }

            if (absmax == 0.f)
            {
                descale_ptr[g] = 0.f;
                for (int k = 0; k < max_kk; k++)
                    outptr0[k0 + k] = 0;
                continue;
            }

            volatile double scale_fp64 = 127.0 / (double)absmax;
            const float scale = (float)scale_fp64;
            descale_ptr[g] = absmax / 127.f;

            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v = ptrA[(size_t)k * A_hstep];
                if (input_scale_ptr)
                {
                    v *= input_scale_ptr[k];
                    // preserve multiplication order for consistent rounding
                    asm volatile(""
                                 : "+f"(v));
                }
                outptr0[k] = float2int8(v * scale);
            }
        }
    }
}

static void gemm_transB_packed_tile_wq_int8(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int K, int block_size)
{
    const signed char* pAT = AT_tile;
    const int A_hstep = AT_tile.w;
    const float* pAT_descales = AT_descales_tile;
    const int A_descales_hstep = AT_descales_tile.w;
    const signed char* pBT = BT_tile;
    const float* pBT_descales = BT_descales_tile;
    float* outptr = topT_tile;

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
                    __m256i _s2 = __lasx_xvmulwev_h_b(_pA, _pB1);
                    __m256i _s3 = __lasx_xvmulwev_h_b(_pA1, _pB1);
                    __m256i _s4 = __lasx_xvmulwev_h_b(_pA2, _pB0);
                    __m256i _s5 = __lasx_xvmulwev_h_b(_pA3, _pB0);
                    __m256i _s6 = __lasx_xvmulwev_h_b(_pA2, _pB1);
                    __m256i _s7 = __lasx_xvmulwev_h_b(_pA3, _pB1);
                    _s0 = __lasx_xvmaddwod_h_b(_s0, _pA, _pB0);
                    _s1 = __lasx_xvmaddwod_h_b(_s1, _pA1, _pB0);
                    _s2 = __lasx_xvmaddwod_h_b(_s2, _pA, _pB1);
                    _s3 = __lasx_xvmaddwod_h_b(_s3, _pA1, _pB1);
                    _s4 = __lasx_xvmaddwod_h_b(_s4, _pA2, _pB0);
                    _s5 = __lasx_xvmaddwod_h_b(_s5, _pA3, _pB0);
                    _s6 = __lasx_xvmaddwod_h_b(_s6, _pA2, _pB1);
                    _s7 = __lasx_xvmaddwod_h_b(_s7, _pA3, _pB1);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s0, _s0));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvhaddw_w_h(_s1, _s1));
                    _sum2 = __lasx_xvadd_w(_sum2, __lasx_xvhaddw_w_h(_s2, _s2));
                    _sum3 = __lasx_xvadd_w(_sum3, __lasx_xvhaddw_w_h(_s3, _s3));
                    _sum4 = __lasx_xvadd_w(_sum4, __lasx_xvhaddw_w_h(_s4, _s4));
                    _sum5 = __lasx_xvadd_w(_sum5, __lasx_xvhaddw_w_h(_s5, _s5));
                    _sum6 = __lasx_xvadd_w(_sum6, __lasx_xvhaddw_w_h(_s6, _s6));
                    _sum7 = __lasx_xvadd_w(_sum7, __lasx_xvhaddw_w_h(_s7, _s7));
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
                    __m256i _s2 = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA, _pB1), _pA, _pB1);
                    __m256i _s3 = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    __m256i _s4 = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA2, _pB0), _pA2, _pB0);
                    __m256i _s5 = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA3, _pB0), _pA3, _pB0);
                    __m256i _s6 = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA2, _pB1), _pA2, _pB1);
                    __m256i _s7 = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA3, _pB1), _pA3, _pB1);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(_s0));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_vext2xv_w_h(_s1));
                    _sum2 = __lasx_xvadd_w(_sum2, __lasx_vext2xv_w_h(_s2));
                    _sum3 = __lasx_xvadd_w(_sum3, __lasx_vext2xv_w_h(_s3));
                    _sum4 = __lasx_xvadd_w(_sum4, __lasx_vext2xv_w_h(_s4));
                    _sum5 = __lasx_xvadd_w(_sum5, __lasx_vext2xv_w_h(_s5));
                    _sum6 = __lasx_xvadd_w(_sum6, __lasx_vext2xv_w_h(_s6));
                    _sum7 = __lasx_xvadd_w(_sum7, __lasx_vext2xv_w_h(_s7));
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
                    __m256i _s2 = __lasx_xvmul_h(_pA, _pB1);
                    __m256i _s3 = __lasx_xvmul_h(_pA1, _pB1);
                    __m256i _s4 = __lasx_xvmul_h(_pA2, _pB0);
                    __m256i _s5 = __lasx_xvmul_h(_pA3, _pB0);
                    __m256i _s6 = __lasx_xvmul_h(_pA2, _pB1);
                    __m256i _s7 = __lasx_xvmul_h(_pA3, _pB1);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(_s0));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_vext2xv_w_h(_s1));
                    _sum2 = __lasx_xvadd_w(_sum2, __lasx_vext2xv_w_h(_s2));
                    _sum3 = __lasx_xvadd_w(_sum3, __lasx_vext2xv_w_h(_s3));
                    _sum4 = __lasx_xvadd_w(_sum4, __lasx_vext2xv_w_h(_s4));
                    _sum5 = __lasx_xvadd_w(_sum5, __lasx_vext2xv_w_h(_s5));
                    _sum6 = __lasx_xvadd_w(_sum6, __lasx_vext2xv_w_h(_s6));
                    _sum7 = __lasx_xvadd_w(_sum7, __lasx_vext2xv_w_h(_s7));
                    pB += 8;
                    pA += 8;
                }

                __m256 _bscale = (__m256)__lasx_xvld(pB_descales, 0);
                __m256 _bscale1 = (__m256)__lasx_xvshuf4i_w((__m256i)_bscale, _LSX_SHUFFLE(1, 0, 3, 2));
                __m256 _ascale = (__m256)__lasx_xvld(pA_descales, 0);
                __m256 _ascale1 = (__m256)__lasx_xvshuf4i_w((__m256i)_ascale, _LSX_SHUFFLE(0, 3, 2, 1));
                __m256 _ascale2 = (__m256)__lasx_xvpermi_q((__m256i)_ascale, (__m256i)_ascale, _LSX_SHUFFLE(0, 0, 0, 1));
                __m256 _ascale3 = (__m256)__lasx_xvshuf4i_w((__m256i)_ascale2, _LSX_SHUFFLE(0, 3, 2, 1));
                __m256 _out0 = k == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr, 0);
                __m256 _out1 = k == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 8, 0);
                __m256 _out2 = k == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 16, 0);
                __m256 _out3 = k == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 24, 0);
                __m256 _out4 = k == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 32, 0);
                __m256 _out5 = k == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 40, 0);
                __m256 _out6 = k == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 48, 0);
                __m256 _out7 = k == 0 ? (__m256)__lasx_xvldi(0) : (__m256)__lasx_xvld(outptr + 56, 0);
                _out0 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum0), __lasx_xvfmul_s(_ascale, _bscale), _out0);
                _out1 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum1), __lasx_xvfmul_s(_ascale1, _bscale), _out1);
                _out2 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum2), __lasx_xvfmul_s(_ascale, _bscale1), _out2);
                _out3 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum3), __lasx_xvfmul_s(_ascale1, _bscale1), _out3);
                _out4 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum4), __lasx_xvfmul_s(_ascale2, _bscale), _out4);
                _out5 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum5), __lasx_xvfmul_s(_ascale3, _bscale), _out5);
                _out6 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum6), __lasx_xvfmul_s(_ascale2, _bscale1), _out6);
                _out7 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum7), __lasx_xvfmul_s(_ascale3, _bscale1), _out7);
                __lasx_xvst(_out0, outptr, 0);
                __lasx_xvst(_out1, outptr + 8, 0);
                __lasx_xvst(_out2, outptr + 16, 0);
                __lasx_xvst(_out3, outptr + 24, 0);
                __lasx_xvst(_out4, outptr + 32, 0);
                __lasx_xvst(_out5, outptr + 40, 0);
                __lasx_xvst(_out6, outptr + 48, 0);
                __lasx_xvst(_out7, outptr + 56, 0);
                pA_descales += 8;
                pB_descales += 8;
            }
            outptr += 64;
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
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
                __m128 _out00 = k == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr, 0);
                __m128 _out01 = k == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 4, 0);
                __m128 _out10 = k == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 8, 0);
                __m128 _out11 = k == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 12, 0);
                __m128 _out20 = k == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 16, 0);
                __m128 _out21 = k == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 20, 0);
                __m128 _out30 = k == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 24, 0);
                __m128 _out31 = k == 0 ? (__m128)__lsx_vldi(0) : (__m128)__lsx_vld(outptr + 28, 0);
                _out00 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum00), __lsx_vfmul_s(_ascale0, _bscale), _out00);
                _out01 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum01), __lsx_vfmul_s(_ascale1, _bscale), _out01);
                _out10 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum10), __lsx_vfmul_s(_ascale0, _bscaler), _out10);
                _out11 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum11), __lsx_vfmul_s(_ascale1, _bscaler), _out11);
                _out20 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum20), __lsx_vfmul_s(_ascale0r, _bscale), _out20);
                _out21 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum21), __lsx_vfmul_s(_ascale1r, _bscale), _out21);
                _out30 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum30), __lsx_vfmul_s(_ascale0r, _bscaler), _out30);
                _out31 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum31), __lsx_vfmul_s(_ascale1r, _bscaler), _out31);
                __lsx_vst((__m128i)_out00, outptr, 0);
                __lsx_vst((__m128i)_out01, outptr + 4, 0);
                __lsx_vst((__m128i)_out10, outptr + 8, 0);
                __lsx_vst((__m128i)_out11, outptr + 12, 0);
                __lsx_vst((__m128i)_out20, outptr + 16, 0);
                __lsx_vst((__m128i)_out21, outptr + 20, 0);
                __lsx_vst((__m128i)_out30, outptr + 24, 0);
                __lsx_vst((__m128i)_out31, outptr + 28, 0);
                pA_descales += 8;
                pB_descales += 4;
            }
            outptr += 32;
        }

        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _out00 = (__m128)__lsx_vldi(0);
            __m128 _out01 = (__m128)__lsx_vldi(0);
            __m128 _out10 = (__m128)__lsx_vldi(0);
            __m128 _out11 = (__m128)__lsx_vldi(0);
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
                    __m128i _pB = __lsx_vreplgr2vr_w(*(const int*)pB);
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
                    __m128i _pB0 = __lsx_vreplgr2vr_h((signed char)pB[0]);
                    __m128i _pB1 = __lsx_vreplgr2vr_h((signed char)pB[1]);
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
        }
        for (; jj < max_jj; jj++)
        {
            __m128 _out0 = (__m128)__lsx_vldi(0);
            __m128 _out1 = (__m128)__lsx_vldi(0);
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
                    __m128i _pB = __lsx_vreplgr2vr_w(*(const int*)pB);
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
                    __m128i _pB = __lsx_vreplgr2vr_h((unsigned char)pB[0] | ((unsigned char)pB[1] << 8));
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
        }

        pAT += K * 8;
        pAT_descales += (K + block_size - 1) / block_size * 8;
    }
#endif // __loongarch_sx
#if __loongarch_sx
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;

        int jj = 0;
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            const signed char* pB0 = pB;
            const signed char* pB1 = pB + 8 * K;
            const float* pB_descales0 = pB_descales;
            const float* pB_descales1 = pB_descales + 8 * ((K + block_size - 1) / block_size);
            __m256 _out00 = (__m256)__lasx_xvldi(0);
            __m256 _out01 = (__m256)__lasx_xvldi(0);
            __m256 _out10 = (__m256)__lasx_xvldi(0);
            __m256 _out11 = (__m256)__lasx_xvldi(0);
            __m256 _out20 = (__m256)__lasx_xvldi(0);
            __m256 _out21 = (__m256)__lasx_xvldi(0);
            __m256 _out30 = (__m256)__lasx_xvldi(0);
            __m256 _out31 = (__m256)__lasx_xvldi(0);

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
                    __m256i _pB0 = __lasx_xvld(pB0, 0);
                    __m256i _pB1 = __lasx_xvld(pB1, 0);
                    __m128i _pA = __lsx_vld(pA, 0);
                    __m128i _pA0_128 = __lsx_vreplvei_w(_pA, 0);
                    __m128i _pA1_128 = __lsx_vreplvei_w(_pA, 1);
                    __m128i _pA2_128 = __lsx_vreplvei_w(_pA, 2);
                    __m128i _pA3_128 = __lsx_vreplvei_w(_pA, 3);
                    __m256i _pA0 = __lasx_concat_128(_pA0_128, _pA0_128);
                    __m256i _pA1 = __lasx_concat_128(_pA1_128, _pA1_128);
                    __m256i _pA2 = __lasx_concat_128(_pA2_128, _pA2_128);
                    __m256i _pA3 = __lasx_concat_128(_pA3_128, _pA3_128);
                    __m256i _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA0, _pB0), _pA0, _pB0);
                    _sum00 = __lasx_xvadd_w(_sum00, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA0, _pB1), _pA0, _pB1);
                    _sum01 = __lasx_xvadd_w(_sum01, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA1, _pB0), _pA1, _pB0);
                    _sum10 = __lasx_xvadd_w(_sum10, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    _sum11 = __lasx_xvadd_w(_sum11, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA2, _pB0), _pA2, _pB0);
                    _sum20 = __lasx_xvadd_w(_sum20, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA2, _pB1), _pA2, _pB1);
                    _sum21 = __lasx_xvadd_w(_sum21, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA3, _pB0), _pA3, _pB0);
                    _sum30 = __lasx_xvadd_w(_sum30, __lasx_xvhaddw_w_h(_s, _s));
                    _s = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA3, _pB1), _pA3, _pB1);
                    _sum31 = __lasx_xvadd_w(_sum31, __lasx_xvhaddw_w_h(_s, _s));
                    pB0 += 32;
                    pB1 += 32;
                    pA += 16;
                }
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
                    const int a0123 = __lsx_vpickve2gr_w(__lsx_vldrepl_w(pA, 0), 0);
                    __m128i _s = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)a0123), _pB0);
                    _sum00 = __lasx_xvadd_w(_sum00, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)a0123), _pB1);
                    _sum01 = __lasx_xvadd_w(_sum01, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)(a0123 >> 8)), _pB0);
                    _sum10 = __lasx_xvadd_w(_sum10, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)(a0123 >> 8)), _pB1);
                    _sum11 = __lasx_xvadd_w(_sum11, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)(a0123 >> 16)), _pB0);
                    _sum20 = __lasx_xvadd_w(_sum20, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)(a0123 >> 16)), _pB1);
                    _sum21 = __lasx_xvadd_w(_sum21, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)(a0123 >> 24)), _pB0);
                    _sum30 = __lasx_xvadd_w(_sum30, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)(a0123 >> 24)), _pB1);
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

            pB = pB1;
            pB_descales = pB_descales1;

            __lasx_xvst(_out00, outptr + (ii + 0) * max_jj + jj, 0);
            __lasx_xvst(_out01, outptr + (ii + 0) * max_jj + jj + 8, 0);
            __lasx_xvst(_out10, outptr + (ii + 1) * max_jj + jj, 0);
            __lasx_xvst(_out11, outptr + (ii + 1) * max_jj + jj + 8, 0);
            __lasx_xvst(_out20, outptr + (ii + 2) * max_jj + jj, 0);
            __lasx_xvst(_out21, outptr + (ii + 2) * max_jj + jj + 8, 0);
            __lasx_xvst(_out30, outptr + (ii + 3) * max_jj + jj, 0);
            __lasx_xvst(_out31, outptr + (ii + 3) * max_jj + jj + 8, 0);
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _out0 = (__m256)__lasx_xvldi(0);
            __m256 _out1 = (__m256)__lasx_xvldi(0);
            __m256 _out2 = (__m256)__lasx_xvldi(0);
            __m256 _out3 = (__m256)__lasx_xvldi(0);

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
                    __m256i _pB = __lasx_xvld(pB, 0);
                    __m128i _pA = __lsx_vld(pA, 0);
                    __m128i _pA0_128 = __lsx_vreplvei_w(_pA, 0);
                    __m128i _pA1_128 = __lsx_vreplvei_w(_pA, 1);
                    __m128i _pA2_128 = __lsx_vreplvei_w(_pA, 2);
                    __m128i _pA3_128 = __lsx_vreplvei_w(_pA, 3);
                    __m256i _pA0 = __lasx_concat_128(_pA0_128, _pA0_128);
                    __m256i _pA1 = __lasx_concat_128(_pA1_128, _pA1_128);
                    __m256i _pA2 = __lasx_concat_128(_pA2_128, _pA2_128);
                    __m256i _pA3 = __lasx_concat_128(_pA3_128, _pA3_128);
                    __m256i _s0 = __lasx_xvmulwev_h_b(_pA0, _pB);
                    __m256i _s1 = __lasx_xvmulwev_h_b(_pA1, _pB);
                    __m256i _s2 = __lasx_xvmulwev_h_b(_pA2, _pB);
                    __m256i _s3 = __lasx_xvmulwev_h_b(_pA3, _pB);
                    _s0 = __lasx_xvmaddwod_h_b(_s0, _pA0, _pB);
                    _s1 = __lasx_xvmaddwod_h_b(_s1, _pA1, _pB);
                    _s2 = __lasx_xvmaddwod_h_b(_s2, _pA2, _pB);
                    _s3 = __lasx_xvmaddwod_h_b(_s3, _pA3, _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s0, _s0));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvhaddw_w_h(_s1, _s1));
                    _sum2 = __lasx_xvadd_w(_sum2, __lasx_xvhaddw_w_h(_s2, _s2));
                    _sum3 = __lasx_xvadd_w(_sum3, __lasx_xvhaddw_w_h(_s3, _s3));
                    pB += 32;
                    pA += 16;
                }
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
                    const int a0123 = __lsx_vpickve2gr_w(__lsx_vldrepl_w(pA, 0), 0);
                    __m128i _s0 = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)a0123), _pB);
                    __m128i _s1 = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)(a0123 >> 8)), _pB);
                    __m128i _s2 = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)(a0123 >> 16)), _pB);
                    __m128i _s3 = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)(a0123 >> 24)), _pB);
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

            __lasx_xvst(_out0, outptr + (ii + 0) * max_jj + jj, 0);
            __lasx_xvst(_out1, outptr + (ii + 1) * max_jj + jj, 0);
            __lasx_xvst(_out2, outptr + (ii + 2) * max_jj + jj, 0);
            __lasx_xvst(_out3, outptr + (ii + 3) * max_jj + jj, 0);
        }
#endif
#if __loongarch_sx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB0 = pB;
            const signed char* pB1 = pB + 4 * K;
            const float* pB_descales0 = pB_descales;
            const float* pB_descales1 = pB_descales + 4 * ((K + block_size - 1) / block_size);
            __m128 _out00 = (__m128)__lsx_vldi(0);
            __m128 _out01 = (__m128)__lsx_vldi(0);
            __m128 _out10 = (__m128)__lsx_vldi(0);
            __m128 _out11 = (__m128)__lsx_vldi(0);
            __m128 _out20 = (__m128)__lsx_vldi(0);
            __m128 _out21 = (__m128)__lsx_vldi(0);
            __m128 _out30 = (__m128)__lsx_vldi(0);
            __m128 _out31 = (__m128)__lsx_vldi(0);
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
                    __m128i _pB0 = __lsx_vld(pB0, 0);
                    __m128i _pB1 = __lsx_vld(pB1, 0);
                    __m128i _pA = __lsx_vld(pA, 0);
                    __m128i _pA0 = __lsx_vreplvei_w(_pA, 0);
                    __m128i _pA1 = __lsx_vreplvei_w(_pA, 1);
                    __m128i _pA2 = __lsx_vreplvei_w(_pA, 2);
                    __m128i _pA3 = __lsx_vreplvei_w(_pA, 3);
                    __m128i _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB0), _pA0, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB1), _pA0, _pB1);
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB0), _pA1, _pB0);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB1), _pA1, _pB1);
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA2, _pB0), _pA2, _pB0);
                    _sum20 = __lsx_vadd_w(_sum20, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA2, _pB1), _pA2, _pB1);
                    _sum21 = __lsx_vadd_w(_sum21, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA3, _pB0), _pA3, _pB0);
                    _sum30 = __lsx_vadd_w(_sum30, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA3, _pB1), _pA3, _pB1);
                    _sum31 = __lsx_vadd_w(_sum31, __lsx_vhaddw_w_h(_s, _s));
                    pB0 += 16;
                    pB1 += 16;
                    pA += 16;
                }
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
                    const int a0123 = __lsx_vpickve2gr_w(__lsx_vldrepl_w(pA, 0), 0);
                    __m128i _s = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)a0123), _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)a0123), _pB1);
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)(a0123 >> 8)), _pB0);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)(a0123 >> 8)), _pB1);
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)(a0123 >> 16)), _pB0);
                    _sum20 = __lsx_vadd_w(_sum20, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)(a0123 >> 16)), _pB1);
                    _sum21 = __lsx_vadd_w(_sum21, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)(a0123 >> 24)), _pB0);
                    _sum30 = __lsx_vadd_w(_sum30, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)(a0123 >> 24)), _pB1);
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
            pB = pB1;
            pB_descales = pB_descales1;
            __lsx_vst((__m128i)_out00, outptr + (ii + 0) * max_jj + jj, 0);
            __lsx_vst((__m128i)_out01, outptr + (ii + 0) * max_jj + jj + 4, 0);
            __lsx_vst((__m128i)_out10, outptr + (ii + 1) * max_jj + jj, 0);
            __lsx_vst((__m128i)_out11, outptr + (ii + 1) * max_jj + jj + 4, 0);
            __lsx_vst((__m128i)_out20, outptr + (ii + 2) * max_jj + jj, 0);
            __lsx_vst((__m128i)_out21, outptr + (ii + 2) * max_jj + jj + 4, 0);
            __lsx_vst((__m128i)_out30, outptr + (ii + 3) * max_jj + jj, 0);
            __lsx_vst((__m128i)_out31, outptr + (ii + 3) * max_jj + jj + 4, 0);
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _out0 = (__m128)__lsx_vldi(0);
            __m128 _out1 = (__m128)__lsx_vldi(0);
            __m128 _out2 = (__m128)__lsx_vldi(0);
            __m128 _out3 = (__m128)__lsx_vldi(0);
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
                    __m128i _pB = __lsx_vld(pB, 0);
                    __m128i _pA = __lsx_vld(pA, 0);
                    __m128i _pA0 = __lsx_vreplvei_w(_pA, 0);
                    __m128i _pA1 = __lsx_vreplvei_w(_pA, 1);
                    __m128i _pA2 = __lsx_vreplvei_w(_pA, 2);
                    __m128i _pA3 = __lsx_vreplvei_w(_pA, 3);
                    __m128i _s0 = __lsx_vmulwev_h_b(_pA0, _pB);
                    __m128i _s1 = __lsx_vmulwev_h_b(_pA1, _pB);
                    __m128i _s2 = __lsx_vmulwev_h_b(_pA2, _pB);
                    __m128i _s3 = __lsx_vmulwev_h_b(_pA3, _pB);
                    _s0 = __lsx_vmaddwod_h_b(_s0, _pA0, _pB);
                    _s1 = __lsx_vmaddwod_h_b(_s1, _pA1, _pB);
                    _s2 = __lsx_vmaddwod_h_b(_s2, _pA2, _pB);
                    _s3 = __lsx_vmaddwod_h_b(_s3, _pA3, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s1, _s1));
                    _sum2 = __lsx_vadd_w(_sum2, __lsx_vhaddw_w_h(_s2, _s2));
                    _sum3 = __lsx_vadd_w(_sum3, __lsx_vhaddw_w_h(_s3, _s3));
                    pB += 16;
                    pA += 16;
                }
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
                    const int a0123 = __lsx_vpickve2gr_w(__lsx_vldrepl_w(pA, 0), 0);
                    __m128i _s0 = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)a0123), _pB);
                    __m128i _s1 = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)(a0123 >> 8)), _pB);
                    __m128i _s2 = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)(a0123 >> 16)), _pB);
                    __m128i _s3 = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)(a0123 >> 24)), _pB);
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
            __lsx_vst((__m128i)_out0, outptr + (ii + 0) * max_jj + jj, 0);
            __lsx_vst((__m128i)_out1, outptr + (ii + 1) * max_jj + jj, 0);
            __lsx_vst((__m128i)_out2, outptr + (ii + 2) * max_jj + jj, 0);
            __lsx_vst((__m128i)_out3, outptr + (ii + 3) * max_jj + jj, 0);
        }
#endif
#if __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _out0 = (__m128)__lsx_vldi(0);
            __m128 _out1 = (__m128)__lsx_vldi(0);
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
                    int b01 = (unsigned char)pB[0] | ((unsigned char)pB[2] << 8);
                    __m128i _pB0 = __lsx_vreplgr2vr_w(b01);
                    _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                    _pB0 = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(1, 0, 1, 0));
                    __m128i _pB1 = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m128i _s0 = __lsx_vmul_h(_pA0, _pB0);
                    __m128i _s1 = __lsx_vmul_h(_pA0, _pB1);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    b01 = (unsigned char)pB[1] | ((unsigned char)pB[3] << 8);
                    _pB0 = __lsx_vreplgr2vr_w(b01);
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
                    int b01 = (unsigned char)pB[0] | ((unsigned char)pB[1] << 8);
                    __m128i _pB0 = __lsx_vreplgr2vr_w(b01);
                    _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                    _pB0 = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(1, 0, 1, 0));
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
                const __m128 _ascale = (__m128)__lsx_vld(pA_descales, 0);
                _out0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sumc0), __lsx_vfmul_s(_ascale, __lsx_vreplfr2vr_s(pB_descales[0])), _out0);
                _out1 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sumc1), __lsx_vfmul_s(_ascale, __lsx_vreplfr2vr_s(pB_descales[1])), _out1);
                pA_descales += 4;
                pB_descales += 2;
            }
            __lsx_vstelm_w((__m128i)_out0, outptr + (ii + 0) * max_jj + jj, 0, 0);
            __lsx_vstelm_w((__m128i)_out1, outptr + (ii + 0) * max_jj + jj + 1, 0, 0);
            __lsx_vstelm_w((__m128i)_out0, outptr + (ii + 1) * max_jj + jj, 0, 1);
            __lsx_vstelm_w((__m128i)_out1, outptr + (ii + 1) * max_jj + jj + 1, 0, 1);
            __lsx_vstelm_w((__m128i)_out0, outptr + (ii + 2) * max_jj + jj, 0, 2);
            __lsx_vstelm_w((__m128i)_out1, outptr + (ii + 2) * max_jj + jj + 1, 0, 2);
            __lsx_vstelm_w((__m128i)_out0, outptr + (ii + 3) * max_jj + jj, 0, 3);
            __lsx_vstelm_w((__m128i)_out1, outptr + (ii + 3) * max_jj + jj + 1, 0, 3);
        }
        for (; jj < max_jj; jj++)
        {
            __m128 _out0 = (__m128)__lsx_vldi(0);
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
                    __m128i _pB = __lsx_vreplgr2vr_h((unsigned char)pB[0] | ((unsigned char)pB[1] << 8));
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
                const __m128 _scale = __lsx_vfmul_s((__m128)__lsx_vld(pA_descales, 0), __lsx_vreplfr2vr_s(*pB_descales++));
                _out0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum0), _scale, _out0);
                pA_descales += 4;
            }
            __lsx_vstelm_w((__m128i)_out0, outptr + (ii + 0) * max_jj + jj, 0, 0);
            __lsx_vstelm_w((__m128i)_out0, outptr + (ii + 1) * max_jj + jj, 0, 1);
            __lsx_vstelm_w((__m128i)_out0, outptr + (ii + 2) * max_jj + jj, 0, 2);
            __lsx_vstelm_w((__m128i)_out0, outptr + (ii + 3) * max_jj + jj, 0, 3);
        }
#endif
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
            const signed char* pB0 = pB;
            const signed char* pB1 = pB + 8 * K;
            const float* pB_descales0 = pB_descales;
            const float* pB_descales1 = pB_descales + 8 * ((K + block_size - 1) / block_size);
            __m256 _out00 = (__m256)__lasx_xvldi(0);
            __m256 _out01 = (__m256)__lasx_xvldi(0);
            __m256 _out10 = (__m256)__lasx_xvldi(0);
            __m256 _out11 = (__m256)__lasx_xvldi(0);
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
                    __m128i _pAs = __lsx_vldrepl_d(pA, 0);
                    __m256i _pA0 = __lasx_xvreplgr2vr_w(__lsx_vpickve2gr_w(_pAs, 0));
                    __m256i _pA1 = __lasx_xvreplgr2vr_w(__lsx_vpickve2gr_w(_pAs, 1));
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
                    __m128i _pAs = __lsx_vldrepl_h(pA, 0);
                    const int a0 = (signed char)__lsx_vpickve2gr_b(_pAs, 0);
                    const int a1 = (signed char)__lsx_vpickve2gr_b(_pAs, 1);
                    __m128i _s = __lsx_vmul_h(__lsx_vreplgr2vr_h(a0), _pB0);
                    _sum00 = __lasx_xvadd_w(_sum00, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h(a0), _pB1);
                    _sum01 = __lasx_xvadd_w(_sum01, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h(a1), _pB0);
                    _sum10 = __lasx_xvadd_w(_sum10, __lasx_vext2xv_w_h(__lasx_cast_128(_s)));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h(a1), _pB1);
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
            pB = pB1;
            pB_descales = pB_descales1;
            __lasx_xvst(_out00, outptr + (ii + 0) * max_jj + jj, 0);
            __lasx_xvst(_out01, outptr + (ii + 0) * max_jj + jj + 8, 0);
            __lasx_xvst(_out10, outptr + (ii + 1) * max_jj + jj, 0);
            __lasx_xvst(_out11, outptr + (ii + 1) * max_jj + jj + 8, 0);
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _out0 = (__m256)__lasx_xvldi(0);
            __m256 _out1 = (__m256)__lasx_xvldi(0);
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
                    __m128i _pAs = __lsx_vldrepl_d(pA, 0);
                    __m256i _pA0 = __lasx_xvreplgr2vr_w(__lsx_vpickve2gr_w(_pAs, 0));
                    __m256i _pA1 = __lasx_xvreplgr2vr_w(__lsx_vpickve2gr_w(_pAs, 1));
                    __m256i _s0 = __lasx_xvmulwev_h_b(_pA0, _pB);
                    __m256i _s1 = __lasx_xvmulwev_h_b(_pA1, _pB);
                    _s0 = __lasx_xvmaddwod_h_b(_s0, _pA0, _pB);
                    _s1 = __lasx_xvmaddwod_h_b(_s1, _pA1, _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s0, _s0));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvhaddw_w_h(_s1, _s1));
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
                    __m128i _pAs = __lsx_vldrepl_h(pA, 0);
                    __m128i _s0 = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)__lsx_vpickve2gr_b(_pAs, 0)), _pB);
                    __m128i _s1 = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)__lsx_vpickve2gr_b(_pAs, 1)), _pB);
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
            __lasx_xvst(_out0, outptr + (ii + 0) * max_jj + jj, 0);
            __lasx_xvst(_out1, outptr + (ii + 1) * max_jj + jj, 0);
        }
#endif
#if __loongarch_sx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB0 = pB;
            const signed char* pB1 = pB + 4 * K;
            const float* pB_descales0 = pB_descales;
            const float* pB_descales1 = pB_descales + 4 * ((K + block_size - 1) / block_size);
            __m128 _out00 = (__m128)__lsx_vldi(0);
            __m128 _out01 = (__m128)__lsx_vldi(0);
            __m128 _out10 = (__m128)__lsx_vldi(0);
            __m128 _out11 = (__m128)__lsx_vldi(0);
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
                    __m128i _pAs = __lsx_vldrepl_d(pA, 0);
                    __m128i _pA0 = __lsx_vreplvei_w(_pAs, 0);
                    __m128i _pA1 = __lsx_vreplvei_w(_pAs, 1);
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
                    __m128i _pAs = __lsx_vldrepl_h(pA, 0);
                    const int a0 = (signed char)__lsx_vpickve2gr_b(_pAs, 0);
                    const int a1 = (signed char)__lsx_vpickve2gr_b(_pAs, 1);
                    __m128i _s = __lsx_vmul_h(__lsx_vreplgr2vr_h(a0), _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h(a0), _pB1);
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h(a1), _pB0);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(__lsx_vreplgr2vr_h(a1), _pB1);
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
            pB = pB1;
            pB_descales = pB_descales1;
            __lsx_vst((__m128i)_out00, outptr + (ii + 0) * max_jj + jj, 0);
            __lsx_vst((__m128i)_out01, outptr + (ii + 0) * max_jj + jj + 4, 0);
            __lsx_vst((__m128i)_out10, outptr + (ii + 1) * max_jj + jj, 0);
            __lsx_vst((__m128i)_out11, outptr + (ii + 1) * max_jj + jj + 4, 0);
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _out0 = (__m128)__lsx_vldi(0);
            __m128 _out1 = (__m128)__lsx_vldi(0);
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
                    __m128i _pAs = __lsx_vldrepl_d(pA, 0);
                    __m128i _pA0 = __lsx_vreplvei_w(_pAs, 0);
                    __m128i _pA1 = __lsx_vreplvei_w(_pAs, 1);
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB), _pA0, _pB);
                    __m128i _s1 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA1, _pB), _pA1, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s1, _s1));
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
                    __m128i _pAs = __lsx_vldrepl_h(pA, 0);
                    __m128i _s0 = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)__lsx_vpickve2gr_b(_pAs, 0)), _pB);
                    __m128i _s1 = __lsx_vmul_h(__lsx_vreplgr2vr_h((signed char)__lsx_vpickve2gr_b(_pAs, 1)), _pB);
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
            __lsx_vst((__m128i)_out0, outptr + (ii + 0) * max_jj + jj, 0);
            __lsx_vst((__m128i)_out1, outptr + (ii + 1) * max_jj + jj, 0);
        }
#endif
        for (; jj + 1 < max_jj; jj += 2)
        {
            float _out00 = 0.f;
            float _out01 = 0.f;
            float _out10 = 0.f;
            float _out11 = 0.f;
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
            outptr[(ii + 0) * max_jj + jj] = _out00;
            outptr[(ii + 0) * max_jj + jj + 1] = _out01;
            outptr[(ii + 1) * max_jj + jj] = _out10;
            outptr[(ii + 1) * max_jj + jj + 1] = _out11;
        }
        for (; jj < max_jj; jj++)
        {
            float _out0 = 0.f;
            float _out1 = 0.f;
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
            outptr[(ii + 0) * max_jj + jj] = _out0;
            outptr[(ii + 1) * max_jj + jj] = _out1;
        }
        pAT += A_hstep * 2;
        pAT_descales += A_descales_hstep * 2;
    }
#endif // __loongarch_sx
    for (; ii < max_ii; ii++)
    {
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;

        int jj = 0;
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            const signed char* pB0 = pB;
            const signed char* pB1 = pB + 8 * K;
            const float* pB_descales0 = pB_descales;
            const float* pB_descales1 = pB_descales + 8 * ((K + block_size - 1) / block_size);
            __m256 _out00 = (__m256)__lasx_xvldi(0);
            __m256 _out01 = (__m256)__lasx_xvldi(0);
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
            pB = pB1;
            pB_descales = pB_descales1;
            __lasx_xvst(_out00, outptr + ii * max_jj + jj, 0);
            __lasx_xvst(_out01, outptr + ii * max_jj + jj + 8, 0);
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _out0 = (__m256)__lasx_xvldi(0);
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
            __lasx_xvst(_out0, outptr + ii * max_jj + jj, 0);
        }
#endif
#if __loongarch_sx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB0 = pB;
            const signed char* pB1 = pB + 4 * K;
            const float* pB_descales0 = pB_descales;
            const float* pB_descales1 = pB_descales + 4 * ((K + block_size - 1) / block_size);
            __m128 _out00 = (__m128)__lsx_vldi(0);
            __m128 _out01 = (__m128)__lsx_vldi(0);
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
            pB = pB1;
            pB_descales = pB_descales1;
            __lsx_vst((__m128i)_out00, outptr + ii * max_jj + jj, 0);
            __lsx_vst((__m128i)_out01, outptr + ii * max_jj + jj + 4, 0);
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _out0 = (__m128)__lsx_vldi(0);
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
            __lsx_vst((__m128i)_out0, outptr + ii * max_jj + jj, 0);
        }
#endif
        for (; jj + 1 < max_jj; jj += 2)
        {
            float _out0 = 0.f;
            float _out1 = 0.f;
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
            outptr[ii * max_jj + jj] = _out0;
            outptr[ii * max_jj + jj + 1] = _out1;
        }
        for (; jj < max_jj; jj++)
        {
            float _out0 = 0.f;
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
            outptr[ii * max_jj + jj] = _out0;
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
        const __m256 _alpha256 = (__m256)__lasx_xvreplfr2vr_s(alpha);
        const __m256 _beta256 = (__m256)__lasx_xvreplfr2vr_s(beta);
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
                    const __m256 _c = (__m256)__lasx_xvreplfr2vr_s(c0);
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
                    const __m256 _c0 = (__m256)__lasx_xvld(pC, 0);
                    const __m256 _c1 = (__m256)__lasx_xvld(pC + c_hstep, 0);
                    const __m256 _c2 = (__m256)__lasx_xvld(pC + c_hstep * 2, 0);
                    const __m256 _c3 = (__m256)__lasx_xvld(pC + c_hstep * 3, 0);
                    const __m256 _c4 = (__m256)__lasx_xvld(pC + c_hstep * 4, 0);
                    const __m256 _c5 = (__m256)__lasx_xvld(pC + c_hstep * 5, 0);
                    const __m256 _c6 = (__m256)__lasx_xvld(pC + c_hstep * 6, 0);
                    const __m256 _c7 = (__m256)__lasx_xvld(pC + c_hstep * 7, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
        }
#endif
        const __m128 _alpha128 = __lsx_vreplfr2vr_s(alpha);
        const __m128 _beta128 = __lsx_vreplfr2vr_s(beta);
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
                    const __m128 _c = __lsx_vreplfr2vr_s(c0);
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
                    const __m128 _c0 = (__m128)__lsx_vld(pC, 0);
                    const __m128 _c1 = (__m128)__lsx_vld(pC + c_hstep, 0);
                    const __m128 _c2 = (__m128)__lsx_vld(pC + c_hstep * 2, 0);
                    const __m128 _c3 = (__m128)__lsx_vld(pC + c_hstep * 3, 0);
                    const __m128 _c4 = (__m128)__lsx_vld(pC + c_hstep * 4, 0);
                    const __m128 _c5 = (__m128)__lsx_vld(pC + c_hstep * 5, 0);
                    const __m128 _c6 = (__m128)__lsx_vld(pC + c_hstep * 6, 0);
                    const __m128 _c7 = (__m128)__lsx_vld(pC + c_hstep * 7, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 4;
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
                    const __m128 _c = __lsx_vreplfr2vr_s(c0);
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
                    const __m128 _c0 = (__m128)__lsx_vldrepl_d(pC, 0);
                    const __m128 _c1 = (__m128)__lsx_vldrepl_d(pC + c_hstep, 0);
                    const __m128 _c2 = (__m128)__lsx_vldrepl_d(pC + c_hstep * 2, 0);
                    const __m128 _c3 = (__m128)__lsx_vldrepl_d(pC + c_hstep * 3, 0);
                    const __m128 _c4 = (__m128)__lsx_vldrepl_d(pC + c_hstep * 4, 0);
                    const __m128 _c5 = (__m128)__lsx_vldrepl_d(pC + c_hstep * 5, 0);
                    const __m128 _c6 = (__m128)__lsx_vldrepl_d(pC + c_hstep * 6, 0);
                    const __m128 _c7 = (__m128)__lsx_vldrepl_d(pC + c_hstep * 7, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
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
                    const __m128 _c = __lsx_vreplfr2vr_s(c0);
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
                }
                if (broadcast_type_C == 4)
                {
                    const __m128 _c = __lsx_vreplfr2vr_s(beta == 1.f ? pC[0] : pC[0] * beta);
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f4 = __lsx_vfadd_s(_f4, _c);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
        }
    }
#endif // __loongarch_sx
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
        const __m256 _alpha256 = (__m256)__lasx_xvreplfr2vr_s(alpha);
        const __m256 _beta256 = (__m256)__lasx_xvreplfr2vr_s(beta);
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _f00 = (__m256)__lasx_xvld(pp + (ii + 0) * max_jj + jj, 0);
            __m256 _f01 = (__m256)__lasx_xvld(pp + (ii + 0) * max_jj + jj + 8, 0);
            __m256 _f10 = (__m256)__lasx_xvld(pp + (ii + 1) * max_jj + jj, 0);
            __m256 _f11 = (__m256)__lasx_xvld(pp + (ii + 1) * max_jj + jj + 8, 0);
            __m256 _f20 = (__m256)__lasx_xvld(pp + (ii + 2) * max_jj + jj, 0);
            __m256 _f21 = (__m256)__lasx_xvld(pp + (ii + 2) * max_jj + jj + 8, 0);
            __m256 _f30 = (__m256)__lasx_xvld(pp + (ii + 3) * max_jj + jj, 0);
            __m256 _f31 = (__m256)__lasx_xvld(pp + (ii + 3) * max_jj + jj + 8, 0);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    const __m256 _c = (__m256)__lasx_xvreplfr2vr_s(c0);
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
                    const __m256 _c0 = (__m256)__lasx_xvreplfr2vr_s(c0);
                    const __m256 _c1 = (__m256)__lasx_xvreplfr2vr_s(c1);
                    const __m256 _c2 = (__m256)__lasx_xvreplfr2vr_s(c2);
                    const __m256 _c3 = (__m256)__lasx_xvreplfr2vr_s(c3);
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
                    const __m256 _c00 = (__m256)__lasx_xvld(pC, 0);
                    const __m256 _c01 = (__m256)__lasx_xvld(pC + 8, 0);
                    const __m256 _c10 = (__m256)__lasx_xvld(pC + c_hstep, 0);
                    const __m256 _c11 = (__m256)__lasx_xvld(pC + c_hstep + 8, 0);
                    const __m256 _c20 = (__m256)__lasx_xvld(pC + c_hstep * 2, 0);
                    const __m256 _c21 = (__m256)__lasx_xvld(pC + c_hstep * 2 + 8, 0);
                    const __m256 _c30 = (__m256)__lasx_xvld(pC + c_hstep * 3, 0);
                    const __m256 _c31 = (__m256)__lasx_xvld(pC + c_hstep * 3 + 8, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = (__m256)__lasx_xvld(pp + (ii + 0) * max_jj + jj, 0);
            __m256 _f1 = (__m256)__lasx_xvld(pp + (ii + 1) * max_jj + jj, 0);
            __m256 _f2 = (__m256)__lasx_xvld(pp + (ii + 2) * max_jj + jj, 0);
            __m256 _f3 = (__m256)__lasx_xvld(pp + (ii + 3) * max_jj + jj, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
        }
#endif
#if __loongarch_sx
        const __m128 _alpha128 = __lsx_vreplfr2vr_s(alpha);
        const __m128 _beta128 = __lsx_vreplfr2vr_s(beta);
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f00 = (__m128)__lsx_vld(pp + (ii + 0) * max_jj + jj, 0);
            __m128 _f01 = (__m128)__lsx_vld(pp + (ii + 0) * max_jj + jj + 4, 0);
            __m128 _f10 = (__m128)__lsx_vld(pp + (ii + 1) * max_jj + jj, 0);
            __m128 _f11 = (__m128)__lsx_vld(pp + (ii + 1) * max_jj + jj + 4, 0);
            __m128 _f20 = (__m128)__lsx_vld(pp + (ii + 2) * max_jj + jj, 0);
            __m128 _f21 = (__m128)__lsx_vld(pp + (ii + 2) * max_jj + jj + 4, 0);
            __m128 _f30 = (__m128)__lsx_vld(pp + (ii + 3) * max_jj + jj, 0);
            __m128 _f31 = (__m128)__lsx_vld(pp + (ii + 3) * max_jj + jj + 4, 0);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    const __m128 _c = __lsx_vreplfr2vr_s(c0);
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
                    const __m128 _c0 = __lsx_vreplfr2vr_s(c0);
                    const __m128 _c1 = __lsx_vreplfr2vr_s(c1);
                    const __m128 _c2 = __lsx_vreplfr2vr_s(c2);
                    const __m128 _c3 = __lsx_vreplfr2vr_s(c3);
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
                    const __m128 _c00 = (__m128)__lsx_vld(pC, 0);
                    const __m128 _c01 = (__m128)__lsx_vld(pC + 4, 0);
                    const __m128 _c10 = (__m128)__lsx_vld(pC + c_hstep, 0);
                    const __m128 _c11 = (__m128)__lsx_vld(pC + c_hstep + 4, 0);
                    const __m128 _c20 = (__m128)__lsx_vld(pC + c_hstep * 2, 0);
                    const __m128 _c21 = (__m128)__lsx_vld(pC + c_hstep * 2 + 4, 0);
                    const __m128 _c30 = (__m128)__lsx_vld(pC + c_hstep * 3, 0);
                    const __m128 _c31 = (__m128)__lsx_vld(pC + c_hstep * 3 + 4, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp + (ii + 0) * max_jj + jj, 0);
            __m128 _f1 = (__m128)__lsx_vld(pp + (ii + 1) * max_jj + jj, 0);
            __m128 _f2 = (__m128)__lsx_vld(pp + (ii + 2) * max_jj + jj, 0);
            __m128 _f3 = (__m128)__lsx_vld(pp + (ii + 3) * max_jj + jj, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _f0 = (__m128)__lsx_vldrepl_d(pp + (ii + 0) * max_jj + jj, 0);
            __m128 _f1 = (__m128)__lsx_vldrepl_d(pp + (ii + 1) * max_jj + jj, 0);
            __m128 _f2 = (__m128)__lsx_vldrepl_d(pp + (ii + 2) * max_jj + jj, 0);
            __m128 _f3 = (__m128)__lsx_vldrepl_d(pp + (ii + 3) * max_jj + jj, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
        }
#endif
#if __loongarch_sx
        for (; jj < max_jj; jj++)
        {
            __m128i _fi = __lsx_vreplgr2vr_w(((const int*)(pp + (ii + 0) * max_jj + jj))[0]);
            _fi = __lsx_vinsgr2vr_w(_fi, ((const int*)(pp + (ii + 1) * max_jj + jj))[0], 1);
            _fi = __lsx_vinsgr2vr_w(_fi, ((const int*)(pp + (ii + 2) * max_jj + jj))[0], 2);
            _fi = __lsx_vinsgr2vr_w(_fi, ((const int*)(pp + (ii + 3) * max_jj + jj))[0], 3);
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
                }
                if (broadcast_type_C == 4)
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(beta == 1.f ? pC[0] : pC[0] * beta));
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
        }
#else
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[(ii + 0) * max_jj + jj];
            float f1 = pp[(ii + 1) * max_jj + jj];
            float f2 = pp[(ii + 2) * max_jj + jj];
            float f3 = pp[(ii + 3) * max_jj + jj];
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    f0 += c0;
                    f1 += c0;
                    f2 += c0;
                    f3 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                    f1 += c1;
                    f2 += c2;
                    f3 += c3;
                }
                if (broadcast_type_C == 3)
                {
                    if (beta == 1.f)
                    {
                        f0 += pC[0];
                        f1 += pC[c_hstep];
                        f2 += pC[c_hstep * 2];
                        f3 += pC[c_hstep * 3];
                    }
                    else
                    {
                        f0 += pC[0] * beta;
                        f1 += pC[c_hstep] * beta;
                        f2 += pC[c_hstep * 2] * beta;
                        f3 += pC[c_hstep * 3] * beta;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float c = beta == 1.f ? pC[0] : pC[0] * beta;
                    f0 += c;
                    f1 += c;
                    f2 += c;
                    f3 += c;
                }
            }
            if (alpha != 1.f)
            {
                f0 *= alpha;
                f1 *= alpha;
                f2 *= alpha;
                f3 *= alpha;
            }
            p0[0] = f0;
            p1[0] = f1;
            p2[0] = f2;
            p3[0] = f3;
            p0++;
            p1++;
            p2++;
            p3++;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
        }
#endif
    }
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
#if __loongarch_asx
        const __m256 _alpha256 = (__m256)__lasx_xvreplfr2vr_s(alpha);
        const __m256 _beta256 = (__m256)__lasx_xvreplfr2vr_s(beta);
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _f00 = (__m256)__lasx_xvld(pp + (ii + 0) * max_jj + jj, 0);
            __m256 _f01 = (__m256)__lasx_xvld(pp + (ii + 0) * max_jj + jj + 8, 0);
            __m256 _f10 = (__m256)__lasx_xvld(pp + (ii + 1) * max_jj + jj, 0);
            __m256 _f11 = (__m256)__lasx_xvld(pp + (ii + 1) * max_jj + jj + 8, 0);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    const __m256 _c = (__m256)__lasx_xvreplfr2vr_s(c0);
                    _f00 = __lasx_xvfadd_s(_f00, _c);
                    _f01 = __lasx_xvfadd_s(_f01, _c);
                    _f10 = __lasx_xvfadd_s(_f10, _c);
                    _f11 = __lasx_xvfadd_s(_f11, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    const __m256 _c0 = (__m256)__lasx_xvreplfr2vr_s(c0);
                    const __m256 _c1 = (__m256)__lasx_xvreplfr2vr_s(c1);
                    _f00 = __lasx_xvfadd_s(_f00, _c0);
                    _f01 = __lasx_xvfadd_s(_f01, _c0);
                    _f10 = __lasx_xvfadd_s(_f10, _c1);
                    _f11 = __lasx_xvfadd_s(_f11, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    const __m256 _c00 = (__m256)__lasx_xvld(pC, 0);
                    const __m256 _c01 = (__m256)__lasx_xvld(pC + 8, 0);
                    const __m256 _c10 = (__m256)__lasx_xvld(pC + c_hstep, 0);
                    const __m256 _c11 = (__m256)__lasx_xvld(pC + c_hstep + 8, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = (__m256)__lasx_xvld(pp + (ii + 0) * max_jj + jj, 0);
            __m256 _f1 = (__m256)__lasx_xvld(pp + (ii + 1) * max_jj + jj, 0);
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
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c = (__m256)__lasx_xvld(pC, 0);
                    if (beta != 1.f)
                        _c = __lasx_xvfmul_s(_c, _beta256);
                    _f0 = __lasx_xvfadd_s(_f0, _c);
                    _f1 = __lasx_xvfadd_s(_f1, _c);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
        }
#endif
#if __loongarch_sx
        const __m128 _alpha128 = __lsx_vreplfr2vr_s(alpha);
        const __m128 _beta128 = __lsx_vreplfr2vr_s(beta);
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f00 = (__m128)__lsx_vld(pp + (ii + 0) * max_jj + jj, 0);
            __m128 _f01 = (__m128)__lsx_vld(pp + (ii + 0) * max_jj + jj + 4, 0);
            __m128 _f10 = (__m128)__lsx_vld(pp + (ii + 1) * max_jj + jj, 0);
            __m128 _f11 = (__m128)__lsx_vld(pp + (ii + 1) * max_jj + jj + 4, 0);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    const __m128 _c = __lsx_vreplfr2vr_s(c0);
                    _f00 = __lsx_vfadd_s(_f00, _c);
                    _f01 = __lsx_vfadd_s(_f01, _c);
                    _f10 = __lsx_vfadd_s(_f10, _c);
                    _f11 = __lsx_vfadd_s(_f11, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    const __m128 _c0 = __lsx_vreplfr2vr_s(c0);
                    const __m128 _c1 = __lsx_vreplfr2vr_s(c1);
                    _f00 = __lsx_vfadd_s(_f00, _c0);
                    _f01 = __lsx_vfadd_s(_f01, _c0);
                    _f10 = __lsx_vfadd_s(_f10, _c1);
                    _f11 = __lsx_vfadd_s(_f11, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    const __m128 _c00 = (__m128)__lsx_vld(pC, 0);
                    const __m128 _c01 = (__m128)__lsx_vld(pC + 4, 0);
                    const __m128 _c10 = (__m128)__lsx_vld(pC + c_hstep, 0);
                    const __m128 _c11 = (__m128)__lsx_vld(pC + c_hstep + 4, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp + (ii + 0) * max_jj + jj, 0);
            __m128 _f1 = (__m128)__lsx_vld(pp + (ii + 1) * max_jj + jj, 0);
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
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = (__m128)__lsx_vld(pC, 0);
                    if (beta != 1.f)
                        _c = __lsx_vfmul_s(_c, _beta128);
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _f0 = (__m128)__lsx_vldrepl_d(pp + (ii + 0) * max_jj + jj, 0);
            __m128 _f1 = (__m128)__lsx_vldrepl_d(pp + (ii + 1) * max_jj + jj, 0);
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
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = (__m128)__lsx_vldrepl_d(pC, 0);
                    if (beta != 1.f)
                        _c = __lsx_vfmul_s(_c, _beta128);
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
        }
#endif
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[(ii + 0) * max_jj + jj];
            float f1 = pp[(ii + 1) * max_jj + jj];
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
                }
                if (broadcast_type_C == 4)
                {
                    float c = beta == 1.f ? pC[0] : pC[0] * beta;
                    f0 += c;
                    f1 += c;
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
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
#if __loongarch_asx
        const __m256 _alpha256 = (__m256)__lasx_xvreplfr2vr_s(alpha);
        const __m256 _beta256 = (__m256)__lasx_xvreplfr2vr_s(beta);
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _f0 = (__m256)__lasx_xvld(pp + ii * max_jj + jj, 0);
            __m256 _f1 = (__m256)__lasx_xvld(pp + ii * max_jj + jj + 8, 0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    const __m256 _c = (__m256)__lasx_xvreplfr2vr_s(c0);
                    _f0 = __lasx_xvfadd_s(_f0, _c);
                    _f1 = __lasx_xvfadd_s(_f1, _c);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    const __m256 _c0 = (__m256)__lasx_xvld(pC, 0);
                    const __m256 _c1 = (__m256)__lasx_xvld(pC + 8, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = (__m256)__lasx_xvld(pp + ii * max_jj + jj, 0);
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
                }
            }
            if (alpha != 1.f) _f0 = __lasx_xvfmul_s(_f0, _alpha256);
            __lasx_xvst(_f0, p0, 0);
            p0 += 8;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
        }
#endif
#if __loongarch_sx
        const __m128 _alpha128 = __lsx_vreplfr2vr_s(alpha);
        const __m128 _beta128 = __lsx_vreplfr2vr_s(beta);
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp + ii * max_jj + jj, 0);
            __m128 _f1 = (__m128)__lsx_vld(pp + ii * max_jj + jj + 4, 0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    const __m128 _c = __lsx_vreplfr2vr_s(c0);
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    const __m128 _c0 = (__m128)__lsx_vld(pC, 0);
                    const __m128 _c1 = (__m128)__lsx_vld(pC + 4, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp + ii * max_jj + jj, 0);
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
                }
            }
            if (alpha != 1.f) _f0 = __lsx_vfmul_s(_f0, _alpha128);
            __lsx_vst((__m128i)_f0, p0, 0);
            p0 += 4;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _f0 = (__m128)__lsx_vldrepl_d(pp + ii * max_jj + jj, 0);
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
                }
            }
            if (alpha != 1.f) _f0 = __lsx_vfmul_s(_f0, _alpha128);
            __lsx_vstelm_d((__m128i)_f0, p0, 0, 0);
            p0 += 2;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
        }
#endif
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[ii * max_jj + jj];
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    f0 += c0;
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    f0 += beta == 1.f ? pC[0] : pC[0] * beta;
            }
            if (alpha != 1.f)
                f0 *= alpha;
            p0[0] = f0;
            p0++;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
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
                const __m128 _beta = __lsx_vreplfr2vr_s(beta);
                _c0 = __lsx_vfmul_s(_c0, _beta);
                _c1 = __lsx_vfmul_s(_c1, _beta);
            }
        }

        if (pC && broadcast_type_C == 3)
            pC += (size_t)(i + ii) * c_hstep + j;
        if (pC && broadcast_type_C == 4)
            pC += j;

        const __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
        const __m128 _beta = __lsx_vreplfr2vr_s(beta);
        int jj = 0;
#if __loongarch_asx
        const __m256 _alpha256 = (__m256)__lasx_xvreplfr2vr_s(alpha);
        const __m256 _beta256 = (__m256)__lasx_xvreplfr2vr_s(beta);
        const __m256 _c256 = __lasx_concat_128_s(_c0, _c1);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
        }
#endif
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 4;
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
                }
                if (broadcast_type_C == 4)
                {
                    const __m128 _cc0 = __lsx_vreplfr2vr_s(pC[0] * beta);
                    const __m128 _cc1 = __lsx_vreplfr2vr_s(pC[1] * beta);
                    _f0 = __lsx_vfadd_s(_f0, _cc0);
                    _f1 = __lsx_vfadd_s(_f1, _cc0);
                    _f2 = __lsx_vfadd_s(_f2, _cc1);
                    _f3 = __lsx_vfadd_s(_f3, _cc1);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
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
                }
                if (broadcast_type_C == 4)
                {
                    const __m128 _cc = __lsx_vreplfr2vr_s(pC[0] * beta);
                    _f0 = __lsx_vfadd_s(_f0, _cc);
                    _f1 = __lsx_vfadd_s(_f1, _cc);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* pp0 = pp + (ii + 0) * max_jj;
        const float* pp1 = pp + (ii + 1) * max_jj;
        const float* pp2 = pp + (ii + 2) * max_jj;
        const float* pp3 = pp + (ii + 3) * max_jj;
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

        const __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
        const __m128 _beta = __lsx_vreplfr2vr_s(beta);
        int jj = 0;
#if __loongarch_asx
        const __m256 _alpha256 = (__m256)__lasx_xvreplfr2vr_s(alpha);
        const __m256 _beta256 = (__m256)__lasx_xvreplfr2vr_s(beta);
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _f00 = (__m256)__lasx_xvld(pp0 + jj, 0);
            __m256 _f01 = (__m256)__lasx_xvld(pp0 + jj + 8, 0);
            __m256 _f10 = (__m256)__lasx_xvld(pp1 + jj, 0);
            __m256 _f11 = (__m256)__lasx_xvld(pp1 + jj + 8, 0);
            __m256 _f20 = (__m256)__lasx_xvld(pp2 + jj, 0);
            __m256 _f21 = (__m256)__lasx_xvld(pp2 + jj + 8, 0);
            __m256 _f30 = (__m256)__lasx_xvld(pp3 + jj, 0);
            __m256 _f31 = (__m256)__lasx_xvld(pp3 + jj + 8, 0);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    const __m256 _cc = (__m256)__lasx_xvreplfr2vr_s(pC[0] * beta);
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
                    const __m256 _c0 = (__m256)__lasx_xvreplfr2vr_s(pC[i + ii] * beta);
                    const __m256 _c1 = (__m256)__lasx_xvreplfr2vr_s(pC[i + ii + 1] * beta);
                    const __m256 _c2 = (__m256)__lasx_xvreplfr2vr_s(pC[i + ii + 2] * beta);
                    const __m256 _c3 = (__m256)__lasx_xvreplfr2vr_s(pC[i + ii + 3] * beta);
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
                    const __m256 _c00 = (__m256)__lasx_xvld(pC, 0);
                    const __m256 _c01 = (__m256)__lasx_xvld(pC + 8, 0);
                    const __m256 _c10 = (__m256)__lasx_xvld(pC + c_hstep, 0);
                    const __m256 _c11 = (__m256)__lasx_xvld(pC + c_hstep + 8, 0);
                    const __m256 _c20 = (__m256)__lasx_xvld(pC + c_hstep * 2, 0);
                    const __m256 _c21 = (__m256)__lasx_xvld(pC + c_hstep * 2 + 8, 0);
                    const __m256 _c30 = (__m256)__lasx_xvld(pC + c_hstep * 3, 0);
                    const __m256 _c31 = (__m256)__lasx_xvld(pC + c_hstep * 3 + 8, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = (__m256)__lasx_xvld(pp0 + jj, 0);
            __m256 _f1 = (__m256)__lasx_xvld(pp1 + jj, 0);
            __m256 _f2 = (__m256)__lasx_xvld(pp2 + jj, 0);
            __m256 _f3 = (__m256)__lasx_xvld(pp3 + jj, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
        }
#endif
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f00 = (__m128)__lsx_vld(pp0 + jj, 0);
            __m128 _f01 = (__m128)__lsx_vld(pp0 + jj + 4, 0);
            __m128 _f10 = (__m128)__lsx_vld(pp1 + jj, 0);
            __m128 _f11 = (__m128)__lsx_vld(pp1 + jj + 4, 0);
            __m128 _f20 = (__m128)__lsx_vld(pp2 + jj, 0);
            __m128 _f21 = (__m128)__lsx_vld(pp2 + jj + 4, 0);
            __m128 _f30 = (__m128)__lsx_vld(pp3 + jj, 0);
            __m128 _f31 = (__m128)__lsx_vld(pp3 + jj + 4, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp0 + jj, 0);
            __m128 _f1 = (__m128)__lsx_vld(pp1 + jj, 0);
            __m128 _f2 = (__m128)__lsx_vld(pp2 + jj, 0);
            __m128 _f3 = (__m128)__lsx_vld(pp3 + jj, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128i _r0 = __lsx_vldrepl_d(pp0 + jj, 0);
            __m128i _r1 = __lsx_vldrepl_d(pp1 + jj, 0);
            __m128i _r2 = __lsx_vldrepl_d(pp2 + jj, 0);
            __m128i _r3 = __lsx_vldrepl_d(pp3 + jj, 0);
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
                    const __m128 _cc0 = (__m128)__lsx_vilvl_d(_t1, _t0);
                    const __m128 _cc1 = (__m128)__lsx_vilvh_d(_t1, _t0);
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
                }
                if (broadcast_type_C == 4)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(pC[0] * beta));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vreplfr2vr_s(pC[1] * beta));
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
        }
        for (; jj < max_jj; jj++)
        {
            __m128i _fi = __lsx_vldrepl_w(pp0 + jj, 0);
            _fi = __lsx_vinsgr2vr_w(_fi, ((const int*)(pp1 + jj))[0], 1);
            _fi = __lsx_vinsgr2vr_w(_fi, ((const int*)(pp2 + jj))[0], 2);
            _fi = __lsx_vinsgr2vr_w(_fi, ((const int*)(pp3 + jj))[0], 3);
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
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c4 = __lsx_vreplfr2vr_s(pC[0]);
                    if (beta == 1.f)
                        _f = __lsx_vfadd_s(_f, _c4);
                    else
                        _f = __lsx_vfmadd_s(_c4, _beta, _f);
                }
            }
            if (alpha != 1.f)
                _f = __lsx_vfmul_s(_f, _alpha);
            __lsx_vst((__m128i)_f, p0, 0);
            p0 += M;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
        }
    }
#endif
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* pp0 = pp + (ii + 0) * max_jj;
        const float* pp1 = pp + (ii + 1) * max_jj;
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
        const __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
        const __m128 _beta = __lsx_vreplfr2vr_s(beta);
#if __loongarch_asx
        const __m256 _alpha256 = (__m256)__lasx_xvreplfr2vr_s(alpha);
        const __m256 _beta256 = (__m256)__lasx_xvreplfr2vr_s(beta);
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _f00 = (__m256)__lasx_xvld(pp0 + jj, 0);
            __m256 _f01 = (__m256)__lasx_xvld(pp0 + jj + 8, 0);
            __m256 _f10 = (__m256)__lasx_xvld(pp1 + jj, 0);
            __m256 _f11 = (__m256)__lasx_xvld(pp1 + jj + 8, 0);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    const __m256 _cc = (__m256)__lasx_xvreplfr2vr_s(c0);
                    _f00 = __lasx_xvfadd_s(_f00, _cc);
                    _f01 = __lasx_xvfadd_s(_f01, _cc);
                    _f10 = __lasx_xvfadd_s(_f10, _cc);
                    _f11 = __lasx_xvfadd_s(_f11, _cc);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    const __m256 _c0 = (__m256)__lasx_xvreplfr2vr_s(c0);
                    const __m256 _c1 = (__m256)__lasx_xvreplfr2vr_s(c1);
                    _f00 = __lasx_xvfadd_s(_f00, _c0);
                    _f01 = __lasx_xvfadd_s(_f01, _c0);
                    _f10 = __lasx_xvfadd_s(_f10, _c1);
                    _f11 = __lasx_xvfadd_s(_f11, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    const __m256 _c00 = (__m256)__lasx_xvld(pC, 0);
                    const __m256 _c01 = (__m256)__lasx_xvld(pC + 8, 0);
                    const __m256 _c10 = (__m256)__lasx_xvld(pC + c_hstep, 0);
                    const __m256 _c11 = (__m256)__lasx_xvld(pC + c_hstep + 8, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = (__m256)__lasx_xvld(pp0 + jj, 0);
            __m256 _f1 = (__m256)__lasx_xvld(pp1 + jj, 0);
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
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c4 = (__m256)__lasx_xvld(pC, 0);
                    if (beta != 1.f)
                        _c4 = __lasx_xvfmul_s(_c4, _beta256);
                    _f0 = __lasx_xvfadd_s(_f0, _c4);
                    _f1 = __lasx_xvfadd_s(_f1, _c4);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
        }
#endif
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f00 = (__m128)__lsx_vld(pp0 + jj, 0);
            __m128 _f01 = (__m128)__lsx_vld(pp0 + jj + 4, 0);
            __m128 _f10 = (__m128)__lsx_vld(pp1 + jj, 0);
            __m128 _f11 = (__m128)__lsx_vld(pp1 + jj + 4, 0);
            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    const __m128 _cc = __lsx_vreplfr2vr_s(c0);
                    _f00 = __lsx_vfadd_s(_f00, _cc);
                    _f01 = __lsx_vfadd_s(_f01, _cc);
                    _f10 = __lsx_vfadd_s(_f10, _cc);
                    _f11 = __lsx_vfadd_s(_f11, _cc);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    const __m128 _c0 = __lsx_vreplfr2vr_s(c0);
                    const __m128 _c1 = __lsx_vreplfr2vr_s(c1);
                    _f00 = __lsx_vfadd_s(_f00, _c0);
                    _f01 = __lsx_vfadd_s(_f01, _c0);
                    _f10 = __lsx_vfadd_s(_f10, _c1);
                    _f11 = __lsx_vfadd_s(_f11, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    const __m128 _c00 = (__m128)__lsx_vld(pC, 0);
                    const __m128 _c01 = (__m128)__lsx_vld(pC + 4, 0);
                    const __m128 _c10 = (__m128)__lsx_vld(pC + c_hstep, 0);
                    const __m128 _c11 = (__m128)__lsx_vld(pC + c_hstep + 4, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp0 + jj, 0);
            __m128 _f1 = (__m128)__lsx_vld(pp1 + jj, 0);
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
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c4 = (__m128)__lsx_vld(pC, 0);
                    if (beta != 1.f)
                        _c4 = __lsx_vfmul_s(_c4, _beta);
                    _f0 = __lsx_vfadd_s(_f0, _c4);
                    _f1 = __lsx_vfadd_s(_f1, _c4);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128i _r0 = __lsx_vldrepl_d(pp0 + jj, 0);
            __m128i _r1 = __lsx_vldrepl_d(pp1 + jj, 0);
            __m128 _f = (__m128)__lsx_vilvl_w(_r1, _r0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f = __lsx_vfadd_s(_f, _c);
                if (broadcast_type_C == 3)
                {
                    _r0 = __lsx_vldrepl_d(pC, 0);
                    _r1 = __lsx_vldrepl_d(pC + c_hstep, 0);
                    const __m128 _cc = (__m128)__lsx_vilvl_w(_r1, _r0);
                    if (beta == 1.f)
                        _f = __lsx_vfadd_s(_f, _cc);
                    else
                        _f = __lsx_vfmadd_s(_cc, _beta, _f);
                }
                if (broadcast_type_C == 4)
                {
                    __m128i _cc = __lsx_vldrepl_d(pC, 0);
                    _cc = __lsx_vilvl_w(_cc, _cc);
                    if (beta == 1.f)
                        _f = __lsx_vfadd_s(_f, (__m128)_cc);
                    else
                        _f = __lsx_vfmadd_s((__m128)_cc, _beta, _f);
                }
            }
            if (alpha != 1.f)
                _f = __lsx_vfmul_s(_f, _alpha);
            __lsx_vstelm_d((__m128i)_f, p0, 0, 0);
            __lsx_vstelm_d((__m128i)_f, p0 + M, 0, 1);
            p0 += M * 2;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
        }
        for (; jj < max_jj; jj++)
        {
            __m128i _fi = __lsx_vldrepl_w(pp0 + jj, 0);
            _fi = __lsx_vinsgr2vr_w(_fi, ((const int*)(pp1 + jj))[0], 1);
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
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c4 = __lsx_vreplfr2vr_s(pC[0]);
                    if (beta == 1.f)
                        _f = __lsx_vfadd_s(_f, _c4);
                    else
                        _f = __lsx_vfmadd_s(_c4, _beta, _f);
                }
            }
            if (alpha != 1.f)
                _f = __lsx_vfmul_s(_f, _alpha);
            __lsx_vstelm_d((__m128i)_f, p0, 0, 0);
            p0 += M;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
        }
#endif
        for (; jj < max_jj; jj++)
        {
            float f0 = pp0[jj];
            float f1 = pp1[jj];
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
                }
                if (broadcast_type_C == 4)
                {
                    float c = beta == 1.f ? pC[0] : pC[0] * beta;
                    f0 += c;
                    f1 += c;
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const float* pp0 = pp + ii * max_jj;
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
#if __loongarch_asx
        const __m256 _alpha256 = (__m256)__lasx_xvreplfr2vr_s(alpha);
        const __m256 _beta256 = (__m256)__lasx_xvreplfr2vr_s(beta);
        const __m256 _c256 = (__m256)__lasx_xvreplfr2vr_s(c0);
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _f0 = (__m256)__lasx_xvld(pp0 + jj, 0);
            __m256 _f1 = (__m256)__lasx_xvld(pp0 + jj + 8, 0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lasx_xvfadd_s(_f0, _c256);
                    _f1 = __lasx_xvfadd_s(_f1, _c256);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    const __m256 _c0 = (__m256)__lasx_xvld(pC, 0);
                    const __m256 _c1 = (__m256)__lasx_xvld(pC + 8, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = (__m256)__lasx_xvld(pp0 + jj, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
        }
#endif
#if __loongarch_sx
        const __m128 _alpha128 = __lsx_vreplfr2vr_s(alpha);
        const __m128 _beta128 = __lsx_vreplfr2vr_s(beta);
        const __m128 _c128 = __lsx_vreplfr2vr_s(c0);
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp0 + jj, 0);
            __m128 _f1 = (__m128)__lsx_vld(pp0 + jj + 4, 0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, _c128);
                    _f1 = __lsx_vfadd_s(_f1, _c128);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    const __m128 _c0 = (__m128)__lsx_vld(pC, 0);
                    const __m128 _c1 = (__m128)__lsx_vld(pC + 4, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp0 + jj, 0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _f0 = (__m128)__lsx_vldrepl_d(pp0 + jj, 0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f0 = __lsx_vfadd_s(_f0, _c128);
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    const __m128 _cc = (__m128)__lsx_vldrepl_d(pC, 0);
                    if (beta == 1.f)
                        _f0 = __lsx_vfadd_s(_f0, _cc);
                    else
                        _f0 = __lsx_vfmadd_s(_cc, _beta128, _f0);
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
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
        }
#endif
        for (; jj < max_jj; jj++)
        {
            float f0 = pp0[jj];
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    f0 += c0;
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    f0 += beta == 1.f ? pC[0] : pC[0] * beta;
            }
            if (alpha != 1.f)
                f0 *= alpha;
            p0[0] = f0;
            p0 += M;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
        }
    }
}

static void get_optimal_tile_mnk_wq_int8(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    const int tile_size = std::max(1, (int)((float)l2_cache_size / 2 / sizeof(signed char) / std::max(1, K)));

#if __loongarch_sx
    const int tile_m_align = M >= nT * 8 ? 8 : M >= nT * 4 ? 4 : M >= nT * 2 ? 2 : 1;
#if __loongarch_asx
    const int tile_n_align = tile_m_align == 8 ? 8 : 16;
#else
    const int tile_n_align = tile_m_align == 8 ? 4 : 8;
#endif
#else
    const int tile_m_align = M >= nT * 4 ? 4 : M >= nT * 2 ? 2 : 1;
    const int tile_n_align = 2;
#endif
    TILE_M = tile_m_align;
    TILE_N = std::max(tile_n_align, tile_size / tile_n_align * tile_n_align);
    TILE_K = K;

    if (N > 0)
    {
        const int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + tile_n_align - 1) / tile_n_align * tile_n_align);
    }

    // always take constant TILE_N value when provided
    if (constant_TILE_N > 0)
        TILE_N = (constant_TILE_N + tile_n_align - 1) / tile_n_align * tile_n_align;

    (void)constant_TILE_M;
    (void)constant_TILE_K;
}

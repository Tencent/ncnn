// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pack_B_tile_wq_int8(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size)
{
    const int block_count = (K + block_size - 1) / block_size;
    signed char* pp = BT_tile;
    float* pd = BT_descales_tile;

    int jj = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; jj + 7 < max_jj; jj += 8)
    {
        const signed char* p0 = B.row<const signed char>(j + jj);
        const signed char* p1 = B.row<const signed char>(j + jj + 1);
        const signed char* p2 = B.row<const signed char>(j + jj + 2);
        const signed char* p3 = B.row<const signed char>(j + jj + 3);
        const signed char* p4 = B.row<const signed char>(j + jj + 4);
        const signed char* p5 = B.row<const signed char>(j + jj + 5);
        const signed char* p6 = B.row<const signed char>(j + jj + 6);
        const signed char* p7 = B.row<const signed char>(j + jj + 7);
        const float* ps0 = B_scales.row(j + jj);
        const float* ps1 = B_scales.row(j + jj + 1);
        const float* ps2 = B_scales.row(j + jj + 2);
        const float* ps3 = B_scales.row(j + jj + 3);
        const float* ps4 = B_scales.row(j + jj + 4);
        const float* ps5 = B_scales.row(j + jj + 5);
        const float* ps6 = B_scales.row(j + jj + 6);
        const float* ps7 = B_scales.row(j + jj + 7);

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
            for (; kk < max_kk; kk++)
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

            *pd++ = 1.f / *ps0++;
            *pd++ = 1.f / *ps1++;
            *pd++ = 1.f / *ps2++;
            *pd++ = 1.f / *ps3++;
            *pd++ = 1.f / *ps4++;
            *pd++ = 1.f / *ps5++;
            *pd++ = 1.f / *ps6++;
            *pd++ = 1.f / *ps7++;
        }
    }
#endif // __loongarch_asx
    for (; jj + 3 < max_jj; jj += 4)
    {
        const signed char* p0 = B.row<const signed char>(j + jj);
        const signed char* p1 = B.row<const signed char>(j + jj + 1);
        const signed char* p2 = B.row<const signed char>(j + jj + 2);
        const signed char* p3 = B.row<const signed char>(j + jj + 3);
        const float* ps0 = B_scales.row(j + jj);
        const float* ps1 = B_scales.row(j + jj + 1);
        const float* ps2 = B_scales.row(j + jj + 2);
        const float* ps3 = B_scales.row(j + jj + 3);

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
            for (; kk < max_kk; kk++)
            {
                pp[0] = *p0++;
                pp[1] = *p1++;
                pp[2] = *p2++;
                pp[3] = *p3++;
                pp += 4;
            }

            *pd++ = 1.f / *ps0++;
            *pd++ = 1.f / *ps1++;
            *pd++ = 1.f / *ps2++;
            *pd++ = 1.f / *ps3++;
        }
    }
#endif // __loongarch_sx

    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = B.row<const signed char>(j + jj);
        const signed char* p1 = B.row<const signed char>(j + jj + 1);
        const float* ps0 = B_scales.row(j + jj);
        const float* ps1 = B_scales.row(j + jj + 1);

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
            for (; kk < max_kk; kk++)
            {
                pp[0] = *p0++;
                pp[1] = *p1++;
                pp += 2;
            }

            *pd++ = 1.f / *ps0++;
            *pd++ = 1.f / *ps1++;
        }
    }

    for (; jj < max_jj; jj++)
    {
        const signed char* p0 = B.row<const signed char>(j + jj);
        const float* ps0 = B_scales.row(j + jj);
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
            for (; kk < max_kk; kk++)
                *pp++ = *p0++;

            *pd++ = 1.f / *ps0++;
        }
    }
}

static void quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
#if NCNN_BF16
    if (A.elembits() == 16)
    {
        quantize_A_tile_wq_int8_bf16s(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif // NCNN_BF16

#if NCNN_RUNTIME_CPU && NCNN_MMI && !__loongarch_sx && !__mips_loongson_mmi
    if (A.elempack == 1 && ncnn::cpu_support_loongson_mmi())
    {
        quantize_A_tile_wq_int8_loongson_mmi(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

    const int elempack = A.elempack;
    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int block_count = (max_kk + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    if (input_scales.empty())
    {
        int ii = 0;
#if __loongarch_sx
        for (; ii + 7 < max_ii; ii += 8)
        {
            const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k * elempack;
            const float* p1 = p0 + A_hstep * elempack;
            const float* p2 = 0;
            const float* p3 = 0;
            const float* p4 = 0;
            const float* p5 = 0;
            const float* p6 = 0;
            const float* p7 = 0;
            if (elempack == 1)
            {
                p2 = p1 + A_hstep;
                p3 = p2 + A_hstep;
                p4 = p3 + A_hstep;
                p5 = p4 + A_hstep;
                p6 = p5 + A_hstep;
                p7 = p6 + A_hstep;
            }

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                float absmax2 = 0.f;
                float absmax3 = 0.f;
                float absmax4 = 0.f;
                float absmax5 = 0.f;
                float absmax6 = 0.f;
                float absmax7 = 0.f;

                const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                __m128 _absmax0 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax1 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax2 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax3 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax4 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax5 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax6 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax7 = (__m128)__lsx_vreplgr2vr_w(0);
#if __loongarch_asx
                __m256 _absmax8 = (__m256)__lasx_xvreplgr2vr_w(0);
#endif // __loongarch_asx

                const float* p0a = p0;
                const float* p1a = p1;
                const float* p2a = p2;
                const float* p3a = p3;
                const float* p4a = p4;
                const float* p5a = p5;
                const float* p6a = p6;
                const float* p7a = p7;
                int kk = 0;

#if __loongarch_asx
                if (elempack == 8)
                {
                    const __m256i _abs_mask8 = (__m256i)__lasx_xvreplgr2vr_w(0x7fffffff);
                    for (; kk < max_kk0; kk++)
                    {
                        __m256 _p = (__m256)__lasx_xvld(p0a, 0);
                        _absmax8 = __lasx_xvfmax_s(_absmax8, (__m256)__lasx_xvand_v((__m256i)_p, _abs_mask8));
                        p0a += 8;
                    }

                    float absmax[8];
                    __lasx_xvst((__m256i)_absmax8, absmax, 0);
                    absmax0 = absmax[0];
                    absmax1 = absmax[1];
                    absmax2 = absmax[2];
                    absmax3 = absmax[3];
                    absmax4 = absmax[4];
                    absmax5 = absmax[5];
                    absmax6 = absmax[6];
                    absmax7 = absmax[7];
                }
#endif // __loongarch_asx

                if (elempack == 4)
                {
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p0 = (__m128)__lsx_vld(p0a, 0);
                        __m128 _p1 = (__m128)__lsx_vld(p1a, 0);
                        _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_p0, _abs_mask));
                        _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_p1, _abs_mask));
                        p0a += 4;
                        p1a += 4;
                    }

                    float absmax[8];
                    __lsx_vst((__m128i)_absmax0, absmax, 0);
                    __lsx_vst((__m128i)_absmax1, absmax + 4, 0);
                    absmax0 = absmax[0];
                    absmax1 = absmax[1];
                    absmax2 = absmax[2];
                    absmax3 = absmax[3];
                    absmax4 = absmax[4];
                    absmax5 = absmax[5];
                    absmax6 = absmax[6];
                    absmax7 = absmax[7];
                }

                if (elempack == 1)
                {
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = (__m128)__lsx_vld(p0a, 0);
                        __m128 _p1 = (__m128)__lsx_vld(p1a, 0);
                        __m128 _p2 = (__m128)__lsx_vld(p2a, 0);
                        __m128 _p3 = (__m128)__lsx_vld(p3a, 0);
                        __m128 _p4 = (__m128)__lsx_vld(p4a, 0);
                        __m128 _p5 = (__m128)__lsx_vld(p5a, 0);
                        __m128 _p6 = (__m128)__lsx_vld(p6a, 0);
                        __m128 _p7 = (__m128)__lsx_vld(p7a, 0);
                        _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_p0, _abs_mask));
                        _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_p1, _abs_mask));
                        _absmax2 = __lsx_vfmax_s(_absmax2, (__m128)__lsx_vand_v((__m128i)_p2, _abs_mask));
                        _absmax3 = __lsx_vfmax_s(_absmax3, (__m128)__lsx_vand_v((__m128i)_p3, _abs_mask));
                        _absmax4 = __lsx_vfmax_s(_absmax4, (__m128)__lsx_vand_v((__m128i)_p4, _abs_mask));
                        _absmax5 = __lsx_vfmax_s(_absmax5, (__m128)__lsx_vand_v((__m128i)_p5, _abs_mask));
                        _absmax6 = __lsx_vfmax_s(_absmax6, (__m128)__lsx_vand_v((__m128i)_p6, _abs_mask));
                        _absmax7 = __lsx_vfmax_s(_absmax7, (__m128)__lsx_vand_v((__m128i)_p7, _abs_mask));
                        p0a += 4;
                        p1a += 4;
                        p2a += 4;
                        p3a += 4;
                        p4a += 4;
                        p5a += 4;
                        p6a += 4;
                        p7a += 4;
                    }

                    absmax0 = __lsx_reduce_fmax_s(_absmax0);
                    absmax1 = __lsx_reduce_fmax_s(_absmax1);
                    absmax2 = __lsx_reduce_fmax_s(_absmax2);
                    absmax3 = __lsx_reduce_fmax_s(_absmax3);
                    absmax4 = __lsx_reduce_fmax_s(_absmax4);
                    absmax5 = __lsx_reduce_fmax_s(_absmax5);
                    absmax6 = __lsx_reduce_fmax_s(_absmax6);
                    absmax7 = __lsx_reduce_fmax_s(_absmax7);

                    for (; kk < max_kk0; kk++)
                    {
                        absmax0 = std::max(absmax0, fabsf(*p0a++));
                        absmax1 = std::max(absmax1, fabsf(*p1a++));
                        absmax2 = std::max(absmax2, fabsf(*p2a++));
                        absmax3 = std::max(absmax3, fabsf(*p3a++));
                        absmax4 = std::max(absmax4, fabsf(*p4a++));
                        absmax5 = std::max(absmax5, fabsf(*p5a++));
                        absmax6 = std::max(absmax6, fabsf(*p6a++));
                        absmax7 = std::max(absmax7, fabsf(*p7a++));
                    }
                }

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
                const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
                const float scale4 = absmax4 == 0.f ? 1.f : 127.f / absmax4;
                const float scale5 = absmax5 == 0.f ? 1.f : 127.f / absmax5;
                const float scale6 = absmax6 == 0.f ? 1.f : 127.f / absmax6;
                const float scale7 = absmax7 == 0.f ? 1.f : 127.f / absmax7;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd[2] = absmax2 / 127.f;
                pd[3] = absmax3 / 127.f;
                pd[4] = absmax4 / 127.f;
                pd[5] = absmax5 / 127.f;
                pd[6] = absmax6 / 127.f;
                pd[7] = absmax7 / 127.f;
                pd += 8;

#if __loongarch_asx
                if (elempack == 8)
                {
                    const __m256 _one = __lasx_xvreplfr2vr_s(1.f);
                    const __m256 _v127 = __lasx_xvreplfr2vr_s(127.f);
                    __m256i _zeromask = (__m256i)__lasx_xvfcmp_ceq_s(_absmax8, (__m256)__lasx_xvreplgr2vr_w(0));
                    __m256 _absmax_safe = (__m256)__lasx_xvbitsel_v((__m256i)_absmax8, (__m256i)_one, _zeromask);
                    __m256 _scale = __lasx_xvfdiv_s(_v127, _absmax_safe);

                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m256 _p0 = __lasx_xvfmul_s((__m256)__lasx_xvld(p0, 0), _scale);
                        __m256 _p1 = __lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 8, 0), _scale);
                        __m256 _p2 = __lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 16, 0), _scale);
                        __m256 _p3 = __lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 24, 0), _scale);
                        transpose8x4_ps(_p0, _p1, _p2, _p3);
                        __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(_p0)), pp, 0, 0);
                        __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(_p1)), pp + 8, 0, 0);
                        __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(_p2)), pp + 16, 0, 0);
                        __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(_p3)), pp + 24, 0, 0);
                        pp += 32;
                        p0 += 32;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m256 _p = __lasx_xvfmul_s((__m256)__lasx_xvld(p0, 0), _scale);
                        __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(_p)), pp, 0, 0);
                        pp += 8;
                        p0 += 8;
                    }
                }
#endif // __loongarch_asx

                if (elempack == 4)
                {
                    __m128i _scale01 = __lsx_vilvl_w((__m128i)__lsx_vreplfr2vr_s(scale1), (__m128i)__lsx_vreplfr2vr_s(scale0));
                    __m128i _scale23 = __lsx_vilvl_w((__m128i)__lsx_vreplfr2vr_s(scale3), (__m128i)__lsx_vreplfr2vr_s(scale2));
                    __m128 _scale0 = (__m128)__lsx_vilvl_d(_scale23, _scale01);
                    __m128i _scale45 = __lsx_vilvl_w((__m128i)__lsx_vreplfr2vr_s(scale5), (__m128i)__lsx_vreplfr2vr_s(scale4));
                    __m128i _scale67 = __lsx_vilvl_w((__m128i)__lsx_vreplfr2vr_s(scale7), (__m128i)__lsx_vreplfr2vr_s(scale6));
                    __m128 _scale1 = (__m128)__lsx_vilvl_d(_scale67, _scale45);

                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = __lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _scale0);
                        __m128 _p1 = __lsx_vfmul_s((__m128)__lsx_vld(p0 + 4, 0), _scale0);
                        __m128 _p2 = __lsx_vfmul_s((__m128)__lsx_vld(p0 + 8, 0), _scale0);
                        __m128 _p3 = __lsx_vfmul_s((__m128)__lsx_vld(p0 + 12, 0), _scale0);
                        transpose4x4_ps(_p0, _p1, _p2, _p3);

                        __m128 _p4 = __lsx_vfmul_s((__m128)__lsx_vld(p1, 0), _scale1);
                        __m128 _p5 = __lsx_vfmul_s((__m128)__lsx_vld(p1 + 4, 0), _scale1);
                        __m128 _p6 = __lsx_vfmul_s((__m128)__lsx_vld(p1 + 8, 0), _scale1);
                        __m128 _p7 = __lsx_vfmul_s((__m128)__lsx_vld(p1 + 12, 0), _scale1);
                        transpose4x4_ps(_p4, _p5, _p6, _p7);

                        ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                        ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                        ((int64_t*)pp)[2] = float2int8(_p4, _p5);
                        ((int64_t*)pp)[3] = float2int8(_p6, _p7);
                        pp += 32;
                        p0 += 16;
                        p1 += 16;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p0 = __lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _scale0);
                        __m128 _p1 = __lsx_vfmul_s((__m128)__lsx_vld(p1, 0), _scale1);
                        ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                        pp += 8;
                        p0 += 4;
                        p1 += 4;
                    }
                }

                if (elempack == 1)
                {
                    __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
                    __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
                    __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
                    __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
                    __m128 _scale4 = __lsx_vreplfr2vr_s(scale4);
                    __m128 _scale5 = __lsx_vreplfr2vr_s(scale5);
                    __m128 _scale6 = __lsx_vreplfr2vr_s(scale6);
                    __m128 _scale7 = __lsx_vreplfr2vr_s(scale7);
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = (__m128)__lsx_vld(p0, 0);
                        __m128 _p1 = (__m128)__lsx_vld(p1, 0);
                        __m128 _p2 = (__m128)__lsx_vld(p2, 0);
                        __m128 _p3 = (__m128)__lsx_vld(p3, 0);
                        __m128 _p4 = (__m128)__lsx_vld(p4, 0);
                        __m128 _p5 = (__m128)__lsx_vld(p5, 0);
                        __m128 _p6 = (__m128)__lsx_vld(p6, 0);
                        __m128 _p7 = (__m128)__lsx_vld(p7, 0);
                        _p0 = __lsx_vfmul_s(_p0, _scale0);
                        _p1 = __lsx_vfmul_s(_p1, _scale1);
                        _p2 = __lsx_vfmul_s(_p2, _scale2);
                        _p3 = __lsx_vfmul_s(_p3, _scale3);
                        _p4 = __lsx_vfmul_s(_p4, _scale4);
                        _p5 = __lsx_vfmul_s(_p5, _scale5);
                        _p6 = __lsx_vfmul_s(_p6, _scale6);
                        _p7 = __lsx_vfmul_s(_p7, _scale7);

                        ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                        ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                        ((int64_t*)pp)[2] = float2int8(_p4, _p5);
                        ((int64_t*)pp)[3] = float2int8(_p6, _p7);
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
                    for (; kk < max_kk0; kk++)
                    {
                        pp[0] = float2int8(*p0 * scale0);
                        pp[1] = float2int8(*p1 * scale1);
                        pp[2] = float2int8(*p2 * scale2);
                        pp[3] = float2int8(*p3 * scale3);
                        pp[4] = float2int8(*p4 * scale4);
                        pp[5] = float2int8(*p5 * scale5);
                        pp[6] = float2int8(*p6 * scale6);
                        pp[7] = float2int8(*p7 * scale7);
                        pp += 8;
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
            }
        }
        for (; ii + 3 < max_ii; ii += 4)
        {
            const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k * elempack;
            const float* p1 = 0;
            const float* p2 = 0;
            const float* p3 = 0;
            if (elempack == 1)
            {
                p1 = p0 + A_hstep;
                p2 = p1 + A_hstep;
                p3 = p2 + A_hstep;
            }

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                float absmax2 = 0.f;
                float absmax3 = 0.f;

                const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                __m128 _absmax0 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax1 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax2 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax3 = (__m128)__lsx_vreplgr2vr_w(0);

                const float* p0a = p0;
                const float* p1a = p1;
                const float* p2a = p2;
                const float* p3a = p3;
                int kk = 0;

                if (elempack == 4)
                {
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p = (__m128)__lsx_vld(p0a, 0);
                        _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_p, _abs_mask));
                        p0a += 4;
                    }

                    float absmax[4];
                    __lsx_vst((__m128i)_absmax0, absmax, 0);
                    absmax0 = absmax[0];
                    absmax1 = absmax[1];
                    absmax2 = absmax[2];
                    absmax3 = absmax[3];
                }

                if (elempack == 1)
                {
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = (__m128)__lsx_vld(p0a, 0);
                        __m128 _p1 = (__m128)__lsx_vld(p1a, 0);
                        __m128 _p2 = (__m128)__lsx_vld(p2a, 0);
                        __m128 _p3 = (__m128)__lsx_vld(p3a, 0);
                        _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_p0, _abs_mask));
                        _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_p1, _abs_mask));
                        _absmax2 = __lsx_vfmax_s(_absmax2, (__m128)__lsx_vand_v((__m128i)_p2, _abs_mask));
                        _absmax3 = __lsx_vfmax_s(_absmax3, (__m128)__lsx_vand_v((__m128i)_p3, _abs_mask));
                        p0a += 4;
                        p1a += 4;
                        p2a += 4;
                        p3a += 4;
                    }
                    absmax0 = __lsx_reduce_fmax_s(_absmax0);
                    absmax1 = __lsx_reduce_fmax_s(_absmax1);
                    absmax2 = __lsx_reduce_fmax_s(_absmax2);
                    absmax3 = __lsx_reduce_fmax_s(_absmax3);

                    for (; kk < max_kk0; kk++)
                    {
                        float v0 = *p0a++;
                        float v1 = *p1a++;
                        float v2 = *p2a++;
                        float v3 = *p3a++;
                        absmax0 = std::max(absmax0, fabsf(v0));
                        absmax1 = std::max(absmax1, fabsf(v1));
                        absmax2 = std::max(absmax2, fabsf(v2));
                        absmax3 = std::max(absmax3, fabsf(v3));
                    }
                }

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
                const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd[2] = absmax2 / 127.f;
                pd[3] = absmax3 / 127.f;
                pd += 4;

                if (elempack == 4)
                {
                    __m128i _scale01 = __lsx_vilvl_w((__m128i)__lsx_vreplfr2vr_s(scale1), (__m128i)__lsx_vreplfr2vr_s(scale0));
                    __m128i _scale23 = __lsx_vilvl_w((__m128i)__lsx_vreplfr2vr_s(scale3), (__m128i)__lsx_vreplfr2vr_s(scale2));
                    __m128 _scale = (__m128)__lsx_vilvl_d(_scale23, _scale01);

                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = __lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _scale);
                        __m128 _p1 = __lsx_vfmul_s((__m128)__lsx_vld(p0 + 4, 0), _scale);
                        __m128 _p2 = __lsx_vfmul_s((__m128)__lsx_vld(p0 + 8, 0), _scale);
                        __m128 _p3 = __lsx_vfmul_s((__m128)__lsx_vld(p0 + 12, 0), _scale);
                        transpose4x4_ps(_p0, _p1, _p2, _p3);

                        ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                        ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                        pp += 16;
                        p0 += 16;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p = __lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _scale);
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p), 0);
                        pp += 4;
                        p0 += 4;
                    }
                }

                if (elempack == 1)
                {
                    __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
                    __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
                    __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
                    __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = (__m128)__lsx_vld(p0, 0);
                        __m128 _p1 = (__m128)__lsx_vld(p1, 0);
                        __m128 _p2 = (__m128)__lsx_vld(p2, 0);
                        __m128 _p3 = (__m128)__lsx_vld(p3, 0);
                        _p0 = __lsx_vfmul_s(_p0, _scale0);
                        _p1 = __lsx_vfmul_s(_p1, _scale1);
                        _p2 = __lsx_vfmul_s(_p2, _scale2);
                        _p3 = __lsx_vfmul_s(_p3, _scale3);

                        ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                        ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                        pp += 16;
                        p0 += 4;
                        p1 += 4;
                        p2 += 4;
                        p3 += 4;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float v0 = *p0++;
                        float v1 = *p1++;
                        float v2 = *p2++;
                        float v3 = *p3++;
                        pp[0] = float2int8(v0 * scale0);
                        pp[1] = float2int8(v1 * scale1);
                        pp[2] = float2int8(v2 * scale2);
                        pp[3] = float2int8(v3 * scale3);
                        pp += 4;
                    }
                }
            }
        }
#endif // __loongarch_sx
        for (; ii + 1 < max_ii; ii += 2)
        {
            const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k;
            const float* p1 = p0 + A_hstep;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const float* p0a = p0;
                const float* p1a = p1;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v0 = *p0a++;
                    float v1 = *p1a++;
                    absmax0 = std::max(absmax0, fabsf(v0));
                    absmax1 = std::max(absmax1, fabsf(v1));
                }

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd += 2;

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float v00 = p0[0];
                    float v01 = p0[1];
                    float v02 = p0[2];
                    float v03 = p0[3];
                    float v10 = p1[0];
                    float v11 = p1[1];
                    float v12 = p1[2];
                    float v13 = p1[3];
                    pp[0] = float2int8(v00 * scale0);
                    pp[1] = float2int8(v01 * scale0);
                    pp[2] = float2int8(v02 * scale0);
                    pp[3] = float2int8(v03 * scale0);
                    pp[4] = float2int8(v10 * scale1);
                    pp[5] = float2int8(v11 * scale1);
                    pp[6] = float2int8(v12 * scale1);
                    pp[7] = float2int8(v13 * scale1);
                    pp += 8;
                    p0 += 4;
                    p1 += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    float v0 = *p0++;
                    float v1 = *p1++;
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale1);
                    pp += 2;
                }
            }
        }
        for (; ii < max_ii; ii++)
        {
            const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                const float* p0a = p0;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v0 = *p0a++;
                    absmax0 = std::max(absmax0, fabsf(v0));
                }

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                *pd++ = absmax0 / 127.f;

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float v0 = p0[0];
                    float v1 = p0[1];
                    float v2 = p0[2];
                    float v3 = p0[3];
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale0);
                    pp[2] = float2int8(v2 * scale0);
                    pp[3] = float2int8(v3 * scale0);
                    pp += 4;
                    p0 += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    float v0 = *p0++;
                    *pp++ = float2int8(v0 * scale0);
                }
            }
        }
        return;
    }

    const float* input_scale_ptr = (const float*)input_scales + k;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k * elempack;
        const float* p1 = p0 + A_hstep * elempack;
        const float* p2 = 0;
        const float* p3 = 0;
        const float* p4 = 0;
        const float* p5 = 0;
        const float* p6 = 0;
        const float* p7 = 0;
        if (elempack == 1)
        {
            p2 = p1 + A_hstep;
            p3 = p2 + A_hstep;
            p4 = p3 + A_hstep;
            p5 = p4 + A_hstep;
            p6 = p5 + A_hstep;
            p7 = p6 + A_hstep;
        }

        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            float absmax2 = 0.f;
            float absmax3 = 0.f;
            float absmax4 = 0.f;
            float absmax5 = 0.f;
            float absmax6 = 0.f;
            float absmax7 = 0.f;

            const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
            __m128 _absmax0 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _absmax1 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _absmax2 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _absmax3 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _absmax4 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _absmax5 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _absmax6 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _absmax7 = (__m128)__lsx_vreplgr2vr_w(0);
#if __loongarch_asx
            __m256 _absmax8 = (__m256)__lasx_xvreplgr2vr_w(0);
#endif // __loongarch_asx

            const float* p0a = p0;
            const float* p1a = p1;
            const float* p2a = p2;
            const float* p3a = p3;
            const float* p4a = p4;
            const float* p5a = p5;
            const float* p6a = p6;
            const float* p7a = p7;
            const float* psa = ps;
            int kk = 0;

#if __loongarch_asx
            if (elempack == 8)
            {
                const __m256i _abs_mask8 = (__m256i)__lasx_xvreplgr2vr_w(0x7fffffff);
                for (; kk < max_kk0; kk++)
                {
                    __m256 _s = __lasx_xvreplfr2vr_s(*psa++);
                    __m256 _p = __lasx_xvfmul_s((__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a, 0), _abs_mask8), _s);
                    _absmax8 = __lasx_xvfmax_s(_absmax8, _p);
                    p0a += 8;
                }

                float absmax[8];
                __lasx_xvst((__m256i)_absmax8, absmax, 0);
                absmax0 = absmax[0];
                absmax1 = absmax[1];
                absmax2 = absmax[2];
                absmax3 = absmax[3];
                absmax4 = absmax[4];
                absmax5 = absmax[5];
                absmax6 = absmax[6];
                absmax7 = absmax[7];
            }
#endif // __loongarch_asx

            if (elempack == 4)
            {
                for (; kk < max_kk0; kk++)
                {
                    __m128 _s = __lsx_vreplfr2vr_s(*psa++);
                    __m128 _p0 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a, 0), _abs_mask), _s);
                    __m128 _p1 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)__lsx_vld(p1a, 0), _abs_mask), _s);
                    _absmax0 = __lsx_vfmax_s(_absmax0, _p0);
                    _absmax1 = __lsx_vfmax_s(_absmax1, _p1);
                    p0a += 4;
                    p1a += 4;
                }

                float absmax[8];
                __lsx_vst((__m128i)_absmax0, absmax, 0);
                __lsx_vst((__m128i)_absmax1, absmax + 4, 0);
                absmax0 = absmax[0];
                absmax1 = absmax[1];
                absmax2 = absmax[2];
                absmax3 = absmax[3];
                absmax4 = absmax[4];
                absmax5 = absmax[5];
                absmax6 = absmax[6];
                absmax7 = absmax[7];
            }

            if (elempack == 1)
            {
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = (__m128)__lsx_vld(p0a, 0);
                    __m128 _p1 = (__m128)__lsx_vld(p1a, 0);
                    __m128 _p2 = (__m128)__lsx_vld(p2a, 0);
                    __m128 _p3 = (__m128)__lsx_vld(p3a, 0);
                    __m128 _p4 = (__m128)__lsx_vld(p4a, 0);
                    __m128 _p5 = (__m128)__lsx_vld(p5a, 0);
                    __m128 _p6 = (__m128)__lsx_vld(p6a, 0);
                    __m128 _p7 = (__m128)__lsx_vld(p7a, 0);
                    __m128 _s = (__m128)__lsx_vld(psa, 0);
                    _p0 = (__m128)__lsx_vand_v((__m128i)_p0, _abs_mask);
                    _p0 = __lsx_vfmul_s(_p0, _s);
                    _p1 = (__m128)__lsx_vand_v((__m128i)_p1, _abs_mask);
                    _p1 = __lsx_vfmul_s(_p1, _s);
                    _p2 = (__m128)__lsx_vand_v((__m128i)_p2, _abs_mask);
                    _p2 = __lsx_vfmul_s(_p2, _s);
                    _p3 = (__m128)__lsx_vand_v((__m128i)_p3, _abs_mask);
                    _p3 = __lsx_vfmul_s(_p3, _s);
                    _p4 = (__m128)__lsx_vand_v((__m128i)_p4, _abs_mask);
                    _p4 = __lsx_vfmul_s(_p4, _s);
                    _p5 = (__m128)__lsx_vand_v((__m128i)_p5, _abs_mask);
                    _p5 = __lsx_vfmul_s(_p5, _s);
                    _p6 = (__m128)__lsx_vand_v((__m128i)_p6, _abs_mask);
                    _p6 = __lsx_vfmul_s(_p6, _s);
                    _p7 = (__m128)__lsx_vand_v((__m128i)_p7, _abs_mask);
                    _p7 = __lsx_vfmul_s(_p7, _s);
                    _absmax0 = __lsx_vfmax_s(_absmax0, _p0);
                    _absmax1 = __lsx_vfmax_s(_absmax1, _p1);
                    _absmax2 = __lsx_vfmax_s(_absmax2, _p2);
                    _absmax3 = __lsx_vfmax_s(_absmax3, _p3);
                    _absmax4 = __lsx_vfmax_s(_absmax4, _p4);
                    _absmax5 = __lsx_vfmax_s(_absmax5, _p5);
                    _absmax6 = __lsx_vfmax_s(_absmax6, _p6);
                    _absmax7 = __lsx_vfmax_s(_absmax7, _p7);
                    p0a += 4;
                    p1a += 4;
                    p2a += 4;
                    p3a += 4;
                    p4a += 4;
                    p5a += 4;
                    p6a += 4;
                    p7a += 4;
                    psa += 4;
                }

                absmax0 = __lsx_reduce_fmax_s(_absmax0);
                absmax1 = __lsx_reduce_fmax_s(_absmax1);
                absmax2 = __lsx_reduce_fmax_s(_absmax2);
                absmax3 = __lsx_reduce_fmax_s(_absmax3);
                absmax4 = __lsx_reduce_fmax_s(_absmax4);
                absmax5 = __lsx_reduce_fmax_s(_absmax5);
                absmax6 = __lsx_reduce_fmax_s(_absmax6);
                absmax7 = __lsx_reduce_fmax_s(_absmax7);

                for (; kk < max_kk0; kk++)
                {
                    const float s = *psa++;
                    absmax0 = std::max(absmax0, fabsf(*p0a++) * s);
                    absmax1 = std::max(absmax1, fabsf(*p1a++) * s);
                    absmax2 = std::max(absmax2, fabsf(*p2a++) * s);
                    absmax3 = std::max(absmax3, fabsf(*p3a++) * s);
                    absmax4 = std::max(absmax4, fabsf(*p4a++) * s);
                    absmax5 = std::max(absmax5, fabsf(*p5a++) * s);
                    absmax6 = std::max(absmax6, fabsf(*p6a++) * s);
                    absmax7 = std::max(absmax7, fabsf(*p7a++) * s);
                }
            }

            const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
            const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
            const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
            const float scale4 = absmax4 == 0.f ? 1.f : 127.f / absmax4;
            const float scale5 = absmax5 == 0.f ? 1.f : 127.f / absmax5;
            const float scale6 = absmax6 == 0.f ? 1.f : 127.f / absmax6;
            const float scale7 = absmax7 == 0.f ? 1.f : 127.f / absmax7;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd[2] = absmax2 / 127.f;
            pd[3] = absmax3 / 127.f;
            pd[4] = absmax4 / 127.f;
            pd[5] = absmax5 / 127.f;
            pd[6] = absmax6 / 127.f;
            pd[7] = absmax7 / 127.f;
            pd += 8;

#if __loongarch_asx
            if (elempack == 8)
            {
                const __m256 _one = __lasx_xvreplfr2vr_s(1.f);
                const __m256 _v127 = __lasx_xvreplfr2vr_s(127.f);
                __m256i _zeromask = (__m256i)__lasx_xvfcmp_ceq_s(_absmax8, (__m256)__lasx_xvreplgr2vr_w(0));
                __m256 _absmax_safe = (__m256)__lasx_xvbitsel_v((__m256i)_absmax8, (__m256i)_one, _zeromask);
                __m256 _scale = __lasx_xvfdiv_s(_v127, _absmax_safe);

                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256 _s0 = __lasx_xvreplfr2vr_s(ps[0]);
                    __m256 _s1 = __lasx_xvreplfr2vr_s(ps[1]);
                    __m256 _s2 = __lasx_xvreplfr2vr_s(ps[2]);
                    __m256 _s3 = __lasx_xvreplfr2vr_s(ps[3]);
                    __m256 _p0 = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0, 0), _s0), _scale);
                    __m256 _p1 = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 8, 0), _s1), _scale);
                    __m256 _p2 = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 16, 0), _s2), _scale);
                    __m256 _p3 = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 24, 0), _s3), _scale);
                    transpose8x4_ps(_p0, _p1, _p2, _p3);
                    __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(_p0)), pp, 0, 0);
                    __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(_p1)), pp + 8, 0, 0);
                    __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(_p2)), pp + 16, 0, 0);
                    __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(_p3)), pp + 24, 0, 0);
                    pp += 32;
                    p0 += 32;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m256 _s = __lasx_xvreplfr2vr_s(*ps++);
                    __m256 _p = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0, 0), _s), _scale);
                    __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(_p)), pp, 0, 0);
                    pp += 8;
                    p0 += 8;
                }
            }
#endif // __loongarch_asx

            if (elempack == 4)
            {
                __m128i _scale01 = __lsx_vilvl_w((__m128i)__lsx_vreplfr2vr_s(scale1), (__m128i)__lsx_vreplfr2vr_s(scale0));
                __m128i _scale23 = __lsx_vilvl_w((__m128i)__lsx_vreplfr2vr_s(scale3), (__m128i)__lsx_vreplfr2vr_s(scale2));
                __m128 _scale0 = (__m128)__lsx_vilvl_d(_scale23, _scale01);
                __m128i _scale45 = __lsx_vilvl_w((__m128i)__lsx_vreplfr2vr_s(scale5), (__m128i)__lsx_vreplfr2vr_s(scale4));
                __m128i _scale67 = __lsx_vilvl_w((__m128i)__lsx_vreplfr2vr_s(scale7), (__m128i)__lsx_vreplfr2vr_s(scale6));
                __m128 _scale1 = (__m128)__lsx_vilvl_d(_scale67, _scale45);

                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _s0 = __lsx_vreplfr2vr_s(ps[0]);
                    __m128 _s1 = __lsx_vreplfr2vr_s(ps[1]);
                    __m128 _s2 = __lsx_vreplfr2vr_s(ps[2]);
                    __m128 _s3 = __lsx_vreplfr2vr_s(ps[3]);
                    __m128 _p0 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _s0), _scale0);
                    __m128 _p1 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0 + 4, 0), _s1), _scale0);
                    __m128 _p2 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0 + 8, 0), _s2), _scale0);
                    __m128 _p3 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0 + 12, 0), _s3), _scale0);
                    transpose4x4_ps(_p0, _p1, _p2, _p3);

                    __m128 _p4 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p1, 0), _s0), _scale1);
                    __m128 _p5 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p1 + 4, 0), _s1), _scale1);
                    __m128 _p6 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p1 + 8, 0), _s2), _scale1);
                    __m128 _p7 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p1 + 12, 0), _s3), _scale1);
                    transpose4x4_ps(_p4, _p5, _p6, _p7);

                    ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                    ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                    ((int64_t*)pp)[2] = float2int8(_p4, _p5);
                    ((int64_t*)pp)[3] = float2int8(_p6, _p7);
                    pp += 32;
                    p0 += 16;
                    p1 += 16;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128 _s = __lsx_vreplfr2vr_s(*ps++);
                    __m128 _p0 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _s), _scale0);
                    __m128 _p1 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p1, 0), _s), _scale1);
                    ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                    pp += 8;
                    p0 += 4;
                    p1 += 4;
                }
            }
            if (elempack == 1)
            {
                __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
                __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
                __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
                __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
                __m128 _scale4 = __lsx_vreplfr2vr_s(scale4);
                __m128 _scale5 = __lsx_vreplfr2vr_s(scale5);
                __m128 _scale6 = __lsx_vreplfr2vr_s(scale6);
                __m128 _scale7 = __lsx_vreplfr2vr_s(scale7);
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _p1 = (__m128)__lsx_vld(p1, 0);
                    __m128 _p2 = (__m128)__lsx_vld(p2, 0);
                    __m128 _p3 = (__m128)__lsx_vld(p3, 0);
                    __m128 _p4 = (__m128)__lsx_vld(p4, 0);
                    __m128 _p5 = (__m128)__lsx_vld(p5, 0);
                    __m128 _p6 = (__m128)__lsx_vld(p6, 0);
                    __m128 _p7 = (__m128)__lsx_vld(p7, 0);
                    __m128 _s = (__m128)__lsx_vld(ps, 0);
                    _p0 = __lsx_vfmul_s(_p0, _s);
                    _p1 = __lsx_vfmul_s(_p1, _s);
                    _p2 = __lsx_vfmul_s(_p2, _s);
                    _p3 = __lsx_vfmul_s(_p3, _s);
                    _p4 = __lsx_vfmul_s(_p4, _s);
                    _p5 = __lsx_vfmul_s(_p5, _s);
                    _p6 = __lsx_vfmul_s(_p6, _s);
                    _p7 = __lsx_vfmul_s(_p7, _s);
                    _p0 = __lsx_vfmul_s(_p0, _scale0);
                    _p1 = __lsx_vfmul_s(_p1, _scale1);
                    _p2 = __lsx_vfmul_s(_p2, _scale2);
                    _p3 = __lsx_vfmul_s(_p3, _scale3);
                    _p4 = __lsx_vfmul_s(_p4, _scale4);
                    _p5 = __lsx_vfmul_s(_p5, _scale5);
                    _p6 = __lsx_vfmul_s(_p6, _scale6);
                    _p7 = __lsx_vfmul_s(_p7, _scale7);

                    ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                    ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                    ((int64_t*)pp)[2] = float2int8(_p4, _p5);
                    ((int64_t*)pp)[3] = float2int8(_p6, _p7);
                    pp += 32;
                    p0 += 4;
                    p1 += 4;
                    p2 += 4;
                    p3 += 4;
                    p4 += 4;
                    p5 += 4;
                    p6 += 4;
                    p7 += 4;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    const float s = *ps++;
                    pp[0] = float2int8(*p0 * s * scale0);
                    pp[1] = float2int8(*p1 * s * scale1);
                    pp[2] = float2int8(*p2 * s * scale2);
                    pp[3] = float2int8(*p3 * s * scale3);
                    pp[4] = float2int8(*p4 * s * scale4);
                    pp[5] = float2int8(*p5 * s * scale5);
                    pp[6] = float2int8(*p6 * s * scale6);
                    pp[7] = float2int8(*p7 * s * scale7);
                    pp += 8;
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
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k * elempack;
        const float* p1 = 0;
        const float* p2 = 0;
        const float* p3 = 0;
        if (elempack == 1)
        {
            p1 = p0 + A_hstep;
            p2 = p1 + A_hstep;
            p3 = p2 + A_hstep;
        }

        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            float absmax2 = 0.f;
            float absmax3 = 0.f;

            const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
            __m128 _absmax0 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _absmax1 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _absmax2 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _absmax3 = (__m128)__lsx_vreplgr2vr_w(0);

            const float* p0a = p0;
            const float* p1a = p1;
            const float* p2a = p2;
            const float* p3a = p3;
            const float* psa = ps;
            int kk = 0;

            if (elempack == 4)
            {
                for (; kk < max_kk0; kk++)
                {
                    __m128 _s = __lsx_vreplfr2vr_s(*psa++);
                    __m128 _p = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a, 0), _abs_mask), _s);
                    _absmax0 = __lsx_vfmax_s(_absmax0, _p);
                    p0a += 4;
                }

                float absmax[4];
                __lsx_vst((__m128i)_absmax0, absmax, 0);
                absmax0 = absmax[0];
                absmax1 = absmax[1];
                absmax2 = absmax[2];
                absmax3 = absmax[3];
            }

            if (elempack == 1)
            {
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = (__m128)__lsx_vld(p0a, 0);
                    __m128 _p1 = (__m128)__lsx_vld(p1a, 0);
                    __m128 _p2 = (__m128)__lsx_vld(p2a, 0);
                    __m128 _p3 = (__m128)__lsx_vld(p3a, 0);
                    __m128 _s = (__m128)__lsx_vld(psa, 0);
                    _p0 = (__m128)__lsx_vand_v((__m128i)_p0, _abs_mask);
                    _p0 = __lsx_vfmul_s(_p0, _s);
                    _p1 = (__m128)__lsx_vand_v((__m128i)_p1, _abs_mask);
                    _p1 = __lsx_vfmul_s(_p1, _s);
                    _p2 = (__m128)__lsx_vand_v((__m128i)_p2, _abs_mask);
                    _p2 = __lsx_vfmul_s(_p2, _s);
                    _p3 = (__m128)__lsx_vand_v((__m128i)_p3, _abs_mask);
                    _p3 = __lsx_vfmul_s(_p3, _s);
                    _absmax0 = __lsx_vfmax_s(_absmax0, _p0);
                    _absmax1 = __lsx_vfmax_s(_absmax1, _p1);
                    _absmax2 = __lsx_vfmax_s(_absmax2, _p2);
                    _absmax3 = __lsx_vfmax_s(_absmax3, _p3);
                    p0a += 4;
                    p1a += 4;
                    p2a += 4;
                    p3a += 4;
                    psa += 4;
                }
                absmax0 = __lsx_reduce_fmax_s(_absmax0);
                absmax1 = __lsx_reduce_fmax_s(_absmax1);
                absmax2 = __lsx_reduce_fmax_s(_absmax2);
                absmax3 = __lsx_reduce_fmax_s(_absmax3);

                for (; kk < max_kk0; kk++)
                {
                    float v0 = *p0a++;
                    float v1 = *p1a++;
                    float v2 = *p2a++;
                    float v3 = *p3a++;
                    const float s = *psa++;

                    absmax0 = std::max(absmax0, fabsf(v0) * s);
                    absmax1 = std::max(absmax1, fabsf(v1) * s);
                    absmax2 = std::max(absmax2, fabsf(v2) * s);
                    absmax3 = std::max(absmax3, fabsf(v3) * s);
                }
            }

            const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
            const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
            const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd[2] = absmax2 / 127.f;
            pd[3] = absmax3 / 127.f;
            pd += 4;

            if (elempack == 4)
            {
                __m128i _scale01 = __lsx_vilvl_w((__m128i)__lsx_vreplfr2vr_s(scale1), (__m128i)__lsx_vreplfr2vr_s(scale0));
                __m128i _scale23 = __lsx_vilvl_w((__m128i)__lsx_vreplfr2vr_s(scale3), (__m128i)__lsx_vreplfr2vr_s(scale2));
                __m128 _scale = (__m128)__lsx_vilvl_d(_scale23, _scale01);

                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _s0 = __lsx_vreplfr2vr_s(ps[0]);
                    __m128 _s1 = __lsx_vreplfr2vr_s(ps[1]);
                    __m128 _s2 = __lsx_vreplfr2vr_s(ps[2]);
                    __m128 _s3 = __lsx_vreplfr2vr_s(ps[3]);
                    __m128 _p0 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _s0), _scale);
                    __m128 _p1 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0 + 4, 0), _s1), _scale);
                    __m128 _p2 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0 + 8, 0), _s2), _scale);
                    __m128 _p3 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0 + 12, 0), _s3), _scale);
                    transpose4x4_ps(_p0, _p1, _p2, _p3);

                    ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                    ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                    pp += 16;
                    p0 += 16;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128 _s = __lsx_vreplfr2vr_s(*ps++);
                    __m128 _p = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _s), _scale);
                    ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p), 0);
                    pp += 4;
                    p0 += 4;
                }
            }

            if (elempack == 1)
            {
                __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
                __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
                __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
                __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _p1 = (__m128)__lsx_vld(p1, 0);
                    __m128 _p2 = (__m128)__lsx_vld(p2, 0);
                    __m128 _p3 = (__m128)__lsx_vld(p3, 0);
                    __m128 _s = (__m128)__lsx_vld(ps, 0);
                    _p0 = __lsx_vfmul_s(_p0, _s);
                    _p1 = __lsx_vfmul_s(_p1, _s);
                    _p2 = __lsx_vfmul_s(_p2, _s);
                    _p3 = __lsx_vfmul_s(_p3, _s);
                    _p0 = __lsx_vfmul_s(_p0, _scale0);
                    _p1 = __lsx_vfmul_s(_p1, _scale1);
                    _p2 = __lsx_vfmul_s(_p2, _scale2);
                    _p3 = __lsx_vfmul_s(_p3, _scale3);

                    ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                    ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                    pp += 16;
                    p0 += 4;
                    p1 += 4;
                    p2 += 4;
                    p3 += 4;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    float v0 = *p0++;
                    float v1 = *p1++;
                    float v2 = *p2++;
                    float v3 = *p3++;
                    const float s = *ps++;
                    v0 *= s;
                    v1 *= s;
                    v2 *= s;
                    v3 *= s;
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale1);
                    pp[2] = float2int8(v2 * scale2);
                    pp[3] = float2int8(v3 * scale3);
                    pp += 4;
                }
            }
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k;
        const float* p1 = p0 + A_hstep;

        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            const float* p0a = p0;
            const float* p1a = p1;
            const float* psa = ps;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                float v0 = *p0a++;
                float v1 = *p1a++;
                const float s = *psa++;

                absmax0 = std::max(absmax0, fabsf(v0) * s);
                absmax1 = std::max(absmax1, fabsf(v1) * s);
            }

            const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            pd += 2;

            int kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float v00 = p0[0];
                float v01 = p0[1];
                float v02 = p0[2];
                float v03 = p0[3];
                float v10 = p1[0];
                float v11 = p1[1];
                float v12 = p1[2];
                float v13 = p1[3];
                v00 *= ps[0];
                v01 *= ps[1];
                v02 *= ps[2];
                v03 *= ps[3];
                v10 *= ps[0];
                v11 *= ps[1];
                v12 *= ps[2];
                v13 *= ps[3];
                pp[0] = float2int8(v00 * scale0);
                pp[1] = float2int8(v01 * scale0);
                pp[2] = float2int8(v02 * scale0);
                pp[3] = float2int8(v03 * scale0);
                pp[4] = float2int8(v10 * scale1);
                pp[5] = float2int8(v11 * scale1);
                pp[6] = float2int8(v12 * scale1);
                pp[7] = float2int8(v13 * scale1);
                pp += 8;
                p0 += 4;
                p1 += 4;
                ps += 4;
            }
            for (; kk < max_kk0; kk++)
            {
                float v0 = *p0++;
                float v1 = *p1++;
                const float s = *ps++;
                v0 *= s;
                v1 *= s;
                pp[0] = float2int8(v0 * scale0);
                pp[1] = float2int8(v1 * scale1);
                pp += 2;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        const float* p0 = (const float*)A + (size_t)(i + ii) * A_hstep + k;

        const float* ps = input_scale_ptr;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax0 = 0.f;
            const float* p0a = p0;
            const float* psa = ps;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                float v0 = *p0a++;
                const float s = *psa++;
                absmax0 = std::max(absmax0, fabsf(v0) * s);
            }

            const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
            *pd++ = absmax0 / 127.f;

            int kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float v0 = p0[0];
                float v1 = p0[1];
                float v2 = p0[2];
                float v3 = p0[3];
                v0 *= ps[0];
                v1 *= ps[1];
                v2 *= ps[2];
                v3 *= ps[3];
                pp[0] = float2int8(v0 * scale0);
                pp[1] = float2int8(v1 * scale0);
                pp[2] = float2int8(v2 * scale0);
                pp[3] = float2int8(v3 * scale0);
                pp += 4;
                p0 += 4;
                ps += 4;
            }
            for (; kk < max_kk0; kk++)
            {
                float v0 = *p0++;
                v0 *= *ps++;
                *pp++ = float2int8(v0 * scale0);
            }
        }
    }
}

// group-major, row-major within each K4/K1 fragment

static void transpose_quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
#if NCNN_BF16
    if (A.elembits() == 16)
    {
        transpose_quantize_A_tile_wq_int8_bf16s(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif // NCNN_BF16

    const int elempack = A.elempack;
    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int block_count = (max_kk + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    if (input_scales.empty())
    {
        int ii = 0;
#if __loongarch_sx
        for (; ii + 7 < max_ii; ii += 8)
        {
            const float* p0 = (const float*)A + (size_t)(k / elempack) * A_hstep * elempack + (i + ii) * elempack;

#if __loongarch_asx
            if (elempack == 8)
            {
                const __m256i _abs_mask = (__m256i)__lasx_xvreplgr2vr_w(0x7fffffff);
                for (int g = 0; g < block_count; g++)
                {
                    const int k0 = g * block_size;
                    const int max_kk0 = std::min(max_kk - k0, block_size);
                    const float* pg = (const float*)A + (size_t)((k + k0) / 8) * A_hstep * 8 + (i + ii) * 8;
                    __m256 _a0 = (__m256)__lasx_xvreplgr2vr_w(0);
                    __m256 _a1 = _a0;
                    __m256 _a2 = _a0;
                    __m256 _a3 = _a0;
                    __m256 _a4 = _a0;
                    __m256 _a5 = _a0;
                    __m256 _a6 = _a0;
                    __m256 _a7 = _a0;
                    int kk = 0;
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        _a0 = __lasx_xvfmax_s(_a0, (__m256)__lasx_xvand_v(__lasx_xvld(pg, 0), _abs_mask));
                        _a1 = __lasx_xvfmax_s(_a1, (__m256)__lasx_xvand_v(__lasx_xvld(pg + 8, 0), _abs_mask));
                        _a2 = __lasx_xvfmax_s(_a2, (__m256)__lasx_xvand_v(__lasx_xvld(pg + 16, 0), _abs_mask));
                        _a3 = __lasx_xvfmax_s(_a3, (__m256)__lasx_xvand_v(__lasx_xvld(pg + 24, 0), _abs_mask));
                        _a4 = __lasx_xvfmax_s(_a4, (__m256)__lasx_xvand_v(__lasx_xvld(pg + 32, 0), _abs_mask));
                        _a5 = __lasx_xvfmax_s(_a5, (__m256)__lasx_xvand_v(__lasx_xvld(pg + 40, 0), _abs_mask));
                        _a6 = __lasx_xvfmax_s(_a6, (__m256)__lasx_xvand_v(__lasx_xvld(pg + 48, 0), _abs_mask));
                        _a7 = __lasx_xvfmax_s(_a7, (__m256)__lasx_xvand_v(__lasx_xvld(pg + 56, 0), _abs_mask));
                        pg += A_hstep * 8;
                    }
                    float absmax0 = __lasx_reduce_fmax_s(_a0);
                    float absmax1 = __lasx_reduce_fmax_s(_a1);
                    float absmax2 = __lasx_reduce_fmax_s(_a2);
                    float absmax3 = __lasx_reduce_fmax_s(_a3);
                    float absmax4 = __lasx_reduce_fmax_s(_a4);
                    float absmax5 = __lasx_reduce_fmax_s(_a5);
                    float absmax6 = __lasx_reduce_fmax_s(_a6);
                    float absmax7 = __lasx_reduce_fmax_s(_a7);
                    for (; kk < max_kk0; kk++)
                    {
                        absmax0 = std::max(absmax0, fabsf(pg[0]));
                        absmax1 = std::max(absmax1, fabsf(pg[8]));
                        absmax2 = std::max(absmax2, fabsf(pg[16]));
                        absmax3 = std::max(absmax3, fabsf(pg[24]));
                        absmax4 = std::max(absmax4, fabsf(pg[32]));
                        absmax5 = std::max(absmax5, fabsf(pg[40]));
                        absmax6 = std::max(absmax6, fabsf(pg[48]));
                        absmax7 = std::max(absmax7, fabsf(pg[56]));
                        pg++;
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
                    __m256 _s0 = __lasx_xvreplfr2vr_s(absmax0 == 0.f ? 1.f : 127.f / absmax0);
                    __m256 _s1 = __lasx_xvreplfr2vr_s(absmax1 == 0.f ? 1.f : 127.f / absmax1);
                    __m256 _s2 = __lasx_xvreplfr2vr_s(absmax2 == 0.f ? 1.f : 127.f / absmax2);
                    __m256 _s3 = __lasx_xvreplfr2vr_s(absmax3 == 0.f ? 1.f : 127.f / absmax3);
                    __m256 _s4 = __lasx_xvreplfr2vr_s(absmax4 == 0.f ? 1.f : 127.f / absmax4);
                    __m256 _s5 = __lasx_xvreplfr2vr_s(absmax5 == 0.f ? 1.f : 127.f / absmax5);
                    __m256 _s6 = __lasx_xvreplfr2vr_s(absmax6 == 0.f ? 1.f : 127.f / absmax6);
                    __m256 _s7 = __lasx_xvreplfr2vr_s(absmax7 == 0.f ? 1.f : 127.f / absmax7);
                    pg = (const float*)A + (size_t)((k + k0) / 8) * A_hstep * 8 + (i + ii) * 8;
                    kk = 0;
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        __m256 _p0 = __lasx_xvfmul_s((__m256)__lasx_xvld(pg, 0), _s0);
                        __m256 _p1 = __lasx_xvfmul_s((__m256)__lasx_xvld(pg + 8, 0), _s1);
                        __m256 _p2 = __lasx_xvfmul_s((__m256)__lasx_xvld(pg + 16, 0), _s2);
                        __m256 _p3 = __lasx_xvfmul_s((__m256)__lasx_xvld(pg + 24, 0), _s3);
                        __m256 _p4 = __lasx_xvfmul_s((__m256)__lasx_xvld(pg + 32, 0), _s4);
                        __m256 _p5 = __lasx_xvfmul_s((__m256)__lasx_xvld(pg + 40, 0), _s5);
                        __m256 _p6 = __lasx_xvfmul_s((__m256)__lasx_xvld(pg + 48, 0), _s6);
                        __m256 _p7 = __lasx_xvfmul_s((__m256)__lasx_xvld(pg + 56, 0), _s7);
                        ((int64_t*)pp)[0] = float2int8(__lasx_extract_128_lo_s(_p0), __lasx_extract_128_lo_s(_p1));
                        ((int64_t*)pp)[1] = float2int8(__lasx_extract_128_lo_s(_p2), __lasx_extract_128_lo_s(_p3));
                        ((int64_t*)pp)[2] = float2int8(__lasx_extract_128_lo_s(_p4), __lasx_extract_128_lo_s(_p5));
                        ((int64_t*)pp)[3] = float2int8(__lasx_extract_128_lo_s(_p6), __lasx_extract_128_lo_s(_p7));
                        ((int64_t*)pp)[4] = float2int8(__lasx_extract_128_hi_s(_p0), __lasx_extract_128_hi_s(_p1));
                        ((int64_t*)pp)[5] = float2int8(__lasx_extract_128_hi_s(_p2), __lasx_extract_128_hi_s(_p3));
                        ((int64_t*)pp)[6] = float2int8(__lasx_extract_128_hi_s(_p4), __lasx_extract_128_hi_s(_p5));
                        ((int64_t*)pp)[7] = float2int8(__lasx_extract_128_hi_s(_p6), __lasx_extract_128_hi_s(_p7));
                        pp += 64;
                        pg += A_hstep * 8;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        pp[0] = float2int8(pg[0] * (absmax0 == 0.f ? 1.f : 127.f / absmax0));
                        pp[1] = float2int8(pg[8] * (absmax1 == 0.f ? 1.f : 127.f / absmax1));
                        pp[2] = float2int8(pg[16] * (absmax2 == 0.f ? 1.f : 127.f / absmax2));
                        pp[3] = float2int8(pg[24] * (absmax3 == 0.f ? 1.f : 127.f / absmax3));
                        pp[4] = float2int8(pg[32] * (absmax4 == 0.f ? 1.f : 127.f / absmax4));
                        pp[5] = float2int8(pg[40] * (absmax5 == 0.f ? 1.f : 127.f / absmax5));
                        pp[6] = float2int8(pg[48] * (absmax6 == 0.f ? 1.f : 127.f / absmax6));
                        pp[7] = float2int8(pg[56] * (absmax7 == 0.f ? 1.f : 127.f / absmax7));
                        pp += 8;
                        pg++;
                    }
                }
            }
#endif // __loongarch_asx

            if (elempack == 4)
            {
                const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
                for (int g = 0; g < block_count; g++)
                {
                    const int k0 = g * block_size;
                    const int max_kk0 = std::min(max_kk - k0, block_size);
                    const float* pg = (const float*)A + (size_t)((k + k0) / 4) * A_hstep * 4 + (i + ii) * 4;
                    __m128 _a0 = (__m128)__lsx_vldi(0);
                    __m128 _a1 = _a0;
                    __m128 _a2 = _a0;
                    __m128 _a3 = _a0;
                    __m128 _a4 = _a0;
                    __m128 _a5 = _a0;
                    __m128 _a6 = _a0;
                    __m128 _a7 = _a0;
                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        _a0 = __lsx_vfmax_s(_a0, (__m128)__lsx_vand_v(__lsx_vld(pg, 0), _abs_mask));
                        _a1 = __lsx_vfmax_s(_a1, (__m128)__lsx_vand_v(__lsx_vld(pg + 4, 0), _abs_mask));
                        _a2 = __lsx_vfmax_s(_a2, (__m128)__lsx_vand_v(__lsx_vld(pg + 8, 0), _abs_mask));
                        _a3 = __lsx_vfmax_s(_a3, (__m128)__lsx_vand_v(__lsx_vld(pg + 12, 0), _abs_mask));
                        _a4 = __lsx_vfmax_s(_a4, (__m128)__lsx_vand_v(__lsx_vld(pg + 16, 0), _abs_mask));
                        _a5 = __lsx_vfmax_s(_a5, (__m128)__lsx_vand_v(__lsx_vld(pg + 20, 0), _abs_mask));
                        _a6 = __lsx_vfmax_s(_a6, (__m128)__lsx_vand_v(__lsx_vld(pg + 24, 0), _abs_mask));
                        _a7 = __lsx_vfmax_s(_a7, (__m128)__lsx_vand_v(__lsx_vld(pg + 28, 0), _abs_mask));
                        pg += A_hstep * 4;
                    }
                    float absmax0 = __lsx_reduce_fmax_s(_a0);
                    float absmax1 = __lsx_reduce_fmax_s(_a1);
                    float absmax2 = __lsx_reduce_fmax_s(_a2);
                    float absmax3 = __lsx_reduce_fmax_s(_a3);
                    float absmax4 = __lsx_reduce_fmax_s(_a4);
                    float absmax5 = __lsx_reduce_fmax_s(_a5);
                    float absmax6 = __lsx_reduce_fmax_s(_a6);
                    float absmax7 = __lsx_reduce_fmax_s(_a7);
                    for (; kk < max_kk0; kk++)
                    {
                        absmax0 = std::max(absmax0, fabsf(pg[0]));
                        absmax1 = std::max(absmax1, fabsf(pg[4]));
                        absmax2 = std::max(absmax2, fabsf(pg[8]));
                        absmax3 = std::max(absmax3, fabsf(pg[12]));
                        absmax4 = std::max(absmax4, fabsf(pg[16]));
                        absmax5 = std::max(absmax5, fabsf(pg[20]));
                        absmax6 = std::max(absmax6, fabsf(pg[24]));
                        absmax7 = std::max(absmax7, fabsf(pg[28]));
                        pg++;
                    }
                    const float s0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    const float s1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                    const float s2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
                    const float s3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
                    const float s4 = absmax4 == 0.f ? 1.f : 127.f / absmax4;
                    const float s5 = absmax5 == 0.f ? 1.f : 127.f / absmax5;
                    const float s6 = absmax6 == 0.f ? 1.f : 127.f / absmax6;
                    const float s7 = absmax7 == 0.f ? 1.f : 127.f / absmax7;
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    pd[2] = absmax2 / 127.f;
                    pd[3] = absmax3 / 127.f;
                    pd[4] = absmax4 / 127.f;
                    pd[5] = absmax5 / 127.f;
                    pd[6] = absmax6 / 127.f;
                    pd[7] = absmax7 / 127.f;
                    pd += 8;
                    pg = (const float*)A + (size_t)((k + k0) / 4) * A_hstep * 4 + (i + ii) * 4;
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        ((int64_t*)pp)[0] = float2int8(__lsx_vfmul_s((__m128)__lsx_vld(pg, 0), __lsx_vreplfr2vr_s(s0)), __lsx_vfmul_s((__m128)__lsx_vld(pg + 4, 0), __lsx_vreplfr2vr_s(s1)));
                        ((int64_t*)pp)[1] = float2int8(__lsx_vfmul_s((__m128)__lsx_vld(pg + 8, 0), __lsx_vreplfr2vr_s(s2)), __lsx_vfmul_s((__m128)__lsx_vld(pg + 12, 0), __lsx_vreplfr2vr_s(s3)));
                        ((int64_t*)pp)[2] = float2int8(__lsx_vfmul_s((__m128)__lsx_vld(pg + 16, 0), __lsx_vreplfr2vr_s(s4)), __lsx_vfmul_s((__m128)__lsx_vld(pg + 20, 0), __lsx_vreplfr2vr_s(s5)));
                        ((int64_t*)pp)[3] = float2int8(__lsx_vfmul_s((__m128)__lsx_vld(pg + 24, 0), __lsx_vreplfr2vr_s(s6)), __lsx_vfmul_s((__m128)__lsx_vld(pg + 28, 0), __lsx_vreplfr2vr_s(s7)));
                        pp += 32;
                        pg += A_hstep * 4;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        pp[0] = float2int8(pg[0] * s0);
                        pp[1] = float2int8(pg[4] * s1);
                        pp[2] = float2int8(pg[8] * s2);
                        pp[3] = float2int8(pg[12] * s3);
                        pp[4] = float2int8(pg[16] * s4);
                        pp[5] = float2int8(pg[20] * s5);
                        pp[6] = float2int8(pg[24] * s6);
                        pp[7] = float2int8(pg[28] * s7);
                        pp += 8;
                        pg++;
                    }
                }
            }

            if (elempack == 1)
            {
                for (int g = 0; g < block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax0 = (__m128)__lsx_vldi(0);
                    __m128 _absmax1 = (__m128)__lsx_vldi(0);
                    const float* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        __m128 _p0 = (__m128)__lsx_vld(p0a, 0);
                        __m128 _p1 = (__m128)__lsx_vld(p0a + 4, 0);
                        _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_p0, _abs_mask));
                        _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_p1, _abs_mask));
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

                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
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
                        transpose4x4_ps(_p0, _p1, _p2, _p3);
                        transpose4x4_ps(_p4, _p5, _p6, _p7);
                        *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_p0, _scale0), __lsx_vfmul_s(_p1, _scale1));
                        *((int64_t*)(pp + 8)) = float2int8(__lsx_vfmul_s(_p2, _scale2), __lsx_vfmul_s(_p3, _scale3));
                        *((int64_t*)(pp + 16)) = float2int8(__lsx_vfmul_s(_p4, _scale4), __lsx_vfmul_s(_p5, _scale5));
                        *((int64_t*)(pp + 24)) = float2int8(__lsx_vfmul_s(_p6, _scale6), __lsx_vfmul_s(_p7, _scale7));
                        pp += 32;
                        p0 += (size_t)4 * A_hstep;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p0 = (__m128)__lsx_vld(p0, 0);
                        __m128 _p1 = (__m128)__lsx_vld(p0 + 4, 0);
                        _p0 = __lsx_vfmul_s(_p0, _scales0);
                        _p1 = __lsx_vfmul_s(_p1, _scales1);
                        *((int64_t*)pp) = float2int8(_p0, _p1);
                        pp += 8;
                        p0 += A_hstep;
                    }
                }
            }
        }
        for (; ii + 3 < max_ii; ii += 4)
        {
            const float* p0 = (const float*)A + (size_t)(k / elempack) * A_hstep * elempack + (i + ii) * elempack;

#if __loongarch_asx
            if (elempack == 8)
            {
                for (int g = 0; g < block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m256i _abs_mask = (__m256i)__lasx_xvreplgr2vr_w(0x7fffffff);
                    __m256 _absmax0 = (__m256)__lasx_xvreplgr2vr_w(0);
                    __m256 _absmax1 = _absmax0;
                    __m256 _absmax2 = _absmax0;
                    __m256 _absmax3 = _absmax0;
                    const float* p0a = p0;
                    for (int kk = 0; kk + 7 < max_kk0; kk += 8)
                    {
                        _absmax0 = __lasx_xvfmax_s(_absmax0, (__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a, 0), _abs_mask));
                        _absmax1 = __lasx_xvfmax_s(_absmax1, (__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a + 8, 0), _abs_mask));
                        _absmax2 = __lasx_xvfmax_s(_absmax2, (__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a + 16, 0), _abs_mask));
                        _absmax3 = __lasx_xvfmax_s(_absmax3, (__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a + 24, 0), _abs_mask));
                        p0a += A_hstep * 8;
                    }

                    const float absmax0 = __lasx_reduce_fmax_s(_absmax0);
                    const float absmax1 = __lasx_reduce_fmax_s(_absmax1);
                    const float absmax2 = __lasx_reduce_fmax_s(_absmax2);
                    const float absmax3 = __lasx_reduce_fmax_s(_absmax3);
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    pd[2] = absmax2 / 127.f;
                    pd[3] = absmax3 / 127.f;
                    pd += 4;

                    __m256 _scale0 = __lasx_xvreplfr2vr_s(absmax0 == 0.f ? 1.f : 127.f / absmax0);
                    __m256 _scale1 = __lasx_xvreplfr2vr_s(absmax1 == 0.f ? 1.f : 127.f / absmax1);
                    __m256 _scale2 = __lasx_xvreplfr2vr_s(absmax2 == 0.f ? 1.f : 127.f / absmax2);
                    __m256 _scale3 = __lasx_xvreplfr2vr_s(absmax3 == 0.f ? 1.f : 127.f / absmax3);
                    for (int kk = 0; kk + 7 < max_kk0; kk += 8)
                    {
                        __m256 _p0 = __lasx_xvfmul_s((__m256)__lasx_xvld(p0, 0), _scale0);
                        __m256 _p1 = __lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 8, 0), _scale1);
                        __m256 _p2 = __lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 16, 0), _scale2);
                        __m256 _p3 = __lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 24, 0), _scale3);
                        ((int64_t*)pp)[0] = float2int8(__lasx_extract_128_lo_s(_p0), __lasx_extract_128_lo_s(_p1));
                        ((int64_t*)pp)[1] = float2int8(__lasx_extract_128_lo_s(_p2), __lasx_extract_128_lo_s(_p3));
                        ((int64_t*)pp)[2] = float2int8(__lasx_extract_128_hi_s(_p0), __lasx_extract_128_hi_s(_p1));
                        ((int64_t*)pp)[3] = float2int8(__lasx_extract_128_hi_s(_p2), __lasx_extract_128_hi_s(_p3));
                        pp += 32;
                        p0 += A_hstep * 8;
                    }
                }
            }
#endif // __loongarch_asx

            if (elempack == 4)
            {
                for (int g = 0; g < block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax0 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax1 = _absmax0;
                    __m128 _absmax2 = _absmax0;
                    __m128 _absmax3 = _absmax0;
                    const float* p0a = p0;
                    for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                    {
                        _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a, 0), _abs_mask));
                        _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a + 4, 0), _abs_mask));
                        _absmax2 = __lsx_vfmax_s(_absmax2, (__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a + 8, 0), _abs_mask));
                        _absmax3 = __lsx_vfmax_s(_absmax3, (__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a + 12, 0), _abs_mask));
                        p0a += A_hstep * 4;
                    }

                    const float absmax0 = __lsx_reduce_fmax_s(_absmax0);
                    const float absmax1 = __lsx_reduce_fmax_s(_absmax1);
                    const float absmax2 = __lsx_reduce_fmax_s(_absmax2);
                    const float absmax3 = __lsx_reduce_fmax_s(_absmax3);
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    pd[2] = absmax2 / 127.f;
                    pd[3] = absmax3 / 127.f;
                    pd += 4;

                    __m128 _scale0 = __lsx_vreplfr2vr_s(absmax0 == 0.f ? 1.f : 127.f / absmax0);
                    __m128 _scale1 = __lsx_vreplfr2vr_s(absmax1 == 0.f ? 1.f : 127.f / absmax1);
                    __m128 _scale2 = __lsx_vreplfr2vr_s(absmax2 == 0.f ? 1.f : 127.f / absmax2);
                    __m128 _scale3 = __lsx_vreplfr2vr_s(absmax3 == 0.f ? 1.f : 127.f / absmax3);
                    for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = __lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _scale0);
                        __m128 _p1 = __lsx_vfmul_s((__m128)__lsx_vld(p0 + 4, 0), _scale1);
                        __m128 _p2 = __lsx_vfmul_s((__m128)__lsx_vld(p0 + 8, 0), _scale2);
                        __m128 _p3 = __lsx_vfmul_s((__m128)__lsx_vld(p0 + 12, 0), _scale3);
                        ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                        ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                        pp += 16;
                        p0 += A_hstep * 4;
                    }
                }
            }
            if (elempack == 1)
            {
                for (int g = 0; g < block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax = (__m128)__lsx_vreplgr2vr_w(0);
                    const float* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        _absmax = __lsx_vfmax_s(_absmax, (__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a, 0), _abs_mask));
                        p0a += A_hstep;
                    }

                    const float absmax0 = __lsx_reduce_fmax_s((__m128)__lsx_vreplvei_w((__m128i)_absmax, 0));
                    const float absmax1 = __lsx_reduce_fmax_s((__m128)__lsx_vreplvei_w((__m128i)_absmax, 1));
                    const float absmax2 = __lsx_reduce_fmax_s((__m128)__lsx_vreplvei_w((__m128i)_absmax, 2));
                    const float absmax3 = __lsx_reduce_fmax_s((__m128)__lsx_vreplvei_w((__m128i)_absmax, 3));
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    pd[2] = absmax2 / 127.f;
                    pd[3] = absmax3 / 127.f;
                    pd += 4;

                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                    const float scale2 = absmax2 == 0.f ? 1.f : 127.f / absmax2;
                    const float scale3 = absmax3 == 0.f ? 1.f : 127.f / absmax3;
                    const float scales[4] = {scale0, scale1, scale2, scale3};
                    __m128 _scale = (__m128)__lsx_vld(scales, 0);
                    __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
                    __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
                    __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
                    __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        const float* p1 = p0 + A_hstep;
                        const float* p2 = p1 + A_hstep;
                        const float* p3 = p2 + A_hstep;
                        __m128 _p0 = (__m128)__lsx_vld(p0, 0);
                        __m128 _p1 = (__m128)__lsx_vld(p1, 0);
                        __m128 _p2 = (__m128)__lsx_vld(p2, 0);
                        __m128 _p3 = (__m128)__lsx_vld(p3, 0);
                        transpose4x4_ps(_p0, _p1, _p2, _p3);
                        ((int64_t*)pp)[0] = float2int8(__lsx_vfmul_s(_p0, _scale0), __lsx_vfmul_s(_p1, _scale1));
                        ((int64_t*)pp)[1] = float2int8(__lsx_vfmul_s(_p2, _scale2), __lsx_vfmul_s(_p3, _scale3));
                        pp += 16;
                        p0 = p3 + A_hstep;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        ((int*)pp)[0] = __lsx_vpickve2gr_w(float2int8(__lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _scale)), 0);
                        pp += 4;
                        p0 += A_hstep;
                    }
                }
            }
        }
#endif // __loongarch_sx
        for (; ii + 1 < max_ii; ii += 2)
        {
            const float* p0 = (const float*)A + (size_t)(k / elempack) * A_hstep * elempack + (i + ii) * elempack;
#if __loongarch_sx
#if __loongarch_asx
            if (elempack == 8)
            {
                for (int g = 0; g < block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m256i _abs_mask = (__m256i)__lasx_xvreplgr2vr_w(0x7fffffff);
                    __m256 _absmax0 = (__m256)__lasx_xvreplgr2vr_w(0);
                    __m256 _absmax1 = _absmax0;
                    const float* p0a = p0;
                    for (int kk = 0; kk + 7 < max_kk0; kk += 8)
                    {
                        _absmax0 = __lasx_xvfmax_s(_absmax0, (__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a, 0), _abs_mask));
                        _absmax1 = __lasx_xvfmax_s(_absmax1, (__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a + 8, 0), _abs_mask));
                        p0a += A_hstep * 8;
                    }
                    const float absmax0 = __lasx_reduce_fmax_s(_absmax0);
                    const float absmax1 = __lasx_reduce_fmax_s(_absmax1);
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    pd += 2;
                    __m256 _scale0 = __lasx_xvreplfr2vr_s(absmax0 == 0.f ? 1.f : 127.f / absmax0);
                    __m256 _scale1 = __lasx_xvreplfr2vr_s(absmax1 == 0.f ? 1.f : 127.f / absmax1);
                    for (int kk = 0; kk + 7 < max_kk0; kk += 8)
                    {
                        __m256 _p0 = __lasx_xvfmul_s((__m256)__lasx_xvld(p0, 0), _scale0);
                        __m256 _p1 = __lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 8, 0), _scale1);
                        ((int64_t*)pp)[0] = float2int8(__lasx_extract_128_lo_s(_p0), __lasx_extract_128_lo_s(_p1));
                        ((int64_t*)pp)[1] = float2int8(__lasx_extract_128_hi_s(_p0), __lasx_extract_128_hi_s(_p1));
                        pp += 16;
                        p0 += A_hstep * 8;
                    }
                }
            }
#endif // __loongarch_asx

            if (elempack == 4)
            {
                for (int g = 0; g < block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax0 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax1 = _absmax0;
                    const float* p0a = p0;
                    for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                    {
                        _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a, 0), _abs_mask));
                        _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a + 4, 0), _abs_mask));
                        p0a += A_hstep * 4;
                    }
                    const float absmax0 = __lsx_reduce_fmax_s(_absmax0);
                    const float absmax1 = __lsx_reduce_fmax_s(_absmax1);
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    pd += 2;
                    __m128 _scale0 = __lsx_vreplfr2vr_s(absmax0 == 0.f ? 1.f : 127.f / absmax0);
                    __m128 _scale1 = __lsx_vreplfr2vr_s(absmax1 == 0.f ? 1.f : 127.f / absmax1);
                    for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                    {
                        ((int64_t*)pp)[0] = float2int8(__lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _scale0), __lsx_vfmul_s((__m128)__lsx_vld(p0 + 4, 0), _scale1));
                        pp += 8;
                        p0 += A_hstep * 4;
                    }
                }
            }
#endif // __loongarch_sx

            if (elempack == 1)
            {
                for (int g = 0; g < block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    float absmax0 = 0.f;
                    float absmax1 = 0.f;
                    const float* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        absmax0 = std::max(absmax0, fabsf(p0a[0]));
                        absmax1 = std::max(absmax1, fabsf(p0a[1]));
                        p0a += A_hstep;
                    }
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    pd += 2;
                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        const float* p1 = p0 + A_hstep;
                        const float* p2 = p1 + A_hstep;
                        const float* p3 = p2 + A_hstep;
                        pp[0] = float2int8(p0[0] * scale0);
                        pp[1] = float2int8(p1[0] * scale0);
                        pp[2] = float2int8(p2[0] * scale0);
                        pp[3] = float2int8(p3[0] * scale0);
                        pp[4] = float2int8(p0[1] * scale1);
                        pp[5] = float2int8(p1[1] * scale1);
                        pp[6] = float2int8(p2[1] * scale1);
                        pp[7] = float2int8(p3[1] * scale1);
                        pp += 8;
                        p0 = p3 + A_hstep;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        pp[0] = float2int8(p0[0] * scale0);
                        pp[1] = float2int8(p0[1] * scale1);
                        pp += 2;
                        p0 += A_hstep;
                    }
                }
            }
        }
        for (; ii < max_ii; ii++)
        {
            const float* p0 = (const float*)A + (size_t)(k / elempack) * A_hstep * elempack + (i + ii) * elempack;

#if __loongarch_asx
            if (elempack == 8)
            {
                for (int g = 0; g < block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m256i _abs_mask = (__m256i)__lasx_xvreplgr2vr_w(0x7fffffff);
                    __m256 _absmax = (__m256)__lasx_xvreplgr2vr_w(0);
                    const float* p0a = p0;
                    for (int kk = 0; kk + 7 < max_kk0; kk += 8)
                    {
                        _absmax = __lasx_xvfmax_s(_absmax, (__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a, 0), _abs_mask));
                        p0a += A_hstep * 8;
                    }
                    const float absmax = __lasx_reduce_fmax_s(_absmax);
                    *pd++ = absmax / 127.f;
                    __m256 _scale = __lasx_xvreplfr2vr_s(absmax == 0.f ? 1.f : 127.f / absmax);
                    for (int kk = 0; kk + 7 < max_kk0; kk += 8)
                    {
                        __m256 _p = __lasx_xvfmul_s((__m256)__lasx_xvld(p0, 0), _scale);
                        ((int*)pp)[0] = __lsx_vpickve2gr_w(float2int8(__lasx_extract_128_lo_s(_p)), 0);
                        ((int*)pp)[1] = __lsx_vpickve2gr_w(float2int8(__lasx_extract_128_hi_s(_p)), 0);
                        pp += 8;
                        p0 += A_hstep * 8;
                    }
                }
            }
#endif // __loongarch_asx

#if __loongarch_sx
            if (elempack == 4)
            {
                for (int g = 0; g < block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax = (__m128)__lsx_vreplgr2vr_w(0);
                    const float* p0a = p0;
                    for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                    {
                        _absmax = __lsx_vfmax_s(_absmax, (__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a, 0), _abs_mask));
                        p0a += A_hstep * 4;
                    }
                    const float absmax = __lsx_reduce_fmax_s(_absmax);
                    *pd++ = absmax / 127.f;
                    __m128 _scale = __lsx_vreplfr2vr_s(absmax == 0.f ? 1.f : 127.f / absmax);
                    for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                    {
                        ((int*)pp)[0] = __lsx_vpickve2gr_w(float2int8(__lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _scale)), 0);
                        pp += 4;
                        p0 += A_hstep * 4;
                    }
                }
            }
#endif // __loongarch_sx

            if (elempack == 1)
            {
                for (int g = 0; g < block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    float absmax = 0.f;
                    const float* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        absmax = std::max(absmax, fabsf(*p0a));
                        p0a += A_hstep;
                    }
                    const float scale = absmax == 0.f ? 1.f : 127.f / absmax;
                    *pd++ = absmax / 127.f;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        *pp++ = float2int8(*p0 * scale);
                        p0 += A_hstep;
                    }
                }
            }
        }

        return;
    }

    const float* input_scale_ptr = (const float*)input_scales + k;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + (size_t)(k / elempack) * A_hstep * elempack + (i + ii) * elempack;
        const float* ps = input_scale_ptr;

#if __loongarch_asx
        if (elempack == 8)
        {
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const __m256i _abs_mask = (__m256i)__lasx_xvreplgr2vr_w(0x7fffffff);
                __m256 _absmax0 = (__m256)__lasx_xvreplgr2vr_w(0);
                __m256 _absmax1 = _absmax0;
                __m256 _absmax2 = _absmax0;
                __m256 _absmax3 = _absmax0;
                __m256 _absmax4 = _absmax0;
                __m256 _absmax5 = _absmax0;
                __m256 _absmax6 = _absmax0;
                __m256 _absmax7 = _absmax0;
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk + 7 < max_kk0; kk += 8)
                {
                    __m256 _s = (__m256)__lasx_xvld(psa, 0);
                    _absmax0 = __lasx_xvfmax_s(_absmax0, __lasx_xvfmul_s((__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a, 0), _abs_mask), _s));
                    _absmax1 = __lasx_xvfmax_s(_absmax1, __lasx_xvfmul_s((__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a + 8, 0), _abs_mask), _s));
                    _absmax2 = __lasx_xvfmax_s(_absmax2, __lasx_xvfmul_s((__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a + 16, 0), _abs_mask), _s));
                    _absmax3 = __lasx_xvfmax_s(_absmax3, __lasx_xvfmul_s((__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a + 24, 0), _abs_mask), _s));
                    _absmax4 = __lasx_xvfmax_s(_absmax4, __lasx_xvfmul_s((__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a + 32, 0), _abs_mask), _s));
                    _absmax5 = __lasx_xvfmax_s(_absmax5, __lasx_xvfmul_s((__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a + 40, 0), _abs_mask), _s));
                    _absmax6 = __lasx_xvfmax_s(_absmax6, __lasx_xvfmul_s((__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a + 48, 0), _abs_mask), _s));
                    _absmax7 = __lasx_xvfmax_s(_absmax7, __lasx_xvfmul_s((__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a + 56, 0), _abs_mask), _s));
                    p0a += A_hstep * 8;
                    psa += 8;
                }
                const float absmax0 = __lasx_reduce_fmax_s(_absmax0);
                const float absmax1 = __lasx_reduce_fmax_s(_absmax1);
                const float absmax2 = __lasx_reduce_fmax_s(_absmax2);
                const float absmax3 = __lasx_reduce_fmax_s(_absmax3);
                const float absmax4 = __lasx_reduce_fmax_s(_absmax4);
                const float absmax5 = __lasx_reduce_fmax_s(_absmax5);
                const float absmax6 = __lasx_reduce_fmax_s(_absmax6);
                const float absmax7 = __lasx_reduce_fmax_s(_absmax7);
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd[2] = absmax2 / 127.f;
                pd[3] = absmax3 / 127.f;
                pd[4] = absmax4 / 127.f;
                pd[5] = absmax5 / 127.f;
                pd[6] = absmax6 / 127.f;
                pd[7] = absmax7 / 127.f;
                pd += 8;
                __m256 _scale0 = __lasx_xvreplfr2vr_s(absmax0 == 0.f ? 1.f : 127.f / absmax0);
                __m256 _scale1 = __lasx_xvreplfr2vr_s(absmax1 == 0.f ? 1.f : 127.f / absmax1);
                __m256 _scale2 = __lasx_xvreplfr2vr_s(absmax2 == 0.f ? 1.f : 127.f / absmax2);
                __m256 _scale3 = __lasx_xvreplfr2vr_s(absmax3 == 0.f ? 1.f : 127.f / absmax3);
                __m256 _scale4 = __lasx_xvreplfr2vr_s(absmax4 == 0.f ? 1.f : 127.f / absmax4);
                __m256 _scale5 = __lasx_xvreplfr2vr_s(absmax5 == 0.f ? 1.f : 127.f / absmax5);
                __m256 _scale6 = __lasx_xvreplfr2vr_s(absmax6 == 0.f ? 1.f : 127.f / absmax6);
                __m256 _scale7 = __lasx_xvreplfr2vr_s(absmax7 == 0.f ? 1.f : 127.f / absmax7);
                for (int kk = 0; kk + 7 < max_kk0; kk += 8)
                {
                    __m256 _s = (__m256)__lasx_xvld(ps, 0);
                    __m256 _p0 = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0, 0), _s), _scale0);
                    __m256 _p1 = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 8, 0), _s), _scale1);
                    __m256 _p2 = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 16, 0), _s), _scale2);
                    __m256 _p3 = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 24, 0), _s), _scale3);
                    __m256 _p4 = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 32, 0), _s), _scale4);
                    __m256 _p5 = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 40, 0), _s), _scale5);
                    __m256 _p6 = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 48, 0), _s), _scale6);
                    __m256 _p7 = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 56, 0), _s), _scale7);
                    ((int64_t*)pp)[0] = float2int8(__lasx_extract_128_lo_s(_p0), __lasx_extract_128_lo_s(_p1));
                    ((int64_t*)pp)[1] = float2int8(__lasx_extract_128_lo_s(_p2), __lasx_extract_128_lo_s(_p3));
                    ((int64_t*)pp)[2] = float2int8(__lasx_extract_128_lo_s(_p4), __lasx_extract_128_lo_s(_p5));
                    ((int64_t*)pp)[3] = float2int8(__lasx_extract_128_lo_s(_p6), __lasx_extract_128_lo_s(_p7));
                    ((int64_t*)pp)[4] = float2int8(__lasx_extract_128_hi_s(_p0), __lasx_extract_128_hi_s(_p1));
                    ((int64_t*)pp)[5] = float2int8(__lasx_extract_128_hi_s(_p2), __lasx_extract_128_hi_s(_p3));
                    ((int64_t*)pp)[6] = float2int8(__lasx_extract_128_hi_s(_p4), __lasx_extract_128_hi_s(_p5));
                    ((int64_t*)pp)[7] = float2int8(__lasx_extract_128_hi_s(_p6), __lasx_extract_128_hi_s(_p7));
                    pp += 64;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
            }
        }
#endif // __loongarch_asx

        if (elempack == 4)
        {
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                __m128 _absmax0 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax1 = _absmax0;
                __m128 _absmax2 = _absmax0;
                __m128 _absmax3 = _absmax0;
                __m128 _absmax4 = _absmax0;
                __m128 _absmax5 = _absmax0;
                __m128 _absmax6 = _absmax0;
                __m128 _absmax7 = _absmax0;
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _s = (__m128)__lsx_vld(psa, 0);
                    _absmax0 = __lsx_vfmax_s(_absmax0, __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a, 0), _abs_mask), _s));
                    _absmax1 = __lsx_vfmax_s(_absmax1, __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a + 4, 0), _abs_mask), _s));
                    _absmax2 = __lsx_vfmax_s(_absmax2, __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a + 8, 0), _abs_mask), _s));
                    _absmax3 = __lsx_vfmax_s(_absmax3, __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a + 12, 0), _abs_mask), _s));
                    _absmax4 = __lsx_vfmax_s(_absmax4, __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a + 16, 0), _abs_mask), _s));
                    _absmax5 = __lsx_vfmax_s(_absmax5, __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a + 20, 0), _abs_mask), _s));
                    _absmax6 = __lsx_vfmax_s(_absmax6, __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a + 24, 0), _abs_mask), _s));
                    _absmax7 = __lsx_vfmax_s(_absmax7, __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a + 28, 0), _abs_mask), _s));
                    p0a += A_hstep * 4;
                    psa += 4;
                }
                const float absmax0 = __lsx_reduce_fmax_s(_absmax0);
                const float absmax1 = __lsx_reduce_fmax_s(_absmax1);
                const float absmax2 = __lsx_reduce_fmax_s(_absmax2);
                const float absmax3 = __lsx_reduce_fmax_s(_absmax3);
                const float absmax4 = __lsx_reduce_fmax_s(_absmax4);
                const float absmax5 = __lsx_reduce_fmax_s(_absmax5);
                const float absmax6 = __lsx_reduce_fmax_s(_absmax6);
                const float absmax7 = __lsx_reduce_fmax_s(_absmax7);
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd[2] = absmax2 / 127.f;
                pd[3] = absmax3 / 127.f;
                pd[4] = absmax4 / 127.f;
                pd[5] = absmax5 / 127.f;
                pd[6] = absmax6 / 127.f;
                pd[7] = absmax7 / 127.f;
                pd += 8;
                __m128 _scale0 = __lsx_vreplfr2vr_s(absmax0 == 0.f ? 1.f : 127.f / absmax0);
                __m128 _scale1 = __lsx_vreplfr2vr_s(absmax1 == 0.f ? 1.f : 127.f / absmax1);
                __m128 _scale2 = __lsx_vreplfr2vr_s(absmax2 == 0.f ? 1.f : 127.f / absmax2);
                __m128 _scale3 = __lsx_vreplfr2vr_s(absmax3 == 0.f ? 1.f : 127.f / absmax3);
                __m128 _scale4 = __lsx_vreplfr2vr_s(absmax4 == 0.f ? 1.f : 127.f / absmax4);
                __m128 _scale5 = __lsx_vreplfr2vr_s(absmax5 == 0.f ? 1.f : 127.f / absmax5);
                __m128 _scale6 = __lsx_vreplfr2vr_s(absmax6 == 0.f ? 1.f : 127.f / absmax6);
                __m128 _scale7 = __lsx_vreplfr2vr_s(absmax7 == 0.f ? 1.f : 127.f / absmax7);
                for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _s = (__m128)__lsx_vld(ps, 0);
                    __m128 _p0 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _s), _scale0);
                    __m128 _p1 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0 + 4, 0), _s), _scale1);
                    __m128 _p2 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0 + 8, 0), _s), _scale2);
                    __m128 _p3 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0 + 12, 0), _s), _scale3);
                    __m128 _p4 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0 + 16, 0), _s), _scale4);
                    __m128 _p5 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0 + 20, 0), _s), _scale5);
                    __m128 _p6 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0 + 24, 0), _s), _scale6);
                    __m128 _p7 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0 + 28, 0), _s), _scale7);
                    ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                    ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                    ((int64_t*)pp)[2] = float2int8(_p4, _p5);
                    ((int64_t*)pp)[3] = float2int8(_p6, _p7);
                    pp += 32;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
            }
        }

        if (elempack == 1)
        {
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
                __m128 _absmax0 = (__m128)__lsx_vldi(0);
                __m128 _absmax1 = (__m128)__lsx_vldi(0);
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m128 _p0 = (__m128)__lsx_vld(p0a, 0);
                    __m128 _p1 = (__m128)__lsx_vld(p0a + 4, 0);
                    __m128 _s = __lsx_vreplfr2vr_s(*psa++);
                    _p0 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p0, _abs_mask), _s);
                    _p1 = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p1, _abs_mask), _s);
                    _absmax0 = __lsx_vfmax_s(_absmax0, _p0);
                    _absmax1 = __lsx_vfmax_s(_absmax1, _p1);
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

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
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
                    __m128 _s0 = __lsx_vreplfr2vr_s(ps[0]);
                    __m128 _s1 = __lsx_vreplfr2vr_s(ps[1]);
                    __m128 _s2 = __lsx_vreplfr2vr_s(ps[2]);
                    __m128 _s3 = __lsx_vreplfr2vr_s(ps[3]);
                    _p0 = __lsx_vfmul_s(_p0, _s0);
                    _p1 = __lsx_vfmul_s(_p1, _s1);
                    _p2 = __lsx_vfmul_s(_p2, _s2);
                    _p3 = __lsx_vfmul_s(_p3, _s3);
                    _p4 = __lsx_vfmul_s(_p4, _s0);
                    _p5 = __lsx_vfmul_s(_p5, _s1);
                    _p6 = __lsx_vfmul_s(_p6, _s2);
                    _p7 = __lsx_vfmul_s(_p7, _s3);
                    transpose4x4_ps(_p0, _p1, _p2, _p3);
                    transpose4x4_ps(_p4, _p5, _p6, _p7);
                    *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_p0, _scale0), __lsx_vfmul_s(_p1, _scale1));
                    *((int64_t*)(pp + 8)) = float2int8(__lsx_vfmul_s(_p2, _scale2), __lsx_vfmul_s(_p3, _scale3));
                    *((int64_t*)(pp + 16)) = float2int8(__lsx_vfmul_s(_p4, _scale4), __lsx_vfmul_s(_p5, _scale5));
                    *((int64_t*)(pp + 24)) = float2int8(__lsx_vfmul_s(_p6, _scale6), __lsx_vfmul_s(_p7, _scale7));
                    pp += 32;
                    p0 += (size_t)4 * A_hstep;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128 _p0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _p1 = (__m128)__lsx_vld(p0 + 4, 0);
                    __m128 _s = __lsx_vreplfr2vr_s(*ps++);
                    _p0 = __lsx_vfmul_s(_p0, _s);
                    _p1 = __lsx_vfmul_s(_p1, _s);
                    _p0 = __lsx_vfmul_s(_p0, _scales0);
                    _p1 = __lsx_vfmul_s(_p1, _scales1);
                    *((int64_t*)pp) = float2int8(_p0, _p1);
                    pp += 8;
                    p0 += A_hstep;
                }
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (size_t)(k / elempack) * A_hstep * elempack + (i + ii) * elempack;
        const float* ps = input_scale_ptr;

#if __loongarch_asx
        if (elempack == 8)
        {
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const __m256i _abs_mask = (__m256i)__lasx_xvreplgr2vr_w(0x7fffffff);
                __m256 _absmax0 = (__m256)__lasx_xvreplgr2vr_w(0);
                __m256 _absmax1 = _absmax0;
                __m256 _absmax2 = _absmax0;
                __m256 _absmax3 = _absmax0;
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk + 7 < max_kk0; kk += 8)
                {
                    __m256 _s = (__m256)__lasx_xvld(psa, 0);
                    _absmax0 = __lasx_xvfmax_s(_absmax0, __lasx_xvfmul_s((__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a, 0), _abs_mask), _s));
                    _absmax1 = __lasx_xvfmax_s(_absmax1, __lasx_xvfmul_s((__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a + 8, 0), _abs_mask), _s));
                    _absmax2 = __lasx_xvfmax_s(_absmax2, __lasx_xvfmul_s((__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a + 16, 0), _abs_mask), _s));
                    _absmax3 = __lasx_xvfmax_s(_absmax3, __lasx_xvfmul_s((__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a + 24, 0), _abs_mask), _s));
                    p0a += A_hstep * 8;
                    psa += 8;
                }
                const float absmax0 = __lasx_reduce_fmax_s(_absmax0);
                const float absmax1 = __lasx_reduce_fmax_s(_absmax1);
                const float absmax2 = __lasx_reduce_fmax_s(_absmax2);
                const float absmax3 = __lasx_reduce_fmax_s(_absmax3);
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd[2] = absmax2 / 127.f;
                pd[3] = absmax3 / 127.f;
                pd += 4;
                __m256 _scale0 = __lasx_xvreplfr2vr_s(absmax0 == 0.f ? 1.f : 127.f / absmax0);
                __m256 _scale1 = __lasx_xvreplfr2vr_s(absmax1 == 0.f ? 1.f : 127.f / absmax1);
                __m256 _scale2 = __lasx_xvreplfr2vr_s(absmax2 == 0.f ? 1.f : 127.f / absmax2);
                __m256 _scale3 = __lasx_xvreplfr2vr_s(absmax3 == 0.f ? 1.f : 127.f / absmax3);
                for (int kk = 0; kk + 7 < max_kk0; kk += 8)
                {
                    __m256 _s = (__m256)__lasx_xvld(ps, 0);
                    __m256 _p0 = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0, 0), _s), _scale0);
                    __m256 _p1 = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 8, 0), _s), _scale1);
                    __m256 _p2 = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 16, 0), _s), _scale2);
                    __m256 _p3 = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 24, 0), _s), _scale3);
                    ((int64_t*)pp)[0] = float2int8(__lasx_extract_128_lo_s(_p0), __lasx_extract_128_lo_s(_p1));
                    ((int64_t*)pp)[1] = float2int8(__lasx_extract_128_lo_s(_p2), __lasx_extract_128_lo_s(_p3));
                    ((int64_t*)pp)[2] = float2int8(__lasx_extract_128_hi_s(_p0), __lasx_extract_128_hi_s(_p1));
                    ((int64_t*)pp)[3] = float2int8(__lasx_extract_128_hi_s(_p2), __lasx_extract_128_hi_s(_p3));
                    pp += 32;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
            }
        }
#endif // __loongarch_asx

        if (elempack == 4)
        {
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                __m128 _absmax0 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax1 = _absmax0;
                __m128 _absmax2 = _absmax0;
                __m128 _absmax3 = _absmax0;
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _s = (__m128)__lsx_vld(psa, 0);
                    _absmax0 = __lsx_vfmax_s(_absmax0, __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a, 0), _abs_mask), _s));
                    _absmax1 = __lsx_vfmax_s(_absmax1, __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a + 4, 0), _abs_mask), _s));
                    _absmax2 = __lsx_vfmax_s(_absmax2, __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a + 8, 0), _abs_mask), _s));
                    _absmax3 = __lsx_vfmax_s(_absmax3, __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a + 12, 0), _abs_mask), _s));
                    p0a += A_hstep * 4;
                    psa += 4;
                }
                const float absmax0 = __lsx_reduce_fmax_s(_absmax0);
                const float absmax1 = __lsx_reduce_fmax_s(_absmax1);
                const float absmax2 = __lsx_reduce_fmax_s(_absmax2);
                const float absmax3 = __lsx_reduce_fmax_s(_absmax3);
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd[2] = absmax2 / 127.f;
                pd[3] = absmax3 / 127.f;
                pd += 4;
                __m128 _scale0 = __lsx_vreplfr2vr_s(absmax0 == 0.f ? 1.f : 127.f / absmax0);
                __m128 _scale1 = __lsx_vreplfr2vr_s(absmax1 == 0.f ? 1.f : 127.f / absmax1);
                __m128 _scale2 = __lsx_vreplfr2vr_s(absmax2 == 0.f ? 1.f : 127.f / absmax2);
                __m128 _scale3 = __lsx_vreplfr2vr_s(absmax3 == 0.f ? 1.f : 127.f / absmax3);
                for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _s = (__m128)__lsx_vld(ps, 0);
                    __m128 _p0 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _s), _scale0);
                    __m128 _p1 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0 + 4, 0), _s), _scale1);
                    __m128 _p2 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0 + 8, 0), _s), _scale2);
                    __m128 _p3 = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0 + 12, 0), _s), _scale3);
                    ((int64_t*)pp)[0] = float2int8(_p0, _p1);
                    ((int64_t*)pp)[1] = float2int8(_p2, _p3);
                    pp += 16;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
            }
        }

        if (elempack == 1)
        {
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);

                const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);
                __m128 _absmax = (__m128)__lsx_vldi(0);
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m128 _p = (__m128)__lsx_vld(p0a, 0);
                    _p = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p, _abs_mask), __lsx_vreplfr2vr_s(*psa++));
                    _absmax = __lsx_vfmax_s(_absmax, _p);
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

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const float* p1 = p0 + A_hstep;
                    const float* p2 = p1 + A_hstep;
                    const float* p3 = p2 + A_hstep;
                    __m128 _p0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _p1 = (__m128)__lsx_vld(p1, 0);
                    __m128 _p2 = (__m128)__lsx_vld(p2, 0);
                    __m128 _p3 = (__m128)__lsx_vld(p3, 0);
                    _p0 = __lsx_vfmul_s(_p0, __lsx_vreplfr2vr_s(ps[0]));
                    _p1 = __lsx_vfmul_s(_p1, __lsx_vreplfr2vr_s(ps[1]));
                    _p2 = __lsx_vfmul_s(_p2, __lsx_vreplfr2vr_s(ps[2]));
                    _p3 = __lsx_vfmul_s(_p3, __lsx_vreplfr2vr_s(ps[3]));
                    transpose4x4_ps(_p0, _p1, _p2, _p3);
                    *((int64_t*)pp) = float2int8(__lsx_vfmul_s(_p0, __lsx_vreplfr2vr_s(scales[0])), __lsx_vfmul_s(_p1, __lsx_vreplfr2vr_s(scales[1])));
                    *((int64_t*)(pp + 8)) = float2int8(__lsx_vfmul_s(_p2, __lsx_vreplfr2vr_s(scales[2])), __lsx_vfmul_s(_p3, __lsx_vreplfr2vr_s(scales[3])));
                    pp += 16;
                    p0 += (size_t)4 * A_hstep;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128 _p = (__m128)__lsx_vld(p0, 0);
                    _p = __lsx_vfmul_s(_p, __lsx_vreplfr2vr_s(*ps++));
                    const int q = __lsx_vpickve2gr_w(float2int8(__lsx_vfmul_s(_p, _scale)), 0);
                    pp[0] = (signed char)q;
                    pp[1] = (signed char)(q >> 8);
                    pp[2] = (signed char)(q >> 16);
                    pp[3] = (signed char)(q >> 24);
                    pp += 4;
                    p0 += A_hstep;
                }
            }
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (size_t)(k / elempack) * A_hstep * elempack + (i + ii) * elempack;
#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            const float* ps = input_scale_ptr;
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const __m256i _abs_mask = (__m256i)__lasx_xvreplgr2vr_w(0x7fffffff);
                __m256 _absmax0 = (__m256)__lasx_xvreplgr2vr_w(0);
                __m256 _absmax1 = _absmax0;
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk + 7 < max_kk0; kk += 8)
                {
                    __m256 _s = (__m256)__lasx_xvld(psa, 0);
                    _absmax0 = __lasx_xvfmax_s(_absmax0, __lasx_xvfmul_s((__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a, 0), _abs_mask), _s));
                    _absmax1 = __lasx_xvfmax_s(_absmax1, __lasx_xvfmul_s((__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a + 8, 0), _abs_mask), _s));
                    p0a += A_hstep * 8;
                    psa += 8;
                }
                const float absmax0 = __lasx_reduce_fmax_s(_absmax0);
                const float absmax1 = __lasx_reduce_fmax_s(_absmax1);
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd += 2;
                __m256 _scale0 = __lasx_xvreplfr2vr_s(absmax0 == 0.f ? 1.f : 127.f / absmax0);
                __m256 _scale1 = __lasx_xvreplfr2vr_s(absmax1 == 0.f ? 1.f : 127.f / absmax1);
                for (int kk = 0; kk + 7 < max_kk0; kk += 8)
                {
                    __m256 _s = (__m256)__lasx_xvld(ps, 0);
                    __m256 _p0 = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0, 0), _s), _scale0);
                    __m256 _p1 = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0 + 8, 0), _s), _scale1);
                    ((int64_t*)pp)[0] = float2int8(__lasx_extract_128_lo_s(_p0), __lasx_extract_128_lo_s(_p1));
                    ((int64_t*)pp)[1] = float2int8(__lasx_extract_128_hi_s(_p0), __lasx_extract_128_hi_s(_p1));
                    pp += 16;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
            }
        }
#endif // __loongarch_asx

        if (elempack == 4)
        {
            const float* ps = input_scale_ptr;
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                __m128 _absmax0 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax1 = _absmax0;
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _s = (__m128)__lsx_vld(psa, 0);
                    _absmax0 = __lsx_vfmax_s(_absmax0, __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a, 0), _abs_mask), _s));
                    _absmax1 = __lsx_vfmax_s(_absmax1, __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a + 4, 0), _abs_mask), _s));
                    p0a += A_hstep * 4;
                    psa += 4;
                }
                const float absmax0 = __lsx_reduce_fmax_s(_absmax0);
                const float absmax1 = __lsx_reduce_fmax_s(_absmax1);
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd += 2;
                __m128 _scale0 = __lsx_vreplfr2vr_s(absmax0 == 0.f ? 1.f : 127.f / absmax0);
                __m128 _scale1 = __lsx_vreplfr2vr_s(absmax1 == 0.f ? 1.f : 127.f / absmax1);
                for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _s = (__m128)__lsx_vld(ps, 0);
                    ((int64_t*)pp)[0] = float2int8(__lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _s), _scale0), __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0 + 4, 0), _s), _scale1));
                    pp += 8;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
            }
        }
#endif // __loongarch_sx

        if (elempack == 1)
        {
#if __loongarch_sx
            const __m128i _abs_mask = __lsx_vreplgr2vr_w(0x7fffffff);

            const float* ps = input_scale_ptr;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                __m128 _absmax = (__m128)__lsx_vldi(0);
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m128 _p = (__m128)__lsx_vldrepl_d(p0a, 0);
                    _p = __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)_p, _abs_mask), __lsx_vreplfr2vr_s(*psa++));
                    _absmax = __lsx_vfmax_s(_absmax, _p);
                    p0a += A_hstep;
                }
                float absmax[2];
                __lsx_vstelm_d((__m128i)_absmax, absmax, 0, 0);
                pd[0] = absmax[0] / 127.f;
                pd[1] = absmax[1] / 127.f;
                pd += 2;
                const float scale0 = absmax[0] == 0.f ? 0.f : 127.f / absmax[0];
                const float scale1 = absmax[1] == 0.f ? 0.f : 127.f / absmax[1];
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const float* p1 = p0 + A_hstep;
                    const float* p2 = p1 + A_hstep;
                    const float* p3 = p2 + A_hstep;
                    const float s0 = ps[0];
                    const float s1 = ps[1];
                    const float s2 = ps[2];
                    const float s3 = ps[3];
                    pp[0] = float2int8(p0[0] * s0 * scale0);
                    pp[1] = float2int8(p1[0] * s1 * scale0);
                    pp[2] = float2int8(p2[0] * s2 * scale0);
                    pp[3] = float2int8(p3[0] * s3 * scale0);
                    pp[4] = float2int8(p0[1] * s0 * scale1);
                    pp[5] = float2int8(p1[1] * s1 * scale1);
                    pp[6] = float2int8(p2[1] * s2 * scale1);
                    pp[7] = float2int8(p3[1] * s3 * scale1);
                    pp += 8;
                    p0 += (size_t)4 * A_hstep;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    const float s = *ps++;
                    pp[0] = float2int8(p0[0] * s * scale0);
                    pp[1] = float2int8(p0[1] * s * scale1);
                    pp += 2;
                    p0 += A_hstep;
                }
            }
#else
            const float* ps = input_scale_ptr;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    const float s = *psa++;
                    absmax0 = std::max(absmax0, fabsf(p0a[0]) * s);
                    absmax1 = std::max(absmax1, fabsf(p0a[1]) * s);
                    p0a += A_hstep;
                }
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                pd += 2;
                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const float* p1 = p0 + A_hstep;
                    const float* p2 = p1 + A_hstep;
                    const float* p3 = p2 + A_hstep;
                    const float s0 = ps[0];
                    const float s1 = ps[1];
                    const float s2 = ps[2];
                    const float s3 = ps[3];
                    pp[0] = float2int8(p0[0] * s0 * scale0);
                    pp[1] = float2int8(p1[0] * s1 * scale0);
                    pp[2] = float2int8(p2[0] * s2 * scale0);
                    pp[3] = float2int8(p3[0] * s3 * scale0);
                    pp[4] = float2int8(p0[1] * s0 * scale1);
                    pp[5] = float2int8(p1[1] * s1 * scale1);
                    pp[6] = float2int8(p2[1] * s2 * scale1);
                    pp[7] = float2int8(p3[1] * s3 * scale1);
                    pp += 8;
                    p0 += (size_t)4 * A_hstep;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    const float s = *ps++;
                    pp[0] = float2int8(p0[0] * s * scale0);
                    pp[1] = float2int8(p0[1] * s * scale1);
                    pp += 2;
                    p0 += A_hstep;
                }
            }
#endif // __loongarch_sx
        }
    }
    for (; ii < max_ii; ii++)
    {
        const float* p0 = (const float*)A + (size_t)(k / elempack) * A_hstep * elempack + (i + ii) * elempack;
        const float* ps = input_scale_ptr;

#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const __m256i _abs_mask = (__m256i)__lasx_xvreplgr2vr_w(0x7fffffff);
                __m256 _absmax = (__m256)__lasx_xvreplgr2vr_w(0);
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk + 7 < max_kk0; kk += 8)
                {
                    _absmax = __lasx_xvfmax_s(_absmax, __lasx_xvfmul_s((__m256)__lasx_xvand_v((__m256i)__lasx_xvld(p0a, 0), _abs_mask), (__m256)__lasx_xvld(psa, 0)));
                    p0a += A_hstep * 8;
                    psa += 8;
                }
                const float absmax = __lasx_reduce_fmax_s(_absmax);
                *pd++ = absmax / 127.f;
                __m256 _scale = __lasx_xvreplfr2vr_s(absmax == 0.f ? 1.f : 127.f / absmax);
                for (int kk = 0; kk + 7 < max_kk0; kk += 8)
                {
                    __m256 _p = __lasx_xvfmul_s(__lasx_xvfmul_s((__m256)__lasx_xvld(p0, 0), (__m256)__lasx_xvld(ps, 0)), _scale);
                    ((int*)pp)[0] = __lsx_vpickve2gr_w(float2int8(__lasx_extract_128_lo_s(_p)), 0);
                    ((int*)pp)[1] = __lsx_vpickve2gr_w(float2int8(__lasx_extract_128_hi_s(_p)), 0);
                    pp += 8;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
            }
        }
#endif // __loongarch_asx

        if (elempack == 4)
        {
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                __m128 _absmax = (__m128)__lsx_vreplgr2vr_w(0);
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                {
                    _absmax = __lsx_vfmax_s(_absmax, __lsx_vfmul_s((__m128)__lsx_vand_v((__m128i)__lsx_vld(p0a, 0), _abs_mask), (__m128)__lsx_vld(psa, 0)));
                    p0a += A_hstep * 4;
                    psa += 4;
                }
                const float absmax = __lsx_reduce_fmax_s(_absmax);
                *pd++ = absmax / 127.f;
                __m128 _scale = __lsx_vreplfr2vr_s(absmax == 0.f ? 1.f : 127.f / absmax);
                for (int kk = 0; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p = __lsx_vfmul_s(__lsx_vfmul_s((__m128)__lsx_vld(p0, 0), (__m128)__lsx_vld(ps, 0)), _scale);
                    ((int*)pp)[0] = __lsx_vpickve2gr_w(float2int8(_p), 0);
                    pp += 4;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
            }
        }
#endif // __loongarch_sx

        if (elempack == 1)
        {
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);

                float absmax = 0.f;
                const float* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v = *p0a;
                    v = fabsf(v) * *psa++;
                    absmax = std::max(absmax, v);
                    p0a += A_hstep;
                }

                if (absmax == 0.f)
                {
                    *pd++ = 0.f;
                    for (int kk = 0; kk < max_kk0; kk++)
                        *pp++ = 0;
                    p0 += (size_t)max_kk0 * A_hstep;
                    ps += max_kk0;
                    continue;
                }

                const float scale = 127.f / absmax;
                *pd++ = absmax / 127.f;

                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v = *p0;
                    v *= *ps++;
                    *pp++ = float2int8(v * scale);
                    p0 += A_hstep;
                }
            }
        }
    }
}

static void gemm_transB_packed_tile_wq_int8(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size)
{
    const signed char* pAT = AT_tile;
    const float* pAT_descales = AT_descales_tile;
    const signed char* pBT = BT_tile;
    const float* pBT_descales = BT_descales_tile;
    float* outptr = topT_tile;
    const int A_hstep = AT_tile.w;
    const int A_descales_hstep = AT_descales_tile.w;
    const int block_count = (K + block_size - 1) / block_size;
    const int block_start = k / block_size;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB = pB_panel + (size_t)8 * k;
            const float* pB_descales = pB_descales_panel + (size_t)8 * block_start;
            __m256 _fsum0;
            __m256 _fsum1;
            __m256 _fsum2;
            __m256 _fsum3;
            __m256 _fsum4;
            __m256 _fsum5;
            __m256 _fsum6;
            __m256 _fsum7;
            if (k == 0)
            {
                _fsum0 = (__m256)__lasx_xvldi(0);
                _fsum1 = (__m256)__lasx_xvldi(0);
                _fsum2 = (__m256)__lasx_xvldi(0);
                _fsum3 = (__m256)__lasx_xvldi(0);
                _fsum4 = (__m256)__lasx_xvldi(0);
                _fsum5 = (__m256)__lasx_xvldi(0);
                _fsum6 = (__m256)__lasx_xvldi(0);
                _fsum7 = (__m256)__lasx_xvldi(0);
            }
            else
            {
                _fsum0 = (__m256)__lasx_xvld(outptr, 0);
                _fsum1 = (__m256)__lasx_xvld(outptr + 8, 0);
                _fsum2 = (__m256)__lasx_xvld(outptr + 16, 0);
                _fsum3 = (__m256)__lasx_xvld(outptr + 24, 0);
                _fsum4 = (__m256)__lasx_xvld(outptr + 32, 0);
                _fsum5 = (__m256)__lasx_xvld(outptr + 40, 0);
                _fsum6 = (__m256)__lasx_xvld(outptr + 48, 0);
                _fsum7 = (__m256)__lasx_xvld(outptr + 56, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m256i _sum0 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum1 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum2 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum3 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum4 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum5 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum6 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum7 = __lasx_xvreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256i _pA = __lasx_xvld(pA, 0);
                    __m256i _pA1 = __lasx_xvshuf4i_w(_pA, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m256i _pB0 = __lasx_xvld(pB, 0);

                    __m256i _s0 = __lasx_xvmulwev_h_b(_pA, _pB0);
                    __m256i _s1 = __lasx_xvmulwev_h_b(_pA1, _pB0);
                    _s0 = __lasx_xvmaddwod_h_b(_s0, _pA, _pB0);
                    _s1 = __lasx_xvmaddwod_h_b(_s1, _pA1, _pB0);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s0, _s0));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvhaddw_w_h(_s1, _s1));
                    __m256i _pB1 = __lasx_xvshuf4i_w(_pB0, _LSX_SHUFFLE(1, 0, 3, 2));
                    _s0 = __lasx_xvmulwev_h_b(_pA, _pB1);
                    _s1 = __lasx_xvmulwev_h_b(_pA1, _pB1);
                    _s0 = __lasx_xvmaddwod_h_b(_s0, _pA, _pB1);
                    _s1 = __lasx_xvmaddwod_h_b(_s1, _pA1, _pB1);
                    _sum2 = __lasx_xvadd_w(_sum2, __lasx_xvhaddw_w_h(_s0, _s0));
                    _sum3 = __lasx_xvadd_w(_sum3, __lasx_xvhaddw_w_h(_s1, _s1));
                    __m256i _pA2 = __lasx_xvpermi_q(_pA, _pA, _LSX_SHUFFLE(0, 0, 0, 1));
                    __m256i _pA3 = __lasx_xvshuf4i_w(_pA2, _LSX_SHUFFLE(0, 3, 2, 1));
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
                for (; kk < max_kk0; kk++)
                {
                    __m256i _pA = __lasx_xvldrepl_d(pA, 0);
                    _pA = __lasx_xvilvl_b(__lasx_xvslti_b(_pA, 0), _pA);
                    __m256i _pA1 = __lasx_xvshuf4i_h(_pA, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m256i _pB0 = __lasx_xvldrepl_d(pB, 0);
                    _pB0 = __lasx_xvilvl_b(__lasx_xvslti_b(_pB0, 0), _pB0);
                    __m256i _s0 = __lasx_xvmul_h(_pA, _pB0);
                    __m256i _s1 = __lasx_xvmul_h(_pA1, _pB0);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(_s0));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_vext2xv_w_h(_s1));
                    __m256i _pB1 = __lasx_xvshuf4i_h(_pB0, _LSX_SHUFFLE(1, 0, 3, 2));
                    _s0 = __lasx_xvmul_h(_pA, _pB1);
                    _s1 = __lasx_xvmul_h(_pA1, _pB1);
                    _sum2 = __lasx_xvadd_w(_sum2, __lasx_vext2xv_w_h(_s0));
                    _sum3 = __lasx_xvadd_w(_sum3, __lasx_vext2xv_w_h(_s1));
                    __m256i _pA2 = __lasx_xvshuf4i_w(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m256i _pA3 = __lasx_xvshuf4i_h(_pA2, _LSX_SHUFFLE(0, 3, 2, 1));
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
                _fsum0 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum0), __lasx_xvfmul_s(_ascale, _bscale), _fsum0);
                _fsum1 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum1), __lasx_xvfmul_s(_ascale1, _bscale), _fsum1);
                _fsum2 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum2), __lasx_xvfmul_s(_ascale, _bscale1), _fsum2);
                _fsum3 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum3), __lasx_xvfmul_s(_ascale1, _bscale1), _fsum3);
                _fsum4 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum4), __lasx_xvfmul_s(_ascale2, _bscale), _fsum4);
                _fsum5 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum5), __lasx_xvfmul_s(_ascale3, _bscale), _fsum5);
                _fsum6 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum6), __lasx_xvfmul_s(_ascale2, _bscale1), _fsum6);
                _fsum7 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum7), __lasx_xvfmul_s(_ascale3, _bscale1), _fsum7);
                pA_descales += 8;
                pB_descales += 8;
            }
            __lasx_xvst(_fsum0, outptr, 0);
            __lasx_xvst(_fsum1, outptr + 8, 0);
            __lasx_xvst(_fsum2, outptr + 16, 0);
            __lasx_xvst(_fsum3, outptr + 24, 0);
            __lasx_xvst(_fsum4, outptr + 32, 0);
            __lasx_xvst(_fsum5, outptr + 40, 0);
            __lasx_xvst(_fsum6, outptr + 48, 0);
            __lasx_xvst(_fsum7, outptr + 56, 0);
            outptr += 64;
            pB_panel += (size_t)8 * K;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)4 * k;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            __m128 _fsum00;
            __m128 _fsum01;
            __m128 _fsum10;
            __m128 _fsum11;
            __m128 _fsum20;
            __m128 _fsum21;
            __m128 _fsum30;
            __m128 _fsum31;
            if (k == 0)
            {
                _fsum00 = (__m128)__lsx_vldi(0);
                _fsum01 = (__m128)__lsx_vldi(0);
                _fsum10 = (__m128)__lsx_vldi(0);
                _fsum11 = (__m128)__lsx_vldi(0);
                _fsum20 = (__m128)__lsx_vldi(0);
                _fsum21 = (__m128)__lsx_vldi(0);
                _fsum30 = (__m128)__lsx_vldi(0);
                _fsum31 = (__m128)__lsx_vldi(0);
            }
            else
            {
                _fsum00 = (__m128)__lsx_vld(outptr, 0);
                _fsum01 = (__m128)__lsx_vld(outptr + 4, 0);
                _fsum10 = (__m128)__lsx_vld(outptr + 8, 0);
                _fsum11 = (__m128)__lsx_vld(outptr + 12, 0);
                _fsum20 = (__m128)__lsx_vld(outptr + 16, 0);
                _fsum21 = (__m128)__lsx_vld(outptr + 20, 0);
                _fsum30 = (__m128)__lsx_vld(outptr + 24, 0);
                _fsum31 = (__m128)__lsx_vld(outptr + 28, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum00 = __lsx_vreplgr2vr_w(0);
                __m128i _sum01 = __lsx_vreplgr2vr_w(0);
                __m128i _sum10 = __lsx_vreplgr2vr_w(0);
                __m128i _sum11 = __lsx_vreplgr2vr_w(0);
                __m128i _sum20 = __lsx_vreplgr2vr_w(0);
                __m128i _sum21 = __lsx_vreplgr2vr_w(0);
                __m128i _sum30 = __lsx_vreplgr2vr_w(0);
                __m128i _sum31 = __lsx_vreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA0 = __lsx_vld(pA, 0);
                    __m128i _pA1 = __lsx_vld(pA + 16, 0);
                    __m128i _pB0 = __lsx_vld(pB, 0);

                    __m128i _s = __lsx_vmulwev_h_b(_pA0, _pB0);
                    _s = __lsx_vmaddwod_h_b(_s, _pA0, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmulwev_h_b(_pA1, _pB0);
                    _s = __lsx_vmaddwod_h_b(_s, _pA1, _pB0);
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vhaddw_w_h(_s, _s));

                    __m128i _pB0r = __lsx_vshuf4i_w(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                    _s = __lsx_vmulwev_h_b(_pA0, _pB0r);
                    _s = __lsx_vmaddwod_h_b(_s, _pA0, _pB0r);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmulwev_h_b(_pA1, _pB0r);
                    _s = __lsx_vmaddwod_h_b(_s, _pA1, _pB0r);
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vhaddw_w_h(_s, _s));

                    __m128i _pA0r = __lsx_vshuf4i_w(_pA0, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m128i _pA1r = __lsx_vshuf4i_w(_pA1, _LSX_SHUFFLE(1, 0, 3, 2));
                    _s = __lsx_vmulwev_h_b(_pA0r, _pB0);
                    _s = __lsx_vmaddwod_h_b(_s, _pA0r, _pB0);
                    _sum20 = __lsx_vadd_w(_sum20, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmulwev_h_b(_pA1r, _pB0);
                    _s = __lsx_vmaddwod_h_b(_s, _pA1r, _pB0);
                    _sum21 = __lsx_vadd_w(_sum21, __lsx_vhaddw_w_h(_s, _s));

                    _s = __lsx_vmulwev_h_b(_pA0r, _pB0r);
                    _s = __lsx_vmaddwod_h_b(_s, _pA0r, _pB0r);
                    _sum30 = __lsx_vadd_w(_sum30, __lsx_vhaddw_w_h(_s, _s));
                    _s = __lsx_vmulwev_h_b(_pA1r, _pB0r);
                    _s = __lsx_vmaddwod_h_b(_s, _pA1r, _pB0r);
                    _sum31 = __lsx_vadd_w(_sum31, __lsx_vhaddw_w_h(_s, _s));
                    pB += 16;
                    pA += 32;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA0 = __lsx_vldrepl_w(pA, 0);
                    _pA0 = __lsx_vilvl_b(__lsx_vslti_b(_pA0, 0), _pA0);
                    __m128i _pA1 = __lsx_vldrepl_w(pA + 4, 0);
                    _pA1 = __lsx_vilvl_b(__lsx_vslti_b(_pA1, 0), _pA1);
                    __m128i _pB0 = __lsx_vldrepl_w(pB, 0);
                    _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                    __m128i _s = __lsx_vmul_h(_pA0, _pB0);
                    _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA1, _pB0);
                    _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    __m128i _pB0r = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                    _s = __lsx_vmul_h(_pA0, _pB0r);
                    _sum10 = __lsx_vadd_w(_sum10, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA1, _pB0r);
                    _sum11 = __lsx_vadd_w(_sum11, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    __m128i _pA0r = __lsx_vshuf4i_h(_pA0, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m128i _pA1r = __lsx_vshuf4i_h(_pA1, _LSX_SHUFFLE(1, 0, 3, 2));
                    _s = __lsx_vmul_h(_pA0r, _pB0);
                    _sum20 = __lsx_vadd_w(_sum20, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA1r, _pB0);
                    _sum21 = __lsx_vadd_w(_sum21, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA0r, _pB0r);
                    _sum30 = __lsx_vadd_w(_sum30, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    _s = __lsx_vmul_h(_pA1r, _pB0r);
                    _sum31 = __lsx_vadd_w(_sum31, __lsx_vilvl_h(__lsx_vslti_h(_s, 0), _s));
                    pB += 4;
                    pA += 8;
                }

                __m128 _bscale = (__m128)__lsx_vld(pB_descales, 0);
                __m128 _bscaler = (__m128)__lsx_vshuf4i_w((__m128i)_bscale, _LSX_SHUFFLE(0, 3, 2, 1));
                __m128 _ascale0 = (__m128)__lsx_vld(pA_descales, 0);
                __m128 _ascale1 = (__m128)__lsx_vld(pA_descales + 4, 0);
                __m128 _ascale0r = (__m128)__lsx_vshuf4i_w((__m128i)_ascale0, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _ascale1r = (__m128)__lsx_vshuf4i_w((__m128i)_ascale1, _LSX_SHUFFLE(1, 0, 3, 2));
                _fsum00 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum00), __lsx_vfmul_s(_ascale0, _bscale), _fsum00);
                _fsum01 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum01), __lsx_vfmul_s(_ascale1, _bscale), _fsum01);
                _fsum10 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum10), __lsx_vfmul_s(_ascale0, _bscaler), _fsum10);
                _fsum11 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum11), __lsx_vfmul_s(_ascale1, _bscaler), _fsum11);
                _fsum20 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum20), __lsx_vfmul_s(_ascale0r, _bscale), _fsum20);
                _fsum21 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum21), __lsx_vfmul_s(_ascale1r, _bscale), _fsum21);
                _fsum30 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum30), __lsx_vfmul_s(_ascale0r, _bscaler), _fsum30);
                _fsum31 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum31), __lsx_vfmul_s(_ascale1r, _bscaler), _fsum31);
                pA_descales += 8;
                pB_descales += 4;
            }
            __lsx_vst((__m128i)_fsum00, outptr, 0);
            __lsx_vst((__m128i)_fsum01, outptr + 4, 0);
            __lsx_vst((__m128i)_fsum10, outptr + 8, 0);
            __lsx_vst((__m128i)_fsum11, outptr + 12, 0);
            __lsx_vst((__m128i)_fsum20, outptr + 16, 0);
            __lsx_vst((__m128i)_fsum21, outptr + 20, 0);
            __lsx_vst((__m128i)_fsum30, outptr + 24, 0);
            __lsx_vst((__m128i)_fsum31, outptr + 28, 0);
            outptr += 32;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }

        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            __m128 _fsum00;
            __m128 _fsum01;
            __m128 _fsum10;
            __m128 _fsum11;
            if (k == 0)
            {
                _fsum00 = (__m128)__lsx_vldi(0);
                _fsum01 = (__m128)__lsx_vldi(0);
                _fsum10 = (__m128)__lsx_vldi(0);
                _fsum11 = (__m128)__lsx_vldi(0);
            }
            else
            {
                _fsum00 = (__m128)__lsx_vld(outptr, 0);
                _fsum01 = (__m128)__lsx_vld(outptr + 4, 0);
                _fsum10 = (__m128)__lsx_vld(outptr + 8, 0);
                _fsum11 = (__m128)__lsx_vld(outptr + 12, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum00 = __lsx_vreplgr2vr_w(0);
                __m128i _sum01 = __lsx_vreplgr2vr_w(0);
                __m128i _sum10 = __lsx_vreplgr2vr_w(0);
                __m128i _sum11 = __lsx_vreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
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
                for (; kk < max_kk0; kk++)
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
                _fsum00 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum00), __lsx_vfmul_s(_ascale0, _bscale), _fsum00);
                _fsum01 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum01), __lsx_vfmul_s(_ascale1, _bscale), _fsum01);
                _bscale = __lsx_vreplfr2vr_s(pB_descales[1]);
                _fsum10 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum10), __lsx_vfmul_s(_ascale0, _bscale), _fsum10);
                _fsum11 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum11), __lsx_vfmul_s(_ascale1, _bscale), _fsum11);
                pA_descales += 8;
                pB_descales += 2;
            }
            __lsx_vst((__m128i)_fsum00, outptr, 0);
            __lsx_vst((__m128i)_fsum01, outptr + 4, 0);
            __lsx_vst((__m128i)_fsum10, outptr + 8, 0);
            __lsx_vst((__m128i)_fsum11, outptr + 12, 0);
            outptr += 16;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + k;
            const float* pB_descales = pB_descales_panel + block_start;
            __m128 _fsum0;
            __m128 _fsum1;
            if (k == 0)
            {
                _fsum0 = (__m128)__lsx_vldi(0);
                _fsum1 = (__m128)__lsx_vldi(0);
            }
            else
            {
                _fsum0 = (__m128)__lsx_vld(outptr, 0);
                _fsum1 = (__m128)__lsx_vld(outptr + 4, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                __m128i _sum1 = __lsx_vreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
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
                for (; kk < max_kk0; kk++)
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
                _fsum0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum0), __lsx_vfmul_s((__m128)__lsx_vld(pA_descales, 0), _bscale), _fsum0);
                _fsum1 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum1), __lsx_vfmul_s((__m128)__lsx_vld(pA_descales + 4, 0), _bscale), _fsum1);
                pA_descales += 8;
            }
            __lsx_vst((__m128i)_fsum0, outptr, 0);
            __lsx_vst((__m128i)_fsum1, outptr + 4, 0);
            outptr += 8;
            pB_panel += K;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep * 8;
        pAT_descales += A_descales_hstep * 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB = pB_panel + (size_t)8 * k;
            const float* pB_descales = pB_descales_panel + (size_t)8 * block_start;
            __m256 _fsum0;
            __m256 _fsum1;
            __m256 _fsum2;
            __m256 _fsum3;
            if (k == 0)
            {
                _fsum0 = (__m256)__lasx_xvldi(0);
                _fsum1 = (__m256)__lasx_xvldi(0);
                _fsum2 = (__m256)__lasx_xvldi(0);
                _fsum3 = (__m256)__lasx_xvldi(0);
            }
            else
            {
                _fsum0 = (__m256)__lasx_xvld(outptr, 0);
                _fsum1 = (__m256)__lasx_xvld(outptr + 8, 0);
                _fsum2 = (__m256)__lasx_xvld(outptr + 16, 0);
                _fsum3 = (__m256)__lasx_xvld(outptr + 24, 0);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m256i _sum0 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum1 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum2 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum3 = __lasx_xvreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
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
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA4 = __lsx_vldrepl_w(pA, 0);
                    _pA4 = __lsx_vilvl_b(__lsx_vslti_b(_pA4, 0), _pA4);
                    __m128i _pB04 = __lsx_vldrepl_w(pB, 0);
                    _pB04 = __lsx_vilvl_b(__lsx_vslti_b(_pB04, 0), _pB04);
                    __m128i _pB48 = __lsx_vldrepl_w(pB + 4, 0);
                    _pB48 = __lsx_vilvl_b(__lsx_vslti_b(_pB48, 0), _pB48);
                    __m128i _pA4r = __lsx_vshuf4i_h(_pA4, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m128i _pB04r = __lsx_vshuf4i_h(_pB04, _LSX_SHUFFLE(0, 3, 2, 1));
                    __m128i _pB48r = __lsx_vshuf4i_h(_pB48, _LSX_SHUFFLE(0, 3, 2, 1));

                    __m128i _s0 = __lsx_vmul_h(_pA4, _pB04);
                    __m128i _s1 = __lsx_vmul_h(_pA4, _pB04r);
                    __m128i _s2 = __lsx_vmul_h(_pA4r, _pB04);
                    __m128i _s3 = __lsx_vmul_h(_pA4r, _pB04r);
                    __m128i _s4 = __lsx_vmul_h(_pA4, _pB48);
                    __m128i _s5 = __lsx_vmul_h(_pA4, _pB48r);
                    __m128i _s6 = __lsx_vmul_h(_pA4r, _pB48);
                    __m128i _s7 = __lsx_vmul_h(_pA4r, _pB48r);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_concat_128(__lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0), __lsx_vilvl_h(__lsx_vslti_h(_s4, 0), _s4)));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_concat_128(__lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1), __lsx_vilvl_h(__lsx_vslti_h(_s5, 0), _s5)));
                    _sum2 = __lasx_xvadd_w(_sum2, __lasx_concat_128(__lsx_vilvl_h(__lsx_vslti_h(_s2, 0), _s2), __lsx_vilvl_h(__lsx_vslti_h(_s6, 0), _s6)));
                    _sum3 = __lasx_xvadd_w(_sum3, __lasx_concat_128(__lsx_vilvl_h(__lsx_vslti_h(_s3, 0), _s3), __lsx_vilvl_h(__lsx_vslti_h(_s7, 0), _s7)));
                    pB += 8;
                    pA += 4;
                }

                __m256 _bscale = (__m256)__lasx_xvld(pB_descales, 0);
                __m128i _ascale4 = __lsx_vld(pA_descales, 0);
                __m256 _ascale = (__m256)__lasx_concat_128(_ascale4, _ascale4);
                __m256 _ascale1 = (__m256)__lasx_xvshuf4i_w((__m256i)_ascale, _LSX_SHUFFLE(1, 0, 3, 2));
                __m256 _bscaler = (__m256)__lasx_xvshuf4i_w((__m256i)_bscale, _LSX_SHUFFLE(0, 3, 2, 1));
                _fsum0 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum0), __lasx_xvfmul_s(_bscale, _ascale), _fsum0);
                _fsum1 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum1), __lasx_xvfmul_s(_bscaler, _ascale), _fsum1);
                _fsum2 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum2), __lasx_xvfmul_s(_bscale, _ascale1), _fsum2);
                _fsum3 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum3), __lasx_xvfmul_s(_bscaler, _ascale1), _fsum3);
                pA_descales += 4;
                pB_descales += 8;
            }

            __lasx_xvst(_fsum0, outptr, 0);
            __lasx_xvst(_fsum1, outptr + 8, 0);
            __lasx_xvst(_fsum2, outptr + 16, 0);
            __lasx_xvst(_fsum3, outptr + 24, 0);
            outptr += 32;
            pB_panel += (size_t)8 * K;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)4 * k;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            __m128 _fsum0;
            __m128 _fsum1;
            __m128 _fsum2;
            __m128 _fsum3;
            if (k == 0)
            {
                _fsum0 = (__m128)__lsx_vldi(0);
                _fsum1 = (__m128)__lsx_vldi(0);
                _fsum2 = (__m128)__lsx_vldi(0);
                _fsum3 = (__m128)__lsx_vldi(0);
            }
            else
            {
                _fsum0 = (__m128)__lsx_vld(outptr, 0);
                _fsum1 = (__m128)__lsx_vld(outptr + 4, 0);
                _fsum2 = (__m128)__lsx_vld(outptr + 8, 0);
                _fsum3 = (__m128)__lsx_vld(outptr + 12, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                __m128i _sum1 = __lsx_vreplgr2vr_w(0);
                __m128i _sum2 = __lsx_vreplgr2vr_w(0);
                __m128i _sum3 = __lsx_vreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
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
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA = __lsx_vldrepl_w(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _pA1 = __lsx_vshuf4i_h(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                    __m128i _pB = __lsx_vldrepl_w(pB, 0);
                    _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);
                    __m128i _pB1 = __lsx_vshuf4i_h(_pB, _LSX_SHUFFLE(0, 3, 2, 1));

                    __m128i _s0 = __lsx_vmul_h(_pA, _pB);
                    __m128i _s1 = __lsx_vmul_h(_pA, _pB1);
                    __m128i _s2 = __lsx_vmul_h(_pA1, _pB);
                    __m128i _s3 = __lsx_vmul_h(_pA1, _pB1);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                    _sum2 = __lsx_vadd_w(_sum2, __lsx_vilvl_h(__lsx_vslti_h(_s2, 0), _s2));
                    _sum3 = __lsx_vadd_w(_sum3, __lsx_vilvl_h(__lsx_vslti_h(_s3, 0), _s3));
                    pB += 4;
                    pA += 4;
                }
                __m128 _bscale = (__m128)__lsx_vld(pB_descales, 0);
                __m128 _ascale = (__m128)__lsx_vld(pA_descales, 0);
                __m128 _ascale1 = (__m128)__lsx_vshuf4i_w((__m128i)_ascale, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _bscale1 = (__m128)__lsx_vshuf4i_w((__m128i)_bscale, _LSX_SHUFFLE(0, 3, 2, 1));
                _fsum0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum0), __lsx_vfmul_s(_bscale, _ascale), _fsum0);
                _fsum1 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum1), __lsx_vfmul_s(_bscale1, _ascale), _fsum1);
                _fsum2 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum2), __lsx_vfmul_s(_bscale, _ascale1), _fsum2);
                _fsum3 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum3), __lsx_vfmul_s(_bscale1, _ascale1), _fsum3);
                pA_descales += 4;
                pB_descales += 4;
            }
            __lsx_vst((__m128i)_fsum0, outptr, 0);
            __lsx_vst((__m128i)_fsum1, outptr + 4, 0);
            __lsx_vst((__m128i)_fsum2, outptr + 8, 0);
            __lsx_vst((__m128i)_fsum3, outptr + 12, 0);
            outptr += 16;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            __m128 _fsum0;
            __m128 _fsum1;
            if (k == 0)
            {
                _fsum0 = (__m128)__lsx_vldi(0);
                _fsum1 = (__m128)__lsx_vldi(0);
            }
            else
            {
                _fsum0 = (__m128)__lsx_vld(outptr, 0);
                _fsum1 = (__m128)__lsx_vld(outptr + 4, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                __m128i _sum1 = __lsx_vreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
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
                for (; kk < max_kk0; kk++)
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
                __m128 _ascale = (__m128)__lsx_vld(pA_descales, 0);
                __m128 _bscale0 = (__m128)__lsx_vldrepl_d(pB_descales, 0);
                __m128 _bscale1 = (__m128)__lsx_vshuf4i_w((__m128i)_bscale0, _LSX_SHUFFLE(2, 3, 0, 1));
                _fsum0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum0), __lsx_vfmul_s(_ascale, _bscale0), _fsum0);
                _fsum1 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum1), __lsx_vfmul_s(_ascale, _bscale1), _fsum1);
                pA_descales += 4;
                pB_descales += 2;
            }
            __lsx_vst((__m128i)_fsum0, outptr, 0);
            __lsx_vst((__m128i)_fsum1, outptr + 4, 0);
            outptr += 8;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + k;
            const float* pB_descales = pB_descales_panel + block_start;
            __m128 _fsum0;
            if (k == 0)
            {
                _fsum0 = (__m128)__lsx_vldi(0);
            }
            else
            {
                _fsum0 = (__m128)__lsx_vld(outptr, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pA = __lsx_vld(pA, 0);
                    __m128i _pB = __lsx_vldrepl_w(pB, 0);
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA, _pB), _pA, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                    pA += 16;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pA = __lsx_vldrepl_w(pA, 0);
                    _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                    __m128i _s0 = __lsx_vmul_h(_pA, __lsx_vreplgr2vr_h(pB[0]));
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    pA += 4;
                    pB++;
                }
                __m128 _scale = __lsx_vfmul_s((__m128)__lsx_vld(pA_descales, 0), __lsx_vreplfr2vr_s(*pB_descales++));
                _fsum0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum0), _scale, _fsum0);
                pA_descales += 4;
            }
            __lsx_vst((__m128i)_fsum0, outptr, 0);
            outptr += 4;
            pB_panel += K;
            pB_descales_panel += block_count;
        }
        pAT += A_hstep * 4;
        pAT_descales += A_descales_hstep * 4;
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB = pB_panel + (size_t)8 * k;
            const float* pB_descales = pB_descales_panel + (size_t)8 * block_start;
            __m256 _fsum0;
            __m256 _fsum1;
            if (k == 0)
            {
                _fsum0 = (__m256)__lasx_xvldi(0);
                _fsum1 = (__m256)__lasx_xvldi(0);
            }
            else
            {
                _fsum0 = (__m256)__lasx_xvld(outptr, 0);
                _fsum1 = (__m256)__lasx_xvld(outptr + 8, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m256i _sum0 = __lasx_xvreplgr2vr_w(0);
                __m256i _sum1 = __lasx_xvreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
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
                for (; kk < max_kk0; kk++)
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
                _fsum0 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum0), __lasx_xvfmul_s(_bscale, (__m256)__lasx_xvreplfr2vr_s(pA_descales[0])), _fsum0);
                _fsum1 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum1), __lasx_xvfmul_s(_bscale, (__m256)__lasx_xvreplfr2vr_s(pA_descales[1])), _fsum1);
                pA_descales += 2;
                pB_descales += 8;
            }
            __lasx_xvst(_fsum0, outptr, 0);
            __lasx_xvst(_fsum1, outptr + 8, 0);
            outptr += 16;
            pB_panel += (size_t)8 * K;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)4 * k;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            __m128 _fsum0;
            __m128 _fsum1;
            if (k == 0)
            {
                _fsum0 = (__m128)__lsx_vldi(0);
                _fsum1 = (__m128)__lsx_vldi(0);
            }
            else
            {
                _fsum0 = (__m128)__lsx_vld(outptr, 0);
                _fsum1 = (__m128)__lsx_vld(outptr + 4, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                __m128i _sum1 = __lsx_vreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
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
                for (; kk < max_kk0; kk++)
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
                _fsum0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum0), __lsx_vfmul_s(_bscale, __lsx_vreplfr2vr_s(pA_descales[0])), _fsum0);
                _fsum1 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum1), __lsx_vfmul_s(_bscale, __lsx_vreplfr2vr_s(pA_descales[1])), _fsum1);
                pA_descales += 2;
                pB_descales += 4;
            }
            __lsx_vst((__m128i)_fsum0, outptr, 0);
            __lsx_vst((__m128i)_fsum1, outptr + 4, 0);
            outptr += 8;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            float out00;
            float out01;
            float out10;
            float out11;
            if (k == 0)
            {
                out00 = 0.f;
                out01 = 0.f;
                out10 = 0.f;
                out11 = 0.f;
            }
            else
            {
                out00 = outptr[0];
                out01 = outptr[1];
                out10 = outptr[2];
                out11 = outptr[3];
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int sum00 = 0;
                int sum01 = 0;
                int sum10 = 0;
                int sum11 = 0;
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    sum00 += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    sum01 += pA[0] * pB[4] + pA[1] * pB[5] + pA[2] * pB[6] + pA[3] * pB[7];
                    sum10 += pA[4] * pB[0] + pA[5] * pB[1] + pA[6] * pB[2] + pA[7] * pB[3];
                    sum11 += pA[4] * pB[4] + pA[5] * pB[5] + pA[6] * pB[6] + pA[7] * pB[7];
                    pB += 8;
                    pA += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    sum00 += pA[0] * pB[0];
                    sum01 += pA[0] * pB[1];
                    sum10 += pA[1] * pB[0];
                    sum11 += pA[1] * pB[1];
                    pB += 2;
                    pA += 2;
                }
                const float bscale0 = pB_descales[0];
                const float bscale1 = pB_descales[1];
                const float ascale0 = pA_descales[0];
                const float ascale1 = pA_descales[1];
                out00 += sum00 * ascale0 * bscale0;
                out01 += sum01 * ascale0 * bscale1;
                out10 += sum10 * ascale1 * bscale0;
                out11 += sum11 * ascale1 * bscale1;
                pA_descales += 2;
                pB_descales += 2;
            }
            outptr[0] = out00;
            outptr[1] = out01;
            outptr[2] = out10;
            outptr[3] = out11;
            outptr += 4;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + k;
            const float* pB_descales = pB_descales_panel + block_start;
            float out0;
            float out1;
            if (k == 0)
            {
                out0 = 0.f;
                out1 = 0.f;
            }
            else
            {
                out0 = outptr[0];
                out1 = outptr[1];
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int sum0 = 0;
                int sum1 = 0;
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    sum0 += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    sum1 += pA[4] * pB[0] + pA[5] * pB[1] + pA[6] * pB[2] + pA[7] * pB[3];
                    pB += 4;
                    pA += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    sum0 += pA[0] * pB[0];
                    sum1 += pA[1] * pB[0];
                    pB++;
                    pA += 2;
                }
                const float bscale = *pB_descales++;
                out0 += sum0 * pA_descales[0] * bscale;
                out1 += sum1 * pA_descales[1] * bscale;
                pA_descales += 2;
            }
            outptr[0] = out0;
            outptr[1] = out1;
            outptr += 2;
            pB_panel += K;
            pB_descales_panel += block_count;
        }
        pAT += A_hstep * 2;
        pAT_descales += A_descales_hstep * 2;
    }
    for (; ii < max_ii; ii++)
    {
        const signed char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB = pB_panel + (size_t)8 * k;
            const float* pB_descales = pB_descales_panel + (size_t)8 * block_start;
            __m256 _fsum0;
            if (k == 0)
            {
                _fsum0 = (__m256)__lasx_xvldi(0);
            }
            else
            {
                _fsum0 = (__m256)__lasx_xvld(outptr, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m256i _sum0 = __lasx_xvreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256i _pB = __lasx_xvld(pB, 0);
                    __m256i _pA0 = __lasx_xvldrepl_w(pA, 0);
                    __m256i _s0 = __lasx_xvmaddwod_h_b(__lasx_xvmulwev_h_b(_pA0, _pB), _pA0, _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s0, _s0));
                    pB += 32;
                    pA += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pB = __lsx_vldrepl_d(pB, 0);
                    _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);
                    __m128i _s0 = __lsx_vmul_h(__lsx_vreplgr2vr_h(pA[0]), _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(__lasx_cast_128(_s0)));
                    pB += 8;
                    pA++;
                }
                __m256 _bscale = (__m256)__lasx_xvld(pB_descales, 0);
                _fsum0 = __lasx_xvfmadd_s((__m256)__lasx_xvffint_s_w(_sum0), __lasx_xvfmul_s(_bscale, (__m256)__lasx_xvreplfr2vr_s(*pA_descales++)), _fsum0);
                pB_descales += 8;
            }
            __lasx_xvst(_fsum0, outptr, 0);
            outptr += 8;
            pB_panel += (size_t)8 * K;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pB = pB_panel + (size_t)4 * k;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            __m128 _fsum0;
            if (k == 0)
            {
                _fsum0 = (__m128)__lsx_vldi(0);
            }
            else
            {
                _fsum0 = (__m128)__lsx_vld(outptr, 0);
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128i _pB = __lsx_vld(pB, 0);
                    __m128i _pA0 = __lsx_vldrepl_w(pA, 0);
                    __m128i _s0 = __lsx_vmaddwod_h_b(__lsx_vmulwev_h_b(_pA0, _pB), _pA0, _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                    pB += 16;
                    pA += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128i _pB = __lsx_vldrepl_w(pB, 0);
                    _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);
                    __m128i _s0 = __lsx_vmul_h(__lsx_vreplgr2vr_h(pA[0]), _pB);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                    pB += 4;
                    pA++;
                }
                __m128 _bscale = (__m128)__lsx_vld(pB_descales, 0);
                _fsum0 = __lsx_vfmadd_s((__m128)__lsx_vffint_s_w(_sum0), __lsx_vfmul_s(_bscale, __lsx_vreplfr2vr_s(*pA_descales++)), _fsum0);
                pB_descales += 4;
            }
            __lsx_vst((__m128i)_fsum0, outptr, 0);
            outptr += 4;
            pB_panel += (size_t)4 * K;
            pB_descales_panel += (size_t)4 * block_count;
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pB = pB_panel + (size_t)2 * k;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            float out0;
            float out1;
            if (k == 0)
            {
                out0 = 0.f;
                out1 = 0.f;
            }
            else
            {
                out0 = outptr[0];
                out1 = outptr[1];
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int sum0 = 0;
                int sum1 = 0;
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    sum0 += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    sum1 += pA[0] * pB[4] + pA[1] * pB[5] + pA[2] * pB[6] + pA[3] * pB[7];
                    pB += 8;
                    pA += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    sum0 += pA[0] * pB[0];
                    sum1 += pA[0] * pB[1];
                    pB += 2;
                    pA++;
                }
                const float ascale = *pA_descales++;
                out0 += sum0 * ascale * pB_descales[0];
                out1 += sum1 * ascale * pB_descales[1];
                pB_descales += 2;
            }
            outptr[0] = out0;
            outptr[1] = out1;
            outptr += 2;
            pB_panel += (size_t)2 * K;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pB = pB_panel + k;
            const float* pB_descales = pB_descales_panel + block_start;
            float out0;
            if (k == 0)
            {
                out0 = 0.f;
            }
            else
            {
                out0 = outptr[0];
            }
            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                int sum0 = 0;
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    sum0 += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                    pB += 4;
                    pA += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    sum0 += pA[0] * pB[0];
                    pB++;
                    pA++;
                }
                out0 += sum0 * *pA_descales++ * *pB_descales++;
            }
            *outptr++ = out0;
            pB_panel += K;
            pB_descales_panel += block_count;
        }
        pAT += A_hstep;
        pAT_descales += A_descales_hstep;
    }
}

static void unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta, int output_elemtype)
{
    const float* pp = topT;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const int c_elempack = C.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    const int out_elempack = top_blob.elempack;
    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* p0f = output_elemtype == 1 ? (float*)top_blob + (size_t)(i + ii) * out_hstep + j * out_elempack : 0;
        unsigned short* p0 = output_elemtype == 3 ? (unsigned short*)top_blob + (size_t)(i + ii) * out_hstep + j * out_elempack : 0;

        float c0 = 0.f;
        float c1 = 0.f;
        float c2 = 0.f;
        float c3 = 0.f;
        float c4 = 0.f;
        float c5 = 0.f;
        float c6 = 0.f;
        float c7 = 0.f;
        const float* pC = C;
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
                pC += i + ii;
                c0 = pC[0] * beta;
                c1 = pC[1] * beta;
                c2 = pC[2] * beta;
                c3 = pC[3] * beta;
                c4 = pC[4] * beta;
                c5 = pC[5] * beta;
                c6 = pC[6] * beta;
                c7 = pC[7] * beta;
            }
            if (broadcast_type_C == 3)
            {
                pC += (size_t)(i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC += j;
            }
        }

        int jj = 0;
#if __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const float* pC0 = pC;
            __m256i _r0 = __lasx_xvld(pp, 0);
            __m256i _r1 = __lasx_xvld(pp + 8, 0);
            __m256i _r2 = __lasx_xvld(pp + 16, 0);
            __m256i _r3 = __lasx_xvld(pp + 24, 0);
            __m256i _r4 = __lasx_xvld(pp + 32, 0);
            __m256i _r5 = __lasx_xvld(pp + 40, 0);
            __m256i _r6 = __lasx_xvld(pp + 48, 0);
            __m256i _r7 = __lasx_xvld(pp + 56, 0);
            __m256i _tmp0 = _r0;
            __m256i _tmp1 = __lasx_xvshuf4i_w(_r1, _LSX_SHUFFLE(2, 1, 0, 3));
            __m256i _tmp2 = _r2;
            __m256i _tmp3 = __lasx_xvshuf4i_w(_r3, _LSX_SHUFFLE(2, 1, 0, 3));
            __m256i _tmp4 = _r4;
            __m256i _tmp5 = __lasx_xvshuf4i_w(_r5, _LSX_SHUFFLE(2, 1, 0, 3));
            __m256i _tmp6 = _r6;
            __m256i _tmp7 = __lasx_xvshuf4i_w(_r7, _LSX_SHUFFLE(2, 1, 0, 3));
            _r0 = __lasx_xvilvl_w(_tmp3, _tmp0);
            _r1 = __lasx_xvilvh_w(_tmp3, _tmp0);
            _r2 = __lasx_xvilvl_w(_tmp1, _tmp2);
            _r3 = __lasx_xvilvh_w(_tmp1, _tmp2);
            _r4 = __lasx_xvilvl_w(_tmp7, _tmp4);
            _r5 = __lasx_xvilvh_w(_tmp7, _tmp4);
            _r6 = __lasx_xvilvl_w(_tmp5, _tmp6);
            _r7 = __lasx_xvilvh_w(_tmp5, _tmp6);
            _tmp0 = __lasx_xvilvl_d(_r2, _r0);
            _tmp1 = __lasx_xvilvh_d(_r2, _r0);
            _tmp2 = __lasx_xvilvl_d(_r1, _r3);
            _tmp3 = __lasx_xvilvh_d(_r1, _r3);
            _tmp4 = __lasx_xvilvl_d(_r6, _r4);
            _tmp5 = __lasx_xvilvh_d(_r6, _r4);
            _tmp6 = __lasx_xvilvl_d(_r5, _r7);
            _tmp7 = __lasx_xvilvh_d(_r5, _r7);
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
            if (pC0)
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
                    __m256 _c0;
                    __m256 _c1;
                    __m256 _c2;
                    __m256 _c3;
                    __m256 _c4;
                    __m256 _c5;
                    __m256 _c6;
                    __m256 _c7;
                    if (c_elempack == 8)
                    {
                        _c0 = (__m256)__lasx_xvld(pC0, 0);
                        _c1 = (__m256)__lasx_xvld(pC0 + 8, 0);
                        _c2 = (__m256)__lasx_xvld(pC0 + 16, 0);
                        _c3 = (__m256)__lasx_xvld(pC0 + 24, 0);
                        _c4 = (__m256)__lasx_xvld(pC0 + 32, 0);
                        _c5 = (__m256)__lasx_xvld(pC0 + 40, 0);
                        _c6 = (__m256)__lasx_xvld(pC0 + 48, 0);
                        _c7 = (__m256)__lasx_xvld(pC0 + 56, 0);
                        transpose8x8_ps(_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7);
                    }
                    else if (c_elempack == 4)
                    {
                        const float* pC1 = pC0 + c_hstep * 4;
                        _c0 = __lasx_concat_128_s((__m128)__lsx_vld(pC0, 0), (__m128)__lsx_vld(pC1, 0));
                        _c1 = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 4, 0), (__m128)__lsx_vld(pC1 + 4, 0));
                        _c2 = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 8, 0), (__m128)__lsx_vld(pC1 + 8, 0));
                        _c3 = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 12, 0), (__m128)__lsx_vld(pC1 + 12, 0));
                        _c4 = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 16, 0), (__m128)__lsx_vld(pC1 + 16, 0));
                        _c5 = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 20, 0), (__m128)__lsx_vld(pC1 + 20, 0));
                        _c6 = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 24, 0), (__m128)__lsx_vld(pC1 + 24, 0));
                        _c7 = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 28, 0), (__m128)__lsx_vld(pC1 + 28, 0));
                        transpose8x8_ps(_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7);
                    }
                    else
                    {
                        _c0 = (__m256)__lasx_xvld(pC0, 0);
                        _c1 = (__m256)__lasx_xvld(pC0 + c_hstep, 0);
                        _c2 = (__m256)__lasx_xvld(pC0 + c_hstep * 2, 0);
                        _c3 = (__m256)__lasx_xvld(pC0 + c_hstep * 3, 0);
                        _c4 = (__m256)__lasx_xvld(pC0 + c_hstep * 4, 0);
                        _c5 = (__m256)__lasx_xvld(pC0 + c_hstep * 5, 0);
                        _c6 = (__m256)__lasx_xvld(pC0 + c_hstep * 6, 0);
                        _c7 = (__m256)__lasx_xvld(pC0 + c_hstep * 7, 0);
                    }
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
                        __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                        _f0 = __lasx_xvfmadd_s(_c0, _beta, _f0);
                        _f1 = __lasx_xvfmadd_s(_c1, _beta, _f1);
                        _f2 = __lasx_xvfmadd_s(_c2, _beta, _f2);
                        _f3 = __lasx_xvfmadd_s(_c3, _beta, _f3);
                        _f4 = __lasx_xvfmadd_s(_c4, _beta, _f4);
                        _f5 = __lasx_xvfmadd_s(_c5, _beta, _f5);
                        _f6 = __lasx_xvfmadd_s(_c6, _beta, _f6);
                        _f7 = __lasx_xvfmadd_s(_c7, _beta, _f7);
                    }
                    pC += 8 * c_elempack;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c = (__m256)__lasx_xvld(pC0, 0);
                    if (beta != 1.f)
                    {
                        __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                        _c = __lasx_xvfmul_s(_c, _beta);
                    }
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
                __m256 _alpha = (__m256)__lasx_xvreplfr2vr_s(alpha);
                _f0 = __lasx_xvfmul_s(_f0, _alpha);
                _f1 = __lasx_xvfmul_s(_f1, _alpha);
                _f2 = __lasx_xvfmul_s(_f2, _alpha);
                _f3 = __lasx_xvfmul_s(_f3, _alpha);
                _f4 = __lasx_xvfmul_s(_f4, _alpha);
                _f5 = __lasx_xvfmul_s(_f5, _alpha);
                _f6 = __lasx_xvfmul_s(_f6, _alpha);
                _f7 = __lasx_xvfmul_s(_f7, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (out_elempack == 8)
                {
                    transpose8x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                    __lasx_xvst(_f0, p0f, 0);
                    __lasx_xvst(_f1, p0f + 8, 0);
                    __lasx_xvst(_f2, p0f + 16, 0);
                    __lasx_xvst(_f3, p0f + 24, 0);
                    __lasx_xvst(_f4, p0f + 32, 0);
                    __lasx_xvst(_f5, p0f + 40, 0);
                    __lasx_xvst(_f6, p0f + 48, 0);
                    __lasx_xvst(_f7, p0f + 56, 0);
                }
                if (out_elempack == 4)
                {
                    float* p1f = p0f + out_hstep * 4;
                    transpose8x4_ps(_f0, _f1, _f2, _f3);
                    transpose8x4_ps(_f4, _f5, _f6, _f7);
                    __lasx_xvst(_f0, p0f, 0);
                    __lasx_xvst(_f1, p0f + 8, 0);
                    __lasx_xvst(_f2, p0f + 16, 0);
                    __lasx_xvst(_f3, p0f + 24, 0);
                    __lasx_xvst(_f4, p1f, 0);
                    __lasx_xvst(_f5, p1f + 8, 0);
                    __lasx_xvst(_f6, p1f + 16, 0);
                    __lasx_xvst(_f7, p1f + 24, 0);
                }
                if (out_elempack == 1)
                {
                    __lasx_xvst(_f0, p0f, 0);
                    __lasx_xvst(_f1, p0f + out_hstep, 0);
                    __lasx_xvst(_f2, p0f + out_hstep * 2, 0);
                    __lasx_xvst(_f3, p0f + out_hstep * 3, 0);
                    __lasx_xvst(_f4, p0f + out_hstep * 4, 0);
                    __lasx_xvst(_f5, p0f + out_hstep * 5, 0);
                    __lasx_xvst(_f6, p0f + out_hstep * 6, 0);
                    __lasx_xvst(_f7, p0f + out_hstep * 7, 0);
                }
                p0f += 8 * out_elempack;
            }
            else
            {
                if (out_elempack == 8)
                {
                    transpose8x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                    __lsx_vst(float2bfloat_lasx(_f0), p0, 0);
                    __lsx_vst(float2bfloat_lasx(_f1), p0 + 8, 0);
                    __lsx_vst(float2bfloat_lasx(_f2), p0 + 16, 0);
                    __lsx_vst(float2bfloat_lasx(_f3), p0 + 24, 0);
                    __lsx_vst(float2bfloat_lasx(_f4), p0 + 32, 0);
                    __lsx_vst(float2bfloat_lasx(_f5), p0 + 40, 0);
                    __lsx_vst(float2bfloat_lasx(_f6), p0 + 48, 0);
                    __lsx_vst(float2bfloat_lasx(_f7), p0 + 56, 0);
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    transpose8x4_ps(_f0, _f1, _f2, _f3);
                    transpose8x4_ps(_f4, _f5, _f6, _f7);
                    __lsx_vst(float2bfloat_lasx(_f0), p0, 0);
                    __lsx_vst(float2bfloat_lasx(_f1), p0 + 8, 0);
                    __lsx_vst(float2bfloat_lasx(_f2), p0 + 16, 0);
                    __lsx_vst(float2bfloat_lasx(_f3), p0 + 24, 0);
                    __lsx_vst(float2bfloat_lasx(_f4), p1, 0);
                    __lsx_vst(float2bfloat_lasx(_f5), p1 + 8, 0);
                    __lsx_vst(float2bfloat_lasx(_f6), p1 + 16, 0);
                    __lsx_vst(float2bfloat_lasx(_f7), p1 + 24, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vst(float2bfloat_lasx(_f0), p0, 0);
                    __lsx_vst(float2bfloat_lasx(_f1), p0 + out_hstep, 0);
                    __lsx_vst(float2bfloat_lasx(_f2), p0 + out_hstep * 2, 0);
                    __lsx_vst(float2bfloat_lasx(_f3), p0 + out_hstep * 3, 0);
                    __lsx_vst(float2bfloat_lasx(_f4), p0 + out_hstep * 4, 0);
                    __lsx_vst(float2bfloat_lasx(_f5), p0 + out_hstep * 5, 0);
                    __lsx_vst(float2bfloat_lasx(_f6), p0 + out_hstep * 6, 0);
                    __lsx_vst(float2bfloat_lasx(_f7), p0 + out_hstep * 7, 0);
                }
                p0 += 8 * out_elempack;
            }
            pp += 64;
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _r0 = __lsx_vld(pp, 0);
            __m128i _r1 = __lsx_vld(pp + 8, 0);
            __m128i _r2 = __lsx_vld(pp + 16, 0);
            __m128i _r3 = __lsx_vld(pp + 24, 0);
            _r2 = __lsx_vshuf4i_w(_r2, _LSX_SHUFFLE(1, 0, 3, 2));
            _r3 = __lsx_vshuf4i_w(_r3, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_r0, _r1, _r2, _r3);
            _r1 = __lsx_vshuf4i_w(_r1, _LSX_SHUFFLE(2, 1, 0, 3));
            _r2 = __lsx_vshuf4i_w(_r2, _LSX_SHUFFLE(1, 0, 3, 2));
            _r3 = __lsx_vshuf4i_w(_r3, _LSX_SHUFFLE(0, 3, 2, 1));
            __m128i _r4 = __lsx_vld(pp + 4, 0);
            __m128i _r5 = __lsx_vld(pp + 12, 0);
            __m128i _r6 = __lsx_vld(pp + 20, 0);
            __m128i _r7 = __lsx_vld(pp + 28, 0);
            pp += 32;
            _r6 = __lsx_vshuf4i_w(_r6, _LSX_SHUFFLE(1, 0, 3, 2));
            _r7 = __lsx_vshuf4i_w(_r7, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_r4, _r5, _r6, _r7);
            _r5 = __lsx_vshuf4i_w(_r5, _LSX_SHUFFLE(2, 1, 0, 3));
            _r6 = __lsx_vshuf4i_w(_r6, _LSX_SHUFFLE(1, 0, 3, 2));
            _r7 = __lsx_vshuf4i_w(_r7, _LSX_SHUFFLE(0, 3, 2, 1));
            __m128 _f0 = (__m128)_r0;
            __m128 _f1 = (__m128)_r1;
            __m128 _f2 = (__m128)_r2;
            __m128 _f3 = (__m128)_r3;
            __m128 _f4 = (__m128)_r4;
            __m128 _f5 = (__m128)_r5;
            __m128 _f6 = (__m128)_r6;
            __m128 _f7 = (__m128)_r7;
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
                    __m128 _c0;
                    __m128 _c1;
                    __m128 _c2;
                    __m128 _c3;
                    __m128 _c4;
                    __m128 _c5;
                    __m128 _c6;
                    __m128 _c7;
                    if (c_elempack == 8)
                    {
                        _c0 = (__m128)__lsx_vld(pC, 0);
                        _c1 = (__m128)__lsx_vld(pC + 8, 0);
                        _c2 = (__m128)__lsx_vld(pC + 16, 0);
                        _c3 = (__m128)__lsx_vld(pC + 24, 0);
                        _c4 = (__m128)__lsx_vld(pC + 4, 0);
                        _c5 = (__m128)__lsx_vld(pC + 12, 0);
                        _c6 = (__m128)__lsx_vld(pC + 20, 0);
                        _c7 = (__m128)__lsx_vld(pC + 28, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        transpose4x4_ps(_c4, _c5, _c6, _c7);
                    }
                    else if (c_elempack == 4)
                    {
                        const float* pC1 = pC + c_hstep * 4;
                        _c0 = (__m128)__lsx_vld(pC, 0);
                        _c1 = (__m128)__lsx_vld(pC + 4, 0);
                        _c2 = (__m128)__lsx_vld(pC + 8, 0);
                        _c3 = (__m128)__lsx_vld(pC + 12, 0);
                        _c4 = (__m128)__lsx_vld(pC1, 0);
                        _c5 = (__m128)__lsx_vld(pC1 + 4, 0);
                        _c6 = (__m128)__lsx_vld(pC1 + 8, 0);
                        _c7 = (__m128)__lsx_vld(pC1 + 12, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        transpose4x4_ps(_c4, _c5, _c6, _c7);
                    }
                    else
                    {
                        _c0 = (__m128)__lsx_vld(pC, 0);
                        _c1 = (__m128)__lsx_vld(pC + c_hstep, 0);
                        _c2 = (__m128)__lsx_vld(pC + c_hstep * 2, 0);
                        _c3 = (__m128)__lsx_vld(pC + c_hstep * 3, 0);
                        _c4 = (__m128)__lsx_vld(pC + c_hstep * 4, 0);
                        _c5 = (__m128)__lsx_vld(pC + c_hstep * 5, 0);
                        _c6 = (__m128)__lsx_vld(pC + c_hstep * 6, 0);
                        _c7 = (__m128)__lsx_vld(pC + c_hstep * 7, 0);
                    }
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
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _f0 = __lsx_vfmadd_s(_c0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s(_c1, _beta, _f1);
                        _f2 = __lsx_vfmadd_s(_c2, _beta, _f2);
                        _f3 = __lsx_vfmadd_s(_c3, _beta, _f3);
                        _f4 = __lsx_vfmadd_s(_c4, _beta, _f4);
                        _f5 = __lsx_vfmadd_s(_c5, _beta, _f5);
                        _f6 = __lsx_vfmadd_s(_c6, _beta, _f6);
                        _f7 = __lsx_vfmadd_s(_c7, _beta, _f7);
                    }
                    pC += 4 * c_elempack;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = (__m128)__lsx_vld(pC, 0);
                    if (beta != 1.f)
                    {
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _c = __lsx_vfmul_s(_c, _beta);
                    }
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
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
                _f2 = __lsx_vfmul_s(_f2, _alpha);
                _f3 = __lsx_vfmul_s(_f3, _alpha);
                _f4 = __lsx_vfmul_s(_f4, _alpha);
                _f5 = __lsx_vfmul_s(_f5, _alpha);
                _f6 = __lsx_vfmul_s(_f6, _alpha);
                _f7 = __lsx_vfmul_s(_f7, _alpha);
            }

            if (output_elemtype == 1)
            {
#if __loongarch_asx
                if (out_elempack == 8)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __lasx_xvst(__lasx_concat_128((__m128i)_f0, (__m128i)_f4), p0f, 0);
                    __lasx_xvst(__lasx_concat_128((__m128i)_f1, (__m128i)_f5), p0f + 8, 0);
                    __lasx_xvst(__lasx_concat_128((__m128i)_f2, (__m128i)_f6), p0f + 16, 0);
                    __lasx_xvst(__lasx_concat_128((__m128i)_f3, (__m128i)_f7), p0f + 24, 0);
                }
#endif // __loongarch_asx
                if (out_elempack == 4)
                {
                    float* p1f = p0f + out_hstep * 4;
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __lsx_vst((__m128i)_f0, p0f, 0);
                    __lsx_vst((__m128i)_f1, p0f + 4, 0);
                    __lsx_vst((__m128i)_f2, p0f + 8, 0);
                    __lsx_vst((__m128i)_f3, p0f + 12, 0);
                    __lsx_vst((__m128i)_f4, p1f, 0);
                    __lsx_vst((__m128i)_f5, p1f + 4, 0);
                    __lsx_vst((__m128i)_f6, p1f + 8, 0);
                    __lsx_vst((__m128i)_f7, p1f + 12, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vst((__m128i)_f0, p0f, 0);
                    __lsx_vst((__m128i)_f1, p0f + out_hstep, 0);
                    __lsx_vst((__m128i)_f2, p0f + out_hstep * 2, 0);
                    __lsx_vst((__m128i)_f3, p0f + out_hstep * 3, 0);
                    __lsx_vst((__m128i)_f4, p0f + out_hstep * 4, 0);
                    __lsx_vst((__m128i)_f5, p0f + out_hstep * 5, 0);
                    __lsx_vst((__m128i)_f6, p0f + out_hstep * 6, 0);
                    __lsx_vst((__m128i)_f7, p0f + out_hstep * 7, 0);
                }
                p0f += 4 * out_elempack;
            }
            else
            {
                if (out_elempack == 8)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __lsx_vst(float2bfloat_lsx(_f0, _f4), p0, 0);
                    __lsx_vst(float2bfloat_lsx(_f1, _f5), p0 + 8, 0);
                    __lsx_vst(float2bfloat_lsx(_f2, _f6), p0 + 16, 0);
                    __lsx_vst(float2bfloat_lsx(_f3, _f7), p0 + 24, 0);
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __lsx_vstelm_d(float2bfloat_lsx(_f0), p0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f1), p0 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f2), p0 + 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f3), p0 + 12, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f4), p1, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f5), p1 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f6), p1 + 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f7), p1 + 12, 0, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vstelm_d(float2bfloat_lsx(_f0), p0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f1), p0 + out_hstep, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f2), p0 + out_hstep * 2, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f3), p0 + out_hstep * 3, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f4), p0 + out_hstep * 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f5), p0 + out_hstep * 5, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f6), p0 + out_hstep * 6, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f7), p0 + out_hstep * 7, 0, 0);
                }
                p0 += 4 * out_elempack;
            }
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
            pp += 16;
            __m128 _f0 = (__m128)_fi0;
            __m128 _f1 = (__m128)_fi1;
            __m128 _f2 = (__m128)_fi2;
            __m128 _f3 = (__m128)_fi3;
            __m128 _f4 = (__m128)_fi4;
            __m128 _f5 = (__m128)_fi5;
            __m128 _f6 = (__m128)_fi6;
            __m128 _f7 = (__m128)_fi7;
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
                    __m128i _ci0;
                    __m128i _ci1;
                    __m128i _ci2;
                    __m128i _ci3;
                    __m128i _ci4;
                    __m128i _ci5;
                    __m128i _ci6;
                    __m128i _ci7;
                    if (c_elempack == 8)
                    {
                        _ci0 = __lsx_vldrepl_w(pC, 0);
                        _ci1 = __lsx_vldrepl_w(pC + 1, 0);
                        _ci2 = __lsx_vldrepl_w(pC + 2, 0);
                        _ci3 = __lsx_vldrepl_w(pC + 3, 0);
                        _ci4 = __lsx_vldrepl_w(pC + 4, 0);
                        _ci5 = __lsx_vldrepl_w(pC + 5, 0);
                        _ci6 = __lsx_vldrepl_w(pC + 6, 0);
                        _ci7 = __lsx_vldrepl_w(pC + 7, 0);
                        _ci0 = __lsx_vinsgr2vr_w(_ci0, ((const int*)(pC + 8))[0], 1);
                        _ci1 = __lsx_vinsgr2vr_w(_ci1, ((const int*)(pC + 9))[0], 1);
                        _ci2 = __lsx_vinsgr2vr_w(_ci2, ((const int*)(pC + 10))[0], 1);
                        _ci3 = __lsx_vinsgr2vr_w(_ci3, ((const int*)(pC + 11))[0], 1);
                        _ci4 = __lsx_vinsgr2vr_w(_ci4, ((const int*)(pC + 12))[0], 1);
                        _ci5 = __lsx_vinsgr2vr_w(_ci5, ((const int*)(pC + 13))[0], 1);
                        _ci6 = __lsx_vinsgr2vr_w(_ci6, ((const int*)(pC + 14))[0], 1);
                        _ci7 = __lsx_vinsgr2vr_w(_ci7, ((const int*)(pC + 15))[0], 1);
                    }
                    else if (c_elempack == 4)
                    {
                        const float* pC1 = pC + c_hstep * 4;
                        _ci0 = __lsx_vldrepl_w(pC, 0);
                        _ci1 = __lsx_vldrepl_w(pC + 1, 0);
                        _ci2 = __lsx_vldrepl_w(pC + 2, 0);
                        _ci3 = __lsx_vldrepl_w(pC + 3, 0);
                        _ci4 = __lsx_vldrepl_w(pC1, 0);
                        _ci5 = __lsx_vldrepl_w(pC1 + 1, 0);
                        _ci6 = __lsx_vldrepl_w(pC1 + 2, 0);
                        _ci7 = __lsx_vldrepl_w(pC1 + 3, 0);
                        _ci0 = __lsx_vinsgr2vr_w(_ci0, ((const int*)(pC + 4))[0], 1);
                        _ci1 = __lsx_vinsgr2vr_w(_ci1, ((const int*)(pC + 5))[0], 1);
                        _ci2 = __lsx_vinsgr2vr_w(_ci2, ((const int*)(pC + 6))[0], 1);
                        _ci3 = __lsx_vinsgr2vr_w(_ci3, ((const int*)(pC + 7))[0], 1);
                        _ci4 = __lsx_vinsgr2vr_w(_ci4, ((const int*)(pC1 + 4))[0], 1);
                        _ci5 = __lsx_vinsgr2vr_w(_ci5, ((const int*)(pC1 + 5))[0], 1);
                        _ci6 = __lsx_vinsgr2vr_w(_ci6, ((const int*)(pC1 + 6))[0], 1);
                        _ci7 = __lsx_vinsgr2vr_w(_ci7, ((const int*)(pC1 + 7))[0], 1);
                    }
                    else
                    {
                        _ci0 = __lsx_vldrepl_d(pC, 0);
                        _ci1 = __lsx_vldrepl_d(pC + c_hstep, 0);
                        _ci2 = __lsx_vldrepl_d(pC + c_hstep * 2, 0);
                        _ci3 = __lsx_vldrepl_d(pC + c_hstep * 3, 0);
                        _ci4 = __lsx_vldrepl_d(pC + c_hstep * 4, 0);
                        _ci5 = __lsx_vldrepl_d(pC + c_hstep * 5, 0);
                        _ci6 = __lsx_vldrepl_d(pC + c_hstep * 6, 0);
                        _ci7 = __lsx_vldrepl_d(pC + c_hstep * 7, 0);
                    }
                    __m128 _c0 = (__m128)_ci0;
                    __m128 _c1 = (__m128)_ci1;
                    __m128 _c2 = (__m128)_ci2;
                    __m128 _c3 = (__m128)_ci3;
                    __m128 _c4 = (__m128)_ci4;
                    __m128 _c5 = (__m128)_ci5;
                    __m128 _c6 = (__m128)_ci6;
                    __m128 _c7 = (__m128)_ci7;
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
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _f0 = __lsx_vfmadd_s(_c0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s(_c1, _beta, _f1);
                        _f2 = __lsx_vfmadd_s(_c2, _beta, _f2);
                        _f3 = __lsx_vfmadd_s(_c3, _beta, _f3);
                        _f4 = __lsx_vfmadd_s(_c4, _beta, _f4);
                        _f5 = __lsx_vfmadd_s(_c5, _beta, _f5);
                        _f6 = __lsx_vfmadd_s(_c6, _beta, _f6);
                        _f7 = __lsx_vfmadd_s(_c7, _beta, _f7);
                    }
                    pC += 2 * c_elempack;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = (__m128)__lsx_vldrepl_d(pC, 0);
                    if (beta != 1.f)
                    {
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _c = __lsx_vfmul_s(_c, _beta);
                    }
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                    _f2 = __lsx_vfadd_s(_f2, _c);
                    _f3 = __lsx_vfadd_s(_f3, _c);
                    _f4 = __lsx_vfadd_s(_f4, _c);
                    _f5 = __lsx_vfadd_s(_f5, _c);
                    _f6 = __lsx_vfadd_s(_f6, _c);
                    _f7 = __lsx_vfadd_s(_f7, _c);
                    pC += 2 * c_elempack;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
                _f2 = __lsx_vfmul_s(_f2, _alpha);
                _f3 = __lsx_vfmul_s(_f3, _alpha);
                _f4 = __lsx_vfmul_s(_f4, _alpha);
                _f5 = __lsx_vfmul_s(_f5, _alpha);
                _f6 = __lsx_vfmul_s(_f6, _alpha);
                _f7 = __lsx_vfmul_s(_f7, _alpha);
            }

            if (output_elemtype == 1)
            {
#if __loongarch_asx
                if (out_elempack == 8)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __lasx_xvst(__lasx_concat_128((__m128i)_f0, (__m128i)_f4), p0f, 0);
                    __lasx_xvst(__lasx_concat_128((__m128i)_f1, (__m128i)_f5), p0f + 8, 0);
                }
#endif // __loongarch_asx
                if (out_elempack == 4)
                {
                    float* p1f = p0f + out_hstep * 4;
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __lsx_vst((__m128i)_f0, p0f, 0);
                    __lsx_vst((__m128i)_f1, p0f + 4, 0);
                    __lsx_vst((__m128i)_f4, p1f, 0);
                    __lsx_vst((__m128i)_f5, p1f + 4, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vstelm_d((__m128i)_f0, p0f, 0, 0);
                    __lsx_vstelm_d((__m128i)_f1, p0f + out_hstep, 0, 0);
                    __lsx_vstelm_d((__m128i)_f2, p0f + out_hstep * 2, 0, 0);
                    __lsx_vstelm_d((__m128i)_f3, p0f + out_hstep * 3, 0, 0);
                    __lsx_vstelm_d((__m128i)_f4, p0f + out_hstep * 4, 0, 0);
                    __lsx_vstelm_d((__m128i)_f5, p0f + out_hstep * 5, 0, 0);
                    __lsx_vstelm_d((__m128i)_f6, p0f + out_hstep * 6, 0, 0);
                    __lsx_vstelm_d((__m128i)_f7, p0f + out_hstep * 7, 0, 0);
                }
                p0f += 2 * out_elempack;
            }
            else
            {
                if (out_elempack == 8)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __lsx_vst(float2bfloat_lsx(_f0, _f4), p0, 0);
                    __lsx_vst(float2bfloat_lsx(_f1, _f5), p0 + 8, 0);
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __lsx_vstelm_d(float2bfloat_lsx(_f0), p0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f1), p0 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f4), p1, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f5), p1 + 4, 0, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vstelm_w(float2bfloat_lsx((__m128)((__m128i)_f0)), p0, 0, 0);
                    __lsx_vstelm_w(float2bfloat_lsx((__m128)((__m128i)_f1)), p0 + out_hstep, 0, 0);
                    __lsx_vstelm_w(float2bfloat_lsx((__m128)((__m128i)_f2)), p0 + out_hstep * 2, 0, 0);
                    __lsx_vstelm_w(float2bfloat_lsx((__m128)((__m128i)_f3)), p0 + out_hstep * 3, 0, 0);
                    __lsx_vstelm_w(float2bfloat_lsx((__m128)((__m128i)_f4)), p0 + out_hstep * 4, 0, 0);
                    __lsx_vstelm_w(float2bfloat_lsx((__m128)((__m128i)_f5)), p0 + out_hstep * 5, 0, 0);
                    __lsx_vstelm_w(float2bfloat_lsx((__m128)((__m128i)_f6)), p0 + out_hstep * 6, 0, 0);
                    __lsx_vstelm_w(float2bfloat_lsx((__m128)((__m128i)_f7)), p0 + out_hstep * 7, 0, 0);
                }
                p0 += 2 * out_elempack;
            }
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
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vfmul_s((__m128)__lsx_vld((const float*)C + i + ii, 0), _beta));
                    _f4 = __lsx_vfadd_s(_f4, __lsx_vfmul_s((__m128)__lsx_vld((const float*)C + i + ii + 4, 0), _beta));
                }
                if (broadcast_type_C == 3)
                {
                    __m128i _c0;
                    __m128i _c4;
                    if (c_elempack == 8)
                    {
                        _c0 = __lsx_vld(pC, 0);
                        _c4 = __lsx_vld(pC + 4, 0);
                    }
                    else if (c_elempack == 4)
                    {
                        _c0 = __lsx_vld(pC, 0);
                        _c4 = __lsx_vld(pC + c_hstep * 4, 0);
                    }
                    else
                    {
                        _c0 = __lsx_vldrepl_w(pC, 0);
                        _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep))[0], 1);
                        _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep * 2))[0], 2);
                        _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep * 3))[0], 3);
                        _c4 = __lsx_vldrepl_w(pC + c_hstep * 4, 0);
                        _c4 = __lsx_vinsgr2vr_w(_c4, ((const int*)(pC + c_hstep * 5))[0], 1);
                        _c4 = __lsx_vinsgr2vr_w(_c4, ((const int*)(pC + c_hstep * 6))[0], 2);
                        _c4 = __lsx_vinsgr2vr_w(_c4, ((const int*)(pC + c_hstep * 7))[0], 3);
                    }
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, (__m128)_c0);
                        _f4 = __lsx_vfadd_s(_f4, (__m128)_c4);
                    }
                    else
                    {
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _f0 = __lsx_vfmadd_s((__m128)_c0, _beta, _f0);
                        _f4 = __lsx_vfmadd_s((__m128)_c4, _beta, _f4);
                    }
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = __lsx_vreplfr2vr_s(pC[0] * beta);
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f4 = __lsx_vfadd_s(_f4, _c);
                    pC++;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f4 = __lsx_vfmul_s(_f4, _alpha);
            }

            if (output_elemtype == 1)
            {
#if __loongarch_asx
                if (out_elempack == 8)
                {
                    __lasx_xvst(__lasx_concat_128((__m128i)_f0, (__m128i)_f4), p0f, 0);
                }
#endif // __loongarch_asx
                if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_f0, p0f, 0);
                    __lsx_vst((__m128i)_f4, p0f + out_hstep * 4, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vstelm_w((__m128i)_f0, p0f, 0, 0);
                    __lsx_vstelm_w((__m128i)_f0, p0f + out_hstep, 0, 1);
                    __lsx_vstelm_w((__m128i)_f0, p0f + out_hstep * 2, 0, 2);
                    __lsx_vstelm_w((__m128i)_f0, p0f + out_hstep * 3, 0, 3);
                    __lsx_vstelm_w((__m128i)_f4, p0f + out_hstep * 4, 0, 0);
                    __lsx_vstelm_w((__m128i)_f4, p0f + out_hstep * 5, 0, 1);
                    __lsx_vstelm_w((__m128i)_f4, p0f + out_hstep * 6, 0, 2);
                    __lsx_vstelm_w((__m128i)_f4, p0f + out_hstep * 7, 0, 3);
                }
                p0f += out_elempack;
            }
            else
            {
                if (out_elempack == 8)
                {
                    __lsx_vst(float2bfloat_lsx(_f0, _f4), p0, 0);
                }
                if (out_elempack == 4)
                {
                    __lsx_vstelm_d(float2bfloat_lsx(_f0), p0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f4), p0 + out_hstep * 4, 0, 0);
                }
                if (out_elempack == 1)
                {
                    __m128i _q0 = float2bfloat_lsx(_f0);
                    __m128i _q4 = float2bfloat_lsx(_f4);
                    __lsx_vstelm_h(_q0, p0, 0, 0);
                    __lsx_vstelm_h(_q0, p0 + out_hstep, 0, 1);
                    __lsx_vstelm_h(_q0, p0 + out_hstep * 2, 0, 2);
                    __lsx_vstelm_h(_q0, p0 + out_hstep * 3, 0, 3);
                    __lsx_vstelm_h(_q4, p0 + out_hstep * 4, 0, 0);
                    __lsx_vstelm_h(_q4, p0 + out_hstep * 5, 0, 1);
                    __lsx_vstelm_h(_q4, p0 + out_hstep * 6, 0, 2);
                    __lsx_vstelm_h(_q4, p0 + out_hstep * 7, 0, 3);
                }
                p0 += out_elempack;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* p0f = output_elemtype == 1 ? (float*)top_blob + (size_t)(i + ii) * out_hstep + j * out_elempack : 0;
        unsigned short* p0 = output_elemtype == 3 ? (unsigned short*)top_blob + (size_t)(i + ii) * out_hstep + j * out_elempack : 0;

        float c0 = 0.f;
        float c1 = 0.f;
        float c2 = 0.f;
        float c3 = 0.f;
        const float* pC = C;
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
                pC += i + ii;
                c0 = pC[0] * beta;
                c1 = pC[1] * beta;
                c2 = pC[2] * beta;
                c3 = pC[3] * beta;
            }
            if (broadcast_type_C == 3)
            {
                pC += (size_t)(i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC += j;
            }
        }

        int jj = 0;
#if __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const float* pC0 = pC;
            __m256i _s0 = __lasx_xvld(pp, 0);
            __m256i _s1 = __lasx_xvld(pp + 8, 0);
            __m256i _s2 = __lasx_xvld(pp + 16, 0);
            __m256i _s3 = __lasx_xvld(pp + 24, 0);

            _s2 = __lasx_xvshuf4i_w(_s2, _LSX_SHUFFLE(1, 0, 3, 2));
            _s3 = __lasx_xvshuf4i_w(_s3, _LSX_SHUFFLE(1, 0, 3, 2));
            {
                __m256i _tmp0 = __lasx_xvilvl_w(_s1, _s0);
                __m256i _tmp1 = __lasx_xvilvh_w(_s1, _s0);
                __m256i _tmp2 = __lasx_xvilvl_w(_s3, _s2);
                __m256i _tmp3 = __lasx_xvilvh_w(_s3, _s2);
                _s0 = __lasx_xvilvl_d(_tmp2, _tmp0);
                _s1 = __lasx_xvilvh_d(_tmp2, _tmp0);
                _s2 = __lasx_xvilvl_d(_tmp3, _tmp1);
                _s3 = __lasx_xvilvh_d(_tmp3, _tmp1);
            }
            _s1 = __lasx_xvshuf4i_w(_s1, _LSX_SHUFFLE(2, 1, 0, 3));
            _s2 = __lasx_xvshuf4i_w(_s2, _LSX_SHUFFLE(1, 0, 3, 2));
            _s3 = __lasx_xvshuf4i_w(_s3, _LSX_SHUFFLE(0, 3, 2, 1));

            __m256 _f0 = (__m256)_s0;
            __m256 _f1 = (__m256)_s1;
            __m256 _f2 = (__m256)_s2;
            __m256 _f3 = (__m256)_s3;
            if (pC0)
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
                    __m256 _c0;
                    __m256 _c1;
                    __m256 _c2;
                    __m256 _c3;
                    if (c_elempack == 4)
                    {
                        __m128 _c00 = (__m128)__lsx_vld(pC0, 0);
                        __m128 _c10 = (__m128)__lsx_vld(pC0 + 4, 0);
                        __m128 _c20 = (__m128)__lsx_vld(pC0 + 8, 0);
                        __m128 _c30 = (__m128)__lsx_vld(pC0 + 12, 0);
                        transpose4x4_ps(_c00, _c10, _c20, _c30);

                        __m128 _c01 = (__m128)__lsx_vld(pC0 + 16, 0);
                        __m128 _c11 = (__m128)__lsx_vld(pC0 + 20, 0);
                        __m128 _c21 = (__m128)__lsx_vld(pC0 + 24, 0);
                        __m128 _c31 = (__m128)__lsx_vld(pC0 + 28, 0);
                        transpose4x4_ps(_c01, _c11, _c21, _c31);

                        _c0 = __lasx_concat_128_s(_c00, _c01);
                        _c1 = __lasx_concat_128_s(_c10, _c11);
                        _c2 = __lasx_concat_128_s(_c20, _c21);
                        _c3 = __lasx_concat_128_s(_c30, _c31);
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = (__m256)__lasx_xvld(pC0, 0);
                        _c1 = (__m256)__lasx_xvld(pC0 + c_hstep, 0);
                        _c2 = (__m256)__lasx_xvld(pC0 + c_hstep * 2, 0);
                        _c3 = (__m256)__lasx_xvld(pC0 + c_hstep * 3, 0);
                    }
                    if (beta == 1.f)
                    {
                        _f0 = __lasx_xvfadd_s(_f0, _c0);
                        _f1 = __lasx_xvfadd_s(_f1, _c1);
                        _f2 = __lasx_xvfadd_s(_f2, _c2);
                        _f3 = __lasx_xvfadd_s(_f3, _c3);
                    }
                    else
                    {
                        __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                        _f0 = __lasx_xvfmadd_s(_c0, _beta, _f0);
                        _f1 = __lasx_xvfmadd_s(_c1, _beta, _f1);
                        _f2 = __lasx_xvfmadd_s(_c2, _beta, _f2);
                        _f3 = __lasx_xvfmadd_s(_c3, _beta, _f3);
                    }
                    pC += 8 * c_elempack;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c = (__m256)__lasx_xvld(pC0, 0);
                    if (beta != 1.f)
                    {
                        __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                        _c = __lasx_xvfmul_s(_c, _beta);
                    }
                    _f0 = __lasx_xvfadd_s(_f0, _c);
                    _f1 = __lasx_xvfadd_s(_f1, _c);
                    _f2 = __lasx_xvfadd_s(_f2, _c);
                    _f3 = __lasx_xvfadd_s(_f3, _c);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m256 _alpha = (__m256)__lasx_xvreplfr2vr_s(alpha);
                _f0 = __lasx_xvfmul_s(_f0, _alpha);
                _f1 = __lasx_xvfmul_s(_f1, _alpha);
                _f2 = __lasx_xvfmul_s(_f2, _alpha);
                _f3 = __lasx_xvfmul_s(_f3, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    transpose8x4_ps(_f0, _f1, _f2, _f3);
                    __lasx_xvst(_f0, p0f, 0);
                    __lasx_xvst(_f1, p0f + 8, 0);
                    __lasx_xvst(_f2, p0f + 16, 0);
                    __lasx_xvst(_f3, p0f + 24, 0);
                }
                if (out_elempack == 1)
                {
                    __lasx_xvst(_f0, p0f, 0);
                    __lasx_xvst(_f1, p0f + out_hstep, 0);
                    __lasx_xvst(_f2, p0f + out_hstep * 2, 0);
                    __lasx_xvst(_f3, p0f + out_hstep * 3, 0);
                }
            }
            else
            {
                if (out_elempack == 4)
                {
                    transpose8x4_ps(_f0, _f1, _f2, _f3);
                    __lsx_vst(float2bfloat_lasx(_f0), p0, 0);
                    __lsx_vst(float2bfloat_lasx(_f1), p0 + 8, 0);
                    __lsx_vst(float2bfloat_lasx(_f2), p0 + 16, 0);
                    __lsx_vst(float2bfloat_lasx(_f3), p0 + 24, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vst(float2bfloat_lasx(_f0), p0, 0);
                    __lsx_vst(float2bfloat_lasx(_f1), p0 + out_hstep, 0);
                    __lsx_vst(float2bfloat_lasx(_f2), p0 + out_hstep * 2, 0);
                    __lsx_vst(float2bfloat_lasx(_f3), p0 + out_hstep * 3, 0);
                }
            }
            pp += 32;
            if (output_elemtype == 1)
                p0f += 8 * out_elempack;
            else
                p0 += 8 * out_elempack;
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _s0 = __lsx_vld(pp, 0);
            __m128i _s1 = __lsx_vld(pp + 4, 0);
            __m128i _s2 = __lsx_vld(pp + 8, 0);
            __m128i _s3 = __lsx_vld(pp + 12, 0);
            pp += 16;

            _s2 = __lsx_vshuf4i_w(_s2, _LSX_SHUFFLE(1, 0, 3, 2));
            _s3 = __lsx_vshuf4i_w(_s3, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_s0, _s1, _s2, _s3);
            _s1 = __lsx_vshuf4i_w(_s1, _LSX_SHUFFLE(2, 1, 0, 3));
            _s2 = __lsx_vshuf4i_w(_s2, _LSX_SHUFFLE(1, 0, 3, 2));
            _s3 = __lsx_vshuf4i_w(_s3, _LSX_SHUFFLE(0, 3, 2, 1));

            __m128 _f0 = (__m128)_s0;
            __m128 _f1 = (__m128)_s1;
            __m128 _f2 = (__m128)_s2;
            __m128 _f3 = (__m128)_s3;
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
                    __m128 _c0;
                    __m128 _c1;
                    __m128 _c2;
                    __m128 _c3;
                    if (c_elempack == 4)
                    {
                        _c0 = (__m128)__lsx_vld(pC, 0);
                        _c1 = (__m128)__lsx_vld(pC + 4, 0);
                        _c2 = (__m128)__lsx_vld(pC + 8, 0);
                        _c3 = (__m128)__lsx_vld(pC + 12, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = (__m128)__lsx_vld(pC, 0);
                        _c1 = (__m128)__lsx_vld(pC + c_hstep, 0);
                        _c2 = (__m128)__lsx_vld(pC + c_hstep * 2, 0);
                        _c3 = (__m128)__lsx_vld(pC + c_hstep * 3, 0);
                    }
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                        _f1 = __lsx_vfadd_s(_f1, _c1);
                        _f2 = __lsx_vfadd_s(_f2, _c2);
                        _f3 = __lsx_vfadd_s(_f3, _c3);
                    }
                    else
                    {
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _f0 = __lsx_vfmadd_s(_c0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s(_c1, _beta, _f1);
                        _f2 = __lsx_vfmadd_s(_c2, _beta, _f2);
                        _f3 = __lsx_vfmadd_s(_c3, _beta, _f3);
                    }
                    pC += 4 * c_elempack;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = (__m128)__lsx_vld(pC, 0);
                    if (beta != 1.f)
                    {
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _c = __lsx_vfmul_s(_c, _beta);
                    }
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                    _f2 = __lsx_vfadd_s(_f2, _c);
                    _f3 = __lsx_vfadd_s(_f3, _c);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
                _f2 = __lsx_vfmul_s(_f2, _alpha);
                _f3 = __lsx_vfmul_s(_f3, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    __lsx_vst((__m128i)_f0, p0f, 0);
                    __lsx_vst((__m128i)_f1, p0f + 4, 0);
                    __lsx_vst((__m128i)_f2, p0f + 8, 0);
                    __lsx_vst((__m128i)_f3, p0f + 12, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vst((__m128i)_f0, p0f, 0);
                    __lsx_vst((__m128i)_f1, p0f + out_hstep, 0);
                    __lsx_vst((__m128i)_f2, p0f + out_hstep * 2, 0);
                    __lsx_vst((__m128i)_f3, p0f + out_hstep * 3, 0);
                }
                p0f += 4 * out_elempack;
            }
            else
            {
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    __lsx_vstelm_d(float2bfloat_lsx(_f0), p0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f1), p0 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f2), p0 + 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f3), p0 + 12, 0, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vstelm_d(float2bfloat_lsx(_f0), p0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f1), p0 + out_hstep, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f2), p0 + out_hstep * 2, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f3), p0 + out_hstep * 3, 0, 0);
                }
                p0 += 4 * out_elempack;
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128i _s0 = __lsx_vld(pp, 0);
            __m128i _s1 = __lsx_vld(pp + 4, 0);
            pp += 8;

            __m128i _s0e = __lsx_vshuf4i_w(_s0, _LSX_SHUFFLE(3, 1, 2, 0));
            __m128i _s0o = __lsx_vshuf4i_w(_s0, _LSX_SHUFFLE(2, 0, 3, 1));
            __m128i _s1e = __lsx_vshuf4i_w(_s1, _LSX_SHUFFLE(3, 1, 2, 0));
            __m128i _s1o = __lsx_vshuf4i_w(_s1, _LSX_SHUFFLE(2, 0, 3, 1));
            __m128i _c0 = __lsx_vilvl_w(_s1o, _s0e);
            __m128i _c1 = __lsx_vilvl_w(_s0o, _s1e);
            __m128i _t0 = __lsx_vilvl_w(_c1, _c0);
            __m128i _t1 = __lsx_vilvh_w(_c1, _c0);
            __m128 _f0 = (__m128)_t0;
            __m128 _f1 = (__m128)__lsx_vreplvei_d(_t0, 1);
            __m128 _f2 = (__m128)_t1;
            __m128 _f3 = (__m128)__lsx_vreplvei_d(_t1, 1);
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
                    __m128i _c0;
                    __m128i _c1;
                    __m128i _c2;
                    __m128i _c3;
                    if (c_elempack == 4)
                    {
                        _c0 = __lsx_vreplgr2vr_w(((const int*)pC)[0]);
                        _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + 4))[0], 1);
                        _c0 = __lsx_vreplvei_d(_c0, 0);
                        _c1 = __lsx_vreplgr2vr_w(((const int*)pC)[1]);
                        _c1 = __lsx_vinsgr2vr_w(_c1, ((const int*)(pC + 4))[1], 1);
                        _c1 = __lsx_vreplvei_d(_c1, 0);
                        _c2 = __lsx_vreplgr2vr_w(((const int*)pC)[2]);
                        _c2 = __lsx_vinsgr2vr_w(_c2, ((const int*)(pC + 4))[2], 1);
                        _c2 = __lsx_vreplvei_d(_c2, 0);
                        _c3 = __lsx_vreplgr2vr_w(((const int*)pC)[3]);
                        _c3 = __lsx_vinsgr2vr_w(_c3, ((const int*)(pC + 4))[3], 1);
                        _c3 = __lsx_vreplvei_d(_c3, 0);
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = __lsx_vldrepl_d(pC, 0);
                        _c1 = __lsx_vldrepl_d(pC + c_hstep, 0);
                        _c2 = __lsx_vldrepl_d(pC + c_hstep * 2, 0);
                        _c3 = __lsx_vldrepl_d(pC + c_hstep * 3, 0);
                    }
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, (__m128)_c0);
                        _f1 = __lsx_vfadd_s(_f1, (__m128)_c1);
                        _f2 = __lsx_vfadd_s(_f2, (__m128)_c2);
                        _f3 = __lsx_vfadd_s(_f3, (__m128)_c3);
                    }
                    else
                    {
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _f0 = __lsx_vfmadd_s((__m128)_c0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s((__m128)_c1, _beta, _f1);
                        _f2 = __lsx_vfmadd_s((__m128)_c2, _beta, _f2);
                        _f3 = __lsx_vfmadd_s((__m128)_c3, _beta, _f3);
                    }
                    pC += 2 * c_elempack;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = (__m128)__lsx_vldrepl_d(pC, 0);
                    if (beta != 1.f)
                    {
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _c = __lsx_vfmul_s(_c, _beta);
                    }
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                    _f2 = __lsx_vfadd_s(_f2, _c);
                    _f3 = __lsx_vfadd_s(_f3, _c);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
                _f2 = __lsx_vfmul_s(_f2, _alpha);
                _f3 = __lsx_vfmul_s(_f3, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    __lsx_vst((__m128i)_f0, p0f, 0);
                    __lsx_vst((__m128i)_f1, p0f + 4, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vstelm_d((__m128i)_f0, p0f, 0, 0);
                    __lsx_vstelm_d((__m128i)_f1, p0f + out_hstep, 0, 0);
                    __lsx_vstelm_d((__m128i)_f2, p0f + out_hstep * 2, 0, 0);
                    __lsx_vstelm_d((__m128i)_f3, p0f + out_hstep * 3, 0, 0);
                }
                p0f += 2 * out_elempack;
            }
            else
            {
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    __lsx_vstelm_d(float2bfloat_lsx(_f0), p0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f1), p0 + 4, 0, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vstelm_w(float2bfloat_lsx(_f0), p0, 0, 0);
                    __lsx_vstelm_w(float2bfloat_lsx(_f1), p0 + out_hstep, 0, 0);
                    __lsx_vstelm_w(float2bfloat_lsx(_f2), p0 + out_hstep * 2, 0, 0);
                    __lsx_vstelm_w(float2bfloat_lsx(_f3), p0 + out_hstep * 3, 0, 0);
                }
                p0 += 2 * out_elempack;
            }
        }
        for (; jj < max_jj; jj++)
        {
            __m128i _fi = __lsx_vld(pp, 0);
            pp += 4;
            __m128 _f0 = (__m128)_fi;
            if (pC)
            {
                __m128 _beta = __lsx_vreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(c0));
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vfmul_s((__m128)__lsx_vld((const float*)C + i + ii, 0), _beta));
                if (broadcast_type_C == 3)
                {
                    __m128i _c0;
                    if (c_elempack == 4)
                    {
                        _c0 = __lsx_vld(pC, 0);
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = __lsx_vreplgr2vr_w(((const int*)pC)[0]);
                        _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep))[0], 1);
                        _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep * 2))[0], 2);
                        _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep * 3))[0], 3);
                    }
                    if (beta == 1.f)
                        _f0 = __lsx_vfadd_s(_f0, (__m128)_c0);
                    else
                        _f0 = __lsx_vfmadd_s((__m128)_c0, __lsx_vreplfr2vr_s(beta), _f0);
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(pC[0] * beta));
                    pC++;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_f0, p0f, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vstelm_w((__m128i)_f0, p0f, 0, 0);
                    __lsx_vstelm_w((__m128i)_f0, p0f + out_hstep, 0, 1);
                    __lsx_vstelm_w((__m128i)_f0, p0f + out_hstep * 2, 0, 2);
                    __lsx_vstelm_w((__m128i)_f0, p0f + out_hstep * 3, 0, 3);
                }
                p0f += out_elempack;
            }
            else
            {
                __m128i _q = float2bfloat_lsx(_f0);
                if (out_elempack == 4)
                {
                    __lsx_vstelm_d(_q, p0, 0, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vstelm_h(_q, p0, 0, 0);
                    __lsx_vstelm_h(_q, p0 + out_hstep, 0, 1);
                    __lsx_vstelm_h(_q, p0 + out_hstep * 2, 0, 2);
                    __lsx_vstelm_h(_q, p0 + out_hstep * 3, 0, 3);
                }
                p0 += out_elempack;
            }
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* p0f = output_elemtype == 1 ? (float*)top_blob + (size_t)(i + ii) * out_hstep + j : 0;
        unsigned short* p0 = output_elemtype == 3 ? (unsigned short*)top_blob + (size_t)(i + ii) * out_hstep + j : 0;

        float c0 = 0.f;
        float c1 = 0.f;
        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
                c1 = c0;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC += i + ii;
                c0 = pC[0] * beta;
                c1 = pC[1] * beta;
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
        for (; jj + 7 < max_jj; jj += 8)
        {
            const float* pC0 = pC;
            __m256 _f0 = (__m256)__lasx_xvld(pp, 0);
            __m256 _f1 = (__m256)__lasx_xvld(pp + 8, 0);
            if (pC0)
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
                    __m256 _c0 = (__m256)__lasx_xvld(pC0, 0);
                    __m256 _c1 = (__m256)__lasx_xvld(pC0 + c_hstep, 0);
                    if (beta == 1.f)
                    {
                        _f0 = __lasx_xvfadd_s(_f0, _c0);
                        _f1 = __lasx_xvfadd_s(_f1, _c1);
                    }
                    else
                    {
                        __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                        _f0 = __lasx_xvfmadd_s(_c0, _beta, _f0);
                        _f1 = __lasx_xvfmadd_s(_c1, _beta, _f1);
                    }
                    pC += 8 * c_elempack;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c = (__m256)__lasx_xvld(pC0, 0);
                    if (beta != 1.f)
                    {
                        __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                        _c = __lasx_xvfmul_s(_c, _beta);
                    }
                    _f0 = __lasx_xvfadd_s(_f0, _c);
                    _f1 = __lasx_xvfadd_s(_f1, _c);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m256 _alpha = (__m256)__lasx_xvreplfr2vr_s(alpha);
                _f0 = __lasx_xvfmul_s(_f0, _alpha);
                _f1 = __lasx_xvfmul_s(_f1, _alpha);
            }

            if (output_elemtype == 1)
            {
                __lasx_xvst(_f0, p0f, 0);
                __lasx_xvst(_f1, p0f + out_hstep, 0);
            }
            else
            {
                __lsx_vst(float2bfloat_lasx((__m256)_f0), p0, 0);
                __lsx_vst(float2bfloat_lasx((__m256)_f1), p0 + out_hstep, 0);
            }
            pp += 16;
            if (output_elemtype == 1)
                p0f += 8;
            else
                p0 += 8;
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp, 0);
            __m128 _f1 = (__m128)__lsx_vld(pp + 4, 0);
            pp += 8;
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
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _f0 = __lsx_vfmadd_s(_c0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s(_c1, _beta, _f1);
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = (__m128)__lsx_vld(pC, 0);
                    if (beta != 1.f)
                    {
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _c = __lsx_vfmul_s(_c, _beta);
                    }
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
            }

            if (output_elemtype == 1)
            {
                __lsx_vst((__m128i)_f0, p0f, 0);
                __lsx_vst((__m128i)_f1, p0f + out_hstep, 0);
                p0f += 4;
            }
            else
            {
                __lsx_vstelm_d(float2bfloat_lsx((__m128)((__m128i)_f0)), p0, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)((__m128i)_f1)), p0 + out_hstep, 0, 0);
                p0 += 4;
            }
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            const float* pC0 = pC;
#if __loongarch_sx
            __m128 _f0 = (__m128)__lsx_vldrepl_d(pp, 0);
            __m128 _f1 = (__m128)__lsx_vldrepl_d(pp + 2, 0);
            if (pC0)
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
                    __m128 _c0 = (__m128)__lsx_vldrepl_d(pC0, 0);
                    __m128 _c1 = (__m128)__lsx_vldrepl_d(pC0 + c_hstep, 0);
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                        _f1 = __lsx_vfadd_s(_f1, _c1);
                    }
                    else
                    {
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _f0 = __lsx_vfmadd_s(_c0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s(_c1, _beta, _f1);
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = (__m128)__lsx_vldrepl_d(pC0, 0);
                    if (beta != 1.f)
                    {
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _c = __lsx_vfmul_s(_c, _beta);
                    }
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
            }

            if (output_elemtype == 1)
            {
                __lsx_vstelm_d((__m128i)_f0, p0f, 0, 0);
                __lsx_vstelm_d((__m128i)_f1, p0f + out_hstep, 0, 0);
            }
            else
            {
                __lsx_vstelm_w(float2bfloat_lsx((__m128)((__m128i)_f0)), p0, 0, 0);
                __lsx_vstelm_w(float2bfloat_lsx((__m128)((__m128i)_f1)), p0 + out_hstep, 0, 0);
            }
#else
            float f00 = pp[0];
            float f01 = pp[1];
            float f10 = pp[2];
            float f11 = pp[3];
            if (pC0)
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
                        f00 += pC0[0];
                        f01 += pC0[1];
                        f10 += pC0[c_hstep];
                        f11 += pC0[c_hstep + 1];
                    }
                    else
                    {
                        f00 += pC0[0] * beta;
                        f01 += pC0[1] * beta;
                        f10 += pC0[c_hstep] * beta;
                        f11 += pC0[c_hstep + 1] * beta;
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    const float cc0 = pC0[0] * beta;
                    const float cc1 = pC0[1] * beta;
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

            if (output_elemtype == 1)
            {
                p0f[0] = f00;
                p0f[1] = f01;
                p0f[out_hstep + 0] = f10;
                p0f[out_hstep + 1] = f11;
            }
            else
            {
                p0[0] = float32_to_bfloat16(f00);
                p0[1] = float32_to_bfloat16(f01);
                p0[out_hstep + 0] = float32_to_bfloat16(f10);
                p0[out_hstep + 1] = float32_to_bfloat16(f11);
            }
#endif // __loongarch_sx
            pp += 4;
            if (output_elemtype == 1)
                p0f += 2;
            else
                p0 += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[0];
            float f1 = pp[1];
            pp += 2;
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
                    float c = pC[0] * beta;
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

            if (output_elemtype == 1)
            {
                p0f[0] = f0;
                p0f[out_hstep + 0] = f1;
                p0f++;
            }
            else
            {
                p0[0] = float32_to_bfloat16(f0);
                p0[out_hstep + 0] = float32_to_bfloat16(f1);
                p0++;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        float* p0f = output_elemtype == 1 ? (float*)top_blob + (size_t)(i + ii) * out_hstep + j : 0;
        unsigned short* p0 = output_elemtype == 3 ? (unsigned short*)top_blob + (size_t)(i + ii) * out_hstep + j : 0;

        float c0 = 0.f;
        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC += i + ii;
                c0 = pC[0] * beta;
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
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = (__m256)__lasx_xvld(pp, 0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f0 = __lasx_xvfadd_s(_f0, (__m256)__lasx_xvreplfr2vr_s(c0));
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                    __m256 _c0 = (__m256)__lasx_xvld(pC, 0);
                    if (beta == 1.f)
                        _f0 = __lasx_xvfadd_s(_f0, _c0);
                    else
                        _f0 = __lasx_xvfmadd_s(_c0, _beta, _f0);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m256 _alpha = (__m256)__lasx_xvreplfr2vr_s(alpha);
                _f0 = __lasx_xvfmul_s(_f0, _alpha);
            }

            if (output_elemtype == 1)
            {
                __lasx_xvst(_f0, p0f, 0);
            }
            else
            {
                __lsx_vst(float2bfloat_lasx((__m256)_f0), p0, 0);
            }
            pp += 8;
            if (output_elemtype == 1)
                p0f += 8;
            else
                p0 += 8;
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp, 0);
            pp += 4;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(c0));
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    __m128 _c0 = (__m128)__lsx_vld(pC, 0);
                    pC += 4;
                    if (beta == 1.f)
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                    else
                        _f0 = __lsx_vfmadd_s(_c0, _beta, _f0);
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
            }

            if (output_elemtype == 1)
            {
                __lsx_vst((__m128i)_f0, p0f, 0);
                p0f += 4;
            }
            else
            {
                __lsx_vstelm_d(float2bfloat_lsx((__m128)((__m128i)_f0)), p0, 0, 0);
                p0 += 4;
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _f0 = (__m128)__lsx_vldrepl_d(pp, 0);
            pp += 2;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(c0));
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    __m128 _c0 = (__m128)__lsx_vldrepl_d(pC, 0);
                    pC += 2;
                    if (beta == 1.f)
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                    else
                        _f0 = __lsx_vfmadd_s(_c0, _beta, _f0);
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
            }

            if (output_elemtype == 1)
            {
                __lsx_vstelm_d((__m128i)_f0, p0f, 0, 0);
                p0f += 2;
            }
            else
            {
                __lsx_vstelm_w(float2bfloat_lsx((__m128)((__m128i)_f0)), p0, 0, 0);
                p0 += 2;
            }
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
                    f0 += pC[0] * beta;
                    pC++;
                }
            }

            if (alpha != 1.f)
                f0 *= alpha;
            if (output_elemtype == 1)
            {
                p0f[0] = f0;
                p0f++;
            }
            else
            {
                p0[0] = float32_to_bfloat16(f0);
                p0++;
            }
        }
    }
}

static void transpose_unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta, int output_elemtype)
{
    const float* pp = topT;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const int c_elempack = C.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    const int out_elempack = top_blob.elempack;
    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* p0f = output_elemtype == 1 ? (float*)top_blob + (size_t)j * out_hstep + (i + ii) * out_elempack : 0;
        unsigned short* p0 = output_elemtype == 3 ? (unsigned short*)top_blob + (size_t)j * out_hstep + (i + ii) * out_elempack : 0;
        const float* pC = C;

        __m128 _c0 = __lsx_vreplfr2vr_s(0.f);
        __m128 _c1 = __lsx_vreplfr2vr_s(0.f);
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = __lsx_vreplfr2vr_s(pC[0] * beta);
                _c1 = _c0;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC += i + ii;
                _c0 = (__m128)__lsx_vld(pC, 0);
                _c1 = (__m128)__lsx_vld(pC + 4, 0);
                if (beta != 1.f)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    _c0 = __lsx_vfmul_s(_c0, _beta);
                    _c1 = __lsx_vfmul_s(_c1, _beta);
                }
            }

            if (broadcast_type_C == 3)
            {
                pC += (size_t)(i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC += j;
            }
        }

        int jj = 0;
#if __loongarch_asx
        __m256 _c256 = (__m256)__lasx_concat_128((__m128i)_c0, (__m128i)_c1);
        for (; jj + 7 < max_jj; jj += 8)
        {
            const float* pC0 = pC;
            __m256i _r0 = __lasx_xvld(pp, 0);
            __m256i _r1 = __lasx_xvld(pp + 8, 0);
            __m256i _r2 = __lasx_xvld(pp + 16, 0);
            __m256i _r3 = __lasx_xvld(pp + 24, 0);
            __m256i _r4 = __lasx_xvld(pp + 32, 0);
            __m256i _r5 = __lasx_xvld(pp + 40, 0);
            __m256i _r6 = __lasx_xvld(pp + 48, 0);
            __m256i _r7 = __lasx_xvld(pp + 56, 0);
            __m256i _tmp0 = _r0;
            __m256i _tmp1 = __lasx_xvshuf4i_w(_r1, _LSX_SHUFFLE(2, 1, 0, 3));
            __m256i _tmp2 = _r2;
            __m256i _tmp3 = __lasx_xvshuf4i_w(_r3, _LSX_SHUFFLE(2, 1, 0, 3));
            __m256i _tmp4 = _r4;
            __m256i _tmp5 = __lasx_xvshuf4i_w(_r5, _LSX_SHUFFLE(2, 1, 0, 3));
            __m256i _tmp6 = _r6;
            __m256i _tmp7 = __lasx_xvshuf4i_w(_r7, _LSX_SHUFFLE(2, 1, 0, 3));
            _r0 = __lasx_xvilvl_w(_tmp3, _tmp0);
            _r1 = __lasx_xvilvh_w(_tmp3, _tmp0);
            _r2 = __lasx_xvilvl_w(_tmp1, _tmp2);
            _r3 = __lasx_xvilvh_w(_tmp1, _tmp2);
            _r4 = __lasx_xvilvl_w(_tmp7, _tmp4);
            _r5 = __lasx_xvilvh_w(_tmp7, _tmp4);
            _r6 = __lasx_xvilvl_w(_tmp5, _tmp6);
            _r7 = __lasx_xvilvh_w(_tmp5, _tmp6);
            _tmp0 = __lasx_xvilvl_d(_r2, _r0);
            _tmp1 = __lasx_xvilvh_d(_r2, _r0);
            _tmp2 = __lasx_xvilvl_d(_r1, _r3);
            _tmp3 = __lasx_xvilvh_d(_r1, _r3);
            _tmp4 = __lasx_xvilvl_d(_r6, _r4);
            _tmp5 = __lasx_xvilvh_d(_r6, _r4);
            _tmp6 = __lasx_xvilvl_d(_r5, _r7);
            _tmp7 = __lasx_xvilvh_d(_r5, _r7);
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
            transpose8x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
            if (pC0)
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
                    __m256 _cc0;
                    __m256 _cc1;
                    __m256 _cc2;
                    __m256 _cc3;
                    __m256 _cc4;
                    __m256 _cc5;
                    __m256 _cc6;
                    __m256 _cc7;
                    if (c_elempack == 8)
                    {
                        _cc0 = (__m256)__lasx_xvld(pC0, 0);
                        _cc1 = (__m256)__lasx_xvld(pC0 + 8, 0);
                        _cc2 = (__m256)__lasx_xvld(pC0 + 16, 0);
                        _cc3 = (__m256)__lasx_xvld(pC0 + 24, 0);
                        _cc4 = (__m256)__lasx_xvld(pC0 + 32, 0);
                        _cc5 = (__m256)__lasx_xvld(pC0 + 40, 0);
                        _cc6 = (__m256)__lasx_xvld(pC0 + 48, 0);
                        _cc7 = (__m256)__lasx_xvld(pC0 + 56, 0);
                    }
                    else if (c_elempack == 4)
                    {
                        _cc0 = __lasx_concat_128_s((__m128)__lsx_vld(pC0, 0), (__m128)__lsx_vld(pC0 + c_hstep * 4, 0));
                        _cc1 = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 4, 0), (__m128)__lsx_vld(pC0 + c_hstep * 4 + 4, 0));
                        _cc2 = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 8, 0), (__m128)__lsx_vld(pC0 + c_hstep * 4 + 8, 0));
                        _cc3 = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 12, 0), (__m128)__lsx_vld(pC0 + c_hstep * 4 + 12, 0));
                        _cc4 = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 16, 0), (__m128)__lsx_vld(pC0 + c_hstep * 4 + 16, 0));
                        _cc5 = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 20, 0), (__m128)__lsx_vld(pC0 + c_hstep * 4 + 20, 0));
                        _cc6 = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 24, 0), (__m128)__lsx_vld(pC0 + c_hstep * 4 + 24, 0));
                        _cc7 = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 28, 0), (__m128)__lsx_vld(pC0 + c_hstep * 4 + 28, 0));
                    }
                    else // if (c_elempack == 1)
                    {
                        _cc0 = (__m256)__lasx_xvld(pC0, 0);
                        _cc1 = (__m256)__lasx_xvld(pC0 + c_hstep, 0);
                        _cc2 = (__m256)__lasx_xvld(pC0 + c_hstep * 2, 0);
                        _cc3 = (__m256)__lasx_xvld(pC0 + c_hstep * 3, 0);
                        _cc4 = (__m256)__lasx_xvld(pC0 + c_hstep * 4, 0);
                        _cc5 = (__m256)__lasx_xvld(pC0 + c_hstep * 5, 0);
                        _cc6 = (__m256)__lasx_xvld(pC0 + c_hstep * 6, 0);
                        _cc7 = (__m256)__lasx_xvld(pC0 + c_hstep * 7, 0);
                        transpose8x8_ps(_cc0, _cc1, _cc2, _cc3, _cc4, _cc5, _cc6, _cc7);
                    }
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
                        __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                        _f0 = __lasx_xvfmadd_s(_cc0, _beta, _f0);
                        _f1 = __lasx_xvfmadd_s(_cc1, _beta, _f1);
                        _f2 = __lasx_xvfmadd_s(_cc2, _beta, _f2);
                        _f3 = __lasx_xvfmadd_s(_cc3, _beta, _f3);
                        _f4 = __lasx_xvfmadd_s(_cc4, _beta, _f4);
                        _f5 = __lasx_xvfmadd_s(_cc5, _beta, _f5);
                        _f6 = __lasx_xvfmadd_s(_cc6, _beta, _f6);
                        _f7 = __lasx_xvfmadd_s(_cc7, _beta, _f7);
                    }
                    pC += 8 * c_elempack;
                }
                if (broadcast_type_C == 4)
                {
                    _f0 = __lasx_xvfadd_s(_f0, (__m256)__lasx_xvreplfr2vr_s(pC0[0] * beta));
                    _f1 = __lasx_xvfadd_s(_f1, (__m256)__lasx_xvreplfr2vr_s(pC0[1] * beta));
                    _f2 = __lasx_xvfadd_s(_f2, (__m256)__lasx_xvreplfr2vr_s(pC0[2] * beta));
                    _f3 = __lasx_xvfadd_s(_f3, (__m256)__lasx_xvreplfr2vr_s(pC0[3] * beta));
                    _f4 = __lasx_xvfadd_s(_f4, (__m256)__lasx_xvreplfr2vr_s(pC0[4] * beta));
                    _f5 = __lasx_xvfadd_s(_f5, (__m256)__lasx_xvreplfr2vr_s(pC0[5] * beta));
                    _f6 = __lasx_xvfadd_s(_f6, (__m256)__lasx_xvreplfr2vr_s(pC0[6] * beta));
                    _f7 = __lasx_xvfadd_s(_f7, (__m256)__lasx_xvreplfr2vr_s(pC0[7] * beta));
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m256 _alpha = (__m256)__lasx_xvreplfr2vr_s(alpha);
                _f0 = __lasx_xvfmul_s(_f0, _alpha);
                _f1 = __lasx_xvfmul_s(_f1, _alpha);
                _f2 = __lasx_xvfmul_s(_f2, _alpha);
                _f3 = __lasx_xvfmul_s(_f3, _alpha);
                _f4 = __lasx_xvfmul_s(_f4, _alpha);
                _f5 = __lasx_xvfmul_s(_f5, _alpha);
                _f6 = __lasx_xvfmul_s(_f6, _alpha);
                _f7 = __lasx_xvfmul_s(_f7, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (out_elempack == 8)
                {
                    transpose8x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                    __lasx_xvst(_f0, p0f, 0);
                    __lasx_xvst(_f1, p0f + 8, 0);
                    __lasx_xvst(_f2, p0f + 16, 0);
                    __lasx_xvst(_f3, p0f + 24, 0);
                    __lasx_xvst(_f4, p0f + 32, 0);
                    __lasx_xvst(_f5, p0f + 40, 0);
                    __lasx_xvst(_f6, p0f + 48, 0);
                    __lasx_xvst(_f7, p0f + 56, 0);
                }
                if (out_elempack == 4)
                {
                    float* p1f = p0f + out_hstep * 4;
                    transpose8x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                    __lsx_vst(__lasx_extract_128_lo((__m256i)_f0), p0f, 0);
                    __lsx_vst(__lasx_extract_128_lo((__m256i)_f1), p0f + 4, 0);
                    __lsx_vst(__lasx_extract_128_lo((__m256i)_f2), p0f + 8, 0);
                    __lsx_vst(__lasx_extract_128_lo((__m256i)_f3), p0f + 12, 0);
                    __lsx_vst(__lasx_extract_128_lo((__m256i)_f4), p0f + 16, 0);
                    __lsx_vst(__lasx_extract_128_lo((__m256i)_f5), p0f + 20, 0);
                    __lsx_vst(__lasx_extract_128_lo((__m256i)_f6), p0f + 24, 0);
                    __lsx_vst(__lasx_extract_128_lo((__m256i)_f7), p0f + 28, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_f0), p1f, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_f1), p1f + 4, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_f2), p1f + 8, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_f3), p1f + 12, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_f4), p1f + 16, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_f5), p1f + 20, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_f6), p1f + 24, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_f7), p1f + 28, 0);
                }
                if (out_elempack == 1)
                {
                    __lasx_xvst(_f0, p0f, 0);
                    __lasx_xvst(_f1, p0f + out_hstep, 0);
                    __lasx_xvst(_f2, p0f + out_hstep * 2, 0);
                    __lasx_xvst(_f3, p0f + out_hstep * 3, 0);
                    __lasx_xvst(_f4, p0f + out_hstep * 4, 0);
                    __lasx_xvst(_f5, p0f + out_hstep * 5, 0);
                    __lasx_xvst(_f6, p0f + out_hstep * 6, 0);
                    __lasx_xvst(_f7, p0f + out_hstep * 7, 0);
                }
                p0f += out_hstep * 8;
            }
            else
            {
                if (out_elempack == 8)
                {
                    transpose8x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                    __lsx_vst(float2bfloat_lasx(_f0), p0, 0);
                    __lsx_vst(float2bfloat_lasx(_f1), p0 + 8, 0);
                    __lsx_vst(float2bfloat_lasx(_f2), p0 + 16, 0);
                    __lsx_vst(float2bfloat_lasx(_f3), p0 + 24, 0);
                    __lsx_vst(float2bfloat_lasx(_f4), p0 + 32, 0);
                    __lsx_vst(float2bfloat_lasx(_f5), p0 + 40, 0);
                    __lsx_vst(float2bfloat_lasx(_f6), p0 + 48, 0);
                    __lsx_vst(float2bfloat_lasx(_f7), p0 + 56, 0);
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    transpose8x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                    __m128i _q0 = float2bfloat_lasx(_f0);
                    __m128i _q1 = float2bfloat_lasx(_f1);
                    __m128i _q2 = float2bfloat_lasx(_f2);
                    __m128i _q3 = float2bfloat_lasx(_f3);
                    __m128i _q4 = float2bfloat_lasx(_f4);
                    __m128i _q5 = float2bfloat_lasx(_f5);
                    __m128i _q6 = float2bfloat_lasx(_f6);
                    __m128i _q7 = float2bfloat_lasx(_f7);
                    __lsx_vstelm_d(_q0, p0, 0, 0);
                    __lsx_vstelm_d(_q1, p0 + 4, 0, 0);
                    __lsx_vstelm_d(_q2, p0 + 8, 0, 0);
                    __lsx_vstelm_d(_q3, p0 + 12, 0, 0);
                    __lsx_vstelm_d(_q4, p0 + 16, 0, 0);
                    __lsx_vstelm_d(_q5, p0 + 20, 0, 0);
                    __lsx_vstelm_d(_q6, p0 + 24, 0, 0);
                    __lsx_vstelm_d(_q7, p0 + 28, 0, 0);
                    __lsx_vstelm_d(_q0, p1, 0, 1);
                    __lsx_vstelm_d(_q1, p1 + 4, 0, 1);
                    __lsx_vstelm_d(_q2, p1 + 8, 0, 1);
                    __lsx_vstelm_d(_q3, p1 + 12, 0, 1);
                    __lsx_vstelm_d(_q4, p1 + 16, 0, 1);
                    __lsx_vstelm_d(_q5, p1 + 20, 0, 1);
                    __lsx_vstelm_d(_q6, p1 + 24, 0, 1);
                    __lsx_vstelm_d(_q7, p1 + 28, 0, 1);
                }
                if (out_elempack == 1)
                {
                    __lsx_vst(float2bfloat_lasx(_f0), p0, 0);
                    __lsx_vst(float2bfloat_lasx(_f1), p0 + out_hstep, 0);
                    __lsx_vst(float2bfloat_lasx(_f2), p0 + out_hstep * 2, 0);
                    __lsx_vst(float2bfloat_lasx(_f3), p0 + out_hstep * 3, 0);
                    __lsx_vst(float2bfloat_lasx(_f4), p0 + out_hstep * 4, 0);
                    __lsx_vst(float2bfloat_lasx(_f5), p0 + out_hstep * 5, 0);
                    __lsx_vst(float2bfloat_lasx(_f6), p0 + out_hstep * 6, 0);
                    __lsx_vst(float2bfloat_lasx(_f7), p0 + out_hstep * 7, 0);
                }
                p0 += out_hstep * 8;
            }
            pp += 64;
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _r0 = __lsx_vld(pp, 0);
            __m128i _r1 = __lsx_vld(pp + 8, 0);
            __m128i _r2 = __lsx_vld(pp + 16, 0);
            __m128i _r3 = __lsx_vld(pp + 24, 0);
            _r2 = __lsx_vshuf4i_w(_r2, _LSX_SHUFFLE(1, 0, 3, 2));
            _r3 = __lsx_vshuf4i_w(_r3, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_r0, _r1, _r2, _r3);
            _r1 = __lsx_vshuf4i_w(_r1, _LSX_SHUFFLE(2, 1, 0, 3));
            _r2 = __lsx_vshuf4i_w(_r2, _LSX_SHUFFLE(1, 0, 3, 2));
            _r3 = __lsx_vshuf4i_w(_r3, _LSX_SHUFFLE(0, 3, 2, 1));
            __m128i _r4 = __lsx_vld(pp + 4, 0);
            __m128i _r5 = __lsx_vld(pp + 12, 0);
            __m128i _r6 = __lsx_vld(pp + 20, 0);
            __m128i _r7 = __lsx_vld(pp + 28, 0);
            pp += 32;
            _r6 = __lsx_vshuf4i_w(_r6, _LSX_SHUFFLE(1, 0, 3, 2));
            _r7 = __lsx_vshuf4i_w(_r7, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_r4, _r5, _r6, _r7);
            _r5 = __lsx_vshuf4i_w(_r5, _LSX_SHUFFLE(2, 1, 0, 3));
            _r6 = __lsx_vshuf4i_w(_r6, _LSX_SHUFFLE(1, 0, 3, 2));
            _r7 = __lsx_vshuf4i_w(_r7, _LSX_SHUFFLE(0, 3, 2, 1));
            __m128 _f0 = (__m128)_r0;
            __m128 _f1 = (__m128)_r1;
            __m128 _f2 = (__m128)_r2;
            __m128 _f3 = (__m128)_r3;
            __m128 _f4 = (__m128)_r4;
            __m128 _f5 = (__m128)_r5;
            __m128 _f6 = (__m128)_r6;
            __m128 _f7 = (__m128)_r7;
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
                    __m128 _cc0;
                    __m128 _cc1;
                    __m128 _cc2;
                    __m128 _cc3;
                    __m128 _cc4;
                    __m128 _cc5;
                    __m128 _cc6;
                    __m128 _cc7;
                    if (c_elempack == 8)
                    {
                        _cc0 = (__m128)__lsx_vld(pC, 0);
                        _cc4 = (__m128)__lsx_vld(pC + 4, 0);
                        _cc1 = (__m128)__lsx_vld(pC + 8, 0);
                        _cc5 = (__m128)__lsx_vld(pC + 12, 0);
                        _cc2 = (__m128)__lsx_vld(pC + 16, 0);
                        _cc6 = (__m128)__lsx_vld(pC + 20, 0);
                        _cc3 = (__m128)__lsx_vld(pC + 24, 0);
                        _cc7 = (__m128)__lsx_vld(pC + 28, 0);
                    }
                    else if (c_elempack == 4)
                    {
                        _cc0 = (__m128)__lsx_vld(pC, 0);
                        _cc1 = (__m128)__lsx_vld(pC + 4, 0);
                        _cc2 = (__m128)__lsx_vld(pC + 8, 0);
                        _cc3 = (__m128)__lsx_vld(pC + 12, 0);
                        _cc4 = (__m128)__lsx_vld(pC + c_hstep * 4, 0);
                        _cc5 = (__m128)__lsx_vld(pC + c_hstep * 4 + 4, 0);
                        _cc6 = (__m128)__lsx_vld(pC + c_hstep * 4 + 8, 0);
                        _cc7 = (__m128)__lsx_vld(pC + c_hstep * 4 + 12, 0);
                    }
                    else // if (c_elempack == 1)
                    {
                        _cc0 = (__m128)__lsx_vld(pC, 0);
                        _cc1 = (__m128)__lsx_vld(pC + c_hstep, 0);
                        _cc2 = (__m128)__lsx_vld(pC + c_hstep * 2, 0);
                        _cc3 = (__m128)__lsx_vld(pC + c_hstep * 3, 0);
                        _cc4 = (__m128)__lsx_vld(pC + c_hstep * 4, 0);
                        _cc5 = (__m128)__lsx_vld(pC + c_hstep * 5, 0);
                        _cc6 = (__m128)__lsx_vld(pC + c_hstep * 6, 0);
                        _cc7 = (__m128)__lsx_vld(pC + c_hstep * 7, 0);
                        transpose4x4_ps(_cc0, _cc1, _cc2, _cc3);
                        transpose4x4_ps(_cc4, _cc5, _cc6, _cc7);
                    }
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
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _f0 = __lsx_vfmadd_s(_cc0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s(_cc1, _beta, _f1);
                        _f2 = __lsx_vfmadd_s(_cc2, _beta, _f2);
                        _f3 = __lsx_vfmadd_s(_cc3, _beta, _f3);
                        _f4 = __lsx_vfmadd_s(_cc4, _beta, _f4);
                        _f5 = __lsx_vfmadd_s(_cc5, _beta, _f5);
                        _f6 = __lsx_vfmadd_s(_cc6, _beta, _f6);
                        _f7 = __lsx_vfmadd_s(_cc7, _beta, _f7);
                    }
                    pC += 4 * c_elempack;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _cc = (__m128)__lsx_vld(pC, 0);
                    if (beta != 1.f)
                    {
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _cc = __lsx_vfmul_s(_cc, _beta);
                    }
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
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
                _f2 = __lsx_vfmul_s(_f2, _alpha);
                _f3 = __lsx_vfmul_s(_f3, _alpha);
                _f4 = __lsx_vfmul_s(_f4, _alpha);
                _f5 = __lsx_vfmul_s(_f5, _alpha);
                _f6 = __lsx_vfmul_s(_f6, _alpha);
                _f7 = __lsx_vfmul_s(_f7, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __lsx_vst((__m128i)_f0, p0f, 0);
                    __lsx_vst((__m128i)_f1, p0f + 4, 0);
                    __lsx_vst((__m128i)_f2, p0f + 8, 0);
                    __lsx_vst((__m128i)_f3, p0f + 12, 0);
                    __lsx_vst((__m128i)_f4, p0f + 16, 0);
                    __lsx_vst((__m128i)_f5, p0f + 20, 0);
                    __lsx_vst((__m128i)_f6, p0f + 24, 0);
                    __lsx_vst((__m128i)_f7, p0f + 28, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vst((__m128i)_f0, p0f, 0);
                    __lsx_vst((__m128i)_f4, p0f + 4, 0);
                    __lsx_vst((__m128i)_f1, p0f + out_hstep, 0);
                    __lsx_vst((__m128i)_f5, p0f + out_hstep + 4, 0);
                    __lsx_vst((__m128i)_f2, p0f + out_hstep * 2, 0);
                    __lsx_vst((__m128i)_f6, p0f + out_hstep * 2 + 4, 0);
                    __lsx_vst((__m128i)_f3, p0f + out_hstep * 3, 0);
                    __lsx_vst((__m128i)_f7, p0f + out_hstep * 3 + 4, 0);
                }
                p0f += out_hstep * 4;
            }
            else
            {
                if (out_elempack == 8)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    const int out_lane = jj % 8;
                    __lsx_vstelm_d(float2bfloat_lsx(_f0), p0 + out_lane, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f1), p0 + 8 + out_lane, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f2), p0 + 16 + out_lane, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f3), p0 + 24 + out_lane, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f4), p0 + 32 + out_lane, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f5), p0 + 40 + out_lane, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f6), p0 + 48 + out_lane, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f7), p0 + 56 + out_lane, 0, 0);
                }
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    transpose4x4_ps(_f4, _f5, _f6, _f7);
                    __lsx_vstelm_d(float2bfloat_lsx(_f0), p0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f1), p0 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f2), p0 + 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f3), p0 + 12, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f4), p0 + 16, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f5), p0 + 20, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f6), p0 + 24, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f7), p0 + 28, 0, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vstelm_d(float2bfloat_lsx(_f0), p0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f4), p0 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f1), p0 + out_hstep, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f5), p0 + out_hstep + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f2), p0 + out_hstep * 2, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f6), p0 + out_hstep * 2 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f3), p0 + out_hstep * 3, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f7), p0 + out_hstep * 3 + 4, 0, 0);
                }
                if (out_elempack == 8)
                {
                    if (jj % 8 == 4)
                        p0 += out_hstep * 8;
                }
                else
                {
                    p0 += out_hstep * 4;
                }
            }
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
                    __m128i _ci0;
                    __m128i _ci1;
                    __m128i _ci2;
                    __m128i _ci3;
                    if (c_elempack == 8)
                    {
                        _ci0 = __lsx_vld(pC, 0);
                        _ci1 = __lsx_vld(pC + 4, 0);
                        _ci2 = __lsx_vld(pC + 8, 0);
                        _ci3 = __lsx_vld(pC + 12, 0);
                    }
                    else if (c_elempack == 4)
                    {
                        _ci0 = __lsx_vld(pC, 0);
                        _ci1 = __lsx_vld(pC + c_hstep * 4, 0);
                        _ci2 = __lsx_vld(pC + 4, 0);
                        _ci3 = __lsx_vld(pC + c_hstep * 4 + 4, 0);
                    }
                    else // if (c_elempack == 1)
                    {
                        _ci0 = __lsx_vreplgr2vr_w(((const int*)pC)[0]);
                        _ci0 = __lsx_vinsgr2vr_w(_ci0, ((const int*)(pC + c_hstep))[0], 1);
                        _ci0 = __lsx_vinsgr2vr_w(_ci0, ((const int*)(pC + c_hstep * 2))[0], 2);
                        _ci0 = __lsx_vinsgr2vr_w(_ci0, ((const int*)(pC + c_hstep * 3))[0], 3);
                        _ci1 = __lsx_vreplgr2vr_w(((const int*)(pC + c_hstep * 4))[0]);
                        _ci1 = __lsx_vinsgr2vr_w(_ci1, ((const int*)(pC + c_hstep * 5))[0], 1);
                        _ci1 = __lsx_vinsgr2vr_w(_ci1, ((const int*)(pC + c_hstep * 6))[0], 2);
                        _ci1 = __lsx_vinsgr2vr_w(_ci1, ((const int*)(pC + c_hstep * 7))[0], 3);
                        _ci2 = __lsx_vreplgr2vr_w(((const int*)pC)[1]);
                        _ci2 = __lsx_vinsgr2vr_w(_ci2, ((const int*)(pC + c_hstep))[1], 1);
                        _ci2 = __lsx_vinsgr2vr_w(_ci2, ((const int*)(pC + c_hstep * 2))[1], 2);
                        _ci2 = __lsx_vinsgr2vr_w(_ci2, ((const int*)(pC + c_hstep * 3))[1], 3);
                        _ci3 = __lsx_vreplgr2vr_w(((const int*)(pC + c_hstep * 4))[1]);
                        _ci3 = __lsx_vinsgr2vr_w(_ci3, ((const int*)(pC + c_hstep * 5))[1], 1);
                        _ci3 = __lsx_vinsgr2vr_w(_ci3, ((const int*)(pC + c_hstep * 6))[1], 2);
                        _ci3 = __lsx_vinsgr2vr_w(_ci3, ((const int*)(pC + c_hstep * 7))[1], 3);
                    }
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, (__m128)_ci0);
                        _f1 = __lsx_vfadd_s(_f1, (__m128)_ci1);
                        _f2 = __lsx_vfadd_s(_f2, (__m128)_ci2);
                        _f3 = __lsx_vfadd_s(_f3, (__m128)_ci3);
                    }
                    else
                    {
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _f0 = __lsx_vfmadd_s((__m128)_ci0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s((__m128)_ci1, _beta, _f1);
                        _f2 = __lsx_vfmadd_s((__m128)_ci2, _beta, _f2);
                        _f3 = __lsx_vfmadd_s((__m128)_ci3, _beta, _f3);
                    }
                    pC += 2 * c_elempack;
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
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
                _f2 = __lsx_vfmul_s(_f2, _alpha);
                _f3 = __lsx_vfmul_s(_f3, _alpha);
            }

            if (output_elemtype == 1)
            {
                __lsx_vst((__m128i)_f0, p0f, 0);
                __lsx_vst((__m128i)_f1, p0f + 4, 0);
                __lsx_vst((__m128i)_f2, p0f + out_hstep, 0);
                __lsx_vst((__m128i)_f3, p0f + out_hstep + 4, 0);
                p0f += out_hstep * 2;
            }
            else
            {
                __lsx_vstelm_d(float2bfloat_lsx((__m128)((__m128i)_f0)), p0, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)((__m128i)_f1)), p0 + 4, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)((__m128i)_f2)), p0 + out_hstep, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)((__m128i)_f3)), p0 + out_hstep + 4, 0, 0);
                p0 += out_hstep * 2;
            }
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
                    __m128i _ci0;
                    __m128i _ci1;
                    if (c_elempack == 8)
                    {
                        _ci0 = __lsx_vld(pC, 0);
                        _ci1 = __lsx_vld(pC + 4, 0);
                    }
                    else if (c_elempack == 4)
                    {
                        _ci0 = __lsx_vld(pC, 0);
                        _ci1 = __lsx_vld(pC + c_hstep * 4, 0);
                    }
                    else // if (c_elempack == 1)
                    {
                        _ci0 = __lsx_vreplgr2vr_w(((const int*)pC)[0]);
                        _ci0 = __lsx_vinsgr2vr_w(_ci0, ((const int*)(pC + c_hstep))[0], 1);
                        _ci0 = __lsx_vinsgr2vr_w(_ci0, ((const int*)(pC + c_hstep * 2))[0], 2);
                        _ci0 = __lsx_vinsgr2vr_w(_ci0, ((const int*)(pC + c_hstep * 3))[0], 3);
                        _ci1 = __lsx_vreplgr2vr_w(((const int*)(pC + c_hstep * 4))[0]);
                        _ci1 = __lsx_vinsgr2vr_w(_ci1, ((const int*)(pC + c_hstep * 5))[0], 1);
                        _ci1 = __lsx_vinsgr2vr_w(_ci1, ((const int*)(pC + c_hstep * 6))[0], 2);
                        _ci1 = __lsx_vinsgr2vr_w(_ci1, ((const int*)(pC + c_hstep * 7))[0], 3);
                    }
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, (__m128)_ci0);
                        _f1 = __lsx_vfadd_s(_f1, (__m128)_ci1);
                    }
                    else
                    {
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _f0 = __lsx_vfmadd_s((__m128)_ci0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s((__m128)_ci1, _beta, _f1);
                    }
                    pC += c_elempack;
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
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
            }

            if (output_elemtype == 1)
            {
                __lsx_vst((__m128i)_f0, p0f, 0);
                __lsx_vst((__m128i)_f1, p0f + 4, 0);
                p0f += out_hstep;
            }
            else
            {
                __lsx_vstelm_d(float2bfloat_lsx((__m128)((__m128i)_f0)), p0, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)((__m128i)_f1)), p0 + 4, 0, 0);
                p0 += out_hstep;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* p0f = output_elemtype == 1 ? (float*)top_blob + (size_t)j * out_hstep + (i + ii) * out_elempack : 0;
        unsigned short* p0 = output_elemtype == 3 ? (unsigned short*)top_blob + (size_t)j * out_hstep + (i + ii) * out_elempack : 0;
        const float* pC = C;

        __m128 _c = __lsx_vreplfr2vr_s(0.f);
        if (pC)
        {
            if (broadcast_type_C == 0)
                _c = __lsx_vreplfr2vr_s(pC[0] * beta);
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC += i + ii;
                _c = (__m128)__lsx_vld(pC, 0);
                if (beta != 1.f)
                    _c = __lsx_vfmul_s(_c, __lsx_vreplfr2vr_s(beta));
            }

            if (broadcast_type_C == 3)
            {
                pC += (size_t)(i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC += j;
            }
        }

        int jj = 0;
#if __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const float* pC0 = pC;
            __m256i _s0 = __lasx_xvld(pp, 0);
            __m256i _s1 = __lasx_xvld(pp + 8, 0);
            __m256i _s2 = __lasx_xvld(pp + 16, 0);
            __m256i _s3 = __lasx_xvld(pp + 24, 0);

            _s2 = __lasx_xvshuf4i_w(_s2, _LSX_SHUFFLE(1, 0, 3, 2));
            _s3 = __lasx_xvshuf4i_w(_s3, _LSX_SHUFFLE(1, 0, 3, 2));
            {
                __m256i _tmp0 = __lasx_xvilvl_w(_s1, _s0);
                __m256i _tmp1 = __lasx_xvilvh_w(_s1, _s0);
                __m256i _tmp2 = __lasx_xvilvl_w(_s3, _s2);
                __m256i _tmp3 = __lasx_xvilvh_w(_s3, _s2);
                _s0 = __lasx_xvilvl_d(_tmp2, _tmp0);
                _s1 = __lasx_xvilvh_d(_tmp2, _tmp0);
                _s2 = __lasx_xvilvl_d(_tmp3, _tmp1);
                _s3 = __lasx_xvilvh_d(_tmp3, _tmp1);
            }
            _s1 = __lasx_xvshuf4i_w(_s1, _LSX_SHUFFLE(2, 1, 0, 3));
            _s2 = __lasx_xvshuf4i_w(_s2, _LSX_SHUFFLE(1, 0, 3, 2));
            _s3 = __lasx_xvshuf4i_w(_s3, _LSX_SHUFFLE(0, 3, 2, 1));

            __m256 _f0 = (__m256)_s0;
            __m256 _f1 = (__m256)_s1;
            __m256 _f2 = (__m256)_s2;
            __m256 _f3 = (__m256)_s3;
            if (pC0)
            {
                if (broadcast_type_C == 0)
                {
                    __m256 _c0 = (__m256)__lasx_xvreplfr2vr_s(pC0[0] * beta);
                    _f0 = __lasx_xvfadd_s(_f0, _c0);
                    _f1 = __lasx_xvfadd_s(_f1, _c0);
                    _f2 = __lasx_xvfadd_s(_f2, _c0);
                    _f3 = __lasx_xvfadd_s(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lasx_xvfadd_s(_f0, (__m256)__lasx_xvreplfr2vr_s(pC0[0] * beta));
                    _f1 = __lasx_xvfadd_s(_f1, (__m256)__lasx_xvreplfr2vr_s(pC0[1] * beta));
                    _f2 = __lasx_xvfadd_s(_f2, (__m256)__lasx_xvreplfr2vr_s(pC0[2] * beta));
                    _f3 = __lasx_xvfadd_s(_f3, (__m256)__lasx_xvreplfr2vr_s(pC0[3] * beta));
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _cc0;
                    __m256 _cc1;
                    __m256 _cc2;
                    __m256 _cc3;
                    if (c_elempack == 4)
                    {
                        __m128 _c00 = (__m128)__lsx_vld(pC0, 0);
                        __m128 _c10 = (__m128)__lsx_vld(pC0 + 4, 0);
                        __m128 _c20 = (__m128)__lsx_vld(pC0 + 8, 0);
                        __m128 _c30 = (__m128)__lsx_vld(pC0 + 12, 0);
                        transpose4x4_ps(_c00, _c10, _c20, _c30);

                        __m128 _c01 = (__m128)__lsx_vld(pC0 + 16, 0);
                        __m128 _c11 = (__m128)__lsx_vld(pC0 + 20, 0);
                        __m128 _c21 = (__m128)__lsx_vld(pC0 + 24, 0);
                        __m128 _c31 = (__m128)__lsx_vld(pC0 + 28, 0);
                        transpose4x4_ps(_c01, _c11, _c21, _c31);

                        _cc0 = __lasx_concat_128_s(_c00, _c01);
                        _cc1 = __lasx_concat_128_s(_c10, _c11);
                        _cc2 = __lasx_concat_128_s(_c20, _c21);
                        _cc3 = __lasx_concat_128_s(_c30, _c31);
                    }
                    else // if (c_elempack == 1)
                    {
                        _cc0 = (__m256)__lasx_xvld(pC0, 0);
                        _cc1 = (__m256)__lasx_xvld(pC0 + c_hstep, 0);
                        _cc2 = (__m256)__lasx_xvld(pC0 + c_hstep * 2, 0);
                        _cc3 = (__m256)__lasx_xvld(pC0 + c_hstep * 3, 0);
                    }
                    if (beta == 1.f)
                    {
                        _f0 = __lasx_xvfadd_s(_f0, _cc0);
                        _f1 = __lasx_xvfadd_s(_f1, _cc1);
                        _f2 = __lasx_xvfadd_s(_f2, _cc2);
                        _f3 = __lasx_xvfadd_s(_f3, _cc3);
                    }
                    else
                    {
                        __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                        _f0 = __lasx_xvfmadd_s(_cc0, _beta, _f0);
                        _f1 = __lasx_xvfmadd_s(_cc1, _beta, _f1);
                        _f2 = __lasx_xvfmadd_s(_cc2, _beta, _f2);
                        _f3 = __lasx_xvfmadd_s(_cc3, _beta, _f3);
                    }
                    pC += 8 * c_elempack;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c4 = (__m256)__lasx_xvld(pC0, 0);
                    if (beta != 1.f)
                    {
                        __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                        _c4 = __lasx_xvfmul_s(_c4, _beta);
                    }
                    _f0 = __lasx_xvfadd_s(_f0, _c4);
                    _f1 = __lasx_xvfadd_s(_f1, _c4);
                    _f2 = __lasx_xvfadd_s(_f2, _c4);
                    _f3 = __lasx_xvfadd_s(_f3, _c4);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m256 _alpha = (__m256)__lasx_xvreplfr2vr_s(alpha);
                _f0 = __lasx_xvfmul_s(_f0, _alpha);
                _f1 = __lasx_xvfmul_s(_f1, _alpha);
                _f2 = __lasx_xvfmul_s(_f2, _alpha);
                _f3 = __lasx_xvfmul_s(_f3, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (out_elempack == 8)
                {
                    __lasx_xvst(_f0, p0f, 0);
                    __lasx_xvst(_f1, p0f + 8, 0);
                    __lasx_xvst(_f2, p0f + 16, 0);
                    __lasx_xvst(_f3, p0f + 24, 0);
                }
                if (out_elempack == 4)
                {
                    float* p1f = p0f + out_hstep * 4;
                    __lsx_vst(__lasx_extract_128_lo((__m256i)_f0), p0f, 0);
                    __lsx_vst(__lasx_extract_128_lo((__m256i)_f1), p0f + 4, 0);
                    __lsx_vst(__lasx_extract_128_lo((__m256i)_f2), p0f + 8, 0);
                    __lsx_vst(__lasx_extract_128_lo((__m256i)_f3), p0f + 12, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_f0), p1f, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_f1), p1f + 4, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_f2), p1f + 8, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_f3), p1f + 12, 0);
                }
                if (out_elempack == 1)
                {
                    __m256i _tmp0 = __lasx_xvilvl_w((__m256i)_f1, (__m256i)_f0);
                    __m256i _tmp1 = __lasx_xvilvh_w((__m256i)_f1, (__m256i)_f0);
                    __m256i _tmp2 = __lasx_xvilvl_w((__m256i)_f3, (__m256i)_f2);
                    __m256i _tmp3 = __lasx_xvilvh_w((__m256i)_f3, (__m256i)_f2);
                    __m256i _r0 = __lasx_xvilvl_d(_tmp2, _tmp0);
                    __m256i _r1 = __lasx_xvilvh_d(_tmp2, _tmp0);
                    __m256i _r2 = __lasx_xvilvl_d(_tmp3, _tmp1);
                    __m256i _r3 = __lasx_xvilvh_d(_tmp3, _tmp1);
                    __lsx_vst(__lasx_extract_128_lo(_r0), p0f, 0);
                    __lsx_vst(__lasx_extract_128_lo(_r1), p0f + out_hstep, 0);
                    __lsx_vst(__lasx_extract_128_lo(_r2), p0f + out_hstep * 2, 0);
                    __lsx_vst(__lasx_extract_128_lo(_r3), p0f + out_hstep * 3, 0);
                    __lsx_vst(__lasx_extract_128_hi(_r0), p0f + out_hstep * 4, 0);
                    __lsx_vst(__lasx_extract_128_hi(_r1), p0f + out_hstep * 5, 0);
                    __lsx_vst(__lasx_extract_128_hi(_r2), p0f + out_hstep * 6, 0);
                    __lsx_vst(__lasx_extract_128_hi(_r3), p0f + out_hstep * 7, 0);
                }
            }
            else
            {
                if (out_elempack == 8)
                {
                    __lsx_vst(float2bfloat_lasx(_f0), p0, 0);
                    __lsx_vst(float2bfloat_lasx(_f1), p0 + 8, 0);
                    __lsx_vst(float2bfloat_lasx(_f2), p0 + 16, 0);
                    __lsx_vst(float2bfloat_lasx(_f3), p0 + 24, 0);
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    __m128i _q0 = float2bfloat_lasx(_f0);
                    __m128i _q1 = float2bfloat_lasx(_f1);
                    __m128i _q2 = float2bfloat_lasx(_f2);
                    __m128i _q3 = float2bfloat_lasx(_f3);
                    __lsx_vstelm_d(_q0, p0, 0, 0);
                    __lsx_vstelm_d(_q1, p0 + 4, 0, 0);
                    __lsx_vstelm_d(_q2, p0 + 8, 0, 0);
                    __lsx_vstelm_d(_q3, p0 + 12, 0, 0);
                    __lsx_vstelm_d(_q0, p1, 0, 1);
                    __lsx_vstelm_d(_q1, p1 + 4, 0, 1);
                    __lsx_vstelm_d(_q2, p1 + 8, 0, 1);
                    __lsx_vstelm_d(_q3, p1 + 12, 0, 1);
                }
                if (out_elempack == 1)
                {
                    __m256i _tmp0 = __lasx_xvilvl_w((__m256i)_f1, (__m256i)_f0);
                    __m256i _tmp1 = __lasx_xvilvh_w((__m256i)_f1, (__m256i)_f0);
                    __m256i _tmp2 = __lasx_xvilvl_w((__m256i)_f3, (__m256i)_f2);
                    __m256i _tmp3 = __lasx_xvilvh_w((__m256i)_f3, (__m256i)_f2);
                    __m256i _r0 = __lasx_xvilvl_d(_tmp2, _tmp0);
                    __m256i _r1 = __lasx_xvilvh_d(_tmp2, _tmp0);
                    __m256i _r2 = __lasx_xvilvl_d(_tmp3, _tmp1);
                    __m256i _r3 = __lasx_xvilvh_d(_tmp3, _tmp1);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)(__lasx_extract_128_lo(_r0))), p0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)(__lasx_extract_128_lo(_r1))), p0 + out_hstep, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)(__lasx_extract_128_lo(_r2))), p0 + out_hstep * 2, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)(__lasx_extract_128_lo(_r3))), p0 + out_hstep * 3, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)(__lasx_extract_128_hi(_r0))), p0 + out_hstep * 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)(__lasx_extract_128_hi(_r1))), p0 + out_hstep * 5, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)(__lasx_extract_128_hi(_r2))), p0 + out_hstep * 6, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)(__lasx_extract_128_hi(_r3))), p0 + out_hstep * 7, 0, 0);
                }
            }
            pp += 32;
            if (output_elemtype == 1)
                p0f += out_hstep * 8;
            else
                p0 += out_hstep * 8;
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _s0 = __lsx_vld(pp, 0);
            __m128i _s1 = __lsx_vld(pp + 4, 0);
            __m128i _s2 = __lsx_vld(pp + 8, 0);
            __m128i _s3 = __lsx_vld(pp + 12, 0);
            pp += 16;

            _s2 = __lsx_vshuf4i_w(_s2, _LSX_SHUFFLE(1, 0, 3, 2));
            _s3 = __lsx_vshuf4i_w(_s3, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_s0, _s1, _s2, _s3);
            _s1 = __lsx_vshuf4i_w(_s1, _LSX_SHUFFLE(2, 1, 0, 3));
            _s2 = __lsx_vshuf4i_w(_s2, _LSX_SHUFFLE(1, 0, 3, 2));
            _s3 = __lsx_vshuf4i_w(_s3, _LSX_SHUFFLE(0, 3, 2, 1));

            __m128 _f0 = (__m128)_s0;
            __m128 _f1 = (__m128)_s1;
            __m128 _f2 = (__m128)_s2;
            __m128 _f3 = (__m128)_s3;
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
                    __m128 _c0;
                    __m128 _c1;
                    __m128 _c2;
                    __m128 _c3;
                    if (c_elempack == 4)
                    {
                        _c0 = (__m128)__lsx_vld(pC, 0);
                        _c1 = (__m128)__lsx_vld(pC + 4, 0);
                        _c2 = (__m128)__lsx_vld(pC + 8, 0);
                        _c3 = (__m128)__lsx_vld(pC + 12, 0);
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = (__m128)__lsx_vld(pC, 0);
                        _c1 = (__m128)__lsx_vld(pC + c_hstep, 0);
                        _c2 = (__m128)__lsx_vld(pC + c_hstep * 2, 0);
                        _c3 = (__m128)__lsx_vld(pC + c_hstep * 3, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                    }
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                        _f1 = __lsx_vfadd_s(_f1, _c1);
                        _f2 = __lsx_vfadd_s(_f2, _c2);
                        _f3 = __lsx_vfadd_s(_f3, _c3);
                    }
                    else
                    {
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _f0 = __lsx_vfmadd_s(_c0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s(_c1, _beta, _f1);
                        _f2 = __lsx_vfmadd_s(_c2, _beta, _f2);
                        _f3 = __lsx_vfmadd_s(_c3, _beta, _f3);
                    }
                    pC += 4 * c_elempack;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c4 = (__m128)__lsx_vld(pC, 0);
                    if (beta != 1.f)
                    {
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _c4 = __lsx_vfmul_s(_c4, _beta);
                    }
                    _f0 = __lsx_vfadd_s(_f0, (__m128)__lsx_vreplvei_w((__m128i)_c4, 0));
                    _f1 = __lsx_vfadd_s(_f1, (__m128)__lsx_vreplvei_w((__m128i)_c4, 1));
                    _f2 = __lsx_vfadd_s(_f2, (__m128)__lsx_vreplvei_w((__m128i)_c4, 2));
                    _f3 = __lsx_vfadd_s(_f3, (__m128)__lsx_vreplvei_w((__m128i)_c4, 3));
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
                _f2 = __lsx_vfmul_s(_f2, _alpha);
                _f3 = __lsx_vfmul_s(_f3, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    __lsx_vst((__m128i)_f0, p0f, 0);
                    __lsx_vst((__m128i)_f1, p0f + 4, 0);
                    __lsx_vst((__m128i)_f2, p0f + 8, 0);
                    __lsx_vst((__m128i)_f3, p0f + 12, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vst((__m128i)_f0, p0f, 0);
                    __lsx_vst((__m128i)_f1, p0f + out_hstep, 0);
                    __lsx_vst((__m128i)_f2, p0f + out_hstep * 2, 0);
                    __lsx_vst((__m128i)_f3, p0f + out_hstep * 3, 0);
                }
                p0f += out_hstep * 4;
            }
            else
            {
                if (out_elempack == 8)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    const int out_lane = jj % 8;
                    __lsx_vstelm_d(float2bfloat_lsx(_f0), p0 + out_lane, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f1), p0 + 8 + out_lane, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f2), p0 + 16 + out_lane, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f3), p0 + 24 + out_lane, 0, 0);
                }
                if (out_elempack == 4)
                {
                    transpose4x4_ps(_f0, _f1, _f2, _f3);
                    __lsx_vstelm_d(float2bfloat_lsx(_f0), p0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f1), p0 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f2), p0 + 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f3), p0 + 12, 0, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vstelm_d(float2bfloat_lsx(_f0), p0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f1), p0 + out_hstep, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f2), p0 + out_hstep * 2, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f3), p0 + out_hstep * 3, 0, 0);
                }
                if (out_elempack == 8)
                {
                    if (jj % 8 == 4)
                        p0 += out_hstep * 8;
                }
                else
                {
                    p0 += out_hstep * 4;
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128i _s0 = __lsx_vld(pp, 0);
            __m128i _s1 = __lsx_vld(pp + 4, 0);
            pp += 8;

            __m128i _s0e = __lsx_vshuf4i_w(_s0, _LSX_SHUFFLE(3, 1, 2, 0));
            __m128i _s0o = __lsx_vshuf4i_w(_s0, _LSX_SHUFFLE(2, 0, 3, 1));
            __m128i _s1e = __lsx_vshuf4i_w(_s1, _LSX_SHUFFLE(3, 1, 2, 0));
            __m128i _s1o = __lsx_vshuf4i_w(_s1, _LSX_SHUFFLE(2, 0, 3, 1));
            __m128 _f0 = (__m128)__lsx_vilvl_w(_s1o, _s0e);
            __m128 _f1 = (__m128)__lsx_vilvl_w(_s0o, _s1e);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, _c);
                    _f1 = __lsx_vfadd_s(_f1, _c);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _cc0;
                    __m128 _cc1;
                    if (c_elempack == 4)
                    {
                        _cc0 = (__m128)__lsx_vld(pC, 0);
                        _cc1 = (__m128)__lsx_vld(pC + 4, 0);
                    }
                    else // if (c_elempack == 1)
                    {
                        __m128i _c0 = __lsx_vreplgr2vr_w(((const int*)pC)[0]);
                        _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep))[0], 1);
                        _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep * 2))[0], 2);
                        _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep * 3))[0], 3);
                        __m128i _c1 = __lsx_vreplgr2vr_w(((const int*)pC)[1]);
                        _c1 = __lsx_vinsgr2vr_w(_c1, ((const int*)(pC + c_hstep))[1], 1);
                        _c1 = __lsx_vinsgr2vr_w(_c1, ((const int*)(pC + c_hstep * 2))[1], 2);
                        _c1 = __lsx_vinsgr2vr_w(_c1, ((const int*)(pC + c_hstep * 3))[1], 3);
                        _cc0 = (__m128)_c0;
                        _cc1 = (__m128)_c1;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = __lsx_vfadd_s(_f0, _cc0);
                        _f1 = __lsx_vfadd_s(_f1, _cc1);
                    }
                    else
                    {
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _f0 = __lsx_vfmadd_s(_cc0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s(_cc1, _beta, _f1);
                    }
                    pC += 2 * c_elempack;
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
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
            }

            if (output_elemtype == 1)
            {
                __lsx_vst((__m128i)_f0, p0f, 0);
                __lsx_vst((__m128i)_f1, p0f + out_hstep, 0);
                p0f += out_hstep * 2;
            }
            else
            {
                __lsx_vstelm_d(float2bfloat_lsx((__m128)((__m128i)_f0)), p0, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)((__m128i)_f1)), p0 + out_hstep, 0, 0);
                p0 += out_hstep * 2;
            }
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
                    __m128i _ci;
                    if (c_elempack == 4)
                    {
                        _ci = __lsx_vld(pC, 0);
                    }
                    else // if (c_elempack == 1)
                    {
                        _ci = __lsx_vreplgr2vr_w(((const int*)pC)[0]);
                        _ci = __lsx_vinsgr2vr_w(_ci, ((const int*)(pC + c_hstep))[0], 1);
                        _ci = __lsx_vinsgr2vr_w(_ci, ((const int*)(pC + c_hstep * 2))[0], 2);
                        _ci = __lsx_vinsgr2vr_w(_ci, ((const int*)(pC + c_hstep * 3))[0], 3);
                    }
                    if (beta == 1.f)
                        _f = __lsx_vfadd_s(_f, (__m128)_ci);
                    else
                        _f = __lsx_vfmadd_s((__m128)_ci, __lsx_vreplfr2vr_s(beta), _f);
                    pC += c_elempack;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    __m128 _c4 = __lsx_vreplfr2vr_s(pC[0]);
                    if (beta == 1.f)
                        _f = __lsx_vfadd_s(_f, _c4);
                    else
                        _f = __lsx_vfmadd_s(_c4, _beta, _f);
                    pC++;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f = __lsx_vfmul_s(_f, _alpha);
            }

            if (output_elemtype == 1)
            {
                __lsx_vst((__m128i)_f, p0f, 0);
                p0f += out_hstep;
            }
            else
            {
                __lsx_vstelm_d(float2bfloat_lsx((__m128)((__m128i)_f)), p0, 0, 0);
                p0 += out_hstep;
            }
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* p0f = output_elemtype == 1 ? (float*)top_blob + (size_t)j * out_hstep + (i + ii) * out_elempack : 0;
        unsigned short* p0 = output_elemtype == 3 ? (unsigned short*)top_blob + (size_t)j * out_hstep + (i + ii) * out_elempack : 0;
        const float* pC = C;

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
                pC += i + ii;
                c0 = pC[0] * beta;
                c1 = pC[1] * beta;
            }
            if (broadcast_type_C == 3)
            {
                pC += (size_t)(i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC += j;
            }
        }

        int jj = 0;
#if __loongarch_sx
        __m128 _c = __lsx_vreplfr2vr_s(0.f);
        if (pC)
        {
            if (broadcast_type_C == 0)
                _c = __lsx_vreplfr2vr_s(c0);
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                __m128 _c0 = __lsx_vreplfr2vr_s(c0);
                __m128 _c1 = __lsx_vreplfr2vr_s(c1);
                _c = (__m128)__lsx_vilvl_w((__m128i)_c1, (__m128i)_c0);
            }
        }
#if __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const float* pC0 = pC;
            __m256 _f0 = (__m256)__lasx_xvld(pp, 0);
            __m256 _f1 = (__m256)__lasx_xvld(pp + 8, 0);
            if (pC0)
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
                        _f0 = __lasx_xvfadd_s(_f0, (__m256)__lasx_xvld(pC0, 0));
                        _f1 = __lasx_xvfadd_s(_f1, (__m256)__lasx_xvld(pC0 + c_hstep, 0));
                    }
                    else
                    {
                        __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                        _f0 = __lasx_xvfmadd_s((__m256)__lasx_xvld(pC0, 0), _beta, _f0);
                        _f1 = __lasx_xvfmadd_s((__m256)__lasx_xvld(pC0 + c_hstep, 0), _beta, _f1);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c4 = (__m256)__lasx_xvld(pC0, 0);
                    if (beta != 1.f)
                    {
                        __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                        _c4 = __lasx_xvfmul_s(_c4, _beta);
                    }
                    _f0 = __lasx_xvfadd_s(_f0, _c4);
                    _f1 = __lasx_xvfadd_s(_f1, _c4);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m256 _alpha = (__m256)__lasx_xvreplfr2vr_s(alpha);
                _f0 = __lasx_xvfmul_s(_f0, _alpha);
                _f1 = __lasx_xvfmul_s(_f1, _alpha);
            }

            __m256i _tmp0 = __lasx_xvilvl_w((__m256i)_f1, (__m256i)_f0);
            __m256i _tmp1 = __lasx_xvilvh_w((__m256i)_f1, (__m256i)_f0);
            if (output_elemtype == 1)
            {
                if (out_elempack == 8)
                {
                    __lasx_xvst(_f0, p0f, 0);
                    __lasx_xvst(_f1, p0f + 8, 0);
                }
                if (out_elempack == 4)
                {
                    float* p1f = p0f + out_hstep * 4;
                    __lsx_vst(__lasx_extract_128_lo((__m256i)_f0), p0f, 0);
                    __lsx_vst(__lasx_extract_128_lo((__m256i)_f1), p0f + 4, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_f0), p1f, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_f1), p1f + 4, 0);
                }
                if (out_elempack == 1)
                {
                    __lasx_xvstelm_d(_tmp0, p0f, 0, 0);
                    __lasx_xvstelm_d(_tmp0, p0f + out_hstep, 0, 1);
                    __lasx_xvstelm_d(_tmp1, p0f + out_hstep * 2, 0, 0);
                    __lasx_xvstelm_d(_tmp1, p0f + out_hstep * 3, 0, 1);
                    __lasx_xvstelm_d(_tmp0, p0f + out_hstep * 4, 0, 2);
                    __lasx_xvstelm_d(_tmp0, p0f + out_hstep * 5, 0, 3);
                    __lasx_xvstelm_d(_tmp1, p0f + out_hstep * 6, 0, 2);
                    __lasx_xvstelm_d(_tmp1, p0f + out_hstep * 7, 0, 3);
                }
            }
            else
            {
                if (out_elempack == 8)
                {
                    __lsx_vst(float2bfloat_lasx(_f0), p0, 0);
                    __lsx_vst(float2bfloat_lasx(_f1), p0 + 8, 0);
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    __m128i _q0 = float2bfloat_lasx(_f0);
                    __m128i _q1 = float2bfloat_lasx(_f1);
                    __lsx_vstelm_d(_q0, p0, 0, 0);
                    __lsx_vstelm_d(_q1, p0 + 4, 0, 0);
                    __lsx_vstelm_d(_q0, p1, 0, 1);
                    __lsx_vstelm_d(_q1, p1 + 4, 0, 1);
                }
                if (out_elempack == 1)
                {
                    __lsx_vstelm_w(float2bfloat_lasx((__m256)_tmp0), p0, 0, 0);
                    __lsx_vstelm_w(float2bfloat_lasx((__m256)_tmp0), p0 + out_hstep, 0, 1);
                    __lsx_vstelm_w(float2bfloat_lasx((__m256)_tmp1), p0 + out_hstep * 2, 0, 0);
                    __lsx_vstelm_w(float2bfloat_lasx((__m256)_tmp1), p0 + out_hstep * 3, 0, 1);
                    __lsx_vstelm_w(float2bfloat_lasx((__m256)_tmp0), p0 + out_hstep * 4, 0, 2);
                    __lsx_vstelm_w(float2bfloat_lasx((__m256)_tmp0), p0 + out_hstep * 5, 0, 3);
                    __lsx_vstelm_w(float2bfloat_lasx((__m256)_tmp1), p0 + out_hstep * 6, 0, 2);
                    __lsx_vstelm_w(float2bfloat_lasx((__m256)_tmp1), p0 + out_hstep * 7, 0, 3);
                }
            }
            pp += 16;
            if (output_elemtype == 1)
                p0f += out_hstep * 8;
            else
                p0 += out_hstep * 8;
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp, 0);
            __m128 _f1 = (__m128)__lsx_vld(pp + 4, 0);
            pp += 8;
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
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _f0 = __lsx_vfmadd_s(_c0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s(_c1, _beta, _f1);
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c4 = (__m128)__lsx_vld(pC, 0);
                    if (beta != 1.f)
                    {
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _c4 = __lsx_vfmul_s(_c4, _beta);
                    }
                    _f0 = __lsx_vfadd_s(_f0, _c4);
                    _f1 = __lsx_vfadd_s(_f1, _c4);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_f0, p0f, 0);
                    __lsx_vst((__m128i)_f1, p0f + 4, 0);
                }
                if (out_elempack == 1)
                {
                    __m128i _tmp0 = __lsx_vilvl_w((__m128i)_f1, (__m128i)_f0);
                    __m128i _tmp1 = __lsx_vilvh_w((__m128i)_f1, (__m128i)_f0);
                    __lsx_vstelm_d(_tmp0, p0f, 0, 0);
                    __lsx_vstelm_d(_tmp0, p0f + out_hstep, 0, 1);
                    __lsx_vstelm_d(_tmp1, p0f + out_hstep * 2, 0, 0);
                    __lsx_vstelm_d(_tmp1, p0f + out_hstep * 3, 0, 1);
                }
                p0f += out_hstep * 4;
            }
            else
            {
                if (out_elempack == 8)
                {
                    const int out_lane = jj % 8;
                    __lsx_vstelm_d(float2bfloat_lsx(_f0), p0 + out_lane, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f1), p0 + 8 + out_lane, 0, 0);
                }
                if (out_elempack == 4)
                {
                    __lsx_vstelm_d(float2bfloat_lsx(_f0), p0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_f1), p0 + 4, 0, 0);
                }
                if (out_elempack == 1)
                {
                    __m128i _tmp0 = __lsx_vilvl_w((__m128i)_f1, (__m128i)_f0);
                    __m128i _tmp1 = __lsx_vilvh_w((__m128i)_f1, (__m128i)_f0);
                    __lsx_vstelm_w(float2bfloat_lsx((__m128)(_tmp0)), p0, 0, 0);
                    __lsx_vstelm_w(float2bfloat_lsx((__m128)(_tmp0)), p0 + out_hstep, 0, 1);
                    __lsx_vstelm_w(float2bfloat_lsx((__m128)(_tmp1)), p0 + out_hstep * 2, 0, 0);
                    __lsx_vstelm_w(float2bfloat_lsx((__m128)(_tmp1)), p0 + out_hstep * 3, 0, 1);
                }
                if (out_elempack == 8)
                {
                    if (jj % 8 == 4)
                        p0 += out_hstep * 8;
                }
                else
                {
                    p0 += out_hstep * 4;
                }
            }
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            const float* pC0 = pC;
#if __loongarch_sx
            __m128 _f = (__m128)__lsx_vshuf4i_w(__lsx_vld(pp, 0), _LSX_SHUFFLE(3, 1, 2, 0));
            __m128i _r0;
            __m128i _r1;
            if (pC0)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f = __lsx_vfadd_s(_f, _c);
                if (broadcast_type_C == 3)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    _r0 = __lsx_vldrepl_d(pC0, 0);
                    _r1 = __lsx_vldrepl_d(pC0 + c_hstep, 0);
                    __m128 _cc = (__m128)__lsx_vilvl_w(_r1, _r0);
                    if (beta == 1.f)
                        _f = __lsx_vfadd_s(_f, _cc);
                    else
                        _f = __lsx_vfmadd_s(_cc, _beta, _f);
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    __m128i _cc = __lsx_vldrepl_d(pC0, 0);
                    _cc = __lsx_vilvl_w(_cc, _cc);
                    if (beta == 1.f)
                        _f = __lsx_vfadd_s(_f, (__m128)_cc);
                    else
                        _f = __lsx_vfmadd_s((__m128)_cc, _beta, _f);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f = __lsx_vfmul_s(_f, _alpha);
            }

            if (output_elemtype == 1)
            {
                __lsx_vstelm_d((__m128i)_f, p0f, 0, 0);
                __lsx_vstelm_d((__m128i)_f, p0f + out_hstep, 0, 1);
            }
            else
            {
                __lsx_vstelm_w(float2bfloat_lsx((__m128)((__m128i)_f)), p0, 0, 0);
                __lsx_vstelm_w(float2bfloat_lsx((__m128)((__m128i)_f)), p0 + out_hstep, 0, 1);
            }
#else
            float f00 = pp[0];
            float f01 = pp[1];
            float f10 = pp[2];
            float f11 = pp[3];
            if (pC0)
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
                        f00 += pC0[0];
                        f01 += pC0[1];
                        f10 += pC0[c_hstep];
                        f11 += pC0[c_hstep + 1];
                    }
                    else
                    {
                        f00 += pC0[0] * beta;
                        f01 += pC0[1] * beta;
                        f10 += pC0[c_hstep] * beta;
                        f11 += pC0[c_hstep + 1] * beta;
                    }
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    const float cc0 = pC0[0] * beta;
                    const float cc1 = pC0[1] * beta;
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

            if (output_elemtype == 1)
            {
                p0f[0] = f00;
                p0f[1] = f10;
                p0f[out_hstep] = f01;
                p0f[out_hstep + 1] = f11;
            }
            else
            {
                p0[0] = float32_to_bfloat16(f00);
                p0[1] = float32_to_bfloat16(f10);
                p0[out_hstep] = float32_to_bfloat16(f01);
                p0[out_hstep + 1] = float32_to_bfloat16(f11);
            }
#endif // __loongarch_sx
            pp += 4;
            if (output_elemtype == 1)
                p0f += out_hstep * 2;
            else
                p0 += out_hstep * 2;
        }
        for (; jj < max_jj; jj++)
        {
#if __loongarch_sx
            __m128i _fi = __lsx_vldrepl_d(pp, 0);
            __m128 _f = (__m128)_fi;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f = __lsx_vfadd_s(_f, _c);
                if (broadcast_type_C == 3)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
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
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    __m128 _c4 = __lsx_vreplfr2vr_s(pC[0]);
                    if (beta == 1.f)
                        _f = __lsx_vfadd_s(_f, _c4);
                    else
                        _f = __lsx_vfmadd_s(_c4, _beta, _f);
                    pC++;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f = __lsx_vfmul_s(_f, _alpha);
            }

            if (output_elemtype == 1)
            {
                __lsx_vstelm_d((__m128i)_f, p0f, 0, 0);
            }
            else
            {
                __lsx_vstelm_w(float2bfloat_lsx((__m128)((__m128i)_f)), p0, 0, 0);
            }
#else
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
                    float c = pC[0] * beta;
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

            if (output_elemtype == 1)
            {
                p0f[0] = f0;
                p0f[1] = f1;
            }
            else
            {
                p0[0] = float32_to_bfloat16(f0);
                p0[1] = float32_to_bfloat16(f1);
            }
#endif // __loongarch_sx
            pp += 2;
            if (output_elemtype == 1)
                p0f += out_hstep;
            else
                p0 += out_hstep;
        }
    }
    for (; ii < max_ii; ii++)
    {
        float* p0f = output_elemtype == 1 ? (float*)top_blob + (size_t)j * out_hstep + (i + ii) * out_elempack : 0;
        unsigned short* p0 = output_elemtype == 3 ? (unsigned short*)top_blob + (size_t)j * out_hstep + (i + ii) * out_elempack : 0;
        const float* pC = C;

        float c0 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC += i + ii;
                c0 = pC[0] * beta;
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
        __m128 _c128 = __lsx_vreplfr2vr_s(c0);
#if __loongarch_asx
        __m256 _c256 = (__m256)__lasx_xvreplfr2vr_s(c0);
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = (__m256)__lasx_xvld(pp, 0);
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f0 = __lasx_xvfadd_s(_f0, _c256);
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                    __m256 _c0 = (__m256)__lasx_xvld(pC, 0);
                    if (beta == 1.f)
                        _f0 = __lasx_xvfadd_s(_f0, _c0);
                    else
                        _f0 = __lasx_xvfmadd_s(_c0, _beta, _f0);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m256 _alpha = (__m256)__lasx_xvreplfr2vr_s(alpha);
                _f0 = __lasx_xvfmul_s(_f0, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (out_elempack == 8)
                {
                    __lasx_xvst(_f0, p0f, 0);
                }
                if (out_elempack == 4)
                {
                    __lsx_vst(__lasx_extract_128_lo((__m256i)_f0), p0f, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_f0), p0f + out_hstep * 4, 0);
                }
                if (out_elempack == 1)
                {
                    if (out_hstep == 1)
                    {
                        __lasx_xvst(_f0, p0f, 0);
                    }
                    else
                    {
                        __lasx_xvstelm_w((__m256i)_f0, p0f, 0, 0);
                        __lasx_xvstelm_w((__m256i)_f0, p0f + out_hstep, 0, 1);
                        __lasx_xvstelm_w((__m256i)_f0, p0f + out_hstep * 2, 0, 2);
                        __lasx_xvstelm_w((__m256i)_f0, p0f + out_hstep * 3, 0, 3);
                        __lasx_xvstelm_w((__m256i)_f0, p0f + out_hstep * 4, 0, 4);
                        __lasx_xvstelm_w((__m256i)_f0, p0f + out_hstep * 5, 0, 5);
                        __lasx_xvstelm_w((__m256i)_f0, p0f + out_hstep * 6, 0, 6);
                        __lasx_xvstelm_w((__m256i)_f0, p0f + out_hstep * 7, 0, 7);
                    }
                }
            }
            else
            {
                if (out_elempack == 8)
                {
                    __lsx_vst(float2bfloat_lasx(_f0), p0, 0);
                }
                if (out_elempack == 4)
                {
                    __m128i _q = float2bfloat_lasx(_f0);
                    __lsx_vstelm_d(_q, p0, 0, 0);
                    __lsx_vstelm_d(_q, p0 + out_hstep * 4, 0, 1);
                }
                if (out_elempack == 1)
                {
                    if (out_hstep == 1)
                    {
                        __lsx_vst(float2bfloat_lasx((__m256)_f0), p0, 0);
                    }
                    else
                    {
                        __lsx_vstelm_h(float2bfloat_lasx((__m256)(__m256i)_f0), p0, 0, 0);
                        __lsx_vstelm_h(float2bfloat_lasx((__m256)(__m256i)_f0), p0 + out_hstep, 0, 1);
                        __lsx_vstelm_h(float2bfloat_lasx((__m256)(__m256i)_f0), p0 + out_hstep * 2, 0, 2);
                        __lsx_vstelm_h(float2bfloat_lasx((__m256)(__m256i)_f0), p0 + out_hstep * 3, 0, 3);
                        __lsx_vstelm_h(float2bfloat_lasx((__m256)(__m256i)_f0), p0 + out_hstep * 4, 0, 4);
                        __lsx_vstelm_h(float2bfloat_lasx((__m256)(__m256i)_f0), p0 + out_hstep * 5, 0, 5);
                        __lsx_vstelm_h(float2bfloat_lasx((__m256)(__m256i)_f0), p0 + out_hstep * 6, 0, 6);
                        __lsx_vstelm_h(float2bfloat_lasx((__m256)(__m256i)_f0), p0 + out_hstep * 7, 0, 7);
                    }
                }
            }
            pp += 8;
            if (output_elemtype == 1)
                p0f += out_hstep * 8;
            else
                p0 += out_hstep * 8;
        }
#endif // __loongarch_asx
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = (__m128)__lsx_vld(pp, 0);
            pp += 4;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f0 = __lsx_vfadd_s(_f0, _c128);
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    __m128 _c0 = (__m128)__lsx_vld(pC, 0);
                    pC += 4;
                    if (beta == 1.f)
                        _f0 = __lsx_vfadd_s(_f0, _c0);
                    else
                        _f0 = __lsx_vfmadd_s(_c0, _beta, _f0);
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_f0, p0f, 0);
                }
                if (out_elempack == 1)
                {
                    if (out_hstep == 1)
                    {
                        __lsx_vst((__m128i)_f0, p0f, 0);
                    }
                    else
                    {
                        __lsx_vstelm_w((__m128i)_f0, p0f, 0, 0);
                        __lsx_vstelm_w((__m128i)_f0, p0f + out_hstep, 0, 1);
                        __lsx_vstelm_w((__m128i)_f0, p0f + out_hstep * 2, 0, 2);
                        __lsx_vstelm_w((__m128i)_f0, p0f + out_hstep * 3, 0, 3);
                    }
                }
            }
            else
            {
                if (out_elempack == 8)
                {
                    __lsx_vstelm_d(float2bfloat_lsx(_f0), p0 + jj % 8, 0, 0);
                }
                if (out_elempack == 4)
                {
                    __lsx_vstelm_d(float2bfloat_lsx(_f0), p0, 0, 0);
                }
                if (out_elempack == 1)
                {
                    if (out_hstep == 1)
                    {
                        __lsx_vstelm_d(float2bfloat_lsx((__m128)((__m128i)_f0)), p0, 0, 0);
                    }
                    else
                    {
                        __lsx_vstelm_h(float2bfloat_lsx((__m128)((__m128i)_f0)), p0, 0, 0);
                        __lsx_vstelm_h(float2bfloat_lsx((__m128)((__m128i)_f0)), p0 + out_hstep, 0, 1);
                        __lsx_vstelm_h(float2bfloat_lsx((__m128)((__m128i)_f0)), p0 + out_hstep * 2, 0, 2);
                        __lsx_vstelm_h(float2bfloat_lsx((__m128)((__m128i)_f0)), p0 + out_hstep * 3, 0, 3);
                    }
                }
            }

            if (output_elemtype == 1)
                p0f += out_hstep * 4;
            else
            {
                if (out_elempack == 8)
                {
                    if (jj % 8 == 4)
                        p0 += out_hstep * 8;
                }
                else
                {
                    p0 += out_hstep * 4;
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _f0 = (__m128)__lsx_vldrepl_d(pp, 0);
            pp += 2;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    _f0 = __lsx_vfadd_s(_f0, _c128);
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    __m128 _cc = (__m128)__lsx_vldrepl_d(pC, 0);
                    pC += 2;
                    if (beta == 1.f)
                        _f0 = __lsx_vfadd_s(_f0, _cc);
                    else
                        _f0 = __lsx_vfmadd_s(_cc, _beta, _f0);
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
            }

            if (output_elemtype == 1)
            {
                if (out_hstep == 1)
                {
                    __lsx_vstelm_d((__m128i)_f0, p0f, 0, 0);
                }
                else
                {
                    __lsx_vstelm_w((__m128i)_f0, p0f, 0, 0);
                    __lsx_vstelm_w((__m128i)_f0, p0f + out_hstep, 0, 1);
                }
            }
            else
            {
                if (out_hstep == 1)
                {
                    __lsx_vstelm_w(float2bfloat_lsx((__m128)((__m128i)_f0)), p0, 0, 0);
                }
                else
                {
                    __lsx_vstelm_h(float2bfloat_lsx((__m128)((__m128i)_f0)), p0, 0, 0);
                    __lsx_vstelm_h(float2bfloat_lsx((__m128)((__m128i)_f0)), p0 + out_hstep, 0, 1);
                }
            }

            if (output_elemtype == 1)
                p0f += out_hstep * 2;
            else
                p0 += out_hstep * 2;
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
                    f0 += pC[0] * beta;
                    pC++;
                }
            }

            if (alpha != 1.f)
                f0 *= alpha;
            if (output_elemtype == 1)
            {
                p0f[0] = f0;
                p0f += out_hstep;
            }
            else
            {
                p0[0] = float32_to_bfloat16(f0);
                p0 += out_hstep;
            }
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

#if __loongarch_sx
    TILE_M = std::max(8, tile_size / 8 * 8);
#if __loongarch_asx
    TILE_N = std::max(8, tile_size / 8 * 8);
#else
    TILE_N = std::max(4, tile_size / 4 * 4);
#endif
#else
    TILE_M = std::max(2, tile_size / 2 * 2);
    TILE_N = std::max(2, tile_size / 2 * 2);
#endif

    TILE_K = std::max(block_size, tile_size / block_size * block_size);

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + block_size - 1) / block_size * block_size);
        TILE_K = std::min(TILE_K, K);

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(signed char) / TILE_K);

#if __loongarch_sx
            TILE_M = std::max(8, tile_size / 8 * 8);
#if __loongarch_asx
            TILE_N = std::max(8, tile_size / 8 * 8);
#else
            TILE_N = std::max(4, tile_size / 4 * 4);
#endif
#else
            TILE_M = std::max(2, tile_size / 2 * 2);
            TILE_N = std::max(2, tile_size / 2 * 2);
#endif
        }
    }

    TILE_M *= std::min(nT, get_physical_cpu_count());

    if (M > 0)
    {
        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __loongarch_sx
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __loongarch_sx
#if __loongarch_asx
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8);
#else
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#endif
#else
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 1) / 2 * 2);
#endif
    }

    if (nT > 1)
    {
#if __loongarch_sx
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#else
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
    }

    // always take constant TILE_M/N/K value when provided
    if (constant_TILE_M > 0)
    {
#if __loongarch_sx
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
#else
        TILE_M = (constant_TILE_M + 1) / 2 * 2;
#endif
    }

    if (constant_TILE_N > 0)
    {
#if __loongarch_sx
#if __loongarch_asx
        TILE_N = (constant_TILE_N + 7) / 8 * 8;
#else
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
#endif
#else
        TILE_N = (constant_TILE_N + 1) / 2 * 2;
#endif
    }

    if (constant_TILE_K > 0)
    {
        TILE_K = std::max(block_size, constant_TILE_K / block_size * block_size);
        if (K > 0)
            TILE_K = std::min(TILE_K, K);
    }
}

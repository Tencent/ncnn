// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void quantize_A_tile_wq_int8_bf16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
#if __loongarch_sx
    const int elempack = A.elempack;
#endif // __loongarch_sx
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int local_block_count = (max_kk + block_size - 1) / block_size;

    if (input_scales.empty())
    {
        int ii = 0;
#if __loongarch_sx
        for (; ii + 7 < max_ii; ii += 8)
        {
            if (elempack == 8)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k * 8;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax0 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax1 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        __m128 _p0 = bfloat2float_lsx(p0a);
                        __m128 _p1 = bfloat2float_lsx(p0a + 4);
                        _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_p0, _abs_mask));
                        _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_p1, _abs_mask));
                        p0a += 8;
                    }

                    const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                    __lsx_vst(__lsx_vfdiv_s(_absmax0, _v127), pd, 0);
                    __lsx_vst(__lsx_vfdiv_s(_absmax1, _v127), pd + 4, 0);
                    pd += 8;

                    const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax03 = (__m128)__lsx_vbitsel_v((__m128i)_absmax0, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax0, _zero));
                    __m128 _absmax47 = (__m128)__lsx_vbitsel_v((__m128i)_absmax1, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax1, _zero));
                    __m128 _scale03 = __lsx_vfdiv_s(_v127, _absmax03);
                    __m128 _scale47 = __lsx_vfdiv_s(_v127, _absmax47);

                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = bfloat2float_lsx(p0);
                        __m128 _p1 = bfloat2float_lsx(p0 + 8);
                        __m128 _p2 = bfloat2float_lsx(p0 + 16);
                        __m128 _p3 = bfloat2float_lsx(p0 + 24);
                        __m128 _p4 = bfloat2float_lsx(p0 + 4);
                        __m128 _p5 = bfloat2float_lsx(p0 + 12);
                        __m128 _p6 = bfloat2float_lsx(p0 + 20);
                        __m128 _p7 = bfloat2float_lsx(p0 + 28);
                        __m128i _q0 = float2int8(__lsx_vfmul_s(_p0, _scale03));
                        __m128i _q1 = float2int8(__lsx_vfmul_s(_p1, _scale03));
                        __m128i _q2 = float2int8(__lsx_vfmul_s(_p2, _scale03));
                        __m128i _q3 = float2int8(__lsx_vfmul_s(_p3, _scale03));
                        __m128i _q01 = __lsx_vilvl_b(_q1, _q0);
                        __m128i _q23 = __lsx_vilvl_b(_q3, _q2);
                        __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp, 0);
                        _q0 = float2int8(__lsx_vfmul_s(_p4, _scale47));
                        _q1 = float2int8(__lsx_vfmul_s(_p5, _scale47));
                        _q2 = float2int8(__lsx_vfmul_s(_p6, _scale47));
                        _q3 = float2int8(__lsx_vfmul_s(_p7, _scale47));
                        _q01 = __lsx_vilvl_b(_q1, _q0);
                        _q23 = __lsx_vilvl_b(_q3, _q2);
                        __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp + 16, 0);
                        pp += 32;
                        p0 += 32;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p0 = bfloat2float_lsx(p0);
                        __m128 _p1 = bfloat2float_lsx(p0 + 4);
                        _p0 = __lsx_vfmul_s(_p0, _scale03);
                        _p1 = __lsx_vfmul_s(_p1, _scale47);
                        __m128i _q0 = float2int8(_p0);
                        __m128i _q1 = float2int8(_p1);
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)_q0, 0);
                        ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)_q1, 0);
                        pp += 8;
                        p0 += 8;
                    }
                }
            }
            if (elempack == 4)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k * 4;
                const unsigned short* p1 = p0 + A_hstep * 4;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax0 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax1 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    const unsigned short* p1a = p1;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        __m128 _p0 = bfloat2float_lsx(p0a);
                        __m128 _p1 = bfloat2float_lsx(p1a);
                        _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_p0, _abs_mask));
                        _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_p1, _abs_mask));
                        p0a += 4;
                        p1a += 4;
                    }

                    const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                    __lsx_vst(__lsx_vfdiv_s(_absmax0, _v127), pd, 0);
                    __lsx_vst(__lsx_vfdiv_s(_absmax1, _v127), pd + 4, 0);
                    pd += 8;

                    const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax03 = (__m128)__lsx_vbitsel_v((__m128i)_absmax0, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax0, _zero));
                    __m128 _absmax47 = (__m128)__lsx_vbitsel_v((__m128i)_absmax1, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax1, _zero));
                    __m128 _scale03 = __lsx_vfdiv_s(_v127, _absmax03);
                    __m128 _scale47 = __lsx_vfdiv_s(_v127, _absmax47);

                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = bfloat2float_lsx(p0);
                        __m128 _p1 = bfloat2float_lsx(p0 + 4);
                        __m128 _p2 = bfloat2float_lsx(p0 + 8);
                        __m128 _p3 = bfloat2float_lsx(p0 + 12);
                        __m128 _p4 = bfloat2float_lsx(p1);
                        __m128 _p5 = bfloat2float_lsx(p1 + 4);
                        __m128 _p6 = bfloat2float_lsx(p1 + 8);
                        __m128 _p7 = bfloat2float_lsx(p1 + 12);
                        __m128i _q0 = float2int8(__lsx_vfmul_s(_p0, _scale03));
                        __m128i _q1 = float2int8(__lsx_vfmul_s(_p1, _scale03));
                        __m128i _q2 = float2int8(__lsx_vfmul_s(_p2, _scale03));
                        __m128i _q3 = float2int8(__lsx_vfmul_s(_p3, _scale03));
                        __m128i _q01 = __lsx_vilvl_b(_q1, _q0);
                        __m128i _q23 = __lsx_vilvl_b(_q3, _q2);
                        __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp, 0);
                        _q0 = float2int8(__lsx_vfmul_s(_p4, _scale47));
                        _q1 = float2int8(__lsx_vfmul_s(_p5, _scale47));
                        _q2 = float2int8(__lsx_vfmul_s(_p6, _scale47));
                        _q3 = float2int8(__lsx_vfmul_s(_p7, _scale47));
                        _q01 = __lsx_vilvl_b(_q1, _q0);
                        _q23 = __lsx_vilvl_b(_q3, _q2);
                        __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp + 16, 0);
                        pp += 32;
                        p0 += 16;
                        p1 += 16;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p0 = bfloat2float_lsx(p0);
                        __m128 _p1 = bfloat2float_lsx(p1);
                        _p0 = __lsx_vfmul_s(_p0, _scale03);
                        _p1 = __lsx_vfmul_s(_p1, _scale47);
                        __m128i _q0 = float2int8(_p0);
                        __m128i _q1 = float2int8(_p1);
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)_q0, 0);
                        ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)_q1, 0);
                        pp += 8;
                        p0 += 4;
                        p1 += 4;
                    }
                }
            }
            if (elempack == 1)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k;
                const unsigned short* p1 = p0 + A_hstep;
                const unsigned short* p2 = p1 + A_hstep;
                const unsigned short* p3 = p2 + A_hstep;
                const unsigned short* p4 = p3 + A_hstep;
                const unsigned short* p5 = p4 + A_hstep;
                const unsigned short* p6 = p5 + A_hstep;
                const unsigned short* p7 = p6 + A_hstep;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax0 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax1 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax2 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax3 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax4 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax5 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax6 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax7 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    const unsigned short* p1a = p1;
                    const unsigned short* p2a = p2;
                    const unsigned short* p3a = p3;
                    const unsigned short* p4a = p4;
                    const unsigned short* p5a = p5;
                    const unsigned short* p6a = p6;
                    const unsigned short* p7a = p7;
                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = bfloat2float_lsx(p0a);
                        __m128 _p1 = bfloat2float_lsx(p1a);
                        __m128 _p2 = bfloat2float_lsx(p2a);
                        __m128 _p3 = bfloat2float_lsx(p3a);
                        __m128 _p4 = bfloat2float_lsx(p4a);
                        __m128 _p5 = bfloat2float_lsx(p5a);
                        __m128 _p6 = bfloat2float_lsx(p6a);
                        __m128 _p7 = bfloat2float_lsx(p7a);
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

                    transpose4x4_ps(_absmax0, _absmax1, _absmax2, _absmax3);
                    transpose4x4_ps(_absmax4, _absmax5, _absmax6, _absmax7);
                    _absmax0 = __lsx_vfmax_s(__lsx_vfmax_s(_absmax0, _absmax1), __lsx_vfmax_s(_absmax2, _absmax3));
                    _absmax1 = __lsx_vfmax_s(__lsx_vfmax_s(_absmax4, _absmax5), __lsx_vfmax_s(_absmax6, _absmax7));

                    for (; kk < max_kk0; kk++)
                    {
                        __m128i _p0 = __lsx_vldrepl_h(p0a, 0);
                        _p0 = __lsx_vinsgr2vr_h(_p0, *p1a, 1);
                        _p0 = __lsx_vinsgr2vr_h(_p0, *p2a, 2);
                        _p0 = __lsx_vinsgr2vr_h(_p0, *p3a, 3);
                        __m128i _p1 = __lsx_vldrepl_h(p4a, 0);
                        _p1 = __lsx_vinsgr2vr_h(_p1, *p5a, 1);
                        _p1 = __lsx_vinsgr2vr_h(_p1, *p6a, 2);
                        _p1 = __lsx_vinsgr2vr_h(_p1, *p7a, 3);
                        _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)bfloat2float_lsx(_p0), _abs_mask));
                        _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)bfloat2float_lsx(_p1), _abs_mask));
                        p0a++;
                        p1a++;
                        p2a++;
                        p3a++;
                        p4a++;
                        p5a++;
                        p6a++;
                        p7a++;
                    }

                    const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                    __lsx_vst(__lsx_vfdiv_s(_absmax0, _v127), pd, 0);
                    __lsx_vst(__lsx_vfdiv_s(_absmax1, _v127), pd + 4, 0);
                    pd += 8;

                    const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax03 = (__m128)__lsx_vbitsel_v((__m128i)_absmax0, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax0, _zero));
                    __m128 _absmax47 = (__m128)__lsx_vbitsel_v((__m128i)_absmax1, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax1, _zero));
                    __m128 _scale03 = __lsx_vfdiv_s(_v127, _absmax03);
                    __m128 _scale47 = __lsx_vfdiv_s(_v127, _absmax47);
                    __m128 _scale0 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 0);
                    __m128 _scale1 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 1);
                    __m128 _scale2 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 2);
                    __m128 _scale3 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 3);
                    __m128 _scale4 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 0);
                    __m128 _scale5 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 1);
                    __m128 _scale6 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 2);
                    __m128 _scale7 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 3);
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = bfloat2float_lsx(p0);
                        __m128 _p1 = bfloat2float_lsx(p1);
                        __m128 _p2 = bfloat2float_lsx(p2);
                        __m128 _p3 = bfloat2float_lsx(p3);
                        __m128 _p4 = bfloat2float_lsx(p4);
                        __m128 _p5 = bfloat2float_lsx(p5);
                        __m128 _p6 = bfloat2float_lsx(p6);
                        __m128 _p7 = bfloat2float_lsx(p7);
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
                        __m128i _p0 = __lsx_vldrepl_h(p0, 0);
                        _p0 = __lsx_vinsgr2vr_h(_p0, *p1, 1);
                        _p0 = __lsx_vinsgr2vr_h(_p0, *p2, 2);
                        _p0 = __lsx_vinsgr2vr_h(_p0, *p3, 3);
                        __m128i _p1 = __lsx_vldrepl_h(p4, 0);
                        _p1 = __lsx_vinsgr2vr_h(_p1, *p5, 1);
                        _p1 = __lsx_vinsgr2vr_h(_p1, *p6, 2);
                        _p1 = __lsx_vinsgr2vr_h(_p1, *p7, 3);
                        ((int64_t*)pp)[0] = float2int8(__lsx_vfmul_s(bfloat2float_lsx(_p0), _scale03), __lsx_vfmul_s(bfloat2float_lsx(_p1), _scale47));
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
            const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k * elempack;
            const unsigned short* p1 = 0;
            const unsigned short* p2 = 0;
            const unsigned short* p3 = 0;
            if (elempack == 1)
            {
                p1 = p0 + A_hstep;
                p2 = p1 + A_hstep;
                p3 = p2 + A_hstep;
            }

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                __m128 _absmax0 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax1 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax2 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax3 = (__m128)__lsx_vreplgr2vr_w(0);

                const unsigned short* p0a = p0;
                const unsigned short* p1a = p1;
                const unsigned short* p2a = p2;
                const unsigned short* p3a = p3;
                int kk = 0;

                if (elempack == 4)
                {
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p = bfloat2float_lsx(p0a);
                        _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_p, _abs_mask));
                        p0a += 4;
                    }
                }

                if (elempack == 1)
                {
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = bfloat2float_lsx(p0a);
                        __m128 _p1 = bfloat2float_lsx(p1a);
                        __m128 _p2 = bfloat2float_lsx(p2a);
                        __m128 _p3 = bfloat2float_lsx(p3a);
                        _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_p0, _abs_mask));
                        _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_p1, _abs_mask));
                        _absmax2 = __lsx_vfmax_s(_absmax2, (__m128)__lsx_vand_v((__m128i)_p2, _abs_mask));
                        _absmax3 = __lsx_vfmax_s(_absmax3, (__m128)__lsx_vand_v((__m128i)_p3, _abs_mask));
                        p0a += 4;
                        p1a += 4;
                        p2a += 4;
                        p3a += 4;
                    }

                    transpose4x4_ps(_absmax0, _absmax1, _absmax2, _absmax3);
                    _absmax0 = __lsx_vfmax_s(__lsx_vfmax_s(_absmax0, _absmax1), __lsx_vfmax_s(_absmax2, _absmax3));

                    for (; kk < max_kk0; kk++)
                    {
                        __m128i _p = __lsx_vldrepl_h(p0a, 0);
                        _p = __lsx_vinsgr2vr_h(_p, *p1a, 1);
                        _p = __lsx_vinsgr2vr_h(_p, *p2a, 2);
                        _p = __lsx_vinsgr2vr_h(_p, *p3a, 3);
                        _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)bfloat2float_lsx(_p), _abs_mask));
                        p0a++;
                        p1a++;
                        p2a++;
                        p3a++;
                    }
                }

                const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                __lsx_vst(__lsx_vfdiv_s(_absmax0, _v127), pd, 0);
                pd += 4;

                const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax = (__m128)__lsx_vbitsel_v((__m128i)_absmax0, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax0, _zero));
                __m128 _scale = __lsx_vfdiv_s(_v127, _absmax);

                if (elempack == 4)
                {
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = __lsx_vfmul_s(bfloat2float_lsx(p0), _scale);
                        __m128 _p1 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 4), _scale);
                        __m128 _p2 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 8), _scale);
                        __m128 _p3 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 12), _scale);

                        __m128i _q0 = float2int8(_p0);
                        __m128i _q1 = float2int8(_p1);
                        __m128i _q2 = float2int8(_p2);
                        __m128i _q3 = float2int8(_p3);
                        __m128i _q01 = __lsx_vilvl_b(_q1, _q0);
                        __m128i _q23 = __lsx_vilvl_b(_q3, _q2);
                        __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp, 0);
                        pp += 16;
                        p0 += 16;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p = __lsx_vfmul_s(bfloat2float_lsx(p0), _scale);
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p), 0);
                        pp += 4;
                        p0 += 4;
                    }
                }

                if (elempack == 1)
                {
                    __m128 _scale0 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 0);
                    __m128 _scale1 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 1);
                    __m128 _scale2 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 2);
                    __m128 _scale3 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 3);
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m128 _p0 = bfloat2float_lsx(p0);
                        __m128 _p1 = bfloat2float_lsx(p1);
                        __m128 _p2 = bfloat2float_lsx(p2);
                        __m128 _p3 = bfloat2float_lsx(p3);
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
                        __m128i _p = __lsx_vldrepl_h(p0, 0);
                        _p = __lsx_vinsgr2vr_h(_p, *p1, 1);
                        _p = __lsx_vinsgr2vr_h(_p, *p2, 2);
                        _p = __lsx_vinsgr2vr_h(_p, *p3, 3);
                        ((int*)pp)[0] = __lsx_vpickve2gr_w(float2int8(__lsx_vfmul_s(bfloat2float_lsx(_p), _scale)), 0);
                        pp += 4;
                        p0++;
                        p1++;
                        p2++;
                        p3++;
                    }
                }
            }
        }
#endif // __loongarch_sx
        for (; ii + 1 < max_ii; ii += 2)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k;
            const unsigned short* p1 = p0 + A_hstep;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const unsigned short* p0a = p0;
                const unsigned short* p1a = p1;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(*p0a++);
                    float v1 = bfloat16_to_float32(*p1a++);
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
                    float v00 = bfloat16_to_float32(p0[0]);
                    float v01 = bfloat16_to_float32(p0[1]);
                    float v02 = bfloat16_to_float32(p0[2]);
                    float v03 = bfloat16_to_float32(p0[3]);
                    float v10 = bfloat16_to_float32(p1[0]);
                    float v11 = bfloat16_to_float32(p1[1]);
                    float v12 = bfloat16_to_float32(p1[2]);
                    float v13 = bfloat16_to_float32(p1[3]);
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
                    float v0 = bfloat16_to_float32(*p0++);
                    float v1 = bfloat16_to_float32(*p1++);
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale1);
                    pp += 2;
                }
            }
        }
        for (; ii < max_ii; ii++)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                const unsigned short* p0a = p0;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(*p0a++);
                    absmax0 = std::max(absmax0, fabsf(v0));
                }

                const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                *pd++ = absmax0 / 127.f;

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float v0 = bfloat16_to_float32(p0[0]);
                    float v1 = bfloat16_to_float32(p0[1]);
                    float v2 = bfloat16_to_float32(p0[2]);
                    float v3 = bfloat16_to_float32(p0[3]);
                    pp[0] = float2int8(v0 * scale0);
                    pp[1] = float2int8(v1 * scale0);
                    pp[2] = float2int8(v2 * scale0);
                    pp[3] = float2int8(v3 * scale0);
                    pp += 4;
                    p0 += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    float v0 = bfloat16_to_float32(*p0++);
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
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k * 8;
            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                __m128 _absmax0123 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax4567 = (__m128)__lsx_vreplgr2vr_w(0);

                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m128 _s = __lsx_vreplfr2vr_s(*psa++);
                    __m128 _p0 = __lsx_vfmul_s(bfloat2float_lsx(p0a), _s);
                    __m128 _p1 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 4), _s);
                    _absmax0123 = __lsx_vfmax_s(_absmax0123, (__m128)__lsx_vand_v((__m128i)_p0, _abs_mask));
                    _absmax4567 = __lsx_vfmax_s(_absmax4567, (__m128)__lsx_vand_v((__m128i)_p1, _abs_mask));
                    p0a += 8;
                }

                const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                __lsx_vst(__lsx_vfdiv_s(_absmax0123, _v127), pd, 0);
                __lsx_vst(__lsx_vfdiv_s(_absmax4567, _v127), pd + 4, 0);
                pd += 8;

                const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax03 = (__m128)__lsx_vbitsel_v((__m128i)_absmax0123, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax0123, _zero));
                __m128 _absmax47 = (__m128)__lsx_vbitsel_v((__m128i)_absmax4567, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax4567, _zero));
                __m128 _scale03 = __lsx_vfdiv_s(_v127, _absmax03);
                __m128 _scale47 = __lsx_vfdiv_s(_v127, _absmax47);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = __lsx_vfmul_s(bfloat2float_lsx(p0), __lsx_vreplfr2vr_s(ps[0]));
                    __m128 _p1 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 8), __lsx_vreplfr2vr_s(ps[1]));
                    __m128 _p2 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 16), __lsx_vreplfr2vr_s(ps[2]));
                    __m128 _p3 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 24), __lsx_vreplfr2vr_s(ps[3]));
                    __m128 _p4 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 4), __lsx_vreplfr2vr_s(ps[0]));
                    __m128 _p5 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 12), __lsx_vreplfr2vr_s(ps[1]));
                    __m128 _p6 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 20), __lsx_vreplfr2vr_s(ps[2]));
                    __m128 _p7 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 28), __lsx_vreplfr2vr_s(ps[3]));
                    __m128i _q0 = float2int8(__lsx_vfmul_s(_p0, _scale03));
                    __m128i _q1 = float2int8(__lsx_vfmul_s(_p1, _scale03));
                    __m128i _q2 = float2int8(__lsx_vfmul_s(_p2, _scale03));
                    __m128i _q3 = float2int8(__lsx_vfmul_s(_p3, _scale03));
                    __m128i _q01 = __lsx_vilvl_b(_q1, _q0);
                    __m128i _q23 = __lsx_vilvl_b(_q3, _q2);
                    __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp, 0);
                    _q0 = float2int8(__lsx_vfmul_s(_p4, _scale47));
                    _q1 = float2int8(__lsx_vfmul_s(_p5, _scale47));
                    _q2 = float2int8(__lsx_vfmul_s(_p6, _scale47));
                    _q3 = float2int8(__lsx_vfmul_s(_p7, _scale47));
                    _q01 = __lsx_vilvl_b(_q1, _q0);
                    _q23 = __lsx_vilvl_b(_q3, _q2);
                    __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp + 16, 0);
                    pp += 32;
                    p0 += 32;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128 _s = __lsx_vreplfr2vr_s(*ps++);
                    __m128 _p0 = __lsx_vfmul_s(bfloat2float_lsx(p0), _s);
                    __m128 _p1 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 4), _s);
                    __m128i _q0 = float2int8(__lsx_vfmul_s(_p0, _scale03));
                    __m128i _q1 = float2int8(__lsx_vfmul_s(_p1, _scale47));
                    ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)_q0, 0);
                    ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)_q1, 0);
                    pp += 8;
                    p0 += 8;
                }
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k * 4;
            const unsigned short* p1 = p0 + A_hstep * 4;
            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                __m128 _absmax0123 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax4567 = (__m128)__lsx_vreplgr2vr_w(0);

                const unsigned short* p0a = p0;
                const unsigned short* p1a = p1;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m128 _s = __lsx_vreplfr2vr_s(*psa++);
                    __m128 _p0 = __lsx_vfmul_s(bfloat2float_lsx(p0a), _s);
                    __m128 _p1 = __lsx_vfmul_s(bfloat2float_lsx(p1a), _s);
                    _absmax0123 = __lsx_vfmax_s(_absmax0123, (__m128)__lsx_vand_v((__m128i)_p0, _abs_mask));
                    _absmax4567 = __lsx_vfmax_s(_absmax4567, (__m128)__lsx_vand_v((__m128i)_p1, _abs_mask));
                    p0a += 4;
                    p1a += 4;
                }

                const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                __lsx_vst(__lsx_vfdiv_s(_absmax0123, _v127), pd, 0);
                __lsx_vst(__lsx_vfdiv_s(_absmax4567, _v127), pd + 4, 0);
                pd += 8;

                const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax03 = (__m128)__lsx_vbitsel_v((__m128i)_absmax0123, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax0123, _zero));
                __m128 _absmax47 = (__m128)__lsx_vbitsel_v((__m128i)_absmax4567, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax4567, _zero));
                __m128 _scale03 = __lsx_vfdiv_s(_v127, _absmax03);
                __m128 _scale47 = __lsx_vfdiv_s(_v127, _absmax47);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = __lsx_vfmul_s(bfloat2float_lsx(p0), __lsx_vreplfr2vr_s(ps[0]));
                    __m128 _p1 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 4), __lsx_vreplfr2vr_s(ps[1]));
                    __m128 _p2 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 8), __lsx_vreplfr2vr_s(ps[2]));
                    __m128 _p3 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 12), __lsx_vreplfr2vr_s(ps[3]));
                    __m128 _p4 = __lsx_vfmul_s(bfloat2float_lsx(p1), __lsx_vreplfr2vr_s(ps[0]));
                    __m128 _p5 = __lsx_vfmul_s(bfloat2float_lsx(p1 + 4), __lsx_vreplfr2vr_s(ps[1]));
                    __m128 _p6 = __lsx_vfmul_s(bfloat2float_lsx(p1 + 8), __lsx_vreplfr2vr_s(ps[2]));
                    __m128 _p7 = __lsx_vfmul_s(bfloat2float_lsx(p1 + 12), __lsx_vreplfr2vr_s(ps[3]));
                    __m128i _q0 = float2int8(__lsx_vfmul_s(_p0, _scale03));
                    __m128i _q1 = float2int8(__lsx_vfmul_s(_p1, _scale03));
                    __m128i _q2 = float2int8(__lsx_vfmul_s(_p2, _scale03));
                    __m128i _q3 = float2int8(__lsx_vfmul_s(_p3, _scale03));
                    __m128i _q01 = __lsx_vilvl_b(_q1, _q0);
                    __m128i _q23 = __lsx_vilvl_b(_q3, _q2);
                    __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp, 0);
                    _q0 = float2int8(__lsx_vfmul_s(_p4, _scale47));
                    _q1 = float2int8(__lsx_vfmul_s(_p5, _scale47));
                    _q2 = float2int8(__lsx_vfmul_s(_p6, _scale47));
                    _q3 = float2int8(__lsx_vfmul_s(_p7, _scale47));
                    _q01 = __lsx_vilvl_b(_q1, _q0);
                    _q23 = __lsx_vilvl_b(_q3, _q2);
                    __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp + 16, 0);
                    pp += 32;
                    p0 += 16;
                    p1 += 16;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128 _s = __lsx_vreplfr2vr_s(*ps++);
                    __m128 _p0 = __lsx_vfmul_s(bfloat2float_lsx(p0), _s);
                    __m128 _p1 = __lsx_vfmul_s(bfloat2float_lsx(p1), _s);
                    __m128i _q0 = float2int8(__lsx_vfmul_s(_p0, _scale03));
                    __m128i _q1 = float2int8(__lsx_vfmul_s(_p1, _scale47));
                    ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)_q0, 0);
                    ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)_q1, 0);
                    pp += 8;
                    p0 += 4;
                    p1 += 4;
                }
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k;
            const unsigned short* p1 = p0 + A_hstep;
            const unsigned short* p2 = p1 + A_hstep;
            const unsigned short* p3 = p2 + A_hstep;
            const unsigned short* p4 = p3 + A_hstep;
            const unsigned short* p5 = p4 + A_hstep;
            const unsigned short* p6 = p5 + A_hstep;
            const unsigned short* p7 = p6 + A_hstep;

            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                __m128 _absmax0 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax1 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax2 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax3 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax4 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax5 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax6 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax7 = (__m128)__lsx_vreplgr2vr_w(0);

                const unsigned short* p0a = p0;
                const unsigned short* p1a = p1;
                const unsigned short* p2a = p2;
                const unsigned short* p3a = p3;
                const unsigned short* p4a = p4;
                const unsigned short* p5a = p5;
                const unsigned short* p6a = p6;
                const unsigned short* p7a = p7;
                const float* psa = ps;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = bfloat2float_lsx(p0a);
                    __m128 _p1 = bfloat2float_lsx(p1a);
                    __m128 _p2 = bfloat2float_lsx(p2a);
                    __m128 _p3 = bfloat2float_lsx(p3a);
                    __m128 _p4 = bfloat2float_lsx(p4a);
                    __m128 _p5 = bfloat2float_lsx(p5a);
                    __m128 _p6 = bfloat2float_lsx(p6a);
                    __m128 _p7 = bfloat2float_lsx(p7a);
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

                transpose4x4_ps(_absmax0, _absmax1, _absmax2, _absmax3);
                transpose4x4_ps(_absmax4, _absmax5, _absmax6, _absmax7);
                _absmax0 = __lsx_vfmax_s(__lsx_vfmax_s(_absmax0, _absmax1), __lsx_vfmax_s(_absmax2, _absmax3));
                _absmax1 = __lsx_vfmax_s(__lsx_vfmax_s(_absmax4, _absmax5), __lsx_vfmax_s(_absmax6, _absmax7));

                for (; kk < max_kk0; kk++)
                {
                    const float s = *psa++;
                    __m128i _p0 = __lsx_vldrepl_h(p0a, 0);
                    _p0 = __lsx_vinsgr2vr_h(_p0, *p1a, 1);
                    _p0 = __lsx_vinsgr2vr_h(_p0, *p2a, 2);
                    _p0 = __lsx_vinsgr2vr_h(_p0, *p3a, 3);
                    __m128i _p1 = __lsx_vldrepl_h(p4a, 0);
                    _p1 = __lsx_vinsgr2vr_h(_p1, *p5a, 1);
                    _p1 = __lsx_vinsgr2vr_h(_p1, *p6a, 2);
                    _p1 = __lsx_vinsgr2vr_h(_p1, *p7a, 3);
                    __m128 _s = __lsx_vreplfr2vr_s(s);
                    __m128 _p0f = __lsx_vfmul_s(bfloat2float_lsx(_p0), _s);
                    __m128 _p1f = __lsx_vfmul_s(bfloat2float_lsx(_p1), _s);
                    _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_p0f, _abs_mask));
                    _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_p1f, _abs_mask));
                    p0a++;
                    p1a++;
                    p2a++;
                    p3a++;
                    p4a++;
                    p5a++;
                    p6a++;
                    p7a++;
                }

                const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                __lsx_vst(__lsx_vfdiv_s(_absmax0, _v127), pd, 0);
                __lsx_vst(__lsx_vfdiv_s(_absmax1, _v127), pd + 4, 0);
                pd += 8;

                const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax03 = (__m128)__lsx_vbitsel_v((__m128i)_absmax0, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax0, _zero));
                __m128 _absmax47 = (__m128)__lsx_vbitsel_v((__m128i)_absmax1, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax1, _zero));
                __m128 _scale03 = __lsx_vfdiv_s(_v127, _absmax03);
                __m128 _scale47 = __lsx_vfdiv_s(_v127, _absmax47);
                __m128 _scale0 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 0);
                __m128 _scale1 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 1);
                __m128 _scale2 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 2);
                __m128 _scale3 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 3);
                __m128 _scale4 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 0);
                __m128 _scale5 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 1);
                __m128 _scale6 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 2);
                __m128 _scale7 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 3);
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = bfloat2float_lsx(p0);
                    __m128 _p1 = bfloat2float_lsx(p1);
                    __m128 _p2 = bfloat2float_lsx(p2);
                    __m128 _p3 = bfloat2float_lsx(p3);
                    __m128 _p4 = bfloat2float_lsx(p4);
                    __m128 _p5 = bfloat2float_lsx(p5);
                    __m128 _p6 = bfloat2float_lsx(p6);
                    __m128 _p7 = bfloat2float_lsx(p7);
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
                    __m128i _p0 = __lsx_vldrepl_h(p0, 0);
                    _p0 = __lsx_vinsgr2vr_h(_p0, *p1, 1);
                    _p0 = __lsx_vinsgr2vr_h(_p0, *p2, 2);
                    _p0 = __lsx_vinsgr2vr_h(_p0, *p3, 3);
                    __m128i _p1 = __lsx_vldrepl_h(p4, 0);
                    _p1 = __lsx_vinsgr2vr_h(_p1, *p5, 1);
                    _p1 = __lsx_vinsgr2vr_h(_p1, *p6, 2);
                    _p1 = __lsx_vinsgr2vr_h(_p1, *p7, 3);
                    __m128 _s = __lsx_vreplfr2vr_s(*ps++);
                    __m128 _p0f = __lsx_vfmul_s(bfloat2float_lsx(_p0), _s);
                    __m128 _p1f = __lsx_vfmul_s(bfloat2float_lsx(_p1), _s);
                    ((int64_t*)pp)[0] = float2int8(__lsx_vfmul_s(_p0f, _scale03), __lsx_vfmul_s(_p1f, _scale47));
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
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k * 4;
            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                __m128 _absmax0123 = (__m128)__lsx_vreplgr2vr_w(0);

                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m128 _p = __lsx_vfmul_s(bfloat2float_lsx(p0a), __lsx_vreplfr2vr_s(*psa++));
                    _absmax0123 = __lsx_vfmax_s(_absmax0123, (__m128)__lsx_vand_v((__m128i)_p, _abs_mask));
                    p0a += 4;
                }

                const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                __lsx_vst(__lsx_vfdiv_s(_absmax0123, _v127), pd, 0);
                pd += 4;

                const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax = (__m128)__lsx_vbitsel_v((__m128i)_absmax0123, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax0123, _zero));
                __m128 _scale = __lsx_vfdiv_s(_v127, _absmax);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = __lsx_vfmul_s(bfloat2float_lsx(p0), __lsx_vreplfr2vr_s(ps[0]));
                    __m128 _p1 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 4), __lsx_vreplfr2vr_s(ps[1]));
                    __m128 _p2 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 8), __lsx_vreplfr2vr_s(ps[2]));
                    __m128 _p3 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 12), __lsx_vreplfr2vr_s(ps[3]));
                    __m128i _q0 = float2int8(__lsx_vfmul_s(_p0, _scale));
                    __m128i _q1 = float2int8(__lsx_vfmul_s(_p1, _scale));
                    __m128i _q2 = float2int8(__lsx_vfmul_s(_p2, _scale));
                    __m128i _q3 = float2int8(__lsx_vfmul_s(_p3, _scale));
                    __m128i _q01 = __lsx_vilvl_b(_q1, _q0);
                    __m128i _q23 = __lsx_vilvl_b(_q3, _q2);
                    __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp, 0);
                    pp += 16;
                    p0 += 16;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m128 _p = __lsx_vfmul_s(bfloat2float_lsx(p0), __lsx_vreplfr2vr_s(*ps++));
                    __m128i _q = float2int8(__lsx_vfmul_s(_p, _scale));
                    ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)_q, 0);
                    pp += 4;
                    p0 += 4;
                }
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k;
            const unsigned short* p1 = p0 + A_hstep;
            const unsigned short* p2 = p1 + A_hstep;
            const unsigned short* p3 = p2 + A_hstep;

            const float* ps = input_scale_ptr;

            for (int g = 0; g < local_block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                __m128 _absmax0 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax1 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax2 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax3 = (__m128)__lsx_vreplgr2vr_w(0);

                const unsigned short* p0a = p0;
                const unsigned short* p1a = p1;
                const unsigned short* p2a = p2;
                const unsigned short* p3a = p3;
                const float* psa = ps;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = bfloat2float_lsx(p0a);
                    __m128 _p1 = bfloat2float_lsx(p1a);
                    __m128 _p2 = bfloat2float_lsx(p2a);
                    __m128 _p3 = bfloat2float_lsx(p3a);
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

                transpose4x4_ps(_absmax0, _absmax1, _absmax2, _absmax3);
                _absmax0 = __lsx_vfmax_s(__lsx_vfmax_s(_absmax0, _absmax1), __lsx_vfmax_s(_absmax2, _absmax3));

                for (; kk < max_kk0; kk++)
                {
                    __m128i _p = __lsx_vldrepl_h(p0a, 0);
                    _p = __lsx_vinsgr2vr_h(_p, *p1a, 1);
                    _p = __lsx_vinsgr2vr_h(_p, *p2a, 2);
                    _p = __lsx_vinsgr2vr_h(_p, *p3a, 3);
                    __m128 _pf = __lsx_vfmul_s(bfloat2float_lsx(_p), __lsx_vreplfr2vr_s(*psa++));
                    _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_pf, _abs_mask));
                    p0a++;
                    p1a++;
                    p2a++;
                    p3a++;
                }

                const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                __lsx_vst(__lsx_vfdiv_s(_absmax0, _v127), pd, 0);
                pd += 4;

                const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _absmax = (__m128)__lsx_vbitsel_v((__m128i)_absmax0, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax0, _zero));
                __m128 _scale = __lsx_vfdiv_s(_v127, _absmax);
                __m128 _scale0 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 0);
                __m128 _scale1 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 1);
                __m128 _scale2 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 2);
                __m128 _scale3 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 3);
                kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m128 _p0 = bfloat2float_lsx(p0);
                    __m128 _p1 = bfloat2float_lsx(p1);
                    __m128 _p2 = bfloat2float_lsx(p2);
                    __m128 _p3 = bfloat2float_lsx(p3);
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
                    __m128i _p = __lsx_vldrepl_h(p0, 0);
                    _p = __lsx_vinsgr2vr_h(_p, *p1, 1);
                    _p = __lsx_vinsgr2vr_h(_p, *p2, 2);
                    _p = __lsx_vinsgr2vr_h(_p, *p3, 3);
                    __m128 _pf = __lsx_vfmul_s(bfloat2float_lsx(_p), __lsx_vreplfr2vr_s(*ps++));
                    ((int*)pp)[0] = __lsx_vpickve2gr_w(float2int8(__lsx_vfmul_s(_pf, _scale)), 0);
                    pp += 4;
                    p0++;
                    p1++;
                    p2++;
                    p3++;
                }
            }
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k;
        const unsigned short* p1 = p0 + A_hstep;

        const float* ps = input_scale_ptr;

        for (int g = 0; g < local_block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            const unsigned short* p0a = p0;
            const unsigned short* p1a = p1;
            const float* psa = ps;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                float v0 = bfloat16_to_float32(*p0a++);
                float v1 = bfloat16_to_float32(*p1a++);
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
                float v00 = bfloat16_to_float32(p0[0]);
                float v01 = bfloat16_to_float32(p0[1]);
                float v02 = bfloat16_to_float32(p0[2]);
                float v03 = bfloat16_to_float32(p0[3]);
                float v10 = bfloat16_to_float32(p1[0]);
                float v11 = bfloat16_to_float32(p1[1]);
                float v12 = bfloat16_to_float32(p1[2]);
                float v13 = bfloat16_to_float32(p1[3]);
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
                float v0 = bfloat16_to_float32(*p0++);
                float v1 = bfloat16_to_float32(*p1++);
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
        const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k;

        const float* ps = input_scale_ptr;

        for (int g = 0; g < local_block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax0 = 0.f;
            const unsigned short* p0a = p0;
            const float* psa = ps;
            for (int kk = 0; kk < max_kk0; kk++)
            {
                float v0 = bfloat16_to_float32(*p0a++);
                const float s = *psa++;
                absmax0 = std::max(absmax0, fabsf(v0) * s);
            }

            const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
            *pd++ = absmax0 / 127.f;

            int kk = 0;
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float v0 = bfloat16_to_float32(p0[0]);
                float v1 = bfloat16_to_float32(p0[1]);
                float v2 = bfloat16_to_float32(p0[2]);
                float v3 = bfloat16_to_float32(p0[3]);
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
                float v0 = bfloat16_to_float32(*p0++);
                v0 *= *ps++;
                *pp++ = float2int8(v0 * scale0);
            }
        }
    }
}

static void transpose_quantize_A_tile_wq_int8_bf16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
#if __loongarch_sx
    const int elempack = A.elempack;
#endif // __loongarch_sx
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int local_block_count = (max_kk + block_size - 1) / block_size;

    if (input_scales.empty())
    {
        int ii = 0;
#if __loongarch_sx
        for (; ii + 7 < max_ii; ii += 8)
        {
            if (elempack == 8)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 8;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax00 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax01 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax10 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax11 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax20 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax21 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax30 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax31 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax40 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax41 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax50 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax51 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax60 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax61 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax70 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax71 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        __m128 _p00 = bfloat2float_lsx(p0a);
                        _absmax00 = __lsx_vfmax_s(_absmax00, (__m128)__lsx_vand_v((__m128i)_p00, _abs_mask));
                        __m128 _p01 = bfloat2float_lsx(p0a + 4);
                        _absmax01 = __lsx_vfmax_s(_absmax01, (__m128)__lsx_vand_v((__m128i)_p01, _abs_mask));
                        __m128 _p10 = bfloat2float_lsx(p0a + 8);
                        _absmax10 = __lsx_vfmax_s(_absmax10, (__m128)__lsx_vand_v((__m128i)_p10, _abs_mask));
                        __m128 _p11 = bfloat2float_lsx(p0a + 12);
                        _absmax11 = __lsx_vfmax_s(_absmax11, (__m128)__lsx_vand_v((__m128i)_p11, _abs_mask));
                        __m128 _p20 = bfloat2float_lsx(p0a + 16);
                        _absmax20 = __lsx_vfmax_s(_absmax20, (__m128)__lsx_vand_v((__m128i)_p20, _abs_mask));
                        __m128 _p21 = bfloat2float_lsx(p0a + 20);
                        _absmax21 = __lsx_vfmax_s(_absmax21, (__m128)__lsx_vand_v((__m128i)_p21, _abs_mask));
                        __m128 _p30 = bfloat2float_lsx(p0a + 24);
                        _absmax30 = __lsx_vfmax_s(_absmax30, (__m128)__lsx_vand_v((__m128i)_p30, _abs_mask));
                        __m128 _p31 = bfloat2float_lsx(p0a + 28);
                        _absmax31 = __lsx_vfmax_s(_absmax31, (__m128)__lsx_vand_v((__m128i)_p31, _abs_mask));
                        __m128 _p40 = bfloat2float_lsx(p0a + 32);
                        _absmax40 = __lsx_vfmax_s(_absmax40, (__m128)__lsx_vand_v((__m128i)_p40, _abs_mask));
                        __m128 _p41 = bfloat2float_lsx(p0a + 36);
                        _absmax41 = __lsx_vfmax_s(_absmax41, (__m128)__lsx_vand_v((__m128i)_p41, _abs_mask));
                        __m128 _p50 = bfloat2float_lsx(p0a + 40);
                        _absmax50 = __lsx_vfmax_s(_absmax50, (__m128)__lsx_vand_v((__m128i)_p50, _abs_mask));
                        __m128 _p51 = bfloat2float_lsx(p0a + 44);
                        _absmax51 = __lsx_vfmax_s(_absmax51, (__m128)__lsx_vand_v((__m128i)_p51, _abs_mask));
                        __m128 _p60 = bfloat2float_lsx(p0a + 48);
                        _absmax60 = __lsx_vfmax_s(_absmax60, (__m128)__lsx_vand_v((__m128i)_p60, _abs_mask));
                        __m128 _p61 = bfloat2float_lsx(p0a + 52);
                        _absmax61 = __lsx_vfmax_s(_absmax61, (__m128)__lsx_vand_v((__m128i)_p61, _abs_mask));
                        __m128 _p70 = bfloat2float_lsx(p0a + 56);
                        _absmax70 = __lsx_vfmax_s(_absmax70, (__m128)__lsx_vand_v((__m128i)_p70, _abs_mask));
                        __m128 _p71 = bfloat2float_lsx(p0a + 60);
                        _absmax71 = __lsx_vfmax_s(_absmax71, (__m128)__lsx_vand_v((__m128i)_p71, _abs_mask));
                        p0a += A_hstep * 8;
                    }
                    _absmax00 = __lsx_vfmax_s(_absmax00, _absmax01);
                    _absmax10 = __lsx_vfmax_s(_absmax10, _absmax11);
                    _absmax20 = __lsx_vfmax_s(_absmax20, _absmax21);
                    _absmax30 = __lsx_vfmax_s(_absmax30, _absmax31);
                    _absmax40 = __lsx_vfmax_s(_absmax40, _absmax41);
                    _absmax50 = __lsx_vfmax_s(_absmax50, _absmax51);
                    _absmax60 = __lsx_vfmax_s(_absmax60, _absmax61);
                    _absmax70 = __lsx_vfmax_s(_absmax70, _absmax71);

                    transpose4x4_ps(_absmax00, _absmax10, _absmax20, _absmax30);
                    transpose4x4_ps(_absmax40, _absmax50, _absmax60, _absmax70);
                    __m128 _absmax03 = __lsx_vfmax_s(__lsx_vfmax_s(_absmax00, _absmax10), __lsx_vfmax_s(_absmax20, _absmax30));
                    __m128 _absmax47 = __lsx_vfmax_s(__lsx_vfmax_s(_absmax40, _absmax50), __lsx_vfmax_s(_absmax60, _absmax70));

                    const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                    __lsx_vst(__lsx_vfdiv_s(_absmax03, _v127), pd, 0);
                    __lsx_vst(__lsx_vfdiv_s(_absmax47, _v127), pd + 4, 0);
                    pd += 8;

                    const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                    _absmax03 = (__m128)__lsx_vbitsel_v((__m128i)_absmax03, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax03, _zero));
                    _absmax47 = (__m128)__lsx_vbitsel_v((__m128i)_absmax47, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax47, _zero));
                    __m128 _scale03 = __lsx_vfdiv_s(_v127, _absmax03);
                    __m128 _scale47 = __lsx_vfdiv_s(_v127, _absmax47);
                    __m128 _scale0 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 0);
                    __m128 _scale1 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 1);
                    __m128 _scale2 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 2);
                    __m128 _scale3 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 3);
                    __m128 _scale4 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 0);
                    __m128 _scale5 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 1);
                    __m128 _scale6 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 2);
                    __m128 _scale7 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 3);

                    int kk = 0;
                    for (; kk < max_kk0; kk += 8)
                    {
                        __m128 _p00 = bfloat2float_lsx(p0);
                        _p00 = __lsx_vfmul_s(_p00, _scale0);
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p00), 0);
                        __m128 _p01 = bfloat2float_lsx(p0 + 4);
                        _p01 = __lsx_vfmul_s(_p01, _scale0);
                        ((int*)pp)[8] = __lsx_vpickve2gr_w((__m128i)float2int8(_p01), 0);
                        __m128 _p10 = bfloat2float_lsx(p0 + 8);
                        _p10 = __lsx_vfmul_s(_p10, _scale1);
                        ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)float2int8(_p10), 0);
                        __m128 _p11 = bfloat2float_lsx(p0 + 12);
                        _p11 = __lsx_vfmul_s(_p11, _scale1);
                        ((int*)pp)[9] = __lsx_vpickve2gr_w((__m128i)float2int8(_p11), 0);
                        __m128 _p20 = bfloat2float_lsx(p0 + 16);
                        _p20 = __lsx_vfmul_s(_p20, _scale2);
                        ((int*)pp)[2] = __lsx_vpickve2gr_w((__m128i)float2int8(_p20), 0);
                        __m128 _p21 = bfloat2float_lsx(p0 + 20);
                        _p21 = __lsx_vfmul_s(_p21, _scale2);
                        ((int*)pp)[10] = __lsx_vpickve2gr_w((__m128i)float2int8(_p21), 0);
                        __m128 _p30 = bfloat2float_lsx(p0 + 24);
                        _p30 = __lsx_vfmul_s(_p30, _scale3);
                        ((int*)pp)[3] = __lsx_vpickve2gr_w((__m128i)float2int8(_p30), 0);
                        __m128 _p31 = bfloat2float_lsx(p0 + 28);
                        _p31 = __lsx_vfmul_s(_p31, _scale3);
                        ((int*)pp)[11] = __lsx_vpickve2gr_w((__m128i)float2int8(_p31), 0);
                        __m128 _p40 = bfloat2float_lsx(p0 + 32);
                        _p40 = __lsx_vfmul_s(_p40, _scale4);
                        ((int*)pp)[4] = __lsx_vpickve2gr_w((__m128i)float2int8(_p40), 0);
                        __m128 _p41 = bfloat2float_lsx(p0 + 36);
                        _p41 = __lsx_vfmul_s(_p41, _scale4);
                        ((int*)pp)[12] = __lsx_vpickve2gr_w((__m128i)float2int8(_p41), 0);
                        __m128 _p50 = bfloat2float_lsx(p0 + 40);
                        _p50 = __lsx_vfmul_s(_p50, _scale5);
                        ((int*)pp)[5] = __lsx_vpickve2gr_w((__m128i)float2int8(_p50), 0);
                        __m128 _p51 = bfloat2float_lsx(p0 + 44);
                        _p51 = __lsx_vfmul_s(_p51, _scale5);
                        ((int*)pp)[13] = __lsx_vpickve2gr_w((__m128i)float2int8(_p51), 0);
                        __m128 _p60 = bfloat2float_lsx(p0 + 48);
                        _p60 = __lsx_vfmul_s(_p60, _scale6);
                        ((int*)pp)[6] = __lsx_vpickve2gr_w((__m128i)float2int8(_p60), 0);
                        __m128 _p61 = bfloat2float_lsx(p0 + 52);
                        _p61 = __lsx_vfmul_s(_p61, _scale6);
                        ((int*)pp)[14] = __lsx_vpickve2gr_w((__m128i)float2int8(_p61), 0);
                        __m128 _p70 = bfloat2float_lsx(p0 + 56);
                        _p70 = __lsx_vfmul_s(_p70, _scale7);
                        ((int*)pp)[7] = __lsx_vpickve2gr_w((__m128i)float2int8(_p70), 0);
                        __m128 _p71 = bfloat2float_lsx(p0 + 60);
                        _p71 = __lsx_vfmul_s(_p71, _scale7);
                        ((int*)pp)[15] = __lsx_vpickve2gr_w((__m128i)float2int8(_p71), 0);
                        pp += 64;
                        p0 += A_hstep * 8;
                    }
                }
            }
            if (elempack == 4)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 4;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax00 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax10 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax20 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax30 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax40 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax50 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax60 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax70 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        __m128 _p00 = bfloat2float_lsx(p0a);
                        _absmax00 = __lsx_vfmax_s(_absmax00, (__m128)__lsx_vand_v((__m128i)_p00, _abs_mask));
                        __m128 _p10 = bfloat2float_lsx(p0a + 4);
                        _absmax10 = __lsx_vfmax_s(_absmax10, (__m128)__lsx_vand_v((__m128i)_p10, _abs_mask));
                        __m128 _p20 = bfloat2float_lsx(p0a + 8);
                        _absmax20 = __lsx_vfmax_s(_absmax20, (__m128)__lsx_vand_v((__m128i)_p20, _abs_mask));
                        __m128 _p30 = bfloat2float_lsx(p0a + 12);
                        _absmax30 = __lsx_vfmax_s(_absmax30, (__m128)__lsx_vand_v((__m128i)_p30, _abs_mask));
                        __m128 _p40 = bfloat2float_lsx(p0a + 16);
                        _absmax40 = __lsx_vfmax_s(_absmax40, (__m128)__lsx_vand_v((__m128i)_p40, _abs_mask));
                        __m128 _p50 = bfloat2float_lsx(p0a + 20);
                        _absmax50 = __lsx_vfmax_s(_absmax50, (__m128)__lsx_vand_v((__m128i)_p50, _abs_mask));
                        __m128 _p60 = bfloat2float_lsx(p0a + 24);
                        _absmax60 = __lsx_vfmax_s(_absmax60, (__m128)__lsx_vand_v((__m128i)_p60, _abs_mask));
                        __m128 _p70 = bfloat2float_lsx(p0a + 28);
                        _absmax70 = __lsx_vfmax_s(_absmax70, (__m128)__lsx_vand_v((__m128i)_p70, _abs_mask));
                        p0a += A_hstep * 4;
                    }
                    transpose4x4_ps(_absmax00, _absmax10, _absmax20, _absmax30);
                    transpose4x4_ps(_absmax40, _absmax50, _absmax60, _absmax70);
                    __m128 _absmax03 = __lsx_vfmax_s(__lsx_vfmax_s(_absmax00, _absmax10), __lsx_vfmax_s(_absmax20, _absmax30));
                    __m128 _absmax47 = __lsx_vfmax_s(__lsx_vfmax_s(_absmax40, _absmax50), __lsx_vfmax_s(_absmax60, _absmax70));

                    const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                    __lsx_vst(__lsx_vfdiv_s(_absmax03, _v127), pd, 0);
                    __lsx_vst(__lsx_vfdiv_s(_absmax47, _v127), pd + 4, 0);
                    pd += 8;

                    const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                    _absmax03 = (__m128)__lsx_vbitsel_v((__m128i)_absmax03, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax03, _zero));
                    _absmax47 = (__m128)__lsx_vbitsel_v((__m128i)_absmax47, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax47, _zero));
                    __m128 _scale03 = __lsx_vfdiv_s(_v127, _absmax03);
                    __m128 _scale47 = __lsx_vfdiv_s(_v127, _absmax47);
                    __m128 _scale0 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 0);
                    __m128 _scale1 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 1);
                    __m128 _scale2 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 2);
                    __m128 _scale3 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 3);
                    __m128 _scale4 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 0);
                    __m128 _scale5 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 1);
                    __m128 _scale6 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 2);
                    __m128 _scale7 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 3);

                    int kk = 0;
                    for (; kk < max_kk0; kk += 4)
                    {
                        __m128 _p00 = bfloat2float_lsx(p0);
                        _p00 = __lsx_vfmul_s(_p00, _scale0);
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p00), 0);
                        __m128 _p10 = bfloat2float_lsx(p0 + 4);
                        _p10 = __lsx_vfmul_s(_p10, _scale1);
                        ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)float2int8(_p10), 0);
                        __m128 _p20 = bfloat2float_lsx(p0 + 8);
                        _p20 = __lsx_vfmul_s(_p20, _scale2);
                        ((int*)pp)[2] = __lsx_vpickve2gr_w((__m128i)float2int8(_p20), 0);
                        __m128 _p30 = bfloat2float_lsx(p0 + 12);
                        _p30 = __lsx_vfmul_s(_p30, _scale3);
                        ((int*)pp)[3] = __lsx_vpickve2gr_w((__m128i)float2int8(_p30), 0);
                        __m128 _p40 = bfloat2float_lsx(p0 + 16);
                        _p40 = __lsx_vfmul_s(_p40, _scale4);
                        ((int*)pp)[4] = __lsx_vpickve2gr_w((__m128i)float2int8(_p40), 0);
                        __m128 _p50 = bfloat2float_lsx(p0 + 20);
                        _p50 = __lsx_vfmul_s(_p50, _scale5);
                        ((int*)pp)[5] = __lsx_vpickve2gr_w((__m128i)float2int8(_p50), 0);
                        __m128 _p60 = bfloat2float_lsx(p0 + 24);
                        _p60 = __lsx_vfmul_s(_p60, _scale6);
                        ((int*)pp)[6] = __lsx_vpickve2gr_w((__m128i)float2int8(_p60), 0);
                        __m128 _p70 = bfloat2float_lsx(p0 + 28);
                        _p70 = __lsx_vfmul_s(_p70, _scale7);
                        ((int*)pp)[7] = __lsx_vpickve2gr_w((__m128i)float2int8(_p70), 0);
                        pp += 32;
                        p0 += A_hstep * 4;
                    }
                }
            }

            if (elempack == 1)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax0 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax1 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    int kk = 0;
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p0 = bfloat2float_lsx(p0a);
                        __m128 _p1 = bfloat2float_lsx(p0a + 4);
                        _absmax0 = __lsx_vfmax_s(_absmax0, (__m128)__lsx_vand_v((__m128i)_p0, _abs_mask));
                        _absmax1 = __lsx_vfmax_s(_absmax1, (__m128)__lsx_vand_v((__m128i)_p1, _abs_mask));
                        p0a += A_hstep;
                    }

                    const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                    __lsx_vst(__lsx_vfdiv_s(_absmax0, _v127), pd, 0);
                    __lsx_vst(__lsx_vfdiv_s(_absmax1, _v127), pd + 4, 0);
                    pd += 8;

                    const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax03 = (__m128)__lsx_vbitsel_v((__m128i)_absmax0, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax0, _zero));
                    __m128 _absmax47 = (__m128)__lsx_vbitsel_v((__m128i)_absmax1, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax1, _zero));
                    __m128 _scale03 = __lsx_vfdiv_s(_v127, _absmax03);
                    __m128 _scale47 = __lsx_vfdiv_s(_v127, _absmax47);
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        const unsigned short* p1 = p0 + A_hstep;
                        const unsigned short* p2 = p1 + A_hstep;
                        const unsigned short* p3 = p2 + A_hstep;
                        __m128 _p0 = bfloat2float_lsx(p0);
                        __m128 _p1 = bfloat2float_lsx(p1);
                        __m128 _p2 = bfloat2float_lsx(p2);
                        __m128 _p3 = bfloat2float_lsx(p3);
                        __m128 _p4 = bfloat2float_lsx(p0 + 4);
                        __m128 _p5 = bfloat2float_lsx(p1 + 4);
                        __m128 _p6 = bfloat2float_lsx(p2 + 4);
                        __m128 _p7 = bfloat2float_lsx(p3 + 4);
                        __m128i _q0 = float2int8(__lsx_vfmul_s(_p0, _scale03));
                        __m128i _q1 = float2int8(__lsx_vfmul_s(_p1, _scale03));
                        __m128i _q2 = float2int8(__lsx_vfmul_s(_p2, _scale03));
                        __m128i _q3 = float2int8(__lsx_vfmul_s(_p3, _scale03));
                        __m128i _q01 = __lsx_vilvl_b(_q1, _q0);
                        __m128i _q23 = __lsx_vilvl_b(_q3, _q2);
                        __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp, 0);
                        _q0 = float2int8(__lsx_vfmul_s(_p4, _scale47));
                        _q1 = float2int8(__lsx_vfmul_s(_p5, _scale47));
                        _q2 = float2int8(__lsx_vfmul_s(_p6, _scale47));
                        _q3 = float2int8(__lsx_vfmul_s(_p7, _scale47));
                        _q01 = __lsx_vilvl_b(_q1, _q0);
                        _q23 = __lsx_vilvl_b(_q3, _q2);
                        __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp + 16, 0);
                        pp += 32;
                        p0 = p3 + A_hstep;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p0 = bfloat2float_lsx(p0);
                        __m128 _p1 = bfloat2float_lsx(p0 + 4);
                        __m128i _q0 = float2int8(__lsx_vfmul_s(_p0, _scale03));
                        __m128i _q1 = float2int8(__lsx_vfmul_s(_p1, _scale47));
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)_q0, 0);
                        ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)_q1, 0);
                        pp += 8;
                        p0 += A_hstep;
                    }
                }
            }
        }
        for (; ii + 3 < max_ii; ii += 4)
        {
            if (elempack == 8)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 8;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax00 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax01 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax10 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax11 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax20 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax21 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax30 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax31 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        __m128 _p00 = bfloat2float_lsx(p0a);
                        _absmax00 = __lsx_vfmax_s(_absmax00, (__m128)__lsx_vand_v((__m128i)_p00, _abs_mask));
                        __m128 _p01 = bfloat2float_lsx(p0a + 4);
                        _absmax01 = __lsx_vfmax_s(_absmax01, (__m128)__lsx_vand_v((__m128i)_p01, _abs_mask));
                        __m128 _p10 = bfloat2float_lsx(p0a + 8);
                        _absmax10 = __lsx_vfmax_s(_absmax10, (__m128)__lsx_vand_v((__m128i)_p10, _abs_mask));
                        __m128 _p11 = bfloat2float_lsx(p0a + 12);
                        _absmax11 = __lsx_vfmax_s(_absmax11, (__m128)__lsx_vand_v((__m128i)_p11, _abs_mask));
                        __m128 _p20 = bfloat2float_lsx(p0a + 16);
                        _absmax20 = __lsx_vfmax_s(_absmax20, (__m128)__lsx_vand_v((__m128i)_p20, _abs_mask));
                        __m128 _p21 = bfloat2float_lsx(p0a + 20);
                        _absmax21 = __lsx_vfmax_s(_absmax21, (__m128)__lsx_vand_v((__m128i)_p21, _abs_mask));
                        __m128 _p30 = bfloat2float_lsx(p0a + 24);
                        _absmax30 = __lsx_vfmax_s(_absmax30, (__m128)__lsx_vand_v((__m128i)_p30, _abs_mask));
                        __m128 _p31 = bfloat2float_lsx(p0a + 28);
                        _absmax31 = __lsx_vfmax_s(_absmax31, (__m128)__lsx_vand_v((__m128i)_p31, _abs_mask));
                        p0a += A_hstep * 8;
                    }
                    _absmax00 = __lsx_vfmax_s(_absmax00, _absmax01);
                    _absmax10 = __lsx_vfmax_s(_absmax10, _absmax11);
                    _absmax20 = __lsx_vfmax_s(_absmax20, _absmax21);
                    _absmax30 = __lsx_vfmax_s(_absmax30, _absmax31);

                    transpose4x4_ps(_absmax00, _absmax10, _absmax20, _absmax30);
                    __m128 _absmax = __lsx_vfmax_s(__lsx_vfmax_s(_absmax00, _absmax10), __lsx_vfmax_s(_absmax20, _absmax30));

                    const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                    __lsx_vst(__lsx_vfdiv_s(_absmax, _v127), pd, 0);
                    pd += 4;

                    const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                    _absmax = (__m128)__lsx_vbitsel_v((__m128i)_absmax, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax, _zero));
                    __m128 _scale = __lsx_vfdiv_s(_v127, _absmax);
                    __m128 _scale0 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 0);
                    __m128 _scale1 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 1);
                    __m128 _scale2 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 2);
                    __m128 _scale3 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 3);

                    int kk = 0;
                    for (; kk < max_kk0; kk += 8)
                    {
                        __m128 _p00 = bfloat2float_lsx(p0);
                        _p00 = __lsx_vfmul_s(_p00, _scale0);
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p00), 0);
                        __m128 _p01 = bfloat2float_lsx(p0 + 4);
                        _p01 = __lsx_vfmul_s(_p01, _scale0);
                        ((int*)pp)[4] = __lsx_vpickve2gr_w((__m128i)float2int8(_p01), 0);
                        __m128 _p10 = bfloat2float_lsx(p0 + 8);
                        _p10 = __lsx_vfmul_s(_p10, _scale1);
                        ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)float2int8(_p10), 0);
                        __m128 _p11 = bfloat2float_lsx(p0 + 12);
                        _p11 = __lsx_vfmul_s(_p11, _scale1);
                        ((int*)pp)[5] = __lsx_vpickve2gr_w((__m128i)float2int8(_p11), 0);
                        __m128 _p20 = bfloat2float_lsx(p0 + 16);
                        _p20 = __lsx_vfmul_s(_p20, _scale2);
                        ((int*)pp)[2] = __lsx_vpickve2gr_w((__m128i)float2int8(_p20), 0);
                        __m128 _p21 = bfloat2float_lsx(p0 + 20);
                        _p21 = __lsx_vfmul_s(_p21, _scale2);
                        ((int*)pp)[6] = __lsx_vpickve2gr_w((__m128i)float2int8(_p21), 0);
                        __m128 _p30 = bfloat2float_lsx(p0 + 24);
                        _p30 = __lsx_vfmul_s(_p30, _scale3);
                        ((int*)pp)[3] = __lsx_vpickve2gr_w((__m128i)float2int8(_p30), 0);
                        __m128 _p31 = bfloat2float_lsx(p0 + 28);
                        _p31 = __lsx_vfmul_s(_p31, _scale3);
                        ((int*)pp)[7] = __lsx_vpickve2gr_w((__m128i)float2int8(_p31), 0);
                        pp += 32;
                        p0 += A_hstep * 8;
                    }
                }
            }
            if (elempack == 4)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 4;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax00 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax10 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax20 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax30 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        __m128 _p00 = bfloat2float_lsx(p0a);
                        _absmax00 = __lsx_vfmax_s(_absmax00, (__m128)__lsx_vand_v((__m128i)_p00, _abs_mask));
                        __m128 _p10 = bfloat2float_lsx(p0a + 4);
                        _absmax10 = __lsx_vfmax_s(_absmax10, (__m128)__lsx_vand_v((__m128i)_p10, _abs_mask));
                        __m128 _p20 = bfloat2float_lsx(p0a + 8);
                        _absmax20 = __lsx_vfmax_s(_absmax20, (__m128)__lsx_vand_v((__m128i)_p20, _abs_mask));
                        __m128 _p30 = bfloat2float_lsx(p0a + 12);
                        _absmax30 = __lsx_vfmax_s(_absmax30, (__m128)__lsx_vand_v((__m128i)_p30, _abs_mask));
                        p0a += A_hstep * 4;
                    }
                    transpose4x4_ps(_absmax00, _absmax10, _absmax20, _absmax30);
                    __m128 _absmax = __lsx_vfmax_s(__lsx_vfmax_s(_absmax00, _absmax10), __lsx_vfmax_s(_absmax20, _absmax30));

                    const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                    __lsx_vst(__lsx_vfdiv_s(_absmax, _v127), pd, 0);
                    pd += 4;

                    const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                    _absmax = (__m128)__lsx_vbitsel_v((__m128i)_absmax, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax, _zero));
                    __m128 _scale = __lsx_vfdiv_s(_v127, _absmax);
                    __m128 _scale0 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 0);
                    __m128 _scale1 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 1);
                    __m128 _scale2 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 2);
                    __m128 _scale3 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 3);

                    int kk = 0;
                    for (; kk < max_kk0; kk += 4)
                    {
                        __m128 _p00 = bfloat2float_lsx(p0);
                        _p00 = __lsx_vfmul_s(_p00, _scale0);
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p00), 0);
                        __m128 _p10 = bfloat2float_lsx(p0 + 4);
                        _p10 = __lsx_vfmul_s(_p10, _scale1);
                        ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)float2int8(_p10), 0);
                        __m128 _p20 = bfloat2float_lsx(p0 + 8);
                        _p20 = __lsx_vfmul_s(_p20, _scale2);
                        ((int*)pp)[2] = __lsx_vpickve2gr_w((__m128i)float2int8(_p20), 0);
                        __m128 _p30 = bfloat2float_lsx(p0 + 12);
                        _p30 = __lsx_vfmul_s(_p30, _scale3);
                        ((int*)pp)[3] = __lsx_vpickve2gr_w((__m128i)float2int8(_p30), 0);
                        pp += 16;
                        p0 += A_hstep * 4;
                    }
                }
            }

            if (elempack == 1)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    int kk = 0;
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p = bfloat2float_lsx(p0a);
                        _absmax = __lsx_vfmax_s(_absmax, (__m128)__lsx_vand_v((__m128i)_p, _abs_mask));
                        p0a += A_hstep;
                    }

                    const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                    __lsx_vst(__lsx_vfdiv_s(_absmax, _v127), pd, 0);
                    pd += 4;

                    const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                    _absmax = (__m128)__lsx_vbitsel_v((__m128i)_absmax, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax, _zero));
                    __m128 _scale = __lsx_vfdiv_s(_v127, _absmax);
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        const unsigned short* p1 = p0 + A_hstep;
                        const unsigned short* p2 = p1 + A_hstep;
                        const unsigned short* p3 = p2 + A_hstep;
                        __m128 _p0 = bfloat2float_lsx(p0);
                        __m128 _p1 = bfloat2float_lsx(p1);
                        __m128 _p2 = bfloat2float_lsx(p2);
                        __m128 _p3 = bfloat2float_lsx(p3);
                        __m128i _q0 = float2int8(__lsx_vfmul_s(_p0, _scale));
                        __m128i _q1 = float2int8(__lsx_vfmul_s(_p1, _scale));
                        __m128i _q2 = float2int8(__lsx_vfmul_s(_p2, _scale));
                        __m128i _q3 = float2int8(__lsx_vfmul_s(_p3, _scale));
                        __m128i _q01 = __lsx_vilvl_b(_q1, _q0);
                        __m128i _q23 = __lsx_vilvl_b(_q3, _q2);
                        __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp, 0);
                        pp += 16;
                        p0 = p3 + A_hstep;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        ((int*)pp)[0] = __lsx_vpickve2gr_w(float2int8(__lsx_vfmul_s(bfloat2float_lsx(p0), _scale)), 0);
                        pp += 4;
                        p0 += A_hstep;
                    }
                }
            }
        }
#endif // __loongarch_sx
        for (; ii + 1 < max_ii; ii += 2)
        {
#if __loongarch_sx
            if (elempack == 8)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 8;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax00 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax01 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax10 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax11 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        __m128 _p00 = bfloat2float_lsx(p0a);
                        _absmax00 = __lsx_vfmax_s(_absmax00, (__m128)__lsx_vand_v((__m128i)_p00, _abs_mask));
                        __m128 _p01 = bfloat2float_lsx(p0a + 4);
                        _absmax01 = __lsx_vfmax_s(_absmax01, (__m128)__lsx_vand_v((__m128i)_p01, _abs_mask));
                        __m128 _p10 = bfloat2float_lsx(p0a + 8);
                        _absmax10 = __lsx_vfmax_s(_absmax10, (__m128)__lsx_vand_v((__m128i)_p10, _abs_mask));
                        __m128 _p11 = bfloat2float_lsx(p0a + 12);
                        _absmax11 = __lsx_vfmax_s(_absmax11, (__m128)__lsx_vand_v((__m128i)_p11, _abs_mask));
                        p0a += A_hstep * 8;
                    }
                    _absmax00 = __lsx_vfmax_s(_absmax00, _absmax01);
                    const float absmax0 = __lsx_reduce_fmax_s(_absmax00);
                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    pd[0] = absmax0 / 127.f;
                    _absmax10 = __lsx_vfmax_s(_absmax10, _absmax11);
                    const float absmax1 = __lsx_reduce_fmax_s(_absmax10);
                    const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                    pd[1] = absmax1 / 127.f;
                    pd += 2;

                    int kk = 0;
                    for (; kk < max_kk0; kk += 8)
                    {
                        __m128 _p00 = bfloat2float_lsx(p0);
                        _p00 = __lsx_vfmul_s(_p00, __lsx_vreplfr2vr_s(scale0));
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p00), 0);
                        __m128 _p01 = bfloat2float_lsx(p0 + 4);
                        _p01 = __lsx_vfmul_s(_p01, __lsx_vreplfr2vr_s(scale0));
                        ((int*)pp)[2] = __lsx_vpickve2gr_w((__m128i)float2int8(_p01), 0);
                        __m128 _p10 = bfloat2float_lsx(p0 + 8);
                        _p10 = __lsx_vfmul_s(_p10, __lsx_vreplfr2vr_s(scale1));
                        ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)float2int8(_p10), 0);
                        __m128 _p11 = bfloat2float_lsx(p0 + 12);
                        _p11 = __lsx_vfmul_s(_p11, __lsx_vreplfr2vr_s(scale1));
                        ((int*)pp)[3] = __lsx_vpickve2gr_w((__m128i)float2int8(_p11), 0);
                        pp += 16;
                        p0 += A_hstep * 8;
                    }
                }
            }
            if (elempack == 4)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 4;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax00 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax10 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        __m128 _p00 = bfloat2float_lsx(p0a);
                        _absmax00 = __lsx_vfmax_s(_absmax00, (__m128)__lsx_vand_v((__m128i)_p00, _abs_mask));
                        __m128 _p10 = bfloat2float_lsx(p0a + 4);
                        _absmax10 = __lsx_vfmax_s(_absmax10, (__m128)__lsx_vand_v((__m128i)_p10, _abs_mask));
                        p0a += A_hstep * 4;
                    }
                    const float absmax0 = __lsx_reduce_fmax_s(_absmax00);
                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    pd[0] = absmax0 / 127.f;
                    const float absmax1 = __lsx_reduce_fmax_s(_absmax10);
                    const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                    pd[1] = absmax1 / 127.f;
                    pd += 2;

                    int kk = 0;
                    for (; kk < max_kk0; kk += 4)
                    {
                        __m128 _p00 = bfloat2float_lsx(p0);
                        _p00 = __lsx_vfmul_s(_p00, __lsx_vreplfr2vr_s(scale0));
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p00), 0);
                        __m128 _p10 = bfloat2float_lsx(p0 + 4);
                        _p10 = __lsx_vfmul_s(_p10, __lsx_vreplfr2vr_s(scale1));
                        ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)float2int8(_p10), 0);
                        pp += 8;
                        p0 += A_hstep * 4;
                    }
                }
            }

            if (elempack == 1)
#endif // __loongarch_sx
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;
                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    float absmax0 = 0.f;
                    float absmax1 = 0.f;
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float v0 = bfloat16_to_float32(p0a[0]);
                        float v1 = bfloat16_to_float32(p0a[1]);
                        absmax0 = std::max(absmax0, fabsf(v0));
                        absmax1 = std::max(absmax1, fabsf(v1));
                        p0a += A_hstep;
                    }

                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    pd += 2;

                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float v00 = bfloat16_to_float32(p0[0]);
                        float v10 = bfloat16_to_float32(p0[1]);
                        float v01 = bfloat16_to_float32(p0[A_hstep]);
                        float v11 = bfloat16_to_float32(p0[A_hstep + 1]);
                        float v02 = bfloat16_to_float32(p0[A_hstep * 2]);
                        float v12 = bfloat16_to_float32(p0[A_hstep * 2 + 1]);
                        float v03 = bfloat16_to_float32(p0[A_hstep * 3]);
                        float v13 = bfloat16_to_float32(p0[A_hstep * 3 + 1]);
                        pp[0] = float2int8(v00 * scale0);
                        pp[1] = float2int8(v01 * scale0);
                        pp[2] = float2int8(v02 * scale0);
                        pp[3] = float2int8(v03 * scale0);
                        pp[4] = float2int8(v10 * scale1);
                        pp[5] = float2int8(v11 * scale1);
                        pp[6] = float2int8(v12 * scale1);
                        pp[7] = float2int8(v13 * scale1);
                        p0 += A_hstep * 4;
                        pp += 8;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float v0 = bfloat16_to_float32(p0[0]);
                        float v1 = bfloat16_to_float32(p0[1]);
                        pp[0] = float2int8(v0 * scale0);
                        pp[1] = float2int8(v1 * scale1);
                        pp += 2;
                        p0 += A_hstep;
                    }
                }
            }
        }
        for (; ii < max_ii; ii++)
        {
#if __loongarch_sx
            if (elempack == 8)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 8;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax00 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax01 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        __m128 _p00 = bfloat2float_lsx(p0a);
                        _absmax00 = __lsx_vfmax_s(_absmax00, (__m128)__lsx_vand_v((__m128i)_p00, _abs_mask));
                        __m128 _p01 = bfloat2float_lsx(p0a + 4);
                        _absmax01 = __lsx_vfmax_s(_absmax01, (__m128)__lsx_vand_v((__m128i)_p01, _abs_mask));
                        p0a += A_hstep * 8;
                    }
                    _absmax00 = __lsx_vfmax_s(_absmax00, _absmax01);
                    const float absmax0 = __lsx_reduce_fmax_s(_absmax00);
                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    pd[0] = absmax0 / 127.f;
                    pd += 1;

                    int kk = 0;
                    for (; kk < max_kk0; kk += 8)
                    {
                        __m128 _p00 = bfloat2float_lsx(p0);
                        _p00 = __lsx_vfmul_s(_p00, __lsx_vreplfr2vr_s(scale0));
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p00), 0);
                        __m128 _p01 = bfloat2float_lsx(p0 + 4);
                        _p01 = __lsx_vfmul_s(_p01, __lsx_vreplfr2vr_s(scale0));
                        ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)float2int8(_p01), 0);
                        pp += 8;
                        p0 += A_hstep * 8;
                    }
                }
            }
            if (elempack == 4)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 4;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax00 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        __m128 _p00 = bfloat2float_lsx(p0a);
                        _absmax00 = __lsx_vfmax_s(_absmax00, (__m128)__lsx_vand_v((__m128i)_p00, _abs_mask));
                        p0a += A_hstep * 4;
                    }
                    const float absmax0 = __lsx_reduce_fmax_s(_absmax00);
                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    pd[0] = absmax0 / 127.f;
                    pd += 1;

                    int kk = 0;
                    for (; kk < max_kk0; kk += 4)
                    {
                        __m128 _p00 = bfloat2float_lsx(p0);
                        _p00 = __lsx_vfmul_s(_p00, __lsx_vreplfr2vr_s(scale0));
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p00), 0);
                        pp += 4;
                        p0 += A_hstep * 4;
                    }
                }
            }

            if (elempack == 1)
#endif // __loongarch_sx
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    float absmax0 = 0.f;
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float v0 = bfloat16_to_float32(*p0a);
                        absmax0 = std::max(absmax0, fabsf(v0));
                        p0a += A_hstep;
                    }

                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    *pd++ = absmax0 / 127.f;

                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float v0 = bfloat16_to_float32(p0[0]);
                        float v1 = bfloat16_to_float32(p0[A_hstep]);
                        float v2 = bfloat16_to_float32(p0[A_hstep * 2]);
                        float v3 = bfloat16_to_float32(p0[A_hstep * 3]);
                        pp[0] = float2int8(v0 * scale0);
                        pp[1] = float2int8(v1 * scale0);
                        pp[2] = float2int8(v2 * scale0);
                        pp[3] = float2int8(v3 * scale0);
                        p0 += A_hstep * 4;
                        pp += 4;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float v0 = bfloat16_to_float32(*p0);
                        *pp++ = float2int8(v0 * scale0);
                        p0 += A_hstep;
                    }
                }
            }
        }
    }
    else
    {
        const float* input_scale_ptr = (const float*)input_scales + k;

        int ii = 0;
#if __loongarch_sx
        for (; ii + 7 < max_ii; ii += 8)
        {
            if (elempack == 8)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 8;
                const float* ps = input_scale_ptr;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax00 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax01 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax10 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax11 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax20 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax21 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax30 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax31 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax40 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax41 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax50 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax51 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax60 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax61 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax70 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax71 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        __m128 _s0 = (__m128)__lsx_vld(psa, 0);
                        __m128 _s1 = (__m128)__lsx_vld(psa + 4, 0);
                        __m128 _p00 = __lsx_vfmul_s(bfloat2float_lsx(p0a), _s0);
                        _absmax00 = __lsx_vfmax_s(_absmax00, (__m128)__lsx_vand_v((__m128i)_p00, _abs_mask));
                        __m128 _p01 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 4), _s1);
                        _absmax01 = __lsx_vfmax_s(_absmax01, (__m128)__lsx_vand_v((__m128i)_p01, _abs_mask));
                        __m128 _p10 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 8), _s0);
                        _absmax10 = __lsx_vfmax_s(_absmax10, (__m128)__lsx_vand_v((__m128i)_p10, _abs_mask));
                        __m128 _p11 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 12), _s1);
                        _absmax11 = __lsx_vfmax_s(_absmax11, (__m128)__lsx_vand_v((__m128i)_p11, _abs_mask));
                        __m128 _p20 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 16), _s0);
                        _absmax20 = __lsx_vfmax_s(_absmax20, (__m128)__lsx_vand_v((__m128i)_p20, _abs_mask));
                        __m128 _p21 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 20), _s1);
                        _absmax21 = __lsx_vfmax_s(_absmax21, (__m128)__lsx_vand_v((__m128i)_p21, _abs_mask));
                        __m128 _p30 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 24), _s0);
                        _absmax30 = __lsx_vfmax_s(_absmax30, (__m128)__lsx_vand_v((__m128i)_p30, _abs_mask));
                        __m128 _p31 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 28), _s1);
                        _absmax31 = __lsx_vfmax_s(_absmax31, (__m128)__lsx_vand_v((__m128i)_p31, _abs_mask));
                        __m128 _p40 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 32), _s0);
                        _absmax40 = __lsx_vfmax_s(_absmax40, (__m128)__lsx_vand_v((__m128i)_p40, _abs_mask));
                        __m128 _p41 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 36), _s1);
                        _absmax41 = __lsx_vfmax_s(_absmax41, (__m128)__lsx_vand_v((__m128i)_p41, _abs_mask));
                        __m128 _p50 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 40), _s0);
                        _absmax50 = __lsx_vfmax_s(_absmax50, (__m128)__lsx_vand_v((__m128i)_p50, _abs_mask));
                        __m128 _p51 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 44), _s1);
                        _absmax51 = __lsx_vfmax_s(_absmax51, (__m128)__lsx_vand_v((__m128i)_p51, _abs_mask));
                        __m128 _p60 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 48), _s0);
                        _absmax60 = __lsx_vfmax_s(_absmax60, (__m128)__lsx_vand_v((__m128i)_p60, _abs_mask));
                        __m128 _p61 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 52), _s1);
                        _absmax61 = __lsx_vfmax_s(_absmax61, (__m128)__lsx_vand_v((__m128i)_p61, _abs_mask));
                        __m128 _p70 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 56), _s0);
                        _absmax70 = __lsx_vfmax_s(_absmax70, (__m128)__lsx_vand_v((__m128i)_p70, _abs_mask));
                        __m128 _p71 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 60), _s1);
                        _absmax71 = __lsx_vfmax_s(_absmax71, (__m128)__lsx_vand_v((__m128i)_p71, _abs_mask));
                        p0a += A_hstep * 8;
                        psa += 8;
                    }
                    _absmax00 = __lsx_vfmax_s(_absmax00, _absmax01);
                    _absmax10 = __lsx_vfmax_s(_absmax10, _absmax11);
                    _absmax20 = __lsx_vfmax_s(_absmax20, _absmax21);
                    _absmax30 = __lsx_vfmax_s(_absmax30, _absmax31);
                    _absmax40 = __lsx_vfmax_s(_absmax40, _absmax41);
                    _absmax50 = __lsx_vfmax_s(_absmax50, _absmax51);
                    _absmax60 = __lsx_vfmax_s(_absmax60, _absmax61);
                    _absmax70 = __lsx_vfmax_s(_absmax70, _absmax71);

                    transpose4x4_ps(_absmax00, _absmax10, _absmax20, _absmax30);
                    transpose4x4_ps(_absmax40, _absmax50, _absmax60, _absmax70);
                    __m128 _absmax03 = __lsx_vfmax_s(__lsx_vfmax_s(_absmax00, _absmax10), __lsx_vfmax_s(_absmax20, _absmax30));
                    __m128 _absmax47 = __lsx_vfmax_s(__lsx_vfmax_s(_absmax40, _absmax50), __lsx_vfmax_s(_absmax60, _absmax70));

                    const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                    __lsx_vst(__lsx_vfdiv_s(_absmax03, _v127), pd, 0);
                    __lsx_vst(__lsx_vfdiv_s(_absmax47, _v127), pd + 4, 0);
                    pd += 8;

                    const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                    _absmax03 = (__m128)__lsx_vbitsel_v((__m128i)_absmax03, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax03, _zero));
                    _absmax47 = (__m128)__lsx_vbitsel_v((__m128i)_absmax47, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax47, _zero));
                    __m128 _scale03 = __lsx_vfdiv_s(_v127, _absmax03);
                    __m128 _scale47 = __lsx_vfdiv_s(_v127, _absmax47);
                    __m128 _scale0 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 0);
                    __m128 _scale1 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 1);
                    __m128 _scale2 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 2);
                    __m128 _scale3 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 3);
                    __m128 _scale4 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 0);
                    __m128 _scale5 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 1);
                    __m128 _scale6 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 2);
                    __m128 _scale7 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 3);

                    int kk = 0;
                    for (; kk < max_kk0; kk += 8)
                    {
                        __m128 _s0 = (__m128)__lsx_vld(ps, 0);
                        __m128 _s1 = (__m128)__lsx_vld(ps + 4, 0);
                        __m128 _p00 = __lsx_vfmul_s(bfloat2float_lsx(p0), _s0);
                        _p00 = __lsx_vfmul_s(_p00, _scale0);
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p00), 0);
                        __m128 _p01 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 4), _s1);
                        _p01 = __lsx_vfmul_s(_p01, _scale0);
                        ((int*)pp)[8] = __lsx_vpickve2gr_w((__m128i)float2int8(_p01), 0);
                        __m128 _p10 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 8), _s0);
                        _p10 = __lsx_vfmul_s(_p10, _scale1);
                        ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)float2int8(_p10), 0);
                        __m128 _p11 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 12), _s1);
                        _p11 = __lsx_vfmul_s(_p11, _scale1);
                        ((int*)pp)[9] = __lsx_vpickve2gr_w((__m128i)float2int8(_p11), 0);
                        __m128 _p20 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 16), _s0);
                        _p20 = __lsx_vfmul_s(_p20, _scale2);
                        ((int*)pp)[2] = __lsx_vpickve2gr_w((__m128i)float2int8(_p20), 0);
                        __m128 _p21 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 20), _s1);
                        _p21 = __lsx_vfmul_s(_p21, _scale2);
                        ((int*)pp)[10] = __lsx_vpickve2gr_w((__m128i)float2int8(_p21), 0);
                        __m128 _p30 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 24), _s0);
                        _p30 = __lsx_vfmul_s(_p30, _scale3);
                        ((int*)pp)[3] = __lsx_vpickve2gr_w((__m128i)float2int8(_p30), 0);
                        __m128 _p31 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 28), _s1);
                        _p31 = __lsx_vfmul_s(_p31, _scale3);
                        ((int*)pp)[11] = __lsx_vpickve2gr_w((__m128i)float2int8(_p31), 0);
                        __m128 _p40 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 32), _s0);
                        _p40 = __lsx_vfmul_s(_p40, _scale4);
                        ((int*)pp)[4] = __lsx_vpickve2gr_w((__m128i)float2int8(_p40), 0);
                        __m128 _p41 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 36), _s1);
                        _p41 = __lsx_vfmul_s(_p41, _scale4);
                        ((int*)pp)[12] = __lsx_vpickve2gr_w((__m128i)float2int8(_p41), 0);
                        __m128 _p50 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 40), _s0);
                        _p50 = __lsx_vfmul_s(_p50, _scale5);
                        ((int*)pp)[5] = __lsx_vpickve2gr_w((__m128i)float2int8(_p50), 0);
                        __m128 _p51 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 44), _s1);
                        _p51 = __lsx_vfmul_s(_p51, _scale5);
                        ((int*)pp)[13] = __lsx_vpickve2gr_w((__m128i)float2int8(_p51), 0);
                        __m128 _p60 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 48), _s0);
                        _p60 = __lsx_vfmul_s(_p60, _scale6);
                        ((int*)pp)[6] = __lsx_vpickve2gr_w((__m128i)float2int8(_p60), 0);
                        __m128 _p61 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 52), _s1);
                        _p61 = __lsx_vfmul_s(_p61, _scale6);
                        ((int*)pp)[14] = __lsx_vpickve2gr_w((__m128i)float2int8(_p61), 0);
                        __m128 _p70 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 56), _s0);
                        _p70 = __lsx_vfmul_s(_p70, _scale7);
                        ((int*)pp)[7] = __lsx_vpickve2gr_w((__m128i)float2int8(_p70), 0);
                        __m128 _p71 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 60), _s1);
                        _p71 = __lsx_vfmul_s(_p71, _scale7);
                        ((int*)pp)[15] = __lsx_vpickve2gr_w((__m128i)float2int8(_p71), 0);
                        pp += 64;
                        p0 += A_hstep * 8;
                        ps += 8;
                    }
                }
            }
            if (elempack == 4)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 4;
                const float* ps = input_scale_ptr;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax00 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax10 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax20 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax30 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax40 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax50 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax60 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax70 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        __m128 _s0 = (__m128)__lsx_vld(psa, 0);
                        __m128 _p00 = __lsx_vfmul_s(bfloat2float_lsx(p0a), _s0);
                        _absmax00 = __lsx_vfmax_s(_absmax00, (__m128)__lsx_vand_v((__m128i)_p00, _abs_mask));
                        __m128 _p10 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 4), _s0);
                        _absmax10 = __lsx_vfmax_s(_absmax10, (__m128)__lsx_vand_v((__m128i)_p10, _abs_mask));
                        __m128 _p20 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 8), _s0);
                        _absmax20 = __lsx_vfmax_s(_absmax20, (__m128)__lsx_vand_v((__m128i)_p20, _abs_mask));
                        __m128 _p30 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 12), _s0);
                        _absmax30 = __lsx_vfmax_s(_absmax30, (__m128)__lsx_vand_v((__m128i)_p30, _abs_mask));
                        __m128 _p40 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 16), _s0);
                        _absmax40 = __lsx_vfmax_s(_absmax40, (__m128)__lsx_vand_v((__m128i)_p40, _abs_mask));
                        __m128 _p50 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 20), _s0);
                        _absmax50 = __lsx_vfmax_s(_absmax50, (__m128)__lsx_vand_v((__m128i)_p50, _abs_mask));
                        __m128 _p60 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 24), _s0);
                        _absmax60 = __lsx_vfmax_s(_absmax60, (__m128)__lsx_vand_v((__m128i)_p60, _abs_mask));
                        __m128 _p70 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 28), _s0);
                        _absmax70 = __lsx_vfmax_s(_absmax70, (__m128)__lsx_vand_v((__m128i)_p70, _abs_mask));
                        p0a += A_hstep * 4;
                        psa += 4;
                    }
                    transpose4x4_ps(_absmax00, _absmax10, _absmax20, _absmax30);
                    transpose4x4_ps(_absmax40, _absmax50, _absmax60, _absmax70);
                    __m128 _absmax03 = __lsx_vfmax_s(__lsx_vfmax_s(_absmax00, _absmax10), __lsx_vfmax_s(_absmax20, _absmax30));
                    __m128 _absmax47 = __lsx_vfmax_s(__lsx_vfmax_s(_absmax40, _absmax50), __lsx_vfmax_s(_absmax60, _absmax70));

                    const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                    __lsx_vst(__lsx_vfdiv_s(_absmax03, _v127), pd, 0);
                    __lsx_vst(__lsx_vfdiv_s(_absmax47, _v127), pd + 4, 0);
                    pd += 8;

                    const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                    _absmax03 = (__m128)__lsx_vbitsel_v((__m128i)_absmax03, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax03, _zero));
                    _absmax47 = (__m128)__lsx_vbitsel_v((__m128i)_absmax47, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax47, _zero));
                    __m128 _scale03 = __lsx_vfdiv_s(_v127, _absmax03);
                    __m128 _scale47 = __lsx_vfdiv_s(_v127, _absmax47);
                    __m128 _scale0 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 0);
                    __m128 _scale1 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 1);
                    __m128 _scale2 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 2);
                    __m128 _scale3 = (__m128)__lsx_vreplvei_w((__m128i)_scale03, 3);
                    __m128 _scale4 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 0);
                    __m128 _scale5 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 1);
                    __m128 _scale6 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 2);
                    __m128 _scale7 = (__m128)__lsx_vreplvei_w((__m128i)_scale47, 3);

                    int kk = 0;
                    for (; kk < max_kk0; kk += 4)
                    {
                        __m128 _s0 = (__m128)__lsx_vld(ps, 0);
                        __m128 _p00 = __lsx_vfmul_s(bfloat2float_lsx(p0), _s0);
                        _p00 = __lsx_vfmul_s(_p00, _scale0);
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p00), 0);
                        __m128 _p10 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 4), _s0);
                        _p10 = __lsx_vfmul_s(_p10, _scale1);
                        ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)float2int8(_p10), 0);
                        __m128 _p20 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 8), _s0);
                        _p20 = __lsx_vfmul_s(_p20, _scale2);
                        ((int*)pp)[2] = __lsx_vpickve2gr_w((__m128i)float2int8(_p20), 0);
                        __m128 _p30 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 12), _s0);
                        _p30 = __lsx_vfmul_s(_p30, _scale3);
                        ((int*)pp)[3] = __lsx_vpickve2gr_w((__m128i)float2int8(_p30), 0);
                        __m128 _p40 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 16), _s0);
                        _p40 = __lsx_vfmul_s(_p40, _scale4);
                        ((int*)pp)[4] = __lsx_vpickve2gr_w((__m128i)float2int8(_p40), 0);
                        __m128 _p50 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 20), _s0);
                        _p50 = __lsx_vfmul_s(_p50, _scale5);
                        ((int*)pp)[5] = __lsx_vpickve2gr_w((__m128i)float2int8(_p50), 0);
                        __m128 _p60 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 24), _s0);
                        _p60 = __lsx_vfmul_s(_p60, _scale6);
                        ((int*)pp)[6] = __lsx_vpickve2gr_w((__m128i)float2int8(_p60), 0);
                        __m128 _p70 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 28), _s0);
                        _p70 = __lsx_vfmul_s(_p70, _scale7);
                        ((int*)pp)[7] = __lsx_vpickve2gr_w((__m128i)float2int8(_p70), 0);
                        pp += 32;
                        p0 += A_hstep * 4;
                        ps += 4;
                    }
                }
            }

            if (elempack == 1)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;

                const float* ps = input_scale_ptr;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax0 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax1 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    int kk = 0;
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p0 = bfloat2float_lsx(p0a);
                        __m128 _p1 = bfloat2float_lsx(p0a + 4);
                        __m128 _s = __lsx_vreplfr2vr_s(*psa++);
                        _p0 = (__m128)__lsx_vand_v((__m128i)_p0, _abs_mask);
                        _p0 = __lsx_vfmul_s(_p0, _s);
                        _p1 = (__m128)__lsx_vand_v((__m128i)_p1, _abs_mask);
                        _p1 = __lsx_vfmul_s(_p1, _s);
                        _absmax0 = __lsx_vfmax_s(_absmax0, _p0);
                        _absmax1 = __lsx_vfmax_s(_absmax1, _p1);
                        p0a += A_hstep;
                    }

                    const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                    __lsx_vst(__lsx_vfdiv_s(_absmax0, _v127), pd, 0);
                    __lsx_vst(__lsx_vfdiv_s(_absmax1, _v127), pd + 4, 0);
                    pd += 8;

                    const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax03 = (__m128)__lsx_vbitsel_v((__m128i)_absmax0, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax0, _zero));
                    __m128 _absmax47 = (__m128)__lsx_vbitsel_v((__m128i)_absmax1, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax1, _zero));
                    __m128 _scale03 = __lsx_vfdiv_s(_v127, _absmax03);
                    __m128 _scale47 = __lsx_vfdiv_s(_v127, _absmax47);
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        const unsigned short* p1 = p0 + A_hstep;
                        const unsigned short* p2 = p1 + A_hstep;
                        const unsigned short* p3 = p2 + A_hstep;
                        __m128 _p0 = bfloat2float_lsx(p0);
                        __m128 _p1 = bfloat2float_lsx(p1);
                        __m128 _p2 = bfloat2float_lsx(p2);
                        __m128 _p3 = bfloat2float_lsx(p3);
                        __m128 _p4 = bfloat2float_lsx(p0 + 4);
                        __m128 _p5 = bfloat2float_lsx(p1 + 4);
                        __m128 _p6 = bfloat2float_lsx(p2 + 4);
                        __m128 _p7 = bfloat2float_lsx(p3 + 4);
                        __m128i _q0 = float2int8(__lsx_vfmul_s(__lsx_vfmul_s(_p0, __lsx_vreplfr2vr_s(ps[0])), _scale03));
                        __m128i _q1 = float2int8(__lsx_vfmul_s(__lsx_vfmul_s(_p1, __lsx_vreplfr2vr_s(ps[1])), _scale03));
                        __m128i _q2 = float2int8(__lsx_vfmul_s(__lsx_vfmul_s(_p2, __lsx_vreplfr2vr_s(ps[2])), _scale03));
                        __m128i _q3 = float2int8(__lsx_vfmul_s(__lsx_vfmul_s(_p3, __lsx_vreplfr2vr_s(ps[3])), _scale03));
                        __m128i _q01 = __lsx_vilvl_b(_q1, _q0);
                        __m128i _q23 = __lsx_vilvl_b(_q3, _q2);
                        __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp, 0);
                        _q0 = float2int8(__lsx_vfmul_s(__lsx_vfmul_s(_p4, __lsx_vreplfr2vr_s(ps[0])), _scale47));
                        _q1 = float2int8(__lsx_vfmul_s(__lsx_vfmul_s(_p5, __lsx_vreplfr2vr_s(ps[1])), _scale47));
                        _q2 = float2int8(__lsx_vfmul_s(__lsx_vfmul_s(_p6, __lsx_vreplfr2vr_s(ps[2])), _scale47));
                        _q3 = float2int8(__lsx_vfmul_s(__lsx_vfmul_s(_p7, __lsx_vreplfr2vr_s(ps[3])), _scale47));
                        _q01 = __lsx_vilvl_b(_q1, _q0);
                        _q23 = __lsx_vilvl_b(_q3, _q2);
                        __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp + 16, 0);
                        pp += 32;
                        p0 = p3 + A_hstep;
                        ps += 4;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        const float s = *ps++;
                        __m128 _p0 = __lsx_vfmul_s(bfloat2float_lsx(p0), __lsx_vreplfr2vr_s(s));
                        __m128 _p1 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 4), __lsx_vreplfr2vr_s(s));
                        __m128i _q0 = float2int8(__lsx_vfmul_s(_p0, _scale03));
                        __m128i _q1 = float2int8(__lsx_vfmul_s(_p1, _scale47));
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)_q0, 0);
                        ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)_q1, 0);
                        pp += 8;
                        p0 += A_hstep;
                    }
                }
            }
        }
        for (; ii + 3 < max_ii; ii += 4)
        {
            if (elempack == 8)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 8;
                const float* ps = input_scale_ptr;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax00 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax01 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax10 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax11 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax20 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax21 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax30 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax31 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        __m128 _s0 = (__m128)__lsx_vld(psa, 0);
                        __m128 _s1 = (__m128)__lsx_vld(psa + 4, 0);
                        __m128 _p00 = __lsx_vfmul_s(bfloat2float_lsx(p0a), _s0);
                        _absmax00 = __lsx_vfmax_s(_absmax00, (__m128)__lsx_vand_v((__m128i)_p00, _abs_mask));
                        __m128 _p01 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 4), _s1);
                        _absmax01 = __lsx_vfmax_s(_absmax01, (__m128)__lsx_vand_v((__m128i)_p01, _abs_mask));
                        __m128 _p10 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 8), _s0);
                        _absmax10 = __lsx_vfmax_s(_absmax10, (__m128)__lsx_vand_v((__m128i)_p10, _abs_mask));
                        __m128 _p11 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 12), _s1);
                        _absmax11 = __lsx_vfmax_s(_absmax11, (__m128)__lsx_vand_v((__m128i)_p11, _abs_mask));
                        __m128 _p20 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 16), _s0);
                        _absmax20 = __lsx_vfmax_s(_absmax20, (__m128)__lsx_vand_v((__m128i)_p20, _abs_mask));
                        __m128 _p21 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 20), _s1);
                        _absmax21 = __lsx_vfmax_s(_absmax21, (__m128)__lsx_vand_v((__m128i)_p21, _abs_mask));
                        __m128 _p30 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 24), _s0);
                        _absmax30 = __lsx_vfmax_s(_absmax30, (__m128)__lsx_vand_v((__m128i)_p30, _abs_mask));
                        __m128 _p31 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 28), _s1);
                        _absmax31 = __lsx_vfmax_s(_absmax31, (__m128)__lsx_vand_v((__m128i)_p31, _abs_mask));
                        p0a += A_hstep * 8;
                        psa += 8;
                    }
                    _absmax00 = __lsx_vfmax_s(_absmax00, _absmax01);
                    _absmax10 = __lsx_vfmax_s(_absmax10, _absmax11);
                    _absmax20 = __lsx_vfmax_s(_absmax20, _absmax21);
                    _absmax30 = __lsx_vfmax_s(_absmax30, _absmax31);

                    transpose4x4_ps(_absmax00, _absmax10, _absmax20, _absmax30);
                    __m128 _absmax = __lsx_vfmax_s(__lsx_vfmax_s(_absmax00, _absmax10), __lsx_vfmax_s(_absmax20, _absmax30));

                    const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                    __lsx_vst(__lsx_vfdiv_s(_absmax, _v127), pd, 0);
                    pd += 4;

                    const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                    _absmax = (__m128)__lsx_vbitsel_v((__m128i)_absmax, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax, _zero));
                    __m128 _scale = __lsx_vfdiv_s(_v127, _absmax);
                    __m128 _scale0 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 0);
                    __m128 _scale1 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 1);
                    __m128 _scale2 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 2);
                    __m128 _scale3 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 3);

                    int kk = 0;
                    for (; kk < max_kk0; kk += 8)
                    {
                        __m128 _s0 = (__m128)__lsx_vld(ps, 0);
                        __m128 _s1 = (__m128)__lsx_vld(ps + 4, 0);
                        __m128 _p00 = __lsx_vfmul_s(bfloat2float_lsx(p0), _s0);
                        _p00 = __lsx_vfmul_s(_p00, _scale0);
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p00), 0);
                        __m128 _p01 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 4), _s1);
                        _p01 = __lsx_vfmul_s(_p01, _scale0);
                        ((int*)pp)[4] = __lsx_vpickve2gr_w((__m128i)float2int8(_p01), 0);
                        __m128 _p10 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 8), _s0);
                        _p10 = __lsx_vfmul_s(_p10, _scale1);
                        ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)float2int8(_p10), 0);
                        __m128 _p11 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 12), _s1);
                        _p11 = __lsx_vfmul_s(_p11, _scale1);
                        ((int*)pp)[5] = __lsx_vpickve2gr_w((__m128i)float2int8(_p11), 0);
                        __m128 _p20 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 16), _s0);
                        _p20 = __lsx_vfmul_s(_p20, _scale2);
                        ((int*)pp)[2] = __lsx_vpickve2gr_w((__m128i)float2int8(_p20), 0);
                        __m128 _p21 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 20), _s1);
                        _p21 = __lsx_vfmul_s(_p21, _scale2);
                        ((int*)pp)[6] = __lsx_vpickve2gr_w((__m128i)float2int8(_p21), 0);
                        __m128 _p30 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 24), _s0);
                        _p30 = __lsx_vfmul_s(_p30, _scale3);
                        ((int*)pp)[3] = __lsx_vpickve2gr_w((__m128i)float2int8(_p30), 0);
                        __m128 _p31 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 28), _s1);
                        _p31 = __lsx_vfmul_s(_p31, _scale3);
                        ((int*)pp)[7] = __lsx_vpickve2gr_w((__m128i)float2int8(_p31), 0);
                        pp += 32;
                        p0 += A_hstep * 8;
                        ps += 8;
                    }
                }
            }
            if (elempack == 4)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 4;
                const float* ps = input_scale_ptr;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax00 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax10 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax20 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax30 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        __m128 _s0 = (__m128)__lsx_vld(psa, 0);
                        __m128 _p00 = __lsx_vfmul_s(bfloat2float_lsx(p0a), _s0);
                        _absmax00 = __lsx_vfmax_s(_absmax00, (__m128)__lsx_vand_v((__m128i)_p00, _abs_mask));
                        __m128 _p10 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 4), _s0);
                        _absmax10 = __lsx_vfmax_s(_absmax10, (__m128)__lsx_vand_v((__m128i)_p10, _abs_mask));
                        __m128 _p20 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 8), _s0);
                        _absmax20 = __lsx_vfmax_s(_absmax20, (__m128)__lsx_vand_v((__m128i)_p20, _abs_mask));
                        __m128 _p30 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 12), _s0);
                        _absmax30 = __lsx_vfmax_s(_absmax30, (__m128)__lsx_vand_v((__m128i)_p30, _abs_mask));
                        p0a += A_hstep * 4;
                        psa += 4;
                    }
                    transpose4x4_ps(_absmax00, _absmax10, _absmax20, _absmax30);
                    __m128 _absmax = __lsx_vfmax_s(__lsx_vfmax_s(_absmax00, _absmax10), __lsx_vfmax_s(_absmax20, _absmax30));

                    const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                    __lsx_vst(__lsx_vfdiv_s(_absmax, _v127), pd, 0);
                    pd += 4;

                    const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                    _absmax = (__m128)__lsx_vbitsel_v((__m128i)_absmax, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax, _zero));
                    __m128 _scale = __lsx_vfdiv_s(_v127, _absmax);
                    __m128 _scale0 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 0);
                    __m128 _scale1 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 1);
                    __m128 _scale2 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 2);
                    __m128 _scale3 = (__m128)__lsx_vreplvei_w((__m128i)_scale, 3);

                    int kk = 0;
                    for (; kk < max_kk0; kk += 4)
                    {
                        __m128 _s0 = (__m128)__lsx_vld(ps, 0);
                        __m128 _p00 = __lsx_vfmul_s(bfloat2float_lsx(p0), _s0);
                        _p00 = __lsx_vfmul_s(_p00, _scale0);
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p00), 0);
                        __m128 _p10 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 4), _s0);
                        _p10 = __lsx_vfmul_s(_p10, _scale1);
                        ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)float2int8(_p10), 0);
                        __m128 _p20 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 8), _s0);
                        _p20 = __lsx_vfmul_s(_p20, _scale2);
                        ((int*)pp)[2] = __lsx_vpickve2gr_w((__m128i)float2int8(_p20), 0);
                        __m128 _p30 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 12), _s0);
                        _p30 = __lsx_vfmul_s(_p30, _scale3);
                        ((int*)pp)[3] = __lsx_vpickve2gr_w((__m128i)float2int8(_p30), 0);
                        pp += 16;
                        p0 += A_hstep * 4;
                        ps += 4;
                    }
                }
            }

            if (elempack == 1)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;

                const float* ps = input_scale_ptr;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    int kk = 0;
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p = bfloat2float_lsx(p0a);
                        _p = (__m128)__lsx_vand_v((__m128i)_p, _abs_mask);
                        _p = __lsx_vfmul_s(_p, __lsx_vreplfr2vr_s(*psa++));
                        _absmax = __lsx_vfmax_s(_absmax, _p);
                        p0a += A_hstep;
                    }

                    const __m128 _v127 = __lsx_vreplfr2vr_s(127.f);
                    __lsx_vst(__lsx_vfdiv_s(_absmax, _v127), pd, 0);
                    pd += 4;

                    const __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                    _absmax = (__m128)__lsx_vbitsel_v((__m128i)_absmax, (__m128i)_v127, __lsx_vfcmp_ceq_s(_absmax, _zero));
                    __m128 _scale = __lsx_vfdiv_s(_v127, _absmax);
                    kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        const unsigned short* p1 = p0 + A_hstep;
                        const unsigned short* p2 = p1 + A_hstep;
                        const unsigned short* p3 = p2 + A_hstep;
                        __m128 _p0 = bfloat2float_lsx(p0);
                        __m128 _p1 = bfloat2float_lsx(p1);
                        __m128 _p2 = bfloat2float_lsx(p2);
                        __m128 _p3 = bfloat2float_lsx(p3);
                        __m128i _q0 = float2int8(__lsx_vfmul_s(__lsx_vfmul_s(_p0, __lsx_vreplfr2vr_s(ps[0])), _scale));
                        __m128i _q1 = float2int8(__lsx_vfmul_s(__lsx_vfmul_s(_p1, __lsx_vreplfr2vr_s(ps[1])), _scale));
                        __m128i _q2 = float2int8(__lsx_vfmul_s(__lsx_vfmul_s(_p2, __lsx_vreplfr2vr_s(ps[2])), _scale));
                        __m128i _q3 = float2int8(__lsx_vfmul_s(__lsx_vfmul_s(_p3, __lsx_vreplfr2vr_s(ps[3])), _scale));
                        __m128i _q01 = __lsx_vilvl_b(_q1, _q0);
                        __m128i _q23 = __lsx_vilvl_b(_q3, _q2);
                        __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp, 0);
                        pp += 16;
                        p0 = p3 + A_hstep;
                        ps += 4;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m128 _p = __lsx_vfmul_s(bfloat2float_lsx(p0), __lsx_vreplfr2vr_s(*ps++));
                        ((int*)pp)[0] = __lsx_vpickve2gr_w(float2int8(__lsx_vfmul_s(_p, _scale)), 0);
                        pp += 4;
                        p0 += A_hstep;
                    }
                }
            }
        }
#endif // __loongarch_sx
        for (; ii + 1 < max_ii; ii += 2)
        {
#if __loongarch_sx
            if (elempack == 8)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 8;
                const float* ps = input_scale_ptr;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax00 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax01 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax10 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax11 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        __m128 _s0 = (__m128)__lsx_vld(psa, 0);
                        __m128 _s1 = (__m128)__lsx_vld(psa + 4, 0);
                        __m128 _p00 = __lsx_vfmul_s(bfloat2float_lsx(p0a), _s0);
                        _absmax00 = __lsx_vfmax_s(_absmax00, (__m128)__lsx_vand_v((__m128i)_p00, _abs_mask));
                        __m128 _p01 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 4), _s1);
                        _absmax01 = __lsx_vfmax_s(_absmax01, (__m128)__lsx_vand_v((__m128i)_p01, _abs_mask));
                        __m128 _p10 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 8), _s0);
                        _absmax10 = __lsx_vfmax_s(_absmax10, (__m128)__lsx_vand_v((__m128i)_p10, _abs_mask));
                        __m128 _p11 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 12), _s1);
                        _absmax11 = __lsx_vfmax_s(_absmax11, (__m128)__lsx_vand_v((__m128i)_p11, _abs_mask));
                        p0a += A_hstep * 8;
                        psa += 8;
                    }
                    _absmax00 = __lsx_vfmax_s(_absmax00, _absmax01);
                    const float absmax0 = __lsx_reduce_fmax_s(_absmax00);
                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    pd[0] = absmax0 / 127.f;
                    _absmax10 = __lsx_vfmax_s(_absmax10, _absmax11);
                    const float absmax1 = __lsx_reduce_fmax_s(_absmax10);
                    const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                    pd[1] = absmax1 / 127.f;
                    pd += 2;

                    int kk = 0;
                    for (; kk < max_kk0; kk += 8)
                    {
                        __m128 _s0 = (__m128)__lsx_vld(ps, 0);
                        __m128 _s1 = (__m128)__lsx_vld(ps + 4, 0);
                        __m128 _p00 = __lsx_vfmul_s(bfloat2float_lsx(p0), _s0);
                        _p00 = __lsx_vfmul_s(_p00, __lsx_vreplfr2vr_s(scale0));
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p00), 0);
                        __m128 _p01 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 4), _s1);
                        _p01 = __lsx_vfmul_s(_p01, __lsx_vreplfr2vr_s(scale0));
                        ((int*)pp)[2] = __lsx_vpickve2gr_w((__m128i)float2int8(_p01), 0);
                        __m128 _p10 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 8), _s0);
                        _p10 = __lsx_vfmul_s(_p10, __lsx_vreplfr2vr_s(scale1));
                        ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)float2int8(_p10), 0);
                        __m128 _p11 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 12), _s1);
                        _p11 = __lsx_vfmul_s(_p11, __lsx_vreplfr2vr_s(scale1));
                        ((int*)pp)[3] = __lsx_vpickve2gr_w((__m128i)float2int8(_p11), 0);
                        pp += 16;
                        p0 += A_hstep * 8;
                        ps += 8;
                    }
                }
            }
            if (elempack == 4)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 4;
                const float* ps = input_scale_ptr;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax00 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax10 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        __m128 _s0 = (__m128)__lsx_vld(psa, 0);
                        __m128 _p00 = __lsx_vfmul_s(bfloat2float_lsx(p0a), _s0);
                        _absmax00 = __lsx_vfmax_s(_absmax00, (__m128)__lsx_vand_v((__m128i)_p00, _abs_mask));
                        __m128 _p10 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 4), _s0);
                        _absmax10 = __lsx_vfmax_s(_absmax10, (__m128)__lsx_vand_v((__m128i)_p10, _abs_mask));
                        p0a += A_hstep * 4;
                        psa += 4;
                    }
                    const float absmax0 = __lsx_reduce_fmax_s(_absmax00);
                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    pd[0] = absmax0 / 127.f;
                    const float absmax1 = __lsx_reduce_fmax_s(_absmax10);
                    const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                    pd[1] = absmax1 / 127.f;
                    pd += 2;

                    int kk = 0;
                    for (; kk < max_kk0; kk += 4)
                    {
                        __m128 _s0 = (__m128)__lsx_vld(ps, 0);
                        __m128 _p00 = __lsx_vfmul_s(bfloat2float_lsx(p0), _s0);
                        _p00 = __lsx_vfmul_s(_p00, __lsx_vreplfr2vr_s(scale0));
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p00), 0);
                        __m128 _p10 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 4), _s0);
                        _p10 = __lsx_vfmul_s(_p10, __lsx_vreplfr2vr_s(scale1));
                        ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)float2int8(_p10), 0);
                        pp += 8;
                        p0 += A_hstep * 4;
                        ps += 4;
                    }
                }
            }

            if (elempack == 1)
#endif // __loongarch_sx
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;
                const float* ps = input_scale_ptr;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    float absmax0 = 0.f;
                    float absmax1 = 0.f;
                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float v0 = bfloat16_to_float32(p0a[0]);
                        float v1 = bfloat16_to_float32(p0a[1]);
                        const float s = *psa++;

                        absmax0 = std::max(absmax0, fabsf(v0) * s);
                        absmax1 = std::max(absmax1, fabsf(v1) * s);
                        p0a += A_hstep;
                    }

                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    const float scale1 = absmax1 == 0.f ? 1.f : 127.f / absmax1;
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    pd += 2;

                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float v00 = bfloat16_to_float32(p0[0]);
                        float v10 = bfloat16_to_float32(p0[1]);
                        float v01 = bfloat16_to_float32(p0[A_hstep]);
                        float v11 = bfloat16_to_float32(p0[A_hstep + 1]);
                        float v02 = bfloat16_to_float32(p0[A_hstep * 2]);
                        float v12 = bfloat16_to_float32(p0[A_hstep * 2 + 1]);
                        float v03 = bfloat16_to_float32(p0[A_hstep * 3]);
                        float v13 = bfloat16_to_float32(p0[A_hstep * 3 + 1]);
                        v00 *= ps[0];
                        v10 *= ps[0];
                        v01 *= ps[1];
                        v11 *= ps[1];
                        v02 *= ps[2];
                        v12 *= ps[2];
                        v03 *= ps[3];
                        v13 *= ps[3];
                        ps += 4;
                        pp[0] = float2int8(v00 * scale0);
                        pp[1] = float2int8(v01 * scale0);
                        pp[2] = float2int8(v02 * scale0);
                        pp[3] = float2int8(v03 * scale0);
                        pp[4] = float2int8(v10 * scale1);
                        pp[5] = float2int8(v11 * scale1);
                        pp[6] = float2int8(v12 * scale1);
                        pp[7] = float2int8(v13 * scale1);
                        p0 += A_hstep * 4;
                        pp += 8;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float v0 = bfloat16_to_float32(p0[0]);
                        float v1 = bfloat16_to_float32(p0[1]);
                        const float s = *ps++;
                        v0 *= s;
                        v1 *= s;
                        pp[0] = float2int8(v0 * scale0);
                        pp[1] = float2int8(v1 * scale1);
                        pp += 2;
                        p0 += A_hstep;
                    }
                }
            }
        }
        for (; ii < max_ii; ii++)
        {
#if __loongarch_sx
            if (elempack == 8)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 8;
                const float* ps = input_scale_ptr;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax00 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _absmax01 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        __m128 _s0 = (__m128)__lsx_vld(psa, 0);
                        __m128 _s1 = (__m128)__lsx_vld(psa + 4, 0);
                        __m128 _p00 = __lsx_vfmul_s(bfloat2float_lsx(p0a), _s0);
                        _absmax00 = __lsx_vfmax_s(_absmax00, (__m128)__lsx_vand_v((__m128i)_p00, _abs_mask));
                        __m128 _p01 = __lsx_vfmul_s(bfloat2float_lsx(p0a + 4), _s1);
                        _absmax01 = __lsx_vfmax_s(_absmax01, (__m128)__lsx_vand_v((__m128i)_p01, _abs_mask));
                        p0a += A_hstep * 8;
                        psa += 8;
                    }
                    _absmax00 = __lsx_vfmax_s(_absmax00, _absmax01);
                    const float absmax0 = __lsx_reduce_fmax_s(_absmax00);
                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    pd[0] = absmax0 / 127.f;
                    pd += 1;

                    int kk = 0;
                    for (; kk < max_kk0; kk += 8)
                    {
                        __m128 _s0 = (__m128)__lsx_vld(ps, 0);
                        __m128 _s1 = (__m128)__lsx_vld(ps + 4, 0);
                        __m128 _p00 = __lsx_vfmul_s(bfloat2float_lsx(p0), _s0);
                        _p00 = __lsx_vfmul_s(_p00, __lsx_vreplfr2vr_s(scale0));
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p00), 0);
                        __m128 _p01 = __lsx_vfmul_s(bfloat2float_lsx(p0 + 4), _s1);
                        _p01 = __lsx_vfmul_s(_p01, __lsx_vreplfr2vr_s(scale0));
                        ((int*)pp)[1] = __lsx_vpickve2gr_w((__m128i)float2int8(_p01), 0);
                        pp += 8;
                        p0 += A_hstep * 8;
                        ps += 8;
                    }
                }
            }
            if (elempack == 4)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * 4;
                const float* ps = input_scale_ptr;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    const __m128i _abs_mask = (__m128i)__lsx_vreplgr2vr_w(0x7fffffff);
                    __m128 _absmax00 = (__m128)__lsx_vreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        __m128 _s0 = (__m128)__lsx_vld(psa, 0);
                        __m128 _p00 = __lsx_vfmul_s(bfloat2float_lsx(p0a), _s0);
                        _absmax00 = __lsx_vfmax_s(_absmax00, (__m128)__lsx_vand_v((__m128i)_p00, _abs_mask));
                        p0a += A_hstep * 4;
                        psa += 4;
                    }
                    const float absmax0 = __lsx_reduce_fmax_s(_absmax00);
                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    pd[0] = absmax0 / 127.f;
                    pd += 1;

                    int kk = 0;
                    for (; kk < max_kk0; kk += 4)
                    {
                        __m128 _s0 = (__m128)__lsx_vld(ps, 0);
                        __m128 _p00 = __lsx_vfmul_s(bfloat2float_lsx(p0), _s0);
                        _p00 = __lsx_vfmul_s(_p00, __lsx_vreplfr2vr_s(scale0));
                        ((int*)pp)[0] = __lsx_vpickve2gr_w((__m128i)float2int8(_p00), 0);
                        pp += 4;
                        p0 += A_hstep * 4;
                        ps += 4;
                    }
                }
            }

            if (elempack == 1)
#endif // __loongarch_sx
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + i + ii;

                const float* ps = input_scale_ptr;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                    float absmax0 = 0.f;
                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float v0 = bfloat16_to_float32(*p0a);
                        const float s = *psa++;
                        absmax0 = std::max(absmax0, fabsf(v0) * s);
                        p0a += A_hstep;
                    }

                    const float scale0 = absmax0 == 0.f ? 1.f : 127.f / absmax0;
                    *pd++ = absmax0 / 127.f;

                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float v0 = bfloat16_to_float32(p0[0]);
                        float v1 = bfloat16_to_float32(p0[A_hstep]);
                        float v2 = bfloat16_to_float32(p0[A_hstep * 2]);
                        float v3 = bfloat16_to_float32(p0[A_hstep * 3]);
                        v0 *= ps[0];
                        v1 *= ps[1];
                        v2 *= ps[2];
                        v3 *= ps[3];
                        ps += 4;
                        pp[0] = float2int8(v0 * scale0);
                        pp[1] = float2int8(v1 * scale0);
                        pp[2] = float2int8(v2 * scale0);
                        pp[3] = float2int8(v3 * scale0);
                        p0 += A_hstep * 4;
                        pp += 4;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float v0 = bfloat16_to_float32(*p0);
                        v0 *= *ps++;
                        *pp++ = float2int8(v0 * scale0);
                        p0 += A_hstep;
                    }
                }
            }
        }
    }
}

static void unpack_output_tile_wq_int8_bf16s(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta, int output_transpose)
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
        unsigned short* p0;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + (size_t)j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (size_t)(i + ii) * out_hstep + j * out_elempack;
        }

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
                        _c0 = (__m256)__lasx_xvld(pC, 0);
                        _c1 = (__m256)__lasx_xvld(pC + 8, 0);
                        _c2 = (__m256)__lasx_xvld(pC + 16, 0);
                        _c3 = (__m256)__lasx_xvld(pC + 24, 0);
                        _c4 = (__m256)__lasx_xvld(pC + 32, 0);
                        _c5 = (__m256)__lasx_xvld(pC + 40, 0);
                        _c6 = (__m256)__lasx_xvld(pC + 48, 0);
                        _c7 = (__m256)__lasx_xvld(pC + 56, 0);
                        transpose8x8_ps(_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7);
                    }
                    else if (c_elempack == 4)
                    {
                        const float* pC1 = pC + c_hstep * 4;
                        _c0 = __lasx_concat_128_s((__m128)__lsx_vld(pC, 0), (__m128)__lsx_vld(pC1, 0));
                        _c1 = __lasx_concat_128_s((__m128)__lsx_vld(pC + 4, 0), (__m128)__lsx_vld(pC1 + 4, 0));
                        _c2 = __lasx_concat_128_s((__m128)__lsx_vld(pC + 8, 0), (__m128)__lsx_vld(pC1 + 8, 0));
                        _c3 = __lasx_concat_128_s((__m128)__lsx_vld(pC + 12, 0), (__m128)__lsx_vld(pC1 + 12, 0));
                        _c4 = __lasx_concat_128_s((__m128)__lsx_vld(pC + 16, 0), (__m128)__lsx_vld(pC1 + 16, 0));
                        _c5 = __lasx_concat_128_s((__m128)__lsx_vld(pC + 20, 0), (__m128)__lsx_vld(pC1 + 20, 0));
                        _c6 = __lasx_concat_128_s((__m128)__lsx_vld(pC + 24, 0), (__m128)__lsx_vld(pC1 + 24, 0));
                        _c7 = __lasx_concat_128_s((__m128)__lsx_vld(pC + 28, 0), (__m128)__lsx_vld(pC1 + 28, 0));
                        transpose8x8_ps(_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7);
                    }
                    else
                    {
                        _c0 = (__m256)__lasx_xvld(pC, 0);
                        _c1 = (__m256)__lasx_xvld(pC + c_hstep, 0);
                        _c2 = (__m256)__lasx_xvld(pC + c_hstep * 2, 0);
                        _c3 = (__m256)__lasx_xvld(pC + c_hstep * 3, 0);
                        _c4 = (__m256)__lasx_xvld(pC + c_hstep * 4, 0);
                        _c5 = (__m256)__lasx_xvld(pC + c_hstep * 5, 0);
                        _c6 = (__m256)__lasx_xvld(pC + c_hstep * 6, 0);
                        _c7 = (__m256)__lasx_xvld(pC + c_hstep * 7, 0);
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
                    __m256 _c = (__m256)__lasx_xvld(pC, 0);
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

            __m128i _q0 = float2bfloat_lasx(_f0);
            __m128i _q1 = float2bfloat_lasx(_f1);
            __m128i _q2 = float2bfloat_lasx(_f2);
            __m128i _q3 = float2bfloat_lasx(_f3);
            __m128i _q4 = float2bfloat_lasx(_f4);
            __m128i _q5 = float2bfloat_lasx(_f5);
            __m128i _q6 = float2bfloat_lasx(_f6);
            __m128i _q7 = float2bfloat_lasx(_f7);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    __lsx_vst(_q0, p0, 0);
                    __lsx_vst(_q1, p0 + 8, 0);
                    __lsx_vst(_q2, p0 + 16, 0);
                    __lsx_vst(_q3, p0 + 24, 0);
                    __lsx_vst(_q4, p0 + 32, 0);
                    __lsx_vst(_q5, p0 + 40, 0);
                    __lsx_vst(_q6, p0 + 48, 0);
                    __lsx_vst(_q7, p0 + 56, 0);
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
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
                    transpose8x8_epi16(_q0, _q1, _q2, _q3, _q4, _q5, _q6, _q7);
                    __lsx_vst(_q0, p0, 0);
                    __lsx_vst(_q1, p0 + out_hstep, 0);
                    __lsx_vst(_q2, p0 + out_hstep * 2, 0);
                    __lsx_vst(_q3, p0 + out_hstep * 3, 0);
                    __lsx_vst(_q4, p0 + out_hstep * 4, 0);
                    __lsx_vst(_q5, p0 + out_hstep * 5, 0);
                    __lsx_vst(_q6, p0 + out_hstep * 6, 0);
                    __lsx_vst(_q7, p0 + out_hstep * 7, 0);
                }
                p0 += out_hstep * 8;
            }
            else
            {
                if (out_elempack == 8)
                {
                    transpose8x8_epi16(_q0, _q1, _q2, _q3, _q4, _q5, _q6, _q7);
                    __lsx_vst(_q0, p0, 0);
                    __lsx_vst(_q1, p0 + 8, 0);
                    __lsx_vst(_q2, p0 + 16, 0);
                    __lsx_vst(_q3, p0 + 24, 0);
                    __lsx_vst(_q4, p0 + 32, 0);
                    __lsx_vst(_q5, p0 + 40, 0);
                    __lsx_vst(_q6, p0 + 48, 0);
                    __lsx_vst(_q7, p0 + 56, 0);
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    transpose8x4_epi16(_q0, _q1, _q2, _q3);
                    transpose8x4_epi16(_q4, _q5, _q6, _q7);
                    __lsx_vst(_q0, p0, 0);
                    __lsx_vst(_q1, p0 + 8, 0);
                    __lsx_vst(_q2, p0 + 16, 0);
                    __lsx_vst(_q3, p0 + 24, 0);
                    __lsx_vst(_q4, p1, 0);
                    __lsx_vst(_q5, p1 + 8, 0);
                    __lsx_vst(_q6, p1 + 16, 0);
                    __lsx_vst(_q7, p1 + 24, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vst(_q0, p0, 0);
                    __lsx_vst(_q1, p0 + out_hstep, 0);
                    __lsx_vst(_q2, p0 + out_hstep * 2, 0);
                    __lsx_vst(_q3, p0 + out_hstep * 3, 0);
                    __lsx_vst(_q4, p0 + out_hstep * 4, 0);
                    __lsx_vst(_q5, p0 + out_hstep * 5, 0);
                    __lsx_vst(_q6, p0 + out_hstep * 6, 0);
                    __lsx_vst(_q7, p0 + out_hstep * 7, 0);
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
#if __loongarch_asx
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
                    else
#endif // __loongarch_asx
                        if (c_elempack == 4)
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

            __m128i _q0 = float2bfloat_lsx(_f0);
            __m128i _q1 = float2bfloat_lsx(_f1);
            __m128i _q2 = float2bfloat_lsx(_f2);
            __m128i _q3 = float2bfloat_lsx(_f3);
            __m128i _q4 = float2bfloat_lsx(_f4);
            __m128i _q5 = float2bfloat_lsx(_f5);
            __m128i _q6 = float2bfloat_lsx(_f6);
            __m128i _q7 = float2bfloat_lsx(_f7);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    const int out_lane = jj % 8;
                    __lsx_vstelm_d(_q0, p0 + out_lane, 0, 0);
                    __lsx_vstelm_d(_q1, p0 + 8 + out_lane, 0, 0);
                    __lsx_vstelm_d(_q2, p0 + 16 + out_lane, 0, 0);
                    __lsx_vstelm_d(_q3, p0 + 24 + out_lane, 0, 0);
                    __lsx_vstelm_d(_q4, p0 + 32 + out_lane, 0, 0);
                    __lsx_vstelm_d(_q5, p0 + 40 + out_lane, 0, 0);
                    __lsx_vstelm_d(_q6, p0 + 48 + out_lane, 0, 0);
                    __lsx_vstelm_d(_q7, p0 + 56 + out_lane, 0, 0);
                }
                if (out_elempack == 4)
                {
                    __lsx_vstelm_d(_q0, p0, 0, 0);
                    __lsx_vstelm_d(_q1, p0 + 4, 0, 0);
                    __lsx_vstelm_d(_q2, p0 + 8, 0, 0);
                    __lsx_vstelm_d(_q3, p0 + 12, 0, 0);
                    __lsx_vstelm_d(_q4, p0 + 16, 0, 0);
                    __lsx_vstelm_d(_q5, p0 + 20, 0, 0);
                    __lsx_vstelm_d(_q6, p0 + 24, 0, 0);
                    __lsx_vstelm_d(_q7, p0 + 28, 0, 0);
                }
                if (out_elempack == 1)
                {
                    transpose4x4_epi16(_q0, _q1, _q2, _q3);
                    transpose4x4_epi16(_q4, _q5, _q6, _q7);
                    __lsx_vstelm_d(_q0, p0, 0, 0);
                    __lsx_vstelm_d(_q4, p0 + 4, 0, 0);
                    __lsx_vstelm_d(_q0, p0 + out_hstep, 0, 1);
                    __lsx_vstelm_d(_q4, p0 + out_hstep + 4, 0, 1);
                    __lsx_vstelm_d(_q1, p0 + out_hstep * 2, 0, 0);
                    __lsx_vstelm_d(_q5, p0 + out_hstep * 2 + 4, 0, 0);
                    __lsx_vstelm_d(_q1, p0 + out_hstep * 3, 0, 1);
                    __lsx_vstelm_d(_q5, p0 + out_hstep * 3 + 4, 0, 1);
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
            else
            {
                if (out_elempack == 8)
                {
                    transpose4x4_epi16(_q0, _q1, _q2, _q3);
                    transpose4x4_epi16(_q4, _q5, _q6, _q7);
                    __lsx_vst(__lsx_vilvl_d(_q4, _q0), p0, 0);
                    __lsx_vst(__lsx_vilvh_d(_q4, _q0), p0 + 8, 0);
                    __lsx_vst(__lsx_vilvl_d(_q5, _q1), p0 + 16, 0);
                    __lsx_vst(__lsx_vilvh_d(_q5, _q1), p0 + 24, 0);
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    transpose4x4_epi16(_q0, _q1, _q2, _q3);
                    transpose4x4_epi16(_q4, _q5, _q6, _q7);
                    __lsx_vstelm_d(_q0, p0, 0, 0);
                    __lsx_vstelm_d(_q0, p0 + 4, 0, 1);
                    __lsx_vstelm_d(_q1, p0 + 8, 0, 0);
                    __lsx_vstelm_d(_q1, p0 + 12, 0, 1);
                    __lsx_vstelm_d(_q4, p1, 0, 0);
                    __lsx_vstelm_d(_q4, p1 + 4, 0, 1);
                    __lsx_vstelm_d(_q5, p1 + 8, 0, 0);
                    __lsx_vstelm_d(_q5, p1 + 12, 0, 1);
                }
                if (out_elempack == 1)
                {
                    __lsx_vstelm_d(_q0, p0, 0, 0);
                    __lsx_vstelm_d(_q1, p0 + out_hstep, 0, 0);
                    __lsx_vstelm_d(_q2, p0 + out_hstep * 2, 0, 0);
                    __lsx_vstelm_d(_q3, p0 + out_hstep * 3, 0, 0);
                    __lsx_vstelm_d(_q4, p0 + out_hstep * 4, 0, 0);
                    __lsx_vstelm_d(_q5, p0 + out_hstep * 5, 0, 0);
                    __lsx_vstelm_d(_q6, p0 + out_hstep * 6, 0, 0);
                    __lsx_vstelm_d(_q7, p0 + out_hstep * 7, 0, 0);
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
#if __loongarch_asx
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
                    else
#endif // __loongarch_asx
                        if (c_elempack == 4)
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

            __m128i _q0 = float2bfloat_lsx(_f0);
            __m128i _q1 = float2bfloat_lsx(_f1);
            __m128i _q2 = float2bfloat_lsx(_f2);
            __m128i _q3 = float2bfloat_lsx(_f3);
            __m128i _q4 = float2bfloat_lsx(_f4);
            __m128i _q5 = float2bfloat_lsx(_f5);
            __m128i _q6 = float2bfloat_lsx(_f6);
            __m128i _q7 = float2bfloat_lsx(_f7);

            if (output_transpose)
            {
                transpose4x4_epi16(_q0, _q1, _q2, _q3);
                transpose4x4_epi16(_q4, _q5, _q6, _q7);
                __lsx_vst(__lsx_vilvl_d(_q4, _q0), p0, 0);
                __lsx_vst(__lsx_vilvh_d(_q4, _q0), p0 + out_hstep, 0);
                p0 += out_hstep * 2;
            }
            else
            {
                if (out_elempack == 8)
                {
                    transpose4x4_epi16(_q0, _q1, _q2, _q3);
                    transpose4x4_epi16(_q4, _q5, _q6, _q7);
                    __lsx_vst(__lsx_vilvl_d(_q4, _q0), p0, 0);
                    __lsx_vst(__lsx_vilvh_d(_q4, _q0), p0 + 8, 0);
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
                    transpose4x4_epi16(_q0, _q1, _q2, _q3);
                    transpose4x4_epi16(_q4, _q5, _q6, _q7);
                    __lsx_vstelm_d(_q0, p0, 0, 0);
                    __lsx_vstelm_d(_q0, p0 + 4, 0, 1);
                    __lsx_vstelm_d(_q4, p1, 0, 0);
                    __lsx_vstelm_d(_q4, p1 + 4, 0, 1);
                }
                if (out_elempack == 1)
                {
                    __lsx_vstelm_w(_q0, p0, 0, 0);
                    __lsx_vstelm_w(_q1, p0 + out_hstep, 0, 0);
                    __lsx_vstelm_w(_q2, p0 + out_hstep * 2, 0, 0);
                    __lsx_vstelm_w(_q3, p0 + out_hstep * 3, 0, 0);
                    __lsx_vstelm_w(_q4, p0 + out_hstep * 4, 0, 0);
                    __lsx_vstelm_w(_q5, p0 + out_hstep * 5, 0, 0);
                    __lsx_vstelm_w(_q6, p0 + out_hstep * 6, 0, 0);
                    __lsx_vstelm_w(_q7, p0 + out_hstep * 7, 0, 0);
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
#if __loongarch_asx
                    if (c_elempack == 8)
                    {
                        _c0 = __lsx_vld(pC, 0);
                        _c4 = __lsx_vld(pC + 4, 0);
                    }
                    else
#endif // __loongarch_asx
                        if (c_elempack == 4)
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

            __m128i _q0 = float2bfloat_lsx(_f0);
            __m128i _q4 = float2bfloat_lsx(_f4);

            if (output_transpose)
            {
                __lsx_vst(__lsx_vilvl_d(_q4, _q0), p0, 0);
                p0 += out_hstep;
            }
            else
            {
                if (out_elempack == 8)
                {
                    __lsx_vst(__lsx_vilvl_d(_q4, _q0), p0, 0);
                }
                if (out_elempack == 4)
                {
                    __lsx_vstelm_d(_q0, p0, 0, 0);
                    __lsx_vstelm_d(_q4, p0 + out_hstep * 4, 0, 0);
                }
                if (out_elempack == 1)
                {
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
        unsigned short* p0;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + (size_t)j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (size_t)(i + ii) * out_hstep + j * out_elempack;
        }

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
                    __m256 _c0;
                    __m256 _c1;
                    __m256 _c2;
                    __m256 _c3;
                    if (c_elempack == 4)
                    {
                        __m128 _c00 = (__m128)__lsx_vld(pC, 0);
                        __m128 _c10 = (__m128)__lsx_vld(pC + 4, 0);
                        __m128 _c20 = (__m128)__lsx_vld(pC + 8, 0);
                        __m128 _c30 = (__m128)__lsx_vld(pC + 12, 0);
                        transpose4x4_ps(_c00, _c10, _c20, _c30);

                        __m128 _c01 = (__m128)__lsx_vld(pC + 16, 0);
                        __m128 _c11 = (__m128)__lsx_vld(pC + 20, 0);
                        __m128 _c21 = (__m128)__lsx_vld(pC + 24, 0);
                        __m128 _c31 = (__m128)__lsx_vld(pC + 28, 0);
                        transpose4x4_ps(_c01, _c11, _c21, _c31);

                        _c0 = __lasx_concat_128_s(_c00, _c01);
                        _c1 = __lasx_concat_128_s(_c10, _c11);
                        _c2 = __lasx_concat_128_s(_c20, _c21);
                        _c3 = __lasx_concat_128_s(_c30, _c31);
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = (__m256)__lasx_xvld(pC, 0);
                        _c1 = (__m256)__lasx_xvld(pC + c_hstep, 0);
                        _c2 = (__m256)__lasx_xvld(pC + c_hstep * 2, 0);
                        _c3 = (__m256)__lasx_xvld(pC + c_hstep * 3, 0);
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
                    __m256 _c = (__m256)__lasx_xvld(pC, 0);
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

            __m128i _q0 = float2bfloat_lasx(_f0);
            __m128i _q1 = float2bfloat_lasx(_f1);
            __m128i _q2 = float2bfloat_lasx(_f2);
            __m128i _q3 = float2bfloat_lasx(_f3);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    __lsx_vst(_q0, p0, 0);
                    __lsx_vst(_q1, p0 + 8, 0);
                    __lsx_vst(_q2, p0 + 16, 0);
                    __lsx_vst(_q3, p0 + 24, 0);
                }
                if (out_elempack == 4)
                {
                    unsigned short* p1 = p0 + out_hstep * 4;
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
                    transpose8x4_epi16(_q0, _q1, _q2, _q3);
                    __lsx_vstelm_d(_q0, p0, 0, 0);
                    __lsx_vstelm_d(_q0, p0 + out_hstep, 0, 1);
                    __lsx_vstelm_d(_q1, p0 + out_hstep * 2, 0, 0);
                    __lsx_vstelm_d(_q1, p0 + out_hstep * 3, 0, 1);
                    __lsx_vstelm_d(_q2, p0 + out_hstep * 4, 0, 0);
                    __lsx_vstelm_d(_q2, p0 + out_hstep * 5, 0, 1);
                    __lsx_vstelm_d(_q3, p0 + out_hstep * 6, 0, 0);
                    __lsx_vstelm_d(_q3, p0 + out_hstep * 7, 0, 1);
                }
                p0 += out_hstep * 8;
            }
            else
            {
                if (out_elempack == 4)
                {
                    transpose8x4_epi16(_q0, _q1, _q2, _q3);
                    __lsx_vst(_q0, p0, 0);
                    __lsx_vst(_q1, p0 + 8, 0);
                    __lsx_vst(_q2, p0 + 16, 0);
                    __lsx_vst(_q3, p0 + 24, 0);
                }
                if (out_elempack == 1)
                {
                    __lsx_vst(_q0, p0, 0);
                    __lsx_vst(_q1, p0 + out_hstep, 0);
                    __lsx_vst(_q2, p0 + out_hstep * 2, 0);
                    __lsx_vst(_q3, p0 + out_hstep * 3, 0);
                }
                p0 += 8 * out_elempack;
            }
            pp += 32;
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

            __m128i _q0 = float2bfloat_lsx(_f0);
            __m128i _q1 = float2bfloat_lsx(_f1);
            __m128i _q2 = float2bfloat_lsx(_f2);
            __m128i _q3 = float2bfloat_lsx(_f3);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    const int out_lane = jj % 8;
                    __lsx_vstelm_d(_q0, p0 + out_lane, 0, 0);
                    __lsx_vstelm_d(_q1, p0 + 8 + out_lane, 0, 0);
                    __lsx_vstelm_d(_q2, p0 + 16 + out_lane, 0, 0);
                    __lsx_vstelm_d(_q3, p0 + 24 + out_lane, 0, 0);
                }
                if (out_elempack == 4)
                {
                    __lsx_vstelm_d(_q0, p0, 0, 0);
                    __lsx_vstelm_d(_q1, p0 + 4, 0, 0);
                    __lsx_vstelm_d(_q2, p0 + 8, 0, 0);
                    __lsx_vstelm_d(_q3, p0 + 12, 0, 0);
                }
                if (out_elempack == 1)
                {
                    transpose4x4_epi16(_q0, _q1, _q2, _q3);
                    __lsx_vstelm_d(_q0, p0, 0, 0);
                    __lsx_vstelm_d(_q0, p0 + out_hstep, 0, 1);
                    __lsx_vstelm_d(_q1, p0 + out_hstep * 2, 0, 0);
                    __lsx_vstelm_d(_q1, p0 + out_hstep * 3, 0, 1);
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
            else
            {
                if (out_elempack == 4)
                {
                    transpose4x4_epi16(_q0, _q1, _q2, _q3);
                    __lsx_vstelm_d(_q0, p0, 0, 0);
                    __lsx_vstelm_d(_q0, p0 + 4, 0, 1);
                    __lsx_vstelm_d(_q1, p0 + 8, 0, 0);
                    __lsx_vstelm_d(_q1, p0 + 12, 0, 1);
                }
                if (out_elempack == 1)
                {
                    __lsx_vstelm_d(_q0, p0, 0, 0);
                    __lsx_vstelm_d(_q1, p0 + out_hstep, 0, 0);
                    __lsx_vstelm_d(_q2, p0 + out_hstep * 2, 0, 0);
                    __lsx_vstelm_d(_q3, p0 + out_hstep * 3, 0, 0);
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

            __m128i _q0 = float2bfloat_lsx(_f0);
            __m128i _q1 = float2bfloat_lsx(_f1);
            __m128i _q2 = float2bfloat_lsx(_f2);
            __m128i _q3 = float2bfloat_lsx(_f3);

            if (output_transpose)
            {
                transpose4x4_epi16(_q0, _q1, _q2, _q3);
                __lsx_vstelm_d(_q0, p0, 0, 0);
                __lsx_vstelm_d(_q0, p0 + out_hstep, 0, 1);
                p0 += out_hstep * 2;
            }
            else
            {
                if (out_elempack == 4)
                {
                    transpose4x4_epi16(_q0, _q1, _q2, _q3);
                    __lsx_vstelm_d(_q0, p0, 0, 0);
                    __lsx_vstelm_d(_q0, p0 + 4, 0, 1);
                }
                if (out_elempack == 1)
                {
                    __lsx_vstelm_w(_q0, p0, 0, 0);
                    __lsx_vstelm_w(_q1, p0 + out_hstep, 0, 0);
                    __lsx_vstelm_w(_q2, p0 + out_hstep * 2, 0, 0);
                    __lsx_vstelm_w(_q3, p0 + out_hstep * 3, 0, 0);
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

            __m128i _q = float2bfloat_lsx(_f0);

            if (output_transpose)
            {
                __lsx_vstelm_d(_q, p0, 0, 0);
                p0 += out_hstep;
            }
            else
            {
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
        unsigned short* p0;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + (size_t)j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (size_t)(i + ii) * out_hstep + j * out_elempack;
        }

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
                        __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                        _f0 = __lasx_xvfmadd_s(_c0, _beta, _f0);
                        _f1 = __lasx_xvfmadd_s(_c1, _beta, _f1);
                    }
                    pC += 8 * c_elempack;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c = (__m256)__lasx_xvld(pC, 0);
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

            __m128i _q0 = float2bfloat_lasx(_f0);
            __m128i _q1 = float2bfloat_lasx(_f1);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    __lsx_vst(_q0, p0, 0);
                    __lsx_vst(_q1, p0 + 8, 0);
                }
                if (out_elempack == 4)
                {
                    __lsx_vstelm_d(_q0, p0, 0, 0);
                    __lsx_vstelm_d(_q1, p0 + 4, 0, 0);
                    __lsx_vstelm_d(_q0, p0 + out_hstep * 4, 0, 1);
                    __lsx_vstelm_d(_q1, p0 + out_hstep * 4 + 4, 0, 1);
                }
                if (out_elempack == 1)
                {
                    __m128i _t0 = __lsx_vilvl_h(_q1, _q0);
                    __m128i _t1 = __lsx_vilvh_h(_q1, _q0);
                    __lsx_vstelm_w(_t0, p0, 0, 0);
                    __lsx_vstelm_w(_t0, p0 + out_hstep, 0, 1);
                    __lsx_vstelm_w(_t0, p0 + out_hstep * 2, 0, 2);
                    __lsx_vstelm_w(_t0, p0 + out_hstep * 3, 0, 3);
                    __lsx_vstelm_w(_t1, p0 + out_hstep * 4, 0, 0);
                    __lsx_vstelm_w(_t1, p0 + out_hstep * 5, 0, 1);
                    __lsx_vstelm_w(_t1, p0 + out_hstep * 6, 0, 2);
                    __lsx_vstelm_w(_t1, p0 + out_hstep * 7, 0, 3);
                }
                p0 += out_hstep * 8;
            }
            else
            {
                __lsx_vst(_q0, p0, 0);
                __lsx_vst(_q1, p0 + out_hstep, 0);
                p0 += 8;
            }
            pp += 16;
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

            __m128i _q0 = float2bfloat_lsx(_f0);
            __m128i _q1 = float2bfloat_lsx(_f1);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    const int out_lane = jj % 8;
                    __lsx_vstelm_d(_q0, p0 + out_lane, 0, 0);
                    __lsx_vstelm_d(_q1, p0 + 8 + out_lane, 0, 0);
                }
                if (out_elempack == 4)
                {
                    __lsx_vstelm_d(_q0, p0, 0, 0);
                    __lsx_vstelm_d(_q1, p0 + 4, 0, 0);
                }
                if (out_elempack == 1)
                {
                    __m128i _t = __lsx_vilvl_h(_q1, _q0);
                    __lsx_vstelm_w(_t, p0, 0, 0);
                    __lsx_vstelm_w(_t, p0 + out_hstep, 0, 1);
                    __lsx_vstelm_w(_t, p0 + out_hstep * 2, 0, 2);
                    __lsx_vstelm_w(_t, p0 + out_hstep * 3, 0, 3);
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
            else
            {
                __lsx_vstelm_d(_q0, p0, 0, 0);
                __lsx_vstelm_d(_q1, p0 + out_hstep, 0, 0);
                p0 += 4;
            }
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __loongarch_sx
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
                        __m128 _beta = __lsx_vreplfr2vr_s(beta);
                        _f0 = __lsx_vfmadd_s(_c0, _beta, _f0);
                        _f1 = __lsx_vfmadd_s(_c1, _beta, _f1);
                    }
                    pC += 2;
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
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
            }

            __m128i _q0 = float2bfloat_lsx(_f0);
            __m128i _q1 = float2bfloat_lsx(_f1);

            if (output_transpose)
            {
                __m128i _t = __lsx_vilvl_h(_q1, _q0);
                __lsx_vstelm_w(_t, p0, 0, 0);
                __lsx_vstelm_w(_t, p0 + out_hstep, 0, 1);
                p0 += out_hstep * 2;
            }
            else
            {
                __lsx_vstelm_w(_q0, p0, 0, 0);
                __lsx_vstelm_w(_q1, p0 + out_hstep, 0, 0);
                p0 += 2;
            }
#else
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
                    const float cc0 = pC[0] * beta;
                    const float cc1 = pC[1] * beta;
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

            if (output_transpose)
            {
                p0[0] = float32_to_bfloat16(f00);
                p0[1] = float32_to_bfloat16(f10);
                p0[out_hstep] = float32_to_bfloat16(f01);
                p0[out_hstep + 1] = float32_to_bfloat16(f11);
                p0 += out_hstep * 2;
            }
            else
            {
                p0[0] = float32_to_bfloat16(f00);
                p0[1] = float32_to_bfloat16(f01);
                p0[out_hstep + 0] = float32_to_bfloat16(f10);
                p0[out_hstep + 1] = float32_to_bfloat16(f11);
                p0 += 2;
            }
#endif // __loongarch_sx
            pp += 4;
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

            if (output_transpose)
            {
                p0[0] = float32_to_bfloat16(f0);
                p0[1] = float32_to_bfloat16(f1);
                p0 += out_hstep;
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
        unsigned short* p0;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + (size_t)j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (size_t)(i + ii) * out_hstep + j * out_elempack;
        }

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

            __m128i _bf = float2bfloat_lasx(_f0);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    __lsx_vst(_bf, p0, 0);
                }
                if (out_elempack == 4)
                {
                    __lsx_vstelm_d(_bf, p0, 0, 0);
                    __lsx_vstelm_d(_bf, p0 + out_hstep * 4, 0, 1);
                }
                if (out_elempack == 1)
                {
                    __lsx_vstelm_h(_bf, p0, 0, 0);
                    __lsx_vstelm_h(_bf, p0 + out_hstep, 0, 1);
                    __lsx_vstelm_h(_bf, p0 + out_hstep * 2, 0, 2);
                    __lsx_vstelm_h(_bf, p0 + out_hstep * 3, 0, 3);
                    __lsx_vstelm_h(_bf, p0 + out_hstep * 4, 0, 4);
                    __lsx_vstelm_h(_bf, p0 + out_hstep * 5, 0, 5);
                    __lsx_vstelm_h(_bf, p0 + out_hstep * 6, 0, 6);
                    __lsx_vstelm_h(_bf, p0 + out_hstep * 7, 0, 7);
                }
                p0 += out_hstep * 8;
            }
            else
            {
                __lsx_vst(_bf, p0, 0);
                p0 += 8;
            }
            pp += 8;
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

            __m128i _bf = float2bfloat_lsx(_f0);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    __lsx_vstelm_d(_bf, p0 + jj % 8, 0, 0);
                    if (jj % 8 == 4)
                        p0 += out_hstep * 8;
                }
                if (out_elempack == 4)
                {
                    __lsx_vstelm_d(_bf, p0, 0, 0);
                    p0 += out_hstep * 4;
                }
                if (out_elempack == 1)
                {
                    __lsx_vstelm_h(_bf, p0, 0, 0);
                    __lsx_vstelm_h(_bf, p0 + out_hstep, 0, 1);
                    __lsx_vstelm_h(_bf, p0 + out_hstep * 2, 0, 2);
                    __lsx_vstelm_h(_bf, p0 + out_hstep * 3, 0, 3);
                    p0 += out_hstep * 4;
                }
            }
            else
            {
                __lsx_vstelm_d(_bf, p0, 0, 0);
                p0 += 4;
            }
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            float f0 = pp[0];
            float f1 = pp[1];
            pp += 2;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                    f1 += c0;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    f0 += pC[0] * beta;
                    f1 += pC[1] * beta;
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                f0 *= alpha;
                f1 *= alpha;
            }

            if (output_transpose)
            {
                p0[0] = float32_to_bfloat16(f0);
                p0[out_hstep] = float32_to_bfloat16(f1);
                p0 += out_hstep * 2;
            }
            else
            {
                p0[0] = float32_to_bfloat16(f0);
                p0[1] = float32_to_bfloat16(f1);
                p0 += 2;
            }
        }
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
            if (output_transpose)
            {
                p0[0] = float32_to_bfloat16(f0);
                p0 += out_hstep;
            }
            else
            {
                p0[0] = float32_to_bfloat16(f0);
                p0++;
            }
        }
    }
}

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
#if __loongarch_asx
                    const __m256i _abs_mask = (__m256i)__lasx_xvreplgr2vr_w(0x7fffffff);
                    __m256 _absmax = (__m256)__lasx_xvreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(p0a, 0));
                        _absmax = __lasx_xvfmax_s(_absmax, (__m256)__lasx_xvand_v((__m256i)_p, _abs_mask));
                        p0a += 8;
                    }

                    const __m256 _v127 = __lasx_xvreplfr2vr_s(127.f);
                    __lasx_xvst(__lasx_xvfdiv_s(_absmax, _v127), pd, 0);
                    pd += 8;

                    __m256i _zeromask = (__m256i)__lasx_xvfcmp_ceq_s(_absmax, (__m256)__lasx_xvreplgr2vr_w(0));
                    _absmax = (__m256)__lasx_xvbitsel_v((__m256i)_absmax, (__m256i)_v127, _zeromask);
                    __m256 _scale = __lasx_xvfdiv_s(_v127, _absmax);

                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m256 _p0 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0, 0)), _scale);
                        __m256 _p1 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0 + 8, 0)), _scale);
                        __m256 _p2 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0 + 16, 0)), _scale);
                        __m256 _p3 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0 + 24, 0)), _scale);
                        __m128i _q0 = __lasx_extract_128_lo(float2int8(_p0));
                        __m128i _q1 = __lasx_extract_128_lo(float2int8(_p1));
                        __m128i _q2 = __lasx_extract_128_lo(float2int8(_p2));
                        __m128i _q3 = __lasx_extract_128_lo(float2int8(_p3));
                        __m128i _q01 = __lsx_vilvl_b(_q1, _q0);
                        __m128i _q23 = __lsx_vilvl_b(_q3, _q2);
                        __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp, 0);
                        __lsx_vst(__lsx_vilvh_h(_q23, _q01), pp + 16, 0);
                        pp += 32;
                        p0 += 32;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m256 _p = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0, 0)), _scale);
                        __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(_p)), pp, 0, 0);
                        pp += 8;
                        p0 += 8;
                    }
#else
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
#endif // __loongarch_asx
                }
            }
            if (elempack == 4)
            {
                const unsigned short* p0 = (const unsigned short*)A + (size_t)(i + ii) * A_hstep + k * 4;
                const unsigned short* p1 = p0 + A_hstep * 4;

                for (int g = 0; g < local_block_count; g++)
                {
                    const int max_kk0 = std::min(max_kk - g * block_size, block_size);
#if __loongarch_asx
                    const __m256i _abs_mask = (__m256i)__lasx_xvreplgr2vr_w(0x7fffffff);
                    __m256 _absmax = (__m256)__lasx_xvreplgr2vr_w(0);

                    const unsigned short* p0a = p0;
                    const unsigned short* p1a = p1;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        __m128i _p01 = __lsx_vilvl_d(__lsx_vldrepl_d(p1a, 0), __lsx_vldrepl_d(p0a, 0));
                        __m256 _p = bfloat2float_lasx(_p01);
                        _absmax = __lasx_xvfmax_s(_absmax, (__m256)__lasx_xvand_v((__m256i)_p, _abs_mask));
                        p0a += 4;
                        p1a += 4;
                    }

                    const __m256 _v127 = __lasx_xvreplfr2vr_s(127.f);
                    __lasx_xvst(__lasx_xvfdiv_s(_absmax, _v127), pd, 0);
                    pd += 8;

                    __m256i _zeromask = (__m256i)__lasx_xvfcmp_ceq_s(_absmax, (__m256)__lasx_xvreplgr2vr_w(0));
                    _absmax = (__m256)__lasx_xvbitsel_v((__m256i)_absmax, (__m256i)_v127, _zeromask);
                    __m256 _scale = __lasx_xvfdiv_s(_v127, _absmax);

                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        __m256 _p0 = bfloat2float_lasx(__lsx_vilvl_d(__lsx_vldrepl_d(p1, 0), __lsx_vldrepl_d(p0, 0)));
                        __m256 _p1 = bfloat2float_lasx(__lsx_vilvl_d(__lsx_vldrepl_d(p1 + 4, 0), __lsx_vldrepl_d(p0 + 4, 0)));
                        __m256 _p2 = bfloat2float_lasx(__lsx_vilvl_d(__lsx_vldrepl_d(p1 + 8, 0), __lsx_vldrepl_d(p0 + 8, 0)));
                        __m256 _p3 = bfloat2float_lasx(__lsx_vilvl_d(__lsx_vldrepl_d(p1 + 12, 0), __lsx_vldrepl_d(p0 + 12, 0)));
                        __m128i _q0 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p0, _scale)));
                        __m128i _q1 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p1, _scale)));
                        __m128i _q2 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p2, _scale)));
                        __m128i _q3 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p3, _scale)));
                        __m128i _q01 = __lsx_vilvl_b(_q1, _q0);
                        __m128i _q23 = __lsx_vilvl_b(_q3, _q2);
                        __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp, 0);
                        __lsx_vst(__lsx_vilvh_h(_q23, _q01), pp + 16, 0);
                        pp += 32;
                        p0 += 16;
                        p1 += 16;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m256 _p = bfloat2float_lasx(__lsx_vilvl_d(__lsx_vldrepl_d(p1, 0), __lsx_vldrepl_d(p0, 0)));
                        __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p, _scale))), pp, 0, 0);
                        pp += 8;
                        p0 += 4;
                        p1 += 4;
                    }
#else
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
#endif // __loongarch_asx
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
#if __loongarch_asx
                const __m256i _abs_mask = (__m256i)__lasx_xvreplgr2vr_w(0x7fffffff);
                __m256 _absmax = (__m256)__lasx_xvreplgr2vr_w(0);

                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m256 _s = __lasx_xvreplfr2vr_s(*psa++);
                    __m256 _p = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0a, 0)), _s);
                    _absmax = __lasx_xvfmax_s(_absmax, (__m256)__lasx_xvand_v((__m256i)_p, _abs_mask));
                    p0a += 8;
                }

                const __m256 _v127 = __lasx_xvreplfr2vr_s(127.f);
                __lasx_xvst(__lasx_xvfdiv_s(_absmax, _v127), pd, 0);
                pd += 8;

                __m256i _zeromask = (__m256i)__lasx_xvfcmp_ceq_s(_absmax, (__m256)__lasx_xvreplgr2vr_w(0));
                _absmax = (__m256)__lasx_xvbitsel_v((__m256i)_absmax, (__m256i)_v127, _zeromask);
                __m256 _scale = __lasx_xvfdiv_s(_v127, _absmax);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256 _p0 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0, 0)), __lasx_xvreplfr2vr_s(ps[0]));
                    __m256 _p1 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0 + 8, 0)), __lasx_xvreplfr2vr_s(ps[1]));
                    __m256 _p2 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0 + 16, 0)), __lasx_xvreplfr2vr_s(ps[2]));
                    __m256 _p3 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0 + 24, 0)), __lasx_xvreplfr2vr_s(ps[3]));
                    __m128i _q0 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p0, _scale)));
                    __m128i _q1 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p1, _scale)));
                    __m128i _q2 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p2, _scale)));
                    __m128i _q3 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p3, _scale)));
                    __m128i _q01 = __lsx_vilvl_b(_q1, _q0);
                    __m128i _q23 = __lsx_vilvl_b(_q3, _q2);
                    __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp, 0);
                    __lsx_vst(__lsx_vilvh_h(_q23, _q01), pp + 16, 0);
                    pp += 32;
                    p0 += 32;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m256 _p = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0, 0)), __lasx_xvreplfr2vr_s(*ps++));
                    __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p, _scale))), pp, 0, 0);
                    pp += 8;
                    p0 += 8;
                }
#else
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
#endif // __loongarch_asx
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
#if __loongarch_asx
                const __m256i _abs_mask = (__m256i)__lasx_xvreplgr2vr_w(0x7fffffff);
                __m256 _absmax = (__m256)__lasx_xvreplgr2vr_w(0);

                const unsigned short* p0a = p0;
                const unsigned short* p1a = p1;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    __m256 _p = bfloat2float_lasx(__lsx_vilvl_d(__lsx_vldrepl_d(p1a, 0), __lsx_vldrepl_d(p0a, 0)));
                    _p = __lasx_xvfmul_s(_p, __lasx_xvreplfr2vr_s(*psa++));
                    _absmax = __lasx_xvfmax_s(_absmax, (__m256)__lasx_xvand_v((__m256i)_p, _abs_mask));
                    p0a += 4;
                    p1a += 4;
                }

                const __m256 _v127 = __lasx_xvreplfr2vr_s(127.f);
                __lasx_xvst(__lasx_xvfdiv_s(_absmax, _v127), pd, 0);
                pd += 8;

                __m256i _zeromask = (__m256i)__lasx_xvfcmp_ceq_s(_absmax, (__m256)__lasx_xvreplgr2vr_w(0));
                _absmax = (__m256)__lasx_xvbitsel_v((__m256i)_absmax, (__m256i)_v127, _zeromask);
                __m256 _scale = __lasx_xvfdiv_s(_v127, _absmax);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    __m256 _p0 = bfloat2float_lasx(__lsx_vilvl_d(__lsx_vldrepl_d(p1, 0), __lsx_vldrepl_d(p0, 0)));
                    __m256 _p1 = bfloat2float_lasx(__lsx_vilvl_d(__lsx_vldrepl_d(p1 + 4, 0), __lsx_vldrepl_d(p0 + 4, 0)));
                    __m256 _p2 = bfloat2float_lasx(__lsx_vilvl_d(__lsx_vldrepl_d(p1 + 8, 0), __lsx_vldrepl_d(p0 + 8, 0)));
                    __m256 _p3 = bfloat2float_lasx(__lsx_vilvl_d(__lsx_vldrepl_d(p1 + 12, 0), __lsx_vldrepl_d(p0 + 12, 0)));
                    __m128i _q0 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(__lasx_xvfmul_s(_p0, __lasx_xvreplfr2vr_s(ps[0])), _scale)));
                    __m128i _q1 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(__lasx_xvfmul_s(_p1, __lasx_xvreplfr2vr_s(ps[1])), _scale)));
                    __m128i _q2 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(__lasx_xvfmul_s(_p2, __lasx_xvreplfr2vr_s(ps[2])), _scale)));
                    __m128i _q3 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(__lasx_xvfmul_s(_p3, __lasx_xvreplfr2vr_s(ps[3])), _scale)));
                    __m128i _q01 = __lsx_vilvl_b(_q1, _q0);
                    __m128i _q23 = __lsx_vilvl_b(_q3, _q2);
                    __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp, 0);
                    __lsx_vst(__lsx_vilvh_h(_q23, _q01), pp + 16, 0);
                    pp += 32;
                    p0 += 16;
                    p1 += 16;
                    ps += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    __m256 _p = bfloat2float_lasx(__lsx_vilvl_d(__lsx_vldrepl_d(p1, 0), __lsx_vldrepl_d(p0, 0)));
                    _p = __lasx_xvfmul_s(__lasx_xvfmul_s(_p, __lasx_xvreplfr2vr_s(*ps++)), _scale);
                    __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(_p)), pp, 0, 0);
                    pp += 8;
                    p0 += 4;
                    p1 += 4;
                }
#else
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
#endif // __loongarch_asx
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
#if __loongarch_asx
                    const __m256i _abs_mask = (__m256i)__lasx_xvreplgr2vr_w(0x7fffffff);
                    __m256 _absmax0 = (__m256)__lasx_xvreplgr2vr_w(0);
                    __m256 _absmax1 = _absmax0;
                    __m256 _absmax2 = _absmax0;
                    __m256 _absmax3 = _absmax0;
                    __m256 _absmax4 = _absmax0;
                    __m256 _absmax5 = _absmax0;
                    __m256 _absmax6 = _absmax0;
                    __m256 _absmax7 = _absmax0;

                    const unsigned short* p0a = p0;
                    int kk = 0;
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        _absmax0 = __lasx_xvfmax_s(_absmax0, (__m256)__lasx_xvand_v((__m256i)bfloat2float_lasx((__m128i)__lsx_vld(p0a, 0)), _abs_mask));
                        _absmax1 = __lasx_xvfmax_s(_absmax1, (__m256)__lasx_xvand_v((__m256i)bfloat2float_lasx((__m128i)__lsx_vld(p0a + 8, 0)), _abs_mask));
                        _absmax2 = __lasx_xvfmax_s(_absmax2, (__m256)__lasx_xvand_v((__m256i)bfloat2float_lasx((__m128i)__lsx_vld(p0a + 16, 0)), _abs_mask));
                        _absmax3 = __lasx_xvfmax_s(_absmax3, (__m256)__lasx_xvand_v((__m256i)bfloat2float_lasx((__m128i)__lsx_vld(p0a + 24, 0)), _abs_mask));
                        _absmax4 = __lasx_xvfmax_s(_absmax4, (__m256)__lasx_xvand_v((__m256i)bfloat2float_lasx((__m128i)__lsx_vld(p0a + 32, 0)), _abs_mask));
                        _absmax5 = __lasx_xvfmax_s(_absmax5, (__m256)__lasx_xvand_v((__m256i)bfloat2float_lasx((__m128i)__lsx_vld(p0a + 40, 0)), _abs_mask));
                        _absmax6 = __lasx_xvfmax_s(_absmax6, (__m256)__lasx_xvand_v((__m256i)bfloat2float_lasx((__m128i)__lsx_vld(p0a + 48, 0)), _abs_mask));
                        _absmax7 = __lasx_xvfmax_s(_absmax7, (__m256)__lasx_xvand_v((__m256i)bfloat2float_lasx((__m128i)__lsx_vld(p0a + 56, 0)), _abs_mask));
                        p0a += A_hstep * 8;
                    }

                    transpose8x8_ps(_absmax0, _absmax1, _absmax2, _absmax3, _absmax4, _absmax5, _absmax6, _absmax7);
                    _absmax0 = __lasx_xvfmax_s(__lasx_xvfmax_s(__lasx_xvfmax_s(_absmax0, _absmax1), __lasx_xvfmax_s(_absmax2, _absmax3)), __lasx_xvfmax_s(__lasx_xvfmax_s(_absmax4, _absmax5), __lasx_xvfmax_s(_absmax6, _absmax7)));
                    for (; kk < max_kk0; kk++)
                    {
                        __m128i _p = __lsx_vldrepl_h(p0a, 0);
                        _p = __lsx_vinsgr2vr_h(_p, p0a[8], 1);
                        _p = __lsx_vinsgr2vr_h(_p, p0a[16], 2);
                        _p = __lsx_vinsgr2vr_h(_p, p0a[24], 3);
                        _p = __lsx_vinsgr2vr_h(_p, p0a[32], 4);
                        _p = __lsx_vinsgr2vr_h(_p, p0a[40], 5);
                        _p = __lsx_vinsgr2vr_h(_p, p0a[48], 6);
                        _p = __lsx_vinsgr2vr_h(_p, p0a[56], 7);
                        _absmax0 = __lasx_xvfmax_s(_absmax0, (__m256)__lasx_xvand_v((__m256i)bfloat2float_lasx(_p), _abs_mask));
                        p0a++;
                    }

                    const __m256 _v127 = __lasx_xvreplfr2vr_s(127.f);
                    __lasx_xvst(__lasx_xvfdiv_s(_absmax0, _v127), pd, 0);
                    pd += 8;

                    __m256i _zeromask = (__m256i)__lasx_xvfcmp_ceq_s(_absmax0, (__m256)__lasx_xvreplgr2vr_w(0));
                    _absmax0 = (__m256)__lasx_xvbitsel_v((__m256i)_absmax0, (__m256i)_v127, _zeromask);
                    __m256 _scale = __lasx_xvfdiv_s(_v127, _absmax0);

                    kk = 0;
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        __m256 _p0 = bfloat2float_lasx((__m128i)__lsx_vld(p0, 0));
                        __m256 _p1 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 8, 0));
                        __m256 _p2 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 16, 0));
                        __m256 _p3 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 24, 0));
                        __m256 _p4 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 32, 0));
                        __m256 _p5 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 40, 0));
                        __m256 _p6 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 48, 0));
                        __m256 _p7 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 56, 0));
                        transpose8x8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);
                        __m128i _q0 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p0, _scale)));
                        __m128i _q1 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p1, _scale)));
                        __m128i _q2 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p2, _scale)));
                        __m128i _q3 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p3, _scale)));
                        __m128i _q01 = __lsx_vilvl_b(_q1, _q0);
                        __m128i _q23 = __lsx_vilvl_b(_q3, _q2);
                        __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp, 0);
                        __lsx_vst(__lsx_vilvh_h(_q23, _q01), pp + 16, 0);
                        _q0 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p4, _scale)));
                        _q1 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p5, _scale)));
                        _q2 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p6, _scale)));
                        _q3 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p7, _scale)));
                        _q01 = __lsx_vilvl_b(_q1, _q0);
                        _q23 = __lsx_vilvl_b(_q3, _q2);
                        __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp + 32, 0);
                        __lsx_vst(__lsx_vilvh_h(_q23, _q01), pp + 48, 0);
                        pp += 64;
                        p0 += A_hstep * 8;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m128i _p = __lsx_vldrepl_h(p0, 0);
                        _p = __lsx_vinsgr2vr_h(_p, p0[8], 1);
                        _p = __lsx_vinsgr2vr_h(_p, p0[16], 2);
                        _p = __lsx_vinsgr2vr_h(_p, p0[24], 3);
                        _p = __lsx_vinsgr2vr_h(_p, p0[32], 4);
                        _p = __lsx_vinsgr2vr_h(_p, p0[40], 5);
                        _p = __lsx_vinsgr2vr_h(_p, p0[48], 6);
                        _p = __lsx_vinsgr2vr_h(_p, p0[56], 7);
                        __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(bfloat2float_lasx(_p), _scale))), pp, 0, 0);
                        pp += 8;
                        p0++;
                    }
#else
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
#endif // __loongarch_asx
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
#if __loongarch_asx
                    const __m256i _abs_mask = (__m256i)__lasx_xvreplgr2vr_w(0x7fffffff);
                    __m256 _absmax0 = (__m256)__lasx_xvreplgr2vr_w(0);
                    __m256 _absmax1 = _absmax0;
                    __m256 _absmax2 = _absmax0;
                    __m256 _absmax3 = _absmax0;
                    __m256 _absmax4 = _absmax0;
                    __m256 _absmax5 = _absmax0;
                    __m256 _absmax6 = _absmax0;
                    __m256 _absmax7 = _absmax0;

                    const unsigned short* p0a = p0;
                    const float* psa = ps;
                    int kk = 0;
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        __m256 _s = (__m256)__lasx_xvld(psa, 0);
                        __m256 _p0 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0a, 0)), _s);
                        __m256 _p1 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0a + 8, 0)), _s);
                        __m256 _p2 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0a + 16, 0)), _s);
                        __m256 _p3 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0a + 24, 0)), _s);
                        __m256 _p4 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0a + 32, 0)), _s);
                        __m256 _p5 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0a + 40, 0)), _s);
                        __m256 _p6 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0a + 48, 0)), _s);
                        __m256 _p7 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0a + 56, 0)), _s);
                        _absmax0 = __lasx_xvfmax_s(_absmax0, (__m256)__lasx_xvand_v((__m256i)_p0, _abs_mask));
                        _absmax1 = __lasx_xvfmax_s(_absmax1, (__m256)__lasx_xvand_v((__m256i)_p1, _abs_mask));
                        _absmax2 = __lasx_xvfmax_s(_absmax2, (__m256)__lasx_xvand_v((__m256i)_p2, _abs_mask));
                        _absmax3 = __lasx_xvfmax_s(_absmax3, (__m256)__lasx_xvand_v((__m256i)_p3, _abs_mask));
                        _absmax4 = __lasx_xvfmax_s(_absmax4, (__m256)__lasx_xvand_v((__m256i)_p4, _abs_mask));
                        _absmax5 = __lasx_xvfmax_s(_absmax5, (__m256)__lasx_xvand_v((__m256i)_p5, _abs_mask));
                        _absmax6 = __lasx_xvfmax_s(_absmax6, (__m256)__lasx_xvand_v((__m256i)_p6, _abs_mask));
                        _absmax7 = __lasx_xvfmax_s(_absmax7, (__m256)__lasx_xvand_v((__m256i)_p7, _abs_mask));
                        p0a += A_hstep * 8;
                        psa += 8;
                    }

                    transpose8x8_ps(_absmax0, _absmax1, _absmax2, _absmax3, _absmax4, _absmax5, _absmax6, _absmax7);
                    _absmax0 = __lasx_xvfmax_s(__lasx_xvfmax_s(__lasx_xvfmax_s(_absmax0, _absmax1), __lasx_xvfmax_s(_absmax2, _absmax3)), __lasx_xvfmax_s(__lasx_xvfmax_s(_absmax4, _absmax5), __lasx_xvfmax_s(_absmax6, _absmax7)));
                    for (; kk < max_kk0; kk++)
                    {
                        __m128i _p = __lsx_vldrepl_h(p0a, 0);
                        _p = __lsx_vinsgr2vr_h(_p, p0a[8], 1);
                        _p = __lsx_vinsgr2vr_h(_p, p0a[16], 2);
                        _p = __lsx_vinsgr2vr_h(_p, p0a[24], 3);
                        _p = __lsx_vinsgr2vr_h(_p, p0a[32], 4);
                        _p = __lsx_vinsgr2vr_h(_p, p0a[40], 5);
                        _p = __lsx_vinsgr2vr_h(_p, p0a[48], 6);
                        _p = __lsx_vinsgr2vr_h(_p, p0a[56], 7);
                        __m256 _v = __lasx_xvfmul_s(bfloat2float_lasx(_p), __lasx_xvreplfr2vr_s(*psa++));
                        _absmax0 = __lasx_xvfmax_s(_absmax0, (__m256)__lasx_xvand_v((__m256i)_v, _abs_mask));
                        p0a++;
                    }

                    const __m256 _v127 = __lasx_xvreplfr2vr_s(127.f);
                    __lasx_xvst(__lasx_xvfdiv_s(_absmax0, _v127), pd, 0);
                    pd += 8;

                    __m256i _zeromask = (__m256i)__lasx_xvfcmp_ceq_s(_absmax0, (__m256)__lasx_xvreplgr2vr_w(0));
                    _absmax0 = (__m256)__lasx_xvbitsel_v((__m256i)_absmax0, (__m256i)_v127, _zeromask);
                    __m256 _scale = __lasx_xvfdiv_s(_v127, _absmax0);

                    kk = 0;
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        __m256 _s = (__m256)__lasx_xvld(ps, 0);
                        __m256 _p0 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0, 0)), _s);
                        __m256 _p1 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0 + 8, 0)), _s);
                        __m256 _p2 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0 + 16, 0)), _s);
                        __m256 _p3 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0 + 24, 0)), _s);
                        __m256 _p4 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0 + 32, 0)), _s);
                        __m256 _p5 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0 + 40, 0)), _s);
                        __m256 _p6 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0 + 48, 0)), _s);
                        __m256 _p7 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i)__lsx_vld(p0 + 56, 0)), _s);
                        transpose8x8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);
                        __m128i _q0 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p0, _scale)));
                        __m128i _q1 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p1, _scale)));
                        __m128i _q2 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p2, _scale)));
                        __m128i _q3 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p3, _scale)));
                        __m128i _q01 = __lsx_vilvl_b(_q1, _q0);
                        __m128i _q23 = __lsx_vilvl_b(_q3, _q2);
                        __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp, 0);
                        __lsx_vst(__lsx_vilvh_h(_q23, _q01), pp + 16, 0);
                        _q0 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p4, _scale)));
                        _q1 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p5, _scale)));
                        _q2 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p6, _scale)));
                        _q3 = __lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_p7, _scale)));
                        _q01 = __lsx_vilvl_b(_q1, _q0);
                        _q23 = __lsx_vilvl_b(_q3, _q2);
                        __lsx_vst(__lsx_vilvl_h(_q23, _q01), pp + 32, 0);
                        __lsx_vst(__lsx_vilvh_h(_q23, _q01), pp + 48, 0);
                        pp += 64;
                        p0 += A_hstep * 8;
                        ps += 8;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        __m128i _p = __lsx_vldrepl_h(p0, 0);
                        _p = __lsx_vinsgr2vr_h(_p, p0[8], 1);
                        _p = __lsx_vinsgr2vr_h(_p, p0[16], 2);
                        _p = __lsx_vinsgr2vr_h(_p, p0[24], 3);
                        _p = __lsx_vinsgr2vr_h(_p, p0[32], 4);
                        _p = __lsx_vinsgr2vr_h(_p, p0[40], 5);
                        _p = __lsx_vinsgr2vr_h(_p, p0[48], 6);
                        _p = __lsx_vinsgr2vr_h(_p, p0[56], 7);
                        __m256 _v = __lasx_xvfmul_s(bfloat2float_lasx(_p), __lasx_xvreplfr2vr_s(*ps++));
                        __lsx_vstelm_d(__lasx_extract_128_lo(float2int8(__lasx_xvfmul_s(_v, _scale))), pp, 0, 0);
                        pp += 8;
                        p0++;
                    }
#else
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
#endif // __loongarch_asx
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

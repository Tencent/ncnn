// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convolution_im2col_pack_A_tile_bf16s(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    // A = (pa, maxk, inch/pa), outch
    const int A_hstep = A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;
        const float* p4 = (const float*)A + (i + ii + 4) * A_hstep + k;
        const float* p5 = (const float*)A + (i + ii + 5) * A_hstep + k;
        const float* p6 = (const float*)A + (i + ii + 6) * A_hstep + k;
        const float* p7 = (const float*)A + (i + ii + 7) * A_hstep + k;

        int kk = 0;
        for (; kk + 7 < max_kk; kk += 8)
        {
            __m256 _r0 = (__m256)__lasx_xvld(p0, 0);
            __m256 _r1 = (__m256)__lasx_xvld(p1, 0);
            __m256 _r2 = (__m256)__lasx_xvld(p2, 0);
            __m256 _r3 = (__m256)__lasx_xvld(p3, 0);
            __m256 _r4 = (__m256)__lasx_xvld(p4, 0);
            __m256 _r5 = (__m256)__lasx_xvld(p5, 0);
            __m256 _r6 = (__m256)__lasx_xvld(p6, 0);
            __m256 _r7 = (__m256)__lasx_xvld(p7, 0);
            transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            __lsx_vst(float2bfloat_lasx(_r0), pp, 0);
            __lsx_vst(float2bfloat_lasx(_r1), pp + 8, 0);
            __lsx_vst(float2bfloat_lasx(_r2), pp + 8 * 2, 0);
            __lsx_vst(float2bfloat_lasx(_r3), pp + 8 * 3, 0);
            __lsx_vst(float2bfloat_lasx(_r4), pp + 8 * 4, 0);
            __lsx_vst(float2bfloat_lasx(_r5), pp + 8 * 5, 0);
            __lsx_vst(float2bfloat_lasx(_r6), pp + 8 * 6, 0);
            __lsx_vst(float2bfloat_lasx(_r7), pp + 8 * 7, 0);
            pp += 64;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
            p4 += 8;
            p5 += 8;
            p6 += 8;
            p7 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128 _r0 = (__m128)__lsx_vld(p0, 0);
            __m128 _r1 = (__m128)__lsx_vld(p1, 0);
            __m128 _r2 = (__m128)__lsx_vld(p2, 0);
            __m128 _r3 = (__m128)__lsx_vld(p3, 0);
            __m128 _r4 = (__m128)__lsx_vld(p4, 0);
            __m128 _r5 = (__m128)__lsx_vld(p5, 0);
            __m128 _r6 = (__m128)__lsx_vld(p6, 0);
            __m128 _r7 = (__m128)__lsx_vld(p7, 0);
            transpose4x4_ps(_r0, _r1, _r2, _r3);
            transpose4x4_ps(_r4, _r5, _r6, _r7);
            __m128i _r0_bf16 = float2bfloat_lsx(_r0, _r4);
            __m128i _r1_bf16 = float2bfloat_lsx(_r1, _r5);
            __m128i _r2_bf16 = float2bfloat_lsx(_r2, _r6);
            __m128i _r3_bf16 = float2bfloat_lsx(_r3, _r7);
            __lsx_vst(_r0_bf16, pp, 0);
            __lsx_vst(_r1_bf16, pp + 8, 0);
            __lsx_vst(_r2_bf16, pp + 8 * 2, 0);
            __lsx_vst(_r3_bf16, pp + 8 * 3, 0);
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
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp[2] = float32_to_bfloat16(p2[0]);
            pp[3] = float32_to_bfloat16(p3[0]);
            pp[4] = float32_to_bfloat16(p4[0]);
            pp[5] = float32_to_bfloat16(p5[0]);
            pp[6] = float32_to_bfloat16(p6[0]);
            pp[7] = float32_to_bfloat16(p7[0]);
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
#endif // __loongarch_asx
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128 _r0 = (__m128)__lsx_vld(p0, 0);
            __m128 _r1 = (__m128)__lsx_vld(p1, 0);
            __m128 _r2 = (__m128)__lsx_vld(p2, 0);
            __m128 _r3 = (__m128)__lsx_vld(p3, 0);
            transpose4x4_ps(_r0, _r1, _r2, _r3);
            __lsx_vstelm_d(float2bfloat_lsx(_r0, _r0), pp, 0, 0);
            __lsx_vstelm_d(float2bfloat_lsx(_r1, _r1), pp + 4, 0, 0);
            __lsx_vstelm_d(float2bfloat_lsx(_r2, _r2), pp + 8, 0, 0);
            __lsx_vstelm_d(float2bfloat_lsx(_r3, _r3), pp + 12, 0, 0);
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp[2] = float32_to_bfloat16(p2[0]);
            pp[3] = float32_to_bfloat16(p3[0]);
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
#if __loongarch_sx
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128 _r0 = (__m128)__lsx_vld(p0, 0);
            __m128 _r1 = (__m128)__lsx_vld(p1, 0);
            __m128i _t0 = __lsx_vilvl_w((__m128i)_r1, (__m128i)_r0);
            __m128i _t1 = __lsx_vilvh_w((__m128i)_r1, (__m128i)_r0);
            __lsx_vstelm_d(float2bfloat_lsx((__m128)_t0, (__m128)_t0), pp, 0, 0);
            __lsx_vstelm_d(float2bfloat_lsx((__m128)_t1, (__m128)_t1), pp + 4, 0, 0);
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
#endif // __loongarch_sx
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        int kk = 0;
#if __loongarch_sx
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128 _r0 = (__m128)__lsx_vld(p0, 0);
            __lsx_vstelm_d(float2bfloat_lsx(_r0, _r0), pp, 0, 0);
            pp += 4;
            p0 += 4;
        }
#endif // __loongarch_sx
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp++;
            p0++;
        }
    }
}

static void convolution_gemm_transB_packed_tile_bf16s(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end, int activation_type, const Mat& activation_params)
{
    const int out_elempack = top_blob.elempack;
    const int out_hstep = (int)top_blob.cstep;

    const unsigned short* pAT = AT_tile;
    const unsigned short* pBT = BT_tile;
    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            const unsigned short* pA = pAT;

            __m256 _sum0;
            __m256 _sum1;
            __m256 _sum2;
            __m256 _sum3;
            __m256 _sum4;
            __m256 _sum5;
            __m256 _sum6;
            __m256 _sum7;
            __m256 _sum8;
            __m256 _sum9;
            __m256 _sum10;
            __m256 _sum11;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m256)__lasx_xvld(pC, 0);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                    _sum8 = _sum0;
                    _sum9 = _sum0;
                    _sum10 = _sum0;
                    _sum11 = _sum0;
                }
                else
                {
                    _sum0 = (__m256)__lasx_xvldi(0);
                    _sum1 = (__m256)__lasx_xvldi(0);
                    _sum2 = (__m256)__lasx_xvldi(0);
                    _sum3 = (__m256)__lasx_xvldi(0);
                    _sum4 = (__m256)__lasx_xvldi(0);
                    _sum5 = (__m256)__lasx_xvldi(0);
                    _sum6 = (__m256)__lasx_xvldi(0);
                    _sum7 = (__m256)__lasx_xvldi(0);
                    _sum8 = (__m256)__lasx_xvldi(0);
                    _sum9 = (__m256)__lasx_xvldi(0);
                    _sum10 = (__m256)__lasx_xvldi(0);
                    _sum11 = (__m256)__lasx_xvldi(0);
                }
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr + 0, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
                _sum2 = (__m256)__lasx_xvld(outptr + 16, 0);
                _sum3 = (__m256)__lasx_xvld(outptr + 24, 0);
                _sum4 = (__m256)__lasx_xvld(outptr + 32, 0);
                _sum5 = (__m256)__lasx_xvld(outptr + 40, 0);
                _sum6 = (__m256)__lasx_xvld(outptr + 48, 0);
                _sum7 = (__m256)__lasx_xvld(outptr + 56, 0);
                _sum8 = (__m256)__lasx_xvld(outptr + 64, 0);
                _sum9 = (__m256)__lasx_xvld(outptr + 72, 0);
                _sum10 = (__m256)__lasx_xvld(outptr + 80, 0);
                _sum11 = (__m256)__lasx_xvld(outptr + 88, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = bfloat2float_lasx((__m128i)__lsx_vld(pA, 0));

                _sum0 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[1])), _sum1);
                _sum2 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[2])), _sum2);
                _sum3 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[3])), _sum3);
                _sum4 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[4])), _sum4);
                _sum5 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[5])), _sum5);
                _sum6 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[6])), _sum6);
                _sum7 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[7])), _sum7);
                _sum8 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[8])), _sum8);
                _sum9 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[9])), _sum9);
                _sum10 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[10])), _sum10);
                _sum11 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[11])), _sum11);

                pA += 8;
                pB += 12;
            }

            if (k_end)
            {
                _sum0 = activation_lasx(_sum0, activation_type, activation_params);
                _sum1 = activation_lasx(_sum1, activation_type, activation_params);
                _sum2 = activation_lasx(_sum2, activation_type, activation_params);
                _sum3 = activation_lasx(_sum3, activation_type, activation_params);
                _sum4 = activation_lasx(_sum4, activation_type, activation_params);
                _sum5 = activation_lasx(_sum5, activation_type, activation_params);
                _sum6 = activation_lasx(_sum6, activation_type, activation_params);
                _sum7 = activation_lasx(_sum7, activation_type, activation_params);
                _sum8 = activation_lasx(_sum8, activation_type, activation_params);
                _sum9 = activation_lasx(_sum9, activation_type, activation_params);
                _sum10 = activation_lasx(_sum10, activation_type, activation_params);
                _sum11 = activation_lasx(_sum11, activation_type, activation_params);

                if (out_elempack == 8)
                {
                    __lsx_vst(float2bfloat_lasx(_sum0), outptr0 + 0, 0);
                    __lsx_vst(float2bfloat_lasx(_sum1), outptr0 + 8, 0);
                    __lsx_vst(float2bfloat_lasx(_sum2), outptr0 + 16, 0);
                    __lsx_vst(float2bfloat_lasx(_sum3), outptr0 + 24, 0);
                    __lsx_vst(float2bfloat_lasx(_sum4), outptr0 + 32, 0);
                    __lsx_vst(float2bfloat_lasx(_sum5), outptr0 + 40, 0);
                    __lsx_vst(float2bfloat_lasx(_sum6), outptr0 + 48, 0);
                    __lsx_vst(float2bfloat_lasx(_sum7), outptr0 + 56, 0);
                    __lsx_vst(float2bfloat_lasx(_sum8), outptr0 + 64, 0);
                    __lsx_vst(float2bfloat_lasx(_sum9), outptr0 + 72, 0);
                    __lsx_vst(float2bfloat_lasx(_sum10), outptr0 + 80, 0);
                    __lsx_vst(float2bfloat_lasx(_sum11), outptr0 + 88, 0);
                    outptr0 += 96;
                }
                if (out_elempack == 4)
                {
                    __m128 _sum0_lo = (__m128)__lasx_extract_lo128((__m256i)_sum0);
                    __m128 _sum0_hi = (__m128)__lasx_extract_hi128((__m256i)_sum0);
                    __m128 _sum1_lo = (__m128)__lasx_extract_lo128((__m256i)_sum1);
                    __m128 _sum1_hi = (__m128)__lasx_extract_hi128((__m256i)_sum1);
                    __m128 _sum2_lo = (__m128)__lasx_extract_lo128((__m256i)_sum2);
                    __m128 _sum2_hi = (__m128)__lasx_extract_hi128((__m256i)_sum2);
                    __m128 _sum3_lo = (__m128)__lasx_extract_lo128((__m256i)_sum3);
                    __m128 _sum3_hi = (__m128)__lasx_extract_hi128((__m256i)_sum3);
                    __m128 _sum4_lo = (__m128)__lasx_extract_lo128((__m256i)_sum4);
                    __m128 _sum4_hi = (__m128)__lasx_extract_hi128((__m256i)_sum4);
                    __m128 _sum5_lo = (__m128)__lasx_extract_lo128((__m256i)_sum5);
                    __m128 _sum5_hi = (__m128)__lasx_extract_hi128((__m256i)_sum5);
                    __m128 _sum6_lo = (__m128)__lasx_extract_lo128((__m256i)_sum6);
                    __m128 _sum6_hi = (__m128)__lasx_extract_hi128((__m256i)_sum6);
                    __m128 _sum7_lo = (__m128)__lasx_extract_lo128((__m256i)_sum7);
                    __m128 _sum7_hi = (__m128)__lasx_extract_hi128((__m256i)_sum7);
                    __m128 _sum8_lo = (__m128)__lasx_extract_lo128((__m256i)_sum8);
                    __m128 _sum8_hi = (__m128)__lasx_extract_hi128((__m256i)_sum8);
                    __m128 _sum9_lo = (__m128)__lasx_extract_lo128((__m256i)_sum9);
                    __m128 _sum9_hi = (__m128)__lasx_extract_hi128((__m256i)_sum9);
                    __m128 _sum10_lo = (__m128)__lasx_extract_lo128((__m256i)_sum10);
                    __m128 _sum10_hi = (__m128)__lasx_extract_hi128((__m256i)_sum10);
                    __m128 _sum11_lo = (__m128)__lasx_extract_lo128((__m256i)_sum11);
                    __m128 _sum11_hi = (__m128)__lasx_extract_hi128((__m256i)_sum11);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum0_lo, _sum0_lo), outptr0 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum1_lo, _sum1_lo), outptr0 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum2_lo, _sum2_lo), outptr0 + 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum3_lo, _sum3_lo), outptr0 + 12, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum4_lo, _sum4_lo), outptr0 + 16, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum5_lo, _sum5_lo), outptr0 + 20, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum6_lo, _sum6_lo), outptr0 + 24, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum7_lo, _sum7_lo), outptr0 + 28, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum8_lo, _sum8_lo), outptr0 + 32, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum9_lo, _sum9_lo), outptr0 + 36, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum10_lo, _sum10_lo), outptr0 + 40, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum11_lo, _sum11_lo), outptr0 + 44, 0, 0);
                    outptr0 += 48;
                    __lsx_vstelm_d(float2bfloat_lsx(_sum0_hi, _sum0_hi), outptr0 + out_hstep * 4 - 48 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum1_hi, _sum1_hi), outptr0 + out_hstep * 4 - 48 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum2_hi, _sum2_hi), outptr0 + out_hstep * 4 - 48 + 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum3_hi, _sum3_hi), outptr0 + out_hstep * 4 - 48 + 12, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum4_hi, _sum4_hi), outptr0 + out_hstep * 4 - 48 + 16, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum5_hi, _sum5_hi), outptr0 + out_hstep * 4 - 48 + 20, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum6_hi, _sum6_hi), outptr0 + out_hstep * 4 - 48 + 24, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum7_hi, _sum7_hi), outptr0 + out_hstep * 4 - 48 + 28, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum8_hi, _sum8_hi), outptr0 + out_hstep * 4 - 48 + 32, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum9_hi, _sum9_hi), outptr0 + out_hstep * 4 - 48 + 36, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum10_hi, _sum10_hi), outptr0 + out_hstep * 4 - 48 + 40, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum11_hi, _sum11_hi), outptr0 + out_hstep * 4 - 48 + 44, 0, 0);
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    __lasx_xvst((__m256i)_sum0, sum0, 0);
                    float sum1[8];
                    __lasx_xvst((__m256i)_sum1, sum1, 0);
                    float sum2[8];
                    __lasx_xvst((__m256i)_sum2, sum2, 0);
                    float sum3[8];
                    __lasx_xvst((__m256i)_sum3, sum3, 0);
                    float sum4[8];
                    __lasx_xvst((__m256i)_sum4, sum4, 0);
                    float sum5[8];
                    __lasx_xvst((__m256i)_sum5, sum5, 0);
                    float sum6[8];
                    __lasx_xvst((__m256i)_sum6, sum6, 0);
                    float sum7[8];
                    __lasx_xvst((__m256i)_sum7, sum7, 0);
                    float sum8[8];
                    __lasx_xvst((__m256i)_sum8, sum8, 0);
                    float sum9[8];
                    __lasx_xvst((__m256i)_sum9, sum9, 0);
                    float sum10[8];
                    __lasx_xvst((__m256i)_sum10, sum10, 0);
                    float sum11[8];
                    __lasx_xvst((__m256i)_sum11, sum11, 0);
                    outptr0[out_hstep * 0 + 0] = float32_to_bfloat16(sum0[0]);
                    outptr0[out_hstep * 0 + 1] = float32_to_bfloat16(sum1[0]);
                    outptr0[out_hstep * 0 + 2] = float32_to_bfloat16(sum2[0]);
                    outptr0[out_hstep * 0 + 3] = float32_to_bfloat16(sum3[0]);
                    outptr0[out_hstep * 0 + 4] = float32_to_bfloat16(sum4[0]);
                    outptr0[out_hstep * 0 + 5] = float32_to_bfloat16(sum5[0]);
                    outptr0[out_hstep * 0 + 6] = float32_to_bfloat16(sum6[0]);
                    outptr0[out_hstep * 0 + 7] = float32_to_bfloat16(sum7[0]);
                    outptr0[out_hstep * 0 + 8] = float32_to_bfloat16(sum8[0]);
                    outptr0[out_hstep * 0 + 9] = float32_to_bfloat16(sum9[0]);
                    outptr0[out_hstep * 0 + 10] = float32_to_bfloat16(sum10[0]);
                    outptr0[out_hstep * 0 + 11] = float32_to_bfloat16(sum11[0]);
                    outptr0[out_hstep * 1 + 0] = float32_to_bfloat16(sum0[1]);
                    outptr0[out_hstep * 1 + 1] = float32_to_bfloat16(sum1[1]);
                    outptr0[out_hstep * 1 + 2] = float32_to_bfloat16(sum2[1]);
                    outptr0[out_hstep * 1 + 3] = float32_to_bfloat16(sum3[1]);
                    outptr0[out_hstep * 1 + 4] = float32_to_bfloat16(sum4[1]);
                    outptr0[out_hstep * 1 + 5] = float32_to_bfloat16(sum5[1]);
                    outptr0[out_hstep * 1 + 6] = float32_to_bfloat16(sum6[1]);
                    outptr0[out_hstep * 1 + 7] = float32_to_bfloat16(sum7[1]);
                    outptr0[out_hstep * 1 + 8] = float32_to_bfloat16(sum8[1]);
                    outptr0[out_hstep * 1 + 9] = float32_to_bfloat16(sum9[1]);
                    outptr0[out_hstep * 1 + 10] = float32_to_bfloat16(sum10[1]);
                    outptr0[out_hstep * 1 + 11] = float32_to_bfloat16(sum11[1]);
                    outptr0[out_hstep * 2 + 0] = float32_to_bfloat16(sum0[2]);
                    outptr0[out_hstep * 2 + 1] = float32_to_bfloat16(sum1[2]);
                    outptr0[out_hstep * 2 + 2] = float32_to_bfloat16(sum2[2]);
                    outptr0[out_hstep * 2 + 3] = float32_to_bfloat16(sum3[2]);
                    outptr0[out_hstep * 2 + 4] = float32_to_bfloat16(sum4[2]);
                    outptr0[out_hstep * 2 + 5] = float32_to_bfloat16(sum5[2]);
                    outptr0[out_hstep * 2 + 6] = float32_to_bfloat16(sum6[2]);
                    outptr0[out_hstep * 2 + 7] = float32_to_bfloat16(sum7[2]);
                    outptr0[out_hstep * 2 + 8] = float32_to_bfloat16(sum8[2]);
                    outptr0[out_hstep * 2 + 9] = float32_to_bfloat16(sum9[2]);
                    outptr0[out_hstep * 2 + 10] = float32_to_bfloat16(sum10[2]);
                    outptr0[out_hstep * 2 + 11] = float32_to_bfloat16(sum11[2]);
                    outptr0[out_hstep * 3 + 0] = float32_to_bfloat16(sum0[3]);
                    outptr0[out_hstep * 3 + 1] = float32_to_bfloat16(sum1[3]);
                    outptr0[out_hstep * 3 + 2] = float32_to_bfloat16(sum2[3]);
                    outptr0[out_hstep * 3 + 3] = float32_to_bfloat16(sum3[3]);
                    outptr0[out_hstep * 3 + 4] = float32_to_bfloat16(sum4[3]);
                    outptr0[out_hstep * 3 + 5] = float32_to_bfloat16(sum5[3]);
                    outptr0[out_hstep * 3 + 6] = float32_to_bfloat16(sum6[3]);
                    outptr0[out_hstep * 3 + 7] = float32_to_bfloat16(sum7[3]);
                    outptr0[out_hstep * 3 + 8] = float32_to_bfloat16(sum8[3]);
                    outptr0[out_hstep * 3 + 9] = float32_to_bfloat16(sum9[3]);
                    outptr0[out_hstep * 3 + 10] = float32_to_bfloat16(sum10[3]);
                    outptr0[out_hstep * 3 + 11] = float32_to_bfloat16(sum11[3]);
                    outptr0[out_hstep * 4 + 0] = float32_to_bfloat16(sum0[4]);
                    outptr0[out_hstep * 4 + 1] = float32_to_bfloat16(sum1[4]);
                    outptr0[out_hstep * 4 + 2] = float32_to_bfloat16(sum2[4]);
                    outptr0[out_hstep * 4 + 3] = float32_to_bfloat16(sum3[4]);
                    outptr0[out_hstep * 4 + 4] = float32_to_bfloat16(sum4[4]);
                    outptr0[out_hstep * 4 + 5] = float32_to_bfloat16(sum5[4]);
                    outptr0[out_hstep * 4 + 6] = float32_to_bfloat16(sum6[4]);
                    outptr0[out_hstep * 4 + 7] = float32_to_bfloat16(sum7[4]);
                    outptr0[out_hstep * 4 + 8] = float32_to_bfloat16(sum8[4]);
                    outptr0[out_hstep * 4 + 9] = float32_to_bfloat16(sum9[4]);
                    outptr0[out_hstep * 4 + 10] = float32_to_bfloat16(sum10[4]);
                    outptr0[out_hstep * 4 + 11] = float32_to_bfloat16(sum11[4]);
                    outptr0[out_hstep * 5 + 0] = float32_to_bfloat16(sum0[5]);
                    outptr0[out_hstep * 5 + 1] = float32_to_bfloat16(sum1[5]);
                    outptr0[out_hstep * 5 + 2] = float32_to_bfloat16(sum2[5]);
                    outptr0[out_hstep * 5 + 3] = float32_to_bfloat16(sum3[5]);
                    outptr0[out_hstep * 5 + 4] = float32_to_bfloat16(sum4[5]);
                    outptr0[out_hstep * 5 + 5] = float32_to_bfloat16(sum5[5]);
                    outptr0[out_hstep * 5 + 6] = float32_to_bfloat16(sum6[5]);
                    outptr0[out_hstep * 5 + 7] = float32_to_bfloat16(sum7[5]);
                    outptr0[out_hstep * 5 + 8] = float32_to_bfloat16(sum8[5]);
                    outptr0[out_hstep * 5 + 9] = float32_to_bfloat16(sum9[5]);
                    outptr0[out_hstep * 5 + 10] = float32_to_bfloat16(sum10[5]);
                    outptr0[out_hstep * 5 + 11] = float32_to_bfloat16(sum11[5]);
                    outptr0[out_hstep * 6 + 0] = float32_to_bfloat16(sum0[6]);
                    outptr0[out_hstep * 6 + 1] = float32_to_bfloat16(sum1[6]);
                    outptr0[out_hstep * 6 + 2] = float32_to_bfloat16(sum2[6]);
                    outptr0[out_hstep * 6 + 3] = float32_to_bfloat16(sum3[6]);
                    outptr0[out_hstep * 6 + 4] = float32_to_bfloat16(sum4[6]);
                    outptr0[out_hstep * 6 + 5] = float32_to_bfloat16(sum5[6]);
                    outptr0[out_hstep * 6 + 6] = float32_to_bfloat16(sum6[6]);
                    outptr0[out_hstep * 6 + 7] = float32_to_bfloat16(sum7[6]);
                    outptr0[out_hstep * 6 + 8] = float32_to_bfloat16(sum8[6]);
                    outptr0[out_hstep * 6 + 9] = float32_to_bfloat16(sum9[6]);
                    outptr0[out_hstep * 6 + 10] = float32_to_bfloat16(sum10[6]);
                    outptr0[out_hstep * 6 + 11] = float32_to_bfloat16(sum11[6]);
                    outptr0[out_hstep * 7 + 0] = float32_to_bfloat16(sum0[7]);
                    outptr0[out_hstep * 7 + 1] = float32_to_bfloat16(sum1[7]);
                    outptr0[out_hstep * 7 + 2] = float32_to_bfloat16(sum2[7]);
                    outptr0[out_hstep * 7 + 3] = float32_to_bfloat16(sum3[7]);
                    outptr0[out_hstep * 7 + 4] = float32_to_bfloat16(sum4[7]);
                    outptr0[out_hstep * 7 + 5] = float32_to_bfloat16(sum5[7]);
                    outptr0[out_hstep * 7 + 6] = float32_to_bfloat16(sum6[7]);
                    outptr0[out_hstep * 7 + 7] = float32_to_bfloat16(sum7[7]);
                    outptr0[out_hstep * 7 + 8] = float32_to_bfloat16(sum8[7]);
                    outptr0[out_hstep * 7 + 9] = float32_to_bfloat16(sum9[7]);
                    outptr0[out_hstep * 7 + 10] = float32_to_bfloat16(sum10[7]);
                    outptr0[out_hstep * 7 + 11] = float32_to_bfloat16(sum11[7]);
                    outptr0 += 12;
                }
            }
            else
            {
                __lasx_xvst((__m256i)_sum0, outptr + 0, 0);
                __lasx_xvst((__m256i)_sum1, outptr + 8, 0);
                __lasx_xvst((__m256i)_sum2, outptr + 16, 0);
                __lasx_xvst((__m256i)_sum3, outptr + 24, 0);
                __lasx_xvst((__m256i)_sum4, outptr + 32, 0);
                __lasx_xvst((__m256i)_sum5, outptr + 40, 0);
                __lasx_xvst((__m256i)_sum6, outptr + 48, 0);
                __lasx_xvst((__m256i)_sum7, outptr + 56, 0);
                __lasx_xvst((__m256i)_sum8, outptr + 64, 0);
                __lasx_xvst((__m256i)_sum9, outptr + 72, 0);
                __lasx_xvst((__m256i)_sum10, outptr + 80, 0);
                __lasx_xvst((__m256i)_sum11, outptr + 88, 0);
            }

            outptr += 96;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

            __m256 _sum0;
            __m256 _sum1;
            __m256 _sum2;
            __m256 _sum3;
            __m256 _sum4;
            __m256 _sum5;
            __m256 _sum6;
            __m256 _sum7;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m256)__lasx_xvld(pC, 0);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                }
                else
                {
                    _sum0 = (__m256)__lasx_xvldi(0);
                    _sum1 = (__m256)__lasx_xvldi(0);
                    _sum2 = (__m256)__lasx_xvldi(0);
                    _sum3 = (__m256)__lasx_xvldi(0);
                    _sum4 = (__m256)__lasx_xvldi(0);
                    _sum5 = (__m256)__lasx_xvldi(0);
                    _sum6 = (__m256)__lasx_xvldi(0);
                    _sum7 = (__m256)__lasx_xvldi(0);
                }
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr + 0, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
                _sum2 = (__m256)__lasx_xvld(outptr + 16, 0);
                _sum3 = (__m256)__lasx_xvld(outptr + 24, 0);
                _sum4 = (__m256)__lasx_xvld(outptr + 32, 0);
                _sum5 = (__m256)__lasx_xvld(outptr + 40, 0);
                _sum6 = (__m256)__lasx_xvld(outptr + 48, 0);
                _sum7 = (__m256)__lasx_xvld(outptr + 56, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = bfloat2float_lasx((__m128i)__lsx_vld(pA, 0));

                _sum0 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[1])), _sum1);
                _sum2 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[2])), _sum2);
                _sum3 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[3])), _sum3);
                _sum4 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[4])), _sum4);
                _sum5 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[5])), _sum5);
                _sum6 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[6])), _sum6);
                _sum7 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[7])), _sum7);

                pA += 8;
                pB += 8;
            }

            if (k_end)
            {
                _sum0 = activation_lasx(_sum0, activation_type, activation_params);
                _sum1 = activation_lasx(_sum1, activation_type, activation_params);
                _sum2 = activation_lasx(_sum2, activation_type, activation_params);
                _sum3 = activation_lasx(_sum3, activation_type, activation_params);
                _sum4 = activation_lasx(_sum4, activation_type, activation_params);
                _sum5 = activation_lasx(_sum5, activation_type, activation_params);
                _sum6 = activation_lasx(_sum6, activation_type, activation_params);
                _sum7 = activation_lasx(_sum7, activation_type, activation_params);

                if (out_elempack == 8)
                {
                    __lsx_vst(float2bfloat_lasx(_sum0), outptr0 + 0, 0);
                    __lsx_vst(float2bfloat_lasx(_sum1), outptr0 + 8, 0);
                    __lsx_vst(float2bfloat_lasx(_sum2), outptr0 + 16, 0);
                    __lsx_vst(float2bfloat_lasx(_sum3), outptr0 + 24, 0);
                    __lsx_vst(float2bfloat_lasx(_sum4), outptr0 + 32, 0);
                    __lsx_vst(float2bfloat_lasx(_sum5), outptr0 + 40, 0);
                    __lsx_vst(float2bfloat_lasx(_sum6), outptr0 + 48, 0);
                    __lsx_vst(float2bfloat_lasx(_sum7), outptr0 + 56, 0);
                    outptr0 += 64;
                }
                if (out_elempack == 4)
                {
                    __m128 _sum0_lo = (__m128)__lasx_extract_lo128((__m256i)_sum0);
                    __m128 _sum0_hi = (__m128)__lasx_extract_hi128((__m256i)_sum0);
                    __m128 _sum1_lo = (__m128)__lasx_extract_lo128((__m256i)_sum1);
                    __m128 _sum1_hi = (__m128)__lasx_extract_hi128((__m256i)_sum1);
                    __m128 _sum2_lo = (__m128)__lasx_extract_lo128((__m256i)_sum2);
                    __m128 _sum2_hi = (__m128)__lasx_extract_hi128((__m256i)_sum2);
                    __m128 _sum3_lo = (__m128)__lasx_extract_lo128((__m256i)_sum3);
                    __m128 _sum3_hi = (__m128)__lasx_extract_hi128((__m256i)_sum3);
                    __m128 _sum4_lo = (__m128)__lasx_extract_lo128((__m256i)_sum4);
                    __m128 _sum4_hi = (__m128)__lasx_extract_hi128((__m256i)_sum4);
                    __m128 _sum5_lo = (__m128)__lasx_extract_lo128((__m256i)_sum5);
                    __m128 _sum5_hi = (__m128)__lasx_extract_hi128((__m256i)_sum5);
                    __m128 _sum6_lo = (__m128)__lasx_extract_lo128((__m256i)_sum6);
                    __m128 _sum6_hi = (__m128)__lasx_extract_hi128((__m256i)_sum6);
                    __m128 _sum7_lo = (__m128)__lasx_extract_lo128((__m256i)_sum7);
                    __m128 _sum7_hi = (__m128)__lasx_extract_hi128((__m256i)_sum7);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum0_lo, _sum0_lo), outptr0 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum1_lo, _sum1_lo), outptr0 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum2_lo, _sum2_lo), outptr0 + 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum3_lo, _sum3_lo), outptr0 + 12, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum4_lo, _sum4_lo), outptr0 + 16, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum5_lo, _sum5_lo), outptr0 + 20, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum6_lo, _sum6_lo), outptr0 + 24, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum7_lo, _sum7_lo), outptr0 + 28, 0, 0);
                    outptr0 += 32;
                    __lsx_vstelm_d(float2bfloat_lsx(_sum0_hi, _sum0_hi), outptr0 + out_hstep * 4 - 32 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum1_hi, _sum1_hi), outptr0 + out_hstep * 4 - 32 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum2_hi, _sum2_hi), outptr0 + out_hstep * 4 - 32 + 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum3_hi, _sum3_hi), outptr0 + out_hstep * 4 - 32 + 12, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum4_hi, _sum4_hi), outptr0 + out_hstep * 4 - 32 + 16, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum5_hi, _sum5_hi), outptr0 + out_hstep * 4 - 32 + 20, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum6_hi, _sum6_hi), outptr0 + out_hstep * 4 - 32 + 24, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum7_hi, _sum7_hi), outptr0 + out_hstep * 4 - 32 + 28, 0, 0);
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    __lasx_xvst((__m256i)_sum0, sum0, 0);
                    float sum1[8];
                    __lasx_xvst((__m256i)_sum1, sum1, 0);
                    float sum2[8];
                    __lasx_xvst((__m256i)_sum2, sum2, 0);
                    float sum3[8];
                    __lasx_xvst((__m256i)_sum3, sum3, 0);
                    float sum4[8];
                    __lasx_xvst((__m256i)_sum4, sum4, 0);
                    float sum5[8];
                    __lasx_xvst((__m256i)_sum5, sum5, 0);
                    float sum6[8];
                    __lasx_xvst((__m256i)_sum6, sum6, 0);
                    float sum7[8];
                    __lasx_xvst((__m256i)_sum7, sum7, 0);
                    outptr0[out_hstep * 0 + 0] = float32_to_bfloat16(sum0[0]);
                    outptr0[out_hstep * 0 + 1] = float32_to_bfloat16(sum1[0]);
                    outptr0[out_hstep * 0 + 2] = float32_to_bfloat16(sum2[0]);
                    outptr0[out_hstep * 0 + 3] = float32_to_bfloat16(sum3[0]);
                    outptr0[out_hstep * 0 + 4] = float32_to_bfloat16(sum4[0]);
                    outptr0[out_hstep * 0 + 5] = float32_to_bfloat16(sum5[0]);
                    outptr0[out_hstep * 0 + 6] = float32_to_bfloat16(sum6[0]);
                    outptr0[out_hstep * 0 + 7] = float32_to_bfloat16(sum7[0]);
                    outptr0[out_hstep * 1 + 0] = float32_to_bfloat16(sum0[1]);
                    outptr0[out_hstep * 1 + 1] = float32_to_bfloat16(sum1[1]);
                    outptr0[out_hstep * 1 + 2] = float32_to_bfloat16(sum2[1]);
                    outptr0[out_hstep * 1 + 3] = float32_to_bfloat16(sum3[1]);
                    outptr0[out_hstep * 1 + 4] = float32_to_bfloat16(sum4[1]);
                    outptr0[out_hstep * 1 + 5] = float32_to_bfloat16(sum5[1]);
                    outptr0[out_hstep * 1 + 6] = float32_to_bfloat16(sum6[1]);
                    outptr0[out_hstep * 1 + 7] = float32_to_bfloat16(sum7[1]);
                    outptr0[out_hstep * 2 + 0] = float32_to_bfloat16(sum0[2]);
                    outptr0[out_hstep * 2 + 1] = float32_to_bfloat16(sum1[2]);
                    outptr0[out_hstep * 2 + 2] = float32_to_bfloat16(sum2[2]);
                    outptr0[out_hstep * 2 + 3] = float32_to_bfloat16(sum3[2]);
                    outptr0[out_hstep * 2 + 4] = float32_to_bfloat16(sum4[2]);
                    outptr0[out_hstep * 2 + 5] = float32_to_bfloat16(sum5[2]);
                    outptr0[out_hstep * 2 + 6] = float32_to_bfloat16(sum6[2]);
                    outptr0[out_hstep * 2 + 7] = float32_to_bfloat16(sum7[2]);
                    outptr0[out_hstep * 3 + 0] = float32_to_bfloat16(sum0[3]);
                    outptr0[out_hstep * 3 + 1] = float32_to_bfloat16(sum1[3]);
                    outptr0[out_hstep * 3 + 2] = float32_to_bfloat16(sum2[3]);
                    outptr0[out_hstep * 3 + 3] = float32_to_bfloat16(sum3[3]);
                    outptr0[out_hstep * 3 + 4] = float32_to_bfloat16(sum4[3]);
                    outptr0[out_hstep * 3 + 5] = float32_to_bfloat16(sum5[3]);
                    outptr0[out_hstep * 3 + 6] = float32_to_bfloat16(sum6[3]);
                    outptr0[out_hstep * 3 + 7] = float32_to_bfloat16(sum7[3]);
                    outptr0[out_hstep * 4 + 0] = float32_to_bfloat16(sum0[4]);
                    outptr0[out_hstep * 4 + 1] = float32_to_bfloat16(sum1[4]);
                    outptr0[out_hstep * 4 + 2] = float32_to_bfloat16(sum2[4]);
                    outptr0[out_hstep * 4 + 3] = float32_to_bfloat16(sum3[4]);
                    outptr0[out_hstep * 4 + 4] = float32_to_bfloat16(sum4[4]);
                    outptr0[out_hstep * 4 + 5] = float32_to_bfloat16(sum5[4]);
                    outptr0[out_hstep * 4 + 6] = float32_to_bfloat16(sum6[4]);
                    outptr0[out_hstep * 4 + 7] = float32_to_bfloat16(sum7[4]);
                    outptr0[out_hstep * 5 + 0] = float32_to_bfloat16(sum0[5]);
                    outptr0[out_hstep * 5 + 1] = float32_to_bfloat16(sum1[5]);
                    outptr0[out_hstep * 5 + 2] = float32_to_bfloat16(sum2[5]);
                    outptr0[out_hstep * 5 + 3] = float32_to_bfloat16(sum3[5]);
                    outptr0[out_hstep * 5 + 4] = float32_to_bfloat16(sum4[5]);
                    outptr0[out_hstep * 5 + 5] = float32_to_bfloat16(sum5[5]);
                    outptr0[out_hstep * 5 + 6] = float32_to_bfloat16(sum6[5]);
                    outptr0[out_hstep * 5 + 7] = float32_to_bfloat16(sum7[5]);
                    outptr0[out_hstep * 6 + 0] = float32_to_bfloat16(sum0[6]);
                    outptr0[out_hstep * 6 + 1] = float32_to_bfloat16(sum1[6]);
                    outptr0[out_hstep * 6 + 2] = float32_to_bfloat16(sum2[6]);
                    outptr0[out_hstep * 6 + 3] = float32_to_bfloat16(sum3[6]);
                    outptr0[out_hstep * 6 + 4] = float32_to_bfloat16(sum4[6]);
                    outptr0[out_hstep * 6 + 5] = float32_to_bfloat16(sum5[6]);
                    outptr0[out_hstep * 6 + 6] = float32_to_bfloat16(sum6[6]);
                    outptr0[out_hstep * 6 + 7] = float32_to_bfloat16(sum7[6]);
                    outptr0[out_hstep * 7 + 0] = float32_to_bfloat16(sum0[7]);
                    outptr0[out_hstep * 7 + 1] = float32_to_bfloat16(sum1[7]);
                    outptr0[out_hstep * 7 + 2] = float32_to_bfloat16(sum2[7]);
                    outptr0[out_hstep * 7 + 3] = float32_to_bfloat16(sum3[7]);
                    outptr0[out_hstep * 7 + 4] = float32_to_bfloat16(sum4[7]);
                    outptr0[out_hstep * 7 + 5] = float32_to_bfloat16(sum5[7]);
                    outptr0[out_hstep * 7 + 6] = float32_to_bfloat16(sum6[7]);
                    outptr0[out_hstep * 7 + 7] = float32_to_bfloat16(sum7[7]);
                    outptr0 += 8;
                }
            }
            else
            {
                __lasx_xvst((__m256i)_sum0, outptr + 0, 0);
                __lasx_xvst((__m256i)_sum1, outptr + 8, 0);
                __lasx_xvst((__m256i)_sum2, outptr + 16, 0);
                __lasx_xvst((__m256i)_sum3, outptr + 24, 0);
                __lasx_xvst((__m256i)_sum4, outptr + 32, 0);
                __lasx_xvst((__m256i)_sum5, outptr + 40, 0);
                __lasx_xvst((__m256i)_sum6, outptr + 48, 0);
                __lasx_xvst((__m256i)_sum7, outptr + 56, 0);
            }

            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

            __m256 _sum0;
            __m256 _sum1;
            __m256 _sum2;
            __m256 _sum3;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m256)__lasx_xvld(pC, 0);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
                else
                {
                    _sum0 = (__m256)__lasx_xvldi(0);
                    _sum1 = (__m256)__lasx_xvldi(0);
                    _sum2 = (__m256)__lasx_xvldi(0);
                    _sum3 = (__m256)__lasx_xvldi(0);
                }
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr + 0, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
                _sum2 = (__m256)__lasx_xvld(outptr + 16, 0);
                _sum3 = (__m256)__lasx_xvld(outptr + 24, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = bfloat2float_lasx((__m128i)__lsx_vld(pA, 0));

                _sum0 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[1])), _sum1);
                _sum2 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[2])), _sum2);
                _sum3 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[3])), _sum3);

                pA += 8;
                pB += 4;
            }

            if (k_end)
            {
                _sum0 = activation_lasx(_sum0, activation_type, activation_params);
                _sum1 = activation_lasx(_sum1, activation_type, activation_params);
                _sum2 = activation_lasx(_sum2, activation_type, activation_params);
                _sum3 = activation_lasx(_sum3, activation_type, activation_params);

                if (out_elempack == 8)
                {
                    __lsx_vst(float2bfloat_lasx(_sum0), outptr0 + 0, 0);
                    __lsx_vst(float2bfloat_lasx(_sum1), outptr0 + 8, 0);
                    __lsx_vst(float2bfloat_lasx(_sum2), outptr0 + 16, 0);
                    __lsx_vst(float2bfloat_lasx(_sum3), outptr0 + 24, 0);
                    outptr0 += 32;
                }
                if (out_elempack == 4)
                {
                    __m128 _sum0_lo = (__m128)__lasx_extract_lo128((__m256i)_sum0);
                    __m128 _sum0_hi = (__m128)__lasx_extract_hi128((__m256i)_sum0);
                    __m128 _sum1_lo = (__m128)__lasx_extract_lo128((__m256i)_sum1);
                    __m128 _sum1_hi = (__m128)__lasx_extract_hi128((__m256i)_sum1);
                    __m128 _sum2_lo = (__m128)__lasx_extract_lo128((__m256i)_sum2);
                    __m128 _sum2_hi = (__m128)__lasx_extract_hi128((__m256i)_sum2);
                    __m128 _sum3_lo = (__m128)__lasx_extract_lo128((__m256i)_sum3);
                    __m128 _sum3_hi = (__m128)__lasx_extract_hi128((__m256i)_sum3);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum0_lo, _sum0_lo), outptr0 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum1_lo, _sum1_lo), outptr0 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum2_lo, _sum2_lo), outptr0 + 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum3_lo, _sum3_lo), outptr0 + 12, 0, 0);
                    outptr0 += 16;
                    __lsx_vstelm_d(float2bfloat_lsx(_sum0_hi, _sum0_hi), outptr0 + out_hstep * 4 - 16 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum1_hi, _sum1_hi), outptr0 + out_hstep * 4 - 16 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum2_hi, _sum2_hi), outptr0 + out_hstep * 4 - 16 + 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum3_hi, _sum3_hi), outptr0 + out_hstep * 4 - 16 + 12, 0, 0);
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    __lasx_xvst((__m256i)_sum0, sum0, 0);
                    float sum1[8];
                    __lasx_xvst((__m256i)_sum1, sum1, 0);
                    float sum2[8];
                    __lasx_xvst((__m256i)_sum2, sum2, 0);
                    float sum3[8];
                    __lasx_xvst((__m256i)_sum3, sum3, 0);
                    outptr0[out_hstep * 0 + 0] = float32_to_bfloat16(sum0[0]);
                    outptr0[out_hstep * 0 + 1] = float32_to_bfloat16(sum1[0]);
                    outptr0[out_hstep * 0 + 2] = float32_to_bfloat16(sum2[0]);
                    outptr0[out_hstep * 0 + 3] = float32_to_bfloat16(sum3[0]);
                    outptr0[out_hstep * 1 + 0] = float32_to_bfloat16(sum0[1]);
                    outptr0[out_hstep * 1 + 1] = float32_to_bfloat16(sum1[1]);
                    outptr0[out_hstep * 1 + 2] = float32_to_bfloat16(sum2[1]);
                    outptr0[out_hstep * 1 + 3] = float32_to_bfloat16(sum3[1]);
                    outptr0[out_hstep * 2 + 0] = float32_to_bfloat16(sum0[2]);
                    outptr0[out_hstep * 2 + 1] = float32_to_bfloat16(sum1[2]);
                    outptr0[out_hstep * 2 + 2] = float32_to_bfloat16(sum2[2]);
                    outptr0[out_hstep * 2 + 3] = float32_to_bfloat16(sum3[2]);
                    outptr0[out_hstep * 3 + 0] = float32_to_bfloat16(sum0[3]);
                    outptr0[out_hstep * 3 + 1] = float32_to_bfloat16(sum1[3]);
                    outptr0[out_hstep * 3 + 2] = float32_to_bfloat16(sum2[3]);
                    outptr0[out_hstep * 3 + 3] = float32_to_bfloat16(sum3[3]);
                    outptr0[out_hstep * 4 + 0] = float32_to_bfloat16(sum0[4]);
                    outptr0[out_hstep * 4 + 1] = float32_to_bfloat16(sum1[4]);
                    outptr0[out_hstep * 4 + 2] = float32_to_bfloat16(sum2[4]);
                    outptr0[out_hstep * 4 + 3] = float32_to_bfloat16(sum3[4]);
                    outptr0[out_hstep * 5 + 0] = float32_to_bfloat16(sum0[5]);
                    outptr0[out_hstep * 5 + 1] = float32_to_bfloat16(sum1[5]);
                    outptr0[out_hstep * 5 + 2] = float32_to_bfloat16(sum2[5]);
                    outptr0[out_hstep * 5 + 3] = float32_to_bfloat16(sum3[5]);
                    outptr0[out_hstep * 6 + 0] = float32_to_bfloat16(sum0[6]);
                    outptr0[out_hstep * 6 + 1] = float32_to_bfloat16(sum1[6]);
                    outptr0[out_hstep * 6 + 2] = float32_to_bfloat16(sum2[6]);
                    outptr0[out_hstep * 6 + 3] = float32_to_bfloat16(sum3[6]);
                    outptr0[out_hstep * 7 + 0] = float32_to_bfloat16(sum0[7]);
                    outptr0[out_hstep * 7 + 1] = float32_to_bfloat16(sum1[7]);
                    outptr0[out_hstep * 7 + 2] = float32_to_bfloat16(sum2[7]);
                    outptr0[out_hstep * 7 + 3] = float32_to_bfloat16(sum3[7]);
                    outptr0 += 4;
                }
            }
            else
            {
                __lasx_xvst((__m256i)_sum0, outptr + 0, 0);
                __lasx_xvst((__m256i)_sum1, outptr + 8, 0);
                __lasx_xvst((__m256i)_sum2, outptr + 16, 0);
                __lasx_xvst((__m256i)_sum3, outptr + 24, 0);
            }

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* pA = pAT;

            __m256 _sum0;
            __m256 _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m256)__lasx_xvld(pC, 0);
                    _sum1 = _sum0;
                }
                else
                {
                    _sum0 = (__m256)__lasx_xvldi(0);
                    _sum1 = (__m256)__lasx_xvldi(0);
                }
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr + 0, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = bfloat2float_lasx((__m128i)__lsx_vld(pA, 0));

                _sum0 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[1])), _sum1);

                pA += 8;
                pB += 2;
            }

            if (k_end)
            {
                _sum0 = activation_lasx(_sum0, activation_type, activation_params);
                _sum1 = activation_lasx(_sum1, activation_type, activation_params);

                if (out_elempack == 8)
                {
                    __lsx_vst(float2bfloat_lasx(_sum0), outptr0 + 0, 0);
                    __lsx_vst(float2bfloat_lasx(_sum1), outptr0 + 8, 0);
                    outptr0 += 16;
                }
                if (out_elempack == 4)
                {
                    __m128 _sum0_lo = (__m128)__lasx_extract_lo128((__m256i)_sum0);
                    __m128 _sum0_hi = (__m128)__lasx_extract_hi128((__m256i)_sum0);
                    __m128 _sum1_lo = (__m128)__lasx_extract_lo128((__m256i)_sum1);
                    __m128 _sum1_hi = (__m128)__lasx_extract_hi128((__m256i)_sum1);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum0_lo, _sum0_lo), outptr0 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum1_lo, _sum1_lo), outptr0 + 4, 0, 0);
                    outptr0 += 8;
                    __lsx_vstelm_d(float2bfloat_lsx(_sum0_hi, _sum0_hi), outptr0 + out_hstep * 4 - 8 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum1_hi, _sum1_hi), outptr0 + out_hstep * 4 - 8 + 4, 0, 0);
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    __lasx_xvst((__m256i)_sum0, sum0, 0);
                    float sum1[8];
                    __lasx_xvst((__m256i)_sum1, sum1, 0);
                    outptr0[out_hstep * 0 + 0] = float32_to_bfloat16(sum0[0]);
                    outptr0[out_hstep * 0 + 1] = float32_to_bfloat16(sum1[0]);
                    outptr0[out_hstep * 1 + 0] = float32_to_bfloat16(sum0[1]);
                    outptr0[out_hstep * 1 + 1] = float32_to_bfloat16(sum1[1]);
                    outptr0[out_hstep * 2 + 0] = float32_to_bfloat16(sum0[2]);
                    outptr0[out_hstep * 2 + 1] = float32_to_bfloat16(sum1[2]);
                    outptr0[out_hstep * 3 + 0] = float32_to_bfloat16(sum0[3]);
                    outptr0[out_hstep * 3 + 1] = float32_to_bfloat16(sum1[3]);
                    outptr0[out_hstep * 4 + 0] = float32_to_bfloat16(sum0[4]);
                    outptr0[out_hstep * 4 + 1] = float32_to_bfloat16(sum1[4]);
                    outptr0[out_hstep * 5 + 0] = float32_to_bfloat16(sum0[5]);
                    outptr0[out_hstep * 5 + 1] = float32_to_bfloat16(sum1[5]);
                    outptr0[out_hstep * 6 + 0] = float32_to_bfloat16(sum0[6]);
                    outptr0[out_hstep * 6 + 1] = float32_to_bfloat16(sum1[6]);
                    outptr0[out_hstep * 7 + 0] = float32_to_bfloat16(sum0[7]);
                    outptr0[out_hstep * 7 + 1] = float32_to_bfloat16(sum1[7]);
                    outptr0 += 2;
                }
            }
            else
            {
                __lasx_xvst((__m256i)_sum0, outptr + 0, 0);
                __lasx_xvst((__m256i)_sum1, outptr + 8, 0);
            }

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            const unsigned short* pA = pAT;

            __m256 _sum0;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m256)__lasx_xvld(pC, 0);
                }
                else
                {
                    _sum0 = (__m256)__lasx_xvldi(0);
                }
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr + 0, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = bfloat2float_lasx((__m128i)__lsx_vld(pA, 0));

                _sum0 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[0])), _sum0);

                pA += 8;
                pB += 1;
            }

            if (k_end)
            {
                _sum0 = activation_lasx(_sum0, activation_type, activation_params);

                if (out_elempack == 8)
                {
                    __lsx_vst(float2bfloat_lasx(_sum0), outptr0 + 0, 0);
                    outptr0 += 8;
                }
                if (out_elempack == 4)
                {
                    __m128 _sum0_lo = (__m128)__lasx_extract_lo128((__m256i)_sum0);
                    __m128 _sum0_hi = (__m128)__lasx_extract_hi128((__m256i)_sum0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum0_lo, _sum0_lo), outptr0 + 0, 0, 0);
                    outptr0 += 4;
                    __lsx_vstelm_d(float2bfloat_lsx(_sum0_hi, _sum0_hi), outptr0 + out_hstep * 4 - 4 + 0, 0, 0);
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    __lasx_xvst((__m256i)_sum0, sum0, 0);
                    outptr0[out_hstep * 0 + 0] = float32_to_bfloat16(sum0[0]);
                    outptr0[out_hstep * 1 + 0] = float32_to_bfloat16(sum0[1]);
                    outptr0[out_hstep * 2 + 0] = float32_to_bfloat16(sum0[2]);
                    outptr0[out_hstep * 3 + 0] = float32_to_bfloat16(sum0[3]);
                    outptr0[out_hstep * 4 + 0] = float32_to_bfloat16(sum0[4]);
                    outptr0[out_hstep * 5 + 0] = float32_to_bfloat16(sum0[5]);
                    outptr0[out_hstep * 6 + 0] = float32_to_bfloat16(sum0[6]);
                    outptr0[out_hstep * 7 + 0] = float32_to_bfloat16(sum0[7]);
                    outptr0 += 1;
                }
            }
            else
            {
                __lasx_xvst((__m256i)_sum0, outptr + 0, 0);
            }

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
#endif // __loongarch_asx
    for (; ii + 3 < max_ii; ii += 4)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            const unsigned short* pA = pAT;

            __m128 _sum0;
            __m128 _sum1;
            __m128 _sum2;
            __m128 _sum3;
            __m128 _sum4;
            __m128 _sum5;
            __m128 _sum6;
            __m128 _sum7;
            __m128 _sum8;
            __m128 _sum9;
            __m128 _sum10;
            __m128 _sum11;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m128)__lsx_vld(pC, 0);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                    _sum8 = _sum0;
                    _sum9 = _sum0;
                    _sum10 = _sum0;
                    _sum11 = _sum0;
                }
                else
                {
                    _sum0 = (__m128)__lsx_vldi(0);
                    _sum1 = (__m128)__lsx_vldi(0);
                    _sum2 = (__m128)__lsx_vldi(0);
                    _sum3 = (__m128)__lsx_vldi(0);
                    _sum4 = (__m128)__lsx_vldi(0);
                    _sum5 = (__m128)__lsx_vldi(0);
                    _sum6 = (__m128)__lsx_vldi(0);
                    _sum7 = (__m128)__lsx_vldi(0);
                    _sum8 = (__m128)__lsx_vldi(0);
                    _sum9 = (__m128)__lsx_vldi(0);
                    _sum10 = (__m128)__lsx_vldi(0);
                    _sum11 = (__m128)__lsx_vldi(0);
                }
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr + 0, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
                _sum2 = (__m128)__lsx_vld(outptr + 8, 0);
                _sum3 = (__m128)__lsx_vld(outptr + 12, 0);
                _sum4 = (__m128)__lsx_vld(outptr + 16, 0);
                _sum5 = (__m128)__lsx_vld(outptr + 20, 0);
                _sum6 = (__m128)__lsx_vld(outptr + 24, 0);
                _sum7 = (__m128)__lsx_vld(outptr + 28, 0);
                _sum8 = (__m128)__lsx_vld(outptr + 32, 0);
                _sum9 = (__m128)__lsx_vld(outptr + 36, 0);
                _sum10 = (__m128)__lsx_vld(outptr + 40, 0);
                _sum11 = (__m128)__lsx_vld(outptr + 44, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = bfloat2float_lsx((__m128i)__lsx_vld(pA, 0));

                _sum0 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[1])), _sum1);
                _sum2 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[2])), _sum2);
                _sum3 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[3])), _sum3);
                _sum4 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[4])), _sum4);
                _sum5 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[5])), _sum5);
                _sum6 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[6])), _sum6);
                _sum7 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[7])), _sum7);
                _sum8 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[8])), _sum8);
                _sum9 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[9])), _sum9);
                _sum10 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[10])), _sum10);
                _sum11 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[11])), _sum11);

                pA += 4;
                pB += 12;
            }

            if (k_end)
            {
                _sum0 = activation_lsx(_sum0, activation_type, activation_params);
                _sum1 = activation_lsx(_sum1, activation_type, activation_params);
                _sum2 = activation_lsx(_sum2, activation_type, activation_params);
                _sum3 = activation_lsx(_sum3, activation_type, activation_params);
                _sum4 = activation_lsx(_sum4, activation_type, activation_params);
                _sum5 = activation_lsx(_sum5, activation_type, activation_params);
                _sum6 = activation_lsx(_sum6, activation_type, activation_params);
                _sum7 = activation_lsx(_sum7, activation_type, activation_params);
                _sum8 = activation_lsx(_sum8, activation_type, activation_params);
                _sum9 = activation_lsx(_sum9, activation_type, activation_params);
                _sum10 = activation_lsx(_sum10, activation_type, activation_params);
                _sum11 = activation_lsx(_sum11, activation_type, activation_params);

                if (out_elempack == 4)
                {
                    __lsx_vstelm_d(float2bfloat_lsx(_sum0, _sum0), outptr0 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum1, _sum1), outptr0 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum2, _sum2), outptr0 + 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum3, _sum3), outptr0 + 12, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum4, _sum4), outptr0 + 16, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum5, _sum5), outptr0 + 20, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum6, _sum6), outptr0 + 24, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum7, _sum7), outptr0 + 28, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum8, _sum8), outptr0 + 32, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum9, _sum9), outptr0 + 36, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum10, _sum10), outptr0 + 40, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum11, _sum11), outptr0 + 44, 0, 0);
                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                    transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);
                    transpose4x4_ps(_sum8, _sum9, _sum10, _sum11);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum0, _sum0), outptr0 + out_hstep * 0 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum4, _sum4), outptr0 + out_hstep * 0 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum8, _sum8), outptr0 + out_hstep * 0 + 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum1, _sum1), outptr0 + out_hstep * 1 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum5, _sum5), outptr0 + out_hstep * 1 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum9, _sum9), outptr0 + out_hstep * 1 + 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum2, _sum2), outptr0 + out_hstep * 2 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum6, _sum6), outptr0 + out_hstep * 2 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum10, _sum10), outptr0 + out_hstep * 2 + 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum3, _sum3), outptr0 + out_hstep * 3 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum7, _sum7), outptr0 + out_hstep * 3 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum11, _sum11), outptr0 + out_hstep * 3 + 8, 0, 0);
                    outptr0 += 12;
                }
            }
            else
            {
                __lsx_vst((__m128i)_sum0, outptr + 0, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);
                __lsx_vst((__m128i)_sum2, outptr + 8, 0);
                __lsx_vst((__m128i)_sum3, outptr + 12, 0);
                __lsx_vst((__m128i)_sum4, outptr + 16, 0);
                __lsx_vst((__m128i)_sum5, outptr + 20, 0);
                __lsx_vst((__m128i)_sum6, outptr + 24, 0);
                __lsx_vst((__m128i)_sum7, outptr + 28, 0);
                __lsx_vst((__m128i)_sum8, outptr + 32, 0);
                __lsx_vst((__m128i)_sum9, outptr + 36, 0);
                __lsx_vst((__m128i)_sum10, outptr + 40, 0);
                __lsx_vst((__m128i)_sum11, outptr + 44, 0);
            }

            outptr += 48;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

            __m128 _sum0;
            __m128 _sum1;
            __m128 _sum2;
            __m128 _sum3;
            __m128 _sum4;
            __m128 _sum5;
            __m128 _sum6;
            __m128 _sum7;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m128)__lsx_vld(pC, 0);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                }
                else
                {
                    _sum0 = (__m128)__lsx_vldi(0);
                    _sum1 = (__m128)__lsx_vldi(0);
                    _sum2 = (__m128)__lsx_vldi(0);
                    _sum3 = (__m128)__lsx_vldi(0);
                    _sum4 = (__m128)__lsx_vldi(0);
                    _sum5 = (__m128)__lsx_vldi(0);
                    _sum6 = (__m128)__lsx_vldi(0);
                    _sum7 = (__m128)__lsx_vldi(0);
                }
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr + 0, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
                _sum2 = (__m128)__lsx_vld(outptr + 8, 0);
                _sum3 = (__m128)__lsx_vld(outptr + 12, 0);
                _sum4 = (__m128)__lsx_vld(outptr + 16, 0);
                _sum5 = (__m128)__lsx_vld(outptr + 20, 0);
                _sum6 = (__m128)__lsx_vld(outptr + 24, 0);
                _sum7 = (__m128)__lsx_vld(outptr + 28, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = bfloat2float_lsx((__m128i)__lsx_vld(pA, 0));

                _sum0 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[1])), _sum1);
                _sum2 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[2])), _sum2);
                _sum3 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[3])), _sum3);
                _sum4 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[4])), _sum4);
                _sum5 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[5])), _sum5);
                _sum6 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[6])), _sum6);
                _sum7 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[7])), _sum7);

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                _sum0 = activation_lsx(_sum0, activation_type, activation_params);
                _sum1 = activation_lsx(_sum1, activation_type, activation_params);
                _sum2 = activation_lsx(_sum2, activation_type, activation_params);
                _sum3 = activation_lsx(_sum3, activation_type, activation_params);
                _sum4 = activation_lsx(_sum4, activation_type, activation_params);
                _sum5 = activation_lsx(_sum5, activation_type, activation_params);
                _sum6 = activation_lsx(_sum6, activation_type, activation_params);
                _sum7 = activation_lsx(_sum7, activation_type, activation_params);

                if (out_elempack == 4)
                {
                    __lsx_vstelm_d(float2bfloat_lsx(_sum0, _sum0), outptr0 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum1, _sum1), outptr0 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum2, _sum2), outptr0 + 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum3, _sum3), outptr0 + 12, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum4, _sum4), outptr0 + 16, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum5, _sum5), outptr0 + 20, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum6, _sum6), outptr0 + 24, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum7, _sum7), outptr0 + 28, 0, 0);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                    transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum0, _sum0), outptr0 + out_hstep * 0 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum4, _sum4), outptr0 + out_hstep * 0 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum1, _sum1), outptr0 + out_hstep * 1 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum5, _sum5), outptr0 + out_hstep * 1 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum2, _sum2), outptr0 + out_hstep * 2 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum6, _sum6), outptr0 + out_hstep * 2 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum3, _sum3), outptr0 + out_hstep * 3 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum7, _sum7), outptr0 + out_hstep * 3 + 4, 0, 0);
                    outptr0 += 8;
                }
            }
            else
            {
                __lsx_vst((__m128i)_sum0, outptr + 0, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);
                __lsx_vst((__m128i)_sum2, outptr + 8, 0);
                __lsx_vst((__m128i)_sum3, outptr + 12, 0);
                __lsx_vst((__m128i)_sum4, outptr + 16, 0);
                __lsx_vst((__m128i)_sum5, outptr + 20, 0);
                __lsx_vst((__m128i)_sum6, outptr + 24, 0);
                __lsx_vst((__m128i)_sum7, outptr + 28, 0);
            }

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

            __m128 _sum0;
            __m128 _sum1;
            __m128 _sum2;
            __m128 _sum3;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m128)__lsx_vld(pC, 0);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
                else
                {
                    _sum0 = (__m128)__lsx_vldi(0);
                    _sum1 = (__m128)__lsx_vldi(0);
                    _sum2 = (__m128)__lsx_vldi(0);
                    _sum3 = (__m128)__lsx_vldi(0);
                }
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr + 0, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
                _sum2 = (__m128)__lsx_vld(outptr + 8, 0);
                _sum3 = (__m128)__lsx_vld(outptr + 12, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = bfloat2float_lsx((__m128i)__lsx_vld(pA, 0));

                _sum0 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[1])), _sum1);
                _sum2 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[2])), _sum2);
                _sum3 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[3])), _sum3);

                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                _sum0 = activation_lsx(_sum0, activation_type, activation_params);
                _sum1 = activation_lsx(_sum1, activation_type, activation_params);
                _sum2 = activation_lsx(_sum2, activation_type, activation_params);
                _sum3 = activation_lsx(_sum3, activation_type, activation_params);

                if (out_elempack == 4)
                {
                    __lsx_vstelm_d(float2bfloat_lsx(_sum0, _sum0), outptr0 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum1, _sum1), outptr0 + 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum2, _sum2), outptr0 + 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum3, _sum3), outptr0 + 12, 0, 0);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum0, _sum0), outptr0 + out_hstep * 0 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum1, _sum1), outptr0 + out_hstep * 1 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum2, _sum2), outptr0 + out_hstep * 2 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum3, _sum3), outptr0 + out_hstep * 3 + 0, 0, 0);
                    outptr0 += 4;
                }
            }
            else
            {
                __lsx_vst((__m128i)_sum0, outptr + 0, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);
                __lsx_vst((__m128i)_sum2, outptr + 8, 0);
                __lsx_vst((__m128i)_sum3, outptr + 12, 0);
            }

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* pA = pAT;

            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m128)__lsx_vld(pC, 0);
                    _sum1 = _sum0;
                }
                else
                {
                    _sum0 = (__m128)__lsx_vldi(0);
                    _sum1 = (__m128)__lsx_vldi(0);
                }
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr + 0, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = bfloat2float_lsx((__m128i)__lsx_vld(pA, 0));

                _sum0 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[1])), _sum1);

                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                _sum0 = activation_lsx(_sum0, activation_type, activation_params);
                _sum1 = activation_lsx(_sum1, activation_type, activation_params);

                if (out_elempack == 4)
                {
                    __lsx_vstelm_d(float2bfloat_lsx(_sum0, _sum0), outptr0 + 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_sum1, _sum1), outptr0 + 4, 0, 0);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    outptr0[out_hstep * 0 + 0] = float32_to_bfloat16(((const float*)&_sum0)[0]);
                    outptr0[out_hstep * 0 + 1] = float32_to_bfloat16(((const float*)&_sum1)[0]);
                    outptr0[out_hstep * 1 + 0] = float32_to_bfloat16(((const float*)&_sum0)[1]);
                    outptr0[out_hstep * 1 + 1] = float32_to_bfloat16(((const float*)&_sum1)[1]);
                    outptr0[out_hstep * 2 + 0] = float32_to_bfloat16(((const float*)&_sum0)[2]);
                    outptr0[out_hstep * 2 + 1] = float32_to_bfloat16(((const float*)&_sum1)[2]);
                    outptr0[out_hstep * 3 + 0] = float32_to_bfloat16(((const float*)&_sum0)[3]);
                    outptr0[out_hstep * 3 + 1] = float32_to_bfloat16(((const float*)&_sum1)[3]);
                    outptr0 += 2;
                }
            }
            else
            {
                __lsx_vst((__m128i)_sum0, outptr + 0, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);
            }

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            const unsigned short* pA = pAT;

            __m128 _sum0;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m128)__lsx_vld(pC, 0);
                }
                else
                {
                    _sum0 = (__m128)__lsx_vldi(0);
                }
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr + 0, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = bfloat2float_lsx((__m128i)__lsx_vld(pA, 0));

                _sum0 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(bfloat16_to_float32(pB[0])), _sum0);

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                _sum0 = activation_lsx(_sum0, activation_type, activation_params);

                if (out_elempack == 4)
                {
                    __lsx_vstelm_d(float2bfloat_lsx(_sum0, _sum0), outptr0 + 0, 0, 0);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    outptr0[out_hstep * 0 + 0] = float32_to_bfloat16(((const float*)&_sum0)[0]);
                    outptr0[out_hstep * 1 + 0] = float32_to_bfloat16(((const float*)&_sum0)[1]);
                    outptr0[out_hstep * 2 + 0] = float32_to_bfloat16(((const float*)&_sum0)[2]);
                    outptr0[out_hstep * 3 + 0] = float32_to_bfloat16(((const float*)&_sum0)[3]);
                    outptr0 += 1;
                }
            }
            else
            {
                __lsx_vst((__m128i)_sum0, outptr + 0, 0);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            const unsigned short* pA = pAT;

            float _sum00 = 0.f;
            float _sum10 = 0.f;
            float _sum01 = 0.f;
            float _sum11 = 0.f;
            float _sum02 = 0.f;
            float _sum12 = 0.f;
            float _sum03 = 0.f;
            float _sum13 = 0.f;
            float _sum04 = 0.f;
            float _sum14 = 0.f;
            float _sum05 = 0.f;
            float _sum15 = 0.f;
            float _sum06 = 0.f;
            float _sum16 = 0.f;
            float _sum07 = 0.f;
            float _sum17 = 0.f;
            float _sum08 = 0.f;
            float _sum18 = 0.f;
            float _sum09 = 0.f;
            float _sum19 = 0.f;
            float _sum0a = 0.f;
            float _sum1a = 0.f;
            float _sum0b = 0.f;
            float _sum1b = 0.f;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = pC[0];
                    _sum10 = pC[1];
                    _sum01 = pC[0];
                    _sum11 = pC[1];
                    _sum02 = pC[0];
                    _sum12 = pC[1];
                    _sum03 = pC[0];
                    _sum13 = pC[1];
                    _sum04 = pC[0];
                    _sum14 = pC[1];
                    _sum05 = pC[0];
                    _sum15 = pC[1];
                    _sum06 = pC[0];
                    _sum16 = pC[1];
                    _sum07 = pC[0];
                    _sum17 = pC[1];
                    _sum08 = pC[0];
                    _sum18 = pC[1];
                    _sum09 = pC[0];
                    _sum19 = pC[1];
                    _sum0a = pC[0];
                    _sum1a = pC[1];
                    _sum0b = pC[0];
                    _sum1b = pC[1];
                }
            }
            else
            {
                _sum00 = outptr[0];
                _sum10 = outptr[1];
                _sum01 = outptr[2];
                _sum11 = outptr[3];
                _sum02 = outptr[4];
                _sum12 = outptr[5];
                _sum03 = outptr[6];
                _sum13 = outptr[7];
                _sum04 = outptr[8];
                _sum14 = outptr[9];
                _sum05 = outptr[10];
                _sum15 = outptr[11];
                _sum06 = outptr[12];
                _sum16 = outptr[13];
                _sum07 = outptr[14];
                _sum17 = outptr[15];
                _sum08 = outptr[16];
                _sum18 = outptr[17];
                _sum09 = outptr[18];
                _sum19 = outptr[19];
                _sum0a = outptr[20];
                _sum1a = outptr[21];
                _sum0b = outptr[22];
                _sum1b = outptr[23];
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float _pA0 = bfloat16_to_float32(pA[0]);
                float _pA1 = bfloat16_to_float32(pA[1]);
                float _pB0 = bfloat16_to_float32(pB[0]);
                float _pB1 = bfloat16_to_float32(pB[1]);
                float _pB2 = bfloat16_to_float32(pB[2]);
                float _pB3 = bfloat16_to_float32(pB[3]);
                float _pB4 = bfloat16_to_float32(pB[4]);
                float _pB5 = bfloat16_to_float32(pB[5]);
                float _pB6 = bfloat16_to_float32(pB[6]);
                float _pB7 = bfloat16_to_float32(pB[7]);
                float _pB8 = bfloat16_to_float32(pB[8]);
                float _pB9 = bfloat16_to_float32(pB[9]);
                float _pBa = bfloat16_to_float32(pB[10]);
                float _pBb = bfloat16_to_float32(pB[11]);
                _sum00 += _pA0 * _pB0;
                _sum10 += _pA1 * _pB0;
                _sum01 += _pA0 * _pB1;
                _sum11 += _pA1 * _pB1;
                _sum02 += _pA0 * _pB2;
                _sum12 += _pA1 * _pB2;
                _sum03 += _pA0 * _pB3;
                _sum13 += _pA1 * _pB3;
                _sum04 += _pA0 * _pB4;
                _sum14 += _pA1 * _pB4;
                _sum05 += _pA0 * _pB5;
                _sum15 += _pA1 * _pB5;
                _sum06 += _pA0 * _pB6;
                _sum16 += _pA1 * _pB6;
                _sum07 += _pA0 * _pB7;
                _sum17 += _pA1 * _pB7;
                _sum08 += _pA0 * _pB8;
                _sum18 += _pA1 * _pB8;
                _sum09 += _pA0 * _pB9;
                _sum19 += _pA1 * _pB9;
                _sum0a += _pA0 * _pBa;
                _sum1a += _pA1 * _pBa;
                _sum0b += _pA0 * _pBb;
                _sum1b += _pA1 * _pBb;
                pA += 2;
                pB += 12;
            }

            if (k_end)
            {
                _sum00 = activation_ss(_sum00, activation_type, activation_params);
                _sum10 = activation_ss(_sum10, activation_type, activation_params);
                _sum01 = activation_ss(_sum01, activation_type, activation_params);
                _sum11 = activation_ss(_sum11, activation_type, activation_params);
                _sum02 = activation_ss(_sum02, activation_type, activation_params);
                _sum12 = activation_ss(_sum12, activation_type, activation_params);
                _sum03 = activation_ss(_sum03, activation_type, activation_params);
                _sum13 = activation_ss(_sum13, activation_type, activation_params);
                _sum04 = activation_ss(_sum04, activation_type, activation_params);
                _sum14 = activation_ss(_sum14, activation_type, activation_params);
                _sum05 = activation_ss(_sum05, activation_type, activation_params);
                _sum15 = activation_ss(_sum15, activation_type, activation_params);
                _sum06 = activation_ss(_sum06, activation_type, activation_params);
                _sum16 = activation_ss(_sum16, activation_type, activation_params);
                _sum07 = activation_ss(_sum07, activation_type, activation_params);
                _sum17 = activation_ss(_sum17, activation_type, activation_params);
                _sum08 = activation_ss(_sum08, activation_type, activation_params);
                _sum18 = activation_ss(_sum18, activation_type, activation_params);
                _sum09 = activation_ss(_sum09, activation_type, activation_params);
                _sum19 = activation_ss(_sum19, activation_type, activation_params);
                _sum0a = activation_ss(_sum0a, activation_type, activation_params);
                _sum1a = activation_ss(_sum1a, activation_type, activation_params);
                _sum0b = activation_ss(_sum0b, activation_type, activation_params);
                _sum1b = activation_ss(_sum1b, activation_type, activation_params);
                outptr0[out_hstep * 0 + 0] = float32_to_bfloat16(_sum00);
                outptr0[out_hstep * 0 + 1] = float32_to_bfloat16(_sum01);
                outptr0[out_hstep * 0 + 2] = float32_to_bfloat16(_sum02);
                outptr0[out_hstep * 0 + 3] = float32_to_bfloat16(_sum03);
                outptr0[out_hstep * 0 + 4] = float32_to_bfloat16(_sum04);
                outptr0[out_hstep * 0 + 5] = float32_to_bfloat16(_sum05);
                outptr0[out_hstep * 0 + 6] = float32_to_bfloat16(_sum06);
                outptr0[out_hstep * 0 + 7] = float32_to_bfloat16(_sum07);
                outptr0[out_hstep * 0 + 8] = float32_to_bfloat16(_sum08);
                outptr0[out_hstep * 0 + 9] = float32_to_bfloat16(_sum09);
                outptr0[out_hstep * 0 + 10] = float32_to_bfloat16(_sum0a);
                outptr0[out_hstep * 0 + 11] = float32_to_bfloat16(_sum0b);
                outptr0[out_hstep * 1 + 0] = float32_to_bfloat16(_sum10);
                outptr0[out_hstep * 1 + 1] = float32_to_bfloat16(_sum11);
                outptr0[out_hstep * 1 + 2] = float32_to_bfloat16(_sum12);
                outptr0[out_hstep * 1 + 3] = float32_to_bfloat16(_sum13);
                outptr0[out_hstep * 1 + 4] = float32_to_bfloat16(_sum14);
                outptr0[out_hstep * 1 + 5] = float32_to_bfloat16(_sum15);
                outptr0[out_hstep * 1 + 6] = float32_to_bfloat16(_sum16);
                outptr0[out_hstep * 1 + 7] = float32_to_bfloat16(_sum17);
                outptr0[out_hstep * 1 + 8] = float32_to_bfloat16(_sum18);
                outptr0[out_hstep * 1 + 9] = float32_to_bfloat16(_sum19);
                outptr0[out_hstep * 1 + 10] = float32_to_bfloat16(_sum1a);
                outptr0[out_hstep * 1 + 11] = float32_to_bfloat16(_sum1b);
                outptr0 += 12;
            }
            else
            {
                outptr[0] = _sum00;
                outptr[1] = _sum10;
                outptr[2] = _sum01;
                outptr[3] = _sum11;
                outptr[4] = _sum02;
                outptr[5] = _sum12;
                outptr[6] = _sum03;
                outptr[7] = _sum13;
                outptr[8] = _sum04;
                outptr[9] = _sum14;
                outptr[10] = _sum05;
                outptr[11] = _sum15;
                outptr[12] = _sum06;
                outptr[13] = _sum16;
                outptr[14] = _sum07;
                outptr[15] = _sum17;
                outptr[16] = _sum08;
                outptr[17] = _sum18;
                outptr[18] = _sum09;
                outptr[19] = _sum19;
                outptr[20] = _sum0a;
                outptr[21] = _sum1a;
                outptr[22] = _sum0b;
                outptr[23] = _sum1b;
            }

            outptr += 24;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

            float _sum00 = 0.f;
            float _sum10 = 0.f;
            float _sum01 = 0.f;
            float _sum11 = 0.f;
            float _sum02 = 0.f;
            float _sum12 = 0.f;
            float _sum03 = 0.f;
            float _sum13 = 0.f;
            float _sum04 = 0.f;
            float _sum14 = 0.f;
            float _sum05 = 0.f;
            float _sum15 = 0.f;
            float _sum06 = 0.f;
            float _sum16 = 0.f;
            float _sum07 = 0.f;
            float _sum17 = 0.f;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = pC[0];
                    _sum10 = pC[1];
                    _sum01 = pC[0];
                    _sum11 = pC[1];
                    _sum02 = pC[0];
                    _sum12 = pC[1];
                    _sum03 = pC[0];
                    _sum13 = pC[1];
                    _sum04 = pC[0];
                    _sum14 = pC[1];
                    _sum05 = pC[0];
                    _sum15 = pC[1];
                    _sum06 = pC[0];
                    _sum16 = pC[1];
                    _sum07 = pC[0];
                    _sum17 = pC[1];
                }
            }
            else
            {
                _sum00 = outptr[0];
                _sum10 = outptr[1];
                _sum01 = outptr[2];
                _sum11 = outptr[3];
                _sum02 = outptr[4];
                _sum12 = outptr[5];
                _sum03 = outptr[6];
                _sum13 = outptr[7];
                _sum04 = outptr[8];
                _sum14 = outptr[9];
                _sum05 = outptr[10];
                _sum15 = outptr[11];
                _sum06 = outptr[12];
                _sum16 = outptr[13];
                _sum07 = outptr[14];
                _sum17 = outptr[15];
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float _pA0 = bfloat16_to_float32(pA[0]);
                float _pA1 = bfloat16_to_float32(pA[1]);
                float _pB0 = bfloat16_to_float32(pB[0]);
                float _pB1 = bfloat16_to_float32(pB[1]);
                float _pB2 = bfloat16_to_float32(pB[2]);
                float _pB3 = bfloat16_to_float32(pB[3]);
                float _pB4 = bfloat16_to_float32(pB[4]);
                float _pB5 = bfloat16_to_float32(pB[5]);
                float _pB6 = bfloat16_to_float32(pB[6]);
                float _pB7 = bfloat16_to_float32(pB[7]);
                _sum00 += _pA0 * _pB0;
                _sum10 += _pA1 * _pB0;
                _sum01 += _pA0 * _pB1;
                _sum11 += _pA1 * _pB1;
                _sum02 += _pA0 * _pB2;
                _sum12 += _pA1 * _pB2;
                _sum03 += _pA0 * _pB3;
                _sum13 += _pA1 * _pB3;
                _sum04 += _pA0 * _pB4;
                _sum14 += _pA1 * _pB4;
                _sum05 += _pA0 * _pB5;
                _sum15 += _pA1 * _pB5;
                _sum06 += _pA0 * _pB6;
                _sum16 += _pA1 * _pB6;
                _sum07 += _pA0 * _pB7;
                _sum17 += _pA1 * _pB7;
                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                _sum00 = activation_ss(_sum00, activation_type, activation_params);
                _sum10 = activation_ss(_sum10, activation_type, activation_params);
                _sum01 = activation_ss(_sum01, activation_type, activation_params);
                _sum11 = activation_ss(_sum11, activation_type, activation_params);
                _sum02 = activation_ss(_sum02, activation_type, activation_params);
                _sum12 = activation_ss(_sum12, activation_type, activation_params);
                _sum03 = activation_ss(_sum03, activation_type, activation_params);
                _sum13 = activation_ss(_sum13, activation_type, activation_params);
                _sum04 = activation_ss(_sum04, activation_type, activation_params);
                _sum14 = activation_ss(_sum14, activation_type, activation_params);
                _sum05 = activation_ss(_sum05, activation_type, activation_params);
                _sum15 = activation_ss(_sum15, activation_type, activation_params);
                _sum06 = activation_ss(_sum06, activation_type, activation_params);
                _sum16 = activation_ss(_sum16, activation_type, activation_params);
                _sum07 = activation_ss(_sum07, activation_type, activation_params);
                _sum17 = activation_ss(_sum17, activation_type, activation_params);
                outptr0[out_hstep * 0 + 0] = float32_to_bfloat16(_sum00);
                outptr0[out_hstep * 0 + 1] = float32_to_bfloat16(_sum01);
                outptr0[out_hstep * 0 + 2] = float32_to_bfloat16(_sum02);
                outptr0[out_hstep * 0 + 3] = float32_to_bfloat16(_sum03);
                outptr0[out_hstep * 0 + 4] = float32_to_bfloat16(_sum04);
                outptr0[out_hstep * 0 + 5] = float32_to_bfloat16(_sum05);
                outptr0[out_hstep * 0 + 6] = float32_to_bfloat16(_sum06);
                outptr0[out_hstep * 0 + 7] = float32_to_bfloat16(_sum07);
                outptr0[out_hstep * 1 + 0] = float32_to_bfloat16(_sum10);
                outptr0[out_hstep * 1 + 1] = float32_to_bfloat16(_sum11);
                outptr0[out_hstep * 1 + 2] = float32_to_bfloat16(_sum12);
                outptr0[out_hstep * 1 + 3] = float32_to_bfloat16(_sum13);
                outptr0[out_hstep * 1 + 4] = float32_to_bfloat16(_sum14);
                outptr0[out_hstep * 1 + 5] = float32_to_bfloat16(_sum15);
                outptr0[out_hstep * 1 + 6] = float32_to_bfloat16(_sum16);
                outptr0[out_hstep * 1 + 7] = float32_to_bfloat16(_sum17);
                outptr0 += 8;
            }
            else
            {
                outptr[0] = _sum00;
                outptr[1] = _sum10;
                outptr[2] = _sum01;
                outptr[3] = _sum11;
                outptr[4] = _sum02;
                outptr[5] = _sum12;
                outptr[6] = _sum03;
                outptr[7] = _sum13;
                outptr[8] = _sum04;
                outptr[9] = _sum14;
                outptr[10] = _sum05;
                outptr[11] = _sum15;
                outptr[12] = _sum06;
                outptr[13] = _sum16;
                outptr[14] = _sum07;
                outptr[15] = _sum17;
            }

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

            float _sum00 = 0.f;
            float _sum10 = 0.f;
            float _sum01 = 0.f;
            float _sum11 = 0.f;
            float _sum02 = 0.f;
            float _sum12 = 0.f;
            float _sum03 = 0.f;
            float _sum13 = 0.f;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = pC[0];
                    _sum10 = pC[1];
                    _sum01 = pC[0];
                    _sum11 = pC[1];
                    _sum02 = pC[0];
                    _sum12 = pC[1];
                    _sum03 = pC[0];
                    _sum13 = pC[1];
                }
            }
            else
            {
                _sum00 = outptr[0];
                _sum10 = outptr[1];
                _sum01 = outptr[2];
                _sum11 = outptr[3];
                _sum02 = outptr[4];
                _sum12 = outptr[5];
                _sum03 = outptr[6];
                _sum13 = outptr[7];
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float _pA0 = bfloat16_to_float32(pA[0]);
                float _pA1 = bfloat16_to_float32(pA[1]);
                float _pB0 = bfloat16_to_float32(pB[0]);
                float _pB1 = bfloat16_to_float32(pB[1]);
                float _pB2 = bfloat16_to_float32(pB[2]);
                float _pB3 = bfloat16_to_float32(pB[3]);
                _sum00 += _pA0 * _pB0;
                _sum10 += _pA1 * _pB0;
                _sum01 += _pA0 * _pB1;
                _sum11 += _pA1 * _pB1;
                _sum02 += _pA0 * _pB2;
                _sum12 += _pA1 * _pB2;
                _sum03 += _pA0 * _pB3;
                _sum13 += _pA1 * _pB3;
                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                _sum00 = activation_ss(_sum00, activation_type, activation_params);
                _sum10 = activation_ss(_sum10, activation_type, activation_params);
                _sum01 = activation_ss(_sum01, activation_type, activation_params);
                _sum11 = activation_ss(_sum11, activation_type, activation_params);
                _sum02 = activation_ss(_sum02, activation_type, activation_params);
                _sum12 = activation_ss(_sum12, activation_type, activation_params);
                _sum03 = activation_ss(_sum03, activation_type, activation_params);
                _sum13 = activation_ss(_sum13, activation_type, activation_params);
                outptr0[out_hstep * 0 + 0] = float32_to_bfloat16(_sum00);
                outptr0[out_hstep * 0 + 1] = float32_to_bfloat16(_sum01);
                outptr0[out_hstep * 0 + 2] = float32_to_bfloat16(_sum02);
                outptr0[out_hstep * 0 + 3] = float32_to_bfloat16(_sum03);
                outptr0[out_hstep * 1 + 0] = float32_to_bfloat16(_sum10);
                outptr0[out_hstep * 1 + 1] = float32_to_bfloat16(_sum11);
                outptr0[out_hstep * 1 + 2] = float32_to_bfloat16(_sum12);
                outptr0[out_hstep * 1 + 3] = float32_to_bfloat16(_sum13);
                outptr0 += 4;
            }
            else
            {
                outptr[0] = _sum00;
                outptr[1] = _sum10;
                outptr[2] = _sum01;
                outptr[3] = _sum11;
                outptr[4] = _sum02;
                outptr[5] = _sum12;
                outptr[6] = _sum03;
                outptr[7] = _sum13;
            }

            outptr += 8;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* pA = pAT;

            float _sum00 = 0.f;
            float _sum10 = 0.f;
            float _sum01 = 0.f;
            float _sum11 = 0.f;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = pC[0];
                    _sum10 = pC[1];
                    _sum01 = pC[0];
                    _sum11 = pC[1];
                }
                else
                {
                    _sum00 = 0.f;
                    _sum10 = 0.f;
                    _sum01 = 0.f;
                    _sum11 = 0.f;
                }
            }
            else
            {
                _sum00 = outptr[0];
                _sum10 = outptr[1];
                _sum01 = outptr[2];
                _sum11 = outptr[3];
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float _pA0 = bfloat16_to_float32(pA[0]);
                float _pA1 = bfloat16_to_float32(pA[1]);
                float _pB0 = bfloat16_to_float32(pB[0]);
                float _pB1 = bfloat16_to_float32(pB[1]);
                _sum00 += _pA0 * _pB0;
                _sum10 += _pA1 * _pB0;
                _sum01 += _pA0 * _pB1;
                _sum11 += _pA1 * _pB1;
                pA += 2;
                pB += 2;
            }

            if (k_end)
            {
                _sum00 = activation_ss(_sum00, activation_type, activation_params);
                _sum10 = activation_ss(_sum10, activation_type, activation_params);
                _sum01 = activation_ss(_sum01, activation_type, activation_params);
                _sum11 = activation_ss(_sum11, activation_type, activation_params);
                outptr0[out_hstep * 0 + 0] = float32_to_bfloat16(_sum00);
                outptr0[out_hstep * 0 + 1] = float32_to_bfloat16(_sum01);
                outptr0[out_hstep * 1 + 0] = float32_to_bfloat16(_sum10);
                outptr0[out_hstep * 1 + 1] = float32_to_bfloat16(_sum11);
                outptr0 += 2;
            }
            else
            {
                outptr[0] = _sum00;
                outptr[1] = _sum10;
                outptr[2] = _sum01;
                outptr[3] = _sum11;
            }

            outptr += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            const unsigned short* pA = pAT;

            float _sum00 = 0.f;
            float _sum10 = 0.f;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = pC[0];
                    _sum10 = pC[1];
                }
                else
                {
                    _sum00 = 0.f;
                    _sum10 = 0.f;
                }
            }
            else
            {
                _sum00 = outptr[0];
                _sum10 = outptr[1];
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float _pA0 = bfloat16_to_float32(pA[0]);
                float _pA1 = bfloat16_to_float32(pA[1]);
                float _pB0 = bfloat16_to_float32(pB[0]);
                _sum00 += _pA0 * _pB0;
                _sum10 += _pA1 * _pB0;
                pA += 2;
                pB += 1;
            }

            if (k_end)
            {
                _sum00 = activation_ss(_sum00, activation_type, activation_params);
                _sum10 = activation_ss(_sum10, activation_type, activation_params);
                outptr0[out_hstep * 0 + 0] = float32_to_bfloat16(_sum00);
                outptr0[out_hstep * 1 + 0] = float32_to_bfloat16(_sum10);
                outptr0 += 1;
            }
            else
            {
                outptr[0] = _sum00;
                outptr[1] = _sum10;
            }

            outptr += 2;
        }

        pAT += max_kk * 2;
    }
    for (; ii < max_ii; ii += 1)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            const unsigned short* pA = pAT;

            float _sum0 = 0.f;
            float _sum1 = 0.f;
            float _sum2 = 0.f;
            float _sum3 = 0.f;
            float _sum4 = 0.f;
            float _sum5 = 0.f;
            float _sum6 = 0.f;
            float _sum7 = 0.f;
            float _sum8 = 0.f;
            float _sum9 = 0.f;
            float _suma = 0.f;
            float _sumb = 0.f;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = pC[0];
                    _sum1 = pC[0];
                    _sum2 = pC[0];
                    _sum3 = pC[0];
                    _sum4 = pC[0];
                    _sum5 = pC[0];
                    _sum6 = pC[0];
                    _sum7 = pC[0];
                    _sum8 = pC[0];
                    _sum9 = pC[0];
                    _suma = pC[0];
                    _sumb = pC[0];
                }
            }
            else
            {
                _sum0 = outptr[0];
                _sum1 = outptr[1];
                _sum2 = outptr[2];
                _sum3 = outptr[3];
                _sum4 = outptr[4];
                _sum5 = outptr[5];
                _sum6 = outptr[6];
                _sum7 = outptr[7];
                _sum8 = outptr[8];
                _sum9 = outptr[9];
                _suma = outptr[10];
                _sumb = outptr[11];
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float _pA = bfloat16_to_float32(pA[0]);
                _sum0 += _pA * bfloat16_to_float32(pB[0]);
                _sum1 += _pA * bfloat16_to_float32(pB[1]);
                _sum2 += _pA * bfloat16_to_float32(pB[2]);
                _sum3 += _pA * bfloat16_to_float32(pB[3]);
                _sum4 += _pA * bfloat16_to_float32(pB[4]);
                _sum5 += _pA * bfloat16_to_float32(pB[5]);
                _sum6 += _pA * bfloat16_to_float32(pB[6]);
                _sum7 += _pA * bfloat16_to_float32(pB[7]);
                _sum8 += _pA * bfloat16_to_float32(pB[8]);
                _sum9 += _pA * bfloat16_to_float32(pB[9]);
                _suma += _pA * bfloat16_to_float32(pB[10]);
                _sumb += _pA * bfloat16_to_float32(pB[11]);
                pA += 1;
                pB += 12;
            }

            if (k_end)
            {
                _sum0 = activation_ss(_sum0, activation_type, activation_params);
                _sum1 = activation_ss(_sum1, activation_type, activation_params);
                _sum2 = activation_ss(_sum2, activation_type, activation_params);
                _sum3 = activation_ss(_sum3, activation_type, activation_params);
                _sum4 = activation_ss(_sum4, activation_type, activation_params);
                _sum5 = activation_ss(_sum5, activation_type, activation_params);
                _sum6 = activation_ss(_sum6, activation_type, activation_params);
                _sum7 = activation_ss(_sum7, activation_type, activation_params);
                _sum8 = activation_ss(_sum8, activation_type, activation_params);
                _sum9 = activation_ss(_sum9, activation_type, activation_params);
                _suma = activation_ss(_suma, activation_type, activation_params);
                _sumb = activation_ss(_sumb, activation_type, activation_params);
                outptr0[0] = float32_to_bfloat16(_sum0);
                outptr0[1] = float32_to_bfloat16(_sum1);
                outptr0[2] = float32_to_bfloat16(_sum2);
                outptr0[3] = float32_to_bfloat16(_sum3);
                outptr0[4] = float32_to_bfloat16(_sum4);
                outptr0[5] = float32_to_bfloat16(_sum5);
                outptr0[6] = float32_to_bfloat16(_sum6);
                outptr0[7] = float32_to_bfloat16(_sum7);
                outptr0[8] = float32_to_bfloat16(_sum8);
                outptr0[9] = float32_to_bfloat16(_sum9);
                outptr0[10] = float32_to_bfloat16(_suma);
                outptr0[11] = float32_to_bfloat16(_sumb);
                outptr0 += 12;
            }
            else
            {
                outptr[0] = _sum0;
                outptr[1] = _sum1;
                outptr[2] = _sum2;
                outptr[3] = _sum3;
                outptr[4] = _sum4;
                outptr[5] = _sum5;
                outptr[6] = _sum6;
                outptr[7] = _sum7;
                outptr[8] = _sum8;
                outptr[9] = _sum9;
                outptr[10] = _suma;
                outptr[11] = _sumb;
            }

            outptr += 12;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

            float _sum0 = 0.f;
            float _sum1 = 0.f;
            float _sum2 = 0.f;
            float _sum3 = 0.f;
            float _sum4 = 0.f;
            float _sum5 = 0.f;
            float _sum6 = 0.f;
            float _sum7 = 0.f;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = pC[0];
                    _sum1 = pC[0];
                    _sum2 = pC[0];
                    _sum3 = pC[0];
                    _sum4 = pC[0];
                    _sum5 = pC[0];
                    _sum6 = pC[0];
                    _sum7 = pC[0];
                }
            }
            else
            {
                _sum0 = outptr[0];
                _sum1 = outptr[1];
                _sum2 = outptr[2];
                _sum3 = outptr[3];
                _sum4 = outptr[4];
                _sum5 = outptr[5];
                _sum6 = outptr[6];
                _sum7 = outptr[7];
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float _pA = bfloat16_to_float32(pA[0]);
                _sum0 += _pA * bfloat16_to_float32(pB[0]);
                _sum1 += _pA * bfloat16_to_float32(pB[1]);
                _sum2 += _pA * bfloat16_to_float32(pB[2]);
                _sum3 += _pA * bfloat16_to_float32(pB[3]);
                _sum4 += _pA * bfloat16_to_float32(pB[4]);
                _sum5 += _pA * bfloat16_to_float32(pB[5]);
                _sum6 += _pA * bfloat16_to_float32(pB[6]);
                _sum7 += _pA * bfloat16_to_float32(pB[7]);
                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                _sum0 = activation_ss(_sum0, activation_type, activation_params);
                _sum1 = activation_ss(_sum1, activation_type, activation_params);
                _sum2 = activation_ss(_sum2, activation_type, activation_params);
                _sum3 = activation_ss(_sum3, activation_type, activation_params);
                _sum4 = activation_ss(_sum4, activation_type, activation_params);
                _sum5 = activation_ss(_sum5, activation_type, activation_params);
                _sum6 = activation_ss(_sum6, activation_type, activation_params);
                _sum7 = activation_ss(_sum7, activation_type, activation_params);
                outptr0[0] = float32_to_bfloat16(_sum0);
                outptr0[1] = float32_to_bfloat16(_sum1);
                outptr0[2] = float32_to_bfloat16(_sum2);
                outptr0[3] = float32_to_bfloat16(_sum3);
                outptr0[4] = float32_to_bfloat16(_sum4);
                outptr0[5] = float32_to_bfloat16(_sum5);
                outptr0[6] = float32_to_bfloat16(_sum6);
                outptr0[7] = float32_to_bfloat16(_sum7);
                outptr0 += 8;
            }
            else
            {
                outptr[0] = _sum0;
                outptr[1] = _sum1;
                outptr[2] = _sum2;
                outptr[3] = _sum3;
                outptr[4] = _sum4;
                outptr[5] = _sum5;
                outptr[6] = _sum6;
                outptr[7] = _sum7;
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

            float _sum0 = 0.f;
            float _sum1 = 0.f;
            float _sum2 = 0.f;
            float _sum3 = 0.f;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = pC[0];
                    _sum1 = pC[0];
                    _sum2 = pC[0];
                    _sum3 = pC[0];
                }
            }
            else
            {
                _sum0 = outptr[0];
                _sum1 = outptr[1];
                _sum2 = outptr[2];
                _sum3 = outptr[3];
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float _pA = bfloat16_to_float32(pA[0]);
                _sum0 += _pA * bfloat16_to_float32(pB[0]);
                _sum1 += _pA * bfloat16_to_float32(pB[1]);
                _sum2 += _pA * bfloat16_to_float32(pB[2]);
                _sum3 += _pA * bfloat16_to_float32(pB[3]);
                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                _sum0 = activation_ss(_sum0, activation_type, activation_params);
                _sum1 = activation_ss(_sum1, activation_type, activation_params);
                _sum2 = activation_ss(_sum2, activation_type, activation_params);
                _sum3 = activation_ss(_sum3, activation_type, activation_params);
                outptr0[0] = float32_to_bfloat16(_sum0);
                outptr0[1] = float32_to_bfloat16(_sum1);
                outptr0[2] = float32_to_bfloat16(_sum2);
                outptr0[3] = float32_to_bfloat16(_sum3);
                outptr0 += 4;
            }
            else
            {
                outptr[0] = _sum0;
                outptr[1] = _sum1;
                outptr[2] = _sum2;
                outptr[3] = _sum3;
            }

            outptr += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* pA = pAT;

            float _sum0 = 0.f;
            float _sum1 = 0.f;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = pC[0];
                    _sum1 = pC[0];
                }
                else
                {
                    _sum0 = 0.f;
                    _sum1 = 0.f;
                }
            }
            else
            {
                _sum0 = outptr[0];
                _sum1 = outptr[1];
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float _pA = bfloat16_to_float32(pA[0]);
                _sum0 += _pA * bfloat16_to_float32(pB[0]);
                _sum1 += _pA * bfloat16_to_float32(pB[1]);
                pA += 1;
                pB += 2;
            }

            if (k_end)
            {
                _sum0 = activation_ss(_sum0, activation_type, activation_params);
                _sum1 = activation_ss(_sum1, activation_type, activation_params);
                outptr0[0] = float32_to_bfloat16(_sum0);
                outptr0[1] = float32_to_bfloat16(_sum1);
                outptr0 += 2;
            }
            else
            {
                outptr[0] = _sum0;
                outptr[1] = _sum1;
            }

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            const unsigned short* pA = pAT;

            float _sum0 = 0.f;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = pC[0];
                }
                else
                {
                    _sum0 = 0.f;
                }
            }
            else
            {
                _sum0 = outptr[0];
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float _pA = bfloat16_to_float32(pA[0]);
                _sum0 += _pA * bfloat16_to_float32(pB[0]);
                pA += 1;
                pB += 1;
            }

            if (k_end)
            {
                _sum0 = activation_ss(_sum0, activation_type, activation_params);
                outptr0[0] = float32_to_bfloat16(_sum0);
                outptr0 += 1;
            }
            else
            {
                outptr[0] = _sum0;
            }

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void convolution_im2col_gemm_get_optimal_tile_mnk_bf16s(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const int l2_cache_size_fp32 = (int)(get_cpu_level2_cache_size() / sizeof(unsigned short));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve K
    {
        // try not to split K
#if __loongarch_sx
#if __loongarch_asx
        int tile_size = (l2_cache_size_fp32 - 32) / 16;
#else
        int tile_size = (l2_cache_size_fp32 - 16) / 8;
#endif
#else
        int tile_size = (l2_cache_size_fp32 - 2) / 3;
#endif

#if __loongarch_sx
        TILE_K = std::max(4, tile_size / 4 * 4);
#else
        TILE_K = std::max(2, tile_size / 2 * 2);
#endif

        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __loongarch_sx
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif
    }

    // solve M
    {
#if __loongarch_sx
#if __loongarch_asx
        int nn_M = (M + 7) / 8;
#else
        int nn_M = (M + 3) / 4;
#endif
#else
        int nn_M = (M + 3) / 4;
#endif

#if __loongarch_sx
#if __loongarch_asx
        TILE_M = std::max(8, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::max(4, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#endif
#else
        TILE_M = std::max(2, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    {
        TILE_M *= std::min(nT, get_physical_cpu_count());

        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __loongarch_sx
#if __loongarch_asx
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#endif
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif

        if (nT > 1)
        {
#if __loongarch_sx
#if __loongarch_asx
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#else
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 3) / 4 * 4);
#endif
#else
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
        }
    }

    if (N > 0)
    {
        int tile_size;
        if (TILE_K >= K)
        {
            tile_size = (l2_cache_size_fp32 - TILE_M * TILE_K) / TILE_K;
        }
        else
        {
            tile_size = (l2_cache_size_fp32 - TILE_M * TILE_K) / (TILE_M + TILE_K);
        }

#if __loongarch_sx
        TILE_N = std::max(4, tile_size / 4 * 4);
#else
        TILE_N = std::max(1, tile_size);
#endif

        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __loongarch_sx
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif

#if __loongarch_sx
        TILE_N = std::max(4, TILE_N);
#else
        TILE_N = std::max(1, TILE_N);
#endif
    }
}

static void convolution_im2col_input_tile_conv1x1s1d1_bf16s(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    const int elempack = bottom_blob.elempack;

    unsigned short* pp = B;

    int jj = 0;
#if __loongarch_sx
    for (; jj + 11 < max_jj; jj += 12)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                // 12 positions x 8 channels of bf16
                // convert to fp32, transpose, convert back
                __m256 _r0 = bfloat2float_lasx((__m128i)__lsx_vld(p0, 0));
                __m256 _r1 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 8, 0));
                __m256 _r2 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 16, 0));
                __m256 _r3 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 24, 0));
                __m256 _r4 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 32, 0));
                __m256 _r5 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 40, 0));
                __m256 _r6 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 48, 0));
                __m256 _r7 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 56, 0));
                __m256 _r8 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 64, 0));
                __m256 _r9 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 72, 0));
                __m256 _ra = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 80, 0));
                __m256 _rb = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 88, 0));
                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                transpose8x4_ps(_r8, _r9, _ra, _rb);
                // After transpose8x8: _r0=ch0[pos0-7], ..., _r7=ch7[pos0-7]
                // After transpose8x4: _r8=[ch0:pos8-11|ch1:pos8-11], _r9=[ch2|ch3], _ra=[ch4|ch5], _rb=[ch6|ch7]
                __lsx_vst(float2bfloat_lasx(_r0), pp, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_r8), (__m128)__lasx_extract_lo128((__m256i)_r8)), pp + 8, 0, 0);
                __lsx_vst(float2bfloat_lasx(_r1), pp + 12, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_r8), (__m128)__lasx_extract_hi128((__m256i)_r8)), pp + 20, 0, 0);
                __lsx_vst(float2bfloat_lasx(_r2), pp + 24, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_r9), (__m128)__lasx_extract_lo128((__m256i)_r9)), pp + 32, 0, 0);
                __lsx_vst(float2bfloat_lasx(_r3), pp + 36, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_r9), (__m128)__lasx_extract_hi128((__m256i)_r9)), pp + 44, 0, 0);
                __lsx_vst(float2bfloat_lasx(_r4), pp + 48, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_ra), (__m128)__lasx_extract_lo128((__m256i)_ra)), pp + 56, 0, 0);
                __lsx_vst(float2bfloat_lasx(_r5), pp + 60, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_ra), (__m128)__lasx_extract_hi128((__m256i)_ra)), pp + 68, 0, 0);
                __lsx_vst(float2bfloat_lasx(_r6), pp + 72, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_rb), (__m128)__lasx_extract_lo128((__m256i)_rb)), pp + 80, 0, 0);
                __lsx_vst(float2bfloat_lasx(_r7), pp + 84, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_rb), (__m128)__lasx_extract_hi128((__m256i)_rb)), pp + 92, 0, 0);
                pp += 96;
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __loongarch_asx
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // 12 positions x 4 channels -> transpose to 4 channels x 12 positions
                __m128 _r0 = bfloat2float_lsx((__m128i)__lsx_vld(p0, 0));
                __m128 _r1 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 4, 0));
                __m128 _r2 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 8, 0));
                __m128 _r3 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 12, 0));
                __m128 _r4 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 16, 0));
                __m128 _r5 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 20, 0));
                __m128 _r6 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 24, 0));
                __m128 _r7 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 28, 0));
                __m128 _r8 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 32, 0));
                __m128 _r9 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 36, 0));
                __m128 _ra = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 40, 0));
                __m128 _rb = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 44, 0));
                transpose4x4_ps(_r0, _r1, _r2, _r3);
                transpose4x4_ps(_r4, _r5, _r6, _r7);
                transpose4x4_ps(_r8, _r9, _ra, _rb);
                __lsx_vstelm_d(float2bfloat_lsx(_r0, _r0), pp, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r4, _r4), pp + 4, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r8, _r8), pp + 8, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r1, _r1), pp + 12, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r5, _r5), pp + 16, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r9, _r9), pp + 20, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r2, _r2), pp + 24, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r6, _r6), pp + 28, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_ra, _ra), pp + 32, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r3, _r3), pp + 36, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r7, _r7), pp + 40, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_rb, _rb), pp + 44, 0, 0);
                pp += 48;
                p0 += bottom_blob.cstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __lsx_vst(__lsx_vld(p0, 0), pp, 0);
                __lsx_vstelm_d(__lsx_vld(p0 + 8, 0), pp + 8, 0, 0);
                pp += 12;
                p0 += bottom_blob.cstep;
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = bfloat2float_lasx((__m128i)__lsx_vld(p0, 0));
                __m256 _r1 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 8, 0));
                __m256 _r2 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 16, 0));
                __m256 _r3 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 24, 0));
                __m256 _r4 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 32, 0));
                __m256 _r5 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 40, 0));
                __m256 _r6 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 48, 0));
                __m256 _r7 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 56, 0));
                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                __lsx_vst(float2bfloat_lasx(_r0), pp, 0);
                __lsx_vst(float2bfloat_lasx(_r1), pp + 8, 0);
                __lsx_vst(float2bfloat_lasx(_r2), pp + 16, 0);
                __lsx_vst(float2bfloat_lasx(_r3), pp + 24, 0);
                __lsx_vst(float2bfloat_lasx(_r4), pp + 32, 0);
                __lsx_vst(float2bfloat_lasx(_r5), pp + 40, 0);
                __lsx_vst(float2bfloat_lasx(_r6), pp + 48, 0);
                __lsx_vst(float2bfloat_lasx(_r7), pp + 56, 0);
                pp += 64;
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __loongarch_asx
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = bfloat2float_lsx((__m128i)__lsx_vld(p0, 0));
                __m128 _r1 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 4, 0));
                __m128 _r2 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 8, 0));
                __m128 _r3 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 12, 0));
                __m128 _r4 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 16, 0));
                __m128 _r5 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 20, 0));
                __m128 _r6 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 24, 0));
                __m128 _r7 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 28, 0));
                transpose4x4_ps(_r0, _r1, _r2, _r3);
                transpose4x4_ps(_r4, _r5, _r6, _r7);
                __lsx_vstelm_d(float2bfloat_lsx(_r0, _r0), pp, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r4, _r4), pp + 4, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r1, _r1), pp + 8, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r5, _r5), pp + 12, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r2, _r2), pp + 16, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r6, _r6), pp + 20, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r3, _r3), pp + 24, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r7, _r7), pp + 28, 0, 0);
                pp += 32;
                p0 += bottom_blob.cstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __lsx_vst(__lsx_vld(p0, 0), pp, 0);
                pp += 8;
                p0 += bottom_blob.cstep;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = bfloat2float_lasx((__m128i)__lsx_vld(p0, 0));
                __m256 _r1 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 8, 0));
                __m256 _r2 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 16, 0));
                __m256 _r3 = bfloat2float_lasx((__m128i)__lsx_vld(p0 + 24, 0));
                transpose8x4_ps(_r0, _r1, _r2, _r3);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_r0), (__m128)__lasx_extract_lo128((__m256i)_r0)), pp, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_r0), (__m128)__lasx_extract_hi128((__m256i)_r0)), pp + 4, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_r1), (__m128)__lasx_extract_lo128((__m256i)_r1)), pp + 8, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_r1), (__m128)__lasx_extract_hi128((__m256i)_r1)), pp + 12, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_r2), (__m128)__lasx_extract_lo128((__m256i)_r2)), pp + 16, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_r2), (__m128)__lasx_extract_hi128((__m256i)_r2)), pp + 20, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_r3), (__m128)__lasx_extract_lo128((__m256i)_r3)), pp + 24, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_r3), (__m128)__lasx_extract_hi128((__m256i)_r3)), pp + 28, 0, 0);
                pp += 32;
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __loongarch_asx
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = bfloat2float_lsx((__m128i)__lsx_vld(p0, 0));
                __m128 _r1 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 4, 0));
                __m128 _r2 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 8, 0));
                __m128 _r3 = bfloat2float_lsx((__m128i)__lsx_vld(p0 + 12, 0));
                transpose4x4_ps(_r0, _r1, _r2, _r3);
                __lsx_vstelm_d(float2bfloat_lsx(_r0, _r0), pp, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r1, _r1), pp + 4, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r2, _r2), pp + 8, 0, 0);
                __lsx_vstelm_d(float2bfloat_lsx(_r3, _r3), pp + 12, 0, 0);
                pp += 16;
                p0 += bottom_blob.cstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __lsx_vstelm_d(__lsx_vld(p0, 0), pp, 0, 0);
                pp += 4;
                p0 += bottom_blob.cstep;
            }
        }
    }
#endif // __loongarch_sx
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                // 2x8 -> 8x2 transpose in bf16 domain
                __m128i _r0 = __lsx_vld(p0, 0);
                __m128i _r1 = __lsx_vld(p0 + 8, 0);
                __m128i _t0 = __lsx_vilvl_h(_r1, _r0);
                __m128i _t1 = __lsx_vilvh_h(_r1, _r0);
                __lsx_vst(_t0, pp, 0);
                __lsx_vst(_t1, pp + 8, 0);
                pp += 16;
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __loongarch_asx
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // 2x4 -> 4x2 transpose in bf16
                __m128i _r0 = __lsx_vld(p0, 0);                                                   // 8 bf16 values: j0c0 j0c1 j0c2 j0c3 j1c0 j1c1 j1c2 j1c3
                __m128i _t0 = __lsx_vilvl_h(__lsx_vshuf4i_h(_r0, _LSX_SHUFFLE(1, 0, 3, 2)), _r0); // interleave to c0j0 c0j1 c1j0 c1j1
                // Actually just do scalar for 2x4
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp[2] = p0[1];
                pp[3] = p0[5];
                pp[4] = p0[2];
                pp[5] = p0[6];
                pp[6] = p0[3];
                pp[7] = p0[7];
                pp += 8;
                p0 += bottom_blob.cstep * 4;
            }
        }
#endif // __loongarch_sx
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += bottom_blob.cstep;
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __lsx_vst(__lsx_vld(p0, 0), pp, 0);
                pp += 8;
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __loongarch_asx
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __lsx_vstelm_d(__lsx_vld(p0, 0), pp, 0, 0);
                pp += 4;
                p0 += bottom_blob.cstep * 4;
            }
        }
#endif // __loongarch_sx
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += bottom_blob.cstep;
            }
        }
    }
}

static inline void convolution_im2col_input_tile_impl_bf16s(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int outw = (w - kernel_extent_w) / stride_w + 1;

    const int maxk = kernel_w * kernel_h;

    unsigned short* pp = B;

    int jj = 0;
#if __loongarch_sx
    for (; jj + 11 < max_jj; jj += 12)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dy4 = (j + jj + 4) / outw;
        int dy5 = (j + jj + 5) / outw;
        int dy6 = (j + jj + 6) / outw;
        int dy7 = (j + jj + 7) / outw;
        int dy8 = (j + jj + 8) / outw;
        int dy9 = (j + jj + 9) / outw;
        int dya = (j + jj + 10) / outw;
        int dyb = (j + jj + 11) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;
        int dx4 = (j + jj + 4) % outw;
        int dx5 = (j + jj + 5) % outw;
        int dx6 = (j + jj + 6) % outw;
        int dx7 = (j + jj + 7) % outw;
        int dx8 = (j + jj + 8) % outw;
        int dx9 = (j + jj + 9) % outw;
        int dxa = (j + jj + 10) % outw;
        int dxb = (j + jj + 11) % outw;

        if (dy0 == dyb)
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const unsigned short* sptr = (const unsigned short*)img.row(y0) + x0 * elempack;

#if __loongarch_asx
                if (elempack == 8)
                {
                    __m256 _r0 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 0, 0));
                    __m256 _r1 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 8, 0));
                    __m256 _r2 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 16, 0));
                    __m256 _r3 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 24, 0));
                    __m256 _r4 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 32, 0));
                    __m256 _r5 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 40, 0));
                    __m256 _r6 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 48, 0));
                    __m256 _r7 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 56, 0));
                    __m256 _r8 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 64, 0));
                    __m256 _r9 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 72, 0));
                    __m256 _ra = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 80, 0));
                    __m256 _rb = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 88, 0));
                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                    transpose8x4_ps(_r8, _r9, _ra, _rb);
                    __lsx_vst(float2bfloat_lasx(_r0), pp + 12 * 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_r8), (__m128)__lasx_extract_lo128((__m256i)_r8)), pp + 12 * 0 + 8, 0, 0);
                    __lsx_vst(float2bfloat_lasx(_r1), pp + 12 * 1, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_r8), (__m128)__lasx_extract_hi128((__m256i)_r8)), pp + 12 * 1 + 8, 0, 0);
                    __lsx_vst(float2bfloat_lasx(_r2), pp + 12 * 2, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_r9), (__m128)__lasx_extract_lo128((__m256i)_r9)), pp + 12 * 2 + 8, 0, 0);
                    __lsx_vst(float2bfloat_lasx(_r3), pp + 12 * 3, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_r9), (__m128)__lasx_extract_hi128((__m256i)_r9)), pp + 12 * 3 + 8, 0, 0);
                    __lsx_vst(float2bfloat_lasx(_r4), pp + 12 * 4, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_ra), (__m128)__lasx_extract_lo128((__m256i)_ra)), pp + 12 * 4 + 8, 0, 0);
                    __lsx_vst(float2bfloat_lasx(_r5), pp + 12 * 5, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_ra), (__m128)__lasx_extract_hi128((__m256i)_ra)), pp + 12 * 5 + 8, 0, 0);
                    __lsx_vst(float2bfloat_lasx(_r6), pp + 12 * 6, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_rb), (__m128)__lasx_extract_lo128((__m256i)_rb)), pp + 12 * 6 + 8, 0, 0);
                    __lsx_vst(float2bfloat_lasx(_r7), pp + 12 * 7, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_rb), (__m128)__lasx_extract_hi128((__m256i)_rb)), pp + 12 * 7 + 8, 0, 0);
                    pp += 96;
                }
#endif // __loongarch_asx
                if (elempack == 4)
                {
                    __m128 _r0 = bfloat2float_lsx(sptr + stride_w * 0);
                    __m128 _r1 = bfloat2float_lsx(sptr + stride_w * 4);
                    __m128 _r2 = bfloat2float_lsx(sptr + stride_w * 8);
                    __m128 _r3 = bfloat2float_lsx(sptr + stride_w * 12);
                    __m128 _r4 = bfloat2float_lsx(sptr + stride_w * 16);
                    __m128 _r5 = bfloat2float_lsx(sptr + stride_w * 20);
                    __m128 _r6 = bfloat2float_lsx(sptr + stride_w * 24);
                    __m128 _r7 = bfloat2float_lsx(sptr + stride_w * 28);
                    __m128 _r8 = bfloat2float_lsx(sptr + stride_w * 32);
                    __m128 _r9 = bfloat2float_lsx(sptr + stride_w * 36);
                    __m128 _ra = bfloat2float_lsx(sptr + stride_w * 40);
                    __m128 _rb = bfloat2float_lsx(sptr + stride_w * 44);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    transpose4x4_ps(_r8, _r9, _ra, _rb);
                    __lsx_vstelm_d(float2bfloat_lsx(_r0, _r0), pp + 4 * 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r4, _r4), pp + 4 * 1, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r8, _r8), pp + 4 * 2, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r1, _r1), pp + 4 * 3, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r5, _r5), pp + 4 * 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r9, _r9), pp + 4 * 5, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r2, _r2), pp + 4 * 6, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r6, _r6), pp + 4 * 7, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_ra, _ra), pp + 4 * 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r3, _r3), pp + 4 * 9, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r7, _r7), pp + 4 * 10, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_rb, _rb), pp + 4 * 11, 0, 0);
                    pp += 48;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w * 1];
                    pp[2] = sptr[stride_w * 2];
                    pp[3] = sptr[stride_w * 3];
                    pp[4] = sptr[stride_w * 4];
                    pp[5] = sptr[stride_w * 5];
                    pp[6] = sptr[stride_w * 6];
                    pp[7] = sptr[stride_w * 7];
                    pp[8] = sptr[stride_w * 8];
                    pp[9] = sptr[stride_w * 9];
                    pp[10] = sptr[stride_w * 10];
                    pp[11] = sptr[stride_w * 11];
                    pp += 12;
                }
            }
        }
        else
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int x2 = stride_w * dx2 + dilation_w * v;
                int x3 = stride_w * dx3 + dilation_w * v;
                int x4 = stride_w * dx4 + dilation_w * v;
                int x5 = stride_w * dx5 + dilation_w * v;
                int x6 = stride_w * dx6 + dilation_w * v;
                int x7 = stride_w * dx7 + dilation_w * v;
                int x8 = stride_w * dx8 + dilation_w * v;
                int x9 = stride_w * dx9 + dilation_w * v;
                int xa = stride_w * dxa + dilation_w * v;
                int xb = stride_w * dxb + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;
                int y4 = stride_h * dy4 + dilation_h * u;
                int y5 = stride_h * dy5 + dilation_h * u;
                int y6 = stride_h * dy6 + dilation_h * u;
                int y7 = stride_h * dy7 + dilation_h * u;
                int y8 = stride_h * dy8 + dilation_h * u;
                int y9 = stride_h * dy9 + dilation_h * u;
                int ya = stride_h * dya + dilation_h * u;
                int yb = stride_h * dyb + dilation_h * u;

                const unsigned short* sptr0 = (const unsigned short*)img.row(y0) + x0 * elempack;
                const unsigned short* sptr1 = (const unsigned short*)img.row(y1) + x1 * elempack;
                const unsigned short* sptr2 = (const unsigned short*)img.row(y2) + x2 * elempack;
                const unsigned short* sptr3 = (const unsigned short*)img.row(y3) + x3 * elempack;
                const unsigned short* sptr4 = (const unsigned short*)img.row(y4) + x4 * elempack;
                const unsigned short* sptr5 = (const unsigned short*)img.row(y5) + x5 * elempack;
                const unsigned short* sptr6 = (const unsigned short*)img.row(y6) + x6 * elempack;
                const unsigned short* sptr7 = (const unsigned short*)img.row(y7) + x7 * elempack;
                const unsigned short* sptr8 = (const unsigned short*)img.row(y8) + x8 * elempack;
                const unsigned short* sptr9 = (const unsigned short*)img.row(y9) + x9 * elempack;
                const unsigned short* sptra = (const unsigned short*)img.row(ya) + xa * elempack;
                const unsigned short* sptrb = (const unsigned short*)img.row(yb) + xb * elempack;

#if __loongarch_asx
                if (elempack == 8)
                {
                    __m256 _r0 = bfloat2float_lasx((__m128i)__lsx_vld(sptr0, 0));
                    __m256 _r1 = bfloat2float_lasx((__m128i)__lsx_vld(sptr1, 0));
                    __m256 _r2 = bfloat2float_lasx((__m128i)__lsx_vld(sptr2, 0));
                    __m256 _r3 = bfloat2float_lasx((__m128i)__lsx_vld(sptr3, 0));
                    __m256 _r4 = bfloat2float_lasx((__m128i)__lsx_vld(sptr4, 0));
                    __m256 _r5 = bfloat2float_lasx((__m128i)__lsx_vld(sptr5, 0));
                    __m256 _r6 = bfloat2float_lasx((__m128i)__lsx_vld(sptr6, 0));
                    __m256 _r7 = bfloat2float_lasx((__m128i)__lsx_vld(sptr7, 0));
                    __m256 _r8 = bfloat2float_lasx((__m128i)__lsx_vld(sptr8, 0));
                    __m256 _r9 = bfloat2float_lasx((__m128i)__lsx_vld(sptr9, 0));
                    __m256 _ra = bfloat2float_lasx((__m128i)__lsx_vld(sptra, 0));
                    __m256 _rb = bfloat2float_lasx((__m128i)__lsx_vld(sptrb, 0));
                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                    transpose8x4_ps(_r8, _r9, _ra, _rb);
                    __lsx_vst(float2bfloat_lasx(_r0), pp + 12 * 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_r8), (__m128)__lasx_extract_lo128((__m256i)_r8)), pp + 12 * 0 + 8, 0, 0);
                    __lsx_vst(float2bfloat_lasx(_r1), pp + 12 * 1, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_r8), (__m128)__lasx_extract_hi128((__m256i)_r8)), pp + 12 * 1 + 8, 0, 0);
                    __lsx_vst(float2bfloat_lasx(_r2), pp + 12 * 2, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_r9), (__m128)__lasx_extract_lo128((__m256i)_r9)), pp + 12 * 2 + 8, 0, 0);
                    __lsx_vst(float2bfloat_lasx(_r3), pp + 12 * 3, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_r9), (__m128)__lasx_extract_hi128((__m256i)_r9)), pp + 12 * 3 + 8, 0, 0);
                    __lsx_vst(float2bfloat_lasx(_r4), pp + 12 * 4, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_ra), (__m128)__lasx_extract_lo128((__m256i)_ra)), pp + 12 * 4 + 8, 0, 0);
                    __lsx_vst(float2bfloat_lasx(_r5), pp + 12 * 5, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_ra), (__m128)__lasx_extract_hi128((__m256i)_ra)), pp + 12 * 5 + 8, 0, 0);
                    __lsx_vst(float2bfloat_lasx(_r6), pp + 12 * 6, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_rb), (__m128)__lasx_extract_lo128((__m256i)_rb)), pp + 12 * 6 + 8, 0, 0);
                    __lsx_vst(float2bfloat_lasx(_r7), pp + 12 * 7, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_rb), (__m128)__lasx_extract_hi128((__m256i)_rb)), pp + 12 * 7 + 8, 0, 0);
                    pp += 96;
                }
#endif // __loongarch_asx
                if (elempack == 4)
                {
                    __m128 _r0 = bfloat2float_lsx(sptr0);
                    __m128 _r1 = bfloat2float_lsx(sptr1);
                    __m128 _r2 = bfloat2float_lsx(sptr2);
                    __m128 _r3 = bfloat2float_lsx(sptr3);
                    __m128 _r4 = bfloat2float_lsx(sptr4);
                    __m128 _r5 = bfloat2float_lsx(sptr5);
                    __m128 _r6 = bfloat2float_lsx(sptr6);
                    __m128 _r7 = bfloat2float_lsx(sptr7);
                    __m128 _r8 = bfloat2float_lsx(sptr8);
                    __m128 _r9 = bfloat2float_lsx(sptr9);
                    __m128 _ra = bfloat2float_lsx(sptra);
                    __m128 _rb = bfloat2float_lsx(sptrb);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    transpose4x4_ps(_r8, _r9, _ra, _rb);
                    __lsx_vstelm_d(float2bfloat_lsx(_r0, _r0), pp + 4 * 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r4, _r4), pp + 4 * 1, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r8, _r8), pp + 4 * 2, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r1, _r1), pp + 4 * 3, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r5, _r5), pp + 4 * 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r9, _r9), pp + 4 * 5, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r2, _r2), pp + 4 * 6, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r6, _r6), pp + 4 * 7, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_ra, _ra), pp + 4 * 8, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r3, _r3), pp + 4 * 9, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r7, _r7), pp + 4 * 10, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_rb, _rb), pp + 4 * 11, 0, 0);
                    pp += 48;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp[4] = sptr4[0];
                    pp[5] = sptr5[0];
                    pp[6] = sptr6[0];
                    pp[7] = sptr7[0];
                    pp[8] = sptr8[0];
                    pp[9] = sptr9[0];
                    pp[10] = sptra[0];
                    pp[11] = sptrb[0];
                    pp += 12;
                }
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dy4 = (j + jj + 4) / outw;
        int dy5 = (j + jj + 5) / outw;
        int dy6 = (j + jj + 6) / outw;
        int dy7 = (j + jj + 7) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;
        int dx4 = (j + jj + 4) % outw;
        int dx5 = (j + jj + 5) % outw;
        int dx6 = (j + jj + 6) % outw;
        int dx7 = (j + jj + 7) % outw;

        if (dy0 == dy7)
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const unsigned short* sptr = (const unsigned short*)img.row(y0) + x0 * elempack;

#if __loongarch_asx
                if (elempack == 8)
                {
                    __m256 _r0 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 0, 0));
                    __m256 _r1 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 8, 0));
                    __m256 _r2 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 16, 0));
                    __m256 _r3 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 24, 0));
                    __m256 _r4 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 32, 0));
                    __m256 _r5 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 40, 0));
                    __m256 _r6 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 48, 0));
                    __m256 _r7 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 56, 0));
                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                    __lsx_vst(float2bfloat_lasx(_r0), pp + 8 * 0, 0);
                    __lsx_vst(float2bfloat_lasx(_r1), pp + 8 * 1, 0);
                    __lsx_vst(float2bfloat_lasx(_r2), pp + 8 * 2, 0);
                    __lsx_vst(float2bfloat_lasx(_r3), pp + 8 * 3, 0);
                    __lsx_vst(float2bfloat_lasx(_r4), pp + 8 * 4, 0);
                    __lsx_vst(float2bfloat_lasx(_r5), pp + 8 * 5, 0);
                    __lsx_vst(float2bfloat_lasx(_r6), pp + 8 * 6, 0);
                    __lsx_vst(float2bfloat_lasx(_r7), pp + 8 * 7, 0);
                    pp += 64;
                }
#endif // __loongarch_asx
                if (elempack == 4)
                {
                    __m128 _r0 = bfloat2float_lsx(sptr + stride_w * 0);
                    __m128 _r1 = bfloat2float_lsx(sptr + stride_w * 4);
                    __m128 _r2 = bfloat2float_lsx(sptr + stride_w * 8);
                    __m128 _r3 = bfloat2float_lsx(sptr + stride_w * 12);
                    __m128 _r4 = bfloat2float_lsx(sptr + stride_w * 16);
                    __m128 _r5 = bfloat2float_lsx(sptr + stride_w * 20);
                    __m128 _r6 = bfloat2float_lsx(sptr + stride_w * 24);
                    __m128 _r7 = bfloat2float_lsx(sptr + stride_w * 28);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    __lsx_vstelm_d(float2bfloat_lsx(_r0, _r0), pp + 4 * 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r4, _r4), pp + 4 * 1, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r1, _r1), pp + 4 * 2, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r5, _r5), pp + 4 * 3, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r2, _r2), pp + 4 * 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r6, _r6), pp + 4 * 5, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r3, _r3), pp + 4 * 6, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r7, _r7), pp + 4 * 7, 0, 0);
                    pp += 32;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w * 1];
                    pp[2] = sptr[stride_w * 2];
                    pp[3] = sptr[stride_w * 3];
                    pp[4] = sptr[stride_w * 4];
                    pp[5] = sptr[stride_w * 5];
                    pp[6] = sptr[stride_w * 6];
                    pp[7] = sptr[stride_w * 7];
                    pp += 8;
                }
            }
        }
        else
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int x2 = stride_w * dx2 + dilation_w * v;
                int x3 = stride_w * dx3 + dilation_w * v;
                int x4 = stride_w * dx4 + dilation_w * v;
                int x5 = stride_w * dx5 + dilation_w * v;
                int x6 = stride_w * dx6 + dilation_w * v;
                int x7 = stride_w * dx7 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;
                int y4 = stride_h * dy4 + dilation_h * u;
                int y5 = stride_h * dy5 + dilation_h * u;
                int y6 = stride_h * dy6 + dilation_h * u;
                int y7 = stride_h * dy7 + dilation_h * u;

                const unsigned short* sptr0 = (const unsigned short*)img.row(y0) + x0 * elempack;
                const unsigned short* sptr1 = (const unsigned short*)img.row(y1) + x1 * elempack;
                const unsigned short* sptr2 = (const unsigned short*)img.row(y2) + x2 * elempack;
                const unsigned short* sptr3 = (const unsigned short*)img.row(y3) + x3 * elempack;
                const unsigned short* sptr4 = (const unsigned short*)img.row(y4) + x4 * elempack;
                const unsigned short* sptr5 = (const unsigned short*)img.row(y5) + x5 * elempack;
                const unsigned short* sptr6 = (const unsigned short*)img.row(y6) + x6 * elempack;
                const unsigned short* sptr7 = (const unsigned short*)img.row(y7) + x7 * elempack;

#if __loongarch_asx
                if (elempack == 8)
                {
                    __m256 _r0 = bfloat2float_lasx((__m128i)__lsx_vld(sptr0, 0));
                    __m256 _r1 = bfloat2float_lasx((__m128i)__lsx_vld(sptr1, 0));
                    __m256 _r2 = bfloat2float_lasx((__m128i)__lsx_vld(sptr2, 0));
                    __m256 _r3 = bfloat2float_lasx((__m128i)__lsx_vld(sptr3, 0));
                    __m256 _r4 = bfloat2float_lasx((__m128i)__lsx_vld(sptr4, 0));
                    __m256 _r5 = bfloat2float_lasx((__m128i)__lsx_vld(sptr5, 0));
                    __m256 _r6 = bfloat2float_lasx((__m128i)__lsx_vld(sptr6, 0));
                    __m256 _r7 = bfloat2float_lasx((__m128i)__lsx_vld(sptr7, 0));
                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                    __lsx_vst(float2bfloat_lasx(_r0), pp + 8 * 0, 0);
                    __lsx_vst(float2bfloat_lasx(_r1), pp + 8 * 1, 0);
                    __lsx_vst(float2bfloat_lasx(_r2), pp + 8 * 2, 0);
                    __lsx_vst(float2bfloat_lasx(_r3), pp + 8 * 3, 0);
                    __lsx_vst(float2bfloat_lasx(_r4), pp + 8 * 4, 0);
                    __lsx_vst(float2bfloat_lasx(_r5), pp + 8 * 5, 0);
                    __lsx_vst(float2bfloat_lasx(_r6), pp + 8 * 6, 0);
                    __lsx_vst(float2bfloat_lasx(_r7), pp + 8 * 7, 0);
                    pp += 64;
                }
#endif // __loongarch_asx
                if (elempack == 4)
                {
                    __m128 _r0 = bfloat2float_lsx(sptr0);
                    __m128 _r1 = bfloat2float_lsx(sptr1);
                    __m128 _r2 = bfloat2float_lsx(sptr2);
                    __m128 _r3 = bfloat2float_lsx(sptr3);
                    __m128 _r4 = bfloat2float_lsx(sptr4);
                    __m128 _r5 = bfloat2float_lsx(sptr5);
                    __m128 _r6 = bfloat2float_lsx(sptr6);
                    __m128 _r7 = bfloat2float_lsx(sptr7);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    __lsx_vstelm_d(float2bfloat_lsx(_r0, _r0), pp + 4 * 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r4, _r4), pp + 4 * 1, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r1, _r1), pp + 4 * 2, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r5, _r5), pp + 4 * 3, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r2, _r2), pp + 4 * 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r6, _r6), pp + 4 * 5, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r3, _r3), pp + 4 * 6, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r7, _r7), pp + 4 * 7, 0, 0);
                    pp += 32;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp[4] = sptr4[0];
                    pp[5] = sptr5[0];
                    pp[6] = sptr6[0];
                    pp[7] = sptr7[0];
                    pp += 8;
                }
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;

        if (dy0 == dy3)
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const unsigned short* sptr = (const unsigned short*)img.row(y0) + x0 * elempack;

#if __loongarch_asx
                if (elempack == 8)
                {
                    __m256 _r0 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 0, 0));
                    __m256 _r1 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 8, 0));
                    __m256 _r2 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 16, 0));
                    __m256 _r3 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 24, 0));
                    transpose8x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_r0), (__m128)__lasx_extract_lo128((__m256i)_r0)), pp + 4 * 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_r0), (__m128)__lasx_extract_hi128((__m256i)_r0)), pp + 4 * 1, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_r1), (__m128)__lasx_extract_lo128((__m256i)_r1)), pp + 4 * 2, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_r1), (__m128)__lasx_extract_hi128((__m256i)_r1)), pp + 4 * 3, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_r2), (__m128)__lasx_extract_lo128((__m256i)_r2)), pp + 4 * 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_r2), (__m128)__lasx_extract_hi128((__m256i)_r2)), pp + 4 * 5, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_r3), (__m128)__lasx_extract_lo128((__m256i)_r3)), pp + 4 * 6, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_r3), (__m128)__lasx_extract_hi128((__m256i)_r3)), pp + 4 * 7, 0, 0);
                    pp += 32;
                }
#endif // __loongarch_asx
                if (elempack == 4)
                {
                    __m128 _r0 = bfloat2float_lsx(sptr + stride_w * 0);
                    __m128 _r1 = bfloat2float_lsx(sptr + stride_w * 4);
                    __m128 _r2 = bfloat2float_lsx(sptr + stride_w * 8);
                    __m128 _r3 = bfloat2float_lsx(sptr + stride_w * 12);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vstelm_d(float2bfloat_lsx(_r0, _r0), pp + 4 * 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r1, _r1), pp + 4 * 1, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r2, _r2), pp + 4 * 2, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r3, _r3), pp + 4 * 3, 0, 0);
                    pp += 16;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w * 1];
                    pp[2] = sptr[stride_w * 2];
                    pp[3] = sptr[stride_w * 3];
                    pp += 4;
                }
            }
        }
        else
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int x2 = stride_w * dx2 + dilation_w * v;
                int x3 = stride_w * dx3 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;

                const unsigned short* sptr0 = (const unsigned short*)img.row(y0) + x0 * elempack;
                const unsigned short* sptr1 = (const unsigned short*)img.row(y1) + x1 * elempack;
                const unsigned short* sptr2 = (const unsigned short*)img.row(y2) + x2 * elempack;
                const unsigned short* sptr3 = (const unsigned short*)img.row(y3) + x3 * elempack;

#if __loongarch_asx
                if (elempack == 8)
                {
                    __m256 _r0 = bfloat2float_lasx((__m128i)__lsx_vld(sptr0, 0));
                    __m256 _r1 = bfloat2float_lasx((__m128i)__lsx_vld(sptr1, 0));
                    __m256 _r2 = bfloat2float_lasx((__m128i)__lsx_vld(sptr2, 0));
                    __m256 _r3 = bfloat2float_lasx((__m128i)__lsx_vld(sptr3, 0));
                    transpose8x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_r0), (__m128)__lasx_extract_lo128((__m256i)_r0)), pp + 4 * 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_r0), (__m128)__lasx_extract_hi128((__m256i)_r0)), pp + 4 * 1, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_r1), (__m128)__lasx_extract_lo128((__m256i)_r1)), pp + 4 * 2, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_r1), (__m128)__lasx_extract_hi128((__m256i)_r1)), pp + 4 * 3, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_r2), (__m128)__lasx_extract_lo128((__m256i)_r2)), pp + 4 * 4, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_r2), (__m128)__lasx_extract_hi128((__m256i)_r2)), pp + 4 * 5, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_lo128((__m256i)_r3), (__m128)__lasx_extract_lo128((__m256i)_r3)), pp + 4 * 6, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx((__m128)__lasx_extract_hi128((__m256i)_r3), (__m128)__lasx_extract_hi128((__m256i)_r3)), pp + 4 * 7, 0, 0);
                    pp += 32;
                }
#endif // __loongarch_asx
                if (elempack == 4)
                {
                    __m128 _r0 = bfloat2float_lsx(sptr0);
                    __m128 _r1 = bfloat2float_lsx(sptr1);
                    __m128 _r2 = bfloat2float_lsx(sptr2);
                    __m128 _r3 = bfloat2float_lsx(sptr3);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vstelm_d(float2bfloat_lsx(_r0, _r0), pp + 4 * 0, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r1, _r1), pp + 4 * 1, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r2, _r2), pp + 4 * 2, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_r3, _r3), pp + 4 * 3, 0, 0);
                    pp += 16;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp += 4;
                }
            }
        }
    }
#endif // __loongarch_sx
    for (; jj + 1 < max_jj; jj += 2)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;

        if (dy0 == dy1)
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const unsigned short* sptr = (const unsigned short*)img.row(y0) + x0 * elempack;

#if __loongarch_sx
#if __loongarch_asx
                if (elempack == 8)
                {
                    __m256 _r0 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 0, 0));
                    __m256 _r1 = bfloat2float_lasx((__m128i)__lsx_vld(sptr + stride_w * 8, 0));
                    transpose8x2_ps(_r0, _r1);
                    __lsx_vst(float2bfloat_lasx(_r0), pp, 0);
                    __lsx_vst(float2bfloat_lasx(_r1), pp + 8, 0);
                    pp += 16;
                }
#endif // __loongarch_asx
                if (elempack == 4)
                {
                    __m128 _r0 = bfloat2float_lsx(sptr + stride_w * 0);
                    __m128 _r1 = bfloat2float_lsx(sptr + stride_w * 4);
                    __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_r1, (__m128i)_r0);
                    __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_r1, (__m128i)_r0);
                    __lsx_vstelm_d(float2bfloat_lsx(_tmp0, _tmp0), pp, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_tmp1, _tmp1), pp + 4, 0, 0);
                    pp += 8;
                }
#endif // __loongarch_sx
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w * 1];
                    pp += 2;
                }
            }
        }
        else
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;

                const unsigned short* sptr0 = (const unsigned short*)img.row(y0) + x0 * elempack;
                const unsigned short* sptr1 = (const unsigned short*)img.row(y1) + x1 * elempack;

#if __loongarch_sx
#if __loongarch_asx
                if (elempack == 8)
                {
                    __m256 _r0 = bfloat2float_lasx((__m128i)__lsx_vld(sptr0, 0));
                    __m256 _r1 = bfloat2float_lasx((__m128i)__lsx_vld(sptr1, 0));
                    transpose8x2_ps(_r0, _r1);
                    __lsx_vst(float2bfloat_lasx(_r0), pp, 0);
                    __lsx_vst(float2bfloat_lasx(_r1), pp + 8, 0);
                    pp += 16;
                }
#endif // __loongarch_asx
                if (elempack == 4)
                {
                    __m128 _r0 = bfloat2float_lsx(sptr0);
                    __m128 _r1 = bfloat2float_lsx(sptr1);
                    __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_r1, (__m128i)_r0);
                    __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_r1, (__m128i)_r0);
                    __lsx_vstelm_d(float2bfloat_lsx(_tmp0, _tmp0), pp, 0, 0);
                    __lsx_vstelm_d(float2bfloat_lsx(_tmp1, _tmp1), pp + 4, 0, 0);
                    pp += 8;
                }
#endif // __loongarch_sx
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp += 2;
                }
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        int dy0 = (j + jj) / outw;
        int dx0 = (j + jj) % outw;

        int kk = 0;
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x0 = stride_w * dx0 + dilation_w * v;
            int y0 = stride_h * dy0 + dilation_h * u;

            const unsigned short* sptr = (const unsigned short*)img.row(y0) + x0 * elempack;

#if __loongarch_sx
#if __loongarch_asx
            if (elempack == 8)
            {
                __lsx_vst(__lsx_vld(sptr, 0), pp, 0);
                pp += 8;
            }
#endif // __loongarch_asx
            if (elempack == 4)
            {
                __lsx_vstelm_d(__lsx_vld(sptr, 0), pp, 0, 0);
                pp += 4;
            }
#endif // __loongarch_sx
            if (elempack == 1)
            {
                pp[0] = sptr[0];
                pp += 1;
            }
        }
    }
}

template<int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h>
static void convolution_im2col_input_tile_lsx_bf16s(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    convolution_im2col_input_tile_impl_bf16s(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

template void convolution_im2col_input_tile_lsx_bf16s<1, 1, 1, 1, 1, 1>(const Mat&, Mat&, int, int, int, int);
template void convolution_im2col_input_tile_lsx_bf16s<1, 1, 1, 1, 2, 2>(const Mat&, Mat&, int, int, int, int);
template void convolution_im2col_input_tile_lsx_bf16s<2, 2, 1, 1, 1, 1>(const Mat&, Mat&, int, int, int, int);
template void convolution_im2col_input_tile_lsx_bf16s<3, 3, 1, 1, 1, 1>(const Mat&, Mat&, int, int, int, int);
template void convolution_im2col_input_tile_lsx_bf16s<3, 3, 1, 1, 2, 2>(const Mat&, Mat&, int, int, int, int);
template void convolution_im2col_input_tile_lsx_bf16s<5, 5, 1, 1, 1, 1>(const Mat&, Mat&, int, int, int, int);
template void convolution_im2col_input_tile_lsx_bf16s<5, 5, 1, 1, 2, 2>(const Mat&, Mat&, int, int, int, int);
template void convolution_im2col_input_tile_lsx_bf16s<7, 7, 1, 1, 2, 2>(const Mat&, Mat&, int, int, int, int);

static void convolution_im2col_input_tile_bf16s(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_lsx_bf16s<1, 1, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_lsx_bf16s<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }
    if (kernel_w == 2 && kernel_h == 2 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_lsx_bf16s<2, 2, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }
    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_lsx_bf16s<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }
    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_lsx_bf16s<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }
    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_lsx_bf16s<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }
    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_lsx_bf16s<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }
    if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_lsx_bf16s<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    convolution_im2col_input_tile_impl_bf16s(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

static void convolution_im2col_gemm_transform_kernel_bf16s(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_bf16s(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    int elempack = 1;
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
#if __loongarch_asx
        elempack = inch % 8 == 0 ? 8 : inch % 4 == 0 ? 4 : 1;
#else
        elempack = inch % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __loongarch_sx

    // maxk-inch-outch to pa-maxk-inch/pa-outch
    Mat A_data;
    if (maxk == 1)
    {
        A_data = kernel.reshape(maxk * inch, outch);
    }
    else
    {
        Mat weight_data_r2 = kernel.reshape(maxk, inch, outch);

        A_data.create(maxk * inch, outch);

        for (int q = 0; q < outch; q += 1)
        {
            float* g00 = A_data.row(q);

            for (int p = 0; p + (elempack - 1) < inch; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        const float* k00 = weight_data_r2.channel(q).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }

    AT.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, 2u, 1);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            convolution_im2col_pack_A_tile_bf16s(A_data, AT_tile, i, max_ii, k, max_kk);
        }
    }
}

static int convolution_im2col_gemm_bf16s(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, int nT, const Option& opt)
{
    const int maxk = kernel_w * kernel_h;

    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_bf16s(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;

    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min((N - j), TILE_N);
        const int max_kk = std::min((K - k), TILE_K);

        Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

        // im2col
        convolution_im2col_input_tile_bf16s(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
    }

    Mat topT_tileX;
    if (K > TILE_K)
    {
        topT_tileX.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT_tileX.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat topT_tile;
        if (K > TILE_K)
            topT_tile = topT_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                const Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = k + TILE_K >= K;

                convolution_gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, bias, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end, activation_type, activation_params);
            }
        }
    }

    return 0;
}

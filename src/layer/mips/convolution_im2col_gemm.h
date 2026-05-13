// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convolution_im2col_pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    // A = (pa, maxk, inch/pa), outch
    const int A_hstep = A.w;

    float* pp = AT;

    int ii = 0;
#if __mips_msa
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
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + 16);
            __builtin_prefetch(p1 + 16);
            __builtin_prefetch(p2 + 16);
            __builtin_prefetch(p3 + 16);
            __builtin_prefetch(p4 + 16);
            __builtin_prefetch(p5 + 16);
            __builtin_prefetch(p6 + 16);
            __builtin_prefetch(p7 + 16);
            v4f32 _r0 = (v4f32)__msa_ld_w(p0, 0);
            v4f32 _r1 = (v4f32)__msa_ld_w(p1, 0);
            v4f32 _r2 = (v4f32)__msa_ld_w(p2, 0);
            v4f32 _r3 = (v4f32)__msa_ld_w(p3, 0);
            v4f32 _r4 = (v4f32)__msa_ld_w(p4, 0);
            v4f32 _r5 = (v4f32)__msa_ld_w(p5, 0);
            v4f32 _r6 = (v4f32)__msa_ld_w(p6, 0);
            v4f32 _r7 = (v4f32)__msa_ld_w(p7, 0);
            transpose4x4_ps(_r0, _r1, _r2, _r3);
            transpose4x4_ps(_r4, _r5, _r6, _r7);
            __msa_st_w((v4i32)_r0, pp, 0);
            __msa_st_w((v4i32)_r4, pp + 4, 0);
            __msa_st_w((v4i32)_r1, pp + 8, 0);
            __msa_st_w((v4i32)_r5, pp + 12, 0);
            __msa_st_w((v4i32)_r2, pp + 16, 0);
            __msa_st_w((v4i32)_r6, pp + 20, 0);
            __msa_st_w((v4i32)_r3, pp + 24, 0);
            __msa_st_w((v4i32)_r7, pp + 28, 0);
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
            __builtin_prefetch(p0 + 16);
            __builtin_prefetch(p1 + 16);
            __builtin_prefetch(p2 + 16);
            __builtin_prefetch(p3 + 16);
            __builtin_prefetch(p4 + 16);
            __builtin_prefetch(p5 + 16);
            __builtin_prefetch(p6 + 16);
            __builtin_prefetch(p7 + 16);
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[4] = p4[0];
            pp[5] = p5[0];
            pp[6] = p6[0];
            pp[7] = p7[0];
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
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + 16);
            __builtin_prefetch(p1 + 16);
            __builtin_prefetch(p2 + 16);
            __builtin_prefetch(p3 + 16);
            v4f32 _r0 = (v4f32)__msa_ld_w(p0, 0);
            v4f32 _r1 = (v4f32)__msa_ld_w(p1, 0);
            v4f32 _r2 = (v4f32)__msa_ld_w(p2, 0);
            v4f32 _r3 = (v4f32)__msa_ld_w(p3, 0);
            transpose4x4_ps(_r0, _r1, _r2, _r3);
            __msa_st_w((v4i32)_r0, pp, 0);
            __msa_st_w((v4i32)_r1, pp + 4, 0);
            __msa_st_w((v4i32)_r2, pp + 8, 0);
            __msa_st_w((v4i32)_r3, pp + 12, 0);
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            __builtin_prefetch(p0 + 16);
            __builtin_prefetch(p1 + 16);
            __builtin_prefetch(p2 + 16);
            __builtin_prefetch(p3 + 16);
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
#if __mips_msa
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + 16);
            __builtin_prefetch(p1 + 16);
            v4f32 _r0 = (v4f32)__msa_ld_w(p0, 0);
            v4f32 _r1 = (v4f32)__msa_ld_w(p1, 0);
            v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
            v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
            __msa_st_w((v4i32)_tmp0, pp, 0);
            __msa_st_w((v4i32)_tmp1, pp + 4, 0);
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
#endif // __mips_msa
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        int kk = 0;
#if __mips_msa
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + 16);
            __msa_st_w((v4i32)(v4f32)__msa_ld_w(p0, 0), pp, 0);
            pp += 4;
            p0 += 4;
        }
#endif // __mips_msa
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void convolution_gemm_transB_packed_tile(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    // NCNN_LOGE("convolution_gemm_transB_packed_tile %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const int out_elempack = top_blob.elempack;
    const int out_hstep = (int)top_blob.cstep;

    const float* pAT = AT_tile;
    const float* pBT = BT_tile;
    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;
        const float* pC0 = pC ? (const float*)CT_tile + i + ii : 0;

        v4f32 _bias0 = (v4f32)__msa_fill_w(0);
        v4f32 _bias1 = (v4f32)__msa_fill_w(0);
        if (pC0)
        {
            _bias0 = (v4f32)__msa_ld_w(pC0, 0);
            _bias1 = (v4f32)__msa_ld_w(pC0 + 4, 0);
        }

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            v4f32 _sum00;
            v4f32 _sum01;
            v4f32 _sum10;
            v4f32 _sum11;
            v4f32 _sum20;
            v4f32 _sum21;
            v4f32 _sum30;
            v4f32 _sum31;
            v4f32 _sum40;
            v4f32 _sum41;
            v4f32 _sum50;
            v4f32 _sum51;
            v4f32 _sum60;
            v4f32 _sum61;
            v4f32 _sum70;
            v4f32 _sum71;

            if (k == 0)
            {
                _sum00 = (v4f32)__msa_fill_w(0);
                _sum01 = (v4f32)__msa_fill_w(0);
                _sum10 = (v4f32)__msa_fill_w(0);
                _sum11 = (v4f32)__msa_fill_w(0);
                _sum20 = (v4f32)__msa_fill_w(0);
                _sum21 = (v4f32)__msa_fill_w(0);
                _sum30 = (v4f32)__msa_fill_w(0);
                _sum31 = (v4f32)__msa_fill_w(0);
                _sum40 = (v4f32)__msa_fill_w(0);
                _sum41 = (v4f32)__msa_fill_w(0);
                _sum50 = (v4f32)__msa_fill_w(0);
                _sum51 = (v4f32)__msa_fill_w(0);
                _sum60 = (v4f32)__msa_fill_w(0);
                _sum61 = (v4f32)__msa_fill_w(0);
                _sum70 = (v4f32)__msa_fill_w(0);
                _sum71 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _sum00 = (v4f32)__msa_ld_w(outptr, 0);
                _sum01 = (v4f32)__msa_ld_w(outptr + 4, 0);
                _sum10 = (v4f32)__msa_ld_w(outptr + 8, 0);
                _sum11 = (v4f32)__msa_ld_w(outptr + 12, 0);
                _sum20 = (v4f32)__msa_ld_w(outptr + 16, 0);
                _sum21 = (v4f32)__msa_ld_w(outptr + 20, 0);
                _sum30 = (v4f32)__msa_ld_w(outptr + 24, 0);
                _sum31 = (v4f32)__msa_ld_w(outptr + 28, 0);
                _sum40 = (v4f32)__msa_ld_w(outptr + 32, 0);
                _sum41 = (v4f32)__msa_ld_w(outptr + 36, 0);
                _sum50 = (v4f32)__msa_ld_w(outptr + 40, 0);
                _sum51 = (v4f32)__msa_ld_w(outptr + 44, 0);
                _sum60 = (v4f32)__msa_ld_w(outptr + 48, 0);
                _sum61 = (v4f32)__msa_ld_w(outptr + 52, 0);
                _sum70 = (v4f32)__msa_ld_w(outptr + 56, 0);
                _sum71 = (v4f32)__msa_ld_w(outptr + 60, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 16);
                v4f32 _pA0 = (v4f32)__msa_ld_w(pA, 0);
                v4f32 _pA1 = (v4f32)__msa_ld_w(pA + 4, 0);
                v4f32 _pA0r = (v4f32)__msa_shf_w((v4i32)_pA0, _MSA_SHUFFLE(1, 0, 3, 2));
                v4f32 _pA1r = (v4f32)__msa_shf_w((v4i32)_pA1, _MSA_SHUFFLE(1, 0, 3, 2));
                v4f32 _pB0 = (v4f32)__msa_ld_w(pB, 0);
                v4f32 _pB1 = (v4f32)__msa_ld_w(pB + 4, 0);
                v4f32 _pB0r = (v4f32)__msa_shf_w((v4i32)_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                v4f32 _pB1r = (v4f32)__msa_shf_w((v4i32)_pB1, _MSA_SHUFFLE(0, 3, 2, 1));

                _sum00 = __ncnn_msa_fmadd_w(_sum00, _pA0, _pB0);
                _sum10 = __ncnn_msa_fmadd_w(_sum10, _pA0, _pB0r);
                _sum20 = __ncnn_msa_fmadd_w(_sum20, _pA0r, _pB0);
                _sum30 = __ncnn_msa_fmadd_w(_sum30, _pA0r, _pB0r);
                _sum40 = __ncnn_msa_fmadd_w(_sum40, _pA0, _pB1);
                _sum50 = __ncnn_msa_fmadd_w(_sum50, _pA0, _pB1r);
                _sum60 = __ncnn_msa_fmadd_w(_sum60, _pA0r, _pB1);
                _sum70 = __ncnn_msa_fmadd_w(_sum70, _pA0r, _pB1r);
                _sum01 = __ncnn_msa_fmadd_w(_sum01, _pA1, _pB0);
                _sum11 = __ncnn_msa_fmadd_w(_sum11, _pA1, _pB0r);
                _sum21 = __ncnn_msa_fmadd_w(_sum21, _pA1r, _pB0);
                _sum31 = __ncnn_msa_fmadd_w(_sum31, _pA1r, _pB0r);
                _sum41 = __ncnn_msa_fmadd_w(_sum41, _pA1, _pB1);
                _sum51 = __ncnn_msa_fmadd_w(_sum51, _pA1, _pB1r);
                _sum61 = __ncnn_msa_fmadd_w(_sum61, _pA1r, _pB1);
                _sum71 = __ncnn_msa_fmadd_w(_sum71, _pA1r, _pB1r);
                pA += 8;
                pB += 8;
            }

            if (k_end)
            {
                _sum10 = (v4f32)__msa_shf_w((v4i32)_sum10, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum30 = (v4f32)__msa_shf_w((v4i32)_sum30, _MSA_SHUFFLE(2, 1, 0, 3));
                {
                    v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_sum30, (v4i32)_sum00);
                    v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_sum30, (v4i32)_sum00);
                    v4f32 _tmp2 = (v4f32)__msa_ilvr_w((v4i32)_sum10, (v4i32)_sum20);
                    v4f32 _tmp3 = (v4f32)__msa_ilvl_w((v4i32)_sum10, (v4i32)_sum20);
                    _sum00 = (v4f32)__msa_ilvr_d((v2i64)_tmp2, (v2i64)_tmp0);
                    _sum10 = (v4f32)__msa_ilvl_d((v2i64)_tmp2, (v2i64)_tmp0);
                    _sum20 = (v4f32)__msa_ilvr_d((v2i64)_tmp1, (v2i64)_tmp3);
                    _sum30 = (v4f32)__msa_ilvl_d((v2i64)_tmp1, (v2i64)_tmp3);
                }
                _sum10 = (v4f32)__msa_shf_w((v4i32)_sum10, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum30 = (v4f32)__msa_shf_w((v4i32)_sum30, _MSA_SHUFFLE(2, 1, 0, 3));

                _sum50 = (v4f32)__msa_shf_w((v4i32)_sum50, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum70 = (v4f32)__msa_shf_w((v4i32)_sum70, _MSA_SHUFFLE(2, 1, 0, 3));
                {
                    v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_sum70, (v4i32)_sum40);
                    v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_sum70, (v4i32)_sum40);
                    v4f32 _tmp2 = (v4f32)__msa_ilvr_w((v4i32)_sum50, (v4i32)_sum60);
                    v4f32 _tmp3 = (v4f32)__msa_ilvl_w((v4i32)_sum50, (v4i32)_sum60);
                    _sum40 = (v4f32)__msa_ilvr_d((v2i64)_tmp2, (v2i64)_tmp0);
                    _sum50 = (v4f32)__msa_ilvl_d((v2i64)_tmp2, (v2i64)_tmp0);
                    _sum60 = (v4f32)__msa_ilvr_d((v2i64)_tmp1, (v2i64)_tmp3);
                    _sum70 = (v4f32)__msa_ilvl_d((v2i64)_tmp1, (v2i64)_tmp3);
                }
                _sum50 = (v4f32)__msa_shf_w((v4i32)_sum50, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum70 = (v4f32)__msa_shf_w((v4i32)_sum70, _MSA_SHUFFLE(2, 1, 0, 3));

                _sum11 = (v4f32)__msa_shf_w((v4i32)_sum11, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum31 = (v4f32)__msa_shf_w((v4i32)_sum31, _MSA_SHUFFLE(2, 1, 0, 3));
                {
                    v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_sum31, (v4i32)_sum01);
                    v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_sum31, (v4i32)_sum01);
                    v4f32 _tmp2 = (v4f32)__msa_ilvr_w((v4i32)_sum11, (v4i32)_sum21);
                    v4f32 _tmp3 = (v4f32)__msa_ilvl_w((v4i32)_sum11, (v4i32)_sum21);
                    _sum01 = (v4f32)__msa_ilvr_d((v2i64)_tmp2, (v2i64)_tmp0);
                    _sum11 = (v4f32)__msa_ilvl_d((v2i64)_tmp2, (v2i64)_tmp0);
                    _sum21 = (v4f32)__msa_ilvr_d((v2i64)_tmp1, (v2i64)_tmp3);
                    _sum31 = (v4f32)__msa_ilvl_d((v2i64)_tmp1, (v2i64)_tmp3);
                }
                _sum11 = (v4f32)__msa_shf_w((v4i32)_sum11, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum31 = (v4f32)__msa_shf_w((v4i32)_sum31, _MSA_SHUFFLE(2, 1, 0, 3));

                _sum51 = (v4f32)__msa_shf_w((v4i32)_sum51, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum71 = (v4f32)__msa_shf_w((v4i32)_sum71, _MSA_SHUFFLE(2, 1, 0, 3));
                {
                    v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_sum71, (v4i32)_sum41);
                    v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_sum71, (v4i32)_sum41);
                    v4f32 _tmp2 = (v4f32)__msa_ilvr_w((v4i32)_sum51, (v4i32)_sum61);
                    v4f32 _tmp3 = (v4f32)__msa_ilvl_w((v4i32)_sum51, (v4i32)_sum61);
                    _sum41 = (v4f32)__msa_ilvr_d((v2i64)_tmp2, (v2i64)_tmp0);
                    _sum51 = (v4f32)__msa_ilvl_d((v2i64)_tmp2, (v2i64)_tmp0);
                    _sum61 = (v4f32)__msa_ilvr_d((v2i64)_tmp1, (v2i64)_tmp3);
                    _sum71 = (v4f32)__msa_ilvl_d((v2i64)_tmp1, (v2i64)_tmp3);
                }
                _sum51 = (v4f32)__msa_shf_w((v4i32)_sum51, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum71 = (v4f32)__msa_shf_w((v4i32)_sum71, _MSA_SHUFFLE(2, 1, 0, 3));

                if (pC0)
                {
                    _sum00 = __msa_fadd_w(_sum00, _bias0);
                    _sum10 = __msa_fadd_w(_sum10, _bias0);
                    _sum20 = __msa_fadd_w(_sum20, _bias0);
                    _sum30 = __msa_fadd_w(_sum30, _bias0);
                    _sum40 = __msa_fadd_w(_sum40, _bias0);
                    _sum50 = __msa_fadd_w(_sum50, _bias0);
                    _sum60 = __msa_fadd_w(_sum60, _bias0);
                    _sum70 = __msa_fadd_w(_sum70, _bias0);
                    _sum01 = __msa_fadd_w(_sum01, _bias1);
                    _sum11 = __msa_fadd_w(_sum11, _bias1);
                    _sum21 = __msa_fadd_w(_sum21, _bias1);
                    _sum31 = __msa_fadd_w(_sum31, _bias1);
                    _sum41 = __msa_fadd_w(_sum41, _bias1);
                    _sum51 = __msa_fadd_w(_sum51, _bias1);
                    _sum61 = __msa_fadd_w(_sum61, _bias1);
                    _sum71 = __msa_fadd_w(_sum71, _bias1);
                }

                if (out_elempack == 4)
                {
                    float* outptr1 = outptr0 + out_hstep * 4;
                    __msa_st_w((v4i32)_sum00, outptr0, 0);
                    __msa_st_w((v4i32)_sum01, outptr1, 0);
                    __msa_st_w((v4i32)_sum10, outptr0 + 4, 0);
                    __msa_st_w((v4i32)_sum11, outptr1 + 4, 0);
                    __msa_st_w((v4i32)_sum20, outptr0 + 8, 0);
                    __msa_st_w((v4i32)_sum21, outptr1 + 8, 0);
                    __msa_st_w((v4i32)_sum30, outptr0 + 12, 0);
                    __msa_st_w((v4i32)_sum31, outptr1 + 12, 0);
                    __msa_st_w((v4i32)_sum40, outptr0 + 16, 0);
                    __msa_st_w((v4i32)_sum41, outptr1 + 16, 0);
                    __msa_st_w((v4i32)_sum50, outptr0 + 20, 0);
                    __msa_st_w((v4i32)_sum51, outptr1 + 20, 0);
                    __msa_st_w((v4i32)_sum60, outptr0 + 24, 0);
                    __msa_st_w((v4i32)_sum61, outptr1 + 24, 0);
                    __msa_st_w((v4i32)_sum70, outptr0 + 28, 0);
                    __msa_st_w((v4i32)_sum71, outptr1 + 28, 0);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    v4f32 _r0 = _sum00;
                    v4f32 _r1 = _sum10;
                    v4f32 _r2 = _sum20;
                    v4f32 _r3 = _sum30;
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    v4f32 _r4 = _sum40;
                    v4f32 _r5 = _sum50;
                    v4f32 _r6 = _sum60;
                    v4f32 _r7 = _sum70;
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    v4f32 _r8 = _sum01;
                    v4f32 _r9 = _sum11;
                    v4f32 _ra = _sum21;
                    v4f32 _rb = _sum31;
                    transpose4x4_ps(_r8, _r9, _ra, _rb);
                    v4f32 _rc = _sum41;
                    v4f32 _rd = _sum51;
                    v4f32 _re = _sum61;
                    v4f32 _rf = _sum71;
                    transpose4x4_ps(_rc, _rd, _re, _rf);

                    __msa_st_w((v4i32)_r0, outptr0, 0);
                    __msa_st_w((v4i32)_r4, outptr0 + 4, 0);
                    __msa_st_w((v4i32)_r1, outptr0 + out_hstep, 0);
                    __msa_st_w((v4i32)_r5, outptr0 + out_hstep + 4, 0);
                    __msa_st_w((v4i32)_r2, outptr0 + out_hstep * 2, 0);
                    __msa_st_w((v4i32)_r6, outptr0 + out_hstep * 2 + 4, 0);
                    __msa_st_w((v4i32)_r3, outptr0 + out_hstep * 3, 0);
                    __msa_st_w((v4i32)_r7, outptr0 + out_hstep * 3 + 4, 0);
                    __msa_st_w((v4i32)_r8, outptr0 + out_hstep * 4, 0);
                    __msa_st_w((v4i32)_rc, outptr0 + out_hstep * 4 + 4, 0);
                    __msa_st_w((v4i32)_r9, outptr0 + out_hstep * 5, 0);
                    __msa_st_w((v4i32)_rd, outptr0 + out_hstep * 5 + 4, 0);
                    __msa_st_w((v4i32)_ra, outptr0 + out_hstep * 6, 0);
                    __msa_st_w((v4i32)_re, outptr0 + out_hstep * 6 + 4, 0);
                    __msa_st_w((v4i32)_rb, outptr0 + out_hstep * 7, 0);
                    __msa_st_w((v4i32)_rf, outptr0 + out_hstep * 7 + 4, 0);
                    outptr0 += 8;
                }
            }
            else
            {
                __msa_st_w((v4i32)_sum00, outptr, 0);
                __msa_st_w((v4i32)_sum01, outptr + 4, 0);
                __msa_st_w((v4i32)_sum10, outptr + 8, 0);
                __msa_st_w((v4i32)_sum11, outptr + 12, 0);
                __msa_st_w((v4i32)_sum20, outptr + 16, 0);
                __msa_st_w((v4i32)_sum21, outptr + 20, 0);
                __msa_st_w((v4i32)_sum30, outptr + 24, 0);
                __msa_st_w((v4i32)_sum31, outptr + 28, 0);
                __msa_st_w((v4i32)_sum40, outptr + 32, 0);
                __msa_st_w((v4i32)_sum41, outptr + 36, 0);
                __msa_st_w((v4i32)_sum50, outptr + 40, 0);
                __msa_st_w((v4i32)_sum51, outptr + 44, 0);
                __msa_st_w((v4i32)_sum60, outptr + 48, 0);
                __msa_st_w((v4i32)_sum61, outptr + 52, 0);
                __msa_st_w((v4i32)_sum70, outptr + 56, 0);
                __msa_st_w((v4i32)_sum71, outptr + 60, 0);
            }

            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            v4f32 _sum00;
            v4f32 _sum01;
            v4f32 _sum10;
            v4f32 _sum11;
            v4f32 _sum20;
            v4f32 _sum21;
            v4f32 _sum30;
            v4f32 _sum31;

            if (k == 0)
            {
                _sum00 = (v4f32)__msa_fill_w(0);
                _sum01 = (v4f32)__msa_fill_w(0);
                _sum10 = (v4f32)__msa_fill_w(0);
                _sum11 = (v4f32)__msa_fill_w(0);
                _sum20 = (v4f32)__msa_fill_w(0);
                _sum21 = (v4f32)__msa_fill_w(0);
                _sum30 = (v4f32)__msa_fill_w(0);
                _sum31 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _sum00 = (v4f32)__msa_ld_w(outptr, 0);
                _sum01 = (v4f32)__msa_ld_w(outptr + 4, 0);
                _sum10 = (v4f32)__msa_ld_w(outptr + 8, 0);
                _sum11 = (v4f32)__msa_ld_w(outptr + 12, 0);
                _sum20 = (v4f32)__msa_ld_w(outptr + 16, 0);
                _sum21 = (v4f32)__msa_ld_w(outptr + 20, 0);
                _sum30 = (v4f32)__msa_ld_w(outptr + 24, 0);
                _sum31 = (v4f32)__msa_ld_w(outptr + 28, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 16);
                v4f32 _pA0 = (v4f32)__msa_ld_w(pA, 0);
                v4f32 _pA1 = (v4f32)__msa_ld_w(pA + 4, 0);
                v4f32 _pA0r = (v4f32)__msa_shf_w((v4i32)_pA0, _MSA_SHUFFLE(1, 0, 3, 2));
                v4f32 _pA1r = (v4f32)__msa_shf_w((v4i32)_pA1, _MSA_SHUFFLE(1, 0, 3, 2));
                v4f32 _pB = (v4f32)__msa_ld_w(pB, 0);
                v4f32 _pB1 = (v4f32)__msa_shf_w((v4i32)_pB, _MSA_SHUFFLE(0, 3, 2, 1));

                _sum00 = __ncnn_msa_fmadd_w(_sum00, _pA0, _pB);
                _sum10 = __ncnn_msa_fmadd_w(_sum10, _pA0, _pB1);
                _sum20 = __ncnn_msa_fmadd_w(_sum20, _pA0r, _pB);
                _sum30 = __ncnn_msa_fmadd_w(_sum30, _pA0r, _pB1);
                _sum01 = __ncnn_msa_fmadd_w(_sum01, _pA1, _pB);
                _sum11 = __ncnn_msa_fmadd_w(_sum11, _pA1, _pB1);
                _sum21 = __ncnn_msa_fmadd_w(_sum21, _pA1r, _pB);
                _sum31 = __ncnn_msa_fmadd_w(_sum31, _pA1r, _pB1);
                pA += 8;
                pB += 4;
            }

            if (k_end)
            {
                _sum10 = (v4f32)__msa_shf_w((v4i32)_sum10, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum30 = (v4f32)__msa_shf_w((v4i32)_sum30, _MSA_SHUFFLE(2, 1, 0, 3));
                {
                    v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_sum30, (v4i32)_sum00);
                    v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_sum30, (v4i32)_sum00);
                    v4f32 _tmp2 = (v4f32)__msa_ilvr_w((v4i32)_sum10, (v4i32)_sum20);
                    v4f32 _tmp3 = (v4f32)__msa_ilvl_w((v4i32)_sum10, (v4i32)_sum20);
                    _sum00 = (v4f32)__msa_ilvr_d((v2i64)_tmp2, (v2i64)_tmp0);
                    _sum10 = (v4f32)__msa_ilvl_d((v2i64)_tmp2, (v2i64)_tmp0);
                    _sum20 = (v4f32)__msa_ilvr_d((v2i64)_tmp1, (v2i64)_tmp3);
                    _sum30 = (v4f32)__msa_ilvl_d((v2i64)_tmp1, (v2i64)_tmp3);
                }
                _sum10 = (v4f32)__msa_shf_w((v4i32)_sum10, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum30 = (v4f32)__msa_shf_w((v4i32)_sum30, _MSA_SHUFFLE(2, 1, 0, 3));

                _sum11 = (v4f32)__msa_shf_w((v4i32)_sum11, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum31 = (v4f32)__msa_shf_w((v4i32)_sum31, _MSA_SHUFFLE(2, 1, 0, 3));
                {
                    v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_sum31, (v4i32)_sum01);
                    v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_sum31, (v4i32)_sum01);
                    v4f32 _tmp2 = (v4f32)__msa_ilvr_w((v4i32)_sum11, (v4i32)_sum21);
                    v4f32 _tmp3 = (v4f32)__msa_ilvl_w((v4i32)_sum11, (v4i32)_sum21);
                    _sum01 = (v4f32)__msa_ilvr_d((v2i64)_tmp2, (v2i64)_tmp0);
                    _sum11 = (v4f32)__msa_ilvl_d((v2i64)_tmp2, (v2i64)_tmp0);
                    _sum21 = (v4f32)__msa_ilvr_d((v2i64)_tmp1, (v2i64)_tmp3);
                    _sum31 = (v4f32)__msa_ilvl_d((v2i64)_tmp1, (v2i64)_tmp3);
                }
                _sum11 = (v4f32)__msa_shf_w((v4i32)_sum11, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum31 = (v4f32)__msa_shf_w((v4i32)_sum31, _MSA_SHUFFLE(2, 1, 0, 3));

                if (pC0)
                {
                    _sum00 = __msa_fadd_w(_sum00, _bias0);
                    _sum10 = __msa_fadd_w(_sum10, _bias0);
                    _sum20 = __msa_fadd_w(_sum20, _bias0);
                    _sum30 = __msa_fadd_w(_sum30, _bias0);
                    _sum01 = __msa_fadd_w(_sum01, _bias1);
                    _sum11 = __msa_fadd_w(_sum11, _bias1);
                    _sum21 = __msa_fadd_w(_sum21, _bias1);
                    _sum31 = __msa_fadd_w(_sum31, _bias1);
                }

                if (out_elempack == 4)
                {
                    float* outptr1 = outptr0 + out_hstep * 4;
                    __msa_st_w((v4i32)_sum00, outptr0, 0);
                    __msa_st_w((v4i32)_sum01, outptr1, 0);
                    __msa_st_w((v4i32)_sum10, outptr0 + 4, 0);
                    __msa_st_w((v4i32)_sum11, outptr1 + 4, 0);
                    __msa_st_w((v4i32)_sum20, outptr0 + 8, 0);
                    __msa_st_w((v4i32)_sum21, outptr1 + 8, 0);
                    __msa_st_w((v4i32)_sum30, outptr0 + 12, 0);
                    __msa_st_w((v4i32)_sum31, outptr1 + 12, 0);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    v4f32 _r0 = _sum00;
                    v4f32 _r1 = _sum10;
                    v4f32 _r2 = _sum20;
                    v4f32 _r3 = _sum30;
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    v4f32 _r4 = _sum01;
                    v4f32 _r5 = _sum11;
                    v4f32 _r6 = _sum21;
                    v4f32 _r7 = _sum31;
                    transpose4x4_ps(_r4, _r5, _r6, _r7);

                    __msa_st_w((v4i32)_r0, outptr0, 0);
                    __msa_st_w((v4i32)_r1, outptr0 + out_hstep, 0);
                    __msa_st_w((v4i32)_r2, outptr0 + out_hstep * 2, 0);
                    __msa_st_w((v4i32)_r3, outptr0 + out_hstep * 3, 0);
                    __msa_st_w((v4i32)_r4, outptr0 + out_hstep * 4, 0);
                    __msa_st_w((v4i32)_r5, outptr0 + out_hstep * 5, 0);
                    __msa_st_w((v4i32)_r6, outptr0 + out_hstep * 6, 0);
                    __msa_st_w((v4i32)_r7, outptr0 + out_hstep * 7, 0);
                    outptr0 += 4;
                }
            }
            else
            {
                __msa_st_w((v4i32)_sum00, outptr, 0);
                __msa_st_w((v4i32)_sum01, outptr + 4, 0);
                __msa_st_w((v4i32)_sum10, outptr + 8, 0);
                __msa_st_w((v4i32)_sum11, outptr + 12, 0);
                __msa_st_w((v4i32)_sum20, outptr + 16, 0);
                __msa_st_w((v4i32)_sum21, outptr + 20, 0);
                __msa_st_w((v4i32)_sum30, outptr + 24, 0);
                __msa_st_w((v4i32)_sum31, outptr + 28, 0);
            }

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            v4f32 _sum00;
            v4f32 _sum01;
            v4f32 _sum10;
            v4f32 _sum11;

            if (k == 0)
            {
                _sum00 = (v4f32)__msa_fill_w(0);
                _sum01 = (v4f32)__msa_fill_w(0);
                _sum10 = (v4f32)__msa_fill_w(0);
                _sum11 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _sum00 = (v4f32)__msa_ld_w(outptr, 0);
                _sum01 = (v4f32)__msa_ld_w(outptr + 4, 0);
                _sum10 = (v4f32)__msa_ld_w(outptr + 8, 0);
                _sum11 = (v4f32)__msa_ld_w(outptr + 12, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 8);
                v4f32 _pA0 = (v4f32)__msa_ld_w(pA, 0);
                v4f32 _pA1 = (v4f32)__msa_ld_w(pA + 4, 0);
                v4f32 _pB = (v4f32)__msa_fill_d_ptr(pB);
                v4f32 _pB1 = (v4f32)__msa_shf_w((v4i32)_pB, _MSA_SHUFFLE(2, 3, 0, 1));

                _sum00 = __ncnn_msa_fmadd_w(_sum00, _pA0, _pB);
                _sum10 = __ncnn_msa_fmadd_w(_sum10, _pA0, _pB1);
                _sum01 = __ncnn_msa_fmadd_w(_sum01, _pA1, _pB);
                _sum11 = __ncnn_msa_fmadd_w(_sum11, _pA1, _pB1);
                pA += 8;
                pB += 2;
            }

            if (k_end)
            {
                {
                    v4f32 _tmp0 = (v4f32)__msa_shf_w((v4i32)_sum00, _MSA_SHUFFLE(3, 1, 2, 0));
                    v4f32 _tmp1 = (v4f32)__msa_shf_w((v4i32)_sum10, _MSA_SHUFFLE(0, 2, 3, 1));
                    _sum00 = (v4f32)__msa_ilvr_w((v4i32)_tmp1, (v4i32)_tmp0);
                    _sum10 = (v4f32)__msa_ilvl_w((v4i32)_tmp1, (v4i32)_tmp0);
                    _sum10 = (v4f32)__msa_shf_w((v4i32)_sum10, _MSA_SHUFFLE(2, 1, 0, 3));
                }
                {
                    v4f32 _tmp0 = (v4f32)__msa_shf_w((v4i32)_sum01, _MSA_SHUFFLE(3, 1, 2, 0));
                    v4f32 _tmp1 = (v4f32)__msa_shf_w((v4i32)_sum11, _MSA_SHUFFLE(0, 2, 3, 1));
                    _sum01 = (v4f32)__msa_ilvr_w((v4i32)_tmp1, (v4i32)_tmp0);
                    _sum11 = (v4f32)__msa_ilvl_w((v4i32)_tmp1, (v4i32)_tmp0);
                    _sum11 = (v4f32)__msa_shf_w((v4i32)_sum11, _MSA_SHUFFLE(2, 1, 0, 3));
                }

                if (pC0)
                {
                    _sum00 = __msa_fadd_w(_sum00, _bias0);
                    _sum10 = __msa_fadd_w(_sum10, _bias0);
                    _sum01 = __msa_fadd_w(_sum01, _bias1);
                    _sum11 = __msa_fadd_w(_sum11, _bias1);
                }

                if (out_elempack == 4)
                {
                    float* outptr1 = outptr0 + out_hstep * 4;
                    __msa_st_w((v4i32)_sum00, outptr0, 0);
                    __msa_st_w((v4i32)_sum01, outptr1, 0);
                    __msa_st_w((v4i32)_sum10, outptr0 + 4, 0);
                    __msa_st_w((v4i32)_sum11, outptr1 + 4, 0);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    uint64_t v0 = (uint32_t)__msa_copy_s_w((v4i32)_sum00, 0) | ((uint64_t)(uint32_t)__msa_copy_s_w((v4i32)_sum10, 0) << 32);
                    uint64_t v1 = (uint32_t)__msa_copy_s_w((v4i32)_sum00, 1) | ((uint64_t)(uint32_t)__msa_copy_s_w((v4i32)_sum10, 1) << 32);
                    uint64_t v2 = (uint32_t)__msa_copy_s_w((v4i32)_sum00, 2) | ((uint64_t)(uint32_t)__msa_copy_s_w((v4i32)_sum10, 2) << 32);
                    uint64_t v3 = (uint32_t)__msa_copy_s_w((v4i32)_sum00, 3) | ((uint64_t)(uint32_t)__msa_copy_s_w((v4i32)_sum10, 3) << 32);
                    uint64_t v4 = (uint32_t)__msa_copy_s_w((v4i32)_sum01, 0) | ((uint64_t)(uint32_t)__msa_copy_s_w((v4i32)_sum11, 0) << 32);
                    uint64_t v5 = (uint32_t)__msa_copy_s_w((v4i32)_sum01, 1) | ((uint64_t)(uint32_t)__msa_copy_s_w((v4i32)_sum11, 1) << 32);
                    uint64_t v6 = (uint32_t)__msa_copy_s_w((v4i32)_sum01, 2) | ((uint64_t)(uint32_t)__msa_copy_s_w((v4i32)_sum11, 2) << 32);
                    uint64_t v7 = (uint32_t)__msa_copy_s_w((v4i32)_sum01, 3) | ((uint64_t)(uint32_t)__msa_copy_s_w((v4i32)_sum11, 3) << 32);
                    *(uint64_t*)outptr0 = v0;
                    *(uint64_t*)(outptr0 + out_hstep) = v1;
                    *(uint64_t*)(outptr0 + out_hstep * 2) = v2;
                    *(uint64_t*)(outptr0 + out_hstep * 3) = v3;
                    *(uint64_t*)(outptr0 + out_hstep * 4) = v4;
                    *(uint64_t*)(outptr0 + out_hstep * 5) = v5;
                    *(uint64_t*)(outptr0 + out_hstep * 6) = v6;
                    *(uint64_t*)(outptr0 + out_hstep * 7) = v7;
                    outptr0 += 2;
                }
            }
            else
            {
                __msa_st_w((v4i32)_sum00, outptr, 0);
                __msa_st_w((v4i32)_sum01, outptr + 4, 0);
                __msa_st_w((v4i32)_sum10, outptr + 8, 0);
                __msa_st_w((v4i32)_sum11, outptr + 12, 0);
            }

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            v4f32 _sum0;
            v4f32 _sum1;

            if (k == 0)
            {
                _sum0 = (v4f32)__msa_fill_w(0);
                _sum1 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _sum0 = (v4f32)__msa_ld_w(outptr, 0);
                _sum1 = (v4f32)__msa_ld_w(outptr + 4, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 4);
                v4f32 _pA0 = (v4f32)__msa_ld_w(pA, 0);
                v4f32 _pA1 = (v4f32)__msa_ld_w(pA + 4, 0);
                _sum0 = __ncnn_msa_fmadd_w(_sum0, _pA0, __msa_fill_w_f32(pB[0]));
                _sum1 = __ncnn_msa_fmadd_w(_sum1, _pA1, __msa_fill_w_f32(pB[0]));
                pA += 8;
                pB += 1;
            }

            if (k_end)
            {
                if (pC0)
                {
                    _sum0 = __msa_fadd_w(_sum0, _bias0);
                    _sum1 = __msa_fadd_w(_sum1, _bias1);
                }

                if (out_elempack == 4)
                {
                    float* outptr1 = outptr0 + out_hstep * 4;
                    __msa_st_w((v4i32)_sum0, outptr0, 0);
                    __msa_st_w((v4i32)_sum1, outptr1, 0);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    uint32_t v0 = __msa_copy_s_w((v4i32)_sum0, 0);
                    uint32_t v1 = __msa_copy_s_w((v4i32)_sum0, 1);
                    uint32_t v2 = __msa_copy_s_w((v4i32)_sum0, 2);
                    uint32_t v3 = __msa_copy_s_w((v4i32)_sum0, 3);
                    uint32_t v4 = __msa_copy_s_w((v4i32)_sum1, 0);
                    uint32_t v5 = __msa_copy_s_w((v4i32)_sum1, 1);
                    uint32_t v6 = __msa_copy_s_w((v4i32)_sum1, 2);
                    uint32_t v7 = __msa_copy_s_w((v4i32)_sum1, 3);
                    *(uint32_t*)outptr0 = v0;
                    *(uint32_t*)(outptr0 + out_hstep) = v1;
                    *(uint32_t*)(outptr0 + out_hstep * 2) = v2;
                    *(uint32_t*)(outptr0 + out_hstep * 3) = v3;
                    *(uint32_t*)(outptr0 + out_hstep * 4) = v4;
                    *(uint32_t*)(outptr0 + out_hstep * 5) = v5;
                    *(uint32_t*)(outptr0 + out_hstep * 6) = v6;
                    *(uint32_t*)(outptr0 + out_hstep * 7) = v7;
                    outptr0++;
                }
            }
            else
            {
                __msa_st_w((v4i32)_sum0, outptr, 0);
                __msa_st_w((v4i32)_sum1, outptr + 4, 0);
            }

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            const float* pA = pAT;

            v4f32 _sum0;
            v4f32 _sum1;
            v4f32 _sum2;
            v4f32 _sum3;
            v4f32 _sum4;
            v4f32 _sum5;
            v4f32 _sum6;
            v4f32 _sum7;

            if (k == 0)
            {
                _sum0 = (v4f32)__msa_fill_w(0);
                _sum1 = (v4f32)__msa_fill_w(0);
                _sum2 = (v4f32)__msa_fill_w(0);
                _sum3 = (v4f32)__msa_fill_w(0);
                _sum4 = (v4f32)__msa_fill_w(0);
                _sum5 = (v4f32)__msa_fill_w(0);
                _sum6 = (v4f32)__msa_fill_w(0);
                _sum7 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _sum0 = (v4f32)__msa_ld_w(outptr, 0);
                _sum1 = (v4f32)__msa_ld_w(outptr + 4 * 1, 0);
                _sum2 = (v4f32)__msa_ld_w(outptr + 4 * 2, 0);
                _sum3 = (v4f32)__msa_ld_w(outptr + 4 * 3, 0);
                _sum4 = (v4f32)__msa_ld_w(outptr + 4 * 4, 0);
                _sum5 = (v4f32)__msa_ld_w(outptr + 4 * 5, 0);
                _sum6 = (v4f32)__msa_ld_w(outptr + 4 * 6, 0);
                _sum7 = (v4f32)__msa_ld_w(outptr + 4 * 7, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 16);
                v4f32 _pA = (v4f32)__msa_ld_w(pA, 0);
                v4f32 _pA1 = (v4f32)__msa_shf_w((v4i32)_pA, _MSA_SHUFFLE(1, 0, 3, 2));
                v4f32 _pB0 = (v4f32)__msa_ld_w(pB, 0);
                v4f32 _pB1 = (v4f32)__msa_ld_w(pB + 4, 0);
                v4f32 _pB0r = (v4f32)__msa_shf_w((v4i32)_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                v4f32 _pB1r = (v4f32)__msa_shf_w((v4i32)_pB1, _MSA_SHUFFLE(0, 3, 2, 1));

                _sum0 = __ncnn_msa_fmadd_w(_sum0, _pA, _pB0);
                _sum1 = __ncnn_msa_fmadd_w(_sum1, _pA, _pB0r);
                _sum2 = __ncnn_msa_fmadd_w(_sum2, _pA1, _pB0);
                _sum3 = __ncnn_msa_fmadd_w(_sum3, _pA1, _pB0r);
                _sum4 = __ncnn_msa_fmadd_w(_sum4, _pA, _pB1);
                _sum5 = __ncnn_msa_fmadd_w(_sum5, _pA, _pB1r);
                _sum6 = __ncnn_msa_fmadd_w(_sum6, _pA1, _pB1);
                _sum7 = __ncnn_msa_fmadd_w(_sum7, _pA1, _pB1r);

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                _sum1 = (v4f32)__msa_shf_w((v4i32)_sum1, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum3 = (v4f32)__msa_shf_w((v4i32)_sum3, _MSA_SHUFFLE(2, 1, 0, 3));
                {
                    v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_sum3, (v4i32)_sum0);
                    v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_sum3, (v4i32)_sum0);
                    v4f32 _tmp2 = (v4f32)__msa_ilvr_w((v4i32)_sum1, (v4i32)_sum2);
                    v4f32 _tmp3 = (v4f32)__msa_ilvl_w((v4i32)_sum1, (v4i32)_sum2);
                    _sum0 = (v4f32)__msa_ilvr_d((v2i64)_tmp2, (v2i64)_tmp0);
                    _sum1 = (v4f32)__msa_ilvl_d((v2i64)_tmp2, (v2i64)_tmp0);
                    _sum2 = (v4f32)__msa_ilvr_d((v2i64)_tmp1, (v2i64)_tmp3);
                    _sum3 = (v4f32)__msa_ilvl_d((v2i64)_tmp1, (v2i64)_tmp3);
                }
                _sum1 = (v4f32)__msa_shf_w((v4i32)_sum1, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum3 = (v4f32)__msa_shf_w((v4i32)_sum3, _MSA_SHUFFLE(2, 1, 0, 3));

                _sum5 = (v4f32)__msa_shf_w((v4i32)_sum5, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum7 = (v4f32)__msa_shf_w((v4i32)_sum7, _MSA_SHUFFLE(2, 1, 0, 3));
                {
                    v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_sum7, (v4i32)_sum4);
                    v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_sum7, (v4i32)_sum4);
                    v4f32 _tmp2 = (v4f32)__msa_ilvr_w((v4i32)_sum5, (v4i32)_sum6);
                    v4f32 _tmp3 = (v4f32)__msa_ilvl_w((v4i32)_sum5, (v4i32)_sum6);
                    _sum4 = (v4f32)__msa_ilvr_d((v2i64)_tmp2, (v2i64)_tmp0);
                    _sum5 = (v4f32)__msa_ilvl_d((v2i64)_tmp2, (v2i64)_tmp0);
                    _sum6 = (v4f32)__msa_ilvr_d((v2i64)_tmp1, (v2i64)_tmp3);
                    _sum7 = (v4f32)__msa_ilvl_d((v2i64)_tmp1, (v2i64)_tmp3);
                }
                _sum5 = (v4f32)__msa_shf_w((v4i32)_sum5, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum7 = (v4f32)__msa_shf_w((v4i32)_sum7, _MSA_SHUFFLE(2, 1, 0, 3));
                if (pC)
                {
                    v4f32 _bias = (v4f32)__msa_ld_w(pC, 0);
                    _sum0 = __msa_fadd_w(_sum0, _bias);
                    _sum1 = __msa_fadd_w(_sum1, _bias);
                    _sum2 = __msa_fadd_w(_sum2, _bias);
                    _sum3 = __msa_fadd_w(_sum3, _bias);
                    _sum4 = __msa_fadd_w(_sum4, _bias);
                    _sum5 = __msa_fadd_w(_sum5, _bias);
                    _sum6 = __msa_fadd_w(_sum6, _bias);
                    _sum7 = __msa_fadd_w(_sum7, _bias);
                }
                if (out_elempack == 4)
                {
                    __msa_st_w((v4i32)_sum0, outptr0, 0);
                    __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);
                    __msa_st_w((v4i32)_sum2, outptr0 + 4 * 2, 0);
                    __msa_st_w((v4i32)_sum3, outptr0 + 4 * 3, 0);
                    __msa_st_w((v4i32)_sum4, outptr0 + 4 * 4, 0);
                    __msa_st_w((v4i32)_sum5, outptr0 + 4 * 5, 0);
                    __msa_st_w((v4i32)_sum6, outptr0 + 4 * 6, 0);
                    __msa_st_w((v4i32)_sum7, outptr0 + 4 * 7, 0);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                    transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);

                    __msa_st_w((v4i32)_sum0, outptr0, 0);
                    __msa_st_w((v4i32)_sum1, outptr0 + out_hstep * 1, 0);
                    __msa_st_w((v4i32)_sum2, outptr0 + out_hstep * 2, 0);
                    __msa_st_w((v4i32)_sum3, outptr0 + out_hstep * 3, 0);
                    __msa_st_w((v4i32)_sum4, outptr0 + 4, 0);
                    __msa_st_w((v4i32)_sum5, outptr0 + out_hstep * 1 + 4, 0);
                    __msa_st_w((v4i32)_sum6, outptr0 + out_hstep * 2 + 4, 0);
                    __msa_st_w((v4i32)_sum7, outptr0 + out_hstep * 3 + 4, 0);
                    outptr0 += 8;
                }
            }
            else
            {
                __msa_st_w((v4i32)_sum0, outptr, 0);
                __msa_st_w((v4i32)_sum1, outptr + 4, 0);
                __msa_st_w((v4i32)_sum2, outptr + 4 * 2, 0);
                __msa_st_w((v4i32)_sum3, outptr + 4 * 3, 0);
                __msa_st_w((v4i32)_sum4, outptr + 4 * 4, 0);
                __msa_st_w((v4i32)_sum5, outptr + 4 * 5, 0);
                __msa_st_w((v4i32)_sum6, outptr + 4 * 6, 0);
                __msa_st_w((v4i32)_sum7, outptr + 4 * 7, 0);
            }

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const float* pA = pAT;

            v4f32 _sum0;
            v4f32 _sum1;
            v4f32 _sum2;
            v4f32 _sum3;

            if (k == 0)
            {
                _sum0 = (v4f32)__msa_fill_w(0);
                _sum1 = (v4f32)__msa_fill_w(0);
                _sum2 = (v4f32)__msa_fill_w(0);
                _sum3 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _sum0 = (v4f32)__msa_ld_w(outptr, 0);
                _sum1 = (v4f32)__msa_ld_w(outptr + 4 * 1, 0);
                _sum2 = (v4f32)__msa_ld_w(outptr + 4 * 2, 0);
                _sum3 = (v4f32)__msa_ld_w(outptr + 4 * 3, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 16);
                v4f32 _pA = (v4f32)__msa_ld_w(pA, 0);
                v4f32 _pA1 = (v4f32)__msa_shf_w((v4i32)_pA, _MSA_SHUFFLE(1, 0, 3, 2));
                v4f32 _pB = (v4f32)__msa_ld_w(pB, 0);
                v4f32 _pB1 = (v4f32)__msa_shf_w((v4i32)_pB, _MSA_SHUFFLE(0, 3, 2, 1));

                _sum0 = __ncnn_msa_fmadd_w(_sum0, _pA, _pB);
                _sum1 = __ncnn_msa_fmadd_w(_sum1, _pA, _pB1);
                _sum2 = __ncnn_msa_fmadd_w(_sum2, _pA1, _pB);
                _sum3 = __ncnn_msa_fmadd_w(_sum3, _pA1, _pB1);

                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                _sum1 = (v4f32)__msa_shf_w((v4i32)_sum1, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum3 = (v4f32)__msa_shf_w((v4i32)_sum3, _MSA_SHUFFLE(2, 1, 0, 3));
                {
                    v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_sum3, (v4i32)_sum0);
                    v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_sum3, (v4i32)_sum0);
                    v4f32 _tmp2 = (v4f32)__msa_ilvr_w((v4i32)_sum1, (v4i32)_sum2);
                    v4f32 _tmp3 = (v4f32)__msa_ilvl_w((v4i32)_sum1, (v4i32)_sum2);
                    _sum0 = (v4f32)__msa_ilvr_d((v2i64)_tmp2, (v2i64)_tmp0);
                    _sum1 = (v4f32)__msa_ilvl_d((v2i64)_tmp2, (v2i64)_tmp0);
                    _sum2 = (v4f32)__msa_ilvr_d((v2i64)_tmp1, (v2i64)_tmp3);
                    _sum3 = (v4f32)__msa_ilvl_d((v2i64)_tmp1, (v2i64)_tmp3);
                }
                _sum1 = (v4f32)__msa_shf_w((v4i32)_sum1, _MSA_SHUFFLE(2, 1, 0, 3));
                _sum3 = (v4f32)__msa_shf_w((v4i32)_sum3, _MSA_SHUFFLE(2, 1, 0, 3));
                if (pC)
                {
                    v4f32 _bias = (v4f32)__msa_ld_w(pC, 0);
                    _sum0 = __msa_fadd_w(_sum0, _bias);
                    _sum1 = __msa_fadd_w(_sum1, _bias);
                    _sum2 = __msa_fadd_w(_sum2, _bias);
                    _sum3 = __msa_fadd_w(_sum3, _bias);
                }
                if (out_elempack == 4)
                {
                    __msa_st_w((v4i32)_sum0, outptr0, 0);
                    __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);
                    __msa_st_w((v4i32)_sum2, outptr0 + 4 * 2, 0);
                    __msa_st_w((v4i32)_sum3, outptr0 + 4 * 3, 0);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);

                    __msa_st_w((v4i32)_sum0, outptr0, 0);
                    __msa_st_w((v4i32)_sum1, outptr0 + out_hstep * 1, 0);
                    __msa_st_w((v4i32)_sum2, outptr0 + out_hstep * 2, 0);
                    __msa_st_w((v4i32)_sum3, outptr0 + out_hstep * 3, 0);
                    outptr0 += 4;
                }
            }
            else
            {
                __msa_st_w((v4i32)_sum0, outptr, 0);
                __msa_st_w((v4i32)_sum1, outptr + 4, 0);
                __msa_st_w((v4i32)_sum2, outptr + 4 * 2, 0);
                __msa_st_w((v4i32)_sum3, outptr + 4 * 3, 0);
            }

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const float* pA = pAT;

            v4f32 _sum0;
            v4f32 _sum1;

            if (k == 0)
            {
                _sum0 = (v4f32)__msa_fill_w(0);
                _sum1 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _sum0 = (v4f32)__msa_ld_w(outptr, 0);
                _sum1 = (v4f32)__msa_ld_w(outptr + 4, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 8);
                v4f32 _pA = (v4f32)__msa_ld_w(pA, 0);
                v4f32 _pB = (v4f32)__msa_fill_d_ptr(pB);
                v4f32 _pB1 = (v4f32)__msa_shf_w((v4i32)_pB, _MSA_SHUFFLE(2, 3, 0, 1));

                _sum0 = __ncnn_msa_fmadd_w(_sum0, _pA, _pB);
                _sum1 = __ncnn_msa_fmadd_w(_sum1, _pA, _pB1);

                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                {
                    v4f32 _tmp0 = (v4f32)__msa_shf_w((v4i32)_sum0, _MSA_SHUFFLE(3, 1, 2, 0));
                    v4f32 _tmp1 = (v4f32)__msa_shf_w((v4i32)_sum1, _MSA_SHUFFLE(0, 2, 3, 1));
                    _sum0 = (v4f32)__msa_ilvr_w((v4i32)_tmp1, (v4i32)_tmp0);
                    _sum1 = (v4f32)__msa_ilvl_w((v4i32)_tmp1, (v4i32)_tmp0);
                    _sum1 = (v4f32)__msa_shf_w((v4i32)_sum1, _MSA_SHUFFLE(2, 1, 0, 3));
                }
                if (pC)
                {
                    v4f32 _bias = (v4f32)__msa_ld_w(pC, 0);
                    _sum0 = __msa_fadd_w(_sum0, _bias);
                    _sum1 = __msa_fadd_w(_sum1, _bias);
                }
                if (out_elempack == 4)
                {
                    __msa_st_w((v4i32)_sum0, outptr0, 0);
                    __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    float sum1[4];
                    __msa_st_w((v4i32)_sum0, sum0, 0);
                    __msa_st_w((v4i32)_sum1, sum1, 0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[1] = sum1[0];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0 += 2;
                }
            }
            else
            {
                __msa_st_w((v4i32)_sum0, outptr, 0);
                __msa_st_w((v4i32)_sum1, outptr + 4, 0);
            }

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            const float* pA = pAT;

            v4f32 _sum0;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (v4f32)__msa_ld_w(pC, 0);
                }
                else
                {
                    _sum0 = (v4f32)__msa_fill_w(0);
                }
            }
            else
            {
                _sum0 = (v4f32)__msa_ld_w(outptr, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 4);
                v4f32 _pA = (v4f32)__msa_ld_w(pA, 0);

                _sum0 = __ncnn_msa_fmadd_w(_sum0, _pA, __msa_fill_w_f32(pB[0]));

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    __msa_st_w((v4i32)_sum0, outptr0, 0);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    __msa_st_w((v4i32)_sum0, sum0, 0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0++;
                }
            }
            else
            {
                __msa_st_w((v4i32)_sum0, outptr, 0);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j;

        const float* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if __mips_msa
        for (; jj + 7 < max_jj; jj += 8)
        {
            v4f32 _sum00;
            v4f32 _sum01;
            v4f32 _sum10;
            v4f32 _sum11;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = __msa_fill_w_f32(pC[0]);
                    _sum01 = __msa_fill_w_f32(pC[0]);
                    _sum10 = __msa_fill_w_f32(pC[1]);
                    _sum11 = __msa_fill_w_f32(pC[1]);
                }
                else
                {
                    _sum00 = (v4f32)__msa_fill_w(0);
                    _sum01 = (v4f32)__msa_fill_w(0);
                    _sum10 = (v4f32)__msa_fill_w(0);
                    _sum11 = (v4f32)__msa_fill_w(0);
                }
            }
            else
            {
                v4f32 _tmp0 = (v4f32)__msa_ld_w(outptr, 0);
                v4f32 _tmp1 = (v4f32)__msa_ld_w(outptr + 4, 0);
                v4f32 _tmp2 = (v4f32)__msa_ld_w(outptr + 8, 0);
                v4f32 _tmp3 = (v4f32)__msa_ld_w(outptr + 12, 0);
                _sum00 = (v4f32)__msa_pckev_w((v4i32)_tmp1, (v4i32)_tmp0);
                _sum01 = (v4f32)__msa_pckev_w((v4i32)_tmp3, (v4i32)_tmp2);
                _sum10 = (v4f32)__msa_pckod_w((v4i32)_tmp1, (v4i32)_tmp0);
                _sum11 = (v4f32)__msa_pckod_w((v4i32)_tmp3, (v4i32)_tmp2);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 16);
                v4f32 _pB0 = (v4f32)__msa_ld_w(pB, 0);
                v4f32 _pB1 = (v4f32)__msa_ld_w(pB + 4, 0);

                v4f32 _pA0 = __msa_fill_w_f32(pA[0]);
                _sum00 = __ncnn_msa_fmadd_w(_sum00, _pA0, _pB0);
                _sum01 = __ncnn_msa_fmadd_w(_sum01, _pA0, _pB1);
                v4f32 _pA1 = __msa_fill_w_f32(pA[1]);
                _sum10 = __ncnn_msa_fmadd_w(_sum10, _pA1, _pB0);
                _sum11 = __ncnn_msa_fmadd_w(_sum11, _pA1, _pB1);

                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __msa_st_w((v4i32)_sum00, outptr0, 0);
                    __msa_st_w((v4i32)_sum01, outptr0 + 4, 0);
                    __msa_st_w((v4i32)_sum10, outptr0 + out_hstep, 0);
                    __msa_st_w((v4i32)_sum11, outptr0 + out_hstep + 4, 0);
                    outptr0 += 8;
                }
            }
            else
            {
                v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_sum10, (v4i32)_sum00);
                v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_sum10, (v4i32)_sum00);
                v4f32 _tmp2 = (v4f32)__msa_ilvr_w((v4i32)_sum11, (v4i32)_sum01);
                v4f32 _tmp3 = (v4f32)__msa_ilvl_w((v4i32)_sum11, (v4i32)_sum01);
                __msa_st_w((v4i32)_tmp0, outptr, 0);
                __msa_st_w((v4i32)_tmp1, outptr + 4, 0);
                __msa_st_w((v4i32)_tmp2, outptr + 8, 0);
                __msa_st_w((v4i32)_tmp3, outptr + 12, 0);
            }

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            v4f32 _sum0;
            v4f32 _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = __msa_fill_w_f32(pC[0]);
                    _sum1 = __msa_fill_w_f32(pC[1]);
                }
                else
                {
                    _sum0 = (v4f32)__msa_fill_w(0);
                    _sum1 = (v4f32)__msa_fill_w(0);
                }
            }
            else
            {
                v4f32 _tmp0 = (v4f32)__msa_ld_w(outptr, 0);
                v4f32 _tmp1 = (v4f32)__msa_ld_w(outptr + 4, 0);
                _sum0 = (v4f32)__msa_pckev_w((v4i32)_tmp1, (v4i32)_tmp0);
                _sum1 = (v4f32)__msa_pckod_w((v4i32)_tmp1, (v4i32)_tmp0);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 16);
                v4f32 _pB = (v4f32)__msa_ld_w(pB, 0);

                _sum0 = __ncnn_msa_fmadd_w(_sum0, __msa_fill_w_f32(pA[0]), _pB);
                _sum1 = __ncnn_msa_fmadd_w(_sum1, __msa_fill_w_f32(pA[1]), _pB);

                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __msa_st_w((v4i32)_sum0, outptr0, 0);
                    __msa_st_w((v4i32)_sum1, outptr0 + out_hstep, 0);
                    outptr0 += 4;
                }
            }
            else
            {
                v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_sum1, (v4i32)_sum0);
                v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_sum1, (v4i32)_sum0);
                __msa_st_w((v4i32)_tmp0, outptr, 0);
                __msa_st_w((v4i32)_tmp1, outptr + 4, 0);
            }

            outptr += 8;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum00;
            float sum01;
            float sum10;
            float sum11;

            if (k == 0)
            {
                if (pC)
                {
                    sum00 = pC[0];
                    sum01 = pC[1];
                    sum10 = pC[0];
                    sum11 = pC[1];
                }
                else
                {
                    sum00 = 0.f;
                    sum01 = 0.f;
                    sum10 = 0.f;
                    sum11 = 0.f;
                }
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum00 += pA[0] * pB[0];
                sum01 += pA[1] * pB[0];
                sum10 += pA[0] * pB[1];
                sum11 += pA[1] * pB[1];

                pA += 2;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum00;
                    outptr0[1] = sum10;
                    outptr0[out_hstep] = sum01;
                    outptr0[out_hstep + 1] = sum11;
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum01;
                outptr[2] = sum10;
                outptr[3] = sum11;
            }

            outptr += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[1];
                }
                else
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[0];
                pA += 2;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum0;
                    outptr0[out_hstep] = sum1;
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }

        pAT += max_kk * 2;
    }
    for (; ii < max_ii; ii += 1)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j;

        const float* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if __mips_msa
        for (; jj + 7 < max_jj; jj += 8)
        {
            v4f32 _sum0;
            v4f32 _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = __msa_fill_w_f32(pC[0]);
                    _sum1 = __msa_fill_w_f32(pC[0]);
                }
                else
                {
                    _sum0 = (v4f32)__msa_fill_w(0);
                    _sum1 = (v4f32)__msa_fill_w(0);
                }
            }
            else
            {
                _sum0 = (v4f32)__msa_ld_w(outptr, 0);
                _sum1 = (v4f32)__msa_ld_w(outptr + 4, 0);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 16);
                v4f32 _pB0 = (v4f32)__msa_ld_w(pB, 0);
                v4f32 _pB1 = (v4f32)__msa_ld_w(pB + 4, 0);

                v4f32 _pA0 = __msa_fill_w_f32(pA[0]);
                _sum0 = __ncnn_msa_fmadd_w(_sum0, _pA0, _pB0);
                _sum1 = __ncnn_msa_fmadd_w(_sum1, _pA0, _pB1);

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __msa_st_w((v4i32)_sum0, outptr0, 0);
                    __msa_st_w((v4i32)_sum1, outptr0 + 4, 0);
                    outptr0 += 8;
                }
            }
            else
            {
                __msa_st_w((v4i32)_sum0, outptr, 0);
                __msa_st_w((v4i32)_sum1, outptr + 4, 0);
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            v4f32 _sum;

            if (k == 0)
            {
                if (pC)
                {
                    _sum = __msa_fill_w_f32(pC[0]);
                }
                else
                {
                    _sum = (v4f32)__msa_fill_w(0);
                }
            }
            else
            {
                _sum = (v4f32)__msa_ld_w(outptr, 0);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 16);
                v4f32 _pB = (v4f32)__msa_ld_w(pB, 0);

                _sum = __ncnn_msa_fmadd_w(_sum, __msa_fill_w_f32(pA[0]), _pB);

                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __msa_st_w((v4i32)_sum, outptr0, 0);
                    outptr0 += 4;
                }
            }
            else
            {
                __msa_st_w((v4i32)_sum, outptr, 0);
            }

            outptr += 4;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[0];
                }
                else
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];

                pA += 1;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum0;
                    outptr0[1] = sum1;
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float sum;

            if (k == 0)
            {
                if (pC)
                {
                    sum = pC[0];
                }
                else
                {
                    sum = 0.f;
                }
            }
            else
            {
                sum = outptr[0];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum += pA[0] * pB[0];
                pA += 1;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum;
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum;
            }

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void convolution_im2col_gemm_get_optimal_tile_mnk(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const int l2_cache_size_fp32 = (int)(get_cpu_level2_cache_size() / sizeof(float));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve K
    {
        // try not to split K
#if __mips_msa
        int tile_size = (l2_cache_size_fp32 - 16) / 8;
#else
        int tile_size = (l2_cache_size_fp32 - 2) / 3;
#endif

#if __mips_msa
        TILE_K = std::max(4, tile_size / 4 * 4);
#else
        TILE_K = std::max(2, tile_size / 2 * 2);
#endif

        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __mips_msa
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif
    }

    // solve M
    {
#if __mips_msa
        int nn_M = (M + 7) / 8;
#else
        int nn_M = (M + 3) / 4;
#endif

#if __mips_msa
        TILE_M = std::max(8, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::max(2, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    {
        TILE_M *= std::min(nT, get_physical_cpu_count());

        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __mips_msa
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif

        if (nT > 1)
        {
#if __mips_msa
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#else
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
        }
    }

    if (N > 0)
    {
#if __mips_msa
        TILE_N = 8;
#else
        int tile_size;
        if (TILE_K >= K)
        {
            tile_size = (l2_cache_size_fp32 - TILE_M * TILE_K) / TILE_K;
        }
        else
        {
            tile_size = (l2_cache_size_fp32 - TILE_M * TILE_K) / (TILE_M + TILE_K);
        }

        TILE_N = std::max(1, tile_size);

        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);

        TILE_N = std::max(1, TILE_N);
#endif
    }
}

static void convolution_im2col_input_tile_conv1x1s1d1(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    const int elempack = bottom_blob.elempack;

    float* pp = B;

    int jj = 0;
#if __mips_msa
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                __builtin_prefetch(p0 + bottom_blob.cstep * 4);
                v4f32 _r0 = (v4f32)__msa_ld_w(p0, 0);
                v4f32 _r1 = (v4f32)__msa_ld_w(p0 + 4, 0);
                v4f32 _r2 = (v4f32)__msa_ld_w(p0 + 4 * 2, 0);
                v4f32 _r3 = (v4f32)__msa_ld_w(p0 + 4 * 3, 0);
                v4f32 _r4 = (v4f32)__msa_ld_w(p0 + 4 * 4, 0);
                v4f32 _r5 = (v4f32)__msa_ld_w(p0 + 4 * 5, 0);
                v4f32 _r6 = (v4f32)__msa_ld_w(p0 + 4 * 6, 0);
                v4f32 _r7 = (v4f32)__msa_ld_w(p0 + 4 * 7, 0);
                transpose4x4_ps(_r0, _r1, _r2, _r3);
                transpose4x4_ps(_r4, _r5, _r6, _r7);
                __msa_st_w((v4i32)_r0, pp, 0);
                __msa_st_w((v4i32)_r4, pp + 4 * 1, 0);
                __msa_st_w((v4i32)_r1, pp + 4 * 2, 0);
                __msa_st_w((v4i32)_r5, pp + 4 * 3, 0);
                __msa_st_w((v4i32)_r2, pp + 4 * 4, 0);
                __msa_st_w((v4i32)_r6, pp + 4 * 5, 0);
                __msa_st_w((v4i32)_r3, pp + 4 * 6, 0);
                __msa_st_w((v4i32)_r7, pp + 4 * 7, 0);
                pp += 32;
                p0 += bottom_blob.cstep * 4;
            }
        }

        if (elempack == 1)
        {
            const float* p0 = (const float*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __builtin_prefetch(p0 + bottom_blob.cstep);
                v4f32 _r0 = (v4f32)__msa_ld_w(p0, 0);
                v4f32 _r1 = (v4f32)__msa_ld_w(p0 + 4, 0);
                __msa_st_w((v4i32)_r0, pp, 0);
                __msa_st_w((v4i32)_r1, pp + 4, 0);
                pp += 8;
                p0 += bottom_blob.cstep;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                __builtin_prefetch(p0 + bottom_blob.cstep * 4);
                v4f32 _r0 = (v4f32)__msa_ld_w(p0, 0);
                v4f32 _r1 = (v4f32)__msa_ld_w(p0 + 4, 0);
                v4f32 _r2 = (v4f32)__msa_ld_w(p0 + 4 * 2, 0);
                v4f32 _r3 = (v4f32)__msa_ld_w(p0 + 4 * 3, 0);
                transpose4x4_ps(_r0, _r1, _r2, _r3);
                __msa_st_w((v4i32)_r0, pp, 0);
                __msa_st_w((v4i32)_r1, pp + 4 * 1, 0);
                __msa_st_w((v4i32)_r2, pp + 4 * 2, 0);
                __msa_st_w((v4i32)_r3, pp + 4 * 3, 0);
                pp += 16;
                p0 += bottom_blob.cstep * 4;
            }
        }

        if (elempack == 1)
        {
            const float* p0 = (const float*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __builtin_prefetch(p0 + bottom_blob.cstep);
                __msa_st_w((v4i32)(v4f32)__msa_ld_w(p0, 0), pp, 0);
                pp += 4;
                p0 += bottom_blob.cstep;
            }
        }
    }
#endif // __mips_msa
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __mips_msa
        if (elempack == 4)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                __builtin_prefetch(p0 + bottom_blob.cstep * 4);
                // transpose4x2
                v4f32 _r0 = (v4f32)__msa_ld_w(p0, 0);
                v4f32 _r1 = (v4f32)__msa_ld_w(p0 + 4, 0);
                v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
                v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
                __msa_st_w((v4i32)_tmp0, pp, 0);
                __msa_st_w((v4i32)_tmp1, pp + 4, 0);
                pp += 8;
                p0 += bottom_blob.cstep * 4;
            }
        }
#endif // __mips_msa

        if (elempack == 1)
        {
            const float* p0 = (const float*)bottom_blob.channel(k) + (j + jj);

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
#if __mips_msa
        if (elempack == 4)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                __builtin_prefetch(p0 + bottom_blob.cstep * 4);
                __msa_st_w((v4i32)(v4f32)__msa_ld_w(p0, 0), pp, 0);
                pp += 4;
                p0 += bottom_blob.cstep * 4;
            }
        }
#endif // __mips_msa

        if (elempack == 1)
        {
            const float* p0 = (const float*)bottom_blob.channel(k) + (j + jj);

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

static inline void convolution_im2col_input_tile_impl(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    const int w = bottom_blob.w;
    // const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int outw = (w - kernel_extent_w) / stride_w + 1;

    // j max_jj     outw*outh    split w and h

    // k max_kk     pa*maxk*(inch/pa)    split inch

    // k/max_kk shall be multiple of maxk

    const int maxk = kernel_w * kernel_h;

    float* pp = B;

    int jj = 0;
#if __mips_msa
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

                const float* sptr = img.row(y0) + x0 * elempack;

                if (elempack == 4)
                {
                    v4f32 _r0 = (v4f32)__msa_ld_w(sptr, 0);
                    v4f32 _r1 = (v4f32)__msa_ld_w(sptr + stride_w * 4, 0);
                    v4f32 _r2 = (v4f32)__msa_ld_w(sptr + stride_w * 8, 0);
                    v4f32 _r3 = (v4f32)__msa_ld_w(sptr + stride_w * 12, 0);
                    v4f32 _r4 = (v4f32)__msa_ld_w(sptr + stride_w * 16, 0);
                    v4f32 _r5 = (v4f32)__msa_ld_w(sptr + stride_w * 20, 0);
                    v4f32 _r6 = (v4f32)__msa_ld_w(sptr + stride_w * 24, 0);
                    v4f32 _r7 = (v4f32)__msa_ld_w(sptr + stride_w * 28, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    __msa_st_w((v4i32)_r0, pp, 0);
                    __msa_st_w((v4i32)_r4, pp + 4 * 1, 0);
                    __msa_st_w((v4i32)_r1, pp + 4 * 2, 0);
                    __msa_st_w((v4i32)_r5, pp + 4 * 3, 0);
                    __msa_st_w((v4i32)_r2, pp + 4 * 4, 0);
                    __msa_st_w((v4i32)_r6, pp + 4 * 5, 0);
                    __msa_st_w((v4i32)_r3, pp + 4 * 6, 0);
                    __msa_st_w((v4i32)_r7, pp + 4 * 7, 0);
                    pp += 32;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
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

                const float* sptr0 = img.row(y0) + x0 * elempack;
                const float* sptr1 = img.row(y1) + x1 * elempack;
                const float* sptr2 = img.row(y2) + x2 * elempack;
                const float* sptr3 = img.row(y3) + x3 * elempack;
                const float* sptr4 = img.row(y4) + x4 * elempack;
                const float* sptr5 = img.row(y5) + x5 * elempack;
                const float* sptr6 = img.row(y6) + x6 * elempack;
                const float* sptr7 = img.row(y7) + x7 * elempack;

                if (elempack == 4)
                {
                    v4f32 _r0 = (v4f32)__msa_ld_w(sptr0, 0);
                    v4f32 _r1 = (v4f32)__msa_ld_w(sptr1, 0);
                    v4f32 _r2 = (v4f32)__msa_ld_w(sptr2, 0);
                    v4f32 _r3 = (v4f32)__msa_ld_w(sptr3, 0);
                    v4f32 _r4 = (v4f32)__msa_ld_w(sptr4, 0);
                    v4f32 _r5 = (v4f32)__msa_ld_w(sptr5, 0);
                    v4f32 _r6 = (v4f32)__msa_ld_w(sptr6, 0);
                    v4f32 _r7 = (v4f32)__msa_ld_w(sptr7, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    __msa_st_w((v4i32)_r0, pp, 0);
                    __msa_st_w((v4i32)_r4, pp + 4 * 1, 0);
                    __msa_st_w((v4i32)_r1, pp + 4 * 2, 0);
                    __msa_st_w((v4i32)_r5, pp + 4 * 3, 0);
                    __msa_st_w((v4i32)_r2, pp + 4 * 4, 0);
                    __msa_st_w((v4i32)_r6, pp + 4 * 5, 0);
                    __msa_st_w((v4i32)_r3, pp + 4 * 6, 0);
                    __msa_st_w((v4i32)_r7, pp + 4 * 7, 0);
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

                const float* sptr = img.row(y0) + x0 * elempack;

                if (elempack == 4)
                {
                    v4f32 _r0 = (v4f32)__msa_ld_w(sptr, 0);
                    v4f32 _r1 = (v4f32)__msa_ld_w(sptr + stride_w * 4, 0);
                    v4f32 _r2 = (v4f32)__msa_ld_w(sptr + stride_w * 8, 0);
                    v4f32 _r3 = (v4f32)__msa_ld_w(sptr + stride_w * 12, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __msa_st_w((v4i32)_r0, pp, 0);
                    __msa_st_w((v4i32)_r1, pp + 4 * 1, 0);
                    __msa_st_w((v4i32)_r2, pp + 4 * 2, 0);
                    __msa_st_w((v4i32)_r3, pp + 4 * 3, 0);
                    pp += 16;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
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

                const float* sptr0 = img.row(y0) + x0 * elempack;
                const float* sptr1 = img.row(y1) + x1 * elempack;
                const float* sptr2 = img.row(y2) + x2 * elempack;
                const float* sptr3 = img.row(y3) + x3 * elempack;

                if (elempack == 4)
                {
                    v4f32 _r0 = (v4f32)__msa_ld_w(sptr0, 0);
                    v4f32 _r1 = (v4f32)__msa_ld_w(sptr1, 0);
                    v4f32 _r2 = (v4f32)__msa_ld_w(sptr2, 0);
                    v4f32 _r3 = (v4f32)__msa_ld_w(sptr3, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __msa_st_w((v4i32)_r0, pp, 0);
                    __msa_st_w((v4i32)_r1, pp + 4 * 1, 0);
                    __msa_st_w((v4i32)_r2, pp + 4 * 2, 0);
                    __msa_st_w((v4i32)_r3, pp + 4 * 3, 0);
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
#endif // __mips_msa
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

                const float* sptr = img.row(y0) + x0 * elempack;

#if __mips_msa
                if (elempack == 4)
                {
                    v4f32 _r0 = (v4f32)__msa_ld_w(sptr, 0);
                    v4f32 _r1 = (v4f32)__msa_ld_w(sptr + stride_w * 4, 0);
                    v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
                    v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
                    __msa_st_w((v4i32)_tmp0, pp, 0);
                    __msa_st_w((v4i32)_tmp1, pp + 4, 0);
                    pp += 8;
                }
#endif // __mips_msa
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
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

                const float* sptr0 = img.row(y0) + x0 * elempack;
                const float* sptr1 = img.row(y1) + x1 * elempack;

#if __mips_msa
                if (elempack == 4)
                {
                    v4f32 _r0 = (v4f32)__msa_ld_w(sptr0, 0);
                    v4f32 _r1 = (v4f32)__msa_ld_w(sptr1, 0);
                    v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
                    v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
                    __msa_st_w((v4i32)_tmp0, pp, 0);
                    __msa_st_w((v4i32)_tmp1, pp + 4, 0);
                    pp += 8;
                }
#endif // __mips_msa
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp += 2;
                }
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
        int dy = (j + jj) / outw;
        int dx = (j + jj) % outw;

        int kk = 0;
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x = stride_w * dx + dilation_w * v;
            int y = stride_h * dy + dilation_h * u;

            const float* sptr = img.row(y) + x * elempack;

#if __mips_msa
            if (elempack == 4)
            {
                __msa_st_w((v4i32)(v4f32)__msa_ld_w(sptr, 0), pp, 0);
                pp += 4;
            }
#endif // __mips_msa
            if (elempack == 1)
            {
                pp[0] = sptr[0];
                pp += 1;
            }
        }
    }
}

template<int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h>
static void convolution_im2col_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    convolution_im2col_input_tile_impl(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

template void convolution_im2col_input_tile<1, 1, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile<3, 3, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile<3, 3, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile<5, 5, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile<5, 5, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile<7, 7, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);

static void convolution_im2col_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_conv1x1s1d1(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 1 && kernel_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    convolution_im2col_input_tile_impl(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

static void convolution_im2col_gemm_transform_kernel(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    // NCNN_LOGE("convolution_im2col_gemm_transform_kernel");
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    int elempack = 1;
#if __mips_msa
    if (opt.use_packing_layout)
    {
        elempack = inch % 4 == 0 ? 4 : 1;
    }
#endif // __mips_msa

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

    AT.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            convolution_im2col_pack_A_tile(A_data, AT_tile, i, max_ii, k, max_kk);
        }
    }
}

static int convolution_im2col_gemm(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
    const int maxk = kernel_w * kernel_h;

    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.workspace_allocator);
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
        convolution_im2col_input_tile(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
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

                convolution_gemm_transB_packed_tile(AT_tile, BT_tile, bias, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end);
            }
        }
    }

    return 0;
}

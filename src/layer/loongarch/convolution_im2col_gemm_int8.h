// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convolution_im2col_pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    // A = (pa, maxk, inch/pa), outch
    const int A_hstep = A.w;

    signed char* pp = AT;

    int ii = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;
        const signed char* p2 = (const signed char*)A + (i + ii + 2) * A_hstep + k;
        const signed char* p3 = (const signed char*)A + (i + ii + 3) * A_hstep + k;
        const signed char* p4 = (const signed char*)A + (i + ii + 4) * A_hstep + k;
        const signed char* p5 = (const signed char*)A + (i + ii + 5) * A_hstep + k;
        const signed char* p6 = (const signed char*)A + (i + ii + 6) * A_hstep + k;
        const signed char* p7 = (const signed char*)A + (i + ii + 7) * A_hstep + k;

        int kk = 0;
        for (; kk + 7 < max_kk; kk += 8)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[4] = p4[0];
            pp[5] = p5[0];
            pp[6] = p6[0];
            pp[7] = p7[0];
            pp[8] = p0[1];
            pp[9] = p1[1];
            pp[10] = p2[1];
            pp[11] = p3[1];
            pp[12] = p4[1];
            pp[13] = p5[1];
            pp[14] = p6[1];
            pp[15] = p7[1];
            pp[16] = p0[2];
            pp[17] = p1[2];
            pp[18] = p2[2];
            pp[19] = p3[2];
            pp[20] = p4[2];
            pp[21] = p5[2];
            pp[22] = p6[2];
            pp[23] = p7[2];
            pp[24] = p0[3];
            pp[25] = p1[3];
            pp[26] = p2[3];
            pp[27] = p3[3];
            pp[28] = p4[3];
            pp[29] = p5[3];
            pp[30] = p6[3];
            pp[31] = p7[3];
            pp[32] = p0[4];
            pp[33] = p1[4];
            pp[34] = p2[4];
            pp[35] = p3[4];
            pp[36] = p4[4];
            pp[37] = p5[4];
            pp[38] = p6[4];
            pp[39] = p7[4];
            pp[40] = p0[5];
            pp[41] = p1[5];
            pp[42] = p2[5];
            pp[43] = p3[5];
            pp[44] = p4[5];
            pp[45] = p5[5];
            pp[46] = p6[5];
            pp[47] = p7[5];
            pp[48] = p0[6];
            pp[49] = p1[6];
            pp[50] = p2[6];
            pp[51] = p3[6];
            pp[52] = p4[6];
            pp[53] = p5[6];
            pp[54] = p6[6];
            pp[55] = p7[6];
            pp[56] = p0[7];
            pp[57] = p1[7];
            pp[58] = p2[7];
            pp[59] = p3[7];
            pp[60] = p4[7];
            pp[61] = p5[7];
            pp[62] = p6[7];
            pp[63] = p7[7];
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
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[4] = p4[0];
            pp[5] = p5[0];
            pp[6] = p6[0];
            pp[7] = p7[0];
            pp[8] = p0[1];
            pp[9] = p1[1];
            pp[10] = p2[1];
            pp[11] = p3[1];
            pp[12] = p4[1];
            pp[13] = p5[1];
            pp[14] = p6[1];
            pp[15] = p7[1];
            pp[16] = p0[2];
            pp[17] = p1[2];
            pp[18] = p2[2];
            pp[19] = p3[2];
            pp[20] = p4[2];
            pp[21] = p5[2];
            pp[22] = p6[2];
            pp[23] = p7[2];
            pp[24] = p0[3];
            pp[25] = p1[3];
            pp[26] = p2[3];
            pp[27] = p3[3];
            pp[28] = p4[3];
            pp[29] = p5[3];
            pp[30] = p6[3];
            pp[31] = p7[3];
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
#endif // __loongarch_asx
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;
        const signed char* p2 = (const signed char*)A + (i + ii + 2) * A_hstep + k;
        const signed char* p3 = (const signed char*)A + (i + ii + 3) * A_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[4] = p0[1];
            pp[5] = p1[1];
            pp[6] = p2[1];
            pp[7] = p3[1];
            pp[8] = p0[2];
            pp[9] = p1[2];
            pp[10] = p2[2];
            pp[11] = p3[2];
            pp[12] = p0[3];
            pp[13] = p1[3];
            pp[14] = p2[3];
            pp[15] = p3[3];
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < max_kk; kk++)
        {
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
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
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
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void convolution_gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
    // NCNN_LOGE("convolution_gemm_transB_packed_tile_int8 %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const signed char* pAT = AT_tile;
    const signed char* pBT = BT_tile;

    int* outptr = topT_tile;

    int ii = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* pB = pBT;

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            __m256i _sum0;
            __m256i _sum1;
            __m256i _sum2;
            __m256i _sum3;

            if (k == 0)
            {
                _sum0 = __lasx_xvreplgr2vr_w(0);
                _sum1 = __lasx_xvreplgr2vr_w(0);
                _sum2 = __lasx_xvreplgr2vr_w(0);
                _sum3 = __lasx_xvreplgr2vr_w(0);
            }
            else
            {
                _sum0 = __lasx_xvld(outptr, 0);
                _sum1 = __lasx_xvld(outptr + 8, 0);
                _sum2 = __lasx_xvld(outptr + 16, 0);
                _sum3 = __lasx_xvld(outptr + 24, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256i _pA = __lasx_xvreplgr2vr_d(*(long long*)pA);
                _pA = __lasx_xvilvl_b(__lasx_xvslti_b(_pA, 0), _pA);

                __m256i _pB0 = __lasx_xvreplgr2vr_h(pB[0]);
                __m256i _pB1 = __lasx_xvreplgr2vr_h(pB[1]);
                __m256i _pB2 = __lasx_xvreplgr2vr_h(pB[2]);
                __m256i _pB3 = __lasx_xvreplgr2vr_h(pB[3]);

                __m256i _s0 = __lasx_xvmul_h(_pA, _pB0);
                __m256i _s1 = __lasx_xvmul_h(_pA, _pB1);
                __m256i _s2 = __lasx_xvmul_h(_pA, _pB2);
                __m256i _s3 = __lasx_xvmul_h(_pA, _pB3);

                _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s0, _s0));
                _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvhaddw_w_h(_s1, _s1));
                _sum2 = __lasx_xvadd_w(_sum2, __lasx_xvhaddw_w_h(_s2, _s2));
                _sum3 = __lasx_xvadd_w(_sum3, __lasx_xvhaddw_w_h(_s3, _s3));

                pA += 8;
                pB += 4;
            }

            __lasx_xvst(_sum0, outptr, 0);
            __lasx_xvst(_sum1, outptr + 8, 0);
            __lasx_xvst(_sum2, outptr + 16, 0);
            __lasx_xvst(_sum3, outptr + 24, 0);

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            __m256i _sum0;
            __m256i _sum1;

            if (k == 0)
            {
                _sum0 = __lasx_xvreplgr2vr_w(0);
                _sum1 = __lasx_xvreplgr2vr_w(0);
            }
            else
            {
                _sum0 = __lasx_xvld(outptr, 0);
                _sum1 = __lasx_xvld(outptr + 8, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256i _pA = __lasx_xvreplgr2vr_d(*(long long*)pA);
                _pA = __lasx_xvilvl_b(__lasx_xvslti_b(_pA, 0), _pA);

                __m256i _pB0 = __lasx_xvreplgr2vr_h(pB[0]);
                __m256i _pB1 = __lasx_xvreplgr2vr_h(pB[1]);

                __m256i _s0 = __lasx_xvmul_h(_pA, _pB0);
                __m256i _s1 = __lasx_xvmul_h(_pA, _pB1);

                _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s0, _s0));
                _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvhaddw_w_h(_s1, _s1));

                pA += 8;
                pB += 2;
            }

            __lasx_xvst(_sum0, outptr, 0);
            __lasx_xvst(_sum1, outptr + 8, 0);

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            __m256i _sum0;

            if (k == 0)
            {
                _sum0 = __lasx_xvreplgr2vr_w(0);
            }
            else
            {
                _sum0 = __lasx_xvld(outptr, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256i _pA = __lasx_xvreplgr2vr_d(*(long long*)pA);
                _pA = __lasx_xvilvl_b(__lasx_xvslti_b(_pA, 0), _pA);

                __m256i _pB0 = __lasx_xvreplgr2vr_h(pB[0]);

                __m256i _s0 = __lasx_xvmul_h(_pA, _pB0);

                _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s0, _s0));

                pA += 8;
                pB += 1;
            }

            __lasx_xvst(_sum0, outptr, 0);

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
#endif // __loongarch_asx
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* pB = pBT;

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            __m128i _sum0;
            __m128i _sum1;
            __m128i _sum2;
            __m128i _sum3;

            if (k == 0)
            {
                _sum0 = __lsx_vreplgr2vr_w(0);
                _sum1 = __lsx_vreplgr2vr_w(0);
                _sum2 = __lsx_vreplgr2vr_w(0);
                _sum3 = __lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum0 = __lsx_vld(outptr, 0);
                _sum1 = __lsx_vld(outptr + 4, 0);
                _sum2 = __lsx_vld(outptr + 8, 0);
                _sum3 = __lsx_vld(outptr + 12, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = __lsx_vreplgr2vr_d(*(int*)pA);
                _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);

                __m128i _pB0 = __lsx_vreplgr2vr_h(pB[0]);
                __m128i _pB1 = __lsx_vreplgr2vr_h(pB[1]);
                __m128i _pB2 = __lsx_vreplgr2vr_h(pB[2]);
                __m128i _pB3 = __lsx_vreplgr2vr_h(pB[3]);

                __m128i _s0 = __lsx_vmul_h(_pA, _pB0);
                __m128i _s1 = __lsx_vmul_h(_pA, _pB1);
                __m128i _s2 = __lsx_vmul_h(_pA, _pB2);
                __m128i _s3 = __lsx_vmul_h(_pA, _pB3);

                _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                _sum2 = __lsx_vadd_w(_sum2, __lsx_vilvl_h(__lsx_vslti_h(_s2, 0), _s2));
                _sum3 = __lsx_vadd_w(_sum3, __lsx_vilvl_h(__lsx_vslti_h(_s3, 0), _s3));

                pA += 4;
                pB += 4;
            }

            __lsx_vst(_sum0, outptr, 0);
            __lsx_vst(_sum1, outptr + 4, 0);
            __lsx_vst(_sum2, outptr + 8, 0);
            __lsx_vst(_sum3, outptr + 12, 0);

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            __m128i _sum0;
            __m128i _sum1;

            if (k == 0)
            {
                _sum0 = __lsx_vreplgr2vr_w(0);
                _sum1 = __lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum0 = __lsx_vld(outptr, 0);
                _sum1 = __lsx_vld(outptr + 4, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = __lsx_vreplgr2vr_d(*(int*)pA);
                _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);

                __m128i _pB0 = __lsx_vreplgr2vr_h(pB[0]);
                __m128i _pB1 = __lsx_vreplgr2vr_h(pB[1]);

                __m128i _s0 = __lsx_vmul_h(_pA, _pB0);
                __m128i _s1 = __lsx_vmul_h(_pA, _pB1);

                _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                pA += 4;
                pB += 2;
            }

            __lsx_vst(_sum0, outptr, 0);
            __lsx_vst(_sum1, outptr + 4, 0);

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            __m128i _sum0;

            if (k == 0)
            {
                _sum0 = __lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum0 = __lsx_vld(outptr, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = __lsx_vreplgr2vr_d(*(int*)pA);
                _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);

                __m128i _pB0 = __lsx_vreplgr2vr_h(pB[0]);

                __m128i _s0 = __lsx_vmul_h(_pA, _pB0);

                _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));

                pA += 4;
                pB += 1;
            }

            __lsx_vst(_sum0, outptr, 0);

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB = pBT;

        int jj = 0;
        for (; jj + 1 < max_jj; jj += 2)
        {
            int sum00;
            int sum01;
            int sum10;
            int sum11;

            if (k == 0)
            {
                sum00 = 0;
                sum01 = 0;
                sum10 = 0;
                sum11 = 0;
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            const signed char* pA = pAT;
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

            outptr[0] = sum00;
            outptr[1] = sum01;
            outptr[2] = sum10;
            outptr[3] = sum11;

            outptr += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            int sum0;
            int sum1;

            if (k == 0)
            {
                sum0 = 0;
                sum1 = 0;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[0];
                pA += 2;
                pB += 1;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;

            outptr += 2;
        }

        pAT += max_kk * 2;
    }
    for (; ii < max_ii; ii += 1)
    {
        const signed char* pB = pBT;

        int jj = 0;
        for (; jj + 1 < max_jj; jj += 2)
        {
            int sum0;
            int sum1;

            if (k == 0)
            {
                sum0 = 0;
                sum1 = 0;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];

                pA += 1;
                pB += 2;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            int sum;

            if (k == 0)
            {
                sum = 0;
            }
            else
            {
                sum = outptr[0];
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum += pA[0] * pB[0];
                pA += 1;
                pB += 1;
            }

            outptr[0] = sum;

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void unpack_output_tile_int32(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    const int out_elempack = top_blob.elempack;
    const int out_hstep = (int)top_blob.cstep;

    const int* pp = topT;

    int ii = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        int* p0 = (int*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m256i _f0 = __lasx_xvld(pp, 0);
            __m256i _f1 = __lasx_xvld(pp + 8, 0);
            __m256i _f2 = __lasx_xvld(pp + 16, 0);
            __m256i _f3 = __lasx_xvld(pp + 24, 0);

            if (out_elempack == 4)
            {
                // ii+0 ii+1 ii+2 ii+3 | ii+4 ii+5 ii+6 ii+7  -> p0[0..3] p0[4..7] for jj+0
                // need to deinterleave and store
                __lsx_vst(__lasx_extract_lo128(_f0), p0, 0);
                __lsx_vst(__lasx_extract_hi128(_f0), p0 + 4, 0);
                __lsx_vst(__lasx_extract_lo128(_f1), p0 + 8, 0);
                __lsx_vst(__lasx_extract_hi128(_f1), p0 + 12, 0);
                __lsx_vst(__lasx_extract_lo128(_f2), p0 + 16, 0);
                __lsx_vst(__lasx_extract_hi128(_f2), p0 + 20, 0);
                __lsx_vst(__lasx_extract_lo128(_f3), p0 + 24, 0);
                __lsx_vst(__lasx_extract_hi128(_f3), p0 + 28, 0);
                p0 += 32;
            }
            if (out_elempack == 1)
            {
                // transpose 8x4
                __m256i _tmp0 = __lasx_xvilvl_w(_f1, _f0);
                __m256i _tmp1 = __lasx_xvilvh_w(_f1, _f0);
                __m256i _tmp2 = __lasx_xvilvl_w(_f3, _f2);
                __m256i _tmp3 = __lasx_xvilvh_w(_f3, _f2);
                _f0 = __lasx_xvilvl_d(_tmp2, _tmp0);
                _f1 = __lasx_xvilvh_d(_tmp2, _tmp0);
                _f2 = __lasx_xvilvl_d(_tmp3, _tmp1);
                _f3 = __lasx_xvilvh_d(_tmp3, _tmp1);

                __lsx_vst(__lasx_extract_lo128(_f0), p0, 0);
                __lsx_vst(__lasx_extract_hi128(_f0), p0 + 4, 0);
                __lsx_vst(__lasx_extract_lo128(_f1), p0 + out_hstep, 0);
                __lsx_vst(__lasx_extract_hi128(_f1), p0 + out_hstep + 4, 0);
                __lsx_vst(__lasx_extract_lo128(_f2), p0 + out_hstep * 2, 0);
                __lsx_vst(__lasx_extract_hi128(_f2), p0 + out_hstep * 2 + 4, 0);
                __lsx_vst(__lasx_extract_lo128(_f3), p0 + out_hstep * 3, 0);
                __lsx_vst(__lasx_extract_hi128(_f3), p0 + out_hstep * 3 + 4, 0);
                p0 += 4;
            }

            pp += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m256i _f0 = __lasx_xvld(pp, 0);
            __m256i _f1 = __lasx_xvld(pp + 8, 0);

            if (out_elempack == 4)
            {
                __lsx_vst(__lasx_extract_lo128(_f0), p0, 0);
                __lsx_vst(__lasx_extract_hi128(_f0), p0 + 4, 0);
                __lsx_vst(__lasx_extract_lo128(_f1), p0 + 8, 0);
                __lsx_vst(__lasx_extract_hi128(_f1), p0 + 12, 0);
                p0 += 16;
            }
            if (out_elempack == 1)
            {
                int tmp[16];
                __lasx_xvst(_f0, tmp, 0);
                __lasx_xvst(_f1, tmp + 8, 0);

                p0[0] = tmp[0];
                p0[1] = tmp[8];
                p0[out_hstep] = tmp[1];
                p0[out_hstep + 1] = tmp[9];
                p0[out_hstep * 2] = tmp[2];
                p0[out_hstep * 2 + 1] = tmp[10];
                p0[out_hstep * 3] = tmp[3];
                p0[out_hstep * 3 + 1] = tmp[11];
                p0[out_hstep * 4] = tmp[4];
                p0[out_hstep * 4 + 1] = tmp[12];
                p0[out_hstep * 5] = tmp[5];
                p0[out_hstep * 5 + 1] = tmp[13];
                p0[out_hstep * 6] = tmp[6];
                p0[out_hstep * 6 + 1] = tmp[14];
                p0[out_hstep * 7] = tmp[7];
                p0[out_hstep * 7 + 1] = tmp[15];
                p0 += 2;
            }

            pp += 16;
        }
        for (; jj < max_jj; jj++)
        {
            __m256i _f0 = __lasx_xvld(pp, 0);

            if (out_elempack == 4)
            {
                __lsx_vst(__lasx_extract_lo128(_f0), p0, 0);
                __lsx_vst(__lasx_extract_hi128(_f0), p0 + 4, 0);
                p0 += 8;
            }
            if (out_elempack == 1)
            {
                int tmp[8];
                __lasx_xvst(_f0, tmp, 0);

                p0[0] = tmp[0];
                p0[out_hstep] = tmp[1];
                p0[out_hstep * 2] = tmp[2];
                p0[out_hstep * 3] = tmp[3];
                p0[out_hstep * 4] = tmp[4];
                p0[out_hstep * 5] = tmp[5];
                p0[out_hstep * 6] = tmp[6];
                p0[out_hstep * 7] = tmp[7];
                p0++;
            }

            pp += 8;
        }
    }
#endif // __loongarch_asx
    for (; ii + 3 < max_ii; ii += 4)
    {
        int* p0 = (int*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _f0 = __lsx_vld(pp, 0);
            __m128i _f1 = __lsx_vld(pp + 4, 0);
            __m128i _f2 = __lsx_vld(pp + 8, 0);
            __m128i _f3 = __lsx_vld(pp + 12, 0);

            if (out_elempack == 4)
            {
                __lsx_vst(_f0, p0, 0);
                __lsx_vst(_f1, p0 + 4, 0);
                __lsx_vst(_f2, p0 + 8, 0);
                __lsx_vst(_f3, p0 + 12, 0);
                p0 += 16;
            }
            if (out_elempack == 1)
            {
                transpose4x4_ps((__m128&)_f0, (__m128&)_f1, (__m128&)_f2, (__m128&)_f3);

                __lsx_vst(_f0, p0, 0);
                __lsx_vst(_f1, p0 + out_hstep, 0);
                __lsx_vst(_f2, p0 + out_hstep * 2, 0);
                __lsx_vst(_f3, p0 + out_hstep * 3, 0);
                p0 += 4;
            }

            pp += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128i _f0 = __lsx_vld(pp, 0);
            __m128i _f1 = __lsx_vld(pp + 4, 0);

            if (out_elempack == 4)
            {
                __lsx_vst(_f0, p0, 0);
                __lsx_vst(_f1, p0 + 4, 0);
                p0 += 8;
            }
            if (out_elempack == 1)
            {
                int tmp[8];
                __lsx_vst(_f0, tmp, 0);
                __lsx_vst(_f1, tmp + 4, 0);

                p0[0] = tmp[0];
                p0[out_hstep] = tmp[1];
                p0[out_hstep * 2] = tmp[2];
                p0[out_hstep * 3] = tmp[3];
                p0[1] = tmp[4];
                p0[out_hstep + 1] = tmp[5];
                p0[out_hstep * 2 + 1] = tmp[6];
                p0[out_hstep * 3 + 1] = tmp[7];
                p0 += 2;
            }

            pp += 8;
        }
        for (; jj < max_jj; jj++)
        {
            __m128i _f0 = __lsx_vld(pp, 0);

            if (out_elempack == 4)
            {
                __lsx_vst(_f0, p0, 0);
                p0 += 4;
            }
            if (out_elempack == 1)
            {
                int tmp[4];
                __lsx_vst(_f0, tmp, 0);

                p0[0] = tmp[0];
                p0[out_hstep] = tmp[1];
                p0[out_hstep * 2] = tmp[2];
                p0[out_hstep * 3] = tmp[3];
                p0++;
            }

            pp += 4;
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        int* p0 = (int*)top_blob + (i + ii) * out_hstep + j;

        int jj = 0;
        for (; jj + 1 < max_jj; jj += 2)
        {
            p0[0] = pp[0];
            p0[1] = pp[2];
            p0[out_hstep] = pp[1];
            p0[out_hstep + 1] = pp[3];
            p0 += 2;

            pp += 4;
        }
        for (; jj < max_jj; jj++)
        {
            p0[0] = pp[0];
            p0[out_hstep] = pp[1];
            p0++;

            pp += 2;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        int* p0 = (int*)top_blob + (i + ii) * out_hstep + j;

        int jj = 0;
        for (; jj + 1 < max_jj; jj += 2)
        {
            p0[0] = pp[0];
            p0[1] = pp[1];
            p0 += 2;

            pp += 2;
        }
        for (; jj < max_jj; jj++)
        {
            p0[0] = pp[0];
            p0++;

            pp += 1;
        }
    }
}

static void convolution_im2col_gemm_get_optimal_tile_mnk_int8(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const int l2_cache_size_int8 = (int)(get_cpu_level2_cache_size() / sizeof(signed char));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve K
    {
        // try not to split K
#if __loongarch_sx
        int tile_size = (l2_cache_size_int8 - 16) / 8;
#else
        int tile_size = (l2_cache_size_int8 - 2) / 3;
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
        int nn_M = (M + 15) / 16;
#else
        int nn_M = (M + 7) / 8;
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
            tile_size = (l2_cache_size_int8 - TILE_M * TILE_K) / TILE_K;
        }
        else
        {
            tile_size = (l2_cache_size_int8 - TILE_M * TILE_K) / (TILE_M + TILE_K);
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

static void convolution_im2col_input_tile_conv1x1s1d1_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    const int elempack = bottom_blob.elempack;

    signed char* pp = B;

    int jj = 0;
#if __loongarch_sx
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += bottom_blob.cstep;
            }
        }
    }
#endif // __loongarch_sx
    for (; jj + 1 < max_jj; jj += 2)
    {
        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

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
        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

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

static inline void convolution_im2col_input_tile_impl_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
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

    signed char* pp = B;

    int jj = 0;
#if __loongarch_sx
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

                const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

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

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;
                const signed char* sptr2 = img.row<const signed char>(y2) + x2 * elempack;
                const signed char* sptr3 = img.row<const signed char>(y3) + x3 * elempack;

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

                const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

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

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;

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

            const signed char* sptr = img.row<const signed char>(y) + x * elempack;

            if (elempack == 1)
            {
                pp[0] = sptr[0];
                pp += 1;
            }
        }
    }
}

template<int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h>
void convolution_im2col_input_tile_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    convolution_im2col_input_tile_impl_int8(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

template void convolution_im2col_input_tile_int8<1, 1, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<3, 3, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<3, 3, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<5, 5, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<5, 5, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<7, 7, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);

static void convolution_im2col_input_tile_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_conv1x1s1d1_int8(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 1 && kernel_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_int8<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_int8<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_int8<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_int8<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_int8<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_int8<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    convolution_im2col_input_tile_impl_int8(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

static void convolution_im2col_gemm_transform_kernel_int8(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    // NCNN_LOGE("convolution_im2col_gemm_transform_kernel_int8");
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_int8(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    // maxk-inch-outch to pa-maxk-inch/pa-outch
    Mat A_data;
    if (maxk == 1)
    {
        A_data = kernel.reshape(maxk * inch, outch);
    }
    else
    {
        Mat weight_data_r2 = kernel.reshape(maxk, inch, outch);

        A_data.create(maxk * inch, outch, (size_t)1u, 1);

        for (int q = 0; q < outch; q += 1)
        {
            signed char* g00 = A_data.row<signed char>(q);

            for (int p = 0; p < inch; p += 1)
            {
                for (int k = 0; k < maxk; k++)
                {
                    const signed char* k00 = weight_data_r2.channel(q).row<const signed char>(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }

    AT.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, (size_t)1u, 1);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            convolution_im2col_pack_A_tile_int8(A_data, AT_tile, i, max_ii, k, max_kk);
        }
    }
}

static int convolution_im2col_gemm_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
    const int maxk = kernel_w * kernel_h;

    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_int8(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 1u, opt.workspace_allocator);
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
        convolution_im2col_input_tile_int8(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
    }

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                const Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                convolution_gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_int32(topT_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

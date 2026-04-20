// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pack_A_tile_int8(const Mat& A, Mat& AT, int batch, int max_ii, int max_kk)
{
    const int N = max_kk * batch;

    for (int b = 0; b < batch; b++)
    {
        short* pp = AT.row<short>(b);

        int ii = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; ii + 7 < max_ii; ii += 8)
        {
            const short* p0 = (const short*)A + ii * N + b;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[batch];
                pp[2] = p0[N];
                pp[3] = p0[N + batch];
                pp[4] = p0[N * 2];
                pp[5] = p0[N * 2 + batch];
                pp[6] = p0[N * 3];
                pp[7] = p0[N * 3 + batch];
                pp[8] = p0[N * 4];
                pp[9] = p0[N * 4 + batch];
                pp[10] = p0[N * 5];
                pp[11] = p0[N * 5 + batch];
                pp[12] = p0[N * 6];
                pp[13] = p0[N * 6 + batch];
                pp[14] = p0[N * 7];
                pp[15] = p0[N * 7 + batch];
                p0 += batch * 2;
                pp += 16;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[N];
                pp[2] = p0[N * 2];
                pp[3] = p0[N * 3];
                pp[4] = p0[N * 4];
                pp[5] = p0[N * 5];
                pp[6] = p0[N * 6];
                pp[7] = p0[N * 7];
                p0 += batch;
                pp += 8;
            }
        }
#endif // __loongarch_asx
        for (; ii + 3 < max_ii; ii += 4)
        {
            const short* p0 = (const short*)A + ii * N + b;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[batch];
                pp[2] = p0[N];
                pp[3] = p0[N + batch];
                pp[4] = p0[N * 2];
                pp[5] = p0[N * 2 + batch];
                pp[6] = p0[N * 3];
                pp[7] = p0[N * 3 + batch];
                p0 += batch * 2;
                pp += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[N];
                pp[2] = p0[N * 2];
                pp[3] = p0[N * 3];
                p0 += batch;
                pp += 4;
            }
        }
#endif // __loongarch_sx
        for (; ii + 1 < max_ii; ii += 2)
        {
            const short* p0 = (const short*)A + ii * N + b;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[batch];
                pp[2] = p0[N];
                pp[3] = p0[N + batch];
                p0 += batch * 2;
                pp += 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[N];
                p0 += batch;
                pp += 2;
            }
        }
        for (; ii < max_ii; ii++)
        {
            const short* p0 = (const short*)A + ii * N + b;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                p0 += batch;
                pp += 1;
            }
        }
    }
}

static void transpose_pack_B_tile_int8(const Mat& B, Mat& BT, int batch, int max_jj, int max_kk, int nT)
{
    #pragma omp parallel for num_threads(nT)
    for (int b = 0; b < batch; b++)
    {
        short* pp = BT.row<short>(b);

        int jj = 0;
#if __loongarch_sx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const short* p0 = B;

            int kk = 0;
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = __lsx_vld(p0, 0);
                __m128i _r1 = __lsx_vld((p0 + 8), 0);
                __m128i _r2 = __lsx_vld((p0 + 8 * 2), 0);
                __m128i _r3 = __lsx_vld((p0 + 8 * 3), 0);
                __m128i _r4 = __lsx_vld((p0 + 8 * 4), 0);
                __m128i _r5 = __lsx_vld((p0 + 8 * 5), 0);
                __m128i _r6 = __lsx_vld((p0 + 8 * 6), 0);
                __m128i _r7 = __lsx_vld((p0 + 8 * 7), 0);
                transpose4x8_epi32(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                __lsx_vst(_r0, pp, 0);
                __lsx_vst(_r1, (pp + 8), 0);
                __lsx_vst(_r2, (pp + 8 * 2), 0);
                __lsx_vst(_r3, (pp + 8 * 3), 0);
                __lsx_vst(_r4, (pp + 8 * 4), 0);
                __lsx_vst(_r5, (pp + 8 * 5), 0);
                __lsx_vst(_r6, (pp + 8 * 6), 0);
                __lsx_vst(_r7, (pp + 8 * 7), 0);
                p0 += max_jj * batch * 8;
                pp += 64;
            }
            p0 -= (b * max_jj + jj) * 8;
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _r0 = __lsx_vld(p0, 0);
                __m128i _r1 = __lsx_vld((p0 + 8), 0);
                __lsx_vst(_r0, pp, 0);
                __lsx_vst(_r1, (pp + 8), 0);
                p0 += max_jj * batch * 2;
                pp += 16;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                __m128i _r0 = __lsx_vld(p0, 0);
                __lsx_vst(_r0, pp, 0);
                p0 += max_jj * batch;
                pp += 8;
            }
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const short* p0 = B;

            int kk = 0;
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = __lsx_vld(p0, 0);
                __m128i _r1 = __lsx_vld((p0 + 8), 0);
                __m128i _r2 = __lsx_vld((p0 + 8 * 2), 0);
                __m128i _r3 = __lsx_vld((p0 + 8 * 3), 0);
                transpose4x4_epi32(_r0, _r1, _r2, _r3);
                __lsx_vst(_r0, pp, 0);
                __lsx_vst(_r1, (pp + 8), 0);
                __lsx_vst(_r2, (pp + 8 * 2), 0);
                __lsx_vst(_r3, (pp + 8 * 3), 0);
                p0 += max_jj * batch * 8;
                pp += 32;
            }
            p0 -= (b * max_jj + jj) * 8;
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _r0 = __lsx_vld(p0, 0);
                __lsx_vst(_r0, pp, 0);
                p0 += max_jj * batch * 2;
                pp += 8;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                p0 += max_jj * batch;
                pp += 4;
            }
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            const short* p0 = B;

            int kk = 0;
#if __loongarch_sx
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = __lsx_vld(p0, 0);
                __m128i _r1 = __lsx_vld((p0 + 8), 0);
                __m128i _tmp0 = __lsx_vilvl_w(_r1, _r0);
                __m128i _tmp1 = __lsx_vilvh_w(_r1, _r0);
                __lsx_vst(_tmp0, pp, 0);
                __lsx_vst(_tmp1, (pp + 8), 0);
                p0 += max_jj * batch * 8;
                pp += 16;
            }
            p0 -= (b * max_jj + jj) * 8;
#endif // __loongarch_sx
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                p0 += max_jj * batch * 2;
                pp += 4;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                p0 += max_jj * batch;
                pp += 2;
            }
        }
        for (; jj < max_jj; jj++)
        {
            const short* p0 = B;

            int kk = 0;
#if __loongarch_sx
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = __lsx_vld(p0, 0);
                __lsx_vst(_r0, pp, 0);
                p0 += max_jj * batch * 8;
                pp += 8;
            }
            p0 -= (b * max_jj + jj) * 8;
#endif // __loongarch_sx
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                p0 += max_jj * batch * 2;
                pp += 2;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                p0 += max_jj * batch;
                pp += 1;
            }
        }
    }
}

static void gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& top_blob, int batch, int max_ii, int max_jj, int k, int max_kk, bool k_end)
{
    int* outptr = top_blob;

    int ii = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        for (int b = 0; b < batch; b++)
        {
            const short* pAT = AT_tile.row<const short>(b) + max_kk * ii;
            const short* pB = BT_tile.row<const short>(b);

            int jj = 0;
            for (; jj + 3 < max_jj; jj += 4)
            {
                const short* pA = pAT;

                __m256i _sum0;
                __m256i _sum1;
                __m256i _sum2;
                __m256i _sum3;

                if (k == 0)
                {
                    _sum0 = __lasx_xvldi(0);
                    _sum1 = __lasx_xvldi(0);
                    _sum2 = __lasx_xvldi(0);
                    _sum3 = __lasx_xvldi(0);
                }
                else
                {
                    _sum0 = __lasx_xvld(outptr, 0);
                    _sum1 = __lasx_xvld(outptr + 8, 0);
                    _sum2 = __lasx_xvld(outptr + 16, 0);
                    _sum3 = __lasx_xvld(outptr + 24, 0);
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m256i _pA0 = __lasx_xvld(pA, 0);
                    __m128i _pB = __lsx_vld(pB, 0);
                    __m256i _pA1 = __lasx_xvshuf4i_w(_pA0, 78);
                    __m256i _pB0 = __lasx_xvpermi_q(__lsx_to_lasx(_pB), __lsx_to_lasx(_pB), 0x00);
                    __m256i _pB1 = __lasx_xvshuf4i_w(_pB0, 57);

                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvadd_w(__lasx_xvmulwev_w_h(_pA0, _pB0), __lasx_xvmulwod_w_h(_pA0, _pB0)));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvadd_w(__lasx_xvmulwev_w_h(_pA0, _pB1), __lasx_xvmulwod_w_h(_pA0, _pB1)));
                    _sum2 = __lasx_xvadd_w(_sum2, __lasx_xvadd_w(__lasx_xvmulwev_w_h(_pA1, _pB0), __lasx_xvmulwod_w_h(_pA1, _pB0)));
                    _sum3 = __lasx_xvadd_w(_sum3, __lasx_xvadd_w(__lasx_xvmulwev_w_h(_pA1, _pB1), __lasx_xvmulwod_w_h(_pA1, _pB1)));

                    pA += 16;
                    pB += 8;
                }
                for (; kk < max_kk; kk++)
                {
                    __m256i _pA0 = __lasx_xvsext_w_h(__lsx_to_lasx(__lsx_vld(pA, 0)));
                    __m256i _pB0 = __lasx_xvsext_w_h(__lsx_to_lasx(__lsx_vldrepl_d(pB, 0)));
                    __m256i _pA1 = __lasx_xvpermi_d(_pA0, 78);
                    __m256i _pB1 = __lasx_xvshuf4i_w(_pB0, 57);

                    __m256i _s0 = __lasx_xvmul_w(_pA0, _pB0);
                    __m256i _s1 = __lasx_xvmul_w(_pA0, _pB1);
                    __m256i _s2 = __lasx_xvmul_w(_pA1, _pB0);
                    __m256i _s3 = __lasx_xvmul_w(_pA1, _pB1);
                    _sum0 = __lasx_xvadd_w(_sum0, _s0);
                    _sum1 = __lasx_xvadd_w(_sum1, _s1);
                    _sum2 = __lasx_xvadd_w(_sum2, _s2);
                    _sum3 = __lasx_xvadd_w(_sum3, _s3);

                    pA += 8;
                    pB += 4;
                }

                if (k_end)
                {
                    // from
                    //      00 11 22 33 40 51 62 73
                    //      01 12 23 30 41 52 63 70
                    //      20 31 02 13 60 71 42 53
                    //      21 32 03 10 61 72 43 50
                    // to
                    //      00 10 20 30 40 50 60 70
                    //      01 11 21 31 41 51 61 71
                    //      02 12 22 32 42 52 62 72
                    //      03 13 23 33 43 53 63 73
                    {
                        _sum1 = __lasx_xvshuf4i_w(_sum1, 147);
                        _sum3 = __lasx_xvshuf4i_w(_sum3, 147);
                        __m256i _tmp0 = __lasx_xvilvl_w(_sum3, _sum0);
                        __m256i _tmp1 = __lasx_xvilvh_w(_sum3, _sum0);
                        __m256i _tmp2 = __lasx_xvilvl_w(_sum1, _sum2);
                        __m256i _tmp3 = __lasx_xvilvh_w(_sum1, _sum2);
                        _sum0 = __lasx_xvilvl_d(_tmp2, _tmp0);
                        _sum1 = __lasx_xvilvh_d(_tmp2, _tmp0);
                        _sum2 = __lasx_xvilvl_d(_tmp1, _tmp3);
                        _sum3 = __lasx_xvilvh_d(_tmp1, _tmp3);
                        _sum1 = __lasx_xvshuf4i_w(_sum1, 147);
                        _sum3 = __lasx_xvshuf4i_w(_sum3, 147);
                    }
                }

                __lasx_xvst(_sum0, outptr, 0);
                __lasx_xvst(_sum1, outptr + 8, 0);
                __lasx_xvst(_sum2, outptr + 16, 0);
                __lasx_xvst(_sum3, outptr + 24, 0);
                outptr += 32;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const short* pA = pAT;

                __m256i _sum0;
                __m256i _sum1;

                if (k == 0)
                {
                    _sum0 = __lasx_xvldi(0);
                    _sum1 = __lasx_xvldi(0);
                }
                else
                {
                    _sum0 = __lasx_xvld(outptr, 0);
                    _sum1 = __lasx_xvld(outptr + 8, 0);
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m256i _pA = __lasx_xvld(pA, 0);
                    __m256i _pB0 = __lasx_xvldrepl_d(pB, 0);
                    __m256i _pB1 = __lasx_xvshuf4i_w(_pB0, 177);

                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvadd_w(__lasx_xvmulwev_w_h(_pA, _pB0), __lasx_xvmulwod_w_h(_pA, _pB0)));
                    _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvadd_w(__lasx_xvmulwev_w_h(_pA, _pB1), __lasx_xvmulwod_w_h(_pA, _pB1)));

                    pA += 16;
                    pB += 4;
                }
                for (; kk < max_kk; kk++)
                {
                    __m256i _pA = __lasx_xvsext_w_h(__lsx_to_lasx(__lsx_vld(pA, 0)));
                    __m256i _pB0 = __lasx_xvsext_w_h(__lsx_to_lasx(__lsx_vldrepl_w(pB, 0)));
                    __m256i _pB1 = __lasx_xvshuf4i_w(_pB0, 177);

                    __m256i _s0 = __lasx_xvmul_w(_pA, _pB0);
                    __m256i _s1 = __lasx_xvmul_w(_pA, _pB1);
                    _sum0 = __lasx_xvadd_w(_sum0, _s0);
                    _sum1 = __lasx_xvadd_w(_sum1, _s1);

                    pA += 8;
                    pB += 2;
                }

                if (k_end)
                {
                    // from
                    //      00 11 20 31 40 51 60 71
                    //      01 10 21 30 41 50 61 70
                    // to
                    //      00 10 20 30 40 50 60 70
                    //      01 11 21 31 41 51 61 71
                    {
                        __m256i _tmp0 = __lasx_xvshuf4i_w(_sum0, 216);
                        __m256i _tmp1 = __lasx_xvshuf4i_w(_sum1, 45);
                        _sum0 = __lasx_xvilvl_w(_tmp1, _tmp0);
                        _sum1 = __lasx_xvilvh_w(_tmp1, _tmp0);
                        _sum1 = __lasx_xvshuf4i_w(_sum1, 147);
                    }
                }

                __lasx_xvst(_sum0, outptr, 0);
                __lasx_xvst(_sum1, outptr + 8, 0);
                outptr += 16;
            }
            for (; jj < max_jj; jj++)
            {
                const short* pA = pAT;

                __m256i _sum0;

                if (k == 0)
                {
                    _sum0 = __lasx_xvldi(0);
                }
                else
                {
                    _sum0 = __lasx_xvld(outptr, 0);
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m256i _pA = __lasx_xvld(pA, 0);
                    __m256i _pB = __lasx_xvldrepl_w(pB, 0);
                    _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvadd_w(__lasx_xvmulwev_w_h(_pA, _pB), __lasx_xvmulwod_w_h(_pA, _pB)));

                    pA += 16;
                    pB += 2;
                }
                for (; kk < max_kk; kk++)
                {
                    __m256i _pA = __lasx_xvsext_w_h(__lsx_to_lasx(__lsx_vld(pA, 0)));
                    __m256i _pB = __lasx_xvreplgr2vr_w(pB[0]);
                    __m256i _s0 = __lasx_xvmul_w(_pA, _pB);
                    _sum0 = __lasx_xvadd_w(_sum0, _s0);

                    pA += 8;
                    pB += 1;
                }

                __lasx_xvst(_sum0, outptr, 0);
                outptr += 8;
            }
        }
    }
#endif // __loongarch_asx
    for (; ii + 3 < max_ii; ii += 4)
    {
        for (int b = 0; b < batch; b++)
        {
            const short* pAT = AT_tile.row<const short>(b) + max_kk * ii;
            const short* pB = BT_tile.row<const short>(b);

            int jj = 0;
            for (; jj + 7 < max_jj; jj += 8)
            {
                const short* pA = pAT;

                __m128i _sum0;
                __m128i _sum1;
                __m128i _sum2;
                __m128i _sum3;
                __m128i _sum4;
                __m128i _sum5;
                __m128i _sum6;
                __m128i _sum7;

                if (k == 0)
                {
                    _sum0 = __lsx_vldi(0);
                    _sum1 = __lsx_vldi(0);
                    _sum2 = __lsx_vldi(0);
                    _sum3 = __lsx_vldi(0);
                    _sum4 = __lsx_vldi(0);
                    _sum5 = __lsx_vldi(0);
                    _sum6 = __lsx_vldi(0);
                    _sum7 = __lsx_vldi(0);
                }
                else
                {
                    _sum0 = __lsx_vld(outptr, 0);
                    _sum1 = __lsx_vld((outptr + 4), 0);
                    _sum2 = __lsx_vld((outptr + 8), 0);
                    _sum3 = __lsx_vld((outptr + 12), 0);
                    _sum4 = __lsx_vld((outptr + 16), 0);
                    _sum5 = __lsx_vld((outptr + 20), 0);
                    _sum6 = __lsx_vld((outptr + 24), 0);
                    _sum7 = __lsx_vld((outptr + 28), 0);
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m128i _pA0 = __lsx_vld(pA, 0);
                    __m128i _pB0 = __lsx_vld(pB, 0);
                    __m128i _pB1 = __lsx_vld((pB + 8), 0);
                    __m128i _pA1 = __lsx_vshuf4i_w(_pA0, 78);
                    __m128i _pB2 = __lsx_vshuf4i_w(_pB0, 57);
                    __m128i _pB3 = __lsx_vshuf4i_w(_pB1, 57);

                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA0, _pB0), __lsx_vmulwod_w_h(_pA0, _pB0)));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA0, _pB1), __lsx_vmulwod_w_h(_pA0, _pB1)));
                    _sum2 = __lsx_vadd_w(_sum2, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA0, _pB2), __lsx_vmulwod_w_h(_pA0, _pB2)));
                    _sum3 = __lsx_vadd_w(_sum3, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA0, _pB3), __lsx_vmulwod_w_h(_pA0, _pB3)));
                    _sum4 = __lsx_vadd_w(_sum4, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA1, _pB0), __lsx_vmulwod_w_h(_pA1, _pB0)));
                    _sum5 = __lsx_vadd_w(_sum5, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA1, _pB1), __lsx_vmulwod_w_h(_pA1, _pB1)));
                    _sum6 = __lsx_vadd_w(_sum6, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA1, _pB2), __lsx_vmulwod_w_h(_pA1, _pB2)));
                    _sum7 = __lsx_vadd_w(_sum7, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA1, _pB3), __lsx_vmulwod_w_h(_pA1, _pB3)));

                    pA += 8;
                    pB += 16;
                }
                for (; kk < max_kk; kk++)
                {
                    __m128i _pA = __lsx_vldrepl_d(pA, 0);
                    __m128i _pB = __lsx_vld(pB, 0);

                    __m128i _pA0 = _pA;
                    __m128i _pA1 = __lsx_vshuf4i_w(_pA, 177);
                    __m128i _pB01 = _pB;
                    __m128i _pB23 = __lsx_vshuf4i_h(_pB, 57);
                    __m128i _sl0 = __lsx_vmul_h(_pA0, _pB01);
                    __m128i _sh0 = __lsx_vmuh_h(_pA0, _pB01);
                    __m128i _sl1 = __lsx_vmul_h(_pA0, _pB23);
                    __m128i _sh1 = __lsx_vmuh_h(_pA0, _pB23);
                    __m128i _sl2 = __lsx_vmul_h(_pA1, _pB01);
                    __m128i _sh2 = __lsx_vmuh_h(_pA1, _pB01);
                    __m128i _sl3 = __lsx_vmul_h(_pA1, _pB23);
                    __m128i _sh3 = __lsx_vmuh_h(_pA1, _pB23);
                    __m128i _s0 = __lsx_vilvl_h(_sh0, _sl0);
                    __m128i _s1 = __lsx_vilvh_h(_sh0, _sl0);
                    __m128i _s2 = __lsx_vilvl_h(_sh1, _sl1);
                    __m128i _s3 = __lsx_vilvh_h(_sh1, _sl1);
                    __m128i _s4 = __lsx_vilvl_h(_sh2, _sl2);
                    __m128i _s5 = __lsx_vilvh_h(_sh2, _sl2);
                    __m128i _s6 = __lsx_vilvl_h(_sh3, _sl3);
                    __m128i _s7 = __lsx_vilvh_h(_sh3, _sl3);
                    _sum0 = __lsx_vadd_w(_sum0, _s0);
                    _sum1 = __lsx_vadd_w(_sum1, _s1);
                    _sum2 = __lsx_vadd_w(_sum2, _s2);
                    _sum3 = __lsx_vadd_w(_sum3, _s3);
                    _sum4 = __lsx_vadd_w(_sum4, _s4);
                    _sum5 = __lsx_vadd_w(_sum5, _s5);
                    _sum6 = __lsx_vadd_w(_sum6, _s6);
                    _sum7 = __lsx_vadd_w(_sum7, _s7);

                    pA += 4;
                    pB += 8;
                }

                if (k_end)
                {
                    // from
                    //      00 11 22 33  04 15 26 37
                    //      01 12 23 30  05 16 27 34
                    //      20 31 02 13  24 35 06 17
                    //      21 32 03 10  25 36 07 14
                    // to
                    //      00 10 20 30  04 14 24 34
                    //      01 11 21 31  05 15 25 35
                    //      02 12 22 32  06 16 26 36
                    //      03 13 23 33  07 17 27 37
                    {
                        _sum2 = __lsx_vshuf4i_w(_sum2, 147);
                        _sum3 = __lsx_vshuf4i_w(_sum3, 147);
                        _sum6 = __lsx_vshuf4i_w(_sum6, 147);
                        _sum7 = __lsx_vshuf4i_w(_sum7, 147);
                        __m128i _tmp0 = __lsx_vilvl_w(_sum6, _sum0);
                        __m128i _tmp1 = __lsx_vilvh_w(_sum6, _sum0);
                        __m128i _tmp2 = __lsx_vilvl_w(_sum7, _sum1);
                        __m128i _tmp3 = __lsx_vilvh_w(_sum7, _sum1);
                        __m128i _tmp4 = __lsx_vilvl_w(_sum2, _sum4);
                        __m128i _tmp5 = __lsx_vilvh_w(_sum2, _sum4);
                        __m128i _tmp6 = __lsx_vilvl_w(_sum3, _sum5);
                        __m128i _tmp7 = __lsx_vilvh_w(_sum3, _sum5);
                        _sum0 = __lsx_vilvl_d(_tmp4, _tmp0);
                        _sum1 = __lsx_vilvh_d(_tmp4, _tmp0);
                        _sum2 = __lsx_vilvl_d(_tmp1, _tmp5);
                        _sum3 = __lsx_vilvh_d(_tmp1, _tmp5);
                        _sum4 = __lsx_vilvl_d(_tmp6, _tmp2);
                        _sum5 = __lsx_vilvh_d(_tmp6, _tmp2);
                        _sum6 = __lsx_vilvl_d(_tmp3, _tmp7);
                        _sum7 = __lsx_vilvh_d(_tmp3, _tmp7);
                        _sum1 = __lsx_vshuf4i_w(_sum1, 147);
                        _sum3 = __lsx_vshuf4i_w(_sum3, 147);
                        _sum5 = __lsx_vshuf4i_w(_sum5, 147);
                        _sum7 = __lsx_vshuf4i_w(_sum7, 147);
                    }
                }

                __lsx_vst(_sum0, outptr, 0);
                __lsx_vst(_sum1, (outptr + 4), 0);
                __lsx_vst(_sum2, (outptr + 8), 0);
                __lsx_vst(_sum3, (outptr + 12), 0);
                __lsx_vst(_sum4, (outptr + 16), 0);
                __lsx_vst(_sum5, (outptr + 20), 0);
                __lsx_vst(_sum6, (outptr + 24), 0);
                __lsx_vst(_sum7, (outptr + 28), 0);
                outptr += 32;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const short* pA = pAT;

                __m128i _sum0;
                __m128i _sum1;
                __m128i _sum2;
                __m128i _sum3;

                if (k == 0)
                {
                    _sum0 = __lsx_vldi(0);
                    _sum1 = __lsx_vldi(0);
                    _sum2 = __lsx_vldi(0);
                    _sum3 = __lsx_vldi(0);
                }
                else
                {
                    _sum0 = __lsx_vld(outptr, 0);
                    _sum1 = __lsx_vld((outptr + 4), 0);
                    _sum2 = __lsx_vld((outptr + 8), 0);
                    _sum3 = __lsx_vld((outptr + 12), 0);
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m128i _pA0 = __lsx_vld(pA, 0);
                    __m128i _pB0 = __lsx_vld(pB, 0);
                    __m128i _pA1 = __lsx_vshuf4i_w(_pA0, 78);
                    __m128i _pB1 = __lsx_vshuf4i_w(_pB0, 57);

                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA0, _pB0), __lsx_vmulwod_w_h(_pA0, _pB0)));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA0, _pB1), __lsx_vmulwod_w_h(_pA0, _pB1)));
                    _sum2 = __lsx_vadd_w(_sum2, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA1, _pB0), __lsx_vmulwod_w_h(_pA1, _pB0)));
                    _sum3 = __lsx_vadd_w(_sum3, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA1, _pB1), __lsx_vmulwod_w_h(_pA1, _pB1)));

                    pA += 8;
                    pB += 8;
                }
                for (; kk < max_kk; kk++)
                {
                    __m128i _pA = __lsx_vldrepl_d(pA, 0);
                    __m128i _pB = __lsx_vldrepl_d(pB, 0);
                    __m128i _pA0 = _pA;
                    __m128i _pA1 = __lsx_vshuf4i_w(_pA, 177);
                    __m128i _pB01 = __lsx_vilvl_d(__lsx_vshuf4i_h(_pB, 57), _pB);
                    __m128i _sl0 = __lsx_vmul_h(_pA0, _pB01);
                    __m128i _sh0 = __lsx_vmuh_h(_pA0, _pB01);
                    __m128i _sl1 = __lsx_vmul_h(_pA1, _pB01);
                    __m128i _sh1 = __lsx_vmuh_h(_pA1, _pB01);
                    __m128i _s0 = __lsx_vilvl_h(_sh0, _sl0);
                    __m128i _s1 = __lsx_vilvh_h(_sh0, _sl0);
                    __m128i _s2 = __lsx_vilvl_h(_sh1, _sl1);
                    __m128i _s3 = __lsx_vilvh_h(_sh1, _sl1);
                    _sum0 = __lsx_vadd_w(_sum0, _s0);
                    _sum1 = __lsx_vadd_w(_sum1, _s1);
                    _sum2 = __lsx_vadd_w(_sum2, _s2);
                    _sum3 = __lsx_vadd_w(_sum3, _s3);

                    pA += 4;
                    pB += 4;
                }

                if (k_end)
                {
                    // from
                    //      00 11 22 33
                    //      01 12 23 30
                    //      20 31 02 13
                    //      21 32 03 10
                    // to
                    //      00 10 20 30
                    //      01 11 21 31
                    //      02 12 22 32
                    //      03 13 23 33
                    {
                        _sum1 = __lsx_vshuf4i_w(_sum1, 147);
                        _sum3 = __lsx_vshuf4i_w(_sum3, 147);
                        __m128i _tmp0 = __lsx_vilvl_w(_sum3, _sum0);
                        __m128i _tmp1 = __lsx_vilvh_w(_sum3, _sum0);
                        __m128i _tmp2 = __lsx_vilvl_w(_sum1, _sum2);
                        __m128i _tmp3 = __lsx_vilvh_w(_sum1, _sum2);
                        _sum0 = __lsx_vilvl_d(_tmp2, _tmp0);
                        _sum1 = __lsx_vilvh_d(_tmp2, _tmp0);
                        _sum2 = __lsx_vilvl_d(_tmp1, _tmp3);
                        _sum3 = __lsx_vilvh_d(_tmp1, _tmp3);
                        _sum1 = __lsx_vshuf4i_w(_sum1, 147);
                        _sum3 = __lsx_vshuf4i_w(_sum3, 147);
                    }
                }

                __lsx_vst(_sum0, outptr, 0);
                __lsx_vst(_sum1, (outptr + 4), 0);
                __lsx_vst(_sum2, (outptr + 8), 0);
                __lsx_vst(_sum3, (outptr + 12), 0);
                outptr += 16;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const short* pA = pAT;

                __m128i _sum0;
                __m128i _sum1;

                if (k == 0)
                {
                    _sum0 = __lsx_vldi(0);
                    _sum1 = __lsx_vldi(0);
                }
                else
                {
                    _sum0 = __lsx_vld(outptr, 0);
                    _sum1 = __lsx_vld((outptr + 4), 0);
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m128i _pA = __lsx_vld(pA, 0);
                    __m128i _pB0 = __lsx_vldrepl_d(pB, 0);
                    __m128i _pB1 = __lsx_vshuf4i_w(_pB0, 177);

                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA, _pB0), __lsx_vmulwod_w_h(_pA, _pB0)));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA, _pB1), __lsx_vmulwod_w_h(_pA, _pB1)));

                    pA += 8;
                    pB += 4;
                }
                for (; kk < max_kk; kk++)
                {
                    __m128i _pA = __lsx_vldrepl_d(pA, 0);
                    __m128i _pB = __lsx_vreplgr2vr_w(*(int*)(pB));

                    __m128i _pB01 = __lsx_vilvl_d(__lsx_vshuf4i_h(_pB, 17), _pB);
                    __m128i _sl = __lsx_vmul_h(_pA, _pB01);
                    __m128i _sh = __lsx_vmuh_h(_pA, _pB01);
                    __m128i _s0 = __lsx_vilvl_h(_sh, _sl);
                    __m128i _s1 = __lsx_vilvh_h(_sh, _sl);
                    _sum0 = __lsx_vadd_w(_sum0, _s0);
                    _sum1 = __lsx_vadd_w(_sum1, _s1);

                    pA += 4;
                    pB += 2;
                }

                if (k_end)
                {
                    // from
                    //      00 11 20 31
                    //      01 10 21 30
                    // to
                    //      00 10 20 30
                    //      01 11 21 31
                    {
                        __m128i _tmp0 = __lsx_vshuf4i_w(_sum0, 216);
                        __m128i _tmp1 = __lsx_vshuf4i_w(_sum1, 45);
                        _sum0 = __lsx_vilvl_w(_tmp1, _tmp0);
                        _sum1 = __lsx_vilvh_w(_tmp1, _tmp0);
                        _sum1 = __lsx_vshuf4i_w(_sum1, 147);
                    }
                }

                __lsx_vst(_sum0, outptr, 0);
                __lsx_vst(_sum1, (outptr + 4), 0);
                outptr += 8;
            }
            for (; jj < max_jj; jj++)
            {
                const short* pA = pAT;

                __m128i _sum0;

                if (k == 0)
                {
                    _sum0 = __lsx_vldi(0);
                }
                else
                {
                    _sum0 = __lsx_vld(outptr, 0);
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m128i _pA = __lsx_vld(pA, 0);
                    __m128i _pB = __lsx_vreplgr2vr_w(*(int*)(pB));

                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA, _pB), __lsx_vmulwod_w_h(_pA, _pB)));

                    pA += 8;
                    pB += 2;
                }
                for (; kk < max_kk; kk++)
                {
                    __m128i _pA = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)pA, 0);
                    __m128i _pB = __lsx_vreplgr2vr_h(pB[0]);

                    __m128i _sl = __lsx_vmul_h(_pA, _pB);
                    __m128i _sh = __lsx_vmuh_h(_pA, _pB);
                    __m128i _s0 = __lsx_vilvl_h(_sh, _sl);
                    _sum0 = __lsx_vadd_w(_sum0, _s0);

                    pA += 4;
                    pB += 1;
                }

                __lsx_vst(_sum0, outptr, 0);
                outptr += 4;
            }
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        for (int b = 0; b < batch; b++)
        {
            const short* pAT = AT_tile.row<const short>(b) + max_kk * ii;
            const short* pB = BT_tile.row<const short>(b);

            int jj = 0;
#if __loongarch_sx
            for (; jj + 7 < max_jj; jj += 8)
            {
                const short* pA = pAT;

                __m128i _sum0;
                __m128i _sum1;
                __m128i _sum2;
                __m128i _sum3;

                if (k == 0)
                {
                    _sum0 = __lsx_vldi(0);
                    _sum1 = __lsx_vldi(0);
                    _sum2 = __lsx_vldi(0);
                    _sum3 = __lsx_vldi(0);
                }
                else
                {
                    _sum0 = __lsx_vld(outptr, 0);
                    _sum1 = __lsx_vld((outptr + 4), 0);
                    _sum2 = __lsx_vld((outptr + 8), 0);
                    _sum3 = __lsx_vld((outptr + 12), 0);
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m128i _pA0 = __lsx_vreplgr2vr_w(*(int*)(pA));
                    __m128i _pA1 = __lsx_vreplgr2vr_w(*(int*)((pA + 2)));
                    __m128i _pB0 = __lsx_vld(pB, 0);
                    __m128i _pB1 = __lsx_vld((pB + 8), 0);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA0, _pB0), __lsx_vmulwod_w_h(_pA0, _pB0)));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA0, _pB1), __lsx_vmulwod_w_h(_pA0, _pB1)));
                    _sum2 = __lsx_vadd_w(_sum2, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA1, _pB0), __lsx_vmulwod_w_h(_pA1, _pB0)));
                    _sum3 = __lsx_vadd_w(_sum3, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA1, _pB1), __lsx_vmulwod_w_h(_pA1, _pB1)));

                    pA += 4;
                    pB += 16;
                }
                for (; kk < max_kk; kk++)
                {
                    __m128i _pB = __lsx_vld(pB, 0);
                    __m128i _pA0 = __lsx_vreplgr2vr_h(pA[0]);
                    __m128i _pA1 = __lsx_vreplgr2vr_h(pA[1]);

                    __m128i _sl0 = __lsx_vmul_h(_pA0, _pB);
                    __m128i _sh0 = __lsx_vmuh_h(_pA0, _pB);
                    __m128i _sl1 = __lsx_vmul_h(_pA1, _pB);
                    __m128i _sh1 = __lsx_vmuh_h(_pA1, _pB);
                    __m128i _s0 = __lsx_vilvl_h(_sh0, _sl0);
                    __m128i _s1 = __lsx_vilvh_h(_sh0, _sl0);
                    __m128i _s2 = __lsx_vilvl_h(_sh1, _sl1);
                    __m128i _s3 = __lsx_vilvh_h(_sh1, _sl1);
                    _sum0 = __lsx_vadd_w(_sum0, _s0);
                    _sum1 = __lsx_vadd_w(_sum1, _s1);
                    _sum2 = __lsx_vadd_w(_sum2, _s2);
                    _sum3 = __lsx_vadd_w(_sum3, _s3);
                    pA += 2;
                    pB += 8;
                }

                if (k_end)
                {
                    __m128i _tmp0 = __lsx_vilvl_w(_sum2, _sum0);
                    __m128i _tmp1 = __lsx_vilvh_w(_sum2, _sum0);
                    __m128i _tmp2 = __lsx_vilvl_w(_sum3, _sum1);
                    __m128i _tmp3 = __lsx_vilvh_w(_sum3, _sum1);
                    _sum0 = _tmp0;
                    _sum1 = _tmp1;
                    _sum2 = _tmp2;
                    _sum3 = _tmp3;
                }

                __lsx_vst(_sum0, outptr, 0);
                __lsx_vst(_sum1, (outptr + 4), 0);
                __lsx_vst(_sum2, (outptr + 8), 0);
                __lsx_vst(_sum3, (outptr + 12), 0);
                outptr += 16;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const short* pA = pAT;

                __m128i _sum0;
                __m128i _sum1;

                if (k == 0)
                {
                    _sum0 = __lsx_vldi(0);
                    _sum1 = __lsx_vldi(0);
                }
                else
                {
                    _sum0 = __lsx_vld(outptr, 0);
                    _sum1 = __lsx_vld((outptr + 4), 0);
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m128i _pA0 = __lsx_vreplgr2vr_w(*(int*)(pA));
                    __m128i _pA1 = __lsx_vreplgr2vr_w(*(int*)((pA + 2)));
                    __m128i _pB = __lsx_vld(pB, 0);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA0, _pB), __lsx_vmulwod_w_h(_pA0, _pB)));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA1, _pB), __lsx_vmulwod_w_h(_pA1, _pB)));

                    pA += 4;
                    pB += 8;
                }
                for (; kk < max_kk; kk++)
                {
                    __m128i _pA0 = __lsx_vreplgr2vr_h(pA[0]);
                    __m128i _pA1 = __lsx_vreplgr2vr_h(pA[1]);
                    __m128i _pB = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)pB, 0);
                    __m128i _sl0 = __lsx_vmul_h(_pA0, _pB);
                    __m128i _sh0 = __lsx_vmuh_h(_pA0, _pB);
                    __m128i _sl1 = __lsx_vmul_h(_pA1, _pB);
                    __m128i _sh1 = __lsx_vmuh_h(_pA1, _pB);
                    __m128i _s0 = __lsx_vilvl_h(_sh0, _sl0);
                    __m128i _s1 = __lsx_vilvl_h(_sh1, _sl1);
                    _sum0 = __lsx_vadd_w(_sum0, _s0);
                    _sum1 = __lsx_vadd_w(_sum1, _s1);
                    pA += 2;
                    pB += 4;
                }

                if (k_end)
                {
                    __m128i _tmp0 = __lsx_vilvl_w(_sum1, _sum0);
                    __m128i _tmp1 = __lsx_vilvh_w(_sum1, _sum0);
                    _sum0 = _tmp0;
                    _sum1 = _tmp1;
                }

                __lsx_vst(_sum0, outptr, 0);
                __lsx_vst(_sum1, (outptr + 4), 0);
                outptr += 2 * 4;
            }
#endif // __loongarch_sx
            for (; jj + 1 < max_jj; jj += 2)
            {
                const short* pA = pAT;

                int sum00 = 0;
                int sum01 = 0;
                int sum10 = 0;
                int sum11 = 0;

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

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    sum00 += pA[0] * pB[0];
                    sum00 += pA[1] * pB[1];
                    sum01 += pA[2] * pB[0];
                    sum01 += pA[3] * pB[1];
                    sum10 += pA[0] * pB[2];
                    sum10 += pA[1] * pB[3];
                    sum11 += pA[2] * pB[2];
                    sum11 += pA[3] * pB[3];

                    pA += 4;
                    pB += 4;
                }
                for (; kk < max_kk; kk++)
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
                outptr += 2 * 2;
            }
            for (; jj < max_jj; jj++)
            {
                const short* pA = pAT;

                int sum0 = 0;
                int sum1 = 0;

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

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    sum0 += pA[0] * pB[0];
                    sum0 += pA[1] * pB[1];
                    sum1 += pA[2] * pB[0];
                    sum1 += pA[3] * pB[1];
                    pA += 4;
                    pB += 2;
                }
                for (; kk < max_kk; kk++)
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
        }
    }
    for (; ii < max_ii; ii++)
    {
        for (int b = 0; b < batch; b++)
        {
            const short* pAT = AT_tile.row<const short>(b) + max_kk * ii;
            const short* pB = BT_tile.row<const short>(b);

            int jj = 0;
#if __loongarch_sx
            for (; jj + 7 < max_jj; jj += 8)
            {
                const short* pA = pAT;

                __m128i _sum0;
                __m128i _sum1;

                if (k == 0)
                {
                    _sum0 = __lsx_vldi(0);
                    _sum1 = __lsx_vldi(0);
                }
                else
                {
                    _sum0 = __lsx_vld(outptr, 0);
                    _sum1 = __lsx_vld((outptr + 4), 0);
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m128i _pA = __lsx_vreplgr2vr_w(*(int*)(pA));
                    __m128i _pB0 = __lsx_vld(pB, 0);
                    __m128i _pB1 = __lsx_vld((pB + 8), 0);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA, _pB0), __lsx_vmulwod_w_h(_pA, _pB0)));
                    _sum1 = __lsx_vadd_w(_sum1, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA, _pB1), __lsx_vmulwod_w_h(_pA, _pB1)));

                    pA += 2;
                    pB += 16;
                }
                for (; kk < max_kk; kk++)
                {
                    __m128i _pA = __lsx_vreplgr2vr_h(pA[0]);
                    __m128i _pB = __lsx_vld(pB, 0);
                    __m128i _sl = __lsx_vmul_h(_pA, _pB);
                    __m128i _sh = __lsx_vmuh_h(_pA, _pB);
                    __m128i _s0 = __lsx_vilvl_h(_sh, _sl);
                    __m128i _s1 = __lsx_vilvh_h(_sh, _sl);
                    _sum0 = __lsx_vadd_w(_sum0, _s0);
                    _sum1 = __lsx_vadd_w(_sum1, _s1);
                    pA += 1;
                    pB += 8;
                }

                __lsx_vst(_sum0, outptr, 0);
                __lsx_vst(_sum1, (outptr + 4), 0);
                outptr += 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const short* pA = pAT;

                __m128i _sum0;

                if (k == 0)
                {
                    _sum0 = __lsx_vldi(0);
                }
                else
                {
                    _sum0 = __lsx_vld(outptr, 0);
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m128i _pA = __lsx_vreplgr2vr_w(*(int*)(pA));
                    __m128i _pB = __lsx_vld(pB, 0);
                    _sum0 = __lsx_vadd_w(_sum0, __lsx_vadd_w(__lsx_vmulwev_w_h(_pA, _pB), __lsx_vmulwod_w_h(_pA, _pB)));

                    pA += 2;
                    pB += 8;
                }
                for (; kk < max_kk; kk++)
                {
                    __m128i _pA = __lsx_vreplgr2vr_h(pA[0]);
                    __m128i _pB = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)pB, 0);
                    __m128i _sl = __lsx_vmul_h(_pA, _pB);
                    __m128i _sh = __lsx_vmuh_h(_pA, _pB);
                    __m128i _s0 = __lsx_vilvl_h(_sh, _sl);
                    _sum0 = __lsx_vadd_w(_sum0, _s0);
                    pA += 1;
                    pB += 4;
                }

                __lsx_vst(_sum0, outptr, 0);
                outptr += 4;
            }
#endif // __loongarch_sx
            for (; jj + 1 < max_jj; jj += 2)
            {
                const short* pA = pAT;

                int sum0 = 0;
                int sum1 = 0;

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

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    sum0 += pA[0] * pB[0];
                    sum0 += pA[1] * pB[1];
                    sum1 += pA[0] * pB[2];
                    sum1 += pA[1] * pB[3];
                    pA += 2;
                    pB += 4;
                }
                for (; kk < max_kk; kk++)
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
            for (; jj < max_jj; jj++)
            {
                const short* pA = pAT;

                int sum = 0;

                if (k == 0)
                {
                    sum = 0;
                }
                else
                {
                    sum = outptr[0];
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    sum += pA[0] * pB[0];
                    pA += 1;
                    pB += 1;
                }

                outptr[0] = sum;
                outptr += 1;
            }
        }
    }
}

static void get_optimal_tile_mnk_int8(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size_int8 = (int)(get_cpu_level2_cache_size() / sizeof(short));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve M
    {
        int tile_size = (int)sqrt((float)l2_cache_size_int8 / 3);

#if __loongarch_asx
        TILE_M = std::max(8, tile_size / 8 * 8);
#elif __loongarch_sx
        TILE_M = std::max(4, tile_size / 4 * 4);
#else
        TILE_M = std::max(2, tile_size / 2 * 2);
#endif

        TILE_M *= std::min(nT, get_physical_cpu_count());

        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __loongarch_asx
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#elif __loongarch_sx
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif

        if (nT > 1)
        {
#if __loongarch_asx
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#elif __loongarch_sx
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 3) / 4 * 4);
#else
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
        }
    }

    // solve K
    {
        int tile_size = (int)(sqrt((float)l2_cache_size_int8) - TILE_M);

#if __loongarch_asx
        TILE_K = std::max(8, tile_size / 8 * 8);
#elif __loongarch_sx
        TILE_K = std::max(4, tile_size / 4 * 4);
#else
        TILE_K = std::max(2, tile_size / 2 * 2);
#endif

        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __loongarch_asx
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);
#elif __loongarch_sx
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif
    }

    if (N > 0)
    {
        int tile_size = (int)((l2_cache_size_int8 - TILE_M * TILE_K) / (TILE_M * 2 + TILE_K));

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
    }
}

static inline void conv3x3s1_winograd23_transform_kernel_tile_int8(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    // const signed char ktm[4][3] = {
    //     {2, 0, 0},
    //     {1, 1, 1},
    //     {1, -1, 1},
    //     {0, 0, 2}
    // };

    short* ptmp = A;

    int ii = 0;
    for (; ii < max_ii; ii++)
    {
        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            short tmp[4][3];

            const signed char* k0 = (const signed char*)kernel + (i + ii) * inch * 9 + (k + kk) * 9;

            for (int m = 0; m < 3; m++)
            {
                signed char r0 = k0[0];
                signed char r1 = k0[1];
                signed char r2 = k0[2];

                tmp[0][m] = r0 * 2;
                tmp[1][m] = r0 + r1 + r2;
                tmp[2][m] = r0 - r1 + r2;
                tmp[3][m] = r2 * 2;

                k0 += 3;
            }

            for (int m = 0; m < 4; m++)
            {
                short r0 = tmp[m][0];
                short r1 = tmp[m][1];
                short r2 = tmp[m][2];

                short z0 = r0 * 2;
                short z1 = r0 + r1 + r2;
                short z2 = r0 - r1 + r2;
                short z3 = r2 * 2;

                ptmp[0] = z0;
                ptmp[1] = z1;
                ptmp[2] = z2;
                ptmp[3] = z3;
                ptmp += 4;
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_kernel_int8(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 16;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat A_tileX(B * TILE_M * TILE_K, 1, opt.num_threads, 2u, (Allocator*)0);

    AT.create(TILE_K * TILE_M, B, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, 2u, (Allocator*)0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat A_tile = A_tileX.channel(get_omp_thread_num());

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            conv3x3s1_winograd23_transform_kernel_tile_int8(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            pack_A_tile_int8(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd23_transform_input_tile_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    // const signed char itm[4][4] = {
    //     {1,  0, -1,  0},
    //     {0,  1,  1,  0},
    //     {0, -1,  1,  0},
    //     {0, -1,  0,  1}
    // };

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const size_t N = bottom_blob.cstep * elempack;

    const int w_tiles = (w - 1) / 2;

    int nn_max_kk = 0;
    int remain_max_kk_start = 0;
#if __loongarch_sx
#if __loongarch_asx
    nn_max_kk = (max_kk - remain_max_kk_start) / 16;
    #pragma omp parallel for num_threads(nT)
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 16;

        __attribute__((aligned(32))) short tmp[4][4][16];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0 = bottom_blob.channel((k + kk) / elempack).row<const signed char>(ti * 2) + (tj * 2) * elempack;

            for (int m = 0; m < 4; m++)
            {
                __m256i _r0 = __lasx_xvldi(0);
                __m256i _r1 = __lasx_xvldi(0);
                __m256i _r2 = __lasx_xvldi(0);
                __m256i _r3 = __lasx_xvldi(0);

                if (ti * 2 + m < h)
                {
                    if (elempack == 8)
                    {
                        const signed char* r1 = r0 + N;

                        __m128i _t0 = __lsx_vld(r0, 0);
                        __m128i _t1 = __lsx_vld(r1, 0);
                        __m128i _t01 = __lsx_vilvl_d(_t1, _t0);
                        _r0 = __lasx_xvsext_h_b(__lsx_to_lasx(_t01));
                        if (tj * 2 + 1 < w)
                        {
                            __m128i _t0 = __lsx_vld(r0 + 8, 0);
                            __m128i _t1 = __lsx_vld(r1 + 8, 0);
                            __m128i _t01 = __lsx_vilvl_d(_t1, _t0);
                            _r1 = __lasx_xvsext_h_b(__lsx_to_lasx(_t01));
                        }
                        if (tj * 2 + 2 < w)
                        {
                            __m128i _t0 = __lsx_vld(r0 + 16, 0);
                            __m128i _t1 = __lsx_vld(r1 + 16, 0);
                            __m128i _t01 = __lsx_vilvl_d(_t1, _t0);
                            _r2 = __lasx_xvsext_h_b(__lsx_to_lasx(_t01));
                        }
                        if (tj * 2 + 3 < w)
                        {
                            __m128i _t0 = __lsx_vld(r0 + 24, 0);
                            __m128i _t1 = __lsx_vld(r1 + 24, 0);
                            __m128i _t01 = __lsx_vilvl_d(_t1, _t0);
                            _r3 = __lasx_xvsext_h_b(__lsx_to_lasx(_t01));
                        }
                    }
                    if (elempack == 1)
                    {
                        const signed char* r1 = r0 + N;
                        const signed char* r2 = r0 + N * 2;
                        const signed char* r3 = r0 + N * 3;
                        const signed char* r4 = r0 + N * 4;
                        const signed char* r5 = r0 + N * 5;
                        const signed char* r6 = r0 + N * 6;
                        const signed char* r7 = r0 + N * 7;
                        const signed char* r8 = r0 + N * 8;
                        const signed char* r9 = r0 + N * 9;
                        const signed char* ra = r0 + N * 10;
                        const signed char* rb = r0 + N * 11;
                        const signed char* rc = r0 + N * 12;
                        const signed char* rd = r0 + N * 13;
                        const signed char* re = r0 + N * 14;
                        const signed char* rf = r0 + N * 15;

                        {
                            __m128i _t0 = __lsx_vld(r0 + 0, 0);
                            __m128i _t1 = __lsx_vld(r1 + 0, 0);
                            __m128i _t2 = __lsx_vld(r2 + 0, 0);
                            __m128i _t3 = __lsx_vld(r3 + 0, 0);
                            __m128i _t4 = __lsx_vld(r4 + 0, 0);
                            __m128i _t5 = __lsx_vld(r5 + 0, 0);
                            __m128i _t6 = __lsx_vld(r6 + 0, 0);
                            __m128i _t7 = __lsx_vld(r7 + 0, 0);
                            __m128i _t8 = __lsx_vld(r8 + 0, 0);
                            __m128i _t9 = __lsx_vld(r9 + 0, 0);
                            __m128i _ta = __lsx_vld(ra + 0, 0);
                            __m128i _tb = __lsx_vld(rb + 0, 0);
                            __m128i _tc = __lsx_vld(rc + 0, 0);
                            __m128i _td = __lsx_vld(rd + 0, 0);
                            __m128i _te = __lsx_vld(re + 0, 0);
                            __m128i _tf = __lsx_vld(rf + 0, 0);

                            __m128i _t01 = __lsx_vilvl_b(_t1, _t0);
                            __m128i _t23 = __lsx_vilvl_b(_t3, _t2);
                            __m128i _t45 = __lsx_vilvl_b(_t5, _t4);
                            __m128i _t67 = __lsx_vilvl_b(_t7, _t6);
                            __m128i _t89 = __lsx_vilvl_b(_t9, _t8);
                            __m128i _tab = __lsx_vilvl_b(_tb, _ta);
                            __m128i _tcd = __lsx_vilvl_b(_td, _tc);
                            __m128i _tef = __lsx_vilvl_b(_tf, _te);
                            __m128i _u0 = __lsx_vilvl_h(_t23, _t01);
                            __m128i _u1 = __lsx_vilvl_h(_t67, _t45);
                            __m128i _u2 = __lsx_vilvl_h(_tab, _t89);
                            __m128i _u3 = __lsx_vilvl_h(_tef, _tcd);
                            __m128i _v0 = __lsx_vilvl_w(_u1, _u0);
                            __m128i _v1 = __lsx_vilvl_w(_u3, _u2);
                            __m128i _val = __lsx_vilvl_d(_v1, _v0);
                            _r0 = __lasx_xvsext_h_b(__lsx_to_lasx(_val));
                        }
                        if (tj * 2 + 1 < w)
                        {
                            __m128i _t0 = __lsx_vld(r0 + 1, 0);
                            __m128i _t1 = __lsx_vld(r1 + 1, 0);
                            __m128i _t2 = __lsx_vld(r2 + 1, 0);
                            __m128i _t3 = __lsx_vld(r3 + 1, 0);
                            __m128i _t4 = __lsx_vld(r4 + 1, 0);
                            __m128i _t5 = __lsx_vld(r5 + 1, 0);
                            __m128i _t6 = __lsx_vld(r6 + 1, 0);
                            __m128i _t7 = __lsx_vld(r7 + 1, 0);
                            __m128i _t8 = __lsx_vld(r8 + 1, 0);
                            __m128i _t9 = __lsx_vld(r9 + 1, 0);
                            __m128i _ta = __lsx_vld(ra + 1, 0);
                            __m128i _tb = __lsx_vld(rb + 1, 0);
                            __m128i _tc = __lsx_vld(rc + 1, 0);
                            __m128i _td = __lsx_vld(rd + 1, 0);
                            __m128i _te = __lsx_vld(re + 1, 0);
                            __m128i _tf = __lsx_vld(rf + 1, 0);

                            __m128i _t01 = __lsx_vilvl_b(_t1, _t0);
                            __m128i _t23 = __lsx_vilvl_b(_t3, _t2);
                            __m128i _t45 = __lsx_vilvl_b(_t5, _t4);
                            __m128i _t67 = __lsx_vilvl_b(_t7, _t6);
                            __m128i _t89 = __lsx_vilvl_b(_t9, _t8);
                            __m128i _tab = __lsx_vilvl_b(_tb, _ta);
                            __m128i _tcd = __lsx_vilvl_b(_td, _tc);
                            __m128i _tef = __lsx_vilvl_b(_tf, _te);
                            __m128i _u0 = __lsx_vilvl_h(_t23, _t01);
                            __m128i _u1 = __lsx_vilvl_h(_t67, _t45);
                            __m128i _u2 = __lsx_vilvl_h(_tab, _t89);
                            __m128i _u3 = __lsx_vilvl_h(_tef, _tcd);
                            __m128i _v0 = __lsx_vilvl_w(_u1, _u0);
                            __m128i _v1 = __lsx_vilvl_w(_u3, _u2);
                            __m128i _val = __lsx_vilvl_d(_v1, _v0);
                            _r1 = __lasx_xvsext_h_b(__lsx_to_lasx(_val));
                        }
                        if (tj * 2 + 2 < w)
                        {
                            __m128i _t0 = __lsx_vld(r0 + 2, 0);
                            __m128i _t1 = __lsx_vld(r1 + 2, 0);
                            __m128i _t2 = __lsx_vld(r2 + 2, 0);
                            __m128i _t3 = __lsx_vld(r3 + 2, 0);
                            __m128i _t4 = __lsx_vld(r4 + 2, 0);
                            __m128i _t5 = __lsx_vld(r5 + 2, 0);
                            __m128i _t6 = __lsx_vld(r6 + 2, 0);
                            __m128i _t7 = __lsx_vld(r7 + 2, 0);
                            __m128i _t8 = __lsx_vld(r8 + 2, 0);
                            __m128i _t9 = __lsx_vld(r9 + 2, 0);
                            __m128i _ta = __lsx_vld(ra + 2, 0);
                            __m128i _tb = __lsx_vld(rb + 2, 0);
                            __m128i _tc = __lsx_vld(rc + 2, 0);
                            __m128i _td = __lsx_vld(rd + 2, 0);
                            __m128i _te = __lsx_vld(re + 2, 0);
                            __m128i _tf = __lsx_vld(rf + 2, 0);

                            __m128i _t01 = __lsx_vilvl_b(_t1, _t0);
                            __m128i _t23 = __lsx_vilvl_b(_t3, _t2);
                            __m128i _t45 = __lsx_vilvl_b(_t5, _t4);
                            __m128i _t67 = __lsx_vilvl_b(_t7, _t6);
                            __m128i _t89 = __lsx_vilvl_b(_t9, _t8);
                            __m128i _tab = __lsx_vilvl_b(_tb, _ta);
                            __m128i _tcd = __lsx_vilvl_b(_td, _tc);
                            __m128i _tef = __lsx_vilvl_b(_tf, _te);
                            __m128i _u0 = __lsx_vilvl_h(_t23, _t01);
                            __m128i _u1 = __lsx_vilvl_h(_t67, _t45);
                            __m128i _u2 = __lsx_vilvl_h(_tab, _t89);
                            __m128i _u3 = __lsx_vilvl_h(_tef, _tcd);
                            __m128i _v0 = __lsx_vilvl_w(_u1, _u0);
                            __m128i _v1 = __lsx_vilvl_w(_u3, _u2);
                            __m128i _val = __lsx_vilvl_d(_v1, _v0);
                            _r2 = __lasx_xvsext_h_b(__lsx_to_lasx(_val));
                        }
                        if (tj * 2 + 3 < w)
                        {
                            __m128i _t0 = __lsx_vld(r0 + 3, 0);
                            __m128i _t1 = __lsx_vld(r1 + 3, 0);
                            __m128i _t2 = __lsx_vld(r2 + 3, 0);
                            __m128i _t3 = __lsx_vld(r3 + 3, 0);
                            __m128i _t4 = __lsx_vld(r4 + 3, 0);
                            __m128i _t5 = __lsx_vld(r5 + 3, 0);
                            __m128i _t6 = __lsx_vld(r6 + 3, 0);
                            __m128i _t7 = __lsx_vld(r7 + 3, 0);
                            __m128i _t8 = __lsx_vld(r8 + 3, 0);
                            __m128i _t9 = __lsx_vld(r9 + 3, 0);
                            __m128i _ta = __lsx_vld(ra + 3, 0);
                            __m128i _tb = __lsx_vld(rb + 3, 0);
                            __m128i _tc = __lsx_vld(rc + 3, 0);
                            __m128i _td = __lsx_vld(rd + 3, 0);
                            __m128i _te = __lsx_vld(re + 3, 0);
                            __m128i _tf = __lsx_vld(rf + 3, 0);

                            __m128i _t01 = __lsx_vilvl_b(_t1, _t0);
                            __m128i _t23 = __lsx_vilvl_b(_t3, _t2);
                            __m128i _t45 = __lsx_vilvl_b(_t5, _t4);
                            __m128i _t67 = __lsx_vilvl_b(_t7, _t6);
                            __m128i _t89 = __lsx_vilvl_b(_t9, _t8);
                            __m128i _tab = __lsx_vilvl_b(_tb, _ta);
                            __m128i _tcd = __lsx_vilvl_b(_td, _tc);
                            __m128i _tef = __lsx_vilvl_b(_tf, _te);
                            __m128i _u0 = __lsx_vilvl_h(_t23, _t01);
                            __m128i _u1 = __lsx_vilvl_h(_t67, _t45);
                            __m128i _u2 = __lsx_vilvl_h(_tab, _t89);
                            __m128i _u3 = __lsx_vilvl_h(_tef, _tcd);
                            __m128i _v0 = __lsx_vilvl_w(_u1, _u0);
                            __m128i _v1 = __lsx_vilvl_w(_u3, _u2);
                            __m128i _val = __lsx_vilvl_d(_v1, _v0);
                            _r3 = __lasx_xvsext_h_b(__lsx_to_lasx(_val));
                        }
                    }
                }

                __m256i _tmp0 = __lasx_xvsub_h(_r0, _r2);
                __m256i _tmp1 = __lasx_xvadd_h(_r1, _r2);
                __m256i _tmp2 = __lasx_xvsub_h(_r2, _r1);
                __m256i _tmp3 = __lasx_xvsub_h(_r3, _r1);

                __lasx_xvst(_tmp0, (short*)tmp[0][m], 0);
                __lasx_xvst(_tmp1, (short*)tmp[1][m], 0);
                __lasx_xvst(_tmp2, (short*)tmp[2][m], 0);
                __lasx_xvst(_tmp3, (short*)tmp[3][m], 0);

                r0 += w * elempack;
            }

            short* p0 = (short*)B + kk * max_jj * 16 + jj * 16;
            short* p1 = p0 + max_jj * 16 * 1;
            short* p2 = p0 + max_jj * 16 * 2;
            short* p3 = p0 + max_jj * 16 * 3;

            for (int m = 0; m < 4; m++)
            {
                __m256i _r0 = __lasx_xvld((const __m256i*)tmp[m][0], 0);
                __m256i _r1 = __lasx_xvld((const __m256i*)tmp[m][1], 0);
                __m256i _r2 = __lasx_xvld((const __m256i*)tmp[m][2], 0);
                __m256i _r3 = __lasx_xvld((const __m256i*)tmp[m][3], 0);

                __m256i _tmp0 = __lasx_xvsub_h(_r0, _r2);
                __m256i _tmp1 = __lasx_xvadd_h(_r1, _r2);
                __m256i _tmp2 = __lasx_xvsub_h(_r2, _r1);
                __m256i _tmp3 = __lasx_xvsub_h(_r3, _r1);

                __lasx_xvst(_tmp0, p0, 0);
                __lasx_xvst(_tmp1, p1, 0);
                __lasx_xvst(_tmp2, p2, 0);
                __lasx_xvst(_tmp3, p3, 0);

                p0 += max_jj * 4 * 16;
                p1 += max_jj * 4 * 16;
                p2 += max_jj * 4 * 16;
                p3 += max_jj * 4 * 16;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 16;
    nn_max_kk = (max_kk - remain_max_kk_start) / 8;
#else // __loongarch_asx
    nn_max_kk = (max_kk - remain_max_kk_start) / 8;
    #pragma omp parallel for num_threads(nT)
#endif // __loongarch_asx
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 8;

        __attribute__((aligned(16))) short tmp[4][4][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0 = bottom_blob.channel((k + kk) / elempack).row<const signed char>(ti * 2) + (tj * 2) * elempack;

            for (int m = 0; m < 4; m++)
            {
                __m128i _r0 = __lsx_vldi(0);
                __m128i _r1 = __lsx_vldi(0);
                __m128i _r2 = __lsx_vldi(0);
                __m128i _r3 = __lsx_vldi(0);

                if (ti * 2 + m < h)
                {
                    if (elempack == 8)
                    {
                        _r0 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)r0, 0);
                        _r0 = __lsx_vsllwil_h_b(_r0, 0);
                        if (tj * 2 + 1 < w)
                        {
                            _r1 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)(r0 + 8), 0);
                            _r1 = __lsx_vsllwil_h_b(_r1, 0);
                        }
                        if (tj * 2 + 2 < w)
                        {
                            _r2 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)(r0 + 16), 0);
                            _r2 = __lsx_vsllwil_h_b(_r2, 0);
                        }
                        if (tj * 2 + 3 < w)
                        {
                            _r3 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)(r0 + 24), 0);
                            _r3 = __lsx_vsllwil_h_b(_r3, 0);
                        }
                    }
                    if (elempack == 1)
                    {
                        const signed char* r1 = r0 + N;
                        const signed char* r2 = r0 + N * 2;
                        const signed char* r3 = r0 + N * 3;
                        const signed char* r4 = r0 + N * 4;
                        const signed char* r5 = r0 + N * 5;
                        const signed char* r6 = r0 + N * 6;
                        const signed char* r7 = r0 + N * 7;

                        __m128i _t0 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)r0, 0);
                        __m128i _t1 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)r1, 0);
                        __m128i _t2 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)r2, 0);
                        __m128i _t3 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)r3, 0);
                        __m128i _t4 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)r4, 0);
                        __m128i _t5 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)r5, 0);
                        __m128i _t6 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)r6, 0);
                        __m128i _t7 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)r7, 0);

                        __m128i _t01 = __lsx_vilvl_b(_t1, _t0);
                        __m128i _t23 = __lsx_vilvl_b(_t3, _t2);
                        __m128i _t45 = __lsx_vilvl_b(_t5, _t4);
                        __m128i _t67 = __lsx_vilvl_b(_t7, _t6);
                        _t0 = __lsx_vilvl_h(_t23, _t01);
                        _t1 = __lsx_vilvl_h(_t67, _t45);
                        _t2 = __lsx_vilvl_w(_t1, _t0);
                        _t3 = __lsx_vilvh_w(_t1, _t0);

                        __m128i _extt2 = __lsx_vsrai_b(_t2, 7);
                        __m128i _extt3 = __lsx_vsrai_b(_t3, 7);

                        _r0 = __lsx_vilvl_b(_extt2, _t2);
                        if (tj * 2 + 1 < w) _r1 = __lsx_vilvh_b(_extt2, _t2);
                        if (tj * 2 + 2 < w) _r2 = __lsx_vilvl_b(_extt3, _t3);
                        if (tj * 2 + 3 < w) _r3 = __lsx_vilvh_b(_extt3, _t3);
                    }
                }

                __m128i _tmp0 = __lsx_vsub_h(_r0, _r2);
                __m128i _tmp1 = __lsx_vadd_h(_r1, _r2);
                __m128i _tmp2 = __lsx_vsub_h(_r2, _r1);
                __m128i _tmp3 = __lsx_vsub_h(_r3, _r1);

                __lsx_vst(_tmp0, tmp[0][m], 0);
                __lsx_vst(_tmp1, tmp[1][m], 0);
                __lsx_vst(_tmp2, tmp[2][m], 0);
                __lsx_vst(_tmp3, tmp[3][m], 0);

                r0 += w * elempack;
            }

            short* p0 = (short*)B + kk * max_jj * 16 + jj * 8;
            short* p1 = p0 + max_jj * 8;
            short* p2 = p0 + max_jj * 8 * 2;
            short* p3 = p0 + max_jj * 8 * 3;

            for (int m = 0; m < 4; m++)
            {
                __m128i _r0 = __lsx_vld(tmp[m][0], 0);
                __m128i _r1 = __lsx_vld(tmp[m][1], 0);
                __m128i _r2 = __lsx_vld(tmp[m][2], 0);
                __m128i _r3 = __lsx_vld(tmp[m][3], 0);

                __m128i _tmp0 = __lsx_vsub_h(_r0, _r2);
                __m128i _tmp1 = __lsx_vadd_h(_r1, _r2);
                __m128i _tmp2 = __lsx_vsub_h(_r2, _r1);
                __m128i _tmp3 = __lsx_vsub_h(_r3, _r1);

                __lsx_vst(_tmp0, p0, 0);
                __lsx_vst(_tmp1, p1, 0);
                __lsx_vst(_tmp2, p2, 0);
                __lsx_vst(_tmp3, p3, 0);

                p0 += max_jj * 4 * 8;
                p1 += max_jj * 4 * 8;
                p2 += max_jj * 4 * 8;
                p3 += max_jj * 4 * 8;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 8;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
#else
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    #pragma omp parallel for num_threads(nT)
#endif // __loongarch_sx
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

        short tmp[4][4][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0 = bottom_blob.channel(k + kk).row<const signed char>(ti * 2) + (tj * 2);

            for (int m = 0; m < 4; m++)
            {
                signed char r00 = 0;
                signed char r01 = 0;
                signed char r10 = 0;
                signed char r11 = 0;
                signed char r20 = 0;
                signed char r21 = 0;
                signed char r30 = 0;
                signed char r31 = 0;

                if (ti * 2 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const signed char* r1 = r0 + N;

                        r00 = r0[0];
                        r01 = r1[0];
                        if (tj * 2 + 1 < w)
                        {
                            r10 = r0[1];
                            r11 = r1[1];
                        }
                        if (tj * 2 + 2 < w)
                        {
                            r20 = r0[2];
                            r21 = r1[2];
                        }
                        if (tj * 2 + 3 < w)
                        {
                            r30 = r0[3];
                            r31 = r1[3];
                        }
                    }
                }

                tmp[0][m][0] = r00 - r20;
                tmp[0][m][1] = r01 - r21;
                tmp[1][m][0] = r10 + r20;
                tmp[1][m][1] = r11 + r21;
                tmp[2][m][0] = r20 - r10;
                tmp[2][m][1] = r21 - r11;
                tmp[3][m][0] = r30 - r10;
                tmp[3][m][1] = r31 - r11;

                r0 += w;
            }

            short* p0 = (short*)B + kk * max_jj * 16 + jj * 2;
            short* p1 = p0 + max_jj * 2;
            short* p2 = p0 + max_jj * 2 * 2;
            short* p3 = p0 + max_jj * 2 * 3;

            for (int m = 0; m < 4; m++)
            {
                short r00 = tmp[m][0][0];
                short r01 = tmp[m][0][1];
                short r10 = tmp[m][1][0];
                short r11 = tmp[m][1][1];
                short r20 = tmp[m][2][0];
                short r21 = tmp[m][2][1];
                short r30 = tmp[m][3][0];
                short r31 = tmp[m][3][1];

                p0[0] = r00 - r20;
                p0[1] = r01 - r21;
                p1[0] = r10 + r20;
                p1[1] = r11 + r21;
                p2[0] = r20 - r10;
                p2[1] = r21 - r11;
                p3[0] = r30 - r10;
                p3[1] = r31 - r11;

                p0 += max_jj * 4 * 2;
                p1 += max_jj * 4 * 2;
                p2 += max_jj * 4 * 2;
                p3 += max_jj * 4 * 2;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 2;
    for (int kk = remain_max_kk_start; kk < max_kk; kk++)
    {
        short tmp[4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0123 = bottom_blob.channel(k + kk).row<const signed char>(ti * 2) + (tj * 2);

            for (int m = 0; m < 4; m++)
            {
                signed char r0 = 0;
                signed char r1 = 0;
                signed char r2 = 0;
                signed char r3 = 0;

                if (ti * 2 + m < h)
                {
                    // if (elempack == 1)
                    {
                        r0 = r0123[0];
                        if (tj * 2 + 1 < w) r1 = r0123[1];
                        if (tj * 2 + 2 < w) r2 = r0123[2];
                        if (tj * 2 + 3 < w) r3 = r0123[3];
                    }
                }

                tmp[0][m] = r0 - r2;
                tmp[1][m] = r1 + r2;
                tmp[2][m] = r2 - r1;
                tmp[3][m] = r3 - r1;

                r0123 += w;
            }

            short* p0 = (short*)B + kk * max_jj * 16 + jj;
            short* p1 = p0 + max_jj;
            short* p2 = p0 + max_jj * 2;
            short* p3 = p0 + max_jj * 3;

            for (int m = 0; m < 4; m++)
            {
                short r0 = tmp[m][0];
                short r1 = tmp[m][1];
                short r2 = tmp[m][2];
                short r3 = tmp[m][3];

                p0[0] = r0 - r2;
                p1[0] = r1 + r2;
                p2[0] = r2 - r1;
                p3[0] = r3 - r1;

                p0 += max_jj * 4;
                p1 += max_jj * 4;
                p2 += max_jj * 4;
                p3 += max_jj * 4;
            }
        }
    }
}

static inline void conv3x3s1_winograd23_transform_output_tile_int8(const Mat& top_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    // const int otm[2][4] = {
    //     {1,  1,  1,  0},
    //     {0,  1, -1,  1}
    // };

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const size_t N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 1) / 2;

    int ii = 0;
#if __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        __attribute__((aligned(32))) int tmp[2][4][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 16 + jj * 8;
            const int* r1 = r0 + max_jj * 8 * 1;
            const int* r2 = r0 + max_jj * 8 * 2;
            const int* r3 = r0 + max_jj * 8 * 3;

            for (int m = 0; m < 4; m++)
            {
                __m256i _r0 = __lasx_xvld(r0, 0);
                __m256i _r1 = __lasx_xvld(r1, 0);
                __m256i _r2 = __lasx_xvld(r2, 0);
                __m256i _r3 = __lasx_xvld(r3, 0);

                __m256i _tmp0 = __lasx_xvadd_w(__lasx_xvadd_w(_r0, _r1), _r2);
                __m256i _tmp1 = __lasx_xvadd_w(__lasx_xvsub_w(_r1, _r2), _r3);

                __lasx_xvst(_tmp0, (int*)tmp[0][m], 0);
                __lasx_xvst(_tmp1, (int*)tmp[1][m], 0);

                r0 += max_jj * 4 * 8;
                r1 += max_jj * 4 * 8;
                r2 += max_jj * 4 * 8;
                r3 += max_jj * 4 * 8;
            }

            int* outptr0 = top_blob.channel((i + ii) / out_elempack).row<int>(ti * 2) + (tj * 2) * out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                __m256i _r0 = __lasx_xvld((const __m256i*)tmp[m][0], 0);
                __m256i _r1 = __lasx_xvld((const __m256i*)tmp[m][1], 0);
                __m256i _r2 = __lasx_xvld((const __m256i*)tmp[m][2], 0);
                __m256i _r3 = __lasx_xvld((const __m256i*)tmp[m][3], 0);

                __m256i _tmp0 = __lasx_xvadd_w(__lasx_xvadd_w(_r0, _r1), _r2);
                __m256i _tmp1 = __lasx_xvadd_w(__lasx_xvsub_w(_r1, _r2), _r3);

                _tmp0 = __lasx_xvsrai_w(_tmp0, 2);
                _tmp1 = __lasx_xvsrai_w(_tmp1, 2);

                if (out_elempack == 8)
                {
                    __lasx_xvst(_tmp0, outptr0, 0);
                    if (tj * 2 + 1 < outw) __lasx_xvst(_tmp1, outptr0 + 8, 0);
                    if (tj * 2 + 1 < outw)
                    {
                        __lasx_xvst(_tmp1, outptr0 + 8, 0);
                    }
                }
                if (out_elempack == 4)
                {
                    int* outptr1 = outptr0 + N;

                    __lsx_vst(__lasx_extract_lo128(_tmp0), outptr0, 0);
                    __lsx_vst((__m128i)__lasx_xvpermi_q(_tmp0, _tmp0, 0x11), outptr1, 0);
                    if (tj * 2 + 1 < outw)
                    {
                        __lsx_vst(__lasx_extract_lo128(_tmp1), outptr0 + 4, 0);
                        __lsx_vst((__m128i)__lasx_xvpermi_q(_tmp1, _tmp1, 0x11), outptr1 + 4, 0);
                    }
                }
                if (out_elempack == 1)
                {
                    int tmp0[8];
                    int tmp1[8];
                    __lasx_xvst(_tmp0, tmp0, 0);
                    __lasx_xvst(_tmp1, tmp1, 0);

                    int* outptr1 = outptr0 + N * 1;
                    int* outptr2 = outptr0 + N * 2;
                    int* outptr3 = outptr0 + N * 3;
                    int* outptr4 = outptr0 + N * 4;
                    int* outptr5 = outptr0 + N * 5;
                    int* outptr6 = outptr0 + N * 6;
                    int* outptr7 = outptr0 + N * 7;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    outptr4[0] = tmp0[4];
                    outptr5[0] = tmp0[5];
                    outptr6[0] = tmp0[6];
                    outptr7[0] = tmp0[7];
                    if (tj * 2 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                        outptr4[1] = tmp1[4];
                        outptr5[1] = tmp1[5];
                        outptr6[1] = tmp1[6];
                        outptr7[1] = tmp1[7];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __loongarch_asx
#if __loongarch_sx
    for (; ii + 3 < max_ii; ii += 4)
    {
        __attribute__((aligned(16))) int tmp[2][4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 16 + jj * 4;
            const int* r1 = r0 + max_jj * 4;
            const int* r2 = r0 + max_jj * 4 * 2;
            const int* r3 = r0 + max_jj * 4 * 3;

            for (int m = 0; m < 4; m++)
            {
                __m128i _r0 = __lsx_vld(r0, 0);
                __m128i _r1 = __lsx_vld(r1, 0);
                __m128i _r2 = __lsx_vld(r2, 0);
                __m128i _r3 = __lsx_vld(r3, 0);

                __m128i _tmp0 = __lsx_vadd_w(__lsx_vadd_w(_r0, _r1), _r2);
                __m128i _tmp1 = __lsx_vadd_w(__lsx_vsub_w(_r1, _r2), _r3);

                __lsx_vst(_tmp0, tmp[0][m], 0);
                __lsx_vst(_tmp1, tmp[1][m], 0);

                r0 += max_jj * 4 * 4;
                r1 += max_jj * 4 * 4;
                r2 += max_jj * 4 * 4;
                r3 += max_jj * 4 * 4;
            }

            int* outptr0 = top_blob.channel((i + ii) / out_elempack).row<int>(ti * 2) + (tj * 2) * out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                __m128i _r0 = __lsx_vld(tmp[m][0], 0);
                __m128i _r1 = __lsx_vld(tmp[m][1], 0);
                __m128i _r2 = __lsx_vld(tmp[m][2], 0);
                __m128i _r3 = __lsx_vld(tmp[m][3], 0);

                __m128i _tmp0 = __lsx_vadd_w(__lsx_vadd_w(_r0, _r1), _r2);
                __m128i _tmp1 = __lsx_vadd_w(__lsx_vsub_w(_r1, _r2), _r3);

                _tmp0 = __lsx_vsrai_w(_tmp0, 2);
                _tmp1 = __lsx_vsrai_w(_tmp1, 2);

                if (out_elempack == 4)
                {
                    __lsx_vst(_tmp0, outptr0, 0);
                    if (tj * 2 + 1 < outw) __lsx_vst(_tmp1, (outptr0 + 4), 0);
                }
                if (out_elempack == 1)
                {
                    int tmp0[4];
                    int tmp1[4];
                    __lsx_vst(_tmp0, tmp0, 0);
                    __lsx_vst(_tmp1, tmp1, 0);

                    int* outptr1 = outptr0 + N;
                    int* outptr2 = outptr0 + N * 2;
                    int* outptr3 = outptr0 + N * 3;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];

                    if (tj * 2 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        int tmp[2][4][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 16 + jj * 2;
            const int* r1 = r0 + max_jj * 2;
            const int* r2 = r0 + max_jj * 2 * 2;
            const int* r3 = r0 + max_jj * 2 * 3;

            for (int m = 0; m < 4; m++)
            {
                tmp[0][m][0] = r0[0] + r1[0] + r2[0];
                tmp[0][m][1] = r0[1] + r1[1] + r2[1];
                tmp[1][m][0] = r1[0] - r2[0] + r3[0];
                tmp[1][m][1] = r1[1] - r2[1] + r3[1];

                r0 += max_jj * 4 * 2;
                r1 += max_jj * 4 * 2;
                r2 += max_jj * 4 * 2;
                r3 += max_jj * 4 * 2;
            }

            int* outptr0 = top_blob.channel(i + ii).row<int>(ti * 2) + (tj * 2);

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                int tmp00 = tmp[m][0][0] + tmp[m][1][0] + tmp[m][2][0];
                int tmp01 = tmp[m][0][1] + tmp[m][1][1] + tmp[m][2][1];
                int tmp10 = tmp[m][1][0] - tmp[m][2][0] + tmp[m][3][0];
                int tmp11 = tmp[m][1][1] - tmp[m][2][1] + tmp[m][3][1];

                tmp00 = tmp00 >> 2;
                tmp01 = tmp01 >> 2;
                tmp10 = tmp10 >> 2;
                tmp11 = tmp11 >> 2;

                // if (out_elempack == 1)
                {
                    int* outptr1 = outptr0 + N;

                    outptr0[0] = tmp00;
                    outptr1[0] = tmp01;
                    if (tj * 2 + 1 < outw)
                    {
                        outptr0[1] = tmp10;
                        outptr1[1] = tmp11;
                    }
                }

                outptr0 += outw;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        int tmp[2][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 16 + jj;
            const int* r1 = r0 + max_jj;
            const int* r2 = r0 + max_jj * 2;
            const int* r3 = r0 + max_jj * 3;

            for (int m = 0; m < 4; m++)
            {
                tmp[0][m] = r0[0] + r1[0] + r2[0];
                tmp[1][m] = r1[0] - r2[0] + r3[0];

                r0 += max_jj * 4;
                r1 += max_jj * 4;
                r2 += max_jj * 4;
                r3 += max_jj * 4;
            }

            int* outptr0 = top_blob.channel(i + ii).row<int>(ti * 2) + (tj * 2);

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                int tmp0 = tmp[m][0] + tmp[m][1] + tmp[m][2];
                int tmp1 = tmp[m][1] - tmp[m][2] + tmp[m][3];

                tmp0 = tmp0 >> 2;
                tmp1 = tmp1 >> 2;

                // if (out_elempack == 1)
                {
                    outptr0[0] = tmp0;
                    if (tj * 2 + 1 < outw) outptr0[1] = tmp1;
                }

                outptr0 += outw;
            }
        }
    }
}

static int conv3x3s1_winograd23_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt)
{
    int outw = top_blob.w;
    int outh = top_blob.h;

    // pad to 2n+2, winograd F(2,3)
    int w_tiles = (outw + 1) / 2;
    int h_tiles = (outh + 1) / 2;
    int tiles = w_tiles * h_tiles;

    const int M = top_blob.c * top_blob.elempack;
    const int N = tiles;
    const int K = bottom_blob.c * bottom_blob.elempack;
    const int B = 16;

    // NCNN_LOGE("conv3x3s1_winograd23_int8 %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;

    if (nT > 1 && nn_NK < nT)
    {
        Mat B_tile(TILE_N * B * TILE_K, 2u, opt.workspace_allocator);
        if (B_tile.empty())
            return -100;

        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            // transform input
            conv3x3s1_winograd23_transform_input_tile_int8(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile_int8(B_tile, BT_tile, B, max_jj, max_kk, nT);
        }
    }
    else
    {
        Mat B_tileX(TILE_N * B * TILE_K, 1, nT, 2u, opt.workspace_allocator);
        if (B_tileX.empty())
            return -100;

        #pragma omp parallel for num_threads(nT)
        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            Mat B_tile = B_tileX.channel(get_omp_thread_num());

            // transform input
            conv3x3s1_winograd23_transform_input_tile_int8(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile_int8(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (top_tileX.empty())
        return -100;

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat top_tile = top_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

                const Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

                bool k_end = k + TILE_K >= K;

                gemm_transB_packed_tile_int8(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk, k_end);
            }

            // transform output
            conv3x3s1_winograd23_transform_output_tile_int8(top_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

static inline void conv3x3s1_winograd43_transform_kernel_tile_int8(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    // const short ktm[6][3] = {
    //     {6, 0, 0},
    //     {-4, -4, -4},
    //     {-4, 4, -4},
    //     {1, 2, 4},
    //     {1, -2, 4},
    //     {0, 0, 6}
    // };

    short* ptmp = A;

    int ii = 0;
    for (; ii < max_ii; ii++)
    {
        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            short tmp[6][3];

            const signed char* k0 = (const signed char*)kernel + (i + ii) * inch * 9 + (k + kk) * 9;

            for (int m = 0; m < 3; m++)
            {
                signed char r0 = k0[0];
                signed char r1 = k0[1];
                signed char r2 = k0[2];

                tmp[0][m] = r0 * 6;
                tmp[1][m] = -r0 * 4 - r1 * 4 - r2 * 4;
                tmp[2][m] = -r0 * 4 + r1 * 4 - r2 * 4;
                tmp[3][m] = r0 + r1 * 2 + r2 * 4;
                tmp[4][m] = r0 - r1 * 2 + r2 * 4;
                tmp[5][m] = r2 * 6;

                k0 += 3;
            }

            for (int m = 0; m < 6; m++)
            {
                short r0 = tmp[m][0];
                short r1 = tmp[m][1];
                short r2 = tmp[m][2];

                short z0 = r0 * 6;
                short z1 = -r0 * 4 - r1 * 4 - r2 * 4;
                short z2 = -r0 * 4 + r1 * 4 - r2 * 4;
                short z3 = r0 + r1 * 2 + r2 * 4;
                short z4 = r0 - r1 * 2 + r2 * 4;
                short z5 = r2 * 6;

                ptmp[0] = z0;
                ptmp[1] = z1;
                ptmp[2] = z2;
                ptmp[3] = z3;
                ptmp[4] = z4;
                ptmp[5] = z5;
                ptmp += 6;
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_kernel_int8(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 36;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat A_tileX(B * TILE_M * TILE_K, 1, opt.num_threads, 4u, (Allocator*)0);

    AT.create(TILE_K * TILE_M, B, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, 4u, (Allocator*)0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat A_tile = A_tileX.channel(get_omp_thread_num());

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            conv3x3s1_winograd43_transform_kernel_tile_int8(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            pack_A_tile_int8(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd43_transform_input_tile_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    // const float itm[4][4] = {
    //     {4,  0, -5,  0, 1, 0},
    //     {0, -4, -4,  1, 1, 0},
    //     {0,  4, -4, -1, 1, 0},
    //     {0, -2, -1,  2, 1, 0},
    //     {0,  2, -1, -2, 1, 0},
    //     {0,  4,  0, -5, 0, 1}
    // };

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const size_t N = bottom_blob.cstep * elempack;

    const int w_tiles = (w + 1) / 4;

    int nn_max_kk = 0;
    int remain_max_kk_start = 0;
#if __loongarch_sx
#if __loongarch_asx
    nn_max_kk = (max_kk - remain_max_kk_start) / 16;
    #pragma omp parallel for num_threads(nT)
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 16;

        __attribute__((aligned(32))) short tmp[6][6][16];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0 = bottom_blob.channel((k + kk) / elempack).row<const signed char>(ti * 4) + (tj * 4) * elempack;

            for (int m = 0; m < 6; m++)
            {
                __m256i _r0 = __lasx_xvldi(0);
                __m256i _r1 = __lasx_xvldi(0);
                __m256i _r2 = __lasx_xvldi(0);
                __m256i _r3 = __lasx_xvldi(0);
                __m256i _r4 = __lasx_xvldi(0);
                __m256i _r5 = __lasx_xvldi(0);

                if (ti * 4 + m < h)
                {
                    if (elempack == 8)
                    {
                        const signed char* r1 = r0 + N;

                        __m128i _t0 = __lsx_vld(r0, 0);
                        __m128i _t1 = __lsx_vld(r1, 0);
                        __m128i _t01 = __lsx_vilvl_d(_t1, _t0);
                        _r0 = __lasx_xvsext_h_b(__lsx_to_lasx(_t01));
                        if (tj * 4 + 1 < w)
                        {
                            __m128i _t0 = __lsx_vld(r0 + 8, 0);
                            __m128i _t1 = __lsx_vld(r1 + 8, 0);
                            __m128i _t01 = __lsx_vilvl_d(_t1, _t0);
                            _r1 = __lasx_xvsext_h_b(__lsx_to_lasx(_t01));
                        }
                        if (tj * 4 + 2 < w)
                        {
                            __m128i _t0 = __lsx_vld(r0 + 16, 0);
                            __m128i _t1 = __lsx_vld(r1 + 16, 0);
                            __m128i _t01 = __lsx_vilvl_d(_t1, _t0);
                            _r2 = __lasx_xvsext_h_b(__lsx_to_lasx(_t01));
                        }
                        if (tj * 4 + 3 < w)
                        {
                            __m128i _t0 = __lsx_vld(r0 + 24, 0);
                            __m128i _t1 = __lsx_vld(r1 + 24, 0);
                            __m128i _t01 = __lsx_vilvl_d(_t1, _t0);
                            _r3 = __lasx_xvsext_h_b(__lsx_to_lasx(_t01));
                        }
                        if (tj * 4 + 4 < w)
                        {
                            __m128i _t0 = __lsx_vld(r0 + 32, 0);
                            __m128i _t1 = __lsx_vld(r1 + 32, 0);
                            __m128i _t01 = __lsx_vilvl_d(_t1, _t0);
                            _r4 = __lasx_xvsext_h_b(__lsx_to_lasx(_t01));
                        }
                        if (tj * 4 + 5 < w)
                        {
                            __m128i _t0 = __lsx_vld(r0 + 40, 0);
                            __m128i _t1 = __lsx_vld(r1 + 40, 0);
                            __m128i _t01 = __lsx_vilvl_d(_t1, _t0);
                            _r5 = __lasx_xvsext_h_b(__lsx_to_lasx(_t01));
                        }
                    }
                    if (elempack == 1)
                    {
                        const signed char* r1 = r0 + N;
                        const signed char* r2 = r0 + N * 2;
                        const signed char* r3 = r0 + N * 3;
                        const signed char* r4 = r0 + N * 4;
                        const signed char* r5 = r0 + N * 5;
                        const signed char* r6 = r0 + N * 6;
                        const signed char* r7 = r0 + N * 7;
                        const signed char* r8 = r0 + N * 8;
                        const signed char* r9 = r0 + N * 9;
                        const signed char* ra = r0 + N * 10;
                        const signed char* rb = r0 + N * 11;
                        const signed char* rc = r0 + N * 12;
                        const signed char* rd = r0 + N * 13;
                        const signed char* re = r0 + N * 14;
                        const signed char* rf = r0 + N * 15;

                        {
                            __m128i _t0 = __lsx_vld(r0 + 0, 0);
                            __m128i _t1 = __lsx_vld(r1 + 0, 0);
                            __m128i _t2 = __lsx_vld(r2 + 0, 0);
                            __m128i _t3 = __lsx_vld(r3 + 0, 0);
                            __m128i _t4 = __lsx_vld(r4 + 0, 0);
                            __m128i _t5 = __lsx_vld(r5 + 0, 0);
                            __m128i _t6 = __lsx_vld(r6 + 0, 0);
                            __m128i _t7 = __lsx_vld(r7 + 0, 0);
                            __m128i _t8 = __lsx_vld(r8 + 0, 0);
                            __m128i _t9 = __lsx_vld(r9 + 0, 0);
                            __m128i _ta = __lsx_vld(ra + 0, 0);
                            __m128i _tb = __lsx_vld(rb + 0, 0);
                            __m128i _tc = __lsx_vld(rc + 0, 0);
                            __m128i _td = __lsx_vld(rd + 0, 0);
                            __m128i _te = __lsx_vld(re + 0, 0);
                            __m128i _tf = __lsx_vld(rf + 0, 0);

                            __m128i _t01 = __lsx_vilvl_b(_t1, _t0);
                            __m128i _t23 = __lsx_vilvl_b(_t3, _t2);
                            __m128i _t45 = __lsx_vilvl_b(_t5, _t4);
                            __m128i _t67 = __lsx_vilvl_b(_t7, _t6);
                            __m128i _t89 = __lsx_vilvl_b(_t9, _t8);
                            __m128i _tab = __lsx_vilvl_b(_tb, _ta);
                            __m128i _tcd = __lsx_vilvl_b(_td, _tc);
                            __m128i _tef = __lsx_vilvl_b(_tf, _te);
                            __m128i _u0 = __lsx_vilvl_h(_t23, _t01);
                            __m128i _u1 = __lsx_vilvl_h(_t67, _t45);
                            __m128i _u2 = __lsx_vilvl_h(_tab, _t89);
                            __m128i _u3 = __lsx_vilvl_h(_tef, _tcd);
                            __m128i _v0 = __lsx_vilvl_w(_u1, _u0);
                            __m128i _v1 = __lsx_vilvl_w(_u3, _u2);
                            __m128i _val = __lsx_vilvl_d(_v1, _v0);
                            _r0 = __lasx_xvsext_h_b(__lsx_to_lasx(_val));
                        }
                        if (tj * 4 + 1 < w)
                        {
                            __m128i _t0 = __lsx_vld(r0 + 1, 0);
                            __m128i _t1 = __lsx_vld(r1 + 1, 0);
                            __m128i _t2 = __lsx_vld(r2 + 1, 0);
                            __m128i _t3 = __lsx_vld(r3 + 1, 0);
                            __m128i _t4 = __lsx_vld(r4 + 1, 0);
                            __m128i _t5 = __lsx_vld(r5 + 1, 0);
                            __m128i _t6 = __lsx_vld(r6 + 1, 0);
                            __m128i _t7 = __lsx_vld(r7 + 1, 0);
                            __m128i _t8 = __lsx_vld(r8 + 1, 0);
                            __m128i _t9 = __lsx_vld(r9 + 1, 0);
                            __m128i _ta = __lsx_vld(ra + 1, 0);
                            __m128i _tb = __lsx_vld(rb + 1, 0);
                            __m128i _tc = __lsx_vld(rc + 1, 0);
                            __m128i _td = __lsx_vld(rd + 1, 0);
                            __m128i _te = __lsx_vld(re + 1, 0);
                            __m128i _tf = __lsx_vld(rf + 1, 0);

                            __m128i _t01 = __lsx_vilvl_b(_t1, _t0);
                            __m128i _t23 = __lsx_vilvl_b(_t3, _t2);
                            __m128i _t45 = __lsx_vilvl_b(_t5, _t4);
                            __m128i _t67 = __lsx_vilvl_b(_t7, _t6);
                            __m128i _t89 = __lsx_vilvl_b(_t9, _t8);
                            __m128i _tab = __lsx_vilvl_b(_tb, _ta);
                            __m128i _tcd = __lsx_vilvl_b(_td, _tc);
                            __m128i _tef = __lsx_vilvl_b(_tf, _te);
                            __m128i _u0 = __lsx_vilvl_h(_t23, _t01);
                            __m128i _u1 = __lsx_vilvl_h(_t67, _t45);
                            __m128i _u2 = __lsx_vilvl_h(_tab, _t89);
                            __m128i _u3 = __lsx_vilvl_h(_tef, _tcd);
                            __m128i _v0 = __lsx_vilvl_w(_u1, _u0);
                            __m128i _v1 = __lsx_vilvl_w(_u3, _u2);
                            __m128i _val = __lsx_vilvl_d(_v1, _v0);
                            _r1 = __lasx_xvsext_h_b(__lsx_to_lasx(_val));
                        }
                        if (tj * 4 + 2 < w)
                        {
                            __m128i _t0 = __lsx_vld(r0 + 2, 0);
                            __m128i _t1 = __lsx_vld(r1 + 2, 0);
                            __m128i _t2 = __lsx_vld(r2 + 2, 0);
                            __m128i _t3 = __lsx_vld(r3 + 2, 0);
                            __m128i _t4 = __lsx_vld(r4 + 2, 0);
                            __m128i _t5 = __lsx_vld(r5 + 2, 0);
                            __m128i _t6 = __lsx_vld(r6 + 2, 0);
                            __m128i _t7 = __lsx_vld(r7 + 2, 0);
                            __m128i _t8 = __lsx_vld(r8 + 2, 0);
                            __m128i _t9 = __lsx_vld(r9 + 2, 0);
                            __m128i _ta = __lsx_vld(ra + 2, 0);
                            __m128i _tb = __lsx_vld(rb + 2, 0);
                            __m128i _tc = __lsx_vld(rc + 2, 0);
                            __m128i _td = __lsx_vld(rd + 2, 0);
                            __m128i _te = __lsx_vld(re + 2, 0);
                            __m128i _tf = __lsx_vld(rf + 2, 0);

                            __m128i _t01 = __lsx_vilvl_b(_t1, _t0);
                            __m128i _t23 = __lsx_vilvl_b(_t3, _t2);
                            __m128i _t45 = __lsx_vilvl_b(_t5, _t4);
                            __m128i _t67 = __lsx_vilvl_b(_t7, _t6);
                            __m128i _t89 = __lsx_vilvl_b(_t9, _t8);
                            __m128i _tab = __lsx_vilvl_b(_tb, _ta);
                            __m128i _tcd = __lsx_vilvl_b(_td, _tc);
                            __m128i _tef = __lsx_vilvl_b(_tf, _te);
                            __m128i _u0 = __lsx_vilvl_h(_t23, _t01);
                            __m128i _u1 = __lsx_vilvl_h(_t67, _t45);
                            __m128i _u2 = __lsx_vilvl_h(_tab, _t89);
                            __m128i _u3 = __lsx_vilvl_h(_tef, _tcd);
                            __m128i _v0 = __lsx_vilvl_w(_u1, _u0);
                            __m128i _v1 = __lsx_vilvl_w(_u3, _u2);
                            __m128i _val = __lsx_vilvl_d(_v1, _v0);
                            _r2 = __lasx_xvsext_h_b(__lsx_to_lasx(_val));
                        }
                        if (tj * 4 + 3 < w)
                        {
                            __m128i _t0 = __lsx_vld(r0 + 3, 0);
                            __m128i _t1 = __lsx_vld(r1 + 3, 0);
                            __m128i _t2 = __lsx_vld(r2 + 3, 0);
                            __m128i _t3 = __lsx_vld(r3 + 3, 0);
                            __m128i _t4 = __lsx_vld(r4 + 3, 0);
                            __m128i _t5 = __lsx_vld(r5 + 3, 0);
                            __m128i _t6 = __lsx_vld(r6 + 3, 0);
                            __m128i _t7 = __lsx_vld(r7 + 3, 0);
                            __m128i _t8 = __lsx_vld(r8 + 3, 0);
                            __m128i _t9 = __lsx_vld(r9 + 3, 0);
                            __m128i _ta = __lsx_vld(ra + 3, 0);
                            __m128i _tb = __lsx_vld(rb + 3, 0);
                            __m128i _tc = __lsx_vld(rc + 3, 0);
                            __m128i _td = __lsx_vld(rd + 3, 0);
                            __m128i _te = __lsx_vld(re + 3, 0);
                            __m128i _tf = __lsx_vld(rf + 3, 0);

                            __m128i _t01 = __lsx_vilvl_b(_t1, _t0);
                            __m128i _t23 = __lsx_vilvl_b(_t3, _t2);
                            __m128i _t45 = __lsx_vilvl_b(_t5, _t4);
                            __m128i _t67 = __lsx_vilvl_b(_t7, _t6);
                            __m128i _t89 = __lsx_vilvl_b(_t9, _t8);
                            __m128i _tab = __lsx_vilvl_b(_tb, _ta);
                            __m128i _tcd = __lsx_vilvl_b(_td, _tc);
                            __m128i _tef = __lsx_vilvl_b(_tf, _te);
                            __m128i _u0 = __lsx_vilvl_h(_t23, _t01);
                            __m128i _u1 = __lsx_vilvl_h(_t67, _t45);
                            __m128i _u2 = __lsx_vilvl_h(_tab, _t89);
                            __m128i _u3 = __lsx_vilvl_h(_tef, _tcd);
                            __m128i _v0 = __lsx_vilvl_w(_u1, _u0);
                            __m128i _v1 = __lsx_vilvl_w(_u3, _u2);
                            __m128i _val = __lsx_vilvl_d(_v1, _v0);
                            _r3 = __lasx_xvsext_h_b(__lsx_to_lasx(_val));
                        }
                        if (tj * 4 + 4 < w)
                        {
                            __m128i _t0 = __lsx_vld(r0 + 4, 0);
                            __m128i _t1 = __lsx_vld(r1 + 4, 0);
                            __m128i _t2 = __lsx_vld(r2 + 4, 0);
                            __m128i _t3 = __lsx_vld(r3 + 4, 0);
                            __m128i _t4 = __lsx_vld(r4 + 4, 0);
                            __m128i _t5 = __lsx_vld(r5 + 4, 0);
                            __m128i _t6 = __lsx_vld(r6 + 4, 0);
                            __m128i _t7 = __lsx_vld(r7 + 4, 0);
                            __m128i _t8 = __lsx_vld(r8 + 4, 0);
                            __m128i _t9 = __lsx_vld(r9 + 4, 0);
                            __m128i _ta = __lsx_vld(ra + 4, 0);
                            __m128i _tb = __lsx_vld(rb + 4, 0);
                            __m128i _tc = __lsx_vld(rc + 4, 0);
                            __m128i _td = __lsx_vld(rd + 4, 0);
                            __m128i _te = __lsx_vld(re + 4, 0);
                            __m128i _tf = __lsx_vld(rf + 4, 0);

                            __m128i _t01 = __lsx_vilvl_b(_t1, _t0);
                            __m128i _t23 = __lsx_vilvl_b(_t3, _t2);
                            __m128i _t45 = __lsx_vilvl_b(_t5, _t4);
                            __m128i _t67 = __lsx_vilvl_b(_t7, _t6);
                            __m128i _t89 = __lsx_vilvl_b(_t9, _t8);
                            __m128i _tab = __lsx_vilvl_b(_tb, _ta);
                            __m128i _tcd = __lsx_vilvl_b(_td, _tc);
                            __m128i _tef = __lsx_vilvl_b(_tf, _te);
                            __m128i _u0 = __lsx_vilvl_h(_t23, _t01);
                            __m128i _u1 = __lsx_vilvl_h(_t67, _t45);
                            __m128i _u2 = __lsx_vilvl_h(_tab, _t89);
                            __m128i _u3 = __lsx_vilvl_h(_tef, _tcd);
                            __m128i _v0 = __lsx_vilvl_w(_u1, _u0);
                            __m128i _v1 = __lsx_vilvl_w(_u3, _u2);
                            __m128i _val = __lsx_vilvl_d(_v1, _v0);
                            _r4 = __lasx_xvsext_h_b(__lsx_to_lasx(_val));
                        }
                        if (tj * 4 + 5 < w)
                        {
                            __m128i _t0 = __lsx_vld(r0 + 5, 0);
                            __m128i _t1 = __lsx_vld(r1 + 5, 0);
                            __m128i _t2 = __lsx_vld(r2 + 5, 0);
                            __m128i _t3 = __lsx_vld(r3 + 5, 0);
                            __m128i _t4 = __lsx_vld(r4 + 5, 0);
                            __m128i _t5 = __lsx_vld(r5 + 5, 0);
                            __m128i _t6 = __lsx_vld(r6 + 5, 0);
                            __m128i _t7 = __lsx_vld(r7 + 5, 0);
                            __m128i _t8 = __lsx_vld(r8 + 5, 0);
                            __m128i _t9 = __lsx_vld(r9 + 5, 0);
                            __m128i _ta = __lsx_vld(ra + 5, 0);
                            __m128i _tb = __lsx_vld(rb + 5, 0);
                            __m128i _tc = __lsx_vld(rc + 5, 0);
                            __m128i _td = __lsx_vld(rd + 5, 0);
                            __m128i _te = __lsx_vld(re + 5, 0);
                            __m128i _tf = __lsx_vld(rf + 5, 0);

                            __m128i _t01 = __lsx_vilvl_b(_t1, _t0);
                            __m128i _t23 = __lsx_vilvl_b(_t3, _t2);
                            __m128i _t45 = __lsx_vilvl_b(_t5, _t4);
                            __m128i _t67 = __lsx_vilvl_b(_t7, _t6);
                            __m128i _t89 = __lsx_vilvl_b(_t9, _t8);
                            __m128i _tab = __lsx_vilvl_b(_tb, _ta);
                            __m128i _tcd = __lsx_vilvl_b(_td, _tc);
                            __m128i _tef = __lsx_vilvl_b(_tf, _te);
                            __m128i _u0 = __lsx_vilvl_h(_t23, _t01);
                            __m128i _u1 = __lsx_vilvl_h(_t67, _t45);
                            __m128i _u2 = __lsx_vilvl_h(_tab, _t89);
                            __m128i _u3 = __lsx_vilvl_h(_tef, _tcd);
                            __m128i _v0 = __lsx_vilvl_w(_u1, _u0);
                            __m128i _v1 = __lsx_vilvl_w(_u3, _u2);
                            __m128i _val = __lsx_vilvl_d(_v1, _v0);
                            _r5 = __lasx_xvsext_h_b(__lsx_to_lasx(_val));
                        }
                    }
                }

                __m256i _v4 = __lasx_xvreplgr2vr_h(4);
                __m256i _v5 = __lasx_xvreplgr2vr_h(5);
                __m256i _v2 = __lasx_xvreplgr2vr_h(2);

                __m256i _tmp12a = __lasx_xvsub_h(_r3, __lasx_xvmul_h(_r1, _v4));
                __m256i _tmp12b = __lasx_xvsub_h(_r4, __lasx_xvmul_h(_r2, _v4));
                __m256i _tmp34a = __lasx_xvmul_h(__lasx_xvsub_h(_r3, _r1), _v2);
                __m256i _tmp34b = __lasx_xvsub_h(_r4, _r2);

                __m256i _tmp0 = __lasx_xvadd_h(_r4, __lasx_xvsub_h(__lasx_xvmul_h(_r0, _v4), __lasx_xvmul_h(_r2, _v5)));
                __m256i _tmp1 = __lasx_xvadd_h(_tmp12b, _tmp12a);
                __m256i _tmp2 = __lasx_xvsub_h(_tmp12b, _tmp12a);
                __m256i _tmp3 = __lasx_xvadd_h(_tmp34b, _tmp34a);
                __m256i _tmp4 = __lasx_xvsub_h(_tmp34b, _tmp34a);
                __m256i _tmp5 = __lasx_xvadd_h(_r5, __lasx_xvsub_h(__lasx_xvmul_h(_r1, _v4), __lasx_xvmul_h(_r3, _v5)));

                __lasx_xvst(_tmp0, (short*)tmp[0][m], 0);
                __lasx_xvst(_tmp1, (short*)tmp[1][m], 0);
                __lasx_xvst(_tmp2, (short*)tmp[2][m], 0);
                __lasx_xvst(_tmp3, (short*)tmp[3][m], 0);
                __lasx_xvst(_tmp4, (short*)tmp[4][m], 0);
                __lasx_xvst(_tmp5, (short*)tmp[5][m], 0);

                r0 += w * elempack;
            }

            short* p0 = (short*)B + kk * max_jj * 36 + jj * 16;
            short* p1 = p0 + max_jj * 16 * 1;
            short* p2 = p0 + max_jj * 16 * 2;
            short* p3 = p0 + max_jj * 16 * 3;
            short* p4 = p0 + max_jj * 16 * 4;
            short* p5 = p0 + max_jj * 16 * 5;

            for (int m = 0; m < 6; m++)
            {
                __m256i _r0 = __lasx_xvld((const __m256i*)tmp[m][0], 0);
                __m256i _r1 = __lasx_xvld((const __m256i*)tmp[m][1], 0);
                __m256i _r2 = __lasx_xvld((const __m256i*)tmp[m][2], 0);
                __m256i _r3 = __lasx_xvld((const __m256i*)tmp[m][3], 0);
                __m256i _r4 = __lasx_xvld((const __m256i*)tmp[m][4], 0);
                __m256i _r5 = __lasx_xvld((const __m256i*)tmp[m][5], 0);

                __m256i _v4 = __lasx_xvreplgr2vr_h(4);
                __m256i _v5 = __lasx_xvreplgr2vr_h(5);
                __m256i _v2 = __lasx_xvreplgr2vr_h(2);

                __m256i _tmp12a = __lasx_xvsub_h(_r3, __lasx_xvmul_h(_r1, _v4));
                __m256i _tmp12b = __lasx_xvsub_h(_r4, __lasx_xvmul_h(_r2, _v4));
                __m256i _tmp34a = __lasx_xvmul_h(__lasx_xvsub_h(_r3, _r1), _v2);
                __m256i _tmp34b = __lasx_xvsub_h(_r4, _r2);

                __m256i _tmp0 = __lasx_xvadd_h(_r4, __lasx_xvsub_h(__lasx_xvmul_h(_r0, _v4), __lasx_xvmul_h(_r2, _v5)));
                __m256i _tmp1 = __lasx_xvadd_h(_tmp12b, _tmp12a);
                __m256i _tmp2 = __lasx_xvsub_h(_tmp12b, _tmp12a);
                __m256i _tmp3 = __lasx_xvadd_h(_tmp34b, _tmp34a);
                __m256i _tmp4 = __lasx_xvsub_h(_tmp34b, _tmp34a);
                __m256i _tmp5 = __lasx_xvadd_h(_r5, __lasx_xvsub_h(__lasx_xvmul_h(_r1, _v4), __lasx_xvmul_h(_r3, _v5)));

                __lasx_xvst(_tmp0, p0, 0);
                __lasx_xvst(_tmp1, p1, 0);
                __lasx_xvst(_tmp2, p2, 0);
                __lasx_xvst(_tmp3, p3, 0);
                __lasx_xvst(_tmp4, p4, 0);
                __lasx_xvst(_tmp5, p5, 0);

                p0 += max_jj * 6 * 16;
                p1 += max_jj * 6 * 16;
                p2 += max_jj * 6 * 16;
                p3 += max_jj * 6 * 16;
                p4 += max_jj * 6 * 16;
                p5 += max_jj * 6 * 16;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 16;
    nn_max_kk = (max_kk - remain_max_kk_start) / 8;
#else // __loongarch_asx
    nn_max_kk = (max_kk - remain_max_kk_start) / 8;
    #pragma omp parallel for num_threads(nT)
#endif // __loongarch_asx
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 8;

        __attribute__((aligned(16))) short tmp[6][6][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0 = bottom_blob.channel((k + kk) / elempack).row<const signed char>(ti * 4) + (tj * 4) * elempack;

            __m128i _v2 = __lsx_vreplgr2vr_h(2);
            __m128i _v4 = __lsx_vreplgr2vr_h(4);
            __m128i _v5 = __lsx_vreplgr2vr_h(5);

            for (int m = 0; m < 6; m++)
            {
                __m128i _r0 = __lsx_vldi(0);
                __m128i _r1 = __lsx_vldi(0);
                __m128i _r2 = __lsx_vldi(0);
                __m128i _r3 = __lsx_vldi(0);
                __m128i _r4 = __lsx_vldi(0);
                __m128i _r5 = __lsx_vldi(0);

                if (ti * 4 + m < h)
                {
                    if (elempack == 8)
                    {
                        _r0 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)r0, 0);
                        _r0 = __lsx_vsllwil_h_b(_r0, 0);
                        if (tj * 4 + 1 < w)
                        {
                            _r1 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)(r0 + 8), 0);
                            _r1 = __lsx_vsllwil_h_b(_r1, 0);
                        }
                        if (tj * 4 + 2 < w)
                        {
                            _r2 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)(r0 + 16), 0);
                            _r2 = __lsx_vsllwil_h_b(_r2, 0);
                        }
                        if (tj * 4 + 3 < w)
                        {
                            _r3 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)(r0 + 24), 0);
                            _r3 = __lsx_vsllwil_h_b(_r3, 0);
                        }
                        if (tj * 4 + 4 < w)
                        {
                            _r4 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)(r0 + 32), 0);
                            _r4 = __lsx_vsllwil_h_b(_r4, 0);
                        }
                        if (tj * 4 + 5 < w)
                        {
                            _r5 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)(r0 + 40), 0);
                            _r5 = __lsx_vsllwil_h_b(_r5, 0);
                        }
                    }
                    if (elempack == 1)
                    {
                        const signed char* r1 = r0 + N;
                        const signed char* r2 = r0 + N * 2;
                        const signed char* r3 = r0 + N * 3;
                        const signed char* r4 = r0 + N * 4;
                        const signed char* r5 = r0 + N * 5;
                        const signed char* r6 = r0 + N * 6;
                        const signed char* r7 = r0 + N * 7;

                        __m128i _t0 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)r0, 0);
                        __m128i _t1 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)r1, 0);
                        __m128i _t2 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)r2, 0);
                        __m128i _t3 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)r3, 0);
                        __m128i _t4 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)r4, 0);
                        __m128i _t5 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)r5, 0);
                        __m128i _t6 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)r6, 0);
                        __m128i _t7 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(int64_t*)r7, 0);

                        __m128i _t01 = __lsx_vilvl_b(_t1, _t0);
                        __m128i _t23 = __lsx_vilvl_b(_t3, _t2);
                        __m128i _t45 = __lsx_vilvl_b(_t5, _t4);
                        __m128i _t67 = __lsx_vilvl_b(_t7, _t6);
                        _t0 = __lsx_vilvl_h(_t23, _t01);
                        _t1 = __lsx_vilvl_h(_t67, _t45);
                        _t2 = __lsx_vilvl_w(_t1, _t0);
                        _t3 = __lsx_vilvh_w(_t1, _t0);

                        __m128i _extt2 = __lsx_vsrai_b(_t2, 7);
                        __m128i _extt3 = __lsx_vsrai_b(_t3, 7);

                        _r0 = __lsx_vilvl_b(_extt2, _t2);
                        if (tj * 4 + 1 < w) _r1 = __lsx_vilvh_b(_extt2, _t2);
                        if (tj * 4 + 2 < w) _r2 = __lsx_vilvl_b(_extt3, _t3);
                        if (tj * 4 + 3 < w) _r3 = __lsx_vilvh_b(_extt3, _t3);
                        if (tj * 4 + 4 < w)
                        {
                            __attribute__((aligned(16))) short _r4tmp[8] = {(short)r0[4], (short)r1[4], (short)r2[4], (short)r3[4], (short)r4[4], (short)r5[4], (short)r6[4], (short)r7[4]};
                            _r4 = __lsx_vld(_r4tmp, 0);
                        }
                        if (tj * 4 + 5 < w)
                        {
                            __attribute__((aligned(16))) short _r5tmp[8] = {(short)r0[5], (short)r1[5], (short)r2[5], (short)r3[5], (short)r4[5], (short)r5[5], (short)r6[5], (short)r7[5]};
                            _r5 = __lsx_vld(_r5tmp, 0);
                        }
                    }
                }

                __m128i _tmp12a = __lsx_vsub_h(_r3, __lsx_vmul_h(_r1, _v4));
                __m128i _tmp12b = __lsx_vsub_h(_r4, __lsx_vmul_h(_r2, _v4));
                __m128i _tmp34a = __lsx_vmul_h(__lsx_vsub_h(_r3, _r1), _v2);
                __m128i _tmp34b = __lsx_vsub_h(_r4, _r2);

                __m128i _tmp0 = __lsx_vadd_h(_r4, __lsx_vsub_h(__lsx_vmul_h(_r0, _v4), __lsx_vmul_h(_r2, _v5)));
                __m128i _tmp1 = __lsx_vadd_h(_tmp12b, _tmp12a);
                __m128i _tmp2 = __lsx_vsub_h(_tmp12b, _tmp12a);
                __m128i _tmp3 = __lsx_vadd_h(_tmp34b, _tmp34a);
                __m128i _tmp4 = __lsx_vsub_h(_tmp34b, _tmp34a);
                __m128i _tmp5 = __lsx_vadd_h(_r5, __lsx_vsub_h(__lsx_vmul_h(_r1, _v4), __lsx_vmul_h(_r3, _v5)));

                __lsx_vst(_tmp0, tmp[0][m], 0);
                __lsx_vst(_tmp1, tmp[1][m], 0);
                __lsx_vst(_tmp2, tmp[2][m], 0);
                __lsx_vst(_tmp3, tmp[3][m], 0);
                __lsx_vst(_tmp4, tmp[4][m], 0);
                __lsx_vst(_tmp5, tmp[5][m], 0);

                r0 += w * elempack;
            }

            short* p0 = (short*)B + kk * max_jj * 36 + jj * 8;
            short* p1 = p0 + max_jj * 8;
            short* p2 = p0 + max_jj * 8 * 2;
            short* p3 = p0 + max_jj * 8 * 3;
            short* p4 = p0 + max_jj * 8 * 4;
            short* p5 = p0 + max_jj * 8 * 5;

            for (int m = 0; m < 6; m++)
            {
                __m128i _r0 = __lsx_vld(tmp[m][0], 0);
                __m128i _r1 = __lsx_vld(tmp[m][1], 0);
                __m128i _r2 = __lsx_vld(tmp[m][2], 0);
                __m128i _r3 = __lsx_vld(tmp[m][3], 0);
                __m128i _r4 = __lsx_vld(tmp[m][4], 0);
                __m128i _r5 = __lsx_vld(tmp[m][5], 0);

                __m128i _tmp12a = __lsx_vsub_h(_r3, __lsx_vmul_h(_r1, _v4));
                __m128i _tmp12b = __lsx_vsub_h(_r4, __lsx_vmul_h(_r2, _v4));
                __m128i _tmp34a = __lsx_vmul_h(__lsx_vsub_h(_r3, _r1), _v2);
                __m128i _tmp34b = __lsx_vsub_h(_r4, _r2);

                __m128i _tmp0 = __lsx_vadd_h(_r4, __lsx_vsub_h(__lsx_vmul_h(_r0, _v4), __lsx_vmul_h(_r2, _v5)));
                __m128i _tmp1 = __lsx_vadd_h(_tmp12b, _tmp12a);
                __m128i _tmp2 = __lsx_vsub_h(_tmp12b, _tmp12a);
                __m128i _tmp3 = __lsx_vadd_h(_tmp34b, _tmp34a);
                __m128i _tmp4 = __lsx_vsub_h(_tmp34b, _tmp34a);
                __m128i _tmp5 = __lsx_vadd_h(_r5, __lsx_vsub_h(__lsx_vmul_h(_r1, _v4), __lsx_vmul_h(_r3, _v5)));

                __lsx_vst(_tmp0, p0, 0);
                __lsx_vst(_tmp1, p1, 0);
                __lsx_vst(_tmp2, p2, 0);
                __lsx_vst(_tmp3, p3, 0);
                __lsx_vst(_tmp4, p4, 0);
                __lsx_vst(_tmp5, p5, 0);

                p0 += max_jj * 6 * 8;
                p1 += max_jj * 6 * 8;
                p2 += max_jj * 6 * 8;
                p3 += max_jj * 6 * 8;
                p4 += max_jj * 6 * 8;
                p5 += max_jj * 6 * 8;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 8;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
#else
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    #pragma omp parallel for num_threads(nT)
#endif // __loongarch_sx
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

        short tmp[6][6][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0 = bottom_blob.channel(k + kk).row<const signed char>(ti * 4) + (tj * 4);

            for (int m = 0; m < 6; m++)
            {
                signed char r00 = 0;
                signed char r01 = 0;
                signed char r10 = 0;
                signed char r11 = 0;
                signed char r20 = 0;
                signed char r21 = 0;
                signed char r30 = 0;
                signed char r31 = 0;
                signed char r40 = 0;
                signed char r41 = 0;
                signed char r50 = 0;
                signed char r51 = 0;

                if (ti * 4 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const signed char* r1 = r0 + N;

                        r00 = r0[0];
                        r01 = r1[0];
                        if (tj * 4 + 1 < w)
                        {
                            r10 = r0[1];
                            r11 = r1[1];
                        }
                        if (tj * 4 + 2 < w)
                        {
                            r20 = r0[2];
                            r21 = r1[2];
                        }
                        if (tj * 4 + 3 < w)
                        {
                            r30 = r0[3];
                            r31 = r1[3];
                        }
                        if (tj * 4 + 4 < w)
                        {
                            r40 = r0[4];
                            r41 = r1[4];
                        }
                        if (tj * 4 + 5 < w)
                        {
                            r50 = r0[5];
                            r51 = r1[5];
                        }
                    }
                }

                short tmp120a = r30 - r10 * 4;
                short tmp121a = r31 - r11 * 4;
                short tmp120b = r40 - r20 * 4;
                short tmp121b = r41 - r21 * 4;
                short tmp340a = (r30 - r10) * 2;
                short tmp341a = (r31 - r11) * 2;
                short tmp340b = r40 - r20;
                short tmp341b = r41 - r21;

                tmp[0][m][0] = r40 + r00 * 4 - r20 * 5;
                tmp[0][m][1] = r41 + r01 * 4 - r21 * 5;
                tmp[1][m][0] = tmp120b + tmp120a;
                tmp[1][m][1] = tmp121b + tmp121a;
                tmp[2][m][0] = tmp120b - tmp120a;
                tmp[2][m][1] = tmp121b - tmp121a;
                tmp[3][m][0] = tmp340b + tmp340a;
                tmp[3][m][1] = tmp341b + tmp341a;
                tmp[4][m][0] = tmp340b - tmp340a;
                tmp[4][m][1] = tmp341b - tmp341a;
                tmp[5][m][0] = r50 + r10 * 4 - r30 * 5;
                tmp[5][m][1] = r51 + r11 * 4 - r31 * 5;

                r0 += w;
            }

            short* p0 = (short*)B + kk * max_jj * 36 + jj * 2;
            short* p1 = p0 + max_jj * 2;
            short* p2 = p0 + max_jj * 2 * 2;
            short* p3 = p0 + max_jj * 2 * 3;
            short* p4 = p0 + max_jj * 2 * 4;
            short* p5 = p0 + max_jj * 2 * 5;

            for (int m = 0; m < 6; m++)
            {
                short r00 = tmp[m][0][0];
                short r01 = tmp[m][0][1];
                short r10 = tmp[m][1][0];
                short r11 = tmp[m][1][1];
                short r20 = tmp[m][2][0];
                short r21 = tmp[m][2][1];
                short r30 = tmp[m][3][0];
                short r31 = tmp[m][3][1];
                short r40 = tmp[m][4][0];
                short r41 = tmp[m][4][1];
                short r50 = tmp[m][5][0];
                short r51 = tmp[m][5][1];

                short tmp120a = r30 - r10 * 4;
                short tmp121a = r31 - r11 * 4;
                short tmp120b = r40 - r20 * 4;
                short tmp121b = r41 - r21 * 4;
                short tmp340a = (r30 - r10) * 2;
                short tmp341a = (r31 - r11) * 2;
                short tmp340b = r40 - r20;
                short tmp341b = r41 - r21;

                p0[0] = r40 + r00 * 4 - r20 * 5;
                p0[1] = r41 + r01 * 4 - r21 * 5;
                p1[0] = tmp120b + tmp120a;
                p1[1] = tmp121b + tmp121a;
                p2[0] = tmp120b - tmp120a;
                p2[1] = tmp121b - tmp121a;
                p3[0] = tmp340b + tmp340a;
                p3[1] = tmp341b + tmp341a;
                p4[0] = tmp340b - tmp340a;
                p4[1] = tmp341b - tmp341a;
                p5[0] = r50 + r10 * 4 - r30 * 5;
                p5[1] = r51 + r11 * 4 - r31 * 5;

                p0 += max_jj * 6 * 2;
                p1 += max_jj * 6 * 2;
                p2 += max_jj * 6 * 2;
                p3 += max_jj * 6 * 2;
                p4 += max_jj * 6 * 2;
                p5 += max_jj * 6 * 2;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 2;
    for (int kk = remain_max_kk_start; kk < max_kk; kk++)
    {
        short tmp[6][6];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0123 = bottom_blob.channel(k + kk).row<const signed char>(ti * 4) + (tj * 4);

            for (int m = 0; m < 6; m++)
            {
                signed char r0 = 0;
                signed char r1 = 0;
                signed char r2 = 0;
                signed char r3 = 0;
                signed char r4 = 0;
                signed char r5 = 0;

                if (ti * 4 + m < h)
                {
                    // if (elempack == 1)
                    {
                        r0 = r0123[0];
                        if (tj * 4 + 1 < w) r1 = r0123[1];
                        if (tj * 4 + 2 < w) r2 = r0123[2];
                        if (tj * 4 + 3 < w) r3 = r0123[3];
                        if (tj * 4 + 4 < w) r4 = r0123[4];
                        if (tj * 4 + 5 < w) r5 = r0123[5];
                    }
                }

                short tmp12a = r3 - r1 * 4;
                short tmp12b = r4 - r2 * 4;
                short tmp34a = (r3 - r1) * 2;
                short tmp34b = r4 - r2;

                tmp[0][m] = r4 + r0 * 4 - r2 * 5;
                tmp[1][m] = tmp12b + tmp12a;
                tmp[2][m] = tmp12b - tmp12a;
                tmp[3][m] = tmp34b + tmp34a;
                tmp[4][m] = tmp34b - tmp34a;
                tmp[5][m] = r5 + r1 * 4 - r3 * 5;

                r0123 += w;
            }

            short* p0 = (short*)B + kk * max_jj * 36 + jj;
            short* p1 = p0 + max_jj;
            short* p2 = p0 + max_jj * 2;
            short* p3 = p0 + max_jj * 3;
            short* p4 = p0 + max_jj * 4;
            short* p5 = p0 + max_jj * 5;

            for (int m = 0; m < 6; m++)
            {
                short r0 = tmp[m][0];
                short r1 = tmp[m][1];
                short r2 = tmp[m][2];
                short r3 = tmp[m][3];
                short r4 = tmp[m][4];
                short r5 = tmp[m][5];

                short tmp12a = r3 - r1 * 4;
                short tmp12b = r4 - r2 * 4;
                short tmp34a = (r3 - r1) * 2;
                short tmp34b = r4 - r2;

                p0[0] = r4 + r0 * 4 - r2 * 5;
                p1[0] = tmp12b + tmp12a;
                p2[0] = tmp12b - tmp12a;
                p3[0] = tmp34b + tmp34a;
                p4[0] = tmp34b - tmp34a;
                p5[0] = r5 + r1 * 4 - r3 * 5;

                p0 += max_jj * 6;
                p1 += max_jj * 6;
                p2 += max_jj * 6;
                p3 += max_jj * 6;
                p4 += max_jj * 6;
                p5 += max_jj * 6;
            }
        }
    }
}

static inline void conv3x3s1_winograd43_transform_output_tile_int8(const Mat& top_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    // const int otm[4][6] = {
    //     {1, 1,  1, 1,  1, 0},
    //     {0, 1, -1, 2, -2, 0},
    //     {0, 1,  1, 4,  4, 0},
    //     {0, 1, -1, 8, -8, 1}
    // };

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const size_t N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 3) / 4;

    int ii = 0;
#if __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        __attribute__((aligned(32))) int tmp[4][6][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 36 + jj * 8;
            const int* r1 = r0 + max_jj * 8 * 1;
            const int* r2 = r0 + max_jj * 8 * 2;
            const int* r3 = r0 + max_jj * 8 * 3;
            const int* r4 = r0 + max_jj * 8 * 4;
            const int* r5 = r0 + max_jj * 8 * 5;

            for (int m = 0; m < 5; m++)
            {
                __m256i _r0 = __lasx_xvld(r0, 0);
                __m256i _r1 = __lasx_xvld(r1, 0);
                __m256i _r2 = __lasx_xvld(r2, 0);
                __m256i _r3 = __lasx_xvld(r3, 0);
                __m256i _r4 = __lasx_xvld(r4, 0);
                __m256i _r5 = __lasx_xvld(r5, 0);

                __m256i _tmp02a = __lasx_xvadd_w(_r1, _r2);
                __m256i _tmp02b = __lasx_xvadd_w(_r3, _r4);
                __m256i _tmp13a = __lasx_xvsub_w(_r1, _r2);
                __m256i _tmp13b = __lasx_xvsub_w(_r3, _r4);

                __m256i _tmp0 = __lasx_xvadd_w(__lasx_xvadd_w(_tmp02a, _tmp02b), _r0);
                __m256i _tmp1 = __lasx_xvadd_w(_tmp13a, __lasx_xvslli_w(_tmp13b, 1));
                __m256i _tmp2 = __lasx_xvadd_w(_tmp02a, __lasx_xvslli_w(_tmp02b, 2));
                __m256i _tmp3 = __lasx_xvadd_w(__lasx_xvadd_w(_tmp13a, __lasx_xvslli_w(_tmp13b, 3)), __lasx_xvslli_w(_r5, 2));

                __lasx_xvst(_tmp0, (int*)tmp[0][m], 0);
                __lasx_xvst(_tmp1, (int*)tmp[1][m], 0);
                __lasx_xvst(_tmp2, (int*)tmp[2][m], 0);
                __lasx_xvst(_tmp3, (int*)tmp[3][m], 0);

                r0 += max_jj * 6 * 8;
                r1 += max_jj * 6 * 8;
                r2 += max_jj * 6 * 8;
                r3 += max_jj * 6 * 8;
                r4 += max_jj * 6 * 8;
                r5 += max_jj * 6 * 8;
            }
            for (int m = 5; m < 6; m++)
            {
                __m256i _r0 = __lasx_xvld(r0, 0);
                __m256i _r1 = __lasx_xvld(r1, 0);
                __m256i _r2 = __lasx_xvld(r2, 0);
                __m256i _r3 = __lasx_xvld(r3, 0);
                __m256i _r4 = __lasx_xvld(r4, 0);
                __m256i _r5 = __lasx_xvld(r5, 0);

                __m256i _tmp02a = __lasx_xvadd_w(_r1, _r2);
                __m256i _tmp02b = __lasx_xvadd_w(_r3, _r4);
                __m256i _tmp13a = __lasx_xvsub_w(_r1, _r2);
                __m256i _tmp13b = __lasx_xvsub_w(_r3, _r4);

                __m256i _tmp0 = __lasx_xvadd_w(__lasx_xvadd_w(_tmp02a, _tmp02b), _r0);
                __m256i _tmp1 = __lasx_xvadd_w(_tmp13a, __lasx_xvslli_w(_tmp13b, 1));
                __m256i _tmp2 = __lasx_xvadd_w(_tmp02a, __lasx_xvslli_w(_tmp02b, 2));
                __m256i _tmp3 = __lasx_xvadd_w(__lasx_xvadd_w(_tmp13a, __lasx_xvslli_w(_tmp13b, 3)), __lasx_xvslli_w(_r5, 2));

                _tmp0 = __lasx_xvslli_w(_tmp0, 2);
                _tmp1 = __lasx_xvslli_w(_tmp1, 2);
                _tmp2 = __lasx_xvslli_w(_tmp2, 2);
                _tmp3 = __lasx_xvslli_w(_tmp3, 2);

                __lasx_xvst(_tmp0, (int*)tmp[0][m], 0);
                __lasx_xvst(_tmp1, (int*)tmp[1][m], 0);
                __lasx_xvst(_tmp2, (int*)tmp[2][m], 0);
                __lasx_xvst(_tmp3, (int*)tmp[3][m], 0);

                r0 += max_jj * 6 * 8;
                r1 += max_jj * 6 * 8;
                r2 += max_jj * 6 * 8;
                r3 += max_jj * 6 * 8;
                r4 += max_jj * 6 * 8;
                r5 += max_jj * 6 * 8;
            }

            int* outptr0 = top_blob.channel((i + ii) / out_elempack).row<int>(ti * 4) + (tj * 4) * out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                __m256i _r0 = __lasx_xvld((const __m256i*)tmp[m][0], 0);
                __m256i _r1 = __lasx_xvld((const __m256i*)tmp[m][1], 0);
                __m256i _r2 = __lasx_xvld((const __m256i*)tmp[m][2], 0);
                __m256i _r3 = __lasx_xvld((const __m256i*)tmp[m][3], 0);
                __m256i _r4 = __lasx_xvld((const __m256i*)tmp[m][4], 0);
                __m256i _r5 = __lasx_xvld((const __m256i*)tmp[m][5], 0);

                __m256i _tmp02a = __lasx_xvadd_w(_r1, _r2);
                __m256i _tmp02b = __lasx_xvadd_w(_r3, _r4);
                __m256i _tmp13a = __lasx_xvsub_w(_r1, _r2);
                __m256i _tmp13b = __lasx_xvsub_w(_r3, _r4);

                __m256i _tmp0 = __lasx_xvadd_w(__lasx_xvadd_w(_tmp02a, _tmp02b), _r0);
                __m256i _tmp1 = __lasx_xvadd_w(_tmp13a, __lasx_xvslli_w(_tmp13b, 1));
                __m256i _tmp2 = __lasx_xvadd_w(_tmp02a, __lasx_xvslli_w(_tmp02b, 2));
                __m256i _tmp3 = __lasx_xvadd_w(__lasx_xvadd_w(_tmp13a, __lasx_xvslli_w(_tmp13b, 3)), _r5);

                // TODO use integer trick for division by 576
                __m256 _v576 = (__m256)__lasx_xvreplfr2vr_s(1.0 / 576);
                _tmp0 = __lasx_xvftintrz_w_s((__m256)__lasx_xvfmul_s((__m256)__lasx_xvffint_s_w(_tmp0), _v576));
                _tmp1 = __lasx_xvftintrz_w_s((__m256)__lasx_xvfmul_s((__m256)__lasx_xvffint_s_w(_tmp1), _v576));
                _tmp2 = __lasx_xvftintrz_w_s((__m256)__lasx_xvfmul_s((__m256)__lasx_xvffint_s_w(_tmp2), _v576));
                _tmp3 = __lasx_xvftintrz_w_s((__m256)__lasx_xvfmul_s((__m256)__lasx_xvffint_s_w(_tmp3), _v576));

                if (out_elempack == 8)
                {
                    __lasx_xvst(_tmp0, outptr0, 0);
                    if (tj * 4 + 1 < outw) __lasx_xvst(_tmp1, outptr0 + 8, 0);
                    if (tj * 4 + 2 < outw) __lasx_xvst(_tmp2, outptr0 + 16, 0);
                    if (tj * 4 + 3 < outw) __lasx_xvst(_tmp3, outptr0 + 24, 0);
                }
                if (out_elempack == 4)
                {
                    int* outptr1 = outptr0 + N;

                    __lsx_vst(__lasx_extract_lo128(_tmp0), outptr0, 0);
                    __lsx_vst((__m128i)__lasx_xvpermi_q(_tmp0, _tmp0, 0x11), outptr1, 0);
                    if (tj * 4 + 1 < outw)
                    {
                        __lsx_vst(__lasx_extract_lo128(_tmp1), outptr0 + 4, 0);
                        __lsx_vst((__m128i)__lasx_xvpermi_q(_tmp1, _tmp1, 0x11), outptr1 + 4, 0);
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        __lsx_vst(__lasx_extract_lo128(_tmp2), outptr0 + 8, 0);
                        __lsx_vst((__m128i)__lasx_xvpermi_q(_tmp2, _tmp2, 0x11), outptr1 + 8, 0);
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        __lsx_vst(__lasx_extract_lo128(_tmp3), outptr0 + 12, 0);
                        __lsx_vst((__m128i)__lasx_xvpermi_q(_tmp3, _tmp3, 0x11), outptr1 + 12, 0);
                    }
                }
                if (out_elempack == 1)
                {
                    int tmp0[8];
                    int tmp1[8];
                    int tmp2[8];
                    int tmp3[8];
                    __lasx_xvst(_tmp0, tmp0, 0);
                    __lasx_xvst(_tmp1, tmp1, 0);
                    __lasx_xvst(_tmp2, tmp2, 0);
                    __lasx_xvst(_tmp3, tmp3, 0);

                    int* outptr1 = outptr0 + N * 1;
                    int* outptr2 = outptr0 + N * 2;
                    int* outptr3 = outptr0 + N * 3;
                    int* outptr4 = outptr0 + N * 4;
                    int* outptr5 = outptr0 + N * 5;
                    int* outptr6 = outptr0 + N * 6;
                    int* outptr7 = outptr0 + N * 7;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    outptr4[0] = tmp0[4];
                    outptr5[0] = tmp0[5];
                    outptr6[0] = tmp0[6];
                    outptr7[0] = tmp0[7];
                    if (tj * 4 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                        outptr4[1] = tmp1[4];
                        outptr5[1] = tmp1[5];
                        outptr6[1] = tmp1[6];
                        outptr7[1] = tmp1[7];
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        outptr0[2] = tmp2[0];
                        outptr1[2] = tmp2[1];
                        outptr2[2] = tmp2[2];
                        outptr3[2] = tmp2[3];
                        outptr4[2] = tmp2[4];
                        outptr5[2] = tmp2[5];
                        outptr6[2] = tmp2[6];
                        outptr7[2] = tmp2[7];
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        outptr0[3] = tmp3[0];
                        outptr1[3] = tmp3[1];
                        outptr2[3] = tmp3[2];
                        outptr3[3] = tmp3[3];
                        outptr4[3] = tmp3[4];
                        outptr5[3] = tmp3[5];
                        outptr6[3] = tmp3[6];
                        outptr7[3] = tmp3[7];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __loongarch_asx
#if __loongarch_sx
    for (; ii + 3 < max_ii; ii += 4)
    {
        __attribute__((aligned(16))) int tmp[4][6][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 36 + jj * 4;
            const int* r1 = r0 + max_jj * 4;
            const int* r2 = r0 + max_jj * 4 * 2;
            const int* r3 = r0 + max_jj * 4 * 3;
            const int* r4 = r0 + max_jj * 4 * 4;
            const int* r5 = r0 + max_jj * 4 * 5;

            for (int m = 0; m < 5; m++)
            {
                __m128i _r0 = __lsx_vld(r0, 0);
                __m128i _r1 = __lsx_vld(r1, 0);
                __m128i _r2 = __lsx_vld(r2, 0);
                __m128i _r3 = __lsx_vld(r3, 0);
                __m128i _r4 = __lsx_vld(r4, 0);
                __m128i _r5 = __lsx_vld(r5, 0);

                __m128i _tmp02a = __lsx_vadd_w(_r1, _r2);
                __m128i _tmp02b = __lsx_vadd_w(_r3, _r4);
                __m128i _tmp13a = __lsx_vsub_w(_r1, _r2);
                __m128i _tmp13b = __lsx_vsub_w(_r3, _r4);

                __m128i _tmp0 = __lsx_vadd_w(__lsx_vadd_w(_tmp02a, _tmp02b), _r0);
                __m128i _tmp1 = __lsx_vadd_w(_tmp13a, __lsx_vslli_w(_tmp13b, 1));
                __m128i _tmp2 = __lsx_vadd_w(_tmp02a, __lsx_vslli_w(_tmp02b, 2));
                __m128i _tmp3 = __lsx_vadd_w(__lsx_vadd_w(_tmp13a, __lsx_vslli_w(_tmp13b, 3)), __lsx_vslli_w(_r5, 2));

                __lsx_vst(_tmp0, tmp[0][m], 0);
                __lsx_vst(_tmp1, tmp[1][m], 0);
                __lsx_vst(_tmp2, tmp[2][m], 0);
                __lsx_vst(_tmp3, tmp[3][m], 0);

                r0 += max_jj * 6 * 4;
                r1 += max_jj * 6 * 4;
                r2 += max_jj * 6 * 4;
                r3 += max_jj * 6 * 4;
                r4 += max_jj * 6 * 4;
                r5 += max_jj * 6 * 4;
            }
            for (int m = 5; m < 6; m++)
            {
                __m128i _r0 = __lsx_vld(r0, 0);
                __m128i _r1 = __lsx_vld(r1, 0);
                __m128i _r2 = __lsx_vld(r2, 0);
                __m128i _r3 = __lsx_vld(r3, 0);
                __m128i _r4 = __lsx_vld(r4, 0);
                __m128i _r5 = __lsx_vld(r5, 0);

                __m128i _tmp02a = __lsx_vadd_w(_r1, _r2);
                __m128i _tmp02b = __lsx_vadd_w(_r3, _r4);
                __m128i _tmp13a = __lsx_vsub_w(_r1, _r2);
                __m128i _tmp13b = __lsx_vsub_w(_r3, _r4);

                __m128i _tmp0 = __lsx_vadd_w(__lsx_vadd_w(_tmp02a, _tmp02b), _r0);
                __m128i _tmp1 = __lsx_vadd_w(_tmp13a, __lsx_vslli_w(_tmp13b, 1));
                __m128i _tmp2 = __lsx_vadd_w(_tmp02a, __lsx_vslli_w(_tmp02b, 2));
                __m128i _tmp3 = __lsx_vadd_w(__lsx_vadd_w(_tmp13a, __lsx_vslli_w(_tmp13b, 3)), __lsx_vslli_w(_r5, 2));

                _tmp0 = __lsx_vslli_w(_tmp0, 2);
                _tmp1 = __lsx_vslli_w(_tmp1, 2);
                _tmp2 = __lsx_vslli_w(_tmp2, 2);
                _tmp3 = __lsx_vslli_w(_tmp3, 2);

                __lsx_vst(_tmp0, tmp[0][m], 0);
                __lsx_vst(_tmp1, tmp[1][m], 0);
                __lsx_vst(_tmp2, tmp[2][m], 0);
                __lsx_vst(_tmp3, tmp[3][m], 0);

                r0 += max_jj * 6 * 4;
                r1 += max_jj * 6 * 4;
                r2 += max_jj * 6 * 4;
                r3 += max_jj * 6 * 4;
                r4 += max_jj * 6 * 4;
                r5 += max_jj * 6 * 4;
            }

            int* outptr0 = top_blob.channel((i + ii) / out_elempack).row<int>(ti * 4) + (tj * 4) * out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                __m128i _r0 = __lsx_vld(tmp[m][0], 0);
                __m128i _r1 = __lsx_vld(tmp[m][1], 0);
                __m128i _r2 = __lsx_vld(tmp[m][2], 0);
                __m128i _r3 = __lsx_vld(tmp[m][3], 0);
                __m128i _r4 = __lsx_vld(tmp[m][4], 0);
                __m128i _r5 = __lsx_vld(tmp[m][5], 0);

                __m128i _tmp02a = __lsx_vadd_w(_r1, _r2);
                __m128i _tmp02b = __lsx_vadd_w(_r3, _r4);
                __m128i _tmp13a = __lsx_vsub_w(_r1, _r2);
                __m128i _tmp13b = __lsx_vsub_w(_r3, _r4);

                __m128i _tmp0 = __lsx_vadd_w(__lsx_vadd_w(_tmp02a, _tmp02b), _r0);
                __m128i _tmp1 = __lsx_vadd_w(_tmp13a, __lsx_vslli_w(_tmp13b, 1));
                __m128i _tmp2 = __lsx_vadd_w(_tmp02a, __lsx_vslli_w(_tmp02b, 2));
                __m128i _tmp3 = __lsx_vadd_w(__lsx_vadd_w(_tmp13a, __lsx_vslli_w(_tmp13b, 3)), _r5);

                // TODO use integer trick for division by 576
                float _1_576 = 1.0f / 576;
                __m128i _v576 = (__m128i)__lsx_vreplfr2vr_s(_1_576);
                _tmp0 = (__m128i)__lsx_vftintrz_w_s(__lsx_vfmul_s((__m128)__lsx_vffint_s_w(_tmp0), (__m128)_v576));
                _tmp1 = (__m128i)__lsx_vftintrz_w_s(__lsx_vfmul_s((__m128)__lsx_vffint_s_w(_tmp1), (__m128)_v576));
                _tmp2 = (__m128i)__lsx_vftintrz_w_s(__lsx_vfmul_s((__m128)__lsx_vffint_s_w(_tmp2), (__m128)_v576));
                _tmp3 = (__m128i)__lsx_vftintrz_w_s(__lsx_vfmul_s((__m128)__lsx_vffint_s_w(_tmp3), (__m128)_v576));

                if (out_elempack == 4)
                {
                    __lsx_vst(_tmp0, outptr0, 0);
                    if (tj * 4 + 1 < outw) __lsx_vst(_tmp1, (outptr0 + 4), 0);
                    if (tj * 4 + 2 < outw) __lsx_vst(_tmp2, (outptr0 + 8), 0);
                    if (tj * 4 + 3 < outw) __lsx_vst(_tmp3, (outptr0 + 12), 0);
                }
                if (out_elempack == 1)
                {
                    int tmp0[4];
                    int tmp1[4];
                    int tmp2[4];
                    int tmp3[4];
                    __lsx_vst(_tmp0, tmp0, 0);
                    __lsx_vst(_tmp1, tmp1, 0);
                    __lsx_vst(_tmp2, tmp2, 0);
                    __lsx_vst(_tmp3, tmp3, 0);

                    int* outptr1 = outptr0 + N;
                    int* outptr2 = outptr0 + N * 2;
                    int* outptr3 = outptr0 + N * 3;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    if (tj * 4 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        outptr0[2] = tmp2[0];
                        outptr1[2] = tmp2[1];
                        outptr2[2] = tmp2[2];
                        outptr3[2] = tmp2[3];
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        outptr0[3] = tmp3[0];
                        outptr1[3] = tmp3[1];
                        outptr2[3] = tmp3[2];
                        outptr3[3] = tmp3[3];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        int tmp[4][6][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 36 + jj * 2;
            const int* r1 = r0 + max_jj * 2;
            const int* r2 = r0 + max_jj * 2 * 2;
            const int* r3 = r0 + max_jj * 2 * 3;
            const int* r4 = r0 + max_jj * 2 * 4;
            const int* r5 = r0 + max_jj * 2 * 5;

            for (int m = 0; m < 5; m++)
            {
                int tmp02a0 = r1[0] + r2[0];
                int tmp02a1 = r1[1] + r2[1];
                int tmp02b0 = r3[0] + r4[0];
                int tmp02b1 = r3[1] + r4[1];
                int tmp13a0 = r1[0] - r2[0];
                int tmp13a1 = r1[1] - r2[1];
                int tmp13b0 = r3[0] - r4[0];
                int tmp13b1 = r3[1] - r4[1];

                int tmp00 = tmp02a0 + tmp02b0 + r0[0];
                int tmp01 = tmp02a1 + tmp02b1 + r0[1];
                int tmp10 = tmp13a0 + tmp13b0 * 2;
                int tmp11 = tmp13a1 + tmp13b1 * 2;
                int tmp20 = tmp02a0 + tmp02b0 * 4;
                int tmp21 = tmp02a1 + tmp02b1 * 4;
                int tmp30 = tmp13a0 + tmp13b0 * 8 + r5[0] * 4;
                int tmp31 = tmp13a1 + tmp13b1 * 8 + r5[1] * 4;

                tmp[0][m][0] = tmp00;
                tmp[0][m][1] = tmp01;
                tmp[1][m][0] = tmp10;
                tmp[1][m][1] = tmp11;
                tmp[2][m][0] = tmp20;
                tmp[2][m][1] = tmp21;
                tmp[3][m][0] = tmp30;
                tmp[3][m][1] = tmp31;

                r0 += max_jj * 6 * 2;
                r1 += max_jj * 6 * 2;
                r2 += max_jj * 6 * 2;
                r3 += max_jj * 6 * 2;
                r4 += max_jj * 6 * 2;
                r5 += max_jj * 6 * 2;
            }
            for (int m = 5; m < 6; m++)
            {
                int tmp02a0 = r1[0] + r2[0];
                int tmp02a1 = r1[1] + r2[1];
                int tmp02b0 = r3[0] + r4[0];
                int tmp02b1 = r3[1] + r4[1];
                int tmp13a0 = r1[0] - r2[0];
                int tmp13a1 = r1[1] - r2[1];
                int tmp13b0 = r3[0] - r4[0];
                int tmp13b1 = r3[1] - r4[1];

                int tmp00 = tmp02a0 + tmp02b0 + r0[0];
                int tmp01 = tmp02a1 + tmp02b1 + r0[1];
                int tmp10 = tmp13a0 + tmp13b0 * 2;
                int tmp11 = tmp13a1 + tmp13b1 * 2;
                int tmp20 = tmp02a0 + tmp02b0 * 4;
                int tmp21 = tmp02a1 + tmp02b1 * 4;
                int tmp30 = tmp13a0 + tmp13b0 * 8 + r5[0] * 4;
                int tmp31 = tmp13a1 + tmp13b1 * 8 + r5[1] * 4;

                tmp00 = tmp00 * 4;
                tmp01 = tmp01 * 4;
                tmp10 = tmp10 * 4;
                tmp11 = tmp11 * 4;
                tmp20 = tmp20 * 4;
                tmp21 = tmp21 * 4;
                tmp30 = tmp30 * 4;
                tmp31 = tmp31 * 4;

                tmp[0][m][0] = tmp00;
                tmp[0][m][1] = tmp01;
                tmp[1][m][0] = tmp10;
                tmp[1][m][1] = tmp11;
                tmp[2][m][0] = tmp20;
                tmp[2][m][1] = tmp21;
                tmp[3][m][0] = tmp30;
                tmp[3][m][1] = tmp31;

                r0 += max_jj * 6 * 2;
                r1 += max_jj * 6 * 2;
                r2 += max_jj * 6 * 2;
                r3 += max_jj * 6 * 2;
                r4 += max_jj * 6 * 2;
                r5 += max_jj * 6 * 2;
            }

            int* outptr0 = top_blob.channel(i + ii).row<int>(ti * 4) + (tj * 4);

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                int tmp02a0 = tmp[m][1][0] + tmp[m][2][0];
                int tmp02a1 = tmp[m][1][1] + tmp[m][2][1];
                int tmp02b0 = tmp[m][3][0] + tmp[m][4][0];
                int tmp02b1 = tmp[m][3][1] + tmp[m][4][1];
                int tmp13a0 = tmp[m][1][0] - tmp[m][2][0];
                int tmp13a1 = tmp[m][1][1] - tmp[m][2][1];
                int tmp13b0 = tmp[m][3][0] - tmp[m][4][0];
                int tmp13b1 = tmp[m][3][1] - tmp[m][4][1];

                int tmp00 = tmp02a0 + tmp02b0 + tmp[m][0][0];
                int tmp01 = tmp02a1 + tmp02b1 + tmp[m][0][1];
                int tmp10 = tmp13a0 + tmp13b0 * 2;
                int tmp11 = tmp13a1 + tmp13b1 * 2;
                int tmp20 = tmp02a0 + tmp02b0 * 4;
                int tmp21 = tmp02a1 + tmp02b1 * 4;
                int tmp30 = tmp13a0 + tmp13b0 * 8 + tmp[m][5][0];
                int tmp31 = tmp13a1 + tmp13b1 * 8 + tmp[m][5][1];

                tmp00 = tmp00 / 576;
                tmp01 = tmp01 / 576;
                tmp10 = tmp10 / 576;
                tmp11 = tmp11 / 576;
                tmp20 = tmp20 / 576;
                tmp21 = tmp21 / 576;
                tmp30 = tmp30 / 576;
                tmp31 = tmp31 / 576;

                // if (out_elempack == 1)
                {
                    int* outptr1 = outptr0 + N;

                    outptr0[0] = tmp00;
                    outptr1[0] = tmp01;
                    if (tj * 4 + 1 < outw)
                    {
                        outptr0[1] = tmp10;
                        outptr1[1] = tmp11;
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        outptr0[2] = tmp20;
                        outptr1[2] = tmp21;
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        outptr0[3] = tmp30;
                        outptr1[3] = tmp31;
                    }
                }

                outptr0 += outw;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        int tmp[4][6];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 36 + jj;
            const int* r1 = r0 + max_jj;
            const int* r2 = r0 + max_jj * 2;
            const int* r3 = r0 + max_jj * 3;
            const int* r4 = r0 + max_jj * 4;
            const int* r5 = r0 + max_jj * 5;

            for (int m = 0; m < 5; m++)
            {
                int tmp02a = r1[0] + r2[0];
                int tmp02b = r3[0] + r4[0];
                int tmp13a = r1[0] - r2[0];
                int tmp13b = r3[0] - r4[0];

                int tmp0 = tmp02a + tmp02b + r0[0];
                int tmp1 = tmp13a + tmp13b * 2;
                int tmp2 = tmp02a + tmp02b * 4;
                int tmp3 = tmp13a + tmp13b * 8 + r5[0] * 4;

                tmp[0][m] = tmp0;
                tmp[1][m] = tmp1;
                tmp[2][m] = tmp2;
                tmp[3][m] = tmp3;

                r0 += max_jj * 6;
                r1 += max_jj * 6;
                r2 += max_jj * 6;
                r3 += max_jj * 6;
                r4 += max_jj * 6;
                r5 += max_jj * 6;
            }
            for (int m = 5; m < 6; m++)
            {
                int tmp02a = r1[0] + r2[0];
                int tmp02b = r3[0] + r4[0];
                int tmp13a = r1[0] - r2[0];
                int tmp13b = r3[0] - r4[0];

                int tmp0 = tmp02a + tmp02b + r0[0];
                int tmp1 = tmp13a + tmp13b * 2;
                int tmp2 = tmp02a + tmp02b * 4;
                int tmp3 = tmp13a + tmp13b * 8 + r5[0] * 4;

                tmp0 = tmp0 * 4;
                tmp1 = tmp1 * 4;
                tmp2 = tmp2 * 4;
                tmp3 = tmp3 * 4;

                tmp[0][m] = tmp0;
                tmp[1][m] = tmp1;
                tmp[2][m] = tmp2;
                tmp[3][m] = tmp3;

                r0 += max_jj * 6;
                r1 += max_jj * 6;
                r2 += max_jj * 6;
                r3 += max_jj * 6;
                r4 += max_jj * 6;
                r5 += max_jj * 6;
            }

            int* outptr0 = top_blob.channel(i + ii).row<int>(ti * 4) + (tj * 4);

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                int tmp02a = tmp[m][1] + tmp[m][2];
                int tmp02b = tmp[m][3] + tmp[m][4];
                int tmp13a = tmp[m][1] - tmp[m][2];
                int tmp13b = tmp[m][3] - tmp[m][4];

                int tmp0 = tmp02a + tmp02b + tmp[m][0];
                int tmp1 = tmp13a + tmp13b * 2;
                int tmp2 = tmp02a + tmp02b * 4;
                int tmp3 = tmp13a + tmp13b * 8 + tmp[m][5];

                tmp0 = tmp0 / 576;
                tmp1 = tmp1 / 576;
                tmp2 = tmp2 / 576;
                tmp3 = tmp3 / 576;

                // if (out_elempack == 1)
                {
                    outptr0[0] = tmp0;
                    if (tj * 4 + 1 < outw) outptr0[1] = tmp1;
                    if (tj * 4 + 2 < outw) outptr0[2] = tmp2;
                    if (tj * 4 + 3 < outw) outptr0[3] = tmp3;
                }

                outptr0 += outw;
            }
        }
    }
}

static int conv3x3s1_winograd43_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt)
{
    int outw = top_blob.w;
    int outh = top_blob.h;

    // pad to 4n+2, winograd F(4,3)
    int w_tiles = (outw + 3) / 4;
    int h_tiles = (outh + 3) / 4;
    int tiles = w_tiles * h_tiles;

    const int M = top_blob.c * top_blob.elempack;
    const int N = tiles;
    const int K = bottom_blob.c * bottom_blob.elempack;
    const int B = 36;

    // NCNN_LOGE("conv3x3s1_winograd43_int8 %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;

    if (nT > 1 && nn_NK < nT)
    {
        Mat B_tile(TILE_N * B * TILE_K, 4u, opt.workspace_allocator);
        if (B_tile.empty())
            return -100;

        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            // transform input
            conv3x3s1_winograd43_transform_input_tile_int8(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile_int8(B_tile, BT_tile, B, max_jj, max_kk, nT);
        }
    }
    else
    {
        Mat B_tileX(TILE_N * B * TILE_K, 1, nT, 4u, opt.workspace_allocator);
        if (B_tileX.empty())
            return -100;

        #pragma omp parallel for num_threads(nT)
        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            Mat B_tile = B_tileX.channel(get_omp_thread_num());

            // transform input
            conv3x3s1_winograd43_transform_input_tile_int8(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile_int8(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (top_tileX.empty())
        return -100;

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat top_tile = top_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

                const Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

                bool k_end = k + TILE_K >= K;

                gemm_transB_packed_tile_int8(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk, k_end);
            }

            // transform output
            conv3x3s1_winograd43_transform_output_tile_int8(top_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

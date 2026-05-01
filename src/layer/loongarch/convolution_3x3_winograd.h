// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pack_A_tile(const Mat& A, Mat& AT, int batch, int max_ii, int max_kk)
{
    const int N = max_kk * batch;

    for (int b = 0; b < batch; b++)
    {
        float* pp = AT.row(b);

        int ii = 0;
#if __loongarch_sx
        for (; ii + 7 < max_ii; ii += 8)
        {
            const float* p0 = (const float*)A + ii * N + b;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[N];
                pp[2] = p0[2 * N];
                pp[3] = p0[3 * N];
                pp[4] = p0[4 * N];
                pp[5] = p0[5 * N];
                pp[6] = p0[6 * N];
                pp[7] = p0[7 * N];
                p0 += batch;
                pp += 8;
            }
        }
        for (; ii + 3 < max_ii; ii += 4)
        {
            const float* p0 = (const float*)A + ii * N + b;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[N];
                pp[2] = p0[2 * N];
                pp[3] = p0[3 * N];
                p0 += batch;
                pp += 4;
            }
        }
#endif // __loongarch_sx
        for (; ii + 1 < max_ii; ii += 2)
        {
            const float* p0 = (const float*)A + ii * N + b;

            int kk = 0;
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
            const float* p0 = (const float*)A + ii * N + b;

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

static void transpose_pack_B_tile(const Mat& B, Mat& BT, int batch, int max_jj, int max_kk, int nT)
{
    #pragma omp parallel for num_threads(nT)
    for (int b = 0; b < batch; b++)
    {
        float* pp = BT.row(b);

        int jj = 0;
#if __loongarch_sx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const float* p0 = B;

            int kk = 0;
#if __loongarch_asx
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = (__m256)__lasx_xvld(p0, 0);
                __m256 _r1 = (__m256)__lasx_xvld(p0 + 8, 0);
                __m256 _r2 = (__m256)__lasx_xvld(p0 + 8 * 2, 0);
                __m256 _r3 = (__m256)__lasx_xvld(p0 + 8 * 3, 0);
                __m256 _r4 = (__m256)__lasx_xvld(p0 + 8 * 4, 0);
                __m256 _r5 = (__m256)__lasx_xvld(p0 + 8 * 5, 0);
                __m256 _r6 = (__m256)__lasx_xvld(p0 + 8 * 6, 0);
                __m256 _r7 = (__m256)__lasx_xvld(p0 + 8 * 7, 0);
                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                __lasx_xvst((__m256i)_r0, pp, 0);
                __lasx_xvst((__m256i)_r1, pp + 8, 0);
                __lasx_xvst((__m256i)_r2, pp + 8 * 2, 0);
                __lasx_xvst((__m256i)_r3, pp + 8 * 3, 0);
                __lasx_xvst((__m256i)_r4, pp + 8 * 4, 0);
                __lasx_xvst((__m256i)_r5, pp + 8 * 5, 0);
                __lasx_xvst((__m256i)_r6, pp + 8 * 6, 0);
                __lasx_xvst((__m256i)_r7, pp + 8 * 7, 0);
                p0 += max_jj * batch * 8;
                pp += 64;
            }
            p0 -= (b * max_jj + jj) * 8;
#endif // __loongarch_asx
            p0 += (b * max_jj + jj) * 4;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                __m128 _r2 = (__m128)__lsx_vld(p0 + 4 * 2, 0);
                __m128 _r3 = (__m128)__lsx_vld(p0 + 4 * 3, 0);
                __m128 _r4 = (__m128)__lsx_vld(p0 + 4 * 4, 0);
                __m128 _r5 = (__m128)__lsx_vld(p0 + 4 * 5, 0);
                __m128 _r6 = (__m128)__lsx_vld(p0 + 4 * 6, 0);
                __m128 _r7 = (__m128)__lsx_vld(p0 + 4 * 7, 0);
                {
                    __m128i _01l = __lsx_vilvl_w((__m128i)_r1, (__m128i)_r0);
                    __m128i _01h = __lsx_vilvh_w((__m128i)_r1, (__m128i)_r0);
                    __m128i _23l = __lsx_vilvl_w((__m128i)_r3, (__m128i)_r2);
                    __m128i _23h = __lsx_vilvh_w((__m128i)_r3, (__m128i)_r2);
                    _r0 = (__m128)__lsx_vilvl_d(_23l, _01l);
                    _r1 = (__m128)__lsx_vilvh_d(_23l, _01l);
                    _r2 = (__m128)__lsx_vilvl_d(_23h, _01h);
                    _r3 = (__m128)__lsx_vilvh_d(_23h, _01h);
                }
                {
                    __m128i _01l = __lsx_vilvl_w((__m128i)_r5, (__m128i)_r4);
                    __m128i _01h = __lsx_vilvh_w((__m128i)_r5, (__m128i)_r4);
                    __m128i _23l = __lsx_vilvl_w((__m128i)_r7, (__m128i)_r6);
                    __m128i _23h = __lsx_vilvh_w((__m128i)_r7, (__m128i)_r6);
                    _r4 = (__m128)__lsx_vilvl_d(_23l, _01l);
                    _r5 = (__m128)__lsx_vilvh_d(_23l, _01l);
                    _r6 = (__m128)__lsx_vilvl_d(_23h, _01h);
                    _r7 = (__m128)__lsx_vilvh_d(_23h, _01h);
                }
                __lsx_vst((__m128i)_r0, pp, 0);
                __lsx_vst((__m128i)_r4, pp + 4, 0);
                __lsx_vst((__m128i)_r1, pp + 8, 0);
                __lsx_vst((__m128i)_r5, pp + 12, 0);
                __lsx_vst((__m128i)_r2, pp + 16, 0);
                __lsx_vst((__m128i)_r6, pp + 20, 0);
                __lsx_vst((__m128i)_r3, pp + 24, 0);
                __lsx_vst((__m128i)_r7, pp + 28, 0);
                p0 += max_jj * batch * 4;
                pp += 32;
            }
            p0 -= (b * max_jj + jj) * 4;
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[2];
                pp[2] = p0[4];
                pp[3] = p0[6];
                pp[4] = p0[8];
                pp[5] = p0[10];
                pp[6] = p0[12];
                pp[7] = p0[14];
                pp[8] = p0[1];
                pp[9] = p0[3];
                pp[10] = p0[5];
                pp[11] = p0[7];
                pp[12] = p0[9];
                pp[13] = p0[11];
                pp[14] = p0[13];
                pp[15] = p0[15];
                p0 += max_jj * batch * 2;
                pp += 16;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p0[4];
                pp[5] = p0[5];
                pp[6] = p0[6];
                pp[7] = p0[7];
                p0 += max_jj * batch;
                pp += 8;
            }
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const float* p0 = B;

            int kk = 0;
#if __loongarch_asx
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = (__m256)__lasx_xvld(p0, 0);
                __m256 _r1 = (__m256)__lasx_xvld(p0 + 8, 0);
                __m256 _r2 = (__m256)__lasx_xvld(p0 + 8 * 2, 0);
                __m256 _r3 = (__m256)__lasx_xvld(p0 + 8 * 3, 0);
                transpose8x4_ps(_r0, _r1, _r2, _r3);
                __lasx_xvst((__m256i)_r0, pp, 0);
                __lasx_xvst((__m256i)_r1, pp + 8, 0);
                __lasx_xvst((__m256i)_r2, pp + 8 * 2, 0);
                __lasx_xvst((__m256i)_r3, pp + 8 * 3, 0);
                p0 += max_jj * batch * 8;
                pp += 32;
            }
            p0 -= (b * max_jj + jj) * 8;
#endif // __loongarch_asx
            p0 += (b * max_jj + jj) * 4;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                __m128 _r2 = (__m128)__lsx_vld(p0 + 4 * 2, 0);
                __m128 _r3 = (__m128)__lsx_vld(p0 + 4 * 3, 0);
                {
                    __m128i _01l = __lsx_vilvl_w((__m128i)_r1, (__m128i)_r0);
                    __m128i _01h = __lsx_vilvh_w((__m128i)_r1, (__m128i)_r0);
                    __m128i _23l = __lsx_vilvl_w((__m128i)_r3, (__m128i)_r2);
                    __m128i _23h = __lsx_vilvh_w((__m128i)_r3, (__m128i)_r2);
                    _r0 = (__m128)__lsx_vilvl_d(_23l, _01l);
                    _r1 = (__m128)__lsx_vilvh_d(_23l, _01l);
                    _r2 = (__m128)__lsx_vilvl_d(_23h, _01h);
                    _r3 = (__m128)__lsx_vilvh_d(_23h, _01h);
                }
                __lsx_vst((__m128i)_r0, pp, 0);
                __lsx_vst((__m128i)_r1, pp + 4, 0);
                __lsx_vst((__m128i)_r2, pp + 8, 0);
                __lsx_vst((__m128i)_r3, pp + 12, 0);
                p0 += max_jj * batch * 4;
                pp += 16;
            }
            p0 -= (b * max_jj + jj) * 4;
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[2];
                pp[2] = p0[4];
                pp[3] = p0[6];
                pp[4] = p0[1];
                pp[5] = p0[3];
                pp[6] = p0[5];
                pp[7] = p0[7];
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
            const float* p0 = B;

            int kk = 0;
#if __loongarch_sx
#if __loongarch_asx
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = (__m256)__lasx_xvld(p0, 0);
                __m256 _r1 = (__m256)__lasx_xvld(p0 + 8, 0);
                transpose8x2_ps(_r0, _r1);
                __lasx_xvst((__m256i)_r0, pp, 0);
                __lasx_xvst((__m256i)_r1, pp + 8, 0);
                p0 += max_jj * batch * 8;
                pp += 16;
            }
            p0 -= (b * max_jj + jj) * 8;
#endif // __loongarch_asx
            p0 += (b * max_jj + jj) * 4;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_r1, (__m128i)_r0);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_r1, (__m128i)_r0);
                __lsx_vst((__m128i)_tmp0, pp, 0);
                __lsx_vst((__m128i)_tmp1, pp + 4, 0);
                p0 += max_jj * batch * 4;
                pp += 8;
            }
            p0 -= (b * max_jj + jj) * 4;
#endif // __loongarch_sx
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[2];
                pp[2] = p0[1];
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
            const float* p0 = B;

            int kk = 0;
#if __loongarch_sx
#if __loongarch_asx
            p0 += (b * max_jj + jj) * 8;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = (__m256)__lasx_xvld(p0, 0);
                __lasx_xvst((__m256i)_r0, pp, 0);
                p0 += max_jj * batch * 8;
                pp += 8;
            }
            p0 -= (b * max_jj + jj) * 8;
#endif // __loongarch_asx
            p0 += (b * max_jj + jj) * 4;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                __lsx_vst((__m128i)_r0, pp, 0);
                p0 += max_jj * batch * 4;
                pp += 4;
            }
            p0 -= (b * max_jj + jj) * 4;
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

static void gemm_transB_packed_tile(const Mat& AT_tile, const Mat& BT_tile, Mat& top_blob, int batch, int max_ii, int max_jj, int k, int max_kk)
{
    float* outptr = top_blob;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        for (int b = 0; b < batch; b++)
        {
            const float* pAT = AT_tile.row(b) + max_kk * ii;
            const float* pB = BT_tile.row(b);

            int jj = 0;
#if __loongarch_asx
            for (; jj + 7 < max_jj; jj += 8)
            {
                const float* pA = pAT;

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
                    _sum0 = (__m256)__lasx_xvldi(0);
                    _sum1 = (__m256)__lasx_xvldi(0);
                    _sum2 = (__m256)__lasx_xvldi(0);
                    _sum3 = (__m256)__lasx_xvldi(0);
                    _sum4 = (__m256)__lasx_xvldi(0);
                    _sum5 = (__m256)__lasx_xvldi(0);
                    _sum6 = (__m256)__lasx_xvldi(0);
                    _sum7 = (__m256)__lasx_xvldi(0);
                }
                else
                {
                    _sum0 = (__m256)__lasx_xvld(outptr, 0);
                    _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
                    _sum2 = (__m256)__lasx_xvld(outptr + 16, 0);
                    _sum3 = (__m256)__lasx_xvld(outptr + 24, 0);
                    _sum4 = (__m256)__lasx_xvld(outptr + 32, 0);
                    _sum5 = (__m256)__lasx_xvld(outptr + 40, 0);
                    _sum6 = (__m256)__lasx_xvld(outptr + 48, 0);
                    _sum7 = (__m256)__lasx_xvld(outptr + 56, 0);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m256 _pA = (__m256)__lasx_xvld(pA, 0);
                    _sum0 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[0]), _sum0);
                    _sum1 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[1]), _sum1);
                    _sum2 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[2]), _sum2);
                    _sum3 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[3]), _sum3);
                    _sum4 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[4]), _sum4);
                    _sum5 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[5]), _sum5);
                    _sum6 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[6]), _sum6);
                    _sum7 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[7]), _sum7);

                    pA += 8;
                    pB += 8;
                }

                __lasx_xvst((__m256i)_sum0, outptr, 0);
                __lasx_xvst((__m256i)_sum1, outptr + 8, 0);
                __lasx_xvst((__m256i)_sum2, outptr + 8 * 2, 0);
                __lasx_xvst((__m256i)_sum3, outptr + 8 * 3, 0);
                __lasx_xvst((__m256i)_sum4, outptr + 8 * 4, 0);
                __lasx_xvst((__m256i)_sum5, outptr + 8 * 5, 0);
                __lasx_xvst((__m256i)_sum6, outptr + 8 * 6, 0);
                __lasx_xvst((__m256i)_sum7, outptr + 8 * 7, 0);
                outptr += 8 * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const float* pA = pAT;

                __m256 _sum0;
                __m256 _sum1;
                __m256 _sum2;
                __m256 _sum3;

                if (k == 0)
                {
                    _sum0 = (__m256)__lasx_xvldi(0);
                    _sum1 = (__m256)__lasx_xvldi(0);
                    _sum2 = (__m256)__lasx_xvldi(0);
                    _sum3 = (__m256)__lasx_xvldi(0);
                }
                else
                {
                    _sum0 = (__m256)__lasx_xvld(outptr, 0);
                    _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
                    _sum2 = (__m256)__lasx_xvld(outptr + 16, 0);
                    _sum3 = (__m256)__lasx_xvld(outptr + 24, 0);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m256 _pA = (__m256)__lasx_xvld(pA, 0);
                    _sum0 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[0]), _sum0);
                    _sum1 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[1]), _sum1);
                    _sum2 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[2]), _sum2);
                    _sum3 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[3]), _sum3);

                    pA += 8;
                    pB += 4;
                }

                __lasx_xvst((__m256i)_sum0, outptr, 0);
                __lasx_xvst((__m256i)_sum1, outptr + 8, 0);
                __lasx_xvst((__m256i)_sum2, outptr + 8 * 2, 0);
                __lasx_xvst((__m256i)_sum3, outptr + 8 * 3, 0);
                outptr += 8 * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const float* pA = pAT;

                __m256 _sum0;
                __m256 _sum1;

                if (k == 0)
                {
                    _sum0 = (__m256)__lasx_xvldi(0);
                    _sum1 = (__m256)__lasx_xvldi(0);
                }
                else
                {
                    _sum0 = (__m256)__lasx_xvld(outptr, 0);
                    _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m256 _pA = (__m256)__lasx_xvld(pA, 0);
                    _sum0 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[0]), _sum0);
                    _sum1 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[1]), _sum1);

                    pA += 8;
                    pB += 2;
                }

                __lasx_xvst((__m256i)_sum0, outptr, 0);
                __lasx_xvst((__m256i)_sum1, outptr + 8, 0);
                outptr += 8 * 2;
            }
            for (; jj < max_jj; jj++)
            {
                const float* pA = pAT;

                __m256 _sum;

                if (k == 0)
                {
                    _sum = (__m256)__lasx_xvldi(0);
                }
                else
                {
                    _sum = (__m256)__lasx_xvld(outptr, 0);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m256 _pA = (__m256)__lasx_xvld(pA, 0);
                    __m256 _pB = (__m256)__lasx_xvreplfr2vr_s(pB[0]);
                    _sum = __lasx_xvfmadd_s(_pA, _pB, _sum);

                    pA += 8;
                    pB += 1;
                }

                __lasx_xvst((__m256i)_sum, outptr, 0);
                outptr += 8;
            }
#else // __loongarch_asx
            float* outptr0 = outptr + b * max_jj * 4;
            float* outptr1 = outptr + (batch + b) * max_jj * 4;

            for (; jj + 7 < max_jj; jj += 8)
            {
                const float* pA = pAT;

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
                __m128 _sum12;
                __m128 _sum13;
                __m128 _sum14;
                __m128 _sum15;

                if (k == 0)
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
                    _sum12 = (__m128)__lsx_vldi(0);
                    _sum13 = (__m128)__lsx_vldi(0);
                    _sum14 = (__m128)__lsx_vldi(0);
                    _sum15 = (__m128)__lsx_vldi(0);
                }
                else
                {
                    _sum0 = (__m128)__lsx_vld(outptr0, 0);
                    _sum1 = (__m128)__lsx_vld(outptr0 + 4, 0);
                    _sum2 = (__m128)__lsx_vld(outptr0 + 8, 0);
                    _sum3 = (__m128)__lsx_vld(outptr0 + 12, 0);
                    _sum4 = (__m128)__lsx_vld(outptr0 + 16, 0);
                    _sum5 = (__m128)__lsx_vld(outptr0 + 20, 0);
                    _sum6 = (__m128)__lsx_vld(outptr0 + 24, 0);
                    _sum7 = (__m128)__lsx_vld(outptr0 + 28, 0);
                    _sum8 = (__m128)__lsx_vld(outptr1, 0);
                    _sum9 = (__m128)__lsx_vld(outptr1 + 4, 0);
                    _sum10 = (__m128)__lsx_vld(outptr1 + 8, 0);
                    _sum11 = (__m128)__lsx_vld(outptr1 + 12, 0);
                    _sum12 = (__m128)__lsx_vld(outptr1 + 16, 0);
                    _sum13 = (__m128)__lsx_vld(outptr1 + 20, 0);
                    _sum14 = (__m128)__lsx_vld(outptr1 + 24, 0);
                    _sum15 = (__m128)__lsx_vld(outptr1 + 28, 0);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA0 = (__m128)__lsx_vld(pA, 0);
                    __m128 _pA1 = (__m128)__lsx_vld(pA + 4, 0);
                    __m128 _pB0 = (__m128)__lsx_vreplfr2vr_s(pB[0]);
                    __m128 _pB1 = (__m128)__lsx_vreplfr2vr_s(pB[1]);
                    __m128 _pB2 = (__m128)__lsx_vreplfr2vr_s(pB[2]);
                    __m128 _pB3 = (__m128)__lsx_vreplfr2vr_s(pB[3]);
                    __m128 _pB4 = (__m128)__lsx_vreplfr2vr_s(pB[4]);
                    __m128 _pB5 = (__m128)__lsx_vreplfr2vr_s(pB[5]);
                    __m128 _pB6 = (__m128)__lsx_vreplfr2vr_s(pB[6]);
                    __m128 _pB7 = (__m128)__lsx_vreplfr2vr_s(pB[7]);
                    _sum0 = __lsx_vfmadd_s(_pA0, _pB0, _sum0);
                    _sum1 = __lsx_vfmadd_s(_pA0, _pB1, _sum1);
                    _sum2 = __lsx_vfmadd_s(_pA0, _pB2, _sum2);
                    _sum3 = __lsx_vfmadd_s(_pA0, _pB3, _sum3);
                    _sum4 = __lsx_vfmadd_s(_pA0, _pB4, _sum4);
                    _sum5 = __lsx_vfmadd_s(_pA0, _pB5, _sum5);
                    _sum6 = __lsx_vfmadd_s(_pA0, _pB6, _sum6);
                    _sum7 = __lsx_vfmadd_s(_pA0, _pB7, _sum7);
                    _sum8 = __lsx_vfmadd_s(_pA1, _pB0, _sum8);
                    _sum9 = __lsx_vfmadd_s(_pA1, _pB1, _sum9);
                    _sum10 = __lsx_vfmadd_s(_pA1, _pB2, _sum10);
                    _sum11 = __lsx_vfmadd_s(_pA1, _pB3, _sum11);
                    _sum12 = __lsx_vfmadd_s(_pA1, _pB4, _sum12);
                    _sum13 = __lsx_vfmadd_s(_pA1, _pB5, _sum13);
                    _sum14 = __lsx_vfmadd_s(_pA1, _pB6, _sum14);
                    _sum15 = __lsx_vfmadd_s(_pA1, _pB7, _sum15);

                    pA += 8;
                    pB += 8;
                }

                __lsx_vst((__m128i)_sum0, outptr0, 0);
                __lsx_vst((__m128i)_sum1, outptr0 + 4, 0);
                __lsx_vst((__m128i)_sum2, outptr0 + 4 * 2, 0);
                __lsx_vst((__m128i)_sum3, outptr0 + 4 * 3, 0);
                __lsx_vst((__m128i)_sum4, outptr0 + 4 * 4, 0);
                __lsx_vst((__m128i)_sum5, outptr0 + 4 * 5, 0);
                __lsx_vst((__m128i)_sum6, outptr0 + 4 * 6, 0);
                __lsx_vst((__m128i)_sum7, outptr0 + 4 * 7, 0);
                __lsx_vst((__m128i)_sum8, outptr1, 0);
                __lsx_vst((__m128i)_sum9, outptr1 + 4, 0);
                __lsx_vst((__m128i)_sum10, outptr1 + 4 * 2, 0);
                __lsx_vst((__m128i)_sum11, outptr1 + 4 * 3, 0);
                __lsx_vst((__m128i)_sum12, outptr1 + 4 * 4, 0);
                __lsx_vst((__m128i)_sum13, outptr1 + 4 * 5, 0);
                __lsx_vst((__m128i)_sum14, outptr1 + 4 * 6, 0);
                __lsx_vst((__m128i)_sum15, outptr1 + 4 * 7, 0);
                outptr0 += 4 * 8;
                outptr1 += 4 * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const float* pA = pAT;

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
                    _sum0 = (__m128)__lsx_vldi(0);
                    _sum1 = (__m128)__lsx_vldi(0);
                    _sum2 = (__m128)__lsx_vldi(0);
                    _sum3 = (__m128)__lsx_vldi(0);
                    _sum4 = (__m128)__lsx_vldi(0);
                    _sum5 = (__m128)__lsx_vldi(0);
                    _sum6 = (__m128)__lsx_vldi(0);
                    _sum7 = (__m128)__lsx_vldi(0);
                }
                else
                {
                    _sum0 = (__m128)__lsx_vld(outptr0, 0);
                    _sum1 = (__m128)__lsx_vld(outptr0 + 4, 0);
                    _sum2 = (__m128)__lsx_vld(outptr0 + 8, 0);
                    _sum3 = (__m128)__lsx_vld(outptr0 + 12, 0);
                    _sum4 = (__m128)__lsx_vld(outptr1, 0);
                    _sum5 = (__m128)__lsx_vld(outptr1 + 4, 0);
                    _sum6 = (__m128)__lsx_vld(outptr1 + 8, 0);
                    _sum7 = (__m128)__lsx_vld(outptr1 + 12, 0);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA0 = (__m128)__lsx_vld(pA, 0);
                    __m128 _pA1 = (__m128)__lsx_vld(pA + 4, 0);
                    __m128 _pB0 = (__m128)__lsx_vreplfr2vr_s(pB[0]);
                    __m128 _pB1 = (__m128)__lsx_vreplfr2vr_s(pB[1]);
                    __m128 _pB2 = (__m128)__lsx_vreplfr2vr_s(pB[2]);
                    __m128 _pB3 = (__m128)__lsx_vreplfr2vr_s(pB[3]);
                    _sum0 = __lsx_vfmadd_s(_pA0, _pB0, _sum0);
                    _sum1 = __lsx_vfmadd_s(_pA0, _pB1, _sum1);
                    _sum2 = __lsx_vfmadd_s(_pA0, _pB2, _sum2);
                    _sum3 = __lsx_vfmadd_s(_pA0, _pB3, _sum3);
                    _sum4 = __lsx_vfmadd_s(_pA1, _pB0, _sum4);
                    _sum5 = __lsx_vfmadd_s(_pA1, _pB1, _sum5);
                    _sum6 = __lsx_vfmadd_s(_pA1, _pB2, _sum6);
                    _sum7 = __lsx_vfmadd_s(_pA1, _pB3, _sum7);

                    pA += 8;
                    pB += 4;
                }

                __lsx_vst((__m128i)_sum0, outptr0, 0);
                __lsx_vst((__m128i)_sum1, outptr0 + 4, 0);
                __lsx_vst((__m128i)_sum2, outptr0 + 4 * 2, 0);
                __lsx_vst((__m128i)_sum3, outptr0 + 4 * 3, 0);
                __lsx_vst((__m128i)_sum4, outptr1, 0);
                __lsx_vst((__m128i)_sum5, outptr1 + 4, 0);
                __lsx_vst((__m128i)_sum6, outptr1 + 4 * 2, 0);
                __lsx_vst((__m128i)_sum7, outptr1 + 4 * 3, 0);
                outptr0 += 4 * 4;
                outptr1 += 4 * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const float* pA = pAT;

                __m128 _sum0;
                __m128 _sum1;
                __m128 _sum2;
                __m128 _sum3;

                if (k == 0)
                {
                    _sum0 = (__m128)__lsx_vldi(0);
                    _sum1 = (__m128)__lsx_vldi(0);
                    _sum2 = (__m128)__lsx_vldi(0);
                    _sum3 = (__m128)__lsx_vldi(0);
                }
                else
                {
                    _sum0 = (__m128)__lsx_vld(outptr0, 0);
                    _sum1 = (__m128)__lsx_vld(outptr0 + 4, 0);
                    _sum2 = (__m128)__lsx_vld(outptr1, 0);
                    _sum3 = (__m128)__lsx_vld(outptr1 + 4, 0);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA0 = (__m128)__lsx_vld(pA, 0);
                    __m128 _pA1 = (__m128)__lsx_vld(pA + 4, 0);
                    __m128 _pB0 = (__m128)__lsx_vreplfr2vr_s(pB[0]);
                    __m128 _pB1 = (__m128)__lsx_vreplfr2vr_s(pB[1]);
                    _sum0 = __lsx_vfmadd_s(_pA0, _pB0, _sum0);
                    _sum1 = __lsx_vfmadd_s(_pA0, _pB1, _sum1);
                    _sum2 = __lsx_vfmadd_s(_pA1, _pB0, _sum2);
                    _sum3 = __lsx_vfmadd_s(_pA1, _pB1, _sum3);

                    pA += 8;
                    pB += 2;
                }

                __lsx_vst((__m128i)_sum0, outptr0, 0);
                __lsx_vst((__m128i)_sum1, outptr0 + 4, 0);
                __lsx_vst((__m128i)_sum2, outptr1, 0);
                __lsx_vst((__m128i)_sum3, outptr1 + 4, 0);
                outptr0 += 4 * 2;
                outptr1 += 4 * 2;
            }
            for (; jj < max_jj; jj++)
            {
                const float* pA = pAT;

                __m128 _sum0;
                __m128 _sum1;

                if (k == 0)
                {
                    _sum0 = (__m128)__lsx_vldi(0);
                    _sum1 = (__m128)__lsx_vldi(0);
                }
                else
                {
                    _sum0 = (__m128)__lsx_vld(outptr0, 0);
                    _sum1 = (__m128)__lsx_vld(outptr1, 0);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA0 = (__m128)__lsx_vld(pA, 0);
                    __m128 _pA1 = (__m128)__lsx_vld(pA + 4, 0);
                    __m128 _pB = (__m128)__lsx_vreplfr2vr_s(pB[0]);
                    _sum0 = __lsx_vfmadd_s(_pA0, _pB, _sum0);
                    _sum1 = __lsx_vfmadd_s(_pA1, _pB, _sum1);

                    pA += 8;
                    pB += 1;
                }

                __lsx_vst((__m128i)_sum0, outptr0, 0);
                __lsx_vst((__m128i)_sum1, outptr1, 0);
                outptr0 += 4;
                outptr1 += 4;
            }
#endif // __loongarch_asx
        }
#if !__loongarch_asx
        outptr += max_jj * batch * 8;
#endif
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        for (int b = 0; b < batch; b++)
        {
            const float* pAT = AT_tile.row(b) + max_kk * ii;
            const float* pB = BT_tile.row(b);

            int jj = 0;
            for (; jj + 7 < max_jj; jj += 8)
            {
                const float* pA = pAT;

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
                    _sum0 = (__m128)__lsx_vldi(0);
                    _sum1 = (__m128)__lsx_vldi(0);
                    _sum2 = (__m128)__lsx_vldi(0);
                    _sum3 = (__m128)__lsx_vldi(0);
                    _sum4 = (__m128)__lsx_vldi(0);
                    _sum5 = (__m128)__lsx_vldi(0);
                    _sum6 = (__m128)__lsx_vldi(0);
                    _sum7 = (__m128)__lsx_vldi(0);
                }
                else
                {
                    _sum0 = (__m128)__lsx_vld(outptr, 0);
                    _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
                    _sum2 = (__m128)__lsx_vld(outptr + 8, 0);
                    _sum3 = (__m128)__lsx_vld(outptr + 12, 0);
                    _sum4 = (__m128)__lsx_vld(outptr + 16, 0);
                    _sum5 = (__m128)__lsx_vld(outptr + 20, 0);
                    _sum6 = (__m128)__lsx_vld(outptr + 24, 0);
                    _sum7 = (__m128)__lsx_vld(outptr + 28, 0);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA = (__m128)__lsx_vld(pA, 0);
                    _sum0 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[0]), _sum0);
                    _sum1 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[1]), _sum1);
                    _sum2 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[2]), _sum2);
                    _sum3 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[3]), _sum3);
                    _sum4 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[4]), _sum4);
                    _sum5 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[5]), _sum5);
                    _sum6 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[6]), _sum6);
                    _sum7 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[7]), _sum7);

                    pA += 4;
                    pB += 8;
                }

                __lsx_vst((__m128i)_sum0, outptr, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);
                __lsx_vst((__m128i)_sum2, outptr + 4 * 2, 0);
                __lsx_vst((__m128i)_sum3, outptr + 4 * 3, 0);
                __lsx_vst((__m128i)_sum4, outptr + 4 * 4, 0);
                __lsx_vst((__m128i)_sum5, outptr + 4 * 5, 0);
                __lsx_vst((__m128i)_sum6, outptr + 4 * 6, 0);
                __lsx_vst((__m128i)_sum7, outptr + 4 * 7, 0);
                outptr += 4 * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const float* pA = pAT;

                __m128 _sum0;
                __m128 _sum1;
                __m128 _sum2;
                __m128 _sum3;

                if (k == 0)
                {
                    _sum0 = (__m128)__lsx_vldi(0);
                    _sum1 = (__m128)__lsx_vldi(0);
                    _sum2 = (__m128)__lsx_vldi(0);
                    _sum3 = (__m128)__lsx_vldi(0);
                }
                else
                {
                    _sum0 = (__m128)__lsx_vld(outptr, 0);
                    _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
                    _sum2 = (__m128)__lsx_vld(outptr + 8, 0);
                    _sum3 = (__m128)__lsx_vld(outptr + 12, 0);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA = (__m128)__lsx_vld(pA, 0);
                    _sum0 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[0]), _sum0);
                    _sum1 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[1]), _sum1);
                    _sum2 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[2]), _sum2);
                    _sum3 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[3]), _sum3);

                    pA += 4;
                    pB += 4;
                }

                __lsx_vst((__m128i)_sum0, outptr, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);
                __lsx_vst((__m128i)_sum2, outptr + 4 * 2, 0);
                __lsx_vst((__m128i)_sum3, outptr + 4 * 3, 0);
                outptr += 4 * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const float* pA = pAT;

                __m128 _sum0;
                __m128 _sum1;

                if (k == 0)
                {
                    _sum0 = (__m128)__lsx_vldi(0);
                    _sum1 = (__m128)__lsx_vldi(0);
                }
                else
                {
                    _sum0 = (__m128)__lsx_vld(outptr, 0);
                    _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA = (__m128)__lsx_vld(pA, 0);
                    _sum0 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[0]), _sum0);
                    _sum1 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[1]), _sum1);

                    pA += 4;
                    pB += 2;
                }

                __lsx_vst((__m128i)_sum0, outptr, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);
                outptr += 4 * 2;
            }
            for (; jj < max_jj; jj++)
            {
                const float* pA = pAT;

                __m128 _sum;

                if (k == 0)
                {
                    _sum = (__m128)__lsx_vldi(0);
                }
                else
                {
                    _sum = (__m128)__lsx_vld(outptr, 0);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA = (__m128)__lsx_vld(pA, 0);
                    _sum = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[0]), _sum);

                    pA += 4;
                    pB += 1;
                }

                __lsx_vst((__m128i)_sum, outptr, 0);
                outptr += 4;
            }
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        for (int b = 0; b < batch; b++)
        {
            const float* pAT = AT_tile.row(b) + max_kk * ii;
            const float* pB = BT_tile.row(b);

            int jj = 0;
#if __loongarch_sx
            for (; jj + 7 < max_jj; jj += 8)
            {
                const float* pA = pAT;

                __m128 _sum0;
                __m128 _sum1;
                __m128 _sum2;
                __m128 _sum3;

                if (k == 0)
                {
                    _sum0 = (__m128)__lsx_vldi(0);
                    _sum1 = (__m128)__lsx_vldi(0);
                    _sum2 = (__m128)__lsx_vldi(0);
                    _sum3 = (__m128)__lsx_vldi(0);
                }
                else
                {
                    __m128 _tmp0 = (__m128)__lsx_vld(outptr, 0);
                    __m128 _tmp1 = (__m128)__lsx_vld(outptr + 4, 0);
                    __m128 _tmp2 = (__m128)__lsx_vld(outptr + 8, 0);
                    __m128 _tmp3 = (__m128)__lsx_vld(outptr + 12, 0);
                    _sum0 = (__m128)__lsx_vpickev_w((__m128i)_tmp1, (__m128i)_tmp0);
                    _sum1 = (__m128)__lsx_vpickev_w((__m128i)_tmp3, (__m128i)_tmp2);
                    _sum2 = (__m128)__lsx_vpickod_w((__m128i)_tmp1, (__m128i)_tmp0);
                    _sum3 = (__m128)__lsx_vpickod_w((__m128i)_tmp3, (__m128i)_tmp2);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA0 = (__m128)__lsx_vreplfr2vr_s(pA[0]);
                    __m128 _pA1 = (__m128)__lsx_vreplfr2vr_s(pA[1]);
                    __m128 _pB0 = (__m128)__lsx_vld(pB, 0);
                    __m128 _pB1 = (__m128)__lsx_vld(pB + 4, 0);
                    _sum0 = __lsx_vfmadd_s(_pA0, _pB0, _sum0);
                    _sum1 = __lsx_vfmadd_s(_pA0, _pB1, _sum1);
                    _sum2 = __lsx_vfmadd_s(_pA1, _pB0, _sum2);
                    _sum3 = __lsx_vfmadd_s(_pA1, _pB1, _sum3);
                    pA += 2;
                    pB += 8;
                }

                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_sum2, (__m128i)_sum0);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_sum2, (__m128i)_sum0);
                __m128 _tmp2 = (__m128)__lsx_vilvl_w((__m128i)_sum3, (__m128i)_sum1);
                __m128 _tmp3 = (__m128)__lsx_vilvh_w((__m128i)_sum3, (__m128i)_sum1);
                __lsx_vst((__m128i)_tmp0, outptr, 0);
                __lsx_vst((__m128i)_tmp1, outptr + 4, 0);
                __lsx_vst((__m128i)_tmp2, outptr + 8, 0);
                __lsx_vst((__m128i)_tmp3, outptr + 12, 0);
                outptr += 2 * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const float* pA = pAT;

                __m128 _sum0;
                __m128 _sum1;

                if (k == 0)
                {
                    _sum0 = (__m128)__lsx_vldi(0);
                    _sum1 = (__m128)__lsx_vldi(0);
                }
                else
                {
                    __m128 _tmp0 = (__m128)__lsx_vld(outptr, 0);
                    __m128 _tmp1 = (__m128)__lsx_vld(outptr + 4, 0);
                    _sum0 = (__m128)__lsx_vpickev_w((__m128i)_tmp1, (__m128i)_tmp0);
                    _sum1 = (__m128)__lsx_vpickod_w((__m128i)_tmp1, (__m128i)_tmp0);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pB = (__m128)__lsx_vld(pB, 0);
                    _sum0 = __lsx_vfmadd_s((__m128)__lsx_vreplfr2vr_s(pA[0]), _pB, _sum0);
                    _sum1 = __lsx_vfmadd_s((__m128)__lsx_vreplfr2vr_s(pA[1]), _pB, _sum1);
                    pA += 2;
                    pB += 4;
                }

                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_sum1, (__m128i)_sum0);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_sum1, (__m128i)_sum0);
                __lsx_vst((__m128i)_tmp0, outptr, 0);
                __lsx_vst((__m128i)_tmp1, outptr + 4, 0);
                outptr += 2 * 4;
            }
#endif // __loongarch_sx
            for (; jj + 1 < max_jj; jj += 2)
            {
                const float* pA = pAT;

                float sum00 = 0.f;
                float sum01 = 0.f;
                float sum10 = 0.f;
                float sum11 = 0.f;

                if (k == 0)
                {
                    sum00 = 0.f;
                    sum01 = 0.f;
                    sum10 = 0.f;
                    sum11 = 0.f;
                }
                else
                {
                    sum00 = outptr[0];
                    sum01 = outptr[1];
                    sum10 = outptr[2];
                    sum11 = outptr[3];
                }

                int kk = 0;
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
                const float* pA = pAT;

                float sum0 = 0.f;
                float sum1 = 0.f;

                if (k == 0)
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                }
                else
                {
                    sum0 = outptr[0];
                    sum1 = outptr[1];
                }

                int kk = 0;
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
            const float* pAT = AT_tile.row(b) + max_kk * ii;
            const float* pB = BT_tile.row(b);

            int jj = 0;
#if __loongarch_sx
            for (; jj + 7 < max_jj; jj += 8)
            {
                const float* pA = pAT;

                __m128 _sum0;
                __m128 _sum1;

                if (k == 0)
                {
                    _sum0 = (__m128)__lsx_vldi(0);
                    _sum1 = (__m128)__lsx_vldi(0);
                }
                else
                {
                    _sum0 = (__m128)__lsx_vld(outptr, 0);
                    _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA = (__m128)__lsx_vreplfr2vr_s(pA[0]);
                    __m128 _pB0 = (__m128)__lsx_vld(pB, 0);
                    __m128 _pB1 = (__m128)__lsx_vld(pB + 4, 0);
                    _sum0 = __lsx_vfmadd_s(_pA, _pB0, _sum0);
                    _sum1 = __lsx_vfmadd_s(_pA, _pB1, _sum1);
                    pA += 1;
                    pB += 8;
                }

                __lsx_vst((__m128i)_sum0, outptr, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);
                outptr += 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const float* pA = pAT;

                __m128 _sum;

                if (k == 0)
                {
                    _sum = (__m128)__lsx_vldi(0);
                }
                else
                {
                    _sum = (__m128)__lsx_vld(outptr, 0);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA = (__m128)__lsx_vreplfr2vr_s(pA[0]);
                    __m128 _pB = (__m128)__lsx_vld(pB, 0);
                    _sum = __lsx_vfmadd_s(_pA, _pB, _sum);
                    pA += 1;
                    pB += 4;
                }

                __lsx_vst((__m128i)_sum, outptr, 0);
                outptr += 4;
            }
#endif // __loongarch_sx
            for (; jj + 1 < max_jj; jj += 2)
            {
                const float* pA = pAT;

                float sum0 = 0.f;
                float sum1 = 0.f;

                if (k == 0)
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                }
                else
                {
                    sum0 = outptr[0];
                    sum1 = outptr[1];
                }

                int kk = 0;
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
                const float* pA = pAT;

                float sum = 0.f;

                if (k == 0)
                {
                    sum = 0.f;
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

static void get_optimal_tile_mnk(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve M
    {
        int tile_size = (int)sqrt((float)l2_cache_size / sizeof(float) / 3);

#if __loongarch_sx
        TILE_M = std::max(8, tile_size / 8 * 8);
#else
        TILE_M = std::max(2, tile_size / 2 * 2);
#endif

        TILE_M *= std::min(nT, get_physical_cpu_count());

        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __loongarch_sx
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif

        if (nT > 1)
        {
#if __loongarch_sx
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#else
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
        }
    }

    // solve K
    {
        int tile_size = (int)(sqrt((float)l2_cache_size / sizeof(float)) - TILE_M);

#if __loongarch_sx
        TILE_K = std::max(8, tile_size / 8 * 8);
#else
        TILE_K = std::max(2, tile_size / 2 * 2);
#endif

        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __loongarch_sx
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif
    }

    if (N > 0)
    {
        int tile_size = (int)(((float)l2_cache_size / sizeof(float) - TILE_M * TILE_K) / (TILE_M + TILE_K));

#if __loongarch_sx
        TILE_N = std::max(8, tile_size / 8 * 8);
#else
        TILE_N = std::max(1, tile_size);
#endif

        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __loongarch_sx
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif
    }
}

static inline void conv3x3s1_winograd23_transform_kernel_tile(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    // const float ktm[4][3] = {
    //     {1.0f, 0.0f, 0.0f},
    //     {1.0f / 2, 1.0f / 2, 1.0f / 2},
    //     {1.0f / 2, -1.0f / 2, 1.0f / 2},
    //     {0.0f, 0.0f, 1.0f}
    // };

    float* ptmp = A;

    int ii = 0;
    for (; ii < max_ii; ii++)
    {
        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            float tmp[4][3];

            const float* k0 = (const float*)kernel + (i + ii) * inch * 9 + (k + kk) * 9;

            for (int m = 0; m < 3; m++)
            {
                float r0 = k0[0];
                float r1 = k0[1];
                float r2 = k0[2];

                tmp[0][m] = r0;
                tmp[1][m] = r0 * 0.5f + r1 * 0.5f + r2 * 0.5f;
                tmp[2][m] = r0 * 0.5f - r1 * 0.5f + r2 * 0.5f;
                tmp[3][m] = r2;

                k0 += 3;
            }

            for (int m = 0; m < 4; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];

                float z0 = r0;
                float z1 = r0 * 0.5f + r1 * 0.5f + r2 * 0.5f;
                float z2 = r0 * 0.5f - r1 * 0.5f + r2 * 0.5f;
                float z3 = r2;

                ptmp[0] = z0;
                ptmp[1] = z1;
                ptmp[2] = z2;
                ptmp[3] = z3;
                ptmp += 4;
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_kernel(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 16;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

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

            conv3x3s1_winograd23_transform_kernel_tile(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            pack_A_tile(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd23_transform_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    // const float itm[4][4] = {
    //     {1.0f,  0.0f, -1.0f,  0.0f},
    //     {0.0f,  1.0f,  1.00f, 0.0f},
    //     {0.0f, -1.0f,  1.00f, 0.0f},
    //     {0.0f, -1.0f,  0.00f, 1.0f}
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
    nn_max_kk = max_kk / 8;
    #pragma omp parallel for num_threads(nT)
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 8;

        __attribute__((aligned(32))) float tmp[4][4][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 2) + (tj * 2) * elempack;

            for (int m = 0; m < 4; m++)
            {
                __m256 _r0 = (__m256)__lasx_xvldi(0);
                __m256 _r1 = (__m256)__lasx_xvldi(0);
                __m256 _r2 = (__m256)__lasx_xvldi(0);
                __m256 _r3 = (__m256)__lasx_xvldi(0);

                if (ti * 2 + m < h)
                {
                    if (elempack == 8)
                    {
                        _r0 = (__m256)__lasx_xvld(r0, 0);
                        if (tj * 2 + 1 < w) _r1 = (__m256)__lasx_xvld(r0 + 8, 0);
                        if (tj * 2 + 2 < w) _r2 = (__m256)__lasx_xvld(r0 + 16, 0);
                        if (tj * 2 + 3 < w) _r3 = (__m256)__lasx_xvld(r0 + 24, 0);
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;

                        _r0 = (__m256)__lasx_concat_128((__m128i)(__m128)__lsx_vld(r0, 0), (__m128i)(__m128)__lsx_vld(r1, 0));
                        if (tj * 2 + 1 < w) _r1 = (__m256)__lasx_concat_128((__m128i)(__m128)__lsx_vld(r0 + 4, 0), (__m128i)(__m128)__lsx_vld(r1 + 4, 0));
                        if (tj * 2 + 2 < w) _r2 = (__m256)__lasx_concat_128((__m128i)(__m128)__lsx_vld(r0 + 8, 0), (__m128i)(__m128)__lsx_vld(r1 + 8, 0));
                        if (tj * 2 + 3 < w) _r3 = (__m256)__lasx_concat_128((__m128i)(__m128)__lsx_vld(r0 + 12, 0), (__m128i)(__m128)__lsx_vld(r1 + 12, 0));
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;
                        const float* r4 = r0 + N * 4;
                        const float* r5 = r0 + N * 5;
                        const float* r6 = r0 + N * 6;
                        const float* r7 = r0 + N * 7;

                        __m128 _t0 = (__m128)__lsx_vld(r0, 0);
                        __m128 _t1 = (__m128)__lsx_vld(r1, 0);
                        __m128 _t2 = (__m128)__lsx_vld(r2, 0);
                        __m128 _t3 = (__m128)__lsx_vld(r3, 0);
                        __m128 _t4 = (__m128)__lsx_vld(r4, 0);
                        __m128 _t5 = (__m128)__lsx_vld(r5, 0);
                        __m128 _t6 = (__m128)__lsx_vld(r6, 0);
                        __m128 _t7 = (__m128)__lsx_vld(r7, 0);

                        {
                            __m128i _01l = __lsx_vilvl_w((__m128i)_t1, (__m128i)_t0);
                            __m128i _01h = __lsx_vilvh_w((__m128i)_t1, (__m128i)_t0);
                            __m128i _23l = __lsx_vilvl_w((__m128i)_t3, (__m128i)_t2);
                            __m128i _23h = __lsx_vilvh_w((__m128i)_t3, (__m128i)_t2);
                            _t0 = (__m128)__lsx_vilvl_d(_23l, _01l);
                            _t1 = (__m128)__lsx_vilvh_d(_23l, _01l);
                            _t2 = (__m128)__lsx_vilvl_d(_23h, _01h);
                            _t3 = (__m128)__lsx_vilvh_d(_23h, _01h);
                        }
                        {
                            __m128i _01l = __lsx_vilvl_w((__m128i)_t5, (__m128i)_t4);
                            __m128i _01h = __lsx_vilvh_w((__m128i)_t5, (__m128i)_t4);
                            __m128i _23l = __lsx_vilvl_w((__m128i)_t7, (__m128i)_t6);
                            __m128i _23h = __lsx_vilvh_w((__m128i)_t7, (__m128i)_t6);
                            _t4 = (__m128)__lsx_vilvl_d(_23l, _01l);
                            _t5 = (__m128)__lsx_vilvh_d(_23l, _01l);
                            _t6 = (__m128)__lsx_vilvl_d(_23h, _01h);
                            _t7 = (__m128)__lsx_vilvh_d(_23h, _01h);
                        }

                        _r0 = (__m256)__lasx_concat_128((__m128i)_t0, (__m128i)_t4);
                        if (tj * 2 + 1 < w) _r1 = (__m256)__lasx_concat_128((__m128i)_t1, (__m128i)_t5);
                        if (tj * 2 + 2 < w) _r2 = (__m256)__lasx_concat_128((__m128i)_t2, (__m128i)_t6);
                        if (tj * 2 + 3 < w) _r3 = (__m256)__lasx_concat_128((__m128i)_t3, (__m128i)_t7);
                    }
                }

                __m256 _tmp0 = __lasx_xvfsub_s(_r0, _r2);
                __m256 _tmp1 = __lasx_xvfadd_s(_r1, _r2);
                __m256 _tmp2 = __lasx_xvfsub_s(_r2, _r1);
                __m256 _tmp3 = __lasx_xvfsub_s(_r3, _r1);

                __lasx_xvst((__m256i)_tmp0, tmp[0][m], 0);
                __lasx_xvst((__m256i)_tmp1, tmp[1][m], 0);
                __lasx_xvst((__m256i)_tmp2, tmp[2][m], 0);
                __lasx_xvst((__m256i)_tmp3, tmp[3][m], 0);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 16 + jj * 8;
            float* p1 = p0 + max_jj * 8;
            float* p2 = p0 + max_jj * 8 * 2;
            float* p3 = p0 + max_jj * 8 * 3;

            for (int m = 0; m < 4; m++)
            {
                __m256 _r0 = (__m256)__lasx_xvld(tmp[m][0], 0);
                __m256 _r1 = (__m256)__lasx_xvld(tmp[m][1], 0);
                __m256 _r2 = (__m256)__lasx_xvld(tmp[m][2], 0);
                __m256 _r3 = (__m256)__lasx_xvld(tmp[m][3], 0);

                __m256 _tmp0 = __lasx_xvfsub_s(_r0, _r2);
                __m256 _tmp1 = __lasx_xvfadd_s(_r1, _r2);
                __m256 _tmp2 = __lasx_xvfsub_s(_r2, _r1);
                __m256 _tmp3 = __lasx_xvfsub_s(_r3, _r1);

                __lasx_xvst((__m256i)_tmp0, p0, 0);
                __lasx_xvst((__m256i)_tmp1, p1, 0);
                __lasx_xvst((__m256i)_tmp2, p2, 0);
                __lasx_xvst((__m256i)_tmp3, p3, 0);

                p0 += max_jj * 4 * 8;
                p1 += max_jj * 4 * 8;
                p2 += max_jj * 4 * 8;
                p3 += max_jj * 4 * 8;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 8;
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
    #pragma omp parallel for num_threads(nT)
#else // __loongarch_asx
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
    #pragma omp parallel for num_threads(nT)
#endif // __loongarch_asx
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 4;

        __attribute__((aligned(16))) float tmp[4][4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 2) + (tj * 2) * elempack;

            for (int m = 0; m < 4; m++)
            {
                __m128 _r0 = (__m128)__lsx_vldi(0);
                __m128 _r1 = (__m128)__lsx_vldi(0);
                __m128 _r2 = (__m128)__lsx_vldi(0);
                __m128 _r3 = (__m128)__lsx_vldi(0);

                if (ti * 2 + m < h)
                {
                    if (elempack == 4)
                    {
                        _r0 = (__m128)__lsx_vld(r0, 0);
                        if (tj * 2 + 1 < w) _r1 = (__m128)__lsx_vld(r0 + 4, 0);
                        if (tj * 2 + 2 < w) _r2 = (__m128)__lsx_vld(r0 + 8, 0);
                        if (tj * 2 + 3 < w) _r3 = (__m128)__lsx_vld(r0 + 12, 0);
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;

                        __m128 _t0 = (__m128)__lsx_vld(r0, 0);
                        __m128 _t1 = (__m128)__lsx_vld(r1, 0);
                        __m128 _t2 = (__m128)__lsx_vld(r2, 0);
                        __m128 _t3 = (__m128)__lsx_vld(r3, 0);

                        {
                            __m128i _01l = __lsx_vilvl_w((__m128i)_t1, (__m128i)_t0);
                            __m128i _01h = __lsx_vilvh_w((__m128i)_t1, (__m128i)_t0);
                            __m128i _23l = __lsx_vilvl_w((__m128i)_t3, (__m128i)_t2);
                            __m128i _23h = __lsx_vilvh_w((__m128i)_t3, (__m128i)_t2);
                            _t0 = (__m128)__lsx_vilvl_d(_23l, _01l);
                            _t1 = (__m128)__lsx_vilvh_d(_23l, _01l);
                            _t2 = (__m128)__lsx_vilvl_d(_23h, _01h);
                            _t3 = (__m128)__lsx_vilvh_d(_23h, _01h);
                        }

                        _r0 = _t0;
                        if (tj * 2 + 1 < w) _r1 = _t1;
                        if (tj * 2 + 2 < w) _r2 = _t2;
                        if (tj * 2 + 3 < w) _r3 = _t3;
                    }
                }

                __m128 _tmp0 = __lsx_vfsub_s(_r0, _r2);
                __m128 _tmp1 = __lsx_vfadd_s(_r1, _r2);
                __m128 _tmp2 = __lsx_vfsub_s(_r2, _r1);
                __m128 _tmp3 = __lsx_vfsub_s(_r3, _r1);

                __lsx_vst((__m128i)_tmp0, tmp[0][m], 0);
                __lsx_vst((__m128i)_tmp1, tmp[1][m], 0);
                __lsx_vst((__m128i)_tmp2, tmp[2][m], 0);
                __lsx_vst((__m128i)_tmp3, tmp[3][m], 0);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 16 + jj * 4;
            float* p1 = p0 + max_jj * 4;
            float* p2 = p0 + max_jj * 4 * 2;
            float* p3 = p0 + max_jj * 4 * 3;

            for (int m = 0; m < 4; m++)
            {
                __m128 _r0 = (__m128)__lsx_vld(tmp[m][0], 0);
                __m128 _r1 = (__m128)__lsx_vld(tmp[m][1], 0);
                __m128 _r2 = (__m128)__lsx_vld(tmp[m][2], 0);
                __m128 _r3 = (__m128)__lsx_vld(tmp[m][3], 0);

                __m128 _tmp0 = __lsx_vfsub_s(_r0, _r2);
                __m128 _tmp1 = __lsx_vfadd_s(_r1, _r2);
                __m128 _tmp2 = __lsx_vfsub_s(_r2, _r1);
                __m128 _tmp3 = __lsx_vfsub_s(_r3, _r1);

                __lsx_vst((__m128i)_tmp0, p0, 0);
                __lsx_vst((__m128i)_tmp1, p1, 0);
                __lsx_vst((__m128i)_tmp2, p2, 0);
                __lsx_vst((__m128i)_tmp3, p3, 0);

                p0 += max_jj * 4 * 4;
                p1 += max_jj * 4 * 4;
                p2 += max_jj * 4 * 4;
                p3 += max_jj * 4 * 4;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 4;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
#else // __loongarch_sx
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    #pragma omp parallel for num_threads(nT)
#endif // __loongarch_sx
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

        float tmp[4][4][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel(k + kk).row(ti * 2) + (tj * 2);

            for (int m = 0; m < 4; m++)
            {
                float r00 = 0.f;
                float r01 = 0.f;
                float r10 = 0.f;
                float r11 = 0.f;
                float r20 = 0.f;
                float r21 = 0.f;
                float r30 = 0.f;
                float r31 = 0.f;

                if (ti * 2 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const float* r1 = r0 + N;

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

            float* p0 = (float*)B + kk * max_jj * 16 + jj * 2;
            float* p1 = p0 + max_jj * 2;
            float* p2 = p0 + max_jj * 2 * 2;
            float* p3 = p0 + max_jj * 2 * 3;

            for (int m = 0; m < 4; m++)
            {
                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];

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
        float tmp[4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0123 = bottom_blob.channel(k + kk).row(ti * 2) + (tj * 2);

            for (int m = 0; m < 4; m++)
            {
                float r0 = 0.f;
                float r1 = 0.f;
                float r2 = 0.f;
                float r3 = 0.f;

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

            float* p0 = (float*)B + kk * max_jj * 16 + jj;
            float* p1 = p0 + max_jj;
            float* p2 = p0 + max_jj * 2;
            float* p3 = p0 + max_jj * 3;

            for (int m = 0; m < 4; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];

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

static inline void conv3x3s1_winograd23_transform_output_tile(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
{
    // const float otm[2][4] = {
    //     {1.0f,  1.0f,  1.0f,  0.0f},
    //     {0.0f,  1.0f, -1.0f,  1.0f}
    // };

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const size_t N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 1) / 2;

    const float* biasptr = bias;

    int ii = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        __m256 _bias0 = biasptr ? (__m256)__lasx_xvld(biasptr + i + ii, 0) : (__m256)__lasx_xvldi(0);

        __attribute__((aligned(32))) float tmp[2][4][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 16 + jj * 8;
            const float* r1 = r0 + max_jj * 8;
            const float* r2 = r0 + max_jj * 8 * 2;
            const float* r3 = r0 + max_jj * 8 * 3;

            for (int m = 0; m < 4; m++)
            {
                __m256 _r0 = (__m256)__lasx_xvld(r0, 0);
                __m256 _r1 = (__m256)__lasx_xvld(r1, 0);
                __m256 _r2 = (__m256)__lasx_xvld(r2, 0);
                __m256 _r3 = (__m256)__lasx_xvld(r3, 0);

                __m256 _tmp0 = __lasx_xvfadd_s(__lasx_xvfadd_s(_r0, _r1), _r2);
                __m256 _tmp1 = __lasx_xvfadd_s(__lasx_xvfsub_s(_r1, _r2), _r3);

                __lasx_xvst((__m256i)_tmp0, tmp[0][m], 0);
                __lasx_xvst((__m256i)_tmp1, tmp[1][m], 0);

                r0 += max_jj * 4 * 8;
                r1 += max_jj * 4 * 8;
                r2 += max_jj * 4 * 8;
                r3 += max_jj * 4 * 8;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 2) + (tj * 2) * out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                __m256 _r0 = (__m256)__lasx_xvld(tmp[m][0], 0);
                __m256 _r1 = (__m256)__lasx_xvld(tmp[m][1], 0);
                __m256 _r2 = (__m256)__lasx_xvld(tmp[m][2], 0);
                __m256 _r3 = (__m256)__lasx_xvld(tmp[m][3], 0);

                __m256 _tmp0 = __lasx_xvfadd_s(_bias0, __lasx_xvfadd_s(__lasx_xvfadd_s(_r0, _r1), _r2));
                __m256 _tmp1 = __lasx_xvfadd_s(_bias0, __lasx_xvfadd_s(__lasx_xvfsub_s(_r1, _r2), _r3));

                if (out_elempack == 8)
                {
                    __lasx_xvst((__m256i)_tmp0, outptr0, 0);
                    if (tj * 2 + 1 < outw)
                    {
                        __lasx_xvst((__m256i)_tmp1, outptr0 + 8, 0);
                    }
                }
                if (out_elempack == 4)
                {
                    float* outptr1 = outptr0 + N;

                    __lsx_vst(__lasx_extract_128_lo((__m256i)_tmp0), outptr0, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_tmp0), outptr1, 0);
                    if (tj * 2 + 1 < outw)
                    {
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_tmp1), outptr0 + 4, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_tmp1), outptr1 + 4, 0);
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[8];
                    float tmp1[8];
                    __lasx_xvst((__m256i)_tmp0, tmp0, 0);
                    __lasx_xvst((__m256i)_tmp1, tmp1, 0);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;
                    float* outptr4 = outptr0 + N * 4;
                    float* outptr5 = outptr0 + N * 5;
                    float* outptr6 = outptr0 + N * 6;
                    float* outptr7 = outptr0 + N * 7;

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
    for (; ii + 3 < max_ii; ii += 4)
    {
        __m128 _bias0 = biasptr ? (__m128)__lsx_vld(biasptr + i + ii, 0) : (__m128)__lsx_vldi(0);

        __attribute__((aligned(16))) float tmp[2][4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 16 + jj * 4;
            const float* r1 = r0 + max_jj * 4;
            const float* r2 = r0 + max_jj * 4 * 2;
            const float* r3 = r0 + max_jj * 4 * 3;

            for (int m = 0; m < 4; m++)
            {
                __m128 _r0 = (__m128)__lsx_vld(r0, 0);
                __m128 _r1 = (__m128)__lsx_vld(r1, 0);
                __m128 _r2 = (__m128)__lsx_vld(r2, 0);
                __m128 _r3 = (__m128)__lsx_vld(r3, 0);

                __m128 _tmp0 = __lsx_vfadd_s(__lsx_vfadd_s(_r0, _r1), _r2);
                __m128 _tmp1 = __lsx_vfadd_s(__lsx_vfsub_s(_r1, _r2), _r3);

                __lsx_vst((__m128i)_tmp0, tmp[0][m], 0);
                __lsx_vst((__m128i)_tmp1, tmp[1][m], 0);

                r0 += max_jj * 4 * 4;
                r1 += max_jj * 4 * 4;
                r2 += max_jj * 4 * 4;
                r3 += max_jj * 4 * 4;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 2) + (tj * 2) * out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                __m128 _r0 = (__m128)__lsx_vld(tmp[m][0], 0);
                __m128 _r1 = (__m128)__lsx_vld(tmp[m][1], 0);
                __m128 _r2 = (__m128)__lsx_vld(tmp[m][2], 0);
                __m128 _r3 = (__m128)__lsx_vld(tmp[m][3], 0);

                __m128 _tmp0 = __lsx_vfadd_s(_bias0, __lsx_vfadd_s(__lsx_vfadd_s(_r0, _r1), _r2));
                __m128 _tmp1 = __lsx_vfadd_s(_bias0, __lsx_vfadd_s(__lsx_vfsub_s(_r1, _r2), _r3));

                if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_tmp0, outptr0, 0);
                    if (tj * 2 + 1 < outw) __lsx_vst((__m128i)_tmp1, outptr0 + 4, 0);
                }
                if (out_elempack == 1)
                {
                    float tmp0[4];
                    float tmp1[4];
                    __lsx_vst((__m128i)_tmp0, tmp0, 0);
                    __lsx_vst((__m128i)_tmp1, tmp1, 0);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;

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
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;
        float bias1 = biasptr ? biasptr[i + ii + 1] : 0.f;

        float tmp[2][4][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 16 + jj * 2;
            const float* r1 = r0 + max_jj * 2;
            const float* r2 = r0 + max_jj * 2 * 2;
            const float* r3 = r0 + max_jj * 2 * 3;

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

            float* outptr0 = top_blob.channel(i + ii).row(ti * 2) + (tj * 2);

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];

                float tmp00 = bias0 + r00 + r10 + r20;
                float tmp01 = bias1 + r01 + r11 + r21;
                float tmp10 = bias0 + r10 - r20 + r30;
                float tmp11 = bias1 + r11 - r21 + r31;

                // if (out_elempack == 1)
                {
                    float* outptr1 = outptr0 + N;

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
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;

        float tmp[2][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 16 + jj;
            const float* r1 = r0 + max_jj;
            const float* r2 = r0 + max_jj * 2;
            const float* r3 = r0 + max_jj * 3;

            for (int m = 0; m < 4; m++)
            {
                tmp[0][m] = r0[0] + r1[0] + r2[0];
                tmp[1][m] = r1[0] - r2[0] + r3[0];

                r0 += max_jj * 4;
                r1 += max_jj * 4;
                r2 += max_jj * 4;
                r3 += max_jj * 4;
            }

            float* outptr0 = top_blob.channel(i + ii).row(ti * 2) + (tj * 2);

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];

                float tmp0 = bias0 + r0 + r1 + r2;
                float tmp1 = bias0 + r1 - r2 + r3;

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

static int conv3x3s1_winograd23(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, const Option& opt)
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

    // NCNN_LOGE("conv3x3s1_winograd23 %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, TILE_M, TILE_N, TILE_K, nT);

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
            conv3x3s1_winograd23_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, nT);
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
            conv3x3s1_winograd23_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, 1);
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

                gemm_transB_packed_tile(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk);
            }

            // transform output
            conv3x3s1_winograd23_transform_output_tile(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

static inline void conv3x3s1_winograd43_transform_kernel_tile(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    float* ptmp = A;

    int ii = 0;
    for (; ii < max_ii; ii++)
    {
        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            const float sq2 = 1.41421356237f;
            // const float ktm[6][3] = {
            //     {1.0f, 0.0f, 0.0f},
            //     {-2.0f / 3, -sq2 / 3, -1.0f / 3},
            //     {-2.0f / 3, sq2 / 3, -1.0f / 3},
            //     {1.0f / 6, sq2 / 6, 1.0f / 3},
            //     {1.0f / 6, -sq2 / 6, 1.0f / 3},
            //     {0.0f, 0.0f, 1.0f}
            // };
            const float ktm0 = 2.0f / 3;
            const float ktm1 = sq2 / 3;
            const float ktm2 = 1.0f / 3;
            const float ktm3 = 1.0f / 6;
            const float ktm4 = sq2 / 6;

            float tmp[6][3];

            const float* k0 = (const float*)kernel + (i + ii) * inch * 9 + (k + kk) * 9;

            for (int m = 0; m < 3; m++)
            {
                float r0 = k0[0];
                float r1 = k0[1];
                float r2 = k0[2];

                tmp[0][m] = r0;
                tmp[1][m] = -r0 * ktm0 - r1 * ktm1 - r2 * ktm2;
                tmp[2][m] = -r0 * ktm0 + r1 * ktm1 - r2 * ktm2;
                tmp[3][m] = r0 * ktm3 + r1 * ktm4 + r2 * ktm2;
                tmp[4][m] = r0 * ktm3 - r1 * ktm4 + r2 * ktm2;
                tmp[5][m] = r2;

                k0 += 3;
            }

            for (int m = 0; m < 6; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];

                float z0 = r0;
                float z1 = -r0 * ktm0 - r1 * ktm1 - r2 * ktm2;
                float z2 = -r0 * ktm0 + r1 * ktm1 - r2 * ktm2;
                float z3 = r0 * ktm3 + r1 * ktm4 + r2 * ktm2;
                float z4 = r0 * ktm3 - r1 * ktm4 + r2 * ktm2;
                float z5 = r2;

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

static void conv3x3s1_winograd43_transform_kernel(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 36;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

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

            conv3x3s1_winograd43_transform_kernel_tile(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            pack_A_tile(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd43_transform_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    const float sq2 = 1.41421356237;
    const float sq2_d2 = 1.41421356237 / 2;

    // const float itm[6][6] = {
    //     {1.0f,  0.0f,  -2.5f,  0.0f,  1.0f, 0.0f},
    //     {0.0f, -sq2,   -2.0f,  sq2/2, 1.0f, 0.0f},
    //     {0.0f,  sq2,   -2.0f, -sq2/2, 1.0f, 0.0f},
    //     {0.0f, -sq2/2, -0.5f,  sq2,   1.0f, 0.0f},
    //     {0.0f,  sq2/2, -0.5f, -sq2,   1.0f, 0.0f},
    //     {0.0f,  1.0f,   0.0f,  -2.5f, 0.0f, 1.0f}
    // };

    // 0 =  r00 + r04 - 2.5f * r02
    // 1 = -(sq2 * r01 - sq2_d2 * r03) + (r04 - 2 * r02)
    // 2 =  (sq2 * r01 - sq2_d2 * r03) + (r04 - 2 * r02)
    // 3 =  (sq2 * r03 - sq2_d2 * r01) + (r04 - 0.5f * r02)
    // 4 = -(sq2 * r03 - sq2_d2 * r01) + (r04 - 0.5f * r02)
    // 5 =  r01 + r05 - 2.5f * r03

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const size_t N = bottom_blob.cstep * elempack;

    const int w_tiles = (w + 1) / 4;

    int nn_max_kk = 0;
    int remain_max_kk_start = 0;
#if __loongarch_sx
#if __loongarch_asx
    nn_max_kk = max_kk / 8;
    #pragma omp parallel for num_threads(nT)
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 8;

        __attribute__((aligned(32))) float tmp[6][6][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 4) + (tj * 4) * elempack;

            __m256 _vm2_5 = (__m256)__lasx_xvreplfr2vr_s(-2.5f);
            __m256 _vsq2 = (__m256)__lasx_xvreplfr2vr_s(sq2);
            __m256 _vmsq2_d2 = (__m256)__lasx_xvreplfr2vr_s(-sq2_d2);
            __m256 _vm2 = (__m256)__lasx_xvreplfr2vr_s(-2.f);
            __m256 _vm0_5 = (__m256)__lasx_xvreplfr2vr_s(-0.5f);

            for (int m = 0; m < 6; m++)
            {
                __m256 _r0 = (__m256)__lasx_xvldi(0);
                __m256 _r1 = (__m256)__lasx_xvldi(0);
                __m256 _r2 = (__m256)__lasx_xvldi(0);
                __m256 _r3 = (__m256)__lasx_xvldi(0);
                __m256 _r4 = (__m256)__lasx_xvldi(0);
                __m256 _r5 = (__m256)__lasx_xvldi(0);

                if (ti * 4 + m < h)
                {
                    if (elempack == 8)
                    {
                        _r0 = (__m256)__lasx_xvld(r0, 0);
                        if (tj * 4 + 1 < w) _r1 = (__m256)__lasx_xvld(r0 + 8, 0);
                        if (tj * 4 + 2 < w) _r2 = (__m256)__lasx_xvld(r0 + 16, 0);
                        if (tj * 4 + 3 < w) _r3 = (__m256)__lasx_xvld(r0 + 24, 0);
                        if (tj * 4 + 4 < w) _r4 = (__m256)__lasx_xvld(r0 + 32, 0);
                        if (tj * 4 + 5 < w) _r5 = (__m256)__lasx_xvld(r0 + 40, 0);
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;

                        _r0 = (__m256)__lasx_concat_128((__m128i)(__m128)__lsx_vld(r0, 0), (__m128i)(__m128)__lsx_vld(r1, 0));
                        if (tj * 4 + 1 < w) _r1 = (__m256)__lasx_concat_128((__m128i)(__m128)__lsx_vld(r0 + 4, 0), (__m128i)(__m128)__lsx_vld(r1 + 4, 0));
                        if (tj * 4 + 2 < w) _r2 = (__m256)__lasx_concat_128((__m128i)(__m128)__lsx_vld(r0 + 8, 0), (__m128i)(__m128)__lsx_vld(r1 + 8, 0));
                        if (tj * 4 + 3 < w) _r3 = (__m256)__lasx_concat_128((__m128i)(__m128)__lsx_vld(r0 + 12, 0), (__m128i)(__m128)__lsx_vld(r1 + 12, 0));
                        if (tj * 4 + 4 < w) _r4 = (__m256)__lasx_concat_128((__m128i)(__m128)__lsx_vld(r0 + 16, 0), (__m128i)(__m128)__lsx_vld(r1 + 16, 0));
                        if (tj * 4 + 5 < w) _r5 = (__m256)__lasx_concat_128((__m128i)(__m128)__lsx_vld(r0 + 20, 0), (__m128i)(__m128)__lsx_vld(r1 + 20, 0));
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;
                        const float* r4 = r0 + N * 4;
                        const float* r5 = r0 + N * 5;
                        const float* r6 = r0 + N * 6;
                        const float* r7 = r0 + N * 7;

                        __m128 _t0 = (__m128)__lsx_vld(r0, 0);
                        __m128 _t1 = (__m128)__lsx_vld(r1, 0);
                        __m128 _t2 = (__m128)__lsx_vld(r2, 0);
                        __m128 _t3 = (__m128)__lsx_vld(r3, 0);
                        __m128 _t4 = (__m128)__lsx_vld(r4, 0);
                        __m128 _t5 = (__m128)__lsx_vld(r5, 0);
                        __m128 _t6 = (__m128)__lsx_vld(r6, 0);
                        __m128 _t7 = (__m128)__lsx_vld(r7, 0);

                        {
                            __m128i _01l = __lsx_vilvl_w((__m128i)_t1, (__m128i)_t0);
                            __m128i _01h = __lsx_vilvh_w((__m128i)_t1, (__m128i)_t0);
                            __m128i _23l = __lsx_vilvl_w((__m128i)_t3, (__m128i)_t2);
                            __m128i _23h = __lsx_vilvh_w((__m128i)_t3, (__m128i)_t2);
                            _t0 = (__m128)__lsx_vilvl_d(_23l, _01l);
                            _t1 = (__m128)__lsx_vilvh_d(_23l, _01l);
                            _t2 = (__m128)__lsx_vilvl_d(_23h, _01h);
                            _t3 = (__m128)__lsx_vilvh_d(_23h, _01h);
                        }
                        {
                            __m128i _01l = __lsx_vilvl_w((__m128i)_t5, (__m128i)_t4);
                            __m128i _01h = __lsx_vilvh_w((__m128i)_t5, (__m128i)_t4);
                            __m128i _23l = __lsx_vilvl_w((__m128i)_t7, (__m128i)_t6);
                            __m128i _23h = __lsx_vilvh_w((__m128i)_t7, (__m128i)_t6);
                            _t4 = (__m128)__lsx_vilvl_d(_23l, _01l);
                            _t5 = (__m128)__lsx_vilvh_d(_23l, _01l);
                            _t6 = (__m128)__lsx_vilvl_d(_23h, _01h);
                            _t7 = (__m128)__lsx_vilvh_d(_23h, _01h);
                        }

                        _r0 = (__m256)__lasx_concat_128((__m128i)_t0, (__m128i)_t4);
                        if (tj * 4 + 1 < w) _r1 = (__m256)__lasx_concat_128((__m128i)_t1, (__m128i)_t5);
                        if (tj * 4 + 2 < w) _r2 = (__m256)__lasx_concat_128((__m128i)_t2, (__m128i)_t6);
                        if (tj * 4 + 3 < w) _r3 = (__m256)__lasx_concat_128((__m128i)_t3, (__m128i)_t7);
                        if (tj * 4 + 4 < w) _r4 = ({ float __set_tmp[8] = {r0[4], r1[4], r2[4], r3[4], r4[4], r5[4], r6[4], r7[4]}; (__m256)__lasx_xvld(__set_tmp, 0); });
                        if (tj * 4 + 5 < w) _r5 = ({ float __set_tmp[8] = {r0[5], r1[5], r2[5], r3[5], r4[5], r5[5], r6[5], r7[5]}; (__m256)__lasx_xvld(__set_tmp, 0); });
                    }
                }

                __m256 _tmp12a = __lasx_xvfmadd_s(_vmsq2_d2, _r3, __lasx_xvfmul_s(_r1, _vsq2));
                __m256 _tmp12b = __lasx_xvfmadd_s(_vm2, _r2, _r4);
                __m256 _tmp34a = __lasx_xvfmadd_s(_vmsq2_d2, _r1, __lasx_xvfmul_s(_r3, _vsq2));
                __m256 _tmp34b = __lasx_xvfmadd_s(_vm0_5, _r2, _r4);

                __m256 _tmp0 = __lasx_xvfmadd_s(_vm2_5, _r2, __lasx_xvfadd_s(_r0, _r4));
                __m256 _tmp1 = __lasx_xvfsub_s(_tmp12b, _tmp12a);
                __m256 _tmp2 = __lasx_xvfadd_s(_tmp12b, _tmp12a);
                __m256 _tmp3 = __lasx_xvfadd_s(_tmp34b, _tmp34a);
                __m256 _tmp4 = __lasx_xvfsub_s(_tmp34b, _tmp34a);
                __m256 _tmp5 = __lasx_xvfmadd_s(_vm2_5, _r3, __lasx_xvfadd_s(_r1, _r5));

                __lasx_xvst((__m256i)_tmp0, tmp[0][m], 0);
                __lasx_xvst((__m256i)_tmp1, tmp[1][m], 0);
                __lasx_xvst((__m256i)_tmp2, tmp[2][m], 0);
                __lasx_xvst((__m256i)_tmp3, tmp[3][m], 0);
                __lasx_xvst((__m256i)_tmp4, tmp[4][m], 0);
                __lasx_xvst((__m256i)_tmp5, tmp[5][m], 0);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 36 + jj * 8;
            float* p1 = p0 + max_jj * 8;
            float* p2 = p0 + max_jj * 8 * 2;
            float* p3 = p0 + max_jj * 8 * 3;
            float* p4 = p0 + max_jj * 8 * 4;
            float* p5 = p0 + max_jj * 8 * 5;

            for (int m = 0; m < 6; m++)
            {
                __m256 _r0 = (__m256)__lasx_xvld(tmp[m][0], 0);
                __m256 _r1 = (__m256)__lasx_xvld(tmp[m][1], 0);
                __m256 _r2 = (__m256)__lasx_xvld(tmp[m][2], 0);
                __m256 _r3 = (__m256)__lasx_xvld(tmp[m][3], 0);
                __m256 _r4 = (__m256)__lasx_xvld(tmp[m][4], 0);
                __m256 _r5 = (__m256)__lasx_xvld(tmp[m][5], 0);

                __m256 _tmp12a = __lasx_xvfmadd_s(_vmsq2_d2, _r3, __lasx_xvfmul_s(_r1, _vsq2));
                __m256 _tmp12b = __lasx_xvfmadd_s(_vm2, _r2, _r4);
                __m256 _tmp34a = __lasx_xvfmadd_s(_vmsq2_d2, _r1, __lasx_xvfmul_s(_r3, _vsq2));
                __m256 _tmp34b = __lasx_xvfmadd_s(_vm0_5, _r2, _r4);

                __m256 _tmp0 = __lasx_xvfmadd_s(_vm2_5, _r2, __lasx_xvfadd_s(_r0, _r4));
                __m256 _tmp1 = __lasx_xvfsub_s(_tmp12b, _tmp12a);
                __m256 _tmp2 = __lasx_xvfadd_s(_tmp12b, _tmp12a);
                __m256 _tmp3 = __lasx_xvfadd_s(_tmp34b, _tmp34a);
                __m256 _tmp4 = __lasx_xvfsub_s(_tmp34b, _tmp34a);
                __m256 _tmp5 = __lasx_xvfmadd_s(_vm2_5, _r3, __lasx_xvfadd_s(_r1, _r5));

                __lasx_xvst((__m256i)_tmp0, p0, 0);
                __lasx_xvst((__m256i)_tmp1, p1, 0);
                __lasx_xvst((__m256i)_tmp2, p2, 0);
                __lasx_xvst((__m256i)_tmp3, p3, 0);
                __lasx_xvst((__m256i)_tmp4, p4, 0);
                __lasx_xvst((__m256i)_tmp5, p5, 0);

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
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
    #pragma omp parallel for num_threads(nT)
#else // __loongarch_asx
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
    #pragma omp parallel for num_threads(nT)
#endif // __loongarch_asx
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 4;

        __attribute__((aligned(16))) float tmp[6][6][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 4) + (tj * 4) * elempack;

            __m128 _vm2_5 = (__m128)__lsx_vreplfr2vr_s(-2.5f);
            __m128 _vsq2 = (__m128)__lsx_vreplfr2vr_s(sq2);
            __m128 _vmsq2_d2 = (__m128)__lsx_vreplfr2vr_s(-sq2_d2);
            __m128 _vm2 = (__m128)__lsx_vreplfr2vr_s(-2.f);
            __m128 _vm0_5 = (__m128)__lsx_vreplfr2vr_s(-0.5f);

            for (int m = 0; m < 6; m++)
            {
                __m128 _r0 = (__m128)__lsx_vldi(0);
                __m128 _r1 = (__m128)__lsx_vldi(0);
                __m128 _r2 = (__m128)__lsx_vldi(0);
                __m128 _r3 = (__m128)__lsx_vldi(0);
                __m128 _r4 = (__m128)__lsx_vldi(0);
                __m128 _r5 = (__m128)__lsx_vldi(0);

                if (ti * 4 + m < h)
                {
                    if (elempack == 4)
                    {
                        _r0 = (__m128)__lsx_vld(r0, 0);
                        if (tj * 4 + 1 < w) _r1 = (__m128)__lsx_vld(r0 + 4, 0);
                        if (tj * 4 + 2 < w) _r2 = (__m128)__lsx_vld(r0 + 8, 0);
                        if (tj * 4 + 3 < w) _r3 = (__m128)__lsx_vld(r0 + 12, 0);
                        if (tj * 4 + 4 < w) _r4 = (__m128)__lsx_vld(r0 + 16, 0);
                        if (tj * 4 + 5 < w) _r5 = (__m128)__lsx_vld(r0 + 20, 0);
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;

                        __m128 _t0 = (__m128)__lsx_vld(r0, 0);
                        __m128 _t1 = (__m128)__lsx_vld(r1, 0);
                        __m128 _t2 = (__m128)__lsx_vld(r2, 0);
                        __m128 _t3 = (__m128)__lsx_vld(r3, 0);

                        {
                            __m128i _01l = __lsx_vilvl_w((__m128i)_t1, (__m128i)_t0);
                            __m128i _01h = __lsx_vilvh_w((__m128i)_t1, (__m128i)_t0);
                            __m128i _23l = __lsx_vilvl_w((__m128i)_t3, (__m128i)_t2);
                            __m128i _23h = __lsx_vilvh_w((__m128i)_t3, (__m128i)_t2);
                            _t0 = (__m128)__lsx_vilvl_d(_23l, _01l);
                            _t1 = (__m128)__lsx_vilvh_d(_23l, _01l);
                            _t2 = (__m128)__lsx_vilvl_d(_23h, _01h);
                            _t3 = (__m128)__lsx_vilvh_d(_23h, _01h);
                        }

                        _r0 = _t0;
                        if (tj * 4 + 1 < w) _r1 = _t1;
                        if (tj * 4 + 2 < w) _r2 = _t2;
                        if (tj * 4 + 3 < w) _r3 = _t3;
                        if (tj * 4 + 4 < w)
                        {
                            float __set_tmp[4] = {r0[4], r1[4], r2[4], r3[4]};
                            _r4 = (__m128)__lsx_vld(__set_tmp, 0);
                        }
                        if (tj * 4 + 5 < w)
                        {
                            float __set_tmp[4] = {r0[5], r1[5], r2[5], r3[5]};
                            _r5 = (__m128)__lsx_vld(__set_tmp, 0);
                        }
                    }
                }

                __m128 _tmp12a = __lsx_vfmadd_s(_vmsq2_d2, _r3, __lsx_vfmul_s(_r1, _vsq2));
                __m128 _tmp12b = __lsx_vfmadd_s(_vm2, _r2, _r4);
                __m128 _tmp34a = __lsx_vfmadd_s(_vmsq2_d2, _r1, __lsx_vfmul_s(_r3, _vsq2));
                __m128 _tmp34b = __lsx_vfmadd_s(_vm0_5, _r2, _r4);

                __m128 _tmp0 = __lsx_vfmadd_s(_vm2_5, _r2, __lsx_vfadd_s(_r0, _r4));
                __m128 _tmp1 = __lsx_vfsub_s(_tmp12b, _tmp12a);
                __m128 _tmp2 = __lsx_vfadd_s(_tmp12b, _tmp12a);
                __m128 _tmp3 = __lsx_vfadd_s(_tmp34b, _tmp34a);
                __m128 _tmp4 = __lsx_vfsub_s(_tmp34b, _tmp34a);
                __m128 _tmp5 = __lsx_vfmadd_s(_vm2_5, _r3, __lsx_vfadd_s(_r1, _r5));

                __lsx_vst((__m128i)_tmp0, tmp[0][m], 0);
                __lsx_vst((__m128i)_tmp1, tmp[1][m], 0);
                __lsx_vst((__m128i)_tmp2, tmp[2][m], 0);
                __lsx_vst((__m128i)_tmp3, tmp[3][m], 0);
                __lsx_vst((__m128i)_tmp4, tmp[4][m], 0);
                __lsx_vst((__m128i)_tmp5, tmp[5][m], 0);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 36 + jj * 4;
            float* p1 = p0 + max_jj * 4;
            float* p2 = p0 + max_jj * 4 * 2;
            float* p3 = p0 + max_jj * 4 * 3;
            float* p4 = p0 + max_jj * 4 * 4;
            float* p5 = p0 + max_jj * 4 * 5;

            for (int m = 0; m < 6; m++)
            {
                __m128 _r0 = (__m128)__lsx_vld(tmp[m][0], 0);
                __m128 _r1 = (__m128)__lsx_vld(tmp[m][1], 0);
                __m128 _r2 = (__m128)__lsx_vld(tmp[m][2], 0);
                __m128 _r3 = (__m128)__lsx_vld(tmp[m][3], 0);
                __m128 _r4 = (__m128)__lsx_vld(tmp[m][4], 0);
                __m128 _r5 = (__m128)__lsx_vld(tmp[m][5], 0);

                __m128 _tmp12a = __lsx_vfmadd_s(_vmsq2_d2, _r3, __lsx_vfmul_s(_r1, _vsq2));
                __m128 _tmp12b = __lsx_vfmadd_s(_vm2, _r2, _r4);
                __m128 _tmp34a = __lsx_vfmadd_s(_vmsq2_d2, _r1, __lsx_vfmul_s(_r3, _vsq2));
                __m128 _tmp34b = __lsx_vfmadd_s(_vm0_5, _r2, _r4);

                __m128 _tmp0 = __lsx_vfmadd_s(_vm2_5, _r2, __lsx_vfadd_s(_r0, _r4));
                __m128 _tmp1 = __lsx_vfsub_s(_tmp12b, _tmp12a);
                __m128 _tmp2 = __lsx_vfadd_s(_tmp12b, _tmp12a);
                __m128 _tmp3 = __lsx_vfadd_s(_tmp34b, _tmp34a);
                __m128 _tmp4 = __lsx_vfsub_s(_tmp34b, _tmp34a);
                __m128 _tmp5 = __lsx_vfmadd_s(_vm2_5, _r3, __lsx_vfadd_s(_r1, _r5));

                __lsx_vst((__m128i)_tmp0, p0, 0);
                __lsx_vst((__m128i)_tmp1, p1, 0);
                __lsx_vst((__m128i)_tmp2, p2, 0);
                __lsx_vst((__m128i)_tmp3, p3, 0);
                __lsx_vst((__m128i)_tmp4, p4, 0);
                __lsx_vst((__m128i)_tmp5, p5, 0);

                p0 += max_jj * 6 * 4;
                p1 += max_jj * 6 * 4;
                p2 += max_jj * 6 * 4;
                p3 += max_jj * 6 * 4;
                p4 += max_jj * 6 * 4;
                p5 += max_jj * 6 * 4;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 4;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
#else // __loongarch_sx
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    #pragma omp parallel for num_threads(nT)
#endif // __loongarch_sx
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

        float tmp[6][6][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel(k + kk).row(ti * 4) + (tj * 4);

            for (int m = 0; m < 6; m++)
            {
                float r00 = 0.f;
                float r01 = 0.f;
                float r10 = 0.f;
                float r11 = 0.f;
                float r20 = 0.f;
                float r21 = 0.f;
                float r30 = 0.f;
                float r31 = 0.f;
                float r40 = 0.f;
                float r41 = 0.f;
                float r50 = 0.f;
                float r51 = 0.f;

                if (ti * 4 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const float* r1 = r0 + N;

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

                float tmp12a0 = sq2 * r10 - sq2_d2 * r30;
                float tmp12a1 = sq2 * r11 - sq2_d2 * r31;
                float tmp12b0 = r40 - 2 * r20;
                float tmp12b1 = r41 - 2 * r21;
                float tmp34a0 = sq2 * r30 - sq2_d2 * r10;
                float tmp34a1 = sq2 * r31 - sq2_d2 * r11;
                float tmp34b0 = r40 - 0.5f * r20;
                float tmp34b1 = r41 - 0.5f * r21;

                tmp[0][m][0] = r00 + r40 - 2.5f * r20;
                tmp[0][m][1] = r01 + r41 - 2.5f * r21;
                tmp[1][m][0] = tmp12b0 - tmp12a0;
                tmp[1][m][1] = tmp12b1 - tmp12a1;
                tmp[2][m][0] = tmp12b0 + tmp12a0;
                tmp[2][m][1] = tmp12b1 + tmp12a1;
                tmp[3][m][0] = tmp34b0 + tmp34a0;
                tmp[3][m][1] = tmp34b1 + tmp34a1;
                tmp[4][m][0] = tmp34b0 - tmp34a0;
                tmp[4][m][1] = tmp34b1 - tmp34a1;
                tmp[5][m][0] = r10 + r50 - 2.5f * r30;
                tmp[5][m][1] = r11 + r51 - 2.5f * r31;

                r0 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 36 + jj * 2;
            float* p1 = p0 + max_jj * 2;
            float* p2 = p0 + max_jj * 2 * 2;
            float* p3 = p0 + max_jj * 2 * 3;
            float* p4 = p0 + max_jj * 2 * 4;
            float* p5 = p0 + max_jj * 2 * 5;

            for (int m = 0; m < 6; m++)
            {
                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];
                float r40 = tmp[m][4][0];
                float r41 = tmp[m][4][1];
                float r50 = tmp[m][5][0];
                float r51 = tmp[m][5][1];

                float tmp12a0 = sq2 * r10 - sq2_d2 * r30;
                float tmp12a1 = sq2 * r11 - sq2_d2 * r31;
                float tmp12b0 = r40 - 2 * r20;
                float tmp12b1 = r41 - 2 * r21;
                float tmp34a0 = sq2 * r30 - sq2_d2 * r10;
                float tmp34a1 = sq2 * r31 - sq2_d2 * r11;
                float tmp34b0 = r40 - 0.5f * r20;
                float tmp34b1 = r41 - 0.5f * r21;

                p0[0] = r00 + r40 - 2.5f * r20;
                p0[1] = r01 + r41 - 2.5f * r21;
                p1[0] = tmp12b0 - tmp12a0;
                p1[1] = tmp12b1 - tmp12a1;
                p2[0] = tmp12b0 + tmp12a0;
                p2[1] = tmp12b1 + tmp12a1;
                p3[0] = tmp34b0 + tmp34a0;
                p3[1] = tmp34b1 + tmp34a1;
                p4[0] = tmp34b0 - tmp34a0;
                p4[1] = tmp34b1 - tmp34a1;
                p5[0] = r10 + r50 - 2.5f * r30;
                p5[1] = r11 + r51 - 2.5f * r31;

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
        float tmp[6][6];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0123 = bottom_blob.channel(k + kk).row(ti * 4) + (tj * 4);

            for (int m = 0; m < 6; m++)
            {
                float r0 = 0.f;
                float r1 = 0.f;
                float r2 = 0.f;
                float r3 = 0.f;
                float r4 = 0.f;
                float r5 = 0.f;

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

                float tmp12a = sq2 * r1 - sq2_d2 * r3;
                float tmp12b = r4 - 2 * r2;
                float tmp34a = sq2 * r3 - sq2_d2 * r1;
                float tmp34b = r4 - 0.5f * r2;

                tmp[0][m] = r0 + r4 - 2.5f * r2;
                tmp[1][m] = tmp12b - tmp12a;
                tmp[2][m] = tmp12b + tmp12a;
                tmp[3][m] = tmp34b + tmp34a;
                tmp[4][m] = tmp34b - tmp34a;
                tmp[5][m] = r1 + r5 - 2.5f * r3;

                r0123 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 36 + jj;
            float* p1 = p0 + max_jj;
            float* p2 = p0 + max_jj * 2;
            float* p3 = p0 + max_jj * 3;
            float* p4 = p0 + max_jj * 4;
            float* p5 = p0 + max_jj * 5;

            for (int m = 0; m < 6; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];
                float r4 = tmp[m][4];
                float r5 = tmp[m][5];

                float tmp12a = sq2 * r1 - sq2_d2 * r3;
                float tmp12b = r4 - 2 * r2;
                float tmp34a = sq2 * r3 - sq2_d2 * r1;
                float tmp34b = r4 - 0.5f * r2;

                p0[0] = r0 + r4 - 2.5f * r2;
                p1[0] = tmp12b - tmp12a;
                p2[0] = tmp12b + tmp12a;
                p3[0] = tmp34b + tmp34a;
                p4[0] = tmp34b - tmp34a;
                p5[0] = r1 + r5 - 2.5f * r3;

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

static inline void conv3x3s1_winograd43_transform_output_tile(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
{
    const float sq2 = 1.41421356237;
    const float sq2_m2 = 1.41421356237 * 2;
    const float sq2_d2 = 1.41421356237 / 2;
    const float sq2_d4 = 1.41421356237 / 4;

    // const float otm[4][6] = {
    //     {1.0f, 1.0f,   1.0f,  1.0f,  1.0f,   0.0f},
    //     {0.0f, sq2/2, -sq2/2, sq2,   -sq2,   0.0f},
    //     {0.0f, 0.5f,   0.5f,  2.0f,  2.0f,   0.0f},
    //     {0.0f, sq2/4, -sq2/4, sq2*2, -sq2*2, 1.0f}
    // };

    // 0 = r00 + (r01 + r02) + (r03 + r04)
    // 1 =       (r01 - r02) * sq2_d2 + (r03 - r04) * sq2
    // 2 =       (r01 + r02) * 0.5f + (r03 + r04) * 2
    // 3 = r05 + (r01 - r02) * sq2_d4 + (r03 - r04) * sq2_m2

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const size_t N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 3) / 4;

    const float* biasptr = bias;

    int ii = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        __m256 _bias0 = biasptr ? (__m256)__lasx_xvld(biasptr + i + ii, 0) : (__m256)__lasx_xvldi(0);

        __attribute__((aligned(32))) float tmp[4][6][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 36 + jj * 8;
            const float* r1 = r0 + max_jj * 8;
            const float* r2 = r0 + max_jj * 8 * 2;
            const float* r3 = r0 + max_jj * 8 * 3;
            const float* r4 = r0 + max_jj * 8 * 4;
            const float* r5 = r0 + max_jj * 8 * 5;

            __m256 _vsq2 = (__m256)__lasx_xvreplfr2vr_s(sq2);
            __m256 _vsq2_d2 = (__m256)__lasx_xvreplfr2vr_s(sq2_d2);
            __m256 _vsq2_d4 = (__m256)__lasx_xvreplfr2vr_s(sq2_d4);
            __m256 _vsq2_m2 = (__m256)__lasx_xvreplfr2vr_s(sq2_m2);
            __m256 _v0_5 = (__m256)__lasx_xvreplfr2vr_s(0.5f);
            __m256 _v2 = (__m256)__lasx_xvreplfr2vr_s(2.f);

            for (int m = 0; m < 6; m++)
            {
                __m256 _r0 = (__m256)__lasx_xvld(r0, 0);
                __m256 _r1 = (__m256)__lasx_xvld(r1, 0);
                __m256 _r2 = (__m256)__lasx_xvld(r2, 0);
                __m256 _r3 = (__m256)__lasx_xvld(r3, 0);
                __m256 _r4 = (__m256)__lasx_xvld(r4, 0);
                __m256 _r5 = (__m256)__lasx_xvld(r5, 0);

                __m256 _tmp02a = __lasx_xvfadd_s(_r1, _r2);
                __m256 _tmp02b = __lasx_xvfadd_s(_r3, _r4);
                __m256 _tmp13a = __lasx_xvfsub_s(_r1, _r2);
                __m256 _tmp13b = __lasx_xvfsub_s(_r3, _r4);

                __m256 _tmp0 = __lasx_xvfadd_s(__lasx_xvfadd_s(_r0, _tmp02a), _tmp02b);
                __m256 _tmp1 = __lasx_xvfmadd_s(_tmp13b, _vsq2, __lasx_xvfmul_s(_tmp13a, _vsq2_d2));
                __m256 _tmp2 = __lasx_xvfmadd_s(_tmp02b, _v2, __lasx_xvfmul_s(_tmp02a, _v0_5));
                __m256 _tmp3 = __lasx_xvfmadd_s(_tmp13b, _vsq2_m2, __lasx_xvfmadd_s(_tmp13a, _vsq2_d4, _r5));

                __lasx_xvst((__m256i)_tmp0, tmp[0][m], 0);
                __lasx_xvst((__m256i)_tmp1, tmp[1][m], 0);
                __lasx_xvst((__m256i)_tmp2, tmp[2][m], 0);
                __lasx_xvst((__m256i)_tmp3, tmp[3][m], 0);

                r0 += max_jj * 6 * 8;
                r1 += max_jj * 6 * 8;
                r2 += max_jj * 6 * 8;
                r3 += max_jj * 6 * 8;
                r4 += max_jj * 6 * 8;
                r5 += max_jj * 6 * 8;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 4) + (tj * 4) * out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                __m256 _r0 = (__m256)__lasx_xvld(tmp[m][0], 0);
                __m256 _r1 = (__m256)__lasx_xvld(tmp[m][1], 0);
                __m256 _r2 = (__m256)__lasx_xvld(tmp[m][2], 0);
                __m256 _r3 = (__m256)__lasx_xvld(tmp[m][3], 0);
                __m256 _r4 = (__m256)__lasx_xvld(tmp[m][4], 0);
                __m256 _r5 = (__m256)__lasx_xvld(tmp[m][5], 0);

                __m256 _tmp02a = __lasx_xvfadd_s(_r1, _r2);
                __m256 _tmp02b = __lasx_xvfadd_s(_r3, _r4);
                __m256 _tmp13a = __lasx_xvfsub_s(_r1, _r2);
                __m256 _tmp13b = __lasx_xvfsub_s(_r3, _r4);

                __m256 _tmp0 = __lasx_xvfadd_s(__lasx_xvfadd_s(_r0, _tmp02a), __lasx_xvfadd_s(_tmp02b, _bias0));
                __m256 _tmp1 = __lasx_xvfmadd_s(_tmp13b, _vsq2, __lasx_xvfmadd_s(_tmp13a, _vsq2_d2, _bias0));
                __m256 _tmp2 = __lasx_xvfmadd_s(_tmp02b, _v2, __lasx_xvfmadd_s(_tmp02a, _v0_5, _bias0));
                __m256 _tmp3 = __lasx_xvfmadd_s(_tmp13b, _vsq2_m2, __lasx_xvfmadd_s(_tmp13a, _vsq2_d4, __lasx_xvfadd_s(_r5, _bias0)));

                if (out_elempack == 8)
                {
                    __lasx_xvst((__m256i)_tmp0, outptr0, 0);
                    if (tj * 4 + 1 < outw) __lasx_xvst((__m256i)_tmp1, outptr0 + 8, 0);
                    if (tj * 4 + 2 < outw) __lasx_xvst((__m256i)_tmp2, outptr0 + 16, 0);
                    if (tj * 4 + 3 < outw) __lasx_xvst((__m256i)_tmp3, outptr0 + 24, 0);
                }
                if (out_elempack == 4)
                {
                    float* outptr1 = outptr0 + N;

                    __lsx_vst(__lasx_extract_128_lo((__m256i)_tmp0), outptr0, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_tmp0), outptr1, 0);
                    if (tj * 4 + 1 < outw)
                    {
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_tmp1), outptr0 + 4, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_tmp1), outptr1 + 4, 0);
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_tmp2), outptr0 + 8, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_tmp2), outptr1 + 8, 0);
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_tmp3), outptr0 + 12, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_tmp3), outptr1 + 12, 0);
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[8];
                    float tmp1[8];
                    float tmp2[8];
                    float tmp3[8];
                    __lasx_xvst((__m256i)_tmp0, tmp0, 0);
                    __lasx_xvst((__m256i)_tmp1, tmp1, 0);
                    __lasx_xvst((__m256i)_tmp2, tmp2, 0);
                    __lasx_xvst((__m256i)_tmp3, tmp3, 0);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;
                    float* outptr4 = outptr0 + N * 4;
                    float* outptr5 = outptr0 + N * 5;
                    float* outptr6 = outptr0 + N * 6;
                    float* outptr7 = outptr0 + N * 7;

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
    for (; ii + 3 < max_ii; ii += 4)
    {
        __m128 _bias0 = biasptr ? (__m128)__lsx_vld(biasptr + i + ii, 0) : (__m128)__lsx_vldi(0);

        __attribute__((aligned(16))) float tmp[4][6][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 36 + jj * 4;
            const float* r1 = r0 + max_jj * 4;
            const float* r2 = r0 + max_jj * 4 * 2;
            const float* r3 = r0 + max_jj * 4 * 3;
            const float* r4 = r0 + max_jj * 4 * 4;
            const float* r5 = r0 + max_jj * 4 * 5;

            __m128 _vsq2 = (__m128)__lsx_vreplfr2vr_s(sq2);
            __m128 _vsq2_d2 = (__m128)__lsx_vreplfr2vr_s(sq2_d2);
            __m128 _vsq2_d4 = (__m128)__lsx_vreplfr2vr_s(sq2_d4);
            __m128 _vsq2_m2 = (__m128)__lsx_vreplfr2vr_s(sq2_m2);
            __m128 _v0_5 = (__m128)__lsx_vreplfr2vr_s(0.5f);
            __m128 _v2 = (__m128)__lsx_vreplfr2vr_s(2.f);

            for (int m = 0; m < 6; m++)
            {
                __m128 _r0 = (__m128)__lsx_vld(r0, 0);
                __m128 _r1 = (__m128)__lsx_vld(r1, 0);
                __m128 _r2 = (__m128)__lsx_vld(r2, 0);
                __m128 _r3 = (__m128)__lsx_vld(r3, 0);
                __m128 _r4 = (__m128)__lsx_vld(r4, 0);
                __m128 _r5 = (__m128)__lsx_vld(r5, 0);

                __m128 _tmp02a = __lsx_vfadd_s(_r1, _r2);
                __m128 _tmp02b = __lsx_vfadd_s(_r3, _r4);
                __m128 _tmp13a = __lsx_vfsub_s(_r1, _r2);
                __m128 _tmp13b = __lsx_vfsub_s(_r3, _r4);

                __m128 _tmp0 = __lsx_vfadd_s(__lsx_vfadd_s(_r0, _tmp02a), _tmp02b);
                __m128 _tmp1 = __lsx_vfmadd_s(_tmp13b, _vsq2, __lsx_vfmul_s(_tmp13a, _vsq2_d2));
                __m128 _tmp2 = __lsx_vfmadd_s(_tmp02b, _v2, __lsx_vfmul_s(_tmp02a, _v0_5));
                __m128 _tmp3 = __lsx_vfmadd_s(_tmp13b, _vsq2_m2, __lsx_vfmadd_s(_tmp13a, _vsq2_d4, _r5));

                __lsx_vst((__m128i)_tmp0, tmp[0][m], 0);
                __lsx_vst((__m128i)_tmp1, tmp[1][m], 0);
                __lsx_vst((__m128i)_tmp2, tmp[2][m], 0);
                __lsx_vst((__m128i)_tmp3, tmp[3][m], 0);

                r0 += max_jj * 6 * 4;
                r1 += max_jj * 6 * 4;
                r2 += max_jj * 6 * 4;
                r3 += max_jj * 6 * 4;
                r4 += max_jj * 6 * 4;
                r5 += max_jj * 6 * 4;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 4) + (tj * 4) * out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                __m128 _r0 = (__m128)__lsx_vld(tmp[m][0], 0);
                __m128 _r1 = (__m128)__lsx_vld(tmp[m][1], 0);
                __m128 _r2 = (__m128)__lsx_vld(tmp[m][2], 0);
                __m128 _r3 = (__m128)__lsx_vld(tmp[m][3], 0);
                __m128 _r4 = (__m128)__lsx_vld(tmp[m][4], 0);
                __m128 _r5 = (__m128)__lsx_vld(tmp[m][5], 0);

                __m128 _tmp02a = __lsx_vfadd_s(_r1, _r2);
                __m128 _tmp02b = __lsx_vfadd_s(_r3, _r4);
                __m128 _tmp13a = __lsx_vfsub_s(_r1, _r2);
                __m128 _tmp13b = __lsx_vfsub_s(_r3, _r4);

                __m128 _tmp0 = __lsx_vfadd_s(__lsx_vfadd_s(_r0, _tmp02a), __lsx_vfadd_s(_tmp02b, _bias0));
                __m128 _tmp1 = __lsx_vfmadd_s(_tmp13b, _vsq2, __lsx_vfmadd_s(_tmp13a, _vsq2_d2, _bias0));
                __m128 _tmp2 = __lsx_vfmadd_s(_tmp02b, _v2, __lsx_vfmadd_s(_tmp02a, _v0_5, _bias0));
                __m128 _tmp3 = __lsx_vfmadd_s(_tmp13b, _vsq2_m2, __lsx_vfmadd_s(_tmp13a, _vsq2_d4, __lsx_vfadd_s(_r5, _bias0)));

                if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_tmp0, outptr0, 0);
                    if (tj * 4 + 1 < outw) __lsx_vst((__m128i)_tmp1, outptr0 + 4, 0);
                    if (tj * 4 + 2 < outw) __lsx_vst((__m128i)_tmp2, outptr0 + 8, 0);
                    if (tj * 4 + 3 < outw) __lsx_vst((__m128i)_tmp3, outptr0 + 12, 0);
                }
                if (out_elempack == 1)
                {
                    float tmp0[4];
                    float tmp1[4];
                    float tmp2[4];
                    float tmp3[4];
                    __lsx_vst((__m128i)_tmp0, tmp0, 0);
                    __lsx_vst((__m128i)_tmp1, tmp1, 0);
                    __lsx_vst((__m128i)_tmp2, tmp2, 0);
                    __lsx_vst((__m128i)_tmp3, tmp3, 0);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;

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
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;
        float bias1 = biasptr ? biasptr[i + ii + 1] : 0.f;

        float tmp[4][6][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 36 + jj * 2;
            const float* r1 = r0 + max_jj * 2;
            const float* r2 = r0 + max_jj * 2 * 2;
            const float* r3 = r0 + max_jj * 2 * 3;
            const float* r4 = r0 + max_jj * 2 * 4;
            const float* r5 = r0 + max_jj * 2 * 5;

            for (int m = 0; m < 6; m++)
            {
                float tmp02a0 = r1[0] + r2[0];
                float tmp02a1 = r1[1] + r2[1];
                float tmp02b0 = r3[0] + r4[0];
                float tmp02b1 = r3[1] + r4[1];
                float tmp13a0 = r1[0] - r2[0];
                float tmp13a1 = r1[1] - r2[1];
                float tmp13b0 = r3[0] - r4[0];
                float tmp13b1 = r3[1] - r4[1];

                tmp[0][m][0] = r0[0] + tmp02a0 + tmp02b0;
                tmp[0][m][1] = r0[1] + tmp02a1 + tmp02b1;
                tmp[1][m][0] = tmp13a0 * sq2_d2 + tmp13b0 * sq2;
                tmp[1][m][1] = tmp13a1 * sq2_d2 + tmp13b1 * sq2;
                tmp[2][m][0] = tmp02a0 * 0.5f + tmp02b0 * 2;
                tmp[2][m][1] = tmp02a1 * 0.5f + tmp02b1 * 2;
                tmp[3][m][0] = r5[0] + tmp13a0 * sq2_d4 + tmp13b0 * sq2_m2;
                tmp[3][m][1] = r5[1] + tmp13a1 * sq2_d4 + tmp13b1 * sq2_m2;

                r0 += max_jj * 6 * 2;
                r1 += max_jj * 6 * 2;
                r2 += max_jj * 6 * 2;
                r3 += max_jj * 6 * 2;
                r4 += max_jj * 6 * 2;
                r5 += max_jj * 6 * 2;
            }

            float* outptr0 = top_blob.channel(i + ii).row(ti * 4) + (tj * 4);

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];
                float r40 = tmp[m][4][0];
                float r41 = tmp[m][4][1];
                float r50 = tmp[m][5][0];
                float r51 = tmp[m][5][1];

                float tmp02a0 = r10 + r20;
                float tmp02a1 = r11 + r21;
                float tmp02b0 = r30 + r40;
                float tmp02b1 = r31 + r41;
                float tmp13a0 = r10 - r20;
                float tmp13a1 = r11 - r21;
                float tmp13b0 = r30 - r40;
                float tmp13b1 = r31 - r41;

                float tmp00 = bias0 + r00 + tmp02a0 + tmp02b0;
                float tmp01 = bias1 + r01 + tmp02a1 + tmp02b1;
                float tmp10 = bias0 + tmp13a0 * sq2_d2 + tmp13b0 * sq2;
                float tmp11 = bias1 + tmp13a1 * sq2_d2 + tmp13b1 * sq2;
                float tmp20 = bias0 + tmp02a0 * 0.5f + tmp02b0 * 2;
                float tmp21 = bias1 + tmp02a1 * 0.5f + tmp02b1 * 2;
                float tmp30 = bias0 + r50 + tmp13a0 * sq2_d4 + tmp13b0 * sq2_m2;
                float tmp31 = bias1 + r51 + tmp13a1 * sq2_d4 + tmp13b1 * sq2_m2;

                // if (out_elempack == 1)
                {
                    float* outptr1 = outptr0 + N;

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
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;

        float tmp[4][6];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 36 + jj;
            const float* r1 = r0 + max_jj;
            const float* r2 = r0 + max_jj * 2;
            const float* r3 = r0 + max_jj * 3;
            const float* r4 = r0 + max_jj * 4;
            const float* r5 = r0 + max_jj * 5;

            for (int m = 0; m < 6; m++)
            {
                float tmp02a = r1[0] + r2[0];
                float tmp02b = r3[0] + r4[0];
                float tmp13a = r1[0] - r2[0];
                float tmp13b = r3[0] - r4[0];

                tmp[0][m] = r0[0] + tmp02a + tmp02b;
                tmp[1][m] = tmp13a * sq2_d2 + tmp13b * sq2;
                tmp[2][m] = tmp02a * 0.5f + tmp02b * 2;
                tmp[3][m] = r5[0] + tmp13a * sq2_d4 + tmp13b * sq2_m2;

                r0 += max_jj * 6;
                r1 += max_jj * 6;
                r2 += max_jj * 6;
                r3 += max_jj * 6;
                r4 += max_jj * 6;
                r5 += max_jj * 6;
            }

            float* outptr0 = top_blob.channel(i + ii).row(ti * 4) + (tj * 4);

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];
                float r4 = tmp[m][4];
                float r5 = tmp[m][5];

                float tmp02a = r1 + r2;
                float tmp02b = r3 + r4;
                float tmp13a = r1 - r2;
                float tmp13b = r3 - r4;

                float tmp0 = bias0 + r0 + tmp02a + tmp02b;
                float tmp1 = bias0 + tmp13a * sq2_d2 + tmp13b * sq2;
                float tmp2 = bias0 + tmp02a * 0.5f + tmp02b * 2;
                float tmp3 = bias0 + r5 + tmp13a * sq2_d4 + tmp13b * sq2_m2;

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

static int conv3x3s1_winograd43(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, const Option& opt)
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

    // NCNN_LOGE("conv3x3s1_winograd43 %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, TILE_M, TILE_N, TILE_K, nT);

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
            conv3x3s1_winograd43_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, nT);
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
            conv3x3s1_winograd43_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, 1);
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

                gemm_transB_packed_tile(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk);
            }

            // transform output
            conv3x3s1_winograd43_transform_output_tile(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

static inline void conv3x3s1_winograd63_transform_kernel_tile(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    float* ptmp = A;

    int ii = 0;
    for (; ii < max_ii; ii++)
    {
        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            // const float ktm[8][3] = {
            //     {1.0f, 0.0f, 0.0f},
            //     {-2.0f / 9, -2.0f / 9, -2.0f / 9},
            //     {-2.0f / 9, 2.0f / 9, -2.0f / 9},
            //     {1.0f / 90, 1.0f / 45, 2.0f / 45},
            //     {1.0f / 90, -1.0f / 45, 2.0f / 45},
            //     {1.0f / 45, 1.0f / 90, 1.0f / 180},
            //     {1.0f / 45, -1.0f / 90, 1.0f / 180},
            //     {0.0f, 0.0f, 1.0f}
            // };
            const float ktm0 = 2.0f / 9;
            const float ktm1 = 1.0f / 45;
            const float ktm2 = 2.0f / 45;
            const float ktm3 = 1.0f / 90;
            const float ktm4 = 1.0f / 180;

            float tmp[8][3];

            const float* k0 = (const float*)kernel + (i + ii) * inch * 9 + (k + kk) * 9;

            for (int m = 0; m < 3; m++)
            {
                float r0 = k0[0];
                float r1 = k0[1];
                float r2 = k0[2];

                tmp[0][m] = r0;
                tmp[1][m] = -r0 * ktm0 - r1 * ktm0 - r2 * ktm0;
                tmp[2][m] = -r0 * ktm0 + r1 * ktm0 - r2 * ktm0;
                tmp[3][m] = r0 * ktm3 + r1 * ktm1 + r2 * ktm2;
                tmp[4][m] = r0 * ktm3 - r1 * ktm1 + r2 * ktm2;
                tmp[5][m] = r0 * ktm1 + r1 * ktm3 + r2 * ktm4;
                tmp[6][m] = r0 * ktm1 - r1 * ktm3 + r2 * ktm4;
                tmp[7][m] = r2;

                k0 += 3;
            }

            for (int m = 0; m < 8; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];

                float z0 = r0;
                float z1 = -r0 * ktm0 - r1 * ktm0 - r2 * ktm0;
                float z2 = -r0 * ktm0 + r1 * ktm0 - r2 * ktm0;
                float z3 = r0 * ktm3 + r1 * ktm1 + r2 * ktm2;
                float z4 = r0 * ktm3 - r1 * ktm1 + r2 * ktm2;
                float z5 = r0 * ktm1 + r1 * ktm3 + r2 * ktm4;
                float z6 = r0 * ktm1 - r1 * ktm3 + r2 * ktm4;
                float z7 = r2;

                ptmp[0] = z0;
                ptmp[1] = z1;
                ptmp[2] = z2;
                ptmp[3] = z3;
                ptmp[4] = z4;
                ptmp[5] = z5;
                ptmp[6] = z6;
                ptmp[7] = z7;
                ptmp += 8;
            }
        }
    }
}

static void conv3x3s1_winograd63_transform_kernel(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 64;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

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

            conv3x3s1_winograd63_transform_kernel_tile(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            pack_A_tile(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd63_transform_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    // const float itm[8][8] = {
    //     {1.0f, 0.0f,-5.25f, 0.00f, 5.25f, 0.00f,-1.0f, 0.0f},
    //     {0.0f, 1.0f, 1.00f,-4.25f,-4.25f, 1.00f, 1.0f, 0.0f},
    //     {0.0f,-1.0f, 1.00f, 4.25f,-4.25f,-1.00f, 1.0f, 0.0f},
    //     {0.0f, 0.5f, 0.25f,-2.50f,-1.25f, 2.00f, 1.0f, 0.0f},
    //     {0.0f,-0.5f, 0.25f, 2.50f,-1.25f,-2.00f, 1.0f, 0.0f},
    //     {0.0f, 2.0f, 4.00f,-2.50f,-5.00f, 0.50f, 1.0f, 0.0f},
    //     {0.0f,-2.0f, 4.00f, 2.50f,-5.00f,-0.50f, 1.0f, 0.0f},
    //     {0.0f,-1.0f, 0.00f, 5.25f, 0.00f,-5.25f, 0.0f, 1.0f}
    // };

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const size_t N = bottom_blob.cstep * elempack;

    const int w_tiles = (w + 3) / 6;

    int nn_max_kk = 0;
    int remain_max_kk_start = 0;
#if __loongarch_sx
#if __loongarch_asx
    nn_max_kk = max_kk / 8;
    #pragma omp parallel for num_threads(nT)
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 8;

        __attribute__((aligned(32))) float tmp[8][8][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 6) + (tj * 6) * elempack;

            for (int m = 0; m < 8; m++)
            {
                __m256 _r0 = (__m256)__lasx_xvldi(0);
                __m256 _r1 = (__m256)__lasx_xvldi(0);
                __m256 _r2 = (__m256)__lasx_xvldi(0);
                __m256 _r3 = (__m256)__lasx_xvldi(0);
                __m256 _r4 = (__m256)__lasx_xvldi(0);
                __m256 _r5 = (__m256)__lasx_xvldi(0);
                __m256 _r6 = (__m256)__lasx_xvldi(0);
                __m256 _r7 = (__m256)__lasx_xvldi(0);

                if (ti * 6 + m < h)
                {
                    if (elempack == 8)
                    {
                        _r0 = (__m256)__lasx_xvld(r0, 0);
                        if (tj * 6 + 1 < w) _r1 = (__m256)__lasx_xvld(r0 + 8, 0);
                        if (tj * 6 + 2 < w) _r2 = (__m256)__lasx_xvld(r0 + 16, 0);
                        if (tj * 6 + 3 < w) _r3 = (__m256)__lasx_xvld(r0 + 24, 0);
                        if (tj * 6 + 4 < w) _r4 = (__m256)__lasx_xvld(r0 + 32, 0);
                        if (tj * 6 + 5 < w) _r5 = (__m256)__lasx_xvld(r0 + 40, 0);
                        if (tj * 6 + 6 < w) _r6 = (__m256)__lasx_xvld(r0 + 48, 0);
                        if (tj * 6 + 7 < w) _r7 = (__m256)__lasx_xvld(r0 + 56, 0);
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;

                        _r0 = (__m256)__lasx_concat_128((__m128i)(__m128)__lsx_vld(r0, 0), (__m128i)(__m128)__lsx_vld(r1, 0));
                        if (tj * 6 + 1 < w) _r1 = (__m256)__lasx_concat_128((__m128i)(__m128)__lsx_vld(r0 + 4, 0), (__m128i)(__m128)__lsx_vld(r1 + 4, 0));
                        if (tj * 6 + 2 < w) _r2 = (__m256)__lasx_concat_128((__m128i)(__m128)__lsx_vld(r0 + 8, 0), (__m128i)(__m128)__lsx_vld(r1 + 8, 0));
                        if (tj * 6 + 3 < w) _r3 = (__m256)__lasx_concat_128((__m128i)(__m128)__lsx_vld(r0 + 12, 0), (__m128i)(__m128)__lsx_vld(r1 + 12, 0));
                        if (tj * 6 + 4 < w) _r4 = (__m256)__lasx_concat_128((__m128i)(__m128)__lsx_vld(r0 + 16, 0), (__m128i)(__m128)__lsx_vld(r1 + 16, 0));
                        if (tj * 6 + 5 < w) _r5 = (__m256)__lasx_concat_128((__m128i)(__m128)__lsx_vld(r0 + 20, 0), (__m128i)(__m128)__lsx_vld(r1 + 20, 0));
                        if (tj * 6 + 6 < w) _r6 = (__m256)__lasx_concat_128((__m128i)(__m128)__lsx_vld(r0 + 24, 0), (__m128i)(__m128)__lsx_vld(r1 + 24, 0));
                        if (tj * 6 + 7 < w) _r7 = (__m256)__lasx_concat_128((__m128i)(__m128)__lsx_vld(r0 + 28, 0), (__m128i)(__m128)__lsx_vld(r1 + 28, 0));
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;
                        const float* r4 = r0 + N * 4;
                        const float* r5 = r0 + N * 5;
                        const float* r6 = r0 + N * 6;
                        const float* r7 = r0 + N * 7;

                        __m128 _t0 = (__m128)__lsx_vld(r0, 0);
                        __m128 _t1 = (__m128)__lsx_vld(r1, 0);
                        __m128 _t2 = (__m128)__lsx_vld(r2, 0);
                        __m128 _t3 = (__m128)__lsx_vld(r3, 0);
                        __m128 _t4 = (__m128)__lsx_vld(r4, 0);
                        __m128 _t5 = (__m128)__lsx_vld(r5, 0);
                        __m128 _t6 = (__m128)__lsx_vld(r6, 0);
                        __m128 _t7 = (__m128)__lsx_vld(r7, 0);

                        {
                            __m128i _01l = __lsx_vilvl_w((__m128i)_t1, (__m128i)_t0);
                            __m128i _01h = __lsx_vilvh_w((__m128i)_t1, (__m128i)_t0);
                            __m128i _23l = __lsx_vilvl_w((__m128i)_t3, (__m128i)_t2);
                            __m128i _23h = __lsx_vilvh_w((__m128i)_t3, (__m128i)_t2);
                            _t0 = (__m128)__lsx_vilvl_d(_23l, _01l);
                            _t1 = (__m128)__lsx_vilvh_d(_23l, _01l);
                            _t2 = (__m128)__lsx_vilvl_d(_23h, _01h);
                            _t3 = (__m128)__lsx_vilvh_d(_23h, _01h);
                        }
                        {
                            __m128i _01l = __lsx_vilvl_w((__m128i)_t5, (__m128i)_t4);
                            __m128i _01h = __lsx_vilvh_w((__m128i)_t5, (__m128i)_t4);
                            __m128i _23l = __lsx_vilvl_w((__m128i)_t7, (__m128i)_t6);
                            __m128i _23h = __lsx_vilvh_w((__m128i)_t7, (__m128i)_t6);
                            _t4 = (__m128)__lsx_vilvl_d(_23l, _01l);
                            _t5 = (__m128)__lsx_vilvh_d(_23l, _01l);
                            _t6 = (__m128)__lsx_vilvl_d(_23h, _01h);
                            _t7 = (__m128)__lsx_vilvh_d(_23h, _01h);
                        }

                        _r0 = (__m256)__lasx_concat_128((__m128i)_t0, (__m128i)_t4);
                        if (tj * 6 + 1 < w) _r1 = (__m256)__lasx_concat_128((__m128i)_t1, (__m128i)_t5);
                        if (tj * 6 + 2 < w) _r2 = (__m256)__lasx_concat_128((__m128i)_t2, (__m128i)_t6);
                        if (tj * 6 + 3 < w) _r3 = (__m256)__lasx_concat_128((__m128i)_t3, (__m128i)_t7);
                        if (tj * 6 + 4 < w)
                        {
                            _t0 = (__m128)__lsx_vld(r0 + 4, 0);
                            _t1 = (__m128)__lsx_vld(r1 + 4, 0);
                            _t2 = (__m128)__lsx_vld(r2 + 4, 0);
                            _t3 = (__m128)__lsx_vld(r3 + 4, 0);
                            _t4 = (__m128)__lsx_vld(r4 + 4, 0);
                            _t5 = (__m128)__lsx_vld(r5 + 4, 0);
                            _t6 = (__m128)__lsx_vld(r6 + 4, 0);
                            _t7 = (__m128)__lsx_vld(r7 + 4, 0);

                            {
                                __m128i _01l = __lsx_vilvl_w((__m128i)_t1, (__m128i)_t0);
                                __m128i _01h = __lsx_vilvh_w((__m128i)_t1, (__m128i)_t0);
                                __m128i _23l = __lsx_vilvl_w((__m128i)_t3, (__m128i)_t2);
                                __m128i _23h = __lsx_vilvh_w((__m128i)_t3, (__m128i)_t2);
                                _t0 = (__m128)__lsx_vilvl_d(_23l, _01l);
                                _t1 = (__m128)__lsx_vilvh_d(_23l, _01l);
                                _t2 = (__m128)__lsx_vilvl_d(_23h, _01h);
                                _t3 = (__m128)__lsx_vilvh_d(_23h, _01h);
                            }
                            {
                                __m128i _01l = __lsx_vilvl_w((__m128i)_t5, (__m128i)_t4);
                                __m128i _01h = __lsx_vilvh_w((__m128i)_t5, (__m128i)_t4);
                                __m128i _23l = __lsx_vilvl_w((__m128i)_t7, (__m128i)_t6);
                                __m128i _23h = __lsx_vilvh_w((__m128i)_t7, (__m128i)_t6);
                                _t4 = (__m128)__lsx_vilvl_d(_23l, _01l);
                                _t5 = (__m128)__lsx_vilvh_d(_23l, _01l);
                                _t6 = (__m128)__lsx_vilvl_d(_23h, _01h);
                                _t7 = (__m128)__lsx_vilvh_d(_23h, _01h);
                            }

                            _r4 = (__m256)__lasx_concat_128((__m128i)_t0, (__m128i)_t4);
                            if (tj * 6 + 5 < w) _r5 = (__m256)__lasx_concat_128((__m128i)_t1, (__m128i)_t5);
                            if (tj * 6 + 6 < w) _r6 = (__m256)__lasx_concat_128((__m128i)_t2, (__m128i)_t6);
                            if (tj * 6 + 7 < w) _r7 = (__m256)__lasx_concat_128((__m128i)_t3, (__m128i)_t7);
                        }
                    }
                }

                __m256 _v5_25 = (__m256)__lasx_xvreplfr2vr_s(5.25f);
                __m256 _vm4_25 = (__m256)__lasx_xvreplfr2vr_s(-4.25f);
                __m256 _vm1_25 = (__m256)__lasx_xvreplfr2vr_s(-1.25f);
                __m256 _v0_25 = (__m256)__lasx_xvreplfr2vr_s(0.25f);
                __m256 _vm2_5 = (__m256)__lasx_xvreplfr2vr_s(-2.5f);
                __m256 _v0_5 = (__m256)__lasx_xvreplfr2vr_s(0.5f);
                __m256 _v2 = (__m256)__lasx_xvreplfr2vr_s(2.f);
                __m256 _v4 = (__m256)__lasx_xvreplfr2vr_s(4.f);

                __m256 _tmp12a = __lasx_xvfmadd_s(_vm4_25, _r4, __lasx_xvfadd_s(_r2, _r6));
                __m256 _tmp12b = __lasx_xvfmadd_s(_vm4_25, _r3, __lasx_xvfadd_s(_r1, _r5));
                __m256 _tmp34a = __lasx_xvfmadd_s(_vm1_25, _r4, __lasx_xvfmadd_s(_v0_25, _r2, _r6));
                __m256 _tmp34b = __lasx_xvfmadd_s(_v2, _r5, __lasx_xvfmadd_s(_vm2_5, _r3, __lasx_xvfmul_s(_r1, _v0_5)));
                __m256 _tmp56a = __lasx_xvfmadd_s(_v4, __lasx_xvfmadd_s(_vm1_25, _r4, _r2), _r6);
                __m256 _tmp56b = __lasx_xvfmadd_s(_v0_5, _r5, __lasx_xvfmadd_s(_vm2_5, _r3, __lasx_xvfmul_s(_r1, _v2)));

                __m256 _tmp0 = __lasx_xvfmadd_s(_v5_25, __lasx_xvfsub_s(_r4, _r2), __lasx_xvfsub_s(_r0, _r6));
                __m256 _tmp1 = __lasx_xvfadd_s(_tmp12a, _tmp12b);
                __m256 _tmp2 = __lasx_xvfsub_s(_tmp12a, _tmp12b);
                __m256 _tmp3 = __lasx_xvfadd_s(_tmp34a, _tmp34b);
                __m256 _tmp4 = __lasx_xvfsub_s(_tmp34a, _tmp34b);
                __m256 _tmp5 = __lasx_xvfadd_s(_tmp56a, _tmp56b);
                __m256 _tmp6 = __lasx_xvfsub_s(_tmp56a, _tmp56b);
                __m256 _tmp7 = __lasx_xvfmadd_s(_v5_25, __lasx_xvfsub_s(_r3, _r5), __lasx_xvfsub_s(_r7, _r1));

                __lasx_xvst((__m256i)_tmp0, tmp[0][m], 0);
                __lasx_xvst((__m256i)_tmp1, tmp[1][m], 0);
                __lasx_xvst((__m256i)_tmp2, tmp[2][m], 0);
                __lasx_xvst((__m256i)_tmp3, tmp[3][m], 0);
                __lasx_xvst((__m256i)_tmp4, tmp[4][m], 0);
                __lasx_xvst((__m256i)_tmp5, tmp[5][m], 0);
                __lasx_xvst((__m256i)_tmp6, tmp[6][m], 0);
                __lasx_xvst((__m256i)_tmp7, tmp[7][m], 0);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 64 + jj * 8;
            float* p1 = p0 + max_jj * 8;
            float* p2 = p0 + max_jj * 8 * 2;
            float* p3 = p0 + max_jj * 8 * 3;
            float* p4 = p0 + max_jj * 8 * 4;
            float* p5 = p0 + max_jj * 8 * 5;
            float* p6 = p0 + max_jj * 8 * 6;
            float* p7 = p0 + max_jj * 8 * 7;

            for (int m = 0; m < 8; m++)
            {
                __m256 _r0 = (__m256)__lasx_xvld(tmp[m][0], 0);
                __m256 _r1 = (__m256)__lasx_xvld(tmp[m][1], 0);
                __m256 _r2 = (__m256)__lasx_xvld(tmp[m][2], 0);
                __m256 _r3 = (__m256)__lasx_xvld(tmp[m][3], 0);
                __m256 _r4 = (__m256)__lasx_xvld(tmp[m][4], 0);
                __m256 _r5 = (__m256)__lasx_xvld(tmp[m][5], 0);
                __m256 _r6 = (__m256)__lasx_xvld(tmp[m][6], 0);
                __m256 _r7 = (__m256)__lasx_xvld(tmp[m][7], 0);

                __m256 _v5_25 = (__m256)__lasx_xvreplfr2vr_s(5.25f);
                __m256 _vm4_25 = (__m256)__lasx_xvreplfr2vr_s(-4.25f);
                __m256 _vm1_25 = (__m256)__lasx_xvreplfr2vr_s(-1.25f);
                __m256 _v0_25 = (__m256)__lasx_xvreplfr2vr_s(0.25f);
                __m256 _vm2_5 = (__m256)__lasx_xvreplfr2vr_s(-2.5f);
                __m256 _v0_5 = (__m256)__lasx_xvreplfr2vr_s(0.5f);
                __m256 _v2 = (__m256)__lasx_xvreplfr2vr_s(2.f);
                __m256 _v4 = (__m256)__lasx_xvreplfr2vr_s(4.f);

                __m256 _tmp12a = __lasx_xvfmadd_s(_vm4_25, _r4, __lasx_xvfadd_s(_r2, _r6));
                __m256 _tmp12b = __lasx_xvfmadd_s(_vm4_25, _r3, __lasx_xvfadd_s(_r1, _r5));
                __m256 _tmp34a = __lasx_xvfmadd_s(_vm1_25, _r4, __lasx_xvfmadd_s(_v0_25, _r2, _r6));
                __m256 _tmp34b = __lasx_xvfmadd_s(_v2, _r5, __lasx_xvfmadd_s(_vm2_5, _r3, __lasx_xvfmul_s(_r1, _v0_5)));
                __m256 _tmp56a = __lasx_xvfmadd_s(_v4, __lasx_xvfmadd_s(_vm1_25, _r4, _r2), _r6);
                __m256 _tmp56b = __lasx_xvfmadd_s(_v0_5, _r5, __lasx_xvfmadd_s(_vm2_5, _r3, __lasx_xvfmul_s(_r1, _v2)));

                __m256 _tmp0 = __lasx_xvfmadd_s(_v5_25, __lasx_xvfsub_s(_r4, _r2), __lasx_xvfsub_s(_r0, _r6));
                __m256 _tmp1 = __lasx_xvfadd_s(_tmp12a, _tmp12b);
                __m256 _tmp2 = __lasx_xvfsub_s(_tmp12a, _tmp12b);
                __m256 _tmp3 = __lasx_xvfadd_s(_tmp34a, _tmp34b);
                __m256 _tmp4 = __lasx_xvfsub_s(_tmp34a, _tmp34b);
                __m256 _tmp5 = __lasx_xvfadd_s(_tmp56a, _tmp56b);
                __m256 _tmp6 = __lasx_xvfsub_s(_tmp56a, _tmp56b);
                __m256 _tmp7 = __lasx_xvfmadd_s(_v5_25, __lasx_xvfsub_s(_r3, _r5), __lasx_xvfsub_s(_r7, _r1));

                __lasx_xvst((__m256i)_tmp0, p0, 0);
                __lasx_xvst((__m256i)_tmp1, p1, 0);
                __lasx_xvst((__m256i)_tmp2, p2, 0);
                __lasx_xvst((__m256i)_tmp3, p3, 0);
                __lasx_xvst((__m256i)_tmp4, p4, 0);
                __lasx_xvst((__m256i)_tmp5, p5, 0);
                __lasx_xvst((__m256i)_tmp6, p6, 0);
                __lasx_xvst((__m256i)_tmp7, p7, 0);

                p0 += max_jj * 8 * 8;
                p1 += max_jj * 8 * 8;
                p2 += max_jj * 8 * 8;
                p3 += max_jj * 8 * 8;
                p4 += max_jj * 8 * 8;
                p5 += max_jj * 8 * 8;
                p6 += max_jj * 8 * 8;
                p7 += max_jj * 8 * 8;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 8;
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
    #pragma omp parallel for num_threads(nT)
#else // __loongarch_asx
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
    #pragma omp parallel for num_threads(nT)
#endif // __loongarch_asx
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 4;

        __attribute__((aligned(16))) float tmp[8][8][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel((k + kk) / elempack).row(ti * 6) + (tj * 6) * elempack;

            for (int m = 0; m < 8; m++)
            {
                __m128 _r0 = (__m128)__lsx_vldi(0);
                __m128 _r1 = (__m128)__lsx_vldi(0);
                __m128 _r2 = (__m128)__lsx_vldi(0);
                __m128 _r3 = (__m128)__lsx_vldi(0);
                __m128 _r4 = (__m128)__lsx_vldi(0);
                __m128 _r5 = (__m128)__lsx_vldi(0);
                __m128 _r6 = (__m128)__lsx_vldi(0);
                __m128 _r7 = (__m128)__lsx_vldi(0);

                if (ti * 6 + m < h)
                {
                    if (elempack == 4)
                    {
                        _r0 = (__m128)__lsx_vld(r0, 0);
                        if (tj * 6 + 1 < w) _r1 = (__m128)__lsx_vld(r0 + 4, 0);
                        if (tj * 6 + 2 < w) _r2 = (__m128)__lsx_vld(r0 + 8, 0);
                        if (tj * 6 + 3 < w) _r3 = (__m128)__lsx_vld(r0 + 12, 0);
                        if (tj * 6 + 4 < w) _r4 = (__m128)__lsx_vld(r0 + 16, 0);
                        if (tj * 6 + 5 < w) _r5 = (__m128)__lsx_vld(r0 + 20, 0);
                        if (tj * 6 + 6 < w) _r6 = (__m128)__lsx_vld(r0 + 24, 0);
                        if (tj * 6 + 7 < w) _r7 = (__m128)__lsx_vld(r0 + 28, 0);
                    }
                    if (elempack == 1)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;

                        __m128 _t0 = (__m128)__lsx_vld(r0, 0);
                        __m128 _t1 = (__m128)__lsx_vld(r1, 0);
                        __m128 _t2 = (__m128)__lsx_vld(r2, 0);
                        __m128 _t3 = (__m128)__lsx_vld(r3, 0);

                        {
                            __m128i _01l = __lsx_vilvl_w((__m128i)_t1, (__m128i)_t0);
                            __m128i _01h = __lsx_vilvh_w((__m128i)_t1, (__m128i)_t0);
                            __m128i _23l = __lsx_vilvl_w((__m128i)_t3, (__m128i)_t2);
                            __m128i _23h = __lsx_vilvh_w((__m128i)_t3, (__m128i)_t2);
                            _t0 = (__m128)__lsx_vilvl_d(_23l, _01l);
                            _t1 = (__m128)__lsx_vilvh_d(_23l, _01l);
                            _t2 = (__m128)__lsx_vilvl_d(_23h, _01h);
                            _t3 = (__m128)__lsx_vilvh_d(_23h, _01h);
                        }

                        _r0 = _t0;
                        if (tj * 6 + 1 < w) _r1 = _t1;
                        if (tj * 6 + 2 < w) _r2 = _t2;
                        if (tj * 6 + 3 < w) _r3 = _t3;
                        if (tj * 6 + 4 < w)
                        {
                            _t0 = (__m128)__lsx_vld(r0 + 4, 0);
                            _t1 = (__m128)__lsx_vld(r1 + 4, 0);
                            _t2 = (__m128)__lsx_vld(r2 + 4, 0);
                            _t3 = (__m128)__lsx_vld(r3 + 4, 0);

                            {
                                __m128i _01l = __lsx_vilvl_w((__m128i)_t1, (__m128i)_t0);
                                __m128i _01h = __lsx_vilvh_w((__m128i)_t1, (__m128i)_t0);
                                __m128i _23l = __lsx_vilvl_w((__m128i)_t3, (__m128i)_t2);
                                __m128i _23h = __lsx_vilvh_w((__m128i)_t3, (__m128i)_t2);
                                _t0 = (__m128)__lsx_vilvl_d(_23l, _01l);
                                _t1 = (__m128)__lsx_vilvh_d(_23l, _01l);
                                _t2 = (__m128)__lsx_vilvl_d(_23h, _01h);
                                _t3 = (__m128)__lsx_vilvh_d(_23h, _01h);
                            }

                            _r4 = _t0;
                            if (tj * 6 + 5 < w) _r5 = _t1;
                            if (tj * 6 + 6 < w) _r6 = _t2;
                            if (tj * 6 + 7 < w) _r7 = _t3;
                        }
                    }
                }

                __m128 _v5_25 = (__m128)__lsx_vreplfr2vr_s(5.25f);
                __m128 _vm4_25 = (__m128)__lsx_vreplfr2vr_s(-4.25f);
                __m128 _vm1_25 = (__m128)__lsx_vreplfr2vr_s(-1.25f);
                __m128 _v0_25 = (__m128)__lsx_vreplfr2vr_s(0.25f);
                __m128 _vm2_5 = (__m128)__lsx_vreplfr2vr_s(-2.5f);
                __m128 _v0_5 = (__m128)__lsx_vreplfr2vr_s(0.5f);
                __m128 _v2 = (__m128)__lsx_vreplfr2vr_s(2.f);
                __m128 _v4 = (__m128)__lsx_vreplfr2vr_s(4.f);

                __m128 _tmp12a = __lsx_vfmadd_s(_vm4_25, _r4, __lsx_vfadd_s(_r2, _r6));
                __m128 _tmp12b = __lsx_vfmadd_s(_vm4_25, _r3, __lsx_vfadd_s(_r1, _r5));
                __m128 _tmp34a = __lsx_vfmadd_s(_vm1_25, _r4, __lsx_vfmadd_s(_v0_25, _r2, _r6));
                __m128 _tmp34b = __lsx_vfmadd_s(_v2, _r5, __lsx_vfmadd_s(_vm2_5, _r3, __lsx_vfmul_s(_r1, _v0_5)));
                __m128 _tmp56a = __lsx_vfmadd_s(_v4, __lsx_vfmadd_s(_vm1_25, _r4, _r2), _r6);
                __m128 _tmp56b = __lsx_vfmadd_s(_v0_5, _r5, __lsx_vfmadd_s(_vm2_5, _r3, __lsx_vfmul_s(_r1, _v2)));

                __m128 _tmp0 = __lsx_vfmadd_s(_v5_25, __lsx_vfsub_s(_r4, _r2), __lsx_vfsub_s(_r0, _r6));
                __m128 _tmp1 = __lsx_vfadd_s(_tmp12a, _tmp12b);
                __m128 _tmp2 = __lsx_vfsub_s(_tmp12a, _tmp12b);
                __m128 _tmp3 = __lsx_vfadd_s(_tmp34a, _tmp34b);
                __m128 _tmp4 = __lsx_vfsub_s(_tmp34a, _tmp34b);
                __m128 _tmp5 = __lsx_vfadd_s(_tmp56a, _tmp56b);
                __m128 _tmp6 = __lsx_vfsub_s(_tmp56a, _tmp56b);
                __m128 _tmp7 = __lsx_vfmadd_s(_v5_25, __lsx_vfsub_s(_r3, _r5), __lsx_vfsub_s(_r7, _r1));

                __lsx_vst((__m128i)_tmp0, tmp[0][m], 0);
                __lsx_vst((__m128i)_tmp1, tmp[1][m], 0);
                __lsx_vst((__m128i)_tmp2, tmp[2][m], 0);
                __lsx_vst((__m128i)_tmp3, tmp[3][m], 0);
                __lsx_vst((__m128i)_tmp4, tmp[4][m], 0);
                __lsx_vst((__m128i)_tmp5, tmp[5][m], 0);
                __lsx_vst((__m128i)_tmp6, tmp[6][m], 0);
                __lsx_vst((__m128i)_tmp7, tmp[7][m], 0);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 64 + jj * 4;
            float* p1 = p0 + max_jj * 4;
            float* p2 = p0 + max_jj * 4 * 2;
            float* p3 = p0 + max_jj * 4 * 3;
            float* p4 = p0 + max_jj * 4 * 4;
            float* p5 = p0 + max_jj * 4 * 5;
            float* p6 = p0 + max_jj * 4 * 6;
            float* p7 = p0 + max_jj * 4 * 7;

            for (int m = 0; m < 8; m++)
            {
                __m128 _r0 = (__m128)__lsx_vld(tmp[m][0], 0);
                __m128 _r1 = (__m128)__lsx_vld(tmp[m][1], 0);
                __m128 _r2 = (__m128)__lsx_vld(tmp[m][2], 0);
                __m128 _r3 = (__m128)__lsx_vld(tmp[m][3], 0);
                __m128 _r4 = (__m128)__lsx_vld(tmp[m][4], 0);
                __m128 _r5 = (__m128)__lsx_vld(tmp[m][5], 0);
                __m128 _r6 = (__m128)__lsx_vld(tmp[m][6], 0);
                __m128 _r7 = (__m128)__lsx_vld(tmp[m][7], 0);

                __m128 _v5_25 = (__m128)__lsx_vreplfr2vr_s(5.25f);
                __m128 _vm4_25 = (__m128)__lsx_vreplfr2vr_s(-4.25f);
                __m128 _vm1_25 = (__m128)__lsx_vreplfr2vr_s(-1.25f);
                __m128 _v0_25 = (__m128)__lsx_vreplfr2vr_s(0.25f);
                __m128 _vm2_5 = (__m128)__lsx_vreplfr2vr_s(-2.5f);
                __m128 _v0_5 = (__m128)__lsx_vreplfr2vr_s(0.5f);
                __m128 _v2 = (__m128)__lsx_vreplfr2vr_s(2.f);
                __m128 _v4 = (__m128)__lsx_vreplfr2vr_s(4.f);

                __m128 _tmp12a = __lsx_vfmadd_s(_vm4_25, _r4, __lsx_vfadd_s(_r2, _r6));
                __m128 _tmp12b = __lsx_vfmadd_s(_vm4_25, _r3, __lsx_vfadd_s(_r1, _r5));
                __m128 _tmp34a = __lsx_vfmadd_s(_vm1_25, _r4, __lsx_vfmadd_s(_v0_25, _r2, _r6));
                __m128 _tmp34b = __lsx_vfmadd_s(_v2, _r5, __lsx_vfmadd_s(_vm2_5, _r3, __lsx_vfmul_s(_r1, _v0_5)));
                __m128 _tmp56a = __lsx_vfmadd_s(_v4, __lsx_vfmadd_s(_vm1_25, _r4, _r2), _r6);
                __m128 _tmp56b = __lsx_vfmadd_s(_v0_5, _r5, __lsx_vfmadd_s(_vm2_5, _r3, __lsx_vfmul_s(_r1, _v2)));

                __m128 _tmp0 = __lsx_vfmadd_s(_v5_25, __lsx_vfsub_s(_r4, _r2), __lsx_vfsub_s(_r0, _r6));
                __m128 _tmp1 = __lsx_vfadd_s(_tmp12a, _tmp12b);
                __m128 _tmp2 = __lsx_vfsub_s(_tmp12a, _tmp12b);
                __m128 _tmp3 = __lsx_vfadd_s(_tmp34a, _tmp34b);
                __m128 _tmp4 = __lsx_vfsub_s(_tmp34a, _tmp34b);
                __m128 _tmp5 = __lsx_vfadd_s(_tmp56a, _tmp56b);
                __m128 _tmp6 = __lsx_vfsub_s(_tmp56a, _tmp56b);
                __m128 _tmp7 = __lsx_vfmadd_s(_v5_25, __lsx_vfsub_s(_r3, _r5), __lsx_vfsub_s(_r7, _r1));

                __lsx_vst((__m128i)_tmp0, p0, 0);
                __lsx_vst((__m128i)_tmp1, p1, 0);
                __lsx_vst((__m128i)_tmp2, p2, 0);
                __lsx_vst((__m128i)_tmp3, p3, 0);
                __lsx_vst((__m128i)_tmp4, p4, 0);
                __lsx_vst((__m128i)_tmp5, p5, 0);
                __lsx_vst((__m128i)_tmp6, p6, 0);
                __lsx_vst((__m128i)_tmp7, p7, 0);

                p0 += max_jj * 8 * 4;
                p1 += max_jj * 8 * 4;
                p2 += max_jj * 8 * 4;
                p3 += max_jj * 8 * 4;
                p4 += max_jj * 8 * 4;
                p5 += max_jj * 8 * 4;
                p6 += max_jj * 8 * 4;
                p7 += max_jj * 8 * 4;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 4;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
#else // __loongarch_sx
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    #pragma omp parallel for num_threads(nT)
#endif // __loongarch_sx
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

        float tmp[8][8][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = bottom_blob.channel(k + kk).row(ti * 6) + (tj * 6);

            for (int m = 0; m < 8; m++)
            {
                float r00 = 0.f;
                float r01 = 0.f;
                float r10 = 0.f;
                float r11 = 0.f;
                float r20 = 0.f;
                float r21 = 0.f;
                float r30 = 0.f;
                float r31 = 0.f;
                float r40 = 0.f;
                float r41 = 0.f;
                float r50 = 0.f;
                float r51 = 0.f;
                float r60 = 0.f;
                float r61 = 0.f;
                float r70 = 0.f;
                float r71 = 0.f;

                if (ti * 6 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const float* r1 = r0 + N;

                        r00 = r0[0];
                        r01 = r1[0];
                        if (tj * 6 + 1 < w)
                        {
                            r10 = r0[1];
                            r11 = r1[1];
                        }
                        if (tj * 6 + 2 < w)
                        {
                            r20 = r0[2];
                            r21 = r1[2];
                        }
                        if (tj * 6 + 3 < w)
                        {
                            r30 = r0[3];
                            r31 = r1[3];
                        }
                        if (tj * 6 + 4 < w)
                        {
                            r40 = r0[4];
                            r41 = r1[4];
                        }
                        if (tj * 6 + 5 < w)
                        {
                            r50 = r0[5];
                            r51 = r1[5];
                        }
                        if (tj * 6 + 6 < w)
                        {
                            r60 = r0[6];
                            r61 = r1[6];
                        }
                        if (tj * 6 + 7 < w)
                        {
                            r70 = r0[7];
                            r71 = r1[7];
                        }
                    }
                }

                float tmp12a0 = r20 + r60 - r40 * 4.25f;
                float tmp12a1 = r21 + r61 - r41 * 4.25f;
                float tmp12b0 = r10 + r50 - r30 * 4.25f;
                float tmp12b1 = r11 + r51 - r31 * 4.25f;
                float tmp34a0 = r60 + r20 * 0.25f - r40 * 1.25f;
                float tmp34a1 = r61 + r21 * 0.25f - r41 * 1.25f;
                float tmp34b0 = r10 * 0.5f - r30 * 2.5f + r50 * 2.f;
                float tmp34b1 = r11 * 0.5f - r31 * 2.5f + r51 * 2.f;
                float tmp56a0 = r20 * 4.f - r40 * 5.f + r60;
                float tmp56a1 = r21 * 4.f - r41 * 5.f + r61;
                float tmp56b0 = r10 * 2.f - r30 * 2.5f + r50 * 0.5f;
                float tmp56b1 = r11 * 2.f - r31 * 2.5f + r51 * 0.5f;

                tmp[0][m][0] = r00 - r60 + (r40 - r20) * 5.25f;
                tmp[0][m][1] = r01 - r61 + (r41 - r21) * 5.25f;
                tmp[1][m][0] = tmp12a0 + tmp12b0;
                tmp[1][m][1] = tmp12a1 + tmp12b1;
                tmp[2][m][0] = tmp12a0 - tmp12b0;
                tmp[2][m][1] = tmp12a1 - tmp12b1;
                tmp[3][m][0] = tmp34a0 + tmp34b0;
                tmp[3][m][1] = tmp34a1 + tmp34b1;
                tmp[4][m][0] = tmp34a0 - tmp34b0;
                tmp[4][m][1] = tmp34a1 - tmp34b1;
                tmp[5][m][0] = tmp56a0 + tmp56b0;
                tmp[5][m][1] = tmp56a1 + tmp56b1;
                tmp[6][m][0] = tmp56a0 - tmp56b0;
                tmp[6][m][1] = tmp56a1 - tmp56b1;
                tmp[7][m][0] = r70 - r10 + (r30 - r50) * 5.25f;
                tmp[7][m][1] = r71 - r11 + (r31 - r51) * 5.25f;

                r0 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 64 + jj * 2;
            float* p1 = p0 + max_jj * 2;
            float* p2 = p0 + max_jj * 2 * 2;
            float* p3 = p0 + max_jj * 2 * 3;
            float* p4 = p0 + max_jj * 2 * 4;
            float* p5 = p0 + max_jj * 2 * 5;
            float* p6 = p0 + max_jj * 2 * 6;
            float* p7 = p0 + max_jj * 2 * 7;

            for (int m = 0; m < 8; m++)
            {
                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];
                float r40 = tmp[m][4][0];
                float r41 = tmp[m][4][1];
                float r50 = tmp[m][5][0];
                float r51 = tmp[m][5][1];
                float r60 = tmp[m][6][0];
                float r61 = tmp[m][6][1];
                float r70 = tmp[m][7][0];
                float r71 = tmp[m][7][1];

                float tmp12a0 = r20 + r60 - r40 * 4.25f;
                float tmp12a1 = r21 + r61 - r41 * 4.25f;
                float tmp12b0 = r10 + r50 - r30 * 4.25f;
                float tmp12b1 = r11 + r51 - r31 * 4.25f;
                float tmp34a0 = r60 + r20 * 0.25f - r40 * 1.25f;
                float tmp34a1 = r61 + r21 * 0.25f - r41 * 1.25f;
                float tmp34b0 = r10 * 0.5f - r30 * 2.5f + r50 * 2.f;
                float tmp34b1 = r11 * 0.5f - r31 * 2.5f + r51 * 2.f;
                float tmp56a0 = r20 * 4.f - r40 * 5.f + r60;
                float tmp56a1 = r21 * 4.f - r41 * 5.f + r61;
                float tmp56b0 = r10 * 2.f - r30 * 2.5f + r50 * 0.5f;
                float tmp56b1 = r11 * 2.f - r31 * 2.5f + r51 * 0.5f;

                p0[0] = r00 - r60 + (r40 - r20) * 5.25f;
                p0[1] = r01 - r61 + (r41 - r21) * 5.25f;
                p1[0] = tmp12a0 + tmp12b0;
                p1[1] = tmp12a1 + tmp12b1;
                p2[0] = tmp12a0 - tmp12b0;
                p2[1] = tmp12a1 - tmp12b1;
                p3[0] = tmp34a0 + tmp34b0;
                p3[1] = tmp34a1 + tmp34b1;
                p4[0] = tmp34a0 - tmp34b0;
                p4[1] = tmp34a1 - tmp34b1;
                p5[0] = tmp56a0 + tmp56b0;
                p5[1] = tmp56a1 + tmp56b1;
                p6[0] = tmp56a0 - tmp56b0;
                p6[1] = tmp56a1 - tmp56b1;
                p7[0] = r70 - r10 + (r30 - r50) * 5.25f;
                p7[1] = r71 - r11 + (r31 - r51) * 5.25f;

                p0 += max_jj * 8 * 2;
                p1 += max_jj * 8 * 2;
                p2 += max_jj * 8 * 2;
                p3 += max_jj * 8 * 2;
                p4 += max_jj * 8 * 2;
                p5 += max_jj * 8 * 2;
                p6 += max_jj * 8 * 2;
                p7 += max_jj * 8 * 2;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 2;
    for (int kk = remain_max_kk_start; kk < max_kk; kk++)
    {
        float tmp[8][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0123 = bottom_blob.channel(k + kk).row(ti * 6) + (tj * 6);

            for (int m = 0; m < 8; m++)
            {
                float r0 = 0.f;
                float r1 = 0.f;
                float r2 = 0.f;
                float r3 = 0.f;
                float r4 = 0.f;
                float r5 = 0.f;
                float r6 = 0.f;
                float r7 = 0.f;

                if (ti * 6 + m < h)
                {
                    // if (elempack == 1)
                    {
                        r0 = r0123[0];
                        if (tj * 6 + 1 < w) r1 = r0123[1];
                        if (tj * 6 + 2 < w) r2 = r0123[2];
                        if (tj * 6 + 3 < w) r3 = r0123[3];
                        if (tj * 6 + 4 < w) r4 = r0123[4];
                        if (tj * 6 + 5 < w) r5 = r0123[5];
                        if (tj * 6 + 6 < w) r6 = r0123[6];
                        if (tj * 6 + 7 < w) r7 = r0123[7];
                    }
                }

                float tmp12a = r2 + r6 - r4 * 4.25f;
                float tmp12b = r1 + r5 - r3 * 4.25f;
                float tmp34a = r6 + r2 * 0.25f - r4 * 1.25f;
                float tmp34b = r1 * 0.5f - r3 * 2.5f + r5 * 2.f;
                float tmp56a = r2 * 4.f - r4 * 5.f + r6;
                float tmp56b = r1 * 2.f - r3 * 2.5f + r5 * 0.5f;

                tmp[0][m] = r0 - r6 + (r4 - r2) * 5.25f;
                tmp[1][m] = tmp12a + tmp12b;
                tmp[2][m] = tmp12a - tmp12b;
                tmp[3][m] = tmp34a + tmp34b;
                tmp[4][m] = tmp34a - tmp34b;
                tmp[5][m] = tmp56a + tmp56b;
                tmp[6][m] = tmp56a - tmp56b;
                tmp[7][m] = r7 - r1 + (r3 - r5) * 5.25f;

                r0123 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 64 + jj;
            float* p1 = p0 + max_jj;
            float* p2 = p0 + max_jj * 2;
            float* p3 = p0 + max_jj * 3;
            float* p4 = p0 + max_jj * 4;
            float* p5 = p0 + max_jj * 5;
            float* p6 = p0 + max_jj * 6;
            float* p7 = p0 + max_jj * 7;

            for (int m = 0; m < 8; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];
                float r4 = tmp[m][4];
                float r5 = tmp[m][5];
                float r6 = tmp[m][6];
                float r7 = tmp[m][7];

                float tmp12a = r2 + r6 - r4 * 4.25f;
                float tmp12b = r1 + r5 - r3 * 4.25f;
                float tmp34a = r6 + r2 * 0.25f - r4 * 1.25f;
                float tmp34b = r1 * 0.5f - r3 * 2.5f + r5 * 2.f;
                float tmp56a = r2 * 4.f - r4 * 5.f + r6;
                float tmp56b = r1 * 2.f - r3 * 2.5f + r5 * 0.5f;

                p0[0] = r0 - r6 + (r4 - r2) * 5.25f;
                p1[0] = tmp12a + tmp12b;
                p2[0] = tmp12a - tmp12b;
                p3[0] = tmp34a + tmp34b;
                p4[0] = tmp34a - tmp34b;
                p5[0] = tmp56a + tmp56b;
                p6[0] = tmp56a - tmp56b;
                p7[0] = r7 - r1 + (r3 - r5) * 5.25f;

                p0 += max_jj * 8;
                p1 += max_jj * 8;
                p2 += max_jj * 8;
                p3 += max_jj * 8;
                p4 += max_jj * 8;
                p5 += max_jj * 8;
                p6 += max_jj * 8;
                p7 += max_jj * 8;
            }
        }
    }
}

static inline void conv3x3s1_winograd63_transform_output_tile(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
{
    // const float otm[6][8] = {
    //     {1.0f, 1.0f,  1.0f,  1.0f,  1.0f, 32.0f, 32.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f,  2.0f, -2.0f, 16.0f,-16.0f, 0.0f},
    //     {0.0f, 1.0f,  1.0f,  4.0f,  4.0f,  8.0f,  8.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f,  8.0f, -8.0f,  4.0f, -4.0f, 0.0f},
    //     {0.0f, 1.0f,  1.0f, 16.0f, 16.0f,  2.0f,  2.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f, 32.0f,-32.0f,  1.0f, -1.0f, 1.0f}
    // };

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const size_t N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 5) / 6;

    const float* biasptr = bias;

    int ii = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        __m256 _bias0 = biasptr ? (__m256)__lasx_xvld(biasptr + i + ii, 0) : (__m256)__lasx_xvldi(0);

        __attribute__((aligned(32))) float tmp[6][8][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 64 + jj * 8;
            const float* r1 = r0 + max_jj * 8;
            const float* r2 = r0 + max_jj * 8 * 2;
            const float* r3 = r0 + max_jj * 8 * 3;
            const float* r4 = r0 + max_jj * 8 * 4;
            const float* r5 = r0 + max_jj * 8 * 5;
            const float* r6 = r0 + max_jj * 8 * 6;
            const float* r7 = r0 + max_jj * 8 * 7;

            __m256 _v32 = (__m256)__lasx_xvreplfr2vr_s(32.f);
            __m256 _v16 = (__m256)__lasx_xvreplfr2vr_s(16.f);
            __m256 _v8 = (__m256)__lasx_xvreplfr2vr_s(8.f);
            __m256 _v4 = (__m256)__lasx_xvreplfr2vr_s(4.f);
            __m256 _v2 = (__m256)__lasx_xvreplfr2vr_s(2.f);

            for (int m = 0; m < 8; m++)
            {
                __m256 _r0 = (__m256)__lasx_xvld(r0, 0);
                __m256 _r1 = (__m256)__lasx_xvld(r1, 0);
                __m256 _r2 = (__m256)__lasx_xvld(r2, 0);
                __m256 _r3 = (__m256)__lasx_xvld(r3, 0);
                __m256 _r4 = (__m256)__lasx_xvld(r4, 0);
                __m256 _r5 = (__m256)__lasx_xvld(r5, 0);
                __m256 _r6 = (__m256)__lasx_xvld(r6, 0);
                __m256 _r7 = (__m256)__lasx_xvld(r7, 0);

                __m256 _tmp024a = __lasx_xvfadd_s(_r1, _r2);
                __m256 _tmp135a = __lasx_xvfsub_s(_r1, _r2);
                __m256 _tmp024b = __lasx_xvfadd_s(_r3, _r4);
                __m256 _tmp135b = __lasx_xvfsub_s(_r3, _r4);
                __m256 _tmp024c = __lasx_xvfadd_s(_r5, _r6);
                __m256 _tmp135c = __lasx_xvfsub_s(_r5, _r6);
                __m256 _tmp0 = __lasx_xvfadd_s(__lasx_xvfadd_s(_r0, _tmp024a), __lasx_xvfmadd_s(_v32, _tmp024c, _tmp024b));
                __m256 _tmp1 = __lasx_xvfmadd_s(_v16, _tmp135c, __lasx_xvfmadd_s(_v2, _tmp135b, _tmp135a));
                __m256 _tmp2 = __lasx_xvfmadd_s(_v8, _tmp024c, __lasx_xvfmadd_s(_v4, _tmp024b, _tmp024a));
                __m256 _tmp3 = __lasx_xvfmadd_s(_v4, _tmp135c, __lasx_xvfmadd_s(_v8, _tmp135b, _tmp135a));
                __m256 _tmp4 = __lasx_xvfmadd_s(_v2, _tmp024c, __lasx_xvfmadd_s(_v16, _tmp024b, _tmp024a));
                __m256 _tmp5 = __lasx_xvfadd_s(__lasx_xvfadd_s(_r7, _tmp135a), __lasx_xvfmadd_s(_v32, _tmp135b, _tmp135c));

                __lasx_xvst((__m256i)_tmp0, tmp[0][m], 0);
                __lasx_xvst((__m256i)_tmp1, tmp[1][m], 0);
                __lasx_xvst((__m256i)_tmp2, tmp[2][m], 0);
                __lasx_xvst((__m256i)_tmp3, tmp[3][m], 0);
                __lasx_xvst((__m256i)_tmp4, tmp[4][m], 0);
                __lasx_xvst((__m256i)_tmp5, tmp[5][m], 0);

                r0 += max_jj * 8 * 8;
                r1 += max_jj * 8 * 8;
                r2 += max_jj * 8 * 8;
                r3 += max_jj * 8 * 8;
                r4 += max_jj * 8 * 8;
                r5 += max_jj * 8 * 8;
                r6 += max_jj * 8 * 8;
                r7 += max_jj * 8 * 8;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 6) + (tj * 6) * out_elempack;

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                __m256 _r0 = (__m256)__lasx_xvld(tmp[m][0], 0);
                __m256 _r1 = (__m256)__lasx_xvld(tmp[m][1], 0);
                __m256 _r2 = (__m256)__lasx_xvld(tmp[m][2], 0);
                __m256 _r3 = (__m256)__lasx_xvld(tmp[m][3], 0);
                __m256 _r4 = (__m256)__lasx_xvld(tmp[m][4], 0);
                __m256 _r5 = (__m256)__lasx_xvld(tmp[m][5], 0);
                __m256 _r6 = (__m256)__lasx_xvld(tmp[m][6], 0);
                __m256 _r7 = (__m256)__lasx_xvld(tmp[m][7], 0);

                __m256 _tmp024a = __lasx_xvfadd_s(_r1, _r2);
                __m256 _tmp135a = __lasx_xvfsub_s(_r1, _r2);
                __m256 _tmp024b = __lasx_xvfadd_s(_r3, _r4);
                __m256 _tmp135b = __lasx_xvfsub_s(_r3, _r4);
                __m256 _tmp024c = __lasx_xvfadd_s(_r5, _r6);
                __m256 _tmp135c = __lasx_xvfsub_s(_r5, _r6);
                __m256 _tmp0 = __lasx_xvfadd_s(_bias0, __lasx_xvfadd_s(__lasx_xvfadd_s(_r0, _tmp024a), __lasx_xvfmadd_s(_v32, _tmp024c, _tmp024b)));
                __m256 _tmp1 = __lasx_xvfadd_s(_bias0, __lasx_xvfmadd_s(_v16, _tmp135c, __lasx_xvfmadd_s(_v2, _tmp135b, _tmp135a)));
                __m256 _tmp2 = __lasx_xvfadd_s(_bias0, __lasx_xvfmadd_s(_v8, _tmp024c, __lasx_xvfmadd_s(_v4, _tmp024b, _tmp024a)));
                __m256 _tmp3 = __lasx_xvfadd_s(_bias0, __lasx_xvfmadd_s(_v4, _tmp135c, __lasx_xvfmadd_s(_v8, _tmp135b, _tmp135a)));
                __m256 _tmp4 = __lasx_xvfadd_s(_bias0, __lasx_xvfmadd_s(_v2, _tmp024c, __lasx_xvfmadd_s(_v16, _tmp024b, _tmp024a)));
                __m256 _tmp5 = __lasx_xvfadd_s(_bias0, __lasx_xvfadd_s(__lasx_xvfadd_s(_r7, _tmp135a), __lasx_xvfmadd_s(_v32, _tmp135b, _tmp135c)));

                if (out_elempack == 8)
                {
                    __lasx_xvst((__m256i)_tmp0, outptr0, 0);
                    if (tj * 6 + 1 < outw) __lasx_xvst((__m256i)_tmp1, outptr0 + 8, 0);
                    if (tj * 6 + 2 < outw) __lasx_xvst((__m256i)_tmp2, outptr0 + 16, 0);
                    if (tj * 6 + 3 < outw) __lasx_xvst((__m256i)_tmp3, outptr0 + 24, 0);
                    if (tj * 6 + 4 < outw) __lasx_xvst((__m256i)_tmp4, outptr0 + 32, 0);
                    if (tj * 6 + 5 < outw) __lasx_xvst((__m256i)_tmp5, outptr0 + 40, 0);
                }
                if (out_elempack == 4)
                {
                    float* outptr1 = outptr0 + N;

                    __lsx_vst(__lasx_extract_128_lo((__m256i)_tmp0), outptr0, 0);
                    __lsx_vst(__lasx_extract_128_hi((__m256i)_tmp0), outptr1, 0);
                    if (tj * 6 + 1 < outw)
                    {
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_tmp1), outptr0 + 4, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_tmp1), outptr1 + 4, 0);
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_tmp2), outptr0 + 8, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_tmp2), outptr1 + 8, 0);
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_tmp3), outptr0 + 12, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_tmp3), outptr1 + 12, 0);
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_tmp4), outptr0 + 16, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_tmp4), outptr1 + 16, 0);
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_tmp5), outptr0 + 20, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_tmp5), outptr1 + 20, 0);
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[8];
                    float tmp1[8];
                    float tmp2[8];
                    float tmp3[8];
                    float tmp4[8];
                    float tmp5[8];
                    __lasx_xvst((__m256i)_tmp0, tmp0, 0);
                    __lasx_xvst((__m256i)_tmp1, tmp1, 0);
                    __lasx_xvst((__m256i)_tmp2, tmp2, 0);
                    __lasx_xvst((__m256i)_tmp3, tmp3, 0);
                    __lasx_xvst((__m256i)_tmp4, tmp4, 0);
                    __lasx_xvst((__m256i)_tmp5, tmp5, 0);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;
                    float* outptr4 = outptr0 + N * 4;
                    float* outptr5 = outptr0 + N * 5;
                    float* outptr6 = outptr0 + N * 6;
                    float* outptr7 = outptr0 + N * 7;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    outptr4[0] = tmp0[4];
                    outptr5[0] = tmp0[5];
                    outptr6[0] = tmp0[6];
                    outptr7[0] = tmp0[7];
                    if (tj * 6 + 1 < outw)
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
                    if (tj * 6 + 2 < outw)
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
                    if (tj * 6 + 3 < outw)
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
                    if (tj * 6 + 4 < outw)
                    {
                        outptr0[4] = tmp4[0];
                        outptr1[4] = tmp4[1];
                        outptr2[4] = tmp4[2];
                        outptr3[4] = tmp4[3];
                        outptr4[4] = tmp4[4];
                        outptr5[4] = tmp4[5];
                        outptr6[4] = tmp4[6];
                        outptr7[4] = tmp4[7];
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        outptr0[5] = tmp5[0];
                        outptr1[5] = tmp5[1];
                        outptr2[5] = tmp5[2];
                        outptr3[5] = tmp5[3];
                        outptr4[5] = tmp5[4];
                        outptr5[5] = tmp5[5];
                        outptr6[5] = tmp5[6];
                        outptr7[5] = tmp5[7];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __loongarch_asx
    for (; ii + 3 < max_ii; ii += 4)
    {
        __m128 _bias0 = biasptr ? (__m128)__lsx_vld(biasptr + i + ii, 0) : (__m128)__lsx_vldi(0);

        __attribute__((aligned(16))) float tmp[6][8][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 64 + jj * 4;
            const float* r1 = r0 + max_jj * 4;
            const float* r2 = r0 + max_jj * 4 * 2;
            const float* r3 = r0 + max_jj * 4 * 3;
            const float* r4 = r0 + max_jj * 4 * 4;
            const float* r5 = r0 + max_jj * 4 * 5;
            const float* r6 = r0 + max_jj * 4 * 6;
            const float* r7 = r0 + max_jj * 4 * 7;

            __m128 _v32 = (__m128)__lsx_vreplfr2vr_s(32.f);
            __m128 _v16 = (__m128)__lsx_vreplfr2vr_s(16.f);
            __m128 _v8 = (__m128)__lsx_vreplfr2vr_s(8.f);
            __m128 _v4 = (__m128)__lsx_vreplfr2vr_s(4.f);
            __m128 _v2 = (__m128)__lsx_vreplfr2vr_s(2.f);

            for (int m = 0; m < 8; m++)
            {
                __m128 _r0 = (__m128)__lsx_vld(r0, 0);
                __m128 _r1 = (__m128)__lsx_vld(r1, 0);
                __m128 _r2 = (__m128)__lsx_vld(r2, 0);
                __m128 _r3 = (__m128)__lsx_vld(r3, 0);
                __m128 _r4 = (__m128)__lsx_vld(r4, 0);
                __m128 _r5 = (__m128)__lsx_vld(r5, 0);
                __m128 _r6 = (__m128)__lsx_vld(r6, 0);
                __m128 _r7 = (__m128)__lsx_vld(r7, 0);

                __m128 _tmp024a = __lsx_vfadd_s(_r1, _r2);
                __m128 _tmp135a = __lsx_vfsub_s(_r1, _r2);
                __m128 _tmp024b = __lsx_vfadd_s(_r3, _r4);
                __m128 _tmp135b = __lsx_vfsub_s(_r3, _r4);
                __m128 _tmp024c = __lsx_vfadd_s(_r5, _r6);
                __m128 _tmp135c = __lsx_vfsub_s(_r5, _r6);
                __m128 _tmp0 = __lsx_vfadd_s(__lsx_vfadd_s(_r0, _tmp024a), __lsx_vfmadd_s(_v32, _tmp024c, _tmp024b));
                __m128 _tmp1 = __lsx_vfmadd_s(_v16, _tmp135c, __lsx_vfmadd_s(_v2, _tmp135b, _tmp135a));
                __m128 _tmp2 = __lsx_vfmadd_s(_v8, _tmp024c, __lsx_vfmadd_s(_v4, _tmp024b, _tmp024a));
                __m128 _tmp3 = __lsx_vfmadd_s(_v4, _tmp135c, __lsx_vfmadd_s(_v8, _tmp135b, _tmp135a));
                __m128 _tmp4 = __lsx_vfmadd_s(_v2, _tmp024c, __lsx_vfmadd_s(_v16, _tmp024b, _tmp024a));
                __m128 _tmp5 = __lsx_vfadd_s(__lsx_vfadd_s(_r7, _tmp135a), __lsx_vfmadd_s(_v32, _tmp135b, _tmp135c));

                __lsx_vst((__m128i)_tmp0, tmp[0][m], 0);
                __lsx_vst((__m128i)_tmp1, tmp[1][m], 0);
                __lsx_vst((__m128i)_tmp2, tmp[2][m], 0);
                __lsx_vst((__m128i)_tmp3, tmp[3][m], 0);
                __lsx_vst((__m128i)_tmp4, tmp[4][m], 0);
                __lsx_vst((__m128i)_tmp5, tmp[5][m], 0);

                r0 += max_jj * 8 * 4;
                r1 += max_jj * 8 * 4;
                r2 += max_jj * 8 * 4;
                r3 += max_jj * 8 * 4;
                r4 += max_jj * 8 * 4;
                r5 += max_jj * 8 * 4;
                r6 += max_jj * 8 * 4;
                r7 += max_jj * 8 * 4;
            }

            float* outptr0 = top_blob.channel((i + ii) / out_elempack).row(ti * 6) + (tj * 6) * out_elempack;

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                __m128 _r0 = (__m128)__lsx_vld(tmp[m][0], 0);
                __m128 _r1 = (__m128)__lsx_vld(tmp[m][1], 0);
                __m128 _r2 = (__m128)__lsx_vld(tmp[m][2], 0);
                __m128 _r3 = (__m128)__lsx_vld(tmp[m][3], 0);
                __m128 _r4 = (__m128)__lsx_vld(tmp[m][4], 0);
                __m128 _r5 = (__m128)__lsx_vld(tmp[m][5], 0);
                __m128 _r6 = (__m128)__lsx_vld(tmp[m][6], 0);
                __m128 _r7 = (__m128)__lsx_vld(tmp[m][7], 0);

                __m128 _tmp024a = __lsx_vfadd_s(_r1, _r2);
                __m128 _tmp135a = __lsx_vfsub_s(_r1, _r2);
                __m128 _tmp024b = __lsx_vfadd_s(_r3, _r4);
                __m128 _tmp135b = __lsx_vfsub_s(_r3, _r4);
                __m128 _tmp024c = __lsx_vfadd_s(_r5, _r6);
                __m128 _tmp135c = __lsx_vfsub_s(_r5, _r6);
                __m128 _tmp0 = __lsx_vfadd_s(_bias0, __lsx_vfadd_s(__lsx_vfadd_s(_r0, _tmp024a), __lsx_vfmadd_s(_v32, _tmp024c, _tmp024b)));
                __m128 _tmp1 = __lsx_vfadd_s(_bias0, __lsx_vfmadd_s(_v16, _tmp135c, __lsx_vfmadd_s(_v2, _tmp135b, _tmp135a)));
                __m128 _tmp2 = __lsx_vfadd_s(_bias0, __lsx_vfmadd_s(_v8, _tmp024c, __lsx_vfmadd_s(_v4, _tmp024b, _tmp024a)));
                __m128 _tmp3 = __lsx_vfadd_s(_bias0, __lsx_vfmadd_s(_v4, _tmp135c, __lsx_vfmadd_s(_v8, _tmp135b, _tmp135a)));
                __m128 _tmp4 = __lsx_vfadd_s(_bias0, __lsx_vfmadd_s(_v2, _tmp024c, __lsx_vfmadd_s(_v16, _tmp024b, _tmp024a)));
                __m128 _tmp5 = __lsx_vfadd_s(_bias0, __lsx_vfadd_s(__lsx_vfadd_s(_r7, _tmp135a), __lsx_vfmadd_s(_v32, _tmp135b, _tmp135c)));

                if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_tmp0, outptr0, 0);
                    if (tj * 6 + 1 < outw) __lsx_vst((__m128i)_tmp1, outptr0 + 4, 0);
                    if (tj * 6 + 2 < outw) __lsx_vst((__m128i)_tmp2, outptr0 + 8, 0);
                    if (tj * 6 + 3 < outw) __lsx_vst((__m128i)_tmp3, outptr0 + 12, 0);
                    if (tj * 6 + 4 < outw) __lsx_vst((__m128i)_tmp4, outptr0 + 16, 0);
                    if (tj * 6 + 5 < outw) __lsx_vst((__m128i)_tmp5, outptr0 + 20, 0);
                }
                if (out_elempack == 1)
                {
                    float tmp0[4];
                    float tmp1[4];
                    float tmp2[4];
                    float tmp3[4];
                    float tmp4[4];
                    float tmp5[4];
                    __lsx_vst((__m128i)_tmp0, tmp0, 0);
                    __lsx_vst((__m128i)_tmp1, tmp1, 0);
                    __lsx_vst((__m128i)_tmp2, tmp2, 0);
                    __lsx_vst((__m128i)_tmp3, tmp3, 0);
                    __lsx_vst((__m128i)_tmp4, tmp4, 0);
                    __lsx_vst((__m128i)_tmp5, tmp5, 0);

                    float* outptr1 = outptr0 + N;
                    float* outptr2 = outptr0 + N * 2;
                    float* outptr3 = outptr0 + N * 3;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    if (tj * 6 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        outptr0[2] = tmp2[0];
                        outptr1[2] = tmp2[1];
                        outptr2[2] = tmp2[2];
                        outptr3[2] = tmp2[3];
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        outptr0[3] = tmp3[0];
                        outptr1[3] = tmp3[1];
                        outptr2[3] = tmp3[2];
                        outptr3[3] = tmp3[3];
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        outptr0[4] = tmp4[0];
                        outptr1[4] = tmp4[1];
                        outptr2[4] = tmp4[2];
                        outptr3[4] = tmp4[3];
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        outptr0[5] = tmp5[0];
                        outptr1[5] = tmp5[1];
                        outptr2[5] = tmp5[2];
                        outptr3[5] = tmp5[3];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;
        float bias1 = biasptr ? biasptr[i + ii + 1] : 0.f;

        float tmp[6][8][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 64 + jj * 2;
            const float* r1 = r0 + max_jj * 2;
            const float* r2 = r0 + max_jj * 2 * 2;
            const float* r3 = r0 + max_jj * 2 * 3;
            const float* r4 = r0 + max_jj * 2 * 4;
            const float* r5 = r0 + max_jj * 2 * 5;
            const float* r6 = r0 + max_jj * 2 * 6;
            const float* r7 = r0 + max_jj * 2 * 7;

            for (int m = 0; m < 8; m++)
            {
                float tmp024a0 = r1[0] + r2[0];
                float tmp024a1 = r1[1] + r2[1];
                float tmp135a0 = r1[0] - r2[0];
                float tmp135a1 = r1[1] - r2[1];
                float tmp024b0 = r3[0] + r4[0];
                float tmp024b1 = r3[1] + r4[1];
                float tmp135b0 = r3[0] - r4[0];
                float tmp135b1 = r3[1] - r4[1];
                float tmp024c0 = r5[0] + r6[0];
                float tmp024c1 = r5[1] + r6[1];
                float tmp135c0 = r5[0] - r6[0];
                float tmp135c1 = r5[1] - r6[1];

                tmp[0][m][0] = r0[0] + tmp024a0 + tmp024b0 + tmp024c0 * 32;
                tmp[0][m][1] = r0[1] + tmp024a1 + tmp024b1 + tmp024c1 * 32;
                tmp[1][m][0] = tmp135a0 + tmp135b0 + tmp135b0 + tmp135c0 * 16;
                tmp[1][m][1] = tmp135a1 + tmp135b1 + tmp135b1 + tmp135c1 * 16;
                tmp[2][m][0] = tmp024a0 + tmp024b0 * 4 + tmp024c0 * 8;
                tmp[2][m][1] = tmp024a1 + tmp024b1 * 4 + tmp024c1 * 8;
                tmp[3][m][0] = tmp135a0 + tmp135b0 * 8 + tmp135c0 * 4;
                tmp[3][m][1] = tmp135a1 + tmp135b1 * 8 + tmp135c1 * 4;
                tmp[4][m][0] = tmp024a0 + tmp024b0 * 16 + tmp024c0 + tmp024c0;
                tmp[4][m][1] = tmp024a1 + tmp024b1 * 16 + tmp024c1 + tmp024c1;
                tmp[5][m][0] = r7[0] + tmp135a0 + tmp135b0 * 32 + tmp135c0;
                tmp[5][m][1] = r7[1] + tmp135a1 + tmp135b1 * 32 + tmp135c1;

                r0 += max_jj * 8 * 2;
                r1 += max_jj * 8 * 2;
                r2 += max_jj * 8 * 2;
                r3 += max_jj * 8 * 2;
                r4 += max_jj * 8 * 2;
                r5 += max_jj * 8 * 2;
                r6 += max_jj * 8 * 2;
                r7 += max_jj * 8 * 2;
            }

            float* outptr0 = top_blob.channel(i + ii).row(ti * 6) + (tj * 6);

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];
                float r40 = tmp[m][4][0];
                float r41 = tmp[m][4][1];
                float r50 = tmp[m][5][0];
                float r51 = tmp[m][5][1];
                float r60 = tmp[m][6][0];
                float r61 = tmp[m][6][1];
                float r70 = tmp[m][7][0];
                float r71 = tmp[m][7][1];

                float tmp024a0 = r10 + r20;
                float tmp024a1 = r11 + r21;
                float tmp135a0 = r10 - r20;
                float tmp135a1 = r11 - r21;
                float tmp024b0 = r30 + r40;
                float tmp024b1 = r31 + r41;
                float tmp135b0 = r30 - r40;
                float tmp135b1 = r31 - r41;
                float tmp024c0 = r50 + r60;
                float tmp024c1 = r51 + r61;
                float tmp135c0 = r50 - r60;
                float tmp135c1 = r51 - r61;

                float tmp00 = bias0 + r00 + tmp024a0 + tmp024b0 + tmp024c0 * 32;
                float tmp01 = bias1 + r01 + tmp024a1 + tmp024b1 + tmp024c1 * 32;
                float tmp10 = bias0 + tmp135a0 + tmp135b0 + tmp135b0 + tmp135c0 * 16;
                float tmp11 = bias1 + tmp135a1 + tmp135b1 + tmp135b1 + tmp135c1 * 16;
                float tmp20 = bias0 + tmp024a0 + tmp024b0 * 4 + tmp024c0 * 8;
                float tmp21 = bias1 + tmp024a1 + tmp024b1 * 4 + tmp024c1 * 8;
                float tmp30 = bias0 + tmp135a0 + tmp135b0 * 8 + tmp135c0 * 4;
                float tmp31 = bias1 + tmp135a1 + tmp135b1 * 8 + tmp135c1 * 4;
                float tmp40 = bias0 + tmp024a0 + tmp024b0 * 16 + tmp024c0 + tmp024c0;
                float tmp41 = bias1 + tmp024a1 + tmp024b1 * 16 + tmp024c1 + tmp024c1;
                float tmp50 = bias0 + r70 + tmp135a0 + tmp135b0 * 32 + tmp135c0;
                float tmp51 = bias1 + r71 + tmp135a1 + tmp135b1 * 32 + tmp135c1;

                // if (out_elempack == 1)
                {
                    float* outptr1 = outptr0 + N;

                    outptr0[0] = tmp00;
                    outptr1[0] = tmp01;
                    if (tj * 6 + 1 < outw)
                    {
                        outptr0[1] = tmp10;
                        outptr1[1] = tmp11;
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        outptr0[2] = tmp20;
                        outptr1[2] = tmp21;
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        outptr0[3] = tmp30;
                        outptr1[3] = tmp31;
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        outptr0[4] = tmp40;
                        outptr1[4] = tmp41;
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        outptr0[5] = tmp50;
                        outptr1[5] = tmp51;
                    }
                }

                outptr0 += outw;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;

        float tmp[6][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 64 + jj;
            const float* r1 = r0 + max_jj;
            const float* r2 = r0 + max_jj * 2;
            const float* r3 = r0 + max_jj * 3;
            const float* r4 = r0 + max_jj * 4;
            const float* r5 = r0 + max_jj * 5;
            const float* r6 = r0 + max_jj * 6;
            const float* r7 = r0 + max_jj * 7;

            for (int m = 0; m < 8; m++)
            {
                float tmp024a = r1[0] + r2[0];
                float tmp135a = r1[0] - r2[0];
                float tmp024b = r3[0] + r4[0];
                float tmp135b = r3[0] - r4[0];
                float tmp024c = r5[0] + r6[0];
                float tmp135c = r5[0] - r6[0];

                tmp[0][m] = r0[0] + tmp024a + tmp024b + tmp024c * 32;
                tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
                tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
                tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
                tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;
                tmp[5][m] = r7[0] + tmp135a + tmp135b * 32 + tmp135c;

                r0 += max_jj * 8;
                r1 += max_jj * 8;
                r2 += max_jj * 8;
                r3 += max_jj * 8;
                r4 += max_jj * 8;
                r5 += max_jj * 8;
                r6 += max_jj * 8;
                r7 += max_jj * 8;
            }

            float* outptr0 = top_blob.channel(i + ii).row(ti * 6) + (tj * 6);

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];
                float r4 = tmp[m][4];
                float r5 = tmp[m][5];
                float r6 = tmp[m][6];
                float r7 = tmp[m][7];

                float tmp024a = r1 + r2;
                float tmp135a = r1 - r2;
                float tmp024b = r3 + r4;
                float tmp135b = r3 - r4;
                float tmp024c = r5 + r6;
                float tmp135c = r5 - r6;

                float tmp0 = bias0 + r0 + tmp024a + tmp024b + tmp024c * 32;
                float tmp1 = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16;
                float tmp2 = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
                float tmp3 = bias0 + tmp135a + tmp135b * 8 + tmp135c * 4;
                float tmp4 = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;
                float tmp5 = bias0 + r7 + tmp135a + tmp135b * 32 + tmp135c;

                // if (out_elempack == 1)
                {
                    outptr0[0] = tmp0;
                    if (tj * 6 + 1 < outw) outptr0[1] = tmp1;
                    if (tj * 6 + 2 < outw) outptr0[2] = tmp2;
                    if (tj * 6 + 3 < outw) outptr0[3] = tmp3;
                    if (tj * 6 + 4 < outw) outptr0[4] = tmp4;
                    if (tj * 6 + 5 < outw) outptr0[5] = tmp5;
                }

                outptr0 += outw;
            }
        }
    }
}

static int conv3x3s1_winograd63(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, const Option& opt)
{
    int outw = top_blob.w;
    int outh = top_blob.h;

    // pad to 6n+2, winograd F(6,3)
    int w_tiles = (outw + 5) / 6;
    int h_tiles = (outh + 5) / 6;
    int tiles = w_tiles * h_tiles;

    const int M = top_blob.c * top_blob.elempack;
    const int N = tiles;
    const int K = bottom_blob.c * bottom_blob.elempack;
    const int B = 64;

    // NCNN_LOGE("conv3x3s1_winograd63 %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, TILE_M, TILE_N, TILE_K, nT);

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
            conv3x3s1_winograd63_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, nT);
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
            conv3x3s1_winograd63_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, 1);
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

                gemm_transB_packed_tile(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk);
            }

            // transform output
            conv3x3s1_winograd63_transform_output_tile(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

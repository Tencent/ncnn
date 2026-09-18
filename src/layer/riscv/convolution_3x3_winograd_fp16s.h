// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void conv3x3s1_winograd_pack_A_tile_fp16sa_rvv(const Mat& A, Mat& AT, int batch, int max_ii, int max_kk)
{
    const int N = max_kk * batch;

    for (int b = 0; b < batch; b++)
    {
        __fp16* pp = AT.row<__fp16>(b);

        int ii = 0;
#if __riscv_zvfh
        const int packn = csrr_vlenb() / 2;
        const size_t vl = __riscv_vsetvl_e16m1(packn);

        for (; ii + (packn - 1) < max_ii; ii += packn)
        {
            const __fp16* p0 = (const __fp16*)A + ii * N + b;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vfloat16m1_t _p = __riscv_vlse16_v_f16m1(p0, N * sizeof(__fp16), vl);
                __riscv_vse16_v_f16m1(pp, _p, vl);
                p0 += batch;
                pp += packn;
            }
        }

        for (; ii + 7 < max_ii; ii += 8)
        {
            const __fp16* p0 = (const __fp16*)A + ii * N + b;

            int kk = 0;
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

        for (; ii + 3 < max_ii; ii += 4)
        {
            const __fp16* p0 = (const __fp16*)A + ii * N + b;

            int kk = 0;
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
#endif // __riscv_zvfh

        for (; ii + 1 < max_ii; ii += 2)
        {
            const __fp16* p0 = (const __fp16*)A + ii * N + b;

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
            const __fp16* p0 = (const __fp16*)A + ii * N + b;

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

static void conv3x3s1_winograd_transpose_pack_B_tile_fp16sa_rvv(const Mat& B, Mat& BT, int batch, int max_jj, int max_kk, int nT)
{
    #pragma omp parallel for num_threads(nT)
    for (int b = 0; b < batch; b++)
    {
        __fp16* pp = BT.row<__fp16>(b);

        int jj = 0;
#if __riscv_zvfh
        const int packn = csrr_vlenb() / 2;
        const size_t vl = __riscv_vsetvl_e16m1(packn);

        for (; jj + 15 < max_jj; jj += 16)
        {
            const __fp16* p0 = (const __fp16*)B + b * max_jj * packn + jj * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                vfloat16m1_t _val0 = __riscv_vle16_v_f16m1(p0, vl);
                vfloat16m1_t _val1 = __riscv_vle16_v_f16m1(p0 + packn, vl);
                vfloat16m1_t _val2 = __riscv_vle16_v_f16m1(p0 + packn * 2, vl);
                vfloat16m1_t _val3 = __riscv_vle16_v_f16m1(p0 + packn * 3, vl);
                vfloat16m1_t _val4 = __riscv_vle16_v_f16m1(p0 + packn * 4, vl);
                vfloat16m1_t _val5 = __riscv_vle16_v_f16m1(p0 + packn * 5, vl);
                vfloat16m1_t _val6 = __riscv_vle16_v_f16m1(p0 + packn * 6, vl);
                vfloat16m1_t _val7 = __riscv_vle16_v_f16m1(p0 + packn * 7, vl);
                __riscv_vssseg8e16_v_f16m1x8(pp, 16 * sizeof(__fp16), __riscv_vcreate_v_f16m1x8(_val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7), vl);

                _val0 = __riscv_vle16_v_f16m1(p0 + packn * 8, vl);
                _val1 = __riscv_vle16_v_f16m1(p0 + packn * 9, vl);
                _val2 = __riscv_vle16_v_f16m1(p0 + packn * 10, vl);
                _val3 = __riscv_vle16_v_f16m1(p0 + packn * 11, vl);
                _val4 = __riscv_vle16_v_f16m1(p0 + packn * 12, vl);
                _val5 = __riscv_vle16_v_f16m1(p0 + packn * 13, vl);
                _val6 = __riscv_vle16_v_f16m1(p0 + packn * 14, vl);
                _val7 = __riscv_vle16_v_f16m1(p0 + packn * 15, vl);
                __riscv_vssseg8e16_v_f16m1x8(pp + 8, 16 * sizeof(__fp16), __riscv_vcreate_v_f16m1x8(_val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7), vl);

                p0 += max_jj * batch * packn;
                pp += packn * 16;
            }

            p0 = (const __fp16*)B + kk * max_jj * batch + b * max_jj + jj;
            const size_t vl16 = __riscv_vsetvl_e16m2(16);
            for (; kk < max_kk; kk++)
            {
                vfloat16m2_t _p = __riscv_vle16_v_f16m2(p0, vl16);
                __riscv_vse16_v_f16m2(pp, _p, vl16);
                p0 += max_jj * batch;
                pp += 16;
            }
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            const __fp16* p0 = (const __fp16*)B + b * max_jj * packn + jj * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                vfloat16m1_t _val0 = __riscv_vle16_v_f16m1(p0, vl);
                vfloat16m1_t _val1 = __riscv_vle16_v_f16m1(p0 + packn, vl);
                vfloat16m1_t _val2 = __riscv_vle16_v_f16m1(p0 + packn * 2, vl);
                vfloat16m1_t _val3 = __riscv_vle16_v_f16m1(p0 + packn * 3, vl);
                vfloat16m1_t _val4 = __riscv_vle16_v_f16m1(p0 + packn * 4, vl);
                vfloat16m1_t _val5 = __riscv_vle16_v_f16m1(p0 + packn * 5, vl);
                vfloat16m1_t _val6 = __riscv_vle16_v_f16m1(p0 + packn * 6, vl);
                vfloat16m1_t _val7 = __riscv_vle16_v_f16m1(p0 + packn * 7, vl);
                __riscv_vsseg8e16_v_f16m1x8(pp, __riscv_vcreate_v_f16m1x8(_val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7), vl);

                p0 += max_jj * batch * packn;
                pp += packn * 8;
            }

            p0 = (const __fp16*)B + kk * max_jj * batch + b * max_jj + jj;
            const size_t vl8 = __riscv_vsetvl_e16m1(8);
            for (; kk < max_kk; kk++)
            {
                vfloat16m1_t _p = __riscv_vle16_v_f16m1(p0, vl8);
                __riscv_vse16_v_f16m1(pp, _p, vl8);
                p0 += max_jj * batch;
                pp += 8;
            }
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const __fp16* p0 = (const __fp16*)B + b * max_jj * packn + jj * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                vfloat16m1_t _val0 = __riscv_vle16_v_f16m1(p0, vl);
                vfloat16m1_t _val1 = __riscv_vle16_v_f16m1(p0 + packn, vl);
                vfloat16m1_t _val2 = __riscv_vle16_v_f16m1(p0 + packn * 2, vl);
                vfloat16m1_t _val3 = __riscv_vle16_v_f16m1(p0 + packn * 3, vl);
                __riscv_vsseg4e16_v_f16m1x4(pp, __riscv_vcreate_v_f16m1x4(_val0, _val1, _val2, _val3), vl);

                p0 += max_jj * batch * packn;
                pp += packn * 4;
            }

            p0 = (const __fp16*)B + kk * max_jj * batch + b * max_jj + jj;
            const size_t vl4 = __riscv_vsetvl_e16m1(4);
            for (; kk < max_kk; kk++)
            {
                vfloat16m1_t _p = __riscv_vle16_v_f16m1(p0, vl4);
                __riscv_vse16_v_f16m1(pp, _p, vl4);
                p0 += max_jj * batch;
                pp += 4;
            }
        }
#endif // __riscv_zvfh
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __riscv_zvfh
            const __fp16* p0 = (const __fp16*)B + b * max_jj * packn + jj * packn;
#else
            const __fp16* p0 = (const __fp16*)B + b * max_jj + jj;
#endif

            int kk = 0;
#if __riscv_zvfh
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                vfloat16m1_t _val0 = __riscv_vle16_v_f16m1(p0, vl);
                vfloat16m1_t _val1 = __riscv_vle16_v_f16m1(p0 + packn, vl);
                __riscv_vsseg2e16_v_f16m1x2(pp, __riscv_vcreate_v_f16m1x2(_val0, _val1), vl);

                p0 += max_jj * batch * packn;
                pp += packn * 2;
            }

            p0 = (const __fp16*)B + kk * max_jj * batch + b * max_jj + jj;
#endif // __riscv_zvfh
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
#if __riscv_zvfh
            const __fp16* p0 = (const __fp16*)B + b * max_jj * packn + jj * packn;
#else
            const __fp16* p0 = (const __fp16*)B + b * max_jj + jj;
#endif

            int kk = 0;
#if __riscv_zvfh
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                vfloat16m1_t _val = __riscv_vle16_v_f16m1(p0, vl);
                __riscv_vse16_v_f16m1(pp, _val, vl);

                p0 += max_jj * batch * packn;
                pp += packn;
            }

            p0 = (const __fp16*)B + kk * max_jj * batch + b * max_jj + jj;
#endif // __riscv_zvfh
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                p0 += max_jj * batch;
                pp += 1;
            }
        }
    }
}

static void conv3x3s1_winograd_gemm_transB_packed_tile_fp16sa_rvv(const Mat& AT_tile, const Mat& BT_tile, Mat& top_blob, int batch, int max_ii, int max_jj, int k, int max_kk)
{
    __fp16* outptr = top_blob;

    int ii = 0;
#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
    size_t vl;
    if (packn == 8)
        vl = __riscv_vsetvl_e16m1(8);
    else if (packn == 16)
        vl = __riscv_vsetvl_e16m1(16);
    else
        vl = __riscv_vsetvl_e16m1(packn);
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        for (int b = 0; b < batch; b++)
        {
            const __fp16* pAT = AT_tile.row<__fp16>(b) + max_kk * ii;
            const __fp16* pB = BT_tile.row<__fp16>(b);

            int jj = 0;
            for (; jj + 15 < max_jj; jj += 16)
            {
                const __fp16* pA = pAT;

                vfloat16m1_t _sum0;
                vfloat16m1_t _sum1;
                vfloat16m1_t _sum2;
                vfloat16m1_t _sum3;
                vfloat16m1_t _sum4;
                vfloat16m1_t _sum5;
                vfloat16m1_t _sum6;
                vfloat16m1_t _sum7;
                vfloat16m1_t _sum8;
                vfloat16m1_t _sum9;
                vfloat16m1_t _suma;
                vfloat16m1_t _sumb;
                vfloat16m1_t _sumc;
                vfloat16m1_t _sumd;
                vfloat16m1_t _sume;
                vfloat16m1_t _sumf;

                if (k == 0)
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                    _sum8 = _sum0;
                    _sum9 = _sum0;
                    _suma = _sum0;
                    _sumb = _sum0;
                    _sumc = _sum0;
                    _sumd = _sum0;
                    _sume = _sum0;
                    _sumf = _sum0;
                }
                else
                {
                    _sum0 = __riscv_vle16_v_f16m1(outptr, vl);
                    _sum1 = __riscv_vle16_v_f16m1(outptr + packn, vl);
                    _sum2 = __riscv_vle16_v_f16m1(outptr + packn * 2, vl);
                    _sum3 = __riscv_vle16_v_f16m1(outptr + packn * 3, vl);
                    _sum4 = __riscv_vle16_v_f16m1(outptr + packn * 4, vl);
                    _sum5 = __riscv_vle16_v_f16m1(outptr + packn * 5, vl);
                    _sum6 = __riscv_vle16_v_f16m1(outptr + packn * 6, vl);
                    _sum7 = __riscv_vle16_v_f16m1(outptr + packn * 7, vl);
                    _sum8 = __riscv_vle16_v_f16m1(outptr + packn * 8, vl);
                    _sum9 = __riscv_vle16_v_f16m1(outptr + packn * 9, vl);
                    _suma = __riscv_vle16_v_f16m1(outptr + packn * 10, vl);
                    _sumb = __riscv_vle16_v_f16m1(outptr + packn * 11, vl);
                    _sumc = __riscv_vle16_v_f16m1(outptr + packn * 12, vl);
                    _sumd = __riscv_vle16_v_f16m1(outptr + packn * 13, vl);
                    _sume = __riscv_vle16_v_f16m1(outptr + packn * 14, vl);
                    _sumf = __riscv_vle16_v_f16m1(outptr + packn * 15, vl);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    vfloat16m1_t _pA = __riscv_vle16_v_f16m1(pA, vl);
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);
                    _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pB[1], _pA, vl);
                    _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, pB[2], _pA, vl);
                    _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, pB[3], _pA, vl);
                    _sum4 = __riscv_vfmacc_vf_f16m1(_sum4, pB[4], _pA, vl);
                    _sum5 = __riscv_vfmacc_vf_f16m1(_sum5, pB[5], _pA, vl);
                    _sum6 = __riscv_vfmacc_vf_f16m1(_sum6, pB[6], _pA, vl);
                    _sum7 = __riscv_vfmacc_vf_f16m1(_sum7, pB[7], _pA, vl);
                    _sum8 = __riscv_vfmacc_vf_f16m1(_sum8, pB[8], _pA, vl);
                    _sum9 = __riscv_vfmacc_vf_f16m1(_sum9, pB[9], _pA, vl);
                    _suma = __riscv_vfmacc_vf_f16m1(_suma, pB[10], _pA, vl);
                    _sumb = __riscv_vfmacc_vf_f16m1(_sumb, pB[11], _pA, vl);
                    _sumc = __riscv_vfmacc_vf_f16m1(_sumc, pB[12], _pA, vl);
                    _sumd = __riscv_vfmacc_vf_f16m1(_sumd, pB[13], _pA, vl);
                    _sume = __riscv_vfmacc_vf_f16m1(_sume, pB[14], _pA, vl);
                    _sumf = __riscv_vfmacc_vf_f16m1(_sumf, pB[15], _pA, vl);

                    pA += packn;
                    pB += 16;
                }

                __riscv_vse16_v_f16m1(outptr, _sum0, vl);
                __riscv_vse16_v_f16m1(outptr + packn, _sum1, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 2, _sum2, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 3, _sum3, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 4, _sum4, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 5, _sum5, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 6, _sum6, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 7, _sum7, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 8, _sum8, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 9, _sum9, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 10, _suma, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 11, _sumb, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 12, _sumc, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 13, _sumd, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 14, _sume, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 15, _sumf, vl);
                outptr += packn * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                const __fp16* pA = pAT;

                vfloat16m1_t _sum0;
                vfloat16m1_t _sum1;
                vfloat16m1_t _sum2;
                vfloat16m1_t _sum3;
                vfloat16m1_t _sum4;
                vfloat16m1_t _sum5;
                vfloat16m1_t _sum6;
                vfloat16m1_t _sum7;

                if (k == 0)
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);
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
                    _sum0 = __riscv_vle16_v_f16m1(outptr, vl);
                    _sum1 = __riscv_vle16_v_f16m1(outptr + packn, vl);
                    _sum2 = __riscv_vle16_v_f16m1(outptr + packn * 2, vl);
                    _sum3 = __riscv_vle16_v_f16m1(outptr + packn * 3, vl);
                    _sum4 = __riscv_vle16_v_f16m1(outptr + packn * 4, vl);
                    _sum5 = __riscv_vle16_v_f16m1(outptr + packn * 5, vl);
                    _sum6 = __riscv_vle16_v_f16m1(outptr + packn * 6, vl);
                    _sum7 = __riscv_vle16_v_f16m1(outptr + packn * 7, vl);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    vfloat16m1_t _pA = __riscv_vle16_v_f16m1(pA, vl);
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);
                    _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pB[1], _pA, vl);
                    _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, pB[2], _pA, vl);
                    _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, pB[3], _pA, vl);
                    _sum4 = __riscv_vfmacc_vf_f16m1(_sum4, pB[4], _pA, vl);
                    _sum5 = __riscv_vfmacc_vf_f16m1(_sum5, pB[5], _pA, vl);
                    _sum6 = __riscv_vfmacc_vf_f16m1(_sum6, pB[6], _pA, vl);
                    _sum7 = __riscv_vfmacc_vf_f16m1(_sum7, pB[7], _pA, vl);

                    pA += packn;
                    pB += 8;
                }

                __riscv_vse16_v_f16m1(outptr, _sum0, vl);
                __riscv_vse16_v_f16m1(outptr + packn, _sum1, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 2, _sum2, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 3, _sum3, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 4, _sum4, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 5, _sum5, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 6, _sum6, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 7, _sum7, vl);
                outptr += packn * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const __fp16* pA = pAT;

                vfloat16m1_t _sum0;
                vfloat16m1_t _sum1;
                vfloat16m1_t _sum2;
                vfloat16m1_t _sum3;

                if (k == 0)
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
                else
                {
                    _sum0 = __riscv_vle16_v_f16m1(outptr, vl);
                    _sum1 = __riscv_vle16_v_f16m1(outptr + packn, vl);
                    _sum2 = __riscv_vle16_v_f16m1(outptr + packn * 2, vl);
                    _sum3 = __riscv_vle16_v_f16m1(outptr + packn * 3, vl);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    vfloat16m1_t _pA = __riscv_vle16_v_f16m1(pA, vl);
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);
                    _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pB[1], _pA, vl);
                    _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, pB[2], _pA, vl);
                    _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, pB[3], _pA, vl);

                    pA += packn;
                    pB += 4;
                }

                __riscv_vse16_v_f16m1(outptr, _sum0, vl);
                __riscv_vse16_v_f16m1(outptr + packn, _sum1, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 2, _sum2, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 3, _sum3, vl);
                outptr += packn * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const __fp16* pA = pAT;

                vfloat16m1_t _sum0;
                vfloat16m1_t _sum1;

                if (k == 0)
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);
                    _sum1 = _sum0;
                }
                else
                {
                    _sum0 = __riscv_vle16_v_f16m1(outptr, vl);
                    _sum1 = __riscv_vle16_v_f16m1(outptr + packn, vl);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    vfloat16m1_t _pA = __riscv_vle16_v_f16m1(pA, vl);
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);
                    _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pB[1], _pA, vl);

                    pA += packn;
                    pB += 2;
                }

                __riscv_vse16_v_f16m1(outptr, _sum0, vl);
                __riscv_vse16_v_f16m1(outptr + packn, _sum1, vl);
                outptr += packn * 2;
            }
            for (; jj < max_jj; jj++)
            {
                const __fp16* pA = pAT;

                vfloat16m1_t _sum;

                if (k == 0)
                    _sum = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);
                else
                    _sum = __riscv_vle16_v_f16m1(outptr, vl);

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    vfloat16m1_t _pA = __riscv_vle16_v_f16m1(pA, vl);
                    _sum = __riscv_vfmacc_vf_f16m1(_sum, pB[0], _pA, vl);

                    pA += packn;
                    pB += 1;
                }

                __riscv_vse16_v_f16m1(outptr, _sum, vl);
                outptr += packn;
            }
        }
    }

    for (; ii + 7 < max_ii; ii += 8)
    {
        __fp16* outptr0_base = outptr;
        __fp16* outptr1_base = outptr + max_jj * batch * 4;

        for (int b = 0; b < batch; b++)
        {
            const __fp16* pAT = AT_tile.row<__fp16>(b) + max_kk * ii;
            const __fp16* pB = BT_tile.row<__fp16>(b);

            __fp16* outptr0 = outptr0_base + b * max_jj * 4;
            __fp16* outptr1 = outptr1_base + b * max_jj * 4;

            int jj = 0;
            for (; jj + 15 < max_jj; jj += 16)
            {
                const __fp16* pA = pAT;

                const size_t vl16 = __riscv_vsetvl_e16m1(16);

                vfloat16m1_t _sum0;
                vfloat16m1_t _sum1;
                vfloat16m1_t _sum2;
                vfloat16m1_t _sum3;
                vfloat16m1_t _sum4;
                vfloat16m1_t _sum5;
                vfloat16m1_t _sum6;
                vfloat16m1_t _sum7;

                if (k == 0)
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl16);
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
                    _sum0 = __riscv_vlse16_v_f16m1(outptr0, 4 * sizeof(__fp16), vl16);
                    _sum1 = __riscv_vlse16_v_f16m1(outptr0 + 1, 4 * sizeof(__fp16), vl16);
                    _sum2 = __riscv_vlse16_v_f16m1(outptr0 + 2, 4 * sizeof(__fp16), vl16);
                    _sum3 = __riscv_vlse16_v_f16m1(outptr0 + 3, 4 * sizeof(__fp16), vl16);
                    _sum4 = __riscv_vlse16_v_f16m1(outptr1, 4 * sizeof(__fp16), vl16);
                    _sum5 = __riscv_vlse16_v_f16m1(outptr1 + 1, 4 * sizeof(__fp16), vl16);
                    _sum6 = __riscv_vlse16_v_f16m1(outptr1 + 2, 4 * sizeof(__fp16), vl16);
                    _sum7 = __riscv_vlse16_v_f16m1(outptr1 + 3, 4 * sizeof(__fp16), vl16);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    vfloat16m1_t _val = __riscv_vle16_v_f16m1(pB, vl16);
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], _val, vl16);
                    _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pA[1], _val, vl16);
                    _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, pA[2], _val, vl16);
                    _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, pA[3], _val, vl16);
                    _sum4 = __riscv_vfmacc_vf_f16m1(_sum4, pA[4], _val, vl16);
                    _sum5 = __riscv_vfmacc_vf_f16m1(_sum5, pA[5], _val, vl16);
                    _sum6 = __riscv_vfmacc_vf_f16m1(_sum6, pA[6], _val, vl16);
                    _sum7 = __riscv_vfmacc_vf_f16m1(_sum7, pA[7], _val, vl16);

                    pA += 8;
                    pB += 16;
                }

                __riscv_vsse16_v_f16m1(outptr0, 4 * sizeof(__fp16), _sum0, vl16);
                __riscv_vsse16_v_f16m1(outptr0 + 1, 4 * sizeof(__fp16), _sum1, vl16);
                __riscv_vsse16_v_f16m1(outptr0 + 2, 4 * sizeof(__fp16), _sum2, vl16);
                __riscv_vsse16_v_f16m1(outptr0 + 3, 4 * sizeof(__fp16), _sum3, vl16);
                __riscv_vsse16_v_f16m1(outptr1, 4 * sizeof(__fp16), _sum4, vl16);
                __riscv_vsse16_v_f16m1(outptr1 + 1, 4 * sizeof(__fp16), _sum5, vl16);
                __riscv_vsse16_v_f16m1(outptr1 + 2, 4 * sizeof(__fp16), _sum6, vl16);
                __riscv_vsse16_v_f16m1(outptr1 + 3, 4 * sizeof(__fp16), _sum7, vl16);

                outptr0 += 4 * 16;
                outptr1 += 4 * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                const __fp16* pA = pAT;

                const size_t vl8 = __riscv_vsetvl_e16m1(8);

                vfloat16m1_t _sum0;
                vfloat16m1_t _sum1;
                vfloat16m1_t _sum2;
                vfloat16m1_t _sum3;
                vfloat16m1_t _sum4;
                vfloat16m1_t _sum5;
                vfloat16m1_t _sum6;
                vfloat16m1_t _sum7;

                if (k == 0)
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl8);
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
                    _sum0 = __riscv_vlse16_v_f16m1(outptr0, 4 * sizeof(__fp16), vl8);
                    _sum1 = __riscv_vlse16_v_f16m1(outptr0 + 1, 4 * sizeof(__fp16), vl8);
                    _sum2 = __riscv_vlse16_v_f16m1(outptr0 + 2, 4 * sizeof(__fp16), vl8);
                    _sum3 = __riscv_vlse16_v_f16m1(outptr0 + 3, 4 * sizeof(__fp16), vl8);
                    _sum4 = __riscv_vlse16_v_f16m1(outptr1, 4 * sizeof(__fp16), vl8);
                    _sum5 = __riscv_vlse16_v_f16m1(outptr1 + 1, 4 * sizeof(__fp16), vl8);
                    _sum6 = __riscv_vlse16_v_f16m1(outptr1 + 2, 4 * sizeof(__fp16), vl8);
                    _sum7 = __riscv_vlse16_v_f16m1(outptr1 + 3, 4 * sizeof(__fp16), vl8);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    vfloat16m1_t _val = __riscv_vle16_v_f16m1(pB, vl8);
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], _val, vl8);
                    _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pA[1], _val, vl8);
                    _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, pA[2], _val, vl8);
                    _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, pA[3], _val, vl8);
                    _sum4 = __riscv_vfmacc_vf_f16m1(_sum4, pA[4], _val, vl8);
                    _sum5 = __riscv_vfmacc_vf_f16m1(_sum5, pA[5], _val, vl8);
                    _sum6 = __riscv_vfmacc_vf_f16m1(_sum6, pA[6], _val, vl8);
                    _sum7 = __riscv_vfmacc_vf_f16m1(_sum7, pA[7], _val, vl8);

                    pA += 8;
                    pB += 8;
                }

                __riscv_vsse16_v_f16m1(outptr0, 4 * sizeof(__fp16), _sum0, vl8);
                __riscv_vsse16_v_f16m1(outptr0 + 1, 4 * sizeof(__fp16), _sum1, vl8);
                __riscv_vsse16_v_f16m1(outptr0 + 2, 4 * sizeof(__fp16), _sum2, vl8);
                __riscv_vsse16_v_f16m1(outptr0 + 3, 4 * sizeof(__fp16), _sum3, vl8);
                __riscv_vsse16_v_f16m1(outptr1, 4 * sizeof(__fp16), _sum4, vl8);
                __riscv_vsse16_v_f16m1(outptr1 + 1, 4 * sizeof(__fp16), _sum5, vl8);
                __riscv_vsse16_v_f16m1(outptr1 + 2, 4 * sizeof(__fp16), _sum6, vl8);
                __riscv_vsse16_v_f16m1(outptr1 + 3, 4 * sizeof(__fp16), _sum7, vl8);

                outptr0 += 4 * 8;
                outptr1 += 4 * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const __fp16* pA = pAT;

                const size_t vl4 = __riscv_vsetvl_e16m1(4);

                vfloat16m1_t _sum0;
                vfloat16m1_t _sum1;
                vfloat16m1_t _sum2;
                vfloat16m1_t _sum3;
                vfloat16m1_t _sum4;
                vfloat16m1_t _sum5;
                vfloat16m1_t _sum6;
                vfloat16m1_t _sum7;

                if (k == 0)
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl4);
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
                    _sum0 = __riscv_vlse16_v_f16m1(outptr0, 4 * sizeof(__fp16), vl4);
                    _sum1 = __riscv_vlse16_v_f16m1(outptr0 + 1, 4 * sizeof(__fp16), vl4);
                    _sum2 = __riscv_vlse16_v_f16m1(outptr0 + 2, 4 * sizeof(__fp16), vl4);
                    _sum3 = __riscv_vlse16_v_f16m1(outptr0 + 3, 4 * sizeof(__fp16), vl4);
                    _sum4 = __riscv_vlse16_v_f16m1(outptr1, 4 * sizeof(__fp16), vl4);
                    _sum5 = __riscv_vlse16_v_f16m1(outptr1 + 1, 4 * sizeof(__fp16), vl4);
                    _sum6 = __riscv_vlse16_v_f16m1(outptr1 + 2, 4 * sizeof(__fp16), vl4);
                    _sum7 = __riscv_vlse16_v_f16m1(outptr1 + 3, 4 * sizeof(__fp16), vl4);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    vfloat16m1_t _val = __riscv_vle16_v_f16m1(pB, vl4);
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], _val, vl4);
                    _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pA[1], _val, vl4);
                    _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, pA[2], _val, vl4);
                    _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, pA[3], _val, vl4);
                    _sum4 = __riscv_vfmacc_vf_f16m1(_sum4, pA[4], _val, vl4);
                    _sum5 = __riscv_vfmacc_vf_f16m1(_sum5, pA[5], _val, vl4);
                    _sum6 = __riscv_vfmacc_vf_f16m1(_sum6, pA[6], _val, vl4);
                    _sum7 = __riscv_vfmacc_vf_f16m1(_sum7, pA[7], _val, vl4);

                    pA += 8;
                    pB += 4;
                }

                __riscv_vsse16_v_f16m1(outptr0, 4 * sizeof(__fp16), _sum0, vl4);
                __riscv_vsse16_v_f16m1(outptr0 + 1, 4 * sizeof(__fp16), _sum1, vl4);
                __riscv_vsse16_v_f16m1(outptr0 + 2, 4 * sizeof(__fp16), _sum2, vl4);
                __riscv_vsse16_v_f16m1(outptr0 + 3, 4 * sizeof(__fp16), _sum3, vl4);
                __riscv_vsse16_v_f16m1(outptr1, 4 * sizeof(__fp16), _sum4, vl4);
                __riscv_vsse16_v_f16m1(outptr1 + 1, 4 * sizeof(__fp16), _sum5, vl4);
                __riscv_vsse16_v_f16m1(outptr1 + 2, 4 * sizeof(__fp16), _sum6, vl4);
                __riscv_vsse16_v_f16m1(outptr1 + 3, 4 * sizeof(__fp16), _sum7, vl4);

                outptr0 += 4 * 4;
                outptr1 += 4 * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const __fp16* pA = pAT;

                __fp16 sum00 = (__fp16)0.f;
                __fp16 sum01 = (__fp16)0.f;
                __fp16 sum02 = (__fp16)0.f;
                __fp16 sum03 = (__fp16)0.f;
                __fp16 sum04 = (__fp16)0.f;
                __fp16 sum05 = (__fp16)0.f;
                __fp16 sum06 = (__fp16)0.f;
                __fp16 sum07 = (__fp16)0.f;
                __fp16 sum10 = (__fp16)0.f;
                __fp16 sum11 = (__fp16)0.f;
                __fp16 sum12 = (__fp16)0.f;
                __fp16 sum13 = (__fp16)0.f;
                __fp16 sum14 = (__fp16)0.f;
                __fp16 sum15 = (__fp16)0.f;
                __fp16 sum16 = (__fp16)0.f;
                __fp16 sum17 = (__fp16)0.f;

                if (k != 0)
                {
                    sum00 = outptr0[0];
                    sum01 = outptr0[1];
                    sum02 = outptr0[2];
                    sum03 = outptr0[3];
                    sum10 = outptr0[4];
                    sum11 = outptr0[5];
                    sum12 = outptr0[6];
                    sum13 = outptr0[7];
                    sum04 = outptr1[0];
                    sum05 = outptr1[1];
                    sum06 = outptr1[2];
                    sum07 = outptr1[3];
                    sum14 = outptr1[4];
                    sum15 = outptr1[5];
                    sum16 = outptr1[6];
                    sum17 = outptr1[7];
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    sum00 += pA[0] * pB[0];
                    sum01 += pA[1] * pB[0];
                    sum02 += pA[2] * pB[0];
                    sum03 += pA[3] * pB[0];
                    sum04 += pA[4] * pB[0];
                    sum05 += pA[5] * pB[0];
                    sum06 += pA[6] * pB[0];
                    sum07 += pA[7] * pB[0];
                    sum10 += pA[0] * pB[1];
                    sum11 += pA[1] * pB[1];
                    sum12 += pA[2] * pB[1];
                    sum13 += pA[3] * pB[1];
                    sum14 += pA[4] * pB[1];
                    sum15 += pA[5] * pB[1];
                    sum16 += pA[6] * pB[1];
                    sum17 += pA[7] * pB[1];
                    pA += 8;
                    pB += 2;
                }

                outptr0[0] = sum00;
                outptr0[1] = sum01;
                outptr0[2] = sum02;
                outptr0[3] = sum03;
                outptr0[4] = sum10;
                outptr0[5] = sum11;
                outptr0[6] = sum12;
                outptr0[7] = sum13;
                outptr1[0] = sum04;
                outptr1[1] = sum05;
                outptr1[2] = sum06;
                outptr1[3] = sum07;
                outptr1[4] = sum14;
                outptr1[5] = sum15;
                outptr1[6] = sum16;
                outptr1[7] = sum17;

                outptr0 += 4 * 2;
                outptr1 += 4 * 2;
            }
            for (; jj < max_jj; jj++)
            {
                const __fp16* pA = pAT;

                __fp16 sum0 = (__fp16)0.f;
                __fp16 sum1 = (__fp16)0.f;
                __fp16 sum2 = (__fp16)0.f;
                __fp16 sum3 = (__fp16)0.f;
                __fp16 sum4 = (__fp16)0.f;
                __fp16 sum5 = (__fp16)0.f;
                __fp16 sum6 = (__fp16)0.f;
                __fp16 sum7 = (__fp16)0.f;

                if (k != 0)
                {
                    sum0 = outptr0[0];
                    sum1 = outptr0[1];
                    sum2 = outptr0[2];
                    sum3 = outptr0[3];
                    sum4 = outptr1[0];
                    sum5 = outptr1[1];
                    sum6 = outptr1[2];
                    sum7 = outptr1[3];
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    sum0 += pA[0] * pB[0];
                    sum1 += pA[1] * pB[0];
                    sum2 += pA[2] * pB[0];
                    sum3 += pA[3] * pB[0];
                    sum4 += pA[4] * pB[0];
                    sum5 += pA[5] * pB[0];
                    sum6 += pA[6] * pB[0];
                    sum7 += pA[7] * pB[0];
                    pA += 8;
                    pB += 1;
                }

                outptr0[0] = sum0;
                outptr0[1] = sum1;
                outptr0[2] = sum2;
                outptr0[3] = sum3;
                outptr1[0] = sum4;
                outptr1[1] = sum5;
                outptr1[2] = sum6;
                outptr1[3] = sum7;

                outptr0 += 4;
                outptr1 += 4;
            }
        }

        outptr += max_jj * batch * 8;
    }

    for (; ii + 3 < max_ii; ii += 4)
    {
        for (int b = 0; b < batch; b++)
        {
            const __fp16* pAT = AT_tile.row<__fp16>(b) + max_kk * ii;
            const __fp16* pB = BT_tile.row<__fp16>(b);

            int jj = 0;
            for (; jj + 15 < max_jj; jj += 16)
            {
                const __fp16* pA = pAT;

                if (packn == 8)
                {
                    const size_t vl16 = __riscv_vsetvl_e16m2(16);

                    vfloat16m2_t _sum0;
                    vfloat16m2_t _sum1;
                    vfloat16m2_t _sum2;
                    vfloat16m2_t _sum3;

                    if (k == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m2((__fp16)0.f, vl16);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    else
                    {
                        _sum0 = __riscv_vlse16_v_f16m2(outptr, 4 * sizeof(__fp16), vl16);
                        _sum1 = __riscv_vlse16_v_f16m2(outptr + 1, 4 * sizeof(__fp16), vl16);
                        _sum2 = __riscv_vlse16_v_f16m2(outptr + 2, 4 * sizeof(__fp16), vl16);
                        _sum3 = __riscv_vlse16_v_f16m2(outptr + 3, 4 * sizeof(__fp16), vl16);
                    }

                    int kk = 0;
                    for (; kk < max_kk; kk++)
                    {
                        vfloat16m2_t _val = __riscv_vle16_v_f16m2(pB, vl16);
                        _sum0 = __riscv_vfmacc_vf_f16m2(_sum0, pA[0], _val, vl16);
                        _sum1 = __riscv_vfmacc_vf_f16m2(_sum1, pA[1], _val, vl16);
                        _sum2 = __riscv_vfmacc_vf_f16m2(_sum2, pA[2], _val, vl16);
                        _sum3 = __riscv_vfmacc_vf_f16m2(_sum3, pA[3], _val, vl16);

                        pA += 4;
                        pB += 16;
                    }

                    __riscv_vsse16_v_f16m2(outptr, 4 * sizeof(__fp16), _sum0, vl16);
                    __riscv_vsse16_v_f16m2(outptr + 1, 4 * sizeof(__fp16), _sum1, vl16);
                    __riscv_vsse16_v_f16m2(outptr + 2, 4 * sizeof(__fp16), _sum2, vl16);
                    __riscv_vsse16_v_f16m2(outptr + 3, 4 * sizeof(__fp16), _sum3, vl16);
                }
                else
                {
                    const size_t vl16 = __riscv_vsetvl_e16m1(16);

                    vfloat16m1_t _sum0;
                    vfloat16m1_t _sum1;
                    vfloat16m1_t _sum2;
                    vfloat16m1_t _sum3;

                    if (k == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl16);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    else
                    {
                        _sum0 = __riscv_vlse16_v_f16m1(outptr, 4 * sizeof(__fp16), vl16);
                        _sum1 = __riscv_vlse16_v_f16m1(outptr + 1, 4 * sizeof(__fp16), vl16);
                        _sum2 = __riscv_vlse16_v_f16m1(outptr + 2, 4 * sizeof(__fp16), vl16);
                        _sum3 = __riscv_vlse16_v_f16m1(outptr + 3, 4 * sizeof(__fp16), vl16);
                    }

                    int kk = 0;
                    for (; kk < max_kk; kk++)
                    {
                        vfloat16m1_t _val = __riscv_vle16_v_f16m1(pB, vl16);
                        _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], _val, vl16);
                        _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pA[1], _val, vl16);
                        _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, pA[2], _val, vl16);
                        _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, pA[3], _val, vl16);

                        pA += 4;
                        pB += 16;
                    }

                    __riscv_vsse16_v_f16m1(outptr, 4 * sizeof(__fp16), _sum0, vl16);
                    __riscv_vsse16_v_f16m1(outptr + 1, 4 * sizeof(__fp16), _sum1, vl16);
                    __riscv_vsse16_v_f16m1(outptr + 2, 4 * sizeof(__fp16), _sum2, vl16);
                    __riscv_vsse16_v_f16m1(outptr + 3, 4 * sizeof(__fp16), _sum3, vl16);
                }

                outptr += 4 * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                const __fp16* pA = pAT;

                const size_t vl8 = __riscv_vsetvl_e16m1(8);

                vfloat16m1_t _sum0;
                vfloat16m1_t _sum1;
                vfloat16m1_t _sum2;
                vfloat16m1_t _sum3;

                if (k == 0)
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl8);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
                else
                {
                    _sum0 = __riscv_vlse16_v_f16m1(outptr, 4 * sizeof(__fp16), vl8);
                    _sum1 = __riscv_vlse16_v_f16m1(outptr + 1, 4 * sizeof(__fp16), vl8);
                    _sum2 = __riscv_vlse16_v_f16m1(outptr + 2, 4 * sizeof(__fp16), vl8);
                    _sum3 = __riscv_vlse16_v_f16m1(outptr + 3, 4 * sizeof(__fp16), vl8);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    vfloat16m1_t _val = __riscv_vle16_v_f16m1(pB, vl8);
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], _val, vl8);
                    _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pA[1], _val, vl8);
                    _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, pA[2], _val, vl8);
                    _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, pA[3], _val, vl8);

                    pA += 4;
                    pB += 8;
                }

                __riscv_vsse16_v_f16m1(outptr, 4 * sizeof(__fp16), _sum0, vl8);
                __riscv_vsse16_v_f16m1(outptr + 1, 4 * sizeof(__fp16), _sum1, vl8);
                __riscv_vsse16_v_f16m1(outptr + 2, 4 * sizeof(__fp16), _sum2, vl8);
                __riscv_vsse16_v_f16m1(outptr + 3, 4 * sizeof(__fp16), _sum3, vl8);

                outptr += 4 * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const __fp16* pA = pAT;

                const size_t vl4 = __riscv_vsetvl_e16m1(4);

                vfloat16m1_t _sum0;
                vfloat16m1_t _sum1;
                vfloat16m1_t _sum2;
                vfloat16m1_t _sum3;

                if (k == 0)
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl4);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
                else
                {
                    _sum0 = __riscv_vlse16_v_f16m1(outptr, 4 * sizeof(__fp16), vl4);
                    _sum1 = __riscv_vlse16_v_f16m1(outptr + 1, 4 * sizeof(__fp16), vl4);
                    _sum2 = __riscv_vlse16_v_f16m1(outptr + 2, 4 * sizeof(__fp16), vl4);
                    _sum3 = __riscv_vlse16_v_f16m1(outptr + 3, 4 * sizeof(__fp16), vl4);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    vfloat16m1_t _val = __riscv_vle16_v_f16m1(pB, vl4);
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], _val, vl4);
                    _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pA[1], _val, vl4);
                    _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, pA[2], _val, vl4);
                    _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, pA[3], _val, vl4);

                    pA += 4;
                    pB += 4;
                }

                __riscv_vsse16_v_f16m1(outptr, 4 * sizeof(__fp16), _sum0, vl4);
                __riscv_vsse16_v_f16m1(outptr + 1, 4 * sizeof(__fp16), _sum1, vl4);
                __riscv_vsse16_v_f16m1(outptr + 2, 4 * sizeof(__fp16), _sum2, vl4);
                __riscv_vsse16_v_f16m1(outptr + 3, 4 * sizeof(__fp16), _sum3, vl4);

                outptr += 4 * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const __fp16* pA = pAT;

                __fp16 sum00 = (__fp16)0.f;
                __fp16 sum01 = (__fp16)0.f;
                __fp16 sum02 = (__fp16)0.f;
                __fp16 sum03 = (__fp16)0.f;
                __fp16 sum10 = (__fp16)0.f;
                __fp16 sum11 = (__fp16)0.f;
                __fp16 sum12 = (__fp16)0.f;
                __fp16 sum13 = (__fp16)0.f;

                if (k != 0)
                {
                    sum00 = outptr[0];
                    sum01 = outptr[1];
                    sum02 = outptr[2];
                    sum03 = outptr[3];
                    sum10 = outptr[4];
                    sum11 = outptr[5];
                    sum12 = outptr[6];
                    sum13 = outptr[7];
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    sum00 += pA[0] * pB[0];
                    sum01 += pA[1] * pB[0];
                    sum02 += pA[2] * pB[0];
                    sum03 += pA[3] * pB[0];
                    sum10 += pA[0] * pB[1];
                    sum11 += pA[1] * pB[1];
                    sum12 += pA[2] * pB[1];
                    sum13 += pA[3] * pB[1];
                    pA += 4;
                    pB += 2;
                }

                outptr[0] = sum00;
                outptr[1] = sum01;
                outptr[2] = sum02;
                outptr[3] = sum03;
                outptr[4] = sum10;
                outptr[5] = sum11;
                outptr[6] = sum12;
                outptr[7] = sum13;

                outptr += 4 * 2;
            }
            for (; jj < max_jj; jj++)
            {
                const __fp16* pA = pAT;

                __fp16 sum0 = (__fp16)0.f;
                __fp16 sum1 = (__fp16)0.f;
                __fp16 sum2 = (__fp16)0.f;
                __fp16 sum3 = (__fp16)0.f;

                if (k != 0)
                {
                    sum0 = outptr[0];
                    sum1 = outptr[1];
                    sum2 = outptr[2];
                    sum3 = outptr[3];
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    sum0 += pA[0] * pB[0];
                    sum1 += pA[1] * pB[0];
                    sum2 += pA[2] * pB[0];
                    sum3 += pA[3] * pB[0];
                    pA += 4;
                    pB += 1;
                }

                outptr[0] = sum0;
                outptr[1] = sum1;
                outptr[2] = sum2;
                outptr[3] = sum3;

                outptr += 4;
            }
        }
    }
#endif // __riscv_zvfh

    for (; ii + 1 < max_ii; ii += 2)
    {
        for (int b = 0; b < batch; b++)
        {
            const __fp16* pAT = AT_tile.row<__fp16>(b) + max_kk * ii;
            const __fp16* pB = BT_tile.row<__fp16>(b);

            int jj = 0;
#if __riscv_zvfh
            for (; jj + 15 < max_jj; jj += 16)
            {
                const __fp16* pA = pAT;

                if (packn == 8)
                {
                    const size_t vl16 = __riscv_vsetvl_e16m2(16);

                    vfloat16m2_t _sum0;
                    vfloat16m2_t _sum1;

                    if (k == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m2((__fp16)0.f, vl16);
                        _sum1 = _sum0;
                    }
                    else
                    {
                        _sum0 = __riscv_vlse16_v_f16m2(outptr, 2 * sizeof(__fp16), vl16);
                        _sum1 = __riscv_vlse16_v_f16m2(outptr + 1, 2 * sizeof(__fp16), vl16);
                    }

                    int kk = 0;
                    for (; kk < max_kk; kk++)
                    {
                        vfloat16m2_t _val = __riscv_vle16_v_f16m2(pB, vl16);
                        _sum0 = __riscv_vfmacc_vf_f16m2(_sum0, pA[0], _val, vl16);
                        _sum1 = __riscv_vfmacc_vf_f16m2(_sum1, pA[1], _val, vl16);

                        pA += 2;
                        pB += 16;
                    }

                    __riscv_vsse16_v_f16m2(outptr, 2 * sizeof(__fp16), _sum0, vl16);
                    __riscv_vsse16_v_f16m2(outptr + 1, 2 * sizeof(__fp16), _sum1, vl16);
                }
                else
                {
                    const size_t vl16 = __riscv_vsetvl_e16m1(16);

                    vfloat16m1_t _sum0;
                    vfloat16m1_t _sum1;

                    if (k == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl16);
                        _sum1 = _sum0;
                    }
                    else
                    {
                        _sum0 = __riscv_vlse16_v_f16m1(outptr, 2 * sizeof(__fp16), vl16);
                        _sum1 = __riscv_vlse16_v_f16m1(outptr + 1, 2 * sizeof(__fp16), vl16);
                    }

                    int kk = 0;
                    for (; kk < max_kk; kk++)
                    {
                        vfloat16m1_t _val = __riscv_vle16_v_f16m1(pB, vl16);
                        _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], _val, vl16);
                        _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pA[1], _val, vl16);

                        pA += 2;
                        pB += 16;
                    }

                    __riscv_vsse16_v_f16m1(outptr, 2 * sizeof(__fp16), _sum0, vl16);
                    __riscv_vsse16_v_f16m1(outptr + 1, 2 * sizeof(__fp16), _sum1, vl16);
                }

                outptr += 2 * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                const __fp16* pA = pAT;

                const size_t vl8 = __riscv_vsetvl_e16m1(8);

                vfloat16m1_t _sum0;
                vfloat16m1_t _sum1;

                if (k == 0)
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl8);
                    _sum1 = _sum0;
                }
                else
                {
                    _sum0 = __riscv_vlse16_v_f16m1(outptr, 2 * sizeof(__fp16), vl8);
                    _sum1 = __riscv_vlse16_v_f16m1(outptr + 1, 2 * sizeof(__fp16), vl8);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    vfloat16m1_t _val = __riscv_vle16_v_f16m1(pB, vl8);
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], _val, vl8);
                    _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pA[1], _val, vl8);

                    pA += 2;
                    pB += 8;
                }

                __riscv_vsse16_v_f16m1(outptr, 2 * sizeof(__fp16), _sum0, vl8);
                __riscv_vsse16_v_f16m1(outptr + 1, 2 * sizeof(__fp16), _sum1, vl8);
                outptr += 2 * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const __fp16* pA = pAT;

                const size_t vl4 = __riscv_vsetvl_e16m1(4);

                vfloat16m1_t _sum0;
                vfloat16m1_t _sum1;

                if (k == 0)
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl4);
                    _sum1 = _sum0;
                }
                else
                {
                    _sum0 = __riscv_vlse16_v_f16m1(outptr, 2 * sizeof(__fp16), vl4);
                    _sum1 = __riscv_vlse16_v_f16m1(outptr + 1, 2 * sizeof(__fp16), vl4);
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    vfloat16m1_t _val = __riscv_vle16_v_f16m1(pB, vl4);
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], _val, vl4);
                    _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pA[1], _val, vl4);

                    pA += 2;
                    pB += 4;
                }

                __riscv_vsse16_v_f16m1(outptr, 2 * sizeof(__fp16), _sum0, vl4);
                __riscv_vsse16_v_f16m1(outptr + 1, 2 * sizeof(__fp16), _sum1, vl4);
                outptr += 2 * 4;
            }
#endif // __riscv_zvfh
            for (; jj + 1 < max_jj; jj += 2)
            {
                const __fp16* pA = pAT;

                __fp16 sum00 = (__fp16)0.f;
                __fp16 sum01 = (__fp16)0.f;
                __fp16 sum10 = (__fp16)0.f;
                __fp16 sum11 = (__fp16)0.f;

                if (k == 0)
                {
                    sum00 = (__fp16)0.f;
                    sum01 = (__fp16)0.f;
                    sum10 = (__fp16)0.f;
                    sum11 = (__fp16)0.f;
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
                const __fp16* pA = pAT;

                __fp16 sum0 = (__fp16)0.f;
                __fp16 sum1 = (__fp16)0.f;

                if (k == 0)
                {
                    sum0 = (__fp16)0.f;
                    sum1 = (__fp16)0.f;
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
            const __fp16* pAT = AT_tile.row<__fp16>(b) + max_kk * ii;
            const __fp16* pB = BT_tile.row<__fp16>(b);

            int jj = 0;
#if __riscv_zvfh
            for (; jj + 15 < max_jj; jj += 16)
            {
                const __fp16* pA = pAT;

                if (packn == 8)
                {
                    const size_t vl16 = __riscv_vsetvl_e16m2(16);

                    vfloat16m2_t _sum0;

                    if (k == 0)
                        _sum0 = __riscv_vfmv_v_f_f16m2((__fp16)0.f, vl16);
                    else
                        _sum0 = __riscv_vle16_v_f16m2(outptr, vl16);

                    int kk = 0;
                    for (; kk < max_kk; kk++)
                    {
                        _sum0 = __riscv_vfmacc_vf_f16m2(_sum0, pA[0], __riscv_vle16_v_f16m2(pB, vl16), vl16);

                        pA += 1;
                        pB += 16;
                    }

                    __riscv_vse16_v_f16m2(outptr, _sum0, vl16);
                }
                else
                {
                    const size_t vl16 = __riscv_vsetvl_e16m1(16);

                    vfloat16m1_t _sum0;

                    if (k == 0)
                        _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl16);
                    else
                        _sum0 = __riscv_vle16_v_f16m1(outptr, vl16);

                    int kk = 0;
                    for (; kk < max_kk; kk++)
                    {
                        _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], __riscv_vle16_v_f16m1(pB, vl16), vl16);

                        pA += 1;
                        pB += 16;
                    }

                    __riscv_vse16_v_f16m1(outptr, _sum0, vl16);
                }

                outptr += 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                const __fp16* pA = pAT;

                const size_t vl8 = __riscv_vsetvl_e16m1(8);

                vfloat16m1_t _sum0;

                if (k == 0)
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl8);
                else
                    _sum0 = __riscv_vle16_v_f16m1(outptr, vl8);

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], __riscv_vle16_v_f16m1(pB, vl8), vl8);

                    pA += 1;
                    pB += 8;
                }

                __riscv_vse16_v_f16m1(outptr, _sum0, vl8);
                outptr += 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const __fp16* pA = pAT;

                const size_t vl4 = __riscv_vsetvl_e16m1(4);

                vfloat16m1_t _sum0;

                if (k == 0)
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl4);
                else
                    _sum0 = __riscv_vle16_v_f16m1(outptr, vl4);

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], __riscv_vle16_v_f16m1(pB, vl4), vl4);

                    pA += 1;
                    pB += 4;
                }

                __riscv_vse16_v_f16m1(outptr, _sum0, vl4);
                outptr += 4;
            }
#endif // __riscv_zvfh
            for (; jj + 1 < max_jj; jj += 2)
            {
                const __fp16* pA = pAT;

                __fp16 sum0 = (__fp16)0.f;
                __fp16 sum1 = (__fp16)0.f;

                if (k == 0)
                {
                    sum0 = (__fp16)0.f;
                    sum1 = (__fp16)0.f;
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
                const __fp16* pA = pAT;

                __fp16 sum = (__fp16)0.f;

                if (k == 0)
                {
                    sum = (__fp16)0.f;
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

static void conv3x3s1_winograd_get_optimal_tile_mnk_fp16sa_rvv(int M, int N, int K, int B, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const int l2_cache_size_fp16 = (int)(get_cpu_level2_cache_size() / sizeof(__fp16));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // we shall take B into account for batched gemm, but that will be slower on arm in practice, why ?
    (void)B;

#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
#endif

    // solve K
    {
        // try not to split K
#if __riscv_zvfh
        int tile_size = (l2_cache_size_fp16 - 16 * packn) / (16 + packn);
#else
        int tile_size = (l2_cache_size_fp16 - 2) / 3;
#endif

#if __riscv_zvfh
        TILE_K = std::max(packn, tile_size / packn * packn);
#else
        TILE_K = std::max(2, tile_size / 2 * 2);
#endif

        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __riscv_zvfh
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + packn - 1) / packn * packn);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif
    }

    // solve M
    {
#if __riscv_zvfh
        TILE_M = packn;
#else
        TILE_M = 2;
#endif
    }

    {
        TILE_M *= std::min(nT, get_physical_cpu_count());

        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __riscv_zvfh
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + packn - 1) / packn * packn);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif

        if (nT > 1)
        {
#if __riscv_zvfh
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + packn - 1) / packn * packn);
#else
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
        }

#if __riscv_zvfh
        TILE_M = std::max(packn, TILE_M);
#else
        TILE_M = std::max(2, TILE_M);
#endif
    }

    if (N > 0)
    {
        int tile_size;
        if (TILE_K >= K)
        {
            tile_size = (l2_cache_size_fp16 - TILE_M * TILE_K) / TILE_K;
        }
        else
        {
            tile_size = (l2_cache_size_fp16 - TILE_M * TILE_K) / (TILE_M + TILE_K);
        }

#if __riscv_zvfh
        TILE_N = std::max(16, tile_size / 16 * 16);
#else
        TILE_N = std::max(1, tile_size);
#endif

        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __riscv_zvfh
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 15) / 16 * 16);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif

#if __riscv_zvfh
        TILE_N = std::max(16, TILE_N);
#else
        TILE_N = std::max(1, TILE_N);
#endif
    }
}

static inline void conv3x3s1_winograd23_transform_kernel_tile_fp16sa_rvv(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    // const float ktm[4][3] = {
    //     {1.0f, 0.0f, 0.0f},
    //     {1.0f / 2, 1.0f / 2, 1.0f / 2},
    //     {1.0f / 2, -1.0f / 2, 1.0f / 2},
    //     {0.0f, 0.0f, 1.0f}
    // };

    __fp16* ptmp = A;

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

                ptmp[0] = (__fp16)z0;
                ptmp[1] = (__fp16)z1;
                ptmp[2] = (__fp16)z2;
                ptmp[3] = (__fp16)z3;
                ptmp += 4;
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_kernel_fp16sa_rvv(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 16;

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk_fp16sa_rvv(M, 0, K, B, TILE_M, TILE_N, TILE_K, opt.num_threads);

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

            conv3x3s1_winograd23_transform_kernel_tile_fp16sa_rvv(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            conv3x3s1_winograd_pack_A_tile_fp16sa_rvv(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd23_transform_input_tile_fp16sa_rvv(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    // const __fp16 itm[4][4] = {
    //     {1.0f,  0.0f, -1.0f,  0.0f},
    //     {0.0f,  1.0f,  1.00f, 0.0f},
    //     {0.0f, -1.0f,  1.00f, 0.0f},
    //     {0.0f, -1.0f,  0.00f, 1.0f}
    // };

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;

    const int w_tiles = (w - 1) / 2;

    int kk_start = 0;
#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
    const size_t N = bottom_blob.cstep * elempack;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
    {
        const int nn_kk = max_kk / packn;

        #pragma omp parallel for num_threads(nT)
        for (int kk_pack = 0; kk_pack < nn_kk; kk_pack++)
        {
            const int kk = kk_pack * packn;

            __fp16 tmp[4][4][packn];

            int jj = 0;
            for (; jj < max_jj; jj++)
            {
                int ti = (j + jj) / w_tiles;
                int tj = (j + jj) % w_tiles;

                const __fp16* r0123 = bottom_blob.channel((k + kk) / elempack).row<__fp16>(ti * 2) + (tj * 2) * elempack + (k + kk) % elempack;

                for (int m = 0; m < 4; m++)
                {
                    vfloat16m1_t _r0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);
                    vfloat16m1_t _r1 = _r0;
                    vfloat16m1_t _r2 = _r0;
                    vfloat16m1_t _r3 = _r0;

                    if (ti * 2 + m < h)
                    {
                        if (elempack == packn)
                        {
                            _r0 = __riscv_vle16_v_f16m1(r0123, vl);
                            if (tj * 2 + 1 < w) _r1 = __riscv_vle16_v_f16m1(r0123 + elempack, vl);
                            if (tj * 2 + 2 < w) _r2 = __riscv_vle16_v_f16m1(r0123 + elempack * 2, vl);
                            if (tj * 2 + 3 < w) _r3 = __riscv_vle16_v_f16m1(r0123 + elempack * 3, vl);
                        }
                        else // if (elempack == 1)
                        {
                            _r0 = __riscv_vlse16_v_f16m1(r0123, N * sizeof(__fp16), vl);
                            if (tj * 2 + 1 < w) _r1 = __riscv_vlse16_v_f16m1(r0123 + elempack, N * sizeof(__fp16), vl);
                            if (tj * 2 + 2 < w) _r2 = __riscv_vlse16_v_f16m1(r0123 + elempack * 2, N * sizeof(__fp16), vl);
                            if (tj * 2 + 3 < w) _r3 = __riscv_vlse16_v_f16m1(r0123 + elempack * 3, N * sizeof(__fp16), vl);
                        }
                    }

                    __riscv_vse16_v_f16m1(tmp[0][m], __riscv_vfsub_vv_f16m1(_r0, _r2, vl), vl);
                    __riscv_vse16_v_f16m1(tmp[1][m], __riscv_vfadd_vv_f16m1(_r1, _r2, vl), vl);
                    __riscv_vse16_v_f16m1(tmp[2][m], __riscv_vfsub_vv_f16m1(_r2, _r1, vl), vl);
                    __riscv_vse16_v_f16m1(tmp[3][m], __riscv_vfsub_vv_f16m1(_r3, _r1, vl), vl);

                    r0123 += w * elempack;
                }

                __fp16* p0 = (__fp16*)B + kk * max_jj * 16 + jj * packn;
                __fp16* p1 = p0 + max_jj * packn;
                __fp16* p2 = p0 + max_jj * packn * 2;
                __fp16* p3 = p0 + max_jj * packn * 3;

                for (int m = 0; m < 4; m++)
                {
                    vfloat16m1_t _r0 = __riscv_vle16_v_f16m1(tmp[m][0], vl);
                    vfloat16m1_t _r1 = __riscv_vle16_v_f16m1(tmp[m][1], vl);
                    vfloat16m1_t _r2 = __riscv_vle16_v_f16m1(tmp[m][2], vl);
                    vfloat16m1_t _r3 = __riscv_vle16_v_f16m1(tmp[m][3], vl);

                    __riscv_vse16_v_f16m1(p0, __riscv_vfsub_vv_f16m1(_r0, _r2, vl), vl);
                    __riscv_vse16_v_f16m1(p1, __riscv_vfadd_vv_f16m1(_r1, _r2, vl), vl);
                    __riscv_vse16_v_f16m1(p2, __riscv_vfsub_vv_f16m1(_r2, _r1, vl), vl);
                    __riscv_vse16_v_f16m1(p3, __riscv_vfsub_vv_f16m1(_r3, _r1, vl), vl);

                    p0 += max_jj * 4 * packn;
                    p1 += max_jj * 4 * packn;
                    p2 += max_jj * 4 * packn;
                    p3 += max_jj * 4 * packn;
                }
            }
        }

        kk_start = nn_kk * packn;
    }
#endif // __riscv_zvfh

    #pragma omp parallel for num_threads(nT)
    for (int kk = kk_start; kk < max_kk; kk++)
    {
        __fp16 tmp[4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0123 = bottom_blob.channel((k + kk) / elempack).row<__fp16>(ti * 2) + (tj * 2) * elempack + (k + kk) % elempack;

            for (int m = 0; m < 4; m++)
            {
                __fp16 r0 = (__fp16)0.f;
                __fp16 r1 = (__fp16)0.f;
                __fp16 r2 = (__fp16)0.f;
                __fp16 r3 = (__fp16)0.f;

                if (ti * 2 + m < h)
                {
                    // if (elempack == 1)
                    {
                        r0 = r0123[0];
                        if (tj * 2 + 1 < w) r1 = r0123[elempack];
                        if (tj * 2 + 2 < w) r2 = r0123[elempack * 2];
                        if (tj * 2 + 3 < w) r3 = r0123[elempack * 3];
                    }
                }

                tmp[0][m] = r0 - r2;
                tmp[1][m] = r1 + r2;
                tmp[2][m] = r2 - r1;
                tmp[3][m] = r3 - r1;

                r0123 += w * elempack;
            }

            __fp16* p0 = (__fp16*)B + kk * max_jj * 16 + jj;
            __fp16* p1 = p0 + max_jj;
            __fp16* p2 = p0 + max_jj * 2;
            __fp16* p3 = p0 + max_jj * 3;

            for (int m = 0; m < 4; m++)
            {
                __fp16 r0 = tmp[m][0];
                __fp16 r1 = tmp[m][1];
                __fp16 r2 = tmp[m][2];
                __fp16 r3 = tmp[m][3];

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

static inline void conv3x3s1_winograd23_transform_output_tile_fp16sa_rvv(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
{
    // const __fp16 otm[2][4] = {
    //     {1.0f,  1.0f,  1.0f,  0.0f},
    //     {0.0f,  1.0f, -1.0f,  1.0f}
    // };

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
#if __riscv_zvfh
    const size_t out_hstep = top_blob.cstep;
#endif

    const int w_tiles = (outw + 1) / 2;

    const __fp16* biasptr = bias;

    int ii = 0;
#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
    size_t vl;
    if (packn == 8)
        vl = __riscv_vsetvl_e16m1(8);
    else if (packn == 16)
        vl = __riscv_vsetvl_e16m1(16);
    else
        vl = __riscv_vsetvl_e16m1(packn);

    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        vfloat16m1_t _bias0 = biasptr ? __riscv_vle16_v_f16m1(biasptr + i + ii, vl) : __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);

        __fp16 tmp[2][4][packn];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 16 + jj * packn;
            const __fp16* r1 = r0 + max_jj * packn;
            const __fp16* r2 = r0 + max_jj * packn * 2;
            const __fp16* r3 = r0 + max_jj * packn * 3;

            for (int m = 0; m < 4; m++)
            {
                vfloat16m1_t _r0 = __riscv_vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r1 = __riscv_vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r2 = __riscv_vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r3 = __riscv_vle16_v_f16m1(r3, vl);

                vfloat16m1_t _tmp0 = __riscv_vfadd_vv_f16m1(__riscv_vfadd_vv_f16m1(_r0, _r1, vl), _r2, vl);
                vfloat16m1_t _tmp1 = __riscv_vfadd_vv_f16m1(__riscv_vfsub_vv_f16m1(_r1, _r2, vl), _r3, vl);
                __riscv_vse16_v_f16m1(tmp[0][m], _tmp0, vl);
                __riscv_vse16_v_f16m1(tmp[1][m], _tmp1, vl);

                r0 += max_jj * 4 * packn;
                r1 += max_jj * 4 * packn;
                r2 += max_jj * 4 * packn;
                r3 += max_jj * 4 * packn;
            }

            __fp16* outptr0 = top_blob.channel((i + ii) / out_elempack).row<__fp16>(ti * 2) + (tj * 2) * out_elempack + (i + ii) % out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                vfloat16m1_t _r0 = __riscv_vle16_v_f16m1(tmp[m][0], vl);
                vfloat16m1_t _r1 = __riscv_vle16_v_f16m1(tmp[m][1], vl);
                vfloat16m1_t _r2 = __riscv_vle16_v_f16m1(tmp[m][2], vl);
                vfloat16m1_t _r3 = __riscv_vle16_v_f16m1(tmp[m][3], vl);

                vfloat16m1_t _tmp0 = __riscv_vfadd_vv_f16m1(_bias0, __riscv_vfadd_vv_f16m1(__riscv_vfadd_vv_f16m1(_r0, _r1, vl), _r2, vl), vl);
                vfloat16m1_t _tmp1 = __riscv_vfadd_vv_f16m1(_bias0, __riscv_vfadd_vv_f16m1(__riscv_vfsub_vv_f16m1(_r1, _r2, vl), _r3, vl), vl);

                if (out_elempack == packn)
                {
                    __riscv_vse16_v_f16m1(outptr0, _tmp0, vl);
                    if (tj * 2 + 1 < outw)
                        __riscv_vse16_v_f16m1(outptr0 + out_elempack, _tmp1, vl);
                }
                else
                {
                    __riscv_vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _tmp0, vl);
                    if (tj * 2 + 1 < outw)
                        __riscv_vsse16_v_f16m1(outptr0 + out_elempack, out_hstep * sizeof(__fp16), _tmp1, vl);
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __riscv_zvfh

#if __riscv_zvfh
    const size_t vl4 = __riscv_vsetvl_e16m1(4);
    for (; ii + 3 < max_ii; ii += 4)
    {
        vfloat16m1_t _bias0 = biasptr ? __riscv_vle16_v_f16m1(biasptr + i + ii, vl4) : __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl4);

        __fp16 tmp[2][4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 16 + jj * 4;
            const __fp16* r1 = r0 + max_jj * 4;
            const __fp16* r2 = r0 + max_jj * 4 * 2;
            const __fp16* r3 = r0 + max_jj * 4 * 3;

            for (int m = 0; m < 4; m++)
            {
                vfloat16m1_t _r0 = __riscv_vle16_v_f16m1(r0, vl4);
                vfloat16m1_t _r1 = __riscv_vle16_v_f16m1(r1, vl4);
                vfloat16m1_t _r2 = __riscv_vle16_v_f16m1(r2, vl4);
                vfloat16m1_t _r3 = __riscv_vle16_v_f16m1(r3, vl4);

                vfloat16m1_t _tmp0 = __riscv_vfadd_vv_f16m1(__riscv_vfadd_vv_f16m1(_r0, _r1, vl4), _r2, vl4);
                vfloat16m1_t _tmp1 = __riscv_vfadd_vv_f16m1(__riscv_vfsub_vv_f16m1(_r1, _r2, vl4), _r3, vl4);
                __riscv_vse16_v_f16m1(tmp[0][m], _tmp0, vl4);
                __riscv_vse16_v_f16m1(tmp[1][m], _tmp1, vl4);

                r0 += max_jj * 4 * 4;
                r1 += max_jj * 4 * 4;
                r2 += max_jj * 4 * 4;
                r3 += max_jj * 4 * 4;
            }

            __fp16* outptr0 = top_blob.channel((i + ii) / out_elempack).row<__fp16>(ti * 2) + (tj * 2) * out_elempack + (i + ii) % out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                vfloat16m1_t _r0 = __riscv_vle16_v_f16m1(tmp[m][0], vl4);
                vfloat16m1_t _r1 = __riscv_vle16_v_f16m1(tmp[m][1], vl4);
                vfloat16m1_t _r2 = __riscv_vle16_v_f16m1(tmp[m][2], vl4);
                vfloat16m1_t _r3 = __riscv_vle16_v_f16m1(tmp[m][3], vl4);

                vfloat16m1_t _tmp0 = __riscv_vfadd_vv_f16m1(_bias0, __riscv_vfadd_vv_f16m1(__riscv_vfadd_vv_f16m1(_r0, _r1, vl4), _r2, vl4), vl4);
                vfloat16m1_t _tmp1 = __riscv_vfadd_vv_f16m1(_bias0, __riscv_vfadd_vv_f16m1(__riscv_vfsub_vv_f16m1(_r1, _r2, vl4), _r3, vl4), vl4);

                if (out_elempack == packn)
                {
                    __riscv_vse16_v_f16m1(outptr0, _tmp0, vl4);
                    if (tj * 2 + 1 < outw)
                        __riscv_vse16_v_f16m1(outptr0 + out_elempack, _tmp1, vl4);
                }
                else
                {
                    __riscv_vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _tmp0, vl4);
                    if (tj * 2 + 1 < outw)
                        __riscv_vsse16_v_f16m1(outptr0 + out_elempack, out_hstep * sizeof(__fp16), _tmp1, vl4);
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __riscv_zvfh

    for (; ii + 1 < max_ii; ii += 2)
    {
        __fp16 bias0 = biasptr ? biasptr[i + ii] : (__fp16)0.f;
        __fp16 bias1 = biasptr ? biasptr[i + ii + 1] : (__fp16)0.f;

        __fp16 tmp[2][4][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 16 + jj * 2;
            const __fp16* r1 = r0 + max_jj * 2;
            const __fp16* r2 = r0 + max_jj * 2 * 2;
            const __fp16* r3 = r0 + max_jj * 2 * 3;

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

            __fp16* outptr0 = top_blob.channel((i + ii) / out_elempack).row<__fp16>(ti * 2) + (tj * 2) * out_elempack + (i + ii) % out_elempack;
            __fp16* outptr1 = top_blob.channel((i + ii + 1) / out_elempack).row<__fp16>(ti * 2) + (tj * 2) * out_elempack + (i + ii + 1) % out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                __fp16 r00 = tmp[m][0][0];
                __fp16 r01 = tmp[m][0][1];
                __fp16 r10 = tmp[m][1][0];
                __fp16 r11 = tmp[m][1][1];
                __fp16 r20 = tmp[m][2][0];
                __fp16 r21 = tmp[m][2][1];
                __fp16 r30 = tmp[m][3][0];
                __fp16 r31 = tmp[m][3][1];

                __fp16 tmp00 = bias0 + r00 + r10 + r20;
                __fp16 tmp01 = bias1 + r01 + r11 + r21;
                __fp16 tmp10 = bias0 + r10 - r20 + r30;
                __fp16 tmp11 = bias1 + r11 - r21 + r31;

                // if (out_elempack == 1)
                {
                    outptr0[0] = tmp00;
                    outptr1[0] = tmp01;
                    if (tj * 2 + 1 < outw)
                    {
                        outptr0[out_elempack] = tmp10;
                        outptr1[out_elempack] = tmp11;
                    }
                }

                outptr0 += outw * out_elempack;
                outptr1 += outw * out_elempack;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        __fp16 bias0 = biasptr ? biasptr[i + ii] : (__fp16)0.f;

        __fp16 tmp[2][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 16 + jj;
            const __fp16* r1 = r0 + max_jj;
            const __fp16* r2 = r0 + max_jj * 2;
            const __fp16* r3 = r0 + max_jj * 3;

            for (int m = 0; m < 4; m++)
            {
                tmp[0][m] = r0[0] + r1[0] + r2[0];
                tmp[1][m] = r1[0] - r2[0] + r3[0];

                r0 += max_jj * 4;
                r1 += max_jj * 4;
                r2 += max_jj * 4;
                r3 += max_jj * 4;
            }

            __fp16* outptr0 = top_blob.channel((i + ii) / out_elempack).row<__fp16>(ti * 2) + (tj * 2) * out_elempack + (i + ii) % out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                __fp16 r0 = tmp[m][0];
                __fp16 r1 = tmp[m][1];
                __fp16 r2 = tmp[m][2];
                __fp16 r3 = tmp[m][3];

                __fp16 tmp0 = bias0 + r0 + r1 + r2;
                __fp16 tmp1 = bias0 + r1 - r2 + r3;

                // if (out_elempack == 1)
                {
                    outptr0[0] = tmp0;
                    if (tj * 2 + 1 < outw) outptr0[out_elempack] = tmp1;
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
}

static int conv3x3s1_winograd23_fp16sa_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, const Option& opt)
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
    conv3x3s1_winograd_get_optimal_tile_mnk_fp16sa_rvv(M, N, K, B, TILE_M, TILE_N, TILE_K, nT);

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
            conv3x3s1_winograd23_transform_input_tile_fp16sa_rvv(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile_fp16sa_rvv(B_tile, BT_tile, B, max_jj, max_kk, nT);
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
            conv3x3s1_winograd23_transform_input_tile_fp16sa_rvv(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile_fp16sa_rvv(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 2u, opt.workspace_allocator);
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

                conv3x3s1_winograd_gemm_transB_packed_tile_fp16sa_rvv(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk);
            }

            // transform output
            conv3x3s1_winograd23_transform_output_tile_fp16sa_rvv(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

static inline void conv3x3s1_winograd43_transform_kernel_tile_fp16sa_rvv(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    __fp16* ptmp = A;

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

                ptmp[0] = (__fp16)z0;
                ptmp[1] = (__fp16)z1;
                ptmp[2] = (__fp16)z2;
                ptmp[3] = (__fp16)z3;
                ptmp[4] = (__fp16)z4;
                ptmp[5] = (__fp16)z5;
                ptmp += 6;
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_kernel_fp16sa_rvv(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 36;

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk_fp16sa_rvv(M, 0, K, B, TILE_M, TILE_N, TILE_K, opt.num_threads);

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

            conv3x3s1_winograd43_transform_kernel_tile_fp16sa_rvv(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            conv3x3s1_winograd_pack_A_tile_fp16sa_rvv(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd43_transform_input_tile_fp16sa_rvv(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    const __fp16 sq2 = (__fp16)1.41421356237f;
    const __fp16 sq2_d2 = (__fp16)(1.41421356237f / 2);

    // const __fp16 itm[6][6] = {
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

    const int w_tiles = (w + 1) / 4;

    int kk_start = 0;
#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
    const size_t N = bottom_blob.cstep * elempack;
    const size_t vl = __riscv_vsetvl_e16m1(packn);

    {
        const int nn_kk = max_kk / packn;

        #pragma omp parallel for num_threads(nT)
        for (int kk_pack = 0; kk_pack < nn_kk; kk_pack++)
        {
            const int kk = kk_pack * packn;

            __fp16 tmp[6][6][packn];

            int jj = 0;
            for (; jj < max_jj; jj++)
            {
                int ti = (j + jj) / w_tiles;
                int tj = (j + jj) % w_tiles;

                const __fp16* r0123 = bottom_blob.channel((k + kk) / elempack).row<__fp16>(ti * 4) + (tj * 4) * elempack + (k + kk) % elempack;

                for (int m = 0; m < 6; m++)
                {
                    vfloat16m1_t _r0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);
                    vfloat16m1_t _r1 = _r0;
                    vfloat16m1_t _r2 = _r0;
                    vfloat16m1_t _r3 = _r0;
                    vfloat16m1_t _r4 = _r0;
                    vfloat16m1_t _r5 = _r0;

                    if (ti * 4 + m < h)
                    {
                        if (elempack == packn)
                        {
                            _r0 = __riscv_vle16_v_f16m1(r0123, vl);
                            if (tj * 4 + 1 < w) _r1 = __riscv_vle16_v_f16m1(r0123 + elempack, vl);
                            if (tj * 4 + 2 < w) _r2 = __riscv_vle16_v_f16m1(r0123 + elempack * 2, vl);
                            if (tj * 4 + 3 < w) _r3 = __riscv_vle16_v_f16m1(r0123 + elempack * 3, vl);
                            if (tj * 4 + 4 < w) _r4 = __riscv_vle16_v_f16m1(r0123 + elempack * 4, vl);
                            if (tj * 4 + 5 < w) _r5 = __riscv_vle16_v_f16m1(r0123 + elempack * 5, vl);
                        }
                        else // if (elempack == 1)
                        {
                            _r0 = __riscv_vlse16_v_f16m1(r0123, N * sizeof(__fp16), vl);
                            if (tj * 4 + 1 < w) _r1 = __riscv_vlse16_v_f16m1(r0123 + elempack, N * sizeof(__fp16), vl);
                            if (tj * 4 + 2 < w) _r2 = __riscv_vlse16_v_f16m1(r0123 + elempack * 2, N * sizeof(__fp16), vl);
                            if (tj * 4 + 3 < w) _r3 = __riscv_vlse16_v_f16m1(r0123 + elempack * 3, N * sizeof(__fp16), vl);
                            if (tj * 4 + 4 < w) _r4 = __riscv_vlse16_v_f16m1(r0123 + elempack * 4, N * sizeof(__fp16), vl);
                            if (tj * 4 + 5 < w) _r5 = __riscv_vlse16_v_f16m1(r0123 + elempack * 5, N * sizeof(__fp16), vl);
                        }
                    }

                    vfloat16m1_t _tmp12a = __riscv_vfmacc_vf_f16m1(__riscv_vfmul_vf_f16m1(_r1, sq2, vl), -sq2_d2, _r3, vl);
                    vfloat16m1_t _tmp12b = __riscv_vfmacc_vf_f16m1(_r4, (__fp16)-2.f, _r2, vl);
                    vfloat16m1_t _tmp34a = __riscv_vfmacc_vf_f16m1(__riscv_vfmul_vf_f16m1(_r3, sq2, vl), -sq2_d2, _r1, vl);
                    vfloat16m1_t _tmp34b = __riscv_vfmacc_vf_f16m1(_r4, (__fp16)-0.5f, _r2, vl);

                    __riscv_vse16_v_f16m1(tmp[0][m], __riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(_r0, _r4, vl), (__fp16)-2.5f, _r2, vl), vl);
                    __riscv_vse16_v_f16m1(tmp[1][m], __riscv_vfsub_vv_f16m1(_tmp12b, _tmp12a, vl), vl);
                    __riscv_vse16_v_f16m1(tmp[2][m], __riscv_vfadd_vv_f16m1(_tmp12b, _tmp12a, vl), vl);
                    __riscv_vse16_v_f16m1(tmp[3][m], __riscv_vfadd_vv_f16m1(_tmp34b, _tmp34a, vl), vl);
                    __riscv_vse16_v_f16m1(tmp[4][m], __riscv_vfsub_vv_f16m1(_tmp34b, _tmp34a, vl), vl);
                    __riscv_vse16_v_f16m1(tmp[5][m], __riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(_r1, _r5, vl), (__fp16)-2.5f, _r3, vl), vl);

                    r0123 += w * elempack;
                }

                __fp16* p0 = (__fp16*)B + kk * max_jj * 36 + jj * packn;
                __fp16* p1 = p0 + max_jj * packn;
                __fp16* p2 = p0 + max_jj * packn * 2;
                __fp16* p3 = p0 + max_jj * packn * 3;
                __fp16* p4 = p0 + max_jj * packn * 4;
                __fp16* p5 = p0 + max_jj * packn * 5;

                for (int m = 0; m < 6; m++)
                {
                    vfloat16m1_t _r0 = __riscv_vle16_v_f16m1(tmp[m][0], vl);
                    vfloat16m1_t _r1 = __riscv_vle16_v_f16m1(tmp[m][1], vl);
                    vfloat16m1_t _r2 = __riscv_vle16_v_f16m1(tmp[m][2], vl);
                    vfloat16m1_t _r3 = __riscv_vle16_v_f16m1(tmp[m][3], vl);
                    vfloat16m1_t _r4 = __riscv_vle16_v_f16m1(tmp[m][4], vl);
                    vfloat16m1_t _r5 = __riscv_vle16_v_f16m1(tmp[m][5], vl);

                    vfloat16m1_t _tmp12a = __riscv_vfmacc_vf_f16m1(__riscv_vfmul_vf_f16m1(_r1, sq2, vl), -sq2_d2, _r3, vl);
                    vfloat16m1_t _tmp12b = __riscv_vfmacc_vf_f16m1(_r4, (__fp16)-2.f, _r2, vl);
                    vfloat16m1_t _tmp34a = __riscv_vfmacc_vf_f16m1(__riscv_vfmul_vf_f16m1(_r3, sq2, vl), -sq2_d2, _r1, vl);
                    vfloat16m1_t _tmp34b = __riscv_vfmacc_vf_f16m1(_r4, (__fp16)-0.5f, _r2, vl);

                    __riscv_vse16_v_f16m1(p0, __riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(_r0, _r4, vl), (__fp16)-2.5f, _r2, vl), vl);
                    __riscv_vse16_v_f16m1(p1, __riscv_vfsub_vv_f16m1(_tmp12b, _tmp12a, vl), vl);
                    __riscv_vse16_v_f16m1(p2, __riscv_vfadd_vv_f16m1(_tmp12b, _tmp12a, vl), vl);
                    __riscv_vse16_v_f16m1(p3, __riscv_vfadd_vv_f16m1(_tmp34b, _tmp34a, vl), vl);
                    __riscv_vse16_v_f16m1(p4, __riscv_vfsub_vv_f16m1(_tmp34b, _tmp34a, vl), vl);
                    __riscv_vse16_v_f16m1(p5, __riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(_r1, _r5, vl), (__fp16)-2.5f, _r3, vl), vl);

                    p0 += max_jj * 6 * packn;
                    p1 += max_jj * 6 * packn;
                    p2 += max_jj * 6 * packn;
                    p3 += max_jj * 6 * packn;
                    p4 += max_jj * 6 * packn;
                    p5 += max_jj * 6 * packn;
                }
            }
        }

        kk_start = nn_kk * packn;
    }
#endif // __riscv_zvfh

    #pragma omp parallel for num_threads(nT)
    for (int kk = kk_start; kk < max_kk; kk++)
    {
        __fp16 tmp[6][6];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0123 = bottom_blob.channel((k + kk) / elempack).row<__fp16>(ti * 4) + (tj * 4) * elempack + (k + kk) % elempack;

            for (int m = 0; m < 6; m++)
            {
                __fp16 r0 = (__fp16)0.f;
                __fp16 r1 = (__fp16)0.f;
                __fp16 r2 = (__fp16)0.f;
                __fp16 r3 = (__fp16)0.f;
                __fp16 r4 = (__fp16)0.f;
                __fp16 r5 = (__fp16)0.f;

                if (ti * 4 + m < h)
                {
                    // if (elempack == 1)
                    {
                        r0 = r0123[0];
                        if (tj * 4 + 1 < w) r1 = r0123[elempack];
                        if (tj * 4 + 2 < w) r2 = r0123[elempack * 2];
                        if (tj * 4 + 3 < w) r3 = r0123[elempack * 3];
                        if (tj * 4 + 4 < w) r4 = r0123[elempack * 4];
                        if (tj * 4 + 5 < w) r5 = r0123[elempack * 5];
                    }
                }

                __fp16 tmp12a = sq2 * r1 - sq2_d2 * r3;
                __fp16 tmp12b = r4 - 2 * r2;
                __fp16 tmp34a = sq2 * r3 - sq2_d2 * r1;
                __fp16 tmp34b = r4 - (__fp16)0.5f * r2;

                tmp[0][m] = r0 + r4 - (__fp16)2.5f * r2;
                tmp[1][m] = tmp12b - tmp12a;
                tmp[2][m] = tmp12b + tmp12a;
                tmp[3][m] = tmp34b + tmp34a;
                tmp[4][m] = tmp34b - tmp34a;
                tmp[5][m] = r1 + r5 - (__fp16)2.5f * r3;

                r0123 += w * elempack;
            }

            __fp16* p0 = (__fp16*)B + kk * max_jj * 36 + jj;
            __fp16* p1 = p0 + max_jj;
            __fp16* p2 = p0 + max_jj * 2;
            __fp16* p3 = p0 + max_jj * 3;
            __fp16* p4 = p0 + max_jj * 4;
            __fp16* p5 = p0 + max_jj * 5;

            for (int m = 0; m < 6; m++)
            {
                __fp16 r0 = tmp[m][0];
                __fp16 r1 = tmp[m][1];
                __fp16 r2 = tmp[m][2];
                __fp16 r3 = tmp[m][3];
                __fp16 r4 = tmp[m][4];
                __fp16 r5 = tmp[m][5];

                __fp16 tmp12a = sq2 * r1 - sq2_d2 * r3;
                __fp16 tmp12b = r4 - 2 * r2;
                __fp16 tmp34a = sq2 * r3 - sq2_d2 * r1;
                __fp16 tmp34b = r4 - (__fp16)0.5f * r2;

                p0[0] = r0 + r4 - (__fp16)2.5f * r2;
                p1[0] = tmp12b - tmp12a;
                p2[0] = tmp12b + tmp12a;
                p3[0] = tmp34b + tmp34a;
                p4[0] = tmp34b - tmp34a;
                p5[0] = r1 + r5 - (__fp16)2.5f * r3;

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

static inline void conv3x3s1_winograd43_transform_output_tile_fp16sa_rvv(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
{
    const __fp16 sq2 = (__fp16)1.41421356237f;
    const __fp16 sq2_m2 = (__fp16)(1.41421356237f * 2);
    const __fp16 sq2_d2 = (__fp16)(1.41421356237f / 2);
    const __fp16 sq2_d4 = (__fp16)(1.41421356237f / 4);

    // const __fp16 otm[4][6] = {
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
#if __riscv_zvfh
    const size_t out_hstep = top_blob.cstep;
#endif

    const int w_tiles = (outw + 3) / 4;

    const __fp16* biasptr = bias;

    int ii = 0;
#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
    size_t vl;
    if (packn == 8)
        vl = __riscv_vsetvl_e16m1(8);
    else if (packn == 16)
        vl = __riscv_vsetvl_e16m1(16);
    else
        vl = __riscv_vsetvl_e16m1(packn);

    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        vfloat16m1_t _bias0 = biasptr ? __riscv_vle16_v_f16m1(biasptr + i + ii, vl) : __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);

        __fp16 tmp[4][6][packn];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 36 + jj * packn;
            const __fp16* r1 = r0 + max_jj * packn;
            const __fp16* r2 = r0 + max_jj * packn * 2;
            const __fp16* r3 = r0 + max_jj * packn * 3;
            const __fp16* r4 = r0 + max_jj * packn * 4;
            const __fp16* r5 = r0 + max_jj * packn * 5;

            for (int m = 0; m < 6; m++)
            {
                vfloat16m1_t _r0 = __riscv_vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r1 = __riscv_vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r2 = __riscv_vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r3 = __riscv_vle16_v_f16m1(r3, vl);
                vfloat16m1_t _r4 = __riscv_vle16_v_f16m1(r4, vl);
                vfloat16m1_t _r5 = __riscv_vle16_v_f16m1(r5, vl);

                vfloat16m1_t _tmp02a = __riscv_vfadd_vv_f16m1(_r1, _r2, vl);
                vfloat16m1_t _tmp02b = __riscv_vfadd_vv_f16m1(_r3, _r4, vl);
                vfloat16m1_t _tmp13a = __riscv_vfsub_vv_f16m1(_r1, _r2, vl);
                vfloat16m1_t _tmp13b = __riscv_vfsub_vv_f16m1(_r3, _r4, vl);

                vfloat16m1_t _tmp0 = __riscv_vfadd_vv_f16m1(__riscv_vfadd_vv_f16m1(_r0, _tmp02a, vl), _tmp02b, vl);
                vfloat16m1_t _tmp1 = __riscv_vfmacc_vf_f16m1(__riscv_vfmul_vf_f16m1(_tmp13a, sq2_d2, vl), sq2, _tmp13b, vl);
                vfloat16m1_t _tmp2 = __riscv_vfmacc_vf_f16m1(__riscv_vfmul_vf_f16m1(_tmp02a, (__fp16)0.5f, vl), (__fp16)2.f, _tmp02b, vl);
                vfloat16m1_t _tmp3 = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(_r5, sq2_d4, _tmp13a, vl), sq2_m2, _tmp13b, vl);
                __riscv_vse16_v_f16m1(tmp[0][m], _tmp0, vl);
                __riscv_vse16_v_f16m1(tmp[1][m], _tmp1, vl);
                __riscv_vse16_v_f16m1(tmp[2][m], _tmp2, vl);
                __riscv_vse16_v_f16m1(tmp[3][m], _tmp3, vl);

                r0 += max_jj * 6 * packn;
                r1 += max_jj * 6 * packn;
                r2 += max_jj * 6 * packn;
                r3 += max_jj * 6 * packn;
                r4 += max_jj * 6 * packn;
                r5 += max_jj * 6 * packn;
            }

            __fp16* outptr0 = top_blob.channel((i + ii) / out_elempack).row<__fp16>(ti * 4) + (tj * 4) * out_elempack + (i + ii) % out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                vfloat16m1_t _r0 = __riscv_vle16_v_f16m1(tmp[m][0], vl);
                vfloat16m1_t _r1 = __riscv_vle16_v_f16m1(tmp[m][1], vl);
                vfloat16m1_t _r2 = __riscv_vle16_v_f16m1(tmp[m][2], vl);
                vfloat16m1_t _r3 = __riscv_vle16_v_f16m1(tmp[m][3], vl);
                vfloat16m1_t _r4 = __riscv_vle16_v_f16m1(tmp[m][4], vl);
                vfloat16m1_t _r5 = __riscv_vle16_v_f16m1(tmp[m][5], vl);

                vfloat16m1_t _tmp02a = __riscv_vfadd_vv_f16m1(_r1, _r2, vl);
                vfloat16m1_t _tmp02b = __riscv_vfadd_vv_f16m1(_r3, _r4, vl);
                vfloat16m1_t _tmp13a = __riscv_vfsub_vv_f16m1(_r1, _r2, vl);
                vfloat16m1_t _tmp13b = __riscv_vfsub_vv_f16m1(_r3, _r4, vl);

                vfloat16m1_t _tmp0 = __riscv_vfadd_vv_f16m1(_bias0, __riscv_vfadd_vv_f16m1(__riscv_vfadd_vv_f16m1(_r0, _tmp02a, vl), _tmp02b, vl), vl);
                vfloat16m1_t _tmp1 = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(_bias0, sq2_d2, _tmp13a, vl), sq2, _tmp13b, vl);
                vfloat16m1_t _tmp2 = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(_bias0, (__fp16)0.5f, _tmp02a, vl), (__fp16)2.f, _tmp02b, vl);
                vfloat16m1_t _tmp3 = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(_bias0, _r5, vl), sq2_d4, _tmp13a, vl), sq2_m2, _tmp13b, vl);

                if (out_elempack == packn)
                {
                    __riscv_vse16_v_f16m1(outptr0, _tmp0, vl);
                    if (tj * 4 + 1 < outw) __riscv_vse16_v_f16m1(outptr0 + out_elempack, _tmp1, vl);
                    if (tj * 4 + 2 < outw) __riscv_vse16_v_f16m1(outptr0 + out_elempack * 2, _tmp2, vl);
                    if (tj * 4 + 3 < outw) __riscv_vse16_v_f16m1(outptr0 + out_elempack * 3, _tmp3, vl);
                }
                else
                {
                    __riscv_vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _tmp0, vl);
                    if (tj * 4 + 1 < outw) __riscv_vsse16_v_f16m1(outptr0 + out_elempack, out_hstep * sizeof(__fp16), _tmp1, vl);
                    if (tj * 4 + 2 < outw) __riscv_vsse16_v_f16m1(outptr0 + out_elempack * 2, out_hstep * sizeof(__fp16), _tmp2, vl);
                    if (tj * 4 + 3 < outw) __riscv_vsse16_v_f16m1(outptr0 + out_elempack * 3, out_hstep * sizeof(__fp16), _tmp3, vl);
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __riscv_zvfh

#if __riscv_zvfh
    const size_t vl4 = __riscv_vsetvl_e16m1(4);
    for (; ii + 3 < max_ii; ii += 4)
    {
        vfloat16m1_t _bias0 = biasptr ? __riscv_vle16_v_f16m1(biasptr + i + ii, vl4) : __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl4);

        __fp16 tmp[4][6][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 36 + jj * 4;
            const __fp16* r1 = r0 + max_jj * 4;
            const __fp16* r2 = r0 + max_jj * 4 * 2;
            const __fp16* r3 = r0 + max_jj * 4 * 3;
            const __fp16* r4 = r0 + max_jj * 4 * 4;
            const __fp16* r5 = r0 + max_jj * 4 * 5;

            for (int m = 0; m < 6; m++)
            {
                vfloat16m1_t _r0 = __riscv_vle16_v_f16m1(r0, vl4);
                vfloat16m1_t _r1 = __riscv_vle16_v_f16m1(r1, vl4);
                vfloat16m1_t _r2 = __riscv_vle16_v_f16m1(r2, vl4);
                vfloat16m1_t _r3 = __riscv_vle16_v_f16m1(r3, vl4);
                vfloat16m1_t _r4 = __riscv_vle16_v_f16m1(r4, vl4);
                vfloat16m1_t _r5 = __riscv_vle16_v_f16m1(r5, vl4);

                vfloat16m1_t _tmp02a = __riscv_vfadd_vv_f16m1(_r1, _r2, vl4);
                vfloat16m1_t _tmp02b = __riscv_vfadd_vv_f16m1(_r3, _r4, vl4);
                vfloat16m1_t _tmp13a = __riscv_vfsub_vv_f16m1(_r1, _r2, vl4);
                vfloat16m1_t _tmp13b = __riscv_vfsub_vv_f16m1(_r3, _r4, vl4);

                vfloat16m1_t _tmp0 = __riscv_vfadd_vv_f16m1(__riscv_vfadd_vv_f16m1(_r0, _tmp02a, vl4), _tmp02b, vl4);
                vfloat16m1_t _tmp1 = __riscv_vfmacc_vf_f16m1(__riscv_vfmul_vf_f16m1(_tmp13a, sq2_d2, vl4), sq2, _tmp13b, vl4);
                vfloat16m1_t _tmp2 = __riscv_vfmacc_vf_f16m1(__riscv_vfmul_vf_f16m1(_tmp02a, (__fp16)0.5f, vl4), (__fp16)2.f, _tmp02b, vl4);
                vfloat16m1_t _tmp3 = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(_r5, sq2_d4, _tmp13a, vl4), sq2_m2, _tmp13b, vl4);
                __riscv_vse16_v_f16m1(tmp[0][m], _tmp0, vl4);
                __riscv_vse16_v_f16m1(tmp[1][m], _tmp1, vl4);
                __riscv_vse16_v_f16m1(tmp[2][m], _tmp2, vl4);
                __riscv_vse16_v_f16m1(tmp[3][m], _tmp3, vl4);

                r0 += max_jj * 6 * 4;
                r1 += max_jj * 6 * 4;
                r2 += max_jj * 6 * 4;
                r3 += max_jj * 6 * 4;
                r4 += max_jj * 6 * 4;
                r5 += max_jj * 6 * 4;
            }

            __fp16* outptr0 = top_blob.channel((i + ii) / out_elempack).row<__fp16>(ti * 4) + (tj * 4) * out_elempack + (i + ii) % out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                vfloat16m1_t _r0 = __riscv_vle16_v_f16m1(tmp[m][0], vl4);
                vfloat16m1_t _r1 = __riscv_vle16_v_f16m1(tmp[m][1], vl4);
                vfloat16m1_t _r2 = __riscv_vle16_v_f16m1(tmp[m][2], vl4);
                vfloat16m1_t _r3 = __riscv_vle16_v_f16m1(tmp[m][3], vl4);
                vfloat16m1_t _r4 = __riscv_vle16_v_f16m1(tmp[m][4], vl4);
                vfloat16m1_t _r5 = __riscv_vle16_v_f16m1(tmp[m][5], vl4);

                vfloat16m1_t _tmp02a = __riscv_vfadd_vv_f16m1(_r1, _r2, vl4);
                vfloat16m1_t _tmp02b = __riscv_vfadd_vv_f16m1(_r3, _r4, vl4);
                vfloat16m1_t _tmp13a = __riscv_vfsub_vv_f16m1(_r1, _r2, vl4);
                vfloat16m1_t _tmp13b = __riscv_vfsub_vv_f16m1(_r3, _r4, vl4);

                vfloat16m1_t _tmp0 = __riscv_vfadd_vv_f16m1(_bias0, __riscv_vfadd_vv_f16m1(__riscv_vfadd_vv_f16m1(_r0, _tmp02a, vl4), _tmp02b, vl4), vl4);
                vfloat16m1_t _tmp1 = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(_bias0, sq2_d2, _tmp13a, vl4), sq2, _tmp13b, vl4);
                vfloat16m1_t _tmp2 = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(_bias0, (__fp16)0.5f, _tmp02a, vl4), (__fp16)2.f, _tmp02b, vl4);
                vfloat16m1_t _tmp3 = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(_bias0, _r5, vl4), sq2_d4, _tmp13a, vl4), sq2_m2, _tmp13b, vl4);

                if (out_elempack == packn)
                {
                    __riscv_vse16_v_f16m1(outptr0, _tmp0, vl4);
                    if (tj * 4 + 1 < outw) __riscv_vse16_v_f16m1(outptr0 + out_elempack, _tmp1, vl4);
                    if (tj * 4 + 2 < outw) __riscv_vse16_v_f16m1(outptr0 + out_elempack * 2, _tmp2, vl4);
                    if (tj * 4 + 3 < outw) __riscv_vse16_v_f16m1(outptr0 + out_elempack * 3, _tmp3, vl4);
                }
                else
                {
                    __riscv_vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _tmp0, vl4);
                    if (tj * 4 + 1 < outw) __riscv_vsse16_v_f16m1(outptr0 + out_elempack, out_hstep * sizeof(__fp16), _tmp1, vl4);
                    if (tj * 4 + 2 < outw) __riscv_vsse16_v_f16m1(outptr0 + out_elempack * 2, out_hstep * sizeof(__fp16), _tmp2, vl4);
                    if (tj * 4 + 3 < outw) __riscv_vsse16_v_f16m1(outptr0 + out_elempack * 3, out_hstep * sizeof(__fp16), _tmp3, vl4);
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __riscv_zvfh

    for (; ii + 1 < max_ii; ii += 2)
    {
        __fp16 bias0 = biasptr ? biasptr[i + ii] : (__fp16)0.f;
        __fp16 bias1 = biasptr ? biasptr[i + ii + 1] : (__fp16)0.f;

        __fp16 tmp[4][6][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 36 + jj * 2;
            const __fp16* r1 = r0 + max_jj * 2;
            const __fp16* r2 = r0 + max_jj * 2 * 2;
            const __fp16* r3 = r0 + max_jj * 2 * 3;
            const __fp16* r4 = r0 + max_jj * 2 * 4;
            const __fp16* r5 = r0 + max_jj * 2 * 5;

            for (int m = 0; m < 6; m++)
            {
                __fp16 tmp02a0 = r1[0] + r2[0];
                __fp16 tmp02a1 = r1[1] + r2[1];
                __fp16 tmp02b0 = r3[0] + r4[0];
                __fp16 tmp02b1 = r3[1] + r4[1];
                __fp16 tmp13a0 = r1[0] - r2[0];
                __fp16 tmp13a1 = r1[1] - r2[1];
                __fp16 tmp13b0 = r3[0] - r4[0];
                __fp16 tmp13b1 = r3[1] - r4[1];

                tmp[0][m][0] = r0[0] + tmp02a0 + tmp02b0;
                tmp[0][m][1] = r0[1] + tmp02a1 + tmp02b1;
                tmp[1][m][0] = tmp13a0 * sq2_d2 + tmp13b0 * sq2;
                tmp[1][m][1] = tmp13a1 * sq2_d2 + tmp13b1 * sq2;
                tmp[2][m][0] = tmp02a0 * (__fp16)0.5f + tmp02b0 * 2;
                tmp[2][m][1] = tmp02a1 * (__fp16)0.5f + tmp02b1 * 2;
                tmp[3][m][0] = r5[0] + tmp13a0 * sq2_d4 + tmp13b0 * sq2_m2;
                tmp[3][m][1] = r5[1] + tmp13a1 * sq2_d4 + tmp13b1 * sq2_m2;

                r0 += max_jj * 6 * 2;
                r1 += max_jj * 6 * 2;
                r2 += max_jj * 6 * 2;
                r3 += max_jj * 6 * 2;
                r4 += max_jj * 6 * 2;
                r5 += max_jj * 6 * 2;
            }

            __fp16* outptr0 = top_blob.channel((i + ii) / out_elempack).row<__fp16>(ti * 4) + (tj * 4) * out_elempack + (i + ii) % out_elempack;
            __fp16* outptr1 = top_blob.channel((i + ii + 1) / out_elempack).row<__fp16>(ti * 4) + (tj * 4) * out_elempack + (i + ii + 1) % out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                __fp16 r00 = tmp[m][0][0];
                __fp16 r01 = tmp[m][0][1];
                __fp16 r10 = tmp[m][1][0];
                __fp16 r11 = tmp[m][1][1];
                __fp16 r20 = tmp[m][2][0];
                __fp16 r21 = tmp[m][2][1];
                __fp16 r30 = tmp[m][3][0];
                __fp16 r31 = tmp[m][3][1];
                __fp16 r40 = tmp[m][4][0];
                __fp16 r41 = tmp[m][4][1];
                __fp16 r50 = tmp[m][5][0];
                __fp16 r51 = tmp[m][5][1];

                __fp16 tmp02a0 = r10 + r20;
                __fp16 tmp02a1 = r11 + r21;
                __fp16 tmp02b0 = r30 + r40;
                __fp16 tmp02b1 = r31 + r41;
                __fp16 tmp13a0 = r10 - r20;
                __fp16 tmp13a1 = r11 - r21;
                __fp16 tmp13b0 = r30 - r40;
                __fp16 tmp13b1 = r31 - r41;

                __fp16 tmp00 = bias0 + r00 + tmp02a0 + tmp02b0;
                __fp16 tmp01 = bias1 + r01 + tmp02a1 + tmp02b1;
                __fp16 tmp10 = bias0 + tmp13a0 * sq2_d2 + tmp13b0 * sq2;
                __fp16 tmp11 = bias1 + tmp13a1 * sq2_d2 + tmp13b1 * sq2;
                __fp16 tmp20 = bias0 + tmp02a0 * (__fp16)0.5f + tmp02b0 * 2;
                __fp16 tmp21 = bias1 + tmp02a1 * (__fp16)0.5f + tmp02b1 * 2;
                __fp16 tmp30 = bias0 + r50 + tmp13a0 * sq2_d4 + tmp13b0 * sq2_m2;
                __fp16 tmp31 = bias1 + r51 + tmp13a1 * sq2_d4 + tmp13b1 * sq2_m2;

                // if (out_elempack == 1)
                {
                    outptr0[0] = tmp00;
                    outptr1[0] = tmp01;
                    if (tj * 4 + 1 < outw)
                    {
                        outptr0[out_elempack] = tmp10;
                        outptr1[out_elempack] = tmp11;
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        outptr0[out_elempack * 2] = tmp20;
                        outptr1[out_elempack * 2] = tmp21;
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        outptr0[out_elempack * 3] = tmp30;
                        outptr1[out_elempack * 3] = tmp31;
                    }
                }

                outptr0 += outw * out_elempack;
                outptr1 += outw * out_elempack;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        __fp16 bias0 = biasptr ? biasptr[i + ii] : (__fp16)0.f;

        __fp16 tmp[4][6];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 36 + jj;
            const __fp16* r1 = r0 + max_jj;
            const __fp16* r2 = r0 + max_jj * 2;
            const __fp16* r3 = r0 + max_jj * 3;
            const __fp16* r4 = r0 + max_jj * 4;
            const __fp16* r5 = r0 + max_jj * 5;

            for (int m = 0; m < 6; m++)
            {
                __fp16 tmp02a = r1[0] + r2[0];
                __fp16 tmp02b = r3[0] + r4[0];
                __fp16 tmp13a = r1[0] - r2[0];
                __fp16 tmp13b = r3[0] - r4[0];

                tmp[0][m] = r0[0] + tmp02a + tmp02b;
                tmp[1][m] = tmp13a * sq2_d2 + tmp13b * sq2;
                tmp[2][m] = tmp02a * (__fp16)0.5f + tmp02b * 2;
                tmp[3][m] = r5[0] + tmp13a * sq2_d4 + tmp13b * sq2_m2;

                r0 += max_jj * 6;
                r1 += max_jj * 6;
                r2 += max_jj * 6;
                r3 += max_jj * 6;
                r4 += max_jj * 6;
                r5 += max_jj * 6;
            }

            __fp16* outptr0 = top_blob.channel((i + ii) / out_elempack).row<__fp16>(ti * 4) + (tj * 4) * out_elempack + (i + ii) % out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                __fp16 r0 = tmp[m][0];
                __fp16 r1 = tmp[m][1];
                __fp16 r2 = tmp[m][2];
                __fp16 r3 = tmp[m][3];
                __fp16 r4 = tmp[m][4];
                __fp16 r5 = tmp[m][5];

                __fp16 tmp02a = r1 + r2;
                __fp16 tmp02b = r3 + r4;
                __fp16 tmp13a = r1 - r2;
                __fp16 tmp13b = r3 - r4;

                __fp16 tmp0 = bias0 + r0 + tmp02a + tmp02b;
                __fp16 tmp1 = bias0 + tmp13a * sq2_d2 + tmp13b * sq2;
                __fp16 tmp2 = bias0 + tmp02a * (__fp16)0.5f + tmp02b * 2;
                __fp16 tmp3 = bias0 + r5 + tmp13a * sq2_d4 + tmp13b * sq2_m2;

                // if (out_elempack == 1)
                {
                    outptr0[0] = tmp0;
                    if (tj * 4 + 1 < outw) outptr0[out_elempack] = tmp1;
                    if (tj * 4 + 2 < outw) outptr0[out_elempack * 2] = tmp2;
                    if (tj * 4 + 3 < outw) outptr0[out_elempack * 3] = tmp3;
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
}

static int conv3x3s1_winograd43_fp16sa_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, const Option& opt)
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
    conv3x3s1_winograd_get_optimal_tile_mnk_fp16sa_rvv(M, N, K, B, TILE_M, TILE_N, TILE_K, nT);

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
            conv3x3s1_winograd43_transform_input_tile_fp16sa_rvv(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile_fp16sa_rvv(B_tile, BT_tile, B, max_jj, max_kk, nT);
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
            conv3x3s1_winograd43_transform_input_tile_fp16sa_rvv(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile_fp16sa_rvv(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 2u, opt.workspace_allocator);
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

                conv3x3s1_winograd_gemm_transB_packed_tile_fp16sa_rvv(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk);
            }

            // transform output
            conv3x3s1_winograd43_transform_output_tile_fp16sa_rvv(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

static inline void conv3x3s1_winograd63_transform_kernel_tile_fp16sa_rvv(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    __fp16* ptmp = A;

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

                ptmp[0] = (__fp16)z0;
                ptmp[1] = (__fp16)z1;
                ptmp[2] = (__fp16)z2;
                ptmp[3] = (__fp16)z3;
                ptmp[4] = (__fp16)z4;
                ptmp[5] = (__fp16)z5;
                ptmp[6] = (__fp16)z6;
                ptmp[7] = (__fp16)z7;
                ptmp += 8;
            }
        }
    }
}

static void conv3x3s1_winograd63_transform_kernel_fp16sa_rvv(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 64;

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk_fp16sa_rvv(M, 0, K, B, TILE_M, TILE_N, TILE_K, opt.num_threads);

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

            conv3x3s1_winograd63_transform_kernel_tile_fp16sa_rvv(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            conv3x3s1_winograd_pack_A_tile_fp16sa_rvv(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd63_transform_input_tile_fp16sa_rvv(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    // const __fp16 itm[8][8] = {
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

    const int w_tiles = (w + 3) / 6;

    int kk_start = 0;
#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
    const size_t N = bottom_blob.cstep * elempack;
    const size_t vl = __riscv_vsetvl_e16m1(packn);

    {
        const int nn_kk = max_kk / packn;

        #pragma omp parallel for num_threads(nT)
        for (int kk_pack = 0; kk_pack < nn_kk; kk_pack++)
        {
            const int kk = kk_pack * packn;

            __fp16 tmp[8][8][packn];

            int jj = 0;
            for (; jj < max_jj; jj++)
            {
                int ti = (j + jj) / w_tiles;
                int tj = (j + jj) % w_tiles;

                const __fp16* r0123 = bottom_blob.channel((k + kk) / elempack).row<__fp16>(ti * 6) + (tj * 6) * elempack + (k + kk) % elempack;

                for (int m = 0; m < 8; m++)
                {
                    vfloat16m1_t _r0;
                    vfloat16m1_t _r1;
                    vfloat16m1_t _r2;
                    vfloat16m1_t _r3;
                    vfloat16m1_t _r4;
                    vfloat16m1_t _r5;
                    vfloat16m1_t _r6;
                    vfloat16m1_t _r7;

                    _r0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);
                    _r1 = _r0;
                    _r2 = _r0;
                    _r3 = _r0;
                    _r4 = _r0;
                    _r5 = _r0;
                    _r6 = _r0;
                    _r7 = _r0;

                    if (ti * 6 + m < h)
                    {
                        if (elempack == packn)
                        {
                            _r0 = __riscv_vle16_v_f16m1(r0123, vl);
                            if (tj * 6 + 1 < w) _r1 = __riscv_vle16_v_f16m1(r0123 + elempack, vl);
                            if (tj * 6 + 2 < w) _r2 = __riscv_vle16_v_f16m1(r0123 + elempack * 2, vl);
                            if (tj * 6 + 3 < w) _r3 = __riscv_vle16_v_f16m1(r0123 + elempack * 3, vl);
                            if (tj * 6 + 4 < w) _r4 = __riscv_vle16_v_f16m1(r0123 + elempack * 4, vl);
                            if (tj * 6 + 5 < w) _r5 = __riscv_vle16_v_f16m1(r0123 + elempack * 5, vl);
                            if (tj * 6 + 6 < w) _r6 = __riscv_vle16_v_f16m1(r0123 + elempack * 6, vl);
                            if (tj * 6 + 7 < w) _r7 = __riscv_vle16_v_f16m1(r0123 + elempack * 7, vl);
                        }
                        else // if (elempack == 1)
                        {
                            _r0 = __riscv_vlse16_v_f16m1(r0123, N * sizeof(__fp16), vl);
                            if (tj * 6 + 1 < w) _r1 = __riscv_vlse16_v_f16m1(r0123 + elempack, N * sizeof(__fp16), vl);
                            if (tj * 6 + 2 < w) _r2 = __riscv_vlse16_v_f16m1(r0123 + elempack * 2, N * sizeof(__fp16), vl);
                            if (tj * 6 + 3 < w) _r3 = __riscv_vlse16_v_f16m1(r0123 + elempack * 3, N * sizeof(__fp16), vl);
                            if (tj * 6 + 4 < w) _r4 = __riscv_vlse16_v_f16m1(r0123 + elempack * 4, N * sizeof(__fp16), vl);
                            if (tj * 6 + 5 < w) _r5 = __riscv_vlse16_v_f16m1(r0123 + elempack * 5, N * sizeof(__fp16), vl);
                            if (tj * 6 + 6 < w) _r6 = __riscv_vlse16_v_f16m1(r0123 + elempack * 6, N * sizeof(__fp16), vl);
                            if (tj * 6 + 7 < w) _r7 = __riscv_vlse16_v_f16m1(r0123 + elempack * 7, N * sizeof(__fp16), vl);
                        }
                    }

                    vfloat16m1_t _tmp12a = __riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(_r2, _r6, vl), (__fp16)-4.25f, _r4, vl);
                    vfloat16m1_t _tmp12b = __riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(_r1, _r5, vl), (__fp16)-4.25f, _r3, vl);
                    vfloat16m1_t _tmp34a = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(_r6, (__fp16)0.25f, _r2, vl), (__fp16)-1.25f, _r4, vl);
                    vfloat16m1_t _tmp34b = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(__riscv_vfmul_vf_f16m1(_r1, (__fp16)0.5f, vl), (__fp16)-2.5f, _r3, vl), (__fp16)2.f, _r5, vl);
                    vfloat16m1_t _tmp56a = __riscv_vfmacc_vf_f16m1(_r6, (__fp16)4.f, __riscv_vfmacc_vf_f16m1(_r2, (__fp16)-1.25f, _r4, vl), vl);
                    vfloat16m1_t _tmp56b = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(__riscv_vfmul_vf_f16m1(_r1, (__fp16)2.f, vl), (__fp16)-2.5f, _r3, vl), (__fp16)0.5f, _r5, vl);

                    __riscv_vse16_v_f16m1(tmp[0][m], __riscv_vfmacc_vf_f16m1(__riscv_vfsub_vv_f16m1(_r0, _r6, vl), (__fp16)5.25f, __riscv_vfsub_vv_f16m1(_r4, _r2, vl), vl), vl);
                    __riscv_vse16_v_f16m1(tmp[1][m], __riscv_vfadd_vv_f16m1(_tmp12a, _tmp12b, vl), vl);
                    __riscv_vse16_v_f16m1(tmp[2][m], __riscv_vfsub_vv_f16m1(_tmp12a, _tmp12b, vl), vl);
                    __riscv_vse16_v_f16m1(tmp[3][m], __riscv_vfadd_vv_f16m1(_tmp34a, _tmp34b, vl), vl);
                    __riscv_vse16_v_f16m1(tmp[4][m], __riscv_vfsub_vv_f16m1(_tmp34a, _tmp34b, vl), vl);
                    __riscv_vse16_v_f16m1(tmp[5][m], __riscv_vfadd_vv_f16m1(_tmp56a, _tmp56b, vl), vl);
                    __riscv_vse16_v_f16m1(tmp[6][m], __riscv_vfsub_vv_f16m1(_tmp56a, _tmp56b, vl), vl);
                    __riscv_vse16_v_f16m1(tmp[7][m], __riscv_vfmacc_vf_f16m1(__riscv_vfsub_vv_f16m1(_r7, _r1, vl), (__fp16)5.25f, __riscv_vfsub_vv_f16m1(_r3, _r5, vl), vl), vl);

                    r0123 += w * elempack;
                }

                __fp16* p0 = (__fp16*)B + kk * max_jj * 64 + jj * packn;
                __fp16* p1 = p0 + max_jj * packn;
                __fp16* p2 = p0 + max_jj * packn * 2;
                __fp16* p3 = p0 + max_jj * packn * 3;
                __fp16* p4 = p0 + max_jj * packn * 4;
                __fp16* p5 = p0 + max_jj * packn * 5;
                __fp16* p6 = p0 + max_jj * packn * 6;
                __fp16* p7 = p0 + max_jj * packn * 7;

                for (int m = 0; m < 8; m++)
                {
                    vfloat16m1_t _r0 = __riscv_vle16_v_f16m1(tmp[m][0], vl);
                    vfloat16m1_t _r1 = __riscv_vle16_v_f16m1(tmp[m][1], vl);
                    vfloat16m1_t _r2 = __riscv_vle16_v_f16m1(tmp[m][2], vl);
                    vfloat16m1_t _r3 = __riscv_vle16_v_f16m1(tmp[m][3], vl);
                    vfloat16m1_t _r4 = __riscv_vle16_v_f16m1(tmp[m][4], vl);
                    vfloat16m1_t _r5 = __riscv_vle16_v_f16m1(tmp[m][5], vl);
                    vfloat16m1_t _r6 = __riscv_vle16_v_f16m1(tmp[m][6], vl);
                    vfloat16m1_t _r7 = __riscv_vle16_v_f16m1(tmp[m][7], vl);

                    vfloat16m1_t _tmp12a = __riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(_r2, _r6, vl), (__fp16)-4.25f, _r4, vl);
                    vfloat16m1_t _tmp12b = __riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(_r1, _r5, vl), (__fp16)-4.25f, _r3, vl);
                    vfloat16m1_t _tmp34a = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(_r6, (__fp16)0.25f, _r2, vl), (__fp16)-1.25f, _r4, vl);
                    vfloat16m1_t _tmp34b = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(__riscv_vfmul_vf_f16m1(_r1, (__fp16)0.5f, vl), (__fp16)-2.5f, _r3, vl), (__fp16)2.f, _r5, vl);
                    vfloat16m1_t _tmp56a = __riscv_vfmacc_vf_f16m1(_r6, (__fp16)4.f, __riscv_vfmacc_vf_f16m1(_r2, (__fp16)-1.25f, _r4, vl), vl);
                    vfloat16m1_t _tmp56b = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(__riscv_vfmul_vf_f16m1(_r1, (__fp16)2.f, vl), (__fp16)-2.5f, _r3, vl), (__fp16)0.5f, _r5, vl);

                    __riscv_vse16_v_f16m1(p0, __riscv_vfmacc_vf_f16m1(__riscv_vfsub_vv_f16m1(_r0, _r6, vl), (__fp16)5.25f, __riscv_vfsub_vv_f16m1(_r4, _r2, vl), vl), vl);
                    __riscv_vse16_v_f16m1(p1, __riscv_vfadd_vv_f16m1(_tmp12a, _tmp12b, vl), vl);
                    __riscv_vse16_v_f16m1(p2, __riscv_vfsub_vv_f16m1(_tmp12a, _tmp12b, vl), vl);
                    __riscv_vse16_v_f16m1(p3, __riscv_vfadd_vv_f16m1(_tmp34a, _tmp34b, vl), vl);
                    __riscv_vse16_v_f16m1(p4, __riscv_vfsub_vv_f16m1(_tmp34a, _tmp34b, vl), vl);
                    __riscv_vse16_v_f16m1(p5, __riscv_vfadd_vv_f16m1(_tmp56a, _tmp56b, vl), vl);
                    __riscv_vse16_v_f16m1(p6, __riscv_vfsub_vv_f16m1(_tmp56a, _tmp56b, vl), vl);
                    __riscv_vse16_v_f16m1(p7, __riscv_vfmacc_vf_f16m1(__riscv_vfsub_vv_f16m1(_r7, _r1, vl), (__fp16)5.25f, __riscv_vfsub_vv_f16m1(_r3, _r5, vl), vl), vl);

                    p0 += max_jj * 8 * packn;
                    p1 += max_jj * 8 * packn;
                    p2 += max_jj * 8 * packn;
                    p3 += max_jj * 8 * packn;
                    p4 += max_jj * 8 * packn;
                    p5 += max_jj * 8 * packn;
                    p6 += max_jj * 8 * packn;
                    p7 += max_jj * 8 * packn;
                }
            }
        }

        kk_start = nn_kk * packn;
    }
#endif // __riscv_zvfh

    #pragma omp parallel for num_threads(nT)
    for (int kk = kk_start; kk < max_kk; kk++)
    {
        __fp16 tmp[8][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0123 = bottom_blob.channel((k + kk) / elempack).row<__fp16>(ti * 6) + (tj * 6) * elempack + (k + kk) % elempack;

            for (int m = 0; m < 8; m++)
            {
                __fp16 r0 = (__fp16)0.f;
                __fp16 r1 = (__fp16)0.f;
                __fp16 r2 = (__fp16)0.f;
                __fp16 r3 = (__fp16)0.f;
                __fp16 r4 = (__fp16)0.f;
                __fp16 r5 = (__fp16)0.f;
                __fp16 r6 = (__fp16)0.f;
                __fp16 r7 = (__fp16)0.f;

                if (ti * 6 + m < h)
                {
                    // if (elempack == 1)
                    {
                        r0 = r0123[0];
                        if (tj * 6 + 1 < w) r1 = r0123[elempack];
                        if (tj * 6 + 2 < w) r2 = r0123[elempack * 2];
                        if (tj * 6 + 3 < w) r3 = r0123[elempack * 3];
                        if (tj * 6 + 4 < w) r4 = r0123[elempack * 4];
                        if (tj * 6 + 5 < w) r5 = r0123[elempack * 5];
                        if (tj * 6 + 6 < w) r6 = r0123[elempack * 6];
                        if (tj * 6 + 7 < w) r7 = r0123[elempack * 7];
                    }
                }

                __fp16 tmp12a = r2 + r6 - r4 * (__fp16)4.25f;
                __fp16 tmp12b = r1 + r5 - r3 * (__fp16)4.25f;
                __fp16 tmp34a = r6 + r2 * (__fp16)0.25f - r4 * (__fp16)1.25f;
                __fp16 tmp34b = r1 * (__fp16)0.5f - r3 * (__fp16)2.5f + r5 * (__fp16)2.f;
                __fp16 tmp56a = r2 * (__fp16)4.f - r4 * (__fp16)5.f + r6;
                __fp16 tmp56b = r1 * (__fp16)2.f - r3 * (__fp16)2.5f + r5 * (__fp16)0.5f;

                tmp[0][m] = r0 - r6 + (r4 - r2) * (__fp16)5.25f;
                tmp[1][m] = tmp12a + tmp12b;
                tmp[2][m] = tmp12a - tmp12b;
                tmp[3][m] = tmp34a + tmp34b;
                tmp[4][m] = tmp34a - tmp34b;
                tmp[5][m] = tmp56a + tmp56b;
                tmp[6][m] = tmp56a - tmp56b;
                tmp[7][m] = r7 - r1 + (r3 - r5) * (__fp16)5.25f;

                r0123 += w * elempack;
            }

            __fp16* p0 = (__fp16*)B + kk * max_jj * 64 + jj;
            __fp16* p1 = p0 + max_jj;
            __fp16* p2 = p0 + max_jj * 2;
            __fp16* p3 = p0 + max_jj * 3;
            __fp16* p4 = p0 + max_jj * 4;
            __fp16* p5 = p0 + max_jj * 5;
            __fp16* p6 = p0 + max_jj * 6;
            __fp16* p7 = p0 + max_jj * 7;

            for (int m = 0; m < 8; m++)
            {
                __fp16 r0 = tmp[m][0];
                __fp16 r1 = tmp[m][1];
                __fp16 r2 = tmp[m][2];
                __fp16 r3 = tmp[m][3];
                __fp16 r4 = tmp[m][4];
                __fp16 r5 = tmp[m][5];
                __fp16 r6 = tmp[m][6];
                __fp16 r7 = tmp[m][7];

                __fp16 tmp12a = r2 + r6 - r4 * (__fp16)4.25f;
                __fp16 tmp12b = r1 + r5 - r3 * (__fp16)4.25f;
                __fp16 tmp34a = r6 + r2 * (__fp16)0.25f - r4 * (__fp16)1.25f;
                __fp16 tmp34b = r1 * (__fp16)0.5f - r3 * (__fp16)2.5f + r5 * (__fp16)2.f;
                __fp16 tmp56a = r2 * (__fp16)4.f - r4 * (__fp16)5.f + r6;
                __fp16 tmp56b = r1 * (__fp16)2.f - r3 * (__fp16)2.5f + r5 * (__fp16)0.5f;

                p0[0] = r0 - r6 + (r4 - r2) * (__fp16)5.25f;
                p1[0] = tmp12a + tmp12b;
                p2[0] = tmp12a - tmp12b;
                p3[0] = tmp34a + tmp34b;
                p4[0] = tmp34a - tmp34b;
                p5[0] = tmp56a + tmp56b;
                p6[0] = tmp56a - tmp56b;
                p7[0] = r7 - r1 + (r3 - r5) * (__fp16)5.25f;

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

static inline void conv3x3s1_winograd63_transform_output_tile_fp16sa_rvv(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
{
    // const __fp16 otm[6][8] = {
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
#if __riscv_zvfh
    const size_t out_hstep = top_blob.cstep;
#endif

    const int w_tiles = (outw + 5) / 6;

    const __fp16* biasptr = bias;

    int ii = 0;
#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
    size_t vl;
    if (packn == 8)
        vl = __riscv_vsetvl_e16m1(8);
    else if (packn == 16)
        vl = __riscv_vsetvl_e16m1(16);
    else
        vl = __riscv_vsetvl_e16m1(packn);

    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        vfloat16m1_t _bias0 = biasptr ? __riscv_vle16_v_f16m1(biasptr + i + ii, vl) : __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);

        __fp16 tmp[6][8][packn];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 64 + jj * packn;
            const __fp16* r1 = r0 + max_jj * packn;
            const __fp16* r2 = r0 + max_jj * packn * 2;
            const __fp16* r3 = r0 + max_jj * packn * 3;
            const __fp16* r4 = r0 + max_jj * packn * 4;
            const __fp16* r5 = r0 + max_jj * packn * 5;
            const __fp16* r6 = r0 + max_jj * packn * 6;
            const __fp16* r7 = r0 + max_jj * packn * 7;

            for (int m = 0; m < 8; m++)
            {
                vfloat16m1_t _r0 = __riscv_vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r1 = __riscv_vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r2 = __riscv_vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r3 = __riscv_vle16_v_f16m1(r3, vl);
                vfloat16m1_t _r4 = __riscv_vle16_v_f16m1(r4, vl);
                vfloat16m1_t _r5 = __riscv_vle16_v_f16m1(r5, vl);
                vfloat16m1_t _r6 = __riscv_vle16_v_f16m1(r6, vl);
                vfloat16m1_t _r7 = __riscv_vle16_v_f16m1(r7, vl);

                vfloat16m1_t _tmp024a = __riscv_vfadd_vv_f16m1(_r1, _r2, vl);
                vfloat16m1_t _tmp135a = __riscv_vfsub_vv_f16m1(_r1, _r2, vl);
                vfloat16m1_t _tmp024b = __riscv_vfadd_vv_f16m1(_r3, _r4, vl);
                vfloat16m1_t _tmp135b = __riscv_vfsub_vv_f16m1(_r3, _r4, vl);
                vfloat16m1_t _tmp024c = __riscv_vfadd_vv_f16m1(_r5, _r6, vl);
                vfloat16m1_t _tmp135c = __riscv_vfsub_vv_f16m1(_r5, _r6, vl);

                vfloat16m1_t _tmp0 = __riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(__riscv_vfadd_vv_f16m1(_r0, _tmp024a, vl), _tmp024b, vl), (__fp16)32.f, _tmp024c, vl);
                vfloat16m1_t _tmp1 = __riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(_tmp135a, __riscv_vfadd_vv_f16m1(_tmp135b, _tmp135b, vl), vl), (__fp16)16.f, _tmp135c, vl);
                vfloat16m1_t _tmp2 = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(_tmp024a, (__fp16)4.f, _tmp024b, vl), (__fp16)8.f, _tmp024c, vl);
                vfloat16m1_t _tmp3 = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(_tmp135a, (__fp16)8.f, _tmp135b, vl), (__fp16)4.f, _tmp135c, vl);
                vfloat16m1_t _tmp4 = __riscv_vfadd_vv_f16m1(__riscv_vfmacc_vf_f16m1(_tmp024a, (__fp16)16.f, _tmp024b, vl), __riscv_vfadd_vv_f16m1(_tmp024c, _tmp024c, vl), vl);
                vfloat16m1_t _tmp5 = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(_r7, _tmp135a, vl), (__fp16)32.f, _tmp135b, vl), (__fp16)1.f, _tmp135c, vl);
                __riscv_vse16_v_f16m1(tmp[0][m], _tmp0, vl);
                __riscv_vse16_v_f16m1(tmp[1][m], _tmp1, vl);
                __riscv_vse16_v_f16m1(tmp[2][m], _tmp2, vl);
                __riscv_vse16_v_f16m1(tmp[3][m], _tmp3, vl);
                __riscv_vse16_v_f16m1(tmp[4][m], _tmp4, vl);
                __riscv_vse16_v_f16m1(tmp[5][m], _tmp5, vl);

                r0 += max_jj * 8 * packn;
                r1 += max_jj * 8 * packn;
                r2 += max_jj * 8 * packn;
                r3 += max_jj * 8 * packn;
                r4 += max_jj * 8 * packn;
                r5 += max_jj * 8 * packn;
                r6 += max_jj * 8 * packn;
                r7 += max_jj * 8 * packn;
            }

            __fp16* outptr0 = top_blob.channel((i + ii) / out_elempack).row<__fp16>(ti * 6) + (tj * 6) * out_elempack + (i + ii) % out_elempack;

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                vfloat16m1_t _r0 = __riscv_vle16_v_f16m1(tmp[m][0], vl);
                vfloat16m1_t _r1 = __riscv_vle16_v_f16m1(tmp[m][1], vl);
                vfloat16m1_t _r2 = __riscv_vle16_v_f16m1(tmp[m][2], vl);
                vfloat16m1_t _r3 = __riscv_vle16_v_f16m1(tmp[m][3], vl);
                vfloat16m1_t _r4 = __riscv_vle16_v_f16m1(tmp[m][4], vl);
                vfloat16m1_t _r5 = __riscv_vle16_v_f16m1(tmp[m][5], vl);
                vfloat16m1_t _r6 = __riscv_vle16_v_f16m1(tmp[m][6], vl);
                vfloat16m1_t _r7 = __riscv_vle16_v_f16m1(tmp[m][7], vl);

                vfloat16m1_t _tmp024a = __riscv_vfadd_vv_f16m1(_r1, _r2, vl);
                vfloat16m1_t _tmp135a = __riscv_vfsub_vv_f16m1(_r1, _r2, vl);
                vfloat16m1_t _tmp024b = __riscv_vfadd_vv_f16m1(_r3, _r4, vl);
                vfloat16m1_t _tmp135b = __riscv_vfsub_vv_f16m1(_r3, _r4, vl);
                vfloat16m1_t _tmp024c = __riscv_vfadd_vv_f16m1(_r5, _r6, vl);
                vfloat16m1_t _tmp135c = __riscv_vfsub_vv_f16m1(_r5, _r6, vl);

                vfloat16m1_t _tmp0 = __riscv_vfadd_vv_f16m1(_bias0, __riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(__riscv_vfadd_vv_f16m1(_r0, _tmp024a, vl), _tmp024b, vl), (__fp16)32.f, _tmp024c, vl), vl);
                vfloat16m1_t _tmp1 = __riscv_vfadd_vv_f16m1(_bias0, __riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(_tmp135a, __riscv_vfadd_vv_f16m1(_tmp135b, _tmp135b, vl), vl), (__fp16)16.f, _tmp135c, vl), vl);
                vfloat16m1_t _tmp2 = __riscv_vfadd_vv_f16m1(_bias0, __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(_tmp024a, (__fp16)4.f, _tmp024b, vl), (__fp16)8.f, _tmp024c, vl), vl);
                vfloat16m1_t _tmp3 = __riscv_vfadd_vv_f16m1(_bias0, __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(_tmp135a, (__fp16)8.f, _tmp135b, vl), (__fp16)4.f, _tmp135c, vl), vl);
                vfloat16m1_t _tmp4 = __riscv_vfadd_vv_f16m1(_bias0, __riscv_vfadd_vv_f16m1(__riscv_vfmacc_vf_f16m1(_tmp024a, (__fp16)16.f, _tmp024b, vl), __riscv_vfadd_vv_f16m1(_tmp024c, _tmp024c, vl), vl), vl);
                vfloat16m1_t _tmp5 = __riscv_vfadd_vv_f16m1(_bias0, __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(_r7, _tmp135a, vl), (__fp16)32.f, _tmp135b, vl), (__fp16)1.f, _tmp135c, vl), vl);

                if (out_elempack == packn)
                {
                    __riscv_vse16_v_f16m1(outptr0, _tmp0, vl);
                    if (tj * 6 + 1 < outw) __riscv_vse16_v_f16m1(outptr0 + out_elempack, _tmp1, vl);
                    if (tj * 6 + 2 < outw) __riscv_vse16_v_f16m1(outptr0 + out_elempack * 2, _tmp2, vl);
                    if (tj * 6 + 3 < outw) __riscv_vse16_v_f16m1(outptr0 + out_elempack * 3, _tmp3, vl);
                    if (tj * 6 + 4 < outw) __riscv_vse16_v_f16m1(outptr0 + out_elempack * 4, _tmp4, vl);
                    if (tj * 6 + 5 < outw) __riscv_vse16_v_f16m1(outptr0 + out_elempack * 5, _tmp5, vl);
                }
                else
                {
                    __riscv_vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _tmp0, vl);
                    if (tj * 6 + 1 < outw) __riscv_vsse16_v_f16m1(outptr0 + out_elempack, out_hstep * sizeof(__fp16), _tmp1, vl);
                    if (tj * 6 + 2 < outw) __riscv_vsse16_v_f16m1(outptr0 + out_elempack * 2, out_hstep * sizeof(__fp16), _tmp2, vl);
                    if (tj * 6 + 3 < outw) __riscv_vsse16_v_f16m1(outptr0 + out_elempack * 3, out_hstep * sizeof(__fp16), _tmp3, vl);
                    if (tj * 6 + 4 < outw) __riscv_vsse16_v_f16m1(outptr0 + out_elempack * 4, out_hstep * sizeof(__fp16), _tmp4, vl);
                    if (tj * 6 + 5 < outw) __riscv_vsse16_v_f16m1(outptr0 + out_elempack * 5, out_hstep * sizeof(__fp16), _tmp5, vl);
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __riscv_zvfh

#if __riscv_zvfh
    const size_t vl4 = __riscv_vsetvl_e16m1(4);
    for (; ii + 3 < max_ii; ii += 4)
    {
        vfloat16m1_t _bias0 = biasptr ? __riscv_vle16_v_f16m1(biasptr + i + ii, vl4) : __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl4);

        __fp16 tmp[6][8][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 64 + jj * 4;
            const __fp16* r1 = r0 + max_jj * 4;
            const __fp16* r2 = r0 + max_jj * 4 * 2;
            const __fp16* r3 = r0 + max_jj * 4 * 3;
            const __fp16* r4 = r0 + max_jj * 4 * 4;
            const __fp16* r5 = r0 + max_jj * 4 * 5;
            const __fp16* r6 = r0 + max_jj * 4 * 6;
            const __fp16* r7 = r0 + max_jj * 4 * 7;

            for (int m = 0; m < 8; m++)
            {
                vfloat16m1_t _r0 = __riscv_vle16_v_f16m1(r0, vl4);
                vfloat16m1_t _r1 = __riscv_vle16_v_f16m1(r1, vl4);
                vfloat16m1_t _r2 = __riscv_vle16_v_f16m1(r2, vl4);
                vfloat16m1_t _r3 = __riscv_vle16_v_f16m1(r3, vl4);
                vfloat16m1_t _r4 = __riscv_vle16_v_f16m1(r4, vl4);
                vfloat16m1_t _r5 = __riscv_vle16_v_f16m1(r5, vl4);
                vfloat16m1_t _r6 = __riscv_vle16_v_f16m1(r6, vl4);
                vfloat16m1_t _r7 = __riscv_vle16_v_f16m1(r7, vl4);

                vfloat16m1_t _tmp024a = __riscv_vfadd_vv_f16m1(_r1, _r2, vl4);
                vfloat16m1_t _tmp135a = __riscv_vfsub_vv_f16m1(_r1, _r2, vl4);
                vfloat16m1_t _tmp024b = __riscv_vfadd_vv_f16m1(_r3, _r4, vl4);
                vfloat16m1_t _tmp135b = __riscv_vfsub_vv_f16m1(_r3, _r4, vl4);
                vfloat16m1_t _tmp024c = __riscv_vfadd_vv_f16m1(_r5, _r6, vl4);
                vfloat16m1_t _tmp135c = __riscv_vfsub_vv_f16m1(_r5, _r6, vl4);

                vfloat16m1_t _tmp0 = __riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(__riscv_vfadd_vv_f16m1(_r0, _tmp024a, vl4), _tmp024b, vl4), (__fp16)32.f, _tmp024c, vl4);
                vfloat16m1_t _tmp1 = __riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(_tmp135a, __riscv_vfadd_vv_f16m1(_tmp135b, _tmp135b, vl4), vl4), (__fp16)16.f, _tmp135c, vl4);
                vfloat16m1_t _tmp2 = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(_tmp024a, (__fp16)4.f, _tmp024b, vl4), (__fp16)8.f, _tmp024c, vl4);
                vfloat16m1_t _tmp3 = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(_tmp135a, (__fp16)8.f, _tmp135b, vl4), (__fp16)4.f, _tmp135c, vl4);
                vfloat16m1_t _tmp4 = __riscv_vfadd_vv_f16m1(__riscv_vfmacc_vf_f16m1(_tmp024a, (__fp16)16.f, _tmp024b, vl4), __riscv_vfadd_vv_f16m1(_tmp024c, _tmp024c, vl4), vl4);
                vfloat16m1_t _tmp5 = __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(_r7, _tmp135a, vl4), (__fp16)32.f, _tmp135b, vl4), (__fp16)1.f, _tmp135c, vl4);
                __riscv_vse16_v_f16m1(tmp[0][m], _tmp0, vl4);
                __riscv_vse16_v_f16m1(tmp[1][m], _tmp1, vl4);
                __riscv_vse16_v_f16m1(tmp[2][m], _tmp2, vl4);
                __riscv_vse16_v_f16m1(tmp[3][m], _tmp3, vl4);
                __riscv_vse16_v_f16m1(tmp[4][m], _tmp4, vl4);
                __riscv_vse16_v_f16m1(tmp[5][m], _tmp5, vl4);

                r0 += max_jj * 8 * 4;
                r1 += max_jj * 8 * 4;
                r2 += max_jj * 8 * 4;
                r3 += max_jj * 8 * 4;
                r4 += max_jj * 8 * 4;
                r5 += max_jj * 8 * 4;
                r6 += max_jj * 8 * 4;
                r7 += max_jj * 8 * 4;
            }

            __fp16* outptr0 = top_blob.channel((i + ii) / out_elempack).row<__fp16>(ti * 6) + (tj * 6) * out_elempack + (i + ii) % out_elempack;

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                vfloat16m1_t _r0 = __riscv_vle16_v_f16m1(tmp[m][0], vl4);
                vfloat16m1_t _r1 = __riscv_vle16_v_f16m1(tmp[m][1], vl4);
                vfloat16m1_t _r2 = __riscv_vle16_v_f16m1(tmp[m][2], vl4);
                vfloat16m1_t _r3 = __riscv_vle16_v_f16m1(tmp[m][3], vl4);
                vfloat16m1_t _r4 = __riscv_vle16_v_f16m1(tmp[m][4], vl4);
                vfloat16m1_t _r5 = __riscv_vle16_v_f16m1(tmp[m][5], vl4);
                vfloat16m1_t _r6 = __riscv_vle16_v_f16m1(tmp[m][6], vl4);
                vfloat16m1_t _r7 = __riscv_vle16_v_f16m1(tmp[m][7], vl4);

                vfloat16m1_t _tmp024a = __riscv_vfadd_vv_f16m1(_r1, _r2, vl4);
                vfloat16m1_t _tmp135a = __riscv_vfsub_vv_f16m1(_r1, _r2, vl4);
                vfloat16m1_t _tmp024b = __riscv_vfadd_vv_f16m1(_r3, _r4, vl4);
                vfloat16m1_t _tmp135b = __riscv_vfsub_vv_f16m1(_r3, _r4, vl4);
                vfloat16m1_t _tmp024c = __riscv_vfadd_vv_f16m1(_r5, _r6, vl4);
                vfloat16m1_t _tmp135c = __riscv_vfsub_vv_f16m1(_r5, _r6, vl4);

                vfloat16m1_t _tmp0 = __riscv_vfadd_vv_f16m1(_bias0, __riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(__riscv_vfadd_vv_f16m1(_r0, _tmp024a, vl4), _tmp024b, vl4), (__fp16)32.f, _tmp024c, vl4), vl4);
                vfloat16m1_t _tmp1 = __riscv_vfadd_vv_f16m1(_bias0, __riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(_tmp135a, __riscv_vfadd_vv_f16m1(_tmp135b, _tmp135b, vl4), vl4), (__fp16)16.f, _tmp135c, vl4), vl4);
                vfloat16m1_t _tmp2 = __riscv_vfadd_vv_f16m1(_bias0, __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(_tmp024a, (__fp16)4.f, _tmp024b, vl4), (__fp16)8.f, _tmp024c, vl4), vl4);
                vfloat16m1_t _tmp3 = __riscv_vfadd_vv_f16m1(_bias0, __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(_tmp135a, (__fp16)8.f, _tmp135b, vl4), (__fp16)4.f, _tmp135c, vl4), vl4);
                vfloat16m1_t _tmp4 = __riscv_vfadd_vv_f16m1(_bias0, __riscv_vfadd_vv_f16m1(__riscv_vfmacc_vf_f16m1(_tmp024a, (__fp16)16.f, _tmp024b, vl4), __riscv_vfadd_vv_f16m1(_tmp024c, _tmp024c, vl4), vl4), vl4);
                vfloat16m1_t _tmp5 = __riscv_vfadd_vv_f16m1(_bias0, __riscv_vfmacc_vf_f16m1(__riscv_vfmacc_vf_f16m1(__riscv_vfadd_vv_f16m1(_r7, _tmp135a, vl4), (__fp16)32.f, _tmp135b, vl4), (__fp16)1.f, _tmp135c, vl4), vl4);

                if (out_elempack == packn)
                {
                    __riscv_vse16_v_f16m1(outptr0, _tmp0, vl4);
                    if (tj * 6 + 1 < outw) __riscv_vse16_v_f16m1(outptr0 + out_elempack, _tmp1, vl4);
                    if (tj * 6 + 2 < outw) __riscv_vse16_v_f16m1(outptr0 + out_elempack * 2, _tmp2, vl4);
                    if (tj * 6 + 3 < outw) __riscv_vse16_v_f16m1(outptr0 + out_elempack * 3, _tmp3, vl4);
                    if (tj * 6 + 4 < outw) __riscv_vse16_v_f16m1(outptr0 + out_elempack * 4, _tmp4, vl4);
                    if (tj * 6 + 5 < outw) __riscv_vse16_v_f16m1(outptr0 + out_elempack * 5, _tmp5, vl4);
                }
                else
                {
                    __riscv_vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _tmp0, vl4);
                    if (tj * 6 + 1 < outw) __riscv_vsse16_v_f16m1(outptr0 + out_elempack, out_hstep * sizeof(__fp16), _tmp1, vl4);
                    if (tj * 6 + 2 < outw) __riscv_vsse16_v_f16m1(outptr0 + out_elempack * 2, out_hstep * sizeof(__fp16), _tmp2, vl4);
                    if (tj * 6 + 3 < outw) __riscv_vsse16_v_f16m1(outptr0 + out_elempack * 3, out_hstep * sizeof(__fp16), _tmp3, vl4);
                    if (tj * 6 + 4 < outw) __riscv_vsse16_v_f16m1(outptr0 + out_elempack * 4, out_hstep * sizeof(__fp16), _tmp4, vl4);
                    if (tj * 6 + 5 < outw) __riscv_vsse16_v_f16m1(outptr0 + out_elempack * 5, out_hstep * sizeof(__fp16), _tmp5, vl4);
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __riscv_zvfh

    for (; ii + 1 < max_ii; ii += 2)
    {
        __fp16 bias0 = biasptr ? biasptr[i + ii] : (__fp16)0.f;
        __fp16 bias1 = biasptr ? biasptr[i + ii + 1] : (__fp16)0.f;

        __fp16 tmp[6][8][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 64 + jj * 2;
            const __fp16* r1 = r0 + max_jj * 2;
            const __fp16* r2 = r0 + max_jj * 2 * 2;
            const __fp16* r3 = r0 + max_jj * 2 * 3;
            const __fp16* r4 = r0 + max_jj * 2 * 4;
            const __fp16* r5 = r0 + max_jj * 2 * 5;
            const __fp16* r6 = r0 + max_jj * 2 * 6;
            const __fp16* r7 = r0 + max_jj * 2 * 7;

            for (int m = 0; m < 8; m++)
            {
                __fp16 tmp024a0 = r1[0] + r2[0];
                __fp16 tmp024a1 = r1[1] + r2[1];
                __fp16 tmp135a0 = r1[0] - r2[0];
                __fp16 tmp135a1 = r1[1] - r2[1];
                __fp16 tmp024b0 = r3[0] + r4[0];
                __fp16 tmp024b1 = r3[1] + r4[1];
                __fp16 tmp135b0 = r3[0] - r4[0];
                __fp16 tmp135b1 = r3[1] - r4[1];
                __fp16 tmp024c0 = r5[0] + r6[0];
                __fp16 tmp024c1 = r5[1] + r6[1];
                __fp16 tmp135c0 = r5[0] - r6[0];
                __fp16 tmp135c1 = r5[1] - r6[1];

                tmp[0][m][0] = r0[0] + tmp024a0 + tmp024b0 + tmp024c0 * (__fp16)32;
                tmp[0][m][1] = r0[1] + tmp024a1 + tmp024b1 + tmp024c1 * (__fp16)32;
                tmp[1][m][0] = tmp135a0 + tmp135b0 + tmp135b0 + tmp135c0 * (__fp16)16;
                tmp[1][m][1] = tmp135a1 + tmp135b1 + tmp135b1 + tmp135c1 * (__fp16)16;
                tmp[2][m][0] = tmp024a0 + tmp024b0 * (__fp16)4 + tmp024c0 * (__fp16)8;
                tmp[2][m][1] = tmp024a1 + tmp024b1 * (__fp16)4 + tmp024c1 * (__fp16)8;
                tmp[3][m][0] = tmp135a0 + tmp135b0 * (__fp16)8 + tmp135c0 * (__fp16)4;
                tmp[3][m][1] = tmp135a1 + tmp135b1 * (__fp16)8 + tmp135c1 * (__fp16)4;
                tmp[4][m][0] = tmp024a0 + tmp024b0 * (__fp16)16 + tmp024c0 + tmp024c0;
                tmp[4][m][1] = tmp024a1 + tmp024b1 * (__fp16)16 + tmp024c1 + tmp024c1;
                tmp[5][m][0] = r7[0] + tmp135a0 + tmp135b0 * (__fp16)32 + tmp135c0;
                tmp[5][m][1] = r7[1] + tmp135a1 + tmp135b1 * (__fp16)32 + tmp135c1;

                r0 += max_jj * 8 * 2;
                r1 += max_jj * 8 * 2;
                r2 += max_jj * 8 * 2;
                r3 += max_jj * 8 * 2;
                r4 += max_jj * 8 * 2;
                r5 += max_jj * 8 * 2;
                r6 += max_jj * 8 * 2;
                r7 += max_jj * 8 * 2;
            }

            __fp16* outptr0 = top_blob.channel((i + ii) / out_elempack).row<__fp16>(ti * 6) + (tj * 6) * out_elempack + (i + ii) % out_elempack;
            __fp16* outptr1 = top_blob.channel((i + ii + 1) / out_elempack).row<__fp16>(ti * 6) + (tj * 6) * out_elempack + (i + ii + 1) % out_elempack;

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                __fp16 r00 = tmp[m][0][0];
                __fp16 r01 = tmp[m][0][1];
                __fp16 r10 = tmp[m][1][0];
                __fp16 r11 = tmp[m][1][1];
                __fp16 r20 = tmp[m][2][0];
                __fp16 r21 = tmp[m][2][1];
                __fp16 r30 = tmp[m][3][0];
                __fp16 r31 = tmp[m][3][1];
                __fp16 r40 = tmp[m][4][0];
                __fp16 r41 = tmp[m][4][1];
                __fp16 r50 = tmp[m][5][0];
                __fp16 r51 = tmp[m][5][1];
                __fp16 r60 = tmp[m][6][0];
                __fp16 r61 = tmp[m][6][1];
                __fp16 r70 = tmp[m][7][0];
                __fp16 r71 = tmp[m][7][1];

                __fp16 tmp024a0 = r10 + r20;
                __fp16 tmp024a1 = r11 + r21;
                __fp16 tmp135a0 = r10 - r20;
                __fp16 tmp135a1 = r11 - r21;
                __fp16 tmp024b0 = r30 + r40;
                __fp16 tmp024b1 = r31 + r41;
                __fp16 tmp135b0 = r30 - r40;
                __fp16 tmp135b1 = r31 - r41;
                __fp16 tmp024c0 = r50 + r60;
                __fp16 tmp024c1 = r51 + r61;
                __fp16 tmp135c0 = r50 - r60;
                __fp16 tmp135c1 = r51 - r61;

                __fp16 tmp00 = bias0 + r00 + tmp024a0 + tmp024b0 + tmp024c0 * (__fp16)32;
                __fp16 tmp01 = bias1 + r01 + tmp024a1 + tmp024b1 + tmp024c1 * (__fp16)32;
                __fp16 tmp10 = bias0 + tmp135a0 + tmp135b0 + tmp135b0 + tmp135c0 * (__fp16)16;
                __fp16 tmp11 = bias1 + tmp135a1 + tmp135b1 + tmp135b1 + tmp135c1 * (__fp16)16;
                __fp16 tmp20 = bias0 + tmp024a0 + tmp024b0 * (__fp16)4 + tmp024c0 * (__fp16)8;
                __fp16 tmp21 = bias1 + tmp024a1 + tmp024b1 * (__fp16)4 + tmp024c1 * (__fp16)8;
                __fp16 tmp30 = bias0 + tmp135a0 + tmp135b0 * (__fp16)8 + tmp135c0 * (__fp16)4;
                __fp16 tmp31 = bias1 + tmp135a1 + tmp135b1 * (__fp16)8 + tmp135c1 * (__fp16)4;
                __fp16 tmp40 = bias0 + tmp024a0 + tmp024b0 * (__fp16)16 + tmp024c0 + tmp024c0;
                __fp16 tmp41 = bias1 + tmp024a1 + tmp024b1 * (__fp16)16 + tmp024c1 + tmp024c1;
                __fp16 tmp50 = bias0 + r70 + tmp135a0 + tmp135b0 * (__fp16)32 + tmp135c0;
                __fp16 tmp51 = bias1 + r71 + tmp135a1 + tmp135b1 * (__fp16)32 + tmp135c1;

                // if (out_elempack == 1)
                {
                    outptr0[0] = tmp00;
                    outptr1[0] = tmp01;
                    if (tj * 6 + 1 < outw)
                    {
                        outptr0[out_elempack] = tmp10;
                        outptr1[out_elempack] = tmp11;
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        outptr0[out_elempack * 2] = tmp20;
                        outptr1[out_elempack * 2] = tmp21;
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        outptr0[out_elempack * 3] = tmp30;
                        outptr1[out_elempack * 3] = tmp31;
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        outptr0[out_elempack * 4] = tmp40;
                        outptr1[out_elempack * 4] = tmp41;
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        outptr0[out_elempack * 5] = tmp50;
                        outptr1[out_elempack * 5] = tmp51;
                    }
                }

                outptr0 += outw * out_elempack;
                outptr1 += outw * out_elempack;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        __fp16 bias0 = biasptr ? biasptr[i + ii] : (__fp16)0.f;

        __fp16 tmp[6][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const __fp16* r0 = (const __fp16*)top_tile + ii * max_jj * 64 + jj;
            const __fp16* r1 = r0 + max_jj;
            const __fp16* r2 = r0 + max_jj * 2;
            const __fp16* r3 = r0 + max_jj * 3;
            const __fp16* r4 = r0 + max_jj * 4;
            const __fp16* r5 = r0 + max_jj * 5;
            const __fp16* r6 = r0 + max_jj * 6;
            const __fp16* r7 = r0 + max_jj * 7;

            for (int m = 0; m < 8; m++)
            {
                __fp16 tmp024a = r1[0] + r2[0];
                __fp16 tmp135a = r1[0] - r2[0];
                __fp16 tmp024b = r3[0] + r4[0];
                __fp16 tmp135b = r3[0] - r4[0];
                __fp16 tmp024c = r5[0] + r6[0];
                __fp16 tmp135c = r5[0] - r6[0];

                tmp[0][m] = r0[0] + tmp024a + tmp024b + tmp024c * (__fp16)32;
                tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * (__fp16)16;
                tmp[2][m] = tmp024a + tmp024b * (__fp16)4 + tmp024c * (__fp16)8;
                tmp[3][m] = tmp135a + tmp135b * (__fp16)8 + tmp135c * (__fp16)4;
                tmp[4][m] = tmp024a + tmp024b * (__fp16)16 + tmp024c + tmp024c;
                tmp[5][m] = r7[0] + tmp135a + tmp135b * (__fp16)32 + tmp135c;

                r0 += max_jj * 8;
                r1 += max_jj * 8;
                r2 += max_jj * 8;
                r3 += max_jj * 8;
                r4 += max_jj * 8;
                r5 += max_jj * 8;
                r6 += max_jj * 8;
                r7 += max_jj * 8;
            }

            __fp16* outptr0 = top_blob.channel((i + ii) / out_elempack).row<__fp16>(ti * 6) + (tj * 6) * out_elempack + (i + ii) % out_elempack;

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                __fp16 r0 = tmp[m][0];
                __fp16 r1 = tmp[m][1];
                __fp16 r2 = tmp[m][2];
                __fp16 r3 = tmp[m][3];
                __fp16 r4 = tmp[m][4];
                __fp16 r5 = tmp[m][5];
                __fp16 r6 = tmp[m][6];
                __fp16 r7 = tmp[m][7];

                __fp16 tmp024a = r1 + r2;
                __fp16 tmp135a = r1 - r2;
                __fp16 tmp024b = r3 + r4;
                __fp16 tmp135b = r3 - r4;
                __fp16 tmp024c = r5 + r6;
                __fp16 tmp135c = r5 - r6;

                __fp16 tmp0 = bias0 + r0 + tmp024a + tmp024b + tmp024c * (__fp16)32;
                __fp16 tmp1 = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * (__fp16)16;
                __fp16 tmp2 = bias0 + tmp024a + tmp024b * (__fp16)4 + tmp024c * (__fp16)8;
                __fp16 tmp3 = bias0 + tmp135a + tmp135b * (__fp16)8 + tmp135c * (__fp16)4;
                __fp16 tmp4 = bias0 + tmp024a + tmp024b * (__fp16)16 + tmp024c + tmp024c;
                __fp16 tmp5 = bias0 + r7 + tmp135a + tmp135b * (__fp16)32 + tmp135c;

                // if (out_elempack == 1)
                {
                    outptr0[0] = tmp0;
                    if (tj * 6 + 1 < outw) outptr0[out_elempack] = tmp1;
                    if (tj * 6 + 2 < outw) outptr0[out_elempack * 2] = tmp2;
                    if (tj * 6 + 3 < outw) outptr0[out_elempack * 3] = tmp3;
                    if (tj * 6 + 4 < outw) outptr0[out_elempack * 4] = tmp4;
                    if (tj * 6 + 5 < outw) outptr0[out_elempack * 5] = tmp5;
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
}

static int conv3x3s1_winograd63_fp16sa_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, const Option& opt)
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
    conv3x3s1_winograd_get_optimal_tile_mnk_fp16sa_rvv(M, N, K, B, TILE_M, TILE_N, TILE_K, nT);

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
            conv3x3s1_winograd63_transform_input_tile_fp16sa_rvv(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile_fp16sa_rvv(B_tile, BT_tile, B, max_jj, max_kk, nT);
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
            conv3x3s1_winograd63_transform_input_tile_fp16sa_rvv(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile_fp16sa_rvv(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 2u, opt.workspace_allocator);
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

                conv3x3s1_winograd_gemm_transB_packed_tile_fp16sa_rvv(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk);
            }

            // transform output
            conv3x3s1_winograd63_transform_output_tile_fp16sa_rvv(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

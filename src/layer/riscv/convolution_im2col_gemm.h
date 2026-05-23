// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convolution_im2col_pack_A_tile_rvv(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    // A = (elempack, maxk, inch/elempack), outch
    const int A_hstep = A.w;

    float* pp = AT;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        for (int kk = 0; kk < max_kk; kk++)
        {
            vfloat32m1_t _r0 = __riscv_vlse32_v_f32m1(p0, A_hstep * sizeof(float), vl);
            __riscv_vse32_v_f32m1(pp, _r0, vl);

            pp += packn;
            p0++;
        }
    }
#else
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }
#endif // __riscv_vector

    for (; ii < max_ii; ii++)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp++;
            p0++;
        }
    }
}

#if __riscv_vector
static void convolution_im2col_pack_A_tile_direct_rvv(const Mat& weight_data_r2, Mat& AT, int i, int max_ii, int k, int max_kk, int maxk, int elempack)
{
    float* pp = AT;

    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

    int ii = 0;
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        for (int kk = 0; kk < max_kk; kk++)
        {
            int kk_global = k + kk;
            int p = kk_global / (maxk * elempack) * elempack + kk_global % elempack;
            int uv = (kk_global / elempack) % maxk;

            const float* k00 = weight_data_r2.channel(i + ii).row(p) + uv;

            vfloat32m1_t _r0 = __riscv_vlse32_v_f32m1(k00, weight_data_r2.cstep * sizeof(float), vl);
            __riscv_vse32_v_f32m1(pp, _r0, vl);

            pp += packn;
        }
    }

    for (; ii < max_ii; ii++)
    {
        for (int kk = 0; kk < max_kk; kk++)
        {
            int kk_global = k + kk;
            int p = kk_global / (maxk * elempack) * elempack + kk_global % elempack;
            int uv = (kk_global / elempack) % maxk;

            const float* k00 = weight_data_r2.channel(i + ii).row(p);

            pp[0] = k00[uv];
            pp++;
        }
    }
}
#endif // __riscv_vector

static void convolution_gemm_transB_packed_tile_rvv(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.cstep;

    const float* pAT = AT_tile;
    const float* pBT = BT_tile;
    const float* biasptr = CT_tile;

    float* outptr = topT_tile;


    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        float* outptr0;
        if (out_elempack == packn)
            outptr0 = (float*)top_blob.channel((i + ii) / packn) + j * packn;
        else // if (out_elempack == 1)
            outptr0 = (float*)top_blob.channel(i + ii) + j;

        const float* pB = pBT;
        const float* pC = biasptr ? biasptr + i + ii : 0;

        int jj = 0;
        for (; jj + 15 < max_jj; jj += 16)
        {
            const float* pA = pAT;

            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;
            vfloat32m1_t _sum2;
            vfloat32m1_t _sum3;
            vfloat32m1_t _sum4;
            vfloat32m1_t _sum5;
            vfloat32m1_t _sum6;
            vfloat32m1_t _sum7;
            vfloat32m1_t _sum8;
            vfloat32m1_t _sum9;
            vfloat32m1_t _suma;
            vfloat32m1_t _sumb;
            vfloat32m1_t _sumc;
            vfloat32m1_t _sumd;
            vfloat32m1_t _sume;
            vfloat32m1_t _sumf;

            if (k == 0)
            {
                if (pC)
                    _sum0 = __riscv_vle32_v_f32m1(pC, vl);
                else
                    _sum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);

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
                _sum0 = __riscv_vle32_v_f32m1(outptr, vl);
                _sum1 = __riscv_vle32_v_f32m1(outptr + packn, vl);
                _sum2 = __riscv_vle32_v_f32m1(outptr + packn * 2, vl);
                _sum3 = __riscv_vle32_v_f32m1(outptr + packn * 3, vl);
                _sum4 = __riscv_vle32_v_f32m1(outptr + packn * 4, vl);
                _sum5 = __riscv_vle32_v_f32m1(outptr + packn * 5, vl);
                _sum6 = __riscv_vle32_v_f32m1(outptr + packn * 6, vl);
                _sum7 = __riscv_vle32_v_f32m1(outptr + packn * 7, vl);
                _sum8 = __riscv_vle32_v_f32m1(outptr + packn * 8, vl);
                _sum9 = __riscv_vle32_v_f32m1(outptr + packn * 9, vl);
                _suma = __riscv_vle32_v_f32m1(outptr + packn * 10, vl);
                _sumb = __riscv_vle32_v_f32m1(outptr + packn * 11, vl);
                _sumc = __riscv_vle32_v_f32m1(outptr + packn * 12, vl);
                _sumd = __riscv_vle32_v_f32m1(outptr + packn * 13, vl);
                _sume = __riscv_vle32_v_f32m1(outptr + packn * 14, vl);
                _sumf = __riscv_vle32_v_f32m1(outptr + packn * 15, vl);
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat32m1_t _w = __riscv_vle32_v_f32m1(pA, vl);
                _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, pB[0], _w, vl);
                _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, pB[1], _w, vl);
                _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, pB[2], _w, vl);
                _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, pB[3], _w, vl);
                _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, pB[4], _w, vl);
                _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, pB[5], _w, vl);
                _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, pB[6], _w, vl);
                _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, pB[7], _w, vl);
                _sum8 = __riscv_vfmacc_vf_f32m1(_sum8, pB[8], _w, vl);
                _sum9 = __riscv_vfmacc_vf_f32m1(_sum9, pB[9], _w, vl);
                _suma = __riscv_vfmacc_vf_f32m1(_suma, pB[10], _w, vl);
                _sumb = __riscv_vfmacc_vf_f32m1(_sumb, pB[11], _w, vl);
                _sumc = __riscv_vfmacc_vf_f32m1(_sumc, pB[12], _w, vl);
                _sumd = __riscv_vfmacc_vf_f32m1(_sumd, pB[13], _w, vl);
                _sume = __riscv_vfmacc_vf_f32m1(_sume, pB[14], _w, vl);
                _sumf = __riscv_vfmacc_vf_f32m1(_sumf, pB[15], _w, vl);

                pA += packn;
                pB += 16;
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse32_v_f32m1(outptr0, _sum0, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn, _sum1, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 2, _sum2, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 3, _sum3, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 4, _sum4, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 5, _sum5, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 6, _sum6, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 7, _sum7, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 8, _sum8, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 9, _sum9, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 10, _suma, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 11, _sumb, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 12, _sumc, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 13, _sumd, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 14, _sume, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 15, _sumf, vl);
                    outptr0 += packn * 16;
                }
                else // if (out_elempack == 1)
                {
                    __riscv_vsse32_v_f32m1(outptr0, out_hstep * sizeof(float), _sum0, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 1, out_hstep * sizeof(float), _sum1, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 2, out_hstep * sizeof(float), _sum2, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 3, out_hstep * sizeof(float), _sum3, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 4, out_hstep * sizeof(float), _sum4, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 5, out_hstep * sizeof(float), _sum5, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 6, out_hstep * sizeof(float), _sum6, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 7, out_hstep * sizeof(float), _sum7, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 8, out_hstep * sizeof(float), _sum8, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 9, out_hstep * sizeof(float), _sum9, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 10, out_hstep * sizeof(float), _suma, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 11, out_hstep * sizeof(float), _sumb, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 12, out_hstep * sizeof(float), _sumc, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 13, out_hstep * sizeof(float), _sumd, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 14, out_hstep * sizeof(float), _sume, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 15, out_hstep * sizeof(float), _sumf, vl);
                    outptr0 += 16;
                }
            }
            else
            {
                __riscv_vse32_v_f32m1(outptr, _sum0, vl);
                __riscv_vse32_v_f32m1(outptr + packn, _sum1, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 2, _sum2, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 3, _sum3, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 4, _sum4, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 5, _sum5, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 6, _sum6, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 7, _sum7, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 8, _sum8, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 9, _sum9, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 10, _suma, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 11, _sumb, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 12, _sumc, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 13, _sumd, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 14, _sume, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 15, _sumf, vl);
            }

            outptr += packn * 16;
        }

        for (; jj + 7 < max_jj; jj += 8)
        {
            const float* pA = pAT;

            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;
            vfloat32m1_t _sum2;
            vfloat32m1_t _sum3;
            vfloat32m1_t _sum4;
            vfloat32m1_t _sum5;
            vfloat32m1_t _sum6;
            vfloat32m1_t _sum7;

            if (k == 0)
            {
                if (pC)
                    _sum0 = __riscv_vle32_v_f32m1(pC, vl);
                else
                    _sum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);

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
                _sum0 = __riscv_vle32_v_f32m1(outptr, vl);
                _sum1 = __riscv_vle32_v_f32m1(outptr + packn, vl);
                _sum2 = __riscv_vle32_v_f32m1(outptr + packn * 2, vl);
                _sum3 = __riscv_vle32_v_f32m1(outptr + packn * 3, vl);
                _sum4 = __riscv_vle32_v_f32m1(outptr + packn * 4, vl);
                _sum5 = __riscv_vle32_v_f32m1(outptr + packn * 5, vl);
                _sum6 = __riscv_vle32_v_f32m1(outptr + packn * 6, vl);
                _sum7 = __riscv_vle32_v_f32m1(outptr + packn * 7, vl);
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat32m1_t _w = __riscv_vle32_v_f32m1(pA, vl);
                _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, pB[0], _w, vl);
                _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, pB[1], _w, vl);
                _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, pB[2], _w, vl);
                _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, pB[3], _w, vl);
                _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, pB[4], _w, vl);
                _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, pB[5], _w, vl);
                _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, pB[6], _w, vl);
                _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, pB[7], _w, vl);

                pA += packn;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse32_v_f32m1(outptr0, _sum0, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn, _sum1, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 2, _sum2, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 3, _sum3, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 4, _sum4, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 5, _sum5, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 6, _sum6, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 7, _sum7, vl);
                    outptr0 += packn * 8;
                }
                else // if (out_elempack == 1)
                {
                    __riscv_vsse32_v_f32m1(outptr0, out_hstep * sizeof(float), _sum0, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 1, out_hstep * sizeof(float), _sum1, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 2, out_hstep * sizeof(float), _sum2, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 3, out_hstep * sizeof(float), _sum3, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 4, out_hstep * sizeof(float), _sum4, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 5, out_hstep * sizeof(float), _sum5, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 6, out_hstep * sizeof(float), _sum6, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 7, out_hstep * sizeof(float), _sum7, vl);
                    outptr0 += 8;
                }
            }
            else
            {
                __riscv_vse32_v_f32m1(outptr, _sum0, vl);
                __riscv_vse32_v_f32m1(outptr + packn, _sum1, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 2, _sum2, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 3, _sum3, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 4, _sum4, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 5, _sum5, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 6, _sum6, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 7, _sum7, vl);
            }

            outptr += packn * 8;
        }

        for (; jj + 3 < max_jj; jj += 4)
        {
            const float* pA = pAT;

            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;
            vfloat32m1_t _sum2;
            vfloat32m1_t _sum3;

            if (k == 0)
            {
                if (pC)
                    _sum0 = __riscv_vle32_v_f32m1(pC, vl);
                else
                    _sum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);

                _sum1 = _sum0;
                _sum2 = _sum0;
                _sum3 = _sum0;
            }
            else
            {
                _sum0 = __riscv_vle32_v_f32m1(outptr, vl);
                _sum1 = __riscv_vle32_v_f32m1(outptr + packn, vl);
                _sum2 = __riscv_vle32_v_f32m1(outptr + packn * 2, vl);
                _sum3 = __riscv_vle32_v_f32m1(outptr + packn * 3, vl);
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat32m1_t _w = __riscv_vle32_v_f32m1(pA, vl);
                _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, pB[0], _w, vl);
                _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, pB[1], _w, vl);
                _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, pB[2], _w, vl);
                _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, pB[3], _w, vl);

                pA += packn;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse32_v_f32m1(outptr0, _sum0, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn, _sum1, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 2, _sum2, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn * 3, _sum3, vl);
                    outptr0 += packn * 4;
                }
                else // if (out_elempack == 1)
                {
                    __riscv_vsse32_v_f32m1(outptr0, out_hstep * sizeof(float), _sum0, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 1, out_hstep * sizeof(float), _sum1, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 2, out_hstep * sizeof(float), _sum2, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 3, out_hstep * sizeof(float), _sum3, vl);
                    outptr0 += 4;
                }
            }
            else
            {
                __riscv_vse32_v_f32m1(outptr, _sum0, vl);
                __riscv_vse32_v_f32m1(outptr + packn, _sum1, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 2, _sum2, vl);
                __riscv_vse32_v_f32m1(outptr + packn * 3, _sum3, vl);
            }

            outptr += packn * 4;
        }

        for (; jj + 1 < max_jj; jj += 2)
        {
            const float* pA = pAT;

            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;

            if (k == 0)
            {
                if (pC)
                    _sum0 = __riscv_vle32_v_f32m1(pC, vl);
                else
                    _sum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);

                _sum1 = _sum0;
            }
            else
            {
                _sum0 = __riscv_vle32_v_f32m1(outptr, vl);
                _sum1 = __riscv_vle32_v_f32m1(outptr + packn, vl);
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat32m1_t _w = __riscv_vle32_v_f32m1(pA, vl);
                _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, pB[0], _w, vl);
                _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, pB[1], _w, vl);

                pA += packn;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse32_v_f32m1(outptr0, _sum0, vl);
                    __riscv_vse32_v_f32m1(outptr0 + packn, _sum1, vl);
                    outptr0 += packn * 2;
                }
                else // if (out_elempack == 1)
                {
                    __riscv_vsse32_v_f32m1(outptr0, out_hstep * sizeof(float), _sum0, vl);
                    __riscv_vsse32_v_f32m1(outptr0 + 1, out_hstep * sizeof(float), _sum1, vl);
                    outptr0 += 2;
                }
            }
            else
            {
                __riscv_vse32_v_f32m1(outptr, _sum0, vl);
                __riscv_vse32_v_f32m1(outptr + packn, _sum1, vl);
            }

            outptr += packn * 2;
        }

        for (; jj < max_jj; jj++)
        {
            const float* pA = pAT;

            vfloat32m1_t _sum0;

            if (k == 0)
            {
                if (pC)
                    _sum0 = __riscv_vle32_v_f32m1(pC, vl);
                else
                    _sum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);

            }
            else
            {
                _sum0 = __riscv_vle32_v_f32m1(outptr, vl);
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat32m1_t _w = __riscv_vle32_v_f32m1(pA, vl);
                _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, pB[0], _w, vl);

                pA += packn;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse32_v_f32m1(outptr0, _sum0, vl);
                    outptr0 += packn;
                }
                else // if (out_elempack == 1)
                {
                    __riscv_vsse32_v_f32m1(outptr0, out_hstep * sizeof(float), _sum0, vl);
                    outptr0 += 1;
                }
            }
            else
            {
                __riscv_vse32_v_f32m1(outptr, _sum0, vl);
            }

            outptr += packn;
        }

        pAT += max_kk * packn;
    }
#else
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* outptr0 = (float*)top_blob.channel(i + ii) + j;

        const float* pB = pBT;
        const float* pC = biasptr ? biasptr + i + ii : 0;

        int jj = 0;
        for (; jj + 1 < max_jj; jj += 2)
        {
            const float* pA = pAT;

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

            for (int kk = 0; kk < max_kk; kk++)
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
                outptr0[0] = sum00;
                outptr0[1] = sum10;
                outptr0[out_hstep] = sum01;
                outptr0[out_hstep + 1] = sum11;
                outptr0 += 2;
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

        for (; jj < max_jj; jj++)
        {
            const float* pA = pAT;

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

            for (int kk = 0; kk < max_kk; kk++)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[0];

                pA += 2;
                pB += 1;
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0[out_hstep] = sum1;
                outptr0++;
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
#endif // __riscv_vector

    for (; ii < max_ii; ii++)
    {
        float* outptr0 = (float*)top_blob.channel((i + ii) / out_elempack) + j * out_elempack + (i + ii) % out_elempack;

        const float* pB = pBT;
        const float* pC = biasptr ? biasptr + i + ii : 0;

        int jj = 0;
#if __riscv_vector
        for (; jj + 15 < max_jj; jj += 16)
        {
            const float* pA = pAT;

            float sum0;
            float sum1;
            float sum2;
            float sum3;
            float sum4;
            float sum5;
            float sum6;
            float sum7;
            float sum8;
            float sum9;
            float suma;
            float sumb;
            float sumc;
            float sumd;
            float sume;
            float sumf;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[0];
                    sum2 = pC[0];
                    sum3 = pC[0];
                    sum4 = pC[0];
                    sum5 = pC[0];
                    sum6 = pC[0];
                    sum7 = pC[0];
                    sum8 = pC[0];
                    sum9 = pC[0];
                    suma = pC[0];
                    sumb = pC[0];
                    sumc = pC[0];
                    sumd = pC[0];
                    sume = pC[0];
                    sumf = pC[0];
                }
                else
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                    sum2 = 0.f;
                    sum3 = 0.f;
                    sum4 = 0.f;
                    sum5 = 0.f;
                    sum6 = 0.f;
                    sum7 = 0.f;
                    sum8 = 0.f;
                    sum9 = 0.f;
                    suma = 0.f;
                    sumb = 0.f;
                    sumc = 0.f;
                    sumd = 0.f;
                    sume = 0.f;
                    sumf = 0.f;
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
                sum2 = outptr[2];
                sum3 = outptr[3];
                sum4 = outptr[4];
                sum5 = outptr[5];
                sum6 = outptr[6];
                sum7 = outptr[7];
                sum8 = outptr[8];
                sum9 = outptr[9];
                suma = outptr[10];
                sumb = outptr[11];
                sumc = outptr[12];
                sumd = outptr[13];
                sume = outptr[14];
                sumf = outptr[15];
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                float w = pA[0];
                sum0 += w * pB[0];
                sum1 += w * pB[1];
                sum2 += w * pB[2];
                sum3 += w * pB[3];
                sum4 += w * pB[4];
                sum5 += w * pB[5];
                sum6 += w * pB[6];
                sum7 += w * pB[7];
                sum8 += w * pB[8];
                sum9 += w * pB[9];
                suma += w * pB[10];
                sumb += w * pB[11];
                sumc += w * pB[12];
                sumd += w * pB[13];
                sume += w * pB[14];
                sumf += w * pB[15];

                pA += 1;
                pB += 16;
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0[out_elempack] = sum1;
                outptr0[out_elempack * 2] = sum2;
                outptr0[out_elempack * 3] = sum3;
                outptr0[out_elempack * 4] = sum4;
                outptr0[out_elempack * 5] = sum5;
                outptr0[out_elempack * 6] = sum6;
                outptr0[out_elempack * 7] = sum7;
                outptr0[out_elempack * 8] = sum8;
                outptr0[out_elempack * 9] = sum9;
                outptr0[out_elempack * 10] = suma;
                outptr0[out_elempack * 11] = sumb;
                outptr0[out_elempack * 12] = sumc;
                outptr0[out_elempack * 13] = sumd;
                outptr0[out_elempack * 14] = sume;
                outptr0[out_elempack * 15] = sumf;
                outptr0 += out_elempack * 16;
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
                outptr[2] = sum2;
                outptr[3] = sum3;
                outptr[4] = sum4;
                outptr[5] = sum5;
                outptr[6] = sum6;
                outptr[7] = sum7;
                outptr[8] = sum8;
                outptr[9] = sum9;
                outptr[10] = suma;
                outptr[11] = sumb;
                outptr[12] = sumc;
                outptr[13] = sumd;
                outptr[14] = sume;
                outptr[15] = sumf;
            }

            outptr += 16;
        }

        for (; jj + 7 < max_jj; jj += 8)
        {
            const float* pA = pAT;

            float sum0;
            float sum1;
            float sum2;
            float sum3;
            float sum4;
            float sum5;
            float sum6;
            float sum7;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[0];
                    sum2 = pC[0];
                    sum3 = pC[0];
                    sum4 = pC[0];
                    sum5 = pC[0];
                    sum6 = pC[0];
                    sum7 = pC[0];
                }
                else
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                    sum2 = 0.f;
                    sum3 = 0.f;
                    sum4 = 0.f;
                    sum5 = 0.f;
                    sum6 = 0.f;
                    sum7 = 0.f;
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
                sum2 = outptr[2];
                sum3 = outptr[3];
                sum4 = outptr[4];
                sum5 = outptr[5];
                sum6 = outptr[6];
                sum7 = outptr[7];
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                float w = pA[0];
                sum0 += w * pB[0];
                sum1 += w * pB[1];
                sum2 += w * pB[2];
                sum3 += w * pB[3];
                sum4 += w * pB[4];
                sum5 += w * pB[5];
                sum6 += w * pB[6];
                sum7 += w * pB[7];

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0[out_elempack] = sum1;
                outptr0[out_elempack * 2] = sum2;
                outptr0[out_elempack * 3] = sum3;
                outptr0[out_elempack * 4] = sum4;
                outptr0[out_elempack * 5] = sum5;
                outptr0[out_elempack * 6] = sum6;
                outptr0[out_elempack * 7] = sum7;
                outptr0 += out_elempack * 8;
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
                outptr[2] = sum2;
                outptr[3] = sum3;
                outptr[4] = sum4;
                outptr[5] = sum5;
                outptr[6] = sum6;
                outptr[7] = sum7;
            }

            outptr += 8;
        }

        for (; jj + 3 < max_jj; jj += 4)
        {
            const float* pA = pAT;

            float sum0;
            float sum1;
            float sum2;
            float sum3;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[0];
                    sum2 = pC[0];
                    sum3 = pC[0];
                }
                else
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                    sum2 = 0.f;
                    sum3 = 0.f;
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
                sum2 = outptr[2];
                sum3 = outptr[3];
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                float w = pA[0];
                sum0 += w * pB[0];
                sum1 += w * pB[1];
                sum2 += w * pB[2];
                sum3 += w * pB[3];

                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0[out_elempack] = sum1;
                outptr0[out_elempack * 2] = sum2;
                outptr0[out_elempack * 3] = sum3;
                outptr0 += out_elempack * 4;
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
                outptr[2] = sum2;
                outptr[3] = sum3;
            }

            outptr += 4;
        }

        for (; jj + 1 < max_jj; jj += 2)
        {
            const float* pA = pAT;

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

            for (int kk = 0; kk < max_kk; kk++)
            {
                float w = pA[0];
                sum0 += w * pB[0];
                sum1 += w * pB[1];

                pA += 1;
                pB += 2;
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0[out_elempack] = sum1;
                outptr0 += out_elempack * 2;
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }

#else
        for (; jj + 1 < max_jj; jj += 2)
        {
            const float* pA = pAT;

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

            for (int kk = 0; kk < max_kk; kk++)
            {
                float w = pA[0];
                sum0 += w * pB[0];
                sum1 += w * pB[1];

                pA += 1;
                pB += 2;
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0[out_elempack] = sum1;
                outptr0 += out_elempack * 2;
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }

#endif // __riscv_vector
        for (; jj < max_jj; jj++)
        {
            const float* pA = pAT;

            float sum0;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                }
                else
                {
                    sum0 = 0.f;
                }
            }
            else
            {
                sum0 = outptr[0];
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                sum0 += pA[0] * pB[0];

                pA += 1;
                pB += 1;
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0 += out_elempack;
            }
            else
            {
                outptr[0] = sum0;
            }

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void convolution_im2col_gemm_get_optimal_tile_mnk_rvv(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    const int l2_cache_size_fp32 = (int)(get_cpu_level2_cache_size() / sizeof(float));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int packn = 1;
#if __riscv_vector
    packn = csrr_vlenb() / 4;
#endif // __riscv_vector

    // solve K
    {
        int tile_size = (l2_cache_size_fp32 - packn * 4) / (packn * 2);

        TILE_K = std::max(packn, tile_size / packn * packn);

        int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + packn - 1) / packn * packn);
    }

    // solve M
    {
        int nn_M = (M + packn * 4 - 1) / (packn * 4);

        TILE_M = std::max(packn, ((M + nn_M - 1) / nn_M + packn - 1) / packn * packn);
    }

    {
        TILE_M *= std::min(nT, get_physical_cpu_count());

        int nn_M = (M + TILE_M - 1) / TILE_M;
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + packn - 1) / packn * packn);

        if (nT > 1)
        {
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + packn - 1) / packn * packn);
        }
    }

    TILE_N = 0;
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

#if __riscv_vector
        TILE_N = std::max(16, tile_size / 16 * 16);
#else
        TILE_N = std::max(1, tile_size);
#endif

        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __riscv_vector
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 15) / 16 * 16);
        TILE_N = std::max(16, TILE_N);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
        TILE_N = std::max(1, TILE_N);
#endif
    }
}

static void convolution_im2col_input_tile_conv1x1s1d1_rvv(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    // B = (N, maxk, inch/elempack, elempack)
    float* pp = B;

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
    const int elempack = bottom_blob.elempack;
#endif // __riscv_vector

    int jj = 0;
#if __riscv_vector
    for (; jj + 15 < max_jj; jj += 16)
    {
        if (elempack == packn)
        {
            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                int p = k / elempack + kk;

                const float* sptr = (const float*)bottom_blob.channel(p) + (j + jj) * elempack;
                const int stride = elempack;
                if (packn == 4)
                {
                    const size_t vl4 = __riscv_vsetvl_e32m4(16);
                    vfloat32m4_t _val0 = __riscv_vlse32_v_f32m4(sptr + 0, stride * sizeof(float), vl4);
                    vfloat32m4_t _val1 = __riscv_vlse32_v_f32m4(sptr + 1, stride * sizeof(float), vl4);
                    vfloat32m4_t _val2 = __riscv_vlse32_v_f32m4(sptr + 2, stride * sizeof(float), vl4);
                    vfloat32m4_t _val3 = __riscv_vlse32_v_f32m4(sptr + 3, stride * sizeof(float), vl4);
                    __riscv_vse32_v_f32m4(pp, _val0, vl4);
                    __riscv_vse32_v_f32m4(pp + 1 * 16, _val1, vl4);
                    __riscv_vse32_v_f32m4(pp + 2 * 16, _val2, vl4);
                    __riscv_vse32_v_f32m4(pp + 3 * 16, _val3, vl4);
                }
                else if (packn == 8)
                {
                    const size_t vl2 = __riscv_vsetvl_e32m2(16);
                    vfloat32m2_t _val0 = __riscv_vlse32_v_f32m2(sptr + 0, stride * sizeof(float), vl2);
                    vfloat32m2_t _val1 = __riscv_vlse32_v_f32m2(sptr + 1, stride * sizeof(float), vl2);
                    vfloat32m2_t _val2 = __riscv_vlse32_v_f32m2(sptr + 2, stride * sizeof(float), vl2);
                    vfloat32m2_t _val3 = __riscv_vlse32_v_f32m2(sptr + 3, stride * sizeof(float), vl2);
                    vfloat32m2_t _val4 = __riscv_vlse32_v_f32m2(sptr + 4, stride * sizeof(float), vl2);
                    vfloat32m2_t _val5 = __riscv_vlse32_v_f32m2(sptr + 5, stride * sizeof(float), vl2);
                    vfloat32m2_t _val6 = __riscv_vlse32_v_f32m2(sptr + 6, stride * sizeof(float), vl2);
                    vfloat32m2_t _val7 = __riscv_vlse32_v_f32m2(sptr + 7, stride * sizeof(float), vl2);
                    __riscv_vse32_v_f32m2(pp, _val0, vl2);
                    __riscv_vse32_v_f32m2(pp + 1 * 16, _val1, vl2);
                    __riscv_vse32_v_f32m2(pp + 2 * 16, _val2, vl2);
                    __riscv_vse32_v_f32m2(pp + 3 * 16, _val3, vl2);
                    __riscv_vse32_v_f32m2(pp + 4 * 16, _val4, vl2);
                    __riscv_vse32_v_f32m2(pp + 5 * 16, _val5, vl2);
                    __riscv_vse32_v_f32m2(pp + 6 * 16, _val6, vl2);
                    __riscv_vse32_v_f32m2(pp + 7 * 16, _val7, vl2);
                }
                else
                {
                    for (int n = 0; n < 16; n++)
                    {
                        vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + n * stride, vl);
                        __riscv_vsse32_v_f32m1(pp + n, 16 * sizeof(float), _val, vl);
                    }
                }
                pp += elempack * 16;
            }
        }
        if (elempack == 1)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                int p = k + kk;

                const float* sptr = (const float*)bottom_blob.channel(p) + j + jj;
                int n = 0;
                while (n < 16)
                {
                    const size_t vl1 = __riscv_vsetvl_e32m1(16 - n);
                    vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + n, vl1);
                    __riscv_vse32_v_f32m1(pp + n, _val, vl1);
                    n += vl1;
                }
                pp += 16;
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == packn)
        {
            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                int p = k / elempack + kk;

                const float* sptr = (const float*)bottom_blob.channel(p) + (j + jj) * elempack;
                const int stride = elempack;
                if (packn == 4)
                {
                    const size_t vl4 = __riscv_vsetvl_e32m4(8);
                    vfloat32m4_t _val0 = __riscv_vlse32_v_f32m4(sptr + 0, stride * sizeof(float), vl4);
                    vfloat32m4_t _val1 = __riscv_vlse32_v_f32m4(sptr + 1, stride * sizeof(float), vl4);
                    vfloat32m4_t _val2 = __riscv_vlse32_v_f32m4(sptr + 2, stride * sizeof(float), vl4);
                    vfloat32m4_t _val3 = __riscv_vlse32_v_f32m4(sptr + 3, stride * sizeof(float), vl4);
                    __riscv_vse32_v_f32m4(pp, _val0, vl4);
                    __riscv_vse32_v_f32m4(pp + 1 * 8, _val1, vl4);
                    __riscv_vse32_v_f32m4(pp + 2 * 8, _val2, vl4);
                    __riscv_vse32_v_f32m4(pp + 3 * 8, _val3, vl4);
                }
                else if (packn == 8)
                {
                    const size_t vl2 = __riscv_vsetvl_e32m2(8);
                    vfloat32m2_t _val0 = __riscv_vlse32_v_f32m2(sptr + 0, stride * sizeof(float), vl2);
                    vfloat32m2_t _val1 = __riscv_vlse32_v_f32m2(sptr + 1, stride * sizeof(float), vl2);
                    vfloat32m2_t _val2 = __riscv_vlse32_v_f32m2(sptr + 2, stride * sizeof(float), vl2);
                    vfloat32m2_t _val3 = __riscv_vlse32_v_f32m2(sptr + 3, stride * sizeof(float), vl2);
                    vfloat32m2_t _val4 = __riscv_vlse32_v_f32m2(sptr + 4, stride * sizeof(float), vl2);
                    vfloat32m2_t _val5 = __riscv_vlse32_v_f32m2(sptr + 5, stride * sizeof(float), vl2);
                    vfloat32m2_t _val6 = __riscv_vlse32_v_f32m2(sptr + 6, stride * sizeof(float), vl2);
                    vfloat32m2_t _val7 = __riscv_vlse32_v_f32m2(sptr + 7, stride * sizeof(float), vl2);
                    __riscv_vse32_v_f32m2(pp, _val0, vl2);
                    __riscv_vse32_v_f32m2(pp + 1 * 8, _val1, vl2);
                    __riscv_vse32_v_f32m2(pp + 2 * 8, _val2, vl2);
                    __riscv_vse32_v_f32m2(pp + 3 * 8, _val3, vl2);
                    __riscv_vse32_v_f32m2(pp + 4 * 8, _val4, vl2);
                    __riscv_vse32_v_f32m2(pp + 5 * 8, _val5, vl2);
                    __riscv_vse32_v_f32m2(pp + 6 * 8, _val6, vl2);
                    __riscv_vse32_v_f32m2(pp + 7 * 8, _val7, vl2);
                }
                else
                {
                    for (int n = 0; n < 8; n++)
                    {
                        vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + n * stride, vl);
                        __riscv_vsse32_v_f32m1(pp + n, 8 * sizeof(float), _val, vl);
                    }
                }
                pp += elempack * 8;
            }
        }
        if (elempack == 1)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                int p = k + kk;

                const float* sptr = (const float*)bottom_blob.channel(p) + j + jj;
                int n = 0;
                while (n < 8)
                {
                    const size_t vl1 = __riscv_vsetvl_e32m1(8 - n);
                    vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + n, vl1);
                    __riscv_vse32_v_f32m1(pp + n, _val, vl1);
                    n += vl1;
                }
                pp += 8;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == packn)
        {
            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                int p = k / elempack + kk;

                const float* sptr = (const float*)bottom_blob.channel(p) + (j + jj) * elempack;
                const int stride = elempack;
                if (packn == 4)
                {
                    const size_t vl4 = __riscv_vsetvl_e32m4(4);
                    vfloat32m4_t _val0 = __riscv_vlse32_v_f32m4(sptr + 0, stride * sizeof(float), vl4);
                    vfloat32m4_t _val1 = __riscv_vlse32_v_f32m4(sptr + 1, stride * sizeof(float), vl4);
                    vfloat32m4_t _val2 = __riscv_vlse32_v_f32m4(sptr + 2, stride * sizeof(float), vl4);
                    vfloat32m4_t _val3 = __riscv_vlse32_v_f32m4(sptr + 3, stride * sizeof(float), vl4);
                    __riscv_vse32_v_f32m4(pp, _val0, vl4);
                    __riscv_vse32_v_f32m4(pp + 1 * 4, _val1, vl4);
                    __riscv_vse32_v_f32m4(pp + 2 * 4, _val2, vl4);
                    __riscv_vse32_v_f32m4(pp + 3 * 4, _val3, vl4);
                }
                else if (packn == 8)
                {
                    const size_t vl2 = __riscv_vsetvl_e32m2(4);
                    vfloat32m2_t _val0 = __riscv_vlse32_v_f32m2(sptr + 0, stride * sizeof(float), vl2);
                    vfloat32m2_t _val1 = __riscv_vlse32_v_f32m2(sptr + 1, stride * sizeof(float), vl2);
                    vfloat32m2_t _val2 = __riscv_vlse32_v_f32m2(sptr + 2, stride * sizeof(float), vl2);
                    vfloat32m2_t _val3 = __riscv_vlse32_v_f32m2(sptr + 3, stride * sizeof(float), vl2);
                    vfloat32m2_t _val4 = __riscv_vlse32_v_f32m2(sptr + 4, stride * sizeof(float), vl2);
                    vfloat32m2_t _val5 = __riscv_vlse32_v_f32m2(sptr + 5, stride * sizeof(float), vl2);
                    vfloat32m2_t _val6 = __riscv_vlse32_v_f32m2(sptr + 6, stride * sizeof(float), vl2);
                    vfloat32m2_t _val7 = __riscv_vlse32_v_f32m2(sptr + 7, stride * sizeof(float), vl2);
                    __riscv_vse32_v_f32m2(pp, _val0, vl2);
                    __riscv_vse32_v_f32m2(pp + 1 * 4, _val1, vl2);
                    __riscv_vse32_v_f32m2(pp + 2 * 4, _val2, vl2);
                    __riscv_vse32_v_f32m2(pp + 3 * 4, _val3, vl2);
                    __riscv_vse32_v_f32m2(pp + 4 * 4, _val4, vl2);
                    __riscv_vse32_v_f32m2(pp + 5 * 4, _val5, vl2);
                    __riscv_vse32_v_f32m2(pp + 6 * 4, _val6, vl2);
                    __riscv_vse32_v_f32m2(pp + 7 * 4, _val7, vl2);
                }
                else
                {
                    for (int n = 0; n < 4; n++)
                    {
                        vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + n * stride, vl);
                        __riscv_vsse32_v_f32m1(pp + n, 4 * sizeof(float), _val, vl);
                    }
                }
                pp += elempack * 4;
            }
        }
        if (elempack == 1)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                int p = k + kk;

                const float* sptr = (const float*)bottom_blob.channel(p) + j + jj;
                int n = 0;
                while (n < 4)
                {
                    const size_t vl1 = __riscv_vsetvl_e32m1(4 - n);
                    vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + n, vl1);
                    __riscv_vse32_v_f32m1(pp + n, _val, vl1);
                    n += vl1;
                }
                pp += 4;
            }
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        if (elempack == packn)
        {
            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                int p = k / elempack + kk;

                const float* sptr = (const float*)bottom_blob.channel(p) + (j + jj) * elempack;
                const int stride = elempack;
                if (packn == 4)
                {
                    const size_t vl4 = __riscv_vsetvl_e32m4(2);
                    vfloat32m4_t _val0 = __riscv_vlse32_v_f32m4(sptr + 0, stride * sizeof(float), vl4);
                    vfloat32m4_t _val1 = __riscv_vlse32_v_f32m4(sptr + 1, stride * sizeof(float), vl4);
                    vfloat32m4_t _val2 = __riscv_vlse32_v_f32m4(sptr + 2, stride * sizeof(float), vl4);
                    vfloat32m4_t _val3 = __riscv_vlse32_v_f32m4(sptr + 3, stride * sizeof(float), vl4);
                    __riscv_vse32_v_f32m4(pp, _val0, vl4);
                    __riscv_vse32_v_f32m4(pp + 1 * 2, _val1, vl4);
                    __riscv_vse32_v_f32m4(pp + 2 * 2, _val2, vl4);
                    __riscv_vse32_v_f32m4(pp + 3 * 2, _val3, vl4);
                }
                else if (packn == 8)
                {
                    const size_t vl2 = __riscv_vsetvl_e32m2(2);
                    vfloat32m2_t _val0 = __riscv_vlse32_v_f32m2(sptr + 0, stride * sizeof(float), vl2);
                    vfloat32m2_t _val1 = __riscv_vlse32_v_f32m2(sptr + 1, stride * sizeof(float), vl2);
                    vfloat32m2_t _val2 = __riscv_vlse32_v_f32m2(sptr + 2, stride * sizeof(float), vl2);
                    vfloat32m2_t _val3 = __riscv_vlse32_v_f32m2(sptr + 3, stride * sizeof(float), vl2);
                    vfloat32m2_t _val4 = __riscv_vlse32_v_f32m2(sptr + 4, stride * sizeof(float), vl2);
                    vfloat32m2_t _val5 = __riscv_vlse32_v_f32m2(sptr + 5, stride * sizeof(float), vl2);
                    vfloat32m2_t _val6 = __riscv_vlse32_v_f32m2(sptr + 6, stride * sizeof(float), vl2);
                    vfloat32m2_t _val7 = __riscv_vlse32_v_f32m2(sptr + 7, stride * sizeof(float), vl2);
                    __riscv_vse32_v_f32m2(pp, _val0, vl2);
                    __riscv_vse32_v_f32m2(pp + 1 * 2, _val1, vl2);
                    __riscv_vse32_v_f32m2(pp + 2 * 2, _val2, vl2);
                    __riscv_vse32_v_f32m2(pp + 3 * 2, _val3, vl2);
                    __riscv_vse32_v_f32m2(pp + 4 * 2, _val4, vl2);
                    __riscv_vse32_v_f32m2(pp + 5 * 2, _val5, vl2);
                    __riscv_vse32_v_f32m2(pp + 6 * 2, _val6, vl2);
                    __riscv_vse32_v_f32m2(pp + 7 * 2, _val7, vl2);
                }
                else
                {
                    for (int n = 0; n < 2; n++)
                    {
                        vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + n * stride, vl);
                        __riscv_vsse32_v_f32m1(pp + n, 2 * sizeof(float), _val, vl);
                    }
                }
                pp += elempack * 2;
            }
        }
        if (elempack == 1)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                int p = k + kk;

                const float* sptr = (const float*)bottom_blob.channel(p) + j + jj;
                int n = 0;
                while (n < 2)
                {
                    const size_t vl1 = __riscv_vsetvl_e32m1(2 - n);
                    vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + n, vl1);
                    __riscv_vse32_v_f32m1(pp + n, _val, vl1);
                    n += vl1;
                }
                pp += 2;
            }
        }
    }
#else
    for (; jj + 1 < max_jj; jj += 2)
    {
        for (int kk = 0; kk < max_kk; kk++)
        {
            int p = k + kk;

            const float* sptr = (const float*)bottom_blob.channel(p) + j + jj;
            pp[0] = sptr[0];
            pp[1] = sptr[1];
            pp += 2;
        }
    }
#endif // __riscv_vector
    for (; jj < max_jj; jj++)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                int p = k / elempack + kk;

                const float* sptr = (const float*)bottom_blob.channel(p) + (j + jj) * elempack;
                vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr, vl);
                __riscv_vse32_v_f32m1(pp, _val, vl);
                pp += elempack;
            }
        }
        if (elempack == 1)
#endif // __riscv_vector
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                int p = k + kk;

                const float* sptr = (const float*)bottom_blob.channel(p) + j + jj;
                pp[0] = sptr[0];
                pp++;
            }
        }
    }
}

static inline void convolution_im2col_input_tile_impl_rvv(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    const int w = bottom_blob.w;
#if __riscv_vector
    const int elempack = bottom_blob.elempack;
#endif // __riscv_vector

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int maxk = kernel_w * kernel_h;

    // B = (N, maxk, inch/elempack, elempack)
    float* pp = B;

    int jj = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
    for (; jj + 15 < max_jj; jj += 16)
    {
        int dy0 = (j + jj) / outw;
        int dy15 = (j + jj + 15) / outw;
        int dx0 = (j + jj) % outw;

        if (dy0 == dy15)
        {
            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const float* sptr = img.row(y0) + x0 * elempack;

                if (elempack == packn)
                {
                    const int stride = stride_w * elempack;
                    if (packn == 4)
                    {
                        const size_t vl4 = __riscv_vsetvl_e32m4(16);
                        vfloat32m4_t _val0 = __riscv_vlse32_v_f32m4(sptr + 0, stride * sizeof(float), vl4);
                        vfloat32m4_t _val1 = __riscv_vlse32_v_f32m4(sptr + 1, stride * sizeof(float), vl4);
                        vfloat32m4_t _val2 = __riscv_vlse32_v_f32m4(sptr + 2, stride * sizeof(float), vl4);
                        vfloat32m4_t _val3 = __riscv_vlse32_v_f32m4(sptr + 3, stride * sizeof(float), vl4);
                        __riscv_vse32_v_f32m4(pp, _val0, vl4);
                        __riscv_vse32_v_f32m4(pp + 1 * 16, _val1, vl4);
                        __riscv_vse32_v_f32m4(pp + 2 * 16, _val2, vl4);
                        __riscv_vse32_v_f32m4(pp + 3 * 16, _val3, vl4);
                    }
                    else if (packn == 8)
                    {
                        const size_t vl2 = __riscv_vsetvl_e32m2(16);
                        vfloat32m2_t _val0 = __riscv_vlse32_v_f32m2(sptr + 0, stride * sizeof(float), vl2);
                        vfloat32m2_t _val1 = __riscv_vlse32_v_f32m2(sptr + 1, stride * sizeof(float), vl2);
                        vfloat32m2_t _val2 = __riscv_vlse32_v_f32m2(sptr + 2, stride * sizeof(float), vl2);
                        vfloat32m2_t _val3 = __riscv_vlse32_v_f32m2(sptr + 3, stride * sizeof(float), vl2);
                        vfloat32m2_t _val4 = __riscv_vlse32_v_f32m2(sptr + 4, stride * sizeof(float), vl2);
                        vfloat32m2_t _val5 = __riscv_vlse32_v_f32m2(sptr + 5, stride * sizeof(float), vl2);
                        vfloat32m2_t _val6 = __riscv_vlse32_v_f32m2(sptr + 6, stride * sizeof(float), vl2);
                        vfloat32m2_t _val7 = __riscv_vlse32_v_f32m2(sptr + 7, stride * sizeof(float), vl2);
                        __riscv_vse32_v_f32m2(pp, _val0, vl2);
                        __riscv_vse32_v_f32m2(pp + 1 * 16, _val1, vl2);
                        __riscv_vse32_v_f32m2(pp + 2 * 16, _val2, vl2);
                        __riscv_vse32_v_f32m2(pp + 3 * 16, _val3, vl2);
                        __riscv_vse32_v_f32m2(pp + 4 * 16, _val4, vl2);
                        __riscv_vse32_v_f32m2(pp + 5 * 16, _val5, vl2);
                        __riscv_vse32_v_f32m2(pp + 6 * 16, _val6, vl2);
                        __riscv_vse32_v_f32m2(pp + 7 * 16, _val7, vl2);
                    }
                    else
                    {
                        for (int n = 0; n < 16; n++)
                        {
                            vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + n * stride, vl);
                            __riscv_vsse32_v_f32m1(pp + n, 16 * sizeof(float), _val, vl);
                        }
                    }
                    pp += elempack * 16;
                }
                if (elempack == 1)
                {
                    int n = 0;
                    while (n < 16)
                    {
                        const size_t vl1 = __riscv_vsetvl_e32m1(16 - n);
                        if (stride_w == 1)
                        {
                            vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + n, vl1);
                            __riscv_vse32_v_f32m1(pp + n, _val, vl1);
                        }
                        else
                        {
                            vfloat32m1_t _val = __riscv_vlse32_v_f32m1(sptr + n * stride_w, stride_w * sizeof(float), vl1);
                            __riscv_vse32_v_f32m1(pp + n, _val, vl1);
                        }
                        n += vl1;
                    }
                    pp += 16;
                }
            }
        }
        else
        {
            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                if (elempack == packn)
                {
                    for (int n = 0; n < 16; n++)
                    {
                        int dy = (j + jj + n) / outw;
                        int dx = (j + jj + n) % outw;
                        int x = stride_w * dx + dilation_w * v;
                        int y = stride_h * dy + dilation_h * u;

                        const float* sptr = img.row(y) + x * elempack;
                        vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr, vl);
                        __riscv_vsse32_v_f32m1(pp + n, 16 * sizeof(float), _val, vl);
                    }
                    pp += elempack * 16;
                }
                if (elempack == 1)
                {
                    const float* base = (const float*)img;
                    const int w = img.w;

                    unsigned int index[16];
                    for (int n = 0; n < 16; n++)
                    {
                        int dy = (j + jj + n) / outw;
                        int dx = (j + jj + n) % outw;
                        int x = stride_w * dx + dilation_w * v;
                        int y = stride_h * dy + dilation_h * u;
                        index[n] = (y * w + x) * sizeof(float);
                    }

                    int n = 0;
                    while (n < 16)
                    {
                        const size_t vl1 = __riscv_vsetvl_e32m1(16 - n);
                        vuint32m1_t _index = __riscv_vle32_v_u32m1(index + n, vl1);
                        vfloat32m1_t _val = __riscv_vloxei32_v_f32m1(base, _index, vl1);
                        __riscv_vse32_v_f32m1(pp + n, _val, vl1);
                        n += vl1;
                    }
                    pp += 16;
                }
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        int dy0 = (j + jj) / outw;
        int dy7 = (j + jj + 7) / outw;
        int dx0 = (j + jj) % outw;

        if (dy0 == dy7)
        {
            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const float* sptr = img.row(y0) + x0 * elempack;

                if (elempack == packn)
                {
                    const int stride = stride_w * elempack;
                    if (packn == 4)
                    {
                        const size_t vl4 = __riscv_vsetvl_e32m4(8);
                        vfloat32m4_t _val0 = __riscv_vlse32_v_f32m4(sptr + 0, stride * sizeof(float), vl4);
                        vfloat32m4_t _val1 = __riscv_vlse32_v_f32m4(sptr + 1, stride * sizeof(float), vl4);
                        vfloat32m4_t _val2 = __riscv_vlse32_v_f32m4(sptr + 2, stride * sizeof(float), vl4);
                        vfloat32m4_t _val3 = __riscv_vlse32_v_f32m4(sptr + 3, stride * sizeof(float), vl4);
                        __riscv_vse32_v_f32m4(pp, _val0, vl4);
                        __riscv_vse32_v_f32m4(pp + 1 * 8, _val1, vl4);
                        __riscv_vse32_v_f32m4(pp + 2 * 8, _val2, vl4);
                        __riscv_vse32_v_f32m4(pp + 3 * 8, _val3, vl4);
                    }
                    else if (packn == 8)
                    {
                        const size_t vl2 = __riscv_vsetvl_e32m2(8);
                        vfloat32m2_t _val0 = __riscv_vlse32_v_f32m2(sptr + 0, stride * sizeof(float), vl2);
                        vfloat32m2_t _val1 = __riscv_vlse32_v_f32m2(sptr + 1, stride * sizeof(float), vl2);
                        vfloat32m2_t _val2 = __riscv_vlse32_v_f32m2(sptr + 2, stride * sizeof(float), vl2);
                        vfloat32m2_t _val3 = __riscv_vlse32_v_f32m2(sptr + 3, stride * sizeof(float), vl2);
                        vfloat32m2_t _val4 = __riscv_vlse32_v_f32m2(sptr + 4, stride * sizeof(float), vl2);
                        vfloat32m2_t _val5 = __riscv_vlse32_v_f32m2(sptr + 5, stride * sizeof(float), vl2);
                        vfloat32m2_t _val6 = __riscv_vlse32_v_f32m2(sptr + 6, stride * sizeof(float), vl2);
                        vfloat32m2_t _val7 = __riscv_vlse32_v_f32m2(sptr + 7, stride * sizeof(float), vl2);
                        __riscv_vse32_v_f32m2(pp, _val0, vl2);
                        __riscv_vse32_v_f32m2(pp + 1 * 8, _val1, vl2);
                        __riscv_vse32_v_f32m2(pp + 2 * 8, _val2, vl2);
                        __riscv_vse32_v_f32m2(pp + 3 * 8, _val3, vl2);
                        __riscv_vse32_v_f32m2(pp + 4 * 8, _val4, vl2);
                        __riscv_vse32_v_f32m2(pp + 5 * 8, _val5, vl2);
                        __riscv_vse32_v_f32m2(pp + 6 * 8, _val6, vl2);
                        __riscv_vse32_v_f32m2(pp + 7 * 8, _val7, vl2);
                    }
                    else
                    {
                        for (int n = 0; n < 8; n++)
                        {
                            vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + n * stride, vl);
                            __riscv_vsse32_v_f32m1(pp + n, 8 * sizeof(float), _val, vl);
                        }
                    }
                    pp += elempack * 8;
                }
                if (elempack == 1)
                {
                    int n = 0;
                    while (n < 8)
                    {
                        const size_t vl1 = __riscv_vsetvl_e32m1(8 - n);
                        if (stride_w == 1)
                        {
                            vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + n, vl1);
                            __riscv_vse32_v_f32m1(pp + n, _val, vl1);
                        }
                        else
                        {
                            vfloat32m1_t _val = __riscv_vlse32_v_f32m1(sptr + n * stride_w, stride_w * sizeof(float), vl1);
                            __riscv_vse32_v_f32m1(pp + n, _val, vl1);
                        }
                        n += vl1;
                    }
                    pp += 8;
                }
            }
        }
        else
        {
            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                if (elempack == packn)
                {
                    for (int n = 0; n < 8; n++)
                    {
                        int dy = (j + jj + n) / outw;
                        int dx = (j + jj + n) % outw;
                        int x = stride_w * dx + dilation_w * v;
                        int y = stride_h * dy + dilation_h * u;

                        const float* sptr = img.row(y) + x * elempack;
                        vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr, vl);
                        __riscv_vsse32_v_f32m1(pp + n, 8 * sizeof(float), _val, vl);
                    }
                    pp += elempack * 8;
                }
                if (elempack == 1)
                {
                    const float* base = (const float*)img;
                    const int w = img.w;

                    unsigned int index[16];
                    for (int n = 0; n < 8; n++)
                    {
                        int dy = (j + jj + n) / outw;
                        int dx = (j + jj + n) % outw;
                        int x = stride_w * dx + dilation_w * v;
                        int y = stride_h * dy + dilation_h * u;
                        index[n] = (y * w + x) * sizeof(float);
                    }

                    int n = 0;
                    while (n < 8)
                    {
                        const size_t vl1 = __riscv_vsetvl_e32m1(8 - n);
                        vuint32m1_t _index = __riscv_vle32_v_u32m1(index + n, vl1);
                        vfloat32m1_t _val = __riscv_vloxei32_v_f32m1(base, _index, vl1);
                        __riscv_vse32_v_f32m1(pp + n, _val, vl1);
                        n += vl1;
                    }
                    pp += 8;
                }
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        int dy0 = (j + jj) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dx0 = (j + jj) % outw;

        if (dy0 == dy3)
        {
            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const float* sptr = img.row(y0) + x0 * elempack;

                if (elempack == packn)
                {
                    const int stride = stride_w * elempack;
                    if (packn == 4)
                    {
                        const size_t vl4 = __riscv_vsetvl_e32m4(4);
                        vfloat32m4_t _val0 = __riscv_vlse32_v_f32m4(sptr + 0, stride * sizeof(float), vl4);
                        vfloat32m4_t _val1 = __riscv_vlse32_v_f32m4(sptr + 1, stride * sizeof(float), vl4);
                        vfloat32m4_t _val2 = __riscv_vlse32_v_f32m4(sptr + 2, stride * sizeof(float), vl4);
                        vfloat32m4_t _val3 = __riscv_vlse32_v_f32m4(sptr + 3, stride * sizeof(float), vl4);
                        __riscv_vse32_v_f32m4(pp, _val0, vl4);
                        __riscv_vse32_v_f32m4(pp + 1 * 4, _val1, vl4);
                        __riscv_vse32_v_f32m4(pp + 2 * 4, _val2, vl4);
                        __riscv_vse32_v_f32m4(pp + 3 * 4, _val3, vl4);
                    }
                    else if (packn == 8)
                    {
                        const size_t vl2 = __riscv_vsetvl_e32m2(4);
                        vfloat32m2_t _val0 = __riscv_vlse32_v_f32m2(sptr + 0, stride * sizeof(float), vl2);
                        vfloat32m2_t _val1 = __riscv_vlse32_v_f32m2(sptr + 1, stride * sizeof(float), vl2);
                        vfloat32m2_t _val2 = __riscv_vlse32_v_f32m2(sptr + 2, stride * sizeof(float), vl2);
                        vfloat32m2_t _val3 = __riscv_vlse32_v_f32m2(sptr + 3, stride * sizeof(float), vl2);
                        vfloat32m2_t _val4 = __riscv_vlse32_v_f32m2(sptr + 4, stride * sizeof(float), vl2);
                        vfloat32m2_t _val5 = __riscv_vlse32_v_f32m2(sptr + 5, stride * sizeof(float), vl2);
                        vfloat32m2_t _val6 = __riscv_vlse32_v_f32m2(sptr + 6, stride * sizeof(float), vl2);
                        vfloat32m2_t _val7 = __riscv_vlse32_v_f32m2(sptr + 7, stride * sizeof(float), vl2);
                        __riscv_vse32_v_f32m2(pp, _val0, vl2);
                        __riscv_vse32_v_f32m2(pp + 1 * 4, _val1, vl2);
                        __riscv_vse32_v_f32m2(pp + 2 * 4, _val2, vl2);
                        __riscv_vse32_v_f32m2(pp + 3 * 4, _val3, vl2);
                        __riscv_vse32_v_f32m2(pp + 4 * 4, _val4, vl2);
                        __riscv_vse32_v_f32m2(pp + 5 * 4, _val5, vl2);
                        __riscv_vse32_v_f32m2(pp + 6 * 4, _val6, vl2);
                        __riscv_vse32_v_f32m2(pp + 7 * 4, _val7, vl2);
                    }
                    else
                    {
                        for (int n = 0; n < 4; n++)
                        {
                            vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + n * stride, vl);
                            __riscv_vsse32_v_f32m1(pp + n, 4 * sizeof(float), _val, vl);
                        }
                    }
                    pp += elempack * 4;
                }
                if (elempack == 1)
                {
                    int n = 0;
                    while (n < 4)
                    {
                        const size_t vl1 = __riscv_vsetvl_e32m1(4 - n);
                        if (stride_w == 1)
                        {
                            vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + n, vl1);
                            __riscv_vse32_v_f32m1(pp + n, _val, vl1);
                        }
                        else
                        {
                            vfloat32m1_t _val = __riscv_vlse32_v_f32m1(sptr + n * stride_w, stride_w * sizeof(float), vl1);
                            __riscv_vse32_v_f32m1(pp + n, _val, vl1);
                        }
                        n += vl1;
                    }
                    pp += 4;
                }
            }
        }
        else
        {
            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                if (elempack == packn)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        int dy = (j + jj + n) / outw;
                        int dx = (j + jj + n) % outw;
                        int x = stride_w * dx + dilation_w * v;
                        int y = stride_h * dy + dilation_h * u;

                        const float* sptr = img.row(y) + x * elempack;
                        vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr, vl);
                        __riscv_vsse32_v_f32m1(pp + n, 4 * sizeof(float), _val, vl);
                    }
                    pp += elempack * 4;
                }
                if (elempack == 1)
                {
                    const float* base = (const float*)img;
                    const int w = img.w;

                    unsigned int index[16];
                    for (int n = 0; n < 4; n++)
                    {
                        int dy = (j + jj + n) / outw;
                        int dx = (j + jj + n) % outw;
                        int x = stride_w * dx + dilation_w * v;
                        int y = stride_h * dy + dilation_h * u;
                        index[n] = (y * w + x) * sizeof(float);
                    }

                    int n = 0;
                    while (n < 4)
                    {
                        const size_t vl1 = __riscv_vsetvl_e32m1(4 - n);
                        vuint32m1_t _index = __riscv_vle32_v_u32m1(index + n, vl1);
                        vfloat32m1_t _val = __riscv_vloxei32_v_f32m1(base, _index, vl1);
                        __riscv_vse32_v_f32m1(pp + n, _val, vl1);
                        n += vl1;
                    }
                    pp += 4;
                }
            }
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dx0 = (j + jj) % outw;

        if (dy0 == dy1)
        {
            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const float* sptr = img.row(y0) + x0 * elempack;

                if (elempack == packn)
                {
                    const int stride = stride_w * elempack;
                    if (packn == 4)
                    {
                        const size_t vl4 = __riscv_vsetvl_e32m4(2);
                        vfloat32m4_t _val0 = __riscv_vlse32_v_f32m4(sptr + 0, stride * sizeof(float), vl4);
                        vfloat32m4_t _val1 = __riscv_vlse32_v_f32m4(sptr + 1, stride * sizeof(float), vl4);
                        vfloat32m4_t _val2 = __riscv_vlse32_v_f32m4(sptr + 2, stride * sizeof(float), vl4);
                        vfloat32m4_t _val3 = __riscv_vlse32_v_f32m4(sptr + 3, stride * sizeof(float), vl4);
                        __riscv_vse32_v_f32m4(pp, _val0, vl4);
                        __riscv_vse32_v_f32m4(pp + 1 * 2, _val1, vl4);
                        __riscv_vse32_v_f32m4(pp + 2 * 2, _val2, vl4);
                        __riscv_vse32_v_f32m4(pp + 3 * 2, _val3, vl4);
                    }
                    else if (packn == 8)
                    {
                        const size_t vl2 = __riscv_vsetvl_e32m2(2);
                        vfloat32m2_t _val0 = __riscv_vlse32_v_f32m2(sptr + 0, stride * sizeof(float), vl2);
                        vfloat32m2_t _val1 = __riscv_vlse32_v_f32m2(sptr + 1, stride * sizeof(float), vl2);
                        vfloat32m2_t _val2 = __riscv_vlse32_v_f32m2(sptr + 2, stride * sizeof(float), vl2);
                        vfloat32m2_t _val3 = __riscv_vlse32_v_f32m2(sptr + 3, stride * sizeof(float), vl2);
                        vfloat32m2_t _val4 = __riscv_vlse32_v_f32m2(sptr + 4, stride * sizeof(float), vl2);
                        vfloat32m2_t _val5 = __riscv_vlse32_v_f32m2(sptr + 5, stride * sizeof(float), vl2);
                        vfloat32m2_t _val6 = __riscv_vlse32_v_f32m2(sptr + 6, stride * sizeof(float), vl2);
                        vfloat32m2_t _val7 = __riscv_vlse32_v_f32m2(sptr + 7, stride * sizeof(float), vl2);
                        __riscv_vse32_v_f32m2(pp, _val0, vl2);
                        __riscv_vse32_v_f32m2(pp + 1 * 2, _val1, vl2);
                        __riscv_vse32_v_f32m2(pp + 2 * 2, _val2, vl2);
                        __riscv_vse32_v_f32m2(pp + 3 * 2, _val3, vl2);
                        __riscv_vse32_v_f32m2(pp + 4 * 2, _val4, vl2);
                        __riscv_vse32_v_f32m2(pp + 5 * 2, _val5, vl2);
                        __riscv_vse32_v_f32m2(pp + 6 * 2, _val6, vl2);
                        __riscv_vse32_v_f32m2(pp + 7 * 2, _val7, vl2);
                    }
                    else
                    {
                        for (int n = 0; n < 2; n++)
                        {
                            vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + n * stride, vl);
                            __riscv_vsse32_v_f32m1(pp + n, 2 * sizeof(float), _val, vl);
                        }
                    }
                    pp += elempack * 2;
                }
                if (elempack == 1)
                {
                    int n = 0;
                    while (n < 2)
                    {
                        const size_t vl1 = __riscv_vsetvl_e32m1(2 - n);
                        if (stride_w == 1)
                        {
                            vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + n, vl1);
                            __riscv_vse32_v_f32m1(pp + n, _val, vl1);
                        }
                        else
                        {
                            vfloat32m1_t _val = __riscv_vlse32_v_f32m1(sptr + n * stride_w, stride_w * sizeof(float), vl1);
                            __riscv_vse32_v_f32m1(pp + n, _val, vl1);
                        }
                        n += vl1;
                    }
                    pp += 2;
                }
            }
        }
        else
        {
            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                if (elempack == packn)
                {
                    for (int n = 0; n < 2; n++)
                    {
                        int dy = (j + jj + n) / outw;
                        int dx = (j + jj + n) % outw;
                        int x = stride_w * dx + dilation_w * v;
                        int y = stride_h * dy + dilation_h * u;

                        const float* sptr = img.row(y) + x * elempack;
                        vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr, vl);
                        __riscv_vsse32_v_f32m1(pp + n, 2 * sizeof(float), _val, vl);
                    }
                    pp += elempack * 2;
                }
                if (elempack == 1)
                {
                    const float* base = (const float*)img;
                    const int w = img.w;

                    unsigned int index[16];
                    for (int n = 0; n < 2; n++)
                    {
                        int dy = (j + jj + n) / outw;
                        int dx = (j + jj + n) % outw;
                        int x = stride_w * dx + dilation_w * v;
                        int y = stride_h * dy + dilation_h * u;
                        index[n] = (y * w + x) * sizeof(float);
                    }

                    int n = 0;
                    while (n < 2)
                    {
                        const size_t vl1 = __riscv_vsetvl_e32m1(2 - n);
                        vuint32m1_t _index = __riscv_vle32_v_u32m1(index + n, vl1);
                        vfloat32m1_t _val = __riscv_vloxei32_v_f32m1(base, _index, vl1);
                        __riscv_vse32_v_f32m1(pp + n, _val, vl1);
                        n += vl1;
                    }
                    pp += 2;
                }
            }
        }
    }
#else
    for (; jj + 1 < max_jj; jj += 2)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;

        for (int kk = 0; kk < max_kk; kk++)
        {
            int p = (k + kk) / maxk;
            int uv = (k + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x0 = stride_w * dx0 + dilation_w * v;
            int y0 = stride_h * dy0 + dilation_h * u;
            int x1 = stride_w * dx1 + dilation_w * v;
            int y1 = stride_h * dy1 + dilation_h * u;

            pp[0] = img.row(y0)[x0];
            pp[1] = img.row(y1)[x1];
            pp += 2;
        }
    }
#endif // __riscv_vector
    for (; jj < max_jj; jj++)
    {
        int dy = (j + jj) / outw;
        int dx = (j + jj) % outw;

#if __riscv_vector
        for (int kk = 0; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x = stride_w * dx + dilation_w * v;
            int y = stride_h * dy + dilation_h * u;

            const float* sptr = img.row(y) + x * elempack;

            if (elempack == packn)
            {
                vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr, vl);
                __riscv_vse32_v_f32m1(pp, _val, vl);
                pp += elempack;
            }
            if (elempack == 1)
            {
                pp[0] = sptr[0];
                pp++;
            }
        }
#else
        for (int kk = 0; kk < max_kk; kk++)
        {
            int p = (k + kk) / maxk;
            int uv = (k + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x = stride_w * dx + dilation_w * v;
            int y = stride_h * dy + dilation_h * u;

            const float* sptr = img.row(y) + x;
            pp[0] = sptr[0];
            pp++;
        }
#endif // __riscv_vector
    }
}

template<int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h>
static inline void convolution_im2col_input_tile_impl_rvv(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    convolution_im2col_input_tile_impl_rvv(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

static void convolution_im2col_input_tile_rvv(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_conv1x1s1d1_rvv(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_impl_rvv<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_impl_rvv<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_impl_rvv<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_impl_rvv<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_impl_rvv<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_impl_rvv<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    convolution_im2col_input_tile_impl_rvv(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

static void convolution_im2col_gemm_transform_kernel_rvv(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_rvv(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    int elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        int packn = csrr_vlenb() / 4;
        elempack = inch % packn == 0 ? packn : 1;
    }
#endif // __riscv_vector

#if __riscv_vector
    {
        const int packn = csrr_vlenb() / 4;

        if (packn == 4 || packn == 8)
        {
            Mat weight_data_r2 = kernel.reshape(maxk, inch, outch);

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

                    convolution_im2col_pack_A_tile_direct_rvv(weight_data_r2, AT_tile, i, max_ii, k, max_kk, maxk, elempack);
                }
            }

            return;
        }
    }
#endif // __riscv_vector

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

            convolution_im2col_pack_A_tile_rvv(A_data, AT_tile, i, max_ii, k, max_kk);
        }
    }
}

static int convolution_im2col_gemm_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
    const int maxk = kernel_w * kernel_h;

    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_rvv(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

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

        convolution_im2col_input_tile_rvv(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
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

                convolution_gemm_transB_packed_tile_rvv(AT_tile, BT_tile, bias, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end);
            }
        }
    }

    return 0;
}

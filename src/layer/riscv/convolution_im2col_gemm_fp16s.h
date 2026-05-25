// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convolution_im2col_pack_A_tile_fp16sa_rvv(const Mat& kernel, Mat& AT, int i, int max_ii, int k, int max_kk, int maxk, int K, int elempack)
{
    const float* kernel_ptr = kernel;
    __fp16* pp = AT;

    int ii = 0;
#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);

    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        const float* kptr = kernel_ptr + (i + ii) * K;

        for (int kk = 0; kk < max_kk; kk++)
        {
            const int kk_global = k + kk;
            const int p = kk_global / (maxk * elempack) * elempack + kk_global % elempack;
            const int uv = (kk_global / elempack) % maxk;

            const float* k00 = kptr + p * maxk + uv;

            vfloat32m2_t _r0 = __riscv_vlse32_v_f32m2(k00, K * sizeof(float), vl);
            __riscv_vse16_v_f16m1(pp, __riscv_vfncvt_f_f_w_f16m1(_r0, vl), vl);

            pp += packn;
        }
    }

    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* kptr0 = kernel_ptr + (i + ii) * K;
        const float* kptr1 = kernel_ptr + (i + ii + 1) * K;
        const float* kptr2 = kernel_ptr + (i + ii + 2) * K;
        const float* kptr3 = kernel_ptr + (i + ii + 3) * K;
        const float* kptr4 = kernel_ptr + (i + ii + 4) * K;
        const float* kptr5 = kernel_ptr + (i + ii + 5) * K;
        const float* kptr6 = kernel_ptr + (i + ii + 6) * K;
        const float* kptr7 = kernel_ptr + (i + ii + 7) * K;

        for (int kk = 0; kk < max_kk; kk++)
        {
            const int kk_global = k + kk;
            const int p = kk_global / (maxk * elempack) * elempack + kk_global % elempack;
            const int uv = (kk_global / elempack) % maxk;

            const int offset = p * maxk + uv;

            pp[0] = (__fp16)kptr0[offset];
            pp[1] = (__fp16)kptr1[offset];
            pp[2] = (__fp16)kptr2[offset];
            pp[3] = (__fp16)kptr3[offset];
            pp[4] = (__fp16)kptr4[offset];
            pp[5] = (__fp16)kptr5[offset];
            pp[6] = (__fp16)kptr6[offset];
            pp[7] = (__fp16)kptr7[offset];
            pp += 8;
        }
    }

    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* kptr0 = kernel_ptr + (i + ii) * K;
        const float* kptr1 = kernel_ptr + (i + ii + 1) * K;
        const float* kptr2 = kernel_ptr + (i + ii + 2) * K;
        const float* kptr3 = kernel_ptr + (i + ii + 3) * K;

        for (int kk = 0; kk < max_kk; kk++)
        {
            const int kk_global = k + kk;
            const int p = kk_global / (maxk * elempack) * elempack + kk_global % elempack;
            const int uv = (kk_global / elempack) % maxk;

            const int offset = p * maxk + uv;

            pp[0] = (__fp16)kptr0[offset];
            pp[1] = (__fp16)kptr1[offset];
            pp[2] = (__fp16)kptr2[offset];
            pp[3] = (__fp16)kptr3[offset];
            pp += 4;
        }
    }
#endif // __riscv_zvfh

    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* kptr0 = kernel_ptr + (i + ii) * K;
        const float* kptr1 = kernel_ptr + (i + ii + 1) * K;

        for (int kk = 0; kk < max_kk; kk++)
        {
            const int kk_global = k + kk;
            const int p = kk_global / (maxk * elempack) * elempack + kk_global % elempack;
            const int uv = (kk_global / elempack) % maxk;

            const int offset = p * maxk + uv;

            pp[0] = (__fp16)kptr0[offset];
            pp[1] = (__fp16)kptr1[offset];
            pp += 2;
        }
    }

    for (; ii < max_ii; ii++)
    {
        const float* kptr = kernel_ptr + (i + ii) * K;

        for (int kk = 0; kk < max_kk; kk++)
        {
            const int kk_global = k + kk;
            const int p = kk_global / (maxk * elempack) * elempack + kk_global % elempack;
            const int uv = (kk_global / elempack) % maxk;

            pp[0] = (__fp16)kptr[p * maxk + uv];
            pp++;
        }
    }
}

static void convolution_gemm_transB_packed_tile_fp16sa_rvv(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.cstep;

    const __fp16* pAT = AT_tile;
    const __fp16* pBT = BT_tile;
    const __fp16* biasptr = CT_tile;

    __fp16* outptr = topT_tile;

    int ii = 0;
#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        __fp16* outptr0;
        if (out_elempack == packn)
            outptr0 = (__fp16*)top_blob.channel((i + ii) / packn) + j * packn;
        else // if (out_elempack == 1)
            outptr0 = (__fp16*)top_blob.channel(i + ii) + j;

        const __fp16* pB = pBT;
        const __fp16* pC = biasptr ? biasptr + i + ii : 0;

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
                if (pC)
                    _sum0 = __riscv_vle16_v_f16m1(pC, vl);
                else
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

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m1_t _w = __riscv_vle16_v_f16m1(pA, vl);
                _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pB[0], _w, vl);
                _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pB[1], _w, vl);
                _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, pB[2], _w, vl);
                _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, pB[3], _w, vl);
                _sum4 = __riscv_vfmacc_vf_f16m1(_sum4, pB[4], _w, vl);
                _sum5 = __riscv_vfmacc_vf_f16m1(_sum5, pB[5], _w, vl);
                _sum6 = __riscv_vfmacc_vf_f16m1(_sum6, pB[6], _w, vl);
                _sum7 = __riscv_vfmacc_vf_f16m1(_sum7, pB[7], _w, vl);
                _sum8 = __riscv_vfmacc_vf_f16m1(_sum8, pB[8], _w, vl);
                _sum9 = __riscv_vfmacc_vf_f16m1(_sum9, pB[9], _w, vl);
                _suma = __riscv_vfmacc_vf_f16m1(_suma, pB[10], _w, vl);
                _sumb = __riscv_vfmacc_vf_f16m1(_sumb, pB[11], _w, vl);
                _sumc = __riscv_vfmacc_vf_f16m1(_sumc, pB[12], _w, vl);
                _sumd = __riscv_vfmacc_vf_f16m1(_sumd, pB[13], _w, vl);
                _sume = __riscv_vfmacc_vf_f16m1(_sume, pB[14], _w, vl);
                _sumf = __riscv_vfmacc_vf_f16m1(_sumf, pB[15], _w, vl);

                pA += packn;
                pB += 16;
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse16_v_f16m1(outptr0, _sum0, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn, _sum1, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 2, _sum2, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 3, _sum3, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 4, _sum4, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 5, _sum5, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 6, _sum6, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 7, _sum7, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 8, _sum8, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 9, _sum9, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 10, _suma, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 11, _sumb, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 12, _sumc, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 13, _sumd, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 14, _sume, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 15, _sumf, vl);
                    outptr0 += packn * 16;
                }
                else // if (out_elempack == 1)
                {
                    __riscv_vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 2, out_hstep * sizeof(__fp16), _sum2, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 3, out_hstep * sizeof(__fp16), _sum3, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 4, out_hstep * sizeof(__fp16), _sum4, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 5, out_hstep * sizeof(__fp16), _sum5, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 6, out_hstep * sizeof(__fp16), _sum6, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 7, out_hstep * sizeof(__fp16), _sum7, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 8, out_hstep * sizeof(__fp16), _sum8, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 9, out_hstep * sizeof(__fp16), _sum9, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 10, out_hstep * sizeof(__fp16), _suma, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 11, out_hstep * sizeof(__fp16), _sumb, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 12, out_hstep * sizeof(__fp16), _sumc, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 13, out_hstep * sizeof(__fp16), _sumd, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 14, out_hstep * sizeof(__fp16), _sume, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 15, out_hstep * sizeof(__fp16), _sumf, vl);
                    outptr0 += 16;
                }
            }
            else
            {
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
            }

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
                if (pC)
                    _sum0 = __riscv_vle16_v_f16m1(pC, vl);
                else
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

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m1_t _w = __riscv_vle16_v_f16m1(pA, vl);
                _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pB[0], _w, vl);
                _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pB[1], _w, vl);
                _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, pB[2], _w, vl);
                _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, pB[3], _w, vl);
                _sum4 = __riscv_vfmacc_vf_f16m1(_sum4, pB[4], _w, vl);
                _sum5 = __riscv_vfmacc_vf_f16m1(_sum5, pB[5], _w, vl);
                _sum6 = __riscv_vfmacc_vf_f16m1(_sum6, pB[6], _w, vl);
                _sum7 = __riscv_vfmacc_vf_f16m1(_sum7, pB[7], _w, vl);

                pA += packn;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse16_v_f16m1(outptr0, _sum0, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn, _sum1, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 2, _sum2, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 3, _sum3, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 4, _sum4, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 5, _sum5, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 6, _sum6, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 7, _sum7, vl);
                    outptr0 += packn * 8;
                }
                else // if (out_elempack == 1)
                {
                    __riscv_vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 2, out_hstep * sizeof(__fp16), _sum2, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 3, out_hstep * sizeof(__fp16), _sum3, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 4, out_hstep * sizeof(__fp16), _sum4, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 5, out_hstep * sizeof(__fp16), _sum5, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 6, out_hstep * sizeof(__fp16), _sum6, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 7, out_hstep * sizeof(__fp16), _sum7, vl);
                    outptr0 += 8;
                }
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl);
                __riscv_vse16_v_f16m1(outptr + packn, _sum1, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 2, _sum2, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 3, _sum3, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 4, _sum4, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 5, _sum5, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 6, _sum6, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 7, _sum7, vl);
            }

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
                if (pC)
                    _sum0 = __riscv_vle16_v_f16m1(pC, vl);
                else
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

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m1_t _w = __riscv_vle16_v_f16m1(pA, vl);
                _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pB[0], _w, vl);
                _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pB[1], _w, vl);
                _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, pB[2], _w, vl);
                _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, pB[3], _w, vl);

                pA += packn;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse16_v_f16m1(outptr0, _sum0, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn, _sum1, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 2, _sum2, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 3, _sum3, vl);
                    outptr0 += packn * 4;
                }
                else // if (out_elempack == 1)
                {
                    __riscv_vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 2, out_hstep * sizeof(__fp16), _sum2, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 3, out_hstep * sizeof(__fp16), _sum3, vl);
                    outptr0 += 4;
                }
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl);
                __riscv_vse16_v_f16m1(outptr + packn, _sum1, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 2, _sum2, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 3, _sum3, vl);
            }

            outptr += packn * 4;
        }

        for (; jj + 1 < max_jj; jj += 2)
        {
            const __fp16* pA = pAT;

            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;

            if (k == 0)
            {
                if (pC)
                    _sum0 = __riscv_vle16_v_f16m1(pC, vl);
                else
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);

                _sum1 = _sum0;
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m1(outptr, vl);
                _sum1 = __riscv_vle16_v_f16m1(outptr + packn, vl);
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m1_t _w = __riscv_vle16_v_f16m1(pA, vl);
                _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pB[0], _w, vl);
                _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pB[1], _w, vl);

                pA += packn;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse16_v_f16m1(outptr0, _sum0, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn, _sum1, vl);
                    outptr0 += packn * 2;
                }
                else // if (out_elempack == 1)
                {
                    __riscv_vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0, vl);
                    __riscv_vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1, vl);
                    outptr0 += 2;
                }
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl);
                __riscv_vse16_v_f16m1(outptr + packn, _sum1, vl);
            }

            outptr += packn * 2;
        }

        for (; jj < max_jj; jj++)
        {
            const __fp16* pA = pAT;

            vfloat16m1_t _sum0;

            if (k == 0)
            {
                if (pC)
                    _sum0 = __riscv_vle16_v_f16m1(pC, vl);
                else
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m1(outptr, vl);
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m1_t _w = __riscv_vle16_v_f16m1(pA, vl);
                _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pB[0], _w, vl);

                pA += packn;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse16_v_f16m1(outptr0, _sum0, vl);
                    outptr0 += packn;
                }
                else // if (out_elempack == 1)
                {
                    __riscv_vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0, vl);
                    outptr0 += 1;
                }
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl);
            }

            outptr += packn;
        }

        pAT += max_kk * packn;
    }

    for (; ii + 7 < max_ii; ii += 8)
    {
        __fp16* outptr0 = (__fp16*)top_blob.channel(i + ii) + j;

        const __fp16* pB = pBT;
        const __fp16* pC = biasptr ? biasptr + i + ii : 0;

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
                if (pC)
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl16);
                    _sum1 = __riscv_vfmv_v_f_f16m1(pC[1], vl16);
                    _sum2 = __riscv_vfmv_v_f_f16m1(pC[2], vl16);
                    _sum3 = __riscv_vfmv_v_f_f16m1(pC[3], vl16);
                    _sum4 = __riscv_vfmv_v_f_f16m1(pC[4], vl16);
                    _sum5 = __riscv_vfmv_v_f_f16m1(pC[5], vl16);
                    _sum6 = __riscv_vfmv_v_f_f16m1(pC[6], vl16);
                    _sum7 = __riscv_vfmv_v_f_f16m1(pC[7], vl16);
                }
                else
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
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m1(outptr, vl16);
                _sum1 = __riscv_vle16_v_f16m1(outptr + 16, vl16);
                _sum2 = __riscv_vle16_v_f16m1(outptr + 16 * 2, vl16);
                _sum3 = __riscv_vle16_v_f16m1(outptr + 16 * 3, vl16);
                _sum4 = __riscv_vle16_v_f16m1(outptr + 16 * 4, vl16);
                _sum5 = __riscv_vle16_v_f16m1(outptr + 16 * 5, vl16);
                _sum6 = __riscv_vle16_v_f16m1(outptr + 16 * 6, vl16);
                _sum7 = __riscv_vle16_v_f16m1(outptr + 16 * 7, vl16);
            }

            for (int kk = 0; kk < max_kk; kk++)
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

            if (k_end)
            {
                __riscv_vse16_v_f16m1(outptr0, _sum0, vl16);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep, _sum1, vl16);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 2, _sum2, vl16);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 3, _sum3, vl16);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 4, _sum4, vl16);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 5, _sum5, vl16);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 6, _sum6, vl16);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 7, _sum7, vl16);
                outptr0 += 16;
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl16);
                __riscv_vse16_v_f16m1(outptr + 16, _sum1, vl16);
                __riscv_vse16_v_f16m1(outptr + 16 * 2, _sum2, vl16);
                __riscv_vse16_v_f16m1(outptr + 16 * 3, _sum3, vl16);
                __riscv_vse16_v_f16m1(outptr + 16 * 4, _sum4, vl16);
                __riscv_vse16_v_f16m1(outptr + 16 * 5, _sum5, vl16);
                __riscv_vse16_v_f16m1(outptr + 16 * 6, _sum6, vl16);
                __riscv_vse16_v_f16m1(outptr + 16 * 7, _sum7, vl16);
            }

            outptr += 128;
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
                if (pC)
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl8);
                    _sum1 = __riscv_vfmv_v_f_f16m1(pC[1], vl8);
                    _sum2 = __riscv_vfmv_v_f_f16m1(pC[2], vl8);
                    _sum3 = __riscv_vfmv_v_f_f16m1(pC[3], vl8);
                    _sum4 = __riscv_vfmv_v_f_f16m1(pC[4], vl8);
                    _sum5 = __riscv_vfmv_v_f_f16m1(pC[5], vl8);
                    _sum6 = __riscv_vfmv_v_f_f16m1(pC[6], vl8);
                    _sum7 = __riscv_vfmv_v_f_f16m1(pC[7], vl8);
                }
                else
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
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m1(outptr, vl8);
                _sum1 = __riscv_vle16_v_f16m1(outptr + 8, vl8);
                _sum2 = __riscv_vle16_v_f16m1(outptr + 8 * 2, vl8);
                _sum3 = __riscv_vle16_v_f16m1(outptr + 8 * 3, vl8);
                _sum4 = __riscv_vle16_v_f16m1(outptr + 8 * 4, vl8);
                _sum5 = __riscv_vle16_v_f16m1(outptr + 8 * 5, vl8);
                _sum6 = __riscv_vle16_v_f16m1(outptr + 8 * 6, vl8);
                _sum7 = __riscv_vle16_v_f16m1(outptr + 8 * 7, vl8);
            }

            for (int kk = 0; kk < max_kk; kk++)
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

            if (k_end)
            {
                __riscv_vse16_v_f16m1(outptr0, _sum0, vl8);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep, _sum1, vl8);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 2, _sum2, vl8);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 3, _sum3, vl8);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 4, _sum4, vl8);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 5, _sum5, vl8);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 6, _sum6, vl8);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 7, _sum7, vl8);
                outptr0 += 8;
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl8);
                __riscv_vse16_v_f16m1(outptr + 8, _sum1, vl8);
                __riscv_vse16_v_f16m1(outptr + 8 * 2, _sum2, vl8);
                __riscv_vse16_v_f16m1(outptr + 8 * 3, _sum3, vl8);
                __riscv_vse16_v_f16m1(outptr + 8 * 4, _sum4, vl8);
                __riscv_vse16_v_f16m1(outptr + 8 * 5, _sum5, vl8);
                __riscv_vse16_v_f16m1(outptr + 8 * 6, _sum6, vl8);
                __riscv_vse16_v_f16m1(outptr + 8 * 7, _sum7, vl8);
            }

            outptr += 64;
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
                if (pC)
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl4);
                    _sum1 = __riscv_vfmv_v_f_f16m1(pC[1], vl4);
                    _sum2 = __riscv_vfmv_v_f_f16m1(pC[2], vl4);
                    _sum3 = __riscv_vfmv_v_f_f16m1(pC[3], vl4);
                    _sum4 = __riscv_vfmv_v_f_f16m1(pC[4], vl4);
                    _sum5 = __riscv_vfmv_v_f_f16m1(pC[5], vl4);
                    _sum6 = __riscv_vfmv_v_f_f16m1(pC[6], vl4);
                    _sum7 = __riscv_vfmv_v_f_f16m1(pC[7], vl4);
                }
                else
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
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m1(outptr, vl4);
                _sum1 = __riscv_vle16_v_f16m1(outptr + 4, vl4);
                _sum2 = __riscv_vle16_v_f16m1(outptr + 4 * 2, vl4);
                _sum3 = __riscv_vle16_v_f16m1(outptr + 4 * 3, vl4);
                _sum4 = __riscv_vle16_v_f16m1(outptr + 4 * 4, vl4);
                _sum5 = __riscv_vle16_v_f16m1(outptr + 4 * 5, vl4);
                _sum6 = __riscv_vle16_v_f16m1(outptr + 4 * 6, vl4);
                _sum7 = __riscv_vle16_v_f16m1(outptr + 4 * 7, vl4);
            }

            for (int kk = 0; kk < max_kk; kk++)
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

            if (k_end)
            {
                __riscv_vse16_v_f16m1(outptr0, _sum0, vl4);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep, _sum1, vl4);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 2, _sum2, vl4);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 3, _sum3, vl4);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 4, _sum4, vl4);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 5, _sum5, vl4);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 6, _sum6, vl4);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 7, _sum7, vl4);
                outptr0 += 4;
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl4);
                __riscv_vse16_v_f16m1(outptr + 4, _sum1, vl4);
                __riscv_vse16_v_f16m1(outptr + 4 * 2, _sum2, vl4);
                __riscv_vse16_v_f16m1(outptr + 4 * 3, _sum3, vl4);
                __riscv_vse16_v_f16m1(outptr + 4 * 4, _sum4, vl4);
                __riscv_vse16_v_f16m1(outptr + 4 * 5, _sum5, vl4);
                __riscv_vse16_v_f16m1(outptr + 4 * 6, _sum6, vl4);
                __riscv_vse16_v_f16m1(outptr + 4 * 7, _sum7, vl4);
            }

            outptr += 32;
        }

        for (; jj + 1 < max_jj; jj += 2)
        {
            const __fp16* pA = pAT;

            __fp16 sum00;
            __fp16 sum01;
            __fp16 sum02;
            __fp16 sum03;
            __fp16 sum04;
            __fp16 sum05;
            __fp16 sum06;
            __fp16 sum07;
            __fp16 sum10;
            __fp16 sum11;
            __fp16 sum12;
            __fp16 sum13;
            __fp16 sum14;
            __fp16 sum15;
            __fp16 sum16;
            __fp16 sum17;

            if (k == 0)
            {
                if (pC)
                {
                    sum00 = pC[0];
                    sum01 = pC[1];
                    sum02 = pC[2];
                    sum03 = pC[3];
                    sum04 = pC[4];
                    sum05 = pC[5];
                    sum06 = pC[6];
                    sum07 = pC[7];
                    sum10 = pC[0];
                    sum11 = pC[1];
                    sum12 = pC[2];
                    sum13 = pC[3];
                    sum14 = pC[4];
                    sum15 = pC[5];
                    sum16 = pC[6];
                    sum17 = pC[7];
                }
                else
                {
                    sum00 = (__fp16)0.f;
                    sum01 = (__fp16)0.f;
                    sum02 = (__fp16)0.f;
                    sum03 = (__fp16)0.f;
                    sum04 = (__fp16)0.f;
                    sum05 = (__fp16)0.f;
                    sum06 = (__fp16)0.f;
                    sum07 = (__fp16)0.f;
                    sum10 = (__fp16)0.f;
                    sum11 = (__fp16)0.f;
                    sum12 = (__fp16)0.f;
                    sum13 = (__fp16)0.f;
                    sum14 = (__fp16)0.f;
                    sum15 = (__fp16)0.f;
                    sum16 = (__fp16)0.f;
                    sum17 = (__fp16)0.f;
                }
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum02 = outptr[2];
                sum03 = outptr[3];
                sum04 = outptr[4];
                sum05 = outptr[5];
                sum06 = outptr[6];
                sum07 = outptr[7];
                sum10 = outptr[8];
                sum11 = outptr[9];
                sum12 = outptr[10];
                sum13 = outptr[11];
                sum14 = outptr[12];
                sum15 = outptr[13];
                sum16 = outptr[14];
                sum17 = outptr[15];
            }

            for (int kk = 0; kk < max_kk; kk++)
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

            if (k_end)
            {
                outptr0[0] = sum00;
                outptr0[1] = sum10;
                outptr0[out_hstep] = sum01;
                outptr0[out_hstep + 1] = sum11;
                outptr0[out_hstep * 2] = sum02;
                outptr0[out_hstep * 2 + 1] = sum12;
                outptr0[out_hstep * 3] = sum03;
                outptr0[out_hstep * 3 + 1] = sum13;
                outptr0[out_hstep * 4] = sum04;
                outptr0[out_hstep * 4 + 1] = sum14;
                outptr0[out_hstep * 5] = sum05;
                outptr0[out_hstep * 5 + 1] = sum15;
                outptr0[out_hstep * 6] = sum06;
                outptr0[out_hstep * 6 + 1] = sum16;
                outptr0[out_hstep * 7] = sum07;
                outptr0[out_hstep * 7 + 1] = sum17;
                outptr0 += 2;
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum01;
                outptr[2] = sum02;
                outptr[3] = sum03;
                outptr[4] = sum04;
                outptr[5] = sum05;
                outptr[6] = sum06;
                outptr[7] = sum07;
                outptr[8] = sum10;
                outptr[9] = sum11;
                outptr[10] = sum12;
                outptr[11] = sum13;
                outptr[12] = sum14;
                outptr[13] = sum15;
                outptr[14] = sum16;
                outptr[15] = sum17;
            }

            outptr += 16;
        }

        for (; jj < max_jj; jj++)
        {
            const __fp16* pA = pAT;

            __fp16 sum0;
            __fp16 sum1;
            __fp16 sum2;
            __fp16 sum3;
            __fp16 sum4;
            __fp16 sum5;
            __fp16 sum6;
            __fp16 sum7;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[1];
                    sum2 = pC[2];
                    sum3 = pC[3];
                    sum4 = pC[4];
                    sum5 = pC[5];
                    sum6 = pC[6];
                    sum7 = pC[7];
                }
                else
                {
                    sum0 = (__fp16)0.f;
                    sum1 = (__fp16)0.f;
                    sum2 = (__fp16)0.f;
                    sum3 = (__fp16)0.f;
                    sum4 = (__fp16)0.f;
                    sum5 = (__fp16)0.f;
                    sum6 = (__fp16)0.f;
                    sum7 = (__fp16)0.f;
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

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0[out_hstep] = sum1;
                outptr0[out_hstep * 2] = sum2;
                outptr0[out_hstep * 3] = sum3;
                outptr0[out_hstep * 4] = sum4;
                outptr0[out_hstep * 5] = sum5;
                outptr0[out_hstep * 6] = sum6;
                outptr0[out_hstep * 7] = sum7;
                outptr0++;
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

        pAT += max_kk * 8;
    }

    for (; ii + 3 < max_ii; ii += 4)
    {
        __fp16* outptr0 = (__fp16*)top_blob.channel(i + ii) + j;

        const __fp16* pB = pBT;
        const __fp16* pC = biasptr ? biasptr + i + ii : 0;

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
                    if (pC)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m2(pC[0], vl16);
                        _sum1 = __riscv_vfmv_v_f_f16m2(pC[1], vl16);
                        _sum2 = __riscv_vfmv_v_f_f16m2(pC[2], vl16);
                        _sum3 = __riscv_vfmv_v_f_f16m2(pC[3], vl16);
                    }
                    else
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m2((__fp16)0.f, vl16);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                }
                else
                {
                    _sum0 = __riscv_vle16_v_f16m2(outptr, vl16);
                    _sum1 = __riscv_vle16_v_f16m2(outptr + 16, vl16);
                    _sum2 = __riscv_vle16_v_f16m2(outptr + 16 * 2, vl16);
                    _sum3 = __riscv_vle16_v_f16m2(outptr + 16 * 3, vl16);
                }

                for (int kk = 0; kk < max_kk; kk++)
                {
                    vfloat16m2_t _val = __riscv_vle16_v_f16m2(pB, vl16);
                    _sum0 = __riscv_vfmacc_vf_f16m2(_sum0, pA[0], _val, vl16);
                    _sum1 = __riscv_vfmacc_vf_f16m2(_sum1, pA[1], _val, vl16);
                    _sum2 = __riscv_vfmacc_vf_f16m2(_sum2, pA[2], _val, vl16);
                    _sum3 = __riscv_vfmacc_vf_f16m2(_sum3, pA[3], _val, vl16);

                    pA += 4;
                    pB += 16;
                }

                if (k_end)
                {
                    __riscv_vse16_v_f16m2(outptr0, _sum0, vl16);
                    __riscv_vse16_v_f16m2(outptr0 + out_hstep, _sum1, vl16);
                    __riscv_vse16_v_f16m2(outptr0 + out_hstep * 2, _sum2, vl16);
                    __riscv_vse16_v_f16m2(outptr0 + out_hstep * 3, _sum3, vl16);
                    outptr0 += 16;
                }
                else
                {
                    __riscv_vse16_v_f16m2(outptr, _sum0, vl16);
                    __riscv_vse16_v_f16m2(outptr + 16, _sum1, vl16);
                    __riscv_vse16_v_f16m2(outptr + 16 * 2, _sum2, vl16);
                    __riscv_vse16_v_f16m2(outptr + 16 * 3, _sum3, vl16);
                }
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
                    if (pC)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl16);
                        _sum1 = __riscv_vfmv_v_f_f16m1(pC[1], vl16);
                        _sum2 = __riscv_vfmv_v_f_f16m1(pC[2], vl16);
                        _sum3 = __riscv_vfmv_v_f_f16m1(pC[3], vl16);
                    }
                    else
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl16);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                }
                else
                {
                    _sum0 = __riscv_vle16_v_f16m1(outptr, vl16);
                    _sum1 = __riscv_vle16_v_f16m1(outptr + 16, vl16);
                    _sum2 = __riscv_vle16_v_f16m1(outptr + 16 * 2, vl16);
                    _sum3 = __riscv_vle16_v_f16m1(outptr + 16 * 3, vl16);
                }

                for (int kk = 0; kk < max_kk; kk++)
                {
                    vfloat16m1_t _val = __riscv_vle16_v_f16m1(pB, vl16);
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], _val, vl16);
                    _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pA[1], _val, vl16);
                    _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, pA[2], _val, vl16);
                    _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, pA[3], _val, vl16);

                    pA += 4;
                    pB += 16;
                }

                if (k_end)
                {
#if defined(__GNUC__) && !defined(__clang__)
                    // gcc may emit wrong writeback for this packn=16 4x16 tail.
                    // Keep a live stack slot to avoid that codegen pattern.
                    __fp16 tmp;
                    __asm__ volatile(""
                                     :
                                     : "r"(&tmp)
                                     : "memory");
#endif

                    __riscv_vse16_v_f16m1(outptr0, _sum0, vl16);
                    __riscv_vse16_v_f16m1(outptr0 + out_hstep, _sum1, vl16);
                    __riscv_vse16_v_f16m1(outptr0 + out_hstep * 2, _sum2, vl16);
                    __riscv_vse16_v_f16m1(outptr0 + out_hstep * 3, _sum3, vl16);
                    outptr0 += 16;
                }
                else
                {
                    __riscv_vse16_v_f16m1(outptr, _sum0, vl16);
                    __riscv_vse16_v_f16m1(outptr + 16, _sum1, vl16);
                    __riscv_vse16_v_f16m1(outptr + 16 * 2, _sum2, vl16);
                    __riscv_vse16_v_f16m1(outptr + 16 * 3, _sum3, vl16);
                }
            }

            outptr += 64;
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
                if (pC)
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl8);
                    _sum1 = __riscv_vfmv_v_f_f16m1(pC[1], vl8);
                    _sum2 = __riscv_vfmv_v_f_f16m1(pC[2], vl8);
                    _sum3 = __riscv_vfmv_v_f_f16m1(pC[3], vl8);
                }
                else
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl8);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m1(outptr, vl8);
                _sum1 = __riscv_vle16_v_f16m1(outptr + 8, vl8);
                _sum2 = __riscv_vle16_v_f16m1(outptr + 8 * 2, vl8);
                _sum3 = __riscv_vle16_v_f16m1(outptr + 8 * 3, vl8);
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m1_t _val = __riscv_vle16_v_f16m1(pB, vl8);
                _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], _val, vl8);
                _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pA[1], _val, vl8);
                _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, pA[2], _val, vl8);
                _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, pA[3], _val, vl8);

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                __riscv_vse16_v_f16m1(outptr0, _sum0, vl8);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep, _sum1, vl8);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 2, _sum2, vl8);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 3, _sum3, vl8);
                outptr0 += 8;
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl8);
                __riscv_vse16_v_f16m1(outptr + 8, _sum1, vl8);
                __riscv_vse16_v_f16m1(outptr + 8 * 2, _sum2, vl8);
                __riscv_vse16_v_f16m1(outptr + 8 * 3, _sum3, vl8);
            }

            outptr += 32;
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
                if (pC)
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl4);
                    _sum1 = __riscv_vfmv_v_f_f16m1(pC[1], vl4);
                    _sum2 = __riscv_vfmv_v_f_f16m1(pC[2], vl4);
                    _sum3 = __riscv_vfmv_v_f_f16m1(pC[3], vl4);
                }
                else
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl4);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m1(outptr, vl4);
                _sum1 = __riscv_vle16_v_f16m1(outptr + 4, vl4);
                _sum2 = __riscv_vle16_v_f16m1(outptr + 4 * 2, vl4);
                _sum3 = __riscv_vle16_v_f16m1(outptr + 4 * 3, vl4);
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m1_t _val = __riscv_vle16_v_f16m1(pB, vl4);
                _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], _val, vl4);
                _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pA[1], _val, vl4);
                _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, pA[2], _val, vl4);
                _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, pA[3], _val, vl4);

                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                __riscv_vse16_v_f16m1(outptr0, _sum0, vl4);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep, _sum1, vl4);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 2, _sum2, vl4);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep * 3, _sum3, vl4);
                outptr0 += 4;
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl4);
                __riscv_vse16_v_f16m1(outptr + 4, _sum1, vl4);
                __riscv_vse16_v_f16m1(outptr + 4 * 2, _sum2, vl4);
                __riscv_vse16_v_f16m1(outptr + 4 * 3, _sum3, vl4);
            }

            outptr += 16;
        }

        for (; jj + 1 < max_jj; jj += 2)
        {
            const __fp16* pA = pAT;

            __fp16 sum00;
            __fp16 sum01;
            __fp16 sum02;
            __fp16 sum03;
            __fp16 sum10;
            __fp16 sum11;
            __fp16 sum12;
            __fp16 sum13;

            if (k == 0)
            {
                if (pC)
                {
                    sum00 = pC[0];
                    sum01 = pC[1];
                    sum02 = pC[2];
                    sum03 = pC[3];
                    sum10 = pC[0];
                    sum11 = pC[1];
                    sum12 = pC[2];
                    sum13 = pC[3];
                }
                else
                {
                    sum00 = (__fp16)0.f;
                    sum01 = (__fp16)0.f;
                    sum02 = (__fp16)0.f;
                    sum03 = (__fp16)0.f;
                    sum10 = (__fp16)0.f;
                    sum11 = (__fp16)0.f;
                    sum12 = (__fp16)0.f;
                    sum13 = (__fp16)0.f;
                }
            }
            else
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

            for (int kk = 0; kk < max_kk; kk++)
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

            if (k_end)
            {
                outptr0[0] = sum00;
                outptr0[1] = sum10;
                outptr0[out_hstep] = sum01;
                outptr0[out_hstep + 1] = sum11;
                outptr0[out_hstep * 2] = sum02;
                outptr0[out_hstep * 2 + 1] = sum12;
                outptr0[out_hstep * 3] = sum03;
                outptr0[out_hstep * 3 + 1] = sum13;
                outptr0 += 2;
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum01;
                outptr[2] = sum02;
                outptr[3] = sum03;
                outptr[4] = sum10;
                outptr[5] = sum11;
                outptr[6] = sum12;
                outptr[7] = sum13;
            }

            outptr += 8;
        }

        for (; jj < max_jj; jj++)
        {
            const __fp16* pA = pAT;

            __fp16 sum0;
            __fp16 sum1;
            __fp16 sum2;
            __fp16 sum3;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[1];
                    sum2 = pC[2];
                    sum3 = pC[3];
                }
                else
                {
                    sum0 = (__fp16)0.f;
                    sum1 = (__fp16)0.f;
                    sum2 = (__fp16)0.f;
                    sum3 = (__fp16)0.f;
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
                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[0];
                sum2 += pA[2] * pB[0];
                sum3 += pA[3] * pB[0];

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0[out_hstep] = sum1;
                outptr0[out_hstep * 2] = sum2;
                outptr0[out_hstep * 3] = sum3;
                outptr0++;
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

        pAT += max_kk * 4;
    }
#endif // __riscv_zvfh

    for (; ii + 1 < max_ii; ii += 2)
    {
        __fp16* outptr0 = (__fp16*)top_blob.channel(i + ii) + j;

        const __fp16* pB = pBT;
        const __fp16* pC = biasptr ? biasptr + i + ii : 0;

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
                    if (pC)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m2(pC[0], vl16);
                        _sum1 = __riscv_vfmv_v_f_f16m2(pC[1], vl16);
                    }
                    else
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m2((__fp16)0.f, vl16);
                        _sum1 = __riscv_vfmv_v_f_f16m2((__fp16)0.f, vl16);
                    }
                }
                else
                {
                    _sum0 = __riscv_vle16_v_f16m2(outptr, vl16);
                    _sum1 = __riscv_vle16_v_f16m2(outptr + 16, vl16);
                }

                for (int kk = 0; kk < max_kk; kk++)
                {
                    vfloat16m2_t _val = __riscv_vle16_v_f16m2(pB, vl16);
                    _sum0 = __riscv_vfmacc_vf_f16m2(_sum0, pA[0], _val, vl16);
                    _sum1 = __riscv_vfmacc_vf_f16m2(_sum1, pA[1], _val, vl16);

                    pA += 2;
                    pB += 16;
                }

                if (k_end)
                {
                    __riscv_vse16_v_f16m2(outptr0, _sum0, vl16);
                    __riscv_vse16_v_f16m2(outptr0 + out_hstep, _sum1, vl16);
                    outptr0 += 16;
                }
                else
                {
                    __riscv_vse16_v_f16m2(outptr, _sum0, vl16);
                    __riscv_vse16_v_f16m2(outptr + 16, _sum1, vl16);
                }
            }
            else
            {
                const size_t vl16 = __riscv_vsetvl_e16m1(16);

                vfloat16m1_t _sum0;
                vfloat16m1_t _sum1;

                if (k == 0)
                {
                    if (pC)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl16);
                        _sum1 = __riscv_vfmv_v_f_f16m1(pC[1], vl16);
                    }
                    else
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl16);
                        _sum1 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl16);
                    }
                }
                else
                {
                    _sum0 = __riscv_vle16_v_f16m1(outptr, vl16);
                    _sum1 = __riscv_vle16_v_f16m1(outptr + 16, vl16);
                }

                for (int kk = 0; kk < max_kk; kk++)
                {
                    vfloat16m1_t _val = __riscv_vle16_v_f16m1(pB, vl16);
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], _val, vl16);
                    _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pA[1], _val, vl16);

                    pA += 2;
                    pB += 16;
                }

                if (k_end)
                {
                    __riscv_vse16_v_f16m1(outptr0, _sum0, vl16);
                    __riscv_vse16_v_f16m1(outptr0 + out_hstep, _sum1, vl16);
                    outptr0 += 16;
                }
                else
                {
                    __riscv_vse16_v_f16m1(outptr, _sum0, vl16);
                    __riscv_vse16_v_f16m1(outptr + 16, _sum1, vl16);
                }
            }

            outptr += 32;
        }

        for (; jj + 7 < max_jj; jj += 8)
        {
            const __fp16* pA = pAT;

            const size_t vl8 = __riscv_vsetvl_e16m1(8);

            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl8);
                    _sum1 = __riscv_vfmv_v_f_f16m1(pC[1], vl8);
                }
                else
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl8);
                    _sum1 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl8);
                }
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m1(outptr, vl8);
                _sum1 = __riscv_vle16_v_f16m1(outptr + 8, vl8);
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m1_t _val = __riscv_vle16_v_f16m1(pB, vl8);
                _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], _val, vl8);
                _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pA[1], _val, vl8);

                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                __riscv_vse16_v_f16m1(outptr0, _sum0, vl8);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep, _sum1, vl8);
                outptr0 += 8;
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl8);
                __riscv_vse16_v_f16m1(outptr + 8, _sum1, vl8);
            }

            outptr += 16;
        }

        for (; jj + 3 < max_jj; jj += 4)
        {
            const __fp16* pA = pAT;

            const size_t vl4 = __riscv_vsetvl_e16m1(4);

            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl4);
                    _sum1 = __riscv_vfmv_v_f_f16m1(pC[1], vl4);
                }
                else
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl4);
                    _sum1 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl4);
                }
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m1(outptr, vl4);
                _sum1 = __riscv_vle16_v_f16m1(outptr + 4, vl4);
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m1_t _val = __riscv_vle16_v_f16m1(pB, vl4);
                _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], _val, vl4);
                _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pA[1], _val, vl4);

                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                __riscv_vse16_v_f16m1(outptr0, _sum0, vl4);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep, _sum1, vl4);
                outptr0 += 4;
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl4);
                __riscv_vse16_v_f16m1(outptr + 4, _sum1, vl4);
            }

            outptr += 8;
        }

#endif // __riscv_zvfh
        for (; jj + 1 < max_jj; jj += 2)
        {
            const __fp16* pA = pAT;

            __fp16 sum00;
            __fp16 sum01;
            __fp16 sum10;
            __fp16 sum11;

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
                    sum00 = (__fp16)0.f;
                    sum01 = (__fp16)0.f;
                    sum10 = (__fp16)0.f;
                    sum11 = (__fp16)0.f;
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
            const __fp16* pA = pAT;

            __fp16 sum0;
            __fp16 sum1;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[1];
                }
                else
                {
                    sum0 = (__fp16)0.f;
                    sum1 = (__fp16)0.f;
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

    for (; ii < max_ii; ii++)
    {
        __fp16* outptr0 = (__fp16*)top_blob.channel((i + ii) / out_elempack) + j * out_elempack + (i + ii) % out_elempack;

        const __fp16* pB = pBT;
        const __fp16* pC = biasptr ? biasptr + i + ii : 0;

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
                {
                    if (pC)
                        _sum0 = __riscv_vfmv_v_f_f16m2(pC[0], vl16);
                    else
                        _sum0 = __riscv_vfmv_v_f_f16m2((__fp16)0.f, vl16);
                }
                else
                {
                    _sum0 = __riscv_vle16_v_f16m2(outptr, vl16);
                }

                for (int kk = 0; kk < max_kk; kk++)
                {
                    _sum0 = __riscv_vfmacc_vf_f16m2(_sum0, pA[0], __riscv_vle16_v_f16m2(pB, vl16), vl16);

                    pA += 1;
                    pB += 16;
                }

                if (k_end)
                {
                    if (out_elempack == 1)
                        __riscv_vse16_v_f16m2(outptr0, _sum0, vl16);
                    else
                        __riscv_vsse16_v_f16m2(outptr0, out_elempack * sizeof(__fp16), _sum0, vl16);

                    outptr0 += out_elempack * 16;
                }
                else
                {
                    __riscv_vse16_v_f16m2(outptr, _sum0, vl16);
                }
            }
            else
            {
                const size_t vl16 = __riscv_vsetvl_e16m1(16);

                vfloat16m1_t _sum0;

                if (k == 0)
                {
                    if (pC)
                        _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl16);
                    else
                        _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl16);
                }
                else
                {
                    _sum0 = __riscv_vle16_v_f16m1(outptr, vl16);
                }

                for (int kk = 0; kk < max_kk; kk++)
                {
                    _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], __riscv_vle16_v_f16m1(pB, vl16), vl16);

                    pA += 1;
                    pB += 16;
                }

                if (k_end)
                {
                    if (out_elempack == 1)
                        __riscv_vse16_v_f16m1(outptr0, _sum0, vl16);
                    else
                        __riscv_vsse16_v_f16m1(outptr0, out_elempack * sizeof(__fp16), _sum0, vl16);

                    outptr0 += out_elempack * 16;
                }
                else
                {
                    __riscv_vse16_v_f16m1(outptr, _sum0, vl16);
                }
            }

            outptr += 16;
        }

        for (; jj + 7 < max_jj; jj += 8)
        {
            const __fp16* pA = pAT;

            const size_t vl8 = __riscv_vsetvl_e16m1(8);

            vfloat16m1_t _sum0;

            if (k == 0)
            {
                if (pC)
                    _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl8);
                else
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl8);
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m1(outptr, vl8);
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], __riscv_vle16_v_f16m1(pB, vl8), vl8);

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 1)
                    __riscv_vse16_v_f16m1(outptr0, _sum0, vl8);
                else
                    __riscv_vsse16_v_f16m1(outptr0, out_elempack * sizeof(__fp16), _sum0, vl8);

                outptr0 += out_elempack * 8;
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl8);
            }

            outptr += 8;
        }

        for (; jj + 3 < max_jj; jj += 4)
        {
            const __fp16* pA = pAT;

            const size_t vl4 = __riscv_vsetvl_e16m1(4);

            vfloat16m1_t _sum0;

            if (k == 0)
            {
                if (pC)
                    _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl4);
                else
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl4);
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m1(outptr, vl4);
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], __riscv_vle16_v_f16m1(pB, vl4), vl4);

                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 1)
                    __riscv_vse16_v_f16m1(outptr0, _sum0, vl4);
                else
                    __riscv_vsse16_v_f16m1(outptr0, out_elempack * sizeof(__fp16), _sum0, vl4);

                outptr0 += out_elempack * 4;
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl4);
            }

            outptr += 4;
        }

#endif // __riscv_zvfh
        for (; jj + 1 < max_jj; jj += 2)
        {
            const __fp16* pA = pAT;

            __fp16 sum0;
            __fp16 sum1;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[0];
                }
                else
                {
                    sum0 = (__fp16)0.f;
                    sum1 = (__fp16)0.f;
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                __fp16 w = pA[0];
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

        for (; jj < max_jj; jj++)
        {
            const __fp16* pA = pAT;

            __fp16 sum0;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                }
                else
                {
                    sum0 = (__fp16)0.f;
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

static void convolution_im2col_gemm_get_optimal_tile_mnk_fp16sa_rvv(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    const int l2_cache_size_fp16 = (int)(get_cpu_level2_cache_size() / sizeof(unsigned short));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
#endif

    // solve K
    {
#if __riscv_zvfh
        int tile_size = (l2_cache_size_fp16 - packn * 4) / (packn * 2);
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
        int nn_M = (M + packn * 4 - 1) / (packn * 4);

        TILE_M = std::max(packn, ((M + nn_M - 1) / nn_M + packn - 1) / packn * packn);
#else
        int nn_M = (M + 7) / 8;

        TILE_M = std::max(2, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
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
    }

    TILE_N = 0;
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
        TILE_N = std::max(16, TILE_N);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
        TILE_N = std::max(1, TILE_N);
#endif
    }
}

static void convolution_im2col_input_tile_conv1x1s1d1_fp16sa_rvv(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    // B = (N, maxk, inch/elempack, elempack)
    __fp16* pp = B;

#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
    const int elempack = bottom_blob.elempack;
#endif // __riscv_zvfh

    int jj = 0;
#if __riscv_zvfh
    for (; jj + 15 < max_jj; jj += 16)
    {
        if (elempack == packn)
        {
            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                int p = k / elempack + kk;

                const __fp16* sptr = (const __fp16*)bottom_blob.channel(p) + (j + jj) * elempack;
                const int stride = elempack;
                if (packn == 8)
                {
                    const size_t vl2 = __riscv_vsetvl_e16m2(16);
                    vfloat16m2_t _val0 = __riscv_vlse16_v_f16m2(sptr + 0, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val1 = __riscv_vlse16_v_f16m2(sptr + 1, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val2 = __riscv_vlse16_v_f16m2(sptr + 2, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val3 = __riscv_vlse16_v_f16m2(sptr + 3, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val4 = __riscv_vlse16_v_f16m2(sptr + 4, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val5 = __riscv_vlse16_v_f16m2(sptr + 5, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val6 = __riscv_vlse16_v_f16m2(sptr + 6, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val7 = __riscv_vlse16_v_f16m2(sptr + 7, stride * sizeof(__fp16), vl2);
                    __riscv_vse16_v_f16m2(pp, _val0, vl2);
                    __riscv_vse16_v_f16m2(pp + 1 * 16, _val1, vl2);
                    __riscv_vse16_v_f16m2(pp + 2 * 16, _val2, vl2);
                    __riscv_vse16_v_f16m2(pp + 3 * 16, _val3, vl2);
                    __riscv_vse16_v_f16m2(pp + 4 * 16, _val4, vl2);
                    __riscv_vse16_v_f16m2(pp + 5 * 16, _val5, vl2);
                    __riscv_vse16_v_f16m2(pp + 6 * 16, _val6, vl2);
                    __riscv_vse16_v_f16m2(pp + 7 * 16, _val7, vl2);
                }
                else if (packn == 16)
                {
                    vfloat16m1_t _val0 = __riscv_vle16_v_f16m1(sptr, vl);
                    vfloat16m1_t _val1 = __riscv_vle16_v_f16m1(sptr + stride, vl);
                    vfloat16m1_t _val2 = __riscv_vle16_v_f16m1(sptr + stride * 2, vl);
                    vfloat16m1_t _val3 = __riscv_vle16_v_f16m1(sptr + stride * 3, vl);
                    vfloat16m1_t _val4 = __riscv_vle16_v_f16m1(sptr + stride * 4, vl);
                    vfloat16m1_t _val5 = __riscv_vle16_v_f16m1(sptr + stride * 5, vl);
                    vfloat16m1_t _val6 = __riscv_vle16_v_f16m1(sptr + stride * 6, vl);
                    vfloat16m1_t _val7 = __riscv_vle16_v_f16m1(sptr + stride * 7, vl);
                    __riscv_vssseg8e16_v_f16m1x8(pp, 16 * sizeof(__fp16), __riscv_vcreate_v_f16m1x8(_val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7), vl);

                    _val0 = __riscv_vle16_v_f16m1(sptr + stride * 8, vl);
                    _val1 = __riscv_vle16_v_f16m1(sptr + stride * 9, vl);
                    _val2 = __riscv_vle16_v_f16m1(sptr + stride * 10, vl);
                    _val3 = __riscv_vle16_v_f16m1(sptr + stride * 11, vl);
                    _val4 = __riscv_vle16_v_f16m1(sptr + stride * 12, vl);
                    _val5 = __riscv_vle16_v_f16m1(sptr + stride * 13, vl);
                    _val6 = __riscv_vle16_v_f16m1(sptr + stride * 14, vl);
                    _val7 = __riscv_vle16_v_f16m1(sptr + stride * 15, vl);
                    __riscv_vssseg8e16_v_f16m1x8(pp + 8, 16 * sizeof(__fp16), __riscv_vcreate_v_f16m1x8(_val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7), vl);
                }
                else
                {
                    for (int n = 0; n < 16; n++)
                    {
                        vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr + n * stride, vl);
                        __riscv_vsse16_v_f16m1(pp + n, 16 * sizeof(__fp16), _val, vl);
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

                const __fp16* sptr = (const __fp16*)bottom_blob.channel(p) + j + jj;
                if (packn == 8)
                {
                    const size_t vl2 = __riscv_vsetvl_e16m2(16);
                    vfloat16m2_t _val = __riscv_vle16_v_f16m2(sptr, vl2);
                    __riscv_vse16_v_f16m2(pp, _val, vl2);
                }
                else
                {
                    const size_t vl1 = __riscv_vsetvl_e16m1(16);
                    vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr, vl1);
                    __riscv_vse16_v_f16m1(pp, _val, vl1);
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

                const __fp16* sptr = (const __fp16*)bottom_blob.channel(p) + (j + jj) * elempack;
                const int stride = elempack;
                if (packn == 8)
                {
                    const size_t vl2 = __riscv_vsetvl_e16m2(8);
                    vfloat16m2_t _val0 = __riscv_vlse16_v_f16m2(sptr + 0, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val1 = __riscv_vlse16_v_f16m2(sptr + 1, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val2 = __riscv_vlse16_v_f16m2(sptr + 2, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val3 = __riscv_vlse16_v_f16m2(sptr + 3, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val4 = __riscv_vlse16_v_f16m2(sptr + 4, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val5 = __riscv_vlse16_v_f16m2(sptr + 5, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val6 = __riscv_vlse16_v_f16m2(sptr + 6, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val7 = __riscv_vlse16_v_f16m2(sptr + 7, stride * sizeof(__fp16), vl2);
                    __riscv_vse16_v_f16m2(pp, _val0, vl2);
                    __riscv_vse16_v_f16m2(pp + 1 * 8, _val1, vl2);
                    __riscv_vse16_v_f16m2(pp + 2 * 8, _val2, vl2);
                    __riscv_vse16_v_f16m2(pp + 3 * 8, _val3, vl2);
                    __riscv_vse16_v_f16m2(pp + 4 * 8, _val4, vl2);
                    __riscv_vse16_v_f16m2(pp + 5 * 8, _val5, vl2);
                    __riscv_vse16_v_f16m2(pp + 6 * 8, _val6, vl2);
                    __riscv_vse16_v_f16m2(pp + 7 * 8, _val7, vl2);
                }
                else if (packn == 16)
                {
                    vfloat16m1_t _val0 = __riscv_vle16_v_f16m1(sptr, vl);
                    vfloat16m1_t _val1 = __riscv_vle16_v_f16m1(sptr + stride, vl);
                    vfloat16m1_t _val2 = __riscv_vle16_v_f16m1(sptr + stride * 2, vl);
                    vfloat16m1_t _val3 = __riscv_vle16_v_f16m1(sptr + stride * 3, vl);
                    vfloat16m1_t _val4 = __riscv_vle16_v_f16m1(sptr + stride * 4, vl);
                    vfloat16m1_t _val5 = __riscv_vle16_v_f16m1(sptr + stride * 5, vl);
                    vfloat16m1_t _val6 = __riscv_vle16_v_f16m1(sptr + stride * 6, vl);
                    vfloat16m1_t _val7 = __riscv_vle16_v_f16m1(sptr + stride * 7, vl);
                    __riscv_vssseg8e16_v_f16m1x8(pp, 8 * sizeof(__fp16), __riscv_vcreate_v_f16m1x8(_val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7), vl);
                }
                else
                {
                    for (int n = 0; n < 8; n++)
                    {
                        vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr + n * stride, vl);
                        __riscv_vsse16_v_f16m1(pp + n, 8 * sizeof(__fp16), _val, vl);
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

                const __fp16* sptr = (const __fp16*)bottom_blob.channel(p) + j + jj;
                const size_t vl1 = __riscv_vsetvl_e16m1(8);
                vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr, vl1);
                __riscv_vse16_v_f16m1(pp, _val, vl1);
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

                const __fp16* sptr = (const __fp16*)bottom_blob.channel(p) + (j + jj) * elempack;
                const int stride = elempack;
                if (packn == 8)
                {
                    const size_t vl2 = __riscv_vsetvl_e16m2(4);
                    vfloat16m2_t _val0 = __riscv_vlse16_v_f16m2(sptr + 0, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val1 = __riscv_vlse16_v_f16m2(sptr + 1, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val2 = __riscv_vlse16_v_f16m2(sptr + 2, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val3 = __riscv_vlse16_v_f16m2(sptr + 3, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val4 = __riscv_vlse16_v_f16m2(sptr + 4, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val5 = __riscv_vlse16_v_f16m2(sptr + 5, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val6 = __riscv_vlse16_v_f16m2(sptr + 6, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val7 = __riscv_vlse16_v_f16m2(sptr + 7, stride * sizeof(__fp16), vl2);
                    __riscv_vse16_v_f16m2(pp, _val0, vl2);
                    __riscv_vse16_v_f16m2(pp + 1 * 4, _val1, vl2);
                    __riscv_vse16_v_f16m2(pp + 2 * 4, _val2, vl2);
                    __riscv_vse16_v_f16m2(pp + 3 * 4, _val3, vl2);
                    __riscv_vse16_v_f16m2(pp + 4 * 4, _val4, vl2);
                    __riscv_vse16_v_f16m2(pp + 5 * 4, _val5, vl2);
                    __riscv_vse16_v_f16m2(pp + 6 * 4, _val6, vl2);
                    __riscv_vse16_v_f16m2(pp + 7 * 4, _val7, vl2);
                }
                else if (packn == 16)
                {
                    vfloat16m1_t _val0 = __riscv_vle16_v_f16m1(sptr, vl);
                    vfloat16m1_t _val1 = __riscv_vle16_v_f16m1(sptr + stride, vl);
                    vfloat16m1_t _val2 = __riscv_vle16_v_f16m1(sptr + stride * 2, vl);
                    vfloat16m1_t _val3 = __riscv_vle16_v_f16m1(sptr + stride * 3, vl);
                    __riscv_vssseg4e16_v_f16m1x4(pp, 4 * sizeof(__fp16), __riscv_vcreate_v_f16m1x4(_val0, _val1, _val2, _val3), vl);
                }
                else
                {
                    for (int n = 0; n < 4; n++)
                    {
                        vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr + n * stride, vl);
                        __riscv_vsse16_v_f16m1(pp + n, 4 * sizeof(__fp16), _val, vl);
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

                const __fp16* sptr = (const __fp16*)bottom_blob.channel(p) + j + jj;
                const size_t vl1 = __riscv_vsetvl_e16m1(4);
                vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr, vl1);
                __riscv_vse16_v_f16m1(pp, _val, vl1);
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

                const __fp16* sptr = (const __fp16*)bottom_blob.channel(p) + (j + jj) * elempack;
                const int stride = elempack;
                if (packn == 8)
                {
                    const size_t vl2 = __riscv_vsetvl_e16m2(2);
                    vfloat16m2_t _val0 = __riscv_vlse16_v_f16m2(sptr + 0, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val1 = __riscv_vlse16_v_f16m2(sptr + 1, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val2 = __riscv_vlse16_v_f16m2(sptr + 2, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val3 = __riscv_vlse16_v_f16m2(sptr + 3, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val4 = __riscv_vlse16_v_f16m2(sptr + 4, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val5 = __riscv_vlse16_v_f16m2(sptr + 5, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val6 = __riscv_vlse16_v_f16m2(sptr + 6, stride * sizeof(__fp16), vl2);
                    vfloat16m2_t _val7 = __riscv_vlse16_v_f16m2(sptr + 7, stride * sizeof(__fp16), vl2);
                    __riscv_vse16_v_f16m2(pp, _val0, vl2);
                    __riscv_vse16_v_f16m2(pp + 1 * 2, _val1, vl2);
                    __riscv_vse16_v_f16m2(pp + 2 * 2, _val2, vl2);
                    __riscv_vse16_v_f16m2(pp + 3 * 2, _val3, vl2);
                    __riscv_vse16_v_f16m2(pp + 4 * 2, _val4, vl2);
                    __riscv_vse16_v_f16m2(pp + 5 * 2, _val5, vl2);
                    __riscv_vse16_v_f16m2(pp + 6 * 2, _val6, vl2);
                    __riscv_vse16_v_f16m2(pp + 7 * 2, _val7, vl2);
                }
                else if (packn == 16)
                {
                    vfloat16m1_t _val0 = __riscv_vle16_v_f16m1(sptr, vl);
                    vfloat16m1_t _val1 = __riscv_vle16_v_f16m1(sptr + stride, vl);
                    __riscv_vssseg2e16_v_f16m1x2(pp, 2 * sizeof(__fp16), __riscv_vcreate_v_f16m1x2(_val0, _val1), vl);
                }
                else
                {
                    for (int n = 0; n < 2; n++)
                    {
                        vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr + n * stride, vl);
                        __riscv_vsse16_v_f16m1(pp + n, 2 * sizeof(__fp16), _val, vl);
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

                const __fp16* sptr = (const __fp16*)bottom_blob.channel(p) + j + jj;
                const size_t vl1 = __riscv_vsetvl_e16m1(2);
                vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr, vl1);
                __riscv_vse16_v_f16m1(pp, _val, vl1);
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

            const __fp16* sptr = (const __fp16*)bottom_blob.channel(p) + j + jj;
            pp[0] = sptr[0];
            pp[1] = sptr[1];
            pp += 2;
        }
    }
#endif // __riscv_zvfh
    for (; jj < max_jj; jj++)
    {
#if __riscv_zvfh
        if (elempack == packn)
        {
            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                int p = k / elempack + kk;

                const __fp16* sptr = (const __fp16*)bottom_blob.channel(p) + (j + jj) * elempack;
                vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr, vl);
                __riscv_vse16_v_f16m1(pp, _val, vl);
                pp += elempack;
            }
        }
        if (elempack == 1)
#endif // __riscv_zvfh
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                int p = k + kk;

                const __fp16* sptr = (const __fp16*)bottom_blob.channel(p) + j + jj;
                pp[0] = sptr[0];
                pp++;
            }
        }
    }
}

static inline void convolution_im2col_input_tile_impl_fp16sa_rvv(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    const int w = bottom_blob.w;
#if __riscv_zvfh
    const int elempack = bottom_blob.elempack;
#endif // __riscv_zvfh

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int maxk = kernel_w * kernel_h;

    // B = (N, maxk, inch/elempack, elempack)
    __fp16* pp = B;

    int jj = 0;
#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
    for (; jj + 15 < max_jj; jj += 16)
    {
        int dy0 = (j + jj) / outw;
        int dy15 = (j + jj + 15) / outw;
        int dx0 = (j + jj) % outw;

        if (dy0 == dy15)
        {
            int p = (k / elempack) / maxk;
            int uv = (k / elempack) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const __fp16* sptr = img.row<const __fp16>(y0) + x0 * elempack;

                if (elempack == packn)
                {
                    const int stride = stride_w * elempack;
                    if (packn == 8)
                    {
                        const size_t vl2 = __riscv_vsetvl_e16m2(16);
                        vfloat16m2_t _val0 = __riscv_vlse16_v_f16m2(sptr + 0, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val1 = __riscv_vlse16_v_f16m2(sptr + 1, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val2 = __riscv_vlse16_v_f16m2(sptr + 2, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val3 = __riscv_vlse16_v_f16m2(sptr + 3, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val4 = __riscv_vlse16_v_f16m2(sptr + 4, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val5 = __riscv_vlse16_v_f16m2(sptr + 5, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val6 = __riscv_vlse16_v_f16m2(sptr + 6, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val7 = __riscv_vlse16_v_f16m2(sptr + 7, stride * sizeof(__fp16), vl2);
                        __riscv_vse16_v_f16m2(pp, _val0, vl2);
                        __riscv_vse16_v_f16m2(pp + 1 * 16, _val1, vl2);
                        __riscv_vse16_v_f16m2(pp + 2 * 16, _val2, vl2);
                        __riscv_vse16_v_f16m2(pp + 3 * 16, _val3, vl2);
                        __riscv_vse16_v_f16m2(pp + 4 * 16, _val4, vl2);
                        __riscv_vse16_v_f16m2(pp + 5 * 16, _val5, vl2);
                        __riscv_vse16_v_f16m2(pp + 6 * 16, _val6, vl2);
                        __riscv_vse16_v_f16m2(pp + 7 * 16, _val7, vl2);
                    }
                    else if (packn == 16)
                    {
                        vfloat16m1_t _val0 = __riscv_vle16_v_f16m1(sptr, vl);
                        vfloat16m1_t _val1 = __riscv_vle16_v_f16m1(sptr + stride, vl);
                        vfloat16m1_t _val2 = __riscv_vle16_v_f16m1(sptr + stride * 2, vl);
                        vfloat16m1_t _val3 = __riscv_vle16_v_f16m1(sptr + stride * 3, vl);
                        vfloat16m1_t _val4 = __riscv_vle16_v_f16m1(sptr + stride * 4, vl);
                        vfloat16m1_t _val5 = __riscv_vle16_v_f16m1(sptr + stride * 5, vl);
                        vfloat16m1_t _val6 = __riscv_vle16_v_f16m1(sptr + stride * 6, vl);
                        vfloat16m1_t _val7 = __riscv_vle16_v_f16m1(sptr + stride * 7, vl);
                        __riscv_vssseg8e16_v_f16m1x8(pp, 16 * sizeof(__fp16), __riscv_vcreate_v_f16m1x8(_val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7), vl);

                        _val0 = __riscv_vle16_v_f16m1(sptr + stride * 8, vl);
                        _val1 = __riscv_vle16_v_f16m1(sptr + stride * 9, vl);
                        _val2 = __riscv_vle16_v_f16m1(sptr + stride * 10, vl);
                        _val3 = __riscv_vle16_v_f16m1(sptr + stride * 11, vl);
                        _val4 = __riscv_vle16_v_f16m1(sptr + stride * 12, vl);
                        _val5 = __riscv_vle16_v_f16m1(sptr + stride * 13, vl);
                        _val6 = __riscv_vle16_v_f16m1(sptr + stride * 14, vl);
                        _val7 = __riscv_vle16_v_f16m1(sptr + stride * 15, vl);
                        __riscv_vssseg8e16_v_f16m1x8(pp + 8, 16 * sizeof(__fp16), __riscv_vcreate_v_f16m1x8(_val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7), vl);
                    }
                    else
                    {
                        for (int n = 0; n < 16; n++)
                        {
                            vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr + n * stride, vl);
                            __riscv_vsse16_v_f16m1(pp + n, 16 * sizeof(__fp16), _val, vl);
                        }
                    }
                    pp += elempack * 16;
                }
                if (elempack == 1)
                {
                    const size_t vl16 = __riscv_vsetvl_e16m2(16);
                    if (stride_w == 1)
                    {
                        vfloat16m2_t _val = __riscv_vle16_v_f16m2(sptr, vl16);
                        __riscv_vse16_v_f16m2(pp, _val, vl16);
                    }
                    else
                    {
                        vfloat16m2_t _val = __riscv_vlse16_v_f16m2(sptr, stride_w * sizeof(__fp16), vl16);
                        __riscv_vse16_v_f16m2(pp, _val, vl16);
                    }
                    pp += 16;
                }

                v++;
                if (v == kernel_w)
                {
                    v = 0;
                    u++;
                    if (u == kernel_h)
                    {
                        u = 0;
                        p++;
                    }
                }
            }
        }
        else
        {
            int nn_size = 0;
            int nn_offset[16];
            int nn_count[16];
            int dy_table[16];
            int dx_table[16];

            int n = 0;
            while (n < 16)
            {
                int dy = (j + jj + n) / outw;
                int dx = (j + jj + n) % outw;
                int nn = outw - dx;
                if (nn > 16 - n)
                    nn = 16 - n;

                nn_offset[nn_size] = n;
                nn_count[nn_size] = nn;
                dy_table[nn_size] = dy;
                dx_table[nn_size] = dx;
                nn_size++;

                n += nn;
            }

            int p = (k / elempack) / maxk;
            int uv = (k / elempack) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                const Mat img = bottom_blob.channel(p);

                if (elempack == packn)
                {
                    for (int n = 0; n < 16; n++)
                    {
                        int dy = (j + jj + n) / outw;
                        int dx = (j + jj + n) % outw;
                        int x = stride_w * dx + dilation_w * v;
                        int y = stride_h * dy + dilation_h * u;

                        const __fp16* sptr = img.row<const __fp16>(y) + x * elempack;
                        vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr, vl);
                        __riscv_vsse16_v_f16m1(pp + n, 16 * sizeof(__fp16), _val, vl);
                    }
                    pp += elempack * 16;
                }
                if (elempack == 1)
                {
                    if (stride_w == 1 && dy15 == dy0 + 1)
                    {
                        int nn0 = outw - dx0;
                        int y0 = stride_h * dy0 + dilation_h * u;
                        int y1 = stride_h * (dy0 + 1) + dilation_h * u;

                        const __fp16* sptr0 = img.row<const __fp16>(y0) + dx0 + dilation_w * v;
                        const __fp16* sptr1 = img.row<const __fp16>(y1) + dilation_w * v;

                        int n = 0;
                        while (n < nn0)
                        {
                            const size_t vl1 = __riscv_vsetvl_e16m1(nn0 - n);
                            vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr0 + n, vl1);
                            __riscv_vse16_v_f16m1(pp + n, _val, vl1);
                            n += vl1;
                        }
                        while (n < 16)
                        {
                            const size_t vl1 = __riscv_vsetvl_e16m1(16 - n);
                            vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr1 + n - nn0, vl1);
                            __riscv_vse16_v_f16m1(pp + n, _val, vl1);
                            n += vl1;
                        }
                    }
                    else
                    {
                        for (int s = 0; s < nn_size; s++)
                        {
                            int nn = nn_count[s];
                            int x = stride_w * dx_table[s] + dilation_w * v;
                            int y = stride_h * dy_table[s] + dilation_h * u;
                            const __fp16* sptr = img.row<const __fp16>(y) + x;
                            __fp16* outptr = pp + nn_offset[s];

                            int n = 0;
                            while (n < nn)
                            {
                                const size_t vl1 = __riscv_vsetvl_e16m1(nn - n);
                                if (stride_w == 1)
                                {
                                    vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr + n, vl1);
                                    __riscv_vse16_v_f16m1(outptr + n, _val, vl1);
                                }
                                else
                                {
                                    vfloat16m1_t _val = __riscv_vlse16_v_f16m1(sptr + n * stride_w, stride_w * sizeof(__fp16), vl1);
                                    __riscv_vse16_v_f16m1(outptr + n, _val, vl1);
                                }
                                n += vl1;
                            }
                        }
                    }
                    pp += 16;
                }

                v++;
                if (v == kernel_w)
                {
                    v = 0;
                    u++;
                    if (u == kernel_h)
                    {
                        u = 0;
                        p++;
                    }
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
            int p = (k / elempack) / maxk;
            int uv = (k / elempack) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const __fp16* sptr = img.row<const __fp16>(y0) + x0 * elempack;

                if (elempack == packn)
                {
                    const int stride = stride_w * elempack;
                    if (packn == 8)
                    {
                        const size_t vl2 = __riscv_vsetvl_e16m2(8);
                        vfloat16m2_t _val0 = __riscv_vlse16_v_f16m2(sptr + 0, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val1 = __riscv_vlse16_v_f16m2(sptr + 1, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val2 = __riscv_vlse16_v_f16m2(sptr + 2, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val3 = __riscv_vlse16_v_f16m2(sptr + 3, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val4 = __riscv_vlse16_v_f16m2(sptr + 4, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val5 = __riscv_vlse16_v_f16m2(sptr + 5, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val6 = __riscv_vlse16_v_f16m2(sptr + 6, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val7 = __riscv_vlse16_v_f16m2(sptr + 7, stride * sizeof(__fp16), vl2);
                        __riscv_vse16_v_f16m2(pp, _val0, vl2);
                        __riscv_vse16_v_f16m2(pp + 1 * 8, _val1, vl2);
                        __riscv_vse16_v_f16m2(pp + 2 * 8, _val2, vl2);
                        __riscv_vse16_v_f16m2(pp + 3 * 8, _val3, vl2);
                        __riscv_vse16_v_f16m2(pp + 4 * 8, _val4, vl2);
                        __riscv_vse16_v_f16m2(pp + 5 * 8, _val5, vl2);
                        __riscv_vse16_v_f16m2(pp + 6 * 8, _val6, vl2);
                        __riscv_vse16_v_f16m2(pp + 7 * 8, _val7, vl2);
                    }
                    else if (packn == 16)
                    {
                        vfloat16m1_t _val0 = __riscv_vle16_v_f16m1(sptr, vl);
                        vfloat16m1_t _val1 = __riscv_vle16_v_f16m1(sptr + stride, vl);
                        vfloat16m1_t _val2 = __riscv_vle16_v_f16m1(sptr + stride * 2, vl);
                        vfloat16m1_t _val3 = __riscv_vle16_v_f16m1(sptr + stride * 3, vl);
                        vfloat16m1_t _val4 = __riscv_vle16_v_f16m1(sptr + stride * 4, vl);
                        vfloat16m1_t _val5 = __riscv_vle16_v_f16m1(sptr + stride * 5, vl);
                        vfloat16m1_t _val6 = __riscv_vle16_v_f16m1(sptr + stride * 6, vl);
                        vfloat16m1_t _val7 = __riscv_vle16_v_f16m1(sptr + stride * 7, vl);
                        __riscv_vssseg8e16_v_f16m1x8(pp, 8 * sizeof(__fp16), __riscv_vcreate_v_f16m1x8(_val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7), vl);
                    }
                    else
                    {
                        for (int n = 0; n < 8; n++)
                        {
                            vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr + n * stride, vl);
                            __riscv_vsse16_v_f16m1(pp + n, 8 * sizeof(__fp16), _val, vl);
                        }
                    }
                    pp += elempack * 8;
                }
                if (elempack == 1)
                {
                    const size_t vl8 = __riscv_vsetvl_e16m1(8);
                    if (stride_w == 1)
                    {
                        vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr, vl8);
                        __riscv_vse16_v_f16m1(pp, _val, vl8);
                    }
                    else
                    {
                        vfloat16m1_t _val = __riscv_vlse16_v_f16m1(sptr, stride_w * sizeof(__fp16), vl8);
                        __riscv_vse16_v_f16m1(pp, _val, vl8);
                    }
                    pp += 8;
                }

                v++;
                if (v == kernel_w)
                {
                    v = 0;
                    u++;
                    if (u == kernel_h)
                    {
                        u = 0;
                        p++;
                    }
                }
            }
        }
        else
        {
            int nn_size = 0;
            int nn_offset[8];
            int nn_count[8];
            int dy_table[8];
            int dx_table[8];

            int n = 0;
            while (n < 8)
            {
                int dy = (j + jj + n) / outw;
                int dx = (j + jj + n) % outw;
                int nn = outw - dx;
                if (nn > 8 - n)
                    nn = 8 - n;

                nn_offset[nn_size] = n;
                nn_count[nn_size] = nn;
                dy_table[nn_size] = dy;
                dx_table[nn_size] = dx;
                nn_size++;

                n += nn;
            }

            int p = (k / elempack) / maxk;
            int uv = (k / elempack) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                const Mat img = bottom_blob.channel(p);

                if (elempack == packn)
                {
                    for (int n = 0; n < 8; n++)
                    {
                        int dy = (j + jj + n) / outw;
                        int dx = (j + jj + n) % outw;
                        int x = stride_w * dx + dilation_w * v;
                        int y = stride_h * dy + dilation_h * u;

                        const __fp16* sptr = img.row<const __fp16>(y) + x * elempack;
                        vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr, vl);
                        __riscv_vsse16_v_f16m1(pp + n, 8 * sizeof(__fp16), _val, vl);
                    }
                    pp += elempack * 8;
                }
                if (elempack == 1)
                {
                    if (stride_w == 1 && dy7 == dy0 + 1)
                    {
                        int nn0 = outw - dx0;
                        int y0 = stride_h * dy0 + dilation_h * u;
                        int y1 = stride_h * (dy0 + 1) + dilation_h * u;

                        const __fp16* sptr0 = img.row<const __fp16>(y0) + dx0 + dilation_w * v;
                        const __fp16* sptr1 = img.row<const __fp16>(y1) + dilation_w * v;

                        int n = 0;
                        while (n < nn0)
                        {
                            const size_t vl1 = __riscv_vsetvl_e16m1(nn0 - n);
                            vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr0 + n, vl1);
                            __riscv_vse16_v_f16m1(pp + n, _val, vl1);
                            n += vl1;
                        }
                        while (n < 8)
                        {
                            const size_t vl1 = __riscv_vsetvl_e16m1(8 - n);
                            vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr1 + n - nn0, vl1);
                            __riscv_vse16_v_f16m1(pp + n, _val, vl1);
                            n += vl1;
                        }
                    }
                    else
                    {
                        for (int s = 0; s < nn_size; s++)
                        {
                            int nn = nn_count[s];
                            int x = stride_w * dx_table[s] + dilation_w * v;
                            int y = stride_h * dy_table[s] + dilation_h * u;
                            const __fp16* sptr = img.row<const __fp16>(y) + x;
                            __fp16* outptr = pp + nn_offset[s];

                            int n = 0;
                            while (n < nn)
                            {
                                const size_t vl1 = __riscv_vsetvl_e16m1(nn - n);
                                if (stride_w == 1)
                                {
                                    vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr + n, vl1);
                                    __riscv_vse16_v_f16m1(outptr + n, _val, vl1);
                                }
                                else
                                {
                                    vfloat16m1_t _val = __riscv_vlse16_v_f16m1(sptr + n * stride_w, stride_w * sizeof(__fp16), vl1);
                                    __riscv_vse16_v_f16m1(outptr + n, _val, vl1);
                                }
                                n += vl1;
                            }
                        }
                    }
                    pp += 8;
                }

                v++;
                if (v == kernel_w)
                {
                    v = 0;
                    u++;
                    if (u == kernel_h)
                    {
                        u = 0;
                        p++;
                    }
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
            int p = (k / elempack) / maxk;
            int uv = (k / elempack) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const __fp16* sptr = img.row<const __fp16>(y0) + x0 * elempack;

                if (elempack == packn)
                {
                    const int stride = stride_w * elempack;
                    if (packn == 8)
                    {
                        const size_t vl2 = __riscv_vsetvl_e16m2(4);
                        vfloat16m2_t _val0 = __riscv_vlse16_v_f16m2(sptr + 0, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val1 = __riscv_vlse16_v_f16m2(sptr + 1, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val2 = __riscv_vlse16_v_f16m2(sptr + 2, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val3 = __riscv_vlse16_v_f16m2(sptr + 3, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val4 = __riscv_vlse16_v_f16m2(sptr + 4, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val5 = __riscv_vlse16_v_f16m2(sptr + 5, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val6 = __riscv_vlse16_v_f16m2(sptr + 6, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val7 = __riscv_vlse16_v_f16m2(sptr + 7, stride * sizeof(__fp16), vl2);
                        __riscv_vse16_v_f16m2(pp, _val0, vl2);
                        __riscv_vse16_v_f16m2(pp + 1 * 4, _val1, vl2);
                        __riscv_vse16_v_f16m2(pp + 2 * 4, _val2, vl2);
                        __riscv_vse16_v_f16m2(pp + 3 * 4, _val3, vl2);
                        __riscv_vse16_v_f16m2(pp + 4 * 4, _val4, vl2);
                        __riscv_vse16_v_f16m2(pp + 5 * 4, _val5, vl2);
                        __riscv_vse16_v_f16m2(pp + 6 * 4, _val6, vl2);
                        __riscv_vse16_v_f16m2(pp + 7 * 4, _val7, vl2);
                    }
                    else if (packn == 16)
                    {
                        vfloat16m1_t _val0 = __riscv_vle16_v_f16m1(sptr, vl);
                        vfloat16m1_t _val1 = __riscv_vle16_v_f16m1(sptr + stride, vl);
                        vfloat16m1_t _val2 = __riscv_vle16_v_f16m1(sptr + stride * 2, vl);
                        vfloat16m1_t _val3 = __riscv_vle16_v_f16m1(sptr + stride * 3, vl);
                        __riscv_vssseg4e16_v_f16m1x4(pp, 4 * sizeof(__fp16), __riscv_vcreate_v_f16m1x4(_val0, _val1, _val2, _val3), vl);
                    }
                    else
                    {
                        for (int n = 0; n < 4; n++)
                        {
                            vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr + n * stride, vl);
                            __riscv_vsse16_v_f16m1(pp + n, 4 * sizeof(__fp16), _val, vl);
                        }
                    }
                    pp += elempack * 4;
                }
                if (elempack == 1)
                {
                    const size_t vl4 = __riscv_vsetvl_e16m1(4);
                    if (stride_w == 1)
                    {
                        vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr, vl4);
                        __riscv_vse16_v_f16m1(pp, _val, vl4);
                    }
                    else
                    {
                        vfloat16m1_t _val = __riscv_vlse16_v_f16m1(sptr, stride_w * sizeof(__fp16), vl4);
                        __riscv_vse16_v_f16m1(pp, _val, vl4);
                    }
                    pp += 4;
                }

                v++;
                if (v == kernel_w)
                {
                    v = 0;
                    u++;
                    if (u == kernel_h)
                    {
                        u = 0;
                        p++;
                    }
                }
            }
        }
        else
        {
            int nn_size = 0;
            int nn_offset[4];
            int nn_count[4];
            int dy_table[4];
            int dx_table[4];

            int n = 0;
            while (n < 4)
            {
                int dy = (j + jj + n) / outw;
                int dx = (j + jj + n) % outw;
                int nn = outw - dx;
                if (nn > 4 - n)
                    nn = 4 - n;

                nn_offset[nn_size] = n;
                nn_count[nn_size] = nn;
                dy_table[nn_size] = dy;
                dx_table[nn_size] = dx;
                nn_size++;

                n += nn;
            }

            int p = (k / elempack) / maxk;
            int uv = (k / elempack) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                const Mat img = bottom_blob.channel(p);

                if (elempack == packn)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        int dy = (j + jj + n) / outw;
                        int dx = (j + jj + n) % outw;
                        int x = stride_w * dx + dilation_w * v;
                        int y = stride_h * dy + dilation_h * u;

                        const __fp16* sptr = img.row<const __fp16>(y) + x * elempack;
                        vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr, vl);
                        __riscv_vsse16_v_f16m1(pp + n, 4 * sizeof(__fp16), _val, vl);
                    }
                    pp += elempack * 4;
                }
                if (elempack == 1)
                {
                    if (stride_w == 1 && dy3 == dy0 + 1)
                    {
                        int nn0 = outw - dx0;
                        int y0 = stride_h * dy0 + dilation_h * u;
                        int y1 = stride_h * (dy0 + 1) + dilation_h * u;

                        const __fp16* sptr0 = img.row<const __fp16>(y0) + dx0 + dilation_w * v;
                        const __fp16* sptr1 = img.row<const __fp16>(y1) + dilation_w * v;

                        int n = 0;
                        while (n < nn0)
                        {
                            const size_t vl1 = __riscv_vsetvl_e16m1(nn0 - n);
                            vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr0 + n, vl1);
                            __riscv_vse16_v_f16m1(pp + n, _val, vl1);
                            n += vl1;
                        }
                        while (n < 4)
                        {
                            const size_t vl1 = __riscv_vsetvl_e16m1(4 - n);
                            vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr1 + n - nn0, vl1);
                            __riscv_vse16_v_f16m1(pp + n, _val, vl1);
                            n += vl1;
                        }
                    }
                    else
                    {
                        for (int s = 0; s < nn_size; s++)
                        {
                            int nn = nn_count[s];
                            int x = stride_w * dx_table[s] + dilation_w * v;
                            int y = stride_h * dy_table[s] + dilation_h * u;
                            const __fp16* sptr = img.row<const __fp16>(y) + x;
                            __fp16* outptr = pp + nn_offset[s];

                            int n = 0;
                            while (n < nn)
                            {
                                const size_t vl1 = __riscv_vsetvl_e16m1(nn - n);
                                if (stride_w == 1)
                                {
                                    vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr + n, vl1);
                                    __riscv_vse16_v_f16m1(outptr + n, _val, vl1);
                                }
                                else
                                {
                                    vfloat16m1_t _val = __riscv_vlse16_v_f16m1(sptr + n * stride_w, stride_w * sizeof(__fp16), vl1);
                                    __riscv_vse16_v_f16m1(outptr + n, _val, vl1);
                                }
                                n += vl1;
                            }
                        }
                    }
                    pp += 4;
                }

                v++;
                if (v == kernel_w)
                {
                    v = 0;
                    u++;
                    if (u == kernel_h)
                    {
                        u = 0;
                        p++;
                    }
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
            int p = (k / elempack) / maxk;
            int uv = (k / elempack) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const __fp16* sptr = img.row<const __fp16>(y0) + x0 * elempack;

                if (elempack == packn)
                {
                    const int stride = stride_w * elempack;
                    if (packn == 8)
                    {
                        const size_t vl2 = __riscv_vsetvl_e16m2(2);
                        vfloat16m2_t _val0 = __riscv_vlse16_v_f16m2(sptr + 0, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val1 = __riscv_vlse16_v_f16m2(sptr + 1, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val2 = __riscv_vlse16_v_f16m2(sptr + 2, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val3 = __riscv_vlse16_v_f16m2(sptr + 3, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val4 = __riscv_vlse16_v_f16m2(sptr + 4, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val5 = __riscv_vlse16_v_f16m2(sptr + 5, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val6 = __riscv_vlse16_v_f16m2(sptr + 6, stride * sizeof(__fp16), vl2);
                        vfloat16m2_t _val7 = __riscv_vlse16_v_f16m2(sptr + 7, stride * sizeof(__fp16), vl2);
                        __riscv_vse16_v_f16m2(pp, _val0, vl2);
                        __riscv_vse16_v_f16m2(pp + 1 * 2, _val1, vl2);
                        __riscv_vse16_v_f16m2(pp + 2 * 2, _val2, vl2);
                        __riscv_vse16_v_f16m2(pp + 3 * 2, _val3, vl2);
                        __riscv_vse16_v_f16m2(pp + 4 * 2, _val4, vl2);
                        __riscv_vse16_v_f16m2(pp + 5 * 2, _val5, vl2);
                        __riscv_vse16_v_f16m2(pp + 6 * 2, _val6, vl2);
                        __riscv_vse16_v_f16m2(pp + 7 * 2, _val7, vl2);
                    }
                    else if (packn == 16)
                    {
                        vfloat16m1_t _val0 = __riscv_vle16_v_f16m1(sptr, vl);
                        vfloat16m1_t _val1 = __riscv_vle16_v_f16m1(sptr + stride, vl);
                        __riscv_vssseg2e16_v_f16m1x2(pp, 2 * sizeof(__fp16), __riscv_vcreate_v_f16m1x2(_val0, _val1), vl);
                    }
                    else
                    {
                        for (int n = 0; n < 2; n++)
                        {
                            vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr + n * stride, vl);
                            __riscv_vsse16_v_f16m1(pp + n, 2 * sizeof(__fp16), _val, vl);
                        }
                    }
                    pp += elempack * 2;
                }
                if (elempack == 1)
                {
                    const size_t vl2 = __riscv_vsetvl_e16m1(2);
                    if (stride_w == 1)
                    {
                        vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr, vl2);
                        __riscv_vse16_v_f16m1(pp, _val, vl2);
                    }
                    else
                    {
                        vfloat16m1_t _val = __riscv_vlse16_v_f16m1(sptr, stride_w * sizeof(__fp16), vl2);
                        __riscv_vse16_v_f16m1(pp, _val, vl2);
                    }
                    pp += 2;
                }

                v++;
                if (v == kernel_w)
                {
                    v = 0;
                    u++;
                    if (u == kernel_h)
                    {
                        u = 0;
                        p++;
                    }
                }
            }
        }
        else
        {
            int nn_size = 0;
            int nn_offset[2];
            int nn_count[2];
            int dy_table[2];
            int dx_table[2];

            int n = 0;
            while (n < 2)
            {
                int dy = (j + jj + n) / outw;
                int dx = (j + jj + n) % outw;
                int nn = outw - dx;
                if (nn > 2 - n)
                    nn = 2 - n;

                nn_offset[nn_size] = n;
                nn_count[nn_size] = nn;
                dy_table[nn_size] = dy;
                dx_table[nn_size] = dx;
                nn_size++;

                n += nn;
            }

            int p = (k / elempack) / maxk;
            int uv = (k / elempack) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            for (int kk = 0; kk < max_kk / elempack; kk++)
            {
                const Mat img = bottom_blob.channel(p);

                if (elempack == packn)
                {
                    for (int n = 0; n < 2; n++)
                    {
                        int dy = (j + jj + n) / outw;
                        int dx = (j + jj + n) % outw;
                        int x = stride_w * dx + dilation_w * v;
                        int y = stride_h * dy + dilation_h * u;

                        const __fp16* sptr = img.row<const __fp16>(y) + x * elempack;
                        vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr, vl);
                        __riscv_vsse16_v_f16m1(pp + n, 2 * sizeof(__fp16), _val, vl);
                    }
                    pp += elempack * 2;
                }
                if (elempack == 1)
                {
                    if (stride_w == 1 && dy1 == dy0 + 1)
                    {
                        int nn0 = outw - dx0;
                        int y0 = stride_h * dy0 + dilation_h * u;
                        int y1 = stride_h * (dy0 + 1) + dilation_h * u;

                        const __fp16* sptr0 = img.row<const __fp16>(y0) + dx0 + dilation_w * v;
                        const __fp16* sptr1 = img.row<const __fp16>(y1) + dilation_w * v;

                        int n = 0;
                        while (n < nn0)
                        {
                            const size_t vl1 = __riscv_vsetvl_e16m1(nn0 - n);
                            vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr0 + n, vl1);
                            __riscv_vse16_v_f16m1(pp + n, _val, vl1);
                            n += vl1;
                        }
                        while (n < 2)
                        {
                            const size_t vl1 = __riscv_vsetvl_e16m1(2 - n);
                            vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr1 + n - nn0, vl1);
                            __riscv_vse16_v_f16m1(pp + n, _val, vl1);
                            n += vl1;
                        }
                    }
                    else
                    {
                        for (int s = 0; s < nn_size; s++)
                        {
                            int nn = nn_count[s];
                            int x = stride_w * dx_table[s] + dilation_w * v;
                            int y = stride_h * dy_table[s] + dilation_h * u;
                            const __fp16* sptr = img.row<const __fp16>(y) + x;
                            __fp16* outptr = pp + nn_offset[s];

                            int n = 0;
                            while (n < nn)
                            {
                                const size_t vl1 = __riscv_vsetvl_e16m1(nn - n);
                                if (stride_w == 1)
                                {
                                    vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr + n, vl1);
                                    __riscv_vse16_v_f16m1(outptr + n, _val, vl1);
                                }
                                else
                                {
                                    vfloat16m1_t _val = __riscv_vlse16_v_f16m1(sptr + n * stride_w, stride_w * sizeof(__fp16), vl1);
                                    __riscv_vse16_v_f16m1(outptr + n, _val, vl1);
                                }
                                n += vl1;
                            }
                        }
                    }
                    pp += 2;
                }

                v++;
                if (v == kernel_w)
                {
                    v = 0;
                    u++;
                    if (u == kernel_h)
                    {
                        u = 0;
                        p++;
                    }
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

            pp[0] = img.row<const __fp16>(y0)[x0];
            pp[1] = img.row<const __fp16>(y1)[x1];
            pp += 2;
        }
    }
#endif // __riscv_zvfh
    for (; jj < max_jj; jj++)
    {
        int dy = (j + jj) / outw;
        int dx = (j + jj) % outw;

#if __riscv_zvfh
        int p = (k / elempack) / maxk;
        int uv = (k / elempack) % maxk;
        int u = uv / kernel_w;
        int v = uv % kernel_w;

        for (int kk = 0; kk < max_kk / elempack; kk++)
        {
            const Mat img = bottom_blob.channel(p);

            int x = stride_w * dx + dilation_w * v;
            int y = stride_h * dy + dilation_h * u;

            const __fp16* sptr = img.row<const __fp16>(y) + x * elempack;

            if (elempack == packn)
            {
                vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr, vl);
                __riscv_vse16_v_f16m1(pp, _val, vl);
                pp += elempack;
            }
            if (elempack == 1)
            {
                pp[0] = sptr[0];
                pp++;
            }

            v++;
            if (v == kernel_w)
            {
                v = 0;
                u++;
                if (u == kernel_h)
                {
                    u = 0;
                    p++;
                }
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

            const __fp16* sptr = img.row<const __fp16>(y) + x;
            pp[0] = sptr[0];
            pp++;
        }
#endif // __riscv_zvfh
    }
}

template<int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h>
static inline void convolution_im2col_input_tile_impl_fp16sa_rvv(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    convolution_im2col_input_tile_impl_fp16sa_rvv(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

static void convolution_im2col_input_tile_fp16sa_rvv(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_conv1x1s1d1_fp16sa_rvv(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_impl_fp16sa_rvv<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_impl_fp16sa_rvv<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_impl_fp16sa_rvv<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_impl_fp16sa_rvv<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_impl_fp16sa_rvv<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_impl_fp16sa_rvv<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    convolution_im2col_input_tile_impl_fp16sa_rvv(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

static void convolution_im2col_gemm_transform_kernel_fp16sa_rvv(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_fp16sa_rvv(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    int elempack = 1;
#if __riscv_zvfh
    if (opt.use_packing_layout)
    {
        int packn = csrr_vlenb() / 2;
        elempack = inch % packn == 0 ? packn : 1;
    }
#endif // __riscv_zvfh

    // maxk-inch-outch to pa-maxk-inch/pa-outch
    AT.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, (size_t)2u);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            convolution_im2col_pack_A_tile_fp16sa_rvv(kernel, AT_tile, i, max_ii, k, max_kk, maxk, K, elempack);
        }
    }
}

static int convolution_im2col_gemm_fp16sa_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
    const int maxk = kernel_w * kernel_h;

    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_fp16sa_rvv(M, N, K, TILE_M, TILE_N, TILE_K, nT);

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

        convolution_im2col_input_tile_fp16sa_rvv(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
    }

    Mat topT_tileX;
    if (K > TILE_K)
    {
        topT_tileX.create(TILE_N * TILE_M, 1, nT, 2u, opt.workspace_allocator);
        if (topT_tileX.empty())
            return -100;
    }

    if (nT > nn_M)
    {
        const int nn_MN = nn_M * nn_N;

        #pragma omp parallel for num_threads(nT)
        for (int ppij = 0; ppij < nn_MN; ppij++)
        {
            const int ppi = ppij / nn_N;
            const int ppj = ppij % nn_N;

            const int i = ppi * TILE_M;
            const int j = ppj * TILE_N;

            Mat topT_tile;
            if (K > TILE_K)
                topT_tile = topT_tileX.channel(get_omp_thread_num());

            const int max_ii = std::min((M - i), TILE_M);
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                const Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = k + TILE_K >= K;

                convolution_gemm_transB_packed_tile_fp16sa_rvv(AT_tile, BT_tile, bias, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end);
            }
        }
    }
    else
    {
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

                    convolution_gemm_transB_packed_tile_fp16sa_rvv(AT_tile, BT_tile, bias, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end);
                }
            }
        }
    }

    return 0;
}

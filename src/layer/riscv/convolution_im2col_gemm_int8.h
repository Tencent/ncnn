// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convolution_im2col_pack_A_tile_int8_rvv(const Mat& kernel, Mat& AT, int i, int max_ii, int k, int max_kk, int K)
{
    const signed char* kernel_ptr = kernel;
    signed char* pp = AT;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb();

    // Cap AVL to preserve full-packn AT/topT blocks with balanced VL selection.
    while (ii < max_ii)
    {
        const size_t vl = __riscv_vsetvl_e8m1(std::min(max_ii - ii, packn));
        const signed char* pA = kernel_ptr + (i + ii) * K + k;
        for (int kk = 0; kk < max_kk; kk++)
        {
            vint8m1_t _a = __riscv_vlse8_v_i8m1(pA + kk, K, vl);
            __riscv_vse8_v_i8m1(pp, _a, vl);
            pp += vl;
        }

        ii += vl;
    }
#else
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pA0 = kernel_ptr + (i + ii) * K + k;
        const signed char* pA1 = kernel_ptr + (i + ii + 1) * K + k;
        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = pA0[kk];
            pp[1] = pA1[kk];
            pp += 2;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const signed char* pA = kernel_ptr + (i + ii) * K + k;
        memcpy(pp, pA, max_kk);
        pp += max_kk;
    }
#endif // __riscv_vector
}

#if __riscv_vector
// Keep wide accumulators local to the kernel. Inlining into the OpenMP loop
// makes GCC spill vector registers inside the K loop.
__attribute__((noinline))
#endif
static void
convolution_gemm_transB_packed_tile_int8_rvv(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    const signed char* pAT = AT_tile;
    const signed char* pBT = BT_tile;

    int* outptr = topT_tile;

    int ii = 0;
#if __riscv_vector
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.cstep;
    const int packn = csrr_vlenb();
    const int packn_int32 = packn / 4;
    const size_t vl_int32 = __riscv_vsetvl_e32m1(packn_int32);

    while (ii < max_ii)
    {
        const size_t vl = __riscv_vsetvl_e32m4(std::min(max_ii - ii, packn));
        int* outptr0 = top_blob.channel((i + ii) / out_elempack);
        outptr0 += j * out_elempack;

        const signed char* pB = pBT;

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT + ii * max_kk;

            vint32m4_t _sum0;
            vint32m4_t _sum1;
            vint32m4_t _sum2;
            vint32m4_t _sum3;

            if (k == 0)
            {
                _sum0 = __riscv_vmv_v_x_i32m4(0, vl);
                _sum1 = _sum0;
                _sum2 = _sum0;
                _sum3 = _sum0;
            }
            else
            {
                _sum0 = __riscv_vle32_v_i32m4(outptr, vl);
                _sum1 = __riscv_vle32_v_i32m4(outptr + vl, vl);
                _sum2 = __riscv_vle32_v_i32m4(outptr + vl * 2, vl);
                _sum3 = __riscv_vle32_v_i32m4(outptr + vl * 3, vl);
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                vint8m1_t _a = __riscv_vle8_v_i8m1(pA, vl);
#if __riscv_xtheadvector
                vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(_a, 0, vl);
#else
                vint16m2_t _a16 = __riscv_vsext_vf2_i16m2(_a, vl);
#endif // __riscv_xtheadvector
                _sum0 = __riscv_vwmacc_vx_i32m4(_sum0, (short)pB[0], _a16, vl);
                _sum1 = __riscv_vwmacc_vx_i32m4(_sum1, (short)pB[1], _a16, vl);
                _sum2 = __riscv_vwmacc_vx_i32m4(_sum2, (short)pB[2], _a16, vl);
                _sum3 = __riscv_vwmacc_vx_i32m4(_sum3, (short)pB[3], _a16, vl);

                pA += vl;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == packn_int32)
                {
                    __riscv_vse32_v_i32m1(outptr0, __riscv_vget_v_i32m4_i32m1(_sum0, 0), vl_int32);
                    __riscv_vse32_v_i32m1(outptr0 + packn_int32, __riscv_vget_v_i32m4_i32m1(_sum1, 0), vl_int32);
                    __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum2, 0), vl_int32);
                    __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 3, __riscv_vget_v_i32m4_i32m1(_sum3, 0), vl_int32);
                    if (vl > (size_t)packn_int32 * 1)
                    {
                        __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum0, 1), vl_int32);
                        __riscv_vse32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum1, 1), vl_int32);
                        __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 2 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum2, 1), vl_int32);
                        __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 3 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum3, 1), vl_int32);
                    }
                    if (vl > (size_t)packn_int32 * 2)
                    {
                        __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum0, 2), vl_int32);
                        __riscv_vse32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum1, 2), vl_int32);
                        __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 2 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum2, 2), vl_int32);
                        __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 3 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum3, 2), vl_int32);
                    }
                    if (vl > (size_t)packn_int32 * 3)
                    {
                        __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 3, __riscv_vget_v_i32m4_i32m1(_sum0, 3), vl_int32);
                        __riscv_vse32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 3, __riscv_vget_v_i32m4_i32m1(_sum1, 3), vl_int32);
                        __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 2 + out_hstep * packn_int32 * 3, __riscv_vget_v_i32m4_i32m1(_sum2, 3), vl_int32);
                        __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 3 + out_hstep * packn_int32 * 3, __riscv_vget_v_i32m4_i32m1(_sum3, 3), vl_int32);
                    }
                    outptr0 += packn_int32 * 4;
                }
                else // if (out_elempack == 1)
                {
                    __riscv_vsse32_v_i32m4(outptr0, out_hstep * sizeof(int), _sum0, vl);
                    __riscv_vsse32_v_i32m4(outptr0 + 1, out_hstep * sizeof(int), _sum1, vl);
                    __riscv_vsse32_v_i32m4(outptr0 + 2, out_hstep * sizeof(int), _sum2, vl);
                    __riscv_vsse32_v_i32m4(outptr0 + 3, out_hstep * sizeof(int), _sum3, vl);
                    outptr0 += 4;
                }
            }
            else
            {
                __riscv_vse32_v_i32m4(outptr, _sum0, vl);
                __riscv_vse32_v_i32m4(outptr + vl, _sum1, vl);
                __riscv_vse32_v_i32m4(outptr + vl * 2, _sum2, vl);
                __riscv_vse32_v_i32m4(outptr + vl * 3, _sum3, vl);
            }

            if (outptr)
                outptr += vl * 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT + ii * max_kk;

            vint32m4_t _sum0;
            vint32m4_t _sum1;

            if (k == 0)
            {
                _sum0 = __riscv_vmv_v_x_i32m4(0, vl);
                _sum1 = _sum0;
            }
            else
            {
                _sum0 = __riscv_vle32_v_i32m4(outptr, vl);
                _sum1 = __riscv_vle32_v_i32m4(outptr + vl, vl);
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                vint8m1_t _a = __riscv_vle8_v_i8m1(pA, vl);
#if __riscv_xtheadvector
                vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(_a, 0, vl);
#else
                vint16m2_t _a16 = __riscv_vsext_vf2_i16m2(_a, vl);
#endif // __riscv_xtheadvector
                _sum0 = __riscv_vwmacc_vx_i32m4(_sum0, (short)pB[0], _a16, vl);
                _sum1 = __riscv_vwmacc_vx_i32m4(_sum1, (short)pB[1], _a16, vl);

                pA += vl;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == packn_int32)
                {
                    __riscv_vse32_v_i32m1(outptr0, __riscv_vget_v_i32m4_i32m1(_sum0, 0), vl_int32);
                    __riscv_vse32_v_i32m1(outptr0 + packn_int32, __riscv_vget_v_i32m4_i32m1(_sum1, 0), vl_int32);
                    if (vl > (size_t)packn_int32 * 1)
                    {
                        __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum0, 1), vl_int32);
                        __riscv_vse32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum1, 1), vl_int32);
                    }
                    if (vl > (size_t)packn_int32 * 2)
                    {
                        __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum0, 2), vl_int32);
                        __riscv_vse32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum1, 2), vl_int32);
                    }
                    if (vl > (size_t)packn_int32 * 3)
                    {
                        __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 3, __riscv_vget_v_i32m4_i32m1(_sum0, 3), vl_int32);
                        __riscv_vse32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 3, __riscv_vget_v_i32m4_i32m1(_sum1, 3), vl_int32);
                    }
                    outptr0 += packn_int32 * 2;
                }
                else // if (out_elempack == 1)
                {
                    __riscv_vsse32_v_i32m4(outptr0, out_hstep * sizeof(int), _sum0, vl);
                    __riscv_vsse32_v_i32m4(outptr0 + 1, out_hstep * sizeof(int), _sum1, vl);
                    outptr0 += 2;
                }
            }
            else
            {
                __riscv_vse32_v_i32m4(outptr, _sum0, vl);
                __riscv_vse32_v_i32m4(outptr + vl, _sum1, vl);
            }

            if (outptr)
                outptr += vl * 2;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pA = pAT + ii * max_kk;

            vint32m4_t _sum0;

            if (k == 0)
            {
                _sum0 = __riscv_vmv_v_x_i32m4(0, vl);
            }
            else
            {
                _sum0 = __riscv_vle32_v_i32m4(outptr, vl);
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                vint8m1_t _a = __riscv_vle8_v_i8m1(pA, vl);
#if __riscv_xtheadvector
                vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(_a, 0, vl);
#else
                vint16m2_t _a16 = __riscv_vsext_vf2_i16m2(_a, vl);
#endif // __riscv_xtheadvector
                _sum0 = __riscv_vwmacc_vx_i32m4(_sum0, (short)pB[0], _a16, vl);

                pA += vl;
                pB++;
            }

            if (k_end)
            {
                if (out_elempack == packn_int32)
                {
                    __riscv_vse32_v_i32m1(outptr0, __riscv_vget_v_i32m4_i32m1(_sum0, 0), vl_int32);
                    if (vl > (size_t)packn_int32 * 1)
                    {
                        __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum0, 1), vl_int32);
                    }
                    if (vl > (size_t)packn_int32 * 2)
                    {
                        __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum0, 2), vl_int32);
                    }
                    if (vl > (size_t)packn_int32 * 3)
                    {
                        __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 3, __riscv_vget_v_i32m4_i32m1(_sum0, 3), vl_int32);
                    }
                    outptr0 += packn_int32;
                }
                else // if (out_elempack == 1)
                {
                    __riscv_vsse32_v_i32m4(outptr0, out_hstep * sizeof(int), _sum0, vl);
                    outptr0++;
                }
            }
            else
            {
                __riscv_vse32_v_i32m4(outptr, _sum0, vl);
            }

            if (outptr)
                outptr += vl;
        }

        ii += vl;
    }
#else
    for (; ii + 1 < max_ii; ii += 2)
    {
        int* outptr0 = top_blob.channel(i + ii);
        int* outptr1 = top_blob.channel(i + ii + 1);
        outptr0 += j;
        outptr1 += j;

        const signed char* pB = pBT;

        int jj = 0;
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT + ii * max_kk;

            int sum00 = k == 0 ? 0 : outptr[0];
            int sum10 = k == 0 ? 0 : outptr[1];
            int sum01 = k == 0 ? 0 : outptr[2];
            int sum11 = k == 0 ? 0 : outptr[3];

            for (int kk = 0; kk < max_kk; kk++)
            {
                const int a0 = pA[0];
                const int a1 = pA[1];
                const int b0 = pB[0];
                const int b1 = pB[1];
                sum00 += a0 * b0;
                sum10 += a1 * b0;
                sum01 += a0 * b1;
                sum11 += a1 * b1;

                pA += 2;
                pB += 2;
            }

            if (k_end)
            {
                outptr0[0] = sum00;
                outptr1[0] = sum10;
                outptr0[1] = sum01;
                outptr1[1] = sum11;
                outptr0 += 2;
                outptr1 += 2;
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum10;
                outptr[2] = sum01;
                outptr[3] = sum11;
            }

            if (outptr)
                outptr += 4;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pA = pAT + ii * max_kk;

            int sum00 = k == 0 ? 0 : outptr[0];
            int sum10 = k == 0 ? 0 : outptr[1];

            for (int kk = 0; kk < max_kk; kk++)
            {
                const int a0 = pA[0];
                const int a1 = pA[1];
                const int b0 = pB[0];
                sum00 += a0 * b0;
                sum10 += a1 * b0;

                pA += 2;
                pB++;
            }

            if (k_end)
            {
                outptr0[0] = sum00;
                outptr1[0] = sum10;
                outptr0++;
                outptr1++;
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum10;
            }

            if (outptr)
                outptr += 2;
        }
    }
    for (; ii < max_ii; ii++)
    {
        int* outptr0 = top_blob.channel(i + ii);
        outptr0 += j;

        const signed char* pB = pBT;

        int jj = 0;
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT + ii * max_kk;

            int sum0 = k == 0 ? 0 : outptr[0];
            int sum1 = k == 0 ? 0 : outptr[1];

            for (int kk = 0; kk < max_kk; kk++)
            {
                const int a = pA[0];
                sum0 += a * pB[0];
                sum1 += a * pB[1];

                pA++;
                pB += 2;
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0[1] = sum1;
                outptr0 += 2;
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            if (outptr)
                outptr += 2;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pA = pAT + ii * max_kk;

            int sum0 = k == 0 ? 0 : outptr[0];

            for (int kk = 0; kk < max_kk; kk++)
            {
                const int a = pA[0];
                sum0 += a * pB[0];

                pA++;
                pB++;
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0++;
            }
            else
            {
                outptr[0] = sum0;
            }

            if (outptr)
                outptr += 1;
        }
    }
#endif // __riscv_vector
}

static void convolution_im2col_gemm_get_optimal_tile_mnk_int8_rvv(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    if (nT == 0)
        nT = get_physical_big_cpu_count();

#if __riscv_vector
    const int packn = csrr_vlenb();
#else
    const int packn = 1;
#endif
    const int l2_cache_size = std::max(16384, (int)get_cpu_level2_cache_size());

    // Balance output-channel tiles, keeping vector groups intact.
    int nn_M = (M + packn * 4 - 1) / (packn * 4);
    TILE_M = ((M + nn_M - 1) / nn_M + packn - 1) / packn * packn;
    TILE_M *= std::min(nT, get_physical_cpu_count());
    nn_M = (M + TILE_M - 1) / TILE_M;
    TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + packn - 1) / packn * packn);
    if (nT > 1)
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + packn - 1) / packn * packn);

    // Account for int8 A/B and int32 accumulators in the cache budget.
    int tile_size = (l2_cache_size - TILE_M * 4 * 4) / (TILE_M + 4);
    TILE_K = std::max(8, tile_size / 8 * 8);
    const int nn_K = (K + TILE_K - 1) / TILE_K;
    TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);

    TILE_N = 0;
    if (N > 0)
    {
        if (TILE_K >= K)
            tile_size = (l2_cache_size - TILE_M * TILE_K) / TILE_K;
        else
            tile_size = (l2_cache_size - TILE_M * TILE_K) / (TILE_K + TILE_M * 4);
        TILE_N = std::max(4, tile_size / 4 * 4);
        int nn_N = (N + TILE_N - 1) / TILE_N;
        nn_N = std::max(nn_N, std::min(nT, (N + 3) / 4));
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
    }
}

static void convolution_im2col_input_tile_conv1x1s1d1_int8_rvv(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    const int elempack = bottom_blob.elempack;
    signed char* pp = B;

    int jj = 0;
#if __riscv_vector
    for (; jj + 3 < max_jj; jj += 4)
    {
        for (int kk = 0; kk < max_kk; kk++)
        {
            const int q = k + kk;
            const signed char* pB = bottom_blob.channel(q / elempack);
            pB += (j + jj) * elempack + q % elempack;
            const size_t vl = __riscv_vsetvl_e8m1(4);
            vint8m1_t _b;
            if (elempack == 1)
                _b = __riscv_vle8_v_i8m1(pB, vl);
            else
                _b = __riscv_vlse8_v_i8m1(pB, elempack, vl);
            __riscv_vse8_v_i8m1(pp, _b, vl);
            pp += 4;
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
        for (int kk = 0; kk < max_kk; kk++)
        {
            const int q = k + kk;
            const signed char* pB = bottom_blob.channel(q / elempack);
            pB += (j + jj) * elempack + q % elempack;
#if __riscv_vector
            const size_t vl = __riscv_vsetvl_e8m1(2);
            vint8m1_t _b;
            if (elempack == 1)
                _b = __riscv_vle8_v_i8m1(pB, vl);
            else
                _b = __riscv_vlse8_v_i8m1(pB, elempack, vl);
            __riscv_vse8_v_i8m1(pp, _b, vl);
#else
            pp[0] = pB[0];
            pp[1] = pB[elempack];
#endif // __riscv_vector
            pp += 2;
        }
    }
    for (; jj < max_jj; jj++)
    {
        for (int kk = 0; kk < max_kk; kk++)
        {
            const int q = k + kk;
            const signed char* pB = bottom_blob.channel(q / elempack);
            pB += (j + jj) * elempack + q % elempack;
            pp[0] = pB[0];
            pp++;
        }
    }
}

static void convolution_im2col_input_tile_int8_rvv(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_conv1x1s1d1_int8_rvv(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const int maxk = kernel_w * kernel_h;
    const int outw = (w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    signed char* pp = B;

    int jj = 0;
#if __riscv_vector
    for (; jj + 3 < max_jj; jj += 4)
    {
        const int dy0 = (j + jj) / outw;
        const int dy3 = (j + jj + 3) / outw;
        const int dx0 = (j + jj) % outw;

        const int q = k / maxk;
        const int uv = k % maxk;
        int u = uv / kernel_w;
        int v = uv % kernel_w;
        int lane = q % elempack;
        const signed char* img = bottom_blob.channel(q / elempack);

        if (dy0 == dy3)
        {
            const int offset = (dy0 * stride_h * w + dx0 * stride_w) * elempack;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const signed char* pB = img + offset + (u * dilation_h * w + v * dilation_w) * elempack + lane;
                // All spatial blocks and row segments fit in e8m1.
                const size_t vl = 4;
                vint8m1_t _b;
                if (stride_w == 1 && elempack == 1)
                    _b = __riscv_vle8_v_i8m1(pB, vl);
                else
                    _b = __riscv_vlse8_v_i8m1(pB, stride_w * elempack, vl);
                __riscv_vse8_v_i8m1(pp, _b, vl);
                pp += 4;

                v++;
                if (v == kernel_w)
                {
                    v = 0;
                    u++;
                    if (u == kernel_h)
                    {
                        u = 0;
                        lane++;
                        if (lane == elempack)
                        {
                            lane = 0;
                            img += cstep;
                        }
                    }
                }
            }
        }
        else
        {
            int nn_size = 0;
            int nn_offset[4];
            int nn_count[4];
            int offset_table[4];
            int x = 0;
            int dy = dy0;
            int dx = dx0;
            while (x < 4)
            {
                const int count = std::min(4 - x, outw - dx);
                nn_offset[nn_size] = x;
                nn_count[nn_size] = count;
                offset_table[nn_size] = (dy * stride_h * w + dx * stride_w) * elempack;
                nn_size++;
                x += count;
                dy++;
                dx = 0;
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                const signed char* img0 = img + (u * dilation_h * w + v * dilation_w) * elempack + lane;
                for (int t = 0; t < nn_size; t++)
                {
                    const signed char* pB = img0 + offset_table[t];
                    signed char* outptr = pp + nn_offset[t];
                    const int count = nn_count[t];
                    const size_t vl = count;
                    vint8m1_t _b;
                    if (stride_w == 1 && elempack == 1)
                        _b = __riscv_vle8_v_i8m1(pB, vl);
                    else
                        _b = __riscv_vlse8_v_i8m1(pB, stride_w * elempack, vl);
                    __riscv_vse8_v_i8m1(outptr, _b, vl);
                }
                pp += 4;

                v++;
                if (v == kernel_w)
                {
                    v = 0;
                    u++;
                    if (u == kernel_h)
                    {
                        u = 0;
                        lane++;
                        if (lane == elempack)
                        {
                            lane = 0;
                            img += cstep;
                        }
                    }
                }
            }
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
        const int dy0 = (j + jj) / outw;
        const int dy1 = (j + jj + 1) / outw;
        const int dx0 = (j + jj) % outw;

        const int q = k / maxk;
        const int uv = k % maxk;
        int u = uv / kernel_w;
        int v = uv % kernel_w;
        int lane = q % elempack;
        const signed char* img = bottom_blob.channel(q / elempack);

        if (dy0 == dy1)
        {
            const int offset = (dy0 * stride_h * w + dx0 * stride_w) * elempack;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const signed char* pB = img + offset + (u * dilation_h * w + v * dilation_w) * elempack + lane;
#if __riscv_vector
                const size_t vl = 2;
                vint8m1_t _b;
                if (stride_w == 1 && elempack == 1)
                    _b = __riscv_vle8_v_i8m1(pB, vl);
                else
                    _b = __riscv_vlse8_v_i8m1(pB, stride_w * elempack, vl);
                __riscv_vse8_v_i8m1(pp, _b, vl);
#else
                pp[0] = pB[0];
                pp[1] = pB[1 * stride_w * elempack];
#endif // __riscv_vector
                pp += 2;

                v++;
                if (v == kernel_w)
                {
                    v = 0;
                    u++;
                    if (u == kernel_h)
                    {
                        u = 0;
                        lane++;
                        if (lane == elempack)
                        {
                            lane = 0;
                            img += cstep;
                        }
                    }
                }
            }
        }
        else
        {
            int nn_size = 0;
            int nn_offset[2];
            int nn_count[2];
            int offset_table[2];
            int x = 0;
            int dy = dy0;
            int dx = dx0;
            while (x < 2)
            {
                const int count = std::min(2 - x, outw - dx);
                nn_offset[nn_size] = x;
                nn_count[nn_size] = count;
                offset_table[nn_size] = (dy * stride_h * w + dx * stride_w) * elempack;
                nn_size++;
                x += count;
                dy++;
                dx = 0;
            }

            for (int kk = 0; kk < max_kk; kk++)
            {
                const signed char* img0 = img + (u * dilation_h * w + v * dilation_w) * elempack + lane;
                for (int t = 0; t < nn_size; t++)
                {
                    const signed char* pB = img0 + offset_table[t];
                    signed char* outptr = pp + nn_offset[t];
                    const int count = nn_count[t];
#if __riscv_vector
                    const size_t vl = count;
                    vint8m1_t _b;
                    if (stride_w == 1 && elempack == 1)
                        _b = __riscv_vle8_v_i8m1(pB, vl);
                    else
                        _b = __riscv_vlse8_v_i8m1(pB, stride_w * elempack, vl);
                    __riscv_vse8_v_i8m1(outptr, _b, vl);
#else
                    for (int n = 0; n < count; n++)
                        outptr[n] = pB[n * stride_w * elempack];
#endif // __riscv_vector
                }
                pp += 2;

                v++;
                if (v == kernel_w)
                {
                    v = 0;
                    u++;
                    if (u == kernel_h)
                    {
                        u = 0;
                        lane++;
                        if (lane == elempack)
                        {
                            lane = 0;
                            img += cstep;
                        }
                    }
                }
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
        const int dy = (j + jj) / outw;
        const int dx = (j + jj) % outw;
        const int offset = (dy * stride_h * w + dx * stride_w) * elempack;

        const int q = k / maxk;
        const int uv = k % maxk;
        int u = uv / kernel_w;
        int v = uv % kernel_w;
        int lane = q % elempack;
        const signed char* img = bottom_blob.channel(q / elempack);

        for (int kk = 0; kk < max_kk; kk++)
        {
            const signed char* pB = img + offset + (u * dilation_h * w + v * dilation_w) * elempack + lane;
            pp[0] = pB[0];
            pp++;

            v++;
            if (v == kernel_w)
            {
                v = 0;
                u++;
                if (u == kernel_h)
                {
                    u = 0;
                    lane++;
                    if (lane == elempack)
                    {
                        lane = 0;
                        img += cstep;
                    }
                }
            }
        }
    }
}

static int convolution_im2col_gemm_transform_kernel_int8_rvv(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    const int M = outch;
    const int K = inch * kernel_w * kernel_h;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_int8_rvv(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_K = (K + TILE_K - 1) / TILE_K;
    AT.create(TILE_M * TILE_K, nn_K, nn_M, (size_t)1u, 1);
    if (AT.empty())
        return -100;

    // K is input_channel * maxk + kernel_offset, independent of input packing.
    // Interleave output channels for reuse across spatial positions.
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;
        const int max_ii = std::min(M - i, TILE_M);

        for (int ppk = 0; ppk < nn_K; ppk++)
        {
            const int k = ppk * TILE_K;
            const int max_kk = std::min(K - k, TILE_K);
            Mat AT_tile = AT.channel(ppi).row_range(ppk, 1);
            convolution_im2col_pack_A_tile_int8_rvv(kernel, AT_tile, i, max_ii, k, max_kk, K);
        }
    }

    return 0;
}

static int convolution_im2col_gemm_int8_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * kernel_w * kernel_h;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_int8_rvv(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // Prefer parallel spatial tiles, reusing each im2col panel across M.
    // Split M to occupy the threads, then bound split-K workspace below.
    int nn_Mgroup = std::min(nn_M, (nT + nn_N - 1) / nn_N);
    if (nn_K > 1)
    {
        // Keep each worker's input panel and partial sums within the cache
        // budget when possible, retaining B reuse across its assigned M tiles.
        const size_t l2_cache_size = std::max(16384, (int)get_cpu_level2_cache_size());
        const size_t bt_tile_size = (size_t)TILE_K * TILE_N;
        const size_t topT_tile_size = (size_t)TILE_M * TILE_N * sizeof(int);
        const size_t topT_budget = l2_cache_size > bt_tile_size ? l2_cache_size - bt_tile_size : 0;
        const int max_M_tiles = std::max(1, (int)(topT_budget / topT_tile_size));
        nn_Mgroup = std::max(nn_Mgroup, (nn_M + max_M_tiles - 1) / max_M_tiles);
    }
    const int nn_MN = nn_Mgroup * nn_N;
    const int num_threads = std::min(nT, nn_MN);

    // Bounded workspace: one K/N panel per worker, never a full im2col matrix.
    // Each worker owns one partial-sum slice for every M tile in its group.
    Mat BT_tileX(TILE_K * TILE_N, 1, num_threads, (size_t)1u, 1, opt.workspace_allocator);
    if (BT_tileX.empty())
        return -100;

    Mat topT_tileX;
    if (nn_K > 1)
    {
        const int nn_M_per_group = (nn_M + nn_Mgroup - 1) / nn_Mgroup;
        topT_tileX.create(TILE_M * TILE_N, nn_M_per_group, num_threads, (size_t)4u, 1, opt.workspace_allocator);
        if (topT_tileX.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(num_threads)
    for (int ppij = 0; ppij < nn_MN; ppij++)
    {
        const int ppj = ppij / nn_Mgroup;
        const int ppi_group = ppij % nn_Mgroup;
        const int ppi_start = nn_M * ppi_group / nn_Mgroup;
        const int ppi_end = nn_M * (ppi_group + 1) / nn_Mgroup;
        const int j = ppj * TILE_N;
        const int max_jj = std::min(N - j, TILE_N);
        const int tid = get_omp_thread_num();
        Mat BT_tile = BT_tileX.channel(tid);

        for (int ppk = 0; ppk < nn_K; ppk++)
        {
            const int k = ppk * TILE_K;
            const int max_kk = std::min(K - k, TILE_K);
            convolution_im2col_input_tile_int8_rvv(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);

            for (int ppi = ppi_start; ppi < ppi_end; ppi++)
            {
                const int i = ppi * TILE_M;
                const int max_ii = std::min(M - i, TILE_M);
                const Mat AT_tile = AT.channel(ppi).row_range(ppk, 1);
                Mat topT_tile;
                if (nn_K > 1)
                    topT_tile = topT_tileX.channel(tid).row_range(ppi - ppi_start, 1);

                const bool k_end = ppk + 1 == nn_K;
                convolution_gemm_transB_packed_tile_int8_rvv(AT_tile, BT_tile, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end);
            }
        }
    }

    return 0;
}

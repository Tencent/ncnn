// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if __riscv_vector
// Keep the widening accumulators local to the kernel, outside the OpenMP loop.
__attribute__((noinline))
#endif
static void
conv3x3s1_winograd_gemm_transB_packed_tile_int8_rvv(const Mat& AT_tile, const Mat& BT_tile, Mat& top_tile, int batch, int max_ii, int max_jj, int k, int max_kk)
{
    for (int b = 0; b < batch; b++)
    {
        const short* pAT = AT_tile.row<const short>(b);
        const short* pBT = BT_tile.row<const short>(b);

        int ii = 0;
#if __riscv_vector
        const int packn = csrr_vlenb() / 2;

        while (ii < max_ii)
        {
            const size_t vl = __riscv_vsetvl_e16m1(std::min(max_ii - ii, packn));
            int* outptr = (int*)top_tile + ii * max_jj * batch + b * max_jj * vl;
            const short* pB = pBT;

            int jj = 0;
            for (; jj + 7 < max_jj; jj += 8)
            {
                const short* pA = pAT + ii * max_kk;

                vint32m2_t _sum0;
                vint32m2_t _sum1;
                vint32m2_t _sum2;
                vint32m2_t _sum3;
                vint32m2_t _sum4;
                vint32m2_t _sum5;
                vint32m2_t _sum6;
                vint32m2_t _sum7;

                if (k == 0)
                {
                    _sum0 = __riscv_vmv_v_x_i32m2(0, vl);
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
                    _sum0 = __riscv_vle32_v_i32m2(outptr, vl);
                    _sum1 = __riscv_vle32_v_i32m2(outptr + vl, vl);
                    _sum2 = __riscv_vle32_v_i32m2(outptr + vl * 2, vl);
                    _sum3 = __riscv_vle32_v_i32m2(outptr + vl * 3, vl);
                    _sum4 = __riscv_vle32_v_i32m2(outptr + vl * 4, vl);
                    _sum5 = __riscv_vle32_v_i32m2(outptr + vl * 5, vl);
                    _sum6 = __riscv_vle32_v_i32m2(outptr + vl * 6, vl);
                    _sum7 = __riscv_vle32_v_i32m2(outptr + vl * 7, vl);
                }

                for (int kk = 0; kk < max_kk; kk++)
                {
                    vint16m1_t _a = __riscv_vle16_v_i16m1(pA, vl);
                    _sum0 = __riscv_vwmacc_vx_i32m2(_sum0, pB[0], _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m2(_sum1, pB[1], _a, vl);
                    _sum2 = __riscv_vwmacc_vx_i32m2(_sum2, pB[2], _a, vl);
                    _sum3 = __riscv_vwmacc_vx_i32m2(_sum3, pB[3], _a, vl);
                    _sum4 = __riscv_vwmacc_vx_i32m2(_sum4, pB[4], _a, vl);
                    _sum5 = __riscv_vwmacc_vx_i32m2(_sum5, pB[5], _a, vl);
                    _sum6 = __riscv_vwmacc_vx_i32m2(_sum6, pB[6], _a, vl);
                    _sum7 = __riscv_vwmacc_vx_i32m2(_sum7, pB[7], _a, vl);

                    pA += vl;
                    pB += 8;
                }

                __riscv_vse32_v_i32m2(outptr, _sum0, vl);
                __riscv_vse32_v_i32m2(outptr + vl, _sum1, vl);
                __riscv_vse32_v_i32m2(outptr + vl * 2, _sum2, vl);
                __riscv_vse32_v_i32m2(outptr + vl * 3, _sum3, vl);
                __riscv_vse32_v_i32m2(outptr + vl * 4, _sum4, vl);
                __riscv_vse32_v_i32m2(outptr + vl * 5, _sum5, vl);
                __riscv_vse32_v_i32m2(outptr + vl * 6, _sum6, vl);
                __riscv_vse32_v_i32m2(outptr + vl * 7, _sum7, vl);
                outptr += vl * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const short* pA = pAT + ii * max_kk;

                vint32m2_t _sum0;
                vint32m2_t _sum1;
                vint32m2_t _sum2;
                vint32m2_t _sum3;

                if (k == 0)
                {
                    _sum0 = __riscv_vmv_v_x_i32m2(0, vl);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
                else
                {
                    _sum0 = __riscv_vle32_v_i32m2(outptr, vl);
                    _sum1 = __riscv_vle32_v_i32m2(outptr + vl, vl);
                    _sum2 = __riscv_vle32_v_i32m2(outptr + vl * 2, vl);
                    _sum3 = __riscv_vle32_v_i32m2(outptr + vl * 3, vl);
                }

                for (int kk = 0; kk < max_kk; kk++)
                {
                    vint16m1_t _a = __riscv_vle16_v_i16m1(pA, vl);
                    _sum0 = __riscv_vwmacc_vx_i32m2(_sum0, pB[0], _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m2(_sum1, pB[1], _a, vl);
                    _sum2 = __riscv_vwmacc_vx_i32m2(_sum2, pB[2], _a, vl);
                    _sum3 = __riscv_vwmacc_vx_i32m2(_sum3, pB[3], _a, vl);

                    pA += vl;
                    pB += 4;
                }

                __riscv_vse32_v_i32m2(outptr, _sum0, vl);
                __riscv_vse32_v_i32m2(outptr + vl, _sum1, vl);
                __riscv_vse32_v_i32m2(outptr + vl * 2, _sum2, vl);
                __riscv_vse32_v_i32m2(outptr + vl * 3, _sum3, vl);
                outptr += vl * 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const short* pA = pAT + ii * max_kk;

                vint32m2_t _sum0;
                vint32m2_t _sum1;

                if (k == 0)
                {
                    _sum0 = __riscv_vmv_v_x_i32m2(0, vl);
                    _sum1 = _sum0;
                }
                else
                {
                    _sum0 = __riscv_vle32_v_i32m2(outptr, vl);
                    _sum1 = __riscv_vle32_v_i32m2(outptr + vl, vl);
                }

                for (int kk = 0; kk < max_kk; kk++)
                {
                    vint16m1_t _a = __riscv_vle16_v_i16m1(pA, vl);
                    _sum0 = __riscv_vwmacc_vx_i32m2(_sum0, pB[0], _a, vl);
                    _sum1 = __riscv_vwmacc_vx_i32m2(_sum1, pB[1], _a, vl);

                    pA += vl;
                    pB += 2;
                }

                __riscv_vse32_v_i32m2(outptr, _sum0, vl);
                __riscv_vse32_v_i32m2(outptr + vl, _sum1, vl);
                outptr += vl * 2;
            }
            for (; jj < max_jj; jj++)
            {
                const short* pA = pAT + ii * max_kk;

                vint32m2_t _sum0;

                if (k == 0)
                {
                    _sum0 = __riscv_vmv_v_x_i32m2(0, vl);
                }
                else
                {
                    _sum0 = __riscv_vle32_v_i32m2(outptr, vl);
                }

                for (int kk = 0; kk < max_kk; kk++)
                {
                    vint16m1_t _a = __riscv_vle16_v_i16m1(pA, vl);
                    _sum0 = __riscv_vwmacc_vx_i32m2(_sum0, pB[0], _a, vl);

                    pA += vl;
                    pB += 1;
                }

                __riscv_vse32_v_i32m2(outptr, _sum0, vl);
                outptr += vl;
            }

            ii += vl;
        }
#else
        // Match the defined wrapping of the RVV accumulators for F(4,3).
        for (; ii + 1 < max_ii; ii += 2)
        {
            unsigned int* outptr = (unsigned int*)top_tile + ii * max_jj * batch + b * max_jj * 2;
            const short* pB = pBT;

            int jj = 0;
            for (; jj + 1 < max_jj; jj += 2)
            {
                const short* pA = pAT + ii * max_kk;

                unsigned int sum00 = k == 0 ? 0 : outptr[0];
                unsigned int sum10 = k == 0 ? 0 : outptr[1];
                unsigned int sum01 = k == 0 ? 0 : outptr[2];
                unsigned int sum11 = k == 0 ? 0 : outptr[3];

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

                outptr[0] = sum00;
                outptr[1] = sum10;
                outptr[2] = sum01;
                outptr[3] = sum11;
                outptr += 4;
            }
            for (; jj < max_jj; jj++)
            {
                const short* pA = pAT + ii * max_kk;

                unsigned int sum00 = k == 0 ? 0 : outptr[0];
                unsigned int sum10 = k == 0 ? 0 : outptr[1];

                for (int kk = 0; kk < max_kk; kk++)
                {
                    const int a0 = pA[0];
                    const int a1 = pA[1];
                    const int b0 = pB[0];
                    sum00 += a0 * b0;
                    sum10 += a1 * b0;

                    pA += 2;
                    pB += 1;
                }

                outptr[0] = sum00;
                outptr[1] = sum10;
                outptr += 2;
            }
        }
        for (; ii < max_ii; ii++)
        {
            unsigned int* outptr = (unsigned int*)top_tile + ii * max_jj * batch + b * max_jj;
            const short* pB = pBT;

            int jj = 0;
            for (; jj + 1 < max_jj; jj += 2)
            {
                const short* pA = pAT + ii * max_kk;

                unsigned int sum0 = k == 0 ? 0 : outptr[0];
                unsigned int sum1 = k == 0 ? 0 : outptr[1];

                for (int kk = 0; kk < max_kk; kk++)
                {
                    const int a = pA[0];
                    sum0 += a * pB[0];
                    sum1 += a * pB[1];

                    pA += 1;
                    pB += 2;
                }

                outptr[0] = sum0;
                outptr[1] = sum1;
                outptr += 2;
            }
            for (; jj < max_jj; jj++)
            {
                const short* pA = pAT + ii * max_kk;

                unsigned int sum0 = k == 0 ? 0 : outptr[0];

                for (int kk = 0; kk < max_kk; kk++)
                {
                    const int a = pA[0];
                    sum0 += a * pB[0];

                    pA += 1;
                    pB += 1;
                }

                outptr[0] = sum0;
                outptr += 1;
            }
        }
#endif // __riscv_vector
    }
}

static void conv3x3s1_winograd_get_optimal_tile_mnk_int8_rvv(int M, int N, int K, int B, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    if (nT == 0)
        nT = get_physical_big_cpu_count();

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
#else
    const int packn = 2;
#endif
    // Budget all transform coefficients, including int32 partial sums.
    const int l2_cache_size = std::max(16384, (int)get_cpu_level2_cache_size()) / B;

    // solve M
    {
        int nn_M = (M + packn * 2 - 1) / (packn * 2);
        TILE_M = ((M + nn_M - 1) / nn_M + packn - 1) / packn * packn;
        TILE_M *= std::min(nT, get_physical_cpu_count());
        nn_M = (M + TILE_M - 1) / TILE_M;
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + packn - 1) / packn * packn);
        if (nT > 1)
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + packn - 1) / packn * packn);
    }

    // solve K
    {
        int tile_size = (l2_cache_size - TILE_M * 4 * 4) / (TILE_M * 2 + 4 * 2);
        TILE_K = std::max(8, tile_size / 8 * 8);
        const int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);
    }

    // solve N
    TILE_N = 0;
    if (N > 0)
    {
        int tile_size = (l2_cache_size - TILE_M * TILE_K * 2) / (TILE_K * 2 + TILE_M * 4);
        TILE_N = std::max(4, tile_size / 4 * 4);
        int nn_N = (N + TILE_N - 1) / TILE_N;
        nn_N = std::max(nn_N, std::min(nT, (N + 3) / 4));
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
    }
}

static void conv3x3s1_winograd23_transform_kernel_tile_int8_rvv(const Mat& kernel, Mat& AT, int inch, int i, int max_ii, int k, int max_kk)
{
    // G' = 2G, so the two-dimensional output transform is divided by 4.
    // const signed char ktm[4][3] = {
    //     {2, 0, 0},
    //     {1, 1, 1},
    //     {1, -1, 1},
    //     {0, 0, 2}
    // };

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
#else
    const int packn = 2;
#endif

    int ii = 0;
    while (ii < max_ii)
    {
        const int max_ii_block = std::min(max_ii - ii, packn);
        for (int kk = 0; kk < max_kk; kk++)
        {
            for (int iii = 0; iii < max_ii_block; iii++)
            {
                short tmp[4][3];
                const signed char* k0 = (const signed char*)kernel + (i + ii + iii) * inch * 9 + (k + kk) * 9;

                for (int m = 0; m < 3; m++)
                {
                    const int r0 = k0[0];
                    const int r1 = k0[1];
                    const int r2 = k0[2];

                    tmp[0][m] = r0 * 2;
                    tmp[1][m] = r0 + r1 + r2;
                    tmp[2][m] = r0 - r1 + r2;
                    tmp[3][m] = r2 * 2;
                    k0 += 3;
                }

                for (int m = 0; m < 4; m++)
                {
                    const int r0 = tmp[m][0];
                    const int r1 = tmp[m][1];
                    const int r2 = tmp[m][2];

                    const int offset = ii * max_kk + kk * max_ii_block + iii;
                    AT.row<short>(m * 4 + 0)[offset] = r0 * 2;
                    AT.row<short>(m * 4 + 1)[offset] = r0 + r1 + r2;
                    AT.row<short>(m * 4 + 2)[offset] = r0 - r1 + r2;
                    AT.row<short>(m * 4 + 3)[offset] = r2 * 2;
                }
            }
        }
        ii += max_ii_block;
    }
}

static int conv3x3s1_winograd23_transform_kernel_int8_rvv(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 16;

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk_int8_rvv(M, 0, K, B, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_K = (K + TILE_K - 1) / TILE_K;
    AT.create(TILE_M * TILE_K, B, nn_K, nn_M, (size_t)2u, 1);
    if (AT.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;
        const int max_ii = std::min(M - i, TILE_M);

        for (int ppk = 0; ppk < nn_K; ppk++)
        {
            const int k = ppk * TILE_K;
            const int max_kk = std::min(K - k, TILE_K);
            Mat AT_tile = AT.channel(ppi).depth(ppk);
            conv3x3s1_winograd23_transform_kernel_tile_int8_rvv(kernel, AT_tile, inch, i, max_ii, k, max_kk);
        }
    }

    return 0;
}

static inline void conv3x3s1_winograd23_transform_input_tile_int8_rvv(const Mat& bottom_blob, Mat& BT, int j, int max_jj, int k, int max_kk)
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
    const int w_tiles = (w - 1) / 2;
#if __riscv_vector
    const int packn = csrr_vlenb();
    const size_t N = bottom_blob.cstep * elempack;
#endif

    // Fuse the transform and the BT packing. Each row contains jj blocks,
    // with the channels interleaved inside each block of spatial tiles.
    int jj = 0;
    while (jj < max_jj)
    {
        int max_jj_block = max_jj - jj >= 2 ? 2 : 1;
#if __riscv_vector
        if (max_jj - jj >= 4)
            max_jj_block = 4;
        if (max_jj - jj >= 8)
            max_jj_block = 8;
#endif

        for (int jjj = 0; jjj < max_jj_block; jjj++)
        {
            const int ti = (j + jj + jjj) / w_tiles;
            const int tj = (j + jj + jjj) % w_tiles;

            int kk = 0;
#if __riscv_vector
            while (kk < max_kk)
            {
                const int q = k + kk;
                const int lane = q % elempack;
                int max_vl = std::min(max_kk - kk, packn);
                if (elempack != 1)
                    max_vl = std::min(max_vl, elempack - lane);
                const size_t vl = __riscv_vsetvl_e8m1(max_vl);

                short tmp[4][4][packn];

                for (int m = 0; m < 4; m++)
                {
                    vint8m1_t _r0 = __riscv_vmv_v_x_i8m1(0, vl);
                    vint8m1_t _r1 = _r0;
                    vint8m1_t _r2 = _r0;
                    vint8m1_t _r3 = _r0;

                    if (ti * 2 + m < h && tj * 2 < w)
                    {
                        const signed char* rptr = bottom_blob.channel(q / elempack).row<const signed char>(ti * 2 + m) + tj * 2 * elempack + lane;
                        if (elempack != 1)
                        {
                            _r0 = __riscv_vle8_v_i8m1(rptr, vl);
                            if (tj * 2 + 1 < w) _r1 = __riscv_vle8_v_i8m1(rptr + elempack, vl);
                            if (tj * 2 + 2 < w) _r2 = __riscv_vle8_v_i8m1(rptr + elempack * 2, vl);
                            if (tj * 2 + 3 < w) _r3 = __riscv_vle8_v_i8m1(rptr + elempack * 3, vl);
                        }
                        else
                        {
                            _r0 = __riscv_vlse8_v_i8m1(rptr, N, vl);
                            if (tj * 2 + 1 < w) _r1 = __riscv_vlse8_v_i8m1(rptr + 1, N, vl);
                            if (tj * 2 + 2 < w) _r2 = __riscv_vlse8_v_i8m1(rptr + 2, N, vl);
                            if (tj * 2 + 3 < w) _r3 = __riscv_vlse8_v_i8m1(rptr + 3, N, vl);
                        }
                    }

                    __riscv_vse16_v_i16m2(tmp[0][m], __riscv_vwsub_vv_i16m2(_r0, _r2, vl), vl);
                    __riscv_vse16_v_i16m2(tmp[1][m], __riscv_vwadd_vv_i16m2(_r1, _r2, vl), vl);
                    __riscv_vse16_v_i16m2(tmp[2][m], __riscv_vwsub_vv_i16m2(_r2, _r1, vl), vl);
                    __riscv_vse16_v_i16m2(tmp[3][m], __riscv_vwsub_vv_i16m2(_r3, _r1, vl), vl);
                }

                const int offset = jj * max_kk + kk * max_jj_block + jjj;
                for (int m = 0; m < 4; m++)
                {
                    vint16m2_t _r0 = __riscv_vle16_v_i16m2(tmp[m][0], vl);
                    vint16m2_t _r1 = __riscv_vle16_v_i16m2(tmp[m][1], vl);
                    vint16m2_t _r2 = __riscv_vle16_v_i16m2(tmp[m][2], vl);
                    vint16m2_t _r3 = __riscv_vle16_v_i16m2(tmp[m][3], vl);

                    __riscv_vsse16_v_i16m2(BT.row<short>(m * 4) + offset, max_jj_block * sizeof(short), __riscv_vsub_vv_i16m2(_r0, _r2, vl), vl);
                    __riscv_vsse16_v_i16m2(BT.row<short>(m * 4 + 1) + offset, max_jj_block * sizeof(short), __riscv_vadd_vv_i16m2(_r1, _r2, vl), vl);
                    __riscv_vsse16_v_i16m2(BT.row<short>(m * 4 + 2) + offset, max_jj_block * sizeof(short), __riscv_vsub_vv_i16m2(_r2, _r1, vl), vl);
                    __riscv_vsse16_v_i16m2(BT.row<short>(m * 4 + 3) + offset, max_jj_block * sizeof(short), __riscv_vsub_vv_i16m2(_r3, _r1, vl), vl);
                }

                kk += vl;
            }
#else
            for (; kk < max_kk; kk++)
            {
                short tmp[4][4];

                const int q = k + kk;
                for (int m = 0; m < 4; m++)
                {
                    signed char r0 = 0;
                    signed char r1 = 0;
                    signed char r2 = 0;
                    signed char r3 = 0;

                    if (ti * 2 + m < h && tj * 2 < w)
                    {
                        const signed char* rptr = bottom_blob.channel(q / elempack).row<const signed char>(ti * 2 + m) + tj * 2 * elempack + q % elempack;
                        r0 = rptr[0];
                        if (tj * 2 + 1 < w) r1 = rptr[elempack];
                        if (tj * 2 + 2 < w) r2 = rptr[elempack * 2];
                        if (tj * 2 + 3 < w) r3 = rptr[elempack * 3];
                    }

                    tmp[0][m] = r0 - r2;
                    tmp[1][m] = r1 + r2;
                    tmp[2][m] = r2 - r1;
                    tmp[3][m] = r3 - r1;
                }

                const int offset = jj * max_kk + kk * max_jj_block + jjj;
                for (int m = 0; m < 4; m++)
                {
                    short r0 = tmp[m][0];
                    short r1 = tmp[m][1];
                    short r2 = tmp[m][2];
                    short r3 = tmp[m][3];

                    BT.row<short>(m * 4)[offset] = r0 - r2;
                    BT.row<short>(m * 4 + 1)[offset] = r1 + r2;
                    BT.row<short>(m * 4 + 2)[offset] = r2 - r1;
                    BT.row<short>(m * 4 + 3)[offset] = r3 - r1;
                }
            }
#endif // __riscv_vector
        }

        jj += max_jj_block;
    }
}

static inline void conv3x3s1_winograd23_transform_output_tile_int8_rvv(const Mat& top_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    // const int otm[2][4] = {
    //     {1,  1,  1,  0},
    //     {0,  1, -1,  1}
    // };

    // The kernel transform uses 2G, so the exact result is divided by four.
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int w_tiles = (outw + 1) / 2;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t out_hstep = top_blob.cstep;

    while (ii < max_ii)
    {
        const int max_ii_block = std::min(max_ii - ii, packn);
        const size_t vl = __riscv_vsetvl_e32m2(max_ii_block);

        int tmp[2][4][packn];

        for (int jj = 0; jj < max_jj; jj++)
        {
            const int ti = (j + jj) / w_tiles;
            const int tj = (j + jj) % w_tiles;

            for (int m = 0; m < 4; m++)
            {
                const int* r0 = (const int*)top_tile + ii * max_jj * 16 + (m * 4 * max_jj + jj) * max_ii_block;
                const int* r1 = r0 + max_jj * max_ii_block;
                const int* r2 = r0 + max_jj * max_ii_block * 2;
                const int* r3 = r0 + max_jj * max_ii_block * 3;

                vint32m2_t _r0 = __riscv_vle32_v_i32m2(r0, vl);
                vint32m2_t _r1 = __riscv_vle32_v_i32m2(r1, vl);
                vint32m2_t _r2 = __riscv_vle32_v_i32m2(r2, vl);
                vint32m2_t _r3 = __riscv_vle32_v_i32m2(r3, vl);

                vint32m2_t _tmp0 = __riscv_vadd_vv_i32m2(__riscv_vadd_vv_i32m2(_r0, _r1, vl), _r2, vl);
                vint32m2_t _tmp1 = __riscv_vadd_vv_i32m2(__riscv_vsub_vv_i32m2(_r1, _r2, vl), _r3, vl);
                __riscv_vse32_v_i32m2(tmp[0][m], _tmp0, vl);
                __riscv_vse32_v_i32m2(tmp[1][m], _tmp1, vl);
            }

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh || tj * 2 >= outw)
                    continue;

                vint32m2_t _r0 = __riscv_vle32_v_i32m2(tmp[m][0], vl);
                vint32m2_t _r1 = __riscv_vle32_v_i32m2(tmp[m][1], vl);
                vint32m2_t _r2 = __riscv_vle32_v_i32m2(tmp[m][2], vl);
                vint32m2_t _r3 = __riscv_vle32_v_i32m2(tmp[m][3], vl);

                vint32m2_t _tmp0 = __riscv_vsra_vx_i32m2(__riscv_vadd_vv_i32m2(__riscv_vadd_vv_i32m2(_r0, _r1, vl), _r2, vl), 2, vl);
                vint32m2_t _tmp1 = __riscv_vsra_vx_i32m2(__riscv_vadd_vv_i32m2(__riscv_vsub_vv_i32m2(_r1, _r2, vl), _r3, vl), 2, vl);

                if (out_elempack == 1)
                {
                    int* outptr = top_blob.channel(i + ii).row<int>(ti * 2 + m) + tj * 2;
                    __riscv_vsse32_v_i32m2(outptr, out_hstep * sizeof(int), _tmp0, vl);
                    if (tj * 2 + 1 < outw)
                        __riscv_vsse32_v_i32m2(outptr + 1, out_hstep * sizeof(int), _tmp1, vl);
                }
                else
                {
                    // TILE_M and the GEMM blocks are aligned to packn channels.
                    const size_t vl0 = __riscv_vsetvl_e32m1(std::min(max_ii_block, out_elempack));
                    int* outptr0 = top_blob.channel((i + ii) / out_elempack).row<int>(ti * 2 + m) + tj * 2 * out_elempack;
                    __riscv_vse32_v_i32m1(outptr0, __riscv_vget_v_i32m2_i32m1(_tmp0, 0), vl0);
                    if (tj * 2 + 1 < outw)
                        __riscv_vse32_v_i32m1(outptr0 + out_elempack, __riscv_vget_v_i32m2_i32m1(_tmp1, 0), vl0);

                    if (max_ii_block > out_elempack)
                    {
                        const size_t vl1 = __riscv_vsetvl_e32m1(max_ii_block - out_elempack);
                        int* outptr1 = top_blob.channel((i + ii) / out_elempack + 1).row<int>(ti * 2 + m) + tj * 2 * out_elempack;
                        __riscv_vse32_v_i32m1(outptr1, __riscv_vget_v_i32m2_i32m1(_tmp0, 1), vl1);
                        if (tj * 2 + 1 < outw)
                            __riscv_vse32_v_i32m1(outptr1 + out_elempack, __riscv_vget_v_i32m2_i32m1(_tmp1, 1), vl1);
                    }
                }
            }
        }

        ii += max_ii_block;
    }
#else
    for (; ii + 1 < max_ii; ii += 2)
    {
        int tmp[2][4][2];

        for (int jj = 0; jj < max_jj; jj++)
        {
            const int ti = (j + jj) / w_tiles;
            const int tj = (j + jj) % w_tiles;

            for (int m = 0; m < 4; m++)
            {
                const int* r0 = (const int*)top_tile + ii * max_jj * 16 + m * max_jj * 8 + jj * 2;
                const int* r1 = r0 + max_jj * 2;
                const int* r2 = r0 + max_jj * 4;
                const int* r3 = r0 + max_jj * 6;

                tmp[0][m][0] = r0[0] + r1[0] + r2[0];
                tmp[0][m][1] = r0[1] + r1[1] + r2[1];
                tmp[1][m][0] = r1[0] - r2[0] + r3[0];
                tmp[1][m][1] = r1[1] - r2[1] + r3[1];
            }

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh || tj * 2 >= outw)
                    continue;

                const int tmp00 = (tmp[m][0][0] + tmp[m][1][0] + tmp[m][2][0]) / 4;
                const int tmp01 = (tmp[m][0][1] + tmp[m][1][1] + tmp[m][2][1]) / 4;
                const int tmp10 = (tmp[m][1][0] - tmp[m][2][0] + tmp[m][3][0]) / 4;
                const int tmp11 = (tmp[m][1][1] - tmp[m][2][1] + tmp[m][3][1]) / 4;

                int* outptr0 = top_blob.channel((i + ii) / out_elempack).row<int>(ti * 2 + m) + tj * 2 * out_elempack + (i + ii) % out_elempack;
                int* outptr1 = top_blob.channel((i + ii + 1) / out_elempack).row<int>(ti * 2 + m) + tj * 2 * out_elempack + (i + ii + 1) % out_elempack;
                outptr0[0] = tmp00;
                outptr1[0] = tmp01;
                if (tj * 2 + 1 < outw)
                {
                    outptr0[out_elempack] = tmp10;
                    outptr1[out_elempack] = tmp11;
                }
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        int tmp[2][4];

        for (int jj = 0; jj < max_jj; jj++)
        {
            const int ti = (j + jj) / w_tiles;
            const int tj = (j + jj) % w_tiles;

            for (int m = 0; m < 4; m++)
            {
                const int* r0 = (const int*)top_tile + ii * max_jj * 16 + m * max_jj * 4 + jj;
                const int* r1 = r0 + max_jj;
                const int* r2 = r0 + max_jj * 2;
                const int* r3 = r0 + max_jj * 3;

                tmp[0][m] = r0[0] + r1[0] + r2[0];
                tmp[1][m] = r1[0] - r2[0] + r3[0];
            }

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh || tj * 2 >= outw)
                    continue;

                const int tmp0 = (tmp[m][0] + tmp[m][1] + tmp[m][2]) / 4;
                const int tmp1 = (tmp[m][1] - tmp[m][2] + tmp[m][3]) / 4;

                int* outptr = top_blob.channel((i + ii) / out_elempack).row<int>(ti * 2 + m) + tj * 2 * out_elempack + (i + ii) % out_elempack;
                outptr[0] = tmp0;
                if (tj * 2 + 1 < outw)
                    outptr[out_elempack] = tmp1;
            }
        }
    }
#endif // __riscv_vector
}

static int conv3x3s1_winograd23_int8_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt)
{
    const int w_tiles = (top_blob.w + 1) / 2;
    const int h_tiles = (top_blob.h + 1) / 2;

    const int M = top_blob.c * top_blob.elempack;
    const int N = w_tiles * h_tiles;
    const int K = bottom_blob.c * bottom_blob.elempack;
    const int B = 16;

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk_int8_rvv(M, N, K, B, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // Reuse each transformed input panel across a group of output channels.
    // Keep the per-worker panel and all split-K partial sums cache-bounded.
    int nn_Mgroup = std::min(nn_M, (nT + nn_N - 1) / nn_N);
    const size_t l2_cache_size = std::max(16384, (int)get_cpu_level2_cache_size());
    const size_t bt_tile_size = (size_t)TILE_K * TILE_N * B * sizeof(short);
    const size_t topT_tile_size = (size_t)TILE_M * TILE_N * B * sizeof(int);
    const size_t topT_budget = l2_cache_size > bt_tile_size ? l2_cache_size - bt_tile_size : 0;
    const int max_M_tiles = std::max(1, (int)(topT_budget / topT_tile_size));
    nn_Mgroup = std::max(nn_Mgroup, (nn_M + max_M_tiles - 1) / max_M_tiles);
    const int nn_MN = nn_Mgroup * nn_N;
    const int num_threads = std::min(nT, nn_MN);

    Mat BT_tileX(TILE_K * TILE_N, B, num_threads, (size_t)2u, 1, opt.workspace_allocator);
    if (BT_tileX.empty())
        return -100;

    const int nn_M_per_group = (nn_M + nn_Mgroup - 1) / nn_Mgroup;
    Mat topT_tileX(TILE_M * TILE_N * B, nn_M_per_group, num_threads, (size_t)4u, 1, opt.workspace_allocator);
    if (topT_tileX.empty())
        return -100;

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
            conv3x3s1_winograd23_transform_input_tile_int8_rvv(bottom_blob, BT_tile, j, max_jj, k, max_kk);

            for (int ppi = ppi_start; ppi < ppi_end; ppi++)
            {
                const int i = ppi * TILE_M;
                const int max_ii = std::min(M - i, TILE_M);
                const Mat AT_tile = AT.channel(ppi).depth(ppk);
                Mat top_tile = topT_tileX.channel(tid).row_range(ppi - ppi_start, 1);

                conv3x3s1_winograd_gemm_transB_packed_tile_int8_rvv(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk);

                if (ppk + 1 == nn_K)
                    conv3x3s1_winograd23_transform_output_tile_int8_rvv(top_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }

    return 0;
}

static void conv3x3s1_winograd43_transform_kernel_tile_int8_rvv(const Mat& kernel, Mat& AT, int inch, int i, int max_ii, int k, int max_kk)
{
    // const short ktm[6][3] = {
    //     {6, 0, 0},
    //     {-4, -4, -4},
    //     {-4, 4, -4},
    //     {1, 2, 4},
    //     {1, -2, 4},
    //     {0, 0, 6}
    // };

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
#else
    const int packn = 2;
#endif

    int ii = 0;
    while (ii < max_ii)
    {
        const int max_ii_block = std::min(max_ii - ii, packn);
        for (int kk = 0; kk < max_kk; kk++)
        {
            for (int iii = 0; iii < max_ii_block; iii++)
            {
                short tmp[6][3];
                const signed char* k0 = (const signed char*)kernel + (i + ii + iii) * inch * 9 + (k + kk) * 9;

                for (int m = 0; m < 3; m++)
                {
                    const int r0 = k0[0];
                    const int r1 = k0[1];
                    const int r2 = k0[2];

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
                    const int r0 = tmp[m][0];
                    const int r1 = tmp[m][1];
                    const int r2 = tmp[m][2];

                    const int offset = ii * max_kk + kk * max_ii_block + iii;
                    AT.row<short>(m * 6 + 0)[offset] = r0 * 6;
                    AT.row<short>(m * 6 + 1)[offset] = -r0 * 4 - r1 * 4 - r2 * 4;
                    AT.row<short>(m * 6 + 2)[offset] = -r0 * 4 + r1 * 4 - r2 * 4;
                    AT.row<short>(m * 6 + 3)[offset] = r0 + r1 * 2 + r2 * 4;
                    AT.row<short>(m * 6 + 4)[offset] = r0 - r1 * 2 + r2 * 4;
                    AT.row<short>(m * 6 + 5)[offset] = r2 * 6;
                }
            }
        }
        ii += max_ii_block;
    }
}

static int conv3x3s1_winograd43_transform_kernel_int8_rvv(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 36;

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk_int8_rvv(M, 0, K, B, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_K = (K + TILE_K - 1) / TILE_K;
    AT.create(TILE_M * TILE_K, B, nn_K, nn_M, (size_t)2u, 1);
    if (AT.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;
        const int max_ii = std::min(M - i, TILE_M);

        for (int ppk = 0; ppk < nn_K; ppk++)
        {
            const int k = ppk * TILE_K;
            const int max_kk = std::min(K - k, TILE_K);
            Mat AT_tile = AT.channel(ppi).depth(ppk);
            conv3x3s1_winograd43_transform_kernel_tile_int8_rvv(kernel, AT_tile, inch, i, max_ii, k, max_kk);
        }
    }

    return 0;
}

static inline void conv3x3s1_winograd43_transform_input_tile_int8_rvv(const Mat& bottom_blob, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    // const signed char itm[6][6] = {
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
    const int w_tiles = (w + 1) / 4;
#if __riscv_vector
    const int packn = csrr_vlenb();
    const size_t N = bottom_blob.cstep * elempack;
#endif

    // Fuse the transform and the BT packing. Each row contains jj blocks,
    // with the channels interleaved inside each block of spatial tiles.
    int jj = 0;
    while (jj < max_jj)
    {
        int max_jj_block = max_jj - jj >= 2 ? 2 : 1;
#if __riscv_vector
        if (max_jj - jj >= 4)
            max_jj_block = 4;
        if (max_jj - jj >= 8)
            max_jj_block = 8;
#endif

        for (int jjj = 0; jjj < max_jj_block; jjj++)
        {
            const int ti = (j + jj + jjj) / w_tiles;
            const int tj = (j + jj + jjj) % w_tiles;

            int kk = 0;
#if __riscv_vector
            while (kk < max_kk)
            {
                const int q = k + kk;
                const int lane = q % elempack;
                int max_vl = std::min(max_kk - kk, packn);
                if (elempack != 1)
                    max_vl = std::min(max_vl, elempack - lane);
                const size_t vl = __riscv_vsetvl_e8m1(max_vl);

                short tmp[6][6][packn];

                for (int m = 0; m < 6; m++)
                {
                    vint8m1_t _r0_8 = __riscv_vmv_v_x_i8m1(0, vl);
                    vint8m1_t _r1_8 = _r0_8;
                    vint8m1_t _r2_8 = _r0_8;
                    vint8m1_t _r3_8 = _r0_8;
                    vint8m1_t _r4_8 = _r0_8;
                    vint8m1_t _r5_8 = _r0_8;

                    if (ti * 4 + m < h && tj * 4 < w)
                    {
                        const signed char* rptr = bottom_blob.channel(q / elempack).row<const signed char>(ti * 4 + m) + tj * 4 * elempack + lane;
                        if (elempack != 1)
                        {
                            _r0_8 = __riscv_vle8_v_i8m1(rptr, vl);
                            if (tj * 4 + 1 < w) _r1_8 = __riscv_vle8_v_i8m1(rptr + elempack, vl);
                            if (tj * 4 + 2 < w) _r2_8 = __riscv_vle8_v_i8m1(rptr + elempack * 2, vl);
                            if (tj * 4 + 3 < w) _r3_8 = __riscv_vle8_v_i8m1(rptr + elempack * 3, vl);
                            if (tj * 4 + 4 < w) _r4_8 = __riscv_vle8_v_i8m1(rptr + elempack * 4, vl);
                            if (tj * 4 + 5 < w) _r5_8 = __riscv_vle8_v_i8m1(rptr + elempack * 5, vl);
                        }
                        else
                        {
                            _r0_8 = __riscv_vlse8_v_i8m1(rptr, N, vl);
                            if (tj * 4 + 1 < w) _r1_8 = __riscv_vlse8_v_i8m1(rptr + 1, N, vl);
                            if (tj * 4 + 2 < w) _r2_8 = __riscv_vlse8_v_i8m1(rptr + 2, N, vl);
                            if (tj * 4 + 3 < w) _r3_8 = __riscv_vlse8_v_i8m1(rptr + 3, N, vl);
                            if (tj * 4 + 4 < w) _r4_8 = __riscv_vlse8_v_i8m1(rptr + 4, N, vl);
                            if (tj * 4 + 5 < w) _r5_8 = __riscv_vlse8_v_i8m1(rptr + 5, N, vl);
                        }
                    }

                    vint16m2_t _r0 = __riscv_vwadd_vx_i16m2(_r0_8, 0, vl);
                    vint16m2_t _r1 = __riscv_vwadd_vx_i16m2(_r1_8, 0, vl);
                    vint16m2_t _r2 = __riscv_vwadd_vx_i16m2(_r2_8, 0, vl);
                    vint16m2_t _r3 = __riscv_vwadd_vx_i16m2(_r3_8, 0, vl);
                    vint16m2_t _r4 = __riscv_vwadd_vx_i16m2(_r4_8, 0, vl);
                    vint16m2_t _r5 = __riscv_vwadd_vx_i16m2(_r5_8, 0, vl);

                    vint16m2_t _tmp12a = __riscv_vsub_vv_i16m2(_r3, __riscv_vsll_vx_i16m2(_r1, 2, vl), vl);
                    vint16m2_t _tmp12b = __riscv_vsub_vv_i16m2(_r4, __riscv_vsll_vx_i16m2(_r2, 2, vl), vl);
                    vint16m2_t _tmp34a = __riscv_vsll_vx_i16m2(__riscv_vsub_vv_i16m2(_r3, _r1, vl), 1, vl);
                    vint16m2_t _tmp34b = __riscv_vsub_vv_i16m2(_r4, _r2, vl);

                    vint16m2_t _tmp0 = __riscv_vsub_vv_i16m2(__riscv_vadd_vv_i16m2(_r4, __riscv_vsll_vx_i16m2(_r0, 2, vl), vl), __riscv_vmul_vx_i16m2(_r2, 5, vl), vl);
                    vint16m2_t _tmp1 = __riscv_vadd_vv_i16m2(_tmp12b, _tmp12a, vl);
                    vint16m2_t _tmp2 = __riscv_vsub_vv_i16m2(_tmp12b, _tmp12a, vl);
                    vint16m2_t _tmp3 = __riscv_vadd_vv_i16m2(_tmp34b, _tmp34a, vl);
                    vint16m2_t _tmp4 = __riscv_vsub_vv_i16m2(_tmp34b, _tmp34a, vl);
                    vint16m2_t _tmp5 = __riscv_vsub_vv_i16m2(__riscv_vadd_vv_i16m2(_r5, __riscv_vsll_vx_i16m2(_r1, 2, vl), vl), __riscv_vmul_vx_i16m2(_r3, 5, vl), vl);

                    __riscv_vse16_v_i16m2(tmp[0][m], _tmp0, vl);
                    __riscv_vse16_v_i16m2(tmp[1][m], _tmp1, vl);
                    __riscv_vse16_v_i16m2(tmp[2][m], _tmp2, vl);
                    __riscv_vse16_v_i16m2(tmp[3][m], _tmp3, vl);
                    __riscv_vse16_v_i16m2(tmp[4][m], _tmp4, vl);
                    __riscv_vse16_v_i16m2(tmp[5][m], _tmp5, vl);
                }

                const int offset = jj * max_kk + kk * max_jj_block + jjj;
                for (int m = 0; m < 6; m++)
                {
                    vint16m2_t _r0 = __riscv_vle16_v_i16m2(tmp[m][0], vl);
                    vint16m2_t _r1 = __riscv_vle16_v_i16m2(tmp[m][1], vl);
                    vint16m2_t _r2 = __riscv_vle16_v_i16m2(tmp[m][2], vl);
                    vint16m2_t _r3 = __riscv_vle16_v_i16m2(tmp[m][3], vl);
                    vint16m2_t _r4 = __riscv_vle16_v_i16m2(tmp[m][4], vl);
                    vint16m2_t _r5 = __riscv_vle16_v_i16m2(tmp[m][5], vl);

                    vint16m2_t _tmp12a = __riscv_vsub_vv_i16m2(_r3, __riscv_vsll_vx_i16m2(_r1, 2, vl), vl);
                    vint16m2_t _tmp12b = __riscv_vsub_vv_i16m2(_r4, __riscv_vsll_vx_i16m2(_r2, 2, vl), vl);
                    vint16m2_t _tmp34a = __riscv_vsll_vx_i16m2(__riscv_vsub_vv_i16m2(_r3, _r1, vl), 1, vl);
                    vint16m2_t _tmp34b = __riscv_vsub_vv_i16m2(_r4, _r2, vl);

                    vint16m2_t _tmp0 = __riscv_vsub_vv_i16m2(__riscv_vadd_vv_i16m2(_r4, __riscv_vsll_vx_i16m2(_r0, 2, vl), vl), __riscv_vmul_vx_i16m2(_r2, 5, vl), vl);
                    vint16m2_t _tmp1 = __riscv_vadd_vv_i16m2(_tmp12b, _tmp12a, vl);
                    vint16m2_t _tmp2 = __riscv_vsub_vv_i16m2(_tmp12b, _tmp12a, vl);
                    vint16m2_t _tmp3 = __riscv_vadd_vv_i16m2(_tmp34b, _tmp34a, vl);
                    vint16m2_t _tmp4 = __riscv_vsub_vv_i16m2(_tmp34b, _tmp34a, vl);
                    vint16m2_t _tmp5 = __riscv_vsub_vv_i16m2(__riscv_vadd_vv_i16m2(_r5, __riscv_vsll_vx_i16m2(_r1, 2, vl), vl), __riscv_vmul_vx_i16m2(_r3, 5, vl), vl);

                    __riscv_vsse16_v_i16m2(BT.row<short>(m * 6) + offset, max_jj_block * sizeof(short), _tmp0, vl);
                    __riscv_vsse16_v_i16m2(BT.row<short>(m * 6 + 1) + offset, max_jj_block * sizeof(short), _tmp1, vl);
                    __riscv_vsse16_v_i16m2(BT.row<short>(m * 6 + 2) + offset, max_jj_block * sizeof(short), _tmp2, vl);
                    __riscv_vsse16_v_i16m2(BT.row<short>(m * 6 + 3) + offset, max_jj_block * sizeof(short), _tmp3, vl);
                    __riscv_vsse16_v_i16m2(BT.row<short>(m * 6 + 4) + offset, max_jj_block * sizeof(short), _tmp4, vl);
                    __riscv_vsse16_v_i16m2(BT.row<short>(m * 6 + 5) + offset, max_jj_block * sizeof(short), _tmp5, vl);
                }

                kk += vl;
            }
#else
            for (; kk < max_kk; kk++)
            {
                short tmp[6][6];

                const int q = k + kk;
                for (int m = 0; m < 6; m++)
                {
                    signed char r0 = 0;
                    signed char r1 = 0;
                    signed char r2 = 0;
                    signed char r3 = 0;
                    signed char r4 = 0;
                    signed char r5 = 0;

                    if (ti * 4 + m < h && tj * 4 < w)
                    {
                        const signed char* rptr = bottom_blob.channel(q / elempack).row<const signed char>(ti * 4 + m) + tj * 4 * elempack + q % elempack;
                        r0 = rptr[0];
                        if (tj * 4 + 1 < w) r1 = rptr[elempack];
                        if (tj * 4 + 2 < w) r2 = rptr[elempack * 2];
                        if (tj * 4 + 3 < w) r3 = rptr[elempack * 3];
                        if (tj * 4 + 4 < w) r4 = rptr[elempack * 4];
                        if (tj * 4 + 5 < w) r5 = rptr[elempack * 5];
                    }

                    const short tmp12a = r3 - r1 * 4;
                    const short tmp12b = r4 - r2 * 4;
                    const short tmp34a = (r3 - r1) * 2;
                    const short tmp34b = r4 - r2;

                    tmp[0][m] = r4 + r0 * 4 - r2 * 5;
                    tmp[1][m] = tmp12b + tmp12a;
                    tmp[2][m] = tmp12b - tmp12a;
                    tmp[3][m] = tmp34b + tmp34a;
                    tmp[4][m] = tmp34b - tmp34a;
                    tmp[5][m] = r5 + r1 * 4 - r3 * 5;
                }

                const int offset = jj * max_kk + kk * max_jj_block + jjj;
                for (int m = 0; m < 6; m++)
                {
                    const short r0 = tmp[m][0];
                    const short r1 = tmp[m][1];
                    const short r2 = tmp[m][2];
                    const short r3 = tmp[m][3];
                    const short r4 = tmp[m][4];
                    const short r5 = tmp[m][5];

                    const short tmp12a = r3 - r1 * 4;
                    const short tmp12b = r4 - r2 * 4;
                    const short tmp34a = (r3 - r1) * 2;
                    const short tmp34b = r4 - r2;

                    BT.row<short>(m * 6)[offset] = r4 + r0 * 4 - r2 * 5;
                    BT.row<short>(m * 6 + 1)[offset] = tmp12b + tmp12a;
                    BT.row<short>(m * 6 + 2)[offset] = tmp12b - tmp12a;
                    BT.row<short>(m * 6 + 3)[offset] = tmp34b + tmp34a;
                    BT.row<short>(m * 6 + 4)[offset] = tmp34b - tmp34a;
                    BT.row<short>(m * 6 + 5)[offset] = r5 + r1 * 4 - r3 * 5;
                }
            }
#endif // __riscv_vector
        }

        jj += max_jj_block;
    }
}

static inline void conv3x3s1_winograd43_transform_output_tile_int8_rvv(const Mat& top_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    // The combined transforms scale the output by 576.
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int w_tiles = (outw + 3) / 4;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t out_hstep = top_blob.cstep;

    while (ii < max_ii)
    {
        const int max_ii_block = std::min(max_ii - ii, packn);
        const size_t vl = __riscv_vsetvl_e32m2(max_ii_block);
        int tmp[4][6][packn];

        for (int jj = 0; jj < max_jj; jj++)
        {
            const int ti = (j + jj) / w_tiles;
            const int tj = (j + jj) % w_tiles;

            for (int m = 0; m < 6; m++)
            {
                const int* r0 = (const int*)top_tile + ii * max_jj * 36 + (m * 6 * max_jj + jj) * max_ii_block;
                const int* r1 = r0 + max_jj * max_ii_block;
                const int* r2 = r0 + max_jj * max_ii_block * 2;
                const int* r3 = r0 + max_jj * max_ii_block * 3;
                const int* r4 = r0 + max_jj * max_ii_block * 4;
                const int* r5 = r0 + max_jj * max_ii_block * 5;

                vint32m2_t _r0 = __riscv_vle32_v_i32m2(r0, vl);
                vint32m2_t _r1 = __riscv_vle32_v_i32m2(r1, vl);
                vint32m2_t _r2 = __riscv_vle32_v_i32m2(r2, vl);
                vint32m2_t _r3 = __riscv_vle32_v_i32m2(r3, vl);
                vint32m2_t _r4 = __riscv_vle32_v_i32m2(r4, vl);
                vint32m2_t _r5 = __riscv_vle32_v_i32m2(r5, vl);

                vint32m2_t _tmp12a = __riscv_vsub_vv_i32m2(_r1, _r2, vl);
                vint32m2_t _tmp12b = __riscv_vadd_vv_i32m2(_r1, _r2, vl);
                vint32m2_t _tmp34a = __riscv_vsub_vv_i32m2(_r3, _r4, vl);
                vint32m2_t _tmp34b = __riscv_vadd_vv_i32m2(_r3, _r4, vl);

                vint32m2_t _tmp0 = __riscv_vadd_vv_i32m2(__riscv_vadd_vv_i32m2(_r0, _tmp12b, vl), _tmp34b, vl);
                vint32m2_t _tmp1 = __riscv_vadd_vv_i32m2(_tmp12a, __riscv_vsll_vx_i32m2(_tmp34a, 1, vl), vl);
                vint32m2_t _tmp2 = __riscv_vadd_vv_i32m2(_tmp12b, __riscv_vsll_vx_i32m2(_tmp34b, 2, vl), vl);
                vint32m2_t _tmp3 = __riscv_vadd_vv_i32m2(__riscv_vadd_vv_i32m2(_tmp12a, __riscv_vsll_vx_i32m2(_tmp34a, 3, vl), vl), __riscv_vsll_vx_i32m2(_r5, 2, vl), vl);
                if (m == 5)
                {
                    _tmp0 = __riscv_vsll_vx_i32m2(_tmp0, 2, vl);
                    _tmp1 = __riscv_vsll_vx_i32m2(_tmp1, 2, vl);
                    _tmp2 = __riscv_vsll_vx_i32m2(_tmp2, 2, vl);
                    _tmp3 = __riscv_vsll_vx_i32m2(_tmp3, 2, vl);
                }

                __riscv_vse32_v_i32m2(tmp[0][m], _tmp0, vl);
                __riscv_vse32_v_i32m2(tmp[1][m], _tmp1, vl);
                __riscv_vse32_v_i32m2(tmp[2][m], _tmp2, vl);
                __riscv_vse32_v_i32m2(tmp[3][m], _tmp3, vl);
            }

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh || tj * 4 >= outw)
                    continue;

                vint32m2_t _r0 = __riscv_vle32_v_i32m2(tmp[m][0], vl);
                vint32m2_t _r1 = __riscv_vle32_v_i32m2(tmp[m][1], vl);
                vint32m2_t _r2 = __riscv_vle32_v_i32m2(tmp[m][2], vl);
                vint32m2_t _r3 = __riscv_vle32_v_i32m2(tmp[m][3], vl);
                vint32m2_t _r4 = __riscv_vle32_v_i32m2(tmp[m][4], vl);
                vint32m2_t _r5 = __riscv_vle32_v_i32m2(tmp[m][5], vl);

                vint32m2_t _tmp12a = __riscv_vsub_vv_i32m2(_r1, _r2, vl);
                vint32m2_t _tmp12b = __riscv_vadd_vv_i32m2(_r1, _r2, vl);
                vint32m2_t _tmp34a = __riscv_vsub_vv_i32m2(_r3, _r4, vl);
                vint32m2_t _tmp34b = __riscv_vadd_vv_i32m2(_r3, _r4, vl);

                vint32m2_t _tmp0 = __riscv_vadd_vv_i32m2(__riscv_vadd_vv_i32m2(_r0, _tmp12b, vl), _tmp34b, vl);
                vint32m2_t _tmp1 = __riscv_vadd_vv_i32m2(_tmp12a, __riscv_vsll_vx_i32m2(_tmp34a, 1, vl), vl);
                vint32m2_t _tmp2 = __riscv_vadd_vv_i32m2(_tmp12b, __riscv_vsll_vx_i32m2(_tmp34b, 2, vl), vl);
                vint32m2_t _tmp3 = __riscv_vadd_vv_i32m2(__riscv_vadd_vv_i32m2(_tmp12a, __riscv_vsll_vx_i32m2(_tmp34a, 3, vl), vl), _r5, vl);

#if __riscv_xtheadvector
                _tmp0 = __riscv_vfcvt_x_f_v_i32m2_rm(__riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(_tmp0, vl), 1.f / 576, vl), __RISCV_FRM_RTZ, vl);
                _tmp1 = __riscv_vfcvt_x_f_v_i32m2_rm(__riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(_tmp1, vl), 1.f / 576, vl), __RISCV_FRM_RTZ, vl);
                _tmp2 = __riscv_vfcvt_x_f_v_i32m2_rm(__riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(_tmp2, vl), 1.f / 576, vl), __RISCV_FRM_RTZ, vl);
                _tmp3 = __riscv_vfcvt_x_f_v_i32m2_rm(__riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(_tmp3, vl), 1.f / 576, vl), __RISCV_FRM_RTZ, vl);
#else
                _tmp0 = __riscv_vfcvt_rtz_x_f_v_i32m2(__riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(_tmp0, vl), 1.f / 576, vl), vl);
                _tmp1 = __riscv_vfcvt_rtz_x_f_v_i32m2(__riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(_tmp1, vl), 1.f / 576, vl), vl);
                _tmp2 = __riscv_vfcvt_rtz_x_f_v_i32m2(__riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(_tmp2, vl), 1.f / 576, vl), vl);
                _tmp3 = __riscv_vfcvt_rtz_x_f_v_i32m2(__riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(_tmp3, vl), 1.f / 576, vl), vl);
#endif

                if (out_elempack == 1)
                {
                    int* outptr = top_blob.channel(i + ii).row<int>(ti * 4 + m) + tj * 4;
                    __riscv_vsse32_v_i32m2(outptr, out_hstep * sizeof(int), _tmp0, vl);
                    if (tj * 4 + 1 < outw)
                        __riscv_vsse32_v_i32m2(outptr + 1, out_hstep * sizeof(int), _tmp1, vl);
                    if (tj * 4 + 2 < outw)
                        __riscv_vsse32_v_i32m2(outptr + 2, out_hstep * sizeof(int), _tmp2, vl);
                    if (tj * 4 + 3 < outw)
                        __riscv_vsse32_v_i32m2(outptr + 3, out_hstep * sizeof(int), _tmp3, vl);
                }
                else
                {
                    const size_t vl0 = __riscv_vsetvl_e32m1(std::min(max_ii_block, out_elempack));
                    int* outptr0 = top_blob.channel((i + ii) / out_elempack).row<int>(ti * 4 + m) + tj * 4 * out_elempack;
                    __riscv_vse32_v_i32m1(outptr0, __riscv_vget_v_i32m2_i32m1(_tmp0, 0), vl0);
                    if (tj * 4 + 1 < outw)
                        __riscv_vse32_v_i32m1(outptr0 + out_elempack, __riscv_vget_v_i32m2_i32m1(_tmp1, 0), vl0);
                    if (tj * 4 + 2 < outw)
                        __riscv_vse32_v_i32m1(outptr0 + out_elempack * 2, __riscv_vget_v_i32m2_i32m1(_tmp2, 0), vl0);
                    if (tj * 4 + 3 < outw)
                        __riscv_vse32_v_i32m1(outptr0 + out_elempack * 3, __riscv_vget_v_i32m2_i32m1(_tmp3, 0), vl0);

                    if (max_ii_block > out_elempack)
                    {
                        const size_t vl1 = __riscv_vsetvl_e32m1(max_ii_block - out_elempack);
                        int* outptr1 = top_blob.channel((i + ii) / out_elempack + 1).row<int>(ti * 4 + m) + tj * 4 * out_elempack;
                        __riscv_vse32_v_i32m1(outptr1, __riscv_vget_v_i32m2_i32m1(_tmp0, 1), vl1);
                        if (tj * 4 + 1 < outw)
                            __riscv_vse32_v_i32m1(outptr1 + out_elempack, __riscv_vget_v_i32m2_i32m1(_tmp1, 1), vl1);
                        if (tj * 4 + 2 < outw)
                            __riscv_vse32_v_i32m1(outptr1 + out_elempack * 2, __riscv_vget_v_i32m2_i32m1(_tmp2, 1), vl1);
                        if (tj * 4 + 3 < outw)
                            __riscv_vse32_v_i32m1(outptr1 + out_elempack * 3, __riscv_vget_v_i32m2_i32m1(_tmp3, 1), vl1);
                    }
                }
            }
        }

        ii += max_ii_block;
    }
#else
    while (ii < max_ii)
    {
        const int max_ii_block = std::min(max_ii - ii, 2);
        unsigned int tmp[4][6][2];

        for (int jj = 0; jj < max_jj; jj++)
        {
            const int ti = (j + jj) / w_tiles;
            const int tj = (j + jj) % w_tiles;

            for (int m = 0; m < 6; m++)
            {
                for (int iii = 0; iii < max_ii_block; iii++)
                {
                    const unsigned int* r0 = (const unsigned int*)top_tile + ii * max_jj * 36 + (m * 6 * max_jj + jj) * max_ii_block + iii;
                    const unsigned int* r1 = r0 + max_jj * max_ii_block;
                    const unsigned int* r2 = r0 + max_jj * max_ii_block * 2;
                    const unsigned int* r3 = r0 + max_jj * max_ii_block * 3;
                    const unsigned int* r4 = r0 + max_jj * max_ii_block * 4;
                    const unsigned int* r5 = r0 + max_jj * max_ii_block * 5;

                    const unsigned int tmp12a = r1[0] - r2[0];
                    const unsigned int tmp12b = r1[0] + r2[0];
                    const unsigned int tmp34a = r3[0] - r4[0];
                    const unsigned int tmp34b = r3[0] + r4[0];

                    tmp[0][m][iii] = r0[0] + tmp12b + tmp34b;
                    tmp[1][m][iii] = tmp12a + tmp34a * 2;
                    tmp[2][m][iii] = tmp12b + tmp34b * 4;
                    tmp[3][m][iii] = tmp12a + tmp34a * 8 + r5[0] * 4;
                    if (m == 5)
                    {
                        tmp[0][m][iii] *= 4;
                        tmp[1][m][iii] *= 4;
                        tmp[2][m][iii] *= 4;
                        tmp[3][m][iii] *= 4;
                    }
                }
            }

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh || tj * 4 >= outw)
                    continue;

                for (int iii = 0; iii < max_ii_block; iii++)
                {
                    const unsigned int tmp12a = tmp[m][1][iii] - tmp[m][2][iii];
                    const unsigned int tmp12b = tmp[m][1][iii] + tmp[m][2][iii];
                    const unsigned int tmp34a = tmp[m][3][iii] - tmp[m][4][iii];
                    const unsigned int tmp34b = tmp[m][3][iii] + tmp[m][4][iii];

                    const unsigned int sum0 = tmp[m][0][iii] + tmp12b + tmp34b;
                    const unsigned int sum1 = tmp12a + tmp34a * 2;
                    const unsigned int sum2 = tmp12b + tmp34b * 4;
                    const unsigned int sum3 = tmp12a + tmp34a * 8 + tmp[m][5][iii];

                    const int tmp0 = (int)sum0 / 576;
                    const int tmp1 = (int)sum1 / 576;
                    const int tmp2 = (int)sum2 / 576;
                    const int tmp3 = (int)sum3 / 576;

                    const int q = i + ii + iii;
                    int* outptr = top_blob.channel(q / out_elempack).row<int>(ti * 4 + m) + tj * 4 * out_elempack + q % out_elempack;
                    outptr[0] = tmp0;
                    if (tj * 4 + 1 < outw)
                        outptr[out_elempack] = tmp1;
                    if (tj * 4 + 2 < outw)
                        outptr[out_elempack * 2] = tmp2;
                    if (tj * 4 + 3 < outw)
                        outptr[out_elempack * 3] = tmp3;
                }
            }
        }

        ii += max_ii_block;
    }
#endif // __riscv_vector
}

static int conv3x3s1_winograd43_int8_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt)
{
    const int w_tiles = (top_blob.w + 3) / 4;
    const int h_tiles = (top_blob.h + 3) / 4;

    const int M = top_blob.c * top_blob.elempack;
    const int N = w_tiles * h_tiles;
    const int K = bottom_blob.c * bottom_blob.elempack;
    const int B = 36;

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk_int8_rvv(M, N, K, B, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // Reuse each transformed input panel across a group of output channels.
    // Keep the per-worker panel and all split-K partial sums cache-bounded.
    int nn_Mgroup = std::min(nn_M, (nT + nn_N - 1) / nn_N);
    const size_t l2_cache_size = std::max(16384, (int)get_cpu_level2_cache_size());
    const size_t bt_tile_size = (size_t)TILE_K * TILE_N * B * sizeof(short);
    const size_t topT_tile_size = (size_t)TILE_M * TILE_N * B * sizeof(int);
    const size_t topT_budget = l2_cache_size > bt_tile_size ? l2_cache_size - bt_tile_size : 0;
    const int max_M_tiles = std::max(1, (int)(topT_budget / topT_tile_size));
    nn_Mgroup = std::max(nn_Mgroup, (nn_M + max_M_tiles - 1) / max_M_tiles);
    const int nn_MN = nn_Mgroup * nn_N;
    const int num_threads = std::min(nT, nn_MN);

    Mat BT_tileX(TILE_K * TILE_N, B, num_threads, (size_t)2u, 1, opt.workspace_allocator);
    if (BT_tileX.empty())
        return -100;

    const int nn_M_per_group = (nn_M + nn_Mgroup - 1) / nn_Mgroup;
    Mat topT_tileX(TILE_M * TILE_N * B, nn_M_per_group, num_threads, (size_t)4u, 1, opt.workspace_allocator);
    if (topT_tileX.empty())
        return -100;

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
            conv3x3s1_winograd43_transform_input_tile_int8_rvv(bottom_blob, BT_tile, j, max_jj, k, max_kk);

            for (int ppi = ppi_start; ppi < ppi_end; ppi++)
            {
                const int i = ppi * TILE_M;
                const int max_ii = std::min(M - i, TILE_M);
                const Mat AT_tile = AT.channel(ppi).depth(ppk);
                Mat top_tile = topT_tileX.channel(tid).row_range(ppi - ppi_start, 1);

                conv3x3s1_winograd_gemm_transB_packed_tile_int8_rvv(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk);

                if (ppk + 1 == nn_K)
                    conv3x3s1_winograd43_transform_output_tile_int8_rvv(top_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }

    return 0;
}

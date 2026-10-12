// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convolution_im2col_gemm_get_optimal_tile_mnk_int8_rvv(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    if (nT == 0)
        nT = get_physical_big_cpu_count();

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
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
    int tile_size = (l2_cache_size - TILE_M * 8 * 4) / (TILE_M + 8);
    TILE_K = std::max(8, tile_size / 8 * 8);
    const int nn_K = (K + TILE_K - 1) / TILE_K;
    TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);

    TILE_N = 0;
    if (N > 0)
    {
        tile_size = (l2_cache_size - TILE_M * TILE_K) / (TILE_K + TILE_M * 4);
        TILE_N = std::max(8, tile_size / 8 * 8);
        int nn_N = (N + TILE_N - 1) / TILE_N;
        nn_N = std::max(nn_N, std::min(nT, (N + 7) / 8));
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8);
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
    // Interleave output channels so one weight vector serves eight positions.
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;
        const int max_ii = std::min(M - i, TILE_M);

        for (int ppk = 0; ppk < nn_K; ppk++)
        {
            const int k = ppk * TILE_K;
            const int max_kk = std::min(K - k, TILE_K);
            signed char* pp = AT.channel(ppi).row<signed char>(ppk);

            int ii = 0;
#if __riscv_vector
            const int packn = csrr_vlenb() / 4;
            for (; ii < max_ii; ii += packn)
            {
                const size_t vl = __riscv_vsetvl_e32m1(std::min(max_ii - ii, packn));
                const signed char* pA = (const signed char*)kernel + (i + ii) * K + k;
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vint8m1_t _a = __riscv_vlse8_v_i8m1(pA + kk, K, vl);
                    __riscv_vse8_v_i8m1(pp, _a, vl);
                    pp += vl;
                }
            }
#else
            for (; ii < max_ii; ii++)
            {
                const signed char* pA = (const signed char*)kernel + (i + ii) * K + k;
                memcpy(pp, pA, max_kk);
                pp += max_kk;
            }
#endif // __riscv_vector
        }
    }

    return 0;
}

static void convolution_im2col_input_tile_conv1x1s1d1_int8_rvv(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    const int elempack = bottom_blob.elempack;
    signed char* pp = B;

    int jj = 0;
    while (jj < max_jj)
    {
        const int nn = max_jj - jj >= 8 ? 8 : max_jj - jj >= 4 ? 4 : max_jj - jj >= 2 ? 2 : 1;
        for (int kk = 0; kk < max_kk; kk++)
        {
            const int q = k + kk;
            const signed char* pB = bottom_blob.channel(q / elempack);
            pB += (j + jj) * elempack + q % elempack;
#if __riscv_vector
            const size_t vl = __riscv_vsetvl_e8m1(nn);
            vint8m1_t _b;
            if (elempack == 1)
                _b = __riscv_vle8_v_i8m1(pB, vl);
            else
                _b = __riscv_vlse8_v_i8m1(pB, elempack, vl);
            __riscv_vse8_v_i8m1(pp, _b, vl);
#else
            for (int x = 0; x < nn; x++)
                pp[x] = pB[x * elempack];
#endif // __riscv_vector
            pp += nn;
        }
        jj += nn;
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
    const int maxk = kernel_w * kernel_h;
    const int outw = (w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    signed char* pp = B;

    int jj = 0;
    while (jj < max_jj)
    {
        const int nn = max_jj - jj >= 8 ? 8 : max_jj - jj >= 4 ? 4 : max_jj - jj >= 2 ? 2 : 1;
        for (int kk = 0; kk < max_kk; kk++)
        {
            const int q = (k + kk) / maxk;
            const int uv = (k + kk) % maxk;
            const int u = uv / kernel_w;
            const int v = uv % kernel_w;
            const signed char* img = bottom_blob.channel(q / elempack);

            // A spatial microtile may cross output rows.
            int x = 0;
            while (x < nn)
            {
                const int dy = (j + jj + x) / outw;
                const int dx = (j + jj + x) % outw;
                const int count = std::min(nn - x, outw - dx);
                const signed char* pB = img + ((dy * stride_h + u * dilation_h) * w + dx * stride_w + v * dilation_w) * elempack + q % elempack;
#if __riscv_vector
                const size_t vl = __riscv_vsetvl_e8m1(count);
                vint8m1_t _b;
                if (stride_w == 1 && elempack == 1)
                    _b = __riscv_vle8_v_i8m1(pB, vl);
                else
                    _b = __riscv_vlse8_v_i8m1(pB, stride_w * elempack, vl);
                __riscv_vse8_v_i8m1(pp + x, _b, vl);
#else
                for (int t = 0; t < count; t++)
                    pp[x + t] = pB[t * stride_w * elempack];
#endif // __riscv_vector
                x += count;
            }
            pp += nn;
        }
        jj += nn;
    }
}

#if __riscv_vector
static inline vint32m1_t convolution_gemm_load_int32_rvv(const int* ptr, size_t out_hstep, int out_elempack, size_t vl)
{
    if (out_elempack == 1)
        return __riscv_vlse32_v_i32m1(ptr, out_hstep * sizeof(int), vl);

    return __riscv_vle32_v_i32m1(ptr, vl);
}

static inline void convolution_gemm_store_int32_rvv(int* ptr, vint32m1_t v, size_t out_hstep, int out_elempack, size_t vl)
{
    if (out_elempack == 1)
        __riscv_vsse32_v_i32m1(ptr, out_hstep * sizeof(int), v, vl);
    else
        __riscv_vse32_v_i32m1(ptr, v, vl);
}
#endif // __riscv_vector

template<int N>
static inline void convolution_gemm_transB_packed_tile_int8_rvv(const signed char* pAT, const signed char* pBT, Mat& top_blob, int i, int max_ii, int j, int k, int max_kk)
{
#if __riscv_vector
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.cstep;
    const int packn = csrr_vlenb() / 4;
    for (int ii = 0; ii < max_ii; ii += packn)
    {
        const size_t vl = __riscv_vsetvl_e32m1(std::min(max_ii - ii, packn));
        int* outptr = top_blob.channel((i + ii) / out_elempack);
        outptr += j * out_elempack;
        vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t _sum1 = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t _sum2 = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t _sum3 = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t _sum4 = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t _sum5 = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t _sum6 = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t _sum7 = __riscv_vmv_v_x_i32m1(0, vl);
        if (k != 0)
        {
            _sum0 = convolution_gemm_load_int32_rvv(outptr, out_hstep, out_elempack, vl);
            if (N > 1)
                _sum1 = convolution_gemm_load_int32_rvv(outptr + 1 * out_elempack, out_hstep, out_elempack, vl);
            if (N > 2)
                _sum2 = convolution_gemm_load_int32_rvv(outptr + 2 * out_elempack, out_hstep, out_elempack, vl);
            if (N > 3)
                _sum3 = convolution_gemm_load_int32_rvv(outptr + 3 * out_elempack, out_hstep, out_elempack, vl);
            if (N > 4)
                _sum4 = convolution_gemm_load_int32_rvv(outptr + 4 * out_elempack, out_hstep, out_elempack, vl);
            if (N > 5)
                _sum5 = convolution_gemm_load_int32_rvv(outptr + 5 * out_elempack, out_hstep, out_elempack, vl);
            if (N > 6)
                _sum6 = convolution_gemm_load_int32_rvv(outptr + 6 * out_elempack, out_hstep, out_elempack, vl);
            if (N > 7)
                _sum7 = convolution_gemm_load_int32_rvv(outptr + 7 * out_elempack, out_hstep, out_elempack, vl);
        }

        const signed char* pA = pAT + ii * max_kk;
        const signed char* pB = pBT;
        for (int kk = 0; kk < max_kk; kk++)
        {
#if __riscv_xtheadvector
            // Vector 0.7 has no fractional LMUL or vsext instructions.
            vint8m1_t _a = __riscv_vle8_v_i8m1(pA, vl);
            vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(_a, 0, vl);
            vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
            vint32m1_t _a32low = __riscv_vget_v_i32m4_i32m1(_a32, 0);
            _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (int)pB[0], _a32low, vl);
            if (N > 1)
                _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (int)pB[1], _a32low, vl);
            if (N > 2)
                _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (int)pB[2], _a32low, vl);
            if (N > 3)
                _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (int)pB[3], _a32low, vl);
            if (N > 4)
                _sum4 = __riscv_vmacc_vx_i32m1(_sum4, (int)pB[4], _a32low, vl);
            if (N > 5)
                _sum5 = __riscv_vmacc_vx_i32m1(_sum5, (int)pB[5], _a32low, vl);
            if (N > 6)
                _sum6 = __riscv_vmacc_vx_i32m1(_sum6, (int)pB[6], _a32low, vl);
            if (N > 7)
                _sum7 = __riscv_vmacc_vx_i32m1(_sum7, (int)pB[7], _a32low, vl);
#else
            vint8mf4_t _a = __riscv_vle8_v_i8mf4(pA, vl);
            vint16mf2_t _a16 = __riscv_vsext_vf2_i16mf2(_a, vl);
            _sum0 = __riscv_vwmacc_vx_i32m1(_sum0, (short)pB[0], _a16, vl);
            if (N > 1)
                _sum1 = __riscv_vwmacc_vx_i32m1(_sum1, (short)pB[1], _a16, vl);
            if (N > 2)
                _sum2 = __riscv_vwmacc_vx_i32m1(_sum2, (short)pB[2], _a16, vl);
            if (N > 3)
                _sum3 = __riscv_vwmacc_vx_i32m1(_sum3, (short)pB[3], _a16, vl);
            if (N > 4)
                _sum4 = __riscv_vwmacc_vx_i32m1(_sum4, (short)pB[4], _a16, vl);
            if (N > 5)
                _sum5 = __riscv_vwmacc_vx_i32m1(_sum5, (short)pB[5], _a16, vl);
            if (N > 6)
                _sum6 = __riscv_vwmacc_vx_i32m1(_sum6, (short)pB[6], _a16, vl);
            if (N > 7)
                _sum7 = __riscv_vwmacc_vx_i32m1(_sum7, (short)pB[7], _a16, vl);
#endif // __riscv_xtheadvector
            pA += vl;
            pB += N;
        }

        convolution_gemm_store_int32_rvv(outptr, _sum0, out_hstep, out_elempack, vl);
        if (N > 1)
            convolution_gemm_store_int32_rvv(outptr + 1 * out_elempack, _sum1, out_hstep, out_elempack, vl);
        if (N > 2)
            convolution_gemm_store_int32_rvv(outptr + 2 * out_elempack, _sum2, out_hstep, out_elempack, vl);
        if (N > 3)
            convolution_gemm_store_int32_rvv(outptr + 3 * out_elempack, _sum3, out_hstep, out_elempack, vl);
        if (N > 4)
            convolution_gemm_store_int32_rvv(outptr + 4 * out_elempack, _sum4, out_hstep, out_elempack, vl);
        if (N > 5)
            convolution_gemm_store_int32_rvv(outptr + 5 * out_elempack, _sum5, out_hstep, out_elempack, vl);
        if (N > 6)
            convolution_gemm_store_int32_rvv(outptr + 6 * out_elempack, _sum6, out_hstep, out_elempack, vl);
        if (N > 7)
            convolution_gemm_store_int32_rvv(outptr + 7 * out_elempack, _sum7, out_hstep, out_elempack, vl);
    }
#else
    for (int ii = 0; ii < max_ii; ii++)
    {
        int* outptr = top_blob.channel(i + ii);
        outptr += j;
        int sums[N];
        for (int jj = 0; jj < N; jj++)
            sums[jj] = k == 0 ? 0 : outptr[jj];

        const signed char* pA = pAT + ii * max_kk;
        const signed char* pB = pBT;
        for (int kk = 0; kk < max_kk; kk++)
        {
            const int a = pA[kk];
            for (int jj = 0; jj < N; jj++)
                sums[jj] += a * pB[jj];
            pB += N;
        }
        for (int jj = 0; jj < N; jj++)
            outptr[jj] = sums[jj];
    }
#endif // __riscv_vector
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
    // Split M only when there are too few spatial tiles to occupy the threads.
    const int nn_Mgroup = std::min(nn_M, (nT + nn_N - 1) / nn_N);
    const int nn_MN = nn_Mgroup * nn_N;
    const int num_threads = std::min(nT, nn_MN);

    // Bounded workspace: one K/N panel per worker, never a full im2col matrix.
    // Partial sums live in the existing int32 output between K tiles.
    Mat BT_tileX(TILE_K * TILE_N, 1, num_threads, (size_t)1u, 1, opt.workspace_allocator);
    if (BT_tileX.empty())
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
        Mat BT_tile = BT_tileX.channel(get_omp_thread_num());

        for (int ppk = 0; ppk < nn_K; ppk++)
        {
            const int k = ppk * TILE_K;
            const int max_kk = std::min(K - k, TILE_K);
            convolution_im2col_input_tile_int8_rvv(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);

            for (int ppi = ppi_start; ppi < ppi_end; ppi++)
            {
                const int i = ppi * TILE_M;
                const int max_ii = std::min(M - i, TILE_M);
                const signed char* pAT = AT.channel(ppi).row<const signed char>(ppk);
                const signed char* pBT = BT_tile;

                int jj = 0;
                for (; jj + 7 < max_jj; jj += 8)
                {
                    convolution_gemm_transB_packed_tile_int8_rvv<8>(pAT, pBT, top_blob, i, max_ii, j + jj, k, max_kk);
                    pBT += max_kk * 8;
                }
                for (; jj + 3 < max_jj; jj += 4)
                {
                    convolution_gemm_transB_packed_tile_int8_rvv<4>(pAT, pBT, top_blob, i, max_ii, j + jj, k, max_kk);
                    pBT += max_kk * 4;
                }
                for (; jj + 1 < max_jj; jj += 2)
                {
                    convolution_gemm_transB_packed_tile_int8_rvv<2>(pAT, pBT, top_blob, i, max_ii, j + jj, k, max_kk);
                    pBT += max_kk * 2;
                }
                for (; jj < max_jj; jj++)
                {
                    convolution_gemm_transB_packed_tile_int8_rvv<1>(pAT, pBT, top_blob, i, max_ii, j + jj, k, max_kk);
                    pBT += max_kk;
                }
            }
        }
    }

    return 0;
}

// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
        tile_size = (l2_cache_size - TILE_M * TILE_K) / (TILE_K + TILE_M * 4);
        TILE_N = std::max(4, tile_size / 4 * 4);
        int nn_N = (N + TILE_N - 1) / TILE_N;
        nn_N = std::max(nn_N, std::min(nT, (N + 3) / 4));
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
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
    // Interleave output channels so one weight vector serves four positions.
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
            const int packn = csrr_vlenb();
            const size_t vl = __riscv_vsetvl_e8m1(packn);

            for (; ii + (packn - 1) < max_ii; ii += packn)
            {
                const signed char* pA = (const signed char*)kernel + (i + ii) * K + k;
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vint8m1_t _a = __riscv_vlse8_v_i8m1(pA + kk, K, vl);
                    __riscv_vse8_v_i8m1(pp, _a, vl);
                    pp += packn;
                }
            }
            if (ii < max_ii)
            {
                const size_t vl = __riscv_vsetvl_e8m1(max_ii - ii);
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
        const int nn = max_jj - jj >= 4 ? 4 : max_jj - jj >= 2 ? 2 : 1;
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
        const int nn = max_jj - jj >= 4 ? 4 : max_jj - jj >= 2 ? 2 : 1;
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
// Keep wide accumulators local to the kernel. Inlining into the OpenMP loop
// makes GCC spill vector registers inside the K loop.
__attribute__((noinline))
#endif
static void
convolution_gemm_transB_packed_tile_int8_rvv(const signed char* pAT, const signed char* pBT, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
    int ii = 0;
#if __riscv_vector
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.cstep;
    const int packn = csrr_vlenb();
    const int packn_int32 = packn / 4;
    const size_t vl = __riscv_vsetvl_e32m4(packn);
    const size_t vl_int32 = __riscv_vsetvl_e32m1(packn_int32);

    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
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
                if (out_elempack == packn_int32)
                {
                    vint32m1_t _s00 = __riscv_vle32_v_i32m1(outptr0, vl_int32);
                    vint32m1_t _s01 = __riscv_vle32_v_i32m1(outptr0 + out_hstep * packn_int32 * 1, vl_int32);
                    vint32m1_t _s02 = __riscv_vle32_v_i32m1(outptr0 + out_hstep * packn_int32 * 2, vl_int32);
                    vint32m1_t _s03 = __riscv_vle32_v_i32m1(outptr0 + out_hstep * packn_int32 * 3, vl_int32);
                    _sum0 = __riscv_vcreate_v_i32m1_i32m4(_s00, _s01, _s02, _s03);
                    vint32m1_t _s10 = __riscv_vle32_v_i32m1(outptr0 + packn_int32, vl_int32);
                    vint32m1_t _s11 = __riscv_vle32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 1, vl_int32);
                    vint32m1_t _s12 = __riscv_vle32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 2, vl_int32);
                    vint32m1_t _s13 = __riscv_vle32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 3, vl_int32);
                    _sum1 = __riscv_vcreate_v_i32m1_i32m4(_s10, _s11, _s12, _s13);
                    vint32m1_t _s20 = __riscv_vle32_v_i32m1(outptr0 + packn_int32 * 2, vl_int32);
                    vint32m1_t _s21 = __riscv_vle32_v_i32m1(outptr0 + packn_int32 * 2 + out_hstep * packn_int32 * 1, vl_int32);
                    vint32m1_t _s22 = __riscv_vle32_v_i32m1(outptr0 + packn_int32 * 2 + out_hstep * packn_int32 * 2, vl_int32);
                    vint32m1_t _s23 = __riscv_vle32_v_i32m1(outptr0 + packn_int32 * 2 + out_hstep * packn_int32 * 3, vl_int32);
                    _sum2 = __riscv_vcreate_v_i32m1_i32m4(_s20, _s21, _s22, _s23);
                    vint32m1_t _s30 = __riscv_vle32_v_i32m1(outptr0 + packn_int32 * 3, vl_int32);
                    vint32m1_t _s31 = __riscv_vle32_v_i32m1(outptr0 + packn_int32 * 3 + out_hstep * packn_int32 * 1, vl_int32);
                    vint32m1_t _s32 = __riscv_vle32_v_i32m1(outptr0 + packn_int32 * 3 + out_hstep * packn_int32 * 2, vl_int32);
                    vint32m1_t _s33 = __riscv_vle32_v_i32m1(outptr0 + packn_int32 * 3 + out_hstep * packn_int32 * 3, vl_int32);
                    _sum3 = __riscv_vcreate_v_i32m1_i32m4(_s30, _s31, _s32, _s33);
                }
                else // if (out_elempack == 1)
                {
                    _sum0 = __riscv_vlse32_v_i32m4(outptr0, out_hstep * sizeof(int), vl);
                    _sum1 = __riscv_vlse32_v_i32m4(outptr0 + 1, out_hstep * sizeof(int), vl);
                    _sum2 = __riscv_vlse32_v_i32m4(outptr0 + 2, out_hstep * sizeof(int), vl);
                    _sum3 = __riscv_vlse32_v_i32m4(outptr0 + 3, out_hstep * sizeof(int), vl);
                }
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

                pA += packn;
                pB += 4;
            }

            if (out_elempack == packn_int32)
            {
                __riscv_vse32_v_i32m1(outptr0, __riscv_vget_v_i32m4_i32m1(_sum0, 0), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32, __riscv_vget_v_i32m4_i32m1(_sum1, 0), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum2, 0), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 3, __riscv_vget_v_i32m4_i32m1(_sum3, 0), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum0, 1), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum1, 1), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 2 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum2, 1), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 3 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum3, 1), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum0, 2), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum1, 2), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 2 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum2, 2), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 3 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum3, 2), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 3, __riscv_vget_v_i32m4_i32m1(_sum0, 3), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 3, __riscv_vget_v_i32m4_i32m1(_sum1, 3), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 2 + out_hstep * packn_int32 * 3, __riscv_vget_v_i32m4_i32m1(_sum2, 3), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 3 + out_hstep * packn_int32 * 3, __riscv_vget_v_i32m4_i32m1(_sum3, 3), vl_int32);
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
                if (out_elempack == packn_int32)
                {
                    vint32m1_t _s00 = __riscv_vle32_v_i32m1(outptr0, vl_int32);
                    vint32m1_t _s01 = __riscv_vle32_v_i32m1(outptr0 + out_hstep * packn_int32 * 1, vl_int32);
                    vint32m1_t _s02 = __riscv_vle32_v_i32m1(outptr0 + out_hstep * packn_int32 * 2, vl_int32);
                    vint32m1_t _s03 = __riscv_vle32_v_i32m1(outptr0 + out_hstep * packn_int32 * 3, vl_int32);
                    _sum0 = __riscv_vcreate_v_i32m1_i32m4(_s00, _s01, _s02, _s03);
                    vint32m1_t _s10 = __riscv_vle32_v_i32m1(outptr0 + packn_int32, vl_int32);
                    vint32m1_t _s11 = __riscv_vle32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 1, vl_int32);
                    vint32m1_t _s12 = __riscv_vle32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 2, vl_int32);
                    vint32m1_t _s13 = __riscv_vle32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 3, vl_int32);
                    _sum1 = __riscv_vcreate_v_i32m1_i32m4(_s10, _s11, _s12, _s13);
                }
                else // if (out_elempack == 1)
                {
                    _sum0 = __riscv_vlse32_v_i32m4(outptr0, out_hstep * sizeof(int), vl);
                    _sum1 = __riscv_vlse32_v_i32m4(outptr0 + 1, out_hstep * sizeof(int), vl);
                }
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

                pA += packn;
                pB += 2;
            }

            if (out_elempack == packn_int32)
            {
                __riscv_vse32_v_i32m1(outptr0, __riscv_vget_v_i32m4_i32m1(_sum0, 0), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32, __riscv_vget_v_i32m4_i32m1(_sum1, 0), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum0, 1), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum1, 1), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum0, 2), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum1, 2), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 3, __riscv_vget_v_i32m4_i32m1(_sum0, 3), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 3, __riscv_vget_v_i32m4_i32m1(_sum1, 3), vl_int32);
                outptr0 += packn_int32 * 2;
            }
            else // if (out_elempack == 1)
            {
                __riscv_vsse32_v_i32m4(outptr0, out_hstep * sizeof(int), _sum0, vl);
                __riscv_vsse32_v_i32m4(outptr0 + 1, out_hstep * sizeof(int), _sum1, vl);
                outptr0 += 2;
            }
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
                if (out_elempack == packn_int32)
                {
                    vint32m1_t _s00 = __riscv_vle32_v_i32m1(outptr0, vl_int32);
                    vint32m1_t _s01 = __riscv_vle32_v_i32m1(outptr0 + out_hstep * packn_int32 * 1, vl_int32);
                    vint32m1_t _s02 = __riscv_vle32_v_i32m1(outptr0 + out_hstep * packn_int32 * 2, vl_int32);
                    vint32m1_t _s03 = __riscv_vle32_v_i32m1(outptr0 + out_hstep * packn_int32 * 3, vl_int32);
                    _sum0 = __riscv_vcreate_v_i32m1_i32m4(_s00, _s01, _s02, _s03);
                }
                else // if (out_elempack == 1)
                {
                    _sum0 = __riscv_vlse32_v_i32m4(outptr0, out_hstep * sizeof(int), vl);
                }
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

                pA += packn;
                pB++;
            }

            if (out_elempack == packn_int32)
            {
                __riscv_vse32_v_i32m1(outptr0, __riscv_vget_v_i32m4_i32m1(_sum0, 0), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum0, 1), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum0, 2), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 3, __riscv_vget_v_i32m4_i32m1(_sum0, 3), vl_int32);
                outptr0 += packn_int32;
            }
            else // if (out_elempack == 1)
            {
                __riscv_vsse32_v_i32m4(outptr0, out_hstep * sizeof(int), _sum0, vl);
                outptr0++;
            }
        }
    }

    if (ii < max_ii)
    {
        const size_t vl = __riscv_vsetvl_e32m4(max_ii - ii);
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
                if (out_elempack == packn_int32)
                {
                    _sum0 = __riscv_vmv_v_x_i32m4(0, vl);
                    _sum0 = __riscv_vset_v_i32m1_i32m4(_sum0, 0, __riscv_vle32_v_i32m1(outptr0, vl_int32));
                    if (max_ii - ii > packn_int32 * 1)
                        _sum0 = __riscv_vset_v_i32m1_i32m4(_sum0, 1, __riscv_vle32_v_i32m1(outptr0 + out_hstep * packn_int32 * 1, vl_int32));
                    if (max_ii - ii > packn_int32 * 2)
                        _sum0 = __riscv_vset_v_i32m1_i32m4(_sum0, 2, __riscv_vle32_v_i32m1(outptr0 + out_hstep * packn_int32 * 2, vl_int32));
                    if (max_ii - ii > packn_int32 * 3)
                        _sum0 = __riscv_vset_v_i32m1_i32m4(_sum0, 3, __riscv_vle32_v_i32m1(outptr0 + out_hstep * packn_int32 * 3, vl_int32));
                    _sum1 = __riscv_vmv_v_x_i32m4(0, vl);
                    _sum1 = __riscv_vset_v_i32m1_i32m4(_sum1, 0, __riscv_vle32_v_i32m1(outptr0 + packn_int32, vl_int32));
                    if (max_ii - ii > packn_int32 * 1)
                        _sum1 = __riscv_vset_v_i32m1_i32m4(_sum1, 1, __riscv_vle32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 1, vl_int32));
                    if (max_ii - ii > packn_int32 * 2)
                        _sum1 = __riscv_vset_v_i32m1_i32m4(_sum1, 2, __riscv_vle32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 2, vl_int32));
                    if (max_ii - ii > packn_int32 * 3)
                        _sum1 = __riscv_vset_v_i32m1_i32m4(_sum1, 3, __riscv_vle32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 3, vl_int32));
                    _sum2 = __riscv_vmv_v_x_i32m4(0, vl);
                    _sum2 = __riscv_vset_v_i32m1_i32m4(_sum2, 0, __riscv_vle32_v_i32m1(outptr0 + packn_int32 * 2, vl_int32));
                    if (max_ii - ii > packn_int32 * 1)
                        _sum2 = __riscv_vset_v_i32m1_i32m4(_sum2, 1, __riscv_vle32_v_i32m1(outptr0 + packn_int32 * 2 + out_hstep * packn_int32 * 1, vl_int32));
                    if (max_ii - ii > packn_int32 * 2)
                        _sum2 = __riscv_vset_v_i32m1_i32m4(_sum2, 2, __riscv_vle32_v_i32m1(outptr0 + packn_int32 * 2 + out_hstep * packn_int32 * 2, vl_int32));
                    if (max_ii - ii > packn_int32 * 3)
                        _sum2 = __riscv_vset_v_i32m1_i32m4(_sum2, 3, __riscv_vle32_v_i32m1(outptr0 + packn_int32 * 2 + out_hstep * packn_int32 * 3, vl_int32));
                    _sum3 = __riscv_vmv_v_x_i32m4(0, vl);
                    _sum3 = __riscv_vset_v_i32m1_i32m4(_sum3, 0, __riscv_vle32_v_i32m1(outptr0 + packn_int32 * 3, vl_int32));
                    if (max_ii - ii > packn_int32 * 1)
                        _sum3 = __riscv_vset_v_i32m1_i32m4(_sum3, 1, __riscv_vle32_v_i32m1(outptr0 + packn_int32 * 3 + out_hstep * packn_int32 * 1, vl_int32));
                    if (max_ii - ii > packn_int32 * 2)
                        _sum3 = __riscv_vset_v_i32m1_i32m4(_sum3, 2, __riscv_vle32_v_i32m1(outptr0 + packn_int32 * 3 + out_hstep * packn_int32 * 2, vl_int32));
                    if (max_ii - ii > packn_int32 * 3)
                        _sum3 = __riscv_vset_v_i32m1_i32m4(_sum3, 3, __riscv_vle32_v_i32m1(outptr0 + packn_int32 * 3 + out_hstep * packn_int32 * 3, vl_int32));
                }
                else // if (out_elempack == 1)
                {
                    _sum0 = __riscv_vlse32_v_i32m4(outptr0, out_hstep * sizeof(int), vl);
                    _sum1 = __riscv_vlse32_v_i32m4(outptr0 + 1, out_hstep * sizeof(int), vl);
                    _sum2 = __riscv_vlse32_v_i32m4(outptr0 + 2, out_hstep * sizeof(int), vl);
                    _sum3 = __riscv_vlse32_v_i32m4(outptr0 + 3, out_hstep * sizeof(int), vl);
                }
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

            if (out_elempack == packn_int32)
            {
                __riscv_vse32_v_i32m1(outptr0, __riscv_vget_v_i32m4_i32m1(_sum0, 0), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32, __riscv_vget_v_i32m4_i32m1(_sum1, 0), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum2, 0), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 3, __riscv_vget_v_i32m4_i32m1(_sum3, 0), vl_int32);
                if (max_ii - ii > packn_int32 * 1)
                {
                    __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum0, 1), vl_int32);
                    __riscv_vse32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum1, 1), vl_int32);
                    __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 2 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum2, 1), vl_int32);
                    __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 3 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum3, 1), vl_int32);
                }
                if (max_ii - ii > packn_int32 * 2)
                {
                    __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum0, 2), vl_int32);
                    __riscv_vse32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum1, 2), vl_int32);
                    __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 2 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum2, 2), vl_int32);
                    __riscv_vse32_v_i32m1(outptr0 + packn_int32 * 3 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum3, 2), vl_int32);
                }
                if (max_ii - ii > packn_int32 * 3)
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
                if (out_elempack == packn_int32)
                {
                    _sum0 = __riscv_vmv_v_x_i32m4(0, vl);
                    _sum0 = __riscv_vset_v_i32m1_i32m4(_sum0, 0, __riscv_vle32_v_i32m1(outptr0, vl_int32));
                    if (max_ii - ii > packn_int32 * 1)
                        _sum0 = __riscv_vset_v_i32m1_i32m4(_sum0, 1, __riscv_vle32_v_i32m1(outptr0 + out_hstep * packn_int32 * 1, vl_int32));
                    if (max_ii - ii > packn_int32 * 2)
                        _sum0 = __riscv_vset_v_i32m1_i32m4(_sum0, 2, __riscv_vle32_v_i32m1(outptr0 + out_hstep * packn_int32 * 2, vl_int32));
                    if (max_ii - ii > packn_int32 * 3)
                        _sum0 = __riscv_vset_v_i32m1_i32m4(_sum0, 3, __riscv_vle32_v_i32m1(outptr0 + out_hstep * packn_int32 * 3, vl_int32));
                    _sum1 = __riscv_vmv_v_x_i32m4(0, vl);
                    _sum1 = __riscv_vset_v_i32m1_i32m4(_sum1, 0, __riscv_vle32_v_i32m1(outptr0 + packn_int32, vl_int32));
                    if (max_ii - ii > packn_int32 * 1)
                        _sum1 = __riscv_vset_v_i32m1_i32m4(_sum1, 1, __riscv_vle32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 1, vl_int32));
                    if (max_ii - ii > packn_int32 * 2)
                        _sum1 = __riscv_vset_v_i32m1_i32m4(_sum1, 2, __riscv_vle32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 2, vl_int32));
                    if (max_ii - ii > packn_int32 * 3)
                        _sum1 = __riscv_vset_v_i32m1_i32m4(_sum1, 3, __riscv_vle32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 3, vl_int32));
                }
                else // if (out_elempack == 1)
                {
                    _sum0 = __riscv_vlse32_v_i32m4(outptr0, out_hstep * sizeof(int), vl);
                    _sum1 = __riscv_vlse32_v_i32m4(outptr0 + 1, out_hstep * sizeof(int), vl);
                }
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

            if (out_elempack == packn_int32)
            {
                __riscv_vse32_v_i32m1(outptr0, __riscv_vget_v_i32m4_i32m1(_sum0, 0), vl_int32);
                __riscv_vse32_v_i32m1(outptr0 + packn_int32, __riscv_vget_v_i32m4_i32m1(_sum1, 0), vl_int32);
                if (max_ii - ii > packn_int32 * 1)
                {
                    __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum0, 1), vl_int32);
                    __riscv_vse32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum1, 1), vl_int32);
                }
                if (max_ii - ii > packn_int32 * 2)
                {
                    __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum0, 2), vl_int32);
                    __riscv_vse32_v_i32m1(outptr0 + packn_int32 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum1, 2), vl_int32);
                }
                if (max_ii - ii > packn_int32 * 3)
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
                if (out_elempack == packn_int32)
                {
                    _sum0 = __riscv_vmv_v_x_i32m4(0, vl);
                    _sum0 = __riscv_vset_v_i32m1_i32m4(_sum0, 0, __riscv_vle32_v_i32m1(outptr0, vl_int32));
                    if (max_ii - ii > packn_int32 * 1)
                        _sum0 = __riscv_vset_v_i32m1_i32m4(_sum0, 1, __riscv_vle32_v_i32m1(outptr0 + out_hstep * packn_int32 * 1, vl_int32));
                    if (max_ii - ii > packn_int32 * 2)
                        _sum0 = __riscv_vset_v_i32m1_i32m4(_sum0, 2, __riscv_vle32_v_i32m1(outptr0 + out_hstep * packn_int32 * 2, vl_int32));
                    if (max_ii - ii > packn_int32 * 3)
                        _sum0 = __riscv_vset_v_i32m1_i32m4(_sum0, 3, __riscv_vle32_v_i32m1(outptr0 + out_hstep * packn_int32 * 3, vl_int32));
                }
                else // if (out_elempack == 1)
                {
                    _sum0 = __riscv_vlse32_v_i32m4(outptr0, out_hstep * sizeof(int), vl);
                }
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

            if (out_elempack == packn_int32)
            {
                __riscv_vse32_v_i32m1(outptr0, __riscv_vget_v_i32m4_i32m1(_sum0, 0), vl_int32);
                if (max_ii - ii > packn_int32 * 1)
                {
                    __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 1, __riscv_vget_v_i32m4_i32m1(_sum0, 1), vl_int32);
                }
                if (max_ii - ii > packn_int32 * 2)
                {
                    __riscv_vse32_v_i32m1(outptr0 + out_hstep * packn_int32 * 2, __riscv_vget_v_i32m4_i32m1(_sum0, 2), vl_int32);
                }
                if (max_ii - ii > packn_int32 * 3)
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
    }
#else
    for (; ii < max_ii; ii++)
    {
        int* outptr0 = top_blob.channel(i + ii);
        outptr0 += j;

        const signed char* pB = pBT;

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT + ii * max_kk;

            int sum0 = k == 0 ? 0 : outptr0[0];
            int sum1 = k == 0 ? 0 : outptr0[1];
            int sum2 = k == 0 ? 0 : outptr0[2];
            int sum3 = k == 0 ? 0 : outptr0[3];

            for (int kk = 0; kk < max_kk; kk++)
            {
                const int a = pA[0];
                sum0 += a * pB[0];
                sum1 += a * pB[1];
                sum2 += a * pB[2];
                sum3 += a * pB[3];

                pA++;
                pB += 4;
            }

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0[2] = sum2;
            outptr0[3] = sum3;
            outptr0 += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT + ii * max_kk;

            int sum0 = k == 0 ? 0 : outptr0[0];
            int sum1 = k == 0 ? 0 : outptr0[1];

            for (int kk = 0; kk < max_kk; kk++)
            {
                const int a = pA[0];
                sum0 += a * pB[0];
                sum1 += a * pB[1];

                pA++;
                pB += 2;
            }

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0 += 2;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pA = pAT + ii * max_kk;

            int sum0 = k == 0 ? 0 : outptr0[0];

            for (int kk = 0; kk < max_kk; kk++)
            {
                const int a = pA[0];
                sum0 += a * pB[0];

                pA++;
                pB++;
            }

            outptr0[0] = sum0;
            outptr0++;
        }
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

                convolution_gemm_transB_packed_tile_int8_rvv(pAT, pBT, top_blob, i, max_ii, j, max_jj, k, max_kk);
            }
        }
    }

    return 0;
}

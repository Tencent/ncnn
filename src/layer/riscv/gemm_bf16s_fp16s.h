// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pack_A_tile_bf16_fp16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
#endif

    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __riscv_vector
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        if (elempack == packn)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * packn;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __riscv_vse16_v_u16m1(pp, __riscv_vle16_v_u16m1(p0, vl), vl);
                pp += packn;
                p0 += packn;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __riscv_vse16_v_u16m1(pp, __riscv_vlse16_v_u16m1(p0, A_hstep * sizeof(unsigned short), vl), vl);
                pp += packn;
                p0++;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;

            int kk = 0;
#if __riscv_vector
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                vuint16m1_t v0 = __riscv_vle16_v_u16m1(p0, vl);
                vuint16m1_t v1 = __riscv_vle16_v_u16m1(p1, vl);
                __riscv_vsseg2e16_v_u16m1x2(pp, __riscv_vcreate_v_u16m1x2(v0, v1), vl);
                pp += packn * 2;
                p0 += packn;
                p1 += packn;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp += 2;
                p0++;
                p1++;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;

            int kk = 0;
#if __riscv_vector
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                __riscv_vse16_v_u16m1(pp, __riscv_vle16_v_u16m1(p0, vl), vl);
                pp += packn;
                p0 += packn;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_A_tile_bf16_fp16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
#endif

    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __riscv_vector
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        if (elempack == packn)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                // transposeNxN
                for (int l = 0; l < packn; l++)
                {
                    __riscv_vse16_v_u16m1(pp, __riscv_vlse16_v_u16m1(p0 + l, packn * sizeof(unsigned short), vl), vl);
                    pp += packn;
                }
                p0 += A_hstep * packn;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __riscv_vse16_v_u16m1(pp, __riscv_vle16_v_u16m1(p0, vl), vl);
                pp += packn;
                p0 += A_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                vuint16m1_t v0 = __riscv_vle16_v_u16m1(p0, vl);
                vuint16m1_t v1 = __riscv_vle16_v_u16m1(p0 + packn, vl);
                __riscv_vsseg2e16_v_u16m1x2(pp, __riscv_vcreate_v_u16m1x2(v0, v1), vl);
                pp += packn * 2;
                p0 += A_hstep * packn;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += A_hstep;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                __riscv_vse16_v_u16m1(pp, __riscv_vle16_v_u16m1(p0, vl), vl);
                pp += packn;
                p0 += A_hstep * packn;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += A_hstep;
            }
        }
    }
}

static void pack_B_tile_bf16_fp16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
#endif

    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __riscv_vector
    for (; jj + (packn - 1) < max_jj; jj += packn)
    {
        if (elempack == packn)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * packn;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __riscv_vse16_v_u16m1(pp, __riscv_vle16_v_u16m1(p0, vl), vl);
                pp += packn;
                p0 += packn;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __riscv_vse16_v_u16m1(pp, __riscv_vlse16_v_u16m1(p0, B_hstep * sizeof(unsigned short), vl), vl);
                pp += packn;
                p0++;
            }
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;

            int kk = 0;
#if __riscv_vector
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                vuint16m1_t v0 = __riscv_vle16_v_u16m1(p0, vl);
                vuint16m1_t v1 = __riscv_vle16_v_u16m1(p1, vl);
                __riscv_vsseg2e16_v_u16m1x2(pp, __riscv_vcreate_v_u16m1x2(v0, v1), vl);
                pp += packn * 2;
                p0 += packn;
                p1 += packn;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp += 2;
                p0++;
                p1++;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

            int kk = 0;
#if __riscv_vector
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                __riscv_vse16_v_u16m1(pp, __riscv_vle16_v_u16m1(p0, vl), vl);
                pp += packn;
                p0 += packn;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_B_tile_bf16_fp16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
#endif

    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __riscv_vector
    for (; jj + (packn - 1) < max_jj; jj += packn)
    {
        if (elempack == packn)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                // transposeNxN
                for (int l = 0; l < packn; l++)
                {
                    __riscv_vse16_v_u16m1(pp, __riscv_vlse16_v_u16m1(p0 + l, packn * sizeof(unsigned short), vl), vl);
                    pp += packn;
                }
                p0 += B_hstep * packn;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __riscv_vse16_v_u16m1(pp, __riscv_vle16_v_u16m1(p0, vl), vl);
                pp += packn;
                p0 += B_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                vuint16m1_t v0 = __riscv_vle16_v_u16m1(p0, vl);
                vuint16m1_t v1 = __riscv_vle16_v_u16m1(p0 + packn, vl);
                __riscv_vsseg2e16_v_u16m1x2(pp, __riscv_vcreate_v_u16m1x2(v0, v1), vl);
                pp += packn * 2;
                p0 += B_hstep * packn;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += B_hstep;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                __riscv_vse16_v_u16m1(pp, __riscv_vle16_v_u16m1(p0, vl), vl);
                pp += packn;
                p0 += B_hstep * packn;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += B_hstep;
            }
        }
    }
}

static void transpose_unpack_output_tile_bf16_fp16(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
#endif

    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const unsigned short* pp = topT;

    int ii = 0;
#if __riscv_vector
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        if (out_elempack == packn)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * packn;

            for (int jj = 0; jj + (packn - 1) < max_jj; jj += packn)
            {
                // transposeNxN
                for (int l = 0; l < packn; l++)
                {
                    __riscv_vsse16_v_u16m1(p0 + l, packn * sizeof(unsigned short), __riscv_vle16_v_u16m1(pp, vl), vl);
                    pp += packn;
                }
                p0 += out_hstep * packn;
            }
        }
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                vuint16m1_t _r0 = __riscv_vle16_v_u16m1(pp, vl);
                __riscv_vse16_v_u16m1(p0, _r0, vl);
                pp += packn;
                p0 += out_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __riscv_vector
        if (out_elempack == packn)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * packn;

            for (int jj = 0; jj + (packn - 1) < max_jj; jj += packn)
            {
                vuint16m1x2_t _s0 = __riscv_vlseg2e16_v_u16m1x2(pp, vl);
                __riscv_vse16_v_u16m1(p0, __riscv_vget_v_u16m1x2_u16m1(_s0, 0), vl);
                __riscv_vse16_v_u16m1(p0 + packn, __riscv_vget_v_u16m1x2_u16m1(_s0, 1), vl);
                pp += packn * 2;
                p0 += out_hstep * packn;
            }
        }
#endif // __riscv_vector
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = pp[0];
                p0[1] = pp[1];
                pp += 2;
                p0 += out_hstep;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
#if __riscv_vector
        if (out_elempack == packn)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * packn;

            for (int jj = 0; jj + (packn - 1) < max_jj; jj += packn)
            {
                vuint16m1_t _r0 = __riscv_vle16_v_u16m1(pp, vl);
                __riscv_vse16_v_u16m1(p0, _r0, vl);
                pp += packn;
                p0 += out_hstep * packn;
            }
        }
#endif // __riscv_vector
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = pp[0];
                pp += 1;
                p0 += out_hstep;
            }
        }
    }
}

static void get_optimal_tile_mnk_bf16s_fp16s(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrtf((float)l2_cache_size / (2 * sizeof(unsigned short) + sizeof(float)));

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
#else
    const int packn = 4;
#endif

    TILE_M = std::max(packn, tile_size / packn * packn);
    TILE_N = std::max(packn, tile_size / packn * packn);
    TILE_K = std::max(packn, tile_size / packn * packn);

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + (packn - 1)) / packn * packn);

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(unsigned short) / TILE_K);

            TILE_M = std::max(packn, tile_size / packn * packn);
            TILE_N = std::max(packn, tile_size / packn * packn);
        }
    }

    TILE_M *= std::min(nT, get_physical_cpu_count());

    if (M > 0)
    {
        int nn_M = (M + TILE_M - 1) / TILE_M;
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + (packn - 1)) / packn * packn);
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + (packn - 1)) / packn * packn);
    }

    if (nT > 1)
    {
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + (packn - 1)) / packn * packn);
    }

    // always take constant TILE_M/N/K value when provided
    if (constant_TILE_M > 0)
    {
        TILE_M = (constant_TILE_M + (packn - 1)) / packn * packn;
    }

    if (constant_TILE_N > 0)
    {
        TILE_N = (constant_TILE_N + (packn - 1)) / packn * packn;
    }

    if (constant_TILE_K > 0)
    {
        TILE_K = (constant_TILE_K + (packn - 1)) / packn * packn;
    }
}

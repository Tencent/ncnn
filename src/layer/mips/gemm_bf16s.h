// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pack_A_tile_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __mips_msa
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;
            const unsigned short* p2 = (const unsigned short*)A + (i + ii + 2) * A_hstep + k;
            const unsigned short* p3 = (const unsigned short*)A + (i + ii + 3) * A_hstep + k;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp += 4;
                p0++;
                p1++;
                p2++;
                p3++;
            }
        }
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
        const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;

        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;

        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_A_tile_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __mips_msa
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // p0 has 4 rows of elempack=4, stride A_hstep*4
                // layout: p0[0..3] = col0 of ii0..ii3, p0[4..7] = col1 of ii0..ii3, etc.
                // We want output: [kk0: ii0,ii1,ii2,ii3, kk1: ii0,ii1,ii2,ii3, ...]
                // With elempack=4, p0[0..15] = 4 columns x 4 ii, but layout is:
                // row k+kk+0: p0[0]=ii0, p0[4]=ii1, p0[8]=ii2, p0[12]=ii3
                // Actually elempack=4 means consecutive 4 elements per pack group
                // p0 points to row (k/4)*A_hstep + (i+ii)*4
                // For transposed A: row=k, col=i+ii
                // elempack=4 means k-dimension is packed by 4
                // p0[0..3] = 4 k-values for ii0, p0[4..7] = 4 k-values for ii1, etc.
                // Wait - for transpose_pack_A, A is indexed as A[k][i], so row=k, col=i
                // With elempack=4, the k dimension is packed
                // p0 = A + k*A_hstep + (i+ii)*4
                // p0[0..3] = ii0's 4 packed k values, p0[4..7] = ii1's 4 packed k values
                // We need to transpose: output is [kk0: ii0,ii1,ii2,ii3, kk1: ii0,ii1,ii2,ii3, ...]
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp[2] = p0[8];
                pp[3] = p0[12];
                pp[4] = p0[1];
                pp[5] = p0[5];
                pp[6] = p0[9];
                pp[7] = p0[13];
                pp[8] = p0[2];
                pp[9] = p0[6];
                pp[10] = p0[10];
                pp[11] = p0[14];
                pp[12] = p0[3];
                pp[13] = p0[7];
                pp[14] = p0[11];
                pp[15] = p0[15];
                pp += 16;
                p0 += A_hstep * 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp[2] = p0[8];
                pp[3] = p0[12];
                pp += 4;
                p0++;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += A_hstep;
            }
        }
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * elempack;
#if __mips_msa
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp[2] = p0[1];
                pp[3] = p0[5];
                pp[4] = p0[2];
                pp[5] = p0[6];
                pp[6] = p0[3];
                pp[7] = p0[7];
                pp += 8;
                p0 += A_hstep * 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp += 2;
                p0++;
            }
        }
#endif // __mips_msa
        if (elempack == 1)
        {
            const unsigned short* p0r = (const unsigned short*)A + k * A_hstep + (i + ii);

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0r[0];
                pp[1] = p0r[1];
                pp += 2;
                p0r += A_hstep;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * elempack;
#if __mips_msa
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += A_hstep * 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
#endif // __mips_msa
        if (elempack == 1)
        {
            const unsigned short* p0r = (const unsigned short*)A + k * A_hstep + (i + ii);

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0r[0];
                pp += 1;
                p0r += A_hstep;
            }
        }
    }
}

static void pack_B_tile_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __mips_msa
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k * 4;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 8) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p1[0];
                pp[5] = p1[1];
                pp[6] = p1[2];
                pp[7] = p1[3];
                pp[8] = p2[0];
                pp[9] = p2[1];
                pp[10] = p2[2];
                pp[11] = p2[3];
                pp += 12;
                p0 += 4;
                p1 += 4;
                p2 += 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 2) * B_hstep + k;
            const unsigned short* p3 = (const unsigned short*)B + (j + jj + 3) * B_hstep + k;
            const unsigned short* p4 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k;
            const unsigned short* p5 = (const unsigned short*)B + (j + jj + 5) * B_hstep + k;
            const unsigned short* p6 = (const unsigned short*)B + (j + jj + 6) * B_hstep + k;
            const unsigned short* p7 = (const unsigned short*)B + (j + jj + 7) * B_hstep + k;
            const unsigned short* p8 = (const unsigned short*)B + (j + jj + 8) * B_hstep + k;
            const unsigned short* p9 = (const unsigned short*)B + (j + jj + 9) * B_hstep + k;
            const unsigned short* pa = (const unsigned short*)B + (j + jj + 10) * B_hstep + k;
            const unsigned short* pb = (const unsigned short*)B + (j + jj + 11) * B_hstep + k;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp[4] = p4[0];
                pp[5] = p5[0];
                pp[6] = p6[0];
                pp[7] = p7[0];
                pp[8] = p8[0];
                pp[9] = p9[0];
                pp[10] = pa[0];
                pp[11] = pb[0];
                pp += 12;
                p0++;
                p1++;
                p2++;
                p3++;
                p4++;
                p5++;
                p6++;
                p7++;
                p8++;
                p9++;
                pa++;
                pb++;
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p1[0];
                pp[5] = p1[1];
                pp[6] = p1[2];
                pp[7] = p1[3];
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 2) * B_hstep + k;
            const unsigned short* p3 = (const unsigned short*)B + (j + jj + 3) * B_hstep + k;
            const unsigned short* p4 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k;
            const unsigned short* p5 = (const unsigned short*)B + (j + jj + 5) * B_hstep + k;
            const unsigned short* p6 = (const unsigned short*)B + (j + jj + 6) * B_hstep + k;
            const unsigned short* p7 = (const unsigned short*)B + (j + jj + 7) * B_hstep + k;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp[4] = p4[0];
                pp[5] = p5[0];
                pp[6] = p6[0];
                pp[7] = p7[0];
                pp += 8;
                p0++;
                p1++;
                p2++;
                p3++;
                p4++;
                p5++;
                p6++;
                p7++;
            }
        }
    }
#endif // __mips_msa
    for (; jj + 3 < max_jj; jj += 4)
    {
        const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * elempack;
#if __mips_msa
        if (elempack == 4)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
#endif // __mips_msa
        {
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 2) * B_hstep + k;
            const unsigned short* p3 = (const unsigned short*)B + (j + jj + 3) * B_hstep + k;
            p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp += 4;
                p0++;
                p1++;
                p2++;
                p3++;
            }
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
        const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;

        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; jj < max_jj; jj++)
    {
        const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_B_tile_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __mips_msa
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;
            const unsigned short* p1 = (const unsigned short*)B + k * B_hstep + (j + jj + 4) * 4;
            const unsigned short* p2 = (const unsigned short*)B + k * B_hstep + (j + jj + 8) * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp[2] = p0[8];
                pp[3] = p0[12];
                pp[4] = p1[0];
                pp[5] = p1[4];
                pp[6] = p1[8];
                pp[7] = p1[12];
                pp[8] = p2[0];
                pp[9] = p2[4];
                pp[10] = p2[8];
                pp[11] = p2[12];
                pp += 12;
                p0++;
                p1++;
                p2++;
                if ((kk + 1) % 4 == 0)
                {
                    p0 += B_hstep * 4 - 4;
                    p1 += B_hstep * 4 - 4;
                    p2 += B_hstep * 4 - 4;
                }
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p0[4];
                pp[5] = p0[5];
                pp[6] = p0[6];
                pp[7] = p0[7];
                pp[8] = p0[8];
                pp[9] = p0[9];
                pp[10] = p0[10];
                pp[11] = p0[11];
                pp += 12;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;
            const unsigned short* p1 = (const unsigned short*)B + k * B_hstep + (j + jj + 4) * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp[2] = p0[8];
                pp[3] = p0[12];
                pp[4] = p1[0];
                pp[5] = p1[4];
                pp[6] = p1[8];
                pp[7] = p1[12];
                pp += 8;
                p0++;
                p1++;
                if ((kk + 1) % 4 == 0)
                {
                    p0 += B_hstep * 4 - 4;
                    p1 += B_hstep * 4 - 4;
                }
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p0[4];
                pp[5] = p0[5];
                pp[6] = p0[6];
                pp[7] = p0[7];
                pp += 8;
                p0 += B_hstep;
            }
        }
    }
#endif // __mips_msa
    for (; jj + 3 < max_jj; jj += 4)
    {
        const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * elempack;
#if __mips_msa
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp[2] = p0[8];
                pp[3] = p0[12];
                pp[4] = p0[1];
                pp[5] = p0[5];
                pp[6] = p0[9];
                pp[7] = p0[13];
                pp[8] = p0[2];
                pp[9] = p0[6];
                pp[10] = p0[10];
                pp[11] = p0[14];
                pp[12] = p0[3];
                pp[13] = p0[7];
                pp[14] = p0[11];
                pp[15] = p0[15];
                pp += 16;
                p0 += B_hstep * 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp[2] = p0[8];
                pp[3] = p0[12];
                pp += 4;
                p0++;
            }
        }
        if (elempack == 1)
#endif // __mips_msa
        {
            p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * elempack;
#if __mips_msa
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp[2] = p0[1];
                pp[3] = p0[5];
                pp[4] = p0[2];
                pp[5] = p0[6];
                pp[6] = p0[3];
                pp[7] = p0[7];
                pp += 8;
                p0 += B_hstep * 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp += 2;
                p0++;
            }
        }
        if (elempack == 1)
#endif // __mips_msa
        {
            p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += B_hstep;
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
        const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * elempack;
#if __mips_msa
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += B_hstep * 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
        if (elempack == 1)
#endif // __mips_msa
        {
            p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += B_hstep;
            }
        }
    }
}

static void gemm_transB_packed_tile_bf16s(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
    // NCNN_LOGE("gemm_transB_packed_tile_bf16s %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    (void)i;
    (void)j;

    const unsigned short* pAT = AT_tile;
    const unsigned short* pBT = BT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __mips_msa
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            const unsigned short* pA = pAT;

            v4f32 _sum0;
            v4f32 _sum1;
            v4f32 _sum2;
            v4f32 _sum3;
            v4f32 _sum4;
            v4f32 _sum5;
            v4f32 _sum6;
            v4f32 _sum7;
            v4f32 _sum8;
            v4f32 _sum9;
            v4f32 _suma;
            v4f32 _sumb;

            if (k == 0)
            {
                _sum0 = (v4f32)__msa_fill_w(0);
                _sum1 = (v4f32)__msa_fill_w(0);
                _sum2 = (v4f32)__msa_fill_w(0);
                _sum3 = (v4f32)__msa_fill_w(0);
                _sum4 = (v4f32)__msa_fill_w(0);
                _sum5 = (v4f32)__msa_fill_w(0);
                _sum6 = (v4f32)__msa_fill_w(0);
                _sum7 = (v4f32)__msa_fill_w(0);
                _sum8 = (v4f32)__msa_fill_w(0);
                _sum9 = (v4f32)__msa_fill_w(0);
                _suma = (v4f32)__msa_fill_w(0);
                _sumb = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _sum0 = (v4f32)__msa_ld_w(outptr, 0);
                _sum1 = (v4f32)__msa_ld_w(outptr + 4, 0);
                _sum2 = (v4f32)__msa_ld_w(outptr + 4 * 2, 0);
                _sum3 = (v4f32)__msa_ld_w(outptr + 4 * 3, 0);
                _sum4 = (v4f32)__msa_ld_w(outptr + 4 * 4, 0);
                _sum5 = (v4f32)__msa_ld_w(outptr + 4 * 5, 0);
                _sum6 = (v4f32)__msa_ld_w(outptr + 4 * 6, 0);
                _sum7 = (v4f32)__msa_ld_w(outptr + 4 * 7, 0);
                _sum8 = (v4f32)__msa_ld_w(outptr + 4 * 8, 0);
                _sum9 = (v4f32)__msa_ld_w(outptr + 4 * 9, 0);
                _suma = (v4f32)__msa_ld_w(outptr + 4 * 10, 0);
                _sumb = (v4f32)__msa_ld_w(outptr + 4 * 11, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                v4f32 _pA = bfloat2float_msa(pA);

                _sum0 = __msa_fmadd_w(_sum0, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[0])));
                _sum1 = __msa_fmadd_w(_sum1, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[1])));
                _sum2 = __msa_fmadd_w(_sum2, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[2])));
                _sum3 = __msa_fmadd_w(_sum3, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[3])));
                _sum4 = __msa_fmadd_w(_sum4, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[4])));
                _sum5 = __msa_fmadd_w(_sum5, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[5])));
                _sum6 = __msa_fmadd_w(_sum6, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[6])));
                _sum7 = __msa_fmadd_w(_sum7, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[7])));
                _sum8 = __msa_fmadd_w(_sum8, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[8])));
                _sum9 = __msa_fmadd_w(_sum9, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[9])));
                _suma = __msa_fmadd_w(_suma, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[10])));
                _sumb = __msa_fmadd_w(_sumb, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[11])));

                pA += 4;
                pB += 12;
            }

            __msa_st_w((v4i32)_sum0, outptr, 0);
            __msa_st_w((v4i32)_sum1, outptr + 4, 0);
            __msa_st_w((v4i32)_sum2, outptr + 4 * 2, 0);
            __msa_st_w((v4i32)_sum3, outptr + 4 * 3, 0);
            __msa_st_w((v4i32)_sum4, outptr + 4 * 4, 0);
            __msa_st_w((v4i32)_sum5, outptr + 4 * 5, 0);
            __msa_st_w((v4i32)_sum6, outptr + 4 * 6, 0);
            __msa_st_w((v4i32)_sum7, outptr + 4 * 7, 0);
            __msa_st_w((v4i32)_sum8, outptr + 4 * 8, 0);
            __msa_st_w((v4i32)_sum9, outptr + 4 * 9, 0);
            __msa_st_w((v4i32)_suma, outptr + 4 * 10, 0);
            __msa_st_w((v4i32)_sumb, outptr + 4 * 11, 0);

            outptr += 48;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

            v4f32 _sum0;
            v4f32 _sum1;
            v4f32 _sum2;
            v4f32 _sum3;
            v4f32 _sum4;
            v4f32 _sum5;
            v4f32 _sum6;
            v4f32 _sum7;

            if (k == 0)
            {
                _sum0 = (v4f32)__msa_fill_w(0);
                _sum1 = (v4f32)__msa_fill_w(0);
                _sum2 = (v4f32)__msa_fill_w(0);
                _sum3 = (v4f32)__msa_fill_w(0);
                _sum4 = (v4f32)__msa_fill_w(0);
                _sum5 = (v4f32)__msa_fill_w(0);
                _sum6 = (v4f32)__msa_fill_w(0);
                _sum7 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _sum0 = (v4f32)__msa_ld_w(outptr, 0);
                _sum1 = (v4f32)__msa_ld_w(outptr + 4, 0);
                _sum2 = (v4f32)__msa_ld_w(outptr + 4 * 2, 0);
                _sum3 = (v4f32)__msa_ld_w(outptr + 4 * 3, 0);
                _sum4 = (v4f32)__msa_ld_w(outptr + 4 * 4, 0);
                _sum5 = (v4f32)__msa_ld_w(outptr + 4 * 5, 0);
                _sum6 = (v4f32)__msa_ld_w(outptr + 4 * 6, 0);
                _sum7 = (v4f32)__msa_ld_w(outptr + 4 * 7, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                v4f32 _pA = bfloat2float_msa(pA);

                _sum0 = __msa_fmadd_w(_sum0, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[0])));
                _sum1 = __msa_fmadd_w(_sum1, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[1])));
                _sum2 = __msa_fmadd_w(_sum2, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[2])));
                _sum3 = __msa_fmadd_w(_sum3, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[3])));
                _sum4 = __msa_fmadd_w(_sum4, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[4])));
                _sum5 = __msa_fmadd_w(_sum5, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[5])));
                _sum6 = __msa_fmadd_w(_sum6, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[6])));
                _sum7 = __msa_fmadd_w(_sum7, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[7])));

                pA += 4;
                pB += 8;
            }

            __msa_st_w((v4i32)_sum0, outptr, 0);
            __msa_st_w((v4i32)_sum1, outptr + 4, 0);
            __msa_st_w((v4i32)_sum2, outptr + 4 * 2, 0);
            __msa_st_w((v4i32)_sum3, outptr + 4 * 3, 0);
            __msa_st_w((v4i32)_sum4, outptr + 4 * 4, 0);
            __msa_st_w((v4i32)_sum5, outptr + 4 * 5, 0);
            __msa_st_w((v4i32)_sum6, outptr + 4 * 6, 0);
            __msa_st_w((v4i32)_sum7, outptr + 4 * 7, 0);

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

            v4f32 _sum0;
            v4f32 _sum1;
            v4f32 _sum2;
            v4f32 _sum3;

            if (k == 0)
            {
                _sum0 = (v4f32)__msa_fill_w(0);
                _sum1 = (v4f32)__msa_fill_w(0);
                _sum2 = (v4f32)__msa_fill_w(0);
                _sum3 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _sum0 = (v4f32)__msa_ld_w(outptr, 0);
                _sum1 = (v4f32)__msa_ld_w(outptr + 4, 0);
                _sum2 = (v4f32)__msa_ld_w(outptr + 4 * 2, 0);
                _sum3 = (v4f32)__msa_ld_w(outptr + 4 * 3, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                v4f32 _pA = bfloat2float_msa(pA);

                _sum0 = __msa_fmadd_w(_sum0, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[0])));
                _sum1 = __msa_fmadd_w(_sum1, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[1])));
                _sum2 = __msa_fmadd_w(_sum2, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[2])));
                _sum3 = __msa_fmadd_w(_sum3, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[3])));

                pA += 4;
                pB += 4;
            }

            __msa_st_w((v4i32)_sum0, outptr, 0);
            __msa_st_w((v4i32)_sum1, outptr + 4, 0);
            __msa_st_w((v4i32)_sum2, outptr + 4 * 2, 0);
            __msa_st_w((v4i32)_sum3, outptr + 4 * 3, 0);

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* pA = pAT;

            v4f32 _sum0;
            v4f32 _sum1;

            if (k == 0)
            {
                _sum0 = (v4f32)__msa_fill_w(0);
                _sum1 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _sum0 = (v4f32)__msa_ld_w(outptr, 0);
                _sum1 = (v4f32)__msa_ld_w(outptr + 4, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                v4f32 _pA = bfloat2float_msa(pA);

                _sum0 = __msa_fmadd_w(_sum0, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[0])));
                _sum1 = __msa_fmadd_w(_sum1, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[1])));

                pA += 4;
                pB += 2;
            }

            __msa_st_w((v4i32)_sum0, outptr, 0);
            __msa_st_w((v4i32)_sum1, outptr + 4, 0);

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            const unsigned short* pA = pAT;

            v4f32 _sum0;

            if (k == 0)
            {
                _sum0 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _sum0 = (v4f32)__msa_ld_w(outptr, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                v4f32 _pA = bfloat2float_msa(pA);

                _sum0 = __msa_fmadd_w(_sum0, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[0])));

                pA += 4;
                pB += 1;
            }

            __msa_st_w((v4i32)_sum0, outptr, 0);

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
#if __mips_msa
        for (; jj + 11 < max_jj; jj += 12)
        {
            float sum0[12];
            float sum1[12];

            if (k == 0)
            {
                for (int c = 0; c < 12; c++)
                {
                    sum0[c] = 0.f;
                    sum1[c] = 0.f;
                }
            }
            else
            {
                for (int c = 0; c < 12; c++)
                {
                    sum0[c] = outptr[c * 2];
                    sum1[c] = outptr[c * 2 + 1];
                }
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                float a1 = bfloat16_to_float32(pA[1]);
                for (int c = 0; c < 12; c++)
                {
                    float b = bfloat16_to_float32(pB[c]);
                    sum0[c] += a0 * b;
                    sum1[c] += a1 * b;
                }
                pA += 2;
                pB += 12;
            }

            for (int c = 0; c < 12; c++)
            {
                outptr[c * 2] = sum0[c];
                outptr[c * 2 + 1] = sum1[c];
            }

            outptr += 24;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            float sum0[8];
            float sum1[8];

            if (k == 0)
            {
                for (int c = 0; c < 8; c++)
                {
                    sum0[c] = 0.f;
                    sum1[c] = 0.f;
                }
            }
            else
            {
                for (int c = 0; c < 8; c++)
                {
                    sum0[c] = outptr[c * 2];
                    sum1[c] = outptr[c * 2 + 1];
                }
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                float a1 = bfloat16_to_float32(pA[1]);
                for (int c = 0; c < 8; c++)
                {
                    float b = bfloat16_to_float32(pB[c]);
                    sum0[c] += a0 * b;
                    sum1[c] += a1 * b;
                }
                pA += 2;
                pB += 8;
            }

            for (int c = 0; c < 8; c++)
            {
                outptr[c * 2] = sum0[c];
                outptr[c * 2 + 1] = sum1[c];
            }

            outptr += 16;
        }
#endif // __mips_msa
        for (; jj + 3 < max_jj; jj += 4)
        {
            float sum00 = 0.f;
            float sum01 = 0.f;
            float sum10 = 0.f;
            float sum11 = 0.f;
            float sum20 = 0.f;
            float sum21 = 0.f;
            float sum30 = 0.f;
            float sum31 = 0.f;

            if (k != 0)
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
                sum20 = outptr[4];
                sum21 = outptr[5];
                sum30 = outptr[6];
                sum31 = outptr[7];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                float a1 = bfloat16_to_float32(pA[1]);
                sum00 += a0 * bfloat16_to_float32(pB[0]);
                sum01 += a1 * bfloat16_to_float32(pB[0]);
                sum10 += a0 * bfloat16_to_float32(pB[1]);
                sum11 += a1 * bfloat16_to_float32(pB[1]);
                sum20 += a0 * bfloat16_to_float32(pB[2]);
                sum21 += a1 * bfloat16_to_float32(pB[2]);
                sum30 += a0 * bfloat16_to_float32(pB[3]);
                sum31 += a1 * bfloat16_to_float32(pB[3]);
                pA += 2;
                pB += 4;
            }

            outptr[0] = sum00;
            outptr[1] = sum01;
            outptr[2] = sum10;
            outptr[3] = sum11;
            outptr[4] = sum20;
            outptr[5] = sum21;
            outptr[6] = sum30;
            outptr[7] = sum31;

            outptr += 8;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum00 = 0.f;
            float sum01 = 0.f;
            float sum10 = 0.f;
            float sum11 = 0.f;

            if (k != 0)
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                float a1 = bfloat16_to_float32(pA[1]);
                float b0 = bfloat16_to_float32(pB[0]);
                float b1 = bfloat16_to_float32(pB[1]);
                sum00 += a0 * b0;
                sum01 += a1 * b0;
                sum10 += a0 * b1;
                sum11 += a1 * b1;
                pA += 2;
                pB += 2;
            }

            outptr[0] = sum00;
            outptr[1] = sum01;
            outptr[2] = sum10;
            outptr[3] = sum11;

            outptr += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            float sum0 = 0.f;
            float sum1 = 0.f;

            if (k != 0)
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                float b0 = bfloat16_to_float32(pB[0]);
                sum0 += bfloat16_to_float32(pA[0]) * b0;
                sum1 += bfloat16_to_float32(pA[1]) * b0;
                pA += 2;
                pB += 1;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;

            outptr += 2;
        }

        pAT += max_kk * 2;
    }
    for (; ii < max_ii; ii += 1)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
#if __mips_msa
        for (; jj + 11 < max_jj; jj += 12)
        {
            float sum[12];

            if (k == 0)
            {
                for (int c = 0; c < 12; c++)
                    sum[c] = 0.f;
            }
            else
            {
                for (int c = 0; c < 12; c++)
                    sum[c] = outptr[c];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                for (int c = 0; c < 12; c++)
                    sum[c] += a0 * bfloat16_to_float32(pB[c]);
                pA += 1;
                pB += 12;
            }

            for (int c = 0; c < 12; c++)
                outptr[c] = sum[c];

            outptr += 12;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            float sum[8];

            if (k == 0)
            {
                for (int c = 0; c < 8; c++)
                    sum[c] = 0.f;
            }
            else
            {
                for (int c = 0; c < 8; c++)
                    sum[c] = outptr[c];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                for (int c = 0; c < 8; c++)
                    sum[c] += a0 * bfloat16_to_float32(pB[c]);
                pA += 1;
                pB += 8;
            }

            for (int c = 0; c < 8; c++)
                outptr[c] = sum[c];

            outptr += 8;
        }
#endif // __mips_msa
        for (; jj + 3 < max_jj; jj += 4)
        {
            float sum0 = 0.f;
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;

            if (k != 0)
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
                sum2 = outptr[2];
                sum3 = outptr[3];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                sum0 += a0 * bfloat16_to_float32(pB[0]);
                sum1 += a0 * bfloat16_to_float32(pB[1]);
                sum2 += a0 * bfloat16_to_float32(pB[2]);
                sum3 += a0 * bfloat16_to_float32(pB[3]);
                pA += 1;
                pB += 4;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr[2] = sum2;
            outptr[3] = sum3;

            outptr += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0 = 0.f;
            float sum1 = 0.f;

            if (k != 0)
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                sum0 += a0 * bfloat16_to_float32(pB[0]);
                sum1 += a0 * bfloat16_to_float32(pB[1]);
                pA += 1;
                pB += 2;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float sum = 0.f;

            if (k != 0)
            {
                sum = outptr[0];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                sum += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);
                pA += 1;
                pB += 1;
            }

            outptr[0] = sum;

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void unpack_output_tile_fp32_to_bf16(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta, int output_transpose, int output_elemtype)
{
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const float* pp = topT;

    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const int c_elempack = C.elempack;
    const float* pC = C;

    int ii = 0;
#if __mips_msa
    for (; ii + 3 < max_ii; ii += 4)
    {
        unsigned short* p0 = (unsigned short*)top_blob + (i + ii) / out_elempack * out_hstep * out_elempack + j * out_elempack;
        float* p0f32 = (float*)top_blob + (i + ii) / out_elempack * out_hstep * out_elempack + j * out_elempack;

        const float* pCi = pC;
        if (pCi)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pCi = (const float*)C + (i + ii);
            }
            if (broadcast_type_C == 4)
            {
                pCi = (const float*)C + j;
            }
        }

        v4f32 _valpha = __msa_fill_w_f32(alpha);

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            v4f32 _sum0;
            v4f32 _sum1;
            v4f32 _sum2;
            v4f32 _sum3;
            v4f32 _sum4;
            v4f32 _sum5;
            v4f32 _sum6;
            v4f32 _sum7;
            v4f32 _sum8;
            v4f32 _sum9;
            v4f32 _suma;
            v4f32 _sumb;

            _sum0 = (v4f32)__msa_ld_w(pp, 0);
            _sum1 = (v4f32)__msa_ld_w(pp + 4, 0);
            _sum2 = (v4f32)__msa_ld_w(pp + 4 * 2, 0);
            _sum3 = (v4f32)__msa_ld_w(pp + 4 * 3, 0);
            _sum4 = (v4f32)__msa_ld_w(pp + 4 * 4, 0);
            _sum5 = (v4f32)__msa_ld_w(pp + 4 * 5, 0);
            _sum6 = (v4f32)__msa_ld_w(pp + 4 * 6, 0);
            _sum7 = (v4f32)__msa_ld_w(pp + 4 * 7, 0);
            _sum8 = (v4f32)__msa_ld_w(pp + 4 * 8, 0);
            _sum9 = (v4f32)__msa_ld_w(pp + 4 * 9, 0);
            _suma = (v4f32)__msa_ld_w(pp + 4 * 10, 0);
            _sumb = (v4f32)__msa_ld_w(pp + 4 * 11, 0);
            pp += 48;

            if (pCi)
            {
                v4f32 _beta = __msa_fill_w_f32(beta);
                if (broadcast_type_C == 0)
                {
                    v4f32 _c = __msa_fill_w_f32(pCi[0]);
                    _c = __msa_fmul_w(_c, _beta);
                    _sum0 = __msa_fadd_w(_sum0, _c);
                    _sum1 = __msa_fadd_w(_sum1, _c);
                    _sum2 = __msa_fadd_w(_sum2, _c);
                    _sum3 = __msa_fadd_w(_sum3, _c);
                    _sum4 = __msa_fadd_w(_sum4, _c);
                    _sum5 = __msa_fadd_w(_sum5, _c);
                    _sum6 = __msa_fadd_w(_sum6, _c);
                    _sum7 = __msa_fadd_w(_sum7, _c);
                    _sum8 = __msa_fadd_w(_sum8, _c);
                    _sum9 = __msa_fadd_w(_sum9, _c);
                    _suma = __msa_fadd_w(_suma, _c);
                    _sumb = __msa_fadd_w(_sumb, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _c = (v4f32)__msa_ld_w(pCi, 0);
                    _c = __msa_fmul_w(_c, _beta);
                    _sum0 = __msa_fadd_w(_sum0, _c);
                    _sum1 = __msa_fadd_w(_sum1, _c);
                    _sum2 = __msa_fadd_w(_sum2, _c);
                    _sum3 = __msa_fadd_w(_sum3, _c);
                    _sum4 = __msa_fadd_w(_sum4, _c);
                    _sum5 = __msa_fadd_w(_sum5, _c);
                    _sum6 = __msa_fadd_w(_sum6, _c);
                    _sum7 = __msa_fadd_w(_sum7, _c);
                    _sum8 = __msa_fadd_w(_sum8, _c);
                    _sum9 = __msa_fadd_w(_sum9, _c);
                    _suma = __msa_fadd_w(_suma, _c);
                    _sumb = __msa_fadd_w(_sumb, _c);
                }
                if (broadcast_type_C == 3)
                {
                    // broadcast_type_C == 3 means full C matrix
                    // C is stored in topT_tile in pack_A_tile order: jj-major, 4 floats per jj
                    if (c_elempack == 4)
                    {
                        const float* pCj = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + (j + jj) * 4;
                        _sum0 = __msa_fmadd_w(_sum0, _beta, (v4f32)__msa_ld_w(pCj, 0));
                        _sum1 = __msa_fmadd_w(_sum1, _beta, (v4f32)__msa_ld_w(pCj + 4, 0));
                        _sum2 = __msa_fmadd_w(_sum2, _beta, (v4f32)__msa_ld_w(pCj + 8, 0));
                        _sum3 = __msa_fmadd_w(_sum3, _beta, (v4f32)__msa_ld_w(pCj + 12, 0));
                        _sum4 = __msa_fmadd_w(_sum4, _beta, (v4f32)__msa_ld_w(pCj + 16, 0));
                        _sum5 = __msa_fmadd_w(_sum5, _beta, (v4f32)__msa_ld_w(pCj + 20, 0));
                        _sum6 = __msa_fmadd_w(_sum6, _beta, (v4f32)__msa_ld_w(pCj + 24, 0));
                        _sum7 = __msa_fmadd_w(_sum7, _beta, (v4f32)__msa_ld_w(pCj + 28, 0));
                        _sum8 = __msa_fmadd_w(_sum8, _beta, (v4f32)__msa_ld_w(pCj + 32, 0));
                        _sum9 = __msa_fmadd_w(_sum9, _beta, (v4f32)__msa_ld_w(pCj + 36, 0));
                        _suma = __msa_fmadd_w(_suma, _beta, (v4f32)__msa_ld_w(pCj + 40, 0));
                        _sumb = __msa_fmadd_w(_sumb, _beta, (v4f32)__msa_ld_w(pCj + 44, 0));
                    }
                    if (c_elempack == 1)
                    {
                        const float* pCj = (const float*)C + (i + ii) * c_hstep + (j + jj);
                        v4f32* sums[12] = {&_sum0, &_sum1, &_sum2, &_sum3, &_sum4, &_sum5, &_sum6, &_sum7, &_sum8, &_sum9, &_suma, &_sumb};
                        for (int c = 0; c < 12; c++)
                        {
                            float tmp[4];
                            tmp[0] = pCj[0];
                            tmp[1] = pCj[c_hstep];
                            tmp[2] = pCj[c_hstep * 2];
                            tmp[3] = pCj[c_hstep * 3];
                            v4f32 _cv = (v4f32)__msa_ld_w(tmp, 0);
                            *sums[c] = __msa_fmadd_w(*sums[c], _beta, _cv);
                            pCj += 1;
                        }
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _sum0 = __msa_fmadd_w(_sum0, _beta, __msa_fill_w_f32(pCi[0]));
                    _sum1 = __msa_fmadd_w(_sum1, _beta, __msa_fill_w_f32(pCi[1]));
                    _sum2 = __msa_fmadd_w(_sum2, _beta, __msa_fill_w_f32(pCi[2]));
                    _sum3 = __msa_fmadd_w(_sum3, _beta, __msa_fill_w_f32(pCi[3]));
                    _sum4 = __msa_fmadd_w(_sum4, _beta, __msa_fill_w_f32(pCi[4]));
                    _sum5 = __msa_fmadd_w(_sum5, _beta, __msa_fill_w_f32(pCi[5]));
                    _sum6 = __msa_fmadd_w(_sum6, _beta, __msa_fill_w_f32(pCi[6]));
                    _sum7 = __msa_fmadd_w(_sum7, _beta, __msa_fill_w_f32(pCi[7]));
                    _sum8 = __msa_fmadd_w(_sum8, _beta, __msa_fill_w_f32(pCi[8]));
                    _sum9 = __msa_fmadd_w(_sum9, _beta, __msa_fill_w_f32(pCi[9]));
                    _suma = __msa_fmadd_w(_suma, _beta, __msa_fill_w_f32(pCi[10]));
                    _sumb = __msa_fmadd_w(_sumb, _beta, __msa_fill_w_f32(pCi[11]));
                    pCi += 12;
                }
            }

            _sum0 = __msa_fmul_w(_sum0, _valpha);
            _sum1 = __msa_fmul_w(_sum1, _valpha);
            _sum2 = __msa_fmul_w(_sum2, _valpha);
            _sum3 = __msa_fmul_w(_sum3, _valpha);
            _sum4 = __msa_fmul_w(_sum4, _valpha);
            _sum5 = __msa_fmul_w(_sum5, _valpha);
            _sum6 = __msa_fmul_w(_sum6, _valpha);
            _sum7 = __msa_fmul_w(_sum7, _valpha);
            _sum8 = __msa_fmul_w(_sum8, _valpha);
            _sum9 = __msa_fmul_w(_sum9, _valpha);
            _suma = __msa_fmul_w(_suma, _valpha);
            _sumb = __msa_fmul_w(_sumb, _valpha);

            if (output_transpose)
            {
                if (output_elemtype == 1)
                {
                    // fp32 output, transposed: row=j, col=i
                    float tmp0[4], tmp1[4], tmp2[4], tmp3[4];
                    float tmp4[4], tmp5[4], tmp6[4], tmp7[4];
                    float tmp8[4], tmp9[4], tmpa[4], tmpb[4];
                    __msa_st_w((v4i32)_sum0, tmp0, 0);
                    __msa_st_w((v4i32)_sum1, tmp1, 0);
                    __msa_st_w((v4i32)_sum2, tmp2, 0);
                    __msa_st_w((v4i32)_sum3, tmp3, 0);
                    __msa_st_w((v4i32)_sum4, tmp4, 0);
                    __msa_st_w((v4i32)_sum5, tmp5, 0);
                    __msa_st_w((v4i32)_sum6, tmp6, 0);
                    __msa_st_w((v4i32)_sum7, tmp7, 0);
                    __msa_st_w((v4i32)_sum8, tmp8, 0);
                    __msa_st_w((v4i32)_sum9, tmp9, 0);
                    __msa_st_w((v4i32)_suma, tmpa, 0);
                    __msa_st_w((v4i32)_sumb, tmpb, 0);
                    float* ptmp[12] = {tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmpa, tmpb};
                    for (int c = 0; c < 12; c++)
                    {
                        int col = j + jj + c;
                        for (int r = 0; r < 4; r++)
                        {
                            int row = i + ii + r;
                            size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                            ((float*)top_blob)[offset] = ptmp[c][r];
                        }
                    }
                }
                else
                {
                    // bf16 output, transposed
                    float tmp0[4], tmp1[4], tmp2[4], tmp3[4];
                    float tmp4[4], tmp5[4], tmp6[4], tmp7[4];
                    float tmp8[4], tmp9[4], tmpa[4], tmpb[4];
                    __msa_st_w((v4i32)_sum0, tmp0, 0);
                    __msa_st_w((v4i32)_sum1, tmp1, 0);
                    __msa_st_w((v4i32)_sum2, tmp2, 0);
                    __msa_st_w((v4i32)_sum3, tmp3, 0);
                    __msa_st_w((v4i32)_sum4, tmp4, 0);
                    __msa_st_w((v4i32)_sum5, tmp5, 0);
                    __msa_st_w((v4i32)_sum6, tmp6, 0);
                    __msa_st_w((v4i32)_sum7, tmp7, 0);
                    __msa_st_w((v4i32)_sum8, tmp8, 0);
                    __msa_st_w((v4i32)_sum9, tmp9, 0);
                    __msa_st_w((v4i32)_suma, tmpa, 0);
                    __msa_st_w((v4i32)_sumb, tmpb, 0);
                    float* ptmp2[12] = {tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmpa, tmpb};
                    for (int c = 0; c < 12; c++)
                    {
                        int col = j + jj + c;
                        for (int r = 0; r < 4; r++)
                        {
                            int row = i + ii + r;
                            size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(ptmp2[c][r]);
                        }
                    }
                }
            }
            else
            {
                if (out_elempack == 4)
                {
                    if (output_elemtype == 1)
                    {
                        __msa_st_w((v4i32)_sum0, p0f32, 0);
                        __msa_st_w((v4i32)_sum1, p0f32 + 4, 0);
                        __msa_st_w((v4i32)_sum2, p0f32 + 4 * 2, 0);
                        __msa_st_w((v4i32)_sum3, p0f32 + 4 * 3, 0);
                        __msa_st_w((v4i32)_sum4, p0f32 + 4 * 4, 0);
                        __msa_st_w((v4i32)_sum5, p0f32 + 4 * 5, 0);
                        __msa_st_w((v4i32)_sum6, p0f32 + 4 * 6, 0);
                        __msa_st_w((v4i32)_sum7, p0f32 + 4 * 7, 0);
                        __msa_st_w((v4i32)_sum8, p0f32 + 4 * 8, 0);
                        __msa_st_w((v4i32)_sum9, p0f32 + 4 * 9, 0);
                        __msa_st_w((v4i32)_suma, p0f32 + 4 * 10, 0);
                        __msa_st_w((v4i32)_sumb, p0f32 + 4 * 11, 0);
                        p0f32 += 48;
                    }
                    else
                    {
                        float2bfloat_msa_store(_sum0, p0);
                        float2bfloat_msa_store(_sum1, p0 + 4);
                        float2bfloat_msa_store(_sum2, p0 + 4 * 2);
                        float2bfloat_msa_store(_sum3, p0 + 4 * 3);
                        float2bfloat_msa_store(_sum4, p0 + 4 * 4);
                        float2bfloat_msa_store(_sum5, p0 + 4 * 5);
                        float2bfloat_msa_store(_sum6, p0 + 4 * 6);
                        float2bfloat_msa_store(_sum7, p0 + 4 * 7);
                        float2bfloat_msa_store(_sum8, p0 + 4 * 8);
                        float2bfloat_msa_store(_sum9, p0 + 4 * 9);
                        float2bfloat_msa_store(_suma, p0 + 4 * 10);
                        float2bfloat_msa_store(_sumb, p0 + 4 * 11);
                        p0 += 48;
                    }
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                    transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);
                    transpose4x4_ps(_sum8, _sum9, _suma, _sumb);

                    if (output_elemtype == 1)
                    {
                        __msa_st_w((v4i32)_sum0, p0f32, 0);
                        __msa_st_w((v4i32)_sum4, p0f32 + 4, 0);
                        __msa_st_w((v4i32)_sum8, p0f32 + 8, 0);
                        __msa_st_w((v4i32)_sum1, p0f32 + out_hstep, 0);
                        __msa_st_w((v4i32)_sum5, p0f32 + out_hstep + 4, 0);
                        __msa_st_w((v4i32)_sum9, p0f32 + out_hstep + 8, 0);
                        __msa_st_w((v4i32)_sum2, p0f32 + out_hstep * 2, 0);
                        __msa_st_w((v4i32)_sum6, p0f32 + out_hstep * 2 + 4, 0);
                        __msa_st_w((v4i32)_suma, p0f32 + out_hstep * 2 + 8, 0);
                        __msa_st_w((v4i32)_sum3, p0f32 + out_hstep * 3, 0);
                        __msa_st_w((v4i32)_sum7, p0f32 + out_hstep * 3 + 4, 0);
                        __msa_st_w((v4i32)_sumb, p0f32 + out_hstep * 3 + 8, 0);
                        p0f32 += 12;
                    }
                    else
                    {
                        float2bfloat_msa_store(_sum0, p0);
                        float2bfloat_msa_store(_sum4, p0 + 4);
                        float2bfloat_msa_store(_sum8, p0 + 8);
                        float2bfloat_msa_store(_sum1, p0 + out_hstep);
                        float2bfloat_msa_store(_sum5, p0 + out_hstep + 4);
                        float2bfloat_msa_store(_sum9, p0 + out_hstep + 8);
                        float2bfloat_msa_store(_sum2, p0 + out_hstep * 2);
                        float2bfloat_msa_store(_sum6, p0 + out_hstep * 2 + 4);
                        float2bfloat_msa_store(_suma, p0 + out_hstep * 2 + 8);
                        float2bfloat_msa_store(_sum3, p0 + out_hstep * 3);
                        float2bfloat_msa_store(_sum7, p0 + out_hstep * 3 + 4);
                        float2bfloat_msa_store(_sumb, p0 + out_hstep * 3 + 8);
                        p0 += 12;
                    }
                }
            }
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            v4f32 _sum0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _sum1 = (v4f32)__msa_ld_w(pp + 4, 0);
            v4f32 _sum2 = (v4f32)__msa_ld_w(pp + 4 * 2, 0);
            v4f32 _sum3 = (v4f32)__msa_ld_w(pp + 4 * 3, 0);
            v4f32 _sum4 = (v4f32)__msa_ld_w(pp + 4 * 4, 0);
            v4f32 _sum5 = (v4f32)__msa_ld_w(pp + 4 * 5, 0);
            v4f32 _sum6 = (v4f32)__msa_ld_w(pp + 4 * 6, 0);
            v4f32 _sum7 = (v4f32)__msa_ld_w(pp + 4 * 7, 0);
            pp += 32;

            if (pCi)
            {
                v4f32 _beta = __msa_fill_w_f32(beta);
                if (broadcast_type_C == 0)
                {
                    v4f32 _c = __msa_fmul_w(__msa_fill_w_f32(pCi[0]), _beta);
                    _sum0 = __msa_fadd_w(_sum0, _c);
                    _sum1 = __msa_fadd_w(_sum1, _c);
                    _sum2 = __msa_fadd_w(_sum2, _c);
                    _sum3 = __msa_fadd_w(_sum3, _c);
                    _sum4 = __msa_fadd_w(_sum4, _c);
                    _sum5 = __msa_fadd_w(_sum5, _c);
                    _sum6 = __msa_fadd_w(_sum6, _c);
                    _sum7 = __msa_fadd_w(_sum7, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _c = __msa_fmul_w((v4f32)__msa_ld_w(pCi, 0), _beta);
                    _sum0 = __msa_fadd_w(_sum0, _c);
                    _sum1 = __msa_fadd_w(_sum1, _c);
                    _sum2 = __msa_fadd_w(_sum2, _c);
                    _sum3 = __msa_fadd_w(_sum3, _c);
                    _sum4 = __msa_fadd_w(_sum4, _c);
                    _sum5 = __msa_fadd_w(_sum5, _c);
                    _sum6 = __msa_fadd_w(_sum6, _c);
                    _sum7 = __msa_fadd_w(_sum7, _c);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        const float* pCj = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + (j + jj) * 4;
                        _sum0 = __msa_fmadd_w(_sum0, _beta, (v4f32)__msa_ld_w(pCj, 0));
                        _sum1 = __msa_fmadd_w(_sum1, _beta, (v4f32)__msa_ld_w(pCj + 4, 0));
                        _sum2 = __msa_fmadd_w(_sum2, _beta, (v4f32)__msa_ld_w(pCj + 8, 0));
                        _sum3 = __msa_fmadd_w(_sum3, _beta, (v4f32)__msa_ld_w(pCj + 12, 0));
                        _sum4 = __msa_fmadd_w(_sum4, _beta, (v4f32)__msa_ld_w(pCj + 16, 0));
                        _sum5 = __msa_fmadd_w(_sum5, _beta, (v4f32)__msa_ld_w(pCj + 20, 0));
                        _sum6 = __msa_fmadd_w(_sum6, _beta, (v4f32)__msa_ld_w(pCj + 24, 0));
                        _sum7 = __msa_fmadd_w(_sum7, _beta, (v4f32)__msa_ld_w(pCj + 28, 0));
                    }
                    if (c_elempack == 1)
                    {
                        const float* pCj = (const float*)C + (i + ii) * c_hstep + (j + jj);
                        v4f32* sums[8] = {&_sum0, &_sum1, &_sum2, &_sum3, &_sum4, &_sum5, &_sum6, &_sum7};
                        for (int c = 0; c < 8; c++)
                        {
                            float tmp[4];
                            tmp[0] = pCj[0];
                            tmp[1] = pCj[c_hstep];
                            tmp[2] = pCj[c_hstep * 2];
                            tmp[3] = pCj[c_hstep * 3];
                            v4f32 _cv = (v4f32)__msa_ld_w(tmp, 0);
                            *sums[c] = __msa_fmadd_w(*sums[c], _beta, _cv);
                            pCj += 1;
                        }
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _sum0 = __msa_fmadd_w(_sum0, _beta, __msa_fill_w_f32(pCi[0]));
                    _sum1 = __msa_fmadd_w(_sum1, _beta, __msa_fill_w_f32(pCi[1]));
                    _sum2 = __msa_fmadd_w(_sum2, _beta, __msa_fill_w_f32(pCi[2]));
                    _sum3 = __msa_fmadd_w(_sum3, _beta, __msa_fill_w_f32(pCi[3]));
                    _sum4 = __msa_fmadd_w(_sum4, _beta, __msa_fill_w_f32(pCi[4]));
                    _sum5 = __msa_fmadd_w(_sum5, _beta, __msa_fill_w_f32(pCi[5]));
                    _sum6 = __msa_fmadd_w(_sum6, _beta, __msa_fill_w_f32(pCi[6]));
                    _sum7 = __msa_fmadd_w(_sum7, _beta, __msa_fill_w_f32(pCi[7]));
                    pCi += 8;
                }
            }

            _sum0 = __msa_fmul_w(_sum0, _valpha);
            _sum1 = __msa_fmul_w(_sum1, _valpha);
            _sum2 = __msa_fmul_w(_sum2, _valpha);
            _sum3 = __msa_fmul_w(_sum3, _valpha);
            _sum4 = __msa_fmul_w(_sum4, _valpha);
            _sum5 = __msa_fmul_w(_sum5, _valpha);
            _sum6 = __msa_fmul_w(_sum6, _valpha);
            _sum7 = __msa_fmul_w(_sum7, _valpha);

            if (output_transpose)
            {
                float tmp0[4], tmp1[4], tmp2[4], tmp3[4], tmp4[4], tmp5[4], tmp6[4], tmp7[4];
                __msa_st_w((v4i32)_sum0, tmp0, 0);
                __msa_st_w((v4i32)_sum1, tmp1, 0);
                __msa_st_w((v4i32)_sum2, tmp2, 0);
                __msa_st_w((v4i32)_sum3, tmp3, 0);
                __msa_st_w((v4i32)_sum4, tmp4, 0);
                __msa_st_w((v4i32)_sum5, tmp5, 0);
                __msa_st_w((v4i32)_sum6, tmp6, 0);
                __msa_st_w((v4i32)_sum7, tmp7, 0);
                float* ptmp[8] = {tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7};
                for (int c = 0; c < 8; c++)
                {
                    int col = j + jj + c;
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = ptmp[c][r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(ptmp[c][r]);
                    }
                }
            }
            else
            {
                if (out_elempack == 4)
                {
                    if (output_elemtype == 1)
                    {
                        __msa_st_w((v4i32)_sum0, p0f32, 0);
                        __msa_st_w((v4i32)_sum1, p0f32 + 4, 0);
                        __msa_st_w((v4i32)_sum2, p0f32 + 8, 0);
                        __msa_st_w((v4i32)_sum3, p0f32 + 12, 0);
                        __msa_st_w((v4i32)_sum4, p0f32 + 16, 0);
                        __msa_st_w((v4i32)_sum5, p0f32 + 20, 0);
                        __msa_st_w((v4i32)_sum6, p0f32 + 24, 0);
                        __msa_st_w((v4i32)_sum7, p0f32 + 28, 0);
                        p0f32 += 32;
                    }
                    else
                    {
                        float2bfloat_msa_store(_sum0, p0);
                        float2bfloat_msa_store(_sum1, p0 + 4);
                        float2bfloat_msa_store(_sum2, p0 + 8);
                        float2bfloat_msa_store(_sum3, p0 + 12);
                        float2bfloat_msa_store(_sum4, p0 + 16);
                        float2bfloat_msa_store(_sum5, p0 + 20);
                        float2bfloat_msa_store(_sum6, p0 + 24);
                        float2bfloat_msa_store(_sum7, p0 + 28);
                        p0 += 32;
                    }
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                    transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);
                    if (output_elemtype == 1)
                    {
                        __msa_st_w((v4i32)_sum0, p0f32, 0);
                        __msa_st_w((v4i32)_sum4, p0f32 + 4, 0);
                        __msa_st_w((v4i32)_sum1, p0f32 + out_hstep, 0);
                        __msa_st_w((v4i32)_sum5, p0f32 + out_hstep + 4, 0);
                        __msa_st_w((v4i32)_sum2, p0f32 + out_hstep * 2, 0);
                        __msa_st_w((v4i32)_sum6, p0f32 + out_hstep * 2 + 4, 0);
                        __msa_st_w((v4i32)_sum3, p0f32 + out_hstep * 3, 0);
                        __msa_st_w((v4i32)_sum7, p0f32 + out_hstep * 3 + 4, 0);
                        p0f32 += 8;
                    }
                    else
                    {
                        float2bfloat_msa_store(_sum0, p0);
                        float2bfloat_msa_store(_sum4, p0 + 4);
                        float2bfloat_msa_store(_sum1, p0 + out_hstep);
                        float2bfloat_msa_store(_sum5, p0 + out_hstep + 4);
                        float2bfloat_msa_store(_sum2, p0 + out_hstep * 2);
                        float2bfloat_msa_store(_sum6, p0 + out_hstep * 2 + 4);
                        float2bfloat_msa_store(_sum3, p0 + out_hstep * 3);
                        float2bfloat_msa_store(_sum7, p0 + out_hstep * 3 + 4);
                        p0 += 8;
                    }
                }
            }
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            v4f32 _sum0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _sum1 = (v4f32)__msa_ld_w(pp + 4, 0);
            v4f32 _sum2 = (v4f32)__msa_ld_w(pp + 8, 0);
            v4f32 _sum3 = (v4f32)__msa_ld_w(pp + 12, 0);
            pp += 16;

            if (pCi)
            {
                v4f32 _beta = __msa_fill_w_f32(beta);
                if (broadcast_type_C == 0)
                {
                    v4f32 _c = __msa_fmul_w(__msa_fill_w_f32(pCi[0]), _beta);
                    _sum0 = __msa_fadd_w(_sum0, _c);
                    _sum1 = __msa_fadd_w(_sum1, _c);
                    _sum2 = __msa_fadd_w(_sum2, _c);
                    _sum3 = __msa_fadd_w(_sum3, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _c = __msa_fmul_w((v4f32)__msa_ld_w(pCi, 0), _beta);
                    _sum0 = __msa_fadd_w(_sum0, _c);
                    _sum1 = __msa_fadd_w(_sum1, _c);
                    _sum2 = __msa_fadd_w(_sum2, _c);
                    _sum3 = __msa_fadd_w(_sum3, _c);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        const float* pCj = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + (j + jj) * 4;
                        _sum0 = __msa_fmadd_w(_sum0, _beta, (v4f32)__msa_ld_w(pCj, 0));
                        _sum1 = __msa_fmadd_w(_sum1, _beta, (v4f32)__msa_ld_w(pCj + 4, 0));
                        _sum2 = __msa_fmadd_w(_sum2, _beta, (v4f32)__msa_ld_w(pCj + 8, 0));
                        _sum3 = __msa_fmadd_w(_sum3, _beta, (v4f32)__msa_ld_w(pCj + 12, 0));
                    }
                    if (c_elempack == 1)
                    {
                        const float* pCj = (const float*)C + (i + ii) * c_hstep + (j + jj);
                        v4f32* sums[4] = {&_sum0, &_sum1, &_sum2, &_sum3};
                        for (int c = 0; c < 4; c++)
                        {
                            float tmp[4];
                            tmp[0] = pCj[0];
                            tmp[1] = pCj[c_hstep];
                            tmp[2] = pCj[c_hstep * 2];
                            tmp[3] = pCj[c_hstep * 3];
                            v4f32 _cv = (v4f32)__msa_ld_w(tmp, 0);
                            *sums[c] = __msa_fmadd_w(*sums[c], _beta, _cv);
                            pCj += 1;
                        }
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _sum0 = __msa_fmadd_w(_sum0, _beta, __msa_fill_w_f32(pCi[0]));
                    _sum1 = __msa_fmadd_w(_sum1, _beta, __msa_fill_w_f32(pCi[1]));
                    _sum2 = __msa_fmadd_w(_sum2, _beta, __msa_fill_w_f32(pCi[2]));
                    _sum3 = __msa_fmadd_w(_sum3, _beta, __msa_fill_w_f32(pCi[3]));
                    pCi += 4;
                }
            }

            _sum0 = __msa_fmul_w(_sum0, _valpha);
            _sum1 = __msa_fmul_w(_sum1, _valpha);
            _sum2 = __msa_fmul_w(_sum2, _valpha);
            _sum3 = __msa_fmul_w(_sum3, _valpha);

            if (output_transpose)
            {
                float tmp0[4], tmp1[4], tmp2[4], tmp3[4];
                __msa_st_w((v4i32)_sum0, tmp0, 0);
                __msa_st_w((v4i32)_sum1, tmp1, 0);
                __msa_st_w((v4i32)_sum2, tmp2, 0);
                __msa_st_w((v4i32)_sum3, tmp3, 0);
                float* ptmp[4] = {tmp0, tmp1, tmp2, tmp3};
                for (int c = 0; c < 4; c++)
                {
                    int col = j + jj + c;
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = ptmp[c][r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(ptmp[c][r]);
                    }
                }
            }
            else
            {
                if (out_elempack == 4)
                {
                    if (output_elemtype == 1)
                    {
                        __msa_st_w((v4i32)_sum0, p0f32, 0);
                        __msa_st_w((v4i32)_sum1, p0f32 + 4, 0);
                        __msa_st_w((v4i32)_sum2, p0f32 + 8, 0);
                        __msa_st_w((v4i32)_sum3, p0f32 + 12, 0);
                        p0f32 += 16;
                    }
                    else
                    {
                        float2bfloat_msa_store(_sum0, p0);
                        float2bfloat_msa_store(_sum1, p0 + 4);
                        float2bfloat_msa_store(_sum2, p0 + 8);
                        float2bfloat_msa_store(_sum3, p0 + 12);
                        p0 += 16;
                    }
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                    if (output_elemtype == 1)
                    {
                        __msa_st_w((v4i32)_sum0, p0f32, 0);
                        __msa_st_w((v4i32)_sum1, p0f32 + out_hstep, 0);
                        __msa_st_w((v4i32)_sum2, p0f32 + out_hstep * 2, 0);
                        __msa_st_w((v4i32)_sum3, p0f32 + out_hstep * 3, 0);
                        p0f32 += 4;
                    }
                    else
                    {
                        float2bfloat_msa_store(_sum0, p0);
                        float2bfloat_msa_store(_sum1, p0 + out_hstep);
                        float2bfloat_msa_store(_sum2, p0 + out_hstep * 2);
                        float2bfloat_msa_store(_sum3, p0 + out_hstep * 3);
                        p0 += 4;
                    }
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            v4f32 _sum0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _sum1 = (v4f32)__msa_ld_w(pp + 4, 0);
            pp += 8;

            if (pCi)
            {
                v4f32 _beta = __msa_fill_w_f32(beta);
                if (broadcast_type_C == 0)
                {
                    v4f32 _c = __msa_fmul_w(__msa_fill_w_f32(pCi[0]), _beta);
                    _sum0 = __msa_fadd_w(_sum0, _c);
                    _sum1 = __msa_fadd_w(_sum1, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _c = __msa_fmul_w((v4f32)__msa_ld_w(pCi, 0), _beta);
                    _sum0 = __msa_fadd_w(_sum0, _c);
                    _sum1 = __msa_fadd_w(_sum1, _c);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        const float* pCj = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + (j + jj) * 4;
                        _sum0 = __msa_fmadd_w(_sum0, _beta, (v4f32)__msa_ld_w(pCj, 0));
                        _sum1 = __msa_fmadd_w(_sum1, _beta, (v4f32)__msa_ld_w(pCj + 4, 0));
                    }
                    if (c_elempack == 1)
                    {
                        const float* pCj = (const float*)C + (i + ii) * c_hstep + (j + jj);
                        float tmp0[4], tmp1[4];
                        tmp0[0] = pCj[0];
                        tmp0[1] = pCj[c_hstep];
                        tmp0[2] = pCj[c_hstep * 2];
                        tmp0[3] = pCj[c_hstep * 3];
                        tmp1[0] = pCj[1];
                        tmp1[1] = pCj[c_hstep + 1];
                        tmp1[2] = pCj[c_hstep * 2 + 1];
                        tmp1[3] = pCj[c_hstep * 3 + 1];
                        _sum0 = __msa_fmadd_w(_sum0, _beta, (v4f32)__msa_ld_w(tmp0, 0));
                        _sum1 = __msa_fmadd_w(_sum1, _beta, (v4f32)__msa_ld_w(tmp1, 0));
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _sum0 = __msa_fmadd_w(_sum0, _beta, __msa_fill_w_f32(pCi[0]));
                    _sum1 = __msa_fmadd_w(_sum1, _beta, __msa_fill_w_f32(pCi[1]));
                    pCi += 2;
                }
            }

            _sum0 = __msa_fmul_w(_sum0, _valpha);
            _sum1 = __msa_fmul_w(_sum1, _valpha);

            if (output_transpose)
            {
                float tmp0[4], tmp1[4];
                __msa_st_w((v4i32)_sum0, tmp0, 0);
                __msa_st_w((v4i32)_sum1, tmp1, 0);
                for (int c = 0; c < 2; c++)
                {
                    float* t = c == 0 ? tmp0 : tmp1;
                    int col = j + jj + c;
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = t[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(t[r]);
                    }
                }
            }
            else
            {
                if (out_elempack == 4)
                {
                    if (output_elemtype == 1)
                    {
                        __msa_st_w((v4i32)_sum0, p0f32, 0);
                        __msa_st_w((v4i32)_sum1, p0f32 + 4, 0);
                        p0f32 += 8;
                    }
                    else
                    {
                        float2bfloat_msa_store(_sum0, p0);
                        float2bfloat_msa_store(_sum1, p0 + 4);
                        p0 += 8;
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[4], tmp1[4];
                    __msa_st_w((v4i32)_sum0, tmp0, 0);
                    __msa_st_w((v4i32)_sum1, tmp1, 0);
                    if (output_elemtype == 1)
                    {
                        float* p0f = (float*)p0f32;
                        p0f[0] = tmp0[0];
                        p0f[1] = tmp1[0];
                        p0f[out_hstep] = tmp0[1];
                        p0f[out_hstep + 1] = tmp1[1];
                        p0f[out_hstep * 2] = tmp0[2];
                        p0f[out_hstep * 2 + 1] = tmp1[2];
                        p0f[out_hstep * 3] = tmp0[3];
                        p0f[out_hstep * 3 + 1] = tmp1[3];
                        p0f32 += 2;
                    }
                    else
                    {
                        p0[0] = float32_to_bfloat16(tmp0[0]);
                        p0[1] = float32_to_bfloat16(tmp1[0]);
                        p0[out_hstep] = float32_to_bfloat16(tmp0[1]);
                        p0[out_hstep + 1] = float32_to_bfloat16(tmp1[1]);
                        p0[out_hstep * 2] = float32_to_bfloat16(tmp0[2]);
                        p0[out_hstep * 2 + 1] = float32_to_bfloat16(tmp1[2]);
                        p0[out_hstep * 3] = float32_to_bfloat16(tmp0[3]);
                        p0[out_hstep * 3 + 1] = float32_to_bfloat16(tmp1[3]);
                        p0 += 2;
                    }
                }
            }
        }
        for (; jj < max_jj; jj += 1)
        {
            v4f32 _sum0 = (v4f32)__msa_ld_w(pp, 0);
            pp += 4;

            if (pCi)
            {
                v4f32 _beta = __msa_fill_w_f32(beta);
                if (broadcast_type_C == 0)
                {
                    _sum0 = __msa_fadd_w(_sum0, __msa_fmul_w(__msa_fill_w_f32(pCi[0]), _beta));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _sum0 = __msa_fadd_w(_sum0, __msa_fmul_w((v4f32)__msa_ld_w(pCi, 0), _beta));
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        const float* pCj = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + (j + jj) * 4;
                        _sum0 = __msa_fmadd_w(_sum0, _beta, (v4f32)__msa_ld_w(pCj, 0));
                    }
                    if (c_elempack == 1)
                    {
                        const float* pCj = (const float*)C + (i + ii) * c_hstep + (j + jj);
                        float tmp[4];
                        tmp[0] = pCj[0];
                        tmp[1] = pCj[c_hstep];
                        tmp[2] = pCj[c_hstep * 2];
                        tmp[3] = pCj[c_hstep * 3];
                        _sum0 = __msa_fmadd_w(_sum0, _beta, (v4f32)__msa_ld_w(tmp, 0));
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _sum0 = __msa_fmadd_w(_sum0, _beta, __msa_fill_w_f32(pCi[0]));
                    pCi += 1;
                }
            }

            _sum0 = __msa_fmul_w(_sum0, _valpha);

            if (output_transpose)
            {
                float tmp[4];
                __msa_st_w((v4i32)_sum0, tmp, 0);
                int col = j + jj;
                for (int r = 0; r < 4; r++)
                {
                    int row = i + ii + r;
                    size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                    if (output_elemtype == 1)
                        ((float*)top_blob)[offset] = tmp[r];
                    else
                        ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp[r]);
                }
            }
            else
            {
                if (out_elempack == 4)
                {
                    if (output_elemtype == 1)
                    {
                        __msa_st_w((v4i32)_sum0, p0f32, 0);
                        p0f32 += 4;
                    }
                    else
                    {
                        float2bfloat_msa_store(_sum0, p0);
                        p0 += 4;
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp[4];
                    __msa_st_w((v4i32)_sum0, tmp, 0);
                    if (output_elemtype == 1)
                    {
                        p0f32[0] = tmp[0];
                        p0f32[out_hstep] = tmp[1];
                        p0f32[out_hstep * 2] = tmp[2];
                        p0f32[out_hstep * 3] = tmp[3];
                        p0f32++;
                    }
                    else
                    {
                        p0[0] = float32_to_bfloat16(tmp[0]);
                        p0[out_hstep] = float32_to_bfloat16(tmp[1]);
                        p0[out_hstep * 2] = float32_to_bfloat16(tmp[2]);
                        p0[out_hstep * 3] = float32_to_bfloat16(tmp[3]);
                        p0++;
                    }
                }
            }
        }
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* pCi = pC;
        if (pCi)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pCi = (const float*)C + (i + ii);
            if (broadcast_type_C == 4)
                pCi = (const float*)C + j;
        }

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            // topT layout: jj-major, 2 floats per jj (ii0, ii1)
            float sum0 = pp[0];
            float sum1 = pp[1];
            pp += 2;

            if (pCi)
            {
                if (broadcast_type_C == 0)
                {
                    sum0 += pCi[0] * beta;
                    sum1 += pCi[0] * beta;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    sum0 += pCi[0] * beta;
                    sum1 += pCi[1] * beta;
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 1)
                    {
                        sum0 += ((const float*)C + (i + ii) * c_hstep + (j + jj))[0] * beta;
                        sum1 += ((const float*)C + (i + ii + 1) * c_hstep + (j + jj))[0] * beta;
                    }
                    else
                    {
                        // c_elempack == 4, but ii is in scalar range
                        sum0 += ((const float*)C + ((j + jj) / c_elempack) * c_hstep * c_elempack + (i + ii) * c_elempack + (j + jj) % c_elempack)[0] * beta;
                        sum1 += ((const float*)C + ((j + jj) / c_elempack) * c_hstep * c_elempack + (i + ii + 1) * c_elempack + (j + jj) % c_elempack)[0] * beta;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    sum0 += pCi[0] * beta;
                    sum1 += pCi[0] * beta;
                    pCi += 1;
                }
            }

            sum0 *= alpha;
            sum1 *= alpha;

            if (output_transpose)
            {
                if (output_elemtype == 1)
                {
                    int col = j + jj;
                    size_t offset0 = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)(i + ii) * out_elempack + col % out_elempack;
                    size_t offset1 = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)(i + ii + 1) * out_elempack + col % out_elempack;
                    ((float*)top_blob)[offset0] = sum0;
                    ((float*)top_blob)[offset1] = sum1;
                }
                else
                {
                    int col = j + jj;
                    size_t offset0 = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)(i + ii) * out_elempack + col % out_elempack;
                    size_t offset1 = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)(i + ii + 1) * out_elempack + col % out_elempack;
                    ((unsigned short*)top_blob)[offset0] = float32_to_bfloat16(sum0);
                    ((unsigned short*)top_blob)[offset1] = float32_to_bfloat16(sum1);
                }
            }
            else
            {
                if (output_elemtype == 1)
                {
                    ((float*)top_blob)[(i + ii) * out_hstep + (j + jj)] = sum0;
                    ((float*)top_blob)[(i + ii + 1) * out_hstep + (j + jj)] = sum1;
                }
                else
                {
                    ((unsigned short*)top_blob)[(i + ii) * out_hstep + (j + jj)] = float32_to_bfloat16(sum0);
                    ((unsigned short*)top_blob)[(i + ii + 1) * out_hstep + (j + jj)] = float32_to_bfloat16(sum1);
                }
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        const float* pCi = pC;
        if (pCi)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pCi = (const float*)C + (i + ii);
            if (broadcast_type_C == 4)
                pCi = (const float*)C + j;
        }

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            float sum = pp[0];
            pp += 1;

            if (pCi)
            {
                if (broadcast_type_C == 0)
                {
                    sum += pCi[0] * beta;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    sum += pCi[0] * beta;
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 1)
                    {
                        sum += ((const float*)C + (i + ii) * c_hstep + (j + jj))[0] * beta;
                    }
                    else
                    {
                        sum += ((const float*)C + ((j + jj) / c_elempack) * c_hstep * c_elempack + (i + ii) * c_elempack + (j + jj) % c_elempack)[0] * beta;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    sum += pCi[0] * beta;
                    pCi += 1;
                }
            }

            sum *= alpha;

            if (output_transpose)
            {
                int col = j + jj;
                int row = i + ii;
                size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                if (output_elemtype == 1)
                    ((float*)top_blob)[offset] = sum;
                else
                    ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(sum);
            }
            else
            {
                if (output_elemtype == 1)
                    ((float*)top_blob)[(i + ii) * out_hstep + (j + jj)] = sum;
                else
                    ((unsigned short*)top_blob)[(i + ii) * out_hstep + (j + jj)] = float32_to_bfloat16(sum);
            }
        }
    }
}

static void get_optimal_tile_mnk_bf16(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // bf16 uses 2 bytes per element for A and B, but 4 bytes for accumulator
    int tile_size = (int)sqrtf((float)l2_cache_size / (2 * sizeof(unsigned short) + sizeof(float)));

#if __mips_msa
    TILE_M = std::max(4, tile_size / 4 * 4);
    TILE_N = std::max(4, tile_size / 4 * 4);
    TILE_K = std::max(4, tile_size / 4 * 4);
#else
    TILE_M = std::max(2, tile_size / 2 * 2);
    TILE_N = std::max(1, tile_size);
    TILE_K = std::max(2, tile_size / 2 * 2);
#endif

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __mips_msa
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(unsigned short) / TILE_K);
#if __mips_msa
            TILE_M = std::max(4, tile_size / 4 * 4);
            TILE_N = std::max(4, tile_size / 4 * 4);
#else
            TILE_M = std::max(2, tile_size / 2 * 2);
            TILE_N = std::max(1, tile_size);
#endif
        }
    }

    TILE_M *= std::min(nT, get_physical_cpu_count());

    if (M > 0)
    {
        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __mips_msa
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __mips_msa
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif
    }

    if (nT > 1)
    {
#if __mips_msa
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
    }

    if (constant_TILE_M > 0)
    {
#if __mips_msa
        TILE_M = (constant_TILE_M + 3) / 4 * 4;
#else
        TILE_M = (constant_TILE_M + 1) / 2 * 2;
#endif
    }
    if (constant_TILE_N > 0)
    {
#if __mips_msa
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
#else
        TILE_N = constant_TILE_N;
#endif
    }
    if (constant_TILE_K > 0)
    {
#if __mips_msa
        TILE_K = (constant_TILE_K + 3) / 4 * 4;
#else
        TILE_K = (constant_TILE_K + 1) / 2 * 2;
#endif
    }
}

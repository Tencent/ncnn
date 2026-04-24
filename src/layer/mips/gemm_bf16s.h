// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pack_A_tile_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 4) * A_hstep + k * 4;

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
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;
            const unsigned short* p2 = (const unsigned short*)A + (i + ii + 2) * A_hstep + k;
            const unsigned short* p3 = (const unsigned short*)A + (i + ii + 3) * A_hstep + k;
            const unsigned short* p4 = (const unsigned short*)A + (i + ii + 4) * A_hstep + k;
            const unsigned short* p5 = (const unsigned short*)A + (i + ii + 5) * A_hstep + k;
            const unsigned short* p6 = (const unsigned short*)A + (i + ii + 6) * A_hstep + k;
            const unsigned short* p7 = (const unsigned short*)A + (i + ii + 7) * A_hstep + k;

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
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp[2] = p0[8];
                pp[3] = p0[12];
                pp[4] = p0[16];
                pp[5] = p0[20];
                pp[6] = p0[24];
                pp[7] = p0[28];
                pp[8] = p0[1];
                pp[9] = p0[5];
                pp[10] = p0[9];
                pp[11] = p0[13];
                pp[12] = p0[17];
                pp[13] = p0[21];
                pp[14] = p0[25];
                pp[15] = p0[29];
                pp[16] = p0[2];
                pp[17] = p0[6];
                pp[18] = p0[10];
                pp[19] = p0[14];
                pp[20] = p0[18];
                pp[21] = p0[22];
                pp[22] = p0[26];
                pp[23] = p0[30];
                pp[24] = p0[3];
                pp[25] = p0[7];
                pp[26] = p0[11];
                pp[27] = p0[15];
                pp[28] = p0[19];
                pp[29] = p0[23];
                pp[30] = p0[27];
                pp[31] = p0[31];
                pp += 32;
                p0 += A_hstep * 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp[2] = p0[8];
                pp[3] = p0[12];
                pp[4] = p0[16];
                pp[5] = p0[20];
                pp[6] = p0[24];
                pp[7] = p0[28];
                pp += 8;
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
                pp[4] = p0[4];
                pp[5] = p0[5];
                pp[6] = p0[6];
                pp[7] = p0[7];
                pp += 8;
                p0 += A_hstep;
            }
        }
    }
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
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            v4f32 _sum00;
            v4f32 _sum01;
            v4f32 _sum10;
            v4f32 _sum11;
            v4f32 _sum20;
            v4f32 _sum21;
            v4f32 _sum30;
            v4f32 _sum31;
            v4f32 _sum40;
            v4f32 _sum41;
            v4f32 _sum50;
            v4f32 _sum51;
            v4f32 _sum60;
            v4f32 _sum61;
            v4f32 _sum70;
            v4f32 _sum71;

            if (k == 0)
            {
                _sum00 = (v4f32)__msa_fill_w(0);
                _sum01 = (v4f32)__msa_fill_w(0);
                _sum10 = (v4f32)__msa_fill_w(0);
                _sum11 = (v4f32)__msa_fill_w(0);
                _sum20 = (v4f32)__msa_fill_w(0);
                _sum21 = (v4f32)__msa_fill_w(0);
                _sum30 = (v4f32)__msa_fill_w(0);
                _sum31 = (v4f32)__msa_fill_w(0);
                _sum40 = (v4f32)__msa_fill_w(0);
                _sum41 = (v4f32)__msa_fill_w(0);
                _sum50 = (v4f32)__msa_fill_w(0);
                _sum51 = (v4f32)__msa_fill_w(0);
                _sum60 = (v4f32)__msa_fill_w(0);
                _sum61 = (v4f32)__msa_fill_w(0);
                _sum70 = (v4f32)__msa_fill_w(0);
                _sum71 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _sum00 = (v4f32)__msa_ld_w(outptr, 0);
                _sum01 = (v4f32)__msa_ld_w(outptr + 4, 0);
                _sum10 = (v4f32)__msa_ld_w(outptr + 8, 0);
                _sum11 = (v4f32)__msa_ld_w(outptr + 12, 0);
                _sum20 = (v4f32)__msa_ld_w(outptr + 16, 0);
                _sum21 = (v4f32)__msa_ld_w(outptr + 20, 0);
                _sum30 = (v4f32)__msa_ld_w(outptr + 24, 0);
                _sum31 = (v4f32)__msa_ld_w(outptr + 28, 0);
                _sum40 = (v4f32)__msa_ld_w(outptr + 32, 0);
                _sum41 = (v4f32)__msa_ld_w(outptr + 36, 0);
                _sum50 = (v4f32)__msa_ld_w(outptr + 40, 0);
                _sum51 = (v4f32)__msa_ld_w(outptr + 44, 0);
                _sum60 = (v4f32)__msa_ld_w(outptr + 48, 0);
                _sum61 = (v4f32)__msa_ld_w(outptr + 52, 0);
                _sum70 = (v4f32)__msa_ld_w(outptr + 56, 0);
                _sum71 = (v4f32)__msa_ld_w(outptr + 60, 0);
            }

            const unsigned short* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                v4f32 _pA0 = bfloat2float_msa(pA);
                v4f32 _pA1 = bfloat2float_msa(pA + 4);
                _sum00 = __msa_fmadd_w(_sum00, _pA0, __msa_fill_w_f32(bfloat16_to_float32(pB[0])));
                _sum01 = __msa_fmadd_w(_sum01, _pA1, __msa_fill_w_f32(bfloat16_to_float32(pB[0])));
                _sum10 = __msa_fmadd_w(_sum10, _pA0, __msa_fill_w_f32(bfloat16_to_float32(pB[1])));
                _sum11 = __msa_fmadd_w(_sum11, _pA1, __msa_fill_w_f32(bfloat16_to_float32(pB[1])));
                _sum20 = __msa_fmadd_w(_sum20, _pA0, __msa_fill_w_f32(bfloat16_to_float32(pB[2])));
                _sum21 = __msa_fmadd_w(_sum21, _pA1, __msa_fill_w_f32(bfloat16_to_float32(pB[2])));
                _sum30 = __msa_fmadd_w(_sum30, _pA0, __msa_fill_w_f32(bfloat16_to_float32(pB[3])));
                _sum31 = __msa_fmadd_w(_sum31, _pA1, __msa_fill_w_f32(bfloat16_to_float32(pB[3])));
                _sum40 = __msa_fmadd_w(_sum40, _pA0, __msa_fill_w_f32(bfloat16_to_float32(pB[4])));
                _sum41 = __msa_fmadd_w(_sum41, _pA1, __msa_fill_w_f32(bfloat16_to_float32(pB[4])));
                _sum50 = __msa_fmadd_w(_sum50, _pA0, __msa_fill_w_f32(bfloat16_to_float32(pB[5])));
                _sum51 = __msa_fmadd_w(_sum51, _pA1, __msa_fill_w_f32(bfloat16_to_float32(pB[5])));
                _sum60 = __msa_fmadd_w(_sum60, _pA0, __msa_fill_w_f32(bfloat16_to_float32(pB[6])));
                _sum61 = __msa_fmadd_w(_sum61, _pA1, __msa_fill_w_f32(bfloat16_to_float32(pB[6])));
                _sum70 = __msa_fmadd_w(_sum70, _pA0, __msa_fill_w_f32(bfloat16_to_float32(pB[7])));
                _sum71 = __msa_fmadd_w(_sum71, _pA1, __msa_fill_w_f32(bfloat16_to_float32(pB[7])));
                pA += 8;
                pB += 8;
            }

            __msa_st_w((v4i32)_sum00, outptr, 0);
            __msa_st_w((v4i32)_sum01, outptr + 4, 0);
            __msa_st_w((v4i32)_sum10, outptr + 8, 0);
            __msa_st_w((v4i32)_sum11, outptr + 12, 0);
            __msa_st_w((v4i32)_sum20, outptr + 16, 0);
            __msa_st_w((v4i32)_sum21, outptr + 20, 0);
            __msa_st_w((v4i32)_sum30, outptr + 24, 0);
            __msa_st_w((v4i32)_sum31, outptr + 28, 0);
            __msa_st_w((v4i32)_sum40, outptr + 32, 0);
            __msa_st_w((v4i32)_sum41, outptr + 36, 0);
            __msa_st_w((v4i32)_sum50, outptr + 40, 0);
            __msa_st_w((v4i32)_sum51, outptr + 44, 0);
            __msa_st_w((v4i32)_sum60, outptr + 48, 0);
            __msa_st_w((v4i32)_sum61, outptr + 52, 0);
            __msa_st_w((v4i32)_sum70, outptr + 56, 0);
            __msa_st_w((v4i32)_sum71, outptr + 60, 0);

            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            v4f32 _sum00;
            v4f32 _sum01;
            v4f32 _sum10;
            v4f32 _sum11;
            v4f32 _sum20;
            v4f32 _sum21;
            v4f32 _sum30;
            v4f32 _sum31;

            if (k == 0)
            {
                _sum00 = (v4f32)__msa_fill_w(0);
                _sum01 = (v4f32)__msa_fill_w(0);
                _sum10 = (v4f32)__msa_fill_w(0);
                _sum11 = (v4f32)__msa_fill_w(0);
                _sum20 = (v4f32)__msa_fill_w(0);
                _sum21 = (v4f32)__msa_fill_w(0);
                _sum30 = (v4f32)__msa_fill_w(0);
                _sum31 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _sum00 = (v4f32)__msa_ld_w(outptr, 0);
                _sum01 = (v4f32)__msa_ld_w(outptr + 4, 0);
                _sum10 = (v4f32)__msa_ld_w(outptr + 8, 0);
                _sum11 = (v4f32)__msa_ld_w(outptr + 12, 0);
                _sum20 = (v4f32)__msa_ld_w(outptr + 16, 0);
                _sum21 = (v4f32)__msa_ld_w(outptr + 20, 0);
                _sum30 = (v4f32)__msa_ld_w(outptr + 24, 0);
                _sum31 = (v4f32)__msa_ld_w(outptr + 28, 0);
            }

            const unsigned short* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                v4f32 _pA0 = bfloat2float_msa(pA);
                v4f32 _pA1 = bfloat2float_msa(pA + 4);
                _sum00 = __msa_fmadd_w(_sum00, _pA0, __msa_fill_w_f32(bfloat16_to_float32(pB[0])));
                _sum01 = __msa_fmadd_w(_sum01, _pA1, __msa_fill_w_f32(bfloat16_to_float32(pB[0])));
                _sum10 = __msa_fmadd_w(_sum10, _pA0, __msa_fill_w_f32(bfloat16_to_float32(pB[1])));
                _sum11 = __msa_fmadd_w(_sum11, _pA1, __msa_fill_w_f32(bfloat16_to_float32(pB[1])));
                _sum20 = __msa_fmadd_w(_sum20, _pA0, __msa_fill_w_f32(bfloat16_to_float32(pB[2])));
                _sum21 = __msa_fmadd_w(_sum21, _pA1, __msa_fill_w_f32(bfloat16_to_float32(pB[2])));
                _sum30 = __msa_fmadd_w(_sum30, _pA0, __msa_fill_w_f32(bfloat16_to_float32(pB[3])));
                _sum31 = __msa_fmadd_w(_sum31, _pA1, __msa_fill_w_f32(bfloat16_to_float32(pB[3])));
                pA += 8;
                pB += 4;
            }

            __msa_st_w((v4i32)_sum00, outptr, 0);
            __msa_st_w((v4i32)_sum01, outptr + 4, 0);
            __msa_st_w((v4i32)_sum10, outptr + 8, 0);
            __msa_st_w((v4i32)_sum11, outptr + 12, 0);
            __msa_st_w((v4i32)_sum20, outptr + 16, 0);
            __msa_st_w((v4i32)_sum21, outptr + 20, 0);
            __msa_st_w((v4i32)_sum30, outptr + 24, 0);
            __msa_st_w((v4i32)_sum31, outptr + 28, 0);

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            v4f32 _sum00;
            v4f32 _sum01;
            v4f32 _sum10;
            v4f32 _sum11;

            if (k == 0)
            {
                _sum00 = (v4f32)__msa_fill_w(0);
                _sum01 = (v4f32)__msa_fill_w(0);
                _sum10 = (v4f32)__msa_fill_w(0);
                _sum11 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _sum00 = (v4f32)__msa_ld_w(outptr, 0);
                _sum01 = (v4f32)__msa_ld_w(outptr + 4, 0);
                _sum10 = (v4f32)__msa_ld_w(outptr + 8, 0);
                _sum11 = (v4f32)__msa_ld_w(outptr + 12, 0);
            }

            const unsigned short* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                v4f32 _pA0 = bfloat2float_msa(pA);
                v4f32 _pA1 = bfloat2float_msa(pA + 4);
                _sum00 = __msa_fmadd_w(_sum00, _pA0, __msa_fill_w_f32(bfloat16_to_float32(pB[0])));
                _sum01 = __msa_fmadd_w(_sum01, _pA1, __msa_fill_w_f32(bfloat16_to_float32(pB[0])));
                _sum10 = __msa_fmadd_w(_sum10, _pA0, __msa_fill_w_f32(bfloat16_to_float32(pB[1])));
                _sum11 = __msa_fmadd_w(_sum11, _pA1, __msa_fill_w_f32(bfloat16_to_float32(pB[1])));
                pA += 8;
                pB += 2;
            }

            __msa_st_w((v4i32)_sum00, outptr, 0);
            __msa_st_w((v4i32)_sum01, outptr + 4, 0);
            __msa_st_w((v4i32)_sum10, outptr + 8, 0);
            __msa_st_w((v4i32)_sum11, outptr + 12, 0);

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
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

            const unsigned short* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                v4f32 _pA0 = bfloat2float_msa(pA);
                v4f32 _pA1 = bfloat2float_msa(pA + 4);
                v4f32 _pB = __msa_fill_w_f32(bfloat16_to_float32(pB[0]));
                _sum0 = __msa_fmadd_w(_sum0, _pA0, _pB);
                _sum1 = __msa_fmadd_w(_sum1, _pA1, _pB);
                pA += 8;
                pB += 1;
            }

            __msa_st_w((v4i32)_sum0, outptr, 0);
            __msa_st_w((v4i32)_sum1, outptr + 4, 0);

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
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
        for (; jj + 7 < max_jj; jj += 8)
        {
            float sum00;
            float sum01;
            float sum10;
            float sum11;
            float sum20;
            float sum21;
            float sum30;
            float sum31;
            float sum40;
            float sum41;
            float sum50;
            float sum51;
            float sum60;
            float sum61;
            float sum70;
            float sum71;

            if (k == 0)
            {
                sum00 = 0.f;
                sum01 = 0.f;
                sum10 = 0.f;
                sum11 = 0.f;
                sum20 = 0.f;
                sum21 = 0.f;
                sum30 = 0.f;
                sum31 = 0.f;
                sum40 = 0.f;
                sum41 = 0.f;
                sum50 = 0.f;
                sum51 = 0.f;
                sum60 = 0.f;
                sum61 = 0.f;
                sum70 = 0.f;
                sum71 = 0.f;
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
                sum20 = outptr[4];
                sum21 = outptr[5];
                sum30 = outptr[6];
                sum31 = outptr[7];
                sum40 = outptr[8];
                sum41 = outptr[9];
                sum50 = outptr[10];
                sum51 = outptr[11];
                sum60 = outptr[12];
                sum61 = outptr[13];
                sum70 = outptr[14];
                sum71 = outptr[15];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                float a1 = bfloat16_to_float32(pA[1]);
                float b0 = bfloat16_to_float32(pB[0]);
                float b1 = bfloat16_to_float32(pB[1]);
                float b2 = bfloat16_to_float32(pB[2]);
                float b3 = bfloat16_to_float32(pB[3]);
                float b4 = bfloat16_to_float32(pB[4]);
                float b5 = bfloat16_to_float32(pB[5]);
                float b6 = bfloat16_to_float32(pB[6]);
                float b7 = bfloat16_to_float32(pB[7]);
                sum00 += a0 * b0;
                sum01 += a1 * b0;
                sum10 += a0 * b1;
                sum11 += a1 * b1;
                sum20 += a0 * b2;
                sum21 += a1 * b2;
                sum30 += a0 * b3;
                sum31 += a1 * b3;
                sum40 += a0 * b4;
                sum41 += a1 * b4;
                sum50 += a0 * b5;
                sum51 += a1 * b5;
                sum60 += a0 * b6;
                sum61 += a1 * b6;
                sum70 += a0 * b7;
                sum71 += a1 * b7;
                pA += 2;
                pB += 8;
            }

            outptr[0] = sum00;
            outptr[1] = sum01;
            outptr[2] = sum10;
            outptr[3] = sum11;
            outptr[4] = sum20;
            outptr[5] = sum21;
            outptr[6] = sum30;
            outptr[7] = sum31;
            outptr[8] = sum40;
            outptr[9] = sum41;
            outptr[10] = sum50;
            outptr[11] = sum51;
            outptr[12] = sum60;
            outptr[13] = sum61;
            outptr[14] = sum70;
            outptr[15] = sum71;

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
        for (; jj + 7 < max_jj; jj += 8)
        {
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
                sum0 = 0.f;
                sum1 = 0.f;
                sum2 = 0.f;
                sum3 = 0.f;
                sum4 = 0.f;
                sum5 = 0.f;
                sum6 = 0.f;
                sum7 = 0.f;
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

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                sum0 += a0 * bfloat16_to_float32(pB[0]);
                sum1 += a0 * bfloat16_to_float32(pB[1]);
                sum2 += a0 * bfloat16_to_float32(pB[2]);
                sum3 += a0 * bfloat16_to_float32(pB[3]);
                sum4 += a0 * bfloat16_to_float32(pB[4]);
                sum5 += a0 * bfloat16_to_float32(pB[5]);
                sum6 += a0 * bfloat16_to_float32(pB[6]);
                sum7 += a0 * bfloat16_to_float32(pB[7]);
                pA += 1;
                pB += 8;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr[2] = sum2;
            outptr[3] = sum3;
            outptr[4] = sum4;
            outptr[5] = sum5;
            outptr[6] = sum6;
            outptr[7] = sum7;

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
    for (; ii + 7 < max_ii; ii += 8)
    {
        unsigned short* p0;
        float* p0f32;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
            p0f32 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (i + ii) / out_elempack * out_hstep * out_elempack + j * out_elempack;
            p0f32 = (float*)top_blob + (i + ii) / out_elempack * out_hstep * out_elempack + j * out_elempack;
        }

        const float* pC0 = pC;
        if (pC0)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC0 = (const float*)C + (i + ii);
            }
            if (broadcast_type_C == 4)
            {
                pC0 = (const float*)C + j;
            }
        }

        v4f32 _valpha = __msa_fill_w_f32(alpha);

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            v4f32 _sum00 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _sum01 = (v4f32)__msa_ld_w(pp + 4, 0);
            v4f32 _sum10 = (v4f32)__msa_ld_w(pp + 8, 0);
            v4f32 _sum11 = (v4f32)__msa_ld_w(pp + 12, 0);
            v4f32 _sum20 = (v4f32)__msa_ld_w(pp + 16, 0);
            v4f32 _sum21 = (v4f32)__msa_ld_w(pp + 20, 0);
            v4f32 _sum30 = (v4f32)__msa_ld_w(pp + 24, 0);
            v4f32 _sum31 = (v4f32)__msa_ld_w(pp + 28, 0);
            v4f32 _sum40 = (v4f32)__msa_ld_w(pp + 32, 0);
            v4f32 _sum41 = (v4f32)__msa_ld_w(pp + 36, 0);
            v4f32 _sum50 = (v4f32)__msa_ld_w(pp + 40, 0);
            v4f32 _sum51 = (v4f32)__msa_ld_w(pp + 44, 0);
            v4f32 _sum60 = (v4f32)__msa_ld_w(pp + 48, 0);
            v4f32 _sum61 = (v4f32)__msa_ld_w(pp + 52, 0);
            v4f32 _sum70 = (v4f32)__msa_ld_w(pp + 56, 0);
            v4f32 _sum71 = (v4f32)__msa_ld_w(pp + 60, 0);
            pp += 64;

            if (pC0)
            {
                v4f32 _beta = __msa_fill_w_f32(beta);
                if (broadcast_type_C == 0)
                {
                    v4f32 _c = __msa_fmul_w(__msa_fill_w_f32(pC0[0]), _beta);
                    _sum00 = __msa_fadd_w(_sum00, _c);
                    _sum01 = __msa_fadd_w(_sum01, _c);
                    _sum10 = __msa_fadd_w(_sum10, _c);
                    _sum11 = __msa_fadd_w(_sum11, _c);
                    _sum20 = __msa_fadd_w(_sum20, _c);
                    _sum21 = __msa_fadd_w(_sum21, _c);
                    _sum30 = __msa_fadd_w(_sum30, _c);
                    _sum31 = __msa_fadd_w(_sum31, _c);
                    _sum40 = __msa_fadd_w(_sum40, _c);
                    _sum41 = __msa_fadd_w(_sum41, _c);
                    _sum50 = __msa_fadd_w(_sum50, _c);
                    _sum51 = __msa_fadd_w(_sum51, _c);
                    _sum60 = __msa_fadd_w(_sum60, _c);
                    _sum61 = __msa_fadd_w(_sum61, _c);
                    _sum70 = __msa_fadd_w(_sum70, _c);
                    _sum71 = __msa_fadd_w(_sum71, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _c0 = __msa_fmul_w((v4f32)__msa_ld_w(pC0, 0), _beta);
                    v4f32 _c1 = __msa_fmul_w((v4f32)__msa_ld_w(pC0 + 4, 0), _beta);
                    _sum00 = __msa_fadd_w(_sum00, _c0);
                    _sum01 = __msa_fadd_w(_sum01, _c1);
                    _sum10 = __msa_fadd_w(_sum10, _c0);
                    _sum11 = __msa_fadd_w(_sum11, _c1);
                    _sum20 = __msa_fadd_w(_sum20, _c0);
                    _sum21 = __msa_fadd_w(_sum21, _c1);
                    _sum30 = __msa_fadd_w(_sum30, _c0);
                    _sum31 = __msa_fadd_w(_sum31, _c1);
                    _sum40 = __msa_fadd_w(_sum40, _c0);
                    _sum41 = __msa_fadd_w(_sum41, _c1);
                    _sum50 = __msa_fadd_w(_sum50, _c0);
                    _sum51 = __msa_fadd_w(_sum51, _c1);
                    _sum60 = __msa_fadd_w(_sum60, _c0);
                    _sum61 = __msa_fadd_w(_sum61, _c1);
                    _sum70 = __msa_fadd_w(_sum70, _c0);
                    _sum71 = __msa_fadd_w(_sum71, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        const float* pCj0 = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + (j + jj) * 4;
                        const float* pCj1 = pCj0 + c_hstep * 4;
                        _sum00 = __msa_fmadd_w(_sum00, _beta, (v4f32)__msa_ld_w(pCj0, 0));
                        _sum01 = __msa_fmadd_w(_sum01, _beta, (v4f32)__msa_ld_w(pCj1, 0));
                        _sum10 = __msa_fmadd_w(_sum10, _beta, (v4f32)__msa_ld_w(pCj0 + 4, 0));
                        _sum11 = __msa_fmadd_w(_sum11, _beta, (v4f32)__msa_ld_w(pCj1 + 4, 0));
                        _sum20 = __msa_fmadd_w(_sum20, _beta, (v4f32)__msa_ld_w(pCj0 + 8, 0));
                        _sum21 = __msa_fmadd_w(_sum21, _beta, (v4f32)__msa_ld_w(pCj1 + 8, 0));
                        _sum30 = __msa_fmadd_w(_sum30, _beta, (v4f32)__msa_ld_w(pCj0 + 12, 0));
                        _sum31 = __msa_fmadd_w(_sum31, _beta, (v4f32)__msa_ld_w(pCj1 + 12, 0));
                        _sum40 = __msa_fmadd_w(_sum40, _beta, (v4f32)__msa_ld_w(pCj0 + 16, 0));
                        _sum41 = __msa_fmadd_w(_sum41, _beta, (v4f32)__msa_ld_w(pCj1 + 16, 0));
                        _sum50 = __msa_fmadd_w(_sum50, _beta, (v4f32)__msa_ld_w(pCj0 + 20, 0));
                        _sum51 = __msa_fmadd_w(_sum51, _beta, (v4f32)__msa_ld_w(pCj1 + 20, 0));
                        _sum60 = __msa_fmadd_w(_sum60, _beta, (v4f32)__msa_ld_w(pCj0 + 24, 0));
                        _sum61 = __msa_fmadd_w(_sum61, _beta, (v4f32)__msa_ld_w(pCj1 + 24, 0));
                        _sum70 = __msa_fmadd_w(_sum70, _beta, (v4f32)__msa_ld_w(pCj0 + 28, 0));
                        _sum71 = __msa_fmadd_w(_sum71, _beta, (v4f32)__msa_ld_w(pCj1 + 28, 0));
                    }
                    if (c_elempack == 1)
                    {
                        const float* pCj = (const float*)C + (i + ii) * c_hstep + (j + jj);
                        v4f32 _c0 = (v4f32)__msa_ld_w(pCj, 0);
                        v4f32 _c1 = (v4f32)__msa_ld_w(pCj + c_hstep, 0);
                        v4f32 _c2 = (v4f32)__msa_ld_w(pCj + c_hstep * 2, 0);
                        v4f32 _c3 = (v4f32)__msa_ld_w(pCj + c_hstep * 3, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _sum00 = __msa_fmadd_w(_sum00, _beta, _c0);
                        _sum10 = __msa_fmadd_w(_sum10, _beta, _c1);
                        _sum20 = __msa_fmadd_w(_sum20, _beta, _c2);
                        _sum30 = __msa_fmadd_w(_sum30, _beta, _c3);

                        _c0 = (v4f32)__msa_ld_w(pCj + c_hstep * 4, 0);
                        _c1 = (v4f32)__msa_ld_w(pCj + c_hstep * 5, 0);
                        _c2 = (v4f32)__msa_ld_w(pCj + c_hstep * 6, 0);
                        _c3 = (v4f32)__msa_ld_w(pCj + c_hstep * 7, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _sum01 = __msa_fmadd_w(_sum01, _beta, _c0);
                        _sum11 = __msa_fmadd_w(_sum11, _beta, _c1);
                        _sum21 = __msa_fmadd_w(_sum21, _beta, _c2);
                        _sum31 = __msa_fmadd_w(_sum31, _beta, _c3);

                        pCj += 4;

                        _c0 = (v4f32)__msa_ld_w(pCj, 0);
                        _c1 = (v4f32)__msa_ld_w(pCj + c_hstep, 0);
                        _c2 = (v4f32)__msa_ld_w(pCj + c_hstep * 2, 0);
                        _c3 = (v4f32)__msa_ld_w(pCj + c_hstep * 3, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _sum40 = __msa_fmadd_w(_sum40, _beta, _c0);
                        _sum50 = __msa_fmadd_w(_sum50, _beta, _c1);
                        _sum60 = __msa_fmadd_w(_sum60, _beta, _c2);
                        _sum70 = __msa_fmadd_w(_sum70, _beta, _c3);

                        _c0 = (v4f32)__msa_ld_w(pCj + c_hstep * 4, 0);
                        _c1 = (v4f32)__msa_ld_w(pCj + c_hstep * 5, 0);
                        _c2 = (v4f32)__msa_ld_w(pCj + c_hstep * 6, 0);
                        _c3 = (v4f32)__msa_ld_w(pCj + c_hstep * 7, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _sum41 = __msa_fmadd_w(_sum41, _beta, _c0);
                        _sum51 = __msa_fmadd_w(_sum51, _beta, _c1);
                        _sum61 = __msa_fmadd_w(_sum61, _beta, _c2);
                        _sum71 = __msa_fmadd_w(_sum71, _beta, _c3);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _c0 = __msa_fill_w_f32(pC0[0]);
                    v4f32 _c1 = __msa_fill_w_f32(pC0[1]);
                    v4f32 _c2 = __msa_fill_w_f32(pC0[2]);
                    v4f32 _c3 = __msa_fill_w_f32(pC0[3]);
                    v4f32 _c4 = __msa_fill_w_f32(pC0[4]);
                    v4f32 _c5 = __msa_fill_w_f32(pC0[5]);
                    v4f32 _c6 = __msa_fill_w_f32(pC0[6]);
                    v4f32 _c7 = __msa_fill_w_f32(pC0[7]);
                    _sum00 = __msa_fmadd_w(_sum00, _beta, _c0);
                    _sum01 = __msa_fmadd_w(_sum01, _beta, _c0);
                    _sum10 = __msa_fmadd_w(_sum10, _beta, _c1);
                    _sum11 = __msa_fmadd_w(_sum11, _beta, _c1);
                    _sum20 = __msa_fmadd_w(_sum20, _beta, _c2);
                    _sum21 = __msa_fmadd_w(_sum21, _beta, _c2);
                    _sum30 = __msa_fmadd_w(_sum30, _beta, _c3);
                    _sum31 = __msa_fmadd_w(_sum31, _beta, _c3);
                    _sum40 = __msa_fmadd_w(_sum40, _beta, _c4);
                    _sum41 = __msa_fmadd_w(_sum41, _beta, _c4);
                    _sum50 = __msa_fmadd_w(_sum50, _beta, _c5);
                    _sum51 = __msa_fmadd_w(_sum51, _beta, _c5);
                    _sum60 = __msa_fmadd_w(_sum60, _beta, _c6);
                    _sum61 = __msa_fmadd_w(_sum61, _beta, _c6);
                    _sum70 = __msa_fmadd_w(_sum70, _beta, _c7);
                    _sum71 = __msa_fmadd_w(_sum71, _beta, _c7);
                    pC0 += 8;
                }
            }

            _sum00 = __msa_fmul_w(_sum00, _valpha);
            _sum01 = __msa_fmul_w(_sum01, _valpha);
            _sum10 = __msa_fmul_w(_sum10, _valpha);
            _sum11 = __msa_fmul_w(_sum11, _valpha);
            _sum20 = __msa_fmul_w(_sum20, _valpha);
            _sum21 = __msa_fmul_w(_sum21, _valpha);
            _sum30 = __msa_fmul_w(_sum30, _valpha);
            _sum31 = __msa_fmul_w(_sum31, _valpha);
            _sum40 = __msa_fmul_w(_sum40, _valpha);
            _sum41 = __msa_fmul_w(_sum41, _valpha);
            _sum50 = __msa_fmul_w(_sum50, _valpha);
            _sum51 = __msa_fmul_w(_sum51, _valpha);
            _sum60 = __msa_fmul_w(_sum60, _valpha);
            _sum61 = __msa_fmul_w(_sum61, _valpha);
            _sum70 = __msa_fmul_w(_sum70, _valpha);
            _sum71 = __msa_fmul_w(_sum71, _valpha);

            if (output_transpose)
            {
                {
                    float tmp0[4];
                    float tmp1[4];
                    __msa_st_w((v4i32)_sum00, tmp0, 0);
                    __msa_st_w((v4i32)_sum01, tmp1, 0);

                    int col = j + jj;
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp0[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp0[r]);
                    }
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + 4 + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp1[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp1[r]);
                    }
                }
                {
                    float tmp0[4];
                    float tmp1[4];
                    __msa_st_w((v4i32)_sum10, tmp0, 0);
                    __msa_st_w((v4i32)_sum11, tmp1, 0);

                    int col = j + jj + 1;
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp0[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp0[r]);
                    }
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + 4 + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp1[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp1[r]);
                    }
                }
                {
                    float tmp0[4];
                    float tmp1[4];
                    __msa_st_w((v4i32)_sum20, tmp0, 0);
                    __msa_st_w((v4i32)_sum21, tmp1, 0);

                    int col = j + jj + 2;
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp0[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp0[r]);
                    }
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + 4 + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp1[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp1[r]);
                    }
                }
                {
                    float tmp0[4];
                    float tmp1[4];
                    __msa_st_w((v4i32)_sum30, tmp0, 0);
                    __msa_st_w((v4i32)_sum31, tmp1, 0);

                    int col = j + jj + 3;
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp0[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp0[r]);
                    }
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + 4 + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp1[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp1[r]);
                    }
                }
                {
                    float tmp0[4];
                    float tmp1[4];
                    __msa_st_w((v4i32)_sum40, tmp0, 0);
                    __msa_st_w((v4i32)_sum41, tmp1, 0);

                    int col = j + jj + 4;
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp0[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp0[r]);
                    }
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + 4 + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp1[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp1[r]);
                    }
                }
                {
                    float tmp0[4];
                    float tmp1[4];
                    __msa_st_w((v4i32)_sum50, tmp0, 0);
                    __msa_st_w((v4i32)_sum51, tmp1, 0);

                    int col = j + jj + 5;
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp0[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp0[r]);
                    }
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + 4 + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp1[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp1[r]);
                    }
                }
                {
                    float tmp0[4];
                    float tmp1[4];
                    __msa_st_w((v4i32)_sum60, tmp0, 0);
                    __msa_st_w((v4i32)_sum61, tmp1, 0);

                    int col = j + jj + 6;
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp0[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp0[r]);
                    }
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + 4 + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp1[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp1[r]);
                    }
                }
                {
                    float tmp0[4];
                    float tmp1[4];
                    __msa_st_w((v4i32)_sum70, tmp0, 0);
                    __msa_st_w((v4i32)_sum71, tmp1, 0);

                    int col = j + jj + 7;
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp0[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp0[r]);
                    }
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + 4 + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp1[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp1[r]);
                    }
                }
            }
            else
            {
                if (out_elempack == 4)
                {
                    if (output_elemtype == 1)
                    {
                        float* p1f32 = p0f32 + out_hstep * 4;
                        __msa_st_w((v4i32)_sum00, p0f32, 0);
                        __msa_st_w((v4i32)_sum01, p1f32, 0);
                        __msa_st_w((v4i32)_sum10, p0f32 + 4, 0);
                        __msa_st_w((v4i32)_sum11, p1f32 + 4, 0);
                        __msa_st_w((v4i32)_sum20, p0f32 + 8, 0);
                        __msa_st_w((v4i32)_sum21, p1f32 + 8, 0);
                        __msa_st_w((v4i32)_sum30, p0f32 + 12, 0);
                        __msa_st_w((v4i32)_sum31, p1f32 + 12, 0);
                        __msa_st_w((v4i32)_sum40, p0f32 + 16, 0);
                        __msa_st_w((v4i32)_sum41, p1f32 + 16, 0);
                        __msa_st_w((v4i32)_sum50, p0f32 + 20, 0);
                        __msa_st_w((v4i32)_sum51, p1f32 + 20, 0);
                        __msa_st_w((v4i32)_sum60, p0f32 + 24, 0);
                        __msa_st_w((v4i32)_sum61, p1f32 + 24, 0);
                        __msa_st_w((v4i32)_sum70, p0f32 + 28, 0);
                        __msa_st_w((v4i32)_sum71, p1f32 + 28, 0);
                        p0f32 += 32;
                    }
                    else
                    {
                        unsigned short* p1 = p0 + out_hstep * 4;
                        float2bfloat_msa_store(_sum00, p0);
                        float2bfloat_msa_store(_sum01, p1);
                        float2bfloat_msa_store(_sum10, p0 + 4);
                        float2bfloat_msa_store(_sum11, p1 + 4);
                        float2bfloat_msa_store(_sum20, p0 + 8);
                        float2bfloat_msa_store(_sum21, p1 + 8);
                        float2bfloat_msa_store(_sum30, p0 + 12);
                        float2bfloat_msa_store(_sum31, p1 + 12);
                        float2bfloat_msa_store(_sum40, p0 + 16);
                        float2bfloat_msa_store(_sum41, p1 + 16);
                        float2bfloat_msa_store(_sum50, p0 + 20);
                        float2bfloat_msa_store(_sum51, p1 + 20);
                        float2bfloat_msa_store(_sum60, p0 + 24);
                        float2bfloat_msa_store(_sum61, p1 + 24);
                        float2bfloat_msa_store(_sum70, p0 + 28);
                        float2bfloat_msa_store(_sum71, p1 + 28);
                        p0 += 32;
                    }
                }
                if (out_elempack == 1)
                {
                    v4f32 _r0 = _sum00;
                    v4f32 _r1 = _sum10;
                    v4f32 _r2 = _sum20;
                    v4f32 _r3 = _sum30;
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    v4f32 _r4 = _sum40;
                    v4f32 _r5 = _sum50;
                    v4f32 _r6 = _sum60;
                    v4f32 _r7 = _sum70;
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    v4f32 _r8 = _sum01;
                    v4f32 _r9 = _sum11;
                    v4f32 _ra = _sum21;
                    v4f32 _rb = _sum31;
                    transpose4x4_ps(_r8, _r9, _ra, _rb);
                    v4f32 _rc = _sum41;
                    v4f32 _rd = _sum51;
                    v4f32 _re = _sum61;
                    v4f32 _rf = _sum71;
                    transpose4x4_ps(_rc, _rd, _re, _rf);

                    if (output_elemtype == 1)
                    {
                        __msa_st_w((v4i32)_r0, p0f32, 0);
                        __msa_st_w((v4i32)_r4, p0f32 + 4, 0);
                        __msa_st_w((v4i32)_r1, p0f32 + out_hstep, 0);
                        __msa_st_w((v4i32)_r5, p0f32 + out_hstep + 4, 0);
                        __msa_st_w((v4i32)_r2, p0f32 + out_hstep * 2, 0);
                        __msa_st_w((v4i32)_r6, p0f32 + out_hstep * 2 + 4, 0);
                        __msa_st_w((v4i32)_r3, p0f32 + out_hstep * 3, 0);
                        __msa_st_w((v4i32)_r7, p0f32 + out_hstep * 3 + 4, 0);
                        __msa_st_w((v4i32)_r8, p0f32 + out_hstep * 4, 0);
                        __msa_st_w((v4i32)_rc, p0f32 + out_hstep * 4 + 4, 0);
                        __msa_st_w((v4i32)_r9, p0f32 + out_hstep * 5, 0);
                        __msa_st_w((v4i32)_rd, p0f32 + out_hstep * 5 + 4, 0);
                        __msa_st_w((v4i32)_ra, p0f32 + out_hstep * 6, 0);
                        __msa_st_w((v4i32)_re, p0f32 + out_hstep * 6 + 4, 0);
                        __msa_st_w((v4i32)_rb, p0f32 + out_hstep * 7, 0);
                        __msa_st_w((v4i32)_rf, p0f32 + out_hstep * 7 + 4, 0);
                        p0f32 += 8;
                    }
                    else
                    {
                        float2bfloat_msa_store(_r0, p0);
                        float2bfloat_msa_store(_r4, p0 + 4);
                        float2bfloat_msa_store(_r1, p0 + out_hstep);
                        float2bfloat_msa_store(_r5, p0 + out_hstep + 4);
                        float2bfloat_msa_store(_r2, p0 + out_hstep * 2);
                        float2bfloat_msa_store(_r6, p0 + out_hstep * 2 + 4);
                        float2bfloat_msa_store(_r3, p0 + out_hstep * 3);
                        float2bfloat_msa_store(_r7, p0 + out_hstep * 3 + 4);
                        float2bfloat_msa_store(_r8, p0 + out_hstep * 4);
                        float2bfloat_msa_store(_rc, p0 + out_hstep * 4 + 4);
                        float2bfloat_msa_store(_r9, p0 + out_hstep * 5);
                        float2bfloat_msa_store(_rd, p0 + out_hstep * 5 + 4);
                        float2bfloat_msa_store(_ra, p0 + out_hstep * 6);
                        float2bfloat_msa_store(_re, p0 + out_hstep * 6 + 4);
                        float2bfloat_msa_store(_rb, p0 + out_hstep * 7);
                        float2bfloat_msa_store(_rf, p0 + out_hstep * 7 + 4);
                        p0 += 8;
                    }
                }
            }
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            v4f32 _sum00 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _sum01 = (v4f32)__msa_ld_w(pp + 4, 0);
            v4f32 _sum10 = (v4f32)__msa_ld_w(pp + 8, 0);
            v4f32 _sum11 = (v4f32)__msa_ld_w(pp + 12, 0);
            v4f32 _sum20 = (v4f32)__msa_ld_w(pp + 16, 0);
            v4f32 _sum21 = (v4f32)__msa_ld_w(pp + 20, 0);
            v4f32 _sum30 = (v4f32)__msa_ld_w(pp + 24, 0);
            v4f32 _sum31 = (v4f32)__msa_ld_w(pp + 28, 0);
            pp += 32;

            if (pC0)
            {
                v4f32 _beta = __msa_fill_w_f32(beta);
                if (broadcast_type_C == 0)
                {
                    v4f32 _c = __msa_fmul_w(__msa_fill_w_f32(pC0[0]), _beta);
                    _sum00 = __msa_fadd_w(_sum00, _c);
                    _sum01 = __msa_fadd_w(_sum01, _c);
                    _sum10 = __msa_fadd_w(_sum10, _c);
                    _sum11 = __msa_fadd_w(_sum11, _c);
                    _sum20 = __msa_fadd_w(_sum20, _c);
                    _sum21 = __msa_fadd_w(_sum21, _c);
                    _sum30 = __msa_fadd_w(_sum30, _c);
                    _sum31 = __msa_fadd_w(_sum31, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _c0 = __msa_fmul_w((v4f32)__msa_ld_w(pC0, 0), _beta);
                    v4f32 _c1 = __msa_fmul_w((v4f32)__msa_ld_w(pC0 + 4, 0), _beta);
                    _sum00 = __msa_fadd_w(_sum00, _c0);
                    _sum01 = __msa_fadd_w(_sum01, _c1);
                    _sum10 = __msa_fadd_w(_sum10, _c0);
                    _sum11 = __msa_fadd_w(_sum11, _c1);
                    _sum20 = __msa_fadd_w(_sum20, _c0);
                    _sum21 = __msa_fadd_w(_sum21, _c1);
                    _sum30 = __msa_fadd_w(_sum30, _c0);
                    _sum31 = __msa_fadd_w(_sum31, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        const float* pCj0 = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + (j + jj) * 4;
                        const float* pCj1 = pCj0 + c_hstep * 4;
                        _sum00 = __msa_fmadd_w(_sum00, _beta, (v4f32)__msa_ld_w(pCj0, 0));
                        _sum01 = __msa_fmadd_w(_sum01, _beta, (v4f32)__msa_ld_w(pCj1, 0));
                        _sum10 = __msa_fmadd_w(_sum10, _beta, (v4f32)__msa_ld_w(pCj0 + 4, 0));
                        _sum11 = __msa_fmadd_w(_sum11, _beta, (v4f32)__msa_ld_w(pCj1 + 4, 0));
                        _sum20 = __msa_fmadd_w(_sum20, _beta, (v4f32)__msa_ld_w(pCj0 + 8, 0));
                        _sum21 = __msa_fmadd_w(_sum21, _beta, (v4f32)__msa_ld_w(pCj1 + 8, 0));
                        _sum30 = __msa_fmadd_w(_sum30, _beta, (v4f32)__msa_ld_w(pCj0 + 12, 0));
                        _sum31 = __msa_fmadd_w(_sum31, _beta, (v4f32)__msa_ld_w(pCj1 + 12, 0));
                    }
                    if (c_elempack == 1)
                    {
                        const float* pCj = (const float*)C + (i + ii) * c_hstep + (j + jj);
                        float tmp0[4];
                        float tmp1[4];
                        tmp0[0] = pCj[0];
                        tmp0[1] = pCj[c_hstep];
                        tmp0[2] = pCj[c_hstep * 2];
                        tmp0[3] = pCj[c_hstep * 3];
                        tmp1[0] = pCj[c_hstep * 4];
                        tmp1[1] = pCj[c_hstep * 5];
                        tmp1[2] = pCj[c_hstep * 6];
                        tmp1[3] = pCj[c_hstep * 7];
                        _sum00 = __msa_fmadd_w(_sum00, _beta, (v4f32)__msa_ld_w(tmp0, 0));
                        _sum01 = __msa_fmadd_w(_sum01, _beta, (v4f32)__msa_ld_w(tmp1, 0));
                        pCj += 1;
                        tmp0[0] = pCj[0];
                        tmp0[1] = pCj[c_hstep];
                        tmp0[2] = pCj[c_hstep * 2];
                        tmp0[3] = pCj[c_hstep * 3];
                        tmp1[0] = pCj[c_hstep * 4];
                        tmp1[1] = pCj[c_hstep * 5];
                        tmp1[2] = pCj[c_hstep * 6];
                        tmp1[3] = pCj[c_hstep * 7];
                        _sum10 = __msa_fmadd_w(_sum10, _beta, (v4f32)__msa_ld_w(tmp0, 0));
                        _sum11 = __msa_fmadd_w(_sum11, _beta, (v4f32)__msa_ld_w(tmp1, 0));
                        pCj += 1;
                        tmp0[0] = pCj[0];
                        tmp0[1] = pCj[c_hstep];
                        tmp0[2] = pCj[c_hstep * 2];
                        tmp0[3] = pCj[c_hstep * 3];
                        tmp1[0] = pCj[c_hstep * 4];
                        tmp1[1] = pCj[c_hstep * 5];
                        tmp1[2] = pCj[c_hstep * 6];
                        tmp1[3] = pCj[c_hstep * 7];
                        _sum20 = __msa_fmadd_w(_sum20, _beta, (v4f32)__msa_ld_w(tmp0, 0));
                        _sum21 = __msa_fmadd_w(_sum21, _beta, (v4f32)__msa_ld_w(tmp1, 0));
                        pCj += 1;
                        tmp0[0] = pCj[0];
                        tmp0[1] = pCj[c_hstep];
                        tmp0[2] = pCj[c_hstep * 2];
                        tmp0[3] = pCj[c_hstep * 3];
                        tmp1[0] = pCj[c_hstep * 4];
                        tmp1[1] = pCj[c_hstep * 5];
                        tmp1[2] = pCj[c_hstep * 6];
                        tmp1[3] = pCj[c_hstep * 7];
                        _sum30 = __msa_fmadd_w(_sum30, _beta, (v4f32)__msa_ld_w(tmp0, 0));
                        _sum31 = __msa_fmadd_w(_sum31, _beta, (v4f32)__msa_ld_w(tmp1, 0));
                    }
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _c0 = __msa_fill_w_f32(pC0[0]);
                    v4f32 _c1 = __msa_fill_w_f32(pC0[1]);
                    v4f32 _c2 = __msa_fill_w_f32(pC0[2]);
                    v4f32 _c3 = __msa_fill_w_f32(pC0[3]);
                    _sum00 = __msa_fmadd_w(_sum00, _beta, _c0);
                    _sum01 = __msa_fmadd_w(_sum01, _beta, _c0);
                    _sum10 = __msa_fmadd_w(_sum10, _beta, _c1);
                    _sum11 = __msa_fmadd_w(_sum11, _beta, _c1);
                    _sum20 = __msa_fmadd_w(_sum20, _beta, _c2);
                    _sum21 = __msa_fmadd_w(_sum21, _beta, _c2);
                    _sum30 = __msa_fmadd_w(_sum30, _beta, _c3);
                    _sum31 = __msa_fmadd_w(_sum31, _beta, _c3);
                    pC0 += 4;
                }
            }

            _sum00 = __msa_fmul_w(_sum00, _valpha);
            _sum01 = __msa_fmul_w(_sum01, _valpha);
            _sum10 = __msa_fmul_w(_sum10, _valpha);
            _sum11 = __msa_fmul_w(_sum11, _valpha);
            _sum20 = __msa_fmul_w(_sum20, _valpha);
            _sum21 = __msa_fmul_w(_sum21, _valpha);
            _sum30 = __msa_fmul_w(_sum30, _valpha);
            _sum31 = __msa_fmul_w(_sum31, _valpha);

            if (output_transpose)
            {
                {
                    float tmp0[4];
                    float tmp1[4];
                    __msa_st_w((v4i32)_sum00, tmp0, 0);
                    __msa_st_w((v4i32)_sum01, tmp1, 0);

                    int col = j + jj;
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp0[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp0[r]);
                    }
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + 4 + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp1[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp1[r]);
                    }
                }
                {
                    float tmp0[4];
                    float tmp1[4];
                    __msa_st_w((v4i32)_sum10, tmp0, 0);
                    __msa_st_w((v4i32)_sum11, tmp1, 0);

                    int col = j + jj + 1;
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp0[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp0[r]);
                    }
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + 4 + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp1[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp1[r]);
                    }
                }
                {
                    float tmp0[4];
                    float tmp1[4];
                    __msa_st_w((v4i32)_sum20, tmp0, 0);
                    __msa_st_w((v4i32)_sum21, tmp1, 0);

                    int col = j + jj + 2;
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp0[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp0[r]);
                    }
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + 4 + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp1[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp1[r]);
                    }
                }
                {
                    float tmp0[4];
                    float tmp1[4];
                    __msa_st_w((v4i32)_sum30, tmp0, 0);
                    __msa_st_w((v4i32)_sum31, tmp1, 0);

                    int col = j + jj + 3;
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp0[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp0[r]);
                    }
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + 4 + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp1[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp1[r]);
                    }
                }
            }
            else
            {
                if (out_elempack == 4)
                {
                    if (output_elemtype == 1)
                    {
                        float* p1f32 = p0f32 + out_hstep * 4;
                        __msa_st_w((v4i32)_sum00, p0f32, 0);
                        __msa_st_w((v4i32)_sum01, p1f32, 0);
                        __msa_st_w((v4i32)_sum10, p0f32 + 4, 0);
                        __msa_st_w((v4i32)_sum11, p1f32 + 4, 0);
                        __msa_st_w((v4i32)_sum20, p0f32 + 8, 0);
                        __msa_st_w((v4i32)_sum21, p1f32 + 8, 0);
                        __msa_st_w((v4i32)_sum30, p0f32 + 12, 0);
                        __msa_st_w((v4i32)_sum31, p1f32 + 12, 0);
                        p0f32 += 16;
                    }
                    else
                    {
                        unsigned short* p1 = p0 + out_hstep * 4;
                        float2bfloat_msa_store(_sum00, p0);
                        float2bfloat_msa_store(_sum01, p1);
                        float2bfloat_msa_store(_sum10, p0 + 4);
                        float2bfloat_msa_store(_sum11, p1 + 4);
                        float2bfloat_msa_store(_sum20, p0 + 8);
                        float2bfloat_msa_store(_sum21, p1 + 8);
                        float2bfloat_msa_store(_sum30, p0 + 12);
                        float2bfloat_msa_store(_sum31, p1 + 12);
                        p0 += 16;
                    }
                }
                if (out_elempack == 1)
                {
                    v4f32 _r0 = _sum00;
                    v4f32 _r1 = _sum10;
                    v4f32 _r2 = _sum20;
                    v4f32 _r3 = _sum30;
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    v4f32 _r4 = _sum01;
                    v4f32 _r5 = _sum11;
                    v4f32 _r6 = _sum21;
                    v4f32 _r7 = _sum31;
                    transpose4x4_ps(_r4, _r5, _r6, _r7);

                    if (output_elemtype == 1)
                    {
                        __msa_st_w((v4i32)_r0, p0f32, 0);
                        __msa_st_w((v4i32)_r1, p0f32 + out_hstep, 0);
                        __msa_st_w((v4i32)_r2, p0f32 + out_hstep * 2, 0);
                        __msa_st_w((v4i32)_r3, p0f32 + out_hstep * 3, 0);
                        __msa_st_w((v4i32)_r4, p0f32 + out_hstep * 4, 0);
                        __msa_st_w((v4i32)_r5, p0f32 + out_hstep * 5, 0);
                        __msa_st_w((v4i32)_r6, p0f32 + out_hstep * 6, 0);
                        __msa_st_w((v4i32)_r7, p0f32 + out_hstep * 7, 0);
                        p0f32 += 4;
                    }
                    else
                    {
                        float2bfloat_msa_store(_r0, p0);
                        float2bfloat_msa_store(_r1, p0 + out_hstep);
                        float2bfloat_msa_store(_r2, p0 + out_hstep * 2);
                        float2bfloat_msa_store(_r3, p0 + out_hstep * 3);
                        float2bfloat_msa_store(_r4, p0 + out_hstep * 4);
                        float2bfloat_msa_store(_r5, p0 + out_hstep * 5);
                        float2bfloat_msa_store(_r6, p0 + out_hstep * 6);
                        float2bfloat_msa_store(_r7, p0 + out_hstep * 7);
                        p0 += 4;
                    }
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            v4f32 _sum00 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _sum01 = (v4f32)__msa_ld_w(pp + 4, 0);
            v4f32 _sum10 = (v4f32)__msa_ld_w(pp + 8, 0);
            v4f32 _sum11 = (v4f32)__msa_ld_w(pp + 12, 0);
            pp += 16;

            if (pC0)
            {
                v4f32 _beta = __msa_fill_w_f32(beta);
                if (broadcast_type_C == 0)
                {
                    v4f32 _c = __msa_fmul_w(__msa_fill_w_f32(pC0[0]), _beta);
                    _sum00 = __msa_fadd_w(_sum00, _c);
                    _sum01 = __msa_fadd_w(_sum01, _c);
                    _sum10 = __msa_fadd_w(_sum10, _c);
                    _sum11 = __msa_fadd_w(_sum11, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _c0 = __msa_fmul_w((v4f32)__msa_ld_w(pC0, 0), _beta);
                    v4f32 _c1 = __msa_fmul_w((v4f32)__msa_ld_w(pC0 + 4, 0), _beta);
                    _sum00 = __msa_fadd_w(_sum00, _c0);
                    _sum01 = __msa_fadd_w(_sum01, _c1);
                    _sum10 = __msa_fadd_w(_sum10, _c0);
                    _sum11 = __msa_fadd_w(_sum11, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        const float* pCj0 = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + (j + jj) * 4;
                        const float* pCj1 = pCj0 + c_hstep * 4;
                        _sum00 = __msa_fmadd_w(_sum00, _beta, (v4f32)__msa_ld_w(pCj0, 0));
                        _sum01 = __msa_fmadd_w(_sum01, _beta, (v4f32)__msa_ld_w(pCj1, 0));
                        _sum10 = __msa_fmadd_w(_sum10, _beta, (v4f32)__msa_ld_w(pCj0 + 4, 0));
                        _sum11 = __msa_fmadd_w(_sum11, _beta, (v4f32)__msa_ld_w(pCj1 + 4, 0));
                    }
                    if (c_elempack == 1)
                    {
                        const float* pCj = (const float*)C + (i + ii) * c_hstep + (j + jj);
                        float tmp0[4], tmp1[4], tmp2[4], tmp3[4];
                        tmp0[0] = pCj[0];
                        tmp0[1] = pCj[c_hstep];
                        tmp0[2] = pCj[c_hstep * 2];
                        tmp0[3] = pCj[c_hstep * 3];
                        tmp1[0] = pCj[c_hstep * 4];
                        tmp1[1] = pCj[c_hstep * 5];
                        tmp1[2] = pCj[c_hstep * 6];
                        tmp1[3] = pCj[c_hstep * 7];
                        tmp2[0] = pCj[1];
                        tmp2[1] = pCj[c_hstep + 1];
                        tmp2[2] = pCj[c_hstep * 2 + 1];
                        tmp2[3] = pCj[c_hstep * 3 + 1];
                        tmp3[0] = pCj[c_hstep * 4 + 1];
                        tmp3[1] = pCj[c_hstep * 5 + 1];
                        tmp3[2] = pCj[c_hstep * 6 + 1];
                        tmp3[3] = pCj[c_hstep * 7 + 1];
                        _sum00 = __msa_fmadd_w(_sum00, _beta, (v4f32)__msa_ld_w(tmp0, 0));
                        _sum01 = __msa_fmadd_w(_sum01, _beta, (v4f32)__msa_ld_w(tmp1, 0));
                        _sum10 = __msa_fmadd_w(_sum10, _beta, (v4f32)__msa_ld_w(tmp2, 0));
                        _sum11 = __msa_fmadd_w(_sum11, _beta, (v4f32)__msa_ld_w(tmp3, 0));
                    }
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _c0 = __msa_fill_w_f32(pC0[0]);
                    v4f32 _c1 = __msa_fill_w_f32(pC0[1]);
                    _sum00 = __msa_fmadd_w(_sum00, _beta, _c0);
                    _sum01 = __msa_fmadd_w(_sum01, _beta, _c0);
                    _sum10 = __msa_fmadd_w(_sum10, _beta, _c1);
                    _sum11 = __msa_fmadd_w(_sum11, _beta, _c1);
                    pC0 += 2;
                }
            }

            _sum00 = __msa_fmul_w(_sum00, _valpha);
            _sum01 = __msa_fmul_w(_sum01, _valpha);
            _sum10 = __msa_fmul_w(_sum10, _valpha);
            _sum11 = __msa_fmul_w(_sum11, _valpha);

            if (output_transpose)
            {
                {
                    float tmp0[4];
                    float tmp1[4];
                    __msa_st_w((v4i32)_sum00, tmp0, 0);
                    __msa_st_w((v4i32)_sum01, tmp1, 0);
                    int col = j + jj;
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp0[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp0[r]);
                    }
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + 4 + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp1[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp1[r]);
                    }
                }
                {
                    float tmp0[4];
                    float tmp1[4];
                    __msa_st_w((v4i32)_sum10, tmp0, 0);
                    __msa_st_w((v4i32)_sum11, tmp1, 0);
                    int col = j + jj + 1;
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp0[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp0[r]);
                    }
                    for (int r = 0; r < 4; r++)
                    {
                        int row = i + ii + 4 + r;
                        size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                        if (output_elemtype == 1)
                            ((float*)top_blob)[offset] = tmp1[r];
                        else
                            ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp1[r]);
                    }
                }
            }
            else
            {
                if (out_elempack == 4)
                {
                    if (output_elemtype == 1)
                    {
                        float* p1f32 = p0f32 + out_hstep * 4;
                        __msa_st_w((v4i32)_sum00, p0f32, 0);
                        __msa_st_w((v4i32)_sum01, p1f32, 0);
                        __msa_st_w((v4i32)_sum10, p0f32 + 4, 0);
                        __msa_st_w((v4i32)_sum11, p1f32 + 4, 0);
                        p0f32 += 8;
                    }
                    else
                    {
                        unsigned short* p1 = p0 + out_hstep * 4;
                        float2bfloat_msa_store(_sum00, p0);
                        float2bfloat_msa_store(_sum01, p1);
                        float2bfloat_msa_store(_sum10, p0 + 4);
                        float2bfloat_msa_store(_sum11, p1 + 4);
                        p0 += 8;
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[4], tmp1[4], tmp2[4], tmp3[4];
                    __msa_st_w((v4i32)_sum00, tmp0, 0);
                    __msa_st_w((v4i32)_sum01, tmp1, 0);
                    __msa_st_w((v4i32)_sum10, tmp2, 0);
                    __msa_st_w((v4i32)_sum11, tmp3, 0);
                    if (output_elemtype == 1)
                    {
                        p0f32[0] = tmp0[0];
                        p0f32[1] = tmp2[0];
                        p0f32[out_hstep] = tmp0[1];
                        p0f32[out_hstep + 1] = tmp2[1];
                        p0f32[out_hstep * 2] = tmp0[2];
                        p0f32[out_hstep * 2 + 1] = tmp2[2];
                        p0f32[out_hstep * 3] = tmp0[3];
                        p0f32[out_hstep * 3 + 1] = tmp2[3];
                        p0f32[out_hstep * 4] = tmp1[0];
                        p0f32[out_hstep * 4 + 1] = tmp3[0];
                        p0f32[out_hstep * 5] = tmp1[1];
                        p0f32[out_hstep * 5 + 1] = tmp3[1];
                        p0f32[out_hstep * 6] = tmp1[2];
                        p0f32[out_hstep * 6 + 1] = tmp3[2];
                        p0f32[out_hstep * 7] = tmp1[3];
                        p0f32[out_hstep * 7 + 1] = tmp3[3];
                        p0f32 += 2;
                    }
                    else
                    {
                        p0[0] = float32_to_bfloat16(tmp0[0]);
                        p0[1] = float32_to_bfloat16(tmp2[0]);
                        p0[out_hstep] = float32_to_bfloat16(tmp0[1]);
                        p0[out_hstep + 1] = float32_to_bfloat16(tmp2[1]);
                        p0[out_hstep * 2] = float32_to_bfloat16(tmp0[2]);
                        p0[out_hstep * 2 + 1] = float32_to_bfloat16(tmp2[2]);
                        p0[out_hstep * 3] = float32_to_bfloat16(tmp0[3]);
                        p0[out_hstep * 3 + 1] = float32_to_bfloat16(tmp2[3]);
                        p0[out_hstep * 4] = float32_to_bfloat16(tmp1[0]);
                        p0[out_hstep * 4 + 1] = float32_to_bfloat16(tmp3[0]);
                        p0[out_hstep * 5] = float32_to_bfloat16(tmp1[1]);
                        p0[out_hstep * 5 + 1] = float32_to_bfloat16(tmp3[1]);
                        p0[out_hstep * 6] = float32_to_bfloat16(tmp1[2]);
                        p0[out_hstep * 6 + 1] = float32_to_bfloat16(tmp3[2]);
                        p0[out_hstep * 7] = float32_to_bfloat16(tmp1[3]);
                        p0[out_hstep * 7 + 1] = float32_to_bfloat16(tmp3[3]);
                        p0 += 2;
                    }
                }
            }
        }
        for (; jj < max_jj; jj += 1)
        {
            v4f32 _sum0 = (v4f32)__msa_ld_w(pp, 0);
            v4f32 _sum1 = (v4f32)__msa_ld_w(pp + 4, 0);
            pp += 8;

            if (pC0)
            {
                v4f32 _beta = __msa_fill_w_f32(beta);
                if (broadcast_type_C == 0)
                {
                    v4f32 _c = __msa_fmul_w(__msa_fill_w_f32(pC0[0]), _beta);
                    _sum0 = __msa_fadd_w(_sum0, _c);
                    _sum1 = __msa_fadd_w(_sum1, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _sum0 = __msa_fadd_w(_sum0, __msa_fmul_w((v4f32)__msa_ld_w(pC0, 0), _beta));
                    _sum1 = __msa_fadd_w(_sum1, __msa_fmul_w((v4f32)__msa_ld_w(pC0 + 4, 0), _beta));
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        const float* pCj0 = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + (j + jj) * 4;
                        const float* pCj1 = pCj0 + c_hstep * 4;
                        _sum0 = __msa_fmadd_w(_sum0, _beta, (v4f32)__msa_ld_w(pCj0, 0));
                        _sum1 = __msa_fmadd_w(_sum1, _beta, (v4f32)__msa_ld_w(pCj1, 0));
                    }
                    if (c_elempack == 1)
                    {
                        const float* pCj = (const float*)C + (i + ii) * c_hstep + (j + jj);
                        float tmp0[4], tmp1[4];
                        tmp0[0] = pCj[0];
                        tmp0[1] = pCj[c_hstep];
                        tmp0[2] = pCj[c_hstep * 2];
                        tmp0[3] = pCj[c_hstep * 3];
                        tmp1[0] = pCj[c_hstep * 4];
                        tmp1[1] = pCj[c_hstep * 5];
                        tmp1[2] = pCj[c_hstep * 6];
                        tmp1[3] = pCj[c_hstep * 7];
                        _sum0 = __msa_fmadd_w(_sum0, _beta, (v4f32)__msa_ld_w(tmp0, 0));
                        _sum1 = __msa_fmadd_w(_sum1, _beta, (v4f32)__msa_ld_w(tmp1, 0));
                    }
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _c = __msa_fill_w_f32(pC0[0]);
                    _sum0 = __msa_fmadd_w(_sum0, _beta, _c);
                    _sum1 = __msa_fmadd_w(_sum1, _beta, _c);
                    pC0 += 1;
                }
            }

            _sum0 = __msa_fmul_w(_sum0, _valpha);
            _sum1 = __msa_fmul_w(_sum1, _valpha);

            if (output_transpose)
            {
                float tmp0[4];
                float tmp1[4];
                __msa_st_w((v4i32)_sum0, tmp0, 0);
                __msa_st_w((v4i32)_sum1, tmp1, 0);
                int col = j + jj;
                for (int r = 0; r < 4; r++)
                {
                    int row = i + ii + r;
                    size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                    if (output_elemtype == 1)
                        ((float*)top_blob)[offset] = tmp0[r];
                    else
                        ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp0[r]);
                }
                for (int r = 0; r < 4; r++)
                {
                    int row = i + ii + 4 + r;
                    size_t offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                    if (output_elemtype == 1)
                        ((float*)top_blob)[offset] = tmp1[r];
                    else
                        ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp1[r]);
                }
            }
            else
            {
                if (out_elempack == 4)
                {
                    if (output_elemtype == 1)
                    {
                        float* p1f32 = p0f32 + out_hstep * 4;
                        __msa_st_w((v4i32)_sum0, p0f32, 0);
                        __msa_st_w((v4i32)_sum1, p1f32, 0);
                        p0f32 += 4;
                    }
                    else
                    {
                        unsigned short* p1 = p0 + out_hstep * 4;
                        float2bfloat_msa_store(_sum0, p0);
                        float2bfloat_msa_store(_sum1, p1);
                        p0 += 4;
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[4];
                    float tmp1[4];
                    __msa_st_w((v4i32)_sum0, tmp0, 0);
                    __msa_st_w((v4i32)_sum1, tmp1, 0);
                    if (output_elemtype == 1)
                    {
                        p0f32[0] = tmp0[0];
                        p0f32[out_hstep] = tmp0[1];
                        p0f32[out_hstep * 2] = tmp0[2];
                        p0f32[out_hstep * 3] = tmp0[3];
                        p0f32[out_hstep * 4] = tmp1[0];
                        p0f32[out_hstep * 5] = tmp1[1];
                        p0f32[out_hstep * 6] = tmp1[2];
                        p0f32[out_hstep * 7] = tmp1[3];
                        p0f32++;
                    }
                    else
                    {
                        p0[0] = float32_to_bfloat16(tmp0[0]);
                        p0[out_hstep] = float32_to_bfloat16(tmp0[1]);
                        p0[out_hstep * 2] = float32_to_bfloat16(tmp0[2]);
                        p0[out_hstep * 3] = float32_to_bfloat16(tmp0[3]);
                        p0[out_hstep * 4] = float32_to_bfloat16(tmp1[0]);
                        p0[out_hstep * 5] = float32_to_bfloat16(tmp1[1]);
                        p0[out_hstep * 6] = float32_to_bfloat16(tmp1[2]);
                        p0[out_hstep * 7] = float32_to_bfloat16(tmp1[3]);
                        p0++;
                    }
                }
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        unsigned short* p0;
        float* p0f32;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
            p0f32 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (i + ii) / out_elempack * out_hstep * out_elempack + j * out_elempack;
            p0f32 = (float*)top_blob + (i + ii) / out_elempack * out_hstep * out_elempack + j * out_elempack;
        }

        const float* pC0 = pC;
        if (pC0)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC0 = (const float*)C + (i + ii);
            }
            if (broadcast_type_C == 4)
            {
                pC0 = (const float*)C + j;
            }
        }

        v4f32 _valpha = __msa_fill_w_f32(alpha);

        int jj = 0;
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

            if (pC0)
            {
                v4f32 _beta = __msa_fill_w_f32(beta);
                if (broadcast_type_C == 0)
                {
                    v4f32 _c = __msa_fmul_w(__msa_fill_w_f32(pC0[0]), _beta);
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
                    v4f32 _c = __msa_fmul_w((v4f32)__msa_ld_w(pC0, 0), _beta);
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
                        v4f32 _c0 = (v4f32)__msa_ld_w(pCj, 0);
                        v4f32 _c1 = (v4f32)__msa_ld_w(pCj + c_hstep, 0);
                        v4f32 _c2 = (v4f32)__msa_ld_w(pCj + c_hstep * 2, 0);
                        v4f32 _c3 = (v4f32)__msa_ld_w(pCj + c_hstep * 3, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _sum0 = __msa_fmadd_w(_sum0, _beta, _c0);
                        _sum1 = __msa_fmadd_w(_sum1, _beta, _c1);
                        _sum2 = __msa_fmadd_w(_sum2, _beta, _c2);
                        _sum3 = __msa_fmadd_w(_sum3, _beta, _c3);

                        pCj += 4;

                        _c0 = (v4f32)__msa_ld_w(pCj, 0);
                        _c1 = (v4f32)__msa_ld_w(pCj + c_hstep, 0);
                        _c2 = (v4f32)__msa_ld_w(pCj + c_hstep * 2, 0);
                        _c3 = (v4f32)__msa_ld_w(pCj + c_hstep * 3, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _sum4 = __msa_fmadd_w(_sum4, _beta, _c0);
                        _sum5 = __msa_fmadd_w(_sum5, _beta, _c1);
                        _sum6 = __msa_fmadd_w(_sum6, _beta, _c2);
                        _sum7 = __msa_fmadd_w(_sum7, _beta, _c3);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _sum0 = __msa_fmadd_w(_sum0, _beta, __msa_fill_w_f32(pC0[0]));
                    _sum1 = __msa_fmadd_w(_sum1, _beta, __msa_fill_w_f32(pC0[1]));
                    _sum2 = __msa_fmadd_w(_sum2, _beta, __msa_fill_w_f32(pC0[2]));
                    _sum3 = __msa_fmadd_w(_sum3, _beta, __msa_fill_w_f32(pC0[3]));
                    _sum4 = __msa_fmadd_w(_sum4, _beta, __msa_fill_w_f32(pC0[4]));
                    _sum5 = __msa_fmadd_w(_sum5, _beta, __msa_fill_w_f32(pC0[5]));
                    _sum6 = __msa_fmadd_w(_sum6, _beta, __msa_fill_w_f32(pC0[6]));
                    _sum7 = __msa_fmadd_w(_sum7, _beta, __msa_fill_w_f32(pC0[7]));
                    pC0 += 8;
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
                if (output_elemtype == 1)
                {
                    if (out_elempack == 4)
                    {
                        float* p1f32 = p0f32 + out_hstep * 4;

                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);

                        __msa_st_w((v4i32)_sum0, p0f32, 0);
                        __msa_st_w((v4i32)_sum1, p0f32 + 4, 0);
                        __msa_st_w((v4i32)_sum2, p0f32 + 8, 0);
                        __msa_st_w((v4i32)_sum3, p0f32 + 12, 0);
                        __msa_st_w((v4i32)_sum4, p1f32, 0);
                        __msa_st_w((v4i32)_sum5, p1f32 + 4, 0);
                        __msa_st_w((v4i32)_sum6, p1f32 + 8, 0);
                        __msa_st_w((v4i32)_sum7, p1f32 + 12, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __msa_st_w((v4i32)_sum0, p0f32, 0);
                        __msa_st_w((v4i32)_sum1, p0f32 + out_hstep, 0);
                        __msa_st_w((v4i32)_sum2, p0f32 + out_hstep * 2, 0);
                        __msa_st_w((v4i32)_sum3, p0f32 + out_hstep * 3, 0);
                        __msa_st_w((v4i32)_sum4, p0f32 + out_hstep * 4, 0);
                        __msa_st_w((v4i32)_sum5, p0f32 + out_hstep * 5, 0);
                        __msa_st_w((v4i32)_sum6, p0f32 + out_hstep * 6, 0);
                        __msa_st_w((v4i32)_sum7, p0f32 + out_hstep * 7, 0);
                    }
                    p0f32 += out_hstep * 8;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        unsigned short* p1 = p0 + out_hstep * 4;

                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);

                        float2bfloat_msa_store(_sum0, p0);
                        float2bfloat_msa_store(_sum1, p0 + 4);
                        float2bfloat_msa_store(_sum2, p0 + 8);
                        float2bfloat_msa_store(_sum3, p0 + 12);
                        float2bfloat_msa_store(_sum4, p1);
                        float2bfloat_msa_store(_sum5, p1 + 4);
                        float2bfloat_msa_store(_sum6, p1 + 8);
                        float2bfloat_msa_store(_sum7, p1 + 12);
                    }
                    if (out_elempack == 1)
                    {
                        float2bfloat_msa_store(_sum0, p0);
                        float2bfloat_msa_store(_sum1, p0 + out_hstep);
                        float2bfloat_msa_store(_sum2, p0 + out_hstep * 2);
                        float2bfloat_msa_store(_sum3, p0 + out_hstep * 3);
                        float2bfloat_msa_store(_sum4, p0 + out_hstep * 4);
                        float2bfloat_msa_store(_sum5, p0 + out_hstep * 5);
                        float2bfloat_msa_store(_sum6, p0 + out_hstep * 6);
                        float2bfloat_msa_store(_sum7, p0 + out_hstep * 7);
                    }
                    p0 += out_hstep * 8;
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

            if (pC0)
            {
                v4f32 _beta = __msa_fill_w_f32(beta);
                if (broadcast_type_C == 0)
                {
                    v4f32 _c = __msa_fmul_w(__msa_fill_w_f32(pC0[0]), _beta);
                    _sum0 = __msa_fadd_w(_sum0, _c);
                    _sum1 = __msa_fadd_w(_sum1, _c);
                    _sum2 = __msa_fadd_w(_sum2, _c);
                    _sum3 = __msa_fadd_w(_sum3, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _c = __msa_fmul_w((v4f32)__msa_ld_w(pC0, 0), _beta);
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
                        v4f32 _c0 = (v4f32)__msa_ld_w(pCj, 0);
                        v4f32 _c1 = (v4f32)__msa_ld_w(pCj + c_hstep, 0);
                        v4f32 _c2 = (v4f32)__msa_ld_w(pCj + c_hstep * 2, 0);
                        v4f32 _c3 = (v4f32)__msa_ld_w(pCj + c_hstep * 3, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _sum0 = __msa_fmadd_w(_sum0, _beta, _c0);
                        _sum1 = __msa_fmadd_w(_sum1, _beta, _c1);
                        _sum2 = __msa_fmadd_w(_sum2, _beta, _c2);
                        _sum3 = __msa_fmadd_w(_sum3, _beta, _c3);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _sum0 = __msa_fmadd_w(_sum0, _beta, __msa_fill_w_f32(pC0[0]));
                    _sum1 = __msa_fmadd_w(_sum1, _beta, __msa_fill_w_f32(pC0[1]));
                    _sum2 = __msa_fmadd_w(_sum2, _beta, __msa_fill_w_f32(pC0[2]));
                    _sum3 = __msa_fmadd_w(_sum3, _beta, __msa_fill_w_f32(pC0[3]));
                    pC0 += 4;
                }
            }

            _sum0 = __msa_fmul_w(_sum0, _valpha);
            _sum1 = __msa_fmul_w(_sum1, _valpha);
            _sum2 = __msa_fmul_w(_sum2, _valpha);
            _sum3 = __msa_fmul_w(_sum3, _valpha);

            if (output_transpose)
            {
                if (output_elemtype == 1)
                {
                    if (out_elempack == 4)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        __msa_st_w((v4i32)_sum0, p0f32, 0);
                        __msa_st_w((v4i32)_sum1, p0f32 + 4, 0);
                        __msa_st_w((v4i32)_sum2, p0f32 + 8, 0);
                        __msa_st_w((v4i32)_sum3, p0f32 + 12, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __msa_st_w((v4i32)_sum0, p0f32, 0);
                        __msa_st_w((v4i32)_sum1, p0f32 + out_hstep, 0);
                        __msa_st_w((v4i32)_sum2, p0f32 + out_hstep * 2, 0);
                        __msa_st_w((v4i32)_sum3, p0f32 + out_hstep * 3, 0);
                    }
                    p0f32 += out_hstep * 4;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        float2bfloat_msa_store(_sum0, p0);
                        float2bfloat_msa_store(_sum1, p0 + 4);
                        float2bfloat_msa_store(_sum2, p0 + 8);
                        float2bfloat_msa_store(_sum3, p0 + 12);
                    }
                    if (out_elempack == 1)
                    {
                        float2bfloat_msa_store(_sum0, p0);
                        float2bfloat_msa_store(_sum1, p0 + out_hstep);
                        float2bfloat_msa_store(_sum2, p0 + out_hstep * 2);
                        float2bfloat_msa_store(_sum3, p0 + out_hstep * 3);
                    }
                    p0 += out_hstep * 4;
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

            if (pC0)
            {
                v4f32 _beta = __msa_fill_w_f32(beta);
                if (broadcast_type_C == 0)
                {
                    v4f32 _c = __msa_fmul_w(__msa_fill_w_f32(pC0[0]), _beta);
                    _sum0 = __msa_fadd_w(_sum0, _c);
                    _sum1 = __msa_fadd_w(_sum1, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _c = __msa_fmul_w((v4f32)__msa_ld_w(pC0, 0), _beta);
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
                    _sum0 = __msa_fmadd_w(_sum0, _beta, __msa_fill_w_f32(pC0[0]));
                    _sum1 = __msa_fmadd_w(_sum1, _beta, __msa_fill_w_f32(pC0[1]));
                    pC0 += 2;
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

            if (pC0)
            {
                v4f32 _beta = __msa_fill_w_f32(beta);
                if (broadcast_type_C == 0)
                {
                    _sum0 = __msa_fadd_w(_sum0, __msa_fmul_w(__msa_fill_w_f32(pC0[0]), _beta));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _sum0 = __msa_fadd_w(_sum0, __msa_fmul_w((v4f32)__msa_ld_w(pC0, 0), _beta));
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
                    _sum0 = __msa_fmadd_w(_sum0, _beta, __msa_fill_w_f32(pC0[0]));
                    pC0 += 1;
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
        const float* pC0 = pC;
        if (pC0)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pC0 = (const float*)C + (i + ii);
            if (broadcast_type_C == 4)
                pC0 = (const float*)C + j;
        }

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            // topT layout: jj-major, 2 floats per jj (ii0, ii1)
            float sum0 = pp[0];
            float sum1 = pp[1];
            pp += 2;

            if (pC0)
            {
                if (broadcast_type_C == 0)
                {
                    sum0 += pC0[0] * beta;
                    sum1 += pC0[0] * beta;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    sum0 += pC0[0] * beta;
                    sum1 += pC0[1] * beta;
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
                    sum0 += pC0[0] * beta;
                    sum1 += pC0[0] * beta;
                    pC0 += 1;
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
        const float* pC0 = pC;
        if (pC0)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pC0 = (const float*)C + (i + ii);
            if (broadcast_type_C == 4)
                pC0 = (const float*)C + j;
        }

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            float sum = pp[0];
            pp += 1;

            if (pC0)
            {
                if (broadcast_type_C == 0)
                {
                    sum += pC0[0] * beta;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    sum += pC0[0] * beta;
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
                    sum += pC0[0] * beta;
                    pC0 += 1;
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
    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_N = std::max(8, tile_size / 8 * 8);
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
            TILE_M = std::max(8, tile_size / 8 * 8);
            TILE_N = std::max(8, tile_size / 8 * 8);
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
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __mips_msa
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif
    }

    if (nT > 1)
    {
#if __mips_msa
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#else
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
    }

    if (constant_TILE_M > 0)
    {
#if __mips_msa
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
#else
        TILE_M = (constant_TILE_M + 1) / 2 * 2;
#endif
    }
    if (constant_TILE_N > 0)
    {
#if __mips_msa
        TILE_N = (constant_TILE_N + 7) / 8 * 8;
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

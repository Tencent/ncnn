// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pack_A_tile_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if __mips_msa
    const int elempack = A.elempack;
#endif
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
                __builtin_prefetch(p0 + A_hstep);
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
                __builtin_prefetch(p0 + 32);
                __builtin_prefetch(p1 + 32);
                __builtin_prefetch(p2 + 32);
                __builtin_prefetch(p3 + 32);
                __builtin_prefetch(p4 + 32);
                __builtin_prefetch(p5 + 32);
                __builtin_prefetch(p6 + 32);
                __builtin_prefetch(p7 + 32);
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
                __builtin_prefetch(p0 + 32);
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
                __builtin_prefetch(p0 + 32);
                __builtin_prefetch(p1 + 32);
                __builtin_prefetch(p2 + 32);
                __builtin_prefetch(p3 + 32);
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
                __builtin_prefetch(p0 + A_hstep * 4);
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
                __builtin_prefetch(p0 + A_hstep);
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
                __builtin_prefetch(p0 + A_hstep * 4);
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
                __builtin_prefetch(p0 + A_hstep);
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
#if __mips_msa
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __builtin_prefetch(p0 + A_hstep * 4);
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
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            for (int kk = 0; kk < max_kk; kk++)
            {
                __builtin_prefetch(p0 + A_hstep);
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += A_hstep;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
#if __mips_msa
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __builtin_prefetch(p0 + A_hstep * 4);
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
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            for (int kk = 0; kk < max_kk; kk++)
            {
                __builtin_prefetch(p0 + A_hstep);
                pp[0] = p0[0];
                pp += 1;
                p0 += A_hstep;
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
                __builtin_prefetch(p0 + 32);
                __builtin_prefetch(p1 + 32);
                __builtin_prefetch(p2 + 32);
                __builtin_prefetch(p3 + 32);
                __builtin_prefetch(p4 + 32);
                __builtin_prefetch(p5 + 32);
                __builtin_prefetch(p6 + 32);
                __builtin_prefetch(p7 + 32);
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
#if __mips_msa
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __builtin_prefetch(p0 + 32);
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += 4;
            }
        }
#endif // __mips_msa
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 2) * B_hstep + k;
            const unsigned short* p3 = (const unsigned short*)B + (j + jj + 3) * B_hstep + k;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __builtin_prefetch(p0 + 32);
                __builtin_prefetch(p1 + 32);
                __builtin_prefetch(p2 + 32);
                __builtin_prefetch(p3 + 32);
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
                __builtin_prefetch(p0 + 32);
                __builtin_prefetch(p1 + 32);
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
                __builtin_prefetch(p0 + B_hstep);
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
#if __mips_msa
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __builtin_prefetch(p0 + B_hstep * 4);
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
#endif // __mips_msa
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            for (int kk = 0; kk < max_kk; kk++)
            {
                __builtin_prefetch(p0 + B_hstep);
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
#if __mips_msa
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __builtin_prefetch(p0 + B_hstep * 4);
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
#endif // __mips_msa
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            for (int kk = 0; kk < max_kk; kk++)
            {
                __builtin_prefetch(p0 + B_hstep);
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += B_hstep;
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
#if __mips_msa
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __builtin_prefetch(p0 + B_hstep * 4);
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
#endif // __mips_msa
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            for (int kk = 0; kk < max_kk; kk++)
            {
                __builtin_prefetch(p0 + B_hstep);
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
                __builtin_prefetch(pA + 32);
                __builtin_prefetch(pB + 32);
                v4f32 _pA0 = bfloat2float_msa(pA);
                v4f32 _pA1 = bfloat2float_msa(pA + 4);
                v4f32 _pA0r = (v4f32)__msa_shf_w((v4i32)_pA0, _MSA_SHUFFLE(1, 0, 3, 2));
                v4f32 _pA1r = (v4f32)__msa_shf_w((v4i32)_pA1, _MSA_SHUFFLE(1, 0, 3, 2));
                v4f32 _pB0 = bfloat2float_msa(pB);
                v4f32 _pB1 = bfloat2float_msa(pB + 4);
                v4f32 _pB0r = (v4f32)__msa_shf_w((v4i32)_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                v4f32 _pB1r = (v4f32)__msa_shf_w((v4i32)_pB1, _MSA_SHUFFLE(0, 3, 2, 1));

                _sum00 = __ncnn_msa_fmadd_w(_sum00, _pA0, _pB0);
                _sum10 = __ncnn_msa_fmadd_w(_sum10, _pA0, _pB0r);
                _sum20 = __ncnn_msa_fmadd_w(_sum20, _pA0r, _pB0);
                _sum30 = __ncnn_msa_fmadd_w(_sum30, _pA0r, _pB0r);
                _sum40 = __ncnn_msa_fmadd_w(_sum40, _pA0, _pB1);
                _sum50 = __ncnn_msa_fmadd_w(_sum50, _pA0, _pB1r);
                _sum60 = __ncnn_msa_fmadd_w(_sum60, _pA0r, _pB1);
                _sum70 = __ncnn_msa_fmadd_w(_sum70, _pA0r, _pB1r);
                _sum01 = __ncnn_msa_fmadd_w(_sum01, _pA1, _pB0);
                _sum11 = __ncnn_msa_fmadd_w(_sum11, _pA1, _pB0r);
                _sum21 = __ncnn_msa_fmadd_w(_sum21, _pA1r, _pB0);
                _sum31 = __ncnn_msa_fmadd_w(_sum31, _pA1r, _pB0r);
                _sum41 = __ncnn_msa_fmadd_w(_sum41, _pA1, _pB1);
                _sum51 = __ncnn_msa_fmadd_w(_sum51, _pA1, _pB1r);
                _sum61 = __ncnn_msa_fmadd_w(_sum61, _pA1r, _pB1);
                _sum71 = __ncnn_msa_fmadd_w(_sum71, _pA1r, _pB1r);
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
                __builtin_prefetch(pA + 32);
                __builtin_prefetch(pB + 16);
                v4f32 _pA0 = bfloat2float_msa(pA);
                v4f32 _pA1 = bfloat2float_msa(pA + 4);
                v4f32 _pA0r = (v4f32)__msa_shf_w((v4i32)_pA0, _MSA_SHUFFLE(1, 0, 3, 2));
                v4f32 _pA1r = (v4f32)__msa_shf_w((v4i32)_pA1, _MSA_SHUFFLE(1, 0, 3, 2));
                v4f32 _pB = bfloat2float_msa(pB);
                v4f32 _pB1 = (v4f32)__msa_shf_w((v4i32)_pB, _MSA_SHUFFLE(0, 3, 2, 1));

                _sum00 = __ncnn_msa_fmadd_w(_sum00, _pA0, _pB);
                _sum10 = __ncnn_msa_fmadd_w(_sum10, _pA0, _pB1);
                _sum20 = __ncnn_msa_fmadd_w(_sum20, _pA0r, _pB);
                _sum30 = __ncnn_msa_fmadd_w(_sum30, _pA0r, _pB1);
                _sum01 = __ncnn_msa_fmadd_w(_sum01, _pA1, _pB);
                _sum11 = __ncnn_msa_fmadd_w(_sum11, _pA1, _pB1);
                _sum21 = __ncnn_msa_fmadd_w(_sum21, _pA1r, _pB);
                _sum31 = __ncnn_msa_fmadd_w(_sum31, _pA1r, _pB1);
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
                __builtin_prefetch(pA + 32);
                __builtin_prefetch(pB + 8);
                v4f32 _pA0 = bfloat2float_msa(pA);
                v4f32 _pA1 = bfloat2float_msa(pA + 4);
                int v;
                memcpy(&v, pB, 4);
                v8i16 _zero = (v8i16)__msa_fill_w(0);
                v8i16 _raw = (v8i16)__msa_fill_w(v);
                v4f32 _pB = (v4f32)__msa_ilvr_h(_raw, _zero);
                v4f32 _pB1 = (v4f32)__msa_shf_w((v4i32)_pB, _MSA_SHUFFLE(2, 3, 0, 1));

                _sum00 = __ncnn_msa_fmadd_w(_sum00, _pA0, _pB);
                _sum10 = __ncnn_msa_fmadd_w(_sum10, _pA0, _pB1);
                _sum01 = __ncnn_msa_fmadd_w(_sum01, _pA1, _pB);
                _sum11 = __ncnn_msa_fmadd_w(_sum11, _pA1, _pB1);
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
                __builtin_prefetch(pA + 32);
                __builtin_prefetch(pB + 4);
                v4f32 _pA0 = bfloat2float_msa(pA);
                v4f32 _pA1 = bfloat2float_msa(pA + 4);
                v4f32 _pB = __msa_fill_w_f32(bfloat16_to_float32(pB[0]));
                _sum0 = __ncnn_msa_fmadd_w(_sum0, _pA0, _pB);
                _sum1 = __ncnn_msa_fmadd_w(_sum1, _pA1, _pB);
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
                __builtin_prefetch(pA + 32);
                __builtin_prefetch(pB + 32);
                v4f32 _pA = bfloat2float_msa(pA);
                v4f32 _pA1 = (v4f32)__msa_shf_w((v4i32)_pA, _MSA_SHUFFLE(1, 0, 3, 2));
                v4f32 _pB0 = bfloat2float_msa(pB);
                v4f32 _pB1 = bfloat2float_msa(pB + 4);
                v4f32 _pB0r = (v4f32)__msa_shf_w((v4i32)_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                v4f32 _pB1r = (v4f32)__msa_shf_w((v4i32)_pB1, _MSA_SHUFFLE(0, 3, 2, 1));

                _sum0 = __ncnn_msa_fmadd_w(_sum0, _pA, _pB0);
                _sum1 = __ncnn_msa_fmadd_w(_sum1, _pA, _pB0r);
                _sum2 = __ncnn_msa_fmadd_w(_sum2, _pA1, _pB0);
                _sum3 = __ncnn_msa_fmadd_w(_sum3, _pA1, _pB0r);
                _sum4 = __ncnn_msa_fmadd_w(_sum4, _pA, _pB1);
                _sum5 = __ncnn_msa_fmadd_w(_sum5, _pA, _pB1r);
                _sum6 = __ncnn_msa_fmadd_w(_sum6, _pA1, _pB1);
                _sum7 = __ncnn_msa_fmadd_w(_sum7, _pA1, _pB1r);

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
                __builtin_prefetch(pA + 32);
                __builtin_prefetch(pB + 16);
                v4f32 _pA = bfloat2float_msa(pA);
                v4f32 _pA1 = (v4f32)__msa_shf_w((v4i32)_pA, _MSA_SHUFFLE(1, 0, 3, 2));
                v4f32 _pB = bfloat2float_msa(pB);
                v4f32 _pB1 = (v4f32)__msa_shf_w((v4i32)_pB, _MSA_SHUFFLE(0, 3, 2, 1));

                _sum0 = __ncnn_msa_fmadd_w(_sum0, _pA, _pB);
                _sum1 = __ncnn_msa_fmadd_w(_sum1, _pA, _pB1);
                _sum2 = __ncnn_msa_fmadd_w(_sum2, _pA1, _pB);
                _sum3 = __ncnn_msa_fmadd_w(_sum3, _pA1, _pB1);

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
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 8);
                v4f32 _pA = bfloat2float_msa(pA);
                int v;
                memcpy(&v, pB, 4);
                v8i16 _zero = (v8i16)__msa_fill_w(0);
                v8i16 _raw = (v8i16)__msa_fill_w(v);
                v4f32 _pB = (v4f32)__msa_ilvr_h(_raw, _zero);
                v4f32 _pB1 = (v4f32)__msa_shf_w((v4i32)_pB, _MSA_SHUFFLE(2, 3, 0, 1));

                _sum0 = __ncnn_msa_fmadd_w(_sum0, _pA, _pB);
                _sum1 = __ncnn_msa_fmadd_w(_sum1, _pA, _pB1);

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
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 4);
                v4f32 _pA = bfloat2float_msa(pA);

                _sum0 = __ncnn_msa_fmadd_w(_sum0, _pA, __msa_fill_w_f32(bfloat16_to_float32(pB[0])));

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
                _sum2 = (v4f32)__msa_ld_w(outptr + 8, 0);
                _sum3 = (v4f32)__msa_ld_w(outptr + 12, 0);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 32);
                v4f32 _pA = bfloat2float_msa(pA);
                v4f32 _pA0 = (v4f32)__msa_ilvr_d((v2i64)_pA, (v2i64)_pA);
                v4f32 _pA1 = (v4f32)__msa_ilvl_d((v2i64)_pA, (v2i64)_pA);
                v4f32 _pB0 = bfloat2float_msa(pB);
                v4f32 _pB1 = bfloat2float_msa(pB + 4);
                v4f32 _pB2 = bfloat2float_msa(pB + 8);
                v4f32 _pB3 = bfloat2float_msa(pB + 12);
                v4f32 _pB01 = (v4f32)__msa_ilvr_w((v4i32)_pB0, (v4i32)_pB0);
                v4f32 _pB23 = (v4f32)__msa_ilvl_w((v4i32)_pB0, (v4i32)_pB0);
                v4f32 _pB45 = (v4f32)__msa_ilvr_w((v4i32)_pB1, (v4i32)_pB1);
                v4f32 _pB67 = (v4f32)__msa_ilvl_w((v4i32)_pB1, (v4i32)_pB1);
                _sum0 = __ncnn_msa_fmadd_w(_sum0, _pA0, _pB01);
                _sum1 = __ncnn_msa_fmadd_w(_sum1, _pA0, _pB23);
                _sum2 = __ncnn_msa_fmadd_w(_sum2, _pA0, _pB45);
                _sum3 = __ncnn_msa_fmadd_w(_sum3, _pA0, _pB67);
                _pB01 = (v4f32)__msa_ilvr_w((v4i32)_pB2, (v4i32)_pB2);
                _pB23 = (v4f32)__msa_ilvl_w((v4i32)_pB2, (v4i32)_pB2);
                _pB45 = (v4f32)__msa_ilvr_w((v4i32)_pB3, (v4i32)_pB3);
                _pB67 = (v4f32)__msa_ilvl_w((v4i32)_pB3, (v4i32)_pB3);
                _sum0 = __ncnn_msa_fmadd_w(_sum0, _pA1, _pB01);
                _sum1 = __ncnn_msa_fmadd_w(_sum1, _pA1, _pB23);
                _sum2 = __ncnn_msa_fmadd_w(_sum2, _pA1, _pB45);
                _sum3 = __ncnn_msa_fmadd_w(_sum3, _pA1, _pB67);
                pA += 4;
                pB += 16;
            }

            __msa_st_w((v4i32)_sum0, outptr, 0);
            __msa_st_w((v4i32)_sum1, outptr + 4, 0);
            __msa_st_w((v4i32)_sum2, outptr + 8, 0);
            __msa_st_w((v4i32)_sum3, outptr + 12, 0);

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
                outptr[0] += a0 * b0;
                outptr[1] += a1 * b0;
                outptr[2] += a0 * b1;
                outptr[3] += a1 * b1;
                outptr[4] += a0 * b2;
                outptr[5] += a1 * b2;
                outptr[6] += a0 * b3;
                outptr[7] += a1 * b3;
                outptr[8] += a0 * b4;
                outptr[9] += a1 * b4;
                outptr[10] += a0 * b5;
                outptr[11] += a1 * b5;
                outptr[12] += a0 * b6;
                outptr[13] += a1 * b6;
                outptr[14] += a0 * b7;
                outptr[15] += a1 * b7;
                pA += 2;
                pB += 8;
            }

            outptr += 16;
        }
#endif // __mips_msa

        for (; jj + 3 < max_jj; jj += 4)
        {
#if __mips_msa
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
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 16);
                v4f32 _pA = bfloat2float_msa(pA);
                v4f32 _pA0 = (v4f32)__msa_ilvr_d((v2i64)_pA, (v2i64)_pA);
                v4f32 _pA1 = (v4f32)__msa_ilvl_d((v2i64)_pA, (v2i64)_pA);
                v4f32 _pB0 = bfloat2float_msa(pB);
                v4f32 _pB1 = bfloat2float_msa(pB + 4);
                v4f32 _pB01 = (v4f32)__msa_ilvr_w((v4i32)_pB0, (v4i32)_pB0);
                v4f32 _pB23 = (v4f32)__msa_ilvl_w((v4i32)_pB0, (v4i32)_pB0);
                _sum0 = __ncnn_msa_fmadd_w(_sum0, _pA0, _pB01);
                _sum1 = __ncnn_msa_fmadd_w(_sum1, _pA0, _pB23);
                _pB01 = (v4f32)__msa_ilvr_w((v4i32)_pB1, (v4i32)_pB1);
                _pB23 = (v4f32)__msa_ilvl_w((v4i32)_pB1, (v4i32)_pB1);
                _sum0 = __ncnn_msa_fmadd_w(_sum0, _pA1, _pB01);
                _sum1 = __ncnn_msa_fmadd_w(_sum1, _pA1, _pB23);
                pA += 4;
                pB += 8;
            }

            __msa_st_w((v4i32)_sum0, outptr, 0);
            __msa_st_w((v4i32)_sum1, outptr + 4, 0);

            for (; kk < max_kk; kk++)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                float a1 = bfloat16_to_float32(pA[1]);
                outptr[0] += a0 * bfloat16_to_float32(pB[0]);
                outptr[1] += a1 * bfloat16_to_float32(pB[0]);
                outptr[2] += a0 * bfloat16_to_float32(pB[1]);
                outptr[3] += a1 * bfloat16_to_float32(pB[1]);
                outptr[4] += a0 * bfloat16_to_float32(pB[2]);
                outptr[5] += a1 * bfloat16_to_float32(pB[2]);
                outptr[6] += a0 * bfloat16_to_float32(pB[3]);
                outptr[7] += a1 * bfloat16_to_float32(pB[3]);
                pA += 2;
                pB += 4;
            }
#else
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
            for (int kk = 0; kk < max_kk; kk++)
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
#endif // __mips_msa

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
            for (int kk = 0; kk < max_kk; kk++)
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
            for (int kk = 0; kk < max_kk; kk++)
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
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 32);
                v4f32 _pA = (v4f32)__msa_fill_w((int)((unsigned int)pA[0] << 16));
                v4f32 _pB0 = bfloat2float_msa(pB);
                v4f32 _pB1 = bfloat2float_msa(pB + 4);
                _sum0 = __ncnn_msa_fmadd_w(_sum0, _pA, _pB0);
                _sum1 = __ncnn_msa_fmadd_w(_sum1, _pA, _pB1);
                pA += 1;
                pB += 8;
            }

            __msa_st_w((v4i32)_sum0, outptr, 0);
            __msa_st_w((v4i32)_sum1, outptr + 4, 0);

            outptr += 8;
        }
#endif // __mips_msa

        for (; jj + 3 < max_jj; jj += 4)
        {
#if __mips_msa
            v4f32 _sum0;

            if (k == 0)
            {
                _sum0 = (v4f32)__msa_fill_w(0);
            }
            else
            {
                _sum0 = (v4f32)__msa_ld_w(outptr, 0);
            }

            const unsigned short* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 16);
                v4f32 _pA = (v4f32)__msa_fill_w((int)((unsigned int)pA[0] << 16));
                v4f32 _pB = bfloat2float_msa(pB);
                _sum0 = __ncnn_msa_fmadd_w(_sum0, _pA, _pB);
                pA += 1;
                pB += 4;
            }

            __msa_st_w((v4i32)_sum0, outptr, 0);
#else
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
            for (int kk = 0; kk < max_kk; kk++)
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
#endif // __mips_msa

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
            for (int kk = 0; kk < max_kk; kk++)
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
            for (int kk = 0; kk < max_kk; kk++)
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

// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pack_A_tile_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 8;
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
                p0 += 8;
            }
        }
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
#if __loongarch_asx
        if (elempack == 8)
        {
            const int i_pack = (i + ii) / 8;
            const int i_lane = (i + ii) % 8;
            const unsigned short* p0 = (const unsigned short*)A + (size_t)i_pack * A_hstep * 8 + k * 8 + i_lane;
            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += A_hstep * 8;
            }
        }
#endif // __loongarch_asx
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
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
        const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;

#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            const int i_pack = (i + ii) / 8;
            const int i_lane = (i + ii) % 8;
            p0 = (const unsigned short*)A + (size_t)i_pack * A_hstep * 8 + k * 8 + i_lane;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += A_hstep * 8;
            }
        }
#endif // __loongarch_asx
#endif // __loongarch_sx
        for (int kk = 0; kk < max_kk; kk++)
        {
            if (elempack == 1)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp += 2;
                p0++;
                p1++;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;

#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            const int i_pack = (i + ii) / 8;
            const int i_lane = (i + ii) % 8;
            p0 = (const unsigned short*)A + (size_t)i_pack * A_hstep * 8 + k * 8 + i_lane;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += A_hstep * 8;
            }
        }
#endif // __loongarch_asx
#endif // __loongarch_sx
        for (int kk = 0; kk < max_kk; kk++)
        {
            if (elempack == 1)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_A_tile_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 8)
        {
            if ((k & 7) == 0)
            {
                const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;
                for (int kk = 0; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[8];
                    pp[2] = p0[16];
                    pp[3] = p0[24];
                    pp[4] = p0[32];
                    pp[5] = p0[40];
                    pp[6] = p0[48];
                    pp[7] = p0[56];
                    pp += 8;
                    p0++;
                    if ((kk + 1) % 8 == 0)
                    {
                        p0 += A_hstep * 8 - 8;
                    }
                }
            }
            else
            {
                const int k_pack = k / 8;
                const int k_lane = k % 8;
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k_pack * A_hstep * 8 + (i + ii) * 8 + k_lane;
                for (int kk = 0; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[8];
                    pp[2] = p0[16];
                    pp[3] = p0[24];
                    pp[4] = p0[32];
                    pp[5] = p0[40];
                    pp[6] = p0[48];
                    pp[7] = p0[56];
                    pp += 8;
                    p0++;
                    if ((k_lane + kk + 1) % 8 == 0)
                    {
                        p0 += A_hstep * 8 - 8;
                    }
                }
            }
        }
        if (elempack == 4)
        {
            const int k_pack = k / 4;
            const int k_lane = k % 4;
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k_pack * A_hstep * 4 + (i + ii) * 4 + k_lane;
            for (int kk = 0; kk < max_kk; kk++)
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
                if ((k_lane + kk + 1) % 4 == 0)
                {
                    p0 += A_hstep * 4 - 4;
                }
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
#if __loongarch_asx
        if (elempack == 8)
        {
            if ((k & 7) == 0)
            {
                const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

                int kk = 0;
                for (; kk + 7 < max_kk; kk += 8)
                {
                    for (int q = 0; q < 8; q++)
                    {
                        pp[q * 4] = p0[q];
                        pp[q * 4 + 1] = p0[8 + q];
                        pp[q * 4 + 2] = p0[16 + q];
                        pp[q * 4 + 3] = p0[24 + q];
                    }
                    pp += 32;
                    p0 += A_hstep * 8;
                }
                for (; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[8];
                    pp[2] = p0[16];
                    pp[3] = p0[24];
                    pp += 4;
                    p0++;
                    if ((kk + 1) % 8 == 0)
                    {
                        p0 += A_hstep * 8 - 8;
                    }
                }
            }
            else
            {
                const int k_pack = k / 8;
                const int k_lane = k % 8;
                const unsigned short* p0 = (const unsigned short*)A + (size_t)k_pack * A_hstep * 8 + (i + ii) * 8 + k_lane;

                for (int kk = 0; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[8];
                    pp[2] = p0[16];
                    pp[3] = p0[24];
                    pp += 4;
                    p0++;
                    if ((k_lane + kk + 1) % 8 == 0)
                    {
                        p0 += A_hstep * 8 - 8;
                    }
                }
            }
        }
#endif // __loongarch_asx
        if (elempack == 4)
        {
            const int k_pack = k / 4;
            const int k_lane = k % 4;
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k_pack * A_hstep * 4 + (i + ii) * 4 + k_lane;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp[2] = p0[8];
                pp[3] = p0[12];
                pp += 4;
                p0++;
                if ((k_lane + kk + 1) % 4 == 0)
                {
                    p0 += A_hstep * 4 - 4;
                }
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
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * elempack;
#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            if ((k & 7) == 0)
            {
                int kk = 0;
                for (; kk + 7 < max_kk; kk += 8)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[8];
                    pp[2] = p0[1];
                    pp[3] = p0[9];
                    pp[4] = p0[2];
                    pp[5] = p0[10];
                    pp[6] = p0[3];
                    pp[7] = p0[11];
                    pp[8] = p0[4];
                    pp[9] = p0[12];
                    pp[10] = p0[5];
                    pp[11] = p0[13];
                    pp[12] = p0[6];
                    pp[13] = p0[14];
                    pp[14] = p0[7];
                    pp[15] = p0[15];
                    pp += 16;
                    p0 += A_hstep * 8;
                }
                for (; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[8];
                    pp += 2;
                    p0++;
                    if ((kk + 1) % 8 == 0)
                    {
                        p0 += A_hstep * 8 - 8;
                    }
                }
            }
            else
            {
                const int k_pack = k / 8;
                const int k_lane = k % 8;
                p0 = (const unsigned short*)A + (size_t)k_pack * A_hstep * 8 + (i + ii) * 8 + k_lane;

                for (int kk = 0; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[8];
                    pp += 2;
                    p0++;
                    if ((k_lane + kk + 1) % 8 == 0)
                    {
                        p0 += A_hstep * 8 - 8;
                    }
                }
            }
        }
#endif // __loongarch_asx
        if (elempack == 4)
        {
            const int k_pack = k / 4;
            const int k_lane = k % 4;
            p0 = (const unsigned short*)A + (size_t)k_pack * A_hstep * 4 + (i + ii) * 4 + k_lane;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp += 2;
                p0++;
                if ((k_lane + kk + 1) % 4 == 0)
                {
                    p0 += A_hstep * 4 - 4;
                }
            }
        }
#endif // __loongarch_sx
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
#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            if ((k & 7) == 0)
            {
                int kk = 0;
                for (; kk + 7 < max_kk; kk += 8)
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
                    p0 += A_hstep * 8;
                }
                for (; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp += 1;
                    p0++;
                    if ((kk + 1) % 8 == 0)
                    {
                        p0 += A_hstep * 8 - 8;
                    }
                }
            }
            else
            {
                const int k_pack = k / 8;
                const int k_lane = k % 8;
                p0 = (const unsigned short*)A + (size_t)k_pack * A_hstep * 8 + (i + ii) * 8 + k_lane;

                for (int kk = 0; kk < max_kk; kk++)
                {
                    pp[0] = p0[0];
                    pp += 1;
                    p0++;
                    if ((k_lane + kk + 1) % 8 == 0)
                    {
                        p0 += A_hstep * 8 - 8;
                    }
                }
            }
        }
#endif // __loongarch_asx
        if (elempack == 4)
        {
            const int k_pack = k / 4;
            const int k_lane = k % 4;
            p0 = (const unsigned short*)A + (size_t)k_pack * A_hstep * 4 + (i + ii) * 4 + k_lane;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
                if ((k_lane + kk + 1) % 4 == 0)
                {
                    p0 += A_hstep * 4 - 4;
                }
            }
        }
#endif // __loongarch_sx
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
#if __loongarch_sx
    for (; jj + 7 < max_jj; jj += 8)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            if (((j + jj) & 7) == 0)
            {
                const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 8;

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
                    p0 += 8;
                }
            }
            else
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    for (int c = 0; c < 8; c++)
                    {
                        const int col = j + jj + c;
                        const int j_pack = col / 8;
                        const int j_lane = col % 8;
                        const unsigned short* p = (const unsigned short*)B + (size_t)j_pack * B_hstep * 8 + (k + kk) * 8 + j_lane;
                        pp[c] = p[0];
                    }
                    pp += 8;
                }
            }
        }
#endif // __loongarch_asx
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
#endif // __loongarch_sx
    for (; jj + 3 < max_jj; jj += 4)
    {
        const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * elempack;
#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            const int j_pack = (j + jj) / 8;
            const int j_lane = (j + jj) % 8;
            p0 = (const unsigned short*)B + (size_t)j_pack * B_hstep * 8 + k * 8 + j_lane;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += 8;
            }
        }
#endif // __loongarch_asx
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
#endif // __loongarch_sx
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

#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            const int j_pack = (j + jj) / 8;
            const int j_lane = (j + jj) % 8;
            p0 = (const unsigned short*)B + (size_t)j_pack * B_hstep * 8 + k * 8 + j_lane;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += 8;
            }
            continue;
        }
#endif // __loongarch_asx
#endif // __loongarch_sx
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

#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            const int j_pack = (j + jj) / 8;
            const int j_lane = (j + jj) % 8;
            p0 = (const unsigned short*)B + (size_t)j_pack * B_hstep * 8 + k * 8 + j_lane;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += 8;
            }
            continue;
        }
#endif // __loongarch_asx
#endif // __loongarch_sx
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
#if __loongarch_sx
    for (; jj + 7 < max_jj; jj += 8)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[8];
                pp[2] = p0[16];
                pp[3] = p0[24];
                pp[4] = p0[32];
                pp[5] = p0[40];
                pp[6] = p0[48];
                pp[7] = p0[56];
                pp += 8;
                p0++;
                if ((kk + 1) % 8 == 0)
                {
                    p0 += B_hstep * 8 - 8;
                }
            }
        }
#endif // __loongarch_asx
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
#endif // __loongarch_sx
    for (; jj + 3 < max_jj; jj += 4)
    {
        const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * elempack;
#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = p0[0];
                pp[1] = p0[8];
                pp[2] = p0[16];
                pp[3] = p0[24];
                pp[4] = p0[1];
                pp[5] = p0[9];
                pp[6] = p0[17];
                pp[7] = p0[25];
                pp[8] = p0[2];
                pp[9] = p0[10];
                pp[10] = p0[18];
                pp[11] = p0[26];
                pp[12] = p0[3];
                pp[13] = p0[11];
                pp[14] = p0[19];
                pp[15] = p0[27];
                pp[16] = p0[4];
                pp[17] = p0[12];
                pp[18] = p0[20];
                pp[19] = p0[28];
                pp[20] = p0[5];
                pp[21] = p0[13];
                pp[22] = p0[21];
                pp[23] = p0[29];
                pp[24] = p0[6];
                pp[25] = p0[14];
                pp[26] = p0[22];
                pp[27] = p0[30];
                pp[28] = p0[7];
                pp[29] = p0[15];
                pp[30] = p0[23];
                pp[31] = p0[31];
                pp += 32;
                p0 += B_hstep * 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[8];
                pp[2] = p0[16];
                pp[3] = p0[24];
                pp += 4;
                p0++;
            }
        }
#endif // __loongarch_asx
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
#endif // __loongarch_sx
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
#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = p0[0];
                pp[1] = p0[8];
                pp[2] = p0[1];
                pp[3] = p0[9];
                pp[4] = p0[2];
                pp[5] = p0[10];
                pp[6] = p0[3];
                pp[7] = p0[11];
                pp[8] = p0[4];
                pp[9] = p0[12];
                pp[10] = p0[5];
                pp[11] = p0[13];
                pp[12] = p0[6];
                pp[13] = p0[14];
                pp[14] = p0[7];
                pp[15] = p0[15];
                pp += 16;
                p0 += B_hstep * 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[8];
                pp += 2;
                p0++;
            }
        }
#endif // __loongarch_asx
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
#endif // __loongarch_sx
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
#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
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
                p0 += B_hstep * 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
#endif // __loongarch_asx
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
#endif // __loongarch_sx
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
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
#if __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

            __m256 _sum0 = (__m256)__lasx_xvldi(0);
            __m256 _sum1 = (__m256)__lasx_xvldi(0);
            __m256 _sum2 = (__m256)__lasx_xvldi(0);
            __m256 _sum3 = (__m256)__lasx_xvldi(0);
            __m256 _sum4 = (__m256)__lasx_xvldi(0);
            __m256 _sum5 = (__m256)__lasx_xvldi(0);
            __m256 _sum6 = (__m256)__lasx_xvldi(0);
            __m256 _sum7 = (__m256)__lasx_xvldi(0);

            if (k != 0)
            {
                _sum0 = (__m256)__lasx_xvld(outptr, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
                _sum2 = (__m256)__lasx_xvld(outptr + 16, 0);
                _sum3 = (__m256)__lasx_xvld(outptr + 24, 0);
                _sum4 = (__m256)__lasx_xvld(outptr + 32, 0);
                _sum5 = (__m256)__lasx_xvld(outptr + 40, 0);
                _sum6 = (__m256)__lasx_xvld(outptr + 48, 0);
                _sum7 = (__m256)__lasx_xvld(outptr + 56, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __m256 _pA = bfloat2float_lasx((__m128i)__lsx_vld(pA, 0));
                __m256 _pA1 = (__m256)__lasx_xvshuf4i_w((__m256i)_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m256 _pB0 = bfloat2float_lasx((__m128i)__lsx_vld(pB, 0));
                __m256 _pB1 = (__m256)__lasx_xvshuf4i_w((__m256i)_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                __m256 _pB2 = (__m256)__lasx_xvpermi_q((__m256i)_pB0, (__m256i)_pB0, _LSX_SHUFFLE(0, 0, 0, 1));
                __m256 _pB3 = (__m256)__lasx_xvshuf4i_w((__m256i)_pB2, _LSX_SHUFFLE(0, 3, 2, 1));

                _sum0 = __lasx_xvfmadd_s(_pA, _pB0, _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA, _pB1, _sum1);
                _sum2 = __lasx_xvfmadd_s(_pA1, _pB0, _sum2);
                _sum3 = __lasx_xvfmadd_s(_pA1, _pB1, _sum3);
                _sum4 = __lasx_xvfmadd_s(_pA, _pB2, _sum4);
                _sum5 = __lasx_xvfmadd_s(_pA, _pB3, _sum5);
                _sum6 = __lasx_xvfmadd_s(_pA1, _pB2, _sum6);
                _sum7 = __lasx_xvfmadd_s(_pA1, _pB3, _sum7);

                pA += 8;
                pB += 8;
            }

            __lasx_xvst((__m256i)_sum0, outptr, 0);
            __lasx_xvst((__m256i)_sum1, outptr + 8, 0);
            __lasx_xvst((__m256i)_sum2, outptr + 16, 0);
            __lasx_xvst((__m256i)_sum3, outptr + 24, 0);
            __lasx_xvst((__m256i)_sum4, outptr + 32, 0);
            __lasx_xvst((__m256i)_sum5, outptr + 40, 0);
            __lasx_xvst((__m256i)_sum6, outptr + 48, 0);
            __lasx_xvst((__m256i)_sum7, outptr + 56, 0);

            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

            __m256 _sum0 = (__m256)__lasx_xvldi(0);
            __m256 _sum1 = (__m256)__lasx_xvldi(0);
            __m256 _sum2 = (__m256)__lasx_xvldi(0);
            __m256 _sum3 = (__m256)__lasx_xvldi(0);

            if (k != 0)
            {
                _sum0 = (__m256)__lasx_xvld(outptr, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
                _sum2 = (__m256)__lasx_xvld(outptr + 16, 0);
                _sum3 = (__m256)__lasx_xvld(outptr + 24, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __m256 _pA = bfloat2float_lasx((__m128i)__lsx_vld(pA, 0));
                __m256 _pA1 = (__m256)__lasx_xvshuf4i_w((__m256i)_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _pB4 = bfloat2float_lsx(pB);
                __m256 _pB = __lasx_concat_128_s(_pB4, _pB4);
                __m256 _pB1 = (__m256)__lasx_xvshuf4i_w((__m256i)_pB, _LSX_SHUFFLE(0, 3, 2, 1));

                _sum0 = __lasx_xvfmadd_s(_pA, _pB, _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA, _pB1, _sum1);
                _sum2 = __lasx_xvfmadd_s(_pA1, _pB, _sum2);
                _sum3 = __lasx_xvfmadd_s(_pA1, _pB1, _sum3);

                pA += 8;
                pB += 4;
            }

            __lasx_xvst((__m256i)_sum0, outptr, 0);
            __lasx_xvst((__m256i)_sum1, outptr + 8, 0);
            __lasx_xvst((__m256i)_sum2, outptr + 16, 0);
            __lasx_xvst((__m256i)_sum3, outptr + 24, 0);

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* pA = pAT;

            __m256 _sum0 = (__m256)__lasx_xvldi(0);
            __m256 _sum1 = (__m256)__lasx_xvldi(0);

            if (k != 0)
            {
                _sum0 = (__m256)__lasx_xvld(outptr, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __m256 _pA = bfloat2float_lasx((__m128i)__lsx_vld(pA, 0));
                __m128 _pB01 = bfloat2float_lsx(__lsx_vldrepl_w(pB, 0));
                __m256 _pB = __lasx_concat_128_s(_pB01, _pB01);
                __m256 _pB1 = (__m256)__lasx_xvshuf4i_w((__m256i)_pB, _LSX_SHUFFLE(2, 3, 0, 1));

                _sum0 = __lasx_xvfmadd_s(_pA, _pB, _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA, _pB1, _sum1);

                pA += 8;
                pB += 2;
            }

            __lasx_xvst((__m256i)_sum0, outptr, 0);
            __lasx_xvst((__m256i)_sum1, outptr + 8, 0);

            outptr += 16;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned short* pA = pAT;

            __m256 _sum0 = (__m256)__lasx_xvldi(0);

            if (k != 0)
            {
                _sum0 = (__m256)__lasx_xvld(outptr, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __m256 _pA = bfloat2float_lasx((__m128i)__lsx_vld(pA, 0));

                __m256 _pB = (__m256)__lasx_xvreplgr2vr_w((int)((unsigned int)pB[0] << 16));
                _sum0 = __lasx_xvfmadd_s(_pA, _pB, _sum0);

                pA += 8;
                pB += 1;
            }

            __lasx_xvst((__m256i)_sum0, outptr, 0);

            outptr += 8;
        }
#else // __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _sum00;
            __m128 _sum01;
            __m128 _sum10;
            __m128 _sum11;
            __m128 _sum20;
            __m128 _sum21;
            __m128 _sum30;
            __m128 _sum31;
            __m128 _sum40;
            __m128 _sum41;
            __m128 _sum50;
            __m128 _sum51;
            __m128 _sum60;
            __m128 _sum61;
            __m128 _sum70;
            __m128 _sum71;

            if (k == 0)
            {
                _sum00 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum01 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum10 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum11 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum20 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum21 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum30 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum31 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum40 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum41 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum50 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum51 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum60 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum61 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum70 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum71 = (__m128)__lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum00 = (__m128)__lsx_vld(outptr, 0);
                _sum01 = (__m128)__lsx_vld(outptr + 4, 0);
                _sum10 = (__m128)__lsx_vld(outptr + 8, 0);
                _sum11 = (__m128)__lsx_vld(outptr + 12, 0);
                _sum20 = (__m128)__lsx_vld(outptr + 16, 0);
                _sum21 = (__m128)__lsx_vld(outptr + 20, 0);
                _sum30 = (__m128)__lsx_vld(outptr + 24, 0);
                _sum31 = (__m128)__lsx_vld(outptr + 28, 0);
                _sum40 = (__m128)__lsx_vld(outptr + 32, 0);
                _sum41 = (__m128)__lsx_vld(outptr + 36, 0);
                _sum50 = (__m128)__lsx_vld(outptr + 40, 0);
                _sum51 = (__m128)__lsx_vld(outptr + 44, 0);
                _sum60 = (__m128)__lsx_vld(outptr + 48, 0);
                _sum61 = (__m128)__lsx_vld(outptr + 52, 0);
                _sum70 = (__m128)__lsx_vld(outptr + 56, 0);
                _sum71 = (__m128)__lsx_vld(outptr + 60, 0);
            }

            const unsigned short* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pA0 = bfloat2float_lsx(pA);
                __m128 _pA1 = bfloat2float_lsx(pA + 4);
                __m128 _pA0r = (__m128)__lsx_vshuf4i_w((__m128i)_pA0, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _pA1r = (__m128)__lsx_vshuf4i_w((__m128i)_pA1, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _pB0 = bfloat2float_lsx(pB);
                __m128 _pB1 = bfloat2float_lsx(pB + 4);
                __m128 _pB0r = (__m128)__lsx_vshuf4i_w((__m128i)_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                __m128 _pB1r = (__m128)__lsx_vshuf4i_w((__m128i)_pB1, _LSX_SHUFFLE(0, 3, 2, 1));

                _sum00 = __lsx_vfmadd_s(_pA0, _pB0, _sum00);
                _sum10 = __lsx_vfmadd_s(_pA0, _pB0r, _sum10);
                _sum20 = __lsx_vfmadd_s(_pA0r, _pB0, _sum20);
                _sum30 = __lsx_vfmadd_s(_pA0r, _pB0r, _sum30);
                _sum40 = __lsx_vfmadd_s(_pA0, _pB1, _sum40);
                _sum50 = __lsx_vfmadd_s(_pA0, _pB1r, _sum50);
                _sum60 = __lsx_vfmadd_s(_pA0r, _pB1, _sum60);
                _sum70 = __lsx_vfmadd_s(_pA0r, _pB1r, _sum70);
                _sum01 = __lsx_vfmadd_s(_pA1, _pB0, _sum01);
                _sum11 = __lsx_vfmadd_s(_pA1, _pB0r, _sum11);
                _sum21 = __lsx_vfmadd_s(_pA1r, _pB0, _sum21);
                _sum31 = __lsx_vfmadd_s(_pA1r, _pB0r, _sum31);
                _sum41 = __lsx_vfmadd_s(_pA1, _pB1, _sum41);
                _sum51 = __lsx_vfmadd_s(_pA1, _pB1r, _sum51);
                _sum61 = __lsx_vfmadd_s(_pA1r, _pB1, _sum61);
                _sum71 = __lsx_vfmadd_s(_pA1r, _pB1r, _sum71);

                pA += 8;
                pB += 8;
            }

            __lsx_vst((__m128i)_sum00, outptr, 0);
            __lsx_vst((__m128i)_sum01, outptr + 4, 0);
            __lsx_vst((__m128i)_sum10, outptr + 8, 0);
            __lsx_vst((__m128i)_sum11, outptr + 12, 0);
            __lsx_vst((__m128i)_sum20, outptr + 16, 0);
            __lsx_vst((__m128i)_sum21, outptr + 20, 0);
            __lsx_vst((__m128i)_sum30, outptr + 24, 0);
            __lsx_vst((__m128i)_sum31, outptr + 28, 0);
            __lsx_vst((__m128i)_sum40, outptr + 32, 0);
            __lsx_vst((__m128i)_sum41, outptr + 36, 0);
            __lsx_vst((__m128i)_sum50, outptr + 40, 0);
            __lsx_vst((__m128i)_sum51, outptr + 44, 0);
            __lsx_vst((__m128i)_sum60, outptr + 48, 0);
            __lsx_vst((__m128i)_sum61, outptr + 52, 0);
            __lsx_vst((__m128i)_sum70, outptr + 56, 0);
            __lsx_vst((__m128i)_sum71, outptr + 60, 0);

            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _sum00;
            __m128 _sum01;
            __m128 _sum10;
            __m128 _sum11;
            __m128 _sum20;
            __m128 _sum21;
            __m128 _sum30;
            __m128 _sum31;

            if (k == 0)
            {
                _sum00 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum01 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum10 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum11 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum20 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum21 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum30 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum31 = (__m128)__lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum00 = (__m128)__lsx_vld(outptr, 0);
                _sum01 = (__m128)__lsx_vld(outptr + 4, 0);
                _sum10 = (__m128)__lsx_vld(outptr + 8, 0);
                _sum11 = (__m128)__lsx_vld(outptr + 12, 0);
                _sum20 = (__m128)__lsx_vld(outptr + 16, 0);
                _sum21 = (__m128)__lsx_vld(outptr + 20, 0);
                _sum30 = (__m128)__lsx_vld(outptr + 24, 0);
                _sum31 = (__m128)__lsx_vld(outptr + 28, 0);
            }

            const unsigned short* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pA0 = bfloat2float_lsx(pA);
                __m128 _pA1 = bfloat2float_lsx(pA + 4);
                __m128 _pA0r = (__m128)__lsx_vshuf4i_w((__m128i)_pA0, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _pA1r = (__m128)__lsx_vshuf4i_w((__m128i)_pA1, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _pB = bfloat2float_lsx(pB);
                __m128 _pB1 = (__m128)__lsx_vshuf4i_w((__m128i)_pB, _LSX_SHUFFLE(0, 3, 2, 1));

                _sum00 = __lsx_vfmadd_s(_pA0, _pB, _sum00);
                _sum10 = __lsx_vfmadd_s(_pA0, _pB1, _sum10);
                _sum20 = __lsx_vfmadd_s(_pA0r, _pB, _sum20);
                _sum30 = __lsx_vfmadd_s(_pA0r, _pB1, _sum30);
                _sum01 = __lsx_vfmadd_s(_pA1, _pB, _sum01);
                _sum11 = __lsx_vfmadd_s(_pA1, _pB1, _sum11);
                _sum21 = __lsx_vfmadd_s(_pA1r, _pB, _sum21);
                _sum31 = __lsx_vfmadd_s(_pA1r, _pB1, _sum31);

                pA += 8;
                pB += 4;
            }

            __lsx_vst((__m128i)_sum00, outptr, 0);
            __lsx_vst((__m128i)_sum01, outptr + 4, 0);
            __lsx_vst((__m128i)_sum10, outptr + 8, 0);
            __lsx_vst((__m128i)_sum11, outptr + 12, 0);
            __lsx_vst((__m128i)_sum20, outptr + 16, 0);
            __lsx_vst((__m128i)_sum21, outptr + 20, 0);
            __lsx_vst((__m128i)_sum30, outptr + 24, 0);
            __lsx_vst((__m128i)_sum31, outptr + 28, 0);

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _sum00;
            __m128 _sum01;
            __m128 _sum10;
            __m128 _sum11;

            if (k == 0)
            {
                _sum00 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum01 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum10 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum11 = (__m128)__lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum00 = (__m128)__lsx_vld(outptr, 0);
                _sum01 = (__m128)__lsx_vld(outptr + 4, 0);
                _sum10 = (__m128)__lsx_vld(outptr + 8, 0);
                _sum11 = (__m128)__lsx_vld(outptr + 12, 0);
            }

            const unsigned short* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pA0 = bfloat2float_lsx(pA);
                __m128 _pA1 = bfloat2float_lsx(pA + 4);
                __m128 _pB = bfloat2float_lsx(__lsx_vldrepl_w(pB, 0));
                __m128 _pB1 = (__m128)__lsx_vshuf4i_w((__m128i)_pB, _LSX_SHUFFLE(2, 3, 0, 1));
                _sum00 = __lsx_vfmadd_s(_pA0, _pB, _sum00);
                _sum10 = __lsx_vfmadd_s(_pA0, _pB1, _sum10);
                _sum01 = __lsx_vfmadd_s(_pA1, _pB, _sum01);
                _sum11 = __lsx_vfmadd_s(_pA1, _pB1, _sum11);
                pA += 8;
                pB += 2;
            }

            __lsx_vst((__m128i)_sum00, outptr, 0);
            __lsx_vst((__m128i)_sum01, outptr + 4, 0);
            __lsx_vst((__m128i)_sum10, outptr + 8, 0);
            __lsx_vst((__m128i)_sum11, outptr + 12, 0);

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
            }

            const unsigned short* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pA0 = bfloat2float_lsx(pA);
                __m128 _pA1 = bfloat2float_lsx(pA + 4);
                __m128 _pB = (__m128)__lsx_vreplgr2vr_w((int)((unsigned int)pB[0] << 16));
                _sum0 = __lsx_vfmadd_s(_pA0, _pB, _sum0);
                _sum1 = __lsx_vfmadd_s(_pA1, _pB, _sum1);
                pA += 8;
                pB += 1;
            }

            __lsx_vst((__m128i)_sum0, outptr, 0);
            __lsx_vst((__m128i)_sum1, outptr + 4, 0);

            outptr += 8;
        }

#endif // __loongarch_asx
        pAT += max_kk * 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

            __m128 _sum0;
            __m128 _sum1;
            __m128 _sum2;
            __m128 _sum3;
            __m128 _sum4;
            __m128 _sum5;
            __m128 _sum6;
            __m128 _sum7;

            if (k == 0)
            {
                _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum3 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum4 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum5 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum6 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum7 = (__m128)__lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
                _sum2 = (__m128)__lsx_vld(outptr + 4 * 2, 0);
                _sum3 = (__m128)__lsx_vld(outptr + 4 * 3, 0);
                _sum4 = (__m128)__lsx_vld(outptr + 4 * 4, 0);
                _sum5 = (__m128)__lsx_vld(outptr + 4 * 5, 0);
                _sum6 = (__m128)__lsx_vld(outptr + 4 * 6, 0);
                _sum7 = (__m128)__lsx_vld(outptr + 4 * 7, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = bfloat2float_lsx(pA);
                __m128 _pA1 = (__m128)__lsx_vshuf4i_w((__m128i)_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _pB0 = bfloat2float_lsx(pB);
                __m128 _pB1 = bfloat2float_lsx(pB + 4);
                __m128 _pB0r = (__m128)__lsx_vshuf4i_w((__m128i)_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                __m128 _pB1r = (__m128)__lsx_vshuf4i_w((__m128i)_pB1, _LSX_SHUFFLE(0, 3, 2, 1));

                _sum0 = __lsx_vfmadd_s(_pA, _pB0, _sum0);
                _sum1 = __lsx_vfmadd_s(_pA, _pB0r, _sum1);
                _sum2 = __lsx_vfmadd_s(_pA1, _pB0, _sum2);
                _sum3 = __lsx_vfmadd_s(_pA1, _pB0r, _sum3);
                _sum4 = __lsx_vfmadd_s(_pA, _pB1, _sum4);
                _sum5 = __lsx_vfmadd_s(_pA, _pB1r, _sum5);
                _sum6 = __lsx_vfmadd_s(_pA1, _pB1, _sum6);
                _sum7 = __lsx_vfmadd_s(_pA1, _pB1r, _sum7);

                pA += 4;
                pB += 8;
            }

            __lsx_vst((__m128i)_sum0, outptr, 0);
            __lsx_vst((__m128i)_sum1, outptr + 4, 0);
            __lsx_vst((__m128i)_sum2, outptr + 4 * 2, 0);
            __lsx_vst((__m128i)_sum3, outptr + 4 * 3, 0);
            __lsx_vst((__m128i)_sum4, outptr + 4 * 4, 0);
            __lsx_vst((__m128i)_sum5, outptr + 4 * 5, 0);
            __lsx_vst((__m128i)_sum6, outptr + 4 * 6, 0);
            __lsx_vst((__m128i)_sum7, outptr + 4 * 7, 0);

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

            __m128 _sum0;
            __m128 _sum1;
            __m128 _sum2;
            __m128 _sum3;

            if (k == 0)
            {
                _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum3 = (__m128)__lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
                _sum2 = (__m128)__lsx_vld(outptr + 4 * 2, 0);
                _sum3 = (__m128)__lsx_vld(outptr + 4 * 3, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = bfloat2float_lsx(pA);
                __m128 _pA1 = (__m128)__lsx_vshuf4i_w((__m128i)_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _pB = bfloat2float_lsx(pB);
                __m128 _pB1 = (__m128)__lsx_vshuf4i_w((__m128i)_pB, _LSX_SHUFFLE(0, 3, 2, 1));

                _sum0 = __lsx_vfmadd_s(_pA, _pB, _sum0);
                _sum1 = __lsx_vfmadd_s(_pA, _pB1, _sum1);
                _sum2 = __lsx_vfmadd_s(_pA1, _pB, _sum2);
                _sum3 = __lsx_vfmadd_s(_pA1, _pB1, _sum3);

                pA += 4;
                pB += 4;
            }

            __lsx_vst((__m128i)_sum0, outptr, 0);
            __lsx_vst((__m128i)_sum1, outptr + 4, 0);
            __lsx_vst((__m128i)_sum2, outptr + 4 * 2, 0);
            __lsx_vst((__m128i)_sum3, outptr + 4 * 3, 0);

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* pA = pAT;

            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = bfloat2float_lsx(pA);
                __m128 _pB = bfloat2float_lsx(__lsx_vldrepl_w(pB, 0));
                __m128 _pB1 = (__m128)__lsx_vshuf4i_w((__m128i)_pB, _LSX_SHUFFLE(2, 3, 0, 1));

                _sum0 = __lsx_vfmadd_s(_pA, _pB, _sum0);
                _sum1 = __lsx_vfmadd_s(_pA, _pB1, _sum1);

                pA += 4;
                pB += 2;
            }

            __lsx_vst((__m128i)_sum0, outptr, 0);
            __lsx_vst((__m128i)_sum1, outptr + 4, 0);

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            const unsigned short* pA = pAT;

            __m128 _sum0;

            if (k == 0)
            {
                _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = bfloat2float_lsx(pA);

                __m128 _pB = (__m128)__lsx_vreplgr2vr_w((int)((unsigned int)pB[0] << 16));
                _sum0 = __lsx_vfmadd_s(_pA, _pB, _sum0);

                pA += 4;
                pB += 1;
            }

            __lsx_vst((__m128i)_sum0, outptr, 0);

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
#if __loongarch_sx
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _sum0;
            __m128 _sum1;
            __m128 _sum2;
            __m128 _sum3;

            if (k == 0)
            {
                _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum3 = (__m128)__lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
                _sum2 = (__m128)__lsx_vld(outptr + 8, 0);
                _sum3 = (__m128)__lsx_vld(outptr + 12, 0);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128 _pA = bfloat2float_lsx(pA);
                __m128 _pA0 = (__m128)__lsx_vilvl_d((__m128i)_pA, (__m128i)_pA);
                __m128 _pA1 = (__m128)__lsx_vilvh_d((__m128i)_pA, (__m128i)_pA);
                __m128 _pB0 = bfloat2float_lsx(pB);
                __m128 _pB1 = bfloat2float_lsx(pB + 4);
                __m128 _pB2 = bfloat2float_lsx(pB + 8);
                __m128 _pB3 = bfloat2float_lsx(pB + 12);
                __m128 _pB01 = (__m128)__lsx_vilvl_w((__m128i)_pB0, (__m128i)_pB0);
                __m128 _pB23 = (__m128)__lsx_vilvh_w((__m128i)_pB0, (__m128i)_pB0);
                __m128 _pB45 = (__m128)__lsx_vilvl_w((__m128i)_pB1, (__m128i)_pB1);
                __m128 _pB67 = (__m128)__lsx_vilvh_w((__m128i)_pB1, (__m128i)_pB1);
                _sum0 = __lsx_vfmadd_s(_pA0, _pB01, _sum0);
                _sum1 = __lsx_vfmadd_s(_pA0, _pB23, _sum1);
                _sum2 = __lsx_vfmadd_s(_pA0, _pB45, _sum2);
                _sum3 = __lsx_vfmadd_s(_pA0, _pB67, _sum3);
                _pB01 = (__m128)__lsx_vilvl_w((__m128i)_pB2, (__m128i)_pB2);
                _pB23 = (__m128)__lsx_vilvh_w((__m128i)_pB2, (__m128i)_pB2);
                _pB45 = (__m128)__lsx_vilvl_w((__m128i)_pB3, (__m128i)_pB3);
                _pB67 = (__m128)__lsx_vilvh_w((__m128i)_pB3, (__m128i)_pB3);
                _sum0 = __lsx_vfmadd_s(_pA1, _pB01, _sum0);
                _sum1 = __lsx_vfmadd_s(_pA1, _pB23, _sum1);
                _sum2 = __lsx_vfmadd_s(_pA1, _pB45, _sum2);
                _sum3 = __lsx_vfmadd_s(_pA1, _pB67, _sum3);
                pA += 4;
                pB += 16;
            }

            __lsx_vst((__m128i)_sum0, outptr, 0);
            __lsx_vst((__m128i)_sum1, outptr + 4, 0);
            __lsx_vst((__m128i)_sum2, outptr + 8, 0);
            __lsx_vst((__m128i)_sum3, outptr + 12, 0);

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
                outptr[8] += a0 * bfloat16_to_float32(pB[4]);
                outptr[9] += a1 * bfloat16_to_float32(pB[4]);
                outptr[10] += a0 * bfloat16_to_float32(pB[5]);
                outptr[11] += a1 * bfloat16_to_float32(pB[5]);
                outptr[12] += a0 * bfloat16_to_float32(pB[6]);
                outptr[13] += a1 * bfloat16_to_float32(pB[6]);
                outptr[14] += a0 * bfloat16_to_float32(pB[7]);
                outptr[15] += a1 * bfloat16_to_float32(pB[7]);
                pA += 2;
                pB += 8;
            }

            outptr += 16;
        }
#endif // __loongarch_sx
        for (; jj + 3 < max_jj; jj += 4)
        {
#if __loongarch_sx
            {
                __m128 _sum0;
                __m128 _sum1;

                if (k == 0)
                {
                    _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                }
                else
                {
                    _sum0 = (__m128)__lsx_vld(outptr, 0);
                    _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
                }

                const unsigned short* pA = pAT;
                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    __m128 _pA = bfloat2float_lsx(pA);
                    __m128 _pA0 = (__m128)__lsx_vilvl_d((__m128i)_pA, (__m128i)_pA);
                    __m128 _pA1 = (__m128)__lsx_vilvh_d((__m128i)_pA, (__m128i)_pA);
                    __m128 _pB0 = bfloat2float_lsx(pB);
                    __m128 _pB1 = bfloat2float_lsx(pB + 4);
                    __m128 _pB01 = (__m128)__lsx_vilvl_w((__m128i)_pB0, (__m128i)_pB0);
                    __m128 _pB23 = (__m128)__lsx_vilvh_w((__m128i)_pB0, (__m128i)_pB0);
                    _sum0 = __lsx_vfmadd_s(_pA0, _pB01, _sum0);
                    _sum1 = __lsx_vfmadd_s(_pA0, _pB23, _sum1);
                    _pB01 = (__m128)__lsx_vilvl_w((__m128i)_pB1, (__m128i)_pB1);
                    _pB23 = (__m128)__lsx_vilvh_w((__m128i)_pB1, (__m128i)_pB1);
                    _sum0 = __lsx_vfmadd_s(_pA1, _pB01, _sum0);
                    _sum1 = __lsx_vfmadd_s(_pA1, _pB23, _sum1);
                    pA += 4;
                    pB += 8;
                }

                __lsx_vst((__m128i)_sum0, outptr, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);

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

                outptr += 8;
                continue;
            }
#endif // __loongarch_sx
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
#if __loongarch_sx
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __m128 _pB0 = bfloat2float_lsx(pB);
                __m128 _pB1 = bfloat2float_lsx(pB + 4);
                __m128 _pA0 = (__m128)__lsx_vreplgr2vr_w((int)((unsigned int)pA[0] << 16));
                _sum0 = __lsx_vfmadd_s(_pA0, _pB0, _sum0);
                _sum1 = __lsx_vfmadd_s(_pA0, _pB1, _sum1);
                pA += 1;
                pB += 8;
            }

            __lsx_vst((__m128i)_sum0, outptr, 0);
            __lsx_vst((__m128i)_sum1, outptr + 4, 0);

            outptr += 8;
        }
#endif // __loongarch_sx
        for (; jj + 3 < max_jj; jj += 4)
        {
#if __loongarch_sx
            {
                __m128 _sum0;

                if (k == 0)
                {
                    _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                }
                else
                {
                    _sum0 = (__m128)__lsx_vld(outptr, 0);
                }

                const unsigned short* pA = pAT;
                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    __m128 _pA = (__m128)__lsx_vreplgr2vr_w((int)((unsigned int)pA[0] << 16));
                    __m128 _pB = bfloat2float_lsx(pB);
                    _sum0 = __lsx_vfmadd_s(_pA, _pB, _sum0);
                    pA += 1;
                    pB += 4;
                }

                __lsx_vst((__m128i)_sum0, outptr, 0);

                outptr += 4;
                continue;
            }
#endif // __loongarch_sx
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

static void get_optimal_tile_mnk_bf16(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // bf16 uses 2 bytes per element for A and B, but 4 bytes for accumulator
    int tile_size = (int)sqrtf((float)l2_cache_size / (2 * sizeof(unsigned short) + sizeof(float)));

#if __loongarch_sx
    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_N = std::max(8, tile_size / 8 * 8);
#if __loongarch_asx
    TILE_K = std::max(8, tile_size / 8 * 8);
#else
    TILE_K = std::max(4, tile_size / 4 * 4);
#endif
#else
    TILE_M = std::max(2, tile_size / 2 * 2);
    TILE_N = std::max(1, tile_size);
    TILE_K = std::max(2, tile_size / 2 * 2);
#endif

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __loongarch_sx
#if __loongarch_asx
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
#endif
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(unsigned short) / TILE_K);
#if __loongarch_sx
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
#if __loongarch_sx
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __loongarch_sx
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif
    }

    if (nT > 1)
    {
#if __loongarch_sx
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#else
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
    }

    if (constant_TILE_M > 0)
    {
#if __loongarch_sx
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
#else
        TILE_M = (constant_TILE_M + 1) / 2 * 2;
#endif
    }
    if (constant_TILE_N > 0)
    {
#if __loongarch_sx
        TILE_N = (constant_TILE_N + 7) / 8 * 8;
#else
        TILE_N = constant_TILE_N;
#endif
    }
    if (constant_TILE_K > 0)
    {
#if __loongarch_sx
#if __loongarch_asx
        TILE_K = (constant_TILE_K + 7) / 8 * 8;
#else
        TILE_K = (constant_TILE_K + 3) / 4 * 4;
#endif
#else
        TILE_K = (constant_TILE_K + 1) / 2 * 2;
#endif
    }
}

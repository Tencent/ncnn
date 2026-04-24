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
#endif
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
#endif
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
#if __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
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

                _sum0 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[1])), _sum1);
                _sum2 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[2])), _sum2);
                _sum3 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[3])), _sum3);
                _sum4 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[4])), _sum4);
                _sum5 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[5])), _sum5);
                _sum6 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[6])), _sum6);
                _sum7 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[7])), _sum7);

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

                _sum0 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[1])), _sum1);
                _sum2 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[2])), _sum2);
                _sum3 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[3])), _sum3);

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

                _sum0 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[1])), _sum1);

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

                _sum0 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[0])), _sum0);

                pA += 8;
                pB += 1;
            }

            __lasx_xvst((__m256i)_sum0, outptr, 0);

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
#endif // __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
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

                __m128 _pB0 = __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[0]));
                __m128 _pB1 = __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[1]));
                __m128 _pB2 = __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[2]));
                __m128 _pB3 = __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[3]));
                __m128 _pB4 = __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[4]));
                __m128 _pB5 = __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[5]));
                __m128 _pB6 = __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[6]));
                __m128 _pB7 = __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[7]));

                _sum00 = __lsx_vfmadd_s(_pA0, _pB0, _sum00);
                _sum01 = __lsx_vfmadd_s(_pA1, _pB0, _sum01);
                _sum10 = __lsx_vfmadd_s(_pA0, _pB1, _sum10);
                _sum11 = __lsx_vfmadd_s(_pA1, _pB1, _sum11);
                _sum20 = __lsx_vfmadd_s(_pA0, _pB2, _sum20);
                _sum21 = __lsx_vfmadd_s(_pA1, _pB2, _sum21);
                _sum30 = __lsx_vfmadd_s(_pA0, _pB3, _sum30);
                _sum31 = __lsx_vfmadd_s(_pA1, _pB3, _sum31);
                _sum40 = __lsx_vfmadd_s(_pA0, _pB4, _sum40);
                _sum41 = __lsx_vfmadd_s(_pA1, _pB4, _sum41);
                _sum50 = __lsx_vfmadd_s(_pA0, _pB5, _sum50);
                _sum51 = __lsx_vfmadd_s(_pA1, _pB5, _sum51);
                _sum60 = __lsx_vfmadd_s(_pA0, _pB6, _sum60);
                _sum61 = __lsx_vfmadd_s(_pA1, _pB6, _sum61);
                _sum70 = __lsx_vfmadd_s(_pA0, _pB7, _sum70);
                _sum71 = __lsx_vfmadd_s(_pA1, _pB7, _sum71);

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
                __m128 _pB0 = __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[0]));
                __m128 _pB1 = __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[1]));
                __m128 _pB2 = __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[2]));
                __m128 _pB3 = __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[3]));

                _sum00 = __lsx_vfmadd_s(_pA0, _pB0, _sum00);
                _sum01 = __lsx_vfmadd_s(_pA1, _pB0, _sum01);
                _sum10 = __lsx_vfmadd_s(_pA0, _pB1, _sum10);
                _sum11 = __lsx_vfmadd_s(_pA1, _pB1, _sum11);
                _sum20 = __lsx_vfmadd_s(_pA0, _pB2, _sum20);
                _sum21 = __lsx_vfmadd_s(_pA1, _pB2, _sum21);
                _sum30 = __lsx_vfmadd_s(_pA0, _pB3, _sum30);
                _sum31 = __lsx_vfmadd_s(_pA1, _pB3, _sum31);

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
                __m128 _pB0 = __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[0]));
                __m128 _pB1 = __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[1]));
                _sum00 = __lsx_vfmadd_s(_pA0, _pB0, _sum00);
                _sum01 = __lsx_vfmadd_s(_pA1, _pB0, _sum01);
                _sum10 = __lsx_vfmadd_s(_pA0, _pB1, _sum10);
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
                __m128 _pB = __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[0]));
                _sum0 = __lsx_vfmadd_s(_pA0, _pB, _sum0);
                _sum1 = __lsx_vfmadd_s(_pA1, _pB, _sum1);
                pA += 8;
                pB += 1;
            }

            __lsx_vst((__m128i)_sum0, outptr, 0);
            __lsx_vst((__m128i)_sum1, outptr + 4, 0);

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

                _sum0 = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[1])), _sum1);
                _sum2 = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[2])), _sum2);
                _sum3 = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[3])), _sum3);
                _sum4 = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[4])), _sum4);
                _sum5 = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[5])), _sum5);
                _sum6 = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[6])), _sum6);
                _sum7 = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[7])), _sum7);

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

                _sum0 = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[1])), _sum1);
                _sum2 = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[2])), _sum2);
                _sum3 = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[3])), _sum3);

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

                _sum0 = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[1])), _sum1);

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

                _sum0 = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[0])), _sum0);

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
                __m128 _tmp0 = (__m128)__lsx_vld(outptr, 0);
                __m128 _tmp1 = (__m128)__lsx_vld(outptr + 4, 0);
                __m128 _tmp2 = (__m128)__lsx_vld(outptr + 8, 0);
                __m128 _tmp3 = (__m128)__lsx_vld(outptr + 12, 0);
                _sum00 = (__m128)__lsx_vpickev_w((__m128i)_tmp1, (__m128i)_tmp0);
                _sum01 = (__m128)__lsx_vpickev_w((__m128i)_tmp3, (__m128i)_tmp2);
                _sum10 = (__m128)__lsx_vpickod_w((__m128i)_tmp1, (__m128i)_tmp0);
                _sum11 = (__m128)__lsx_vpickod_w((__m128i)_tmp3, (__m128i)_tmp2);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __m128 _pB0 = bfloat2float_lsx(pB);
                __m128 _pB1 = bfloat2float_lsx(pB + 4);

                __m128 _pA0 = __lsx_vreplfr2vr_s(bfloat16_to_float32(pA[0]));
                _sum00 = __lsx_vfmadd_s(_pA0, _pB0, _sum00);
                _sum01 = __lsx_vfmadd_s(_pA0, _pB1, _sum01);
                __m128 _pA1 = __lsx_vreplfr2vr_s(bfloat16_to_float32(pA[1]));
                _sum10 = __lsx_vfmadd_s(_pA1, _pB0, _sum10);
                _sum11 = __lsx_vfmadd_s(_pA1, _pB1, _sum11);

                pA += 2;
                pB += 8;
            }

            __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_sum10, (__m128i)_sum00);
            __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_sum10, (__m128i)_sum00);
            __m128 _tmp2 = (__m128)__lsx_vilvl_w((__m128i)_sum11, (__m128i)_sum01);
            __m128 _tmp3 = (__m128)__lsx_vilvh_w((__m128i)_sum11, (__m128i)_sum01);
            __lsx_vst((__m128i)_tmp0, outptr, 0);
            __lsx_vst((__m128i)_tmp1, outptr + 4, 0);
            __lsx_vst((__m128i)_tmp2, outptr + 8, 0);
            __lsx_vst((__m128i)_tmp3, outptr + 12, 0);

            outptr += 16;
        }
#endif // __loongarch_sx
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
                __m128 _pA0 = __lsx_vreplfr2vr_s(bfloat16_to_float32(pA[0]));
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
#if __loongarch_sx
#if __loongarch_asx
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
            p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
            p0f32 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        const float* pC0 = pC;
        if (pC0)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pC0 = (const float*)C + (i + ii);
            if (broadcast_type_C == 3)
            {
                if (c_elempack == 8)
                    pC0 = (const float*)C + (size_t)((i + ii) / 8) * c_hstep * 8 + j * 8;
                else if (c_elempack == 4)
                    pC0 = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + j * 4;
                else
                    pC0 = (const float*)C + (size_t)(i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
                pC0 = (const float*)C + j;
        }

        __m256 _valpha = (__m256)__lasx_xvreplfr2vr_s(alpha);

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _sum0 = (__m256)__lasx_xvld(pp, 0);
            __m256 _sum1 = (__m256)__lasx_xvld(pp + 8, 0);
            __m256 _sum2 = (__m256)__lasx_xvld(pp + 16, 0);
            __m256 _sum3 = (__m256)__lasx_xvld(pp + 24, 0);
            __m256 _sum4 = (__m256)__lasx_xvld(pp + 32, 0);
            __m256 _sum5 = (__m256)__lasx_xvld(pp + 40, 0);
            __m256 _sum6 = (__m256)__lasx_xvld(pp + 48, 0);
            __m256 _sum7 = (__m256)__lasx_xvld(pp + 56, 0);
            pp += 64;

            if (pC0)
            {
                __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pC0[0]);
                    _sum0 = __lasx_xvfmadd_s(_c, _beta, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_c, _beta, _sum1);
                    _sum2 = __lasx_xvfmadd_s(_c, _beta, _sum2);
                    _sum3 = __lasx_xvfmadd_s(_c, _beta, _sum3);
                    _sum4 = __lasx_xvfmadd_s(_c, _beta, _sum4);
                    _sum5 = __lasx_xvfmadd_s(_c, _beta, _sum5);
                    _sum6 = __lasx_xvfmadd_s(_c, _beta, _sum6);
                    _sum7 = __lasx_xvfmadd_s(_c, _beta, _sum7);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m256 _c = (__m256)__lasx_xvld(pC0, 0);
                    _sum0 = __lasx_xvfmadd_s(_c, _beta, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_c, _beta, _sum1);
                    _sum2 = __lasx_xvfmadd_s(_c, _beta, _sum2);
                    _sum3 = __lasx_xvfmadd_s(_c, _beta, _sum3);
                    _sum4 = __lasx_xvfmadd_s(_c, _beta, _sum4);
                    _sum5 = __lasx_xvfmadd_s(_c, _beta, _sum5);
                    _sum6 = __lasx_xvfmadd_s(_c, _beta, _sum6);
                    _sum7 = __lasx_xvfmadd_s(_c, _beta, _sum7);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c0;
                    __m256 _c1;
                    __m256 _c2;
                    __m256 _c3;
                    __m256 _c4;
                    __m256 _c5;
                    __m256 _c6;
                    __m256 _c7;
                    if (c_elempack == 8)
                    {
                        _c0 = (__m256)__lasx_xvld(pC0, 0);
                        _c1 = (__m256)__lasx_xvld(pC0 + 8, 0);
                        _c2 = (__m256)__lasx_xvld(pC0 + 16, 0);
                        _c3 = (__m256)__lasx_xvld(pC0 + 24, 0);
                        _c4 = (__m256)__lasx_xvld(pC0 + 32, 0);
                        _c5 = (__m256)__lasx_xvld(pC0 + 40, 0);
                        _c6 = (__m256)__lasx_xvld(pC0 + 48, 0);
                        _c7 = (__m256)__lasx_xvld(pC0 + 56, 0);
                        pC0 += 64;
                    }
                    else if (c_elempack == 4)
                    {
                        __m128 _c0l = (__m128)__lsx_vld(pC0, 0);
                        __m128 _c1l = (__m128)__lsx_vld(pC0 + 4, 0);
                        __m128 _c2l = (__m128)__lsx_vld(pC0 + 8, 0);
                        __m128 _c3l = (__m128)__lsx_vld(pC0 + 12, 0);
                        __m128 _c4l = (__m128)__lsx_vld(pC0 + 16, 0);
                        __m128 _c5l = (__m128)__lsx_vld(pC0 + 20, 0);
                        __m128 _c6l = (__m128)__lsx_vld(pC0 + 24, 0);
                        __m128 _c7l = (__m128)__lsx_vld(pC0 + 28, 0);
                        const float* pC1 = pC0 + c_hstep * 4;
                        __m128 _c0h = (__m128)__lsx_vld(pC1, 0);
                        __m128 _c1h = (__m128)__lsx_vld(pC1 + 4, 0);
                        __m128 _c2h = (__m128)__lsx_vld(pC1 + 8, 0);
                        __m128 _c3h = (__m128)__lsx_vld(pC1 + 12, 0);
                        __m128 _c4h = (__m128)__lsx_vld(pC1 + 16, 0);
                        __m128 _c5h = (__m128)__lsx_vld(pC1 + 20, 0);
                        __m128 _c6h = (__m128)__lsx_vld(pC1 + 24, 0);
                        __m128 _c7h = (__m128)__lsx_vld(pC1 + 28, 0);
                        _c0 = combine4x2_ps(_c0l, _c0h);
                        _c1 = combine4x2_ps(_c1l, _c1h);
                        _c2 = combine4x2_ps(_c2l, _c2h);
                        _c3 = combine4x2_ps(_c3l, _c3h);
                        _c4 = combine4x2_ps(_c4l, _c4h);
                        _c5 = combine4x2_ps(_c5l, _c5h);
                        _c6 = combine4x2_ps(_c6l, _c6h);
                        _c7 = combine4x2_ps(_c7l, _c7h);
                        pC0 += 32;
                    }
                    else // c_elempack == 1
                    {
                        _c0 = (__m256)__lasx_xvld(pC0, 0);
                        _c1 = (__m256)__lasx_xvld(pC0 + c_hstep, 0);
                        _c2 = (__m256)__lasx_xvld(pC0 + c_hstep * 2, 0);
                        _c3 = (__m256)__lasx_xvld(pC0 + c_hstep * 3, 0);
                        _c4 = (__m256)__lasx_xvld(pC0 + c_hstep * 4, 0);
                        _c5 = (__m256)__lasx_xvld(pC0 + c_hstep * 5, 0);
                        _c6 = (__m256)__lasx_xvld(pC0 + c_hstep * 6, 0);
                        _c7 = (__m256)__lasx_xvld(pC0 + c_hstep * 7, 0);
                        transpose8x8_ps(_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7);
                        pC0 += 8;
                    }
                    _sum0 = __lasx_xvfmadd_s(_c0, _beta, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_c1, _beta, _sum1);
                    _sum2 = __lasx_xvfmadd_s(_c2, _beta, _sum2);
                    _sum3 = __lasx_xvfmadd_s(_c3, _beta, _sum3);
                    _sum4 = __lasx_xvfmadd_s(_c4, _beta, _sum4);
                    _sum5 = __lasx_xvfmadd_s(_c5, _beta, _sum5);
                    _sum6 = __lasx_xvfmadd_s(_c6, _beta, _sum6);
                    _sum7 = __lasx_xvfmadd_s(_c7, _beta, _sum7);
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c0 = (__m256)__lasx_xvreplfr2vr_s(pC0[0]);
                    __m256 _c1 = (__m256)__lasx_xvreplfr2vr_s(pC0[1]);
                    __m256 _c2 = (__m256)__lasx_xvreplfr2vr_s(pC0[2]);
                    __m256 _c3 = (__m256)__lasx_xvreplfr2vr_s(pC0[3]);
                    __m256 _c4 = (__m256)__lasx_xvreplfr2vr_s(pC0[4]);
                    __m256 _c5 = (__m256)__lasx_xvreplfr2vr_s(pC0[5]);
                    __m256 _c6 = (__m256)__lasx_xvreplfr2vr_s(pC0[6]);
                    __m256 _c7 = (__m256)__lasx_xvreplfr2vr_s(pC0[7]);
                    _sum0 = __lasx_xvfmadd_s(_c0, _beta, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_c1, _beta, _sum1);
                    _sum2 = __lasx_xvfmadd_s(_c2, _beta, _sum2);
                    _sum3 = __lasx_xvfmadd_s(_c3, _beta, _sum3);
                    _sum4 = __lasx_xvfmadd_s(_c4, _beta, _sum4);
                    _sum5 = __lasx_xvfmadd_s(_c5, _beta, _sum5);
                    _sum6 = __lasx_xvfmadd_s(_c6, _beta, _sum6);
                    _sum7 = __lasx_xvfmadd_s(_c7, _beta, _sum7);
                    pC0 += 8;
                }
            }

            _sum0 = __lasx_xvfmul_s(_sum0, _valpha);
            _sum1 = __lasx_xvfmul_s(_sum1, _valpha);
            _sum2 = __lasx_xvfmul_s(_sum2, _valpha);
            _sum3 = __lasx_xvfmul_s(_sum3, _valpha);
            _sum4 = __lasx_xvfmul_s(_sum4, _valpha);
            _sum5 = __lasx_xvfmul_s(_sum5, _valpha);
            _sum6 = __lasx_xvfmul_s(_sum6, _valpha);
            _sum7 = __lasx_xvfmul_s(_sum7, _valpha);

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 8)
                    {
                        transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
                        __lasx_xvst(_sum0, p0f32, 0);
                        __lasx_xvst(_sum1, p0f32 + 8, 0);
                        __lasx_xvst(_sum2, p0f32 + 16, 0);
                        __lasx_xvst(_sum3, p0f32 + 24, 0);
                        __lasx_xvst(_sum4, p0f32 + 32, 0);
                        __lasx_xvst(_sum5, p0f32 + 40, 0);
                        __lasx_xvst(_sum6, p0f32 + 48, 0);
                        __lasx_xvst(_sum7, p0f32 + 56, 0);
                    }
                    if (out_elempack == 4)
                    {
                        transpose8x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose8x4_ps(_sum4, _sum5, _sum6, _sum7);
                        __lasx_xvst(_sum0, p0f32, 0);
                        __lasx_xvst(_sum1, p0f32 + 8, 0);
                        __lasx_xvst(_sum2, p0f32 + 16, 0);
                        __lasx_xvst(_sum3, p0f32 + 24, 0);
                        __lasx_xvst(_sum4, p0f32 + out_hstep * 4, 0);
                        __lasx_xvst(_sum5, p0f32 + out_hstep * 4 + 8, 0);
                        __lasx_xvst(_sum6, p0f32 + out_hstep * 4 + 16, 0);
                        __lasx_xvst(_sum7, p0f32 + out_hstep * 4 + 24, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lasx_xvst(_sum0, p0f32, 0);
                        __lasx_xvst(_sum1, p0f32 + out_hstep, 0);
                        __lasx_xvst(_sum2, p0f32 + out_hstep * 2, 0);
                        __lasx_xvst(_sum3, p0f32 + out_hstep * 3, 0);
                        __lasx_xvst(_sum4, p0f32 + out_hstep * 4, 0);
                        __lasx_xvst(_sum5, p0f32 + out_hstep * 5, 0);
                        __lasx_xvst(_sum6, p0f32 + out_hstep * 6, 0);
                        __lasx_xvst(_sum7, p0f32 + out_hstep * 7, 0);
                    }
                    p0f32 += out_hstep * 8;
                }
                else
                {
                    if (out_elempack == 8)
                    {
                        __lasx_xvst(_sum0, p0f32, 0);
                        __lasx_xvst(_sum1, p0f32 + 8, 0);
                        __lasx_xvst(_sum2, p0f32 + 16, 0);
                        __lasx_xvst(_sum3, p0f32 + 24, 0);
                        __lasx_xvst(_sum4, p0f32 + 32, 0);
                        __lasx_xvst(_sum5, p0f32 + 40, 0);
                        __lasx_xvst(_sum6, p0f32 + 48, 0);
                        __lasx_xvst(_sum7, p0f32 + 56, 0);
                        p0f32 += 64;
                    }
                    if (out_elempack == 4)
                    {
                        float* p1f32 = p0f32 + out_hstep * 4;
                        __lsx_vst(__lasx_extract_lo128((__m256i)_sum0), p0f32, 0);
                        __lsx_vst(__lasx_extract_lo128((__m256i)_sum1), p0f32 + 4, 0);
                        __lsx_vst(__lasx_extract_lo128((__m256i)_sum2), p0f32 + 8, 0);
                        __lsx_vst(__lasx_extract_lo128((__m256i)_sum3), p0f32 + 12, 0);
                        __lsx_vst(__lasx_extract_lo128((__m256i)_sum4), p0f32 + 16, 0);
                        __lsx_vst(__lasx_extract_lo128((__m256i)_sum5), p0f32 + 20, 0);
                        __lsx_vst(__lasx_extract_lo128((__m256i)_sum6), p0f32 + 24, 0);
                        __lsx_vst(__lasx_extract_lo128((__m256i)_sum7), p0f32 + 28, 0);
                        __lsx_vst(__lasx_extract_hi128((__m256i)_sum0), p1f32, 0);
                        __lsx_vst(__lasx_extract_hi128((__m256i)_sum1), p1f32 + 4, 0);
                        __lsx_vst(__lasx_extract_hi128((__m256i)_sum2), p1f32 + 8, 0);
                        __lsx_vst(__lasx_extract_hi128((__m256i)_sum3), p1f32 + 12, 0);
                        __lsx_vst(__lasx_extract_hi128((__m256i)_sum4), p1f32 + 16, 0);
                        __lsx_vst(__lasx_extract_hi128((__m256i)_sum5), p1f32 + 20, 0);
                        __lsx_vst(__lasx_extract_hi128((__m256i)_sum6), p1f32 + 24, 0);
                        __lsx_vst(__lasx_extract_hi128((__m256i)_sum7), p1f32 + 28, 0);
                        p0f32 += 32;
                    }
                    if (out_elempack == 1)
                    {
                        transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
                        __lasx_xvst(_sum0, p0f32, 0);
                        __lasx_xvst(_sum1, p0f32 + out_hstep, 0);
                        __lasx_xvst(_sum2, p0f32 + out_hstep * 2, 0);
                        __lasx_xvst(_sum3, p0f32 + out_hstep * 3, 0);
                        __lasx_xvst(_sum4, p0f32 + out_hstep * 4, 0);
                        __lasx_xvst(_sum5, p0f32 + out_hstep * 5, 0);
                        __lasx_xvst(_sum6, p0f32 + out_hstep * 6, 0);
                        __lasx_xvst(_sum7, p0f32 + out_hstep * 7, 0);
                        p0f32 += 8;
                    }
                }
            }
            else
            {
                if (output_transpose)
                {
                    if (out_elempack == 8)
                    {
                        transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
                        __lsx_vst(float2bfloat_lasx(_sum0), p0, 0);
                        __lsx_vst(float2bfloat_lasx(_sum1), p0 + 8, 0);
                        __lsx_vst(float2bfloat_lasx(_sum2), p0 + 16, 0);
                        __lsx_vst(float2bfloat_lasx(_sum3), p0 + 24, 0);
                        __lsx_vst(float2bfloat_lasx(_sum4), p0 + 32, 0);
                        __lsx_vst(float2bfloat_lasx(_sum5), p0 + 40, 0);
                        __lsx_vst(float2bfloat_lasx(_sum6), p0 + 48, 0);
                        __lsx_vst(float2bfloat_lasx(_sum7), p0 + 56, 0);
                    }
                    if (out_elempack == 4)
                    {
                        transpose8x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose8x4_ps(_sum4, _sum5, _sum6, _sum7);
                        __lsx_vst(float2bfloat_lasx(_sum0), p0, 0);
                        __lsx_vst(float2bfloat_lasx(_sum1), p0 + 8, 0);
                        __lsx_vst(float2bfloat_lasx(_sum2), p0 + 16, 0);
                        __lsx_vst(float2bfloat_lasx(_sum3), p0 + 24, 0);
                        __lsx_vst(float2bfloat_lasx(_sum4), p0 + out_hstep * 4, 0);
                        __lsx_vst(float2bfloat_lasx(_sum5), p0 + out_hstep * 4 + 8, 0);
                        __lsx_vst(float2bfloat_lasx(_sum6), p0 + out_hstep * 4 + 16, 0);
                        __lsx_vst(float2bfloat_lasx(_sum7), p0 + out_hstep * 4 + 24, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vst(float2bfloat_lasx(_sum0), p0, 0);
                        __lsx_vst(float2bfloat_lasx(_sum1), p0 + out_hstep, 0);
                        __lsx_vst(float2bfloat_lasx(_sum2), p0 + out_hstep * 2, 0);
                        __lsx_vst(float2bfloat_lasx(_sum3), p0 + out_hstep * 3, 0);
                        __lsx_vst(float2bfloat_lasx(_sum4), p0 + out_hstep * 4, 0);
                        __lsx_vst(float2bfloat_lasx(_sum5), p0 + out_hstep * 5, 0);
                        __lsx_vst(float2bfloat_lasx(_sum6), p0 + out_hstep * 6, 0);
                        __lsx_vst(float2bfloat_lasx(_sum7), p0 + out_hstep * 7, 0);
                    }
                    p0 += out_hstep * 8;
                }
                else
                {
                    if (out_elempack == 8)
                    {
                        __lsx_vst(float2bfloat_lasx(_sum0), p0, 0);
                        __lsx_vst(float2bfloat_lasx(_sum1), p0 + 8, 0);
                        __lsx_vst(float2bfloat_lasx(_sum2), p0 + 16, 0);
                        __lsx_vst(float2bfloat_lasx(_sum3), p0 + 24, 0);
                        __lsx_vst(float2bfloat_lasx(_sum4), p0 + 32, 0);
                        __lsx_vst(float2bfloat_lasx(_sum5), p0 + 40, 0);
                        __lsx_vst(float2bfloat_lasx(_sum6), p0 + 48, 0);
                        __lsx_vst(float2bfloat_lasx(_sum7), p0 + 56, 0);
                        p0 += 64;
                    }
                    if (out_elempack == 4)
                    {
                        unsigned short* p1 = p0 + out_hstep * 4;
                        __m128i _bf0 = float2bfloat_lasx(_sum0);
                        __m128i _bf1 = float2bfloat_lasx(_sum1);
                        __m128i _bf2 = float2bfloat_lasx(_sum2);
                        __m128i _bf3 = float2bfloat_lasx(_sum3);
                        __m128i _bf4 = float2bfloat_lasx(_sum4);
                        __m128i _bf5 = float2bfloat_lasx(_sum5);
                        __m128i _bf6 = float2bfloat_lasx(_sum6);
                        __m128i _bf7 = float2bfloat_lasx(_sum7);
                        __lsx_vstelm_d(_bf0, p0, 0, 0);
                        __lsx_vstelm_d(_bf1, p0 + 4, 0, 0);
                        __lsx_vstelm_d(_bf2, p0 + 8, 0, 0);
                        __lsx_vstelm_d(_bf3, p0 + 12, 0, 0);
                        __lsx_vstelm_d(_bf4, p0 + 16, 0, 0);
                        __lsx_vstelm_d(_bf5, p0 + 20, 0, 0);
                        __lsx_vstelm_d(_bf6, p0 + 24, 0, 0);
                        __lsx_vstelm_d(_bf7, p0 + 28, 0, 0);
                        __lsx_vstelm_d(_bf0, p1, 0, 1);
                        __lsx_vstelm_d(_bf1, p1 + 4, 0, 1);
                        __lsx_vstelm_d(_bf2, p1 + 8, 0, 1);
                        __lsx_vstelm_d(_bf3, p1 + 12, 0, 1);
                        __lsx_vstelm_d(_bf4, p1 + 16, 0, 1);
                        __lsx_vstelm_d(_bf5, p1 + 20, 0, 1);
                        __lsx_vstelm_d(_bf6, p1 + 24, 0, 1);
                        __lsx_vstelm_d(_bf7, p1 + 28, 0, 1);
                        p0 += 32;
                    }
                    if (out_elempack == 1)
                    {
                        transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
                        __lsx_vst(float2bfloat_lasx(_sum0), p0, 0);
                        __lsx_vst(float2bfloat_lasx(_sum1), p0 + out_hstep, 0);
                        __lsx_vst(float2bfloat_lasx(_sum2), p0 + out_hstep * 2, 0);
                        __lsx_vst(float2bfloat_lasx(_sum3), p0 + out_hstep * 3, 0);
                        __lsx_vst(float2bfloat_lasx(_sum4), p0 + out_hstep * 4, 0);
                        __lsx_vst(float2bfloat_lasx(_sum5), p0 + out_hstep * 5, 0);
                        __lsx_vst(float2bfloat_lasx(_sum6), p0 + out_hstep * 6, 0);
                        __lsx_vst(float2bfloat_lasx(_sum7), p0 + out_hstep * 7, 0);
                        p0 += 8;
                    }
                }
            }
        }
        for (; jj < max_jj; jj++)
        {
            __m256 _sum = (__m256)__lasx_xvld(pp, 0);
            pp += 8;

            if (pC0)
            {
                __m256 _beta_v = (__m256)__lasx_xvreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pC0[0]);
                    _sum = __lasx_xvfmadd_s(_c, _beta_v, _sum);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m256 _c = (__m256)__lasx_xvld(pC0, 0);
                    _sum = __lasx_xvfmadd_s(_c, _beta_v, _sum);
                }
                if (broadcast_type_C == 3)
                {
                    // Load 8 values from C matrix for this (ii, jj) position
                    if (c_elempack == 8)
                    {
                        __m256 _c = (__m256)__lasx_xvld(pC0, 0);
                        _sum = __lasx_xvfmadd_s(_c, _beta_v, _sum);
                        pC0 += 8;
                    }
                    else if (c_elempack == 4)
                    {
                        __m128 _cl = (__m128)__lsx_vld(pC0, 0);
                        __m128 _ch = (__m128)__lsx_vld(pC0 + c_hstep * 4, 0);
                        __m256 _c = combine4x2_ps(_cl, _ch);
                        _sum = __lasx_xvfmadd_s(_c, _beta_v, _sum);
                        pC0 += 4;
                    }
                    else // c_elempack == 1
                    {
                        float ctmp[8];
                        for (int r = 0; r < 8; r++)
                            ctmp[r] = pC0[c_hstep * r];
                        __m256 _c = (__m256)__lasx_xvld(ctmp, 0);
                        _sum = __lasx_xvfmadd_s(_c, _beta_v, _sum);
                        pC0 += 1;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pC0[0]);
                    _sum = __lasx_xvfmadd_s(_c, _beta_v, _sum);
                    pC0 += 1;
                }
            }

            _sum = __lasx_xvfmul_s(_sum, _valpha);

            float tmp[8];
            __lasx_xvst((__m256i)_sum, tmp, 0);

            if (output_transpose)
            {
                const int col = j + jj;
                if (out_elempack == 8)
                {
                    const int lane = col % 8;
                    unsigned short* p0t = (unsigned short*)top_blob + (size_t)(col / 8 * 8) * out_hstep + (i + ii) * 8;
                    float* p0f32t = (float*)top_blob + (size_t)(col / 8 * 8) * out_hstep + (i + ii) * 8;

                    if (output_elemtype == 1)
                    {
                        p0f32t[lane] = tmp[0];
                        p0f32t[8 + lane] = tmp[1];
                        p0f32t[16 + lane] = tmp[2];
                        p0f32t[24 + lane] = tmp[3];
                        p0f32t[32 + lane] = tmp[4];
                        p0f32t[40 + lane] = tmp[5];
                        p0f32t[48 + lane] = tmp[6];
                        p0f32t[56 + lane] = tmp[7];
                    }
                    else
                    {
                        p0t[lane] = float32_to_bfloat16(tmp[0]);
                        p0t[8 + lane] = float32_to_bfloat16(tmp[1]);
                        p0t[16 + lane] = float32_to_bfloat16(tmp[2]);
                        p0t[24 + lane] = float32_to_bfloat16(tmp[3]);
                        p0t[32 + lane] = float32_to_bfloat16(tmp[4]);
                        p0t[40 + lane] = float32_to_bfloat16(tmp[5]);
                        p0t[48 + lane] = float32_to_bfloat16(tmp[6]);
                        p0t[56 + lane] = float32_to_bfloat16(tmp[7]);
                    }
                }
                if (out_elempack == 4)
                {
                    const int lane = col % 4;
                    unsigned short* p0t = (unsigned short*)top_blob + (size_t)(col / 4 * 4) * out_hstep + (i + ii) * 4;
                    float* p0f32t = (float*)top_blob + (size_t)(col / 4 * 4) * out_hstep + (i + ii) * 4;

                    if (output_elemtype == 1)
                    {
                        p0f32t[lane] = tmp[0];
                        p0f32t[4 + lane] = tmp[1];
                        p0f32t[8 + lane] = tmp[2];
                        p0f32t[12 + lane] = tmp[3];
                        p0f32t[16 + lane] = tmp[4];
                        p0f32t[20 + lane] = tmp[5];
                        p0f32t[24 + lane] = tmp[6];
                        p0f32t[28 + lane] = tmp[7];
                    }
                    else
                    {
                        p0t[lane] = float32_to_bfloat16(tmp[0]);
                        p0t[4 + lane] = float32_to_bfloat16(tmp[1]);
                        p0t[8 + lane] = float32_to_bfloat16(tmp[2]);
                        p0t[12 + lane] = float32_to_bfloat16(tmp[3]);
                        p0t[16 + lane] = float32_to_bfloat16(tmp[4]);
                        p0t[20 + lane] = float32_to_bfloat16(tmp[5]);
                        p0t[24 + lane] = float32_to_bfloat16(tmp[6]);
                        p0t[28 + lane] = float32_to_bfloat16(tmp[7]);
                    }
                }
                if (out_elempack == 1)
                {
                    unsigned short* p0t = (unsigned short*)top_blob + col * out_hstep + (i + ii);
                    float* p0f32t = (float*)top_blob + col * out_hstep + (i + ii);

                    if (output_elemtype == 1)
                    {
                        __lasx_xvst(_sum, p0f32t, 0);
                    }
                    else
                    {
                        __lsx_vst(float2bfloat_lasx(_sum), p0t, 0);
                    }
                }
            }
            else
            {
                if (out_elempack == 8)
                {
                    if (output_elemtype == 1)
                    {
                        __lasx_xvst(_sum, p0f32, 0);
                        p0f32 += 8;
                    }
                    else
                    {
                        __lsx_vst(float2bfloat_lasx(_sum), p0, 0);
                        p0 += 8;
                    }
                }
                if (out_elempack == 4)
                {
                    if (output_elemtype == 1)
                    {
                        __lsx_vst(__lasx_extract_lo128((__m256i)_sum), p0f32, 0);
                        __lsx_vst(__lasx_extract_hi128((__m256i)_sum), p0f32 + out_hstep * 4, 0);
                        p0f32 += 4;
                    }
                    else
                    {
                        __m128i _bf = float2bfloat_lasx(_sum);
                        __lsx_vstelm_d(_bf, p0, 0, 0);
                        __lsx_vstelm_d(_bf, p0 + out_hstep * 4, 0, 1);
                        p0 += 4;
                    }
                }
                if (out_elempack == 1)
                {
                    if (output_elemtype == 1)
                    {
                        p0f32[0] = tmp[0];
                        p0f32[out_hstep] = tmp[1];
                        p0f32[out_hstep * 2] = tmp[2];
                        p0f32[out_hstep * 3] = tmp[3];
                        p0f32[out_hstep * 4] = tmp[4];
                        p0f32[out_hstep * 5] = tmp[5];
                        p0f32[out_hstep * 6] = tmp[6];
                        p0f32[out_hstep * 7] = tmp[7];
                        p0f32++;
                    }
                    else
                    {
                        p0[0] = float32_to_bfloat16(tmp[0]);
                        p0[out_hstep] = float32_to_bfloat16(tmp[1]);
                        p0[out_hstep * 2] = float32_to_bfloat16(tmp[2]);
                        p0[out_hstep * 3] = float32_to_bfloat16(tmp[3]);
                        p0[out_hstep * 4] = float32_to_bfloat16(tmp[4]);
                        p0[out_hstep * 5] = float32_to_bfloat16(tmp[5]);
                        p0[out_hstep * 6] = float32_to_bfloat16(tmp[6]);
                        p0[out_hstep * 7] = float32_to_bfloat16(tmp[7]);
                        p0++;
                    }
                }
            }
        }
    }
#endif // __loongarch_asx
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
            p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
            p0f32 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;
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

        __m128 _valpha = __lsx_vreplfr2vr_s(alpha);

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _sum00 = (__m128)__lsx_vld(pp, 0);
            __m128 _sum01 = (__m128)__lsx_vld(pp + 4, 0);
            __m128 _sum10 = (__m128)__lsx_vld(pp + 8, 0);
            __m128 _sum11 = (__m128)__lsx_vld(pp + 12, 0);
            __m128 _sum20 = (__m128)__lsx_vld(pp + 16, 0);
            __m128 _sum21 = (__m128)__lsx_vld(pp + 20, 0);
            __m128 _sum30 = (__m128)__lsx_vld(pp + 24, 0);
            __m128 _sum31 = (__m128)__lsx_vld(pp + 28, 0);
            __m128 _sum40 = (__m128)__lsx_vld(pp + 32, 0);
            __m128 _sum41 = (__m128)__lsx_vld(pp + 36, 0);
            __m128 _sum50 = (__m128)__lsx_vld(pp + 40, 0);
            __m128 _sum51 = (__m128)__lsx_vld(pp + 44, 0);
            __m128 _sum60 = (__m128)__lsx_vld(pp + 48, 0);
            __m128 _sum61 = (__m128)__lsx_vld(pp + 52, 0);
            __m128 _sum70 = (__m128)__lsx_vld(pp + 56, 0);
            __m128 _sum71 = (__m128)__lsx_vld(pp + 60, 0);
            pp += 64;

            if (pC0)
            {
                __m128 _beta = __lsx_vreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    __m128 _c = __lsx_vfmul_s(__lsx_vreplfr2vr_s(pC0[0]), _beta);
                    _sum00 = __lsx_vfadd_s(_sum00, _c);
                    _sum01 = __lsx_vfadd_s(_sum01, _c);
                    _sum10 = __lsx_vfadd_s(_sum10, _c);
                    _sum11 = __lsx_vfadd_s(_sum11, _c);
                    _sum20 = __lsx_vfadd_s(_sum20, _c);
                    _sum21 = __lsx_vfadd_s(_sum21, _c);
                    _sum30 = __lsx_vfadd_s(_sum30, _c);
                    _sum31 = __lsx_vfadd_s(_sum31, _c);
                    _sum40 = __lsx_vfadd_s(_sum40, _c);
                    _sum41 = __lsx_vfadd_s(_sum41, _c);
                    _sum50 = __lsx_vfadd_s(_sum50, _c);
                    _sum51 = __lsx_vfadd_s(_sum51, _c);
                    _sum60 = __lsx_vfadd_s(_sum60, _c);
                    _sum61 = __lsx_vfadd_s(_sum61, _c);
                    _sum70 = __lsx_vfadd_s(_sum70, _c);
                    _sum71 = __lsx_vfadd_s(_sum71, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c0 = __lsx_vfmul_s((__m128)__lsx_vld(pC0, 0), _beta);
                    __m128 _c1 = __lsx_vfmul_s((__m128)__lsx_vld(pC0 + 4, 0), _beta);
                    _sum00 = __lsx_vfadd_s(_sum00, _c0);
                    _sum01 = __lsx_vfadd_s(_sum01, _c1);
                    _sum10 = __lsx_vfadd_s(_sum10, _c0);
                    _sum11 = __lsx_vfadd_s(_sum11, _c1);
                    _sum20 = __lsx_vfadd_s(_sum20, _c0);
                    _sum21 = __lsx_vfadd_s(_sum21, _c1);
                    _sum30 = __lsx_vfadd_s(_sum30, _c0);
                    _sum31 = __lsx_vfadd_s(_sum31, _c1);
                    _sum40 = __lsx_vfadd_s(_sum40, _c0);
                    _sum41 = __lsx_vfadd_s(_sum41, _c1);
                    _sum50 = __lsx_vfadd_s(_sum50, _c0);
                    _sum51 = __lsx_vfadd_s(_sum51, _c1);
                    _sum60 = __lsx_vfadd_s(_sum60, _c0);
                    _sum61 = __lsx_vfadd_s(_sum61, _c1);
                    _sum70 = __lsx_vfadd_s(_sum70, _c0);
                    _sum71 = __lsx_vfadd_s(_sum71, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        const float* pCj0 = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + (j + jj) * 4;
                        const float* pCj1 = pCj0 + c_hstep * 4;
                        _sum00 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj0, 0), _sum00);
                        _sum01 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj1, 0), _sum01);
                        _sum10 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj0 + 4, 0), _sum10);
                        _sum11 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj1 + 4, 0), _sum11);
                        _sum20 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj0 + 8, 0), _sum20);
                        _sum21 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj1 + 8, 0), _sum21);
                        _sum30 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj0 + 12, 0), _sum30);
                        _sum31 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj1 + 12, 0), _sum31);
                        _sum40 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj0 + 16, 0), _sum40);
                        _sum41 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj1 + 16, 0), _sum41);
                        _sum50 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj0 + 20, 0), _sum50);
                        _sum51 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj1 + 20, 0), _sum51);
                        _sum60 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj0 + 24, 0), _sum60);
                        _sum61 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj1 + 24, 0), _sum61);
                        _sum70 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj0 + 28, 0), _sum70);
                        _sum71 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj1 + 28, 0), _sum71);
                    }
                    if (c_elempack == 1)
                    {
                        const float* pCj = (const float*)C + (i + ii) * c_hstep + (j + jj);
                        __m128 _c0 = (__m128)__lsx_vld(pCj, 0);
                        __m128 _c1 = (__m128)__lsx_vld(pCj + c_hstep, 0);
                        __m128 _c2 = (__m128)__lsx_vld(pCj + c_hstep * 2, 0);
                        __m128 _c3 = (__m128)__lsx_vld(pCj + c_hstep * 3, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _sum00 = __lsx_vfmadd_s(_beta, _c0, _sum00);
                        _sum10 = __lsx_vfmadd_s(_beta, _c1, _sum10);
                        _sum20 = __lsx_vfmadd_s(_beta, _c2, _sum20);
                        _sum30 = __lsx_vfmadd_s(_beta, _c3, _sum30);

                        _c0 = (__m128)__lsx_vld(pCj + c_hstep * 4, 0);
                        _c1 = (__m128)__lsx_vld(pCj + c_hstep * 5, 0);
                        _c2 = (__m128)__lsx_vld(pCj + c_hstep * 6, 0);
                        _c3 = (__m128)__lsx_vld(pCj + c_hstep * 7, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _sum01 = __lsx_vfmadd_s(_beta, _c0, _sum01);
                        _sum11 = __lsx_vfmadd_s(_beta, _c1, _sum11);
                        _sum21 = __lsx_vfmadd_s(_beta, _c2, _sum21);
                        _sum31 = __lsx_vfmadd_s(_beta, _c3, _sum31);

                        pCj += 4;

                        _c0 = (__m128)__lsx_vld(pCj, 0);
                        _c1 = (__m128)__lsx_vld(pCj + c_hstep, 0);
                        _c2 = (__m128)__lsx_vld(pCj + c_hstep * 2, 0);
                        _c3 = (__m128)__lsx_vld(pCj + c_hstep * 3, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _sum40 = __lsx_vfmadd_s(_beta, _c0, _sum40);
                        _sum50 = __lsx_vfmadd_s(_beta, _c1, _sum50);
                        _sum60 = __lsx_vfmadd_s(_beta, _c2, _sum60);
                        _sum70 = __lsx_vfmadd_s(_beta, _c3, _sum70);

                        _c0 = (__m128)__lsx_vld(pCj + c_hstep * 4, 0);
                        _c1 = (__m128)__lsx_vld(pCj + c_hstep * 5, 0);
                        _c2 = (__m128)__lsx_vld(pCj + c_hstep * 6, 0);
                        _c3 = (__m128)__lsx_vld(pCj + c_hstep * 7, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _sum41 = __lsx_vfmadd_s(_beta, _c0, _sum41);
                        _sum51 = __lsx_vfmadd_s(_beta, _c1, _sum51);
                        _sum61 = __lsx_vfmadd_s(_beta, _c2, _sum61);
                        _sum71 = __lsx_vfmadd_s(_beta, _c3, _sum71);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(pC0[0]);
                    __m128 _c1 = __lsx_vreplfr2vr_s(pC0[1]);
                    __m128 _c2 = __lsx_vreplfr2vr_s(pC0[2]);
                    __m128 _c3 = __lsx_vreplfr2vr_s(pC0[3]);
                    _sum00 = __lsx_vfmadd_s(_beta, _c0, _sum00);
                    _sum01 = __lsx_vfmadd_s(_beta, _c0, _sum01);
                    _sum10 = __lsx_vfmadd_s(_beta, _c1, _sum10);
                    _sum11 = __lsx_vfmadd_s(_beta, _c1, _sum11);
                    _sum20 = __lsx_vfmadd_s(_beta, _c2, _sum20);
                    _sum21 = __lsx_vfmadd_s(_beta, _c2, _sum21);
                    _sum30 = __lsx_vfmadd_s(_beta, _c3, _sum30);
                    _sum31 = __lsx_vfmadd_s(_beta, _c3, _sum31);
                    _c0 = __lsx_vreplfr2vr_s(pC0[4]);
                    _c1 = __lsx_vreplfr2vr_s(pC0[5]);
                    _c2 = __lsx_vreplfr2vr_s(pC0[6]);
                    _c3 = __lsx_vreplfr2vr_s(pC0[7]);
                    _sum40 = __lsx_vfmadd_s(_beta, _c0, _sum40);
                    _sum41 = __lsx_vfmadd_s(_beta, _c0, _sum41);
                    _sum50 = __lsx_vfmadd_s(_beta, _c1, _sum50);
                    _sum51 = __lsx_vfmadd_s(_beta, _c1, _sum51);
                    _sum60 = __lsx_vfmadd_s(_beta, _c2, _sum60);
                    _sum61 = __lsx_vfmadd_s(_beta, _c2, _sum61);
                    _sum70 = __lsx_vfmadd_s(_beta, _c3, _sum70);
                    _sum71 = __lsx_vfmadd_s(_beta, _c3, _sum71);
                    pC0 += 8;
                }
            }

            _sum00 = __lsx_vfmul_s(_sum00, _valpha);
            _sum01 = __lsx_vfmul_s(_sum01, _valpha);
            _sum10 = __lsx_vfmul_s(_sum10, _valpha);
            _sum11 = __lsx_vfmul_s(_sum11, _valpha);
            _sum20 = __lsx_vfmul_s(_sum20, _valpha);
            _sum21 = __lsx_vfmul_s(_sum21, _valpha);
            _sum30 = __lsx_vfmul_s(_sum30, _valpha);
            _sum31 = __lsx_vfmul_s(_sum31, _valpha);
            _sum40 = __lsx_vfmul_s(_sum40, _valpha);
            _sum41 = __lsx_vfmul_s(_sum41, _valpha);
            _sum50 = __lsx_vfmul_s(_sum50, _valpha);
            _sum51 = __lsx_vfmul_s(_sum51, _valpha);
            _sum60 = __lsx_vfmul_s(_sum60, _valpha);
            _sum61 = __lsx_vfmul_s(_sum61, _valpha);
            _sum70 = __lsx_vfmul_s(_sum70, _valpha);
            _sum71 = __lsx_vfmul_s(_sum71, _valpha);

            if (output_transpose)
            {
                if (output_elemtype == 1)
                {
                    if (out_elempack == 4)
                    {
                        float* p1f32 = p0f32 + out_hstep * 4;

                        transpose4x4_ps(_sum00, _sum10, _sum20, _sum30);
                        transpose4x4_ps(_sum40, _sum50, _sum60, _sum70);
                        transpose4x4_ps(_sum01, _sum11, _sum21, _sum31);
                        transpose4x4_ps(_sum41, _sum51, _sum61, _sum71);

                        __lsx_vst((__m128i)_sum00, p0f32, 0);
                        __lsx_vst((__m128i)_sum10, p0f32 + 4, 0);
                        __lsx_vst((__m128i)_sum20, p0f32 + 8, 0);
                        __lsx_vst((__m128i)_sum30, p0f32 + 12, 0);
                        __lsx_vst((__m128i)_sum01, p0f32 + 16, 0);
                        __lsx_vst((__m128i)_sum11, p0f32 + 20, 0);
                        __lsx_vst((__m128i)_sum21, p0f32 + 24, 0);
                        __lsx_vst((__m128i)_sum31, p0f32 + 28, 0);

                        __lsx_vst((__m128i)_sum40, p1f32, 0);
                        __lsx_vst((__m128i)_sum50, p1f32 + 4, 0);
                        __lsx_vst((__m128i)_sum60, p1f32 + 8, 0);
                        __lsx_vst((__m128i)_sum70, p1f32 + 12, 0);
                        __lsx_vst((__m128i)_sum41, p1f32 + 16, 0);
                        __lsx_vst((__m128i)_sum51, p1f32 + 20, 0);
                        __lsx_vst((__m128i)_sum61, p1f32 + 24, 0);
                        __lsx_vst((__m128i)_sum71, p1f32 + 28, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vst((__m128i)_sum00, p0f32, 0);
                        __lsx_vst((__m128i)_sum01, p0f32 + 4, 0);
                        __lsx_vst((__m128i)_sum10, p0f32 + out_hstep, 0);
                        __lsx_vst((__m128i)_sum11, p0f32 + out_hstep + 4, 0);
                        __lsx_vst((__m128i)_sum20, p0f32 + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_sum21, p0f32 + out_hstep * 2 + 4, 0);
                        __lsx_vst((__m128i)_sum30, p0f32 + out_hstep * 3, 0);
                        __lsx_vst((__m128i)_sum31, p0f32 + out_hstep * 3 + 4, 0);
                        __lsx_vst((__m128i)_sum40, p0f32 + out_hstep * 4, 0);
                        __lsx_vst((__m128i)_sum41, p0f32 + out_hstep * 4 + 4, 0);
                        __lsx_vst((__m128i)_sum50, p0f32 + out_hstep * 5, 0);
                        __lsx_vst((__m128i)_sum51, p0f32 + out_hstep * 5 + 4, 0);
                        __lsx_vst((__m128i)_sum60, p0f32 + out_hstep * 6, 0);
                        __lsx_vst((__m128i)_sum61, p0f32 + out_hstep * 6 + 4, 0);
                        __lsx_vst((__m128i)_sum70, p0f32 + out_hstep * 7, 0);
                        __lsx_vst((__m128i)_sum71, p0f32 + out_hstep * 7 + 4, 0);
                    }
                    p0f32 += out_hstep * 8;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        unsigned short* p1 = p0 + out_hstep * 4;

                        transpose4x4_ps(_sum00, _sum10, _sum20, _sum30);
                        transpose4x4_ps(_sum40, _sum50, _sum60, _sum70);
                        transpose4x4_ps(_sum01, _sum11, _sum21, _sum31);
                        transpose4x4_ps(_sum41, _sum51, _sum61, _sum71);

                        __lsx_vstelm_d(float2bfloat_lsx(_sum00), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum10), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum20), p0 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum30), p0 + 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum01), p0 + 16, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum11), p0 + 20, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum21), p0 + 24, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum31), p0 + 28, 0, 0);

                        __lsx_vstelm_d(float2bfloat_lsx(_sum40), p1, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum50), p1 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum60), p1 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum70), p1 + 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum41), p1 + 16, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum51), p1 + 20, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum61), p1 + 24, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum71), p1 + 28, 0, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vstelm_d(float2bfloat_lsx(_sum00), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum01), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum10), p0 + out_hstep, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum11), p0 + out_hstep + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum20), p0 + out_hstep * 2, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum21), p0 + out_hstep * 2 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum30), p0 + out_hstep * 3, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum31), p0 + out_hstep * 3 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum40), p0 + out_hstep * 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum41), p0 + out_hstep * 4 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum50), p0 + out_hstep * 5, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum51), p0 + out_hstep * 5 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum60), p0 + out_hstep * 6, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum61), p0 + out_hstep * 6 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum70), p0 + out_hstep * 7, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum71), p0 + out_hstep * 7 + 4, 0, 0);
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
                        float* p1f32 = p0f32 + out_hstep * 4;
                        __lsx_vst((__m128i)_sum00, p0f32, 0);
                        __lsx_vst((__m128i)_sum01, p1f32, 0);
                        __lsx_vst((__m128i)_sum10, p0f32 + 4, 0);
                        __lsx_vst((__m128i)_sum11, p1f32 + 4, 0);
                        __lsx_vst((__m128i)_sum20, p0f32 + 8, 0);
                        __lsx_vst((__m128i)_sum21, p1f32 + 8, 0);
                        __lsx_vst((__m128i)_sum30, p0f32 + 12, 0);
                        __lsx_vst((__m128i)_sum31, p1f32 + 12, 0);
                        __lsx_vst((__m128i)_sum40, p0f32 + 16, 0);
                        __lsx_vst((__m128i)_sum41, p1f32 + 16, 0);
                        __lsx_vst((__m128i)_sum50, p0f32 + 20, 0);
                        __lsx_vst((__m128i)_sum51, p1f32 + 20, 0);
                        __lsx_vst((__m128i)_sum60, p0f32 + 24, 0);
                        __lsx_vst((__m128i)_sum61, p1f32 + 24, 0);
                        __lsx_vst((__m128i)_sum70, p0f32 + 28, 0);
                        __lsx_vst((__m128i)_sum71, p1f32 + 28, 0);
                        p0f32 += 32;
                    }
                    else
                    {
                        unsigned short* p1 = p0 + out_hstep * 4;
                        __lsx_vstelm_d(float2bfloat_lsx(_sum00), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum01), p1, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum10), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum11), p1 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum20), p0 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum21), p1 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum30), p0 + 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum31), p1 + 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum40), p0 + 16, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum41), p1 + 16, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum50), p0 + 20, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum51), p1 + 20, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum60), p0 + 24, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum61), p1 + 24, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum70), p0 + 28, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum71), p1 + 28, 0, 0);
                        p0 += 32;
                    }
                }
                if (out_elempack == 1)
                {
                    __m128 _r0 = _sum00;
                    __m128 _r1 = _sum10;
                    __m128 _r2 = _sum20;
                    __m128 _r3 = _sum30;
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __m128 _r4 = _sum40;
                    __m128 _r5 = _sum50;
                    __m128 _r6 = _sum60;
                    __m128 _r7 = _sum70;
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    __m128 _r8 = _sum01;
                    __m128 _r9 = _sum11;
                    __m128 _ra = _sum21;
                    __m128 _rb = _sum31;
                    transpose4x4_ps(_r8, _r9, _ra, _rb);
                    __m128 _rc = _sum41;
                    __m128 _rd = _sum51;
                    __m128 _re = _sum61;
                    __m128 _rf = _sum71;
                    transpose4x4_ps(_rc, _rd, _re, _rf);

                    if (output_elemtype == 1)
                    {
                        __lsx_vst((__m128i)_r0, p0f32, 0);
                        __lsx_vst((__m128i)_r4, p0f32 + 4, 0);
                        __lsx_vst((__m128i)_r1, p0f32 + out_hstep, 0);
                        __lsx_vst((__m128i)_r5, p0f32 + out_hstep + 4, 0);
                        __lsx_vst((__m128i)_r2, p0f32 + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_r6, p0f32 + out_hstep * 2 + 4, 0);
                        __lsx_vst((__m128i)_r3, p0f32 + out_hstep * 3, 0);
                        __lsx_vst((__m128i)_r7, p0f32 + out_hstep * 3 + 4, 0);
                        __lsx_vst((__m128i)_r8, p0f32 + out_hstep * 4, 0);
                        __lsx_vst((__m128i)_rc, p0f32 + out_hstep * 4 + 4, 0);
                        __lsx_vst((__m128i)_r9, p0f32 + out_hstep * 5, 0);
                        __lsx_vst((__m128i)_rd, p0f32 + out_hstep * 5 + 4, 0);
                        __lsx_vst((__m128i)_ra, p0f32 + out_hstep * 6, 0);
                        __lsx_vst((__m128i)_re, p0f32 + out_hstep * 6 + 4, 0);
                        __lsx_vst((__m128i)_rb, p0f32 + out_hstep * 7, 0);
                        __lsx_vst((__m128i)_rf, p0f32 + out_hstep * 7 + 4, 0);
                        p0f32 += 8;
                    }
                    else
                    {
                        __lsx_vstelm_d(float2bfloat_lsx(_r0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r4), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r1), p0 + out_hstep, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r5), p0 + out_hstep + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r2), p0 + out_hstep * 2, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r6), p0 + out_hstep * 2 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r3), p0 + out_hstep * 3, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r7), p0 + out_hstep * 3 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r8), p0 + out_hstep * 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_rc), p0 + out_hstep * 4 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r9), p0 + out_hstep * 5, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_rd), p0 + out_hstep * 5 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_ra), p0 + out_hstep * 6, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_re), p0 + out_hstep * 6 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_rb), p0 + out_hstep * 7, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_rf), p0 + out_hstep * 7 + 4, 0, 0);
                        p0 += 8;
                    }
                }
            }
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _sum00 = (__m128)__lsx_vld(pp, 0);
            __m128 _sum01 = (__m128)__lsx_vld(pp + 4, 0);
            __m128 _sum10 = (__m128)__lsx_vld(pp + 8, 0);
            __m128 _sum11 = (__m128)__lsx_vld(pp + 12, 0);
            __m128 _sum20 = (__m128)__lsx_vld(pp + 16, 0);
            __m128 _sum21 = (__m128)__lsx_vld(pp + 20, 0);
            __m128 _sum30 = (__m128)__lsx_vld(pp + 24, 0);
            __m128 _sum31 = (__m128)__lsx_vld(pp + 28, 0);
            pp += 32;

            if (pC0)
            {
                __m128 _beta = __lsx_vreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    __m128 _c = __lsx_vfmul_s(__lsx_vreplfr2vr_s(pC0[0]), _beta);
                    _sum00 = __lsx_vfadd_s(_sum00, _c);
                    _sum01 = __lsx_vfadd_s(_sum01, _c);
                    _sum10 = __lsx_vfadd_s(_sum10, _c);
                    _sum11 = __lsx_vfadd_s(_sum11, _c);
                    _sum20 = __lsx_vfadd_s(_sum20, _c);
                    _sum21 = __lsx_vfadd_s(_sum21, _c);
                    _sum30 = __lsx_vfadd_s(_sum30, _c);
                    _sum31 = __lsx_vfadd_s(_sum31, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c0 = __lsx_vfmul_s((__m128)__lsx_vld(pC0, 0), _beta);
                    __m128 _c1 = __lsx_vfmul_s((__m128)__lsx_vld(pC0 + 4, 0), _beta);
                    _sum00 = __lsx_vfadd_s(_sum00, _c0);
                    _sum01 = __lsx_vfadd_s(_sum01, _c1);
                    _sum10 = __lsx_vfadd_s(_sum10, _c0);
                    _sum11 = __lsx_vfadd_s(_sum11, _c1);
                    _sum20 = __lsx_vfadd_s(_sum20, _c0);
                    _sum21 = __lsx_vfadd_s(_sum21, _c1);
                    _sum30 = __lsx_vfadd_s(_sum30, _c0);
                    _sum31 = __lsx_vfadd_s(_sum31, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        const float* pCj0 = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + (j + jj) * 4;
                        const float* pCj1 = pCj0 + c_hstep * 4;
                        _sum00 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj0, 0), _sum00);
                        _sum01 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj1, 0), _sum01);
                        _sum10 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj0 + 4, 0), _sum10);
                        _sum11 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj1 + 4, 0), _sum11);
                        _sum20 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj0 + 8, 0), _sum20);
                        _sum21 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj1 + 8, 0), _sum21);
                        _sum30 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj0 + 12, 0), _sum30);
                        _sum31 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj1 + 12, 0), _sum31);
                    }
                    if (c_elempack == 1)
                    {
                        const float* pCj = (const float*)C + (i + ii) * c_hstep + (j + jj);
                        __m128 _c0 = (__m128)__lsx_vld(pCj, 0);
                        __m128 _c1 = (__m128)__lsx_vld(pCj + c_hstep, 0);
                        __m128 _c2 = (__m128)__lsx_vld(pCj + c_hstep * 2, 0);
                        __m128 _c3 = (__m128)__lsx_vld(pCj + c_hstep * 3, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _sum00 = __lsx_vfmadd_s(_beta, _c0, _sum00);
                        _sum10 = __lsx_vfmadd_s(_beta, _c1, _sum10);
                        _sum20 = __lsx_vfmadd_s(_beta, _c2, _sum20);
                        _sum30 = __lsx_vfmadd_s(_beta, _c3, _sum30);

                        _c0 = (__m128)__lsx_vld(pCj + c_hstep * 4, 0);
                        _c1 = (__m128)__lsx_vld(pCj + c_hstep * 5, 0);
                        _c2 = (__m128)__lsx_vld(pCj + c_hstep * 6, 0);
                        _c3 = (__m128)__lsx_vld(pCj + c_hstep * 7, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _sum01 = __lsx_vfmadd_s(_beta, _c0, _sum01);
                        _sum11 = __lsx_vfmadd_s(_beta, _c1, _sum11);
                        _sum21 = __lsx_vfmadd_s(_beta, _c2, _sum21);
                        _sum31 = __lsx_vfmadd_s(_beta, _c3, _sum31);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(pC0[0]);
                    __m128 _c1 = __lsx_vreplfr2vr_s(pC0[1]);
                    __m128 _c2 = __lsx_vreplfr2vr_s(pC0[2]);
                    __m128 _c3 = __lsx_vreplfr2vr_s(pC0[3]);
                    _sum00 = __lsx_vfmadd_s(_beta, _c0, _sum00);
                    _sum01 = __lsx_vfmadd_s(_beta, _c0, _sum01);
                    _sum10 = __lsx_vfmadd_s(_beta, _c1, _sum10);
                    _sum11 = __lsx_vfmadd_s(_beta, _c1, _sum11);
                    _sum20 = __lsx_vfmadd_s(_beta, _c2, _sum20);
                    _sum21 = __lsx_vfmadd_s(_beta, _c2, _sum21);
                    _sum30 = __lsx_vfmadd_s(_beta, _c3, _sum30);
                    _sum31 = __lsx_vfmadd_s(_beta, _c3, _sum31);
                    pC0 += 4;
                }
            }

            _sum00 = __lsx_vfmul_s(_sum00, _valpha);
            _sum01 = __lsx_vfmul_s(_sum01, _valpha);
            _sum10 = __lsx_vfmul_s(_sum10, _valpha);
            _sum11 = __lsx_vfmul_s(_sum11, _valpha);
            _sum20 = __lsx_vfmul_s(_sum20, _valpha);
            _sum21 = __lsx_vfmul_s(_sum21, _valpha);
            _sum30 = __lsx_vfmul_s(_sum30, _valpha);
            _sum31 = __lsx_vfmul_s(_sum31, _valpha);

            if (output_transpose)
            {
                unsigned short* p0t = (unsigned short*)top_blob + (j + jj) * out_hstep + (i + ii) * out_elempack;
                float* p0f32t = (float*)top_blob + (j + jj) * out_hstep + (i + ii) * out_elempack;

                if (output_elemtype == 1)
                {
                    if (out_elempack == 4)
                    {
                        transpose4x4_ps(_sum00, _sum10, _sum20, _sum30);
                        transpose4x4_ps(_sum01, _sum11, _sum21, _sum31);

                        __lsx_vst((__m128i)_sum00, p0f32t, 0);
                        __lsx_vst((__m128i)_sum10, p0f32t + 4, 0);
                        __lsx_vst((__m128i)_sum20, p0f32t + 4 * 2, 0);
                        __lsx_vst((__m128i)_sum30, p0f32t + 4 * 3, 0);
                        __lsx_vst((__m128i)_sum01, p0f32t + 4 * 4, 0);
                        __lsx_vst((__m128i)_sum11, p0f32t + 4 * 5, 0);
                        __lsx_vst((__m128i)_sum21, p0f32t + 4 * 6, 0);
                        __lsx_vst((__m128i)_sum31, p0f32t + 4 * 7, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vst((__m128i)_sum00, p0f32t, 0);
                        __lsx_vst((__m128i)_sum01, p0f32t + 4, 0);
                        __lsx_vst((__m128i)_sum10, p0f32t + out_hstep, 0);
                        __lsx_vst((__m128i)_sum11, p0f32t + out_hstep + 4, 0);
                        __lsx_vst((__m128i)_sum20, p0f32t + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_sum21, p0f32t + out_hstep * 2 + 4, 0);
                        __lsx_vst((__m128i)_sum30, p0f32t + out_hstep * 3, 0);
                        __lsx_vst((__m128i)_sum31, p0f32t + out_hstep * 3 + 4, 0);
                    }
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        transpose4x4_ps(_sum00, _sum10, _sum20, _sum30);
                        transpose4x4_ps(_sum01, _sum11, _sum21, _sum31);

                        __lsx_vstelm_d(float2bfloat_lsx(_sum00), p0t, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum10), p0t + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum20), p0t + 4 * 2, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum30), p0t + 4 * 3, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum01), p0t + 4 * 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum11), p0t + 4 * 5, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum21), p0t + 4 * 6, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum31), p0t + 4 * 7, 0, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vstelm_d(float2bfloat_lsx(_sum00), p0t, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum01), p0t + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum10), p0t + out_hstep, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum11), p0t + out_hstep + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum20), p0t + out_hstep * 2, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum21), p0t + out_hstep * 2 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum30), p0t + out_hstep * 3, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum31), p0t + out_hstep * 3 + 4, 0, 0);
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
                        __lsx_vst((__m128i)_sum00, p0f32, 0);
                        __lsx_vst((__m128i)_sum01, p1f32, 0);
                        __lsx_vst((__m128i)_sum10, p0f32 + 4, 0);
                        __lsx_vst((__m128i)_sum11, p1f32 + 4, 0);
                        __lsx_vst((__m128i)_sum20, p0f32 + 8, 0);
                        __lsx_vst((__m128i)_sum21, p1f32 + 8, 0);
                        __lsx_vst((__m128i)_sum30, p0f32 + 12, 0);
                        __lsx_vst((__m128i)_sum31, p1f32 + 12, 0);
                        p0f32 += 16;
                    }
                    else
                    {
                        unsigned short* p1 = p0 + out_hstep * 4;
                        __lsx_vstelm_d(float2bfloat_lsx(_sum00), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum01), p1, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum10), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum11), p1 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum20), p0 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum21), p1 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum30), p0 + 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum31), p1 + 12, 0, 0);
                        p0 += 16;
                    }
                }
                if (out_elempack == 1)
                {
                    __m128 _r0 = _sum00;
                    __m128 _r1 = _sum10;
                    __m128 _r2 = _sum20;
                    __m128 _r3 = _sum30;
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __m128 _r4 = _sum01;
                    __m128 _r5 = _sum11;
                    __m128 _r6 = _sum21;
                    __m128 _r7 = _sum31;
                    transpose4x4_ps(_r4, _r5, _r6, _r7);

                    if (output_elemtype == 1)
                    {
                        __lsx_vst((__m128i)_r0, p0f32, 0);
                        __lsx_vst((__m128i)_r1, p0f32 + out_hstep, 0);
                        __lsx_vst((__m128i)_r2, p0f32 + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_r3, p0f32 + out_hstep * 3, 0);
                        __lsx_vst((__m128i)_r4, p0f32 + out_hstep * 4, 0);
                        __lsx_vst((__m128i)_r5, p0f32 + out_hstep * 5, 0);
                        __lsx_vst((__m128i)_r6, p0f32 + out_hstep * 6, 0);
                        __lsx_vst((__m128i)_r7, p0f32 + out_hstep * 7, 0);
                        p0f32 += 4;
                    }
                    else
                    {
                        __lsx_vstelm_d(float2bfloat_lsx(_r0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r1), p0 + out_hstep, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r2), p0 + out_hstep * 2, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r3), p0 + out_hstep * 3, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r4), p0 + out_hstep * 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r5), p0 + out_hstep * 5, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r6), p0 + out_hstep * 6, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r7), p0 + out_hstep * 7, 0, 0);
                        p0 += 4;
                    }
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _sum00 = (__m128)__lsx_vld(pp, 0);
            __m128 _sum01 = (__m128)__lsx_vld(pp + 4, 0);
            __m128 _sum10 = (__m128)__lsx_vld(pp + 8, 0);
            __m128 _sum11 = (__m128)__lsx_vld(pp + 12, 0);
            pp += 16;

            if (pC0)
            {
                __m128 _beta = __lsx_vreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    __m128 _c = __lsx_vfmul_s(__lsx_vreplfr2vr_s(pC0[0]), _beta);
                    _sum00 = __lsx_vfadd_s(_sum00, _c);
                    _sum01 = __lsx_vfadd_s(_sum01, _c);
                    _sum10 = __lsx_vfadd_s(_sum10, _c);
                    _sum11 = __lsx_vfadd_s(_sum11, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c0 = __lsx_vfmul_s((__m128)__lsx_vld(pC0, 0), _beta);
                    __m128 _c1 = __lsx_vfmul_s((__m128)__lsx_vld(pC0 + 4, 0), _beta);
                    _sum00 = __lsx_vfadd_s(_sum00, _c0);
                    _sum01 = __lsx_vfadd_s(_sum01, _c1);
                    _sum10 = __lsx_vfadd_s(_sum10, _c0);
                    _sum11 = __lsx_vfadd_s(_sum11, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        const float* pCj0 = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + (j + jj) * 4;
                        const float* pCj1 = pCj0 + c_hstep * 4;
                        _sum00 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj0, 0), _sum00);
                        _sum01 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj1, 0), _sum01);
                        _sum10 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj0 + 4, 0), _sum10);
                        _sum11 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj1 + 4, 0), _sum11);
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
                        _sum00 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(tmp0, 0), _sum00);
                        _sum01 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(tmp1, 0), _sum01);
                        _sum10 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(tmp2, 0), _sum10);
                        _sum11 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(tmp3, 0), _sum11);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(pC0[0]);
                    __m128 _c1 = __lsx_vreplfr2vr_s(pC0[1]);
                    _sum00 = __lsx_vfmadd_s(_beta, _c0, _sum00);
                    _sum01 = __lsx_vfmadd_s(_beta, _c0, _sum01);
                    _sum10 = __lsx_vfmadd_s(_beta, _c1, _sum10);
                    _sum11 = __lsx_vfmadd_s(_beta, _c1, _sum11);
                    pC0 += 2;
                }
            }

            _sum00 = __lsx_vfmul_s(_sum00, _valpha);
            _sum01 = __lsx_vfmul_s(_sum01, _valpha);
            _sum10 = __lsx_vfmul_s(_sum10, _valpha);
            _sum11 = __lsx_vfmul_s(_sum11, _valpha);

            if (output_transpose)
            {
                unsigned short* p0t = (unsigned short*)top_blob + (j + jj) * out_hstep + (i + ii) * out_elempack;
                float* p0f32t = (float*)top_blob + (j + jj) * out_hstep + (i + ii) * out_elempack;

                if (out_elempack == 4)
                {
                    float tmp0[4];
                    float tmp1[4];
                    float tmp2[4];
                    float tmp3[4];
                    __lsx_vst((__m128i)_sum00, tmp0, 0);
                    __lsx_vst((__m128i)_sum01, tmp1, 0);
                    __lsx_vst((__m128i)_sum10, tmp2, 0);
                    __lsx_vst((__m128i)_sum11, tmp3, 0);

                    if (output_elemtype == 1)
                    {
                        p0f32t[0] = tmp0[0];
                        p0f32t[1] = tmp2[0];
                        p0f32t[4] = tmp0[1];
                        p0f32t[5] = tmp2[1];
                        p0f32t[8] = tmp0[2];
                        p0f32t[9] = tmp2[2];
                        p0f32t[12] = tmp0[3];
                        p0f32t[13] = tmp2[3];
                        p0f32t[16] = tmp1[0];
                        p0f32t[17] = tmp3[0];
                        p0f32t[20] = tmp1[1];
                        p0f32t[21] = tmp3[1];
                        p0f32t[24] = tmp1[2];
                        p0f32t[25] = tmp3[2];
                        p0f32t[28] = tmp1[3];
                        p0f32t[29] = tmp3[3];
                    }
                    else
                    {
                        p0t[0] = float32_to_bfloat16(tmp0[0]);
                        p0t[1] = float32_to_bfloat16(tmp2[0]);
                        p0t[4] = float32_to_bfloat16(tmp0[1]);
                        p0t[5] = float32_to_bfloat16(tmp2[1]);
                        p0t[8] = float32_to_bfloat16(tmp0[2]);
                        p0t[9] = float32_to_bfloat16(tmp2[2]);
                        p0t[12] = float32_to_bfloat16(tmp0[3]);
                        p0t[13] = float32_to_bfloat16(tmp2[3]);
                        p0t[16] = float32_to_bfloat16(tmp1[0]);
                        p0t[17] = float32_to_bfloat16(tmp3[0]);
                        p0t[20] = float32_to_bfloat16(tmp1[1]);
                        p0t[21] = float32_to_bfloat16(tmp3[1]);
                        p0t[24] = float32_to_bfloat16(tmp1[2]);
                        p0t[25] = float32_to_bfloat16(tmp3[2]);
                        p0t[28] = float32_to_bfloat16(tmp1[3]);
                        p0t[29] = float32_to_bfloat16(tmp3[3]);
                    }
                }
                if (out_elempack == 1)
                {
                    if (output_elemtype == 1)
                    {
                        __lsx_vst((__m128i)_sum00, p0f32t, 0);
                        __lsx_vst((__m128i)_sum01, p0f32t + 4, 0);
                        __lsx_vst((__m128i)_sum10, p0f32t + out_hstep, 0);
                        __lsx_vst((__m128i)_sum11, p0f32t + out_hstep + 4, 0);
                    }
                    else
                    {
                        __lsx_vstelm_d(float2bfloat_lsx(_sum00), p0t, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum01), p0t + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum10), p0t + out_hstep, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum11), p0t + out_hstep + 4, 0, 0);
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
                        __lsx_vst((__m128i)_sum00, p0f32, 0);
                        __lsx_vst((__m128i)_sum01, p1f32, 0);
                        __lsx_vst((__m128i)_sum10, p0f32 + 4, 0);
                        __lsx_vst((__m128i)_sum11, p1f32 + 4, 0);
                        p0f32 += 8;
                    }
                    else
                    {
                        unsigned short* p1 = p0 + out_hstep * 4;
                        __lsx_vstelm_d(float2bfloat_lsx(_sum00), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum01), p1, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum10), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum11), p1 + 4, 0, 0);
                        p0 += 8;
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[4], tmp1[4], tmp2[4], tmp3[4];
                    __lsx_vst((__m128i)_sum00, tmp0, 0);
                    __lsx_vst((__m128i)_sum01, tmp1, 0);
                    __lsx_vst((__m128i)_sum10, tmp2, 0);
                    __lsx_vst((__m128i)_sum11, tmp3, 0);
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
            __m128 _sum0 = (__m128)__lsx_vld(pp, 0);
            __m128 _sum1 = (__m128)__lsx_vld(pp + 4, 0);
            pp += 8;

            if (pC0)
            {
                __m128 _beta = __lsx_vreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    __m128 _c = __lsx_vfmul_s(__lsx_vreplfr2vr_s(pC0[0]), _beta);
                    _sum0 = __lsx_vfadd_s(_sum0, _c);
                    _sum1 = __lsx_vfadd_s(_sum1, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _sum0 = __lsx_vfadd_s(_sum0, __lsx_vfmul_s((__m128)__lsx_vld(pC0, 0), _beta));
                    _sum1 = __lsx_vfadd_s(_sum1, __lsx_vfmul_s((__m128)__lsx_vld(pC0 + 4, 0), _beta));
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        const float* pCj0 = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + (j + jj) * 4;
                        const float* pCj1 = pCj0 + c_hstep * 4;
                        _sum0 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj0, 0), _sum0);
                        _sum1 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj1, 0), _sum1);
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
                        _sum0 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(tmp0, 0), _sum0);
                        _sum1 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(tmp1, 0), _sum1);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = __lsx_vreplfr2vr_s(pC0[0]);
                    _sum0 = __lsx_vfmadd_s(_beta, _c, _sum0);
                    _sum1 = __lsx_vfmadd_s(_beta, _c, _sum1);
                    pC0 += 1;
                }
            }

            _sum0 = __lsx_vfmul_s(_sum0, _valpha);
            _sum1 = __lsx_vfmul_s(_sum1, _valpha);

            if (output_transpose)
            {
                float tmp0[4];
                float tmp1[4];
                __lsx_vst((__m128i)_sum0, tmp0, 0);
                __lsx_vst((__m128i)_sum1, tmp1, 0);
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
                        __lsx_vst((__m128i)_sum0, p0f32, 0);
                        __lsx_vst((__m128i)_sum1, p1f32, 0);
                        p0f32 += 4;
                    }
                    else
                    {
                        unsigned short* p1 = p0 + out_hstep * 4;
                        __lsx_vstelm_d(float2bfloat_lsx(_sum0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum1), p1, 0, 0);
                        p0 += 4;
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[4];
                    float tmp1[4];
                    __lsx_vst((__m128i)_sum0, tmp0, 0);
                    __lsx_vst((__m128i)_sum1, tmp1, 0);
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
            p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
            p0f32 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;
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

        __m128 _valpha = __lsx_vreplfr2vr_s(alpha);

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _sum0 = (__m128)__lsx_vld(pp, 0);
            __m128 _sum1 = (__m128)__lsx_vld(pp + 4, 0);
            __m128 _sum2 = (__m128)__lsx_vld(pp + 4 * 2, 0);
            __m128 _sum3 = (__m128)__lsx_vld(pp + 4 * 3, 0);
            __m128 _sum4 = (__m128)__lsx_vld(pp + 4 * 4, 0);
            __m128 _sum5 = (__m128)__lsx_vld(pp + 4 * 5, 0);
            __m128 _sum6 = (__m128)__lsx_vld(pp + 4 * 6, 0);
            __m128 _sum7 = (__m128)__lsx_vld(pp + 4 * 7, 0);
            pp += 32;

            if (pC0)
            {
                __m128 _beta = __lsx_vreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    __m128 _c = __lsx_vfmul_s(__lsx_vreplfr2vr_s(pC0[0]), _beta);
                    _sum0 = __lsx_vfadd_s(_sum0, _c);
                    _sum1 = __lsx_vfadd_s(_sum1, _c);
                    _sum2 = __lsx_vfadd_s(_sum2, _c);
                    _sum3 = __lsx_vfadd_s(_sum3, _c);
                    _sum4 = __lsx_vfadd_s(_sum4, _c);
                    _sum5 = __lsx_vfadd_s(_sum5, _c);
                    _sum6 = __lsx_vfadd_s(_sum6, _c);
                    _sum7 = __lsx_vfadd_s(_sum7, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c = __lsx_vfmul_s((__m128)__lsx_vld(pC0, 0), _beta);
                    _sum0 = __lsx_vfadd_s(_sum0, _c);
                    _sum1 = __lsx_vfadd_s(_sum1, _c);
                    _sum2 = __lsx_vfadd_s(_sum2, _c);
                    _sum3 = __lsx_vfadd_s(_sum3, _c);
                    _sum4 = __lsx_vfadd_s(_sum4, _c);
                    _sum5 = __lsx_vfadd_s(_sum5, _c);
                    _sum6 = __lsx_vfadd_s(_sum6, _c);
                    _sum7 = __lsx_vfadd_s(_sum7, _c);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        const float* pCj = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + (j + jj) * 4;
                        _sum0 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj, 0), _sum0);
                        _sum1 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj + 4, 0), _sum1);
                        _sum2 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj + 8, 0), _sum2);
                        _sum3 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj + 12, 0), _sum3);
                        _sum4 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj + 16, 0), _sum4);
                        _sum5 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj + 20, 0), _sum5);
                        _sum6 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj + 24, 0), _sum6);
                        _sum7 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj + 28, 0), _sum7);
                    }
                    if (c_elempack == 1)
                    {
                        const float* pCj = (const float*)C + (i + ii) * c_hstep + (j + jj);

                        __m128 _c0 = (__m128)__lsx_vld(pCj, 0);
                        __m128 _c1 = (__m128)__lsx_vld(pCj + c_hstep, 0);
                        __m128 _c2 = (__m128)__lsx_vld(pCj + c_hstep * 2, 0);
                        __m128 _c3 = (__m128)__lsx_vld(pCj + c_hstep * 3, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _sum0 = __lsx_vfmadd_s(_beta, _c0, _sum0);
                        _sum1 = __lsx_vfmadd_s(_beta, _c1, _sum1);
                        _sum2 = __lsx_vfmadd_s(_beta, _c2, _sum2);
                        _sum3 = __lsx_vfmadd_s(_beta, _c3, _sum3);

                        _c0 = (__m128)__lsx_vld(pCj + 4, 0);
                        _c1 = (__m128)__lsx_vld(pCj + c_hstep + 4, 0);
                        _c2 = (__m128)__lsx_vld(pCj + c_hstep * 2 + 4, 0);
                        _c3 = (__m128)__lsx_vld(pCj + c_hstep * 3 + 4, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _sum4 = __lsx_vfmadd_s(_beta, _c0, _sum4);
                        _sum5 = __lsx_vfmadd_s(_beta, _c1, _sum5);
                        _sum6 = __lsx_vfmadd_s(_beta, _c2, _sum6);
                        _sum7 = __lsx_vfmadd_s(_beta, _c3, _sum7);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _sum0 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pC0[0]), _sum0);
                    _sum1 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pC0[1]), _sum1);
                    _sum2 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pC0[2]), _sum2);
                    _sum3 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pC0[3]), _sum3);
                    _sum4 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pC0[4]), _sum4);
                    _sum5 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pC0[5]), _sum5);
                    _sum6 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pC0[6]), _sum6);
                    _sum7 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pC0[7]), _sum7);
                    pC0 += 8;
                }
            }

            _sum0 = __lsx_vfmul_s(_sum0, _valpha);
            _sum1 = __lsx_vfmul_s(_sum1, _valpha);
            _sum2 = __lsx_vfmul_s(_sum2, _valpha);
            _sum3 = __lsx_vfmul_s(_sum3, _valpha);
            _sum4 = __lsx_vfmul_s(_sum4, _valpha);
            _sum5 = __lsx_vfmul_s(_sum5, _valpha);
            _sum6 = __lsx_vfmul_s(_sum6, _valpha);
            _sum7 = __lsx_vfmul_s(_sum7, _valpha);

            if (output_transpose)
            {
                if (output_elemtype == 1)
                {
                    if (out_elempack == 8)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);

                        __lsx_vst((__m128i)_sum0, p0f32, 0);
                        __lsx_vst((__m128i)_sum4, p0f32 + 4, 0);
                        __lsx_vst((__m128i)_sum1, p0f32 + 8, 0);
                        __lsx_vst((__m128i)_sum5, p0f32 + 12, 0);
                        __lsx_vst((__m128i)_sum2, p0f32 + 16, 0);
                        __lsx_vst((__m128i)_sum6, p0f32 + 20, 0);
                        __lsx_vst((__m128i)_sum3, p0f32 + 24, 0);
                        __lsx_vst((__m128i)_sum7, p0f32 + 28, 0);
                    }
                    if (out_elempack == 4)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);

                        __lsx_vst((__m128i)_sum0, p0f32, 0);
                        __lsx_vst((__m128i)_sum1, p0f32 + 4, 0);
                        __lsx_vst((__m128i)_sum2, p0f32 + 4 * 2, 0);
                        __lsx_vst((__m128i)_sum3, p0f32 + 4 * 3, 0);
                        __lsx_vst((__m128i)_sum4, p0f32 + out_hstep * 4, 0);
                        __lsx_vst((__m128i)_sum5, p0f32 + out_hstep * 4 + 4, 0);
                        __lsx_vst((__m128i)_sum6, p0f32 + out_hstep * 4 + 4 * 2, 0);
                        __lsx_vst((__m128i)_sum7, p0f32 + out_hstep * 4 + 4 * 3, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vst((__m128i)_sum0, p0f32, 0);
                        __lsx_vst((__m128i)_sum1, p0f32 + out_hstep, 0);
                        __lsx_vst((__m128i)_sum2, p0f32 + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_sum3, p0f32 + out_hstep * 3, 0);
                        __lsx_vst((__m128i)_sum4, p0f32 + out_hstep * 4, 0);
                        __lsx_vst((__m128i)_sum5, p0f32 + out_hstep * 5, 0);
                        __lsx_vst((__m128i)_sum6, p0f32 + out_hstep * 6, 0);
                        __lsx_vst((__m128i)_sum7, p0f32 + out_hstep * 7, 0);
                    }
                    p0f32 += out_hstep * 8;
                }
                else
                {
                    if (out_elempack == 8)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);

                        __lsx_vstelm_d(float2bfloat_lsx(_sum0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum4), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum1), p0 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum5), p0 + 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum2), p0 + 16, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum6), p0 + 20, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum3), p0 + 24, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum7), p0 + 28, 0, 0);
                    }
                    if (out_elempack == 4)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);

                        __lsx_vstelm_d(float2bfloat_lsx(_sum0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum1), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum2), p0 + 4 * 2, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum3), p0 + 4 * 3, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum4), p0 + out_hstep * 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum5), p0 + out_hstep * 4 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum6), p0 + out_hstep * 4 + 4 * 2, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum7), p0 + out_hstep * 4 + 4 * 3, 0, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vstelm_d(float2bfloat_lsx(_sum0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum1), p0 + out_hstep, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum2), p0 + out_hstep * 2, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum3), p0 + out_hstep * 3, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum4), p0 + out_hstep * 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum5), p0 + out_hstep * 5, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum6), p0 + out_hstep * 6, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum7), p0 + out_hstep * 7, 0, 0);
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
                        __lsx_vst((__m128i)_sum0, p0f32, 0);
                        __lsx_vst((__m128i)_sum1, p0f32 + 4, 0);
                        __lsx_vst((__m128i)_sum2, p0f32 + 8, 0);
                        __lsx_vst((__m128i)_sum3, p0f32 + 12, 0);
                        __lsx_vst((__m128i)_sum4, p0f32 + 16, 0);
                        __lsx_vst((__m128i)_sum5, p0f32 + 20, 0);
                        __lsx_vst((__m128i)_sum6, p0f32 + 24, 0);
                        __lsx_vst((__m128i)_sum7, p0f32 + 28, 0);
                        p0f32 += 32;
                    }
                    else
                    {
                        __lsx_vstelm_d(float2bfloat_lsx(_sum0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum1), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum2), p0 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum3), p0 + 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum4), p0 + 16, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum5), p0 + 20, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum6), p0 + 24, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum7), p0 + 28, 0, 0);
                        p0 += 32;
                    }
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                    transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);
                    if (output_elemtype == 1)
                    {
                        __lsx_vst((__m128i)_sum0, p0f32, 0);
                        __lsx_vst((__m128i)_sum4, p0f32 + 4, 0);
                        __lsx_vst((__m128i)_sum1, p0f32 + out_hstep, 0);
                        __lsx_vst((__m128i)_sum5, p0f32 + out_hstep + 4, 0);
                        __lsx_vst((__m128i)_sum2, p0f32 + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_sum6, p0f32 + out_hstep * 2 + 4, 0);
                        __lsx_vst((__m128i)_sum3, p0f32 + out_hstep * 3, 0);
                        __lsx_vst((__m128i)_sum7, p0f32 + out_hstep * 3 + 4, 0);
                        p0f32 += 8;
                    }
                    else
                    {
                        __lsx_vstelm_d(float2bfloat_lsx(_sum0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum4), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum1), p0 + out_hstep, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum5), p0 + out_hstep + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum2), p0 + out_hstep * 2, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum6), p0 + out_hstep * 2 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum3), p0 + out_hstep * 3, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum7), p0 + out_hstep * 3 + 4, 0, 0);
                        p0 += 8;
                    }
                }
            }
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _sum0 = (__m128)__lsx_vld(pp, 0);
            __m128 _sum1 = (__m128)__lsx_vld(pp + 4, 0);
            __m128 _sum2 = (__m128)__lsx_vld(pp + 8, 0);
            __m128 _sum3 = (__m128)__lsx_vld(pp + 12, 0);
            pp += 16;

            if (pC0)
            {
                __m128 _beta = __lsx_vreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    __m128 _c = __lsx_vfmul_s(__lsx_vreplfr2vr_s(pC0[0]), _beta);
                    _sum0 = __lsx_vfadd_s(_sum0, _c);
                    _sum1 = __lsx_vfadd_s(_sum1, _c);
                    _sum2 = __lsx_vfadd_s(_sum2, _c);
                    _sum3 = __lsx_vfadd_s(_sum3, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c = __lsx_vfmul_s((__m128)__lsx_vld(pC0, 0), _beta);
                    _sum0 = __lsx_vfadd_s(_sum0, _c);
                    _sum1 = __lsx_vfadd_s(_sum1, _c);
                    _sum2 = __lsx_vfadd_s(_sum2, _c);
                    _sum3 = __lsx_vfadd_s(_sum3, _c);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        const float* pCj = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + (j + jj) * 4;
                        _sum0 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj, 0), _sum0);
                        _sum1 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj + 4, 0), _sum1);
                        _sum2 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj + 8, 0), _sum2);
                        _sum3 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj + 12, 0), _sum3);
                    }
                    if (c_elempack == 1)
                    {
                        const float* pCj = (const float*)C + (i + ii) * c_hstep + (j + jj);
                        __m128 _c0 = (__m128)__lsx_vld(pCj, 0);
                        __m128 _c1 = (__m128)__lsx_vld(pCj + c_hstep, 0);
                        __m128 _c2 = (__m128)__lsx_vld(pCj + c_hstep * 2, 0);
                        __m128 _c3 = (__m128)__lsx_vld(pCj + c_hstep * 3, 0);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        _sum0 = __lsx_vfmadd_s(_beta, _c0, _sum0);
                        _sum1 = __lsx_vfmadd_s(_beta, _c1, _sum1);
                        _sum2 = __lsx_vfmadd_s(_beta, _c2, _sum2);
                        _sum3 = __lsx_vfmadd_s(_beta, _c3, _sum3);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _sum0 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pC0[0]), _sum0);
                    _sum1 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pC0[1]), _sum1);
                    _sum2 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pC0[2]), _sum2);
                    _sum3 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pC0[3]), _sum3);
                    pC0 += 4;
                }
            }

            _sum0 = __lsx_vfmul_s(_sum0, _valpha);
            _sum1 = __lsx_vfmul_s(_sum1, _valpha);
            _sum2 = __lsx_vfmul_s(_sum2, _valpha);
            _sum3 = __lsx_vfmul_s(_sum3, _valpha);

            if (output_transpose)
            {
                if (output_elemtype == 1)
                {
                    if (out_elempack == 8)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        __lsx_vst((__m128i)_sum0, p0f32, 0);
                        __lsx_vst((__m128i)_sum1, p0f32 + 8, 0);
                        __lsx_vst((__m128i)_sum2, p0f32 + 16, 0);
                        __lsx_vst((__m128i)_sum3, p0f32 + 24, 0);
                    }
                    if (out_elempack == 4)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        __lsx_vst((__m128i)_sum0, p0f32, 0);
                        __lsx_vst((__m128i)_sum1, p0f32 + 4, 0);
                        __lsx_vst((__m128i)_sum2, p0f32 + 4 * 2, 0);
                        __lsx_vst((__m128i)_sum3, p0f32 + 4 * 3, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vst((__m128i)_sum0, p0f32, 0);
                        __lsx_vst((__m128i)_sum1, p0f32 + out_hstep, 0);
                        __lsx_vst((__m128i)_sum2, p0f32 + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_sum3, p0f32 + out_hstep * 3, 0);
                    }
                    p0f32 += out_hstep * 4;
                }
                else
                {
                    if (out_elempack == 8)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum1), p0 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum2), p0 + 16, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum3), p0 + 24, 0, 0);
                    }
                    if (out_elempack == 4)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum1), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum2), p0 + 4 * 2, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum3), p0 + 4 * 3, 0, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vstelm_d(float2bfloat_lsx(_sum0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum1), p0 + out_hstep, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum2), p0 + out_hstep * 2, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum3), p0 + out_hstep * 3, 0, 0);
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
                        __lsx_vst((__m128i)_sum0, p0f32, 0);
                        __lsx_vst((__m128i)_sum1, p0f32 + 4, 0);
                        __lsx_vst((__m128i)_sum2, p0f32 + 8, 0);
                        __lsx_vst((__m128i)_sum3, p0f32 + 12, 0);
                        p0f32 += 16;
                    }
                    else
                    {
                        __lsx_vstelm_d(float2bfloat_lsx(_sum0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum1), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum2), p0 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum3), p0 + 12, 0, 0);
                        p0 += 16;
                    }
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                    if (output_elemtype == 1)
                    {
                        __lsx_vst((__m128i)_sum0, p0f32, 0);
                        __lsx_vst((__m128i)_sum1, p0f32 + out_hstep, 0);
                        __lsx_vst((__m128i)_sum2, p0f32 + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_sum3, p0f32 + out_hstep * 3, 0);
                        p0f32 += 4;
                    }
                    else
                    {
                        __lsx_vstelm_d(float2bfloat_lsx(_sum0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum1), p0 + out_hstep, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum2), p0 + out_hstep * 2, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum3), p0 + out_hstep * 3, 0, 0);
                        p0 += 4;
                    }
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _sum0 = (__m128)__lsx_vld(pp, 0);
            __m128 _sum1 = (__m128)__lsx_vld(pp + 4, 0);
            pp += 8;

            if (pC0)
            {
                __m128 _beta = __lsx_vreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    __m128 _c = __lsx_vfmul_s(__lsx_vreplfr2vr_s(pC0[0]), _beta);
                    _sum0 = __lsx_vfadd_s(_sum0, _c);
                    _sum1 = __lsx_vfadd_s(_sum1, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c = __lsx_vfmul_s((__m128)__lsx_vld(pC0, 0), _beta);
                    _sum0 = __lsx_vfadd_s(_sum0, _c);
                    _sum1 = __lsx_vfadd_s(_sum1, _c);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        const float* pCj = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + (j + jj) * 4;
                        _sum0 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj, 0), _sum0);
                        _sum1 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj + 4, 0), _sum1);
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
                        _sum0 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(tmp0, 0), _sum0);
                        _sum1 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(tmp1, 0), _sum1);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _sum0 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pC0[0]), _sum0);
                    _sum1 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pC0[1]), _sum1);
                    pC0 += 2;
                }
            }

            _sum0 = __lsx_vfmul_s(_sum0, _valpha);
            _sum1 = __lsx_vfmul_s(_sum1, _valpha);

            if (output_transpose)
            {
                float tmp0[4], tmp1[4];
                __lsx_vst((__m128i)_sum0, tmp0, 0);
                __lsx_vst((__m128i)_sum1, tmp1, 0);
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
                        __lsx_vst((__m128i)_sum0, p0f32, 0);
                        __lsx_vst((__m128i)_sum1, p0f32 + 4, 0);
                        p0f32 += 8;
                    }
                    else
                    {
                        __lsx_vstelm_d(float2bfloat_lsx(_sum0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum1), p0 + 4, 0, 0);
                        p0 += 8;
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp0[4], tmp1[4];
                    __lsx_vst((__m128i)_sum0, tmp0, 0);
                    __lsx_vst((__m128i)_sum1, tmp1, 0);
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
            __m128 _sum0 = (__m128)__lsx_vld(pp, 0);
            pp += 4;

            if (pC0)
            {
                __m128 _beta = __lsx_vreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    _sum0 = __lsx_vfadd_s(_sum0, __lsx_vfmul_s(__lsx_vreplfr2vr_s(pC0[0]), _beta));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _sum0 = __lsx_vfadd_s(_sum0, __lsx_vfmul_s((__m128)__lsx_vld(pC0, 0), _beta));
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        const float* pCj = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + (j + jj) * 4;
                        _sum0 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj, 0), _sum0);
                    }
                    if (c_elempack == 1)
                    {
                        const float* pCj = (const float*)C + (i + ii) * c_hstep + (j + jj);
                        float tmp[4];
                        tmp[0] = pCj[0];
                        tmp[1] = pCj[c_hstep];
                        tmp[2] = pCj[c_hstep * 2];
                        tmp[3] = pCj[c_hstep * 3];
                        _sum0 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(tmp, 0), _sum0);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _sum0 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pC0[0]), _sum0);
                    pC0 += 1;
                }
            }

            _sum0 = __lsx_vfmul_s(_sum0, _valpha);

            if (output_transpose)
            {
                float tmp[4];
                __lsx_vst((__m128i)_sum0, tmp, 0);
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
                        __lsx_vst((__m128i)_sum0, p0f32, 0);
                        p0f32 += 4;
                    }
                    else
                    {
                        __lsx_vstelm_d(float2bfloat_lsx(_sum0), p0, 0, 0);
                        p0 += 4;
                    }
                }
                if (out_elempack == 1)
                {
                    float tmp[4];
                    __lsx_vst((__m128i)_sum0, tmp, 0);
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
#endif // __loongarch_sx
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

#if __loongarch_sx
#if __loongarch_asx
    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_N = std::max(8, tile_size / 8 * 8);
    TILE_K = std::max(8, tile_size / 8 * 8);
#else
    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_N = std::max(8, tile_size / 8 * 8);
    TILE_K = std::max(8, tile_size / 8 * 8);
#endif // __loongarch_asx
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
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);
#endif // __loongarch_asx
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(unsigned short) / TILE_K);
#if __loongarch_sx
#if __loongarch_asx
            TILE_M = std::max(8, tile_size / 8 * 8);
            TILE_N = std::max(8, tile_size / 8 * 8);
#else
            TILE_M = std::max(8, tile_size / 8 * 8);
            TILE_N = std::max(8, tile_size / 8 * 8);
#endif // __loongarch_asx
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
#if __loongarch_asx
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#endif // __loongarch_asx
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __loongarch_sx
#if __loongarch_asx
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8);
#else
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8);
#endif // __loongarch_asx
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif
    }

    if (nT > 1)
    {
#if __loongarch_sx
#if __loongarch_asx
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#else
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#endif // __loongarch_asx
#else
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
    }

    if (constant_TILE_M > 0)
    {
#if __loongarch_sx
#if __loongarch_asx
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
#else
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
#endif // __loongarch_asx
#else
        TILE_M = (constant_TILE_M + 1) / 2 * 2;
#endif
    }
    if (constant_TILE_N > 0)
    {
#if __loongarch_sx
#if __loongarch_asx
        TILE_N = (constant_TILE_N + 7) / 8 * 8;
#else
        TILE_N = (constant_TILE_N + 7) / 8 * 8;
#endif // __loongarch_asx
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
        TILE_K = (constant_TILE_K + 7) / 8 * 8;
#endif // __loongarch_asx
#else
        TILE_K = (constant_TILE_K + 1) / 2 * 2;
#endif
    }
}

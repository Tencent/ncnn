// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if __loongarch_sx
// Store 4 bf16 values to potentially unaligned pointer
static NCNN_FORCEINLINE void float2bfloat_lsx_store(const __m128& v0, unsigned short* ptr)
{
    __m128i _bf16 = float2bfloat_lsx(v0);
    int64_t val = __lsx_vpickve2gr_d(_bf16, 0);
    __builtin_memcpy(ptr, &val, sizeof(int64_t));
}

static NCNN_FORCEINLINE void store_fp32_lsx(const __m128& v0, float* ptr)
{
#if __loongarch_asx
    __lsx_vst((__m128i)v0, ptr, 0);
#else
    float tmp[4];
    __lsx_vst((__m128i)v0, tmp, 0);
    __builtin_memcpy(ptr, tmp, sizeof(tmp));
#endif
}

#if __loongarch_asx
static NCNN_FORCEINLINE void store_fp32_lasx(const __m256& v0, float* ptr)
{
    float tmp[8];
    __lasx_xvst((__m256i)v0, tmp, 0);
    __builtin_memcpy(ptr, tmp, sizeof(tmp));
}
#endif // __loongarch_asx
#endif // __loongarch_sx

static void pack_A_tile_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __loongarch_sx
#if __loongarch_asx
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
#endif // __loongarch_asx
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
#if __loongarch_asx
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
#endif // __loongarch_asx
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
    for (; jj + 11 < max_jj; jj += 12)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            if (((j + jj) & 7) == 0)
            {
                const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 8;
                const unsigned short* p1 = (const unsigned short*)B + (j + jj + 8) * B_hstep + k * 8;

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
                    pp[8] = p1[0];
                    pp[9] = p1[1];
                    pp[10] = p1[2];
                    pp[11] = p1[3];
                    pp += 12;
                    p0 += 8;
                    p1 += 8;
                }
            }
            else
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    for (int c = 0; c < 12; c++)
                    {
                        const int col = j + jj + c;
                        const int j_pack = col / 8;
                        const int j_lane = col % 8;
                        const unsigned short* p = (const unsigned short*)B + (size_t)j_pack * B_hstep * 8 + (k + kk) * 8 + j_lane;
                        pp[c] = p[0];
                    }
                    pp += 12;
                }
            }
        }
#endif // __loongarch_asx
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
    for (; jj + 11 < max_jj; jj += 12)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;
            const unsigned short* p1 = (const unsigned short*)B + k * B_hstep + (j + jj + 8) * 8;

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
                pp[8] = p1[0];
                pp[9] = p1[8];
                pp[10] = p1[16];
                pp[11] = p1[24];
                pp += 12;
                p0++;
                p1++;
                if ((kk + 1) % 8 == 0)
                {
                    p0 += B_hstep * 8 - 8;
                    p1 += B_hstep * 8 - 8;
                }
            }
        }
#endif // __loongarch_asx
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
        for (; jj + 11 < max_jj; jj += 12)
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
            __m256 _sum8 = (__m256)__lasx_xvldi(0);
            __m256 _sum9 = (__m256)__lasx_xvldi(0);
            __m256 _suma = (__m256)__lasx_xvldi(0);
            __m256 _sumb = (__m256)__lasx_xvldi(0);

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
                _sum8 = (__m256)__lasx_xvld(outptr + 64, 0);
                _sum9 = (__m256)__lasx_xvld(outptr + 72, 0);
                _suma = (__m256)__lasx_xvld(outptr + 80, 0);
                _sumb = (__m256)__lasx_xvld(outptr + 88, 0);
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
                _sum8 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[8])), _sum8);
                _sum9 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[9])), _sum9);
                _suma = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[10])), _suma);
                _sumb = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(bfloat16_to_float32(pB[11])), _sumb);

                pA += 8;
                pB += 12;
            }

            store_fp32_lasx(_sum0, outptr);
            store_fp32_lasx(_sum1, outptr + 8);
            store_fp32_lasx(_sum2, outptr + 16);
            store_fp32_lasx(_sum3, outptr + 24);
            store_fp32_lasx(_sum4, outptr + 32);
            store_fp32_lasx(_sum5, outptr + 40);
            store_fp32_lasx(_sum6, outptr + 48);
            store_fp32_lasx(_sum7, outptr + 56);
            store_fp32_lasx(_sum8, outptr + 64);
            store_fp32_lasx(_sum9, outptr + 72);
            store_fp32_lasx(_suma, outptr + 80);
            store_fp32_lasx(_sumb, outptr + 88);

            outptr += 96;
        }
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

            store_fp32_lasx(_sum0, outptr);
            store_fp32_lasx(_sum1, outptr + 8);
            store_fp32_lasx(_sum2, outptr + 16);
            store_fp32_lasx(_sum3, outptr + 24);
            store_fp32_lasx(_sum4, outptr + 32);
            store_fp32_lasx(_sum5, outptr + 40);
            store_fp32_lasx(_sum6, outptr + 48);
            store_fp32_lasx(_sum7, outptr + 56);

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

            store_fp32_lasx(_sum0, outptr);
            store_fp32_lasx(_sum1, outptr + 8);
            store_fp32_lasx(_sum2, outptr + 16);
            store_fp32_lasx(_sum3, outptr + 24);

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

            store_fp32_lasx(_sum0, outptr);
            store_fp32_lasx(_sum1, outptr + 8);

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

            store_fp32_lasx(_sum0, outptr);

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
#endif // __loongarch_asx
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
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
            __m128 _sum8;
            __m128 _sum9;
            __m128 _suma;
            __m128 _sumb;

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
                _sum8 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum9 = (__m128)__lsx_vreplgr2vr_w(0);
                _suma = (__m128)__lsx_vreplgr2vr_w(0);
                _sumb = (__m128)__lsx_vreplgr2vr_w(0);
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
                _sum8 = (__m128)__lsx_vld(outptr + 4 * 8, 0);
                _sum9 = (__m128)__lsx_vld(outptr + 4 * 9, 0);
                _suma = (__m128)__lsx_vld(outptr + 4 * 10, 0);
                _sumb = (__m128)__lsx_vld(outptr + 4 * 11, 0);
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
                _sum8 = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[8])), _sum8);
                _sum9 = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[9])), _sum9);
                _suma = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[10])), _suma);
                _sumb = __lsx_vfmadd_s(_pA, __lsx_vreplfr2vr_s(bfloat16_to_float32(pB[11])), _sumb);

                pA += 4;
                pB += 12;
            }

#if __loongarch_asx
            __lsx_vst((__m128i)_sum0, outptr, 0);
            __lsx_vst((__m128i)_sum1, outptr + 4, 0);
            __lsx_vst((__m128i)_sum2, outptr + 4 * 2, 0);
            __lsx_vst((__m128i)_sum3, outptr + 4 * 3, 0);
            __lsx_vst((__m128i)_sum4, outptr + 4 * 4, 0);
            __lsx_vst((__m128i)_sum5, outptr + 4 * 5, 0);
            __lsx_vst((__m128i)_sum6, outptr + 4 * 6, 0);
            __lsx_vst((__m128i)_sum7, outptr + 4 * 7, 0);
            __lsx_vst((__m128i)_sum8, outptr + 4 * 8, 0);
            __lsx_vst((__m128i)_sum9, outptr + 4 * 9, 0);
            __lsx_vst((__m128i)_suma, outptr + 4 * 10, 0);
            __lsx_vst((__m128i)_sumb, outptr + 4 * 11, 0);
#else
            store_fp32_lsx(_sum0, outptr);
            store_fp32_lsx(_sum1, outptr + 4);
            store_fp32_lsx(_sum2, outptr + 4 * 2);
            store_fp32_lsx(_sum3, outptr + 4 * 3);
            store_fp32_lsx(_sum4, outptr + 4 * 4);
            store_fp32_lsx(_sum5, outptr + 4 * 5);
            store_fp32_lsx(_sum6, outptr + 4 * 6);
            store_fp32_lsx(_sum7, outptr + 4 * 7);
            store_fp32_lsx(_sum8, outptr + 4 * 8);
            store_fp32_lsx(_sum9, outptr + 4 * 9);
            store_fp32_lsx(_suma, outptr + 4 * 10);
            store_fp32_lsx(_sumb, outptr + 4 * 11);
#endif

            outptr += 48;
        }
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

#if __loongarch_asx
            __lsx_vst((__m128i)_sum0, outptr, 0);
            __lsx_vst((__m128i)_sum1, outptr + 4, 0);
            __lsx_vst((__m128i)_sum2, outptr + 4 * 2, 0);
            __lsx_vst((__m128i)_sum3, outptr + 4 * 3, 0);
            __lsx_vst((__m128i)_sum4, outptr + 4 * 4, 0);
            __lsx_vst((__m128i)_sum5, outptr + 4 * 5, 0);
            __lsx_vst((__m128i)_sum6, outptr + 4 * 6, 0);
            __lsx_vst((__m128i)_sum7, outptr + 4 * 7, 0);
#else
            store_fp32_lsx(_sum0, outptr);
            store_fp32_lsx(_sum1, outptr + 4);
            store_fp32_lsx(_sum2, outptr + 4 * 2);
            store_fp32_lsx(_sum3, outptr + 4 * 3);
            store_fp32_lsx(_sum4, outptr + 4 * 4);
            store_fp32_lsx(_sum5, outptr + 4 * 5);
            store_fp32_lsx(_sum6, outptr + 4 * 6);
            store_fp32_lsx(_sum7, outptr + 4 * 7);
#endif

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

#if __loongarch_asx
            __lsx_vst((__m128i)_sum0, outptr, 0);
            __lsx_vst((__m128i)_sum1, outptr + 4, 0);
            __lsx_vst((__m128i)_sum2, outptr + 4 * 2, 0);
            __lsx_vst((__m128i)_sum3, outptr + 4 * 3, 0);
#else
            store_fp32_lsx(_sum0, outptr);
            store_fp32_lsx(_sum1, outptr + 4);
            store_fp32_lsx(_sum2, outptr + 4 * 2);
            store_fp32_lsx(_sum3, outptr + 4 * 3);
#endif

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

#if __loongarch_asx
            __lsx_vst((__m128i)_sum0, outptr, 0);
            __lsx_vst((__m128i)_sum1, outptr + 4, 0);
#else
            store_fp32_lsx(_sum0, outptr);
            store_fp32_lsx(_sum1, outptr + 4);
#endif

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

#if __loongarch_asx
            __lsx_vst((__m128i)_sum0, outptr, 0);
#else
            store_fp32_lsx(_sum0, outptr);
#endif

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
        const float* pCi = pC;
        if (pCi)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pCi = (const float*)C + (i + ii);
            if (broadcast_type_C == 3)
            {
                if (c_elempack == 8)
                    pCi = (const float*)C + (size_t)((i + ii) / 8) * c_hstep * 8 + j * 8;
                else if (c_elempack == 4)
                    pCi = (const float*)C + (size_t)((i + ii) / 4) * c_hstep * 4 + j * 4;
                else
                    pCi = (const float*)C + (size_t)(i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
                pCi = (const float*)C + j;
        }

        __m256 _valpha = (__m256)__lasx_xvreplfr2vr_s(alpha);

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            __m256 _sum = (__m256)__lasx_xvld(pp, 0);
            pp += 8;

            if (pCi)
            {
                __m256 _beta_v = (__m256)__lasx_xvreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                    _sum = __lasx_xvfmadd_s(_c, _beta_v, _sum);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m256 _c = (__m256)__lasx_xvld(pCi, 0);
                    _sum = __lasx_xvfmadd_s(_c, _beta_v, _sum);
                }
                if (broadcast_type_C == 3)
                {
                    // Load 8 values from C matrix for this (ii, jj) position
                    if (c_elempack == 8)
                    {
                        __m256 _c = (__m256)__lasx_xvld(pCi, 0);
                        _sum = __lasx_xvfmadd_s(_c, _beta_v, _sum);
                        pCi += 8;
                    }
                    else if (c_elempack == 4)
                    {
                        __m128 _cl = (__m128)__lsx_vld(pCi, 0);
                        __m128 _ch = (__m128)__lsx_vld(pCi + c_hstep * 4, 0);
                        __m256 _c = combine4x2_ps(_cl, _ch);
                        _sum = __lasx_xvfmadd_s(_c, _beta_v, _sum);
                        pCi += 4;
                    }
                    else // c_elempack == 1
                    {
                        float ctmp[8];
                        for (int r = 0; r < 8; r++)
                            ctmp[r] = pCi[c_hstep * r];
                        __m256 _c = (__m256)__lasx_xvld(ctmp, 0);
                        _sum = __lasx_xvfmadd_s(_c, _beta_v, _sum);
                        pCi += 1;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                    _sum = __lasx_xvfmadd_s(_c, _beta_v, _sum);
                    pCi += 1;
                }
            }

            _sum = __lasx_xvfmul_s(_sum, _valpha);

            // Scatter write to output using per-element addressing for correctness with all elempacks
            float tmp[8];
            __lasx_xvst((__m256i)_sum, tmp, 0);
            int col = j + jj;
            for (int r = 0; r < 8; r++)
            {
                int row = i + ii + r;
                size_t offset;
                if (output_transpose)
                    offset = (size_t)(col / out_elempack) * out_hstep * out_elempack + (size_t)row * out_elempack + col % out_elempack;
                else
                    offset = (size_t)(row / out_elempack) * out_hstep * out_elempack + (size_t)col * out_elempack + row % out_elempack;

                if (output_elemtype == 1)
                    ((float*)top_blob)[offset] = tmp[r];
                else
                    ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(tmp[r]);
            }
        }
    }
#endif // __loongarch_asx
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

        __m128 _valpha = __lsx_vreplfr2vr_s(alpha);

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            __m128 _sum0;
            __m128 _sum1;
            __m128 _sum2;
            __m128 _sum3;
            __m128 _sum4;
            __m128 _sum5;
            __m128 _sum6;
            __m128 _sum7;
            __m128 _sum8;
            __m128 _sum9;
            __m128 _suma;
            __m128 _sumb;

            _sum0 = (__m128)__lsx_vld(pp, 0);
            _sum1 = (__m128)__lsx_vld(pp + 4, 0);
            _sum2 = (__m128)__lsx_vld(pp + 4 * 2, 0);
            _sum3 = (__m128)__lsx_vld(pp + 4 * 3, 0);
            _sum4 = (__m128)__lsx_vld(pp + 4 * 4, 0);
            _sum5 = (__m128)__lsx_vld(pp + 4 * 5, 0);
            _sum6 = (__m128)__lsx_vld(pp + 4 * 6, 0);
            _sum7 = (__m128)__lsx_vld(pp + 4 * 7, 0);
            _sum8 = (__m128)__lsx_vld(pp + 4 * 8, 0);
            _sum9 = (__m128)__lsx_vld(pp + 4 * 9, 0);
            _suma = (__m128)__lsx_vld(pp + 4 * 10, 0);
            _sumb = (__m128)__lsx_vld(pp + 4 * 11, 0);
            pp += 48;

            if (pCi)
            {
                __m128 _beta = __lsx_vreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    __m128 _c = __lsx_vreplfr2vr_s(pCi[0]);
                    _c = __lsx_vfmul_s(_c, _beta);
                    _sum0 = __lsx_vfadd_s(_sum0, _c);
                    _sum1 = __lsx_vfadd_s(_sum1, _c);
                    _sum2 = __lsx_vfadd_s(_sum2, _c);
                    _sum3 = __lsx_vfadd_s(_sum3, _c);
                    _sum4 = __lsx_vfadd_s(_sum4, _c);
                    _sum5 = __lsx_vfadd_s(_sum5, _c);
                    _sum6 = __lsx_vfadd_s(_sum6, _c);
                    _sum7 = __lsx_vfadd_s(_sum7, _c);
                    _sum8 = __lsx_vfadd_s(_sum8, _c);
                    _sum9 = __lsx_vfadd_s(_sum9, _c);
                    _suma = __lsx_vfadd_s(_suma, _c);
                    _sumb = __lsx_vfadd_s(_sumb, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c = (__m128)__lsx_vld(pCi, 0);
                    _c = __lsx_vfmul_s(_c, _beta);
                    _sum0 = __lsx_vfadd_s(_sum0, _c);
                    _sum1 = __lsx_vfadd_s(_sum1, _c);
                    _sum2 = __lsx_vfadd_s(_sum2, _c);
                    _sum3 = __lsx_vfadd_s(_sum3, _c);
                    _sum4 = __lsx_vfadd_s(_sum4, _c);
                    _sum5 = __lsx_vfadd_s(_sum5, _c);
                    _sum6 = __lsx_vfadd_s(_sum6, _c);
                    _sum7 = __lsx_vfadd_s(_sum7, _c);
                    _sum8 = __lsx_vfadd_s(_sum8, _c);
                    _sum9 = __lsx_vfadd_s(_sum9, _c);
                    _suma = __lsx_vfadd_s(_suma, _c);
                    _sumb = __lsx_vfadd_s(_sumb, _c);
                }
                if (broadcast_type_C == 3)
                {
                    // broadcast_type_C == 3 means full C matrix
                    // C is stored in topT_tile in pack_A_tile order: jj-major, 4 floats per jj
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
                        _sum8 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj + 32, 0), _sum8);
                        _sum9 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj + 36, 0), _sum9);
                        _suma = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj + 40, 0), _suma);
                        _sumb = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pCj + 44, 0), _sumb);
                    }
                    if (c_elempack == 1)
                    {
                        const float* pCj = (const float*)C + (i + ii) * c_hstep + (j + jj);
                        __m128* sums[12] = {&_sum0, &_sum1, &_sum2, &_sum3, &_sum4, &_sum5, &_sum6, &_sum7, &_sum8, &_sum9, &_suma, &_sumb};
                        for (int c = 0; c < 12; c++)
                        {
                            float tmp[4];
                            tmp[0] = pCj[0];
                            tmp[1] = pCj[c_hstep];
                            tmp[2] = pCj[c_hstep * 2];
                            tmp[3] = pCj[c_hstep * 3];
                            __m128 _cv = (__m128)__lsx_vld(tmp, 0);
                            *sums[c] = __lsx_vfmadd_s(_beta, _cv, *sums[c]);
                            pCj += 1;
                        }
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _sum0 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[0]), _sum0);
                    _sum1 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[1]), _sum1);
                    _sum2 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[2]), _sum2);
                    _sum3 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[3]), _sum3);
                    _sum4 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[4]), _sum4);
                    _sum5 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[5]), _sum5);
                    _sum6 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[6]), _sum6);
                    _sum7 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[7]), _sum7);
                    _sum8 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[8]), _sum8);
                    _sum9 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[9]), _sum9);
                    _suma = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[10]), _suma);
                    _sumb = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[11]), _sumb);
                    pCi += 12;
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
            _sum8 = __lsx_vfmul_s(_sum8, _valpha);
            _sum9 = __lsx_vfmul_s(_sum9, _valpha);
            _suma = __lsx_vfmul_s(_suma, _valpha);
            _sumb = __lsx_vfmul_s(_sumb, _valpha);

            if (output_transpose)
            {
                if (output_elemtype == 1)
                {
                    // fp32 output, transposed: row=j, col=i
                    float tmp0[4], tmp1[4], tmp2[4], tmp3[4];
                    float tmp4[4], tmp5[4], tmp6[4], tmp7[4];
                    float tmp8[4], tmp9[4], tmpa[4], tmpb[4];
                    __lsx_vst((__m128i)_sum0, tmp0, 0);
                    __lsx_vst((__m128i)_sum1, tmp1, 0);
                    __lsx_vst((__m128i)_sum2, tmp2, 0);
                    __lsx_vst((__m128i)_sum3, tmp3, 0);
                    __lsx_vst((__m128i)_sum4, tmp4, 0);
                    __lsx_vst((__m128i)_sum5, tmp5, 0);
                    __lsx_vst((__m128i)_sum6, tmp6, 0);
                    __lsx_vst((__m128i)_sum7, tmp7, 0);
                    __lsx_vst((__m128i)_sum8, tmp8, 0);
                    __lsx_vst((__m128i)_sum9, tmp9, 0);
                    __lsx_vst((__m128i)_suma, tmpa, 0);
                    __lsx_vst((__m128i)_sumb, tmpb, 0);
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
                    __lsx_vst((__m128i)_sum0, tmp0, 0);
                    __lsx_vst((__m128i)_sum1, tmp1, 0);
                    __lsx_vst((__m128i)_sum2, tmp2, 0);
                    __lsx_vst((__m128i)_sum3, tmp3, 0);
                    __lsx_vst((__m128i)_sum4, tmp4, 0);
                    __lsx_vst((__m128i)_sum5, tmp5, 0);
                    __lsx_vst((__m128i)_sum6, tmp6, 0);
                    __lsx_vst((__m128i)_sum7, tmp7, 0);
                    __lsx_vst((__m128i)_sum8, tmp8, 0);
                    __lsx_vst((__m128i)_sum9, tmp9, 0);
                    __lsx_vst((__m128i)_suma, tmpa, 0);
                    __lsx_vst((__m128i)_sumb, tmpb, 0);
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
                        __lsx_vst((__m128i)_sum0, p0f32, 0);
                        __lsx_vst((__m128i)_sum1, p0f32 + 4, 0);
                        __lsx_vst((__m128i)_sum2, p0f32 + 4 * 2, 0);
                        __lsx_vst((__m128i)_sum3, p0f32 + 4 * 3, 0);
                        __lsx_vst((__m128i)_sum4, p0f32 + 4 * 4, 0);
                        __lsx_vst((__m128i)_sum5, p0f32 + 4 * 5, 0);
                        __lsx_vst((__m128i)_sum6, p0f32 + 4 * 6, 0);
                        __lsx_vst((__m128i)_sum7, p0f32 + 4 * 7, 0);
                        __lsx_vst((__m128i)_sum8, p0f32 + 4 * 8, 0);
                        __lsx_vst((__m128i)_sum9, p0f32 + 4 * 9, 0);
                        __lsx_vst((__m128i)_suma, p0f32 + 4 * 10, 0);
                        __lsx_vst((__m128i)_sumb, p0f32 + 4 * 11, 0);
                        p0f32 += 48;
                    }
                    else
                    {
                        float2bfloat_lsx_store(_sum0, p0);
                        float2bfloat_lsx_store(_sum1, p0 + 4);
                        float2bfloat_lsx_store(_sum2, p0 + 4 * 2);
                        float2bfloat_lsx_store(_sum3, p0 + 4 * 3);
                        float2bfloat_lsx_store(_sum4, p0 + 4 * 4);
                        float2bfloat_lsx_store(_sum5, p0 + 4 * 5);
                        float2bfloat_lsx_store(_sum6, p0 + 4 * 6);
                        float2bfloat_lsx_store(_sum7, p0 + 4 * 7);
                        float2bfloat_lsx_store(_sum8, p0 + 4 * 8);
                        float2bfloat_lsx_store(_sum9, p0 + 4 * 9);
                        float2bfloat_lsx_store(_suma, p0 + 4 * 10);
                        float2bfloat_lsx_store(_sumb, p0 + 4 * 11);
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
                        __lsx_vst((__m128i)_sum0, p0f32, 0);
                        __lsx_vst((__m128i)_sum4, p0f32 + 4, 0);
                        __lsx_vst((__m128i)_sum8, p0f32 + 8, 0);
                        __lsx_vst((__m128i)_sum1, p0f32 + out_hstep, 0);
                        __lsx_vst((__m128i)_sum5, p0f32 + out_hstep + 4, 0);
                        __lsx_vst((__m128i)_sum9, p0f32 + out_hstep + 8, 0);
                        __lsx_vst((__m128i)_sum2, p0f32 + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_sum6, p0f32 + out_hstep * 2 + 4, 0);
                        __lsx_vst((__m128i)_suma, p0f32 + out_hstep * 2 + 8, 0);
                        __lsx_vst((__m128i)_sum3, p0f32 + out_hstep * 3, 0);
                        __lsx_vst((__m128i)_sum7, p0f32 + out_hstep * 3 + 4, 0);
                        __lsx_vst((__m128i)_sumb, p0f32 + out_hstep * 3 + 8, 0);
                        p0f32 += 12;
                    }
                    else
                    {
                        float2bfloat_lsx_store(_sum0, p0);
                        float2bfloat_lsx_store(_sum4, p0 + 4);
                        float2bfloat_lsx_store(_sum8, p0 + 8);
                        float2bfloat_lsx_store(_sum1, p0 + out_hstep);
                        float2bfloat_lsx_store(_sum5, p0 + out_hstep + 4);
                        float2bfloat_lsx_store(_sum9, p0 + out_hstep + 8);
                        float2bfloat_lsx_store(_sum2, p0 + out_hstep * 2);
                        float2bfloat_lsx_store(_sum6, p0 + out_hstep * 2 + 4);
                        float2bfloat_lsx_store(_suma, p0 + out_hstep * 2 + 8);
                        float2bfloat_lsx_store(_sum3, p0 + out_hstep * 3);
                        float2bfloat_lsx_store(_sum7, p0 + out_hstep * 3 + 4);
                        float2bfloat_lsx_store(_sumb, p0 + out_hstep * 3 + 8);
                        p0 += 12;
                    }
                }
            }
        }
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

            if (pCi)
            {
                __m128 _beta = __lsx_vreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    __m128 _c = __lsx_vfmul_s(__lsx_vreplfr2vr_s(pCi[0]), _beta);
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
                    __m128 _c = __lsx_vfmul_s((__m128)__lsx_vld(pCi, 0), _beta);
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
                        __m128 sums[8] = {_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7};
                        for (int c = 0; c < 8; c++)
                        {
                            float tmp[4];
                            tmp[0] = pCj[0];
                            tmp[1] = pCj[c_hstep];
                            tmp[2] = pCj[c_hstep * 2];
                            tmp[3] = pCj[c_hstep * 3];
                            __m128 _cv = (__m128)__lsx_vld(tmp, 0);
                            sums[c] = __lsx_vfmadd_s(_beta, _cv, sums[c]);
                            pCj += 1;
                        }
                        _sum0 = sums[0];
                        _sum1 = sums[1];
                        _sum2 = sums[2];
                        _sum3 = sums[3];
                        _sum4 = sums[4];
                        _sum5 = sums[5];
                        _sum6 = sums[6];
                        _sum7 = sums[7];
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _sum0 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[0]), _sum0);
                    _sum1 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[1]), _sum1);
                    _sum2 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[2]), _sum2);
                    _sum3 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[3]), _sum3);
                    _sum4 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[4]), _sum4);
                    _sum5 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[5]), _sum5);
                    _sum6 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[6]), _sum6);
                    _sum7 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[7]), _sum7);
                    pCi += 8;
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
                float tmp0[4], tmp1[4], tmp2[4], tmp3[4], tmp4[4], tmp5[4], tmp6[4], tmp7[4];
                __lsx_vst((__m128i)_sum0, tmp0, 0);
                __lsx_vst((__m128i)_sum1, tmp1, 0);
                __lsx_vst((__m128i)_sum2, tmp2, 0);
                __lsx_vst((__m128i)_sum3, tmp3, 0);
                __lsx_vst((__m128i)_sum4, tmp4, 0);
                __lsx_vst((__m128i)_sum5, tmp5, 0);
                __lsx_vst((__m128i)_sum6, tmp6, 0);
                __lsx_vst((__m128i)_sum7, tmp7, 0);
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
                        float2bfloat_lsx_store(_sum0, p0);
                        float2bfloat_lsx_store(_sum1, p0 + 4);
                        float2bfloat_lsx_store(_sum2, p0 + 8);
                        float2bfloat_lsx_store(_sum3, p0 + 12);
                        float2bfloat_lsx_store(_sum4, p0 + 16);
                        float2bfloat_lsx_store(_sum5, p0 + 20);
                        float2bfloat_lsx_store(_sum6, p0 + 24);
                        float2bfloat_lsx_store(_sum7, p0 + 28);
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
                        float2bfloat_lsx_store(_sum0, p0);
                        float2bfloat_lsx_store(_sum4, p0 + 4);
                        float2bfloat_lsx_store(_sum1, p0 + out_hstep);
                        float2bfloat_lsx_store(_sum5, p0 + out_hstep + 4);
                        float2bfloat_lsx_store(_sum2, p0 + out_hstep * 2);
                        float2bfloat_lsx_store(_sum6, p0 + out_hstep * 2 + 4);
                        float2bfloat_lsx_store(_sum3, p0 + out_hstep * 3);
                        float2bfloat_lsx_store(_sum7, p0 + out_hstep * 3 + 4);
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

            if (pCi)
            {
                __m128 _beta = __lsx_vreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    __m128 _c = __lsx_vfmul_s(__lsx_vreplfr2vr_s(pCi[0]), _beta);
                    _sum0 = __lsx_vfadd_s(_sum0, _c);
                    _sum1 = __lsx_vfadd_s(_sum1, _c);
                    _sum2 = __lsx_vfadd_s(_sum2, _c);
                    _sum3 = __lsx_vfadd_s(_sum3, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c = __lsx_vfmul_s((__m128)__lsx_vld(pCi, 0), _beta);
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
                        __m128 sums[4] = {_sum0, _sum1, _sum2, _sum3};
                        for (int c = 0; c < 4; c++)
                        {
                            float tmp[4];
                            tmp[0] = pCj[0];
                            tmp[1] = pCj[c_hstep];
                            tmp[2] = pCj[c_hstep * 2];
                            tmp[3] = pCj[c_hstep * 3];
                            __m128 _cv = (__m128)__lsx_vld(tmp, 0);
                            sums[c] = __lsx_vfmadd_s(_beta, _cv, sums[c]);
                            pCj += 1;
                        }
                        _sum0 = sums[0];
                        _sum1 = sums[1];
                        _sum2 = sums[2];
                        _sum3 = sums[3];
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _sum0 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[0]), _sum0);
                    _sum1 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[1]), _sum1);
                    _sum2 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[2]), _sum2);
                    _sum3 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[3]), _sum3);
                    pCi += 4;
                }
            }

            _sum0 = __lsx_vfmul_s(_sum0, _valpha);
            _sum1 = __lsx_vfmul_s(_sum1, _valpha);
            _sum2 = __lsx_vfmul_s(_sum2, _valpha);
            _sum3 = __lsx_vfmul_s(_sum3, _valpha);

            if (output_transpose)
            {
                float tmp0[4], tmp1[4], tmp2[4], tmp3[4];
                __lsx_vst((__m128i)_sum0, tmp0, 0);
                __lsx_vst((__m128i)_sum1, tmp1, 0);
                __lsx_vst((__m128i)_sum2, tmp2, 0);
                __lsx_vst((__m128i)_sum3, tmp3, 0);
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
                        __lsx_vst((__m128i)_sum0, p0f32, 0);
                        __lsx_vst((__m128i)_sum1, p0f32 + 4, 0);
                        __lsx_vst((__m128i)_sum2, p0f32 + 8, 0);
                        __lsx_vst((__m128i)_sum3, p0f32 + 12, 0);
                        p0f32 += 16;
                    }
                    else
                    {
                        float2bfloat_lsx_store(_sum0, p0);
                        float2bfloat_lsx_store(_sum1, p0 + 4);
                        float2bfloat_lsx_store(_sum2, p0 + 8);
                        float2bfloat_lsx_store(_sum3, p0 + 12);
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
                        float2bfloat_lsx_store(_sum0, p0);
                        float2bfloat_lsx_store(_sum1, p0 + out_hstep);
                        float2bfloat_lsx_store(_sum2, p0 + out_hstep * 2);
                        float2bfloat_lsx_store(_sum3, p0 + out_hstep * 3);
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

            if (pCi)
            {
                __m128 _beta = __lsx_vreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    __m128 _c = __lsx_vfmul_s(__lsx_vreplfr2vr_s(pCi[0]), _beta);
                    _sum0 = __lsx_vfadd_s(_sum0, _c);
                    _sum1 = __lsx_vfadd_s(_sum1, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c = __lsx_vfmul_s((__m128)__lsx_vld(pCi, 0), _beta);
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
                    _sum0 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[0]), _sum0);
                    _sum1 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[1]), _sum1);
                    pCi += 2;
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
                        float2bfloat_lsx_store(_sum0, p0);
                        float2bfloat_lsx_store(_sum1, p0 + 4);
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

            if (pCi)
            {
                __m128 _beta = __lsx_vreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    _sum0 = __lsx_vfadd_s(_sum0, __lsx_vfmul_s(__lsx_vreplfr2vr_s(pCi[0]), _beta));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _sum0 = __lsx_vfadd_s(_sum0, __lsx_vfmul_s((__m128)__lsx_vld(pCi, 0), _beta));
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
                    _sum0 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pCi[0]), _sum0);
                    pCi += 1;
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
                        float2bfloat_lsx_store(_sum0, p0);
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

#if __loongarch_sx
#if __loongarch_asx
    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_N = std::max(8, tile_size / 8 * 8);
    TILE_K = std::max(8, tile_size / 8 * 8);
#else
    TILE_M = std::max(4, tile_size / 4 * 4);
    TILE_N = std::max(4, tile_size / 4 * 4);
    TILE_K = std::max(4, tile_size / 4 * 4);
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
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
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
            TILE_M = std::max(4, tile_size / 4 * 4);
            TILE_N = std::max(4, tile_size / 4 * 4);
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
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
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
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
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
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 3) / 4 * 4);
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
        TILE_M = (constant_TILE_M + 3) / 4 * 4;
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
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
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
        TILE_K = (constant_TILE_K + 3) / 4 * 4;
#endif // __loongarch_asx
#else
        TILE_K = (constant_TILE_K + 1) / 2 * 2;
#endif
    }
}

// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static NCNN_FORCEINLINE signed char gemm_float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static void pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    signed char* pp = AT;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;
        const signed char* p2 = (const signed char*)A + (i + ii + 2) * A_hstep + k;
        const signed char* p3 = (const signed char*)A + (i + ii + 3) * A_hstep + k;
        const signed char* p4 = (const signed char*)A + (i + ii + 4) * A_hstep + k;
        const signed char* p5 = (const signed char*)A + (i + ii + 5) * A_hstep + k;
        const signed char* p6 = (const signed char*)A + (i + ii + 6) * A_hstep + k;
        const signed char* p7 = (const signed char*)A + (i + ii + 7) * A_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
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
            pp[12] = p3[0];
            pp[13] = p3[1];
            pp[14] = p3[2];
            pp[15] = p3[3];
            pp[16] = p4[0];
            pp[17] = p4[1];
            pp[18] = p4[2];
            pp[19] = p4[3];
            pp[20] = p5[0];
            pp[21] = p5[1];
            pp[22] = p5[2];
            pp[23] = p5[3];
            pp[24] = p6[0];
            pp[25] = p6[1];
            pp[26] = p6[2];
            pp[27] = p6[3];
            pp[28] = p7[0];
            pp[29] = p7[1];
            pp[30] = p7[2];
            pp[31] = p7[3];
            pp += 32;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
            p4 += 4;
            p5 += 4;
            p6 += 4;
            p7 += 4;
        }
        for (; kk < max_kk; kk++)
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
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;
        const signed char* p2 = (const signed char*)A + (i + ii + 2) * A_hstep + k;
        const signed char* p3 = (const signed char*)A + (i + ii + 3) * A_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
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
            pp[12] = p3[0];
            pp[13] = p3[1];
            pp[14] = p3[2];
            pp[15] = p3[3];
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < max_kk; kk++)
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
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
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
        for (; kk < max_kk; kk++)
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
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    signed char* pp = AT;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* p0 = (const signed char*)A + (size_t)k * A_hstep + i + ii;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            const signed char* p1 = p0 + A_hstep;
            const signed char* p2 = p0 + A_hstep * 2;
            const signed char* p3 = p0 + A_hstep * 3;

            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[4] = p0[1];
            pp[5] = p1[1];
            pp[6] = p2[1];
            pp[7] = p3[1];
            pp[8] = p0[2];
            pp[9] = p1[2];
            pp[10] = p2[2];
            pp[11] = p3[2];
            pp[12] = p0[3];
            pp[13] = p1[3];
            pp[14] = p2[3];
            pp[15] = p3[3];
            pp[16] = p0[4];
            pp[17] = p1[4];
            pp[18] = p2[4];
            pp[19] = p3[4];
            pp[20] = p0[5];
            pp[21] = p1[5];
            pp[22] = p2[5];
            pp[23] = p3[5];
            pp[24] = p0[6];
            pp[25] = p1[6];
            pp[26] = p2[6];
            pp[27] = p3[6];
            pp[28] = p0[7];
            pp[29] = p1[7];
            pp[30] = p2[7];
            pp[31] = p3[7];
            pp += 32;
            p0 += A_hstep * 4;
        }
        for (; kk < max_kk; kk++)
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
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* p0 = (const signed char*)A + (size_t)k * A_hstep + i + ii;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            const signed char* p1 = p0 + A_hstep;
            const signed char* p2 = p0 + A_hstep * 2;
            const signed char* p3 = p0 + A_hstep * 3;

            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[4] = p0[1];
            pp[5] = p1[1];
            pp[6] = p2[1];
            pp[7] = p3[1];
            pp[8] = p0[2];
            pp[9] = p1[2];
            pp[10] = p2[2];
            pp[11] = p3[2];
            pp[12] = p0[3];
            pp[13] = p1[3];
            pp[14] = p2[3];
            pp[15] = p3[3];
            pp += 16;
            p0 += A_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp += 4;
            p0 += A_hstep;
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* p0 = (const signed char*)A + (size_t)k * A_hstep + i + ii;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            const signed char* p1 = p0 + A_hstep;
            const signed char* p2 = p0 + A_hstep * 2;
            const signed char* p3 = p0 + A_hstep * 3;

            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[4] = p0[1];
            pp[5] = p1[1];
            pp[6] = p2[1];
            pp[7] = p3[1];
            pp += 8;
            p0 += A_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp += 2;
            p0 += A_hstep;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const signed char* p0 = (const signed char*)A + (size_t)k * A_hstep + i + ii;

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0 += A_hstep;
        }
    }
}

static void pack_B_tile_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    signed char* pp = BT;

    int jj = 0;
#if __loongarch_sx
    for (; jj + 7 < max_jj; jj += 8)
    {
        const signed char* p0 = (const signed char*)B + (j + jj) * B_hstep + k;
        const signed char* p1 = (const signed char*)B + (j + jj + 1) * B_hstep + k;
        const signed char* p2 = (const signed char*)B + (j + jj + 2) * B_hstep + k;
        const signed char* p3 = (const signed char*)B + (j + jj + 3) * B_hstep + k;
        const signed char* p4 = (const signed char*)B + (j + jj + 4) * B_hstep + k;
        const signed char* p5 = (const signed char*)B + (j + jj + 5) * B_hstep + k;
        const signed char* p6 = (const signed char*)B + (j + jj + 6) * B_hstep + k;
        const signed char* p7 = (const signed char*)B + (j + jj + 7) * B_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
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
            pp[12] = p3[0];
            pp[13] = p3[1];
            pp[14] = p3[2];
            pp[15] = p3[3];
            pp[16] = p4[0];
            pp[17] = p4[1];
            pp[18] = p4[2];
            pp[19] = p4[3];
            pp[20] = p5[0];
            pp[21] = p5[1];
            pp[22] = p5[2];
            pp[23] = p5[3];
            pp[24] = p6[0];
            pp[25] = p6[1];
            pp[26] = p6[2];
            pp[27] = p6[3];
            pp[28] = p7[0];
            pp[29] = p7[1];
            pp[30] = p7[2];
            pp[31] = p7[3];
            pp += 32;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
            p4 += 4;
            p5 += 4;
            p6 += 4;
            p7 += 4;
        }
        for (; kk < max_kk; kk++)
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
    for (; jj + 3 < max_jj; jj += 4)
    {
        const signed char* p0 = (const signed char*)B + (j + jj) * B_hstep + k;
        const signed char* p1 = (const signed char*)B + (j + jj + 1) * B_hstep + k;
        const signed char* p2 = (const signed char*)B + (j + jj + 2) * B_hstep + k;
        const signed char* p3 = (const signed char*)B + (j + jj + 3) * B_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
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
            pp[12] = p3[0];
            pp[13] = p3[1];
            pp[14] = p3[2];
            pp[15] = p3[3];
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < max_kk; kk++)
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
#endif // __loongarch_sx
    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = (const signed char*)B + (j + jj) * B_hstep + k;
        const signed char* p1 = (const signed char*)B + (j + jj + 1) * B_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
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
        for (; kk < max_kk; kk++)
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
        const signed char* p0 = (const signed char*)B + (j + jj) * B_hstep + k;

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_B_tile_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    signed char* pp = BT;

    int jj = 0;
#if __loongarch_sx
    for (; jj + 7 < max_jj; jj += 8)
    {
        const signed char* p0 = (const signed char*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            const signed char* p1 = p0 + B_hstep;
            const signed char* p2 = p0 + B_hstep * 2;
            const signed char* p3 = p0 + B_hstep * 3;

            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[4] = p0[1];
            pp[5] = p1[1];
            pp[6] = p2[1];
            pp[7] = p3[1];
            pp[8] = p0[2];
            pp[9] = p1[2];
            pp[10] = p2[2];
            pp[11] = p3[2];
            pp[12] = p0[3];
            pp[13] = p1[3];
            pp[14] = p2[3];
            pp[15] = p3[3];
            pp[16] = p0[4];
            pp[17] = p1[4];
            pp[18] = p2[4];
            pp[19] = p3[4];
            pp[20] = p0[5];
            pp[21] = p1[5];
            pp[22] = p2[5];
            pp[23] = p3[5];
            pp[24] = p0[6];
            pp[25] = p1[6];
            pp[26] = p2[6];
            pp[27] = p3[6];
            pp[28] = p0[7];
            pp[29] = p1[7];
            pp[30] = p2[7];
            pp[31] = p3[7];
            pp += 32;
            p0 += B_hstep * 4;
        }
        for (; kk < max_kk; kk++)
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
    for (; jj + 3 < max_jj; jj += 4)
    {
        const signed char* p0 = (const signed char*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            const signed char* p1 = p0 + B_hstep;
            const signed char* p2 = p0 + B_hstep * 2;
            const signed char* p3 = p0 + B_hstep * 3;

            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[4] = p0[1];
            pp[5] = p1[1];
            pp[6] = p2[1];
            pp[7] = p3[1];
            pp[8] = p0[2];
            pp[9] = p1[2];
            pp[10] = p2[2];
            pp[11] = p3[2];
            pp[12] = p0[3];
            pp[13] = p1[3];
            pp[14] = p2[3];
            pp[15] = p3[3];
            pp += 16;
            p0 += B_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp += 4;
            p0 += B_hstep;
        }
    }
#endif // __loongarch_sx
    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = (const signed char*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            const signed char* p1 = p0 + B_hstep;
            const signed char* p2 = p0 + B_hstep * 2;
            const signed char* p3 = p0 + B_hstep * 3;

            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[4] = p0[1];
            pp[5] = p1[1];
            pp[6] = p2[1];
            pp[7] = p3[1];
            pp += 8;
            p0 += B_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp += 2;
            p0 += B_hstep;
        }
    }
    for (; jj < max_jj; jj++)
    {
        const signed char* p0 = (const signed char*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0 += B_hstep;
        }
    }
}

static void compute_A_tile_int8_scales(const Mat& A, Mat& scales, float B_scale, Mat& out_descales, int i, int max_ii)
{
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    for (int ii = 0; ii < max_ii; ii++)
    {
        const float* ptr = (const float*)A + (i + ii) * A_hstep;

        float absmax = 0.f;
        for (int kk = 0; kk < A.w; kk++)
            absmax = std::max(absmax, (float)fabs(ptr[kk]));

        float scale = absmax == 0.f ? 1.f : 127.f / absmax;
        scales[i + ii] = scale;
        out_descales[i + ii] = 1.f / (scale * B_scale);
    }
}

static void transpose_compute_A_tile_int8_scales(const Mat& A, Mat& scales, float B_scale, Mat& out_descales, int i, int max_ii)
{
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
    const int K = A.dims == 3 ? A.c : A.h;

    for (int ii = 0; ii < max_ii; ii++)
    {
        const float* ptr = (const float*)A + i + ii;

        float absmax = 0.f;
        for (int kk = 0; kk < K; kk++)
        {
            absmax = std::max(absmax, (float)fabs(ptr[0]));
            ptr += A_hstep;
        }

        float scale = absmax == 0.f ? 1.f : 127.f / absmax;
        scales[i + ii] = scale;
        out_descales[i + ii] = 1.f / (scale * B_scale);
    }
}

static void pack_A_tile_fp32_to_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    signed char* pp = AT;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;
        const float* p4 = (const float*)A + (i + ii + 4) * A_hstep + k;
        const float* p5 = (const float*)A + (i + ii + 5) * A_hstep + k;
        const float* p6 = (const float*)A + (i + ii + 6) * A_hstep + k;
        const float* p7 = (const float*)A + (i + ii + 7) * A_hstep + k;
        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
        const float scale2 = scales[i + ii + 2];
        const float scale3 = scales[i + ii + 3];
        const float scale4 = scales[i + ii + 4];
        const float scale5 = scales[i + ii + 5];
        const float scale6 = scales[i + ii + 6];
        const float scale7 = scales[i + ii + 7];
        __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
        __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
        __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
        __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
        __m128 _scale4 = __lsx_vreplfr2vr_s(scale4);
        __m128 _scale5 = __lsx_vreplfr2vr_s(scale5);
        __m128 _scale6 = __lsx_vreplfr2vr_s(scale6);
        __m128 _scale7 = __lsx_vreplfr2vr_s(scale7);

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + 16);
            __builtin_prefetch(p1 + 16);
            __builtin_prefetch(p2 + 16);
            __builtin_prefetch(p3 + 16);
            __builtin_prefetch(p4 + 16);
            __builtin_prefetch(p5 + 16);
            __builtin_prefetch(p6 + 16);
            __builtin_prefetch(p7 + 16);
            __m128 _p0 = __lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _scale0);
            __m128 _p1 = __lsx_vfmul_s((__m128)__lsx_vld(p1, 0), _scale1);
            __m128 _p2 = __lsx_vfmul_s((__m128)__lsx_vld(p2, 0), _scale2);
            __m128 _p3 = __lsx_vfmul_s((__m128)__lsx_vld(p3, 0), _scale3);
            __m128 _p4 = __lsx_vfmul_s((__m128)__lsx_vld(p4, 0), _scale4);
            __m128 _p5 = __lsx_vfmul_s((__m128)__lsx_vld(p5, 0), _scale5);
            __m128 _p6 = __lsx_vfmul_s((__m128)__lsx_vld(p6, 0), _scale6);
            __m128 _p7 = __lsx_vfmul_s((__m128)__lsx_vld(p7, 0), _scale7);

            *((int64_t*)pp) = float2int8(_p0, _p1);
            *((int64_t*)(pp + 8)) = float2int8(_p2, _p3);
            *((int64_t*)(pp + 16)) = float2int8(_p4, _p5);
            *((int64_t*)(pp + 24)) = float2int8(_p6, _p7);
            pp += 32;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
            p4 += 4;
            p5 += 4;
            p6 += 4;
            p7 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp[1] = gemm_float2int8(p1[0] * scale1);
            pp[2] = gemm_float2int8(p2[0] * scale2);
            pp[3] = gemm_float2int8(p3[0] * scale3);
            pp[4] = gemm_float2int8(p4[0] * scale4);
            pp[5] = gemm_float2int8(p5[0] * scale5);
            pp[6] = gemm_float2int8(p6[0] * scale6);
            pp[7] = gemm_float2int8(p7[0] * scale7);
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
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;
        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
        const float scale2 = scales[i + ii + 2];
        const float scale3 = scales[i + ii + 3];
        __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
        __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
        __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
        __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + 16);
            __builtin_prefetch(p1 + 16);
            __builtin_prefetch(p2 + 16);
            __builtin_prefetch(p3 + 16);
            __m128 _p0 = __lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _scale0);
            __m128 _p1 = __lsx_vfmul_s((__m128)__lsx_vld(p1, 0), _scale1);
            __m128 _p2 = __lsx_vfmul_s((__m128)__lsx_vld(p2, 0), _scale2);
            __m128 _p3 = __lsx_vfmul_s((__m128)__lsx_vld(p3, 0), _scale3);

            *((int64_t*)pp) = float2int8(_p0, _p1);
            *((int64_t*)(pp + 8)) = float2int8(_p2, _p3);
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp[1] = gemm_float2int8(p1[0] * scale1);
            pp[2] = gemm_float2int8(p2[0] * scale2);
            pp[3] = gemm_float2int8(p3[0] * scale3);
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp[1] = gemm_float2int8(p0[1] * scale0);
            pp[2] = gemm_float2int8(p0[2] * scale0);
            pp[3] = gemm_float2int8(p0[3] * scale0);
            pp[4] = gemm_float2int8(p1[0] * scale1);
            pp[5] = gemm_float2int8(p1[1] * scale1);
            pp[6] = gemm_float2int8(p1[2] * scale1);
            pp[7] = gemm_float2int8(p1[3] * scale1);
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp[1] = gemm_float2int8(p1[0] * scale1);
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float scale0 = scales[i + ii];

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_A_tile_fp32_to_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    signed char* pp = AT;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;
        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
        const float scale2 = scales[i + ii + 2];
        const float scale3 = scales[i + ii + 3];
        const float scale4 = scales[i + ii + 4];
        const float scale5 = scales[i + ii + 5];
        const float scale6 = scales[i + ii + 6];
        const float scale7 = scales[i + ii + 7];
        __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
        __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
        __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
        __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);
        __m128 _scale4 = __lsx_vreplfr2vr_s(scale4);
        __m128 _scale5 = __lsx_vreplfr2vr_s(scale5);
        __m128 _scale6 = __lsx_vreplfr2vr_s(scale6);
        __m128 _scale7 = __lsx_vreplfr2vr_s(scale7);

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + A_hstep * 4);
            const float* p1 = p0 + A_hstep;
            const float* p2 = p1 + A_hstep;
            const float* p3 = p2 + A_hstep;
            __m128 _p0 = (__m128)__lsx_vld(p0, 0);
            __m128 _p1 = (__m128)__lsx_vld(p1, 0);
            __m128 _p2 = (__m128)__lsx_vld(p2, 0);
            __m128 _p3 = (__m128)__lsx_vld(p3, 0);
            transpose4x4_ps(_p0, _p1, _p2, _p3);
            _p0 = __lsx_vfmul_s(_p0, _scale0);
            _p1 = __lsx_vfmul_s(_p1, _scale1);
            _p2 = __lsx_vfmul_s(_p2, _scale2);
            _p3 = __lsx_vfmul_s(_p3, _scale3);

            __m128 _p4 = (__m128)__lsx_vld(p0 + 4, 0);
            __m128 _p5 = (__m128)__lsx_vld(p1 + 4, 0);
            __m128 _p6 = (__m128)__lsx_vld(p2 + 4, 0);
            __m128 _p7 = (__m128)__lsx_vld(p3 + 4, 0);
            transpose4x4_ps(_p4, _p5, _p6, _p7);
            _p4 = __lsx_vfmul_s(_p4, _scale4);
            _p5 = __lsx_vfmul_s(_p5, _scale5);
            _p6 = __lsx_vfmul_s(_p6, _scale6);
            _p7 = __lsx_vfmul_s(_p7, _scale7);

            *((int64_t*)pp) = float2int8(_p0, _p1);
            *((int64_t*)(pp + 8)) = float2int8(_p2, _p3);
            *((int64_t*)(pp + 16)) = float2int8(_p4, _p5);
            *((int64_t*)(pp + 24)) = float2int8(_p6, _p7);
            pp += 32;
            p0 += A_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp[1] = gemm_float2int8(p0[1] * scale1);
            pp[2] = gemm_float2int8(p0[2] * scale2);
            pp[3] = gemm_float2int8(p0[3] * scale3);
            pp[4] = gemm_float2int8(p0[4] * scale4);
            pp[5] = gemm_float2int8(p0[5] * scale5);
            pp[6] = gemm_float2int8(p0[6] * scale6);
            pp[7] = gemm_float2int8(p0[7] * scale7);
            pp += 8;
            p0 += A_hstep;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;
        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
        const float scale2 = scales[i + ii + 2];
        const float scale3 = scales[i + ii + 3];
        __m128 _scale0 = __lsx_vreplfr2vr_s(scale0);
        __m128 _scale1 = __lsx_vreplfr2vr_s(scale1);
        __m128 _scale2 = __lsx_vreplfr2vr_s(scale2);
        __m128 _scale3 = __lsx_vreplfr2vr_s(scale3);

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + A_hstep * 4);
            const float* p1 = p0 + A_hstep;
            const float* p2 = p1 + A_hstep;
            const float* p3 = p2 + A_hstep;
            __m128 _p0 = (__m128)__lsx_vld(p0, 0);
            __m128 _p1 = (__m128)__lsx_vld(p1, 0);
            __m128 _p2 = (__m128)__lsx_vld(p2, 0);
            __m128 _p3 = (__m128)__lsx_vld(p3, 0);
            transpose4x4_ps(_p0, _p1, _p2, _p3);
            _p0 = __lsx_vfmul_s(_p0, _scale0);
            _p1 = __lsx_vfmul_s(_p1, _scale1);
            _p2 = __lsx_vfmul_s(_p2, _scale2);
            _p3 = __lsx_vfmul_s(_p3, _scale3);

            *((int64_t*)pp) = float2int8(_p0, _p1);
            *((int64_t*)(pp + 8)) = float2int8(_p2, _p3);
            pp += 16;
            p0 += A_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp[1] = gemm_float2int8(p0[1] * scale1);
            pp[2] = gemm_float2int8(p0[2] * scale2);
            pp[3] = gemm_float2int8(p0[3] * scale3);
            pp += 4;
            p0 += A_hstep;
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;
        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            const float* p1 = p0 + A_hstep;
            const float* p2 = p0 + A_hstep * 2;
            const float* p3 = p0 + A_hstep * 3;

            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp[1] = gemm_float2int8(p1[0] * scale0);
            pp[2] = gemm_float2int8(p2[0] * scale0);
            pp[3] = gemm_float2int8(p3[0] * scale0);
            pp[4] = gemm_float2int8(p0[1] * scale1);
            pp[5] = gemm_float2int8(p1[1] * scale1);
            pp[6] = gemm_float2int8(p2[1] * scale1);
            pp[7] = gemm_float2int8(p3[1] * scale1);
            pp += 8;
            p0 += A_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp[1] = gemm_float2int8(p0[1] * scale1);
            pp += 2;
            p0 += A_hstep;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;
        const float scale0 = scales[i + ii];

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp += 1;
            p0 += A_hstep;
        }
    }
}

static void compute_B_int8_scale(const Mat& B, float& scale)
{
    float absmax = 0.f;

    const int H = B.dims == 3 ? B.c : B.h;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    for (int y = 0; y < H; y++)
    {
        const float* ptr = (const float*)B + y * B_hstep;
        for (int x = 0; x < B.w; x++)
            absmax = std::max(absmax, (float)fabs(ptr[x]));
    }

    scale = absmax == 0.f ? 1.f : 127.f / absmax;
}

static void pack_B_tile_fp32_to_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    signed char* pp = BT;

    int jj = 0;
#if __loongarch_sx
    __m128 _scale = __lsx_vreplfr2vr_s(scale);

    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
        const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
        const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;
        const float* p4 = (const float*)B + (j + jj + 4) * B_hstep + k;
        const float* p5 = (const float*)B + (j + jj + 5) * B_hstep + k;
        const float* p6 = (const float*)B + (j + jj + 6) * B_hstep + k;
        const float* p7 = (const float*)B + (j + jj + 7) * B_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + 16);
            __builtin_prefetch(p1 + 16);
            __builtin_prefetch(p2 + 16);
            __builtin_prefetch(p3 + 16);
            __builtin_prefetch(p4 + 16);
            __builtin_prefetch(p5 + 16);
            __builtin_prefetch(p6 + 16);
            __builtin_prefetch(p7 + 16);
            __m128 _p0 = __lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _scale);
            __m128 _p1 = __lsx_vfmul_s((__m128)__lsx_vld(p1, 0), _scale);
            __m128 _p2 = __lsx_vfmul_s((__m128)__lsx_vld(p2, 0), _scale);
            __m128 _p3 = __lsx_vfmul_s((__m128)__lsx_vld(p3, 0), _scale);
            __m128 _p4 = __lsx_vfmul_s((__m128)__lsx_vld(p4, 0), _scale);
            __m128 _p5 = __lsx_vfmul_s((__m128)__lsx_vld(p5, 0), _scale);
            __m128 _p6 = __lsx_vfmul_s((__m128)__lsx_vld(p6, 0), _scale);
            __m128 _p7 = __lsx_vfmul_s((__m128)__lsx_vld(p7, 0), _scale);

            *((int64_t*)pp) = float2int8(_p0, _p1);
            *((int64_t*)(pp + 8)) = float2int8(_p2, _p3);
            *((int64_t*)(pp + 16)) = float2int8(_p4, _p5);
            *((int64_t*)(pp + 24)) = float2int8(_p6, _p7);
            pp += 32;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
            p4 += 4;
            p5 += 4;
            p6 += 4;
            p7 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp[1] = gemm_float2int8(p1[0] * scale);
            pp[2] = gemm_float2int8(p2[0] * scale);
            pp[3] = gemm_float2int8(p3[0] * scale);
            pp[4] = gemm_float2int8(p4[0] * scale);
            pp[5] = gemm_float2int8(p5[0] * scale);
            pp[6] = gemm_float2int8(p6[0] * scale);
            pp[7] = gemm_float2int8(p7[0] * scale);
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
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
        const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
        const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + 16);
            __builtin_prefetch(p1 + 16);
            __builtin_prefetch(p2 + 16);
            __builtin_prefetch(p3 + 16);
            __m128 _p0 = __lsx_vfmul_s((__m128)__lsx_vld(p0, 0), _scale);
            __m128 _p1 = __lsx_vfmul_s((__m128)__lsx_vld(p1, 0), _scale);
            __m128 _p2 = __lsx_vfmul_s((__m128)__lsx_vld(p2, 0), _scale);
            __m128 _p3 = __lsx_vfmul_s((__m128)__lsx_vld(p3, 0), _scale);

            *((int64_t*)pp) = float2int8(_p0, _p1);
            *((int64_t*)(pp + 8)) = float2int8(_p2, _p3);
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp[1] = gemm_float2int8(p1[0] * scale);
            pp[2] = gemm_float2int8(p2[0] * scale);
            pp[3] = gemm_float2int8(p3[0] * scale);
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
    }
#endif // __loongarch_sx
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp[1] = gemm_float2int8(p0[1] * scale);
            pp[2] = gemm_float2int8(p0[2] * scale);
            pp[3] = gemm_float2int8(p0[3] * scale);
            pp[4] = gemm_float2int8(p1[0] * scale);
            pp[5] = gemm_float2int8(p1[1] * scale);
            pp[6] = gemm_float2int8(p1[2] * scale);
            pp[7] = gemm_float2int8(p1[3] * scale);
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp[1] = gemm_float2int8(p1[0] * scale);
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; jj < max_jj; jj++)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_B_tile_fp32_to_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    signed char* pp = BT;

    int jj = 0;
#if __loongarch_sx
    __m128 _scale = __lsx_vreplfr2vr_s(scale);

    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + B_hstep * 4);
            const float* p1 = p0 + B_hstep;
            const float* p2 = p1 + B_hstep;
            const float* p3 = p2 + B_hstep;
            __m128 _p0 = (__m128)__lsx_vld(p0, 0);
            __m128 _p1 = (__m128)__lsx_vld(p1, 0);
            __m128 _p2 = (__m128)__lsx_vld(p2, 0);
            __m128 _p3 = (__m128)__lsx_vld(p3, 0);
            transpose4x4_ps(_p0, _p1, _p2, _p3);
            _p0 = __lsx_vfmul_s(_p0, _scale);
            _p1 = __lsx_vfmul_s(_p1, _scale);
            _p2 = __lsx_vfmul_s(_p2, _scale);
            _p3 = __lsx_vfmul_s(_p3, _scale);

            __m128 _p4 = (__m128)__lsx_vld(p0 + 4, 0);
            __m128 _p5 = (__m128)__lsx_vld(p1 + 4, 0);
            __m128 _p6 = (__m128)__lsx_vld(p2 + 4, 0);
            __m128 _p7 = (__m128)__lsx_vld(p3 + 4, 0);
            transpose4x4_ps(_p4, _p5, _p6, _p7);
            _p4 = __lsx_vfmul_s(_p4, _scale);
            _p5 = __lsx_vfmul_s(_p5, _scale);
            _p6 = __lsx_vfmul_s(_p6, _scale);
            _p7 = __lsx_vfmul_s(_p7, _scale);

            *((int64_t*)pp) = float2int8(_p0, _p1);
            *((int64_t*)(pp + 8)) = float2int8(_p2, _p3);
            *((int64_t*)(pp + 16)) = float2int8(_p4, _p5);
            *((int64_t*)(pp + 24)) = float2int8(_p6, _p7);
            pp += 32;
            p0 += B_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp[1] = gemm_float2int8(p0[1] * scale);
            pp[2] = gemm_float2int8(p0[2] * scale);
            pp[3] = gemm_float2int8(p0[3] * scale);
            pp[4] = gemm_float2int8(p0[4] * scale);
            pp[5] = gemm_float2int8(p0[5] * scale);
            pp[6] = gemm_float2int8(p0[6] * scale);
            pp[7] = gemm_float2int8(p0[7] * scale);
            pp += 8;
            p0 += B_hstep;
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + B_hstep * 4);
            const float* p1 = p0 + B_hstep;
            const float* p2 = p1 + B_hstep;
            const float* p3 = p2 + B_hstep;
            __m128 _p0 = (__m128)__lsx_vld(p0, 0);
            __m128 _p1 = (__m128)__lsx_vld(p1, 0);
            __m128 _p2 = (__m128)__lsx_vld(p2, 0);
            __m128 _p3 = (__m128)__lsx_vld(p3, 0);
            transpose4x4_ps(_p0, _p1, _p2, _p3);
            _p0 = __lsx_vfmul_s(_p0, _scale);
            _p1 = __lsx_vfmul_s(_p1, _scale);
            _p2 = __lsx_vfmul_s(_p2, _scale);
            _p3 = __lsx_vfmul_s(_p3, _scale);

            *((int64_t*)pp) = float2int8(_p0, _p1);
            *((int64_t*)(pp + 8)) = float2int8(_p2, _p3);
            pp += 16;
            p0 += B_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp[1] = gemm_float2int8(p0[1] * scale);
            pp[2] = gemm_float2int8(p0[2] * scale);
            pp[3] = gemm_float2int8(p0[3] * scale);
            pp += 4;
            p0 += B_hstep;
        }
    }
#endif // __loongarch_sx
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            const float* p1 = p0 + B_hstep;
            const float* p2 = p0 + B_hstep * 2;
            const float* p3 = p0 + B_hstep * 3;

            pp[0] = gemm_float2int8(p0[0] * scale);
            pp[1] = gemm_float2int8(p1[0] * scale);
            pp[2] = gemm_float2int8(p2[0] * scale);
            pp[3] = gemm_float2int8(p3[0] * scale);
            pp[4] = gemm_float2int8(p0[1] * scale);
            pp[5] = gemm_float2int8(p1[1] * scale);
            pp[6] = gemm_float2int8(p2[1] * scale);
            pp[7] = gemm_float2int8(p3[1] * scale);
            pp += 8;
            p0 += B_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp[1] = gemm_float2int8(p0[1] * scale);
            pp += 2;
            p0 += B_hstep;
        }
    }
    for (; jj < max_jj; jj++)
    {
        const float* p0 = (const float*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp += 1;
            p0 += B_hstep;
        }
    }
}

static void gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
    // NCNN_LOGE("gemm_transB_packed_tile_int8 %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    // actually we only depend the global k==0 condition
    (void)i;
    (void)j;

    const signed char* pAT = AT_tile;
    const signed char* pBT = BT_tile;

    int* outptr = topT_tile;

    int ii = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* pB = pBT;

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

            __m256i _sum0;
            __m256i _sum1;
            __m256i _sum2;
            __m256i _sum3;
            __m256i _sum4;
            __m256i _sum5;
            __m256i _sum6;
            __m256i _sum7;

            if (k == 0)
            {
                _sum0 = __lasx_xvreplgr2vr_w(0);
                _sum1 = __lasx_xvreplgr2vr_w(0);
                _sum2 = __lasx_xvreplgr2vr_w(0);
                _sum3 = __lasx_xvreplgr2vr_w(0);
                _sum4 = __lasx_xvreplgr2vr_w(0);
                _sum5 = __lasx_xvreplgr2vr_w(0);
                _sum6 = __lasx_xvreplgr2vr_w(0);
                _sum7 = __lasx_xvreplgr2vr_w(0);
            }
            else
            {
                _sum0 = __lasx_xvld(outptr, 0);
                _sum1 = __lasx_xvld(outptr + 8, 0);
                _sum2 = __lasx_xvld(outptr + 16, 0);
                _sum3 = __lasx_xvld(outptr + 24, 0);
                _sum4 = __lasx_xvld(outptr + 32, 0);
                _sum5 = __lasx_xvld(outptr + 40, 0);
                _sum6 = __lasx_xvld(outptr + 48, 0);
                _sum7 = __lasx_xvld(outptr + 56, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256i _pA = __lasx_xvld(pA, 0);
                __m256i _pA1 = __lasx_xvshuf4i_w(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m256i _pB0 = __lasx_xvld(pB, 0);
                __m256i _pB1 = __lasx_xvshuf4i_w(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                __m256i _pB2 = __lasx_xvpermi_q(_pB0, _pB0, _LSX_SHUFFLE(0, 0, 0, 1));
                __m256i _pB3 = __lasx_xvshuf4i_w(_pB2, _LSX_SHUFFLE(0, 3, 2, 1));

                __m256i _s0 = __lasx_xvmulwev_h_b(_pA, _pB0);
                _s0 = __lasx_xvmaddwod_h_b(_s0, _pA, _pB0);
                _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s0, _s0));

                __m256i _s1 = __lasx_xvmulwev_h_b(_pA, _pB1);
                _s1 = __lasx_xvmaddwod_h_b(_s1, _pA, _pB1);
                _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvhaddw_w_h(_s1, _s1));

                __m256i _s2 = __lasx_xvmulwev_h_b(_pA1, _pB0);
                _s2 = __lasx_xvmaddwod_h_b(_s2, _pA1, _pB0);
                _sum2 = __lasx_xvadd_w(_sum2, __lasx_xvhaddw_w_h(_s2, _s2));

                __m256i _s3 = __lasx_xvmulwev_h_b(_pA1, _pB1);
                _s3 = __lasx_xvmaddwod_h_b(_s3, _pA1, _pB1);
                _sum3 = __lasx_xvadd_w(_sum3, __lasx_xvhaddw_w_h(_s3, _s3));

                __m256i _s4 = __lasx_xvmulwev_h_b(_pA, _pB2);
                _s4 = __lasx_xvmaddwod_h_b(_s4, _pA, _pB2);
                _sum4 = __lasx_xvadd_w(_sum4, __lasx_xvhaddw_w_h(_s4, _s4));

                __m256i _s5 = __lasx_xvmulwev_h_b(_pA, _pB3);
                _s5 = __lasx_xvmaddwod_h_b(_s5, _pA, _pB3);
                _sum5 = __lasx_xvadd_w(_sum5, __lasx_xvhaddw_w_h(_s5, _s5));

                __m256i _s6 = __lasx_xvmulwev_h_b(_pA1, _pB2);
                _s6 = __lasx_xvmaddwod_h_b(_s6, _pA1, _pB2);
                _sum6 = __lasx_xvadd_w(_sum6, __lasx_xvhaddw_w_h(_s6, _s6));

                __m256i _s7 = __lasx_xvmulwev_h_b(_pA1, _pB3);
                _s7 = __lasx_xvmaddwod_h_b(_s7, _pA1, _pB3);
                _sum7 = __lasx_xvadd_w(_sum7, __lasx_xvhaddw_w_h(_s7, _s7));

                pA += 32;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m256i _pA = __lasx_xvldrepl_d(pA, 0);
                _pA = __lasx_xvilvl_b(__lasx_xvslti_b(_pA, 0), _pA);
                __m256i _pA1 = __lasx_xvshuf4i_h(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m256i _pB0 = __lasx_xvldrepl_d(pB, 0);
                _pB0 = __lasx_xvilvl_b(__lasx_xvslti_b(_pB0, 0), _pB0);
                __m256i _pB1 = __lasx_xvshuf4i_h(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                __m256i _pB2 = __lasx_xvshuf4i_w(_pB0, _LSX_SHUFFLE(1, 0, 3, 2));
                __m256i _pB3 = __lasx_xvshuf4i_h(_pB2, _LSX_SHUFFLE(0, 3, 2, 1));

                __m256i _s0 = __lasx_xvmul_h(_pA, _pB0);
                _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(_s0));

                __m256i _s1 = __lasx_xvmul_h(_pA, _pB1);
                _sum1 = __lasx_xvadd_w(_sum1, __lasx_vext2xv_w_h(_s1));

                __m256i _s2 = __lasx_xvmul_h(_pA1, _pB0);
                _sum2 = __lasx_xvadd_w(_sum2, __lasx_vext2xv_w_h(_s2));

                __m256i _s3 = __lasx_xvmul_h(_pA1, _pB1);
                _sum3 = __lasx_xvadd_w(_sum3, __lasx_vext2xv_w_h(_s3));

                __m256i _s4 = __lasx_xvmul_h(_pA, _pB2);
                _sum4 = __lasx_xvadd_w(_sum4, __lasx_vext2xv_w_h(_s4));

                __m256i _s5 = __lasx_xvmul_h(_pA, _pB3);
                _sum5 = __lasx_xvadd_w(_sum5, __lasx_vext2xv_w_h(_s5));

                __m256i _s6 = __lasx_xvmul_h(_pA1, _pB2);
                _sum6 = __lasx_xvadd_w(_sum6, __lasx_vext2xv_w_h(_s6));

                __m256i _s7 = __lasx_xvmul_h(_pA1, _pB3);
                _sum7 = __lasx_xvadd_w(_sum7, __lasx_vext2xv_w_h(_s7));

                pA += 8;
                pB += 8;
            }

            __lasx_xvst(_sum0, outptr, 0);
            __lasx_xvst(_sum1, outptr + 8, 0);
            __lasx_xvst(_sum2, outptr + 16, 0);
            __lasx_xvst(_sum3, outptr + 24, 0);
            __lasx_xvst(_sum4, outptr + 32, 0);
            __lasx_xvst(_sum5, outptr + 40, 0);
            __lasx_xvst(_sum6, outptr + 48, 0);
            __lasx_xvst(_sum7, outptr + 56, 0);

            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            __m256i _sum0;
            __m256i _sum1;
            __m256i _sum2;
            __m256i _sum3;

            if (k == 0)
            {
                _sum0 = __lasx_xvreplgr2vr_w(0);
                _sum1 = __lasx_xvreplgr2vr_w(0);
                _sum2 = __lasx_xvreplgr2vr_w(0);
                _sum3 = __lasx_xvreplgr2vr_w(0);
            }
            else
            {
                _sum0 = __lasx_concat_128(__lsx_vld(outptr, 0), __lsx_vld(outptr + 4, 0));
                _sum1 = __lasx_concat_128(__lsx_vld(outptr + 8, 0), __lsx_vld(outptr + 12, 0));
                _sum2 = __lasx_concat_128(__lsx_vld(outptr + 16, 0), __lsx_vld(outptr + 20, 0));
                _sum3 = __lasx_concat_128(__lsx_vld(outptr + 24, 0), __lsx_vld(outptr + 28, 0));
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256i _pA = __lasx_xvld(pA, 0);
                __m256i _pA1 = __lasx_xvshuf4i_w(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128i _pBs = __lsx_vld(pB, 0);
                __m256i _pB = __lasx_concat_128(_pBs, _pBs);
                __m256i _pB1 = __lasx_xvshuf4i_w(_pB, _LSX_SHUFFLE(0, 3, 2, 1));

                __m256i _s0 = __lasx_xvmulwev_h_b(_pA, _pB);
                __m256i _s1 = __lasx_xvmulwev_h_b(_pA, _pB1);
                __m256i _s2 = __lasx_xvmulwev_h_b(_pA1, _pB);
                __m256i _s3 = __lasx_xvmulwev_h_b(_pA1, _pB1);

                _s0 = __lasx_xvmaddwod_h_b(_s0, _pA, _pB);
                _s1 = __lasx_xvmaddwod_h_b(_s1, _pA, _pB1);
                _s2 = __lasx_xvmaddwod_h_b(_s2, _pA1, _pB);
                _s3 = __lasx_xvmaddwod_h_b(_s3, _pA1, _pB1);

                _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s0, _s0));
                _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvhaddw_w_h(_s1, _s1));
                _sum2 = __lasx_xvadd_w(_sum2, __lasx_xvhaddw_w_h(_s2, _s2));
                _sum3 = __lasx_xvadd_w(_sum3, __lasx_xvhaddw_w_h(_s3, _s3));

                pA += 32;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m256i _pA = __lasx_xvldrepl_d(pA, 0);
                _pA = __lasx_xvilvl_b(__lasx_xvslti_b(_pA, 0), _pA);
                __m256i _pA1 = __lasx_xvshuf4i_h(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m256i _pB = __lasx_xvldrepl_w(pB, 0);
                _pB = __lasx_xvilvl_b(__lasx_xvslti_b(_pB, 0), _pB);
                __m256i _pB1 = __lasx_xvshuf4i_h(_pB, _LSX_SHUFFLE(0, 3, 2, 1));

                __m256i _s0 = __lasx_xvmul_h(_pA, _pB);
                __m256i _s1 = __lasx_xvmul_h(_pA, _pB1);
                __m256i _s2 = __lasx_xvmul_h(_pA1, _pB);
                __m256i _s3 = __lasx_xvmul_h(_pA1, _pB1);

                _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(_s0));
                _sum1 = __lasx_xvadd_w(_sum1, __lasx_vext2xv_w_h(_s1));
                _sum2 = __lasx_xvadd_w(_sum2, __lasx_vext2xv_w_h(_s2));
                _sum3 = __lasx_xvadd_w(_sum3, __lasx_vext2xv_w_h(_s3));

                pA += 8;
                pB += 4;
            }

            __lsx_vst(__lasx_extract_128_lo(_sum0), outptr, 0);
            __lsx_vst(__lasx_extract_128_hi(_sum0), outptr + 4, 0);
            __lsx_vst(__lasx_extract_128_lo(_sum1), outptr + 8, 0);
            __lsx_vst(__lasx_extract_128_hi(_sum1), outptr + 12, 0);
            __lsx_vst(__lasx_extract_128_lo(_sum2), outptr + 16, 0);
            __lsx_vst(__lasx_extract_128_hi(_sum2), outptr + 20, 0);
            __lsx_vst(__lasx_extract_128_lo(_sum3), outptr + 24, 0);
            __lsx_vst(__lasx_extract_128_hi(_sum3), outptr + 28, 0);

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            __m256i _sum0;
            __m256i _sum1;

            if (k == 0)
            {
                _sum0 = __lasx_xvreplgr2vr_w(0);
                _sum1 = __lasx_xvreplgr2vr_w(0);
            }
            else
            {
                _sum0 = __lasx_concat_128(__lsx_vld(outptr, 0), __lsx_vld(outptr + 4, 0));
                _sum1 = __lasx_concat_128(__lsx_vld(outptr + 8, 0), __lsx_vld(outptr + 12, 0));
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256i _pA = __lasx_xvld(pA, 0);

                __m128i _pBs = __lsx_vldrepl_d(pB, 0);
                __m256i _pB = __lasx_concat_128(_pBs, _pBs);
                __m256i _pB1 = __lasx_xvshuf4i_w(_pB, _LSX_SHUFFLE(2, 3, 0, 1));

                __m256i _s0 = __lasx_xvmulwev_h_b(_pA, _pB);
                __m256i _s1 = __lasx_xvmulwev_h_b(_pA, _pB1);

                _s0 = __lasx_xvmaddwod_h_b(_s0, _pA, _pB);
                _s1 = __lasx_xvmaddwod_h_b(_s1, _pA, _pB1);

                _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s0, _s0));
                _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvhaddw_w_h(_s1, _s1));

                pA += 32;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m256i _pA = __lasx_xvldrepl_d(pA, 0);
                _pA = __lasx_xvilvl_b(__lasx_xvslti_b(_pA, 0), _pA);
                int b01 = (unsigned char)pB[0] | ((unsigned char)pB[1] << 8);
                __m256i _pB = __lasx_xvreplgr2vr_w(b01);
                _pB = __lasx_xvilvl_b(__lasx_xvslti_b(_pB, 0), _pB);
                _pB = __lasx_xvshuf4i_h(_pB, _LSX_SHUFFLE(1, 0, 1, 0));
                __m256i _pB1 = __lasx_xvshuf4i_h(_pB, _LSX_SHUFFLE(0, 3, 2, 1));

                __m256i _s0 = __lasx_xvmul_h(_pA, _pB);
                __m256i _s1 = __lasx_xvmul_h(_pA, _pB1);

                _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(_s0));
                _sum1 = __lasx_xvadd_w(_sum1, __lasx_vext2xv_w_h(_s1));

                pA += 8;
                pB += 2;
            }

            __lsx_vst(__lasx_extract_128_lo(_sum0), outptr, 0);
            __lsx_vst(__lasx_extract_128_hi(_sum0), outptr + 4, 0);
            __lsx_vst(__lasx_extract_128_lo(_sum1), outptr + 8, 0);
            __lsx_vst(__lasx_extract_128_hi(_sum1), outptr + 12, 0);

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            __m256i _sum0;

            if (k == 0)
            {
                _sum0 = __lasx_xvreplgr2vr_w(0);
            }
            else
            {
                _sum0 = __lasx_xvld(outptr, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256i _pA = __lasx_xvld(pA, 0);
                __m256i _pB0 = __lasx_xvldrepl_w(pB, 0);

                __m256i _s0 = __lasx_xvmulwev_h_b(_pA, _pB0);
                _s0 = __lasx_xvmaddwod_h_b(_s0, _pA, _pB0);

                _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvhaddw_w_h(_s0, _s0));

                pA += 32;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m256i _pA = __lasx_xvldrepl_d(pA, 0);
                _pA = __lasx_xvilvl_b(__lasx_xvslti_b(_pA, 0), _pA);

                __m256i _pB0 = __lasx_xvreplgr2vr_h(pB[0]);

                __m256i _s0 = __lasx_xvmul_h(_pA, _pB0);

                _sum0 = __lasx_xvadd_w(_sum0, __lasx_vext2xv_w_h(_s0));

                pA += 8;
                pB += 1;
            }

            __lasx_xvst(_sum0, outptr, 0);

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
#else  // __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* pB = pBT;

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

            __m128i _sum00;
            __m128i _sum01;
            __m128i _sum10;
            __m128i _sum11;
            __m128i _sum20;
            __m128i _sum21;
            __m128i _sum30;
            __m128i _sum31;
            __m128i _sum40;
            __m128i _sum41;
            __m128i _sum50;
            __m128i _sum51;
            __m128i _sum60;
            __m128i _sum61;
            __m128i _sum70;
            __m128i _sum71;

            if (k == 0)
            {
                _sum00 = __lsx_vreplgr2vr_w(0);
                _sum01 = __lsx_vreplgr2vr_w(0);
                _sum10 = __lsx_vreplgr2vr_w(0);
                _sum11 = __lsx_vreplgr2vr_w(0);
                _sum20 = __lsx_vreplgr2vr_w(0);
                _sum21 = __lsx_vreplgr2vr_w(0);
                _sum30 = __lsx_vreplgr2vr_w(0);
                _sum31 = __lsx_vreplgr2vr_w(0);
                _sum40 = __lsx_vreplgr2vr_w(0);
                _sum41 = __lsx_vreplgr2vr_w(0);
                _sum50 = __lsx_vreplgr2vr_w(0);
                _sum51 = __lsx_vreplgr2vr_w(0);
                _sum60 = __lsx_vreplgr2vr_w(0);
                _sum61 = __lsx_vreplgr2vr_w(0);
                _sum70 = __lsx_vreplgr2vr_w(0);
                _sum71 = __lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum00 = __lsx_vld(outptr, 0);
                _sum01 = __lsx_vld(outptr + 4, 0);
                _sum10 = __lsx_vld(outptr + 8, 0);
                _sum11 = __lsx_vld(outptr + 12, 0);
                _sum20 = __lsx_vld(outptr + 16, 0);
                _sum21 = __lsx_vld(outptr + 20, 0);
                _sum30 = __lsx_vld(outptr + 24, 0);
                _sum31 = __lsx_vld(outptr + 28, 0);
                _sum40 = __lsx_vld(outptr + 32, 0);
                _sum41 = __lsx_vld(outptr + 36, 0);
                _sum50 = __lsx_vld(outptr + 40, 0);
                _sum51 = __lsx_vld(outptr + 44, 0);
                _sum60 = __lsx_vld(outptr + 48, 0);
                _sum61 = __lsx_vld(outptr + 52, 0);
                _sum70 = __lsx_vld(outptr + 56, 0);
                _sum71 = __lsx_vld(outptr + 60, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA0 = __lsx_vld(pA, 0);
                __m128i _pA1 = __lsx_vld(pA + 16, 0);
                __m128i _pA0r = __lsx_vshuf4i_w(_pA0, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128i _pA1r = __lsx_vshuf4i_w(_pA1, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128i _pB0 = __lsx_vld(pB, 0);
                __m128i _pB1 = __lsx_vld(pB + 16, 0);
                __m128i _pB0r = __lsx_vshuf4i_w(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                __m128i _pB1r = __lsx_vshuf4i_w(_pB1, _LSX_SHUFFLE(0, 3, 2, 1));

                __m128i _s0 = __lsx_vmulwev_h_b(_pA0, _pB0);
                __m128i _s1 = __lsx_vmulwev_h_b(_pA1, _pB0);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA0, _pB0);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA1, _pB0);
                _sum00 = __lsx_vadd_w(_sum00, __lsx_vhaddw_w_h(_s0, _s0));
                _sum01 = __lsx_vadd_w(_sum01, __lsx_vhaddw_w_h(_s1, _s1));

                _s0 = __lsx_vmulwev_h_b(_pA0, _pB0r);
                _s1 = __lsx_vmulwev_h_b(_pA1, _pB0r);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA0, _pB0r);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA1, _pB0r);
                _sum10 = __lsx_vadd_w(_sum10, __lsx_vhaddw_w_h(_s0, _s0));
                _sum11 = __lsx_vadd_w(_sum11, __lsx_vhaddw_w_h(_s1, _s1));

                _s0 = __lsx_vmulwev_h_b(_pA0r, _pB0);
                _s1 = __lsx_vmulwev_h_b(_pA1r, _pB0);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA0r, _pB0);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA1r, _pB0);
                _sum20 = __lsx_vadd_w(_sum20, __lsx_vhaddw_w_h(_s0, _s0));
                _sum21 = __lsx_vadd_w(_sum21, __lsx_vhaddw_w_h(_s1, _s1));

                _s0 = __lsx_vmulwev_h_b(_pA0r, _pB0r);
                _s1 = __lsx_vmulwev_h_b(_pA1r, _pB0r);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA0r, _pB0r);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA1r, _pB0r);
                _sum30 = __lsx_vadd_w(_sum30, __lsx_vhaddw_w_h(_s0, _s0));
                _sum31 = __lsx_vadd_w(_sum31, __lsx_vhaddw_w_h(_s1, _s1));

                _s0 = __lsx_vmulwev_h_b(_pA0, _pB1);
                _s1 = __lsx_vmulwev_h_b(_pA1, _pB1);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA0, _pB1);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA1, _pB1);
                _sum40 = __lsx_vadd_w(_sum40, __lsx_vhaddw_w_h(_s0, _s0));
                _sum41 = __lsx_vadd_w(_sum41, __lsx_vhaddw_w_h(_s1, _s1));

                _s0 = __lsx_vmulwev_h_b(_pA0, _pB1r);
                _s1 = __lsx_vmulwev_h_b(_pA1, _pB1r);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA0, _pB1r);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA1, _pB1r);
                _sum50 = __lsx_vadd_w(_sum50, __lsx_vhaddw_w_h(_s0, _s0));
                _sum51 = __lsx_vadd_w(_sum51, __lsx_vhaddw_w_h(_s1, _s1));

                _s0 = __lsx_vmulwev_h_b(_pA0r, _pB1);
                _s1 = __lsx_vmulwev_h_b(_pA1r, _pB1);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA0r, _pB1);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA1r, _pB1);
                _sum60 = __lsx_vadd_w(_sum60, __lsx_vhaddw_w_h(_s0, _s0));
                _sum61 = __lsx_vadd_w(_sum61, __lsx_vhaddw_w_h(_s1, _s1));

                _s0 = __lsx_vmulwev_h_b(_pA0r, _pB1r);
                _s1 = __lsx_vmulwev_h_b(_pA1r, _pB1r);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA0r, _pB1r);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA1r, _pB1r);
                _sum70 = __lsx_vadd_w(_sum70, __lsx_vhaddw_w_h(_s0, _s0));
                _sum71 = __lsx_vadd_w(_sum71, __lsx_vhaddw_w_h(_s1, _s1));

                pA += 32;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA0 = __lsx_vldrepl_w(pA, 0);
                _pA0 = __lsx_vilvl_b(__lsx_vslti_b(_pA0, 0), _pA0);
                __m128i _pA1 = __lsx_vldrepl_w(pA + 4, 0);
                _pA1 = __lsx_vilvl_b(__lsx_vslti_b(_pA1, 0), _pA1);
                __m128i _pA0r = __lsx_vshuf4i_h(_pA0, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128i _pA1r = __lsx_vshuf4i_h(_pA1, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128i _pB0 = __lsx_vldrepl_w(pB, 0);
                _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                __m128i _pB1 = __lsx_vldrepl_w(pB + 4, 0);
                _pB1 = __lsx_vilvl_b(__lsx_vslti_b(_pB1, 0), _pB1);
                __m128i _pB0r = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                __m128i _pB1r = __lsx_vshuf4i_h(_pB1, _LSX_SHUFFLE(0, 3, 2, 1));

                __m128i _s0 = __lsx_vmul_h(_pA0, _pB0);
                __m128i _s1 = __lsx_vmul_h(_pA1, _pB0);
                _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                _s0 = __lsx_vmul_h(_pA0, _pB0r);
                _s1 = __lsx_vmul_h(_pA1, _pB0r);
                _sum10 = __lsx_vadd_w(_sum10, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum11 = __lsx_vadd_w(_sum11, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                _s0 = __lsx_vmul_h(_pA0r, _pB0);
                _s1 = __lsx_vmul_h(_pA1r, _pB0);
                _sum20 = __lsx_vadd_w(_sum20, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum21 = __lsx_vadd_w(_sum21, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                _s0 = __lsx_vmul_h(_pA0r, _pB0r);
                _s1 = __lsx_vmul_h(_pA1r, _pB0r);
                _sum30 = __lsx_vadd_w(_sum30, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum31 = __lsx_vadd_w(_sum31, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                _s0 = __lsx_vmul_h(_pA0, _pB1);
                _s1 = __lsx_vmul_h(_pA1, _pB1);
                _sum40 = __lsx_vadd_w(_sum40, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum41 = __lsx_vadd_w(_sum41, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                _s0 = __lsx_vmul_h(_pA0, _pB1r);
                _s1 = __lsx_vmul_h(_pA1, _pB1r);
                _sum50 = __lsx_vadd_w(_sum50, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum51 = __lsx_vadd_w(_sum51, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                _s0 = __lsx_vmul_h(_pA0r, _pB1);
                _s1 = __lsx_vmul_h(_pA1r, _pB1);
                _sum60 = __lsx_vadd_w(_sum60, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum61 = __lsx_vadd_w(_sum61, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                _s0 = __lsx_vmul_h(_pA0r, _pB1r);
                _s1 = __lsx_vmul_h(_pA1r, _pB1r);
                _sum70 = __lsx_vadd_w(_sum70, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum71 = __lsx_vadd_w(_sum71, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                pA += 8;
                pB += 8;
            }

            __lsx_vst(_sum00, outptr, 0);
            __lsx_vst(_sum01, outptr + 4, 0);
            __lsx_vst(_sum10, outptr + 8, 0);
            __lsx_vst(_sum11, outptr + 12, 0);
            __lsx_vst(_sum20, outptr + 16, 0);
            __lsx_vst(_sum21, outptr + 20, 0);
            __lsx_vst(_sum30, outptr + 24, 0);
            __lsx_vst(_sum31, outptr + 28, 0);
            __lsx_vst(_sum40, outptr + 32, 0);
            __lsx_vst(_sum41, outptr + 36, 0);
            __lsx_vst(_sum50, outptr + 40, 0);
            __lsx_vst(_sum51, outptr + 44, 0);
            __lsx_vst(_sum60, outptr + 48, 0);
            __lsx_vst(_sum61, outptr + 52, 0);
            __lsx_vst(_sum70, outptr + 56, 0);
            __lsx_vst(_sum71, outptr + 60, 0);

            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            __m128i _sum00;
            __m128i _sum01;
            __m128i _sum10;
            __m128i _sum11;
            __m128i _sum20;
            __m128i _sum21;
            __m128i _sum30;
            __m128i _sum31;

            if (k == 0)
            {
                _sum00 = __lsx_vreplgr2vr_w(0);
                _sum01 = __lsx_vreplgr2vr_w(0);
                _sum10 = __lsx_vreplgr2vr_w(0);
                _sum11 = __lsx_vreplgr2vr_w(0);
                _sum20 = __lsx_vreplgr2vr_w(0);
                _sum21 = __lsx_vreplgr2vr_w(0);
                _sum30 = __lsx_vreplgr2vr_w(0);
                _sum31 = __lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum00 = __lsx_vld(outptr, 0);
                _sum01 = __lsx_vld(outptr + 4, 0);
                _sum10 = __lsx_vld(outptr + 8, 0);
                _sum11 = __lsx_vld(outptr + 12, 0);
                _sum20 = __lsx_vld(outptr + 16, 0);
                _sum21 = __lsx_vld(outptr + 20, 0);
                _sum30 = __lsx_vld(outptr + 24, 0);
                _sum31 = __lsx_vld(outptr + 28, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA0 = __lsx_vld(pA, 0);
                __m128i _pA1 = __lsx_vld(pA + 16, 0);
                __m128i _pA0r = __lsx_vshuf4i_w(_pA0, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128i _pA1r = __lsx_vshuf4i_w(_pA1, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128i _pB0 = __lsx_vld(pB, 0);
                __m128i _pB0r = __lsx_vshuf4i_w(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));

                __m128i _s0 = __lsx_vmulwev_h_b(_pA0, _pB0);
                __m128i _s1 = __lsx_vmulwev_h_b(_pA1, _pB0);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA0, _pB0);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA1, _pB0);
                _sum00 = __lsx_vadd_w(_sum00, __lsx_vhaddw_w_h(_s0, _s0));
                _sum01 = __lsx_vadd_w(_sum01, __lsx_vhaddw_w_h(_s1, _s1));

                _s0 = __lsx_vmulwev_h_b(_pA0, _pB0r);
                _s1 = __lsx_vmulwev_h_b(_pA1, _pB0r);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA0, _pB0r);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA1, _pB0r);
                _sum10 = __lsx_vadd_w(_sum10, __lsx_vhaddw_w_h(_s0, _s0));
                _sum11 = __lsx_vadd_w(_sum11, __lsx_vhaddw_w_h(_s1, _s1));

                _s0 = __lsx_vmulwev_h_b(_pA0r, _pB0);
                _s1 = __lsx_vmulwev_h_b(_pA1r, _pB0);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA0r, _pB0);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA1r, _pB0);
                _sum20 = __lsx_vadd_w(_sum20, __lsx_vhaddw_w_h(_s0, _s0));
                _sum21 = __lsx_vadd_w(_sum21, __lsx_vhaddw_w_h(_s1, _s1));

                _s0 = __lsx_vmulwev_h_b(_pA0r, _pB0r);
                _s1 = __lsx_vmulwev_h_b(_pA1r, _pB0r);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA0r, _pB0r);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA1r, _pB0r);
                _sum30 = __lsx_vadd_w(_sum30, __lsx_vhaddw_w_h(_s0, _s0));
                _sum31 = __lsx_vadd_w(_sum31, __lsx_vhaddw_w_h(_s1, _s1));

                pA += 32;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA0 = __lsx_vldrepl_w(pA, 0);
                _pA0 = __lsx_vilvl_b(__lsx_vslti_b(_pA0, 0), _pA0);
                __m128i _pA1 = __lsx_vldrepl_w(pA + 4, 0);
                _pA1 = __lsx_vilvl_b(__lsx_vslti_b(_pA1, 0), _pA1);
                __m128i _pA0r = __lsx_vshuf4i_h(_pA0, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128i _pA1r = __lsx_vshuf4i_h(_pA1, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128i _pB0 = __lsx_vldrepl_w(pB, 0);
                _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                __m128i _pB0r = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));

                __m128i _s0 = __lsx_vmul_h(_pA0, _pB0);
                __m128i _s1 = __lsx_vmul_h(_pA1, _pB0);
                _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                _s0 = __lsx_vmul_h(_pA0, _pB0r);
                _s1 = __lsx_vmul_h(_pA1, _pB0r);
                _sum10 = __lsx_vadd_w(_sum10, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum11 = __lsx_vadd_w(_sum11, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                _s0 = __lsx_vmul_h(_pA0r, _pB0);
                _s1 = __lsx_vmul_h(_pA1r, _pB0);
                _sum20 = __lsx_vadd_w(_sum20, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum21 = __lsx_vadd_w(_sum21, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                _s0 = __lsx_vmul_h(_pA0r, _pB0r);
                _s1 = __lsx_vmul_h(_pA1r, _pB0r);
                _sum30 = __lsx_vadd_w(_sum30, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum31 = __lsx_vadd_w(_sum31, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                pA += 8;
                pB += 4;
            }

            __lsx_vst(_sum00, outptr, 0);
            __lsx_vst(_sum01, outptr + 4, 0);
            __lsx_vst(_sum10, outptr + 8, 0);
            __lsx_vst(_sum11, outptr + 12, 0);
            __lsx_vst(_sum20, outptr + 16, 0);
            __lsx_vst(_sum21, outptr + 20, 0);
            __lsx_vst(_sum30, outptr + 24, 0);
            __lsx_vst(_sum31, outptr + 28, 0);

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            __m128i _sum00;
            __m128i _sum01;
            __m128i _sum10;
            __m128i _sum11;

            if (k == 0)
            {
                _sum00 = __lsx_vreplgr2vr_w(0);
                _sum01 = __lsx_vreplgr2vr_w(0);
                _sum10 = __lsx_vreplgr2vr_w(0);
                _sum11 = __lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum00 = __lsx_vld(outptr, 0);
                _sum01 = __lsx_vld(outptr + 4, 0);
                _sum10 = __lsx_vld(outptr + 8, 0);
                _sum11 = __lsx_vld(outptr + 12, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA0 = __lsx_vld(pA, 0);
                __m128i _pA1 = __lsx_vld(pA + 16, 0);

                __m128i _pB0 = __lsx_vldrepl_d(pB, 0);
                __m128i _pB1 = __lsx_vshuf4i_w(_pB0, _LSX_SHUFFLE(2, 3, 0, 1));

                __m128i _s0 = __lsx_vmulwev_h_b(_pA0, _pB0);
                __m128i _s1 = __lsx_vmulwev_h_b(_pA1, _pB0);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA0, _pB0);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA1, _pB0);
                _sum00 = __lsx_vadd_w(_sum00, __lsx_vhaddw_w_h(_s0, _s0));
                _sum01 = __lsx_vadd_w(_sum01, __lsx_vhaddw_w_h(_s1, _s1));

                _s0 = __lsx_vmulwev_h_b(_pA0, _pB1);
                _s1 = __lsx_vmulwev_h_b(_pA1, _pB1);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA0, _pB1);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA1, _pB1);
                _sum10 = __lsx_vadd_w(_sum10, __lsx_vhaddw_w_h(_s0, _s0));
                _sum11 = __lsx_vadd_w(_sum11, __lsx_vhaddw_w_h(_s1, _s1));

                pA += 32;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA0 = __lsx_vldrepl_w(pA, 0);
                _pA0 = __lsx_vilvl_b(__lsx_vslti_b(_pA0, 0), _pA0);
                __m128i _pA1 = __lsx_vldrepl_w(pA + 4, 0);
                _pA1 = __lsx_vilvl_b(__lsx_vslti_b(_pA1, 0), _pA1);

                int b01 = (unsigned char)pB[0] | ((unsigned char)pB[1] << 8);
                __m128i _pB0 = __lsx_vreplgr2vr_w(b01);
                _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                _pB0 = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(1, 0, 1, 0));
                __m128i _pB1 = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));

                __m128i _s0 = __lsx_vmul_h(_pA0, _pB0);
                __m128i _s1 = __lsx_vmul_h(_pA1, _pB0);
                _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                _s0 = __lsx_vmul_h(_pA0, _pB1);
                _s1 = __lsx_vmul_h(_pA1, _pB1);
                _sum10 = __lsx_vadd_w(_sum10, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum11 = __lsx_vadd_w(_sum11, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                pA += 8;
                pB += 2;
            }

            __lsx_vst(_sum00, outptr, 0);
            __lsx_vst(_sum01, outptr + 4, 0);
            __lsx_vst(_sum10, outptr + 8, 0);
            __lsx_vst(_sum11, outptr + 12, 0);

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            __m128i _sum00;
            __m128i _sum01;

            if (k == 0)
            {
                _sum00 = __lsx_vreplgr2vr_w(0);
                _sum01 = __lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum00 = __lsx_vld(outptr, 0);
                _sum01 = __lsx_vld(outptr + 4, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA0 = __lsx_vld(pA, 0);
                __m128i _pA1 = __lsx_vld(pA + 16, 0);
                __m128i _pB0 = __lsx_vldrepl_w(pB, 0);

                __m128i _s0 = __lsx_vmulwev_h_b(_pA0, _pB0);
                __m128i _s1 = __lsx_vmulwev_h_b(_pA1, _pB0);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA0, _pB0);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA1, _pB0);
                _sum00 = __lsx_vadd_w(_sum00, __lsx_vhaddw_w_h(_s0, _s0));
                _sum01 = __lsx_vadd_w(_sum01, __lsx_vhaddw_w_h(_s1, _s1));

                pA += 32;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA0 = __lsx_vldrepl_w(pA, 0);
                _pA0 = __lsx_vilvl_b(__lsx_vslti_b(_pA0, 0), _pA0);
                __m128i _pA1 = __lsx_vldrepl_w(pA + 4, 0);
                _pA1 = __lsx_vilvl_b(__lsx_vslti_b(_pA1, 0), _pA1);

                __m128i _pB0 = __lsx_vreplgr2vr_h(pB[0]);
                __m128i _s0 = __lsx_vmul_h(_pA0, _pB0);
                __m128i _s1 = __lsx_vmul_h(_pA1, _pB0);
                _sum00 = __lsx_vadd_w(_sum00, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum01 = __lsx_vadd_w(_sum01, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                pA += 8;
                pB += 1;
            }

            __lsx_vst(_sum00, outptr, 0);
            __lsx_vst(_sum01, outptr + 4, 0);

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
#endif // __loongarch_asx
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* pB = pBT;

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

            __m128i _sum0;
            __m128i _sum1;
            __m128i _sum2;
            __m128i _sum3;
            __m128i _sum4;
            __m128i _sum5;
            __m128i _sum6;
            __m128i _sum7;

            if (k == 0)
            {
                _sum0 = __lsx_vreplgr2vr_w(0);
                _sum1 = __lsx_vreplgr2vr_w(0);
                _sum2 = __lsx_vreplgr2vr_w(0);
                _sum3 = __lsx_vreplgr2vr_w(0);
                _sum4 = __lsx_vreplgr2vr_w(0);
                _sum5 = __lsx_vreplgr2vr_w(0);
                _sum6 = __lsx_vreplgr2vr_w(0);
                _sum7 = __lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum0 = __lsx_vld(outptr, 0);
                _sum1 = __lsx_vld(outptr + 4, 0);
                _sum2 = __lsx_vld(outptr + 8, 0);
                _sum3 = __lsx_vld(outptr + 12, 0);
                _sum4 = __lsx_vld(outptr + 16, 0);
                _sum5 = __lsx_vld(outptr + 20, 0);
                _sum6 = __lsx_vld(outptr + 24, 0);
                _sum7 = __lsx_vld(outptr + 28, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA = __lsx_vld(pA, 0);
                __m128i _pA1 = __lsx_vshuf4i_w(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128i _pB0 = __lsx_vld(pB, 0);
                __m128i _pB1 = __lsx_vld(pB + 16, 0);
                __m128i _pB0r = __lsx_vshuf4i_w(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                __m128i _pB1r = __lsx_vshuf4i_w(_pB1, _LSX_SHUFFLE(0, 3, 2, 1));

                __m128i _s0 = __lsx_vmulwev_h_b(_pA, _pB0);
                __m128i _s1 = __lsx_vmulwev_h_b(_pA, _pB0r);
                __m128i _s2 = __lsx_vmulwev_h_b(_pA1, _pB0);
                __m128i _s3 = __lsx_vmulwev_h_b(_pA1, _pB0r);
                __m128i _s4 = __lsx_vmulwev_h_b(_pA, _pB1);
                __m128i _s5 = __lsx_vmulwev_h_b(_pA, _pB1r);
                __m128i _s6 = __lsx_vmulwev_h_b(_pA1, _pB1);
                __m128i _s7 = __lsx_vmulwev_h_b(_pA1, _pB1r);

                _s0 = __lsx_vmaddwod_h_b(_s0, _pA, _pB0);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA, _pB0r);
                _s2 = __lsx_vmaddwod_h_b(_s2, _pA1, _pB0);
                _s3 = __lsx_vmaddwod_h_b(_s3, _pA1, _pB0r);
                _s4 = __lsx_vmaddwod_h_b(_s4, _pA, _pB1);
                _s5 = __lsx_vmaddwod_h_b(_s5, _pA, _pB1r);
                _s6 = __lsx_vmaddwod_h_b(_s6, _pA1, _pB1);
                _s7 = __lsx_vmaddwod_h_b(_s7, _pA1, _pB1r);

                _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s1, _s1));
                _sum2 = __lsx_vadd_w(_sum2, __lsx_vhaddw_w_h(_s2, _s2));
                _sum3 = __lsx_vadd_w(_sum3, __lsx_vhaddw_w_h(_s3, _s3));
                _sum4 = __lsx_vadd_w(_sum4, __lsx_vhaddw_w_h(_s4, _s4));
                _sum5 = __lsx_vadd_w(_sum5, __lsx_vhaddw_w_h(_s5, _s5));
                _sum6 = __lsx_vadd_w(_sum6, __lsx_vhaddw_w_h(_s6, _s6));
                _sum7 = __lsx_vadd_w(_sum7, __lsx_vhaddw_w_h(_s7, _s7));

                pA += 16;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = __lsx_vldrepl_w(pA, 0);
                _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                __m128i _pA1 = __lsx_vshuf4i_h(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128i _pB0 = __lsx_vldrepl_w(pB, 0);
                _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                __m128i _pB1 = __lsx_vldrepl_w(pB + 4, 0);
                _pB1 = __lsx_vilvl_b(__lsx_vslti_b(_pB1, 0), _pB1);
                __m128i _pB0r = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                __m128i _pB1r = __lsx_vshuf4i_h(_pB1, _LSX_SHUFFLE(0, 3, 2, 1));

                __m128i _s0 = __lsx_vmul_h(_pA, _pB0);
                __m128i _s1 = __lsx_vmul_h(_pA, _pB0r);
                __m128i _s2 = __lsx_vmul_h(_pA1, _pB0);
                __m128i _s3 = __lsx_vmul_h(_pA1, _pB0r);
                __m128i _s4 = __lsx_vmul_h(_pA, _pB1);
                __m128i _s5 = __lsx_vmul_h(_pA, _pB1r);
                __m128i _s6 = __lsx_vmul_h(_pA1, _pB1);
                __m128i _s7 = __lsx_vmul_h(_pA1, _pB1r);

                _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                _sum2 = __lsx_vadd_w(_sum2, __lsx_vilvl_h(__lsx_vslti_h(_s2, 0), _s2));
                _sum3 = __lsx_vadd_w(_sum3, __lsx_vilvl_h(__lsx_vslti_h(_s3, 0), _s3));
                _sum4 = __lsx_vadd_w(_sum4, __lsx_vilvl_h(__lsx_vslti_h(_s4, 0), _s4));
                _sum5 = __lsx_vadd_w(_sum5, __lsx_vilvl_h(__lsx_vslti_h(_s5, 0), _s5));
                _sum6 = __lsx_vadd_w(_sum6, __lsx_vilvl_h(__lsx_vslti_h(_s6, 0), _s6));
                _sum7 = __lsx_vadd_w(_sum7, __lsx_vilvl_h(__lsx_vslti_h(_s7, 0), _s7));

                pA += 4;
                pB += 8;
            }

            __lsx_vst(_sum0, outptr, 0);
            __lsx_vst(_sum1, outptr + 4, 0);
            __lsx_vst(_sum2, outptr + 8, 0);
            __lsx_vst(_sum3, outptr + 12, 0);
            __lsx_vst(_sum4, outptr + 16, 0);
            __lsx_vst(_sum5, outptr + 20, 0);
            __lsx_vst(_sum6, outptr + 24, 0);
            __lsx_vst(_sum7, outptr + 28, 0);

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            __m128i _sum0;
            __m128i _sum1;
            __m128i _sum2;
            __m128i _sum3;

            if (k == 0)
            {
                _sum0 = __lsx_vreplgr2vr_w(0);
                _sum1 = __lsx_vreplgr2vr_w(0);
                _sum2 = __lsx_vreplgr2vr_w(0);
                _sum3 = __lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum0 = __lsx_vld(outptr, 0);
                _sum1 = __lsx_vld(outptr + 4, 0);
                _sum2 = __lsx_vld(outptr + 8, 0);
                _sum3 = __lsx_vld(outptr + 12, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA = __lsx_vld(pA, 0);
                __m128i _pA1 = __lsx_vshuf4i_w(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128i _pB = __lsx_vld(pB, 0);
                __m128i _pB1 = __lsx_vshuf4i_w(_pB, _LSX_SHUFFLE(0, 3, 2, 1));

                __m128i _s0 = __lsx_vmulwev_h_b(_pA, _pB);
                __m128i _s1 = __lsx_vmulwev_h_b(_pA, _pB1);
                __m128i _s2 = __lsx_vmulwev_h_b(_pA1, _pB);
                __m128i _s3 = __lsx_vmulwev_h_b(_pA1, _pB1);

                _s0 = __lsx_vmaddwod_h_b(_s0, _pA, _pB);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA, _pB1);
                _s2 = __lsx_vmaddwod_h_b(_s2, _pA1, _pB);
                _s3 = __lsx_vmaddwod_h_b(_s3, _pA1, _pB1);

                _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s1, _s1));
                _sum2 = __lsx_vadd_w(_sum2, __lsx_vhaddw_w_h(_s2, _s2));
                _sum3 = __lsx_vadd_w(_sum3, __lsx_vhaddw_w_h(_s3, _s3));

                pA += 16;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = __lsx_vldrepl_w(pA, 0);
                _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);
                __m128i _pA1 = __lsx_vshuf4i_h(_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128i _pB = __lsx_vldrepl_w(pB, 0);
                _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);
                __m128i _pB1 = __lsx_vshuf4i_h(_pB, _LSX_SHUFFLE(0, 3, 2, 1));

                __m128i _s0 = __lsx_vmul_h(_pA, _pB);
                __m128i _s1 = __lsx_vmul_h(_pA, _pB1);
                __m128i _s2 = __lsx_vmul_h(_pA1, _pB);
                __m128i _s3 = __lsx_vmul_h(_pA1, _pB1);

                _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                _sum2 = __lsx_vadd_w(_sum2, __lsx_vilvl_h(__lsx_vslti_h(_s2, 0), _s2));
                _sum3 = __lsx_vadd_w(_sum3, __lsx_vilvl_h(__lsx_vslti_h(_s3, 0), _s3));

                pA += 4;
                pB += 4;
            }

            __lsx_vst(_sum0, outptr, 0);
            __lsx_vst(_sum1, outptr + 4, 0);
            __lsx_vst(_sum2, outptr + 8, 0);
            __lsx_vst(_sum3, outptr + 12, 0);

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            __m128i _sum0;
            __m128i _sum1;

            if (k == 0)
            {
                _sum0 = __lsx_vreplgr2vr_w(0);
                _sum1 = __lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum0 = __lsx_vld(outptr, 0);
                _sum1 = __lsx_vld(outptr + 4, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA = __lsx_vld(pA, 0);

                __m128i _pB0 = __lsx_vldrepl_d(pB, 0);
                __m128i _pB1 = __lsx_vshuf4i_w(_pB0, _LSX_SHUFFLE(2, 3, 0, 1));

                __m128i _s0 = __lsx_vmulwev_h_b(_pA, _pB0);
                __m128i _s1 = __lsx_vmulwev_h_b(_pA, _pB1);

                _s0 = __lsx_vmaddwod_h_b(_s0, _pA, _pB0);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA, _pB1);

                _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s1, _s1));

                pA += 16;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = __lsx_vldrepl_w(pA, 0);
                _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);

                int b01 = (unsigned char)pB[0] | ((unsigned char)pB[1] << 8);
                __m128i _pB0 = __lsx_vreplgr2vr_w(b01);
                _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                _pB0 = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(1, 0, 1, 0));
                __m128i _pB1 = __lsx_vshuf4i_h(_pB0, _LSX_SHUFFLE(0, 3, 2, 1));

                __m128i _s0 = __lsx_vmul_h(_pA, _pB0);
                __m128i _s1 = __lsx_vmul_h(_pA, _pB1);

                _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                pA += 4;
                pB += 2;
            }

            __lsx_vst(_sum0, outptr, 0);
            __lsx_vst(_sum1, outptr + 4, 0);

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            __m128i _sum0;

            if (k == 0)
            {
                _sum0 = __lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum0 = __lsx_vld(outptr, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA = __lsx_vld(pA, 0);
                __m128i _pB0 = __lsx_vldrepl_w(pB, 0);

                __m128i _s0 = __lsx_vmulwev_h_b(_pA, _pB0);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA, _pB0);

                _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));

                pA += 16;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = __lsx_vldrepl_w(pA, 0);
                _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);

                __m128i _pB0 = __lsx_vreplgr2vr_h(pB[0]);

                __m128i _s0 = __lsx_vmul_h(_pA, _pB0);

                _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));

                pA += 4;
                pB += 1;
            }

            __lsx_vst(_sum0, outptr, 0);

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if __loongarch_sx
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128i _sum0;
            __m128i _sum1;
            __m128i _sum2;
            __m128i _sum3;

            if (k == 0)
            {
                _sum0 = __lsx_vreplgr2vr_w(0);
                _sum1 = __lsx_vreplgr2vr_w(0);
                _sum2 = __lsx_vreplgr2vr_w(0);
                _sum3 = __lsx_vreplgr2vr_w(0);
            }
            else
            {
                __m128i _s0 = __lsx_vld(outptr, 0);
                __m128i _s1 = __lsx_vld(outptr + 4, 0);
                __m128i _s2 = __lsx_vld(outptr + 8, 0);
                __m128i _s3 = __lsx_vld(outptr + 12, 0);

                _sum0 = __lsx_vpickev_w(_s1, _s0);
                _sum1 = __lsx_vpickod_w(_s1, _s0);
                _sum2 = __lsx_vpickev_w(_s3, _s2);
                _sum3 = __lsx_vpickod_w(_s3, _s2);
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA0 = __lsx_vldrepl_w(pA, 0);
                __m128i _pA1 = __lsx_vldrepl_w(pA + 4, 0);
                __m128i _pB0 = __lsx_vld(pB, 0);
                __m128i _pB1 = __lsx_vld(pB + 16, 0);

                __m128i _s0 = __lsx_vmulwev_h_b(_pA0, _pB0);
                __m128i _s1 = __lsx_vmulwev_h_b(_pA1, _pB0);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA0, _pB0);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA1, _pB0);
                _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s1, _s1));

                _s0 = __lsx_vmulwev_h_b(_pA0, _pB1);
                _s1 = __lsx_vmulwev_h_b(_pA1, _pB1);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA0, _pB1);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA1, _pB1);
                _sum2 = __lsx_vadd_w(_sum2, __lsx_vhaddw_w_h(_s0, _s0));
                _sum3 = __lsx_vadd_w(_sum3, __lsx_vhaddw_w_h(_s1, _s1));

                pA += 8;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pB0 = __lsx_vldrepl_w(pB, 0);
                _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                __m128i _pB1 = __lsx_vldrepl_w(pB + 4, 0);
                _pB1 = __lsx_vilvl_b(__lsx_vslti_b(_pB1, 0), _pB1);

                __m128i _pA0 = __lsx_vreplgr2vr_h(pA[0]);
                __m128i _pA1 = __lsx_vreplgr2vr_h(pA[1]);

                __m128i _s0 = __lsx_vmul_h(_pA0, _pB0);
                __m128i _s1 = __lsx_vmul_h(_pA1, _pB0);
                __m128i _s2 = __lsx_vmul_h(_pA0, _pB1);
                __m128i _s3 = __lsx_vmul_h(_pA1, _pB1);

                _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));
                _sum2 = __lsx_vadd_w(_sum2, __lsx_vilvl_h(__lsx_vslti_h(_s2, 0), _s2));
                _sum3 = __lsx_vadd_w(_sum3, __lsx_vilvl_h(__lsx_vslti_h(_s3, 0), _s3));

                pA += 2;
                pB += 8;
            }

            __m128i _s0 = __lsx_vilvl_w(_sum1, _sum0);
            __m128i _s1 = __lsx_vilvh_w(_sum1, _sum0);
            __m128i _s2 = __lsx_vilvl_w(_sum3, _sum2);
            __m128i _s3 = __lsx_vilvh_w(_sum3, _sum2);

            __lsx_vst(_s0, outptr, 0);
            __lsx_vst(_s1, outptr + 4, 0);
            __lsx_vst(_s2, outptr + 8, 0);
            __lsx_vst(_s3, outptr + 12, 0);

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _sum0;
            __m128i _sum1;

            if (k == 0)
            {
                _sum0 = __lsx_vreplgr2vr_w(0);
                _sum1 = __lsx_vreplgr2vr_w(0);
            }
            else
            {
                __m128i _s0 = __lsx_vld(outptr, 0);
                __m128i _s1 = __lsx_vld(outptr + 4, 0);

                _sum0 = __lsx_vpickev_w(_s1, _s0);
                _sum1 = __lsx_vpickod_w(_s1, _s0);
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA0 = __lsx_vldrepl_w(pA, 0);
                __m128i _pA1 = __lsx_vldrepl_w(pA + 4, 0);
                __m128i _pB = __lsx_vld(pB, 0);

                __m128i _s0 = __lsx_vmulwev_h_b(_pA0, _pB);
                __m128i _s1 = __lsx_vmulwev_h_b(_pA1, _pB);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA0, _pB);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA1, _pB);
                _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s1, _s1));

                pA += 8;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pB = __lsx_vldrepl_w(pB, 0);
                _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);

                __m128i _pA0 = __lsx_vreplgr2vr_h(pA[0]);
                __m128i _pA1 = __lsx_vreplgr2vr_h(pA[1]);

                __m128i _s0 = __lsx_vmul_h(_pA0, _pB);
                __m128i _s1 = __lsx_vmul_h(_pA1, _pB);

                _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                pA += 2;
                pB += 4;
            }

            __m128i _s0 = __lsx_vilvl_w(_sum1, _sum0);
            __m128i _s1 = __lsx_vilvh_w(_sum1, _sum0);

            __lsx_vst(_s0, outptr, 0);
            __lsx_vst(_s1, outptr + 4, 0);

            outptr += 8;
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            int sum00;
            int sum01;
            int sum10;
            int sum11;

            if (k == 0)
            {
                sum00 = 0;
                sum01 = 0;
                sum10 = 0;
                sum11 = 0;
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                asm volatile(""
                             :
                             :
                             : "memory");

                sum00 += pA[0] * pB[0];
                sum00 += pA[1] * pB[1];
                sum00 += pA[2] * pB[2];
                sum00 += pA[3] * pB[3];
                sum01 += pA[4] * pB[0];
                sum01 += pA[5] * pB[1];
                sum01 += pA[6] * pB[2];
                sum01 += pA[7] * pB[3];
                sum10 += pA[0] * pB[4];
                sum10 += pA[1] * pB[5];
                sum10 += pA[2] * pB[6];
                sum10 += pA[3] * pB[7];
                sum11 += pA[4] * pB[4];
                sum11 += pA[5] * pB[5];
                sum11 += pA[6] * pB[6];
                sum11 += pA[7] * pB[7];

                pA += 8;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                asm volatile(""
                             :
                             :
                             : "memory");

                sum00 += pA[0] * pB[0];
                sum01 += pA[1] * pB[0];
                sum10 += pA[0] * pB[1];
                sum11 += pA[1] * pB[1];

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
            int sum0;
            int sum1;

            if (k == 0)
            {
                sum0 = 0;
                sum1 = 0;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                asm volatile(""
                             :
                             :
                             : "memory");

                sum0 += pA[0] * pB[0];
                sum0 += pA[1] * pB[1];
                sum0 += pA[2] * pB[2];
                sum0 += pA[3] * pB[3];
                sum1 += pA[4] * pB[0];
                sum1 += pA[5] * pB[1];
                sum1 += pA[6] * pB[2];
                sum1 += pA[7] * pB[3];

                pA += 8;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                asm volatile(""
                             :
                             :
                             : "memory");

                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[0];

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
        const signed char* pB = pBT;

        int jj = 0;
#if __loongarch_sx
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128i _sum0;
            __m128i _sum1;

            if (k == 0)
            {
                _sum0 = __lsx_vreplgr2vr_w(0);
                _sum1 = __lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum0 = __lsx_vld(outptr, 0);
                _sum1 = __lsx_vld(outptr + 4, 0);
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA = __lsx_vldrepl_w(pA, 0);
                __m128i _pB0 = __lsx_vld(pB, 0);
                __m128i _pB1 = __lsx_vld(pB + 16, 0);

                __m128i _s0 = __lsx_vmulwev_h_b(_pA, _pB0);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA, _pB0);
                _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));

                __m128i _s1 = __lsx_vmulwev_h_b(_pA, _pB1);
                _s1 = __lsx_vmaddwod_h_b(_s1, _pA, _pB1);
                _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s1, _s1));

                pA += 4;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pB0 = __lsx_vldrepl_w(pB, 0);
                _pB0 = __lsx_vilvl_b(__lsx_vslti_b(_pB0, 0), _pB0);
                __m128i _pB1 = __lsx_vldrepl_w(pB + 4, 0);
                _pB1 = __lsx_vilvl_b(__lsx_vslti_b(_pB1, 0), _pB1);

                __m128i _pA0 = __lsx_vreplgr2vr_h(pA[0]);

                __m128i _s0 = __lsx_vmul_h(_pA0, _pB0);
                __m128i _s1 = __lsx_vmul_h(_pA0, _pB1);

                _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));
                _sum1 = __lsx_vadd_w(_sum1, __lsx_vilvl_h(__lsx_vslti_h(_s1, 0), _s1));

                pA += 1;
                pB += 8;
            }

            __lsx_vst(_sum0, outptr, 0);
            __lsx_vst(_sum1, outptr + 4, 0);

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _sum0;

            if (k == 0)
            {
                _sum0 = __lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum0 = __lsx_vld(outptr, 0);
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA = __lsx_vldrepl_w(pA, 0);
                __m128i _pB = __lsx_vld(pB, 0);

                __m128i _s0 = __lsx_vmulwev_h_b(_pA, _pB);
                _s0 = __lsx_vmaddwod_h_b(_s0, _pA, _pB);
                _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));

                pA += 4;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pB = __lsx_vldrepl_w(pB, 0);
                _pB = __lsx_vilvl_b(__lsx_vslti_b(_pB, 0), _pB);

                __m128i _pA0 = __lsx_vreplgr2vr_h(pA[0]);

                __m128i _s0 = __lsx_vmul_h(_pA0, _pB);

                _sum0 = __lsx_vadd_w(_sum0, __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0));

                pA += 1;
                pB += 4;
            }

            __lsx_vst(_sum0, outptr, 0);

            outptr += 4;
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            int sum0;
            int sum1;

            if (k == 0)
            {
                sum0 = 0;
                sum1 = 0;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                asm volatile(""
                             :
                             :
                             : "memory");

                sum0 += pA[0] * pB[0];
                sum0 += pA[1] * pB[1];
                sum0 += pA[2] * pB[2];
                sum0 += pA[3] * pB[3];
                sum1 += pA[0] * pB[4];
                sum1 += pA[1] * pB[5];
                sum1 += pA[2] * pB[6];
                sum1 += pA[3] * pB[7];

                pA += 4;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                asm volatile(""
                             :
                             :
                             : "memory");

                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];

                pA += 1;
                pB += 2;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            int sum;

            if (k == 0)
            {
                sum = 0;
            }
            else
            {
                sum = outptr[0];
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum += pA[0] * pB[0];
                pA += 1;
                pB += 1;
            }

            outptr[0] = sum;

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void unpack_output_tile_int32_to_fp32(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta, int output_transpose)
{
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const float* descale_ptr = descales;

    const int* pp = topT;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        __m128 _descale0 = (__m128)__lsx_vld(descale_ptr + i + ii, 0);
        __m128 _descale1 = (__m128)__lsx_vld(descale_ptr + i + ii + 4, 0);

        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + i + ii;
        }
        else
        {
            p0 = (float*)top_blob + (i + ii) * out_hstep + j;
        }

        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (size_t)(i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            __builtin_prefetch(pp + 64);
#if __loongarch_asx
            __m256i _sum0 = __lasx_xvld(pp, 0);
            __m256i _sum1 = __lasx_xvld(pp + 8, 0);
            __m256i _sum2 = __lasx_xvld(pp + 16, 0);
            __m256i _sum3 = __lasx_xvld(pp + 24, 0);
            __m256i _sum4 = __lasx_xvld(pp + 32, 0);
            __m256i _sum5 = __lasx_xvld(pp + 40, 0);
            __m256i _sum6 = __lasx_xvld(pp + 48, 0);
            __m256i _sum7 = __lasx_xvld(pp + 56, 0);

            {
                __m256i _tmp0 = _sum0;
                __m256i _tmp1 = __lasx_xvshuf4i_w(_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256i _tmp2 = _sum2;
                __m256i _tmp3 = __lasx_xvshuf4i_w(_sum3, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256i _tmp4 = _sum4;
                __m256i _tmp5 = __lasx_xvshuf4i_w(_sum5, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256i _tmp6 = _sum6;
                __m256i _tmp7 = __lasx_xvshuf4i_w(_sum7, _LSX_SHUFFLE(2, 1, 0, 3));

                _sum0 = __lasx_xvilvl_w(_tmp3, _tmp0);
                _sum1 = __lasx_xvilvh_w(_tmp3, _tmp0);
                _sum2 = __lasx_xvilvl_w(_tmp1, _tmp2);
                _sum3 = __lasx_xvilvh_w(_tmp1, _tmp2);
                _sum4 = __lasx_xvilvl_w(_tmp7, _tmp4);
                _sum5 = __lasx_xvilvh_w(_tmp7, _tmp4);
                _sum6 = __lasx_xvilvl_w(_tmp5, _tmp6);
                _sum7 = __lasx_xvilvh_w(_tmp5, _tmp6);

                _tmp0 = __lasx_xvilvl_d(_sum2, _sum0);
                _tmp1 = __lasx_xvilvh_d(_sum2, _sum0);
                _tmp2 = __lasx_xvilvl_d(_sum1, _sum3);
                _tmp3 = __lasx_xvilvh_d(_sum1, _sum3);
                _tmp4 = __lasx_xvilvl_d(_sum6, _sum4);
                _tmp5 = __lasx_xvilvh_d(_sum6, _sum4);
                _tmp6 = __lasx_xvilvl_d(_sum5, _sum7);
                _tmp7 = __lasx_xvilvh_d(_sum5, _sum7);

                _tmp1 = __lasx_xvshuf4i_w(_tmp1, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp3 = __lasx_xvshuf4i_w(_tmp3, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp5 = __lasx_xvshuf4i_w(_tmp5, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp7 = __lasx_xvshuf4i_w(_tmp7, _LSX_SHUFFLE(2, 1, 0, 3));

                _sum0 = __lasx_xvpermi_q(_tmp4, _tmp0, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum1 = __lasx_xvpermi_q(_tmp5, _tmp1, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum2 = __lasx_xvpermi_q(_tmp6, _tmp2, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum3 = __lasx_xvpermi_q(_tmp7, _tmp3, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum4 = __lasx_xvpermi_q(_tmp0, _tmp4, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum5 = __lasx_xvpermi_q(_tmp1, _tmp5, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum6 = __lasx_xvpermi_q(_tmp2, _tmp6, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum7 = __lasx_xvpermi_q(_tmp3, _tmp7, _LSX_SHUFFLE(0, 3, 0, 0));
            }

            __m256 _descale = (__m256)__lasx_xvld(descale_ptr + i + ii, 0);
            __m256 _f0 = __lasx_xvfmul_s((__m256)__lasx_xvffint_s_w(_sum0), _descale);
            __m256 _f1 = __lasx_xvfmul_s((__m256)__lasx_xvffint_s_w(_sum1), _descale);
            __m256 _f2 = __lasx_xvfmul_s((__m256)__lasx_xvffint_s_w(_sum2), _descale);
            __m256 _f3 = __lasx_xvfmul_s((__m256)__lasx_xvffint_s_w(_sum3), _descale);
            __m256 _f4 = __lasx_xvfmul_s((__m256)__lasx_xvffint_s_w(_sum4), _descale);
            __m256 _f5 = __lasx_xvfmul_s((__m256)__lasx_xvffint_s_w(_sum5), _descale);
            __m256 _f6 = __lasx_xvfmul_s((__m256)__lasx_xvffint_s_w(_sum6), _descale);
            __m256 _f7 = __lasx_xvfmul_s((__m256)__lasx_xvffint_s_w(_sum7), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m256 _c0 = (__m256)__lasx_xvreplfr2vr_s(pC[0] * beta);
                    _f0 = __lasx_xvfadd_s(_f0, _c0);
                    _f1 = __lasx_xvfadd_s(_f1, _c0);
                    _f2 = __lasx_xvfadd_s(_f2, _c0);
                    _f3 = __lasx_xvfadd_s(_f3, _c0);
                    _f4 = __lasx_xvfadd_s(_f4, _c0);
                    _f5 = __lasx_xvfadd_s(_f5, _c0);
                    _f6 = __lasx_xvfadd_s(_f6, _c0);
                    _f7 = __lasx_xvfadd_s(_f7, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m256 _c0 = (__m256)__lasx_xvld(pC, 0);
                    _c0 = __lasx_xvfmul_s(_c0, (__m256)__lasx_xvreplfr2vr_s(beta));
                    _f0 = __lasx_xvfadd_s(_f0, _c0);
                    _f1 = __lasx_xvfadd_s(_f1, _c0);
                    _f2 = __lasx_xvfadd_s(_f2, _c0);
                    _f3 = __lasx_xvfadd_s(_f3, _c0);
                    _f4 = __lasx_xvfadd_s(_f4, _c0);
                    _f5 = __lasx_xvfadd_s(_f5, _c0);
                    _f6 = __lasx_xvfadd_s(_f6, _c0);
                    _f7 = __lasx_xvfadd_s(_f7, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                    __m256 _c0 = (__m256)__lasx_xvld(pC, 0);
                    __m256 _c1 = (__m256)__lasx_xvld(pC + c_hstep, 0);
                    __m256 _c2 = (__m256)__lasx_xvld(pC + c_hstep * 2, 0);
                    __m256 _c3 = (__m256)__lasx_xvld(pC + c_hstep * 3, 0);
                    __m256 _c4 = (__m256)__lasx_xvld(pC + c_hstep * 4, 0);
                    __m256 _c5 = (__m256)__lasx_xvld(pC + c_hstep * 5, 0);
                    __m256 _c6 = (__m256)__lasx_xvld(pC + c_hstep * 6, 0);
                    __m256 _c7 = (__m256)__lasx_xvld(pC + c_hstep * 7, 0);
                    transpose8x8_ps(_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7);
                    _f0 = __lasx_xvfmadd_s(_c0, _beta, _f0);
                    _f1 = __lasx_xvfmadd_s(_c1, _beta, _f1);
                    _f2 = __lasx_xvfmadd_s(_c2, _beta, _f2);
                    _f3 = __lasx_xvfmadd_s(_c3, _beta, _f3);
                    _f4 = __lasx_xvfmadd_s(_c4, _beta, _f4);
                    _f5 = __lasx_xvfmadd_s(_c5, _beta, _f5);
                    _f6 = __lasx_xvfmadd_s(_c6, _beta, _f6);
                    _f7 = __lasx_xvfmadd_s(_c7, _beta, _f7);
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    _f0 = __lasx_xvfadd_s(_f0, (__m256)__lasx_xvreplfr2vr_s(pC[0] * beta));
                    _f1 = __lasx_xvfadd_s(_f1, (__m256)__lasx_xvreplfr2vr_s(pC[1] * beta));
                    _f2 = __lasx_xvfadd_s(_f2, (__m256)__lasx_xvreplfr2vr_s(pC[2] * beta));
                    _f3 = __lasx_xvfadd_s(_f3, (__m256)__lasx_xvreplfr2vr_s(pC[3] * beta));
                    _f4 = __lasx_xvfadd_s(_f4, (__m256)__lasx_xvreplfr2vr_s(pC[4] * beta));
                    _f5 = __lasx_xvfadd_s(_f5, (__m256)__lasx_xvreplfr2vr_s(pC[5] * beta));
                    _f6 = __lasx_xvfadd_s(_f6, (__m256)__lasx_xvreplfr2vr_s(pC[6] * beta));
                    _f7 = __lasx_xvfadd_s(_f7, (__m256)__lasx_xvreplfr2vr_s(pC[7] * beta));
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m256 _alpha = (__m256)__lasx_xvreplfr2vr_s(alpha);
                _f0 = __lasx_xvfmul_s(_f0, _alpha);
                _f1 = __lasx_xvfmul_s(_f1, _alpha);
                _f2 = __lasx_xvfmul_s(_f2, _alpha);
                _f3 = __lasx_xvfmul_s(_f3, _alpha);
                _f4 = __lasx_xvfmul_s(_f4, _alpha);
                _f5 = __lasx_xvfmul_s(_f5, _alpha);
                _f6 = __lasx_xvfmul_s(_f6, _alpha);
                _f7 = __lasx_xvfmul_s(_f7, _alpha);
            }

            if (output_transpose)
            {
                __lasx_xvst(_f0, p0, 0);
                __lasx_xvst(_f1, p0 + out_hstep, 0);
                __lasx_xvst(_f2, p0 + out_hstep * 2, 0);
                __lasx_xvst(_f3, p0 + out_hstep * 3, 0);
                __lasx_xvst(_f4, p0 + out_hstep * 4, 0);
                __lasx_xvst(_f5, p0 + out_hstep * 5, 0);
                __lasx_xvst(_f6, p0 + out_hstep * 6, 0);
                __lasx_xvst(_f7, p0 + out_hstep * 7, 0);
                p0 += out_hstep * 8;
            }
            else
            {
                transpose8x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                __lasx_xvst(_f0, p0, 0);
                __lasx_xvst(_f1, p0 + out_hstep, 0);
                __lasx_xvst(_f2, p0 + out_hstep * 2, 0);
                __lasx_xvst(_f3, p0 + out_hstep * 3, 0);
                __lasx_xvst(_f4, p0 + out_hstep * 4, 0);
                __lasx_xvst(_f5, p0 + out_hstep * 5, 0);
                __lasx_xvst(_f6, p0 + out_hstep * 6, 0);
                __lasx_xvst(_f7, p0 + out_hstep * 7, 0);
                p0 += 8;
            }

            pp += 64;
#else  // __loongarch_asx
            __m128i _sum0 = __lsx_vld(pp, 0);
            __m128i _sum8 = __lsx_vld(pp + 4, 0);
            __m128i _sum1 = __lsx_vld(pp + 8, 0);
            __m128i _sum9 = __lsx_vld(pp + 12, 0);
            __m128i _sum2 = __lsx_vld(pp + 16, 0);
            __m128i _suma = __lsx_vld(pp + 20, 0);
            __m128i _sum3 = __lsx_vld(pp + 24, 0);
            __m128i _sumb = __lsx_vld(pp + 28, 0);

            _sum2 = __lsx_vshuf4i_w(_sum2, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum3 = __lsx_vshuf4i_w(_sum3, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
            _sum1 = __lsx_vshuf4i_w(_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum2 = __lsx_vshuf4i_w(_sum2, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum3 = __lsx_vshuf4i_w(_sum3, _LSX_SHUFFLE(0, 3, 2, 1));

            _suma = __lsx_vshuf4i_w(_suma, _LSX_SHUFFLE(1, 0, 3, 2));
            _sumb = __lsx_vshuf4i_w(_sumb, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum8, _sum9, _suma, _sumb);
            _sum9 = __lsx_vshuf4i_w(_sum9, _LSX_SHUFFLE(2, 1, 0, 3));
            _suma = __lsx_vshuf4i_w(_suma, _LSX_SHUFFLE(1, 0, 3, 2));
            _sumb = __lsx_vshuf4i_w(_sumb, _LSX_SHUFFLE(0, 3, 2, 1));

            __m128i _sum4 = __lsx_vld(pp + 32, 0);
            __m128i _sumc = __lsx_vld(pp + 36, 0);
            __m128i _sum5 = __lsx_vld(pp + 40, 0);
            __m128i _sumd = __lsx_vld(pp + 44, 0);
            __m128i _sum6 = __lsx_vld(pp + 48, 0);
            __m128i _sume = __lsx_vld(pp + 52, 0);
            __m128i _sum7 = __lsx_vld(pp + 56, 0);
            __m128i _sumf = __lsx_vld(pp + 60, 0);

            _sum6 = __lsx_vshuf4i_w(_sum6, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum7 = __lsx_vshuf4i_w(_sum7, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum4, _sum5, _sum6, _sum7);
            _sum5 = __lsx_vshuf4i_w(_sum5, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum6 = __lsx_vshuf4i_w(_sum6, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum7 = __lsx_vshuf4i_w(_sum7, _LSX_SHUFFLE(0, 3, 2, 1));

            _sume = __lsx_vshuf4i_w(_sume, _LSX_SHUFFLE(1, 0, 3, 2));
            _sumf = __lsx_vshuf4i_w(_sumf, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sumc, _sumd, _sume, _sumf);
            _sumd = __lsx_vshuf4i_w(_sumd, _LSX_SHUFFLE(2, 1, 0, 3));
            _sume = __lsx_vshuf4i_w(_sume, _LSX_SHUFFLE(1, 0, 3, 2));
            _sumf = __lsx_vshuf4i_w(_sumf, _LSX_SHUFFLE(0, 3, 2, 1));

            __m128 _f0 = (__m128)__lsx_vffint_s_w(_sum0);
            __m128 _f4 = (__m128)__lsx_vffint_s_w(_sum4);
            __m128 _f1 = (__m128)__lsx_vffint_s_w(_sum1);
            __m128 _f5 = (__m128)__lsx_vffint_s_w(_sum5);
            __m128 _f2 = (__m128)__lsx_vffint_s_w(_sum2);
            __m128 _f6 = (__m128)__lsx_vffint_s_w(_sum6);
            __m128 _f3 = (__m128)__lsx_vffint_s_w(_sum3);
            __m128 _f7 = (__m128)__lsx_vffint_s_w(_sum7);
            __m128 _f8 = (__m128)__lsx_vffint_s_w(_sum8);
            __m128 _fc = (__m128)__lsx_vffint_s_w(_sumc);
            __m128 _f9 = (__m128)__lsx_vffint_s_w(_sum9);
            __m128 _fd = (__m128)__lsx_vffint_s_w(_sumd);
            __m128 _fa = (__m128)__lsx_vffint_s_w(_suma);
            __m128 _fe = (__m128)__lsx_vffint_s_w(_sume);
            __m128 _fb = (__m128)__lsx_vffint_s_w(_sumb);
            __m128 _ff = (__m128)__lsx_vffint_s_w(_sumf);

            _f0 = __lsx_vfmul_s(_f0, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 0));
            _f4 = __lsx_vfmul_s(_f4, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 0));
            _f1 = __lsx_vfmul_s(_f1, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 1));
            _f5 = __lsx_vfmul_s(_f5, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 1));
            _f2 = __lsx_vfmul_s(_f2, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 2));
            _f6 = __lsx_vfmul_s(_f6, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 2));
            _f3 = __lsx_vfmul_s(_f3, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 3));
            _f7 = __lsx_vfmul_s(_f7, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 3));
            _f8 = __lsx_vfmul_s(_f8, (__m128)__lsx_vreplvei_w((__m128i)_descale1, 0));
            _fc = __lsx_vfmul_s(_fc, (__m128)__lsx_vreplvei_w((__m128i)_descale1, 0));
            _f9 = __lsx_vfmul_s(_f9, (__m128)__lsx_vreplvei_w((__m128i)_descale1, 1));
            _fd = __lsx_vfmul_s(_fd, (__m128)__lsx_vreplvei_w((__m128i)_descale1, 1));
            _fa = __lsx_vfmul_s(_fa, (__m128)__lsx_vreplvei_w((__m128i)_descale1, 2));
            _fe = __lsx_vfmul_s(_fe, (__m128)__lsx_vreplvei_w((__m128i)_descale1, 2));
            _fb = __lsx_vfmul_s(_fb, (__m128)__lsx_vreplvei_w((__m128i)_descale1, 3));
            _ff = __lsx_vfmul_s(_ff, (__m128)__lsx_vreplvei_w((__m128i)_descale1, 3));

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(pC[0] * beta);
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f4 = __lsx_vfadd_s(_f4, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                    _f5 = __lsx_vfadd_s(_f5, _c0);
                    _f2 = __lsx_vfadd_s(_f2, _c0);
                    _f6 = __lsx_vfadd_s(_f6, _c0);
                    _f3 = __lsx_vfadd_s(_f3, _c0);
                    _f7 = __lsx_vfadd_s(_f7, _c0);
                    _f8 = __lsx_vfadd_s(_f8, _c0);
                    _fc = __lsx_vfadd_s(_fc, _c0);
                    _f9 = __lsx_vfadd_s(_f9, _c0);
                    _fd = __lsx_vfadd_s(_fd, _c0);
                    _fa = __lsx_vfadd_s(_fa, _c0);
                    _fe = __lsx_vfadd_s(_fe, _c0);
                    _fb = __lsx_vfadd_s(_fb, _c0);
                    _ff = __lsx_vfadd_s(_ff, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(pC[0] * beta);
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f4 = __lsx_vfadd_s(_f4, _c0);
                    _c0 = __lsx_vreplfr2vr_s(pC[1] * beta);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                    _f5 = __lsx_vfadd_s(_f5, _c0);
                    _c0 = __lsx_vreplfr2vr_s(pC[2] * beta);
                    _f2 = __lsx_vfadd_s(_f2, _c0);
                    _f6 = __lsx_vfadd_s(_f6, _c0);
                    _c0 = __lsx_vreplfr2vr_s(pC[3] * beta);
                    _f3 = __lsx_vfadd_s(_f3, _c0);
                    _f7 = __lsx_vfadd_s(_f7, _c0);
                    _c0 = __lsx_vreplfr2vr_s(pC[4] * beta);
                    _f8 = __lsx_vfadd_s(_f8, _c0);
                    _fc = __lsx_vfadd_s(_fc, _c0);
                    _c0 = __lsx_vreplfr2vr_s(pC[5] * beta);
                    _f9 = __lsx_vfadd_s(_f9, _c0);
                    _fd = __lsx_vfadd_s(_fd, _c0);
                    _c0 = __lsx_vreplfr2vr_s(pC[6] * beta);
                    _fa = __lsx_vfadd_s(_fa, _c0);
                    _fe = __lsx_vfadd_s(_fe, _c0);
                    _c0 = __lsx_vreplfr2vr_s(pC[7] * beta);
                    _fb = __lsx_vfadd_s(_fb, _c0);
                    _ff = __lsx_vfadd_s(_ff, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vfmul_s((__m128)__lsx_vld(pC, 0), _beta));
                    _f4 = __lsx_vfadd_s(_f4, __lsx_vfmul_s((__m128)__lsx_vld(pC + 4, 0), _beta));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep, 0), _beta));
                    _f5 = __lsx_vfadd_s(_f5, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep + 4, 0), _beta));
                    _f2 = __lsx_vfadd_s(_f2, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 2, 0), _beta));
                    _f6 = __lsx_vfadd_s(_f6, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 2 + 4, 0), _beta));
                    _f3 = __lsx_vfadd_s(_f3, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 3, 0), _beta));
                    _f7 = __lsx_vfadd_s(_f7, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 3 + 4, 0), _beta));
                    _f8 = __lsx_vfadd_s(_f8, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 4, 0), _beta));
                    _fc = __lsx_vfadd_s(_fc, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 4 + 4, 0), _beta));
                    _f9 = __lsx_vfadd_s(_f9, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 5, 0), _beta));
                    _fd = __lsx_vfadd_s(_fd, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 5 + 4, 0), _beta));
                    _fa = __lsx_vfadd_s(_fa, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 6, 0), _beta));
                    _fe = __lsx_vfadd_s(_fe, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 6 + 4, 0), _beta));
                    _fb = __lsx_vfadd_s(_fb, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 7, 0), _beta));
                    _ff = __lsx_vfadd_s(_ff, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 7 + 4, 0), _beta));
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    __m128 _c0 = __lsx_vfmul_s((__m128)__lsx_vld(pC, 0), _beta);
                    __m128 _c1 = __lsx_vfmul_s((__m128)__lsx_vld(pC + 4, 0), _beta);
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f4 = __lsx_vfadd_s(_f4, _c1);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                    _f5 = __lsx_vfadd_s(_f5, _c1);
                    _f2 = __lsx_vfadd_s(_f2, _c0);
                    _f6 = __lsx_vfadd_s(_f6, _c1);
                    _f3 = __lsx_vfadd_s(_f3, _c0);
                    _f7 = __lsx_vfadd_s(_f7, _c1);
                    _f8 = __lsx_vfadd_s(_f8, _c0);
                    _fc = __lsx_vfadd_s(_fc, _c1);
                    _f9 = __lsx_vfadd_s(_f9, _c0);
                    _fd = __lsx_vfadd_s(_fd, _c1);
                    _fa = __lsx_vfadd_s(_fa, _c0);
                    _fe = __lsx_vfadd_s(_fe, _c1);
                    _fb = __lsx_vfadd_s(_fb, _c0);
                    _ff = __lsx_vfadd_s(_ff, _c1);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f4 = __lsx_vfmul_s(_f4, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
                _f5 = __lsx_vfmul_s(_f5, _alpha);
                _f2 = __lsx_vfmul_s(_f2, _alpha);
                _f6 = __lsx_vfmul_s(_f6, _alpha);
                _f3 = __lsx_vfmul_s(_f3, _alpha);
                _f7 = __lsx_vfmul_s(_f7, _alpha);
                _f8 = __lsx_vfmul_s(_f8, _alpha);
                _fc = __lsx_vfmul_s(_fc, _alpha);
                _f9 = __lsx_vfmul_s(_f9, _alpha);
                _fd = __lsx_vfmul_s(_fd, _alpha);
                _fa = __lsx_vfmul_s(_fa, _alpha);
                _fe = __lsx_vfmul_s(_fe, _alpha);
                _fb = __lsx_vfmul_s(_fb, _alpha);
                _ff = __lsx_vfmul_s(_ff, _alpha);
            }

            if (output_transpose)
            {
                transpose4x4_ps(_f0, _f1, _f2, _f3);
                transpose4x4_ps(_f4, _f5, _f6, _f7);
                transpose4x4_ps(_f8, _f9, _fa, _fb);
                transpose4x4_ps(_fc, _fd, _fe, _ff);

                __lsx_vst((__m128i)_f0, p0, 0);
                __lsx_vst((__m128i)_f8, p0 + 4, 0);
                __lsx_vst((__m128i)_f1, p0 + out_hstep, 0);
                __lsx_vst((__m128i)_f9, p0 + out_hstep + 4, 0);
                __lsx_vst((__m128i)_f2, p0 + out_hstep * 2, 0);
                __lsx_vst((__m128i)_fa, p0 + out_hstep * 2 + 4, 0);
                __lsx_vst((__m128i)_f3, p0 + out_hstep * 3, 0);
                __lsx_vst((__m128i)_fb, p0 + out_hstep * 3 + 4, 0);
                __lsx_vst((__m128i)_f4, p0 + out_hstep * 4, 0);
                __lsx_vst((__m128i)_fc, p0 + out_hstep * 4 + 4, 0);
                __lsx_vst((__m128i)_f5, p0 + out_hstep * 5, 0);
                __lsx_vst((__m128i)_fd, p0 + out_hstep * 5 + 4, 0);
                __lsx_vst((__m128i)_f6, p0 + out_hstep * 6, 0);
                __lsx_vst((__m128i)_fe, p0 + out_hstep * 6 + 4, 0);
                __lsx_vst((__m128i)_f7, p0 + out_hstep * 7, 0);
                __lsx_vst((__m128i)_ff, p0 + out_hstep * 7 + 4, 0);
                p0 += out_hstep * 8;
            }
            else
            {
                __lsx_vst((__m128i)_f0, p0, 0);
                __lsx_vst((__m128i)_f4, p0 + 4, 0);
                __lsx_vst((__m128i)_f1, p0 + out_hstep, 0);
                __lsx_vst((__m128i)_f5, p0 + out_hstep + 4, 0);
                __lsx_vst((__m128i)_f2, p0 + out_hstep * 2, 0);
                __lsx_vst((__m128i)_f6, p0 + out_hstep * 2 + 4, 0);
                __lsx_vst((__m128i)_f3, p0 + out_hstep * 3, 0);
                __lsx_vst((__m128i)_f7, p0 + out_hstep * 3 + 4, 0);
                __lsx_vst((__m128i)_f8, p0 + out_hstep * 4, 0);
                __lsx_vst((__m128i)_fc, p0 + out_hstep * 4 + 4, 0);
                __lsx_vst((__m128i)_f9, p0 + out_hstep * 5, 0);
                __lsx_vst((__m128i)_fd, p0 + out_hstep * 5 + 4, 0);
                __lsx_vst((__m128i)_fa, p0 + out_hstep * 6, 0);
                __lsx_vst((__m128i)_fe, p0 + out_hstep * 6 + 4, 0);
                __lsx_vst((__m128i)_fb, p0 + out_hstep * 7, 0);
                __lsx_vst((__m128i)_ff, p0 + out_hstep * 7 + 4, 0);
                p0 += 8;
            }
            pp += 64;
#endif // __loongarch_asx
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __builtin_prefetch(pp + 32);
            __m128i _sum0 = __lsx_vld(pp, 0);
            __m128i _sum1 = __lsx_vld(pp + 8, 0);
            __m128i _sum2 = __lsx_vld(pp + 16, 0);
            __m128i _sum3 = __lsx_vld(pp + 24, 0);

            _sum2 = __lsx_vshuf4i_w(_sum2, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum3 = __lsx_vshuf4i_w(_sum3, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
            _sum1 = __lsx_vshuf4i_w(_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum2 = __lsx_vshuf4i_w(_sum2, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum3 = __lsx_vshuf4i_w(_sum3, _LSX_SHUFFLE(0, 3, 2, 1));

            __m128i _sum4 = __lsx_vld(pp + 4, 0);
            __m128i _sum5 = __lsx_vld(pp + 12, 0);
            __m128i _sum6 = __lsx_vld(pp + 20, 0);
            __m128i _sum7 = __lsx_vld(pp + 28, 0);

            _sum6 = __lsx_vshuf4i_w(_sum6, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum7 = __lsx_vshuf4i_w(_sum7, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum4, _sum5, _sum6, _sum7);
            _sum5 = __lsx_vshuf4i_w(_sum5, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum6 = __lsx_vshuf4i_w(_sum6, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum7 = __lsx_vshuf4i_w(_sum7, _LSX_SHUFFLE(0, 3, 2, 1));

            __m128 _f0 = (__m128)__lsx_vffint_s_w(_sum0);
            __m128 _f1 = (__m128)__lsx_vffint_s_w(_sum1);
            __m128 _f2 = (__m128)__lsx_vffint_s_w(_sum2);
            __m128 _f3 = (__m128)__lsx_vffint_s_w(_sum3);
            __m128 _f4 = (__m128)__lsx_vffint_s_w(_sum4);
            __m128 _f5 = (__m128)__lsx_vffint_s_w(_sum5);
            __m128 _f6 = (__m128)__lsx_vffint_s_w(_sum6);
            __m128 _f7 = (__m128)__lsx_vffint_s_w(_sum7);

            _f0 = __lsx_vfmul_s(_f0, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 0));
            _f1 = __lsx_vfmul_s(_f1, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 1));
            _f2 = __lsx_vfmul_s(_f2, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 2));
            _f3 = __lsx_vfmul_s(_f3, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 3));
            _f4 = __lsx_vfmul_s(_f4, (__m128)__lsx_vreplvei_w((__m128i)_descale1, 0));
            _f5 = __lsx_vfmul_s(_f5, (__m128)__lsx_vreplvei_w((__m128i)_descale1, 1));
            _f6 = __lsx_vfmul_s(_f6, (__m128)__lsx_vreplvei_w((__m128i)_descale1, 2));
            _f7 = __lsx_vfmul_s(_f7, (__m128)__lsx_vreplvei_w((__m128i)_descale1, 3));

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(pC[0] * beta);
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                    _f2 = __lsx_vfadd_s(_f2, _c0);
                    _f3 = __lsx_vfadd_s(_f3, _c0);
                    _f4 = __lsx_vfadd_s(_f4, _c0);
                    _f5 = __lsx_vfadd_s(_f5, _c0);
                    _f6 = __lsx_vfadd_s(_f6, _c0);
                    _f7 = __lsx_vfadd_s(_f7, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(pC[0] * beta));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vreplfr2vr_s(pC[1] * beta));
                    _f2 = __lsx_vfadd_s(_f2, __lsx_vreplfr2vr_s(pC[2] * beta));
                    _f3 = __lsx_vfadd_s(_f3, __lsx_vreplfr2vr_s(pC[3] * beta));
                    _f4 = __lsx_vfadd_s(_f4, __lsx_vreplfr2vr_s(pC[4] * beta));
                    _f5 = __lsx_vfadd_s(_f5, __lsx_vreplfr2vr_s(pC[5] * beta));
                    _f6 = __lsx_vfadd_s(_f6, __lsx_vreplfr2vr_s(pC[6] * beta));
                    _f7 = __lsx_vfadd_s(_f7, __lsx_vreplfr2vr_s(pC[7] * beta));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vfmul_s((__m128)__lsx_vld(pC, 0), _beta));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep, 0), _beta));
                    _f2 = __lsx_vfadd_s(_f2, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 2, 0), _beta));
                    _f3 = __lsx_vfadd_s(_f3, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 3, 0), _beta));
                    _f4 = __lsx_vfadd_s(_f4, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 4, 0), _beta));
                    _f5 = __lsx_vfadd_s(_f5, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 5, 0), _beta));
                    _f6 = __lsx_vfadd_s(_f6, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 6, 0), _beta));
                    _f7 = __lsx_vfadd_s(_f7, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 7, 0), _beta));
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c0 = __lsx_vfmul_s((__m128)__lsx_vld(pC, 0), __lsx_vreplfr2vr_s(beta));
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                    _f2 = __lsx_vfadd_s(_f2, _c0);
                    _f3 = __lsx_vfadd_s(_f3, _c0);
                    _f4 = __lsx_vfadd_s(_f4, _c0);
                    _f5 = __lsx_vfadd_s(_f5, _c0);
                    _f6 = __lsx_vfadd_s(_f6, _c0);
                    _f7 = __lsx_vfadd_s(_f7, _c0);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
                _f2 = __lsx_vfmul_s(_f2, _alpha);
                _f3 = __lsx_vfmul_s(_f3, _alpha);
                _f4 = __lsx_vfmul_s(_f4, _alpha);
                _f5 = __lsx_vfmul_s(_f5, _alpha);
                _f6 = __lsx_vfmul_s(_f6, _alpha);
                _f7 = __lsx_vfmul_s(_f7, _alpha);
            }

            if (output_transpose)
            {
                transpose4x4_ps(_f0, _f1, _f2, _f3);
                transpose4x4_ps(_f4, _f5, _f6, _f7);

                __lsx_vst((__m128i)_f0, p0, 0);
                __lsx_vst((__m128i)_f4, p0 + 4, 0);
                __lsx_vst((__m128i)_f1, p0 + out_hstep, 0);
                __lsx_vst((__m128i)_f5, p0 + out_hstep + 4, 0);
                __lsx_vst((__m128i)_f2, p0 + out_hstep * 2, 0);
                __lsx_vst((__m128i)_f6, p0 + out_hstep * 2 + 4, 0);
                __lsx_vst((__m128i)_f3, p0 + out_hstep * 3, 0);
                __lsx_vst((__m128i)_f7, p0 + out_hstep * 3 + 4, 0);
                p0 += out_hstep * 4;
            }
            else
            {
                __lsx_vst((__m128i)_f0, p0, 0);
                __lsx_vst((__m128i)_f1, p0 + out_hstep, 0);
                __lsx_vst((__m128i)_f2, p0 + out_hstep * 2, 0);
                __lsx_vst((__m128i)_f3, p0 + out_hstep * 3, 0);
                __lsx_vst((__m128i)_f4, p0 + out_hstep * 4, 0);
                __lsx_vst((__m128i)_f5, p0 + out_hstep * 5, 0);
                __lsx_vst((__m128i)_f6, p0 + out_hstep * 6, 0);
                __lsx_vst((__m128i)_f7, p0 + out_hstep * 7, 0);
                p0 += 4;
            }
            pp += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __builtin_prefetch(pp + 16);
            __m128i _sum0 = __lsx_vld(pp, 0);
            __m128i _sum1 = __lsx_vld(pp + 4, 0);
            __m128i _sum2 = __lsx_vld(pp + 8, 0);
            __m128i _sum3 = __lsx_vld(pp + 12, 0);

            __m128i _sum0e = __lsx_vshuf4i_w(_sum0, _LSX_SHUFFLE(3, 1, 2, 0));
            __m128i _sum0o = __lsx_vshuf4i_w(_sum0, _LSX_SHUFFLE(2, 0, 3, 1));
            __m128i _sum2e = __lsx_vshuf4i_w(_sum2, _LSX_SHUFFLE(3, 1, 2, 0));
            __m128i _sum2o = __lsx_vshuf4i_w(_sum2, _LSX_SHUFFLE(2, 0, 3, 1));
            __m128i _sum4e = __lsx_vshuf4i_w(_sum1, _LSX_SHUFFLE(3, 1, 2, 0));
            __m128i _sum4o = __lsx_vshuf4i_w(_sum1, _LSX_SHUFFLE(2, 0, 3, 1));
            __m128i _sum6e = __lsx_vshuf4i_w(_sum3, _LSX_SHUFFLE(3, 1, 2, 0));
            __m128i _sum6o = __lsx_vshuf4i_w(_sum3, _LSX_SHUFFLE(2, 0, 3, 1));

            __m128 _f0 = (__m128)__lsx_vffint_s_w((__m128i)__lsx_vilvl_w(_sum2o, _sum0e));
            __m128 _f1 = (__m128)__lsx_vffint_s_w((__m128i)__lsx_vilvl_w(_sum0o, _sum2e));
            __m128 _f4 = (__m128)__lsx_vffint_s_w((__m128i)__lsx_vilvl_w(_sum6o, _sum4e));
            __m128 _f5 = (__m128)__lsx_vffint_s_w((__m128i)__lsx_vilvl_w(_sum4o, _sum6e));

            _f0 = __lsx_vfmul_s(_f0, _descale0);
            _f1 = __lsx_vfmul_s(_f1, _descale0);
            _f4 = __lsx_vfmul_s(_f4, _descale1);
            _f5 = __lsx_vfmul_s(_f5, _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(pC[0] * beta);
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                    _f4 = __lsx_vfadd_s(_f4, _c0);
                    _f5 = __lsx_vfadd_s(_f5, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    __m128 _c0 = __lsx_vfmul_s((__m128)__lsx_vld(pC, 0), _beta);
                    __m128 _c1 = __lsx_vfmul_s((__m128)__lsx_vld(pC + 4, 0), _beta);
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                    _f4 = __lsx_vfadd_s(_f4, _c1);
                    _f5 = __lsx_vfadd_s(_f5, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    __m128i _c0 = __lsx_vreplgr2vr_w(((const int*)pC)[0]);
                    _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep))[0], 1);
                    _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep * 2))[0], 2);
                    _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep * 3))[0], 3);
                    __m128i _c1 = __lsx_vreplgr2vr_w(((const int*)pC)[1]);
                    _c1 = __lsx_vinsgr2vr_w(_c1, ((const int*)(pC + c_hstep))[1], 1);
                    _c1 = __lsx_vinsgr2vr_w(_c1, ((const int*)(pC + c_hstep * 2))[1], 2);
                    _c1 = __lsx_vinsgr2vr_w(_c1, ((const int*)(pC + c_hstep * 3))[1], 3);
                    __m128i _c4 = __lsx_vreplgr2vr_w(((const int*)(pC + c_hstep * 4))[0]);
                    _c4 = __lsx_vinsgr2vr_w(_c4, ((const int*)(pC + c_hstep * 5))[0], 1);
                    _c4 = __lsx_vinsgr2vr_w(_c4, ((const int*)(pC + c_hstep * 6))[0], 2);
                    _c4 = __lsx_vinsgr2vr_w(_c4, ((const int*)(pC + c_hstep * 7))[0], 3);
                    __m128i _c5 = __lsx_vreplgr2vr_w(((const int*)(pC + c_hstep * 4))[1]);
                    _c5 = __lsx_vinsgr2vr_w(_c5, ((const int*)(pC + c_hstep * 5))[1], 1);
                    _c5 = __lsx_vinsgr2vr_w(_c5, ((const int*)(pC + c_hstep * 6))[1], 2);
                    _c5 = __lsx_vinsgr2vr_w(_c5, ((const int*)(pC + c_hstep * 7))[1], 3);
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vfmul_s((__m128)_c0, _beta));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vfmul_s((__m128)_c1, _beta));
                    _f4 = __lsx_vfadd_s(_f4, __lsx_vfmul_s((__m128)_c4, _beta));
                    _f5 = __lsx_vfadd_s(_f5, __lsx_vfmul_s((__m128)_c5, _beta));
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(pC[0] * beta);
                    __m128 _c1 = __lsx_vreplfr2vr_s(pC[1] * beta);
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f4 = __lsx_vfadd_s(_f4, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c1);
                    _f5 = __lsx_vfadd_s(_f5, _c1);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
                _f4 = __lsx_vfmul_s(_f4, _alpha);
                _f5 = __lsx_vfmul_s(_f5, _alpha);
            }

            if (output_transpose)
            {
                __lsx_vst((__m128i)_f0, p0, 0);
                __lsx_vst((__m128i)_f4, p0 + 4, 0);
                __lsx_vst((__m128i)_f1, p0 + out_hstep, 0);
                __lsx_vst((__m128i)_f5, p0 + out_hstep + 4, 0);
                p0 += out_hstep * 2;
            }
            else
            {
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_f1, (__m128i)_f0);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_f1, (__m128i)_f0);
                __m128 _tmp2 = (__m128)__lsx_vilvl_w((__m128i)_f5, (__m128i)_f4);
                __m128 _tmp3 = (__m128)__lsx_vilvh_w((__m128i)_f5, (__m128i)_f4);

                *(int64_t*)p0 = __lsx_vpickve2gr_d((__m128i)_tmp0, 0);
                *(int64_t*)(p0 + out_hstep) = __lsx_vpickve2gr_d((__m128i)_tmp0, 1);
                *(int64_t*)(p0 + out_hstep * 2) = __lsx_vpickve2gr_d((__m128i)_tmp1, 0);
                *(int64_t*)(p0 + out_hstep * 3) = __lsx_vpickve2gr_d((__m128i)_tmp1, 1);
                *(int64_t*)(p0 + out_hstep * 4) = __lsx_vpickve2gr_d((__m128i)_tmp2, 0);
                *(int64_t*)(p0 + out_hstep * 5) = __lsx_vpickve2gr_d((__m128i)_tmp2, 1);
                *(int64_t*)(p0 + out_hstep * 6) = __lsx_vpickve2gr_d((__m128i)_tmp3, 0);
                *(int64_t*)(p0 + out_hstep * 7) = __lsx_vpickve2gr_d((__m128i)_tmp3, 1);
                p0 += 2;
            }
            pp += 16;
        }
        for (; jj < max_jj; jj++)
        {
            __builtin_prefetch(pp + 8);
            __m128 _f0 = (__m128)__lsx_vffint_s_w(__lsx_vld(pp, 0));
            __m128 _f4 = (__m128)__lsx_vffint_s_w(__lsx_vld(pp + 4, 0));

            _f0 = __lsx_vfmul_s(_f0, _descale0);
            _f4 = __lsx_vfmul_s(_f4, _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(pC[0] * beta);
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f4 = __lsx_vfadd_s(_f4, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vfmul_s((__m128)__lsx_vld(pC, 0), _beta));
                    _f4 = __lsx_vfadd_s(_f4, __lsx_vfmul_s((__m128)__lsx_vld(pC + 4, 0), _beta));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    __m128i _c0 = __lsx_vreplgr2vr_w(((const int*)pC)[0]);
                    _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep))[0], 1);
                    _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep * 2))[0], 2);
                    _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep * 3))[0], 3);
                    __m128i _c4 = __lsx_vreplgr2vr_w(((const int*)(pC + c_hstep * 4))[0]);
                    _c4 = __lsx_vinsgr2vr_w(_c4, ((const int*)(pC + c_hstep * 5))[0], 1);
                    _c4 = __lsx_vinsgr2vr_w(_c4, ((const int*)(pC + c_hstep * 6))[0], 2);
                    _c4 = __lsx_vinsgr2vr_w(_c4, ((const int*)(pC + c_hstep * 7))[0], 3);
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vfmul_s((__m128)_c0, _beta));
                    _f4 = __lsx_vfadd_s(_f4, __lsx_vfmul_s((__m128)_c4, _beta));
                    pC += 1;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(pC[0] * beta);
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f4 = __lsx_vfadd_s(_f4, _c0);
                    pC += 1;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f4 = __lsx_vfmul_s(_f4, _alpha);
            }

            if (output_transpose)
            {
                __lsx_vst((__m128i)_f0, p0, 0);
                __lsx_vst((__m128i)_f4, p0 + 4, 0);
                p0 += out_hstep;
            }
            else
            {
                *(int*)p0 = __lsx_vpickve2gr_w((__m128i)_f0, 0);
                *(int*)(p0 + out_hstep) = __lsx_vpickve2gr_w((__m128i)_f0, 1);
                *(int*)(p0 + out_hstep * 2) = __lsx_vpickve2gr_w((__m128i)_f0, 2);
                *(int*)(p0 + out_hstep * 3) = __lsx_vpickve2gr_w((__m128i)_f0, 3);
                *(int*)(p0 + out_hstep * 4) = __lsx_vpickve2gr_w((__m128i)_f4, 0);
                *(int*)(p0 + out_hstep * 5) = __lsx_vpickve2gr_w((__m128i)_f4, 1);
                *(int*)(p0 + out_hstep * 6) = __lsx_vpickve2gr_w((__m128i)_f4, 2);
                *(int*)(p0 + out_hstep * 7) = __lsx_vpickve2gr_w((__m128i)_f4, 3);
                p0 += 1;
            }
            pp += 8;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        __m128 _descale0 = (__m128)__lsx_vld(descale_ptr + i + ii, 0);

        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + i + ii;
        }
        else
        {
            p0 = (float*)top_blob + (i + ii) * out_hstep + j;
        }

        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (size_t)(i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            __builtin_prefetch(pp + 32);
            __m128i _sum0 = __lsx_vld(pp, 0);
            __m128i _sum1 = __lsx_vld(pp + 4, 0);
            __m128i _sum2 = __lsx_vld(pp + 8, 0);
            __m128i _sum3 = __lsx_vld(pp + 12, 0);

            _sum2 = __lsx_vshuf4i_w(_sum2, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum3 = __lsx_vshuf4i_w(_sum3, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
            _sum1 = __lsx_vshuf4i_w(_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum2 = __lsx_vshuf4i_w(_sum2, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum3 = __lsx_vshuf4i_w(_sum3, _LSX_SHUFFLE(0, 3, 2, 1));

            __m128i _sum4 = __lsx_vld(pp + 16, 0);
            __m128i _sum5 = __lsx_vld(pp + 20, 0);
            __m128i _sum6 = __lsx_vld(pp + 24, 0);
            __m128i _sum7 = __lsx_vld(pp + 28, 0);

            _sum6 = __lsx_vshuf4i_w(_sum6, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum7 = __lsx_vshuf4i_w(_sum7, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum4, _sum5, _sum6, _sum7);
            _sum5 = __lsx_vshuf4i_w(_sum5, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum6 = __lsx_vshuf4i_w(_sum6, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum7 = __lsx_vshuf4i_w(_sum7, _LSX_SHUFFLE(0, 3, 2, 1));

            __m128 _f0 = (__m128)__lsx_vffint_s_w(_sum0);
            __m128 _f4 = (__m128)__lsx_vffint_s_w(_sum4);
            __m128 _f1 = (__m128)__lsx_vffint_s_w(_sum1);
            __m128 _f5 = (__m128)__lsx_vffint_s_w(_sum5);
            __m128 _f2 = (__m128)__lsx_vffint_s_w(_sum2);
            __m128 _f6 = (__m128)__lsx_vffint_s_w(_sum6);
            __m128 _f3 = (__m128)__lsx_vffint_s_w(_sum3);
            __m128 _f7 = (__m128)__lsx_vffint_s_w(_sum7);

            _f0 = __lsx_vfmul_s(_f0, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 0));
            _f4 = __lsx_vfmul_s(_f4, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 0));
            _f1 = __lsx_vfmul_s(_f1, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 1));
            _f5 = __lsx_vfmul_s(_f5, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 1));
            _f2 = __lsx_vfmul_s(_f2, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 2));
            _f6 = __lsx_vfmul_s(_f6, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 2));
            _f3 = __lsx_vfmul_s(_f3, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 3));
            _f7 = __lsx_vfmul_s(_f7, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 3));

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(pC[0] * beta);
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f4 = __lsx_vfadd_s(_f4, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                    _f5 = __lsx_vfadd_s(_f5, _c0);
                    _f2 = __lsx_vfadd_s(_f2, _c0);
                    _f6 = __lsx_vfadd_s(_f6, _c0);
                    _f3 = __lsx_vfadd_s(_f3, _c0);
                    _f7 = __lsx_vfadd_s(_f7, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(pC[0] * beta);
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f4 = __lsx_vfadd_s(_f4, _c0);
                    _c0 = __lsx_vreplfr2vr_s(pC[1] * beta);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                    _f5 = __lsx_vfadd_s(_f5, _c0);
                    _c0 = __lsx_vreplfr2vr_s(pC[2] * beta);
                    _f2 = __lsx_vfadd_s(_f2, _c0);
                    _f6 = __lsx_vfadd_s(_f6, _c0);
                    _c0 = __lsx_vreplfr2vr_s(pC[3] * beta);
                    _f3 = __lsx_vfadd_s(_f3, _c0);
                    _f7 = __lsx_vfadd_s(_f7, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vfmul_s((__m128)__lsx_vld(pC, 0), _beta));
                    _f4 = __lsx_vfadd_s(_f4, __lsx_vfmul_s((__m128)__lsx_vld(pC + 4, 0), _beta));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep, 0), _beta));
                    _f5 = __lsx_vfadd_s(_f5, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep + 4, 0), _beta));
                    _f2 = __lsx_vfadd_s(_f2, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 2, 0), _beta));
                    _f6 = __lsx_vfadd_s(_f6, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 2 + 4, 0), _beta));
                    _f3 = __lsx_vfadd_s(_f3, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 3, 0), _beta));
                    _f7 = __lsx_vfadd_s(_f7, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 3 + 4, 0), _beta));
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    __m128 _c0 = __lsx_vfmul_s((__m128)__lsx_vld(pC, 0), _beta);
                    __m128 _c1 = __lsx_vfmul_s((__m128)__lsx_vld(pC + 4, 0), _beta);
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f4 = __lsx_vfadd_s(_f4, _c1);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                    _f5 = __lsx_vfadd_s(_f5, _c1);
                    _f2 = __lsx_vfadd_s(_f2, _c0);
                    _f6 = __lsx_vfadd_s(_f6, _c1);
                    _f3 = __lsx_vfadd_s(_f3, _c0);
                    _f7 = __lsx_vfadd_s(_f7, _c1);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f4 = __lsx_vfmul_s(_f4, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
                _f5 = __lsx_vfmul_s(_f5, _alpha);
                _f2 = __lsx_vfmul_s(_f2, _alpha);
                _f6 = __lsx_vfmul_s(_f6, _alpha);
                _f3 = __lsx_vfmul_s(_f3, _alpha);
                _f7 = __lsx_vfmul_s(_f7, _alpha);
            }

            if (output_transpose)
            {
                transpose4x4_ps(_f0, _f1, _f2, _f3);
                transpose4x4_ps(_f4, _f5, _f6, _f7);

                __lsx_vst((__m128i)_f0, p0, 0);
                __lsx_vst((__m128i)_f1, p0 + out_hstep, 0);
                __lsx_vst((__m128i)_f2, p0 + out_hstep * 2, 0);
                __lsx_vst((__m128i)_f3, p0 + out_hstep * 3, 0);
                __lsx_vst((__m128i)_f4, p0 + out_hstep * 4, 0);
                __lsx_vst((__m128i)_f5, p0 + out_hstep * 5, 0);
                __lsx_vst((__m128i)_f6, p0 + out_hstep * 6, 0);
                __lsx_vst((__m128i)_f7, p0 + out_hstep * 7, 0);
                p0 += out_hstep * 8;
            }
            else
            {
                __lsx_vst((__m128i)_f0, p0, 0);
                __lsx_vst((__m128i)_f4, p0 + 4, 0);
                __lsx_vst((__m128i)_f1, p0 + out_hstep, 0);
                __lsx_vst((__m128i)_f5, p0 + out_hstep + 4, 0);
                __lsx_vst((__m128i)_f2, p0 + out_hstep * 2, 0);
                __lsx_vst((__m128i)_f6, p0 + out_hstep * 2 + 4, 0);
                __lsx_vst((__m128i)_f3, p0 + out_hstep * 3, 0);
                __lsx_vst((__m128i)_f7, p0 + out_hstep * 3 + 4, 0);
                p0 += 8;
            }
            pp += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __builtin_prefetch(pp + 16);
            __m128i _sum0 = __lsx_vld(pp, 0);
            __m128i _sum1 = __lsx_vld(pp + 4, 0);
            __m128i _sum2 = __lsx_vld(pp + 8, 0);
            __m128i _sum3 = __lsx_vld(pp + 12, 0);

            _sum2 = __lsx_vshuf4i_w(_sum2, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum3 = __lsx_vshuf4i_w(_sum3, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
            _sum1 = __lsx_vshuf4i_w(_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum2 = __lsx_vshuf4i_w(_sum2, _LSX_SHUFFLE(1, 0, 3, 2));
            _sum3 = __lsx_vshuf4i_w(_sum3, _LSX_SHUFFLE(0, 3, 2, 1));

            __m128 _f0 = (__m128)__lsx_vffint_s_w(_sum0);
            __m128 _f1 = (__m128)__lsx_vffint_s_w(_sum1);
            __m128 _f2 = (__m128)__lsx_vffint_s_w(_sum2);
            __m128 _f3 = (__m128)__lsx_vffint_s_w(_sum3);

            _f0 = __lsx_vfmul_s(_f0, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 0));
            _f1 = __lsx_vfmul_s(_f1, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 1));
            _f2 = __lsx_vfmul_s(_f2, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 2));
            _f3 = __lsx_vfmul_s(_f3, (__m128)__lsx_vreplvei_w((__m128i)_descale0, 3));

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(pC[0] * beta);
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                    _f2 = __lsx_vfadd_s(_f2, _c0);
                    _f3 = __lsx_vfadd_s(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(pC[0] * beta));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vreplfr2vr_s(pC[1] * beta));
                    _f2 = __lsx_vfadd_s(_f2, __lsx_vreplfr2vr_s(pC[2] * beta));
                    _f3 = __lsx_vfadd_s(_f3, __lsx_vreplfr2vr_s(pC[3] * beta));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vfmul_s((__m128)__lsx_vld(pC, 0), _beta));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep, 0), _beta));
                    _f2 = __lsx_vfadd_s(_f2, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 2, 0), _beta));
                    _f3 = __lsx_vfadd_s(_f3, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep * 3, 0), _beta));
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c0 = __lsx_vfmul_s((__m128)__lsx_vld(pC, 0), __lsx_vreplfr2vr_s(beta));
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                    _f2 = __lsx_vfadd_s(_f2, _c0);
                    _f3 = __lsx_vfadd_s(_f3, _c0);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
                _f2 = __lsx_vfmul_s(_f2, _alpha);
                _f3 = __lsx_vfmul_s(_f3, _alpha);
            }

            if (output_transpose)
            {
                transpose4x4_ps(_f0, _f1, _f2, _f3);

                __lsx_vst((__m128i)_f0, p0, 0);
                __lsx_vst((__m128i)_f1, p0 + out_hstep, 0);
                __lsx_vst((__m128i)_f2, p0 + out_hstep * 2, 0);
                __lsx_vst((__m128i)_f3, p0 + out_hstep * 3, 0);
                p0 += out_hstep * 4;
            }
            else
            {
                __lsx_vst((__m128i)_f0, p0, 0);
                __lsx_vst((__m128i)_f1, p0 + out_hstep, 0);
                __lsx_vst((__m128i)_f2, p0 + out_hstep * 2, 0);
                __lsx_vst((__m128i)_f3, p0 + out_hstep * 3, 0);
                p0 += 4;
            }
            pp += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __builtin_prefetch(pp + 8);
            __m128i _sum0 = __lsx_vld(pp, 0);
            __m128i _sum1 = __lsx_vld(pp + 4, 0);

            __m128i _sum0e = __lsx_vshuf4i_w(_sum0, _LSX_SHUFFLE(3, 1, 2, 0));
            __m128i _sum0o = __lsx_vshuf4i_w(_sum0, _LSX_SHUFFLE(2, 0, 3, 1));
            __m128i _sum1e = __lsx_vshuf4i_w(_sum1, _LSX_SHUFFLE(3, 1, 2, 0));
            __m128i _sum1o = __lsx_vshuf4i_w(_sum1, _LSX_SHUFFLE(2, 0, 3, 1));

            __m128 _f0 = (__m128)__lsx_vffint_s_w((__m128i)__lsx_vilvl_w(_sum1o, _sum0e));
            __m128 _f1 = (__m128)__lsx_vffint_s_w((__m128i)__lsx_vilvl_w(_sum0o, _sum1e));

            _f0 = __lsx_vfmul_s(_f0, _descale0);
            _f1 = __lsx_vfmul_s(_f1, _descale0);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(pC[0] * beta);
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c0 = __lsx_vfmul_s((__m128)__lsx_vld(pC, 0), __lsx_vreplfr2vr_s(beta));
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    __m128i _c0 = __lsx_vreplgr2vr_w(((const int*)pC)[0]);
                    _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep))[0], 1);
                    _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep * 2))[0], 2);
                    _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep * 3))[0], 3);
                    __m128i _c1 = __lsx_vreplgr2vr_w(((const int*)pC)[1]);
                    _c1 = __lsx_vinsgr2vr_w(_c1, ((const int*)(pC + c_hstep))[1], 1);
                    _c1 = __lsx_vinsgr2vr_w(_c1, ((const int*)(pC + c_hstep * 2))[1], 2);
                    _c1 = __lsx_vinsgr2vr_w(_c1, ((const int*)(pC + c_hstep * 3))[1], 3);
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vfmul_s((__m128)_c0, _beta));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vfmul_s((__m128)_c1, _beta));
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(pC[0] * beta));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vreplfr2vr_s(pC[1] * beta));
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
            }

            if (output_transpose)
            {
                __lsx_vst((__m128i)_f0, p0, 0);
                __lsx_vst((__m128i)_f1, p0 + out_hstep, 0);
                p0 += out_hstep * 2;
            }
            else
            {
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_f1, (__m128i)_f0);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_f1, (__m128i)_f0);

                *(int64_t*)p0 = __lsx_vpickve2gr_d((__m128i)_tmp0, 0);
                *(int64_t*)(p0 + out_hstep) = __lsx_vpickve2gr_d((__m128i)_tmp0, 1);
                *(int64_t*)(p0 + out_hstep * 2) = __lsx_vpickve2gr_d((__m128i)_tmp1, 0);
                *(int64_t*)(p0 + out_hstep * 3) = __lsx_vpickve2gr_d((__m128i)_tmp1, 1);
                p0 += 2;
            }
            pp += 8;
        }
        for (; jj < max_jj; jj++)
        {
            __builtin_prefetch(pp + 4);
            __m128 _f0 = (__m128)__lsx_vffint_s_w(__lsx_vld(pp, 0));

            _f0 = __lsx_vfmul_s(_f0, _descale0);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(pC[0] * beta));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vfmul_s((__m128)__lsx_vld(pC, 0), __lsx_vreplfr2vr_s(beta)));
                }
                if (broadcast_type_C == 3)
                {
                    __m128i _c0 = __lsx_vreplgr2vr_w(((const int*)pC)[0]);
                    _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep))[0], 1);
                    _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep * 2))[0], 2);
                    _c0 = __lsx_vinsgr2vr_w(_c0, ((const int*)(pC + c_hstep * 3))[0], 3);
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vfmul_s((__m128)_c0, __lsx_vreplfr2vr_s(beta)));
                    pC += 1;
                }
                if (broadcast_type_C == 4)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(pC[0] * beta));
                    pC += 1;
                }
            }

            if (alpha != 1.f)
            {
                _f0 = __lsx_vfmul_s(_f0, __lsx_vreplfr2vr_s(alpha));
            }

            if (output_transpose)
            {
                __lsx_vst((__m128i)_f0, p0, 0);
                p0 += out_hstep;
            }
            else
            {
                *(int*)p0 = __lsx_vpickve2gr_w((__m128i)_f0, 0);
                *(int*)(p0 + out_hstep) = __lsx_vpickve2gr_w((__m128i)_f0, 1);
                *(int*)(p0 + out_hstep * 2) = __lsx_vpickve2gr_w((__m128i)_f0, 2);
                *(int*)(p0 + out_hstep * 3) = __lsx_vpickve2gr_w((__m128i)_f0, 3);
                p0 += 1;
            }
            pp += 4;
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + i + ii;
        }
        else
        {
            p0 = (float*)top_blob + (i + ii) * out_hstep + j;
        }

        const float descale0 = descale_ptr[i + ii];
        const float descale1 = descale_ptr[i + ii + 1];

        float c0 = 0.f;
        float c1 = 0.f;
        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                c0 = pC[0] * beta;
                c1 = pC[1] * beta;
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (size_t)(i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if __loongarch_sx
        __m128 _descale0 = __lsx_vreplfr2vr_s(descale0);
        __m128 _descale1 = __lsx_vreplfr2vr_s(descale1);
        for (; jj + 7 < max_jj; jj += 8)
        {
            __builtin_prefetch(pp + 16);
            __m128i _sum0 = __lsx_vld(pp, 0);
            __m128i _sum1 = __lsx_vld(pp + 4, 0);
            __m128i _sum2 = __lsx_vld(pp + 8, 0);
            __m128i _sum3 = __lsx_vld(pp + 12, 0);

            __m128i _sum00 = __lsx_vpickev_w(_sum1, _sum0);
            __m128i _sum10 = __lsx_vpickod_w(_sum1, _sum0);
            __m128i _sum01 = __lsx_vpickev_w(_sum3, _sum2);
            __m128i _sum11 = __lsx_vpickod_w(_sum3, _sum2);

            __m128 _f0 = (__m128)__lsx_vffint_s_w(_sum00);
            __m128 _f1 = (__m128)__lsx_vffint_s_w(_sum01);
            __m128 _f2 = (__m128)__lsx_vffint_s_w(_sum10);
            __m128 _f3 = (__m128)__lsx_vffint_s_w(_sum11);

            _f0 = __lsx_vfmul_s(_f0, _descale0);
            _f1 = __lsx_vfmul_s(_f1, _descale0);
            _f2 = __lsx_vfmul_s(_f2, _descale1);
            _f3 = __lsx_vfmul_s(_f3, _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(c0);
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                    _f2 = __lsx_vfadd_s(_f2, _c0);
                    _f3 = __lsx_vfadd_s(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(c0));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vreplfr2vr_s(c0));
                    _f2 = __lsx_vfadd_s(_f2, __lsx_vreplfr2vr_s(c1));
                    _f3 = __lsx_vfadd_s(_f3, __lsx_vreplfr2vr_s(c1));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vfmul_s((__m128)__lsx_vld(pC, 0), _beta));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vfmul_s((__m128)__lsx_vld(pC + 4, 0), _beta));
                    _f2 = __lsx_vfadd_s(_f2, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep, 0), _beta));
                    _f3 = __lsx_vfadd_s(_f3, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep + 4, 0), _beta));
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    __m128 _c0 = __lsx_vfmul_s((__m128)__lsx_vld(pC, 0), _beta);
                    __m128 _c1 = __lsx_vfmul_s((__m128)__lsx_vld(pC + 4, 0), _beta);
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c1);
                    _f2 = __lsx_vfadd_s(_f2, _c0);
                    _f3 = __lsx_vfadd_s(_f3, _c1);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
                _f2 = __lsx_vfmul_s(_f2, _alpha);
                _f3 = __lsx_vfmul_s(_f3, _alpha);
            }

            if (output_transpose)
            {
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_f2, (__m128i)_f0);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_f2, (__m128i)_f0);
                __m128 _tmp2 = (__m128)__lsx_vilvl_w((__m128i)_f3, (__m128i)_f1);
                __m128 _tmp3 = (__m128)__lsx_vilvh_w((__m128i)_f3, (__m128i)_f1);

                *(int64_t*)p0 = __lsx_vpickve2gr_d((__m128i)_tmp0, 0);
                *(int64_t*)(p0 + out_hstep) = __lsx_vpickve2gr_d((__m128i)_tmp0, 1);
                *(int64_t*)(p0 + out_hstep * 2) = __lsx_vpickve2gr_d((__m128i)_tmp1, 0);
                *(int64_t*)(p0 + out_hstep * 3) = __lsx_vpickve2gr_d((__m128i)_tmp1, 1);
                *(int64_t*)(p0 + out_hstep * 4) = __lsx_vpickve2gr_d((__m128i)_tmp2, 0);
                *(int64_t*)(p0 + out_hstep * 5) = __lsx_vpickve2gr_d((__m128i)_tmp2, 1);
                *(int64_t*)(p0 + out_hstep * 6) = __lsx_vpickve2gr_d((__m128i)_tmp3, 0);
                *(int64_t*)(p0 + out_hstep * 7) = __lsx_vpickve2gr_d((__m128i)_tmp3, 1);
                p0 += out_hstep * 8;
            }
            else
            {
                __lsx_vst((__m128i)_f0, p0, 0);
                __lsx_vst((__m128i)_f1, p0 + 4, 0);
                __lsx_vst((__m128i)_f2, p0 + out_hstep, 0);
                __lsx_vst((__m128i)_f3, p0 + out_hstep + 4, 0);
                p0 += 8;
            }
            pp += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __builtin_prefetch(pp + 8);
            __m128i _sum0 = __lsx_vld(pp, 0);
            __m128i _sum1 = __lsx_vld(pp + 4, 0);

            __m128i _sum00 = __lsx_vpickev_w(_sum1, _sum0);
            __m128i _sum10 = __lsx_vpickod_w(_sum1, _sum0);

            __m128 _f0 = (__m128)__lsx_vffint_s_w(_sum00);
            __m128 _f1 = (__m128)__lsx_vffint_s_w(_sum10);

            _f0 = __lsx_vfmul_s(_f0, _descale0);
            _f1 = __lsx_vfmul_s(_f1, _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(c0);
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(c0));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vreplfr2vr_s(c1));
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vfmul_s((__m128)__lsx_vld(pC, 0), _beta));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vfmul_s((__m128)__lsx_vld(pC + c_hstep, 0), _beta));
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c0 = __lsx_vfmul_s((__m128)__lsx_vld(pC, 0), __lsx_vreplfr2vr_s(beta));
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
            }

            if (output_transpose)
            {
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_f1, (__m128i)_f0);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_f1, (__m128i)_f0);

                *(int64_t*)p0 = __lsx_vpickve2gr_d((__m128i)_tmp0, 0);
                *(int64_t*)(p0 + out_hstep) = __lsx_vpickve2gr_d((__m128i)_tmp0, 1);
                *(int64_t*)(p0 + out_hstep * 2) = __lsx_vpickve2gr_d((__m128i)_tmp1, 0);
                *(int64_t*)(p0 + out_hstep * 3) = __lsx_vpickve2gr_d((__m128i)_tmp1, 1);
                p0 += out_hstep * 4;
            }
            else
            {
                __lsx_vst((__m128i)_f0, p0, 0);
                __lsx_vst((__m128i)_f1, p0 + out_hstep, 0);
                p0 += 4;
            }
            pp += 8;
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            float f00 = pp[0] * descale0;
            float f10 = pp[1] * descale1;
            float f01 = pp[2] * descale0;
            float f11 = pp[3] * descale1;

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    f00 += c0;
                    f01 += c0;
                    f10 += c0;
                    f11 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f00 += c0;
                    f01 += c0;
                    f10 += c1;
                    f11 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    f00 += pC[0] * beta;
                    f01 += pC[1] * beta;
                    f10 += pC[c_hstep] * beta;
                    f11 += pC[c_hstep + 1] * beta;
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    f00 += pC[0] * beta;
                    f01 += pC[1] * beta;
                    f10 += pC[0] * beta;
                    f11 += pC[1] * beta;
                    pC += 2;
                }
            }

            f00 *= alpha;
            f01 *= alpha;
            f10 *= alpha;
            f11 *= alpha;

            if (output_transpose)
            {
                p0[0] = f00;
                p0[1] = f10;
                p0[out_hstep] = f01;
                p0[out_hstep + 1] = f11;
                p0 += out_hstep * 2;
            }
            else
            {
                p0[0] = f00;
                p0[1] = f01;
                p0[out_hstep] = f10;
                p0[out_hstep + 1] = f11;
                p0 += 2;
            }
            pp += 4;
        }
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[0] * descale0;
            float f1 = pp[1] * descale1;

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    f0 += c0;
                    f1 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                    f1 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    f0 += pC[0] * beta;
                    f1 += pC[c_hstep] * beta;
                    pC += 1;
                }
                if (broadcast_type_C == 4)
                {
                    f0 += pC[0] * beta;
                    f1 += pC[0] * beta;
                    pC += 1;
                }
            }

            f0 *= alpha;
            f1 *= alpha;

            if (output_transpose)
            {
                p0[0] = f0;
                p0[1] = f1;
                p0 += out_hstep;
            }
            else
            {
                p0[0] = f0;
                p0[out_hstep] = f1;
                p0 += 1;
            }
            pp += 2;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const int row = i + ii;
        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + row;
        }
        else
        {
            p0 = (float*)top_blob + row * out_hstep + j;
        }

        const float descale = descale_ptr[row];

        float c0 = 0.f;
        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + row;
                c0 = pC[0] * beta;
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (size_t)row * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if __loongarch_sx
        __m128 _descale = __lsx_vreplfr2vr_s(descale);
        for (; jj + 7 < max_jj; jj += 8)
        {
            __builtin_prefetch(pp + 8);
            __m128i _sum0 = __lsx_vld(pp, 0);
            __m128i _sum1 = __lsx_vld(pp + 4, 0);

            __m128 _f0 = (__m128)__lsx_vffint_s_w(_sum0);
            __m128 _f1 = (__m128)__lsx_vffint_s_w(_sum1);

            _f0 = __lsx_vfmul_s(_f0, _descale);
            _f1 = __lsx_vfmul_s(_f1, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c0 = __lsx_vreplfr2vr_s(c0);
                    _f0 = __lsx_vfadd_s(_f0, _c0);
                    _f1 = __lsx_vfadd_s(_f1, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    __m128 _beta = __lsx_vreplfr2vr_s(beta);
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vfmul_s((__m128)__lsx_vld(pC, 0), _beta));
                    _f1 = __lsx_vfadd_s(_f1, __lsx_vfmul_s((__m128)__lsx_vld(pC + 4, 0), _beta));
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = __lsx_vreplfr2vr_s(alpha);
                _f0 = __lsx_vfmul_s(_f0, _alpha);
                _f1 = __lsx_vfmul_s(_f1, _alpha);
            }

            if (output_transpose)
            {
                *(int*)p0 = __lsx_vpickve2gr_w((__m128i)_f0, 0);
                *(int*)(p0 + out_hstep) = __lsx_vpickve2gr_w((__m128i)_f0, 1);
                *(int*)(p0 + out_hstep * 2) = __lsx_vpickve2gr_w((__m128i)_f0, 2);
                *(int*)(p0 + out_hstep * 3) = __lsx_vpickve2gr_w((__m128i)_f0, 3);
                *(int*)(p0 + out_hstep * 4) = __lsx_vpickve2gr_w((__m128i)_f1, 0);
                *(int*)(p0 + out_hstep * 5) = __lsx_vpickve2gr_w((__m128i)_f1, 1);
                *(int*)(p0 + out_hstep * 6) = __lsx_vpickve2gr_w((__m128i)_f1, 2);
                *(int*)(p0 + out_hstep * 7) = __lsx_vpickve2gr_w((__m128i)_f1, 3);
                p0 += out_hstep * 8;
            }
            else
            {
                __lsx_vst((__m128i)_f0, p0, 0);
                __lsx_vst((__m128i)_f1, p0 + 4, 0);
                p0 += 8;
            }
            pp += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __builtin_prefetch(pp + 4);
            __m128i _sum0 = __lsx_vld(pp, 0);
            __m128 _f0 = (__m128)__lsx_vffint_s_w(_sum0);

            _f0 = __lsx_vfmul_s(_f0, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vreplfr2vr_s(c0));
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    _f0 = __lsx_vfadd_s(_f0, __lsx_vfmul_s((__m128)__lsx_vld(pC, 0), __lsx_vreplfr2vr_s(beta)));
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                _f0 = __lsx_vfmul_s(_f0, __lsx_vreplfr2vr_s(alpha));
            }

            if (output_transpose)
            {
                *(int*)p0 = __lsx_vpickve2gr_w((__m128i)_f0, 0);
                *(int*)(p0 + out_hstep) = __lsx_vpickve2gr_w((__m128i)_f0, 1);
                *(int*)(p0 + out_hstep * 2) = __lsx_vpickve2gr_w((__m128i)_f0, 2);
                *(int*)(p0 + out_hstep * 3) = __lsx_vpickve2gr_w((__m128i)_f0, 3);
                p0 += out_hstep * 4;
            }
            else
            {
                __lsx_vst((__m128i)_f0, p0, 0);
                p0 += 4;
            }
            pp += 4;
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            float f0 = pp[0] * descale;
            float f1 = pp[1] * descale;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                    f1 += c0;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    f0 += pC[0] * beta;
                    f1 += pC[1] * beta;
                    pC += 2;
                }
            }

            f0 *= alpha;
            f1 *= alpha;

            if (output_transpose)
            {
                p0[0] = f0;
                p0[out_hstep] = f1;
                p0 += out_hstep * 2;
            }
            else
            {
                p0[0] = f0;
                p0[1] = f1;
                p0 += 2;
            }
            pp += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[0] * descale;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    f0 += pC[0] * beta;
                    pC += 1;
                }
            }

            f0 *= alpha;

            p0[0] = f0;

            if (output_transpose)
                p0 += out_hstep;
            else
                p0 += 1;
            pp += 1;
        }
    }
}

static void get_optimal_tile_mnk_int8(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrtf((float)l2_cache_size / (2 * sizeof(signed char) + sizeof(int)));

#if __loongarch_sx
    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_N = std::max(8, tile_size / 8 * 8);
    TILE_K = std::max(8, tile_size / 8 * 8);
#else
    TILE_M = std::max(2, tile_size / 2 * 2);
    TILE_N = std::max(1, tile_size);
    TILE_K = std::max(2, tile_size / 2 * 2);
#endif

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __loongarch_sx
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(signed char) / TILE_K);

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

    // always take constant TILE_M/N/K value when provided
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
        TILE_K = (constant_TILE_K + 7) / 8 * 8;
#else
        TILE_K = (constant_TILE_K + 1) / 2 * 2;
#endif
    }
}

struct gemm_loongarch_int8_omp_args
{
    int TILE_M;
    int TILE_N;
    int TILE_K;
    int broadcast_type_C;
    int transA;
    int output_transpose;
    float alpha;
    float beta;
};

static int gemm_loongarch_int8(const Mat& A, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int transA, int transB, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h);
    const int K = transA ? (A.dims == 3 ? A.c : A.h) : A.w;
    const int N = transB ? (B.dims == 3 ? B.c : B.h) : B.w;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat ATX(TILE_K * TILE_M, nn_K, nT, 1u, opt.workspace_allocator);
    if (ATX.empty())
        return -100;

    Mat BT(TILE_K * TILE_N, nn_K, nn_N, 1u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    Mat A_int8_scales(M, 4u, opt.workspace_allocator);
    if (A_int8_scales.empty())
        return -100;

    float B_int8_scale;
    compute_B_int8_scale(B, B_int8_scale);

    Mat output_descales(M, 4u, opt.workspace_allocator);
    if (output_descales.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;
    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min(N - j, TILE_N);
        const int max_kk = std::min(K - k, TILE_K);

        Mat BT_tile = BT.channel(ppj).row_range(ppk, 1);

        if (transB)
            pack_B_tile_fp32_to_int8(B, BT_tile, j, max_jj, k, max_kk, B_int8_scale);
        else
            transpose_pack_B_tile_fp32_to_int8(B, BT_tile, j, max_jj, k, max_kk, B_int8_scale);
    }

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    const struct gemm_loongarch_int8_omp_args args = {TILE_M, TILE_N, TILE_K, broadcast_type_C, transA, output_transpose, alpha, beta};

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int TILE_M = args.TILE_M;
        const int TILE_N = args.TILE_N;
        const int TILE_K = args.TILE_K;
        const int broadcast_type_C = args.broadcast_type_C;
        const int transA = args.transA;
        const int output_transpose = args.output_transpose;
        const float alpha = args.alpha;
        const float beta = args.beta;

        const int i = ppi * TILE_M;
        const int max_ii = std::min(M - i, TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min(K - k, TILE_K);

                Mat AT_tile = ATX.channel(get_omp_thread_num()).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                if (j == 0)
                {
                    if (k == 0)
                    {
                        if (transA)
                            transpose_compute_A_tile_int8_scales(A, A_int8_scales, B_int8_scale, output_descales, i, max_ii);
                        else
                            compute_A_tile_int8_scales(A, A_int8_scales, B_int8_scale, output_descales, i, max_ii);
                    }

                    if (transA)
                        transpose_pack_A_tile_fp32_to_int8(A, AT_tile, i, max_ii, k, max_kk, A_int8_scales);
                    else
                        pack_A_tile_fp32_to_int8(A, AT_tile, i, max_ii, k, max_kk, A_int8_scales);
                }

                gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_int32_to_fp32(topT_tile, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, output_descales, alpha, beta, output_transpose);
        }
    }

    return 0;
}

static int gemm_AT_loongarch_int8(const Mat& AT, const Mat& A_int8_scales, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int K, int transB, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int N = transB ? (B.dims == 3 ? B.c : B.h) : B.w;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat BT(TILE_K * TILE_N, nn_K, nn_N, 1u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    float B_int8_scale;
    compute_B_int8_scale(B, B_int8_scale);

    Mat output_descales(M, 4u, opt.workspace_allocator);
    if (output_descales.empty())
        return -100;

    for (int i = 0; i < M; i++)
    {
        output_descales[i] = 1.f / (A_int8_scales[i] * B_int8_scale);
    }

    const int nn_NK = nn_N * nn_K;
    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min(N - j, TILE_N);
        const int max_kk = std::min(K - k, TILE_K);

        Mat BT_tile = BT.channel(ppj).row_range(ppk, 1);

        if (transB)
            pack_B_tile_fp32_to_int8(B, BT_tile, j, max_jj, k, max_kk, B_int8_scale);
        else
            transpose_pack_B_tile_fp32_to_int8(B, BT_tile, j, max_jj, k, max_kk, B_int8_scale);
    }

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    const struct gemm_loongarch_int8_omp_args args = {TILE_M, TILE_N, TILE_K, broadcast_type_C, 0, output_transpose, alpha, beta};

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int TILE_M = args.TILE_M;
        const int TILE_N = args.TILE_N;
        const int TILE_K = args.TILE_K;
        const int broadcast_type_C = args.broadcast_type_C;
        const int output_transpose = args.output_transpose;
        const float alpha = args.alpha;
        const float beta = args.beta;

        const int i = ppi * TILE_M;
        const int max_ii = std::min(M - i, TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min(K - k, TILE_K);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_int32_to_fp32(topT_tile, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, output_descales, alpha, beta, output_transpose);
        }
    }

    return 0;
}

static int gemm_BT_loongarch_int8(const Mat& A, const Mat& BT, float B_int8_scale, const Mat& C, Mat& top_blob, int broadcast_type_C, int N, int K, int transA, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat A_int8_scales(M, 4u, opt.workspace_allocator);
    if (A_int8_scales.empty())
        return -100;

    Mat output_descales(M, 4u, opt.workspace_allocator);
    if (output_descales.empty())
        return -100;

    Mat ATX(TILE_K * TILE_M, nn_K, nT, 1u, opt.workspace_allocator);
    if (ATX.empty())
        return -100;

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    const struct gemm_loongarch_int8_omp_args args = {TILE_M, TILE_N, TILE_K, broadcast_type_C, transA, output_transpose, alpha, beta};

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int TILE_M = args.TILE_M;
        const int TILE_N = args.TILE_N;
        const int TILE_K = args.TILE_K;
        const int broadcast_type_C = args.broadcast_type_C;
        const int transA = args.transA;
        const int output_transpose = args.output_transpose;
        const float alpha = args.alpha;
        const float beta = args.beta;

        const int i = ppi * TILE_M;
        const int max_ii = std::min(M - i, TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min(K - k, TILE_K);

                Mat AT_tile = ATX.channel(get_omp_thread_num()).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                if (j == 0)
                {
                    if (k == 0)
                    {
                        if (transA)
                            transpose_compute_A_tile_int8_scales(A, A_int8_scales, B_int8_scale, output_descales, i, max_ii);
                        else
                            compute_A_tile_int8_scales(A, A_int8_scales, B_int8_scale, output_descales, i, max_ii);
                    }

                    if (transA)
                        transpose_pack_A_tile_fp32_to_int8(A, AT_tile, i, max_ii, k, max_kk, A_int8_scales);
                    else
                        pack_A_tile_fp32_to_int8(A, AT_tile, i, max_ii, k, max_kk, A_int8_scales);
                }

                gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_int32_to_fp32(topT_tile, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, output_descales, alpha, beta, output_transpose);
        }
    }

    return 0;
}

static int gemm_AT_BT_loongarch_int8(const Mat& AT, const Mat& A_int8_scales, const Mat& BT, float B_int8_scale, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int N, int K, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat output_descales(M, 4u, opt.workspace_allocator);
    if (output_descales.empty())
        return -100;

    for (int i = 0; i < M; i++)
    {
        output_descales[i] = 1.f / (A_int8_scales[i] * B_int8_scale);
    }

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    const struct gemm_loongarch_int8_omp_args args = {TILE_M, TILE_N, TILE_K, broadcast_type_C, 0, output_transpose, alpha, beta};

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int TILE_M = args.TILE_M;
        const int TILE_N = args.TILE_N;
        const int TILE_K = args.TILE_K;
        const int broadcast_type_C = args.broadcast_type_C;
        const int output_transpose = args.output_transpose;
        const float alpha = args.alpha;
        const float beta = args.beta;

        const int i = ppi * TILE_M;
        const int max_ii = std::min(M - i, TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min(K - k, TILE_K);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_int32_to_fp32(topT_tile, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, output_descales, alpha, beta, output_transpose);
        }
    }

    return 0;
}

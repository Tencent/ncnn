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
#if __mips_msa
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
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
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
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* p0 = (const signed char*)A + (size_t)k * A_hstep + i + ii;

        int kk = 0;
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
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* p0 = (const signed char*)A + (size_t)k * A_hstep + i + ii;

        int kk = 0;
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
#if __mips_msa
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
#endif // __mips_msa
    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = (const signed char*)B + (j + jj) * B_hstep + k;
        const signed char* p1 = (const signed char*)B + (j + jj + 1) * B_hstep + k;

        int kk = 0;
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
#if __mips_msa
    for (; jj + 7 < max_jj; jj += 8)
    {
        const signed char* p0 = (const signed char*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
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
#endif // __mips_msa
    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = (const signed char*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
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
        {
            absmax = std::max(absmax, (float)fabs(ptr[kk]));
        }

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
#if __mips_msa
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

        int kk = 0;
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

        int kk = 0;
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
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];

        int kk = 0;
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
#if __mips_msa
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

        int kk = 0;
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

        int kk = 0;
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
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;
        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];

        int kk = 0;
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
        {
            absmax = std::max(absmax, (float)fabs(ptr[x]));
        }
    }

    scale = absmax == 0.f ? 1.f : 127.f / absmax;
}

static void pack_B_tile_fp32_to_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    signed char* pp = BT;

    int jj = 0;
#if __mips_msa
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
#endif // __mips_msa
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

        int kk = 0;
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
#if __mips_msa
    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
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
#endif // __mips_msa
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
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

    const signed char* pAT = AT_tile;
    const signed char* pBT = BT_tile;

    int* outptr = topT_tile;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* pB = pBT;

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

            v4i32 _sum00;
            v4i32 _sum01;
            v4i32 _sum10;
            v4i32 _sum11;
            v4i32 _sum20;
            v4i32 _sum21;
            v4i32 _sum30;
            v4i32 _sum31;
            v4i32 _sum40;
            v4i32 _sum41;
            v4i32 _sum50;
            v4i32 _sum51;
            v4i32 _sum60;
            v4i32 _sum61;
            v4i32 _sum70;
            v4i32 _sum71;

            if (k == 0)
            {
                _sum00 = __msa_fill_w(0);
                _sum01 = __msa_fill_w(0);
                _sum10 = __msa_fill_w(0);
                _sum11 = __msa_fill_w(0);
                _sum20 = __msa_fill_w(0);
                _sum21 = __msa_fill_w(0);
                _sum30 = __msa_fill_w(0);
                _sum31 = __msa_fill_w(0);
                _sum40 = __msa_fill_w(0);
                _sum41 = __msa_fill_w(0);
                _sum50 = __msa_fill_w(0);
                _sum51 = __msa_fill_w(0);
                _sum60 = __msa_fill_w(0);
                _sum61 = __msa_fill_w(0);
                _sum70 = __msa_fill_w(0);
                _sum71 = __msa_fill_w(0);
            }
            else
            {
                _sum00 = __msa_ld_w(outptr, 0);
                _sum01 = __msa_ld_w(outptr + 4, 0);
                _sum10 = __msa_ld_w(outptr + 8, 0);
                _sum11 = __msa_ld_w(outptr + 12, 0);
                _sum20 = __msa_ld_w(outptr + 16, 0);
                _sum21 = __msa_ld_w(outptr + 20, 0);
                _sum30 = __msa_ld_w(outptr + 24, 0);
                _sum31 = __msa_ld_w(outptr + 28, 0);
                _sum40 = __msa_ld_w(outptr + 32, 0);
                _sum41 = __msa_ld_w(outptr + 36, 0);
                _sum50 = __msa_ld_w(outptr + 40, 0);
                _sum51 = __msa_ld_w(outptr + 44, 0);
                _sum60 = __msa_ld_w(outptr + 48, 0);
                _sum61 = __msa_ld_w(outptr + 52, 0);
                _sum70 = __msa_ld_w(outptr + 56, 0);
                _sum71 = __msa_ld_w(outptr + 60, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                v8i16 _pA0 = (v8i16)__msa_fill_d(*(int*)pA);
                _pA0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA0, 0), (v16i8)_pA0);
                v8i16 _pA1 = (v8i16)__msa_fill_d(*(int*)(pA + 4));
                _pA1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA1, 0), (v16i8)_pA1);

                v8i16 _pB0 = __msa_fill_h(pB[0]);
                v8i16 _s0 = __msa_mulv_h(_pA0, _pB0);
                v8i16 _s1 = __msa_mulv_h(_pA1, _pB0);
                _sum00 = __msa_addv_w(_sum00, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum01 = __msa_addv_w(_sum01, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));

                v8i16 _pB1 = __msa_fill_h(pB[1]);
                _s0 = __msa_mulv_h(_pA0, _pB1);
                _s1 = __msa_mulv_h(_pA1, _pB1);
                _sum10 = __msa_addv_w(_sum10, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum11 = __msa_addv_w(_sum11, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));

                v8i16 _pB2 = __msa_fill_h(pB[2]);
                _s0 = __msa_mulv_h(_pA0, _pB2);
                _s1 = __msa_mulv_h(_pA1, _pB2);
                _sum20 = __msa_addv_w(_sum20, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum21 = __msa_addv_w(_sum21, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));

                v8i16 _pB3 = __msa_fill_h(pB[3]);
                _s0 = __msa_mulv_h(_pA0, _pB3);
                _s1 = __msa_mulv_h(_pA1, _pB3);
                _sum30 = __msa_addv_w(_sum30, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum31 = __msa_addv_w(_sum31, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));

                v8i16 _pB4 = __msa_fill_h(pB[4]);
                _s0 = __msa_mulv_h(_pA0, _pB4);
                _s1 = __msa_mulv_h(_pA1, _pB4);
                _sum40 = __msa_addv_w(_sum40, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum41 = __msa_addv_w(_sum41, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));

                v8i16 _pB5 = __msa_fill_h(pB[5]);
                _s0 = __msa_mulv_h(_pA0, _pB5);
                _s1 = __msa_mulv_h(_pA1, _pB5);
                _sum50 = __msa_addv_w(_sum50, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum51 = __msa_addv_w(_sum51, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));

                v8i16 _pB6 = __msa_fill_h(pB[6]);
                _s0 = __msa_mulv_h(_pA0, _pB6);
                _s1 = __msa_mulv_h(_pA1, _pB6);
                _sum60 = __msa_addv_w(_sum60, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum61 = __msa_addv_w(_sum61, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));

                v8i16 _pB7 = __msa_fill_h(pB[7]);
                _s0 = __msa_mulv_h(_pA0, _pB7);
                _s1 = __msa_mulv_h(_pA1, _pB7);
                _sum70 = __msa_addv_w(_sum70, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum71 = __msa_addv_w(_sum71, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));

                pA += 8;
                pB += 8;
            }

            __msa_st_w(_sum00, outptr, 0);
            __msa_st_w(_sum01, outptr + 4, 0);
            __msa_st_w(_sum10, outptr + 8, 0);
            __msa_st_w(_sum11, outptr + 12, 0);
            __msa_st_w(_sum20, outptr + 16, 0);
            __msa_st_w(_sum21, outptr + 20, 0);
            __msa_st_w(_sum30, outptr + 24, 0);
            __msa_st_w(_sum31, outptr + 28, 0);
            __msa_st_w(_sum40, outptr + 32, 0);
            __msa_st_w(_sum41, outptr + 36, 0);
            __msa_st_w(_sum50, outptr + 40, 0);
            __msa_st_w(_sum51, outptr + 44, 0);
            __msa_st_w(_sum60, outptr + 48, 0);
            __msa_st_w(_sum61, outptr + 52, 0);
            __msa_st_w(_sum70, outptr + 56, 0);
            __msa_st_w(_sum71, outptr + 60, 0);

            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            v4i32 _sum00;
            v4i32 _sum01;
            v4i32 _sum10;
            v4i32 _sum11;
            v4i32 _sum20;
            v4i32 _sum21;
            v4i32 _sum30;
            v4i32 _sum31;

            if (k == 0)
            {
                _sum00 = __msa_fill_w(0);
                _sum01 = __msa_fill_w(0);
                _sum10 = __msa_fill_w(0);
                _sum11 = __msa_fill_w(0);
                _sum20 = __msa_fill_w(0);
                _sum21 = __msa_fill_w(0);
                _sum30 = __msa_fill_w(0);
                _sum31 = __msa_fill_w(0);
            }
            else
            {
                _sum00 = __msa_ld_w(outptr, 0);
                _sum01 = __msa_ld_w(outptr + 4, 0);
                _sum10 = __msa_ld_w(outptr + 8, 0);
                _sum11 = __msa_ld_w(outptr + 12, 0);
                _sum20 = __msa_ld_w(outptr + 16, 0);
                _sum21 = __msa_ld_w(outptr + 20, 0);
                _sum30 = __msa_ld_w(outptr + 24, 0);
                _sum31 = __msa_ld_w(outptr + 28, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                v8i16 _pA0 = (v8i16)__msa_fill_d(*(int*)pA);
                _pA0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA0, 0), (v16i8)_pA0);
                v8i16 _pA1 = (v8i16)__msa_fill_d(*(int*)(pA + 4));
                _pA1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA1, 0), (v16i8)_pA1);

                v8i16 _pB0 = __msa_fill_h(pB[0]);
                v8i16 _s0 = __msa_mulv_h(_pA0, _pB0);
                v8i16 _s1 = __msa_mulv_h(_pA1, _pB0);
                _sum00 = __msa_addv_w(_sum00, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum01 = __msa_addv_w(_sum01, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));

                v8i16 _pB1 = __msa_fill_h(pB[1]);
                _s0 = __msa_mulv_h(_pA0, _pB1);
                _s1 = __msa_mulv_h(_pA1, _pB1);
                _sum10 = __msa_addv_w(_sum10, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum11 = __msa_addv_w(_sum11, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));

                v8i16 _pB2 = __msa_fill_h(pB[2]);
                _s0 = __msa_mulv_h(_pA0, _pB2);
                _s1 = __msa_mulv_h(_pA1, _pB2);
                _sum20 = __msa_addv_w(_sum20, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum21 = __msa_addv_w(_sum21, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));

                v8i16 _pB3 = __msa_fill_h(pB[3]);
                _s0 = __msa_mulv_h(_pA0, _pB3);
                _s1 = __msa_mulv_h(_pA1, _pB3);
                _sum30 = __msa_addv_w(_sum30, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum31 = __msa_addv_w(_sum31, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));

                pA += 8;
                pB += 4;
            }

            __msa_st_w(_sum00, outptr, 0);
            __msa_st_w(_sum01, outptr + 4, 0);
            __msa_st_w(_sum10, outptr + 8, 0);
            __msa_st_w(_sum11, outptr + 12, 0);
            __msa_st_w(_sum20, outptr + 16, 0);
            __msa_st_w(_sum21, outptr + 20, 0);
            __msa_st_w(_sum30, outptr + 24, 0);
            __msa_st_w(_sum31, outptr + 28, 0);

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            v4i32 _sum00;
            v4i32 _sum01;
            v4i32 _sum10;
            v4i32 _sum11;

            if (k == 0)
            {
                _sum00 = __msa_fill_w(0);
                _sum01 = __msa_fill_w(0);
                _sum10 = __msa_fill_w(0);
                _sum11 = __msa_fill_w(0);
            }
            else
            {
                _sum00 = __msa_ld_w(outptr, 0);
                _sum01 = __msa_ld_w(outptr + 4, 0);
                _sum10 = __msa_ld_w(outptr + 8, 0);
                _sum11 = __msa_ld_w(outptr + 12, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                v8i16 _pA0 = (v8i16)__msa_fill_d(*(int*)pA);
                _pA0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA0, 0), (v16i8)_pA0);
                v8i16 _pA1 = (v8i16)__msa_fill_d(*(int*)(pA + 4));
                _pA1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA1, 0), (v16i8)_pA1);

                v8i16 _pB0 = __msa_fill_h(pB[0]);
                v8i16 _s0 = __msa_mulv_h(_pA0, _pB0);
                v8i16 _s1 = __msa_mulv_h(_pA1, _pB0);
                _sum00 = __msa_addv_w(_sum00, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum01 = __msa_addv_w(_sum01, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));

                v8i16 _pB1 = __msa_fill_h(pB[1]);
                _s0 = __msa_mulv_h(_pA0, _pB1);
                _s1 = __msa_mulv_h(_pA1, _pB1);
                _sum10 = __msa_addv_w(_sum10, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum11 = __msa_addv_w(_sum11, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));

                pA += 8;
                pB += 2;
            }

            __msa_st_w(_sum00, outptr, 0);
            __msa_st_w(_sum01, outptr + 4, 0);
            __msa_st_w(_sum10, outptr + 8, 0);
            __msa_st_w(_sum11, outptr + 12, 0);

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            v4i32 _sum00;
            v4i32 _sum01;

            if (k == 0)
            {
                _sum00 = __msa_fill_w(0);
                _sum01 = __msa_fill_w(0);
            }
            else
            {
                _sum00 = __msa_ld_w(outptr, 0);
                _sum01 = __msa_ld_w(outptr + 4, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                v8i16 _pA0 = (v8i16)__msa_fill_d(*(int*)pA);
                _pA0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA0, 0), (v16i8)_pA0);
                v8i16 _pA1 = (v8i16)__msa_fill_d(*(int*)(pA + 4));
                _pA1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA1, 0), (v16i8)_pA1);

                v8i16 _pB0 = __msa_fill_h(pB[0]);
                v8i16 _s0 = __msa_mulv_h(_pA0, _pB0);
                v8i16 _s1 = __msa_mulv_h(_pA1, _pB0);
                _sum00 = __msa_addv_w(_sum00, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum01 = __msa_addv_w(_sum01, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));

                pA += 8;
                pB += 1;
            }

            __msa_st_w(_sum00, outptr, 0);
            __msa_st_w(_sum01, outptr + 4, 0);

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* pB = pBT;

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

            v4i32 _sum0;
            v4i32 _sum1;
            v4i32 _sum2;
            v4i32 _sum3;
            v4i32 _sum4;
            v4i32 _sum5;
            v4i32 _sum6;
            v4i32 _sum7;

            if (k == 0)
            {
                _sum0 = __msa_fill_w(0);
                _sum1 = __msa_fill_w(0);
                _sum2 = __msa_fill_w(0);
                _sum3 = __msa_fill_w(0);
                _sum4 = __msa_fill_w(0);
                _sum5 = __msa_fill_w(0);
                _sum6 = __msa_fill_w(0);
                _sum7 = __msa_fill_w(0);
            }
            else
            {
                _sum0 = __msa_ld_w(outptr, 0);
                _sum1 = __msa_ld_w(outptr + 4, 0);
                _sum2 = __msa_ld_w(outptr + 8, 0);
                _sum3 = __msa_ld_w(outptr + 12, 0);
                _sum4 = __msa_ld_w(outptr + 16, 0);
                _sum5 = __msa_ld_w(outptr + 20, 0);
                _sum6 = __msa_ld_w(outptr + 24, 0);
                _sum7 = __msa_ld_w(outptr + 28, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                v8i16 _pA = (v8i16)__msa_fill_d(*(int*)pA);
                _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);

                v8i16 _pB0 = __msa_fill_h(pB[0]);
                v8i16 _pB1 = __msa_fill_h(pB[1]);
                v8i16 _pB2 = __msa_fill_h(pB[2]);
                v8i16 _pB3 = __msa_fill_h(pB[3]);
                v8i16 _pB4 = __msa_fill_h(pB[4]);
                v8i16 _pB5 = __msa_fill_h(pB[5]);
                v8i16 _pB6 = __msa_fill_h(pB[6]);
                v8i16 _pB7 = __msa_fill_h(pB[7]);

                v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                v8i16 _s1 = __msa_mulv_h(_pA, _pB1);
                v8i16 _s2 = __msa_mulv_h(_pA, _pB2);
                v8i16 _s3 = __msa_mulv_h(_pA, _pB3);
                v8i16 _s4 = __msa_mulv_h(_pA, _pB4);
                v8i16 _s5 = __msa_mulv_h(_pA, _pB5);
                v8i16 _s6 = __msa_mulv_h(_pA, _pB6);
                v8i16 _s7 = __msa_mulv_h(_pA, _pB7);

                _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s2, 0), _s2));
                _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s3, 0), _s3));
                _sum4 = __msa_addv_w(_sum4, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s4, 0), _s4));
                _sum5 = __msa_addv_w(_sum5, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s5, 0), _s5));
                _sum6 = __msa_addv_w(_sum6, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s6, 0), _s6));
                _sum7 = __msa_addv_w(_sum7, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s7, 0), _s7));

                pA += 4;
                pB += 8;
            }

            __msa_st_w(_sum0, outptr, 0);
            __msa_st_w(_sum1, outptr + 4, 0);
            __msa_st_w(_sum2, outptr + 8, 0);
            __msa_st_w(_sum3, outptr + 12, 0);
            __msa_st_w(_sum4, outptr + 16, 0);
            __msa_st_w(_sum5, outptr + 20, 0);
            __msa_st_w(_sum6, outptr + 24, 0);
            __msa_st_w(_sum7, outptr + 28, 0);

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            v4i32 _sum0;
            v4i32 _sum1;
            v4i32 _sum2;
            v4i32 _sum3;

            if (k == 0)
            {
                _sum0 = __msa_fill_w(0);
                _sum1 = __msa_fill_w(0);
                _sum2 = __msa_fill_w(0);
                _sum3 = __msa_fill_w(0);
            }
            else
            {
                _sum0 = __msa_ld_w(outptr, 0);
                _sum1 = __msa_ld_w(outptr + 4, 0);
                _sum2 = __msa_ld_w(outptr + 8, 0);
                _sum3 = __msa_ld_w(outptr + 12, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                v8i16 _pA = (v8i16)__msa_fill_d(*(int*)pA);
                _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);

                v8i16 _pB0 = __msa_fill_h(pB[0]);
                v8i16 _pB1 = __msa_fill_h(pB[1]);
                v8i16 _pB2 = __msa_fill_h(pB[2]);
                v8i16 _pB3 = __msa_fill_h(pB[3]);

                v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                v8i16 _s1 = __msa_mulv_h(_pA, _pB1);
                v8i16 _s2 = __msa_mulv_h(_pA, _pB2);
                v8i16 _s3 = __msa_mulv_h(_pA, _pB3);

                _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s2, 0), _s2));
                _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s3, 0), _s3));

                pA += 4;
                pB += 4;
            }

            __msa_st_w(_sum0, outptr, 0);
            __msa_st_w(_sum1, outptr + 4, 0);
            __msa_st_w(_sum2, outptr + 8, 0);
            __msa_st_w(_sum3, outptr + 12, 0);

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            v4i32 _sum0;
            v4i32 _sum1;

            if (k == 0)
            {
                _sum0 = __msa_fill_w(0);
                _sum1 = __msa_fill_w(0);
            }
            else
            {
                _sum0 = __msa_ld_w(outptr, 0);
                _sum1 = __msa_ld_w(outptr + 4, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                v8i16 _pA = (v8i16)__msa_fill_d(*(int*)pA);
                _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);

                v8i16 _pB0 = __msa_fill_h(pB[0]);
                v8i16 _pB1 = __msa_fill_h(pB[1]);

                v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                v8i16 _s1 = __msa_mulv_h(_pA, _pB1);

                _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));

                pA += 4;
                pB += 2;
            }

            __msa_st_w(_sum0, outptr, 0);
            __msa_st_w(_sum1, outptr + 4, 0);

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            v4i32 _sum0;

            if (k == 0)
            {
                _sum0 = __msa_fill_w(0);
            }
            else
            {
                _sum0 = __msa_ld_w(outptr, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                v8i16 _pA = (v8i16)__msa_fill_d(*(int*)pA);
                _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);

                v8i16 _pB0 = __msa_fill_h(pB[0]);

                v8i16 _s0 = __msa_mulv_h(_pA, _pB0);

                _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));

                pA += 4;
                pB += 1;
            }

            __msa_st_w(_sum0, outptr, 0);

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if __mips_msa
        for (; jj + 7 < max_jj; jj += 8)
        {
            int sum00;
            int sum01;
            int sum10;
            int sum11;
            int sum20;
            int sum21;
            int sum30;
            int sum31;
            int sum40;
            int sum41;
            int sum50;
            int sum51;
            int sum60;
            int sum61;
            int sum70;
            int sum71;

            if (k == 0)
            {
                sum00 = 0;
                sum01 = 0;
                sum10 = 0;
                sum11 = 0;
                sum20 = 0;
                sum21 = 0;
                sum30 = 0;
                sum31 = 0;
                sum40 = 0;
                sum41 = 0;
                sum50 = 0;
                sum51 = 0;
                sum60 = 0;
                sum61 = 0;
                sum70 = 0;
                sum71 = 0;
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

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum00 += pA[0] * pB[0];
                sum01 += pA[1] * pB[0];
                sum10 += pA[0] * pB[1];
                sum11 += pA[1] * pB[1];
                sum20 += pA[0] * pB[2];
                sum21 += pA[1] * pB[2];
                sum30 += pA[0] * pB[3];
                sum31 += pA[1] * pB[3];
                sum40 += pA[0] * pB[4];
                sum41 += pA[1] * pB[4];
                sum50 += pA[0] * pB[5];
                sum51 += pA[1] * pB[5];
                sum60 += pA[0] * pB[6];
                sum61 += pA[1] * pB[6];
                sum70 += pA[0] * pB[7];
                sum71 += pA[1] * pB[7];

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
        for (; jj + 3 < max_jj; jj += 4)
        {
            int sum00;
            int sum01;
            int sum10;
            int sum11;
            int sum20;
            int sum21;
            int sum30;
            int sum31;

            if (k == 0)
            {
                sum00 = 0;
                sum01 = 0;
                sum10 = 0;
                sum11 = 0;
                sum20 = 0;
                sum21 = 0;
                sum30 = 0;
                sum31 = 0;
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
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum00 += pA[0] * pB[0];
                sum01 += pA[1] * pB[0];
                sum10 += pA[0] * pB[1];
                sum11 += pA[1] * pB[1];
                sum20 += pA[0] * pB[2];
                sum21 += pA[1] * pB[2];
                sum30 += pA[0] * pB[3];
                sum31 += pA[1] * pB[3];

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
#endif // __mips_msa
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
            for (; kk < max_kk; kk += 1)
            {
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
            for (; kk < max_kk; kk += 1)
            {
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
#if __mips_msa
        for (; jj + 7 < max_jj; jj += 8)
        {
            int sum0;
            int sum1;
            int sum2;
            int sum3;
            int sum4;
            int sum5;
            int sum6;
            int sum7;

            if (k == 0)
            {
                sum0 = 0;
                sum1 = 0;
                sum2 = 0;
                sum3 = 0;
                sum4 = 0;
                sum5 = 0;
                sum6 = 0;
                sum7 = 0;
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

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];
                sum2 += pA[0] * pB[2];
                sum3 += pA[0] * pB[3];
                sum4 += pA[0] * pB[4];
                sum5 += pA[0] * pB[5];
                sum6 += pA[0] * pB[6];
                sum7 += pA[0] * pB[7];

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
        for (; jj + 3 < max_jj; jj += 4)
        {
            int sum0;
            int sum1;
            int sum2;
            int sum3;

            if (k == 0)
            {
                sum0 = 0;
                sum1 = 0;
                sum2 = 0;
                sum3 = 0;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
                sum2 = outptr[2];
                sum3 = outptr[3];
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];
                sum2 += pA[0] * pB[2];
                sum3 += pA[0] * pB[3];

                pA += 1;
                pB += 4;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr[2] = sum2;
            outptr[3] = sum3;

            outptr += 4;
        }
#endif // __mips_msa
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
            for (; kk < max_kk; kk += 1)
            {
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
    float* outptr = top_blob;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const float* pC = C;
    const float* descale_ptr = descales;

    const int* pp = topT;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            for (int c = 0; c < 8; c++)
            {
                for (int r = 0; r < 8; r++)
                {
                    const int row = i + ii + r;
                    const int col = j + jj + c;
                    float v = pp[c * 8 + r] * descale_ptr[row];
                    if (pC)
                    {
                        if (broadcast_type_C == 0)
                            v += pC[0] * beta;
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                            v += pC[row] * beta;
                        if (broadcast_type_C == 3)
                            v += pC[(size_t)row * c_hstep + col] * beta;
                        if (broadcast_type_C == 4)
                            v += pC[col] * beta;
                    }
                    v *= alpha;
                    if (output_transpose)
                        outptr[(size_t)col * out_hstep + row] = v;
                    else
                        outptr[(size_t)row * out_hstep + col] = v;
                }
            }
            pp += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            for (int c = 0; c < 4; c++)
            {
                for (int r = 0; r < 8; r++)
                {
                    const int row = i + ii + r;
                    const int col = j + jj + c;
                    float v = pp[c * 8 + r] * descale_ptr[row];
                    if (pC)
                    {
                        if (broadcast_type_C == 0)
                            v += pC[0] * beta;
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                            v += pC[row] * beta;
                        if (broadcast_type_C == 3)
                            v += pC[(size_t)row * c_hstep + col] * beta;
                        if (broadcast_type_C == 4)
                            v += pC[col] * beta;
                    }
                    v *= alpha;
                    if (output_transpose)
                        outptr[(size_t)col * out_hstep + row] = v;
                    else
                        outptr[(size_t)row * out_hstep + col] = v;
                }
            }
            pp += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            for (int c = 0; c < 2; c++)
            {
                for (int r = 0; r < 8; r++)
                {
                    const int row = i + ii + r;
                    const int col = j + jj + c;
                    float v = pp[c * 8 + r] * descale_ptr[row];
                    if (pC)
                    {
                        if (broadcast_type_C == 0)
                            v += pC[0] * beta;
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                            v += pC[row] * beta;
                        if (broadcast_type_C == 3)
                            v += pC[(size_t)row * c_hstep + col] * beta;
                        if (broadcast_type_C == 4)
                            v += pC[col] * beta;
                    }
                    v *= alpha;
                    if (output_transpose)
                        outptr[(size_t)col * out_hstep + row] = v;
                    else
                        outptr[(size_t)row * out_hstep + col] = v;
                }
            }
            pp += 16;
        }
        for (; jj < max_jj; jj++)
        {
            for (int r = 0; r < 8; r++)
            {
                const int row = i + ii + r;
                const int col = j + jj;
                float v = pp[r] * descale_ptr[row];
                if (pC)
                {
                    if (broadcast_type_C == 0)
                        v += pC[0] * beta;
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        v += pC[row] * beta;
                    if (broadcast_type_C == 3)
                        v += pC[(size_t)row * c_hstep + col] * beta;
                    if (broadcast_type_C == 4)
                        v += pC[col] * beta;
                }
                v *= alpha;
                if (output_transpose)
                    outptr[(size_t)col * out_hstep + row] = v;
                else
                    outptr[(size_t)row * out_hstep + col] = v;
            }
            pp += 8;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            for (int c = 0; c < 8; c++)
            {
                for (int r = 0; r < 4; r++)
                {
                    const int row = i + ii + r;
                    const int col = j + jj + c;
                    float v = pp[c * 4 + r] * descale_ptr[row];
                    if (pC)
                    {
                        if (broadcast_type_C == 0)
                            v += pC[0] * beta;
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                            v += pC[row] * beta;
                        if (broadcast_type_C == 3)
                            v += pC[(size_t)row * c_hstep + col] * beta;
                        if (broadcast_type_C == 4)
                            v += pC[col] * beta;
                    }
                    v *= alpha;
                    if (output_transpose)
                        outptr[(size_t)col * out_hstep + row] = v;
                    else
                        outptr[(size_t)row * out_hstep + col] = v;
                }
            }
            pp += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            for (int c = 0; c < 4; c++)
            {
                for (int r = 0; r < 4; r++)
                {
                    const int row = i + ii + r;
                    const int col = j + jj + c;
                    float v = pp[c * 4 + r] * descale_ptr[row];
                    if (pC)
                    {
                        if (broadcast_type_C == 0)
                            v += pC[0] * beta;
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                            v += pC[row] * beta;
                        if (broadcast_type_C == 3)
                            v += pC[(size_t)row * c_hstep + col] * beta;
                        if (broadcast_type_C == 4)
                            v += pC[col] * beta;
                    }
                    v *= alpha;
                    if (output_transpose)
                        outptr[(size_t)col * out_hstep + row] = v;
                    else
                        outptr[(size_t)row * out_hstep + col] = v;
                }
            }
            pp += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            for (int c = 0; c < 2; c++)
            {
                for (int r = 0; r < 4; r++)
                {
                    const int row = i + ii + r;
                    const int col = j + jj + c;
                    float v = pp[c * 4 + r] * descale_ptr[row];
                    if (pC)
                    {
                        if (broadcast_type_C == 0)
                            v += pC[0] * beta;
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                            v += pC[row] * beta;
                        if (broadcast_type_C == 3)
                            v += pC[(size_t)row * c_hstep + col] * beta;
                        if (broadcast_type_C == 4)
                            v += pC[col] * beta;
                    }
                    v *= alpha;
                    if (output_transpose)
                        outptr[(size_t)col * out_hstep + row] = v;
                    else
                        outptr[(size_t)row * out_hstep + col] = v;
                }
            }
            pp += 8;
        }
        for (; jj < max_jj; jj++)
        {
            for (int r = 0; r < 4; r++)
            {
                const int row = i + ii + r;
                const int col = j + jj;
                float v = pp[r] * descale_ptr[row];
                if (pC)
                {
                    if (broadcast_type_C == 0)
                        v += pC[0] * beta;
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        v += pC[row] * beta;
                    if (broadcast_type_C == 3)
                        v += pC[(size_t)row * c_hstep + col] * beta;
                    if (broadcast_type_C == 4)
                        v += pC[col] * beta;
                }
                v *= alpha;
                if (output_transpose)
                    outptr[(size_t)col * out_hstep + row] = v;
                else
                    outptr[(size_t)row * out_hstep + col] = v;
            }
            pp += 4;
        }
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        int jj = 0;
#if __mips_msa
        for (; jj + 7 < max_jj; jj += 8)
        {
            for (int c = 0; c < 8; c++)
            {
                for (int r = 0; r < 2; r++)
                {
                    const int row = i + ii + r;
                    const int col = j + jj + c;
                    float v = pp[c * 2 + r] * descale_ptr[row];
                    if (pC)
                    {
                        if (broadcast_type_C == 0)
                            v += pC[0] * beta;
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                            v += pC[row] * beta;
                        if (broadcast_type_C == 3)
                            v += pC[(size_t)row * c_hstep + col] * beta;
                        if (broadcast_type_C == 4)
                            v += pC[col] * beta;
                    }
                    v *= alpha;
                    if (output_transpose)
                        outptr[(size_t)col * out_hstep + row] = v;
                    else
                        outptr[(size_t)row * out_hstep + col] = v;
                }
            }
            pp += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            for (int c = 0; c < 4; c++)
            {
                for (int r = 0; r < 2; r++)
                {
                    const int row = i + ii + r;
                    const int col = j + jj + c;
                    float v = pp[c * 2 + r] * descale_ptr[row];
                    if (pC)
                    {
                        if (broadcast_type_C == 0)
                            v += pC[0] * beta;
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                            v += pC[row] * beta;
                        if (broadcast_type_C == 3)
                            v += pC[(size_t)row * c_hstep + col] * beta;
                        if (broadcast_type_C == 4)
                            v += pC[col] * beta;
                    }
                    v *= alpha;
                    if (output_transpose)
                        outptr[(size_t)col * out_hstep + row] = v;
                    else
                        outptr[(size_t)row * out_hstep + col] = v;
                }
            }
            pp += 8;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            for (int c = 0; c < 2; c++)
            {
                for (int r = 0; r < 2; r++)
                {
                    const int row = i + ii + r;
                    const int col = j + jj + c;
                    float v = pp[c * 2 + r] * descale_ptr[row];
                    if (pC)
                    {
                        if (broadcast_type_C == 0)
                            v += pC[0] * beta;
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                            v += pC[row] * beta;
                        if (broadcast_type_C == 3)
                            v += pC[(size_t)row * c_hstep + col] * beta;
                        if (broadcast_type_C == 4)
                            v += pC[col] * beta;
                    }
                    v *= alpha;
                    if (output_transpose)
                        outptr[(size_t)col * out_hstep + row] = v;
                    else
                        outptr[(size_t)row * out_hstep + col] = v;
                }
            }
            pp += 4;
        }
        for (; jj < max_jj; jj++)
        {
            for (int r = 0; r < 2; r++)
            {
                const int row = i + ii + r;
                const int col = j + jj;
                float v = pp[r] * descale_ptr[row];
                if (pC)
                {
                    if (broadcast_type_C == 0)
                        v += pC[0] * beta;
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        v += pC[row] * beta;
                    if (broadcast_type_C == 3)
                        v += pC[(size_t)row * c_hstep + col] * beta;
                    if (broadcast_type_C == 4)
                        v += pC[col] * beta;
                }
                v *= alpha;
                if (output_transpose)
                    outptr[(size_t)col * out_hstep + row] = v;
                else
                    outptr[(size_t)row * out_hstep + col] = v;
            }
            pp += 2;
        }
    }
    for (; ii < max_ii; ii++)
    {
        int jj = 0;
#if __mips_msa
        for (; jj + 7 < max_jj; jj += 8)
        {
            for (int c = 0; c < 8; c++)
            {
                const int row = i + ii;
                const int col = j + jj + c;
                float v = pp[c] * descale_ptr[row];
                if (pC)
                {
                    if (broadcast_type_C == 0)
                        v += pC[0] * beta;
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        v += pC[row] * beta;
                    if (broadcast_type_C == 3)
                        v += pC[(size_t)row * c_hstep + col] * beta;
                    if (broadcast_type_C == 4)
                        v += pC[col] * beta;
                }
                v *= alpha;
                if (output_transpose)
                    outptr[(size_t)col * out_hstep + row] = v;
                else
                    outptr[(size_t)row * out_hstep + col] = v;
            }
            pp += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            for (int c = 0; c < 4; c++)
            {
                const int row = i + ii;
                const int col = j + jj + c;
                float v = pp[c] * descale_ptr[row];
                if (pC)
                {
                    if (broadcast_type_C == 0)
                        v += pC[0] * beta;
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        v += pC[row] * beta;
                    if (broadcast_type_C == 3)
                        v += pC[(size_t)row * c_hstep + col] * beta;
                    if (broadcast_type_C == 4)
                        v += pC[col] * beta;
                }
                v *= alpha;
                if (output_transpose)
                    outptr[(size_t)col * out_hstep + row] = v;
                else
                    outptr[(size_t)row * out_hstep + col] = v;
            }
            pp += 4;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            for (int c = 0; c < 2; c++)
            {
                const int row = i + ii;
                const int col = j + jj + c;
                float v = pp[c] * descale_ptr[row];
                if (pC)
                {
                    if (broadcast_type_C == 0)
                        v += pC[0] * beta;
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        v += pC[row] * beta;
                    if (broadcast_type_C == 3)
                        v += pC[(size_t)row * c_hstep + col] * beta;
                    if (broadcast_type_C == 4)
                        v += pC[col] * beta;
                }
                v *= alpha;
                if (output_transpose)
                    outptr[(size_t)col * out_hstep + row] = v;
                else
                    outptr[(size_t)row * out_hstep + col] = v;
            }
            pp += 2;
        }
        for (; jj < max_jj; jj++)
        {
            const int row = i + ii;
            const int col = j + jj;
            float v = pp[0] * descale_ptr[row];
            if (pC)
            {
                if (broadcast_type_C == 0)
                    v += pC[0] * beta;
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    v += pC[row] * beta;
                if (broadcast_type_C == 3)
                    v += pC[(size_t)row * c_hstep + col] * beta;
                if (broadcast_type_C == 4)
                    v += pC[col] * beta;
            }
            v *= alpha;
            if (output_transpose)
                outptr[(size_t)col * out_hstep + row] = v;
            else
                outptr[(size_t)row * out_hstep + col] = v;
            pp += 1;
        }
    }
}

static void get_optimal_tile_mnk_int8(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    const int l2_cache_size_int8 = (int)(get_cpu_level2_cache_size() / sizeof(signed char));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    {
#if __mips_msa
        int tile_size = (l2_cache_size_int8 - 16) / 8;
        TILE_K = std::max(8, tile_size / 8 * 8);
#else
        int tile_size = (l2_cache_size_int8 - 2) / 3;
        TILE_K = std::max(2, tile_size / 2 * 2);
#endif

        if (K > 0)
        {
            int nn_K = (K + TILE_K - 1) / TILE_K;
#if __mips_msa
            TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);
#else
            TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif
        }
    }

#if __mips_msa
    TILE_M = 8;
#else
    TILE_M = 2;
#endif
    if (M > 0)
    {
        TILE_M *= std::min(nT, get_physical_cpu_count());
        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __mips_msa
        TILE_M = std::max(8, std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8));
#else
        TILE_M = std::max(2, std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2));
#endif
        if (nT > 1)
        {
#if __mips_msa
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#else
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
        }
    }

    if (N > 0)
    {
        int tile_size = TILE_K >= K ? (l2_cache_size_int8 - TILE_M * TILE_K) / std::max(1, TILE_K) : (l2_cache_size_int8 - TILE_M * TILE_K) / std::max(1, TILE_M + TILE_K);
#if __mips_msa
        TILE_N = std::max(8, tile_size / 8 * 8);
        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::max(8, std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8));
#else
        TILE_N = std::max(1, tile_size);
        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::max(1, std::min(TILE_N, (N + nn_N - 1) / nn_N));
#endif
    }
    else
    {
#if __mips_msa
        TILE_N = 8;
#else
        TILE_N = 1;
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
        TILE_K = (constant_TILE_K + 7) / 8 * 8;
#else
        TILE_K = (constant_TILE_K + 1) / 2 * 2;
#endif
    }
}

struct gemm_mips_int8_omp_args
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

static int gemm_mips_int8(const Mat& A, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int transA, int transB, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
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

    const struct gemm_mips_int8_omp_args args = {TILE_M, TILE_N, TILE_K, broadcast_type_C, transA, output_transpose, alpha, beta};

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

static int gemm_AT_mips_int8(const Mat& AT, const Mat& A_int8_scales, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int K, int transB, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
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

    const struct gemm_mips_int8_omp_args args = {TILE_M, TILE_N, TILE_K, broadcast_type_C, 0, output_transpose, alpha, beta};

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

static int gemm_BT_mips_int8(const Mat& A, const Mat& BT, float B_int8_scale, const Mat& C, Mat& top_blob, int broadcast_type_C, int N, int K, int transA, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
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

    const struct gemm_mips_int8_omp_args args = {TILE_M, TILE_N, TILE_K, broadcast_type_C, transA, output_transpose, alpha, beta};

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

static int gemm_AT_BT_mips_int8(const Mat& AT, const Mat& A_int8_scales, const Mat& BT, float B_int8_scale, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int N, int K, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
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

    const struct gemm_mips_int8_omp_args args = {TILE_M, TILE_N, TILE_K, broadcast_type_C, 0, output_transpose, alpha, beta};

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

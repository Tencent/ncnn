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
#if __loongarch_asx
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
#endif // __loongarch_asx
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
#endif // __loongarch_sx
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
#if __loongarch_sx
#if __loongarch_asx
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
#endif // __loongarch_asx
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
#endif // __loongarch_sx
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
#if __loongarch_sx
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
#endif // __loongarch_sx
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
#if __loongarch_sx
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
#endif // __loongarch_sx
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
#if __loongarch_asx
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
#endif // __loongarch_asx
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
#endif // __loongarch_sx
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
#if __loongarch_sx
#if __loongarch_asx
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
#endif // __loongarch_asx
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
#endif // __loongarch_sx
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
#endif // __loongarch_sx
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
#if __loongarch_sx
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
#endif // __loongarch_sx
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
#if __loongarch_sx
#if __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* pB = pBT;

        int jj = 0;
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
                _sum0 = __lasx_xvld(outptr, 0);
                _sum1 = __lasx_xvld(outptr + 8, 0);
                _sum2 = __lasx_xvld(outptr + 16, 0);
                _sum3 = __lasx_xvld(outptr + 24, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256i _pA = __lasx_xvldrepl_d(pA, 0);
                _pA = __lasx_xvilvl_b(__lasx_xvslti_b(_pA, 0), _pA);

                __m256i _pB0 = __lasx_xvreplgr2vr_h(pB[0]);
                __m256i _pB1 = __lasx_xvreplgr2vr_h(pB[1]);
                __m256i _pB2 = __lasx_xvreplgr2vr_h(pB[2]);
                __m256i _pB3 = __lasx_xvreplgr2vr_h(pB[3]);

                __m256i _s0 = __lasx_xvmul_h(_pA, _pB0);
                __m256i _s1 = __lasx_xvmul_h(_pA, _pB1);
                __m256i _s2 = __lasx_xvmul_h(_pA, _pB2);
                __m256i _s3 = __lasx_xvmul_h(_pA, _pB3);

                _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvsext_w_h(_s0));
                _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvsext_w_h(_s1));
                _sum2 = __lasx_xvadd_w(_sum2, __lasx_xvsext_w_h(_s2));
                _sum3 = __lasx_xvadd_w(_sum3, __lasx_xvsext_w_h(_s3));

                pA += 8;
                pB += 4;
            }

            __lasx_xvst(_sum0, outptr, 0);
            __lasx_xvst(_sum1, outptr + 8, 0);
            __lasx_xvst(_sum2, outptr + 16, 0);
            __lasx_xvst(_sum3, outptr + 24, 0);

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
                _sum0 = __lasx_xvld(outptr, 0);
                _sum1 = __lasx_xvld(outptr + 8, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256i _pA = __lasx_xvldrepl_d(pA, 0);
                _pA = __lasx_xvilvl_b(__lasx_xvslti_b(_pA, 0), _pA);

                __m256i _pB0 = __lasx_xvreplgr2vr_h(pB[0]);
                __m256i _pB1 = __lasx_xvreplgr2vr_h(pB[1]);

                __m256i _s0 = __lasx_xvmul_h(_pA, _pB0);
                __m256i _s1 = __lasx_xvmul_h(_pA, _pB1);

                _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvsext_w_h(_s0));
                _sum1 = __lasx_xvadd_w(_sum1, __lasx_xvsext_w_h(_s1));

                pA += 8;
                pB += 2;
            }

            __lasx_xvst(_sum0, outptr, 0);
            __lasx_xvst(_sum1, outptr + 8, 0);

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
            for (; kk < max_kk; kk += 1)
            {
                __m256i _pA = __lasx_xvldrepl_d(pA, 0);
                _pA = __lasx_xvilvl_b(__lasx_xvslti_b(_pA, 0), _pA);

                __m256i _pB0 = __lasx_xvreplgr2vr_h(pB[0]);

                __m256i _s0 = __lasx_xvmul_h(_pA, _pB0);

                _sum0 = __lasx_xvadd_w(_sum0, __lasx_xvsext_w_h(_s0));

                pA += 8;
                pB += 1;
            }

            __lasx_xvst(_sum0, outptr, 0);

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
#endif // __loongarch_asx
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* pB = pBT;

        int jj = 0;
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
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = __lsx_vldrepl_w(pA, 0);
                _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);

                __m128i _pB0 = __lsx_vreplgr2vr_h(pB[0]);
                __m128i _pB1 = __lsx_vreplgr2vr_h(pB[1]);
                __m128i _pB2 = __lsx_vreplgr2vr_h(pB[2]);
                __m128i _pB3 = __lsx_vreplgr2vr_h(pB[3]);

                __m128i _s0 = __lsx_vmul_h(_pA, _pB0);
                __m128i _s1 = __lsx_vmul_h(_pA, _pB1);
                __m128i _s2 = __lsx_vmul_h(_pA, _pB2);
                __m128i _s3 = __lsx_vmul_h(_pA, _pB3);

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
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = __lsx_vldrepl_w(pA, 0);
                _pA = __lsx_vilvl_b(__lsx_vslti_b(_pA, 0), _pA);

                __m128i _pB0 = __lsx_vreplgr2vr_h(pB[0]);
                __m128i _pB1 = __lsx_vreplgr2vr_h(pB[1]);

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
            for (; kk < max_kk; kk += 1)
            {
                // HACK auto-vectorization leads to wrong result
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
            for (; kk < max_kk; kk += 1)
            {
                // HACK auto-vectorization leads to wrong result
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
            for (; kk < max_kk; kk += 1)
            {
                // HACK auto-vectorization leads to wrong result
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


static NCNN_FORCEINLINE float gemm_int8_get_C(const Mat& C, int broadcast_type_C, int i, int j)
{
    const float* pC = C;
    if (!pC)
        return 0.f;

    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;

    if (broadcast_type_C == 0) return pC[0];
    if (broadcast_type_C == 1 || broadcast_type_C == 2) return pC[i];
    if (broadcast_type_C == 3) return pC[(size_t)i * c_hstep + j];
    if (broadcast_type_C == 4) return pC[j];

    return 0.f;
}

static NCNN_FORCEINLINE void gemm_int8_store_output(float* outptr, size_t out_hstep, int output_transpose, int i, int j, float v)
{
    if (output_transpose)
        outptr[(size_t)j * out_hstep + i] = v;
    else
        outptr[(size_t)i * out_hstep + j] = v;
}

static NCNN_FORCEINLINE void unpack_output_tile_int32_to_fp32_block(const int*& pp, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int rows, int j, int cols, const Mat& descales, float alpha, float beta, int output_transpose)
{
    float* outptr = top_blob;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    const float* descale_ptr = (const float*)descales + i;

    for (int jj = 0; jj < cols; jj++)
    {
        for (int ii = 0; ii < rows; ii++)
        {
            float v = pp[jj * rows + ii] * descale_ptr[ii];
            if (!C.empty())
                v += gemm_int8_get_C(C, broadcast_type_C, i + ii, j + jj) * beta;
            v *= alpha;
            gemm_int8_store_output(outptr, out_hstep, output_transpose, i + ii, j + jj, v);
        }
    }

    pp += rows * cols;
}

static void unpack_output_tile_int32_to_fp32(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta, int output_transpose)
{
    const int* pp = topT;

    int ii = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
            unpack_output_tile_int32_to_fp32_block(pp, C, top_blob, broadcast_type_C, i + ii, 8, j + jj, 4, descales, alpha, beta, output_transpose);
        for (; jj + 1 < max_jj; jj += 2)
            unpack_output_tile_int32_to_fp32_block(pp, C, top_blob, broadcast_type_C, i + ii, 8, j + jj, 2, descales, alpha, beta, output_transpose);
        for (; jj < max_jj; jj++)
            unpack_output_tile_int32_to_fp32_block(pp, C, top_blob, broadcast_type_C, i + ii, 8, j + jj, 1, descales, alpha, beta, output_transpose);
    }
#endif // __loongarch_asx
    for (; ii + 3 < max_ii; ii += 4)
    {
        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
            unpack_output_tile_int32_to_fp32_block(pp, C, top_blob, broadcast_type_C, i + ii, 4, j + jj, 4, descales, alpha, beta, output_transpose);
        for (; jj + 1 < max_jj; jj += 2)
            unpack_output_tile_int32_to_fp32_block(pp, C, top_blob, broadcast_type_C, i + ii, 4, j + jj, 2, descales, alpha, beta, output_transpose);
        for (; jj < max_jj; jj++)
            unpack_output_tile_int32_to_fp32_block(pp, C, top_blob, broadcast_type_C, i + ii, 4, j + jj, 1, descales, alpha, beta, output_transpose);
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        int jj = 0;
#if __loongarch_sx
        for (; jj + 3 < max_jj; jj += 4)
            unpack_output_tile_int32_to_fp32_block(pp, C, top_blob, broadcast_type_C, i + ii, 2, j + jj, 4, descales, alpha, beta, output_transpose);
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
            unpack_output_tile_int32_to_fp32_block(pp, C, top_blob, broadcast_type_C, i + ii, 2, j + jj, 2, descales, alpha, beta, output_transpose);
        for (; jj < max_jj; jj++)
            unpack_output_tile_int32_to_fp32_block(pp, C, top_blob, broadcast_type_C, i + ii, 2, j + jj, 1, descales, alpha, beta, output_transpose);
    }
    for (; ii < max_ii; ii++)
    {
        int jj = 0;
#if __loongarch_sx
        for (; jj + 3 < max_jj; jj += 4)
            unpack_output_tile_int32_to_fp32_block(pp, C, top_blob, broadcast_type_C, i + ii, 1, j + jj, 4, descales, alpha, beta, output_transpose);
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
            unpack_output_tile_int32_to_fp32_block(pp, C, top_blob, broadcast_type_C, i + ii, 1, j + jj, 2, descales, alpha, beta, output_transpose);
        for (; jj < max_jj; jj++)
            unpack_output_tile_int32_to_fp32_block(pp, C, top_blob, broadcast_type_C, i + ii, 1, j + jj, 1, descales, alpha, beta, output_transpose);
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
#if __loongarch_asx
    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_N = std::max(8, tile_size / 8 * 8);
    TILE_K = std::max(8, tile_size / 8 * 8);
#else
    TILE_M = std::max(4, tile_size / 4 * 4);
    TILE_N = std::max(4, tile_size / 4 * 4);
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
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(signed char) / TILE_K);

#if __loongarch_sx
#if __loongarch_asx
            TILE_M = std::max(8, tile_size / 8 * 8);
            TILE_N = std::max(8, tile_size / 8 * 8);
#else
            TILE_M = std::max(4, tile_size / 4 * 4);
            TILE_N = std::max(4, tile_size / 4 * 4);
#endif
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
#endif
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
#endif
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
#endif
#else
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
    }

    // always take constant TILE_M/N/K value when provided
    if (constant_TILE_M > 0)
    {
#if __loongarch_sx
#if __loongarch_asx
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
#else
        TILE_M = (constant_TILE_M + 3) / 4 * 4;
#endif
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
#endif
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

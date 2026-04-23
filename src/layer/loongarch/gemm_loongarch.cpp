// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gemm_loongarch.h"

#if __loongarch_sx
#include "loongarch_usability.h"
#endif // __loongarch_sx

#include "cpu.h"

namespace ncnn {

#if NCNN_BF16
#include "gemm_bf16s.h"
#endif

Gemm_loongarch::Gemm_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx

#if NCNN_BF16
    support_bf16_storage = true;
#endif

    nT = 0;
}

static int resolve_broadcast_type_C(const Mat& C, int M, int N)
{
    int broadcast_type_C = 0;

    if (C.dims == 1 && C.w == 1)
        broadcast_type_C = 0;
    if (C.dims == 1 && C.w * C.elempack == M)
        broadcast_type_C = 1;
    if (C.dims == 1 && C.w * C.elempack == N)
        broadcast_type_C = 4;
    if (C.dims == 2 && C.w == 1 && C.h * C.elempack == M)
        broadcast_type_C = 2;
    if (C.dims == 2 && C.w == N && C.h * C.elempack == M)
        broadcast_type_C = 3;
    if (C.dims == 2 && C.w == N && C.h * C.elempack == 1)
        broadcast_type_C = 4;

    return broadcast_type_C;
}

static void add_matrix_C_fp32(const Mat& C, Mat& top_blob, int output_transpose)
{
    const int M = C.h;
    const int N = C.w;
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    if (output_transpose)
    {
        if (top_blob.dims == 3)
        {
            for (int j = 0; j < N; j++)
            {
                float* outptr = top_blob.channel(j / out_elempack);
                const int lane = j % out_elempack;
                for (int i = 0; i < M; i++)
                {
                    outptr[i * out_elempack + lane] += C.row(i)[j];
                }
            }
        }
        else
        {
            for (int j = 0; j < N; j++)
            {
                float* outptr = (float*)top_blob + (j / out_elempack) * out_hstep * out_elempack;
                const int lane = j % out_elempack;
                for (int i = 0; i < M; i++)
                {
                    outptr[i * out_elempack + lane] += C.row(i)[j];
                }
            }
        }
    }
    else
    {
        if (top_blob.dims == 3)
        {
            for (int i = 0; i < M; i++)
            {
                float* outptr = top_blob.channel(i / out_elempack);
                const int lane = i % out_elempack;
                const float* cptr = C.row(i);
                for (int j = 0; j < N; j++)
                {
                    outptr[j * out_elempack + lane] += cptr[j];
                }
            }
        }
        else
        {
            for (int i = 0; i < M; i++)
            {
                float* outptr = (float*)top_blob + (i / out_elempack) * out_hstep * out_elempack;
                const int lane = i % out_elempack;
                const float* cptr = C.row(i);
                for (int j = 0; j < N; j++)
                {
                    outptr[j * out_elempack + lane] += cptr[j];
                }
            }
        }
    }
}

#if __loongarch_sx
static NCNN_FORCEINLINE void transpose4x4_ps(__m128& _r0, __m128& _r1, __m128& _r2, __m128& _r3)
{
    __m128i _r01r = __lsx_vilvl_w((__m128i)_r1, (__m128i)_r0);
    __m128i _r01l = __lsx_vilvh_w((__m128i)_r1, (__m128i)_r0);
    __m128i _r23r = __lsx_vilvl_w((__m128i)_r3, (__m128i)_r2);
    __m128i _r23l = __lsx_vilvh_w((__m128i)_r3, (__m128i)_r2);
    _r0 = (__m128)__lsx_vilvl_d((__m128i)_r23r, (__m128i)_r01r);
    _r1 = (__m128)__lsx_vilvh_d((__m128i)_r23r, (__m128i)_r01r);
    _r2 = (__m128)__lsx_vilvl_d((__m128i)_r23l, (__m128i)_r01l);
    _r3 = (__m128)__lsx_vilvh_d((__m128i)_r23l, (__m128i)_r01l);
}
#endif // __loongarch_sx

static NCNN_FORCEINLINE float get_packed_matrix_element(const Mat& m, int row, int col)
{
    const int elempack = m.elempack;
    const size_t hstep = m.dims == 3 ? m.cstep : (size_t)m.w;

    return ((const float*)m)[(size_t)(row / elempack) * hstep * elempack + (size_t)col * elempack + row % elempack];
}

#if __loongarch_sx
static NCNN_FORCEINLINE bool use_8row_packed_kernel(int out_elempack)
{
#if __loongarch_asx
    (void)out_elempack;
    return false;
#else
    (void)out_elempack;
    return true;
#endif
}
#endif // __loongarch_sx

static void pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    float* pp = AT;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 3 < max_ii; ii += 4)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = get_packed_matrix_element(A, i + ii, k + kk);
                pp[1] = get_packed_matrix_element(A, i + ii + 1, k + kk);
                pp[2] = get_packed_matrix_element(A, i + ii + 2, k + kk);
                pp[3] = get_packed_matrix_element(A, i + ii + 3, k + kk);
                pp += 4;
            }
        }
        else
#endif
            if (elempack == 4)
            {
                const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;

                for (int kk = 0; kk < max_kk; kk++)
                {
                    __lsx_vst(__lsx_vld(p0, 0), pp, 0);
                    pp += 4;
                    p0 += 4;
                }
            }
            else
            {
                const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
                const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
                const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
                const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(p1, 0);
                    __m128 _r2 = (__m128)__lsx_vld(p2, 0);
                    __m128 _r3 = (__m128)__lsx_vld(p3, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r1, pp + 4, 0);
                    __lsx_vst((__m128i)_r2, pp + 8, 0);
                    __lsx_vst((__m128i)_r3, pp + 12, 0);
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
    }
#endif // __loongarch_sx

    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }

    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        int kk = 0;
#if __loongarch_sx
        for (; kk + 3 < max_kk; kk += 4)
        {
            __lsx_vst(__lsx_vld(p0, 0), pp, 0);
            pp += 4;
            p0 += 4;
        }
#endif // __loongarch_sx
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    float* pp = AT;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 3 < max_ii; ii += 4)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                for (int q = 0; q < 8; q++)
                {
                    pp[q * 4] = get_packed_matrix_element(A, k + kk + q, i + ii);
                    pp[q * 4 + 1] = get_packed_matrix_element(A, k + kk + q, i + ii + 1);
                    pp[q * 4 + 2] = get_packed_matrix_element(A, k + kk + q, i + ii + 2);
                    pp[q * 4 + 3] = get_packed_matrix_element(A, k + kk + q, i + ii + 3);
                }
                pp += 32;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = get_packed_matrix_element(A, k + kk, i + ii);
                pp[1] = get_packed_matrix_element(A, k + kk, i + ii + 1);
                pp[2] = get_packed_matrix_element(A, k + kk, i + ii + 2);
                pp[3] = get_packed_matrix_element(A, k + kk, i + ii + 3);
                pp += 4;
            }
        }
        else
#endif
            if (elempack == 4)
            {
                const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                    __m128 _r2 = (__m128)__lsx_vld(p0 + 8, 0);
                    __m128 _r3 = (__m128)__lsx_vld(p0 + 12, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r1, pp + 4, 0);
                    __lsx_vst((__m128i)_r2, pp + 8, 0);
                    __lsx_vst((__m128i)_r3, pp + 12, 0);
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
            else
            {
                const float* p0 = (const float*)A + k * A_hstep + (i + ii);

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
#if __loongarch_sx
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                float tmp0[4];
                float tmp1[4];
                __lsx_vst((__m128i)_r0, tmp0, 0);
                __lsx_vst((__m128i)_r1, tmp1, 0);
                pp[0] = tmp0[0];
                pp[1] = tmp1[0];
                pp[2] = tmp0[1];
                pp[3] = tmp1[1];
                pp[4] = tmp0[2];
                pp[5] = tmp1[2];
                pp[6] = tmp0[3];
                pp[7] = tmp1[3];
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
        else
#endif // __loongarch_sx
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            for (int kk = 0; kk < max_kk; kk++)
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
#if __loongarch_sx
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __lsx_vst(__lsx_vld(p0, 0), pp, 0);
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
        else
#endif // __loongarch_sx
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += A_hstep;
            }
        }
    }
}

#if __loongarch_sx
static void pack_A_tile_8row(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    float* pp = AT;

    int ii = 0;
    for (; ii + 7 < max_ii; ii += 8)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = get_packed_matrix_element(A, i + ii, k + kk);
                pp[1] = get_packed_matrix_element(A, i + ii + 1, k + kk);
                pp[2] = get_packed_matrix_element(A, i + ii + 2, k + kk);
                pp[3] = get_packed_matrix_element(A, i + ii + 3, k + kk);
                pp[4] = get_packed_matrix_element(A, i + ii + 4, k + kk);
                pp[5] = get_packed_matrix_element(A, i + ii + 5, k + kk);
                pp[6] = get_packed_matrix_element(A, i + ii + 6, k + kk);
                pp[7] = get_packed_matrix_element(A, i + ii + 7, k + kk);
                pp += 8;
            }
        }
        else
#endif
            if (elempack == 4)
            {
                const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;
                const float* p1 = (const float*)A + (i + ii + 4) * A_hstep + k * 4;

                for (int kk = 0; kk < max_kk; kk++)
                {
                    __lsx_vst(__lsx_vld(p0, 0), pp, 0);
                    __lsx_vst(__lsx_vld(p1, 0), pp + 4, 0);
                    pp += 8;
                    p0 += 4;
                    p1 += 4;
                }
            }
            else
            {
                const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
                const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
                const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
                const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;
                const float* p4 = (const float*)A + (i + ii + 4) * A_hstep + k;
                const float* p5 = (const float*)A + (i + ii + 5) * A_hstep + k;
                const float* p6 = (const float*)A + (i + ii + 6) * A_hstep + k;
                const float* p7 = (const float*)A + (i + ii + 7) * A_hstep + k;

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(p1, 0);
                    __m128 _r2 = (__m128)__lsx_vld(p2, 0);
                    __m128 _r3 = (__m128)__lsx_vld(p3, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __m128 _r4 = (__m128)__lsx_vld(p4, 0);
                    __m128 _r5 = (__m128)__lsx_vld(p5, 0);
                    __m128 _r6 = (__m128)__lsx_vld(p6, 0);
                    __m128 _r7 = (__m128)__lsx_vld(p7, 0);
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r4, pp + 4, 0);
                    __lsx_vst((__m128i)_r1, pp + 8, 0);
                    __lsx_vst((__m128i)_r5, pp + 12, 0);
                    __lsx_vst((__m128i)_r2, pp + 16, 0);
                    __lsx_vst((__m128i)_r6, pp + 20, 0);
                    __lsx_vst((__m128i)_r3, pp + 24, 0);
                    __lsx_vst((__m128i)_r7, pp + 28, 0);
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
    }

    for (; ii + 3 < max_ii; ii += 4)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = get_packed_matrix_element(A, i + ii, k + kk);
                pp[1] = get_packed_matrix_element(A, i + ii + 1, k + kk);
                pp[2] = get_packed_matrix_element(A, i + ii + 2, k + kk);
                pp[3] = get_packed_matrix_element(A, i + ii + 3, k + kk);
                pp += 4;
            }
        }
        else
#endif
            if (elempack == 4)
            {
                const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;

                for (int kk = 0; kk < max_kk; kk++)
                {
                    __lsx_vst(__lsx_vld(p0, 0), pp, 0);
                    pp += 4;
                    p0 += 4;
                }
            }
            else
            {
                const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
                const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
                const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
                const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(p1, 0);
                    __m128 _r2 = (__m128)__lsx_vld(p2, 0);
                    __m128 _r3 = (__m128)__lsx_vld(p3, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r1, pp + 4, 0);
                    __lsx_vst((__m128i)_r2, pp + 8, 0);
                    __lsx_vst((__m128i)_r3, pp + 12, 0);
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
    }

    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }

    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __lsx_vst(__lsx_vld(p0, 0), pp, 0);
            pp += 4;
            p0 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_A_tile_8row(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    float* pp = AT;

    int ii = 0;
    for (; ii + 7 < max_ii; ii += 8)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = get_packed_matrix_element(A, k + kk, i + ii);
                pp[1] = get_packed_matrix_element(A, k + kk, i + ii + 1);
                pp[2] = get_packed_matrix_element(A, k + kk, i + ii + 2);
                pp[3] = get_packed_matrix_element(A, k + kk, i + ii + 3);
                pp[4] = get_packed_matrix_element(A, k + kk, i + ii + 4);
                pp[5] = get_packed_matrix_element(A, k + kk, i + ii + 5);
                pp[6] = get_packed_matrix_element(A, k + kk, i + ii + 6);
                pp[7] = get_packed_matrix_element(A, k + kk, i + ii + 7);
                pp += 8;
            }
        }
        else
#endif
            if (elempack == 4)
            {
                const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                    __m128 _r2 = (__m128)__lsx_vld(p0 + 8, 0);
                    __m128 _r3 = (__m128)__lsx_vld(p0 + 12, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __m128 _r4 = (__m128)__lsx_vld(p0 + 16, 0);
                    __m128 _r5 = (__m128)__lsx_vld(p0 + 20, 0);
                    __m128 _r6 = (__m128)__lsx_vld(p0 + 24, 0);
                    __m128 _r7 = (__m128)__lsx_vld(p0 + 28, 0);
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r4, pp + 4, 0);
                    __lsx_vst((__m128i)_r1, pp + 8, 0);
                    __lsx_vst((__m128i)_r5, pp + 12, 0);
                    __lsx_vst((__m128i)_r2, pp + 16, 0);
                    __lsx_vst((__m128i)_r6, pp + 20, 0);
                    __lsx_vst((__m128i)_r3, pp + 24, 0);
                    __lsx_vst((__m128i)_r7, pp + 28, 0);
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
            else
            {
                const float* p0 = (const float*)A + k * A_hstep + (i + ii);

                for (int kk = 0; kk < max_kk; kk++)
                {
                    __lsx_vst(__lsx_vld(p0, 0), pp, 0);
                    __lsx_vst(__lsx_vld(p0 + 4, 0), pp + 4, 0);
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
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                for (int q = 0; q < 8; q++)
                {
                    pp[q * 4] = get_packed_matrix_element(A, k + kk + q, i + ii);
                    pp[q * 4 + 1] = get_packed_matrix_element(A, k + kk + q, i + ii + 1);
                    pp[q * 4 + 2] = get_packed_matrix_element(A, k + kk + q, i + ii + 2);
                    pp[q * 4 + 3] = get_packed_matrix_element(A, k + kk + q, i + ii + 3);
                }
                pp += 32;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = get_packed_matrix_element(A, k + kk, i + ii);
                pp[1] = get_packed_matrix_element(A, k + kk, i + ii + 1);
                pp[2] = get_packed_matrix_element(A, k + kk, i + ii + 2);
                pp[3] = get_packed_matrix_element(A, k + kk, i + ii + 3);
                pp += 4;
            }
        }
        else
#endif
            if (elempack == 4)
            {
                const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                    __m128 _r2 = (__m128)__lsx_vld(p0 + 8, 0);
                    __m128 _r3 = (__m128)__lsx_vld(p0 + 12, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r1, pp + 4, 0);
                    __lsx_vst((__m128i)_r2, pp + 8, 0);
                    __lsx_vst((__m128i)_r3, pp + 12, 0);
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
            else
            {
                const float* p0 = (const float*)A + k * A_hstep + (i + ii);

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

    for (; ii + 1 < max_ii; ii += 2)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                float tmp0[4];
                float tmp1[4];
                __lsx_vst((__m128i)_r0, tmp0, 0);
                __lsx_vst((__m128i)_r1, tmp1, 0);
                pp[0] = tmp0[0];
                pp[1] = tmp1[0];
                pp[2] = tmp0[1];
                pp[3] = tmp1[1];
                pp[4] = tmp0[2];
                pp[5] = tmp1[2];
                pp[6] = tmp0[3];
                pp[7] = tmp1[3];
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
        else
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            for (int kk = 0; kk < max_kk; kk++)
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
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __lsx_vst(__lsx_vld(p0, 0), pp, 0);
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
        else
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += A_hstep;
            }
        }
    }
}
#endif // __loongarch_sx

static void pack_B_tile(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    float* pp = BT;

    int jj = 0;
#if __loongarch_asx
    for (; jj + 15 < max_jj; jj += 16)
    {
        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = get_packed_matrix_element(B, j + jj, k + kk);
            pp[1] = get_packed_matrix_element(B, j + jj + 1, k + kk);
            pp[2] = get_packed_matrix_element(B, j + jj + 2, k + kk);
            pp[3] = get_packed_matrix_element(B, j + jj + 3, k + kk);
            pp[4] = get_packed_matrix_element(B, j + jj + 4, k + kk);
            pp[5] = get_packed_matrix_element(B, j + jj + 5, k + kk);
            pp[6] = get_packed_matrix_element(B, j + jj + 6, k + kk);
            pp[7] = get_packed_matrix_element(B, j + jj + 7, k + kk);
            pp[8] = get_packed_matrix_element(B, j + jj + 8, k + kk);
            pp[9] = get_packed_matrix_element(B, j + jj + 9, k + kk);
            pp[10] = get_packed_matrix_element(B, j + jj + 10, k + kk);
            pp[11] = get_packed_matrix_element(B, j + jj + 11, k + kk);
            pp[12] = get_packed_matrix_element(B, j + jj + 12, k + kk);
            pp[13] = get_packed_matrix_element(B, j + jj + 13, k + kk);
            pp[14] = get_packed_matrix_element(B, j + jj + 14, k + kk);
            pp[15] = get_packed_matrix_element(B, j + jj + 15, k + kk);
            pp += 16;
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = get_packed_matrix_element(B, j + jj, k + kk);
            pp[1] = get_packed_matrix_element(B, j + jj + 1, k + kk);
            pp[2] = get_packed_matrix_element(B, j + jj + 2, k + kk);
            pp[3] = get_packed_matrix_element(B, j + jj + 3, k + kk);
            pp[4] = get_packed_matrix_element(B, j + jj + 4, k + kk);
            pp[5] = get_packed_matrix_element(B, j + jj + 5, k + kk);
            pp[6] = get_packed_matrix_element(B, j + jj + 6, k + kk);
            pp[7] = get_packed_matrix_element(B, j + jj + 7, k + kk);
            pp += 8;
        }
    }
#endif
#if __loongarch_sx
    for (; jj + 7 < max_jj; jj += 8)
    {
        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = get_packed_matrix_element(B, j + jj, k + kk);
            pp[1] = get_packed_matrix_element(B, j + jj + 1, k + kk);
            pp[2] = get_packed_matrix_element(B, j + jj + 2, k + kk);
            pp[3] = get_packed_matrix_element(B, j + jj + 3, k + kk);
            pp[4] = get_packed_matrix_element(B, j + jj + 4, k + kk);
            pp[5] = get_packed_matrix_element(B, j + jj + 5, k + kk);
            pp[6] = get_packed_matrix_element(B, j + jj + 6, k + kk);
            pp[7] = get_packed_matrix_element(B, j + jj + 7, k + kk);
            pp += 8;
        }
    }
#endif // __loongarch_sx
    for (; jj + 3 < max_jj; jj += 4)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = get_packed_matrix_element(B, j + jj, k + kk);
                pp[1] = get_packed_matrix_element(B, j + jj + 1, k + kk);
                pp[2] = get_packed_matrix_element(B, j + jj + 2, k + kk);
                pp[3] = get_packed_matrix_element(B, j + jj + 3, k + kk);
                pp += 4;
            }
        }
        else
#endif
#if __loongarch_sx
            if (elempack == 4)
            {
                const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;

                for (int kk = 0; kk < max_kk; kk++)
                {
                    __lsx_vst(__lsx_vld(p0, 0), pp, 0);
                    pp += 4;
                    p0 += 4;
                }
            }
            else
#endif // __loongarch_sx
            {
                const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
                const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
                const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
                const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;

                int kk = 0;
#if __loongarch_sx
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(p1, 0);
                    __m128 _r2 = (__m128)__lsx_vld(p2, 0);
                    __m128 _r3 = (__m128)__lsx_vld(p3, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r1, pp + 4, 0);
                    __lsx_vst((__m128i)_r2, pp + 8, 0);
                    __lsx_vst((__m128i)_r3, pp + 12, 0);
                    pp += 16;
                    p0 += 4;
                    p1 += 4;
                    p2 += 4;
                    p3 += 4;
                }
#endif // __loongarch_sx
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
    }

    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }

    for (; jj < max_jj; jj += 1)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

        int kk = 0;
#if __loongarch_sx
        for (; kk + 3 < max_kk; kk += 4)
        {
            __lsx_vst(__lsx_vld(p0, 0), pp, 0);
            pp += 4;
            p0 += 4;
        }
#endif // __loongarch_sx
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_B_tile(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    float* pp = BT;

    int jj = 0;
#if __loongarch_asx
    for (; jj + 15 < max_jj; jj += 16)
    {
        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = get_packed_matrix_element(B, k + kk, j + jj);
            pp[1] = get_packed_matrix_element(B, k + kk, j + jj + 1);
            pp[2] = get_packed_matrix_element(B, k + kk, j + jj + 2);
            pp[3] = get_packed_matrix_element(B, k + kk, j + jj + 3);
            pp[4] = get_packed_matrix_element(B, k + kk, j + jj + 4);
            pp[5] = get_packed_matrix_element(B, k + kk, j + jj + 5);
            pp[6] = get_packed_matrix_element(B, k + kk, j + jj + 6);
            pp[7] = get_packed_matrix_element(B, k + kk, j + jj + 7);
            pp[8] = get_packed_matrix_element(B, k + kk, j + jj + 8);
            pp[9] = get_packed_matrix_element(B, k + kk, j + jj + 9);
            pp[10] = get_packed_matrix_element(B, k + kk, j + jj + 10);
            pp[11] = get_packed_matrix_element(B, k + kk, j + jj + 11);
            pp[12] = get_packed_matrix_element(B, k + kk, j + jj + 12);
            pp[13] = get_packed_matrix_element(B, k + kk, j + jj + 13);
            pp[14] = get_packed_matrix_element(B, k + kk, j + jj + 14);
            pp[15] = get_packed_matrix_element(B, k + kk, j + jj + 15);
            pp += 16;
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = get_packed_matrix_element(B, k + kk, j + jj);
            pp[1] = get_packed_matrix_element(B, k + kk, j + jj + 1);
            pp[2] = get_packed_matrix_element(B, k + kk, j + jj + 2);
            pp[3] = get_packed_matrix_element(B, k + kk, j + jj + 3);
            pp[4] = get_packed_matrix_element(B, k + kk, j + jj + 4);
            pp[5] = get_packed_matrix_element(B, k + kk, j + jj + 5);
            pp[6] = get_packed_matrix_element(B, k + kk, j + jj + 6);
            pp[7] = get_packed_matrix_element(B, k + kk, j + jj + 7);
            pp += 8;
        }
    }
#endif
#if __loongarch_sx
    for (; jj + 7 < max_jj; jj += 8)
    {
        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = get_packed_matrix_element(B, k + kk, j + jj);
            pp[1] = get_packed_matrix_element(B, k + kk, j + jj + 1);
            pp[2] = get_packed_matrix_element(B, k + kk, j + jj + 2);
            pp[3] = get_packed_matrix_element(B, k + kk, j + jj + 3);
            pp[4] = get_packed_matrix_element(B, k + kk, j + jj + 4);
            pp[5] = get_packed_matrix_element(B, k + kk, j + jj + 5);
            pp[6] = get_packed_matrix_element(B, k + kk, j + jj + 6);
            pp[7] = get_packed_matrix_element(B, k + kk, j + jj + 7);
            pp += 8;
        }
    }
#endif // __loongarch_sx
    for (; jj + 3 < max_jj; jj += 4)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = get_packed_matrix_element(B, k + kk, j + jj);
                pp[1] = get_packed_matrix_element(B, k + kk, j + jj + 1);
                pp[2] = get_packed_matrix_element(B, k + kk, j + jj + 2);
                pp[3] = get_packed_matrix_element(B, k + kk, j + jj + 3);
                pp += 4;
            }
        }
        else
#endif
#if __loongarch_sx
            if (elempack == 4)
            {
                const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                    __m128 _r2 = (__m128)__lsx_vld(p0 + 8, 0);
                    __m128 _r3 = (__m128)__lsx_vld(p0 + 12, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r1, pp + 4, 0);
                    __lsx_vst((__m128i)_r2, pp + 8, 0);
                    __lsx_vst((__m128i)_r3, pp + 12, 0);
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
            else
#endif // __loongarch_sx
            {
                const float* p0 = (const float*)B + k * B_hstep + (j + jj);

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
#if __loongarch_sx
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                float tmp0[4];
                float tmp1[4];
                __lsx_vst((__m128i)_r0, tmp0, 0);
                __lsx_vst((__m128i)_r1, tmp1, 0);
                pp[0] = tmp0[0];
                pp[1] = tmp1[0];
                pp[2] = tmp0[1];
                pp[3] = tmp1[1];
                pp[4] = tmp0[2];
                pp[5] = tmp1[2];
                pp[6] = tmp0[3];
                pp[7] = tmp1[3];
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
        else
#endif // __loongarch_sx
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            for (int kk = 0; kk < max_kk; kk++)
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
#if __loongarch_sx
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __lsx_vst(__lsx_vld(p0, 0), pp, 0);
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
        else
#endif // __loongarch_sx
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += B_hstep;
            }
        }
    }
}

static void transpose_unpack_output_tile(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const float* pp = topT;

    int ii = 0;
#if __loongarch_sx
    if (use_8row_packed_kernel(out_elempack))
    {
        for (; ii + 7 < max_ii; ii += 8)
        {
            if (out_elempack == 8)
            {
                int jj = 0;
                for (; jj + 7 < max_jj; jj += 8)
                {
                    for (int kk = 0; kk < 2; kk++)
                    {
                        const int col = j + jj + kk * 4;
                        const int lane = col & 7;
                        float* p0 = (float*)top_blob + (size_t)(col / 8 * 8) * out_hstep + (i + ii) * 8;
                        const float* ppk = pp + kk * 32;

                        __m128 _r0 = (__m128)__lsx_vld(ppk, 0);
                        __m128 _r4 = (__m128)__lsx_vld(ppk + 4, 0);
                        __m128 _r1 = (__m128)__lsx_vld(ppk + 8, 0);
                        __m128 _r5 = (__m128)__lsx_vld(ppk + 12, 0);
                        __m128 _r2 = (__m128)__lsx_vld(ppk + 16, 0);
                        __m128 _r6 = (__m128)__lsx_vld(ppk + 20, 0);
                        __m128 _r3 = (__m128)__lsx_vld(ppk + 24, 0);
                        __m128 _r7 = (__m128)__lsx_vld(ppk + 28, 0);
                        transpose4x4_ps(_r0, _r1, _r2, _r3);
                        transpose4x4_ps(_r4, _r5, _r6, _r7);
                        __lsx_vst((__m128i)_r0, p0 + lane, 0);
                        __lsx_vst((__m128i)_r1, p0 + 8 + lane, 0);
                        __lsx_vst((__m128i)_r2, p0 + 16 + lane, 0);
                        __lsx_vst((__m128i)_r3, p0 + 24 + lane, 0);
                        __lsx_vst((__m128i)_r4, p0 + 32 + lane, 0);
                        __lsx_vst((__m128i)_r5, p0 + 40 + lane, 0);
                        __lsx_vst((__m128i)_r6, p0 + 48 + lane, 0);
                        __lsx_vst((__m128i)_r7, p0 + 56 + lane, 0);
                    }
                    pp += 64;
                }
                for (; jj + 3 < max_jj; jj += 4)
                {
                    const int col = j + jj;
                    const int lane = col & 7;
                    float* p0 = (float*)top_blob + (size_t)(col / 8 * 8) * out_hstep + (i + ii) * 8;

                    __m128 _r0 = (__m128)__lsx_vld(pp, 0);
                    __m128 _r4 = (__m128)__lsx_vld(pp + 4, 0);
                    __m128 _r1 = (__m128)__lsx_vld(pp + 8, 0);
                    __m128 _r5 = (__m128)__lsx_vld(pp + 12, 0);
                    __m128 _r2 = (__m128)__lsx_vld(pp + 16, 0);
                    __m128 _r6 = (__m128)__lsx_vld(pp + 20, 0);
                    __m128 _r3 = (__m128)__lsx_vld(pp + 24, 0);
                    __m128 _r7 = (__m128)__lsx_vld(pp + 28, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    __lsx_vst((__m128i)_r0, p0 + lane, 0);
                    __lsx_vst((__m128i)_r1, p0 + 8 + lane, 0);
                    __lsx_vst((__m128i)_r2, p0 + 16 + lane, 0);
                    __lsx_vst((__m128i)_r3, p0 + 24 + lane, 0);
                    __lsx_vst((__m128i)_r4, p0 + 32 + lane, 0);
                    __lsx_vst((__m128i)_r5, p0 + 40 + lane, 0);
                    __lsx_vst((__m128i)_r6, p0 + 48 + lane, 0);
                    __lsx_vst((__m128i)_r7, p0 + 56 + lane, 0);
                    pp += 32;
                }
                for (; jj + 1 < max_jj; jj += 2)
                {
                    for (int c = 0; c < 2; c++)
                    {
                        const int col = j + jj + c;
                        const int lane = col % 8;
                        float* outptr = (float*)top_blob + (size_t)(col / 8 * 8) * out_hstep + (i + ii) * 8;
                        outptr[lane] = pp[c * 8];
                        outptr[8 + lane] = pp[c * 8 + 1];
                        outptr[16 + lane] = pp[c * 8 + 2];
                        outptr[24 + lane] = pp[c * 8 + 3];
                        outptr[32 + lane] = pp[c * 8 + 4];
                        outptr[40 + lane] = pp[c * 8 + 5];
                        outptr[48 + lane] = pp[c * 8 + 6];
                        outptr[56 + lane] = pp[c * 8 + 7];
                    }
                    pp += 16;
                }
                for (; jj < max_jj; jj++)
                {
                    const int lane = (j + jj) % 8;
                    float* outptr = (float*)top_blob + (size_t)((j + jj) / 8 * 8) * out_hstep + (i + ii) * 8;
                    outptr[lane] = pp[0];
                    outptr[8 + lane] = pp[1];
                    outptr[16 + lane] = pp[2];
                    outptr[24 + lane] = pp[3];
                    outptr[32 + lane] = pp[4];
                    outptr[40 + lane] = pp[5];
                    outptr[48 + lane] = pp[6];
                    outptr[56 + lane] = pp[7];
                    pp += 8;
                }
            }
            else if (out_elempack == 4)
            {
                int jj = 0;
                for (; jj + 7 < max_jj; jj += 8)
                {
                    for (int kk = 0; kk < 2; kk++)
                    {
                        float* p0 = (float*)top_blob + (j + jj + kk * 4) * out_hstep + (i + ii) * 4;
                        const float* ppk = pp + kk * 32;

                        __m128 _r0 = (__m128)__lsx_vld(ppk, 0);
                        __m128 _r4 = (__m128)__lsx_vld(ppk + 4, 0);
                        __m128 _r1 = (__m128)__lsx_vld(ppk + 8, 0);
                        __m128 _r5 = (__m128)__lsx_vld(ppk + 12, 0);
                        __m128 _r2 = (__m128)__lsx_vld(ppk + 16, 0);
                        __m128 _r6 = (__m128)__lsx_vld(ppk + 20, 0);
                        __m128 _r3 = (__m128)__lsx_vld(ppk + 24, 0);
                        __m128 _r7 = (__m128)__lsx_vld(ppk + 28, 0);
                        transpose4x4_ps(_r0, _r1, _r2, _r3);
                        transpose4x4_ps(_r4, _r5, _r6, _r7);
                        __lsx_vst((__m128i)_r0, p0, 0);
                        __lsx_vst((__m128i)_r1, p0 + 4, 0);
                        __lsx_vst((__m128i)_r2, p0 + 8, 0);
                        __lsx_vst((__m128i)_r3, p0 + 12, 0);
                        __lsx_vst((__m128i)_r4, p0 + 16, 0);
                        __lsx_vst((__m128i)_r5, p0 + 20, 0);
                        __lsx_vst((__m128i)_r6, p0 + 24, 0);
                        __lsx_vst((__m128i)_r7, p0 + 28, 0);
                    }
                    pp += 64;
                }
                for (; jj + 3 < max_jj; jj += 4)
                {
                    float* p0 = (float*)top_blob + (j + jj) * out_hstep + (i + ii) * 4;

                    __m128 _r0 = (__m128)__lsx_vld(pp, 0);
                    __m128 _r4 = (__m128)__lsx_vld(pp + 4, 0);
                    __m128 _r1 = (__m128)__lsx_vld(pp + 8, 0);
                    __m128 _r5 = (__m128)__lsx_vld(pp + 12, 0);
                    __m128 _r2 = (__m128)__lsx_vld(pp + 16, 0);
                    __m128 _r6 = (__m128)__lsx_vld(pp + 20, 0);
                    __m128 _r3 = (__m128)__lsx_vld(pp + 24, 0);
                    __m128 _r7 = (__m128)__lsx_vld(pp + 28, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    __lsx_vst((__m128i)_r0, p0, 0);
                    __lsx_vst((__m128i)_r1, p0 + 4, 0);
                    __lsx_vst((__m128i)_r2, p0 + 8, 0);
                    __lsx_vst((__m128i)_r3, p0 + 12, 0);
                    __lsx_vst((__m128i)_r4, p0 + 16, 0);
                    __lsx_vst((__m128i)_r5, p0 + 20, 0);
                    __lsx_vst((__m128i)_r6, p0 + 24, 0);
                    __lsx_vst((__m128i)_r7, p0 + 28, 0);
                    pp += 32;
                }
                for (; jj + 1 < max_jj; jj += 2)
                {
                    for (int c = 0; c < 2; c++)
                    {
                        const int col = j + jj + c;
                        const int lane = col % 4;
                        float* outptr = (float*)top_blob + (size_t)(col / 4 * 4) * out_hstep + (i + ii) * 4;
                        outptr[lane] = pp[c * 8];
                        outptr[4 + lane] = pp[c * 8 + 1];
                        outptr[8 + lane] = pp[c * 8 + 2];
                        outptr[12 + lane] = pp[c * 8 + 3];
                        outptr[16 + lane] = pp[c * 8 + 4];
                        outptr[20 + lane] = pp[c * 8 + 5];
                        outptr[24 + lane] = pp[c * 8 + 6];
                        outptr[28 + lane] = pp[c * 8 + 7];
                    }
                    pp += 16;
                }
                for (; jj < max_jj; jj++)
                {
                    const int lane = (j + jj) % 4;
                    float* outptr = (float*)top_blob + (size_t)((j + jj) / 4 * 4) * out_hstep + (i + ii) * 4;
                    outptr[lane] = pp[0];
                    outptr[4 + lane] = pp[1];
                    outptr[8 + lane] = pp[2];
                    outptr[12 + lane] = pp[3];
                    outptr[16 + lane] = pp[4];
                    outptr[20 + lane] = pp[5];
                    outptr[24 + lane] = pp[6];
                    outptr[28 + lane] = pp[7];
                    pp += 8;
                }
            }
            else
            {
                float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

                int jj = 0;
                for (; jj + 7 < max_jj; jj += 8)
                {
                    for (int c = 0; c < 8; c++)
                    {
                        float* outptr = p0 + out_hstep * c;
                        outptr[0] = pp[c * 8];
                        outptr[1] = pp[c * 8 + 1];
                        outptr[2] = pp[c * 8 + 2];
                        outptr[3] = pp[c * 8 + 3];
                        outptr[4] = pp[c * 8 + 4];
                        outptr[5] = pp[c * 8 + 5];
                        outptr[6] = pp[c * 8 + 6];
                        outptr[7] = pp[c * 8 + 7];
                    }
                    pp += 64;
                    p0 += out_hstep * 8;
                }
                for (; jj + 3 < max_jj; jj += 4)
                {
                    for (int c = 0; c < 4; c++)
                    {
                        float* outptr = p0 + out_hstep * c;
                        outptr[0] = pp[c * 8];
                        outptr[1] = pp[c * 8 + 1];
                        outptr[2] = pp[c * 8 + 2];
                        outptr[3] = pp[c * 8 + 3];
                        outptr[4] = pp[c * 8 + 4];
                        outptr[5] = pp[c * 8 + 5];
                        outptr[6] = pp[c * 8 + 6];
                        outptr[7] = pp[c * 8 + 7];
                    }
                    pp += 32;
                    p0 += out_hstep * 4;
                }
                for (; jj + 1 < max_jj; jj += 2)
                {
                    float* outptr = p0;
                    outptr[0] = pp[0];
                    outptr[1] = pp[1];
                    outptr[2] = pp[2];
                    outptr[3] = pp[3];
                    outptr[4] = pp[4];
                    outptr[5] = pp[5];
                    outptr[6] = pp[6];
                    outptr[7] = pp[7];
                    outptr += out_hstep;
                    outptr[0] = pp[8];
                    outptr[1] = pp[9];
                    outptr[2] = pp[10];
                    outptr[3] = pp[11];
                    outptr[4] = pp[12];
                    outptr[5] = pp[13];
                    outptr[6] = pp[14];
                    outptr[7] = pp[15];
                    pp += 16;
                    p0 += out_hstep * 2;
                }
                for (; jj < max_jj; jj++)
                {
                    p0[0] = pp[0];
                    p0[1] = pp[1];
                    p0[2] = pp[2];
                    p0[3] = pp[3];
                    p0[4] = pp[4];
                    p0[5] = pp[5];
                    p0[6] = pp[6];
                    p0[7] = pp[7];
                    pp += 8;
                    p0 += out_hstep;
                }
            }
        }
    }
#endif // __loongarch_sx

#if __loongarch_sx
    for (; ii + 3 < max_ii; ii += 4)
    {
#if __loongarch_sx
        if (out_elempack == 8)
        {
            int jj = 0;
#if __loongarch_asx
            for (; jj + 15 < max_jj; jj += 16)
            {
                for (int c = 0; c < 16; c++)
                {
                    const int col = j + jj + c;
                    const int lane = col % 8;
                    float* outptr = (float*)top_blob + (size_t)(col / 8 * 8) * out_hstep + (i + ii) * 8;
                    outptr[lane] = pp[c];
                    outptr[8 + lane] = pp[16 + c];
                    outptr[16 + lane] = pp[32 + c];
                    outptr[24 + lane] = pp[48 + c];
                }
                pp += 64;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                for (int c = 0; c < 8; c++)
                {
                    const int col = j + jj + c;
                    const int lane = col % 8;
                    float* outptr = (float*)top_blob + (size_t)(col / 8 * 8) * out_hstep + (i + ii) * 8;
                    outptr[lane] = pp[c];
                    outptr[8 + lane] = pp[8 + c];
                    outptr[16 + lane] = pp[16 + c];
                    outptr[24 + lane] = pp[24 + c];
                }
                pp += 32;
            }
#endif
            for (; jj + 3 < max_jj; jj += 4)
            {
                const int col = j + jj;
                const int lane = col & 7;
                float* p0 = (float*)top_blob + (size_t)(col / 8 * 8) * out_hstep + (i + ii) * 8;

                __m128 _r0 = (__m128)__lsx_vld(pp, 0);
                __m128 _r1 = (__m128)__lsx_vld(pp + 4, 0);
                __m128 _r2 = (__m128)__lsx_vld(pp + 8, 0);
                __m128 _r3 = (__m128)__lsx_vld(pp + 12, 0);
                transpose4x4_ps(_r0, _r1, _r2, _r3);
                __lsx_vst((__m128i)_r0, p0 + lane, 0);
                __lsx_vst((__m128i)_r1, p0 + 8 + lane, 0);
                __lsx_vst((__m128i)_r2, p0 + 16 + lane, 0);
                __lsx_vst((__m128i)_r3, p0 + 24 + lane, 0);
                pp += 16;
            }
            for (; jj < max_jj; jj++)
            {
                const int lane = (j + jj) % 8;
                float* outptr = (float*)top_blob + (size_t)((j + jj) / 8 * 8) * out_hstep + (i + ii) * 8;
                outptr[lane] = pp[0];
                outptr[8 + lane] = pp[1];
                outptr[16 + lane] = pp[2];
                outptr[24 + lane] = pp[3];
                pp += 4;
            }
        }
        else if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            int jj = 0;
#if __loongarch_asx
            for (; jj + 15 < max_jj; jj += 16)
            {
                for (int c = 0; c < 16; c++)
                {
                    const int col = j + jj + c;
                    const int lane = col % 4;
                    float* outptr = (float*)top_blob + (size_t)(col / 4 * 4) * out_hstep + (i + ii) * 4;
                    outptr[lane] = pp[c];
                    outptr[4 + lane] = pp[16 + c];
                    outptr[8 + lane] = pp[32 + c];
                    outptr[12 + lane] = pp[48 + c];
                }
                pp += 64;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                for (int c = 0; c < 8; c++)
                {
                    const int col = j + jj + c;
                    const int lane = col % 4;
                    float* outptr = (float*)top_blob + (size_t)(col / 4 * 4) * out_hstep + (i + ii) * 4;
                    outptr[lane] = pp[c];
                    outptr[4 + lane] = pp[8 + c];
                    outptr[8 + lane] = pp[16 + c];
                    outptr[12 + lane] = pp[24 + c];
                }
                pp += 32;
                p0 += out_hstep * 8;
            }
#endif
            for (; jj + 7 < max_jj; jj += 8)
            {
                for (int kk = 0; kk < 2; kk++)
                {
                    const float* ppk = pp + kk * 16;
                    __m128 _r0 = (__m128)__lsx_vld(ppk, 0);
                    __m128 _r1 = (__m128)__lsx_vld(ppk + 4, 0);
                    __m128 _r2 = (__m128)__lsx_vld(ppk + 8, 0);
                    __m128 _r3 = (__m128)__lsx_vld(ppk + 12, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vst((__m128i)_r0, p0, 0);
                    __lsx_vst((__m128i)_r1, p0 + 4, 0);
                    __lsx_vst((__m128i)_r2, p0 + 8, 0);
                    __lsx_vst((__m128i)_r3, p0 + 12, 0);
                    p0 += out_hstep * 4;
                }
                pp += 32;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                __m128 _r0 = (__m128)__lsx_vld(pp, 0);
                __m128 _r1 = (__m128)__lsx_vld(pp + 4, 0);
                __m128 _r2 = (__m128)__lsx_vld(pp + 8, 0);
                __m128 _r3 = (__m128)__lsx_vld(pp + 12, 0);
                transpose4x4_ps(_r0, _r1, _r2, _r3);
                __lsx_vst((__m128i)_r0, p0, 0);
                __lsx_vst((__m128i)_r1, p0 + 4, 0);
                __lsx_vst((__m128i)_r2, p0 + 8, 0);
                __lsx_vst((__m128i)_r3, p0 + 12, 0);
                pp += 16;
                p0 += out_hstep * 4;
            }
        }
        else
#endif // __loongarch_sx
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

#if __loongarch_asx
            int jj = 0;
            for (; jj + 15 < max_jj; jj += 16)
            {
                for (int c = 0; c < 16; c++)
                {
                    float* outptr = p0 + out_hstep * c;
                    outptr[0] = pp[c];
                    outptr[1] = pp[16 + c];
                    outptr[2] = pp[32 + c];
                    outptr[3] = pp[48 + c];
                }
                pp += 64;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                for (int c = 0; c < 8; c++)
                {
                    float* outptr = p0 + out_hstep * c;
                    outptr[0] = pp[c];
                    outptr[1] = pp[8 + c];
                    outptr[2] = pp[16 + c];
                    outptr[3] = pp[24 + c];
                }
                pp += 32;
                p0 += out_hstep * 8;
            }
            for (; jj < max_jj; jj++)
            {
                p0[0] = pp[0];
                p0[1] = pp[1];
                p0[2] = pp[2];
                p0[3] = pp[3];
                pp += 4;
                p0 += out_hstep;
            }
#else
            int jj = 0;
            for (; jj + 7 < max_jj; jj += 8)
            {
                for (int c = 0; c < 8; c++)
                {
                    float* outptr = p0 + out_hstep * c;
                    outptr[0] = pp[c * 4];
                    outptr[1] = pp[c * 4 + 1];
                    outptr[2] = pp[c * 4 + 2];
                    outptr[3] = pp[c * 4 + 3];
                }
                pp += 32;
                p0 += out_hstep * 8;
            }
            for (; jj < max_jj; jj++)
            {
                p0[0] = pp[0];
                p0[1] = pp[1];
                p0[2] = pp[2];
                p0[3] = pp[3];
                pp += 4;
                p0 += out_hstep;
            }
#endif
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __loongarch_sx
        if (out_elempack == 8)
        {
            int jj = 0;
#if __loongarch_asx
            for (; jj + 15 < max_jj; jj += 16)
            {
                for (int c = 0; c < 16; c++)
                {
                    const int col = j + jj + c;
                    const int lane = col % 8;
                    float* outptr = (float*)top_blob + (size_t)(col / 8 * 8) * out_hstep + (i + ii) * 8;
                    outptr[lane] = pp[c];
                    outptr[8 + lane] = pp[16 + c];
                }
                pp += 32;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                for (int c = 0; c < 8; c++)
                {
                    const int col = j + jj + c;
                    const int lane = col % 8;
                    float* outptr = (float*)top_blob + (size_t)(col / 8 * 8) * out_hstep + (i + ii) * 8;
                    outptr[lane] = pp[c];
                    outptr[8 + lane] = pp[8 + c];
                }
                pp += 16;
            }
#endif
            for (; jj + 3 < max_jj; jj += 4)
            {
                for (int c = 0; c < 4; c++)
                {
                    const int col = j + jj + c;
                    const int lane = col % 8;
                    float* outptr = (float*)top_blob + (size_t)(col / 8 * 8) * out_hstep + (i + ii) * 8;
                    outptr[lane] = pp[c * 2];
                    outptr[8 + lane] = pp[c * 2 + 1];
                }
                pp += 8;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                for (int c = 0; c < 2; c++)
                {
                    const int col = j + jj + c;
                    const int lane = col % 8;
                    float* outptr = (float*)top_blob + (size_t)(col / 8 * 8) * out_hstep + (i + ii) * 8;
                    outptr[lane] = pp[c * 2];
                    outptr[8 + lane] = pp[c * 2 + 1];
                }
                pp += 4;
            }
            for (; jj < max_jj; jj++)
            {
                const int lane = (j + jj) % 8;
                float* outptr = (float*)top_blob + (size_t)((j + jj) / 8 * 8) * out_hstep + (i + ii) * 8;
                outptr[lane] = pp[0];
                outptr[8 + lane] = pp[1];
                pp += 2;
            }
        }
        else if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            int jj = 0;
#if __loongarch_asx
            for (; jj + 15 < max_jj; jj += 16)
            {
                for (int c = 0; c < 16; c++)
                {
                    const int col = j + jj + c;
                    const int lane = col % 4;
                    float* outptr = (float*)top_blob + (size_t)(col / 4 * 4) * out_hstep + (i + ii) * 4;
                    outptr[lane] = pp[c];
                    outptr[4 + lane] = pp[16 + c];
                }
                pp += 32;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                for (int c = 0; c < 8; c++)
                {
                    const int col = j + jj + c;
                    const int lane = col % 4;
                    float* outptr = (float*)top_blob + (size_t)(col / 4 * 4) * out_hstep + (i + ii) * 4;
                    outptr[lane] = pp[c];
                    outptr[4 + lane] = pp[8 + c];
                }
                pp += 16;
                p0 += out_hstep * 8;
            }
#endif
            for (; jj + 7 < max_jj; jj += 8)
            {
                for (int c = 0; c < 8; c++)
                {
                    const int col = j + jj + c;
                    const int lane = col % 4;
                    float* outptr = (float*)top_blob + (size_t)(col / 4 * 4) * out_hstep + (i + ii) * 4;
                    outptr[lane] = pp[c * 2];
                    outptr[4 + lane] = pp[c * 2 + 1];
                }
                pp += 16;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                p0[0] = pp[0];
                p0[1] = pp[2];
                p0[2] = pp[4];
                p0[3] = pp[6];
                p0[4] = pp[1];
                p0[5] = pp[3];
                p0[6] = pp[5];
                p0[7] = pp[7];
                pp += 8;
                p0 += out_hstep * 4;
            }
        }
        else
#endif // __loongarch_sx
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

#if __loongarch_asx
            int jj = 0;
            for (; jj + 15 < max_jj; jj += 16)
            {
                for (int c = 0; c < 16; c++)
                {
                    float* outptr = p0 + out_hstep * c;
                    outptr[0] = pp[c];
                    outptr[1] = pp[16 + c];
                }
                pp += 32;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                for (int c = 0; c < 8; c++)
                {
                    float* outptr = p0 + out_hstep * c;
                    outptr[0] = pp[c];
                    outptr[1] = pp[8 + c];
                }
                pp += 16;
                p0 += out_hstep * 8;
            }
            for (; jj < max_jj; jj++)
            {
                p0[0] = pp[0];
                p0[1] = pp[1];
                pp += 2;
                p0 += out_hstep;
            }
#else
            int jj = 0;
            for (; jj + 7 < max_jj; jj += 8)
            {
                for (int c = 0; c < 8; c++)
                {
                    float* outptr = p0 + out_hstep * c;
                    outptr[0] = pp[c * 2];
                    outptr[1] = pp[c * 2 + 1];
                }
                pp += 16;
                p0 += out_hstep * 8;
            }
            for (; jj < max_jj; jj++)
            {
                p0[0] = pp[0];
                p0[1] = pp[1];
                pp += 2;
                p0 += out_hstep;
            }
#endif
        }
    }

    for (; ii < max_ii; ii += 1)
    {
#if __loongarch_sx
        if (out_elempack == 8)
        {
            int jj = 0;
#if __loongarch_asx
            for (; jj + 15 < max_jj; jj += 16)
            {
                for (int c = 0; c < 16; c++)
                {
                    const int col = j + jj + c;
                    float* outptr = (float*)top_blob + (size_t)(col / 8 * 8) * out_hstep + (i + ii) * 8;
                    outptr[col % 8] = pp[c];
                }
                pp += 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                for (int c = 0; c < 8; c++)
                {
                    float* outptr = (float*)top_blob + (size_t)((j + jj + c) / 8 * 8) * out_hstep + (i + ii) * 8;
                    outptr[(j + jj + c) % 8] = pp[c];
                }
                pp += 8;
            }
#endif
            for (; jj + 3 < max_jj; jj += 4)
            {
                for (int c = 0; c < 4; c++)
                {
                    float* outptr = (float*)top_blob + (size_t)((j + jj + c) / 8 * 8) * out_hstep + (i + ii) * 8;
                    outptr[(j + jj + c) % 8] = pp[c];
                }
                pp += 4;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                for (int c = 0; c < 2; c++)
                {
                    float* outptr = (float*)top_blob + (size_t)((j + jj + c) / 8 * 8) * out_hstep + (i + ii) * 8;
                    outptr[(j + jj + c) % 8] = pp[c];
                }
                pp += 2;
            }
            for (; jj < max_jj; jj++)
            {
                float* outptr = (float*)top_blob + (size_t)((j + jj) / 8 * 8) * out_hstep + (i + ii) * 8;
                outptr[(j + jj) % 8] = pp[0];
                pp += 1;
            }
        }
        else if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            int jj = 0;
#if __loongarch_asx
            for (; jj + 15 < max_jj; jj += 16)
            {
                for (int c = 0; c < 16; c++)
                {
                    const int col = j + jj + c;
                    float* outptr = (float*)top_blob + (size_t)(col / 4 * 4) * out_hstep + (i + ii) * 4;
                    outptr[col % 4] = pp[c];
                }
                pp += 16;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                for (int c = 0; c < 8; c++)
                {
                    const int col = j + jj + c;
                    float* outptr = (float*)top_blob + (size_t)(col / 4 * 4) * out_hstep + (i + ii) * 4;
                    outptr[col % 4] = pp[c];
                }
                pp += 8;
                p0 += out_hstep * 8;
            }
#endif
            for (; jj + 7 < max_jj; jj += 8)
            {
                for (int c = 0; c < 8; c++)
                {
                    const int col = j + jj + c;
                    float* outptr = (float*)top_blob + (size_t)(col / 4 * 4) * out_hstep + (i + ii) * 4;
                    outptr[col % 4] = pp[c];
                }
                pp += 8;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                __lsx_vst(__lsx_vld(pp, 0), p0, 0);
                pp += 4;
                p0 += out_hstep * 4;
            }
        }
        else
#endif // __loongarch_sx
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

#if __loongarch_asx
            int jj = 0;
            for (; jj + 15 < max_jj; jj += 16)
            {
                for (int c = 0; c < 16; c++)
                    p0[out_hstep * c] = pp[c];
                pp += 16;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                for (int c = 0; c < 8; c++)
                    p0[out_hstep * c] = pp[c];
                pp += 8;
                p0 += out_hstep * 8;
            }
            for (; jj < max_jj; jj++)
            {
                p0[0] = pp[0];
                pp += 1;
                p0 += out_hstep;
            }
#else
            int jj = 0;
            for (; jj + 7 < max_jj; jj += 8)
            {
                for (int c = 0; c < 8; c++)
                    p0[out_hstep * c] = pp[c];
                pp += 8;
                p0 += out_hstep * 8;
            }
            for (; jj < max_jj; jj++)
            {
                p0[0] = pp[0];
                pp += 1;
                p0 += out_hstep;
            }
#endif
        }
    }
}

static void gemm_transB_packed_tile(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const float* pAT = AT_tile;
    const float* pBT = BT_tile;
    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __loongarch_sx
    if (use_8row_packed_kernel(out_elempack))
    {
        for (; ii + 7 < max_ii; ii += 8)
        {
            float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

            const float* pB = pBT;
            const float* pCi = pC;

            if (pCi)
            {
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    pCi = (const float*)CT_tile + i + ii;
                if (broadcast_type_C == 4)
                    pCi = (const float*)CT_tile + j;
            }

            int jj = 0;
            for (; jj + 7 < max_jj; jj += 8)
            {
                __m128 _sum0[8];
                __m128 _sum1[8];

                if (k == 0)
                {
                    for (int c = 0; c < 8; c++)
                    {
                        _sum0[c] = (__m128)__lsx_vreplgr2vr_w(0);
                        _sum1[c] = (__m128)__lsx_vreplgr2vr_w(0);
                    }

                    if (pCi)
                    {
                        if (broadcast_type_C == 0)
                        {
                            __m128 _c0 = __lsx_vreplfr2vr_s(pCi[0]);
                            for (int c = 0; c < 8; c++)
                            {
                                _sum0[c] = _c0;
                                _sum1[c] = _c0;
                            }
                        }
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        {
                            __m128 _c0 = (__m128)__lsx_vld(pCi, 0);
                            __m128 _c1 = (__m128)__lsx_vld(pCi + 4, 0);
                            for (int c = 0; c < 8; c++)
                            {
                                _sum0[c] = _c0;
                                _sum1[c] = _c1;
                            }
                        }
                        if (broadcast_type_C == 3)
                        {
                            for (int c = 0; c < 8; c++)
                            {
                                _sum0[c] = (__m128)__lsx_vld(pCi + c * 8, 0);
                                _sum1[c] = (__m128)__lsx_vld(pCi + c * 8 + 4, 0);
                            }
                            pCi += 64;
                        }
                        if (broadcast_type_C == 4)
                        {
                            for (int c = 0; c < 8; c++)
                            {
                                __m128 _c0 = __lsx_vreplfr2vr_s(pCi[c]);
                                _sum0[c] = _c0;
                                _sum1[c] = _c0;
                            }
                            pCi += 8;
                        }
                    }
                }
                else
                {
                    for (int c = 0; c < 8; c++)
                    {
                        _sum0[c] = (__m128)__lsx_vld(outptr + c * 8, 0);
                        _sum1[c] = (__m128)__lsx_vld(outptr + c * 8 + 4, 0);
                    }
                }

                const float* pA = pAT;
                for (int kk = 0; kk < max_kk; kk++)
                {
                    __m128 _pA0 = (__m128)__lsx_vld(pA, 0);
                    __m128 _pA1 = (__m128)__lsx_vld(pA + 4, 0);
                    for (int c = 0; c < 8; c++)
                    {
                        __m128 _pB = __lsx_vreplfr2vr_s(pB[c]);
                        _sum0[c] = __lsx_vfmadd_s(_pB, _pA0, _sum0[c]);
                        _sum1[c] = __lsx_vfmadd_s(_pB, _pA1, _sum1[c]);
                    }
                    pA += 8;
                    pB += 8;
                }

                if (k_end)
                {
                    float block[64];
                    for (int c = 0; c < 8; c++)
                    {
                        __lsx_vst((__m128i)_sum0[c], block + c * 8, 0);
                        __lsx_vst((__m128i)_sum1[c], block + c * 8 + 4, 0);
                    }
                    float* outptr1 = (float*)top_blob;
                    for (int c = 0; c < 8; c++)
                    {
                        const int col = j + jj + c;
                        for (int r = 0; r < 8; r++)
                        {
                            const int row = i + ii + r;
                            outptr1[(size_t)(row / out_elempack) * out_hstep * out_elempack + (size_t)col * out_elempack + row % out_elempack] = block[c * 8 + r];
                        }
                    }
                    outptr0 += 8 * out_elempack;
                }
                else
                {
                    for (int c = 0; c < 8; c++)
                    {
                        __lsx_vst((__m128i)_sum0[c], outptr + c * 8, 0);
                        __lsx_vst((__m128i)_sum1[c], outptr + c * 8 + 4, 0);
                    }
                }

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

                    if (pCi)
                    {
                        if (broadcast_type_C == 0)
                        {
                            _sum00 = __lsx_vreplfr2vr_s(pCi[0]);
                            _sum01 = _sum00;
                            _sum10 = _sum00;
                            _sum11 = _sum00;
                            _sum20 = _sum00;
                            _sum21 = _sum00;
                            _sum30 = _sum00;
                            _sum31 = _sum00;
                        }
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        {
                            _sum00 = (__m128)__lsx_vld(pCi, 0);
                            _sum01 = (__m128)__lsx_vld(pCi + 4, 0);
                            _sum10 = _sum00;
                            _sum11 = _sum01;
                            _sum20 = _sum00;
                            _sum21 = _sum01;
                            _sum30 = _sum00;
                            _sum31 = _sum01;
                        }
                        if (broadcast_type_C == 3)
                        {
                            _sum00 = (__m128)__lsx_vld(pCi, 0);
                            _sum01 = (__m128)__lsx_vld(pCi + 4, 0);
                            _sum10 = (__m128)__lsx_vld(pCi + 8, 0);
                            _sum11 = (__m128)__lsx_vld(pCi + 12, 0);
                            _sum20 = (__m128)__lsx_vld(pCi + 16, 0);
                            _sum21 = (__m128)__lsx_vld(pCi + 20, 0);
                            _sum30 = (__m128)__lsx_vld(pCi + 24, 0);
                            _sum31 = (__m128)__lsx_vld(pCi + 28, 0);
                            pCi += 32;
                        }
                        if (broadcast_type_C == 4)
                        {
                            _sum00 = __lsx_vreplfr2vr_s(pCi[0]);
                            _sum10 = __lsx_vreplfr2vr_s(pCi[1]);
                            _sum20 = __lsx_vreplfr2vr_s(pCi[2]);
                            _sum30 = __lsx_vreplfr2vr_s(pCi[3]);
                            _sum01 = _sum00;
                            _sum11 = _sum10;
                            _sum21 = _sum20;
                            _sum31 = _sum30;
                            pCi += 4;
                        }
                    }
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

                const float* pA = pAT;
                for (int kk = 0; kk < max_kk; kk++)
                {
                    __m128 _pA0 = (__m128)__lsx_vld(pA, 0);
                    __m128 _pA1 = (__m128)__lsx_vld(pA + 4, 0);
                    _sum00 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[0]), _pA0, _sum00);
                    _sum01 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[0]), _pA1, _sum01);
                    _sum10 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[1]), _pA0, _sum10);
                    _sum11 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[1]), _pA1, _sum11);
                    _sum20 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[2]), _pA0, _sum20);
                    _sum21 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[2]), _pA1, _sum21);
                    _sum30 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[3]), _pA0, _sum30);
                    _sum31 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[3]), _pA1, _sum31);
                    pA += 8;
                    pB += 4;
                }

                if (k_end)
                {
                    if (out_elempack == 4)
                    {
                        float* outptr1 = outptr0 + out_hstep * 4;
                        __lsx_vst((__m128i)_sum00, outptr0, 0);
                        __lsx_vst((__m128i)_sum10, outptr0 + 4, 0);
                        __lsx_vst((__m128i)_sum20, outptr0 + 8, 0);
                        __lsx_vst((__m128i)_sum30, outptr0 + 12, 0);
                        __lsx_vst((__m128i)_sum01, outptr1, 0);
                        __lsx_vst((__m128i)_sum11, outptr1 + 4, 0);
                        __lsx_vst((__m128i)_sum21, outptr1 + 8, 0);
                        __lsx_vst((__m128i)_sum31, outptr1 + 12, 0);
                        outptr0 += 16;
                    }
                    else
                    {
                        __m128 _r0 = _sum00;
                        __m128 _r1 = _sum10;
                        __m128 _r2 = _sum20;
                        __m128 _r3 = _sum30;
                        transpose4x4_ps(_r0, _r1, _r2, _r3);
                        __lsx_vst((__m128i)_r0, outptr0, 0);
                        __lsx_vst((__m128i)_r1, outptr0 + out_hstep, 0);
                        __lsx_vst((__m128i)_r2, outptr0 + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_r3, outptr0 + out_hstep * 3, 0);
                        __m128 _r4 = _sum01;
                        __m128 _r5 = _sum11;
                        __m128 _r6 = _sum21;
                        __m128 _r7 = _sum31;
                        transpose4x4_ps(_r4, _r5, _r6, _r7);
                        __lsx_vst((__m128i)_r4, outptr0 + out_hstep * 4, 0);
                        __lsx_vst((__m128i)_r5, outptr0 + out_hstep * 5, 0);
                        __lsx_vst((__m128i)_r6, outptr0 + out_hstep * 6, 0);
                        __lsx_vst((__m128i)_r7, outptr0 + out_hstep * 7, 0);
                        outptr0 += 4;
                    }
                }
                else
                {
                    __lsx_vst((__m128i)_sum00, outptr, 0);
                    __lsx_vst((__m128i)_sum01, outptr + 4, 0);
                    __lsx_vst((__m128i)_sum10, outptr + 8, 0);
                    __lsx_vst((__m128i)_sum11, outptr + 12, 0);
                    __lsx_vst((__m128i)_sum20, outptr + 16, 0);
                    __lsx_vst((__m128i)_sum21, outptr + 20, 0);
                    __lsx_vst((__m128i)_sum30, outptr + 24, 0);
                    __lsx_vst((__m128i)_sum31, outptr + 28, 0);
                }

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

                    if (pCi)
                    {
                        if (broadcast_type_C == 0)
                        {
                            _sum00 = __lsx_vreplfr2vr_s(pCi[0]);
                            _sum01 = _sum00;
                            _sum10 = _sum00;
                            _sum11 = _sum00;
                        }
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        {
                            _sum00 = (__m128)__lsx_vld(pCi, 0);
                            _sum01 = (__m128)__lsx_vld(pCi + 4, 0);
                            _sum10 = _sum00;
                            _sum11 = _sum01;
                        }
                        if (broadcast_type_C == 3)
                        {
                            _sum00 = (__m128)__lsx_vld(pCi, 0);
                            _sum01 = (__m128)__lsx_vld(pCi + 4, 0);
                            _sum10 = (__m128)__lsx_vld(pCi + 8, 0);
                            _sum11 = (__m128)__lsx_vld(pCi + 12, 0);
                            pCi += 16;
                        }
                        if (broadcast_type_C == 4)
                        {
                            _sum00 = __lsx_vreplfr2vr_s(pCi[0]);
                            _sum10 = __lsx_vreplfr2vr_s(pCi[1]);
                            _sum01 = _sum00;
                            _sum11 = _sum10;
                            pCi += 2;
                        }
                    }
                }
                else
                {
                    _sum00 = (__m128)__lsx_vld(outptr, 0);
                    _sum01 = (__m128)__lsx_vld(outptr + 4, 0);
                    _sum10 = (__m128)__lsx_vld(outptr + 8, 0);
                    _sum11 = (__m128)__lsx_vld(outptr + 12, 0);
                }

                const float* pA = pAT;
                for (int kk = 0; kk < max_kk; kk++)
                {
                    __m128 _pA0 = (__m128)__lsx_vld(pA, 0);
                    __m128 _pA1 = (__m128)__lsx_vld(pA + 4, 0);
                    _sum00 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[0]), _pA0, _sum00);
                    _sum01 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[0]), _pA1, _sum01);
                    _sum10 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[1]), _pA0, _sum10);
                    _sum11 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[1]), _pA1, _sum11);
                    pA += 8;
                    pB += 2;
                }

                if (k_end)
                {
                    if (out_elempack == 4)
                    {
                        float* outptr1 = outptr0 + out_hstep * 4;
                        __lsx_vst((__m128i)_sum00, outptr0, 0);
                        __lsx_vst((__m128i)_sum10, outptr0 + 4, 0);
                        __lsx_vst((__m128i)_sum01, outptr1, 0);
                        __lsx_vst((__m128i)_sum11, outptr1 + 4, 0);
                        outptr0 += 8;
                    }
                    else
                    {
                        float tmp0[4];
                        float tmp1[4];
                        float tmp2[4];
                        float tmp3[4];
                        __lsx_vst((__m128i)_sum00, tmp0, 0);
                        __lsx_vst((__m128i)_sum10, tmp1, 0);
                        __lsx_vst((__m128i)_sum01, tmp2, 0);
                        __lsx_vst((__m128i)_sum11, tmp3, 0);
                        for (int r = 0; r < 4; r++)
                        {
                            outptr0[out_hstep * r] = tmp0[r];
                            outptr0[out_hstep * r + 1] = tmp1[r];
                            outptr0[out_hstep * (r + 4)] = tmp2[r];
                            outptr0[out_hstep * (r + 4) + 1] = tmp3[r];
                        }
                        outptr0 += 2;
                    }
                }
                else
                {
                    __lsx_vst((__m128i)_sum00, outptr, 0);
                    __lsx_vst((__m128i)_sum01, outptr + 4, 0);
                    __lsx_vst((__m128i)_sum10, outptr + 8, 0);
                    __lsx_vst((__m128i)_sum11, outptr + 12, 0);
                }

                outptr += 16;
            }

            for (; jj < max_jj; jj += 1)
            {
                __m128 _sum00;
                __m128 _sum01;

                if (k == 0)
                {
                    _sum00 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum01 = (__m128)__lsx_vreplgr2vr_w(0);

                    if (pCi)
                    {
                        if (broadcast_type_C == 0)
                        {
                            _sum00 = __lsx_vreplfr2vr_s(pCi[0]);
                            _sum01 = _sum00;
                        }
                        if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        {
                            _sum00 = (__m128)__lsx_vld(pCi, 0);
                            _sum01 = (__m128)__lsx_vld(pCi + 4, 0);
                        }
                        if (broadcast_type_C == 3)
                        {
                            _sum00 = (__m128)__lsx_vld(pCi, 0);
                            _sum01 = (__m128)__lsx_vld(pCi + 4, 0);
                            pCi += 8;
                        }
                        if (broadcast_type_C == 4)
                        {
                            _sum00 = __lsx_vreplfr2vr_s(pCi[0]);
                            _sum01 = _sum00;
                            pCi += 1;
                        }
                    }
                }
                else
                {
                    _sum00 = (__m128)__lsx_vld(outptr, 0);
                    _sum01 = (__m128)__lsx_vld(outptr + 4, 0);
                }

                const float* pA = pAT;
                for (int kk = 0; kk < max_kk; kk++)
                {
                    __m128 _pA0 = (__m128)__lsx_vld(pA, 0);
                    __m128 _pA1 = (__m128)__lsx_vld(pA + 4, 0);
                    _sum00 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[0]), _pA0, _sum00);
                    _sum01 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[0]), _pA1, _sum01);
                    pA += 8;
                    pB += 1;
                }

                if (k_end)
                {
                    if (out_elempack == 4)
                    {
                        float* outptr1 = outptr0 + out_hstep * 4;
                        __lsx_vst((__m128i)_sum00, outptr0, 0);
                        __lsx_vst((__m128i)_sum01, outptr1, 0);
                        outptr0 += 4;
                    }
                    else
                    {
                        float tmp0[4];
                        float tmp1[4];
                        __lsx_vst((__m128i)_sum00, tmp0, 0);
                        __lsx_vst((__m128i)_sum01, tmp1, 0);
                        for (int r = 0; r < 4; r++)
                        {
                            outptr0[out_hstep * r] = tmp0[r];
                            outptr0[out_hstep * (r + 4)] = tmp1[r];
                        }
                        outptr0 += 1;
                    }
                }
                else
                {
                    __lsx_vst((__m128i)_sum00, outptr, 0);
                    __lsx_vst((__m128i)_sum01, outptr + 4, 0);
                }

                outptr += 8;
            }

            pAT += max_kk * 8;
        }
    }

    for (; ii + 3 < max_ii; ii += 4)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;
        const float* pCi = pC;

        if (pCi)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pCi = (const float*)CT_tile + i + ii;
            if (broadcast_type_C == 4)
                pCi = (const float*)CT_tile + j;
        }

        int jj = 0;
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _sum00;
            __m256 _sum01;
            __m256 _sum10;
            __m256 _sum11;
            __m256 _sum20;
            __m256 _sum21;
            __m256 _sum30;
            __m256 _sum31;

            if (k == 0)
            {
                _sum00 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum01 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum10 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum11 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum20 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum21 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum30 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum31 = (__m256)__lasx_xvreplgr2vr_w(0);

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum00 = _c;
                        _sum01 = _c;
                        _sum10 = _c;
                        _sum11 = _c;
                        _sum20 = _c;
                        _sum21 = _c;
                        _sum30 = _c;
                        _sum31 = _c;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum01 = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum10 = (__m256)__lasx_xvreplfr2vr_s(pCi[1]);
                        _sum11 = (__m256)__lasx_xvreplfr2vr_s(pCi[1]);
                        _sum20 = (__m256)__lasx_xvreplfr2vr_s(pCi[2]);
                        _sum21 = (__m256)__lasx_xvreplfr2vr_s(pCi[2]);
                        _sum30 = (__m256)__lasx_xvreplfr2vr_s(pCi[3]);
                        _sum31 = (__m256)__lasx_xvreplfr2vr_s(pCi[3]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        __attribute__((aligned(32))) float ctmp[64];
                        for (int r = 0; r < 4; r++)
                        {
                            for (int c = 0; c < 16; c++)
                            {
                                ctmp[r * 16 + c] = pCi[c * 4 + r];
                            }
                        }
                        _sum00 = (__m256)__lasx_xvld(ctmp, 0);
                        _sum01 = (__m256)__lasx_xvld(ctmp + 8, 0);
                        _sum10 = (__m256)__lasx_xvld(ctmp + 16, 0);
                        _sum11 = (__m256)__lasx_xvld(ctmp + 24, 0);
                        _sum20 = (__m256)__lasx_xvld(ctmp + 32, 0);
                        _sum21 = (__m256)__lasx_xvld(ctmp + 40, 0);
                        _sum30 = (__m256)__lasx_xvld(ctmp + 48, 0);
                        _sum31 = (__m256)__lasx_xvld(ctmp + 56, 0);
                        pCi += 64;
                    }
                    if (broadcast_type_C == 4)
                    {
                        __m256 _c0 = (__m256)__lasx_xvld(pCi, 0);
                        __m256 _c1 = (__m256)__lasx_xvld(pCi + 8, 0);
                        _sum00 = _c0;
                        _sum01 = _c1;
                        _sum10 = _c0;
                        _sum11 = _c1;
                        _sum20 = _c0;
                        _sum21 = _c1;
                        _sum30 = _c0;
                        _sum31 = _c1;
                        pCi += 16;
                    }
                }
            }
            else
            {
                _sum00 = (__m256)__lasx_xvld(outptr, 0);
                _sum01 = (__m256)__lasx_xvld(outptr + 8, 0);
                _sum10 = (__m256)__lasx_xvld(outptr + 16, 0);
                _sum11 = (__m256)__lasx_xvld(outptr + 24, 0);
                _sum20 = (__m256)__lasx_xvld(outptr + 32, 0);
                _sum21 = (__m256)__lasx_xvld(outptr + 40, 0);
                _sum30 = (__m256)__lasx_xvld(outptr + 48, 0);
                _sum31 = (__m256)__lasx_xvld(outptr + 56, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _pB0 = (__m256)__lasx_xvld(pB, 0);
                __m256 _pB1 = (__m256)__lasx_xvld(pB + 8, 0);
                __m256 _a0 = (__m256)__lasx_xvreplfr2vr_s(pA[0]);
                __m256 _a1 = (__m256)__lasx_xvreplfr2vr_s(pA[1]);
                __m256 _a2 = (__m256)__lasx_xvreplfr2vr_s(pA[2]);
                __m256 _a3 = (__m256)__lasx_xvreplfr2vr_s(pA[3]);
                _sum00 = __lasx_xvfmadd_s(_a0, _pB0, _sum00);
                _sum01 = __lasx_xvfmadd_s(_a0, _pB1, _sum01);
                _sum10 = __lasx_xvfmadd_s(_a1, _pB0, _sum10);
                _sum11 = __lasx_xvfmadd_s(_a1, _pB1, _sum11);
                _sum20 = __lasx_xvfmadd_s(_a2, _pB0, _sum20);
                _sum21 = __lasx_xvfmadd_s(_a2, _pB1, _sum21);
                _sum30 = __lasx_xvfmadd_s(_a3, _pB0, _sum30);
                _sum31 = __lasx_xvfmadd_s(_a3, _pB1, _sum31);
                pA += 4;
                pB += 16;
            }

            if (k_end)
            {
                __attribute__((aligned(32))) float tmp[64];
                __lasx_xvst(_sum00, tmp, 0);
                __lasx_xvst(_sum01, tmp + 8, 0);
                __lasx_xvst(_sum10, tmp + 16, 0);
                __lasx_xvst(_sum11, tmp + 24, 0);
                __lasx_xvst(_sum20, tmp + 32, 0);
                __lasx_xvst(_sum21, tmp + 40, 0);
                __lasx_xvst(_sum30, tmp + 48, 0);
                __lasx_xvst(_sum31, tmp + 56, 0);
                float* outptr1 = (float*)top_blob;
                for (int r = 0; r < 4; r++)
                {
                    const int row = i + ii + r;
                    for (int c = 0; c < 16; c++)
                    {
                        const int col = j + jj + c;
                        outptr1[(size_t)(row / out_elempack) * out_hstep * out_elempack + (size_t)col * out_elempack + row % out_elempack] = tmp[r * 16 + c];
                    }
                }
                outptr0 += 16 * out_elempack;
            }
            else
            {
                __lasx_xvst(_sum00, outptr, 0);
                __lasx_xvst(_sum01, outptr + 8, 0);
                __lasx_xvst(_sum10, outptr + 16, 0);
                __lasx_xvst(_sum11, outptr + 24, 0);
                __lasx_xvst(_sum20, outptr + 32, 0);
                __lasx_xvst(_sum21, outptr + 40, 0);
                __lasx_xvst(_sum30, outptr + 48, 0);
                __lasx_xvst(_sum31, outptr + 56, 0);
            }

            outptr += 64;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _sum0;
            __m256 _sum1;
            __m256 _sum2;
            __m256 _sum3;

            if (k == 0)
            {
                _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum1 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum2 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum3 = (__m256)__lasx_xvreplgr2vr_w(0);

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum0 = _c;
                        _sum1 = _c;
                        _sum2 = _c;
                        _sum3 = _c;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum1 = (__m256)__lasx_xvreplfr2vr_s(pCi[1]);
                        _sum2 = (__m256)__lasx_xvreplfr2vr_s(pCi[2]);
                        _sum3 = (__m256)__lasx_xvreplfr2vr_s(pCi[3]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        __attribute__((aligned(32))) float ctmp[32];
                        for (int r = 0; r < 4; r++)
                        {
                            for (int c = 0; c < 8; c++)
                            {
                                ctmp[r * 8 + c] = pCi[c * 4 + r];
                            }
                        }
                        _sum0 = (__m256)__lasx_xvld(ctmp, 0);
                        _sum1 = (__m256)__lasx_xvld(ctmp + 8, 0);
                        _sum2 = (__m256)__lasx_xvld(ctmp + 16, 0);
                        _sum3 = (__m256)__lasx_xvld(ctmp + 24, 0);
                        pCi += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        __m256 _c = (__m256)__lasx_xvld(pCi, 0);
                        _sum0 = _c;
                        _sum1 = _c;
                        _sum2 = _c;
                        _sum3 = _c;
                        pCi += 8;
                    }
                }
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
                _sum2 = (__m256)__lasx_xvld(outptr + 16, 0);
                _sum3 = (__m256)__lasx_xvld(outptr + 24, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _pB = (__m256)__lasx_xvld(pB, 0);
                _sum0 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pA[0]), _pB, _sum0);
                _sum1 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pA[1]), _pB, _sum1);
                _sum2 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pA[2]), _pB, _sum2);
                _sum3 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pA[3]), _pB, _sum3);
                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                __attribute__((aligned(32))) float tmp[32];
                __lasx_xvst(_sum0, tmp, 0);
                __lasx_xvst(_sum1, tmp + 8, 0);
                __lasx_xvst(_sum2, tmp + 16, 0);
                __lasx_xvst(_sum3, tmp + 24, 0);
                float* outptr1 = (float*)top_blob;
                for (int r = 0; r < 4; r++)
                {
                    const int row = i + ii + r;
                    for (int c = 0; c < 8; c++)
                    {
                        const int col = j + jj + c;
                        outptr1[(size_t)(row / out_elempack) * out_hstep * out_elempack + (size_t)col * out_elempack + row % out_elempack] = tmp[r * 8 + c];
                    }
                }
                outptr0 += 8 * out_elempack;
            }
            else
            {
                __lasx_xvst(_sum0, outptr, 0);
                __lasx_xvst(_sum1, outptr + 8, 0);
                __lasx_xvst(_sum2, outptr + 16, 0);
                __lasx_xvst(_sum3, outptr + 24, 0);
            }

            outptr += 32;
        }
#endif
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _sum[8];

            if (k == 0)
            {
                for (int c = 0; c < 8; c++)
                {
                    _sum[c] = (__m128)__lsx_vreplgr2vr_w(0);
                }

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        __m128 _c0 = __lsx_vreplfr2vr_s(pCi[0]);
                        for (int c = 0; c < 8; c++)
                            _sum[c] = _c0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        __m128 _c0 = (__m128)__lsx_vld(pCi, 0);
                        for (int c = 0; c < 8; c++)
                            _sum[c] = _c0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        for (int c = 0; c < 8; c++)
                            _sum[c] = (__m128)__lsx_vld(pCi + c * 4, 0);
                        pCi += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        for (int c = 0; c < 8; c++)
                            _sum[c] = __lsx_vreplfr2vr_s(pCi[c]);
                        pCi += 8;
                    }
                }
            }
            else
            {
                for (int c = 0; c < 8; c++)
                    _sum[c] = (__m128)__lsx_vld(outptr + c * 4, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pA = (__m128)__lsx_vld(pA, 0);
                for (int c = 0; c < 8; c++)
                {
                    _sum[c] = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[c]), _pA, _sum[c]);
                }
                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                float block[32];
                for (int c = 0; c < 8; c++)
                    __lsx_vst((__m128i)_sum[c], block + c * 4, 0);
                float* outptr1 = (float*)top_blob;
                for (int c = 0; c < 8; c++)
                {
                    const int col = j + jj + c;
                    for (int r = 0; r < 4; r++)
                    {
                        const int row = i + ii + r;
                        outptr1[(size_t)(row / out_elempack) * out_hstep * out_elempack + (size_t)col * out_elempack + row % out_elempack] = block[c * 4 + r];
                    }
                }
                outptr0 += 8 * out_elempack;
            }
            else
            {
                for (int c = 0; c < 8; c++)
                    __lsx_vst((__m128i)_sum[c], outptr + c * 4, 0);
            }

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
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

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __lsx_vreplfr2vr_s(pCi[0]);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = (__m128)__lsx_vld(pCi, 0);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = (__m128)__lsx_vld(pCi, 0);
                        _sum1 = (__m128)__lsx_vld(pCi + 4, 0);
                        _sum2 = (__m128)__lsx_vld(pCi + 8, 0);
                        _sum3 = (__m128)__lsx_vld(pCi + 12, 0);
                        pCi += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __lsx_vreplfr2vr_s(pCi[0]);
                        _sum1 = __lsx_vreplfr2vr_s(pCi[1]);
                        _sum2 = __lsx_vreplfr2vr_s(pCi[2]);
                        _sum3 = __lsx_vreplfr2vr_s(pCi[3]);
                        pCi += 4;
                    }
                }
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
                _sum2 = (__m128)__lsx_vld(outptr + 8, 0);
                _sum3 = (__m128)__lsx_vld(outptr + 12, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pA = (__m128)__lsx_vld(pA, 0);
                _sum0 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[0]), _pA, _sum0);
                _sum1 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[1]), _pA, _sum1);
                _sum2 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[2]), _pA, _sum2);
                _sum3 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[3]), _pA, _sum3);
                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    float tmp[16];
                    __lsx_vst((__m128i)_sum0, tmp, 0);
                    __lsx_vst((__m128i)_sum1, tmp + 4, 0);
                    __lsx_vst((__m128i)_sum2, tmp + 8, 0);
                    __lsx_vst((__m128i)_sum3, tmp + 12, 0);
                    float* outptr1 = (float*)top_blob;
                    for (int c = 0; c < 4; c++)
                    {
                        const int col = j + jj + c;
                        for (int r = 0; r < 4; r++)
                        {
                            const int row = i + ii + r;
                            outptr1[(size_t)(row / out_elempack) * out_hstep * out_elempack + (size_t)col * out_elempack + row % out_elempack] = tmp[c * 4 + r];
                        }
                    }
                    outptr0 += 4;
                }
                else if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_sum0, outptr0, 0);
                    __lsx_vst((__m128i)_sum1, outptr0 + 4, 0);
                    __lsx_vst((__m128i)_sum2, outptr0 + 8, 0);
                    __lsx_vst((__m128i)_sum3, outptr0 + 12, 0);
                    outptr0 += 16;
                }
                else
                {
                    __m128 _r0 = _sum0;
                    __m128 _r1 = _sum1;
                    __m128 _r2 = _sum2;
                    __m128 _r3 = _sum3;
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vst((__m128i)_r0, outptr0, 0);
                    __lsx_vst((__m128i)_r1, outptr0 + out_hstep, 0);
                    __lsx_vst((__m128i)_r2, outptr0 + out_hstep * 2, 0);
                    __lsx_vst((__m128i)_r3, outptr0 + out_hstep * 3, 0);
                    outptr0 += 4;
                }
            }
            else
            {
                __lsx_vst((__m128i)_sum0, outptr, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);
                __lsx_vst((__m128i)_sum2, outptr + 8, 0);
                __lsx_vst((__m128i)_sum3, outptr + 12, 0);
            }

            outptr += 16;
        }

        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum1 = (__m128)__lsx_vreplgr2vr_w(0);

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __lsx_vreplfr2vr_s(pCi[0]);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = (__m128)__lsx_vld(pCi, 0);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = (__m128)__lsx_vld(pCi, 0);
                        _sum1 = (__m128)__lsx_vld(pCi + 4, 0);
                        pCi += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __lsx_vreplfr2vr_s(pCi[0]);
                        _sum1 = __lsx_vreplfr2vr_s(pCi[1]);
                        pCi += 2;
                    }
                }
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pA = (__m128)__lsx_vld(pA, 0);
                _sum0 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[0]), _pA, _sum0);
                _sum1 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[1]), _pA, _sum1);
                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    float tmp[8];
                    __lsx_vst((__m128i)_sum0, tmp, 0);
                    __lsx_vst((__m128i)_sum1, tmp + 4, 0);
                    float* outptr1 = (float*)top_blob;
                    for (int c = 0; c < 2; c++)
                    {
                        const int col = j + jj + c;
                        for (int r = 0; r < 4; r++)
                        {
                            const int row = i + ii + r;
                            outptr1[(size_t)(row / out_elempack) * out_hstep * out_elempack + (size_t)col * out_elempack + row % out_elempack] = tmp[c * 4 + r];
                        }
                    }
                    outptr0 += 2;
                }
                else if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_sum0, outptr0, 0);
                    __lsx_vst((__m128i)_sum1, outptr0 + 4, 0);
                    outptr0 += 8;
                }
                else
                {
                    float tmp0[4];
                    float tmp1[4];
                    __lsx_vst((__m128i)_sum0, tmp0, 0);
                    __lsx_vst((__m128i)_sum1, tmp1, 0);
                    for (int r = 0; r < 4; r++)
                    {
                        outptr0[out_hstep * r] = tmp0[r];
                        outptr0[out_hstep * r + 1] = tmp1[r];
                    }
                    outptr0 += 2;
                }
            }
            else
            {
                __lsx_vst((__m128i)_sum0, outptr, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);
            }

            outptr += 8;
        }

        for (; jj < max_jj; jj += 1)
        {
            __m128 _sum0;

            if (k == 0)
            {
                _sum0 = (__m128)__lsx_vreplgr2vr_w(0);

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                        _sum0 = __lsx_vreplfr2vr_s(pCi[0]);
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        _sum0 = (__m128)__lsx_vld(pCi, 0);
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = (__m128)__lsx_vld(pCi, 0);
                        pCi += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __lsx_vreplfr2vr_s(pCi[0]);
                        pCi += 1;
                    }
                }
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pA = (__m128)__lsx_vld(pA, 0);
                _sum0 = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(pB[0]), _pA, _sum0);
                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    float tmp[4];
                    __lsx_vst((__m128i)_sum0, tmp, 0);
                    float* outptr1 = (float*)top_blob;
                    for (int r = 0; r < 4; r++)
                    {
                        const int row = i + ii + r;
                        outptr1[(size_t)(row / out_elempack) * out_hstep * out_elempack + (size_t)(j + jj) * out_elempack + row % out_elempack] = tmp[r];
                    }
                    outptr0 += 1;
                }
                else if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_sum0, outptr0, 0);
                    outptr0 += 4;
                }
                else
                {
                    float tmp0[4];
                    __lsx_vst((__m128i)_sum0, tmp0, 0);
                    for (int r = 0; r < 4; r++)
                    {
                        outptr0[out_hstep * r] = tmp0[r];
                    }
                    outptr0 += 1;
                }
            }
            else
            {
                __lsx_vst((__m128i)_sum0, outptr, 0);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __loongarch_sx

    for (; ii + 1 < max_ii; ii += 2)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j;

        const float* pB = pBT;
        const float* pCi = pC;

        if (pCi)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pCi = (const float*)CT_tile + i + ii;
            if (broadcast_type_C == 4)
                pCi = (const float*)CT_tile + j;
        }

        int jj = 0;
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _sum00;
            __m256 _sum01;
            __m256 _sum10;
            __m256 _sum11;

            if (k == 0)
            {
                _sum00 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum01 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum10 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum11 = (__m256)__lasx_xvreplgr2vr_w(0);

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum00 = _c;
                        _sum01 = _c;
                        _sum10 = _c;
                        _sum11 = _c;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum01 = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum10 = (__m256)__lasx_xvreplfr2vr_s(pCi[1]);
                        _sum11 = (__m256)__lasx_xvreplfr2vr_s(pCi[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        __attribute__((aligned(32))) float ctmp[32];
                        for (int r = 0; r < 2; r++)
                        {
                            for (int c = 0; c < 16; c++)
                            {
                                ctmp[r * 16 + c] = pCi[c * 2 + r];
                            }
                        }
                        _sum00 = (__m256)__lasx_xvld(ctmp, 0);
                        _sum01 = (__m256)__lasx_xvld(ctmp + 8, 0);
                        _sum10 = (__m256)__lasx_xvld(ctmp + 16, 0);
                        _sum11 = (__m256)__lasx_xvld(ctmp + 24, 0);
                        pCi += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        __m256 _c0 = (__m256)__lasx_xvld(pCi, 0);
                        __m256 _c1 = (__m256)__lasx_xvld(pCi + 8, 0);
                        _sum00 = _c0;
                        _sum01 = _c1;
                        _sum10 = _c0;
                        _sum11 = _c1;
                        pCi += 16;
                    }
                }
            }
            else
            {
                _sum00 = (__m256)__lasx_xvld(outptr, 0);
                _sum01 = (__m256)__lasx_xvld(outptr + 8, 0);
                _sum10 = (__m256)__lasx_xvld(outptr + 16, 0);
                _sum11 = (__m256)__lasx_xvld(outptr + 24, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _pB0 = (__m256)__lasx_xvld(pB, 0);
                __m256 _pB1 = (__m256)__lasx_xvld(pB + 8, 0);
                __m256 _a0 = (__m256)__lasx_xvreplfr2vr_s(pA[0]);
                __m256 _a1 = (__m256)__lasx_xvreplfr2vr_s(pA[1]);
                _sum00 = __lasx_xvfmadd_s(_a0, _pB0, _sum00);
                _sum01 = __lasx_xvfmadd_s(_a0, _pB1, _sum01);
                _sum10 = __lasx_xvfmadd_s(_a1, _pB0, _sum10);
                _sum11 = __lasx_xvfmadd_s(_a1, _pB1, _sum11);
                pA += 2;
                pB += 16;
            }

            if (k_end)
            {
                __attribute__((aligned(32))) float tmp[32];
                __lasx_xvst(_sum00, tmp, 0);
                __lasx_xvst(_sum01, tmp + 8, 0);
                __lasx_xvst(_sum10, tmp + 16, 0);
                __lasx_xvst(_sum11, tmp + 24, 0);
                float* outptr1 = (float*)top_blob;
                for (int r = 0; r < 2; r++)
                {
                    const int row = i + ii + r;
                    for (int c = 0; c < 16; c++)
                    {
                        const int col = j + jj + c;
                        outptr1[(size_t)(row / out_elempack) * out_hstep * out_elempack + (size_t)col * out_elempack + row % out_elempack] = tmp[r * 16 + c];
                    }
                }
                outptr0 += 16 * out_elempack;
            }
            else
            {
                __lasx_xvst(_sum00, outptr, 0);
                __lasx_xvst(_sum01, outptr + 8, 0);
                __lasx_xvst(_sum10, outptr + 16, 0);
                __lasx_xvst(_sum11, outptr + 24, 0);
            }

            outptr += 32;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _sum0;
            __m256 _sum1;

            if (k == 0)
            {
                _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum1 = (__m256)__lasx_xvreplgr2vr_w(0);

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum0 = _c;
                        _sum1 = _c;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum1 = (__m256)__lasx_xvreplfr2vr_s(pCi[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        __attribute__((aligned(32))) float ctmp[16];
                        for (int r = 0; r < 2; r++)
                        {
                            for (int c = 0; c < 8; c++)
                            {
                                ctmp[r * 8 + c] = pCi[c * 2 + r];
                            }
                        }
                        _sum0 = (__m256)__lasx_xvld(ctmp, 0);
                        _sum1 = (__m256)__lasx_xvld(ctmp + 8, 0);
                        pCi += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        __m256 _c = (__m256)__lasx_xvld(pCi, 0);
                        _sum0 = _c;
                        _sum1 = _c;
                        pCi += 8;
                    }
                }
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _pB = (__m256)__lasx_xvld(pB, 0);
                _sum0 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pA[0]), _pB, _sum0);
                _sum1 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pA[1]), _pB, _sum1);
                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                __attribute__((aligned(32))) float tmp[16];
                __lasx_xvst(_sum0, tmp, 0);
                __lasx_xvst(_sum1, tmp + 8, 0);
                float* outptr1 = (float*)top_blob;
                for (int r = 0; r < 2; r++)
                {
                    const int row = i + ii + r;
                    for (int c = 0; c < 8; c++)
                    {
                        const int col = j + jj + c;
                        outptr1[(size_t)(row / out_elempack) * out_hstep * out_elempack + (size_t)col * out_elempack + row % out_elempack] = tmp[r * 8 + c];
                    }
                }
                outptr0 += 8 * out_elempack;
            }
            else
            {
                __lasx_xvst(_sum0, outptr, 0);
                __lasx_xvst(_sum1, outptr + 8, 0);
            }

            outptr += 16;
        }
#endif
#if __loongarch_sx
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

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        for (int c = 0; c < 8; c++)
                        {
                            sum0[c] = pCi[0];
                            sum1[c] = pCi[0];
                        }
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        for (int c = 0; c < 8; c++)
                        {
                            sum0[c] = pCi[0];
                            sum1[c] = pCi[1];
                        }
                    }
                    if (broadcast_type_C == 3)
                    {
                        for (int c = 0; c < 8; c++)
                        {
                            sum0[c] = pCi[c * 2];
                            sum1[c] = pCi[c * 2 + 1];
                        }
                        pCi += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        for (int c = 0; c < 8; c++)
                        {
                            sum0[c] = pCi[c];
                            sum1[c] = pCi[c];
                        }
                        pCi += 8;
                    }
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

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const float a0 = pA[0];
                const float a1 = pA[1];
                for (int c = 0; c < 8; c++)
                {
                    sum0[c] += a0 * pB[c];
                    sum1[c] += a1 * pB[c];
                }
                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                float block[16];
                for (int c = 0; c < 8; c++)
                {
                    block[c * 2] = sum0[c];
                    block[c * 2 + 1] = sum1[c];
                }
                float* outptr1 = (float*)top_blob;
                for (int c = 0; c < 8; c++)
                {
                    const int col = j + jj + c;
                    for (int r = 0; r < 2; r++)
                    {
                        const int row = i + ii + r;
                        outptr1[(size_t)(row / out_elempack) * out_hstep * out_elempack + (size_t)col * out_elempack + row % out_elempack] = block[c * 2 + r];
                    }
                }
                outptr0 += 8;
            }
            else
            {
                for (int c = 0; c < 8; c++)
                {
                    outptr[c * 2] = sum0[c];
                    outptr[c * 2 + 1] = sum1[c];
                }
            }

            outptr += 16;
        }
#endif // __loongarch_sx
        for (; jj + 3 < max_jj; jj += 4)
        {
            float sum00;
            float sum01;
            float sum10;
            float sum11;
            float sum20;
            float sum21;
            float sum30;
            float sum31;

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

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        sum00 = pCi[0];
                        sum01 = pCi[0];
                        sum10 = sum00;
                        sum11 = sum00;
                        sum20 = sum00;
                        sum21 = sum00;
                        sum30 = sum00;
                        sum31 = sum00;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum00 = pCi[0];
                        sum01 = pCi[1];
                        sum10 = sum00;
                        sum11 = sum01;
                        sum20 = sum00;
                        sum21 = sum01;
                        sum30 = sum00;
                        sum31 = sum01;
                    }
                    if (broadcast_type_C == 3)
                    {
                        sum00 = pCi[0];
                        sum01 = pCi[1];
                        sum10 = pCi[2];
                        sum11 = pCi[3];
                        sum20 = pCi[4];
                        sum21 = pCi[5];
                        sum30 = pCi[6];
                        sum31 = pCi[7];
                        pCi += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        sum00 = pCi[0];
                        sum10 = pCi[1];
                        sum20 = pCi[2];
                        sum30 = pCi[3];
                        sum01 = sum00;
                        sum11 = sum10;
                        sum21 = sum20;
                        sum31 = sum30;
                        pCi += 4;
                    }
                }
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

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const float a0 = pA[0];
                const float a1 = pA[1];
                sum00 += a0 * pB[0];
                sum01 += a1 * pB[0];
                sum10 += a0 * pB[1];
                sum11 += a1 * pB[1];
                sum20 += a0 * pB[2];
                sum21 += a1 * pB[2];
                sum30 += a0 * pB[3];
                sum31 += a1 * pB[3];
                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                outptr0[0] = sum00;
                outptr0[1] = sum10;
                outptr0[2] = sum20;
                outptr0[3] = sum30;
                outptr0[out_hstep] = sum01;
                outptr0[out_hstep + 1] = sum11;
                outptr0[out_hstep + 2] = sum21;
                outptr0[out_hstep + 3] = sum31;
                outptr0 += 4;
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum01;
                outptr[2] = sum10;
                outptr[3] = sum11;
                outptr[4] = sum20;
                outptr[5] = sum21;
                outptr[6] = sum30;
                outptr[7] = sum31;
            }

            outptr += 8;
        }

        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum00;
            float sum01;
            float sum10;
            float sum11;

            if (k == 0)
            {
                sum00 = 0.f;
                sum01 = 0.f;
                sum10 = 0.f;
                sum11 = 0.f;

                if (pCi)
                {
                    if (broadcast_type_C == 0)
                    {
                        sum00 = pCi[0];
                        sum01 = pCi[0];
                        sum10 = sum00;
                        sum11 = sum00;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum00 = pCi[0];
                        sum01 = pCi[1];
                        sum10 = sum00;
                        sum11 = sum01;
                    }
                    if (broadcast_type_C == 3)
                    {
                        sum00 = pCi[0];
                        sum01 = pCi[1];
                        sum10 = pCi[2];
                        sum11 = pCi[3];
                        pCi += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        sum00 = pCi[0];
                        sum10 = pCi[1];
                        sum01 = sum00;
                        sum11 = sum10;
                        pCi += 2;
                    }
                }
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const float a0 = pA[0];
                const float a1 = pA[1];
                sum00 += a0 * pB[0];
                sum01 += a1 * pB[0];
                sum10 += a0 * pB[1];
                sum11 += a1 * pB[1];
                pA += 2;
                pB += 2;
            }

            if (k_end)
            {
                outptr0[0] = sum00;
                outptr0[1] = sum10;
                outptr0[out_hstep] = sum01;
                outptr0[out_hstep + 1] = sum11;
                outptr0 += 2;
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum01;
                outptr[2] = sum10;
                outptr[3] = sum11;
            }

            outptr += 4;
        }

        for (; jj < max_jj; jj += 1)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                sum0 = 0.f;
                sum1 = 0.f;

                if (pCi)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum0 = pCi[0];
                        sum1 = broadcast_type_C == 0 ? pCi[0] : pCi[1];
                    }
                    if (broadcast_type_C == 3)
                    {
                        sum0 = pCi[0];
                        sum1 = pCi[1];
                        pCi += 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        sum0 = pCi[0];
                        sum1 = pCi[0];
                        pCi += 1;
                    }
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const float a0 = pA[0];
                const float a1 = pA[1];
                sum0 += a0 * pB[0];
                sum1 += a1 * pB[0];
                pA += 2;
                pB += 1;
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0[out_hstep] = sum1;
                outptr0 += 1;
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }

        pAT += max_kk * 2;
    }

    for (; ii < max_ii; ii += 1)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j;

        const float* pB = pBT;
        const float* pCi = pC;

        if (pCi)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pCi = (const float*)CT_tile + i + ii;
            if (broadcast_type_C == 4)
                pCi = (const float*)CT_tile + j;
        }

        int jj = 0;
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _sum0;
            __m256 _sum1;

            if (k == 0)
            {
                _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum1 = (__m256)__lasx_xvreplgr2vr_w(0);

                if (pCi)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                        _sum0 = _c;
                        _sum1 = _c;
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = (__m256)__lasx_xvld(pCi, 0);
                        _sum1 = (__m256)__lasx_xvld(pCi + 8, 0);
                        pCi += 16;
                    }
                }
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _pB0 = (__m256)__lasx_xvld(pB, 0);
                __m256 _pB1 = (__m256)__lasx_xvld(pB + 8, 0);
                __m256 _a0 = (__m256)__lasx_xvreplfr2vr_s(pA[0]);
                _sum0 = __lasx_xvfmadd_s(_a0, _pB0, _sum0);
                _sum1 = __lasx_xvfmadd_s(_a0, _pB1, _sum1);
                pA += 1;
                pB += 16;
            }

            if (k_end)
            {
                __attribute__((aligned(32))) float tmp[16];
                __lasx_xvst(_sum0, tmp, 0);
                __lasx_xvst(_sum1, tmp + 8, 0);
                float* outptr1 = (float*)top_blob;
                const int row = i + ii;
                for (int c = 0; c < 16; c++)
                {
                    const int col = j + jj + c;
                    outptr1[(size_t)(row / out_elempack) * out_hstep * out_elempack + (size_t)col * out_elempack + row % out_elempack] = tmp[c];
                }
                outptr0 += 16 * out_elempack;
            }
            else
            {
                __lasx_xvst(_sum0, outptr, 0);
                __lasx_xvst(_sum1, outptr + 8, 0);
            }

            outptr += 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _sum0;

            if (k == 0)
            {
                _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);

                if (pCi)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = (__m256)__lasx_xvreplfr2vr_s(pCi[0]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = (__m256)__lasx_xvld(pCi, 0);
                        pCi += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = (__m256)__lasx_xvld(pCi, 0);
                        pCi += 8;
                    }
                }
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _pB = (__m256)__lasx_xvld(pB, 0);
                _sum0 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pA[0]), _pB, _sum0);
                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                __attribute__((aligned(32))) float tmp[8];
                __lasx_xvst(_sum0, tmp, 0);
                float* outptr1 = (float*)top_blob;
                const int row = i + ii;
                for (int c = 0; c < 8; c++)
                {
                    const int col = j + jj + c;
                    outptr1[(size_t)(row / out_elempack) * out_hstep * out_elempack + (size_t)col * out_elempack + row % out_elempack] = tmp[c];
                }
                outptr0 += 8 * out_elempack;
            }
            else
            {
                __lasx_xvst(_sum0, outptr, 0);
            }

            outptr += 8;
        }
#endif
#if __loongarch_sx
        for (; jj + 7 < max_jj; jj += 8)
        {
            float sum[8];

            if (k == 0)
            {
                for (int c = 0; c < 8; c++)
                    sum[c] = 0.f;

                if (pCi)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        for (int c = 0; c < 8; c++)
                            sum[c] = pCi[0];
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        for (int c = 0; c < 8; c++)
                            sum[c] = pCi[c];
                        pCi += 8;
                    }
                }
            }
            else
            {
                for (int c = 0; c < 8; c++)
                    sum[c] = outptr[c];
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const float a0 = pA[0];
                for (int c = 0; c < 8; c++)
                    sum[c] += a0 * pB[c];
                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                float* outptr1 = (float*)top_blob;
                const int row = i + ii;
                for (int c = 0; c < 8; c++)
                {
                    const int col = j + jj + c;
                    outptr1[(size_t)(row / out_elempack) * out_hstep * out_elempack + (size_t)col * out_elempack + row % out_elempack] = sum[c];
                }
                outptr0 += 8;
            }
            else
            {
                for (int c = 0; c < 8; c++)
                    outptr[c] = sum[c];
            }

            outptr += 8;
        }
#endif // __loongarch_sx
        for (; jj + 3 < max_jj; jj += 4)
        {
            float sum0;
            float sum1;
            float sum2;
            float sum3;

            if (k == 0)
            {
                sum0 = 0.f;
                sum1 = 0.f;
                sum2 = 0.f;
                sum3 = 0.f;

                if (pCi)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum0 = pCi[0];
                        sum1 = pCi[0];
                        sum2 = pCi[0];
                        sum3 = pCi[0];
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        sum0 = pCi[0];
                        sum1 = pCi[1];
                        sum2 = pCi[2];
                        sum3 = pCi[3];
                        pCi += 4;
                    }
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
                sum2 = outptr[2];
                sum3 = outptr[3];
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const float a0 = pA[0];
                sum0 += a0 * pB[0];
                sum1 += a0 * pB[1];
                sum2 += a0 * pB[2];
                sum3 += a0 * pB[3];
                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0[1] = sum1;
                outptr0[2] = sum2;
                outptr0[3] = sum3;
                outptr0 += 4;
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
                outptr[2] = sum2;
                outptr[3] = sum3;
            }

            outptr += 4;
        }

        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                sum0 = 0.f;
                sum1 = 0.f;

                if (pCi)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum0 = pCi[0];
                        sum1 = pCi[0];
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        sum0 = pCi[0];
                        sum1 = pCi[1];
                        pCi += 2;
                    }
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const float a0 = pA[0];
                sum0 += a0 * pB[0];
                sum1 += a0 * pB[1];
                pA += 1;
                pB += 2;
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0[1] = sum1;
                outptr0 += 2;
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }

        for (; jj < max_jj; jj += 1)
        {
            float sum0;

            if (k == 0)
            {
                sum0 = 0.f;

                if (pCi)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                        sum0 = pCi[0];
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        sum0 = pCi[0];
                        pCi += 1;
                    }
                }
            }
            else
            {
                sum0 = outptr[0];
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                sum0 += pA[0] * pB[0];
                pA += 1;
                pB += 1;
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0 += 1;
            }
            else
            {
                outptr[0] = sum0;
            }

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void get_optimal_tile_mnk(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrtf((float)l2_cache_size / 3 / sizeof(float));

#if __loongarch_sx
    TILE_M = std::max(4, tile_size / 4 * 4);
    TILE_N = std::max(4, tile_size / 4 * 4);
    TILE_K = std::max(4, tile_size / 4 * 4);
#else
    TILE_M = std::max(2, tile_size / 2 * 2);
    TILE_N = std::max(4, tile_size / 4 * 4);
    TILE_K = std::max(2, tile_size / 2 * 2);
#endif

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __loongarch_sx
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(float) / TILE_K);
#if __loongarch_sx
            TILE_M = std::max(4, tile_size / 4 * 4);
            TILE_N = std::max(4, tile_size / 4 * 4);
#else
            TILE_M = std::max(2, tile_size / 2 * 2);
            TILE_N = std::max(4, tile_size / 4 * 4);
#endif
        }
    }

    TILE_M *= std::min(nT, get_physical_cpu_count());

    if (M > 0)
    {
        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __loongarch_sx
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
    }

    if (nT > 1)
    {
#if __loongarch_sx
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
    }

    if (constant_TILE_M > 0)
    {
#if __loongarch_sx
        TILE_M = (constant_TILE_M + 3) / 4 * 4;
#else
        TILE_M = (constant_TILE_M + 1) / 2 * 2;
#endif
    }
    if (constant_TILE_N > 0)
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
    if (constant_TILE_K > 0)
    {
#if __loongarch_sx
        TILE_K = (constant_TILE_K + 3) / 4 * 4;
#else
        TILE_K = (constant_TILE_K + 1) / 2 * 2;
#endif
    }
}

static int gemm_loongarch(const Mat& A, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int transA, int transB, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
    const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    int TILE_M;
    int TILE_N;
    int TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat ATX(TILE_K * TILE_M, nn_K, nT, 4u, opt.workspace_allocator);
    if (ATX.empty())
        return -100;
    Mat BT(TILE_K * TILE_N, nn_K, nn_N, 4u, opt.workspace_allocator);
    if (BT.empty())
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
            pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
        else
            transpose_pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
    }

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

#if __loongarch_sx
    const bool use_8row_kernel = use_8row_packed_kernel(top_blob.elempack);
#endif

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;
        const int M_local = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
        const int K_local = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;
        const int max_ii = std::min(M_local - i, TILE_M);

        Mat topT_tile;
        if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
            topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            if (broadcast_type_C == 3)
            {
#if __loongarch_sx
                if (use_8row_kernel)
                    pack_A_tile_8row(C, topT_tile, i, max_ii, j, max_jj);
                else
#endif
                    pack_A_tile(C, topT_tile, i, max_ii, j, max_jj);
            }

            const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

            for (int k = 0; k < K_local; k += TILE_K)
            {
                const int max_kk = std::min(K_local - k, TILE_K);

                Mat AT_tile = ATX.channel(get_omp_thread_num()).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                if (j == 0)
                {
                    if (transA)
                    {
#if __loongarch_sx
                        if (use_8row_kernel)
                            transpose_pack_A_tile_8row(A, AT_tile, i, max_ii, k, max_kk);
                        else
#endif
                            transpose_pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
                    }
                    else
                    {
#if __loongarch_sx
                        if (use_8row_kernel)
                            pack_A_tile_8row(A, AT_tile, i, max_ii, k, max_kk);
                        else
#endif
                            pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
                    }
                }

                const bool k_end = !output_transpose && k + TILE_K >= K_local;
                gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
                transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

static int gemm_AT_loongarch(const Mat& AT, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int K, int transB, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    int TILE_M;
    int TILE_N;
    int TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat BT(TILE_K * TILE_N, nn_K, nn_N, 4u, opt.workspace_allocator);
    if (BT.empty())
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
            pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
        else
            transpose_pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
    }

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

#if __loongarch_sx
    const bool use_8row_kernel = use_8row_packed_kernel(top_blob.elempack);
#endif

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;
        const int max_ii = std::min(M - i, TILE_M);

        Mat topT_tile;
        if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
            topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            if (broadcast_type_C == 3)
            {
#if __loongarch_sx
                if (use_8row_kernel)
                    pack_A_tile_8row(C, topT_tile, i, max_ii, j, max_jj);
                else
#endif
                    pack_A_tile(C, topT_tile, i, max_ii, j, max_jj);
            }

            const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min(K - k, TILE_K);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                const bool k_end = !output_transpose && k + TILE_K >= K;
                gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
                transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

static int gemm_BT_loongarch(const Mat& A, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int N, int K, int transA, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;

    int TILE_M;
    int TILE_N;
    int TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat ATX(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, nT, 4u, opt.workspace_allocator);
    if (ATX.empty())
        return -100;

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

#if __loongarch_sx
    const bool use_8row_kernel = use_8row_packed_kernel(top_blob.elempack);
#endif

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;
        const int M_local = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
        const int K_local = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;
        const int max_ii = std::min(M_local - i, TILE_M);

        Mat topT_tile;
        if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
            topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            if (broadcast_type_C == 3)
            {
#if __loongarch_sx
                if (use_8row_kernel)
                    pack_A_tile_8row(C, topT_tile, i, max_ii, j, max_jj);
                else
#endif
                    pack_A_tile(C, topT_tile, i, max_ii, j, max_jj);
            }

            const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

            for (int k = 0; k < K_local; k += TILE_K)
            {
                const int max_kk = std::min(K_local - k, TILE_K);

                Mat AT_tile = ATX.channel(get_omp_thread_num()).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                if (j == 0)
                {
                    if (transA)
                    {
#if __loongarch_sx
                        if (use_8row_kernel)
                            transpose_pack_A_tile_8row(A, AT_tile, i, max_ii, k, max_kk);
                        else
#endif
                            transpose_pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
                    }
                    else
                    {
#if __loongarch_sx
                        if (use_8row_kernel)
                            pack_A_tile_8row(A, AT_tile, i, max_ii, k, max_kk);
                        else
#endif
                            pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
                    }
                }

                const bool k_end = !output_transpose && k + TILE_K >= K_local;
                gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
                transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

static int gemm_AT_BT_loongarch(const Mat& AT, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int N, int K, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    int TILE_M;
    int TILE_N;
    int TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

#if __loongarch_sx
    const bool use_8row_kernel = use_8row_packed_kernel(top_blob.elempack);
#endif

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;
        const int max_ii = std::min(M - i, TILE_M);

        Mat topT_tile;
        if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
            topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            if (broadcast_type_C == 3)
            {
#if __loongarch_sx
                if (use_8row_kernel)
                    pack_A_tile_8row(C, topT_tile, i, max_ii, j, max_jj);
                else
#endif
                    pack_A_tile(C, topT_tile, i, max_ii, j, max_jj);
            }

            const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min(K - k, TILE_K);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                const bool k_end = !output_transpose && k + TILE_K >= K;
                gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
                transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

int Gemm_loongarch::create_pipeline(const Option& opt)
{
    AT_data.release();
    BT_data.release();
    CT_data.release();
    nT = 0;

    if (int8_scale_term)
    {
        support_packing = false;
        support_bf16_storage = false;
        return 0;
    }

#if NCNN_BF16
    if (opt.use_bf16_storage)
        return create_pipeline_bf16s(opt);
#endif

    if (constantA)
    {
        const int M = constantM;
        const int K = constantK;
#if __loongarch_sx
        const bool use_8row_kernel = use_8row_packed_kernel(output_elempack);
#endif

        int TILE_M;
        int TILE_N;
        int TILE_K;
        get_optimal_tile_mnk(M, 0, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, opt.num_threads);

        const int nn_M = (M + TILE_M - 1) / TILE_M;
        const int nn_K = (K + TILE_K - 1) / TILE_K;

        AT_data.create(TILE_K * TILE_M, nn_K, nn_M, 4u, (Allocator*)0);
        if (AT_data.empty())
            return -100;

        const int nn_MK = nn_M * nn_K;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ppik = 0; ppik < nn_MK; ppik++)
        {
            const int ppi = ppik / nn_K;
            const int ppk = ppik % nn_K;

            const int i = ppi * TILE_M;
            const int k = ppk * TILE_K;

            const int max_ii = std::min(M - i, TILE_M);
            const int max_kk = std::min(K - k, TILE_K);

            Mat AT_tile = AT_data.channel(ppi).row_range(ppk, 1);

            if (transA)
            {
#if __loongarch_sx
                if (use_8row_kernel)
                    transpose_pack_A_tile_8row(A_data, AT_tile, i, max_ii, k, max_kk);
                else
#endif
                    transpose_pack_A_tile(A_data, AT_tile, i, max_ii, k, max_kk);
            }
            else
            {
#if __loongarch_sx
                if (use_8row_kernel)
                    pack_A_tile_8row(A_data, AT_tile, i, max_ii, k, max_kk);
                else
#endif
                    pack_A_tile(A_data, AT_tile, i, max_ii, k, max_kk);
            }
        }
    }

    if (constantB)
    {
        const int N = constantN;
        const int K = constantK;

        int TILE_M;
        int TILE_N;
        int TILE_K;
        get_optimal_tile_mnk(0, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, opt.num_threads);

        const int nn_N = (N + TILE_N - 1) / TILE_N;
        const int nn_K = (K + TILE_K - 1) / TILE_K;

        BT_data.create(TILE_K * TILE_N, nn_K, nn_N, 4u, (Allocator*)0);
        if (BT_data.empty())
            return -100;

        const int nn_NK = nn_N * nn_K;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min(N - j, TILE_N);
            const int max_kk = std::min(K - k, TILE_K);

            Mat BT_tile = BT_data.channel(ppj).row_range(ppk, 1);

            if (transB)
                pack_B_tile(B_data, BT_tile, j, max_jj, k, max_kk);
            else
                transpose_pack_B_tile(B_data, BT_tile, j, max_jj, k, max_kk);
        }
    }

    if (constantC && constant_broadcast_type_C != -1)
    {
        CT_data = C_data;

#if __loongarch_sx
        if (constant_broadcast_type_C == 3 && opt.use_packing_layout)
        {
#if __loongarch_asx
            int C_elempack = constantM % 8 == 0 ? 8 : constantM % 4 == 0 ? 4 : 1;
#else
            int C_elempack = constantM % 4 == 0 ? 4 : 1;
#endif
            convert_packing(C_data, CT_data, C_elempack, opt);
            if (CT_data.empty())
                return -100;
        }
#endif // __loongarch_sx

        // pre-multiply C with beta
        if (beta != 1.f)
        {
            Mat C2;
            C2.create_like(CT_data);
            if (C2.empty())
                return -100;

            const int size = CT_data.total() * CT_data.elempack;
            for (int i = 0; i < size; i++)
            {
                C2[i] = CT_data[i] * beta;
            }

            CT_data = C2;
        }
    }

    if (constantA || constantB || constantC)
    {
        nT = opt.num_threads;
    }

    return 0;
}

int Gemm_loongarch::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        return Gemm::forward_int8(bottom_blobs, top_blobs, opt);
    }
#endif

    const Mat& bottom_blob = bottom_blobs.empty() ? AT_data : bottom_blobs[0];
    int elembits = bottom_blob.elembits();

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s(bottom_blobs, top_blobs, opt);
#endif

    int M;
    int N;
    if (constantA && constantB)
    {
        M = constantM;
        N = constantN;
    }
    else if (constantA)
    {
        const Mat& B = bottom_blobs[0];
        M = constantM;
        N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;
    }
    else if (constantB)
    {
        const Mat& A = bottom_blobs[0];
        M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
        N = constantN;
    }
    else
    {
        const Mat& A = bottom_blobs[0];
        const Mat& B = bottom_blobs[1];
        M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
        N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;
    }

    Mat C;
    int broadcast_type_C = 0;
    if (constantC)
    {
        C = CT_data;
        broadcast_type_C = constant_broadcast_type_C;
    }
    else
    {
        if (constantA && constantB)
        {
            C = bottom_blobs.size() == 1 ? bottom_blobs[0] : Mat();
        }
        else if (constantA)
        {
            C = bottom_blobs.size() == 2 ? bottom_blobs[1] : Mat();
        }
        else if (constantB)
        {
            C = bottom_blobs.size() == 2 ? bottom_blobs[1] : Mat();
        }
        else
        {
            C = bottom_blobs.size() == 3 ? bottom_blobs[2] : Mat();
        }

        if (!C.empty())
        {
            if (C.dims == 1 && C.w == 1)
            {
                // scalar
                broadcast_type_C = 0;
            }
            if (C.dims == 1 && C.w * C.elempack == M)
            {
                // M
                // auto broadcast from h to w is the ncnn-style convention
                broadcast_type_C = 1;
            }
            if (C.dims == 1 && C.w * C.elempack == N)
            {
                // N
                broadcast_type_C = 4;
            }
            if (C.dims == 2 && C.w == 1 && C.h * C.elempack == M)
            {
                // Mx1
                broadcast_type_C = 2;
            }
            if (C.dims == 2 && C.w == N && C.h * C.elempack == M)
            {
                // MxN
                broadcast_type_C = 3;
            }
            if (C.dims == 2 && C.w == N && C.h * C.elempack == 1)
            {
                // 1xN
                broadcast_type_C = 4;
            }

            // pre-multiply C with beta
            if (beta != 1.f)
            {
                Mat C2;
                C2.create_like(C, opt.workspace_allocator);
                if (C2.empty())
                    return -100;

                const int size = C.total() * C.elempack;
                for (int i = 0; i < size; i++)
                {
                    C2[i] = C[i] * beta;
                }

                C = C2;
            }
        }
    }

    // HACK workaround broadcast_type_C == 3
    // tiled kernel does not handle this combination correctly
    // add C post-hoc instead
    Mat C_matrix_post;
    if (broadcast_type_C == 3)
    {
        if (C.elempack != 1)
        {
            // unpack C to elempack 1 for add_matrix_C_fp32
            Mat C_unpacked;
            convert_packing(C, C_unpacked, 1, opt);
            if (C_unpacked.empty())
                return -100;
            C_matrix_post = C_unpacked;
        }
        else
        {
            C_matrix_post = C;
        }
        C = Mat();
        broadcast_type_C = 0;
    }

    int out_elempack = 1;
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
        int outh = output_transpose ? N : M;
#if __loongarch_asx
        out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
        out_elempack = outh % 4 == 0 ? 4 : 1;
#endif
    }
#endif
    if (output_elempack)
        out_elempack = output_elempack;
    size_t out_elemsize = 4u * out_elempack;

    Mat& top_blob = top_blobs[0];
    if (output_transpose)
    {
        if (output_N1M)
            top_blob.create(M, 1, N / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        else
            top_blob.create(M, N / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    }
    else
    {
        if (output_N1M)
            top_blob.create(N, 1, M / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        else
            top_blob.create(N, M / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    int _nT = nT ? nT : opt.num_threads;
    if (nT != 0 && opt.num_threads != nT)
    {
        // force num_threads the same as in create_pipeline
        // so we could use pre-packed A/B from the same tile config
        NCNN_LOGE("opt.num_threads %d changed, gemm will use load-time value %d", opt.num_threads, nT);
    }

    int ret = 0;
    if (constantA && constantB)
    {
        ret = gemm_AT_BT_loongarch(AT_data, BT_data, C, top_blob, broadcast_type_C, constantM, constantN, constantK, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantA)
    {
        const Mat& B = bottom_blobs[0];
        ret = gemm_AT_loongarch(AT_data, B, C, top_blob, broadcast_type_C, constantM, constantK, transB, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantB)
    {
        const Mat& A = bottom_blobs[0];
        ret = gemm_BT_loongarch(A, BT_data, C, top_blob, broadcast_type_C, constantN, constantK, transA, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else
    {
        const Mat& A = bottom_blobs[0];
        const Mat& B = bottom_blobs[1];
        ret = gemm_loongarch(A, B, C, top_blob, broadcast_type_C, transA, transB, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    if (ret != 0)
        return ret;

    // apply C_matrix_post workaround
    if (!C_matrix_post.empty())
    {
        add_matrix_C_fp32(C_matrix_post, top_blob, output_transpose);
    }

    // multiply top_blob with alpha
    if (alpha != 1.f)
    {
        const int size = top_blob.total() * out_elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < size; i++)
        {
            top_blob[i] *= alpha;
        }
    }

    return 0;
}

#if NCNN_BF16
int Gemm_loongarch::create_pipeline_bf16s(const Option& opt)
{
    if (constantA)
    {
        const int M = constantM;
        const int K = constantK;

        int TILE_M, TILE_N, TILE_K;
        get_optimal_tile_mnk_bf16(M, 0, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, opt.num_threads);

        const int nn_M = (M + TILE_M - 1) / TILE_M;
        const int nn_K = (K + TILE_K - 1) / TILE_K;

        // cast A_data fp32 to bf16
        Mat A_data_bf16;
        cast_float32_to_bfloat16(A_data, A_data_bf16, opt);

        AT_data.create(TILE_K * TILE_M, nn_K, nn_M, 2u, (Allocator*)0);
        if (AT_data.empty())
            return -100;

        const int nn_MK = nn_M * nn_K;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ppik = 0; ppik < nn_MK; ppik++)
        {
            const int ppi = ppik / nn_K;
            const int ppk = ppik % nn_K;

            const int i = ppi * TILE_M;
            const int k = ppk * TILE_K;

            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT_data.channel(i / TILE_M).row_range(k / TILE_K, 1);

            if (transA)
            {
                transpose_pack_A_tile_bf16(A_data_bf16, AT_tile, i, max_ii, k, max_kk);
            }
            else
            {
                pack_A_tile_bf16(A_data_bf16, AT_tile, i, max_ii, k, max_kk);
            }
        }
    }

    if (constantB)
    {
        const int N = constantN;
        const int K = constantK;

        int TILE_M, TILE_N, TILE_K;
        get_optimal_tile_mnk_bf16(0, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, opt.num_threads);

        const int nn_N = (N + TILE_N - 1) / TILE_N;
        const int nn_K = (K + TILE_K - 1) / TILE_K;

        // cast B_data fp32 to bf16
        Mat B_data_bf16;
        cast_float32_to_bfloat16(B_data, B_data_bf16, opt);

        BT_data.create(TILE_K * TILE_N, nn_K, nn_N, 2u, (Allocator*)0);
        if (BT_data.empty())
            return -100;

        const int nn_NK = nn_N * nn_K;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            Mat BT_tile = BT_data.channel(j / TILE_N).row_range(k / TILE_K, 1);

            if (transB)
            {
                pack_B_tile_bf16(B_data_bf16, BT_tile, j, max_jj, k, max_kk);
            }
            else
            {
                transpose_pack_B_tile_bf16(B_data_bf16, BT_tile, j, max_jj, k, max_kk);
            }
        }
    }

    if (constantC && constant_broadcast_type_C != -1)
    {
        CT_data = C_data;

#if __loongarch_sx
        if (constant_broadcast_type_C == 3 && opt.use_packing_layout)
        {
#if __loongarch_asx
            int C_elempack = constantM % 8 == 0 ? 8 : constantM % 4 == 0 ? 4 : 1;
#else
            int C_elempack = constantM % 4 == 0 ? 4 : 1;
#endif
            convert_packing(C_data, CT_data, C_elempack, opt);
            if (CT_data.empty())
                return -100;
        }
#endif // __loongarch_sx
    }

    if (constantA || constantB || constantC)
    {
        nT = opt.num_threads;
    }

    return 0;
}

static int gemm_AT_BT_loongarch_bf16s(const Mat& AT, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int N, int K, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, int output_elemtype, const Option& opt)
{
    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_bf16(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_fp32_to_bf16(topT_tile, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta, output_transpose, output_elemtype);
        }
    }

    return 0;
}

static int gemm_AT_loongarch_bf16s(const Mat& AT, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int K, int transB, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, int output_elemtype, const Option& opt)
{
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_bf16(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat BT(TILE_K * TILE_N, nn_K, nn_N, 2u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;
    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min((N - j), TILE_N);
        const int max_kk = std::min((K - k), TILE_K);

        Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

        if (transB)
        {
            pack_B_tile_bf16(B, BT_tile, j, max_jj, k, max_kk);
        }
        else
        {
            transpose_pack_B_tile_bf16(B, BT_tile, j, max_jj, k, max_kk);
        }
    }

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                // AT is pre-packed bf16
                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_fp32_to_bf16(topT_tile, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta, output_transpose, output_elemtype);
        }
    }

    return 0;
}

static int gemm_BT_loongarch_bf16s(const Mat& A, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int N, int K, int transA, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, int output_elemtype, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_bf16(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    Mat ATX(TILE_K * TILE_M, nn_K, nT, 2u, opt.workspace_allocator);
    if (ATX.empty())
        return -100;

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;

        const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
        const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;

        const int max_ii = std::min((M - i), TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                Mat AT_tile = ATX.channel(get_omp_thread_num()).row_range(k / TILE_K, 1);

                // BT is pre-packed bf16
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                if (j == 0)
                {
                    if (transA)
                    {
                        transpose_pack_A_tile_bf16(A, AT_tile, i, max_ii, k, max_kk);
                    }
                    else
                    {
                        pack_A_tile_bf16(A, AT_tile, i, max_ii, k, max_kk);
                    }
                }

                gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_fp32_to_bf16(topT_tile, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta, output_transpose, output_elemtype);
        }
    }

    return 0;
}

static int gemm_loongarch_bf16s(const Mat& A, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int transA, int transB, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, int output_elemtype, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
    const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_bf16(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat BT(TILE_K * TILE_N, nn_K, nn_N, 2u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;
    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min((N - j), TILE_N);
        const int max_kk = std::min((K - k), TILE_K);

        Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

        if (transB)
        {
            pack_B_tile_bf16(B, BT_tile, j, max_jj, k, max_kk);
        }
        else
        {
            transpose_pack_B_tile_bf16(B, BT_tile, j, max_jj, k, max_kk);
        }
    }

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    Mat ATX(TILE_K * TILE_M, nn_K, nT, 2u, opt.workspace_allocator);
    if (ATX.empty())
        return -100;

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;

        const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
        const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;

        const int max_ii = std::min((M - i), TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                Mat AT_tile = ATX.channel(get_omp_thread_num()).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                if (j == 0)
                {
                    if (transA)
                    {
                        transpose_pack_A_tile_bf16(A, AT_tile, i, max_ii, k, max_kk);
                    }
                    else
                    {
                        pack_A_tile_bf16(A, AT_tile, i, max_ii, k, max_kk);
                    }
                }

                gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_fp32_to_bf16(topT_tile, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta, output_transpose, output_elemtype);
        }
    }

    return 0;
}

int Gemm_loongarch::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& A_input = constantA ? A_data : bottom_blobs[0];
    const Mat& B_input = constantB ? B_data : constantA ? bottom_blobs[0] : bottom_blobs[1];

    int M;
    int N;
    if (constantA && constantB)
    {
        M = constantM;
        N = constantN;
    }
    else if (constantA)
    {
        const Mat& B = bottom_blobs[0];
        M = constantM;
        N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;
    }
    else if (constantB)
    {
        const Mat& A = bottom_blobs[0];
        M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
        N = constantN;
    }
    else
    {
        M = transA ? A_input.w : (A_input.dims == 3 ? A_input.c : A_input.h) * A_input.elempack;
        N = transB ? (B_input.dims == 3 ? B_input.c : B_input.h) * B_input.elempack : B_input.w;
    }

    Mat C;
    int broadcast_type_C = 0;
    if (constantC)
    {
        C = !CT_data.empty() ? CT_data : C_data;
        broadcast_type_C = constant_broadcast_type_C;

        if (!C.empty())
        {
            if (C.elembits() == 16)
            {
                Option opt_cast = opt;
                opt_cast.blob_allocator = opt.workspace_allocator;

                Mat C_fp32;
                cast_bfloat16_to_float32(C, C_fp32, opt_cast);
                if (C_fp32.empty())
                    return -100;
                C = C_fp32;
            }
        }
    }
    else
    {
        if (constantA && constantB)
            C = bottom_blobs.size() == 1 ? bottom_blobs[0] : Mat();
        else if (constantA || constantB)
            C = bottom_blobs.size() == 2 ? bottom_blobs[1] : Mat();
        else
            C = bottom_blobs.size() == 3 ? bottom_blobs[2] : Mat();

        if (!C.empty())
        {
            broadcast_type_C = resolve_broadcast_type_C(C, M, N);

            if (C.elembits() == 16)
            {
                Option opt_cast = opt;
                opt_cast.blob_allocator = opt.workspace_allocator;

                Mat C_fp32;
                cast_bfloat16_to_float32(C, C_fp32, opt_cast);
                if (C_fp32.empty())
                    return -100;
                C = C_fp32;
            }
        }
    }

    int out_elempack = 1;
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
        const int outh = output_transpose ? N : M;
        out_elempack =
#if __loongarch_asx
            outh % 8 == 0 ? 8 :
#endif
            outh % 4 == 0 ? 4 : 1;
    }
#endif
    if (output_elempack)
        out_elempack = output_elempack;

    size_t out_elemsize = (output_elemtype == 1 ? 4u : 2u) * out_elempack;

    Mat& top_blob = top_blobs[0];
    if (output_transpose)
    {
        if (output_N1M)
            top_blob.create(M, 1, N / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        else
            top_blob.create(M, N / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    }
    else
    {
        if (output_N1M)
            top_blob.create(N, 1, M / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        else
            top_blob.create(N, M / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    const int _nT = nT ? nT : opt.num_threads;
    if (nT != 0 && opt.num_threads != nT)
    {
        NCNN_LOGE("opt.num_threads %d changed, gemm will use load-time value %d", opt.num_threads, nT);
    }

    int ret = 0;
    if (constantA && constantB)
    {
        ret = gemm_AT_BT_loongarch_bf16s(AT_data, BT_data, C, top_blob, broadcast_type_C, constantM, constantN, constantK, output_transpose, alpha, beta, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, output_elemtype, opt);
    }
    else if (constantA)
    {
        ret = gemm_AT_loongarch_bf16s(AT_data, B_input, C, top_blob, broadcast_type_C, constantM, constantK, transB, output_transpose, alpha, beta, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, output_elemtype, opt);
    }
    else if (constantB)
    {
        ret = gemm_BT_loongarch_bf16s(A_input, BT_data, C, top_blob, broadcast_type_C, constantN, constantK, transA, output_transpose, alpha, beta, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, output_elemtype, opt);
    }
    else
    {
        ret = gemm_loongarch_bf16s(A_input, B_input, C, top_blob, broadcast_type_C, transA, transB, output_transpose, alpha, beta, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, output_elemtype, opt);
    }
    if (ret != 0)
        return ret;

    return 0;
}
#endif

} // namespace ncnn

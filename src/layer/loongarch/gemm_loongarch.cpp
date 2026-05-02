// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gemm_loongarch.h"

#include "loongarch_usability.h"

#include "cpu.h"

namespace ncnn {

#if NCNN_INT8
#include "gemm_int8.h"
#endif

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

static void pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    float* pp = AT;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 8;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __lasx_xvst(__lasx_xvld(p0, 0), pp, 0);
                pp += 8;
                p0 += 8;
            }
        }
#endif // __loongarch_asx
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
        if (elempack == 1)
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
        if (elempack == 1)
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
        // if (elempack == 1)
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
    }

    for (; ii < max_ii; ii += 1)
    {
        // if (elempack == 1)
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
}

static void transpose_pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    float* pp = AT;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = (__m256)__lasx_xvld(p0, 0);
                __m256 _r1 = (__m256)__lasx_xvld(p0 + 8, 0);
                __m256 _r2 = (__m256)__lasx_xvld(p0 + 16, 0);
                __m256 _r3 = (__m256)__lasx_xvld(p0 + 24, 0);
                __m256 _r4 = (__m256)__lasx_xvld(p0 + 32, 0);
                __m256 _r5 = (__m256)__lasx_xvld(p0 + 40, 0);
                __m256 _r6 = (__m256)__lasx_xvld(p0 + 48, 0);
                __m256 _r7 = (__m256)__lasx_xvld(p0 + 56, 0);
                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                __lasx_xvst(_r0, pp, 0);
                __lasx_xvst(_r1, pp + 8, 0);
                __lasx_xvst(_r2, pp + 16, 0);
                __lasx_xvst(_r3, pp + 24, 0);
                __lasx_xvst(_r4, pp + 32, 0);
                __lasx_xvst(_r5, pp + 40, 0);
                __lasx_xvst(_r6, pp + 48, 0);
                __lasx_xvst(_r7, pp + 56, 0);
                pp += 64;
                p0 += A_hstep * 8;
            }
        }
#endif // __loongarch_asx
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
        }
        if (elempack == 1)
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
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = (__m256)__lasx_xvld(p0, 0);
                __m256 _r1 = (__m256)__lasx_xvld(p0 + 8, 0);
                __m256 _r2 = (__m256)__lasx_xvld(p0 + 16, 0);
                __m256 _r3 = (__m256)__lasx_xvld(p0 + 24, 0);
                transpose8x4_ps(_r0, _r1, _r2, _r3);
                __lasx_xvst(_r0, pp, 0);
                __lasx_xvst(_r1, pp + 8, 0);
                __lasx_xvst(_r2, pp + 16, 0);
                __lasx_xvst(_r3, pp + 24, 0);
                pp += 32;
                p0 += A_hstep * 8;
            }
        }
#endif // __loongarch_asx
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
        }
        if (elempack == 1)
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
#if __loongarch_asx
        if (elempack == 8)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = (__m256)__lasx_xvld(p0, 0);
                __m256 _r1 = (__m256)__lasx_xvld(p0 + 8, 0);
                transpose8x2_ps(_r0, _r1);
                __lasx_xvst(_r0, pp, 0);
                __lasx_xvst(_r1, pp + 8, 0);
                pp += 16;
                p0 += A_hstep * 8;
            }
        }
#endif // __loongarch_asx
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_r1, (__m128i)_r0);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_r1, (__m128i)_r0);
                __lsx_vst((__m128i)_tmp0, pp, 0);
                __lsx_vst((__m128i)_tmp1, pp + 4, 0);
                pp += 8;
                p0 += A_hstep * 4;
            }
        }
#endif // __loongarch_sx
        if (elempack == 1)
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
#if __loongarch_asx
        if (elempack == 8)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __lasx_xvst(__lasx_xvld(p0, 0), pp, 0);
                pp += 8;
                p0 += A_hstep * 8;
            }
        }
#endif // __loongarch_asx
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
        }
#endif // __loongarch_sx
        if (elempack == 1)
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

static void pack_B_tile(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    float* pp = BT;

    int jj = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; jj + 15 < max_jj; jj += 16)
    {
        if (elempack == 8)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 8;
            const float* p1 = (const float*)B + (j + jj + 8) * B_hstep + k * 8;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __lasx_xvst(__lasx_xvld(p0, 0), pp, 0);
                __lasx_xvst(__lasx_xvld(p1, 0), pp + 8, 0);
                pp += 16;
                p0 += 8;
                p1 += 8;
            }
        }
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;
            const float* p1 = (const float*)B + (j + jj + 4) * B_hstep + k * 4;
            const float* p2 = (const float*)B + (j + jj + 8) * B_hstep + k * 4;
            const float* p3 = (const float*)B + (j + jj + 12) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _r0 = __lasx_concat_128_s((__m128)__lsx_vld(p0, 0), (__m128)__lsx_vld(p1, 0));
                __m256 _r1 = __lasx_concat_128_s((__m128)__lsx_vld(p2, 0), (__m128)__lsx_vld(p3, 0));
                __lasx_xvst(_r0, pp, 0);
                __lasx_xvst(_r1, pp + 8, 0);
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
            const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
            const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;
            const float* p4 = (const float*)B + (j + jj + 4) * B_hstep + k;
            const float* p5 = (const float*)B + (j + jj + 5) * B_hstep + k;
            const float* p6 = (const float*)B + (j + jj + 6) * B_hstep + k;
            const float* p7 = (const float*)B + (j + jj + 7) * B_hstep + k;
            const float* p8 = (const float*)B + (j + jj + 8) * B_hstep + k;
            const float* p9 = (const float*)B + (j + jj + 9) * B_hstep + k;
            const float* pa = (const float*)B + (j + jj + 10) * B_hstep + k;
            const float* pb = (const float*)B + (j + jj + 11) * B_hstep + k;
            const float* pc = (const float*)B + (j + jj + 12) * B_hstep + k;
            const float* pd = (const float*)B + (j + jj + 13) * B_hstep + k;
            const float* pe = (const float*)B + (j + jj + 14) * B_hstep + k;
            const float* pf = (const float*)B + (j + jj + 15) * B_hstep + k;

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
                pp[12] = pc[0];
                pp[13] = pd[0];
                pp[14] = pe[0];
                pp[15] = pf[0];
                pp += 16;
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
                pc++;
                pd++;
                pe++;
                pf++;
            }
        }
    }
#endif // __loongarch_asx
    for (; jj + 7 < max_jj; jj += 8)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 8;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __lasx_xvst(__lasx_xvld(p0, 0), pp, 0);
                pp += 8;
                p0 += 8;
            }
        }
#endif // __loongarch_asx
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;
            const float* p1 = (const float*)B + (j + jj + 4) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __lsx_vst(__lsx_vld(p0, 0), pp, 0);
                __lsx_vst(__lsx_vld(p1, 0), pp + 4, 0);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
        }
        if (elempack == 1)
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
#endif // __loongarch_sx
    for (; jj + 3 < max_jj; jj += 4)
    {
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
#endif // __loongarch_sx
        if (elempack == 1)
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
        // if (elempack == 1)
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
    }

    for (; jj < max_jj; jj += 1)
    {
        // if (elempack == 1)
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
}

static void transpose_pack_B_tile(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    float* pp = BT;

    int jj = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; jj + 15 < max_jj; jj += 16)
    {
        if (elempack == 8)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = (__m256)__lasx_xvld(p0, 0);
                __m256 _r1 = (__m256)__lasx_xvld(p0 + 8, 0);
                __m256 _r2 = (__m256)__lasx_xvld(p0 + 16, 0);
                __m256 _r3 = (__m256)__lasx_xvld(p0 + 24, 0);
                __m256 _r4 = (__m256)__lasx_xvld(p0 + 32, 0);
                __m256 _r5 = (__m256)__lasx_xvld(p0 + 40, 0);
                __m256 _r6 = (__m256)__lasx_xvld(p0 + 48, 0);
                __m256 _r7 = (__m256)__lasx_xvld(p0 + 56, 0);
                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                __m256 _s0 = (__m256)__lasx_xvld(p0 + 64, 0);
                __m256 _s1 = (__m256)__lasx_xvld(p0 + 72, 0);
                __m256 _s2 = (__m256)__lasx_xvld(p0 + 80, 0);
                __m256 _s3 = (__m256)__lasx_xvld(p0 + 88, 0);
                __m256 _s4 = (__m256)__lasx_xvld(p0 + 96, 0);
                __m256 _s5 = (__m256)__lasx_xvld(p0 + 104, 0);
                __m256 _s6 = (__m256)__lasx_xvld(p0 + 112, 0);
                __m256 _s7 = (__m256)__lasx_xvld(p0 + 120, 0);
                transpose8x8_ps(_s0, _s1, _s2, _s3, _s4, _s5, _s6, _s7);

                __lasx_xvst(_r0, pp, 0);
                __lasx_xvst(_s0, pp + 8, 0);
                __lasx_xvst(_r1, pp + 16, 0);
                __lasx_xvst(_s1, pp + 24, 0);
                __lasx_xvst(_r2, pp + 32, 0);
                __lasx_xvst(_s2, pp + 40, 0);
                __lasx_xvst(_r3, pp + 48, 0);
                __lasx_xvst(_s3, pp + 56, 0);
                __lasx_xvst(_r4, pp + 64, 0);
                __lasx_xvst(_s4, pp + 72, 0);
                __lasx_xvst(_r5, pp + 80, 0);
                __lasx_xvst(_s5, pp + 88, 0);
                __lasx_xvst(_r6, pp + 96, 0);
                __lasx_xvst(_s6, pp + 104, 0);
                __lasx_xvst(_r7, pp + 112, 0);
                __lasx_xvst(_s7, pp + 120, 0);
                pp += 128;
                p0 += B_hstep * 8;
            }
        }
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
                __m128 _r4 = (__m128)__lsx_vld(p0 + 16, 0);
                __m128 _r5 = (__m128)__lsx_vld(p0 + 20, 0);
                __m128 _r6 = (__m128)__lsx_vld(p0 + 24, 0);
                __m128 _r7 = (__m128)__lsx_vld(p0 + 28, 0);
                transpose4x4_ps(_r4, _r5, _r6, _r7);
                __m128 _s0 = (__m128)__lsx_vld(p0 + 32, 0);
                __m128 _s1 = (__m128)__lsx_vld(p0 + 36, 0);
                __m128 _s2 = (__m128)__lsx_vld(p0 + 40, 0);
                __m128 _s3 = (__m128)__lsx_vld(p0 + 44, 0);
                transpose4x4_ps(_s0, _s1, _s2, _s3);
                __m128 _s4 = (__m128)__lsx_vld(p0 + 48, 0);
                __m128 _s5 = (__m128)__lsx_vld(p0 + 52, 0);
                __m128 _s6 = (__m128)__lsx_vld(p0 + 56, 0);
                __m128 _s7 = (__m128)__lsx_vld(p0 + 60, 0);
                transpose4x4_ps(_s4, _s5, _s6, _s7);
                __lasx_xvst(__lasx_concat_128_s(_r0, _r4), pp, 0);
                __lasx_xvst(__lasx_concat_128_s(_s0, _s4), pp + 8, 0);
                __lasx_xvst(__lasx_concat_128_s(_r1, _r5), pp + 16, 0);
                __lasx_xvst(__lasx_concat_128_s(_s1, _s5), pp + 24, 0);
                __lasx_xvst(__lasx_concat_128_s(_r2, _r6), pp + 32, 0);
                __lasx_xvst(__lasx_concat_128_s(_s2, _s6), pp + 40, 0);
                __lasx_xvst(__lasx_concat_128_s(_r3, _r7), pp + 48, 0);
                __lasx_xvst(__lasx_concat_128_s(_s3, _s7), pp + 56, 0);
                pp += 64;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            for (int kk = 0; kk < max_kk; kk++)
            {
                __lasx_xvst(__lasx_xvld(p0, 0), pp, 0);
                __lasx_xvst(__lasx_xvld(p0 + 8, 0), pp + 8, 0);
                pp += 16;
                p0 += B_hstep;
            }
        }
    }
#endif // __loongarch_asx
    for (; jj + 7 < max_jj; jj += 8)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = (__m256)__lasx_xvld(p0, 0);
                __m256 _r1 = (__m256)__lasx_xvld(p0 + 8, 0);
                __m256 _r2 = (__m256)__lasx_xvld(p0 + 16, 0);
                __m256 _r3 = (__m256)__lasx_xvld(p0 + 24, 0);
                __m256 _r4 = (__m256)__lasx_xvld(p0 + 32, 0);
                __m256 _r5 = (__m256)__lasx_xvld(p0 + 40, 0);
                __m256 _r6 = (__m256)__lasx_xvld(p0 + 48, 0);
                __m256 _r7 = (__m256)__lasx_xvld(p0 + 56, 0);
                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                __lasx_xvst(_r0, pp, 0);
                __lasx_xvst(_r1, pp + 8, 0);
                __lasx_xvst(_r2, pp + 16, 0);
                __lasx_xvst(_r3, pp + 24, 0);
                __lasx_xvst(_r4, pp + 32, 0);
                __lasx_xvst(_r5, pp + 40, 0);
                __lasx_xvst(_r6, pp + 48, 0);
                __lasx_xvst(_r7, pp + 56, 0);
                pp += 64;
                p0 += B_hstep * 8;
            }
        }
#endif // __loongarch_asx
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
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            for (int kk = 0; kk < max_kk; kk++)
            {
                __lsx_vst(__lsx_vld(p0, 0), pp, 0);
                __lsx_vst(__lsx_vld(p0 + 4, 0), pp + 4, 0);
                pp += 8;
                p0 += B_hstep;
            }
        }
    }
#endif // __loongarch_sx
    for (; jj + 3 < max_jj; jj += 4)
    {
#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = (__m256)__lasx_xvld(p0, 0);
                __m256 _r1 = (__m256)__lasx_xvld(p0 + 8, 0);
                __m256 _r2 = (__m256)__lasx_xvld(p0 + 16, 0);
                __m256 _r3 = (__m256)__lasx_xvld(p0 + 24, 0);
                transpose8x4_ps(_r0, _r1, _r2, _r3);
                __lasx_xvst(_r0, pp, 0);
                __lasx_xvst(_r1, pp + 8, 0);
                __lasx_xvst(_r2, pp + 16, 0);
                __lasx_xvst(_r3, pp + 24, 0);
                pp += 32;
                p0 += B_hstep * 8;
            }
        }
#endif // __loongarch_asx
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
        }
#endif // __loongarch_sx
        if (elempack == 1)
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
#if __loongarch_asx
        if (elempack == 8)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = (__m256)__lasx_xvld(p0, 0);
                __m256 _r1 = (__m256)__lasx_xvld(p0 + 8, 0);
                transpose8x2_ps(_r0, _r1);
                __lasx_xvst(_r0, pp, 0);
                __lasx_xvst(_r1, pp + 8, 0);
                pp += 16;
                p0 += B_hstep * 8;
            }
        }
#endif // __loongarch_asx
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_r1, (__m128i)_r0);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_r1, (__m128i)_r0);
                __lsx_vst((__m128i)_tmp0, pp, 0);
                __lsx_vst((__m128i)_tmp1, pp + 4, 0);
                pp += 8;
                p0 += B_hstep * 4;
            }
        }
#endif // __loongarch_sx
        if (elempack == 1)
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
#if __loongarch_asx
        if (elempack == 8)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __lasx_xvst(__lasx_xvld(p0, 0), pp, 0);
                pp += 8;
                p0 += B_hstep * 8;
            }
        }
#endif // __loongarch_asx
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
        }
#endif // __loongarch_sx
        if (elempack == 1)
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

static void unpack_output_tile(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta, int output_transpose, int output_elemtype)
{
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const float* pp = topT;

    const float* pC = C;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        unsigned short* p0;
        float* p0f;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
            p0f = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
            p0f = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        const float* pC0 = pC;
        if (pC0)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC0 = (const float*)C + (i + ii);
            }
            if (broadcast_type_C == 3)
            {
                pC += max_jj * 8;
            }
            if (broadcast_type_C == 4)
            {
                pC0 = (const float*)C + j;
            }
        }

        int jj = 0;
#if __loongarch_asx
        __m256 _valpha = (__m256)__lasx_xvreplfr2vr_s(alpha);

        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _sum0 = (__m256)__lasx_xvld(pp, 0);
            __m256 _sum1 = (__m256)__lasx_xvld(pp + 8, 0);
            __m256 _sum2 = (__m256)__lasx_xvld(pp + 16, 0);
            __m256 _sum3 = (__m256)__lasx_xvld(pp + 24, 0);
            __m256 _sum4 = (__m256)__lasx_xvld(pp + 32, 0);
            __m256 _sum5 = (__m256)__lasx_xvld(pp + 40, 0);
            __m256 _sum6 = (__m256)__lasx_xvld(pp + 48, 0);
            __m256 _sum7 = (__m256)__lasx_xvld(pp + 56, 0);
            __m256 _sum8 = (__m256)__lasx_xvld(pp + 64, 0);
            __m256 _sum9 = (__m256)__lasx_xvld(pp + 72, 0);
            __m256 _suma = (__m256)__lasx_xvld(pp + 80, 0);
            __m256 _sumb = (__m256)__lasx_xvld(pp + 88, 0);
            __m256 _sumc = (__m256)__lasx_xvld(pp + 96, 0);
            __m256 _sumd = (__m256)__lasx_xvld(pp + 104, 0);
            __m256 _sume = (__m256)__lasx_xvld(pp + 112, 0);
            __m256 _sumf = (__m256)__lasx_xvld(pp + 120, 0);
            pp += 128;

            // deshuffle from the shuffle-based 8x16 kernel
            _sum4 = (__m256)__lasx_xvpermi_q((__m256i)_sum4, (__m256i)_sum4, _LSX_SHUFFLE(0, 0, 0, 1));
            _sum5 = (__m256)__lasx_xvpermi_q((__m256i)_sum5, (__m256i)_sum5, _LSX_SHUFFLE(0, 0, 0, 1));
            _sum6 = (__m256)__lasx_xvpermi_q((__m256i)_sum6, (__m256i)_sum6, _LSX_SHUFFLE(0, 0, 0, 1));
            _sum7 = (__m256)__lasx_xvpermi_q((__m256i)_sum7, (__m256i)_sum7, _LSX_SHUFFLE(0, 0, 0, 1));
            _sumc = (__m256)__lasx_xvpermi_q((__m256i)_sumc, (__m256i)_sumc, _LSX_SHUFFLE(0, 0, 0, 1));
            _sumd = (__m256)__lasx_xvpermi_q((__m256i)_sumd, (__m256i)_sumd, _LSX_SHUFFLE(0, 0, 0, 1));
            _sume = (__m256)__lasx_xvpermi_q((__m256i)_sume, (__m256i)_sume, _LSX_SHUFFLE(0, 0, 0, 1));
            _sumf = (__m256)__lasx_xvpermi_q((__m256i)_sumf, (__m256i)_sumf, _LSX_SHUFFLE(0, 0, 0, 1));
            {
                __m256 _tmp0 = _sum0;
                __m256 _tmp1 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp2 = _sum2;
                __m256 _tmp3 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum3, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp4 = _sum4;
                __m256 _tmp5 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum5, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp6 = _sum6;
                __m256 _tmp7 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum7, _LSX_SHUFFLE(2, 1, 0, 3));

                _sum0 = (__m256)__lasx_xvilvl_w((__m256i)_tmp3, (__m256i)_tmp0);
                _sum1 = (__m256)__lasx_xvilvh_w((__m256i)_tmp3, (__m256i)_tmp0);
                _sum2 = (__m256)__lasx_xvilvl_w((__m256i)_tmp1, (__m256i)_tmp2);
                _sum3 = (__m256)__lasx_xvilvh_w((__m256i)_tmp1, (__m256i)_tmp2);
                _sum4 = (__m256)__lasx_xvilvl_w((__m256i)_tmp7, (__m256i)_tmp4);
                _sum5 = (__m256)__lasx_xvilvh_w((__m256i)_tmp7, (__m256i)_tmp4);
                _sum6 = (__m256)__lasx_xvilvl_w((__m256i)_tmp5, (__m256i)_tmp6);
                _sum7 = (__m256)__lasx_xvilvh_w((__m256i)_tmp5, (__m256i)_tmp6);

                _tmp0 = (__m256)__lasx_xvilvl_d((__m256i)_sum2, (__m256i)_sum0);
                _tmp1 = (__m256)__lasx_xvilvh_d((__m256i)_sum2, (__m256i)_sum0);
                _tmp2 = (__m256)__lasx_xvilvl_d((__m256i)_sum1, (__m256i)_sum3);
                _tmp3 = (__m256)__lasx_xvilvh_d((__m256i)_sum1, (__m256i)_sum3);
                _tmp4 = (__m256)__lasx_xvilvl_d((__m256i)_sum6, (__m256i)_sum4);
                _tmp5 = (__m256)__lasx_xvilvh_d((__m256i)_sum6, (__m256i)_sum4);
                _tmp6 = (__m256)__lasx_xvilvl_d((__m256i)_sum5, (__m256i)_sum7);
                _tmp7 = (__m256)__lasx_xvilvh_d((__m256i)_sum5, (__m256i)_sum7);

                _tmp1 = (__m256)__lasx_xvshuf4i_w((__m256i)_tmp1, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp3 = (__m256)__lasx_xvshuf4i_w((__m256i)_tmp3, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp5 = (__m256)__lasx_xvshuf4i_w((__m256i)_tmp5, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp7 = (__m256)__lasx_xvshuf4i_w((__m256i)_tmp7, _LSX_SHUFFLE(2, 1, 0, 3));

                _sum0 = (__m256)__lasx_xvpermi_q((__m256i)_tmp4, (__m256i)_tmp0, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum1 = (__m256)__lasx_xvpermi_q((__m256i)_tmp5, (__m256i)_tmp1, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum2 = (__m256)__lasx_xvpermi_q((__m256i)_tmp6, (__m256i)_tmp2, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum3 = (__m256)__lasx_xvpermi_q((__m256i)_tmp7, (__m256i)_tmp3, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum4 = (__m256)__lasx_xvpermi_q((__m256i)_tmp0, (__m256i)_tmp4, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum5 = (__m256)__lasx_xvpermi_q((__m256i)_tmp1, (__m256i)_tmp5, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum6 = (__m256)__lasx_xvpermi_q((__m256i)_tmp2, (__m256i)_tmp6, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum7 = (__m256)__lasx_xvpermi_q((__m256i)_tmp3, (__m256i)_tmp7, _LSX_SHUFFLE(0, 3, 0, 0));
            }
            {
                __m256 _tmp0 = _sum8;
                __m256 _tmp1 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum9, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp2 = _suma;
                __m256 _tmp3 = (__m256)__lasx_xvshuf4i_w((__m256i)_sumb, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp4 = _sumc;
                __m256 _tmp5 = (__m256)__lasx_xvshuf4i_w((__m256i)_sumd, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp6 = _sume;
                __m256 _tmp7 = (__m256)__lasx_xvshuf4i_w((__m256i)_sumf, _LSX_SHUFFLE(2, 1, 0, 3));

                _sum8 = (__m256)__lasx_xvilvl_w((__m256i)_tmp3, (__m256i)_tmp0);
                _sum9 = (__m256)__lasx_xvilvh_w((__m256i)_tmp3, (__m256i)_tmp0);
                _suma = (__m256)__lasx_xvilvl_w((__m256i)_tmp1, (__m256i)_tmp2);
                _sumb = (__m256)__lasx_xvilvh_w((__m256i)_tmp1, (__m256i)_tmp2);
                _sumc = (__m256)__lasx_xvilvl_w((__m256i)_tmp7, (__m256i)_tmp4);
                _sumd = (__m256)__lasx_xvilvh_w((__m256i)_tmp7, (__m256i)_tmp4);
                _sume = (__m256)__lasx_xvilvl_w((__m256i)_tmp5, (__m256i)_tmp6);
                _sumf = (__m256)__lasx_xvilvh_w((__m256i)_tmp5, (__m256i)_tmp6);

                _tmp0 = (__m256)__lasx_xvilvl_d((__m256i)_suma, (__m256i)_sum8);
                _tmp1 = (__m256)__lasx_xvilvh_d((__m256i)_suma, (__m256i)_sum8);
                _tmp2 = (__m256)__lasx_xvilvl_d((__m256i)_sum9, (__m256i)_sumb);
                _tmp3 = (__m256)__lasx_xvilvh_d((__m256i)_sum9, (__m256i)_sumb);
                _tmp4 = (__m256)__lasx_xvilvl_d((__m256i)_sume, (__m256i)_sumc);
                _tmp5 = (__m256)__lasx_xvilvh_d((__m256i)_sume, (__m256i)_sumc);
                _tmp6 = (__m256)__lasx_xvilvl_d((__m256i)_sumd, (__m256i)_sumf);
                _tmp7 = (__m256)__lasx_xvilvh_d((__m256i)_sumd, (__m256i)_sumf);

                _tmp1 = (__m256)__lasx_xvshuf4i_w((__m256i)_tmp1, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp3 = (__m256)__lasx_xvshuf4i_w((__m256i)_tmp3, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp5 = (__m256)__lasx_xvshuf4i_w((__m256i)_tmp5, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp7 = (__m256)__lasx_xvshuf4i_w((__m256i)_tmp7, _LSX_SHUFFLE(2, 1, 0, 3));

                _sum8 = (__m256)__lasx_xvpermi_q((__m256i)_tmp4, (__m256i)_tmp0, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum9 = (__m256)__lasx_xvpermi_q((__m256i)_tmp5, (__m256i)_tmp1, _LSX_SHUFFLE(0, 3, 0, 0));
                _suma = (__m256)__lasx_xvpermi_q((__m256i)_tmp6, (__m256i)_tmp2, _LSX_SHUFFLE(0, 3, 0, 0));
                _sumb = (__m256)__lasx_xvpermi_q((__m256i)_tmp7, (__m256i)_tmp3, _LSX_SHUFFLE(0, 3, 0, 0));
                _sumc = (__m256)__lasx_xvpermi_q((__m256i)_tmp0, (__m256i)_tmp4, _LSX_SHUFFLE(0, 3, 0, 0));
                _sumd = (__m256)__lasx_xvpermi_q((__m256i)_tmp1, (__m256i)_tmp5, _LSX_SHUFFLE(0, 3, 0, 0));
                _sume = (__m256)__lasx_xvpermi_q((__m256i)_tmp2, (__m256i)_tmp6, _LSX_SHUFFLE(0, 3, 0, 0));
                _sumf = (__m256)__lasx_xvpermi_q((__m256i)_tmp3, (__m256i)_tmp7, _LSX_SHUFFLE(0, 3, 0, 0));
            }

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
                    _sum8 = __lasx_xvfmadd_s(_c, _beta, _sum8);
                    _sum9 = __lasx_xvfmadd_s(_c, _beta, _sum9);
                    _suma = __lasx_xvfmadd_s(_c, _beta, _suma);
                    _sumb = __lasx_xvfmadd_s(_c, _beta, _sumb);
                    _sumc = __lasx_xvfmadd_s(_c, _beta, _sumc);
                    _sumd = __lasx_xvfmadd_s(_c, _beta, _sumd);
                    _sume = __lasx_xvfmadd_s(_c, _beta, _sume);
                    _sumf = __lasx_xvfmadd_s(_c, _beta, _sumf);
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
                    _sum8 = __lasx_xvfmadd_s(_c, _beta, _sum8);
                    _sum9 = __lasx_xvfmadd_s(_c, _beta, _sum9);
                    _suma = __lasx_xvfmadd_s(_c, _beta, _suma);
                    _sumb = __lasx_xvfmadd_s(_c, _beta, _sumb);
                    _sumc = __lasx_xvfmadd_s(_c, _beta, _sumc);
                    _sumd = __lasx_xvfmadd_s(_c, _beta, _sumd);
                    _sume = __lasx_xvfmadd_s(_c, _beta, _sume);
                    _sumf = __lasx_xvfmadd_s(_c, _beta, _sumf);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c0 = (__m256)__lasx_xvld(pC0, 0);
                    __m256 _c1 = (__m256)__lasx_xvld(pC0 + 8, 0);
                    __m256 _c2 = (__m256)__lasx_xvld(pC0 + 16, 0);
                    __m256 _c3 = (__m256)__lasx_xvld(pC0 + 24, 0);
                    __m256 _c4 = (__m256)__lasx_xvld(pC0 + 32, 0);
                    __m256 _c5 = (__m256)__lasx_xvld(pC0 + 40, 0);
                    __m256 _c6 = (__m256)__lasx_xvld(pC0 + 48, 0);
                    __m256 _c7 = (__m256)__lasx_xvld(pC0 + 56, 0);
                    __m256 _c8 = (__m256)__lasx_xvld(pC0 + 64, 0);
                    __m256 _c9 = (__m256)__lasx_xvld(pC0 + 72, 0);
                    __m256 _ca = (__m256)__lasx_xvld(pC0 + 80, 0);
                    __m256 _cb = (__m256)__lasx_xvld(pC0 + 88, 0);
                    __m256 _cc = (__m256)__lasx_xvld(pC0 + 96, 0);
                    __m256 _cd = (__m256)__lasx_xvld(pC0 + 104, 0);
                    __m256 _ce = (__m256)__lasx_xvld(pC0 + 112, 0);
                    __m256 _cf = (__m256)__lasx_xvld(pC0 + 120, 0);
                    _sum0 = __lasx_xvfmadd_s(_c0, _beta, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_c1, _beta, _sum1);
                    _sum2 = __lasx_xvfmadd_s(_c2, _beta, _sum2);
                    _sum3 = __lasx_xvfmadd_s(_c3, _beta, _sum3);
                    _sum4 = __lasx_xvfmadd_s(_c4, _beta, _sum4);
                    _sum5 = __lasx_xvfmadd_s(_c5, _beta, _sum5);
                    _sum6 = __lasx_xvfmadd_s(_c6, _beta, _sum6);
                    _sum7 = __lasx_xvfmadd_s(_c7, _beta, _sum7);
                    _sum8 = __lasx_xvfmadd_s(_c8, _beta, _sum8);
                    _sum9 = __lasx_xvfmadd_s(_c9, _beta, _sum9);
                    _suma = __lasx_xvfmadd_s(_ca, _beta, _suma);
                    _sumb = __lasx_xvfmadd_s(_cb, _beta, _sumb);
                    _sumc = __lasx_xvfmadd_s(_cc, _beta, _sumc);
                    _sumd = __lasx_xvfmadd_s(_cd, _beta, _sumd);
                    _sume = __lasx_xvfmadd_s(_ce, _beta, _sume);
                    _sumf = __lasx_xvfmadd_s(_cf, _beta, _sumf);
                    pC0 += 128;
                }
                if (broadcast_type_C == 4)
                {
                    _sum0 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pC0[0]), _beta, _sum0);
                    _sum1 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pC0[1]), _beta, _sum1);
                    _sum2 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pC0[2]), _beta, _sum2);
                    _sum3 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pC0[3]), _beta, _sum3);
                    _sum4 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pC0[4]), _beta, _sum4);
                    _sum5 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pC0[5]), _beta, _sum5);
                    _sum6 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pC0[6]), _beta, _sum6);
                    _sum7 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pC0[7]), _beta, _sum7);
                    _sum8 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pC0[8]), _beta, _sum8);
                    _sum9 = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pC0[9]), _beta, _sum9);
                    _suma = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pC0[10]), _beta, _suma);
                    _sumb = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pC0[11]), _beta, _sumb);
                    _sumc = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pC0[12]), _beta, _sumc);
                    _sumd = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pC0[13]), _beta, _sumd);
                    _sume = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pC0[14]), _beta, _sume);
                    _sumf = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(pC0[15]), _beta, _sumf);
                    pC0 += 16;
                }
            }

            if (alpha != 1.f)
            {
                _sum0 = __lasx_xvfmul_s(_sum0, _valpha);
                _sum1 = __lasx_xvfmul_s(_sum1, _valpha);
                _sum2 = __lasx_xvfmul_s(_sum2, _valpha);
                _sum3 = __lasx_xvfmul_s(_sum3, _valpha);
                _sum4 = __lasx_xvfmul_s(_sum4, _valpha);
                _sum5 = __lasx_xvfmul_s(_sum5, _valpha);
                _sum6 = __lasx_xvfmul_s(_sum6, _valpha);
                _sum7 = __lasx_xvfmul_s(_sum7, _valpha);
                _sum8 = __lasx_xvfmul_s(_sum8, _valpha);
                _sum9 = __lasx_xvfmul_s(_sum9, _valpha);
                _suma = __lasx_xvfmul_s(_suma, _valpha);
                _sumb = __lasx_xvfmul_s(_sumb, _valpha);
                _sumc = __lasx_xvfmul_s(_sumc, _valpha);
                _sumd = __lasx_xvfmul_s(_sumd, _valpha);
                _sume = __lasx_xvfmul_s(_sume, _valpha);
                _sumf = __lasx_xvfmul_s(_sumf, _valpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    float* p1f = p0f + out_hstep * 8;
                    if (out_elempack == 8)
                    {
                        transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
                        transpose8x8_ps(_sum8, _sum9, _suma, _sumb, _sumc, _sumd, _sume, _sumf);
                        __lasx_xvst(_sum0, p0f, 0);
                        __lasx_xvst(_sum1, p0f + 8, 0);
                        __lasx_xvst(_sum2, p0f + 16, 0);
                        __lasx_xvst(_sum3, p0f + 24, 0);
                        __lasx_xvst(_sum4, p0f + 32, 0);
                        __lasx_xvst(_sum5, p0f + 40, 0);
                        __lasx_xvst(_sum6, p0f + 48, 0);
                        __lasx_xvst(_sum7, p0f + 56, 0);
                        __lasx_xvst(_sum8, p1f, 0);
                        __lasx_xvst(_sum9, p1f + 8, 0);
                        __lasx_xvst(_suma, p1f + 16, 0);
                        __lasx_xvst(_sumb, p1f + 24, 0);
                        __lasx_xvst(_sumc, p1f + 32, 0);
                        __lasx_xvst(_sumd, p1f + 40, 0);
                        __lasx_xvst(_sume, p1f + 48, 0);
                        __lasx_xvst(_sumf, p1f + 56, 0);
                    }
                    if (out_elempack == 4)
                    {
                        transpose8x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose8x4_ps(_sum4, _sum5, _sum6, _sum7);
                        transpose8x4_ps(_sum8, _sum9, _suma, _sumb);
                        transpose8x4_ps(_sumc, _sumd, _sume, _sumf);
                        __lasx_xvst(_sum0, p0f, 0);
                        __lasx_xvst(_sum1, p0f + 8, 0);
                        __lasx_xvst(_sum2, p0f + 16, 0);
                        __lasx_xvst(_sum3, p0f + 24, 0);
                        __lasx_xvst(_sum4, p0f + out_hstep * 4, 0);
                        __lasx_xvst(_sum5, p0f + out_hstep * 4 + 8, 0);
                        __lasx_xvst(_sum6, p0f + out_hstep * 4 + 16, 0);
                        __lasx_xvst(_sum7, p0f + out_hstep * 4 + 24, 0);
                        __lasx_xvst(_sum8, p1f, 0);
                        __lasx_xvst(_sum9, p1f + 8, 0);
                        __lasx_xvst(_suma, p1f + 16, 0);
                        __lasx_xvst(_sumb, p1f + 24, 0);
                        __lasx_xvst(_sumc, p1f + out_hstep * 4, 0);
                        __lasx_xvst(_sumd, p1f + out_hstep * 4 + 8, 0);
                        __lasx_xvst(_sume, p1f + out_hstep * 4 + 16, 0);
                        __lasx_xvst(_sumf, p1f + out_hstep * 4 + 24, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lasx_xvst(_sum0, p0f, 0);
                        __lasx_xvst(_sum1, p0f + out_hstep, 0);
                        __lasx_xvst(_sum2, p0f + out_hstep * 2, 0);
                        __lasx_xvst(_sum3, p0f + out_hstep * 3, 0);
                        __lasx_xvst(_sum4, p0f + out_hstep * 4, 0);
                        __lasx_xvst(_sum5, p0f + out_hstep * 5, 0);
                        __lasx_xvst(_sum6, p0f + out_hstep * 6, 0);
                        __lasx_xvst(_sum7, p0f + out_hstep * 7, 0);
                        __lasx_xvst(_sum8, p1f, 0);
                        __lasx_xvst(_sum9, p1f + out_hstep, 0);
                        __lasx_xvst(_suma, p1f + out_hstep * 2, 0);
                        __lasx_xvst(_sumb, p1f + out_hstep * 3, 0);
                        __lasx_xvst(_sumc, p1f + out_hstep * 4, 0);
                        __lasx_xvst(_sumd, p1f + out_hstep * 5, 0);
                        __lasx_xvst(_sume, p1f + out_hstep * 6, 0);
                        __lasx_xvst(_sumf, p1f + out_hstep * 7, 0);
                    }
                    p0f += out_hstep * 16;
                }
                else
                {
                    if (out_elempack == 8)
                    {
                        __lasx_xvst(_sum0, p0f, 0);
                        __lasx_xvst(_sum1, p0f + 8, 0);
                        __lasx_xvst(_sum2, p0f + 16, 0);
                        __lasx_xvst(_sum3, p0f + 24, 0);
                        __lasx_xvst(_sum4, p0f + 32, 0);
                        __lasx_xvst(_sum5, p0f + 40, 0);
                        __lasx_xvst(_sum6, p0f + 48, 0);
                        __lasx_xvst(_sum7, p0f + 56, 0);
                        __lasx_xvst(_sum8, p0f + 64, 0);
                        __lasx_xvst(_sum9, p0f + 72, 0);
                        __lasx_xvst(_suma, p0f + 80, 0);
                        __lasx_xvst(_sumb, p0f + 88, 0);
                        __lasx_xvst(_sumc, p0f + 96, 0);
                        __lasx_xvst(_sumd, p0f + 104, 0);
                        __lasx_xvst(_sume, p0f + 112, 0);
                        __lasx_xvst(_sumf, p0f + 120, 0);
                        p0f += 128;
                    }
                    if (out_elempack == 4)
                    {
                        float* p1f = p0f + out_hstep * 4;
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum0), p0f, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum1), p0f + 4, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum2), p0f + 8, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum3), p0f + 12, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum4), p0f + 16, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum5), p0f + 20, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum6), p0f + 24, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum7), p0f + 28, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum8), p0f + 32, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum9), p0f + 36, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_suma), p0f + 40, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sumb), p0f + 44, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sumc), p0f + 48, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sumd), p0f + 52, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sume), p0f + 56, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sumf), p0f + 60, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum0), p1f, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum1), p1f + 4, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum2), p1f + 8, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum3), p1f + 12, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum4), p1f + 16, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum5), p1f + 20, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum6), p1f + 24, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum7), p1f + 28, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum8), p1f + 32, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum9), p1f + 36, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_suma), p1f + 40, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sumb), p1f + 44, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sumc), p1f + 48, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sumd), p1f + 52, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sume), p1f + 56, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sumf), p1f + 60, 0);
                        p0f += 64;
                    }
                    if (out_elempack == 1)
                    {
                        transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
                        transpose8x8_ps(_sum8, _sum9, _suma, _sumb, _sumc, _sumd, _sume, _sumf);
                        __lasx_xvst(_sum0, p0f, 0);
                        __lasx_xvst(_sum1, p0f + out_hstep, 0);
                        __lasx_xvst(_sum2, p0f + out_hstep * 2, 0);
                        __lasx_xvst(_sum3, p0f + out_hstep * 3, 0);
                        __lasx_xvst(_sum4, p0f + out_hstep * 4, 0);
                        __lasx_xvst(_sum5, p0f + out_hstep * 5, 0);
                        __lasx_xvst(_sum6, p0f + out_hstep * 6, 0);
                        __lasx_xvst(_sum7, p0f + out_hstep * 7, 0);
                        __lasx_xvst(_sum8, p0f + 8, 0);
                        __lasx_xvst(_sum9, p0f + out_hstep + 8, 0);
                        __lasx_xvst(_suma, p0f + out_hstep * 2 + 8, 0);
                        __lasx_xvst(_sumb, p0f + out_hstep * 3 + 8, 0);
                        __lasx_xvst(_sumc, p0f + out_hstep * 4 + 8, 0);
                        __lasx_xvst(_sumd, p0f + out_hstep * 5 + 8, 0);
                        __lasx_xvst(_sume, p0f + out_hstep * 6 + 8, 0);
                        __lasx_xvst(_sumf, p0f + out_hstep * 7 + 8, 0);
                        p0f += 16;
                    }
                }
            }
            else
            {
                if (output_transpose)
                {
                    unsigned short* p1 = p0 + out_hstep * 8;
                    if (out_elempack == 8)
                    {
                        transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
                        transpose8x8_ps(_sum8, _sum9, _suma, _sumb, _sumc, _sumd, _sume, _sumf);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum0), float2bfloat_lasx(_sum1)), p0, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum2), float2bfloat_lasx(_sum3)), p0 + 16, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum4), float2bfloat_lasx(_sum5)), p0 + 32, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum6), float2bfloat_lasx(_sum7)), p0 + 48, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum8), float2bfloat_lasx(_sum9)), p1, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_suma), float2bfloat_lasx(_sumb)), p1 + 16, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sumc), float2bfloat_lasx(_sumd)), p1 + 32, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sume), float2bfloat_lasx(_sumf)), p1 + 48, 0);
                    }
                    if (out_elempack == 4)
                    {
                        transpose8x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose8x4_ps(_sum4, _sum5, _sum6, _sum7);
                        transpose8x4_ps(_sum8, _sum9, _suma, _sumb);
                        transpose8x4_ps(_sumc, _sumd, _sume, _sumf);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum0), float2bfloat_lasx(_sum1)), p0, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum2), float2bfloat_lasx(_sum3)), p0 + 16, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum4), float2bfloat_lasx(_sum5)), p0 + out_hstep * 4, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum6), float2bfloat_lasx(_sum7)), p0 + out_hstep * 4 + 16, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum8), float2bfloat_lasx(_sum9)), p1, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_suma), float2bfloat_lasx(_sumb)), p1 + 16, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sumc), float2bfloat_lasx(_sumd)), p1 + out_hstep * 4, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sume), float2bfloat_lasx(_sumf)), p1 + out_hstep * 4 + 16, 0);
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
                        __lsx_vst(float2bfloat_lasx(_sum8), p1, 0);
                        __lsx_vst(float2bfloat_lasx(_sum9), p1 + out_hstep, 0);
                        __lsx_vst(float2bfloat_lasx(_suma), p1 + out_hstep * 2, 0);
                        __lsx_vst(float2bfloat_lasx(_sumb), p1 + out_hstep * 3, 0);
                        __lsx_vst(float2bfloat_lasx(_sumc), p1 + out_hstep * 4, 0);
                        __lsx_vst(float2bfloat_lasx(_sumd), p1 + out_hstep * 5, 0);
                        __lsx_vst(float2bfloat_lasx(_sume), p1 + out_hstep * 6, 0);
                        __lsx_vst(float2bfloat_lasx(_sumf), p1 + out_hstep * 7, 0);
                    }
                    p0 += out_hstep * 16;
                }
                else
                {
                    if (out_elempack == 8)
                    {
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum0), float2bfloat_lasx(_sum1)), p0, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum2), float2bfloat_lasx(_sum3)), p0 + 16, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum4), float2bfloat_lasx(_sum5)), p0 + 32, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum6), float2bfloat_lasx(_sum7)), p0 + 48, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum8), float2bfloat_lasx(_sum9)), p0 + 64, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_suma), float2bfloat_lasx(_sumb)), p0 + 80, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sumc), float2bfloat_lasx(_sumd)), p0 + 96, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sume), float2bfloat_lasx(_sumf)), p0 + 112, 0);
                        p0 += 128;
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
                        __m128i _bf8 = float2bfloat_lasx(_sum8);
                        __m128i _bf9 = float2bfloat_lasx(_sum9);
                        __m128i _bfa = float2bfloat_lasx(_suma);
                        __m128i _bfb = float2bfloat_lasx(_sumb);
                        __m128i _bfc = float2bfloat_lasx(_sumc);
                        __m128i _bfd = float2bfloat_lasx(_sumd);
                        __m128i _bfe = float2bfloat_lasx(_sume);
                        __m128i _bff = float2bfloat_lasx(_sumf);
                        __lsx_vstelm_d(_bf0, p0, 0, 0);
                        __lsx_vstelm_d(_bf1, p0 + 4, 0, 0);
                        __lsx_vstelm_d(_bf2, p0 + 8, 0, 0);
                        __lsx_vstelm_d(_bf3, p0 + 12, 0, 0);
                        __lsx_vstelm_d(_bf4, p0 + 16, 0, 0);
                        __lsx_vstelm_d(_bf5, p0 + 20, 0, 0);
                        __lsx_vstelm_d(_bf6, p0 + 24, 0, 0);
                        __lsx_vstelm_d(_bf7, p0 + 28, 0, 0);
                        __lsx_vstelm_d(_bf8, p0 + 32, 0, 0);
                        __lsx_vstelm_d(_bf9, p0 + 36, 0, 0);
                        __lsx_vstelm_d(_bfa, p0 + 40, 0, 0);
                        __lsx_vstelm_d(_bfb, p0 + 44, 0, 0);
                        __lsx_vstelm_d(_bfc, p0 + 48, 0, 0);
                        __lsx_vstelm_d(_bfd, p0 + 52, 0, 0);
                        __lsx_vstelm_d(_bfe, p0 + 56, 0, 0);
                        __lsx_vstelm_d(_bff, p0 + 60, 0, 0);
                        __lsx_vstelm_d(_bf0, p1, 0, 1);
                        __lsx_vstelm_d(_bf1, p1 + 4, 0, 1);
                        __lsx_vstelm_d(_bf2, p1 + 8, 0, 1);
                        __lsx_vstelm_d(_bf3, p1 + 12, 0, 1);
                        __lsx_vstelm_d(_bf4, p1 + 16, 0, 1);
                        __lsx_vstelm_d(_bf5, p1 + 20, 0, 1);
                        __lsx_vstelm_d(_bf6, p1 + 24, 0, 1);
                        __lsx_vstelm_d(_bf7, p1 + 28, 0, 1);
                        __lsx_vstelm_d(_bf8, p1 + 32, 0, 1);
                        __lsx_vstelm_d(_bf9, p1 + 36, 0, 1);
                        __lsx_vstelm_d(_bfa, p1 + 40, 0, 1);
                        __lsx_vstelm_d(_bfb, p1 + 44, 0, 1);
                        __lsx_vstelm_d(_bfc, p1 + 48, 0, 1);
                        __lsx_vstelm_d(_bfd, p1 + 52, 0, 1);
                        __lsx_vstelm_d(_bfe, p1 + 56, 0, 1);
                        __lsx_vstelm_d(_bff, p1 + 60, 0, 1);
                        p0 += 64;
                    }
                    if (out_elempack == 1)
                    {
                        transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
                        transpose8x8_ps(_sum8, _sum9, _suma, _sumb, _sumc, _sumd, _sume, _sumf);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum0), float2bfloat_lasx(_sum8)), p0, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum1), float2bfloat_lasx(_sum9)), p0 + out_hstep, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum2), float2bfloat_lasx(_suma)), p0 + out_hstep * 2, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum3), float2bfloat_lasx(_sumb)), p0 + out_hstep * 3, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum4), float2bfloat_lasx(_sumc)), p0 + out_hstep * 4, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum5), float2bfloat_lasx(_sumd)), p0 + out_hstep * 5, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum6), float2bfloat_lasx(_sume)), p0 + out_hstep * 6, 0);
                        __lasx_xvst(__lasx_concat_128(float2bfloat_lasx(_sum7), float2bfloat_lasx(_sumf)), p0 + out_hstep * 7, 0);
                        p0 += 16;
                    }
                }
            }
        }
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

            // deshuffle from the shuffle-based 8x8 kernel
            {
                __m256 _tmp0 = _sum0;
                __m256 _tmp1 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp2 = _sum2;
                __m256 _tmp3 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum3, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp4 = _sum4;
                __m256 _tmp5 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum5, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp6 = _sum6;
                __m256 _tmp7 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum7, _LSX_SHUFFLE(2, 1, 0, 3));

                _sum0 = (__m256)__lasx_xvilvl_w((__m256i)_tmp3, (__m256i)_tmp0);
                _sum1 = (__m256)__lasx_xvilvh_w((__m256i)_tmp3, (__m256i)_tmp0);
                _sum2 = (__m256)__lasx_xvilvl_w((__m256i)_tmp1, (__m256i)_tmp2);
                _sum3 = (__m256)__lasx_xvilvh_w((__m256i)_tmp1, (__m256i)_tmp2);
                _sum4 = (__m256)__lasx_xvilvl_w((__m256i)_tmp7, (__m256i)_tmp4);
                _sum5 = (__m256)__lasx_xvilvh_w((__m256i)_tmp7, (__m256i)_tmp4);
                _sum6 = (__m256)__lasx_xvilvl_w((__m256i)_tmp5, (__m256i)_tmp6);
                _sum7 = (__m256)__lasx_xvilvh_w((__m256i)_tmp5, (__m256i)_tmp6);

                _tmp0 = (__m256)__lasx_xvilvl_d((__m256i)_sum2, (__m256i)_sum0);
                _tmp1 = (__m256)__lasx_xvilvh_d((__m256i)_sum2, (__m256i)_sum0);
                _tmp2 = (__m256)__lasx_xvilvl_d((__m256i)_sum1, (__m256i)_sum3);
                _tmp3 = (__m256)__lasx_xvilvh_d((__m256i)_sum1, (__m256i)_sum3);
                _tmp4 = (__m256)__lasx_xvilvl_d((__m256i)_sum6, (__m256i)_sum4);
                _tmp5 = (__m256)__lasx_xvilvh_d((__m256i)_sum6, (__m256i)_sum4);
                _tmp6 = (__m256)__lasx_xvilvl_d((__m256i)_sum5, (__m256i)_sum7);
                _tmp7 = (__m256)__lasx_xvilvh_d((__m256i)_sum5, (__m256i)_sum7);

                _tmp1 = (__m256)__lasx_xvshuf4i_w((__m256i)_tmp1, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp3 = (__m256)__lasx_xvshuf4i_w((__m256i)_tmp3, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp5 = (__m256)__lasx_xvshuf4i_w((__m256i)_tmp5, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp7 = (__m256)__lasx_xvshuf4i_w((__m256i)_tmp7, _LSX_SHUFFLE(2, 1, 0, 3));

                _sum0 = (__m256)__lasx_xvpermi_q((__m256i)_tmp4, (__m256i)_tmp0, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum1 = (__m256)__lasx_xvpermi_q((__m256i)_tmp5, (__m256i)_tmp1, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum2 = (__m256)__lasx_xvpermi_q((__m256i)_tmp6, (__m256i)_tmp2, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum3 = (__m256)__lasx_xvpermi_q((__m256i)_tmp7, (__m256i)_tmp3, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum4 = (__m256)__lasx_xvpermi_q((__m256i)_tmp0, (__m256i)_tmp4, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum5 = (__m256)__lasx_xvpermi_q((__m256i)_tmp1, (__m256i)_tmp5, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum6 = (__m256)__lasx_xvpermi_q((__m256i)_tmp2, (__m256i)_tmp6, _LSX_SHUFFLE(0, 3, 0, 0));
                _sum7 = (__m256)__lasx_xvpermi_q((__m256i)_tmp3, (__m256i)_tmp7, _LSX_SHUFFLE(0, 3, 0, 0));
            }

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
                    __m256 _c0 = (__m256)__lasx_xvld(pC0, 0);
                    __m256 _c1 = (__m256)__lasx_xvld(pC0 + 8, 0);
                    __m256 _c2 = (__m256)__lasx_xvld(pC0 + 16, 0);
                    __m256 _c3 = (__m256)__lasx_xvld(pC0 + 24, 0);
                    __m256 _c4 = (__m256)__lasx_xvld(pC0 + 32, 0);
                    __m256 _c5 = (__m256)__lasx_xvld(pC0 + 40, 0);
                    __m256 _c6 = (__m256)__lasx_xvld(pC0 + 48, 0);
                    __m256 _c7 = (__m256)__lasx_xvld(pC0 + 56, 0);
                    _sum0 = __lasx_xvfmadd_s(_c0, _beta, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_c1, _beta, _sum1);
                    _sum2 = __lasx_xvfmadd_s(_c2, _beta, _sum2);
                    _sum3 = __lasx_xvfmadd_s(_c3, _beta, _sum3);
                    _sum4 = __lasx_xvfmadd_s(_c4, _beta, _sum4);
                    _sum5 = __lasx_xvfmadd_s(_c5, _beta, _sum5);
                    _sum6 = __lasx_xvfmadd_s(_c6, _beta, _sum6);
                    _sum7 = __lasx_xvfmadd_s(_c7, _beta, _sum7);
                    pC0 += 64;
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

            if (alpha != 1.f)
            {
                _sum0 = __lasx_xvfmul_s(_sum0, _valpha);
                _sum1 = __lasx_xvfmul_s(_sum1, _valpha);
                _sum2 = __lasx_xvfmul_s(_sum2, _valpha);
                _sum3 = __lasx_xvfmul_s(_sum3, _valpha);
                _sum4 = __lasx_xvfmul_s(_sum4, _valpha);
                _sum5 = __lasx_xvfmul_s(_sum5, _valpha);
                _sum6 = __lasx_xvfmul_s(_sum6, _valpha);
                _sum7 = __lasx_xvfmul_s(_sum7, _valpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 8)
                    {
                        transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
                        __lasx_xvst(_sum0, p0f, 0);
                        __lasx_xvst(_sum1, p0f + 8, 0);
                        __lasx_xvst(_sum2, p0f + 16, 0);
                        __lasx_xvst(_sum3, p0f + 24, 0);
                        __lasx_xvst(_sum4, p0f + 32, 0);
                        __lasx_xvst(_sum5, p0f + 40, 0);
                        __lasx_xvst(_sum6, p0f + 48, 0);
                        __lasx_xvst(_sum7, p0f + 56, 0);
                    }
                    if (out_elempack == 4)
                    {
                        transpose8x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose8x4_ps(_sum4, _sum5, _sum6, _sum7);
                        __lasx_xvst(_sum0, p0f, 0);
                        __lasx_xvst(_sum1, p0f + 8, 0);
                        __lasx_xvst(_sum2, p0f + 16, 0);
                        __lasx_xvst(_sum3, p0f + 24, 0);
                        __lasx_xvst(_sum4, p0f + out_hstep * 4, 0);
                        __lasx_xvst(_sum5, p0f + out_hstep * 4 + 8, 0);
                        __lasx_xvst(_sum6, p0f + out_hstep * 4 + 16, 0);
                        __lasx_xvst(_sum7, p0f + out_hstep * 4 + 24, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lasx_xvst(_sum0, p0f, 0);
                        __lasx_xvst(_sum1, p0f + out_hstep, 0);
                        __lasx_xvst(_sum2, p0f + out_hstep * 2, 0);
                        __lasx_xvst(_sum3, p0f + out_hstep * 3, 0);
                        __lasx_xvst(_sum4, p0f + out_hstep * 4, 0);
                        __lasx_xvst(_sum5, p0f + out_hstep * 5, 0);
                        __lasx_xvst(_sum6, p0f + out_hstep * 6, 0);
                        __lasx_xvst(_sum7, p0f + out_hstep * 7, 0);
                    }
                    p0f += out_hstep * 8;
                }
                else
                {
                    if (out_elempack == 8)
                    {
                        __lasx_xvst(_sum0, p0f, 0);
                        __lasx_xvst(_sum1, p0f + 8, 0);
                        __lasx_xvst(_sum2, p0f + 16, 0);
                        __lasx_xvst(_sum3, p0f + 24, 0);
                        __lasx_xvst(_sum4, p0f + 32, 0);
                        __lasx_xvst(_sum5, p0f + 40, 0);
                        __lasx_xvst(_sum6, p0f + 48, 0);
                        __lasx_xvst(_sum7, p0f + 56, 0);
                        p0f += 64;
                    }
                    if (out_elempack == 4)
                    {
                        float* p1f = p0f + out_hstep * 4;
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum0), p0f, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum1), p0f + 4, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum2), p0f + 8, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum3), p0f + 12, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum4), p0f + 16, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum5), p0f + 20, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum6), p0f + 24, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum7), p0f + 28, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum0), p1f, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum1), p1f + 4, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum2), p1f + 8, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum3), p1f + 12, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum4), p1f + 16, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum5), p1f + 20, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum6), p1f + 24, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum7), p1f + 28, 0);
                        p0f += 32;
                    }
                    if (out_elempack == 1)
                    {
                        transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
                        __lasx_xvst(_sum0, p0f, 0);
                        __lasx_xvst(_sum1, p0f + out_hstep, 0);
                        __lasx_xvst(_sum2, p0f + out_hstep * 2, 0);
                        __lasx_xvst(_sum3, p0f + out_hstep * 3, 0);
                        __lasx_xvst(_sum4, p0f + out_hstep * 4, 0);
                        __lasx_xvst(_sum5, p0f + out_hstep * 5, 0);
                        __lasx_xvst(_sum6, p0f + out_hstep * 6, 0);
                        __lasx_xvst(_sum7, p0f + out_hstep * 7, 0);
                        p0f += 8;
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
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m256 _sum0 = (__m256)__lasx_xvld(pp, 0);
            __m256 _sum1 = (__m256)__lasx_xvld(pp + 8, 0);
            __m256 _sum2 = (__m256)__lasx_xvld(pp + 16, 0);
            __m256 _sum3 = (__m256)__lasx_xvld(pp + 24, 0);
            pp += 32;

            // deshuffle from the shuffle-based 8x4 kernel
            _sum1 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum3 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum3, _LSX_SHUFFLE(2, 1, 0, 3));
            {
                __m256 _tmp0 = (__m256)__lasx_xvilvl_w((__m256i)_sum3, (__m256i)_sum0);
                __m256 _tmp1 = (__m256)__lasx_xvilvh_w((__m256i)_sum3, (__m256i)_sum0);
                __m256 _tmp2 = (__m256)__lasx_xvilvl_w((__m256i)_sum1, (__m256i)_sum2);
                __m256 _tmp3 = (__m256)__lasx_xvilvh_w((__m256i)_sum1, (__m256i)_sum2);
                _sum0 = (__m256)__lasx_xvilvl_d((__m256i)_tmp2, (__m256i)_tmp0);
                _sum1 = (__m256)__lasx_xvilvh_d((__m256i)_tmp2, (__m256i)_tmp0);
                _sum2 = (__m256)__lasx_xvilvl_d((__m256i)_tmp1, (__m256i)_tmp3);
                _sum3 = (__m256)__lasx_xvilvh_d((__m256i)_tmp1, (__m256i)_tmp3);
            }
            _sum1 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum3 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum3, _LSX_SHUFFLE(2, 1, 0, 3));

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
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m256 _c = (__m256)__lasx_xvld(pC0, 0);
                    _sum0 = __lasx_xvfmadd_s(_c, _beta, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_c, _beta, _sum1);
                    _sum2 = __lasx_xvfmadd_s(_c, _beta, _sum2);
                    _sum3 = __lasx_xvfmadd_s(_c, _beta, _sum3);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c0 = (__m256)__lasx_xvld(pC0, 0);
                    __m256 _c1 = (__m256)__lasx_xvld(pC0 + 8, 0);
                    __m256 _c2 = (__m256)__lasx_xvld(pC0 + 16, 0);
                    __m256 _c3 = (__m256)__lasx_xvld(pC0 + 24, 0);
                    _sum0 = __lasx_xvfmadd_s(_c0, _beta, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_c1, _beta, _sum1);
                    _sum2 = __lasx_xvfmadd_s(_c2, _beta, _sum2);
                    _sum3 = __lasx_xvfmadd_s(_c3, _beta, _sum3);
                    pC0 += 32;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c0 = (__m256)__lasx_xvreplfr2vr_s(pC0[0]);
                    __m256 _c1 = (__m256)__lasx_xvreplfr2vr_s(pC0[1]);
                    __m256 _c2 = (__m256)__lasx_xvreplfr2vr_s(pC0[2]);
                    __m256 _c3 = (__m256)__lasx_xvreplfr2vr_s(pC0[3]);
                    _sum0 = __lasx_xvfmadd_s(_c0, _beta, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_c1, _beta, _sum1);
                    _sum2 = __lasx_xvfmadd_s(_c2, _beta, _sum2);
                    _sum3 = __lasx_xvfmadd_s(_c3, _beta, _sum3);
                    pC0 += 4;
                }
            }

            if (alpha != 1.f)
            {
                _sum0 = __lasx_xvfmul_s(_sum0, _valpha);
                _sum1 = __lasx_xvfmul_s(_sum1, _valpha);
                _sum2 = __lasx_xvfmul_s(_sum2, _valpha);
                _sum3 = __lasx_xvfmul_s(_sum3, _valpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 8)
                    {
                        transpose8x4_ps(_sum0, _sum1, _sum2, _sum3);
                        float* p1f = p0f;
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum0), p1f, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum0), p1f + 8, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum1), p1f + 16, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum1), p1f + 24, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum2), p1f + 32, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum2), p1f + 40, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum3), p1f + 48, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum3), p1f + 56, 0);
                    }
                    if (out_elempack == 4)
                    {
                        transpose8x4_ps(_sum0, _sum1, _sum2, _sum3);
                        __lasx_xvst(_sum0, p0f, 0);
                        __lasx_xvst(_sum1, p0f + 8, 0);
                        __lasx_xvst(_sum2, p0f + 16, 0);
                        __lasx_xvst(_sum3, p0f + 24, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lasx_xvst(_sum0, p0f, 0);
                        __lasx_xvst(_sum1, p0f + out_hstep, 0);
                        __lasx_xvst(_sum2, p0f + out_hstep * 2, 0);
                        __lasx_xvst(_sum3, p0f + out_hstep * 3, 0);
                    }
                    p0f += out_hstep * 4;
                }
                else
                {
                    if (out_elempack == 8)
                    {
                        __lasx_xvst(_sum0, p0f, 0);
                        __lasx_xvst(_sum1, p0f + 8, 0);
                        __lasx_xvst(_sum2, p0f + 16, 0);
                        __lasx_xvst(_sum3, p0f + 24, 0);
                        p0f += 32;
                    }
                    if (out_elempack == 4)
                    {
                        float* p1f = p0f + out_hstep * 4;
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum0), p0f, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum1), p0f + 4, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum2), p0f + 8, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum3), p0f + 12, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum0), p1f, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum1), p1f + 4, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum2), p1f + 8, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum3), p1f + 12, 0);
                        p0f += 16;
                    }
                    if (out_elempack == 1)
                    {
                        transpose8x4_ps(_sum0, _sum1, _sum2, _sum3);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum0), p0f, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum0), p0f + out_hstep, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum1), p0f + out_hstep * 2, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum1), p0f + out_hstep * 3, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum2), p0f + out_hstep * 4, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum2), p0f + out_hstep * 5, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum3), p0f + out_hstep * 6, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum3), p0f + out_hstep * 7, 0);
                        p0f += 4;
                    }
                }
            }
            else
            {
                __m128i _bf0 = float2bfloat_lasx(_sum0);
                __m128i _bf1 = float2bfloat_lasx(_sum1);
                __m128i _bf2 = float2bfloat_lasx(_sum2);
                __m128i _bf3 = float2bfloat_lasx(_sum3);

                if (output_transpose)
                {
                    if (out_elempack == 8)
                    {
                        transpose8x4_epi16(_bf0, _bf1, _bf2, _bf3);
                        unsigned short* p1 = p0;
                        __lsx_vstelm_d(_bf0, p1, 0, 0);
                        __lsx_vstelm_d(_bf0, p1 + 8, 0, 1);
                        __lsx_vstelm_d(_bf1, p1 + 16, 0, 0);
                        __lsx_vstelm_d(_bf1, p1 + 24, 0, 1);
                        __lsx_vstelm_d(_bf2, p1 + 32, 0, 0);
                        __lsx_vstelm_d(_bf2, p1 + 40, 0, 1);
                        __lsx_vstelm_d(_bf3, p1 + 48, 0, 0);
                        __lsx_vstelm_d(_bf3, p1 + 56, 0, 1);
                    }
                    if (out_elempack == 4)
                    {
                        transpose8x4_epi16(_bf0, _bf1, _bf2, _bf3);
                        __lsx_vst(_bf0, p0, 0);
                        __lsx_vst(_bf1, p0 + 8, 0);
                        __lsx_vst(_bf2, p0 + 16, 0);
                        __lsx_vst(_bf3, p0 + 24, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vst(_bf0, p0, 0);
                        __lsx_vst(_bf1, p0 + out_hstep, 0);
                        __lsx_vst(_bf2, p0 + out_hstep * 2, 0);
                        __lsx_vst(_bf3, p0 + out_hstep * 3, 0);
                    }
                    p0 += out_hstep * 4;
                }
                else
                {
                    if (out_elempack == 8)
                    {
                        __lsx_vst(_bf0, p0, 0);
                        __lsx_vst(_bf1, p0 + 8, 0);
                        __lsx_vst(_bf2, p0 + 16, 0);
                        __lsx_vst(_bf3, p0 + 24, 0);
                        p0 += 32;
                    }
                    if (out_elempack == 4)
                    {
                        unsigned short* p1 = p0 + out_hstep * 4;
                        __lsx_vstelm_d(_bf0, p0, 0, 0);
                        __lsx_vstelm_d(_bf1, p0 + 4, 0, 0);
                        __lsx_vstelm_d(_bf2, p0 + 8, 0, 0);
                        __lsx_vstelm_d(_bf3, p0 + 12, 0, 0);
                        __lsx_vstelm_d(_bf0, p1, 0, 1);
                        __lsx_vstelm_d(_bf1, p1 + 4, 0, 1);
                        __lsx_vstelm_d(_bf2, p1 + 8, 0, 1);
                        __lsx_vstelm_d(_bf3, p1 + 12, 0, 1);
                        p0 += 16;
                    }
                    if (out_elempack == 1)
                    {
                        transpose8x4_epi16(_bf0, _bf1, _bf2, _bf3);
                        __lsx_vstelm_d(_bf0, p0, 0, 0);
                        __lsx_vstelm_d(_bf0, p0 + out_hstep, 0, 1);
                        __lsx_vstelm_d(_bf1, p0 + out_hstep * 2, 0, 0);
                        __lsx_vstelm_d(_bf1, p0 + out_hstep * 3, 0, 1);
                        __lsx_vstelm_d(_bf2, p0 + out_hstep * 4, 0, 0);
                        __lsx_vstelm_d(_bf2, p0 + out_hstep * 5, 0, 1);
                        __lsx_vstelm_d(_bf3, p0 + out_hstep * 6, 0, 0);
                        __lsx_vstelm_d(_bf3, p0 + out_hstep * 7, 0, 1);
                        p0 += 4;
                    }
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m256 _sum0 = (__m256)__lasx_xvld(pp, 0);
            __m256 _sum1 = (__m256)__lasx_xvld(pp + 8, 0);
            pp += 16;

            // deshuffle from the shuffle-based 8x2 kernel
            {
                __m256 _tmp0 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum0, _LSX_SHUFFLE(3, 1, 2, 0));
                __m256 _tmp1 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum1, _LSX_SHUFFLE(0, 2, 3, 1));
                _sum0 = (__m256)__lasx_xvilvl_w((__m256i)_tmp1, (__m256i)_tmp0);
                _sum1 = (__m256)__lasx_xvilvh_w((__m256i)_tmp1, (__m256i)_tmp0);
                _sum1 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
            }

            if (pC0)
            {
                __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pC0[0]);
                    _sum0 = __lasx_xvfmadd_s(_c, _beta, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_c, _beta, _sum1);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m256 _c = (__m256)__lasx_xvld(pC0, 0);
                    _sum0 = __lasx_xvfmadd_s(_c, _beta, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_c, _beta, _sum1);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c0 = (__m256)__lasx_xvld(pC0, 0);
                    __m256 _c1 = (__m256)__lasx_xvld(pC0 + 8, 0);
                    _sum0 = __lasx_xvfmadd_s(_c0, _beta, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_c1, _beta, _sum1);
                    pC0 += 16;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c0 = (__m256)__lasx_xvreplfr2vr_s(pC0[0]);
                    __m256 _c1 = (__m256)__lasx_xvreplfr2vr_s(pC0[1]);
                    _sum0 = __lasx_xvfmadd_s(_c0, _beta, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_c1, _beta, _sum1);
                    pC0 += 2;
                }
            }

            if (alpha != 1.f)
            {
                _sum0 = __lasx_xvfmul_s(_sum0, _valpha);
                _sum1 = __lasx_xvfmul_s(_sum1, _valpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 8)
                    {
                        __lasx_xvst(_sum0, p0f, 0);
                        __lasx_xvst(_sum1, p0f + 8, 0);
                    }
                    if (out_elempack == 4)
                    {
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum0), p0f, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum1), p0f + 4, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum0), p0f + out_hstep * 4, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum1), p0f + out_hstep * 4 + 4, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lasx_xvst(_sum0, p0f, 0);
                        __lasx_xvst(_sum1, p0f + out_hstep, 0);
                    }
                    p0f += out_hstep * 2;
                }
                else
                {
                    if (out_elempack == 8)
                    {
                        __lasx_xvst(_sum0, p0f, 0);
                        __lasx_xvst(_sum1, p0f + 8, 0);
                        p0f += 16;
                    }
                    if (out_elempack == 4)
                    {
                        float* p1f = p0f + out_hstep * 4;
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum0), p0f, 0);
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum1), p0f + 4, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum0), p1f, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum1), p1f + 4, 0);
                        p0f += 8;
                    }
                    if (out_elempack == 1)
                    {
                        __m256i _tmp0 = __lasx_xvilvl_w((__m256i)_sum1, (__m256i)_sum0);
                        __m256i _tmp1 = __lasx_xvilvh_w((__m256i)_sum1, (__m256i)_sum0);
                        __lasx_xvstelm_d(_tmp0, p0f, 0, 0);
                        __lasx_xvstelm_d(_tmp0, p0f + out_hstep, 0, 1);
                        __lasx_xvstelm_d(_tmp1, p0f + out_hstep * 2, 0, 0);
                        __lasx_xvstelm_d(_tmp1, p0f + out_hstep * 3, 0, 1);
                        __lasx_xvstelm_d(_tmp0, p0f + out_hstep * 4, 0, 2);
                        __lasx_xvstelm_d(_tmp0, p0f + out_hstep * 5, 0, 3);
                        __lasx_xvstelm_d(_tmp1, p0f + out_hstep * 6, 0, 2);
                        __lasx_xvstelm_d(_tmp1, p0f + out_hstep * 7, 0, 3);
                        p0f += 2;
                    }
                }
            }
            else
            {
                __m128i _bf0 = float2bfloat_lasx(_sum0);
                __m128i _bf1 = float2bfloat_lasx(_sum1);

                if (output_transpose)
                {
                    if (out_elempack == 8)
                    {
                        __lsx_vst(_bf0, p0, 0);
                        __lsx_vst(_bf1, p0 + 8, 0);
                    }
                    if (out_elempack == 4)
                    {
                        __lsx_vstelm_d(_bf0, p0, 0, 0);
                        __lsx_vstelm_d(_bf1, p0 + 4, 0, 0);
                        __lsx_vstelm_d(_bf0, p0 + out_hstep * 4, 0, 1);
                        __lsx_vstelm_d(_bf1, p0 + out_hstep * 4 + 4, 0, 1);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vst(_bf0, p0, 0);
                        __lsx_vst(_bf1, p0 + out_hstep, 0);
                    }
                    p0 += out_hstep * 2;
                }
                else
                {
                    if (out_elempack == 8)
                    {
                        __lsx_vst(_bf0, p0, 0);
                        __lsx_vst(_bf1, p0 + 8, 0);
                        p0 += 16;
                    }
                    if (out_elempack == 4)
                    {
                        unsigned short* p1 = p0 + out_hstep * 4;
                        __lsx_vstelm_d(_bf0, p0, 0, 0);
                        __lsx_vstelm_d(_bf1, p0 + 4, 0, 0);
                        __lsx_vstelm_d(_bf0, p1, 0, 1);
                        __lsx_vstelm_d(_bf1, p1 + 4, 0, 1);
                        p0 += 8;
                    }
                    if (out_elempack == 1)
                    {
                        __m128i _tmp0 = __lsx_vilvl_h(_bf1, _bf0);
                        __m128i _tmp1 = __lsx_vilvh_h(_bf1, _bf0);
                        __lsx_vstelm_w(_tmp0, p0, 0, 0);
                        __lsx_vstelm_w(_tmp0, p0 + out_hstep, 0, 1);
                        __lsx_vstelm_w(_tmp0, p0 + out_hstep * 2, 0, 2);
                        __lsx_vstelm_w(_tmp0, p0 + out_hstep * 3, 0, 3);
                        __lsx_vstelm_w(_tmp1, p0 + out_hstep * 4, 0, 0);
                        __lsx_vstelm_w(_tmp1, p0 + out_hstep * 5, 0, 1);
                        __lsx_vstelm_w(_tmp1, p0 + out_hstep * 6, 0, 2);
                        __lsx_vstelm_w(_tmp1, p0 + out_hstep * 7, 0, 3);
                        p0 += 2;
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
                __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pC0[0]);
                    _sum = __lasx_xvfmadd_s(_c, _beta, _sum);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m256 _c = (__m256)__lasx_xvld(pC0, 0);
                    _sum = __lasx_xvfmadd_s(_c, _beta, _sum);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c = (__m256)__lasx_xvld(pC0, 0);
                    _sum = __lasx_xvfmadd_s(_c, _beta, _sum);
                    pC0 += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pC0[0]);
                    _sum = __lasx_xvfmadd_s(_c, _beta, _sum);
                    pC0 += 1;
                }
            }

            if (alpha != 1.f)
            {
                _sum = __lasx_xvfmul_s(_sum, _valpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 8)
                    {
                        __lasx_xvst(_sum, p0f, 0);
                    }
                    if (out_elempack == 4)
                    {
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum), p0f, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum), p0f + out_hstep * 4, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lasx_xvst(_sum, p0f, 0);
                    }
                    p0f += out_hstep;
                }
                else
                {
                    if (out_elempack == 8)
                    {
                        __lasx_xvst(_sum, p0f, 0);
                        p0f += 8;
                    }
                    if (out_elempack == 4)
                    {
                        __lsx_vst(__lasx_extract_128_lo((__m256i)_sum), p0f, 0);
                        __lsx_vst(__lasx_extract_128_hi((__m256i)_sum), p0f + out_hstep * 4, 0);
                        p0f += 4;
                    }
                    if (out_elempack == 1)
                    {
                        __lasx_xvstelm_w((__m256i)_sum, p0f, 0, 0);
                        __lasx_xvstelm_w((__m256i)_sum, p0f + out_hstep, 0, 1);
                        __lasx_xvstelm_w((__m256i)_sum, p0f + out_hstep * 2, 0, 2);
                        __lasx_xvstelm_w((__m256i)_sum, p0f + out_hstep * 3, 0, 3);
                        __lasx_xvstelm_w((__m256i)_sum, p0f + out_hstep * 4, 0, 4);
                        __lasx_xvstelm_w((__m256i)_sum, p0f + out_hstep * 5, 0, 5);
                        __lasx_xvstelm_w((__m256i)_sum, p0f + out_hstep * 6, 0, 6);
                        __lasx_xvstelm_w((__m256i)_sum, p0f + out_hstep * 7, 0, 7);
                        p0f++;
                    }
                }
            }
            else
            {
                __m128i _bf = float2bfloat_lasx(_sum);
                if (output_transpose)
                {
                    if (out_elempack == 8)
                    {
                        __lsx_vst(_bf, p0, 0);
                    }
                    if (out_elempack == 4)
                    {
                        __lsx_vstelm_d(_bf, p0, 0, 0);
                        __lsx_vstelm_d(_bf, p0 + out_hstep * 4, 0, 1);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vst(_bf, p0, 0);
                    }
                    p0 += out_hstep;
                }
                else
                {
                    if (out_elempack == 8)
                    {
                        __lsx_vst(_bf, p0, 0);
                        p0 += 8;
                    }
                    if (out_elempack == 4)
                    {
                        __lsx_vstelm_d(_bf, p0, 0, 0);
                        __lsx_vstelm_d(_bf, p0 + out_hstep * 4, 0, 1);
                        p0 += 4;
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vstelm_h(_bf, p0, 0, 0);
                        __lsx_vstelm_h(_bf, p0 + out_hstep, 0, 1);
                        __lsx_vstelm_h(_bf, p0 + out_hstep * 2, 0, 2);
                        __lsx_vstelm_h(_bf, p0 + out_hstep * 3, 0, 3);
                        __lsx_vstelm_h(_bf, p0 + out_hstep * 4, 0, 4);
                        __lsx_vstelm_h(_bf, p0 + out_hstep * 5, 0, 5);
                        __lsx_vstelm_h(_bf, p0 + out_hstep * 6, 0, 6);
                        __lsx_vstelm_h(_bf, p0 + out_hstep * 7, 0, 7);
                        p0++;
                    }
                }
            }
        }
#else  // __loongarch_asx
        __m128 _valpha = __lsx_vreplfr2vr_s(alpha);

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

            // deshuffle from the shuffle-based 8x8 kernel
            _sum10 = (__m128)__lsx_vshuf4i_w((__m128i)_sum10, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum30 = (__m128)__lsx_vshuf4i_w((__m128i)_sum30, _LSX_SHUFFLE(2, 1, 0, 3));
            {
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_sum30, (__m128i)_sum00);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_sum30, (__m128i)_sum00);
                __m128 _tmp2 = (__m128)__lsx_vilvl_w((__m128i)_sum10, (__m128i)_sum20);
                __m128 _tmp3 = (__m128)__lsx_vilvh_w((__m128i)_sum10, (__m128i)_sum20);
                _sum00 = (__m128)__lsx_vilvl_d((__m128i)_tmp2, (__m128i)_tmp0);
                _sum10 = (__m128)__lsx_vilvh_d((__m128i)_tmp2, (__m128i)_tmp0);
                _sum20 = (__m128)__lsx_vilvl_d((__m128i)_tmp1, (__m128i)_tmp3);
                _sum30 = (__m128)__lsx_vilvh_d((__m128i)_tmp1, (__m128i)_tmp3);
            }
            _sum10 = (__m128)__lsx_vshuf4i_w((__m128i)_sum10, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum30 = (__m128)__lsx_vshuf4i_w((__m128i)_sum30, _LSX_SHUFFLE(2, 1, 0, 3));

            _sum50 = (__m128)__lsx_vshuf4i_w((__m128i)_sum50, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum70 = (__m128)__lsx_vshuf4i_w((__m128i)_sum70, _LSX_SHUFFLE(2, 1, 0, 3));
            {
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_sum70, (__m128i)_sum40);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_sum70, (__m128i)_sum40);
                __m128 _tmp2 = (__m128)__lsx_vilvl_w((__m128i)_sum50, (__m128i)_sum60);
                __m128 _tmp3 = (__m128)__lsx_vilvh_w((__m128i)_sum50, (__m128i)_sum60);
                _sum40 = (__m128)__lsx_vilvl_d((__m128i)_tmp2, (__m128i)_tmp0);
                _sum50 = (__m128)__lsx_vilvh_d((__m128i)_tmp2, (__m128i)_tmp0);
                _sum60 = (__m128)__lsx_vilvl_d((__m128i)_tmp1, (__m128i)_tmp3);
                _sum70 = (__m128)__lsx_vilvh_d((__m128i)_tmp1, (__m128i)_tmp3);
            }
            _sum50 = (__m128)__lsx_vshuf4i_w((__m128i)_sum50, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum70 = (__m128)__lsx_vshuf4i_w((__m128i)_sum70, _LSX_SHUFFLE(2, 1, 0, 3));

            _sum11 = (__m128)__lsx_vshuf4i_w((__m128i)_sum11, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum31 = (__m128)__lsx_vshuf4i_w((__m128i)_sum31, _LSX_SHUFFLE(2, 1, 0, 3));
            {
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_sum31, (__m128i)_sum01);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_sum31, (__m128i)_sum01);
                __m128 _tmp2 = (__m128)__lsx_vilvl_w((__m128i)_sum11, (__m128i)_sum21);
                __m128 _tmp3 = (__m128)__lsx_vilvh_w((__m128i)_sum11, (__m128i)_sum21);
                _sum01 = (__m128)__lsx_vilvl_d((__m128i)_tmp2, (__m128i)_tmp0);
                _sum11 = (__m128)__lsx_vilvh_d((__m128i)_tmp2, (__m128i)_tmp0);
                _sum21 = (__m128)__lsx_vilvl_d((__m128i)_tmp1, (__m128i)_tmp3);
                _sum31 = (__m128)__lsx_vilvh_d((__m128i)_tmp1, (__m128i)_tmp3);
            }
            _sum11 = (__m128)__lsx_vshuf4i_w((__m128i)_sum11, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum31 = (__m128)__lsx_vshuf4i_w((__m128i)_sum31, _LSX_SHUFFLE(2, 1, 0, 3));

            _sum51 = (__m128)__lsx_vshuf4i_w((__m128i)_sum51, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum71 = (__m128)__lsx_vshuf4i_w((__m128i)_sum71, _LSX_SHUFFLE(2, 1, 0, 3));
            {
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_sum71, (__m128i)_sum41);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_sum71, (__m128i)_sum41);
                __m128 _tmp2 = (__m128)__lsx_vilvl_w((__m128i)_sum51, (__m128i)_sum61);
                __m128 _tmp3 = (__m128)__lsx_vilvh_w((__m128i)_sum51, (__m128i)_sum61);
                _sum41 = (__m128)__lsx_vilvl_d((__m128i)_tmp2, (__m128i)_tmp0);
                _sum51 = (__m128)__lsx_vilvh_d((__m128i)_tmp2, (__m128i)_tmp0);
                _sum61 = (__m128)__lsx_vilvl_d((__m128i)_tmp1, (__m128i)_tmp3);
                _sum71 = (__m128)__lsx_vilvh_d((__m128i)_tmp1, (__m128i)_tmp3);
            }
            _sum51 = (__m128)__lsx_vshuf4i_w((__m128i)_sum51, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum71 = (__m128)__lsx_vshuf4i_w((__m128i)_sum71, _LSX_SHUFFLE(2, 1, 0, 3));

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
                    _sum00 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0, 0), _sum00);
                    _sum01 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 4, 0), _sum01);
                    _sum10 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 8, 0), _sum10);
                    _sum11 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 12, 0), _sum11);
                    _sum20 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 16, 0), _sum20);
                    _sum21 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 20, 0), _sum21);
                    _sum30 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 24, 0), _sum30);
                    _sum31 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 28, 0), _sum31);
                    _sum40 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 32, 0), _sum40);
                    _sum41 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 36, 0), _sum41);
                    _sum50 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 40, 0), _sum50);
                    _sum51 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 44, 0), _sum51);
                    _sum60 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 48, 0), _sum60);
                    _sum61 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 52, 0), _sum61);
                    _sum70 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 56, 0), _sum70);
                    _sum71 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 60, 0), _sum71);
                    pC0 += 64;
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

            if (alpha != 1.f)
            {
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
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        float* p1f = p0f + out_hstep * 4;

                        transpose4x4_ps(_sum00, _sum10, _sum20, _sum30);
                        transpose4x4_ps(_sum40, _sum50, _sum60, _sum70);
                        transpose4x4_ps(_sum01, _sum11, _sum21, _sum31);
                        transpose4x4_ps(_sum41, _sum51, _sum61, _sum71);

                        __lsx_vst((__m128i)_sum00, p0f, 0);
                        __lsx_vst((__m128i)_sum10, p0f + 4, 0);
                        __lsx_vst((__m128i)_sum20, p0f + 8, 0);
                        __lsx_vst((__m128i)_sum30, p0f + 12, 0);
                        __lsx_vst((__m128i)_sum01, p0f + 16, 0);
                        __lsx_vst((__m128i)_sum11, p0f + 20, 0);
                        __lsx_vst((__m128i)_sum21, p0f + 24, 0);
                        __lsx_vst((__m128i)_sum31, p0f + 28, 0);

                        __lsx_vst((__m128i)_sum40, p1f, 0);
                        __lsx_vst((__m128i)_sum50, p1f + 4, 0);
                        __lsx_vst((__m128i)_sum60, p1f + 8, 0);
                        __lsx_vst((__m128i)_sum70, p1f + 12, 0);
                        __lsx_vst((__m128i)_sum41, p1f + 16, 0);
                        __lsx_vst((__m128i)_sum51, p1f + 20, 0);
                        __lsx_vst((__m128i)_sum61, p1f + 24, 0);
                        __lsx_vst((__m128i)_sum71, p1f + 28, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vst((__m128i)_sum00, p0f, 0);
                        __lsx_vst((__m128i)_sum01, p0f + 4, 0);
                        __lsx_vst((__m128i)_sum10, p0f + out_hstep, 0);
                        __lsx_vst((__m128i)_sum11, p0f + out_hstep + 4, 0);
                        __lsx_vst((__m128i)_sum20, p0f + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_sum21, p0f + out_hstep * 2 + 4, 0);
                        __lsx_vst((__m128i)_sum30, p0f + out_hstep * 3, 0);
                        __lsx_vst((__m128i)_sum31, p0f + out_hstep * 3 + 4, 0);
                        __lsx_vst((__m128i)_sum40, p0f + out_hstep * 4, 0);
                        __lsx_vst((__m128i)_sum41, p0f + out_hstep * 4 + 4, 0);
                        __lsx_vst((__m128i)_sum50, p0f + out_hstep * 5, 0);
                        __lsx_vst((__m128i)_sum51, p0f + out_hstep * 5 + 4, 0);
                        __lsx_vst((__m128i)_sum60, p0f + out_hstep * 6, 0);
                        __lsx_vst((__m128i)_sum61, p0f + out_hstep * 6 + 4, 0);
                        __lsx_vst((__m128i)_sum70, p0f + out_hstep * 7, 0);
                        __lsx_vst((__m128i)_sum71, p0f + out_hstep * 7 + 4, 0);
                    }
                    p0f += out_hstep * 8;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        float* p1f = p0f + out_hstep * 4;
                        __lsx_vst((__m128i)_sum00, p0f, 0);
                        __lsx_vst((__m128i)_sum01, p1f, 0);
                        __lsx_vst((__m128i)_sum10, p0f + 4, 0);
                        __lsx_vst((__m128i)_sum11, p1f + 4, 0);
                        __lsx_vst((__m128i)_sum20, p0f + 8, 0);
                        __lsx_vst((__m128i)_sum21, p1f + 8, 0);
                        __lsx_vst((__m128i)_sum30, p0f + 12, 0);
                        __lsx_vst((__m128i)_sum31, p1f + 12, 0);
                        __lsx_vst((__m128i)_sum40, p0f + 16, 0);
                        __lsx_vst((__m128i)_sum41, p1f + 16, 0);
                        __lsx_vst((__m128i)_sum50, p0f + 20, 0);
                        __lsx_vst((__m128i)_sum51, p1f + 20, 0);
                        __lsx_vst((__m128i)_sum60, p0f + 24, 0);
                        __lsx_vst((__m128i)_sum61, p1f + 24, 0);
                        __lsx_vst((__m128i)_sum70, p0f + 28, 0);
                        __lsx_vst((__m128i)_sum71, p1f + 28, 0);
                        p0f += 32;
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

                        __lsx_vst((__m128i)_r0, p0f, 0);
                        __lsx_vst((__m128i)_r4, p0f + 4, 0);
                        __lsx_vst((__m128i)_r1, p0f + out_hstep, 0);
                        __lsx_vst((__m128i)_r5, p0f + out_hstep + 4, 0);
                        __lsx_vst((__m128i)_r2, p0f + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_r6, p0f + out_hstep * 2 + 4, 0);
                        __lsx_vst((__m128i)_r3, p0f + out_hstep * 3, 0);
                        __lsx_vst((__m128i)_r7, p0f + out_hstep * 3 + 4, 0);
                        __lsx_vst((__m128i)_r8, p0f + out_hstep * 4, 0);
                        __lsx_vst((__m128i)_rc, p0f + out_hstep * 4 + 4, 0);
                        __lsx_vst((__m128i)_r9, p0f + out_hstep * 5, 0);
                        __lsx_vst((__m128i)_rd, p0f + out_hstep * 5 + 4, 0);
                        __lsx_vst((__m128i)_ra, p0f + out_hstep * 6, 0);
                        __lsx_vst((__m128i)_re, p0f + out_hstep * 6 + 4, 0);
                        __lsx_vst((__m128i)_rb, p0f + out_hstep * 7, 0);
                        __lsx_vst((__m128i)_rf, p0f + out_hstep * 7 + 4, 0);
                        p0f += 8;
                    }
                }
            }
            else
            {
                if (output_transpose)
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
                else
                {
                    if (out_elempack == 4)
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

            // deshuffle from the shuffle-based 8x4 kernel
            _sum10 = (__m128)__lsx_vshuf4i_w((__m128i)_sum10, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum30 = (__m128)__lsx_vshuf4i_w((__m128i)_sum30, _LSX_SHUFFLE(2, 1, 0, 3));
            {
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_sum30, (__m128i)_sum00);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_sum30, (__m128i)_sum00);
                __m128 _tmp2 = (__m128)__lsx_vilvl_w((__m128i)_sum10, (__m128i)_sum20);
                __m128 _tmp3 = (__m128)__lsx_vilvh_w((__m128i)_sum10, (__m128i)_sum20);
                _sum00 = (__m128)__lsx_vilvl_d((__m128i)_tmp2, (__m128i)_tmp0);
                _sum10 = (__m128)__lsx_vilvh_d((__m128i)_tmp2, (__m128i)_tmp0);
                _sum20 = (__m128)__lsx_vilvl_d((__m128i)_tmp1, (__m128i)_tmp3);
                _sum30 = (__m128)__lsx_vilvh_d((__m128i)_tmp1, (__m128i)_tmp3);
            }
            _sum10 = (__m128)__lsx_vshuf4i_w((__m128i)_sum10, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum30 = (__m128)__lsx_vshuf4i_w((__m128i)_sum30, _LSX_SHUFFLE(2, 1, 0, 3));

            _sum11 = (__m128)__lsx_vshuf4i_w((__m128i)_sum11, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum31 = (__m128)__lsx_vshuf4i_w((__m128i)_sum31, _LSX_SHUFFLE(2, 1, 0, 3));
            {
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_sum31, (__m128i)_sum01);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_sum31, (__m128i)_sum01);
                __m128 _tmp2 = (__m128)__lsx_vilvl_w((__m128i)_sum11, (__m128i)_sum21);
                __m128 _tmp3 = (__m128)__lsx_vilvh_w((__m128i)_sum11, (__m128i)_sum21);
                _sum01 = (__m128)__lsx_vilvl_d((__m128i)_tmp2, (__m128i)_tmp0);
                _sum11 = (__m128)__lsx_vilvh_d((__m128i)_tmp2, (__m128i)_tmp0);
                _sum21 = (__m128)__lsx_vilvl_d((__m128i)_tmp1, (__m128i)_tmp3);
                _sum31 = (__m128)__lsx_vilvh_d((__m128i)_tmp1, (__m128i)_tmp3);
            }
            _sum11 = (__m128)__lsx_vshuf4i_w((__m128i)_sum11, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum31 = (__m128)__lsx_vshuf4i_w((__m128i)_sum31, _LSX_SHUFFLE(2, 1, 0, 3));

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
                    _sum00 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0, 0), _sum00);
                    _sum01 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 4, 0), _sum01);
                    _sum10 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 8, 0), _sum10);
                    _sum11 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 12, 0), _sum11);
                    _sum20 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 16, 0), _sum20);
                    _sum21 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 20, 0), _sum21);
                    _sum30 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 24, 0), _sum30);
                    _sum31 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 28, 0), _sum31);
                    pC0 += 32;
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

            if (alpha != 1.f)
            {
                _sum00 = __lsx_vfmul_s(_sum00, _valpha);
                _sum01 = __lsx_vfmul_s(_sum01, _valpha);
                _sum10 = __lsx_vfmul_s(_sum10, _valpha);
                _sum11 = __lsx_vfmul_s(_sum11, _valpha);
                _sum20 = __lsx_vfmul_s(_sum20, _valpha);
                _sum21 = __lsx_vfmul_s(_sum21, _valpha);
                _sum30 = __lsx_vfmul_s(_sum30, _valpha);
                _sum31 = __lsx_vfmul_s(_sum31, _valpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        transpose4x4_ps(_sum00, _sum10, _sum20, _sum30);
                        transpose4x4_ps(_sum01, _sum11, _sum21, _sum31);

                        __lsx_vst((__m128i)_sum00, p0f, 0);
                        __lsx_vst((__m128i)_sum10, p0f + 4, 0);
                        __lsx_vst((__m128i)_sum20, p0f + 4 * 2, 0);
                        __lsx_vst((__m128i)_sum30, p0f + 4 * 3, 0);
                        __lsx_vst((__m128i)_sum01, p0f + 4 * 4, 0);
                        __lsx_vst((__m128i)_sum11, p0f + 4 * 5, 0);
                        __lsx_vst((__m128i)_sum21, p0f + 4 * 6, 0);
                        __lsx_vst((__m128i)_sum31, p0f + 4 * 7, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vst((__m128i)_sum00, p0f, 0);
                        __lsx_vst((__m128i)_sum01, p0f + 4, 0);
                        __lsx_vst((__m128i)_sum10, p0f + out_hstep, 0);
                        __lsx_vst((__m128i)_sum11, p0f + out_hstep + 4, 0);
                        __lsx_vst((__m128i)_sum20, p0f + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_sum21, p0f + out_hstep * 2 + 4, 0);
                        __lsx_vst((__m128i)_sum30, p0f + out_hstep * 3, 0);
                        __lsx_vst((__m128i)_sum31, p0f + out_hstep * 3 + 4, 0);
                    }
                    p0f += out_hstep * 4;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        float* p1f = p0f + out_hstep * 4;
                        __lsx_vst((__m128i)_sum00, p0f, 0);
                        __lsx_vst((__m128i)_sum01, p1f, 0);
                        __lsx_vst((__m128i)_sum10, p0f + 4, 0);
                        __lsx_vst((__m128i)_sum11, p1f + 4, 0);
                        __lsx_vst((__m128i)_sum20, p0f + 8, 0);
                        __lsx_vst((__m128i)_sum21, p1f + 8, 0);
                        __lsx_vst((__m128i)_sum30, p0f + 12, 0);
                        __lsx_vst((__m128i)_sum31, p1f + 12, 0);
                        p0f += 16;
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

                        __lsx_vst((__m128i)_r0, p0f, 0);
                        __lsx_vst((__m128i)_r1, p0f + out_hstep, 0);
                        __lsx_vst((__m128i)_r2, p0f + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_r3, p0f + out_hstep * 3, 0);
                        __lsx_vst((__m128i)_r4, p0f + out_hstep * 4, 0);
                        __lsx_vst((__m128i)_r5, p0f + out_hstep * 5, 0);
                        __lsx_vst((__m128i)_r6, p0f + out_hstep * 6, 0);
                        __lsx_vst((__m128i)_r7, p0f + out_hstep * 7, 0);
                        p0f += 4;
                    }
                }
            }
            else
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        transpose4x4_ps(_sum00, _sum10, _sum20, _sum30);
                        transpose4x4_ps(_sum01, _sum11, _sum21, _sum31);

                        __lsx_vstelm_d(float2bfloat_lsx(_sum00), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum10), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum20), p0 + 4 * 2, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum30), p0 + 4 * 3, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum01), p0 + 4 * 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum11), p0 + 4 * 5, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum21), p0 + 4 * 6, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum31), p0 + 4 * 7, 0, 0);
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
                    }
                    p0 += out_hstep * 4;
                }
                else
                {
                    if (out_elempack == 4)
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

            // deshuffle from the shuffle-based 8x2 kernel
            {
                __m128 _tmp0 = (__m128)__lsx_vshuf4i_w((__m128i)_sum00, _LSX_SHUFFLE(3, 1, 2, 0));
                __m128 _tmp1 = (__m128)__lsx_vshuf4i_w((__m128i)_sum10, _LSX_SHUFFLE(0, 2, 3, 1));
                _sum00 = (__m128)__lsx_vilvl_w((__m128i)_tmp1, (__m128i)_tmp0);
                _sum10 = (__m128)__lsx_vilvh_w((__m128i)_tmp1, (__m128i)_tmp0);
                _sum10 = (__m128)__lsx_vshuf4i_w((__m128i)_sum10, _LSX_SHUFFLE(2, 1, 0, 3));
            }
            {
                __m128 _tmp0 = (__m128)__lsx_vshuf4i_w((__m128i)_sum01, _LSX_SHUFFLE(3, 1, 2, 0));
                __m128 _tmp1 = (__m128)__lsx_vshuf4i_w((__m128i)_sum11, _LSX_SHUFFLE(0, 2, 3, 1));
                _sum01 = (__m128)__lsx_vilvl_w((__m128i)_tmp1, (__m128i)_tmp0);
                _sum11 = (__m128)__lsx_vilvh_w((__m128i)_tmp1, (__m128i)_tmp0);
                _sum11 = (__m128)__lsx_vshuf4i_w((__m128i)_sum11, _LSX_SHUFFLE(2, 1, 0, 3));
            }

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
                    _sum00 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0, 0), _sum00);
                    _sum01 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 4, 0), _sum01);
                    _sum10 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 8, 0), _sum10);
                    _sum11 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 12, 0), _sum11);
                    pC0 += 16;
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

            if (alpha != 1.f)
            {
                _sum00 = __lsx_vfmul_s(_sum00, _valpha);
                _sum01 = __lsx_vfmul_s(_sum01, _valpha);
                _sum10 = __lsx_vfmul_s(_sum10, _valpha);
                _sum11 = __lsx_vfmul_s(_sum11, _valpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        __lsx_vstelm_w((__m128i)_sum00, p0f, 0, 0);
                        __lsx_vstelm_w((__m128i)_sum10, p0f + 1, 0, 0);
                        __lsx_vstelm_w((__m128i)_sum00, p0f + 4, 0, 1);
                        __lsx_vstelm_w((__m128i)_sum10, p0f + 5, 0, 1);
                        __lsx_vstelm_w((__m128i)_sum00, p0f + 8, 0, 2);
                        __lsx_vstelm_w((__m128i)_sum10, p0f + 9, 0, 2);
                        __lsx_vstelm_w((__m128i)_sum00, p0f + 12, 0, 3);
                        __lsx_vstelm_w((__m128i)_sum10, p0f + 13, 0, 3);
                        __lsx_vstelm_w((__m128i)_sum01, p0f + 16, 0, 0);
                        __lsx_vstelm_w((__m128i)_sum11, p0f + 17, 0, 0);
                        __lsx_vstelm_w((__m128i)_sum01, p0f + 20, 0, 1);
                        __lsx_vstelm_w((__m128i)_sum11, p0f + 21, 0, 1);
                        __lsx_vstelm_w((__m128i)_sum01, p0f + 24, 0, 2);
                        __lsx_vstelm_w((__m128i)_sum11, p0f + 25, 0, 2);
                        __lsx_vstelm_w((__m128i)_sum01, p0f + 28, 0, 3);
                        __lsx_vstelm_w((__m128i)_sum11, p0f + 29, 0, 3);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vst((__m128i)_sum00, p0f, 0);
                        __lsx_vst((__m128i)_sum01, p0f + 4, 0);
                        __lsx_vst((__m128i)_sum10, p0f + out_hstep, 0);
                        __lsx_vst((__m128i)_sum11, p0f + out_hstep + 4, 0);
                    }
                    p0f += out_hstep * 2;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        float* p1f = p0f + out_hstep * 4;
                        __lsx_vst((__m128i)_sum00, p0f, 0);
                        __lsx_vst((__m128i)_sum01, p1f, 0);
                        __lsx_vst((__m128i)_sum10, p0f + 4, 0);
                        __lsx_vst((__m128i)_sum11, p1f + 4, 0);
                        p0f += 8;
                    }
                    if (out_elempack == 1)
                    {
                        __m128i _tmp0 = __lsx_vilvl_w((__m128i)_sum10, (__m128i)_sum00);
                        __m128i _tmp1 = __lsx_vilvh_w((__m128i)_sum10, (__m128i)_sum00);
                        __m128i _tmp2 = __lsx_vilvl_w((__m128i)_sum11, (__m128i)_sum01);
                        __m128i _tmp3 = __lsx_vilvh_w((__m128i)_sum11, (__m128i)_sum01);
                        __lsx_vstelm_d(_tmp0, p0f, 0, 0);
                        __lsx_vstelm_d(_tmp0, p0f + out_hstep, 0, 1);
                        __lsx_vstelm_d(_tmp1, p0f + out_hstep * 2, 0, 0);
                        __lsx_vstelm_d(_tmp1, p0f + out_hstep * 3, 0, 1);
                        __lsx_vstelm_d(_tmp2, p0f + out_hstep * 4, 0, 0);
                        __lsx_vstelm_d(_tmp2, p0f + out_hstep * 5, 0, 1);
                        __lsx_vstelm_d(_tmp3, p0f + out_hstep * 6, 0, 0);
                        __lsx_vstelm_d(_tmp3, p0f + out_hstep * 7, 0, 1);
                        p0f += 2;
                    }
                }
            }
            else
            {
                __m128i _bf00 = float2bfloat_lsx(_sum00);
                __m128i _bf01 = float2bfloat_lsx(_sum01);
                __m128i _bf10 = float2bfloat_lsx(_sum10);
                __m128i _bf11 = float2bfloat_lsx(_sum11);

                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        __lsx_vstelm_h(_bf00, p0, 0, 0);
                        __lsx_vstelm_h(_bf10, p0 + 1, 0, 0);
                        __lsx_vstelm_h(_bf00, p0 + 4, 0, 1);
                        __lsx_vstelm_h(_bf10, p0 + 5, 0, 1);
                        __lsx_vstelm_h(_bf00, p0 + 8, 0, 2);
                        __lsx_vstelm_h(_bf10, p0 + 9, 0, 2);
                        __lsx_vstelm_h(_bf00, p0 + 12, 0, 3);
                        __lsx_vstelm_h(_bf10, p0 + 13, 0, 3);
                        __lsx_vstelm_h(_bf01, p0 + 16, 0, 0);
                        __lsx_vstelm_h(_bf11, p0 + 17, 0, 0);
                        __lsx_vstelm_h(_bf01, p0 + 20, 0, 1);
                        __lsx_vstelm_h(_bf11, p0 + 21, 0, 1);
                        __lsx_vstelm_h(_bf01, p0 + 24, 0, 2);
                        __lsx_vstelm_h(_bf11, p0 + 25, 0, 2);
                        __lsx_vstelm_h(_bf01, p0 + 28, 0, 3);
                        __lsx_vstelm_h(_bf11, p0 + 29, 0, 3);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vstelm_d(_bf00, p0, 0, 0);
                        __lsx_vstelm_d(_bf01, p0 + 4, 0, 0);
                        __lsx_vstelm_d(_bf10, p0 + out_hstep, 0, 0);
                        __lsx_vstelm_d(_bf11, p0 + out_hstep + 4, 0, 0);
                    }
                    p0 += out_hstep * 2;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        unsigned short* p1 = p0 + out_hstep * 4;
                        __lsx_vstelm_d(_bf00, p0, 0, 0);
                        __lsx_vstelm_d(_bf01, p1, 0, 0);
                        __lsx_vstelm_d(_bf10, p0 + 4, 0, 0);
                        __lsx_vstelm_d(_bf11, p1 + 4, 0, 0);
                        p0 += 8;
                    }
                    if (out_elempack == 1)
                    {
                        __m128i _tmp0 = __lsx_vilvl_h(_bf10, _bf00);
                        __m128i _tmp2 = __lsx_vilvl_h(_bf11, _bf01);
                        __lsx_vstelm_w(_tmp0, p0, 0, 0);
                        __lsx_vstelm_w(_tmp0, p0 + out_hstep, 0, 1);
                        __lsx_vstelm_w(_tmp0, p0 + out_hstep * 2, 0, 2);
                        __lsx_vstelm_w(_tmp0, p0 + out_hstep * 3, 0, 3);
                        __lsx_vstelm_w(_tmp2, p0 + out_hstep * 4, 0, 0);
                        __lsx_vstelm_w(_tmp2, p0 + out_hstep * 5, 0, 1);
                        __lsx_vstelm_w(_tmp2, p0 + out_hstep * 6, 0, 2);
                        __lsx_vstelm_w(_tmp2, p0 + out_hstep * 7, 0, 3);
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
                    _sum0 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0, 0), _sum0);
                    _sum1 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 4, 0), _sum1);
                    pC0 += 8;
                }
                if (broadcast_type_C == 4)
                {
                    __m128 _c = __lsx_vreplfr2vr_s(pC0[0]);
                    _sum0 = __lsx_vfmadd_s(_beta, _c, _sum0);
                    _sum1 = __lsx_vfmadd_s(_beta, _c, _sum1);
                    pC0 += 1;
                }
            }

            if (alpha != 1.f)
            {
                _sum0 = __lsx_vfmul_s(_sum0, _valpha);
                _sum1 = __lsx_vfmul_s(_sum1, _valpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        __lsx_vstelm_w((__m128i)_sum0, p0f, 0, 0);
                        __lsx_vstelm_w((__m128i)_sum0, p0f + 4, 0, 1);
                        __lsx_vstelm_w((__m128i)_sum0, p0f + 8, 0, 2);
                        __lsx_vstelm_w((__m128i)_sum0, p0f + 12, 0, 3);
                        __lsx_vstelm_w((__m128i)_sum1, p0f + 16, 0, 0);
                        __lsx_vstelm_w((__m128i)_sum1, p0f + 20, 0, 1);
                        __lsx_vstelm_w((__m128i)_sum1, p0f + 24, 0, 2);
                        __lsx_vstelm_w((__m128i)_sum1, p0f + 28, 0, 3);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum1, p0f + 4, 0);
                    }
                    p0f += out_hstep;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        float* p1f = p0f + out_hstep * 4;
                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum1, p1f, 0);
                        p0f += 4;
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vstelm_w((__m128i)_sum0, p0f, 0, 0);
                        __lsx_vstelm_w((__m128i)_sum0, p0f + out_hstep, 0, 1);
                        __lsx_vstelm_w((__m128i)_sum0, p0f + out_hstep * 2, 0, 2);
                        __lsx_vstelm_w((__m128i)_sum0, p0f + out_hstep * 3, 0, 3);
                        __lsx_vstelm_w((__m128i)_sum1, p0f + out_hstep * 4, 0, 0);
                        __lsx_vstelm_w((__m128i)_sum1, p0f + out_hstep * 5, 0, 1);
                        __lsx_vstelm_w((__m128i)_sum1, p0f + out_hstep * 6, 0, 2);
                        __lsx_vstelm_w((__m128i)_sum1, p0f + out_hstep * 7, 0, 3);
                        p0f++;
                    }
                }
            }
            else
            {
                __m128i _bf0 = float2bfloat_lsx(_sum0);
                __m128i _bf1 = float2bfloat_lsx(_sum1);

                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        __lsx_vstelm_h(_bf0, p0, 0, 0);
                        __lsx_vstelm_h(_bf0, p0 + 4, 0, 1);
                        __lsx_vstelm_h(_bf0, p0 + 8, 0, 2);
                        __lsx_vstelm_h(_bf0, p0 + 12, 0, 3);
                        __lsx_vstelm_h(_bf1, p0 + 16, 0, 0);
                        __lsx_vstelm_h(_bf1, p0 + 20, 0, 1);
                        __lsx_vstelm_h(_bf1, p0 + 24, 0, 2);
                        __lsx_vstelm_h(_bf1, p0 + 28, 0, 3);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vstelm_d(_bf0, p0, 0, 0);
                        __lsx_vstelm_d(_bf1, p0 + 4, 0, 0);
                    }
                    p0 += out_hstep;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        unsigned short* p1 = p0 + out_hstep * 4;
                        __lsx_vstelm_d(_bf0, p0, 0, 0);
                        __lsx_vstelm_d(_bf1, p1, 0, 0);
                        p0 += 4;
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vstelm_h(_bf0, p0, 0, 0);
                        __lsx_vstelm_h(_bf0, p0 + out_hstep, 0, 1);
                        __lsx_vstelm_h(_bf0, p0 + out_hstep * 2, 0, 2);
                        __lsx_vstelm_h(_bf0, p0 + out_hstep * 3, 0, 3);
                        __lsx_vstelm_h(_bf1, p0 + out_hstep * 4, 0, 0);
                        __lsx_vstelm_h(_bf1, p0 + out_hstep * 5, 0, 1);
                        __lsx_vstelm_h(_bf1, p0 + out_hstep * 6, 0, 2);
                        __lsx_vstelm_h(_bf1, p0 + out_hstep * 7, 0, 3);
                        p0++;
                    }
                }
            }
        }
#endif // __loongarch_asx
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        unsigned short* p0;
        float* p0f;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
            p0f = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
            p0f = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        const float* pC0 = pC;
        if (pC0)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC0 = (const float*)C + (i + ii);
            }
            if (broadcast_type_C == 3)
            {
                pC += max_jj * 4;
            }
            if (broadcast_type_C == 4)
            {
                pC0 = (const float*)C + j;
            }
        }

        __m128 _valpha = __lsx_vreplfr2vr_s(alpha);

        int jj = 0;
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _sum04 = (__m256)__lasx_xvld(pp, 0);
            __m256 _sum15 = (__m256)__lasx_xvld(pp + 8, 0);
            __m256 _sum26 = (__m256)__lasx_xvld(pp + 16, 0);
            __m256 _sum37 = (__m256)__lasx_xvld(pp + 24, 0);
            __m256 _sum8c = (__m256)__lasx_xvld(pp + 32, 0);
            __m256 _sum9d = (__m256)__lasx_xvld(pp + 40, 0);
            __m256 _sumae = (__m256)__lasx_xvld(pp + 48, 0);
            __m256 _sumbf = (__m256)__lasx_xvld(pp + 56, 0);
            pp += 64;

            _sum15 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum15, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum37 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum37, _LSX_SHUFFLE(2, 1, 0, 3));
            {
                __m256 _tmp0 = (__m256)__lasx_xvilvl_w((__m256i)_sum37, (__m256i)_sum04);
                __m256 _tmp1 = (__m256)__lasx_xvilvh_w((__m256i)_sum37, (__m256i)_sum04);
                __m256 _tmp2 = (__m256)__lasx_xvilvl_w((__m256i)_sum15, (__m256i)_sum26);
                __m256 _tmp3 = (__m256)__lasx_xvilvh_w((__m256i)_sum15, (__m256i)_sum26);
                _sum04 = (__m256)__lasx_xvilvl_d((__m256i)_tmp2, (__m256i)_tmp0);
                _sum15 = (__m256)__lasx_xvilvh_d((__m256i)_tmp2, (__m256i)_tmp0);
                _sum26 = (__m256)__lasx_xvilvl_d((__m256i)_tmp1, (__m256i)_tmp3);
                _sum37 = (__m256)__lasx_xvilvh_d((__m256i)_tmp1, (__m256i)_tmp3);
            }
            _sum15 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum15, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum37 = (__m256)__lasx_xvshuf4i_w((__m256i)_sum37, _LSX_SHUFFLE(2, 1, 0, 3));

            _sum9d = (__m256)__lasx_xvshuf4i_w((__m256i)_sum9d, _LSX_SHUFFLE(2, 1, 0, 3));
            _sumbf = (__m256)__lasx_xvshuf4i_w((__m256i)_sumbf, _LSX_SHUFFLE(2, 1, 0, 3));
            {
                __m256 _tmp0 = (__m256)__lasx_xvilvl_w((__m256i)_sumbf, (__m256i)_sum8c);
                __m256 _tmp1 = (__m256)__lasx_xvilvh_w((__m256i)_sumbf, (__m256i)_sum8c);
                __m256 _tmp2 = (__m256)__lasx_xvilvl_w((__m256i)_sum9d, (__m256i)_sumae);
                __m256 _tmp3 = (__m256)__lasx_xvilvh_w((__m256i)_sum9d, (__m256i)_sumae);
                _sum8c = (__m256)__lasx_xvilvl_d((__m256i)_tmp2, (__m256i)_tmp0);
                _sum9d = (__m256)__lasx_xvilvh_d((__m256i)_tmp2, (__m256i)_tmp0);
                _sumae = (__m256)__lasx_xvilvl_d((__m256i)_tmp1, (__m256i)_tmp3);
                _sumbf = (__m256)__lasx_xvilvh_d((__m256i)_tmp1, (__m256i)_tmp3);
            }
            _sum9d = (__m256)__lasx_xvshuf4i_w((__m256i)_sum9d, _LSX_SHUFFLE(2, 1, 0, 3));
            _sumbf = (__m256)__lasx_xvshuf4i_w((__m256i)_sumbf, _LSX_SHUFFLE(2, 1, 0, 3));

            if (pC0)
            {
                __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta);
                if (broadcast_type_C == 0)
                {
                    __m256 _c = (__m256)__lasx_xvreplfr2vr_s(pC0[0]);
                    _sum04 = __lasx_xvfmadd_s(_c, _beta, _sum04);
                    _sum15 = __lasx_xvfmadd_s(_c, _beta, _sum15);
                    _sum26 = __lasx_xvfmadd_s(_c, _beta, _sum26);
                    _sum37 = __lasx_xvfmadd_s(_c, _beta, _sum37);
                    _sum8c = __lasx_xvfmadd_s(_c, _beta, _sum8c);
                    _sum9d = __lasx_xvfmadd_s(_c, _beta, _sum9d);
                    _sumae = __lasx_xvfmadd_s(_c, _beta, _sumae);
                    _sumbf = __lasx_xvfmadd_s(_c, _beta, _sumbf);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m128 _c0 = (__m128)__lsx_vld(pC0, 0);
                    __m256 _c = __lasx_concat_128_s(_c0, _c0);
                    _sum04 = __lasx_xvfmadd_s(_c, _beta, _sum04);
                    _sum15 = __lasx_xvfmadd_s(_c, _beta, _sum15);
                    _sum26 = __lasx_xvfmadd_s(_c, _beta, _sum26);
                    _sum37 = __lasx_xvfmadd_s(_c, _beta, _sum37);
                    _sum8c = __lasx_xvfmadd_s(_c, _beta, _sum8c);
                    _sum9d = __lasx_xvfmadd_s(_c, _beta, _sum9d);
                    _sumae = __lasx_xvfmadd_s(_c, _beta, _sumae);
                    _sumbf = __lasx_xvfmadd_s(_c, _beta, _sumbf);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c04 = __lasx_concat_128_s((__m128)__lsx_vld(pC0, 0), (__m128)__lsx_vld(pC0 + 16, 0));
                    __m256 _c15 = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 4, 0), (__m128)__lsx_vld(pC0 + 20, 0));
                    __m256 _c26 = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 8, 0), (__m128)__lsx_vld(pC0 + 24, 0));
                    __m256 _c37 = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 12, 0), (__m128)__lsx_vld(pC0 + 28, 0));
                    __m256 _c8c = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 32, 0), (__m128)__lsx_vld(pC0 + 48, 0));
                    __m256 _c9d = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 36, 0), (__m128)__lsx_vld(pC0 + 52, 0));
                    __m256 _cae = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 40, 0), (__m128)__lsx_vld(pC0 + 56, 0));
                    __m256 _cbf = __lasx_concat_128_s((__m128)__lsx_vld(pC0 + 44, 0), (__m128)__lsx_vld(pC0 + 60, 0));
                    _sum04 = __lasx_xvfmadd_s(_c04, _beta, _sum04);
                    _sum15 = __lasx_xvfmadd_s(_c15, _beta, _sum15);
                    _sum26 = __lasx_xvfmadd_s(_c26, _beta, _sum26);
                    _sum37 = __lasx_xvfmadd_s(_c37, _beta, _sum37);
                    _sum8c = __lasx_xvfmadd_s(_c8c, _beta, _sum8c);
                    _sum9d = __lasx_xvfmadd_s(_c9d, _beta, _sum9d);
                    _sumae = __lasx_xvfmadd_s(_cae, _beta, _sumae);
                    _sumbf = __lasx_xvfmadd_s(_cbf, _beta, _sumbf);
                    pC0 += 64;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c04 = __lasx_concat_128_s(__lsx_vreplfr2vr_s(pC0[0]), __lsx_vreplfr2vr_s(pC0[4]));
                    __m256 _c15 = __lasx_concat_128_s(__lsx_vreplfr2vr_s(pC0[1]), __lsx_vreplfr2vr_s(pC0[5]));
                    __m256 _c26 = __lasx_concat_128_s(__lsx_vreplfr2vr_s(pC0[2]), __lsx_vreplfr2vr_s(pC0[6]));
                    __m256 _c37 = __lasx_concat_128_s(__lsx_vreplfr2vr_s(pC0[3]), __lsx_vreplfr2vr_s(pC0[7]));
                    __m256 _c8c = __lasx_concat_128_s(__lsx_vreplfr2vr_s(pC0[8]), __lsx_vreplfr2vr_s(pC0[12]));
                    __m256 _c9d = __lasx_concat_128_s(__lsx_vreplfr2vr_s(pC0[9]), __lsx_vreplfr2vr_s(pC0[13]));
                    __m256 _cae = __lasx_concat_128_s(__lsx_vreplfr2vr_s(pC0[10]), __lsx_vreplfr2vr_s(pC0[14]));
                    __m256 _cbf = __lasx_concat_128_s(__lsx_vreplfr2vr_s(pC0[11]), __lsx_vreplfr2vr_s(pC0[15]));
                    _sum04 = __lasx_xvfmadd_s(_c04, _beta, _sum04);
                    _sum15 = __lasx_xvfmadd_s(_c15, _beta, _sum15);
                    _sum26 = __lasx_xvfmadd_s(_c26, _beta, _sum26);
                    _sum37 = __lasx_xvfmadd_s(_c37, _beta, _sum37);
                    _sum8c = __lasx_xvfmadd_s(_c8c, _beta, _sum8c);
                    _sum9d = __lasx_xvfmadd_s(_c9d, _beta, _sum9d);
                    _sumae = __lasx_xvfmadd_s(_cae, _beta, _sumae);
                    _sumbf = __lasx_xvfmadd_s(_cbf, _beta, _sumbf);
                    pC0 += 16;
                }
            }

            if (alpha != 1.f)
            {
                __m256 _alpha = (__m256)__lasx_xvreplfr2vr_s(alpha);
                _sum04 = __lasx_xvfmul_s(_sum04, _alpha);
                _sum15 = __lasx_xvfmul_s(_sum15, _alpha);
                _sum26 = __lasx_xvfmul_s(_sum26, _alpha);
                _sum37 = __lasx_xvfmul_s(_sum37, _alpha);
                _sum8c = __lasx_xvfmul_s(_sum8c, _alpha);
                _sum9d = __lasx_xvfmul_s(_sum9d, _alpha);
                _sumae = __lasx_xvfmul_s(_sumae, _alpha);
                _sumbf = __lasx_xvfmul_s(_sumbf, _alpha);
            }

            __m128 _sum0 = __lasx_extract_128_lo_s(_sum04);
            __m128 _sum4 = __lasx_extract_128_hi_s(_sum04);
            __m128 _sum1 = __lasx_extract_128_lo_s(_sum15);
            __m128 _sum5 = __lasx_extract_128_hi_s(_sum15);
            __m128 _sum2 = __lasx_extract_128_lo_s(_sum26);
            __m128 _sum6 = __lasx_extract_128_hi_s(_sum26);
            __m128 _sum3 = __lasx_extract_128_lo_s(_sum37);
            __m128 _sum7 = __lasx_extract_128_hi_s(_sum37);
            __m128 _sum8 = __lasx_extract_128_lo_s(_sum8c);
            __m128 _sumc = __lasx_extract_128_hi_s(_sum8c);
            __m128 _sum9 = __lasx_extract_128_lo_s(_sum9d);
            __m128 _sumd = __lasx_extract_128_hi_s(_sum9d);
            __m128 _suma = __lasx_extract_128_lo_s(_sumae);
            __m128 _sume = __lasx_extract_128_hi_s(_sumae);
            __m128 _sumb = __lasx_extract_128_lo_s(_sumbf);
            __m128 _sumf = __lasx_extract_128_hi_s(_sumbf);

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 8)
                    {
                        float* p1f = p0f + out_hstep * 8;
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);
                        transpose4x4_ps(_sum8, _sum9, _suma, _sumb);
                        transpose4x4_ps(_sumc, _sumd, _sume, _sumf);

                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum4, p0f + 4, 0);
                        __lsx_vst((__m128i)_sum1, p0f + 8, 0);
                        __lsx_vst((__m128i)_sum5, p0f + 12, 0);
                        __lsx_vst((__m128i)_sum2, p0f + 16, 0);
                        __lsx_vst((__m128i)_sum6, p0f + 20, 0);
                        __lsx_vst((__m128i)_sum3, p0f + 24, 0);
                        __lsx_vst((__m128i)_sum7, p0f + 28, 0);
                        __lsx_vst((__m128i)_sum8, p1f, 0);
                        __lsx_vst((__m128i)_sumc, p1f + 4, 0);
                        __lsx_vst((__m128i)_sum9, p1f + 8, 0);
                        __lsx_vst((__m128i)_sumd, p1f + 12, 0);
                        __lsx_vst((__m128i)_suma, p1f + 16, 0);
                        __lsx_vst((__m128i)_sume, p1f + 20, 0);
                        __lsx_vst((__m128i)_sumb, p1f + 24, 0);
                        __lsx_vst((__m128i)_sumf, p1f + 28, 0);
                    }
                    if (out_elempack == 4)
                    {
                        float* p1f = p0f + out_hstep * 4;
                        float* p2f = p0f + out_hstep * 8;
                        float* p3f = p0f + out_hstep * 12;
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);
                        transpose4x4_ps(_sum8, _sum9, _suma, _sumb);
                        transpose4x4_ps(_sumc, _sumd, _sume, _sumf);

                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum1, p0f + 4, 0);
                        __lsx_vst((__m128i)_sum2, p0f + 8, 0);
                        __lsx_vst((__m128i)_sum3, p0f + 12, 0);
                        __lsx_vst((__m128i)_sum4, p1f, 0);
                        __lsx_vst((__m128i)_sum5, p1f + 4, 0);
                        __lsx_vst((__m128i)_sum6, p1f + 8, 0);
                        __lsx_vst((__m128i)_sum7, p1f + 12, 0);
                        __lsx_vst((__m128i)_sum8, p2f, 0);
                        __lsx_vst((__m128i)_sum9, p2f + 4, 0);
                        __lsx_vst((__m128i)_suma, p2f + 8, 0);
                        __lsx_vst((__m128i)_sumb, p2f + 12, 0);
                        __lsx_vst((__m128i)_sumc, p3f, 0);
                        __lsx_vst((__m128i)_sumd, p3f + 4, 0);
                        __lsx_vst((__m128i)_sume, p3f + 8, 0);
                        __lsx_vst((__m128i)_sumf, p3f + 12, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum1, p0f + out_hstep, 0);
                        __lsx_vst((__m128i)_sum2, p0f + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_sum3, p0f + out_hstep * 3, 0);
                        __lsx_vst((__m128i)_sum4, p0f + out_hstep * 4, 0);
                        __lsx_vst((__m128i)_sum5, p0f + out_hstep * 5, 0);
                        __lsx_vst((__m128i)_sum6, p0f + out_hstep * 6, 0);
                        __lsx_vst((__m128i)_sum7, p0f + out_hstep * 7, 0);
                        __lsx_vst((__m128i)_sum8, p0f + out_hstep * 8, 0);
                        __lsx_vst((__m128i)_sum9, p0f + out_hstep * 9, 0);
                        __lsx_vst((__m128i)_suma, p0f + out_hstep * 10, 0);
                        __lsx_vst((__m128i)_sumb, p0f + out_hstep * 11, 0);
                        __lsx_vst((__m128i)_sumc, p0f + out_hstep * 12, 0);
                        __lsx_vst((__m128i)_sumd, p0f + out_hstep * 13, 0);
                        __lsx_vst((__m128i)_sume, p0f + out_hstep * 14, 0);
                        __lsx_vst((__m128i)_sumf, p0f + out_hstep * 15, 0);
                    }
                    p0f += out_hstep * 16;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum1, p0f + 4, 0);
                        __lsx_vst((__m128i)_sum2, p0f + 8, 0);
                        __lsx_vst((__m128i)_sum3, p0f + 12, 0);
                        __lsx_vst((__m128i)_sum4, p0f + 16, 0);
                        __lsx_vst((__m128i)_sum5, p0f + 20, 0);
                        __lsx_vst((__m128i)_sum6, p0f + 24, 0);
                        __lsx_vst((__m128i)_sum7, p0f + 28, 0);
                        __lsx_vst((__m128i)_sum8, p0f + 32, 0);
                        __lsx_vst((__m128i)_sum9, p0f + 36, 0);
                        __lsx_vst((__m128i)_suma, p0f + 40, 0);
                        __lsx_vst((__m128i)_sumb, p0f + 44, 0);
                        __lsx_vst((__m128i)_sumc, p0f + 48, 0);
                        __lsx_vst((__m128i)_sumd, p0f + 52, 0);
                        __lsx_vst((__m128i)_sume, p0f + 56, 0);
                        __lsx_vst((__m128i)_sumf, p0f + 60, 0);
                        p0f += 64;
                    }
                    if (out_elempack == 1)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);
                        transpose4x4_ps(_sum8, _sum9, _suma, _sumb);
                        transpose4x4_ps(_sumc, _sumd, _sume, _sumf);
                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum4, p0f + 4, 0);
                        __lsx_vst((__m128i)_sum8, p0f + 8, 0);
                        __lsx_vst((__m128i)_sumc, p0f + 12, 0);
                        __lsx_vst((__m128i)_sum1, p0f + out_hstep, 0);
                        __lsx_vst((__m128i)_sum5, p0f + out_hstep + 4, 0);
                        __lsx_vst((__m128i)_sum9, p0f + out_hstep + 8, 0);
                        __lsx_vst((__m128i)_sumd, p0f + out_hstep + 12, 0);
                        __lsx_vst((__m128i)_sum2, p0f + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_sum6, p0f + out_hstep * 2 + 4, 0);
                        __lsx_vst((__m128i)_suma, p0f + out_hstep * 2 + 8, 0);
                        __lsx_vst((__m128i)_sume, p0f + out_hstep * 2 + 12, 0);
                        __lsx_vst((__m128i)_sum3, p0f + out_hstep * 3, 0);
                        __lsx_vst((__m128i)_sum7, p0f + out_hstep * 3 + 4, 0);
                        __lsx_vst((__m128i)_sumb, p0f + out_hstep * 3 + 8, 0);
                        __lsx_vst((__m128i)_sumf, p0f + out_hstep * 3 + 12, 0);
                        p0f += 16;
                    }
                }
            }
            else
            {
                if (output_transpose)
                {
                    if (out_elempack == 8)
                    {
                        unsigned short* p1 = p0 + out_hstep * 8;
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);
                        transpose4x4_ps(_sum8, _sum9, _suma, _sumb);
                        transpose4x4_ps(_sumc, _sumd, _sume, _sumf);

                        __lsx_vstelm_d(float2bfloat_lsx(_sum0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum4), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum1), p0 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum5), p0 + 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum2), p0 + 16, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum6), p0 + 20, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum3), p0 + 24, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum7), p0 + 28, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum8), p1, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumc), p1 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum9), p1 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumd), p1 + 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_suma), p1 + 16, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sume), p1 + 20, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumb), p1 + 24, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumf), p1 + 28, 0, 0);
                    }
                    if (out_elempack == 4)
                    {
                        unsigned short* p1 = p0 + out_hstep * 4;
                        unsigned short* p2 = p0 + out_hstep * 8;
                        unsigned short* p3 = p0 + out_hstep * 12;
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);
                        transpose4x4_ps(_sum8, _sum9, _suma, _sumb);
                        transpose4x4_ps(_sumc, _sumd, _sume, _sumf);

                        __lsx_vstelm_d(float2bfloat_lsx(_sum0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum1), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum2), p0 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum3), p0 + 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum4), p1, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum5), p1 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum6), p1 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum7), p1 + 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum8), p2, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum9), p2 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_suma), p2 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumb), p2 + 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumc), p3, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumd), p3 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sume), p3 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumf), p3 + 12, 0, 0);
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
                        __lsx_vstelm_d(float2bfloat_lsx(_sum8), p0 + out_hstep * 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum9), p0 + out_hstep * 9, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_suma), p0 + out_hstep * 10, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumb), p0 + out_hstep * 11, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumc), p0 + out_hstep * 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumd), p0 + out_hstep * 13, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sume), p0 + out_hstep * 14, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumf), p0 + out_hstep * 15, 0, 0);
                    }
                    p0 += out_hstep * 16;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        __lsx_vstelm_d(float2bfloat_lsx(_sum0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum1), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum2), p0 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum3), p0 + 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum4), p0 + 16, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum5), p0 + 20, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum6), p0 + 24, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum7), p0 + 28, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum8), p0 + 32, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum9), p0 + 36, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_suma), p0 + 40, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumb), p0 + 44, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumc), p0 + 48, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumd), p0 + 52, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sume), p0 + 56, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumf), p0 + 60, 0, 0);
                        p0 += 64;
                    }
                    if (out_elempack == 1)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);
                        transpose4x4_ps(_sum8, _sum9, _suma, _sumb);
                        transpose4x4_ps(_sumc, _sumd, _sume, _sumf);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum4), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum8), p0 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumc), p0 + 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum1), p0 + out_hstep, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum5), p0 + out_hstep + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum9), p0 + out_hstep + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumd), p0 + out_hstep + 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum2), p0 + out_hstep * 2, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum6), p0 + out_hstep * 2 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_suma), p0 + out_hstep * 2 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sume), p0 + out_hstep * 2 + 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum3), p0 + out_hstep * 3, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum7), p0 + out_hstep * 3 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumb), p0 + out_hstep * 3 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sumf), p0 + out_hstep * 3 + 12, 0, 0);
                        p0 += 16;
                    }
                }
            }
        }
#endif // __loongarch_asx
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

            // deshuffle from the shuffle-based 4x8 kernel
            _sum1 = (__m128)__lsx_vshuf4i_w((__m128i)_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum3 = (__m128)__lsx_vshuf4i_w((__m128i)_sum3, _LSX_SHUFFLE(2, 1, 0, 3));
            {
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_sum3, (__m128i)_sum0);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_sum3, (__m128i)_sum0);
                __m128 _tmp2 = (__m128)__lsx_vilvl_w((__m128i)_sum1, (__m128i)_sum2);
                __m128 _tmp3 = (__m128)__lsx_vilvh_w((__m128i)_sum1, (__m128i)_sum2);
                _sum0 = (__m128)__lsx_vilvl_d((__m128i)_tmp2, (__m128i)_tmp0);
                _sum1 = (__m128)__lsx_vilvh_d((__m128i)_tmp2, (__m128i)_tmp0);
                _sum2 = (__m128)__lsx_vilvl_d((__m128i)_tmp1, (__m128i)_tmp3);
                _sum3 = (__m128)__lsx_vilvh_d((__m128i)_tmp1, (__m128i)_tmp3);
            }
            _sum1 = (__m128)__lsx_vshuf4i_w((__m128i)_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum3 = (__m128)__lsx_vshuf4i_w((__m128i)_sum3, _LSX_SHUFFLE(2, 1, 0, 3));

            _sum5 = (__m128)__lsx_vshuf4i_w((__m128i)_sum5, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum7 = (__m128)__lsx_vshuf4i_w((__m128i)_sum7, _LSX_SHUFFLE(2, 1, 0, 3));
            {
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_sum7, (__m128i)_sum4);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_sum7, (__m128i)_sum4);
                __m128 _tmp2 = (__m128)__lsx_vilvl_w((__m128i)_sum5, (__m128i)_sum6);
                __m128 _tmp3 = (__m128)__lsx_vilvh_w((__m128i)_sum5, (__m128i)_sum6);
                _sum4 = (__m128)__lsx_vilvl_d((__m128i)_tmp2, (__m128i)_tmp0);
                _sum5 = (__m128)__lsx_vilvh_d((__m128i)_tmp2, (__m128i)_tmp0);
                _sum6 = (__m128)__lsx_vilvl_d((__m128i)_tmp1, (__m128i)_tmp3);
                _sum7 = (__m128)__lsx_vilvh_d((__m128i)_tmp1, (__m128i)_tmp3);
            }
            _sum5 = (__m128)__lsx_vshuf4i_w((__m128i)_sum5, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum7 = (__m128)__lsx_vshuf4i_w((__m128i)_sum7, _LSX_SHUFFLE(2, 1, 0, 3));

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
                    _sum0 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0, 0), _sum0);
                    _sum1 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 4, 0), _sum1);
                    _sum2 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 8, 0), _sum2);
                    _sum3 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 12, 0), _sum3);
                    _sum4 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 16, 0), _sum4);
                    _sum5 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 20, 0), _sum5);
                    _sum6 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 24, 0), _sum6);
                    _sum7 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 28, 0), _sum7);
                    pC0 += 32;
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

            if (alpha != 1.f)
            {
                _sum0 = __lsx_vfmul_s(_sum0, _valpha);
                _sum1 = __lsx_vfmul_s(_sum1, _valpha);
                _sum2 = __lsx_vfmul_s(_sum2, _valpha);
                _sum3 = __lsx_vfmul_s(_sum3, _valpha);
                _sum4 = __lsx_vfmul_s(_sum4, _valpha);
                _sum5 = __lsx_vfmul_s(_sum5, _valpha);
                _sum6 = __lsx_vfmul_s(_sum6, _valpha);
                _sum7 = __lsx_vfmul_s(_sum7, _valpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 8)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);

                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum4, p0f + 4, 0);
                        __lsx_vst((__m128i)_sum1, p0f + 8, 0);
                        __lsx_vst((__m128i)_sum5, p0f + 12, 0);
                        __lsx_vst((__m128i)_sum2, p0f + 16, 0);
                        __lsx_vst((__m128i)_sum6, p0f + 20, 0);
                        __lsx_vst((__m128i)_sum3, p0f + 24, 0);
                        __lsx_vst((__m128i)_sum7, p0f + 28, 0);
                    }
                    if (out_elempack == 4)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);

                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum1, p0f + 4, 0);
                        __lsx_vst((__m128i)_sum2, p0f + 4 * 2, 0);
                        __lsx_vst((__m128i)_sum3, p0f + 4 * 3, 0);
                        __lsx_vst((__m128i)_sum4, p0f + out_hstep * 4, 0);
                        __lsx_vst((__m128i)_sum5, p0f + out_hstep * 4 + 4, 0);
                        __lsx_vst((__m128i)_sum6, p0f + out_hstep * 4 + 4 * 2, 0);
                        __lsx_vst((__m128i)_sum7, p0f + out_hstep * 4 + 4 * 3, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum1, p0f + out_hstep, 0);
                        __lsx_vst((__m128i)_sum2, p0f + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_sum3, p0f + out_hstep * 3, 0);
                        __lsx_vst((__m128i)_sum4, p0f + out_hstep * 4, 0);
                        __lsx_vst((__m128i)_sum5, p0f + out_hstep * 5, 0);
                        __lsx_vst((__m128i)_sum6, p0f + out_hstep * 6, 0);
                        __lsx_vst((__m128i)_sum7, p0f + out_hstep * 7, 0);
                    }
                    p0f += out_hstep * 8;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum1, p0f + 4, 0);
                        __lsx_vst((__m128i)_sum2, p0f + 8, 0);
                        __lsx_vst((__m128i)_sum3, p0f + 12, 0);
                        __lsx_vst((__m128i)_sum4, p0f + 16, 0);
                        __lsx_vst((__m128i)_sum5, p0f + 20, 0);
                        __lsx_vst((__m128i)_sum6, p0f + 24, 0);
                        __lsx_vst((__m128i)_sum7, p0f + 28, 0);
                        p0f += 32;
                    }
                    if (out_elempack == 1)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);
                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum4, p0f + 4, 0);
                        __lsx_vst((__m128i)_sum1, p0f + out_hstep, 0);
                        __lsx_vst((__m128i)_sum5, p0f + out_hstep + 4, 0);
                        __lsx_vst((__m128i)_sum2, p0f + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_sum6, p0f + out_hstep * 2 + 4, 0);
                        __lsx_vst((__m128i)_sum3, p0f + out_hstep * 3, 0);
                        __lsx_vst((__m128i)_sum7, p0f + out_hstep * 3 + 4, 0);
                        p0f += 8;
                    }
                }
            }
            else
            {
                if (output_transpose)
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
                else
                {
                    if (out_elempack == 4)
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
                    if (out_elempack == 1)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);
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

            // deshuffle from the shuffle-based 4x4 kernel
            _sum1 = (__m128)__lsx_vshuf4i_w((__m128i)_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum3 = (__m128)__lsx_vshuf4i_w((__m128i)_sum3, _LSX_SHUFFLE(2, 1, 0, 3));
            {
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_sum3, (__m128i)_sum0);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_sum3, (__m128i)_sum0);
                __m128 _tmp2 = (__m128)__lsx_vilvl_w((__m128i)_sum1, (__m128i)_sum2);
                __m128 _tmp3 = (__m128)__lsx_vilvh_w((__m128i)_sum1, (__m128i)_sum2);
                _sum0 = (__m128)__lsx_vilvl_d((__m128i)_tmp2, (__m128i)_tmp0);
                _sum1 = (__m128)__lsx_vilvh_d((__m128i)_tmp2, (__m128i)_tmp0);
                _sum2 = (__m128)__lsx_vilvl_d((__m128i)_tmp1, (__m128i)_tmp3);
                _sum3 = (__m128)__lsx_vilvh_d((__m128i)_tmp1, (__m128i)_tmp3);
            }
            _sum1 = (__m128)__lsx_vshuf4i_w((__m128i)_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
            _sum3 = (__m128)__lsx_vshuf4i_w((__m128i)_sum3, _LSX_SHUFFLE(2, 1, 0, 3));

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
                    _sum0 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0, 0), _sum0);
                    _sum1 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 4, 0), _sum1);
                    _sum2 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 8, 0), _sum2);
                    _sum3 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 12, 0), _sum3);
                    pC0 += 16;
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

            if (alpha != 1.f)
            {
                _sum0 = __lsx_vfmul_s(_sum0, _valpha);
                _sum1 = __lsx_vfmul_s(_sum1, _valpha);
                _sum2 = __lsx_vfmul_s(_sum2, _valpha);
                _sum3 = __lsx_vfmul_s(_sum3, _valpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 8)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum1, p0f + 8, 0);
                        __lsx_vst((__m128i)_sum2, p0f + 16, 0);
                        __lsx_vst((__m128i)_sum3, p0f + 24, 0);
                    }
                    if (out_elempack == 4)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum1, p0f + 4, 0);
                        __lsx_vst((__m128i)_sum2, p0f + 4 * 2, 0);
                        __lsx_vst((__m128i)_sum3, p0f + 4 * 3, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum1, p0f + out_hstep, 0);
                        __lsx_vst((__m128i)_sum2, p0f + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_sum3, p0f + out_hstep * 3, 0);
                    }
                    p0f += out_hstep * 4;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum1, p0f + 4, 0);
                        __lsx_vst((__m128i)_sum2, p0f + 8, 0);
                        __lsx_vst((__m128i)_sum3, p0f + 12, 0);
                        p0f += 16;
                    }
                    if (out_elempack == 1)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum1, p0f + out_hstep, 0);
                        __lsx_vst((__m128i)_sum2, p0f + out_hstep * 2, 0);
                        __lsx_vst((__m128i)_sum3, p0f + out_hstep * 3, 0);
                        p0f += 4;
                    }
                }
            }
            else
            {
                if (output_transpose)
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
                else
                {
                    if (out_elempack == 4)
                    {
                        __lsx_vstelm_d(float2bfloat_lsx(_sum0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum1), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum2), p0 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_sum3), p0 + 12, 0, 0);
                        p0 += 16;
                    }
                    if (out_elempack == 1)
                    {
                        transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
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

            // deshuffle from the shuffle-based 4x2 kernel
            {
                __m128 _tmp0 = (__m128)__lsx_vshuf4i_w((__m128i)_sum0, _LSX_SHUFFLE(3, 1, 2, 0));
                __m128 _tmp1 = (__m128)__lsx_vshuf4i_w((__m128i)_sum1, _LSX_SHUFFLE(0, 2, 3, 1));
                _sum0 = (__m128)__lsx_vilvl_w((__m128i)_tmp1, (__m128i)_tmp0);
                _sum1 = (__m128)__lsx_vilvh_w((__m128i)_tmp1, (__m128i)_tmp0);
                _sum1 = (__m128)__lsx_vshuf4i_w((__m128i)_sum1, _LSX_SHUFFLE(2, 1, 0, 3));
            }

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
                    _sum0 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0, 0), _sum0);
                    _sum1 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0 + 4, 0), _sum1);
                    pC0 += 8;
                }
                if (broadcast_type_C == 4)
                {
                    _sum0 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pC0[0]), _sum0);
                    _sum1 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pC0[1]), _sum1);
                    pC0 += 2;
                }
            }

            if (alpha != 1.f)
            {
                _sum0 = __lsx_vfmul_s(_sum0, _valpha);
                _sum1 = __lsx_vfmul_s(_sum1, _valpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        __lsx_vstelm_w((__m128i)_sum0, p0f, 0, 0);
                        __lsx_vstelm_w((__m128i)_sum1, p0f + 1, 0, 0);
                        __lsx_vstelm_w((__m128i)_sum0, p0f + 4, 0, 1);
                        __lsx_vstelm_w((__m128i)_sum1, p0f + 5, 0, 1);
                        __lsx_vstelm_w((__m128i)_sum0, p0f + 8, 0, 2);
                        __lsx_vstelm_w((__m128i)_sum1, p0f + 9, 0, 2);
                        __lsx_vstelm_w((__m128i)_sum0, p0f + 12, 0, 3);
                        __lsx_vstelm_w((__m128i)_sum1, p0f + 13, 0, 3);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum1, p0f + out_hstep, 0);
                    }
                    p0f += out_hstep * 2;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        __lsx_vst((__m128i)_sum1, p0f + 4, 0);
                        p0f += 8;
                    }
                    if (out_elempack == 1)
                    {
                        __m128i _tmp0 = __lsx_vilvl_w((__m128i)_sum1, (__m128i)_sum0);
                        __m128i _tmp1 = __lsx_vilvh_w((__m128i)_sum1, (__m128i)_sum0);
                        __lsx_vstelm_d(_tmp0, p0f, 0, 0);
                        __lsx_vstelm_d(_tmp0, p0f + out_hstep, 0, 1);
                        __lsx_vstelm_d(_tmp1, p0f + out_hstep * 2, 0, 0);
                        __lsx_vstelm_d(_tmp1, p0f + out_hstep * 3, 0, 1);
                        p0f += 2;
                    }
                }
            }
            else
            {
                __m128i _bf0 = float2bfloat_lsx(_sum0);
                __m128i _bf1 = float2bfloat_lsx(_sum1);

                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        __lsx_vstelm_h(_bf0, p0, 0, 0);
                        __lsx_vstelm_h(_bf1, p0 + 1, 0, 0);
                        __lsx_vstelm_h(_bf0, p0 + 4, 0, 1);
                        __lsx_vstelm_h(_bf1, p0 + 5, 0, 1);
                        __lsx_vstelm_h(_bf0, p0 + 8, 0, 2);
                        __lsx_vstelm_h(_bf1, p0 + 9, 0, 2);
                        __lsx_vstelm_h(_bf0, p0 + 12, 0, 3);
                        __lsx_vstelm_h(_bf1, p0 + 13, 0, 3);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vstelm_d(_bf0, p0, 0, 0);
                        __lsx_vstelm_d(_bf1, p0 + out_hstep, 0, 0);
                    }
                    p0 += out_hstep * 2;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        __lsx_vstelm_d(_bf0, p0, 0, 0);
                        __lsx_vstelm_d(_bf1, p0 + 4, 0, 0);
                        p0 += 8;
                    }
                    if (out_elempack == 1)
                    {
                        __m128i _tmp = __lsx_vilvl_h(_bf1, _bf0);
                        __lsx_vstelm_w(_tmp, p0, 0, 0);
                        __lsx_vstelm_w(_tmp, p0 + out_hstep, 0, 1);
                        __lsx_vstelm_w(_tmp, p0 + out_hstep * 2, 0, 2);
                        __lsx_vstelm_w(_tmp, p0 + out_hstep * 3, 0, 3);
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
                    _sum0 = __lsx_vfmadd_s(_beta, (__m128)__lsx_vld(pC0, 0), _sum0);
                    pC0 += 4;
                }
                if (broadcast_type_C == 4)
                {
                    _sum0 = __lsx_vfmadd_s(_beta, __lsx_vreplfr2vr_s(pC0[0]), _sum0);
                    pC0 += 1;
                }
            }

            if (alpha != 1.f)
            {
                _sum0 = __lsx_vfmul_s(_sum0, _valpha);
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        __lsx_vstelm_w((__m128i)_sum0, p0f, 0, 0);
                        __lsx_vstelm_w((__m128i)_sum0, p0f + 4, 0, 1);
                        __lsx_vstelm_w((__m128i)_sum0, p0f + 8, 0, 2);
                        __lsx_vstelm_w((__m128i)_sum0, p0f + 12, 0, 3);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vst((__m128i)_sum0, p0f, 0);
                    }
                    p0f += out_hstep;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        __lsx_vst((__m128i)_sum0, p0f, 0);
                        p0f += 4;
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vstelm_w((__m128i)_sum0, p0f, 0, 0);
                        __lsx_vstelm_w((__m128i)_sum0, p0f + out_hstep, 0, 1);
                        __lsx_vstelm_w((__m128i)_sum0, p0f + out_hstep * 2, 0, 2);
                        __lsx_vstelm_w((__m128i)_sum0, p0f + out_hstep * 3, 0, 3);
                        p0f++;
                    }
                }
            }
            else
            {
                __m128i _bf = float2bfloat_lsx(_sum0);

                if (output_transpose)
                {
                    if (out_elempack == 4)
                    {
                        __lsx_vstelm_h(_bf, p0, 0, 0);
                        __lsx_vstelm_h(_bf, p0 + 4, 0, 1);
                        __lsx_vstelm_h(_bf, p0 + 8, 0, 2);
                        __lsx_vstelm_h(_bf, p0 + 12, 0, 3);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vstelm_d(_bf, p0, 0, 0);
                    }
                    p0 += out_hstep;
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        __lsx_vstelm_d(_bf, p0, 0, 0);
                        p0 += 4;
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vstelm_h(_bf, p0, 0, 0);
                        __lsx_vstelm_h(_bf, p0 + out_hstep, 0, 1);
                        __lsx_vstelm_h(_bf, p0 + out_hstep * 2, 0, 2);
                        __lsx_vstelm_h(_bf, p0 + out_hstep * 3, 0, 3);
                        p0++;
                    }
                }
            }
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        unsigned short* p0;
        float* p0f;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
            p0f = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
            p0f = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        const float* pC0 = pC;
        if (pC0)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC0 = (const float*)C + (i + ii);
            }
            if (broadcast_type_C == 3)
            {
                pC += max_jj * 2;
            }
            if (broadcast_type_C == 4)
            {
                pC0 = (const float*)C + j;
            }
        }

        int jj = 0;
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _sum0 = (__m256)__lasx_xvld(pp, 0);
            __m256 _sum1 = (__m256)__lasx_xvld(pp + 8, 0);
            __m256 _sum2 = (__m256)__lasx_xvld(pp + 16, 0);
            __m256 _sum3 = (__m256)__lasx_xvld(pp + 24, 0);
            pp += 32;

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
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    __m256 _c0 = (__m256)__lasx_xvreplfr2vr_s(pC0[0]);
                    __m256 _c1 = (__m256)__lasx_xvreplfr2vr_s(pC0[1]);
                    _sum0 = __lasx_xvfmadd_s(_c0, _beta, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_c0, _beta, _sum1);
                    _sum2 = __lasx_xvfmadd_s(_c1, _beta, _sum2);
                    _sum3 = __lasx_xvfmadd_s(_c1, _beta, _sum3);
                }
                if (broadcast_type_C == 3)
                {
                    __m256i _c0 = __lasx_xvld(pC0, 0);
                    __m256i _c1 = __lasx_xvld(pC0 + 8, 0);
                    __m256i _c2 = __lasx_xvld(pC0 + 16, 0);
                    __m256i _c3 = __lasx_xvld(pC0 + 24, 0);
                    __m256 _c0e = (__m256)__lasx_xvpickev_w(_c1, _c0);
                    __m256 _c0o = (__m256)__lasx_xvpickod_w(_c1, _c0);
                    __m256 _c1e = (__m256)__lasx_xvpickev_w(_c3, _c2);
                    __m256 _c1o = (__m256)__lasx_xvpickod_w(_c3, _c2);
                    _c0e = (__m256)__lasx_xvpermi_d((__m256i)_c0e, _LSX_SHUFFLE(3, 1, 2, 0));
                    _c0o = (__m256)__lasx_xvpermi_d((__m256i)_c0o, _LSX_SHUFFLE(3, 1, 2, 0));
                    _c1e = (__m256)__lasx_xvpermi_d((__m256i)_c1e, _LSX_SHUFFLE(3, 1, 2, 0));
                    _c1o = (__m256)__lasx_xvpermi_d((__m256i)_c1o, _LSX_SHUFFLE(3, 1, 2, 0));
                    _sum0 = __lasx_xvfmadd_s(_c0e, _beta, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_c1e, _beta, _sum1);
                    _sum2 = __lasx_xvfmadd_s(_c0o, _beta, _sum2);
                    _sum3 = __lasx_xvfmadd_s(_c1o, _beta, _sum3);
                    pC0 += 32;
                }
                if (broadcast_type_C == 4)
                {
                    __m256 _c0 = (__m256)__lasx_xvld(pC0, 0);
                    __m256 _c1 = (__m256)__lasx_xvld(pC0 + 8, 0);
                    _sum0 = __lasx_xvfmadd_s(_c0, _beta, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_c1, _beta, _sum1);
                    _sum2 = __lasx_xvfmadd_s(_c0, _beta, _sum2);
                    _sum3 = __lasx_xvfmadd_s(_c1, _beta, _sum3);
                    pC0 += 16;
                }
            }

            if (alpha != 1.f)
            {
                __m256 _alpha = (__m256)__lasx_xvreplfr2vr_s(alpha);
                _sum0 = __lasx_xvfmul_s(_sum0, _alpha);
                _sum1 = __lasx_xvfmul_s(_sum1, _alpha);
                _sum2 = __lasx_xvfmul_s(_sum2, _alpha);
                _sum3 = __lasx_xvfmul_s(_sum3, _alpha);
            }

            __m256 _tmp0 = (__m256)__lasx_xvilvl_w((__m256i)_sum2, (__m256i)_sum0);
            __m256 _tmp1 = (__m256)__lasx_xvilvh_w((__m256i)_sum2, (__m256i)_sum0);
            __m256 _tmp2 = (__m256)__lasx_xvilvl_w((__m256i)_sum3, (__m256i)_sum1);
            __m256 _tmp3 = (__m256)__lasx_xvilvh_w((__m256i)_sum3, (__m256i)_sum1);

            __m128 _f0 = __lasx_extract_128_lo_s(_tmp0);
            __m128 _f4 = __lasx_extract_128_hi_s(_tmp0);
            __m128 _f2 = __lasx_extract_128_lo_s(_tmp1);
            __m128 _f6 = __lasx_extract_128_hi_s(_tmp1);
            __m128 _f1 = __lasx_extract_128_lo_s(_tmp2);
            __m128 _f5 = __lasx_extract_128_hi_s(_tmp2);
            __m128 _f3 = __lasx_extract_128_lo_s(_tmp3);
            __m128 _f7 = __lasx_extract_128_hi_s(_tmp3);

            if (output_transpose)
            {
                if (output_elemtype == 1)
                {
                    if (out_elempack == 8)
                    {
                        __m128 _r0 = (__m128)__lsx_vpickev_w((__m128i)_f2, (__m128i)_f0);
                        __m128 _r1 = (__m128)__lsx_vpickev_w((__m128i)_f6, (__m128i)_f4);
                        __m128 _r2 = (__m128)__lsx_vpickev_w((__m128i)_f3, (__m128i)_f1);
                        __m128 _r3 = (__m128)__lsx_vpickev_w((__m128i)_f7, (__m128i)_f5);
                        __m128 _r4 = (__m128)__lsx_vpickod_w((__m128i)_f2, (__m128i)_f0);
                        __m128 _r5 = (__m128)__lsx_vpickod_w((__m128i)_f6, (__m128i)_f4);
                        __m128 _r6 = (__m128)__lsx_vpickod_w((__m128i)_f3, (__m128i)_f1);
                        __m128 _r7 = (__m128)__lsx_vpickod_w((__m128i)_f7, (__m128i)_f5);
                        float* p1f = p0f + out_hstep * 8;
                        __lsx_vst((__m128i)_r0, p0f, 0);
                        __lsx_vst((__m128i)_r1, p0f + 4, 0);
                        __lsx_vst((__m128i)_r4, p0f + 8, 0);
                        __lsx_vst((__m128i)_r5, p0f + 12, 0);
                        __lsx_vst((__m128i)_r2, p1f, 0);
                        __lsx_vst((__m128i)_r3, p1f + 4, 0);
                        __lsx_vst((__m128i)_r6, p1f + 8, 0);
                        __lsx_vst((__m128i)_r7, p1f + 12, 0);
                    }
                    if (out_elempack == 4)
                    {
                        __m128 _r0 = (__m128)__lsx_vpickev_w((__m128i)_f2, (__m128i)_f0);
                        __m128 _r1 = (__m128)__lsx_vpickev_w((__m128i)_f6, (__m128i)_f4);
                        __m128 _r2 = (__m128)__lsx_vpickev_w((__m128i)_f3, (__m128i)_f1);
                        __m128 _r3 = (__m128)__lsx_vpickev_w((__m128i)_f7, (__m128i)_f5);
                        __m128 _r4 = (__m128)__lsx_vpickod_w((__m128i)_f2, (__m128i)_f0);
                        __m128 _r5 = (__m128)__lsx_vpickod_w((__m128i)_f6, (__m128i)_f4);
                        __m128 _r6 = (__m128)__lsx_vpickod_w((__m128i)_f3, (__m128i)_f1);
                        __m128 _r7 = (__m128)__lsx_vpickod_w((__m128i)_f7, (__m128i)_f5);
                        float* p1f = p0f + out_hstep * 4;
                        float* p2f = p0f + out_hstep * 8;
                        float* p3f = p0f + out_hstep * 12;
                        __lsx_vst((__m128i)_r0, p0f, 0);
                        __lsx_vst((__m128i)_r4, p0f + 4, 0);
                        __lsx_vst((__m128i)_r1, p1f, 0);
                        __lsx_vst((__m128i)_r5, p1f + 4, 0);
                        __lsx_vst((__m128i)_r2, p2f, 0);
                        __lsx_vst((__m128i)_r6, p2f + 4, 0);
                        __lsx_vst((__m128i)_r3, p3f, 0);
                        __lsx_vst((__m128i)_r7, p3f + 4, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vstelm_d((__m128i)_f0, p0f, 0, 0);
                        __lsx_vstelm_d((__m128i)_f0, p0f + out_hstep, 0, 1);
                        __lsx_vstelm_d((__m128i)_f2, p0f + out_hstep * 2, 0, 0);
                        __lsx_vstelm_d((__m128i)_f2, p0f + out_hstep * 3, 0, 1);
                        __lsx_vstelm_d((__m128i)_f4, p0f + out_hstep * 4, 0, 0);
                        __lsx_vstelm_d((__m128i)_f4, p0f + out_hstep * 5, 0, 1);
                        __lsx_vstelm_d((__m128i)_f6, p0f + out_hstep * 6, 0, 0);
                        __lsx_vstelm_d((__m128i)_f6, p0f + out_hstep * 7, 0, 1);
                        __lsx_vstelm_d((__m128i)_f1, p0f + out_hstep * 8, 0, 0);
                        __lsx_vstelm_d((__m128i)_f1, p0f + out_hstep * 9, 0, 1);
                        __lsx_vstelm_d((__m128i)_f3, p0f + out_hstep * 10, 0, 0);
                        __lsx_vstelm_d((__m128i)_f3, p0f + out_hstep * 11, 0, 1);
                        __lsx_vstelm_d((__m128i)_f5, p0f + out_hstep * 12, 0, 0);
                        __lsx_vstelm_d((__m128i)_f5, p0f + out_hstep * 13, 0, 1);
                        __lsx_vstelm_d((__m128i)_f7, p0f + out_hstep * 14, 0, 0);
                        __lsx_vstelm_d((__m128i)_f7, p0f + out_hstep * 15, 0, 1);
                    }
                    p0f += out_hstep * 16;
                }
                else
                {
                    __m128i _bf0 = float2bfloat_lsx(_f0);
                    __m128i _bf1 = float2bfloat_lsx(_f1);
                    __m128i _bf2 = float2bfloat_lsx(_f2);
                    __m128i _bf3 = float2bfloat_lsx(_f3);
                    __m128i _bf4 = float2bfloat_lsx(_f4);
                    __m128i _bf5 = float2bfloat_lsx(_f5);
                    __m128i _bf6 = float2bfloat_lsx(_f6);
                    __m128i _bf7 = float2bfloat_lsx(_f7);

                    if (out_elempack == 8)
                    {
                        __m128 _r0 = (__m128)__lsx_vpickev_w((__m128i)_f2, (__m128i)_f0);
                        __m128 _r1 = (__m128)__lsx_vpickev_w((__m128i)_f6, (__m128i)_f4);
                        __m128 _r2 = (__m128)__lsx_vpickev_w((__m128i)_f3, (__m128i)_f1);
                        __m128 _r3 = (__m128)__lsx_vpickev_w((__m128i)_f7, (__m128i)_f5);
                        __m128 _r4 = (__m128)__lsx_vpickod_w((__m128i)_f2, (__m128i)_f0);
                        __m128 _r5 = (__m128)__lsx_vpickod_w((__m128i)_f6, (__m128i)_f4);
                        __m128 _r6 = (__m128)__lsx_vpickod_w((__m128i)_f3, (__m128i)_f1);
                        __m128 _r7 = (__m128)__lsx_vpickod_w((__m128i)_f7, (__m128i)_f5);
                        unsigned short* p1 = p0 + out_hstep * 8;
                        __lsx_vstelm_d(float2bfloat_lsx(_r0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r1), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r4), p0 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r5), p0 + 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r2), p1, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r3), p1 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r6), p1 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r7), p1 + 12, 0, 0);
                    }
                    if (out_elempack == 4)
                    {
                        __m128 _r0 = (__m128)__lsx_vpickev_w((__m128i)_f2, (__m128i)_f0);
                        __m128 _r1 = (__m128)__lsx_vpickev_w((__m128i)_f6, (__m128i)_f4);
                        __m128 _r2 = (__m128)__lsx_vpickev_w((__m128i)_f3, (__m128i)_f1);
                        __m128 _r3 = (__m128)__lsx_vpickev_w((__m128i)_f7, (__m128i)_f5);
                        __m128 _r4 = (__m128)__lsx_vpickod_w((__m128i)_f2, (__m128i)_f0);
                        __m128 _r5 = (__m128)__lsx_vpickod_w((__m128i)_f6, (__m128i)_f4);
                        __m128 _r6 = (__m128)__lsx_vpickod_w((__m128i)_f3, (__m128i)_f1);
                        __m128 _r7 = (__m128)__lsx_vpickod_w((__m128i)_f7, (__m128i)_f5);
                        unsigned short* p1 = p0 + out_hstep * 4;
                        unsigned short* p2 = p0 + out_hstep * 8;
                        unsigned short* p3 = p0 + out_hstep * 12;
                        __lsx_vstelm_d(float2bfloat_lsx(_r0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r4), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r1), p1, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r5), p1 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r2), p2, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r6), p2 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r3), p3, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r7), p3 + 4, 0, 0);
                    }
                    if (out_elempack == 1)
                    {
                        __lsx_vstelm_w(_bf0, p0, 0, 0);
                        __lsx_vstelm_w(_bf0, p0 + out_hstep, 0, 1);
                        __lsx_vstelm_w(_bf2, p0 + out_hstep * 2, 0, 0);
                        __lsx_vstelm_w(_bf2, p0 + out_hstep * 3, 0, 1);
                        __lsx_vstelm_w(_bf4, p0 + out_hstep * 4, 0, 0);
                        __lsx_vstelm_w(_bf4, p0 + out_hstep * 5, 0, 1);
                        __lsx_vstelm_w(_bf6, p0 + out_hstep * 6, 0, 0);
                        __lsx_vstelm_w(_bf6, p0 + out_hstep * 7, 0, 1);
                        __lsx_vstelm_w(_bf1, p0 + out_hstep * 8, 0, 0);
                        __lsx_vstelm_w(_bf1, p0 + out_hstep * 9, 0, 1);
                        __lsx_vstelm_w(_bf3, p0 + out_hstep * 10, 0, 0);
                        __lsx_vstelm_w(_bf3, p0 + out_hstep * 11, 0, 1);
                        __lsx_vstelm_w(_bf5, p0 + out_hstep * 12, 0, 0);
                        __lsx_vstelm_w(_bf5, p0 + out_hstep * 13, 0, 1);
                        __lsx_vstelm_w(_bf7, p0 + out_hstep * 14, 0, 0);
                        __lsx_vstelm_w(_bf7, p0 + out_hstep * 15, 0, 1);
                    }
                    p0 += out_hstep * 16;
                }
            }
            else
            {
                if (output_elemtype == 1)
                {
                    if (out_elempack == 1)
                    {
                        __m128 _r0 = (__m128)__lsx_vpickev_w((__m128i)_f2, (__m128i)_f0);
                        __m128 _r1 = (__m128)__lsx_vpickev_w((__m128i)_f6, (__m128i)_f4);
                        __m128 _r2 = (__m128)__lsx_vpickev_w((__m128i)_f3, (__m128i)_f1);
                        __m128 _r3 = (__m128)__lsx_vpickev_w((__m128i)_f7, (__m128i)_f5);
                        __m128 _r4 = (__m128)__lsx_vpickod_w((__m128i)_f2, (__m128i)_f0);
                        __m128 _r5 = (__m128)__lsx_vpickod_w((__m128i)_f6, (__m128i)_f4);
                        __m128 _r6 = (__m128)__lsx_vpickod_w((__m128i)_f3, (__m128i)_f1);
                        __m128 _r7 = (__m128)__lsx_vpickod_w((__m128i)_f7, (__m128i)_f5);
                        __lsx_vst((__m128i)_r0, p0f, 0);
                        __lsx_vst((__m128i)_r1, p0f + 4, 0);
                        __lsx_vst((__m128i)_r2, p0f + 8, 0);
                        __lsx_vst((__m128i)_r3, p0f + 12, 0);
                        __lsx_vst((__m128i)_r4, p0f + out_hstep, 0);
                        __lsx_vst((__m128i)_r5, p0f + out_hstep + 4, 0);
                        __lsx_vst((__m128i)_r6, p0f + out_hstep + 8, 0);
                        __lsx_vst((__m128i)_r7, p0f + out_hstep + 12, 0);
                        p0f += 16;
                    }
                    if (out_elempack == 4 || out_elempack == 8)
                    {
                        __lsx_vstelm_w((__m128i)_f0, p0f, 0, 0);
                        __lsx_vstelm_w((__m128i)_f2, p0f + 1, 0, 0);
                        __lsx_vstelm_w((__m128i)_f0, p0f + out_elempack, 0, 1);
                        __lsx_vstelm_w((__m128i)_f2, p0f + out_elempack + 1, 0, 1);
                        __lsx_vstelm_w((__m128i)_f0, p0f + out_elempack * 2, 0, 2);
                        __lsx_vstelm_w((__m128i)_f2, p0f + out_elempack * 2 + 1, 0, 2);
                        __lsx_vstelm_w((__m128i)_f0, p0f + out_elempack * 3, 0, 3);
                        __lsx_vstelm_w((__m128i)_f2, p0f + out_elempack * 3 + 1, 0, 3);
                        __lsx_vstelm_w((__m128i)_f1, p0f + out_elempack * 4, 0, 0);
                        __lsx_vstelm_w((__m128i)_f3, p0f + out_elempack * 4 + 1, 0, 0);
                        __lsx_vstelm_w((__m128i)_f1, p0f + out_elempack * 5, 0, 1);
                        __lsx_vstelm_w((__m128i)_f3, p0f + out_elempack * 5 + 1, 0, 1);
                        __lsx_vstelm_w((__m128i)_f1, p0f + out_elempack * 6, 0, 2);
                        __lsx_vstelm_w((__m128i)_f3, p0f + out_elempack * 6 + 1, 0, 2);
                        __lsx_vstelm_w((__m128i)_f1, p0f + out_elempack * 7, 0, 3);
                        __lsx_vstelm_w((__m128i)_f3, p0f + out_elempack * 7 + 1, 0, 3);
                        __lsx_vstelm_w((__m128i)_f4, p0f + out_elempack * 8, 0, 0);
                        __lsx_vstelm_w((__m128i)_f6, p0f + out_elempack * 8 + 1, 0, 0);
                        __lsx_vstelm_w((__m128i)_f4, p0f + out_elempack * 9, 0, 1);
                        __lsx_vstelm_w((__m128i)_f6, p0f + out_elempack * 9 + 1, 0, 1);
                        __lsx_vstelm_w((__m128i)_f4, p0f + out_elempack * 10, 0, 2);
                        __lsx_vstelm_w((__m128i)_f6, p0f + out_elempack * 10 + 1, 0, 2);
                        __lsx_vstelm_w((__m128i)_f4, p0f + out_elempack * 11, 0, 3);
                        __lsx_vstelm_w((__m128i)_f6, p0f + out_elempack * 11 + 1, 0, 3);
                        __lsx_vstelm_w((__m128i)_f5, p0f + out_elempack * 12, 0, 0);
                        __lsx_vstelm_w((__m128i)_f7, p0f + out_elempack * 12 + 1, 0, 0);
                        __lsx_vstelm_w((__m128i)_f5, p0f + out_elempack * 13, 0, 1);
                        __lsx_vstelm_w((__m128i)_f7, p0f + out_elempack * 13 + 1, 0, 1);
                        __lsx_vstelm_w((__m128i)_f5, p0f + out_elempack * 14, 0, 2);
                        __lsx_vstelm_w((__m128i)_f7, p0f + out_elempack * 14 + 1, 0, 2);
                        __lsx_vstelm_w((__m128i)_f5, p0f + out_elempack * 15, 0, 3);
                        __lsx_vstelm_w((__m128i)_f7, p0f + out_elempack * 15 + 1, 0, 3);
                        p0f += 16 * out_elempack;
                    }
                }
                else
                {
                    __m128i _bf0 = float2bfloat_lsx(_f0);
                    __m128i _bf1 = float2bfloat_lsx(_f1);
                    __m128i _bf2 = float2bfloat_lsx(_f2);
                    __m128i _bf3 = float2bfloat_lsx(_f3);
                    __m128i _bf4 = float2bfloat_lsx(_f4);
                    __m128i _bf5 = float2bfloat_lsx(_f5);
                    __m128i _bf6 = float2bfloat_lsx(_f6);
                    __m128i _bf7 = float2bfloat_lsx(_f7);

                    if (out_elempack == 1)
                    {
                        __m128 _r0 = (__m128)__lsx_vpickev_w((__m128i)_f2, (__m128i)_f0);
                        __m128 _r1 = (__m128)__lsx_vpickev_w((__m128i)_f6, (__m128i)_f4);
                        __m128 _r2 = (__m128)__lsx_vpickev_w((__m128i)_f3, (__m128i)_f1);
                        __m128 _r3 = (__m128)__lsx_vpickev_w((__m128i)_f7, (__m128i)_f5);
                        __m128 _r4 = (__m128)__lsx_vpickod_w((__m128i)_f2, (__m128i)_f0);
                        __m128 _r5 = (__m128)__lsx_vpickod_w((__m128i)_f6, (__m128i)_f4);
                        __m128 _r6 = (__m128)__lsx_vpickod_w((__m128i)_f3, (__m128i)_f1);
                        __m128 _r7 = (__m128)__lsx_vpickod_w((__m128i)_f7, (__m128i)_f5);
                        __lsx_vstelm_d(float2bfloat_lsx(_r0), p0, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r1), p0 + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r2), p0 + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r3), p0 + 12, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r4), p0 + out_hstep, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r5), p0 + out_hstep + 4, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r6), p0 + out_hstep + 8, 0, 0);
                        __lsx_vstelm_d(float2bfloat_lsx(_r7), p0 + out_hstep + 12, 0, 0);
                        p0 += 16;
                    }
                    if (out_elempack == 4 || out_elempack == 8)
                    {
                        __lsx_vstelm_h(_bf0, p0, 0, 0);
                        __lsx_vstelm_h(_bf2, p0 + 1, 0, 0);
                        __lsx_vstelm_h(_bf0, p0 + out_elempack, 0, 1);
                        __lsx_vstelm_h(_bf2, p0 + out_elempack + 1, 0, 1);
                        __lsx_vstelm_h(_bf0, p0 + out_elempack * 2, 0, 2);
                        __lsx_vstelm_h(_bf2, p0 + out_elempack * 2 + 1, 0, 2);
                        __lsx_vstelm_h(_bf0, p0 + out_elempack * 3, 0, 3);
                        __lsx_vstelm_h(_bf2, p0 + out_elempack * 3 + 1, 0, 3);
                        __lsx_vstelm_h(_bf1, p0 + out_elempack * 4, 0, 0);
                        __lsx_vstelm_h(_bf3, p0 + out_elempack * 4 + 1, 0, 0);
                        __lsx_vstelm_h(_bf1, p0 + out_elempack * 5, 0, 1);
                        __lsx_vstelm_h(_bf3, p0 + out_elempack * 5 + 1, 0, 1);
                        __lsx_vstelm_h(_bf1, p0 + out_elempack * 6, 0, 2);
                        __lsx_vstelm_h(_bf3, p0 + out_elempack * 6 + 1, 0, 2);
                        __lsx_vstelm_h(_bf1, p0 + out_elempack * 7, 0, 3);
                        __lsx_vstelm_h(_bf3, p0 + out_elempack * 7 + 1, 0, 3);
                        __lsx_vstelm_h(_bf4, p0 + out_elempack * 8, 0, 0);
                        __lsx_vstelm_h(_bf6, p0 + out_elempack * 8 + 1, 0, 0);
                        __lsx_vstelm_h(_bf4, p0 + out_elempack * 9, 0, 1);
                        __lsx_vstelm_h(_bf6, p0 + out_elempack * 9 + 1, 0, 1);
                        __lsx_vstelm_h(_bf4, p0 + out_elempack * 10, 0, 2);
                        __lsx_vstelm_h(_bf6, p0 + out_elempack * 10 + 1, 0, 2);
                        __lsx_vstelm_h(_bf4, p0 + out_elempack * 11, 0, 3);
                        __lsx_vstelm_h(_bf6, p0 + out_elempack * 11 + 1, 0, 3);
                        __lsx_vstelm_h(_bf5, p0 + out_elempack * 12, 0, 0);
                        __lsx_vstelm_h(_bf7, p0 + out_elempack * 12 + 1, 0, 0);
                        __lsx_vstelm_h(_bf5, p0 + out_elempack * 13, 0, 1);
                        __lsx_vstelm_h(_bf7, p0 + out_elempack * 13 + 1, 0, 1);
                        __lsx_vstelm_h(_bf5, p0 + out_elempack * 14, 0, 2);
                        __lsx_vstelm_h(_bf7, p0 + out_elempack * 14 + 1, 0, 2);
                        __lsx_vstelm_h(_bf5, p0 + out_elempack * 15, 0, 3);
                        __lsx_vstelm_h(_bf7, p0 + out_elempack * 15 + 1, 0, 3);
                        p0 += 16 * out_elempack;
                    }
                }
            }
        }
#endif // __loongarch_asx
        for (; jj < max_jj;)
        {
            const int nn = output_transpose ? out_elempack : 1;

            for (int q = 0; q < nn; q++)
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
                        sum0 += pC0[0] * beta;
                        sum1 += pC0[1] * beta;
                        pC0 += 2;
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

                if (output_elemtype == 1)
                {
                    if (output_transpose)
                    {
                        if (out_elempack == 8)
                        {
                            p0f[q] = sum0;
                            p0f[q + 8] = sum1;
                        }
                        if (out_elempack == 4)
                        {
                            p0f[q] = sum0;
                            p0f[q + 4] = sum1;
                        }
                        if (out_elempack == 1)
                        {
                            p0f[0] = sum0;
                            p0f[1] = sum1;
                        }
                    }
                    else
                    {
                        if (out_elempack == 8)
                        {
                            p0f[0] = sum0;
                            p0f[out_hstep] = sum1;
                        }
                        if (out_elempack == 4)
                        {
                            p0f[0] = sum0;
                            p0f[out_hstep] = sum1;
                        }
                        if (out_elempack == 1)
                        {
                            p0f[0] = sum0;
                            p0f[out_hstep] = sum1;
                        }
                    }
                }
                else
                {
                    if (output_transpose)
                    {
                        if (out_elempack == 8)
                        {
                            p0[q] = float32_to_bfloat16(sum0);
                            p0[q + 8] = float32_to_bfloat16(sum1);
                        }
                        if (out_elempack == 4)
                        {
                            p0[q] = float32_to_bfloat16(sum0);
                            p0[q + 4] = float32_to_bfloat16(sum1);
                        }
                        if (out_elempack == 1)
                        {
                            p0[0] = float32_to_bfloat16(sum0);
                            p0[1] = float32_to_bfloat16(sum1);
                        }
                    }
                    else
                    {
                        if (out_elempack == 8)
                        {
                            p0[0] = float32_to_bfloat16(sum0);
                            p0[out_hstep] = float32_to_bfloat16(sum1);
                        }
                        if (out_elempack == 4)
                        {
                            p0[0] = float32_to_bfloat16(sum0);
                            p0[out_hstep] = float32_to_bfloat16(sum1);
                        }
                        if (out_elempack == 1)
                        {
                            p0[0] = float32_to_bfloat16(sum0);
                            p0[out_hstep] = float32_to_bfloat16(sum1);
                        }
                    }
                }
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    p0f += out_hstep * out_elempack;
                    jj += out_elempack;
                }
                else
                {
                    p0f += 1;
                    jj += 1;
                }
            }
            else
            {
                if (output_transpose)
                {
                    p0 += out_hstep * out_elempack;
                    jj += out_elempack;
                }
                else
                {
                    p0 += 1;
                    jj += 1;
                }
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        unsigned short* p0;
        float* p0f;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
            p0f = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
            p0f = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        const float* pC0 = pC;
        if (pC0)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC0 = (const float*)C + (i + ii);
            }
            if (broadcast_type_C == 3)
            {
                pC += max_jj;
            }
            if (broadcast_type_C == 4)
            {
                pC0 = (const float*)C + j;
            }
        }

        int jj = 0;
        for (; jj < max_jj;)
        {
            const int nn = output_transpose ? out_elempack : 1;

            for (int q = 0; q < nn; q++)
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
                        sum += pC0[0] * beta;
                        pC0 += 1;
                    }
                    if (broadcast_type_C == 4)
                    {
                        sum += pC0[0] * beta;
                        pC0 += 1;
                    }
                }

                sum *= alpha;

                if (output_elemtype == 1)
                {
                    if (output_transpose)
                    {
                        if (out_elempack == 8)
                        {
                            p0f[q] = sum;
                        }
                        if (out_elempack == 4)
                        {
                            p0f[q] = sum;
                        }
                        if (out_elempack == 1)
                        {
                            p0f[0] = sum;
                        }
                    }
                    else
                    {
                        if (out_elempack == 8)
                        {
                            p0f[0] = sum;
                        }
                        if (out_elempack == 4)
                        {
                            p0f[0] = sum;
                        }
                        if (out_elempack == 1)
                        {
                            p0f[0] = sum;
                        }
                    }
                }
                else
                {
                    if (output_transpose)
                    {
                        if (out_elempack == 8)
                        {
                            p0[q] = float32_to_bfloat16(sum);
                        }
                        if (out_elempack == 4)
                        {
                            p0[q] = float32_to_bfloat16(sum);
                        }
                        if (out_elempack == 1)
                        {
                            p0[0] = float32_to_bfloat16(sum);
                        }
                    }
                    else
                    {
                        if (out_elempack == 8)
                        {
                            p0[0] = float32_to_bfloat16(sum);
                        }
                        if (out_elempack == 4)
                        {
                            p0[0] = float32_to_bfloat16(sum);
                        }
                        if (out_elempack == 1)
                        {
                            p0[0] = float32_to_bfloat16(sum);
                        }
                    }
                }
            }

            if (output_elemtype == 1)
            {
                if (output_transpose)
                {
                    p0f += out_hstep * out_elempack;
                    jj += out_elempack;
                }
                else
                {
                    p0f += 1;
                    jj += 1;
                }
            }
            else
            {
                if (output_transpose)
                {
                    p0 += out_hstep * out_elempack;
                    jj += out_elempack;
                }
                else
                {
                    p0 += 1;
                    jj += 1;
                }
            }
        }
    }
}

static void gemm_transB_packed_tile(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk)
{
    const float* pAT = AT_tile;
    const float* pBT = BT_tile;
    float* outptr = topT_tile;

    int ii = 0;
#if __loongarch_sx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* pB = pBT;

        int jj = 0;
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _sum0;
            __m256 _sum1;
            __m256 _sum2;
            __m256 _sum3;
            __m256 _sum4;
            __m256 _sum5;
            __m256 _sum6;
            __m256 _sum7;
            __m256 _sum8;
            __m256 _sum9;
            __m256 _suma;
            __m256 _sumb;
            __m256 _sumc;
            __m256 _sumd;
            __m256 _sume;
            __m256 _sumf;

            if (k == 0)
            {
                _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum1 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum2 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum3 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum4 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum5 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum6 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum7 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum8 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum9 = (__m256)__lasx_xvreplgr2vr_w(0);
                _suma = (__m256)__lasx_xvreplgr2vr_w(0);
                _sumb = (__m256)__lasx_xvreplgr2vr_w(0);
                _sumc = (__m256)__lasx_xvreplgr2vr_w(0);
                _sumd = (__m256)__lasx_xvreplgr2vr_w(0);
                _sume = (__m256)__lasx_xvreplgr2vr_w(0);
                _sumf = (__m256)__lasx_xvreplgr2vr_w(0);
            }
            else
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
                _sumc = (__m256)__lasx_xvld(outptr + 96, 0);
                _sumd = (__m256)__lasx_xvld(outptr + 104, 0);
                _sume = (__m256)__lasx_xvld(outptr + 112, 0);
                _sumf = (__m256)__lasx_xvld(outptr + 120, 0);
            }

            const float* pA = pAT;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _pA = (__m256)__lasx_xvld(pA, 0);
                __m256 _pA1 = (__m256)__lasx_xvshuf4i_w((__m256i)_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m256 _pA2 = (__m256)__lasx_xvpermi_q((__m256i)_pA, (__m256i)_pA, _LSX_SHUFFLE(0, 0, 0, 1));
                __m256 _pA3 = (__m256)__lasx_xvpermi_q((__m256i)_pA1, (__m256i)_pA1, _LSX_SHUFFLE(0, 0, 0, 1));
                __m256 _pB0 = (__m256)__lasx_xvld(pB, 0);
                __m256 _pB1 = (__m256)__lasx_xvshuf4i_w((__m256i)_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                __m256 _pB4 = (__m256)__lasx_xvld(pB + 8, 0);
                __m256 _pB5 = (__m256)__lasx_xvshuf4i_w((__m256i)_pB4, _LSX_SHUFFLE(0, 3, 2, 1));

                _sum0 = __lasx_xvfmadd_s(_pA, _pB0, _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA, _pB1, _sum1);
                _sum2 = __lasx_xvfmadd_s(_pA1, _pB0, _sum2);
                _sum3 = __lasx_xvfmadd_s(_pA1, _pB1, _sum3);
                _sum4 = __lasx_xvfmadd_s(_pA2, _pB0, _sum4);
                _sum5 = __lasx_xvfmadd_s(_pA2, _pB1, _sum5);
                _sum6 = __lasx_xvfmadd_s(_pA3, _pB0, _sum6);
                _sum7 = __lasx_xvfmadd_s(_pA3, _pB1, _sum7);
                _sum8 = __lasx_xvfmadd_s(_pA, _pB4, _sum8);
                _sum9 = __lasx_xvfmadd_s(_pA, _pB5, _sum9);
                _suma = __lasx_xvfmadd_s(_pA1, _pB4, _suma);
                _sumb = __lasx_xvfmadd_s(_pA1, _pB5, _sumb);
                _sumc = __lasx_xvfmadd_s(_pA2, _pB4, _sumc);
                _sumd = __lasx_xvfmadd_s(_pA2, _pB5, _sumd);
                _sume = __lasx_xvfmadd_s(_pA3, _pB4, _sume);
                _sumf = __lasx_xvfmadd_s(_pA3, _pB5, _sumf);

                pA += 8;
                pB += 16;
            }

            __lasx_xvst(_sum0, outptr, 0);
            __lasx_xvst(_sum1, outptr + 8, 0);
            __lasx_xvst(_sum2, outptr + 16, 0);
            __lasx_xvst(_sum3, outptr + 24, 0);
            __lasx_xvst(_sum4, outptr + 32, 0);
            __lasx_xvst(_sum5, outptr + 40, 0);
            __lasx_xvst(_sum6, outptr + 48, 0);
            __lasx_xvst(_sum7, outptr + 56, 0);
            __lasx_xvst(_sum8, outptr + 64, 0);
            __lasx_xvst(_sum9, outptr + 72, 0);
            __lasx_xvst(_suma, outptr + 80, 0);
            __lasx_xvst(_sumb, outptr + 88, 0);
            __lasx_xvst(_sumc, outptr + 96, 0);
            __lasx_xvst(_sumd, outptr + 104, 0);
            __lasx_xvst(_sume, outptr + 112, 0);
            __lasx_xvst(_sumf, outptr + 120, 0);

            outptr += 128;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _sum0;
            __m256 _sum1;
            __m256 _sum2;
            __m256 _sum3;
            __m256 _sum4;
            __m256 _sum5;
            __m256 _sum6;
            __m256 _sum7;

            if (k == 0)
            {
                _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum1 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum2 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum3 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum4 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum5 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum6 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum7 = (__m256)__lasx_xvreplgr2vr_w(0);
            }
            else
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

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _pA = (__m256)__lasx_xvld(pA, 0);
                __m256 _pA1 = (__m256)__lasx_xvshuf4i_w((__m256i)_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m256 _pB0 = (__m256)__lasx_xvld(pB, 0);
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
                __m256 _pA = (__m256)__lasx_xvld(pA, 0);
                __m256 _pA1 = (__m256)__lasx_xvshuf4i_w((__m256i)_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _pB4 = (__m128)__lsx_vld(pB, 0);
                __m256 _pB = __lasx_concat_128_s(_pB4, _pB4);
                __m256 _pB1 = (__m256)__lasx_xvshuf4i_w((__m256i)_pB, _LSX_SHUFFLE(0, 3, 2, 1));

                _sum0 = __lasx_xvfmadd_s(_pA, _pB, _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA, _pB1, _sum1);
                _sum2 = __lasx_xvfmadd_s(_pA1, _pB, _sum2);
                _sum3 = __lasx_xvfmadd_s(_pA1, _pB1, _sum3);
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
            __m256 _sum0;
            __m256 _sum1;

            if (k == 0)
            {
                _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum1 = (__m256)__lasx_xvreplgr2vr_w(0);
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _pA = (__m256)__lasx_xvld(pA, 0);
                __m256 _pB = (__m256)__lasx_xvldrepl_d(pB, 0);
                __m256 _pB1 = (__m256)__lasx_xvshuf4i_w((__m256i)_pB, _LSX_SHUFFLE(2, 3, 0, 1));

                _sum0 = __lasx_xvfmadd_s(_pA, _pB, _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA, _pB1, _sum1);
                pA += 8;
                pB += 2;
            }

            __lasx_xvst(_sum0, outptr, 0);
            __lasx_xvst(_sum1, outptr + 8, 0);

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            __m256 _sum0;

            if (k == 0)
            {
                _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _pA = (__m256)__lasx_xvld(pA, 0);
                _sum0 = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(pB[0]), _pA, _sum0);
                pA += 8;
                pB += 1;
            }

            __lasx_xvst(_sum0, outptr, 0);

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

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pA0 = (__m128)__lsx_vld(pA, 0);
                __m128 _pA1 = (__m128)__lsx_vld(pA + 4, 0);
                __m128 _pA0r = (__m128)__lsx_vshuf4i_w((__m128i)_pA0, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _pA1r = (__m128)__lsx_vshuf4i_w((__m128i)_pA1, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _pB0 = (__m128)__lsx_vld(pB, 0);
                __m128 _pB1 = (__m128)__lsx_vld(pB + 4, 0);
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

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pA0 = (__m128)__lsx_vld(pA, 0);
                __m128 _pA1 = (__m128)__lsx_vld(pA + 4, 0);
                __m128 _pA0r = (__m128)__lsx_vshuf4i_w((__m128i)_pA0, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _pA1r = (__m128)__lsx_vshuf4i_w((__m128i)_pA1, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _pB = (__m128)__lsx_vld(pB, 0);
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

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pA0 = (__m128)__lsx_vld(pA, 0);
                __m128 _pA1 = (__m128)__lsx_vld(pA + 4, 0);
                __m128 _pB = (__m128)__lsx_vldrepl_d(pB, 0);
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
            __m128 _sum00;
            __m128 _sum01;

            if (k == 0)
            {
                _sum00 = (__m128)__lsx_vreplgr2vr_w(0);
                _sum01 = (__m128)__lsx_vreplgr2vr_w(0);
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

            __lsx_vst((__m128i)_sum00, outptr, 0);
            __lsx_vst((__m128i)_sum01, outptr + 4, 0);

            outptr += 8;
        }

#endif // __loongarch_asx
        pAT += max_kk * 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* pB = pBT;

        int jj = 0;
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _sum0;
            __m256 _sum1;
            __m256 _sum2;
            __m256 _sum3;
            __m256 _sum4;
            __m256 _sum5;
            __m256 _sum6;
            __m256 _sum7;

            if (k == 0)
            {
                _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum1 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum2 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum3 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum4 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum5 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum6 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum7 = (__m256)__lasx_xvreplgr2vr_w(0);
            }
            else
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

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pA4 = (__m128)__lsx_vld(pA, 0);
                __m128 _pA4r = (__m128)__lsx_vshuf4i_w((__m128i)_pA4, _LSX_SHUFFLE(1, 0, 3, 2));
                __m256 _pA = __lasx_concat_128_s(_pA4, _pA4);
                __m256 _pA1 = __lasx_concat_128_s(_pA4r, _pA4r);
                __m256 _pB0 = (__m256)__lasx_xvld(pB, 0);
                __m256 _pB1 = (__m256)__lasx_xvshuf4i_w((__m256i)_pB0, _LSX_SHUFFLE(0, 3, 2, 1));
                __m256 _pB2 = (__m256)__lasx_xvld(pB + 8, 0);
                __m256 _pB3 = (__m256)__lasx_xvshuf4i_w((__m256i)_pB2, _LSX_SHUFFLE(0, 3, 2, 1));

                _sum0 = __lasx_xvfmadd_s(_pA, _pB0, _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA, _pB1, _sum1);
                _sum2 = __lasx_xvfmadd_s(_pA1, _pB0, _sum2);
                _sum3 = __lasx_xvfmadd_s(_pA1, _pB1, _sum3);
                _sum4 = __lasx_xvfmadd_s(_pA, _pB2, _sum4);
                _sum5 = __lasx_xvfmadd_s(_pA, _pB3, _sum5);
                _sum6 = __lasx_xvfmadd_s(_pA1, _pB2, _sum6);
                _sum7 = __lasx_xvfmadd_s(_pA1, _pB3, _sum7);
                pA += 4;
                pB += 16;
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
#endif // __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
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
                _sum2 = (__m128)__lsx_vld(outptr + 8, 0);
                _sum3 = (__m128)__lsx_vld(outptr + 12, 0);
                _sum4 = (__m128)__lsx_vld(outptr + 16, 0);
                _sum5 = (__m128)__lsx_vld(outptr + 20, 0);
                _sum6 = (__m128)__lsx_vld(outptr + 24, 0);
                _sum7 = (__m128)__lsx_vld(outptr + 28, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pA = (__m128)__lsx_vld(pA, 0);
                __m128 _pA1 = (__m128)__lsx_vshuf4i_w((__m128i)_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _pB0 = (__m128)__lsx_vld(pB, 0);
                __m128 _pB1 = (__m128)__lsx_vld(pB + 4, 0);
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
            __lsx_vst((__m128i)_sum2, outptr + 8, 0);
            __lsx_vst((__m128i)_sum3, outptr + 12, 0);
            __lsx_vst((__m128i)_sum4, outptr + 16, 0);
            __lsx_vst((__m128i)_sum5, outptr + 20, 0);
            __lsx_vst((__m128i)_sum6, outptr + 24, 0);
            __lsx_vst((__m128i)_sum7, outptr + 28, 0);

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
                __m128 _pA1 = (__m128)__lsx_vshuf4i_w((__m128i)_pA, _LSX_SHUFFLE(1, 0, 3, 2));
                __m128 _pB = (__m128)__lsx_vld(pB, 0);
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
            __lsx_vst((__m128i)_sum2, outptr + 8, 0);
            __lsx_vst((__m128i)_sum3, outptr + 12, 0);

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
                __m128 _pB = (__m128)__lsx_vldrepl_d(pB, 0);
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
            __m128 _sum0;

            if (k == 0)
            {
                _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
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

            __lsx_vst((__m128i)_sum0, outptr, 0);

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __loongarch_sx

    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* pB = pBT;

        int jj = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
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
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
                _sum2 = (__m256)__lasx_xvld(outptr + 16, 0);
                _sum3 = (__m256)__lasx_xvld(outptr + 24, 0);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256 _pA0 = (__m256)__lasx_xvldrepl_d(pA, 0);
                __m256 _pA1 = (__m256)__lasx_xvldrepl_d(pA + 2, 0);
                __m256 _pB0 = (__m256)__lasx_xvld(pB, 0);
                __m256 _pB1 = (__m256)__lasx_xvld(pB + 8, 0);
                __m256 _pB2 = (__m256)__lasx_xvld(pB + 16, 0);
                __m256 _pB3 = (__m256)__lasx_xvld(pB + 24, 0);

                __m256 _pA00 = (__m256)__lasx_xvshuf4i_w((__m256i)_pA0, _LSX_SHUFFLE(0, 0, 0, 0));
                __m256 _pA01 = (__m256)__lasx_xvshuf4i_w((__m256i)_pA0, _LSX_SHUFFLE(1, 1, 1, 1));
                __m256 _pA10 = (__m256)__lasx_xvshuf4i_w((__m256i)_pA1, _LSX_SHUFFLE(0, 0, 0, 0));
                __m256 _pA11 = (__m256)__lasx_xvshuf4i_w((__m256i)_pA1, _LSX_SHUFFLE(1, 1, 1, 1));

                _sum0 = __lasx_xvfmadd_s(_pA00, _pB0, _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA00, _pB1, _sum1);
                _sum2 = __lasx_xvfmadd_s(_pA01, _pB0, _sum2);
                _sum3 = __lasx_xvfmadd_s(_pA01, _pB1, _sum3);
                _sum0 = __lasx_xvfmadd_s(_pA10, _pB2, _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA10, _pB3, _sum1);
                _sum2 = __lasx_xvfmadd_s(_pA11, _pB2, _sum2);
                _sum3 = __lasx_xvfmadd_s(_pA11, _pB3, _sum3);

                pA += 4;
                pB += 32;
            }
            for (; kk < max_kk; kk++)
            {
                __m256 _pA = (__m256)__lasx_xvldrepl_d(pA, 0);
                __m256 _pB0 = (__m256)__lasx_xvld(pB, 0);
                __m256 _pB1 = (__m256)__lasx_xvld(pB + 8, 0);

                __m256 _pA0 = (__m256)__lasx_xvshuf4i_w((__m256i)_pA, _LSX_SHUFFLE(0, 0, 0, 0));
                __m256 _pA1 = (__m256)__lasx_xvshuf4i_w((__m256i)_pA, _LSX_SHUFFLE(1, 1, 1, 1));

                _sum0 = __lasx_xvfmadd_s(_pA0, _pB0, _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA0, _pB1, _sum1);
                _sum2 = __lasx_xvfmadd_s(_pA1, _pB0, _sum2);
                _sum3 = __lasx_xvfmadd_s(_pA1, _pB1, _sum3);

                pA += 2;
                pB += 16;
            }

            __lasx_xvst(_sum0, outptr, 0);
            __lasx_xvst(_sum1, outptr + 8, 0);
            __lasx_xvst(_sum2, outptr + 16, 0);
            __lasx_xvst(_sum3, outptr + 24, 0);

            outptr += 32;
        }
#endif // __loongarch_asx
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

            const float* pA = pAT;
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128 _pA = (__m128)__lsx_vld(pA, 0);
                __m128 _pA0 = (__m128)__lsx_vilvl_d((__m128i)_pA, (__m128i)_pA);
                __m128 _pA1 = (__m128)__lsx_vilvh_d((__m128i)_pA, (__m128i)_pA);
                __m128 _pB0 = (__m128)__lsx_vld(pB, 0);
                __m128 _pB1 = (__m128)__lsx_vld(pB + 4, 0);
                __m128 _pB2 = (__m128)__lsx_vld(pB + 8, 0);
                __m128 _pB3 = (__m128)__lsx_vld(pB + 12, 0);
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
                const float a0 = pA[0];
                const float a1 = pA[1];
                outptr[0] += a0 * pB[0];
                outptr[1] += a1 * pB[0];
                outptr[2] += a0 * pB[1];
                outptr[3] += a1 * pB[1];
                outptr[4] += a0 * pB[2];
                outptr[5] += a1 * pB[2];
                outptr[6] += a0 * pB[3];
                outptr[7] += a1 * pB[3];
                outptr[8] += a0 * pB[4];
                outptr[9] += a1 * pB[4];
                outptr[10] += a0 * pB[5];
                outptr[11] += a1 * pB[5];
                outptr[12] += a0 * pB[6];
                outptr[13] += a1 * pB[6];
                outptr[14] += a0 * pB[7];
                outptr[15] += a1 * pB[7];
                pA += 2;
                pB += 8;
            }

            outptr += 16;
        }
#endif // __loongarch_sx
        for (; jj + 3 < max_jj; jj += 4)
        {
#if __loongarch_sx
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

            const float* pA = pAT;
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128 _pA = (__m128)__lsx_vld(pA, 0);
                __m128 _pA0 = (__m128)__lsx_vilvl_d((__m128i)_pA, (__m128i)_pA);
                __m128 _pA1 = (__m128)__lsx_vilvh_d((__m128i)_pA, (__m128i)_pA);
                __m128 _pB0 = (__m128)__lsx_vld(pB, 0);
                __m128 _pB1 = (__m128)__lsx_vld(pB + 4, 0);
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
                const float a0 = pA[0];
                const float a1 = pA[1];
                outptr[0] += a0 * pB[0];
                outptr[1] += a1 * pB[0];
                outptr[2] += a0 * pB[1];
                outptr[3] += a1 * pB[1];
                outptr[4] += a0 * pB[2];
                outptr[5] += a1 * pB[2];
                outptr[6] += a0 * pB[3];
                outptr[7] += a1 * pB[3];
                pA += 2;
                pB += 4;
            }
#else
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

            outptr[0] = sum00;
            outptr[1] = sum01;
            outptr[2] = sum10;
            outptr[3] = sum11;
            outptr[4] = sum20;
            outptr[5] = sum21;
            outptr[6] = sum30;
            outptr[7] = sum31;

#endif // __loongarch_sx
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

            outptr[0] = sum00;
            outptr[1] = sum01;
            outptr[2] = sum10;
            outptr[3] = sum11;

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

            outptr[0] = sum0;
            outptr[1] = sum1;

            outptr += 2;
        }

        pAT += max_kk * 2;
    }

    for (; ii < max_ii; ii += 1)
    {
        const float* pB = pBT;

        int jj = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256 _sum0;
            __m256 _sum1;

            if (k == 0)
            {
                _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
                _sum1 = (__m256)__lasx_xvreplgr2vr_w(0);
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
                __m256 _pA0 = __lasx_xvreplfr2vr_s(pA[0]);
                _sum0 = __lasx_xvfmadd_s(_pA0, _pB0, _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA0, _pB1, _sum1);
                pA += 1;
                pB += 16;
            }

            __lasx_xvst(_sum0, outptr, 0);
            __lasx_xvst(_sum1, outptr + 8, 0);

            outptr += 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _sum0;

            if (k == 0)
            {
                _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m256 _pB = (__m256)__lasx_xvld(pB, 0);
                _sum0 = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(pA[0]), _pB, _sum0);
                pA += 1;
                pB += 8;
            }

            __lasx_xvst(_sum0, outptr, 0);

            outptr += 8;
        }
#endif // __loongarch_asx
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

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pB0 = (__m128)__lsx_vld(pB, 0);
                __m128 _pB1 = (__m128)__lsx_vld(pB + 4, 0);
                __m128 _pA0 = __lsx_vreplfr2vr_s(pA[0]);
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
            __m128 _sum0;

            if (k == 0)
            {
                _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
            }

            const float* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                __m128 _pA = __lsx_vreplfr2vr_s(pA[0]);
                __m128 _pB = (__m128)__lsx_vld(pB, 0);
                _sum0 = __lsx_vfmadd_s(_pA, _pB, _sum0);
                pA += 1;
                pB += 4;
            }

            __lsx_vst((__m128i)_sum0, outptr, 0);
#else
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

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr[2] = sum2;
            outptr[3] = sum3;

#endif // __loongarch_sx
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

            outptr[0] = sum0;
            outptr[1] = sum1;

            outptr += 2;
        }

        for (; jj < max_jj; jj += 1)
        {
            float sum0;

            if (k == 0)
            {
                sum0 = 0.f;
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

            outptr[0] = sum0;

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
    TILE_M = std::max(8, tile_size / 8 * 8);
#if __loongarch_asx
    TILE_N = std::max(16, tile_size / 16 * 16);
#else
    TILE_N = std::max(8, tile_size / 8 * 8);
#endif
#if __loongarch_asx
    TILE_K = std::max(8, tile_size / 8 * 8);
#else
    TILE_K = std::max(4, tile_size / 4 * 4);
#endif
#else
    TILE_M = std::max(2, tile_size / 2 * 2);
    TILE_N = std::max(4, tile_size / 4 * 4);
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
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(float) / TILE_K);
#if __loongarch_sx
            TILE_M = std::max(8, tile_size / 8 * 8);
#if __loongarch_asx
            TILE_N = std::max(16, tile_size / 16 * 16);
#else
            TILE_N = std::max(8, tile_size / 8 * 8);
#endif
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
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __loongarch_sx
#if __loongarch_asx
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 15) / 16 * 16);
#else
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8);
#endif
#else
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
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
#if __loongarch_asx
        TILE_N = (constant_TILE_N + 15) / 16 * 16;
#else
        TILE_N = (constant_TILE_N + 7) / 8 * 8;
#endif
#else
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
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

static int gemm_loongarch(const Mat& A, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int transA, int transB, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
    const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    int TILE_M, TILE_N, TILE_K;
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

        Mat BT_tile = BT.channel(j / TILE_N).row_range(ppk, 1);

        if (transB)
            pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
        else
            transpose_pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
    }

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    Mat CT;
    if (broadcast_type_C == 3)
    {
        CT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (CT.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;
        const int M_local = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
        const int K_local = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;
        const int max_ii = std::min(M_local - i, TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());
        Mat CT_tile;
        if (broadcast_type_C == 3)
            CT_tile = CT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            if (broadcast_type_C == 3)
            {
                pack_A_tile(C, CT_tile, i, max_ii, j, max_jj);
            }

            const Mat& C_tile = broadcast_type_C == 3 ? CT_tile : C;

            for (int k = 0; k < K_local; k += TILE_K)
            {
                const int max_kk = std::min(K_local - k, TILE_K);

                Mat AT_tile = ATX.channel(get_omp_thread_num()).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                if (j == 0)
                {
                    if (transA)
                    {
                        transpose_pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
                    }
                    else
                    {
                        pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
                    }
                }

                gemm_transB_packed_tile(AT_tile, BT_tile, topT_tile, max_ii, max_jj, k, max_kk);
            }

            unpack_output_tile(topT_tile, C_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, 1.f, 1.f, output_transpose, 1);
        }
    }

    return 0;
}

static int gemm_AT_loongarch(const Mat& AT, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int K, int transB, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    int TILE_M, TILE_N, TILE_K;
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

        Mat BT_tile = BT.channel(j / TILE_N).row_range(ppk, 1);

        if (transB)
            pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
        else
            transpose_pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
    }

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    Mat CT;
    if (broadcast_type_C == 3)
    {
        CT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (CT.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;
        const int max_ii = std::min(M - i, TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());
        Mat CT_tile;
        if (broadcast_type_C == 3)
            CT_tile = CT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            if (broadcast_type_C == 3)
            {
                pack_A_tile(C, CT_tile, i, max_ii, j, max_jj);
            }

            const Mat& C_tile = broadcast_type_C == 3 ? CT_tile : C;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min(K - k, TILE_K);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                gemm_transB_packed_tile(AT_tile, BT_tile, topT_tile, max_ii, max_jj, k, max_kk);
            }

            unpack_output_tile(topT_tile, C_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, 1.f, 1.f, output_transpose, 1);
        }
    }

    return 0;
}

static int gemm_BT_loongarch(const Mat& A, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int N, int K, int transA, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat ATX(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, nT, 4u, opt.workspace_allocator);
    if (ATX.empty())
        return -100;

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    Mat CT;
    if (broadcast_type_C == 3)
    {
        CT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (CT.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;
        const int M_local = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
        const int K_local = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;
        const int max_ii = std::min(M_local - i, TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());
        Mat CT_tile;
        if (broadcast_type_C == 3)
            CT_tile = CT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            if (broadcast_type_C == 3)
            {
                pack_A_tile(C, CT_tile, i, max_ii, j, max_jj);
            }

            const Mat& C_tile = broadcast_type_C == 3 ? CT_tile : C;

            for (int k = 0; k < K_local; k += TILE_K)
            {
                const int max_kk = std::min(K_local - k, TILE_K);

                Mat AT_tile = ATX.channel(get_omp_thread_num()).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                if (j == 0)
                {
                    if (transA)
                    {
                        transpose_pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
                    }
                    else
                    {
                        pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
                    }
                }

                gemm_transB_packed_tile(AT_tile, BT_tile, topT_tile, max_ii, max_jj, k, max_kk);
            }

            unpack_output_tile(topT_tile, C_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, 1.f, 1.f, output_transpose, 1);
        }
    }

    return 0;
}

static int gemm_AT_BT_loongarch(const Mat& AT, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int N, int K, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    Mat CT;
    if (broadcast_type_C == 3)
    {
        CT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (CT.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;
        const int max_ii = std::min(M - i, TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());
        Mat CT_tile;
        if (broadcast_type_C == 3)
            CT_tile = CT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            if (broadcast_type_C == 3)
            {
                pack_A_tile(C, CT_tile, i, max_ii, j, max_jj);
            }

            const Mat& C_tile = broadcast_type_C == 3 ? CT_tile : C;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min(K - k, TILE_K);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                gemm_transB_packed_tile(AT_tile, BT_tile, topT_tile, max_ii, max_jj, k, max_kk);
            }

            unpack_output_tile(topT_tile, C_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, 1.f, 1.f, output_transpose, 1);
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

#if NCNN_INT8
    if (int8_scale_term)
    {
        return create_pipeline_int8(opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage)
    {
        return create_pipeline_bf16s(opt);
    }
#endif

    if (constantA)
    {
        const int M = constantM;
        const int K = constantK;
        int TILE_M, TILE_N, TILE_K;
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

            Mat AT_tile = AT_data.channel(i / TILE_M).row_range(k / TILE_K, 1);

            if (transA)
            {
                transpose_pack_A_tile(A_data, AT_tile, i, max_ii, k, max_kk);
            }
            else
            {
                pack_A_tile(A_data, AT_tile, i, max_ii, k, max_kk);
            }
        }

        if (opt.lightmode)
            A_data.release();
    }

    if (constantB)
    {
        const int N = constantN;
        const int K = constantK;

        int TILE_M, TILE_N, TILE_K;
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

            Mat BT_tile = BT_data.channel(j / TILE_N).row_range(k / TILE_K, 1);

            if (transB)
            {
                pack_B_tile(B_data, BT_tile, j, max_jj, k, max_kk);
            }
            else
            {
                transpose_pack_B_tile(B_data, BT_tile, j, max_jj, k, max_kk);
            }
        }

        if (opt.lightmode)
            B_data.release();
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

        if (opt.lightmode)
            C_data.release();
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
#if NCNN_INT8
    if (int8_scale_term)
    {
        return forward_int8(bottom_blobs, top_blobs, opt);
    }
#endif
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

    int out_elempack = 1;
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
        const int outh = output_transpose ? N : M;
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

#if NCNN_INT8
int Gemm_loongarch::create_pipeline_int8(const Option& opt)
{
#if __loongarch_sx
    support_packing = true;
#else
    support_packing = false;
#endif
    support_bf16_storage = false;

    if (constantA)
    {
        const int M = constantM;
        const int K = constantK;

        int TILE_M, TILE_N, TILE_K;
        get_optimal_tile_mnk_int8(M, 0, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, opt.num_threads);

        const int nn_M = (M + TILE_M - 1) / TILE_M;
        const int nn_K = (K + TILE_K - 1) / TILE_K;

        AT_data.create(TILE_K * TILE_M, nn_K, nn_M, 1u, (Allocator*)0);
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

            Mat AT_tile = AT_data.channel(i / TILE_M).row_range(ppk, 1);

            if (transA)
                transpose_pack_A_tile_int8(A_data, AT_tile, i, max_ii, k, max_kk);
            else
                pack_A_tile_int8(A_data, AT_tile, i, max_ii, k, max_kk);
        }

        if (opt.lightmode)
            A_data.release();
    }

    if (constantB)
    {
        const int N = constantN;
        const int K = constantK;

        int TILE_M, TILE_N, TILE_K;
        get_optimal_tile_mnk_int8(0, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, opt.num_threads);

        const int nn_N = (N + TILE_N - 1) / TILE_N;
        const int nn_K = (K + TILE_K - 1) / TILE_K;

        BT_data.create(TILE_K * TILE_N, nn_K, nn_N, 1u, (Allocator*)0);
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

            Mat BT_tile = BT_data.channel(j / TILE_N).row_range(ppk, 1);

            if (transB)
                pack_B_tile_int8(B_data, BT_tile, j, max_jj, k, max_kk);
            else
                transpose_pack_B_tile_int8(B_data, BT_tile, j, max_jj, k, max_kk);
        }

        if (opt.lightmode)
            B_data.release();
    }

    if (constantC && constant_broadcast_type_C != -1)
    {
        CT_data = C_data;

        if (opt.lightmode)
            C_data.release();
    }

    if (constantA || constantB || constantC)
    {
        nT = opt.num_threads;
    }

    return 0;
}

int Gemm_loongarch::forward_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
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
            C = bottom_blobs.size() == 1 ? bottom_blobs[0] : Mat();
        else if (constantA || constantB)
            C = bottom_blobs.size() == 2 ? bottom_blobs[1] : Mat();
        else
            C = bottom_blobs.size() == 3 ? bottom_blobs[2] : Mat();

        if (!C.empty())
            broadcast_type_C = resolve_broadcast_type_C(C, M, N);
    }

    int out_elempack = 1;
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
        const int outh = output_transpose ? N : M;
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
        NCNN_LOGE("opt.num_threads %d changed, gemm will use load-time value %d", opt.num_threads, nT);
    }

    int ret = 0;
    if (constantA && constantB)
    {
        ret = gemm_AT_BT_loongarch_int8(AT_data, A_data_int8_scales, BT_data, B_data_int8_scale, C, top_blob, broadcast_type_C, constantM, constantN, constantK, output_transpose, alpha, beta, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantA)
    {
        const Mat& B = bottom_blobs[0];
        ret = gemm_AT_loongarch_int8(AT_data, A_data_int8_scales, B, C, top_blob, broadcast_type_C, constantM, constantK, transB, output_transpose, alpha, beta, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantB)
    {
        const Mat& A = bottom_blobs[0];
        ret = gemm_BT_loongarch_int8(A, BT_data, B_data_int8_scale, C, top_blob, broadcast_type_C, constantN, constantK, transA, output_transpose, alpha, beta, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else
    {
        const Mat& A = bottom_blobs[0];
        const Mat& B = bottom_blobs[1];
        ret = gemm_loongarch_int8(A, B, C, top_blob, broadcast_type_C, transA, transB, output_transpose, alpha, beta, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }

    return ret;
}

#endif

namespace Gemm_loongarch_utility {
#if NCNN_INT8
void pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    ncnn::pack_A_tile_int8(A, AT, i, max_ii, k, max_kk);
}

void gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
    ncnn::gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
}
#endif
} // namespace Gemm_loongarch_utility

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

    Mat CT;
    if (broadcast_type_C == 3)
    {
        CT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (CT.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());
        Mat CT_tile;
        if (broadcast_type_C == 3)
            CT_tile = CT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            if (broadcast_type_C == 3)
            {
                pack_A_tile(C, CT_tile, i, max_ii, j, max_jj);
            }

            const Mat& C_tile = broadcast_type_C == 3 ? CT_tile : C;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile(topT_tile, C_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta, output_transpose, output_elemtype);
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

    Mat CT;
    if (broadcast_type_C == 3)
    {
        CT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (CT.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());
        Mat CT_tile;
        if (broadcast_type_C == 3)
            CT_tile = CT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            if (broadcast_type_C == 3)
            {
                pack_A_tile(C, CT_tile, i, max_ii, j, max_jj);
            }

            const Mat& C_tile = broadcast_type_C == 3 ? CT_tile : C;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                // AT is pre-packed bf16
                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile(topT_tile, C_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta, output_transpose, output_elemtype);
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

    Mat CT;
    if (broadcast_type_C == 3)
    {
        CT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (CT.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;

        const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
        const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;

        const int max_ii = std::min((M - i), TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());
        Mat CT_tile;
        if (broadcast_type_C == 3)
            CT_tile = CT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            if (broadcast_type_C == 3)
            {
                pack_A_tile(C, CT_tile, i, max_ii, j, max_jj);
            }

            const Mat& C_tile = broadcast_type_C == 3 ? CT_tile : C;

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

            unpack_output_tile(topT_tile, C_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta, output_transpose, output_elemtype);
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

    Mat CT;
    if (broadcast_type_C == 3)
    {
        CT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (CT.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int i = ppi * TILE_M;

        const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
        const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;

        const int max_ii = std::min((M - i), TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());
        Mat CT_tile;
        if (broadcast_type_C == 3)
            CT_tile = CT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            if (broadcast_type_C == 3)
            {
                pack_A_tile(C, CT_tile, i, max_ii, j, max_jj);
            }

            const Mat& C_tile = broadcast_type_C == 3 ? CT_tile : C;

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

            unpack_output_tile(topT_tile, C_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta, output_transpose, output_elemtype);
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
#if __loongarch_asx
        out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
        out_elempack = outh % 4 == 0 ? 4 : 1;
#endif
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

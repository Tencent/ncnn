// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gemm_x86.h"

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"

#include "cpu.h"

namespace ncnn {

#if NCNN_INT8
#include "gemm_int8.h"
#endif

Gemm_x86::Gemm_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__

    nT = 0;
}

static void pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    float* pp = AT;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        if (elempack == 16)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 16;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm512_store_ps(pp, _mm512_load_ps(p0));
                pp += 16;
                p0 += 16;
            }
        }
        if (elempack == 8)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 8;
            const float* p1 = (const float*)A + (i + ii + 8) * A_hstep + k * 8;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm256_store_ps(pp, _mm256_load_ps(p0));
                _mm256_store_ps(pp + 8, _mm256_load_ps(p1));
                pp += 16;
                p0 += 8;
                p1 += 8;
            }
        }
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;
            const float* p1 = (const float*)A + (i + ii + 4) * A_hstep + k * 4;
            const float* p2 = (const float*)A + (i + ii + 8) * A_hstep + k * 4;
            const float* p3 = (const float*)A + (i + ii + 12) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, _mm_load_ps(p0));
                _mm_store_ps(pp + 4, _mm_load_ps(p1));
                _mm_store_ps(pp + 8, _mm_load_ps(p2));
                _mm_store_ps(pp + 12, _mm_load_ps(p3));
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
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
            const float* p8 = (const float*)A + (i + ii + 8) * A_hstep + k;
            const float* p9 = (const float*)A + (i + ii + 9) * A_hstep + k;
            const float* pa = (const float*)A + (i + ii + 10) * A_hstep + k;
            const float* pb = (const float*)A + (i + ii + 11) * A_hstep + k;
            const float* pc = (const float*)A + (i + ii + 12) * A_hstep + k;
            const float* pd = (const float*)A + (i + ii + 13) * A_hstep + k;
            const float* pe = (const float*)A + (i + ii + 14) * A_hstep + k;
            const float* pf = (const float*)A + (i + ii + 15) * A_hstep + k;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_loadu_ps(p0);
                __m512 _r1 = _mm512_loadu_ps(p1);
                __m512 _r2 = _mm512_loadu_ps(p2);
                __m512 _r3 = _mm512_loadu_ps(p3);
                __m512 _r4 = _mm512_loadu_ps(p4);
                __m512 _r5 = _mm512_loadu_ps(p5);
                __m512 _r6 = _mm512_loadu_ps(p6);
                __m512 _r7 = _mm512_loadu_ps(p7);
                __m512 _r8 = _mm512_loadu_ps(p8);
                __m512 _r9 = _mm512_loadu_ps(p9);
                __m512 _ra = _mm512_loadu_ps(pa);
                __m512 _rb = _mm512_loadu_ps(pb);
                __m512 _rc = _mm512_loadu_ps(pc);
                __m512 _rd = _mm512_loadu_ps(pd);
                __m512 _re = _mm512_loadu_ps(pe);
                __m512 _rf = _mm512_loadu_ps(pf);
                transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                _mm512_store_ps(pp, _r0);
                _mm512_store_ps(pp + 16, _r1);
                _mm512_store_ps(pp + 16 * 2, _r2);
                _mm512_store_ps(pp + 16 * 3, _r3);
                _mm512_store_ps(pp + 16 * 4, _r4);
                _mm512_store_ps(pp + 16 * 5, _r5);
                _mm512_store_ps(pp + 16 * 6, _r6);
                _mm512_store_ps(pp + 16 * 7, _r7);
                _mm512_store_ps(pp + 16 * 8, _r8);
                _mm512_store_ps(pp + 16 * 9, _r9);
                _mm512_store_ps(pp + 16 * 10, _ra);
                _mm512_store_ps(pp + 16 * 11, _rb);
                _mm512_store_ps(pp + 16 * 12, _rc);
                _mm512_store_ps(pp + 16 * 13, _rd);
                _mm512_store_ps(pp + 16 * 14, _re);
                _mm512_store_ps(pp + 16 * 15, _rf);
                pp += 256;
                p0 += 16;
                p1 += 16;
                p2 += 16;
                p3 += 16;
                p4 += 16;
                p5 += 16;
                p6 += 16;
                p7 += 16;
                p8 += 16;
                p9 += 16;
                pa += 16;
                pb += 16;
                pc += 16;
                pd += 16;
                pe += 16;
                pf += 16;
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
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 8)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 8;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm256_store_ps(pp, _mm256_load_ps(p0));
                pp += 8;
                p0 += 8;
            }
        }
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;
            const float* p1 = (const float*)A + (i + ii + 4) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, _mm_load_ps(p0));
                _mm_store_ps(pp + 4, _mm_load_ps(p1));
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
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_loadu_ps(p0);
                __m256 _r1 = _mm256_loadu_ps(p1);
                __m256 _r2 = _mm256_loadu_ps(p2);
                __m256 _r3 = _mm256_loadu_ps(p3);
                __m256 _r4 = _mm256_loadu_ps(p4);
                __m256 _r5 = _mm256_loadu_ps(p5);
                __m256 _r6 = _mm256_loadu_ps(p6);
                __m256 _r7 = _mm256_loadu_ps(p7);
                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm256_store_ps(pp, _r0);
                _mm256_store_ps(pp + 8, _r1);
                _mm256_store_ps(pp + 8 * 2, _r2);
                _mm256_store_ps(pp + 8 * 3, _r3);
                _mm256_store_ps(pp + 8 * 4, _r4);
                _mm256_store_ps(pp + 8 * 5, _r5);
                _mm256_store_ps(pp + 8 * 6, _r6);
                _mm256_store_ps(pp + 8 * 7, _r7);
                pp += 64;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
                p4 += 8;
                p5 += 8;
                p6 += 8;
                p7 += 8;
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
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, _mm_load_ps(p0));
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
#if __AVX__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_loadu_ps(p0);
                __m256 _r1 = _mm256_loadu_ps(p1);
                __m256 _r2 = _mm256_loadu_ps(p2);
                __m256 _r3 = _mm256_loadu_ps(p3);
                transpose8x4_ps(_r0, _r1, _r2, _r3);
                _mm256_store_ps(pp, _r0);
                _mm256_store_ps(pp + 8, _r1);
                _mm256_store_ps(pp + 16, _r2);
                _mm256_store_ps(pp + 24, _r3);
                pp += 32;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
            }
#endif // __AVX__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_loadu_ps(p0);
                __m128 _r1 = _mm_loadu_ps(p1);
                __m128 _r2 = _mm_loadu_ps(p2);
                __m128 _r3 = _mm_loadu_ps(p3);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_store_ps(pp, _r0);
                _mm_store_ps(pp + 4, _r1);
                _mm_store_ps(pp + 8, _r2);
                _mm_store_ps(pp + 12, _r3);
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
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

            int kk = 0;
#if __SSE2__
#if __AVX__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_loadu_ps(p0);
                __m256 _r1 = _mm256_loadu_ps(p1);
                transpose8x2_ps(_r0, _r1);
                _mm256_storeu_ps(pp, _r0);
                _mm256_storeu_ps(pp + 8, _r1);
                pp += 16;
                p0 += 8;
                p1 += 8;
            }
#endif // __AVX__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_loadu_ps(p0);
                __m128 _r1 = _mm_loadu_ps(p1);
                __m128 _tmp0 = _mm_unpacklo_ps(_r0, _r1);
                __m128 _tmp1 = _mm_unpackhi_ps(_r0, _r1);
                _mm_store_ps(pp, _tmp0);
                _mm_store_ps(pp + 4, _tmp1);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk++)
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
#if __SSE2__
#if __AVX__
            for (; kk + 7 < max_kk; kk += 8)
            {
                _mm256_storeu_ps(pp, _mm256_loadu_ps(p0));
                pp += 8;
                p0 += 8;
            }
#endif // __AVX__
            for (; kk + 3 < max_kk; kk += 4)
            {
                _mm_storeu_ps(pp, _mm_loadu_ps(p0));
                pp += 4;
                p0 += 4;
            }
#endif // __SSE2__
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
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        if (elempack == 16)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_load_ps(p0);
                __m512 _r1 = _mm512_load_ps(p0 + 16 * 1);
                __m512 _r2 = _mm512_load_ps(p0 + 16 * 2);
                __m512 _r3 = _mm512_load_ps(p0 + 16 * 3);
                __m512 _r4 = _mm512_load_ps(p0 + 16 * 4);
                __m512 _r5 = _mm512_load_ps(p0 + 16 * 5);
                __m512 _r6 = _mm512_load_ps(p0 + 16 * 6);
                __m512 _r7 = _mm512_load_ps(p0 + 16 * 7);
                __m512 _r8 = _mm512_load_ps(p0 + 16 * 8);
                __m512 _r9 = _mm512_load_ps(p0 + 16 * 9);
                __m512 _ra = _mm512_load_ps(p0 + 16 * 10);
                __m512 _rb = _mm512_load_ps(p0 + 16 * 11);
                __m512 _rc = _mm512_load_ps(p0 + 16 * 12);
                __m512 _rd = _mm512_load_ps(p0 + 16 * 13);
                __m512 _re = _mm512_load_ps(p0 + 16 * 14);
                __m512 _rf = _mm512_load_ps(p0 + 16 * 15);
                transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                _mm512_store_ps(pp, _r0);
                _mm512_store_ps(pp + 16 * 1, _r1);
                _mm512_store_ps(pp + 16 * 2, _r2);
                _mm512_store_ps(pp + 16 * 3, _r3);
                _mm512_store_ps(pp + 16 * 4, _r4);
                _mm512_store_ps(pp + 16 * 5, _r5);
                _mm512_store_ps(pp + 16 * 6, _r6);
                _mm512_store_ps(pp + 16 * 7, _r7);
                _mm512_store_ps(pp + 16 * 8, _r8);
                _mm512_store_ps(pp + 16 * 9, _r9);
                _mm512_store_ps(pp + 16 * 10, _ra);
                _mm512_store_ps(pp + 16 * 11, _rb);
                _mm512_store_ps(pp + 16 * 12, _rc);
                _mm512_store_ps(pp + 16 * 13, _rd);
                _mm512_store_ps(pp + 16 * 14, _re);
                _mm512_store_ps(pp + 16 * 15, _rf);
                pp += 256;
                p0 += A_hstep * 16;
            }
        }
        if (elempack == 8)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_load_ps(p0);
                __m256 _r1 = _mm256_load_ps(p0 + 8 * 1);
                __m256 _r2 = _mm256_load_ps(p0 + 8 * 2);
                __m256 _r3 = _mm256_load_ps(p0 + 8 * 3);
                __m256 _r4 = _mm256_load_ps(p0 + 8 * 4);
                __m256 _r5 = _mm256_load_ps(p0 + 8 * 5);
                __m256 _r6 = _mm256_load_ps(p0 + 8 * 6);
                __m256 _r7 = _mm256_load_ps(p0 + 8 * 7);
                __m256 _r8 = _mm256_load_ps(p0 + 8 * 8);
                __m256 _r9 = _mm256_load_ps(p0 + 8 * 9);
                __m256 _ra = _mm256_load_ps(p0 + 8 * 10);
                __m256 _rb = _mm256_load_ps(p0 + 8 * 11);
                __m256 _rc = _mm256_load_ps(p0 + 8 * 12);
                __m256 _rd = _mm256_load_ps(p0 + 8 * 13);
                __m256 _re = _mm256_load_ps(p0 + 8 * 14);
                __m256 _rf = _mm256_load_ps(p0 + 8 * 15);

                transpose8x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                __m512 _rr0 = combine8x2_ps(_r0, _r1);
                __m512 _rr1 = combine8x2_ps(_r2, _r3);
                __m512 _rr2 = combine8x2_ps(_r4, _r5);
                __m512 _rr3 = combine8x2_ps(_r6, _r7);
                __m512 _rr4 = combine8x2_ps(_r8, _r9);
                __m512 _rr5 = combine8x2_ps(_ra, _rb);
                __m512 _rr6 = combine8x2_ps(_rc, _rd);
                __m512 _rr7 = combine8x2_ps(_re, _rf);

                _mm512_store_ps(pp, _rr0);
                _mm512_store_ps(pp + 16 * 1, _rr1);
                _mm512_store_ps(pp + 16 * 2, _rr2);
                _mm512_store_ps(pp + 16 * 3, _rr3);
                _mm512_store_ps(pp + 16 * 4, _rr4);
                _mm512_store_ps(pp + 16 * 5, _rr5);
                _mm512_store_ps(pp + 16 * 6, _rr6);
                _mm512_store_ps(pp + 16 * 7, _rr7);
                pp += 128;
                p0 += A_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_load_ps(p0);
                __m128 _r1 = _mm_load_ps(p0 + 4 * 1);
                __m128 _r2 = _mm_load_ps(p0 + 4 * 2);
                __m128 _r3 = _mm_load_ps(p0 + 4 * 3);
                __m128 _r4 = _mm_load_ps(p0 + 4 * 4);
                __m128 _r5 = _mm_load_ps(p0 + 4 * 5);
                __m128 _r6 = _mm_load_ps(p0 + 4 * 6);
                __m128 _r7 = _mm_load_ps(p0 + 4 * 7);
                __m128 _r8 = _mm_load_ps(p0 + 4 * 8);
                __m128 _r9 = _mm_load_ps(p0 + 4 * 9);
                __m128 _ra = _mm_load_ps(p0 + 4 * 10);
                __m128 _rb = _mm_load_ps(p0 + 4 * 11);
                __m128 _rc = _mm_load_ps(p0 + 4 * 12);
                __m128 _rd = _mm_load_ps(p0 + 4 * 13);
                __m128 _re = _mm_load_ps(p0 + 4 * 14);
                __m128 _rf = _mm_load_ps(p0 + 4 * 15);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);
                _MM_TRANSPOSE4_PS(_rc, _rd, _re, _rf);
                _mm_store_ps(pp, _r0);
                _mm_store_ps(pp + 4 * 1, _r4);
                _mm_store_ps(pp + 4 * 2, _r8);
                _mm_store_ps(pp + 4 * 3, _rc);
                _mm_store_ps(pp + 4 * 4, _r1);
                _mm_store_ps(pp + 4 * 5, _r5);
                _mm_store_ps(pp + 4 * 6, _r9);
                _mm_store_ps(pp + 4 * 7, _rd);
                _mm_store_ps(pp + 4 * 8, _r2);
                _mm_store_ps(pp + 4 * 9, _r6);
                _mm_store_ps(pp + 4 * 10, _ra);
                _mm_store_ps(pp + 4 * 11, _re);
                _mm_store_ps(pp + 4 * 12, _r3);
                _mm_store_ps(pp + 4 * 13, _r7);
                _mm_store_ps(pp + 4 * 14, _rb);
                _mm_store_ps(pp + 4 * 15, _rf);
                pp += 64;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                _mm512_store_ps(pp, _mm512_loadu_ps(p0));
                pp += 16;
                p0 += A_hstep;
            }
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
#if __AVX512F__
        if (elempack == 16)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_load_ps(p0);
                __m512 _r1 = _mm512_load_ps(p0 + 16 * 1);
                __m512 _r2 = _mm512_load_ps(p0 + 16 * 2);
                __m512 _r3 = _mm512_load_ps(p0 + 16 * 3);
                __m512 _r4 = _mm512_load_ps(p0 + 16 * 4);
                __m512 _r5 = _mm512_load_ps(p0 + 16 * 5);
                __m512 _r6 = _mm512_load_ps(p0 + 16 * 6);
                __m512 _r7 = _mm512_load_ps(p0 + 16 * 7);
                transpose16x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm512_store_ps(pp, _r0);
                _mm512_store_ps(pp + 16 * 1, _r1);
                _mm512_store_ps(pp + 16 * 2, _r2);
                _mm512_store_ps(pp + 16 * 3, _r3);
                _mm512_store_ps(pp + 16 * 4, _r4);
                _mm512_store_ps(pp + 16 * 5, _r5);
                _mm512_store_ps(pp + 16 * 6, _r6);
                _mm512_store_ps(pp + 16 * 7, _r7);
                pp += 128;
                p0 += A_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_load_ps(p0);
                __m256 _r1 = _mm256_load_ps(p0 + 8 * 1);
                __m256 _r2 = _mm256_load_ps(p0 + 8 * 2);
                __m256 _r3 = _mm256_load_ps(p0 + 8 * 3);
                __m256 _r4 = _mm256_load_ps(p0 + 8 * 4);
                __m256 _r5 = _mm256_load_ps(p0 + 8 * 5);
                __m256 _r6 = _mm256_load_ps(p0 + 8 * 6);
                __m256 _r7 = _mm256_load_ps(p0 + 8 * 7);
                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm256_store_ps(pp, _r0);
                _mm256_store_ps(pp + 8 * 1, _r1);
                _mm256_store_ps(pp + 8 * 2, _r2);
                _mm256_store_ps(pp + 8 * 3, _r3);
                _mm256_store_ps(pp + 8 * 4, _r4);
                _mm256_store_ps(pp + 8 * 5, _r5);
                _mm256_store_ps(pp + 8 * 6, _r6);
                _mm256_store_ps(pp + 8 * 7, _r7);
                pp += 64;
                p0 += A_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_load_ps(p0);
                __m128 _r1 = _mm_load_ps(p0 + 4 * 1);
                __m128 _r2 = _mm_load_ps(p0 + 4 * 2);
                __m128 _r3 = _mm_load_ps(p0 + 4 * 3);
                __m128 _r4 = _mm_load_ps(p0 + 4 * 4);
                __m128 _r5 = _mm_load_ps(p0 + 4 * 5);
                __m128 _r6 = _mm_load_ps(p0 + 4 * 6);
                __m128 _r7 = _mm_load_ps(p0 + 4 * 7);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                _mm_store_ps(pp, _r0);
                _mm_store_ps(pp + 4 * 1, _r4);
                _mm_store_ps(pp + 4 * 2, _r1);
                _mm_store_ps(pp + 4 * 3, _r5);
                _mm_store_ps(pp + 4 * 4, _r2);
                _mm_store_ps(pp + 4 * 5, _r6);
                _mm_store_ps(pp + 4 * 6, _r3);
                _mm_store_ps(pp + 4 * 7, _r7);
                pp += 32;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                _mm256_store_ps(pp, _mm256_loadu_ps(p0));
                pp += 8;
                p0 += A_hstep;
            }
        }
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_load_ps(p0);
                __m512 _r1 = _mm512_load_ps(p0 + 16 * 1);
                __m512 _r2 = _mm512_load_ps(p0 + 16 * 2);
                __m512 _r3 = _mm512_load_ps(p0 + 16 * 3);
                transpose16x4_ps(_r0, _r1, _r2, _r3);
                _mm512_store_ps(pp, _r0);
                _mm512_store_ps(pp + 16 * 1, _r1);
                _mm512_store_ps(pp + 16 * 2, _r2);
                _mm512_store_ps(pp + 16 * 3, _r3);
                pp += 64;
                p0 += A_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_load_ps(p0);
                __m256 _r1 = _mm256_load_ps(p0 + 8 * 1);
                __m256 _r2 = _mm256_load_ps(p0 + 8 * 2);
                __m256 _r3 = _mm256_load_ps(p0 + 8 * 3);
                transpose8x4_ps(_r0, _r1, _r2, _r3);
                _mm256_store_ps(pp, _r0);
                _mm256_store_ps(pp + 8 * 1, _r1);
                _mm256_store_ps(pp + 8 * 2, _r2);
                _mm256_store_ps(pp + 8 * 3, _r3);
                pp += 32;
                p0 += A_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_load_ps(p0);
                __m128 _r1 = _mm_load_ps(p0 + 4 * 1);
                __m128 _r2 = _mm_load_ps(p0 + 4 * 2);
                __m128 _r3 = _mm_load_ps(p0 + 4 * 3);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_store_ps(pp, _r0);
                _mm_store_ps(pp + 4 * 1, _r1);
                _mm_store_ps(pp + 4 * 2, _r2);
                _mm_store_ps(pp + 4 * 3, _r3);
                pp += 16;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, _mm_loadu_ps(p0));
                pp += 4;
                p0 += A_hstep;
            }
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_load_ps(p0);
                __m512 _r1 = _mm512_load_ps(p0 + 16);
                transpose16x2_ps(_r0, _r1);
                _mm512_store_ps(pp, _r0);
                _mm512_store_ps(pp + 16, _r1);
                pp += 32;
                p0 += A_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_load_ps(p0);
                __m256 _r1 = _mm256_load_ps(p0 + 8);
                transpose8x2_ps(_r0, _r1);
                _mm256_store_ps(pp, _r0);
                _mm256_store_ps(pp + 8, _r1);
                pp += 16;
                p0 += A_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_load_ps(p0);
                __m128 _r1 = _mm_load_ps(p0 + 4);
                __m128 _tmp0 = _mm_unpacklo_ps(_r0, _r1);
                __m128 _tmp1 = _mm_unpackhi_ps(_r0, _r1);
                _mm_store_ps(pp, _tmp0);
                _mm_store_ps(pp + 4, _tmp1);
                pp += 8;
                p0 += A_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
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
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                _mm512_store_ps(pp, _mm512_load_ps(p0));
                pp += 16;
                p0 += A_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                _mm256_store_ps(pp, _mm256_load_ps(p0));
                pp += 8;
                p0 += A_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                _mm_store_ps(pp, _mm_load_ps(p0));
                pp += 4;
                p0 += A_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
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
#if __SSE2__
    for (; jj + 11 < max_jj; jj += 12)
    {
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const float* p0 = (const float*)B + (j + jj) / 16 * 16 * B_hstep + k * 16;
            const float* p1 = (const float*)B + ((j + jj) / 16 * 16 + 16) * B_hstep + k * 16;

            if ((j + jj) % 16 == 0)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm256_storeu_ps(pp, _mm256_load_ps(p0));
                    _mm_store_ps(pp + 8, _mm_load_ps(p0 + 8));
                    pp += 12;
                    p0 += 16;
                }
            }
            if ((j + jj) % 16 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm256_storeu_ps(pp, _mm256_loadu_ps(p0 + 4));
                    _mm_store_ps(pp + 8, _mm_load_ps(p0 + 12));
                    pp += 12;
                    p0 += 16;
                }
            }
            if ((j + jj) % 16 == 8)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm256_storeu_ps(pp, _mm256_load_ps(p0 + 8));
                    _mm_store_ps(pp + 8, _mm_load_ps(p1));
                    pp += 12;
                    p0 += 16;
                    p1 += 16;
                }
            }
            if ((j + jj) % 16 == 12)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm_store_ps(pp, _mm_load_ps(p0 + 12));
                    _mm256_storeu_ps(pp + 4, _mm256_load_ps(p1));
                    pp += 12;
                    p0 += 16;
                    p1 += 16;
                }
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const float* p0 = (const float*)B + (j + jj) / 8 * 8 * B_hstep + k * 8;
            const float* p1 = (const float*)B + ((j + jj) / 8 * 8 + 8) * B_hstep + k * 8;

            if ((j + jj) % 8 == 0)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm256_storeu_ps(pp, _mm256_load_ps(p0));
                    _mm_store_ps(pp + 8, _mm_load_ps(p1));
                    pp += 12;
                    p0 += 8;
                    p1 += 8;
                }
            }
            if ((j + jj) % 8 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm_store_ps(pp, _mm_load_ps(p0 + 4));
                    _mm256_storeu_ps(pp + 4, _mm256_load_ps(p1));
                    pp += 12;
                    p0 += 8;
                    p1 += 8;
                }
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;
            const float* p1 = (const float*)B + (j + jj + 4) * B_hstep + k * 4;
            const float* p2 = (const float*)B + (j + jj + 8) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, _mm_load_ps(p0));
                _mm_store_ps(pp + 4, _mm_load_ps(p1));
                _mm_store_ps(pp + 8, _mm_load_ps(p2));
                pp += 12;
                p0 += 4;
                p1 += 4;
                p2 += 4;
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

            int kk = 0;
#if __AVX__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_loadu_ps(p0);
                __m256 _r1 = _mm256_loadu_ps(p1);
                __m256 _r2 = _mm256_loadu_ps(p2);
                __m256 _r3 = _mm256_loadu_ps(p3);
                __m256 _r4 = _mm256_loadu_ps(p4);
                __m256 _r5 = _mm256_loadu_ps(p5);
                __m256 _r6 = _mm256_loadu_ps(p6);
                __m256 _r7 = _mm256_loadu_ps(p7);
                __m256 _r8 = _mm256_loadu_ps(p8);
                __m256 _r9 = _mm256_loadu_ps(p9);
                __m256 _ra = _mm256_loadu_ps(pa);
                __m256 _rb = _mm256_loadu_ps(pb);
                transpose8x12_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                _mm256_storeu_ps(pp, _r0);
                _mm256_storeu_ps(pp + 8, _r1);
                _mm256_storeu_ps(pp + 8 * 2, _r2);
                _mm256_storeu_ps(pp + 8 * 3, _r3);
                _mm256_storeu_ps(pp + 8 * 4, _r4);
                _mm256_storeu_ps(pp + 8 * 5, _r5);
                _mm256_storeu_ps(pp + 8 * 6, _r6);
                _mm256_storeu_ps(pp + 8 * 7, _r7);
                _mm256_storeu_ps(pp + 8 * 8, _r8);
                _mm256_storeu_ps(pp + 8 * 9, _r9);
                _mm256_storeu_ps(pp + 8 * 10, _ra);
                _mm256_storeu_ps(pp + 8 * 11, _rb);
                pp += 96;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
                p4 += 8;
                p5 += 8;
                p6 += 8;
                p7 += 8;
                p8 += 8;
                p9 += 8;
                pa += 8;
                pb += 8;
            }
#endif // __AVX__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_loadu_ps(p0);
                __m128 _r1 = _mm_loadu_ps(p1);
                __m128 _r2 = _mm_loadu_ps(p2);
                __m128 _r3 = _mm_loadu_ps(p3);
                __m128 _r4 = _mm_loadu_ps(p4);
                __m128 _r5 = _mm_loadu_ps(p5);
                __m128 _r6 = _mm_loadu_ps(p6);
                __m128 _r7 = _mm_loadu_ps(p7);
                __m128 _r8 = _mm_loadu_ps(p8);
                __m128 _r9 = _mm_loadu_ps(p9);
                __m128 _ra = _mm_loadu_ps(pa);
                __m128 _rb = _mm_loadu_ps(pb);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);
                _mm_store_ps(pp, _r0);
                _mm_store_ps(pp + 4, _r4);
                _mm_store_ps(pp + 4 * 2, _r8);
                _mm_store_ps(pp + 4 * 3, _r1);
                _mm_store_ps(pp + 4 * 4, _r5);
                _mm_store_ps(pp + 4 * 5, _r9);
                _mm_store_ps(pp + 4 * 6, _r2);
                _mm_store_ps(pp + 4 * 7, _r6);
                _mm_store_ps(pp + 4 * 8, _ra);
                _mm_store_ps(pp + 4 * 9, _r3);
                _mm_store_ps(pp + 4 * 10, _r7);
                _mm_store_ps(pp + 4 * 11, _rb);
                pp += 48;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
                p4 += 4;
                p5 += 4;
                p6 += 4;
                p7 += 4;
                p8 += 4;
                p9 += 4;
                pa += 4;
                pb += 4;
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
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const float* p0 = (const float*)B + (j + jj) / 16 * 16 * B_hstep + k * 16;
            const float* p1 = (const float*)B + ((j + jj) / 16 * 16 + 16) * B_hstep + k * 16;

            if ((j + jj) % 16 == 0)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm256_storeu_ps(pp, _mm256_load_ps(p0));
                    pp += 8;
                    p0 += 16;
                }
            }
            if ((j + jj) % 16 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm256_storeu_ps(pp, _mm256_loadu_ps(p0 + 4));
                    pp += 8;
                    p0 += 16;
                }
            }
            if ((j + jj) % 16 == 8)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm256_storeu_ps(pp, _mm256_load_ps(p0 + 8));
                    pp += 8;
                    p0 += 16;
                }
            }
            if ((j + jj) % 16 == 12)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm_store_ps(pp, _mm_load_ps(p0 + 12));
                    _mm_store_ps(pp + 4, _mm_load_ps(p1));
                    pp += 8;
                    p0 += 16;
                    p1 += 16;
                }
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const float* p0 = (const float*)B + (j + jj) / 8 * 8 * B_hstep + k * 8;
            const float* p1 = (const float*)B + (j + jj + 8) / 8 * 8 * B_hstep + k * 8;

            if ((j + jj) % 8 == 0)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm256_storeu_ps(pp, _mm256_load_ps(p0));
                    pp += 8;
                    p0 += 8;
                }
            }
            if ((j + jj) % 8 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm_store_ps(pp, _mm_load_ps(p0 + 4));
                    _mm_store_ps(pp + 4, _mm_load_ps(p1));
                    pp += 8;
                    p0 += 8;
                    p1 += 8;
                }
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;
            const float* p1 = (const float*)B + (j + jj + 4) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, _mm_load_ps(p0));
                _mm_store_ps(pp + 4, _mm_load_ps(p1));
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
#if __AVX__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_loadu_ps(p0);
                __m256 _r1 = _mm256_loadu_ps(p1);
                __m256 _r2 = _mm256_loadu_ps(p2);
                __m256 _r3 = _mm256_loadu_ps(p3);
                __m256 _r4 = _mm256_loadu_ps(p4);
                __m256 _r5 = _mm256_loadu_ps(p5);
                __m256 _r6 = _mm256_loadu_ps(p6);
                __m256 _r7 = _mm256_loadu_ps(p7);
                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm256_storeu_ps(pp, _r0);
                _mm256_storeu_ps(pp + 8, _r1);
                _mm256_storeu_ps(pp + 8 * 2, _r2);
                _mm256_storeu_ps(pp + 8 * 3, _r3);
                _mm256_storeu_ps(pp + 8 * 4, _r4);
                _mm256_storeu_ps(pp + 8 * 5, _r5);
                _mm256_storeu_ps(pp + 8 * 6, _r6);
                _mm256_storeu_ps(pp + 8 * 7, _r7);
                pp += 64;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
                p4 += 8;
                p5 += 8;
                p6 += 8;
                p7 += 8;
            }
#endif // __AVX__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_loadu_ps(p0);
                __m128 _r1 = _mm_loadu_ps(p1);
                __m128 _r2 = _mm_loadu_ps(p2);
                __m128 _r3 = _mm_loadu_ps(p3);
                __m128 _r4 = _mm_loadu_ps(p4);
                __m128 _r5 = _mm_loadu_ps(p5);
                __m128 _r6 = _mm_loadu_ps(p6);
                __m128 _r7 = _mm_loadu_ps(p7);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                _mm_store_ps(pp, _r0);
                _mm_store_ps(pp + 4, _r4);
                _mm_store_ps(pp + 4 * 2, _r1);
                _mm_store_ps(pp + 4 * 3, _r5);
                _mm_store_ps(pp + 4 * 4, _r2);
                _mm_store_ps(pp + 4 * 5, _r6);
                _mm_store_ps(pp + 4 * 6, _r3);
                _mm_store_ps(pp + 4 * 7, _r7);
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
    for (; jj + 3 < max_jj; jj += 4)
    {
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const float* p0 = (const float*)B + (j + jj) / 16 * 16 * B_hstep + k * 16;

            if ((j + jj) % 16 == 0)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm_store_ps(pp, _mm_load_ps(p0));
                    pp += 4;
                    p0 += 16;
                }
            }
            if ((j + jj) % 16 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm_store_ps(pp, _mm_load_ps(p0 + 4));
                    pp += 4;
                    p0 += 16;
                }
            }
            if ((j + jj) % 16 == 8)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm_store_ps(pp, _mm_load_ps(p0 + 8));
                    pp += 4;
                    p0 += 16;
                }
            }
            if ((j + jj) % 16 == 12)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm_store_ps(pp, _mm_load_ps(p0 + 12));
                    pp += 4;
                    p0 += 16;
                }
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const float* p0 = (const float*)B + (j + jj) / 8 * 8 * B_hstep + k * 8;

            if ((j + jj) % 8 == 0)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm_store_ps(pp, _mm_load_ps(p0));
                    pp += 4;
                    p0 += 8;
                }
            }
            if ((j + jj) % 8 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm_store_ps(pp, _mm_load_ps(p0 + 4));
                    pp += 4;
                    p0 += 8;
                }
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, _mm_load_ps(p0));
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
            const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
            const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;

            int kk = 0;
#if __AVX__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_loadu_ps(p0);
                __m256 _r1 = _mm256_loadu_ps(p1);
                __m256 _r2 = _mm256_loadu_ps(p2);
                __m256 _r3 = _mm256_loadu_ps(p3);
                transpose8x4_ps(_r0, _r1, _r2, _r3);
                _mm256_storeu_ps(pp, _r0);
                _mm256_storeu_ps(pp + 8, _r1);
                _mm256_storeu_ps(pp + 16, _r2);
                _mm256_storeu_ps(pp + 24, _r3);
                pp += 32;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
            }
#endif // __AVX__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_loadu_ps(p0);
                __m128 _r1 = _mm_loadu_ps(p1);
                __m128 _r2 = _mm_loadu_ps(p2);
                __m128 _r3 = _mm_loadu_ps(p3);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_store_ps(pp, _r0);
                _mm_store_ps(pp + 4, _r1);
                _mm_store_ps(pp + 8, _r2);
                _mm_store_ps(pp + 12, _r3);
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
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

            int kk = 0;
#if __SSE2__
#if __AVX__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_loadu_ps(p0);
                __m256 _r1 = _mm256_loadu_ps(p1);
                transpose8x2_ps(_r0, _r1);
                _mm256_storeu_ps(pp, _r0);
                _mm256_storeu_ps(pp + 8, _r1);
                pp += 16;
                p0 += 8;
                p1 += 8;
            }
#endif // __AVX__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_loadu_ps(p0);
                __m128 _r1 = _mm_loadu_ps(p1);
                __m128 _tmp0 = _mm_unpacklo_ps(_r0, _r1);
                __m128 _tmp1 = _mm_unpackhi_ps(_r0, _r1);
                _mm_store_ps(pp, _tmp0);
                _mm_store_ps(pp + 4, _tmp1);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk++)
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
#if __SSE2__
#if __AVX__
            for (; kk + 7 < max_kk; kk += 8)
            {
                _mm256_storeu_ps(pp, _mm256_loadu_ps(p0));
                pp += 8;
                p0 += 8;
            }
#endif // __AVX__
            for (; kk + 3 < max_kk; kk += 4)
            {
                _mm_storeu_ps(pp, _mm_loadu_ps(p0));
                pp += 4;
                p0 += 4;
            }
#endif // __SSE2__
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
#if __SSE2__
    for (; jj + 11 < max_jj; jj += 12)
    {
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_load_ps(p0);
                __m512 _r1 = _mm512_load_ps(p0 + 16 * 1);
                __m512 _r2 = _mm512_load_ps(p0 + 16 * 2);
                __m512 _r3 = _mm512_load_ps(p0 + 16 * 3);
                __m512 _r4 = _mm512_load_ps(p0 + 16 * 4);
                __m512 _r5 = _mm512_load_ps(p0 + 16 * 5);
                __m512 _r6 = _mm512_load_ps(p0 + 16 * 6);
                __m512 _r7 = _mm512_load_ps(p0 + 16 * 7);
                __m512 _r8 = _mm512_load_ps(p0 + 16 * 8);
                __m512 _r9 = _mm512_load_ps(p0 + 16 * 9);
                __m512 _ra = _mm512_load_ps(p0 + 16 * 10);
                __m512 _rb = _mm512_load_ps(p0 + 16 * 11);
                transpose16x12_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                _mm512_store_ps(pp, _r0);
                _mm512_store_ps(pp + 16 * 1, _r1);
                _mm512_store_ps(pp + 16 * 2, _r2);
                _mm512_store_ps(pp + 16 * 3, _r3);
                _mm512_store_ps(pp + 16 * 4, _r4);
                _mm512_store_ps(pp + 16 * 5, _r5);
                _mm512_store_ps(pp + 16 * 6, _r6);
                _mm512_store_ps(pp + 16 * 7, _r7);
                _mm512_store_ps(pp + 16 * 8, _r8);
                _mm512_store_ps(pp + 16 * 9, _r9);
                _mm512_store_ps(pp + 16 * 10, _ra);
                _mm512_store_ps(pp + 16 * 11, _rb);
                pp += 192;
                p0 += B_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_load_ps(p0);
                __m256 _r1 = _mm256_load_ps(p0 + 8 * 1);
                __m256 _r2 = _mm256_load_ps(p0 + 8 * 2);
                __m256 _r3 = _mm256_load_ps(p0 + 8 * 3);
                __m256 _r4 = _mm256_load_ps(p0 + 8 * 4);
                __m256 _r5 = _mm256_load_ps(p0 + 8 * 5);
                __m256 _r6 = _mm256_load_ps(p0 + 8 * 6);
                __m256 _r7 = _mm256_load_ps(p0 + 8 * 7);
                __m256 _r8 = _mm256_load_ps(p0 + 8 * 8);
                __m256 _r9 = _mm256_load_ps(p0 + 8 * 9);
                __m256 _ra = _mm256_load_ps(p0 + 8 * 10);
                __m256 _rb = _mm256_load_ps(p0 + 8 * 11);
                transpose8x12_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                _mm256_store_ps(pp, _r0);
                _mm256_store_ps(pp + 8 * 1, _r1);
                _mm256_store_ps(pp + 8 * 2, _r2);
                _mm256_store_ps(pp + 8 * 3, _r3);
                _mm256_store_ps(pp + 8 * 4, _r4);
                _mm256_store_ps(pp + 8 * 5, _r5);
                _mm256_store_ps(pp + 8 * 6, _r6);
                _mm256_store_ps(pp + 8 * 7, _r7);
                _mm256_store_ps(pp + 8 * 8, _r8);
                _mm256_store_ps(pp + 8 * 9, _r9);
                _mm256_store_ps(pp + 8 * 10, _ra);
                _mm256_store_ps(pp + 8 * 11, _rb);
                pp += 96;
                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_load_ps(p0);
                __m128 _r1 = _mm_load_ps(p0 + 4 * 1);
                __m128 _r2 = _mm_load_ps(p0 + 4 * 2);
                __m128 _r3 = _mm_load_ps(p0 + 4 * 3);
                __m128 _r4 = _mm_load_ps(p0 + 4 * 4);
                __m128 _r5 = _mm_load_ps(p0 + 4 * 5);
                __m128 _r6 = _mm_load_ps(p0 + 4 * 6);
                __m128 _r7 = _mm_load_ps(p0 + 4 * 7);
                __m128 _r8 = _mm_load_ps(p0 + 4 * 8);
                __m128 _r9 = _mm_load_ps(p0 + 4 * 9);
                __m128 _ra = _mm_load_ps(p0 + 4 * 10);
                __m128 _rb = _mm_load_ps(p0 + 4 * 11);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);
                _mm_store_ps(pp, _r0);
                _mm_store_ps(pp + 4 * 1, _r4);
                _mm_store_ps(pp + 4 * 2, _r8);
                _mm_store_ps(pp + 4 * 3, _r1);
                _mm_store_ps(pp + 4 * 4, _r5);
                _mm_store_ps(pp + 4 * 5, _r9);
                _mm_store_ps(pp + 4 * 6, _r2);
                _mm_store_ps(pp + 4 * 7, _r6);
                _mm_store_ps(pp + 4 * 8, _ra);
                _mm_store_ps(pp + 4 * 9, _r3);
                _mm_store_ps(pp + 4 * 10, _r7);
                _mm_store_ps(pp + 4 * 11, _rb);
                pp += 48;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, _mm_loadu_ps(p0));
                _mm_store_ps(pp + 4, _mm_loadu_ps(p0 + 4));
                _mm_store_ps(pp + 8, _mm_loadu_ps(p0 + 8));
                pp += 12;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_load_ps(p0);
                __m512 _r1 = _mm512_load_ps(p0 + 16 * 1);
                __m512 _r2 = _mm512_load_ps(p0 + 16 * 2);
                __m512 _r3 = _mm512_load_ps(p0 + 16 * 3);
                __m512 _r4 = _mm512_load_ps(p0 + 16 * 4);
                __m512 _r5 = _mm512_load_ps(p0 + 16 * 5);
                __m512 _r6 = _mm512_load_ps(p0 + 16 * 6);
                __m512 _r7 = _mm512_load_ps(p0 + 16 * 7);
                transpose16x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm512_store_ps(pp, _r0);
                _mm512_store_ps(pp + 16 * 1, _r1);
                _mm512_store_ps(pp + 16 * 2, _r2);
                _mm512_store_ps(pp + 16 * 3, _r3);
                _mm512_store_ps(pp + 16 * 4, _r4);
                _mm512_store_ps(pp + 16 * 5, _r5);
                _mm512_store_ps(pp + 16 * 6, _r6);
                _mm512_store_ps(pp + 16 * 7, _r7);
                pp += 128;
                p0 += B_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_load_ps(p0);
                __m256 _r1 = _mm256_load_ps(p0 + 8 * 1);
                __m256 _r2 = _mm256_load_ps(p0 + 8 * 2);
                __m256 _r3 = _mm256_load_ps(p0 + 8 * 3);
                __m256 _r4 = _mm256_load_ps(p0 + 8 * 4);
                __m256 _r5 = _mm256_load_ps(p0 + 8 * 5);
                __m256 _r6 = _mm256_load_ps(p0 + 8 * 6);
                __m256 _r7 = _mm256_load_ps(p0 + 8 * 7);
                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm256_store_ps(pp, _r0);
                _mm256_store_ps(pp + 8 * 1, _r1);
                _mm256_store_ps(pp + 8 * 2, _r2);
                _mm256_store_ps(pp + 8 * 3, _r3);
                _mm256_store_ps(pp + 8 * 4, _r4);
                _mm256_store_ps(pp + 8 * 5, _r5);
                _mm256_store_ps(pp + 8 * 6, _r6);
                _mm256_store_ps(pp + 8 * 7, _r7);
                pp += 64;
                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_load_ps(p0);
                __m128 _r1 = _mm_load_ps(p0 + 4 * 1);
                __m128 _r2 = _mm_load_ps(p0 + 4 * 2);
                __m128 _r3 = _mm_load_ps(p0 + 4 * 3);
                __m128 _r4 = _mm_load_ps(p0 + 4 * 4);
                __m128 _r5 = _mm_load_ps(p0 + 4 * 5);
                __m128 _r6 = _mm_load_ps(p0 + 4 * 6);
                __m128 _r7 = _mm_load_ps(p0 + 4 * 7);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                _mm_store_ps(pp, _r0);
                _mm_store_ps(pp + 4 * 1, _r4);
                _mm_store_ps(pp + 4 * 2, _r1);
                _mm_store_ps(pp + 4 * 3, _r5);
                _mm_store_ps(pp + 4 * 4, _r2);
                _mm_store_ps(pp + 4 * 5, _r6);
                _mm_store_ps(pp + 4 * 6, _r3);
                _mm_store_ps(pp + 4 * 7, _r7);
                pp += 32;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, _mm_loadu_ps(p0));
                _mm_store_ps(pp + 4, _mm_loadu_ps(p0 + 4));
                pp += 8;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_load_ps(p0);
                __m512 _r1 = _mm512_load_ps(p0 + 16 * 1);
                __m512 _r2 = _mm512_load_ps(p0 + 16 * 2);
                __m512 _r3 = _mm512_load_ps(p0 + 16 * 3);
                transpose16x4_ps(_r0, _r1, _r2, _r3);
                _mm512_store_ps(pp, _r0);
                _mm512_store_ps(pp + 16 * 1, _r1);
                _mm512_store_ps(pp + 16 * 2, _r2);
                _mm512_store_ps(pp + 16 * 3, _r3);
                pp += 64;
                p0 += B_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_load_ps(p0);
                __m256 _r1 = _mm256_load_ps(p0 + 8 * 1);
                __m256 _r2 = _mm256_load_ps(p0 + 8 * 2);
                __m256 _r3 = _mm256_load_ps(p0 + 8 * 3);
                transpose8x4_ps(_r0, _r1, _r2, _r3);
                _mm256_store_ps(pp, _r0);
                _mm256_store_ps(pp + 8 * 1, _r1);
                _mm256_store_ps(pp + 8 * 2, _r2);
                _mm256_store_ps(pp + 8 * 3, _r3);
                pp += 32;
                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_load_ps(p0);
                __m128 _r1 = _mm_load_ps(p0 + 4 * 1);
                __m128 _r2 = _mm_load_ps(p0 + 4 * 2);
                __m128 _r3 = _mm_load_ps(p0 + 4 * 3);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_store_ps(pp, _r0);
                _mm_store_ps(pp + 4 * 1, _r1);
                _mm_store_ps(pp + 4 * 2, _r2);
                _mm_store_ps(pp + 4 * 3, _r3);
                pp += 16;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, _mm_loadu_ps(p0));
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = _mm512_load_ps(p0);
                __m512 _r1 = _mm512_load_ps(p0 + 16);
                transpose16x2_ps(_r0, _r1);
                _mm512_store_ps(pp, _r0);
                _mm512_store_ps(pp + 16, _r1);
                pp += 32;
                p0 += B_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = _mm256_load_ps(p0);
                __m256 _r1 = _mm256_load_ps(p0 + 8);
                transpose8x2_ps(_r0, _r1);
                _mm256_store_ps(pp, _r0);
                _mm256_store_ps(pp + 8, _r1);
                pp += 16;
                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = _mm_load_ps(p0);
                __m128 _r1 = _mm_load_ps(p0 + 4);
                __m128 _tmp0 = _mm_unpacklo_ps(_r0, _r1);
                __m128 _tmp1 = _mm_unpackhi_ps(_r0, _r1);
                _mm_store_ps(pp, _tmp0);
                _mm_store_ps(pp + 4, _tmp1);
                pp += 8;
                p0 += B_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
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
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                _mm512_store_ps(pp, _mm512_load_ps(p0));
                pp += 16;
                p0 += B_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                _mm256_store_ps(pp, _mm256_load_ps(p0));
                pp += 8;
                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                _mm_store_ps(pp, _mm_load_ps(p0));
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
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
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        if (out_elempack == 16)
        {
            float* p0 = (float*)top_blob + (j / 16 * 16) * out_hstep + (i + ii) * 16;

            int jj = 0;
            if (j % 16 == 4)
            {
                __m512 _r0 = _mm512_load_ps(pp);
                __m512 _r1 = _mm512_load_ps(pp + 16);
                __m512 _r2 = _mm512_load_ps(pp + 16 * 2);
                __m512 _r3 = _mm512_load_ps(pp + 16 * 3);
                if (max_jj > 4)
                {
                    // assert max_jj > 8
                    __m512 _r4 = _mm512_load_ps(pp + 16 * 4);
                    __m512 _r5 = _mm512_load_ps(pp + 16 * 5);
                    __m512 _r6 = _mm512_load_ps(pp + 16 * 6);
                    __m512 _r7 = _mm512_load_ps(pp + 16 * 7);
                    __m512 _r8 = _mm512_load_ps(pp + 16 * 8);
                    __m512 _r9 = _mm512_load_ps(pp + 16 * 9);
                    __m512 _ra = _mm512_load_ps(pp + 16 * 10);
                    __m512 _rb = _mm512_load_ps(pp + 16 * 11);
                    transpose16x12_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                    _mm256_storeu_ps(p0 + 4, _mm512_extractf32x8_ps(_r0, 0));
                    _mm_store_ps(p0 + 4 + 8, _mm512_extractf32x4_ps(_r0, 2));
                    _mm_store_ps(p0 + 16 + 4, _mm512_extractf32x4_ps(_r0, 3));
                    _mm256_store_ps(p0 + 16 + 4 + 4, _mm512_extractf32x8_ps(_r1, 0));
                    _mm256_storeu_ps(p0 + 16 * 2 + 4, _mm512_extractf32x8_ps(_r1, 1));
                    _mm_store_ps(p0 + 16 * 2 + 4 + 8, _mm512_extractf32x4_ps(_r2, 0));
                    _mm_store_ps(p0 + 16 * 3 + 4, _mm512_extractf32x4_ps(_r2, 1));
                    _mm256_store_ps(p0 + 16 * 3 + 4 + 4, _mm512_extractf32x8_ps(_r2, 1));
                    _mm256_storeu_ps(p0 + 16 * 4 + 4, _mm512_extractf32x8_ps(_r3, 0));
                    _mm_store_ps(p0 + 16 * 4 + 4 + 8, _mm512_extractf32x4_ps(_r3, 2));
                    _mm_store_ps(p0 + 16 * 5 + 4, _mm512_extractf32x4_ps(_r3, 3));
                    _mm256_store_ps(p0 + 16 * 5 + 4 + 4, _mm512_extractf32x8_ps(_r4, 0));
                    _mm256_storeu_ps(p0 + 16 * 6 + 4, _mm512_extractf32x8_ps(_r4, 1));
                    _mm_store_ps(p0 + 16 * 6 + 4 + 8, _mm512_extractf32x4_ps(_r5, 0));
                    _mm_store_ps(p0 + 16 * 7 + 4, _mm512_extractf32x4_ps(_r5, 1));
                    _mm256_store_ps(p0 + 16 * 7 + 4 + 4, _mm512_extractf32x8_ps(_r5, 1));
                    _mm256_storeu_ps(p0 + 16 * 8 + 4, _mm512_extractf32x8_ps(_r6, 0));
                    _mm_store_ps(p0 + 16 * 8 + 4 + 8, _mm512_extractf32x4_ps(_r6, 2));
                    _mm_store_ps(p0 + 16 * 9 + 4, _mm512_extractf32x4_ps(_r6, 3));
                    _mm256_store_ps(p0 + 16 * 9 + 4 + 4, _mm512_extractf32x8_ps(_r7, 0));
                    _mm256_storeu_ps(p0 + 16 * 10 + 4, _mm512_extractf32x8_ps(_r7, 1));
                    _mm_store_ps(p0 + 16 * 10 + 4 + 8, _mm512_extractf32x4_ps(_r8, 0));
                    _mm_store_ps(p0 + 16 * 11 + 4, _mm512_extractf32x4_ps(_r8, 1));
                    _mm256_store_ps(p0 + 16 * 11 + 4 + 4, _mm512_extractf32x8_ps(_r8, 1));
                    _mm256_storeu_ps(p0 + 16 * 12 + 4, _mm512_extractf32x8_ps(_r9, 0));
                    _mm_store_ps(p0 + 16 * 12 + 4 + 8, _mm512_extractf32x4_ps(_r9, 2));
                    _mm_store_ps(p0 + 16 * 13 + 4, _mm512_extractf32x4_ps(_r9, 3));
                    _mm256_store_ps(p0 + 16 * 13 + 4 + 4, _mm512_extractf32x8_ps(_ra, 0));
                    _mm256_storeu_ps(p0 + 16 * 14 + 4, _mm512_extractf32x8_ps(_ra, 1));
                    _mm_store_ps(p0 + 16 * 14 + 4 + 8, _mm512_extractf32x4_ps(_rb, 0));
                    _mm_store_ps(p0 + 16 * 15 + 4, _mm512_extractf32x4_ps(_rb, 1));
                    _mm256_store_ps(p0 + 16 * 15 + 4 + 4, _mm512_extractf32x8_ps(_rb, 1));
                    pp += 192;
                    jj += 12;
                }
                else
                {
                    transpose16x4_ps(_r0, _r1, _r2, _r3);
                    _mm_store_ps(p0 + 4, _mm512_extractf32x4_ps(_r0, 0));
                    _mm_store_ps(p0 + 16 + 4, _mm512_extractf32x4_ps(_r0, 1));
                    _mm_store_ps(p0 + 16 * 2 + 4, _mm512_extractf32x4_ps(_r0, 2));
                    _mm_store_ps(p0 + 16 * 3 + 4, _mm512_extractf32x4_ps(_r0, 3));
                    _mm_store_ps(p0 + 16 * 4 + 4, _mm512_extractf32x4_ps(_r1, 0));
                    _mm_store_ps(p0 + 16 * 5 + 4, _mm512_extractf32x4_ps(_r1, 1));
                    _mm_store_ps(p0 + 16 * 6 + 4, _mm512_extractf32x4_ps(_r1, 2));
                    _mm_store_ps(p0 + 16 * 7 + 4, _mm512_extractf32x4_ps(_r1, 3));
                    _mm_store_ps(p0 + 16 * 8 + 4, _mm512_extractf32x4_ps(_r2, 0));
                    _mm_store_ps(p0 + 16 * 9 + 4, _mm512_extractf32x4_ps(_r2, 1));
                    _mm_store_ps(p0 + 16 * 10 + 4, _mm512_extractf32x4_ps(_r2, 2));
                    _mm_store_ps(p0 + 16 * 11 + 4, _mm512_extractf32x4_ps(_r2, 3));
                    _mm_store_ps(p0 + 16 * 12 + 4, _mm512_extractf32x4_ps(_r3, 0));
                    _mm_store_ps(p0 + 16 * 13 + 4, _mm512_extractf32x4_ps(_r3, 1));
                    _mm_store_ps(p0 + 16 * 14 + 4, _mm512_extractf32x4_ps(_r3, 2));
                    _mm_store_ps(p0 + 16 * 15 + 4, _mm512_extractf32x4_ps(_r3, 3));
                    pp += 64;
                    jj += 4;
                }
                p0 += out_hstep * 16;
            }
            if (j % 16 == 8)
            {
                __m512 _r0 = _mm512_load_ps(pp);
                __m512 _r1 = _mm512_load_ps(pp + 16);
                __m512 _r2 = _mm512_load_ps(pp + 16 * 2);
                __m512 _r3 = _mm512_load_ps(pp + 16 * 3);
                if (max_jj > 4)
                {
                    __m512 _r4 = _mm512_load_ps(pp + 16 * 4);
                    __m512 _r5 = _mm512_load_ps(pp + 16 * 5);
                    __m512 _r6 = _mm512_load_ps(pp + 16 * 6);
                    __m512 _r7 = _mm512_load_ps(pp + 16 * 7);
                    transpose16x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                    _mm256_store_ps(p0 + 8, _mm512_extractf32x8_ps(_r0, 0));
                    _mm256_store_ps(p0 + 16 + 8, _mm512_extractf32x8_ps(_r0, 1));
                    _mm256_store_ps(p0 + 16 * 2 + 8, _mm512_extractf32x8_ps(_r1, 0));
                    _mm256_store_ps(p0 + 16 * 3 + 8, _mm512_extractf32x8_ps(_r1, 1));
                    _mm256_store_ps(p0 + 16 * 4 + 8, _mm512_extractf32x8_ps(_r2, 0));
                    _mm256_store_ps(p0 + 16 * 5 + 8, _mm512_extractf32x8_ps(_r2, 1));
                    _mm256_store_ps(p0 + 16 * 6 + 8, _mm512_extractf32x8_ps(_r3, 0));
                    _mm256_store_ps(p0 + 16 * 7 + 8, _mm512_extractf32x8_ps(_r3, 1));
                    _mm256_store_ps(p0 + 16 * 8 + 8, _mm512_extractf32x8_ps(_r4, 0));
                    _mm256_store_ps(p0 + 16 * 9 + 8, _mm512_extractf32x8_ps(_r4, 1));
                    _mm256_store_ps(p0 + 16 * 10 + 8, _mm512_extractf32x8_ps(_r5, 0));
                    _mm256_store_ps(p0 + 16 * 11 + 8, _mm512_extractf32x8_ps(_r5, 1));
                    _mm256_store_ps(p0 + 16 * 12 + 8, _mm512_extractf32x8_ps(_r6, 0));
                    _mm256_store_ps(p0 + 16 * 13 + 8, _mm512_extractf32x8_ps(_r6, 1));
                    _mm256_store_ps(p0 + 16 * 14 + 8, _mm512_extractf32x8_ps(_r7, 0));
                    _mm256_store_ps(p0 + 16 * 15 + 8, _mm512_extractf32x8_ps(_r7, 1));
                    pp += 128;
                    jj += 8;
                }
                else
                {
                    transpose16x4_ps(_r0, _r1, _r2, _r3);
                    _mm_store_ps(p0 + 8, _mm512_extractf32x4_ps(_r0, 0));
                    _mm_store_ps(p0 + 16 + 8, _mm512_extractf32x4_ps(_r0, 1));
                    _mm_store_ps(p0 + 16 * 2 + 8, _mm512_extractf32x4_ps(_r0, 2));
                    _mm_store_ps(p0 + 16 * 3 + 8, _mm512_extractf32x4_ps(_r0, 3));
                    _mm_store_ps(p0 + 16 * 4 + 8, _mm512_extractf32x4_ps(_r1, 0));
                    _mm_store_ps(p0 + 16 * 5 + 8, _mm512_extractf32x4_ps(_r1, 1));
                    _mm_store_ps(p0 + 16 * 6 + 8, _mm512_extractf32x4_ps(_r1, 2));
                    _mm_store_ps(p0 + 16 * 7 + 8, _mm512_extractf32x4_ps(_r1, 3));
                    _mm_store_ps(p0 + 16 * 8 + 8, _mm512_extractf32x4_ps(_r2, 0));
                    _mm_store_ps(p0 + 16 * 9 + 8, _mm512_extractf32x4_ps(_r2, 1));
                    _mm_store_ps(p0 + 16 * 10 + 8, _mm512_extractf32x4_ps(_r2, 2));
                    _mm_store_ps(p0 + 16 * 11 + 8, _mm512_extractf32x4_ps(_r2, 3));
                    _mm_store_ps(p0 + 16 * 12 + 8, _mm512_extractf32x4_ps(_r3, 0));
                    _mm_store_ps(p0 + 16 * 13 + 8, _mm512_extractf32x4_ps(_r3, 1));
                    _mm_store_ps(p0 + 16 * 14 + 8, _mm512_extractf32x4_ps(_r3, 2));
                    _mm_store_ps(p0 + 16 * 15 + 8, _mm512_extractf32x4_ps(_r3, 3));
                    pp += 64;
                    jj += 4;
                }
                p0 += out_hstep * 16;
            }
            if (j % 16 == 12)
            {
                __m512 _r0 = _mm512_load_ps(pp);
                __m512 _r1 = _mm512_load_ps(pp + 16);
                __m512 _r2 = _mm512_load_ps(pp + 16 * 2);
                __m512 _r3 = _mm512_load_ps(pp + 16 * 3);
                transpose16x4_ps(_r0, _r1, _r2, _r3);
                _mm_store_ps(p0 + 12, _mm512_extractf32x4_ps(_r0, 0));
                _mm_store_ps(p0 + 16 + 12, _mm512_extractf32x4_ps(_r0, 1));
                _mm_store_ps(p0 + 16 * 2 + 12, _mm512_extractf32x4_ps(_r0, 2));
                _mm_store_ps(p0 + 16 * 3 + 12, _mm512_extractf32x4_ps(_r0, 3));
                _mm_store_ps(p0 + 16 * 4 + 12, _mm512_extractf32x4_ps(_r1, 0));
                _mm_store_ps(p0 + 16 * 5 + 12, _mm512_extractf32x4_ps(_r1, 1));
                _mm_store_ps(p0 + 16 * 6 + 12, _mm512_extractf32x4_ps(_r1, 2));
                _mm_store_ps(p0 + 16 * 7 + 12, _mm512_extractf32x4_ps(_r1, 3));
                _mm_store_ps(p0 + 16 * 8 + 12, _mm512_extractf32x4_ps(_r2, 0));
                _mm_store_ps(p0 + 16 * 9 + 12, _mm512_extractf32x4_ps(_r2, 1));
                _mm_store_ps(p0 + 16 * 10 + 12, _mm512_extractf32x4_ps(_r2, 2));
                _mm_store_ps(p0 + 16 * 11 + 12, _mm512_extractf32x4_ps(_r2, 3));
                _mm_store_ps(p0 + 16 * 12 + 12, _mm512_extractf32x4_ps(_r3, 0));
                _mm_store_ps(p0 + 16 * 13 + 12, _mm512_extractf32x4_ps(_r3, 1));
                _mm_store_ps(p0 + 16 * 14 + 12, _mm512_extractf32x4_ps(_r3, 2));
                _mm_store_ps(p0 + 16 * 15 + 12, _mm512_extractf32x4_ps(_r3, 3));
                pp += 64;
                p0 += out_hstep * 16;
                jj += 4;
            }
            for (; jj + 15 < max_jj; jj += 16)
            {
                __m512 _r0 = _mm512_load_ps(pp);
                __m512 _r1 = _mm512_load_ps(pp + 16);
                __m512 _r2 = _mm512_load_ps(pp + 16 * 2);
                __m512 _r3 = _mm512_load_ps(pp + 16 * 3);
                __m512 _r4 = _mm512_load_ps(pp + 16 * 4);
                __m512 _r5 = _mm512_load_ps(pp + 16 * 5);
                __m512 _r6 = _mm512_load_ps(pp + 16 * 6);
                __m512 _r7 = _mm512_load_ps(pp + 16 * 7);
                __m512 _r8 = _mm512_load_ps(pp + 16 * 8);
                __m512 _r9 = _mm512_load_ps(pp + 16 * 9);
                __m512 _ra = _mm512_load_ps(pp + 16 * 10);
                __m512 _rb = _mm512_load_ps(pp + 16 * 11);
                __m512 _rc = _mm512_load_ps(pp + 16 * 12);
                __m512 _rd = _mm512_load_ps(pp + 16 * 13);
                __m512 _re = _mm512_load_ps(pp + 16 * 14);
                __m512 _rf = _mm512_load_ps(pp + 16 * 15);
                transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                _mm512_store_ps(p0, _r0);
                _mm512_store_ps(p0 + 16, _r1);
                _mm512_store_ps(p0 + 16 * 2, _r2);
                _mm512_store_ps(p0 + 16 * 3, _r3);
                _mm512_store_ps(p0 + 16 * 4, _r4);
                _mm512_store_ps(p0 + 16 * 5, _r5);
                _mm512_store_ps(p0 + 16 * 6, _r6);
                _mm512_store_ps(p0 + 16 * 7, _r7);
                _mm512_store_ps(p0 + 16 * 8, _r8);
                _mm512_store_ps(p0 + 16 * 9, _r9);
                _mm512_store_ps(p0 + 16 * 10, _ra);
                _mm512_store_ps(p0 + 16 * 11, _rb);
                _mm512_store_ps(p0 + 16 * 12, _rc);
                _mm512_store_ps(p0 + 16 * 13, _rd);
                _mm512_store_ps(p0 + 16 * 14, _re);
                _mm512_store_ps(p0 + 16 * 15, _rf);
                pp += 256;
                p0 += out_hstep * 16;
            }
            for (; jj + 11 < max_jj; jj += 12)
            {
                __m512 _r0 = _mm512_load_ps(pp);
                __m512 _r1 = _mm512_load_ps(pp + 16);
                __m512 _r2 = _mm512_load_ps(pp + 16 * 2);
                __m512 _r3 = _mm512_load_ps(pp + 16 * 3);
                __m512 _r4 = _mm512_load_ps(pp + 16 * 4);
                __m512 _r5 = _mm512_load_ps(pp + 16 * 5);
                __m512 _r6 = _mm512_load_ps(pp + 16 * 6);
                __m512 _r7 = _mm512_load_ps(pp + 16 * 7);
                __m512 _r8 = _mm512_load_ps(pp + 16 * 8);
                __m512 _r9 = _mm512_load_ps(pp + 16 * 9);
                __m512 _ra = _mm512_load_ps(pp + 16 * 10);
                __m512 _rb = _mm512_load_ps(pp + 16 * 11);
                transpose16x12_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                _mm256_store_ps(p0, _mm512_extractf32x8_ps(_r0, 0));
                _mm_store_ps(p0 + 8, _mm512_extractf32x4_ps(_r0, 2));
                _mm_store_ps(p0 + 16, _mm512_extractf32x4_ps(_r0, 3));
                _mm256_storeu_ps(p0 + 16 + 4, _mm512_extractf32x8_ps(_r1, 0));
                _mm256_store_ps(p0 + 16 * 2, _mm512_extractf32x8_ps(_r1, 1));
                _mm_store_ps(p0 + 16 * 2 + 8, _mm512_extractf32x4_ps(_r2, 0));
                _mm_store_ps(p0 + 16 * 3, _mm512_extractf32x4_ps(_r2, 1));
                _mm256_storeu_ps(p0 + 16 * 3 + 4, _mm512_extractf32x8_ps(_r2, 1));
                _mm256_store_ps(p0 + 16 * 4, _mm512_extractf32x8_ps(_r3, 0));
                _mm_store_ps(p0 + 16 * 4 + 8, _mm512_extractf32x4_ps(_r3, 2));
                _mm_store_ps(p0 + 16 * 5, _mm512_extractf32x4_ps(_r3, 3));
                _mm256_storeu_ps(p0 + 16 * 5 + 4, _mm512_extractf32x8_ps(_r4, 0));
                _mm256_store_ps(p0 + 16 * 6, _mm512_extractf32x8_ps(_r4, 1));
                _mm_store_ps(p0 + 16 * 6 + 8, _mm512_extractf32x4_ps(_r5, 0));
                _mm_store_ps(p0 + 16 * 7, _mm512_extractf32x4_ps(_r5, 1));
                _mm256_storeu_ps(p0 + 16 * 7 + 4, _mm512_extractf32x8_ps(_r5, 1));
                _mm256_store_ps(p0 + 16 * 8, _mm512_extractf32x8_ps(_r6, 0));
                _mm_store_ps(p0 + 16 * 8 + 8, _mm512_extractf32x4_ps(_r6, 2));
                _mm_store_ps(p0 + 16 * 9, _mm512_extractf32x4_ps(_r6, 3));
                _mm256_storeu_ps(p0 + 16 * 9 + 4, _mm512_extractf32x8_ps(_r7, 0));
                _mm256_store_ps(p0 + 16 * 10, _mm512_extractf32x8_ps(_r7, 1));
                _mm_store_ps(p0 + 16 * 10 + 8, _mm512_extractf32x4_ps(_r8, 0));
                _mm_store_ps(p0 + 16 * 11, _mm512_extractf32x4_ps(_r8, 1));
                _mm256_storeu_ps(p0 + 16 * 11 + 4, _mm512_extractf32x8_ps(_r8, 1));
                _mm256_store_ps(p0 + 16 * 12, _mm512_extractf32x8_ps(_r9, 0));
                _mm_store_ps(p0 + 16 * 12 + 8, _mm512_extractf32x4_ps(_r9, 2));
                _mm_store_ps(p0 + 16 * 13, _mm512_extractf32x4_ps(_r9, 3));
                _mm256_storeu_ps(p0 + 16 * 13 + 4, _mm512_extractf32x8_ps(_ra, 0));
                _mm256_store_ps(p0 + 16 * 14, _mm512_extractf32x8_ps(_ra, 1));
                _mm_store_ps(p0 + 16 * 14 + 8, _mm512_extractf32x4_ps(_rb, 0));
                _mm_store_ps(p0 + 16 * 15, _mm512_extractf32x4_ps(_rb, 1));
                _mm256_storeu_ps(p0 + 16 * 15 + 4, _mm512_extractf32x8_ps(_rb, 1));
                pp += 192;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                __m512 _r0 = _mm512_load_ps(pp);
                __m512 _r1 = _mm512_load_ps(pp + 16);
                __m512 _r2 = _mm512_load_ps(pp + 16 * 2);
                __m512 _r3 = _mm512_load_ps(pp + 16 * 3);
                __m512 _r4 = _mm512_load_ps(pp + 16 * 4);
                __m512 _r5 = _mm512_load_ps(pp + 16 * 5);
                __m512 _r6 = _mm512_load_ps(pp + 16 * 6);
                __m512 _r7 = _mm512_load_ps(pp + 16 * 7);
                transpose16x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm256_store_ps(p0, _mm512_extractf32x8_ps(_r0, 0));
                _mm256_store_ps(p0 + 16, _mm512_extractf32x8_ps(_r0, 1));
                _mm256_store_ps(p0 + 16 * 2, _mm512_extractf32x8_ps(_r1, 0));
                _mm256_store_ps(p0 + 16 * 3, _mm512_extractf32x8_ps(_r1, 1));
                _mm256_store_ps(p0 + 16 * 4, _mm512_extractf32x8_ps(_r2, 0));
                _mm256_store_ps(p0 + 16 * 5, _mm512_extractf32x8_ps(_r2, 1));
                _mm256_store_ps(p0 + 16 * 6, _mm512_extractf32x8_ps(_r3, 0));
                _mm256_store_ps(p0 + 16 * 7, _mm512_extractf32x8_ps(_r3, 1));
                _mm256_store_ps(p0 + 16 * 8, _mm512_extractf32x8_ps(_r4, 0));
                _mm256_store_ps(p0 + 16 * 9, _mm512_extractf32x8_ps(_r4, 1));
                _mm256_store_ps(p0 + 16 * 10, _mm512_extractf32x8_ps(_r5, 0));
                _mm256_store_ps(p0 + 16 * 11, _mm512_extractf32x8_ps(_r5, 1));
                _mm256_store_ps(p0 + 16 * 12, _mm512_extractf32x8_ps(_r6, 0));
                _mm256_store_ps(p0 + 16 * 13, _mm512_extractf32x8_ps(_r6, 1));
                _mm256_store_ps(p0 + 16 * 14, _mm512_extractf32x8_ps(_r7, 0));
                _mm256_store_ps(p0 + 16 * 15, _mm512_extractf32x8_ps(_r7, 1));
                pp += 128;
                p0 += out_hstep * 16;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                __m512 _r0 = _mm512_load_ps(pp);
                __m512 _r1 = _mm512_load_ps(pp + 16);
                __m512 _r2 = _mm512_load_ps(pp + 16 * 2);
                __m512 _r3 = _mm512_load_ps(pp + 16 * 3);
                transpose16x4_ps(_r0, _r1, _r2, _r3);
                _mm_store_ps(p0, _mm512_extractf32x4_ps(_r0, 0));
                _mm_store_ps(p0 + 16, _mm512_extractf32x4_ps(_r0, 1));
                _mm_store_ps(p0 + 16 * 2, _mm512_extractf32x4_ps(_r0, 2));
                _mm_store_ps(p0 + 16 * 3, _mm512_extractf32x4_ps(_r0, 3));
                _mm_store_ps(p0 + 16 * 4, _mm512_extractf32x4_ps(_r1, 0));
                _mm_store_ps(p0 + 16 * 5, _mm512_extractf32x4_ps(_r1, 1));
                _mm_store_ps(p0 + 16 * 6, _mm512_extractf32x4_ps(_r1, 2));
                _mm_store_ps(p0 + 16 * 7, _mm512_extractf32x4_ps(_r1, 3));
                _mm_store_ps(p0 + 16 * 8, _mm512_extractf32x4_ps(_r2, 0));
                _mm_store_ps(p0 + 16 * 9, _mm512_extractf32x4_ps(_r2, 1));
                _mm_store_ps(p0 + 16 * 10, _mm512_extractf32x4_ps(_r2, 2));
                _mm_store_ps(p0 + 16 * 11, _mm512_extractf32x4_ps(_r2, 3));
                _mm_store_ps(p0 + 16 * 12, _mm512_extractf32x4_ps(_r3, 0));
                _mm_store_ps(p0 + 16 * 13, _mm512_extractf32x4_ps(_r3, 1));
                _mm_store_ps(p0 + 16 * 14, _mm512_extractf32x4_ps(_r3, 2));
                _mm_store_ps(p0 + 16 * 15, _mm512_extractf32x4_ps(_r3, 3));
                pp += 64;
                p0 += out_hstep * 16;
            }
        }
        if (out_elempack == 8)
        {
            float* p0 = (float*)top_blob + (j / 8 * 8) * out_hstep + (i + ii) * 8;

            int jj = 0;
            if (j % 8 == 4)
            {
                __m512 _r0 = _mm512_load_ps(pp);
                __m512 _r1 = _mm512_load_ps(pp + 16);
                __m512 _r2 = _mm512_load_ps(pp + 16 * 2);
                __m512 _r3 = _mm512_load_ps(pp + 16 * 3);
                transpose16x4_ps(_r0, _r1, _r2, _r3);
                _mm_store_ps(p0 + 4, _mm512_extractf32x4_ps(_r0, 0));
                _mm_store_ps(p0 + 8 + 4, _mm512_extractf32x4_ps(_r0, 1));
                _mm_store_ps(p0 + 8 * 2 + 4, _mm512_extractf32x4_ps(_r0, 2));
                _mm_store_ps(p0 + 8 * 3 + 4, _mm512_extractf32x4_ps(_r0, 3));
                _mm_store_ps(p0 + 8 * 4 + 4, _mm512_extractf32x4_ps(_r1, 0));
                _mm_store_ps(p0 + 8 * 5 + 4, _mm512_extractf32x4_ps(_r1, 1));
                _mm_store_ps(p0 + 8 * 6 + 4, _mm512_extractf32x4_ps(_r1, 2));
                _mm_store_ps(p0 + 8 * 7 + 4, _mm512_extractf32x4_ps(_r1, 3));
                _mm_store_ps(p0 + 8 * 8 + 4, _mm512_extractf32x4_ps(_r2, 0));
                _mm_store_ps(p0 + 8 * 9 + 4, _mm512_extractf32x4_ps(_r2, 1));
                _mm_store_ps(p0 + 8 * 10 + 4, _mm512_extractf32x4_ps(_r2, 2));
                _mm_store_ps(p0 + 8 * 11 + 4, _mm512_extractf32x4_ps(_r2, 3));
                _mm_store_ps(p0 + 8 * 12 + 4, _mm512_extractf32x4_ps(_r3, 0));
                _mm_store_ps(p0 + 8 * 13 + 4, _mm512_extractf32x4_ps(_r3, 1));
                _mm_store_ps(p0 + 8 * 14 + 4, _mm512_extractf32x4_ps(_r3, 2));
                _mm_store_ps(p0 + 8 * 15 + 4, _mm512_extractf32x4_ps(_r3, 3));
                pp += 64;
                p0 += out_hstep * 8;
                jj += 4;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                __m512 _r0 = _mm512_load_ps(pp);
                __m512 _r1 = _mm512_load_ps(pp + 16);
                __m512 _r2 = _mm512_load_ps(pp + 16 * 2);
                __m512 _r3 = _mm512_load_ps(pp + 16 * 3);
                __m512 _r4 = _mm512_load_ps(pp + 16 * 4);
                __m512 _r5 = _mm512_load_ps(pp + 16 * 5);
                __m512 _r6 = _mm512_load_ps(pp + 16 * 6);
                __m512 _r7 = _mm512_load_ps(pp + 16 * 7);
                transpose16x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm512_storeu_ps(p0, _r0);
                _mm512_storeu_ps(p0 + 16, _r1);
                _mm512_storeu_ps(p0 + 16 * 2, _r2);
                _mm512_storeu_ps(p0 + 16 * 3, _r3);
                _mm512_storeu_ps(p0 + 16 * 4, _r4);
                _mm512_storeu_ps(p0 + 16 * 5, _r5);
                _mm512_storeu_ps(p0 + 16 * 6, _r6);
                _mm512_storeu_ps(p0 + 16 * 7, _r7);
                pp += 128;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                __m512 _r0 = _mm512_load_ps(pp);
                __m512 _r1 = _mm512_load_ps(pp + 16);
                __m512 _r2 = _mm512_load_ps(pp + 16 * 2);
                __m512 _r3 = _mm512_load_ps(pp + 16 * 3);
                transpose16x4_ps(_r0, _r1, _r2, _r3);
                _mm_store_ps(p0, _mm512_extractf32x4_ps(_r0, 0));
                _mm_store_ps(p0 + 8, _mm512_extractf32x4_ps(_r0, 1));
                _mm_store_ps(p0 + 8 * 2, _mm512_extractf32x4_ps(_r0, 2));
                _mm_store_ps(p0 + 8 * 3, _mm512_extractf32x4_ps(_r0, 3));
                _mm_store_ps(p0 + 8 * 4, _mm512_extractf32x4_ps(_r1, 0));
                _mm_store_ps(p0 + 8 * 5, _mm512_extractf32x4_ps(_r1, 1));
                _mm_store_ps(p0 + 8 * 6, _mm512_extractf32x4_ps(_r1, 2));
                _mm_store_ps(p0 + 8 * 7, _mm512_extractf32x4_ps(_r1, 3));
                _mm_store_ps(p0 + 8 * 8, _mm512_extractf32x4_ps(_r2, 0));
                _mm_store_ps(p0 + 8 * 9, _mm512_extractf32x4_ps(_r2, 1));
                _mm_store_ps(p0 + 8 * 10, _mm512_extractf32x4_ps(_r2, 2));
                _mm_store_ps(p0 + 8 * 11, _mm512_extractf32x4_ps(_r2, 3));
                _mm_store_ps(p0 + 8 * 12, _mm512_extractf32x4_ps(_r3, 0));
                _mm_store_ps(p0 + 8 * 13, _mm512_extractf32x4_ps(_r3, 1));
                _mm_store_ps(p0 + 8 * 14, _mm512_extractf32x4_ps(_r3, 2));
                _mm_store_ps(p0 + 8 * 15, _mm512_extractf32x4_ps(_r3, 3));
                pp += 64;
                p0 += out_hstep * 8;
            }
        }
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                __m512 _r0 = _mm512_load_ps(pp);
                __m512 _r1 = _mm512_load_ps(pp + 16);
                __m512 _r2 = _mm512_load_ps(pp + 16 * 2);
                __m512 _r3 = _mm512_load_ps(pp + 16 * 3);
                transpose16x4_ps(_r0, _r1, _r2, _r3);
                _mm512_storeu_ps(p0, _r0);
                _mm512_storeu_ps(p0 + 16, _r1);
                _mm512_storeu_ps(p0 + 16 * 2, _r2);
                _mm512_storeu_ps(p0 + 16 * 3, _r3);
                pp += 64;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                __m512 _r0 = _mm512_load_ps(pp);
                _mm512_storeu_ps(p0, _r0);
                pp += 16;
                p0 += out_hstep;
            }
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
#if __AVX512F__
        if (out_elempack == 16)
        {
            float* p0 = (float*)top_blob + (j / 16 * 16) * out_hstep + (i + ii) * 16;

            int jj = 0;
            if (j % 16 == 4)
            {
                __m256 _r0 = _mm256_load_ps(pp);
                __m256 _r1 = _mm256_load_ps(pp + 8);
                __m256 _r2 = _mm256_load_ps(pp + 8 * 2);
                __m256 _r3 = _mm256_load_ps(pp + 8 * 3);
                if (max_jj > 4)
                {
                    // assert max_jj > 8
                    __m256 _r4 = _mm256_load_ps(pp + 8 * 4);
                    __m256 _r5 = _mm256_load_ps(pp + 8 * 5);
                    __m256 _r6 = _mm256_load_ps(pp + 8 * 6);
                    __m256 _r7 = _mm256_load_ps(pp + 8 * 7);
                    __m256 _r8 = _mm256_load_ps(pp + 8 * 8);
                    __m256 _r9 = _mm256_load_ps(pp + 8 * 9);
                    __m256 _ra = _mm256_load_ps(pp + 8 * 10);
                    __m256 _rb = _mm256_load_ps(pp + 8 * 11);
                    transpose8x12_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                    _mm256_storeu_ps(p0 + 4, _r0);
                    _mm_store_ps(p0 + 12, _mm256_extractf128_ps(_r1, 0));
                    _mm_store_ps(p0 + 16 + 4, _mm256_extractf128_ps(_r1, 1));
                    _mm256_store_ps(p0 + 16 + 8, _r2);
                    _mm256_storeu_ps(p0 + 16 * 2 + 4, _r3);
                    _mm_store_ps(p0 + 16 * 2 + 12, _mm256_extractf128_ps(_r4, 0));
                    _mm_store_ps(p0 + 16 * 3 + 4, _mm256_extractf128_ps(_r4, 1));
                    _mm256_store_ps(p0 + 16 * 3 + 8, _r5);
                    _mm256_storeu_ps(p0 + 16 * 4 + 4, _r6);
                    _mm_store_ps(p0 + 16 * 4 + 12, _mm256_extractf128_ps(_r7, 0));
                    _mm_store_ps(p0 + 16 * 5 + 4, _mm256_extractf128_ps(_r7, 1));
                    _mm256_store_ps(p0 + 16 * 5 + 8, _r8);
                    _mm256_storeu_ps(p0 + 16 * 6 + 4, _r9);
                    _mm_store_ps(p0 + 16 * 6 + 12, _mm256_extractf128_ps(_ra, 0));
                    _mm_store_ps(p0 + 16 * 7 + 4, _mm256_extractf128_ps(_ra, 1));
                    _mm256_store_ps(p0 + 16 * 7 + 8, _rb);
                    pp += 96;
                    jj += 12;
                }
                else
                {
                    transpose8x4_ps(_r0, _r1, _r2, _r3);
                    _mm_store_ps(p0 + 4, _mm256_extractf128_ps(_r0, 0));
                    _mm_store_ps(p0 + 16 + 4, _mm256_extractf128_ps(_r0, 1));
                    _mm_store_ps(p0 + 16 * 2 + 4, _mm256_extractf128_ps(_r1, 0));
                    _mm_store_ps(p0 + 16 * 3 + 4, _mm256_extractf128_ps(_r1, 1));
                    _mm_store_ps(p0 + 16 * 4 + 4, _mm256_extractf128_ps(_r2, 0));
                    _mm_store_ps(p0 + 16 * 5 + 4, _mm256_extractf128_ps(_r2, 1));
                    _mm_store_ps(p0 + 16 * 6 + 4, _mm256_extractf128_ps(_r3, 0));
                    _mm_store_ps(p0 + 16 * 7 + 4, _mm256_extractf128_ps(_r3, 1));
                    pp += 32;
                    jj += 4;
                }
                p0 += out_hstep * 16;
            }
            if (j % 16 == 8)
            {
                __m256 _r0 = _mm256_load_ps(pp);
                __m256 _r1 = _mm256_load_ps(pp + 8);
                __m256 _r2 = _mm256_load_ps(pp + 8 * 2);
                __m256 _r3 = _mm256_load_ps(pp + 8 * 3);
                if (max_jj > 4)
                {
                    __m256 _r4 = _mm256_load_ps(pp + 8 * 4);
                    __m256 _r5 = _mm256_load_ps(pp + 8 * 5);
                    __m256 _r6 = _mm256_load_ps(pp + 8 * 6);
                    __m256 _r7 = _mm256_load_ps(pp + 8 * 7);
                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                    _mm256_store_ps(p0 + 8, _r0);
                    _mm256_store_ps(p0 + 16 + 8, _r1);
                    _mm256_store_ps(p0 + 16 * 2 + 8, _r2);
                    _mm256_store_ps(p0 + 16 * 3 + 8, _r3);
                    _mm256_store_ps(p0 + 16 * 4 + 8, _r4);
                    _mm256_store_ps(p0 + 16 * 5 + 8, _r5);
                    _mm256_store_ps(p0 + 16 * 6 + 8, _r6);
                    _mm256_store_ps(p0 + 16 * 7 + 8, _r7);
                    pp += 64;
                    jj += 8;
                }
                else
                {
                    transpose8x4_ps(_r0, _r1, _r2, _r3);
                    _mm_store_ps(p0 + 8, _mm256_extractf128_ps(_r0, 0));
                    _mm_store_ps(p0 + 16 + 8, _mm256_extractf128_ps(_r0, 1));
                    _mm_store_ps(p0 + 16 * 2 + 8, _mm256_extractf128_ps(_r1, 0));
                    _mm_store_ps(p0 + 16 * 3 + 8, _mm256_extractf128_ps(_r1, 1));
                    _mm_store_ps(p0 + 16 * 4 + 8, _mm256_extractf128_ps(_r2, 0));
                    _mm_store_ps(p0 + 16 * 5 + 8, _mm256_extractf128_ps(_r2, 1));
                    _mm_store_ps(p0 + 16 * 6 + 8, _mm256_extractf128_ps(_r3, 0));
                    _mm_store_ps(p0 + 16 * 7 + 8, _mm256_extractf128_ps(_r3, 1));
                    pp += 32;
                    jj += 4;
                }
                p0 += out_hstep * 16;
            }
            if (j % 16 == 12)
            {
                __m256 _r0 = _mm256_load_ps(pp);
                __m256 _r1 = _mm256_load_ps(pp + 8);
                __m256 _r2 = _mm256_load_ps(pp + 8 * 2);
                __m256 _r3 = _mm256_load_ps(pp + 8 * 3);
                transpose8x4_ps(_r0, _r1, _r2, _r3);
                _mm_store_ps(p0 + 12, _mm256_extractf128_ps(_r0, 0));
                _mm_store_ps(p0 + 16 + 12, _mm256_extractf128_ps(_r0, 1));
                _mm_store_ps(p0 + 16 * 2 + 12, _mm256_extractf128_ps(_r1, 0));
                _mm_store_ps(p0 + 16 * 3 + 12, _mm256_extractf128_ps(_r1, 1));
                _mm_store_ps(p0 + 16 * 4 + 12, _mm256_extractf128_ps(_r2, 0));
                _mm_store_ps(p0 + 16 * 5 + 12, _mm256_extractf128_ps(_r2, 1));
                _mm_store_ps(p0 + 16 * 6 + 12, _mm256_extractf128_ps(_r3, 0));
                _mm_store_ps(p0 + 16 * 7 + 12, _mm256_extractf128_ps(_r3, 1));
                pp += 32;
                p0 += out_hstep * 16;
                jj += 4;
            }
            for (; jj + 15 < max_jj; jj += 16)
            {
                __m256 _r0 = _mm256_load_ps(pp);
                __m256 _r1 = _mm256_load_ps(pp + 8);
                __m256 _r2 = _mm256_load_ps(pp + 8 * 2);
                __m256 _r3 = _mm256_load_ps(pp + 8 * 3);
                __m256 _r4 = _mm256_load_ps(pp + 8 * 4);
                __m256 _r5 = _mm256_load_ps(pp + 8 * 5);
                __m256 _r6 = _mm256_load_ps(pp + 8 * 6);
                __m256 _r7 = _mm256_load_ps(pp + 8 * 7);
                __m256 _r8 = _mm256_load_ps(pp + 8 * 8);
                __m256 _r9 = _mm256_load_ps(pp + 8 * 9);
                __m256 _ra = _mm256_load_ps(pp + 8 * 10);
                __m256 _rb = _mm256_load_ps(pp + 8 * 11);
                __m256 _rc = _mm256_load_ps(pp + 8 * 12);
                __m256 _rd = _mm256_load_ps(pp + 8 * 13);
                __m256 _re = _mm256_load_ps(pp + 8 * 14);
                __m256 _rf = _mm256_load_ps(pp + 8 * 15);
                transpose8x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                _mm256_store_ps(p0, _r0);
                _mm256_store_ps(p0 + 8, _r1);
                _mm256_store_ps(p0 + 8 * 2, _r2);
                _mm256_store_ps(p0 + 8 * 3, _r3);
                _mm256_store_ps(p0 + 8 * 4, _r4);
                _mm256_store_ps(p0 + 8 * 5, _r5);
                _mm256_store_ps(p0 + 8 * 6, _r6);
                _mm256_store_ps(p0 + 8 * 7, _r7);
                _mm256_store_ps(p0 + 8 * 8, _r8);
                _mm256_store_ps(p0 + 8 * 9, _r9);
                _mm256_store_ps(p0 + 8 * 10, _ra);
                _mm256_store_ps(p0 + 8 * 11, _rb);
                _mm256_store_ps(p0 + 8 * 12, _rc);
                _mm256_store_ps(p0 + 8 * 13, _rd);
                _mm256_store_ps(p0 + 8 * 14, _re);
                _mm256_store_ps(p0 + 8 * 15, _rf);
                pp += 128;
                p0 += out_hstep * 16;
            }
            for (; jj + 11 < max_jj; jj += 12)
            {
                __m256 _r0 = _mm256_load_ps(pp);
                __m256 _r1 = _mm256_load_ps(pp + 8);
                __m256 _r2 = _mm256_load_ps(pp + 8 * 2);
                __m256 _r3 = _mm256_load_ps(pp + 8 * 3);
                __m256 _r4 = _mm256_load_ps(pp + 8 * 4);
                __m256 _r5 = _mm256_load_ps(pp + 8 * 5);
                __m256 _r6 = _mm256_load_ps(pp + 8 * 6);
                __m256 _r7 = _mm256_load_ps(pp + 8 * 7);
                __m256 _r8 = _mm256_load_ps(pp + 8 * 8);
                __m256 _r9 = _mm256_load_ps(pp + 8 * 9);
                __m256 _ra = _mm256_load_ps(pp + 8 * 10);
                __m256 _rb = _mm256_load_ps(pp + 8 * 11);
                transpose8x12_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb);
                _mm256_store_ps(p0, _r0);
                _mm_store_ps(p0 + 8, _mm256_extractf128_ps(_r1, 0));
                _mm_store_ps(p0 + 16, _mm256_extractf128_ps(_r1, 1));
                _mm256_storeu_ps(p0 + 16 + 4, _r2);
                _mm256_store_ps(p0 + 16 * 2, _r3);
                _mm_store_ps(p0 + 16 * 2 + 8, _mm256_extractf128_ps(_r4, 0));
                _mm_store_ps(p0 + 16 * 3, _mm256_extractf128_ps(_r4, 1));
                _mm256_storeu_ps(p0 + 16 * 3 + 4, _r5);
                _mm256_store_ps(p0 + 16 * 4, _r6);
                _mm_store_ps(p0 + 16 * 4 + 8, _mm256_extractf128_ps(_r7, 0));
                _mm_store_ps(p0 + 16 * 5, _mm256_extractf128_ps(_r7, 1));
                _mm256_storeu_ps(p0 + 16 * 5 + 4, _r8);
                _mm256_store_ps(p0 + 16 * 6, _r9);
                _mm_store_ps(p0 + 16 * 6 + 8, _mm256_extractf128_ps(_ra, 0));
                _mm_store_ps(p0 + 16 * 7, _mm256_extractf128_ps(_ra, 1));
                _mm256_storeu_ps(p0 + 16 * 7 + 4, _rb);
                pp += 96;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                __m256 _r0 = _mm256_load_ps(pp);
                __m256 _r1 = _mm256_load_ps(pp + 8);
                __m256 _r2 = _mm256_load_ps(pp + 8 * 2);
                __m256 _r3 = _mm256_load_ps(pp + 8 * 3);
                __m256 _r4 = _mm256_load_ps(pp + 8 * 4);
                __m256 _r5 = _mm256_load_ps(pp + 8 * 5);
                __m256 _r6 = _mm256_load_ps(pp + 8 * 6);
                __m256 _r7 = _mm256_load_ps(pp + 8 * 7);
                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm256_store_ps(p0, _r0);
                _mm256_store_ps(p0 + 16, _r1);
                _mm256_store_ps(p0 + 16 * 2, _r2);
                _mm256_store_ps(p0 + 16 * 3, _r3);
                _mm256_store_ps(p0 + 16 * 4, _r4);
                _mm256_store_ps(p0 + 16 * 5, _r5);
                _mm256_store_ps(p0 + 16 * 6, _r6);
                _mm256_store_ps(p0 + 16 * 7, _r7);
                pp += 64;
                p0 += out_hstep * 16;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                __m256 _r0 = _mm256_load_ps(pp);
                __m256 _r1 = _mm256_load_ps(pp + 8);
                __m256 _r2 = _mm256_load_ps(pp + 8 * 2);
                __m256 _r3 = _mm256_load_ps(pp + 8 * 3);
                transpose8x4_ps(_r0, _r1, _r2, _r3);
                _mm_store_ps(p0, _mm256_extractf128_ps(_r0, 0));
                _mm_store_ps(p0 + 16, _mm256_extractf128_ps(_r0, 1));
                _mm_store_ps(p0 + 16 * 2, _mm256_extractf128_ps(_r1, 0));
                _mm_store_ps(p0 + 16 * 3, _mm256_extractf128_ps(_r1, 1));
                _mm_store_ps(p0 + 16 * 4, _mm256_extractf128_ps(_r2, 0));
                _mm_store_ps(p0 + 16 * 5, _mm256_extractf128_ps(_r2, 1));
                _mm_store_ps(p0 + 16 * 6, _mm256_extractf128_ps(_r3, 0));
                _mm_store_ps(p0 + 16 * 7, _mm256_extractf128_ps(_r3, 1));
                pp += 32;
                p0 += out_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (out_elempack == 8)
        {
            float* p0 = (float*)top_blob + (j / 8 * 8) * out_hstep + (i + ii) * 8;

            int jj = 0;
            if (j % 8 == 4)
            {
                __m256 _r0 = _mm256_load_ps(pp);
                __m256 _r1 = _mm256_load_ps(pp + 8);
                __m256 _r2 = _mm256_load_ps(pp + 8 * 2);
                __m256 _r3 = _mm256_load_ps(pp + 8 * 3);
                transpose8x4_ps(_r0, _r1, _r2, _r3);
                _mm_store_ps(p0 + 4, _mm256_extractf128_ps(_r0, 0));
                _mm_store_ps(p0 + 8 + 4, _mm256_extractf128_ps(_r0, 1));
                _mm_store_ps(p0 + 8 * 2 + 4, _mm256_extractf128_ps(_r1, 0));
                _mm_store_ps(p0 + 8 * 3 + 4, _mm256_extractf128_ps(_r1, 1));
                _mm_store_ps(p0 + 8 * 4 + 4, _mm256_extractf128_ps(_r2, 0));
                _mm_store_ps(p0 + 8 * 5 + 4, _mm256_extractf128_ps(_r2, 1));
                _mm_store_ps(p0 + 8 * 6 + 4, _mm256_extractf128_ps(_r3, 0));
                _mm_store_ps(p0 + 8 * 7 + 4, _mm256_extractf128_ps(_r3, 1));
                pp += 32;
                p0 += out_hstep * 8;
                jj += 4;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                __m256 _r0 = _mm256_load_ps(pp);
                __m256 _r1 = _mm256_load_ps(pp + 8);
                __m256 _r2 = _mm256_load_ps(pp + 8 * 2);
                __m256 _r3 = _mm256_load_ps(pp + 8 * 3);
                __m256 _r4 = _mm256_load_ps(pp + 8 * 4);
                __m256 _r5 = _mm256_load_ps(pp + 8 * 5);
                __m256 _r6 = _mm256_load_ps(pp + 8 * 6);
                __m256 _r7 = _mm256_load_ps(pp + 8 * 7);
                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm256_storeu_ps(p0, _r0);
                _mm256_storeu_ps(p0 + 8, _r1);
                _mm256_storeu_ps(p0 + 8 * 2, _r2);
                _mm256_storeu_ps(p0 + 8 * 3, _r3);
                _mm256_storeu_ps(p0 + 8 * 4, _r4);
                _mm256_storeu_ps(p0 + 8 * 5, _r5);
                _mm256_storeu_ps(p0 + 8 * 6, _r6);
                _mm256_storeu_ps(p0 + 8 * 7, _r7);
                pp += 64;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                __m256 _r0 = _mm256_load_ps(pp);
                __m256 _r1 = _mm256_load_ps(pp + 8);
                __m256 _r2 = _mm256_load_ps(pp + 8 * 2);
                __m256 _r3 = _mm256_load_ps(pp + 8 * 3);
                transpose8x4_ps(_r0, _r1, _r2, _r3);
                _mm_store_ps(p0, _mm256_extractf128_ps(_r0, 0));
                _mm_store_ps(p0 + 8, _mm256_extractf128_ps(_r0, 1));
                _mm_store_ps(p0 + 8 * 2, _mm256_extractf128_ps(_r1, 0));
                _mm_store_ps(p0 + 8 * 3, _mm256_extractf128_ps(_r1, 1));
                _mm_store_ps(p0 + 8 * 4, _mm256_extractf128_ps(_r2, 0));
                _mm_store_ps(p0 + 8 * 5, _mm256_extractf128_ps(_r2, 1));
                _mm_store_ps(p0 + 8 * 6, _mm256_extractf128_ps(_r3, 0));
                _mm_store_ps(p0 + 8 * 7, _mm256_extractf128_ps(_r3, 1));
                pp += 32;
                p0 += out_hstep * 8;
            }
        }
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                __m256 _r0 = _mm256_load_ps(pp);
                __m256 _r1 = _mm256_load_ps(pp + 8);
                __m256 _r2 = _mm256_load_ps(pp + 8 * 2);
                __m256 _r3 = _mm256_load_ps(pp + 8 * 3);
                transpose8x4_ps(_r0, _r1, _r2, _r3);
                _mm256_storeu_ps(p0, _r0);
                _mm256_storeu_ps(p0 + 8, _r1);
                _mm256_storeu_ps(p0 + 8 * 2, _r2);
                _mm256_storeu_ps(p0 + 8 * 3, _r3);
                pp += 32;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                __m256 _r0 = _mm256_load_ps(pp);
                _mm256_storeu_ps(p0, _r0);
                pp += 8;
                p0 += out_hstep;
            }
        }
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
#if __AVX__
#if __AVX512F__
        if (out_elempack == 16)
        {
            float* p0 = (float*)top_blob + (j / 16 * 16) * out_hstep + (i + ii) * 16;

            int jj = 0;
            if (j % 16 == 4)
            {
                __m128 _r0 = _mm_load_ps(pp);
                __m128 _r1 = _mm_load_ps(pp + 4);
                __m128 _r2 = _mm_load_ps(pp + 4 * 2);
                __m128 _r3 = _mm_load_ps(pp + 4 * 3);
                if (max_jj > 4)
                {
                    // assert max_jj > 8
                    __m128 _r4 = _mm_load_ps(pp + 4 * 4);
                    __m128 _r5 = _mm_load_ps(pp + 4 * 5);
                    __m128 _r6 = _mm_load_ps(pp + 4 * 6);
                    __m128 _r7 = _mm_load_ps(pp + 4 * 7);
                    __m128 _r8 = _mm_load_ps(pp + 4 * 8);
                    __m128 _r9 = _mm_load_ps(pp + 4 * 9);
                    __m128 _ra = _mm_load_ps(pp + 4 * 10);
                    __m128 _rb = _mm_load_ps(pp + 4 * 11);
                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                    _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);
                    _mm_store_ps(p0 + 4, _r0);
                    _mm_store_ps(p0 + 4 + 4, _r4);
                    _mm_store_ps(p0 + 4 + 8, _r8);
                    _mm_store_ps(p0 + 16 + 4, _r1);
                    _mm_store_ps(p0 + 16 + 4 + 4, _r5);
                    _mm_store_ps(p0 + 16 + 4 + 8, _r9);
                    _mm_store_ps(p0 + 16 * 2 + 4, _r2);
                    _mm_store_ps(p0 + 16 * 2 + 4 + 4, _r6);
                    _mm_store_ps(p0 + 16 * 2 + 4 + 8, _ra);
                    _mm_store_ps(p0 + 16 * 3 + 4, _r3);
                    _mm_store_ps(p0 + 16 * 3 + 4 + 4, _r7);
                    _mm_store_ps(p0 + 16 * 3 + 4 + 8, _rb);
                    pp += 48;
                    jj += 12;
                }
                else
                {
                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _mm_store_ps(p0 + 4, _r0);
                    _mm_store_ps(p0 + 16 + 4, _r1);
                    _mm_store_ps(p0 + 16 * 2 + 4, _r2);
                    _mm_store_ps(p0 + 16 * 3 + 4, _r3);
                    pp += 16;
                    jj += 4;
                }
                p0 += out_hstep * 16;
            }
            if (j % 16 == 8)
            {
                __m128 _r0 = _mm_load_ps(pp);
                __m128 _r1 = _mm_load_ps(pp + 4);
                __m128 _r2 = _mm_load_ps(pp + 4 * 2);
                __m128 _r3 = _mm_load_ps(pp + 4 * 3);
                if (max_jj > 4)
                {
                    __m128 _r4 = _mm_load_ps(pp + 4 * 4);
                    __m128 _r5 = _mm_load_ps(pp + 4 * 5);
                    __m128 _r6 = _mm_load_ps(pp + 4 * 6);
                    __m128 _r7 = _mm_load_ps(pp + 4 * 7);
                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                    _mm_store_ps(p0 + 8, _r0);
                    _mm_store_ps(p0 + 8 + 4, _r4);
                    _mm_store_ps(p0 + 16 + 8, _r1);
                    _mm_store_ps(p0 + 16 + 8 + 4, _r5);
                    _mm_store_ps(p0 + 16 * 2 + 8, _r2);
                    _mm_store_ps(p0 + 16 * 2 + 8 + 4, _r6);
                    _mm_store_ps(p0 + 16 * 3 + 8, _r3);
                    _mm_store_ps(p0 + 16 * 3 + 8 + 4, _r7);
                    pp += 32;
                    jj += 8;
                }
                else
                {
                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _mm_store_ps(p0 + 8, _r0);
                    _mm_store_ps(p0 + 16 + 8, _r1);
                    _mm_store_ps(p0 + 16 * 2 + 8, _r2);
                    _mm_store_ps(p0 + 16 * 3 + 8, _r3);
                    pp += 16;
                    jj += 4;
                }
                p0 += out_hstep * 16;
            }
            if (j % 16 == 12)
            {
                __m128 _r0 = _mm_load_ps(pp);
                __m128 _r1 = _mm_load_ps(pp + 4);
                __m128 _r2 = _mm_load_ps(pp + 4 * 2);
                __m128 _r3 = _mm_load_ps(pp + 4 * 3);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_store_ps(p0 + 12, _r0);
                _mm_store_ps(p0 + 16 + 12, _r1);
                _mm_store_ps(p0 + 16 * 2 + 12, _r2);
                _mm_store_ps(p0 + 16 * 3 + 12, _r3);
                pp += 16;
                p0 += out_hstep * 16;
                jj += 4;
            }
            for (; jj + 15 < max_jj; jj += 16)
            {
                __m128 _r0 = _mm_load_ps(pp);
                __m128 _r1 = _mm_load_ps(pp + 4);
                __m128 _r2 = _mm_load_ps(pp + 4 * 2);
                __m128 _r3 = _mm_load_ps(pp + 4 * 3);
                __m128 _r4 = _mm_load_ps(pp + 4 * 4);
                __m128 _r5 = _mm_load_ps(pp + 4 * 5);
                __m128 _r6 = _mm_load_ps(pp + 4 * 6);
                __m128 _r7 = _mm_load_ps(pp + 4 * 7);
                __m128 _r8 = _mm_load_ps(pp + 4 * 8);
                __m128 _r9 = _mm_load_ps(pp + 4 * 9);
                __m128 _ra = _mm_load_ps(pp + 4 * 10);
                __m128 _rb = _mm_load_ps(pp + 4 * 11);
                __m128 _rc = _mm_load_ps(pp + 4 * 12);
                __m128 _rd = _mm_load_ps(pp + 4 * 13);
                __m128 _re = _mm_load_ps(pp + 4 * 14);
                __m128 _rf = _mm_load_ps(pp + 4 * 15);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);
                _MM_TRANSPOSE4_PS(_rc, _rd, _re, _rf);
                _mm_store_ps(p0, _r0);
                _mm_store_ps(p0 + 4, _r4);
                _mm_store_ps(p0 + 4 * 2, _r8);
                _mm_store_ps(p0 + 4 * 3, _rc);
                _mm_store_ps(p0 + 4 * 4, _r1);
                _mm_store_ps(p0 + 4 * 5, _r5);
                _mm_store_ps(p0 + 4 * 6, _r9);
                _mm_store_ps(p0 + 4 * 7, _rd);
                _mm_store_ps(p0 + 4 * 8, _r2);
                _mm_store_ps(p0 + 4 * 9, _r6);
                _mm_store_ps(p0 + 4 * 10, _ra);
                _mm_store_ps(p0 + 4 * 11, _re);
                _mm_store_ps(p0 + 4 * 12, _r3);
                _mm_store_ps(p0 + 4 * 13, _r7);
                _mm_store_ps(p0 + 4 * 14, _rb);
                _mm_store_ps(p0 + 4 * 15, _rf);
                pp += 64;
                p0 += out_hstep * 16;
            }
            for (; jj + 11 < max_jj; jj += 12)
            {
                __m128 _r0 = _mm_load_ps(pp);
                __m128 _r1 = _mm_load_ps(pp + 4);
                __m128 _r2 = _mm_load_ps(pp + 4 * 2);
                __m128 _r3 = _mm_load_ps(pp + 4 * 3);
                __m128 _r4 = _mm_load_ps(pp + 4 * 4);
                __m128 _r5 = _mm_load_ps(pp + 4 * 5);
                __m128 _r6 = _mm_load_ps(pp + 4 * 6);
                __m128 _r7 = _mm_load_ps(pp + 4 * 7);
                __m128 _r8 = _mm_load_ps(pp + 4 * 8);
                __m128 _r9 = _mm_load_ps(pp + 4 * 9);
                __m128 _ra = _mm_load_ps(pp + 4 * 10);
                __m128 _rb = _mm_load_ps(pp + 4 * 11);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);
                _mm_store_ps(p0, _r0);
                _mm_store_ps(p0 + 4, _r4);
                _mm_store_ps(p0 + 8, _r8);
                _mm_store_ps(p0 + 16, _r1);
                _mm_store_ps(p0 + 16 + 4, _r5);
                _mm_store_ps(p0 + 16 + 8, _r9);
                _mm_store_ps(p0 + 16 * 2, _r2);
                _mm_store_ps(p0 + 16 * 2 + 4, _r6);
                _mm_store_ps(p0 + 16 * 2 + 8, _ra);
                _mm_store_ps(p0 + 16 * 3, _r3);
                _mm_store_ps(p0 + 16 * 3 + 4, _r7);
                _mm_store_ps(p0 + 16 * 3 + 8, _rb);
                pp += 48;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                __m128 _r0 = _mm_load_ps(pp);
                __m128 _r1 = _mm_load_ps(pp + 4);
                __m128 _r2 = _mm_load_ps(pp + 4 * 2);
                __m128 _r3 = _mm_load_ps(pp + 4 * 3);
                __m128 _r4 = _mm_load_ps(pp + 4 * 4);
                __m128 _r5 = _mm_load_ps(pp + 4 * 5);
                __m128 _r6 = _mm_load_ps(pp + 4 * 6);
                __m128 _r7 = _mm_load_ps(pp + 4 * 7);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                _mm_store_ps(p0, _r0);
                _mm_store_ps(p0 + 4, _r4);
                _mm_store_ps(p0 + 16, _r1);
                _mm_store_ps(p0 + 16 + 4, _r5);
                _mm_store_ps(p0 + 16 * 2, _r2);
                _mm_store_ps(p0 + 16 * 2 + 4, _r6);
                _mm_store_ps(p0 + 16 * 3, _r3);
                _mm_store_ps(p0 + 16 * 3 + 4, _r7);
                pp += 32;
                p0 += out_hstep * 16;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                __m128 _r0 = _mm_load_ps(pp);
                __m128 _r1 = _mm_load_ps(pp + 4);
                __m128 _r2 = _mm_load_ps(pp + 4 * 2);
                __m128 _r3 = _mm_load_ps(pp + 4 * 3);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_store_ps(p0, _r0);
                _mm_store_ps(p0 + 16, _r1);
                _mm_store_ps(p0 + 16 * 2, _r2);
                _mm_store_ps(p0 + 16 * 3, _r3);
                pp += 16;
                p0 += out_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (out_elempack == 8)
        {
            float* p0 = (float*)top_blob + (j / 8 * 8) * out_hstep + (i + ii) * 8;

            int jj = 0;
            if (j % 8 == 4)
            {
                __m128 _r0 = _mm_load_ps(pp);
                __m128 _r1 = _mm_load_ps(pp + 4);
                __m128 _r2 = _mm_load_ps(pp + 4 * 2);
                __m128 _r3 = _mm_load_ps(pp + 4 * 3);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_store_ps(p0 + 4, _r0);
                _mm_store_ps(p0 + 8 + 4, _r1);
                _mm_store_ps(p0 + 8 * 2 + 4, _r2);
                _mm_store_ps(p0 + 8 * 3 + 4, _r3);
                pp += 16;
                p0 += out_hstep * 8;
                jj += 4;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                __m128 _r0 = _mm_load_ps(pp);
                __m128 _r1 = _mm_load_ps(pp + 4);
                __m128 _r2 = _mm_load_ps(pp + 4 * 2);
                __m128 _r3 = _mm_load_ps(pp + 4 * 3);
                __m128 _r4 = _mm_load_ps(pp + 4 * 4);
                __m128 _r5 = _mm_load_ps(pp + 4 * 5);
                __m128 _r6 = _mm_load_ps(pp + 4 * 6);
                __m128 _r7 = _mm_load_ps(pp + 4 * 7);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                _mm_storeu_ps(p0, _r0);
                _mm_storeu_ps(p0 + 4, _r4);
                _mm_storeu_ps(p0 + 4 * 2, _r1);
                _mm_storeu_ps(p0 + 4 * 3, _r5);
                _mm_storeu_ps(p0 + 4 * 4, _r2);
                _mm_storeu_ps(p0 + 4 * 5, _r6);
                _mm_storeu_ps(p0 + 4 * 6, _r3);
                _mm_storeu_ps(p0 + 4 * 7, _r7);
                pp += 32;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                __m128 _r0 = _mm_load_ps(pp);
                __m128 _r1 = _mm_load_ps(pp + 4);
                __m128 _r2 = _mm_load_ps(pp + 4 * 2);
                __m128 _r3 = _mm_load_ps(pp + 4 * 3);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_store_ps(p0, _r0);
                _mm_store_ps(p0 + 8, _r1);
                _mm_store_ps(p0 + 8 * 2, _r2);
                _mm_store_ps(p0 + 8 * 3, _r3);
                pp += 16;
                p0 += out_hstep * 8;
            }
        }
#endif // __AVX__
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                __m128 _r0 = _mm_load_ps(pp);
                __m128 _r1 = _mm_load_ps(pp + 4);
                __m128 _r2 = _mm_load_ps(pp + 4 * 2);
                __m128 _r3 = _mm_load_ps(pp + 4 * 3);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_storeu_ps(p0, _r0);
                _mm_storeu_ps(p0 + 4, _r1);
                _mm_storeu_ps(p0 + 4 * 2, _r2);
                _mm_storeu_ps(p0 + 4 * 3, _r3);
                pp += 16;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                __m128 _r0 = _mm_load_ps(pp);
                _mm_storeu_ps(p0, _r0);
                pp += 4;
                p0 += out_hstep;
            }
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (out_elempack == 16)
        {
            float* p0 = (float*)top_blob + (j / 16 * 16) * out_hstep + (i + ii) * 16;

            int jj = 0;
            if (j % 16 == 4)
            {
                if (max_jj > 4)
                {
                    // assert max_jj > 8
                    p0[0 + 4] = pp[0];
                    p0[1 + 4] = pp[2];
                    p0[2 + 4] = pp[4];
                    p0[3 + 4] = pp[6];
                    p0[4 + 4] = pp[8];
                    p0[5 + 4] = pp[10];
                    p0[6 + 4] = pp[12];
                    p0[7 + 4] = pp[14];
                    p0[8 + 4] = pp[16];
                    p0[9 + 4] = pp[18];
                    p0[10 + 4] = pp[20];
                    p0[11 + 4] = pp[22];
                    p0[16 + 4] = pp[1];
                    p0[17 + 4] = pp[3];
                    p0[18 + 4] = pp[5];
                    p0[19 + 4] = pp[7];
                    p0[20 + 4] = pp[9];
                    p0[21 + 4] = pp[11];
                    p0[22 + 4] = pp[13];
                    p0[23 + 4] = pp[15];
                    p0[24 + 4] = pp[17];
                    p0[25 + 4] = pp[19];
                    p0[26 + 4] = pp[21];
                    p0[27 + 4] = pp[23];
                    pp += 24;
                    jj += 12;
                }
                else
                {
                    p0[0 + 4] = pp[0];
                    p0[1 + 4] = pp[2];
                    p0[2 + 4] = pp[4];
                    p0[3 + 4] = pp[6];
                    p0[16 + 4] = pp[1];
                    p0[17 + 4] = pp[3];
                    p0[18 + 4] = pp[5];
                    p0[19 + 4] = pp[7];
                    pp += 8;
                    jj += 4;
                }
                p0 += out_hstep * 16;
            }
            if (j % 16 == 8)
            {
                if (max_jj > 4)
                {
                    p0[0 + 8] = pp[0];
                    p0[1 + 8] = pp[2];
                    p0[2 + 8] = pp[4];
                    p0[3 + 8] = pp[6];
                    p0[4 + 8] = pp[8];
                    p0[5 + 8] = pp[10];
                    p0[6 + 8] = pp[12];
                    p0[7 + 8] = pp[14];
                    p0[16 + 8] = pp[1];
                    p0[17 + 8] = pp[3];
                    p0[18 + 8] = pp[5];
                    p0[19 + 8] = pp[7];
                    p0[20 + 8] = pp[9];
                    p0[21 + 8] = pp[11];
                    p0[22 + 8] = pp[13];
                    p0[23 + 8] = pp[15];
                    pp += 16;
                    jj += 8;
                }
                else
                {
                    p0[0 + 8] = pp[0];
                    p0[1 + 8] = pp[2];
                    p0[2 + 8] = pp[4];
                    p0[3 + 8] = pp[6];
                    p0[16 + 8] = pp[1];
                    p0[17 + 8] = pp[3];
                    p0[18 + 8] = pp[5];
                    p0[19 + 8] = pp[7];
                    pp += 8;
                    jj += 4;
                }
                p0 += out_hstep * 16;
            }
            if (j % 16 == 12)
            {
                p0[0 + 12] = pp[0];
                p0[1 + 12] = pp[2];
                p0[2 + 12] = pp[4];
                p0[3 + 12] = pp[6];
                p0[16 + 12] = pp[1];
                p0[17 + 12] = pp[3];
                p0[18 + 12] = pp[5];
                p0[19 + 12] = pp[7];
                pp += 8;
                p0 += out_hstep * 16;
                jj += 4;
            }
            for (; jj + 15 < max_jj; jj += 16)
            {
                p0[0] = pp[0];
                p0[1] = pp[2];
                p0[2] = pp[4];
                p0[3] = pp[6];
                p0[4] = pp[8];
                p0[5] = pp[10];
                p0[6] = pp[12];
                p0[7] = pp[14];
                p0[8] = pp[16];
                p0[9] = pp[18];
                p0[10] = pp[20];
                p0[11] = pp[22];
                p0[12] = pp[24];
                p0[13] = pp[26];
                p0[14] = pp[28];
                p0[15] = pp[30];
                p0[16] = pp[1];
                p0[17] = pp[3];
                p0[18] = pp[5];
                p0[19] = pp[7];
                p0[20] = pp[9];
                p0[21] = pp[11];
                p0[22] = pp[13];
                p0[23] = pp[15];
                p0[24] = pp[17];
                p0[25] = pp[19];
                p0[26] = pp[21];
                p0[27] = pp[23];
                p0[28] = pp[25];
                p0[29] = pp[27];
                p0[30] = pp[29];
                p0[31] = pp[31];
                pp += 32;
                p0 += out_hstep * 16;
            }
            for (; jj + 11 < max_jj; jj += 12)
            {
                p0[0] = pp[0];
                p0[1] = pp[2];
                p0[2] = pp[4];
                p0[3] = pp[6];
                p0[4] = pp[8];
                p0[5] = pp[10];
                p0[6] = pp[12];
                p0[7] = pp[14];
                p0[8] = pp[16];
                p0[9] = pp[18];
                p0[10] = pp[20];
                p0[11] = pp[22];
                p0[16] = pp[1];
                p0[17] = pp[3];
                p0[18] = pp[5];
                p0[19] = pp[7];
                p0[20] = pp[9];
                p0[21] = pp[11];
                p0[22] = pp[13];
                p0[23] = pp[15];
                p0[24] = pp[17];
                p0[25] = pp[19];
                p0[26] = pp[21];
                p0[27] = pp[23];
                pp += 24;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                p0[0] = pp[0];
                p0[1] = pp[2];
                p0[2] = pp[4];
                p0[3] = pp[6];
                p0[4] = pp[8];
                p0[5] = pp[10];
                p0[6] = pp[12];
                p0[7] = pp[14];
                p0[16] = pp[1];
                p0[17] = pp[3];
                p0[18] = pp[5];
                p0[19] = pp[7];
                p0[20] = pp[9];
                p0[21] = pp[11];
                p0[22] = pp[13];
                p0[23] = pp[15];
                pp += 16;
                p0 += out_hstep * 16;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                p0[0] = pp[0];
                p0[1] = pp[2];
                p0[2] = pp[4];
                p0[3] = pp[6];
                p0[16] = pp[1];
                p0[17] = pp[3];
                p0[18] = pp[5];
                p0[19] = pp[7];
                pp += 8;
                p0 += out_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (out_elempack == 8)
        {
            float* p0 = (float*)top_blob + (j / 8 * 8) * out_hstep + (i + ii) * 8;

            int jj = 0;
            if (j % 8 == 4)
            {
                p0[0 + 4] = pp[0];
                p0[1 + 4] = pp[2];
                p0[2 + 4] = pp[4];
                p0[3 + 4] = pp[6];
                p0[8 + 4] = pp[1];
                p0[9 + 4] = pp[3];
                p0[10 + 4] = pp[5];
                p0[11 + 4] = pp[7];
                pp += 8;
                p0 += out_hstep * 8;
                jj += 4;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                p0[0] = pp[0];
                p0[1] = pp[2];
                p0[2] = pp[4];
                p0[3] = pp[6];
                p0[4] = pp[8];
                p0[5] = pp[10];
                p0[6] = pp[12];
                p0[7] = pp[14];
                p0[8] = pp[1];
                p0[9] = pp[3];
                p0[10] = pp[5];
                p0[11] = pp[7];
                p0[12] = pp[9];
                p0[13] = pp[11];
                p0[14] = pp[13];
                p0[15] = pp[15];
                pp += 16;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                p0[0] = pp[0];
                p0[1] = pp[2];
                p0[2] = pp[4];
                p0[3] = pp[6];
                p0[8] = pp[1];
                p0[9] = pp[3];
                p0[10] = pp[5];
                p0[11] = pp[7];
                pp += 8;
                p0 += out_hstep * 8;
            }
        }
#endif // __AVX__
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
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
#endif // __SSE2__
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = pp[0];
                p0[1] = pp[1];
                pp += 2;
                p0 += out_hstep;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (out_elempack == 16)
        {
            float* p0 = (float*)top_blob + (j / 16 * 16) * out_hstep + (i + ii) * 16;

            int jj = 0;
            if (j % 16 == 4)
            {
                if (max_jj > 4)
                {
                    // assert max_jj > 8
                    __m256 _r0 = _mm256_loadu_ps(pp);
                    __m128 _r1 = _mm_loadu_ps(pp + 8);
                    _mm256_storeu_ps(p0 + 4, _r0);
                    _mm_store_ps(p0 + 4 + 8, _r1);
                    pp += 12;
                    jj += 12;
                }
                else
                {
                    __m128 _r0 = _mm_loadu_ps(pp);
                    _mm_storeu_ps(p0 + 4, _r0);
                    pp += 4;
                    jj += 4;
                }
                p0 += out_hstep * 16;
            }
            if (j % 16 == 8)
            {
                if (max_jj > 4)
                {
                    __m256 _r0 = _mm256_loadu_ps(pp);
                    _mm256_store_ps(p0 + 8, _r0);
                    pp += 8;
                    jj += 8;
                }
                else
                {
                    __m128 _r0 = _mm_loadu_ps(pp);
                    _mm_store_ps(p0 + 8, _r0);
                    pp += 4;
                    jj += 4;
                }
                p0 += out_hstep * 16;
            }
            if (j % 16 == 12)
            {
                __m128 _r0 = _mm_loadu_ps(pp);
                _mm_store_ps(p0 + 12, _r0);
                pp += 4;
                p0 += out_hstep * 16;
                jj += 4;
            }
            for (; jj + 15 < max_jj; jj += 16)
            {
                __m512 _r0 = _mm512_loadu_ps(pp);
                _mm512_store_ps(p0, _r0);
                pp += 16;
                p0 += out_hstep * 16;
            }
            for (; jj + 11 < max_jj; jj += 12)
            {
                __m256 _r0 = _mm256_loadu_ps(pp);
                __m128 _r1 = _mm_loadu_ps(pp + 8);
                _mm256_store_ps(p0, _r0);
                _mm_store_ps(p0 + 8, _r1);
                pp += 12;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                __m256 _r0 = _mm256_loadu_ps(pp);
                _mm256_store_ps(p0, _r0);
                pp += 8;
                p0 += out_hstep * 16;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                __m128 _r0 = _mm_loadu_ps(pp);
                _mm_store_ps(p0, _r0);
                pp += 4;
                p0 += out_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (out_elempack == 8)
        {
            float* p0 = (float*)top_blob + (j / 8 * 8) * out_hstep + (i + ii) * 8;

            int jj = 0;
            if (j % 8 == 4)
            {
                p0[0 + 4] = pp[0];
                p0[1 + 4] = pp[1];
                p0[2 + 4] = pp[2];
                p0[3 + 4] = pp[3];
                pp += 4;
                p0 += out_hstep * 8;
                jj += 4;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                __m256 _r0 = _mm256_loadu_ps(pp);
                _mm256_store_ps(p0, _r0);
                pp += 8;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                p0[0] = pp[0];
                p0[1] = pp[1];
                p0[2] = pp[2];
                p0[3] = pp[3];
                pp += 4;
                p0 += out_hstep * 8;
            }
        }
#endif // __AVX__
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                __m128 _r0 = _mm_loadu_ps(pp);
                _mm_store_ps(p0, _r0);
                pp += 4;
                p0 += out_hstep * 4;
            }
        }
#endif // __SSE2__
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = pp[0];
                pp += 1;
                p0 += out_hstep;
            }
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
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            __m512 _sum0;
            __m512 _sum1;
            __m512 _sum2;
            __m512 _sum3;
            __m512 _sum4;
            __m512 _sum5;
            __m512 _sum6;
            __m512 _sum7;
            __m512 _sum8;
            __m512 _sum9;
            __m512 _suma;
            __m512 _sumb;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_ps();
                _sum1 = _mm512_setzero_ps();
                _sum2 = _mm512_setzero_ps();
                _sum3 = _mm512_setzero_ps();
                _sum4 = _mm512_setzero_ps();
                _sum5 = _mm512_setzero_ps();
                _sum6 = _mm512_setzero_ps();
                _sum7 = _mm512_setzero_ps();
                _sum8 = _mm512_setzero_ps();
                _sum9 = _mm512_setzero_ps();
                _suma = _mm512_setzero_ps();
                _sumb = _mm512_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[0]);
                        _sum2 = _mm512_set1_ps(pC[0]);
                        _sum3 = _mm512_set1_ps(pC[0]);
                        _sum4 = _mm512_set1_ps(pC[0]);
                        _sum5 = _mm512_set1_ps(pC[0]);
                        _sum6 = _mm512_set1_ps(pC[0]);
                        _sum7 = _mm512_set1_ps(pC[0]);
                        _sum8 = _mm512_set1_ps(pC[0]);
                        _sum9 = _mm512_set1_ps(pC[0]);
                        _suma = _mm512_set1_ps(pC[0]);
                        _sumb = _mm512_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                        _sum8 = _sum0;
                        _sum9 = _sum0;
                        _suma = _sum0;
                        _sumb = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                        _sum1 = _mm512_loadu_ps(pC + 16);
                        _sum2 = _mm512_loadu_ps(pC + 16 * 2);
                        _sum3 = _mm512_loadu_ps(pC + 16 * 3);
                        _sum4 = _mm512_loadu_ps(pC + 16 * 4);
                        _sum5 = _mm512_loadu_ps(pC + 16 * 5);
                        _sum6 = _mm512_loadu_ps(pC + 16 * 6);
                        _sum7 = _mm512_loadu_ps(pC + 16 * 7);
                        _sum8 = _mm512_loadu_ps(pC + 16 * 8);
                        _sum9 = _mm512_loadu_ps(pC + 16 * 9);
                        _suma = _mm512_loadu_ps(pC + 16 * 10);
                        _sumb = _mm512_loadu_ps(pC + 16 * 11);
                        pC += 192;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[1]);
                        _sum2 = _mm512_set1_ps(pC[2]);
                        _sum3 = _mm512_set1_ps(pC[3]);
                        _sum4 = _mm512_set1_ps(pC[4]);
                        _sum5 = _mm512_set1_ps(pC[5]);
                        _sum6 = _mm512_set1_ps(pC[6]);
                        _sum7 = _mm512_set1_ps(pC[7]);
                        _sum8 = _mm512_set1_ps(pC[8]);
                        _sum9 = _mm512_set1_ps(pC[9]);
                        _suma = _mm512_set1_ps(pC[10]);
                        _sumb = _mm512_set1_ps(pC[11]);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
                _sum1 = _mm512_load_ps(outptr + 16 * 1);
                _sum2 = _mm512_load_ps(outptr + 16 * 2);
                _sum3 = _mm512_load_ps(outptr + 16 * 3);
                _sum4 = _mm512_load_ps(outptr + 16 * 4);
                _sum5 = _mm512_load_ps(outptr + 16 * 5);
                _sum6 = _mm512_load_ps(outptr + 16 * 6);
                _sum7 = _mm512_load_ps(outptr + 16 * 7);
                _sum8 = _mm512_load_ps(outptr + 16 * 8);
                _sum9 = _mm512_load_ps(outptr + 16 * 9);
                _suma = _mm512_load_ps(outptr + 16 * 10);
                _sumb = _mm512_load_ps(outptr + 16 * 11);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m512 _pA = _mm512_load_ps(pA);

                _sum0 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[0]), _sum0);
                _sum1 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[1]), _sum1);
                _sum2 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[2]), _sum2);
                _sum3 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[3]), _sum3);
                _sum4 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[4]), _sum4);
                _sum5 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[5]), _sum5);
                _sum6 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[6]), _sum6);
                _sum7 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[7]), _sum7);
                _sum8 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[8]), _sum8);
                _sum9 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[9]), _sum9);
                _suma = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[10]), _suma);
                _sumb = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[11]), _sumb);

                pA += 16;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr0, _sum0);
                    _mm512_store_ps(outptr0 + 16 * 1, _sum1);
                    _mm512_store_ps(outptr0 + 16 * 2, _sum2);
                    _mm512_store_ps(outptr0 + 16 * 3, _sum3);
                    _mm512_store_ps(outptr0 + 16 * 4, _sum4);
                    _mm512_store_ps(outptr0 + 16 * 5, _sum5);
                    _mm512_store_ps(outptr0 + 16 * 6, _sum6);
                    _mm512_store_ps(outptr0 + 16 * 7, _sum7);
                    _mm512_store_ps(outptr0 + 16 * 8, _sum8);
                    _mm512_store_ps(outptr0 + 16 * 9, _sum9);
                    _mm512_store_ps(outptr0 + 16 * 10, _suma);
                    _mm512_store_ps(outptr0 + 16 * 11, _sumb);
                    outptr0 += 192;
                }
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_sum8, _sum9, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_suma, _sumb, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp8 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp9 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpa = _mm512_shuffle_f32x4(_sum8, _sum9, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpb = _mm512_shuffle_f32x4(_suma, _sumb, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_ps(outptr0, _tmp0);
                    _mm512_storeu_ps(outptr0 + 16, _tmp1);
                    _mm512_storeu_ps(outptr0 + 16 * 2, _tmp2);
                    _mm512_storeu_ps(outptr0 + 16 * 3, _tmp3);
                    _mm512_storeu_ps(outptr0 + 16 * 4, _tmp4);
                    _mm512_storeu_ps(outptr0 + 16 * 5, _tmp5);

                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _tmp6);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16, _tmp7);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 2, _tmp8);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 3, _tmp9);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 4, _tmpa);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 5, _tmpb);

                    outptr0 += 96;
                }
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp8 = _mm512_shuffle_f32x4(_sum8, _sum9, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp9 = _mm512_shuffle_f32x4(_suma, _sumb, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmpa = _mm512_shuffle_f32x4(_sum8, _sum9, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpb = _mm512_shuffle_f32x4(_suma, _sumb, _MM_SHUFFLE(3, 2, 3, 2));

                    _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum4 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum5 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum6 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum8 = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum9 = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(3, 1, 3, 1));
                    _suma = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(2, 0, 2, 0));
                    _sumb = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(outptr0, _sum0);
                    _mm512_storeu_ps(outptr0 + 16, _sum4);
                    _mm512_storeu_ps(outptr0 + 32, _sum8);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4, _sum1);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4 + 16, _sum5);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4 + 32, _sum9);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _sum2);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16, _sum6);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 32, _suma);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12, _sum3);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12 + 16, _sum7);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12 + 32, _sumb);

                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose16x12_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7, _sum8, _sum9, _suma, _sumb);

                    _mm256_storeu_ps(outptr0, _mm512_extractf32x8_ps(_sum0, 0));
                    _mm_storeu_ps(outptr0 + 8, _mm512_extractf32x4_ps(_sum0, 2));
                    _mm_storeu_ps(outptr0 + out_hstep * 1, _mm512_extractf32x4_ps(_sum0, 3));
                    _mm256_storeu_ps(outptr0 + out_hstep * 1 + 4, _mm512_extractf32x8_ps(_sum1, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 2, _mm512_extractf32x8_ps(_sum1, 1));
                    _mm_storeu_ps(outptr0 + out_hstep * 2 + 8, _mm512_extractf32x4_ps(_sum2, 0));
                    _mm_storeu_ps(outptr0 + out_hstep * 3, _mm512_extractf32x4_ps(_sum2, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 3 + 4, _mm512_extractf32x8_ps(_sum2, 1));

                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _mm512_extractf32x8_ps(_sum3, 0));
                    _mm_storeu_ps(outptr0 + out_hstep * 4 + 8, _mm512_extractf32x4_ps(_sum3, 2));
                    _mm_storeu_ps(outptr0 + out_hstep * 5, _mm512_extractf32x4_ps(_sum3, 3));
                    _mm256_storeu_ps(outptr0 + out_hstep * 5 + 4, _mm512_extractf32x8_ps(_sum4, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 6, _mm512_extractf32x8_ps(_sum4, 1));
                    _mm_storeu_ps(outptr0 + out_hstep * 6 + 8, _mm512_extractf32x4_ps(_sum5, 0));
                    _mm_storeu_ps(outptr0 + out_hstep * 7, _mm512_extractf32x4_ps(_sum5, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 7 + 4, _mm512_extractf32x8_ps(_sum5, 1));

                    _mm256_storeu_ps(outptr0 + out_hstep * 8, _mm512_extractf32x8_ps(_sum6, 0));
                    _mm_storeu_ps(outptr0 + out_hstep * 8 + 8, _mm512_extractf32x4_ps(_sum6, 2));
                    _mm_storeu_ps(outptr0 + out_hstep * 9, _mm512_extractf32x4_ps(_sum6, 3));
                    _mm256_storeu_ps(outptr0 + out_hstep * 9 + 4, _mm512_extractf32x8_ps(_sum7, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 10, _mm512_extractf32x8_ps(_sum7, 1));
                    _mm_storeu_ps(outptr0 + out_hstep * 10 + 8, _mm512_extractf32x4_ps(_sum8, 0));
                    _mm_storeu_ps(outptr0 + out_hstep * 11, _mm512_extractf32x4_ps(_sum8, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 11 + 4, _mm512_extractf32x8_ps(_sum8, 1));

                    _mm256_storeu_ps(outptr0 + out_hstep * 12, _mm512_extractf32x8_ps(_sum9, 0));
                    _mm_storeu_ps(outptr0 + out_hstep * 12 + 8, _mm512_extractf32x4_ps(_sum9, 2));
                    _mm_storeu_ps(outptr0 + out_hstep * 13, _mm512_extractf32x4_ps(_sum9, 3));
                    _mm256_storeu_ps(outptr0 + out_hstep * 13 + 4, _mm512_extractf32x8_ps(_suma, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 14, _mm512_extractf32x8_ps(_suma, 1));
                    _mm_storeu_ps(outptr0 + out_hstep * 14 + 8, _mm512_extractf32x4_ps(_sumb, 0));
                    _mm_storeu_ps(outptr0 + out_hstep * 15, _mm512_extractf32x4_ps(_sumb, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 15 + 4, _mm512_extractf32x8_ps(_sumb, 1));

                    outptr0 += 12;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16 * 1, _sum1);
                _mm512_store_ps(outptr + 16 * 2, _sum2);
                _mm512_store_ps(outptr + 16 * 3, _sum3);
                _mm512_store_ps(outptr + 16 * 4, _sum4);
                _mm512_store_ps(outptr + 16 * 5, _sum5);
                _mm512_store_ps(outptr + 16 * 6, _sum6);
                _mm512_store_ps(outptr + 16 * 7, _sum7);
                _mm512_store_ps(outptr + 16 * 8, _sum8);
                _mm512_store_ps(outptr + 16 * 9, _sum9);
                _mm512_store_ps(outptr + 16 * 10, _suma);
                _mm512_store_ps(outptr + 16 * 11, _sumb);
            }

            outptr += 192;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m512 _sum0;
            __m512 _sum1;
            __m512 _sum2;
            __m512 _sum3;
            __m512 _sum4;
            __m512 _sum5;
            __m512 _sum6;
            __m512 _sum7;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_ps();
                _sum1 = _mm512_setzero_ps();
                _sum2 = _mm512_setzero_ps();
                _sum3 = _mm512_setzero_ps();
                _sum4 = _mm512_setzero_ps();
                _sum5 = _mm512_setzero_ps();
                _sum6 = _mm512_setzero_ps();
                _sum7 = _mm512_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[0]);
                        _sum2 = _mm512_set1_ps(pC[0]);
                        _sum3 = _mm512_set1_ps(pC[0]);
                        _sum4 = _mm512_set1_ps(pC[0]);
                        _sum5 = _mm512_set1_ps(pC[0]);
                        _sum6 = _mm512_set1_ps(pC[0]);
                        _sum7 = _mm512_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                        _sum1 = _mm512_loadu_ps(pC + 16);
                        _sum2 = _mm512_loadu_ps(pC + 16 * 2);
                        _sum3 = _mm512_loadu_ps(pC + 16 * 3);
                        _sum4 = _mm512_loadu_ps(pC + 16 * 4);
                        _sum5 = _mm512_loadu_ps(pC + 16 * 5);
                        _sum6 = _mm512_loadu_ps(pC + 16 * 6);
                        _sum7 = _mm512_loadu_ps(pC + 16 * 7);
                        pC += 128;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[1]);
                        _sum2 = _mm512_set1_ps(pC[2]);
                        _sum3 = _mm512_set1_ps(pC[3]);
                        _sum4 = _mm512_set1_ps(pC[4]);
                        _sum5 = _mm512_set1_ps(pC[5]);
                        _sum6 = _mm512_set1_ps(pC[6]);
                        _sum7 = _mm512_set1_ps(pC[7]);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
                _sum1 = _mm512_load_ps(outptr + 16 * 1);
                _sum2 = _mm512_load_ps(outptr + 16 * 2);
                _sum3 = _mm512_load_ps(outptr + 16 * 3);
                _sum4 = _mm512_load_ps(outptr + 16 * 4);
                _sum5 = _mm512_load_ps(outptr + 16 * 5);
                _sum6 = _mm512_load_ps(outptr + 16 * 6);
                _sum7 = _mm512_load_ps(outptr + 16 * 7);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m512 _pA = _mm512_load_ps(pA);

                _sum0 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[0]), _sum0);
                _sum1 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[1]), _sum1);
                _sum2 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[2]), _sum2);
                _sum3 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[3]), _sum3);
                _sum4 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[4]), _sum4);
                _sum5 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[5]), _sum5);
                _sum6 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[6]), _sum6);
                _sum7 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[7]), _sum7);

                pA += 16;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr0, _sum0);
                    _mm512_store_ps(outptr0 + 16 * 1, _sum1);
                    _mm512_store_ps(outptr0 + 16 * 2, _sum2);
                    _mm512_store_ps(outptr0 + 16 * 3, _sum3);
                    _mm512_store_ps(outptr0 + 16 * 4, _sum4);
                    _mm512_store_ps(outptr0 + 16 * 5, _sum5);
                    _mm512_store_ps(outptr0 + 16 * 6, _sum6);
                    _mm512_store_ps(outptr0 + 16 * 7, _sum7);
                    outptr0 += 128;
                }
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_ps(outptr0, _tmp0);
                    _mm512_storeu_ps(outptr0 + 16, _tmp1);
                    _mm512_storeu_ps(outptr0 + 16 * 2, _tmp2);
                    _mm512_storeu_ps(outptr0 + 16 * 3, _tmp3);

                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _tmp4);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16, _tmp5);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 2, _tmp6);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 3, _tmp7);

                    outptr0 += 64;
                }
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 2, 3, 2));

                    _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum4 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum5 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum6 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(outptr0, _sum0);
                    _mm512_storeu_ps(outptr0 + 16, _sum4);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4, _sum1);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4 + 16, _sum5);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _sum2);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16, _sum6);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12, _sum3);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12 + 16, _sum7);

                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose16x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    _mm256_storeu_ps(outptr0, _mm512_extractf32x8_ps(_sum0, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 1, _mm512_extractf32x8_ps(_sum0, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 2, _mm512_extractf32x8_ps(_sum1, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 3, _mm512_extractf32x8_ps(_sum1, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _mm512_extractf32x8_ps(_sum2, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 5, _mm512_extractf32x8_ps(_sum2, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 6, _mm512_extractf32x8_ps(_sum3, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 7, _mm512_extractf32x8_ps(_sum3, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 8, _mm512_extractf32x8_ps(_sum4, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 9, _mm512_extractf32x8_ps(_sum4, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 10, _mm512_extractf32x8_ps(_sum5, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 11, _mm512_extractf32x8_ps(_sum5, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 12, _mm512_extractf32x8_ps(_sum6, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 13, _mm512_extractf32x8_ps(_sum6, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 14, _mm512_extractf32x8_ps(_sum7, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 15, _mm512_extractf32x8_ps(_sum7, 1));

                    outptr0 += 8;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16 * 1, _sum1);
                _mm512_store_ps(outptr + 16 * 2, _sum2);
                _mm512_store_ps(outptr + 16 * 3, _sum3);
                _mm512_store_ps(outptr + 16 * 4, _sum4);
                _mm512_store_ps(outptr + 16 * 5, _sum5);
                _mm512_store_ps(outptr + 16 * 6, _sum6);
                _mm512_store_ps(outptr + 16 * 7, _sum7);
            }

            outptr += 128;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m512 _sum0;
            __m512 _sum1;
            __m512 _sum2;
            __m512 _sum3;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_ps();
                _sum1 = _mm512_setzero_ps();
                _sum2 = _mm512_setzero_ps();
                _sum3 = _mm512_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[0]);
                        _sum2 = _mm512_set1_ps(pC[0]);
                        _sum3 = _mm512_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                        _sum1 = _mm512_loadu_ps(pC + 16);
                        _sum2 = _mm512_loadu_ps(pC + 32);
                        _sum3 = _mm512_loadu_ps(pC + 48);
                        pC += 64;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[1]);
                        _sum2 = _mm512_set1_ps(pC[2]);
                        _sum3 = _mm512_set1_ps(pC[3]);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
                _sum1 = _mm512_load_ps(outptr + 16 * 1);
                _sum2 = _mm512_load_ps(outptr + 16 * 2);
                _sum3 = _mm512_load_ps(outptr + 16 * 3);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m512 _pA = _mm512_load_ps(pA);

                _sum0 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[0]), _sum0);
                _sum1 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[1]), _sum1);
                _sum2 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[2]), _sum2);
                _sum3 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[3]), _sum3);

                pA += 16;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr0, _sum0);
                    _mm512_store_ps(outptr0 + 16 * 1, _sum1);
                    _mm512_store_ps(outptr0 + 16 * 2, _sum2);
                    _mm512_store_ps(outptr0 + 16 * 3, _sum3);
                    outptr0 += 64;
                }
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_ps(outptr0, _tmp0);
                    _mm512_storeu_ps(outptr0 + 16, _tmp1);

                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _tmp2);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16, _tmp3);

                    outptr0 += 32;
                }
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));

                    _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(outptr0, _sum0);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4, _sum1);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _sum2);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12, _sum3);

                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    __m128 _sum0_0 = _mm512_extractf32x4_ps(_sum0, 0);
                    __m128 _sum1_0 = _mm512_extractf32x4_ps(_sum1, 0);
                    __m128 _sum2_0 = _mm512_extractf32x4_ps(_sum2, 0);
                    __m128 _sum3_0 = _mm512_extractf32x4_ps(_sum3, 0);
                    __m128 _sum0_1 = _mm512_extractf32x4_ps(_sum0, 1);
                    __m128 _sum1_1 = _mm512_extractf32x4_ps(_sum1, 1);
                    __m128 _sum2_1 = _mm512_extractf32x4_ps(_sum2, 1);
                    __m128 _sum3_1 = _mm512_extractf32x4_ps(_sum3, 1);
                    __m128 _sum0_2 = _mm512_extractf32x4_ps(_sum0, 2);
                    __m128 _sum1_2 = _mm512_extractf32x4_ps(_sum1, 2);
                    __m128 _sum2_2 = _mm512_extractf32x4_ps(_sum2, 2);
                    __m128 _sum3_2 = _mm512_extractf32x4_ps(_sum3, 2);
                    __m128 _sum0_3 = _mm512_extractf32x4_ps(_sum0, 3);
                    __m128 _sum1_3 = _mm512_extractf32x4_ps(_sum1, 3);
                    __m128 _sum2_3 = _mm512_extractf32x4_ps(_sum2, 3);
                    __m128 _sum3_3 = _mm512_extractf32x4_ps(_sum3, 3);

                    _MM_TRANSPOSE4_PS(_sum0_0, _sum1_0, _sum2_0, _sum3_0);
                    _MM_TRANSPOSE4_PS(_sum0_1, _sum1_1, _sum2_1, _sum3_1);
                    _MM_TRANSPOSE4_PS(_sum0_2, _sum1_2, _sum2_2, _sum3_2);
                    _MM_TRANSPOSE4_PS(_sum0_3, _sum1_3, _sum2_3, _sum3_3);

                    _mm_storeu_ps(outptr0, _sum0_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 1, _sum1_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 2, _sum2_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 3, _sum3_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 4, _sum0_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 5, _sum1_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 6, _sum2_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 7, _sum3_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 8, _sum0_2);
                    _mm_storeu_ps(outptr0 + out_hstep * 9, _sum1_2);
                    _mm_storeu_ps(outptr0 + out_hstep * 10, _sum2_2);
                    _mm_storeu_ps(outptr0 + out_hstep * 11, _sum3_2);
                    _mm_storeu_ps(outptr0 + out_hstep * 12, _sum0_3);
                    _mm_storeu_ps(outptr0 + out_hstep * 13, _sum1_3);
                    _mm_storeu_ps(outptr0 + out_hstep * 14, _sum2_3);
                    _mm_storeu_ps(outptr0 + out_hstep * 15, _sum3_3);

                    outptr0 += 4;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16 * 1, _sum1);
                _mm512_store_ps(outptr + 16 * 2, _sum2);
                _mm512_store_ps(outptr + 16 * 3, _sum3);
            }

            outptr += 64;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m512 _sum0;
            __m512 _sum1;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_ps();
                _sum1 = _mm512_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                        _sum1 = _mm512_loadu_ps(pC + 16);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[1]);
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
                _sum1 = _mm512_load_ps(outptr + 16);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m512 _pA = _mm512_load_ps(pA);

                _sum0 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[0]), _sum0);
                _sum1 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[1]), _sum1);

                pA += 16;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr0, _sum0);
                    _mm512_store_ps(outptr0 + 16, _sum1);
                    outptr0 += 32;
                }
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_ps(outptr0, _tmp0);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _tmp1);

                    outptr0 += 16;
                }
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _mm512_extractf32x4_ps(_sum0, 0));
                    _mm_store_ps(outptr0 + 4, _mm512_extractf32x4_ps(_sum1, 0));

                    _mm_store_ps(outptr0 + out_hstep * 4, _mm512_extractf32x4_ps(_sum0, 1));
                    _mm_store_ps(outptr0 + out_hstep * 4 + 4, _mm512_extractf32x4_ps(_sum1, 1));

                    _mm_store_ps(outptr0 + out_hstep * 8, _mm512_extractf32x4_ps(_sum0, 2));
                    _mm_store_ps(outptr0 + out_hstep * 8 + 4, _mm512_extractf32x4_ps(_sum1, 2));

                    _mm_store_ps(outptr0 + out_hstep * 12, _mm512_extractf32x4_ps(_sum0, 3));
                    _mm_store_ps(outptr0 + out_hstep * 12 + 4, _mm512_extractf32x4_ps(_sum1, 3));
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[16];
                    float sum1[16];
                    _mm512_storeu_ps(sum0, _sum0);
                    _mm512_storeu_ps(sum1, _sum1);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];
                    outptr0[out_hstep * 8] = sum0[8];
                    outptr0[out_hstep * 9] = sum0[9];
                    outptr0[out_hstep * 10] = sum0[10];
                    outptr0[out_hstep * 11] = sum0[11];
                    outptr0[out_hstep * 12] = sum0[12];
                    outptr0[out_hstep * 13] = sum0[13];
                    outptr0[out_hstep * 14] = sum0[14];
                    outptr0[out_hstep * 15] = sum0[15];

                    outptr0[1] = sum1[0];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0[out_hstep * 4 + 1] = sum1[4];
                    outptr0[out_hstep * 5 + 1] = sum1[5];
                    outptr0[out_hstep * 6 + 1] = sum1[6];
                    outptr0[out_hstep * 7 + 1] = sum1[7];
                    outptr0[out_hstep * 8 + 1] = sum1[8];
                    outptr0[out_hstep * 9 + 1] = sum1[9];
                    outptr0[out_hstep * 10 + 1] = sum1[10];
                    outptr0[out_hstep * 11 + 1] = sum1[11];
                    outptr0[out_hstep * 12 + 1] = sum1[12];
                    outptr0[out_hstep * 13 + 1] = sum1[13];
                    outptr0[out_hstep * 14 + 1] = sum1[14];
                    outptr0[out_hstep * 15 + 1] = sum1[15];
                    outptr0 += 2;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16, _sum1);
            }

            outptr += 32;
        }
        for (; jj < max_jj; jj += 1)
        {
            __m512 _sum0;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m512 _pA = _mm512_load_ps(pA);

                _sum0 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[0]), _sum0);

                pA += 16;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr0, _sum0);
                    outptr0 += 16;
                }
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _mm512_extractf32x8_ps(_sum0, 0));
                    _mm256_store_ps(outptr0 + out_hstep * 8, _mm512_extractf32x8_ps(_sum0, 1));
                    outptr0 += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _mm512_extractf32x4_ps(_sum0, 0));
                    _mm_store_ps(outptr0 + out_hstep * 4, _mm512_extractf32x4_ps(_sum0, 1));
                    _mm_store_ps(outptr0 + out_hstep * 8, _mm512_extractf32x4_ps(_sum0, 2));
                    _mm_store_ps(outptr0 + out_hstep * 12, _mm512_extractf32x4_ps(_sum0, 3));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[16];
                    _mm512_storeu_ps(sum0, _sum0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep * 1] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];
                    outptr0[out_hstep * 8] = sum0[8];
                    outptr0[out_hstep * 9] = sum0[9];
                    outptr0[out_hstep * 10] = sum0[10];
                    outptr0[out_hstep * 11] = sum0[11];
                    outptr0[out_hstep * 12] = sum0[12];
                    outptr0[out_hstep * 13] = sum0[13];
                    outptr0[out_hstep * 14] = sum0[14];
                    outptr0[out_hstep * 15] = sum0[15];
                    outptr0++;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
            }

            outptr += 16;
        }

        pAT += max_kk * 16;
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
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

            if (k == 0)
            {
                _sum0 = _mm256_setzero_ps();
                _sum1 = _mm256_setzero_ps();
                _sum2 = _mm256_setzero_ps();
                _sum3 = _mm256_setzero_ps();
                _sum4 = _mm256_setzero_ps();
                _sum5 = _mm256_setzero_ps();
                _sum6 = _mm256_setzero_ps();
                _sum7 = _mm256_setzero_ps();
                _sum8 = _mm256_setzero_ps();
                _sum9 = _mm256_setzero_ps();
                _suma = _mm256_setzero_ps();
                _sumb = _mm256_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                        _sum1 = _mm256_set1_ps(pC[0]);
                        _sum2 = _mm256_set1_ps(pC[0]);
                        _sum3 = _mm256_set1_ps(pC[0]);
                        _sum4 = _mm256_set1_ps(pC[0]);
                        _sum5 = _mm256_set1_ps(pC[0]);
                        _sum6 = _mm256_set1_ps(pC[0]);
                        _sum7 = _mm256_set1_ps(pC[0]);
                        _sum8 = _mm256_set1_ps(pC[0]);
                        _sum9 = _mm256_set1_ps(pC[0]);
                        _suma = _mm256_set1_ps(pC[0]);
                        _sumb = _mm256_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                        _sum8 = _sum0;
                        _sum9 = _sum0;
                        _suma = _sum0;
                        _sumb = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
                        _sum1 = _mm256_loadu_ps(pC + 8);
                        _sum2 = _mm256_loadu_ps(pC + 8 * 2);
                        _sum3 = _mm256_loadu_ps(pC + 8 * 3);
                        _sum4 = _mm256_loadu_ps(pC + 8 * 4);
                        _sum5 = _mm256_loadu_ps(pC + 8 * 5);
                        _sum6 = _mm256_loadu_ps(pC + 8 * 6);
                        _sum7 = _mm256_loadu_ps(pC + 8 * 7);
                        _sum8 = _mm256_loadu_ps(pC + 8 * 8);
                        _sum9 = _mm256_loadu_ps(pC + 8 * 9);
                        _suma = _mm256_loadu_ps(pC + 8 * 10);
                        _sumb = _mm256_loadu_ps(pC + 8 * 11);
                        pC += 96;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                        _sum1 = _mm256_set1_ps(pC[1]);
                        _sum2 = _mm256_set1_ps(pC[2]);
                        _sum3 = _mm256_set1_ps(pC[3]);
                        _sum4 = _mm256_set1_ps(pC[4]);
                        _sum5 = _mm256_set1_ps(pC[5]);
                        _sum6 = _mm256_set1_ps(pC[6]);
                        _sum7 = _mm256_set1_ps(pC[7]);
                        _sum8 = _mm256_set1_ps(pC[8]);
                        _sum9 = _mm256_set1_ps(pC[9]);
                        _suma = _mm256_set1_ps(pC[10]);
                        _sumb = _mm256_set1_ps(pC[11]);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = _mm256_load_ps(outptr);
                _sum1 = _mm256_load_ps(outptr + 8 * 1);
                _sum2 = _mm256_load_ps(outptr + 8 * 2);
                _sum3 = _mm256_load_ps(outptr + 8 * 3);
                _sum4 = _mm256_load_ps(outptr + 8 * 4);
                _sum5 = _mm256_load_ps(outptr + 8 * 5);
                _sum6 = _mm256_load_ps(outptr + 8 * 6);
                _sum7 = _mm256_load_ps(outptr + 8 * 7);
                _sum8 = _mm256_load_ps(outptr + 8 * 8);
                _sum9 = _mm256_load_ps(outptr + 8 * 9);
                _suma = _mm256_load_ps(outptr + 8 * 10);
                _sumb = _mm256_load_ps(outptr + 8 * 11);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = _mm256_load_ps(pA);

                _sum0 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[0]), _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[1]), _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[2]), _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[3]), _sum3);
                _sum4 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[4]), _sum4);
                _sum5 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[5]), _sum5);
                _sum6 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[6]), _sum6);
                _sum7 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[7]), _sum7);
                _sum8 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[8]), _sum8);
                _sum9 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[9]), _sum9);
                _suma = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[10]), _suma);
                _sumb = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[11]), _sumb);

                pA += 8;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _sum0);
                    _mm256_store_ps(outptr0 + 8 * 1, _sum1);
                    _mm256_store_ps(outptr0 + 8 * 2, _sum2);
                    _mm256_store_ps(outptr0 + 8 * 3, _sum3);
                    _mm256_store_ps(outptr0 + 8 * 4, _sum4);
                    _mm256_store_ps(outptr0 + 8 * 5, _sum5);
                    _mm256_store_ps(outptr0 + 8 * 6, _sum6);
                    _mm256_store_ps(outptr0 + 8 * 7, _sum7);
                    _mm256_store_ps(outptr0 + 8 * 8, _sum8);
                    _mm256_store_ps(outptr0 + 8 * 9, _sum9);
                    _mm256_store_ps(outptr0 + 8 * 10, _suma);
                    _mm256_store_ps(outptr0 + 8 * 11, _sumb);
                    outptr0 += 96;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp2 = _mm256_permute2f128_ps(_sum4, _sum5, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp3 = _mm256_permute2f128_ps(_sum6, _sum7, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp4 = _mm256_permute2f128_ps(_sum8, _sum9, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp5 = _mm256_permute2f128_ps(_suma, _sumb, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp6 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp7 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp8 = _mm256_permute2f128_ps(_sum4, _sum5, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp9 = _mm256_permute2f128_ps(_sum6, _sum7, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmpa = _mm256_permute2f128_ps(_sum8, _sum9, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmpb = _mm256_permute2f128_ps(_suma, _sumb, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_ps(outptr0, _tmp0);
                    _mm256_storeu_ps(outptr0 + 8, _tmp1);
                    _mm256_storeu_ps(outptr0 + 8 * 2, _tmp2);
                    _mm256_storeu_ps(outptr0 + 8 * 3, _tmp3);
                    _mm256_storeu_ps(outptr0 + 8 * 4, _tmp4);
                    _mm256_storeu_ps(outptr0 + 8 * 5, _tmp5);

                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _tmp6);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8, _tmp7);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8 * 2, _tmp8);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8 * 3, _tmp9);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8 * 4, _tmpa);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8 * 5, _tmpb);

                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    _mm256_storeu_ps(outptr0, _sum0);
                    _mm256_storeu_ps(outptr0 + out_hstep * 1, _sum1);
                    _mm256_storeu_ps(outptr0 + out_hstep * 2, _sum2);
                    _mm256_storeu_ps(outptr0 + out_hstep * 3, _sum3);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _sum4);
                    _mm256_storeu_ps(outptr0 + out_hstep * 5, _sum5);
                    _mm256_storeu_ps(outptr0 + out_hstep * 6, _sum6);
                    _mm256_storeu_ps(outptr0 + out_hstep * 7, _sum7);

                    __m128 _sum8_0 = _mm256_extractf128_ps(_sum8, 0);
                    __m128 _sum9_0 = _mm256_extractf128_ps(_sum9, 0);
                    __m128 _suma_0 = _mm256_extractf128_ps(_suma, 0);
                    __m128 _sumb_0 = _mm256_extractf128_ps(_sumb, 0);
                    __m128 _sum8_1 = _mm256_extractf128_ps(_sum8, 1);
                    __m128 _sum9_1 = _mm256_extractf128_ps(_sum9, 1);
                    __m128 _suma_1 = _mm256_extractf128_ps(_suma, 1);
                    __m128 _sumb_1 = _mm256_extractf128_ps(_sumb, 1);

                    _MM_TRANSPOSE4_PS(_sum8_0, _sum9_0, _suma_0, _sumb_0);
                    _MM_TRANSPOSE4_PS(_sum8_1, _sum9_1, _suma_1, _sumb_1);

                    _mm_storeu_ps(outptr0 + 8, _sum8_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 1 + 8, _sum9_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 2 + 8, _suma_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 3 + 8, _sumb_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 4 + 8, _sum8_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 5 + 8, _sum9_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 6 + 8, _suma_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 7 + 8, _sumb_1);

                    outptr0 += 12;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8 * 1, _sum1);
                _mm256_store_ps(outptr + 8 * 2, _sum2);
                _mm256_store_ps(outptr + 8 * 3, _sum3);
                _mm256_store_ps(outptr + 8 * 4, _sum4);
                _mm256_store_ps(outptr + 8 * 5, _sum5);
                _mm256_store_ps(outptr + 8 * 6, _sum6);
                _mm256_store_ps(outptr + 8 * 7, _sum7);
                _mm256_store_ps(outptr + 8 * 8, _sum8);
                _mm256_store_ps(outptr + 8 * 9, _sum9);
                _mm256_store_ps(outptr + 8 * 10, _suma);
                _mm256_store_ps(outptr + 8 * 11, _sumb);
            }

            outptr += 96;
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
                _sum0 = _mm256_setzero_ps();
                _sum1 = _mm256_setzero_ps();
                _sum2 = _mm256_setzero_ps();
                _sum3 = _mm256_setzero_ps();
                _sum4 = _mm256_setzero_ps();
                _sum5 = _mm256_setzero_ps();
                _sum6 = _mm256_setzero_ps();
                _sum7 = _mm256_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                        _sum1 = _mm256_set1_ps(pC[0]);
                        _sum2 = _mm256_set1_ps(pC[0]);
                        _sum3 = _mm256_set1_ps(pC[0]);
                        _sum4 = _mm256_set1_ps(pC[0]);
                        _sum5 = _mm256_set1_ps(pC[0]);
                        _sum6 = _mm256_set1_ps(pC[0]);
                        _sum7 = _mm256_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
                        _sum1 = _mm256_loadu_ps(pC + 8);
                        _sum2 = _mm256_loadu_ps(pC + 8 * 2);
                        _sum3 = _mm256_loadu_ps(pC + 8 * 3);
                        _sum4 = _mm256_loadu_ps(pC + 8 * 4);
                        _sum5 = _mm256_loadu_ps(pC + 8 * 5);
                        _sum6 = _mm256_loadu_ps(pC + 8 * 6);
                        _sum7 = _mm256_loadu_ps(pC + 8 * 7);
                        pC += 64;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                        _sum1 = _mm256_set1_ps(pC[1]);
                        _sum2 = _mm256_set1_ps(pC[2]);
                        _sum3 = _mm256_set1_ps(pC[3]);
                        _sum4 = _mm256_set1_ps(pC[4]);
                        _sum5 = _mm256_set1_ps(pC[5]);
                        _sum6 = _mm256_set1_ps(pC[6]);
                        _sum7 = _mm256_set1_ps(pC[7]);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = _mm256_load_ps(outptr);
                _sum1 = _mm256_load_ps(outptr + 8 * 1);
                _sum2 = _mm256_load_ps(outptr + 8 * 2);
                _sum3 = _mm256_load_ps(outptr + 8 * 3);
                _sum4 = _mm256_load_ps(outptr + 8 * 4);
                _sum5 = _mm256_load_ps(outptr + 8 * 5);
                _sum6 = _mm256_load_ps(outptr + 8 * 6);
                _sum7 = _mm256_load_ps(outptr + 8 * 7);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = _mm256_load_ps(pA);

                _sum0 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[0]), _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[1]), _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[2]), _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[3]), _sum3);
                _sum4 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[4]), _sum4);
                _sum5 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[5]), _sum5);
                _sum6 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[6]), _sum6);
                _sum7 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[7]), _sum7);

                pA += 8;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _sum0);
                    _mm256_store_ps(outptr0 + 8 * 1, _sum1);
                    _mm256_store_ps(outptr0 + 8 * 2, _sum2);
                    _mm256_store_ps(outptr0 + 8 * 3, _sum3);
                    _mm256_store_ps(outptr0 + 8 * 4, _sum4);
                    _mm256_store_ps(outptr0 + 8 * 5, _sum5);
                    _mm256_store_ps(outptr0 + 8 * 6, _sum6);
                    _mm256_store_ps(outptr0 + 8 * 7, _sum7);
                    outptr0 += 64;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp2 = _mm256_permute2f128_ps(_sum4, _sum5, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp3 = _mm256_permute2f128_ps(_sum6, _sum7, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp4 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp5 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp6 = _mm256_permute2f128_ps(_sum4, _sum5, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp7 = _mm256_permute2f128_ps(_sum6, _sum7, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_ps(outptr0, _tmp0);
                    _mm256_storeu_ps(outptr0 + 8, _tmp1);
                    _mm256_storeu_ps(outptr0 + 8 * 2, _tmp2);
                    _mm256_storeu_ps(outptr0 + 8 * 3, _tmp3);

                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _tmp4);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8, _tmp5);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8 * 2, _tmp6);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8 * 3, _tmp7);

                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    _mm256_storeu_ps(outptr0, _sum0);
                    _mm256_storeu_ps(outptr0 + out_hstep * 1, _sum1);
                    _mm256_storeu_ps(outptr0 + out_hstep * 2, _sum2);
                    _mm256_storeu_ps(outptr0 + out_hstep * 3, _sum3);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _sum4);
                    _mm256_storeu_ps(outptr0 + out_hstep * 5, _sum5);
                    _mm256_storeu_ps(outptr0 + out_hstep * 6, _sum6);
                    _mm256_storeu_ps(outptr0 + out_hstep * 7, _sum7);

                    outptr0 += 8;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8 * 1, _sum1);
                _mm256_store_ps(outptr + 8 * 2, _sum2);
                _mm256_store_ps(outptr + 8 * 3, _sum3);
                _mm256_store_ps(outptr + 8 * 4, _sum4);
                _mm256_store_ps(outptr + 8 * 5, _sum5);
                _mm256_store_ps(outptr + 8 * 6, _sum6);
                _mm256_store_ps(outptr + 8 * 7, _sum7);
            }

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
                _sum0 = _mm256_setzero_ps();
                _sum1 = _mm256_setzero_ps();
                _sum2 = _mm256_setzero_ps();
                _sum3 = _mm256_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                        _sum1 = _mm256_set1_ps(pC[0]);
                        _sum2 = _mm256_set1_ps(pC[0]);
                        _sum3 = _mm256_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
                        _sum1 = _mm256_loadu_ps(pC + 8);
                        _sum2 = _mm256_loadu_ps(pC + 16);
                        _sum3 = _mm256_loadu_ps(pC + 24);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                        _sum1 = _mm256_set1_ps(pC[1]);
                        _sum2 = _mm256_set1_ps(pC[2]);
                        _sum3 = _mm256_set1_ps(pC[3]);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum0 = _mm256_load_ps(outptr);
                _sum1 = _mm256_load_ps(outptr + 8 * 1);
                _sum2 = _mm256_load_ps(outptr + 8 * 2);
                _sum3 = _mm256_load_ps(outptr + 8 * 3);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = _mm256_load_ps(pA);

                _sum0 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[0]), _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[1]), _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[2]), _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[3]), _sum3);

                pA += 8;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _sum0);
                    _mm256_store_ps(outptr0 + 8 * 1, _sum1);
                    _mm256_store_ps(outptr0 + 8 * 2, _sum2);
                    _mm256_store_ps(outptr0 + 8 * 3, _sum3);
                    outptr0 += 32;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp2 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp3 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_ps(outptr0, _tmp0);
                    _mm256_storeu_ps(outptr0 + 8, _tmp1);

                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _tmp2);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8, _tmp3);

                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    __m128 _sum0_0 = _mm256_extractf128_ps(_sum0, 0);
                    __m128 _sum1_0 = _mm256_extractf128_ps(_sum1, 0);
                    __m128 _sum2_0 = _mm256_extractf128_ps(_sum2, 0);
                    __m128 _sum3_0 = _mm256_extractf128_ps(_sum3, 0);
                    __m128 _sum0_1 = _mm256_extractf128_ps(_sum0, 1);
                    __m128 _sum1_1 = _mm256_extractf128_ps(_sum1, 1);
                    __m128 _sum2_1 = _mm256_extractf128_ps(_sum2, 1);
                    __m128 _sum3_1 = _mm256_extractf128_ps(_sum3, 1);

                    _MM_TRANSPOSE4_PS(_sum0_0, _sum1_0, _sum2_0, _sum3_0);
                    _MM_TRANSPOSE4_PS(_sum0_1, _sum1_1, _sum2_1, _sum3_1);

                    _mm_storeu_ps(outptr0, _sum0_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 1, _sum1_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 2, _sum2_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 3, _sum3_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 4, _sum0_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 5, _sum1_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 6, _sum2_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 7, _sum3_1);

                    outptr0 += 4;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8 * 1, _sum1);
                _mm256_store_ps(outptr + 8 * 2, _sum2);
                _mm256_store_ps(outptr + 8 * 3, _sum3);
            }

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m256 _sum0;
            __m256 _sum1;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_ps();
                _sum1 = _mm256_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                        _sum1 = _mm256_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
                        _sum1 = _mm256_loadu_ps(pC + 8);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                        _sum1 = _mm256_set1_ps(pC[1]);
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum0 = _mm256_load_ps(outptr);
                _sum1 = _mm256_load_ps(outptr + 8);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = _mm256_load_ps(pA);

                _sum0 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[0]), _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[1]), _sum1);

                pA += 8;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _sum0);
                    _mm256_store_ps(outptr0 + 8, _sum1);
                    outptr0 += 16;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_ps(outptr0, _tmp0);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _tmp1);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    float sum1[8];
                    _mm256_storeu_ps(sum0, _sum0);
                    _mm256_storeu_ps(sum1, _sum1);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];

                    outptr0[1] = sum1[0];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0[out_hstep * 4 + 1] = sum1[4];
                    outptr0[out_hstep * 5 + 1] = sum1[5];
                    outptr0[out_hstep * 6 + 1] = sum1[6];
                    outptr0[out_hstep * 7 + 1] = sum1[7];
                    outptr0 += 2;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8, _sum1);
            }

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            __m256 _sum0;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum0 = _mm256_load_ps(outptr);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = _mm256_load_ps(pA);

                _sum0 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[0]), _sum0);

                pA += 8;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _sum0);
                    outptr0 += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _mm256_extractf128_ps(_sum0, 0));
                    _mm_store_ps(outptr0 + out_hstep * 4, _mm256_extractf128_ps(_sum0, 1));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    _mm256_storeu_ps(sum0, _sum0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep * 1] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];
                    outptr0++;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
            }

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

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

            if (k == 0)
            {
                _sum0 = _mm_setzero_ps();
                _sum1 = _mm_setzero_ps();
                _sum2 = _mm_setzero_ps();
                _sum3 = _mm_setzero_ps();
                _sum4 = _mm_setzero_ps();
                _sum5 = _mm_setzero_ps();
                _sum6 = _mm_setzero_ps();
                _sum7 = _mm_setzero_ps();
                _sum8 = _mm_setzero_ps();
                _sum9 = _mm_setzero_ps();
                _suma = _mm_setzero_ps();
                _sumb = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[0]);
                        _sum2 = _mm_set1_ps(pC[0]);
                        _sum3 = _mm_set1_ps(pC[0]);
                        _sum4 = _mm_set1_ps(pC[0]);
                        _sum5 = _mm_set1_ps(pC[0]);
                        _sum6 = _mm_set1_ps(pC[0]);
                        _sum7 = _mm_set1_ps(pC[0]);
                        _sum8 = _mm_set1_ps(pC[0]);
                        _sum9 = _mm_set1_ps(pC[0]);
                        _suma = _mm_set1_ps(pC[0]);
                        _sumb = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                        _sum8 = _sum0;
                        _sum9 = _sum0;
                        _suma = _sum0;
                        _sumb = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _mm_loadu_ps(pC + 4);
                        _sum2 = _mm_loadu_ps(pC + 8);
                        _sum3 = _mm_loadu_ps(pC + 12);
                        _sum4 = _mm_loadu_ps(pC + 16);
                        _sum5 = _mm_loadu_ps(pC + 20);
                        _sum6 = _mm_loadu_ps(pC + 24);
                        _sum7 = _mm_loadu_ps(pC + 28);
                        _sum8 = _mm_loadu_ps(pC + 32);
                        _sum9 = _mm_loadu_ps(pC + 36);
                        _suma = _mm_loadu_ps(pC + 40);
                        _sumb = _mm_loadu_ps(pC + 44);
                        pC += 48;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[1]);
                        _sum2 = _mm_set1_ps(pC[2]);
                        _sum3 = _mm_set1_ps(pC[3]);
                        _sum4 = _mm_set1_ps(pC[4]);
                        _sum5 = _mm_set1_ps(pC[5]);
                        _sum6 = _mm_set1_ps(pC[6]);
                        _sum7 = _mm_set1_ps(pC[7]);
                        _sum8 = _mm_set1_ps(pC[8]);
                        _sum9 = _mm_set1_ps(pC[9]);
                        _suma = _mm_set1_ps(pC[10]);
                        _sumb = _mm_set1_ps(pC[11]);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4 * 1);
                _sum2 = _mm_load_ps(outptr + 4 * 2);
                _sum3 = _mm_load_ps(outptr + 4 * 3);
                _sum4 = _mm_load_ps(outptr + 4 * 4);
                _sum5 = _mm_load_ps(outptr + 4 * 5);
                _sum6 = _mm_load_ps(outptr + 4 * 6);
                _sum7 = _mm_load_ps(outptr + 4 * 7);
                _sum8 = _mm_load_ps(outptr + 4 * 8);
                _sum9 = _mm_load_ps(outptr + 4 * 9);
                _suma = _mm_load_ps(outptr + 4 * 10);
                _sumb = _mm_load_ps(outptr + 4 * 11);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = _mm_load_ps(pA);

                _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[1]), _sum1);
                _sum2 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[2]), _sum2);
                _sum3 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[3]), _sum3);
                _sum4 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[4]), _sum4);
                _sum5 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[5]), _sum5);
                _sum6 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[6]), _sum6);
                _sum7 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[7]), _sum7);
                _sum8 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[8]), _sum8);
                _sum9 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[9]), _sum9);
                _suma = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[10]), _suma);
                _sumb = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[11]), _sumb);

                pA += 4;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _sum0);
                    _mm_store_ps(outptr0 + 4 * 1, _sum1);
                    _mm_store_ps(outptr0 + 4 * 2, _sum2);
                    _mm_store_ps(outptr0 + 4 * 3, _sum3);
                    _mm_store_ps(outptr0 + 4 * 4, _sum4);
                    _mm_store_ps(outptr0 + 4 * 5, _sum5);
                    _mm_store_ps(outptr0 + 4 * 6, _sum6);
                    _mm_store_ps(outptr0 + 4 * 7, _sum7);
                    _mm_store_ps(outptr0 + 4 * 8, _sum8);
                    _mm_store_ps(outptr0 + 4 * 9, _sum9);
                    _mm_store_ps(outptr0 + 4 * 10, _suma);
                    _mm_store_ps(outptr0 + 4 * 11, _sumb);
                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    _MM_TRANSPOSE4_PS(_sum0, _sum1, _sum2, _sum3);
                    _MM_TRANSPOSE4_PS(_sum4, _sum5, _sum6, _sum7);
                    _MM_TRANSPOSE4_PS(_sum8, _sum9, _suma, _sumb);

                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + out_hstep * 1, _sum1);
                    _mm_storeu_ps(outptr0 + out_hstep * 2, _sum2);
                    _mm_storeu_ps(outptr0 + out_hstep * 3, _sum3);
                    _mm_storeu_ps(outptr0 + 4, _sum4);
                    _mm_storeu_ps(outptr0 + out_hstep * 1 + 4, _sum5);
                    _mm_storeu_ps(outptr0 + out_hstep * 2 + 4, _sum6);
                    _mm_storeu_ps(outptr0 + out_hstep * 3 + 4, _sum7);
                    _mm_storeu_ps(outptr0 + 8, _sum8);
                    _mm_storeu_ps(outptr0 + out_hstep * 1 + 8, _sum9);
                    _mm_storeu_ps(outptr0 + out_hstep * 2 + 8, _suma);
                    _mm_storeu_ps(outptr0 + out_hstep * 3 + 8, _sumb);
                    outptr0 += 12;
                }
            }
            else
            {
                _mm_store_ps(outptr, _sum0);
                _mm_store_ps(outptr + 4 * 1, _sum1);
                _mm_store_ps(outptr + 4 * 2, _sum2);
                _mm_store_ps(outptr + 4 * 3, _sum3);
                _mm_store_ps(outptr + 4 * 4, _sum4);
                _mm_store_ps(outptr + 4 * 5, _sum5);
                _mm_store_ps(outptr + 4 * 6, _sum6);
                _mm_store_ps(outptr + 4 * 7, _sum7);
                _mm_store_ps(outptr + 4 * 8, _sum8);
                _mm_store_ps(outptr + 4 * 9, _sum9);
                _mm_store_ps(outptr + 4 * 10, _suma);
                _mm_store_ps(outptr + 4 * 11, _sumb);
            }

            outptr += 48;
        }
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
                _sum0 = _mm_setzero_ps();
                _sum1 = _mm_setzero_ps();
                _sum2 = _mm_setzero_ps();
                _sum3 = _mm_setzero_ps();
                _sum4 = _mm_setzero_ps();
                _sum5 = _mm_setzero_ps();
                _sum6 = _mm_setzero_ps();
                _sum7 = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[0]);
                        _sum2 = _mm_set1_ps(pC[0]);
                        _sum3 = _mm_set1_ps(pC[0]);
                        _sum4 = _mm_set1_ps(pC[0]);
                        _sum5 = _mm_set1_ps(pC[0]);
                        _sum6 = _mm_set1_ps(pC[0]);
                        _sum7 = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _mm_loadu_ps(pC + 4);
                        _sum2 = _mm_loadu_ps(pC + 8);
                        _sum3 = _mm_loadu_ps(pC + 12);
                        _sum4 = _mm_loadu_ps(pC + 16);
                        _sum5 = _mm_loadu_ps(pC + 20);
                        _sum6 = _mm_loadu_ps(pC + 24);
                        _sum7 = _mm_loadu_ps(pC + 28);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[1]);
                        _sum2 = _mm_set1_ps(pC[2]);
                        _sum3 = _mm_set1_ps(pC[3]);
                        _sum4 = _mm_set1_ps(pC[4]);
                        _sum5 = _mm_set1_ps(pC[5]);
                        _sum6 = _mm_set1_ps(pC[6]);
                        _sum7 = _mm_set1_ps(pC[7]);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4 * 1);
                _sum2 = _mm_load_ps(outptr + 4 * 2);
                _sum3 = _mm_load_ps(outptr + 4 * 3);
                _sum4 = _mm_load_ps(outptr + 4 * 4);
                _sum5 = _mm_load_ps(outptr + 4 * 5);
                _sum6 = _mm_load_ps(outptr + 4 * 6);
                _sum7 = _mm_load_ps(outptr + 4 * 7);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = _mm_load_ps(pA);

                _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[1]), _sum1);
                _sum2 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[2]), _sum2);
                _sum3 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[3]), _sum3);
                _sum4 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[4]), _sum4);
                _sum5 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[5]), _sum5);
                _sum6 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[6]), _sum6);
                _sum7 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[7]), _sum7);

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _sum0);
                    _mm_store_ps(outptr0 + 4 * 1, _sum1);
                    _mm_store_ps(outptr0 + 4 * 2, _sum2);
                    _mm_store_ps(outptr0 + 4 * 3, _sum3);
                    _mm_store_ps(outptr0 + 4 * 4, _sum4);
                    _mm_store_ps(outptr0 + 4 * 5, _sum5);
                    _mm_store_ps(outptr0 + 4 * 6, _sum6);
                    _mm_store_ps(outptr0 + 4 * 7, _sum7);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    _MM_TRANSPOSE4_PS(_sum0, _sum1, _sum2, _sum3);
                    _MM_TRANSPOSE4_PS(_sum4, _sum5, _sum6, _sum7);

                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + out_hstep * 1, _sum1);
                    _mm_storeu_ps(outptr0 + out_hstep * 2, _sum2);
                    _mm_storeu_ps(outptr0 + out_hstep * 3, _sum3);
                    _mm_storeu_ps(outptr0 + 4, _sum4);
                    _mm_storeu_ps(outptr0 + out_hstep * 1 + 4, _sum5);
                    _mm_storeu_ps(outptr0 + out_hstep * 2 + 4, _sum6);
                    _mm_storeu_ps(outptr0 + out_hstep * 3 + 4, _sum7);
                    outptr0 += 8;
                }
            }
            else
            {
                _mm_store_ps(outptr, _sum0);
                _mm_store_ps(outptr + 4 * 1, _sum1);
                _mm_store_ps(outptr + 4 * 2, _sum2);
                _mm_store_ps(outptr + 4 * 3, _sum3);
                _mm_store_ps(outptr + 4 * 4, _sum4);
                _mm_store_ps(outptr + 4 * 5, _sum5);
                _mm_store_ps(outptr + 4 * 6, _sum6);
                _mm_store_ps(outptr + 4 * 7, _sum7);
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
                _sum0 = _mm_setzero_ps();
                _sum1 = _mm_setzero_ps();
                _sum2 = _mm_setzero_ps();
                _sum3 = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[0]);
                        _sum2 = _mm_set1_ps(pC[0]);
                        _sum3 = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _mm_loadu_ps(pC + 4);
                        _sum2 = _mm_loadu_ps(pC + 8);
                        _sum3 = _mm_loadu_ps(pC + 12);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[1]);
                        _sum2 = _mm_set1_ps(pC[2]);
                        _sum3 = _mm_set1_ps(pC[3]);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4 * 1);
                _sum2 = _mm_load_ps(outptr + 4 * 2);
                _sum3 = _mm_load_ps(outptr + 4 * 3);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = _mm_load_ps(pA);

                _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[1]), _sum1);
                _sum2 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[2]), _sum2);
                _sum3 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[3]), _sum3);

                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _sum0);
                    _mm_store_ps(outptr0 + 4 * 1, _sum1);
                    _mm_store_ps(outptr0 + 4 * 2, _sum2);
                    _mm_store_ps(outptr0 + 4 * 3, _sum3);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    _MM_TRANSPOSE4_PS(_sum0, _sum1, _sum2, _sum3);

                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + out_hstep * 1, _sum1);
                    _mm_storeu_ps(outptr0 + out_hstep * 2, _sum2);
                    _mm_storeu_ps(outptr0 + out_hstep * 3, _sum3);
                    outptr0 += 4;
                }
            }
            else
            {
                _mm_store_ps(outptr, _sum0);
                _mm_store_ps(outptr + 4 * 1, _sum1);
                _mm_store_ps(outptr + 4 * 2, _sum2);
                _mm_store_ps(outptr + 4 * 3, _sum3);
            }

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                _sum0 = _mm_setzero_ps();
                _sum1 = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _mm_loadu_ps(pC + 4);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[1]);
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = _mm_load_ps(pA);

                _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[1]), _sum1);

                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _sum0);
                    _mm_store_ps(outptr0 + 4, _sum1);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    float sum1[4];
                    _mm_storeu_ps(sum0, _sum0);
                    _mm_storeu_ps(sum1, _sum1);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[1] = sum1[0];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0 += 2;
                }
            }
            else
            {
                _mm_store_ps(outptr, _sum0);
                _mm_store_ps(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            __m128 _sum0;

            if (k == 0)
            {
                _sum0 = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum0 = _mm_load_ps(outptr);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = _mm_load_ps(pA);

                _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum0);

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _sum0);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    _mm_storeu_ps(sum0, _sum0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0++;
                }
            }
            else
            {
                _mm_store_ps(outptr, _sum0);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j;

        const float* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

        int jj = 0;
#if __SSE2__
        for (; jj + 11 < max_jj; jj += 12)
        {
            __m128 _sum00;
            __m128 _sum01;
            __m128 _sum02;
            __m128 _sum10;
            __m128 _sum11;
            __m128 _sum12;

            if (k == 0)
            {
                _sum00 = _mm_setzero_ps();
                _sum01 = _mm_setzero_ps();
                _sum02 = _mm_setzero_ps();
                _sum10 = _mm_setzero_ps();
                _sum11 = _mm_setzero_ps();
                _sum12 = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = _mm_set1_ps(pC[0]);
                        _sum01 = _mm_set1_ps(pC[0]);
                        _sum02 = _mm_set1_ps(pC[0]);
                        _sum10 = _mm_set1_ps(pC[0]);
                        _sum11 = _mm_set1_ps(pC[0]);
                        _sum12 = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = _mm_set1_ps(pC[0]);
                        _sum01 = _mm_set1_ps(pC[0]);
                        _sum02 = _mm_set1_ps(pC[0]);
                        _sum10 = _mm_set1_ps(pC[1]);
                        _sum11 = _mm_set1_ps(pC[1]);
                        _sum12 = _mm_set1_ps(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        __m128 _tmp0 = _mm_loadu_ps(pC);
                        __m128 _tmp1 = _mm_loadu_ps(pC + 4);
                        __m128 _tmp2 = _mm_loadu_ps(pC + 8);
                        __m128 _tmp3 = _mm_loadu_ps(pC + 12);
                        __m128 _tmp4 = _mm_loadu_ps(pC + 16);
                        __m128 _tmp5 = _mm_loadu_ps(pC + 20);
                        _sum00 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _sum01 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                        _sum02 = _mm_shuffle_ps(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                        _sum10 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        _sum11 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                        _sum12 = _mm_shuffle_ps(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                        pC += 24;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = _mm_loadu_ps(pC);
                        _sum01 = _mm_loadu_ps(pC + 4);
                        _sum02 = _mm_loadu_ps(pC + 8);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        _sum12 = _sum02;
                        pC += 12;
                    }
                }
            }
            else
            {
                __m128 _tmp0 = _mm_loadu_ps(outptr);
                __m128 _tmp1 = _mm_loadu_ps(outptr + 4);
                __m128 _tmp2 = _mm_loadu_ps(outptr + 8);
                __m128 _tmp3 = _mm_loadu_ps(outptr + 12);
                __m128 _tmp4 = _mm_loadu_ps(outptr + 16);
                __m128 _tmp5 = _mm_loadu_ps(outptr + 20);
                _sum00 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                _sum01 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                _sum02 = _mm_shuffle_ps(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                _sum10 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                _sum11 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                _sum12 = _mm_shuffle_ps(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB0 = _mm_load_ps(pB);
                __m128 _pB1 = _mm_load_ps(pB + 4);
                __m128 _pB2 = _mm_load_ps(pB + 8);

                __m128 _pA0 = _mm_set1_ps(pA[0]);
                _sum00 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum00);
                _sum01 = _mm_comp_fmadd_ps(_pA0, _pB1, _sum01);
                _sum02 = _mm_comp_fmadd_ps(_pA0, _pB2, _sum02);
                __m128 _pA1 = _mm_set1_ps(pA[1]);
                _sum10 = _mm_comp_fmadd_ps(_pA1, _pB0, _sum10);
                _sum11 = _mm_comp_fmadd_ps(_pA1, _pB1, _sum11);
                _sum12 = _mm_comp_fmadd_ps(_pA1, _pB2, _sum12);

                pA += 2;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_storeu_ps(outptr0, _sum00);
                    _mm_storeu_ps(outptr0 + 4, _sum01);
                    _mm_storeu_ps(outptr0 + 8, _sum02);
                    _mm_storeu_ps(outptr0 + out_hstep, _sum10);
                    _mm_storeu_ps(outptr0 + out_hstep + 4, _sum11);
                    _mm_storeu_ps(outptr0 + out_hstep + 8, _sum12);
                    outptr0 += 12;
                }
            }
            else
            {
                __m128 _tmp0 = _mm_unpacklo_ps(_sum00, _sum10);
                __m128 _tmp1 = _mm_unpackhi_ps(_sum00, _sum10);
                __m128 _tmp2 = _mm_unpacklo_ps(_sum01, _sum11);
                __m128 _tmp3 = _mm_unpackhi_ps(_sum01, _sum11);
                __m128 _tmp4 = _mm_unpacklo_ps(_sum02, _sum12);
                __m128 _tmp5 = _mm_unpackhi_ps(_sum02, _sum12);
                _mm_store_ps(outptr, _tmp0);
                _mm_store_ps(outptr + 4, _tmp1);
                _mm_store_ps(outptr + 8, _tmp2);
                _mm_store_ps(outptr + 12, _tmp3);
                _mm_store_ps(outptr + 16, _tmp4);
                _mm_store_ps(outptr + 20, _tmp5);
            }

            outptr += 24;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _sum00;
            __m128 _sum01;
            __m128 _sum10;
            __m128 _sum11;

            if (k == 0)
            {
                _sum00 = _mm_setzero_ps();
                _sum01 = _mm_setzero_ps();
                _sum10 = _mm_setzero_ps();
                _sum11 = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = _mm_set1_ps(pC[0]);
                        _sum01 = _mm_set1_ps(pC[0]);
                        _sum10 = _mm_set1_ps(pC[0]);
                        _sum11 = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = _mm_set1_ps(pC[0]);
                        _sum01 = _mm_set1_ps(pC[0]);
                        _sum10 = _mm_set1_ps(pC[1]);
                        _sum11 = _mm_set1_ps(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        __m128 _tmp0 = _mm_loadu_ps(pC);
                        __m128 _tmp1 = _mm_loadu_ps(pC + 4);
                        __m128 _tmp2 = _mm_loadu_ps(pC + 8);
                        __m128 _tmp3 = _mm_loadu_ps(pC + 12);
                        _sum00 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _sum01 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                        _sum10 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        _sum11 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = _mm_loadu_ps(pC);
                        _sum01 = _mm_loadu_ps(pC + 4);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        pC += 8;
                    }
                }
            }
            else
            {
                __m128 _tmp0 = _mm_loadu_ps(outptr);
                __m128 _tmp1 = _mm_loadu_ps(outptr + 4);
                __m128 _tmp2 = _mm_loadu_ps(outptr + 8);
                __m128 _tmp3 = _mm_loadu_ps(outptr + 12);
                _sum00 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                _sum01 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                _sum10 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                _sum11 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB0 = _mm_load_ps(pB);
                __m128 _pB1 = _mm_load_ps(pB + 4);

                __m128 _pA0 = _mm_set1_ps(pA[0]);
                _sum00 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum00);
                _sum01 = _mm_comp_fmadd_ps(_pA0, _pB1, _sum01);
                __m128 _pA1 = _mm_set1_ps(pA[1]);
                _sum10 = _mm_comp_fmadd_ps(_pA1, _pB0, _sum10);
                _sum11 = _mm_comp_fmadd_ps(_pA1, _pB1, _sum11);

                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_storeu_ps(outptr0, _sum00);
                    _mm_storeu_ps(outptr0 + 4, _sum01);
                    _mm_storeu_ps(outptr0 + out_hstep, _sum10);
                    _mm_storeu_ps(outptr0 + out_hstep + 4, _sum11);
                    outptr0 += 8;
                }
            }
            else
            {
                __m128 _tmp0 = _mm_unpacklo_ps(_sum00, _sum10);
                __m128 _tmp1 = _mm_unpackhi_ps(_sum00, _sum10);
                __m128 _tmp2 = _mm_unpacklo_ps(_sum01, _sum11);
                __m128 _tmp3 = _mm_unpackhi_ps(_sum01, _sum11);
                _mm_store_ps(outptr, _tmp0);
                _mm_store_ps(outptr + 4, _tmp1);
                _mm_store_ps(outptr + 8, _tmp2);
                _mm_store_ps(outptr + 12, _tmp3);
            }

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                _sum0 = _mm_setzero_ps();
                _sum1 = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        __m128 _tmp0 = _mm_loadu_ps(pC);
                        __m128 _tmp1 = _mm_loadu_ps(pC + 4);
                        _sum0 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _sum1 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _sum0;
                        pC += 4;
                    }
                }
            }
            else
            {
                __m128 _tmp0 = _mm_loadu_ps(outptr);
                __m128 _tmp1 = _mm_loadu_ps(outptr + 4);
                _sum0 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                _sum1 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB = _mm_load_ps(pB);

                _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(pA[0]), _pB, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_mm_set1_ps(pA[1]), _pB, _sum1);

                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + out_hstep, _sum1);
                    outptr0 += 4;
                }
            }
            else
            {
                __m128 _tmp0 = _mm_unpacklo_ps(_sum0, _sum1);
                __m128 _tmp1 = _mm_unpackhi_ps(_sum0, _sum1);
                _mm_storeu_ps(outptr, _tmp0);
                _mm_storeu_ps(outptr + 4, _tmp1);
            }

            outptr += 8;
        }
#endif // __SSE2__
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

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        sum00 = pC[0];
                        sum01 = pC[0];
                        sum10 = pC[0];
                        sum11 = pC[0];
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum00 = pC[0];
                        sum01 = pC[1];
                        sum10 = pC[0];
                        sum11 = pC[1];
                    }
                    if (broadcast_type_C == 3)
                    {
                        sum00 = pC[0];
                        sum01 = pC[1];
                        sum10 = pC[2];
                        sum11 = pC[3];
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        sum00 = pC[0];
                        sum01 = pC[0];
                        sum10 = pC[1];
                        sum11 = pC[1];
                        pC += 2;
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

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum00;
                    outptr0[1] = sum10;
                    outptr0[out_hstep] = sum01;
                    outptr0[out_hstep + 1] = sum11;
                    outptr0 += 2;
                }
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

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        sum0 = pC[0];
                        sum1 = pC[0];
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum0 = pC[0];
                        sum1 = pC[1];
                    }
                    if (broadcast_type_C == 3)
                    {
                        sum0 = pC[0];
                        sum1 = pC[1];
                        pC += 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        sum0 = pC[0];
                        sum1 = pC[0];
                        pC += 1;
                    }
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[0];
                pA += 2;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum0;
                    outptr0[out_hstep] = sum1;
                    outptr0++;
                }
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

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

        int jj = 0;
#if __SSE2__
        for (; jj + 11 < max_jj; jj += 12)
        {
            __m128 _sum0;
            __m128 _sum1;
            __m128 _sum2;

            if (k == 0)
            {
                _sum0 = _mm_setzero_ps();
                _sum1 = _mm_setzero_ps();
                _sum2 = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[0]);
                        _sum2 = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _mm_loadu_ps(pC + 4);
                        _sum2 = _mm_loadu_ps(pC + 8);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = _mm_loadu_ps(outptr);
                _sum1 = _mm_loadu_ps(outptr + 4);
                _sum2 = _mm_loadu_ps(outptr + 8);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB0 = _mm_load_ps(pB);
                __m128 _pB1 = _mm_load_ps(pB + 4);
                __m128 _pB2 = _mm_load_ps(pB + 8);

                __m128 _pA0 = _mm_set1_ps(pA[0]);
                _sum0 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA0, _pB1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_pA0, _pB2, _sum2);

                pA += 1;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + 4, _sum1);
                    _mm_storeu_ps(outptr0 + 8, _sum2);
                    outptr0 += 12;
                }
            }
            else
            {
                _mm_storeu_ps(outptr, _sum0);
                _mm_storeu_ps(outptr + 4, _sum1);
                _mm_storeu_ps(outptr + 8, _sum2);
            }

            outptr += 12;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                _sum0 = _mm_setzero_ps();
                _sum1 = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _mm_loadu_ps(pC + 4);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = _mm_loadu_ps(outptr);
                _sum1 = _mm_loadu_ps(outptr + 4);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB0 = _mm_load_ps(pB);
                __m128 _pB1 = _mm_load_ps(pB + 4);

                __m128 _pA0 = _mm_set1_ps(pA[0]);
                _sum0 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA0, _pB1, _sum1);

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + 4, _sum1);
                    outptr0 += 8;
                }
            }
            else
            {
                _mm_storeu_ps(outptr, _sum0);
                _mm_storeu_ps(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _sum;

            if (k == 0)
            {
                _sum = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum = _mm_loadu_ps(pC);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum = _mm_loadu_ps(outptr);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB = _mm_load_ps(pB);

                _sum = _mm_comp_fmadd_ps(_mm_set1_ps(pA[0]), _pB, _sum);

                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_storeu_ps(outptr0, _sum);
                    outptr0 += 4;
                }
            }
            else
            {
                _mm_storeu_ps(outptr, _sum);
            }

            outptr += 4;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                sum0 = 0.f;
                sum1 = 0.f;

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum0 = pC[0];
                        sum1 = pC[0];
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        sum0 = pC[0];
                        sum1 = pC[1];
                        pC += 2;
                    }
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];

                pA += 1;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum0;
                    outptr0[1] = sum1;
                    outptr0 += 2;
                }
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
            float sum;

            if (k == 0)
            {
                sum = 0.f;

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum = pC[0];
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        sum = pC[0];
                        pC += 1;
                    }
                }
            }
            else
            {
                sum = outptr[0];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum += pA[0] * pB[0];
                pA += 1;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum;
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum;
            }

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void get_optimal_tile_mnk(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrt((float)l2_cache_size / 3 / sizeof(float));

#if __AVX512F__
    TILE_M = std::max(16, tile_size / 16 * 16);
    TILE_N = std::max(4, tile_size / 4 * 4);
    TILE_K = std::max(16, tile_size / 16 * 16);
#elif __AVX__
    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_N = std::max(4, tile_size / 4 * 4);
    TILE_K = std::max(8, tile_size / 8 * 8);
#elif __SSE2__
    TILE_M = std::max(4, tile_size / 4 * 4);
    TILE_N = std::max(4, tile_size / 4 * 4);
    TILE_K = std::max(4, tile_size / 4 * 4);
#else
    TILE_M = std::max(2, tile_size / 2 * 2);
    TILE_N = std::max(1, tile_size);
    TILE_K = std::max(2, tile_size / 2 * 2);
#endif

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __AVX512F__
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 15) / 16 * 16);
#elif __AVX__
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);
#elif __SSE2__
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(float) / TILE_K);

#if __AVX512F__
            TILE_M = std::max(16, tile_size / 16 * 16);
            TILE_N = std::max(4, tile_size / 4 * 4);
#elif __AVX__
            TILE_M = std::max(8, tile_size / 8 * 8);
            TILE_N = std::max(4, tile_size / 4 * 4);
#elif __SSE2__
            TILE_M = std::max(4, tile_size / 4 * 4);
            TILE_N = std::max(4, tile_size / 4 * 4);
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
#if __AVX512F__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 15) / 16 * 16);
#elif __AVX__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#elif __SSE2__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __AVX512F__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#elif __AVX__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#elif __SSE2__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif
    }

    if (nT > 1)
    {
#if __AVX512F__
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 15) / 16 * 16);
#elif __AVX__
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#elif __SSE2__
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
    }

    // always take constant TILE_M/N/K value when provided
    if (constant_TILE_M > 0)
    {
#if __AVX512F__
        TILE_M = (constant_TILE_M + 15) / 16 * 16;
#elif __AVX__
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
#elif __SSE2__
        TILE_M = (constant_TILE_M + 3) / 4 * 4;
#else
        TILE_M = (constant_TILE_M + 1) / 2 * 2;
#endif
    }

    if (constant_TILE_N > 0)
    {
#if __AVX512F__
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
#elif __AVX__
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
#elif __SSE2__
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
#else
        TILE_N = constant_TILE_N;
#endif
    }

    if (constant_TILE_K > 0)
    {
#if __AVX512F__
        TILE_K = (constant_TILE_K + 15) / 16 * 16;
#elif __AVX__
        TILE_K = (constant_TILE_K + 7) / 8 * 8;
#elif __SSE2__
        TILE_K = (constant_TILE_K + 3) / 4 * 4;
#else
        TILE_K = (constant_TILE_K + 1) / 2 * 2;
#endif
    }
}

static int gemm_x86(const Mat& A, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int transA, int transB, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
    const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat BT(TILE_K * TILE_N, nn_K, nn_N, 4u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    // pack B
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
            pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
        }
        else
        {
            transpose_pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
        }
    }

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

    if (nT > nn_M)
    {
        Mat AT(TILE_K * TILE_M, nn_K, nn_M, 4u, opt.workspace_allocator);
        if (AT.empty())
            return -100;

        // pack A
        const int nn_MK = nn_M * nn_K;
        #pragma omp parallel for num_threads(nT)
        for (int ppik = 0; ppik < nn_MK; ppik++)
        {
            const int ppi = ppik / nn_K;
            const int ppk = ppik % nn_K;

            const int i = ppi * TILE_M;
            const int k = ppk * TILE_K;

            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            if (transA)
            {
                transpose_pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
            }
            else
            {
                pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
            }
        }

        const int nn_MN = nn_M * nn_N;
        #pragma omp parallel for num_threads(nT)
        for (int ppij = 0; ppij < nn_MN; ppij++)
        {
            const int ppi = ppij / nn_N;
            const int ppj = ppij % nn_N;

            const int i = ppi * TILE_M;
            const int j = ppj * TILE_N;

            // shadowed variable for less openmp task args
            const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
            const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;

            const int max_ii = std::min((M - i), TILE_M);
            const int max_jj = std::min((N - j), TILE_N);

            Mat topT_tile;
            if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
                topT_tile = topT.channel(get_omp_thread_num());

            if (broadcast_type_C == 3)
            {
                pack_A_tile(C, topT_tile, i, max_ii, j, max_jj);
            }

            const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = !output_transpose && k + TILE_K >= K;

                gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }
    else
    {
        Mat ATX(TILE_K * TILE_M, nn_K, nT, 4u, opt.workspace_allocator);
        if (ATX.empty())
            return -100;

        #pragma omp parallel for num_threads(nT)
        for (int ppi = 0; ppi < nn_M; ppi++)
        {
            const int i = ppi * TILE_M;

            // shadowed variable for less openmp task args
            const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
            const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;

            const int max_ii = std::min((M - i), TILE_M);

            Mat topT_tile;
            if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
                topT_tile = topT.channel(get_omp_thread_num());

            for (int j = 0; j < N; j += TILE_N)
            {
                const int max_jj = std::min((N - j), TILE_N);

                if (broadcast_type_C == 3)
                {
                    pack_A_tile(C, topT_tile, i, max_ii, j, max_jj);
                }

                const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

                for (int k = 0; k < K; k += TILE_K)
                {
                    const int max_kk = std::min((K - k), TILE_K);

                    // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

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

                    bool k_end = !output_transpose && k + TILE_K >= K;

                    gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
                }

                if (output_transpose)
                {
                    transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj);
                }
            }
        }
    }

    return 0;
}

static int gemm_AT_x86(const Mat& AT, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int K, int transB, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat BT(TILE_K * TILE_N, nn_K, nn_N, 4u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    // pack B
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
            pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
        }
        else
        {
            transpose_pack_B_tile(B, BT_tile, j, max_jj, k, max_kk);
        }
    }

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

    const int nn_MN = nn_M * nn_N;
    #pragma omp parallel for num_threads(nT)
    for (int ppij = 0; ppij < nn_MN; ppij++)
    {
        const int ppi = ppij / nn_N;
        const int ppj = ppij % nn_N;

        const int i = ppi * TILE_M;
        const int j = ppj * TILE_N;

        const int max_ii = std::min((M - i), TILE_M);
        const int max_jj = std::min((N - j), TILE_N);

        Mat topT_tile;
        if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
            topT_tile = topT.channel(get_omp_thread_num());

        if (broadcast_type_C == 3)
        {
            pack_A_tile(C, topT_tile, i, max_ii, j, max_jj);
        }

        const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

            bool k_end = !output_transpose && k + TILE_K >= K;

            gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
        }

        if (output_transpose)
        {
            transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

static int gemm_BT_x86(const Mat& A, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int N, int K, int transA, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

    if (nT > nn_M)
    {
        Mat AT(TILE_K * TILE_M, nn_K, nn_M, 4u, opt.workspace_allocator);
        if (AT.empty())
            return -100;

        // pack A
        const int nn_MK = nn_M * nn_K;
        #pragma omp parallel for num_threads(nT)
        for (int ppik = 0; ppik < nn_MK; ppik++)
        {
            const int ppi = ppik / nn_K;
            const int ppk = ppik % nn_K;

            const int i = ppi * TILE_M;
            const int k = ppk * TILE_K;

            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            if (transA)
            {
                transpose_pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
            }
            else
            {
                pack_A_tile(A, AT_tile, i, max_ii, k, max_kk);
            }
        }

        const int nn_MN = nn_M * nn_N;
        #pragma omp parallel for num_threads(nT)
        for (int ppij = 0; ppij < nn_MN; ppij++)
        {
            const int ppi = ppij / nn_N;
            const int ppj = ppij % nn_N;

            const int i = ppi * TILE_M;
            const int j = ppj * TILE_N;

            // shadowed variable for less openmp task args
            const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
            const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;

            const int max_ii = std::min((M - i), TILE_M);
            const int max_jj = std::min((N - j), TILE_N);

            Mat topT_tile;
            if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
                topT_tile = topT.channel(get_omp_thread_num());

            if (broadcast_type_C == 3)
            {
                pack_A_tile(C, topT_tile, i, max_ii, j, max_jj);
            }

            const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = !output_transpose && k + TILE_K >= K;

                gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
            }

            if (output_transpose)
            {
                transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj);
            }
        }
    }
    else
    {
        Mat ATX(TILE_K * TILE_M, nn_K, nT, 4u, opt.workspace_allocator);
        if (ATX.empty())
            return -100;

        #pragma omp parallel for num_threads(nT)
        for (int ppi = 0; ppi < nn_M; ppi++)
        {
            const int i = ppi * TILE_M;

            // shadowed variable for less openmp task args
            const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
            const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;

            const int max_ii = std::min((M - i), TILE_M);

            Mat topT_tile;
            if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
                topT_tile = topT.channel(get_omp_thread_num());

            for (int j = 0; j < N; j += TILE_N)
            {
                const int max_jj = std::min((N - j), TILE_N);

                if (broadcast_type_C == 3)
                {
                    pack_A_tile(C, topT_tile, i, max_ii, j, max_jj);
                }

                const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

                for (int k = 0; k < K; k += TILE_K)
                {
                    const int max_kk = std::min((K - k), TILE_K);

                    // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

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

                    bool k_end = !output_transpose && k + TILE_K >= K;

                    gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
                }

                if (output_transpose)
                {
                    transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj);
                }
            }
        }
    }

    return 0;
}

static int gemm_AT_BT_x86(const Mat& AT, const Mat& BT, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int N, int K, int output_transpose, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;

    Mat topT;
    if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
    {
        topT.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT.empty())
            return -100;
    }

    const int nn_MN = nn_M * nn_N;
    #pragma omp parallel for num_threads(nT)
    for (int ppij = 0; ppij < nn_MN; ppij++)
    {
        const int ppi = ppij / nn_N;
        const int ppj = ppij % nn_N;

        const int i = ppi * TILE_M;
        const int j = ppj * TILE_N;

        const int max_ii = std::min((M - i), TILE_M);
        const int max_jj = std::min((N - j), TILE_N);

        Mat topT_tile;
        if (K > TILE_K || broadcast_type_C == 3 || output_transpose)
            topT_tile = topT.channel(get_omp_thread_num());

        if (broadcast_type_C == 3)
        {
            pack_A_tile(C, topT_tile, i, max_ii, j, max_jj);
        }

        const Mat& CT_tile = broadcast_type_C == 3 ? topT_tile : C;

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

            bool k_end = !output_transpose && k + TILE_K >= K;

            gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
        }

        if (output_transpose)
        {
            transpose_unpack_output_tile(topT_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}

int Gemm_x86::create_pipeline(const Option& opt)
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        return create_pipeline_int8(opt);
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

            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

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

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

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

#if __SSE2__
        if (constant_broadcast_type_C == 3 && opt.use_packing_layout)
        {
#if __AVX512F__
            int C_elempack = constantM % 16 == 0 ? 16 : constantM % 8 == 0 ? 8 : constantM % 4 == 0 ? 4 : 1;
#elif __AVX__
            int C_elempack = constantM % 8 == 0 ? 8 : constantM % 4 == 0 ? 4 : 1;
#else
            int C_elempack = constantM % 4 == 0 ? 4 : 1;
#endif
            convert_packing(C_data, CT_data, C_elempack, opt);
            if (CT_data.empty())
                return -100;
        }
#endif // __SSE2__

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

int Gemm_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        // return Gemm::forward_int8(bottom_blobs, top_blobs, opt);
        return forward_int8(bottom_blobs, top_blobs, opt);
    }
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
#if __SSE2__
    if (opt.use_packing_layout)
    {
        int outh = output_transpose ? N : M;
#if __AVX512F__
        out_elempack = outh % 16 == 0 ? 16 : outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#elif __AVX__
        out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
        out_elempack = outh % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__
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
        ret = gemm_AT_BT_x86(AT_data, BT_data, C, top_blob, broadcast_type_C, constantM, constantN, constantK, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantA)
    {
        const Mat& B = bottom_blobs[0];
        ret = gemm_AT_x86(AT_data, B, C, top_blob, broadcast_type_C, constantM, constantK, transB, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantB)
    {
        const Mat& A = bottom_blobs[0];
        ret = gemm_BT_x86(A, BT_data, C, top_blob, broadcast_type_C, constantN, constantK, transA, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else
    {
        const Mat& A = bottom_blobs[0];
        const Mat& B = bottom_blobs[1];
        ret = gemm_x86(A, B, C, top_blob, broadcast_type_C, transA, transB, output_transpose, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
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
static void compute_A_tile_int8_scales(const Mat& A, Mat& scales, float B_scale, Mat& out_descales, int i, int max_ii)
{
    compute_A_tile_fp32_int8_scales(A, scales, B_scale, out_descales, i, max_ii);
}

static void transpose_compute_A_tile_int8_scales(const Mat& A, Mat& scales, float B_scale, Mat& out_descales, int i, int max_ii)
{
    transpose_compute_A_tile_fp32_int8_scales(A, scales, B_scale, out_descales, i, max_ii);
}

static void pack_A_tile_quantize(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
    pack_A_tile_fp32_to_int8(A, AT, i, max_ii, k, max_kk, scales);
}

static void transpose_pack_A_tile_quantize(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
    transpose_pack_A_tile_fp32_to_int8(A, AT, i, max_ii, k, max_kk, scales);
}

static void compute_B_int8_scale(const Mat& B, float& scale)
{
    compute_B_fp32_int8_scale(B, scale);
}

static void pack_B_tile_quantize(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    pack_B_tile_fp32_to_int8(B, BT, j, max_jj, k, max_kk, scale);
}

static void transpose_pack_B_tile_quantize(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    transpose_pack_B_tile_fp32_to_int8(B, BT, j, max_jj, k, max_kk, scale);
}

static void unpack_output_tile_dequantize(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta, int output_transpose)
{
    unpack_output_tile_int32_to_fp32(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, descales, alpha, beta, output_transpose);
}

struct gemm_x86_int8_omp_args
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

static int gemm_x86_int8(const Mat& A, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int transA, int transB, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    // NCNN_LOGE("gemm_x86_int8");

    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
    const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;
    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    int nn_N = (N + TILE_N - 1) / TILE_N;
    int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat ATX;
#if NCNN_AVX512VNNI || NCNN_AVXVNNI
    bool has_w_shift = false;
    if (TILE_K >= 4)
    {
        has_w_shift = ncnn::cpu_support_x86_avx512_vnni() || ncnn::cpu_support_x86_avx_vnni();
#if NCNN_AVXVNNIINT8
        if (ncnn::cpu_support_x86_avx_vnni_int8())
            has_w_shift = false;
#endif // NCNN_AVXVNNIINT8
    }
    if (has_w_shift)
    {
        int w_shift_count = TILE_M >= 16 ? 16 : TILE_M >= 8 ? 8 : TILE_M >= 4 ? 4 : TILE_M >= 2 ? 2 : 1;
        ATX.create((TILE_K + w_shift_count * 4) * TILE_M, (K + TILE_K - 1) / TILE_K, nT, 1u, opt.workspace_allocator);
    }
    else
#endif // NCNN_AVX512VNNI || NCNN_AVXVNNI
    {
        ATX.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, nT, 1u, opt.workspace_allocator);
    }
    if (ATX.empty())
        return -100;
    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 1u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;

    Mat A_int8_scales(M, 4u, opt.workspace_allocator);
    if (A_int8_scales.empty())
        return -100;

    // dynamic quantize B
    float B_int8_scale;
    compute_B_int8_scale(B, B_int8_scale);

    // const float output_descale = 1.f / (A_int8_scale * B_int8_scale);
    Mat output_descales(M, 4u, opt.workspace_allocator);
    if (output_descales.empty())
        return -100;

    // NCNN_LOGE("arm ds %f %f", 1/A_int8_scale, 1/B_int8_scale);

    // pack B
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
            pack_B_tile_quantize(B, BT_tile, j, max_jj, k, max_kk, B_int8_scale);
        else
            transpose_pack_B_tile_quantize(B, BT_tile, j, max_jj, k, max_kk, B_int8_scale);
    }

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    const struct gemm_x86_int8_omp_args args = {TILE_M, TILE_N, TILE_K, broadcast_type_C, transA, output_transpose, alpha, beta};

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        // shadowed variable for less openmp task args
        const int TILE_M = args.TILE_M;
        const int TILE_N = args.TILE_N;
        const int TILE_K = args.TILE_K;
        const int broadcast_type_C = args.broadcast_type_C;
        const int transA = args.transA;
        const int output_transpose = args.output_transpose;
        const float alpha = args.alpha;
        const float beta = args.beta;
        // const int input_elemtype = args.input_elemtype;
        // const int output_elemtype = args.output_elemtype;

        const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
        const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;

        const int i = ppi * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

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

                        // NCNN_LOGE("A_int8_scales %f  B_int8_scale %f", A_int8_scales[0], B_int8_scale);
                    }

                    if (transA)
                        transpose_pack_A_tile_quantize(A, AT_tile, i, max_ii, k, max_kk, A_int8_scales);
                    else
                        pack_A_tile_quantize(A, AT_tile, i, max_ii, k, max_kk, A_int8_scales);
                }

                gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_dequantize(topT_tile, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, output_descales, alpha, beta, output_transpose);
        }
    }

    return 0;
}

static int gemm_AT_x86_int8(const Mat& AT, const Mat& A_int8_scales, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int K, int transB, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    // NCNN_LOGE("gemm_AT_x86_int8");

    const int N = transB ? (B.dims == 3 ? B.c : B.h) * B.elempack : B.w;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    int nn_N = (N + TILE_N - 1) / TILE_N;
    int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 1u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;

    // dynamic quantize B
    float B_int8_scale;
    compute_B_int8_scale(B, B_int8_scale);

    // NCNN_LOGE("%.4f %.4f", A_int8_scale, B_int8_scale);

    // const float output_descale = 1.f / (A_int8_scale * B_int8_scale);
    Mat output_descales(M, 4u, opt.workspace_allocator);
    if (output_descales.empty())
        return -100;

    for (int i = 0; i < M; i++)
    {
        output_descales[i] = 1.f / (A_int8_scales[i] * B_int8_scale);
    }

    // pack B
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
            pack_B_tile_quantize(B, BT_tile, j, max_jj, k, max_kk, B_int8_scale);
        else
            transpose_pack_B_tile_quantize(B, BT_tile, j, max_jj, k, max_kk, B_int8_scale);
    }

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    const struct gemm_x86_int8_omp_args args = {TILE_M, TILE_N, TILE_K, broadcast_type_C, 0, output_transpose, alpha, beta};

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        // shadowed variable for less openmp task args
        const int TILE_M = args.TILE_M;
        const int TILE_N = args.TILE_N;
        const int TILE_K = args.TILE_K;
        const int broadcast_type_C = args.broadcast_type_C;
        const int output_transpose = args.output_transpose;
        const float alpha = args.alpha;
        const float beta = args.beta;
        // const int output_elemtype = args.output_elemtype;

        const int i = ppi * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_dequantize(topT_tile, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, output_descales, alpha, beta, output_transpose);
        }
    }

    return 0;
}

static int gemm_BT_x86_int8(const Mat& A, const Mat& BT, float B_int8_scale, const Mat& C, Mat& top_blob, int broadcast_type_C, int N, int K, int transA, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    // NCNN_LOGE("gemm_BT_x86_int8");

    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    // int nn_N = (N + TILE_N - 1) / TILE_N;

    Mat A_int8_scales(M, 4u, opt.workspace_allocator);
    if (A_int8_scales.empty())
        return -100;

    // const float output_descale = 1.f / (A_int8_scale * B_int8_scale);
    Mat output_descales(M, 4u, opt.workspace_allocator);
    if (output_descales.empty())
        return -100;

    // NCNN_LOGE("scale %.4f  %.4f", A_int8_scale, B_int8_scale);

    Mat ATX;
#if NCNN_AVX512VNNI || NCNN_AVXVNNI
    bool has_w_shift = false;
    if (TILE_K >= 4)
    {
        has_w_shift = ncnn::cpu_support_x86_avx512_vnni() || ncnn::cpu_support_x86_avx_vnni();
#if NCNN_AVXVNNIINT8
        if (ncnn::cpu_support_x86_avx_vnni_int8())
            has_w_shift = false;
#endif // NCNN_AVXVNNIINT8
    }
    if (has_w_shift)
    {
        int w_shift_count = TILE_M >= 16 ? 16 : TILE_M >= 8 ? 8 : TILE_M >= 4 ? 4 : TILE_M >= 2 ? 2 : 1;
        ATX.create((TILE_K + w_shift_count * 4) * TILE_M, (K + TILE_K - 1) / TILE_K, nT, 1u, opt.workspace_allocator);
    }
    else
#endif // NCNN_AVX512VNNI || NCNN_AVXVNNI
    {
        ATX.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, nT, 1u, opt.workspace_allocator);
    }
    if (ATX.empty())
        return -100;

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    const struct gemm_x86_int8_omp_args args = {TILE_M, TILE_N, TILE_K, broadcast_type_C, transA, output_transpose, alpha, beta};

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        // shadowed variable for less openmp task args
        const int TILE_M = args.TILE_M;
        const int TILE_N = args.TILE_N;
        const int TILE_K = args.TILE_K;
        const int broadcast_type_C = args.broadcast_type_C;
        const int transA = args.transA;
        const int output_transpose = args.output_transpose;
        const float alpha = args.alpha;
        const float beta = args.beta;
        // const int input_elemtype = args.input_elemtype;
        // const int output_elemtype = args.output_elemtype;

        const int i = ppi * TILE_M;

        // shadowed variable for less openmp task args
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

                // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

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

                        // NCNN_LOGE("A_int8_scales %f  B_int8_scale %f", A_int8_scales[0], B_int8_scale);
                    }

                    if (transA)
                        transpose_pack_A_tile_quantize(A, AT_tile, i, max_ii, k, max_kk, A_int8_scales);
                    else
                        pack_A_tile_quantize(A, AT_tile, i, max_ii, k, max_kk, A_int8_scales);
                }

                gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_dequantize(topT_tile, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, output_descales, alpha, beta, output_transpose);
        }
    }

    return 0;
}

static int gemm_AT_BT_x86_int8(const Mat& AT, const Mat& A_int8_scales, const Mat& BT, float B_int8_scale, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int N, int K, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    // NCNN_LOGE("gemm_AT_BT_x86_int8");

    // NCNN_LOGE("M/N/K = %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    // NCNN_LOGE("TILE M/N/K = %d %d %d", TILE_M, TILE_N, TILE_K);

    int nn_M = (M + TILE_M - 1) / TILE_M;
    // int nn_N = (N + TILE_N - 1) / TILE_N;

    // const float output_descale = 1.f / (A_int8_scale * B_int8_scale);
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

    const struct gemm_x86_int8_omp_args args = {TILE_M, TILE_N, TILE_K, broadcast_type_C, 0, output_transpose, alpha, beta};

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        // shadowed variable for less openmp task args
        const int TILE_M = args.TILE_M;
        const int TILE_N = args.TILE_N;
        const int TILE_K = args.TILE_K;
        const int broadcast_type_C = args.broadcast_type_C;
        const int output_transpose = args.output_transpose;
        const float alpha = args.alpha;
        const float beta = args.beta;
        // const int output_elemtype = args.output_elemtype;

        const int i = ppi * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                // NCNN_LOGE("max_ii/jj/kk = %d %d %d", max_ii, max_jj, max_kk);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_dequantize(topT_tile, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, output_descales, alpha, beta, output_transpose);
        }
    }

    return 0;
}

int Gemm_x86::create_pipeline_int8(const Option& opt)
{
    if (constantA)
    {
        const int M = constantM;
        const int K = constantK;

        int TILE_M, TILE_N, TILE_K;
        get_optimal_tile_mnk_int8(M, 0, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, opt.num_threads);

        const int nn_M = (M + TILE_M - 1) / TILE_M;

#if NCNN_AVX512VNNI || NCNN_AVXVNNI
        bool has_w_shift = false;
        if (TILE_K >= 4)
        {
            has_w_shift = ncnn::cpu_support_x86_avx512_vnni() || ncnn::cpu_support_x86_avx_vnni();
#if NCNN_AVXVNNIINT8
            if (ncnn::cpu_support_x86_avx_vnni_int8())
                has_w_shift = false;
#endif // NCNN_AVXVNNIINT8
        }
        if (has_w_shift)
        {
            int w_shift_count = TILE_M >= 16 ? 16 : TILE_M >= 8 ? 8 : TILE_M >= 4 ? 4 : TILE_M >= 2 ? 2 : 1;
            AT_data.create((TILE_K + w_shift_count * 4) * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, 1u, (Allocator*)0);
        }
        else
#endif // NCNN_AVX512VNNI || NCNN_AVXVNNI
        {
            AT_data.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, 1u, (Allocator*)0);
        }
        if (AT_data.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ppj = 0; ppj < nn_M; ppj++)
        {
            const int i = ppj * TILE_M;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_ii = std::min((M - i), TILE_M);
                const int max_kk = std::min((K - k), TILE_K);

                Mat AT_tile = AT_data.channel(i / TILE_M).row_range(k / TILE_K, 1);

                if (transA)
                {
                    transpose_pack_A_tile_int8(A_data, AT_tile, i, max_ii, k, max_kk);
                }
                else
                {
                    pack_A_tile_int8(A_data, AT_tile, i, max_ii, k, max_kk);
                }
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
        get_optimal_tile_mnk_int8(0, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, opt.num_threads);

        const int nn_N = (N + TILE_N - 1) / TILE_N;

        BT_data.create(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 1u, (Allocator*)0);
        if (BT_data.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ppj = 0; ppj < nn_N; ppj++)
        {
            const int j = ppj * TILE_N;

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_jj = std::min((N - j), TILE_N);
                const int max_kk = std::min((K - k), TILE_K);

                Mat BT_tile = BT_data.channel(j / TILE_N).row_range(k / TILE_K, 1);

                if (transB)
                {
                    pack_B_tile_int8(B_data, BT_tile, j, max_jj, k, max_kk);
                }
                else
                {
                    transpose_pack_B_tile_int8(B_data, BT_tile, j, max_jj, k, max_kk);
                }
            }
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

int Gemm_x86::forward_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
        }
    }

    int out_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
        int outh = output_transpose ? N : M;
#if __AVX512F__
        out_elempack = outh % 16 == 0 ? 16 : outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#elif __AVX__
        out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
        out_elempack = outh % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__

    // FIXME use output_elempack
    // int output_elempack = out_elempack > 4 ? 4 : out_elempack;

    if (output_elempack)
        out_elempack = output_elempack;
    size_t out_elemsize = 4u * out_elempack;

    // FIXME use output_elemtype instead of input_elemtype
    // int output_elemtype = input_elemtype;

    // TODO use output_elemtype

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
        ret = gemm_AT_BT_x86_int8(AT_data, A_data_int8_scales, BT_data, B_data_int8_scale, C, top_blob, broadcast_type_C, constantM, constantN, constantK, output_transpose, alpha, beta, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantA)
    {
        const Mat& B = bottom_blobs[0];
        ret = gemm_AT_x86_int8(AT_data, A_data_int8_scales, B, C, top_blob, broadcast_type_C, constantM, constantK, transB, output_transpose, alpha, beta, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else if (constantB)
    {
        const Mat& A = bottom_blobs[0];
        ret = gemm_BT_x86_int8(A, BT_data, B_data_int8_scale, C, top_blob, broadcast_type_C, constantN, constantK, transA, output_transpose, alpha, beta, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }
    else
    {
        const Mat& A = bottom_blobs[0];
        const Mat& B = bottom_blobs[1];
        ret = gemm_x86_int8(A, B, C, top_blob, broadcast_type_C, transA, transB, output_transpose, alpha, beta, constant_TILE_M, constant_TILE_N, constant_TILE_K, _nT, opt);
    }

    return ret;
}
#endif

namespace Gemm_x86_utility {
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
} // namespace Gemm_x86_utility

} // namespace ncnn

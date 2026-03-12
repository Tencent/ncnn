// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pack_A_tile_bf16s(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
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
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 16;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm512_store_ps(pp, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)));
                pp += 16;
                p0 += 16;
            }
        }
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 8;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 8) * A_hstep + k * 8;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm256_store_ps(pp, bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)));
                _mm256_store_ps(pp + 8, bfloat2float_avx(_mm_loadu_si128((const __m128i*)p1)));
                pp += 16;
                p0 += 8;
                p1 += 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 4) * A_hstep + k * 4;
            const unsigned short* p2 = (const unsigned short*)A + (i + ii + 8) * A_hstep + k * 4;
            const unsigned short* p3 = (const unsigned short*)A + (i + ii + 12) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
                _mm_store_ps(pp + 4, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p1)));
                _mm_store_ps(pp + 8, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p2)));
                _mm_store_ps(pp + 12, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p3)));
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
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
            const unsigned short* p8 = (const unsigned short*)A + (i + ii + 8) * A_hstep + k;
            const unsigned short* p9 = (const unsigned short*)A + (i + ii + 9) * A_hstep + k;
            const unsigned short* pa = (const unsigned short*)A + (i + ii + 10) * A_hstep + k;
            const unsigned short* pb = (const unsigned short*)A + (i + ii + 11) * A_hstep + k;
            const unsigned short* pc = (const unsigned short*)A + (i + ii + 12) * A_hstep + k;
            const unsigned short* pd = (const unsigned short*)A + (i + ii + 13) * A_hstep + k;
            const unsigned short* pe = (const unsigned short*)A + (i + ii + 14) * A_hstep + k;
            const unsigned short* pf = (const unsigned short*)A + (i + ii + 15) * A_hstep + k;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0));
                __m512 _r1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p1));
                __m512 _r2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p2));
                __m512 _r3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p3));
                __m512 _r4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p4));
                __m512 _r5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p5));
                __m512 _r6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p6));
                __m512 _r7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p7));
                __m512 _r8 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p8));
                __m512 _r9 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p9));
                __m512 _ra = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pa));
                __m512 _rb = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pb));
                __m512 _rc = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pc));
                __m512 _rd = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pd));
                __m512 _re = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pe));
                __m512 _rf = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pf));
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
                pp[0] = bfloat16_to_float32(p0[0]);
                pp[1] = bfloat16_to_float32(p1[0]);
                pp[2] = bfloat16_to_float32(p2[0]);
                pp[3] = bfloat16_to_float32(p3[0]);
                pp[4] = bfloat16_to_float32(p4[0]);
                pp[5] = bfloat16_to_float32(p5[0]);
                pp[6] = bfloat16_to_float32(p6[0]);
                pp[7] = bfloat16_to_float32(p7[0]);
                pp[8] = bfloat16_to_float32(p8[0]);
                pp[9] = bfloat16_to_float32(p9[0]);
                pp[10] = bfloat16_to_float32(pa[0]);
                pp[11] = bfloat16_to_float32(pb[0]);
                pp[12] = bfloat16_to_float32(pc[0]);
                pp[13] = bfloat16_to_float32(pd[0]);
                pp[14] = bfloat16_to_float32(pe[0]);
                pp[15] = bfloat16_to_float32(pf[0]);
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
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 8;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm256_store_ps(pp, bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)));
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
                _mm_store_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
                _mm_store_ps(pp + 4, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p1)));
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

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0));
                __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p1));
                __m256 _r2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p2));
                __m256 _r3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p3));
                __m256 _r4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p4));
                __m256 _r5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p5));
                __m256 _r6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p6));
                __m256 _r7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p7));
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
                pp[0] = bfloat16_to_float32(p0[0]);
                pp[1] = bfloat16_to_float32(p1[0]);
                pp[2] = bfloat16_to_float32(p2[0]);
                pp[3] = bfloat16_to_float32(p3[0]);
                pp[4] = bfloat16_to_float32(p4[0]);
                pp[5] = bfloat16_to_float32(p5[0]);
                pp[6] = bfloat16_to_float32(p6[0]);
                pp[7] = bfloat16_to_float32(p7[0]);
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
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
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

            int kk = 0;
#if __AVX__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0));
                __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p1));
                __m256 _r2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p2));
                __m256 _r3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p3));
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
                __m128 _r0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0));
                __m128 _r1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p1));
                __m128 _r2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p2));
                __m128 _r3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p3));
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
                pp[0] = bfloat16_to_float32(p0[0]);
                pp[1] = bfloat16_to_float32(p1[0]);
                pp[2] = bfloat16_to_float32(p2[0]);
                pp[3] = bfloat16_to_float32(p3[0]);
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
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;

            int kk = 0;
#if __SSE2__
#if __AVX__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0));
                __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p1));
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
                __m128 _r0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0));
                __m128 _r1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p1));
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
                pp[0] = bfloat16_to_float32(p0[0]);
                pp[1] = bfloat16_to_float32(p1[0]);
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
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;

            int kk = 0;
#if __SSE2__
#if __AVX__
            for (; kk + 7 < max_kk; kk += 8)
            {
                _mm256_storeu_ps(pp, bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)));
                pp += 8;
                p0 += 8;
            }
#endif // __AVX__
            for (; kk + 3 < max_kk; kk += 4)
            {
                _mm_storeu_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
                pp += 4;
                p0 += 4;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                pp[0] = bfloat16_to_float32(p0[0]);
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_A_tile_bf16s(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
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
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 1));
                __m256i _r2 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 2));
                __m256i _r3 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 3));
                __m256i _r4 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 4));
                __m256i _r5 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 5));
                __m256i _r6 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 6));
                __m256i _r7 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 7));
                __m256i _r8 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 8));
                __m256i _r9 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 9));
                __m256i _ra = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 10));
                __m256i _rb = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 11));
                __m256i _rc = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 12));
                __m256i _rd = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 13));
                __m256i _re = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 14));
                __m256i _rf = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 15));
                transpose16x16_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                _mm512_store_ps(pp, bfloat2float_avx512(_r0));
                _mm512_store_ps(pp + 16 * 1, bfloat2float_avx512(_r1));
                _mm512_store_ps(pp + 16 * 2, bfloat2float_avx512(_r2));
                _mm512_store_ps(pp + 16 * 3, bfloat2float_avx512(_r3));
                _mm512_store_ps(pp + 16 * 4, bfloat2float_avx512(_r4));
                _mm512_store_ps(pp + 16 * 5, bfloat2float_avx512(_r5));
                _mm512_store_ps(pp + 16 * 6, bfloat2float_avx512(_r6));
                _mm512_store_ps(pp + 16 * 7, bfloat2float_avx512(_r7));
                _mm512_store_ps(pp + 16 * 8, bfloat2float_avx512(_r8));
                _mm512_store_ps(pp + 16 * 9, bfloat2float_avx512(_r9));
                _mm512_store_ps(pp + 16 * 10, bfloat2float_avx512(_ra));
                _mm512_store_ps(pp + 16 * 11, bfloat2float_avx512(_rb));
                _mm512_store_ps(pp + 16 * 12, bfloat2float_avx512(_rc));
                _mm512_store_ps(pp + 16 * 13, bfloat2float_avx512(_rd));
                _mm512_store_ps(pp + 16 * 14, bfloat2float_avx512(_re));
                _mm512_store_ps(pp + 16 * 15, bfloat2float_avx512(_rf));
                pp += 256;
                p0 += A_hstep * 16;
            }
        }
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 1));
                __m128i _r2 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 2));
                __m128i _r3 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 3));
                __m128i _r4 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 4));
                __m128i _r5 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 5));
                __m128i _r6 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 6));
                __m128i _r7 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 7));
                __m128i _r8 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 8));
                __m128i _r9 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 9));
                __m128i _ra = _mm_loadu_si128((const __m128i*)(p0 + 8 * 10));
                __m128i _rb = _mm_loadu_si128((const __m128i*)(p0 + 8 * 11));
                __m128i _rc = _mm_loadu_si128((const __m128i*)(p0 + 8 * 12));
                __m128i _rd = _mm_loadu_si128((const __m128i*)(p0 + 8 * 13));
                __m128i _re = _mm_loadu_si128((const __m128i*)(p0 + 8 * 14));
                __m128i _rf = _mm_loadu_si128((const __m128i*)(p0 + 8 * 15));
                transpose8x16_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                _mm512_store_ps(pp, bfloat2float_avx512(_mm256_inserti128_si256(_mm256_castsi128_si256(_r0), _r1, 1)));
                _mm512_store_ps(pp + 16 * 1, bfloat2float_avx512(_mm256_inserti128_si256(_mm256_castsi128_si256(_r2), _r3, 1)));
                _mm512_store_ps(pp + 16 * 2, bfloat2float_avx512(_mm256_inserti128_si256(_mm256_castsi128_si256(_r4), _r5, 1)));
                _mm512_store_ps(pp + 16 * 3, bfloat2float_avx512(_mm256_inserti128_si256(_mm256_castsi128_si256(_r6), _r7, 1)));
                _mm512_store_ps(pp + 16 * 4, bfloat2float_avx512(_mm256_inserti128_si256(_mm256_castsi128_si256(_r8), _r9, 1)));
                _mm512_store_ps(pp + 16 * 5, bfloat2float_avx512(_mm256_inserti128_si256(_mm256_castsi128_si256(_ra), _rb, 1)));
                _mm512_store_ps(pp + 16 * 6, bfloat2float_avx512(_mm256_inserti128_si256(_mm256_castsi128_si256(_rc), _rd, 1)));
                _mm512_store_ps(pp + 16 * 7, bfloat2float_avx512(_mm256_inserti128_si256(_mm256_castsi128_si256(_re), _rf, 1)));
                pp += 128;
                p0 += A_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _a0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _a1 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 1));
                __m128i _a2 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 2));
                __m128i _a3 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 3));
                __m128i _b0 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 4));
                __m128i _b1 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 5));
                __m128i _b2 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 6));
                __m128i _b3 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 7));
                __m128i _c0 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 8));
                __m128i _c1 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 9));
                __m128i _c2 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 10));
                __m128i _c3 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 11));
                __m128i _d0 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 12));
                __m128i _d1 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 13));
                __m128i _d2 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 14));
                __m128i _d3 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 15));
                transpose8x4_epi16(_a0, _a1, _a2, _a3);
                transpose8x4_epi16(_b0, _b1, _b2, _b3);
                transpose8x4_epi16(_c0, _c1, _c2, _c3);
                transpose8x4_epi16(_d0, _d1, _d2, _d3);
                __m256i _col0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi64(_a0, _b0)), _mm_unpacklo_epi64(_c0, _d0), 1);
                __m256i _col1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpackhi_epi64(_a0, _b0)), _mm_unpackhi_epi64(_c0, _d0), 1);
                __m256i _col2 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi64(_a1, _b1)), _mm_unpacklo_epi64(_c1, _d1), 1);
                __m256i _col3 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpackhi_epi64(_a1, _b1)), _mm_unpackhi_epi64(_c1, _d1), 1);
                _mm512_store_ps(pp, bfloat2float_avx512(_col0));
                _mm512_store_ps(pp + 16 * 1, bfloat2float_avx512(_col1));
                _mm512_store_ps(pp + 16 * 2, bfloat2float_avx512(_col2));
                _mm512_store_ps(pp + 16 * 3, bfloat2float_avx512(_col3));
                pp += 64;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                _mm512_store_ps(pp, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)));
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
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)(p0 + 16));
                __m256i _r2 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 2));
                __m256i _r3 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 3));
                __m256i _r4 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 4));
                __m256i _r5 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 5));
                __m256i _r6 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 6));
                __m256i _r7 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 7));
                transpose16x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm512_store_ps(pp, bfloat2float_avx512(_r0));
                _mm512_store_ps(pp + 16 * 1, bfloat2float_avx512(_r1));
                _mm512_store_ps(pp + 16 * 2, bfloat2float_avx512(_r2));
                _mm512_store_ps(pp + 16 * 3, bfloat2float_avx512(_r3));
                _mm512_store_ps(pp + 16 * 4, bfloat2float_avx512(_r4));
                _mm512_store_ps(pp + 16 * 5, bfloat2float_avx512(_r5));
                _mm512_store_ps(pp + 16 * 6, bfloat2float_avx512(_r6));
                _mm512_store_ps(pp + 16 * 7, bfloat2float_avx512(_r7));
                pp += 128;
                p0 += A_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 1));
                __m128i _r2 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 2));
                __m128i _r3 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 3));
                __m128i _r4 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 4));
                __m128i _r5 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 5));
                __m128i _r6 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 6));
                __m128i _r7 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 7));
                transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm256_store_ps(pp, bfloat2float_avx(_r0));
                _mm256_store_ps(pp + 8 * 1, bfloat2float_avx(_r1));
                _mm256_store_ps(pp + 8 * 2, bfloat2float_avx(_r2));
                _mm256_store_ps(pp + 8 * 3, bfloat2float_avx(_r3));
                _mm256_store_ps(pp + 8 * 4, bfloat2float_avx(_r4));
                _mm256_store_ps(pp + 8 * 5, bfloat2float_avx(_r5));
                _mm256_store_ps(pp + 8 * 6, bfloat2float_avx(_r6));
                _mm256_store_ps(pp + 8 * 7, bfloat2float_avx(_r7));
                pp += 64;
                p0 += A_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _a0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _a1 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 1));
                __m128i _a2 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 2));
                __m128i _a3 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 3));
                __m128i _b0 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 4));
                __m128i _b1 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 5));
                __m128i _b2 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 6));
                __m128i _b3 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 7));
                transpose8x4_epi16(_a0, _a1, _a2, _a3);
                transpose8x4_epi16(_b0, _b1, _b2, _b3);
                // _a0 = [col0_rows0-3 | col1_rows0-3], _b0 = [col0_rows4-7 | col1_rows4-7]
                _mm_store_ps(pp, bfloat2float_sse(_a0));
                _mm_store_ps(pp + 4 * 1, bfloat2float_sse(_b0));
                _mm_store_ps(pp + 4 * 2, bfloat2float_sse(_mm_unpackhi_epi64(_a0, _a0)));
                _mm_store_ps(pp + 4 * 3, bfloat2float_sse(_mm_unpackhi_epi64(_b0, _b0)));
                _mm_store_ps(pp + 4 * 4, bfloat2float_sse(_a1));
                _mm_store_ps(pp + 4 * 5, bfloat2float_sse(_b1));
                _mm_store_ps(pp + 4 * 6, bfloat2float_sse(_mm_unpackhi_epi64(_a1, _a1)));
                _mm_store_ps(pp + 4 * 7, bfloat2float_sse(_mm_unpackhi_epi64(_b1, _b1)));
                pp += 32;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                _mm256_store_ps(pp, bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)));
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
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0));
                __m512 _r1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 1)));
                __m512 _r2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 2)));
                __m512 _r3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 3)));
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
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 1));
                __m128i _r2 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 2));
                __m128i _r3 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 3));
                transpose8x4_epi16(_r0, _r1, _r2, _r3);
                _mm256_store_ps(pp, bfloat2float_avx(_r0));
                _mm256_store_ps(pp + 8 * 1, bfloat2float_avx(_r1));
                _mm256_store_ps(pp + 8 * 2, bfloat2float_avx(_r2));
                _mm256_store_ps(pp + 8 * 3, bfloat2float_avx(_r3));
                pp += 32;
                p0 += A_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 1));
                __m128i _r2 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 2));
                __m128i _r3 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 3));
                transpose8x4_epi16(_r0, _r1, _r2, _r3);
                // _r0 = [row0_lo | row1_lo], _r1 = [row2_lo | row3_lo], _r2/_r3 = 0
                _mm_store_ps(pp, bfloat2float_sse(_r0));
                _mm_store_ps(pp + 4 * 1, bfloat2float_sse(_mm_unpackhi_epi64(_r0, _r0)));
                _mm_store_ps(pp + 4 * 2, bfloat2float_sse(_r1));
                _mm_store_ps(pp + 4 * 3, bfloat2float_sse(_mm_unpackhi_epi64(_r1, _r1)));
                pp += 16;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
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
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0));
                __m512 _r1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16)));
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
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0));
                __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8)));
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
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _a = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _b = _mm_loadl_epi64((const __m128i*)(p0 + 4));
                __m128i _tmp0 = _mm_unpacklo_epi16(_a, _b);
                _mm_store_ps(pp, bfloat2float_sse(_tmp0));
                _mm_store_ps(pp + 4, bfloat2float_sse(_mm_unpackhi_epi64(_tmp0, _tmp0)));
                pp += 8;
                p0 += A_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = bfloat16_to_float32(p0[0]);
                pp[1] = bfloat16_to_float32(p0[1]);
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
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                _mm512_store_ps(pp, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)));
                pp += 16;
                p0 += A_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                _mm256_store_ps(pp, bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)));
                pp += 8;
                p0 += A_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                _mm_store_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
                pp += 4;
                p0 += A_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = bfloat16_to_float32(p0[0]);
                pp += 1;
                p0 += A_hstep;
            }
        }
    }
}

static void pack_B_tile_bf16s(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    float* pp = BT;

    int jj = 0;
#if __SSE2__
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 16;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm512_storeu_ps(pp, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)));
                pp += 16;
                p0 += 16;
            }
        }
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 8;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 8) * B_hstep + k * 8;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm256_storeu_ps(pp, bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)));
                _mm256_storeu_ps(pp + 8, bfloat2float_avx(_mm_loadu_si128((const __m128i*)p1)));
                pp += 16;
                p0 += 8;
                p1 += 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k * 4;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 8) * B_hstep + k * 4;
            const unsigned short* p3 = (const unsigned short*)B + (j + jj + 12) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
                _mm_store_ps(pp + 4, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p1)));
                _mm_store_ps(pp + 8, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p2)));
                _mm_store_ps(pp + 12, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p3)));
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
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
            const unsigned short* pc = (const unsigned short*)B + (j + jj + 12) * B_hstep + k;
            const unsigned short* pd = (const unsigned short*)B + (j + jj + 13) * B_hstep + k;
            const unsigned short* pe = (const unsigned short*)B + (j + jj + 14) * B_hstep + k;
            const unsigned short* pf = (const unsigned short*)B + (j + jj + 15) * B_hstep + k;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0));
                __m512 _r1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p1));
                __m512 _r2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p2));
                __m512 _r3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p3));
                __m512 _r4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p4));
                __m512 _r5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p5));
                __m512 _r6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p6));
                __m512 _r7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p7));
                __m512 _r8 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p8));
                __m512 _r9 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p9));
                __m512 _ra = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pa));
                __m512 _rb = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pb));
                __m512 _rc = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pc));
                __m512 _rd = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pd));
                __m512 _re = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pe));
                __m512 _rf = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pf));
                transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                _mm512_storeu_ps(pp, _r0);
                _mm512_storeu_ps(pp + 16, _r1);
                _mm512_storeu_ps(pp + 16 * 2, _r2);
                _mm512_storeu_ps(pp + 16 * 3, _r3);
                _mm512_storeu_ps(pp + 16 * 4, _r4);
                _mm512_storeu_ps(pp + 16 * 5, _r5);
                _mm512_storeu_ps(pp + 16 * 6, _r6);
                _mm512_storeu_ps(pp + 16 * 7, _r7);
                _mm512_storeu_ps(pp + 16 * 8, _r8);
                _mm512_storeu_ps(pp + 16 * 9, _r9);
                _mm512_storeu_ps(pp + 16 * 10, _ra);
                _mm512_storeu_ps(pp + 16 * 11, _rb);
                _mm512_storeu_ps(pp + 16 * 12, _rc);
                _mm512_storeu_ps(pp + 16 * 13, _rd);
                _mm512_storeu_ps(pp + 16 * 14, _re);
                _mm512_storeu_ps(pp + 16 * 15, _rf);
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
                pp[0] = bfloat16_to_float32(p0[0]);
                pp[1] = bfloat16_to_float32(p1[0]);
                pp[2] = bfloat16_to_float32(p2[0]);
                pp[3] = bfloat16_to_float32(p3[0]);
                pp[4] = bfloat16_to_float32(p4[0]);
                pp[5] = bfloat16_to_float32(p5[0]);
                pp[6] = bfloat16_to_float32(p6[0]);
                pp[7] = bfloat16_to_float32(p7[0]);
                pp[8] = bfloat16_to_float32(p8[0]);
                pp[9] = bfloat16_to_float32(p9[0]);
                pp[10] = bfloat16_to_float32(pa[0]);
                pp[11] = bfloat16_to_float32(pb[0]);
                pp[12] = bfloat16_to_float32(pc[0]);
                pp[13] = bfloat16_to_float32(pd[0]);
                pp[14] = bfloat16_to_float32(pe[0]);
                pp[15] = bfloat16_to_float32(pf[0]);
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
#else // __AVX512F__
    for (; jj + 11 < max_jj; jj += 12)
    {
#if __AVX__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) / 8 * 8 * B_hstep + k * 8;
            const unsigned short* p1 = (const unsigned short*)B + ((j + jj) / 8 * 8 + 8) * B_hstep + k * 8;

            if ((j + jj) % 8 == 0)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm256_storeu_ps(pp, bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)));
                    _mm_store_ps(pp + 8, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p1)));
                    pp += 12;
                    p0 += 8;
                    p1 += 8;
                }
            }
            if ((j + jj) % 8 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm_store_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4))));
                    _mm256_storeu_ps(pp + 4, bfloat2float_avx(_mm_loadu_si128((const __m128i*)p1)));
                    pp += 12;
                    p0 += 8;
                    p1 += 8;
                }
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k * 4;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 8) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
                _mm_store_ps(pp + 4, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p1)));
                _mm_store_ps(pp + 8, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p2)));
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

            int kk = 0;
#if __AVX__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0));
                __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p1));
                __m256 _r2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p2));
                __m256 _r3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p3));
                __m256 _r4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p4));
                __m256 _r5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p5));
                __m256 _r6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p6));
                __m256 _r7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p7));
                __m256 _r8 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p8));
                __m256 _r9 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p9));
                __m256 _ra = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pa));
                __m256 _rb = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pb));
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
                __m128 _r0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0));
                __m128 _r1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p1));
                __m128 _r2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p2));
                __m128 _r3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p3));
                __m128 _r4 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p4));
                __m128 _r5 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p5));
                __m128 _r6 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p6));
                __m128 _r7 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p7));
                __m128 _r8 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p8));
                __m128 _r9 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p9));
                __m128 _ra = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pa));
                __m128 _rb = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pb));
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
                pp[0] = bfloat16_to_float32(p0[0]);
                pp[1] = bfloat16_to_float32(p1[0]);
                pp[2] = bfloat16_to_float32(p2[0]);
                pp[3] = bfloat16_to_float32(p3[0]);
                pp[4] = bfloat16_to_float32(p4[0]);
                pp[5] = bfloat16_to_float32(p5[0]);
                pp[6] = bfloat16_to_float32(p6[0]);
                pp[7] = bfloat16_to_float32(p7[0]);
                pp[8] = bfloat16_to_float32(p8[0]);
                pp[9] = bfloat16_to_float32(p9[0]);
                pp[10] = bfloat16_to_float32(pa[0]);
                pp[11] = bfloat16_to_float32(pb[0]);
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
#endif // __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
#if __AVX__
        if (elempack == 8)
        {
#if __AVX512F__
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 8;
#else
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) / 8 * 8 * B_hstep + k * 8;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 8) / 8 * 8 * B_hstep + k * 8;

            if ((j + jj) % 8 == 0)
#endif
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm256_storeu_ps(pp, bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)));
                    pp += 8;
                    p0 += 8;
                }
            }
#if !__AVX512F__
            if ((j + jj) % 8 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm_store_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4))));
                    _mm_store_ps(pp + 4, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p1)));
                    pp += 8;
                    p0 += 8;
                    p1 += 8;
                }
            }
#endif // !__AVX512F__
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
                _mm_store_ps(pp + 4, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p1)));
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

            int kk = 0;
#if __AVX__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0));
                __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p1));
                __m256 _r2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p2));
                __m256 _r3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p3));
                __m256 _r4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p4));
                __m256 _r5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p5));
                __m256 _r6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p6));
                __m256 _r7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p7));
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
                __m128 _r0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0));
                __m128 _r1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p1));
                __m128 _r2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p2));
                __m128 _r3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p3));
                __m128 _r4 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p4));
                __m128 _r5 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p5));
                __m128 _r6 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p6));
                __m128 _r7 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p7));
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
                pp[0] = bfloat16_to_float32(p0[0]);
                pp[1] = bfloat16_to_float32(p1[0]);
                pp[2] = bfloat16_to_float32(p2[0]);
                pp[3] = bfloat16_to_float32(p3[0]);
                pp[4] = bfloat16_to_float32(p4[0]);
                pp[5] = bfloat16_to_float32(p5[0]);
                pp[6] = bfloat16_to_float32(p6[0]);
                pp[7] = bfloat16_to_float32(p7[0]);
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
#if __AVX__ && !__AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) / 8 * 8 * B_hstep + k * 8;

            if ((j + jj) % 8 == 0)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm_store_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
                    pp += 4;
                    p0 += 8;
                }
            }
            if ((j + jj) % 8 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    _mm_store_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4))));
                    pp += 4;
                    p0 += 8;
                }
            }
        }
#endif // __AVX__ && !__AVX512F__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 2) * B_hstep + k;
            const unsigned short* p3 = (const unsigned short*)B + (j + jj + 3) * B_hstep + k;

            int kk = 0;
#if __AVX__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0));
                __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p1));
                __m256 _r2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p2));
                __m256 _r3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p3));
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
                __m128 _r0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0));
                __m128 _r1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p1));
                __m128 _r2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p2));
                __m128 _r3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p3));
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
                pp[0] = bfloat16_to_float32(p0[0]);
                pp[1] = bfloat16_to_float32(p1[0]);
                pp[2] = bfloat16_to_float32(p2[0]);
                pp[3] = bfloat16_to_float32(p3[0]);
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
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;

            int kk = 0;
#if __SSE2__
#if __AVX__
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0));
                __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p1));
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
                __m128 _r0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0));
                __m128 _r1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p1));
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
                pp[0] = bfloat16_to_float32(p0[0]);
                pp[1] = bfloat16_to_float32(p1[0]);
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
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

            int kk = 0;
#if __SSE2__
#if __AVX__
            for (; kk + 7 < max_kk; kk += 8)
            {
                _mm256_storeu_ps(pp, bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)));
                pp += 8;
                p0 += 8;
            }
#endif // __AVX__
            for (; kk + 3 < max_kk; kk += 4)
            {
                _mm_storeu_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
                pp += 4;
                p0 += 4;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                pp[0] = bfloat16_to_float32(p0[0]);
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_B_tile_bf16s(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    float* pp = BT;

    int jj = 0;
#if __SSE2__
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 1));
                __m256i _r2 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 2));
                __m256i _r3 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 3));
                __m256i _r4 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 4));
                __m256i _r5 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 5));
                __m256i _r6 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 6));
                __m256i _r7 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 7));
                __m256i _r8 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 8));
                __m256i _r9 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 9));
                __m256i _ra = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 10));
                __m256i _rb = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 11));
                __m256i _rc = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 12));
                __m256i _rd = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 13));
                __m256i _re = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 14));
                __m256i _rf = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 15));
                transpose16x16_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                _mm512_store_ps(pp, bfloat2float_avx512(_r0));
                _mm512_store_ps(pp + 16 * 1, bfloat2float_avx512(_r1));
                _mm512_store_ps(pp + 16 * 2, bfloat2float_avx512(_r2));
                _mm512_store_ps(pp + 16 * 3, bfloat2float_avx512(_r3));
                _mm512_store_ps(pp + 16 * 4, bfloat2float_avx512(_r4));
                _mm512_store_ps(pp + 16 * 5, bfloat2float_avx512(_r5));
                _mm512_store_ps(pp + 16 * 6, bfloat2float_avx512(_r6));
                _mm512_store_ps(pp + 16 * 7, bfloat2float_avx512(_r7));
                _mm512_store_ps(pp + 16 * 8, bfloat2float_avx512(_r8));
                _mm512_store_ps(pp + 16 * 9, bfloat2float_avx512(_r9));
                _mm512_store_ps(pp + 16 * 10, bfloat2float_avx512(_ra));
                _mm512_store_ps(pp + 16 * 11, bfloat2float_avx512(_rb));
                _mm512_store_ps(pp + 16 * 12, bfloat2float_avx512(_rc));
                _mm512_store_ps(pp + 16 * 13, bfloat2float_avx512(_rd));
                _mm512_store_ps(pp + 16 * 14, bfloat2float_avx512(_re));
                _mm512_store_ps(pp + 16 * 15, bfloat2float_avx512(_rf));
                pp += 256;
                p0 += B_hstep * 16;
            }
        }
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 1));
                __m128i _r2 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 2));
                __m128i _r3 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 3));
                __m128i _r4 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 4));
                __m128i _r5 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 5));
                __m128i _r6 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 6));
                __m128i _r7 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 7));
                __m128i _r8 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 8));
                __m128i _r9 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 9));
                __m128i _ra = _mm_loadu_si128((const __m128i*)(p0 + 8 * 10));
                __m128i _rb = _mm_loadu_si128((const __m128i*)(p0 + 8 * 11));
                __m128i _rc = _mm_loadu_si128((const __m128i*)(p0 + 8 * 12));
                __m128i _rd = _mm_loadu_si128((const __m128i*)(p0 + 8 * 13));
                __m128i _re = _mm_loadu_si128((const __m128i*)(p0 + 8 * 14));
                __m128i _rf = _mm_loadu_si128((const __m128i*)(p0 + 8 * 15));
                transpose8x16_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                _mm256_store_ps(pp, bfloat2float_avx(_r0));
                _mm256_store_ps(pp + 8 * 1, bfloat2float_avx(_r1));
                _mm256_store_ps(pp + 8 * 2, bfloat2float_avx(_r2));
                _mm256_store_ps(pp + 8 * 3, bfloat2float_avx(_r3));
                _mm256_store_ps(pp + 8 * 4, bfloat2float_avx(_r4));
                _mm256_store_ps(pp + 8 * 5, bfloat2float_avx(_r5));
                _mm256_store_ps(pp + 8 * 6, bfloat2float_avx(_r6));
                _mm256_store_ps(pp + 8 * 7, bfloat2float_avx(_r7));
                _mm256_store_ps(pp + 8 * 8, bfloat2float_avx(_r8));
                _mm256_store_ps(pp + 8 * 9, bfloat2float_avx(_r9));
                _mm256_store_ps(pp + 8 * 10, bfloat2float_avx(_ra));
                _mm256_store_ps(pp + 8 * 11, bfloat2float_avx(_rb));
                _mm256_store_ps(pp + 8 * 12, bfloat2float_avx(_rc));
                _mm256_store_ps(pp + 8 * 13, bfloat2float_avx(_rd));
                _mm256_store_ps(pp + 8 * 14, bfloat2float_avx(_re));
                _mm256_store_ps(pp + 8 * 15, bfloat2float_avx(_rf));
                pp += 128;
                p0 += B_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _a0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _a1 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 1));
                __m128i _a2 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 2));
                __m128i _a3 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 3));
                __m128i _b0 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 4));
                __m128i _b1 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 5));
                __m128i _b2 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 6));
                __m128i _b3 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 7));
                __m128i _c0 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 8));
                __m128i _c1 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 9));
                __m128i _c2 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 10));
                __m128i _c3 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 11));
                __m128i _d0 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 12));
                __m128i _d1 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 13));
                __m128i _d2 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 14));
                __m128i _d3 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 15));
                transpose8x4_epi16(_a0, _a1, _a2, _a3);
                transpose8x4_epi16(_b0, _b1, _b2, _b3);
                transpose8x4_epi16(_c0, _c1, _c2, _c3);
                transpose8x4_epi16(_d0, _d1, _d2, _d3);
                __m256i _col0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi64(_a0, _b0)), _mm_unpacklo_epi64(_c0, _d0), 1);
                __m256i _col1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpackhi_epi64(_a0, _b0)), _mm_unpackhi_epi64(_c0, _d0), 1);
                __m256i _col2 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi64(_a1, _b1)), _mm_unpacklo_epi64(_c1, _d1), 1);
                __m256i _col3 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpackhi_epi64(_a1, _b1)), _mm_unpackhi_epi64(_c1, _d1), 1);
                _mm512_store_ps(pp, bfloat2float_avx512(_col0));
                _mm512_store_ps(pp + 16 * 1, bfloat2float_avx512(_col1));
                _mm512_store_ps(pp + 16 * 2, bfloat2float_avx512(_col2));
                _mm512_store_ps(pp + 16 * 3, bfloat2float_avx512(_col3));
                pp += 64;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                _mm512_storeu_ps(pp, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)));
                pp += 16;
                p0 += B_hstep;
            }
        }
    }
#else // __AVX512F__
    for (; jj + 11 < max_jj; jj += 12)
    {
#if __AVX__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0));
                __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 1)));
                __m256 _r2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 2)));
                __m256 _r3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 3)));
                __m256 _r4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 4)));
                __m256 _r5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 5)));
                __m256 _r6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 6)));
                __m256 _r7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 7)));
                __m256 _r8 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 8)));
                __m256 _r9 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 9)));
                __m256 _ra = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 10)));
                __m256 _rb = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 11)));
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
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _r0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0));
                __m128 _r1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 1)));
                __m128 _r2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 2)));
                __m128 _r3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 3)));
                __m128 _r4 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 4)));
                __m128 _r5 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 5)));
                __m128 _r6 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 6)));
                __m128 _r7 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 7)));
                __m128 _r8 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 8)));
                __m128 _r9 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 9)));
                __m128 _ra = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 10)));
                __m128 _rb = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 11)));
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
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
                _mm_store_ps(pp + 4, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4))));
                _mm_store_ps(pp + 8, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 8))));
                pp += 12;
                p0 += B_hstep;
            }
        }
    }
#endif // __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)(p0 + 16));
                __m256i _r2 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 2));
                __m256i _r3 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 3));
                __m256i _r4 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 4));
                __m256i _r5 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 5));
                __m256i _r6 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 6));
                __m256i _r7 = _mm256_loadu_si256((const __m256i*)(p0 + 16 * 7));
                transpose16x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm512_store_ps(pp, bfloat2float_avx512(_r0));
                _mm512_store_ps(pp + 16 * 1, bfloat2float_avx512(_r1));
                _mm512_store_ps(pp + 16 * 2, bfloat2float_avx512(_r2));
                _mm512_store_ps(pp + 16 * 3, bfloat2float_avx512(_r3));
                _mm512_store_ps(pp + 16 * 4, bfloat2float_avx512(_r4));
                _mm512_store_ps(pp + 16 * 5, bfloat2float_avx512(_r5));
                _mm512_store_ps(pp + 16 * 6, bfloat2float_avx512(_r6));
                _mm512_store_ps(pp + 16 * 7, bfloat2float_avx512(_r7));
                pp += 128;
                p0 += B_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 1));
                __m128i _r2 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 2));
                __m128i _r3 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 3));
                __m128i _r4 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 4));
                __m128i _r5 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 5));
                __m128i _r6 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 6));
                __m128i _r7 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 7));
                transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm256_store_ps(pp, bfloat2float_avx(_r0));
                _mm256_store_ps(pp + 8 * 1, bfloat2float_avx(_r1));
                _mm256_store_ps(pp + 8 * 2, bfloat2float_avx(_r2));
                _mm256_store_ps(pp + 8 * 3, bfloat2float_avx(_r3));
                _mm256_store_ps(pp + 8 * 4, bfloat2float_avx(_r4));
                _mm256_store_ps(pp + 8 * 5, bfloat2float_avx(_r5));
                _mm256_store_ps(pp + 8 * 6, bfloat2float_avx(_r6));
                _mm256_store_ps(pp + 8 * 7, bfloat2float_avx(_r7));
                pp += 64;
                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _a0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _a1 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 1));
                __m128i _a2 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 2));
                __m128i _a3 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 3));
                __m128i _b0 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 4));
                __m128i _b1 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 5));
                __m128i _b2 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 6));
                __m128i _b3 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 7));
                transpose8x4_epi16(_a0, _a1, _a2, _a3);
                transpose8x4_epi16(_b0, _b1, _b2, _b3);
                // _a0 = [col0_rows0-3 | col1_rows0-3], _b0 = [col0_rows4-7 | col1_rows4-7]
                _mm_store_ps(pp, bfloat2float_sse(_a0));
                _mm_store_ps(pp + 4 * 1, bfloat2float_sse(_b0));
                _mm_store_ps(pp + 4 * 2, bfloat2float_sse(_mm_unpackhi_epi64(_a0, _a0)));
                _mm_store_ps(pp + 4 * 3, bfloat2float_sse(_mm_unpackhi_epi64(_b0, _b0)));
                _mm_store_ps(pp + 4 * 4, bfloat2float_sse(_a1));
                _mm_store_ps(pp + 4 * 5, bfloat2float_sse(_b1));
                _mm_store_ps(pp + 4 * 6, bfloat2float_sse(_mm_unpackhi_epi64(_a1, _a1)));
                _mm_store_ps(pp + 4 * 7, bfloat2float_sse(_mm_unpackhi_epi64(_b1, _b1)));
                pp += 32;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
                _mm_store_ps(pp + 4, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4))));
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
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0));
                __m512 _r1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 1)));
                __m512 _r2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 2)));
                __m512 _r3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 3)));
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
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 1));
                __m128i _r2 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 2));
                __m128i _r3 = _mm_loadu_si128((const __m128i*)(p0 + 8 * 3));
                transpose8x4_epi16(_r0, _r1, _r2, _r3);
                _mm256_store_ps(pp, bfloat2float_avx(_r0));
                _mm256_store_ps(pp + 8 * 1, bfloat2float_avx(_r1));
                _mm256_store_ps(pp + 8 * 2, bfloat2float_avx(_r2));
                _mm256_store_ps(pp + 8 * 3, bfloat2float_avx(_r3));
                pp += 32;
                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 1));
                __m128i _r2 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 2));
                __m128i _r3 = _mm_loadl_epi64((const __m128i*)(p0 + 4 * 3));
                transpose8x4_epi16(_r0, _r1, _r2, _r3);
                // _r0 = [col0_rows0-3 | col1_rows0-3], _r1 = [col2_rows0-3 | col3_rows0-3]
                _mm_store_ps(pp, bfloat2float_sse(_r0));
                _mm_store_ps(pp + 4 * 1, bfloat2float_sse(_mm_unpackhi_epi64(_r0, _r0)));
                _mm_store_ps(pp + 4 * 2, bfloat2float_sse(_r1));
                _mm_store_ps(pp + 4 * 3, bfloat2float_sse(_mm_unpackhi_epi64(_r1, _r1)));
                pp += 16;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                _mm_store_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
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
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512 _r0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0));
                __m512 _r1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16)));
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
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0));
                __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8)));
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
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _a = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _b = _mm_loadl_epi64((const __m128i*)(p0 + 4));
                __m128i _tmp0 = _mm_unpacklo_epi16(_a, _b);
                _mm_store_ps(pp, bfloat2float_sse(_tmp0));
                _mm_store_ps(pp + 4, bfloat2float_sse(_mm_unpackhi_epi64(_tmp0, _tmp0)));
                pp += 8;
                p0 += B_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = bfloat16_to_float32(p0[0]);
                pp[1] = bfloat16_to_float32(p0[1]);
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
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                _mm512_store_ps(pp, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)p0)));
                pp += 16;
                p0 += B_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                _mm256_store_ps(pp, bfloat2float_avx(_mm_loadu_si128((const __m128i*)p0)));
                pp += 8;
                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                _mm_store_ps(pp, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)p0)));
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = bfloat16_to_float32(p0[0]);
                pp += 1;
                p0 += B_hstep;
            }
        }
    }
}

static void unpack_output_tile_bf16s(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj, int output_transpose)
{
    // topT is fp32 packed tile data
    // top_blob output is bf16
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const float* pp = topT;

    if (output_transpose)
    {
        // transpose_unpack: topT layout is [ii][jj] with ii values contiguous for each jj
        // output to top_blob which is transposed (j is row, i is col)
        int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; ii + 15 < max_ii; ii += 16)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                _mm256_storeu_si256((__m256i*)p0, float2bfloat_avx512(_mm512_loadu_ps(pp)));
                pp += 16;
                p0 += out_hstep;
            }
        }
#endif // __AVX512F__
        for (; ii + 7 < max_ii; ii += 8)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                _mm_storeu_si128((__m128i*)p0, float2bfloat_avx(_mm256_loadu_ps(pp)));
                pp += 8;
                p0 += out_hstep;
            }
        }
#endif // __AVX__
        for (; ii + 3 < max_ii; ii += 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                _mm_storel_epi64((__m128i*)p0, float2bfloat_sse(_mm_loadu_ps(pp), _mm_setzero_ps()));
                pp += 4;
                p0 += out_hstep;
            }
        }
#endif // __SSE2__
        for (; ii + 1 < max_ii; ii += 2)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = float32_to_bfloat16(pp[0]);
                p0[1] = float32_to_bfloat16(pp[1]);
                pp += 2;
                p0 += out_hstep;
            }
        }
        for (; ii < max_ii; ii += 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = float32_to_bfloat16(pp[0]);
                pp += 1;
                p0 += out_hstep;
            }
        }
    }
    else
    {
        // non-transpose unpack: topT layout has ii values contiguous for each jj
        // pp[0..ii-1] = results for jj=0, pp[ii..2*ii-1] = results for jj=1, etc.
        // output: row (i+ii+k), col (j+jj), with out_elempack packing along the ii dimension
        // For bf16 output with out_elempack==1: store pp[k] at row (i+ii+k), col (j+jj)
        int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; ii + 15 < max_ii; ii += 16)
        {
            for (int jj = 0; jj < max_jj; jj += 1)
            {
                for (int k = 0; k < 16; k++)
                {
                    *((unsigned short*)top_blob + (i + ii + k) * out_hstep + (j + jj) * out_elempack) = float32_to_bfloat16(pp[k]);
                }
                pp += 16;
            }
        }
#endif // __AVX512F__
        for (; ii + 7 < max_ii; ii += 8)
        {
            for (int jj = 0; jj < max_jj; jj += 1)
            {
                for (int k = 0; k < 8; k++)
                {
                    *((unsigned short*)top_blob + (i + ii + k) * out_hstep + (j + jj) * out_elempack) = float32_to_bfloat16(pp[k]);
                }
                pp += 8;
            }
        }
#endif // __AVX__
        for (; ii + 3 < max_ii; ii += 4)
        {
            for (int jj = 0; jj < max_jj; jj += 1)
            {
                *((unsigned short*)top_blob + (i + ii + 0) * out_hstep + (j + jj) * out_elempack) = float32_to_bfloat16(pp[0]);
                *((unsigned short*)top_blob + (i + ii + 1) * out_hstep + (j + jj) * out_elempack) = float32_to_bfloat16(pp[1]);
                *((unsigned short*)top_blob + (i + ii + 2) * out_hstep + (j + jj) * out_elempack) = float32_to_bfloat16(pp[2]);
                *((unsigned short*)top_blob + (i + ii + 3) * out_hstep + (j + jj) * out_elempack) = float32_to_bfloat16(pp[3]);
                pp += 4;
            }
        }
#endif // __SSE2__
        for (; ii + 1 < max_ii; ii += 2)
        {
            for (int jj = 0; jj < max_jj; jj += 1)
            {
                *((unsigned short*)top_blob + (i + ii + 0) * out_hstep + (j + jj) * out_elempack) = float32_to_bfloat16(pp[0]);
                *((unsigned short*)top_blob + (i + ii + 1) * out_hstep + (j + jj) * out_elempack) = float32_to_bfloat16(pp[1]);
                pp += 2;
            }
        }
        for (; ii < max_ii; ii += 1)
        {
            for (int jj = 0; jj < max_jj; jj += 1)
            {
                *((unsigned short*)top_blob + (i + ii) * out_hstep + (j + jj) * out_elempack) = float32_to_bfloat16(pp[0]);
                pp += 1;
            }
        }
    }
}

static void get_optimal_tile_mnk_bf16s(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrt((float)l2_cache_size / 3 / sizeof(float));

#if __AVX512F__
    TILE_M = std::max(16, tile_size / 16 * 16);
    TILE_N = std::max(16, tile_size / 16 * 16);
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
            TILE_N = std::max(16, tile_size / 16 * 16);
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
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 15) / 16 * 16);
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
        TILE_N = (constant_TILE_N + 15) / 16 * 16;
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

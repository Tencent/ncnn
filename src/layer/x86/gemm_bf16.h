// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// AVX512BF16 native bf16 gemm kernels
// Data stays in bf16 format through pack/gemm, accumulated to fp32 via _mm512_dpbf16_ps

static void pack_A_tile_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = (unsigned short*)AT;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
#if __AVX512BF16__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 16;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                // p0[0..15] = 16 rows at k, p0[16..31] = 16 rows at k+1
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)(p0 + 16));
                __m512i _r0_32 = _mm512_cvtepu16_epi32(_r0);
                __m512i _r1_32 = _mm512_cvtepu16_epi32(_r1);
                __m512i _paired = _mm512_or_si512(_r0_32, _mm512_slli_epi32(_r1_32, 16));
                _mm512_storeu_si512((__m512i*)pp, _paired);
                pp += 32;
                p0 += 32;
            }
            for (; kk < max_kk; kk++)
            {
                __m256i _r = _mm256_loadu_si256((const __m256i*)p0);
                __m512i _r32 = _mm512_cvtepu16_epi32(_r);
                _mm512_storeu_si512((__m512i*)pp, _r32);
                pp += 32;
                p0 += 16;
            }
        }
        else if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 8;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 8) * A_hstep + k * 8;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                // p0[0..7] = rows 0-7 at k, p0[8..15] = rows 0-7 at k+1
                // p1[0..7] = rows 8-15 at k, p1[8..15] = rows 8-15 at k+1
                __m128i _a0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _a1 = _mm_loadu_si128((const __m128i*)(p0 + 8));
                __m128i _b0 = _mm_loadu_si128((const __m128i*)p1);
                __m128i _b1 = _mm_loadu_si128((const __m128i*)(p1 + 8));
                __m256i _a0_32 = _mm256_cvtepu16_epi32(_a0);
                __m256i _a1_32 = _mm256_cvtepu16_epi32(_a1);
                __m256i _b0_32 = _mm256_cvtepu16_epi32(_b0);
                __m256i _b1_32 = _mm256_cvtepu16_epi32(_b1);
                __m256i _paired_lo = _mm256_or_si256(_a0_32, _mm256_slli_epi32(_a1_32, 16));
                __m256i _paired_hi = _mm256_or_si256(_b0_32, _mm256_slli_epi32(_b1_32, 16));
                _mm256_storeu_si256((__m256i*)pp, _paired_lo);
                _mm256_storeu_si256((__m256i*)(pp + 16), _paired_hi);
                pp += 32;
                p0 += 16;
                p1 += 16;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[1]; pp[3] = 0;
                pp[4] = p0[2]; pp[5] = 0;
                pp[6] = p0[3]; pp[7] = 0;
                pp[8] = p0[4]; pp[9] = 0;
                pp[10] = p0[5]; pp[11] = 0;
                pp[12] = p0[6]; pp[13] = 0;
                pp[14] = p0[7]; pp[15] = 0;
                pp[16] = p1[0]; pp[17] = 0;
                pp[18] = p1[1]; pp[19] = 0;
                pp[20] = p1[2]; pp[21] = 0;
                pp[22] = p1[3]; pp[23] = 0;
                pp[24] = p1[4]; pp[25] = 0;
                pp[26] = p1[5]; pp[27] = 0;
                pp[28] = p1[6]; pp[29] = 0;
                pp[30] = p1[7]; pp[31] = 0;
                pp += 32;
                p0 += 8;
                p1 += 8;
            }
        }
        else if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 4) * A_hstep + k * 4;
            const unsigned short* p2 = (const unsigned short*)A + (i + ii + 8) * A_hstep + k * 4;
            const unsigned short* p3 = (const unsigned short*)A + (i + ii + 12) * A_hstep + k * 4;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _a0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _a1 = _mm_loadu_si128((const __m128i*)p1);
                __m128i _a2 = _mm_loadu_si128((const __m128i*)p2);
                __m128i _a3 = _mm_loadu_si128((const __m128i*)p3);
                __m128i _t0 = _mm_unpacklo_epi16(_a0, _mm_srli_si128(_a0, 8));
                __m128i _t1 = _mm_unpacklo_epi16(_a1, _mm_srli_si128(_a1, 8));
                __m128i _t2 = _mm_unpacklo_epi16(_a2, _mm_srli_si128(_a2, 8));
                __m128i _t3 = _mm_unpacklo_epi16(_a3, _mm_srli_si128(_a3, 8));
                _mm_storeu_si128((__m128i*)pp, _t0);
                _mm_storeu_si128((__m128i*)(pp + 8), _t1);
                _mm_storeu_si128((__m128i*)(pp + 16), _t2);
                _mm_storeu_si128((__m128i*)(pp + 24), _t3);
                pp += 32;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[1]; pp[3] = 0;
                pp[4] = p0[2]; pp[5] = 0;
                pp[6] = p0[3]; pp[7] = 0;
                pp[8] = p1[0]; pp[9] = 0;
                pp[10] = p1[1]; pp[11] = 0;
                pp[12] = p1[2]; pp[13] = 0;
                pp[14] = p1[3]; pp[15] = 0;
                pp[16] = p2[0]; pp[17] = 0;
                pp[18] = p2[1]; pp[19] = 0;
                pp[20] = p2[2]; pp[21] = 0;
                pp[22] = p2[3]; pp[23] = 0;
                pp[24] = p3[0]; pp[25] = 0;
                pp[26] = p3[1]; pp[27] = 0;
                pp[28] = p3[2]; pp[29] = 0;
                pp[30] = p3[3]; pp[31] = 0;
                pp += 32;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
        }
        else // elempack == 1
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[1];
                pp[2] = p1[0]; pp[3] = p1[1];
                pp[4] = p2[0]; pp[5] = p2[1];
                pp[6] = p3[0]; pp[7] = p3[1];
                pp[8] = p4[0]; pp[9] = p4[1];
                pp[10] = p5[0]; pp[11] = p5[1];
                pp[12] = p6[0]; pp[13] = p6[1];
                pp[14] = p7[0]; pp[15] = p7[1];
                pp[16] = p8[0]; pp[17] = p8[1];
                pp[18] = p9[0]; pp[19] = p9[1];
                pp[20] = pa[0]; pp[21] = pa[1];
                pp[22] = pb[0]; pp[23] = pb[1];
                pp[24] = pc[0]; pp[25] = pc[1];
                pp[26] = pd[0]; pp[27] = pd[1];
                pp[28] = pe[0]; pp[29] = pe[1];
                pp[30] = pf[0]; pp[31] = pf[1];
                pp += 32;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
                p4 += 2;
                p5 += 2;
                p6 += 2;
                p7 += 2;
                p8 += 2;
                p9 += 2;
                pa += 2;
                pb += 2;
                pc += 2;
                pd += 2;
                pe += 2;
                pf += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p1[0]; pp[3] = 0;
                pp[4] = p2[0]; pp[5] = 0;
                pp[6] = p3[0]; pp[7] = 0;
                pp[8] = p4[0]; pp[9] = 0;
                pp[10] = p5[0]; pp[11] = 0;
                pp[12] = p6[0]; pp[13] = 0;
                pp[14] = p7[0]; pp[15] = 0;
                pp[16] = p8[0]; pp[17] = 0;
                pp[18] = p9[0]; pp[19] = 0;
                pp[20] = pa[0]; pp[21] = 0;
                pp[22] = pb[0]; pp[23] = 0;
                pp[24] = pc[0]; pp[25] = 0;
                pp[26] = pd[0]; pp[27] = 0;
                pp[28] = pe[0]; pp[29] = 0;
                pp[30] = pf[0]; pp[31] = 0;
                pp += 32;
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
#else  // __AVX512BF16__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 16;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm256_storeu_si256((__m256i*)pp, _mm256_loadu_si256((const __m256i*)p0));
                pp += 16;
                p0 += 16;
            }
        }
        else if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 8;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 8) * A_hstep + k * 8;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                _mm_storeu_si128((__m128i*)(pp + 8), _mm_loadu_si128((const __m128i*)p1));
                pp += 16;
                p0 += 8;
                p1 += 8;
            }
        }
        else if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 4) * A_hstep + k * 4;
            const unsigned short* p2 = (const unsigned short*)A + (i + ii + 8) * A_hstep + k * 4;
            const unsigned short* p3 = (const unsigned short*)A + (i + ii + 12) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
                _mm_storel_epi64((__m128i*)(pp + 4), _mm_loadl_epi64((const __m128i*)p1));
                _mm_storel_epi64((__m128i*)(pp + 8), _mm_loadl_epi64((const __m128i*)p2));
                _mm_storel_epi64((__m128i*)(pp + 12), _mm_loadl_epi64((const __m128i*)p3));
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
        }
        else // elempack == 1
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
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)p1);
                __m256i _r2 = _mm256_loadu_si256((const __m256i*)p2);
                __m256i _r3 = _mm256_loadu_si256((const __m256i*)p3);
                __m256i _r4 = _mm256_loadu_si256((const __m256i*)p4);
                __m256i _r5 = _mm256_loadu_si256((const __m256i*)p5);
                __m256i _r6 = _mm256_loadu_si256((const __m256i*)p6);
                __m256i _r7 = _mm256_loadu_si256((const __m256i*)p7);
                __m256i _r8 = _mm256_loadu_si256((const __m256i*)p8);
                __m256i _r9 = _mm256_loadu_si256((const __m256i*)p9);
                __m256i _ra = _mm256_loadu_si256((const __m256i*)pa);
                __m256i _rb = _mm256_loadu_si256((const __m256i*)pb);
                __m256i _rc = _mm256_loadu_si256((const __m256i*)pc);
                __m256i _rd = _mm256_loadu_si256((const __m256i*)pd);
                __m256i _re = _mm256_loadu_si256((const __m256i*)pe);
                __m256i _rf = _mm256_loadu_si256((const __m256i*)pf);
                transpose16x16_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                _mm256_storeu_si256((__m256i*)pp, _r0);
                _mm256_storeu_si256((__m256i*)(pp + 16), _r1);
                _mm256_storeu_si256((__m256i*)(pp + 32), _r2);
                _mm256_storeu_si256((__m256i*)(pp + 48), _r3);
                _mm256_storeu_si256((__m256i*)(pp + 64), _r4);
                _mm256_storeu_si256((__m256i*)(pp + 80), _r5);
                _mm256_storeu_si256((__m256i*)(pp + 96), _r6);
                _mm256_storeu_si256((__m256i*)(pp + 112), _r7);
                _mm256_storeu_si256((__m256i*)(pp + 128), _r8);
                _mm256_storeu_si256((__m256i*)(pp + 144), _r9);
                _mm256_storeu_si256((__m256i*)(pp + 160), _ra);
                _mm256_storeu_si256((__m256i*)(pp + 176), _rb);
                _mm256_storeu_si256((__m256i*)(pp + 192), _rc);
                _mm256_storeu_si256((__m256i*)(pp + 208), _rd);
                _mm256_storeu_si256((__m256i*)(pp + 224), _re);
                _mm256_storeu_si256((__m256i*)(pp + 240), _rf);
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
#endif // __AVX512BF16__
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
#if __AVX512BF16__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 8;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + 8));
                __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
                __m128i _t1 = _mm_unpackhi_epi16(_r0, _r1);
                _mm_storeu_si128((__m128i*)pp, _t0);
                _mm_storeu_si128((__m128i*)(pp + 8), _t1);
                pp += 16;
                p0 += 16;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[1]; pp[3] = 0;
                pp[4] = p0[2]; pp[5] = 0;
                pp[6] = p0[3]; pp[7] = 0;
                pp[8] = p0[4]; pp[9] = 0;
                pp[10] = p0[5]; pp[11] = 0;
                pp[12] = p0[6]; pp[13] = 0;
                pp[14] = p0[7]; pp[15] = 0;
                pp += 16;
                p0 += 8;
            }
        }
        else if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 4) * A_hstep + k * 4;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _a0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _b0 = _mm_loadu_si128((const __m128i*)p1);
                __m128i _t0 = _mm_unpacklo_epi16(_a0, _mm_srli_si128(_a0, 8));
                __m128i _t1 = _mm_unpacklo_epi16(_b0, _mm_srli_si128(_b0, 8));
                _mm_storeu_si128((__m128i*)pp, _t0);
                _mm_storeu_si128((__m128i*)(pp + 8), _t1);
                pp += 16;
                p0 += 8;
                p1 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[1]; pp[3] = 0;
                pp[4] = p0[2]; pp[5] = 0;
                pp[6] = p0[3]; pp[7] = 0;
                pp[8] = p1[0]; pp[9] = 0;
                pp[10] = p1[1]; pp[11] = 0;
                pp[12] = p1[2]; pp[13] = 0;
                pp[14] = p1[3]; pp[15] = 0;
                pp += 16;
                p0 += 4;
                p1 += 4;
            }
        }
        else // elempack == 1
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[1];
                pp[2] = p1[0]; pp[3] = p1[1];
                pp[4] = p2[0]; pp[5] = p2[1];
                pp[6] = p3[0]; pp[7] = p3[1];
                pp[8] = p4[0]; pp[9] = p4[1];
                pp[10] = p5[0]; pp[11] = p5[1];
                pp[12] = p6[0]; pp[13] = p6[1];
                pp[14] = p7[0]; pp[15] = p7[1];
                pp += 16;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
                p4 += 2;
                p5 += 2;
                p6 += 2;
                p7 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p1[0]; pp[3] = 0;
                pp[4] = p2[0]; pp[5] = 0;
                pp[6] = p3[0]; pp[7] = 0;
                pp[8] = p4[0]; pp[9] = 0;
                pp[10] = p5[0]; pp[11] = 0;
                pp[12] = p6[0]; pp[13] = 0;
                pp[14] = p7[0]; pp[15] = 0;
                pp += 16;
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
#else  // __AVX512BF16__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 8;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                pp += 8;
                p0 += 8;
            }
        }
        else if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 4) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
                _mm_storel_epi64((__m128i*)(pp + 4), _mm_loadl_epi64((const __m128i*)p1));
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
        }
        else // elempack == 1
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
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)p1);
                __m128i _r2 = _mm_loadu_si128((const __m128i*)p2);
                __m128i _r3 = _mm_loadu_si128((const __m128i*)p3);
                __m128i _r4 = _mm_loadu_si128((const __m128i*)p4);
                __m128i _r5 = _mm_loadu_si128((const __m128i*)p5);
                __m128i _r6 = _mm_loadu_si128((const __m128i*)p6);
                __m128i _r7 = _mm_loadu_si128((const __m128i*)p7);
                transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm_storeu_si128((__m128i*)pp, _r0);
                _mm_storeu_si128((__m128i*)(pp + 8), _r1);
                _mm_storeu_si128((__m128i*)(pp + 16), _r2);
                _mm_storeu_si128((__m128i*)(pp + 24), _r3);
                _mm_storeu_si128((__m128i*)(pp + 32), _r4);
                _mm_storeu_si128((__m128i*)(pp + 40), _r5);
                _mm_storeu_si128((__m128i*)(pp + 48), _r6);
                _mm_storeu_si128((__m128i*)(pp + 56), _r7);
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
#endif // __AVX512BF16__
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
#if __AVX512BF16__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _t0 = _mm_unpacklo_epi16(_r0, _mm_srli_si128(_r0, 8));
                __m128i _t1 = _mm_unpackhi_epi16(_mm_slli_si128(_r0, 8), _r0);
                (void)_t1;
                _mm_storeu_si128((__m128i*)pp, _t0);
                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[1]; pp[3] = 0;
                pp[4] = p0[2]; pp[5] = 0;
                pp[6] = p0[3]; pp[7] = 0;
                pp += 8;
                p0 += 4;
            }
        }
        else // elempack == 1
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;
            const unsigned short* p2 = (const unsigned short*)A + (i + ii + 2) * A_hstep + k;
            const unsigned short* p3 = (const unsigned short*)A + (i + ii + 3) * A_hstep + k;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[1];
                pp[2] = p1[0]; pp[3] = p1[1];
                pp[4] = p2[0]; pp[5] = p2[1];
                pp[6] = p3[0]; pp[7] = p3[1];
                pp += 8;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p1[0]; pp[3] = 0;
                pp[4] = p2[0]; pp[5] = 0;
                pp[6] = p3[0]; pp[7] = 0;
                pp += 8;
                p0++;
                p1++;
                p2++;
                p3++;
            }
        }
#else  // __AVX512BF16__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
                pp += 4;
                p0 += 4;
            }
        }
        else // elempack == 1
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;
            const unsigned short* p2 = (const unsigned short*)A + (i + ii + 2) * A_hstep + k;
            const unsigned short* p3 = (const unsigned short*)A + (i + ii + 3) * A_hstep + k;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)p1);
                __m128i _r2 = _mm_loadl_epi64((const __m128i*)p2);
                __m128i _r3 = _mm_loadl_epi64((const __m128i*)p3);
                __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
                __m128i _t1 = _mm_unpacklo_epi16(_r2, _r3);
                _r0 = _mm_unpacklo_epi32(_t0, _t1);
                _r1 = _mm_unpackhi_epi32(_t0, _t1);
                _mm_storeu_si128((__m128i*)pp, _r0);
                _mm_storeu_si128((__m128i*)(pp + 8), _r1);
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
#endif // __AVX512BF16__
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __AVX512BF16__
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
        const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0]; pp[1] = p0[1];
            pp[2] = p1[0]; pp[3] = p1[1];
            pp += 4;
            p0 += 2;
            p1 += 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0]; pp[1] = 0;
            pp[2] = p1[0]; pp[3] = 0;
            pp += 4;
            p0++;
            p1++;
        }
#else  // __AVX512BF16__
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
        const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
#endif // __AVX512BF16__
    }
    for (; ii < max_ii; ii += 1)
    {
#if __AVX512BF16__
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;

        int kk = 0;
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp += 2;
            p0 += 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = 0;
            pp += 2;
            p0++;
        }
#else  // __AVX512BF16__
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
#endif // __AVX512BF16__
    }
}

static void transpose_pack_A_tile_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = (unsigned short*)AT;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
#if __AVX512BF16__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                for (int p = 0; p < 8; p++)
                {
                    for (int m = 0; m < 16; m++)
                    {
                        pp[m * 2] = p0[m * 16 + p * 2];
                        pp[m * 2 + 1] = p0[m * 16 + p * 2 + 1];
                    }
                    pp += 32;
                }
                p0 += A_hstep * 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 16;
                for (int m = 0; m < 16; m++)
                {
                    pp[m * 2] = p0[m * 16 + kk_offset];
                    pp[m * 2 + 1] = p0[m * 16 + kk_offset + 1];
                }
                pp += 32;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 16;
                for (int m = 0; m < 16; m++)
                {
                    pp[m * 2] = p0[m * 16 + kk_offset];
                    pp[m * 2 + 1] = 0;
                }
                pp += 32;
            }
        }
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                for (int p = 0; p < 4; p++)
                {
                    for (int m = 0; m < 8; m++)
                    {
                        pp[m * 2] = p0[m * 8 + p * 2];
                        pp[m * 2 + 1] = p0[m * 8 + p * 2 + 1];
                    }
                    const unsigned short* p1 = p0 + 8 * 8;
                    for (int m = 0; m < 8; m++)
                    {
                        pp[16 + m * 2] = p1[m * 8 + p * 2];
                        pp[16 + m * 2 + 1] = p1[m * 8 + p * 2 + 1];
                    }
                    pp += 32;
                }
                p0 += A_hstep * 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 8;
                for (int m = 0; m < 8; m++)
                {
                    pp[m * 2] = p0[m * 8 + kk_offset];
                    pp[m * 2 + 1] = p0[m * 8 + kk_offset + 1];
                }
                const unsigned short* p1 = p0 + 8 * 8;
                for (int m = 0; m < 8; m++)
                {
                    pp[16 + m * 2] = p1[m * 8 + kk_offset];
                    pp[16 + m * 2 + 1] = p1[m * 8 + kk_offset + 1];
                }
                pp += 32;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 8;
                for (int m = 0; m < 8; m++)
                {
                    pp[m * 2] = p0[m * 8 + kk_offset];
                    pp[m * 2 + 1] = 0;
                }
                const unsigned short* p1 = p0 + 8 * 8;
                for (int m = 0; m < 8; m++)
                {
                    pp[16 + m * 2] = p1[m * 8 + kk_offset];
                    pp[16 + m * 2 + 1] = 0;
                }
                pp += 32;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                for (int p = 0; p < 2; p++)
                {
                    for (int m = 0; m < 16; m++)
                    {
                        pp[m * 2] = p0[m * 4 + p * 2];
                        pp[m * 2 + 1] = p0[m * 4 + p * 2 + 1];
                    }
                    pp += 32;
                }
                p0 += A_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 4;
                for (int m = 0; m < 16; m++)
                {
                    pp[m * 2] = p0[m * 4 + kk_offset];
                    pp[m * 2 + 1] = p0[m * 4 + kk_offset + 1];
                }
                pp += 32;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 4;
                for (int m = 0; m < 16; m++)
                {
                    pp[m * 2] = p0[m * 4 + kk_offset];
                    pp[m * 2 + 1] = 0;
                }
                pp += 32;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[A_hstep];
                pp[2] = p0[1]; pp[3] = p0[A_hstep + 1];
                pp[4] = p0[2]; pp[5] = p0[A_hstep + 2];
                pp[6] = p0[3]; pp[7] = p0[A_hstep + 3];
                pp[8] = p0[4]; pp[9] = p0[A_hstep + 4];
                pp[10] = p0[5]; pp[11] = p0[A_hstep + 5];
                pp[12] = p0[6]; pp[13] = p0[A_hstep + 6];
                pp[14] = p0[7]; pp[15] = p0[A_hstep + 7];
                pp[16] = p0[8]; pp[17] = p0[A_hstep + 8];
                pp[18] = p0[9]; pp[19] = p0[A_hstep + 9];
                pp[20] = p0[10]; pp[21] = p0[A_hstep + 10];
                pp[22] = p0[11]; pp[23] = p0[A_hstep + 11];
                pp[24] = p0[12]; pp[25] = p0[A_hstep + 12];
                pp[26] = p0[13]; pp[27] = p0[A_hstep + 13];
                pp[28] = p0[14]; pp[29] = p0[A_hstep + 14];
                pp[30] = p0[15]; pp[31] = p0[A_hstep + 15];
                pp += 32;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[1]; pp[3] = 0;
                pp[4] = p0[2]; pp[5] = 0;
                pp[6] = p0[3]; pp[7] = 0;
                pp[8] = p0[4]; pp[9] = 0;
                pp[10] = p0[5]; pp[11] = 0;
                pp[12] = p0[6]; pp[13] = 0;
                pp[14] = p0[7]; pp[15] = 0;
                pp[16] = p0[8]; pp[17] = 0;
                pp[18] = p0[9]; pp[19] = 0;
                pp[20] = p0[10]; pp[21] = 0;
                pp[22] = p0[11]; pp[23] = 0;
                pp[24] = p0[12]; pp[25] = 0;
                pp[26] = p0[13]; pp[27] = 0;
                pp[28] = p0[14]; pp[29] = 0;
                pp[30] = p0[15]; pp[31] = 0;
                pp += 32;
                p0 += A_hstep;
            }
        }
#else  // __AVX512BF16__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)(p0 + 16));
                __m256i _r2 = _mm256_loadu_si256((const __m256i*)(p0 + 32));
                __m256i _r3 = _mm256_loadu_si256((const __m256i*)(p0 + 48));
                __m256i _r4 = _mm256_loadu_si256((const __m256i*)(p0 + 64));
                __m256i _r5 = _mm256_loadu_si256((const __m256i*)(p0 + 80));
                __m256i _r6 = _mm256_loadu_si256((const __m256i*)(p0 + 96));
                __m256i _r7 = _mm256_loadu_si256((const __m256i*)(p0 + 112));
                __m256i _r8 = _mm256_loadu_si256((const __m256i*)(p0 + 128));
                __m256i _r9 = _mm256_loadu_si256((const __m256i*)(p0 + 144));
                __m256i _ra = _mm256_loadu_si256((const __m256i*)(p0 + 160));
                __m256i _rb = _mm256_loadu_si256((const __m256i*)(p0 + 176));
                __m256i _rc = _mm256_loadu_si256((const __m256i*)(p0 + 192));
                __m256i _rd = _mm256_loadu_si256((const __m256i*)(p0 + 208));
                __m256i _re = _mm256_loadu_si256((const __m256i*)(p0 + 224));
                __m256i _rf = _mm256_loadu_si256((const __m256i*)(p0 + 240));
                transpose16x16_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                _mm256_storeu_si256((__m256i*)pp, _r0);
                _mm256_storeu_si256((__m256i*)(pp + 16), _r1);
                _mm256_storeu_si256((__m256i*)(pp + 32), _r2);
                _mm256_storeu_si256((__m256i*)(pp + 48), _r3);
                _mm256_storeu_si256((__m256i*)(pp + 64), _r4);
                _mm256_storeu_si256((__m256i*)(pp + 80), _r5);
                _mm256_storeu_si256((__m256i*)(pp + 96), _r6);
                _mm256_storeu_si256((__m256i*)(pp + 112), _r7);
                _mm256_storeu_si256((__m256i*)(pp + 128), _r8);
                _mm256_storeu_si256((__m256i*)(pp + 144), _r9);
                _mm256_storeu_si256((__m256i*)(pp + 160), _ra);
                _mm256_storeu_si256((__m256i*)(pp + 176), _rb);
                _mm256_storeu_si256((__m256i*)(pp + 192), _rc);
                _mm256_storeu_si256((__m256i*)(pp + 208), _rd);
                _mm256_storeu_si256((__m256i*)(pp + 224), _re);
                _mm256_storeu_si256((__m256i*)(pp + 240), _rf);
                pp += 256;
                p0 += A_hstep * 16;
            }
            for (; kk < max_kk; kk++)
            {
                _mm256_storeu_si256((__m256i*)pp, _mm256_loadu_si256((const __m256i*)p0));
                pp += 16;
                p0 += A_hstep;
            }
        }
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + 8));
                __m128i _r2 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep));
                __m128i _r3 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep + 8));
                __m128i _r4 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 2));
                __m128i _r5 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 2 + 8));
                __m128i _r6 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 3));
                __m128i _r7 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 3 + 8));
                __m128i _r8 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 4));
                __m128i _r9 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 4 + 8));
                __m128i _ra = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 5));
                __m128i _rb = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 5 + 8));
                __m128i _rc = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 6));
                __m128i _rd = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 6 + 8));
                __m128i _re = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 7));
                __m128i _rf = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 7 + 8));
                transpose8x16_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                _mm256_storeu_si256((__m256i*)pp, _mm256_inserti128_si256(_mm256_castsi128_si256(_r0), _r1, 1));
                _mm256_storeu_si256((__m256i*)(pp + 16), _mm256_inserti128_si256(_mm256_castsi128_si256(_r2), _r3, 1));
                _mm256_storeu_si256((__m256i*)(pp + 32), _mm256_inserti128_si256(_mm256_castsi128_si256(_r4), _r5, 1));
                _mm256_storeu_si256((__m256i*)(pp + 48), _mm256_inserti128_si256(_mm256_castsi128_si256(_r6), _r7, 1));
                _mm256_storeu_si256((__m256i*)(pp + 64), _mm256_inserti128_si256(_mm256_castsi128_si256(_r8), _r9, 1));
                _mm256_storeu_si256((__m256i*)(pp + 80), _mm256_inserti128_si256(_mm256_castsi128_si256(_ra), _rb, 1));
                _mm256_storeu_si256((__m256i*)(pp + 96), _mm256_inserti128_si256(_mm256_castsi128_si256(_rc), _rd, 1));
                _mm256_storeu_si256((__m256i*)(pp + 112), _mm256_inserti128_si256(_mm256_castsi128_si256(_re), _rf, 1));
                pp += 128;
                p0 += A_hstep * 8;
            }
            for (; kk < max_kk; kk++)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                _mm_storeu_si128((__m128i*)(pp + 8), _mm_loadu_si128((const __m128i*)(p0 + 8)));
                pp += 16;
                p0 += A_hstep;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
                _mm_storel_epi64((__m128i*)(pp + 4), _mm_loadl_epi64((const __m128i*)(p0 + 4)));
                _mm_storel_epi64((__m128i*)(pp + 8), _mm_loadl_epi64((const __m128i*)(p0 + 8)));
                _mm_storel_epi64((__m128i*)(pp + 12), _mm_loadl_epi64((const __m128i*)(p0 + 12)));
                pp += 16;
                p0 += A_hstep;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                _mm256_storeu_si256((__m256i*)pp, _mm256_loadu_si256((const __m256i*)p0));
                pp += 16;
                p0 += A_hstep;
            }
        }
#endif // __AVX512BF16__
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
#if __AVX512BF16__
#if __AVX512F__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                for (int p = 0; p < 8; p++)
                {
                    for (int m = 0; m < 8; m++)
                    {
                        pp[m * 2] = p0[m * 16 + p * 2];
                        pp[m * 2 + 1] = p0[m * 16 + p * 2 + 1];
                    }
                    pp += 16;
                }
                p0 += A_hstep * 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 16;
                for (int m = 0; m < 8; m++)
                {
                    pp[m * 2] = p0[m * 16 + kk_offset];
                    pp[m * 2 + 1] = p0[m * 16 + kk_offset + 1];
                }
                pp += 16;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 16;
                for (int m = 0; m < 8; m++)
                {
                    pp[m * 2] = p0[m * 16 + kk_offset];
                    pp[m * 2 + 1] = 0;
                }
                pp += 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                for (int p = 0; p < 4; p++)
                {
                    for (int m = 0; m < 8; m++)
                    {
                        pp[m * 2] = p0[m * 8 + p * 2];
                        pp[m * 2 + 1] = p0[m * 8 + p * 2 + 1];
                    }
                    pp += 16;
                }
                p0 += A_hstep * 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 8;
                for (int m = 0; m < 8; m++)
                {
                    pp[m * 2] = p0[m * 8 + kk_offset];
                    pp[m * 2 + 1] = p0[m * 8 + kk_offset + 1];
                }
                pp += 16;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 8;
                for (int m = 0; m < 8; m++)
                {
                    pp[m * 2] = p0[m * 8 + kk_offset];
                    pp[m * 2 + 1] = 0;
                }
                pp += 16;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                for (int p = 0; p < 2; p++)
                {
                    for (int m = 0; m < 8; m++)
                    {
                        pp[m * 2] = p0[m * 4 + p * 2];
                        pp[m * 2 + 1] = p0[m * 4 + p * 2 + 1];
                    }
                    pp += 16;
                }
                p0 += A_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 4;
                for (int m = 0; m < 8; m++)
                {
                    pp[m * 2] = p0[m * 4 + kk_offset];
                    pp[m * 2 + 1] = p0[m * 4 + kk_offset + 1];
                }
                pp += 16;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 4;
                for (int m = 0; m < 8; m++)
                {
                    pp[m * 2] = p0[m * 4 + kk_offset];
                    pp[m * 2 + 1] = 0;
                }
                pp += 16;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[A_hstep];
                pp[2] = p0[1]; pp[3] = p0[A_hstep + 1];
                pp[4] = p0[2]; pp[5] = p0[A_hstep + 2];
                pp[6] = p0[3]; pp[7] = p0[A_hstep + 3];
                pp[8] = p0[4]; pp[9] = p0[A_hstep + 4];
                pp[10] = p0[5]; pp[11] = p0[A_hstep + 5];
                pp[12] = p0[6]; pp[13] = p0[A_hstep + 6];
                pp[14] = p0[7]; pp[15] = p0[A_hstep + 7];
                pp += 16;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[1]; pp[3] = 0;
                pp[4] = p0[2]; pp[5] = 0;
                pp[6] = p0[3]; pp[7] = 0;
                pp[8] = p0[4]; pp[9] = 0;
                pp[10] = p0[5]; pp[11] = 0;
                pp[12] = p0[6]; pp[13] = 0;
                pp[14] = p0[7]; pp[15] = 0;
                pp += 16;
                p0 += A_hstep;
            }
        }
#else  // __AVX512BF16__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep));
                __m128i _r2 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 2));
                __m128i _r3 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 3));
                __m128i _r4 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 4));
                __m128i _r5 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 5));
                __m128i _r6 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 6));
                __m128i _r7 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep * 7));
                transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm_storeu_si128((__m128i*)pp, _r0);
                _mm_storeu_si128((__m128i*)(pp + 8), _r1);
                _mm_storeu_si128((__m128i*)(pp + 16), _r2);
                _mm_storeu_si128((__m128i*)(pp + 24), _r3);
                _mm_storeu_si128((__m128i*)(pp + 32), _r4);
                _mm_storeu_si128((__m128i*)(pp + 40), _r5);
                _mm_storeu_si128((__m128i*)(pp + 48), _r6);
                _mm_storeu_si128((__m128i*)(pp + 56), _r7);
                pp += 64;
                p0 += A_hstep * 8;
            }
            for (; kk < max_kk; kk++)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                pp += 8;
                p0 += A_hstep;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // 8 columns of 4 kk values each. Extract per-kk rows of 8.
                // p0[0..3] = kk0..kk3 for ii+0
                // p0[4..7] = kk0..kk3 for ii+1
                // ...
                // p0[28..31] = kk0..kk3 for ii+7
                for (int m = 0; m < 4; m++)
                {
                    pp[0] = p0[m];
                    pp[1] = p0[4 + m];
                    pp[2] = p0[8 + m];
                    pp[3] = p0[12 + m];
                    pp[4] = p0[16 + m];
                    pp[5] = p0[20 + m];
                    pp[6] = p0[24 + m];
                    pp[7] = p0[28 + m];
                    pp += 8;
                }
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
#endif // __AVX512BF16__
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
#if __AVX512BF16__
#if __AVX512F__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                for (int p = 0; p < 8; p++)
                {
                    for (int m = 0; m < 4; m++)
                    {
                        pp[m * 2] = p0[m * 16 + p * 2];
                        pp[m * 2 + 1] = p0[m * 16 + p * 2 + 1];
                    }
                    pp += 8;
                }
                p0 += A_hstep * 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 16;
                for (int m = 0; m < 4; m++)
                {
                    pp[m * 2] = p0[m * 16 + kk_offset];
                    pp[m * 2 + 1] = p0[m * 16 + kk_offset + 1];
                }
                pp += 8;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 16;
                for (int m = 0; m < 4; m++)
                {
                    pp[m * 2] = p0[m * 16 + kk_offset];
                    pp[m * 2 + 1] = 0;
                }
                pp += 8;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                for (int p = 0; p < 4; p++)
                {
                    for (int m = 0; m < 4; m++)
                    {
                        pp[m * 2] = p0[m * 8 + p * 2];
                        pp[m * 2 + 1] = p0[m * 8 + p * 2 + 1];
                    }
                    pp += 8;
                }
                p0 += A_hstep * 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 8;
                for (int m = 0; m < 4; m++)
                {
                    pp[m * 2] = p0[m * 8 + kk_offset];
                    pp[m * 2 + 1] = p0[m * 8 + kk_offset + 1];
                }
                pp += 8;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 8;
                for (int m = 0; m < 4; m++)
                {
                    pp[m * 2] = p0[m * 8 + kk_offset];
                    pp[m * 2 + 1] = 0;
                }
                pp += 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // elempack=4: p0[0..3] = k0..k3 for m_i, p0[4..7] = k0..k3 for m_i+1, etc.
                pp[0] = p0[0]; pp[1] = p0[1];    // k0|k1 for m_i
                pp[2] = p0[4]; pp[3] = p0[5];    // k0|k1 for m_i+1
                pp[4] = p0[8]; pp[5] = p0[9];    // k0|k1 for m_i+2
                pp[6] = p0[12]; pp[7] = p0[13];  // k0|k1 for m_i+3
                pp[8] = p0[2]; pp[9] = p0[3];    // k2|k3 for m_i
                pp[10] = p0[6]; pp[11] = p0[7];  // k2|k3 for m_i+1
                pp[12] = p0[10]; pp[13] = p0[11]; // k2|k3 for m_i+2
                pp[14] = p0[14]; pp[15] = p0[15]; // k2|k3 for m_i+3
                pp += 16;
                p0 += A_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[1];
                pp[2] = p0[4]; pp[3] = p0[5];
                pp[4] = p0[8]; pp[5] = p0[9];
                pp[6] = p0[12]; pp[7] = p0[13];
                pp += 8;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[4]; pp[3] = 0;
                pp[4] = p0[8]; pp[5] = 0;
                pp[6] = p0[12]; pp[7] = 0;
                pp += 8;
                p0++;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[A_hstep];
                pp[2] = p0[1]; pp[3] = p0[A_hstep + 1];
                pp[4] = p0[2]; pp[5] = p0[A_hstep + 2];
                pp[6] = p0[3]; pp[7] = p0[A_hstep + 3];
                pp += 8;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[1]; pp[3] = 0;
                pp[4] = p0[2]; pp[5] = 0;
                pp[6] = p0[3]; pp[7] = 0;
                pp += 8;
                p0 += A_hstep;
            }
        }
#else  // __AVX512BF16__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)(p0 + A_hstep));
                __m128i _r2 = _mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 2));
                __m128i _r3 = _mm_loadl_epi64((const __m128i*)(p0 + A_hstep * 3));
                __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
                __m128i _t1 = _mm_unpacklo_epi16(_r2, _r3);
                _r0 = _mm_unpacklo_epi32(_t0, _t1);
                _r1 = _mm_unpackhi_epi32(_t0, _t1);
                _mm_storeu_si128((__m128i*)pp, _r0);
                _mm_storeu_si128((__m128i*)(pp + 8), _r1);
                pp += 16;
                p0 += A_hstep * 4;
            }
            for (; kk < max_kk; kk++)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
                pp += 4;
                p0 += A_hstep;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

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
#endif // __AVX512BF16__
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __AVX512BF16__
#if __AVX512F__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                for (int m = 0; m < 8; m++)
                {
                    pp[0] = p0[m * 2];
                    pp[1] = p0[m * 2 + 1];
                    pp[2] = p0[16 + m * 2];
                    pp[3] = p0[16 + m * 2 + 1];
                    pp += 4;
                }
                p0 += A_hstep * 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 16;
                pp[0] = p0[kk_offset];
                pp[1] = p0[kk_offset + 1];
                pp[2] = p0[16 + kk_offset];
                pp[3] = p0[16 + kk_offset + 1];
                pp += 4;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 16;
                pp[0] = p0[kk_offset];
                pp[1] = 0;
                pp[2] = p0[16 + kk_offset];
                pp[3] = 0;
                pp += 4;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                for (int m = 0; m < 4; m++)
                {
                    pp[0] = p0[m * 2];
                    pp[1] = p0[m * 2 + 1];
                    pp[2] = p0[8 + m * 2];
                    pp[3] = p0[8 + m * 2 + 1];
                    pp += 4;
                }
                p0 += A_hstep * 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 8;
                pp[0] = p0[kk_offset];
                pp[1] = p0[kk_offset + 1];
                pp[2] = p0[8 + kk_offset];
                pp[3] = p0[8 + kk_offset + 1];
                pp += 4;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 8;
                pp[0] = p0[kk_offset];
                pp[1] = 0;
                pp[2] = p0[8 + kk_offset];
                pp[3] = 0;
                pp += 4;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0]; pp[1] = p0[1];
                pp[2] = p0[4]; pp[3] = p0[5];
                pp[4] = p0[2]; pp[5] = p0[3];
                pp[6] = p0[6]; pp[7] = p0[7];
                pp += 8;
                p0 += A_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[1];
                pp[2] = p0[4]; pp[3] = p0[5];
                pp += 4;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[4]; pp[3] = 0;
                pp += 4;
                p0++;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[A_hstep];
                pp[2] = p0[1]; pp[3] = p0[A_hstep + 1];
                pp += 4;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[1]; pp[3] = 0;
                pp += 4;
                p0 += A_hstep;
            }
        }
#else  // __AVX512BF16__
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 16;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[16];
                pp += 2;
                p0 += A_hstep;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[8];
                pp += 2;
                p0 += A_hstep;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp += 2;
                p0 += A_hstep;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += A_hstep;
            }
        }
#endif // __AVX512BF16__
    }
    for (; ii < max_ii; ii += 1)
    {
#if __AVX512BF16__
#if __AVX512F__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                for (int m = 0; m < 8; m++)
                {
                    pp[0] = p0[m * 2];
                    pp[1] = p0[m * 2 + 1];
                    pp += 2;
                }
                p0 += A_hstep * 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 16;
                pp[0] = p0[kk_offset];
                pp[1] = p0[kk_offset + 1];
                pp += 2;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 16;
                pp[0] = p0[kk_offset];
                pp[1] = 0;
                pp += 2;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                for (int m = 0; m < 4; m++)
                {
                    pp[0] = p0[m * 2];
                    pp[1] = p0[m * 2 + 1];
                    pp += 2;
                }
                p0 += A_hstep * 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 8;
                pp[0] = p0[kk_offset];
                pp[1] = p0[kk_offset + 1];
                pp += 2;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 8;
                pp[0] = p0[kk_offset];
                pp[1] = 0;
                pp += 2;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0]; pp[1] = p0[1];
                pp[2] = p0[2]; pp[3] = p0[3];
                pp += 4;
                p0 += A_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[1];
                pp += 2;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp += 2;
                p0++;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[A_hstep];
                pp += 2;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = 0;
                pp += 2;
                p0 += A_hstep;
            }
        }
#else  // __AVX512BF16__
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                _mm256_storeu_si256((__m256i*)pp, _mm256_loadu_si256((const __m256i*)p0));
                pp += 16;
                p0 += A_hstep * 16;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += A_hstep;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                pp += 8;
                p0 += A_hstep * 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += A_hstep;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
                pp += 4;
                p0 += A_hstep * 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += A_hstep;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += A_hstep;
            }
        }
#endif // __AVX512BF16__
    }
}

static void pack_B_tile_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    // pack_B_tile_bf16 is the same as pack_A_tile_bf16 but for B
    // B is transposed relative to A, so pack_B is like transpose_pack_A
    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    unsigned short* pp = (unsigned short*)BT;

    int jj = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
#if __AVX512BF16__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 16;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)(p0 + 16));
                __m512i _r0_32 = _mm512_cvtepu16_epi32(_r0);
                __m512i _r1_32 = _mm512_cvtepu16_epi32(_r1);
                __m512i _paired = _mm512_or_si512(_r0_32, _mm512_slli_epi32(_r1_32, 16));
                _mm512_storeu_si512((__m512i*)pp, _paired);
                pp += 32;
                p0 += 32;
            }
            for (; kk < max_kk; kk++)
            {
                __m256i _r = _mm256_loadu_si256((const __m256i*)p0);
                __m512i _r32 = _mm512_cvtepu16_epi32(_r);
                _mm512_storeu_si512((__m512i*)pp, _r32);
                pp += 32;
                p0 += 16;
            }
        }
        else if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 8;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 8) * B_hstep + k * 8;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                // p0[0..7] = rows 0-7 at k, p0[8..15] = rows 0-7 at k+1
                // p1[0..7] = rows 8-15 at k, p1[8..15] = rows 8-15 at k+1
                __m128i _a0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _a1 = _mm_loadu_si128((const __m128i*)(p0 + 8));
                __m128i _b0 = _mm_loadu_si128((const __m128i*)p1);
                __m128i _b1 = _mm_loadu_si128((const __m128i*)(p1 + 8));
                __m256i _a0_32 = _mm256_cvtepu16_epi32(_a0);
                __m256i _a1_32 = _mm256_cvtepu16_epi32(_a1);
                __m256i _b0_32 = _mm256_cvtepu16_epi32(_b0);
                __m256i _b1_32 = _mm256_cvtepu16_epi32(_b1);
                __m256i _paired_lo = _mm256_or_si256(_a0_32, _mm256_slli_epi32(_a1_32, 16));
                __m256i _paired_hi = _mm256_or_si256(_b0_32, _mm256_slli_epi32(_b1_32, 16));
                _mm256_storeu_si256((__m256i*)pp, _paired_lo);
                _mm256_storeu_si256((__m256i*)(pp + 16), _paired_hi);
                pp += 32;
                p0 += 16;
                p1 += 16;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[1]; pp[3] = 0;
                pp[4] = p0[2]; pp[5] = 0;
                pp[6] = p0[3]; pp[7] = 0;
                pp[8] = p0[4]; pp[9] = 0;
                pp[10] = p0[5]; pp[11] = 0;
                pp[12] = p0[6]; pp[13] = 0;
                pp[14] = p0[7]; pp[15] = 0;
                pp[16] = p1[0]; pp[17] = 0;
                pp[18] = p1[1]; pp[19] = 0;
                pp[20] = p1[2]; pp[21] = 0;
                pp[22] = p1[3]; pp[23] = 0;
                pp[24] = p1[4]; pp[25] = 0;
                pp[26] = p1[5]; pp[27] = 0;
                pp[28] = p1[6]; pp[29] = 0;
                pp[30] = p1[7]; pp[31] = 0;
                pp += 32;
                p0 += 8;
                p1 += 8;
            }
        }
        else if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k * 4;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 8) * B_hstep + k * 4;
            const unsigned short* p3 = (const unsigned short*)B + (j + jj + 12) * B_hstep + k * 4;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _a0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _a1 = _mm_loadu_si128((const __m128i*)p1);
                __m128i _a2 = _mm_loadu_si128((const __m128i*)p2);
                __m128i _a3 = _mm_loadu_si128((const __m128i*)p3);
                __m128i _t0 = _mm_unpacklo_epi16(_a0, _mm_srli_si128(_a0, 8));
                __m128i _t1 = _mm_unpacklo_epi16(_a1, _mm_srli_si128(_a1, 8));
                __m128i _t2 = _mm_unpacklo_epi16(_a2, _mm_srli_si128(_a2, 8));
                __m128i _t3 = _mm_unpacklo_epi16(_a3, _mm_srli_si128(_a3, 8));
                _mm_storeu_si128((__m128i*)pp, _t0);
                _mm_storeu_si128((__m128i*)(pp + 8), _t1);
                _mm_storeu_si128((__m128i*)(pp + 16), _t2);
                _mm_storeu_si128((__m128i*)(pp + 24), _t3);
                pp += 32;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[1]; pp[3] = 0;
                pp[4] = p0[2]; pp[5] = 0;
                pp[6] = p0[3]; pp[7] = 0;
                pp[8] = p1[0]; pp[9] = 0;
                pp[10] = p1[1]; pp[11] = 0;
                pp[12] = p1[2]; pp[13] = 0;
                pp[14] = p1[3]; pp[15] = 0;
                pp[16] = p2[0]; pp[17] = 0;
                pp[18] = p2[1]; pp[19] = 0;
                pp[20] = p2[2]; pp[21] = 0;
                pp[22] = p2[3]; pp[23] = 0;
                pp[24] = p3[0]; pp[25] = 0;
                pp[26] = p3[1]; pp[27] = 0;
                pp[28] = p3[2]; pp[29] = 0;
                pp[30] = p3[3]; pp[31] = 0;
                pp += 32;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
        }
        else // elempack == 1
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[1];
                pp[2] = p1[0]; pp[3] = p1[1];
                pp[4] = p2[0]; pp[5] = p2[1];
                pp[6] = p3[0]; pp[7] = p3[1];
                pp[8] = p4[0]; pp[9] = p4[1];
                pp[10] = p5[0]; pp[11] = p5[1];
                pp[12] = p6[0]; pp[13] = p6[1];
                pp[14] = p7[0]; pp[15] = p7[1];
                pp[16] = p8[0]; pp[17] = p8[1];
                pp[18] = p9[0]; pp[19] = p9[1];
                pp[20] = pa[0]; pp[21] = pa[1];
                pp[22] = pb[0]; pp[23] = pb[1];
                pp[24] = pc[0]; pp[25] = pc[1];
                pp[26] = pd[0]; pp[27] = pd[1];
                pp[28] = pe[0]; pp[29] = pe[1];
                pp[30] = pf[0]; pp[31] = pf[1];
                pp += 32;
                p0 += 2; p1 += 2; p2 += 2; p3 += 2;
                p4 += 2; p5 += 2; p6 += 2; p7 += 2;
                p8 += 2; p9 += 2; pa += 2; pb += 2;
                pc += 2; pd += 2; pe += 2; pf += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p1[0]; pp[3] = 0;
                pp[4] = p2[0]; pp[5] = 0;
                pp[6] = p3[0]; pp[7] = 0;
                pp[8] = p4[0]; pp[9] = 0;
                pp[10] = p5[0]; pp[11] = 0;
                pp[12] = p6[0]; pp[13] = 0;
                pp[14] = p7[0]; pp[15] = 0;
                pp[16] = p8[0]; pp[17] = 0;
                pp[18] = p9[0]; pp[19] = 0;
                pp[20] = pa[0]; pp[21] = 0;
                pp[22] = pb[0]; pp[23] = 0;
                pp[24] = pc[0]; pp[25] = 0;
                pp[26] = pd[0]; pp[27] = 0;
                pp[28] = pe[0]; pp[29] = 0;
                pp[30] = pf[0]; pp[31] = 0;
                pp += 32;
                p0++; p1++; p2++; p3++;
                p4++; p5++; p6++; p7++;
                p8++; p9++; pa++; pb++;
                pc++; pd++; pe++; pf++;
            }
        }
#else  // __AVX512BF16__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 16;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm256_storeu_si256((__m256i*)pp, _mm256_loadu_si256((const __m256i*)p0));
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
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                _mm_storeu_si128((__m128i*)(pp + 8), _mm_loadu_si128((const __m128i*)p1));
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
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
                _mm_storel_epi64((__m128i*)(pp + 4), _mm_loadl_epi64((const __m128i*)p1));
                _mm_storel_epi64((__m128i*)(pp + 8), _mm_loadl_epi64((const __m128i*)p2));
                _mm_storel_epi64((__m128i*)(pp + 12), _mm_loadl_epi64((const __m128i*)p3));
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
        }
        if (elempack == 1)
        {
            // pack 16 rows of B
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
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)p1);
                __m256i _r2 = _mm256_loadu_si256((const __m256i*)p2);
                __m256i _r3 = _mm256_loadu_si256((const __m256i*)p3);
                __m256i _r4 = _mm256_loadu_si256((const __m256i*)p4);
                __m256i _r5 = _mm256_loadu_si256((const __m256i*)p5);
                __m256i _r6 = _mm256_loadu_si256((const __m256i*)p6);
                __m256i _r7 = _mm256_loadu_si256((const __m256i*)p7);
                __m256i _r8 = _mm256_loadu_si256((const __m256i*)p8);
                __m256i _r9 = _mm256_loadu_si256((const __m256i*)p9);
                __m256i _ra = _mm256_loadu_si256((const __m256i*)pa);
                __m256i _rb = _mm256_loadu_si256((const __m256i*)pb);
                __m256i _rc = _mm256_loadu_si256((const __m256i*)pc);
                __m256i _rd = _mm256_loadu_si256((const __m256i*)pd);
                __m256i _re = _mm256_loadu_si256((const __m256i*)pe);
                __m256i _rf = _mm256_loadu_si256((const __m256i*)pf);
                transpose16x16_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                _mm256_storeu_si256((__m256i*)pp, _r0);
                _mm256_storeu_si256((__m256i*)(pp + 16), _r1);
                _mm256_storeu_si256((__m256i*)(pp + 32), _r2);
                _mm256_storeu_si256((__m256i*)(pp + 48), _r3);
                _mm256_storeu_si256((__m256i*)(pp + 64), _r4);
                _mm256_storeu_si256((__m256i*)(pp + 80), _r5);
                _mm256_storeu_si256((__m256i*)(pp + 96), _r6);
                _mm256_storeu_si256((__m256i*)(pp + 112), _r7);
                _mm256_storeu_si256((__m256i*)(pp + 128), _r8);
                _mm256_storeu_si256((__m256i*)(pp + 144), _r9);
                _mm256_storeu_si256((__m256i*)(pp + 160), _ra);
                _mm256_storeu_si256((__m256i*)(pp + 176), _rb);
                _mm256_storeu_si256((__m256i*)(pp + 192), _rc);
                _mm256_storeu_si256((__m256i*)(pp + 208), _rd);
                _mm256_storeu_si256((__m256i*)(pp + 224), _re);
                _mm256_storeu_si256((__m256i*)(pp + 240), _rf);
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
#endif // __AVX512BF16__
    }
#endif // __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
#if __AVX512BF16__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 8;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + 8));
                __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
                __m128i _t1 = _mm_unpackhi_epi16(_r0, _r1);
                _mm_storeu_si128((__m128i*)pp, _t0);
                _mm_storeu_si128((__m128i*)(pp + 8), _t1);
                pp += 16;
                p0 += 16;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[1]; pp[3] = 0;
                pp[4] = p0[2]; pp[5] = 0;
                pp[6] = p0[3]; pp[7] = 0;
                pp[8] = p0[4]; pp[9] = 0;
                pp[10] = p0[5]; pp[11] = 0;
                pp[12] = p0[6]; pp[13] = 0;
                pp[14] = p0[7]; pp[15] = 0;
                pp += 16;
                p0 += 8;
            }
        }
        else if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k * 4;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _a0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _b0 = _mm_loadu_si128((const __m128i*)p1);
                __m128i _t0 = _mm_unpacklo_epi16(_a0, _mm_srli_si128(_a0, 8));
                __m128i _t1 = _mm_unpacklo_epi16(_b0, _mm_srli_si128(_b0, 8));
                _mm_storeu_si128((__m128i*)pp, _t0);
                _mm_storeu_si128((__m128i*)(pp + 8), _t1);
                pp += 16;
                p0 += 8;
                p1 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[1]; pp[3] = 0;
                pp[4] = p0[2]; pp[5] = 0;
                pp[6] = p0[3]; pp[7] = 0;
                pp[8] = p1[0]; pp[9] = 0;
                pp[10] = p1[1]; pp[11] = 0;
                pp[12] = p1[2]; pp[13] = 0;
                pp[14] = p1[3]; pp[15] = 0;
                pp += 16;
                p0 += 4;
                p1 += 4;
            }
        }
        else // elempack == 1
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[1];
                pp[2] = p1[0]; pp[3] = p1[1];
                pp[4] = p2[0]; pp[5] = p2[1];
                pp[6] = p3[0]; pp[7] = p3[1];
                pp[8] = p4[0]; pp[9] = p4[1];
                pp[10] = p5[0]; pp[11] = p5[1];
                pp[12] = p6[0]; pp[13] = p6[1];
                pp[14] = p7[0]; pp[15] = p7[1];
                pp += 16;
                p0 += 2; p1 += 2; p2 += 2; p3 += 2;
                p4 += 2; p5 += 2; p6 += 2; p7 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p1[0]; pp[3] = 0;
                pp[4] = p2[0]; pp[5] = 0;
                pp[6] = p3[0]; pp[7] = 0;
                pp[8] = p4[0]; pp[9] = 0;
                pp[10] = p5[0]; pp[11] = 0;
                pp[12] = p6[0]; pp[13] = 0;
                pp[14] = p7[0]; pp[15] = 0;
                pp += 16;
                p0++; p1++; p2++; p3++;
                p4++; p5++; p6++; p7++;
            }
        }
#else  // __AVX512BF16__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 8;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                pp += 8;
                p0 += 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
                _mm_storel_epi64((__m128i*)(pp + 4), _mm_loadl_epi64((const __m128i*)p1));
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
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)p1);
                __m128i _r2 = _mm_loadu_si128((const __m128i*)p2);
                __m128i _r3 = _mm_loadu_si128((const __m128i*)p3);
                __m128i _r4 = _mm_loadu_si128((const __m128i*)p4);
                __m128i _r5 = _mm_loadu_si128((const __m128i*)p5);
                __m128i _r6 = _mm_loadu_si128((const __m128i*)p6);
                __m128i _r7 = _mm_loadu_si128((const __m128i*)p7);
                transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm_storeu_si128((__m128i*)pp, _r0);
                _mm_storeu_si128((__m128i*)(pp + 8), _r1);
                _mm_storeu_si128((__m128i*)(pp + 16), _r2);
                _mm_storeu_si128((__m128i*)(pp + 24), _r3);
                _mm_storeu_si128((__m128i*)(pp + 32), _r4);
                _mm_storeu_si128((__m128i*)(pp + 40), _r5);
                _mm_storeu_si128((__m128i*)(pp + 48), _r6);
                _mm_storeu_si128((__m128i*)(pp + 56), _r7);
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
#endif // __AVX512BF16__
    }
#endif // __AVX__
    for (; jj + 3 < max_jj; jj += 4)
    {
#if __AVX512BF16__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _t0 = _mm_unpacklo_epi16(_r0, _mm_srli_si128(_r0, 8));
                __m128i _t1 = _mm_unpackhi_epi16(_mm_slli_si128(_r0, 8), _r0);
                (void)_t1;
                _mm_storeu_si128((__m128i*)pp, _t0);
                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[1]; pp[3] = 0;
                pp[4] = p0[2]; pp[5] = 0;
                pp[6] = p0[3]; pp[7] = 0;
                pp += 8;
                p0 += 4;
            }
        }
        else // elempack == 1
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 2) * B_hstep + k;
            const unsigned short* p3 = (const unsigned short*)B + (j + jj + 3) * B_hstep + k;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[1];
                pp[2] = p1[0]; pp[3] = p1[1];
                pp[4] = p2[0]; pp[5] = p2[1];
                pp[6] = p3[0]; pp[7] = p3[1];
                pp += 8;
                p0 += 2; p1 += 2; p2 += 2; p3 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p1[0]; pp[3] = 0;
                pp[4] = p2[0]; pp[5] = 0;
                pp[6] = p3[0]; pp[7] = 0;
                pp += 8;
                p0++; p1++; p2++; p3++;
            }
        }
#else  // __AVX512BF16__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
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
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)p1);
                __m128i _r2 = _mm_loadl_epi64((const __m128i*)p2);
                __m128i _r3 = _mm_loadl_epi64((const __m128i*)p3);
                __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
                __m128i _t1 = _mm_unpacklo_epi16(_r2, _r3);
                _r0 = _mm_unpacklo_epi32(_t0, _t1);
                _r1 = _mm_unpackhi_epi32(_t0, _t1);
                _mm_storeu_si128((__m128i*)pp, _r0);
                _mm_storeu_si128((__m128i*)(pp + 8), _r1);
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
#endif // __AVX512BF16__
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __AVX512BF16__
        const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
        const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;

        int kk = 0;
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0]; pp[1] = p0[1];
            pp[2] = p1[0]; pp[3] = p1[1];
            pp += 4;
            p0 += 2;
            p1 += 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0]; pp[1] = 0;
            pp[2] = p1[0]; pp[3] = 0;
            pp += 4;
            p0++;
            p1++;
        }
#else  // __AVX512BF16__
        const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
        const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
#endif // __AVX512BF16__
    }
    for (; jj < max_jj; jj += 1)
    {
#if __AVX512BF16__
        const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

        int kk = 0;
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp += 2;
            p0 += 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = 0;
            pp += 2;
            p0++;
        }
#else  // __AVX512BF16__
        const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
#endif // __AVX512BF16__
    }
}

static void transpose_pack_B_tile_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    // transpose_pack_B is like pack_A for the B matrix
    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    unsigned short* pp = (unsigned short*)BT;

    int jj = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
#if __AVX512BF16__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 16;
            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                for (int p = 0; p < 8; p++)
                {
                    for (int n = 0; n < 16; n++)
                    {
                        pp[n * 2] = p0[n * 16 + p * 2];
                        pp[n * 2 + 1] = p0[n * 16 + p * 2 + 1];
                    }
                    pp += 32;
                }
                p0 += B_hstep * 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 16;
                for (int n = 0; n < 16; n++)
                {
                    pp[n * 2] = p0[n * 16 + kk_offset];
                    pp[n * 2 + 1] = p0[n * 16 + kk_offset + 1];
                }
                pp += 32;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 16;
                for (int n = 0; n < 16; n++)
                {
                    pp[n * 2] = p0[n * 16 + kk_offset];
                    pp[n * 2 + 1] = 0;
                }
                pp += 32;
            }
        }
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                for (int p = 0; p < 4; p++)
                {
                    for (int n = 0; n < 8; n++)
                    {
                        pp[n * 2] = p0[n * 8 + p * 2];
                        pp[n * 2 + 1] = p0[n * 8 + p * 2 + 1];
                    }
                    const unsigned short* p1 = p0 + 8 * 8;
                    for (int n = 0; n < 8; n++)
                    {
                        pp[16 + n * 2] = p1[n * 8 + p * 2];
                        pp[16 + n * 2 + 1] = p1[n * 8 + p * 2 + 1];
                    }
                    pp += 32;
                }
                p0 += B_hstep * 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 8;
                for (int n = 0; n < 8; n++)
                {
                    pp[n * 2] = p0[n * 8 + kk_offset];
                    pp[n * 2 + 1] = p0[n * 8 + kk_offset + 1];
                }
                const unsigned short* p1 = p0 + 8 * 8;
                for (int n = 0; n < 8; n++)
                {
                    pp[16 + n * 2] = p1[n * 8 + kk_offset];
                    pp[16 + n * 2 + 1] = p1[n * 8 + kk_offset + 1];
                }
                pp += 32;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 8;
                for (int n = 0; n < 8; n++)
                {
                    pp[n * 2] = p0[n * 8 + kk_offset];
                    pp[n * 2 + 1] = 0;
                }
                const unsigned short* p1 = p0 + 8 * 8;
                for (int n = 0; n < 8; n++)
                {
                    pp[16 + n * 2] = p1[n * 8 + kk_offset];
                    pp[16 + n * 2 + 1] = 0;
                }
                pp += 32;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                for (int p = 0; p < 2; p++)
                {
                    for (int n = 0; n < 16; n++)
                    {
                        pp[n * 2] = p0[n * 4 + p * 2];
                        pp[n * 2 + 1] = p0[n * 4 + p * 2 + 1];
                    }
                    pp += 32;
                }
                p0 += B_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 4;
                for (int n = 0; n < 16; n++)
                {
                    pp[n * 2] = p0[n * 4 + kk_offset];
                    pp[n * 2 + 1] = p0[n * 4 + kk_offset + 1];
                }
                pp += 32;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 4;
                for (int n = 0; n < 16; n++)
                {
                    pp[n * 2] = p0[n * 4 + kk_offset];
                    pp[n * 2 + 1] = 0;
                }
                pp += 32;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[B_hstep];
                pp[2] = p0[1]; pp[3] = p0[B_hstep + 1];
                pp[4] = p0[2]; pp[5] = p0[B_hstep + 2];
                pp[6] = p0[3]; pp[7] = p0[B_hstep + 3];
                pp[8] = p0[4]; pp[9] = p0[B_hstep + 4];
                pp[10] = p0[5]; pp[11] = p0[B_hstep + 5];
                pp[12] = p0[6]; pp[13] = p0[B_hstep + 6];
                pp[14] = p0[7]; pp[15] = p0[B_hstep + 7];
                pp[16] = p0[8]; pp[17] = p0[B_hstep + 8];
                pp[18] = p0[9]; pp[19] = p0[B_hstep + 9];
                pp[20] = p0[10]; pp[21] = p0[B_hstep + 10];
                pp[22] = p0[11]; pp[23] = p0[B_hstep + 11];
                pp[24] = p0[12]; pp[25] = p0[B_hstep + 12];
                pp[26] = p0[13]; pp[27] = p0[B_hstep + 13];
                pp[28] = p0[14]; pp[29] = p0[B_hstep + 14];
                pp[30] = p0[15]; pp[31] = p0[B_hstep + 15];
                pp += 32;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[1]; pp[3] = 0;
                pp[4] = p0[2]; pp[5] = 0;
                pp[6] = p0[3]; pp[7] = 0;
                pp[8] = p0[4]; pp[9] = 0;
                pp[10] = p0[5]; pp[11] = 0;
                pp[12] = p0[6]; pp[13] = 0;
                pp[14] = p0[7]; pp[15] = 0;
                pp[16] = p0[8]; pp[17] = 0;
                pp[18] = p0[9]; pp[19] = 0;
                pp[20] = p0[10]; pp[21] = 0;
                pp[22] = p0[11]; pp[23] = 0;
                pp[24] = p0[12]; pp[25] = 0;
                pp[26] = p0[13]; pp[27] = 0;
                pp[28] = p0[14]; pp[29] = 0;
                pp[30] = p0[15]; pp[31] = 0;
                pp += 32;
                p0 += B_hstep;
            }
        }
#else  // __AVX512BF16__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)(p0 + 16));
                __m256i _r2 = _mm256_loadu_si256((const __m256i*)(p0 + 32));
                __m256i _r3 = _mm256_loadu_si256((const __m256i*)(p0 + 48));
                __m256i _r4 = _mm256_loadu_si256((const __m256i*)(p0 + 64));
                __m256i _r5 = _mm256_loadu_si256((const __m256i*)(p0 + 80));
                __m256i _r6 = _mm256_loadu_si256((const __m256i*)(p0 + 96));
                __m256i _r7 = _mm256_loadu_si256((const __m256i*)(p0 + 112));
                __m256i _r8 = _mm256_loadu_si256((const __m256i*)(p0 + 128));
                __m256i _r9 = _mm256_loadu_si256((const __m256i*)(p0 + 144));
                __m256i _ra = _mm256_loadu_si256((const __m256i*)(p0 + 160));
                __m256i _rb = _mm256_loadu_si256((const __m256i*)(p0 + 176));
                __m256i _rc = _mm256_loadu_si256((const __m256i*)(p0 + 192));
                __m256i _rd = _mm256_loadu_si256((const __m256i*)(p0 + 208));
                __m256i _re = _mm256_loadu_si256((const __m256i*)(p0 + 224));
                __m256i _rf = _mm256_loadu_si256((const __m256i*)(p0 + 240));
                transpose16x16_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                _mm256_storeu_si256((__m256i*)pp, _r0);
                _mm256_storeu_si256((__m256i*)(pp + 16), _r1);
                _mm256_storeu_si256((__m256i*)(pp + 32), _r2);
                _mm256_storeu_si256((__m256i*)(pp + 48), _r3);
                _mm256_storeu_si256((__m256i*)(pp + 64), _r4);
                _mm256_storeu_si256((__m256i*)(pp + 80), _r5);
                _mm256_storeu_si256((__m256i*)(pp + 96), _r6);
                _mm256_storeu_si256((__m256i*)(pp + 112), _r7);
                _mm256_storeu_si256((__m256i*)(pp + 128), _r8);
                _mm256_storeu_si256((__m256i*)(pp + 144), _r9);
                _mm256_storeu_si256((__m256i*)(pp + 160), _ra);
                _mm256_storeu_si256((__m256i*)(pp + 176), _rb);
                _mm256_storeu_si256((__m256i*)(pp + 192), _rc);
                _mm256_storeu_si256((__m256i*)(pp + 208), _rd);
                _mm256_storeu_si256((__m256i*)(pp + 224), _re);
                _mm256_storeu_si256((__m256i*)(pp + 240), _rf);
                pp += 256;
                p0 += B_hstep * 16;
            }
            for (; kk < max_kk; kk++)
            {
                _mm256_storeu_si256((__m256i*)pp, _mm256_loadu_si256((const __m256i*)p0));
                pp += 16;
                p0 += B_hstep;
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
                _mm256_storeu_si256((__m256i*)pp, _mm256_inserti128_si256(_mm256_castsi128_si256(_r0), _r1, 1));
                _mm256_storeu_si256((__m256i*)(pp + 16), _mm256_inserti128_si256(_mm256_castsi128_si256(_r2), _r3, 1));
                _mm256_storeu_si256((__m256i*)(pp + 32), _mm256_inserti128_si256(_mm256_castsi128_si256(_r4), _r5, 1));
                _mm256_storeu_si256((__m256i*)(pp + 48), _mm256_inserti128_si256(_mm256_castsi128_si256(_r6), _r7, 1));
                _mm256_storeu_si256((__m256i*)(pp + 64), _mm256_inserti128_si256(_mm256_castsi128_si256(_r8), _r9, 1));
                _mm256_storeu_si256((__m256i*)(pp + 80), _mm256_inserti128_si256(_mm256_castsi128_si256(_ra), _rb, 1));
                _mm256_storeu_si256((__m256i*)(pp + 96), _mm256_inserti128_si256(_mm256_castsi128_si256(_rc), _rd, 1));
                _mm256_storeu_si256((__m256i*)(pp + 112), _mm256_inserti128_si256(_mm256_castsi128_si256(_re), _rf, 1));
                pp += 128;
                p0 += B_hstep * 8;
            }
            for (; kk < max_kk; kk++)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                _mm_storeu_si128((__m128i*)(pp + 8), _mm_loadu_si128((const __m128i*)(p0 + 8)));
                pp += 16;
                p0 += B_hstep;
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
                _mm256_storeu_si256((__m256i*)pp, _col0);
                _mm256_storeu_si256((__m256i*)(pp + 16), _col1);
                _mm256_storeu_si256((__m256i*)(pp + 32), _col2);
                _mm256_storeu_si256((__m256i*)(pp + 48), _col3);
                pp += 64;
                p0 += B_hstep * 4;
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
                pp[8] = p0[32];
                pp[9] = p0[36];
                pp[10] = p0[40];
                pp[11] = p0[44];
                pp[12] = p0[48];
                pp[13] = p0[52];
                pp[14] = p0[56];
                pp[15] = p0[60];
                pp += 16;
                p0 += B_hstep;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                _mm256_storeu_si256((__m256i*)pp, _mm256_loadu_si256((const __m256i*)p0));
                pp += 16;
                p0 += B_hstep;
            }
        }
#endif // __AVX512BF16__
    }
#endif // __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
#if __AVX512BF16__
#if __AVX512F__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                for (int p = 0; p < 8; p++)
                {
                    for (int n = 0; n < 8; n++)
                    {
                        pp[n * 2] = p0[n * 16 + p * 2];
                        pp[n * 2 + 1] = p0[n * 16 + p * 2 + 1];
                    }
                    pp += 16;
                }
                p0 += B_hstep * 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 16;
                for (int n = 0; n < 8; n++)
                {
                    pp[n * 2] = p0[n * 16 + kk_offset];
                    pp[n * 2 + 1] = p0[n * 16 + kk_offset + 1];
                }
                pp += 16;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 16;
                for (int n = 0; n < 8; n++)
                {
                    pp[n * 2] = p0[n * 16 + kk_offset];
                    pp[n * 2 + 1] = 0;
                }
                pp += 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                for (int p = 0; p < 4; p++)
                {
                    for (int n = 0; n < 8; n++)
                    {
                        pp[n * 2] = p0[n * 8 + p * 2];
                        pp[n * 2 + 1] = p0[n * 8 + p * 2 + 1];
                    }
                    pp += 16;
                }
                p0 += B_hstep * 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 8;
                for (int n = 0; n < 8; n++)
                {
                    pp[n * 2] = p0[n * 8 + kk_offset];
                    pp[n * 2 + 1] = p0[n * 8 + kk_offset + 1];
                }
                pp += 16;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 8;
                for (int n = 0; n < 8; n++)
                {
                    pp[n * 2] = p0[n * 8 + kk_offset];
                    pp[n * 2 + 1] = 0;
                }
                pp += 16;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                for (int p = 0; p < 2; p++)
                {
                    for (int n = 0; n < 8; n++)
                    {
                        pp[n * 2] = p0[n * 4 + p * 2];
                        pp[n * 2 + 1] = p0[n * 4 + p * 2 + 1];
                    }
                    pp += 16;
                }
                p0 += B_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 4;
                for (int n = 0; n < 8; n++)
                {
                    pp[n * 2] = p0[n * 4 + kk_offset];
                    pp[n * 2 + 1] = p0[n * 4 + kk_offset + 1];
                }
                pp += 16;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 4;
                for (int n = 0; n < 8; n++)
                {
                    pp[n * 2] = p0[n * 4 + kk_offset];
                    pp[n * 2 + 1] = 0;
                }
                pp += 16;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[B_hstep];
                pp[2] = p0[1]; pp[3] = p0[B_hstep + 1];
                pp[4] = p0[2]; pp[5] = p0[B_hstep + 2];
                pp[6] = p0[3]; pp[7] = p0[B_hstep + 3];
                pp[8] = p0[4]; pp[9] = p0[B_hstep + 4];
                pp[10] = p0[5]; pp[11] = p0[B_hstep + 5];
                pp[12] = p0[6]; pp[13] = p0[B_hstep + 6];
                pp[14] = p0[7]; pp[15] = p0[B_hstep + 7];
                pp += 16;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[1]; pp[3] = 0;
                pp[4] = p0[2]; pp[5] = 0;
                pp[6] = p0[3]; pp[7] = 0;
                pp[8] = p0[4]; pp[9] = 0;
                pp[10] = p0[5]; pp[11] = 0;
                pp[12] = p0[6]; pp[13] = 0;
                pp[14] = p0[7]; pp[15] = 0;
                pp += 16;
                p0 += B_hstep;
            }
        }
#else  // __AVX512BF16__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + B_hstep));
                __m128i _r2 = _mm_loadu_si128((const __m128i*)(p0 + B_hstep * 2));
                __m128i _r3 = _mm_loadu_si128((const __m128i*)(p0 + B_hstep * 3));
                __m128i _r4 = _mm_loadu_si128((const __m128i*)(p0 + B_hstep * 4));
                __m128i _r5 = _mm_loadu_si128((const __m128i*)(p0 + B_hstep * 5));
                __m128i _r6 = _mm_loadu_si128((const __m128i*)(p0 + B_hstep * 6));
                __m128i _r7 = _mm_loadu_si128((const __m128i*)(p0 + B_hstep * 7));
                transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm_storeu_si128((__m128i*)pp, _r0);
                _mm_storeu_si128((__m128i*)(pp + 8), _r1);
                _mm_storeu_si128((__m128i*)(pp + 16), _r2);
                _mm_storeu_si128((__m128i*)(pp + 24), _r3);
                _mm_storeu_si128((__m128i*)(pp + 32), _r4);
                _mm_storeu_si128((__m128i*)(pp + 40), _r5);
                _mm_storeu_si128((__m128i*)(pp + 48), _r6);
                _mm_storeu_si128((__m128i*)(pp + 56), _r7);
                pp += 64;
                p0 += B_hstep * 8;
            }
            for (; kk < max_kk; kk++)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                pp += 8;
                p0 += B_hstep;
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
                transpose8x4_epi16(_a0, _a1, _a2, _a3);
                transpose8x4_epi16(_b0, _b1, _b2, _b3);
                _mm_storel_epi64((__m128i*)pp, _a0);
                _mm_storel_epi64((__m128i*)(pp + 4), _b0);
                _mm_storel_epi64((__m128i*)(pp + 8), _mm_unpackhi_epi64(_a0, _a0));
                _mm_storel_epi64((__m128i*)(pp + 12), _mm_unpackhi_epi64(_b0, _b0));
                _mm_storel_epi64((__m128i*)(pp + 16), _a1);
                _mm_storel_epi64((__m128i*)(pp + 20), _b1);
                _mm_storel_epi64((__m128i*)(pp + 24), _mm_unpackhi_epi64(_a1, _a1));
                _mm_storel_epi64((__m128i*)(pp + 28), _mm_unpackhi_epi64(_b1, _b1));
                pp += 32;
                p0 += B_hstep * 4;
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
                p0 += B_hstep;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                pp += 8;
                p0 += B_hstep;
            }
        }
#endif // __AVX512BF16__
    }
#endif // __AVX__
    for (; jj + 3 < max_jj; jj += 4)
    {
#if __AVX512BF16__
#if __AVX512F__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                for (int p = 0; p < 8; p++)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        pp[n * 2] = p0[n * 16 + p * 2];
                        pp[n * 2 + 1] = p0[n * 16 + p * 2 + 1];
                    }
                    pp += 8;
                }
                p0 += B_hstep * 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 16;
                for (int n = 0; n < 4; n++)
                {
                    pp[n * 2] = p0[n * 16 + kk_offset];
                    pp[n * 2 + 1] = p0[n * 16 + kk_offset + 1];
                }
                pp += 8;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 16;
                for (int n = 0; n < 4; n++)
                {
                    pp[n * 2] = p0[n * 16 + kk_offset];
                    pp[n * 2 + 1] = 0;
                }
                pp += 8;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                for (int p = 0; p < 4; p++)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        pp[n * 2] = p0[n * 8 + p * 2];
                        pp[n * 2 + 1] = p0[n * 8 + p * 2 + 1];
                    }
                    pp += 8;
                }
                p0 += B_hstep * 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 8;
                for (int n = 0; n < 4; n++)
                {
                    pp[n * 2] = p0[n * 8 + kk_offset];
                    pp[n * 2 + 1] = p0[n * 8 + kk_offset + 1];
                }
                pp += 8;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 8;
                for (int n = 0; n < 4; n++)
                {
                    pp[n * 2] = p0[n * 8 + kk_offset];
                    pp[n * 2 + 1] = 0;
                }
                pp += 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // elempack=4: p0[0..3] = k0..k3 for n0, p0[4..7] = k0..k3 for n1, etc.
                pp[0] = p0[0]; pp[1] = p0[1];    // k0|k1 for n0
                pp[2] = p0[4]; pp[3] = p0[5];    // k0|k1 for n1
                pp[4] = p0[8]; pp[5] = p0[9];    // k0|k1 for n2
                pp[6] = p0[12]; pp[7] = p0[13];  // k0|k1 for n3
                pp[8] = p0[2]; pp[9] = p0[3];    // k2|k3 for n0
                pp[10] = p0[6]; pp[11] = p0[7];  // k2|k3 for n1
                pp[12] = p0[10]; pp[13] = p0[11]; // k2|k3 for n2
                pp[14] = p0[14]; pp[15] = p0[15]; // k2|k3 for n3
                pp += 16;
                p0 += B_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[1];
                pp[2] = p0[4]; pp[3] = p0[5];
                pp[4] = p0[8]; pp[5] = p0[9];
                pp[6] = p0[12]; pp[7] = p0[13];
                pp += 8;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[4]; pp[3] = 0;
                pp[4] = p0[8]; pp[5] = 0;
                pp[6] = p0[12]; pp[7] = 0;
                pp += 8;
                p0++;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[B_hstep];
                pp[2] = p0[1]; pp[3] = p0[B_hstep + 1];
                pp[4] = p0[2]; pp[5] = p0[B_hstep + 2];
                pp[6] = p0[3]; pp[7] = p0[B_hstep + 3];
                pp += 8;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[1]; pp[3] = 0;
                pp[4] = p0[2]; pp[5] = 0;
                pp[6] = p0[3]; pp[7] = 0;
                pp += 8;
                p0 += B_hstep;
            }
        }
#else  // __AVX512BF16__
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
                __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
                __m128i _t1 = _mm_unpacklo_epi16(_r2, _r3);
                _r0 = _mm_unpacklo_epi32(_t0, _t1);
                _r1 = _mm_unpackhi_epi32(_t0, _t1);
                _mm_storeu_si128((__m128i*)pp, _r0);
                _mm_storeu_si128((__m128i*)(pp + 8), _r1);
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
                p0 += B_hstep;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

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
#endif // __AVX512BF16__
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __AVX512BF16__
#if __AVX512F__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                for (int m = 0; m < 8; m++)
                {
                    pp[0] = p0[m * 2];
                    pp[1] = p0[m * 2 + 1];
                    pp[2] = p0[16 + m * 2];
                    pp[3] = p0[16 + m * 2 + 1];
                    pp += 4;
                }
                p0 += B_hstep * 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 16;
                pp[0] = p0[kk_offset];
                pp[1] = p0[kk_offset + 1];
                pp[2] = p0[16 + kk_offset];
                pp[3] = p0[16 + kk_offset + 1];
                pp += 4;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 16;
                pp[0] = p0[kk_offset];
                pp[1] = 0;
                pp[2] = p0[16 + kk_offset];
                pp[3] = 0;
                pp += 4;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                for (int m = 0; m < 4; m++)
                {
                    pp[0] = p0[m * 2];
                    pp[1] = p0[m * 2 + 1];
                    pp[2] = p0[8 + m * 2];
                    pp[3] = p0[8 + m * 2 + 1];
                    pp += 4;
                }
                p0 += B_hstep * 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 8;
                pp[0] = p0[kk_offset];
                pp[1] = p0[kk_offset + 1];
                pp[2] = p0[8 + kk_offset];
                pp[3] = p0[8 + kk_offset + 1];
                pp += 4;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 8;
                pp[0] = p0[kk_offset];
                pp[1] = 0;
                pp[2] = p0[8 + kk_offset];
                pp[3] = 0;
                pp += 4;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0]; pp[1] = p0[1];
                pp[2] = p0[4]; pp[3] = p0[5];
                pp[4] = p0[2]; pp[5] = p0[3];
                pp[6] = p0[6]; pp[7] = p0[7];
                pp += 8;
                p0 += B_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[1];
                pp[2] = p0[4]; pp[3] = p0[5];
                pp += 4;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[4]; pp[3] = 0;
                pp += 4;
                p0++;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[B_hstep];
                pp[2] = p0[1]; pp[3] = p0[B_hstep + 1];
                pp += 4;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp[2] = p0[1]; pp[3] = 0;
                pp += 4;
                p0 += B_hstep;
            }
        }
#else  // __AVX512BF16__
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                for (int m = 0; m < 16; m++)
                {
                    pp[0] = p0[m];
                    pp[1] = p0[16 + m];
                    pp += 2;
                }
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
                for (int m = 0; m < 8; m++)
                {
                    pp[0] = p0[m];
                    pp[1] = p0[8 + m];
                    pp += 2;
                }
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
                pp[0] = p0[0]; pp[1] = p0[4];
                pp[2] = p0[1]; pp[3] = p0[5];
                pp[4] = p0[2]; pp[5] = p0[6];
                pp[6] = p0[3]; pp[7] = p0[7];
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
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += B_hstep;
            }
        }
#endif // __AVX512BF16__
    }
    for (; jj < max_jj; jj += 1)
    {
#if __AVX512BF16__
#if __AVX512F__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                for (int m = 0; m < 8; m++)
                {
                    pp[0] = p0[m * 2];
                    pp[1] = p0[m * 2 + 1];
                    pp += 2;
                }
                p0 += B_hstep * 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 16;
                pp[0] = p0[kk_offset];
                pp[1] = p0[kk_offset + 1];
                pp += 2;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 16;
                pp[0] = p0[kk_offset];
                pp[1] = 0;
                pp += 2;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                for (int m = 0; m < 4; m++)
                {
                    pp[0] = p0[m * 2];
                    pp[1] = p0[m * 2 + 1];
                    pp += 2;
                }
                p0 += B_hstep * 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int kk_offset = kk % 8;
                pp[0] = p0[kk_offset];
                pp[1] = p0[kk_offset + 1];
                pp += 2;
            }
            for (; kk < max_kk; kk++)
            {
                int kk_offset = kk % 8;
                pp[0] = p0[kk_offset];
                pp[1] = 0;
                pp += 2;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0]; pp[1] = p0[1];
                pp[2] = p0[2]; pp[3] = p0[3];
                pp += 4;
                p0 += B_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0]; pp[1] = p0[1];
                pp += 2;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0]; pp[1] = 0;
                pp += 2;
                p0++;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[B_hstep];
                pp += 2;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = 0;
                pp += 2;
                p0 += B_hstep;
            }
        }
#else  // __AVX512BF16__
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 16;

            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                _mm256_storeu_si256((__m256i*)pp, _mm256_loadu_si256((const __m256i*)p0));
                pp += 16;
                p0 += B_hstep * 16;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += B_hstep;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                pp += 8;
                p0 += B_hstep * 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += B_hstep;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
                pp += 4;
                p0 += B_hstep * 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += B_hstep;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += B_hstep;
            }
        }
#endif // __AVX512BF16__
    }
}

static void gemm_transB_packed_tile_bf16(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
    // bf16 packed data -> fp32 accumulation via broadcast-based approach
    // Load A vector once per kk step, broadcast each B element, fmadd with A
    // Each sum register holds one column of the output tile

    const unsigned short* pAT = (const unsigned short*)AT_tile;
    const unsigned short* pBT = (const unsigned short*)BT_tile;
    float* outptr = (float*)topT_tile;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 15 < max_jj; jj += 16)
        {
            const unsigned short* pA = pAT;

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
            __m512 _sumc;
            __m512 _sumd;
            __m512 _sume;
            __m512 _sumf;

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
                _sumc = _mm512_setzero_ps();
                _sumd = _mm512_setzero_ps();
                _sume = _mm512_setzero_ps();
                _sumf = _mm512_setzero_ps();
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
                _sum1 = _mm512_load_ps(outptr + 16);
                _sum2 = _mm512_load_ps(outptr + 32);
                _sum3 = _mm512_load_ps(outptr + 48);
                _sum4 = _mm512_load_ps(outptr + 64);
                _sum5 = _mm512_load_ps(outptr + 80);
                _sum6 = _mm512_load_ps(outptr + 96);
                _sum7 = _mm512_load_ps(outptr + 112);
                _sum8 = _mm512_load_ps(outptr + 128);
                _sum9 = _mm512_load_ps(outptr + 128 + 16);
                _suma = _mm512_load_ps(outptr + 128 + 32);
                _sumb = _mm512_load_ps(outptr + 128 + 48);
                _sumc = _mm512_load_ps(outptr + 128 + 64);
                _sumd = _mm512_load_ps(outptr + 128 + 80);
                _sume = _mm512_load_ps(outptr + 128 + 96);
                _sumf = _mm512_load_ps(outptr + 128 + 112);
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);

                __m512i _pA1 = _mm512_alignr_epi8(_pA0, _pA0, 8);
                __m512i _pA2 = _mm512_shuffle_i32x4(_pA0, _pA0, _MM_SHUFFLE(2, 3, 0, 1));
                __m512i _pA3 = _mm512_alignr_epi8(_pA2, _pA2, 8);

                __m512i _pB1 = _mm512_alignr_epi8(_pB0, _pB0, 4);
                __m512i _pB2 = _mm512_shuffle_i32x4(_pB0, _pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pB3 = _mm512_alignr_epi8(_pB2, _pB2, 4);

                _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_pA0, (__m512bh)_pB0);
                _sum1 = _mm512_dpbf16_ps(_sum1, (__m512bh)_pA0, (__m512bh)_pB1);
                _sum2 = _mm512_dpbf16_ps(_sum2, (__m512bh)_pA1, (__m512bh)_pB0);
                _sum3 = _mm512_dpbf16_ps(_sum3, (__m512bh)_pA1, (__m512bh)_pB1);
                _sum4 = _mm512_dpbf16_ps(_sum4, (__m512bh)_pA0, (__m512bh)_pB2);
                _sum5 = _mm512_dpbf16_ps(_sum5, (__m512bh)_pA0, (__m512bh)_pB3);
                _sum6 = _mm512_dpbf16_ps(_sum6, (__m512bh)_pA1, (__m512bh)_pB2);
                _sum7 = _mm512_dpbf16_ps(_sum7, (__m512bh)_pA1, (__m512bh)_pB3);
                _sum8 = _mm512_dpbf16_ps(_sum8, (__m512bh)_pA2, (__m512bh)_pB0);
                _sum9 = _mm512_dpbf16_ps(_sum9, (__m512bh)_pA2, (__m512bh)_pB1);
                _suma = _mm512_dpbf16_ps(_suma, (__m512bh)_pA3, (__m512bh)_pB0);
                _sumb = _mm512_dpbf16_ps(_sumb, (__m512bh)_pA3, (__m512bh)_pB1);
                _sumc = _mm512_dpbf16_ps(_sumc, (__m512bh)_pA2, (__m512bh)_pB2);
                _sumd = _mm512_dpbf16_ps(_sumd, (__m512bh)_pA2, (__m512bh)_pB3);
                _sume = _mm512_dpbf16_ps(_sume, (__m512bh)_pA3, (__m512bh)_pB2);
                _sumf = _mm512_dpbf16_ps(_sumf, (__m512bh)_pA3, (__m512bh)_pB3);

                pA += 32;
                pB += 32;
            }
#else // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m512 _pA0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)pA)), 16));

                _sum0 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[1])), _sum1);
                _sum2 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[2])), _sum2);
                _sum3 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[3])), _sum3);
                _sum4 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[4])), _sum4);
                _sum5 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[5])), _sum5);
                _sum6 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[6])), _sum6);
                _sum7 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[7])), _sum7);
                _sum8 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[8])), _sum8);
                _sum9 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[9])), _sum9);
                _suma = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[10])), _suma);
                _sumb = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[11])), _sumb);
                _sumc = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[12])), _sumc);
                _sumd = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[13])), _sumd);
                _sume = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[14])), _sume);
                _sumf = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[15])), _sumf);

                pA += 16;
                pB += 16;
            }
#endif // __AVX512BF16__

            _mm512_store_ps(outptr, _sum0);
            _mm512_store_ps(outptr + 16, _sum1);
            _mm512_store_ps(outptr + 32, _sum2);
            _mm512_store_ps(outptr + 48, _sum3);
            _mm512_store_ps(outptr + 64, _sum4);
            _mm512_store_ps(outptr + 80, _sum5);
            _mm512_store_ps(outptr + 96, _sum6);
            _mm512_store_ps(outptr + 112, _sum7);
            _mm512_store_ps(outptr + 128, _sum8);
            _mm512_store_ps(outptr + 128 + 16, _sum9);
            _mm512_store_ps(outptr + 128 + 32, _suma);
            _mm512_store_ps(outptr + 128 + 48, _sumb);
            _mm512_store_ps(outptr + 128 + 64, _sumc);
            _mm512_store_ps(outptr + 128 + 80, _sumd);
            _mm512_store_ps(outptr + 128 + 96, _sume);
            _mm512_store_ps(outptr + 128 + 112, _sumf);
            outptr += 256;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

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
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
                _sum1 = _mm512_load_ps(outptr + 16);
                _sum2 = _mm512_load_ps(outptr + 32);
                _sum3 = _mm512_load_ps(outptr + 48);
                _sum4 = _mm512_load_ps(outptr + 64);
                _sum5 = _mm512_load_ps(outptr + 80);
                _sum6 = _mm512_load_ps(outptr + 96);
                _sum7 = _mm512_load_ps(outptr + 112);
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                __m256i _pBB = _mm256_loadu_si256((const __m256i*)pB);
                __m512i _pB0 = combine8x2_epi32(_pBB, _pBB);

                __m512i _pA1 = _mm512_alignr_epi8(_pA0, _pA0, 8);

                __m512i _pB1 = _mm512_alignr_epi8(_pB0, _pB0, 4);
                __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pB3 = _mm512_alignr_epi8(_pB2, _pB2, 4);

                _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_pA0, (__m512bh)_pB0);
                _sum1 = _mm512_dpbf16_ps(_sum1, (__m512bh)_pA0, (__m512bh)_pB1);
                _sum2 = _mm512_dpbf16_ps(_sum2, (__m512bh)_pA1, (__m512bh)_pB0);
                _sum3 = _mm512_dpbf16_ps(_sum3, (__m512bh)_pA1, (__m512bh)_pB1);
                _sum4 = _mm512_dpbf16_ps(_sum4, (__m512bh)_pA0, (__m512bh)_pB2);
                _sum5 = _mm512_dpbf16_ps(_sum5, (__m512bh)_pA0, (__m512bh)_pB3);
                _sum6 = _mm512_dpbf16_ps(_sum6, (__m512bh)_pA1, (__m512bh)_pB2);
                _sum7 = _mm512_dpbf16_ps(_sum7, (__m512bh)_pA1, (__m512bh)_pB3);

                pA += 32;
                pB += 16;
            }
#else // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m512 _pA0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)pA)), 16));

                _sum0 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[1])), _sum1);
                _sum2 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[2])), _sum2);
                _sum3 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[3])), _sum3);
                _sum4 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[4])), _sum4);
                _sum5 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[5])), _sum5);
                _sum6 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[6])), _sum6);
                _sum7 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[7])), _sum7);

                pA += 16;
                pB += 8;
            }
#endif // __AVX512BF16__

            _mm512_store_ps(outptr, _sum0);
            _mm512_store_ps(outptr + 16, _sum1);
            _mm512_store_ps(outptr + 32, _sum2);
            _mm512_store_ps(outptr + 48, _sum3);
            _mm512_store_ps(outptr + 64, _sum4);
            _mm512_store_ps(outptr + 80, _sum5);
            _mm512_store_ps(outptr + 96, _sum6);
            _mm512_store_ps(outptr + 112, _sum7);
            outptr += 128;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

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
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
                _sum1 = _mm512_load_ps(outptr + 16);
                _sum2 = _mm512_load_ps(outptr + 32);
                _sum3 = _mm512_load_ps(outptr + 48);
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                __m512i _pA = _mm512_loadu_si512((const __m512i*)pA);
                _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_pA, (__m512bh)_mm512_set1_epi32(((const int*)pB)[0]));
                _sum1 = _mm512_dpbf16_ps(_sum1, (__m512bh)_pA, (__m512bh)_mm512_set1_epi32(((const int*)pB)[1]));
                _sum2 = _mm512_dpbf16_ps(_sum2, (__m512bh)_pA, (__m512bh)_mm512_set1_epi32(((const int*)pB)[2]));
                _sum3 = _mm512_dpbf16_ps(_sum3, (__m512bh)_pA, (__m512bh)_mm512_set1_epi32(((const int*)pB)[3]));
                pA += 32;
                pB += 8;
            }
#else // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m512 _pA0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)pA)), 16));

                _sum0 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[1])), _sum1);
                _sum2 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[2])), _sum2);
                _sum3 = _mm512_fmadd_ps(_pA0, _mm512_set1_ps(bfloat16_to_float32(pB[3])), _sum3);

                pA += 16;
                pB += 4;
            }
#endif // __AVX512BF16__

            _mm512_store_ps(outptr, _sum0);
            _mm512_store_ps(outptr + 16, _sum1);
            _mm512_store_ps(outptr + 32, _sum2);
            _mm512_store_ps(outptr + 48, _sum3);
            outptr += 64;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* pA = pAT;

            __m512 _sum0;
            __m512 _sum1;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_ps();
                _sum1 = _mm512_setzero_ps();
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
                _sum1 = _mm512_load_ps(outptr + 16);
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                __m512i _pA = _mm512_loadu_si512((const __m512i*)pA);
                _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_pA, (__m512bh)_mm512_set1_epi32(((const int*)pB)[0]));
                _sum1 = _mm512_dpbf16_ps(_sum1, (__m512bh)_pA, (__m512bh)_mm512_set1_epi32(((const int*)pB)[1]));
                pA += 32;
                pB += 4;
            }
#else // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m512 _pA0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)pA)), 16));
                __m512 _pB0 = _mm512_set1_ps(bfloat16_to_float32(pB[0]));
                __m512 _pB1 = _mm512_set1_ps(bfloat16_to_float32(pB[1]));

                _sum0 = _mm512_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm512_fmadd_ps(_pA0, _pB1, _sum1);

                pA += 16;
                pB += 2;
            }
#endif // __AVX512BF16__

            _mm512_store_ps(outptr, _sum0);
            _mm512_store_ps(outptr + 16, _sum1);
            outptr += 32;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned short* pA = pAT;

            __m512 _sum0;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_ps();
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                __m512i _pA = _mm512_loadu_si512((const __m512i*)pA);
                _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_pA, (__m512bh)_mm512_set1_epi32(((const int*)pB)[0]));
                pA += 32;
                pB += 2;
            }
#else // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m512 _pA0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)pA)), 16));
                __m512 _pB0 = _mm512_set1_ps(bfloat16_to_float32(pB[0]));

                _sum0 = _mm512_fmadd_ps(_pA0, _pB0, _sum0);

                pA += 16;
                pB += 1;
            }
#endif // __AVX512BF16__

            _mm512_store_ps(outptr, _sum0);
            outptr += 16;
        }

#if __AVX512BF16__
        pAT += ((max_kk + 1) / 2 * 2) * 16;
#else
        pAT += max_kk * 16;
#endif
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            const unsigned short* pA = pAT;

            __m256 _sum0 = _mm256_setzero_ps();
            __m256 _sum1 = _mm256_setzero_ps();
            __m256 _sum2 = _mm256_setzero_ps();
            __m256 _sum3 = _mm256_setzero_ps();
            __m256 _sum4 = _mm256_setzero_ps();
            __m256 _sum5 = _mm256_setzero_ps();
            __m256 _sum6 = _mm256_setzero_ps();
            __m256 _sum7 = _mm256_setzero_ps();
            __m256 _sum8 = _mm256_setzero_ps();
            __m256 _sum9 = _mm256_setzero_ps();
            __m256 _suma = _mm256_setzero_ps();
            __m256 _sumb = _mm256_setzero_ps();
            __m256 _sumc = _mm256_setzero_ps();
            __m256 _sumd = _mm256_setzero_ps();
            __m256 _sume = _mm256_setzero_ps();
            __m256 _sumf = _mm256_setzero_ps();

            if (k != 0)
            {
                _sum0 = _mm256_load_ps(outptr);
                _sum1 = _mm256_load_ps(outptr + 8);
                _sum2 = _mm256_load_ps(outptr + 16);
                _sum3 = _mm256_load_ps(outptr + 24);
                _sum4 = _mm256_load_ps(outptr + 32);
                _sum5 = _mm256_load_ps(outptr + 40);
                _sum6 = _mm256_load_ps(outptr + 48);
                _sum7 = _mm256_load_ps(outptr + 56);
                _sum8 = _mm256_load_ps(outptr + 64);
                _sum9 = _mm256_load_ps(outptr + 72);
                _suma = _mm256_load_ps(outptr + 80);
                _sumb = _mm256_load_ps(outptr + 88);
                _sumc = _mm256_load_ps(outptr + 96);
                _sumd = _mm256_load_ps(outptr + 104);
                _sume = _mm256_load_ps(outptr + 112);
                _sumf = _mm256_load_ps(outptr + 120);
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[0]));
                _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[1]));
                _sum2 = _mm256_dpbf16_ps(_sum2, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[2]));
                _sum3 = _mm256_dpbf16_ps(_sum3, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[3]));
                _sum4 = _mm256_dpbf16_ps(_sum4, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[4]));
                _sum5 = _mm256_dpbf16_ps(_sum5, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[5]));
                _sum6 = _mm256_dpbf16_ps(_sum6, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[6]));
                _sum7 = _mm256_dpbf16_ps(_sum7, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[7]));
                _sum8 = _mm256_dpbf16_ps(_sum8, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[8]));
                _sum9 = _mm256_dpbf16_ps(_sum9, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[9]));
                _suma = _mm256_dpbf16_ps(_suma, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[10]));
                _sumb = _mm256_dpbf16_ps(_sumb, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[11]));
                _sumc = _mm256_dpbf16_ps(_sumc, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[12]));
                _sumd = _mm256_dpbf16_ps(_sumd, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[13]));
                _sume = _mm256_dpbf16_ps(_sume, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[14]));
                _sumf = _mm256_dpbf16_ps(_sumf, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[15]));
                pA += 16;
                pB += 32;
            }
#else // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m256 _pA = _mm256_setr_ps(bfloat16_to_float32(pA[0]), bfloat16_to_float32(pA[1]), bfloat16_to_float32(pA[2]), bfloat16_to_float32(pA[3]),
                                             bfloat16_to_float32(pA[4]), bfloat16_to_float32(pA[5]), bfloat16_to_float32(pA[6]), bfloat16_to_float32(pA[7]));

                _sum0 = _mm256_fmadd_ps(_pA, _mm256_set1_ps(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = _mm256_fmadd_ps(_pA, _mm256_set1_ps(bfloat16_to_float32(pB[1])), _sum1);
                _sum2 = _mm256_fmadd_ps(_pA, _mm256_set1_ps(bfloat16_to_float32(pB[2])), _sum2);
                _sum3 = _mm256_fmadd_ps(_pA, _mm256_set1_ps(bfloat16_to_float32(pB[3])), _sum3);
                _sum4 = _mm256_fmadd_ps(_pA, _mm256_set1_ps(bfloat16_to_float32(pB[4])), _sum4);
                _sum5 = _mm256_fmadd_ps(_pA, _mm256_set1_ps(bfloat16_to_float32(pB[5])), _sum5);
                _sum6 = _mm256_fmadd_ps(_pA, _mm256_set1_ps(bfloat16_to_float32(pB[6])), _sum6);
                _sum7 = _mm256_fmadd_ps(_pA, _mm256_set1_ps(bfloat16_to_float32(pB[7])), _sum7);
                _sum8 = _mm256_fmadd_ps(_pA, _mm256_set1_ps(bfloat16_to_float32(pB[8])), _sum8);
                _sum9 = _mm256_fmadd_ps(_pA, _mm256_set1_ps(bfloat16_to_float32(pB[9])), _sum9);
                _suma = _mm256_fmadd_ps(_pA, _mm256_set1_ps(bfloat16_to_float32(pB[10])), _suma);
                _sumb = _mm256_fmadd_ps(_pA, _mm256_set1_ps(bfloat16_to_float32(pB[11])), _sumb);
                _sumc = _mm256_fmadd_ps(_pA, _mm256_set1_ps(bfloat16_to_float32(pB[12])), _sumc);
                _sumd = _mm256_fmadd_ps(_pA, _mm256_set1_ps(bfloat16_to_float32(pB[13])), _sumd);
                _sume = _mm256_fmadd_ps(_pA, _mm256_set1_ps(bfloat16_to_float32(pB[14])), _sume);
                _sumf = _mm256_fmadd_ps(_pA, _mm256_set1_ps(bfloat16_to_float32(pB[15])), _sumf);

                pA += 8;
                pB += 16;
            }
#endif // __AVX512BF16__

            _mm256_store_ps(outptr, _sum0);
            _mm256_store_ps(outptr + 8, _sum1);
            _mm256_store_ps(outptr + 16, _sum2);
            _mm256_store_ps(outptr + 24, _sum3);
            _mm256_store_ps(outptr + 32, _sum4);
            _mm256_store_ps(outptr + 40, _sum5);
            _mm256_store_ps(outptr + 48, _sum6);
            _mm256_store_ps(outptr + 56, _sum7);
            _mm256_store_ps(outptr + 64, _sum8);
            _mm256_store_ps(outptr + 72, _sum9);
            _mm256_store_ps(outptr + 80, _suma);
            _mm256_store_ps(outptr + 88, _sumb);
            _mm256_store_ps(outptr + 96, _sumc);
            _mm256_store_ps(outptr + 104, _sumd);
            _mm256_store_ps(outptr + 112, _sume);
            _mm256_store_ps(outptr + 120, _sumf);
            outptr += 128;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

            __m256 _sum0 = _mm256_setzero_ps();
            __m256 _sum1 = _mm256_setzero_ps();
            __m256 _sum2 = _mm256_setzero_ps();
            __m256 _sum3 = _mm256_setzero_ps();
            __m256 _sum4 = _mm256_setzero_ps();
            __m256 _sum5 = _mm256_setzero_ps();
            __m256 _sum6 = _mm256_setzero_ps();
            __m256 _sum7 = _mm256_setzero_ps();

            if (k != 0)
            {
                _sum0 = _mm256_load_ps(outptr);
                _sum1 = _mm256_load_ps(outptr + 8);
                _sum2 = _mm256_load_ps(outptr + 16);
                _sum3 = _mm256_load_ps(outptr + 24);
                _sum4 = _mm256_load_ps(outptr + 32);
                _sum5 = _mm256_load_ps(outptr + 40);
                _sum6 = _mm256_load_ps(outptr + 48);
                _sum7 = _mm256_load_ps(outptr + 56);
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[0]));
                _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[1]));
                _sum2 = _mm256_dpbf16_ps(_sum2, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[2]));
                _sum3 = _mm256_dpbf16_ps(_sum3, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[3]));
                _sum4 = _mm256_dpbf16_ps(_sum4, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[4]));
                _sum5 = _mm256_dpbf16_ps(_sum5, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[5]));
                _sum6 = _mm256_dpbf16_ps(_sum6, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[6]));
                _sum7 = _mm256_dpbf16_ps(_sum7, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[7]));
                pA += 16;
                pB += 16;
            }
#else // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m256 _pA0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)pA)), 16));

                _sum0 = _mm256_fmadd_ps(_pA0, _mm256_set1_ps(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = _mm256_fmadd_ps(_pA0, _mm256_set1_ps(bfloat16_to_float32(pB[1])), _sum1);
                _sum2 = _mm256_fmadd_ps(_pA0, _mm256_set1_ps(bfloat16_to_float32(pB[2])), _sum2);
                _sum3 = _mm256_fmadd_ps(_pA0, _mm256_set1_ps(bfloat16_to_float32(pB[3])), _sum3);
                _sum4 = _mm256_fmadd_ps(_pA0, _mm256_set1_ps(bfloat16_to_float32(pB[4])), _sum4);
                _sum5 = _mm256_fmadd_ps(_pA0, _mm256_set1_ps(bfloat16_to_float32(pB[5])), _sum5);
                _sum6 = _mm256_fmadd_ps(_pA0, _mm256_set1_ps(bfloat16_to_float32(pB[6])), _sum6);
                _sum7 = _mm256_fmadd_ps(_pA0, _mm256_set1_ps(bfloat16_to_float32(pB[7])), _sum7);

                pA += 8;
                pB += 8;
            }
#endif // __AVX512BF16__

            _mm256_store_ps(outptr, _sum0);
            _mm256_store_ps(outptr + 8, _sum1);
            _mm256_store_ps(outptr + 16, _sum2);
            _mm256_store_ps(outptr + 24, _sum3);
            _mm256_store_ps(outptr + 32, _sum4);
            _mm256_store_ps(outptr + 40, _sum5);
            _mm256_store_ps(outptr + 48, _sum6);
            _mm256_store_ps(outptr + 56, _sum7);
            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

            __m256 _sum0 = _mm256_setzero_ps();
            __m256 _sum1 = _mm256_setzero_ps();
            __m256 _sum2 = _mm256_setzero_ps();
            __m256 _sum3 = _mm256_setzero_ps();

            if (k != 0)
            {
                _sum0 = _mm256_load_ps(outptr);
                _sum1 = _mm256_load_ps(outptr + 8);
                _sum2 = _mm256_load_ps(outptr + 16);
                _sum3 = _mm256_load_ps(outptr + 24);
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[0]));
                _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[1]));
                _sum2 = _mm256_dpbf16_ps(_sum2, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[2]));
                _sum3 = _mm256_dpbf16_ps(_sum3, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[3]));
                pA += 16;
                pB += 8;
            }
#else // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m256 _pA0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)pA)), 16));

                _sum0 = _mm256_fmadd_ps(_pA0, _mm256_set1_ps(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = _mm256_fmadd_ps(_pA0, _mm256_set1_ps(bfloat16_to_float32(pB[1])), _sum1);
                _sum2 = _mm256_fmadd_ps(_pA0, _mm256_set1_ps(bfloat16_to_float32(pB[2])), _sum2);
                _sum3 = _mm256_fmadd_ps(_pA0, _mm256_set1_ps(bfloat16_to_float32(pB[3])), _sum3);

                pA += 8;
                pB += 4;
            }
#endif // __AVX512BF16__

            _mm256_store_ps(outptr, _sum0);
            _mm256_store_ps(outptr + 8, _sum1);
            _mm256_store_ps(outptr + 16, _sum2);
            _mm256_store_ps(outptr + 24, _sum3);
            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* pA = pAT;

            __m256 _sum0 = _mm256_setzero_ps();
            __m256 _sum1 = _mm256_setzero_ps();

            if (k != 0)
            {
                _sum0 = _mm256_load_ps(outptr);
                _sum1 = _mm256_load_ps(outptr + 8);
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[0]));
                _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[1]));
                pA += 16;
                pB += 4;
            }
#else // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m256 _pA0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)pA)), 16));
                __m256 _pB0 = _mm256_set1_ps(bfloat16_to_float32(pB[0]));
                __m256 _pB1 = _mm256_set1_ps(bfloat16_to_float32(pB[1]));

                _sum0 = _mm256_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm256_fmadd_ps(_pA0, _pB1, _sum1);

                pA += 8;
                pB += 2;
            }
#endif // __AVX512BF16__

            _mm256_store_ps(outptr, _sum0);
            _mm256_store_ps(outptr + 8, _sum1);
            outptr += 16;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned short* pA = pAT;

            __m256 _sum0 = _mm256_setzero_ps();

            if (k != 0)
            {
                _sum0 = _mm256_load_ps(outptr);
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[0]));
                pA += 16;
                pB += 2;
            }
#else // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m256 _pA0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)pA)), 16));
                __m256 _pB0 = _mm256_set1_ps(bfloat16_to_float32(pB[0]));

                _sum0 = _mm256_fmadd_ps(_pA0, _pB0, _sum0);

                pA += 8;
                pB += 1;
            }
#endif // __AVX512BF16__

            _mm256_store_ps(outptr, _sum0);
            outptr += 8;
        }

#if __AVX512BF16__
        pAT += ((max_kk + 1) / 2 * 2) * 8;
#else
        pAT += max_kk * 8;
#endif
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
#if __AVX__
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            const unsigned short* pA = pAT;

            __m128 _sum0 = _mm_setzero_ps();
            __m128 _sum1 = _mm_setzero_ps();
            __m128 _sum2 = _mm_setzero_ps();
            __m128 _sum3 = _mm_setzero_ps();
            __m128 _sum4 = _mm_setzero_ps();
            __m128 _sum5 = _mm_setzero_ps();
            __m128 _sum6 = _mm_setzero_ps();
            __m128 _sum7 = _mm_setzero_ps();
            __m128 _sum8 = _mm_setzero_ps();
            __m128 _sum9 = _mm_setzero_ps();
            __m128 _suma = _mm_setzero_ps();
            __m128 _sumb = _mm_setzero_ps();
            __m128 _sumc = _mm_setzero_ps();
            __m128 _sumd = _mm_setzero_ps();
            __m128 _sume = _mm_setzero_ps();
            __m128 _sumf = _mm_setzero_ps();

            if (k != 0)
            {
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4);
                _sum2 = _mm_load_ps(outptr + 8);
                _sum3 = _mm_load_ps(outptr + 12);
                _sum4 = _mm_load_ps(outptr + 16);
                _sum5 = _mm_load_ps(outptr + 20);
                _sum6 = _mm_load_ps(outptr + 24);
                _sum7 = _mm_load_ps(outptr + 28);
                _sum8 = _mm_load_ps(outptr + 32);
                _sum9 = _mm_load_ps(outptr + 36);
                _suma = _mm_load_ps(outptr + 40);
                _sumb = _mm_load_ps(outptr + 44);
                _sumc = _mm_load_ps(outptr + 48);
                _sumd = _mm_load_ps(outptr + 52);
                _sume = _mm_load_ps(outptr + 56);
                _sumf = _mm_load_ps(outptr + 60);
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[0]));
                _sum1 = _mm_dpbf16_ps(_sum1, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[1]));
                _sum2 = _mm_dpbf16_ps(_sum2, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[2]));
                _sum3 = _mm_dpbf16_ps(_sum3, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[3]));
                _sum4 = _mm_dpbf16_ps(_sum4, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[4]));
                _sum5 = _mm_dpbf16_ps(_sum5, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[5]));
                _sum6 = _mm_dpbf16_ps(_sum6, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[6]));
                _sum7 = _mm_dpbf16_ps(_sum7, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[7]));
                _sum8 = _mm_dpbf16_ps(_sum8, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[8]));
                _sum9 = _mm_dpbf16_ps(_sum9, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[9]));
                _suma = _mm_dpbf16_ps(_suma, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[10]));
                _sumb = _mm_dpbf16_ps(_sumb, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[11]));
                _sumc = _mm_dpbf16_ps(_sumc, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[12]));
                _sumd = _mm_dpbf16_ps(_sumd, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[13]));
                _sume = _mm_dpbf16_ps(_sume, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[14]));
                _sumf = _mm_dpbf16_ps(_sumf, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[15]));
                pA += 8;
                pB += 32;
            }
#else // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m128 _pA = _mm_setr_ps(bfloat16_to_float32(pA[0]), bfloat16_to_float32(pA[1]), bfloat16_to_float32(pA[2]), bfloat16_to_float32(pA[3]));
                __m128 _pB0 = _mm_set1_ps(bfloat16_to_float32(pB[0]));
                __m128 _pB1 = _mm_set1_ps(bfloat16_to_float32(pB[1]));
                __m128 _pB2 = _mm_set1_ps(bfloat16_to_float32(pB[2]));
                __m128 _pB3 = _mm_set1_ps(bfloat16_to_float32(pB[3]));

                _sum0 = _mm_fmadd_ps(_pA, _pB0, _sum0);
                _sum1 = _mm_fmadd_ps(_pA, _pB1, _sum1);
                _sum2 = _mm_fmadd_ps(_pA, _pB2, _sum2);
                _sum3 = _mm_fmadd_ps(_pA, _pB3, _sum3);

                __m128 _pB4 = _mm_set1_ps(bfloat16_to_float32(pB[4]));
                __m128 _pB5 = _mm_set1_ps(bfloat16_to_float32(pB[5]));
                __m128 _pB6 = _mm_set1_ps(bfloat16_to_float32(pB[6]));
                __m128 _pB7 = _mm_set1_ps(bfloat16_to_float32(pB[7]));

                _sum4 = _mm_fmadd_ps(_pA, _pB4, _sum4);
                _sum5 = _mm_fmadd_ps(_pA, _pB5, _sum5);
                _sum6 = _mm_fmadd_ps(_pA, _pB6, _sum6);
                _sum7 = _mm_fmadd_ps(_pA, _pB7, _sum7);

                __m128 _pB8 = _mm_set1_ps(bfloat16_to_float32(pB[8]));
                __m128 _pB9 = _mm_set1_ps(bfloat16_to_float32(pB[9]));
                __m128 _pBa = _mm_set1_ps(bfloat16_to_float32(pB[10]));
                __m128 _pBb = _mm_set1_ps(bfloat16_to_float32(pB[11]));

                _sum8 = _mm_fmadd_ps(_pA, _pB8, _sum8);
                _sum9 = _mm_fmadd_ps(_pA, _pB9, _sum9);
                _suma = _mm_fmadd_ps(_pA, _pBa, _suma);
                _sumb = _mm_fmadd_ps(_pA, _pBb, _sumb);

                __m128 _pBc = _mm_set1_ps(bfloat16_to_float32(pB[12]));
                __m128 _pBd = _mm_set1_ps(bfloat16_to_float32(pB[13]));
                __m128 _pBe = _mm_set1_ps(bfloat16_to_float32(pB[14]));
                __m128 _pBf = _mm_set1_ps(bfloat16_to_float32(pB[15]));

                _sumc = _mm_fmadd_ps(_pA, _pBc, _sumc);
                _sumd = _mm_fmadd_ps(_pA, _pBd, _sumd);
                _sume = _mm_fmadd_ps(_pA, _pBe, _sume);
                _sumf = _mm_fmadd_ps(_pA, _pBf, _sumf);

                pA += 4;
                pB += 16;
            }
#endif // __AVX512BF16__

            _mm_store_ps(outptr, _sum0);
            _mm_store_ps(outptr + 4, _sum1);
            _mm_store_ps(outptr + 8, _sum2);
            _mm_store_ps(outptr + 12, _sum3);
            _mm_store_ps(outptr + 16, _sum4);
            _mm_store_ps(outptr + 20, _sum5);
            _mm_store_ps(outptr + 24, _sum6);
            _mm_store_ps(outptr + 28, _sum7);
            _mm_store_ps(outptr + 32, _sum8);
            _mm_store_ps(outptr + 36, _sum9);
            _mm_store_ps(outptr + 40, _suma);
            _mm_store_ps(outptr + 44, _sumb);
            _mm_store_ps(outptr + 48, _sumc);
            _mm_store_ps(outptr + 52, _sumd);
            _mm_store_ps(outptr + 56, _sume);
            _mm_store_ps(outptr + 60, _sumf);
            outptr += 64;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

            __m128 _sum0 = _mm_setzero_ps();
            __m128 _sum1 = _mm_setzero_ps();
            __m128 _sum2 = _mm_setzero_ps();
            __m128 _sum3 = _mm_setzero_ps();
            __m128 _sum4 = _mm_setzero_ps();
            __m128 _sum5 = _mm_setzero_ps();
            __m128 _sum6 = _mm_setzero_ps();
            __m128 _sum7 = _mm_setzero_ps();

            if (k != 0)
            {
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4);
                _sum2 = _mm_load_ps(outptr + 8);
                _sum3 = _mm_load_ps(outptr + 12);
                _sum4 = _mm_load_ps(outptr + 16);
                _sum5 = _mm_load_ps(outptr + 20);
                _sum6 = _mm_load_ps(outptr + 24);
                _sum7 = _mm_load_ps(outptr + 28);
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[0]));
                _sum1 = _mm_dpbf16_ps(_sum1, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[1]));
                _sum2 = _mm_dpbf16_ps(_sum2, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[2]));
                _sum3 = _mm_dpbf16_ps(_sum3, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[3]));
                _sum4 = _mm_dpbf16_ps(_sum4, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[4]));
                _sum5 = _mm_dpbf16_ps(_sum5, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[5]));
                _sum6 = _mm_dpbf16_ps(_sum6, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[6]));
                _sum7 = _mm_dpbf16_ps(_sum7, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[7]));
                pA += 8;
                pB += 16;
            }
#else // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m128 _pA = _mm_setr_ps(bfloat16_to_float32(pA[0]), bfloat16_to_float32(pA[1]), bfloat16_to_float32(pA[2]), bfloat16_to_float32(pA[3]));
                __m128 _pB0 = _mm_set1_ps(bfloat16_to_float32(pB[0]));
                __m128 _pB1 = _mm_set1_ps(bfloat16_to_float32(pB[1]));
                __m128 _pB2 = _mm_set1_ps(bfloat16_to_float32(pB[2]));
                __m128 _pB3 = _mm_set1_ps(bfloat16_to_float32(pB[3]));
                __m128 _pB4 = _mm_set1_ps(bfloat16_to_float32(pB[4]));
                __m128 _pB5 = _mm_set1_ps(bfloat16_to_float32(pB[5]));
                __m128 _pB6 = _mm_set1_ps(bfloat16_to_float32(pB[6]));
                __m128 _pB7 = _mm_set1_ps(bfloat16_to_float32(pB[7]));

                _sum0 = _mm_fmadd_ps(_pA, _pB0, _sum0);
                _sum1 = _mm_fmadd_ps(_pA, _pB1, _sum1);
                _sum2 = _mm_fmadd_ps(_pA, _pB2, _sum2);
                _sum3 = _mm_fmadd_ps(_pA, _pB3, _sum3);
                _sum4 = _mm_fmadd_ps(_pA, _pB4, _sum4);
                _sum5 = _mm_fmadd_ps(_pA, _pB5, _sum5);
                _sum6 = _mm_fmadd_ps(_pA, _pB6, _sum6);
                _sum7 = _mm_fmadd_ps(_pA, _pB7, _sum7);

                pA += 4;
                pB += 8;
            }
#endif // __AVX512BF16__

            _mm_store_ps(outptr, _sum0);
            _mm_store_ps(outptr + 4, _sum1);
            _mm_store_ps(outptr + 8, _sum2);
            _mm_store_ps(outptr + 12, _sum3);
            _mm_store_ps(outptr + 16, _sum4);
            _mm_store_ps(outptr + 20, _sum5);
            _mm_store_ps(outptr + 24, _sum6);
            _mm_store_ps(outptr + 28, _sum7);
            outptr += 32;
        }
#endif // __AVX__
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

            __m128 _sum0 = _mm_setzero_ps();
            __m128 _sum1 = _mm_setzero_ps();
            __m128 _sum2 = _mm_setzero_ps();
            __m128 _sum3 = _mm_setzero_ps();

            if (k != 0)
            {
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4);
                _sum2 = _mm_load_ps(outptr + 8);
                _sum3 = _mm_load_ps(outptr + 12);
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[0]));
                _sum1 = _mm_dpbf16_ps(_sum1, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[1]));
                _sum2 = _mm_dpbf16_ps(_sum2, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[2]));
                _sum3 = _mm_dpbf16_ps(_sum3, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[3]));
                pA += 8;
                pB += 8;
            }
#else // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m128 _pA = _mm_setr_ps(bfloat16_to_float32(pA[0]), bfloat16_to_float32(pA[1]), bfloat16_to_float32(pA[2]), bfloat16_to_float32(pA[3]));
                __m128 _pB0 = _mm_set1_ps(bfloat16_to_float32(pB[0]));
                __m128 _pB1 = _mm_set1_ps(bfloat16_to_float32(pB[1]));
                __m128 _pB2 = _mm_set1_ps(bfloat16_to_float32(pB[2]));
                __m128 _pB3 = _mm_set1_ps(bfloat16_to_float32(pB[3]));

                _sum0 = _mm_fmadd_ps(_pA, _pB0, _sum0);
                _sum1 = _mm_fmadd_ps(_pA, _pB1, _sum1);
                _sum2 = _mm_fmadd_ps(_pA, _pB2, _sum2);
                _sum3 = _mm_fmadd_ps(_pA, _pB3, _sum3);

                pA += 4;
                pB += 4;
            }
#endif // __AVX512BF16__

            _mm_store_ps(outptr, _sum0);
            _mm_store_ps(outptr + 4, _sum1);
            _mm_store_ps(outptr + 8, _sum2);
            _mm_store_ps(outptr + 12, _sum3);
            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* pA = pAT;

            __m128 _sum0 = _mm_setzero_ps();
            __m128 _sum1 = _mm_setzero_ps();

            if (k != 0)
            {
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4);
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[0]));
                _sum1 = _mm_dpbf16_ps(_sum1, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[1]));
                pA += 8;
                pB += 4;
            }
#else // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m128 _pA = _mm_setr_ps(bfloat16_to_float32(pA[0]), bfloat16_to_float32(pA[1]), bfloat16_to_float32(pA[2]), bfloat16_to_float32(pA[3]));
                __m128 _pB0 = _mm_set1_ps(bfloat16_to_float32(pB[0]));
                __m128 _pB1 = _mm_set1_ps(bfloat16_to_float32(pB[1]));

                _sum0 = _mm_fmadd_ps(_pA, _pB0, _sum0);
                _sum1 = _mm_fmadd_ps(_pA, _pB1, _sum1);

                pA += 4;
                pB += 2;
            }
#endif // __AVX512BF16__

            _mm_store_ps(outptr, _sum0);
            _mm_store_ps(outptr + 4, _sum1);
            outptr += 8;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned short* pA = pAT;

            __m128 _sum0 = _mm_setzero_ps();

            if (k != 0)
            {
                _sum0 = _mm_load_ps(outptr);
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[0]));
                pA += 8;
                pB += 2;
            }
#else // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m128 _pA = _mm_setr_ps(bfloat16_to_float32(pA[0]), bfloat16_to_float32(pA[1]), bfloat16_to_float32(pA[2]), bfloat16_to_float32(pA[3]));
                __m128 _pB0 = _mm_set1_ps(bfloat16_to_float32(pB[0]));

                _sum0 = _mm_fmadd_ps(_pA, _pB0, _sum0);

                pA += 4;
                pB += 1;
            }
#endif // __AVX512BF16__

            _mm_store_ps(outptr, _sum0);
            outptr += 4;
        }

#if __AVX512BF16__
        pAT += ((max_kk + 1) / 2 * 2) * 4;
#else
        pAT += max_kk * 4;
#endif
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            const unsigned short* pA = pAT;

            float sum00 = 0.f, sum01 = 0.f, sum02 = 0.f, sum03 = 0.f;
            float sum04 = 0.f, sum05 = 0.f, sum06 = 0.f, sum07 = 0.f;
            float sum08 = 0.f, sum09 = 0.f, sum0a = 0.f, sum0b = 0.f;
            float sum0c = 0.f, sum0d = 0.f, sum0e = 0.f, sum0f = 0.f;
            float sum10 = 0.f, sum11 = 0.f, sum12 = 0.f, sum13 = 0.f;
            float sum14 = 0.f, sum15 = 0.f, sum16 = 0.f, sum17 = 0.f;
            float sum18 = 0.f, sum19 = 0.f, sum1a = 0.f, sum1b = 0.f;
            float sum1c = 0.f, sum1d = 0.f, sum1e = 0.f, sum1f = 0.f;

            if (k != 0)
            {
                sum00 = outptr[0]; sum01 = outptr[1]; sum02 = outptr[2]; sum03 = outptr[3];
                sum04 = outptr[4]; sum05 = outptr[5]; sum06 = outptr[6]; sum07 = outptr[7];
                sum08 = outptr[8]; sum09 = outptr[9]; sum0a = outptr[10]; sum0b = outptr[11];
                sum0c = outptr[12]; sum0d = outptr[13]; sum0e = outptr[14]; sum0f = outptr[15];
                sum10 = outptr[16]; sum11 = outptr[17]; sum12 = outptr[18]; sum13 = outptr[19];
                sum14 = outptr[20]; sum15 = outptr[21]; sum16 = outptr[22]; sum17 = outptr[23];
                sum18 = outptr[24]; sum19 = outptr[25]; sum1a = outptr[26]; sum1b = outptr[27];
                sum1c = outptr[28]; sum1d = outptr[29]; sum1e = outptr[30]; sum1f = outptr[31];
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                float a00 = bfloat16_to_float32(pA[0]);
                float a01 = bfloat16_to_float32(pA[1]);
                float a10 = bfloat16_to_float32(pA[2]);
                float a11 = bfloat16_to_float32(pA[3]);
                float b0_k0 = bfloat16_to_float32(pB[0]);
                float b0_k1 = bfloat16_to_float32(pB[1]);
                float b1_k0 = bfloat16_to_float32(pB[2]);
                float b1_k1 = bfloat16_to_float32(pB[3]);
                float b2_k0 = bfloat16_to_float32(pB[4]);
                float b2_k1 = bfloat16_to_float32(pB[5]);
                float b3_k0 = bfloat16_to_float32(pB[6]);
                float b3_k1 = bfloat16_to_float32(pB[7]);
                float b4_k0 = bfloat16_to_float32(pB[8]);
                float b4_k1 = bfloat16_to_float32(pB[9]);
                float b5_k0 = bfloat16_to_float32(pB[10]);
                float b5_k1 = bfloat16_to_float32(pB[11]);
                float b6_k0 = bfloat16_to_float32(pB[12]);
                float b6_k1 = bfloat16_to_float32(pB[13]);
                float b7_k0 = bfloat16_to_float32(pB[14]);
                float b7_k1 = bfloat16_to_float32(pB[15]);
                float b8_k0 = bfloat16_to_float32(pB[16]);
                float b8_k1 = bfloat16_to_float32(pB[17]);
                float b9_k0 = bfloat16_to_float32(pB[18]);
                float b9_k1 = bfloat16_to_float32(pB[19]);
                float ba_k0 = bfloat16_to_float32(pB[20]);
                float ba_k1 = bfloat16_to_float32(pB[21]);
                float bb_k0 = bfloat16_to_float32(pB[22]);
                float bb_k1 = bfloat16_to_float32(pB[23]);
                float bc_k0 = bfloat16_to_float32(pB[24]);
                float bc_k1 = bfloat16_to_float32(pB[25]);
                float bd_k0 = bfloat16_to_float32(pB[26]);
                float bd_k1 = bfloat16_to_float32(pB[27]);
                float be_k0 = bfloat16_to_float32(pB[28]);
                float be_k1 = bfloat16_to_float32(pB[29]);
                float bf_k0 = bfloat16_to_float32(pB[30]);
                float bf_k1 = bfloat16_to_float32(pB[31]);
                sum00 += a00 * b0_k0 + a01 * b0_k1; sum01 += a00 * b1_k0 + a01 * b1_k1;
                sum02 += a00 * b2_k0 + a01 * b2_k1; sum03 += a00 * b3_k0 + a01 * b3_k1;
                sum04 += a00 * b4_k0 + a01 * b4_k1; sum05 += a00 * b5_k0 + a01 * b5_k1;
                sum06 += a00 * b6_k0 + a01 * b6_k1; sum07 += a00 * b7_k0 + a01 * b7_k1;
                sum08 += a00 * b8_k0 + a01 * b8_k1; sum09 += a00 * b9_k0 + a01 * b9_k1;
                sum0a += a00 * ba_k0 + a01 * ba_k1; sum0b += a00 * bb_k0 + a01 * bb_k1;
                sum0c += a00 * bc_k0 + a01 * bc_k1; sum0d += a00 * bd_k0 + a01 * bd_k1;
                sum0e += a00 * be_k0 + a01 * be_k1; sum0f += a00 * bf_k0 + a01 * bf_k1;
                sum10 += a10 * b0_k0 + a11 * b0_k1; sum11 += a10 * b1_k0 + a11 * b1_k1;
                sum12 += a10 * b2_k0 + a11 * b2_k1; sum13 += a10 * b3_k0 + a11 * b3_k1;
                sum14 += a10 * b4_k0 + a11 * b4_k1; sum15 += a10 * b5_k0 + a11 * b5_k1;
                sum16 += a10 * b6_k0 + a11 * b6_k1; sum17 += a10 * b7_k0 + a11 * b7_k1;
                sum18 += a10 * b8_k0 + a11 * b8_k1; sum19 += a10 * b9_k0 + a11 * b9_k1;
                sum1a += a10 * ba_k0 + a11 * ba_k1; sum1b += a10 * bb_k0 + a11 * bb_k1;
                sum1c += a10 * bc_k0 + a11 * bc_k1; sum1d += a10 * bd_k0 + a11 * bd_k1;
                sum1e += a10 * be_k0 + a11 * be_k1; sum1f += a10 * bf_k0 + a11 * bf_k1;
                pA += 4;
                pB += 32;
            }
#else  // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                float a1 = bfloat16_to_float32(pA[1]);
                float b0 = bfloat16_to_float32(pB[0]);
                float b1 = bfloat16_to_float32(pB[1]);
                float b2 = bfloat16_to_float32(pB[2]);
                float b3 = bfloat16_to_float32(pB[3]);
                float b4 = bfloat16_to_float32(pB[4]);
                float b5 = bfloat16_to_float32(pB[5]);
                float b6 = bfloat16_to_float32(pB[6]);
                float b7 = bfloat16_to_float32(pB[7]);
                float b8 = bfloat16_to_float32(pB[8]);
                float b9 = bfloat16_to_float32(pB[9]);
                float ba = bfloat16_to_float32(pB[10]);
                float bb = bfloat16_to_float32(pB[11]);
                float bc = bfloat16_to_float32(pB[12]);
                float bd = bfloat16_to_float32(pB[13]);
                float be = bfloat16_to_float32(pB[14]);
                float bf = bfloat16_to_float32(pB[15]);
                sum00 += a0 * b0; sum01 += a0 * b1; sum02 += a0 * b2; sum03 += a0 * b3;
                sum04 += a0 * b4; sum05 += a0 * b5; sum06 += a0 * b6; sum07 += a0 * b7;
                sum08 += a0 * b8; sum09 += a0 * b9; sum0a += a0 * ba; sum0b += a0 * bb;
                sum0c += a0 * bc; sum0d += a0 * bd; sum0e += a0 * be; sum0f += a0 * bf;
                sum10 += a1 * b0; sum11 += a1 * b1; sum12 += a1 * b2; sum13 += a1 * b3;
                sum14 += a1 * b4; sum15 += a1 * b5; sum16 += a1 * b6; sum17 += a1 * b7;
                sum18 += a1 * b8; sum19 += a1 * b9; sum1a += a1 * ba; sum1b += a1 * bb;
                sum1c += a1 * bc; sum1d += a1 * bd; sum1e += a1 * be; sum1f += a1 * bf;
                pA += 2;
                pB += 16;
            }
#endif // __AVX512BF16__

            outptr[0] = sum00; outptr[1] = sum01; outptr[2] = sum02; outptr[3] = sum03;
            outptr[4] = sum04; outptr[5] = sum05; outptr[6] = sum06; outptr[7] = sum07;
            outptr[8] = sum08; outptr[9] = sum09; outptr[10] = sum0a; outptr[11] = sum0b;
            outptr[12] = sum0c; outptr[13] = sum0d; outptr[14] = sum0e; outptr[15] = sum0f;
            outptr[16] = sum10; outptr[17] = sum11; outptr[18] = sum12; outptr[19] = sum13;
            outptr[20] = sum14; outptr[21] = sum15; outptr[22] = sum16; outptr[23] = sum17;
            outptr[24] = sum18; outptr[25] = sum19; outptr[26] = sum1a; outptr[27] = sum1b;
            outptr[28] = sum1c; outptr[29] = sum1d; outptr[30] = sum1e; outptr[31] = sum1f;
            outptr += 32;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

            float sum00 = 0.f, sum01 = 0.f, sum02 = 0.f, sum03 = 0.f;
            float sum04 = 0.f, sum05 = 0.f, sum06 = 0.f, sum07 = 0.f;
            float sum10 = 0.f, sum11 = 0.f, sum12 = 0.f, sum13 = 0.f;
            float sum14 = 0.f, sum15 = 0.f, sum16 = 0.f, sum17 = 0.f;

            if (k != 0)
            {
                sum00 = outptr[0]; sum01 = outptr[1]; sum02 = outptr[2]; sum03 = outptr[3];
                sum04 = outptr[4]; sum05 = outptr[5]; sum06 = outptr[6]; sum07 = outptr[7];
                sum10 = outptr[8]; sum11 = outptr[9]; sum12 = outptr[10]; sum13 = outptr[11];
                sum14 = outptr[12]; sum15 = outptr[13]; sum16 = outptr[14]; sum17 = outptr[15];
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                float a00 = bfloat16_to_float32(pA[0]);
                float a01 = bfloat16_to_float32(pA[1]);
                float a10 = bfloat16_to_float32(pA[2]);
                float a11 = bfloat16_to_float32(pA[3]);
                float b0_k0 = bfloat16_to_float32(pB[0]);
                float b0_k1 = bfloat16_to_float32(pB[1]);
                float b1_k0 = bfloat16_to_float32(pB[2]);
                float b1_k1 = bfloat16_to_float32(pB[3]);
                float b2_k0 = bfloat16_to_float32(pB[4]);
                float b2_k1 = bfloat16_to_float32(pB[5]);
                float b3_k0 = bfloat16_to_float32(pB[6]);
                float b3_k1 = bfloat16_to_float32(pB[7]);
                float b4_k0 = bfloat16_to_float32(pB[8]);
                float b4_k1 = bfloat16_to_float32(pB[9]);
                float b5_k0 = bfloat16_to_float32(pB[10]);
                float b5_k1 = bfloat16_to_float32(pB[11]);
                float b6_k0 = bfloat16_to_float32(pB[12]);
                float b6_k1 = bfloat16_to_float32(pB[13]);
                float b7_k0 = bfloat16_to_float32(pB[14]);
                float b7_k1 = bfloat16_to_float32(pB[15]);
                sum00 += a00 * b0_k0 + a01 * b0_k1; sum01 += a00 * b1_k0 + a01 * b1_k1;
                sum02 += a00 * b2_k0 + a01 * b2_k1; sum03 += a00 * b3_k0 + a01 * b3_k1;
                sum04 += a00 * b4_k0 + a01 * b4_k1; sum05 += a00 * b5_k0 + a01 * b5_k1;
                sum06 += a00 * b6_k0 + a01 * b6_k1; sum07 += a00 * b7_k0 + a01 * b7_k1;
                sum10 += a10 * b0_k0 + a11 * b0_k1; sum11 += a10 * b1_k0 + a11 * b1_k1;
                sum12 += a10 * b2_k0 + a11 * b2_k1; sum13 += a10 * b3_k0 + a11 * b3_k1;
                sum14 += a10 * b4_k0 + a11 * b4_k1; sum15 += a10 * b5_k0 + a11 * b5_k1;
                sum16 += a10 * b6_k0 + a11 * b6_k1; sum17 += a10 * b7_k0 + a11 * b7_k1;
                pA += 4;
                pB += 16;
            }
#else  // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                float a1 = bfloat16_to_float32(pA[1]);
                float b0 = bfloat16_to_float32(pB[0]);
                float b1 = bfloat16_to_float32(pB[1]);
                float b2 = bfloat16_to_float32(pB[2]);
                float b3 = bfloat16_to_float32(pB[3]);
                float b4 = bfloat16_to_float32(pB[4]);
                float b5 = bfloat16_to_float32(pB[5]);
                float b6 = bfloat16_to_float32(pB[6]);
                float b7 = bfloat16_to_float32(pB[7]);
                sum00 += a0 * b0; sum01 += a0 * b1; sum02 += a0 * b2; sum03 += a0 * b3;
                sum04 += a0 * b4; sum05 += a0 * b5; sum06 += a0 * b6; sum07 += a0 * b7;
                sum10 += a1 * b0; sum11 += a1 * b1; sum12 += a1 * b2; sum13 += a1 * b3;
                sum14 += a1 * b4; sum15 += a1 * b5; sum16 += a1 * b6; sum17 += a1 * b7;
                pA += 2;
                pB += 8;
            }
#endif // __AVX512BF16__

            outptr[0] = sum00; outptr[1] = sum01; outptr[2] = sum02; outptr[3] = sum03;
            outptr[4] = sum04; outptr[5] = sum05; outptr[6] = sum06; outptr[7] = sum07;
            outptr[8] = sum10; outptr[9] = sum11; outptr[10] = sum12; outptr[11] = sum13;
            outptr[12] = sum14; outptr[13] = sum15; outptr[14] = sum16; outptr[15] = sum17;
            outptr += 16;
        }
#endif // __AVX__
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

            float sum00 = 0.f, sum01 = 0.f, sum02 = 0.f, sum03 = 0.f;
            float sum10 = 0.f, sum11 = 0.f, sum12 = 0.f, sum13 = 0.f;

            if (k != 0)
            {
                sum00 = outptr[0]; sum01 = outptr[1]; sum02 = outptr[2]; sum03 = outptr[3];
                sum10 = outptr[4]; sum11 = outptr[5]; sum12 = outptr[6]; sum13 = outptr[7];
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                float a00 = bfloat16_to_float32(pA[0]);
                float a01 = bfloat16_to_float32(pA[1]);
                float a10 = bfloat16_to_float32(pA[2]);
                float a11 = bfloat16_to_float32(pA[3]);
                float b0_k0 = bfloat16_to_float32(pB[0]);
                float b0_k1 = bfloat16_to_float32(pB[1]);
                float b1_k0 = bfloat16_to_float32(pB[2]);
                float b1_k1 = bfloat16_to_float32(pB[3]);
                float b2_k0 = bfloat16_to_float32(pB[4]);
                float b2_k1 = bfloat16_to_float32(pB[5]);
                float b3_k0 = bfloat16_to_float32(pB[6]);
                float b3_k1 = bfloat16_to_float32(pB[7]);
                sum00 += a00 * b0_k0 + a01 * b0_k1; sum01 += a00 * b1_k0 + a01 * b1_k1;
                sum02 += a00 * b2_k0 + a01 * b2_k1; sum03 += a00 * b3_k0 + a01 * b3_k1;
                sum10 += a10 * b0_k0 + a11 * b0_k1; sum11 += a10 * b1_k0 + a11 * b1_k1;
                sum12 += a10 * b2_k0 + a11 * b2_k1; sum13 += a10 * b3_k0 + a11 * b3_k1;
                pA += 4;
                pB += 8;
            }
#else  // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                float a1 = bfloat16_to_float32(pA[1]);
                float b0 = bfloat16_to_float32(pB[0]);
                float b1 = bfloat16_to_float32(pB[1]);
                float b2 = bfloat16_to_float32(pB[2]);
                float b3 = bfloat16_to_float32(pB[3]);
                sum00 += a0 * b0; sum01 += a0 * b1; sum02 += a0 * b2; sum03 += a0 * b3;
                sum10 += a1 * b0; sum11 += a1 * b1; sum12 += a1 * b2; sum13 += a1 * b3;
                pA += 2;
                pB += 4;
            }
#endif // __AVX512BF16__

            outptr[0] = sum00; outptr[1] = sum01; outptr[2] = sum02; outptr[3] = sum03;
            outptr[4] = sum10; outptr[5] = sum11; outptr[6] = sum12; outptr[7] = sum13;
            outptr += 8;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* pA = pAT;

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

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                float a00 = bfloat16_to_float32(pA[0]);
                float a01 = bfloat16_to_float32(pA[1]);
                float a10 = bfloat16_to_float32(pA[2]);
                float a11 = bfloat16_to_float32(pA[3]);
                float b0_k0 = bfloat16_to_float32(pB[0]);
                float b0_k1 = bfloat16_to_float32(pB[1]);
                float b1_k0 = bfloat16_to_float32(pB[2]);
                float b1_k1 = bfloat16_to_float32(pB[3]);
                sum00 += a00 * b0_k0 + a01 * b0_k1;
                sum01 += a00 * b1_k0 + a01 * b1_k1;
                sum10 += a10 * b0_k0 + a11 * b0_k1;
                sum11 += a10 * b1_k0 + a11 * b1_k1;
                pA += 4;
                pB += 4;
            }
#else  // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                float a1 = bfloat16_to_float32(pA[1]);
                float b0 = bfloat16_to_float32(pB[0]);
                float b1 = bfloat16_to_float32(pB[1]);
                sum00 += a0 * b0;
                sum01 += a0 * b1;
                sum10 += a1 * b0;
                sum11 += a1 * b1;
                pA += 2;
                pB += 2;
            }
#endif // __AVX512BF16__

            outptr[0] = sum00;
            outptr[1] = sum01;
            outptr[2] = sum10;
            outptr[3] = sum11;
            outptr += 4;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned short* pA = pAT;

            float sum0 = 0.f;
            float sum1 = 0.f;

            if (k != 0)
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                float a00 = bfloat16_to_float32(pA[0]);
                float a01 = bfloat16_to_float32(pA[1]);
                float a10 = bfloat16_to_float32(pA[2]);
                float a11 = bfloat16_to_float32(pA[3]);
                float b0_k0 = bfloat16_to_float32(pB[0]);
                float b0_k1 = bfloat16_to_float32(pB[1]);
                sum0 += a00 * b0_k0 + a01 * b0_k1;
                sum1 += a10 * b0_k0 + a11 * b0_k1;
                pA += 4;
                pB += 2;
            }
#else  // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                sum0 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);
                sum1 += bfloat16_to_float32(pA[1]) * bfloat16_to_float32(pB[0]);
                pA += 2;
                pB += 1;
            }
#endif // __AVX512BF16__

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
        }

#if __AVX512BF16__
        pAT += ((max_kk + 1) / 2 * 2) * 2;
#else
        pAT += max_kk * 2;
#endif
    }
    for (; ii < max_ii; ii++)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            const unsigned short* pA = pAT;

            float sum0 = 0.f;
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;
            float sum4 = 0.f;
            float sum5 = 0.f;
            float sum6 = 0.f;
            float sum7 = 0.f;
            float sum8 = 0.f;
            float sum9 = 0.f;
            float suma = 0.f;
            float sumb = 0.f;
            float sumc = 0.f;
            float sumd = 0.f;
            float sume = 0.f;
            float sumf = 0.f;

            if (k != 0)
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
                sum2 = outptr[2];
                sum3 = outptr[3];
                sum4 = outptr[4];
                sum5 = outptr[5];
                sum6 = outptr[6];
                sum7 = outptr[7];
                sum8 = outptr[8];
                sum9 = outptr[9];
                suma = outptr[10];
                sumb = outptr[11];
                sumc = outptr[12];
                sumd = outptr[13];
                sume = outptr[14];
                sumf = outptr[15];
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                float a1 = bfloat16_to_float32(pA[1]);
                float b0_k0 = bfloat16_to_float32(pB[0]);
                float b0_k1 = bfloat16_to_float32(pB[1]);
                float b1_k0 = bfloat16_to_float32(pB[2]);
                float b1_k1 = bfloat16_to_float32(pB[3]);
                float b2_k0 = bfloat16_to_float32(pB[4]);
                float b2_k1 = bfloat16_to_float32(pB[5]);
                float b3_k0 = bfloat16_to_float32(pB[6]);
                float b3_k1 = bfloat16_to_float32(pB[7]);
                float b4_k0 = bfloat16_to_float32(pB[8]);
                float b4_k1 = bfloat16_to_float32(pB[9]);
                float b5_k0 = bfloat16_to_float32(pB[10]);
                float b5_k1 = bfloat16_to_float32(pB[11]);
                float b6_k0 = bfloat16_to_float32(pB[12]);
                float b6_k1 = bfloat16_to_float32(pB[13]);
                float b7_k0 = bfloat16_to_float32(pB[14]);
                float b7_k1 = bfloat16_to_float32(pB[15]);
                float b8_k0 = bfloat16_to_float32(pB[16]);
                float b8_k1 = bfloat16_to_float32(pB[17]);
                float b9_k0 = bfloat16_to_float32(pB[18]);
                float b9_k1 = bfloat16_to_float32(pB[19]);
                float ba_k0 = bfloat16_to_float32(pB[20]);
                float ba_k1 = bfloat16_to_float32(pB[21]);
                float bb_k0 = bfloat16_to_float32(pB[22]);
                float bb_k1 = bfloat16_to_float32(pB[23]);
                float bc_k0 = bfloat16_to_float32(pB[24]);
                float bc_k1 = bfloat16_to_float32(pB[25]);
                float bd_k0 = bfloat16_to_float32(pB[26]);
                float bd_k1 = bfloat16_to_float32(pB[27]);
                float be_k0 = bfloat16_to_float32(pB[28]);
                float be_k1 = bfloat16_to_float32(pB[29]);
                float bf_k0 = bfloat16_to_float32(pB[30]);
                float bf_k1 = bfloat16_to_float32(pB[31]);
                sum0 += a0 * b0_k0 + a1 * b0_k1;
                sum1 += a0 * b1_k0 + a1 * b1_k1;
                sum2 += a0 * b2_k0 + a1 * b2_k1;
                sum3 += a0 * b3_k0 + a1 * b3_k1;
                sum4 += a0 * b4_k0 + a1 * b4_k1;
                sum5 += a0 * b5_k0 + a1 * b5_k1;
                sum6 += a0 * b6_k0 + a1 * b6_k1;
                sum7 += a0 * b7_k0 + a1 * b7_k1;
                sum8 += a0 * b8_k0 + a1 * b8_k1;
                sum9 += a0 * b9_k0 + a1 * b9_k1;
                suma += a0 * ba_k0 + a1 * ba_k1;
                sumb += a0 * bb_k0 + a1 * bb_k1;
                sumc += a0 * bc_k0 + a1 * bc_k1;
                sumd += a0 * bd_k0 + a1 * bd_k1;
                sume += a0 * be_k0 + a1 * be_k1;
                sumf += a0 * bf_k0 + a1 * bf_k1;
                pA += 2;
                pB += 32;
            }
#else  // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                sum0 += a0 * bfloat16_to_float32(pB[0]);
                sum1 += a0 * bfloat16_to_float32(pB[1]);
                sum2 += a0 * bfloat16_to_float32(pB[2]);
                sum3 += a0 * bfloat16_to_float32(pB[3]);
                sum4 += a0 * bfloat16_to_float32(pB[4]);
                sum5 += a0 * bfloat16_to_float32(pB[5]);
                sum6 += a0 * bfloat16_to_float32(pB[6]);
                sum7 += a0 * bfloat16_to_float32(pB[7]);
                sum8 += a0 * bfloat16_to_float32(pB[8]);
                sum9 += a0 * bfloat16_to_float32(pB[9]);
                suma += a0 * bfloat16_to_float32(pB[10]);
                sumb += a0 * bfloat16_to_float32(pB[11]);
                sumc += a0 * bfloat16_to_float32(pB[12]);
                sumd += a0 * bfloat16_to_float32(pB[13]);
                sume += a0 * bfloat16_to_float32(pB[14]);
                sumf += a0 * bfloat16_to_float32(pB[15]);
                pA += 1;
                pB += 16;
            }
#endif // __AVX512BF16__

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr[2] = sum2;
            outptr[3] = sum3;
            outptr[4] = sum4;
            outptr[5] = sum5;
            outptr[6] = sum6;
            outptr[7] = sum7;
            outptr[8] = sum8;
            outptr[9] = sum9;
            outptr[10] = suma;
            outptr[11] = sumb;
            outptr[12] = sumc;
            outptr[13] = sumd;
            outptr[14] = sume;
            outptr[15] = sumf;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

            float sum0 = 0.f;
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;
            float sum4 = 0.f;
            float sum5 = 0.f;
            float sum6 = 0.f;
            float sum7 = 0.f;

            if (k != 0)
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

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                float a1 = bfloat16_to_float32(pA[1]);
                float b0_k0 = bfloat16_to_float32(pB[0]);
                float b0_k1 = bfloat16_to_float32(pB[1]);
                float b1_k0 = bfloat16_to_float32(pB[2]);
                float b1_k1 = bfloat16_to_float32(pB[3]);
                float b2_k0 = bfloat16_to_float32(pB[4]);
                float b2_k1 = bfloat16_to_float32(pB[5]);
                float b3_k0 = bfloat16_to_float32(pB[6]);
                float b3_k1 = bfloat16_to_float32(pB[7]);
                float b4_k0 = bfloat16_to_float32(pB[8]);
                float b4_k1 = bfloat16_to_float32(pB[9]);
                float b5_k0 = bfloat16_to_float32(pB[10]);
                float b5_k1 = bfloat16_to_float32(pB[11]);
                float b6_k0 = bfloat16_to_float32(pB[12]);
                float b6_k1 = bfloat16_to_float32(pB[13]);
                float b7_k0 = bfloat16_to_float32(pB[14]);
                float b7_k1 = bfloat16_to_float32(pB[15]);
                sum0 += a0 * b0_k0 + a1 * b0_k1;
                sum1 += a0 * b1_k0 + a1 * b1_k1;
                sum2 += a0 * b2_k0 + a1 * b2_k1;
                sum3 += a0 * b3_k0 + a1 * b3_k1;
                sum4 += a0 * b4_k0 + a1 * b4_k1;
                sum5 += a0 * b5_k0 + a1 * b5_k1;
                sum6 += a0 * b6_k0 + a1 * b6_k1;
                sum7 += a0 * b7_k0 + a1 * b7_k1;
                pA += 2;
                pB += 16;
            }
#else  // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                sum0 += a0 * bfloat16_to_float32(pB[0]);
                sum1 += a0 * bfloat16_to_float32(pB[1]);
                sum2 += a0 * bfloat16_to_float32(pB[2]);
                sum3 += a0 * bfloat16_to_float32(pB[3]);
                sum4 += a0 * bfloat16_to_float32(pB[4]);
                sum5 += a0 * bfloat16_to_float32(pB[5]);
                sum6 += a0 * bfloat16_to_float32(pB[6]);
                sum7 += a0 * bfloat16_to_float32(pB[7]);
                pA += 1;
                pB += 8;
            }
#endif // __AVX512BF16__

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
#endif // __AVX__
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

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

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                float a1 = bfloat16_to_float32(pA[1]);
                float b0_k0 = bfloat16_to_float32(pB[0]);
                float b0_k1 = bfloat16_to_float32(pB[1]);
                float b1_k0 = bfloat16_to_float32(pB[2]);
                float b1_k1 = bfloat16_to_float32(pB[3]);
                float b2_k0 = bfloat16_to_float32(pB[4]);
                float b2_k1 = bfloat16_to_float32(pB[5]);
                float b3_k0 = bfloat16_to_float32(pB[6]);
                float b3_k1 = bfloat16_to_float32(pB[7]);
                sum0 += a0 * b0_k0 + a1 * b0_k1;
                sum1 += a0 * b1_k0 + a1 * b1_k1;
                sum2 += a0 * b2_k0 + a1 * b2_k1;
                sum3 += a0 * b3_k0 + a1 * b3_k1;
                pA += 2;
                pB += 8;
            }
#else  // __AVX512BF16__
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
#endif // __AVX512BF16__

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr[2] = sum2;
            outptr[3] = sum3;
            outptr += 4;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* pA = pAT;

            float sum0 = 0.f;
            float sum1 = 0.f;

            if (k != 0)
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                float a1 = bfloat16_to_float32(pA[1]);
                float b0_k0 = bfloat16_to_float32(pB[0]);
                float b0_k1 = bfloat16_to_float32(pB[1]);
                float b1_k0 = bfloat16_to_float32(pB[2]);
                float b1_k1 = bfloat16_to_float32(pB[3]);
                sum0 += a0 * b0_k0 + a1 * b0_k1;
                sum1 += a0 * b1_k0 + a1 * b1_k1;
                pA += 2;
                pB += 4;
            }
#else  // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                sum0 += a0 * bfloat16_to_float32(pB[0]);
                sum1 += a0 * bfloat16_to_float32(pB[1]);
                pA += 1;
                pB += 2;
            }
#endif // __AVX512BF16__

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned short* pA = pAT;

            float sum = 0.f;

            if (k != 0)
            {
                sum = outptr[0];
            }

            int kk = 0;
#if __AVX512BF16__
            int max_kk_e2 = (max_kk + 1) / 2 * 2;
            for (; kk < max_kk_e2; kk += 2)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                float a1 = bfloat16_to_float32(pA[1]);
                float b0_k0 = bfloat16_to_float32(pB[0]);
                float b0_k1 = bfloat16_to_float32(pB[1]);
                sum += a0 * b0_k0 + a1 * b0_k1;
                pA += 2;
                pB += 2;
            }
#else  // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                sum += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);
                pA += 1;
                pB += 1;
            }
#endif // __AVX512BF16__

            outptr[0] = sum;
            outptr += 1;
        }

#if __AVX512BF16__
        pAT += ((max_kk + 1) / 2 * 2) * 1;
#else
        pAT += max_kk * 1;
#endif
    }
}

static void unpack_output_tile_fp32_to_bf16(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, int output_transpose)
{
    // topT contains fp32 accumulated results in column order from gemm_transB_packed_tile_bf16
    // Each register is one column (16/8/4 rows), no deshuffling needed
    // This function applies C bias, multiplies alpha, and stores as bf16
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const float* pp = topT;

    const float* pC = (const float*)C;
    int c_hstep = 0;
    int c_elempack = 0;
    if (pC)
    {
        c_hstep = C.dims == 3 ? (int)C.cstep : C.w;
        c_elempack = C.elempack;
        if (broadcast_type_C == 3)
        {
            pC = (const float*)C + (i * c_hstep + j) * c_elempack;
        }
        else if (broadcast_type_C == 1 || broadcast_type_C == 2)
        {
            pC = (const float*)C + i;
        }
        else if (broadcast_type_C == 4)
        {
            pC = (const float*)C + j;
        }
    }

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        unsigned short* p0;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        __m512 _descale = _mm512_set1_ps(1.f);
        (void)_descale;

        __m512 _c0 = _mm512_setzero_ps();
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = _mm512_set1_ps(pC[0]);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                _c0 = _mm512_loadu_ps(pC);
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _f0 = _mm512_load_ps(pp);
            __m512 _f1 = _mm512_load_ps(pp + 16);
            __m512 _f2 = _mm512_load_ps(pp + 32);
            __m512 _f3 = _mm512_load_ps(pp + 48);
            __m512 _f4 = _mm512_load_ps(pp + 64);
            __m512 _f5 = _mm512_load_ps(pp + 80);
            __m512 _f6 = _mm512_load_ps(pp + 96);
            __m512 _f7 = _mm512_load_ps(pp + 112);
            __m512 _f8 = _mm512_load_ps(pp + 128);
            __m512 _f9 = _mm512_load_ps(pp + 128 + 16);
            __m512 _fa = _mm512_load_ps(pp + 128 + 32);
            __m512 _fb = _mm512_load_ps(pp + 128 + 48);
            __m512 _fc = _mm512_load_ps(pp + 128 + 64);
            __m512 _fd = _mm512_load_ps(pp + 128 + 80);
            __m512 _fe = _mm512_load_ps(pp + 128 + 96);
            __m512 _ff = _mm512_load_ps(pp + 128 + 112);
            pp += 256;

#if __AVX512BF16__
            // deshuffle from the shuffle-based 16x16 dpbf16_ps kernel
            {
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm512_permute_ps(_f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm512_permute_ps(_f7, _MM_SHUFFLE(2, 1, 0, 3));
                _f9 = _mm512_permute_ps(_f9, _MM_SHUFFLE(2, 1, 0, 3));
                _fb = _mm512_permute_ps(_fb, _MM_SHUFFLE(2, 1, 0, 3));
                _fd = _mm512_permute_ps(_fd, _MM_SHUFFLE(2, 1, 0, 3));
                _ff = _mm512_permute_ps(_ff, _MM_SHUFFLE(2, 1, 0, 3));

                __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f3);
                __m512 _tmp1 = _mm512_unpackhi_ps(_f0, _f3);
                __m512 _tmp2 = _mm512_unpacklo_ps(_f2, _f1);
                __m512 _tmp3 = _mm512_unpackhi_ps(_f2, _f1);
                __m512 _tmp4 = _mm512_unpacklo_ps(_f4, _f7);
                __m512 _tmp5 = _mm512_unpackhi_ps(_f4, _f7);
                __m512 _tmp6 = _mm512_unpacklo_ps(_f6, _f5);
                __m512 _tmp7 = _mm512_unpackhi_ps(_f6, _f5);
                __m512 _tmp8 = _mm512_unpacklo_ps(_f8, _fb);
                __m512 _tmp9 = _mm512_unpackhi_ps(_f8, _fb);
                __m512 _tmpa = _mm512_unpacklo_ps(_fa, _f9);
                __m512 _tmpb = _mm512_unpackhi_ps(_fa, _f9);
                __m512 _tmpc = _mm512_unpacklo_ps(_fc, _ff);
                __m512 _tmpd = _mm512_unpackhi_ps(_fc, _ff);
                __m512 _tmpe = _mm512_unpacklo_ps(_fe, _fd);
                __m512 _tmpf = _mm512_unpackhi_ps(_fe, _fd);

                _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _f2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _f4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                _f5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                _f6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));
                _f7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));
                _f8 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp8), _mm512_castps_pd(_tmpa)));
                _f9 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp8), _mm512_castps_pd(_tmpa)));
                _fa = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmpb), _mm512_castps_pd(_tmp9)));
                _fb = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmpb), _mm512_castps_pd(_tmp9)));
                _fc = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmpc), _mm512_castps_pd(_tmpe)));
                _fd = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmpc), _mm512_castps_pd(_tmpe)));
                _fe = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmpf), _mm512_castps_pd(_tmpd)));
                _ff = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmpf), _mm512_castps_pd(_tmpd)));

                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm512_permute_ps(_f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm512_permute_ps(_f7, _MM_SHUFFLE(2, 1, 0, 3));
                _f9 = _mm512_permute_ps(_f9, _MM_SHUFFLE(2, 1, 0, 3));
                _fb = _mm512_permute_ps(_fb, _MM_SHUFFLE(2, 1, 0, 3));
                _fd = _mm512_permute_ps(_fd, _MM_SHUFFLE(2, 1, 0, 3));
                _ff = _mm512_permute_ps(_ff, _MM_SHUFFLE(2, 1, 0, 3));

                _tmp0 = _mm512_shuffle_f32x4(_f0, _f8, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp1 = _mm512_shuffle_f32x4(_f1, _f9, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp2 = _mm512_shuffle_f32x4(_f2, _fa, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp3 = _mm512_shuffle_f32x4(_f3, _fb, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp4 = _mm512_shuffle_f32x4(_f8, _f0, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp5 = _mm512_shuffle_f32x4(_f9, _f1, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp6 = _mm512_shuffle_f32x4(_fa, _f2, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp7 = _mm512_shuffle_f32x4(_fb, _f3, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp8 = _mm512_shuffle_f32x4(_f4, _fc, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp9 = _mm512_shuffle_f32x4(_f5, _fd, _MM_SHUFFLE(2, 0, 2, 0));
                _tmpa = _mm512_shuffle_f32x4(_f6, _fe, _MM_SHUFFLE(2, 0, 2, 0));
                _tmpb = _mm512_shuffle_f32x4(_f7, _ff, _MM_SHUFFLE(2, 0, 2, 0));
                _tmpc = _mm512_shuffle_f32x4(_fc, _f4, _MM_SHUFFLE(3, 1, 3, 1));
                _tmpd = _mm512_shuffle_f32x4(_fd, _f5, _MM_SHUFFLE(3, 1, 3, 1));
                _tmpe = _mm512_shuffle_f32x4(_fe, _f6, _MM_SHUFFLE(3, 1, 3, 1));
                _tmpf = _mm512_shuffle_f32x4(_ff, _f7, _MM_SHUFFLE(3, 1, 3, 1));

                _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp8, _MM_SHUFFLE(3, 1, 2, 0));
                _f1 = _mm512_shuffle_f32x4(_tmp1, _tmp9, _MM_SHUFFLE(3, 1, 2, 0));
                _f2 = _mm512_shuffle_f32x4(_tmp2, _tmpa, _MM_SHUFFLE(3, 1, 2, 0));
                _f3 = _mm512_shuffle_f32x4(_tmp3, _tmpb, _MM_SHUFFLE(3, 1, 2, 0));
                _f4 = _mm512_shuffle_f32x4(_tmp4, _tmpc, _MM_SHUFFLE(3, 1, 2, 0));
                _f5 = _mm512_shuffle_f32x4(_tmp5, _tmpd, _MM_SHUFFLE(3, 1, 2, 0));
                _f6 = _mm512_shuffle_f32x4(_tmp6, _tmpe, _MM_SHUFFLE(3, 1, 2, 0));
                _f7 = _mm512_shuffle_f32x4(_tmp7, _tmpf, _MM_SHUFFLE(3, 1, 2, 0));
                _f8 = _mm512_shuffle_f32x4(_tmp8, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                _f9 = _mm512_shuffle_f32x4(_tmp9, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                _fa = _mm512_shuffle_f32x4(_tmpa, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                _fb = _mm512_shuffle_f32x4(_tmpb, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                _fc = _mm512_shuffle_f32x4(_tmpc, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                _fd = _mm512_shuffle_f32x4(_tmpd, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                _fe = _mm512_shuffle_f32x4(_tmpe, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                _ff = _mm512_shuffle_f32x4(_tmpf, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));
            }
#endif // __AVX512BF16__

            if (broadcast_type_C == 3)
            {
                __m512 _c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _ca, _cb, _cc, _cd, _ce, _cf;
                if (c_elempack == 16)
                {
                    _c0 = _mm512_loadu_ps(pC);
                    _c1 = _mm512_loadu_ps(pC + 16);
                    _c2 = _mm512_loadu_ps(pC + 32);
                    _c3 = _mm512_loadu_ps(pC + 48);
                    _c4 = _mm512_loadu_ps(pC + 64);
                    _c5 = _mm512_loadu_ps(pC + 80);
                    _c6 = _mm512_loadu_ps(pC + 96);
                    _c7 = _mm512_loadu_ps(pC + 112);
                    _c8 = _mm512_loadu_ps(pC + 128);
                    _c9 = _mm512_loadu_ps(pC + 128 + 16);
                    _ca = _mm512_loadu_ps(pC + 128 + 32);
                    _cb = _mm512_loadu_ps(pC + 128 + 48);
                    _cc = _mm512_loadu_ps(pC + 128 + 64);
                    _cd = _mm512_loadu_ps(pC + 128 + 80);
                    _ce = _mm512_loadu_ps(pC + 128 + 96);
                    _cf = _mm512_loadu_ps(pC + 128 + 112);
                    pC += 256;
                }
                else // c_elempack == 1 or 4 or 8
                {
                    _c0 = _mm512_setzero_ps();
                    _c1 = _mm512_setzero_ps();
                    _c2 = _mm512_setzero_ps();
                    _c3 = _mm512_setzero_ps();
                    _c4 = _mm512_setzero_ps();
                    _c5 = _mm512_setzero_ps();
                    _c6 = _mm512_setzero_ps();
                    _c7 = _mm512_setzero_ps();
                    _c8 = _mm512_setzero_ps();
                    _c9 = _mm512_setzero_ps();
                    _ca = _mm512_setzero_ps();
                    _cb = _mm512_setzero_ps();
                    _cc = _mm512_setzero_ps();
                    _cd = _mm512_setzero_ps();
                    _ce = _mm512_setzero_ps();
                    _cf = _mm512_setzero_ps();
                    for (int t = 0; t < 16; t++)
                    {
                        ((float*)&_c0)[t] = pC[(t / c_elempack) * c_hstep * c_elempack + 0 * c_elempack + t % c_elempack];
                        ((float*)&_c1)[t] = pC[(t / c_elempack) * c_hstep * c_elempack + 1 * c_elempack + t % c_elempack];
                        ((float*)&_c2)[t] = pC[(t / c_elempack) * c_hstep * c_elempack + 2 * c_elempack + t % c_elempack];
                        ((float*)&_c3)[t] = pC[(t / c_elempack) * c_hstep * c_elempack + 3 * c_elempack + t % c_elempack];
                        ((float*)&_c4)[t] = pC[(t / c_elempack) * c_hstep * c_elempack + 4 * c_elempack + t % c_elempack];
                        ((float*)&_c5)[t] = pC[(t / c_elempack) * c_hstep * c_elempack + 5 * c_elempack + t % c_elempack];
                        ((float*)&_c6)[t] = pC[(t / c_elempack) * c_hstep * c_elempack + 6 * c_elempack + t % c_elempack];
                        ((float*)&_c7)[t] = pC[(t / c_elempack) * c_hstep * c_elempack + 7 * c_elempack + t % c_elempack];
                        ((float*)&_c8)[t] = pC[(t / c_elempack) * c_hstep * c_elempack + 8 * c_elempack + t % c_elempack];
                        ((float*)&_c9)[t] = pC[(t / c_elempack) * c_hstep * c_elempack + 9 * c_elempack + t % c_elempack];
                        ((float*)&_ca)[t] = pC[(t / c_elempack) * c_hstep * c_elempack + 10 * c_elempack + t % c_elempack];
                        ((float*)&_cb)[t] = pC[(t / c_elempack) * c_hstep * c_elempack + 11 * c_elempack + t % c_elempack];
                        ((float*)&_cc)[t] = pC[(t / c_elempack) * c_hstep * c_elempack + 12 * c_elempack + t % c_elempack];
                        ((float*)&_cd)[t] = pC[(t / c_elempack) * c_hstep * c_elempack + 13 * c_elempack + t % c_elempack];
                        ((float*)&_ce)[t] = pC[(t / c_elempack) * c_hstep * c_elempack + 14 * c_elempack + t % c_elempack];
                        ((float*)&_cf)[t] = pC[(t / c_elempack) * c_hstep * c_elempack + 15 * c_elempack + t % c_elempack];
                    }
                    pC += 16 * c_elempack;
                }
                _f0 = _mm512_add_ps(_f0, _c0);
                _f1 = _mm512_add_ps(_f1, _c1);
                _f2 = _mm512_add_ps(_f2, _c2);
                _f3 = _mm512_add_ps(_f3, _c3);
                _f4 = _mm512_add_ps(_f4, _c4);
                _f5 = _mm512_add_ps(_f5, _c5);
                _f6 = _mm512_add_ps(_f6, _c6);
                _f7 = _mm512_add_ps(_f7, _c7);
                _f8 = _mm512_add_ps(_f8, _c8);
                _f9 = _mm512_add_ps(_f9, _c9);
                _fa = _mm512_add_ps(_fa, _ca);
                _fb = _mm512_add_ps(_fb, _cb);
                _fc = _mm512_add_ps(_fc, _cc);
                _fd = _mm512_add_ps(_fd, _cd);
                _fe = _mm512_add_ps(_fe, _ce);
                _ff = _mm512_add_ps(_ff, _cf);
            }
            else if (broadcast_type_C == 4)
            {
                _f0 = _mm512_add_ps(_f0, _mm512_set1_ps(pC[0]));
                _f1 = _mm512_add_ps(_f1, _mm512_set1_ps(pC[1]));
                _f2 = _mm512_add_ps(_f2, _mm512_set1_ps(pC[2]));
                _f3 = _mm512_add_ps(_f3, _mm512_set1_ps(pC[3]));
                _f4 = _mm512_add_ps(_f4, _mm512_set1_ps(pC[4]));
                _f5 = _mm512_add_ps(_f5, _mm512_set1_ps(pC[5]));
                _f6 = _mm512_add_ps(_f6, _mm512_set1_ps(pC[6]));
                _f7 = _mm512_add_ps(_f7, _mm512_set1_ps(pC[7]));
                _f8 = _mm512_add_ps(_f8, _mm512_set1_ps(pC[8]));
                _f9 = _mm512_add_ps(_f9, _mm512_set1_ps(pC[9]));
                _fa = _mm512_add_ps(_fa, _mm512_set1_ps(pC[10]));
                _fb = _mm512_add_ps(_fb, _mm512_set1_ps(pC[11]));
                _fc = _mm512_add_ps(_fc, _mm512_set1_ps(pC[12]));
                _fd = _mm512_add_ps(_fd, _mm512_set1_ps(pC[13]));
                _fe = _mm512_add_ps(_fe, _mm512_set1_ps(pC[14]));
                _ff = _mm512_add_ps(_ff, _mm512_set1_ps(pC[15]));
                pC += 16;
            }
            else
            {
                _f0 = _mm512_add_ps(_f0, _c0);
                _f1 = _mm512_add_ps(_f1, _c0);
                _f2 = _mm512_add_ps(_f2, _c0);
                _f3 = _mm512_add_ps(_f3, _c0);
                _f4 = _mm512_add_ps(_f4, _c0);
                _f5 = _mm512_add_ps(_f5, _c0);
                _f6 = _mm512_add_ps(_f6, _c0);
                _f7 = _mm512_add_ps(_f7, _c0);
                _f8 = _mm512_add_ps(_f8, _c0);
                _f9 = _mm512_add_ps(_f9, _c0);
                _fa = _mm512_add_ps(_fa, _c0);
                _fb = _mm512_add_ps(_fb, _c0);
                _fc = _mm512_add_ps(_fc, _c0);
                _fd = _mm512_add_ps(_fd, _c0);
                _fe = _mm512_add_ps(_fe, _c0);
                _ff = _mm512_add_ps(_ff, _c0);
            }

            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
                _f2 = _mm512_mul_ps(_f2, _alpha);
                _f3 = _mm512_mul_ps(_f3, _alpha);
                _f4 = _mm512_mul_ps(_f4, _alpha);
                _f5 = _mm512_mul_ps(_f5, _alpha);
                _f6 = _mm512_mul_ps(_f6, _alpha);
                _f7 = _mm512_mul_ps(_f7, _alpha);
                _f8 = _mm512_mul_ps(_f8, _alpha);
                _f9 = _mm512_mul_ps(_f9, _alpha);
                _fa = _mm512_mul_ps(_fa, _alpha);
                _fb = _mm512_mul_ps(_fb, _alpha);
                _fc = _mm512_mul_ps(_fc, _alpha);
                _fd = _mm512_mul_ps(_fd, _alpha);
                _fe = _mm512_mul_ps(_fe, _alpha);
                _ff = _mm512_mul_ps(_ff, _alpha);
            }

            // store bf16
            if (output_transpose)
            {
                _mm256_storeu_si256((__m256i*)p0, float2bfloat_avx512(_f0));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep), float2bfloat_avx512(_f1));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 2), float2bfloat_avx512(_f2));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 3), float2bfloat_avx512(_f3));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 4), float2bfloat_avx512(_f4));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 5), float2bfloat_avx512(_f5));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 6), float2bfloat_avx512(_f6));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 7), float2bfloat_avx512(_f7));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 8), float2bfloat_avx512(_f8));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 9), float2bfloat_avx512(_f9));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 10), float2bfloat_avx512(_fa));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 11), float2bfloat_avx512(_fb));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 12), float2bfloat_avx512(_fc));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 13), float2bfloat_avx512(_fd));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 14), float2bfloat_avx512(_fe));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 15), float2bfloat_avx512(_ff));
                p0 += 16 * out_hstep;
            }
            else
            {
                transpose16x16_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7, _f8, _f9, _fa, _fb, _fc, _fd, _fe, _ff);
                _mm256_storeu_si256((__m256i*)p0, float2bfloat_avx512(_f0));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep), float2bfloat_avx512(_f1));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 2), float2bfloat_avx512(_f2));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 3), float2bfloat_avx512(_f3));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 4), float2bfloat_avx512(_f4));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 5), float2bfloat_avx512(_f5));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 6), float2bfloat_avx512(_f6));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 7), float2bfloat_avx512(_f7));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 8), float2bfloat_avx512(_f8));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 9), float2bfloat_avx512(_f9));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 10), float2bfloat_avx512(_fa));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 11), float2bfloat_avx512(_fb));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 12), float2bfloat_avx512(_fc));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 13), float2bfloat_avx512(_fd));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 14), float2bfloat_avx512(_fe));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 15), float2bfloat_avx512(_ff));
                p0 += 16;
            }
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m512 _f0 = _mm512_load_ps(pp);
            __m512 _f1 = _mm512_load_ps(pp + 16);
            __m512 _f2 = _mm512_load_ps(pp + 32);
            __m512 _f3 = _mm512_load_ps(pp + 48);
            __m512 _f4 = _mm512_load_ps(pp + 64);
            __m512 _f5 = _mm512_load_ps(pp + 80);
            __m512 _f6 = _mm512_load_ps(pp + 96);
            __m512 _f7 = _mm512_load_ps(pp + 112);
            pp += 128;

#if __AVX512BF16__
            // deshuffle from the shuffle-based 16x8 dpbf16_ps kernel
            {
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm512_permute_ps(_f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm512_permute_ps(_f7, _MM_SHUFFLE(2, 1, 0, 3));

                __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f3);
                __m512 _tmp1 = _mm512_unpackhi_ps(_f0, _f3);
                __m512 _tmp2 = _mm512_unpacklo_ps(_f2, _f1);
                __m512 _tmp3 = _mm512_unpackhi_ps(_f2, _f1);
                __m512 _tmp4 = _mm512_unpacklo_ps(_f4, _f7);
                __m512 _tmp5 = _mm512_unpackhi_ps(_f4, _f7);
                __m512 _tmp6 = _mm512_unpacklo_ps(_f6, _f5);
                __m512 _tmp7 = _mm512_unpackhi_ps(_f6, _f5);

                _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _f2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _f4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                _f5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                _f6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));
                _f7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));

                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm512_permute_ps(_f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm512_permute_ps(_f7, _MM_SHUFFLE(2, 1, 0, 3));

                _tmp0 = _mm512_shuffle_f32x4(_f0, _f4, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp1 = _mm512_shuffle_f32x4(_f1, _f5, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp2 = _mm512_shuffle_f32x4(_f2, _f6, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp3 = _mm512_shuffle_f32x4(_f3, _f7, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp4 = _mm512_shuffle_f32x4(_f0, _f4, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp5 = _mm512_shuffle_f32x4(_f1, _f5, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp6 = _mm512_shuffle_f32x4(_f2, _f6, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp7 = _mm512_shuffle_f32x4(_f3, _f7, _MM_SHUFFLE(2, 3, 3, 2));

                _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(2, 0, 2, 0));
                _f1 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                _f2 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(2, 0, 2, 0));
                _f3 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                _f4 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(1, 3, 1, 3));
                _f5 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(1, 3, 1, 3));
                _f6 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(1, 3, 1, 3));
                _f7 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(1, 3, 1, 3));
            }
#endif // __AVX512BF16__

            if (broadcast_type_C == 3)
            {
                __m512 _ct0 = _mm512_setzero_ps();
                __m512 _ct1 = _mm512_setzero_ps();
                __m512 _ct2 = _mm512_setzero_ps();
                __m512 _ct3 = _mm512_setzero_ps();
                __m512 _ct4 = _mm512_setzero_ps();
                __m512 _ct5 = _mm512_setzero_ps();
                __m512 _ct6 = _mm512_setzero_ps();
                __m512 _ct7 = _mm512_setzero_ps();
                for (int s = 0; s < 16; s++)
                {
                    ((float*)&_ct0)[s] = pC[(s / c_elempack) * c_hstep * c_elempack + 0 * c_elempack + s % c_elempack];
                    ((float*)&_ct1)[s] = pC[(s / c_elempack) * c_hstep * c_elempack + 1 * c_elempack + s % c_elempack];
                    ((float*)&_ct2)[s] = pC[(s / c_elempack) * c_hstep * c_elempack + 2 * c_elempack + s % c_elempack];
                    ((float*)&_ct3)[s] = pC[(s / c_elempack) * c_hstep * c_elempack + 3 * c_elempack + s % c_elempack];
                    ((float*)&_ct4)[s] = pC[(s / c_elempack) * c_hstep * c_elempack + 4 * c_elempack + s % c_elempack];
                    ((float*)&_ct5)[s] = pC[(s / c_elempack) * c_hstep * c_elempack + 5 * c_elempack + s % c_elempack];
                    ((float*)&_ct6)[s] = pC[(s / c_elempack) * c_hstep * c_elempack + 6 * c_elempack + s % c_elempack];
                    ((float*)&_ct7)[s] = pC[(s / c_elempack) * c_hstep * c_elempack + 7 * c_elempack + s % c_elempack];
                }
                _f0 = _mm512_add_ps(_f0, _ct0);
                _f1 = _mm512_add_ps(_f1, _ct1);
                _f2 = _mm512_add_ps(_f2, _ct2);
                _f3 = _mm512_add_ps(_f3, _ct3);
                _f4 = _mm512_add_ps(_f4, _ct4);
                _f5 = _mm512_add_ps(_f5, _ct5);
                _f6 = _mm512_add_ps(_f6, _ct6);
                _f7 = _mm512_add_ps(_f7, _ct7);
                pC += 8 * c_elempack;
            }
            else if (broadcast_type_C == 4)
            {
                _f0 = _mm512_add_ps(_f0, _mm512_set1_ps(pC[0]));
                _f1 = _mm512_add_ps(_f1, _mm512_set1_ps(pC[1]));
                _f2 = _mm512_add_ps(_f2, _mm512_set1_ps(pC[2]));
                _f3 = _mm512_add_ps(_f3, _mm512_set1_ps(pC[3]));
                _f4 = _mm512_add_ps(_f4, _mm512_set1_ps(pC[4]));
                _f5 = _mm512_add_ps(_f5, _mm512_set1_ps(pC[5]));
                _f6 = _mm512_add_ps(_f6, _mm512_set1_ps(pC[6]));
                _f7 = _mm512_add_ps(_f7, _mm512_set1_ps(pC[7]));
                pC += 8;
            }
            else
            {
                _f0 = _mm512_add_ps(_f0, _c0);
                _f1 = _mm512_add_ps(_f1, _c0);
                _f2 = _mm512_add_ps(_f2, _c0);
                _f3 = _mm512_add_ps(_f3, _c0);
                _f4 = _mm512_add_ps(_f4, _c0);
                _f5 = _mm512_add_ps(_f5, _c0);
                _f6 = _mm512_add_ps(_f6, _c0);
                _f7 = _mm512_add_ps(_f7, _c0);
            }

            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
                _f2 = _mm512_mul_ps(_f2, _alpha);
                _f3 = _mm512_mul_ps(_f3, _alpha);
                _f4 = _mm512_mul_ps(_f4, _alpha);
                _f5 = _mm512_mul_ps(_f5, _alpha);
                _f6 = _mm512_mul_ps(_f6, _alpha);
                _f7 = _mm512_mul_ps(_f7, _alpha);
            }

            if (output_transpose)
            {
                _mm256_storeu_si256((__m256i*)p0, float2bfloat_avx512(_f0));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep), float2bfloat_avx512(_f1));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 2), float2bfloat_avx512(_f2));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 3), float2bfloat_avx512(_f3));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 4), float2bfloat_avx512(_f4));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 5), float2bfloat_avx512(_f5));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 6), float2bfloat_avx512(_f6));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 7), float2bfloat_avx512(_f7));
                p0 += 8 * out_hstep;
            }
            else
            {
                // _f0.._f7 are columns (16 rows each), store as rows
                float tmp[128];
                _mm512_storeu_ps(tmp, _f0);
                _mm512_storeu_ps(tmp + 16, _f1);
                _mm512_storeu_ps(tmp + 32, _f2);
                _mm512_storeu_ps(tmp + 48, _f3);
                _mm512_storeu_ps(tmp + 64, _f4);
                _mm512_storeu_ps(tmp + 80, _f5);
                _mm512_storeu_ps(tmp + 96, _f6);
                _mm512_storeu_ps(tmp + 112, _f7);
                for (int s = 0; s < 16; s++)
                {
                    for (int t = 0; t < 8; t++)
                    {
                        *((unsigned short*)top_blob + (i + ii + s) * out_hstep + (j + jj + t)) = float32_to_bfloat16(tmp[t * 16 + s]);
                    }
                }
            }
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m512 _f0 = _mm512_load_ps(pp);
            __m512 _f1 = _mm512_load_ps(pp + 16);
            __m512 _f2 = _mm512_load_ps(pp + 32);
            __m512 _f3 = _mm512_load_ps(pp + 48);
            pp += 64;

            if (broadcast_type_C == 3)
            {
                __m512 _ct0 = _mm512_setzero_ps();
                __m512 _ct1 = _mm512_setzero_ps();
                __m512 _ct2 = _mm512_setzero_ps();
                __m512 _ct3 = _mm512_setzero_ps();
                for (int s = 0; s < 16; s++)
                {
                    ((float*)&_ct0)[s] = pC[(s / c_elempack) * c_hstep * c_elempack + 0 * c_elempack + s % c_elempack];
                    ((float*)&_ct1)[s] = pC[(s / c_elempack) * c_hstep * c_elempack + 1 * c_elempack + s % c_elempack];
                    ((float*)&_ct2)[s] = pC[(s / c_elempack) * c_hstep * c_elempack + 2 * c_elempack + s % c_elempack];
                    ((float*)&_ct3)[s] = pC[(s / c_elempack) * c_hstep * c_elempack + 3 * c_elempack + s % c_elempack];
                }
                _f0 = _mm512_add_ps(_f0, _ct0);
                _f1 = _mm512_add_ps(_f1, _ct1);
                _f2 = _mm512_add_ps(_f2, _ct2);
                _f3 = _mm512_add_ps(_f3, _ct3);
                pC += 4 * c_elempack;
            }
            else if (broadcast_type_C == 4)
            {
                _f0 = _mm512_add_ps(_f0, _mm512_set1_ps(pC[0]));
                _f1 = _mm512_add_ps(_f1, _mm512_set1_ps(pC[1]));
                _f2 = _mm512_add_ps(_f2, _mm512_set1_ps(pC[2]));
                _f3 = _mm512_add_ps(_f3, _mm512_set1_ps(pC[3]));
                pC += 4;
            }
            else
            {
                _f0 = _mm512_add_ps(_f0, _c0);
                _f1 = _mm512_add_ps(_f1, _c0);
                _f2 = _mm512_add_ps(_f2, _c0);
                _f3 = _mm512_add_ps(_f3, _c0);
            }

            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
                _f2 = _mm512_mul_ps(_f2, _alpha);
                _f3 = _mm512_mul_ps(_f3, _alpha);
            }

            if (output_transpose)
            {
                _mm256_storeu_si256((__m256i*)p0, float2bfloat_avx512(_f0));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep), float2bfloat_avx512(_f1));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 2), float2bfloat_avx512(_f2));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 3), float2bfloat_avx512(_f3));
                p0 += 4 * out_hstep;
            }
            else
            {
                float tmp[64];
                _mm512_storeu_ps(tmp, _f0);
                _mm512_storeu_ps(tmp + 16, _f1);
                _mm512_storeu_ps(tmp + 32, _f2);
                _mm512_storeu_ps(tmp + 48, _f3);
                for (int s = 0; s < 16; s++)
                {
                    for (int t = 0; t < 4; t++)
                    {
                        *((unsigned short*)top_blob + (i + ii + s) * out_hstep + (j + jj + t)) = float32_to_bfloat16(tmp[t * 16 + s]);
                    }
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m512 _f0 = _mm512_load_ps(pp);
            __m512 _f1 = _mm512_load_ps(pp + 16);
            pp += 32;

            if (broadcast_type_C == 3)
            {
                __m512 _ct0 = _mm512_setzero_ps();
                __m512 _ct1 = _mm512_setzero_ps();
                for (int s = 0; s < 16; s++)
                {
                    ((float*)&_ct0)[s] = pC[(s / c_elempack) * c_hstep * c_elempack + 0 * c_elempack + s % c_elempack];
                    ((float*)&_ct1)[s] = pC[(s / c_elempack) * c_hstep * c_elempack + 1 * c_elempack + s % c_elempack];
                }
                _f0 = _mm512_add_ps(_f0, _ct0);
                _f1 = _mm512_add_ps(_f1, _ct1);
                pC += 2 * c_elempack;
            }
            else if (broadcast_type_C == 4)
            {
                _f0 = _mm512_add_ps(_f0, _mm512_set1_ps(pC[0]));
                _f1 = _mm512_add_ps(_f1, _mm512_set1_ps(pC[1]));
                pC += 2;
            }
            else
            {
                _f0 = _mm512_add_ps(_f0, _c0);
                _f1 = _mm512_add_ps(_f1, _c0);
            }

            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
            }

            if (output_transpose)
            {
                _mm256_storeu_si256((__m256i*)p0, float2bfloat_avx512(_f0));
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep), float2bfloat_avx512(_f1));
                p0 += 2 * out_hstep;
            }
            else
            {
                float tmp[32];
                _mm512_storeu_ps(tmp, _f0);
                _mm512_storeu_ps(tmp + 16, _f1);
                for (int s = 0; s < 16; s++)
                {
                    for (int t = 0; t < 2; t++)
                    {
                        *((unsigned short*)top_blob + (i + ii + s) * out_hstep + (j + jj + t)) = float32_to_bfloat16(tmp[t * 16 + s]);
                    }
                }
            }
        }
        for (; jj < max_jj; jj++)
        {
            // 16x1: no deshuffle needed
            __m512 _f0 = _mm512_load_ps(pp);
            pp += 16;

            if (broadcast_type_C == 3)
            {
                __m512 _ct = _mm512_setzero_ps();
                for (int s = 0; s < 16; s++)
                {
                    ((float*)&_ct)[s] = pC[(s / c_elempack) * c_hstep * c_elempack + s % c_elempack];
                }
                _f0 = _mm512_add_ps(_f0, _ct);
                pC += 1 * c_elempack;
            }
            else if (broadcast_type_C == 4)
            {
                _f0 = _mm512_add_ps(_f0, _mm512_set1_ps(pC[0]));
                pC += 1;
            }
            else
            {
                _f0 = _mm512_add_ps(_f0, _c0);
            }

            if (alpha != 1.f)
            {
                _f0 = _mm512_mul_ps(_f0, _mm512_set1_ps(alpha));
            }

            if (output_transpose)
            {
                _mm256_storeu_si256((__m256i*)p0, float2bfloat_avx512(_f0));
                p0 += out_hstep;
            }
            else
            {
                for (int s = 0; s < 16; s++)
                {
                    *((unsigned short*)top_blob + (i + ii + s) * out_hstep + (j + jj)) = float32_to_bfloat16(((const float*)&_f0)[s]);
                }
                p0 += out_hstep * 16;
            }
        }

        if (broadcast_type_C == 1 || broadcast_type_C == 2)
        {
            pC += 16;
        }
        if (broadcast_type_C == 4)
        {
            pC -= max_jj;
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        unsigned short* p0;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        __m256 _c0_avx = _mm256_setzero_ps();
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0_avx = _mm256_set1_ps(pC[0]);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                _c0_avx = _mm256_loadu_ps(pC);
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
        }

        int jj = 0;
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            // 8x16: pp has 16 columns, each column is 8 floats (column-major from gemm)
            for (int t = 0; t < 16; t++)
            {
                for (int s = 0; s < 8; s++)
                {
                    float f = pp[t * 8 + s];

                    if (broadcast_type_C == 3)
                    {
                        f += pC[(s / c_elempack) * c_hstep * c_elempack + t * c_elempack + s % c_elempack];
                    }
                    else if (broadcast_type_C == 4)
                    {
                        f += pC[t];
                    }
                    else
                    {
                        f += ((const float*)&_c0_avx)[s];
                    }

                    f *= alpha;

                    if (output_transpose)
                    {
                        *((unsigned short*)top_blob + (j + jj + t) * out_hstep + (i + ii + s)) = float32_to_bfloat16(f);
                    }
                    else
                    {
                        *((unsigned short*)top_blob + (i + ii + s) * out_hstep + (j + jj + t)) = float32_to_bfloat16(f);
                    }
                }
            }
            pp += 128;
            if (output_transpose)
            {
                p0 += 16 * out_hstep;
            }
            else
            {
                p0 += 16;
            }
            if (broadcast_type_C == 3)
            {
                pC += 16 * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC += 16;
            }
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _f0 = _mm256_load_ps(pp);
            __m256 _f1 = _mm256_load_ps(pp + 8);
            __m256 _f2 = _mm256_load_ps(pp + 16);
            __m256 _f3 = _mm256_load_ps(pp + 24);
            __m256 _f4 = _mm256_load_ps(pp + 32);
            __m256 _f5 = _mm256_load_ps(pp + 40);
            __m256 _f6 = _mm256_load_ps(pp + 48);
            __m256 _f7 = _mm256_load_ps(pp + 56);
            pp += 64;

            if (broadcast_type_C == 3)
            {
                if (c_elempack == 8)
                {
                    __m256 _cc0 = _mm256_loadu_ps(pC);
                    __m256 _c1 = _mm256_loadu_ps(pC + 8);
                    __m256 _c2 = _mm256_loadu_ps(pC + 16);
                    __m256 _c3 = _mm256_loadu_ps(pC + 24);
                    __m256 _c4 = _mm256_loadu_ps(pC + 32);
                    __m256 _c5 = _mm256_loadu_ps(pC + 40);
                    __m256 _c6 = _mm256_loadu_ps(pC + 48);
                    __m256 _c7 = _mm256_loadu_ps(pC + 56);
                    _f0 = _mm256_add_ps(_f0, _cc0);
                    _f1 = _mm256_add_ps(_f1, _c1);
                    _f2 = _mm256_add_ps(_f2, _c2);
                    _f3 = _mm256_add_ps(_f3, _c3);
                    _f4 = _mm256_add_ps(_f4, _c4);
                    _f5 = _mm256_add_ps(_f5, _c5);
                    _f6 = _mm256_add_ps(_f6, _c6);
                    _f7 = _mm256_add_ps(_f7, _c7);
                    pC += 64;
                }
                else if (c_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_loadu_ps(pC);
                    __m256 _tmp1 = _mm256_loadu_ps(pC + 8);
                    __m256 _tmp2 = _mm256_loadu_ps(pC + 16);
                    __m256 _tmp3 = _mm256_loadu_ps(pC + 24);
                    __m256 _tmp4 = _mm256_loadu_ps(pC + c_hstep * 4);
                    __m256 _tmp5 = _mm256_loadu_ps(pC + c_hstep * 4 + 8);
                    __m256 _tmp6 = _mm256_loadu_ps(pC + c_hstep * 4 + 16);
                    __m256 _tmp7 = _mm256_loadu_ps(pC + c_hstep * 4 + 24);
                    __m256 _cc0 = _mm256_permute2f128_ps(_tmp0, _tmp4, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _cc1 = _mm256_permute2f128_ps(_tmp0, _tmp4, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _cc2 = _mm256_permute2f128_ps(_tmp1, _tmp5, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _cc3 = _mm256_permute2f128_ps(_tmp1, _tmp5, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _cc4 = _mm256_permute2f128_ps(_tmp2, _tmp6, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _cc5 = _mm256_permute2f128_ps(_tmp2, _tmp6, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _cc6 = _mm256_permute2f128_ps(_tmp3, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _cc7 = _mm256_permute2f128_ps(_tmp3, _tmp7, _MM_SHUFFLE(0, 3, 0, 1));
                    _f0 = _mm256_add_ps(_f0, _cc0);
                    _f1 = _mm256_add_ps(_f1, _cc1);
                    _f2 = _mm256_add_ps(_f2, _cc2);
                    _f3 = _mm256_add_ps(_f3, _cc3);
                    _f4 = _mm256_add_ps(_f4, _cc4);
                    _f5 = _mm256_add_ps(_f5, _cc5);
                    _f6 = _mm256_add_ps(_f6, _cc6);
                    _f7 = _mm256_add_ps(_f7, _cc7);
                    pC += 32;
                }
                else // c_elempack == 1
                {
                    __m128 _cc0 = _mm_loadu_ps(pC);
                    __m128 _cc1 = _mm_loadu_ps(pC + c_hstep);
                    __m128 _cc2 = _mm_loadu_ps(pC + c_hstep * 2);
                    __m128 _cc3 = _mm_loadu_ps(pC + c_hstep * 3);
                    __m128 _cc4 = _mm_loadu_ps(pC + c_hstep * 4);
                    __m128 _cc5 = _mm_loadu_ps(pC + c_hstep * 5);
                    __m128 _cc6 = _mm_loadu_ps(pC + c_hstep * 6);
                    __m128 _cc7 = _mm_loadu_ps(pC + c_hstep * 7);
                    _MM_TRANSPOSE4_PS(_cc0, _cc1, _cc2, _cc3);
                    __m128 _cc0h = _mm_loadu_ps(pC + 4);
                    __m128 _cc1h = _mm_loadu_ps(pC + c_hstep + 4);
                    __m128 _cc2h = _mm_loadu_ps(pC + c_hstep * 2 + 4);
                    __m128 _cc3h = _mm_loadu_ps(pC + c_hstep * 3 + 4);
                    __m128 _cc4h = _mm_loadu_ps(pC + c_hstep * 4 + 4);
                    __m128 _cc5h = _mm_loadu_ps(pC + c_hstep * 5 + 4);
                    __m128 _cc6h = _mm_loadu_ps(pC + c_hstep * 6 + 4);
                    __m128 _cc7h = _mm_loadu_ps(pC + c_hstep * 7 + 4);
                    _MM_TRANSPOSE4_PS(_cc4, _cc5, _cc6, _cc7);
                    _MM_TRANSPOSE4_PS(_cc0h, _cc1h, _cc2h, _cc3h);
                    _MM_TRANSPOSE4_PS(_cc4h, _cc5h, _cc6h, _cc7h);
                    _f0 = _mm256_add_ps(_f0, _mm256_insertf128_ps(_mm256_castps128_ps256(_cc0), _cc0h, 1));
                    _f1 = _mm256_add_ps(_f1, _mm256_insertf128_ps(_mm256_castps128_ps256(_cc1), _cc1h, 1));
                    _f2 = _mm256_add_ps(_f2, _mm256_insertf128_ps(_mm256_castps128_ps256(_cc2), _cc2h, 1));
                    _f3 = _mm256_add_ps(_f3, _mm256_insertf128_ps(_mm256_castps128_ps256(_cc3), _cc3h, 1));
                    _f4 = _mm256_add_ps(_f4, _mm256_insertf128_ps(_mm256_castps128_ps256(_cc4), _cc4h, 1));
                    _f5 = _mm256_add_ps(_f5, _mm256_insertf128_ps(_mm256_castps128_ps256(_cc5), _cc5h, 1));
                    _f6 = _mm256_add_ps(_f6, _mm256_insertf128_ps(_mm256_castps128_ps256(_cc6), _cc6h, 1));
                    _f7 = _mm256_add_ps(_f7, _mm256_insertf128_ps(_mm256_castps128_ps256(_cc7), _cc7h, 1));
                    pC += 8;
                }
            }
            else if (broadcast_type_C == 4)
            {
                _f0 = _mm256_add_ps(_f0, _mm256_set1_ps(pC[0]));
                _f1 = _mm256_add_ps(_f1, _mm256_set1_ps(pC[1]));
                _f2 = _mm256_add_ps(_f2, _mm256_set1_ps(pC[2]));
                _f3 = _mm256_add_ps(_f3, _mm256_set1_ps(pC[3]));
                _f4 = _mm256_add_ps(_f4, _mm256_set1_ps(pC[4]));
                _f5 = _mm256_add_ps(_f5, _mm256_set1_ps(pC[5]));
                _f6 = _mm256_add_ps(_f6, _mm256_set1_ps(pC[6]));
                _f7 = _mm256_add_ps(_f7, _mm256_set1_ps(pC[7]));
                pC += 8;
            }
            else
            {
                _f0 = _mm256_add_ps(_f0, _c0_avx);
                _f1 = _mm256_add_ps(_f1, _c0_avx);
                _f2 = _mm256_add_ps(_f2, _c0_avx);
                _f3 = _mm256_add_ps(_f3, _c0_avx);
                _f4 = _mm256_add_ps(_f4, _c0_avx);
                _f5 = _mm256_add_ps(_f5, _c0_avx);
                _f6 = _mm256_add_ps(_f6, _c0_avx);
                _f7 = _mm256_add_ps(_f7, _c0_avx);
            }

            if (alpha != 1.f)
            {
                __m256 _alpha = _mm256_set1_ps(alpha);
                _f0 = _mm256_mul_ps(_f0, _alpha);
                _f1 = _mm256_mul_ps(_f1, _alpha);
                _f2 = _mm256_mul_ps(_f2, _alpha);
                _f3 = _mm256_mul_ps(_f3, _alpha);
                _f4 = _mm256_mul_ps(_f4, _alpha);
                _f5 = _mm256_mul_ps(_f5, _alpha);
                _f6 = _mm256_mul_ps(_f6, _alpha);
                _f7 = _mm256_mul_ps(_f7, _alpha);
            }

            if (output_transpose)
            {
                _mm_storeu_si128((__m128i*)p0, float2bfloat_avx(_f0));
                _mm_storeu_si128((__m128i*)(p0 + out_hstep), float2bfloat_avx(_f1));
                _mm_storeu_si128((__m128i*)(p0 + out_hstep * 2), float2bfloat_avx(_f2));
                _mm_storeu_si128((__m128i*)(p0 + out_hstep * 3), float2bfloat_avx(_f3));
                _mm_storeu_si128((__m128i*)(p0 + out_hstep * 4), float2bfloat_avx(_f4));
                _mm_storeu_si128((__m128i*)(p0 + out_hstep * 5), float2bfloat_avx(_f5));
                _mm_storeu_si128((__m128i*)(p0 + out_hstep * 6), float2bfloat_avx(_f6));
                _mm_storeu_si128((__m128i*)(p0 + out_hstep * 7), float2bfloat_avx(_f7));
                p0 += 8 * out_hstep;
            }
            else
            {
                // _f0.._f7 are columns (8 rows each), store as rows for non-transpose
                float tmp[64];
                _mm256_storeu_ps(tmp, _f0);
                _mm256_storeu_ps(tmp + 8, _f1);
                _mm256_storeu_ps(tmp + 16, _f2);
                _mm256_storeu_ps(tmp + 24, _f3);
                _mm256_storeu_ps(tmp + 32, _f4);
                _mm256_storeu_ps(tmp + 40, _f5);
                _mm256_storeu_ps(tmp + 48, _f6);
                _mm256_storeu_ps(tmp + 56, _f7);
                for (int s = 0; s < 8; s++)
                {
                    for (int t = 0; t < 8; t++)
                    {
                        *((unsigned short*)top_blob + (i + ii + s) * out_hstep + (j + jj + t)) = float32_to_bfloat16(tmp[t * 8 + s]);
                    }
                }
            }
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m256 _f0 = _mm256_load_ps(pp);
            __m256 _f1 = _mm256_load_ps(pp + 8);
            __m256 _f2 = _mm256_load_ps(pp + 16);
            __m256 _f3 = _mm256_load_ps(pp + 24);
            pp += 32;

            if (broadcast_type_C == 3)
            {
                if (c_elempack == 8)
                {
                    __m256 _cc0 = _mm256_loadu_ps(pC);
                    __m256 _c1 = _mm256_loadu_ps(pC + 8);
                    __m256 _c2 = _mm256_loadu_ps(pC + 16);
                    __m256 _c3 = _mm256_loadu_ps(pC + 24);
                    _f0 = _mm256_add_ps(_f0, _cc0);
                    _f1 = _mm256_add_ps(_f1, _c1);
                    _f2 = _mm256_add_ps(_f2, _c2);
                    _f3 = _mm256_add_ps(_f3, _c3);
                    pC += 32;
                }
                else if (c_elempack == 4)
                {
                    __m256 _cc0 = _mm256_loadu_ps(pC);
                    __m256 _cc1 = _mm256_loadu_ps(pC + 8);
                    __m256 _cc2 = _mm256_loadu_ps(pC + c_hstep * 4);
                    __m256 _cc3 = _mm256_loadu_ps(pC + c_hstep * 4 + 8);
                    _f0 = _mm256_add_ps(_f0, _mm256_permute2f128_ps(_cc0, _cc2, _MM_SHUFFLE(0, 2, 0, 0)));
                    _f1 = _mm256_add_ps(_f1, _mm256_permute2f128_ps(_cc0, _cc2, _MM_SHUFFLE(0, 3, 0, 1)));
                    _f2 = _mm256_add_ps(_f2, _mm256_permute2f128_ps(_cc1, _cc3, _MM_SHUFFLE(0, 2, 0, 0)));
                    _f3 = _mm256_add_ps(_f3, _mm256_permute2f128_ps(_cc1, _cc3, _MM_SHUFFLE(0, 3, 0, 1)));
                    pC += 16;
                }
                else // c_elempack == 1
                {
                    __m128 _cc0 = _mm_loadu_ps(pC);
                    __m128 _cc1 = _mm_loadu_ps(pC + c_hstep);
                    __m128 _cc2 = _mm_loadu_ps(pC + c_hstep * 2);
                    __m128 _cc3 = _mm_loadu_ps(pC + c_hstep * 3);
                    __m128 _cc4 = _mm_loadu_ps(pC + c_hstep * 4);
                    __m128 _cc5 = _mm_loadu_ps(pC + c_hstep * 5);
                    __m128 _cc6 = _mm_loadu_ps(pC + c_hstep * 6);
                    __m128 _cc7 = _mm_loadu_ps(pC + c_hstep * 7);
                    _MM_TRANSPOSE4_PS(_cc0, _cc1, _cc2, _cc3);
                    _MM_TRANSPOSE4_PS(_cc4, _cc5, _cc6, _cc7);
                    _f0 = _mm256_add_ps(_f0, _mm256_insertf128_ps(_mm256_castps128_ps256(_cc0), _cc4, 1));
                    _f1 = _mm256_add_ps(_f1, _mm256_insertf128_ps(_mm256_castps128_ps256(_cc1), _cc5, 1));
                    _f2 = _mm256_add_ps(_f2, _mm256_insertf128_ps(_mm256_castps128_ps256(_cc2), _cc6, 1));
                    _f3 = _mm256_add_ps(_f3, _mm256_insertf128_ps(_mm256_castps128_ps256(_cc3), _cc7, 1));
                    pC += 4;
                }
            }
            else if (broadcast_type_C == 4)
            {
                _f0 = _mm256_add_ps(_f0, _mm256_set1_ps(pC[0]));
                _f1 = _mm256_add_ps(_f1, _mm256_set1_ps(pC[1]));
                _f2 = _mm256_add_ps(_f2, _mm256_set1_ps(pC[2]));
                _f3 = _mm256_add_ps(_f3, _mm256_set1_ps(pC[3]));
                pC += 4;
            }
            else
            {
                _f0 = _mm256_add_ps(_f0, _c0_avx);
                _f1 = _mm256_add_ps(_f1, _c0_avx);
                _f2 = _mm256_add_ps(_f2, _c0_avx);
                _f3 = _mm256_add_ps(_f3, _c0_avx);
            }

            if (alpha != 1.f)
            {
                __m256 _alpha = _mm256_set1_ps(alpha);
                _f0 = _mm256_mul_ps(_f0, _alpha);
                _f1 = _mm256_mul_ps(_f1, _alpha);
                _f2 = _mm256_mul_ps(_f2, _alpha);
                _f3 = _mm256_mul_ps(_f3, _alpha);
            }

            if (output_transpose)
            {
                _mm_storeu_si128((__m128i*)p0, float2bfloat_avx(_f0));
                _mm_storeu_si128((__m128i*)(p0 + out_hstep), float2bfloat_avx(_f1));
                _mm_storeu_si128((__m128i*)(p0 + out_hstep * 2), float2bfloat_avx(_f2));
                _mm_storeu_si128((__m128i*)(p0 + out_hstep * 3), float2bfloat_avx(_f3));
                p0 += 4 * out_hstep;
            }
            else
            {
                float tmp[32];
                _mm256_storeu_ps(tmp, _f0);
                _mm256_storeu_ps(tmp + 8, _f1);
                _mm256_storeu_ps(tmp + 16, _f2);
                _mm256_storeu_ps(tmp + 24, _f3);
                for (int s = 0; s < 8; s++)
                {
                    for (int t = 0; t < 4; t++)
                    {
                        *((unsigned short*)top_blob + (i + ii + s) * out_hstep + (j + jj + t)) = float32_to_bfloat16(tmp[t * 8 + s]);
                    }
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            // 8x2: no shuffle (uses broadcast)
            for (int t = 0; t < 2; t++)
            {
                __m256 _f = _mm256_load_ps(pp);
                pp += 8;

                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 8)
                    {
                        _f = _mm256_add_ps(_f, _mm256_loadu_ps(pC + t * 8));
                    }
                    else if (c_elempack == 4)
                    {
                        __m128 _cc0 = _mm_loadu_ps(pC + t * 4);
                        __m128 _cc1 = _mm_loadu_ps(pC + c_hstep * 4 + t * 4);
                        _f = _mm256_add_ps(_f, _mm256_insertf128_ps(_mm256_castps128_ps256(_cc0), _cc1, 1));
                    }
                    else // c_elempack == 1
                    {
                        __m256 _cc = _mm256_setzero_ps();
                        for (int s = 0; s < 8; s++)
                        {
                            ((float*)&_cc)[s] = pC[s * c_hstep + t];
                        }
                        _f = _mm256_add_ps(_f, _cc);
                    }
                }
                else if (broadcast_type_C == 4)
                {
                    _f = _mm256_add_ps(_f, _mm256_set1_ps(pC[t]));
                }
                else
                {
                    _f = _mm256_add_ps(_f, _c0_avx);
                }

                if (alpha != 1.f)
                {
                    _f = _mm256_mul_ps(_f, _mm256_set1_ps(alpha));
                }

                if (output_transpose)
                {
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * t), float2bfloat_avx(_f));
                }
                else
                {
                    for (int s = 0; s < 8; s++)
                    {
                        *((unsigned short*)top_blob + (i + ii + s) * out_hstep + (j + jj + t)) = float32_to_bfloat16(((const float*)&_f)[s]);
                    }
                }
            }
            if (broadcast_type_C == 3)
            {
                if (c_elempack == 8)
                    pC += 16;
                else if (c_elempack == 4)
                    pC += 8;
                else
                    pC += 2;
            }
            if (broadcast_type_C == 4)
            {
                pC += 2;
            }
            if (output_transpose)
            {
                p0 += 2 * out_hstep;
            }
        }
        for (; jj < max_jj; jj++)
        {
            __m256 _f = _mm256_load_ps(pp);
            pp += 8;

            if (broadcast_type_C == 3)
            {
                if (c_elempack == 8)
                {
                    _f = _mm256_add_ps(_f, _mm256_loadu_ps(pC));
                    pC += 8;
                }
                else if (c_elempack == 4)
                {
                    __m128 _cc0 = _mm_loadu_ps(pC);
                    __m128 _cc1 = _mm_loadu_ps(pC + c_hstep * 4);
                    _f = _mm256_add_ps(_f, _mm256_insertf128_ps(_mm256_castps128_ps256(_cc0), _cc1, 1));
                    pC += 4;
                }
                else // c_elempack == 1
                {
                    __m256 _cc = _mm256_setzero_ps();
                    for (int s = 0; s < 8; s++)
                    {
                        ((float*)&_cc)[s] = pC[s * c_hstep];
                    }
                    _f = _mm256_add_ps(_f, _cc);
                    pC += 1;
                }
            }
            else if (broadcast_type_C == 4)
            {
                _f = _mm256_add_ps(_f, _mm256_set1_ps(pC[0]));
                pC += 1;
            }
            else
            {
                _f = _mm256_add_ps(_f, _c0_avx);
            }

            if (alpha != 1.f)
            {
                _f = _mm256_mul_ps(_f, _mm256_set1_ps(alpha));
            }

            if (output_transpose)
            {
                _mm_storeu_si128((__m128i*)p0, float2bfloat_avx(_f));
                p0 += out_hstep;
            }
            else
            {
                for (int s = 0; s < 8; s++)
                {
                    *((unsigned short*)top_blob + (i + ii + s) * out_hstep + (j + jj)) = float32_to_bfloat16(((const float*)&_f)[s]);
                }
            }
        }

        if (broadcast_type_C == 1 || broadcast_type_C == 2)
        {
            pC += 8;
        }
        if (broadcast_type_C == 4)
        {
            pC -= max_jj;
        }
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        unsigned short* p0;
        if (output_transpose)
        {
            p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        __m128 _c0_sse = _mm_setzero_ps();
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0_sse = _mm_set1_ps(pC[0]);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                _c0_sse = _mm_loadu_ps(pC);
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
        }

        int jj = 0;
#if __AVX__
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            // 4x16: pp has 16 columns, each column is 4 floats
            for (int t = 0; t < 16; t++)
            {
                for (int s = 0; s < 4; s++)
                {
                    float f = pp[t * 4 + s];

                    if (broadcast_type_C == 3)
                    {
                        if (c_elempack == 4)
                        {
                            f += pC[t * 4 + s];
                        }
                        else // c_elempack == 1
                        {
                            f += pC[s * c_hstep + t];
                        }
                    }
                    else if (broadcast_type_C == 4)
                    {
                        f += pC[t];
                    }
                    else
                    {
                        f += ((const float*)&_c0_sse)[s];
                    }

                    f *= alpha;

                    if (output_transpose)
                    {
                        *((unsigned short*)top_blob + (j + jj + t) * out_hstep + (i + ii + s)) = float32_to_bfloat16(f);
                    }
                    else
                    {
                        *((unsigned short*)top_blob + (i + ii + s) * out_hstep + (j + jj + t)) = float32_to_bfloat16(f);
                    }
                }
            }
            pp += 64;
            if (output_transpose)
            {
                p0 += 16 * out_hstep;
            }
            else
            {
                p0 += 16;
            }
            if (broadcast_type_C == 3)
            {
                if (c_elempack == 4)
                    pC += 64;
                else
                    pC += 16;
            }
            if (broadcast_type_C == 4)
            {
                pC += 16;
            }
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            // 4x8: pp has 8 columns, each column is 4 floats
            for (int t = 0; t < 8; t++)
            {
                for (int s = 0; s < 4; s++)
                {
                    float f = pp[t * 4 + s];

                    if (broadcast_type_C == 3)
                    {
                        if (c_elempack == 4)
                        {
                            f += pC[t * 4 + s];
                        }
                        else // c_elempack == 1
                        {
                            f += pC[s * c_hstep + t];
                        }
                    }
                    else if (broadcast_type_C == 4)
                    {
                        f += pC[t];
                    }
                    else
                    {
                        f += ((const float*)&_c0_sse)[s];
                    }

                    f *= alpha;

                    if (output_transpose)
                    {
                        *((unsigned short*)top_blob + (j + jj + t) * out_hstep + (i + ii + s)) = float32_to_bfloat16(f);
                    }
                    else
                    {
                        *((unsigned short*)top_blob + (i + ii + s) * out_hstep + (j + jj + t)) = float32_to_bfloat16(f);
                    }
                }
            }
            pp += 32;
            if (output_transpose)
            {
                p0 += 8 * out_hstep;
            }
            else
            {
                p0 += 8;
            }
            if (broadcast_type_C == 3)
            {
                if (c_elempack == 4)
                    pC += 32;
                else
                    pC += 8;
            }
            if (broadcast_type_C == 4)
            {
                pC += 8;
            }
        }
#endif // __AVX__
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = _mm_load_ps(pp);
            __m128 _f1 = _mm_load_ps(pp + 4);
            __m128 _f2 = _mm_load_ps(pp + 8);
            __m128 _f3 = _mm_load_ps(pp + 12);
            pp += 16;

            // 4x4: broadcast-based gemm, output is in column order (no shuffle)
            // _f0 = column 0 = [a0*b0, a1*b0, a2*b0, a3*b0] summed over k

            if (broadcast_type_C == 3)
            {
                if (c_elempack == 4)
                {
                    __m128 _cc0 = _mm_loadu_ps(pC);
                    __m128 _c1 = _mm_loadu_ps(pC + 4);
                    __m128 _c2 = _mm_loadu_ps(pC + 8);
                    __m128 _c3 = _mm_loadu_ps(pC + 12);
                    _f0 = _mm_add_ps(_f0, _cc0);
                    _f1 = _mm_add_ps(_f1, _c1);
                    _f2 = _mm_add_ps(_f2, _c2);
                    _f3 = _mm_add_ps(_f3, _c3);
                    pC += 16;
                }
                else // c_elempack == 1
                {
                    __m128 _cc0 = _mm_loadu_ps(pC);
                    __m128 _cc1 = _mm_loadu_ps(pC + c_hstep);
                    __m128 _cc2 = _mm_loadu_ps(pC + c_hstep * 2);
                    __m128 _cc3 = _mm_loadu_ps(pC + c_hstep * 3);
                    _MM_TRANSPOSE4_PS(_cc0, _cc1, _cc2, _cc3);
                    _f0 = _mm_add_ps(_f0, _cc0);
                    _f1 = _mm_add_ps(_f1, _cc1);
                    _f2 = _mm_add_ps(_f2, _cc2);
                    _f3 = _mm_add_ps(_f3, _cc3);
                    pC += 4;
                }
            }
            else if (broadcast_type_C == 4)
            {
                _f0 = _mm_add_ps(_f0, _mm_set1_ps(pC[0]));
                _f1 = _mm_add_ps(_f1, _mm_set1_ps(pC[1]));
                _f2 = _mm_add_ps(_f2, _mm_set1_ps(pC[2]));
                _f3 = _mm_add_ps(_f3, _mm_set1_ps(pC[3]));
                pC += 4;
            }
            else
            {
                _f0 = _mm_add_ps(_f0, _c0_sse);
                _f1 = _mm_add_ps(_f1, _c0_sse);
                _f2 = _mm_add_ps(_f2, _c0_sse);
                _f3 = _mm_add_ps(_f3, _c0_sse);
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
                _f2 = _mm_mul_ps(_f2, _alpha);
                _f3 = _mm_mul_ps(_f3, _alpha);
            }

            if (output_transpose)
            {
                _mm_storel_epi64((__m128i*)p0, float2bfloat_sse(_f0, _mm_setzero_ps()));
                _mm_storel_epi64((__m128i*)(p0 + out_hstep), float2bfloat_sse(_f1, _mm_setzero_ps()));
                _mm_storel_epi64((__m128i*)(p0 + out_hstep * 2), float2bfloat_sse(_f2, _mm_setzero_ps()));
                _mm_storel_epi64((__m128i*)(p0 + out_hstep * 3), float2bfloat_sse(_f3, _mm_setzero_ps()));
                p0 += 4 * out_hstep;
            }
            else
            {
                // _f0.._f3 are columns, transpose to rows for row-major output
                _MM_TRANSPOSE4_PS(_f0, _f1, _f2, _f3);

                __m128i _bf01 = float2bfloat_sse(_f0, _f1);
                __m128i _bf23 = float2bfloat_sse(_f2, _f3);

                _mm_storel_epi64((__m128i*)(p0), _bf01);
                _mm_storel_epi64((__m128i*)(p0 + out_hstep), _mm_srli_si128(_bf01, 8));
                _mm_storel_epi64((__m128i*)(p0 + out_hstep * 2), _bf23);
                _mm_storel_epi64((__m128i*)(p0 + out_hstep * 3), _mm_srli_si128(_bf23, 8));
                p0 += 4;
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            // 4x2: no shuffle (uses broadcast)
            for (int t = 0; t < 2; t++)
            {
                __m128 _f = _mm_load_ps(pp);
                pp += 4;

                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        _f = _mm_add_ps(_f, _mm_loadu_ps(pC + t * 4));
                    }
                    else // c_elempack == 1
                    {
                        __m128 _cc = _mm_setr_ps(pC[t], pC[c_hstep + t], pC[c_hstep * 2 + t], pC[c_hstep * 3 + t]);
                        _f = _mm_add_ps(_f, _cc);
                    }
                }
                else if (broadcast_type_C == 4)
                {
                    _f = _mm_add_ps(_f, _mm_set1_ps(pC[t]));
                }
                else
                {
                    _f = _mm_add_ps(_f, _c0_sse);
                }

                if (alpha != 1.f)
                {
                    _f = _mm_mul_ps(_f, _mm_set1_ps(alpha));
                }

                if (output_transpose)
                {
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * t), float2bfloat_sse(_f, _mm_setzero_ps()));
                }
                else
                {
                    for (int s = 0; s < 4; s++)
                    {
                        *((unsigned short*)top_blob + (i + ii + s) * out_hstep + (j + jj + t)) = float32_to_bfloat16(((const float*)&_f)[s]);
                    }
                }
            }
            if (broadcast_type_C == 3)
            {
                if (c_elempack == 4)
                    pC += 8;
                else
                    pC += 2;
            }
            if (broadcast_type_C == 4)
            {
                pC += 2;
            }
            if (output_transpose)
            {
                p0 += 2 * out_hstep;
            }
        }
        for (; jj < max_jj; jj++)
        {
            __m128 _f = _mm_load_ps(pp);
            pp += 4;

            if (broadcast_type_C == 3)
            {
                if (c_elempack == 4)
                {
                    _f = _mm_add_ps(_f, _mm_loadu_ps(pC));
                    pC += 4;
                }
                else // c_elempack == 1
                {
                    __m128 _cc = _mm_setr_ps(pC[0], pC[c_hstep], pC[c_hstep * 2], pC[c_hstep * 3]);
                    _f = _mm_add_ps(_f, _cc);
                    pC += 1;
                }
            }
            else if (broadcast_type_C == 4)
            {
                _f = _mm_add_ps(_f, _mm_set1_ps(pC[0]));
                pC += 1;
            }
            else
            {
                _f = _mm_add_ps(_f, _c0_sse);
            }

            if (alpha != 1.f)
            {
                _f = _mm_mul_ps(_f, _mm_set1_ps(alpha));
            }

            if (output_transpose)
            {
                _mm_storel_epi64((__m128i*)p0, float2bfloat_sse(_f, _mm_setzero_ps()));
                p0 += out_hstep;
            }
            else
            {
                for (int s = 0; s < 4; s++)
                {
                    *((unsigned short*)top_blob + (i + ii + s) * out_hstep + (j + jj)) = float32_to_bfloat16(((const float*)&_f)[s]);
                }
            }
        }

        if (broadcast_type_C == 1 || broadcast_type_C == 2)
        {
            pC += 4;
        }
        if (broadcast_type_C == 4)
        {
            pC -= max_jj;
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        float _c0_scalar[2] = {0.f, 0.f};
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0_scalar[0] = pC[0];
                _c0_scalar[1] = pC[0];
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                _c0_scalar[0] = pC[0];
                _c0_scalar[1] = pC[1];
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
        }

        int jj = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            for (int t = 0; t < 16; t++)
            {
                float f0 = pp[t];
                float f1 = pp[16 + t];

                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 1)
                    {
                        f0 += pC[t];
                        f1 += pC[c_hstep + t];
                    }
                }
                else if (broadcast_type_C == 4)
                {
                    f0 += pC[t];
                    f1 += pC[t];
                }
                else
                {
                    f0 += _c0_scalar[0];
                    f1 += _c0_scalar[1];
                }

                f0 *= alpha;
                f1 *= alpha;

                if (output_transpose)
                {
                    *((unsigned short*)top_blob + (j + jj + t) * out_hstep + (i + ii)) = float32_to_bfloat16(f0);
                    *((unsigned short*)top_blob + (j + jj + t) * out_hstep + (i + ii + 1)) = float32_to_bfloat16(f1);
                }
                else
                {
                    *((unsigned short*)top_blob + (i + ii) * out_hstep + (j + jj + t)) = float32_to_bfloat16(f0);
                    *((unsigned short*)top_blob + (i + ii + 1) * out_hstep + (j + jj + t)) = float32_to_bfloat16(f1);
                }
            }
            pp += 32;
            if (broadcast_type_C == 3)
            {
                if (c_elempack == 1)
                    pC += 16;
            }
            if (broadcast_type_C == 4)
            {
                pC += 16;
            }
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            for (int t = 0; t < 8; t++)
            {
                float f0 = pp[t];
                float f1 = pp[8 + t];

                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 1)
                    {
                        f0 += pC[t];
                        f1 += pC[c_hstep + t];
                    }
                }
                else if (broadcast_type_C == 4)
                {
                    f0 += pC[t];
                    f1 += pC[t];
                }
                else
                {
                    f0 += _c0_scalar[0];
                    f1 += _c0_scalar[1];
                }

                f0 *= alpha;
                f1 *= alpha;

                if (output_transpose)
                {
                    *((unsigned short*)top_blob + (j + jj + t) * out_hstep + (i + ii)) = float32_to_bfloat16(f0);
                    *((unsigned short*)top_blob + (j + jj + t) * out_hstep + (i + ii + 1)) = float32_to_bfloat16(f1);
                }
                else
                {
                    *((unsigned short*)top_blob + (i + ii) * out_hstep + (j + jj + t)) = float32_to_bfloat16(f0);
                    *((unsigned short*)top_blob + (i + ii + 1) * out_hstep + (j + jj + t)) = float32_to_bfloat16(f1);
                }
            }
            pp += 16;
            if (broadcast_type_C == 3)
            {
                if (c_elempack == 1)
                    pC += 8;
            }
            if (broadcast_type_C == 4)
            {
                pC += 8;
            }
        }
#endif // __AVX__
        for (; jj + 3 < max_jj; jj += 4)
        {
            for (int t = 0; t < 4; t++)
            {
                float f0 = pp[t];
                float f1 = pp[4 + t];

                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 1)
                    {
                        f0 += pC[t];
                        f1 += pC[c_hstep + t];
                    }
                }
                else if (broadcast_type_C == 4)
                {
                    f0 += pC[t];
                    f1 += pC[t];
                }
                else
                {
                    f0 += _c0_scalar[0];
                    f1 += _c0_scalar[1];
                }

                f0 *= alpha;
                f1 *= alpha;

                if (output_transpose)
                {
                    *((unsigned short*)top_blob + (j + jj + t) * out_hstep + (i + ii)) = float32_to_bfloat16(f0);
                    *((unsigned short*)top_blob + (j + jj + t) * out_hstep + (i + ii + 1)) = float32_to_bfloat16(f1);
                }
                else
                {
                    *((unsigned short*)top_blob + (i + ii) * out_hstep + (j + jj + t)) = float32_to_bfloat16(f0);
                    *((unsigned short*)top_blob + (i + ii + 1) * out_hstep + (j + jj + t)) = float32_to_bfloat16(f1);
                }
            }
            pp += 8;
            if (broadcast_type_C == 3)
            {
                if (c_elempack == 1)
                    pC += 4;
            }
            if (broadcast_type_C == 4)
            {
                pC += 4;
            }
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            float f00 = pp[0];
            float f01 = pp[1];
            float f10 = pp[2];
            float f11 = pp[3];
            pp += 4;

            if (broadcast_type_C == 3)
            {
                if (c_elempack == 1)
                {
                    f00 += pC[0];
                    f01 += pC[1];
                    f10 += pC[c_hstep];
                    f11 += pC[c_hstep + 1];
                    pC += 2;
                }
            }
            else if (broadcast_type_C == 4)
            {
                f00 += pC[0];
                f01 += pC[1];
                f10 += pC[0];
                f11 += pC[1];
                pC += 2;
            }
            else
            {
                f00 += _c0_scalar[0];
                f01 += _c0_scalar[0];
                f10 += _c0_scalar[1];
                f11 += _c0_scalar[1];
            }

            f00 *= alpha;
            f01 *= alpha;
            f10 *= alpha;
            f11 *= alpha;

            if (output_transpose)
            {
                *((unsigned short*)top_blob + (j + jj) * out_hstep + (i + ii)) = float32_to_bfloat16(f00);
                *((unsigned short*)top_blob + (j + jj) * out_hstep + (i + ii + 1)) = float32_to_bfloat16(f10);
                *((unsigned short*)top_blob + (j + jj + 1) * out_hstep + (i + ii)) = float32_to_bfloat16(f01);
                *((unsigned short*)top_blob + (j + jj + 1) * out_hstep + (i + ii + 1)) = float32_to_bfloat16(f11);
            }
            else
            {
                *((unsigned short*)top_blob + (i + ii) * out_hstep + (j + jj)) = float32_to_bfloat16(f00);
                *((unsigned short*)top_blob + (i + ii) * out_hstep + (j + jj + 1)) = float32_to_bfloat16(f01);
                *((unsigned short*)top_blob + (i + ii + 1) * out_hstep + (j + jj)) = float32_to_bfloat16(f10);
                *((unsigned short*)top_blob + (i + ii + 1) * out_hstep + (j + jj + 1)) = float32_to_bfloat16(f11);
            }
        }
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[0];
            float f1 = pp[1];
            pp += 2;

            if (broadcast_type_C == 3)
            {
                if (c_elempack == 1)
                {
                    f0 += pC[0];
                    f1 += pC[c_hstep];
                    pC += 1;
                }
            }
            else if (broadcast_type_C == 4)
            {
                f0 += pC[0];
                f1 += pC[0];
                pC += 1;
            }
            else
            {
                f0 += _c0_scalar[0];
                f1 += _c0_scalar[1];
            }

            f0 *= alpha;
            f1 *= alpha;

            if (output_transpose)
            {
                *((unsigned short*)top_blob + (j + jj) * out_hstep + (i + ii)) = float32_to_bfloat16(f0);
                *((unsigned short*)top_blob + (j + jj) * out_hstep + (i + ii + 1)) = float32_to_bfloat16(f1);
            }
            else
            {
                *((unsigned short*)top_blob + (i + ii) * out_hstep + (j + jj)) = float32_to_bfloat16(f0);
                *((unsigned short*)top_blob + (i + ii + 1) * out_hstep + (j + jj)) = float32_to_bfloat16(f1);
            }
        }

        if (broadcast_type_C == 1 || broadcast_type_C == 2)
        {
            pC += 2;
        }
        if (broadcast_type_C == 4)
        {
            pC -= max_jj;
        }
    }
    for (; ii < max_ii; ii++)
    {
        float _c0_scalar = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0_scalar = pC[0];
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                _c0_scalar = pC[0];
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
        }

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[0];
            pp += 1;

            if (broadcast_type_C == 3)
            {
                if (c_elempack == 1)
                {
                    f0 += pC[0];
                    pC += 1;
                }
            }
            else if (broadcast_type_C == 4)
            {
                f0 += pC[0];
                pC += 1;
            }
            else
            {
                f0 += _c0_scalar;
            }

            f0 *= alpha;

            if (output_transpose)
            {
                *((unsigned short*)top_blob + (j + jj) * out_hstep + (i + ii)) = float32_to_bfloat16(f0);
            }
            else
            {
                *((unsigned short*)top_blob + (i + ii) * out_hstep + (j + jj)) = float32_to_bfloat16(f0);
            }
        }

        if (broadcast_type_C == 1 || broadcast_type_C == 2)
        {
            pC += 1;
        }
        if (broadcast_type_C == 4)
        {
            pC -= max_jj;
        }
    }
}

static void get_optimal_tile_mnk_bf16(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // bf16 takes half the space of fp32 plus fp32 accumulator
    int tile_size = (int)sqrt((float)l2_cache_size / (2 * sizeof(unsigned short) + sizeof(float)));

#if __AVX512F__
    TILE_M = std::max(16, tile_size / 16 * 16);
    TILE_N = std::max(16, tile_size / 16 * 16);
    TILE_K = std::max(16, tile_size / 16 * 16);
#elif __AVX__
    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_N = std::max(8, tile_size / 8 * 8);
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
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(unsigned short) / TILE_K);

#if __AVX512F__
            TILE_M = std::max(16, tile_size / 16 * 16);
            TILE_N = std::max(16, tile_size / 16 * 16);
#elif __AVX__
            TILE_M = std::max(8, tile_size / 8 * 8);
            TILE_N = std::max(8, tile_size / 8 * 8);
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
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8);
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
        TILE_N = (constant_TILE_N + 7) / 8 * 8;
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

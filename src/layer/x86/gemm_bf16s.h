// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void pack_A_tile_bf16_avx512bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void transpose_pack_A_tile_bf16_avx512bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void pack_B_tile_bf16_avx512bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void transpose_pack_B_tile_bf16_avx512bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void gemm_transB_packed_tile_bf16s_avx512bf16(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk);
#endif

static void pack_A_tile_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        pack_A_tile_bf16_avx512bf16(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("pack_A_tile_bf16 %d %d %d %d", i, max_ii, k, max_kk);
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = (unsigned short*)AT;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;

        if (elempack == 16)
        {
            int kk = 0;
#if __AVX512BF16__
            __m512i _idx = _mm512_set_epi16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512i _p = _mm512_loadu_si512((const __m512i*)p0);
                _p = _mm512_permutexvar_epi16(_idx, _p);
                _mm512_storeu_si512((__m512i*)pp, _p);
                pp += 32;
                p0 += 32;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                _mm256_storeu_si256((__m256i*)pp, _mm256_loadu_si256((const __m256i*)p0));
                pp += 16;
                p0 += 16;
            }
        }
        if (elempack == 8)
        {
            const unsigned short* p1 = p0 + A_hstep * 8;

            int kk = 0;
#if __AVX512BF16__
            __m512i _idx = _mm512_set_epi16(31, 23, 30, 22, 29, 21, 28, 20, 27, 19, 26, 18, 25, 17, 24, 16, 15, 7, 14, 6, 13, 5, 12, 4, 11, 3, 10, 2, 9, 1, 8, 0);
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _a = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _b = _mm256_loadu_si256((const __m256i*)p1);
                __m512i _ab = combine8x2_epi32(_a, _b);
                __m512i _p = _mm512_permutexvar_epi16(_idx, _ab);
                _mm512_storeu_si512((__m512i*)pp, _p);
                pp += 32;
                p0 += 16;
                p1 += 16;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
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
            const unsigned short* p1 = p0 + A_hstep * 4;
            const unsigned short* p2 = p0 + A_hstep * 8;
            const unsigned short* p3 = p0 + A_hstep * 12;

            int kk = 0;
#if __AVX512BF16__
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
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
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
            __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(A_hstep));

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512i _p = _mm512_i32gather_epi32(_vindex, (const int*)p0, sizeof(unsigned short));
                _mm512_storeu_si512((__m512i*)pp, _p);
                pp += 32;
                p0 += 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m512i _p = _mm512_i32gather_epi32(_vindex, (const int*)p0, sizeof(unsigned short));
                __m256i _p16 = _mm512_cvtepi32_epi16(_p);
                _mm256_storeu_si256((__m256i*)pp, _p16);
                pp += 16;
                p0++;
            }
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;

        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512BF16__
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
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                pp += 8;
                p0 += 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p1 = p0 + A_hstep * 4;

            int kk = 0;
#if __AVX512BF16__
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
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
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
#if __AVX2__
            __m256i _vindex = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(A_hstep));
#endif
            const unsigned short* p1 = p0 + A_hstep * 1;
            const unsigned short* p2 = p0 + A_hstep * 2;
            const unsigned short* p3 = p0 + A_hstep * 3;
            const unsigned short* p4 = p0 + A_hstep * 4;
            const unsigned short* p5 = p0 + A_hstep * 5;
            const unsigned short* p6 = p0 + A_hstep * 6;
            const unsigned short* p7 = p0 + A_hstep * 7;

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _p = _mm256_i32gather_epi32((const int*)p0, _vindex, sizeof(unsigned short));
                _mm256_storeu_si256((__m256i*)pp, _p);
                pp += 16;
                p0 += 2;
            }
#else  // __AVX512BF16__
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
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
#if __AVX2__
                __m256i _p = _mm256_i32gather_epi32((const int*)p0, _vindex, sizeof(unsigned short));
                __m128i _p16 = _mm256_comp_cvtepi32_epi16(_p);
                _mm_storeu_si128((__m128i*)pp, _p16);
#else
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp[4] = p4[0];
                pp[5] = p5[0];
                pp[6] = p6[0];
                pp[7] = p7[0];
                p1++;
                p2++;
                p3++;
                p4++;
                p5++;
                p6++;
                p7++;
#endif
                pp += 8;
                p0++;
            }
        }
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;

        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512BF16__
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
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p1 = p0 + A_hstep * 1;
            const unsigned short* p2 = p0 + A_hstep * 2;
            const unsigned short* p3 = p0 + A_hstep * 3;

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p1[0];
                pp[3] = p1[1];
                pp[4] = p2[0];
                pp[5] = p2[1];
                pp[6] = p3[0];
                pp[7] = p3[1];
                pp += 8;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
            }
#else  // __AVX512BF16__
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
#endif // __AVX512BF16__
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
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
        const unsigned short* p1 = p0 + A_hstep;

        // if (elempack == 1)
        {
            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p1[0];
                pp[3] = p1[1];
                pp += 4;
                p0 += 2;
                p1 += 2;
            }
#endif // __AVX512BF16__
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
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;

        // if (elempack == 1)
        {
            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
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
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        transpose_pack_A_tile_bf16_avx512bf16(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("transpose_pack_A_tile_bf16 %d %d %d %d", i, max_ii, k, max_kk);
    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = (unsigned short*)AT;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * elempack;

        if (elempack == 16)
        {
            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512i _r0 = _mm512_loadu_si512((const __m512i*)p0);
                __m512i _r1 = _mm512_loadu_si512((const __m512i*)(p0 + 32));
                __m512i _r2 = _mm512_loadu_si512((const __m512i*)(p0 + 64));
                __m512i _r3 = _mm512_loadu_si512((const __m512i*)(p0 + 96));
                __m512i _r4 = _mm512_loadu_si512((const __m512i*)(p0 + 128));
                __m512i _r5 = _mm512_loadu_si512((const __m512i*)(p0 + 160));
                __m512i _r6 = _mm512_loadu_si512((const __m512i*)(p0 + 192));
                __m512i _r7 = _mm512_loadu_si512((const __m512i*)(p0 + 224));

                __m512i w0 = _mm512_shuffle_i64x2(_r0, _r1, 0x44);
                __m512i w1 = _mm512_shuffle_i64x2(_r0, _r1, 0xEE);
                __m512i w2 = _mm512_shuffle_i64x2(_r2, _r3, 0x44);
                __m512i w3 = _mm512_shuffle_i64x2(_r2, _r3, 0xEE);
                __m512i w4 = _mm512_shuffle_i64x2(_r4, _r5, 0x44);
                __m512i w5 = _mm512_shuffle_i64x2(_r4, _r5, 0xEE);
                __m512i w6 = _mm512_shuffle_i64x2(_r6, _r7, 0x44);
                __m512i w7 = _mm512_shuffle_i64x2(_r6, _r7, 0xEE);

#if __AVX512BF16__
                __m512i a0 = _mm512_unpacklo_epi32(w0, w1);
                __m512i a1 = _mm512_unpackhi_epi32(w0, w1);
                __m512i a2 = _mm512_unpacklo_epi32(w2, w3);
                __m512i a3 = _mm512_unpackhi_epi32(w2, w3);
                __m512i a4 = _mm512_unpacklo_epi32(w4, w5);
                __m512i a5 = _mm512_unpackhi_epi32(w4, w5);
                __m512i a6 = _mm512_unpacklo_epi32(w6, w7);
                __m512i a7 = _mm512_unpackhi_epi32(w6, w7);

                __m512i b0 = _mm512_unpacklo_epi64(a0, a2);
                __m512i b1 = _mm512_unpackhi_epi64(a0, a2);
                __m512i b2 = _mm512_unpacklo_epi64(a1, a3);
                __m512i b3 = _mm512_unpackhi_epi64(a1, a3);
                __m512i b4 = _mm512_unpacklo_epi64(a4, a6);
                __m512i b5 = _mm512_unpackhi_epi64(a4, a6);
                __m512i b6 = _mm512_unpacklo_epi64(a5, a7);
                __m512i b7 = _mm512_unpackhi_epi64(a5, a7);

                __m512i idx_l = _mm512_set_epi32(27, 26, 19, 18, 25, 24, 17, 16, 11, 10, 3, 2, 9, 8, 1, 0);
                __m512i idx_r = _mm512_set_epi32(31, 30, 23, 22, 29, 28, 21, 20, 15, 14, 7, 6, 13, 12, 5, 4);

                __m512i _p0 = _mm512_permutex2var_epi32(b0, idx_l, b4);
                __m512i _p1 = _mm512_permutex2var_epi32(b1, idx_l, b5);
                __m512i _p2 = _mm512_permutex2var_epi32(b2, idx_l, b6);
                __m512i _p3 = _mm512_permutex2var_epi32(b3, idx_l, b7);
                __m512i _p4 = _mm512_permutex2var_epi32(b0, idx_r, b4);
                __m512i _p5 = _mm512_permutex2var_epi32(b1, idx_r, b5);
                __m512i _p6 = _mm512_permutex2var_epi32(b2, idx_r, b6);
                __m512i _p7 = _mm512_permutex2var_epi32(b3, idx_r, b7);
#else  // __AVX512BF16__
                __m512i a0 = _mm512_unpacklo_epi16(w0, w1);
                __m512i a1 = _mm512_unpackhi_epi16(w0, w1);
                __m512i a2 = _mm512_unpacklo_epi16(w2, w3);
                __m512i a3 = _mm512_unpackhi_epi16(w2, w3);
                __m512i a4 = _mm512_unpacklo_epi16(w4, w5);
                __m512i a5 = _mm512_unpackhi_epi16(w4, w5);
                __m512i a6 = _mm512_unpacklo_epi16(w6, w7);
                __m512i a7 = _mm512_unpackhi_epi16(w6, w7);

                __m512i b0 = _mm512_unpacklo_epi32(a0, a2);
                __m512i b1 = _mm512_unpackhi_epi32(a0, a2);
                __m512i b2 = _mm512_unpacklo_epi32(a1, a3);
                __m512i b3 = _mm512_unpackhi_epi32(a1, a3);
                __m512i b4 = _mm512_unpacklo_epi32(a4, a6);
                __m512i b5 = _mm512_unpackhi_epi32(a4, a6);
                __m512i b6 = _mm512_unpacklo_epi32(a5, a7);
                __m512i b7 = _mm512_unpackhi_epi32(a5, a7);

                __m512i c0 = _mm512_unpacklo_epi64(b0, b4);
                __m512i c1 = _mm512_unpackhi_epi64(b0, b4);
                __m512i c2 = _mm512_unpacklo_epi64(b1, b5);
                __m512i c3 = _mm512_unpackhi_epi64(b1, b5);
                __m512i c4 = _mm512_unpacklo_epi64(b2, b6);
                __m512i c5 = _mm512_unpackhi_epi64(b2, b6);
                __m512i c6 = _mm512_unpacklo_epi64(b3, b7);
                __m512i c7 = _mm512_unpackhi_epi64(b3, b7);

                __m512i idx_lo = _mm512_set_epi32(27, 19, 26, 18, 25, 17, 24, 16, 11, 3, 10, 2, 9, 1, 8, 0);
                __m512i idx_hi = _mm512_set_epi32(31, 23, 30, 22, 29, 21, 28, 20, 15, 7, 14, 6, 13, 5, 12, 4);

                __m512i _p0 = _mm512_permutex2var_epi32(c0, idx_lo, c1); // col 0,1
                __m512i _p1 = _mm512_permutex2var_epi32(c2, idx_lo, c3); // col 2,3
                __m512i _p2 = _mm512_permutex2var_epi32(c4, idx_lo, c5); // col 4,5
                __m512i _p3 = _mm512_permutex2var_epi32(c6, idx_lo, c7); // col 6,7
                __m512i _p4 = _mm512_permutex2var_epi32(c0, idx_hi, c1); // col 8,9
                __m512i _p5 = _mm512_permutex2var_epi32(c2, idx_hi, c3); // col A,B
                __m512i _p6 = _mm512_permutex2var_epi32(c4, idx_hi, c5); // col C,D
                __m512i _p7 = _mm512_permutex2var_epi32(c6, idx_hi, c7); // col E,F
#endif // __AVX512BF16__

                _mm512_storeu_si512((__m512i*)pp, _p0);
                _mm512_storeu_si512((__m512i*)(pp + 32), _p1);
                _mm512_storeu_si512((__m512i*)(pp + 64), _p2);
                _mm512_storeu_si512((__m512i*)(pp + 96), _p3);
                _mm512_storeu_si512((__m512i*)(pp + 128), _p4);
                _mm512_storeu_si512((__m512i*)(pp + 160), _p5);
                _mm512_storeu_si512((__m512i*)(pp + 192), _p6);
                _mm512_storeu_si512((__m512i*)(pp + 224), _p7);
                pp += 256;
                p0 += A_hstep * 16;
            }
        }
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m512i _r0 = _mm512_loadu_si512((const __m512i*)p0);
                __m512i _r1 = _mm512_loadu_si512((const __m512i*)(p0 + 32));
                __m512i _r2 = _mm512_loadu_si512((const __m512i*)(p0 + 64));
                __m512i _r3 = _mm512_loadu_si512((const __m512i*)(p0 + 96));
#if __AVX512BF16__
                __m512i idx0 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 28, 24, 20, 16, 12, 8, 4, 0);
                __m512i idx1 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 29, 25, 21, 17, 13, 9, 5, 1);
                __m512i idx2 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 30, 26, 22, 18, 14, 10, 6, 2);
                __m512i idx3 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 31, 27, 23, 19, 15, 11, 7, 3);

                __m512i lo0 = _mm512_permutex2var_epi32(_r0, idx0, _r1);
                __m512i lo1 = _mm512_permutex2var_epi32(_r0, idx1, _r1);
                __m512i lo2 = _mm512_permutex2var_epi32(_r0, idx2, _r1);
                __m512i lo3 = _mm512_permutex2var_epi32(_r0, idx3, _r1);

                __m512i hi0 = _mm512_permutex2var_epi32(_r2, idx0, _r3);
                __m512i hi1 = _mm512_permutex2var_epi32(_r2, idx1, _r3);
                __m512i hi2 = _mm512_permutex2var_epi32(_r2, idx2, _r3);
                __m512i hi3 = _mm512_permutex2var_epi32(_r2, idx3, _r3);

                __m512i _p0 = _mm512_inserti64x4(lo0, _mm512_castsi512_si256(hi0), 1);
                __m512i _p1 = _mm512_inserti64x4(lo1, _mm512_castsi512_si256(hi1), 1);
                __m512i _p2 = _mm512_inserti64x4(lo2, _mm512_castsi512_si256(hi2), 1);
                __m512i _p3 = _mm512_inserti64x4(lo3, _mm512_castsi512_si256(hi3), 1);
#else  // __AVX512BF16__
                __m512i id0 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 28, 24, 20, 16, 12, 8, 4, 0);
                __m512i id1 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 29, 25, 21, 17, 13, 9, 5, 1);
                __m512i id2 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 30, 26, 22, 18, 14, 10, 6, 2);
                __m512i id3 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 31, 27, 23, 19, 15, 11, 7, 3);

                __m512i p0_lo = _mm512_permutex2var_epi32(_r0, id0, _r1);
                __m512i p1_lo = _mm512_permutex2var_epi32(_r0, id1, _r1);
                __m512i p2_lo = _mm512_permutex2var_epi32(_r0, id2, _r1);
                __m512i p3_lo = _mm512_permutex2var_epi32(_r0, id3, _r1);

                __m512i p0_hi = _mm512_permutex2var_epi32(_r2, id0, _r3);
                __m512i p1_hi = _mm512_permutex2var_epi32(_r2, id1, _r3);
                __m512i p2_hi = _mm512_permutex2var_epi32(_r2, id2, _r3);
                __m512i p3_hi = _mm512_permutex2var_epi32(_r2, id3, _r3);

                __m512i cp0 = _mm512_inserti64x4(p0_lo, _mm512_castsi512_si256(p0_hi), 1);
                __m512i cp1 = _mm512_inserti64x4(p1_lo, _mm512_castsi512_si256(p1_hi), 1);
                __m512i cp2 = _mm512_inserti64x4(p2_lo, _mm512_castsi512_si256(p2_hi), 1);
                __m512i cp3 = _mm512_inserti64x4(p3_lo, _mm512_castsi512_si256(p3_hi), 1);

                __m512i shuf = _mm512_set4_epi32(0x0f0e0b0a, 0x07060302, 0x0d0c0908, 0x05040100);
                __m512i pq = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

                __m512i s0 = _mm512_shuffle_epi8(cp0, shuf);
                __m512i s1 = _mm512_shuffle_epi8(cp1, shuf);
                __m512i s2 = _mm512_shuffle_epi8(cp2, shuf);
                __m512i s3 = _mm512_shuffle_epi8(cp3, shuf);

                __m512i _p0 = _mm512_permutexvar_epi64(pq, s0);
                __m512i _p1 = _mm512_permutexvar_epi64(pq, s1);
                __m512i _p2 = _mm512_permutexvar_epi64(pq, s2);
                __m512i _p3 = _mm512_permutexvar_epi64(pq, s3);
#endif // __AVX512BF16__
                _mm512_storeu_si512((__m512i*)pp, _p0);
                _mm512_storeu_si512((__m512i*)(pp + 32), _p1);
                _mm512_storeu_si512((__m512i*)(pp + 64), _p2);
                _mm512_storeu_si512((__m512i*)(pp + 96), _p3);
                pp += 128;
                p0 += A_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512i _r0 = _mm512_loadu_si512((const __m512i*)p0);
                __m512i _r1 = _mm512_loadu_si512((const __m512i*)(p0 + 32));
#if __AVX512BF16__
                __m512i idx_lo = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
                __m512i idx_hi = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
                __m512i _p0 = _mm512_permutex2var_epi32(_r0, idx_lo, _r1);
                __m512i _p1 = _mm512_permutex2var_epi32(_r0, idx_hi, _r1);
#else  // __AVX512BF16__
                __m512i idx_lo = _mm512_set_epi16(61, 57, 53, 49, 45, 41, 37, 33, 29, 25, 21, 17, 13, 9, 5, 1, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0);
                __m512i idx_hi = _mm512_set_epi16(63, 59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11, 7, 3, 62, 58, 54, 50, 46, 42, 38, 34, 30, 26, 22, 18, 14, 10, 6, 2);
                __m512i _p0 = _mm512_permutex2var_epi16(_r0, idx_lo, _r1);
                __m512i _p1 = _mm512_permutex2var_epi16(_r0, idx_hi, _r1);
#endif // __AVX512BF16__
                _mm512_storeu_si512((__m512i*)pp, _p0);
                _mm512_storeu_si512((__m512i*)(pp + 32), _p1);
                pp += 64;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)(p0 + A_hstep));
                transpose16x2_epi16(_r0, _r1);
                _mm256_storeu_si256((__m256i*)pp, _r0);
                _mm256_storeu_si256((__m256i*)(pp + 16), _r1);
                pp += 32;
                p0 += A_hstep * 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                _mm256_storeu_si256((__m256i*)pp, _mm256_loadu_si256((const __m256i*)p0));
                pp += 16;
                p0 += A_hstep;
            }
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * elempack;

#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
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
#if __AVX512BF16__
                transpose8x8_epi32(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
#else  // __AVX512BF16__
                transpose16x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
#endif // __AVX512BF16__
                _mm256_storeu_si256((__m256i*)pp, _r0);
                _mm256_storeu_si256((__m256i*)(pp + 16), _r1);
                _mm256_storeu_si256((__m256i*)(pp + 32), _r2);
                _mm256_storeu_si256((__m256i*)(pp + 48), _r3);
                _mm256_storeu_si256((__m256i*)(pp + 64), _r4);
                _mm256_storeu_si256((__m256i*)(pp + 80), _r5);
                _mm256_storeu_si256((__m256i*)(pp + 96), _r6);
                _mm256_storeu_si256((__m256i*)(pp + 112), _r7);
                pp += 128;
                p0 += A_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + 8));
                __m128i _r2 = _mm_loadu_si128((const __m128i*)(p0 + 16));
                __m128i _r3 = _mm_loadu_si128((const __m128i*)(p0 + 24));
                __m128i _r4 = _mm_loadu_si128((const __m128i*)(p0 + 32));
                __m128i _r5 = _mm_loadu_si128((const __m128i*)(p0 + 40));
                __m128i _r6 = _mm_loadu_si128((const __m128i*)(p0 + 48));
                __m128i _r7 = _mm_loadu_si128((const __m128i*)(p0 + 56));
#if __AVX512BF16__
                transpose4x8_epi32(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
#else  // __AVX512BF16__
                transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
#endif // __AVX512BF16__
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
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)(p0 + 4));
                __m128i _r2 = _mm_loadl_epi64((const __m128i*)(p0 + 8));
                __m128i _r3 = _mm_loadl_epi64((const __m128i*)(p0 + 12));
                __m128i _r4 = _mm_loadl_epi64((const __m128i*)(p0 + 16));
                __m128i _r5 = _mm_loadl_epi64((const __m128i*)(p0 + 20));
                __m128i _r6 = _mm_loadl_epi64((const __m128i*)(p0 + 24));
                __m128i _r7 = _mm_loadl_epi64((const __m128i*)(p0 + 28));
#if __AVX512BF16__
                __m128i _t0 = _mm_unpacklo_epi32(_r0, _r1);
                __m128i _t1 = _mm_unpacklo_epi32(_r2, _r3);
                __m128i _t2 = _mm_unpacklo_epi32(_r4, _r5);
                __m128i _t3 = _mm_unpacklo_epi32(_r6, _r7);
                __m128i _p0 = _mm_unpacklo_epi64(_t0, _t1);
                __m128i _p1 = _mm_unpacklo_epi64(_t2, _t3);
                __m128i _p2 = _mm_unpackhi_epi64(_t0, _t1);
                __m128i _p3 = _mm_unpackhi_epi64(_t2, _t3);
#else  // __AVX512BF16__
                __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
                __m128i _t1 = _mm_unpacklo_epi16(_r2, _r3);
                __m128i _t2 = _mm_unpacklo_epi16(_r4, _r5);
                __m128i _t3 = _mm_unpacklo_epi16(_r6, _r7);
                _r0 = _mm_unpacklo_epi32(_t0, _t1);
                _r1 = _mm_unpackhi_epi32(_t0, _t1);
                _r2 = _mm_unpacklo_epi32(_t2, _t3);
                _r3 = _mm_unpackhi_epi32(_t2, _t3);
                __m128i _p0 = _mm_unpacklo_epi64(_r0, _r2);
                __m128i _p1 = _mm_unpackhi_epi64(_r0, _r2);
                __m128i _p2 = _mm_unpacklo_epi64(_r1, _r3);
                __m128i _p3 = _mm_unpackhi_epi64(_r1, _r3);
#endif // __AVX512BF16__
                _mm_storeu_si128((__m128i*)pp, _p0);
                _mm_storeu_si128((__m128i*)(pp + 8), _p1);
                _mm_storeu_si128((__m128i*)(pp + 16), _p2);
                _mm_storeu_si128((__m128i*)(pp + 24), _p3);
                pp += 32;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + A_hstep));
                __m128i _p0 = _mm_unpacklo_epi16(_r0, _r1);
                __m128i _p1 = _mm_unpackhi_epi16(_r0, _r1);
                _mm_storeu_si128((__m128i*)pp, _p0);
                _mm_storeu_si128((__m128i*)(pp + 8), _p1);
                pp += 16;
                p0 += A_hstep * 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                pp += 8;
                p0 += A_hstep;
            }
        }
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * elempack;

#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)(p0 + 16));
                __m256i _r2 = _mm256_loadu_si256((const __m256i*)(p0 + 32));
                __m256i _r3 = _mm256_loadu_si256((const __m256i*)(p0 + 48));
#if __AVX512BF16__
                transpose8x4_epi32(_r0, _r1, _r2, _r3);
#else  // __AVX512BF16__
                transpose16x4_epi16(_r0, _r1, _r2, _r3);
#endif // __AVX512BF16__
                _mm256_storeu_si256((__m256i*)pp, _r0);
                _mm256_storeu_si256((__m256i*)(pp + 16), _r1);
                _mm256_storeu_si256((__m256i*)(pp + 32), _r2);
                _mm256_storeu_si256((__m256i*)(pp + 48), _r3);
                pp += 64;
                p0 += A_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + 8));
                __m128i _r2 = _mm_loadu_si128((const __m128i*)(p0 + 16));
                __m128i _r3 = _mm_loadu_si128((const __m128i*)(p0 + 24));
#if __AVX512BF16__
                transpose4x4_epi32(_r0, _r1, _r2, _r3);
#else  // __AVX512BF16__
                transpose8x4_epi16(_r0, _r1, _r2, _r3);
#endif // __AVX512BF16__
                _mm_storeu_si128((__m128i*)pp, _r0);
                _mm_storeu_si128((__m128i*)(pp + 8), _r1);
                _mm_storeu_si128((__m128i*)(pp + 16), _r2);
                _mm_storeu_si128((__m128i*)(pp + 24), _r3);
                pp += 32;
                p0 += A_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)(p0 + 4));
                __m128i _r2 = _mm_loadl_epi64((const __m128i*)(p0 + 8));
                __m128i _r3 = _mm_loadl_epi64((const __m128i*)(p0 + 12));
#if __AVX512BF16__
                __m128i _t0 = _mm_unpacklo_epi32(_r0, _r1);
                __m128i _t1 = _mm_unpacklo_epi32(_r2, _r3);
                _r0 = _mm_unpacklo_epi64(_t0, _t1);
                _r1 = _mm_unpackhi_epi64(_t0, _t1);
#else  // __AVX512BF16__
                __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
                __m128i _t1 = _mm_unpacklo_epi16(_r2, _r3);
                _r0 = _mm_unpacklo_epi32(_t0, _t1);
                _r1 = _mm_unpackhi_epi32(_t0, _t1);
#endif // __AVX512BF16__
                _mm_storeu_si128((__m128i*)pp, _r0);
                _mm_storeu_si128((__m128i*)(pp + 8), _r1);
                pp += 16;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _p0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _p1 = _mm_loadl_epi64((const __m128i*)(p0 + A_hstep));
                __m128i _p = _mm_unpacklo_epi16(_p0, _p1);
                _mm_storeu_si128((__m128i*)pp, _p);
                pp += 8;
                p0 += A_hstep * 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
                pp += 4;
                p0 += A_hstep;
            }
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m256i _p0 = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _p1 = _mm256_loadu_si256((const __m256i*)(p0 + 16));
#if __AVX512BF16__
                transpose8x2_epi32(_p0, _p1);
#else  // __AVX512BF16__
                transpose16x2_epi16(_p0, _p1);
#endif // __AVX512BF16__
                _mm256_storeu_si256((__m256i*)pp, _p0);
                _mm256_storeu_si256((__m256i*)(pp + 16), _p1);
                pp += 32;
                p0 += A_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _p0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _p1 = _mm_loadu_si128((const __m128i*)(p0 + 8));
#if __AVX512BF16__
                __m128i _t0 = _mm_unpacklo_epi32(_p0, _p1);
                __m128i _t1 = _mm_unpackhi_epi32(_p0, _p1);
#else  // __AVX512BF16__
                __m128i _t0 = _mm_unpacklo_epi16(_p0, _p1);
                __m128i _t1 = _mm_unpackhi_epi16(_p0, _p1);
#endif // __AVX512BF16__
                _mm_storeu_si128((__m128i*)pp, _t0);
                _mm_storeu_si128((__m128i*)(pp + 8), _t1);
                pp += 16;
                p0 += A_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __AVX512BF16__
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[4];
                pp[3] = p0[5];
                pp[4] = p0[2];
                pp[5] = p0[3];
                pp[6] = p0[6];
                pp[7] = p0[7];
#else  // __AVX512BF16__
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp[2] = p0[1];
                pp[3] = p0[5];
                pp[4] = p0[2];
                pp[5] = p0[6];
                pp[6] = p0[3];
                pp[7] = p0[7];
#endif // __AVX512BF16__
                pp += 8;
                p0 += A_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[A_hstep];
                pp[2] = p0[1];
                pp[3] = p0[A_hstep + 1];
                pp += 4;
                p0 += A_hstep * 2;
            }
#endif // __AVX512BF16__
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
        const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                _mm256_storeu_si256((__m256i*)pp, _mm256_loadu_si256((const __m256i*)p0));
                pp += 16;
                p0 += A_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                pp += 8;
                p0 += A_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
                pp += 4;
                p0 += A_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
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

static void pack_B_tile_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        pack_B_tile_bf16_avx512bf16(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("pack_B_tile_bf16 %d %d %d %d", j, max_jj, k, max_kk);
    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * elempack;

        if (elempack == 16)
        {
            int kk = 0;
#if __AVX512BF16__
            __m512i _idx = _mm512_set_epi16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512i _p = _mm512_loadu_si512((const __m512i*)p0);
                _p = _mm512_permutexvar_epi16(_idx, _p);
                _mm512_storeu_si512((__m512i*)pp, _p);
                pp += 32;
                p0 += 32;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                _mm256_storeu_si256((__m256i*)pp, _mm256_loadu_si256((const __m256i*)p0));
                pp += 16;
                p0 += 16;
            }
        }
        if (elempack == 8)
        {
            const unsigned short* p1 = p0 + B_hstep * 8;

            int kk = 0;
#if __AVX512BF16__
            __m512i _idx = _mm512_set_epi16(31, 23, 30, 22, 29, 21, 28, 20, 27, 19, 26, 18, 25, 17, 24, 16, 15, 7, 14, 6, 13, 5, 12, 4, 11, 3, 10, 2, 9, 1, 8, 0);
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _a = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _b = _mm256_loadu_si256((const __m256i*)p1);
                __m512i _ab = combine8x2_epi32(_a, _b);
                __m512i _p = _mm512_permutexvar_epi16(_idx, _ab);
                _mm512_storeu_si512((__m512i*)pp, _p);
                pp += 32;
                p0 += 16;
                p1 += 16;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
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
            const unsigned short* p1 = p0 + B_hstep * 4;
            const unsigned short* p2 = p0 + B_hstep * 8;
            const unsigned short* p3 = p0 + B_hstep * 12;

            int kk = 0;
#if __AVX512BF16__
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
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
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
            __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(B_hstep));

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512i _p = _mm512_i32gather_epi32(_vindex, (const int*)p0, sizeof(unsigned short));
                _mm512_storeu_si512((__m512i*)pp, _p);
                pp += 32;
                p0 += 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m512i _p = _mm512_i32gather_epi32(_vindex, (const int*)p0, sizeof(unsigned short));
                __m256i _p16 = _mm512_cvtepi32_epi16(_p);
                _mm256_storeu_si256((__m256i*)pp, _p16);
                pp += 16;
                p0++;
            }
        }
    }
#endif // __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * elempack;

#if __AVX__
        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512BF16__
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
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                pp += 8;
                p0 += 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const unsigned short* p1 = p0 + B_hstep * 4;

            int kk = 0;
#if __AVX512BF16__
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
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
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
#if __AVX2__
            __m256i _vindex = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(B_hstep));
#endif
            const unsigned short* p1 = p0 + B_hstep * 1;
            const unsigned short* p2 = p0 + B_hstep * 2;
            const unsigned short* p3 = p0 + B_hstep * 3;
            const unsigned short* p4 = p0 + B_hstep * 4;
            const unsigned short* p5 = p0 + B_hstep * 5;
            const unsigned short* p6 = p0 + B_hstep * 6;
            const unsigned short* p7 = p0 + B_hstep * 7;

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _p = _mm256_i32gather_epi32((const int*)p0, _vindex, sizeof(unsigned short));
                _mm256_storeu_si256((__m256i*)pp, _p);
                pp += 16;
                p0 += 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
#if __AVX2__
                __m256i _p = _mm256_i32gather_epi32((const int*)p0, _vindex, sizeof(unsigned short));
                __m128i _p16 = _mm256_comp_cvtepi32_epi16(_p);
                _mm_storeu_si128((__m128i*)pp, _p16);
#else
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp[4] = p4[0];
                pp[5] = p5[0];
                pp[6] = p6[0];
                pp[7] = p7[0];
                p1++;
                p2++;
                p3++;
                p4++;
                p5++;
                p6++;
                p7++;
#endif
                pp += 8;
                p0++;
            }
        }
    }
#else // defined(__x86_64__) || defined(_M_X64)
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        for (; jj + 15 < max_jj; jj += 16)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * elempack;

            unsigned short* pp1 = pp + max_kk * 4;
            unsigned short* pp2 = pp + max_kk * 8;
            unsigned short* pp3 = pp + max_kk * 12;

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _p0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _p1 = _mm_loadl_epi64((const __m128i*)(p0 + 4));
                __m128i _p2 = _mm_loadl_epi64((const __m128i*)(p0 + 8));
                __m128i _p3 = _mm_loadl_epi64((const __m128i*)(p0 + 12));
                __m128i _p4 = _mm_loadl_epi64((const __m128i*)(p0 + 16));
                __m128i _p5 = _mm_loadl_epi64((const __m128i*)(p0 + 20));
                __m128i _p6 = _mm_loadl_epi64((const __m128i*)(p0 + 24));
                __m128i _p7 = _mm_loadl_epi64((const __m128i*)(p0 + 28));

                __m128i _t0 = _mm_unpacklo_epi16(_p0, _p1);
                __m128i _t1 = _mm_unpacklo_epi16(_p2, _p3);
                __m128i _t2 = _mm_unpacklo_epi16(_p4, _p5);
                __m128i _t3 = _mm_unpacklo_epi16(_p6, _p7);

                _mm_storeu_si128((__m128i*)pp, _t0);
                _mm_storeu_si128((__m128i*)pp1, _t1);
                _mm_storeu_si128((__m128i*)pp2, _t2);
                _mm_storeu_si128((__m128i*)pp3, _t3);

                pp += 8;
                pp1 += 8;
                pp2 += 8;
                pp3 += 8;
                p0 += 32;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m128i _p0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _p1 = _mm_loadu_si128((const __m128i*)(p0 + 8));

                _mm_storel_pd((double*)pp, _mm_castsi128_pd(_p0));
                _mm_storeh_pd((double*)pp1, _mm_castsi128_pd(_p0));
                _mm_storel_pd((double*)pp2, _mm_castsi128_pd(_p1));
                _mm_storeh_pd((double*)pp3, _mm_castsi128_pd(_p1));

                pp += 4;
                pp1 += 4;
                pp2 += 4;
                pp3 += 4;
                p0 += 16;
            }

            pp = pp3;
        }
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * elempack;

            unsigned short* pp1 = pp + max_kk * 4;

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _p0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _p1 = _mm_loadl_epi64((const __m128i*)(p0 + 4));
                __m128i _p2 = _mm_loadl_epi64((const __m128i*)(p0 + 8));
                __m128i _p3 = _mm_loadl_epi64((const __m128i*)(p0 + 12));

                __m128i _t0 = _mm_unpacklo_epi16(_p0, _p1);
                __m128i _t1 = _mm_unpacklo_epi16(_p2, _p3);

                _mm_storeu_si128((__m128i*)pp, _t0);
                _mm_storeu_si128((__m128i*)pp1, _t1);

                pp += 8;
                pp1 += 8;
                p0 += 16;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m128i _p0 = _mm_loadu_si128((const __m128i*)p0);

                _mm_storel_pd((double*)pp, _mm_castsi128_pd(_p0));
                _mm_storeh_pd((double*)pp1, _mm_castsi128_pd(_p0));

                pp += 4;
                pp1 += 4;
                p0 += 8;
            }

            pp = pp1;
        }
    }
#endif // __AVX__
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * elempack;

        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512BF16__
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
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p1 = p0 + B_hstep * 1;
            const unsigned short* p2 = p0 + B_hstep * 2;
            const unsigned short* p3 = p0 + B_hstep * 3;

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p1[0];
                pp[3] = p1[1];
                pp[4] = p2[0];
                pp[5] = p2[1];
                pp[6] = p3[0];
                pp[7] = p3[1];
                pp += 8;
                p0 += 2;
                p1 += 2;
                p2 += 2;
                p3 += 2;
            }
#endif // __AVX512BF16__
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
        const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
        const unsigned short* p1 = p0 + B_hstep;

        // if (elempack == 1)
        {
            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p1[0];
                pp[3] = p1[1];
                pp += 4;
                p0 += 2;
                p1 += 2;
            }
#endif // __AVX512BF16__
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
        const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

        // if (elempack == 1)
        {
            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_B_tile_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        transpose_pack_B_tile_bf16_avx512bf16(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("transpose_pack_B_tile_bf16 %d %d %d %d", j, max_jj, k, max_kk);
    const int elempack = B.elempack;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    unsigned short* pp = (unsigned short*)BT;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * elempack;

        if (elempack == 16)
        {
            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m512i _r0 = _mm512_loadu_si512((const __m512i*)p0);
                __m512i _r1 = _mm512_loadu_si512((const __m512i*)(p0 + 32));
                __m512i _r2 = _mm512_loadu_si512((const __m512i*)(p0 + 64));
                __m512i _r3 = _mm512_loadu_si512((const __m512i*)(p0 + 96));
                __m512i _r4 = _mm512_loadu_si512((const __m512i*)(p0 + 128));
                __m512i _r5 = _mm512_loadu_si512((const __m512i*)(p0 + 160));
                __m512i _r6 = _mm512_loadu_si512((const __m512i*)(p0 + 192));
                __m512i _r7 = _mm512_loadu_si512((const __m512i*)(p0 + 224));

                __m512i w0 = _mm512_shuffle_i64x2(_r0, _r1, 0x44);
                __m512i w1 = _mm512_shuffle_i64x2(_r0, _r1, 0xEE);
                __m512i w2 = _mm512_shuffle_i64x2(_r2, _r3, 0x44);
                __m512i w3 = _mm512_shuffle_i64x2(_r2, _r3, 0xEE);
                __m512i w4 = _mm512_shuffle_i64x2(_r4, _r5, 0x44);
                __m512i w5 = _mm512_shuffle_i64x2(_r4, _r5, 0xEE);
                __m512i w6 = _mm512_shuffle_i64x2(_r6, _r7, 0x44);
                __m512i w7 = _mm512_shuffle_i64x2(_r6, _r7, 0xEE);

#if __AVX512BF16__
                __m512i a0 = _mm512_unpacklo_epi32(w0, w1);
                __m512i a1 = _mm512_unpackhi_epi32(w0, w1);
                __m512i a2 = _mm512_unpacklo_epi32(w2, w3);
                __m512i a3 = _mm512_unpackhi_epi32(w2, w3);
                __m512i a4 = _mm512_unpacklo_epi32(w4, w5);
                __m512i a5 = _mm512_unpackhi_epi32(w4, w5);
                __m512i a6 = _mm512_unpacklo_epi32(w6, w7);
                __m512i a7 = _mm512_unpackhi_epi32(w6, w7);

                __m512i b0 = _mm512_unpacklo_epi64(a0, a2);
                __m512i b1 = _mm512_unpackhi_epi64(a0, a2);
                __m512i b2 = _mm512_unpacklo_epi64(a1, a3);
                __m512i b3 = _mm512_unpackhi_epi64(a1, a3);
                __m512i b4 = _mm512_unpacklo_epi64(a4, a6);
                __m512i b5 = _mm512_unpackhi_epi64(a4, a6);
                __m512i b6 = _mm512_unpacklo_epi64(a5, a7);
                __m512i b7 = _mm512_unpackhi_epi64(a5, a7);

                __m512i idx_l = _mm512_set_epi32(27, 26, 19, 18, 25, 24, 17, 16, 11, 10, 3, 2, 9, 8, 1, 0);
                __m512i idx_r = _mm512_set_epi32(31, 30, 23, 22, 29, 28, 21, 20, 15, 14, 7, 6, 13, 12, 5, 4);

                __m512i _p0 = _mm512_permutex2var_epi32(b0, idx_l, b4);
                __m512i _p1 = _mm512_permutex2var_epi32(b1, idx_l, b5);
                __m512i _p2 = _mm512_permutex2var_epi32(b2, idx_l, b6);
                __m512i _p3 = _mm512_permutex2var_epi32(b3, idx_l, b7);
                __m512i _p4 = _mm512_permutex2var_epi32(b0, idx_r, b4);
                __m512i _p5 = _mm512_permutex2var_epi32(b1, idx_r, b5);
                __m512i _p6 = _mm512_permutex2var_epi32(b2, idx_r, b6);
                __m512i _p7 = _mm512_permutex2var_epi32(b3, idx_r, b7);
#else  // __AVX512BF16__
                __m512i a0 = _mm512_unpacklo_epi16(w0, w1);
                __m512i a1 = _mm512_unpackhi_epi16(w0, w1);
                __m512i a2 = _mm512_unpacklo_epi16(w2, w3);
                __m512i a3 = _mm512_unpackhi_epi16(w2, w3);
                __m512i a4 = _mm512_unpacklo_epi16(w4, w5);
                __m512i a5 = _mm512_unpackhi_epi16(w4, w5);
                __m512i a6 = _mm512_unpacklo_epi16(w6, w7);
                __m512i a7 = _mm512_unpackhi_epi16(w6, w7);

                __m512i b0 = _mm512_unpacklo_epi32(a0, a2);
                __m512i b1 = _mm512_unpackhi_epi32(a0, a2);
                __m512i b2 = _mm512_unpacklo_epi32(a1, a3);
                __m512i b3 = _mm512_unpackhi_epi32(a1, a3);
                __m512i b4 = _mm512_unpacklo_epi32(a4, a6);
                __m512i b5 = _mm512_unpackhi_epi32(a4, a6);
                __m512i b6 = _mm512_unpacklo_epi32(a5, a7);
                __m512i b7 = _mm512_unpackhi_epi32(a5, a7);

                __m512i c0 = _mm512_unpacklo_epi64(b0, b4);
                __m512i c1 = _mm512_unpackhi_epi64(b0, b4);
                __m512i c2 = _mm512_unpacklo_epi64(b1, b5);
                __m512i c3 = _mm512_unpackhi_epi64(b1, b5);
                __m512i c4 = _mm512_unpacklo_epi64(b2, b6);
                __m512i c5 = _mm512_unpackhi_epi64(b2, b6);
                __m512i c6 = _mm512_unpacklo_epi64(b3, b7);
                __m512i c7 = _mm512_unpackhi_epi64(b3, b7);

                __m512i idx_lo = _mm512_set_epi32(27, 19, 26, 18, 25, 17, 24, 16, 11, 3, 10, 2, 9, 1, 8, 0);
                __m512i idx_hi = _mm512_set_epi32(31, 23, 30, 22, 29, 21, 28, 20, 15, 7, 14, 6, 13, 5, 12, 4);

                __m512i _p0 = _mm512_permutex2var_epi32(c0, idx_lo, c1); // col 0,1
                __m512i _p1 = _mm512_permutex2var_epi32(c2, idx_lo, c3); // col 2,3
                __m512i _p2 = _mm512_permutex2var_epi32(c4, idx_lo, c5); // col 4,5
                __m512i _p3 = _mm512_permutex2var_epi32(c6, idx_lo, c7); // col 6,7
                __m512i _p4 = _mm512_permutex2var_epi32(c0, idx_hi, c1); // col 8,9
                __m512i _p5 = _mm512_permutex2var_epi32(c2, idx_hi, c3); // col A,B
                __m512i _p6 = _mm512_permutex2var_epi32(c4, idx_hi, c5); // col C,D
                __m512i _p7 = _mm512_permutex2var_epi32(c6, idx_hi, c7); // col E,F
#endif // __AVX512BF16__

                _mm512_storeu_si512((__m512i*)pp, _p0);
                _mm512_storeu_si512((__m512i*)(pp + 32), _p1);
                _mm512_storeu_si512((__m512i*)(pp + 64), _p2);
                _mm512_storeu_si512((__m512i*)(pp + 96), _p3);
                _mm512_storeu_si512((__m512i*)(pp + 128), _p4);
                _mm512_storeu_si512((__m512i*)(pp + 160), _p5);
                _mm512_storeu_si512((__m512i*)(pp + 192), _p6);
                _mm512_storeu_si512((__m512i*)(pp + 224), _p7);
                pp += 256;
                p0 += B_hstep * 16;
            }
        }
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m512i _r0 = _mm512_loadu_si512((const __m512i*)p0);
                __m512i _r1 = _mm512_loadu_si512((const __m512i*)(p0 + 32));
                __m512i _r2 = _mm512_loadu_si512((const __m512i*)(p0 + 64));
                __m512i _r3 = _mm512_loadu_si512((const __m512i*)(p0 + 96));
#if __AVX512BF16__
                __m512i idx0 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 28, 24, 20, 16, 12, 8, 4, 0);
                __m512i idx1 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 29, 25, 21, 17, 13, 9, 5, 1);
                __m512i idx2 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 30, 26, 22, 18, 14, 10, 6, 2);
                __m512i idx3 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 31, 27, 23, 19, 15, 11, 7, 3);

                __m512i lo0 = _mm512_permutex2var_epi32(_r0, idx0, _r1);
                __m512i lo1 = _mm512_permutex2var_epi32(_r0, idx1, _r1);
                __m512i lo2 = _mm512_permutex2var_epi32(_r0, idx2, _r1);
                __m512i lo3 = _mm512_permutex2var_epi32(_r0, idx3, _r1);

                __m512i hi0 = _mm512_permutex2var_epi32(_r2, idx0, _r3);
                __m512i hi1 = _mm512_permutex2var_epi32(_r2, idx1, _r3);
                __m512i hi2 = _mm512_permutex2var_epi32(_r2, idx2, _r3);
                __m512i hi3 = _mm512_permutex2var_epi32(_r2, idx3, _r3);

                __m512i _p0 = _mm512_inserti64x4(lo0, _mm512_castsi512_si256(hi0), 1);
                __m512i _p1 = _mm512_inserti64x4(lo1, _mm512_castsi512_si256(hi1), 1);
                __m512i _p2 = _mm512_inserti64x4(lo2, _mm512_castsi512_si256(hi2), 1);
                __m512i _p3 = _mm512_inserti64x4(lo3, _mm512_castsi512_si256(hi3), 1);
#else  // __AVX512BF16__
                __m512i id0 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 28, 24, 20, 16, 12, 8, 4, 0);
                __m512i id1 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 29, 25, 21, 17, 13, 9, 5, 1);
                __m512i id2 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 30, 26, 22, 18, 14, 10, 6, 2);
                __m512i id3 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 31, 27, 23, 19, 15, 11, 7, 3);

                __m512i p0_lo = _mm512_permutex2var_epi32(_r0, id0, _r1);
                __m512i p1_lo = _mm512_permutex2var_epi32(_r0, id1, _r1);
                __m512i p2_lo = _mm512_permutex2var_epi32(_r0, id2, _r1);
                __m512i p3_lo = _mm512_permutex2var_epi32(_r0, id3, _r1);

                __m512i p0_hi = _mm512_permutex2var_epi32(_r2, id0, _r3);
                __m512i p1_hi = _mm512_permutex2var_epi32(_r2, id1, _r3);
                __m512i p2_hi = _mm512_permutex2var_epi32(_r2, id2, _r3);
                __m512i p3_hi = _mm512_permutex2var_epi32(_r2, id3, _r3);

                __m512i cp0 = _mm512_inserti64x4(p0_lo, _mm512_castsi512_si256(p0_hi), 1);
                __m512i cp1 = _mm512_inserti64x4(p1_lo, _mm512_castsi512_si256(p1_hi), 1);
                __m512i cp2 = _mm512_inserti64x4(p2_lo, _mm512_castsi512_si256(p2_hi), 1);
                __m512i cp3 = _mm512_inserti64x4(p3_lo, _mm512_castsi512_si256(p3_hi), 1);

                __m512i shuf = _mm512_set4_epi32(0x0f0e0b0a, 0x07060302, 0x0d0c0908, 0x05040100);
                __m512i pq = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

                __m512i s0 = _mm512_shuffle_epi8(cp0, shuf);
                __m512i s1 = _mm512_shuffle_epi8(cp1, shuf);
                __m512i s2 = _mm512_shuffle_epi8(cp2, shuf);
                __m512i s3 = _mm512_shuffle_epi8(cp3, shuf);

                __m512i _p0 = _mm512_permutexvar_epi64(pq, s0);
                __m512i _p1 = _mm512_permutexvar_epi64(pq, s1);
                __m512i _p2 = _mm512_permutexvar_epi64(pq, s2);
                __m512i _p3 = _mm512_permutexvar_epi64(pq, s3);
#endif // __AVX512BF16__
                _mm512_storeu_si512((__m512i*)pp, _p0);
                _mm512_storeu_si512((__m512i*)(pp + 32), _p1);
                _mm512_storeu_si512((__m512i*)(pp + 64), _p2);
                _mm512_storeu_si512((__m512i*)(pp + 96), _p3);
                pp += 128;
                p0 += B_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512i _r0 = _mm512_loadu_si512((const __m512i*)p0);
                __m512i _r1 = _mm512_loadu_si512((const __m512i*)(p0 + 32));
#if __AVX512BF16__
                __m512i idx_lo = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
                __m512i idx_hi = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
                __m512i _p0 = _mm512_permutex2var_epi32(_r0, idx_lo, _r1);
                __m512i _p1 = _mm512_permutex2var_epi32(_r0, idx_hi, _r1);
#else  // __AVX512BF16__
                __m512i idx_lo = _mm512_set_epi16(61, 57, 53, 49, 45, 41, 37, 33, 29, 25, 21, 17, 13, 9, 5, 1, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0);
                __m512i idx_hi = _mm512_set_epi16(63, 59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11, 7, 3, 62, 58, 54, 50, 46, 42, 38, 34, 30, 26, 22, 18, 14, 10, 6, 2);
                __m512i _p0 = _mm512_permutex2var_epi16(_r0, idx_lo, _r1);
                __m512i _p1 = _mm512_permutex2var_epi16(_r0, idx_hi, _r1);
#endif // __AVX512BF16__
                _mm512_storeu_si512((__m512i*)pp, _p0);
                _mm512_storeu_si512((__m512i*)(pp + 32), _p1);
                pp += 64;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)(p0 + B_hstep));
                transpose16x2_epi16(_r0, _r1);
                _mm256_storeu_si256((__m256i*)pp, _r0);
                _mm256_storeu_si256((__m256i*)(pp + 16), _r1);
                pp += 32;
                p0 += B_hstep * 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                _mm256_storeu_si256((__m256i*)pp, _mm256_loadu_si256((const __m256i*)p0));
                pp += 16;
                p0 += B_hstep;
            }
        }
    }
#endif // __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * elempack;

#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
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
#if __AVX512BF16__
                transpose8x8_epi32(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
#else  // __AVX512BF16__
                transpose16x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
#endif // __AVX512BF16__
                _mm256_storeu_si256((__m256i*)pp, _r0);
                _mm256_storeu_si256((__m256i*)(pp + 16), _r1);
                _mm256_storeu_si256((__m256i*)(pp + 32), _r2);
                _mm256_storeu_si256((__m256i*)(pp + 48), _r3);
                _mm256_storeu_si256((__m256i*)(pp + 64), _r4);
                _mm256_storeu_si256((__m256i*)(pp + 80), _r5);
                _mm256_storeu_si256((__m256i*)(pp + 96), _r6);
                _mm256_storeu_si256((__m256i*)(pp + 112), _r7);
                pp += 128;
                p0 += B_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + 8));
                __m128i _r2 = _mm_loadu_si128((const __m128i*)(p0 + 16));
                __m128i _r3 = _mm_loadu_si128((const __m128i*)(p0 + 24));
                __m128i _r4 = _mm_loadu_si128((const __m128i*)(p0 + 32));
                __m128i _r5 = _mm_loadu_si128((const __m128i*)(p0 + 40));
                __m128i _r6 = _mm_loadu_si128((const __m128i*)(p0 + 48));
                __m128i _r7 = _mm_loadu_si128((const __m128i*)(p0 + 56));
#if __AVX512BF16__
                transpose4x8_epi32(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
#else  // __AVX512BF16__
                transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
#endif // __AVX512BF16__
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
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)(p0 + 4));
                __m128i _r2 = _mm_loadl_epi64((const __m128i*)(p0 + 8));
                __m128i _r3 = _mm_loadl_epi64((const __m128i*)(p0 + 12));
                __m128i _r4 = _mm_loadl_epi64((const __m128i*)(p0 + 16));
                __m128i _r5 = _mm_loadl_epi64((const __m128i*)(p0 + 20));
                __m128i _r6 = _mm_loadl_epi64((const __m128i*)(p0 + 24));
                __m128i _r7 = _mm_loadl_epi64((const __m128i*)(p0 + 28));
#if __AVX512BF16__
                __m128i _t0 = _mm_unpacklo_epi32(_r0, _r1);
                __m128i _t1 = _mm_unpacklo_epi32(_r2, _r3);
                __m128i _t2 = _mm_unpacklo_epi32(_r4, _r5);
                __m128i _t3 = _mm_unpacklo_epi32(_r6, _r7);
                __m128i _p0 = _mm_unpacklo_epi64(_t0, _t1);
                __m128i _p1 = _mm_unpacklo_epi64(_t2, _t3);
                __m128i _p2 = _mm_unpackhi_epi64(_t0, _t1);
                __m128i _p3 = _mm_unpackhi_epi64(_t2, _t3);
#else  // __AVX512BF16__
                __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
                __m128i _t1 = _mm_unpacklo_epi16(_r2, _r3);
                __m128i _t2 = _mm_unpacklo_epi16(_r4, _r5);
                __m128i _t3 = _mm_unpacklo_epi16(_r6, _r7);
                _r0 = _mm_unpacklo_epi32(_t0, _t1);
                _r1 = _mm_unpackhi_epi32(_t0, _t1);
                _r2 = _mm_unpacklo_epi32(_t2, _t3);
                _r3 = _mm_unpackhi_epi32(_t2, _t3);
                __m128i _p0 = _mm_unpacklo_epi64(_r0, _r2);
                __m128i _p1 = _mm_unpackhi_epi64(_r0, _r2);
                __m128i _p2 = _mm_unpacklo_epi64(_r1, _r3);
                __m128i _p3 = _mm_unpackhi_epi64(_r1, _r3);
#endif // __AVX512BF16__
                _mm_storeu_si128((__m128i*)pp, _p0);
                _mm_storeu_si128((__m128i*)(pp + 8), _p1);
                _mm_storeu_si128((__m128i*)(pp + 16), _p2);
                _mm_storeu_si128((__m128i*)(pp + 24), _p3);
                pp += 32;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + B_hstep));
                __m128i _p0 = _mm_unpacklo_epi16(_r0, _r1);
                __m128i _p1 = _mm_unpackhi_epi16(_r0, _r1);
                _mm_storeu_si128((__m128i*)pp, _p0);
                _mm_storeu_si128((__m128i*)(pp + 8), _p1);
                pp += 16;
                p0 += B_hstep * 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                pp += 8;
                p0 += B_hstep;
            }
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * elempack;

#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)(p0 + 16));
                __m256i _r2 = _mm256_loadu_si256((const __m256i*)(p0 + 32));
                __m256i _r3 = _mm256_loadu_si256((const __m256i*)(p0 + 48));
#if __AVX512BF16__
                transpose8x4_epi32(_r0, _r1, _r2, _r3);
#else  // __AVX512BF16__
                transpose16x4_epi16(_r0, _r1, _r2, _r3);
#endif // __AVX512BF16__
                _mm256_storeu_si256((__m256i*)pp, _r0);
                _mm256_storeu_si256((__m256i*)(pp + 16), _r1);
                _mm256_storeu_si256((__m256i*)(pp + 32), _r2);
                _mm256_storeu_si256((__m256i*)(pp + 48), _r3);
                pp += 64;
                p0 += B_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + 8));
                __m128i _r2 = _mm_loadu_si128((const __m128i*)(p0 + 16));
                __m128i _r3 = _mm_loadu_si128((const __m128i*)(p0 + 24));
#if __AVX512BF16__
                transpose4x4_epi32(_r0, _r1, _r2, _r3);
#else  // __AVX512BF16__
                transpose8x4_epi16(_r0, _r1, _r2, _r3);
#endif // __AVX512BF16__
                _mm_storeu_si128((__m128i*)pp, _r0);
                _mm_storeu_si128((__m128i*)(pp + 8), _r1);
                _mm_storeu_si128((__m128i*)(pp + 16), _r2);
                _mm_storeu_si128((__m128i*)(pp + 24), _r3);
                pp += 32;
                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)(p0 + 4));
                __m128i _r2 = _mm_loadl_epi64((const __m128i*)(p0 + 8));
                __m128i _r3 = _mm_loadl_epi64((const __m128i*)(p0 + 12));
#if __AVX512BF16__
                __m128i _t0 = _mm_unpacklo_epi32(_r0, _r1);
                __m128i _t1 = _mm_unpacklo_epi32(_r2, _r3);
                _r0 = _mm_unpacklo_epi64(_t0, _t1);
                _r1 = _mm_unpackhi_epi64(_t0, _t1);
#else  // __AVX512BF16__
                __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
                __m128i _t1 = _mm_unpacklo_epi16(_r2, _r3);
                _r0 = _mm_unpacklo_epi32(_t0, _t1);
                _r1 = _mm_unpackhi_epi32(_t0, _t1);
#endif // __AVX512BF16__
                _mm_storeu_si128((__m128i*)pp, _r0);
                _mm_storeu_si128((__m128i*)(pp + 8), _r1);
                pp += 16;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _p0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _p1 = _mm_loadl_epi64((const __m128i*)(p0 + B_hstep));
                __m128i _p = _mm_unpacklo_epi16(_p0, _p1);
                _mm_storeu_si128((__m128i*)pp, _p);
                pp += 8;
                p0 += B_hstep * 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                __m256i _p0 = _mm256_loadu_si256((const __m256i*)p0);
                __m256i _p1 = _mm256_loadu_si256((const __m256i*)(p0 + 16));
#if __AVX512BF16__
                transpose8x2_epi32(_p0, _p1);
#else  // __AVX512BF16__
                transpose16x2_epi16(_p0, _p1);
#endif // __AVX512BF16__
                _mm256_storeu_si256((__m256i*)pp, _p0);
                _mm256_storeu_si256((__m256i*)(pp + 16), _p1);
                pp += 32;
                p0 += B_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                __m128i _p0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _p1 = _mm_loadu_si128((const __m128i*)(p0 + 8));
#if __AVX512BF16__
                __m128i _t0 = _mm_unpacklo_epi32(_p0, _p1);
                __m128i _t1 = _mm_unpackhi_epi32(_p0, _p1);
#else  // __AVX512BF16__
                __m128i _t0 = _mm_unpacklo_epi16(_p0, _p1);
                __m128i _t1 = _mm_unpackhi_epi16(_p0, _p1);
#endif // __AVX512BF16__
                _mm_storeu_si128((__m128i*)pp, _t0);
                _mm_storeu_si128((__m128i*)(pp + 8), _t1);
                pp += 16;
                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __AVX512BF16__
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[4];
                pp[3] = p0[5];
                pp[4] = p0[2];
                pp[5] = p0[3];
                pp[6] = p0[6];
                pp[7] = p0[7];
#else  // __AVX512BF16__
                pp[0] = p0[0];
                pp[1] = p0[4];
                pp[2] = p0[1];
                pp[3] = p0[5];
                pp[4] = p0[2];
                pp[5] = p0[6];
                pp[6] = p0[3];
                pp[7] = p0[7];
#endif // __AVX512BF16__
                pp += 8;
                p0 += B_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[B_hstep];
                pp[2] = p0[1];
                pp[3] = p0[B_hstep + 1];
                pp += 4;
                p0 += B_hstep * 2;
            }
#endif // __AVX512BF16__
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
        const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                _mm256_storeu_si256((__m256i*)pp, _mm256_loadu_si256((const __m256i*)p0));
                pp += 16;
                p0 += B_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                pp += 8;
                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
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

static void gemm_transB_packed_tile_bf16s(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        gemm_transB_packed_tile_bf16s_avx512bf16(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("gemm_transB_packed_tile_bf16s %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);
    // actually we only depend the global k==0 condition
    (void)i;
    (void)j;

    const unsigned short* pAT = AT_tile;
    const unsigned short* pBT = BT_tile;

    float* outptr = topT_tile;

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

            __m512 _sum0 = _mm512_setzero_ps();
            __m512 _sum1 = _mm512_setzero_ps();
            __m512 _sum2 = _mm512_setzero_ps();
            __m512 _sum3 = _mm512_setzero_ps();
            __m512 _sum4 = _mm512_setzero_ps();
            __m512 _sum5 = _mm512_setzero_ps();
            __m512 _sum6 = _mm512_setzero_ps();
            __m512 _sum7 = _mm512_setzero_ps();
            __m512 _sum8 = _mm512_setzero_ps();
            __m512 _sum9 = _mm512_setzero_ps();
            __m512 _suma = _mm512_setzero_ps();
            __m512 _sumb = _mm512_setzero_ps();
            __m512 _sumc = _mm512_setzero_ps();
            __m512 _sumd = _mm512_setzero_ps();
            __m512 _sume = _mm512_setzero_ps();
            __m512 _sumf = _mm512_setzero_ps();

            if (k != 0)
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
            for (; kk + 1 < max_kk; kk += 2)
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
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m512 _pA0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pA));
                __m512 _pB0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pB));

                __m512 _pA1 = _mm512_permute_ps(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512 _pB1 = _mm512_permute_ps(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                __m512 _pA2 = _mm512_shuffle_f32x4(_pA0, _pA0, _MM_SHUFFLE(2, 3, 0, 1));
                __m512 _pB2 = _mm512_shuffle_f32x4(_pB0, _pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512 _pA3 = _mm512_permute_ps(_pA2, _MM_SHUFFLE(1, 0, 3, 2));
                __m512 _pB3 = _mm512_permute_ps(_pB2, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm512_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm512_fmadd_ps(_pA0, _pB1, _sum1);
                _sum2 = _mm512_fmadd_ps(_pA1, _pB0, _sum2);
                _sum3 = _mm512_fmadd_ps(_pA1, _pB1, _sum3);
                _sum4 = _mm512_fmadd_ps(_pA0, _pB2, _sum4);
                _sum5 = _mm512_fmadd_ps(_pA0, _pB3, _sum5);
                _sum6 = _mm512_fmadd_ps(_pA1, _pB2, _sum6);
                _sum7 = _mm512_fmadd_ps(_pA1, _pB3, _sum7);
                _sum8 = _mm512_fmadd_ps(_pA2, _pB0, _sum8);
                _sum9 = _mm512_fmadd_ps(_pA2, _pB1, _sum9);
                _suma = _mm512_fmadd_ps(_pA3, _pB0, _suma);
                _sumb = _mm512_fmadd_ps(_pA3, _pB1, _sumb);
                _sumc = _mm512_fmadd_ps(_pA2, _pB2, _sumc);
                _sumd = _mm512_fmadd_ps(_pA2, _pB3, _sumd);
                _sume = _mm512_fmadd_ps(_pA3, _pB2, _sume);
                _sumf = _mm512_fmadd_ps(_pA3, _pB3, _sumf);

                pA += 16;
                pB += 16;
            }

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
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

            __m512 _sum0 = _mm512_setzero_ps();
            __m512 _sum1 = _mm512_setzero_ps();
            __m512 _sum2 = _mm512_setzero_ps();
            __m512 _sum3 = _mm512_setzero_ps();
            __m512 _sum4 = _mm512_setzero_ps();
            __m512 _sum5 = _mm512_setzero_ps();
            __m512 _sum6 = _mm512_setzero_ps();
            __m512 _sum7 = _mm512_setzero_ps();

            if (k != 0)
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
            for (; kk + 1 < max_kk; kk += 2)
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
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m512 _pA0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pA));
                __m256 _pBB = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pB));
                __m512 _pB0 = _mm512_castsi512_ps(combine8x2_epi32(_mm256_castps_si256(_pBB), _mm256_castps_si256(_pBB)));

                __m512 _pA1 = _mm512_permute_ps(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512 _pB1 = _mm512_permute_ps(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                __m512 _pB2 = _mm512_castsi512_ps(_mm512_permutex_epi64(_mm512_castps_si512(_pB0), _MM_SHUFFLE(1, 0, 3, 2)));
                __m512 _pB3 = _mm512_permute_ps(_pB2, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm512_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm512_fmadd_ps(_pA0, _pB1, _sum1);
                _sum2 = _mm512_fmadd_ps(_pA1, _pB0, _sum2);
                _sum3 = _mm512_fmadd_ps(_pA1, _pB1, _sum3);
                _sum4 = _mm512_fmadd_ps(_pA0, _pB2, _sum4);
                _sum5 = _mm512_fmadd_ps(_pA0, _pB3, _sum5);
                _sum6 = _mm512_fmadd_ps(_pA1, _pB2, _sum6);
                _sum7 = _mm512_fmadd_ps(_pA1, _pB3, _sum7);

                pA += 16;
                pB += 8;
            }

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
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

            __m512 _sum0 = _mm512_setzero_ps();
            __m512 _sum1 = _mm512_setzero_ps();
            __m512 _sum2 = _mm512_setzero_ps();
            __m512 _sum3 = _mm512_setzero_ps();

            if (k != 0)
            {
                _sum0 = _mm512_load_ps(outptr);
                _sum1 = _mm512_load_ps(outptr + 16);
                _sum2 = _mm512_load_ps(outptr + 32);
                _sum3 = _mm512_load_ps(outptr + 48);
            }

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _pB0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)pB));
                __m512i _pA1 = _mm512_alignr_epi8(_pA0, _pA0, 8);
                __m512i _pB1 = _mm512_alignr_epi8(_pB0, _pB0, 4);
                _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_pA0, (__m512bh)_pB0);
                _sum1 = _mm512_dpbf16_ps(_sum1, (__m512bh)_pA0, (__m512bh)_pB1);
                _sum2 = _mm512_dpbf16_ps(_sum2, (__m512bh)_pA1, (__m512bh)_pB0);
                _sum3 = _mm512_dpbf16_ps(_sum3, (__m512bh)_pA1, (__m512bh)_pB1);
                pA += 32;
                pB += 8;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m512 _pA0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pA));
                __m128 _pBs = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pB));
                __m512 _pB0 = _mm512_broadcast_f32x4(_pBs);

                __m512 _pA1 = _mm512_permute_ps(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512 _pB1 = _mm512_permute_ps(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm512_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm512_fmadd_ps(_pA0, _pB1, _sum1);
                _sum2 = _mm512_fmadd_ps(_pA1, _pB0, _sum2);
                _sum3 = _mm512_fmadd_ps(_pA1, _pB1, _sum3);

                pA += 16;
                pB += 4;
            }

            _mm512_store_ps(outptr, _sum0);
            _mm512_store_ps(outptr + 16, _sum1);
            _mm512_store_ps(outptr + 32, _sum2);
            _mm512_store_ps(outptr + 48, _sum3);
            outptr += 64;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* pA = pAT;

            __m512 _sum0 = _mm512_setzero_ps();
            __m512 _sum1 = _mm512_setzero_ps();

            if (k != 0)
            {
                _sum0 = _mm512_load_ps(outptr);
                _sum1 = _mm512_load_ps(outptr + 16);
            }

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512i _pA = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _pB0 = _mm512_castpd_si512(_mm512_set1_pd(((const double*)pB)[0]));
                __m512i _pB1 = _mm512_alignr_epi8(_pB0, _pB0, 4);
                _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_pA, (__m512bh)_pB0);
                _sum1 = _mm512_dpbf16_ps(_sum1, (__m512bh)_pA, (__m512bh)_pB1);
                pA += 32;
                pB += 4;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m512 _pA0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pA));
                __m512 _pB0 = bfloat2float_avx512(_mm256_set1_epi32(((const int*)pB)[0]));
                __m512 _pB1 = _mm512_permute_ps(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm512_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm512_fmadd_ps(_pA0, _pB1, _sum1);

                pA += 16;
                pB += 2;
            }

            _mm512_store_ps(outptr, _sum0);
            _mm512_store_ps(outptr + 16, _sum1);
            outptr += 32;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned short* pA = pAT;

            __m512 _sum0 = _mm512_setzero_ps();

            if (k != 0)
            {
                _sum0 = _mm512_load_ps(outptr);
            }

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512i _pA = _mm512_loadu_si512((const __m512i*)pA);
                _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_pA, (__m512bh)_mm512_set1_epi32(((const int*)pB)[0]));
                pA += 32;
                pB += 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m512 _pA0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pA));
                __m512 _pB0 = _mm512_set1_ps(bfloat16_to_float32(pB[0]));

                _sum0 = _mm512_fmadd_ps(_pA0, _pB0, _sum0);

                pA += 16;
                pB += 1;
            }

            _mm512_store_ps(outptr, _sum0);
            outptr += 16;
        }

        pAT += max_kk * 16;
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            const unsigned short* pA = pAT;

            __m512 _sum0 = _mm512_setzero_ps();
            __m512 _sum1 = _mm512_setzero_ps();
            __m512 _sum2 = _mm512_setzero_ps();
            __m512 _sum3 = _mm512_setzero_ps();
            __m512 _sum4 = _mm512_setzero_ps();
            __m512 _sum5 = _mm512_setzero_ps();
            __m512 _sum6 = _mm512_setzero_ps();
            __m512 _sum7 = _mm512_setzero_ps();

            if (k != 0)
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);
                __m512i _pA00 = combine8x2_epi32(_pA0, _pA0);
                __m512i _pA11 = _mm512_alignr_epi8(_pA00, _pA00, 8);
                __m512i _pB1 = _mm512_alignr_epi8(_pB0, _pB0, 4);
                __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pB3 = _mm512_alignr_epi8(_pB2, _pB2, 4);
                _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_pA00, (__m512bh)_pB0);
                _sum1 = _mm512_dpbf16_ps(_sum1, (__m512bh)_pA00, (__m512bh)_pB1);
                _sum2 = _mm512_dpbf16_ps(_sum2, (__m512bh)_pA11, (__m512bh)_pB0);
                _sum3 = _mm512_dpbf16_ps(_sum3, (__m512bh)_pA11, (__m512bh)_pB1);
                _sum4 = _mm512_dpbf16_ps(_sum4, (__m512bh)_pA00, (__m512bh)_pB2);
                _sum5 = _mm512_dpbf16_ps(_sum5, (__m512bh)_pA00, (__m512bh)_pB3);
                _sum6 = _mm512_dpbf16_ps(_sum6, (__m512bh)_pA11, (__m512bh)_pB2);
                _sum7 = _mm512_dpbf16_ps(_sum7, (__m512bh)_pA11, (__m512bh)_pB3);
                pA += 16;
                pB += 32;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m256 _pAA = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pA));
                __m512 _pA0 = _mm512_castsi512_ps(combine8x2_epi32(_mm256_castps_si256(_pAA), _mm256_castps_si256(_pAA)));
                __m512 _pB0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pB));

                __m512 _pA1 = _mm512_permute_ps(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512 _pB1 = _mm512_permute_ps(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                __m512 _pB2 = _mm512_castsi512_ps(_mm512_permutex_epi64(_mm512_castps_si512(_pB0), _MM_SHUFFLE(1, 0, 3, 2)));
                __m512 _pB3 = _mm512_permute_ps(_pB2, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm512_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm512_fmadd_ps(_pA0, _pB1, _sum1);
                _sum2 = _mm512_fmadd_ps(_pA1, _pB0, _sum2);
                _sum3 = _mm512_fmadd_ps(_pA1, _pB1, _sum3);
                _sum4 = _mm512_fmadd_ps(_pA0, _pB2, _sum4);
                _sum5 = _mm512_fmadd_ps(_pA0, _pB3, _sum5);
                _sum6 = _mm512_fmadd_ps(_pA1, _pB2, _sum6);
                _sum7 = _mm512_fmadd_ps(_pA1, _pB3, _sum7);

                pA += 8;
                pB += 16;
            }

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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB0 = _mm256_loadu_si256((const __m256i*)pB);
                __m256i _pA1 = _mm256_alignr_epi8(_pA0, _pA0, 8);
                __m256i _pB1 = _mm256_alignr_epi8(_pB0, _pB0, 4);
                __m256i _pB2 = _mm256_permute4x64_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB3 = _mm256_alignr_epi8(_pB2, _pB2, 4);
                _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_pA0, (__m256bh)_pB0);
                _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_pA0, (__m256bh)_pB1);
                _sum2 = _mm256_dpbf16_ps(_sum2, (__m256bh)_pA1, (__m256bh)_pB0);
                _sum3 = _mm256_dpbf16_ps(_sum3, (__m256bh)_pA1, (__m256bh)_pB1);
                _sum4 = _mm256_dpbf16_ps(_sum4, (__m256bh)_pA0, (__m256bh)_pB2);
                _sum5 = _mm256_dpbf16_ps(_sum5, (__m256bh)_pA0, (__m256bh)_pB3);
                _sum6 = _mm256_dpbf16_ps(_sum6, (__m256bh)_pA1, (__m256bh)_pB2);
                _sum7 = _mm256_dpbf16_ps(_sum7, (__m256bh)_pA1, (__m256bh)_pB3);
                pA += 16;
                pB += 16;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m256 _pA0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pA));
                __m256 _pB0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pB));

                __m256 _pA1 = _mm256_permute_ps(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256 _pB1 = _mm256_permute_ps(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                __m256 _pB2 = _mm256_permute2f128_ps(_pB0, _pB0, _MM_SHUFFLE(0, 0, 0, 1));
                __m256 _pB3 = _mm256_permute_ps(_pB2, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm256_comp_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_pA0, _pB1, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_pA1, _pB0, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_pA1, _pB1, _sum3);
                _sum4 = _mm256_comp_fmadd_ps(_pA0, _pB2, _sum4);
                _sum5 = _mm256_comp_fmadd_ps(_pA0, _pB3, _sum5);
                _sum6 = _mm256_comp_fmadd_ps(_pA1, _pB2, _sum6);
                _sum7 = _mm256_comp_fmadd_ps(_pA1, _pB3, _sum7);

                pA += 8;
                pB += 8;
            }

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
#endif // defined(__x86_64__) || defined(_M_X64)
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);
                __m256i _pB0 = combine4x2_epi32(_pB, _pB);
                __m256i _pA1 = _mm256_alignr_epi8(_pA0, _pA0, 8);
                __m256i _pB1 = _mm256_alignr_epi8(_pB0, _pB0, 4);
                _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_pA0, (__m256bh)_pB0);
                _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_pA0, (__m256bh)_pB1);
                _sum2 = _mm256_dpbf16_ps(_sum2, (__m256bh)_pA1, (__m256bh)_pB0);
                _sum3 = _mm256_dpbf16_ps(_sum3, (__m256bh)_pA1, (__m256bh)_pB1);
                pA += 16;
                pB += 8;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m256 _pA0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pA));
                __m128 _pBs = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pB));
                __m256 _pB0 = combine4x2_ps(_pBs, _pBs);

                __m256 _pA1 = _mm256_permute_ps(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256 _pB1 = _mm256_permute_ps(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm256_comp_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_pA0, _pB1, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_pA1, _pB0, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_pA1, _pB1, _sum3);

                pA += 8;
                pB += 4;
            }

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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB0 = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)pB));
                __m256i _pB1 = _mm256_alignr_epi8(_pB0, _pB0, 4);
                _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_pA, (__m256bh)_pB0);
                _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_pA, (__m256bh)_pB1);
                pA += 16;
                pB += 4;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m256 _pA0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pA));
                __m256 _pB0 = bfloat2float_avx(_mm_castps_si128(_mm_load1_ps((const float*)pB)));
                __m256 _pB1 = _mm256_permute_ps(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm256_comp_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_pA0, _pB1, _sum1);

                pA += 8;
                pB += 2;
            }

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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_pA, (__m256bh)_mm256_set1_epi32(((const int*)pB)[0]));
                pA += 16;
                pB += 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m256 _pA0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pA));
                __m256 _pB0 = _mm256_set1_ps(bfloat16_to_float32(pB[0]));

                _sum0 = _mm256_comp_fmadd_ps(_pA0, _pB0, _sum0);

                pA += 8;
                pB += 1;
            }

            _mm256_store_ps(outptr, _sum0);
            outptr += 8;
        }

        pAT += max_kk * 8;
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            const unsigned short* pA = pAT;

            __m512 _sum0 = _mm512_setzero_ps();
            __m512 _sum1 = _mm512_setzero_ps();
            __m512 _sum2 = _mm512_setzero_ps();
            __m512 _sum3 = _mm512_setzero_ps();

            if (k != 0)
            {
                _sum0 = _mm512_loadu_ps(outptr);
                _sum1 = _mm512_loadu_ps(outptr + 16);
                _sum2 = _mm512_loadu_ps(outptr + 32);
                _sum3 = _mm512_loadu_ps(outptr + 48);
            }

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512i _pA0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)pA));
                __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);
                __m512i _pA1 = _mm512_alignr_epi8(_pA0, _pA0, 8);
                __m512i _pB1 = _mm512_alignr_epi8(_pB0, _pB0, 4);
                _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_pA0, (__m512bh)_pB0);
                _sum1 = _mm512_dpbf16_ps(_sum1, (__m512bh)_pA0, (__m512bh)_pB1);
                _sum2 = _mm512_dpbf16_ps(_sum2, (__m512bh)_pA1, (__m512bh)_pB0);
                _sum3 = _mm512_dpbf16_ps(_sum3, (__m512bh)_pA1, (__m512bh)_pB1);
                pA += 8;
                pB += 32;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m128 _pAs = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pA));
                __m512 _pA0 = _mm512_broadcast_f32x4(_pAs);
                __m512 _pB0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pB));

                __m512 _pA1 = _mm512_permute_ps(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512 _pB1 = _mm512_permute_ps(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm512_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm512_fmadd_ps(_pA0, _pB1, _sum1);
                _sum2 = _mm512_fmadd_ps(_pA1, _pB0, _sum2);
                _sum3 = _mm512_fmadd_ps(_pA1, _pB1, _sum3);

                pA += 4;
                pB += 16;
            }

            _mm512_storeu_ps(outptr, _sum0);
            _mm512_storeu_ps(outptr + 16, _sum1);
            _mm512_storeu_ps(outptr + 32, _sum2);
            _mm512_storeu_ps(outptr + 48, _sum3);
            outptr += 64;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

#if __AVX__
            __m256 _sum0 = _mm256_setzero_ps();
            __m256 _sum1 = _mm256_setzero_ps();
            __m256 _sum2 = _mm256_setzero_ps();
            __m256 _sum3 = _mm256_setzero_ps();
#else
            __m128 _sum0 = _mm_setzero_ps();
            __m128 _sum1 = _mm_setzero_ps();
            __m128 _sum2 = _mm_setzero_ps();
            __m128 _sum3 = _mm_setzero_ps();
            __m128 _sum4 = _mm_setzero_ps();
            __m128 _sum5 = _mm_setzero_ps();
            __m128 _sum6 = _mm_setzero_ps();
            __m128 _sum7 = _mm_setzero_ps();
#endif

            if (k != 0)
            {
#if __AVX__
                _sum0 = _mm256_load_ps(outptr);
                _sum1 = _mm256_load_ps(outptr + 8);
                _sum2 = _mm256_load_ps(outptr + 16);
                _sum3 = _mm256_load_ps(outptr + 24);
#else
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4);
                _sum2 = _mm_load_ps(outptr + 8);
                _sum3 = _mm_load_ps(outptr + 12);
                _sum4 = _mm_load_ps(outptr + 16);
                _sum5 = _mm_load_ps(outptr + 20);
                _sum6 = _mm_load_ps(outptr + 24);
                _sum7 = _mm_load_ps(outptr + 28);
#endif
            }

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA0 = _mm_loadu_si128((const __m128i*)pA);
                __m256i _pA00 = combine4x2_epi32(_pA0, _pA0);
                __m256i _pB01 = _mm256_loadu_si256((const __m256i*)pB);
                __m256i _pA11 = _mm256_alignr_epi8(_pA00, _pA00, 8);
                __m256i _pB23 = _mm256_alignr_epi8(_pB01, _pB01, 4);
                _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_pA00, (__m256bh)_pB01);
                _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_pA11, (__m256bh)_pB01);
                _sum2 = _mm256_dpbf16_ps(_sum2, (__m256bh)_pA00, (__m256bh)_pB23);
                _sum3 = _mm256_dpbf16_ps(_sum3, (__m256bh)_pA11, (__m256bh)_pB23);
                pA += 8;
                pB += 16;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
#if __AVX__
                __m128 _pA = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pA));
                __m256 _pA0 = combine4x2_ps(_pA, _pA);
                __m256 _pB0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pB));

                __m256 _pA1 = _mm256_permute_ps(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256 _pB1 = _mm256_permute_ps(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm256_comp_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_pA1, _pB0, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_pA0, _pB1, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_pA1, _pB1, _sum3);
#else  // __AVX__
                __m128 _pA0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pA));
                __m128 _pB0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pB));
                __m128 _pB1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pB + 4)));

                __m128 _pA1 = _mm_shuffle_ps(_pA0, _pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m128 _pB0s = _mm_shuffle_ps(_pB0, _pB0, _MM_SHUFFLE(0, 3, 2, 1));
                __m128 _pB1s = _mm_shuffle_ps(_pB1, _pB1, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA0, _pB1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_pA1, _pB0, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_pA1, _pB1, _sum3);
                _sum4 = _mm_comp_fmadd_ps(_pA0, _pB0s, _sum4);
                _sum5 = _mm_comp_fmadd_ps(_pA0, _pB1s, _sum5);
                _sum6 = _mm_comp_fmadd_ps(_pA1, _pB0s, _sum6);
                _sum7 = _mm_comp_fmadd_ps(_pA1, _pB1s, _sum7);
#endif // __AVX__

                pA += 4;
                pB += 8;
            }

#if __AVX__
            _mm256_store_ps(outptr, _sum0);
            _mm256_store_ps(outptr + 8, _sum1);
            _mm256_store_ps(outptr + 16, _sum2);
            _mm256_store_ps(outptr + 24, _sum3);
#else
            _mm_store_ps(outptr, _sum0);
            _mm_store_ps(outptr + 4, _sum1);
            _mm_store_ps(outptr + 8, _sum2);
            _mm_store_ps(outptr + 12, _sum3);
            _mm_store_ps(outptr + 16, _sum4);
            _mm_store_ps(outptr + 20, _sum5);
            _mm_store_ps(outptr + 24, _sum6);
            _mm_store_ps(outptr + 28, _sum7);
#endif
            outptr += 32;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA0 = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB0 = _mm_loadu_si128((const __m128i*)pB);
                __m128i _pA1 = _mm_alignr_epi8(_pA0, _pA0, 8);
                __m128i _pB1 = _mm_alignr_epi8(_pB0, _pB0, 4);
                _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_pA0, (__m128bh)_pB0);
                _sum1 = _mm_dpbf16_ps(_sum1, (__m128bh)_pA0, (__m128bh)_pB1);
                _sum2 = _mm_dpbf16_ps(_sum2, (__m128bh)_pA1, (__m128bh)_pB0);
                _sum3 = _mm_dpbf16_ps(_sum3, (__m128bh)_pA1, (__m128bh)_pB1);
                pA += 8;
                pB += 8;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m128 _pA0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pA));
                __m128 _pB0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pB));

                __m128 _pA1 = _mm_shuffle_ps(_pA0, _pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m128 _pB1 = _mm_shuffle_ps(_pB0, _pB0, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA0, _pB1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_pA1, _pB0, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_pA1, _pB1, _sum3);

                pA += 4;
                pB += 4;
            }

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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB0 = _mm_castpd_si128(_mm_load1_pd((const double*)pB));
                __m128i _pB1 = _mm_alignr_epi8(_pB0, _pB0, 4);
                _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_pA, (__m128bh)_pB0);
                _sum1 = _mm_dpbf16_ps(_sum1, (__m128bh)_pA, (__m128bh)_pB1);
                pA += 8;
                pB += 4;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m128 _pA = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pA));
                __m128 _pB0 = bfloat2float_sse(_mm_castps_si128(_mm_load1_ps((const float*)pB)));
                __m128 _pB1 = _mm_shuffle_ps(_pB0, _pB0, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm_comp_fmadd_ps(_pA, _pB0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA, _pB1, _sum1);

                pA += 4;
                pB += 2;
            }

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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_pA, (__m128bh)_mm_set1_epi32(((const int*)pB)[0]));
                pA += 8;
                pB += 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m128 _pA = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pA));
                __m128 _pB0 = _mm_set1_ps(bfloat16_to_float32(pB[0]));

                _sum0 = _mm_comp_fmadd_ps(_pA, _pB0, _sum0);

                pA += 4;
                pB += 1;
            }

            _mm_store_ps(outptr, _sum0);
            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _sum0 = _mm512_setzero_ps();
            __m512 _sum1 = _mm512_setzero_ps();

            if (k != 0)
            {
                _sum0 = _mm512_loadu_ps(outptr);
                _sum1 = _mm512_loadu_ps(outptr + 16);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512i _pA = _mm512_castpd_si512(_mm512_set1_pd(((const double*)pA)[0]));
                __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);
                __m512i _pB1 = _mm512_alignr_epi8(_pB0, _pB0, 4);
                _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_pA, (__m512bh)_pB0);
                _sum1 = _mm512_dpbf16_ps(_sum1, (__m512bh)_pA, (__m512bh)_pB1);
                pA += 4;
                pB += 32;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m512 _pA0 = bfloat2float_avx512(_mm256_set1_epi32(((const int*)pA)[0]));
                __m512 _pB0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pB));
                __m512 _pB1 = _mm512_permute_ps(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm512_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm512_fmadd_ps(_pA0, _pB1, _sum1);

                pA += 2;
                pB += 16;
            }

            _mm512_storeu_ps(outptr, _sum0);
            _mm512_storeu_ps(outptr + 16, _sum1);
            outptr += 32;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
#if __AVX__
            __m256 _sum0 = _mm256_setzero_ps();
            __m256 _sum1 = _mm256_setzero_ps();
#else
            __m128 _sum0 = _mm_setzero_ps();
            __m128 _sum1 = _mm_setzero_ps();
            __m128 _sum2 = _mm_setzero_ps();
            __m128 _sum3 = _mm_setzero_ps();
#endif

            if (k != 0)
            {
#if __AVX__
                _sum0 = _mm256_loadu_ps(outptr);
                _sum1 = _mm256_loadu_ps(outptr + 8);
#else
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4);
                _sum2 = _mm_load_ps(outptr + 8);
                _sum3 = _mm_load_ps(outptr + 12);
#endif
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA0 = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)pA));
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);
                __m256i _pA1 = _mm256_alignr_epi8(_pA0, _pA0, 4);
                _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_pA0, (__m256bh)_pB);
                _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_pA1, (__m256bh)_pB);
                pA += 4;
                pB += 16;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);
#if __AVX__
                __m256 _pA0 = bfloat2float_avx(_pA);
                __m256 _pA1 = _mm256_permute_ps(_pA0, _MM_SHUFFLE(0, 3, 2, 1));
                __m256 _pB0 = bfloat2float_avx(_pB);

                _sum0 = _mm256_comp_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_pA1, _pB0, _sum1);
#else  // __AVX__
                __m128 _pA0 = bfloat2float_sse(_pA);
                __m128 _pA1 = _mm_shuffle_ps(_pA0, _pA0, _MM_SHUFFLE(0, 3, 2, 1));
                __m128 _pB0 = bfloat2float_sse(_pB);
                __m128 _pB1 = bfloat2float_sse(_mm_srli_si128(_pB, 8));

                _sum0 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA0, _pB1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_pA1, _pB0, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_pA1, _pB1, _sum3);
#endif // __AVX__

                pA += 2;
                pB += 8;
            }

#if __AVX__
            _mm256_storeu_ps(outptr, _sum0);
            _mm256_storeu_ps(outptr + 8, _sum1);
#else
            _mm_store_ps(outptr, _sum0);
            _mm_store_ps(outptr + 4, _sum1);
            _mm_store_ps(outptr + 8, _sum2);
            _mm_store_ps(outptr + 12, _sum3);
#endif
            outptr += 16;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _sum0 = _mm_setzero_ps();
            __m128 _sum1 = _mm_setzero_ps();

            if (k != 0)
            {
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_castpd_si128(_mm_load1_pd((const double*)pA));
                __m128i _pB0 = _mm_loadu_si128((const __m128i*)pB);
                __m128i _pB1 = _mm_alignr_epi8(_pB0, _pB0, 4);
                _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_pA, (__m128bh)_pB0);
                _sum1 = _mm_dpbf16_ps(_sum1, (__m128bh)_pA, (__m128bh)_pB1);
                pA += 4;
                pB += 8;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m128 _pA = bfloat2float_sse(_mm_castps_si128(_mm_load1_ps((const float*)pA)));
                __m128 _pB0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pB));
                __m128 _pB1 = _mm_shuffle_ps(_pB0, _pB0, _MM_SHUFFLE(0, 3, 2, 1));
                _sum0 = _mm_comp_fmadd_ps(_pA, _pB0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA, _pB1, _sum1);
                pA += 2;
                pB += 4;
            }

            _mm_store_ps(outptr, _sum0);
            _mm_store_ps(outptr + 4, _sum1);
            outptr += 8;
        }
#endif // __SSE2__
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
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                float a00 = bfloat16_to_float32(pA[0]);
                float a01 = bfloat16_to_float32(pA[1]);
                float a10 = bfloat16_to_float32(pA[2]);
                float a11 = bfloat16_to_float32(pA[3]);
                float b00 = bfloat16_to_float32(pB[0]);
                float b01 = bfloat16_to_float32(pB[1]);
                float b10 = bfloat16_to_float32(pB[2]);
                float b11 = bfloat16_to_float32(pB[3]);
                sum00 += a00 * b00 + a01 * b01;
                sum01 += a00 * b10 + a01 * b11;
                sum10 += a10 * b00 + a11 * b01;
                sum11 += a10 * b10 + a11 * b11;
                pA += 4;
                pB += 4;
            }
#endif // __AVX512BF16__
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

            outptr[0] = sum00;
            outptr[1] = sum01;
            outptr[2] = sum10;
            outptr[3] = sum11;
            outptr += 4;
        }
        for (; jj < max_jj; jj++)
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
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                float a00 = bfloat16_to_float32(pA[0]);
                float a01 = bfloat16_to_float32(pA[1]);
                float a10 = bfloat16_to_float32(pA[2]);
                float a11 = bfloat16_to_float32(pA[3]);
                float b0 = bfloat16_to_float32(pB[0]);
                float b1 = bfloat16_to_float32(pB[1]);
                sum0 += a00 * b0 + a01 * b1;
                sum1 += a10 * b0 + a11 * b1;
                pA += 4;
                pB += 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                sum0 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);
                sum1 += bfloat16_to_float32(pA[1]) * bfloat16_to_float32(pB[0]);
                pA += 2;
                pB += 1;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
        }

        pAT += max_kk * 2;
    }
    for (; ii < max_ii; ii++)
    {
        const unsigned short* pB = pBT;

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _sum0 = _mm512_setzero_ps();

            if (k != 0)
            {
                _sum0 = _mm512_loadu_ps(outptr);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512i _pA = _mm512_set1_epi32(((const int*)pA)[0]);
                __m512i _pB = _mm512_loadu_si512((const __m512i*)pB);
                _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_pA, (__m512bh)_pB);
                pA += 2;
                pB += 32;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m512 _pA0 = _mm512_set1_ps(bfloat16_to_float32(pA[0]));
                __m512 _pB0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pB));

                _sum0 = _mm512_fmadd_ps(_pA0, _pB0, _sum0);

                pA += 1;
                pB += 16;
            }

            _mm512_storeu_ps(outptr, _sum0);
            outptr += 16;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
#if __AVX__
            __m256 _sum0 = _mm256_setzero_ps();
#else
            __m128 _sum0 = _mm_setzero_ps();
            __m128 _sum1 = _mm_setzero_ps();
#endif

            if (k != 0)
            {
#if __AVX__
                _sum0 = _mm256_loadu_ps(outptr);
#else
                _sum0 = _mm_loadu_ps(outptr);
                _sum1 = _mm_loadu_ps(outptr + 4);
#endif
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __AVX512BF16__
#if _MSC_VER
            __m256 _sum1 = _mm256_setzero_ps();
            __m256i _mask = _mm256_set1_epi32(0xffff0000);
#endif
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_set1_epi32(((const int*)pA)[0]);
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);
#if _MSC_VER
                // msvc crash here  --- nihui
                __m256 _pA0 = _mm256_castsi256_ps(_mm256_slli_epi32(_pA, 16));
                __m256 _pB0 = _mm256_castsi256_ps(_mm256_slli_epi32(_pB, 16));
                __m256 _pA1 = _mm256_castsi256_ps(_mm256_and_si256(_pA, _mask));
                __m256 _pB1 = _mm256_castsi256_ps(_mm256_and_si256(_pB, _mask));
                _sum0 = _mm256_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm256_fmadd_ps(_pA1, _pB1, _sum1);
#else
                _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_pA, (__m256bh)_pB);
#endif
                pA += 2;
                pB += 16;
            }
#if _MSC_VER
            _sum0 = _mm256_add_ps(_sum0, _sum1);
#endif
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
#if __AVX__
                __m256 _pA0 = _mm256_set1_ps(bfloat16_to_float32(pA[0]));
                __m256 _pB0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pB));

                _sum0 = _mm256_comp_fmadd_ps(_pA0, _pB0, _sum0);
#else
                __m128 _pA = _mm_set1_ps(bfloat16_to_float32(pA[0]));
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);
                __m128i _zero = _mm_setzero_si128();
                __m128 _pB0 = _mm_castsi128_ps(_mm_unpacklo_epi16(_zero, _pB));
                __m128 _pB1 = _mm_castsi128_ps(_mm_unpackhi_epi16(_zero, _pB));

                _sum0 = _mm_comp_fmadd_ps(_pA, _pB0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA, _pB1, _sum1);
#endif

                pA += 1;
                pB += 8;
            }

#if __AVX__
            _mm256_storeu_ps(outptr, _sum0);
#else
            _mm_storeu_ps(outptr, _sum0);
            _mm_storeu_ps(outptr + 4, _sum1);
#endif
            outptr += 8;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _sum0 = _mm_setzero_ps();

            if (k != 0)
            {
                _sum0 = _mm_loadu_ps(outptr);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __AVX512BF16__
#if _MSC_VER
            __m128 _sum1 = _mm_setzero_ps();
            __m128i _mask = _mm_set1_epi32(0xffff0000);
#endif
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_set1_epi32(((const int*)pA)[0]);
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);
#if _MSC_VER
                // msvc crash here  --- nihui
                __m128 _pA0 = _mm_castsi128_ps(_mm_slli_epi32(_pA, 16));
                __m128 _pB0 = _mm_castsi128_ps(_mm_slli_epi32(_pB, 16));
                __m128 _pA1 = _mm_castsi128_ps(_mm_and_si128(_pA, _mask));
                __m128 _pB1 = _mm_castsi128_ps(_mm_and_si128(_pB, _mask));
                _sum0 = _mm_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm_fmadd_ps(_pA1, _pB1, _sum1);
#else
                _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_pA, (__m128bh)_pB);
#endif
                pA += 2;
                pB += 8;
            }
#if _MSC_VER
            _sum0 = _mm_add_ps(_sum0, _sum1);
#endif
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m128 _pA0 = _mm_set1_ps(bfloat16_to_float32(pA[0]));
                __m128 _pB0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pB));

                _sum0 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum0);

                pA += 1;
                pB += 4;
            }

            _mm_storeu_ps(outptr, _sum0);
            outptr += 4;
        }
#endif // __SSE2__
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
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                float a1 = bfloat16_to_float32(pA[1]);
                float b00 = bfloat16_to_float32(pB[0]);
                float b01 = bfloat16_to_float32(pB[1]);
                float b10 = bfloat16_to_float32(pB[2]);
                float b11 = bfloat16_to_float32(pB[3]);
                sum0 += a0 * b00 + a1 * b01;
                sum1 += a0 * b10 + a1 * b11;
                pA += 2;
                pB += 4;
            }
#endif // __AVX512BF16__
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                float a1 = bfloat16_to_float32(pA[1]);
                float b00 = bfloat16_to_float32(pB[0]);
                float b01 = bfloat16_to_float32(pB[1]);
                sum += a0 * b00 + a1 * b01;
                pA += 2;
                pB += 2;
            }
#endif // __AVX512BF16__
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

static void unpack_output_tile_fp32_to_bf16(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta, int output_transpose)
{
    // NCNN_LOGE("unpack_output_tile_fp32_to_bf16 %d %d %d %d", i, max_ii, j, max_jj);
    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const int c_elempack = C.elempack;
    const float* pC = C;

    const float* pp = topT;

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

        __m512 _c0 = _mm512_set1_ps(0.f);
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = _mm512_set1_ps(pC[0] * beta);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = _mm512_loadu_ps(pC);
                _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
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

            if (pC)
            {
                if (broadcast_type_C == 0)
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
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
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
                if (broadcast_type_C == 3)
                {
                    __m512 _c1;
                    __m512 _c2;
                    __m512 _c3;
                    __m512 _c4;
                    __m512 _c5;
                    __m512 _c6;
                    __m512 _c7;
                    __m512 _c8;
                    __m512 _c9;
                    __m512 _ca;
                    __m512 _cb;
                    __m512 _cc;
                    __m512 _cd;
                    __m512 _ce;
                    __m512 _cf;
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
                    else if (c_elempack == 8)
                    {
                        __m512 _tmp0 = _mm512_loadu_ps(pC);
                        __m512 _tmp1 = _mm512_loadu_ps(pC + 16);
                        __m512 _tmp2 = _mm512_loadu_ps(pC + 32);
                        __m512 _tmp3 = _mm512_loadu_ps(pC + 48);
                        __m512 _tmp4 = _mm512_loadu_ps(pC + 64);
                        __m512 _tmp5 = _mm512_loadu_ps(pC + 80);
                        __m512 _tmp6 = _mm512_loadu_ps(pC + 96);
                        __m512 _tmp7 = _mm512_loadu_ps(pC + 112);
                        __m512 _tmp8 = _mm512_loadu_ps(pC + c_hstep * 8);
                        __m512 _tmp9 = _mm512_loadu_ps(pC + c_hstep * 8 + 16);
                        __m512 _tmpa = _mm512_loadu_ps(pC + c_hstep * 8 + 32);
                        __m512 _tmpb = _mm512_loadu_ps(pC + c_hstep * 8 + 48);
                        __m512 _tmpc = _mm512_loadu_ps(pC + c_hstep * 8 + 64);
                        __m512 _tmpd = _mm512_loadu_ps(pC + c_hstep * 8 + 80);
                        __m512 _tmpe = _mm512_loadu_ps(pC + c_hstep * 8 + 96);
                        __m512 _tmpf = _mm512_loadu_ps(pC + c_hstep * 8 + 112);

                        _c0 = _mm512_shuffle_f32x4(_tmp0, _tmp8, _MM_SHUFFLE(1, 0, 1, 0));
                        _c1 = _mm512_shuffle_f32x4(_tmp0, _tmp8, _MM_SHUFFLE(3, 2, 3, 2));
                        _c2 = _mm512_shuffle_f32x4(_tmp1, _tmp9, _MM_SHUFFLE(1, 0, 1, 0));
                        _c3 = _mm512_shuffle_f32x4(_tmp1, _tmp9, _MM_SHUFFLE(3, 2, 3, 2));
                        _c4 = _mm512_shuffle_f32x4(_tmp2, _tmpa, _MM_SHUFFLE(1, 0, 1, 0));
                        _c5 = _mm512_shuffle_f32x4(_tmp2, _tmpa, _MM_SHUFFLE(3, 2, 3, 2));
                        _c6 = _mm512_shuffle_f32x4(_tmp3, _tmpb, _MM_SHUFFLE(1, 0, 1, 0));
                        _c7 = _mm512_shuffle_f32x4(_tmp3, _tmpb, _MM_SHUFFLE(3, 2, 3, 2));
                        _c8 = _mm512_shuffle_f32x4(_tmp4, _tmpc, _MM_SHUFFLE(1, 0, 1, 0));
                        _c9 = _mm512_shuffle_f32x4(_tmp4, _tmpc, _MM_SHUFFLE(3, 2, 3, 2));
                        _ca = _mm512_shuffle_f32x4(_tmp5, _tmpd, _MM_SHUFFLE(1, 0, 1, 0));
                        _cb = _mm512_shuffle_f32x4(_tmp5, _tmpd, _MM_SHUFFLE(3, 2, 3, 2));
                        _cc = _mm512_shuffle_f32x4(_tmp6, _tmpe, _MM_SHUFFLE(1, 0, 1, 0));
                        _cd = _mm512_shuffle_f32x4(_tmp6, _tmpe, _MM_SHUFFLE(3, 2, 3, 2));
                        _ce = _mm512_shuffle_f32x4(_tmp7, _tmpf, _MM_SHUFFLE(1, 0, 1, 0));
                        _cf = _mm512_shuffle_f32x4(_tmp7, _tmpf, _MM_SHUFFLE(3, 2, 3, 2));

                        pC += 128;
                    }
                    else if (c_elempack == 4)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + 16);
                        _c2 = _mm512_loadu_ps(pC + 32);
                        _c3 = _mm512_loadu_ps(pC + 48);
                        _c4 = _mm512_loadu_ps(pC + c_hstep * 4);
                        _c5 = _mm512_loadu_ps(pC + c_hstep * 4 + 16);
                        _c6 = _mm512_loadu_ps(pC + c_hstep * 4 + 32);
                        _c7 = _mm512_loadu_ps(pC + c_hstep * 4 + 48);
                        _c8 = _mm512_loadu_ps(pC + c_hstep * 8);
                        _c9 = _mm512_loadu_ps(pC + c_hstep * 8 + 16);
                        _ca = _mm512_loadu_ps(pC + c_hstep * 8 + 32);
                        _cb = _mm512_loadu_ps(pC + c_hstep * 8 + 48);
                        _cc = _mm512_loadu_ps(pC + c_hstep * 12);
                        _cd = _mm512_loadu_ps(pC + c_hstep * 12 + 16);
                        _ce = _mm512_loadu_ps(pC + c_hstep * 12 + 32);
                        _cf = _mm512_loadu_ps(pC + c_hstep * 12 + 48);

                        __m512 _tmp0 = _mm512_shuffle_f32x4(_c0, _c4, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_c8, _cc, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp2 = _mm512_shuffle_f32x4(_c0, _c4, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp3 = _mm512_shuffle_f32x4(_c8, _cc, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp4 = _mm512_shuffle_f32x4(_c1, _c5, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp5 = _mm512_shuffle_f32x4(_c9, _cd, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp6 = _mm512_shuffle_f32x4(_c1, _c5, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp7 = _mm512_shuffle_f32x4(_c9, _cd, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp8 = _mm512_shuffle_f32x4(_c2, _c6, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp9 = _mm512_shuffle_f32x4(_ca, _ce, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmpa = _mm512_shuffle_f32x4(_c2, _c6, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmpb = _mm512_shuffle_f32x4(_ca, _ce, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmpc = _mm512_shuffle_f32x4(_c3, _c7, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmpd = _mm512_shuffle_f32x4(_cb, _cf, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmpe = _mm512_shuffle_f32x4(_c3, _c7, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmpf = _mm512_shuffle_f32x4(_cb, _cf, _MM_SHUFFLE(3, 2, 3, 2));

                        _c0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _c1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        _c2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                        _c3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                        _c4 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                        _c5 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                        _c6 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                        _c7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));
                        _c8 = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(2, 0, 2, 0));
                        _c9 = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(3, 1, 3, 1));
                        _ca = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(2, 0, 2, 0));
                        _cb = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(3, 1, 3, 1));
                        _cc = _mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(2, 0, 2, 0));
                        _cd = _mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(3, 1, 3, 1));
                        _ce = _mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(2, 0, 2, 0));
                        _cf = _mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(3, 1, 3, 1));

                        pC += 64;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + c_hstep);
                        _c2 = _mm512_loadu_ps(pC + c_hstep * 2);
                        _c3 = _mm512_loadu_ps(pC + c_hstep * 3);
                        _c4 = _mm512_loadu_ps(pC + c_hstep * 4);
                        _c5 = _mm512_loadu_ps(pC + c_hstep * 5);
                        _c6 = _mm512_loadu_ps(pC + c_hstep * 6);
                        _c7 = _mm512_loadu_ps(pC + c_hstep * 7);
                        _c8 = _mm512_loadu_ps(pC + c_hstep * 8);
                        _c9 = _mm512_loadu_ps(pC + c_hstep * 9);
                        _ca = _mm512_loadu_ps(pC + c_hstep * 10);
                        _cb = _mm512_loadu_ps(pC + c_hstep * 11);
                        _cc = _mm512_loadu_ps(pC + c_hstep * 12);
                        _cd = _mm512_loadu_ps(pC + c_hstep * 13);
                        _ce = _mm512_loadu_ps(pC + c_hstep * 14);
                        _cf = _mm512_loadu_ps(pC + c_hstep * 15);
                        transpose16x16_ps(_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _ca, _cb, _cc, _cd, _ce, _cf);
                        pC += 16;
                    }
                    if (beta == 1.f)
                    {
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
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm512_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm512_fmadd_ps(_c3, _beta, _f3);
                        _f4 = _mm512_fmadd_ps(_c4, _beta, _f4);
                        _f5 = _mm512_fmadd_ps(_c5, _beta, _f5);
                        _f6 = _mm512_fmadd_ps(_c6, _beta, _f6);
                        _f7 = _mm512_fmadd_ps(_c7, _beta, _f7);
                        _f8 = _mm512_fmadd_ps(_c8, _beta, _f8);
                        _f9 = _mm512_fmadd_ps(_c9, _beta, _f9);
                        _fa = _mm512_fmadd_ps(_ca, _beta, _fa);
                        _fb = _mm512_fmadd_ps(_cb, _beta, _fb);
                        _fc = _mm512_fmadd_ps(_cc, _beta, _fc);
                        _fd = _mm512_fmadd_ps(_cd, _beta, _fd);
                        _fe = _mm512_fmadd_ps(_ce, _beta, _fe);
                        _ff = _mm512_fmadd_ps(_cf, _beta, _ff);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm512_set1_ps(pC[0] * beta);
                    __m512 _c1 = _mm512_set1_ps(pC[1] * beta);
                    __m512 _c2 = _mm512_set1_ps(pC[2] * beta);
                    __m512 _c3 = _mm512_set1_ps(pC[3] * beta);

                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c1);
                    _f2 = _mm512_add_ps(_f2, _c2);
                    _f3 = _mm512_add_ps(_f3, _c3);

                    _c0 = _mm512_set1_ps(pC[4] * beta);
                    _c1 = _mm512_set1_ps(pC[5] * beta);
                    _c2 = _mm512_set1_ps(pC[6] * beta);
                    _c3 = _mm512_set1_ps(pC[7] * beta);

                    _f4 = _mm512_add_ps(_f4, _c0);
                    _f5 = _mm512_add_ps(_f5, _c1);
                    _f6 = _mm512_add_ps(_f6, _c2);
                    _f7 = _mm512_add_ps(_f7, _c3);

                    _c0 = _mm512_set1_ps(pC[8] * beta);
                    _c1 = _mm512_set1_ps(pC[9] * beta);
                    _c2 = _mm512_set1_ps(pC[10] * beta);
                    _c3 = _mm512_set1_ps(pC[11] * beta);

                    _f8 = _mm512_add_ps(_f8, _c0);
                    _f9 = _mm512_add_ps(_f9, _c1);
                    _fa = _mm512_add_ps(_fa, _c2);
                    _fb = _mm512_add_ps(_fb, _c3);

                    _c0 = _mm512_set1_ps(pC[12] * beta);
                    _c1 = _mm512_set1_ps(pC[13] * beta);
                    _c2 = _mm512_set1_ps(pC[14] * beta);
                    _c3 = _mm512_set1_ps(pC[15] * beta);

                    _fc = _mm512_add_ps(_fc, _c0);
                    _fd = _mm512_add_ps(_fd, _c1);
                    _fe = _mm512_add_ps(_fe, _c2);
                    _ff = _mm512_add_ps(_ff, _c3);
                    pC += 16;
                }
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

            __m256i _bf0 = float2bfloat_avx512(_f0);
            __m256i _bf1 = float2bfloat_avx512(_f1);
            __m256i _bf2 = float2bfloat_avx512(_f2);
            __m256i _bf3 = float2bfloat_avx512(_f3);
            __m256i _bf4 = float2bfloat_avx512(_f4);
            __m256i _bf5 = float2bfloat_avx512(_f5);
            __m256i _bf6 = float2bfloat_avx512(_f6);
            __m256i _bf7 = float2bfloat_avx512(_f7);
            __m256i _bf8 = float2bfloat_avx512(_f8);
            __m256i _bf9 = float2bfloat_avx512(_f9);
            __m256i _bfa = float2bfloat_avx512(_fa);
            __m256i _bfb = float2bfloat_avx512(_fb);
            __m256i _bfc = float2bfloat_avx512(_fc);
            __m256i _bfd = float2bfloat_avx512(_fd);
            __m256i _bfe = float2bfloat_avx512(_fe);
            __m256i _bff = float2bfloat_avx512(_ff);

            // store bf16
            if (output_transpose)
            {
                if (out_elempack == 16)
                {
                    transpose16x16_epi16(_bf0, _bf1, _bf2, _bf3, _bf4, _bf5, _bf6, _bf7, _bf8, _bf9, _bfa, _bfb, _bfc, _bfd, _bfe, _bff);

                    _mm256_storeu_si256((__m256i*)p0, _bf0);
                    _mm256_storeu_si256((__m256i*)(p0 + 16), _bf1);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 2), _bf2);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 3), _bf3);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 4), _bf4);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 5), _bf5);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 6), _bf6);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 7), _bf7);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 8), _bf8);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 9), _bf9);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 10), _bfa);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 11), _bfb);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 12), _bfc);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 13), _bfd);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 14), _bfe);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 15), _bff);
                }
                if (out_elempack == 8)
                {
                    transpose16x8_epi16(_bf0, _bf1, _bf2, _bf3, _bf4, _bf5, _bf6, _bf7);
                    transpose16x8_epi16(_bf8, _bf9, _bfa, _bfb, _bfc, _bfd, _bfe, _bff);

                    // after transpose: r[k].low128=[jj0..7 for ii_{2k}], r[k].high128=[jj0..7 for ii_{2k+1}]
                    _mm_storeu_si128((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 2), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 3), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 4), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 5), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 6), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 7), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 8), _mm256_extractf128_si256(_bf4, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 9), _mm256_extractf128_si256(_bf4, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 10), _mm256_extractf128_si256(_bf5, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 11), _mm256_extractf128_si256(_bf5, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 12), _mm256_extractf128_si256(_bf6, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 13), _mm256_extractf128_si256(_bf6, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 14), _mm256_extractf128_si256(_bf7, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 15), _mm256_extractf128_si256(_bf7, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf8, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8), _mm256_extractf128_si256(_bf8, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 2), _mm256_extractf128_si256(_bf9, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 3), _mm256_extractf128_si256(_bf9, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 4), _mm256_extractf128_si256(_bfa, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 5), _mm256_extractf128_si256(_bfa, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 6), _mm256_extractf128_si256(_bfb, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 7), _mm256_extractf128_si256(_bfb, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 8), _mm256_extractf128_si256(_bfc, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 9), _mm256_extractf128_si256(_bfc, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 10), _mm256_extractf128_si256(_bfd, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 11), _mm256_extractf128_si256(_bfd, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 12), _mm256_extractf128_si256(_bfe, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 13), _mm256_extractf128_si256(_bfe, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 14), _mm256_extractf128_si256(_bff, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 15), _mm256_extractf128_si256(_bff, 1));
                }
                if (out_elempack == 4)
                {
                    transpose16x4_epi16(_bf0, _bf1, _bf2, _bf3);
                    transpose16x4_epi16(_bf4, _bf5, _bf6, _bf7);
                    transpose16x4_epi16(_bf8, _bf9, _bfa, _bfb);
                    transpose16x4_epi16(_bfc, _bfd, _bfe, _bff);

                    // after transpose16x4: _bf0 = [jj0-3 for ii0 | jj0-3 for ii1 | jj0-3 for ii2 | jj0-3 for ii3]
                    //                      _bf1 = [jj0-3 for ii4..ii7], _bf2 = [jj0-3 for ii8..ii11], _bf3 = [jj0-3 for ii12..ii15]
                    //                      _bf4 = [jj4-7 for ii0..ii3], _bf5 = [jj4-7 for ii4..ii7], etc.
                    // jj0-3 row
                    _mm_storel_epi64((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storeh_pd((double*)(p0 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storeh_pd((double*)(p0 + 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + 16), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storeh_pd((double*)(p0 + 20), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + 24), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storeh_pd((double*)(p0 + 28), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + 32), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storeh_pd((double*)(p0 + 36), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + 40), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storeh_pd((double*)(p0 + 44), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + 48), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storeh_pd((double*)(p0 + 52), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + 56), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storeh_pd((double*)(p0 + 60), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 1)));
                    // jj4-7 row
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4), _mm256_extractf128_si256(_bf4, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf4, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 8), _mm256_extractf128_si256(_bf4, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf4, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 16), _mm256_extractf128_si256(_bf5, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 20), _mm_castsi128_pd(_mm256_extractf128_si256(_bf5, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 24), _mm256_extractf128_si256(_bf5, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 28), _mm_castsi128_pd(_mm256_extractf128_si256(_bf5, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 32), _mm256_extractf128_si256(_bf6, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 36), _mm_castsi128_pd(_mm256_extractf128_si256(_bf6, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 40), _mm256_extractf128_si256(_bf6, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 44), _mm_castsi128_pd(_mm256_extractf128_si256(_bf6, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 48), _mm256_extractf128_si256(_bf7, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 52), _mm_castsi128_pd(_mm256_extractf128_si256(_bf7, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 56), _mm256_extractf128_si256(_bf7, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 60), _mm_castsi128_pd(_mm256_extractf128_si256(_bf7, 1)));
                    // jj8-11 row
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf8, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 8 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf8, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 8), _mm256_extractf128_si256(_bf8, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 8 + 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf8, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 16), _mm256_extractf128_si256(_bf9, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 8 + 20), _mm_castsi128_pd(_mm256_extractf128_si256(_bf9, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 24), _mm256_extractf128_si256(_bf9, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 8 + 28), _mm_castsi128_pd(_mm256_extractf128_si256(_bf9, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 32), _mm256_extractf128_si256(_bfa, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 8 + 36), _mm_castsi128_pd(_mm256_extractf128_si256(_bfa, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 40), _mm256_extractf128_si256(_bfa, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 8 + 44), _mm_castsi128_pd(_mm256_extractf128_si256(_bfa, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 48), _mm256_extractf128_si256(_bfb, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 8 + 52), _mm_castsi128_pd(_mm256_extractf128_si256(_bfb, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 56), _mm256_extractf128_si256(_bfb, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 8 + 60), _mm_castsi128_pd(_mm256_extractf128_si256(_bfb, 1)));
                    // jj12-15 row
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 12), _mm256_extractf128_si256(_bfc, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bfc, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 12 + 8), _mm256_extractf128_si256(_bfc, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bfc, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 12 + 16), _mm256_extractf128_si256(_bfd, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 20), _mm_castsi128_pd(_mm256_extractf128_si256(_bfd, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 12 + 24), _mm256_extractf128_si256(_bfd, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 28), _mm_castsi128_pd(_mm256_extractf128_si256(_bfd, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 12 + 32), _mm256_extractf128_si256(_bfe, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 36), _mm_castsi128_pd(_mm256_extractf128_si256(_bfe, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 12 + 40), _mm256_extractf128_si256(_bfe, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 44), _mm_castsi128_pd(_mm256_extractf128_si256(_bfe, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 12 + 48), _mm256_extractf128_si256(_bff, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 52), _mm_castsi128_pd(_mm256_extractf128_si256(_bff, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 12 + 56), _mm256_extractf128_si256(_bff, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 60), _mm_castsi128_pd(_mm256_extractf128_si256(_bff, 1)));
                }
                if (out_elempack == 1)
                {
                    _mm256_storeu_si256((__m256i*)p0, _bf0);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep), _bf1);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 2), _bf2);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 3), _bf3);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 4), _bf4);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 5), _bf5);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 6), _bf6);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 7), _bf7);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 8), _bf8);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 9), _bf9);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 10), _bfa);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 11), _bfb);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 12), _bfc);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 13), _bfd);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 14), _bfe);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 15), _bff);
                }
                p0 += out_hstep * 16;
            }
            else
            {
                if (out_elempack == 16)
                {
                    _mm256_storeu_si256((__m256i*)p0, _bf0);
                    _mm256_storeu_si256((__m256i*)(p0 + 16), _bf1);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 2), _bf2);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 3), _bf3);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 4), _bf4);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 5), _bf5);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 6), _bf6);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 7), _bf7);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 8), _bf8);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 9), _bf9);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 10), _bfa);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 11), _bfb);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 12), _bfc);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 13), _bfd);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 14), _bfe);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 15), _bff);
                    p0 += 256;
                }
                if (out_elempack == 8)
                {
                    _mm_storeu_si128((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 2), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 3), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 4), _mm256_extractf128_si256(_bf4, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 5), _mm256_extractf128_si256(_bf5, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 6), _mm256_extractf128_si256(_bf6, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 7), _mm256_extractf128_si256(_bf7, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 8), _mm256_extractf128_si256(_bf8, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 9), _mm256_extractf128_si256(_bf9, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 10), _mm256_extractf128_si256(_bfa, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 11), _mm256_extractf128_si256(_bfb, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 12), _mm256_extractf128_si256(_bfc, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 13), _mm256_extractf128_si256(_bfd, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 14), _mm256_extractf128_si256(_bfe, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 15), _mm256_extractf128_si256(_bff, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 2), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 3), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 4), _mm256_extractf128_si256(_bf4, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 5), _mm256_extractf128_si256(_bf5, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 6), _mm256_extractf128_si256(_bf6, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 7), _mm256_extractf128_si256(_bf7, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 8), _mm256_extractf128_si256(_bf8, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 9), _mm256_extractf128_si256(_bf9, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 10), _mm256_extractf128_si256(_bfa, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 11), _mm256_extractf128_si256(_bfb, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 12), _mm256_extractf128_si256(_bfc, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 13), _mm256_extractf128_si256(_bfd, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 14), _mm256_extractf128_si256(_bfe, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 15), _mm256_extractf128_si256(_bff, 1));
                    p0 += 128;
                }
                if (out_elempack == 4)
                {
                    _mm_storel_epi64((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 2), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 3), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 4), _mm256_extractf128_si256(_bf4, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 5), _mm256_extractf128_si256(_bf5, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 6), _mm256_extractf128_si256(_bf6, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 7), _mm256_extractf128_si256(_bf7, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 8), _mm256_extractf128_si256(_bf8, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 9), _mm256_extractf128_si256(_bf9, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 10), _mm256_extractf128_si256(_bfa, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 11), _mm256_extractf128_si256(_bfb, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 12), _mm256_extractf128_si256(_bfc, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 13), _mm256_extractf128_si256(_bfd, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 14), _mm256_extractf128_si256(_bfe, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 15), _mm256_extractf128_si256(_bff, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 2), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 3), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf4, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 5), _mm_castsi128_pd(_mm256_extractf128_si256(_bf5, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 6), _mm_castsi128_pd(_mm256_extractf128_si256(_bf6, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 7), _mm_castsi128_pd(_mm256_extractf128_si256(_bf7, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 8), _mm_castsi128_pd(_mm256_extractf128_si256(_bf8, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 9), _mm_castsi128_pd(_mm256_extractf128_si256(_bf9, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 10), _mm_castsi128_pd(_mm256_extractf128_si256(_bfa, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 11), _mm_castsi128_pd(_mm256_extractf128_si256(_bfb, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bfc, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 13), _mm_castsi128_pd(_mm256_extractf128_si256(_bfd, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 14), _mm_castsi128_pd(_mm256_extractf128_si256(_bfe, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 15), _mm_castsi128_pd(_mm256_extractf128_si256(_bff, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 2), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 3), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 4), _mm256_extractf128_si256(_bf4, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 5), _mm256_extractf128_si256(_bf5, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 6), _mm256_extractf128_si256(_bf6, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 7), _mm256_extractf128_si256(_bf7, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 8), _mm256_extractf128_si256(_bf8, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 9), _mm256_extractf128_si256(_bf9, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 10), _mm256_extractf128_si256(_bfa, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 11), _mm256_extractf128_si256(_bfb, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 12), _mm256_extractf128_si256(_bfc, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 13), _mm256_extractf128_si256(_bfd, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 14), _mm256_extractf128_si256(_bfe, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 15), _mm256_extractf128_si256(_bff, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 2), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 3), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf4, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 5), _mm_castsi128_pd(_mm256_extractf128_si256(_bf5, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 6), _mm_castsi128_pd(_mm256_extractf128_si256(_bf6, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 7), _mm_castsi128_pd(_mm256_extractf128_si256(_bf7, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 8), _mm_castsi128_pd(_mm256_extractf128_si256(_bf8, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 9), _mm_castsi128_pd(_mm256_extractf128_si256(_bf9, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 10), _mm_castsi128_pd(_mm256_extractf128_si256(_bfa, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 11), _mm_castsi128_pd(_mm256_extractf128_si256(_bfb, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bfc, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 13), _mm_castsi128_pd(_mm256_extractf128_si256(_bfd, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 14), _mm_castsi128_pd(_mm256_extractf128_si256(_bfe, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 15), _mm_castsi128_pd(_mm256_extractf128_si256(_bff, 1)));
                    p0 += 64;
                }
                if (out_elempack == 1)
                {
                    transpose16x16_epi16(_bf0, _bf1, _bf2, _bf3, _bf4, _bf5, _bf6, _bf7, _bf8, _bf9, _bfa, _bfb, _bfc, _bfd, _bfe, _bff);

                    _mm256_storeu_si256((__m256i*)p0, _bf0);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep), _bf1);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 2), _bf2);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 3), _bf3);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 4), _bf4);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 5), _bf5);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 6), _bf6);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 7), _bf7);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 8), _bf8);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 9), _bf9);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 10), _bfa);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 11), _bfb);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 12), _bfc);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 13), _bfd);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 14), _bfe);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 15), _bff);
                    p0 += 16;
                }
            }
        }
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

            if (pC)
            {
                if (broadcast_type_C == 0)
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
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
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
                if (broadcast_type_C == 3)
                {
                    __m512 _c1;
                    __m512 _c2;
                    __m512 _c3;
                    __m512 _c4;
                    __m512 _c5;
                    __m512 _c6;
                    __m512 _c7;
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
                        pC += 128;
                    }
                    else if (c_elempack == 8)
                    {
                        __m512 _tmp0 = _mm512_loadu_ps(pC);
                        __m512 _tmp1 = _mm512_loadu_ps(pC + 16);
                        __m512 _tmp2 = _mm512_loadu_ps(pC + 32);
                        __m512 _tmp3 = _mm512_loadu_ps(pC + 48);
                        __m512 _tmp4 = _mm512_loadu_ps(pC + c_hstep * 8);
                        __m512 _tmp5 = _mm512_loadu_ps(pC + c_hstep * 8 + 16);
                        __m512 _tmp6 = _mm512_loadu_ps(pC + c_hstep * 8 + 32);
                        __m512 _tmp7 = _mm512_loadu_ps(pC + c_hstep * 8 + 48);

                        _c0 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(1, 0, 1, 0));
                        _c1 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(3, 2, 3, 2));
                        _c2 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(1, 0, 1, 0));
                        _c3 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(3, 2, 3, 2));
                        _c4 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
                        _c5 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
                        _c6 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
                        _c7 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));

                        pC += 64;
                    }
                    else if (c_elempack == 4)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + 16);
                        _c2 = _mm512_loadu_ps(pC + c_hstep * 4);
                        _c3 = _mm512_loadu_ps(pC + c_hstep * 4 + 16);
                        _c4 = _mm512_loadu_ps(pC + c_hstep * 8);
                        _c5 = _mm512_loadu_ps(pC + c_hstep * 8 + 16);
                        _c6 = _mm512_loadu_ps(pC + c_hstep * 12);
                        _c7 = _mm512_loadu_ps(pC + c_hstep * 12 + 16);

                        __m512 _tmp0 = _mm512_shuffle_f32x4(_c0, _c2, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_c4, _c6, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp2 = _mm512_shuffle_f32x4(_c0, _c2, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp3 = _mm512_shuffle_f32x4(_c4, _c6, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp4 = _mm512_shuffle_f32x4(_c1, _c3, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp5 = _mm512_shuffle_f32x4(_c5, _c7, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp6 = _mm512_shuffle_f32x4(_c1, _c3, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp7 = _mm512_shuffle_f32x4(_c5, _c7, _MM_SHUFFLE(3, 2, 3, 2));

                        _c0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _c1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        _c2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                        _c3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                        _c4 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                        _c5 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                        _c6 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                        _c7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                        pC += 32;
                    }
                    else // if (c_elempack == 1)
                    {
                        __m256 _cc0 = _mm256_loadu_ps(pC);
                        __m256 _cc1 = _mm256_loadu_ps(pC + c_hstep);
                        __m256 _cc2 = _mm256_loadu_ps(pC + c_hstep * 2);
                        __m256 _cc3 = _mm256_loadu_ps(pC + c_hstep * 3);
                        __m256 _cc4 = _mm256_loadu_ps(pC + c_hstep * 4);
                        __m256 _cc5 = _mm256_loadu_ps(pC + c_hstep * 5);
                        __m256 _cc6 = _mm256_loadu_ps(pC + c_hstep * 6);
                        __m256 _cc7 = _mm256_loadu_ps(pC + c_hstep * 7);
                        __m256 _cc8 = _mm256_loadu_ps(pC + c_hstep * 8);
                        __m256 _cc9 = _mm256_loadu_ps(pC + c_hstep * 9);
                        __m256 _cca = _mm256_loadu_ps(pC + c_hstep * 10);
                        __m256 _ccb = _mm256_loadu_ps(pC + c_hstep * 11);
                        __m256 _ccc = _mm256_loadu_ps(pC + c_hstep * 12);
                        __m256 _ccd = _mm256_loadu_ps(pC + c_hstep * 13);
                        __m256 _cce = _mm256_loadu_ps(pC + c_hstep * 14);
                        __m256 _ccf = _mm256_loadu_ps(pC + c_hstep * 15);
                        transpose8x8_ps(_cc0, _cc1, _cc2, _cc3, _cc4, _cc5, _cc6, _cc7);
                        transpose8x8_ps(_cc8, _cc9, _cca, _ccb, _ccc, _ccd, _cce, _ccf);
                        _c0 = combine8x2_ps(_cc0, _cc8);
                        _c1 = combine8x2_ps(_cc1, _cc9);
                        _c2 = combine8x2_ps(_cc2, _cca);
                        _c3 = combine8x2_ps(_cc3, _ccb);
                        _c4 = combine8x2_ps(_cc4, _ccc);
                        _c5 = combine8x2_ps(_cc5, _ccd);
                        _c6 = combine8x2_ps(_cc6, _cce);
                        _c7 = combine8x2_ps(_cc7, _ccf);
                        pC += 8;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0);
                        _f1 = _mm512_add_ps(_f1, _c1);
                        _f2 = _mm512_add_ps(_f2, _c2);
                        _f3 = _mm512_add_ps(_f3, _c3);
                        _f4 = _mm512_add_ps(_f4, _c4);
                        _f5 = _mm512_add_ps(_f5, _c5);
                        _f6 = _mm512_add_ps(_f6, _c6);
                        _f7 = _mm512_add_ps(_f7, _c7);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm512_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm512_fmadd_ps(_c3, _beta, _f3);
                        _f4 = _mm512_fmadd_ps(_c4, _beta, _f4);
                        _f5 = _mm512_fmadd_ps(_c5, _beta, _f5);
                        _f6 = _mm512_fmadd_ps(_c6, _beta, _f6);
                        _f7 = _mm512_fmadd_ps(_c7, _beta, _f7);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm512_set1_ps(pC[0] * beta);
                    __m512 _c1 = _mm512_set1_ps(pC[1] * beta);
                    __m512 _c2 = _mm512_set1_ps(pC[2] * beta);
                    __m512 _c3 = _mm512_set1_ps(pC[3] * beta);

                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c1);
                    _f2 = _mm512_add_ps(_f2, _c2);
                    _f3 = _mm512_add_ps(_f3, _c3);

                    _c0 = _mm512_set1_ps(pC[4] * beta);
                    _c1 = _mm512_set1_ps(pC[5] * beta);
                    _c2 = _mm512_set1_ps(pC[6] * beta);
                    _c3 = _mm512_set1_ps(pC[7] * beta);

                    _f4 = _mm512_add_ps(_f4, _c0);
                    _f5 = _mm512_add_ps(_f5, _c1);
                    _f6 = _mm512_add_ps(_f6, _c2);
                    _f7 = _mm512_add_ps(_f7, _c3);
                    pC += 8;
                }
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

            __m256i _bf0 = float2bfloat_avx512(_f0);
            __m256i _bf1 = float2bfloat_avx512(_f1);
            __m256i _bf2 = float2bfloat_avx512(_f2);
            __m256i _bf3 = float2bfloat_avx512(_f3);
            __m256i _bf4 = float2bfloat_avx512(_f4);
            __m256i _bf5 = float2bfloat_avx512(_f5);
            __m256i _bf6 = float2bfloat_avx512(_f6);
            __m256i _bf7 = float2bfloat_avx512(_f7);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    transpose16x8_epi16(_bf0, _bf1, _bf2, _bf3, _bf4, _bf5, _bf6, _bf7);

                    // after transpose: r[k].low128=[jj0..7 for ii_{2k}], r[k].high128=[jj0..7 for ii_{2k+1}]
                    _mm_storeu_si128((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 2), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 3), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 4), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 5), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 6), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 7), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 8), _mm256_extractf128_si256(_bf4, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 9), _mm256_extractf128_si256(_bf4, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 10), _mm256_extractf128_si256(_bf5, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 11), _mm256_extractf128_si256(_bf5, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 12), _mm256_extractf128_si256(_bf6, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 13), _mm256_extractf128_si256(_bf6, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 14), _mm256_extractf128_si256(_bf7, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 15), _mm256_extractf128_si256(_bf7, 1));
                }
                if (out_elempack == 4)
                {
                    transpose16x4_epi16(_bf0, _bf1, _bf2, _bf3);
                    transpose16x4_epi16(_bf4, _bf5, _bf6, _bf7);

                    // jj0-3 row: _bf0=ii0-3, _bf1=ii4-7, _bf2=ii8-11, _bf3=ii12-15
                    _mm_storel_epi64((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storeh_pd((double*)(p0 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 2), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storeh_pd((double*)(p0 + 4 * 3), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 4), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storeh_pd((double*)(p0 + 4 * 5), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 6), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storeh_pd((double*)(p0 + 4 * 7), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 8), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storeh_pd((double*)(p0 + 4 * 9), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 10), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storeh_pd((double*)(p0 + 4 * 11), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 12), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storeh_pd((double*)(p0 + 4 * 13), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 14), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storeh_pd((double*)(p0 + 4 * 15), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 1)));
                    // jj4-7 row: _bf4=ii0-3, _bf5=ii4-7, _bf6=ii8-11, _bf7=ii12-15
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4), _mm256_extractf128_si256(_bf4, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf4, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 4 * 2), _mm256_extractf128_si256(_bf4, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 3), _mm_castsi128_pd(_mm256_extractf128_si256(_bf4, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 4 * 4), _mm256_extractf128_si256(_bf5, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 5), _mm_castsi128_pd(_mm256_extractf128_si256(_bf5, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 4 * 6), _mm256_extractf128_si256(_bf5, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 7), _mm_castsi128_pd(_mm256_extractf128_si256(_bf5, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 4 * 8), _mm256_extractf128_si256(_bf6, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 9), _mm_castsi128_pd(_mm256_extractf128_si256(_bf6, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 4 * 10), _mm256_extractf128_si256(_bf6, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 11), _mm_castsi128_pd(_mm256_extractf128_si256(_bf6, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 4 * 12), _mm256_extractf128_si256(_bf7, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 13), _mm_castsi128_pd(_mm256_extractf128_si256(_bf7, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 4 * 14), _mm256_extractf128_si256(_bf7, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 15), _mm_castsi128_pd(_mm256_extractf128_si256(_bf7, 1)));
                }
                if (out_elempack == 1)
                {
                    _mm256_storeu_si256((__m256i*)p0, _bf0);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep), _bf1);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 2), _bf2);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 3), _bf3);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 4), _bf4);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 5), _bf5);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 6), _bf6);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 7), _bf7);
                }
                p0 += out_hstep * 8;
            }
            else
            {
                if (out_elempack == 16)
                {
                    _mm256_storeu_si256((__m256i*)p0, _bf0);
                    _mm256_storeu_si256((__m256i*)(p0 + 16), _bf1);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 2), _bf2);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 3), _bf3);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 4), _bf4);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 5), _bf5);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 6), _bf6);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 7), _bf7);
                    p0 += 128;
                }
                if (out_elempack == 8)
                {
                    _mm_storeu_si128((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 2), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 3), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 4), _mm256_extractf128_si256(_bf4, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 5), _mm256_extractf128_si256(_bf5, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 6), _mm256_extractf128_si256(_bf6, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 7), _mm256_extractf128_si256(_bf7, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 2), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 3), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 4), _mm256_extractf128_si256(_bf4, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 5), _mm256_extractf128_si256(_bf5, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 6), _mm256_extractf128_si256(_bf6, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 7), _mm256_extractf128_si256(_bf7, 1));
                    p0 += 64;
                }
                if (out_elempack == 4)
                {
                    _mm_storel_epi64((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 2), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 3), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 4), _mm256_extractf128_si256(_bf4, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 5), _mm256_extractf128_si256(_bf5, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 6), _mm256_extractf128_si256(_bf6, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 7), _mm256_extractf128_si256(_bf7, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 2), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 3), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf4, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 5), _mm_castsi128_pd(_mm256_extractf128_si256(_bf5, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 6), _mm_castsi128_pd(_mm256_extractf128_si256(_bf6, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 7), _mm_castsi128_pd(_mm256_extractf128_si256(_bf7, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 2), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 3), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 4), _mm256_extractf128_si256(_bf4, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 5), _mm256_extractf128_si256(_bf5, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 6), _mm256_extractf128_si256(_bf6, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 7), _mm256_extractf128_si256(_bf7, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 2), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 3), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf4, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 5), _mm_castsi128_pd(_mm256_extractf128_si256(_bf5, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 6), _mm_castsi128_pd(_mm256_extractf128_si256(_bf6, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 7), _mm_castsi128_pd(_mm256_extractf128_si256(_bf7, 1)));
                    p0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose16x8_epi16(_bf0, _bf1, _bf2, _bf3, _bf4, _bf5, _bf6, _bf7);
                    _mm_storeu_si128((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 2), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 3), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 4), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 5), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 6), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 7), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf4, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 9), _mm256_extractf128_si256(_bf4, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 10), _mm256_extractf128_si256(_bf5, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 11), _mm256_extractf128_si256(_bf5, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 12), _mm256_extractf128_si256(_bf6, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 13), _mm256_extractf128_si256(_bf6, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 14), _mm256_extractf128_si256(_bf7, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 15), _mm256_extractf128_si256(_bf7, 1));
                    p0 += 8;
                }
            }
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m512 _f0 = _mm512_load_ps(pp);
            __m512 _f1 = _mm512_load_ps(pp + 16);
            __m512 _f2 = _mm512_load_ps(pp + 32);
            __m512 _f3 = _mm512_load_ps(pp + 48);
            pp += 64;

            // deshuffle from the shuffle-based 16x4 dpbf16_ps kernel
            {
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f3);
                __m512 _tmp1 = _mm512_unpackhi_ps(_f0, _f3);
                __m512 _tmp2 = _mm512_unpacklo_ps(_f2, _f1);
                __m512 _tmp3 = _mm512_unpackhi_ps(_f2, _f1);
                _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _f2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
            }

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                    _f2 = _mm512_add_ps(_f2, _c0);
                    _f3 = _mm512_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                    _f2 = _mm512_add_ps(_f2, _c0);
                    _f3 = _mm512_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c1;
                    __m512 _c2;
                    __m512 _c3;
                    if (c_elempack == 16)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + 16);
                        _c2 = _mm512_loadu_ps(pC + 32);
                        _c3 = _mm512_loadu_ps(pC + 48);
                        pC += 64;
                    }
                    else if (c_elempack == 8)
                    {
                        __m512 _cc0 = _mm512_loadu_ps(pC);
                        __m512 _cc1 = _mm512_loadu_ps(pC + 16);
                        __m512 _cc2 = _mm512_loadu_ps(pC + c_hstep * 8);
                        __m512 _cc3 = _mm512_loadu_ps(pC + c_hstep * 8 + 16);
                        _c0 = _mm512_shuffle_f32x4(_cc0, _cc2, _MM_SHUFFLE(1, 0, 1, 0));
                        _c1 = _mm512_shuffle_f32x4(_cc0, _cc2, _MM_SHUFFLE(3, 2, 3, 2));
                        _c2 = _mm512_shuffle_f32x4(_cc1, _cc3, _MM_SHUFFLE(1, 0, 1, 0));
                        _c3 = _mm512_shuffle_f32x4(_cc1, _cc3, _MM_SHUFFLE(3, 2, 3, 2));
                        pC += 32;
                    }
                    else if (c_elempack == 4)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + c_hstep * 4);
                        _c2 = _mm512_loadu_ps(pC + c_hstep * 8);
                        _c3 = _mm512_loadu_ps(pC + c_hstep * 12);
                        __m512 _tmp0 = _mm512_shuffle_f32x4(_c0, _c1, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_c2, _c3, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp2 = _mm512_shuffle_f32x4(_c0, _c1, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp3 = _mm512_shuffle_f32x4(_c2, _c3, _MM_SHUFFLE(3, 2, 3, 2));
                        _c0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _c1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        _c2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                        _c3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                        pC += 16;
                    }
                    else // if (c_elempack == 1)
                    {
                        __m128 _cc0 = _mm_loadu_ps(pC);
                        __m128 _cc1 = _mm_loadu_ps(pC + c_hstep);
                        __m128 _cc2 = _mm_loadu_ps(pC + c_hstep * 2);
                        __m128 _cc3 = _mm_loadu_ps(pC + c_hstep * 3);
                        __m128 _cc4 = _mm_loadu_ps(pC + c_hstep * 4);
                        __m128 _cc5 = _mm_loadu_ps(pC + c_hstep * 5);
                        __m128 _cc6 = _mm_loadu_ps(pC + c_hstep * 6);
                        __m128 _cc7 = _mm_loadu_ps(pC + c_hstep * 7);
                        __m128 _cc8 = _mm_loadu_ps(pC + c_hstep * 8);
                        __m128 _cc9 = _mm_loadu_ps(pC + c_hstep * 9);
                        __m128 _cca = _mm_loadu_ps(pC + c_hstep * 10);
                        __m128 _ccb = _mm_loadu_ps(pC + c_hstep * 11);
                        __m128 _ccc = _mm_loadu_ps(pC + c_hstep * 12);
                        __m128 _ccd = _mm_loadu_ps(pC + c_hstep * 13);
                        __m128 _cce = _mm_loadu_ps(pC + c_hstep * 14);
                        __m128 _ccf = _mm_loadu_ps(pC + c_hstep * 15);
                        _MM_TRANSPOSE4_PS(_cc0, _cc1, _cc2, _cc3);
                        _MM_TRANSPOSE4_PS(_cc4, _cc5, _cc6, _cc7);
                        _MM_TRANSPOSE4_PS(_cc8, _cc9, _cca, _ccb);
                        _MM_TRANSPOSE4_PS(_ccc, _ccd, _cce, _ccf);

                        _c0 = combine4x4_ps(_cc0, _cc4, _cc8, _ccc);
                        _c1 = combine4x4_ps(_cc1, _cc5, _cc9, _ccd);
                        _c2 = combine4x4_ps(_cc2, _cc6, _cca, _cce);
                        _c3 = combine4x4_ps(_cc3, _cc7, _ccb, _ccf);

                        pC += 4;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0);
                        _f1 = _mm512_add_ps(_f1, _c1);
                        _f2 = _mm512_add_ps(_f2, _c2);
                        _f3 = _mm512_add_ps(_f3, _c3);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm512_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm512_fmadd_ps(_c3, _beta, _f3);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm512_set1_ps(pC[0] * beta);
                    __m512 _c1 = _mm512_set1_ps(pC[1] * beta);
                    __m512 _c2 = _mm512_set1_ps(pC[2] * beta);
                    __m512 _c3 = _mm512_set1_ps(pC[3] * beta);

                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c1);
                    _f2 = _mm512_add_ps(_f2, _c2);
                    _f3 = _mm512_add_ps(_f3, _c3);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
                _f2 = _mm512_mul_ps(_f2, _alpha);
                _f3 = _mm512_mul_ps(_f3, _alpha);
            }

            __m256i _bf0 = float2bfloat_avx512(_f0);
            __m256i _bf1 = float2bfloat_avx512(_f1);
            __m256i _bf2 = float2bfloat_avx512(_f2);
            __m256i _bf3 = float2bfloat_avx512(_f3);

            if (output_transpose)
            {
                if (out_elempack == 4)
                {
                    transpose16x4_epi16(_bf0, _bf1, _bf2, _bf3);

                    // All jj0-3: _bf0=ii0-3, _bf1=ii4-7, _bf2=ii8-11, _bf3=ii12-15
                    _mm_storel_epi64((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storeh_pd((double*)(p0 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 2), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storeh_pd((double*)(p0 + 4 * 3), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 4), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storeh_pd((double*)(p0 + 4 * 5), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 6), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storeh_pd((double*)(p0 + 4 * 7), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 8), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storeh_pd((double*)(p0 + 4 * 9), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 10), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storeh_pd((double*)(p0 + 4 * 11), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 12), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storeh_pd((double*)(p0 + 4 * 13), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 14), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storeh_pd((double*)(p0 + 4 * 15), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 1)));
                }
                if (out_elempack == 1)
                {
                    _mm256_storeu_si256((__m256i*)p0, _bf0);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep), _bf1);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 2), _bf2);
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 3), _bf3);
                }
                p0 += out_hstep * 4;
            }
            else
            {
                if (out_elempack == 16)
                {
                    _mm256_storeu_si256((__m256i*)p0, _bf0);
                    _mm256_storeu_si256((__m256i*)(p0 + 16), _bf1);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 2), _bf2);
                    _mm256_storeu_si256((__m256i*)(p0 + 16 * 3), _bf3);
                    p0 += 64;
                }
                if (out_elempack == 8)
                {
                    _mm_storeu_si128((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 2), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 3), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 2), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 3), _mm256_extractf128_si256(_bf3, 1));
                    p0 += 32;
                }
                if (out_elempack == 4)
                {
                    _mm_storel_epi64((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 2), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 3), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 2), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 3), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 2), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 3), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 2), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 3), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 1)));
                    p0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose16x4_epi16(_bf0, _bf1, _bf2, _bf3);

                    _mm_storel_epi64((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 2), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 3), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 5), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 6), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 7), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 9), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 10), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 11), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 12), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 13), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 14), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 15), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 1)));
                    p0 += 4;
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m512 _f0 = _mm512_load_ps(pp);
            __m512 _f1 = _mm512_load_ps(pp + 16);
            pp += 32;

            // deshuffle from the shuffle-based 16x2 dpbf16_ps kernel
            {
                __m512 _tmp0 = _mm512_permute_ps(_f0, _MM_SHUFFLE(3, 1, 2, 0));
                __m512 _tmp1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(0, 2, 3, 1));
                _f0 = _mm512_unpacklo_ps(_tmp0, _tmp1);
                _f1 = _mm512_unpackhi_ps(_tmp0, _tmp1);
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
            }

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c1;
                    if (c_elempack == 16)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + 16);
                        pC += 32;
                    }
                    else if (c_elempack == 8)
                    {
                        __m512 _cc0 = _mm512_loadu_ps(pC);
                        __m512 _cc1 = _mm512_loadu_ps(pC + c_hstep * 8);
                        _c0 = _mm512_shuffle_f32x4(_cc0, _cc1, _MM_SHUFFLE(1, 0, 1, 0));
                        _c1 = _mm512_shuffle_f32x4(_cc0, _cc1, _MM_SHUFFLE(3, 2, 3, 2));
                        pC += 16;
                    }
                    else if (c_elempack == 4)
                    {
                        __m128 _cc0 = _mm_loadu_ps(pC);
                        __m128 _cc1 = _mm_loadu_ps(pC + 4);
                        __m128 _cc2 = _mm_loadu_ps(pC + c_hstep * 4);
                        __m128 _cc3 = _mm_loadu_ps(pC + c_hstep * 4 + 4);
                        __m128 _cc4 = _mm_loadu_ps(pC + c_hstep * 8);
                        __m128 _cc5 = _mm_loadu_ps(pC + c_hstep * 8 + 4);
                        __m128 _cc6 = _mm_loadu_ps(pC + c_hstep * 12);
                        __m128 _cc7 = _mm_loadu_ps(pC + c_hstep * 12 + 4);
                        _c0 = combine4x4_ps(_cc0, _cc2, _cc4, _cc6);
                        _c1 = combine4x4_ps(_cc1, _cc3, _cc5, _cc7);
                        pC += 8;
                    }
                    else // if (c_elempack == 1)
                    {
                        __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(c_hstep));
                        _c0 = _mm512_i32gather_ps(_vindex, pC, sizeof(float));
                        _c1 = _mm512_i32gather_ps(_vindex, pC + 1, sizeof(float));
                        pC += 2;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0);
                        _f1 = _mm512_add_ps(_f1, _c1);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1, _beta, _f1);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm512_set1_ps(pC[0] * beta);
                    __m512 _c1 = _mm512_set1_ps(pC[1] * beta);
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c1);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
            }

            __m256i _bf0 = float2bfloat_avx512(_f0);
            __m256i _bf1 = float2bfloat_avx512(_f1);

            if (output_transpose)
            {
                _mm256_storeu_si256((__m256i*)p0, _bf0);
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep), _bf1);
                p0 += out_hstep * 2;
            }
            else
            {
                if (out_elempack == 16)
                {
                    _mm256_storeu_si256((__m256i*)p0, _bf0);
                    _mm256_storeu_si256((__m256i*)(p0 + 16), _bf1);
                    p0 += 32;
                }
                if (out_elempack == 8)
                {
                    _mm_storeu_si128((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8), _mm256_extractf128_si256(_bf1, 1));
                    p0 += 16;
                }
                if (out_elempack == 4)
                {
                    _mm_storel_epi64((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 1)));
                    p0 += 8;
                }
                if (out_elempack == 1)
                {
                    transpose16x2_epi16(_bf0, _bf1);
                    __m512i _bf01 = combine8x2_epi32(_bf0, _bf1);
                    __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(out_hstep));
                    _mm512_i32scatter_epi32(p0, _vindex, _bf01, sizeof(unsigned short));
                    p0 += 2;
                }
            }
        }
        for (; jj < max_jj; jj++)
        {
            __m512 _f0 = _mm512_load_ps(pp);
            pp += 16;

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 16)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        pC += 16;
                    }
                    else if (c_elempack == 8)
                    {
                        __m256 _cc0 = _mm256_loadu_ps(pC);
                        __m256 _cc1 = _mm256_loadu_ps(pC + c_hstep * 8);
                        _c0 = combine8x2_ps(_cc0, _cc1);
                        pC += 8;
                    }
                    else if (c_elempack == 4)
                    {
                        __m128 _cc0 = _mm_loadu_ps(pC);
                        __m128 _cc1 = _mm_loadu_ps(pC + c_hstep * 4);
                        __m128 _cc2 = _mm_loadu_ps(pC + c_hstep * 8);
                        __m128 _cc3 = _mm_loadu_ps(pC + c_hstep * 12);
                        _c0 = combine4x4_ps(_cc0, _cc1, _cc2, _cc3);
                        pC += 4;
                    }
                    else // if (c_elempack == 1)
                    {
                        __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(c_hstep));
                        _c0 = _mm512_i32gather_ps(_vindex, pC, sizeof(float));
                        pC += 1;
                    }
                    _f0 = _mm512_fmadd_ps(_c0, _mm512_set1_ps(beta), _f0);
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm512_set1_ps(pC[0] * beta);
                    _f0 = _mm512_add_ps(_f0, _c0);
                    pC += 1;
                }
            }

            if (alpha != 1.f)
            {
                _f0 = _mm512_mul_ps(_f0, _mm512_set1_ps(alpha));
            }

            __m256i _bf0 = float2bfloat_avx512(_f0);

            if (output_transpose)
            {
                _mm256_storeu_si256((__m256i*)p0, _bf0);
                p0 += out_hstep;
            }
            else
            {
                if (out_elempack == 16)
                {
                    _mm256_storeu_si256((__m256i*)p0, _bf0);
                    p0 += 16;
                }
                if (out_elempack == 8)
                {
                    _mm_storeu_si128((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf0, 1));
                    p0 += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_storel_epi64((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 1)));
                    p0 += 4;
                }
                if (out_elempack == 1)
                {
                    unsigned short tmp[16];
                    _mm256_storeu_si256((__m256i*)tmp, _bf0);

                    p0[0] = tmp[0];
                    p0[out_hstep] = tmp[1];
                    p0[out_hstep * 2] = tmp[2];
                    p0[out_hstep * 3] = tmp[3];
                    p0[out_hstep * 4] = tmp[4];
                    p0[out_hstep * 5] = tmp[5];
                    p0[out_hstep * 6] = tmp[6];
                    p0[out_hstep * 7] = tmp[7];
                    p0[out_hstep * 8] = tmp[8];
                    p0[out_hstep * 9] = tmp[9];
                    p0[out_hstep * 10] = tmp[10];
                    p0[out_hstep * 11] = tmp[11];
                    p0[out_hstep * 12] = tmp[12];
                    p0[out_hstep * 13] = tmp[13];
                    p0[out_hstep * 14] = tmp[14];
                    p0[out_hstep * 15] = tmp[15];
                    p0++;
                }
            }
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

        __m256 _c0 = _mm256_set1_ps(0.f);
#if __AVX512F__
        __m512 _c0_avx512 = _mm512_set1_ps(0.f);
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = _mm256_set1_ps(pC[0] * beta);
#if __AVX512F__
                _c0_avx512 = _mm512_set1_ps(pC[0] * beta);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = _mm256_loadu_ps(pC);
                _c0 = _mm256_mul_ps(_c0, _mm256_set1_ps(beta));
#if __AVX512F__
                _c0_avx512 = _mm512_broadcast_f32x8(_c0);
#endif
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
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
            pp += 128;

            // deshuffle from the shuffle-based 8x16 dpbf16_ps kernel
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
                _tmp1 = _mm512_shuffle_f32x4(_f0, _f4, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp2 = _mm512_shuffle_f32x4(_f1, _f5, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp3 = _mm512_shuffle_f32x4(_f1, _f5, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp4 = _mm512_shuffle_f32x4(_f2, _f6, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp5 = _mm512_shuffle_f32x4(_f2, _f6, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp6 = _mm512_shuffle_f32x4(_f3, _f7, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp7 = _mm512_shuffle_f32x4(_f3, _f7, _MM_SHUFFLE(2, 3, 3, 2));

                _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                _f1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                _f2 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                _f3 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                _f4 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(1, 3, 1, 3));
                _f5 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(1, 3, 1, 3));
                _f6 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(1, 3, 1, 3));
                _f7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(1, 3, 1, 3));
            }

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c0_avx512);
                    _f2 = _mm512_add_ps(_f2, _c0_avx512);
                    _f3 = _mm512_add_ps(_f3, _c0_avx512);
                    _f4 = _mm512_add_ps(_f4, _c0_avx512);
                    _f5 = _mm512_add_ps(_f5, _c0_avx512);
                    _f6 = _mm512_add_ps(_f6, _c0_avx512);
                    _f7 = _mm512_add_ps(_f7, _c0_avx512);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c0_avx512);
                    _f2 = _mm512_add_ps(_f2, _c0_avx512);
                    _f3 = _mm512_add_ps(_f3, _c0_avx512);
                    _f4 = _mm512_add_ps(_f4, _c0_avx512);
                    _f5 = _mm512_add_ps(_f5, _c0_avx512);
                    _f6 = _mm512_add_ps(_f6, _c0_avx512);
                    _f7 = _mm512_add_ps(_f7, _c0_avx512);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c1_avx512;
                    __m512 _c2_avx512;
                    __m512 _c3_avx512;
                    __m512 _c4_avx512;
                    __m512 _c5_avx512;
                    __m512 _c6_avx512;
                    __m512 _c7_avx512;
                    if (c_elempack == 8)
                    {
                        _c0_avx512 = _mm512_loadu_ps(pC);
                        _c1_avx512 = _mm512_loadu_ps(pC + 16);
                        _c2_avx512 = _mm512_loadu_ps(pC + 32);
                        _c3_avx512 = _mm512_loadu_ps(pC + 48);
                        _c4_avx512 = _mm512_loadu_ps(pC + 64);
                        _c5_avx512 = _mm512_loadu_ps(pC + 80);
                        _c6_avx512 = _mm512_loadu_ps(pC + 96);
                        _c7_avx512 = _mm512_loadu_ps(pC + 112);

                        __m512 _tmp0 = _mm512_shuffle_f32x4(_c0_avx512, _c4_avx512, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_c0_avx512, _c4_avx512, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp2 = _mm512_shuffle_f32x4(_c1_avx512, _c5_avx512, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp3 = _mm512_shuffle_f32x4(_c1_avx512, _c5_avx512, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp4 = _mm512_shuffle_f32x4(_c2_avx512, _c6_avx512, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp5 = _mm512_shuffle_f32x4(_c2_avx512, _c6_avx512, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp6 = _mm512_shuffle_f32x4(_c3_avx512, _c7_avx512, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp7 = _mm512_shuffle_f32x4(_c3_avx512, _c7_avx512, _MM_SHUFFLE(3, 2, 3, 2));

                        _c0_avx512 = _tmp0;
                        _c1_avx512 = _tmp1;
                        _c2_avx512 = _tmp2;
                        _c3_avx512 = _tmp3;
                        _c4_avx512 = _tmp4;
                        _c5_avx512 = _tmp5;
                        _c6_avx512 = _tmp6;
                        _c7_avx512 = _tmp7;

                        pC += 128;
                    }
                    else if (c_elempack == 4)
                    {
                        _c0_avx512 = _mm512_loadu_ps(pC);
                        _c1_avx512 = _mm512_loadu_ps(pC + 16);
                        _c2_avx512 = _mm512_loadu_ps(pC + 32);
                        _c3_avx512 = _mm512_loadu_ps(pC + 48);
                        _c4_avx512 = _mm512_loadu_ps(pC + c_hstep * 4);
                        _c5_avx512 = _mm512_loadu_ps(pC + c_hstep * 4 + 16);
                        _c6_avx512 = _mm512_loadu_ps(pC + c_hstep * 4 + 32);
                        _c7_avx512 = _mm512_loadu_ps(pC + c_hstep * 4 + 48);

                        __m512 _tmp0 = _mm512_shuffle_f32x4(_c0_avx512, _c2_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_c0_avx512, _c2_avx512, _MM_SHUFFLE(3, 1, 3, 1));
                        __m512 _tmp2 = _mm512_shuffle_f32x4(_c1_avx512, _c3_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        __m512 _tmp3 = _mm512_shuffle_f32x4(_c1_avx512, _c3_avx512, _MM_SHUFFLE(3, 1, 3, 1));
                        __m512 _tmp4 = _mm512_shuffle_f32x4(_c4_avx512, _c6_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        __m512 _tmp5 = _mm512_shuffle_f32x4(_c4_avx512, _c6_avx512, _MM_SHUFFLE(3, 1, 3, 1));
                        __m512 _tmp6 = _mm512_shuffle_f32x4(_c5_avx512, _c7_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        __m512 _tmp7 = _mm512_shuffle_f32x4(_c5_avx512, _c7_avx512, _MM_SHUFFLE(3, 1, 3, 1));

                        _c0_avx512 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(2, 0, 2, 0));
                        _c1_avx512 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                        _c2_avx512 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(3, 1, 3, 1));
                        _c3_avx512 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                        _c4_avx512 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(2, 0, 2, 0));
                        _c5_avx512 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                        _c6_avx512 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(3, 1, 3, 1));
                        _c7_avx512 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                        _c0_avx512 = _mm512_shuffle_f32x4(_c0_avx512, _c0_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c1_avx512 = _mm512_shuffle_f32x4(_c1_avx512, _c1_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c2_avx512 = _mm512_shuffle_f32x4(_c2_avx512, _c2_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c3_avx512 = _mm512_shuffle_f32x4(_c3_avx512, _c3_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c4_avx512 = _mm512_shuffle_f32x4(_c4_avx512, _c4_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c5_avx512 = _mm512_shuffle_f32x4(_c5_avx512, _c5_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c6_avx512 = _mm512_shuffle_f32x4(_c6_avx512, _c6_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c7_avx512 = _mm512_shuffle_f32x4(_c7_avx512, _c7_avx512, _MM_SHUFFLE(3, 1, 2, 0));

                        pC += 64;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0_avx512 = _mm512_loadu_ps(pC);
                        _c1_avx512 = _mm512_loadu_ps(pC + c_hstep);
                        _c2_avx512 = _mm512_loadu_ps(pC + c_hstep * 2);
                        _c3_avx512 = _mm512_loadu_ps(pC + c_hstep * 3);
                        _c4_avx512 = _mm512_loadu_ps(pC + c_hstep * 4);
                        _c5_avx512 = _mm512_loadu_ps(pC + c_hstep * 5);
                        _c6_avx512 = _mm512_loadu_ps(pC + c_hstep * 6);
                        _c7_avx512 = _mm512_loadu_ps(pC + c_hstep * 7);

                        __m512 _tmp0 = _mm512_unpacklo_ps(_c0_avx512, _c1_avx512);
                        __m512 _tmp1 = _mm512_unpacklo_ps(_c2_avx512, _c3_avx512);
                        __m512 _tmp2 = _mm512_unpacklo_ps(_c4_avx512, _c5_avx512);
                        __m512 _tmp3 = _mm512_unpacklo_ps(_c6_avx512, _c7_avx512);
                        __m512 _tmp4 = _mm512_unpackhi_ps(_c0_avx512, _c1_avx512);
                        __m512 _tmp5 = _mm512_unpackhi_ps(_c2_avx512, _c3_avx512);
                        __m512 _tmp6 = _mm512_unpackhi_ps(_c4_avx512, _c5_avx512);
                        __m512 _tmp7 = _mm512_unpackhi_ps(_c6_avx512, _c7_avx512);

                        _c0_avx512 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                        _c1_avx512 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                        _c2_avx512 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                        _c3_avx512 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                        _c4_avx512 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp5)));
                        _c5_avx512 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp6), _mm512_castps_pd(_tmp7)));
                        _c6_avx512 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp5)));
                        _c7_avx512 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp6), _mm512_castps_pd(_tmp7)));

                        _tmp0 = _mm512_shuffle_f32x4(_c0_avx512, _c1_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        _tmp1 = _mm512_shuffle_f32x4(_c2_avx512, _c3_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        _tmp2 = _mm512_shuffle_f32x4(_c4_avx512, _c5_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        _tmp3 = _mm512_shuffle_f32x4(_c6_avx512, _c7_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        _tmp4 = _mm512_shuffle_f32x4(_c0_avx512, _c1_avx512, _MM_SHUFFLE(3, 1, 3, 1));
                        _tmp5 = _mm512_shuffle_f32x4(_c2_avx512, _c3_avx512, _MM_SHUFFLE(3, 1, 3, 1));
                        _tmp6 = _mm512_shuffle_f32x4(_c4_avx512, _c5_avx512, _MM_SHUFFLE(3, 1, 3, 1));
                        _tmp7 = _mm512_shuffle_f32x4(_c6_avx512, _c7_avx512, _MM_SHUFFLE(3, 1, 3, 1));

                        _c0_avx512 = _mm512_shuffle_f32x4(_tmp0, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                        _c1_avx512 = _mm512_shuffle_f32x4(_tmp1, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                        _c2_avx512 = _mm512_shuffle_f32x4(_tmp2, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                        _c3_avx512 = _mm512_shuffle_f32x4(_tmp3, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                        _c4_avx512 = _mm512_shuffle_f32x4(_tmp4, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                        _c5_avx512 = _mm512_shuffle_f32x4(_tmp5, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                        _c6_avx512 = _mm512_shuffle_f32x4(_tmp6, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                        _c7_avx512 = _mm512_shuffle_f32x4(_tmp7, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));

                        pC += 16;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0_avx512);
                        _f1 = _mm512_add_ps(_f1, _c1_avx512);
                        _f2 = _mm512_add_ps(_f2, _c2_avx512);
                        _f3 = _mm512_add_ps(_f3, _c3_avx512);
                        _f4 = _mm512_add_ps(_f4, _c4_avx512);
                        _f5 = _mm512_add_ps(_f5, _c5_avx512);
                        _f6 = _mm512_add_ps(_f6, _c6_avx512);
                        _f7 = _mm512_add_ps(_f7, _c7_avx512);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0_avx512, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1_avx512, _beta, _f1);
                        _f2 = _mm512_fmadd_ps(_c2_avx512, _beta, _f2);
                        _f3 = _mm512_fmadd_ps(_c3_avx512, _beta, _f3);
                        _f4 = _mm512_fmadd_ps(_c4_avx512, _beta, _f4);
                        _f5 = _mm512_fmadd_ps(_c5_avx512, _beta, _f5);
                        _f6 = _mm512_fmadd_ps(_c6_avx512, _beta, _f6);
                        _f7 = _mm512_fmadd_ps(_c7_avx512, _beta, _f7);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    __m512 _cc = _mm512_loadu_ps(pC);
                    _cc = _mm512_mul_ps(_cc, _mm512_set1_ps(beta));
                    __m512 _cc0 = _mm512_permute_ps(_cc, _MM_SHUFFLE(0, 0, 0, 0));
                    __m512 _cc1 = _mm512_permute_ps(_cc, _MM_SHUFFLE(1, 1, 1, 1));
                    __m512 _cc2 = _mm512_permute_ps(_cc, _MM_SHUFFLE(2, 2, 2, 2));
                    __m512 _cc3 = _mm512_permute_ps(_cc, _MM_SHUFFLE(3, 3, 3, 3));

                    _c0_avx512 = _mm512_shuffle_f32x4(_cc0, _cc0, _MM_SHUFFLE(2, 2, 0, 0));
                    __m512 _c1_avx512 = _mm512_shuffle_f32x4(_cc1, _cc1, _MM_SHUFFLE(2, 2, 0, 0));
                    __m512 _c2_avx512 = _mm512_shuffle_f32x4(_cc2, _cc2, _MM_SHUFFLE(2, 2, 0, 0));
                    __m512 _c3_avx512 = _mm512_shuffle_f32x4(_cc3, _cc3, _MM_SHUFFLE(2, 2, 0, 0));
                    __m512 _c4_avx512 = _mm512_shuffle_f32x4(_cc0, _cc0, _MM_SHUFFLE(3, 3, 1, 1));
                    __m512 _c5_avx512 = _mm512_shuffle_f32x4(_cc1, _cc1, _MM_SHUFFLE(3, 3, 1, 1));
                    __m512 _c6_avx512 = _mm512_shuffle_f32x4(_cc2, _cc2, _MM_SHUFFLE(3, 3, 1, 1));
                    __m512 _c7_avx512 = _mm512_shuffle_f32x4(_cc3, _cc3, _MM_SHUFFLE(3, 3, 1, 1));

                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c1_avx512);
                    _f2 = _mm512_add_ps(_f2, _c2_avx512);
                    _f3 = _mm512_add_ps(_f3, _c3_avx512);
                    _f4 = _mm512_add_ps(_f4, _c4_avx512);
                    _f5 = _mm512_add_ps(_f5, _c5_avx512);
                    _f6 = _mm512_add_ps(_f6, _c6_avx512);
                    _f7 = _mm512_add_ps(_f7, _c7_avx512);

                    pC += 16;
                }
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

            __m256i _bf0 = float2bfloat_avx512(_f0);
            __m256i _bf1 = float2bfloat_avx512(_f1);
            __m256i _bf2 = float2bfloat_avx512(_f2);
            __m256i _bf3 = float2bfloat_avx512(_f3);
            __m256i _bf4 = float2bfloat_avx512(_f4);
            __m256i _bf5 = float2bfloat_avx512(_f5);
            __m256i _bf6 = float2bfloat_avx512(_f6);
            __m256i _bf7 = float2bfloat_avx512(_f7);

            if (output_transpose)
            {
                if (out_elempack == 16)
                {
                    transpose16x8_epi16(_bf0, _bf1, _bf2, _bf3, _bf4, _bf5, _bf6, _bf7);

                    _mm_store_si128((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_store_si128((__m128i*)(p0 + 8), _mm256_extractf128_si256(_bf4, 0));
                    _mm_store_si128((__m128i*)(p0 + 16), _mm256_extractf128_si256(_bf0, 1));
                    _mm_store_si128((__m128i*)(p0 + 16 + 8), _mm256_extractf128_si256(_bf4, 1));
                    _mm_store_si128((__m128i*)(p0 + 32), _mm256_extractf128_si256(_bf1, 0));
                    _mm_store_si128((__m128i*)(p0 + 32 + 8), _mm256_extractf128_si256(_bf5, 0));
                    _mm_store_si128((__m128i*)(p0 + 48), _mm256_extractf128_si256(_bf1, 1));
                    _mm_store_si128((__m128i*)(p0 + 48 + 8), _mm256_extractf128_si256(_bf5, 1));
                    _mm_store_si128((__m128i*)(p0 + 64), _mm256_extractf128_si256(_bf2, 0));
                    _mm_store_si128((__m128i*)(p0 + 64 + 8), _mm256_extractf128_si256(_bf6, 0));
                    _mm_store_si128((__m128i*)(p0 + 80), _mm256_extractf128_si256(_bf2, 1));
                    _mm_store_si128((__m128i*)(p0 + 80 + 8), _mm256_extractf128_si256(_bf6, 1));
                    _mm_store_si128((__m128i*)(p0 + 96), _mm256_extractf128_si256(_bf3, 0));
                    _mm_store_si128((__m128i*)(p0 + 96 + 8), _mm256_extractf128_si256(_bf7, 0));
                    _mm_store_si128((__m128i*)(p0 + 112), _mm256_extractf128_si256(_bf3, 1));
                    _mm_store_si128((__m128i*)(p0 + 112 + 8), _mm256_extractf128_si256(_bf7, 1));
                }
                if (out_elempack == 8)
                {
                    // _bf_k is __m256i with 16 bf16: low128=[ii0..7] for jj=k, high128=[ii0..7] for jj=k+8
                    // Need to transpose to [jj0..7 for ii=k]
                    // Treat low halves as 8x8 block and high halves as another 8x8 block
                    __m128i _bf0l = _mm256_extractf128_si256(_bf0, 0);
                    __m128i _bf1l = _mm256_extractf128_si256(_bf1, 0);
                    __m128i _bf2l = _mm256_extractf128_si256(_bf2, 0);
                    __m128i _bf3l = _mm256_extractf128_si256(_bf3, 0);
                    __m128i _bf4l = _mm256_extractf128_si256(_bf4, 0);
                    __m128i _bf5l = _mm256_extractf128_si256(_bf5, 0);
                    __m128i _bf6l = _mm256_extractf128_si256(_bf6, 0);
                    __m128i _bf7l = _mm256_extractf128_si256(_bf7, 0);
                    transpose8x8_epi16(_bf0l, _bf1l, _bf2l, _bf3l, _bf4l, _bf5l, _bf6l, _bf7l);
                    _mm_storeu_si128((__m128i*)p0, _bf0l);
                    _mm_storeu_si128((__m128i*)(p0 + 8), _bf1l);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 2), _bf2l);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 3), _bf3l);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 4), _bf4l);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 5), _bf5l);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 6), _bf6l);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 7), _bf7l);
                    __m128i _bf0h = _mm256_extractf128_si256(_bf0, 1);
                    __m128i _bf1h = _mm256_extractf128_si256(_bf1, 1);
                    __m128i _bf2h = _mm256_extractf128_si256(_bf2, 1);
                    __m128i _bf3h = _mm256_extractf128_si256(_bf3, 1);
                    __m128i _bf4h = _mm256_extractf128_si256(_bf4, 1);
                    __m128i _bf5h = _mm256_extractf128_si256(_bf5, 1);
                    __m128i _bf6h = _mm256_extractf128_si256(_bf6, 1);
                    __m128i _bf7h = _mm256_extractf128_si256(_bf7, 1);
                    transpose8x8_epi16(_bf0h, _bf1h, _bf2h, _bf3h, _bf4h, _bf5h, _bf6h, _bf7h);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8), _bf0h);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8), _bf1h);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 2), _bf2h);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 3), _bf3h);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 4), _bf4h);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 5), _bf5h);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 6), _bf6h);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8 * 7), _bf7h);
                }
                if (out_elempack == 4)
                {
                    __m128i _bf0l = _mm256_extractf128_si256(_bf0, 0);
                    __m128i _bf1l = _mm256_extractf128_si256(_bf1, 0);
                    __m128i _bf2l = _mm256_extractf128_si256(_bf2, 0);
                    __m128i _bf3l = _mm256_extractf128_si256(_bf3, 0);
                    __m128i _bf4l = _mm256_extractf128_si256(_bf4, 0);
                    __m128i _bf5l = _mm256_extractf128_si256(_bf5, 0);
                    __m128i _bf6l = _mm256_extractf128_si256(_bf6, 0);
                    __m128i _bf7l = _mm256_extractf128_si256(_bf7, 0);
                    transpose8x4_epi16(_bf0l, _bf1l, _bf2l, _bf3l);
                    transpose8x4_epi16(_bf4l, _bf5l, _bf6l, _bf7l);
                    // After transpose: _bf0l=[jj0..3 for ii0 | jj0..3 for ii1]
                    // jj0..3 row
                    _mm_storel_epi64((__m128i*)p0, _bf0l);
                    _mm_storeh_pd((double*)(p0 + 4), _mm_castsi128_pd(_bf0l));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 2), _bf1l);
                    _mm_storeh_pd((double*)(p0 + 4 * 3), _mm_castsi128_pd(_bf1l));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 4), _bf2l);
                    _mm_storeh_pd((double*)(p0 + 4 * 5), _mm_castsi128_pd(_bf2l));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 6), _bf3l);
                    _mm_storeh_pd((double*)(p0 + 4 * 7), _mm_castsi128_pd(_bf3l));
                    // jj4..7 row
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4), _bf4l);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4), _mm_castsi128_pd(_bf4l));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 4 * 2), _bf5l);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 3), _mm_castsi128_pd(_bf5l));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 4 * 4), _bf6l);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 5), _mm_castsi128_pd(_bf6l));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 4 * 6), _bf7l);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 7), _mm_castsi128_pd(_bf7l));
                    __m128i _bf0h = _mm256_extractf128_si256(_bf0, 1);
                    __m128i _bf1h = _mm256_extractf128_si256(_bf1, 1);
                    __m128i _bf2h = _mm256_extractf128_si256(_bf2, 1);
                    __m128i _bf3h = _mm256_extractf128_si256(_bf3, 1);
                    __m128i _bf4h = _mm256_extractf128_si256(_bf4, 1);
                    __m128i _bf5h = _mm256_extractf128_si256(_bf5, 1);
                    __m128i _bf6h = _mm256_extractf128_si256(_bf6, 1);
                    __m128i _bf7h = _mm256_extractf128_si256(_bf7, 1);
                    transpose8x4_epi16(_bf0h, _bf1h, _bf2h, _bf3h);
                    transpose8x4_epi16(_bf4h, _bf5h, _bf6h, _bf7h);
                    // jj8..11 row
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8), _bf0h);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 8 + 4), _mm_castsi128_pd(_bf0h));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 2), _bf1h);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 8 + 4 * 3), _mm_castsi128_pd(_bf1h));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 4), _bf2h);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 8 + 4 * 5), _mm_castsi128_pd(_bf2h));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 6), _bf3h);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 8 + 4 * 7), _mm_castsi128_pd(_bf3h));
                    // jj12..15 row
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 12), _bf4h);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4), _mm_castsi128_pd(_bf4h));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 12 + 4 * 2), _bf5h);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 3), _mm_castsi128_pd(_bf5h));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 12 + 4 * 4), _bf6h);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 5), _mm_castsi128_pd(_bf6h));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 12 + 4 * 6), _bf7h);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 7), _mm_castsi128_pd(_bf7h));
                }
                if (out_elempack == 1)
                {
                    _mm_storeu_si128((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 2), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 3), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 4), _mm256_extractf128_si256(_bf4, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 5), _mm256_extractf128_si256(_bf5, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 6), _mm256_extractf128_si256(_bf6, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 7), _mm256_extractf128_si256(_bf7, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 9), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 10), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 11), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 12), _mm256_extractf128_si256(_bf4, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 13), _mm256_extractf128_si256(_bf5, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 14), _mm256_extractf128_si256(_bf6, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 15), _mm256_extractf128_si256(_bf7, 1));
                }
                p0 += out_hstep * 16;
            }
            else
            {
                if (out_elempack == 8)
                {
                    // _bf_k low128=[ii0..7] for jj=k, high128=[ii0..7] for jj=k+8
                    // Already packed for out_elempack==8: 8 ii values contiguous
                    _mm_storeu_si128((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 2), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 3), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 4), _mm256_extractf128_si256(_bf4, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 5), _mm256_extractf128_si256(_bf5, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 6), _mm256_extractf128_si256(_bf6, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 7), _mm256_extractf128_si256(_bf7, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 9), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 10), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 11), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 12), _mm256_extractf128_si256(_bf4, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 13), _mm256_extractf128_si256(_bf5, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 14), _mm256_extractf128_si256(_bf6, 1));
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 15), _mm256_extractf128_si256(_bf7, 1));
                    p0 += 128;
                }
                if (out_elempack == 4)
                {
                    // low128 low64=[ii0..3] for jj=k, low128 high64=[ii4..7] for jj=k
                    // high128 low64=[ii0..3] for jj=k+8, high128 high64=[ii4..7] for jj=k+8
                    _mm_storel_epi64((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 2), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 3), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 4), _mm256_extractf128_si256(_bf4, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 5), _mm256_extractf128_si256(_bf5, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 6), _mm256_extractf128_si256(_bf6, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 7), _mm256_extractf128_si256(_bf7, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 2), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 3), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf4, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 5), _mm_castsi128_pd(_mm256_extractf128_si256(_bf5, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 6), _mm_castsi128_pd(_mm256_extractf128_si256(_bf6, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 7), _mm_castsi128_pd(_mm256_extractf128_si256(_bf7, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 9), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 10), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 11), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 12), _mm256_extractf128_si256(_bf4, 1));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 13), _mm256_extractf128_si256(_bf5, 1));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 14), _mm256_extractf128_si256(_bf6, 1));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 15), _mm256_extractf128_si256(_bf7, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 8), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 9), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 10), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 11), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf4, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 13), _mm_castsi128_pd(_mm256_extractf128_si256(_bf5, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 14), _mm_castsi128_pd(_mm256_extractf128_si256(_bf6, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 15), _mm_castsi128_pd(_mm256_extractf128_si256(_bf7, 1)));
                    p0 += 64;
                }
                if (out_elempack == 1)
                {
                    // Transpose deshuffled 8x16 to row-major for non-transpose output
                    // Same as int8 out_elempack==1 non-transpose store
                    __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f1);
                    __m512 _tmp1 = _mm512_unpacklo_ps(_f2, _f3);
                    __m512 _tmp2 = _mm512_unpacklo_ps(_f4, _f5);
                    __m512 _tmp3 = _mm512_unpacklo_ps(_f6, _f7);
                    __m512 _tmp4 = _mm512_unpackhi_ps(_f0, _f1);
                    __m512 _tmp5 = _mm512_unpackhi_ps(_f2, _f3);
                    __m512 _tmp6 = _mm512_unpackhi_ps(_f4, _f5);
                    __m512 _tmp7 = _mm512_unpackhi_ps(_f6, _f7);

                    _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                    _f1 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                    _f2 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                    _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                    _f4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp5)));
                    _f5 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp6), _mm512_castps_pd(_tmp7)));
                    _f6 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp5)));
                    _f7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp6), _mm512_castps_pd(_tmp7)));

                    _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp1 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp2 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp3 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp4 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp5 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp6 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp7 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(3, 1, 3, 1));

                    _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                    _f1 = _mm512_shuffle_f32x4(_tmp1, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                    _f2 = _mm512_shuffle_f32x4(_tmp2, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                    _f3 = _mm512_shuffle_f32x4(_tmp3, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                    _f4 = _mm512_shuffle_f32x4(_tmp4, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                    _f5 = _mm512_shuffle_f32x4(_tmp5, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                    _f6 = _mm512_shuffle_f32x4(_tmp6, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                    _f7 = _mm512_shuffle_f32x4(_tmp7, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));

                    _mm256_storeu_si256((__m256i*)p0, float2bfloat_avx512(_f0));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep), float2bfloat_avx512(_f1));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 2), float2bfloat_avx512(_f2));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 3), float2bfloat_avx512(_f3));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 4), float2bfloat_avx512(_f4));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 5), float2bfloat_avx512(_f5));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 6), float2bfloat_avx512(_f6));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 7), float2bfloat_avx512(_f7));
                    p0 += 16;
                }
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

            // deshuffle from the shuffle-based 8x8 dpbf16_ps kernel
            // from
            //      00 11 22 33 04 15 26 37
            //      20 31 02 13 24 35 06 17
            //      01 12 23 30 05 16 27 34
            //      21 32 03 10 25 36 07 14
            //      40 51 62 73 44 55 66 77
            //      60 71 42 53 64 75 46 57
            //      41 52 63 70 45 56 67 74
            //      61 72 43 50 65 76 47 54
            // to
            //      00 10 20 30 40 50 60 70
            //      01 11 21 31 41 51 61 71
            //      02 12 22 32 42 52 62 72
            //      03 13 23 33 43 53 63 73
            //      04 14 24 34 44 54 64 74
            //      05 15 25 35 45 55 65 75
            //      06 16 26 36 46 56 66 76
            //      07 17 27 37 47 57 67 77
            {
                __m256 _tmp0 = _f0;
                __m256 _tmp1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp2 = _f2;
                __m256 _tmp3 = _mm256_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp4 = _f4;
                __m256 _tmp5 = _mm256_shuffle_ps(_f5, _f5, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp6 = _f6;
                __m256 _tmp7 = _mm256_shuffle_ps(_f7, _f7, _MM_SHUFFLE(2, 1, 0, 3));

                _f0 = _mm256_unpacklo_ps(_tmp0, _tmp3);
                _f1 = _mm256_unpackhi_ps(_tmp0, _tmp3);
                _f2 = _mm256_unpacklo_ps(_tmp2, _tmp1);
                _f3 = _mm256_unpackhi_ps(_tmp2, _tmp1);
                _f4 = _mm256_unpacklo_ps(_tmp4, _tmp7);
                _f5 = _mm256_unpackhi_ps(_tmp4, _tmp7);
                _f6 = _mm256_unpacklo_ps(_tmp6, _tmp5);
                _f7 = _mm256_unpackhi_ps(_tmp6, _tmp5);

                _tmp0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_f0), _mm256_castps_pd(_f2)));
                _tmp1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_f0), _mm256_castps_pd(_f2)));
                _tmp2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_f3), _mm256_castps_pd(_f1)));
                _tmp3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_f3), _mm256_castps_pd(_f1)));
                _tmp4 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_f4), _mm256_castps_pd(_f6)));
                _tmp5 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_f4), _mm256_castps_pd(_f6)));
                _tmp6 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_f7), _mm256_castps_pd(_f5)));
                _tmp7 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_f7), _mm256_castps_pd(_f5)));

                _tmp1 = _mm256_shuffle_ps(_tmp1, _tmp1, _MM_SHUFFLE(2, 1, 0, 3));
                _tmp3 = _mm256_shuffle_ps(_tmp3, _tmp3, _MM_SHUFFLE(2, 1, 0, 3));
                _tmp5 = _mm256_shuffle_ps(_tmp5, _tmp5, _MM_SHUFFLE(2, 1, 0, 3));
                _tmp7 = _mm256_shuffle_ps(_tmp7, _tmp7, _MM_SHUFFLE(2, 1, 0, 3));

                _f0 = _mm256_permute2f128_ps(_tmp0, _tmp4, _MM_SHUFFLE(0, 3, 0, 0));
                _f1 = _mm256_permute2f128_ps(_tmp1, _tmp5, _MM_SHUFFLE(0, 3, 0, 0));
                _f2 = _mm256_permute2f128_ps(_tmp2, _tmp6, _MM_SHUFFLE(0, 3, 0, 0));
                _f3 = _mm256_permute2f128_ps(_tmp3, _tmp7, _MM_SHUFFLE(0, 3, 0, 0));
                _f4 = _mm256_permute2f128_ps(_tmp4, _tmp0, _MM_SHUFFLE(0, 3, 0, 0));
                _f5 = _mm256_permute2f128_ps(_tmp5, _tmp1, _MM_SHUFFLE(0, 3, 0, 0));
                _f6 = _mm256_permute2f128_ps(_tmp6, _tmp2, _MM_SHUFFLE(0, 3, 0, 0));
                _f7 = _mm256_permute2f128_ps(_tmp7, _tmp3, _MM_SHUFFLE(0, 3, 0, 0));
            }
            // _f0..7 are now rows: _f0 = row0(8 cols), ..., _f7 = row7(8 cols)

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c0);
                    _f2 = _mm256_add_ps(_f2, _c0);
                    _f3 = _mm256_add_ps(_f3, _c0);
                    _f4 = _mm256_add_ps(_f4, _c0);
                    _f5 = _mm256_add_ps(_f5, _c0);
                    _f6 = _mm256_add_ps(_f6, _c0);
                    _f7 = _mm256_add_ps(_f7, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c0);
                    _f2 = _mm256_add_ps(_f2, _c0);
                    _f3 = _mm256_add_ps(_f3, _c0);
                    _f4 = _mm256_add_ps(_f4, _c0);
                    _f5 = _mm256_add_ps(_f5, _c0);
                    _f6 = _mm256_add_ps(_f6, _c0);
                    _f7 = _mm256_add_ps(_f7, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c1;
                    __m256 _c2;
                    __m256 _c3;
                    __m256 _c4;
                    __m256 _c5;
                    __m256 _c6;
                    __m256 _c7;
                    if (c_elempack == 8)
                    {
                        _c0 = _mm256_loadu_ps(pC);
                        _c1 = _mm256_loadu_ps(pC + 8);
                        _c2 = _mm256_loadu_ps(pC + 16);
                        _c3 = _mm256_loadu_ps(pC + 24);
                        _c4 = _mm256_loadu_ps(pC + 32);
                        _c5 = _mm256_loadu_ps(pC + 40);
                        _c6 = _mm256_loadu_ps(pC + 48);
                        _c7 = _mm256_loadu_ps(pC + 56);
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
                        _c0 = _mm256_permute2f128_ps(_tmp0, _tmp4, _MM_SHUFFLE(0, 2, 0, 0));
                        _c1 = _mm256_permute2f128_ps(_tmp0, _tmp4, _MM_SHUFFLE(0, 3, 0, 1));
                        _c2 = _mm256_permute2f128_ps(_tmp1, _tmp5, _MM_SHUFFLE(0, 2, 0, 0));
                        _c3 = _mm256_permute2f128_ps(_tmp1, _tmp5, _MM_SHUFFLE(0, 3, 0, 1));
                        _c4 = _mm256_permute2f128_ps(_tmp2, _tmp6, _MM_SHUFFLE(0, 2, 0, 0));
                        _c5 = _mm256_permute2f128_ps(_tmp2, _tmp6, _MM_SHUFFLE(0, 3, 0, 1));
                        _c6 = _mm256_permute2f128_ps(_tmp3, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
                        _c7 = _mm256_permute2f128_ps(_tmp3, _tmp7, _MM_SHUFFLE(0, 3, 0, 1));
                        pC += 32;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = _mm256_loadu_ps(pC);
                        _c1 = _mm256_loadu_ps(pC + c_hstep);
                        _c2 = _mm256_loadu_ps(pC + c_hstep * 2);
                        _c3 = _mm256_loadu_ps(pC + c_hstep * 3);
                        _c4 = _mm256_loadu_ps(pC + c_hstep * 4);
                        _c5 = _mm256_loadu_ps(pC + c_hstep * 5);
                        _c6 = _mm256_loadu_ps(pC + c_hstep * 6);
                        _c7 = _mm256_loadu_ps(pC + c_hstep * 7);
                        transpose8x8_ps(_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7);
                        pC += 8;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm256_add_ps(_f0, _c0);
                        _f1 = _mm256_add_ps(_f1, _c1);
                        _f2 = _mm256_add_ps(_f2, _c2);
                        _f3 = _mm256_add_ps(_f3, _c3);
                        _f4 = _mm256_add_ps(_f4, _c4);
                        _f5 = _mm256_add_ps(_f5, _c5);
                        _f6 = _mm256_add_ps(_f6, _c6);
                        _f7 = _mm256_add_ps(_f7, _c7);
                    }
                    else
                    {
                        __m256 _beta = _mm256_set1_ps(beta);
                        _f0 = _mm256_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm256_comp_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm256_comp_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm256_comp_fmadd_ps(_c3, _beta, _f3);
                        _f4 = _mm256_comp_fmadd_ps(_c4, _beta, _f4);
                        _f5 = _mm256_comp_fmadd_ps(_c5, _beta, _f5);
                        _f6 = _mm256_comp_fmadd_ps(_c6, _beta, _f6);
                        _f7 = _mm256_comp_fmadd_ps(_c7, _beta, _f7);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm256_set1_ps(pC[0] * beta);
                    __m256 _c1 = _mm256_set1_ps(pC[1] * beta);
                    __m256 _c2 = _mm256_set1_ps(pC[2] * beta);
                    __m256 _c3 = _mm256_set1_ps(pC[3] * beta);

                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c1);
                    _f2 = _mm256_add_ps(_f2, _c2);
                    _f3 = _mm256_add_ps(_f3, _c3);

                    _c0 = _mm256_set1_ps(pC[4] * beta);
                    _c1 = _mm256_set1_ps(pC[5] * beta);
                    _c2 = _mm256_set1_ps(pC[6] * beta);
                    _c3 = _mm256_set1_ps(pC[7] * beta);

                    _f4 = _mm256_add_ps(_f4, _c0);
                    _f5 = _mm256_add_ps(_f5, _c1);
                    _f6 = _mm256_add_ps(_f6, _c2);
                    _f7 = _mm256_add_ps(_f7, _c3);
                    pC += 8;
                }
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

            __m128i _bf0 = float2bfloat_avx(_f0);
            __m128i _bf1 = float2bfloat_avx(_f1);
            __m128i _bf2 = float2bfloat_avx(_f2);
            __m128i _bf3 = float2bfloat_avx(_f3);
            __m128i _bf4 = float2bfloat_avx(_f4);
            __m128i _bf5 = float2bfloat_avx(_f5);
            __m128i _bf6 = float2bfloat_avx(_f6);
            __m128i _bf7 = float2bfloat_avx(_f7);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    transpose8x8_epi16(_bf0, _bf1, _bf2, _bf3, _bf4, _bf5, _bf6, _bf7);
                    // After transpose: _bf_k = [jj0..jj7 for ii=k]
                    _mm_storeu_si128((__m128i*)p0, _bf0);
                    _mm_storeu_si128((__m128i*)(p0 + 8), _bf1);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 2), _bf2);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 3), _bf3);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 4), _bf4);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 5), _bf5);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 6), _bf6);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 7), _bf7);
                }
                if (out_elempack == 4)
                {
                    transpose8x4_epi16(_bf0, _bf1, _bf2, _bf3);
                    transpose8x4_epi16(_bf4, _bf5, _bf6, _bf7);
                    // After transpose: _bf0=[jj0..3 for ii0 | jj0..3 for ii1]
                    // jj0..3 row
                    _mm_storel_epi64((__m128i*)p0, _bf0);
                    _mm_storeh_pd((double*)(p0 + 4), _mm_castsi128_pd(_bf0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 2), _bf1);
                    _mm_storeh_pd((double*)(p0 + 4 * 3), _mm_castsi128_pd(_bf1));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 4), _bf2);
                    _mm_storeh_pd((double*)(p0 + 4 * 5), _mm_castsi128_pd(_bf2));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 6), _bf3);
                    _mm_storeh_pd((double*)(p0 + 4 * 7), _mm_castsi128_pd(_bf3));
                    // jj4..7 row
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4), _bf4);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4), _mm_castsi128_pd(_bf4));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 4 * 2), _bf5);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 3), _mm_castsi128_pd(_bf5));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 4 * 4), _bf6);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 5), _mm_castsi128_pd(_bf6));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 4 * 6), _bf7);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 7), _mm_castsi128_pd(_bf7));
                }
                if (out_elempack == 1)
                {
                    _mm_storeu_si128((__m128i*)p0, _bf0);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep), _bf1);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 2), _bf2);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 3), _bf3);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 4), _bf4);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 5), _bf5);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 6), _bf6);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 7), _bf7);
                }
                p0 += out_hstep * 8;
            }
            else
            {
                if (out_elempack == 8)
                {
                    // _bf_k = __m128i with 8 bf16 = [ii0..ii7] for jj=k, already packed
                    _mm_storeu_si128((__m128i*)p0, _bf0);
                    _mm_storeu_si128((__m128i*)(p0 + 8), _bf1);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 2), _bf2);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 3), _bf3);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 4), _bf4);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 5), _bf5);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 6), _bf6);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 7), _bf7);
                    p0 += 64;
                }
                if (out_elempack == 4)
                {
                    // low64=[ii0..3] for jj=k, high64=[ii4..7] for jj=k
                    _mm_storel_epi64((__m128i*)p0, _bf0);
                    _mm_storel_epi64((__m128i*)(p0 + 4), _bf1);
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 2), _bf2);
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 3), _bf3);
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 4), _bf4);
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 5), _bf5);
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 6), _bf6);
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 7), _bf7);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4), _mm_castsi128_pd(_bf0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4), _mm_castsi128_pd(_bf1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 2), _mm_castsi128_pd(_bf2));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 3), _mm_castsi128_pd(_bf3));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 4), _mm_castsi128_pd(_bf4));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 5), _mm_castsi128_pd(_bf5));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 6), _mm_castsi128_pd(_bf6));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 7), _mm_castsi128_pd(_bf7));
                    p0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose8x8_epi16(_bf0, _bf1, _bf2, _bf3, _bf4, _bf5, _bf6, _bf7);

                    _mm_storeu_si128((__m128i*)p0, _bf0);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep), _bf1);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 2), _bf2);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 3), _bf3);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 4), _bf4);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 5), _bf5);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 6), _bf6);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 7), _bf7);
                    p0 += 8;
                }
            }
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m256 _f0 = _mm256_load_ps(pp);
            __m256 _f1 = _mm256_load_ps(pp + 8);
            __m256 _f2 = _mm256_load_ps(pp + 16);
            __m256 _f3 = _mm256_load_ps(pp + 24);
            pp += 32;

            // deshuffle from the shuffle-based 8x4 dpbf16_ps kernel
            {
                _f1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm256_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp0 = _mm256_unpacklo_ps(_f0, _f3);
                __m256 _tmp1 = _mm256_unpackhi_ps(_f0, _f3);
                __m256 _tmp2 = _mm256_unpacklo_ps(_f2, _f1);
                __m256 _tmp3 = _mm256_unpackhi_ps(_f2, _f1);
                _f0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp2)));
                _f1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp2)));
                _f2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp3), _mm256_castps_pd(_tmp1)));
                _f3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp3), _mm256_castps_pd(_tmp1)));
                _f1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm256_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
            }

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c0);
                    _f2 = _mm256_add_ps(_f2, _c0);
                    _f3 = _mm256_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c0);
                    _f2 = _mm256_add_ps(_f2, _c0);
                    _f3 = _mm256_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c1;
                    __m256 _c2;
                    __m256 _c3;
                    if (c_elempack == 8)
                    {
                        _c0 = _mm256_loadu_ps(pC);
                        _c1 = _mm256_loadu_ps(pC + 8);
                        _c2 = _mm256_loadu_ps(pC + 16);
                        _c3 = _mm256_loadu_ps(pC + 24);
                        pC += 32;
                    }
                    else if (c_elempack == 4)
                    {
                        __m256 _cc0 = _mm256_loadu_ps(pC);
                        __m256 _cc1 = _mm256_loadu_ps(pC + 8);
                        __m256 _cc2 = _mm256_loadu_ps(pC + c_hstep * 4);
                        __m256 _cc3 = _mm256_loadu_ps(pC + c_hstep * 4 + 8);
                        _c0 = _mm256_permute2f128_ps(_cc0, _cc2, _MM_SHUFFLE(0, 2, 0, 0));
                        _c1 = _mm256_permute2f128_ps(_cc0, _cc2, _MM_SHUFFLE(0, 3, 0, 1));
                        _c2 = _mm256_permute2f128_ps(_cc1, _cc3, _MM_SHUFFLE(0, 2, 0, 0));
                        _c3 = _mm256_permute2f128_ps(_cc1, _cc3, _MM_SHUFFLE(0, 3, 0, 1));
                        pC += 16;
                    }
                    else // if (c_elempack == 1)
                    {
                        // __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                        // _c0 = _mm256_i32gather_ps(pC, _vindex, c_hstep * sizeof(float));
                        // _c1 = _mm256_i32gather_ps(pC + 1, _vindex, c_hstep * sizeof(float));
                        // _c2 = _mm256_i32gather_ps(pC + 2, _vindex, c_hstep * sizeof(float));
                        // _c3 = _mm256_i32gather_ps(pC + 3, _vindex, c_hstep * sizeof(float));

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

                        _c0 = combine4x2_ps(_cc0, _cc4);
                        _c1 = combine4x2_ps(_cc1, _cc5);
                        _c2 = combine4x2_ps(_cc2, _cc6);
                        _c3 = combine4x2_ps(_cc3, _cc7);

                        pC += 4;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm256_add_ps(_f0, _c0);
                        _f1 = _mm256_add_ps(_f1, _c1);
                        _f2 = _mm256_add_ps(_f2, _c2);
                        _f3 = _mm256_add_ps(_f3, _c3);
                    }
                    else
                    {
                        __m256 _beta = _mm256_set1_ps(beta);
                        _f0 = _mm256_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm256_comp_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm256_comp_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm256_comp_fmadd_ps(_c3, _beta, _f3);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm256_set1_ps(pC[0] * beta);
                    __m256 _c1 = _mm256_set1_ps(pC[1] * beta);
                    __m256 _c2 = _mm256_set1_ps(pC[2] * beta);
                    __m256 _c3 = _mm256_set1_ps(pC[3] * beta);

                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c1);
                    _f2 = _mm256_add_ps(_f2, _c2);
                    _f3 = _mm256_add_ps(_f3, _c3);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                __m256 _alpha = _mm256_set1_ps(alpha);
                _f0 = _mm256_mul_ps(_f0, _alpha);
                _f1 = _mm256_mul_ps(_f1, _alpha);
                _f2 = _mm256_mul_ps(_f2, _alpha);
                _f3 = _mm256_mul_ps(_f3, _alpha);
            }

            __m128i _bf0 = float2bfloat_avx(_f0);
            __m128i _bf1 = float2bfloat_avx(_f1);
            __m128i _bf2 = float2bfloat_avx(_f2);
            __m128i _bf3 = float2bfloat_avx(_f3);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    // _bf_k = [ii0..7] for jj=k, need to store 4 jj values for each ii
                    // Store directly - each _bf has 8 ii values for one jj
                    _mm_storeu_si128((__m128i*)p0, _bf0);
                    _mm_storeu_si128((__m128i*)(p0 + 8), _bf1);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 2), _bf2);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 3), _bf3);
                }
                if (out_elempack == 4)
                {
                    transpose8x4_epi16(_bf0, _bf1, _bf2, _bf3);
                    // After: _bf0=[jj0..3 for ii0 | jj0..3 for ii1]
                    _mm_storel_epi64((__m128i*)p0, _bf0);
                    _mm_storeh_pd((double*)(p0 + 4), _mm_castsi128_pd(_bf0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 2), _bf1);
                    _mm_storeh_pd((double*)(p0 + 4 * 3), _mm_castsi128_pd(_bf1));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 4), _bf2);
                    _mm_storeh_pd((double*)(p0 + 4 * 5), _mm_castsi128_pd(_bf2));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 6), _bf3);
                    _mm_storeh_pd((double*)(p0 + 4 * 7), _mm_castsi128_pd(_bf3));
                }
                if (out_elempack == 1)
                {
                    _mm_storeu_si128((__m128i*)p0, _bf0);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep), _bf1);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 2), _bf2);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 3), _bf3);
                }
                p0 += out_hstep * 4;
            }
            else
            {
                if (out_elempack == 8)
                {
                    // _bf_k = [ii0..ii7] for jj=k, already packed
                    _mm_storeu_si128((__m128i*)p0, _bf0);
                    _mm_storeu_si128((__m128i*)(p0 + 8), _bf1);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 2), _bf2);
                    _mm_storeu_si128((__m128i*)(p0 + 8 * 3), _bf3);
                    p0 += 32;
                }
                if (out_elempack == 4)
                {
                    // low64=[ii0..3] for jj=k, high64=[ii4..7] for jj=k
                    _mm_storel_epi64((__m128i*)p0, _bf0);
                    _mm_storel_epi64((__m128i*)(p0 + 4), _bf1);
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 2), _bf2);
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 3), _bf3);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4), _mm_castsi128_pd(_bf0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4), _mm_castsi128_pd(_bf1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 2), _mm_castsi128_pd(_bf2));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 3), _mm_castsi128_pd(_bf3));
                    p0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose8x4_epi16(_bf0, _bf1, _bf2, _bf3);

                    _mm_storel_epi64((__m128i*)p0, _bf0);
                    _mm_storeh_pd((double*)(p0 + out_hstep), _mm_castsi128_pd(_bf0));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 2), _bf1);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 3), _mm_castsi128_pd(_bf1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4), _bf2);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 5), _mm_castsi128_pd(_bf2));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 6), _bf3);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 7), _mm_castsi128_pd(_bf3));
                    p0 += 4;
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m256 _f0 = _mm256_load_ps(pp);
            __m256 _f1 = _mm256_load_ps(pp + 8);
            pp += 16;

            // deshuffle from the shuffle-based 8x2 dpbf16_ps kernel
            {
                __m256 _tmp0 = _mm256_shuffle_ps(_f0, _f0, _MM_SHUFFLE(3, 1, 2, 0));
                __m256 _tmp1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(0, 2, 3, 1));
                _f0 = _mm256_unpacklo_ps(_tmp0, _tmp1);
                _f1 = _mm256_unpackhi_ps(_tmp0, _tmp1);
                _f1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
            }

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c1;
                    if (c_elempack == 8)
                    {
                        _c0 = _mm256_loadu_ps(pC);
                        _c1 = _mm256_loadu_ps(pC + 8);
                        pC += 16;
                    }
                    else if (c_elempack == 4)
                    {
                        __m256 _cc0 = _mm256_loadu_ps(pC);
                        __m256 _cc1 = _mm256_loadu_ps(pC + c_hstep * 4);
                        _c0 = _mm256_permute2f128_ps(_cc0, _cc1, _MM_SHUFFLE(0, 2, 0, 0));
                        _c1 = _mm256_permute2f128_ps(_cc0, _cc1, _MM_SHUFFLE(0, 3, 0, 1));
                        pC += 8;
                    }
                    else // if (c_elempack == 1)
                    {
#if __AVX2__
                        __m256i _vindex = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(c_hstep));
                        _c0 = _mm256_i32gather_ps(pC, _vindex, sizeof(float));
                        _c1 = _mm256_i32gather_ps(pC + 1, _vindex, sizeof(float));
#else
                        _c0 = _mm256_setr_ps(pC[0], pC[c_hstep], pC[c_hstep * 2], pC[c_hstep * 3], pC[c_hstep * 4], pC[c_hstep * 5], pC[c_hstep * 6], pC[c_hstep * 7]);
                        _c1 = _mm256_setr_ps(pC[1], pC[c_hstep + 1], pC[c_hstep * 2 + 1], pC[c_hstep * 3 + 1], pC[c_hstep * 4 + 1], pC[c_hstep * 5 + 1], pC[c_hstep * 6 + 1], pC[c_hstep * 7 + 1]);
#endif
                        pC += 2;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm256_add_ps(_f0, _c0);
                        _f1 = _mm256_add_ps(_f1, _c1);
                    }
                    else
                    {
                        __m256 _beta = _mm256_set1_ps(beta);
                        _f0 = _mm256_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm256_comp_fmadd_ps(_c1, _beta, _f1);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm256_set1_ps(pC[0] * beta);
                    __m256 _c1 = _mm256_set1_ps(pC[1] * beta);
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c1);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                __m256 _alpha = _mm256_set1_ps(alpha);
                _f0 = _mm256_mul_ps(_f0, _alpha);
                _f1 = _mm256_mul_ps(_f1, _alpha);
            }

            __m128i _bf0 = float2bfloat_avx(_f0);
            __m128i _bf1 = float2bfloat_avx(_f1);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    // _bf0=[ii0..7] for jj0, _bf1=[ii0..7] for jj1
                    // Need [jj0,jj1] for each ii group of 8 - but only 2 jj, store as-is
                    _mm_storeu_si128((__m128i*)p0, _bf0);
                    _mm_storeu_si128((__m128i*)(p0 + 8), _bf1);
                }
                if (out_elempack == 4)
                {
                    // _bf0=[ii0..7] for jj0: low64=[ii0..3], high64=[ii4..7]
                    _mm_storel_epi64((__m128i*)p0, _bf0);
                    _mm_storel_epi64((__m128i*)(p0 + 4), _bf1);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4), _mm_castsi128_pd(_bf0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4), _mm_castsi128_pd(_bf1));
                }
                if (out_elempack == 1)
                {
                    _mm_storeu_si128((__m128i*)p0, _bf0);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep), _bf1);
                }
                p0 += out_hstep * 2;
            }
            else
            {
                if (out_elempack == 8)
                {
                    _mm_storeu_si128((__m128i*)p0, _bf0);
                    _mm_storeu_si128((__m128i*)(p0 + 8), _bf1);
                    p0 += 16;
                }
                if (out_elempack == 4)
                {
                    _mm_storel_epi64((__m128i*)p0, _bf0);
                    _mm_storel_epi64((__m128i*)(p0 + 4), _bf1);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4), _mm_castsi128_pd(_bf0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4), _mm_castsi128_pd(_bf1));
                    p0 += 8;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[8];
                    unsigned short sum1[8];
                    _mm_storeu_si128((__m128i*)sum0, _bf0);
                    _mm_storeu_si128((__m128i*)sum1, _bf1);

                    p0[0] = sum0[0];
                    p0[1] = sum1[0];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep + 1] = sum1[1];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 2 + 1] = sum1[2];
                    p0[out_hstep * 3] = sum0[3];
                    p0[out_hstep * 3 + 1] = sum1[3];
                    p0[out_hstep * 4] = sum0[4];
                    p0[out_hstep * 4 + 1] = sum1[4];
                    p0[out_hstep * 5] = sum0[5];
                    p0[out_hstep * 5 + 1] = sum1[5];
                    p0[out_hstep * 6] = sum0[6];
                    p0[out_hstep * 6 + 1] = sum1[6];
                    p0[out_hstep * 7] = sum0[7];
                    p0[out_hstep * 7 + 1] = sum1[7];
                    p0 += 2;
                }
            }
        }
        for (; jj < max_jj; jj++)
        {
            __m256 _f = _mm256_load_ps(pp);
            pp += 8;

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f = _mm256_add_ps(_f, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f = _mm256_add_ps(_f, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 8)
                    {
                        _c0 = _mm256_loadu_ps(pC);
                        pC += 8;
                    }
                    else if (c_elempack == 4)
                    {
                        __m128 _cc0 = _mm_loadu_ps(pC);
                        __m128 _cc1 = _mm_loadu_ps(pC + c_hstep * 4);
                        _c0 = combine4x2_ps(_cc0, _cc1);
                        pC += 4;
                    }
                    else // if (c_elempack == 1)
                    {
#if __AVX2__
                        __m256i _vindex = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(c_hstep));
                        _c0 = _mm256_i32gather_ps(pC, _vindex, sizeof(float));
#else
                        _c0 = _mm256_setr_ps(pC[0], pC[c_hstep], pC[c_hstep * 2], pC[c_hstep * 3], pC[c_hstep * 4], pC[c_hstep * 5], pC[c_hstep * 6], pC[c_hstep * 7]);
#endif
                        pC += 1;
                    }
                    _f = _mm256_comp_fmadd_ps(_c0, _mm256_set1_ps(beta), _f);
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm256_set1_ps(pC[0] * beta);
                    _f = _mm256_add_ps(_f, _c0);
                    pC += 1;
                }
            }

            if (alpha != 1.f)
            {
                _f = _mm256_mul_ps(_f, _mm256_set1_ps(alpha));
            }

            __m128i _bf = float2bfloat_avx(_f);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    _mm_storeu_si128((__m128i*)p0, _bf);
                }
                if (out_elempack == 4)
                {
                    _mm_storel_epi64((__m128i*)p0, _bf);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4), _mm_castsi128_pd(_bf));
                }
                if (out_elempack == 1)
                {
                    _mm_storeu_si128((__m128i*)p0, _bf);
                }
                p0 += out_hstep;
            }
            else
            {
                if (out_elempack == 8)
                {
                    _mm_storeu_si128((__m128i*)p0, _bf);
                    p0 += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_storel_epi64((__m128i*)p0, _bf);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4), _mm_castsi128_pd(_bf));
                    p0 += 4;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[8];
                    _mm_storeu_si128((__m128i*)sum0, _bf);

                    p0[0] = sum0[0];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 3] = sum0[3];
                    p0[out_hstep * 4] = sum0[4];
                    p0[out_hstep * 5] = sum0[5];
                    p0[out_hstep * 6] = sum0[6];
                    p0[out_hstep * 7] = sum0[7];
                    p0++;
                }
            }
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

        __m128 _c0 = _mm_set1_ps(0.f);
#if __AVX512F__
        __m512 _c0_avx512 = _mm512_set1_ps(0.f);
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = _mm_set1_ps(pC[0] * beta);
#if __AVX512F__
                _c0_avx512 = _mm512_set1_ps(pC[0] * beta);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = _mm_loadu_ps(pC);
                _c0 = _mm_mul_ps(_c0, _mm_set1_ps(beta));
#if __AVX512F__
                _c0_avx512 = _mm512_broadcast_f32x4(_c0);
#endif
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _f0 = _mm512_loadu_ps(pp);
            __m512 _f1 = _mm512_loadu_ps(pp + 16);
            __m512 _f2 = _mm512_loadu_ps(pp + 32);
            __m512 _f3 = _mm512_loadu_ps(pp + 48);
            pp += 64;

            // from
            //      00 11 22 33 04 15 26 37 08 19 2a 3b 0c 1d 2e 3f
            //      01 12 23 30 05 16 27 34 09 1a 2b 38 0d 1e 2f 3c
            //      20 31 02 13 24 35 06 17 28 39 0a 1b 2c 3d 0e 1f
            //      21 32 03 10 25 36 07 14 29 3a 0b 18 2d 3e 0f 1c
            // to
            //      00 10 20 30 04 14 24 34 08 18 28 38 0c 1c 2c 3c
            //      01 11 21 31 05 15 25 35 09 19 29 39 0d 1d 2d 3d
            //      02 12 22 32 06 16 26 36 0a 1a 2a 3a 0e 1e 2e 3e
            //      03 13 23 33 07 17 27 37 0b 1b 2b 3b 0f 1f 2f 3f
            {
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f3);
                __m512 _tmp1 = _mm512_unpacklo_ps(_f2, _f1);
                __m512 _tmp2 = _mm512_unpackhi_ps(_f0, _f3);
                __m512 _tmp3 = _mm512_unpackhi_ps(_f2, _f1);
                _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                _f2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp2)));
                _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp2)));
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
            }

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c0_avx512);
                    _f2 = _mm512_add_ps(_f2, _c0_avx512);
                    _f3 = _mm512_add_ps(_f3, _c0_avx512);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c0_avx512);
                    _f2 = _mm512_add_ps(_f2, _c0_avx512);
                    _f3 = _mm512_add_ps(_f3, _c0_avx512);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c1_avx512;
                    __m512 _c2_avx512;
                    __m512 _c3_avx512;
                    if (c_elempack == 4)
                    {
                        _c0_avx512 = _mm512_loadu_ps(pC);
                        _c1_avx512 = _mm512_loadu_ps(pC + 16);
                        _c2_avx512 = _mm512_loadu_ps(pC + 32);
                        _c3_avx512 = _mm512_loadu_ps(pC + 48);
                        pC += 64;

                        __m512 _tmp0 = _mm512_shuffle_f32x4(_c0_avx512, _c1_avx512, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_c2_avx512, _c3_avx512, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp2 = _mm512_shuffle_f32x4(_c0_avx512, _c1_avx512, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp3 = _mm512_shuffle_f32x4(_c2_avx512, _c3_avx512, _MM_SHUFFLE(3, 2, 3, 2));
                        _c0_avx512 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _c1_avx512 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        _c2_avx512 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                        _c3_avx512 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0_avx512 = _mm512_loadu_ps(pC);
                        _c1_avx512 = _mm512_loadu_ps(pC + c_hstep);
                        _c2_avx512 = _mm512_loadu_ps(pC + c_hstep * 2);
                        _c3_avx512 = _mm512_loadu_ps(pC + c_hstep * 3);
                        pC += 16;

                        __m512 _tmp0 = _mm512_unpacklo_ps(_c0_avx512, _c1_avx512);
                        __m512 _tmp1 = _mm512_unpacklo_ps(_c2_avx512, _c3_avx512);
                        __m512 _tmp2 = _mm512_unpackhi_ps(_c0_avx512, _c1_avx512);
                        __m512 _tmp3 = _mm512_unpackhi_ps(_c2_avx512, _c3_avx512);
                        _c0_avx512 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                        _c1_avx512 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                        _c2_avx512 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                        _c3_avx512 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0_avx512);
                        _f1 = _mm512_add_ps(_f1, _c1_avx512);
                        _f2 = _mm512_add_ps(_f2, _c2_avx512);
                        _f3 = _mm512_add_ps(_f3, _c3_avx512);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0_avx512, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1_avx512, _beta, _f1);
                        _f2 = _mm512_fmadd_ps(_c2_avx512, _beta, _f2);
                        _f3 = _mm512_fmadd_ps(_c3_avx512, _beta, _f3);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    __m512 _cc = _mm512_loadu_ps(pC);
                    _cc = _mm512_mul_ps(_cc, _mm512_set1_ps(beta));
                    _c0_avx512 = _mm512_permute_ps(_cc, _MM_SHUFFLE(0, 0, 0, 0));
                    __m512 _c1_avx512 = _mm512_permute_ps(_cc, _MM_SHUFFLE(1, 1, 1, 1));
                    __m512 _c2_avx512 = _mm512_permute_ps(_cc, _MM_SHUFFLE(2, 2, 2, 2));
                    __m512 _c3_avx512 = _mm512_permute_ps(_cc, _MM_SHUFFLE(3, 3, 3, 3));

                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c1_avx512);
                    _f2 = _mm512_add_ps(_f2, _c2_avx512);
                    _f3 = _mm512_add_ps(_f3, _c3_avx512);

                    pC += 16;
                }
            }

            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
                _f2 = _mm512_mul_ps(_f2, _alpha);
                _f3 = _mm512_mul_ps(_f3, _alpha);
            }

            __m256i _bf0 = float2bfloat_avx512(_f0);
            __m256i _bf1 = float2bfloat_avx512(_f1);
            __m256i _bf2 = float2bfloat_avx512(_f2);
            __m256i _bf3 = float2bfloat_avx512(_f3);

            if (output_transpose)
            {
                if (out_elempack == 16)
                {
                    transpose16x4_epi16(_bf0, _bf1, _bf2, _bf3);
                    _mm_storel_epi64((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 8), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 12), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storeh_pd((double*)(p0 + 16), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 0)));
                    _mm_storeh_pd((double*)(p0 + 16 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 0)));
                    _mm_storeh_pd((double*)(p0 + 16 + 8), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 0)));
                    _mm_storeh_pd((double*)(p0 + 16 + 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + 32), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storel_epi64((__m128i*)(p0 + 32 + 4), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storel_epi64((__m128i*)(p0 + 32 + 8), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storel_epi64((__m128i*)(p0 + 32 + 12), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storeh_pd((double*)(p0 + 48), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 1)));
                    _mm_storeh_pd((double*)(p0 + 48 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 1)));
                    _mm_storeh_pd((double*)(p0 + 48 + 8), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 1)));
                    _mm_storeh_pd((double*)(p0 + 48 + 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 1)));
                }
                if (out_elempack == 8)
                {
                    transpose16x4_epi16(_bf0, _bf1, _bf2, _bf3);
                    _mm_storel_epi64((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storeh_pd((double*)(p0 + 8), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 0)));
                    _mm_storeh_pd((double*)(p0 + 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + 16), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storel_epi64((__m128i*)(p0 + 16 + 4), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storeh_pd((double*)(p0 + 16 + 8), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 1)));
                    _mm_storeh_pd((double*)(p0 + 16 + 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 1)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 8 + 8), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 8 + 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 16), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 16 + 4), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 8 + 16 + 8), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 8 + 16 + 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 1)));
                }
                if (out_elempack == 4)
                {
                    // _bf_k layout: low128.low64=[ii0..3 jj=k], low128.high64=[ii0..3 jj=k+4],
                    //               high128.low64=[ii0..3 jj=k+8], high128.high64=[ii0..3 jj=k+12]
                    // 4 _bf regs: _bf0=jj{0,4,8,12}, _bf1=jj{1,5,9,13}, _bf2=jj{2,6,10,14}, _bf3=jj{3,7,11,15}
                    // Need transpose to [jj0..3 for each ii], each stored at ii*4 offset
                    // Use unpack on the low halves: pairs from _bf0/_bf1 and _bf2/_bf3
                    __m128i _bf0l = _mm256_extractf128_si256(_bf0, 0);
                    __m128i _bf1l = _mm256_extractf128_si256(_bf1, 0);
                    __m128i _bf2l = _mm256_extractf128_si256(_bf2, 0);
                    __m128i _bf3l = _mm256_extractf128_si256(_bf3, 0);
                    // _bf0l = [ii0_j0 ii1_j0 ii2_j0 ii3_j0 | ii0_j4 ii1_j4 ii2_j4 ii3_j4]
                    // _bf1l = [ii0_j1 ii1_j1 ii2_j1 ii3_j1 | ii0_j5 ii1_j5 ii2_j5 ii3_j5]
                    __m128i _t0 = _mm_unpacklo_epi16(_bf0l, _bf1l);  // [ii0_j0 ii0_j1 ii1_j0 ii1_j1 ii2_j0 ii2_j1 ii3_j0 ii3_j1]
                    __m128i _t1 = _mm_unpacklo_epi16(_bf2l, _bf3l);  // [ii0_j2 ii0_j3 ii1_j2 ii1_j3 ii2_j2 ii2_j3 ii3_j2 ii3_j3]
                    __m128i _d0 = _mm_unpacklo_epi32(_t0, _t1);      // [ii0_j0 ii0_j1 ii0_j2 ii0_j3 ii1_j0 ii1_j1 ii1_j2 ii1_j3]
                    __m128i _d1 = _mm_unpackhi_epi32(_t0, _t1);      // [ii2_j0..j3 ii3_j0..j3]
                    _mm_storel_epi64((__m128i*)p0, _d0);
                    _mm_storeh_pd((double*)(p0 + 4), _mm_castsi128_pd(_d0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 2), _d1);
                    _mm_storeh_pd((double*)(p0 + 4 * 3), _mm_castsi128_pd(_d1));
                    // jj4..7
                    __m128i _t2 = _mm_unpackhi_epi16(_bf0l, _bf1l);
                    __m128i _t3 = _mm_unpackhi_epi16(_bf2l, _bf3l);
                    __m128i _d2 = _mm_unpacklo_epi32(_t2, _t3);
                    __m128i _d3 = _mm_unpackhi_epi32(_t2, _t3);
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4), _d2);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4), _mm_castsi128_pd(_d2));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 4 * 2), _d3);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 3), _mm_castsi128_pd(_d3));
                    // jj8..15 from high halves
                    __m128i _bf0h = _mm256_extractf128_si256(_bf0, 1);
                    __m128i _bf1h = _mm256_extractf128_si256(_bf1, 1);
                    __m128i _bf2h = _mm256_extractf128_si256(_bf2, 1);
                    __m128i _bf3h = _mm256_extractf128_si256(_bf3, 1);
                    __m128i _t4 = _mm_unpacklo_epi16(_bf0h, _bf1h);
                    __m128i _t5 = _mm_unpacklo_epi16(_bf2h, _bf3h);
                    __m128i _d4 = _mm_unpacklo_epi32(_t4, _t5);
                    __m128i _d5 = _mm_unpackhi_epi32(_t4, _t5);
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8), _d4);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 8 + 4), _mm_castsi128_pd(_d4));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4 * 2), _d5);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 8 + 4 * 3), _mm_castsi128_pd(_d5));
                    __m128i _t6 = _mm_unpackhi_epi16(_bf0h, _bf1h);
                    __m128i _t7 = _mm_unpackhi_epi16(_bf2h, _bf3h);
                    __m128i _d6 = _mm_unpacklo_epi32(_t6, _t7);
                    __m128i _d7 = _mm_unpackhi_epi32(_t6, _t7);
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 12), _d6);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4), _mm_castsi128_pd(_d6));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 12 + 4 * 2), _d7);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4 * 3), _mm_castsi128_pd(_d7));
                }
                if (out_elempack == 1)
                {
                    _mm_storel_epi64((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 2), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 3), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 5), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 6), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 7), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 9), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 10), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 11), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 13), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 14), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 15), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 1)));
                }
                p0 += out_hstep * 16;
            }
            else
            {
                if (out_elempack == 4)
                {
                    // _bf_k layout: low128.low64=[ii0..3] for jj=k, low128.high64=[ii0..3] for jj=k+4
                    //               high128.low64=[ii0..3] for jj=k+8, high128.high64=[ii0..3] for jj=k+12
                    // _bf0 has jj{0,4,8,12}, _bf1 has jj{1,5,9,13}, _bf2 has jj{2,6,10,14}, _bf3 has jj{3,7,11,15}
                    // Store 4 bf16 per jj column, interleaving _bf0..3 for sequential jj
                    _mm_storel_epi64((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));           // jj0
                    _mm_storel_epi64((__m128i*)(p0 + 4), _mm256_extractf128_si256(_bf1, 0));     // jj1
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 2), _mm256_extractf128_si256(_bf2, 0)); // jj2
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 3), _mm256_extractf128_si256(_bf3, 0)); // jj3
                    _mm_storeh_pd((double*)(p0 + 4 * 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 0)));  // jj4
                    _mm_storeh_pd((double*)(p0 + 4 * 5), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 0)));  // jj5
                    _mm_storeh_pd((double*)(p0 + 4 * 6), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 0)));  // jj6
                    _mm_storeh_pd((double*)(p0 + 4 * 7), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 0)));  // jj7
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 8), _mm256_extractf128_si256(_bf0, 1));                // jj8
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 9), _mm256_extractf128_si256(_bf1, 1));                // jj9
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 10), _mm256_extractf128_si256(_bf2, 1));               // jj10
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 11), _mm256_extractf128_si256(_bf3, 1));               // jj11
                    _mm_storeh_pd((double*)(p0 + 4 * 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 1))); // jj12
                    _mm_storeh_pd((double*)(p0 + 4 * 13), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 1))); // jj13
                    _mm_storeh_pd((double*)(p0 + 4 * 14), _mm_castsi128_pd(_mm256_extractf128_si256(_bf2, 1))); // jj14
                    _mm_storeh_pd((double*)(p0 + 4 * 15), _mm_castsi128_pd(_mm256_extractf128_si256(_bf3, 1))); // jj15
                    p0 += 64;
                }
                if (out_elempack == 1)
                {
                    __m512i _idx_r0r1 = _mm512_set_epi16(61, 45, 29, 13, 57, 41, 25, 9, 53, 37, 21, 5, 49, 33, 17, 1, 60, 44, 28, 12, 56, 40, 24, 8, 52, 36, 20, 4, 48, 32, 16, 0);
                    __m512i _idx_r2r3 = _mm512_set_epi16(63, 47, 31, 15, 59, 43, 27, 11, 55, 39, 23, 7, 51, 35, 19, 3, 62, 46, 30, 14, 58, 42, 26, 10, 54, 38, 22, 6, 50, 34, 18, 2);

                    __m512i _bf01 = combine8x2_epi32(_bf0, _bf1);
                    __m512i _bf23 = combine8x2_epi32(_bf2, _bf3);

                    __m512i _t01 = _mm512_permutex2var_epi16(_bf01, _idx_r0r1, _bf23);
                    __m512i _t23 = _mm512_permutex2var_epi16(_bf01, _idx_r2r3, _bf23);

                    _mm256_storeu_si256((__m256i*)p0, _mm512_extracti32x8_epi32(_t01, 0));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep), _mm512_extracti32x8_epi32(_t01, 1));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 2), _mm512_extracti32x8_epi32(_t23, 0));
                    _mm256_storeu_si256((__m256i*)(p0 + out_hstep * 3), _mm512_extracti32x8_epi32(_t23, 1));
                    p0 += 16;
                }
            }
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f0 = _mm_load_ps(pp);
            __m128 _f1 = _mm_load_ps(pp + 4);
            __m128 _f2 = _mm_load_ps(pp + 8);
            __m128 _f3 = _mm_load_ps(pp + 12);
            __m128 _f4 = _mm_load_ps(pp + 16);
            __m128 _f5 = _mm_load_ps(pp + 20);
            __m128 _f6 = _mm_load_ps(pp + 24);
            __m128 _f7 = _mm_load_ps(pp + 28);
            pp += 32;

            // from
            //      00 11 22 33
            //      04 15 26 37
            //      20 31 02 13
            //      24 35 06 17
            //      01 12 23 30
            //      05 16 27 34
            //      21 32 03 10
            //      25 36 07 14
            // to
            //      00 10 20 30
            //      01 11 21 31
            //      02 12 22 32
            //      03 13 23 33
            //      04 14 24 34
            //      05 15 25 35
            //      06 16 26 36
            //      07 17 27 37
            {
                _f4 = _mm_shuffle_ps(_f4, _f4, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm_shuffle_ps(_f5, _f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f6 = _mm_shuffle_ps(_f6, _f6, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm_shuffle_ps(_f7, _f7, _MM_SHUFFLE(2, 1, 0, 3));
                __m128 _tmp0 = _mm_unpacklo_ps(_f0, _f6);
                __m128 _tmp1 = _mm_unpackhi_ps(_f0, _f6);
                __m128 _tmp2 = _mm_unpacklo_ps(_f1, _f7);
                __m128 _tmp3 = _mm_unpackhi_ps(_f1, _f7);
                __m128 _tmp4 = _mm_unpacklo_ps(_f2, _f4);
                __m128 _tmp5 = _mm_unpackhi_ps(_f2, _f4);
                __m128 _tmp6 = _mm_unpacklo_ps(_f3, _f5);
                __m128 _tmp7 = _mm_unpackhi_ps(_f3, _f5);
                _f0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp4)));
                _f1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp4)));
                _f2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp5), _mm_castps_pd(_tmp1)));
                _f3 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp5), _mm_castps_pd(_tmp1)));
                _f4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp2), _mm_castps_pd(_tmp6)));
                _f5 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp2), _mm_castps_pd(_tmp6)));
                _f6 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp7), _mm_castps_pd(_tmp3)));
                _f7 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp7), _mm_castps_pd(_tmp3)));
                _f1 = _mm_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm_shuffle_ps(_f5, _f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm_shuffle_ps(_f7, _f7, _MM_SHUFFLE(2, 1, 0, 3));
            }

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    _f2 = _mm_add_ps(_f2, _c0);
                    _f3 = _mm_add_ps(_f3, _c0);
                    _f4 = _mm_add_ps(_f4, _c0);
                    _f5 = _mm_add_ps(_f5, _c0);
                    _f6 = _mm_add_ps(_f6, _c0);
                    _f7 = _mm_add_ps(_f7, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    _f2 = _mm_add_ps(_f2, _c0);
                    _f3 = _mm_add_ps(_f3, _c0);
                    _f4 = _mm_add_ps(_f4, _c0);
                    _f5 = _mm_add_ps(_f5, _c0);
                    _f6 = _mm_add_ps(_f6, _c0);
                    _f7 = _mm_add_ps(_f7, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c1;
                    __m128 _c2;
                    __m128 _c3;
                    if (c_elempack == 4)
                    {
                        _c0 = _mm_loadu_ps(pC);
                        _c1 = _mm_loadu_ps(pC + 4);
                        _c2 = _mm_loadu_ps(pC + 8);
                        _c3 = _mm_loadu_ps(pC + 12);
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = _mm_loadu_ps(pC);
                        _c1 = _mm_loadu_ps(pC + c_hstep);
                        _c2 = _mm_loadu_ps(pC + c_hstep * 2);
                        _c3 = _mm_loadu_ps(pC + c_hstep * 3);
                        _MM_TRANSPOSE4_PS(_c0, _c1, _c2, _c3);
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                        _f2 = _mm_add_ps(_f2, _c2);
                        _f3 = _mm_add_ps(_f3, _c3);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
                        _f0 = _mm_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_comp_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm_comp_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm_comp_fmadd_ps(_c3, _beta, _f3);
                    }
                    if (c_elempack == 4)
                    {
                        _c0 = _mm_loadu_ps(pC + 16);
                        _c1 = _mm_loadu_ps(pC + 20);
                        _c2 = _mm_loadu_ps(pC + 24);
                        _c3 = _mm_loadu_ps(pC + 28);
                        pC += 32;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = _mm_loadu_ps(pC + 4);
                        _c1 = _mm_loadu_ps(pC + c_hstep + 4);
                        _c2 = _mm_loadu_ps(pC + c_hstep * 2 + 4);
                        _c3 = _mm_loadu_ps(pC + c_hstep * 3 + 4);
                        _MM_TRANSPOSE4_PS(_c0, _c1, _c2, _c3);
                        pC += 8;
                    }
                    if (beta == 1.f)
                    {
                        _f4 = _mm_add_ps(_f4, _c0);
                        _f5 = _mm_add_ps(_f5, _c1);
                        _f6 = _mm_add_ps(_f6, _c2);
                        _f7 = _mm_add_ps(_f7, _c3);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
                        _f4 = _mm_comp_fmadd_ps(_c0, _beta, _f4);
                        _f5 = _mm_comp_fmadd_ps(_c1, _beta, _f5);
                        _f6 = _mm_comp_fmadd_ps(_c2, _beta, _f6);
                        _f7 = _mm_comp_fmadd_ps(_c3, _beta, _f7);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm_set1_ps(pC[0] * beta);
                    __m128 _c1 = _mm_set1_ps(pC[1] * beta);
                    __m128 _c2 = _mm_set1_ps(pC[2] * beta);
                    __m128 _c3 = _mm_set1_ps(pC[3] * beta);

                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c1);
                    _f2 = _mm_add_ps(_f2, _c2);
                    _f3 = _mm_add_ps(_f3, _c3);

                    _c0 = _mm_set1_ps(pC[4] * beta);
                    _c1 = _mm_set1_ps(pC[5] * beta);
                    _c2 = _mm_set1_ps(pC[6] * beta);
                    _c3 = _mm_set1_ps(pC[7] * beta);

                    _f4 = _mm_add_ps(_f4, _c0);
                    _f5 = _mm_add_ps(_f5, _c1);
                    _f6 = _mm_add_ps(_f6, _c2);
                    _f7 = _mm_add_ps(_f7, _c3);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
                _f2 = _mm_mul_ps(_f2, _alpha);
                _f3 = _mm_mul_ps(_f3, _alpha);
                _f4 = _mm_mul_ps(_f4, _alpha);
                _f5 = _mm_mul_ps(_f5, _alpha);
                _f6 = _mm_mul_ps(_f6, _alpha);
                _f7 = _mm_mul_ps(_f7, _alpha);
            }

            __m128i _bf04 = float2bfloat_sse(_f0, _f4);
            __m128i _bf15 = float2bfloat_sse(_f1, _f5);
            __m128i _bf26 = float2bfloat_sse(_f2, _f6);
            __m128i _bf37 = float2bfloat_sse(_f3, _f7);

            if (output_transpose)
            {
#if __AVX__
                if (out_elempack == 8)
                {
                    __m128i _t0 = _mm_unpacklo_epi16(_bf04, _bf15);
                    __m128i _t1 = _mm_unpacklo_epi16(_bf26, _bf37);
                    __m128i _t2 = _mm_unpackhi_epi16(_bf04, _bf15);
                    __m128i _t3 = _mm_unpackhi_epi16(_bf26, _bf37);
                    _bf04 = _mm_unpacklo_epi32(_t0, _t1);
                    _bf15 = _mm_unpacklo_epi32(_t2, _t3);
                    _bf26 = _mm_unpackhi_epi32(_t0, _t1);
                    _bf37 = _mm_unpackhi_epi32(_t2, _t3);
                    _t0 = _mm_unpacklo_epi64(_bf04, _bf15);
                    _t1 = _mm_unpackhi_epi64(_bf04, _bf15);
                    _t2 = _mm_unpacklo_epi64(_bf26, _bf37);
                    _t3 = _mm_unpackhi_epi64(_bf26, _bf37);

                    _mm_storel_epi64((__m128i*)p0, _t0);
                    _mm_storeh_pd((double*)(p0 + 4), _mm_castsi128_pd(_t0));
                    _mm_storel_epi64((__m128i*)(p0 + 8), _t1);
                    _mm_storeh_pd((double*)(p0 + 12), _mm_castsi128_pd(_t1));
                    _mm_storel_epi64((__m128i*)(p0 + 16), _t2);
                    _mm_storeh_pd((double*)(p0 + 20), _mm_castsi128_pd(_t2));
                    _mm_storel_epi64((__m128i*)(p0 + 24), _t3);
                    _mm_storeh_pd((double*)(p0 + 28), _mm_castsi128_pd(_t3));
                }
#endif // __AVX__
                if (out_elempack == 4)
                {
                    // _bf04 = [ii0..3 jj0 | ii0..3 jj4], _bf15 = [ii0..3 jj1 | ii0..3 jj5], etc.
                    // Need transpose to [jj0..3 for each ii]
                    __m128i _t0 = _mm_unpacklo_epi16(_bf04, _bf15);  // [ii0_j0 ii0_j1 ii1_j0 ii1_j1 ii2_j0 ii2_j1 ii3_j0 ii3_j1]
                    __m128i _t1 = _mm_unpacklo_epi16(_bf26, _bf37);  // [ii0_j2 ii0_j3 ii1_j2 ii1_j3 ii2_j2 ii2_j3 ii3_j2 ii3_j3]
                    __m128i _d0 = _mm_unpacklo_epi32(_t0, _t1);      // [ii0_j0..j3 ii1_j0..j3]
                    __m128i _d1 = _mm_unpackhi_epi32(_t0, _t1);      // [ii2_j0..j3 ii3_j0..j3]
                    _mm_storel_epi64((__m128i*)p0, _d0);
                    _mm_storeh_pd((double*)(p0 + 4), _mm_castsi128_pd(_d0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 2), _d1);
                    _mm_storeh_pd((double*)(p0 + 4 * 3), _mm_castsi128_pd(_d1));
                    // jj4..7
                    __m128i _t2 = _mm_unpackhi_epi16(_bf04, _bf15);
                    __m128i _t3 = _mm_unpackhi_epi16(_bf26, _bf37);
                    __m128i _d2 = _mm_unpacklo_epi32(_t2, _t3);
                    __m128i _d3 = _mm_unpackhi_epi32(_t2, _t3);
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4), _d2);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4), _mm_castsi128_pd(_d2));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4 + 4 * 2), _d3);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4 * 3), _mm_castsi128_pd(_d3));
                }
                if (out_elempack == 1)
                {
                    _mm_storel_epi64((__m128i*)p0, _bf04);
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep), _bf15);
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 2), _bf26);
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 3), _bf37);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4), _mm_castsi128_pd(_bf04));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 5), _mm_castsi128_pd(_bf15));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 6), _mm_castsi128_pd(_bf26));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 7), _mm_castsi128_pd(_bf37));
                }
                p0 += out_hstep * 8;
            }
            else
            {
                if (out_elempack == 4)
                {
                    // _bf04 low64=[ii0..3 jj0], high64=[ii0..3 jj4]
                    // Store 4 bf16 per jj, interleaving for sequential jj
                    _mm_storel_epi64((__m128i*)p0, _bf04);                                         // jj0
                    _mm_storel_epi64((__m128i*)(p0 + 4), _bf15);                                   // jj1
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 2), _bf26);                               // jj2
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 3), _bf37);                               // jj3
                    _mm_storeh_pd((double*)(p0 + 4 * 4), _mm_castsi128_pd(_bf04));                 // jj4
                    _mm_storeh_pd((double*)(p0 + 4 * 5), _mm_castsi128_pd(_bf15));                 // jj5
                    _mm_storeh_pd((double*)(p0 + 4 * 6), _mm_castsi128_pd(_bf26));                 // jj6
                    _mm_storeh_pd((double*)(p0 + 4 * 7), _mm_castsi128_pd(_bf37));                 // jj7
                    p0 += 32;
                }
                if (out_elempack == 1)
                {
                    __m128i _t0 = _mm_unpacklo_epi16(_bf04, _bf15);
                    __m128i _t1 = _mm_unpacklo_epi16(_bf26, _bf37);
                    __m128i _t2 = _mm_unpackhi_epi16(_bf04, _bf15);
                    __m128i _t3 = _mm_unpackhi_epi16(_bf26, _bf37);
                    _bf04 = _mm_unpacklo_epi32(_t0, _t1);
                    _bf15 = _mm_unpacklo_epi32(_t2, _t3);
                    _bf26 = _mm_unpackhi_epi32(_t0, _t1);
                    _bf37 = _mm_unpackhi_epi32(_t2, _t3);
                    _t0 = _mm_unpacklo_epi64(_bf04, _bf15);
                    _t1 = _mm_unpackhi_epi64(_bf04, _bf15);
                    _t2 = _mm_unpacklo_epi64(_bf26, _bf37);
                    _t3 = _mm_unpackhi_epi64(_bf26, _bf37);

                    _mm_storeu_si128((__m128i*)p0, _t0);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep), _t1);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 2), _t2);
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 3), _t3);
                    p0 += 8;
                }
            }
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = _mm_load_ps(pp);
            __m128 _f1 = _mm_load_ps(pp + 4);
            __m128 _f2 = _mm_load_ps(pp + 8);
            __m128 _f3 = _mm_load_ps(pp + 12);
            pp += 16;

            // deshuffle from the shuffle-based 4x4 dpbf16_ps kernel
            {
                _f1 = _mm_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m128 _tmp0 = _mm_unpacklo_ps(_f0, _f3);
                __m128 _tmp1 = _mm_unpackhi_ps(_f0, _f3);
                __m128 _tmp2 = _mm_unpacklo_ps(_f2, _f1);
                __m128 _tmp3 = _mm_unpackhi_ps(_f2, _f1);
                _f0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp2)));
                _f1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp2)));
                _f2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp3), _mm_castps_pd(_tmp1)));
                _f3 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp3), _mm_castps_pd(_tmp1)));
                _f1 = _mm_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
            }

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    _f2 = _mm_add_ps(_f2, _c0);
                    _f3 = _mm_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    _f2 = _mm_add_ps(_f2, _c0);
                    _f3 = _mm_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c1;
                    __m128 _c2;
                    __m128 _c3;
                    if (c_elempack == 4)
                    {
                        _c0 = _mm_loadu_ps(pC);
                        _c1 = _mm_loadu_ps(pC + 4);
                        _c2 = _mm_loadu_ps(pC + 8);
                        _c3 = _mm_loadu_ps(pC + 12);
                        pC += 16;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = _mm_loadu_ps(pC);
                        _c1 = _mm_loadu_ps(pC + c_hstep);
                        _c2 = _mm_loadu_ps(pC + c_hstep * 2);
                        _c3 = _mm_loadu_ps(pC + c_hstep * 3);
                        _MM_TRANSPOSE4_PS(_c0, _c1, _c2, _c3);
                        pC += 4;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                        _f2 = _mm_add_ps(_f2, _c2);
                        _f3 = _mm_add_ps(_f3, _c3);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
                        _f0 = _mm_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_comp_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm_comp_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm_comp_fmadd_ps(_c3, _beta, _f3);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm_set1_ps(pC[0] * beta);
                    __m128 _c1 = _mm_set1_ps(pC[1] * beta);
                    __m128 _c2 = _mm_set1_ps(pC[2] * beta);
                    __m128 _c3 = _mm_set1_ps(pC[3] * beta);

                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c1);
                    _f2 = _mm_add_ps(_f2, _c2);
                    _f3 = _mm_add_ps(_f3, _c3);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
                _f2 = _mm_mul_ps(_f2, _alpha);
                _f3 = _mm_mul_ps(_f3, _alpha);
            }

            __m128i _bf02 = float2bfloat_sse(_f0, _f2);
            __m128i _bf13 = float2bfloat_sse(_f1, _f3);

            if (output_transpose)
            {
                if (out_elempack == 4)
                {
                    // _bf02 = [ii0..3 jj0 | ii0..3 jj2], _bf13 = [ii0..3 jj1 | ii0..3 jj3]
                    // Transpose to [jj0..3 for each ii]
                    __m128i _t0 = _mm_unpacklo_epi16(_bf02, _bf13);  // [ii0_j0 ii0_j1 ii1_j0 ii1_j1 ii2_j0 ii2_j1 ii3_j0 ii3_j1]
                    __m128i _t1 = _mm_unpackhi_epi16(_bf02, _bf13);  // [ii0_j2 ii0_j3 ii1_j2 ii1_j3 ii2_j2 ii2_j3 ii3_j2 ii3_j3]
                    __m128i _d0 = _mm_unpacklo_epi32(_t0, _t1);      // [ii0_j0..j3 ii1_j0..j3]
                    __m128i _d1 = _mm_unpackhi_epi32(_t0, _t1);      // [ii2_j0..j3 ii3_j0..j3]
                    _mm_storel_epi64((__m128i*)p0, _d0);
                    _mm_storeh_pd((double*)(p0 + 4), _mm_castsi128_pd(_d0));
                    _mm_storel_epi64((__m128i*)(p0 + 4 * 2), _d1);
                    _mm_storeh_pd((double*)(p0 + 4 * 3), _mm_castsi128_pd(_d1));
                }
                if (out_elempack == 1)
                {
                    _mm_storel_epi64((__m128i*)p0, _bf02);
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep), _bf13);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 2), _mm_castsi128_pd(_bf02));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 3), _mm_castsi128_pd(_bf13));
                }
                p0 += out_hstep * 4;
            }
            else
            {
                if (out_elempack == 4)
                {
                    // _bf02 low64=[ii0..3 jj0], high64=[ii0..3 jj2]
                    // _bf13 low64=[ii0..3 jj1], high64=[ii0..3 jj3]
                    _mm_storel_epi64((__m128i*)p0, _bf02);                                       // jj0
                    _mm_storel_epi64((__m128i*)(p0 + 4), _bf13);                                 // jj1
                    _mm_storeh_pd((double*)(p0 + 4 * 2), _mm_castsi128_pd(_bf02));               // jj2
                    _mm_storeh_pd((double*)(p0 + 4 * 3), _mm_castsi128_pd(_bf13));               // jj3
                    p0 += 16;
                }
                if (out_elempack == 1)
                {
                    __m128i _t0 = _mm_unpacklo_epi16(_bf02, _bf13);
                    __m128i _t1 = _mm_unpackhi_epi16(_bf02, _bf13);
                    _bf02 = _mm_unpacklo_epi32(_t0, _t1);
                    _bf13 = _mm_unpackhi_epi32(_t0, _t1);

                    _mm_storel_epi64((__m128i*)(p0), _bf02);
                    _mm_storeh_pd((double*)(p0 + out_hstep), _mm_castsi128_pd(_bf02));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 2), _bf13);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 3), _mm_castsi128_pd(_bf13));
                    p0 += 4;
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _f0 = _mm_load_ps(pp);
            __m128 _f1 = _mm_load_ps(pp + 4);
            pp += 8;

            // deshuffle from the shuffle-based 4x2 dpbf16_ps kernel
            {
                __m128 _tmp0 = _mm_shuffle_ps(_f0, _f0, _MM_SHUFFLE(3, 1, 2, 0));
                __m128 _tmp1 = _mm_shuffle_ps(_f1, _f1, _MM_SHUFFLE(0, 2, 3, 1));
                _f0 = _mm_unpacklo_ps(_tmp0, _tmp1);
                _f1 = _mm_unpackhi_ps(_tmp0, _tmp1);
                _f1 = _mm_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
            }

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c1;
                    if (c_elempack == 4)
                    {
                        _c0 = _mm_loadu_ps(pC);
                        _c1 = _mm_loadu_ps(pC + 4);
                        pC += 8;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = _mm_setr_ps(pC[0], pC[c_hstep], pC[c_hstep * 2], pC[c_hstep * 3]);
                        _c1 = _mm_setr_ps(pC[1], pC[c_hstep + 1], pC[c_hstep * 2 + 1], pC[c_hstep * 3 + 1]);
                        pC += 2;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
                        _f0 = _mm_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_comp_fmadd_ps(_c1, _beta, _f1);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm_set1_ps(pC[0] * beta);
                    __m128 _c1 = _mm_set1_ps(pC[1] * beta);
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c1);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
            }

            __m128i _bf01 = float2bfloat_sse(_f0, _f1);

            if (output_transpose)
            {
                if (out_elempack == 4)
                {
                    unsigned short sum0[8];
                    _mm_storeu_si128((__m128i*)sum0, _bf01);

                    p0[0] = sum0[0];
                    p0[1] = sum0[4];
                    p0[4] = sum0[1];
                    p0[5] = sum0[5];
                    p0[8] = sum0[2];
                    p0[9] = sum0[6];
                    p0[12] = sum0[3];
                    p0[13] = sum0[7];
                }
                if (out_elempack == 1)
                {
                    _mm_storel_epi64((__m128i*)p0, _bf01);
                    _mm_storeh_pd((double*)(p0 + out_hstep), _mm_castsi128_pd(_bf01));
                }
                p0 += out_hstep * 2;
            }
            else
            {
                if (out_elempack == 4)
                {
                    _mm_storel_epi64((__m128i*)p0, _bf01);
                    _mm_storeh_pd((double*)(p0 + 4), _mm_castsi128_pd(_bf01));
                    p0 += 8;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[8];
                    _mm_storeu_si128((__m128i*)sum0, _bf01);

                    p0[0] = sum0[0];
                    p0[1] = sum0[4];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep + 1] = sum0[5];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 2 + 1] = sum0[6];
                    p0[out_hstep * 3] = sum0[3];
                    p0[out_hstep * 3 + 1] = sum0[7];
                    p0 += 2;
                }
            }
        }
        for (; jj < max_jj; jj++)
        {
            __m128 _f = _mm_load_ps(pp);
            pp += 4;

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f = _mm_add_ps(_f, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f = _mm_add_ps(_f, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        _c0 = _mm_loadu_ps(pC);
                        pC += 4;
                    }
                    else // if (c_elempack == 1)
                    {
                        _c0 = _mm_setr_ps(pC[0], pC[c_hstep], pC[c_hstep * 2], pC[c_hstep * 3]);
                        pC += 1;
                    }
                    _f = _mm_comp_fmadd_ps(_c0, _mm_set1_ps(beta), _f);
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm_set1_ps(pC[0] * beta);
                    _f = _mm_add_ps(_f, _c0);
                    pC += 1;
                }
            }

            if (alpha != 1.f)
            {
                _f = _mm_mul_ps(_f, _mm_set1_ps(alpha));
            }

            __m128i _bf = float2bfloat_sse(_f);

            if (output_transpose)
            {
                if (out_elempack == 4)
                {
                    // _bf = [ii0 ii1 ii2 ii3 ? ? ? ?] low64 has 4 bf16 for one jj
                    // Need to scatter to p0 + ii*4 for each ii
                    unsigned short sum0[4];
                    _mm_storel_epi64((__m128i*)sum0, _bf);
                    p0[0] = sum0[0];
                    p0[4] = sum0[1];
                    p0[4 * 2] = sum0[2];
                    p0[4 * 3] = sum0[3];
                }
                if (out_elempack == 1)
                {
                    _mm_storel_epi64((__m128i*)p0, _bf);
                }
                p0 += out_hstep;
            }
            else
            {
                if (out_elempack == 4)
                {
                    // _bf low64 = [ii0..3] for this jj, already packed
                    _mm_storel_epi64((__m128i*)p0, _bf);
                    p0 += 4;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[4];
                    _mm_storel_epi64((__m128i*)sum0, _bf);

                    p0[0] = sum0[0];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 3] = sum0[3];
                    p0++;
                }
            }
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
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

        float c0 = 0.f;
        float c1 = 0.f;
#if __SSE2__
        __m128 _c0 = _mm_set1_ps(0.f);
        __m128 _c1 = _mm_set1_ps(0.f);
#if __AVX512F__
        __m512 _c0_avx512 = _mm512_set1_ps(0.f);
        __m512 _c1_avx512 = _mm512_set1_ps(0.f);
#endif // __AVX512F__
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
#if __SSE2__
                _c0 = _mm_set1_ps(c0);
#if __AVX512F__
                _c0_avx512 = _mm512_set1_ps(c0);
#endif // __AVX512F__
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                c0 = pC[0] * beta;
                c1 = pC[1] * beta;
#if __SSE2__
                _c0 = _mm_set1_ps(c0);
                _c1 = _mm_set1_ps(c1);
#if __AVX512F__
                _c0_avx512 = _mm512_set1_ps(c0);
                _c1_avx512 = _mm512_set1_ps(c1);
#endif // __AVX512F__
#endif
            }
            if (broadcast_type_C == 3)
            {
                // c_elempack == 1
                pC = (const float*)C + (i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _f0 = _mm512_loadu_ps(pp);
            __m512 _f1 = _mm512_loadu_ps(pp + 16);
            pp += 32;

            // deshuffle from the shuffle-based 2x16 dpbf16_ps kernel
            {
                __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f1);
                __m512 _tmp1 = _mm512_unpackhi_ps(_f0, _f1);
                _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
            }

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c0_avx512);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c1_avx512);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    _c0_avx512 = _mm512_loadu_ps(pC);
                    _c1_avx512 = _mm512_loadu_ps(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0_avx512);
                        _f1 = _mm512_add_ps(_f1, _c1_avx512);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0_avx512, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1_avx512, _beta, _f1);
                    }
                    pC += 16;
                }
                if (broadcast_type_C == 4)
                {
                    _c0_avx512 = _mm512_loadu_ps(pC);
                    _c0_avx512 = _mm512_mul_ps(_c0_avx512, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c0_avx512);
                    pC += 16;
                }
            }

            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
            }

            __m256i _bf0 = float2bfloat_avx512(_f0);
            __m256i _bf1 = float2bfloat_avx512(_f1);

            if (output_transpose)
            {
                if (out_elempack == 16)
                {
                    _mm256_storeu_si256((__m256i*)p0, _bf0);
                    _mm256_storeu_si256((__m256i*)(p0 + 16), _bf1);
                }
                if (out_elempack == 8)
                {
                    _mm_storeu_si128((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storeu_si128((__m128i*)(p0 + 8), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8 + 8), _mm256_extractf128_si256(_bf1, 1));
                }
                if (out_elempack == 4)
                {
                    _mm_storel_epi64((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                    _mm_storel_epi64((__m128i*)(p0 + 4), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 0)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 0)));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8 + 4), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 1)));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 12 + 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf1, 1)));
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[16];
                    unsigned short sum1[16];
                    _mm256_storeu_si256((__m256i*)sum0, _bf0);
                    _mm256_storeu_si256((__m256i*)sum1, _bf1);

                    p0[0] = sum0[0];
                    p0[1] = sum1[0];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep + 1] = sum1[1];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 2 + 1] = sum1[2];
                    p0[out_hstep * 3] = sum0[3];
                    p0[out_hstep * 3 + 1] = sum1[3];
                    p0[out_hstep * 4] = sum0[4];
                    p0[out_hstep * 4 + 1] = sum1[4];
                    p0[out_hstep * 5] = sum0[5];
                    p0[out_hstep * 5 + 1] = sum1[5];
                    p0[out_hstep * 6] = sum0[6];
                    p0[out_hstep * 6 + 1] = sum1[6];
                    p0[out_hstep * 7] = sum0[7];
                    p0[out_hstep * 7 + 1] = sum1[7];
                    p0[out_hstep * 8] = sum0[8];
                    p0[out_hstep * 8 + 1] = sum1[8];
                    p0[out_hstep * 9] = sum0[9];
                    p0[out_hstep * 9 + 1] = sum1[9];
                    p0[out_hstep * 10] = sum0[10];
                    p0[out_hstep * 10 + 1] = sum1[10];
                    p0[out_hstep * 11] = sum0[11];
                    p0[out_hstep * 11 + 1] = sum1[11];
                    p0[out_hstep * 12] = sum0[12];
                    p0[out_hstep * 12 + 1] = sum1[12];
                    p0[out_hstep * 13] = sum0[13];
                    p0[out_hstep * 13 + 1] = sum1[13];
                    p0[out_hstep * 14] = sum0[14];
                    p0[out_hstep * 14 + 1] = sum1[14];
                    p0[out_hstep * 15] = sum0[15];
                    p0[out_hstep * 15 + 1] = sum1[15];
                }
                p0 += out_hstep * 16;
            }
            else
            {
                _mm256_storeu_si256((__m256i*)p0, _bf0);
                _mm256_storeu_si256((__m256i*)(p0 + out_hstep), _bf1);
                p0 += 16;
            }
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f0 = _mm_load_ps(pp);
            __m128 _f1 = _mm_load_ps(pp + 4);
            __m128 _f2 = _mm_load_ps(pp + 8);
            __m128 _f3 = _mm_load_ps(pp + 12);
            pp += 16;

            // 00 11 02 13
            // 04 15 06 17
            // 10 01 12 03
            // 14 05 16 07
            _f2 = _mm_shuffle_ps(_f2, _f2, _MM_SHUFFLE(2, 3, 0, 1));
            _f3 = _mm_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 3, 0, 1));

            __m128 _tmp0 = _mm_unpacklo_ps(_f0, _f2);
            __m128 _tmp1 = _mm_unpackhi_ps(_f0, _f2);
            __m128 _tmp2 = _mm_unpacklo_ps(_f1, _f3);
            __m128 _tmp3 = _mm_unpackhi_ps(_f1, _f3);

            _f0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp1)));
            _f1 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp2), _mm_castps_pd(_tmp3)));
            _f2 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp1)));
            _f3 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp2), _mm_castps_pd(_tmp3)));

            _f2 = _mm_shuffle_ps(_f2, _f2, _MM_SHUFFLE(2, 3, 0, 1));
            _f3 = _mm_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 3, 0, 1));

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    _f2 = _mm_add_ps(_f2, _c0);
                    _f3 = _mm_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    _f2 = _mm_add_ps(_f2, _c1);
                    _f3 = _mm_add_ps(_f3, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    _c0 = _mm_loadu_ps(pC);
                    _c1 = _mm_loadu_ps(pC + 4);
                    __m128 _c2 = _mm_loadu_ps(pC + c_hstep);
                    __m128 _c3 = _mm_loadu_ps(pC + c_hstep + 4);
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                        _f2 = _mm_add_ps(_f2, _c2);
                        _f3 = _mm_add_ps(_f3, _c3);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
                        _f0 = _mm_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_comp_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm_comp_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm_comp_fmadd_ps(_c3, _beta, _f3);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm_loadu_ps(pC);
                    _c1 = _mm_loadu_ps(pC + 4);
                    _c0 = _mm_mul_ps(_c0, _mm_set1_ps(beta));
                    _c1 = _mm_mul_ps(_c1, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c1);
                    _f2 = _mm_add_ps(_f2, _c0);
                    _f3 = _mm_add_ps(_f3, _c1);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
                _f2 = _mm_mul_ps(_f2, _alpha);
                _f3 = _mm_mul_ps(_f3, _alpha);
            }

            __m128i _bf0 = float2bfloat_sse(_f0, _f1);
            __m128i _bf1 = float2bfloat_sse(_f2, _f3);

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    // _bf0=[jj0..3|jj4..7] for ii0, _bf1=[jj0..3|jj4..7] for ii1
                    _mm_storeu_si128((__m128i*)p0, _bf0);
                    _mm_storeu_si128((__m128i*)(p0 + 8), _bf1);
                }
                if (out_elempack == 4)
                {
                    // _bf0 low64=[jj0..3 ii0], high64=[jj4..7 ii0]
                    // _bf1 low64=[jj0..3 ii1], high64=[jj4..7 ii1]
                    _mm_storel_epi64((__m128i*)p0, _bf0);
                    _mm_storel_epi64((__m128i*)(p0 + 4), _bf1);
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4), _mm_castsi128_pd(_bf0));
                    _mm_storeh_pd((double*)(p0 + out_hstep * 4 + 4), _mm_castsi128_pd(_bf1));
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[8];
                    unsigned short sum1[8];
                    _mm_storeu_si128((__m128i*)sum0, _bf0);
                    _mm_storeu_si128((__m128i*)sum1, _bf1);

                    p0[0] = sum0[0];
                    p0[1] = sum1[0];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep + 1] = sum1[1];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 2 + 1] = sum1[2];
                    p0[out_hstep * 3] = sum0[3];
                    p0[out_hstep * 3 + 1] = sum1[3];
                    p0[out_hstep * 4] = sum0[4];
                    p0[out_hstep * 4 + 1] = sum1[4];
                    p0[out_hstep * 5] = sum0[5];
                    p0[out_hstep * 5 + 1] = sum1[5];
                    p0[out_hstep * 6] = sum0[6];
                    p0[out_hstep * 6 + 1] = sum1[6];
                    p0[out_hstep * 7] = sum0[7];
                    p0[out_hstep * 7 + 1] = sum1[7];
                }
                p0 += out_hstep * 8;
            }
            else
            {
                _mm_storeu_si128((__m128i*)p0, _bf0);
                _mm_storeu_si128((__m128i*)(p0 + out_hstep), _bf1);
                p0 += 8;
            }
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = _mm_load_ps(pp);
            __m128 _f1 = _mm_load_ps(pp + 4);
            pp += 8;

            {
                __m128 _tmp0 = _mm_unpacklo_ps(_f0, _f1);
                __m128 _tmp1 = _mm_unpackhi_ps(_f0, _f1);
                _f0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp1)));
                _f1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp1)));
                _f1 = _mm_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
            }

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    _c0 = _mm_loadu_ps(pC);
                    _c1 = _mm_loadu_ps(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
                        _f0 = _mm_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_comp_fmadd_ps(_c1, _beta, _f1);
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm_loadu_ps(pC);
                    _c0 = _mm_mul_ps(_c0, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
            }

            __m128i _bf0 = float2bfloat_sse(_f0, _f1);

            if (output_transpose)
            {
                if (out_elempack == 4)
                {
                    // _bf0 = [jj0..3 for ii0 | jj0..3 for ii1]
                    _mm_storel_epi64((__m128i*)p0, _bf0);
                    _mm_storeh_pd((double*)(p0 + 4), _mm_castsi128_pd(_bf0));
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[8];
                    _mm_storeu_si128((__m128i*)sum0, _bf0);

                    p0[0] = sum0[0];
                    p0[1] = sum0[4];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep + 1] = sum0[5];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 2 + 1] = sum0[6];
                    p0[out_hstep * 3] = sum0[3];
                    p0[out_hstep * 3 + 1] = sum0[7];
                }
                p0 += out_hstep * 4;
            }
            else
            {
                _mm_storel_epi64((__m128i*)p0, _bf0);
                _mm_storel_epi64((__m128i*)(p0 + out_hstep), _mm_srli_si128(_bf0, 8));
                p0 += 4;
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
                    // c_elempack == 1
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

            unsigned short bf00 = float32_to_bfloat16(f00);
            unsigned short bf01 = float32_to_bfloat16(f01);
            unsigned short bf10 = float32_to_bfloat16(f10);
            unsigned short bf11 = float32_to_bfloat16(f11);

            if (output_transpose)
            {
                p0[0] = bf00;
                p0[1] = bf10;
                p0[out_hstep] = bf01;
                p0[out_hstep + 1] = bf11;
                p0 += out_hstep * 2;
            }
            else
            {
                p0[0] = bf00;
                p0[1] = bf01;
                p0[out_hstep] = bf10;
                p0[out_hstep + 1] = bf11;
                p0 += 2;
            }
        }
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[0];
            float f1 = pp[1];
            pp += 2;

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
                    // c_elempack == 1
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

            unsigned short bf0 = float32_to_bfloat16(f0);
            unsigned short bf1 = float32_to_bfloat16(f1);

            if (output_transpose)
            {
                p0[0] = bf0;
                p0[1] = bf1;
                p0 += out_hstep;
            }
            else
            {
                p0[0] = bf0;
                p0[out_hstep] = bf1;
                p0++;
            }
        }
    }
    for (; ii < max_ii; ii++)
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

        float c0 = 0.f;
#if __SSE2__
        __m128 _c0 = _mm_set1_ps(0.f);
#if __AVX512F__
        __m512 _c0_avx512 = _mm512_set1_ps(0.f);
#endif // __AVX512F__
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
#if __SSE2__
                _c0 = _mm_set1_ps(c0);
#if __AVX512F__
                _c0_avx512 = _mm512_set1_ps(c0);
#endif // __AVX512F__
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                c0 = pC[0] * beta;
#if __SSE2__
                _c0 = _mm_set1_ps(c0);
#if __AVX512F__
                _c0_avx512 = _mm512_set1_ps(c0);
#endif // __AVX512F__
#endif
            }
            if (broadcast_type_C == 3)
            {
                // c_elempack == 1
                pC = (const float*)C + (i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _f0 = _mm512_loadu_ps(pp);
            pp += 16;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    _c0_avx512 = _mm512_loadu_ps(pC);
                    _f0 = _mm512_fmadd_ps(_c0_avx512, _mm512_set1_ps(beta), _f0);
                    pC += 16;
                }
            }

            if (alpha != 1.f)
            {
                _f0 = _mm512_mul_ps(_f0, _mm512_set1_ps(alpha));
            }

            __m256i _bf0 = float2bfloat_avx512(_f0);

            if (output_transpose)
            {
                if (out_hstep == 1)
                {
                    _mm256_storeu_si256((__m256i*)p0, _bf0);
                }
                else
                {
                    if (out_elempack == 16)
                    {
                        _mm256_storeu_si256((__m256i*)p0, _bf0);
                    }
                    if (out_elempack == 8)
                    {
                        _mm_storeu_si128((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                        _mm_storeu_si128((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf0, 1));
                    }
                    if (out_elempack == 4)
                    {
                        _mm_storel_epi64((__m128i*)p0, _mm256_extractf128_si256(_bf0, 0));
                        _mm_storeh_pd((double*)(p0 + out_hstep * 4), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 0)));
                        _mm_storel_epi64((__m128i*)(p0 + out_hstep * 8), _mm256_extractf128_si256(_bf0, 1));
                        _mm_storeh_pd((double*)(p0 + out_hstep * 12), _mm_castsi128_pd(_mm256_extractf128_si256(_bf0, 1)));
                    }
                    if (out_elempack == 1)
                    {
                        unsigned short sum0[16];
                        _mm256_storeu_si256((__m256i*)sum0, _bf0);

                        p0[0] = sum0[0];
                        p0[out_hstep] = sum0[1];
                        p0[out_hstep * 2] = sum0[2];
                        p0[out_hstep * 3] = sum0[3];
                        p0[out_hstep * 4] = sum0[4];
                        p0[out_hstep * 5] = sum0[5];
                        p0[out_hstep * 6] = sum0[6];
                        p0[out_hstep * 7] = sum0[7];
                        p0[out_hstep * 8] = sum0[8];
                        p0[out_hstep * 9] = sum0[9];
                        p0[out_hstep * 10] = sum0[10];
                        p0[out_hstep * 11] = sum0[11];
                        p0[out_hstep * 12] = sum0[12];
                        p0[out_hstep * 13] = sum0[13];
                        p0[out_hstep * 14] = sum0[14];
                        p0[out_hstep * 15] = sum0[15];
                    }
                }
                p0 += out_hstep * 16;
            }
            else
            {
                _mm256_storeu_si256((__m256i*)p0, _bf0);
                p0 += 16;
            }
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f0 = _mm_loadu_ps(pp);
            __m128 _f1 = _mm_loadu_ps(pp + 4);
            pp += 8;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    _c0 = _mm_loadu_ps(pC);
                    __m128 _c1 = _mm_loadu_ps(pC + 4);
                    _f0 = _mm_comp_fmadd_ps(_c0, _mm_set1_ps(beta), _f0);
                    _f1 = _mm_comp_fmadd_ps(_c1, _mm_set1_ps(beta), _f1);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
            }

            __m128i _bf0 = float2bfloat_sse(_f0, _f1);

            if (output_transpose)
            {
                if (out_hstep == 1)
                {
                    _mm_storeu_si128((__m128i*)p0, _bf0);
                }
                else
                {
#if __AVX__
                    if (out_elempack == 8)
                    {
                        _mm_storeu_si128((__m128i*)p0, _bf0);
                    }
#endif // __AVX__
                    if (out_elempack == 4)
                    {
                        _mm_storel_epi64((__m128i*)p0, float2bfloat_sse(_f0));
                        _mm_storel_epi64((__m128i*)(p0 + out_hstep * 4), float2bfloat_sse(_f1));
                    }
                    if (out_elempack == 1)
                    {
                        unsigned short sum0[8];
                        _mm_storeu_si128((__m128i*)sum0, _bf0);

                        p0[0] = sum0[0];
                        p0[out_hstep] = sum0[1];
                        p0[out_hstep * 2] = sum0[2];
                        p0[out_hstep * 3] = sum0[3];
                        p0[out_hstep * 4] = sum0[4];
                        p0[out_hstep * 5] = sum0[5];
                        p0[out_hstep * 6] = sum0[6];
                        p0[out_hstep * 7] = sum0[7];
                    }
                }
                p0 += out_hstep * 8;
            }
            else
            {
                _mm_storeu_si128((__m128i*)p0, _bf0);
                p0 += 8;
            }
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = _mm_loadu_ps(pp);
            pp += 4;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    _c0 = _mm_loadu_ps(pC);
                    _f0 = _mm_comp_fmadd_ps(_c0, _mm_set1_ps(beta), _f0);
                    pC += 4;
                }
            }

            _f0 = _mm_mul_ps(_f0, _mm_set1_ps(alpha));

            __m128i _bf0 = float2bfloat_sse(_f0);

            if (output_transpose)
            {
                if (out_hstep == 1)
                {
                    _mm_storel_epi64((__m128i*)p0, _bf0);
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        _mm_storel_epi64((__m128i*)p0, _bf0);
                    }
                    if (out_elempack == 1)
                    {
                        unsigned short sum0[4];
                        _mm_storel_epi64((__m128i*)sum0, _bf0);

                        p0[0] = sum0[0];
                        p0[out_hstep] = sum0[1];
                        p0[out_hstep * 2] = sum0[2];
                        p0[out_hstep * 3] = sum0[3];
                    }
                }
                p0 += out_hstep * 4;
            }
            else
            {
                _mm_storel_epi64((__m128i*)p0, _bf0);
                p0 += 4;
            }
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            float f0 = pp[0];
            float f1 = pp[1];
            pp += 2;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                    f1 += c0;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    f0 += pC[0] * beta;
                    f1 += pC[1] * beta;
                    pC += 2;
                }
            }

            f0 *= alpha;
            f1 *= alpha;

            unsigned short bf0 = float32_to_bfloat16(f0);
            unsigned short bf1 = float32_to_bfloat16(f1);

            if (output_transpose)
            {
                p0[0] = bf0;
                p0[out_hstep] = bf1;
                p0 += out_hstep * 2;
            }
            else
            {
                p0[0] = bf0;
                p0[1] = bf1;
                p0 += 2;
            }
        }
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[0];
            pp += 1;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    f0 += pC[0] * beta;
                    pC += 1;
                }
            }

            f0 *= alpha;

            p0[0] = float32_to_bfloat16(f0);

            if (output_transpose)
            {
                p0 += out_hstep;
            }
            else
            {
                p0++;
            }
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

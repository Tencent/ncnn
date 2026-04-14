// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void convolution_im2col_gemm_transform_kernel_bf16s_avx512bf16(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt);
int convolution_im2col_gemm_bf16s_avx512bf16(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, int nT, const Option& opt);
#endif

static void convolution_im2col_pack_A_tile_bf16s(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    // A = (pa, maxk, inch/pa), outch
    const int A_hstep = A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
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
#if !__AVX512BF16__
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
            _mm256_storeu_si256((__m256i*)pp, float2bfloat_avx512(_r0));
            _mm256_storeu_si256((__m256i*)(pp + 16), float2bfloat_avx512(_r1));
            _mm256_storeu_si256((__m256i*)(pp + 16 * 2), float2bfloat_avx512(_r2));
            _mm256_storeu_si256((__m256i*)(pp + 16 * 3), float2bfloat_avx512(_r3));
            _mm256_storeu_si256((__m256i*)(pp + 16 * 4), float2bfloat_avx512(_r4));
            _mm256_storeu_si256((__m256i*)(pp + 16 * 5), float2bfloat_avx512(_r5));
            _mm256_storeu_si256((__m256i*)(pp + 16 * 6), float2bfloat_avx512(_r6));
            _mm256_storeu_si256((__m256i*)(pp + 16 * 7), float2bfloat_avx512(_r7));
            _mm256_storeu_si256((__m256i*)(pp + 16 * 8), float2bfloat_avx512(_r8));
            _mm256_storeu_si256((__m256i*)(pp + 16 * 9), float2bfloat_avx512(_r9));
            _mm256_storeu_si256((__m256i*)(pp + 16 * 10), float2bfloat_avx512(_ra));
            _mm256_storeu_si256((__m256i*)(pp + 16 * 11), float2bfloat_avx512(_rb));
            _mm256_storeu_si256((__m256i*)(pp + 16 * 12), float2bfloat_avx512(_rc));
            _mm256_storeu_si256((__m256i*)(pp + 16 * 13), float2bfloat_avx512(_rd));
            _mm256_storeu_si256((__m256i*)(pp + 16 * 14), float2bfloat_avx512(_re));
            _mm256_storeu_si256((__m256i*)(pp + 16 * 15), float2bfloat_avx512(_rf));
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
#endif // !__AVX512BF16__
#if __AVX512BF16__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p0[1]);
            pp[2] = float32_to_bfloat16(p1[0]);
            pp[3] = float32_to_bfloat16(p1[1]);
            pp[4] = float32_to_bfloat16(p2[0]);
            pp[5] = float32_to_bfloat16(p2[1]);
            pp[6] = float32_to_bfloat16(p3[0]);
            pp[7] = float32_to_bfloat16(p3[1]);
            pp[8] = float32_to_bfloat16(p4[0]);
            pp[9] = float32_to_bfloat16(p4[1]);
            pp[10] = float32_to_bfloat16(p5[0]);
            pp[11] = float32_to_bfloat16(p5[1]);
            pp[12] = float32_to_bfloat16(p6[0]);
            pp[13] = float32_to_bfloat16(p6[1]);
            pp[14] = float32_to_bfloat16(p7[0]);
            pp[15] = float32_to_bfloat16(p7[1]);
            pp[16] = float32_to_bfloat16(p8[0]);
            pp[17] = float32_to_bfloat16(p8[1]);
            pp[18] = float32_to_bfloat16(p9[0]);
            pp[19] = float32_to_bfloat16(p9[1]);
            pp[20] = float32_to_bfloat16(pa[0]);
            pp[21] = float32_to_bfloat16(pa[1]);
            pp[22] = float32_to_bfloat16(pb[0]);
            pp[23] = float32_to_bfloat16(pb[1]);
            pp[24] = float32_to_bfloat16(pc[0]);
            pp[25] = float32_to_bfloat16(pc[1]);
            pp[26] = float32_to_bfloat16(pd[0]);
            pp[27] = float32_to_bfloat16(pd[1]);
            pp[28] = float32_to_bfloat16(pe[0]);
            pp[29] = float32_to_bfloat16(pe[1]);
            pp[30] = float32_to_bfloat16(pf[0]);
            pp[31] = float32_to_bfloat16(pf[1]);
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
#endif // __AVX512BF16__
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp[2] = float32_to_bfloat16(p2[0]);
            pp[3] = float32_to_bfloat16(p3[0]);
            pp[4] = float32_to_bfloat16(p4[0]);
            pp[5] = float32_to_bfloat16(p5[0]);
            pp[6] = float32_to_bfloat16(p6[0]);
            pp[7] = float32_to_bfloat16(p7[0]);
            pp[8] = float32_to_bfloat16(p8[0]);
            pp[9] = float32_to_bfloat16(p9[0]);
            pp[10] = float32_to_bfloat16(pa[0]);
            pp[11] = float32_to_bfloat16(pb[0]);
            pp[12] = float32_to_bfloat16(pc[0]);
            pp[13] = float32_to_bfloat16(pd[0]);
            pp[14] = float32_to_bfloat16(pe[0]);
            pp[15] = float32_to_bfloat16(pf[0]);
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
#endif // __AVX512F__
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

        int kk = 0;
#if !__AVX512BF16__
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
            _mm_storeu_si128((__m128i*)pp, float2bfloat_avx(_r0));
            _mm_storeu_si128((__m128i*)(pp + 8), float2bfloat_avx(_r1));
            _mm_storeu_si128((__m128i*)(pp + 8 * 2), float2bfloat_avx(_r2));
            _mm_storeu_si128((__m128i*)(pp + 8 * 3), float2bfloat_avx(_r3));
            _mm_storeu_si128((__m128i*)(pp + 8 * 4), float2bfloat_avx(_r4));
            _mm_storeu_si128((__m128i*)(pp + 8 * 5), float2bfloat_avx(_r5));
            _mm_storeu_si128((__m128i*)(pp + 8 * 6), float2bfloat_avx(_r6));
            _mm_storeu_si128((__m128i*)(pp + 8 * 7), float2bfloat_avx(_r7));
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
#endif // !__AVX512BF16__
#if __AVX512BF16__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p0[1]);
            pp[2] = float32_to_bfloat16(p1[0]);
            pp[3] = float32_to_bfloat16(p1[1]);
            pp[4] = float32_to_bfloat16(p2[0]);
            pp[5] = float32_to_bfloat16(p2[1]);
            pp[6] = float32_to_bfloat16(p3[0]);
            pp[7] = float32_to_bfloat16(p3[1]);
            pp[8] = float32_to_bfloat16(p4[0]);
            pp[9] = float32_to_bfloat16(p4[1]);
            pp[10] = float32_to_bfloat16(p5[0]);
            pp[11] = float32_to_bfloat16(p5[1]);
            pp[12] = float32_to_bfloat16(p6[0]);
            pp[13] = float32_to_bfloat16(p6[1]);
            pp[14] = float32_to_bfloat16(p7[0]);
            pp[15] = float32_to_bfloat16(p7[1]);
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
#endif // __AVX512BF16__
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp[2] = float32_to_bfloat16(p2[0]);
            pp[3] = float32_to_bfloat16(p3[0]);
            pp[4] = float32_to_bfloat16(p4[0]);
            pp[5] = float32_to_bfloat16(p5[0]);
            pp[6] = float32_to_bfloat16(p6[0]);
            pp[7] = float32_to_bfloat16(p7[0]);
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
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

        int kk = 0;
#if !__AVX512BF16__
#if __AVX__
        for (; kk + 7 < max_kk; kk += 8)
        {
            __m256 _r0 = _mm256_loadu_ps(p0);
            __m256 _r1 = _mm256_loadu_ps(p1);
            __m256 _r2 = _mm256_loadu_ps(p2);
            __m256 _r3 = _mm256_loadu_ps(p3);
            transpose8x4_ps(_r0, _r1, _r2, _r3);
            _mm_storeu_si128((__m128i*)pp, float2bfloat_avx(_r0));
            _mm_storeu_si128((__m128i*)(pp + 8), float2bfloat_avx(_r1));
            _mm_storeu_si128((__m128i*)(pp + 16), float2bfloat_avx(_r2));
            _mm_storeu_si128((__m128i*)(pp + 24), float2bfloat_avx(_r3));
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
            _mm_storel_epi64((__m128i*)pp, float2bfloat_sse(_r0));
            _mm_storel_epi64((__m128i*)(pp + 4), float2bfloat_sse(_r1));
            _mm_storel_epi64((__m128i*)(pp + 8), float2bfloat_sse(_r2));
            _mm_storel_epi64((__m128i*)(pp + 12), float2bfloat_sse(_r3));
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
#endif // !__AVX512BF16__
#if __AVX512BF16__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p0[1]);
            pp[2] = float32_to_bfloat16(p1[0]);
            pp[3] = float32_to_bfloat16(p1[1]);
            pp[4] = float32_to_bfloat16(p2[0]);
            pp[5] = float32_to_bfloat16(p2[1]);
            pp[6] = float32_to_bfloat16(p3[0]);
            pp[7] = float32_to_bfloat16(p3[1]);
            pp += 8;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
        }
#endif // __AVX512BF16__
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp[2] = float32_to_bfloat16(p2[0]);
            pp[3] = float32_to_bfloat16(p3[0]);
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
#if __SSE2__
#if !__AVX512BF16__
#if __AVX__
        for (; kk + 7 < max_kk; kk += 8)
        {
            __m256 _r0 = _mm256_loadu_ps(p0);
            __m256 _r1 = _mm256_loadu_ps(p1);
            transpose8x2_ps(_r0, _r1);
            _mm_storeu_si128((__m128i*)pp, float2bfloat_avx(_r0));
            _mm_storeu_si128((__m128i*)(pp + 8), float2bfloat_avx(_r1));
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
            _mm_storel_epi64((__m128i*)pp, float2bfloat_sse(_tmp0));
            _mm_storel_epi64((__m128i*)(pp + 4), float2bfloat_sse(_tmp1));
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
#endif // !__AVX512BF16__
#endif // __SSE2__
#if __AVX512BF16__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p0[1]);
            pp[2] = float32_to_bfloat16(p1[0]);
            pp[3] = float32_to_bfloat16(p1[1]);
            pp += 4;
            p0 += 2;
            p1 += 2;
        }
#endif // __AVX512BF16__
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        int kk = 0;
#if __SSE2__
#if !__AVX512BF16__
#if __AVX__
        for (; kk + 7 < max_kk; kk += 8)
        {
            _mm_storeu_si128((__m128i*)pp, float2bfloat_avx(_mm256_loadu_ps(p0)));
            pp += 8;
            p0 += 8;
        }
#endif // __AVX__
        for (; kk + 3 < max_kk; kk += 4)
        {
            _mm_storel_epi64((__m128i*)pp, float2bfloat_sse(_mm_loadu_ps(p0)));
            pp += 4;
            p0 += 4;
        }
#endif // !__AVX512BF16__
#endif // __SSE2__
#if __AVX512BF16__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p0[1]);
            pp += 2;
            p0 += 2;
        }
#endif // __AVX512BF16__
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp += 1;
            p0++;
        }
    }
}

static void convolution_gemm_transB_packed_tile_bf16s(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end, int activation_type, const Mat& activation_params)
{
    // NCNN_LOGE("convolution_gemm_transB_packed_tile %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const int out_elempack = top_blob.elempack;
    const int out_hstep = (int)top_blob.cstep;

    const unsigned short* pAT = AT_tile;
    const unsigned short* pBT = BT_tile;
    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);

                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);
                __m512i _pA2 = _mm512_shuffle_i32x4(_pA0, _pA0, _MM_SHUFFLE(2, 3, 0, 1));
                __m512i _pB2 = _mm512_shuffle_i32x4(_pB0, _pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);

                _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_pA0, (__m512bh)_pB0);
                __m512i _pA3 = _mm512_shuffle_epi32(_pA2, _MM_PERM_BADC);
                _sum1 = _mm512_dpbf16_ps(_sum1, (__m512bh)_pA0, (__m512bh)_pB1);
                _sum2 = _mm512_dpbf16_ps(_sum2, (__m512bh)_pA1, (__m512bh)_pB0);
                _sum3 = _mm512_dpbf16_ps(_sum3, (__m512bh)_pA1, (__m512bh)_pB1);
                _sum8 = _mm512_dpbf16_ps(_sum8, (__m512bh)_pA2, (__m512bh)_pB0);
                _sum9 = _mm512_dpbf16_ps(_sum9, (__m512bh)_pA2, (__m512bh)_pB1);
                __m512i _pB3 = _mm512_shuffle_epi32(_pB2, _MM_PERM_ADCB);
                _sum4 = _mm512_dpbf16_ps(_sum4, (__m512bh)_pA0, (__m512bh)_pB2);
                _sum6 = _mm512_dpbf16_ps(_sum6, (__m512bh)_pA1, (__m512bh)_pB2);
                _suma = _mm512_dpbf16_ps(_suma, (__m512bh)_pA3, (__m512bh)_pB0);
                _sumb = _mm512_dpbf16_ps(_sumb, (__m512bh)_pA3, (__m512bh)_pB1);
                _sumc = _mm512_dpbf16_ps(_sumc, (__m512bh)_pA2, (__m512bh)_pB2);
                _sume = _mm512_dpbf16_ps(_sume, (__m512bh)_pA3, (__m512bh)_pB2);
                _sum5 = _mm512_dpbf16_ps(_sum5, (__m512bh)_pA0, (__m512bh)_pB3);
                _sum7 = _mm512_dpbf16_ps(_sum7, (__m512bh)_pA1, (__m512bh)_pB3);
                _sumd = _mm512_dpbf16_ps(_sumd, (__m512bh)_pA2, (__m512bh)_pB3);
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

            if (k_end)
            {
                // deshuffle
                _sum1 = _mm512_permute_ps(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm512_permute_ps(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                _sum5 = _mm512_permute_ps(_sum5, _MM_SHUFFLE(2, 1, 0, 3));
                _sum7 = _mm512_permute_ps(_sum7, _MM_SHUFFLE(2, 1, 0, 3));
                _sum9 = _mm512_permute_ps(_sum9, _MM_SHUFFLE(2, 1, 0, 3));
                _sumb = _mm512_permute_ps(_sumb, _MM_SHUFFLE(2, 1, 0, 3));
                _sumd = _mm512_permute_ps(_sumd, _MM_SHUFFLE(2, 1, 0, 3));
                _sumf = _mm512_permute_ps(_sumf, _MM_SHUFFLE(2, 1, 0, 3));

                __m512 _tmp0 = _mm512_unpacklo_ps(_sum0, _sum3);
                __m512 _tmp1 = _mm512_unpackhi_ps(_sum0, _sum3);
                __m512 _tmp2 = _mm512_unpacklo_ps(_sum2, _sum1);
                __m512 _tmp3 = _mm512_unpackhi_ps(_sum2, _sum1);
                __m512 _tmp4 = _mm512_unpacklo_ps(_sum4, _sum7);
                __m512 _tmp5 = _mm512_unpackhi_ps(_sum4, _sum7);
                __m512 _tmp6 = _mm512_unpacklo_ps(_sum6, _sum5);
                __m512 _tmp7 = _mm512_unpackhi_ps(_sum6, _sum5);
                __m512 _tmp8 = _mm512_unpacklo_ps(_sum8, _sumb);
                __m512 _tmp9 = _mm512_unpackhi_ps(_sum8, _sumb);
                __m512 _tmpa = _mm512_unpacklo_ps(_suma, _sum9);
                __m512 _tmpb = _mm512_unpackhi_ps(_suma, _sum9);
                __m512 _tmpc = _mm512_unpacklo_ps(_sumc, _sumf);
                __m512 _tmpd = _mm512_unpackhi_ps(_sumc, _sumf);
                __m512 _tmpe = _mm512_unpacklo_ps(_sume, _sumd);
                __m512 _tmpf = _mm512_unpackhi_ps(_sume, _sumd);

                _sum0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _sum1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _sum2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _sum3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _sum4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                _sum5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                _sum6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));
                _sum7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));
                _sum8 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp8), _mm512_castps_pd(_tmpa)));
                _sum9 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp8), _mm512_castps_pd(_tmpa)));
                _suma = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmpb), _mm512_castps_pd(_tmp9)));
                _sumb = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmpb), _mm512_castps_pd(_tmp9)));
                _sumc = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmpc), _mm512_castps_pd(_tmpe)));
                _sumd = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmpc), _mm512_castps_pd(_tmpe)));
                _sume = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmpf), _mm512_castps_pd(_tmpd)));
                _sumf = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmpf), _mm512_castps_pd(_tmpd)));

                _sum1 = _mm512_permute_ps(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm512_permute_ps(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                _sum5 = _mm512_permute_ps(_sum5, _MM_SHUFFLE(2, 1, 0, 3));
                _sum7 = _mm512_permute_ps(_sum7, _MM_SHUFFLE(2, 1, 0, 3));
                _sum9 = _mm512_permute_ps(_sum9, _MM_SHUFFLE(2, 1, 0, 3));
                _sumb = _mm512_permute_ps(_sumb, _MM_SHUFFLE(2, 1, 0, 3));
                _sumd = _mm512_permute_ps(_sumd, _MM_SHUFFLE(2, 1, 0, 3));
                _sumf = _mm512_permute_ps(_sumf, _MM_SHUFFLE(2, 1, 0, 3));

                _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum8, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp1 = _mm512_shuffle_f32x4(_sum1, _sum9, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp2 = _mm512_shuffle_f32x4(_sum2, _suma, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp3 = _mm512_shuffle_f32x4(_sum3, _sumb, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp4 = _mm512_shuffle_f32x4(_sum8, _sum0, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp5 = _mm512_shuffle_f32x4(_sum9, _sum1, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp6 = _mm512_shuffle_f32x4(_suma, _sum2, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp7 = _mm512_shuffle_f32x4(_sumb, _sum3, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp8 = _mm512_shuffle_f32x4(_sum4, _sumc, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp9 = _mm512_shuffle_f32x4(_sum5, _sumd, _MM_SHUFFLE(2, 0, 2, 0));
                _tmpa = _mm512_shuffle_f32x4(_sum6, _sume, _MM_SHUFFLE(2, 0, 2, 0));
                _tmpb = _mm512_shuffle_f32x4(_sum7, _sumf, _MM_SHUFFLE(2, 0, 2, 0));
                _tmpc = _mm512_shuffle_f32x4(_sumc, _sum4, _MM_SHUFFLE(3, 1, 3, 1));
                _tmpd = _mm512_shuffle_f32x4(_sumd, _sum5, _MM_SHUFFLE(3, 1, 3, 1));
                _tmpe = _mm512_shuffle_f32x4(_sume, _sum6, _MM_SHUFFLE(3, 1, 3, 1));
                _tmpf = _mm512_shuffle_f32x4(_sumf, _sum7, _MM_SHUFFLE(3, 1, 3, 1));

                _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp8, _MM_SHUFFLE(3, 1, 2, 0));
                _sum1 = _mm512_shuffle_f32x4(_tmp1, _tmp9, _MM_SHUFFLE(3, 1, 2, 0));
                _sum2 = _mm512_shuffle_f32x4(_tmp2, _tmpa, _MM_SHUFFLE(3, 1, 2, 0));
                _sum3 = _mm512_shuffle_f32x4(_tmp3, _tmpb, _MM_SHUFFLE(3, 1, 2, 0));
                _sum4 = _mm512_shuffle_f32x4(_tmp4, _tmpc, _MM_SHUFFLE(3, 1, 2, 0));
                _sum5 = _mm512_shuffle_f32x4(_tmp5, _tmpd, _MM_SHUFFLE(3, 1, 2, 0));
                _sum6 = _mm512_shuffle_f32x4(_tmp6, _tmpe, _MM_SHUFFLE(3, 1, 2, 0));
                _sum7 = _mm512_shuffle_f32x4(_tmp7, _tmpf, _MM_SHUFFLE(3, 1, 2, 0));
                _sum8 = _mm512_shuffle_f32x4(_tmp8, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                _sum9 = _mm512_shuffle_f32x4(_tmp9, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                _suma = _mm512_shuffle_f32x4(_tmpa, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                _sumb = _mm512_shuffle_f32x4(_tmpb, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                _sumc = _mm512_shuffle_f32x4(_tmpc, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                _sumd = _mm512_shuffle_f32x4(_tmpd, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                _sume = _mm512_shuffle_f32x4(_tmpe, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                _sumf = _mm512_shuffle_f32x4(_tmpf, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));

                if (pC)
                {
                    __m512 _bias = _mm512_loadu_ps(pC);
                    _sum0 = _mm512_add_ps(_sum0, _bias);
                    _sum1 = _mm512_add_ps(_sum1, _bias);
                    _sum2 = _mm512_add_ps(_sum2, _bias);
                    _sum3 = _mm512_add_ps(_sum3, _bias);
                    _sum4 = _mm512_add_ps(_sum4, _bias);
                    _sum5 = _mm512_add_ps(_sum5, _bias);
                    _sum6 = _mm512_add_ps(_sum6, _bias);
                    _sum7 = _mm512_add_ps(_sum7, _bias);
                    _sum8 = _mm512_add_ps(_sum8, _bias);
                    _sum9 = _mm512_add_ps(_sum9, _bias);
                    _suma = _mm512_add_ps(_suma, _bias);
                    _sumb = _mm512_add_ps(_sumb, _bias);
                    _sumc = _mm512_add_ps(_sumc, _bias);
                    _sumd = _mm512_add_ps(_sumd, _bias);
                    _sume = _mm512_add_ps(_sume, _bias);
                    _sumf = _mm512_add_ps(_sumf, _bias);
                }

                _sum0 = activation_avx512(_sum0, activation_type, activation_params);
                _sum1 = activation_avx512(_sum1, activation_type, activation_params);
                _sum2 = activation_avx512(_sum2, activation_type, activation_params);
                _sum3 = activation_avx512(_sum3, activation_type, activation_params);
                _sum4 = activation_avx512(_sum4, activation_type, activation_params);
                _sum5 = activation_avx512(_sum5, activation_type, activation_params);
                _sum6 = activation_avx512(_sum6, activation_type, activation_params);
                _sum7 = activation_avx512(_sum7, activation_type, activation_params);
                _sum8 = activation_avx512(_sum8, activation_type, activation_params);
                _sum9 = activation_avx512(_sum9, activation_type, activation_params);
                _suma = activation_avx512(_suma, activation_type, activation_params);
                _sumb = activation_avx512(_sumb, activation_type, activation_params);
                _sumc = activation_avx512(_sumc, activation_type, activation_params);
                _sumd = activation_avx512(_sumd, activation_type, activation_params);
                _sume = activation_avx512(_sume, activation_type, activation_params);
                _sumf = activation_avx512(_sumf, activation_type, activation_params);

                if (out_elempack == 16)
                {
                    _mm256_storeu_si256((__m256i*)(outptr0), float2bfloat_avx512(_sum0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 16), float2bfloat_avx512(_sum1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 32), float2bfloat_avx512(_sum2));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 48), float2bfloat_avx512(_sum3));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 64), float2bfloat_avx512(_sum4));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 80), float2bfloat_avx512(_sum5));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 96), float2bfloat_avx512(_sum6));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 112), float2bfloat_avx512(_sum7));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 128), float2bfloat_avx512(_sum8));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 128 + 16), float2bfloat_avx512(_sum9));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 128 + 32), float2bfloat_avx512(_suma));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 128 + 48), float2bfloat_avx512(_sumb));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 128 + 64), float2bfloat_avx512(_sumc));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 128 + 80), float2bfloat_avx512(_sumd));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 128 + 96), float2bfloat_avx512(_sume));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 128 + 112), float2bfloat_avx512(_sumf));
                    outptr0 += 256;
                }
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_sum8, _sum9, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_suma, _sumb, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_sumc, _sumd, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_sume, _sumf, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp8 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp9 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpa = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpb = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpc = _mm512_shuffle_f32x4(_sum8, _sum9, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpd = _mm512_shuffle_f32x4(_suma, _sumb, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpe = _mm512_shuffle_f32x4(_sumc, _sumd, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpf = _mm512_shuffle_f32x4(_sume, _sumf, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm256_storeu_si256((__m256i*)(outptr0), float2bfloat_avx512(_tmp0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 16), float2bfloat_avx512(_tmp1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 32), float2bfloat_avx512(_tmp2));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 48), float2bfloat_avx512(_tmp3));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 64), float2bfloat_avx512(_tmp4));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 80), float2bfloat_avx512(_tmp5));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 96), float2bfloat_avx512(_tmp6));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 112), float2bfloat_avx512(_tmp7));

                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8), float2bfloat_avx512(_tmp8));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 16), float2bfloat_avx512(_tmp9));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 32), float2bfloat_avx512(_tmpa));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 48), float2bfloat_avx512(_tmpb));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 64), float2bfloat_avx512(_tmpc));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 80), float2bfloat_avx512(_tmpd));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 96), float2bfloat_avx512(_tmpe));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 112), float2bfloat_avx512(_tmpf));

                    outptr0 += 128;
                }
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_sum8, _sum9, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_suma, _sumb, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_sumc, _sumd, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_sume, _sumf, _MM_SHUFFLE(2, 0, 2, 0));

                    __m512 _tmp8 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp9 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmpa = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmpb = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmpc = _mm512_shuffle_f32x4(_sum8, _sum9, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmpd = _mm512_shuffle_f32x4(_suma, _sumb, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmpe = _mm512_shuffle_f32x4(_sumc, _sumd, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmpf = _mm512_shuffle_f32x4(_sume, _sumf, _MM_SHUFFLE(3, 1, 3, 1));

                    _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum2 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum4 = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum5 = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum6 = _mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum7 = _mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(2, 0, 2, 0));

                    _sum8 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum9 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _suma = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _sumb = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));
                    _sumc = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(3, 1, 3, 1));
                    _sumd = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(3, 1, 3, 1));
                    _sume = _mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(3, 1, 3, 1));
                    _sumf = _mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm256_storeu_si256((__m256i*)(outptr0), float2bfloat_avx512(_sum0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 16), float2bfloat_avx512(_sum1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 32), float2bfloat_avx512(_sum2));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 48), float2bfloat_avx512(_sum3));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4), float2bfloat_avx512(_sum4));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4 + 16), float2bfloat_avx512(_sum5));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4 + 32), float2bfloat_avx512(_sum6));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4 + 48), float2bfloat_avx512(_sum7));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8), float2bfloat_avx512(_sum8));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 16), float2bfloat_avx512(_sum9));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 32), float2bfloat_avx512(_suma));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 48), float2bfloat_avx512(_sumb));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 12), float2bfloat_avx512(_sumc));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 12 + 16), float2bfloat_avx512(_sumd));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 12 + 32), float2bfloat_avx512(_sume));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 12 + 48), float2bfloat_avx512(_sumf));

                    outptr0 += 64;
                }
                if (out_elempack == 1)
                {
                    transpose16x16_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7, _sum8, _sum9, _suma, _sumb, _sumc, _sumd, _sume, _sumf);

                    _mm256_storeu_si256((__m256i*)(outptr0), float2bfloat_avx512(_sum0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep), float2bfloat_avx512(_sum1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 2), float2bfloat_avx512(_sum2));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 3), float2bfloat_avx512(_sum3));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4), float2bfloat_avx512(_sum4));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 5), float2bfloat_avx512(_sum5));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 6), float2bfloat_avx512(_sum6));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 7), float2bfloat_avx512(_sum7));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8), float2bfloat_avx512(_sum8));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 9), float2bfloat_avx512(_sum9));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 10), float2bfloat_avx512(_suma));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 11), float2bfloat_avx512(_sumb));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 12), float2bfloat_avx512(_sumc));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 13), float2bfloat_avx512(_sumd));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 14), float2bfloat_avx512(_sume));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 15), float2bfloat_avx512(_sumf));

                    outptr0 += 16;
                }
            }
            else
            {
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
            }

            outptr += 256;
        }
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                __m256i _pBB = _mm256_loadu_si256((const __m256i*)pB);
                __m512i _pB0 = combine8x2_epi32(_pBB, _pBB);

                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);

                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pB3 = _mm512_shuffle_epi32(_pB2, _MM_PERM_ADCB);

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

            if (k_end)
            {
                // deshuffle
                _sum1 = _mm512_permute_ps(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm512_permute_ps(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                _sum5 = _mm512_permute_ps(_sum5, _MM_SHUFFLE(2, 1, 0, 3));
                _sum7 = _mm512_permute_ps(_sum7, _MM_SHUFFLE(2, 1, 0, 3));

                __m512 _tmp0 = _mm512_unpacklo_ps(_sum0, _sum3);
                __m512 _tmp1 = _mm512_unpackhi_ps(_sum0, _sum3);
                __m512 _tmp2 = _mm512_unpacklo_ps(_sum2, _sum1);
                __m512 _tmp3 = _mm512_unpackhi_ps(_sum2, _sum1);
                __m512 _tmp4 = _mm512_unpacklo_ps(_sum4, _sum7);
                __m512 _tmp5 = _mm512_unpackhi_ps(_sum4, _sum7);
                __m512 _tmp6 = _mm512_unpacklo_ps(_sum6, _sum5);
                __m512 _tmp7 = _mm512_unpackhi_ps(_sum6, _sum5);

                _sum0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _sum1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _sum2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _sum3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _sum4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                _sum5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                _sum6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));
                _sum7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));

                _sum1 = _mm512_permute_ps(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm512_permute_ps(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                _sum5 = _mm512_permute_ps(_sum5, _MM_SHUFFLE(2, 1, 0, 3));
                _sum7 = _mm512_permute_ps(_sum7, _MM_SHUFFLE(2, 1, 0, 3));

                _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum4, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp1 = _mm512_shuffle_f32x4(_sum1, _sum5, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp2 = _mm512_shuffle_f32x4(_sum2, _sum6, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp3 = _mm512_shuffle_f32x4(_sum3, _sum7, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp4 = _mm512_shuffle_f32x4(_sum0, _sum4, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp5 = _mm512_shuffle_f32x4(_sum1, _sum5, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp6 = _mm512_shuffle_f32x4(_sum2, _sum6, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp7 = _mm512_shuffle_f32x4(_sum3, _sum7, _MM_SHUFFLE(2, 3, 3, 2));

                _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(2, 0, 2, 0));
                _sum1 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                _sum2 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(2, 0, 2, 0));
                _sum3 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                _sum4 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(1, 3, 1, 3));
                _sum5 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(1, 3, 1, 3));
                _sum6 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(1, 3, 1, 3));
                _sum7 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(1, 3, 1, 3));

                if (pC)
                {
                    __m512 _bias = _mm512_loadu_ps(pC);
                    _sum0 = _mm512_add_ps(_sum0, _bias);
                    _sum1 = _mm512_add_ps(_sum1, _bias);
                    _sum2 = _mm512_add_ps(_sum2, _bias);
                    _sum3 = _mm512_add_ps(_sum3, _bias);
                    _sum4 = _mm512_add_ps(_sum4, _bias);
                    _sum5 = _mm512_add_ps(_sum5, _bias);
                    _sum6 = _mm512_add_ps(_sum6, _bias);
                    _sum7 = _mm512_add_ps(_sum7, _bias);
                }

                _sum0 = activation_avx512(_sum0, activation_type, activation_params);
                _sum1 = activation_avx512(_sum1, activation_type, activation_params);
                _sum2 = activation_avx512(_sum2, activation_type, activation_params);
                _sum3 = activation_avx512(_sum3, activation_type, activation_params);
                _sum4 = activation_avx512(_sum4, activation_type, activation_params);
                _sum5 = activation_avx512(_sum5, activation_type, activation_params);
                _sum6 = activation_avx512(_sum6, activation_type, activation_params);
                _sum7 = activation_avx512(_sum7, activation_type, activation_params);

                if (out_elempack == 16)
                {
                    _mm256_storeu_si256((__m256i*)(outptr0), float2bfloat_avx512(_sum0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 16), float2bfloat_avx512(_sum1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 32), float2bfloat_avx512(_sum2));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 48), float2bfloat_avx512(_sum3));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 64), float2bfloat_avx512(_sum4));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 80), float2bfloat_avx512(_sum5));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 96), float2bfloat_avx512(_sum6));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 112), float2bfloat_avx512(_sum7));
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

                    _mm256_storeu_si256((__m256i*)(outptr0), float2bfloat_avx512(_tmp0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 16), float2bfloat_avx512(_tmp1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 32), float2bfloat_avx512(_tmp2));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 48), float2bfloat_avx512(_tmp3));

                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8), float2bfloat_avx512(_tmp4));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 16), float2bfloat_avx512(_tmp5));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 32), float2bfloat_avx512(_tmp6));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 48), float2bfloat_avx512(_tmp7));

                    outptr0 += 64;
                }
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 2, 3, 2));

                    _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum2 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum4 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum5 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum6 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm256_storeu_si256((__m256i*)(outptr0), float2bfloat_avx512(_sum0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 16), float2bfloat_avx512(_sum1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4), float2bfloat_avx512(_sum2));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4 + 16), float2bfloat_avx512(_sum3));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8), float2bfloat_avx512(_sum4));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 16), float2bfloat_avx512(_sum5));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 12), float2bfloat_avx512(_sum6));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 12 + 16), float2bfloat_avx512(_sum7));

                    outptr0 += 32;
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
                    __m128 _sum4_0 = _mm512_extractf32x4_ps(_sum4, 0);
                    __m128 _sum5_0 = _mm512_extractf32x4_ps(_sum5, 0);
                    __m128 _sum6_0 = _mm512_extractf32x4_ps(_sum6, 0);
                    __m128 _sum7_0 = _mm512_extractf32x4_ps(_sum7, 0);
                    __m128 _sum4_1 = _mm512_extractf32x4_ps(_sum4, 1);
                    __m128 _sum5_1 = _mm512_extractf32x4_ps(_sum5, 1);
                    __m128 _sum6_1 = _mm512_extractf32x4_ps(_sum6, 1);
                    __m128 _sum7_1 = _mm512_extractf32x4_ps(_sum7, 1);
                    __m128 _sum4_2 = _mm512_extractf32x4_ps(_sum4, 2);
                    __m128 _sum5_2 = _mm512_extractf32x4_ps(_sum5, 2);
                    __m128 _sum6_2 = _mm512_extractf32x4_ps(_sum6, 2);
                    __m128 _sum7_2 = _mm512_extractf32x4_ps(_sum7, 2);
                    __m128 _sum4_3 = _mm512_extractf32x4_ps(_sum4, 3);
                    __m128 _sum5_3 = _mm512_extractf32x4_ps(_sum5, 3);
                    __m128 _sum6_3 = _mm512_extractf32x4_ps(_sum6, 3);
                    __m128 _sum7_3 = _mm512_extractf32x4_ps(_sum7, 3);

                    _MM_TRANSPOSE4_PS(_sum0_0, _sum1_0, _sum2_0, _sum3_0);
                    _MM_TRANSPOSE4_PS(_sum4_0, _sum5_0, _sum6_0, _sum7_0);
                    _MM_TRANSPOSE4_PS(_sum0_1, _sum1_1, _sum2_1, _sum3_1);
                    _MM_TRANSPOSE4_PS(_sum4_1, _sum5_1, _sum6_1, _sum7_1);
                    _MM_TRANSPOSE4_PS(_sum0_2, _sum1_2, _sum2_2, _sum3_2);
                    _MM_TRANSPOSE4_PS(_sum4_2, _sum5_2, _sum6_2, _sum7_2);
                    _MM_TRANSPOSE4_PS(_sum0_3, _sum1_3, _sum2_3, _sum3_3);
                    _MM_TRANSPOSE4_PS(_sum4_3, _sum5_3, _sum6_3, _sum7_3);

                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 0), float2bfloat_sse(_sum0_0, _sum4_0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 1), float2bfloat_sse(_sum1_0, _sum5_0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 2), float2bfloat_sse(_sum2_0, _sum6_0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 3), float2bfloat_sse(_sum3_0, _sum7_0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4), float2bfloat_sse(_sum0_1, _sum4_1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 5), float2bfloat_sse(_sum1_1, _sum5_1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 6), float2bfloat_sse(_sum2_1, _sum6_1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 7), float2bfloat_sse(_sum3_1, _sum7_1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8), float2bfloat_sse(_sum0_2, _sum4_2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 9), float2bfloat_sse(_sum1_2, _sum5_2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 10), float2bfloat_sse(_sum2_2, _sum6_2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 11), float2bfloat_sse(_sum3_2, _sum7_2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 12), float2bfloat_sse(_sum0_3, _sum4_3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 13), float2bfloat_sse(_sum1_3, _sum5_3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 14), float2bfloat_sse(_sum2_3, _sum6_3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 15), float2bfloat_sse(_sum3_3, _sum7_3));

                    outptr0 += 8;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16, _sum1);
                _mm512_store_ps(outptr + 32, _sum2);
                _mm512_store_ps(outptr + 48, _sum3);
                _mm512_store_ps(outptr + 64, _sum4);
                _mm512_store_ps(outptr + 80, _sum5);
                _mm512_store_ps(outptr + 96, _sum6);
                _mm512_store_ps(outptr + 112, _sum7);
            }

            outptr += 128;
        }
#endif // defined(__x86_64__) || defined(_M_X64)

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
                _sum0 = _mm512_load_ps(outptr + 16 * 0);
                _sum1 = _mm512_load_ps(outptr + 16 * 1);
                _sum2 = _mm512_load_ps(outptr + 16 * 2);
                _sum3 = _mm512_load_ps(outptr + 16 * 3);
            }

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _pB0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)pB));
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
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

            if (k_end)
            {
                // deshuffle
                _sum1 = _mm512_permute_ps(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm512_permute_ps(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                __m512 _tmp0 = _mm512_unpacklo_ps(_sum0, _sum3);
                __m512 _tmp1 = _mm512_unpackhi_ps(_sum0, _sum3);
                __m512 _tmp2 = _mm512_unpacklo_ps(_sum2, _sum1);
                __m512 _tmp3 = _mm512_unpackhi_ps(_sum2, _sum1);
                _sum0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _sum1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _sum2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _sum3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _sum1 = _mm512_permute_ps(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm512_permute_ps(_sum3, _MM_SHUFFLE(2, 1, 0, 3));

                if (pC)
                {
                    __m512 _bias = _mm512_loadu_ps(pC);
                    _sum0 = _mm512_add_ps(_sum0, _bias);
                    _sum1 = _mm512_add_ps(_sum1, _bias);
                    _sum2 = _mm512_add_ps(_sum2, _bias);
                    _sum3 = _mm512_add_ps(_sum3, _bias);
                }

                _sum0 = activation_avx512(_sum0, activation_type, activation_params);
                _sum1 = activation_avx512(_sum1, activation_type, activation_params);
                _sum2 = activation_avx512(_sum2, activation_type, activation_params);
                _sum3 = activation_avx512(_sum3, activation_type, activation_params);
                if (out_elempack == 16)
                {
                    _mm256_storeu_si256((__m256i*)(outptr0 + 16 * 0), float2bfloat_avx512(_sum0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 16 * 1), float2bfloat_avx512(_sum1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 16 * 2), float2bfloat_avx512(_sum2));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 16 * 3), float2bfloat_avx512(_sum3));
                    outptr0 += 64;
                }
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm256_storeu_si256((__m256i*)(outptr0), float2bfloat_avx512(_tmp0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 16), float2bfloat_avx512(_tmp1));

                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8), float2bfloat_avx512(_tmp2));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 16), float2bfloat_avx512(_tmp3));

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

                    _mm256_storeu_si256((__m256i*)(outptr0), float2bfloat_avx512(_sum0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4), float2bfloat_avx512(_sum1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8), float2bfloat_avx512(_sum2));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 12), float2bfloat_avx512(_sum3));

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

                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 0), float2bfloat_sse(_sum0_0));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 4), float2bfloat_sse(_sum0_1));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 8), float2bfloat_sse(_sum0_2));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 12), float2bfloat_sse(_sum0_3));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 1), float2bfloat_sse(_sum1_0));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 5), float2bfloat_sse(_sum1_1));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 9), float2bfloat_sse(_sum1_2));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 13), float2bfloat_sse(_sum1_3));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 2), float2bfloat_sse(_sum2_0));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 6), float2bfloat_sse(_sum2_1));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 10), float2bfloat_sse(_sum2_2));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 14), float2bfloat_sse(_sum2_3));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 3), float2bfloat_sse(_sum3_0));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 7), float2bfloat_sse(_sum3_1));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 11), float2bfloat_sse(_sum3_2));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 15), float2bfloat_sse(_sum3_3));

                    outptr0 += 4;
                }
            }
            else
            {
                _mm512_store_ps(outptr + 16 * 0, _sum0);
                _mm512_store_ps(outptr + 16 * 1, _sum1);
                _mm512_store_ps(outptr + 16 * 2, _sum2);
                _mm512_store_ps(outptr + 16 * 3, _sum3);
            }

            outptr += 64;
        }
        for (; jj < max_jj; jj += 1)
        {
            const unsigned short* pA = pAT;

            __m512 _sum0;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = _mm512_loadu_ps(pC);
                }
                else
                {
                    _sum0 = _mm512_setzero_ps();
                }
            }
            else
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
            for (; kk < max_kk; kk += 1)
            {
                __m512 _pA = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pA));

                _sum0 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(bfloat16_to_float32(pB[0])), _sum0);

                pA += 16;
                pB += 1;
            }

            if (k_end)
            {
                _sum0 = activation_avx512(_sum0, activation_type, activation_params);

                if (out_elempack == 16)
                {
                    _mm256_storeu_si256((__m256i*)(outptr0), float2bfloat_avx512(_sum0));
                    outptr0 += 16;
                }
                if (out_elempack == 8)
                {
                    _mm_storeu_si128((__m128i*)(outptr0), float2bfloat_avx(_mm512_extractf32x8_ps(_sum0, 0)));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8), float2bfloat_avx(_mm512_extractf32x8_ps(_sum0, 1)));
                    outptr0 += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_storel_epi64((__m128i*)(outptr0), float2bfloat_sse(_mm512_extractf32x4_ps(_sum0, 0)));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 4), float2bfloat_sse(_mm512_extractf32x4_ps(_sum0, 1)));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 8), float2bfloat_sse(_mm512_extractf32x4_ps(_sum0, 2)));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 12), float2bfloat_sse(_mm512_extractf32x4_ps(_sum0, 3)));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[16];
                    _mm512_storeu_ps(sum0, _sum0);

                    outptr0[0] = float32_to_bfloat16(sum0[0]);
                    outptr0[out_hstep * 1] = float32_to_bfloat16(sum0[1]);
                    outptr0[out_hstep * 2] = float32_to_bfloat16(sum0[2]);
                    outptr0[out_hstep * 3] = float32_to_bfloat16(sum0[3]);
                    outptr0[out_hstep * 4] = float32_to_bfloat16(sum0[4]);
                    outptr0[out_hstep * 5] = float32_to_bfloat16(sum0[5]);
                    outptr0[out_hstep * 6] = float32_to_bfloat16(sum0[6]);
                    outptr0[out_hstep * 7] = float32_to_bfloat16(sum0[7]);
                    outptr0[out_hstep * 8] = float32_to_bfloat16(sum0[8]);
                    outptr0[out_hstep * 9] = float32_to_bfloat16(sum0[9]);
                    outptr0[out_hstep * 10] = float32_to_bfloat16(sum0[10]);
                    outptr0[out_hstep * 11] = float32_to_bfloat16(sum0[11]);
                    outptr0[out_hstep * 12] = float32_to_bfloat16(sum0[12]);
                    outptr0[out_hstep * 13] = float32_to_bfloat16(sum0[13]);
                    outptr0[out_hstep * 14] = float32_to_bfloat16(sum0[14]);
                    outptr0[out_hstep * 15] = float32_to_bfloat16(sum0[15]);
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
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);
                __m512i _pA00 = combine8x2_epi32(_pA0, _pA0);
                __m512i _pA11 = _mm512_shuffle_epi32(_pA00, _MM_PERM_BADC);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pB3 = _mm512_shuffle_epi32(_pB2, _MM_PERM_ADCB);
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

            if (k_end)
            {
                // deshuffle
                _sum1 = _mm512_permute_ps(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm512_permute_ps(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                _sum5 = _mm512_permute_ps(_sum5, _MM_SHUFFLE(2, 1, 0, 3));
                _sum7 = _mm512_permute_ps(_sum7, _MM_SHUFFLE(2, 1, 0, 3));

                __m512 _tmp0 = _mm512_unpacklo_ps(_sum0, _sum3);
                __m512 _tmp1 = _mm512_unpackhi_ps(_sum0, _sum3);
                __m512 _tmp2 = _mm512_unpacklo_ps(_sum2, _sum1);
                __m512 _tmp3 = _mm512_unpackhi_ps(_sum2, _sum1);
                __m512 _tmp4 = _mm512_unpacklo_ps(_sum4, _sum7);
                __m512 _tmp5 = _mm512_unpackhi_ps(_sum4, _sum7);
                __m512 _tmp6 = _mm512_unpacklo_ps(_sum6, _sum5);
                __m512 _tmp7 = _mm512_unpackhi_ps(_sum6, _sum5);

                _sum0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _sum1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _sum2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _sum3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _sum4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                _sum5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                _sum6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));
                _sum7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));

                _sum1 = _mm512_permute_ps(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm512_permute_ps(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                _sum5 = _mm512_permute_ps(_sum5, _MM_SHUFFLE(2, 1, 0, 3));
                _sum7 = _mm512_permute_ps(_sum7, _MM_SHUFFLE(2, 1, 0, 3));

                _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum4, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp1 = _mm512_shuffle_f32x4(_sum0, _sum4, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp2 = _mm512_shuffle_f32x4(_sum1, _sum5, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp3 = _mm512_shuffle_f32x4(_sum1, _sum5, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp4 = _mm512_shuffle_f32x4(_sum2, _sum6, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp5 = _mm512_shuffle_f32x4(_sum2, _sum6, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp6 = _mm512_shuffle_f32x4(_sum3, _sum7, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp7 = _mm512_shuffle_f32x4(_sum3, _sum7, _MM_SHUFFLE(2, 3, 3, 2));

                _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                _sum1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                _sum2 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                _sum3 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                _sum4 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(1, 3, 1, 3));
                _sum5 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(1, 3, 1, 3));
                _sum6 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(1, 3, 1, 3));
                _sum7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(1, 3, 1, 3));

                if (pC)
                {
                    __m256 _bias = _mm256_loadu_ps(pC);
                    __m512 _bias512 = _mm512_castsi512_ps(combine8x2_epi32(_mm256_castps_si256(_bias), _mm256_castps_si256(_bias)));
                    _sum0 = _mm512_add_ps(_sum0, _bias512);
                    _sum1 = _mm512_add_ps(_sum1, _bias512);
                    _sum2 = _mm512_add_ps(_sum2, _bias512);
                    _sum3 = _mm512_add_ps(_sum3, _bias512);
                    _sum4 = _mm512_add_ps(_sum4, _bias512);
                    _sum5 = _mm512_add_ps(_sum5, _bias512);
                    _sum6 = _mm512_add_ps(_sum6, _bias512);
                    _sum7 = _mm512_add_ps(_sum7, _bias512);
                }

                _sum0 = activation_avx512(_sum0, activation_type, activation_params);
                _sum1 = activation_avx512(_sum1, activation_type, activation_params);
                _sum2 = activation_avx512(_sum2, activation_type, activation_params);
                _sum3 = activation_avx512(_sum3, activation_type, activation_params);
                _sum4 = activation_avx512(_sum4, activation_type, activation_params);
                _sum5 = activation_avx512(_sum5, activation_type, activation_params);
                _sum6 = activation_avx512(_sum6, activation_type, activation_params);
                _sum7 = activation_avx512(_sum7, activation_type, activation_params);

                if (out_elempack == 8)
                {
                    __m256i _bf0 = float2bfloat_avx512(_sum0);
                    __m256i _bf1 = float2bfloat_avx512(_sum1);
                    __m256i _bf2 = float2bfloat_avx512(_sum2);
                    __m256i _bf3 = float2bfloat_avx512(_sum3);
                    __m256i _bf4 = float2bfloat_avx512(_sum4);
                    __m256i _bf5 = float2bfloat_avx512(_sum5);
                    __m256i _bf6 = float2bfloat_avx512(_sum6);
                    __m256i _bf7 = float2bfloat_avx512(_sum7);
                    _mm_storeu_si128((__m128i*)(outptr0), _mm256_extractf128_si256(_bf0, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8), _mm256_extractf128_si256(_bf1, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8 * 2), _mm256_extractf128_si256(_bf2, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8 * 3), _mm256_extractf128_si256(_bf3, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8 * 4), _mm256_extractf128_si256(_bf4, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8 * 5), _mm256_extractf128_si256(_bf5, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8 * 6), _mm256_extractf128_si256(_bf6, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8 * 7), _mm256_extractf128_si256(_bf7, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8 * 8), _mm256_extractf128_si256(_bf0, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8 * 9), _mm256_extractf128_si256(_bf1, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8 * 10), _mm256_extractf128_si256(_bf2, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8 * 11), _mm256_extractf128_si256(_bf3, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8 * 12), _mm256_extractf128_si256(_bf4, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8 * 13), _mm256_extractf128_si256(_bf5, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8 * 14), _mm256_extractf128_si256(_bf6, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8 * 15), _mm256_extractf128_si256(_bf7, 1));
                    outptr0 += 128;
                }
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 1, 3, 1));

                    _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum2 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum4 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum5 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum6 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm256_storeu_si256((__m256i*)(outptr0), float2bfloat_avx512(_sum0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 16), float2bfloat_avx512(_sum1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 32), float2bfloat_avx512(_sum2));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 48), float2bfloat_avx512(_sum3));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4), float2bfloat_avx512(_sum4));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4 + 16), float2bfloat_avx512(_sum5));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4 + 32), float2bfloat_avx512(_sum6));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4 + 48), float2bfloat_avx512(_sum7));
                    outptr0 += 64;
                }
                if (out_elempack == 1)
                {
                    __m512 _tmp0 = _mm512_unpacklo_ps(_sum0, _sum1);
                    __m512 _tmp1 = _mm512_unpacklo_ps(_sum2, _sum3);
                    __m512 _tmp2 = _mm512_unpacklo_ps(_sum4, _sum5);
                    __m512 _tmp3 = _mm512_unpacklo_ps(_sum6, _sum7);
                    __m512 _tmp4 = _mm512_unpackhi_ps(_sum0, _sum1);
                    __m512 _tmp5 = _mm512_unpackhi_ps(_sum2, _sum3);
                    __m512 _tmp6 = _mm512_unpackhi_ps(_sum4, _sum5);
                    __m512 _tmp7 = _mm512_unpackhi_ps(_sum6, _sum7);

                    _sum0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                    _sum1 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                    _sum2 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                    _sum3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                    _sum4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp5)));
                    _sum5 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp6), _mm512_castps_pd(_tmp7)));
                    _sum6 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp5)));
                    _sum7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp6), _mm512_castps_pd(_tmp7)));

                    _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp2 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp3 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp4 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp5 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp6 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp7 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 1, 3, 1));

                    _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum1 = _mm512_shuffle_f32x4(_tmp1, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum2 = _mm512_shuffle_f32x4(_tmp2, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum3 = _mm512_shuffle_f32x4(_tmp3, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum4 = _mm512_shuffle_f32x4(_tmp4, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum5 = _mm512_shuffle_f32x4(_tmp5, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum6 = _mm512_shuffle_f32x4(_tmp6, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum7 = _mm512_shuffle_f32x4(_tmp7, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));

                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 0), float2bfloat_avx512(_sum0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 1), float2bfloat_avx512(_sum1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 2), float2bfloat_avx512(_sum2));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 3), float2bfloat_avx512(_sum3));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4), float2bfloat_avx512(_sum4));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 5), float2bfloat_avx512(_sum5));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 6), float2bfloat_avx512(_sum6));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 7), float2bfloat_avx512(_sum7));
                    outptr0 += 16;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16, _sum1);
                _mm512_store_ps(outptr + 32, _sum2);
                _mm512_store_ps(outptr + 48, _sum3);
                _mm512_store_ps(outptr + 64, _sum4);
                _mm512_store_ps(outptr + 80, _sum5);
                _mm512_store_ps(outptr + 96, _sum6);
                _mm512_store_ps(outptr + 112, _sum7);
            }

            outptr += 128;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

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
            }
            else
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
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                __m256i _pB2 = _mm256_permute4x64_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB3 = _mm256_shuffle_epi32(_pB2, _MM_SHUFFLE(0, 3, 2, 1));
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

            if (k_end)
            {
                // deshuffle
                __m256 _tmp0 = _sum0;
                __m256 _tmp1 = _mm256_shuffle_ps(_sum1, _sum1, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp2 = _sum2;
                __m256 _tmp3 = _mm256_shuffle_ps(_sum3, _sum3, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp4 = _sum4;
                __m256 _tmp5 = _mm256_shuffle_ps(_sum5, _sum5, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp6 = _sum6;
                __m256 _tmp7 = _mm256_shuffle_ps(_sum7, _sum7, _MM_SHUFFLE(2, 1, 0, 3));

                _sum0 = _mm256_unpacklo_ps(_tmp0, _tmp3);
                _sum1 = _mm256_unpackhi_ps(_tmp0, _tmp3);
                _sum2 = _mm256_unpacklo_ps(_tmp2, _tmp1);
                _sum3 = _mm256_unpackhi_ps(_tmp2, _tmp1);
                _sum4 = _mm256_unpacklo_ps(_tmp4, _tmp7);
                _sum5 = _mm256_unpackhi_ps(_tmp4, _tmp7);
                _sum6 = _mm256_unpacklo_ps(_tmp6, _tmp5);
                _sum7 = _mm256_unpackhi_ps(_tmp6, _tmp5);

                _tmp0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_sum0), _mm256_castps_pd(_sum2)));
                _tmp1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_sum0), _mm256_castps_pd(_sum2)));
                _tmp2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_sum3), _mm256_castps_pd(_sum1)));
                _tmp3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_sum3), _mm256_castps_pd(_sum1)));
                _tmp4 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_sum4), _mm256_castps_pd(_sum6)));
                _tmp5 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_sum4), _mm256_castps_pd(_sum6)));
                _tmp6 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_sum7), _mm256_castps_pd(_sum5)));
                _tmp7 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_sum7), _mm256_castps_pd(_sum5)));

                _tmp1 = _mm256_shuffle_ps(_tmp1, _tmp1, _MM_SHUFFLE(2, 1, 0, 3));
                _tmp3 = _mm256_shuffle_ps(_tmp3, _tmp3, _MM_SHUFFLE(2, 1, 0, 3));
                _tmp5 = _mm256_shuffle_ps(_tmp5, _tmp5, _MM_SHUFFLE(2, 1, 0, 3));
                _tmp7 = _mm256_shuffle_ps(_tmp7, _tmp7, _MM_SHUFFLE(2, 1, 0, 3));

                _sum0 = _mm256_permute2f128_ps(_tmp0, _tmp4, _MM_SHUFFLE(0, 3, 0, 0));
                _sum1 = _mm256_permute2f128_ps(_tmp1, _tmp5, _MM_SHUFFLE(0, 3, 0, 0));
                _sum2 = _mm256_permute2f128_ps(_tmp2, _tmp6, _MM_SHUFFLE(0, 3, 0, 0));
                _sum3 = _mm256_permute2f128_ps(_tmp3, _tmp7, _MM_SHUFFLE(0, 3, 0, 0));
                _sum4 = _mm256_permute2f128_ps(_tmp4, _tmp0, _MM_SHUFFLE(0, 3, 0, 0));
                _sum5 = _mm256_permute2f128_ps(_tmp5, _tmp1, _MM_SHUFFLE(0, 3, 0, 0));
                _sum6 = _mm256_permute2f128_ps(_tmp6, _tmp2, _MM_SHUFFLE(0, 3, 0, 0));
                _sum7 = _mm256_permute2f128_ps(_tmp7, _tmp3, _MM_SHUFFLE(0, 3, 0, 0));

                if (pC)
                {
                    __m256 _bias = _mm256_loadu_ps(pC);
                    _sum0 = _mm256_add_ps(_sum0, _bias);
                    _sum1 = _mm256_add_ps(_sum1, _bias);
                    _sum2 = _mm256_add_ps(_sum2, _bias);
                    _sum3 = _mm256_add_ps(_sum3, _bias);
                    _sum4 = _mm256_add_ps(_sum4, _bias);
                    _sum5 = _mm256_add_ps(_sum5, _bias);
                    _sum6 = _mm256_add_ps(_sum6, _bias);
                    _sum7 = _mm256_add_ps(_sum7, _bias);
                }

                _sum0 = activation_avx(_sum0, activation_type, activation_params);
                _sum1 = activation_avx(_sum1, activation_type, activation_params);
                _sum2 = activation_avx(_sum2, activation_type, activation_params);
                _sum3 = activation_avx(_sum3, activation_type, activation_params);
                _sum4 = activation_avx(_sum4, activation_type, activation_params);
                _sum5 = activation_avx(_sum5, activation_type, activation_params);
                _sum6 = activation_avx(_sum6, activation_type, activation_params);
                _sum7 = activation_avx(_sum7, activation_type, activation_params);

                if (out_elempack == 8)
                {
                    _mm_storeu_si128((__m128i*)(outptr0), float2bfloat_avx(_sum0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8), float2bfloat_avx(_sum1));
                    _mm_storeu_si128((__m128i*)(outptr0 + 16), float2bfloat_avx(_sum2));
                    _mm_storeu_si128((__m128i*)(outptr0 + 24), float2bfloat_avx(_sum3));
                    _mm_storeu_si128((__m128i*)(outptr0 + 32), float2bfloat_avx(_sum4));
                    _mm_storeu_si128((__m128i*)(outptr0 + 40), float2bfloat_avx(_sum5));
                    _mm_storeu_si128((__m128i*)(outptr0 + 48), float2bfloat_avx(_sum6));
                    _mm_storeu_si128((__m128i*)(outptr0 + 56), float2bfloat_avx(_sum7));
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

                    _mm_storeu_si128((__m128i*)(outptr0), float2bfloat_avx(_tmp0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8), float2bfloat_avx(_tmp1));
                    _mm_storeu_si128((__m128i*)(outptr0 + 16), float2bfloat_avx(_tmp2));
                    _mm_storeu_si128((__m128i*)(outptr0 + 24), float2bfloat_avx(_tmp3));

                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4), float2bfloat_avx(_tmp4));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 8), float2bfloat_avx(_tmp5));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 16), float2bfloat_avx(_tmp6));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 24), float2bfloat_avx(_tmp7));

                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    __m128 _sum0_0 = _mm256_extractf128_ps(_sum0, 0);
                    __m128 _sum1_0 = _mm256_extractf128_ps(_sum1, 0);
                    __m128 _sum2_0 = _mm256_extractf128_ps(_sum2, 0);
                    __m128 _sum3_0 = _mm256_extractf128_ps(_sum3, 0);
                    __m128 _sum4_0 = _mm256_extractf128_ps(_sum4, 0);
                    __m128 _sum5_0 = _mm256_extractf128_ps(_sum5, 0);
                    __m128 _sum6_0 = _mm256_extractf128_ps(_sum6, 0);
                    __m128 _sum7_0 = _mm256_extractf128_ps(_sum7, 0);
                    __m128 _sum0_1 = _mm256_extractf128_ps(_sum0, 1);
                    __m128 _sum1_1 = _mm256_extractf128_ps(_sum1, 1);
                    __m128 _sum2_1 = _mm256_extractf128_ps(_sum2, 1);
                    __m128 _sum3_1 = _mm256_extractf128_ps(_sum3, 1);
                    __m128 _sum4_1 = _mm256_extractf128_ps(_sum4, 1);
                    __m128 _sum5_1 = _mm256_extractf128_ps(_sum5, 1);
                    __m128 _sum6_1 = _mm256_extractf128_ps(_sum6, 1);
                    __m128 _sum7_1 = _mm256_extractf128_ps(_sum7, 1);

                    _MM_TRANSPOSE4_PS(_sum0_0, _sum1_0, _sum2_0, _sum3_0);
                    _MM_TRANSPOSE4_PS(_sum4_0, _sum5_0, _sum6_0, _sum7_0);
                    _MM_TRANSPOSE4_PS(_sum0_1, _sum1_1, _sum2_1, _sum3_1);
                    _MM_TRANSPOSE4_PS(_sum4_1, _sum5_1, _sum6_1, _sum7_1);

                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 0), float2bfloat_sse(_sum0_0, _sum4_0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 1), float2bfloat_sse(_sum1_0, _sum5_0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 2), float2bfloat_sse(_sum2_0, _sum6_0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 3), float2bfloat_sse(_sum3_0, _sum7_0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4), float2bfloat_sse(_sum0_1, _sum4_1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 5), float2bfloat_sse(_sum1_1, _sum5_1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 6), float2bfloat_sse(_sum2_1, _sum6_1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 7), float2bfloat_sse(_sum3_1, _sum7_1));

                    outptr0 += 8;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8, _sum1);
                _mm256_store_ps(outptr + 16, _sum2);
                _mm256_store_ps(outptr + 24, _sum3);
                _mm256_store_ps(outptr + 32, _sum4);
                _mm256_store_ps(outptr + 40, _sum5);
                _mm256_store_ps(outptr + 48, _sum6);
                _mm256_store_ps(outptr + 56, _sum7);
            }

            outptr += 64;
        }
#endif // defined(__x86_64__) || defined(_M_X64)

        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

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
            }
            else
            {
                _sum0 = _mm256_load_ps(outptr + 8 * 0);
                _sum1 = _mm256_load_ps(outptr + 8 * 1);
                _sum2 = _mm256_load_ps(outptr + 8 * 2);
                _sum3 = _mm256_load_ps(outptr + 8 * 3);
            }

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);
                __m256i _pB0 = combine4x2_epi32(_pB, _pB);
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_pA0, (__m256bh)_pB0);
                _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_pA0, (__m256bh)_pB1);
                _sum2 = _mm256_dpbf16_ps(_sum2, (__m256bh)_pA1, (__m256bh)_pB0);
                _sum3 = _mm256_dpbf16_ps(_sum3, (__m256bh)_pA1, (__m256bh)_pB1);
                pA += 16;
                pB += 8;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk += 1)
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

            if (k_end)
            {
                // deshuffle
                _sum1 = _mm256_shuffle_ps(_sum1, _sum1, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm256_shuffle_ps(_sum3, _sum3, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp0 = _mm256_unpacklo_ps(_sum0, _sum3);
                __m256 _tmp1 = _mm256_unpackhi_ps(_sum0, _sum3);
                __m256 _tmp2 = _mm256_unpacklo_ps(_sum2, _sum1);
                __m256 _tmp3 = _mm256_unpackhi_ps(_sum2, _sum1);
                _sum0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp2)));
                _sum1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp2)));
                _sum2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp3), _mm256_castps_pd(_tmp1)));
                _sum3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp3), _mm256_castps_pd(_tmp1)));
                _sum1 = _mm256_shuffle_ps(_sum1, _sum1, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm256_shuffle_ps(_sum3, _sum3, _MM_SHUFFLE(2, 1, 0, 3));

                if (pC)
                {
                    __m256 _bias = _mm256_loadu_ps(pC);
                    _sum0 = _mm256_add_ps(_sum0, _bias);
                    _sum1 = _mm256_add_ps(_sum1, _bias);
                    _sum2 = _mm256_add_ps(_sum2, _bias);
                    _sum3 = _mm256_add_ps(_sum3, _bias);
                }

                _sum0 = activation_avx(_sum0, activation_type, activation_params);
                _sum1 = activation_avx(_sum1, activation_type, activation_params);
                _sum2 = activation_avx(_sum2, activation_type, activation_params);
                _sum3 = activation_avx(_sum3, activation_type, activation_params);
                if (out_elempack == 8)
                {
                    _mm_storeu_si128((__m128i*)(outptr0 + 8 * 0), float2bfloat_avx(_sum0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8 * 1), float2bfloat_avx(_sum1));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8 * 2), float2bfloat_avx(_sum2));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8 * 3), float2bfloat_avx(_sum3));
                    outptr0 += 32;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp2 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp3 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm_storeu_si128((__m128i*)(outptr0), float2bfloat_avx(_tmp0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8), float2bfloat_avx(_tmp1));

                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4), float2bfloat_avx(_tmp2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 8), float2bfloat_avx(_tmp3));

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

                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 0), float2bfloat_sse(_sum0_0));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 1), float2bfloat_sse(_sum1_0));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 2), float2bfloat_sse(_sum2_0));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 3), float2bfloat_sse(_sum3_0));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 4), float2bfloat_sse(_sum0_1));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 5), float2bfloat_sse(_sum1_1));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 6), float2bfloat_sse(_sum2_1));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 7), float2bfloat_sse(_sum3_1));

                    outptr0 += 4;
                }
            }
            else
            {
                _mm256_store_ps(outptr + 8 * 0, _sum0);
                _mm256_store_ps(outptr + 8 * 1, _sum1);
                _mm256_store_ps(outptr + 8 * 2, _sum2);
                _mm256_store_ps(outptr + 8 * 3, _sum3);
            }

            outptr += 32;
        }
        for (; jj < max_jj; jj += 1)
        {
            const unsigned short* pA = pAT;

            __m256 _sum0;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = _mm256_loadu_ps(pC);
                }
                else
                {
                    _sum0 = _mm256_setzero_ps();
                }
            }
            else
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
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pA));

                _sum0 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(bfloat16_to_float32(pB[0])), _sum0);

                pA += 8;
                pB += 1;
            }

            if (k_end)
            {
                _sum0 = activation_avx(_sum0, activation_type, activation_params);
                if (out_elempack == 8)
                {
                    _mm_storeu_si128((__m128i*)(outptr0), float2bfloat_avx(_sum0));
                    outptr0 += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_storel_epi64((__m128i*)(outptr0), float2bfloat_sse(_mm256_extractf128_ps(_sum0, 0)));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 4), float2bfloat_sse(_mm256_extractf128_ps(_sum0, 1)));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    _mm256_storeu_ps(sum0, _sum0);

                    outptr0[0] = float32_to_bfloat16(sum0[0]);
                    outptr0[out_hstep * 1] = float32_to_bfloat16(sum0[1]);
                    outptr0[out_hstep * 2] = float32_to_bfloat16(sum0[2]);
                    outptr0[out_hstep * 3] = float32_to_bfloat16(sum0[3]);
                    outptr0[out_hstep * 4] = float32_to_bfloat16(sum0[4]);
                    outptr0[out_hstep * 5] = float32_to_bfloat16(sum0[5]);
                    outptr0[out_hstep * 6] = float32_to_bfloat16(sum0[6]);
                    outptr0[out_hstep * 7] = float32_to_bfloat16(sum0[7]);
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
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512i _pA0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)pA));
                __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
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

            if (k_end)
            {
                // deshuffle
                _sum1 = _mm512_permute_ps(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm512_permute_ps(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                __m512 _tmp0 = _mm512_unpacklo_ps(_sum0, _sum3);
                __m512 _tmp1 = _mm512_unpacklo_ps(_sum2, _sum1);
                __m512 _tmp2 = _mm512_unpackhi_ps(_sum0, _sum3);
                __m512 _tmp3 = _mm512_unpackhi_ps(_sum2, _sum1);
                _sum0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                _sum1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                _sum2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp2)));
                _sum3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp2)));
                _sum1 = _mm512_permute_ps(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm512_permute_ps(_sum3, _MM_SHUFFLE(2, 1, 0, 3));

                if (pC)
                {
                    __m128 _bias = _mm_loadu_ps(pC);
                    __m512 _bias512 = _mm512_broadcast_f32x4(_bias);
                    _sum0 = _mm512_add_ps(_sum0, _bias512);
                    _sum1 = _mm512_add_ps(_sum1, _bias512);
                    _sum2 = _mm512_add_ps(_sum2, _bias512);
                    _sum3 = _mm512_add_ps(_sum3, _bias512);
                }

                _sum0 = activation_avx512(_sum0, activation_type, activation_params);
                _sum1 = activation_avx512(_sum1, activation_type, activation_params);
                _sum2 = activation_avx512(_sum2, activation_type, activation_params);
                _sum3 = activation_avx512(_sum3, activation_type, activation_params);

                if (out_elempack == 4)
                {
                    // transpose 128-bit lanes from [jj0,jj4,jj8,jj12] to [jj0,jj1,jj2,jj3]
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm256_storeu_si256((__m256i*)(outptr0), float2bfloat_avx512(_sum0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 16), float2bfloat_avx512(_sum1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 32), float2bfloat_avx512(_sum2));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 48), float2bfloat_avx512(_sum3));
                    outptr0 += 64;
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

                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 0), float2bfloat_sse(_sum0_0, _sum0_1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 0 + 8), float2bfloat_sse(_sum0_2, _sum0_3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 1), float2bfloat_sse(_sum1_0, _sum1_1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 1 + 8), float2bfloat_sse(_sum1_2, _sum1_3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 2), float2bfloat_sse(_sum2_0, _sum2_1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 2 + 8), float2bfloat_sse(_sum2_2, _sum2_3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 3), float2bfloat_sse(_sum3_0, _sum3_1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 3 + 8), float2bfloat_sse(_sum3_2, _sum3_3));
                    outptr0 += 16;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16, _sum1);
                _mm512_store_ps(outptr + 32, _sum2);
                _mm512_store_ps(outptr + 48, _sum3);
            }

            outptr += 64;
        }
#endif // __AVX512F__
#if __AVX__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

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
            }
            else
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
                __m128i _pA0 = _mm_loadu_si128((const __m128i*)pA);
                __m256i _pA00 = combine4x2_epi32(_pA0, _pA0);
                __m256i _pB01 = _mm256_loadu_si256((const __m256i*)pB);
                __m256i _pA11 = _mm256_shuffle_epi32(_pA00, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB23 = _mm256_shuffle_epi32(_pB01, _MM_SHUFFLE(0, 3, 2, 1));
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
                __m128 _pA = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pA));
                __m256 _pA0 = combine4x2_ps(_pA, _pA);
                __m256 _pB0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pB));

                __m256 _pA1 = _mm256_permute_ps(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256 _pB1 = _mm256_permute_ps(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm256_comp_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_pA1, _pB0, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_pA0, _pB1, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_pA1, _pB1, _sum3);

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                // deshuffle
                _sum2 = _mm256_shuffle_ps(_sum2, _sum2, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm256_shuffle_ps(_sum3, _sum3, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp0 = _mm256_unpacklo_ps(_sum0, _sum3);
                __m256 _tmp1 = _mm256_unpackhi_ps(_sum0, _sum3);
                __m256 _tmp2 = _mm256_unpacklo_ps(_sum1, _sum2);
                __m256 _tmp3 = _mm256_unpackhi_ps(_sum1, _sum2);
                _sum0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp2)));
                _sum1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp2)));
                _sum2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp3), _mm256_castps_pd(_tmp1)));
                _sum3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp3), _mm256_castps_pd(_tmp1)));
                _sum1 = _mm256_shuffle_ps(_sum1, _sum1, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm256_shuffle_ps(_sum3, _sum3, _MM_SHUFFLE(2, 1, 0, 3));

                if (pC)
                {
                    __m128 _bias = _mm_loadu_ps(pC);
                    __m256 _bias256 = combine4x2_ps(_bias, _bias);
                    _sum0 = _mm256_add_ps(_sum0, _bias256);
                    _sum1 = _mm256_add_ps(_sum1, _bias256);
                    _sum2 = _mm256_add_ps(_sum2, _bias256);
                    _sum3 = _mm256_add_ps(_sum3, _bias256);
                }

                _sum0 = activation_avx(_sum0, activation_type, activation_params);
                _sum1 = activation_avx(_sum1, activation_type, activation_params);
                _sum2 = activation_avx(_sum2, activation_type, activation_params);
                _sum3 = activation_avx(_sum3, activation_type, activation_params);
                if (out_elempack == 4)
                {
                    _mm_storel_epi64((__m128i*)(outptr0), float2bfloat_sse(_mm256_extractf128_ps(_sum0, 0)));
                    _mm_storel_epi64((__m128i*)(outptr0 + 4), float2bfloat_sse(_mm256_extractf128_ps(_sum1, 0)));
                    _mm_storel_epi64((__m128i*)(outptr0 + 8), float2bfloat_sse(_mm256_extractf128_ps(_sum2, 0)));
                    _mm_storel_epi64((__m128i*)(outptr0 + 12), float2bfloat_sse(_mm256_extractf128_ps(_sum3, 0)));
                    _mm_storel_epi64((__m128i*)(outptr0 + 16), float2bfloat_sse(_mm256_extractf128_ps(_sum0, 1)));
                    _mm_storel_epi64((__m128i*)(outptr0 + 20), float2bfloat_sse(_mm256_extractf128_ps(_sum1, 1)));
                    _mm_storel_epi64((__m128i*)(outptr0 + 24), float2bfloat_sse(_mm256_extractf128_ps(_sum2, 1)));
                    _mm_storel_epi64((__m128i*)(outptr0 + 28), float2bfloat_sse(_mm256_extractf128_ps(_sum3, 1)));
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    __m128 _s0l = _mm256_extractf128_ps(_sum0, 0);
                    __m128 _s1l = _mm256_extractf128_ps(_sum1, 0);
                    __m128 _s2l = _mm256_extractf128_ps(_sum2, 0);
                    __m128 _s3l = _mm256_extractf128_ps(_sum3, 0);
                    __m128 _s0h = _mm256_extractf128_ps(_sum0, 1);
                    __m128 _s1h = _mm256_extractf128_ps(_sum1, 1);
                    __m128 _s2h = _mm256_extractf128_ps(_sum2, 1);
                    __m128 _s3h = _mm256_extractf128_ps(_sum3, 1);

                    _MM_TRANSPOSE4_PS(_s0l, _s1l, _s2l, _s3l);
                    _MM_TRANSPOSE4_PS(_s0h, _s1h, _s2h, _s3h);

                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 0), float2bfloat_sse(_s0l, _s0h));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 1), float2bfloat_sse(_s1l, _s1h));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 2), float2bfloat_sse(_s2l, _s2h));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 3), float2bfloat_sse(_s3l, _s3h));

                    outptr0 += 8;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8, _sum1);
                _mm256_store_ps(outptr + 16, _sum2);
                _mm256_store_ps(outptr + 24, _sum3);
            }

            outptr += 32;
        }
#else  // __AVX__
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
                if (pC)
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
                else
                {
                    _sum0 = _mm_setzero_ps();
                    _sum1 = _mm_setzero_ps();
                    _sum2 = _mm_setzero_ps();
                    _sum3 = _mm_setzero_ps();
                    _sum4 = _mm_setzero_ps();
                    _sum5 = _mm_setzero_ps();
                    _sum6 = _mm_setzero_ps();
                    _sum7 = _mm_setzero_ps();
                }
            }
            else
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
            for (; kk < max_kk; kk++)
            {
                __m128 _pA = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pA));

                _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(bfloat16_to_float32(pB[0])), _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(bfloat16_to_float32(pB[1])), _sum1);
                _sum2 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(bfloat16_to_float32(pB[2])), _sum2);
                _sum3 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(bfloat16_to_float32(pB[3])), _sum3);
                _sum4 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(bfloat16_to_float32(pB[4])), _sum4);
                _sum5 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(bfloat16_to_float32(pB[5])), _sum5);
                _sum6 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(bfloat16_to_float32(pB[6])), _sum6);
                _sum7 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(bfloat16_to_float32(pB[7])), _sum7);

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                _sum0 = activation_sse(_sum0, activation_type, activation_params);
                _sum1 = activation_sse(_sum1, activation_type, activation_params);
                _sum2 = activation_sse(_sum2, activation_type, activation_params);
                _sum3 = activation_sse(_sum3, activation_type, activation_params);
                _sum4 = activation_sse(_sum4, activation_type, activation_params);
                _sum5 = activation_sse(_sum5, activation_type, activation_params);
                _sum6 = activation_sse(_sum6, activation_type, activation_params);
                _sum7 = activation_sse(_sum7, activation_type, activation_params);
                if (out_elempack == 4)
                {
                    _mm_storel_epi64((__m128i*)(outptr0), float2bfloat_sse(_sum0));
                    _mm_storel_epi64((__m128i*)(outptr0 + 4), float2bfloat_sse(_sum1));
                    _mm_storel_epi64((__m128i*)(outptr0 + 8), float2bfloat_sse(_sum2));
                    _mm_storel_epi64((__m128i*)(outptr0 + 12), float2bfloat_sse(_sum3));
                    _mm_storel_epi64((__m128i*)(outptr0 + 16), float2bfloat_sse(_sum4));
                    _mm_storel_epi64((__m128i*)(outptr0 + 20), float2bfloat_sse(_sum5));
                    _mm_storel_epi64((__m128i*)(outptr0 + 24), float2bfloat_sse(_sum6));
                    _mm_storel_epi64((__m128i*)(outptr0 + 28), float2bfloat_sse(_sum7));
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    _MM_TRANSPOSE4_PS(_sum0, _sum1, _sum2, _sum3);
                    _MM_TRANSPOSE4_PS(_sum4, _sum5, _sum6, _sum7);

                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 0), float2bfloat_sse(_sum0, _sum4));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 1), float2bfloat_sse(_sum1, _sum5));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 2), float2bfloat_sse(_sum2, _sum6));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 3), float2bfloat_sse(_sum3, _sum7));

                    outptr0 += 8;
                }
            }
            else
            {
                _mm_store_ps(outptr, _sum0);
                _mm_store_ps(outptr + 4, _sum1);
                _mm_store_ps(outptr + 8, _sum2);
                _mm_store_ps(outptr + 12, _sum3);
                _mm_store_ps(outptr + 16, _sum4);
                _mm_store_ps(outptr + 20, _sum5);
                _mm_store_ps(outptr + 24, _sum6);
                _mm_store_ps(outptr + 28, _sum7);
            }

            outptr += 32;
        }
#endif // __AVX__
#endif // defined(__x86_64__) || defined(_M_X64)

        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

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
            }
            else
            {
                _sum0 = _mm_load_ps(outptr + 4 * 0);
                _sum1 = _mm_load_ps(outptr + 4 * 1);
                _sum2 = _mm_load_ps(outptr + 4 * 2);
                _sum3 = _mm_load_ps(outptr + 4 * 3);
            }

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA0 = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB0 = _mm_loadu_si128((const __m128i*)pB);
                __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_pA0, (__m128bh)_pB0);
                _sum1 = _mm_dpbf16_ps(_sum1, (__m128bh)_pA0, (__m128bh)_pB1);
                _sum2 = _mm_dpbf16_ps(_sum2, (__m128bh)_pA1, (__m128bh)_pB0);
                _sum3 = _mm_dpbf16_ps(_sum3, (__m128bh)_pA1, (__m128bh)_pB1);
                pA += 8;
                pB += 8;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk += 1)
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

            if (k_end)
            {
                // deshuffle
                _sum1 = _mm_shuffle_ps(_sum1, _sum1, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm_shuffle_ps(_sum3, _sum3, _MM_SHUFFLE(2, 1, 0, 3));
                __m128 _tmp0 = _mm_unpacklo_ps(_sum0, _sum3);
                __m128 _tmp1 = _mm_unpackhi_ps(_sum0, _sum3);
                __m128 _tmp2 = _mm_unpacklo_ps(_sum2, _sum1);
                __m128 _tmp3 = _mm_unpackhi_ps(_sum2, _sum1);
                _sum0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp2)));
                _sum1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp2)));
                _sum2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp3), _mm_castps_pd(_tmp1)));
                _sum3 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp3), _mm_castps_pd(_tmp1)));
                _sum1 = _mm_shuffle_ps(_sum1, _sum1, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm_shuffle_ps(_sum3, _sum3, _MM_SHUFFLE(2, 1, 0, 3));

                if (pC)
                {
                    __m128 _bias = _mm_loadu_ps(pC);
                    _sum0 = _mm_add_ps(_sum0, _bias);
                    _sum1 = _mm_add_ps(_sum1, _bias);
                    _sum2 = _mm_add_ps(_sum2, _bias);
                    _sum3 = _mm_add_ps(_sum3, _bias);
                }

                _sum0 = activation_sse(_sum0, activation_type, activation_params);
                _sum1 = activation_sse(_sum1, activation_type, activation_params);
                _sum2 = activation_sse(_sum2, activation_type, activation_params);
                _sum3 = activation_sse(_sum3, activation_type, activation_params);
                if (out_elempack == 4)
                {
                    _mm_storel_epi64((__m128i*)(outptr0 + 4 * 0), float2bfloat_sse(_sum0));
                    _mm_storel_epi64((__m128i*)(outptr0 + 4 * 1), float2bfloat_sse(_sum1));
                    _mm_storel_epi64((__m128i*)(outptr0 + 4 * 2), float2bfloat_sse(_sum2));
                    _mm_storel_epi64((__m128i*)(outptr0 + 4 * 3), float2bfloat_sse(_sum3));
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    _MM_TRANSPOSE4_PS(_sum0, _sum1, _sum2, _sum3);

                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 0), float2bfloat_sse(_sum0));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 1), float2bfloat_sse(_sum1));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 2), float2bfloat_sse(_sum2));
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep * 3), float2bfloat_sse(_sum3));
                    outptr0 += 4;
                }
            }
            else
            {
                _mm_store_ps(outptr + 4 * 0, _sum0);
                _mm_store_ps(outptr + 4 * 1, _sum1);
                _mm_store_ps(outptr + 4 * 2, _sum2);
                _mm_store_ps(outptr + 4 * 3, _sum3);
            }

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            const unsigned short* pA = pAT;

            __m128 _sum0;

            if (k == 0)
            {
                _sum0 = _mm_setzero_ps();
            }
            else
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
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pA));

                _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(bfloat16_to_float32(pB[0])), _sum0);

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (pC)
                {
                    _sum0 = _mm_add_ps(_sum0, _mm_loadu_ps(pC));
                }

                _sum0 = activation_sse(_sum0, activation_type, activation_params);
                if (out_elempack == 4)
                {
                    _mm_storel_epi64((__m128i*)(outptr0), float2bfloat_sse(_sum0));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    _mm_storeu_ps(sum0, _sum0);

                    outptr0[0] = float32_to_bfloat16(sum0[0]);
                    outptr0[out_hstep] = float32_to_bfloat16(sum0[1]);
                    outptr0[out_hstep * 2] = float32_to_bfloat16(sum0[2]);
                    outptr0[out_hstep * 3] = float32_to_bfloat16(sum0[3]);
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
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
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
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m512i _pA = _mm512_castpd_si512(_mm512_set1_pd(((const double*)pA)[0]));
                __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
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

            if (k_end)
            {
                // deshuffle
                __m512 _tmp0 = _mm512_unpacklo_ps(_sum0, _sum1);
                __m512 _tmp1 = _mm512_unpackhi_ps(_sum0, _sum1);
                _sum0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                _sum1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                _sum1 = _mm512_permute_ps(_sum1, _MM_SHUFFLE(2, 1, 0, 3));

                if (pC)
                {
                    _sum0 = _mm512_add_ps(_sum0, _mm512_set1_ps(pC[0]));
                    _sum1 = _mm512_add_ps(_sum1, _mm512_set1_ps(pC[1]));
                }

                _sum0 = activation_avx512(_sum0, activation_type, activation_params);
                _sum1 = activation_avx512(_sum1, activation_type, activation_params);

                // if (out_elempack == 1)
                {
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 0), float2bfloat_avx512(_sum0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 1), float2bfloat_avx512(_sum1));
                    outptr0 += 16;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16, _sum1);
            }

            outptr += 32;
        }
#endif // __AVX512F__
#if __AVX__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

            __m256 _sum0;
            __m256 _sum1;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_ps();
                _sum1 = _mm256_setzero_ps();
            }
            else
            {
                _sum0 = _mm256_load_ps(outptr);
                _sum1 = _mm256_load_ps(outptr + 8);
            }

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA0 = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)pA));
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(0, 3, 2, 1));
                _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_pA0, (__m256bh)_pB);
                _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_pA1, (__m256bh)_pB);
                pA += 4;
                pB += 16;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m256 _pA0 = bfloat2float_avx(_pA);
                __m256 _pA1 = _mm256_permute_ps(_pA0, _MM_SHUFFLE(0, 3, 2, 1));
                __m256 _pB0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pB));

                _sum0 = _mm256_comp_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_pA1, _pB0, _sum1);

                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                // deshuffle from the shuffle-based 2x8 dpbf16_ps kernel
                __m128 _f0 = _mm256_extractf128_ps(_sum0, 0);
                __m128 _f1 = _mm256_extractf128_ps(_sum0, 1);
                __m128 _f2 = _mm256_extractf128_ps(_sum1, 0);
                __m128 _f3 = _mm256_extractf128_ps(_sum1, 1);

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
                    __m128 _bias0 = _mm_set1_ps(pC[0]);
                    __m128 _bias1 = _mm_set1_ps(pC[1]);
                    _f0 = _mm_add_ps(_f0, _bias0);
                    _f1 = _mm_add_ps(_f1, _bias0);
                    _f2 = _mm_add_ps(_f2, _bias1);
                    _f3 = _mm_add_ps(_f3, _bias1);
                }

                _f0 = activation_sse(_f0, activation_type, activation_params);
                _f1 = activation_sse(_f1, activation_type, activation_params);
                _f2 = activation_sse(_f2, activation_type, activation_params);
                _f3 = activation_sse(_f3, activation_type, activation_params);

                // if (out_elempack == 1)
                {
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 0), float2bfloat_sse(_f0, _f1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 1), float2bfloat_sse(_f2, _f3));
                    outptr0 += 8;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8, _sum1);
            }

            outptr += 16;
        }
#else  // __AVX__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

            float _sum0[8] = {0.f};
            float _sum1[8] = {0.f};

            if (k == 0)
            {
                if (pC)
                {
                    for (int i = 0; i < 8; i++)
                    {
                        _sum0[i] = pC[0];
                        _sum1[i] = pC[1];
                    }
                }
            }
            else
            {
                for (int i = 0; i < 8; i++)
                {
                    _sum0[i] = ((const float*)outptr)[i];
                    _sum1[i] = ((const float*)outptr)[8 + i];
                }
            }

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                float pA0 = bfloat16_to_float32(pA[0]);
                float pA1 = bfloat16_to_float32(pA[1]);
                for (int i = 0; i < 8; i++)
                {
                    float pBi = bfloat16_to_float32(pB[i]);
                    _sum0[i] += pA0 * pBi;
                    _sum1[i] += pA1 * pBi;
                }
                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                for (int i = 0; i < 8; i++)
                {
                    _sum0[i] = activation_ss(_sum0[i], activation_type, activation_params);
                    _sum1[i] = activation_ss(_sum1[i], activation_type, activation_params);
                }
                if (out_elempack == 1)
                {
                    for (int i = 0; i < 8; i++)
                    {
                        outptr0[out_hstep * 0 + i] = float32_to_bfloat16(_sum0[i]);
                        outptr0[out_hstep * 1 + i] = float32_to_bfloat16(_sum1[i]);
                    }
                    outptr0 += 8;
                }
            }
            else
            {
                for (int i = 0; i < 8; i++)
                {
                    ((float*)outptr)[i] = _sum0[i];
                    ((float*)outptr)[8 + i] = _sum1[i];
                }
            }

            outptr += 16;
        }
#endif // __AVX__
#endif // defined(__x86_64__) || defined(_M_X64)

        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                _sum0 = _mm_setzero_ps();
                _sum1 = _mm_setzero_ps();
            }
            else
            {
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4);
            }

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_castpd_si128(_mm_load1_pd((const double*)pA));
                __m128i _pB0 = _mm_loadu_si128((const __m128i*)pB);
                __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
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

            if (k_end)
            {
                // deshuffle
                __m128 _tmp0 = _mm_unpacklo_ps(_sum0, _sum1);
                __m128 _tmp1 = _mm_unpackhi_ps(_sum0, _sum1);
                _sum0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp1)));
                _sum1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(_tmp0), _mm_castps_pd(_tmp1)));
                _sum1 = _mm_shuffle_ps(_sum1, _sum1, _MM_SHUFFLE(2, 1, 0, 3));

                if (pC)
                {
                    _sum0 = _mm_add_ps(_sum0, _mm_set1_ps(pC[0]));
                    _sum1 = _mm_add_ps(_sum1, _mm_set1_ps(pC[1]));
                }

                _sum0 = activation_sse(_sum0, activation_type, activation_params);
                _sum1 = activation_sse(_sum1, activation_type, activation_params);

                // if (out_elempack == 1)
                {
                    __m128i _bf0 = float2bfloat_sse(_sum0, _sum1);
                    _mm_storel_epi64((__m128i*)outptr0, _bf0);
                    _mm_storel_epi64((__m128i*)(outptr0 + out_hstep), _mm_srli_si128(_bf0, 8));
                    outptr0 += 4;
                }
            }
            else
            {
                _mm_store_ps(outptr, _sum0);
                _mm_store_ps(outptr + 4, _sum1);
            }

            outptr += 8;
        }
#endif // __SSE2__
        for (; jj < max_jj; jj += 1)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[1];
                }
                else
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                }
            }
            else
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
            for (; kk < max_kk; kk += 1)
            {
                sum0 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);
                sum1 += bfloat16_to_float32(pA[1]) * bfloat16_to_float32(pB[0]);
                pA += 2;
                pB += 1;
            }

            if (k_end)
            {
                sum0 = activation_ss(sum0, activation_type, activation_params);
                sum1 = activation_ss(sum1, activation_type, activation_params);

                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(sum0);
                    outptr0[out_hstep] = float32_to_bfloat16(sum1);
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
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
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

            if (k_end)
            {
                if (pC)
                {
                    _sum0 = _mm512_add_ps(_sum0, _mm512_set1_ps(pC[0]));
                }

                _sum0 = activation_avx512(_sum0, activation_type, activation_params);

                // if (out_elempack == 1)
                {
                    _mm256_storeu_si256((__m256i*)outptr0, float2bfloat_avx512(_sum0));
                    outptr0 += 16;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
            }

            outptr += 16;
        }
#endif // __AVX512F__
#if __AVX__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

            __m256 _sum0;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_ps();
            }
            else
            {
                _sum0 = _mm256_load_ps(outptr);
            }

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
                __m256 _pA0 = _mm256_set1_ps(bfloat16_to_float32(pA[0]));
                __m256 _pB0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pB));

                _sum0 = _mm256_comp_fmadd_ps(_pA0, _pB0, _sum0);

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                if (pC)
                {
                    _sum0 = _mm256_add_ps(_sum0, _mm256_set1_ps(pC[0]));
                }

                _sum0 = activation_avx(_sum0, activation_type, activation_params);

                // if (out_elempack == 1)
                {
                    _mm_storeu_si128((__m128i*)outptr0, float2bfloat_sse(_mm256_extractf128_ps(_sum0, 0), _mm256_extractf128_ps(_sum0, 1)));
                    outptr0 += 8;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
            }

            outptr += 8;
        }
#else  // __AVX__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

            float _sum0[8] = {0.f};

            if (k == 0)
            {
                if (pC)
                {
                    for (int i = 0; i < 8; i++)
                    {
                        _sum0[i] = pC[0];
                    }
                }
            }
            else
            {
                for (int i = 0; i < 8; i++)
                {
                    _sum0[i] = ((const float*)outptr)[i];
                }
            }

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                float pA0 = bfloat16_to_float32(pA[0]);
                for (int i = 0; i < 8; i++)
                {
                    float pBi = bfloat16_to_float32(pB[i]);
                    _sum0[i] += pA0 * pBi;
                }
                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                for (int i = 0; i < 8; i++)
                {
                    _sum0[i] = activation_ss(_sum0[i], activation_type, activation_params);
                }
                if (out_elempack == 1)
                {
                    for (int i = 0; i < 8; i++)
                    {
                        outptr0[i] = float32_to_bfloat16(_sum0[i]);
                    }
                    outptr0 += 8;
                }
            }
            else
            {
                for (int i = 0; i < 8; i++)
                {
                    ((float*)outptr)[i] = _sum0[i];
                }
            }

            outptr += 8;
        }
#endif // __AVX__
#endif // defined(__x86_64__) || defined(_M_X64)

        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

            __m128 _sum0;

            if (k == 0)
            {
                _sum0 = _mm_setzero_ps();
            }
            else
            {
                _sum0 = _mm_loadu_ps(outptr);
            }

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

            if (k_end)
            {
                if (pC)
                {
                    _sum0 = _mm_add_ps(_sum0, _mm_set1_ps(pC[0]));
                }

                _sum0 = activation_sse(_sum0, activation_type, activation_params);

                // if (out_elempack == 1)
                {
                    _mm_storel_epi64((__m128i*)outptr0, float2bfloat_sse(_sum0));
                    outptr0 += 4;
                }
            }
            else
            {
                _mm_storeu_ps(outptr, _sum0);
            }

            outptr += 4;
        }
#endif // __SSE2__
        for (; jj < max_jj; jj += 1)
        {
            float sum;

            if (k == 0)
            {
                if (pC)
                {
                    sum = pC[0];
                }
                else
                {
                    sum = 0.f;
                }
            }
            else
            {
                sum = outptr[0];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                float a0 = bfloat16_to_float32(pA[0]);
                float a1 = bfloat16_to_float32(pA[1]);
                float b0 = bfloat16_to_float32(pB[0]);
                float b1 = bfloat16_to_float32(pB[1]);
                sum += a0 * b0 + a1 * b1;
                pA += 2;
                pB += 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk += 1)
            {
                sum += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);
                pA += 1;
                pB += 1;
            }

            if (k_end)
            {
                sum = activation_ss(sum, activation_type, activation_params);

                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(sum);
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

static void convolution_im2col_gemm_get_optimal_tile_mnk_bf16s(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const int l2_cache_size_fp32 = (int)(get_cpu_level2_cache_size() / sizeof(unsigned short));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve K
    {
        // try not to split K
#if __AVX512F__
        int tile_size = (l2_cache_size_fp32 - 64) / 16;
#elif __AVX__
        int tile_size = (l2_cache_size_fp32 - 32) / 8;
#elif __SSE2__
        int tile_size = (l2_cache_size_fp32 - 16) / 8;
#else
        int tile_size = (l2_cache_size_fp32 - 2) / 3;
#endif

#if __AVX512F__
        TILE_K = std::max(16, tile_size / 16 * 16);
#elif __AVX__
        TILE_K = std::max(8, tile_size / 8 * 8);
#elif __SSE2__
        TILE_K = std::max(4, tile_size / 4 * 4);
#else
        TILE_K = std::max(2, tile_size / 2 * 2);
#endif

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
    }

    // solve M
    {
#if __AVX512F__
        int nn_M = (M + 63) / 64;
#elif __AVX__
        int nn_M = (M + 31) / 32;
#elif __SSE2__
        int nn_M = (M + 15) / 16;
#else
        int nn_M = (M + 7) / 8;
#endif

#if __AVX512F__
        TILE_M = std::max(16, ((M + nn_M - 1) / nn_M + 15) / 16 * 16);
#elif __AVX__
        TILE_M = std::max(8, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#elif __SSE2__
        TILE_M = std::max(4, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#else
        TILE_M = std::max(2, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    {
        TILE_M *= std::min(nT, get_physical_cpu_count());

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
    }

    if (N > 0)
    {
        int tile_size;
        if (TILE_K >= K)
        {
            tile_size = (l2_cache_size_fp32 - TILE_M * TILE_K) / TILE_K;
        }
        else
        {
            tile_size = (l2_cache_size_fp32 - TILE_M * TILE_K) / (TILE_M + TILE_K);
        }

#if __AVX512F__
        TILE_N = std::max(4, tile_size / 4 * 4);
#elif __AVX__
        TILE_N = std::max(4, tile_size / 4 * 4);
#elif __SSE2__
        TILE_N = std::max(4, tile_size / 4 * 4);
#else
        TILE_N = std::max(1, tile_size);
#endif

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

#if __AVX512F__
        TILE_N = std::max(4, TILE_N);
#elif __AVX__
        TILE_N = std::max(4, TILE_N);
#elif __SSE2__
        TILE_N = std::max(4, TILE_N);
#else
        TILE_N = std::max(1, TILE_N);
#endif
    }
}

static void convolution_im2col_input_tile_conv1x1s1d1_bf16s(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    const int elempack = bottom_blob.elempack;

    unsigned short* pp = B;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
#if __AVX__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 16) + (j + jj) * 16;

            int kk = 0;
            for (; kk < max_kk / 16; kk++)
            {
                __m512 _r0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0)));
                __m512 _r1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16)));
                __m512 _r2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 2)));
                __m512 _r3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 3)));
                __m512 _r4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 4)));
                __m512 _r5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 5)));
                __m512 _r6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 6)));
                __m512 _r7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 7)));
                __m512 _r8 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 8)));
                __m512 _r9 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 9)));
                __m512 _ra = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 10)));
                __m512 _rb = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 11)));
                __m512 _rc = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 12)));
                __m512 _rd = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 13)));
                __m512 _re = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 14)));
                __m512 _rf = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 15)));
                transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
#if __AVX512BF16__
                __m256i _t0 = float2bfloat_avx512(_r0);
                __m128i _t0l = _mm256_extracti128_si256(_t0, 0);
                __m128i _t0h = _mm256_extracti128_si256(_t0, 1);
                _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_t0l, _t0h));
                _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_t0l, _t0h));
                __m256i _t1 = float2bfloat_avx512(_r1);
                __m128i _t1l = _mm256_extracti128_si256(_t1, 0);
                __m128i _t1h = _mm256_extracti128_si256(_t1, 1);
                _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_t1l, _t1h));
                _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpackhi_epi16(_t1l, _t1h));
                __m256i _t2 = float2bfloat_avx512(_r2);
                __m128i _t2l = _mm256_extracti128_si256(_t2, 0);
                __m128i _t2h = _mm256_extracti128_si256(_t2, 1);
                _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi16(_t2l, _t2h));
                _mm_storeu_si128((__m128i*)(pp + 40), _mm_unpackhi_epi16(_t2l, _t2h));
                __m256i _t3 = float2bfloat_avx512(_r3);
                __m128i _t3l = _mm256_extracti128_si256(_t3, 0);
                __m128i _t3h = _mm256_extracti128_si256(_t3, 1);
                _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpacklo_epi16(_t3l, _t3h));
                _mm_storeu_si128((__m128i*)(pp + 56), _mm_unpackhi_epi16(_t3l, _t3h));
                __m256i _t4 = float2bfloat_avx512(_r4);
                __m128i _t4l = _mm256_extracti128_si256(_t4, 0);
                __m128i _t4h = _mm256_extracti128_si256(_t4, 1);
                _mm_storeu_si128((__m128i*)(pp + 64), _mm_unpacklo_epi16(_t4l, _t4h));
                _mm_storeu_si128((__m128i*)(pp + 72), _mm_unpackhi_epi16(_t4l, _t4h));
                __m256i _t5 = float2bfloat_avx512(_r5);
                __m128i _t5l = _mm256_extracti128_si256(_t5, 0);
                __m128i _t5h = _mm256_extracti128_si256(_t5, 1);
                _mm_storeu_si128((__m128i*)(pp + 80), _mm_unpacklo_epi16(_t5l, _t5h));
                _mm_storeu_si128((__m128i*)(pp + 88), _mm_unpackhi_epi16(_t5l, _t5h));
                __m256i _t6 = float2bfloat_avx512(_r6);
                __m128i _t6l = _mm256_extracti128_si256(_t6, 0);
                __m128i _t6h = _mm256_extracti128_si256(_t6, 1);
                _mm_storeu_si128((__m128i*)(pp + 96), _mm_unpacklo_epi16(_t6l, _t6h));
                _mm_storeu_si128((__m128i*)(pp + 104), _mm_unpackhi_epi16(_t6l, _t6h));
                __m256i _t7 = float2bfloat_avx512(_r7);
                __m128i _t7l = _mm256_extracti128_si256(_t7, 0);
                __m128i _t7h = _mm256_extracti128_si256(_t7, 1);
                _mm_storeu_si128((__m128i*)(pp + 112), _mm_unpacklo_epi16(_t7l, _t7h));
                _mm_storeu_si128((__m128i*)(pp + 120), _mm_unpackhi_epi16(_t7l, _t7h));
                __m256i _t8 = float2bfloat_avx512(_r8);
                __m128i _t8l = _mm256_extracti128_si256(_t8, 0);
                __m128i _t8h = _mm256_extracti128_si256(_t8, 1);
                _mm_storeu_si128((__m128i*)(pp + 128), _mm_unpacklo_epi16(_t8l, _t8h));
                _mm_storeu_si128((__m128i*)(pp + 136), _mm_unpackhi_epi16(_t8l, _t8h));
                __m256i _t9 = float2bfloat_avx512(_r9);
                __m128i _t9l = _mm256_extracti128_si256(_t9, 0);
                __m128i _t9h = _mm256_extracti128_si256(_t9, 1);
                _mm_storeu_si128((__m128i*)(pp + 144), _mm_unpacklo_epi16(_t9l, _t9h));
                _mm_storeu_si128((__m128i*)(pp + 152), _mm_unpackhi_epi16(_t9l, _t9h));
                __m256i _ta = float2bfloat_avx512(_ra);
                __m128i _tal = _mm256_extracti128_si256(_ta, 0);
                __m128i _tah = _mm256_extracti128_si256(_ta, 1);
                _mm_storeu_si128((__m128i*)(pp + 160), _mm_unpacklo_epi16(_tal, _tah));
                _mm_storeu_si128((__m128i*)(pp + 168), _mm_unpackhi_epi16(_tal, _tah));
                __m256i _tb = float2bfloat_avx512(_rb);
                __m128i _tbl = _mm256_extracti128_si256(_tb, 0);
                __m128i _tbh = _mm256_extracti128_si256(_tb, 1);
                _mm_storeu_si128((__m128i*)(pp + 176), _mm_unpacklo_epi16(_tbl, _tbh));
                _mm_storeu_si128((__m128i*)(pp + 184), _mm_unpackhi_epi16(_tbl, _tbh));
                __m256i _tc = float2bfloat_avx512(_rc);
                __m128i _tcl = _mm256_extracti128_si256(_tc, 0);
                __m128i _tch = _mm256_extracti128_si256(_tc, 1);
                _mm_storeu_si128((__m128i*)(pp + 192), _mm_unpacklo_epi16(_tcl, _tch));
                _mm_storeu_si128((__m128i*)(pp + 200), _mm_unpackhi_epi16(_tcl, _tch));
                __m256i _td = float2bfloat_avx512(_rd);
                __m128i _tdl = _mm256_extracti128_si256(_td, 0);
                __m128i _tdh = _mm256_extracti128_si256(_td, 1);
                _mm_storeu_si128((__m128i*)(pp + 208), _mm_unpacklo_epi16(_tdl, _tdh));
                _mm_storeu_si128((__m128i*)(pp + 216), _mm_unpackhi_epi16(_tdl, _tdh));
                __m256i _te = float2bfloat_avx512(_re);
                __m128i _tel = _mm256_extracti128_si256(_te, 0);
                __m128i _teh = _mm256_extracti128_si256(_te, 1);
                _mm_storeu_si128((__m128i*)(pp + 224), _mm_unpacklo_epi16(_tel, _teh));
                _mm_storeu_si128((__m128i*)(pp + 232), _mm_unpackhi_epi16(_tel, _teh));
                __m256i _tf = float2bfloat_avx512(_rf);
                __m128i _tfl = _mm256_extracti128_si256(_tf, 0);
                __m128i _tfh = _mm256_extracti128_si256(_tf, 1);
                _mm_storeu_si128((__m128i*)(pp + 240), _mm_unpacklo_epi16(_tfl, _tfh));
                _mm_storeu_si128((__m128i*)(pp + 248), _mm_unpackhi_epi16(_tfl, _tfh));
#else

                _mm256_storeu_si256((__m256i*)pp, float2bfloat_avx512(_r0));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 1), float2bfloat_avx512(_r1));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 2), float2bfloat_avx512(_r2));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 3), float2bfloat_avx512(_r3));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 4), float2bfloat_avx512(_r4));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 5), float2bfloat_avx512(_r5));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 6), float2bfloat_avx512(_r6));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 7), float2bfloat_avx512(_r7));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 8), float2bfloat_avx512(_r8));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 9), float2bfloat_avx512(_r9));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 10), float2bfloat_avx512(_ra));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 11), float2bfloat_avx512(_rb));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 12), float2bfloat_avx512(_rc));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 13), float2bfloat_avx512(_rd));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 14), float2bfloat_avx512(_re));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 15), float2bfloat_avx512(_rf));
#endif // __AVX512BF16__
                pp += 256;
                p0 += bottom_blob.cstep * 16;
            }
        }
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;
            const unsigned short* p1 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj + 8) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0)));
                __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8)));
                __m256 _r2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 2)));
                __m256 _r3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 3)));
                __m256 _r4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 4)));
                __m256 _r5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 5)));
                __m256 _r6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 6)));
                __m256 _r7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 7)));
                __m256 _r8 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p1)));
                __m256 _r9 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p1 + 8)));
                __m256 _ra = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p1 + 8 * 2)));
                __m256 _rb = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p1 + 8 * 3)));
                __m256 _rc = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p1 + 8 * 4)));
                __m256 _rd = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p1 + 8 * 5)));
                __m256 _re = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p1 + 8 * 6)));
                __m256 _rf = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p1 + 8 * 7)));
                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                transpose8x8_ps(_r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
#if __AVX512BF16__
                __m128i _t0 = float2bfloat_avx(_r0);
                __m128i _t8 = float2bfloat_avx(_r8);
                _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_t0, _t8));
                _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_t0, _t8));
                __m128i _t1 = float2bfloat_avx(_r1);
                __m128i _t9 = float2bfloat_avx(_r9);
                _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_t1, _t9));
                _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpackhi_epi16(_t1, _t9));
                __m128i _t2 = float2bfloat_avx(_r2);
                __m128i _ta = float2bfloat_avx(_ra);
                _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi16(_t2, _ta));
                _mm_storeu_si128((__m128i*)(pp + 40), _mm_unpackhi_epi16(_t2, _ta));
                __m128i _t3 = float2bfloat_avx(_r3);
                __m128i _tb = float2bfloat_avx(_rb);
                _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpacklo_epi16(_t3, _tb));
                _mm_storeu_si128((__m128i*)(pp + 56), _mm_unpackhi_epi16(_t3, _tb));
                __m128i _t4 = float2bfloat_avx(_r4);
                __m128i _tc = float2bfloat_avx(_rc);
                _mm_storeu_si128((__m128i*)(pp + 64), _mm_unpacklo_epi16(_t4, _tc));
                _mm_storeu_si128((__m128i*)(pp + 72), _mm_unpackhi_epi16(_t4, _tc));
                __m128i _t5 = float2bfloat_avx(_r5);
                __m128i _td = float2bfloat_avx(_rd);
                _mm_storeu_si128((__m128i*)(pp + 80), _mm_unpacklo_epi16(_t5, _td));
                _mm_storeu_si128((__m128i*)(pp + 88), _mm_unpackhi_epi16(_t5, _td));
                __m128i _t6 = float2bfloat_avx(_r6);
                __m128i _te = float2bfloat_avx(_re);
                _mm_storeu_si128((__m128i*)(pp + 96), _mm_unpacklo_epi16(_t6, _te));
                _mm_storeu_si128((__m128i*)(pp + 104), _mm_unpackhi_epi16(_t6, _te));
                __m128i _t7 = float2bfloat_avx(_r7);
                __m128i _tf = float2bfloat_avx(_rf);
                _mm_storeu_si128((__m128i*)(pp + 112), _mm_unpacklo_epi16(_t7, _tf));
                _mm_storeu_si128((__m128i*)(pp + 120), _mm_unpackhi_epi16(_t7, _tf));
#else

                _mm_storeu_si128((__m128i*)pp, float2bfloat_avx(_r0));
                _mm_storeu_si128((__m128i*)(pp + 8 * 1), float2bfloat_avx(_r8));
                _mm_storeu_si128((__m128i*)(pp + 8 * 2), float2bfloat_avx(_r1));
                _mm_storeu_si128((__m128i*)(pp + 8 * 3), float2bfloat_avx(_r9));
                _mm_storeu_si128((__m128i*)(pp + 8 * 4), float2bfloat_avx(_r2));
                _mm_storeu_si128((__m128i*)(pp + 8 * 5), float2bfloat_avx(_ra));
                _mm_storeu_si128((__m128i*)(pp + 8 * 6), float2bfloat_avx(_r3));
                _mm_storeu_si128((__m128i*)(pp + 8 * 7), float2bfloat_avx(_rb));
                _mm_storeu_si128((__m128i*)(pp + 8 * 8), float2bfloat_avx(_r4));
                _mm_storeu_si128((__m128i*)(pp + 8 * 9), float2bfloat_avx(_rc));
                _mm_storeu_si128((__m128i*)(pp + 8 * 10), float2bfloat_avx(_r5));
                _mm_storeu_si128((__m128i*)(pp + 8 * 11), float2bfloat_avx(_rd));
                _mm_storeu_si128((__m128i*)(pp + 8 * 12), float2bfloat_avx(_r6));
                _mm_storeu_si128((__m128i*)(pp + 8 * 13), float2bfloat_avx(_re));
                _mm_storeu_si128((__m128i*)(pp + 8 * 14), float2bfloat_avx(_r7));
                _mm_storeu_si128((__m128i*)(pp + 8 * 15), float2bfloat_avx(_rf));
#endif // __AVX512BF16__
                pp += 128;
                p0 += bottom_blob.cstep * 8;
                p1 += bottom_blob.cstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                __m128 _r0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                __m128 _r1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4)));
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
                __m128 _rc = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 12)));
                __m128 _rd = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 13)));
                __m128 _re = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 14)));
                __m128 _rf = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 15)));
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);
                _MM_TRANSPOSE4_PS(_rc, _rd, _re, _rf);
#if __AVX512BF16__
                __m128i _tk0 = _mm_unpacklo_epi64(float2bfloat_sse(_r0), float2bfloat_sse(_r4));
                __m128i _tk8 = _mm_unpacklo_epi64(float2bfloat_sse(_r8), float2bfloat_sse(_rc));
                _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_tk0, _tk8));
                _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_tk0, _tk8));
                __m128i _tk1 = _mm_unpacklo_epi64(float2bfloat_sse(_r1), float2bfloat_sse(_r5));
                __m128i _tk9 = _mm_unpacklo_epi64(float2bfloat_sse(_r9), float2bfloat_sse(_rd));
                _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_tk1, _tk9));
                _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpackhi_epi16(_tk1, _tk9));
                __m128i _tk2 = _mm_unpacklo_epi64(float2bfloat_sse(_r2), float2bfloat_sse(_r6));
                __m128i _tka = _mm_unpacklo_epi64(float2bfloat_sse(_ra), float2bfloat_sse(_re));
                _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi16(_tk2, _tka));
                _mm_storeu_si128((__m128i*)(pp + 40), _mm_unpackhi_epi16(_tk2, _tka));
                __m128i _tk3 = _mm_unpacklo_epi64(float2bfloat_sse(_r3), float2bfloat_sse(_r7));
                __m128i _tkb = _mm_unpacklo_epi64(float2bfloat_sse(_rb), float2bfloat_sse(_rf));
                _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpacklo_epi16(_tk3, _tkb));
                _mm_storeu_si128((__m128i*)(pp + 56), _mm_unpackhi_epi16(_tk3, _tkb));
#else

                _mm_storel_epi64((__m128i*)pp, float2bfloat_sse(_r0));
                _mm_storel_epi64((__m128i*)(pp + 4 * 1), float2bfloat_sse(_r4));
                _mm_storel_epi64((__m128i*)(pp + 4 * 2), float2bfloat_sse(_r8));
                _mm_storel_epi64((__m128i*)(pp + 4 * 3), float2bfloat_sse(_rc));
                _mm_storel_epi64((__m128i*)(pp + 4 * 4), float2bfloat_sse(_r1));
                _mm_storel_epi64((__m128i*)(pp + 4 * 5), float2bfloat_sse(_r5));
                _mm_storel_epi64((__m128i*)(pp + 4 * 6), float2bfloat_sse(_r9));
                _mm_storel_epi64((__m128i*)(pp + 4 * 7), float2bfloat_sse(_rd));
                _mm_storel_epi64((__m128i*)(pp + 4 * 8), float2bfloat_sse(_r2));
                _mm_storel_epi64((__m128i*)(pp + 4 * 9), float2bfloat_sse(_r6));
                _mm_storel_epi64((__m128i*)(pp + 4 * 10), float2bfloat_sse(_ra));
                _mm_storel_epi64((__m128i*)(pp + 4 * 11), float2bfloat_sse(_re));
                _mm_storel_epi64((__m128i*)(pp + 4 * 12), float2bfloat_sse(_r3));
                _mm_storel_epi64((__m128i*)(pp + 4 * 13), float2bfloat_sse(_r7));
                _mm_storel_epi64((__m128i*)(pp + 4 * 14), float2bfloat_sse(_rb));
                _mm_storel_epi64((__m128i*)(pp + 4 * 15), float2bfloat_sse(_rf));
#endif // __AVX512BF16__
                pp += 64;
                p0 += bottom_blob.cstep * 4;
            }
        }

        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _r0 = _mm256_loadu_si256((const __m256i*)(p0));
                __m256i _r1 = _mm256_loadu_si256((const __m256i*)(p0 + bottom_blob.cstep));
                __m256i _t0 = _mm256_unpacklo_epi16(_r0, _r1);
                __m256i _t1 = _mm256_unpackhi_epi16(_r0, _r1);
                _mm256_storeu_si256((__m256i*)pp, _t0);
                _mm256_storeu_si256((__m256i*)(pp + 16), _t1);
                pp += 32;
                p0 += bottom_blob.cstep * 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)(p0)));
                _mm_storeu_si128((__m128i*)(pp + 8), _mm_loadu_si128((const __m128i*)(p0 + 8)));
                pp += 16;
                p0 += bottom_blob.cstep;
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
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 16) + (j + jj) * 16;

            int kk = 0;
            for (; kk < max_kk / 16; kk++)
            {
                __m512 _r0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0)));
                __m512 _r1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16)));
                __m512 _r2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 2)));
                __m512 _r3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 3)));
                __m512 _r4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 4)));
                __m512 _r5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 5)));
                __m512 _r6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 6)));
                __m512 _r7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 7)));
                transpose16x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
#if __AVX512BF16__
                __m256i _t0 = float2bfloat_avx512(_r0);
                __m128i _t0l = _mm256_extracti128_si256(_t0, 0);
                __m128i _t0h = _mm256_extracti128_si256(_t0, 1);
                _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_t0l, _t0h));
                _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_t0l, _t0h));
                __m256i _t1 = float2bfloat_avx512(_r1);
                __m128i _t1l = _mm256_extracti128_si256(_t1, 0);
                __m128i _t1h = _mm256_extracti128_si256(_t1, 1);
                _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_t1l, _t1h));
                _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpackhi_epi16(_t1l, _t1h));
                __m256i _t2 = float2bfloat_avx512(_r2);
                __m128i _t2l = _mm256_extracti128_si256(_t2, 0);
                __m128i _t2h = _mm256_extracti128_si256(_t2, 1);
                _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi16(_t2l, _t2h));
                _mm_storeu_si128((__m128i*)(pp + 40), _mm_unpackhi_epi16(_t2l, _t2h));
                __m256i _t3 = float2bfloat_avx512(_r3);
                __m128i _t3l = _mm256_extracti128_si256(_t3, 0);
                __m128i _t3h = _mm256_extracti128_si256(_t3, 1);
                _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpacklo_epi16(_t3l, _t3h));
                _mm_storeu_si128((__m128i*)(pp + 56), _mm_unpackhi_epi16(_t3l, _t3h));
                __m256i _t4 = float2bfloat_avx512(_r4);
                __m128i _t4l = _mm256_extracti128_si256(_t4, 0);
                __m128i _t4h = _mm256_extracti128_si256(_t4, 1);
                _mm_storeu_si128((__m128i*)(pp + 64), _mm_unpacklo_epi16(_t4l, _t4h));
                _mm_storeu_si128((__m128i*)(pp + 72), _mm_unpackhi_epi16(_t4l, _t4h));
                __m256i _t5 = float2bfloat_avx512(_r5);
                __m128i _t5l = _mm256_extracti128_si256(_t5, 0);
                __m128i _t5h = _mm256_extracti128_si256(_t5, 1);
                _mm_storeu_si128((__m128i*)(pp + 80), _mm_unpacklo_epi16(_t5l, _t5h));
                _mm_storeu_si128((__m128i*)(pp + 88), _mm_unpackhi_epi16(_t5l, _t5h));
                __m256i _t6 = float2bfloat_avx512(_r6);
                __m128i _t6l = _mm256_extracti128_si256(_t6, 0);
                __m128i _t6h = _mm256_extracti128_si256(_t6, 1);
                _mm_storeu_si128((__m128i*)(pp + 96), _mm_unpacklo_epi16(_t6l, _t6h));
                _mm_storeu_si128((__m128i*)(pp + 104), _mm_unpackhi_epi16(_t6l, _t6h));
                __m256i _t7 = float2bfloat_avx512(_r7);
                __m128i _t7l = _mm256_extracti128_si256(_t7, 0);
                __m128i _t7h = _mm256_extracti128_si256(_t7, 1);
                _mm_storeu_si128((__m128i*)(pp + 112), _mm_unpacklo_epi16(_t7l, _t7h));
                _mm_storeu_si128((__m128i*)(pp + 120), _mm_unpackhi_epi16(_t7l, _t7h));
#else

                _mm256_storeu_si256((__m256i*)pp, float2bfloat_avx512(_r0));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 1), float2bfloat_avx512(_r1));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 2), float2bfloat_avx512(_r2));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 3), float2bfloat_avx512(_r3));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 4), float2bfloat_avx512(_r4));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 5), float2bfloat_avx512(_r5));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 6), float2bfloat_avx512(_r6));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 7), float2bfloat_avx512(_r7));
#endif // __AVX512BF16__
                pp += 128;
                p0 += bottom_blob.cstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0)));
                __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8)));
                __m256 _r2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 2)));
                __m256 _r3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 3)));
                __m256 _r4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 4)));
                __m256 _r5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 5)));
                __m256 _r6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 6)));
                __m256 _r7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 7)));
                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
#if __AVX512BF16__
                __m128i _t0 = float2bfloat_avx(_r0);
                __m128i _t1 = float2bfloat_avx(_r1);
                _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_t0, _t1));
                _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_t0, _t1));
                __m128i _t2 = float2bfloat_avx(_r2);
                __m128i _t3 = float2bfloat_avx(_r3);
                _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_t2, _t3));
                _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpackhi_epi16(_t2, _t3));
                __m128i _t4 = float2bfloat_avx(_r4);
                __m128i _t5 = float2bfloat_avx(_r5);
                _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi16(_t4, _t5));
                _mm_storeu_si128((__m128i*)(pp + 40), _mm_unpackhi_epi16(_t4, _t5));
                __m128i _t6 = float2bfloat_avx(_r6);
                __m128i _t7 = float2bfloat_avx(_r7);
                _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpacklo_epi16(_t6, _t7));
                _mm_storeu_si128((__m128i*)(pp + 56), _mm_unpackhi_epi16(_t6, _t7));
#else

                _mm_storeu_si128((__m128i*)pp, float2bfloat_avx(_r0));
                _mm_storeu_si128((__m128i*)(pp + 8 * 1), float2bfloat_avx(_r1));
                _mm_storeu_si128((__m128i*)(pp + 8 * 2), float2bfloat_avx(_r2));
                _mm_storeu_si128((__m128i*)(pp + 8 * 3), float2bfloat_avx(_r3));
                _mm_storeu_si128((__m128i*)(pp + 8 * 4), float2bfloat_avx(_r4));
                _mm_storeu_si128((__m128i*)(pp + 8 * 5), float2bfloat_avx(_r5));
                _mm_storeu_si128((__m128i*)(pp + 8 * 6), float2bfloat_avx(_r6));
                _mm_storeu_si128((__m128i*)(pp + 8 * 7), float2bfloat_avx(_r7));
#endif // __AVX512BF16__
                pp += 64;
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                __m128 _r0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                __m128 _r1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4)));
                __m128 _r2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 2)));
                __m128 _r3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 3)));
                __m128 _r4 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 4)));
                __m128 _r5 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 5)));
                __m128 _r6 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 6)));
                __m128 _r7 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 7)));
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
#if __AVX512BF16__
                __m128i _tk0 = _mm_unpacklo_epi64(float2bfloat_sse(_r0), float2bfloat_sse(_r4));
                __m128i _tk1 = _mm_unpacklo_epi64(float2bfloat_sse(_r1), float2bfloat_sse(_r5));
                _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_tk0, _tk1));
                _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_tk0, _tk1));
                __m128i _tk2 = _mm_unpacklo_epi64(float2bfloat_sse(_r2), float2bfloat_sse(_r6));
                __m128i _tk3 = _mm_unpacklo_epi64(float2bfloat_sse(_r3), float2bfloat_sse(_r7));
                _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_tk2, _tk3));
                _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpackhi_epi16(_tk2, _tk3));
#else

                _mm_storel_epi64((__m128i*)pp, float2bfloat_sse(_r0));
                _mm_storel_epi64((__m128i*)(pp + 4 * 1), float2bfloat_sse(_r4));
                _mm_storel_epi64((__m128i*)(pp + 4 * 2), float2bfloat_sse(_r1));
                _mm_storel_epi64((__m128i*)(pp + 4 * 3), float2bfloat_sse(_r5));
                _mm_storel_epi64((__m128i*)(pp + 4 * 4), float2bfloat_sse(_r2));
                _mm_storel_epi64((__m128i*)(pp + 4 * 5), float2bfloat_sse(_r6));
                _mm_storel_epi64((__m128i*)(pp + 4 * 6), float2bfloat_sse(_r3));
                _mm_storel_epi64((__m128i*)(pp + 4 * 7), float2bfloat_sse(_r7));
#endif // __AVX512BF16__
                pp += 32;
                p0 += bottom_blob.cstep * 4;
            }
        }

        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)(p0));
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + bottom_blob.cstep));
                _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi16(_r0, _r1));
                _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_r0, _r1));
                pp += 16;
                p0 += bottom_blob.cstep * 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)(p0)));
                pp += 8;
                p0 += bottom_blob.cstep;
            }
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)

    for (; jj + 3 < max_jj; jj += 4)
    {
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 16) + (j + jj) * 16;

            int kk = 0;
            for (; kk < max_kk / 16; kk++)
            {
                __m512 _r0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0)));
                __m512 _r1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16)));
                __m512 _r2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 2)));
                __m512 _r3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(p0 + 16 * 3)));
                transpose16x4_ps(_r0, _r1, _r2, _r3);
#if __AVX512BF16__
                __m256i _t0 = float2bfloat_avx512(_r0);
                __m128i _t0a = _mm256_extracti128_si256(_t0, 0);
                __m128i _t0b = _mm256_extracti128_si256(_t0, 1);
                __m128i _t0k01_lo = _mm_unpacklo_epi16(_t0a, _mm_srli_si128(_t0a, 8));
                _mm_storeu_si128((__m128i*)(pp + 0), _t0k01_lo);
                __m128i _t0k23_lo = _mm_unpacklo_epi16(_t0b, _mm_srli_si128(_t0b, 8));
                _mm_storeu_si128((__m128i*)(pp + 8), _t0k23_lo);
                __m256i _t1 = float2bfloat_avx512(_r1);
                __m128i _t1a = _mm256_extracti128_si256(_t1, 0);
                __m128i _t1b = _mm256_extracti128_si256(_t1, 1);
                __m128i _t1k01_lo = _mm_unpacklo_epi16(_t1a, _mm_srli_si128(_t1a, 8));
                _mm_storeu_si128((__m128i*)(pp + 16), _t1k01_lo);
                __m128i _t1k23_lo = _mm_unpacklo_epi16(_t1b, _mm_srli_si128(_t1b, 8));
                _mm_storeu_si128((__m128i*)(pp + 24), _t1k23_lo);
                __m256i _t2 = float2bfloat_avx512(_r2);
                __m128i _t2a = _mm256_extracti128_si256(_t2, 0);
                __m128i _t2b = _mm256_extracti128_si256(_t2, 1);
                __m128i _t2k01_lo = _mm_unpacklo_epi16(_t2a, _mm_srli_si128(_t2a, 8));
                _mm_storeu_si128((__m128i*)(pp + 32), _t2k01_lo);
                __m128i _t2k23_lo = _mm_unpacklo_epi16(_t2b, _mm_srli_si128(_t2b, 8));
                _mm_storeu_si128((__m128i*)(pp + 40), _t2k23_lo);
                __m256i _t3 = float2bfloat_avx512(_r3);
                __m128i _t3a = _mm256_extracti128_si256(_t3, 0);
                __m128i _t3b = _mm256_extracti128_si256(_t3, 1);
                __m128i _t3k01_lo = _mm_unpacklo_epi16(_t3a, _mm_srli_si128(_t3a, 8));
                _mm_storeu_si128((__m128i*)(pp + 48), _t3k01_lo);
                __m128i _t3k23_lo = _mm_unpacklo_epi16(_t3b, _mm_srli_si128(_t3b, 8));
                _mm_storeu_si128((__m128i*)(pp + 56), _t3k23_lo);
#else

                _mm256_storeu_si256((__m256i*)pp, float2bfloat_avx512(_r0));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 1), float2bfloat_avx512(_r1));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 2), float2bfloat_avx512(_r2));
                _mm256_storeu_si256((__m256i*)(pp + 16 * 3), float2bfloat_avx512(_r3));
#endif // __AVX512BF16__
                pp += 64;
                p0 += bottom_blob.cstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0)));
                __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8)));
                __m256 _r2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 2)));
                __m256 _r3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(p0 + 8 * 3)));
                transpose8x4_ps(_r0, _r1, _r2, _r3);
#if __AVX512BF16__
                __m128i _t0 = float2bfloat_avx(_r0);
                _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_t0, _mm_srli_si128(_t0, 8)));
                __m128i _t1 = float2bfloat_avx(_r1);
                _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpacklo_epi16(_t1, _mm_srli_si128(_t1, 8)));
                __m128i _t2 = float2bfloat_avx(_r2);
                _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_t2, _mm_srli_si128(_t2, 8)));
                __m128i _t3 = float2bfloat_avx(_r3);
                _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpacklo_epi16(_t3, _mm_srli_si128(_t3, 8)));
#else

                _mm_storeu_si128((__m128i*)pp, float2bfloat_avx(_r0));
                _mm_storeu_si128((__m128i*)(pp + 8 * 1), float2bfloat_avx(_r1));
                _mm_storeu_si128((__m128i*)(pp + 8 * 2), float2bfloat_avx(_r2));
                _mm_storeu_si128((__m128i*)(pp + 8 * 3), float2bfloat_avx(_r3));
#endif // __AVX512BF16__
                pp += 32;
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                __m128 _r0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0)));
                __m128 _r1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4)));
                __m128 _r2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 2)));
                __m128 _r3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(p0 + 4 * 3)));
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
#if __AVX512BF16__
                __m128i _t0 = float2bfloat_sse(_r0);
                __m128i _t1 = float2bfloat_sse(_r1);
                _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_t0, _t1));
                __m128i _t2 = float2bfloat_sse(_r2);
                __m128i _t3 = float2bfloat_sse(_r3);
                _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpacklo_epi16(_t2, _t3));
#else

                _mm_storel_epi64((__m128i*)pp, float2bfloat_sse(_r0));
                _mm_storel_epi64((__m128i*)(pp + 4 * 1), float2bfloat_sse(_r1));
                _mm_storel_epi64((__m128i*)(pp + 4 * 2), float2bfloat_sse(_r2));
                _mm_storel_epi64((__m128i*)(pp + 4 * 3), float2bfloat_sse(_r3));
#endif // __AVX512BF16__
                pp += 16;
                p0 += bottom_blob.cstep * 4;
            }
        }

        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
#if __AVX512BF16__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)(p0));
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)(p0 + bottom_blob.cstep));
                _mm_storeu_si128((__m128i*)pp, _mm_unpacklo_epi16(_r0, _r1));
                pp += 8;
                p0 += bottom_blob.cstep * 2;
            }
#endif // __AVX512BF16__
            for (; kk < max_kk; kk++)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)(p0)));
                pp += 4;
                p0 += bottom_blob.cstep;
            }
        }
    }
#endif // __SSE2__
    for (; jj < max_jj; jj++)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 16) + (j + jj) * 16;

            int kk = 0;
            for (; kk < max_kk / 16; kk++)
            {
                _mm256_storeu_si256((__m256i*)pp, _mm256_loadu_si256((const __m256i*)(p0)));
                pp += 16;
                p0 += bottom_blob.cstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)(p0)));
                pp += 8;
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)(p0)));
                pp += 4;
                p0 += bottom_blob.cstep * 4;
            }
        }
#endif // __SSE2__

        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += bottom_blob.cstep;
            }
        }
    }
}

static inline void convolution_im2col_input_tile_impl_bf16s(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    const int w = bottom_blob.w;
    // const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int outw = (w - kernel_extent_w) / stride_w + 1;

    // j max_jj     outw*outh    split w and h

    // k max_kk     pa*maxk*(inch/pa)    split inch

    // k/max_kk shall be multiple of maxk

    const int maxk = kernel_w * kernel_h;

    unsigned short* pp = B;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dy4 = (j + jj + 4) / outw;
        int dy5 = (j + jj + 5) / outw;
        int dy6 = (j + jj + 6) / outw;
        int dy7 = (j + jj + 7) / outw;
        int dy8 = (j + jj + 8) / outw;
        int dy9 = (j + jj + 9) / outw;
        int dya = (j + jj + 10) / outw;
        int dyb = (j + jj + 11) / outw;
        int dyc = (j + jj + 12) / outw;
        int dyd = (j + jj + 13) / outw;
        int dye = (j + jj + 14) / outw;
        int dyf = (j + jj + 15) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;
        int dx4 = (j + jj + 4) % outw;
        int dx5 = (j + jj + 5) % outw;
        int dx6 = (j + jj + 6) % outw;
        int dx7 = (j + jj + 7) % outw;
        int dx8 = (j + jj + 8) % outw;
        int dx9 = (j + jj + 9) % outw;
        int dxa = (j + jj + 10) % outw;
        int dxb = (j + jj + 11) % outw;
        int dxc = (j + jj + 12) % outw;
        int dxd = (j + jj + 13) % outw;
        int dxe = (j + jj + 14) % outw;
        int dxf = (j + jj + 15) % outw;

        if (dy0 == dyf)
        {
            int kk = 0;
#if __AVX512BF16__
            if (elempack == 1)
            {
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int u0 = uv0 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    const Mat img0 = bottom_blob.channel(p0);
                    int sx0 = stride_w * dx0 + dilation_w * v0;
                    int sy0 = stride_h * dy0 + dilation_h * u0;
                    const unsigned short* sptr0 = img0.row<const unsigned short>(sy0) + sx0;

                    int p1 = (k + kk + 1) / maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u1 = uv1 / kernel_w;
                    int v1 = uv1 % kernel_w;
                    const Mat img1 = bottom_blob.channel(p1);
                    int sx1 = stride_w * dx0 + dilation_w * v1;
                    int sy1 = stride_h * dy0 + dilation_h * u1;
                    const unsigned short* sptr1 = img1.row<const unsigned short>(sy1) + sx1;

                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr0[stride_w];
                    pp[3] = sptr1[stride_w];
                    pp[4] = sptr0[stride_w * 2];
                    pp[5] = sptr1[stride_w * 2];
                    pp[6] = sptr0[stride_w * 3];
                    pp[7] = sptr1[stride_w * 3];
                    pp[8] = sptr0[stride_w * 4];
                    pp[9] = sptr1[stride_w * 4];
                    pp[10] = sptr0[stride_w * 5];
                    pp[11] = sptr1[stride_w * 5];
                    pp[12] = sptr0[stride_w * 6];
                    pp[13] = sptr1[stride_w * 6];
                    pp[14] = sptr0[stride_w * 7];
                    pp[15] = sptr1[stride_w * 7];
                    pp[16] = sptr0[stride_w * 8];
                    pp[17] = sptr1[stride_w * 8];
                    pp[18] = sptr0[stride_w * 9];
                    pp[19] = sptr1[stride_w * 9];
                    pp[20] = sptr0[stride_w * 10];
                    pp[21] = sptr1[stride_w * 10];
                    pp[22] = sptr0[stride_w * 11];
                    pp[23] = sptr1[stride_w * 11];
                    pp[24] = sptr0[stride_w * 12];
                    pp[25] = sptr1[stride_w * 12];
                    pp[26] = sptr0[stride_w * 13];
                    pp[27] = sptr1[stride_w * 13];
                    pp[28] = sptr0[stride_w * 14];
                    pp[29] = sptr1[stride_w * 14];
                    pp[30] = sptr0[stride_w * 15];
                    pp[31] = sptr1[stride_w * 15];
                    pp += 32;
                }
            }
#endif // __AVX512BF16__
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const unsigned short* sptr = img.row<const unsigned short>(y0) + x0 * elempack;

#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    __m512 _r0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr)));
                    __m512 _r1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 16)));
                    __m512 _r2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 32)));
                    __m512 _r3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 48)));
                    __m512 _r4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 64)));
                    __m512 _r5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 80)));
                    __m512 _r6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 96)));
                    __m512 _r7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 112)));
                    __m512 _r8 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 128)));
                    __m512 _r9 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 144)));
                    __m512 _ra = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 160)));
                    __m512 _rb = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 176)));
                    __m512 _rc = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 192)));
                    __m512 _rd = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 208)));
                    __m512 _re = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 224)));
                    __m512 _rf = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 240)));
                    transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
#if __AVX512BF16__
                    __m256i _t0 = float2bfloat_avx512(_r0);
                    __m128i _t0l = _mm256_extracti128_si256(_t0, 0);
                    __m128i _t0h = _mm256_extracti128_si256(_t0, 1);
                    _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_t0l, _t0h));
                    _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_t0l, _t0h));
                    __m256i _t1 = float2bfloat_avx512(_r1);
                    __m128i _t1l = _mm256_extracti128_si256(_t1, 0);
                    __m128i _t1h = _mm256_extracti128_si256(_t1, 1);
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_t1l, _t1h));
                    _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpackhi_epi16(_t1l, _t1h));
                    __m256i _t2 = float2bfloat_avx512(_r2);
                    __m128i _t2l = _mm256_extracti128_si256(_t2, 0);
                    __m128i _t2h = _mm256_extracti128_si256(_t2, 1);
                    _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi16(_t2l, _t2h));
                    _mm_storeu_si128((__m128i*)(pp + 40), _mm_unpackhi_epi16(_t2l, _t2h));
                    __m256i _t3 = float2bfloat_avx512(_r3);
                    __m128i _t3l = _mm256_extracti128_si256(_t3, 0);
                    __m128i _t3h = _mm256_extracti128_si256(_t3, 1);
                    _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpacklo_epi16(_t3l, _t3h));
                    _mm_storeu_si128((__m128i*)(pp + 56), _mm_unpackhi_epi16(_t3l, _t3h));
                    __m256i _t4 = float2bfloat_avx512(_r4);
                    __m128i _t4l = _mm256_extracti128_si256(_t4, 0);
                    __m128i _t4h = _mm256_extracti128_si256(_t4, 1);
                    _mm_storeu_si128((__m128i*)(pp + 64), _mm_unpacklo_epi16(_t4l, _t4h));
                    _mm_storeu_si128((__m128i*)(pp + 72), _mm_unpackhi_epi16(_t4l, _t4h));
                    __m256i _t5 = float2bfloat_avx512(_r5);
                    __m128i _t5l = _mm256_extracti128_si256(_t5, 0);
                    __m128i _t5h = _mm256_extracti128_si256(_t5, 1);
                    _mm_storeu_si128((__m128i*)(pp + 80), _mm_unpacklo_epi16(_t5l, _t5h));
                    _mm_storeu_si128((__m128i*)(pp + 88), _mm_unpackhi_epi16(_t5l, _t5h));
                    __m256i _t6 = float2bfloat_avx512(_r6);
                    __m128i _t6l = _mm256_extracti128_si256(_t6, 0);
                    __m128i _t6h = _mm256_extracti128_si256(_t6, 1);
                    _mm_storeu_si128((__m128i*)(pp + 96), _mm_unpacklo_epi16(_t6l, _t6h));
                    _mm_storeu_si128((__m128i*)(pp + 104), _mm_unpackhi_epi16(_t6l, _t6h));
                    __m256i _t7 = float2bfloat_avx512(_r7);
                    __m128i _t7l = _mm256_extracti128_si256(_t7, 0);
                    __m128i _t7h = _mm256_extracti128_si256(_t7, 1);
                    _mm_storeu_si128((__m128i*)(pp + 112), _mm_unpacklo_epi16(_t7l, _t7h));
                    _mm_storeu_si128((__m128i*)(pp + 120), _mm_unpackhi_epi16(_t7l, _t7h));
                    __m256i _t8 = float2bfloat_avx512(_r8);
                    __m128i _t8l = _mm256_extracti128_si256(_t8, 0);
                    __m128i _t8h = _mm256_extracti128_si256(_t8, 1);
                    _mm_storeu_si128((__m128i*)(pp + 128), _mm_unpacklo_epi16(_t8l, _t8h));
                    _mm_storeu_si128((__m128i*)(pp + 136), _mm_unpackhi_epi16(_t8l, _t8h));
                    __m256i _t9 = float2bfloat_avx512(_r9);
                    __m128i _t9l = _mm256_extracti128_si256(_t9, 0);
                    __m128i _t9h = _mm256_extracti128_si256(_t9, 1);
                    _mm_storeu_si128((__m128i*)(pp + 144), _mm_unpacklo_epi16(_t9l, _t9h));
                    _mm_storeu_si128((__m128i*)(pp + 152), _mm_unpackhi_epi16(_t9l, _t9h));
                    __m256i _ta = float2bfloat_avx512(_ra);
                    __m128i _tal = _mm256_extracti128_si256(_ta, 0);
                    __m128i _tah = _mm256_extracti128_si256(_ta, 1);
                    _mm_storeu_si128((__m128i*)(pp + 160), _mm_unpacklo_epi16(_tal, _tah));
                    _mm_storeu_si128((__m128i*)(pp + 168), _mm_unpackhi_epi16(_tal, _tah));
                    __m256i _tb = float2bfloat_avx512(_rb);
                    __m128i _tbl = _mm256_extracti128_si256(_tb, 0);
                    __m128i _tbh = _mm256_extracti128_si256(_tb, 1);
                    _mm_storeu_si128((__m128i*)(pp + 176), _mm_unpacklo_epi16(_tbl, _tbh));
                    _mm_storeu_si128((__m128i*)(pp + 184), _mm_unpackhi_epi16(_tbl, _tbh));
                    __m256i _tc = float2bfloat_avx512(_rc);
                    __m128i _tcl = _mm256_extracti128_si256(_tc, 0);
                    __m128i _tch = _mm256_extracti128_si256(_tc, 1);
                    _mm_storeu_si128((__m128i*)(pp + 192), _mm_unpacklo_epi16(_tcl, _tch));
                    _mm_storeu_si128((__m128i*)(pp + 200), _mm_unpackhi_epi16(_tcl, _tch));
                    __m256i _td = float2bfloat_avx512(_rd);
                    __m128i _tdl = _mm256_extracti128_si256(_td, 0);
                    __m128i _tdh = _mm256_extracti128_si256(_td, 1);
                    _mm_storeu_si128((__m128i*)(pp + 208), _mm_unpacklo_epi16(_tdl, _tdh));
                    _mm_storeu_si128((__m128i*)(pp + 216), _mm_unpackhi_epi16(_tdl, _tdh));
                    __m256i _te = float2bfloat_avx512(_re);
                    __m128i _tel = _mm256_extracti128_si256(_te, 0);
                    __m128i _teh = _mm256_extracti128_si256(_te, 1);
                    _mm_storeu_si128((__m128i*)(pp + 224), _mm_unpacklo_epi16(_tel, _teh));
                    _mm_storeu_si128((__m128i*)(pp + 232), _mm_unpackhi_epi16(_tel, _teh));
                    __m256i _tf = float2bfloat_avx512(_rf);
                    __m128i _tfl = _mm256_extracti128_si256(_tf, 0);
                    __m128i _tfh = _mm256_extracti128_si256(_tf, 1);
                    _mm_storeu_si128((__m128i*)(pp + 240), _mm_unpacklo_epi16(_tfl, _tfh));
                    _mm_storeu_si128((__m128i*)(pp + 248), _mm_unpackhi_epi16(_tfl, _tfh));
#else

                    _mm256_storeu_si256((__m256i*)pp, float2bfloat_avx512(_r0));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 1), float2bfloat_avx512(_r1));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 2), float2bfloat_avx512(_r2));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 3), float2bfloat_avx512(_r3));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 4), float2bfloat_avx512(_r4));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 5), float2bfloat_avx512(_r5));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 6), float2bfloat_avx512(_r6));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 7), float2bfloat_avx512(_r7));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 8), float2bfloat_avx512(_r8));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 9), float2bfloat_avx512(_r9));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 10), float2bfloat_avx512(_ra));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 11), float2bfloat_avx512(_rb));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 12), float2bfloat_avx512(_rc));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 13), float2bfloat_avx512(_rd));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 14), float2bfloat_avx512(_re));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 15), float2bfloat_avx512(_rf));
#endif // __AVX512BF16__
                    pp += 256;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr)));
                    __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 8)));
                    __m256 _r2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 16)));
                    __m256 _r3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 24)));
                    __m256 _r4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 32)));
                    __m256 _r5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 40)));
                    __m256 _r6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 48)));
                    __m256 _r7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 56)));
                    __m256 _r8 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 64)));
                    __m256 _r9 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 72)));
                    __m256 _ra = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 80)));
                    __m256 _rb = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 88)));
                    __m256 _rc = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 96)));
                    __m256 _rd = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 104)));
                    __m256 _re = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 112)));
                    __m256 _rf = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 120)));
                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                    transpose8x8_ps(_r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
#if __AVX512BF16__
                    __m128i _t0 = float2bfloat_avx(_r0);
                    __m128i _t8 = float2bfloat_avx(_r8);
                    _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_t0, _t8));
                    _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_t0, _t8));
                    __m128i _t1 = float2bfloat_avx(_r1);
                    __m128i _t9 = float2bfloat_avx(_r9);
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_t1, _t9));
                    _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpackhi_epi16(_t1, _t9));
                    __m128i _t2 = float2bfloat_avx(_r2);
                    __m128i _ta = float2bfloat_avx(_ra);
                    _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi16(_t2, _ta));
                    _mm_storeu_si128((__m128i*)(pp + 40), _mm_unpackhi_epi16(_t2, _ta));
                    __m128i _t3 = float2bfloat_avx(_r3);
                    __m128i _tb = float2bfloat_avx(_rb);
                    _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpacklo_epi16(_t3, _tb));
                    _mm_storeu_si128((__m128i*)(pp + 56), _mm_unpackhi_epi16(_t3, _tb));
                    __m128i _t4 = float2bfloat_avx(_r4);
                    __m128i _tc = float2bfloat_avx(_rc);
                    _mm_storeu_si128((__m128i*)(pp + 64), _mm_unpacklo_epi16(_t4, _tc));
                    _mm_storeu_si128((__m128i*)(pp + 72), _mm_unpackhi_epi16(_t4, _tc));
                    __m128i _t5 = float2bfloat_avx(_r5);
                    __m128i _td = float2bfloat_avx(_rd);
                    _mm_storeu_si128((__m128i*)(pp + 80), _mm_unpacklo_epi16(_t5, _td));
                    _mm_storeu_si128((__m128i*)(pp + 88), _mm_unpackhi_epi16(_t5, _td));
                    __m128i _t6 = float2bfloat_avx(_r6);
                    __m128i _te = float2bfloat_avx(_re);
                    _mm_storeu_si128((__m128i*)(pp + 96), _mm_unpacklo_epi16(_t6, _te));
                    _mm_storeu_si128((__m128i*)(pp + 104), _mm_unpackhi_epi16(_t6, _te));
                    __m128i _t7 = float2bfloat_avx(_r7);
                    __m128i _tf = float2bfloat_avx(_rf);
                    _mm_storeu_si128((__m128i*)(pp + 112), _mm_unpacklo_epi16(_t7, _tf));
                    _mm_storeu_si128((__m128i*)(pp + 120), _mm_unpackhi_epi16(_t7, _tf));
#else

                    _mm_storeu_si128((__m128i*)pp, float2bfloat_avx(_r0));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 1), float2bfloat_avx(_r8));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 2), float2bfloat_avx(_r1));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 3), float2bfloat_avx(_r9));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 4), float2bfloat_avx(_r2));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 5), float2bfloat_avx(_ra));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 6), float2bfloat_avx(_r3));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 7), float2bfloat_avx(_rb));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 8), float2bfloat_avx(_r4));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 9), float2bfloat_avx(_rc));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 10), float2bfloat_avx(_r5));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 11), float2bfloat_avx(_rd));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 12), float2bfloat_avx(_r6));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 13), float2bfloat_avx(_re));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 14), float2bfloat_avx(_r7));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 15), float2bfloat_avx(_rf));
#endif // __AVX512BF16__
                    pp += 128;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    __m128 _r0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr)));
                    __m128 _r1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 4)));
                    __m128 _r2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 8)));
                    __m128 _r3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 12)));
                    __m128 _r4 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 16)));
                    __m128 _r5 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 20)));
                    __m128 _r6 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 24)));
                    __m128 _r7 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 28)));
                    __m128 _r8 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 32)));
                    __m128 _r9 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 36)));
                    __m128 _ra = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 40)));
                    __m128 _rb = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 44)));
                    __m128 _rc = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 48)));
                    __m128 _rd = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 52)));
                    __m128 _re = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 56)));
                    __m128 _rf = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 60)));
                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                    _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);
                    _MM_TRANSPOSE4_PS(_rc, _rd, _re, _rf);
#if __AVX512BF16__
                    __m128i _tk0 = _mm_unpacklo_epi64(float2bfloat_sse(_r0), float2bfloat_sse(_r4));
                    __m128i _tk8 = _mm_unpacklo_epi64(float2bfloat_sse(_r8), float2bfloat_sse(_rc));
                    _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_tk0, _tk8));
                    _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_tk0, _tk8));
                    __m128i _tk1 = _mm_unpacklo_epi64(float2bfloat_sse(_r1), float2bfloat_sse(_r5));
                    __m128i _tk9 = _mm_unpacklo_epi64(float2bfloat_sse(_r9), float2bfloat_sse(_rd));
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_tk1, _tk9));
                    _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpackhi_epi16(_tk1, _tk9));
                    __m128i _tk2 = _mm_unpacklo_epi64(float2bfloat_sse(_r2), float2bfloat_sse(_r6));
                    __m128i _tka = _mm_unpacklo_epi64(float2bfloat_sse(_ra), float2bfloat_sse(_re));
                    _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi16(_tk2, _tka));
                    _mm_storeu_si128((__m128i*)(pp + 40), _mm_unpackhi_epi16(_tk2, _tka));
                    __m128i _tk3 = _mm_unpacklo_epi64(float2bfloat_sse(_r3), float2bfloat_sse(_r7));
                    __m128i _tkb = _mm_unpacklo_epi64(float2bfloat_sse(_rb), float2bfloat_sse(_rf));
                    _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpacklo_epi16(_tk3, _tkb));
                    _mm_storeu_si128((__m128i*)(pp + 56), _mm_unpackhi_epi16(_tk3, _tkb));
#else

                    _mm_storel_epi64((__m128i*)pp, float2bfloat_sse(_r0));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 1), float2bfloat_sse(_r4));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 2), float2bfloat_sse(_r8));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 3), float2bfloat_sse(_rc));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 4), float2bfloat_sse(_r1));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 5), float2bfloat_sse(_r5));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 6), float2bfloat_sse(_r9));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 7), float2bfloat_sse(_rd));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 8), float2bfloat_sse(_r2));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 9), float2bfloat_sse(_r6));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 10), float2bfloat_sse(_ra));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 11), float2bfloat_sse(_re));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 12), float2bfloat_sse(_r3));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 13), float2bfloat_sse(_r7));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 14), float2bfloat_sse(_rb));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 15), float2bfloat_sse(_rf));
#endif // __AVX512BF16__
                    pp += 64;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp[2] = sptr[stride_w * 2];
                    pp[3] = sptr[stride_w * 3];
                    pp[4] = sptr[stride_w * 4];
                    pp[5] = sptr[stride_w * 5];
                    pp[6] = sptr[stride_w * 6];
                    pp[7] = sptr[stride_w * 7];
                    pp[8] = sptr[stride_w * 8];
                    pp[9] = sptr[stride_w * 9];
                    pp[10] = sptr[stride_w * 10];
                    pp[11] = sptr[stride_w * 11];
                    pp[12] = sptr[stride_w * 12];
                    pp[13] = sptr[stride_w * 13];
                    pp[14] = sptr[stride_w * 14];
                    pp[15] = sptr[stride_w * 15];
                    pp += 16;
                }
            }
        }
        else
        {
            int kk = 0;
#if __AVX512BF16__
            if (elempack == 1)
            {
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int u0 = uv0 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    const Mat img0 = bottom_blob.channel(p0);

                    int p1 = (k + kk + 1) / maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u1 = uv1 / kernel_w;
                    int v1 = uv1 % kernel_w;
                    const Mat img1 = bottom_blob.channel(p1);

                    pp[0] = img0.row<const unsigned short>(stride_h * dy0 + dilation_h * u0)[stride_w * dx0 + dilation_w * v0];
                    pp[1] = img1.row<const unsigned short>(stride_h * dy0 + dilation_h * u1)[stride_w * dx0 + dilation_w * v1];
                    pp[2] = img0.row<const unsigned short>(stride_h * dy1 + dilation_h * u0)[stride_w * dx1 + dilation_w * v0];
                    pp[3] = img1.row<const unsigned short>(stride_h * dy1 + dilation_h * u1)[stride_w * dx1 + dilation_w * v1];
                    pp[4] = img0.row<const unsigned short>(stride_h * dy2 + dilation_h * u0)[stride_w * dx2 + dilation_w * v0];
                    pp[5] = img1.row<const unsigned short>(stride_h * dy2 + dilation_h * u1)[stride_w * dx2 + dilation_w * v1];
                    pp[6] = img0.row<const unsigned short>(stride_h * dy3 + dilation_h * u0)[stride_w * dx3 + dilation_w * v0];
                    pp[7] = img1.row<const unsigned short>(stride_h * dy3 + dilation_h * u1)[stride_w * dx3 + dilation_w * v1];
                    pp[8] = img0.row<const unsigned short>(stride_h * dy4 + dilation_h * u0)[stride_w * dx4 + dilation_w * v0];
                    pp[9] = img1.row<const unsigned short>(stride_h * dy4 + dilation_h * u1)[stride_w * dx4 + dilation_w * v1];
                    pp[10] = img0.row<const unsigned short>(stride_h * dy5 + dilation_h * u0)[stride_w * dx5 + dilation_w * v0];
                    pp[11] = img1.row<const unsigned short>(stride_h * dy5 + dilation_h * u1)[stride_w * dx5 + dilation_w * v1];
                    pp[12] = img0.row<const unsigned short>(stride_h * dy6 + dilation_h * u0)[stride_w * dx6 + dilation_w * v0];
                    pp[13] = img1.row<const unsigned short>(stride_h * dy6 + dilation_h * u1)[stride_w * dx6 + dilation_w * v1];
                    pp[14] = img0.row<const unsigned short>(stride_h * dy7 + dilation_h * u0)[stride_w * dx7 + dilation_w * v0];
                    pp[15] = img1.row<const unsigned short>(stride_h * dy7 + dilation_h * u1)[stride_w * dx7 + dilation_w * v1];
                    pp[16] = img0.row<const unsigned short>(stride_h * dy8 + dilation_h * u0)[stride_w * dx8 + dilation_w * v0];
                    pp[17] = img1.row<const unsigned short>(stride_h * dy8 + dilation_h * u1)[stride_w * dx8 + dilation_w * v1];
                    pp[18] = img0.row<const unsigned short>(stride_h * dy9 + dilation_h * u0)[stride_w * dx9 + dilation_w * v0];
                    pp[19] = img1.row<const unsigned short>(stride_h * dy9 + dilation_h * u1)[stride_w * dx9 + dilation_w * v1];
                    pp[20] = img0.row<const unsigned short>(stride_h * dya + dilation_h * u0)[stride_w * dxa + dilation_w * v0];
                    pp[21] = img1.row<const unsigned short>(stride_h * dya + dilation_h * u1)[stride_w * dxa + dilation_w * v1];
                    pp[22] = img0.row<const unsigned short>(stride_h * dyb + dilation_h * u0)[stride_w * dxb + dilation_w * v0];
                    pp[23] = img1.row<const unsigned short>(stride_h * dyb + dilation_h * u1)[stride_w * dxb + dilation_w * v1];
                    pp[24] = img0.row<const unsigned short>(stride_h * dyc + dilation_h * u0)[stride_w * dxc + dilation_w * v0];
                    pp[25] = img1.row<const unsigned short>(stride_h * dyc + dilation_h * u1)[stride_w * dxc + dilation_w * v1];
                    pp[26] = img0.row<const unsigned short>(stride_h * dyd + dilation_h * u0)[stride_w * dxd + dilation_w * v0];
                    pp[27] = img1.row<const unsigned short>(stride_h * dyd + dilation_h * u1)[stride_w * dxd + dilation_w * v1];
                    pp[28] = img0.row<const unsigned short>(stride_h * dye + dilation_h * u0)[stride_w * dxe + dilation_w * v0];
                    pp[29] = img1.row<const unsigned short>(stride_h * dye + dilation_h * u1)[stride_w * dxe + dilation_w * v1];
                    pp[30] = img0.row<const unsigned short>(stride_h * dyf + dilation_h * u0)[stride_w * dxf + dilation_w * v0];
                    pp[31] = img1.row<const unsigned short>(stride_h * dyf + dilation_h * u1)[stride_w * dxf + dilation_w * v1];
                    pp += 32;
                }
            }
#endif // __AVX512BF16__
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int x2 = stride_w * dx2 + dilation_w * v;
                int x3 = stride_w * dx3 + dilation_w * v;
                int x4 = stride_w * dx4 + dilation_w * v;
                int x5 = stride_w * dx5 + dilation_w * v;
                int x6 = stride_w * dx6 + dilation_w * v;
                int x7 = stride_w * dx7 + dilation_w * v;
                int x8 = stride_w * dx8 + dilation_w * v;
                int x9 = stride_w * dx9 + dilation_w * v;
                int xa = stride_w * dxa + dilation_w * v;
                int xb = stride_w * dxb + dilation_w * v;
                int xc = stride_w * dxc + dilation_w * v;
                int xd = stride_w * dxd + dilation_w * v;
                int xe = stride_w * dxe + dilation_w * v;
                int xf = stride_w * dxf + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;
                int y4 = stride_h * dy4 + dilation_h * u;
                int y5 = stride_h * dy5 + dilation_h * u;
                int y6 = stride_h * dy6 + dilation_h * u;
                int y7 = stride_h * dy7 + dilation_h * u;
                int y8 = stride_h * dy8 + dilation_h * u;
                int y9 = stride_h * dy9 + dilation_h * u;
                int ya = stride_h * dya + dilation_h * u;
                int yb = stride_h * dyb + dilation_h * u;
                int yc = stride_h * dyc + dilation_h * u;
                int yd = stride_h * dyd + dilation_h * u;
                int ye = stride_h * dye + dilation_h * u;
                int yf = stride_h * dyf + dilation_h * u;

                const unsigned short* sptr0 = img.row<const unsigned short>(y0) + x0 * elempack;
                const unsigned short* sptr1 = img.row<const unsigned short>(y1) + x1 * elempack;
                const unsigned short* sptr2 = img.row<const unsigned short>(y2) + x2 * elempack;
                const unsigned short* sptr3 = img.row<const unsigned short>(y3) + x3 * elempack;
                const unsigned short* sptr4 = img.row<const unsigned short>(y4) + x4 * elempack;
                const unsigned short* sptr5 = img.row<const unsigned short>(y5) + x5 * elempack;
                const unsigned short* sptr6 = img.row<const unsigned short>(y6) + x6 * elempack;
                const unsigned short* sptr7 = img.row<const unsigned short>(y7) + x7 * elempack;
                const unsigned short* sptr8 = img.row<const unsigned short>(y8) + x8 * elempack;
                const unsigned short* sptr9 = img.row<const unsigned short>(y9) + x9 * elempack;
                const unsigned short* sptra = img.row<const unsigned short>(ya) + xa * elempack;
                const unsigned short* sptrb = img.row<const unsigned short>(yb) + xb * elempack;
                const unsigned short* sptrc = img.row<const unsigned short>(yc) + xc * elempack;
                const unsigned short* sptrd = img.row<const unsigned short>(yd) + xd * elempack;
                const unsigned short* sptre = img.row<const unsigned short>(ye) + xe * elempack;
                const unsigned short* sptrf = img.row<const unsigned short>(yf) + xf * elempack;

#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    __m512 _r0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr0)));
                    __m512 _r1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr1)));
                    __m512 _r2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr2)));
                    __m512 _r3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr3)));
                    __m512 _r4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr4)));
                    __m512 _r5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr5)));
                    __m512 _r6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr6)));
                    __m512 _r7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr7)));
                    __m512 _r8 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr8)));
                    __m512 _r9 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr9)));
                    __m512 _ra = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptra)));
                    __m512 _rb = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptrb)));
                    __m512 _rc = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptrc)));
                    __m512 _rd = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptrd)));
                    __m512 _re = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptre)));
                    __m512 _rf = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptrf)));
                    transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
#if __AVX512BF16__
                    __m256i _t0 = float2bfloat_avx512(_r0);
                    __m128i _t0l = _mm256_extracti128_si256(_t0, 0);
                    __m128i _t0h = _mm256_extracti128_si256(_t0, 1);
                    _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_t0l, _t0h));
                    _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_t0l, _t0h));
                    __m256i _t1 = float2bfloat_avx512(_r1);
                    __m128i _t1l = _mm256_extracti128_si256(_t1, 0);
                    __m128i _t1h = _mm256_extracti128_si256(_t1, 1);
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_t1l, _t1h));
                    _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpackhi_epi16(_t1l, _t1h));
                    __m256i _t2 = float2bfloat_avx512(_r2);
                    __m128i _t2l = _mm256_extracti128_si256(_t2, 0);
                    __m128i _t2h = _mm256_extracti128_si256(_t2, 1);
                    _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi16(_t2l, _t2h));
                    _mm_storeu_si128((__m128i*)(pp + 40), _mm_unpackhi_epi16(_t2l, _t2h));
                    __m256i _t3 = float2bfloat_avx512(_r3);
                    __m128i _t3l = _mm256_extracti128_si256(_t3, 0);
                    __m128i _t3h = _mm256_extracti128_si256(_t3, 1);
                    _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpacklo_epi16(_t3l, _t3h));
                    _mm_storeu_si128((__m128i*)(pp + 56), _mm_unpackhi_epi16(_t3l, _t3h));
                    __m256i _t4 = float2bfloat_avx512(_r4);
                    __m128i _t4l = _mm256_extracti128_si256(_t4, 0);
                    __m128i _t4h = _mm256_extracti128_si256(_t4, 1);
                    _mm_storeu_si128((__m128i*)(pp + 64), _mm_unpacklo_epi16(_t4l, _t4h));
                    _mm_storeu_si128((__m128i*)(pp + 72), _mm_unpackhi_epi16(_t4l, _t4h));
                    __m256i _t5 = float2bfloat_avx512(_r5);
                    __m128i _t5l = _mm256_extracti128_si256(_t5, 0);
                    __m128i _t5h = _mm256_extracti128_si256(_t5, 1);
                    _mm_storeu_si128((__m128i*)(pp + 80), _mm_unpacklo_epi16(_t5l, _t5h));
                    _mm_storeu_si128((__m128i*)(pp + 88), _mm_unpackhi_epi16(_t5l, _t5h));
                    __m256i _t6 = float2bfloat_avx512(_r6);
                    __m128i _t6l = _mm256_extracti128_si256(_t6, 0);
                    __m128i _t6h = _mm256_extracti128_si256(_t6, 1);
                    _mm_storeu_si128((__m128i*)(pp + 96), _mm_unpacklo_epi16(_t6l, _t6h));
                    _mm_storeu_si128((__m128i*)(pp + 104), _mm_unpackhi_epi16(_t6l, _t6h));
                    __m256i _t7 = float2bfloat_avx512(_r7);
                    __m128i _t7l = _mm256_extracti128_si256(_t7, 0);
                    __m128i _t7h = _mm256_extracti128_si256(_t7, 1);
                    _mm_storeu_si128((__m128i*)(pp + 112), _mm_unpacklo_epi16(_t7l, _t7h));
                    _mm_storeu_si128((__m128i*)(pp + 120), _mm_unpackhi_epi16(_t7l, _t7h));
                    __m256i _t8 = float2bfloat_avx512(_r8);
                    __m128i _t8l = _mm256_extracti128_si256(_t8, 0);
                    __m128i _t8h = _mm256_extracti128_si256(_t8, 1);
                    _mm_storeu_si128((__m128i*)(pp + 128), _mm_unpacklo_epi16(_t8l, _t8h));
                    _mm_storeu_si128((__m128i*)(pp + 136), _mm_unpackhi_epi16(_t8l, _t8h));
                    __m256i _t9 = float2bfloat_avx512(_r9);
                    __m128i _t9l = _mm256_extracti128_si256(_t9, 0);
                    __m128i _t9h = _mm256_extracti128_si256(_t9, 1);
                    _mm_storeu_si128((__m128i*)(pp + 144), _mm_unpacklo_epi16(_t9l, _t9h));
                    _mm_storeu_si128((__m128i*)(pp + 152), _mm_unpackhi_epi16(_t9l, _t9h));
                    __m256i _ta = float2bfloat_avx512(_ra);
                    __m128i _tal = _mm256_extracti128_si256(_ta, 0);
                    __m128i _tah = _mm256_extracti128_si256(_ta, 1);
                    _mm_storeu_si128((__m128i*)(pp + 160), _mm_unpacklo_epi16(_tal, _tah));
                    _mm_storeu_si128((__m128i*)(pp + 168), _mm_unpackhi_epi16(_tal, _tah));
                    __m256i _tb = float2bfloat_avx512(_rb);
                    __m128i _tbl = _mm256_extracti128_si256(_tb, 0);
                    __m128i _tbh = _mm256_extracti128_si256(_tb, 1);
                    _mm_storeu_si128((__m128i*)(pp + 176), _mm_unpacklo_epi16(_tbl, _tbh));
                    _mm_storeu_si128((__m128i*)(pp + 184), _mm_unpackhi_epi16(_tbl, _tbh));
                    __m256i _tc = float2bfloat_avx512(_rc);
                    __m128i _tcl = _mm256_extracti128_si256(_tc, 0);
                    __m128i _tch = _mm256_extracti128_si256(_tc, 1);
                    _mm_storeu_si128((__m128i*)(pp + 192), _mm_unpacklo_epi16(_tcl, _tch));
                    _mm_storeu_si128((__m128i*)(pp + 200), _mm_unpackhi_epi16(_tcl, _tch));
                    __m256i _td = float2bfloat_avx512(_rd);
                    __m128i _tdl = _mm256_extracti128_si256(_td, 0);
                    __m128i _tdh = _mm256_extracti128_si256(_td, 1);
                    _mm_storeu_si128((__m128i*)(pp + 208), _mm_unpacklo_epi16(_tdl, _tdh));
                    _mm_storeu_si128((__m128i*)(pp + 216), _mm_unpackhi_epi16(_tdl, _tdh));
                    __m256i _te = float2bfloat_avx512(_re);
                    __m128i _tel = _mm256_extracti128_si256(_te, 0);
                    __m128i _teh = _mm256_extracti128_si256(_te, 1);
                    _mm_storeu_si128((__m128i*)(pp + 224), _mm_unpacklo_epi16(_tel, _teh));
                    _mm_storeu_si128((__m128i*)(pp + 232), _mm_unpackhi_epi16(_tel, _teh));
                    __m256i _tf = float2bfloat_avx512(_rf);
                    __m128i _tfl = _mm256_extracti128_si256(_tf, 0);
                    __m128i _tfh = _mm256_extracti128_si256(_tf, 1);
                    _mm_storeu_si128((__m128i*)(pp + 240), _mm_unpacklo_epi16(_tfl, _tfh));
                    _mm_storeu_si128((__m128i*)(pp + 248), _mm_unpackhi_epi16(_tfl, _tfh));
#else

                    _mm256_storeu_si256((__m256i*)pp, float2bfloat_avx512(_r0));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 1), float2bfloat_avx512(_r1));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 2), float2bfloat_avx512(_r2));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 3), float2bfloat_avx512(_r3));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 4), float2bfloat_avx512(_r4));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 5), float2bfloat_avx512(_r5));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 6), float2bfloat_avx512(_r6));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 7), float2bfloat_avx512(_r7));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 8), float2bfloat_avx512(_r8));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 9), float2bfloat_avx512(_r9));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 10), float2bfloat_avx512(_ra));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 11), float2bfloat_avx512(_rb));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 12), float2bfloat_avx512(_rc));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 13), float2bfloat_avx512(_rd));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 14), float2bfloat_avx512(_re));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 15), float2bfloat_avx512(_rf));
#endif // __AVX512BF16__
                    pp += 256;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr0)));
                    __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr1)));
                    __m256 _r2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr2)));
                    __m256 _r3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr3)));
                    __m256 _r4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr4)));
                    __m256 _r5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr5)));
                    __m256 _r6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr6)));
                    __m256 _r7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr7)));
                    __m256 _r8 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr8)));
                    __m256 _r9 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr9)));
                    __m256 _ra = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptra)));
                    __m256 _rb = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptrb)));
                    __m256 _rc = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptrc)));
                    __m256 _rd = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptrd)));
                    __m256 _re = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptre)));
                    __m256 _rf = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptrf)));
                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                    transpose8x8_ps(_r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
#if __AVX512BF16__
                    __m128i _t0 = float2bfloat_avx(_r0);
                    __m128i _t8 = float2bfloat_avx(_r8);
                    _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_t0, _t8));
                    _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_t0, _t8));
                    __m128i _t1 = float2bfloat_avx(_r1);
                    __m128i _t9 = float2bfloat_avx(_r9);
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_t1, _t9));
                    _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpackhi_epi16(_t1, _t9));
                    __m128i _t2 = float2bfloat_avx(_r2);
                    __m128i _ta = float2bfloat_avx(_ra);
                    _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi16(_t2, _ta));
                    _mm_storeu_si128((__m128i*)(pp + 40), _mm_unpackhi_epi16(_t2, _ta));
                    __m128i _t3 = float2bfloat_avx(_r3);
                    __m128i _tb = float2bfloat_avx(_rb);
                    _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpacklo_epi16(_t3, _tb));
                    _mm_storeu_si128((__m128i*)(pp + 56), _mm_unpackhi_epi16(_t3, _tb));
                    __m128i _t4 = float2bfloat_avx(_r4);
                    __m128i _tc = float2bfloat_avx(_rc);
                    _mm_storeu_si128((__m128i*)(pp + 64), _mm_unpacklo_epi16(_t4, _tc));
                    _mm_storeu_si128((__m128i*)(pp + 72), _mm_unpackhi_epi16(_t4, _tc));
                    __m128i _t5 = float2bfloat_avx(_r5);
                    __m128i _td = float2bfloat_avx(_rd);
                    _mm_storeu_si128((__m128i*)(pp + 80), _mm_unpacklo_epi16(_t5, _td));
                    _mm_storeu_si128((__m128i*)(pp + 88), _mm_unpackhi_epi16(_t5, _td));
                    __m128i _t6 = float2bfloat_avx(_r6);
                    __m128i _te = float2bfloat_avx(_re);
                    _mm_storeu_si128((__m128i*)(pp + 96), _mm_unpacklo_epi16(_t6, _te));
                    _mm_storeu_si128((__m128i*)(pp + 104), _mm_unpackhi_epi16(_t6, _te));
                    __m128i _t7 = float2bfloat_avx(_r7);
                    __m128i _tf = float2bfloat_avx(_rf);
                    _mm_storeu_si128((__m128i*)(pp + 112), _mm_unpacklo_epi16(_t7, _tf));
                    _mm_storeu_si128((__m128i*)(pp + 120), _mm_unpackhi_epi16(_t7, _tf));
#else

                    _mm_storeu_si128((__m128i*)pp, float2bfloat_avx(_r0));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 1), float2bfloat_avx(_r8));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 2), float2bfloat_avx(_r1));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 3), float2bfloat_avx(_r9));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 4), float2bfloat_avx(_r2));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 5), float2bfloat_avx(_ra));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 6), float2bfloat_avx(_r3));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 7), float2bfloat_avx(_rb));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 8), float2bfloat_avx(_r4));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 9), float2bfloat_avx(_rc));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 10), float2bfloat_avx(_r5));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 11), float2bfloat_avx(_rd));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 12), float2bfloat_avx(_r6));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 13), float2bfloat_avx(_re));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 14), float2bfloat_avx(_r7));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 15), float2bfloat_avx(_rf));
#endif // __AVX512BF16__
                    pp += 128;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    __m128 _r0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr0)));
                    __m128 _r1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr1)));
                    __m128 _r2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr2)));
                    __m128 _r3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr3)));
                    __m128 _r4 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr4)));
                    __m128 _r5 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr5)));
                    __m128 _r6 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr6)));
                    __m128 _r7 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr7)));
                    __m128 _r8 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr8)));
                    __m128 _r9 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr9)));
                    __m128 _ra = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptra)));
                    __m128 _rb = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptrb)));
                    __m128 _rc = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptrc)));
                    __m128 _rd = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptrd)));
                    __m128 _re = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptre)));
                    __m128 _rf = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptrf)));
                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                    _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);
                    _MM_TRANSPOSE4_PS(_rc, _rd, _re, _rf);
#if __AVX512BF16__
                    __m128i _tk0 = _mm_unpacklo_epi64(float2bfloat_sse(_r0), float2bfloat_sse(_r4));
                    __m128i _tk8 = _mm_unpacklo_epi64(float2bfloat_sse(_r8), float2bfloat_sse(_rc));
                    _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_tk0, _tk8));
                    _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_tk0, _tk8));
                    __m128i _tk1 = _mm_unpacklo_epi64(float2bfloat_sse(_r1), float2bfloat_sse(_r5));
                    __m128i _tk9 = _mm_unpacklo_epi64(float2bfloat_sse(_r9), float2bfloat_sse(_rd));
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_tk1, _tk9));
                    _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpackhi_epi16(_tk1, _tk9));
                    __m128i _tk2 = _mm_unpacklo_epi64(float2bfloat_sse(_r2), float2bfloat_sse(_r6));
                    __m128i _tka = _mm_unpacklo_epi64(float2bfloat_sse(_ra), float2bfloat_sse(_re));
                    _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi16(_tk2, _tka));
                    _mm_storeu_si128((__m128i*)(pp + 40), _mm_unpackhi_epi16(_tk2, _tka));
                    __m128i _tk3 = _mm_unpacklo_epi64(float2bfloat_sse(_r3), float2bfloat_sse(_r7));
                    __m128i _tkb = _mm_unpacklo_epi64(float2bfloat_sse(_rb), float2bfloat_sse(_rf));
                    _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpacklo_epi16(_tk3, _tkb));
                    _mm_storeu_si128((__m128i*)(pp + 56), _mm_unpackhi_epi16(_tk3, _tkb));
#else

                    _mm_storel_epi64((__m128i*)pp, float2bfloat_sse(_r0));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 1), float2bfloat_sse(_r4));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 2), float2bfloat_sse(_r8));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 3), float2bfloat_sse(_rc));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 4), float2bfloat_sse(_r1));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 5), float2bfloat_sse(_r5));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 6), float2bfloat_sse(_r9));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 7), float2bfloat_sse(_rd));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 8), float2bfloat_sse(_r2));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 9), float2bfloat_sse(_r6));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 10), float2bfloat_sse(_ra));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 11), float2bfloat_sse(_re));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 12), float2bfloat_sse(_r3));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 13), float2bfloat_sse(_r7));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 14), float2bfloat_sse(_rb));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 15), float2bfloat_sse(_rf));
#endif // __AVX512BF16__
                    pp += 64;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp[4] = sptr4[0];
                    pp[5] = sptr5[0];
                    pp[6] = sptr6[0];
                    pp[7] = sptr7[0];
                    pp[8] = sptr8[0];
                    pp[9] = sptr9[0];
                    pp[10] = sptra[0];
                    pp[11] = sptrb[0];
                    pp[12] = sptrc[0];
                    pp[13] = sptrd[0];
                    pp[14] = sptre[0];
                    pp[15] = sptrf[0];
                    pp += 16;
                }
            }
        }
    }
#endif // __AVX512F__

    for (; jj + 7 < max_jj; jj += 8)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dy4 = (j + jj + 4) / outw;
        int dy5 = (j + jj + 5) / outw;
        int dy6 = (j + jj + 6) / outw;
        int dy7 = (j + jj + 7) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;
        int dx4 = (j + jj + 4) % outw;
        int dx5 = (j + jj + 5) % outw;
        int dx6 = (j + jj + 6) % outw;
        int dx7 = (j + jj + 7) % outw;

        if (dy0 == dy7)
        {
            int kk = 0;
#if __AVX512BF16__
            if (elempack == 1)
            {
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int u0 = uv0 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    const Mat img0 = bottom_blob.channel(p0);
                    int sx0 = stride_w * dx0 + dilation_w * v0;
                    int sy0 = stride_h * dy0 + dilation_h * u0;
                    const unsigned short* sptr0 = img0.row<const unsigned short>(sy0) + sx0;

                    int p1 = (k + kk + 1) / maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u1 = uv1 / kernel_w;
                    int v1 = uv1 % kernel_w;
                    const Mat img1 = bottom_blob.channel(p1);
                    int sx1 = stride_w * dx0 + dilation_w * v1;
                    int sy1 = stride_h * dy0 + dilation_h * u1;
                    const unsigned short* sptr1 = img1.row<const unsigned short>(sy1) + sx1;

                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr0[stride_w];
                    pp[3] = sptr1[stride_w];
                    pp[4] = sptr0[stride_w * 2];
                    pp[5] = sptr1[stride_w * 2];
                    pp[6] = sptr0[stride_w * 3];
                    pp[7] = sptr1[stride_w * 3];
                    pp[8] = sptr0[stride_w * 4];
                    pp[9] = sptr1[stride_w * 4];
                    pp[10] = sptr0[stride_w * 5];
                    pp[11] = sptr1[stride_w * 5];
                    pp[12] = sptr0[stride_w * 6];
                    pp[13] = sptr1[stride_w * 6];
                    pp[14] = sptr0[stride_w * 7];
                    pp[15] = sptr1[stride_w * 7];
                    pp += 16;
                }
            }
#endif // __AVX512BF16__
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const unsigned short* sptr = img.row<const unsigned short>(y0) + x0 * elempack;

#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    __m512 _r0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr)));
                    __m512 _r1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 16)));
                    __m512 _r2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 32)));
                    __m512 _r3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 48)));
                    __m512 _r4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 64)));
                    __m512 _r5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 80)));
                    __m512 _r6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 96)));
                    __m512 _r7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 112)));
                    transpose16x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
#if __AVX512BF16__
                    __m256i _t0 = float2bfloat_avx512(_r0);
                    __m128i _t0l = _mm256_extracti128_si256(_t0, 0);
                    __m128i _t0h = _mm256_extracti128_si256(_t0, 1);
                    _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_t0l, _t0h));
                    _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_t0l, _t0h));
                    __m256i _t1 = float2bfloat_avx512(_r1);
                    __m128i _t1l = _mm256_extracti128_si256(_t1, 0);
                    __m128i _t1h = _mm256_extracti128_si256(_t1, 1);
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_t1l, _t1h));
                    _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpackhi_epi16(_t1l, _t1h));
                    __m256i _t2 = float2bfloat_avx512(_r2);
                    __m128i _t2l = _mm256_extracti128_si256(_t2, 0);
                    __m128i _t2h = _mm256_extracti128_si256(_t2, 1);
                    _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi16(_t2l, _t2h));
                    _mm_storeu_si128((__m128i*)(pp + 40), _mm_unpackhi_epi16(_t2l, _t2h));
                    __m256i _t3 = float2bfloat_avx512(_r3);
                    __m128i _t3l = _mm256_extracti128_si256(_t3, 0);
                    __m128i _t3h = _mm256_extracti128_si256(_t3, 1);
                    _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpacklo_epi16(_t3l, _t3h));
                    _mm_storeu_si128((__m128i*)(pp + 56), _mm_unpackhi_epi16(_t3l, _t3h));
                    __m256i _t4 = float2bfloat_avx512(_r4);
                    __m128i _t4l = _mm256_extracti128_si256(_t4, 0);
                    __m128i _t4h = _mm256_extracti128_si256(_t4, 1);
                    _mm_storeu_si128((__m128i*)(pp + 64), _mm_unpacklo_epi16(_t4l, _t4h));
                    _mm_storeu_si128((__m128i*)(pp + 72), _mm_unpackhi_epi16(_t4l, _t4h));
                    __m256i _t5 = float2bfloat_avx512(_r5);
                    __m128i _t5l = _mm256_extracti128_si256(_t5, 0);
                    __m128i _t5h = _mm256_extracti128_si256(_t5, 1);
                    _mm_storeu_si128((__m128i*)(pp + 80), _mm_unpacklo_epi16(_t5l, _t5h));
                    _mm_storeu_si128((__m128i*)(pp + 88), _mm_unpackhi_epi16(_t5l, _t5h));
                    __m256i _t6 = float2bfloat_avx512(_r6);
                    __m128i _t6l = _mm256_extracti128_si256(_t6, 0);
                    __m128i _t6h = _mm256_extracti128_si256(_t6, 1);
                    _mm_storeu_si128((__m128i*)(pp + 96), _mm_unpacklo_epi16(_t6l, _t6h));
                    _mm_storeu_si128((__m128i*)(pp + 104), _mm_unpackhi_epi16(_t6l, _t6h));
                    __m256i _t7 = float2bfloat_avx512(_r7);
                    __m128i _t7l = _mm256_extracti128_si256(_t7, 0);
                    __m128i _t7h = _mm256_extracti128_si256(_t7, 1);
                    _mm_storeu_si128((__m128i*)(pp + 112), _mm_unpacklo_epi16(_t7l, _t7h));
                    _mm_storeu_si128((__m128i*)(pp + 120), _mm_unpackhi_epi16(_t7l, _t7h));
#else

                    _mm256_storeu_si256((__m256i*)pp, float2bfloat_avx512(_r0));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 1), float2bfloat_avx512(_r1));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 2), float2bfloat_avx512(_r2));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 3), float2bfloat_avx512(_r3));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 4), float2bfloat_avx512(_r4));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 5), float2bfloat_avx512(_r5));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 6), float2bfloat_avx512(_r6));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 7), float2bfloat_avx512(_r7));
#endif // __AVX512BF16__
                    pp += 128;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr)));
                    __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 8)));
                    __m256 _r2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 16)));
                    __m256 _r3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 24)));
                    __m256 _r4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 32)));
                    __m256 _r5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 40)));
                    __m256 _r6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 48)));
                    __m256 _r7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 56)));
                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
#if __AVX512BF16__
                    __m128i _t0 = float2bfloat_avx(_r0);
                    __m128i _t1 = float2bfloat_avx(_r1);
                    _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_t0, _t1));
                    _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_t0, _t1));
                    __m128i _t2 = float2bfloat_avx(_r2);
                    __m128i _t3 = float2bfloat_avx(_r3);
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_t2, _t3));
                    _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpackhi_epi16(_t2, _t3));
                    __m128i _t4 = float2bfloat_avx(_r4);
                    __m128i _t5 = float2bfloat_avx(_r5);
                    _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi16(_t4, _t5));
                    _mm_storeu_si128((__m128i*)(pp + 40), _mm_unpackhi_epi16(_t4, _t5));
                    __m128i _t6 = float2bfloat_avx(_r6);
                    __m128i _t7 = float2bfloat_avx(_r7);
                    _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpacklo_epi16(_t6, _t7));
                    _mm_storeu_si128((__m128i*)(pp + 56), _mm_unpackhi_epi16(_t6, _t7));
#else

                    _mm_storeu_si128((__m128i*)pp, float2bfloat_avx(_r0));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 1), float2bfloat_avx(_r1));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 2), float2bfloat_avx(_r2));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 3), float2bfloat_avx(_r3));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 4), float2bfloat_avx(_r4));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 5), float2bfloat_avx(_r5));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 6), float2bfloat_avx(_r6));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 7), float2bfloat_avx(_r7));
#endif // __AVX512BF16__
                    pp += 64;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    __m128 _r0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr)));
                    __m128 _r1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 4)));
                    __m128 _r2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 8)));
                    __m128 _r3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 12)));
                    __m128 _r4 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 16)));
                    __m128 _r5 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 20)));
                    __m128 _r6 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 24)));
                    __m128 _r7 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 28)));
                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
#if __AVX512BF16__
                    __m128i _tk0 = _mm_unpacklo_epi64(float2bfloat_sse(_r0), float2bfloat_sse(_r4));
                    __m128i _tk1 = _mm_unpacklo_epi64(float2bfloat_sse(_r1), float2bfloat_sse(_r5));
                    _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_tk0, _tk1));
                    _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_tk0, _tk1));
                    __m128i _tk2 = _mm_unpacklo_epi64(float2bfloat_sse(_r2), float2bfloat_sse(_r6));
                    __m128i _tk3 = _mm_unpacklo_epi64(float2bfloat_sse(_r3), float2bfloat_sse(_r7));
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_tk2, _tk3));
                    _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpackhi_epi16(_tk2, _tk3));
#else

                    _mm_storel_epi64((__m128i*)pp, float2bfloat_sse(_r0));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 1), float2bfloat_sse(_r4));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 2), float2bfloat_sse(_r1));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 3), float2bfloat_sse(_r5));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 4), float2bfloat_sse(_r2));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 5), float2bfloat_sse(_r6));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 6), float2bfloat_sse(_r3));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 7), float2bfloat_sse(_r7));
#endif // __AVX512BF16__
                    pp += 32;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp[2] = sptr[stride_w * 2];
                    pp[3] = sptr[stride_w * 3];
                    pp[4] = sptr[stride_w * 4];
                    pp[5] = sptr[stride_w * 5];
                    pp[6] = sptr[stride_w * 6];
                    pp[7] = sptr[stride_w * 7];
                    pp += 8;
                }
            }
        }
        else
        {
            int kk = 0;
#if __AVX512BF16__
            if (elempack == 1)
            {
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int u0 = uv0 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    const Mat img0 = bottom_blob.channel(p0);

                    int p1 = (k + kk + 1) / maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u1 = uv1 / kernel_w;
                    int v1 = uv1 % kernel_w;
                    const Mat img1 = bottom_blob.channel(p1);

                    pp[0] = img0.row<const unsigned short>(stride_h * dy0 + dilation_h * u0)[stride_w * dx0 + dilation_w * v0];
                    pp[1] = img1.row<const unsigned short>(stride_h * dy0 + dilation_h * u1)[stride_w * dx0 + dilation_w * v1];
                    pp[2] = img0.row<const unsigned short>(stride_h * dy1 + dilation_h * u0)[stride_w * dx1 + dilation_w * v0];
                    pp[3] = img1.row<const unsigned short>(stride_h * dy1 + dilation_h * u1)[stride_w * dx1 + dilation_w * v1];
                    pp[4] = img0.row<const unsigned short>(stride_h * dy2 + dilation_h * u0)[stride_w * dx2 + dilation_w * v0];
                    pp[5] = img1.row<const unsigned short>(stride_h * dy2 + dilation_h * u1)[stride_w * dx2 + dilation_w * v1];
                    pp[6] = img0.row<const unsigned short>(stride_h * dy3 + dilation_h * u0)[stride_w * dx3 + dilation_w * v0];
                    pp[7] = img1.row<const unsigned short>(stride_h * dy3 + dilation_h * u1)[stride_w * dx3 + dilation_w * v1];
                    pp[8] = img0.row<const unsigned short>(stride_h * dy4 + dilation_h * u0)[stride_w * dx4 + dilation_w * v0];
                    pp[9] = img1.row<const unsigned short>(stride_h * dy4 + dilation_h * u1)[stride_w * dx4 + dilation_w * v1];
                    pp[10] = img0.row<const unsigned short>(stride_h * dy5 + dilation_h * u0)[stride_w * dx5 + dilation_w * v0];
                    pp[11] = img1.row<const unsigned short>(stride_h * dy5 + dilation_h * u1)[stride_w * dx5 + dilation_w * v1];
                    pp[12] = img0.row<const unsigned short>(stride_h * dy6 + dilation_h * u0)[stride_w * dx6 + dilation_w * v0];
                    pp[13] = img1.row<const unsigned short>(stride_h * dy6 + dilation_h * u1)[stride_w * dx6 + dilation_w * v1];
                    pp[14] = img0.row<const unsigned short>(stride_h * dy7 + dilation_h * u0)[stride_w * dx7 + dilation_w * v0];
                    pp[15] = img1.row<const unsigned short>(stride_h * dy7 + dilation_h * u1)[stride_w * dx7 + dilation_w * v1];
                    pp += 16;
                }
            }
#endif // __AVX512BF16__
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int x2 = stride_w * dx2 + dilation_w * v;
                int x3 = stride_w * dx3 + dilation_w * v;
                int x4 = stride_w * dx4 + dilation_w * v;
                int x5 = stride_w * dx5 + dilation_w * v;
                int x6 = stride_w * dx6 + dilation_w * v;
                int x7 = stride_w * dx7 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;
                int y4 = stride_h * dy4 + dilation_h * u;
                int y5 = stride_h * dy5 + dilation_h * u;
                int y6 = stride_h * dy6 + dilation_h * u;
                int y7 = stride_h * dy7 + dilation_h * u;

                const unsigned short* sptr0 = img.row<const unsigned short>(y0) + x0 * elempack;
                const unsigned short* sptr1 = img.row<const unsigned short>(y1) + x1 * elempack;
                const unsigned short* sptr2 = img.row<const unsigned short>(y2) + x2 * elempack;
                const unsigned short* sptr3 = img.row<const unsigned short>(y3) + x3 * elempack;
                const unsigned short* sptr4 = img.row<const unsigned short>(y4) + x4 * elempack;
                const unsigned short* sptr5 = img.row<const unsigned short>(y5) + x5 * elempack;
                const unsigned short* sptr6 = img.row<const unsigned short>(y6) + x6 * elempack;
                const unsigned short* sptr7 = img.row<const unsigned short>(y7) + x7 * elempack;

#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    __m512 _r0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr0)));
                    __m512 _r1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr1)));
                    __m512 _r2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr2)));
                    __m512 _r3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr3)));
                    __m512 _r4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr4)));
                    __m512 _r5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr5)));
                    __m512 _r6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr6)));
                    __m512 _r7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr7)));
                    transpose16x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
#if __AVX512BF16__
                    __m256i _t0 = float2bfloat_avx512(_r0);
                    __m128i _t0l = _mm256_extracti128_si256(_t0, 0);
                    __m128i _t0h = _mm256_extracti128_si256(_t0, 1);
                    _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_t0l, _t0h));
                    _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_t0l, _t0h));
                    __m256i _t1 = float2bfloat_avx512(_r1);
                    __m128i _t1l = _mm256_extracti128_si256(_t1, 0);
                    __m128i _t1h = _mm256_extracti128_si256(_t1, 1);
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_t1l, _t1h));
                    _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpackhi_epi16(_t1l, _t1h));
                    __m256i _t2 = float2bfloat_avx512(_r2);
                    __m128i _t2l = _mm256_extracti128_si256(_t2, 0);
                    __m128i _t2h = _mm256_extracti128_si256(_t2, 1);
                    _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi16(_t2l, _t2h));
                    _mm_storeu_si128((__m128i*)(pp + 40), _mm_unpackhi_epi16(_t2l, _t2h));
                    __m256i _t3 = float2bfloat_avx512(_r3);
                    __m128i _t3l = _mm256_extracti128_si256(_t3, 0);
                    __m128i _t3h = _mm256_extracti128_si256(_t3, 1);
                    _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpacklo_epi16(_t3l, _t3h));
                    _mm_storeu_si128((__m128i*)(pp + 56), _mm_unpackhi_epi16(_t3l, _t3h));
                    __m256i _t4 = float2bfloat_avx512(_r4);
                    __m128i _t4l = _mm256_extracti128_si256(_t4, 0);
                    __m128i _t4h = _mm256_extracti128_si256(_t4, 1);
                    _mm_storeu_si128((__m128i*)(pp + 64), _mm_unpacklo_epi16(_t4l, _t4h));
                    _mm_storeu_si128((__m128i*)(pp + 72), _mm_unpackhi_epi16(_t4l, _t4h));
                    __m256i _t5 = float2bfloat_avx512(_r5);
                    __m128i _t5l = _mm256_extracti128_si256(_t5, 0);
                    __m128i _t5h = _mm256_extracti128_si256(_t5, 1);
                    _mm_storeu_si128((__m128i*)(pp + 80), _mm_unpacklo_epi16(_t5l, _t5h));
                    _mm_storeu_si128((__m128i*)(pp + 88), _mm_unpackhi_epi16(_t5l, _t5h));
                    __m256i _t6 = float2bfloat_avx512(_r6);
                    __m128i _t6l = _mm256_extracti128_si256(_t6, 0);
                    __m128i _t6h = _mm256_extracti128_si256(_t6, 1);
                    _mm_storeu_si128((__m128i*)(pp + 96), _mm_unpacklo_epi16(_t6l, _t6h));
                    _mm_storeu_si128((__m128i*)(pp + 104), _mm_unpackhi_epi16(_t6l, _t6h));
                    __m256i _t7 = float2bfloat_avx512(_r7);
                    __m128i _t7l = _mm256_extracti128_si256(_t7, 0);
                    __m128i _t7h = _mm256_extracti128_si256(_t7, 1);
                    _mm_storeu_si128((__m128i*)(pp + 112), _mm_unpacklo_epi16(_t7l, _t7h));
                    _mm_storeu_si128((__m128i*)(pp + 120), _mm_unpackhi_epi16(_t7l, _t7h));
#else

                    _mm256_storeu_si256((__m256i*)pp, float2bfloat_avx512(_r0));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 1), float2bfloat_avx512(_r1));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 2), float2bfloat_avx512(_r2));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 3), float2bfloat_avx512(_r3));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 4), float2bfloat_avx512(_r4));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 5), float2bfloat_avx512(_r5));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 6), float2bfloat_avx512(_r6));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 7), float2bfloat_avx512(_r7));
#endif // __AVX512BF16__
                    pp += 128;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr0)));
                    __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr1)));
                    __m256 _r2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr2)));
                    __m256 _r3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr3)));
                    __m256 _r4 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr4)));
                    __m256 _r5 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr5)));
                    __m256 _r6 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr6)));
                    __m256 _r7 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr7)));
                    transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
#if __AVX512BF16__
                    __m128i _t0 = float2bfloat_avx(_r0);
                    __m128i _t1 = float2bfloat_avx(_r1);
                    _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_t0, _t1));
                    _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_t0, _t1));
                    __m128i _t2 = float2bfloat_avx(_r2);
                    __m128i _t3 = float2bfloat_avx(_r3);
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_t2, _t3));
                    _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpackhi_epi16(_t2, _t3));
                    __m128i _t4 = float2bfloat_avx(_r4);
                    __m128i _t5 = float2bfloat_avx(_r5);
                    _mm_storeu_si128((__m128i*)(pp + 32), _mm_unpacklo_epi16(_t4, _t5));
                    _mm_storeu_si128((__m128i*)(pp + 40), _mm_unpackhi_epi16(_t4, _t5));
                    __m128i _t6 = float2bfloat_avx(_r6);
                    __m128i _t7 = float2bfloat_avx(_r7);
                    _mm_storeu_si128((__m128i*)(pp + 48), _mm_unpacklo_epi16(_t6, _t7));
                    _mm_storeu_si128((__m128i*)(pp + 56), _mm_unpackhi_epi16(_t6, _t7));
#else

                    _mm_storeu_si128((__m128i*)pp, float2bfloat_avx(_r0));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 1), float2bfloat_avx(_r1));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 2), float2bfloat_avx(_r2));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 3), float2bfloat_avx(_r3));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 4), float2bfloat_avx(_r4));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 5), float2bfloat_avx(_r5));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 6), float2bfloat_avx(_r6));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 7), float2bfloat_avx(_r7));
#endif // __AVX512BF16__
                    pp += 64;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    __m128 _r0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr0)));
                    __m128 _r1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr1)));
                    __m128 _r2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr2)));
                    __m128 _r3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr3)));
                    __m128 _r4 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr4)));
                    __m128 _r5 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr5)));
                    __m128 _r6 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr6)));
                    __m128 _r7 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr7)));
                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
#if __AVX512BF16__
                    __m128i _tk0 = _mm_unpacklo_epi64(float2bfloat_sse(_r0), float2bfloat_sse(_r4));
                    __m128i _tk1 = _mm_unpacklo_epi64(float2bfloat_sse(_r1), float2bfloat_sse(_r5));
                    _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_tk0, _tk1));
                    _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpackhi_epi16(_tk0, _tk1));
                    __m128i _tk2 = _mm_unpacklo_epi64(float2bfloat_sse(_r2), float2bfloat_sse(_r6));
                    __m128i _tk3 = _mm_unpacklo_epi64(float2bfloat_sse(_r3), float2bfloat_sse(_r7));
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_tk2, _tk3));
                    _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpackhi_epi16(_tk2, _tk3));
#else

                    _mm_storel_epi64((__m128i*)pp, float2bfloat_sse(_r0));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 1), float2bfloat_sse(_r4));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 2), float2bfloat_sse(_r1));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 3), float2bfloat_sse(_r5));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 4), float2bfloat_sse(_r2));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 5), float2bfloat_sse(_r6));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 6), float2bfloat_sse(_r3));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 7), float2bfloat_sse(_r7));
#endif // __AVX512BF16__
                    pp += 32;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp[4] = sptr4[0];
                    pp[5] = sptr5[0];
                    pp[6] = sptr6[0];
                    pp[7] = sptr7[0];
                    pp += 8;
                }
            }
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)

    for (; jj + 3 < max_jj; jj += 4)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;

        if (dy0 == dy3)
        {
            int kk = 0;
#if __AVX512BF16__
            if (elempack == 1)
            {
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int u0 = uv0 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    const Mat img0 = bottom_blob.channel(p0);
                    int sx0 = stride_w * dx0 + dilation_w * v0;
                    int sy0 = stride_h * dy0 + dilation_h * u0;
                    const unsigned short* sptr0 = img0.row<const unsigned short>(sy0) + sx0;

                    int p1 = (k + kk + 1) / maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u1 = uv1 / kernel_w;
                    int v1 = uv1 % kernel_w;
                    const Mat img1 = bottom_blob.channel(p1);
                    int sx1 = stride_w * dx0 + dilation_w * v1;
                    int sy1 = stride_h * dy0 + dilation_h * u1;
                    const unsigned short* sptr1 = img1.row<const unsigned short>(sy1) + sx1;

                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr0[stride_w];
                    pp[3] = sptr1[stride_w];
                    pp[4] = sptr0[stride_w * 2];
                    pp[5] = sptr1[stride_w * 2];
                    pp[6] = sptr0[stride_w * 3];
                    pp[7] = sptr1[stride_w * 3];
                    pp += 8;
                }
            }
#endif // __AVX512BF16__
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const unsigned short* sptr = img.row<const unsigned short>(y0) + x0 * elempack;

#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    __m512 _r0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr)));
                    __m512 _r1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 16)));
                    __m512 _r2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 32)));
                    __m512 _r3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr + stride_w * 48)));
                    transpose16x4_ps(_r0, _r1, _r2, _r3);
#if __AVX512BF16__
                    __m256i _t0 = float2bfloat_avx512(_r0);
                    __m128i _t0a = _mm256_extracti128_si256(_t0, 0);
                    __m128i _t0b = _mm256_extracti128_si256(_t0, 1);
                    __m128i _t0k01_lo = _mm_unpacklo_epi16(_t0a, _mm_srli_si128(_t0a, 8));
                    _mm_storeu_si128((__m128i*)(pp + 0), _t0k01_lo);
                    __m128i _t0k23_lo = _mm_unpacklo_epi16(_t0b, _mm_srli_si128(_t0b, 8));
                    _mm_storeu_si128((__m128i*)(pp + 8), _t0k23_lo);
                    __m256i _t1 = float2bfloat_avx512(_r1);
                    __m128i _t1a = _mm256_extracti128_si256(_t1, 0);
                    __m128i _t1b = _mm256_extracti128_si256(_t1, 1);
                    __m128i _t1k01_lo = _mm_unpacklo_epi16(_t1a, _mm_srli_si128(_t1a, 8));
                    _mm_storeu_si128((__m128i*)(pp + 16), _t1k01_lo);
                    __m128i _t1k23_lo = _mm_unpacklo_epi16(_t1b, _mm_srli_si128(_t1b, 8));
                    _mm_storeu_si128((__m128i*)(pp + 24), _t1k23_lo);
                    __m256i _t2 = float2bfloat_avx512(_r2);
                    __m128i _t2a = _mm256_extracti128_si256(_t2, 0);
                    __m128i _t2b = _mm256_extracti128_si256(_t2, 1);
                    __m128i _t2k01_lo = _mm_unpacklo_epi16(_t2a, _mm_srli_si128(_t2a, 8));
                    _mm_storeu_si128((__m128i*)(pp + 32), _t2k01_lo);
                    __m128i _t2k23_lo = _mm_unpacklo_epi16(_t2b, _mm_srli_si128(_t2b, 8));
                    _mm_storeu_si128((__m128i*)(pp + 40), _t2k23_lo);
                    __m256i _t3 = float2bfloat_avx512(_r3);
                    __m128i _t3a = _mm256_extracti128_si256(_t3, 0);
                    __m128i _t3b = _mm256_extracti128_si256(_t3, 1);
                    __m128i _t3k01_lo = _mm_unpacklo_epi16(_t3a, _mm_srli_si128(_t3a, 8));
                    _mm_storeu_si128((__m128i*)(pp + 48), _t3k01_lo);
                    __m128i _t3k23_lo = _mm_unpacklo_epi16(_t3b, _mm_srli_si128(_t3b, 8));
                    _mm_storeu_si128((__m128i*)(pp + 56), _t3k23_lo);
#else

                    _mm256_storeu_si256((__m256i*)pp, float2bfloat_avx512(_r0));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 1), float2bfloat_avx512(_r1));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 2), float2bfloat_avx512(_r2));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 3), float2bfloat_avx512(_r3));
#endif // __AVX512BF16__
                    pp += 64;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr)));
                    __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 8)));
                    __m256 _r2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 16)));
                    __m256 _r3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr + stride_w * 24)));
                    transpose8x4_ps(_r0, _r1, _r2, _r3);
#if __AVX512BF16__
                    __m128i _t0 = float2bfloat_avx(_r0);
                    _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_t0, _mm_srli_si128(_t0, 8)));
                    __m128i _t1 = float2bfloat_avx(_r1);
                    _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpacklo_epi16(_t1, _mm_srli_si128(_t1, 8)));
                    __m128i _t2 = float2bfloat_avx(_r2);
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_t2, _mm_srli_si128(_t2, 8)));
                    __m128i _t3 = float2bfloat_avx(_r3);
                    _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpacklo_epi16(_t3, _mm_srli_si128(_t3, 8)));
#else

                    _mm_storeu_si128((__m128i*)pp, float2bfloat_avx(_r0));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 1), float2bfloat_avx(_r1));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 2), float2bfloat_avx(_r2));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 3), float2bfloat_avx(_r3));
#endif // __AVX512BF16__
                    pp += 32;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    __m128 _r0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr)));
                    __m128 _r1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 4)));
                    __m128 _r2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 8)));
                    __m128 _r3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr + stride_w * 12)));
                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
#if __AVX512BF16__
                    __m128i _t0 = float2bfloat_sse(_r0);
                    __m128i _t1 = float2bfloat_sse(_r1);
                    _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_t0, _t1));
                    __m128i _t2 = float2bfloat_sse(_r2);
                    __m128i _t3 = float2bfloat_sse(_r3);
                    _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpacklo_epi16(_t2, _t3));
#else

                    _mm_storel_epi64((__m128i*)pp, float2bfloat_sse(_r0));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 1), float2bfloat_sse(_r1));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 2), float2bfloat_sse(_r2));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 3), float2bfloat_sse(_r3));
#endif // __AVX512BF16__
                    pp += 16;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp[2] = sptr[stride_w * 2];
                    pp[3] = sptr[stride_w * 3];
                    pp += 4;
                }
            }
        }
        else
        {
            int kk = 0;
#if __AVX512BF16__
            if (elempack == 1)
            {
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int u0 = uv0 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    const Mat img0 = bottom_blob.channel(p0);

                    int p1 = (k + kk + 1) / maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u1 = uv1 / kernel_w;
                    int v1 = uv1 % kernel_w;
                    const Mat img1 = bottom_blob.channel(p1);

                    pp[0] = img0.row<const unsigned short>(stride_h * dy0 + dilation_h * u0)[stride_w * dx0 + dilation_w * v0];
                    pp[1] = img1.row<const unsigned short>(stride_h * dy0 + dilation_h * u1)[stride_w * dx0 + dilation_w * v1];
                    pp[2] = img0.row<const unsigned short>(stride_h * dy1 + dilation_h * u0)[stride_w * dx1 + dilation_w * v0];
                    pp[3] = img1.row<const unsigned short>(stride_h * dy1 + dilation_h * u1)[stride_w * dx1 + dilation_w * v1];
                    pp[4] = img0.row<const unsigned short>(stride_h * dy2 + dilation_h * u0)[stride_w * dx2 + dilation_w * v0];
                    pp[5] = img1.row<const unsigned short>(stride_h * dy2 + dilation_h * u1)[stride_w * dx2 + dilation_w * v1];
                    pp[6] = img0.row<const unsigned short>(stride_h * dy3 + dilation_h * u0)[stride_w * dx3 + dilation_w * v0];
                    pp[7] = img1.row<const unsigned short>(stride_h * dy3 + dilation_h * u1)[stride_w * dx3 + dilation_w * v1];
                    pp += 8;
                }
            }
#endif // __AVX512BF16__
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int x2 = stride_w * dx2 + dilation_w * v;
                int x3 = stride_w * dx3 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;

                const unsigned short* sptr0 = img.row<const unsigned short>(y0) + x0 * elempack;
                const unsigned short* sptr1 = img.row<const unsigned short>(y1) + x1 * elempack;
                const unsigned short* sptr2 = img.row<const unsigned short>(y2) + x2 * elempack;
                const unsigned short* sptr3 = img.row<const unsigned short>(y3) + x3 * elempack;

#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    __m512 _r0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr0)));
                    __m512 _r1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr1)));
                    __m512 _r2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr2)));
                    __m512 _r3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(sptr3)));
                    transpose16x4_ps(_r0, _r1, _r2, _r3);
#if __AVX512BF16__
                    __m256i _t0 = float2bfloat_avx512(_r0);
                    __m128i _t0a = _mm256_extracti128_si256(_t0, 0);
                    __m128i _t0b = _mm256_extracti128_si256(_t0, 1);
                    __m128i _t0k01_lo = _mm_unpacklo_epi16(_t0a, _mm_srli_si128(_t0a, 8));
                    _mm_storeu_si128((__m128i*)(pp + 0), _t0k01_lo);
                    __m128i _t0k23_lo = _mm_unpacklo_epi16(_t0b, _mm_srli_si128(_t0b, 8));
                    _mm_storeu_si128((__m128i*)(pp + 8), _t0k23_lo);
                    __m256i _t1 = float2bfloat_avx512(_r1);
                    __m128i _t1a = _mm256_extracti128_si256(_t1, 0);
                    __m128i _t1b = _mm256_extracti128_si256(_t1, 1);
                    __m128i _t1k01_lo = _mm_unpacklo_epi16(_t1a, _mm_srli_si128(_t1a, 8));
                    _mm_storeu_si128((__m128i*)(pp + 16), _t1k01_lo);
                    __m128i _t1k23_lo = _mm_unpacklo_epi16(_t1b, _mm_srli_si128(_t1b, 8));
                    _mm_storeu_si128((__m128i*)(pp + 24), _t1k23_lo);
                    __m256i _t2 = float2bfloat_avx512(_r2);
                    __m128i _t2a = _mm256_extracti128_si256(_t2, 0);
                    __m128i _t2b = _mm256_extracti128_si256(_t2, 1);
                    __m128i _t2k01_lo = _mm_unpacklo_epi16(_t2a, _mm_srli_si128(_t2a, 8));
                    _mm_storeu_si128((__m128i*)(pp + 32), _t2k01_lo);
                    __m128i _t2k23_lo = _mm_unpacklo_epi16(_t2b, _mm_srli_si128(_t2b, 8));
                    _mm_storeu_si128((__m128i*)(pp + 40), _t2k23_lo);
                    __m256i _t3 = float2bfloat_avx512(_r3);
                    __m128i _t3a = _mm256_extracti128_si256(_t3, 0);
                    __m128i _t3b = _mm256_extracti128_si256(_t3, 1);
                    __m128i _t3k01_lo = _mm_unpacklo_epi16(_t3a, _mm_srli_si128(_t3a, 8));
                    _mm_storeu_si128((__m128i*)(pp + 48), _t3k01_lo);
                    __m128i _t3k23_lo = _mm_unpacklo_epi16(_t3b, _mm_srli_si128(_t3b, 8));
                    _mm_storeu_si128((__m128i*)(pp + 56), _t3k23_lo);
#else

                    _mm256_storeu_si256((__m256i*)pp, float2bfloat_avx512(_r0));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 1), float2bfloat_avx512(_r1));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 2), float2bfloat_avx512(_r2));
                    _mm256_storeu_si256((__m256i*)(pp + 16 * 3), float2bfloat_avx512(_r3));
#endif // __AVX512BF16__
                    pp += 64;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    __m256 _r0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr0)));
                    __m256 _r1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr1)));
                    __m256 _r2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr2)));
                    __m256 _r3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(sptr3)));
                    transpose8x4_ps(_r0, _r1, _r2, _r3);
#if __AVX512BF16__
                    __m128i _t0 = float2bfloat_avx(_r0);
                    _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_t0, _mm_srli_si128(_t0, 8)));
                    __m128i _t1 = float2bfloat_avx(_r1);
                    _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpacklo_epi16(_t1, _mm_srli_si128(_t1, 8)));
                    __m128i _t2 = float2bfloat_avx(_r2);
                    _mm_storeu_si128((__m128i*)(pp + 16), _mm_unpacklo_epi16(_t2, _mm_srli_si128(_t2, 8)));
                    __m128i _t3 = float2bfloat_avx(_r3);
                    _mm_storeu_si128((__m128i*)(pp + 24), _mm_unpacklo_epi16(_t3, _mm_srli_si128(_t3, 8)));
#else

                    _mm_storeu_si128((__m128i*)pp, float2bfloat_avx(_r0));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 1), float2bfloat_avx(_r1));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 2), float2bfloat_avx(_r2));
                    _mm_storeu_si128((__m128i*)(pp + 8 * 3), float2bfloat_avx(_r3));
#endif // __AVX512BF16__
                    pp += 32;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    __m128 _r0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr0)));
                    __m128 _r1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr1)));
                    __m128 _r2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr2)));
                    __m128 _r3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(sptr3)));
                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
#if __AVX512BF16__
                    __m128i _t0 = float2bfloat_sse(_r0);
                    __m128i _t1 = float2bfloat_sse(_r1);
                    _mm_storeu_si128((__m128i*)(pp + 0), _mm_unpacklo_epi16(_t0, _t1));
                    __m128i _t2 = float2bfloat_sse(_r2);
                    __m128i _t3 = float2bfloat_sse(_r3);
                    _mm_storeu_si128((__m128i*)(pp + 8), _mm_unpacklo_epi16(_t2, _t3));
#else

                    _mm_storel_epi64((__m128i*)pp, float2bfloat_sse(_r0));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 1), float2bfloat_sse(_r1));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 2), float2bfloat_sse(_r2));
                    _mm_storel_epi64((__m128i*)(pp + 4 * 3), float2bfloat_sse(_r3));
#endif // __AVX512BF16__
                    pp += 16;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp += 4;
                }
            }
        }
    }
#endif // __SSE2__
    for (; jj < max_jj; jj++)
    {
        int dy = (j + jj) / outw;
        int dx = (j + jj) % outw;

        int kk = 0;
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x = stride_w * dx + dilation_w * v;
            int y = stride_h * dy + dilation_h * u;

            const unsigned short* sptr = img.row<const unsigned short>(y) + x * elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
            if (elempack == 16)
            {
                _mm256_storeu_si256((__m256i*)pp, _mm256_loadu_si256((const __m256i*)(sptr)));
                pp += 16;
            }
#endif // __AVX512F__
            if (elempack == 8)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)(sptr)));
                pp += 8;
            }
#endif // __AVX__
            if (elempack == 4)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)(sptr)));
                pp += 4;
            }
#endif // __SSE2__
            if (elempack == 1)
            {
                pp[0] = sptr[0];
                pp += 1;
            }
        }
    }
}

template<int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h>
#if __AVX512F__
static void convolution_im2col_input_tile_avx512_bf16s(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
#elif __AVX__
static void convolution_im2col_input_tile_avx_bf16s(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
#else
static void convolution_im2col_input_tile_bf16s(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
#endif
{
    convolution_im2col_input_tile_impl_bf16s(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

#if __AVX512F__
template void convolution_im2col_input_tile_avx512_bf16s<1, 1, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx512_bf16s<3, 3, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx512_bf16s<3, 3, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx512_bf16s<5, 5, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx512_bf16s<5, 5, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx512_bf16s<7, 7, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
#elif __AVX__
template void convolution_im2col_input_tile_avx_bf16s<1, 1, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx_bf16s<3, 3, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx_bf16s<3, 3, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx_bf16s<5, 5, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx_bf16s<5, 5, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_avx_bf16s<7, 7, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
#else
template void convolution_im2col_input_tile_bf16s<1, 1, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_bf16s<3, 3, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_bf16s<3, 3, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_bf16s<5, 5, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_bf16s<5, 5, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_bf16s<7, 7, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
#endif

static void convolution_im2col_input_tile_bf16s(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_conv1x1s1d1_bf16s(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 1 && kernel_h == 1 && stride_w == 2 && stride_h == 2)
    {
#if __AVX512F__
        convolution_im2col_input_tile_avx512_bf16s<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#elif __AVX__
        convolution_im2col_input_tile_avx_bf16s<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#else
        convolution_im2col_input_tile_bf16s<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#endif
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
#if __AVX512F__
        convolution_im2col_input_tile_avx512_bf16s<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#elif __AVX__
        convolution_im2col_input_tile_avx_bf16s<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#else
        convolution_im2col_input_tile_bf16s<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#endif
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
#if __AVX512F__
        convolution_im2col_input_tile_avx512_bf16s<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#elif __AVX__
        convolution_im2col_input_tile_avx_bf16s<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#else
        convolution_im2col_input_tile_bf16s<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#endif
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
#if __AVX512F__
        convolution_im2col_input_tile_avx512_bf16s<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#elif __AVX__
        convolution_im2col_input_tile_avx_bf16s<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#else
        convolution_im2col_input_tile_bf16s<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#endif
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
#if __AVX512F__
        convolution_im2col_input_tile_avx512_bf16s<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#elif __AVX__
        convolution_im2col_input_tile_avx_bf16s<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#else
        convolution_im2col_input_tile_bf16s<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#endif
        return;
    }

    if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
#if __AVX512F__
        convolution_im2col_input_tile_avx512_bf16s<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#elif __AVX__
        convolution_im2col_input_tile_avx_bf16s<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#else
        convolution_im2col_input_tile_bf16s<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#endif
        return;
    }

    convolution_im2col_input_tile_impl_bf16s(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

static void convolution_im2col_gemm_transform_kernel_bf16s(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        convolution_im2col_gemm_transform_kernel_bf16s_avx512bf16(kernel, AT, inch, outch, kernel_w, kernel_h, opt);
        return;
    }
#endif

    // NCNN_LOGE("convolution_im2col_gemm_transform_kernel");
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_bf16s(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    int elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX512F__
        elempack = inch % 16 == 0 ? 16 : inch % 8 == 0 ? 8 : inch % 4 == 0 ? 4 : 1;
#elif __AVX__
        elempack = inch % 8 == 0 ? 8 : inch % 4 == 0 ? 4 : 1;
#else
        elempack = inch % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__

    // maxk-inch-outch to pa-maxk-inch/pa-outch
    Mat A_data;
    if (maxk == 1)
    {
        A_data = kernel.reshape(maxk * inch, outch);
    }
    else
    {
        Mat weight_data_r2 = kernel.reshape(maxk, inch, outch);

        A_data.create(maxk * inch, outch);

        for (int q = 0; q < outch; q += 1)
        {
            float* g00 = A_data.row(q);

            for (int p = 0; p + (elempack - 1) < inch; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        const float* k00 = weight_data_r2.channel(q).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }

    AT.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, (size_t)2u);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            convolution_im2col_pack_A_tile_bf16s(A_data, AT_tile, i, max_ii, k, max_kk);
        }
    }
}

static int convolution_im2col_gemm_bf16s(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, int nT, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        return convolution_im2col_gemm_bf16s_avx512bf16(bottom_blob, top_blob, AT, bias, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, nT, opt);
    }
#endif

    const int maxk = kernel_w * kernel_h;

    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_bf16s(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d nT=%d", M, N, K, TILE_M, TILE_N, TILE_K, nT);

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, opt.workspace_allocator);
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

        // im2col
        convolution_im2col_input_tile_bf16s(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
    }

    Mat topT_tileX;
    if (K > TILE_K)
    {
        topT_tileX.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT_tileX.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat topT_tile;
        if (K > TILE_K)
            topT_tile = topT_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                const Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = k + TILE_K >= K;

                convolution_gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, bias, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end, activation_type, activation_params);
            }
        }
    }

    return 0;
}

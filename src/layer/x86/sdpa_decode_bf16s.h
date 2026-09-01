// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
int sdpa_decode_bf16s_avx512bf16(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVX512BF16__
int sdpa_decode_bf16s_avx2(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt);
#endif

static void sdpa_decode_pack_query_bf16s(const Mat& query, Mat& queryT, int q0, int max_qq)
{
#if __SSE2__
    const int head_dim = query.w;
    const size_t q_cstep = query.cstep * query.elempack;
    unsigned short* queryT_ptr = queryT;
    int qq = 0;
#if __AVX__
#if __AVX512F__
    for (; qq + 15 < max_qq; qq += 16)
    {
        const int q = q0 + qq;
        unsigned short* pQ = queryT_ptr + (size_t)qq * head_dim;
        const unsigned short* qptr = query.channel(q);

        int d = 0;
        for (; d + 15 < head_dim; d += 16)
        {
            __m256i _r0 = _mm256_loadu_si256((const __m256i*)qptr);
            __m256i _r1 = _mm256_loadu_si256((const __m256i*)(qptr + q_cstep));
            __m256i _r2 = _mm256_loadu_si256((const __m256i*)(qptr + q_cstep * 2));
            __m256i _r3 = _mm256_loadu_si256((const __m256i*)(qptr + q_cstep * 3));
            __m256i _r4 = _mm256_loadu_si256((const __m256i*)(qptr + q_cstep * 4));
            __m256i _r5 = _mm256_loadu_si256((const __m256i*)(qptr + q_cstep * 5));
            __m256i _r6 = _mm256_loadu_si256((const __m256i*)(qptr + q_cstep * 6));
            __m256i _r7 = _mm256_loadu_si256((const __m256i*)(qptr + q_cstep * 7));
            __m256i _r8 = _mm256_loadu_si256((const __m256i*)(qptr + q_cstep * 8));
            __m256i _r9 = _mm256_loadu_si256((const __m256i*)(qptr + q_cstep * 9));
            __m256i _ra = _mm256_loadu_si256((const __m256i*)(qptr + q_cstep * 10));
            __m256i _rb = _mm256_loadu_si256((const __m256i*)(qptr + q_cstep * 11));
            __m256i _rc = _mm256_loadu_si256((const __m256i*)(qptr + q_cstep * 12));
            __m256i _rd = _mm256_loadu_si256((const __m256i*)(qptr + q_cstep * 13));
            __m256i _re = _mm256_loadu_si256((const __m256i*)(qptr + q_cstep * 14));
            __m256i _rf = _mm256_loadu_si256((const __m256i*)(qptr + q_cstep * 15));
#if __AVX512BF16__
            transpose8x8_epi32(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            transpose8x8_epi32(_r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);

            __m512i _p0 = combine8x2_epi32(_r0, _r8);
            __m512i _p1 = combine8x2_epi32(_r1, _r9);
            __m512i _p2 = combine8x2_epi32(_r2, _ra);
            __m512i _p3 = combine8x2_epi32(_r3, _rb);
            __m512i _p4 = combine8x2_epi32(_r4, _rc);
            __m512i _p5 = combine8x2_epi32(_r5, _rd);
            __m512i _p6 = combine8x2_epi32(_r6, _re);
            __m512i _p7 = combine8x2_epi32(_r7, _rf);

            _mm512_storeu_si512((__m512i*)pQ, _p0);
            _mm512_storeu_si512((__m512i*)(pQ + 32), _p1);
            _mm512_storeu_si512((__m512i*)(pQ + 64), _p2);
            _mm512_storeu_si512((__m512i*)(pQ + 96), _p3);
            _mm512_storeu_si512((__m512i*)(pQ + 128), _p4);
            _mm512_storeu_si512((__m512i*)(pQ + 160), _p5);
            _mm512_storeu_si512((__m512i*)(pQ + 192), _p6);
            _mm512_storeu_si512((__m512i*)(pQ + 224), _p7);
#else
            transpose16x16_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
            _mm256_storeu_si256((__m256i*)pQ, _r0);
            _mm256_storeu_si256((__m256i*)(pQ + 16), _r1);
            _mm256_storeu_si256((__m256i*)(pQ + 32), _r2);
            _mm256_storeu_si256((__m256i*)(pQ + 48), _r3);
            _mm256_storeu_si256((__m256i*)(pQ + 64), _r4);
            _mm256_storeu_si256((__m256i*)(pQ + 80), _r5);
            _mm256_storeu_si256((__m256i*)(pQ + 96), _r6);
            _mm256_storeu_si256((__m256i*)(pQ + 112), _r7);
            _mm256_storeu_si256((__m256i*)(pQ + 128), _r8);
            _mm256_storeu_si256((__m256i*)(pQ + 144), _r9);
            _mm256_storeu_si256((__m256i*)(pQ + 160), _ra);
            _mm256_storeu_si256((__m256i*)(pQ + 176), _rb);
            _mm256_storeu_si256((__m256i*)(pQ + 192), _rc);
            _mm256_storeu_si256((__m256i*)(pQ + 208), _rd);
            _mm256_storeu_si256((__m256i*)(pQ + 224), _re);
            _mm256_storeu_si256((__m256i*)(pQ + 240), _rf);
#endif // __AVX512BF16__
            qptr += 16;
            pQ += 256;
        }
#if __AVX512BF16__
        for (; d + 1 < head_dim; d += 2)
        {
            pQ[0] = qptr[0];
            pQ[1] = qptr[1];
            pQ[2] = qptr[q_cstep];
            pQ[3] = qptr[q_cstep + 1];
            pQ[4] = qptr[q_cstep * 2];
            pQ[5] = qptr[q_cstep * 2 + 1];
            pQ[6] = qptr[q_cstep * 3];
            pQ[7] = qptr[q_cstep * 3 + 1];
            pQ[8] = qptr[q_cstep * 4];
            pQ[9] = qptr[q_cstep * 4 + 1];
            pQ[10] = qptr[q_cstep * 5];
            pQ[11] = qptr[q_cstep * 5 + 1];
            pQ[12] = qptr[q_cstep * 6];
            pQ[13] = qptr[q_cstep * 6 + 1];
            pQ[14] = qptr[q_cstep * 7];
            pQ[15] = qptr[q_cstep * 7 + 1];
            pQ[16] = qptr[q_cstep * 8];
            pQ[17] = qptr[q_cstep * 8 + 1];
            pQ[18] = qptr[q_cstep * 9];
            pQ[19] = qptr[q_cstep * 9 + 1];
            pQ[20] = qptr[q_cstep * 10];
            pQ[21] = qptr[q_cstep * 10 + 1];
            pQ[22] = qptr[q_cstep * 11];
            pQ[23] = qptr[q_cstep * 11 + 1];
            pQ[24] = qptr[q_cstep * 12];
            pQ[25] = qptr[q_cstep * 12 + 1];
            pQ[26] = qptr[q_cstep * 13];
            pQ[27] = qptr[q_cstep * 13 + 1];
            pQ[28] = qptr[q_cstep * 14];
            pQ[29] = qptr[q_cstep * 14 + 1];
            pQ[30] = qptr[q_cstep * 15];
            pQ[31] = qptr[q_cstep * 15 + 1];
            qptr += 2;
            pQ += 32;
        }
#endif // __AVX512BF16__
        for (; d < head_dim; d++)
        {
            pQ[0] = qptr[0];
            pQ[1] = qptr[q_cstep];
            pQ[2] = qptr[q_cstep * 2];
            pQ[3] = qptr[q_cstep * 3];
            pQ[4] = qptr[q_cstep * 4];
            pQ[5] = qptr[q_cstep * 5];
            pQ[6] = qptr[q_cstep * 6];
            pQ[7] = qptr[q_cstep * 7];
            pQ[8] = qptr[q_cstep * 8];
            pQ[9] = qptr[q_cstep * 9];
            pQ[10] = qptr[q_cstep * 10];
            pQ[11] = qptr[q_cstep * 11];
            pQ[12] = qptr[q_cstep * 12];
            pQ[13] = qptr[q_cstep * 13];
            pQ[14] = qptr[q_cstep * 14];
            pQ[15] = qptr[q_cstep * 15];
            qptr++;
            pQ += 16;
        }
    }
#endif // __AVX512F__
    for (; qq + 7 < max_qq; qq += 8)
    {
        const int q = q0 + qq;
        unsigned short* pQ = queryT_ptr + (size_t)qq * head_dim;
        const unsigned short* qptr = query.channel(q);

        int d = 0;
        for (; d + 7 < head_dim; d += 8)
        {
            __m128i _r0 = _mm_loadu_si128((const __m128i*)qptr);
            __m128i _r1 = _mm_loadu_si128((const __m128i*)(qptr + q_cstep));
            __m128i _r2 = _mm_loadu_si128((const __m128i*)(qptr + q_cstep * 2));
            __m128i _r3 = _mm_loadu_si128((const __m128i*)(qptr + q_cstep * 3));
            __m128i _r4 = _mm_loadu_si128((const __m128i*)(qptr + q_cstep * 4));
            __m128i _r5 = _mm_loadu_si128((const __m128i*)(qptr + q_cstep * 5));
            __m128i _r6 = _mm_loadu_si128((const __m128i*)(qptr + q_cstep * 6));
            __m128i _r7 = _mm_loadu_si128((const __m128i*)(qptr + q_cstep * 7));
#if __AVX512BF16__
            transpose4x4_epi32(_r0, _r1, _r2, _r3);
            transpose4x4_epi32(_r4, _r5, _r6, _r7);

            __m256i _p0 = combine4x2_epi32(_r0, _r4);
            __m256i _p1 = combine4x2_epi32(_r1, _r5);
            __m256i _p2 = combine4x2_epi32(_r2, _r6);
            __m256i _p3 = combine4x2_epi32(_r3, _r7);

            _mm256_storeu_si256((__m256i*)pQ, _p0);
            _mm256_storeu_si256((__m256i*)(pQ + 16), _p1);
            _mm256_storeu_si256((__m256i*)(pQ + 32), _p2);
            _mm256_storeu_si256((__m256i*)(pQ + 48), _p3);
#else
            transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            _mm_storeu_si128((__m128i*)pQ, _r0);
            _mm_storeu_si128((__m128i*)(pQ + 8), _r1);
            _mm_storeu_si128((__m128i*)(pQ + 16), _r2);
            _mm_storeu_si128((__m128i*)(pQ + 24), _r3);
            _mm_storeu_si128((__m128i*)(pQ + 32), _r4);
            _mm_storeu_si128((__m128i*)(pQ + 40), _r5);
            _mm_storeu_si128((__m128i*)(pQ + 48), _r6);
            _mm_storeu_si128((__m128i*)(pQ + 56), _r7);
#endif // __AVX512BF16__
            qptr += 8;
            pQ += 64;
        }
#if __AVX512BF16__
        for (; d + 1 < head_dim; d += 2)
        {
            pQ[0] = qptr[0];
            pQ[1] = qptr[1];
            pQ[2] = qptr[q_cstep];
            pQ[3] = qptr[q_cstep + 1];
            pQ[4] = qptr[q_cstep * 2];
            pQ[5] = qptr[q_cstep * 2 + 1];
            pQ[6] = qptr[q_cstep * 3];
            pQ[7] = qptr[q_cstep * 3 + 1];
            pQ[8] = qptr[q_cstep * 4];
            pQ[9] = qptr[q_cstep * 4 + 1];
            pQ[10] = qptr[q_cstep * 5];
            pQ[11] = qptr[q_cstep * 5 + 1];
            pQ[12] = qptr[q_cstep * 6];
            pQ[13] = qptr[q_cstep * 6 + 1];
            pQ[14] = qptr[q_cstep * 7];
            pQ[15] = qptr[q_cstep * 7 + 1];
            qptr += 2;
            pQ += 16;
        }
#endif // __AVX512BF16__
        for (; d < head_dim; d++)
        {
            pQ[0] = qptr[0];
            pQ[1] = qptr[q_cstep];
            pQ[2] = qptr[q_cstep * 2];
            pQ[3] = qptr[q_cstep * 3];
            pQ[4] = qptr[q_cstep * 4];
            pQ[5] = qptr[q_cstep * 5];
            pQ[6] = qptr[q_cstep * 6];
            pQ[7] = qptr[q_cstep * 7];
            qptr++;
            pQ += 8;
        }
    }
#endif // __AVX__
    for (; qq + 3 < max_qq; qq += 4)
    {
        const int q = q0 + qq;
        unsigned short* pQ = queryT_ptr + (size_t)qq * head_dim;
        const unsigned short* qptr = query.channel(q);

        int d = 0;
        for (; d + 3 < head_dim; d += 4)
        {
            __m128i _r0 = _mm_loadl_epi64((const __m128i*)qptr);
            __m128i _r1 = _mm_loadl_epi64((const __m128i*)(qptr + q_cstep));
            __m128i _r2 = _mm_loadl_epi64((const __m128i*)(qptr + q_cstep * 2));
            __m128i _r3 = _mm_loadl_epi64((const __m128i*)(qptr + q_cstep * 3));
#if __AVX512BF16__
            __m128i _tmp0 = _mm_unpacklo_epi32(_r0, _r1);
            __m128i _tmp1 = _mm_unpacklo_epi32(_r2, _r3);
            _r0 = _mm_unpacklo_epi64(_tmp0, _tmp1);
            _r1 = _mm_unpackhi_epi64(_tmp0, _tmp1);
#else
            __m128i _tmp0 = _mm_unpacklo_epi16(_r0, _r1);
            __m128i _tmp1 = _mm_unpacklo_epi16(_r2, _r3);
            _r0 = _mm_unpacklo_epi32(_tmp0, _tmp1);
            _r1 = _mm_unpackhi_epi32(_tmp0, _tmp1);
#endif // __AVX512BF16__
            _mm_storeu_si128((__m128i*)pQ, _r0);
            _mm_storeu_si128((__m128i*)(pQ + 8), _r1);
            qptr += 4;
            pQ += 16;
        }
#if __AVX512BF16__
        for (; d + 1 < head_dim; d += 2)
        {
            pQ[0] = qptr[0];
            pQ[1] = qptr[1];
            pQ[2] = qptr[q_cstep];
            pQ[3] = qptr[q_cstep + 1];
            pQ[4] = qptr[q_cstep * 2];
            pQ[5] = qptr[q_cstep * 2 + 1];
            pQ[6] = qptr[q_cstep * 3];
            pQ[7] = qptr[q_cstep * 3 + 1];
            qptr += 2;
            pQ += 8;
        }
#endif // __AVX512BF16__
        for (; d < head_dim; d++)
        {
            pQ[0] = qptr[0];
            pQ[1] = qptr[q_cstep];
            pQ[2] = qptr[q_cstep * 2];
            pQ[3] = qptr[q_cstep * 3];
            qptr++;
            pQ += 4;
        }
    }
#else
    (void)query;
    (void)queryT;
    (void)q0;
    (void)max_qq;
#endif // __SSE2__
}

static void sdpa_decode_tile_bf16s(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, int q0, int max_qq, int g, int block_n, Mat& workspace)
{
    const int key_seqlen = key.h;
    int qq = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; qq + 15 < max_qq; qq += 16)
    {
        const int q = q0 + qq;
        const int head_dim = query.w;
        const int value_dim = value.w;

        const bool mask_per_head = !attn_mask_blob.empty() && attn_mask_blob.dims == 3 && attn_mask_blob.c > 1;
        const unsigned short* mask = 0;
        if (!attn_mask_blob.empty())
        {
            if (attn_mask_blob.dims == 3)
                mask = attn_mask_blob.channel(mask_per_head ? q : 0);
            else
                mask = attn_mask_blob;
        }
        const int mask_hstep = mask_per_head ? (int)attn_mask_blob.cstep : 0;
        __m512i _mask_index = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(mask_hstep));

        float* scoreT = workspace;
        float* outT = scoreT + block_n * 16;
        Mat queryT_blob(head_dim * 16, (unsigned short*)(outT + value_dim * 16), 2u);
        sdpa_decode_pack_query_bf16s(query, queryT_blob, q, 16);
        const unsigned short* queryT = queryT_blob;
        memset(outT, 0, (size_t)value_dim * 16 * sizeof(float));

        const Mat key_head = key.channel(g);
        const Mat value_head = value.channel(g);
        __m512 _m = _mm512_set1_ps(-FLT_MAX);
        __m512 _l = _mm512_setzero_ps();
        __m512 _scale = _mm512_set1_ps(scale);

        for (int n = 0; n < key_seqlen; n += block_n)
        {
            const int max_jj = std::min(key_seqlen - n, block_n);
            __m512 _block_max = _mm512_set1_ps(-FLT_MAX);
            const unsigned short* pK = key_head.row<const unsigned short>(n);
            float* pS = scoreT;
            const unsigned short* pM = mask ? mask + n : 0;
            for (int j = 0; j < max_jj; j++)
            {
                __m512 _sum0 = _mm512_setzero_ps();
                __m512 _sum1 = _mm512_setzero_ps();
                __m512 _sum2 = _mm512_setzero_ps();
                __m512 _sum3 = _mm512_setzero_ps();
                const unsigned short* pQ = queryT;
                int d = 0;
#if __AVX512BF16__
                for (; d + 1 < head_dim; d += 2)
                {
                    __m512i _q = _mm512_loadu_si512((const __m512i*)pQ);
                    _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(((const int*)pK)[0]));
                    pQ += 32;
                    pK += 2;
                }
#endif // __AVX512BF16__
                for (; d + 3 < head_dim; d += 4)
                {
                    _sum0 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pQ)), _mm512_set1_ps(bfloat16_to_float32(pK[0])), _sum0);
                    _sum1 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pQ + 16))), _mm512_set1_ps(bfloat16_to_float32(pK[1])), _sum1);
                    _sum2 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pQ + 32))), _mm512_set1_ps(bfloat16_to_float32(pK[2])), _sum2);
                    _sum3 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pQ + 48))), _mm512_set1_ps(bfloat16_to_float32(pK[3])), _sum3);
                    pQ += 64;
                    pK += 4;
                }
                for (; d < head_dim; d++)
                {
                    _sum0 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pQ)), _mm512_set1_ps(bfloat16_to_float32(*pK)), _sum0);
                    pQ += 16;
                    pK++;
                }
                __m512 _score = _mm512_mul_ps(_mm512_add_ps(_mm512_add_ps(_sum0, _sum1), _mm512_add_ps(_sum2, _sum3)), _scale);
                if (pM)
                {
                    __m512i _mask_bf16 = _mm512_i32gather_epi32(_mask_index, (const int*)pM, sizeof(unsigned short));
                    __m512 _mask = bfloat2float_avx512(_mm512_cvtepi32_epi16(_mask_bf16));
                    _score = _mm512_add_ps(_score, _mask);
                    pM++;
                }
                _mm512_storeu_ps(pS, _score);
                pS += 16;
                _block_max = _mm512_max_ps(_block_max, _score);
            }

            __m512 _m_new = _mm512_max_ps(_m, _block_max);
            __mmask16 alpha_active = _mm512_cmp_ps_mask(_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
            __m512 _alpha = _mm512_maskz_mov_ps(alpha_active, exp512_ps(_mm512_maskz_sub_ps(alpha_active, _m, _m_new)));

            float* scoreptr = scoreT;
            __m512 _sum0 = _mm512_setzero_ps();
            int j = 0;
#if defined(__x86_64__) || defined(_M_X64)
            __m512 _sum1 = _mm512_setzero_ps();
            __m512 _sum2 = _mm512_setzero_ps();
            __m512 _sum3 = _mm512_setzero_ps();
            for (; j + 3 < max_jj; j += 4)
            {
                __m512 _p0 = _mm512_sub_ps(_mm512_loadu_ps(scoreptr), _m_new);
                _p0 = exp512_ps(_p0);
                _mm512_storeu_ps(scoreptr, _p0);
                _sum0 = _mm512_add_ps(_sum0, _p0);
                __m512 _p1 = _mm512_sub_ps(_mm512_loadu_ps(scoreptr + 16), _m_new);
                _p1 = exp512_ps(_p1);
                _mm512_storeu_ps(scoreptr + 16, _p1);
                _sum1 = _mm512_add_ps(_sum1, _p1);
                __m512 _p2 = _mm512_sub_ps(_mm512_loadu_ps(scoreptr + 32), _m_new);
                _p2 = exp512_ps(_p2);
                _mm512_storeu_ps(scoreptr + 32, _p2);
                _sum2 = _mm512_add_ps(_sum2, _p2);
                __m512 _p3 = _mm512_sub_ps(_mm512_loadu_ps(scoreptr + 48), _m_new);
                _p3 = exp512_ps(_p3);
                _mm512_storeu_ps(scoreptr + 48, _p3);
                _sum3 = _mm512_add_ps(_sum3, _p3);
                scoreptr += 64;
            }
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; j < max_jj; j++)
            {
                __m512 _p = _mm512_sub_ps(_mm512_loadu_ps(scoreptr), _m_new);
                _p = exp512_ps(_p);
                _mm512_storeu_ps(scoreptr, _p);
                scoreptr += 16;
                _sum0 = _mm512_add_ps(_sum0, _p);
            }
            __m512 _sum = _sum0;
#if defined(__x86_64__) || defined(_M_X64)
            _sum = _mm512_add_ps(_mm512_add_ps(_sum, _sum1), _mm512_add_ps(_sum2, _sum3));
#endif // defined(__x86_64__) || defined(_M_X64)
            _l = _mm512_add_ps(_mm512_mul_ps(_l, _alpha), _sum);
            _m = _m_new;

            float* outptr = outT;
            const unsigned short* valueptr = value_head.row<const unsigned short>(n);
            int d = 0;
#if defined(__x86_64__) || defined(_M_X64)
            for (; d + 7 < value_dim; d += 8)
            {
                __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 0), _alpha);
                __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 16), _alpha);
                __m512 _out2 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 32), _alpha);
                __m512 _out3 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 48), _alpha);
                __m512 _out4 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 64), _alpha);
                __m512 _out5 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 80), _alpha);
                __m512 _out6 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 96), _alpha);
                __m512 _out7 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 112), _alpha);
                const unsigned short* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    __m512 _p = _mm512_loadu_ps(pS);
                    _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[0])), _out0);
                    _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[1])), _out1);
                    _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[2])), _out2);
                    _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[3])), _out3);
                    _out4 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[4])), _out4);
                    _out5 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[5])), _out5);
                    _out6 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[6])), _out6);
                    _out7 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[7])), _out7);
                    pS += 16;
                    pV += value_dim;
                }
                _mm512_storeu_ps(outptr + 0, _out0);
                _mm512_storeu_ps(outptr + 16, _out1);
                _mm512_storeu_ps(outptr + 32, _out2);
                _mm512_storeu_ps(outptr + 48, _out3);
                _mm512_storeu_ps(outptr + 64, _out4);
                _mm512_storeu_ps(outptr + 80, _out5);
                _mm512_storeu_ps(outptr + 96, _out6);
                _mm512_storeu_ps(outptr + 112, _out7);
                outptr += 128;
                valueptr += 8;
            }
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; d + 3 < value_dim; d += 4)
            {
                __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 0), _alpha);
                __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 16), _alpha);
                __m512 _out2 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 32), _alpha);
                __m512 _out3 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 48), _alpha);
                const unsigned short* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    __m512 _p = _mm512_loadu_ps(pS);
                    _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[0])), _out0);
                    _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[1])), _out1);
                    _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[2])), _out2);
                    _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[3])), _out3);
                    pS += 16;
                    pV += value_dim;
                }
                _mm512_storeu_ps(outptr + 0, _out0);
                _mm512_storeu_ps(outptr + 16, _out1);
                _mm512_storeu_ps(outptr + 32, _out2);
                _mm512_storeu_ps(outptr + 48, _out3);
                outptr += 64;
                valueptr += 4;
            }
            for (; d < value_dim; d++)
            {
                __m512 _out = _mm512_mul_ps(_mm512_loadu_ps(outptr), _alpha);
                const unsigned short* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    _out = _mm512_fmadd_ps(_mm512_loadu_ps(pS), _mm512_set1_ps(bfloat16_to_float32(*pV)), _out);
                    pS += 16;
                    pV += value_dim;
                }
                _mm512_storeu_ps(outptr, _out);
                outptr += 16;
                valueptr++;
            }
        }

        {
            float* output = top_blob.channel(q);
            const size_t output_cstep = top_blob.cstep;
            __mmask16 nonzero = _mm512_cmp_ps_mask(_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
            __m512 _out_scale = _mm512_maskz_div_ps(nonzero, _mm512_set1_ps(1.f), _l);

            const float* outptr = outT;
            float* p0 = output;
            int d = 0;
            for (; d + 15 < value_dim; d += 16)
            {
                __m512 _r0 = _mm512_mul_ps(_mm512_loadu_ps(outptr), _out_scale);
                __m512 _r1 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 16), _out_scale);
                __m512 _r2 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 32), _out_scale);
                __m512 _r3 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 48), _out_scale);
                __m512 _r4 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 64), _out_scale);
                __m512 _r5 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 80), _out_scale);
                __m512 _r6 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 96), _out_scale);
                __m512 _r7 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 112), _out_scale);
                __m512 _r8 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 128), _out_scale);
                __m512 _r9 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 144), _out_scale);
                __m512 _ra = _mm512_mul_ps(_mm512_loadu_ps(outptr + 160), _out_scale);
                __m512 _rb = _mm512_mul_ps(_mm512_loadu_ps(outptr + 176), _out_scale);
                __m512 _rc = _mm512_mul_ps(_mm512_loadu_ps(outptr + 192), _out_scale);
                __m512 _rd = _mm512_mul_ps(_mm512_loadu_ps(outptr + 208), _out_scale);
                __m512 _re = _mm512_mul_ps(_mm512_loadu_ps(outptr + 224), _out_scale);
                __m512 _rf = _mm512_mul_ps(_mm512_loadu_ps(outptr + 240), _out_scale);
                transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                _mm512_storeu_ps(p0, _r0);
                _mm512_storeu_ps(p0 + output_cstep, _r1);
                _mm512_storeu_ps(p0 + output_cstep * 2, _r2);
                _mm512_storeu_ps(p0 + output_cstep * 3, _r3);
                _mm512_storeu_ps(p0 + output_cstep * 4, _r4);
                _mm512_storeu_ps(p0 + output_cstep * 5, _r5);
                _mm512_storeu_ps(p0 + output_cstep * 6, _r6);
                _mm512_storeu_ps(p0 + output_cstep * 7, _r7);
                _mm512_storeu_ps(p0 + output_cstep * 8, _r8);
                _mm512_storeu_ps(p0 + output_cstep * 9, _r9);
                _mm512_storeu_ps(p0 + output_cstep * 10, _ra);
                _mm512_storeu_ps(p0 + output_cstep * 11, _rb);
                _mm512_storeu_ps(p0 + output_cstep * 12, _rc);
                _mm512_storeu_ps(p0 + output_cstep * 13, _rd);
                _mm512_storeu_ps(p0 + output_cstep * 14, _re);
                _mm512_storeu_ps(p0 + output_cstep * 15, _rf);
                p0 += 16;
                outptr += 256;
            }
            for (; d < value_dim; d++)
            {
                __m512 _r = _mm512_mul_ps(_mm512_loadu_ps(outptr), _out_scale);
                __m128 _r0 = _mm512_extractf32x4_ps(_r, 0);
                __m128 _r1 = _mm512_extractf32x4_ps(_r, 1);
                __m128 _r2 = _mm512_extractf32x4_ps(_r, 2);
                __m128 _r3 = _mm512_extractf32x4_ps(_r, 3);
                *p0 = _mm_cvtss_f32(_r0);
                p0[output_cstep] = _mm_cvtss_f32(_mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(1, 1, 1, 1)));
                p0[output_cstep * 2] = _mm_cvtss_f32(_mm_movehl_ps(_r0, _r0));
                p0[output_cstep * 3] = _mm_cvtss_f32(_mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(3, 3, 3, 3)));
                p0[output_cstep * 4] = _mm_cvtss_f32(_r1);
                p0[output_cstep * 5] = _mm_cvtss_f32(_mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(1, 1, 1, 1)));
                p0[output_cstep * 6] = _mm_cvtss_f32(_mm_movehl_ps(_r1, _r1));
                p0[output_cstep * 7] = _mm_cvtss_f32(_mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(3, 3, 3, 3)));
                p0[output_cstep * 8] = _mm_cvtss_f32(_r2);
                p0[output_cstep * 9] = _mm_cvtss_f32(_mm_shuffle_ps(_r2, _r2, _MM_SHUFFLE(1, 1, 1, 1)));
                p0[output_cstep * 10] = _mm_cvtss_f32(_mm_movehl_ps(_r2, _r2));
                p0[output_cstep * 11] = _mm_cvtss_f32(_mm_shuffle_ps(_r2, _r2, _MM_SHUFFLE(3, 3, 3, 3)));
                p0[output_cstep * 12] = _mm_cvtss_f32(_r3);
                p0[output_cstep * 13] = _mm_cvtss_f32(_mm_shuffle_ps(_r3, _r3, _MM_SHUFFLE(1, 1, 1, 1)));
                p0[output_cstep * 14] = _mm_cvtss_f32(_mm_movehl_ps(_r3, _r3));
                p0[output_cstep * 15] = _mm_cvtss_f32(_mm_shuffle_ps(_r3, _r3, _MM_SHUFFLE(3, 3, 3, 3)));
                p0++;
                outptr += 16;
            }
        }
    }
#endif // __AVX512F__
    for (; qq + 7 < max_qq; qq += 8)
    {
        const int q = q0 + qq;
        const int head_dim = query.w;
        const int value_dim = value.w;

        const bool mask_per_head = !attn_mask_blob.empty() && attn_mask_blob.dims == 3 && attn_mask_blob.c > 1;
        const unsigned short* mask = 0;
        if (!attn_mask_blob.empty())
        {
            if (attn_mask_blob.dims == 3)
                mask = attn_mask_blob.channel(mask_per_head ? q : 0);
            else
                mask = attn_mask_blob;
        }
        const int mask_hstep = mask_per_head ? (int)attn_mask_blob.cstep : 0;
#if __AVX2__
        __m256i _mask_index = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(mask_hstep));
#endif // __AVX2__

        float* scoreT = workspace;
        float* outT = scoreT + block_n * 8;
        Mat queryT_blob(head_dim * 8, (unsigned short*)(outT + value_dim * 8), 2u);
        sdpa_decode_pack_query_bf16s(query, queryT_blob, q, 8);
        const unsigned short* queryT = queryT_blob;
        memset(outT, 0, (size_t)value_dim * 8 * sizeof(float));

        const Mat key_head = key.channel(g);
        const Mat value_head = value.channel(g);
        __m256 _m = _mm256_set1_ps(-FLT_MAX);
        __m256 _l = _mm256_setzero_ps();
        __m256 _scale = _mm256_set1_ps(scale);

        for (int n = 0; n < key_seqlen; n += block_n)
        {
            const int max_jj = std::min(key_seqlen - n, block_n);
            __m256 _block_max = _mm256_set1_ps(-FLT_MAX);
            const unsigned short* pK = key_head.row<const unsigned short>(n);
            float* pS = scoreT;
            const unsigned short* pM = mask ? mask + n : 0;
            for (int j = 0; j < max_jj; j++)
            {
                __m256 _sum0 = _mm256_setzero_ps();
                __m256 _sum1 = _mm256_setzero_ps();
                __m256 _sum2 = _mm256_setzero_ps();
                __m256 _sum3 = _mm256_setzero_ps();
                const unsigned short* pQ = queryT;
                int d = 0;
#if __AVX512BF16__
                for (; d + 1 < head_dim; d += 2)
                {
                    __m256i _q = _mm256_loadu_si256((const __m256i*)pQ);
                    _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(((const int*)pK)[0]));
                    pQ += 16;
                    pK += 2;
                }
#endif // __AVX512BF16__
                for (; d + 3 < head_dim; d += 4)
                {
                    _sum0 = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)pQ)), _mm256_set1_ps(bfloat16_to_float32(pK[0])), _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pQ + 8))), _mm256_set1_ps(bfloat16_to_float32(pK[1])), _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pQ + 16))), _mm256_set1_ps(bfloat16_to_float32(pK[2])), _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pQ + 24))), _mm256_set1_ps(bfloat16_to_float32(pK[3])), _sum3);
                    pQ += 32;
                    pK += 4;
                }
                for (; d < head_dim; d++)
                {
                    _sum0 = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)pQ)), _mm256_set1_ps(bfloat16_to_float32(*pK)), _sum0);
                    pQ += 8;
                    pK++;
                }
                __m256 _score = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(_sum0, _sum1), _mm256_add_ps(_sum2, _sum3)), _scale);
                if (pM)
                {
#if __AVX2__
                    __m256i _mask_bf16 = _mm256_i32gather_epi32((const int*)pM, _mask_index, sizeof(unsigned short));
                    __m256 _mask = _mm256_castsi256_ps(_mm256_slli_epi32(_mask_bf16, 16));
#else
                    __m256 _mask = _mm256_set_ps(bfloat16_to_float32(pM[mask_hstep * 7]), bfloat16_to_float32(pM[mask_hstep * 6]), bfloat16_to_float32(pM[mask_hstep * 5]), bfloat16_to_float32(pM[mask_hstep * 4]), bfloat16_to_float32(pM[mask_hstep * 3]), bfloat16_to_float32(pM[mask_hstep * 2]), bfloat16_to_float32(pM[mask_hstep]), bfloat16_to_float32(pM[0]));
#endif // __AVX2__
                    _score = _mm256_add_ps(_score, _mask);
                    pM++;
                }
                _mm256_storeu_ps(pS, _score);
                pS += 8;
                _block_max = _mm256_max_ps(_block_max, _score);
            }

            __m256 _m_new = _mm256_max_ps(_m, _block_max);
            __m256 _alpha_active = _mm256_cmp_ps(_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
            __m256 _alpha = _mm256_and_ps(_alpha_active, exp256_ps(_mm256_and_ps(_alpha_active, _mm256_sub_ps(_m, _m_new))));

            float* scoreptr = scoreT;
            __m256 _sum0 = _mm256_setzero_ps();
            int j = 0;
#if defined(__x86_64__) || defined(_M_X64)
            __m256 _sum1 = _mm256_setzero_ps();
            __m256 _sum2 = _mm256_setzero_ps();
            __m256 _sum3 = _mm256_setzero_ps();
            for (; j + 3 < max_jj; j += 4)
            {
                __m256 _p0 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr), _m_new));
                __m256 _p1 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr + 8), _m_new));
                __m256 _p2 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr + 16), _m_new));
                __m256 _p3 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr + 24), _m_new));
                _mm256_storeu_ps(scoreptr, _p0);
                _mm256_storeu_ps(scoreptr + 8, _p1);
                _mm256_storeu_ps(scoreptr + 16, _p2);
                _mm256_storeu_ps(scoreptr + 24, _p3);
                _sum0 = _mm256_add_ps(_sum0, _p0);
                _sum1 = _mm256_add_ps(_sum1, _p1);
                _sum2 = _mm256_add_ps(_sum2, _p2);
                _sum3 = _mm256_add_ps(_sum3, _p3);
                scoreptr += 32;
            }
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; j < max_jj; j++)
            {
                __m256 _p = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr), _m_new));
                _mm256_storeu_ps(scoreptr, _p);
                scoreptr += 8;
                _sum0 = _mm256_add_ps(_sum0, _p);
            }
            __m256 _sum = _sum0;
#if defined(__x86_64__) || defined(_M_X64)
            _sum = _mm256_add_ps(_mm256_add_ps(_sum, _sum1), _mm256_add_ps(_sum2, _sum3));
#endif // defined(__x86_64__) || defined(_M_X64)
            _l = _mm256_add_ps(_mm256_mul_ps(_l, _alpha), _sum);
            _m = _m_new;

            float* outptr = outT;
            const unsigned short* valueptr = value_head.row<const unsigned short>(n);
            int d = 0;
            for (; d + 3 < value_dim; d += 4)
            {
                __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr), _alpha);
                __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 8), _alpha);
                __m256 _out2 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 16), _alpha);
                __m256 _out3 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 24), _alpha);
                const unsigned short* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    __m256 _p = _mm256_loadu_ps(pS);
                    _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[0])), _out0);
                    _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[1])), _out1);
                    _out2 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[2])), _out2);
                    _out3 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[3])), _out3);
                    pS += 8;
                    pV += value_dim;
                }
                _mm256_storeu_ps(outptr, _out0);
                _mm256_storeu_ps(outptr + 8, _out1);
                _mm256_storeu_ps(outptr + 16, _out2);
                _mm256_storeu_ps(outptr + 24, _out3);
                outptr += 32;
                valueptr += 4;
            }
            for (; d < value_dim; d++)
            {
                __m256 _out = _mm256_mul_ps(_mm256_loadu_ps(outptr), _alpha);
                const unsigned short* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pS), _mm256_set1_ps(bfloat16_to_float32(*pV)), _out);
                    pS += 8;
                    pV += value_dim;
                }
                _mm256_storeu_ps(outptr, _out);
                outptr += 8;
                valueptr++;
            }
        }

        {
            float* output = top_blob.channel(q);
            const size_t output_cstep = top_blob.cstep;
            __m256 _nonzero = _mm256_cmp_ps(_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
            __m256 _denom = _mm256_blendv_ps(_mm256_set1_ps(1.f), _l, _nonzero);
            __m256 _out_scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(1.f), _denom), _nonzero);

            const float* outptr = outT;
            float* p0 = output;
            int d = 0;
            for (; d + 7 < value_dim; d += 8)
            {
                __m256 _r0 = _mm256_mul_ps(_mm256_loadu_ps(outptr), _out_scale);
                __m256 _r1 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 8), _out_scale);
                __m256 _r2 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 16), _out_scale);
                __m256 _r3 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 24), _out_scale);
                __m256 _r4 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 32), _out_scale);
                __m256 _r5 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 40), _out_scale);
                __m256 _r6 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 48), _out_scale);
                __m256 _r7 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 56), _out_scale);
                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm256_storeu_ps(p0, _r0);
                _mm256_storeu_ps(p0 + output_cstep, _r1);
                _mm256_storeu_ps(p0 + output_cstep * 2, _r2);
                _mm256_storeu_ps(p0 + output_cstep * 3, _r3);
                _mm256_storeu_ps(p0 + output_cstep * 4, _r4);
                _mm256_storeu_ps(p0 + output_cstep * 5, _r5);
                _mm256_storeu_ps(p0 + output_cstep * 6, _r6);
                _mm256_storeu_ps(p0 + output_cstep * 7, _r7);
                p0 += 8;
                outptr += 64;
            }
            for (; d < value_dim; d++)
            {
                __m256 _r = _mm256_mul_ps(_mm256_loadu_ps(outptr), _out_scale);
                __m128 _r0 = _mm256_castps256_ps128(_r);
                __m128 _r1 = _mm256_extractf128_ps(_r, 1);
                *p0 = _mm_cvtss_f32(_r0);
                p0[output_cstep] = _mm_cvtss_f32(_mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(1, 1, 1, 1)));
                p0[output_cstep * 2] = _mm_cvtss_f32(_mm_movehl_ps(_r0, _r0));
                p0[output_cstep * 3] = _mm_cvtss_f32(_mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(3, 3, 3, 3)));
                p0[output_cstep * 4] = _mm_cvtss_f32(_r1);
                p0[output_cstep * 5] = _mm_cvtss_f32(_mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(1, 1, 1, 1)));
                p0[output_cstep * 6] = _mm_cvtss_f32(_mm_movehl_ps(_r1, _r1));
                p0[output_cstep * 7] = _mm_cvtss_f32(_mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(3, 3, 3, 3)));
                p0++;
                outptr += 8;
            }
        }
    }
#endif // __AVX__
    for (; qq + 3 < max_qq; qq += 4)
    {
        const int q = q0 + qq;
        const int head_dim = query.w;
        const int value_dim = value.w;

        const bool mask_per_head = !attn_mask_blob.empty() && attn_mask_blob.dims == 3 && attn_mask_blob.c > 1;
        const unsigned short* mask = 0;
        if (!attn_mask_blob.empty())
        {
            if (attn_mask_blob.dims == 3)
                mask = attn_mask_blob.channel(mask_per_head ? q : 0);
            else
                mask = attn_mask_blob;
        }
        const size_t mask_cstep = mask_per_head ? attn_mask_blob.cstep : 0;

        float* scoreT = workspace;
        float* outT = scoreT + block_n * 4;
        Mat queryT_blob(head_dim * 4, (unsigned short*)(outT + value_dim * 4), 2u);
        sdpa_decode_pack_query_bf16s(query, queryT_blob, q, 4);
        const unsigned short* queryT = queryT_blob;
        memset(outT, 0, (size_t)value_dim * 4 * sizeof(float));

        const Mat key_head = key.channel(g);
        const Mat value_head = value.channel(g);
#if __AVX__
        const size_t key_hstep = (size_t)key_head.w * key_head.elempack;
#endif // __AVX__

        __m128 _m = _mm_set1_ps(-FLT_MAX);
        __m128 _l = _mm_setzero_ps();

        for (int n = 0; n < key_seqlen; n += block_n)
        {
            const int max_jj = std::min(key_seqlen - n, block_n);
            __m128 _block_max = _mm_set1_ps(-FLT_MAX);
            {
                float* pS = scoreT;
                const unsigned short* pM = mask ? mask + n : 0;
                int j = 0;
#if __AVX__
#if __AVX512F__
                for (; j + 3 < max_jj; j += 4)
                {
                    const unsigned short* pK = key_head.row<const unsigned short>(n + j);
                    const unsigned short* pQ = queryT;
                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();
                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m128i _q = _mm_loadu_si128((const __m128i*)pQ);
                        _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_q, (__m128bh)_mm_set1_epi32(((const int*)pK)[0]));
                        _sum1 = _mm_dpbf16_ps(_sum1, (__m128bh)_q, (__m128bh)_mm_set1_epi32(((const int*)(pK + key_hstep))[0]));
                        _sum2 = _mm_dpbf16_ps(_sum2, (__m128bh)_q, (__m128bh)_mm_set1_epi32(((const int*)(pK + key_hstep * 2))[0]));
                        _sum3 = _mm_dpbf16_ps(_sum3, (__m128bh)_q, (__m128bh)_mm_set1_epi32(((const int*)(pK + key_hstep * 3))[0]));
                        pQ += 8;
                        pK += 2;
                    }
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m128 _q = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pQ));
                        _sum0 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(pK[0])), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(pK[key_hstep])), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(pK[key_hstep * 2])), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(pK[key_hstep * 3])), _sum3);
                        pQ += 4;
                        pK++;
                    }
                    __m512 _score = _mm512_mul_ps(combine4x4_ps(_sum0, _sum1, _sum2, _sum3), _mm512_set1_ps(scale));
                    if (pM)
                    {
                        __m512 _mask;
                        if (mask_per_head)
                        {
                            __m128i _m0 = _mm_loadl_epi64((const __m128i*)pM);
                            __m128i _m1 = _mm_loadl_epi64((const __m128i*)(pM + mask_cstep));
                            __m128i _m2 = _mm_loadl_epi64((const __m128i*)(pM + mask_cstep * 2));
                            __m128i _m3 = _mm_loadl_epi64((const __m128i*)(pM + mask_cstep * 3));
                            transpose8x4_epi16(_m0, _m1, _m2, _m3);
                            _mask = combine8x2_ps(bfloat2float_avx(_m0), bfloat2float_avx(_m1));
                        }
                        else
                        {
                            _mask = combine4x4_ps(_mm_set1_ps(bfloat16_to_float32(pM[0])), _mm_set1_ps(bfloat16_to_float32(pM[1])), _mm_set1_ps(bfloat16_to_float32(pM[2])), _mm_set1_ps(bfloat16_to_float32(pM[3])));
                        }
                        _score = _mm512_add_ps(_score, _mask);
                        pM += 4;
                    }
                    _mm512_storeu_ps(pS, _score);
                    pS += 16;
                    __m128 _score0 = _mm512_extractf32x4_ps(_score, 0);
                    __m128 _score1 = _mm512_extractf32x4_ps(_score, 1);
                    __m128 _score2 = _mm512_extractf32x4_ps(_score, 2);
                    __m128 _score3 = _mm512_extractf32x4_ps(_score, 3);
                    _block_max = _mm_max_ps(_block_max, _mm_max_ps(_mm_max_ps(_score0, _score1), _mm_max_ps(_score2, _score3)));
                }
#endif // __AVX512F__
                for (; j + 1 < max_jj; j += 2)
                {
                    const unsigned short* pK = key_head.row<const unsigned short>(n + j);
                    const unsigned short* pQ = queryT;
                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m128i _q = _mm_loadu_si128((const __m128i*)pQ);
                        _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_q, (__m128bh)_mm_set1_epi32(((const int*)pK)[0]));
                        _sum1 = _mm_dpbf16_ps(_sum1, (__m128bh)_q, (__m128bh)_mm_set1_epi32(((const int*)(pK + key_hstep))[0]));
                        pQ += 8;
                        pK += 2;
                    }
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m128 _q = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pQ));
                        _sum0 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(pK[0])), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(pK[key_hstep])), _sum1);
                        pQ += 4;
                        pK++;
                    }
                    __m256 _score = _mm256_mul_ps(combine4x2_ps(_sum0, _sum1), _mm256_set1_ps(scale));
                    if (pM)
                    {
                        __m256 _mask;
                        if (mask_per_head)
                        {
                            __m128i _m0 = _mm_cvtsi32_si128(*(const int*)pM);
                            __m128i _m1 = _mm_cvtsi32_si128(*(const int*)(pM + mask_cstep));
                            __m128i _m2 = _mm_cvtsi32_si128(*(const int*)(pM + mask_cstep * 2));
                            __m128i _m3 = _mm_cvtsi32_si128(*(const int*)(pM + mask_cstep * 3));
                            transpose8x4_epi16(_m0, _m1, _m2, _m3);
                            _mask = bfloat2float_avx(_m0);
                        }
                        else
                        {
                            _mask = combine4x2_ps(_mm_set1_ps(bfloat16_to_float32(pM[0])), _mm_set1_ps(bfloat16_to_float32(pM[1])));
                        }
                        _score = _mm256_add_ps(_score, _mask);
                        pM += 2;
                    }
                    _mm256_storeu_ps(pS, _score);
                    pS += 8;
                    __m128 _score0 = _mm256_castps256_ps128(_score);
                    __m128 _score1 = _mm256_extractf128_ps(_score, 1);
                    _block_max = _mm_max_ps(_block_max, _mm_max_ps(_score0, _score1));
                }
#endif // __AVX__
                for (; j < max_jj; j++)
                {
                    const unsigned short* pK = key_head.row<const unsigned short>(n + j);
                    const unsigned short* pQ = queryT;
                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();
                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m128i _q = _mm_loadu_si128((const __m128i*)pQ);
                        _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_q, (__m128bh)_mm_set1_epi32(((const int*)pK)[0]));
                        pQ += 8;
                        pK += 2;
                    }
#endif // __AVX512BF16__
                    for (; d + 3 < head_dim; d += 4)
                    {
                        _sum0 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pQ)), _mm_set1_ps(bfloat16_to_float32(pK[0])), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pQ + 4))), _mm_set1_ps(bfloat16_to_float32(pK[1])), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pQ + 8))), _mm_set1_ps(bfloat16_to_float32(pK[2])), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pQ + 12))), _mm_set1_ps(bfloat16_to_float32(pK[3])), _sum3);
                        pQ += 16;
                        pK += 4;
                    }
                    for (; d < head_dim; d++)
                    {
                        _sum0 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pQ)), _mm_set1_ps(bfloat16_to_float32(*pK)), _sum0);
                        pQ += 4;
                        pK++;
                    }
                    __m128 _score = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_sum0, _sum1), _mm_add_ps(_sum2, _sum3)), _mm_set1_ps(scale));
                    if (pM)
                    {
                        _score = _mm_add_ps(_score, _mm_set_ps(bfloat16_to_float32(pM[mask_cstep * 3]), bfloat16_to_float32(pM[mask_cstep * 2]), bfloat16_to_float32(pM[mask_cstep]), bfloat16_to_float32(pM[0])));
                        pM++;
                    }
                    _mm_storeu_ps(pS, _score);
                    pS += 4;
                    _block_max = _mm_max_ps(_block_max, _score);
                }
            }

            __m128 _m_new = _mm_max_ps(_m, _block_max);
            __m128 _alpha_active = _mm_cmpneq_ps(_l, _mm_setzero_ps());
            __m128 _alpha = exp_ps(_mm_and_ps(_alpha_active, _mm_sub_ps(_m, _m_new)));
            _alpha = _mm_and_ps(_alpha, _alpha_active);

            float* scoreptr = scoreT;
            __m128 _sum0 = _mm_setzero_ps();
            int j = 0;
#if defined(__x86_64__) || defined(_M_X64)
            __m128 _sum1 = _mm_setzero_ps();
            __m128 _sum2 = _mm_setzero_ps();
            __m128 _sum3 = _mm_setzero_ps();
            for (; j + 3 < max_jj; j += 4)
            {
                __m128 _score = _mm_sub_ps(_mm_loadu_ps(scoreptr), _m_new);
                __m128 _p = exp_ps(_score);
                _mm_storeu_ps(scoreptr, _p);
                _sum0 = _mm_add_ps(_sum0, _p);
                _score = _mm_sub_ps(_mm_loadu_ps(scoreptr + 4), _m_new);
                _p = exp_ps(_score);
                _mm_storeu_ps(scoreptr + 4, _p);
                _sum1 = _mm_add_ps(_sum1, _p);
                _score = _mm_sub_ps(_mm_loadu_ps(scoreptr + 8), _m_new);
                _p = exp_ps(_score);
                _mm_storeu_ps(scoreptr + 8, _p);
                _sum2 = _mm_add_ps(_sum2, _p);
                _score = _mm_sub_ps(_mm_loadu_ps(scoreptr + 12), _m_new);
                _p = exp_ps(_score);
                _mm_storeu_ps(scoreptr + 12, _p);
                _sum3 = _mm_add_ps(_sum3, _p);
                scoreptr += 16;
            }
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; j < max_jj; j++)
            {
                __m128 _score = _mm_sub_ps(_mm_loadu_ps(scoreptr), _m_new);
                __m128 _p = exp_ps(_score);
                _mm_storeu_ps(scoreptr, _p);
                scoreptr += 4;
                _sum0 = _mm_add_ps(_sum0, _p);
            }
            __m128 _sum = _sum0;
#if defined(__x86_64__) || defined(_M_X64)
            _sum = _mm_add_ps(_mm_add_ps(_sum, _sum1), _mm_add_ps(_sum2, _sum3));
#endif // defined(__x86_64__) || defined(_M_X64)

            _l = _mm_add_ps(_mm_mul_ps(_l, _alpha), _sum);
            _m = _m_new;
            float* outptr = outT;
            const unsigned short* value = value_head.row<const unsigned short>(n);
            const unsigned short* valueptr = value;
            int d = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX__
#if __AVX512F__
            for (; d + 15 < value_dim; d += 16)
            {
                __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(outptr), _alpha);
                __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(outptr + 4), _alpha);
                __m128 _out2 = _mm_mul_ps(_mm_loadu_ps(outptr + 8), _alpha);
                __m128 _out3 = _mm_mul_ps(_mm_loadu_ps(outptr + 12), _alpha);
                __m128 _out4 = _mm_mul_ps(_mm_loadu_ps(outptr + 16), _alpha);
                __m128 _out5 = _mm_mul_ps(_mm_loadu_ps(outptr + 20), _alpha);
                __m128 _out6 = _mm_mul_ps(_mm_loadu_ps(outptr + 24), _alpha);
                __m128 _out7 = _mm_mul_ps(_mm_loadu_ps(outptr + 28), _alpha);
                __m128 _out8 = _mm_mul_ps(_mm_loadu_ps(outptr + 32), _alpha);
                __m128 _out9 = _mm_mul_ps(_mm_loadu_ps(outptr + 36), _alpha);
                __m128 _outa = _mm_mul_ps(_mm_loadu_ps(outptr + 40), _alpha);
                __m128 _outb = _mm_mul_ps(_mm_loadu_ps(outptr + 44), _alpha);
                __m128 _outc = _mm_mul_ps(_mm_loadu_ps(outptr + 48), _alpha);
                __m128 _outd = _mm_mul_ps(_mm_loadu_ps(outptr + 52), _alpha);
                __m128 _oute = _mm_mul_ps(_mm_loadu_ps(outptr + 56), _alpha);
                __m128 _outf = _mm_mul_ps(_mm_loadu_ps(outptr + 60), _alpha);
                const unsigned short* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    __m128 _p = _mm_loadu_ps(pS);
                    __m512 _v = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pV));
                    __m128 _v0 = _mm512_extractf32x4_ps(_v, 0);
                    _out0 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v0, _v0, _MM_SHUFFLE(0, 0, 0, 0)), _out0);
                    _out1 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v0, _v0, _MM_SHUFFLE(1, 1, 1, 1)), _out1);
                    _out2 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v0, _v0, _MM_SHUFFLE(2, 2, 2, 2)), _out2);
                    _out3 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v0, _v0, _MM_SHUFFLE(3, 3, 3, 3)), _out3);
                    __m128 _v1 = _mm512_extractf32x4_ps(_v, 1);
                    _out4 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v1, _v1, _MM_SHUFFLE(0, 0, 0, 0)), _out4);
                    _out5 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v1, _v1, _MM_SHUFFLE(1, 1, 1, 1)), _out5);
                    _out6 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v1, _v1, _MM_SHUFFLE(2, 2, 2, 2)), _out6);
                    _out7 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v1, _v1, _MM_SHUFFLE(3, 3, 3, 3)), _out7);
                    __m128 _v2 = _mm512_extractf32x4_ps(_v, 2);
                    _out8 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v2, _v2, _MM_SHUFFLE(0, 0, 0, 0)), _out8);
                    _out9 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v2, _v2, _MM_SHUFFLE(1, 1, 1, 1)), _out9);
                    _outa = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v2, _v2, _MM_SHUFFLE(2, 2, 2, 2)), _outa);
                    _outb = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v2, _v2, _MM_SHUFFLE(3, 3, 3, 3)), _outb);
                    __m128 _v3 = _mm512_extractf32x4_ps(_v, 3);
                    _outc = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v3, _v3, _MM_SHUFFLE(0, 0, 0, 0)), _outc);
                    _outd = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v3, _v3, _MM_SHUFFLE(1, 1, 1, 1)), _outd);
                    _oute = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v3, _v3, _MM_SHUFFLE(2, 2, 2, 2)), _oute);
                    _outf = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v3, _v3, _MM_SHUFFLE(3, 3, 3, 3)), _outf);
                    pS += 4;
                    pV += value_dim;
                }
                _mm_storeu_ps(outptr, _out0);
                _mm_storeu_ps(outptr + 4, _out1);
                _mm_storeu_ps(outptr + 8, _out2);
                _mm_storeu_ps(outptr + 12, _out3);
                _mm_storeu_ps(outptr + 16, _out4);
                _mm_storeu_ps(outptr + 20, _out5);
                _mm_storeu_ps(outptr + 24, _out6);
                _mm_storeu_ps(outptr + 28, _out7);
                _mm_storeu_ps(outptr + 32, _out8);
                _mm_storeu_ps(outptr + 36, _out9);
                _mm_storeu_ps(outptr + 40, _outa);
                _mm_storeu_ps(outptr + 44, _outb);
                _mm_storeu_ps(outptr + 48, _outc);
                _mm_storeu_ps(outptr + 52, _outd);
                _mm_storeu_ps(outptr + 56, _oute);
                _mm_storeu_ps(outptr + 60, _outf);
                outptr += 64;
                valueptr += 16;
            }
#endif // __AVX512F__
            for (; d + 7 < value_dim; d += 8)
            {
                __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(outptr), _alpha);
                __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(outptr + 4), _alpha);
                __m128 _out2 = _mm_mul_ps(_mm_loadu_ps(outptr + 8), _alpha);
                __m128 _out3 = _mm_mul_ps(_mm_loadu_ps(outptr + 12), _alpha);
                __m128 _out4 = _mm_mul_ps(_mm_loadu_ps(outptr + 16), _alpha);
                __m128 _out5 = _mm_mul_ps(_mm_loadu_ps(outptr + 20), _alpha);
                __m128 _out6 = _mm_mul_ps(_mm_loadu_ps(outptr + 24), _alpha);
                __m128 _out7 = _mm_mul_ps(_mm_loadu_ps(outptr + 28), _alpha);
                const unsigned short* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    __m128 _p = _mm_loadu_ps(pS);
                    __m256 _v = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pV));
                    __m128 _v0 = _mm256_castps256_ps128(_v);
                    _out0 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v0, _v0, _MM_SHUFFLE(0, 0, 0, 0)), _out0);
                    _out1 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v0, _v0, _MM_SHUFFLE(1, 1, 1, 1)), _out1);
                    _out2 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v0, _v0, _MM_SHUFFLE(2, 2, 2, 2)), _out2);
                    _out3 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v0, _v0, _MM_SHUFFLE(3, 3, 3, 3)), _out3);
                    __m128 _v1 = _mm256_extractf128_ps(_v, 1);
                    _out4 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v1, _v1, _MM_SHUFFLE(0, 0, 0, 0)), _out4);
                    _out5 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v1, _v1, _MM_SHUFFLE(1, 1, 1, 1)), _out5);
                    _out6 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v1, _v1, _MM_SHUFFLE(2, 2, 2, 2)), _out6);
                    _out7 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v1, _v1, _MM_SHUFFLE(3, 3, 3, 3)), _out7);
                    pS += 4;
                    pV += value_dim;
                }
                _mm_storeu_ps(outptr, _out0);
                _mm_storeu_ps(outptr + 4, _out1);
                _mm_storeu_ps(outptr + 8, _out2);
                _mm_storeu_ps(outptr + 12, _out3);
                _mm_storeu_ps(outptr + 16, _out4);
                _mm_storeu_ps(outptr + 20, _out5);
                _mm_storeu_ps(outptr + 24, _out6);
                _mm_storeu_ps(outptr + 28, _out7);
                outptr += 32;
                valueptr += 8;
            }
#endif // __AVX__
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; d + 3 < value_dim; d += 4)
            {
                __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(outptr), _alpha);
                __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(outptr + 4), _alpha);
                __m128 _out2 = _mm_mul_ps(_mm_loadu_ps(outptr + 8), _alpha);
                __m128 _out3 = _mm_mul_ps(_mm_loadu_ps(outptr + 12), _alpha);
                const unsigned short* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    __m128 _p = _mm_loadu_ps(pS);
                    __m128 _v = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pV));
                    _out0 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v, _v, _MM_SHUFFLE(0, 0, 0, 0)), _out0);
                    _out1 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v, _v, _MM_SHUFFLE(1, 1, 1, 1)), _out1);
                    _out2 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v, _v, _MM_SHUFFLE(2, 2, 2, 2)), _out2);
                    _out3 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v, _v, _MM_SHUFFLE(3, 3, 3, 3)), _out3);
                    pS += 4;
                    pV += value_dim;
                }
                _mm_storeu_ps(outptr, _out0);
                _mm_storeu_ps(outptr + 4, _out1);
                _mm_storeu_ps(outptr + 8, _out2);
                _mm_storeu_ps(outptr + 12, _out3);
                outptr += 16;
                valueptr += 4;
            }
            for (; d < value_dim; d++)
            {
                __m128 _out = _mm_mul_ps(_mm_loadu_ps(outptr), _alpha);
                const unsigned short* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    _out = _mm_comp_fmadd_ps(_mm_loadu_ps(pS), _mm_set1_ps(bfloat16_to_float32(*pV)), _out);
                    pS += 4;
                    pV += value_dim;
                }
                _mm_storeu_ps(outptr, _out);
                outptr += 4;
                valueptr++;
            }
        }

        {
            float* output = top_blob.channel(q);
            const size_t output_cstep = top_blob.cstep;
            __m128 _nonzero = _mm_cmpneq_ps(_l, _mm_setzero_ps());
            __m128 _denom = _mm_or_ps(_mm_and_ps(_nonzero, _l), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
            __m128 _out_scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(1.f), _denom), _nonzero);

            const float* outptr = outT;
            float* p0 = output;
            int d = 0;
            for (; d + 3 < value_dim; d += 4)
            {
                __m128 _r0 = _mm_mul_ps(_mm_loadu_ps(outptr), _out_scale);
                __m128 _r1 = _mm_mul_ps(_mm_loadu_ps(outptr + 4), _out_scale);
                __m128 _r2 = _mm_mul_ps(_mm_loadu_ps(outptr + 8), _out_scale);
                __m128 _r3 = _mm_mul_ps(_mm_loadu_ps(outptr + 12), _out_scale);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_storeu_ps(p0, _r0);
                _mm_storeu_ps(p0 + output_cstep, _r1);
                _mm_storeu_ps(p0 + output_cstep * 2, _r2);
                _mm_storeu_ps(p0 + output_cstep * 3, _r3);
                p0 += 4;
                outptr += 16;
            }
            for (; d < value_dim; d++)
            {
                __m128 _r = _mm_mul_ps(_mm_loadu_ps(outptr), _out_scale);
                *p0 = _mm_cvtss_f32(_r);
                p0[output_cstep] = _mm_cvtss_f32(_mm_shuffle_ps(_r, _r, _MM_SHUFFLE(1, 1, 1, 1)));
                p0[output_cstep * 2] = _mm_cvtss_f32(_mm_movehl_ps(_r, _r));
                p0[output_cstep * 3] = _mm_cvtss_f32(_mm_shuffle_ps(_r, _r, _MM_SHUFFLE(3, 3, 3, 3)));
                p0++;
                outptr += 4;
            }
        }
    }
#endif // __SSE2__

    for (; qq < max_qq; qq++)
    {
        const int q = q0 + qq;
        const int head_dim = query.w;
        const int value_dim = value.w;

        const unsigned short* query_ptr = query.channel(q);
        Mat mask_head;
        if (!attn_mask_blob.empty())
            mask_head = attn_mask_blob.dims == 3 ? attn_mask_blob.channel(attn_mask_blob.c > 1 ? q : 0) : attn_mask_blob;
        const unsigned short* mask = mask_head.empty() ? 0 : mask_head;

        float* workspace_ptr = workspace;
        float* score = workspace_ptr;
        float* out = workspace_ptr + block_n;
        memset(out, 0, value_dim * sizeof(float));

        const Mat key_head = key.channel(g);
        const Mat value_head = value.channel(g);

        float m = -FLT_MAX;
        float l = 0.f;

        for (int n = 0; n < key_seqlen; n += block_n)
        {
            const int max_jj = std::min(key_seqlen - n, block_n);
            float block_max = -FLT_MAX;
            const unsigned short* pK = key_head.row<const unsigned short>(n);
            float* pS = score;
            const unsigned short* pM = mask ? mask + n : 0;
            for (int j = 0; j < max_jj; j++)
            {
                const unsigned short* pQ = query_ptr;
                float sum;
#if __SSE2__
#if __AVX__
#if __AVX512F__
#if __AVX512BF16__
                __m512 _sum_bf16 = _mm512_setzero_ps();
#endif // __AVX512BF16__
                __m512 _sum_avx512 = _mm512_setzero_ps();
#endif // __AVX512F__
                __m256 _sum_avx = _mm256_setzero_ps();
#endif // __AVX__
                __m128 _sum = _mm_setzero_ps();
#endif // __SSE2__
                sum = 0.f;

                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
#if __AVX512BF16__
                for (; i + 31 < head_dim; i += 32)
                {
                    __m512i _q = _mm512_loadu_si512((const __m512i*)pQ);
                    __m512i _k = _mm512_loadu_si512((const __m512i*)pK);
                    _sum_bf16 = _mm512_dpbf16_ps(_sum_bf16, (__m512bh)_q, (__m512bh)_k);
                    pQ += 32;
                    pK += 32;
                }
#endif // __AVX512BF16__
                for (; i + 15 < head_dim; i += 16)
                {
                    __m512 _q = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pQ));
                    __m512 _k = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pK));
                    _sum_avx512 = _mm512_fmadd_ps(_q, _k, _sum_avx512);
                    pQ += 16;
                    pK += 16;
                }
#endif // __AVX512F__
                for (; i + 7 < head_dim; i += 8)
                {
                    __m256 _q = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pQ));
                    __m256 _k = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pK));
                    _sum_avx = _mm256_comp_fmadd_ps(_q, _k, _sum_avx);
                    pQ += 8;
                    pK += 8;
                }
#endif // __AVX__
                for (; i + 3 < head_dim; i += 4)
                {
                    __m128 _q = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pQ));
                    __m128 _k = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pK));
                    _sum = _mm_comp_fmadd_ps(_q, _k, _sum);
                    pQ += 4;
                    pK += 4;
                }
#endif // __SSE2__
                for (; i < head_dim; i++)
                {
                    sum += bfloat16_to_float32(*pQ) * bfloat16_to_float32(*pK);
                    pQ++;
                    pK++;
                }

#if __SSE2__
#if __AVX__
#if __AVX512F__
#if __AVX512BF16__
                sum += _mm512_comp_reduce_add_ps(_sum_bf16);
#endif // __AVX512BF16__
                sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
                sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
                sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__
                float s = sum * scale;
                if (pM)
                {
                    s += bfloat16_to_float32(*pM);
                    pM++;
                }
                *pS++ = s;
                block_max = std::max(block_max, s);
            }

            const float m_new = std::max(m, block_max);
            const float alpha = l == 0.f ? 0.f : expf(m - m_new);
            {
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sum_avx512 = _mm512_setzero_ps();
                __m512 _max_avx512 = _mm512_set1_ps(m_new);
#endif // __AVX512F__
                __m256 _sum_avx = _mm256_setzero_ps();
                __m256 _max_avx = _mm256_set1_ps(m_new);
#endif // __AVX__
                __m128 _sum = _mm_setzero_ps();
                __m128 _max = _mm_set1_ps(m_new);
#endif // __SSE2__
                float sum = 0.f;

                float* pS = score;
                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; i + 15 < max_jj; i += 16)
                {
                    __m512 _p = _mm512_loadu_ps(pS);
                    _p = exp512_ps(_mm512_sub_ps(_p, _max_avx512));
                    _mm512_storeu_ps(pS, _p);
                    pS += 16;
                    _sum_avx512 = _mm512_add_ps(_sum_avx512, _p);
                }
#endif // __AVX512F__
                for (; i + 7 < max_jj; i += 8)
                {
                    __m256 _p = _mm256_loadu_ps(pS);
                    _p = exp256_ps(_mm256_sub_ps(_p, _max_avx));
                    _mm256_storeu_ps(pS, _p);
                    pS += 8;
                    _sum_avx = _mm256_add_ps(_sum_avx, _p);
                }
#endif // __AVX__
                for (; i + 3 < max_jj; i += 4)
                {
                    __m128 _p = _mm_loadu_ps(pS);
                    _p = exp_ps(_mm_sub_ps(_p, _max));
                    _mm_storeu_ps(pS, _p);
                    pS += 4;
                    _sum = _mm_add_ps(_sum, _p);
                }
#endif // __SSE2__
                for (; i < max_jj; i++)
                {
                    *pS = expf(*pS - m_new);
                    sum += *pS;
                    pS++;
                }

#if __SSE2__
#if __AVX__
#if __AVX512F__
                sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
                sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
                sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__

                l = l * alpha + sum;
            }
            m = m_new;

            const unsigned short* value_base = value_head.row<const unsigned short>(n);
            float* outptr = out;
            const unsigned short* valueptr = value_base;
            int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; d + 63 < value_dim; d += 64)
            {
                __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr), _mm512_set1_ps(alpha));
                __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 16), _mm512_set1_ps(alpha));
                __m512 _out2 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 32), _mm512_set1_ps(alpha));
                __m512 _out3 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 48), _mm512_set1_ps(alpha));
                const unsigned short* pV = valueptr;
                const float* pS = score;
                for (int j = 0; j < max_jj; j++)
                {
                    __m512 _p = _mm512_set1_ps(*pS++);
                    _out0 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pV)), _p, _out0);
                    _out1 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pV + 16))), _p, _out1);
                    _out2 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pV + 32))), _p, _out2);
                    _out3 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pV + 48))), _p, _out3);
                    pV += value_dim;
                }
                _mm512_storeu_ps(outptr, _out0);
                _mm512_storeu_ps(outptr + 16, _out1);
                _mm512_storeu_ps(outptr + 32, _out2);
                _mm512_storeu_ps(outptr + 48, _out3);
                outptr += 64;
                valueptr += 64;
            }

            for (; d + 15 < value_dim; d += 16)
            {
                __m512 _out = _mm512_mul_ps(_mm512_loadu_ps(outptr), _mm512_set1_ps(alpha));
                const unsigned short* pV = valueptr;
                const float* pS = score;
                for (int j = 0; j < max_jj; j++)
                {
                    _out = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pV)), _mm512_set1_ps(*pS++), _out);
                    pV += value_dim;
                }
                _mm512_storeu_ps(outptr, _out);
                outptr += 16;
                valueptr += 16;
            }
#else
            for (; d + 15 < value_dim; d += 16)
            {
                __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr), _mm256_set1_ps(alpha));
                __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 8), _mm256_set1_ps(alpha));
                const unsigned short* pV = valueptr;
                const float* pS = score;
                for (int j = 0; j < max_jj; j++)
                {
                    __m256 _p = _mm256_set1_ps(*pS++);
                    _out0 = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)pV)), _p, _out0);
                    _out1 = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pV + 8))), _p, _out1);
                    pV += value_dim;
                }
                _mm256_storeu_ps(outptr, _out0);
                _mm256_storeu_ps(outptr + 8, _out1);
                outptr += 16;
                valueptr += 16;
            }
#endif // __AVX512F__

            for (; d + 7 < value_dim; d += 8)
            {
                __m256 _out = _mm256_mul_ps(_mm256_loadu_ps(outptr), _mm256_set1_ps(alpha));
                const unsigned short* pV = valueptr;
                const float* pS = score;
                for (int j = 0; j < max_jj; j++)
                {
                    _out = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)pV)), _mm256_set1_ps(*pS++), _out);
                    pV += value_dim;
                }
                _mm256_storeu_ps(outptr, _out);
                outptr += 8;
                valueptr += 8;
            }
#else
            for (; d + 7 < value_dim; d += 8)
            {
                __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(outptr), _mm_set1_ps(alpha));
                __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(outptr + 4), _mm_set1_ps(alpha));
                const unsigned short* pV = valueptr;
                const float* pS = score;
                for (int j = 0; j < max_jj; j++)
                {
                    __m128 _p = _mm_set1_ps(*pS++);
                    _out0 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pV)), _p, _out0);
                    _out1 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pV + 4))), _p, _out1);
                    pV += value_dim;
                }
                _mm_storeu_ps(outptr, _out0);
                _mm_storeu_ps(outptr + 4, _out1);
                outptr += 8;
                valueptr += 8;
            }
#endif // __AVX__
            for (; d + 3 < value_dim; d += 4)
            {
                __m128 _out = _mm_mul_ps(_mm_loadu_ps(outptr), _mm_set1_ps(alpha));
                const unsigned short* pV = valueptr;
                const float* pS = score;
                for (int j = 0; j < max_jj; j++)
                {
                    _out = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pV)), _mm_set1_ps(*pS++), _out);
                    pV += value_dim;
                }
                _mm_storeu_ps(outptr, _out);
                outptr += 4;
                valueptr += 4;
            }
#endif // __SSE2__
            for (; d < value_dim; d++)
            {
                float sum = *outptr * alpha;
                const unsigned short* pV = valueptr;
                const float* pS = score;
                for (int j = 0; j < max_jj; j++)
                {
                    sum += *pS++ * bfloat16_to_float32(*pV);
                    pV += value_dim;
                }
                *outptr++ = sum;
                valueptr++;
            }
        }

        {
            float* output = top_blob.channel(q);
            memcpy(output, out, value_dim * sizeof(float));
            if (l != 0.f)
            {
                float inv_sum = 1.f / l;
                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _inv_sum_avx512 = _mm512_set1_ps(inv_sum);
                for (; i + 15 < value_dim; i += 16)
                    _mm512_storeu_ps(output + i, _mm512_mul_ps(_mm512_loadu_ps(output + i), _inv_sum_avx512));
#endif // __AVX512F__
                __m256 _inv_sum_avx = _mm256_set1_ps(inv_sum);
                for (; i + 7 < value_dim; i += 8)
                    _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_loadu_ps(output + i), _inv_sum_avx));
#endif // __AVX__
                __m128 _inv_sum = _mm_set1_ps(inv_sum);
                for (; i + 3 < value_dim; i += 4)
                    _mm_storeu_ps(output + i, _mm_mul_ps(_mm_loadu_ps(output + i), _inv_sum));
#endif // __SSE2__
                for (; i < value_dim; i++)
                    output[i] *= inv_sum;
            }
        }
    }
}

static int sdpa_decode_bf16s(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
        return sdpa_decode_bf16s_avx512bf16(query, key, value, attn_mask_blob, top_blob, scale, opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx2())
        return sdpa_decode_bf16s_avx2(query, key, value, attn_mask_blob, top_blob, scale, opt);
#endif

    const int num_query_heads = query.c;
    const int num_kv_heads = key.c;
    const int key_seqlen = key.h;
    const int value_dim = value.w;
    const int num_query_heads_per_kv_head = num_query_heads / num_kv_heads;
    const int block_q = sdpa_decode_get_optimal_tile_q(num_query_heads_per_kv_head);
    const int num_qblocks = (num_query_heads_per_kv_head + block_q - 1) / block_q;
    const int num_tasks = num_kv_heads * num_qblocks;
    const int nT = std::min(std::max(opt.num_threads, 1), num_tasks);
    const int block_n = sdpa_decode_get_optimal_tile_n(query.w, value_dim, key_seqlen, 2, 2, 2, attn_mask_blob.empty() ? 0 : 2, block_q);

    const bool pack_query = block_q >= 4;
    const size_t score_workspace_size = (size_t)block_q * block_n * sizeof(float);
    const size_t output_workspace_size = (size_t)block_q * value_dim * sizeof(float);
    const size_t query_workspace_size = pack_query ? (size_t)block_q * query.w * sizeof(unsigned short) : 0;
    const size_t workspace_size = alignSize(score_workspace_size + output_workspace_size + query_workspace_size, 64);
    Mat workspace((int)(workspace_size / sizeof(float)), 1, nT, 4u, opt.workspace_allocator);
    if (workspace.empty())
        return -100;

    #pragma omp parallel for num_threads(nT)
    for (int task_id = 0; task_id < num_tasks; task_id++)
    {
        const int g = task_id / num_qblocks;
        const int qblock_id = task_id % num_qblocks;
        const int q0 = g * num_query_heads_per_kv_head + qblock_id * block_q;
        const int max_qq = std::min(num_query_heads_per_kv_head - qblock_id * block_q, block_q);

        Mat workspace_tile = workspace.channel(get_omp_thread_num());
        sdpa_decode_tile_bf16s(query, key, value, attn_mask_blob, top_blob, scale, q0, max_qq, g, block_n, workspace_tile);
    }

    return 0;
}

static void sdpa_decode_kvcache_tile_bf16s(const Mat& query, const Mat& key_cache, const Mat& value_cache, const Mat& attn_mask_blob, Mat& top_blob, float scale, int q0, int max_qq, int g, int block_n, Mat& workspace)
{
    const int head_dim = query.w;
    const int value_dim = value_cache.w;
    const int key_seqlen = key_cache.h;
#if __AVX512F__
    const int NR = 16;
#elif __AVX__
    const int NR = 8;
#elif __SSE2__
    const int NR = 4;
#else
    const int NR = 2;
#endif
    const int score_workspace_size = max_qq * block_n;
    const int out_workspace_size = max_qq * value_dim;
    Mat scoreT = workspace.range(0, score_workspace_size);
    Mat outT = workspace.range(score_workspace_size, out_workspace_size);
#if __SSE2__
    const bool pack_query = max_qq >= 4;
    Mat queryT;
    if (pack_query)
    {
        queryT = Mat(head_dim * max_qq, (unsigned short*)((float*)outT + out_workspace_size), 2u);
        sdpa_decode_pack_query_bf16s(query, queryT, q0, max_qq);
    }
    const unsigned short* queryT_ptr = queryT;
#endif // __SSE2__
    float* scoreT_ptr = scoreT;
    float* outT_ptr = outT;
    const Mat key_cache_head = key_cache.channel(g);
    const Mat value_cache_head = value_cache.channel(g);

    int qq = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; qq + 15 < max_qq; qq += 16)
    {
        const bool mask_per_head = !attn_mask_blob.empty() && attn_mask_blob.dims == 3 && attn_mask_blob.c > 1;
        const unsigned short* mask = 0;
        if (!attn_mask_blob.empty())
        {
            if (attn_mask_blob.dims == 3)
                mask = attn_mask_blob.channel(mask_per_head ? q0 + qq : 0);
            else
                mask = attn_mask_blob;
        }
        const int mask_hstep = mask_per_head ? (int)attn_mask_blob.cstep : 0;
        __m512i _mask_index = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(mask_hstep));

        const unsigned short* queryT_tile = queryT_ptr + (size_t)qq * head_dim;
        float* scoreT_tile = scoreT_ptr + (size_t)qq * block_n;
        float* outT_tile = outT_ptr + (size_t)qq * value_dim;
        __m512 _m = _mm512_set1_ps(-FLT_MAX);
        __m512 _l = _mm512_setzero_ps();
        memset(outT_tile, 0, (size_t)value_dim * 16 * sizeof(float));

        for (int n = 0; n < key_seqlen; n += block_n)
        {
            const int max_jj = std::min(key_seqlen - n, block_n);
            float* scoreptr = scoreT_tile;
            const unsigned short* pM = mask ? mask + n : 0;
            __m512 _block_max = _mm512_set1_ps(-FLT_MAX);
            float* score_panel = scoreptr;

            for (int jj = 0; jj < max_jj; jj += NR)
            {
                const int max_nn = std::min(NR, max_jj - jj);
                const unsigned short* key_panel = (const unsigned short*)key_cache_head + (size_t)(n + jj) * head_dim;
                int j = 0;
#if defined(__x86_64__) || defined(_M_X64)
                for (; j + 15 < max_nn; j += 16)
                {
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
                    const unsigned short* pA = queryT_tile;
                    const unsigned short* pK = key_panel;
                    int d = 0;
#if __AVX512BF16__
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                        __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pK);

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
                        pK += 32;
                    }
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m512 _pA0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pA));
                        __m512 _pB0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pK));

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
                        pK += 16;
                    }

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

                    __m512 _scale = _mm512_set1_ps(scale);
                    _sum0 = _mm512_mul_ps(_sum0, _scale);
                    _sum1 = _mm512_mul_ps(_sum1, _scale);
                    _sum2 = _mm512_mul_ps(_sum2, _scale);
                    _sum3 = _mm512_mul_ps(_sum3, _scale);
                    _sum4 = _mm512_mul_ps(_sum4, _scale);
                    _sum5 = _mm512_mul_ps(_sum5, _scale);
                    _sum6 = _mm512_mul_ps(_sum6, _scale);
                    _sum7 = _mm512_mul_ps(_sum7, _scale);
                    _sum8 = _mm512_mul_ps(_sum8, _scale);
                    _sum9 = _mm512_mul_ps(_sum9, _scale);
                    _suma = _mm512_mul_ps(_suma, _scale);
                    _sumb = _mm512_mul_ps(_sumb, _scale);
                    _sumc = _mm512_mul_ps(_sumc, _scale);
                    _sumd = _mm512_mul_ps(_sumd, _scale);
                    _sume = _mm512_mul_ps(_sume, _scale);
                    _sumf = _mm512_mul_ps(_sumf, _scale);
                    if (pM)
                    {
                        if (mask_per_head)
                        {
                            const unsigned short* pM0 = pM;
                            __m256i _m0 = _mm256_loadu_si256((const __m256i*)pM0);
                            __m256i _m1 = _mm256_loadu_si256((const __m256i*)(pM0 + mask_hstep));
                            __m256i _m2 = _mm256_loadu_si256((const __m256i*)(pM0 + mask_hstep * 2));
                            __m256i _m3 = _mm256_loadu_si256((const __m256i*)(pM0 + mask_hstep * 3));
                            __m256i _m4 = _mm256_loadu_si256((const __m256i*)(pM0 + mask_hstep * 4));
                            __m256i _m5 = _mm256_loadu_si256((const __m256i*)(pM0 + mask_hstep * 5));
                            __m256i _m6 = _mm256_loadu_si256((const __m256i*)(pM0 + mask_hstep * 6));
                            __m256i _m7 = _mm256_loadu_si256((const __m256i*)(pM0 + mask_hstep * 7));
                            __m256i _m8 = _mm256_loadu_si256((const __m256i*)(pM0 + mask_hstep * 8));
                            __m256i _m9 = _mm256_loadu_si256((const __m256i*)(pM0 + mask_hstep * 9));
                            __m256i _ma = _mm256_loadu_si256((const __m256i*)(pM0 + mask_hstep * 10));
                            __m256i _mb = _mm256_loadu_si256((const __m256i*)(pM0 + mask_hstep * 11));
                            __m256i _mc = _mm256_loadu_si256((const __m256i*)(pM0 + mask_hstep * 12));
                            __m256i _md = _mm256_loadu_si256((const __m256i*)(pM0 + mask_hstep * 13));
                            __m256i _me = _mm256_loadu_si256((const __m256i*)(pM0 + mask_hstep * 14));
                            __m256i _mf = _mm256_loadu_si256((const __m256i*)(pM0 + mask_hstep * 15));
                            transpose16x16_epi16(_m0, _m1, _m2, _m3, _m4, _m5, _m6, _m7, _m8, _m9, _ma, _mb, _mc, _md, _me, _mf);
                            _sum0 = _mm512_add_ps(_sum0, bfloat2float_avx512(_m0));
                            _sum1 = _mm512_add_ps(_sum1, bfloat2float_avx512(_m1));
                            _sum2 = _mm512_add_ps(_sum2, bfloat2float_avx512(_m2));
                            _sum3 = _mm512_add_ps(_sum3, bfloat2float_avx512(_m3));
                            _sum4 = _mm512_add_ps(_sum4, bfloat2float_avx512(_m4));
                            _sum5 = _mm512_add_ps(_sum5, bfloat2float_avx512(_m5));
                            _sum6 = _mm512_add_ps(_sum6, bfloat2float_avx512(_m6));
                            _sum7 = _mm512_add_ps(_sum7, bfloat2float_avx512(_m7));
                            _sum8 = _mm512_add_ps(_sum8, bfloat2float_avx512(_m8));
                            _sum9 = _mm512_add_ps(_sum9, bfloat2float_avx512(_m9));
                            _suma = _mm512_add_ps(_suma, bfloat2float_avx512(_ma));
                            _sumb = _mm512_add_ps(_sumb, bfloat2float_avx512(_mb));
                            _sumc = _mm512_add_ps(_sumc, bfloat2float_avx512(_mc));
                            _sumd = _mm512_add_ps(_sumd, bfloat2float_avx512(_md));
                            _sume = _mm512_add_ps(_sume, bfloat2float_avx512(_me));
                            _sumf = _mm512_add_ps(_sumf, bfloat2float_avx512(_mf));
                        }
                        else
                        {
                            _sum0 = _mm512_add_ps(_sum0, _mm512_set1_ps(bfloat16_to_float32(pM[0])));
                            _sum1 = _mm512_add_ps(_sum1, _mm512_set1_ps(bfloat16_to_float32(pM[1])));
                            _sum2 = _mm512_add_ps(_sum2, _mm512_set1_ps(bfloat16_to_float32(pM[2])));
                            _sum3 = _mm512_add_ps(_sum3, _mm512_set1_ps(bfloat16_to_float32(pM[3])));
                            _sum4 = _mm512_add_ps(_sum4, _mm512_set1_ps(bfloat16_to_float32(pM[4])));
                            _sum5 = _mm512_add_ps(_sum5, _mm512_set1_ps(bfloat16_to_float32(pM[5])));
                            _sum6 = _mm512_add_ps(_sum6, _mm512_set1_ps(bfloat16_to_float32(pM[6])));
                            _sum7 = _mm512_add_ps(_sum7, _mm512_set1_ps(bfloat16_to_float32(pM[7])));
                            _sum8 = _mm512_add_ps(_sum8, _mm512_set1_ps(bfloat16_to_float32(pM[8])));
                            _sum9 = _mm512_add_ps(_sum9, _mm512_set1_ps(bfloat16_to_float32(pM[9])));
                            _suma = _mm512_add_ps(_suma, _mm512_set1_ps(bfloat16_to_float32(pM[10])));
                            _sumb = _mm512_add_ps(_sumb, _mm512_set1_ps(bfloat16_to_float32(pM[11])));
                            _sumc = _mm512_add_ps(_sumc, _mm512_set1_ps(bfloat16_to_float32(pM[12])));
                            _sumd = _mm512_add_ps(_sumd, _mm512_set1_ps(bfloat16_to_float32(pM[13])));
                            _sume = _mm512_add_ps(_sume, _mm512_set1_ps(bfloat16_to_float32(pM[14])));
                            _sumf = _mm512_add_ps(_sumf, _mm512_set1_ps(bfloat16_to_float32(pM[15])));
                        }
                        pM += 16;
                    }
                    _mm512_storeu_ps(score_panel, _sum0);
                    _mm512_storeu_ps(score_panel + 16, _sum1);
                    _mm512_storeu_ps(score_panel + 32, _sum2);
                    _mm512_storeu_ps(score_panel + 48, _sum3);
                    _mm512_storeu_ps(score_panel + 64, _sum4);
                    _mm512_storeu_ps(score_panel + 80, _sum5);
                    _mm512_storeu_ps(score_panel + 96, _sum6);
                    _mm512_storeu_ps(score_panel + 112, _sum7);
                    _mm512_storeu_ps(score_panel + 128, _sum8);
                    _mm512_storeu_ps(score_panel + 144, _sum9);
                    _mm512_storeu_ps(score_panel + 160, _suma);
                    _mm512_storeu_ps(score_panel + 176, _sumb);
                    _mm512_storeu_ps(score_panel + 192, _sumc);
                    _mm512_storeu_ps(score_panel + 208, _sumd);
                    _mm512_storeu_ps(score_panel + 224, _sume);
                    _mm512_storeu_ps(score_panel + 240, _sumf);
                    score_panel += 256;
                    _block_max = _mm512_max_ps(_block_max, _mm512_max_ps(_mm512_max_ps(_mm512_max_ps(_sum0, _sum1), _mm512_max_ps(_sum2, _sum3)), _mm512_max_ps(_mm512_max_ps(_sum4, _sum5), _mm512_max_ps(_sum6, _sum7))));
                    _block_max = _mm512_max_ps(_block_max, _mm512_max_ps(_mm512_max_ps(_mm512_max_ps(_sum8, _sum9), _mm512_max_ps(_suma, _sumb)), _mm512_max_ps(_mm512_max_ps(_sumc, _sumd), _mm512_max_ps(_sume, _sumf))));
                }
                for (; j + 7 < max_nn; j += 8)
                {
                    __m512 _sum0 = _mm512_setzero_ps();
                    __m512 _sum1 = _mm512_setzero_ps();
                    __m512 _sum2 = _mm512_setzero_ps();
                    __m512 _sum3 = _mm512_setzero_ps();
                    __m512 _sum4 = _mm512_setzero_ps();
                    __m512 _sum5 = _mm512_setzero_ps();
                    __m512 _sum6 = _mm512_setzero_ps();
                    __m512 _sum7 = _mm512_setzero_ps();

                    const unsigned short* pA = queryT_tile;
                    const unsigned short* pK = key_panel + j;
                    int d = 0;
#if __AVX512BF16__
                    pK = key_panel + j * 2;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                        __m256i _pBB = _mm256_loadu_si256((const __m256i*)pK);
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
                        pK += NR * 2;
                    }
                    pK = key_panel + (size_t)d * NR + j;
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m512 _pA0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pA));
                        __m256 _pBB = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pK));
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
                        pK += NR;
                    }

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

                    __m512 _scale = _mm512_set1_ps(scale);
                    _sum0 = _mm512_mul_ps(_sum0, _scale);
                    _sum1 = _mm512_mul_ps(_sum1, _scale);
                    _sum2 = _mm512_mul_ps(_sum2, _scale);
                    _sum3 = _mm512_mul_ps(_sum3, _scale);
                    _sum4 = _mm512_mul_ps(_sum4, _scale);
                    _sum5 = _mm512_mul_ps(_sum5, _scale);
                    _sum6 = _mm512_mul_ps(_sum6, _scale);
                    _sum7 = _mm512_mul_ps(_sum7, _scale);
                    if (pM)
                    {
                        if (mask_per_head)
                        {
                            const unsigned short* pM0 = pM;
                            __m128i _m0 = _mm_loadu_si128((const __m128i*)pM0);
                            __m128i _m1 = _mm_loadu_si128((const __m128i*)(pM0 + mask_hstep));
                            __m128i _m2 = _mm_loadu_si128((const __m128i*)(pM0 + mask_hstep * 2));
                            __m128i _m3 = _mm_loadu_si128((const __m128i*)(pM0 + mask_hstep * 3));
                            __m128i _m4 = _mm_loadu_si128((const __m128i*)(pM0 + mask_hstep * 4));
                            __m128i _m5 = _mm_loadu_si128((const __m128i*)(pM0 + mask_hstep * 5));
                            __m128i _m6 = _mm_loadu_si128((const __m128i*)(pM0 + mask_hstep * 6));
                            __m128i _m7 = _mm_loadu_si128((const __m128i*)(pM0 + mask_hstep * 7));
                            __m128i _m8 = _mm_loadu_si128((const __m128i*)(pM0 + mask_hstep * 8));
                            __m128i _m9 = _mm_loadu_si128((const __m128i*)(pM0 + mask_hstep * 9));
                            __m128i _ma = _mm_loadu_si128((const __m128i*)(pM0 + mask_hstep * 10));
                            __m128i _mb = _mm_loadu_si128((const __m128i*)(pM0 + mask_hstep * 11));
                            __m128i _mc = _mm_loadu_si128((const __m128i*)(pM0 + mask_hstep * 12));
                            __m128i _md = _mm_loadu_si128((const __m128i*)(pM0 + mask_hstep * 13));
                            __m128i _me = _mm_loadu_si128((const __m128i*)(pM0 + mask_hstep * 14));
                            __m128i _mf = _mm_loadu_si128((const __m128i*)(pM0 + mask_hstep * 15));
                            transpose8x16_epi16(_m0, _m1, _m2, _m3, _m4, _m5, _m6, _m7, _m8, _m9, _ma, _mb, _mc, _md, _me, _mf);
                            _sum0 = _mm512_add_ps(_sum0, bfloat2float_avx512(combine4x2_epi32(_m0, _m1)));
                            _sum1 = _mm512_add_ps(_sum1, bfloat2float_avx512(combine4x2_epi32(_m2, _m3)));
                            _sum2 = _mm512_add_ps(_sum2, bfloat2float_avx512(combine4x2_epi32(_m4, _m5)));
                            _sum3 = _mm512_add_ps(_sum3, bfloat2float_avx512(combine4x2_epi32(_m6, _m7)));
                            _sum4 = _mm512_add_ps(_sum4, bfloat2float_avx512(combine4x2_epi32(_m8, _m9)));
                            _sum5 = _mm512_add_ps(_sum5, bfloat2float_avx512(combine4x2_epi32(_ma, _mb)));
                            _sum6 = _mm512_add_ps(_sum6, bfloat2float_avx512(combine4x2_epi32(_mc, _md)));
                            _sum7 = _mm512_add_ps(_sum7, bfloat2float_avx512(combine4x2_epi32(_me, _mf)));
                        }
                        else
                        {
                            _sum0 = _mm512_add_ps(_sum0, _mm512_set1_ps(bfloat16_to_float32(pM[0])));
                            _sum1 = _mm512_add_ps(_sum1, _mm512_set1_ps(bfloat16_to_float32(pM[1])));
                            _sum2 = _mm512_add_ps(_sum2, _mm512_set1_ps(bfloat16_to_float32(pM[2])));
                            _sum3 = _mm512_add_ps(_sum3, _mm512_set1_ps(bfloat16_to_float32(pM[3])));
                            _sum4 = _mm512_add_ps(_sum4, _mm512_set1_ps(bfloat16_to_float32(pM[4])));
                            _sum5 = _mm512_add_ps(_sum5, _mm512_set1_ps(bfloat16_to_float32(pM[5])));
                            _sum6 = _mm512_add_ps(_sum6, _mm512_set1_ps(bfloat16_to_float32(pM[6])));
                            _sum7 = _mm512_add_ps(_sum7, _mm512_set1_ps(bfloat16_to_float32(pM[7])));
                        }
                        pM += 8;
                    }
                    _mm512_storeu_ps(score_panel, _sum0);
                    _mm512_storeu_ps(score_panel + 16, _sum1);
                    _mm512_storeu_ps(score_panel + 32, _sum2);
                    _mm512_storeu_ps(score_panel + 48, _sum3);
                    _mm512_storeu_ps(score_panel + 64, _sum4);
                    _mm512_storeu_ps(score_panel + 80, _sum5);
                    _mm512_storeu_ps(score_panel + 96, _sum6);
                    _mm512_storeu_ps(score_panel + 112, _sum7);
                    score_panel += 128;
                    _block_max = _mm512_max_ps(_block_max, _mm512_max_ps(_mm512_max_ps(_mm512_max_ps(_sum0, _sum1), _mm512_max_ps(_sum2, _sum3)), _mm512_max_ps(_mm512_max_ps(_sum4, _sum5), _mm512_max_ps(_sum6, _sum7))));
                }
#endif // defined(__x86_64__) || defined(_M_X64)
                for (; j + 3 < max_nn; j += 4)
                {
                    const unsigned short* pA = queryT_tile;
                    const unsigned short* pK = key_panel + j;
                    __m512 _sum0 = _mm512_setzero_ps();
                    __m512 _sum1 = _mm512_setzero_ps();
                    __m512 _sum2 = _mm512_setzero_ps();
                    __m512 _sum3 = _mm512_setzero_ps();
                    int d = 0;
#if __AVX512BF16__
                    pK = key_panel + j * 2;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                        __m512i _pB0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)pK));
                        __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);
                        __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                        _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_pA0, (__m512bh)_pB0);
                        _sum1 = _mm512_dpbf16_ps(_sum1, (__m512bh)_pA0, (__m512bh)_pB1);
                        _sum2 = _mm512_dpbf16_ps(_sum2, (__m512bh)_pA1, (__m512bh)_pB0);
                        _sum3 = _mm512_dpbf16_ps(_sum3, (__m512bh)_pA1, (__m512bh)_pB1);
                        pA += 32;
                        pK += NR * 2;
                    }
                    pK = key_panel + (size_t)d * NR + j;
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m512 _pA0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pA));
                        __m128 _pBs = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pK));
                        __m512 _pB0 = _mm512_broadcast_f32x4(_pBs);
                        __m512 _pA1 = _mm512_permute_ps(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                        __m512 _pB1 = _mm512_permute_ps(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                        _sum0 = _mm512_fmadd_ps(_pA0, _pB0, _sum0);
                        _sum1 = _mm512_fmadd_ps(_pA0, _pB1, _sum1);
                        _sum2 = _mm512_fmadd_ps(_pA1, _pB0, _sum2);
                        _sum3 = _mm512_fmadd_ps(_pA1, _pB1, _sum3);
                        pA += 16;
                        pK += NR;
                    }
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
                    __m512 _scale = _mm512_set1_ps(scale);
                    _sum0 = _mm512_mul_ps(_sum0, _scale);
                    _sum1 = _mm512_mul_ps(_sum1, _scale);
                    _sum2 = _mm512_mul_ps(_sum2, _scale);
                    _sum3 = _mm512_mul_ps(_sum3, _scale);
                    if (pM)
                    {
                        if (mask_per_head)
                        {
                            const unsigned short* pM0 = pM;
                            __m128i _m0 = _mm_loadl_epi64((const __m128i*)pM0);
                            __m128i _m1 = _mm_loadl_epi64((const __m128i*)(pM0 + mask_hstep));
                            __m128i _m2 = _mm_loadl_epi64((const __m128i*)(pM0 + mask_hstep * 2));
                            __m128i _m3 = _mm_loadl_epi64((const __m128i*)(pM0 + mask_hstep * 3));
                            __m128i _m4 = _mm_loadl_epi64((const __m128i*)(pM0 + mask_hstep * 4));
                            __m128i _m5 = _mm_loadl_epi64((const __m128i*)(pM0 + mask_hstep * 5));
                            __m128i _m6 = _mm_loadl_epi64((const __m128i*)(pM0 + mask_hstep * 6));
                            __m128i _m7 = _mm_loadl_epi64((const __m128i*)(pM0 + mask_hstep * 7));
                            __m128i _m8 = _mm_loadl_epi64((const __m128i*)(pM0 + mask_hstep * 8));
                            __m128i _m9 = _mm_loadl_epi64((const __m128i*)(pM0 + mask_hstep * 9));
                            __m128i _ma = _mm_loadl_epi64((const __m128i*)(pM0 + mask_hstep * 10));
                            __m128i _mb = _mm_loadl_epi64((const __m128i*)(pM0 + mask_hstep * 11));
                            __m128i _mc = _mm_loadl_epi64((const __m128i*)(pM0 + mask_hstep * 12));
                            __m128i _md = _mm_loadl_epi64((const __m128i*)(pM0 + mask_hstep * 13));
                            __m128i _me = _mm_loadl_epi64((const __m128i*)(pM0 + mask_hstep * 14));
                            __m128i _mf = _mm_loadl_epi64((const __m128i*)(pM0 + mask_hstep * 15));
                            transpose8x16_epi16(_m0, _m1, _m2, _m3, _m4, _m5, _m6, _m7, _m8, _m9, _ma, _mb, _mc, _md, _me, _mf);
                            _sum0 = _mm512_add_ps(_sum0, bfloat2float_avx512(combine4x2_epi32(_m0, _m1)));
                            _sum1 = _mm512_add_ps(_sum1, bfloat2float_avx512(combine4x2_epi32(_m2, _m3)));
                            _sum2 = _mm512_add_ps(_sum2, bfloat2float_avx512(combine4x2_epi32(_m4, _m5)));
                            _sum3 = _mm512_add_ps(_sum3, bfloat2float_avx512(combine4x2_epi32(_m6, _m7)));
                        }
                        else
                        {
                            _sum0 = _mm512_add_ps(_sum0, _mm512_set1_ps(bfloat16_to_float32(pM[0])));
                            _sum1 = _mm512_add_ps(_sum1, _mm512_set1_ps(bfloat16_to_float32(pM[1])));
                            _sum2 = _mm512_add_ps(_sum2, _mm512_set1_ps(bfloat16_to_float32(pM[2])));
                            _sum3 = _mm512_add_ps(_sum3, _mm512_set1_ps(bfloat16_to_float32(pM[3])));
                        }
                        pM += 4;
                    }
                    _mm512_storeu_ps(score_panel, _sum0);
                    _mm512_storeu_ps(score_panel + 16, _sum1);
                    _mm512_storeu_ps(score_panel + 32, _sum2);
                    _mm512_storeu_ps(score_panel + 48, _sum3);
                    score_panel += 64;
                    _block_max = _mm512_max_ps(_block_max, _mm512_max_ps(_mm512_max_ps(_sum0, _sum1), _mm512_max_ps(_sum2, _sum3)));
                }
                for (; j + 1 < max_nn; j += 2)
                {
                    __m512 _sum0 = _mm512_setzero_ps();
                    __m512 _sum1 = _mm512_setzero_ps();
                    const unsigned short* pA = queryT_tile;
                    int d = 0;
#if __AVX512BF16__
                    const int* pK_pair = (const int*)key_panel + j;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m512i _q = _mm512_loadu_si512((const __m512i*)pA);
                        _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(pK_pair[0]));
                        _sum1 = _mm512_dpbf16_ps(_sum1, (__m512bh)_q, (__m512bh)_mm512_set1_epi32(pK_pair[1]));
                        pA += 32;
                        pK_pair += NR;
                    }
#endif // __AVX512BF16__
                    const unsigned short* pK = key_panel + (size_t)d * NR + j;
                    for (; d < head_dim; d++)
                    {
                        __m512 _q = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pA));
                        _sum0 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(pK[0])), _sum0);
                        _sum1 = _mm512_fmadd_ps(_q, _mm512_set1_ps(bfloat16_to_float32(pK[1])), _sum1);
                        pA += 16;
                        pK += NR;
                    }
                    __m512 _scale = _mm512_set1_ps(scale);
                    _sum0 = _mm512_mul_ps(_sum0, _scale);
                    _sum1 = _mm512_mul_ps(_sum1, _scale);
                    if (pM)
                    {
                        __m512i _mask0 = _mm512_i32gather_epi32(_mask_index, (const int*)pM, sizeof(unsigned short));
                        __m512i _mask1 = _mm512_i32gather_epi32(_mask_index, (const int*)(pM + 1), sizeof(unsigned short));
                        _sum0 = _mm512_add_ps(_sum0, bfloat2float_avx512(_mm512_cvtepi32_epi16(_mask0)));
                        _sum1 = _mm512_add_ps(_sum1, bfloat2float_avx512(_mm512_cvtepi32_epi16(_mask1)));
                        pM += 2;
                    }
                    _mm512_storeu_ps(score_panel, _sum0);
                    _mm512_storeu_ps(score_panel + 16, _sum1);
                    score_panel += 32;
                    _block_max = _mm512_max_ps(_block_max, _mm512_max_ps(_sum0, _sum1));
                }
                for (; j < max_nn; j++)
                {
                    const unsigned short* pK = key_panel + j;
                    __m512 _sum = _mm512_setzero_ps();
                    const unsigned short* pQ = queryT_tile;
                    int d = 0;
#if __AVX512BF16__
                    const int* pK_pair = (const int*)key_panel + j;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        _sum = _mm512_dpbf16_ps(_sum, (__m512bh)_mm512_loadu_si512((const __m512i*)pQ), (__m512bh)_mm512_set1_epi32(*pK_pair));
                        pQ += 32;
                        pK_pair += NR;
                    }
                    pK = key_panel + (size_t)d * NR + j;
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        _sum = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pQ)), _mm512_set1_ps(bfloat16_to_float32(*pK)), _sum);
                        pQ += 16;
                        pK += NR;
                    }
                    _sum = _mm512_mul_ps(_sum, _mm512_set1_ps(scale));
                    if (pM)
                    {
                        __m512i _mask = _mm512_i32gather_epi32(_mask_index, (const int*)pM, sizeof(unsigned short));
                        _sum = _mm512_add_ps(_sum, bfloat2float_avx512(_mm512_cvtepi32_epi16(_mask)));
                        pM++;
                    }
                    _mm512_storeu_ps(score_panel, _sum);
                    score_panel += 16;
                    _block_max = _mm512_max_ps(_block_max, _sum);
                }
            }

            __m512 _m_new = _mm512_max_ps(_m, _block_max);
            __mmask16 alpha_active = _mm512_cmp_ps_mask(_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
            __m512 _alpha = _mm512_maskz_mov_ps(alpha_active, exp512_ps(_mm512_maskz_sub_ps(alpha_active, _m, _m_new)));

            __m512 _sum0 = _mm512_setzero_ps();
            int j = 0;
#if defined(__x86_64__) || defined(_M_X64)
            __m512 _sum1 = _mm512_setzero_ps();
            __m512 _sum2 = _mm512_setzero_ps();
            __m512 _sum3 = _mm512_setzero_ps();
            for (; j + 3 < max_jj; j += 4)
            {
                __m512 _p0 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(scoreptr), _m_new));
                __m512 _p1 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(scoreptr + 16), _m_new));
                __m512 _p2 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(scoreptr + 32), _m_new));
                __m512 _p3 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(scoreptr + 48), _m_new));
                _mm512_storeu_ps(scoreptr, _p0);
                _mm512_storeu_ps(scoreptr + 16, _p1);
                _mm512_storeu_ps(scoreptr + 32, _p2);
                _mm512_storeu_ps(scoreptr + 48, _p3);
                scoreptr += 64;
                _sum0 = _mm512_add_ps(_sum0, _p0);
                _sum1 = _mm512_add_ps(_sum1, _p1);
                _sum2 = _mm512_add_ps(_sum2, _p2);
                _sum3 = _mm512_add_ps(_sum3, _p3);
            }
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; j < max_jj; j++)
            {
                __m512 _p = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(scoreptr), _m_new));
                _mm512_storeu_ps(scoreptr, _p);
                scoreptr += 16;
                _sum0 = _mm512_add_ps(_sum0, _p);
            }
            __m512 _sum = _sum0;
#if defined(__x86_64__) || defined(_M_X64)
            _sum = _mm512_add_ps(_mm512_add_ps(_sum, _sum1), _mm512_add_ps(_sum2, _sum3));
#endif // defined(__x86_64__) || defined(_M_X64)
            _l = _mm512_add_ps(_mm512_mul_ps(_l, _alpha), _sum);
            _m = _m_new;

            float* outptr = outT_tile;
            for (int d = 0; d < value_dim;)
            {
                const int value_panel_width = sdpa_kvcache_value_panel_width(value_dim - d);
                int lane = 0;
#if defined(__x86_64__) || defined(_M_X64)
                for (; lane + 15 < value_panel_width; lane += 16)
                {
                    __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr), _alpha);
                    __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 16), _alpha);
                    __m512 _out2 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 32), _alpha);
                    __m512 _out3 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 48), _alpha);
                    __m512 _out4 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 64), _alpha);
                    __m512 _out5 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 80), _alpha);
                    __m512 _out6 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 96), _alpha);
                    __m512 _out7 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 112), _alpha);
                    __m512 _out8 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 128), _alpha);
                    __m512 _out9 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 144), _alpha);
                    __m512 _outa = _mm512_mul_ps(_mm512_loadu_ps(outptr + 160), _alpha);
                    __m512 _outb = _mm512_mul_ps(_mm512_loadu_ps(outptr + 176), _alpha);
                    __m512 _outc = _mm512_mul_ps(_mm512_loadu_ps(outptr + 192), _alpha);
                    __m512 _outd = _mm512_mul_ps(_mm512_loadu_ps(outptr + 208), _alpha);
                    __m512 _oute = _mm512_mul_ps(_mm512_loadu_ps(outptr + 224), _alpha);
                    __m512 _outf = _mm512_mul_ps(_mm512_loadu_ps(outptr + 240), _alpha);
                    const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 16;
                        const unsigned short* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            __m512 _p = _mm512_loadu_ps(pS);
                            __m512 _v = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pV));
                            __m512 _v0 = _mm512_shuffle_f32x4(_v, _v, _MM_SHUFFLE(0, 0, 0, 0));
                            __m512 _v1 = _mm512_shuffle_f32x4(_v, _v, _MM_SHUFFLE(1, 1, 1, 1));
                            __m512 _v2 = _mm512_shuffle_f32x4(_v, _v, _MM_SHUFFLE(2, 2, 2, 2));
                            __m512 _v3 = _mm512_shuffle_f32x4(_v, _v, _MM_SHUFFLE(3, 3, 3, 3));
                            _out0 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v0, _MM_SHUFFLE(0, 0, 0, 0)), _out0);
                            _out1 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v0, _MM_SHUFFLE(1, 1, 1, 1)), _out1);
                            _out2 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v0, _MM_SHUFFLE(2, 2, 2, 2)), _out2);
                            _out3 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v0, _MM_SHUFFLE(3, 3, 3, 3)), _out3);
                            _out4 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v1, _MM_SHUFFLE(0, 0, 0, 0)), _out4);
                            _out5 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v1, _MM_SHUFFLE(1, 1, 1, 1)), _out5);
                            _out6 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v1, _MM_SHUFFLE(2, 2, 2, 2)), _out6);
                            _out7 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v1, _MM_SHUFFLE(3, 3, 3, 3)), _out7);
                            _out8 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v2, _MM_SHUFFLE(0, 0, 0, 0)), _out8);
                            _out9 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v2, _MM_SHUFFLE(1, 1, 1, 1)), _out9);
                            _outa = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v2, _MM_SHUFFLE(2, 2, 2, 2)), _outa);
                            _outb = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v2, _MM_SHUFFLE(3, 3, 3, 3)), _outb);
                            _outc = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v3, _MM_SHUFFLE(0, 0, 0, 0)), _outc);
                            _outd = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v3, _MM_SHUFFLE(1, 1, 1, 1)), _outd);
                            _oute = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v3, _MM_SHUFFLE(2, 2, 2, 2)), _oute);
                            _outf = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v3, _MM_SHUFFLE(3, 3, 3, 3)), _outf);
                            pS += 16;
                            pV += value_panel_width;
                        }
                        pV_panel += (size_t)NR * value_dim;
                    }
                    _mm512_storeu_ps(outptr, _out0);
                    _mm512_storeu_ps(outptr + 16, _out1);
                    _mm512_storeu_ps(outptr + 32, _out2);
                    _mm512_storeu_ps(outptr + 48, _out3);
                    _mm512_storeu_ps(outptr + 64, _out4);
                    _mm512_storeu_ps(outptr + 80, _out5);
                    _mm512_storeu_ps(outptr + 96, _out6);
                    _mm512_storeu_ps(outptr + 112, _out7);
                    _mm512_storeu_ps(outptr + 128, _out8);
                    _mm512_storeu_ps(outptr + 144, _out9);
                    _mm512_storeu_ps(outptr + 160, _outa);
                    _mm512_storeu_ps(outptr + 176, _outb);
                    _mm512_storeu_ps(outptr + 192, _outc);
                    _mm512_storeu_ps(outptr + 208, _outd);
                    _mm512_storeu_ps(outptr + 224, _oute);
                    _mm512_storeu_ps(outptr + 240, _outf);
                    outptr += 256;
                }
                for (; lane + 7 < value_panel_width; lane += 8)
                {
                    __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr), _alpha);
                    __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 16), _alpha);
                    __m512 _out2 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 32), _alpha);
                    __m512 _out3 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 48), _alpha);
                    __m512 _out4 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 64), _alpha);
                    __m512 _out5 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 80), _alpha);
                    __m512 _out6 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 96), _alpha);
                    __m512 _out7 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 112), _alpha);
                    const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 16;
                        const unsigned short* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            __m512 _p = _mm512_loadu_ps(pS);
                            __m256 _v = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pV));
                            __m512 _v0 = _mm512_broadcast_f32x4(_mm256_castps256_ps128(_v));
                            __m512 _v1 = _mm512_broadcast_f32x4(_mm256_extractf128_ps(_v, 1));
                            _out0 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v0, _MM_SHUFFLE(0, 0, 0, 0)), _out0);
                            _out1 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v0, _MM_SHUFFLE(1, 1, 1, 1)), _out1);
                            _out2 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v0, _MM_SHUFFLE(2, 2, 2, 2)), _out2);
                            _out3 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v0, _MM_SHUFFLE(3, 3, 3, 3)), _out3);
                            _out4 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v1, _MM_SHUFFLE(0, 0, 0, 0)), _out4);
                            _out5 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v1, _MM_SHUFFLE(1, 1, 1, 1)), _out5);
                            _out6 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v1, _MM_SHUFFLE(2, 2, 2, 2)), _out6);
                            _out7 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v1, _MM_SHUFFLE(3, 3, 3, 3)), _out7);
                            pS += 16;
                            pV += value_panel_width;
                        }
                        pV_panel += (size_t)NR * value_dim;
                    }
                    _mm512_storeu_ps(outptr, _out0);
                    _mm512_storeu_ps(outptr + 16, _out1);
                    _mm512_storeu_ps(outptr + 32, _out2);
                    _mm512_storeu_ps(outptr + 48, _out3);
                    _mm512_storeu_ps(outptr + 64, _out4);
                    _mm512_storeu_ps(outptr + 80, _out5);
                    _mm512_storeu_ps(outptr + 96, _out6);
                    _mm512_storeu_ps(outptr + 112, _out7);
                    outptr += 128;
                }
#endif // defined(__x86_64__) || defined(_M_X64)
                for (; lane + 3 < value_panel_width; lane += 4)
                {
                    __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr), _alpha);
                    __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 16), _alpha);
                    __m512 _out2 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 32), _alpha);
                    __m512 _out3 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 48), _alpha);
                    const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 16;
                        const unsigned short* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            __m512 _p = _mm512_loadu_ps(pS);
                            __m512 _v = _mm512_broadcast_f32x4(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pV)));
                            _out0 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v, _MM_SHUFFLE(0, 0, 0, 0)), _out0);
                            _out1 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v, _MM_SHUFFLE(1, 1, 1, 1)), _out1);
                            _out2 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v, _MM_SHUFFLE(2, 2, 2, 2)), _out2);
                            _out3 = _mm512_fmadd_ps(_p, _mm512_permute_ps(_v, _MM_SHUFFLE(3, 3, 3, 3)), _out3);
                            pS += 16;
                            pV += value_panel_width;
                        }
                        pV_panel += (size_t)NR * value_dim;
                    }
                    _mm512_storeu_ps(outptr, _out0);
                    _mm512_storeu_ps(outptr + 16, _out1);
                    _mm512_storeu_ps(outptr + 32, _out2);
                    _mm512_storeu_ps(outptr + 48, _out3);
                    outptr += 64;
                }
                for (; lane + 1 < value_panel_width; lane += 2)
                {
                    __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr), _alpha);
                    __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 16), _alpha);
                    const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 16;
                        const unsigned short* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            __m512 _p = _mm512_loadu_ps(pS);
                            _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[0])), _out0);
                            _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(bfloat16_to_float32(pV[1])), _out1);
                            pS += 16;
                            pV += value_panel_width;
                        }
                        pV_panel += (size_t)NR * value_dim;
                    }
                    _mm512_storeu_ps(outptr, _out0);
                    _mm512_storeu_ps(outptr + 16, _out1);
                    outptr += 32;
                }
                for (; lane < value_panel_width; lane++)
                {
                    __m512 _out = _mm512_mul_ps(_mm512_loadu_ps(outptr), _alpha);
                    const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 16;
                        const unsigned short* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            _out = _mm512_fmadd_ps(_mm512_loadu_ps(pS), _mm512_set1_ps(bfloat16_to_float32(*pV)), _out);
                            pS += 16;
                            pV += value_panel_width;
                        }
                        pV_panel += (size_t)NR * value_dim;
                    }
                    _mm512_storeu_ps(outptr, _out);
                    outptr += 16;
                }
                d += value_panel_width;
            }
        }

        {
            float* output = top_blob.channel(q0 + qq);
            const size_t output_cstep = top_blob.cstep;
            __mmask16 nonzero = _mm512_cmp_ps_mask(_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
            __m512 _out_scale = _mm512_maskz_div_ps(nonzero, _mm512_set1_ps(1.f), _l);
            const float* pO = outT_tile;
            float* p0 = output;
            int d = 0;
            for (; d + 15 < value_dim; d += 16)
            {
                __m512 _r0 = _mm512_mul_ps(_mm512_loadu_ps(pO), _out_scale);
                __m512 _r1 = _mm512_mul_ps(_mm512_loadu_ps(pO + 16), _out_scale);
                __m512 _r2 = _mm512_mul_ps(_mm512_loadu_ps(pO + 32), _out_scale);
                __m512 _r3 = _mm512_mul_ps(_mm512_loadu_ps(pO + 48), _out_scale);
                __m512 _r4 = _mm512_mul_ps(_mm512_loadu_ps(pO + 64), _out_scale);
                __m512 _r5 = _mm512_mul_ps(_mm512_loadu_ps(pO + 80), _out_scale);
                __m512 _r6 = _mm512_mul_ps(_mm512_loadu_ps(pO + 96), _out_scale);
                __m512 _r7 = _mm512_mul_ps(_mm512_loadu_ps(pO + 112), _out_scale);
                __m512 _r8 = _mm512_mul_ps(_mm512_loadu_ps(pO + 128), _out_scale);
                __m512 _r9 = _mm512_mul_ps(_mm512_loadu_ps(pO + 144), _out_scale);
                __m512 _ra = _mm512_mul_ps(_mm512_loadu_ps(pO + 160), _out_scale);
                __m512 _rb = _mm512_mul_ps(_mm512_loadu_ps(pO + 176), _out_scale);
                __m512 _rc = _mm512_mul_ps(_mm512_loadu_ps(pO + 192), _out_scale);
                __m512 _rd = _mm512_mul_ps(_mm512_loadu_ps(pO + 208), _out_scale);
                __m512 _re = _mm512_mul_ps(_mm512_loadu_ps(pO + 224), _out_scale);
                __m512 _rf = _mm512_mul_ps(_mm512_loadu_ps(pO + 240), _out_scale);
                transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
                _mm512_storeu_ps(p0, _r0);
                _mm512_storeu_ps(p0 + output_cstep, _r1);
                _mm512_storeu_ps(p0 + output_cstep * 2, _r2);
                _mm512_storeu_ps(p0 + output_cstep * 3, _r3);
                _mm512_storeu_ps(p0 + output_cstep * 4, _r4);
                _mm512_storeu_ps(p0 + output_cstep * 5, _r5);
                _mm512_storeu_ps(p0 + output_cstep * 6, _r6);
                _mm512_storeu_ps(p0 + output_cstep * 7, _r7);
                _mm512_storeu_ps(p0 + output_cstep * 8, _r8);
                _mm512_storeu_ps(p0 + output_cstep * 9, _r9);
                _mm512_storeu_ps(p0 + output_cstep * 10, _ra);
                _mm512_storeu_ps(p0 + output_cstep * 11, _rb);
                _mm512_storeu_ps(p0 + output_cstep * 12, _rc);
                _mm512_storeu_ps(p0 + output_cstep * 13, _rd);
                _mm512_storeu_ps(p0 + output_cstep * 14, _re);
                _mm512_storeu_ps(p0 + output_cstep * 15, _rf);
                p0 += 16;
                pO += 256;
            }
            for (; d < value_dim; d++)
            {
                __m512 _r = _mm512_mul_ps(_mm512_loadu_ps(pO), _out_scale);
                __m128 _r0 = _mm512_extractf32x4_ps(_r, 0);
                __m128 _r1 = _mm512_extractf32x4_ps(_r, 1);
                __m128 _r2 = _mm512_extractf32x4_ps(_r, 2);
                __m128 _r3 = _mm512_extractf32x4_ps(_r, 3);
                *p0 = _mm_cvtss_f32(_r0);
                p0[output_cstep] = _mm_cvtss_f32(_mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(1, 1, 1, 1)));
                p0[output_cstep * 2] = _mm_cvtss_f32(_mm_movehl_ps(_r0, _r0));
                p0[output_cstep * 3] = _mm_cvtss_f32(_mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(3, 3, 3, 3)));
                p0[output_cstep * 4] = _mm_cvtss_f32(_r1);
                p0[output_cstep * 5] = _mm_cvtss_f32(_mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(1, 1, 1, 1)));
                p0[output_cstep * 6] = _mm_cvtss_f32(_mm_movehl_ps(_r1, _r1));
                p0[output_cstep * 7] = _mm_cvtss_f32(_mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(3, 3, 3, 3)));
                p0[output_cstep * 8] = _mm_cvtss_f32(_r2);
                p0[output_cstep * 9] = _mm_cvtss_f32(_mm_shuffle_ps(_r2, _r2, _MM_SHUFFLE(1, 1, 1, 1)));
                p0[output_cstep * 10] = _mm_cvtss_f32(_mm_movehl_ps(_r2, _r2));
                p0[output_cstep * 11] = _mm_cvtss_f32(_mm_shuffle_ps(_r2, _r2, _MM_SHUFFLE(3, 3, 3, 3)));
                p0[output_cstep * 12] = _mm_cvtss_f32(_r3);
                p0[output_cstep * 13] = _mm_cvtss_f32(_mm_shuffle_ps(_r3, _r3, _MM_SHUFFLE(1, 1, 1, 1)));
                p0[output_cstep * 14] = _mm_cvtss_f32(_mm_movehl_ps(_r3, _r3));
                p0[output_cstep * 15] = _mm_cvtss_f32(_mm_shuffle_ps(_r3, _r3, _MM_SHUFFLE(3, 3, 3, 3)));
                p0++;
                pO += 16;
            }
        }
    }
#endif // __AVX512F__
    for (; qq + 7 < max_qq; qq += 8)
    {
        const bool mask_per_head = !attn_mask_blob.empty() && attn_mask_blob.dims == 3 && attn_mask_blob.c > 1;
        const unsigned short* mask = 0;
        if (!attn_mask_blob.empty())
        {
            if (attn_mask_blob.dims == 3)
                mask = attn_mask_blob.channel(mask_per_head ? q0 + qq : 0);
            else
                mask = attn_mask_blob;
        }
        const int mask_cstep = mask_per_head ? attn_mask_blob.cstep : 0;
#if __AVX2__
        __m256i _mask_index = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(mask_cstep));
#endif // __AVX2__

        const unsigned short* queryT_tile = queryT_ptr + (size_t)qq * head_dim;
        float* scoreT_tile = scoreT_ptr + (size_t)qq * block_n;
        float* outT_tile = outT_ptr + (size_t)qq * value_dim;
        __m256 _m = _mm256_set1_ps(-FLT_MAX);
        __m256 _l = _mm256_setzero_ps();
        memset(outT_tile, 0, (size_t)value_dim * 8 * sizeof(float));

        for (int n = 0; n < key_seqlen; n += block_n)
        {
            const int max_jj = std::min(key_seqlen - n, block_n);
            float* scoreptr = scoreT_tile;
            const unsigned short* pM = mask ? mask + n : 0;
            __m256 _block_max = _mm256_set1_ps(-FLT_MAX);
            float* score_panel = scoreptr;

            for (int jj = 0; jj < max_jj; jj += NR)
            {
                const int max_nn = std::min(NR, max_jj - jj);
                const unsigned short* key_panel = (const unsigned short*)key_cache_head + (size_t)(n + jj) * head_dim;
                int j = 0;
#if defined(__x86_64__) || defined(_M_X64)
                for (; j + 7 < max_nn; j += 8)
                {
                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    __m256 _sum4 = _mm256_setzero_ps();
                    __m256 _sum5 = _mm256_setzero_ps();
                    __m256 _sum6 = _mm256_setzero_ps();
                    __m256 _sum7 = _mm256_setzero_ps();
                    const unsigned short* pA = queryT_tile;
                    int d = 0;
#if __AVX512BF16__
                    const unsigned short* pK_pair = key_panel + j * 2;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                        __m256i _pB0 = _mm256_loadu_si256((const __m256i*)pK_pair);
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
                        pK_pair += NR * 2;
                    }
#endif // __AVX512BF16__
                    const unsigned short* pK = key_panel + (size_t)d * NR + j;
                    for (; d < head_dim; d++)
                    {
                        __m256 _pA0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pA));
                        __m256 _pB0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pK));

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
                        pK += NR;
                    }

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

                    __m256 _scale = _mm256_set1_ps(scale);
                    _sum0 = _mm256_mul_ps(_sum0, _scale);
                    _sum1 = _mm256_mul_ps(_sum1, _scale);
                    _sum2 = _mm256_mul_ps(_sum2, _scale);
                    _sum3 = _mm256_mul_ps(_sum3, _scale);
                    _sum4 = _mm256_mul_ps(_sum4, _scale);
                    _sum5 = _mm256_mul_ps(_sum5, _scale);
                    _sum6 = _mm256_mul_ps(_sum6, _scale);
                    _sum7 = _mm256_mul_ps(_sum7, _scale);
                    if (pM)
                    {
                        if (mask_per_head)
                        {
                            __m128i _m0 = _mm_loadu_si128((const __m128i*)pM);
                            __m128i _m1 = _mm_loadu_si128((const __m128i*)(pM + mask_cstep));
                            __m128i _m2 = _mm_loadu_si128((const __m128i*)(pM + mask_cstep * 2));
                            __m128i _m3 = _mm_loadu_si128((const __m128i*)(pM + mask_cstep * 3));
                            __m128i _m4 = _mm_loadu_si128((const __m128i*)(pM + mask_cstep * 4));
                            __m128i _m5 = _mm_loadu_si128((const __m128i*)(pM + mask_cstep * 5));
                            __m128i _m6 = _mm_loadu_si128((const __m128i*)(pM + mask_cstep * 6));
                            __m128i _m7 = _mm_loadu_si128((const __m128i*)(pM + mask_cstep * 7));
                            transpose8x8_epi16(_m0, _m1, _m2, _m3, _m4, _m5, _m6, _m7);
                            _sum0 = _mm256_add_ps(_sum0, bfloat2float_avx(_m0));
                            _sum1 = _mm256_add_ps(_sum1, bfloat2float_avx(_m1));
                            _sum2 = _mm256_add_ps(_sum2, bfloat2float_avx(_m2));
                            _sum3 = _mm256_add_ps(_sum3, bfloat2float_avx(_m3));
                            _sum4 = _mm256_add_ps(_sum4, bfloat2float_avx(_m4));
                            _sum5 = _mm256_add_ps(_sum5, bfloat2float_avx(_m5));
                            _sum6 = _mm256_add_ps(_sum6, bfloat2float_avx(_m6));
                            _sum7 = _mm256_add_ps(_sum7, bfloat2float_avx(_m7));
                        }
                        else
                        {
                            _sum0 = _mm256_add_ps(_sum0, _mm256_set1_ps(bfloat16_to_float32(pM[0])));
                            _sum1 = _mm256_add_ps(_sum1, _mm256_set1_ps(bfloat16_to_float32(pM[1])));
                            _sum2 = _mm256_add_ps(_sum2, _mm256_set1_ps(bfloat16_to_float32(pM[2])));
                            _sum3 = _mm256_add_ps(_sum3, _mm256_set1_ps(bfloat16_to_float32(pM[3])));
                            _sum4 = _mm256_add_ps(_sum4, _mm256_set1_ps(bfloat16_to_float32(pM[4])));
                            _sum5 = _mm256_add_ps(_sum5, _mm256_set1_ps(bfloat16_to_float32(pM[5])));
                            _sum6 = _mm256_add_ps(_sum6, _mm256_set1_ps(bfloat16_to_float32(pM[6])));
                            _sum7 = _mm256_add_ps(_sum7, _mm256_set1_ps(bfloat16_to_float32(pM[7])));
                        }
                        pM += 8;
                    }
                    _mm256_storeu_ps(score_panel, _sum0);
                    _mm256_storeu_ps(score_panel + 8, _sum1);
                    _mm256_storeu_ps(score_panel + 16, _sum2);
                    _mm256_storeu_ps(score_panel + 24, _sum3);
                    _mm256_storeu_ps(score_panel + 32, _sum4);
                    _mm256_storeu_ps(score_panel + 40, _sum5);
                    _mm256_storeu_ps(score_panel + 48, _sum6);
                    _mm256_storeu_ps(score_panel + 56, _sum7);
                    score_panel += 64;
                    _block_max = _mm256_max_ps(_block_max, _mm256_max_ps(_mm256_max_ps(_mm256_max_ps(_sum0, _sum1), _mm256_max_ps(_sum2, _sum3)), _mm256_max_ps(_mm256_max_ps(_sum4, _sum5), _mm256_max_ps(_sum6, _sum7))));
                }
#endif // defined(__x86_64__) || defined(_M_X64)
                for (; j + 3 < max_nn; j += 4)
                {
                    const unsigned short* pA = queryT_tile;
                    const unsigned short* pK = key_panel + j;
                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    int d = 0;
#if __AVX512BF16__
                    pK = key_panel + j * 2;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                        __m128i _pB = _mm_loadu_si128((const __m128i*)pK);
                        __m256i _pB0 = combine4x2_epi32(_pB, _pB);
                        __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                        __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                        _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_pA0, (__m256bh)_pB0);
                        _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_pA0, (__m256bh)_pB1);
                        _sum2 = _mm256_dpbf16_ps(_sum2, (__m256bh)_pA1, (__m256bh)_pB0);
                        _sum3 = _mm256_dpbf16_ps(_sum3, (__m256bh)_pA1, (__m256bh)_pB1);
                        pA += 16;
                        pK += NR * 2;
                    }
                    pK = key_panel + (size_t)d * NR + j;
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m256 _pA0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pA));
                        __m128 _pBs = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pK));
                        __m256 _pB0 = combine4x2_ps(_pBs, _pBs);
                        __m256 _pA1 = _mm256_permute_ps(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                        __m256 _pB1 = _mm256_permute_ps(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                        _sum0 = _mm256_comp_fmadd_ps(_pA0, _pB0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_pA0, _pB1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_pA1, _pB0, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_pA1, _pB1, _sum3);
                        pA += 8;
                        pK += NR;
                    }
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
                    __m256 _scale = _mm256_set1_ps(scale);
                    _sum0 = _mm256_mul_ps(_sum0, _scale);
                    _sum1 = _mm256_mul_ps(_sum1, _scale);
                    _sum2 = _mm256_mul_ps(_sum2, _scale);
                    _sum3 = _mm256_mul_ps(_sum3, _scale);
                    if (pM)
                    {
                        if (mask_per_head)
                        {
                            __m128i _m0 = _mm_loadl_epi64((const __m128i*)pM);
                            __m128i _m1 = _mm_loadl_epi64((const __m128i*)(pM + mask_cstep));
                            __m128i _m2 = _mm_loadl_epi64((const __m128i*)(pM + mask_cstep * 2));
                            __m128i _m3 = _mm_loadl_epi64((const __m128i*)(pM + mask_cstep * 3));
                            __m128i _m4 = _mm_loadl_epi64((const __m128i*)(pM + mask_cstep * 4));
                            __m128i _m5 = _mm_loadl_epi64((const __m128i*)(pM + mask_cstep * 5));
                            __m128i _m6 = _mm_loadl_epi64((const __m128i*)(pM + mask_cstep * 6));
                            __m128i _m7 = _mm_loadl_epi64((const __m128i*)(pM + mask_cstep * 7));
                            transpose8x8_epi16(_m0, _m1, _m2, _m3, _m4, _m5, _m6, _m7);
                            _sum0 = _mm256_add_ps(_sum0, bfloat2float_avx(_m0));
                            _sum1 = _mm256_add_ps(_sum1, bfloat2float_avx(_m1));
                            _sum2 = _mm256_add_ps(_sum2, bfloat2float_avx(_m2));
                            _sum3 = _mm256_add_ps(_sum3, bfloat2float_avx(_m3));
                        }
                        else
                        {
                            _sum0 = _mm256_add_ps(_sum0, _mm256_set1_ps(bfloat16_to_float32(pM[0])));
                            _sum1 = _mm256_add_ps(_sum1, _mm256_set1_ps(bfloat16_to_float32(pM[1])));
                            _sum2 = _mm256_add_ps(_sum2, _mm256_set1_ps(bfloat16_to_float32(pM[2])));
                            _sum3 = _mm256_add_ps(_sum3, _mm256_set1_ps(bfloat16_to_float32(pM[3])));
                        }
                        pM += 4;
                    }
                    _mm256_storeu_ps(score_panel, _sum0);
                    _mm256_storeu_ps(score_panel + 8, _sum1);
                    _mm256_storeu_ps(score_panel + 16, _sum2);
                    _mm256_storeu_ps(score_panel + 24, _sum3);
                    score_panel += 32;
                    _block_max = _mm256_max_ps(_block_max, _mm256_max_ps(_mm256_max_ps(_sum0, _sum1), _mm256_max_ps(_sum2, _sum3)));
                }
                for (; j + 1 < max_nn; j += 2)
                {
                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    const unsigned short* pA = queryT_tile;
                    int d = 0;
#if __AVX512BF16__
                    const int* pK_pair = (const int*)key_panel + j;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m256i _q = _mm256_loadu_si256((const __m256i*)pA);
                        _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(pK_pair[0]));
                        _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_q, (__m256bh)_mm256_set1_epi32(pK_pair[1]));
                        pA += 16;
                        pK_pair += NR;
                    }
#endif // __AVX512BF16__
                    const unsigned short* pK = key_panel + (size_t)d * NR + j;
                    for (; d < head_dim; d++)
                    {
                        __m256 _q = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pA));
                        _sum0 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(pK[0])), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(bfloat16_to_float32(pK[1])), _sum1);
                        pA += 8;
                        pK += NR;
                    }
                    __m256 _scale = _mm256_set1_ps(scale);
                    _sum0 = _mm256_mul_ps(_sum0, _scale);
                    _sum1 = _mm256_mul_ps(_sum1, _scale);
                    if (pM)
                    {
#if __AVX2__
                        __m256i _mask0_bf16 = _mm256_i32gather_epi32((const int*)pM, _mask_index, sizeof(unsigned short));
                        __m256i _mask1_bf16 = _mm256_i32gather_epi32((const int*)(pM + 1), _mask_index, sizeof(unsigned short));
                        _sum0 = _mm256_add_ps(_sum0, _mm256_castsi256_ps(_mm256_slli_epi32(_mask0_bf16, 16)));
                        _sum1 = _mm256_add_ps(_sum1, _mm256_castsi256_ps(_mm256_slli_epi32(_mask1_bf16, 16)));
#else
                        _sum0 = _mm256_add_ps(_sum0, _mm256_set_ps(bfloat16_to_float32(pM[mask_cstep * 7]), bfloat16_to_float32(pM[mask_cstep * 6]), bfloat16_to_float32(pM[mask_cstep * 5]), bfloat16_to_float32(pM[mask_cstep * 4]), bfloat16_to_float32(pM[mask_cstep * 3]), bfloat16_to_float32(pM[mask_cstep * 2]), bfloat16_to_float32(pM[mask_cstep]), bfloat16_to_float32(pM[0])));
                        _sum1 = _mm256_add_ps(_sum1, _mm256_set_ps(bfloat16_to_float32(pM[mask_cstep * 7 + 1]), bfloat16_to_float32(pM[mask_cstep * 6 + 1]), bfloat16_to_float32(pM[mask_cstep * 5 + 1]), bfloat16_to_float32(pM[mask_cstep * 4 + 1]), bfloat16_to_float32(pM[mask_cstep * 3 + 1]), bfloat16_to_float32(pM[mask_cstep * 2 + 1]), bfloat16_to_float32(pM[mask_cstep + 1]), bfloat16_to_float32(pM[1])));
#endif // __AVX2__
                        pM += 2;
                    }
                    _mm256_storeu_ps(score_panel, _sum0);
                    _mm256_storeu_ps(score_panel + 8, _sum1);
                    score_panel += 16;
                    _block_max = _mm256_max_ps(_block_max, _mm256_max_ps(_sum0, _sum1));
                }
                for (; j < max_nn; j++)
                {
                    const unsigned short* pK = key_panel + j;
                    __m256 _sum = _mm256_setzero_ps();
                    const unsigned short* pQ = queryT_tile;
                    int d = 0;
#if __AVX512BF16__
                    const int* pK_pair = (const int*)key_panel + j;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        _sum = _mm256_dpbf16_ps(_sum, (__m256bh)_mm256_loadu_si256((const __m256i*)pQ), (__m256bh)_mm256_set1_epi32(*pK_pair));
                        pQ += 16;
                        pK_pair += NR;
                    }
                    pK = key_panel + (size_t)d * NR + j;
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        _sum = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)pQ)), _mm256_set1_ps(bfloat16_to_float32(*pK)), _sum);
                        pQ += 8;
                        pK += NR;
                    }
                    _sum = _mm256_mul_ps(_sum, _mm256_set1_ps(scale));
                    if (pM)
                    {
#if __AVX2__
                        __m256i _mask_bf16 = _mm256_i32gather_epi32((const int*)pM, _mask_index, sizeof(unsigned short));
                        __m256 _mask = _mm256_castsi256_ps(_mm256_slli_epi32(_mask_bf16, 16));
#else
                        __m256 _mask = _mm256_set_ps(bfloat16_to_float32(pM[mask_cstep * 7]), bfloat16_to_float32(pM[mask_cstep * 6]), bfloat16_to_float32(pM[mask_cstep * 5]), bfloat16_to_float32(pM[mask_cstep * 4]), bfloat16_to_float32(pM[mask_cstep * 3]), bfloat16_to_float32(pM[mask_cstep * 2]), bfloat16_to_float32(pM[mask_cstep]), bfloat16_to_float32(pM[0]));
#endif // __AVX2__
                        _sum = _mm256_add_ps(_sum, _mask);
                        pM++;
                    }
                    _mm256_storeu_ps(score_panel, _sum);
                    score_panel += 8;
                    _block_max = _mm256_max_ps(_block_max, _sum);
                }
            }

            __m256 _m_new = _mm256_max_ps(_m, _block_max);
            __m256 _alpha_active = _mm256_cmp_ps(_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
            __m256 _alpha = exp256_ps(_mm256_and_ps(_alpha_active, _mm256_sub_ps(_m, _m_new)));
            _alpha = _mm256_and_ps(_alpha, _alpha_active);

            __m256 _sum0 = _mm256_setzero_ps();
            int j = 0;
#if defined(__x86_64__) || defined(_M_X64)
            __m256 _sum1 = _mm256_setzero_ps();
            __m256 _sum2 = _mm256_setzero_ps();
            __m256 _sum3 = _mm256_setzero_ps();
            for (; j + 3 < max_jj; j += 4)
            {
                __m256 _p0 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr), _m_new));
                __m256 _p1 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr + 8), _m_new));
                __m256 _p2 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr + 16), _m_new));
                __m256 _p3 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr + 24), _m_new));
                _mm256_storeu_ps(scoreptr, _p0);
                _mm256_storeu_ps(scoreptr + 8, _p1);
                _mm256_storeu_ps(scoreptr + 16, _p2);
                _mm256_storeu_ps(scoreptr + 24, _p3);
                scoreptr += 32;
                _sum0 = _mm256_add_ps(_sum0, _p0);
                _sum1 = _mm256_add_ps(_sum1, _p1);
                _sum2 = _mm256_add_ps(_sum2, _p2);
                _sum3 = _mm256_add_ps(_sum3, _p3);
            }
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; j < max_jj; j++)
            {
                __m256 _p = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr), _m_new));
                _mm256_storeu_ps(scoreptr, _p);
                scoreptr += 8;
                _sum0 = _mm256_add_ps(_sum0, _p);
            }
            __m256 _sum = _sum0;
#if defined(__x86_64__) || defined(_M_X64)
            _sum = _mm256_add_ps(_mm256_add_ps(_sum, _sum1), _mm256_add_ps(_sum2, _sum3));
#endif // defined(__x86_64__) || defined(_M_X64)
            _l = _mm256_add_ps(_mm256_mul_ps(_l, _alpha), _sum);
            _m = _m_new;

            float* outptr = outT_tile;
            for (int d = 0; d < value_dim;)
            {
                const int value_panel_width = sdpa_kvcache_value_panel_width(value_dim - d);
                int lane = 0;
#if defined(__x86_64__) || defined(_M_X64)
                for (; lane + 7 < value_panel_width; lane += 8)
                {
                    __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr), _alpha);
                    __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 8), _alpha);
                    __m256 _out2 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 16), _alpha);
                    __m256 _out3 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 24), _alpha);
                    __m256 _out4 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 32), _alpha);
                    __m256 _out5 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 40), _alpha);
                    __m256 _out6 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 48), _alpha);
                    __m256 _out7 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 56), _alpha);
                    const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 8;
                        const unsigned short* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            __m256 _p = _mm256_loadu_ps(pS);
                            __m256 _v = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pV));
                            __m256 _v0 = _mm256_permute2f128_ps(_v, _v, 0x00);
                            __m256 _v1 = _mm256_permute2f128_ps(_v, _v, 0x11);
                            _out0 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v0, _MM_SHUFFLE(0, 0, 0, 0)), _out0);
                            _out1 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v0, _MM_SHUFFLE(1, 1, 1, 1)), _out1);
                            _out2 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v0, _MM_SHUFFLE(2, 2, 2, 2)), _out2);
                            _out3 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v0, _MM_SHUFFLE(3, 3, 3, 3)), _out3);
                            _out4 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v1, _MM_SHUFFLE(0, 0, 0, 0)), _out4);
                            _out5 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v1, _MM_SHUFFLE(1, 1, 1, 1)), _out5);
                            _out6 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v1, _MM_SHUFFLE(2, 2, 2, 2)), _out6);
                            _out7 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v1, _MM_SHUFFLE(3, 3, 3, 3)), _out7);
                            pS += 8;
                            pV += value_panel_width;
                        }
                        pV_panel += (size_t)NR * value_dim;
                    }
                    _mm256_storeu_ps(outptr, _out0);
                    _mm256_storeu_ps(outptr + 8, _out1);
                    _mm256_storeu_ps(outptr + 16, _out2);
                    _mm256_storeu_ps(outptr + 24, _out3);
                    _mm256_storeu_ps(outptr + 32, _out4);
                    _mm256_storeu_ps(outptr + 40, _out5);
                    _mm256_storeu_ps(outptr + 48, _out6);
                    _mm256_storeu_ps(outptr + 56, _out7);
                    outptr += 64;
                }
#endif // defined(__x86_64__) || defined(_M_X64)
                for (; lane + 3 < value_panel_width; lane += 4)
                {
                    __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr), _alpha);
                    __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 8), _alpha);
                    __m256 _out2 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 16), _alpha);
                    __m256 _out3 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 24), _alpha);
                    const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 8;
                        const unsigned short* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            __m256 _p = _mm256_loadu_ps(pS);
                            __m128 _v128 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pV));
                            __m256 _v = _mm256_insertf128_ps(_mm256_castps128_ps256(_v128), _v128, 1);
                            _out0 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v, _MM_SHUFFLE(0, 0, 0, 0)), _out0);
                            _out1 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v, _MM_SHUFFLE(1, 1, 1, 1)), _out1);
                            _out2 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v, _MM_SHUFFLE(2, 2, 2, 2)), _out2);
                            _out3 = _mm256_comp_fmadd_ps(_p, _mm256_permute_ps(_v, _MM_SHUFFLE(3, 3, 3, 3)), _out3);
                            pS += 8;
                            pV += value_panel_width;
                        }
                        pV_panel += (size_t)NR * value_dim;
                    }
                    _mm256_storeu_ps(outptr, _out0);
                    _mm256_storeu_ps(outptr + 8, _out1);
                    _mm256_storeu_ps(outptr + 16, _out2);
                    _mm256_storeu_ps(outptr + 24, _out3);
                    outptr += 32;
                }
                for (; lane + 1 < value_panel_width; lane += 2)
                {
                    __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr), _alpha);
                    __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 8), _alpha);
                    const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 8;
                        const unsigned short* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            __m256 _p = _mm256_loadu_ps(pS);
                            _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[0])), _out0);
                            _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(bfloat16_to_float32(pV[1])), _out1);
                            pS += 8;
                            pV += value_panel_width;
                        }
                        pV_panel += (size_t)NR * value_dim;
                    }
                    _mm256_storeu_ps(outptr, _out0);
                    _mm256_storeu_ps(outptr + 8, _out1);
                    outptr += 16;
                }
                for (; lane < value_panel_width; lane++)
                {
                    __m256 _out = _mm256_mul_ps(_mm256_loadu_ps(outptr), _alpha);
                    const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 8;
                        const unsigned short* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pS), _mm256_set1_ps(bfloat16_to_float32(*pV)), _out);
                            pS += 8;
                            pV += value_panel_width;
                        }
                        pV_panel += (size_t)NR * value_dim;
                    }
                    _mm256_storeu_ps(outptr, _out);
                    outptr += 8;
                }
                d += value_panel_width;
            }
        }

        {
            float* output = top_blob.channel(q0 + qq);
            const size_t output_cstep = top_blob.cstep;
            __m256 _nonzero = _mm256_cmp_ps(_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
            __m256 _denom = _mm256_blendv_ps(_mm256_set1_ps(1.f), _l, _nonzero);
            __m256 _out_scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(1.f), _denom), _nonzero);
            const float* pO = outT_tile;
            float* p0 = output;
            int d = 0;
            for (; d + 7 < value_dim; d += 8)
            {
                __m256 _r0 = _mm256_mul_ps(_mm256_loadu_ps(pO), _out_scale);
                __m256 _r1 = _mm256_mul_ps(_mm256_loadu_ps(pO + 8), _out_scale);
                __m256 _r2 = _mm256_mul_ps(_mm256_loadu_ps(pO + 16), _out_scale);
                __m256 _r3 = _mm256_mul_ps(_mm256_loadu_ps(pO + 24), _out_scale);
                __m256 _r4 = _mm256_mul_ps(_mm256_loadu_ps(pO + 32), _out_scale);
                __m256 _r5 = _mm256_mul_ps(_mm256_loadu_ps(pO + 40), _out_scale);
                __m256 _r6 = _mm256_mul_ps(_mm256_loadu_ps(pO + 48), _out_scale);
                __m256 _r7 = _mm256_mul_ps(_mm256_loadu_ps(pO + 56), _out_scale);
                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
                _mm256_storeu_ps(p0, _r0);
                _mm256_storeu_ps(p0 + output_cstep, _r1);
                _mm256_storeu_ps(p0 + output_cstep * 2, _r2);
                _mm256_storeu_ps(p0 + output_cstep * 3, _r3);
                _mm256_storeu_ps(p0 + output_cstep * 4, _r4);
                _mm256_storeu_ps(p0 + output_cstep * 5, _r5);
                _mm256_storeu_ps(p0 + output_cstep * 6, _r6);
                _mm256_storeu_ps(p0 + output_cstep * 7, _r7);
                p0 += 8;
                pO += 64;
            }
            for (; d < value_dim; d++)
            {
                __m256 _r = _mm256_mul_ps(_mm256_loadu_ps(pO), _out_scale);
                __m128 _r0 = _mm256_castps256_ps128(_r);
                __m128 _r1 = _mm256_extractf128_ps(_r, 1);
                *p0 = _mm_cvtss_f32(_r0);
                p0[output_cstep] = _mm_cvtss_f32(_mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(1, 1, 1, 1)));
                p0[output_cstep * 2] = _mm_cvtss_f32(_mm_movehl_ps(_r0, _r0));
                p0[output_cstep * 3] = _mm_cvtss_f32(_mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(3, 3, 3, 3)));
                p0[output_cstep * 4] = _mm_cvtss_f32(_r1);
                p0[output_cstep * 5] = _mm_cvtss_f32(_mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(1, 1, 1, 1)));
                p0[output_cstep * 6] = _mm_cvtss_f32(_mm_movehl_ps(_r1, _r1));
                p0[output_cstep * 7] = _mm_cvtss_f32(_mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(3, 3, 3, 3)));
                p0++;
                pO += 8;
            }
        }
    }
#endif // __AVX__
    for (; qq + 3 < max_qq; qq += 4)
    {
        const bool mask_per_head = !attn_mask_blob.empty() && attn_mask_blob.dims == 3 && attn_mask_blob.c > 1;
        const unsigned short* mask = 0;
        if (!attn_mask_blob.empty())
        {
            if (attn_mask_blob.dims == 3)
                mask = attn_mask_blob.channel(mask_per_head ? q0 + qq : 0);
            else
                mask = attn_mask_blob;
        }
        const size_t mask_cstep = mask_per_head ? attn_mask_blob.cstep : 0;

        const unsigned short* queryT_tile = queryT_ptr + (size_t)qq * head_dim;
        float* scoreT_tile = scoreT_ptr + (size_t)qq * block_n;
        float* outT_tile = outT_ptr + (size_t)qq * value_dim;
        __m128 _m = _mm_set1_ps(-FLT_MAX);
        __m128 _l = _mm_setzero_ps();
        memset(outT_tile, 0, (size_t)value_dim * 4 * sizeof(float));

        for (int n = 0; n < key_seqlen; n += block_n)
        {
            const int max_jj = std::min(key_seqlen - n, block_n);
            float* scoreptr = scoreT_tile;
            const unsigned short* pM = mask ? mask + n : 0;
            __m128 _block_max = _mm_set1_ps(-FLT_MAX);

            float* score_panel = scoreptr;
            for (int jj = 0; jj < max_jj; jj += NR)
            {
                const int max_nn = std::min(NR, max_jj - jj);
                const unsigned short* key_panel = (const unsigned short*)key_cache_head + (size_t)(n + jj) * head_dim;
                int j = 0;
                for (; j + 3 < max_nn; j += 4)
                {
                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();
                    const unsigned short* pA = queryT_tile;
                    int d = 0;
#if __AVX512BF16__
                    const int* pK_pair = (const int*)key_panel + j;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m128i _pA0 = _mm_loadu_si128((const __m128i*)pA);
                        __m128i _pB0 = _mm_loadu_si128((const __m128i*)pK_pair);
                        __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                        __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                        _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_pA0, (__m128bh)_pB0);
                        _sum1 = _mm_dpbf16_ps(_sum1, (__m128bh)_pA0, (__m128bh)_pB1);
                        _sum2 = _mm_dpbf16_ps(_sum2, (__m128bh)_pA1, (__m128bh)_pB0);
                        _sum3 = _mm_dpbf16_ps(_sum3, (__m128bh)_pA1, (__m128bh)_pB1);
                        pA += 8;
                        pK_pair += NR;
                    }
#endif // __AVX512BF16__
                    const unsigned short* pK = key_panel + (size_t)d * NR + j;
                    for (; d < head_dim; d++)
                    {
                        __m128 _pA0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pA));
                        __m128 _pB0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pK));
                        __m128 _pA1 = _mm_shuffle_ps(_pA0, _pA0, _MM_SHUFFLE(1, 0, 3, 2));
                        __m128 _pB1 = _mm_shuffle_ps(_pB0, _pB0, _MM_SHUFFLE(0, 3, 2, 1));
                        _sum0 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_pA0, _pB1, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_pA1, _pB0, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_pA1, _pB1, _sum3);
                        pA += 4;
                        pK += NR;
                    }

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
                    __m128 _scale = _mm_set1_ps(scale);
                    _sum0 = _mm_mul_ps(_sum0, _scale);
                    _sum1 = _mm_mul_ps(_sum1, _scale);
                    _sum2 = _mm_mul_ps(_sum2, _scale);
                    _sum3 = _mm_mul_ps(_sum3, _scale);
                    if (pM)
                    {
                        if (mask_per_head)
                        {
                            __m128i _m0 = _mm_loadl_epi64((const __m128i*)pM);
                            __m128i _m1 = _mm_loadl_epi64((const __m128i*)(pM + mask_cstep));
                            __m128i _m2 = _mm_loadl_epi64((const __m128i*)(pM + mask_cstep * 2));
                            __m128i _m3 = _mm_loadl_epi64((const __m128i*)(pM + mask_cstep * 3));
                            transpose8x4_epi16(_m0, _m1, _m2, _m3);
                            _sum0 = _mm_add_ps(_sum0, bfloat2float_sse(_m0));
                            _sum1 = _mm_add_ps(_sum1, bfloat2float_sse(_mm_srli_si128(_m0, 8)));
                            _sum2 = _mm_add_ps(_sum2, bfloat2float_sse(_m1));
                            _sum3 = _mm_add_ps(_sum3, bfloat2float_sse(_mm_srli_si128(_m1, 8)));
                        }
                        else
                        {
                            _sum0 = _mm_add_ps(_sum0, _mm_set1_ps(bfloat16_to_float32(pM[0])));
                            _sum1 = _mm_add_ps(_sum1, _mm_set1_ps(bfloat16_to_float32(pM[1])));
                            _sum2 = _mm_add_ps(_sum2, _mm_set1_ps(bfloat16_to_float32(pM[2])));
                            _sum3 = _mm_add_ps(_sum3, _mm_set1_ps(bfloat16_to_float32(pM[3])));
                        }
                        pM += 4;
                    }
                    _mm_storeu_ps(score_panel, _sum0);
                    _mm_storeu_ps(score_panel + 4, _sum1);
                    _mm_storeu_ps(score_panel + 8, _sum2);
                    _mm_storeu_ps(score_panel + 12, _sum3);
                    score_panel += 16;
                    _block_max = _mm_max_ps(_block_max, _mm_max_ps(_mm_max_ps(_sum0, _sum1), _mm_max_ps(_sum2, _sum3)));
                }
                for (; j + 1 < max_nn; j += 2)
                {
                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    const unsigned short* pA = queryT_tile;
                    int d = 0;
#if __AVX512BF16__
                    const int* pK_pair = (const int*)key_panel + j;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m128i _q = _mm_loadu_si128((const __m128i*)pA);
                        _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_q, (__m128bh)_mm_set1_epi32(pK_pair[0]));
                        _sum1 = _mm_dpbf16_ps(_sum1, (__m128bh)_q, (__m128bh)_mm_set1_epi32(pK_pair[1]));
                        pA += 8;
                        pK_pair += NR;
                    }
#endif // __AVX512BF16__
                    const unsigned short* pK = key_panel + (size_t)d * NR + j;
                    for (; d < head_dim; d++)
                    {
                        __m128 _q = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pA));
                        _sum0 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(pK[0])), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(pK[1])), _sum1);
                        pA += 4;
                        pK += NR;
                    }
                    __m128 _scale = _mm_set1_ps(scale);
                    _sum0 = _mm_mul_ps(_sum0, _scale);
                    _sum1 = _mm_mul_ps(_sum1, _scale);
                    if (pM)
                    {
                        _sum0 = _mm_add_ps(_sum0, _mm_set_ps(bfloat16_to_float32(pM[mask_cstep * 3]), bfloat16_to_float32(pM[mask_cstep * 2]), bfloat16_to_float32(pM[mask_cstep]), bfloat16_to_float32(pM[0])));
                        _sum1 = _mm_add_ps(_sum1, _mm_set_ps(bfloat16_to_float32(pM[mask_cstep * 3 + 1]), bfloat16_to_float32(pM[mask_cstep * 2 + 1]), bfloat16_to_float32(pM[mask_cstep + 1]), bfloat16_to_float32(pM[1])));
                        pM += 2;
                    }
                    _mm_storeu_ps(score_panel, _sum0);
                    _mm_storeu_ps(score_panel + 4, _sum1);
                    score_panel += 8;
                    _block_max = _mm_max_ps(_block_max, _mm_max_ps(_sum0, _sum1));
                }
                for (; j < max_nn; j++)
                {
                    __m128 _sum = _mm_setzero_ps();
                    const unsigned short* pA = queryT_tile;
                    int d = 0;
#if __AVX512BF16__
                    const int* pK_pair = (const int*)key_panel + j;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        _sum = _mm_dpbf16_ps(_sum, (__m128bh)_mm_loadu_si128((const __m128i*)pA), (__m128bh)_mm_set1_epi32(*pK_pair));
                        pA += 8;
                        pK_pair += NR;
                    }
#endif // __AVX512BF16__
                    const unsigned short* pK = key_panel + (size_t)d * NR + j;
                    for (; d < head_dim; d++)
                    {
                        _sum = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pA)), _mm_set1_ps(bfloat16_to_float32(*pK)), _sum);
                        pA += 4;
                        pK += NR;
                    }
                    _sum = _mm_mul_ps(_sum, _mm_set1_ps(scale));
                    if (pM)
                    {
                        _sum = _mm_add_ps(_sum, _mm_set_ps(bfloat16_to_float32(pM[mask_cstep * 3]), bfloat16_to_float32(pM[mask_cstep * 2]), bfloat16_to_float32(pM[mask_cstep]), bfloat16_to_float32(pM[0])));
                        pM++;
                    }
                    _mm_storeu_ps(score_panel, _sum);
                    score_panel += 4;
                    _block_max = _mm_max_ps(_block_max, _sum);
                }
            }

            __m128 _m_new = _mm_max_ps(_m, _block_max);
            __m128 _alpha_active = _mm_cmpneq_ps(_l, _mm_setzero_ps());
            __m128 _alpha = exp_ps(_mm_and_ps(_alpha_active, _mm_sub_ps(_m, _m_new)));
            _alpha = _mm_and_ps(_alpha, _alpha_active);

            __m128 _sum0 = _mm_setzero_ps();
            int j = 0;
#if defined(__x86_64__) || defined(_M_X64)
            __m128 _sum1 = _mm_setzero_ps();
            __m128 _sum2 = _mm_setzero_ps();
            __m128 _sum3 = _mm_setzero_ps();
            for (; j + 3 < max_jj; j += 4)
            {
                __m128 _p0 = exp_ps(_mm_sub_ps(_mm_loadu_ps(scoreptr), _m_new));
                __m128 _p1 = exp_ps(_mm_sub_ps(_mm_loadu_ps(scoreptr + 4), _m_new));
                __m128 _p2 = exp_ps(_mm_sub_ps(_mm_loadu_ps(scoreptr + 8), _m_new));
                __m128 _p3 = exp_ps(_mm_sub_ps(_mm_loadu_ps(scoreptr + 12), _m_new));
                _mm_storeu_ps(scoreptr, _p0);
                _mm_storeu_ps(scoreptr + 4, _p1);
                _mm_storeu_ps(scoreptr + 8, _p2);
                _mm_storeu_ps(scoreptr + 12, _p3);
                scoreptr += 16;
                _sum0 = _mm_add_ps(_sum0, _p0);
                _sum1 = _mm_add_ps(_sum1, _p1);
                _sum2 = _mm_add_ps(_sum2, _p2);
                _sum3 = _mm_add_ps(_sum3, _p3);
            }
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; j < max_jj; j++)
            {
                __m128 _p = exp_ps(_mm_sub_ps(_mm_loadu_ps(scoreptr), _m_new));
                _mm_storeu_ps(scoreptr, _p);
                scoreptr += 4;
                _sum0 = _mm_add_ps(_sum0, _p);
            }
            __m128 _sum = _sum0;
#if defined(__x86_64__) || defined(_M_X64)
            _sum = _mm_add_ps(_mm_add_ps(_sum, _sum1), _mm_add_ps(_sum2, _sum3));
#endif // defined(__x86_64__) || defined(_M_X64)
            _l = _mm_add_ps(_mm_mul_ps(_l, _alpha), _sum);
            _m = _m_new;

            float* outptr = outT_tile;
            for (int d = 0; d < value_dim;)
            {
                const int value_panel_width = sdpa_kvcache_value_panel_width(value_dim - d);
                int lane = 0;
                for (; lane + 3 < value_panel_width; lane += 4)
                {
                    __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(outptr), _alpha);
                    __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(outptr + 4), _alpha);
                    __m128 _out2 = _mm_mul_ps(_mm_loadu_ps(outptr + 8), _alpha);
                    __m128 _out3 = _mm_mul_ps(_mm_loadu_ps(outptr + 12), _alpha);
                    const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 4;
                        const unsigned short* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            __m128 _p = _mm_loadu_ps(pS);
                            __m128 _v = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pV));
                            _out0 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v, _v, _MM_SHUFFLE(0, 0, 0, 0)), _out0);
                            _out1 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v, _v, _MM_SHUFFLE(1, 1, 1, 1)), _out1);
                            _out2 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v, _v, _MM_SHUFFLE(2, 2, 2, 2)), _out2);
                            _out3 = _mm_comp_fmadd_ps(_p, _mm_shuffle_ps(_v, _v, _MM_SHUFFLE(3, 3, 3, 3)), _out3);
                            pS += 4;
                            pV += value_panel_width;
                        }
                        pV_panel += (size_t)NR * value_dim;
                    }
                    _mm_storeu_ps(outptr, _out0);
                    _mm_storeu_ps(outptr + 4, _out1);
                    _mm_storeu_ps(outptr + 8, _out2);
                    _mm_storeu_ps(outptr + 12, _out3);
                    outptr += 16;
                }
                for (; lane + 1 < value_panel_width; lane += 2)
                {
                    __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(outptr), _alpha);
                    __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(outptr + 4), _alpha);
                    const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 4;
                        const unsigned short* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            __m128 _p = _mm_loadu_ps(pS);
                            _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(bfloat16_to_float32(pV[0])), _out0);
                            _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(bfloat16_to_float32(pV[1])), _out1);
                            pS += 4;
                            pV += value_panel_width;
                        }
                        pV_panel += (size_t)NR * value_dim;
                    }
                    _mm_storeu_ps(outptr, _out0);
                    _mm_storeu_ps(outptr + 4, _out1);
                    outptr += 8;
                }
                for (; lane < value_panel_width; lane++)
                {
                    __m128 _out = _mm_mul_ps(_mm_loadu_ps(outptr), _alpha);
                    const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 4;
                        const unsigned short* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            _out = _mm_comp_fmadd_ps(_mm_loadu_ps(pS), _mm_set1_ps(bfloat16_to_float32(*pV)), _out);
                            pS += 4;
                            pV += value_panel_width;
                        }
                        pV_panel += (size_t)NR * value_dim;
                    }
                    _mm_storeu_ps(outptr, _out);
                    outptr += 4;
                }
                d += value_panel_width;
            }
        }

        {
            float* output = top_blob.channel(q0 + qq);
            const size_t output_cstep = top_blob.cstep;
            __m128 _nonzero = _mm_cmpneq_ps(_l, _mm_setzero_ps());
            __m128 _denom = _mm_or_ps(_mm_and_ps(_nonzero, _l), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
            __m128 _out_scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(1.f), _denom), _nonzero);
            const float* pO = outT_tile;
            float* p0 = output;
            int d = 0;
            for (; d + 3 < value_dim; d += 4)
            {
                __m128 _r0 = _mm_mul_ps(_mm_loadu_ps(pO), _out_scale);
                __m128 _r1 = _mm_mul_ps(_mm_loadu_ps(pO + 4), _out_scale);
                __m128 _r2 = _mm_mul_ps(_mm_loadu_ps(pO + 8), _out_scale);
                __m128 _r3 = _mm_mul_ps(_mm_loadu_ps(pO + 12), _out_scale);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_storeu_ps(p0, _r0);
                _mm_storeu_ps(p0 + output_cstep, _r1);
                _mm_storeu_ps(p0 + output_cstep * 2, _r2);
                _mm_storeu_ps(p0 + output_cstep * 3, _r3);
                p0 += 4;
                pO += 16;
            }
            for (; d < value_dim; d++)
            {
                __m128 _r = _mm_mul_ps(_mm_loadu_ps(pO), _out_scale);
                *p0 = _mm_cvtss_f32(_r);
                p0[output_cstep] = _mm_cvtss_f32(_mm_shuffle_ps(_r, _r, _MM_SHUFFLE(1, 1, 1, 1)));
                p0[output_cstep * 2] = _mm_cvtss_f32(_mm_movehl_ps(_r, _r));
                p0[output_cstep * 3] = _mm_cvtss_f32(_mm_shuffle_ps(_r, _r, _MM_SHUFFLE(3, 3, 3, 3)));
                p0++;
                pO += 4;
            }
        }
    }
#endif // __SSE2__
    for (; qq + 1 < max_qq; qq += 2)
    {
        const int q = q0 + qq;
        const unsigned short* query_ptr = query.channel(q);
        const size_t query_cstep = query.cstep * query.elempack;
        const bool mask_per_head = !attn_mask_blob.empty() && attn_mask_blob.dims == 3 && attn_mask_blob.c > 1;
        const unsigned short* mask = 0;
        if (!attn_mask_blob.empty())
        {
            if (attn_mask_blob.dims == 3)
                mask = attn_mask_blob.channel(mask_per_head ? q : 0);
            else
                mask = attn_mask_blob;
        }
        const size_t mask_cstep = mask_per_head ? attn_mask_blob.cstep : 0;

        float* score0 = scoreT_ptr + (size_t)qq * block_n;
        float* score1 = score0 + block_n;
        float* out0 = outT_ptr + (size_t)qq * value_dim;
        float* out1 = out0 + value_dim;
        memset(out0, 0, (size_t)value_dim * 2 * sizeof(float));
        float m0 = -FLT_MAX;
        float m1 = -FLT_MAX;
        float l0 = 0.f;
        float l1 = 0.f;

        for (int n = 0; n < key_seqlen; n += block_n)
        {
            const int max_jj = std::min(key_seqlen - n, block_n);
            float block_max0 = -FLT_MAX;
            float block_max1 = -FLT_MAX;

            for (int jj = 0; jj < max_jj; jj += NR)
            {
                const int max_nn = std::min(NR, max_jj - jj);
                const unsigned short* key_panel = (const unsigned short*)key_cache_head + (size_t)(n + jj) * head_dim;
                int k = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; k + 15 < max_nn; k += 16)
                {
                    __m512 _sum0 = _mm512_setzero_ps();
                    __m512 _sum1 = _mm512_setzero_ps();
                    const unsigned short* pA = query_ptr;
                    const unsigned short* pK = key_panel + k;
                    int d = 0;
#if __AVX512BF16__
                    pK = key_panel + k * 2;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m512i _k = _mm512_loadu_si512((const __m512i*)pK);
                        _sum0 = _mm512_dpbf16_ps(_sum0, (__m512bh)_mm512_set1_epi32(((const int*)pA)[0]), (__m512bh)_k);
                        _sum1 = _mm512_dpbf16_ps(_sum1, (__m512bh)_mm512_set1_epi32(((const int*)(pA + query_cstep))[0]), (__m512bh)_k);
                        pA += 2;
                        pK += NR * 2;
                    }
                    pK = key_panel + (size_t)d * NR + k;
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m512 _k = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pK));
                        _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(bfloat16_to_float32(pA[0])), _k, _sum0);
                        _sum1 = _mm512_fmadd_ps(_mm512_set1_ps(bfloat16_to_float32(pA[query_cstep])), _k, _sum1);
                        pA++;
                        pK += NR;
                    }
                    _sum0 = _mm512_mul_ps(_sum0, _mm512_set1_ps(scale));
                    _sum1 = _mm512_mul_ps(_sum1, _mm512_set1_ps(scale));
                    if (mask)
                    {
                        const unsigned short* pM = mask + n + jj + k;
                        _sum0 = _mm512_add_ps(_sum0, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pM)));
                        _sum1 = _mm512_add_ps(_sum1, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(pM + mask_cstep))));
                    }
                    _mm512_storeu_ps(score0 + jj + k, _sum0);
                    _mm512_storeu_ps(score1 + jj + k, _sum1);
                    block_max0 = std::max(block_max0, _mm512_comp_reduce_max_ps(_sum0));
                    block_max1 = std::max(block_max1, _mm512_comp_reduce_max_ps(_sum1));
                }
#endif // __AVX512F__
                for (; k + 7 < max_nn; k += 8)
                {
                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    const unsigned short* pA = query_ptr;
                    const unsigned short* pK = key_panel + k;
                    int d = 0;
#if __AVX512BF16__
                    pK = key_panel + k * 2;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m256i _k = _mm256_loadu_si256((const __m256i*)pK);
                        _sum0 = _mm256_dpbf16_ps(_sum0, (__m256bh)_mm256_set1_epi32(((const int*)pA)[0]), (__m256bh)_k);
                        _sum1 = _mm256_dpbf16_ps(_sum1, (__m256bh)_mm256_set1_epi32(((const int*)(pA + query_cstep))[0]), (__m256bh)_k);
                        pA += 2;
                        pK += NR * 2;
                    }
                    pK = key_panel + (size_t)d * NR + k;
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m256 _k = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pK));
                        _sum0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(bfloat16_to_float32(pA[0])), _k, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(bfloat16_to_float32(pA[query_cstep])), _k, _sum1);
                        pA++;
                        pK += NR;
                    }
                    _sum0 = _mm256_mul_ps(_sum0, _mm256_set1_ps(scale));
                    _sum1 = _mm256_mul_ps(_sum1, _mm256_set1_ps(scale));
                    if (mask)
                    {
                        const unsigned short* pM = mask + n + jj + k;
                        _sum0 = _mm256_add_ps(_sum0, bfloat2float_avx(_mm_loadu_si128((const __m128i*)pM)));
                        _sum1 = _mm256_add_ps(_sum1, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(pM + mask_cstep))));
                    }
                    _mm256_storeu_ps(score0 + jj + k, _sum0);
                    _mm256_storeu_ps(score1 + jj + k, _sum1);
                    block_max0 = std::max(block_max0, _mm256_reduce_max_ps(_sum0));
                    block_max1 = std::max(block_max1, _mm256_reduce_max_ps(_sum1));
                }
#endif // __AVX__
                for (; k + 3 < max_nn; k += 4)
                {
                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    const unsigned short* pA = query_ptr;
                    const unsigned short* pK = key_panel + k;
                    int d = 0;
#if __AVX512BF16__
                    pK = key_panel + k * 2;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m128i _k = _mm_loadu_si128((const __m128i*)pK);
                        _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_mm_set1_epi32(((const int*)pA)[0]), (__m128bh)_k);
                        _sum1 = _mm_dpbf16_ps(_sum1, (__m128bh)_mm_set1_epi32(((const int*)(pA + query_cstep))[0]), (__m128bh)_k);
                        pA += 2;
                        pK += NR * 2;
                    }
                    pK = key_panel + (size_t)d * NR + k;
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        __m128 _k = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pK));
                        _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(bfloat16_to_float32(pA[0])), _k, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_mm_set1_ps(bfloat16_to_float32(pA[query_cstep])), _k, _sum1);
                        pA++;
                        pK += NR;
                    }
                    _sum0 = _mm_mul_ps(_sum0, _mm_set1_ps(scale));
                    _sum1 = _mm_mul_ps(_sum1, _mm_set1_ps(scale));
                    if (mask)
                    {
                        const unsigned short* pM = mask + n + jj + k;
                        _sum0 = _mm_add_ps(_sum0, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pM)));
                        _sum1 = _mm_add_ps(_sum1, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pM + mask_cstep))));
                    }
                    _mm_storeu_ps(score0 + jj + k, _sum0);
                    _mm_storeu_ps(score1 + jj + k, _sum1);
                    block_max0 = std::max(block_max0, _mm_reduce_max_ps(_sum0));
                    block_max1 = std::max(block_max1, _mm_reduce_max_ps(_sum1));
                }
#endif // __SSE2__
                for (; k + 1 < max_nn; k += 2)
                {
                    const unsigned short* pK = key_panel + k;
                    float sum00 = 0.f;
                    float sum01 = 0.f;
                    float sum10 = 0.f;
                    float sum11 = 0.f;
                    const unsigned short* pA = query_ptr;
                    int d = 0;
#if __AVX512BF16__
                    pK = key_panel + k * 2;
                    __m128 _sum = _mm_setzero_ps();
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m128i _q0 = _mm_set1_epi32(((const int*)pA)[0]);
                        __m128i _q1 = _mm_set1_epi32(((const int*)(pA + query_cstep))[0]);
                        __m128i _k0 = _mm_set1_epi32(((const int*)pK)[0]);
                        __m128i _k1 = _mm_set1_epi32(((const int*)pK)[1]);
                        __m128i _q = _mm_unpacklo_epi64(_q0, _q1);
                        __m128i _k = _mm_unpacklo_epi32(_k0, _k1);
                        _sum = _mm_dpbf16_ps(_sum, (__m128bh)_q, (__m128bh)_k);
                        pA += 2;
                        pK += NR * 2;
                    }
                    sum00 = _mm_cvtss_f32(_sum);
                    sum01 = _mm_cvtss_f32(_mm_shuffle_ps(_sum, _sum, _MM_SHUFFLE(1, 1, 1, 1)));
                    sum10 = _mm_cvtss_f32(_mm_movehl_ps(_sum, _sum));
                    sum11 = _mm_cvtss_f32(_mm_shuffle_ps(_sum, _sum, _MM_SHUFFLE(3, 3, 3, 3)));
                    pK = key_panel + (size_t)d * NR + k;
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        const float k0 = bfloat16_to_float32(pK[0]);
                        const float k1 = bfloat16_to_float32(pK[1]);
                        const float qv0 = bfloat16_to_float32(pA[0]);
                        const float qv1 = bfloat16_to_float32(pA[query_cstep]);
                        sum00 += qv0 * k0;
                        sum01 += qv0 * k1;
                        sum10 += qv1 * k0;
                        sum11 += qv1 * k1;
                        pA++;
                        pK += NR;
                    }
                    sum00 = sum00 * scale + (mask ? bfloat16_to_float32(mask[n + jj + k]) : 0.f);
                    sum01 = sum01 * scale + (mask ? bfloat16_to_float32(mask[n + jj + k + 1]) : 0.f);
                    sum10 = sum10 * scale + (mask ? bfloat16_to_float32(mask[mask_cstep + n + jj + k]) : 0.f);
                    sum11 = sum11 * scale + (mask ? bfloat16_to_float32(mask[mask_cstep + n + jj + k + 1]) : 0.f);
                    score0[jj + k] = sum00;
                    score0[jj + k + 1] = sum01;
                    score1[jj + k] = sum10;
                    score1[jj + k + 1] = sum11;
                    block_max0 = std::max(block_max0, std::max(sum00, sum01));
                    block_max1 = std::max(block_max1, std::max(sum10, sum11));
                }
                for (; k < max_nn; k++)
                {
                    const unsigned short* pK = key_panel + k;
                    const unsigned short* pA = query_ptr;
                    float sum0 = 0.f;
                    float sum1 = 0.f;
                    int d = 0;
#if __AVX512BF16__
                    pK = key_panel + k * 2;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        sum0 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pK[0]);
                        sum0 += bfloat16_to_float32(pA[1]) * bfloat16_to_float32(pK[1]);
                        sum1 += bfloat16_to_float32(pA[query_cstep]) * bfloat16_to_float32(pK[0]);
                        sum1 += bfloat16_to_float32(pA[query_cstep + 1]) * bfloat16_to_float32(pK[1]);
                        pA += 2;
                        pK += NR * 2;
                    }
                    pK = key_panel + (size_t)d * NR + k;
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        const float v = bfloat16_to_float32(*pK);
                        sum0 += bfloat16_to_float32(pA[0]) * v;
                        sum1 += bfloat16_to_float32(pA[query_cstep]) * v;
                        pA++;
                        pK += NR;
                    }
                    score0[jj + k] = sum0 * scale + (mask ? bfloat16_to_float32(mask[n + jj + k]) : 0.f);
                    score1[jj + k] = sum1 * scale + (mask ? bfloat16_to_float32(mask[mask_cstep + n + jj + k]) : 0.f);
                    block_max0 = std::max(block_max0, score0[jj + k]);
                    block_max1 = std::max(block_max1, score1[jj + k]);
                }
            }

            const float m_new0 = std::max(m0, block_max0);
            const float m_new1 = std::max(m1, block_max1);
            const float alpha0 = l0 == 0.f ? 0.f : expf(m0 - m_new0);
            const float alpha1 = l1 == 0.f ? 0.f : expf(m1 - m_new1);

            float sum0 = 0.f;
            float sum1 = 0.f;
            int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _sum0_avx512 = _mm512_setzero_ps();
            __m512 _sum1_avx512 = _mm512_setzero_ps();
            for (; j + 15 < max_jj; j += 16)
            {
                __m512 _p0 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(score0 + j), _mm512_set1_ps(m_new0)));
                __m512 _p1 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(score1 + j), _mm512_set1_ps(m_new1)));
                _mm512_storeu_ps(score0 + j, _p0);
                _mm512_storeu_ps(score1 + j, _p1);
                _sum0_avx512 = _mm512_add_ps(_sum0_avx512, _p0);
                _sum1_avx512 = _mm512_add_ps(_sum1_avx512, _p1);
            }
            sum0 += _mm512_comp_reduce_add_ps(_sum0_avx512);
            sum1 += _mm512_comp_reduce_add_ps(_sum1_avx512);
#endif // __AVX512F__
            __m256 _sum0_avx = _mm256_setzero_ps();
            __m256 _sum1_avx = _mm256_setzero_ps();
            for (; j + 7 < max_jj; j += 8)
            {
                __m256 _p0 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(score0 + j), _mm256_set1_ps(m_new0)));
                __m256 _p1 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(score1 + j), _mm256_set1_ps(m_new1)));
                _mm256_storeu_ps(score0 + j, _p0);
                _mm256_storeu_ps(score1 + j, _p1);
                _sum0_avx = _mm256_add_ps(_sum0_avx, _p0);
                _sum1_avx = _mm256_add_ps(_sum1_avx, _p1);
            }
            sum0 += _mm256_reduce_add_ps(_sum0_avx);
            sum1 += _mm256_reduce_add_ps(_sum1_avx);
#endif // __AVX__
            __m128 _sum0_sse = _mm_setzero_ps();
            __m128 _sum1_sse = _mm_setzero_ps();
            for (; j + 3 < max_jj; j += 4)
            {
                __m128 _p0 = exp_ps(_mm_sub_ps(_mm_loadu_ps(score0 + j), _mm_set1_ps(m_new0)));
                __m128 _p1 = exp_ps(_mm_sub_ps(_mm_loadu_ps(score1 + j), _mm_set1_ps(m_new1)));
                _mm_storeu_ps(score0 + j, _p0);
                _mm_storeu_ps(score1 + j, _p1);
                _sum0_sse = _mm_add_ps(_sum0_sse, _p0);
                _sum1_sse = _mm_add_ps(_sum1_sse, _p1);
            }
            sum0 += _mm_reduce_add_ps(_sum0_sse);
            sum1 += _mm_reduce_add_ps(_sum1_sse);
#endif // __SSE2__
            for (; j < max_jj; j++)
            {
                score0[j] = expf(score0[j] - m_new0);
                score1[j] = expf(score1[j] - m_new1);
                sum0 += score0[j];
                sum1 += score1[j];
            }
            l0 = l0 * alpha0 + sum0;
            l1 = l1 * alpha1 + sum1;
            m0 = m_new0;
            m1 = m_new1;

            int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; d + 15 < value_dim; d += 16)
            {
                __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(out0 + d), _mm512_set1_ps(alpha0));
                __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(out1 + d), _mm512_set1_ps(alpha1));
                const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pS = score0 + jj;
                    const unsigned short* pV = pV_panel;
                    for (int j = 0; j < max_nn; j++)
                    {
                        __m512 _v = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pV));
                        _out0 = _mm512_fmadd_ps(_v, _mm512_set1_ps(pS[0]), _out0);
                        _out1 = _mm512_fmadd_ps(_v, _mm512_set1_ps(pS[block_n]), _out1);
                        pS++;
                        pV += 16;
                    }
                    pV_panel += (size_t)NR * value_dim;
                }
                _mm512_storeu_ps(out0 + d, _out0);
                _mm512_storeu_ps(out1 + d, _out1);
            }
#endif // __AVX512F__
            for (; d + 7 < value_dim; d += 8)
            {
                __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(out0 + d), _mm256_set1_ps(alpha0));
                __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(out1 + d), _mm256_set1_ps(alpha1));
                const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pS = score0 + jj;
                    const unsigned short* pV = pV_panel;
                    for (int j = 0; j < max_nn; j++)
                    {
                        __m256 _v = bfloat2float_avx(_mm_loadu_si128((const __m128i*)pV));
                        _out0 = _mm256_comp_fmadd_ps(_v, _mm256_set1_ps(pS[0]), _out0);
                        _out1 = _mm256_comp_fmadd_ps(_v, _mm256_set1_ps(pS[block_n]), _out1);
                        pS++;
                        pV += 8;
                    }
                    pV_panel += (size_t)NR * value_dim;
                }
                _mm256_storeu_ps(out0 + d, _out0);
                _mm256_storeu_ps(out1 + d, _out1);
            }
#endif // __AVX__
            for (; d + 3 < value_dim; d += 4)
            {
                __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(out0 + d), _mm_set1_ps(alpha0));
                __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(out1 + d), _mm_set1_ps(alpha1));
                const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pS = score0 + jj;
                    const unsigned short* pV = pV_panel;
                    for (int j = 0; j < max_nn; j++)
                    {
                        __m128 _v = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pV));
                        _out0 = _mm_comp_fmadd_ps(_v, _mm_set1_ps(pS[0]), _out0);
                        _out1 = _mm_comp_fmadd_ps(_v, _mm_set1_ps(pS[block_n]), _out1);
                        pS++;
                        pV += 4;
                    }
                    pV_panel += (size_t)NR * value_dim;
                }
                _mm_storeu_ps(out0 + d, _out0);
                _mm_storeu_ps(out1 + d, _out1);
            }
#endif // __SSE2__
            for (; d + 1 < value_dim; d += 2)
            {
                float sum00 = out0[d] * alpha0;
                float sum01 = out0[d + 1] * alpha0;
                float sum10 = out1[d] * alpha1;
                float sum11 = out1[d + 1] * alpha1;
                const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pS = score0 + jj;
                    const unsigned short* pV = pV_panel;
                    for (int j = 0; j < max_nn; j++)
                    {
                        const float v0 = bfloat16_to_float32(pV[0]);
                        const float v1 = bfloat16_to_float32(pV[1]);
                        sum00 += pS[0] * v0;
                        sum01 += pS[0] * v1;
                        sum10 += pS[block_n] * v0;
                        sum11 += pS[block_n] * v1;
                        pS++;
                        pV += 2;
                    }
                    pV_panel += (size_t)NR * value_dim;
                }
                out0[d] = sum00;
                out0[d + 1] = sum01;
                out1[d] = sum10;
                out1[d + 1] = sum11;
            }
            for (; d < value_dim; d++)
            {
                float sum0 = out0[d] * alpha0;
                float sum1 = out1[d] * alpha1;
                const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pS = score0 + jj;
                    const unsigned short* pV = pV_panel;
                    for (int j = 0; j < max_nn; j++)
                    {
                        const float v = bfloat16_to_float32(*pV++);
                        sum0 += pS[0] * v;
                        sum1 += pS[block_n] * v;
                        pS++;
                    }
                    pV_panel += (size_t)NR * value_dim;
                }
                out0[d] = sum0;
                out1[d] = sum1;
            }
        }

        {
            float* output = top_blob.channel(q);
            const size_t output_cstep = top_blob.cstep;
            float* p0 = output;
            const float inv_sum0 = l0 == 0.f ? 0.f : 1.f / l0;
            const float inv_sum1 = l1 == 0.f ? 0.f : 1.f / l1;
            int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; d + 15 < value_dim; d += 16)
            {
                _mm512_storeu_ps(p0, _mm512_mul_ps(_mm512_loadu_ps(out0 + d), _mm512_set1_ps(inv_sum0)));
                _mm512_storeu_ps(p0 + output_cstep, _mm512_mul_ps(_mm512_loadu_ps(out1 + d), _mm512_set1_ps(inv_sum1)));
                p0 += 16;
            }
#endif // __AVX512F__
            for (; d + 7 < value_dim; d += 8)
            {
                _mm256_storeu_ps(p0, _mm256_mul_ps(_mm256_loadu_ps(out0 + d), _mm256_set1_ps(inv_sum0)));
                _mm256_storeu_ps(p0 + output_cstep, _mm256_mul_ps(_mm256_loadu_ps(out1 + d), _mm256_set1_ps(inv_sum1)));
                p0 += 8;
            }
#endif // __AVX__
            for (; d + 3 < value_dim; d += 4)
            {
                _mm_storeu_ps(p0, _mm_mul_ps(_mm_loadu_ps(out0 + d), _mm_set1_ps(inv_sum0)));
                _mm_storeu_ps(p0 + output_cstep, _mm_mul_ps(_mm_loadu_ps(out1 + d), _mm_set1_ps(inv_sum1)));
                p0 += 4;
            }
#endif // __SSE2__
            for (; d < value_dim; d++)
            {
                *p0 = out0[d] * inv_sum0;
                p0[output_cstep] = out1[d] * inv_sum1;
                p0++;
            }
        }
    }

    for (; qq < max_qq; qq++)
    {
        const int q = q0 + qq;
        const unsigned short* query_ptr = query.channel(q);
        const unsigned short* mask = 0;
        if (!attn_mask_blob.empty())
        {
            if (attn_mask_blob.dims == 3)
                mask = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q : 0);
            else
                mask = attn_mask_blob;
        }

        float* score = scoreT_ptr + (size_t)qq * block_n;
        float* out = outT_ptr + (size_t)qq * value_dim;
        memset(out, 0, (size_t)value_dim * sizeof(float));
        float m = -FLT_MAX;
        float l = 0.f;

        for (int n = 0; n < key_seqlen; n += block_n)
        {
            const int max_jj = std::min(key_seqlen - n, block_n);
            float block_max = -FLT_MAX;

            for (int jj = 0; jj < max_jj; jj += NR)
            {
                const int max_nn = std::min(NR, max_jj - jj);
                const unsigned short* key_panel = (const unsigned short*)key_cache_head + (size_t)(n + jj) * head_dim;
                int k = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; k + 15 < max_nn; k += 16)
                {
                    const unsigned short* pK = key_panel + k;
                    const unsigned short* pA = query_ptr;
                    __m512 _sum = _mm512_setzero_ps();
                    int d = 0;
#if __AVX512BF16__
                    pK = key_panel + k * 2;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m512i _k = _mm512_loadu_si512((const __m512i*)pK);
                        _sum = _mm512_dpbf16_ps(_sum, (__m512bh)_k, (__m512bh)_mm512_set1_epi32(((const int*)pA)[0]));
                        pA += 2;
                        pK += NR * 2;
                    }
                    pK = key_panel + (size_t)d * NR + k;
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        _sum = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pK)), _mm512_set1_ps(bfloat16_to_float32(*pA++)), _sum);
                        pK += NR;
                    }
                    _sum = _mm512_mul_ps(_sum, _mm512_set1_ps(scale));
                    if (mask)
                        _sum = _mm512_add_ps(_sum, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(mask + n + jj + k))));
                    _mm512_storeu_ps(score + jj + k, _sum);
                    block_max = std::max(block_max, _mm512_comp_reduce_max_ps(_sum));
                }
#endif // __AVX512F__
                for (; k + 7 < max_nn; k += 8)
                {
                    const unsigned short* pK = key_panel + k;
                    const unsigned short* pA = query_ptr;
                    __m256 _sum = _mm256_setzero_ps();
                    int d = 0;
#if __AVX512BF16__
                    pK = key_panel + k * 2;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m256i _k = _mm256_loadu_si256((const __m256i*)pK);
                        _sum = _mm256_dpbf16_ps(_sum, (__m256bh)_k, (__m256bh)_mm256_set1_epi32(((const int*)pA)[0]));
                        pA += 2;
                        pK += NR * 2;
                    }
                    pK = key_panel + (size_t)d * NR + k;
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        _sum = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)pK)), _mm256_set1_ps(bfloat16_to_float32(*pA++)), _sum);
                        pK += NR;
                    }
                    _sum = _mm256_mul_ps(_sum, _mm256_set1_ps(scale));
                    if (mask)
                        _sum = _mm256_add_ps(_sum, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(mask + n + jj + k))));
                    _mm256_storeu_ps(score + jj + k, _sum);
                    block_max = std::max(block_max, _mm256_reduce_max_ps(_sum));
                }
#endif // __AVX__
                for (; k + 3 < max_nn; k += 4)
                {
                    const unsigned short* pK = key_panel + k;
                    const unsigned short* pA = query_ptr;
                    __m128 _sum = _mm_setzero_ps();
                    int d = 0;
#if __AVX512BF16__
                    pK = key_panel + k * 2;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m128i _k = _mm_loadu_si128((const __m128i*)pK);
                        _sum = _mm_dpbf16_ps(_sum, (__m128bh)_k, (__m128bh)_mm_set1_epi32(((const int*)pA)[0]));
                        pA += 2;
                        pK += NR * 2;
                    }
                    pK = key_panel + (size_t)d * NR + k;
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        _sum = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pK)), _mm_set1_ps(bfloat16_to_float32(*pA++)), _sum);
                        pK += NR;
                    }
                    _sum = _mm_mul_ps(_sum, _mm_set1_ps(scale));
                    if (mask)
                        _sum = _mm_add_ps(_sum, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(mask + n + jj + k))));
                    _mm_storeu_ps(score + jj + k, _sum);
                    block_max = std::max(block_max, _mm_reduce_max_ps(_sum));
                }
#endif // __SSE2__
                for (; k + 1 < max_nn; k += 2)
                {
                    const unsigned short* pK = key_panel + k;
                    const unsigned short* pA = query_ptr;
                    float sum0 = 0.f;
                    float sum1 = 0.f;
                    int d = 0;
#if __AVX512BF16__
                    pK = key_panel + k * 2;
                    __m128 _sum = _mm_setzero_ps();
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m128i _q = _mm_set1_epi32(((const int*)pA)[0]);
                        __m128i _k0 = _mm_set1_epi32(((const int*)pK)[0]);
                        __m128i _k1 = _mm_set1_epi32(((const int*)pK)[1]);
                        __m128i _k = _mm_unpacklo_epi32(_k0, _k1);
                        _sum = _mm_dpbf16_ps(_sum, (__m128bh)_q, (__m128bh)_k);
                        pA += 2;
                        pK += NR * 2;
                    }
                    sum0 = _mm_cvtss_f32(_sum);
                    sum1 = _mm_cvtss_f32(_mm_shuffle_ps(_sum, _sum, _MM_SHUFFLE(1, 1, 1, 1)));
                    pK = key_panel + (size_t)d * NR + k;
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        const float qv = bfloat16_to_float32(*pA++);
                        sum0 += qv * bfloat16_to_float32(pK[0]);
                        sum1 += qv * bfloat16_to_float32(pK[1]);
                        pK += NR;
                    }
                    score[jj + k] = sum0 * scale + (mask ? bfloat16_to_float32(mask[n + jj + k]) : 0.f);
                    score[jj + k + 1] = sum1 * scale + (mask ? bfloat16_to_float32(mask[n + jj + k + 1]) : 0.f);
                    block_max = std::max(block_max, std::max(score[jj + k], score[jj + k + 1]));
                }
                for (; k < max_nn; k++)
                {
                    const unsigned short* pK = key_panel + k;
                    const unsigned short* pA = query_ptr;
                    float sum0 = 0.f;
                    int d = 0;
#if __AVX512BF16__
                    pK = key_panel + k * 2;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        sum0 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pK[0]);
                        sum0 += bfloat16_to_float32(pA[1]) * bfloat16_to_float32(pK[1]);
                        pA += 2;
                        pK += NR * 2;
                    }
                    pK = key_panel + (size_t)d * NR + k;
#endif // __AVX512BF16__
                    for (; d < head_dim; d++)
                    {
                        sum0 += bfloat16_to_float32(*pA++) * bfloat16_to_float32(*pK);
                        pK += NR;
                    }
                    score[jj + k] = sum0 * scale + (mask ? bfloat16_to_float32(mask[n + jj + k]) : 0.f);
                    block_max = std::max(block_max, score[jj + k]);
                }
            }

            const float m_new = std::max(m, block_max);
            const float alpha = l == 0.f ? 0.f : expf(m - m_new);

            float sum = 0.f;
            int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _sum_avx512 = _mm512_setzero_ps();
            __m512 _max_avx512 = _mm512_set1_ps(m_new);
            for (; j + 15 < max_jj; j += 16)
            {
                __m512 _p = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(score + j), _max_avx512));
                _mm512_storeu_ps(score + j, _p);
                _sum_avx512 = _mm512_add_ps(_sum_avx512, _p);
            }
            sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
            __m256 _sum_avx = _mm256_setzero_ps();
            __m256 _max_avx = _mm256_set1_ps(m_new);
            for (; j + 7 < max_jj; j += 8)
            {
                __m256 _p = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(score + j), _max_avx));
                _mm256_storeu_ps(score + j, _p);
                _sum_avx = _mm256_add_ps(_sum_avx, _p);
            }
            sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
            __m128 _sum_sse = _mm_setzero_ps();
            __m128 _max_sse = _mm_set1_ps(m_new);
            for (; j + 3 < max_jj; j += 4)
            {
                __m128 _p = exp_ps(_mm_sub_ps(_mm_loadu_ps(score + j), _max_sse));
                _mm_storeu_ps(score + j, _p);
                _sum_sse = _mm_add_ps(_sum_sse, _p);
            }
            sum += _mm_reduce_add_ps(_sum_sse);
#endif // __SSE2__
            for (; j < max_jj; j++)
            {
                score[j] = expf(score[j] - m_new);
                sum += score[j];
            }
            l = l * alpha + sum;
            m = m_new;

            int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; d + 15 < value_dim; d += 16)
            {
                __m512 _out = _mm512_mul_ps(_mm512_loadu_ps(out + d), _mm512_set1_ps(alpha));
                const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pS = score + jj;
                    const unsigned short* pV = pV_panel;
                    for (int j = 0; j < max_nn; j++)
                    {
                        _out = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)pV)), _mm512_set1_ps(*pS++), _out);
                        pV += 16;
                    }
                    pV_panel += (size_t)NR * value_dim;
                }
                _mm512_storeu_ps(out + d, _out);
            }
#endif // __AVX512F__
            for (; d + 7 < value_dim; d += 8)
            {
                __m256 _out = _mm256_mul_ps(_mm256_loadu_ps(out + d), _mm256_set1_ps(alpha));
                const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pS = score + jj;
                    const unsigned short* pV = pV_panel;
                    for (int j = 0; j < max_nn; j++)
                    {
                        _out = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)pV)), _mm256_set1_ps(*pS++), _out);
                        pV += 8;
                    }
                    pV_panel += (size_t)NR * value_dim;
                }
                _mm256_storeu_ps(out + d, _out);
            }
#endif // __AVX__
            for (; d + 3 < value_dim; d += 4)
            {
                __m128 _out = _mm_mul_ps(_mm_loadu_ps(out + d), _mm_set1_ps(alpha));
                const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pS = score + jj;
                    const unsigned short* pV = pV_panel;
                    for (int j = 0; j < max_nn; j++)
                    {
                        _out = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pV)), _mm_set1_ps(*pS++), _out);
                        pV += 4;
                    }
                    pV_panel += (size_t)NR * value_dim;
                }
                _mm_storeu_ps(out + d, _out);
            }
#endif // __SSE2__
            for (; d + 1 < value_dim; d += 2)
            {
                float sum0 = out[d] * alpha;
                float sum1 = out[d + 1] * alpha;
                const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pS = score + jj;
                    const unsigned short* pV = pV_panel;
                    for (int j = 0; j < max_nn; j++)
                    {
                        sum0 += *pS * bfloat16_to_float32(pV[0]);
                        sum1 += *pS++ * bfloat16_to_float32(pV[1]);
                        pV += 2;
                    }
                    pV_panel += (size_t)NR * value_dim;
                }
                out[d] = sum0;
                out[d + 1] = sum1;
            }
            for (; d < value_dim; d++)
            {
                float sum0 = out[d] * alpha;
                const unsigned short* pV_panel = (const unsigned short*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pS = score + jj;
                    const unsigned short* pV = pV_panel;
                    for (int j = 0; j < max_nn; j++)
                        sum0 += *pS++ * bfloat16_to_float32(*pV++);
                    pV_panel += (size_t)NR * value_dim;
                }
                out[d] = sum0;
            }
        }

        {
            float* output = top_blob.channel(q);
            const float inv_sum = l == 0.f ? 0.f : 1.f / l;
            int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _inv_sum_avx512 = _mm512_set1_ps(inv_sum);
            for (; d + 15 < value_dim; d += 16)
                _mm512_storeu_ps(output + d, _mm512_mul_ps(_mm512_loadu_ps(out + d), _inv_sum_avx512));
#endif // __AVX512F__
            __m256 _inv_sum_avx = _mm256_set1_ps(inv_sum);
            for (; d + 7 < value_dim; d += 8)
                _mm256_storeu_ps(output + d, _mm256_mul_ps(_mm256_loadu_ps(out + d), _inv_sum_avx));
#endif // __AVX__
            __m128 _inv_sum = _mm_set1_ps(inv_sum);
            for (; d + 3 < value_dim; d += 4)
                _mm_storeu_ps(output + d, _mm_mul_ps(_mm_loadu_ps(out + d), _inv_sum));
#endif // __SSE2__
            for (; d < value_dim; d++)
                output[d] = out[d] * inv_sum;
        }
    }
}

static int sdpa_decode_kvcache_bf16s(const Mat& query, const Mat& key_cache, const Mat& value_cache, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
{
    const int head_dim = query.w;
    const int value_dim = value_cache.w;
    const int key_seqlen = key_cache.h;
    const int num_query_heads = query.c;
    const int num_kv_heads = key_cache.c;
    const int num_query_heads_per_kv_head = num_query_heads / num_kv_heads;
    const int block_q = sdpa_decode_get_optimal_tile_q(num_query_heads_per_kv_head);
#if __AVX512F__
    const int NR = 16;
#elif __AVX__
    const int NR = 8;
#elif __SSE2__
    const int NR = 4;
#else
    const int NR = 2;
#endif
    const int num_qblocks = (num_query_heads_per_kv_head + block_q - 1) / block_q;
    const int num_tasks = num_kv_heads * num_qblocks;
    const int nT = std::min(std::max(opt.num_threads, 1), num_tasks);
    int block_n = sdpa_decode_get_optimal_tile_n(head_dim, value_dim, key_seqlen, 2, 2, 2, attn_mask_blob.empty() ? 0 : 2, block_q);
    block_n = std::max(NR, (block_n + NR - 1) / NR * NR);

    const bool pack_query = block_q >= 4;
    const size_t score_workspace_size = (size_t)block_q * block_n * sizeof(float);
    const size_t output_workspace_size = (size_t)block_q * value_dim * sizeof(float);
    const size_t query_workspace_size = pack_query ? (size_t)block_q * head_dim * sizeof(unsigned short) : 0;
    const size_t workspace_size = alignSize(score_workspace_size + output_workspace_size + query_workspace_size, 64);
    Mat workspace((int)(workspace_size / sizeof(float)), 1, nT, 4u, opt.workspace_allocator);
    if (workspace.empty())
        return -100;

    #pragma omp parallel for num_threads(nT)
    for (int task_id = 0; task_id < num_tasks; task_id++)
    {
        const int g = task_id / num_qblocks;
        const int qblock_id = task_id % num_qblocks;
        const int q0 = g * num_query_heads_per_kv_head + qblock_id * block_q;
        const int max_qq = std::min(num_query_heads_per_kv_head - qblock_id * block_q, block_q);

        Mat workspace_tile = workspace.channel(get_omp_thread_num());
        sdpa_decode_kvcache_tile_bf16s(query, key_cache, value_cache, attn_mask_blob, top_blob, scale, q0, max_qq, g, block_n, workspace_tile);
    }

    return 0;
}

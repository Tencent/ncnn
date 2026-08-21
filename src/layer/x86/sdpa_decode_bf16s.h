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
    unsigned short* queryT_ptr = queryT;
    int qq = 0;
#if __AVX__
#if __AVX512F__
    for (; qq + 15 < max_qq; qq += 16)
    {
        const int q = q0 + qq;
        unsigned short* pQ = queryT_ptr + (size_t)qq * head_dim;
        const unsigned short* qptr0 = query.channel(q);
        const unsigned short* qptr1 = query.channel(q + 1);
        const unsigned short* qptr2 = query.channel(q + 2);
        const unsigned short* qptr3 = query.channel(q + 3);
        const unsigned short* qptr4 = query.channel(q + 4);
        const unsigned short* qptr5 = query.channel(q + 5);
        const unsigned short* qptr6 = query.channel(q + 6);
        const unsigned short* qptr7 = query.channel(q + 7);
        const unsigned short* qptr8 = query.channel(q + 8);
        const unsigned short* qptr9 = query.channel(q + 9);
        const unsigned short* qptra = query.channel(q + 10);
        const unsigned short* qptrb = query.channel(q + 11);
        const unsigned short* qptrc = query.channel(q + 12);
        const unsigned short* qptrd = query.channel(q + 13);
        const unsigned short* qptre = query.channel(q + 14);
        const unsigned short* qptrf = query.channel(q + 15);

        int d = 0;
        for (; d + 15 < head_dim; d += 16)
        {
            __m256i _r0 = _mm256_loadu_si256((const __m256i*)(qptr0 + d));
            __m256i _r1 = _mm256_loadu_si256((const __m256i*)(qptr1 + d));
            __m256i _r2 = _mm256_loadu_si256((const __m256i*)(qptr2 + d));
            __m256i _r3 = _mm256_loadu_si256((const __m256i*)(qptr3 + d));
            __m256i _r4 = _mm256_loadu_si256((const __m256i*)(qptr4 + d));
            __m256i _r5 = _mm256_loadu_si256((const __m256i*)(qptr5 + d));
            __m256i _r6 = _mm256_loadu_si256((const __m256i*)(qptr6 + d));
            __m256i _r7 = _mm256_loadu_si256((const __m256i*)(qptr7 + d));
            __m256i _r8 = _mm256_loadu_si256((const __m256i*)(qptr8 + d));
            __m256i _r9 = _mm256_loadu_si256((const __m256i*)(qptr9 + d));
            __m256i _ra = _mm256_loadu_si256((const __m256i*)(qptra + d));
            __m256i _rb = _mm256_loadu_si256((const __m256i*)(qptrb + d));
            __m256i _rc = _mm256_loadu_si256((const __m256i*)(qptrc + d));
            __m256i _rd = _mm256_loadu_si256((const __m256i*)(qptrd + d));
            __m256i _re = _mm256_loadu_si256((const __m256i*)(qptre + d));
            __m256i _rf = _mm256_loadu_si256((const __m256i*)(qptrf + d));
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
            pQ += 256;
        }
#if __AVX512BF16__
        for (; d + 1 < head_dim; d += 2)
        {
            pQ[0] = qptr0[d];
            pQ[1] = qptr0[d + 1];
            pQ[2] = qptr1[d];
            pQ[3] = qptr1[d + 1];
            pQ[4] = qptr2[d];
            pQ[5] = qptr2[d + 1];
            pQ[6] = qptr3[d];
            pQ[7] = qptr3[d + 1];
            pQ[8] = qptr4[d];
            pQ[9] = qptr4[d + 1];
            pQ[10] = qptr5[d];
            pQ[11] = qptr5[d + 1];
            pQ[12] = qptr6[d];
            pQ[13] = qptr6[d + 1];
            pQ[14] = qptr7[d];
            pQ[15] = qptr7[d + 1];
            pQ[16] = qptr8[d];
            pQ[17] = qptr8[d + 1];
            pQ[18] = qptr9[d];
            pQ[19] = qptr9[d + 1];
            pQ[20] = qptra[d];
            pQ[21] = qptra[d + 1];
            pQ[22] = qptrb[d];
            pQ[23] = qptrb[d + 1];
            pQ[24] = qptrc[d];
            pQ[25] = qptrc[d + 1];
            pQ[26] = qptrd[d];
            pQ[27] = qptrd[d + 1];
            pQ[28] = qptre[d];
            pQ[29] = qptre[d + 1];
            pQ[30] = qptrf[d];
            pQ[31] = qptrf[d + 1];
            pQ += 32;
        }
#endif // __AVX512BF16__
        for (; d < head_dim; d++)
        {
            pQ[0] = qptr0[d];
            pQ[1] = qptr1[d];
            pQ[2] = qptr2[d];
            pQ[3] = qptr3[d];
            pQ[4] = qptr4[d];
            pQ[5] = qptr5[d];
            pQ[6] = qptr6[d];
            pQ[7] = qptr7[d];
            pQ[8] = qptr8[d];
            pQ[9] = qptr9[d];
            pQ[10] = qptra[d];
            pQ[11] = qptrb[d];
            pQ[12] = qptrc[d];
            pQ[13] = qptrd[d];
            pQ[14] = qptre[d];
            pQ[15] = qptrf[d];
            pQ += 16;
        }
    }
#endif // __AVX512F__
    for (; qq + 7 < max_qq; qq += 8)
    {
        const int q = q0 + qq;
        unsigned short* pQ = queryT_ptr + (size_t)qq * head_dim;
        const unsigned short* qptr0 = query.channel(q);
        const unsigned short* qptr1 = query.channel(q + 1);
        const unsigned short* qptr2 = query.channel(q + 2);
        const unsigned short* qptr3 = query.channel(q + 3);
        const unsigned short* qptr4 = query.channel(q + 4);
        const unsigned short* qptr5 = query.channel(q + 5);
        const unsigned short* qptr6 = query.channel(q + 6);
        const unsigned short* qptr7 = query.channel(q + 7);

        int d = 0;
        for (; d + 7 < head_dim; d += 8)
        {
            __m128i _r0 = _mm_loadu_si128((const __m128i*)(qptr0 + d));
            __m128i _r1 = _mm_loadu_si128((const __m128i*)(qptr1 + d));
            __m128i _r2 = _mm_loadu_si128((const __m128i*)(qptr2 + d));
            __m128i _r3 = _mm_loadu_si128((const __m128i*)(qptr3 + d));
            __m128i _r4 = _mm_loadu_si128((const __m128i*)(qptr4 + d));
            __m128i _r5 = _mm_loadu_si128((const __m128i*)(qptr5 + d));
            __m128i _r6 = _mm_loadu_si128((const __m128i*)(qptr6 + d));
            __m128i _r7 = _mm_loadu_si128((const __m128i*)(qptr7 + d));
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
            pQ += 64;
        }
#if __AVX512BF16__
        for (; d + 1 < head_dim; d += 2)
        {
            pQ[0] = qptr0[d];
            pQ[1] = qptr0[d + 1];
            pQ[2] = qptr1[d];
            pQ[3] = qptr1[d + 1];
            pQ[4] = qptr2[d];
            pQ[5] = qptr2[d + 1];
            pQ[6] = qptr3[d];
            pQ[7] = qptr3[d + 1];
            pQ[8] = qptr4[d];
            pQ[9] = qptr4[d + 1];
            pQ[10] = qptr5[d];
            pQ[11] = qptr5[d + 1];
            pQ[12] = qptr6[d];
            pQ[13] = qptr6[d + 1];
            pQ[14] = qptr7[d];
            pQ[15] = qptr7[d + 1];
            pQ += 16;
        }
#endif // __AVX512BF16__
        for (; d < head_dim; d++)
        {
            pQ[0] = qptr0[d];
            pQ[1] = qptr1[d];
            pQ[2] = qptr2[d];
            pQ[3] = qptr3[d];
            pQ[4] = qptr4[d];
            pQ[5] = qptr5[d];
            pQ[6] = qptr6[d];
            pQ[7] = qptr7[d];
            pQ += 8;
        }
    }
#endif // __AVX__
    for (; qq + 3 < max_qq; qq += 4)
    {
        const int q = q0 + qq;
        unsigned short* pQ = queryT_ptr + (size_t)qq * head_dim;
        const unsigned short* qptr0 = query.channel(q);
        const unsigned short* qptr1 = query.channel(q + 1);
        const unsigned short* qptr2 = query.channel(q + 2);
        const unsigned short* qptr3 = query.channel(q + 3);

        int d = 0;
        for (; d + 3 < head_dim; d += 4)
        {
            __m128i _r0 = _mm_loadl_epi64((const __m128i*)(qptr0 + d));
            __m128i _r1 = _mm_loadl_epi64((const __m128i*)(qptr1 + d));
            __m128i _r2 = _mm_loadl_epi64((const __m128i*)(qptr2 + d));
            __m128i _r3 = _mm_loadl_epi64((const __m128i*)(qptr3 + d));
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
            pQ += 16;
        }
#if __AVX512BF16__
        for (; d + 1 < head_dim; d += 2)
        {
            pQ[0] = qptr0[d];
            pQ[1] = qptr0[d + 1];
            pQ[2] = qptr1[d];
            pQ[3] = qptr1[d + 1];
            pQ[4] = qptr2[d];
            pQ[5] = qptr2[d + 1];
            pQ[6] = qptr3[d];
            pQ[7] = qptr3[d + 1];
            pQ += 8;
        }
#endif // __AVX512BF16__
        for (; d < head_dim; d++)
        {
            pQ[0] = qptr0[d];
            pQ[1] = qptr1[d];
            pQ[2] = qptr2[d];
            pQ[3] = qptr3[d];
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

static void sdpa_decode_tile_bf16s(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, int q0, int max_qq, int g, int n_begin, int n_end, int block_n, const Mat& packed_query, Mat& workspace, Mat& state)
{
    (void)packed_query;
    int qq = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; qq + 15 < max_qq; qq += 16)
    {
        const int q = q0 + qq;
        const int head_dim = query.w;
        const int value_dim = value.w;

        Mat state_q;
        if (!state.empty())
            state_q = state.range(qq * (value_dim + 2), (value_dim + 2) * 16);

        const bool mask_per_head = !attn_mask_blob.empty() && attn_mask_blob.dims == 3 && attn_mask_blob.c > 1;
        const unsigned short* mask = 0;
        if (!attn_mask_blob.empty())
        {
            if (attn_mask_blob.dims == 3)
                mask = attn_mask_blob.channel(mask_per_head ? q : 0);
            else
                mask = attn_mask_blob;
        }
        __m512i _mask_index;
        if (mask_per_head)
        {
            const int mask_hstep = attn_mask_blob.cstep;
            _mask_index = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(mask_hstep));
        }

        float* scoreT = workspace;
        float* outT = scoreT + block_n * 16;
        Mat queryT_blob = packed_query;
        if (queryT_blob.empty())
        {
            queryT_blob = Mat(head_dim * 16, (unsigned short*)(outT + value_dim * 16), 2u);
            sdpa_decode_pack_query_bf16s(query, queryT_blob, q, 16);
        }
        const unsigned short* queryT = queryT_blob;
        if (!packed_query.empty())
            queryT += (size_t)qq * head_dim;
        memset(outT, 0, (size_t)value_dim * 16 * sizeof(float));

        const Mat key_head = key.channel(g);
        const Mat value_head = value.channel(g);
        __m512 _m = _mm512_set1_ps(-FLT_MAX);
        __m512 _l = _mm512_setzero_ps();
        const __m512 _scale = _mm512_set1_ps(scale);

        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
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
                    __m512 _mask;
                    if (mask_per_head)
                    {
                        __m512i _mask_bf16 = _mm512_i32gather_epi32(_mask_index, (const int*)pM, sizeof(unsigned short));
                        _mask = bfloat2float_avx512(_mm512_cvtepi32_epi16(_mask_bf16));
                    }
                    else
                    {
                        _mask = _mm512_set1_ps(bfloat16_to_float32(*pM));
                    }
                    _score = _mm512_add_ps(_score, _mask);
                    pM++;
                }
                _mm512_storeu_ps(pS, _score);
                pS += 16;
                _block_max = _mm512_max_ps(_block_max, _score);
            }

            __m512 _m_new = _mm512_max_ps(_m, _block_max);
            const __mmask16 alpha_active = _mm512_cmp_ps_mask(_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
            __m512 _alpha = _mm512_maskz_mov_ps(alpha_active, exp512_ps(_mm512_maskz_sub_ps(alpha_active, _m, _m_new)));

            float* scoreptr = scoreT;
            __m512 _sum0 = _mm512_setzero_ps();
            __m512 _sum1 = _mm512_setzero_ps();
            __m512 _sum2 = _mm512_setzero_ps();
            __m512 _sum3 = _mm512_setzero_ps();
            int j = 0;
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
            for (; j < max_jj; j++)
            {
                __m512 _p = _mm512_sub_ps(_mm512_loadu_ps(scoreptr), _m_new);
                _p = exp512_ps(_p);
                _mm512_storeu_ps(scoreptr, _p);
                scoreptr += 16;
                _sum0 = _mm512_add_ps(_sum0, _p);
            }
            __m512 _sum = _mm512_add_ps(_mm512_add_ps(_sum0, _sum1), _mm512_add_ps(_sum2, _sum3));
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

        if (!state_q.empty())
        {
            float* stateptr = state_q;
            _mm512_storeu_ps(stateptr, _m);
            _mm512_storeu_ps(stateptr + 16, _l);
            memcpy(stateptr + 32, outT, (size_t)value_dim * 16 * sizeof(float));
        }
        else
        {
            float* output0 = top_blob.channel(q + 0);
            float* output1 = top_blob.channel(q + 1);
            float* output2 = top_blob.channel(q + 2);
            float* output3 = top_blob.channel(q + 3);
            float* output4 = top_blob.channel(q + 4);
            float* output5 = top_blob.channel(q + 5);
            float* output6 = top_blob.channel(q + 6);
            float* output7 = top_blob.channel(q + 7);
            float* output8 = top_blob.channel(q + 8);
            float* output9 = top_blob.channel(q + 9);
            float* outputa = top_blob.channel(q + 10);
            float* outputb = top_blob.channel(q + 11);
            float* outputc = top_blob.channel(q + 12);
            float* outputd = top_blob.channel(q + 13);
            float* outpute = top_blob.channel(q + 14);
            float* outputf = top_blob.channel(q + 15);
            const __mmask16 nonzero = _mm512_cmp_ps_mask(_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
            const __m512 _out_scale = _mm512_maskz_div_ps(nonzero, _mm512_set1_ps(1.f), _l);

            const float* outptr = outT;
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
                _mm512_storeu_ps(output0 + d, _r0);
                _mm512_storeu_ps(output1 + d, _r1);
                _mm512_storeu_ps(output2 + d, _r2);
                _mm512_storeu_ps(output3 + d, _r3);
                _mm512_storeu_ps(output4 + d, _r4);
                _mm512_storeu_ps(output5 + d, _r5);
                _mm512_storeu_ps(output6 + d, _r6);
                _mm512_storeu_ps(output7 + d, _r7);
                _mm512_storeu_ps(output8 + d, _r8);
                _mm512_storeu_ps(output9 + d, _r9);
                _mm512_storeu_ps(outputa + d, _ra);
                _mm512_storeu_ps(outputb + d, _rb);
                _mm512_storeu_ps(outputc + d, _rc);
                _mm512_storeu_ps(outputd + d, _rd);
                _mm512_storeu_ps(outpute + d, _re);
                _mm512_storeu_ps(outputf + d, _rf);
                outptr += 256;
            }
            for (; d < value_dim; d++)
            {
                const __m512 _r = _mm512_mul_ps(_mm512_loadu_ps(outptr), _out_scale);
                const __m128 _r0 = _mm512_extractf32x4_ps(_r, 0);
                const __m128 _r1 = _mm512_extractf32x4_ps(_r, 1);
                const __m128 _r2 = _mm512_extractf32x4_ps(_r, 2);
                const __m128 _r3 = _mm512_extractf32x4_ps(_r, 3);
                output0[d] = _mm_cvtss_f32(_r0);
                output1[d] = _mm_cvtss_f32(_mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(1, 1, 1, 1)));
                output2[d] = _mm_cvtss_f32(_mm_movehl_ps(_r0, _r0));
                output3[d] = _mm_cvtss_f32(_mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(3, 3, 3, 3)));
                output4[d] = _mm_cvtss_f32(_r1);
                output5[d] = _mm_cvtss_f32(_mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(1, 1, 1, 1)));
                output6[d] = _mm_cvtss_f32(_mm_movehl_ps(_r1, _r1));
                output7[d] = _mm_cvtss_f32(_mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(3, 3, 3, 3)));
                output8[d] = _mm_cvtss_f32(_r2);
                output9[d] = _mm_cvtss_f32(_mm_shuffle_ps(_r2, _r2, _MM_SHUFFLE(1, 1, 1, 1)));
                outputa[d] = _mm_cvtss_f32(_mm_movehl_ps(_r2, _r2));
                outputb[d] = _mm_cvtss_f32(_mm_shuffle_ps(_r2, _r2, _MM_SHUFFLE(3, 3, 3, 3)));
                outputc[d] = _mm_cvtss_f32(_r3);
                outputd[d] = _mm_cvtss_f32(_mm_shuffle_ps(_r3, _r3, _MM_SHUFFLE(1, 1, 1, 1)));
                outpute[d] = _mm_cvtss_f32(_mm_movehl_ps(_r3, _r3));
                outputf[d] = _mm_cvtss_f32(_mm_shuffle_ps(_r3, _r3, _MM_SHUFFLE(3, 3, 3, 3)));
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

        Mat state_q;
        if (!state.empty())
            state_q = state.range(qq * (value_dim + 2), (value_dim + 2) * 8);

        Mat mask_head0;
        Mat mask_head1;
        Mat mask_head2;
        Mat mask_head3;
        Mat mask_head4;
        Mat mask_head5;
        Mat mask_head6;
        Mat mask_head7;
        if (!attn_mask_blob.empty())
        {
            if (attn_mask_blob.dims == 3)
            {
                mask_head0 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q : 0);
                mask_head1 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q + 1 : 0);
                mask_head2 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q + 2 : 0);
                mask_head3 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q + 3 : 0);
                mask_head4 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q + 4 : 0);
                mask_head5 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q + 5 : 0);
                mask_head6 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q + 6 : 0);
                mask_head7 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q + 7 : 0);
            }
            else
            {
                mask_head0 = attn_mask_blob;
                mask_head1 = attn_mask_blob;
                mask_head2 = attn_mask_blob;
                mask_head3 = attn_mask_blob;
                mask_head4 = attn_mask_blob;
                mask_head5 = attn_mask_blob;
                mask_head6 = attn_mask_blob;
                mask_head7 = attn_mask_blob;
            }
        }
        const unsigned short* mask0 = mask_head0.empty() ? 0 : mask_head0;
        const unsigned short* mask1 = mask_head1.empty() ? 0 : mask_head1;
        const unsigned short* mask2 = mask_head2.empty() ? 0 : mask_head2;
        const unsigned short* mask3 = mask_head3.empty() ? 0 : mask_head3;
        const unsigned short* mask4 = mask_head4.empty() ? 0 : mask_head4;
        const unsigned short* mask5 = mask_head5.empty() ? 0 : mask_head5;
        const unsigned short* mask6 = mask_head6.empty() ? 0 : mask_head6;
        const unsigned short* mask7 = mask_head7.empty() ? 0 : mask_head7;

        float* scoreT = workspace;
        float* outT = scoreT + block_n * 8;
        Mat queryT_blob = packed_query;
        if (queryT_blob.empty())
        {
            queryT_blob = Mat(head_dim * 8, (unsigned short*)(outT + value_dim * 8), 2u);
            sdpa_decode_pack_query_bf16s(query, queryT_blob, q, 8);
        }
        const unsigned short* queryT = queryT_blob;
        if (!packed_query.empty())
            queryT += (size_t)qq * head_dim;
        memset(outT, 0, (size_t)value_dim * 8 * sizeof(float));

        const Mat key_head = key.channel(g);
        const Mat value_head = value.channel(g);
        __m256 _m = _mm256_set1_ps(-FLT_MAX);
        __m256 _l = _mm256_setzero_ps();
        const __m256 _scale = _mm256_set1_ps(scale);

        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
            __m256 _block_max = _mm256_set1_ps(-FLT_MAX);
            const unsigned short* pK = key_head.row<const unsigned short>(n);
            float* pS = scoreT;
            const unsigned short* pM0 = mask0 ? mask0 + n : 0;
            const unsigned short* pM1 = mask1 ? mask1 + n : 0;
            const unsigned short* pM2 = mask2 ? mask2 + n : 0;
            const unsigned short* pM3 = mask3 ? mask3 + n : 0;
            const unsigned short* pM4 = mask4 ? mask4 + n : 0;
            const unsigned short* pM5 = mask5 ? mask5 + n : 0;
            const unsigned short* pM6 = mask6 ? mask6 + n : 0;
            const unsigned short* pM7 = mask7 ? mask7 + n : 0;
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
                if (pM0)
                {
                    _score = _mm256_add_ps(_score, _mm256_set_ps(bfloat16_to_float32(*pM7), bfloat16_to_float32(*pM6), bfloat16_to_float32(*pM5), bfloat16_to_float32(*pM4), bfloat16_to_float32(*pM3), bfloat16_to_float32(*pM2), bfloat16_to_float32(*pM1), bfloat16_to_float32(*pM0)));
                    pM0++;
                    pM1++;
                    pM2++;
                    pM3++;
                    pM4++;
                    pM5++;
                    pM6++;
                    pM7++;
                }
                _mm256_storeu_ps(pS, _score);
                pS += 8;
                _block_max = _mm256_max_ps(_block_max, _score);
            }

            __m256 _m_new = _mm256_max_ps(_m, _block_max);
            const __m256 _alpha_active = _mm256_cmp_ps(_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
            __m256 _alpha = _mm256_and_ps(_alpha_active, exp256_ps(_mm256_and_ps(_alpha_active, _mm256_sub_ps(_m, _m_new))));

            float* scoreptr = scoreT;
            __m256 _sum0 = _mm256_setzero_ps();
            __m256 _sum1 = _mm256_setzero_ps();
            __m256 _sum2 = _mm256_setzero_ps();
            __m256 _sum3 = _mm256_setzero_ps();
            int j = 0;
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
            for (; j < max_jj; j++)
            {
                __m256 _p = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr), _m_new));
                _mm256_storeu_ps(scoreptr, _p);
                scoreptr += 8;
                _sum0 = _mm256_add_ps(_sum0, _p);
            }
            const __m256 _sum = _mm256_add_ps(_mm256_add_ps(_sum0, _sum1), _mm256_add_ps(_sum2, _sum3));
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
                    const __m256 _p = _mm256_loadu_ps(pS);
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

        if (!state_q.empty())
        {
            float* stateptr = state_q;
            _mm256_storeu_ps(stateptr, _m);
            _mm256_storeu_ps(stateptr + 8, _l);
            memcpy(stateptr + 16, outT, (size_t)value_dim * 8 * sizeof(float));
        }
        else
        {
            float* output0 = top_blob.channel(q);
            float* output1 = top_blob.channel(q + 1);
            float* output2 = top_blob.channel(q + 2);
            float* output3 = top_blob.channel(q + 3);
            float* output4 = top_blob.channel(q + 4);
            float* output5 = top_blob.channel(q + 5);
            float* output6 = top_blob.channel(q + 6);
            float* output7 = top_blob.channel(q + 7);
            const __m256 _nonzero = _mm256_cmp_ps(_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
            const __m256 _denom = _mm256_blendv_ps(_mm256_set1_ps(1.f), _l, _nonzero);
            const __m256 _out_scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(1.f), _denom), _nonzero);

            const float* outptr = outT;
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
                _mm256_storeu_ps(output0 + d, _r0);
                _mm256_storeu_ps(output1 + d, _r1);
                _mm256_storeu_ps(output2 + d, _r2);
                _mm256_storeu_ps(output3 + d, _r3);
                _mm256_storeu_ps(output4 + d, _r4);
                _mm256_storeu_ps(output5 + d, _r5);
                _mm256_storeu_ps(output6 + d, _r6);
                _mm256_storeu_ps(output7 + d, _r7);
                outptr += 64;
            }
            for (; d < value_dim; d++)
            {
                const __m256 _r = _mm256_mul_ps(_mm256_loadu_ps(outptr), _out_scale);
                const __m128 _r0 = _mm256_castps256_ps128(_r);
                const __m128 _r1 = _mm256_extractf128_ps(_r, 1);
                output0[d] = _mm_cvtss_f32(_r0);
                output1[d] = _mm_cvtss_f32(_mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(1, 1, 1, 1)));
                output2[d] = _mm_cvtss_f32(_mm_movehl_ps(_r0, _r0));
                output3[d] = _mm_cvtss_f32(_mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(3, 3, 3, 3)));
                output4[d] = _mm_cvtss_f32(_r1);
                output5[d] = _mm_cvtss_f32(_mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(1, 1, 1, 1)));
                output6[d] = _mm_cvtss_f32(_mm_movehl_ps(_r1, _r1));
                output7[d] = _mm_cvtss_f32(_mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(3, 3, 3, 3)));
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

        Mat state_q;
        if (!state.empty())
            state_q = state.range(qq * (value_dim + 2), (value_dim + 2) * 4);

        Mat mask_head0;
        Mat mask_head1;
        Mat mask_head2;
        Mat mask_head3;
        if (!attn_mask_blob.empty())
        {
            if (attn_mask_blob.dims == 3)
            {
                mask_head0 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q : 0);
                mask_head1 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q + 1 : 0);
                mask_head2 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q + 2 : 0);
                mask_head3 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q + 3 : 0);
            }
            else
            {
                mask_head0 = attn_mask_blob;
                mask_head1 = attn_mask_blob;
                mask_head2 = attn_mask_blob;
                mask_head3 = attn_mask_blob;
            }
        }
        const unsigned short* mask0 = mask_head0.empty() ? 0 : mask_head0;
        const unsigned short* mask1 = mask_head1.empty() ? 0 : mask_head1;
        const unsigned short* mask2 = mask_head2.empty() ? 0 : mask_head2;
        const unsigned short* mask3 = mask_head3.empty() ? 0 : mask_head3;

        float* scoreT = workspace;
        float* outT = scoreT + block_n * 4;
        Mat queryT_blob = packed_query;
        if (queryT_blob.empty())
        {
            queryT_blob = Mat(head_dim * 4, (unsigned short*)(outT + value_dim * 4), 2u);
            sdpa_decode_pack_query_bf16s(query, queryT_blob, q, 4);
        }
        const unsigned short* queryT = queryT_blob;
        if (!packed_query.empty())
            queryT += (size_t)qq * head_dim;
        memset(outT, 0, (size_t)value_dim * 4 * sizeof(float));

        const Mat key_head = key.channel(g);
        const Mat value_head = value.channel(g);

        __m128 _m = _mm_set1_ps(-FLT_MAX);
        __m128 _l = _mm_setzero_ps();

        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
            __m128 _block_max = _mm_set1_ps(-FLT_MAX);
            {
                float* pS = scoreT;
                const unsigned short* pM0 = mask0 ? mask0 + n : 0;
                const unsigned short* pM1 = mask1 ? mask1 + n : 0;
                const unsigned short* pM2 = mask2 ? mask2 + n : 0;
                const unsigned short* pM3 = mask3 ? mask3 + n : 0;
                int j = 0;
#if __AVX__
#if __AVX512F__
                for (; j + 3 < max_jj; j += 4)
                {
                    const unsigned short* pK0 = key_head.row<const unsigned short>(n + j);
                    const unsigned short* pK1 = key_head.row<const unsigned short>(n + j + 1);
                    const unsigned short* pK2 = key_head.row<const unsigned short>(n + j + 2);
                    const unsigned short* pK3 = key_head.row<const unsigned short>(n + j + 3);
                    const unsigned short* pQ = queryT;
#if __AVX512BF16__
                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();
                    int d = 0;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m128i _q = _mm_loadu_si128((const __m128i*)pQ);
                        _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_q, (__m128bh)_mm_set1_epi32(((const int*)pK0)[0]));
                        _sum1 = _mm_dpbf16_ps(_sum1, (__m128bh)_q, (__m128bh)_mm_set1_epi32(((const int*)pK1)[0]));
                        _sum2 = _mm_dpbf16_ps(_sum2, (__m128bh)_q, (__m128bh)_mm_set1_epi32(((const int*)pK2)[0]));
                        _sum3 = _mm_dpbf16_ps(_sum3, (__m128bh)_q, (__m128bh)_mm_set1_epi32(((const int*)pK3)[0]));
                        pQ += 8;
                        pK0 += 2;
                        pK1 += 2;
                        pK2 += 2;
                        pK3 += 2;
                    }
                    for (; d < head_dim; d++)
                    {
                        __m128 _q = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pQ));
                        _sum0 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(*pK0++)), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(*pK1++)), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(*pK2++)), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(*pK3++)), _sum3);
                        pQ += 4;
                    }
                    __m512 _score = _mm512_mul_ps(combine4x4_ps(_sum0, _sum1, _sum2, _sum3), _mm512_set1_ps(scale));
#else
                    __m512 _sum0 = _mm512_setzero_ps();
                    __m512 _sum1 = _mm512_setzero_ps();
                    __m512 _sum2 = _mm512_setzero_ps();
                    __m512 _sum3 = _mm512_setzero_ps();
                    int d = 0;
                    for (; d + 3 < head_dim; d += 4)
                    {
                        __m512 _q0 = _mm512_broadcast_f32x4(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pQ)));
                        __m512 _q1 = _mm512_broadcast_f32x4(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pQ + 4))));
                        __m512 _q2 = _mm512_broadcast_f32x4(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pQ + 8))));
                        __m512 _q3 = _mm512_broadcast_f32x4(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pQ + 12))));
                        __m512 _k0 = _mm512_set_ps(bfloat16_to_float32(pK3[0]), bfloat16_to_float32(pK3[0]), bfloat16_to_float32(pK3[0]), bfloat16_to_float32(pK3[0]), bfloat16_to_float32(pK2[0]), bfloat16_to_float32(pK2[0]), bfloat16_to_float32(pK2[0]), bfloat16_to_float32(pK2[0]), bfloat16_to_float32(pK1[0]), bfloat16_to_float32(pK1[0]), bfloat16_to_float32(pK1[0]), bfloat16_to_float32(pK1[0]), bfloat16_to_float32(pK0[0]), bfloat16_to_float32(pK0[0]), bfloat16_to_float32(pK0[0]), bfloat16_to_float32(pK0[0]));
                        __m512 _k1 = _mm512_set_ps(bfloat16_to_float32(pK3[1]), bfloat16_to_float32(pK3[1]), bfloat16_to_float32(pK3[1]), bfloat16_to_float32(pK3[1]), bfloat16_to_float32(pK2[1]), bfloat16_to_float32(pK2[1]), bfloat16_to_float32(pK2[1]), bfloat16_to_float32(pK2[1]), bfloat16_to_float32(pK1[1]), bfloat16_to_float32(pK1[1]), bfloat16_to_float32(pK1[1]), bfloat16_to_float32(pK1[1]), bfloat16_to_float32(pK0[1]), bfloat16_to_float32(pK0[1]), bfloat16_to_float32(pK0[1]), bfloat16_to_float32(pK0[1]));
                        __m512 _k2 = _mm512_set_ps(bfloat16_to_float32(pK3[2]), bfloat16_to_float32(pK3[2]), bfloat16_to_float32(pK3[2]), bfloat16_to_float32(pK3[2]), bfloat16_to_float32(pK2[2]), bfloat16_to_float32(pK2[2]), bfloat16_to_float32(pK2[2]), bfloat16_to_float32(pK2[2]), bfloat16_to_float32(pK1[2]), bfloat16_to_float32(pK1[2]), bfloat16_to_float32(pK1[2]), bfloat16_to_float32(pK1[2]), bfloat16_to_float32(pK0[2]), bfloat16_to_float32(pK0[2]), bfloat16_to_float32(pK0[2]), bfloat16_to_float32(pK0[2]));
                        __m512 _k3 = _mm512_set_ps(bfloat16_to_float32(pK3[3]), bfloat16_to_float32(pK3[3]), bfloat16_to_float32(pK3[3]), bfloat16_to_float32(pK3[3]), bfloat16_to_float32(pK2[3]), bfloat16_to_float32(pK2[3]), bfloat16_to_float32(pK2[3]), bfloat16_to_float32(pK2[3]), bfloat16_to_float32(pK1[3]), bfloat16_to_float32(pK1[3]), bfloat16_to_float32(pK1[3]), bfloat16_to_float32(pK1[3]), bfloat16_to_float32(pK0[3]), bfloat16_to_float32(pK0[3]), bfloat16_to_float32(pK0[3]), bfloat16_to_float32(pK0[3]));
                        _sum0 = _mm512_fmadd_ps(_q0, _k0, _sum0);
                        _sum1 = _mm512_fmadd_ps(_q1, _k1, _sum1);
                        _sum2 = _mm512_fmadd_ps(_q2, _k2, _sum2);
                        _sum3 = _mm512_fmadd_ps(_q3, _k3, _sum3);
                        pQ += 16;
                        pK0 += 4;
                        pK1 += 4;
                        pK2 += 4;
                        pK3 += 4;
                    }
                    for (; d < head_dim; d++)
                    {
                        __m512 _q = _mm512_broadcast_f32x4(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pQ)));
                        __m512 _k = _mm512_set_ps(bfloat16_to_float32(*pK3), bfloat16_to_float32(*pK3), bfloat16_to_float32(*pK3), bfloat16_to_float32(*pK3), bfloat16_to_float32(*pK2), bfloat16_to_float32(*pK2), bfloat16_to_float32(*pK2), bfloat16_to_float32(*pK2), bfloat16_to_float32(*pK1), bfloat16_to_float32(*pK1), bfloat16_to_float32(*pK1), bfloat16_to_float32(*pK1), bfloat16_to_float32(*pK0), bfloat16_to_float32(*pK0), bfloat16_to_float32(*pK0), bfloat16_to_float32(*pK0));
                        _sum0 = _mm512_fmadd_ps(_q, _k, _sum0);
                        pQ += 4;
                        pK0++;
                        pK1++;
                        pK2++;
                        pK3++;
                    }
                    __m512 _score = _mm512_mul_ps(_mm512_add_ps(_mm512_add_ps(_sum0, _sum1), _mm512_add_ps(_sum2, _sum3)), _mm512_set1_ps(scale));
#endif // __AVX512BF16__
                    if (pM0)
                    {
                        __m512 _mask = _mm512_set_ps(bfloat16_to_float32(pM3[3]), bfloat16_to_float32(pM2[3]), bfloat16_to_float32(pM1[3]), bfloat16_to_float32(pM0[3]), bfloat16_to_float32(pM3[2]), bfloat16_to_float32(pM2[2]), bfloat16_to_float32(pM1[2]), bfloat16_to_float32(pM0[2]), bfloat16_to_float32(pM3[1]), bfloat16_to_float32(pM2[1]), bfloat16_to_float32(pM1[1]), bfloat16_to_float32(pM0[1]), bfloat16_to_float32(pM3[0]), bfloat16_to_float32(pM2[0]), bfloat16_to_float32(pM1[0]), bfloat16_to_float32(pM0[0]));
                        _score = _mm512_add_ps(_score, _mask);
                        pM0 += 4;
                        pM1 += 4;
                        pM2 += 4;
                        pM3 += 4;
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
                    const unsigned short* pK0 = key_head.row<const unsigned short>(n + j);
                    const unsigned short* pK1 = key_head.row<const unsigned short>(n + j + 1);
                    const unsigned short* pQ = queryT;
#if __AVX512BF16__
                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    int d = 0;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        __m128i _q = _mm_loadu_si128((const __m128i*)pQ);
                        _sum0 = _mm_dpbf16_ps(_sum0, (__m128bh)_q, (__m128bh)_mm_set1_epi32(((const int*)pK0)[0]));
                        _sum1 = _mm_dpbf16_ps(_sum1, (__m128bh)_q, (__m128bh)_mm_set1_epi32(((const int*)pK1)[0]));
                        pQ += 8;
                        pK0 += 2;
                        pK1 += 2;
                    }
                    for (; d < head_dim; d++)
                    {
                        __m128 _q = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pQ));
                        _sum0 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(*pK0++)), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(bfloat16_to_float32(*pK1++)), _sum1);
                        pQ += 4;
                    }
                    __m256 _score = _mm256_mul_ps(combine4x2_ps(_sum0, _sum1), _mm256_set1_ps(scale));
#else
                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    int d = 0;
                    for (; d + 3 < head_dim; d += 4)
                    {
                        __m128 _q0_128 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pQ));
                        __m128 _q1_128 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pQ + 4)));
                        __m128 _q2_128 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pQ + 8)));
                        __m128 _q3_128 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(pQ + 12)));
                        __m256 _q0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_q0_128), _q0_128, 1);
                        __m256 _q1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_q1_128), _q1_128, 1);
                        __m256 _q2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_q2_128), _q2_128, 1);
                        __m256 _q3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_q3_128), _q3_128, 1);
                        __m256 _k0 = _mm256_set_ps(bfloat16_to_float32(pK1[0]), bfloat16_to_float32(pK1[0]), bfloat16_to_float32(pK1[0]), bfloat16_to_float32(pK1[0]), bfloat16_to_float32(pK0[0]), bfloat16_to_float32(pK0[0]), bfloat16_to_float32(pK0[0]), bfloat16_to_float32(pK0[0]));
                        __m256 _k1 = _mm256_set_ps(bfloat16_to_float32(pK1[1]), bfloat16_to_float32(pK1[1]), bfloat16_to_float32(pK1[1]), bfloat16_to_float32(pK1[1]), bfloat16_to_float32(pK0[1]), bfloat16_to_float32(pK0[1]), bfloat16_to_float32(pK0[1]), bfloat16_to_float32(pK0[1]));
                        __m256 _k2 = _mm256_set_ps(bfloat16_to_float32(pK1[2]), bfloat16_to_float32(pK1[2]), bfloat16_to_float32(pK1[2]), bfloat16_to_float32(pK1[2]), bfloat16_to_float32(pK0[2]), bfloat16_to_float32(pK0[2]), bfloat16_to_float32(pK0[2]), bfloat16_to_float32(pK0[2]));
                        __m256 _k3 = _mm256_set_ps(bfloat16_to_float32(pK1[3]), bfloat16_to_float32(pK1[3]), bfloat16_to_float32(pK1[3]), bfloat16_to_float32(pK1[3]), bfloat16_to_float32(pK0[3]), bfloat16_to_float32(pK0[3]), bfloat16_to_float32(pK0[3]), bfloat16_to_float32(pK0[3]));
                        _sum0 = _mm256_comp_fmadd_ps(_q0, _k0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_q1, _k1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_q2, _k2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_q3, _k3, _sum3);
                        pQ += 16;
                        pK0 += 4;
                        pK1 += 4;
                    }
                    for (; d < head_dim; d++)
                    {
                        __m128 _q128 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)pQ));
                        __m256 _q = _mm256_insertf128_ps(_mm256_castps128_ps256(_q128), _q128, 1);
                        __m256 _k = _mm256_set_ps(bfloat16_to_float32(*pK1), bfloat16_to_float32(*pK1), bfloat16_to_float32(*pK1), bfloat16_to_float32(*pK1), bfloat16_to_float32(*pK0), bfloat16_to_float32(*pK0), bfloat16_to_float32(*pK0), bfloat16_to_float32(*pK0));
                        _sum0 = _mm256_comp_fmadd_ps(_q, _k, _sum0);
                        pQ += 4;
                        pK0++;
                        pK1++;
                    }
                    __m256 _score = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(_sum0, _sum1), _mm256_add_ps(_sum2, _sum3)), _mm256_set1_ps(scale));
#endif // __AVX512BF16__
                    if (pM0)
                    {
                        __m256 _mask = _mm256_set_ps(bfloat16_to_float32(pM3[1]), bfloat16_to_float32(pM2[1]), bfloat16_to_float32(pM1[1]), bfloat16_to_float32(pM0[1]), bfloat16_to_float32(pM3[0]), bfloat16_to_float32(pM2[0]), bfloat16_to_float32(pM1[0]), bfloat16_to_float32(pM0[0]));
                        _score = _mm256_add_ps(_score, _mask);
                        pM0 += 2;
                        pM1 += 2;
                        pM2 += 2;
                        pM3 += 2;
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
                    if (pM0)
                    {
                        _score = _mm_add_ps(_score, _mm_set_ps(bfloat16_to_float32(*pM3), bfloat16_to_float32(*pM2), bfloat16_to_float32(*pM1), bfloat16_to_float32(*pM0)));
                        pM0++;
                        pM1++;
                        pM2++;
                        pM3++;
                    }
                    _mm_storeu_ps(pS, _score);
                    pS += 4;
                    _block_max = _mm_max_ps(_block_max, _score);
                }
            }

            __m128 _m_new = _mm_max_ps(_m, _block_max);
            const __m128 _alpha_active = _mm_cmpneq_ps(_l, _mm_setzero_ps());
            __m128 _alpha = exp_ps(_mm_and_ps(_alpha_active, _mm_sub_ps(_m, _m_new)));
            _alpha = _mm_and_ps(_alpha, _alpha_active);

            float* scoreptr = scoreT;
            __m128 _sum0 = _mm_setzero_ps();
            __m128 _sum1 = _mm_setzero_ps();
            __m128 _sum2 = _mm_setzero_ps();
            __m128 _sum3 = _mm_setzero_ps();
            int j = 0;
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
            for (; j < max_jj; j++)
            {
                __m128 _score = _mm_sub_ps(_mm_loadu_ps(scoreptr), _m_new);
                __m128 _p = exp_ps(_score);
                _mm_storeu_ps(scoreptr, _p);
                scoreptr += 4;
                _sum0 = _mm_add_ps(_sum0, _p);
            }
            __m128 _sum = _mm_add_ps(_mm_add_ps(_sum0, _sum1), _mm_add_ps(_sum2, _sum3));

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

        if (!state_q.empty())
        {
            float* stateptr = state_q;
            _mm_storeu_ps(stateptr, _m);
            _mm_storeu_ps(stateptr + 4, _l);
            memcpy(stateptr + 8, outT, (size_t)value_dim * 4 * sizeof(float));
        }
        else
        {
            float* output0 = top_blob.channel(q);
            float* output1 = top_blob.channel(q + 1);
            float* output2 = top_blob.channel(q + 2);
            float* output3 = top_blob.channel(q + 3);
            const __m128 _nonzero = _mm_cmpneq_ps(_l, _mm_setzero_ps());
            const __m128 _denom = _mm_or_ps(_mm_and_ps(_nonzero, _l), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
            const __m128 _out_scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(1.f), _denom), _nonzero);

            const float* outptr = outT;
            int d = 0;
            for (; d + 3 < value_dim; d += 4)
            {
                __m128 _r0 = _mm_mul_ps(_mm_loadu_ps(outptr), _out_scale);
                __m128 _r1 = _mm_mul_ps(_mm_loadu_ps(outptr + 4), _out_scale);
                __m128 _r2 = _mm_mul_ps(_mm_loadu_ps(outptr + 8), _out_scale);
                __m128 _r3 = _mm_mul_ps(_mm_loadu_ps(outptr + 12), _out_scale);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                _mm_storeu_ps(output0 + d, _r0);
                _mm_storeu_ps(output1 + d, _r1);
                _mm_storeu_ps(output2 + d, _r2);
                _mm_storeu_ps(output3 + d, _r3);
                outptr += 16;
            }
            for (; d < value_dim; d++)
            {
                const __m128 _r = _mm_mul_ps(_mm_loadu_ps(outptr), _out_scale);
                output0[d] = _mm_cvtss_f32(_r);
                output1[d] = _mm_cvtss_f32(_mm_shuffle_ps(_r, _r, _MM_SHUFFLE(1, 1, 1, 1)));
                output2[d] = _mm_cvtss_f32(_mm_movehl_ps(_r, _r));
                output3[d] = _mm_cvtss_f32(_mm_shuffle_ps(_r, _r, _MM_SHUFFLE(3, 3, 3, 3)));
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

        Mat state_q;
        if (!state.empty())
            state_q = state.range(qq * (value_dim + 2), value_dim + 2);

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

        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
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

        if (!state_q.empty())
        {
            float* state_ptr = state_q;
            state_ptr[0] = m;
            state_ptr[1] = l;
            memcpy(state_ptr + 2, out, value_dim * sizeof(float));
        }
        else
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
    const int nT = std::max(opt.num_threads, 1);
    const int block_q = sdpa_decode_get_optimal_tile_q(num_query_heads_per_kv_head, num_kv_heads, nT);
    const int num_qblocks = (num_query_heads_per_kv_head + block_q - 1) / block_q;
    const int num_tasks = num_kv_heads * num_qblocks;
    const int block_n = sdpa_decode_get_optimal_tile_n(query.w, value_dim, key_seqlen, 2, 2, 2, attn_mask_blob.empty() ? 0 : 2, block_q, num_tasks, nT);
    const int num_key_blocks = (key_seqlen + block_n - 1) / block_n;
    const bool use_packed_query = block_q >= 4 && num_query_heads_per_kv_head >= 4;

    int num_kv_chunks = 1;
    if (num_tasks < nT && num_key_blocks >= 2)
    {
        num_kv_chunks = std::min((nT + num_tasks - 1) / num_tasks, num_key_blocks);
        num_kv_chunks = std::max(num_kv_chunks, 1);
    }

    Mat packed_query;
    if (num_kv_chunks > 1 && use_packed_query)
    {
        packed_query.create(query.w * block_q, 1, num_tasks, 2u, opt.workspace_allocator);
        if (packed_query.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int task_id = 0; task_id < num_tasks; task_id++)
        {
            const int g = task_id / num_qblocks;
            const int qblock_id = task_id % num_qblocks;
            const int q0 = g * num_query_heads_per_kv_head + qblock_id * block_q;
            const int max_qq = std::min(num_query_heads_per_kv_head - qblock_id * block_q, block_q);
            Mat queryT = packed_query.channel(task_id);
            sdpa_decode_pack_query_bf16s(query, queryT, q0, max_qq);
        }
    }

    const int query_workspace_size = use_packed_query ? (query.w * block_q + 1) / 2 : 0;
    const int workspace_size = (block_q * (block_n + value_dim) + query_workspace_size + 15) / 16 * 16;
    Mat workspace(workspace_size, 1, nT, 4u, opt.workspace_allocator);
    if (workspace.empty())
        return -100;

    Mat partials;
    if (num_kv_chunks > 1)
    {
        partials.create((value_dim + 2) * block_q, 1, num_tasks * num_kv_chunks, 4u, opt.workspace_allocator);
        if (partials.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ti = 0; ti < num_tasks * num_kv_chunks; ti++)
    {
        const int task_id = ti / num_kv_chunks;
        const int chunk_id = ti % num_kv_chunks;
        const int g = task_id / num_qblocks;
        const int qblock_id = task_id % num_qblocks;
        const int q0 = g * num_query_heads_per_kv_head + qblock_id * block_q;
        const int max_qq = std::min(num_query_heads_per_kv_head - qblock_id * block_q, block_q);
        const int n_begin = chunk_id * num_key_blocks / num_kv_chunks * block_n;
        const int n_end = std::min((chunk_id + 1) * num_key_blocks / num_kv_chunks * block_n, key_seqlen);

        Mat workspace_tile = workspace.channel(get_omp_thread_num());
        Mat state;
        Mat packed_query_tile;
        if (num_kv_chunks > 1)
        {
            state = partials.channel(ti);
            if (!packed_query.empty())
                packed_query_tile = packed_query.channel(task_id);
        }
        sdpa_decode_tile_bf16s(query, key, value, attn_mask_blob, top_blob, scale, q0, max_qq, g, n_begin, n_end, block_n, packed_query_tile, workspace_tile, state);
    }

    if (num_kv_chunks > 1)
        sdpa_decode_reduce(partials, top_blob, workspace, num_tasks, num_qblocks, block_q, num_kv_chunks, num_query_heads_per_kv_head, value_dim, opt);

    return 0;
}

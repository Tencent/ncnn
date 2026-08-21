// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static int sdpa_decode_block_q(int num_query_heads_per_kv_head)
{
#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (num_query_heads_per_kv_head >= 16)
        return 16;
#endif // __AVX512F__
    if (num_query_heads_per_kv_head >= 8)
        return 8;
#else
    (void)num_query_heads_per_kv_head;
#endif // __AVX__
    return 4;
#else
    (void)num_query_heads_per_kv_head;
    return 1;
#endif // __SSE2__
}

static int sdpa_decode_block_n(int head_dim, int value_dim, int key_seqlen, int query_storage_size, int key_storage_size, int value_storage_size, int mask_storage_size, int block_q)
{
    size_t l2_cache_size = get_cpu_level2_cache_size();
    if (l2_cache_size == 0)
        l2_cache_size = 256 * 1024;

    const size_t cache_budget = l2_cache_size * 3 / 4;
    const size_t fixed_size = (size_t)block_q * (head_dim * query_storage_size + value_dim * sizeof(float));
    const size_t size_per_token = (size_t)head_dim * key_storage_size + (size_t)value_dim * value_storage_size + (size_t)block_q * (sizeof(float) + mask_storage_size);

    int block_n = 64;
    if (fixed_size + size_per_token * 256 <= cache_budget)
        block_n = 256;
    else if (fixed_size + size_per_token * 128 <= cache_budget)
        block_n = 128;

    return std::min(block_n, key_seqlen);
}

static void sdpa_decode_pack_query_fp32(const Mat& query, Mat& queryT, float scale, int q0, int max_qq)
{
#if __SSE2__
    const int head_dim = query.w;
    float* queryT_ptr = queryT;
    int qq = 0;
#if __AVX__
#if __AVX512F__
    for (; qq + 15 < max_qq; qq += 16)
    {
        const int q = q0 + qq;
        float* pQ = queryT_ptr + (size_t)qq * head_dim;
        const float* qptr0 = query.channel(q);
        const float* qptr1 = query.channel(q + 1);
        const float* qptr2 = query.channel(q + 2);
        const float* qptr3 = query.channel(q + 3);
        const float* qptr4 = query.channel(q + 4);
        const float* qptr5 = query.channel(q + 5);
        const float* qptr6 = query.channel(q + 6);
        const float* qptr7 = query.channel(q + 7);
        const float* qptr8 = query.channel(q + 8);
        const float* qptr9 = query.channel(q + 9);
        const float* qptra = query.channel(q + 10);
        const float* qptrb = query.channel(q + 11);
        const float* qptrc = query.channel(q + 12);
        const float* qptrd = query.channel(q + 13);
        const float* qptre = query.channel(q + 14);
        const float* qptrf = query.channel(q + 15);

        const __m512 _scale = _mm512_set1_ps(scale);
        int d = 0;
        for (; d + 15 < head_dim; d += 16)
        {
            __m512 _r0 = _mm512_loadu_ps(qptr0 + d);
            __m512 _r1 = _mm512_loadu_ps(qptr1 + d);
            __m512 _r2 = _mm512_loadu_ps(qptr2 + d);
            __m512 _r3 = _mm512_loadu_ps(qptr3 + d);
            __m512 _r4 = _mm512_loadu_ps(qptr4 + d);
            __m512 _r5 = _mm512_loadu_ps(qptr5 + d);
            __m512 _r6 = _mm512_loadu_ps(qptr6 + d);
            __m512 _r7 = _mm512_loadu_ps(qptr7 + d);
            __m512 _r8 = _mm512_loadu_ps(qptr8 + d);
            __m512 _r9 = _mm512_loadu_ps(qptr9 + d);
            __m512 _ra = _mm512_loadu_ps(qptra + d);
            __m512 _rb = _mm512_loadu_ps(qptrb + d);
            __m512 _rc = _mm512_loadu_ps(qptrc + d);
            __m512 _rd = _mm512_loadu_ps(qptrd + d);
            __m512 _re = _mm512_loadu_ps(qptre + d);
            __m512 _rf = _mm512_loadu_ps(qptrf + d);
            transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
            _mm512_storeu_ps(pQ, _mm512_mul_ps(_r0, _scale));
            _mm512_storeu_ps(pQ + 16, _mm512_mul_ps(_r1, _scale));
            _mm512_storeu_ps(pQ + 32, _mm512_mul_ps(_r2, _scale));
            _mm512_storeu_ps(pQ + 48, _mm512_mul_ps(_r3, _scale));
            _mm512_storeu_ps(pQ + 64, _mm512_mul_ps(_r4, _scale));
            _mm512_storeu_ps(pQ + 80, _mm512_mul_ps(_r5, _scale));
            _mm512_storeu_ps(pQ + 96, _mm512_mul_ps(_r6, _scale));
            _mm512_storeu_ps(pQ + 112, _mm512_mul_ps(_r7, _scale));
            _mm512_storeu_ps(pQ + 128, _mm512_mul_ps(_r8, _scale));
            _mm512_storeu_ps(pQ + 144, _mm512_mul_ps(_r9, _scale));
            _mm512_storeu_ps(pQ + 160, _mm512_mul_ps(_ra, _scale));
            _mm512_storeu_ps(pQ + 176, _mm512_mul_ps(_rb, _scale));
            _mm512_storeu_ps(pQ + 192, _mm512_mul_ps(_rc, _scale));
            _mm512_storeu_ps(pQ + 208, _mm512_mul_ps(_rd, _scale));
            _mm512_storeu_ps(pQ + 224, _mm512_mul_ps(_re, _scale));
            _mm512_storeu_ps(pQ + 240, _mm512_mul_ps(_rf, _scale));
            pQ += 256;
        }
        for (; d < head_dim; d++)
        {
            pQ[0] = qptr0[d] * scale;
            pQ[1] = qptr1[d] * scale;
            pQ[2] = qptr2[d] * scale;
            pQ[3] = qptr3[d] * scale;
            pQ[4] = qptr4[d] * scale;
            pQ[5] = qptr5[d] * scale;
            pQ[6] = qptr6[d] * scale;
            pQ[7] = qptr7[d] * scale;
            pQ[8] = qptr8[d] * scale;
            pQ[9] = qptr9[d] * scale;
            pQ[10] = qptra[d] * scale;
            pQ[11] = qptrb[d] * scale;
            pQ[12] = qptrc[d] * scale;
            pQ[13] = qptrd[d] * scale;
            pQ[14] = qptre[d] * scale;
            pQ[15] = qptrf[d] * scale;
            pQ += 16;
        }
    }
#endif // __AVX512F__
    for (; qq + 7 < max_qq; qq += 8)
    {
        const int q = q0 + qq;
        float* pQ = queryT_ptr + (size_t)qq * head_dim;
        const float* qptr0 = query.channel(q);
        const float* qptr1 = query.channel(q + 1);
        const float* qptr2 = query.channel(q + 2);
        const float* qptr3 = query.channel(q + 3);
        const float* qptr4 = query.channel(q + 4);
        const float* qptr5 = query.channel(q + 5);
        const float* qptr6 = query.channel(q + 6);
        const float* qptr7 = query.channel(q + 7);

        const __m256 _scale = _mm256_set1_ps(scale);
        int d = 0;
        for (; d + 7 < head_dim; d += 8)
        {
            __m256 _r0 = _mm256_loadu_ps(qptr0 + d);
            __m256 _r1 = _mm256_loadu_ps(qptr1 + d);
            __m256 _r2 = _mm256_loadu_ps(qptr2 + d);
            __m256 _r3 = _mm256_loadu_ps(qptr3 + d);
            __m256 _r4 = _mm256_loadu_ps(qptr4 + d);
            __m256 _r5 = _mm256_loadu_ps(qptr5 + d);
            __m256 _r6 = _mm256_loadu_ps(qptr6 + d);
            __m256 _r7 = _mm256_loadu_ps(qptr7 + d);
            transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            _mm256_storeu_ps(pQ, _mm256_mul_ps(_r0, _scale));
            _mm256_storeu_ps(pQ + 8, _mm256_mul_ps(_r1, _scale));
            _mm256_storeu_ps(pQ + 16, _mm256_mul_ps(_r2, _scale));
            _mm256_storeu_ps(pQ + 24, _mm256_mul_ps(_r3, _scale));
            _mm256_storeu_ps(pQ + 32, _mm256_mul_ps(_r4, _scale));
            _mm256_storeu_ps(pQ + 40, _mm256_mul_ps(_r5, _scale));
            _mm256_storeu_ps(pQ + 48, _mm256_mul_ps(_r6, _scale));
            _mm256_storeu_ps(pQ + 56, _mm256_mul_ps(_r7, _scale));
            pQ += 64;
        }
        for (; d < head_dim; d++)
        {
            pQ[0] = qptr0[d] * scale;
            pQ[1] = qptr1[d] * scale;
            pQ[2] = qptr2[d] * scale;
            pQ[3] = qptr3[d] * scale;
            pQ[4] = qptr4[d] * scale;
            pQ[5] = qptr5[d] * scale;
            pQ[6] = qptr6[d] * scale;
            pQ[7] = qptr7[d] * scale;
            pQ += 8;
        }
    }
#endif // __AVX__
    for (; qq + 3 < max_qq; qq += 4)
    {
        const int q = q0 + qq;
        float* pQ = queryT_ptr + (size_t)qq * head_dim;
        const float* qptr0 = query.channel(q);
        const float* qptr1 = query.channel(q + 1);
        const float* qptr2 = query.channel(q + 2);
        const float* qptr3 = query.channel(q + 3);

        const __m128 _scale = _mm_set1_ps(scale);
        int d = 0;
        for (; d + 3 < head_dim; d += 4)
        {
            __m128 _r0 = _mm_loadu_ps(qptr0 + d);
            __m128 _r1 = _mm_loadu_ps(qptr1 + d);
            __m128 _r2 = _mm_loadu_ps(qptr2 + d);
            __m128 _r3 = _mm_loadu_ps(qptr3 + d);
            _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
            _mm_storeu_ps(pQ, _mm_mul_ps(_r0, _scale));
            _mm_storeu_ps(pQ + 4, _mm_mul_ps(_r1, _scale));
            _mm_storeu_ps(pQ + 8, _mm_mul_ps(_r2, _scale));
            _mm_storeu_ps(pQ + 12, _mm_mul_ps(_r3, _scale));
            pQ += 16;
        }
        for (; d < head_dim; d++)
        {
            pQ[0] = qptr0[d] * scale;
            pQ[1] = qptr1[d] * scale;
            pQ[2] = qptr2[d] * scale;
            pQ[3] = qptr3[d] * scale;
            pQ += 4;
        }
    }
#else
    (void)query;
    (void)queryT;
    (void)scale;
    (void)q0;
    (void)max_qq;
#endif // __SSE2__
}

static void sdpa_decode_tile_fp32(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, int q0, int max_qq, int g, int n_begin, int n_end, int block_n, const Mat& packed_query, Mat& workspace, Mat& state)
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
        const float* mask = 0;
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
            queryT_blob = Mat(head_dim * 16, outT + value_dim * 16, 4u);
            sdpa_decode_pack_query_fp32(query, queryT_blob, scale, q, 16);
        }
        const float* queryT = queryT_blob;
        if (!packed_query.empty())
            queryT += (size_t)qq * head_dim;
        memset(outT, 0, (size_t)value_dim * 16 * sizeof(float));

        const Mat key_head = key.channel(g);
        const Mat value_head = value.channel(g);
        __m512 _m = _mm512_set1_ps(-FLT_MAX);
        __m512 _l = _mm512_setzero_ps();

        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
            __m512 _block_max = _mm512_set1_ps(-FLT_MAX);
            const float* pK = key_head.row(n);
            float* pS = scoreT;
            const float* pM = mask ? mask + n : 0;
            for (int j = 0; j < max_jj; j++)
            {
                __m512 _sum0 = _mm512_setzero_ps();
                __m512 _sum1 = _mm512_setzero_ps();
                __m512 _sum2 = _mm512_setzero_ps();
                __m512 _sum3 = _mm512_setzero_ps();
                const float* pQ = queryT;
                int d = 0;
                for (; d + 3 < head_dim; d += 4)
                {
                    _sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(pQ), _mm512_set1_ps(pK[0]), _sum0);
                    _sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(pQ + 16), _mm512_set1_ps(pK[1]), _sum1);
                    _sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(pQ + 32), _mm512_set1_ps(pK[2]), _sum2);
                    _sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(pQ + 48), _mm512_set1_ps(pK[3]), _sum3);
                    pQ += 64;
                    pK += 4;
                }
                for (; d < head_dim; d++)
                {
                    _sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(pQ), _mm512_set1_ps(*pK), _sum0);
                    pQ += 16;
                    pK++;
                }
                __m512 _score = _mm512_add_ps(_mm512_add_ps(_sum0, _sum1), _mm512_add_ps(_sum2, _sum3));
                if (pM)
                {
                    __m512 _mask;
                    if (mask_per_head)
                        _mask = _mm512_i32gather_ps(_mask_index, pM, sizeof(float));
                    else
                        _mask = _mm512_set1_ps(*pM);
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
            const float* valueptr = value_head.row(n);
            int d = 0;
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
                const float* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    __m512 _p = _mm512_loadu_ps(pS);
                    _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[0]), _out0);
                    _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[1]), _out1);
                    _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[2]), _out2);
                    _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[3]), _out3);
                    _out4 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[4]), _out4);
                    _out5 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[5]), _out5);
                    _out6 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[6]), _out6);
                    _out7 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[7]), _out7);
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
            for (; d + 3 < value_dim; d += 4)
            {
                __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 0), _alpha);
                __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 16), _alpha);
                __m512 _out2 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 32), _alpha);
                __m512 _out3 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 48), _alpha);
                const float* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    __m512 _p = _mm512_loadu_ps(pS);
                    _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[0]), _out0);
                    _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[1]), _out1);
                    _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[2]), _out2);
                    _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[3]), _out3);
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
                const float* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    _out = _mm512_fmadd_ps(_mm512_loadu_ps(pS), _mm512_set1_ps(*pV), _out);
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
                mask_head0 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q + 0 : 0);
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
        const float* mask0 = mask_head0.empty() ? 0 : mask_head0;
        const float* mask1 = mask_head1.empty() ? 0 : mask_head1;
        const float* mask2 = mask_head2.empty() ? 0 : mask_head2;
        const float* mask3 = mask_head3.empty() ? 0 : mask_head3;
        const float* mask4 = mask_head4.empty() ? 0 : mask_head4;
        const float* mask5 = mask_head5.empty() ? 0 : mask_head5;
        const float* mask6 = mask_head6.empty() ? 0 : mask_head6;
        const float* mask7 = mask_head7.empty() ? 0 : mask_head7;

        float* scoreT = workspace;
        float* outT = scoreT + block_n * 8;
        Mat queryT_blob = packed_query;
        if (queryT_blob.empty())
        {
            queryT_blob = Mat(head_dim * 8, outT + value_dim * 8, 4u);
            sdpa_decode_pack_query_fp32(query, queryT_blob, scale, q, 8);
        }
        const float* queryT = queryT_blob;
        if (!packed_query.empty())
            queryT += (size_t)qq * head_dim;
        memset(outT, 0, (size_t)value_dim * 8 * sizeof(float));

        const Mat key_head = key.channel(g);
        const Mat value_head = value.channel(g);
        __m256 _m = _mm256_set1_ps(-FLT_MAX);
        __m256 _l = _mm256_setzero_ps();

        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
            __m256 _block_max = _mm256_set1_ps(-FLT_MAX);
            const float* pK = key_head.row(n);
            float* pS = scoreT;
            const float* pM0 = mask0 ? mask0 + n : 0;
            const float* pM1 = mask1 ? mask1 + n : 0;
            const float* pM2 = mask2 ? mask2 + n : 0;
            const float* pM3 = mask3 ? mask3 + n : 0;
            const float* pM4 = mask4 ? mask4 + n : 0;
            const float* pM5 = mask5 ? mask5 + n : 0;
            const float* pM6 = mask6 ? mask6 + n : 0;
            const float* pM7 = mask7 ? mask7 + n : 0;
            for (int j = 0; j < max_jj; j++)
            {
                __m256 _sum0 = _mm256_setzero_ps();
                __m256 _sum1 = _mm256_setzero_ps();
                __m256 _sum2 = _mm256_setzero_ps();
                __m256 _sum3 = _mm256_setzero_ps();
                const float* pQ = queryT;
                int d = 0;
                for (; d + 3 < head_dim; d += 4)
                {
                    _sum0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pQ), _mm256_set1_ps(pK[0]), _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pQ + 8), _mm256_set1_ps(pK[1]), _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pQ + 16), _mm256_set1_ps(pK[2]), _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pQ + 24), _mm256_set1_ps(pK[3]), _sum3);
                    pQ += 32;
                    pK += 4;
                }
                for (; d < head_dim; d++)
                {
                    _sum0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pQ), _mm256_set1_ps(*pK), _sum0);
                    pQ += 8;
                    pK++;
                }
                __m256 _score = _mm256_add_ps(_mm256_add_ps(_sum0, _sum1), _mm256_add_ps(_sum2, _sum3));
                if (pM0)
                {
                    _score = _mm256_add_ps(_score, _mm256_set_ps(*pM7, *pM6, *pM5, *pM4, *pM3, *pM2, *pM1, *pM0));
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
                __m256 _p0 = _mm256_sub_ps(_mm256_loadu_ps(scoreptr), _m_new);
                _p0 = exp256_ps(_p0);
                _mm256_storeu_ps(scoreptr, _p0);
                _sum0 = _mm256_add_ps(_sum0, _p0);
                __m256 _p1 = _mm256_sub_ps(_mm256_loadu_ps(scoreptr + 8), _m_new);
                _p1 = exp256_ps(_p1);
                _mm256_storeu_ps(scoreptr + 8, _p1);
                _sum1 = _mm256_add_ps(_sum1, _p1);
                __m256 _p2 = _mm256_sub_ps(_mm256_loadu_ps(scoreptr + 16), _m_new);
                _p2 = exp256_ps(_p2);
                _mm256_storeu_ps(scoreptr + 16, _p2);
                _sum2 = _mm256_add_ps(_sum2, _p2);
                __m256 _p3 = _mm256_sub_ps(_mm256_loadu_ps(scoreptr + 24), _m_new);
                _p3 = exp256_ps(_p3);
                _mm256_storeu_ps(scoreptr + 24, _p3);
                _sum3 = _mm256_add_ps(_sum3, _p3);
                scoreptr += 32;
            }
            for (; j < max_jj; j++)
            {
                __m256 _p = _mm256_sub_ps(_mm256_loadu_ps(scoreptr), _m_new);
                _p = exp256_ps(_p);
                _mm256_storeu_ps(scoreptr, _p);
                scoreptr += 8;
                _sum0 = _mm256_add_ps(_sum0, _p);
            }
            __m256 _sum = _mm256_add_ps(_mm256_add_ps(_sum0, _sum1), _mm256_add_ps(_sum2, _sum3));
            _l = _mm256_add_ps(_mm256_mul_ps(_l, _alpha), _sum);
            _m = _m_new;

            float* outptr = outT;
            const float* valueptr = value_head.row(n);
            int d = 0;
            for (; d + 7 < value_dim; d += 8)
            {
                __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 0), _alpha);
                __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 8), _alpha);
                __m256 _out2 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 16), _alpha);
                __m256 _out3 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 24), _alpha);
                __m256 _out4 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 32), _alpha);
                __m256 _out5 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 40), _alpha);
                __m256 _out6 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 48), _alpha);
                __m256 _out7 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 56), _alpha);
                const float* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    __m256 _p = _mm256_loadu_ps(pS);
                    _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[0]), _out0);
                    _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[1]), _out1);
                    _out2 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[2]), _out2);
                    _out3 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[3]), _out3);
                    _out4 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[4]), _out4);
                    _out5 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[5]), _out5);
                    _out6 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[6]), _out6);
                    _out7 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[7]), _out7);
                    pS += 8;
                    pV += value_dim;
                }
                _mm256_storeu_ps(outptr + 0, _out0);
                _mm256_storeu_ps(outptr + 8, _out1);
                _mm256_storeu_ps(outptr + 16, _out2);
                _mm256_storeu_ps(outptr + 24, _out3);
                _mm256_storeu_ps(outptr + 32, _out4);
                _mm256_storeu_ps(outptr + 40, _out5);
                _mm256_storeu_ps(outptr + 48, _out6);
                _mm256_storeu_ps(outptr + 56, _out7);
                outptr += 64;
                valueptr += 8;
            }
            for (; d + 3 < value_dim; d += 4)
            {
                __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 0), _alpha);
                __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 8), _alpha);
                __m256 _out2 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 16), _alpha);
                __m256 _out3 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 24), _alpha);
                const float* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    __m256 _p = _mm256_loadu_ps(pS);
                    _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[0]), _out0);
                    _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[1]), _out1);
                    _out2 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[2]), _out2);
                    _out3 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[3]), _out3);
                    pS += 8;
                    pV += value_dim;
                }
                _mm256_storeu_ps(outptr + 0, _out0);
                _mm256_storeu_ps(outptr + 8, _out1);
                _mm256_storeu_ps(outptr + 16, _out2);
                _mm256_storeu_ps(outptr + 24, _out3);
                outptr += 32;
                valueptr += 4;
            }
            for (; d < value_dim; d++)
            {
                __m256 _out = _mm256_mul_ps(_mm256_loadu_ps(outptr), _alpha);
                const float* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pS), _mm256_set1_ps(*pV), _out);
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
            float* output0 = top_blob.channel(q + 0);
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

        const float* mask0 = 0;
        const float* mask1 = 0;
        const float* mask2 = 0;
        const float* mask3 = 0;
        if (!attn_mask_blob.empty())
        {
            if (attn_mask_blob.dims == 3)
            {
                mask0 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q : 0);
                mask1 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q + 1 : 0);
                mask2 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q + 2 : 0);
                mask3 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q + 3 : 0);
            }
            else
            {
                mask0 = attn_mask_blob;
                mask1 = attn_mask_blob;
                mask2 = attn_mask_blob;
                mask3 = attn_mask_blob;
            }
        }

        float* scoreT = workspace;
        float* outT = scoreT + block_n * 4;
        Mat queryT_blob = packed_query;
        if (queryT_blob.empty())
        {
            queryT_blob = Mat(head_dim * 4, outT + value_dim * 4, 4u);
            sdpa_decode_pack_query_fp32(query, queryT_blob, scale, q, 4);
        }
        const float* queryT = queryT_blob;
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
                const float* maskptr0 = mask0 ? mask0 + n : 0;
                const float* maskptr1 = mask1 ? mask1 + n : 0;
                const float* maskptr2 = mask2 ? mask2 + n : 0;
                const float* maskptr3 = mask3 ? mask3 + n : 0;
                int j = 0;
#if __AVX__
#if __AVX512F__
                for (; j + 3 < max_jj; j += 4)
                {
                    const float* pK0 = key_head.row(n + j);
                    const float* pK1 = key_head.row(n + j + 1);
                    const float* pK2 = key_head.row(n + j + 2);
                    const float* pK3 = key_head.row(n + j + 3);
                    const float* pQ = queryT;
                    __m512 _sum0 = _mm512_setzero_ps();
                    __m512 _sum1 = _mm512_setzero_ps();
                    __m512 _sum2 = _mm512_setzero_ps();
                    __m512 _sum3 = _mm512_setzero_ps();
                    int d = 0;
                    for (; d + 3 < head_dim; d += 4)
                    {
                        __m512 _q0 = _mm512_broadcast_f32x4(_mm_loadu_ps(pQ));
                        __m512 _q1 = _mm512_broadcast_f32x4(_mm_loadu_ps(pQ + 4));
                        __m512 _q2 = _mm512_broadcast_f32x4(_mm_loadu_ps(pQ + 8));
                        __m512 _q3 = _mm512_broadcast_f32x4(_mm_loadu_ps(pQ + 12));
                        __m512 _k0 = _mm512_set_ps(pK3[0], pK3[0], pK3[0], pK3[0], pK2[0], pK2[0], pK2[0], pK2[0], pK1[0], pK1[0], pK1[0], pK1[0], pK0[0], pK0[0], pK0[0], pK0[0]);
                        __m512 _k1 = _mm512_set_ps(pK3[1], pK3[1], pK3[1], pK3[1], pK2[1], pK2[1], pK2[1], pK2[1], pK1[1], pK1[1], pK1[1], pK1[1], pK0[1], pK0[1], pK0[1], pK0[1]);
                        __m512 _k2 = _mm512_set_ps(pK3[2], pK3[2], pK3[2], pK3[2], pK2[2], pK2[2], pK2[2], pK2[2], pK1[2], pK1[2], pK1[2], pK1[2], pK0[2], pK0[2], pK0[2], pK0[2]);
                        __m512 _k3 = _mm512_set_ps(pK3[3], pK3[3], pK3[3], pK3[3], pK2[3], pK2[3], pK2[3], pK2[3], pK1[3], pK1[3], pK1[3], pK1[3], pK0[3], pK0[3], pK0[3], pK0[3]);
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
                        __m512 _q = _mm512_broadcast_f32x4(_mm_loadu_ps(pQ));
                        __m512 _k = _mm512_set_ps(*pK3, *pK3, *pK3, *pK3, *pK2, *pK2, *pK2, *pK2, *pK1, *pK1, *pK1, *pK1, *pK0, *pK0, *pK0, *pK0);
                        _sum0 = _mm512_fmadd_ps(_q, _k, _sum0);
                        pQ += 4;
                        pK0++;
                        pK1++;
                        pK2++;
                        pK3++;
                    }
                    __m512 _score = _mm512_add_ps(_mm512_add_ps(_sum0, _sum1), _mm512_add_ps(_sum2, _sum3));
                    if (maskptr0)
                    {
                        __m512 _mask = _mm512_set_ps(maskptr3[3], maskptr2[3], maskptr1[3], maskptr0[3], maskptr3[2], maskptr2[2], maskptr1[2], maskptr0[2], maskptr3[1], maskptr2[1], maskptr1[1], maskptr0[1], maskptr3[0], maskptr2[0], maskptr1[0], maskptr0[0]);
                        _score = _mm512_add_ps(_score, _mask);
                        maskptr0 += 4;
                        maskptr1 += 4;
                        maskptr2 += 4;
                        maskptr3 += 4;
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
                    const float* pK0 = key_head.row(n + j);
                    const float* pK1 = key_head.row(n + j + 1);
                    const float* pQ = queryT;
                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    int d = 0;
                    for (; d + 3 < head_dim; d += 4)
                    {
                        __m256 _q0 = _mm256_broadcast_ps((const __m128*)pQ);
                        __m256 _q1 = _mm256_broadcast_ps((const __m128*)(pQ + 4));
                        __m256 _q2 = _mm256_broadcast_ps((const __m128*)(pQ + 8));
                        __m256 _q3 = _mm256_broadcast_ps((const __m128*)(pQ + 12));
                        __m256 _k0 = _mm256_set_ps(pK1[0], pK1[0], pK1[0], pK1[0], pK0[0], pK0[0], pK0[0], pK0[0]);
                        __m256 _k1 = _mm256_set_ps(pK1[1], pK1[1], pK1[1], pK1[1], pK0[1], pK0[1], pK0[1], pK0[1]);
                        __m256 _k2 = _mm256_set_ps(pK1[2], pK1[2], pK1[2], pK1[2], pK0[2], pK0[2], pK0[2], pK0[2]);
                        __m256 _k3 = _mm256_set_ps(pK1[3], pK1[3], pK1[3], pK1[3], pK0[3], pK0[3], pK0[3], pK0[3]);
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
                        __m256 _q = _mm256_broadcast_ps((const __m128*)pQ);
                        __m256 _k = _mm256_set_ps(*pK1, *pK1, *pK1, *pK1, *pK0, *pK0, *pK0, *pK0);
                        _sum0 = _mm256_comp_fmadd_ps(_q, _k, _sum0);
                        pQ += 4;
                        pK0++;
                        pK1++;
                    }
                    __m256 _score = _mm256_add_ps(_mm256_add_ps(_sum0, _sum1), _mm256_add_ps(_sum2, _sum3));
                    if (maskptr0)
                    {
                        __m256 _mask = _mm256_set_ps(maskptr3[1], maskptr2[1], maskptr1[1], maskptr0[1], maskptr3[0], maskptr2[0], maskptr1[0], maskptr0[0]);
                        _score = _mm256_add_ps(_score, _mask);
                        maskptr0 += 2;
                        maskptr1 += 2;
                        maskptr2 += 2;
                        maskptr3 += 2;
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
                    const float* pK = key_head.row(n + j);
                    const float* pQ = queryT;
                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();
                    int d = 0;
                    for (; d + 3 < head_dim; d += 4)
                    {
                        _sum0 = _mm_comp_fmadd_ps(_mm_loadu_ps(pQ), _mm_set1_ps(pK[0]), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_mm_loadu_ps(pQ + 4), _mm_set1_ps(pK[1]), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_mm_loadu_ps(pQ + 8), _mm_set1_ps(pK[2]), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_mm_loadu_ps(pQ + 12), _mm_set1_ps(pK[3]), _sum3);
                        pQ += 16;
                        pK += 4;
                    }
                    for (; d < head_dim; d++)
                    {
                        _sum0 = _mm_comp_fmadd_ps(_mm_loadu_ps(pQ), _mm_set1_ps(*pK), _sum0);
                        pQ += 4;
                        pK++;
                    }
                    __m128 _score = _mm_add_ps(_mm_add_ps(_sum0, _sum1), _mm_add_ps(_sum2, _sum3));
                    if (maskptr0)
                    {
                        _score = _mm_add_ps(_score, _mm_set_ps(*maskptr3, *maskptr2, *maskptr1, *maskptr0));
                        maskptr0++;
                        maskptr1++;
                        maskptr2++;
                        maskptr3++;
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
            const float* value = value_head.row(n);
            const float* valueptr = value;
            int d = 0;
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
                const float* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    __m128 _p = _mm_loadu_ps(pS);
                    _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[0]), _out0);
                    _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[1]), _out1);
                    _out2 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[2]), _out2);
                    _out3 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[3]), _out3);
                    _out4 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[4]), _out4);
                    _out5 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[5]), _out5);
                    _out6 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[6]), _out6);
                    _out7 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[7]), _out7);
                    _out8 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[8]), _out8);
                    _out9 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[9]), _out9);
                    _outa = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[10]), _outa);
                    _outb = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[11]), _outb);
                    _outc = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[12]), _outc);
                    _outd = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[13]), _outd);
                    _oute = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[14]), _oute);
                    _outf = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[15]), _outf);
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
                const float* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    __m128 _p = _mm_loadu_ps(pS);
                    _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[0]), _out0);
                    _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[1]), _out1);
                    _out2 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[2]), _out2);
                    _out3 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[3]), _out3);
                    _out4 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[4]), _out4);
                    _out5 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[5]), _out5);
                    _out6 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[6]), _out6);
                    _out7 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[7]), _out7);
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
            for (; d + 3 < value_dim; d += 4)
            {
                __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(outptr), _alpha);
                __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(outptr + 4), _alpha);
                __m128 _out2 = _mm_mul_ps(_mm_loadu_ps(outptr + 8), _alpha);
                __m128 _out3 = _mm_mul_ps(_mm_loadu_ps(outptr + 12), _alpha);

                const float* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    __m128 _p = _mm_loadu_ps(pS);
                    _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[0]), _out0);
                    _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[1]), _out1);
                    _out2 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[2]), _out2);
                    _out3 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[3]), _out3);
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
                const float* pV = valueptr;
                const float* pS = scoreT;
                for (int j = 0; j < max_jj; j++)
                {
                    _out = _mm_comp_fmadd_ps(_mm_loadu_ps(pS), _mm_set1_ps(*pV), _out);
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

        const float* query_ptr = query.channel(q);
        const float* mask = 0;
        if (!attn_mask_blob.empty())
        {
            if (attn_mask_blob.dims == 3)
                mask = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q : 0);
            else
                mask = attn_mask_blob;
        }

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
            const float* pK = key_head.row(n);
            float* pS = score;
            const float* pM = mask ? mask + n : 0;
            for (int j = 0; j < max_jj; j++)
            {
                const float* pQ = query_ptr;
                float sum;
#if __SSE2__
#if __AVX__
#if __AVX512F__
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
                for (; i + 15 < head_dim; i += 16)
                {
                    _sum_avx512 = _mm512_fmadd_ps(_mm512_loadu_ps(pQ), _mm512_loadu_ps(pK), _sum_avx512);
                    pQ += 16;
                    pK += 16;
                }
#endif // __AVX512F__
                for (; i + 7 < head_dim; i += 8)
                {
                    _sum_avx = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pQ), _mm256_loadu_ps(pK), _sum_avx);
                    pQ += 8;
                    pK += 8;
                }
#endif // __AVX__
                for (; i + 3 < head_dim; i += 4)
                {
                    _sum = _mm_comp_fmadd_ps(_mm_loadu_ps(pQ), _mm_loadu_ps(pK), _sum);
                    pQ += 4;
                    pK += 4;
                }
#endif // __SSE2__
                for (; i < head_dim; i++)
                {
                    sum += *pQ * *pK;
                    pQ++;
                    pK++;
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
                float s = sum * scale;
                if (pM)
                {
                    s += *pM;
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

            const float* value_base = value_head.row(n);
            float* outptr = out;
            const float* valueptr = value_base;
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
                const float* pV = valueptr;
                const float* pS = score;
                for (int j = 0; j < max_jj; j++)
                {
                    __m512 _p = _mm512_set1_ps(*pS++);
                    _out0 = _mm512_fmadd_ps(_mm512_loadu_ps(pV), _p, _out0);
                    _out1 = _mm512_fmadd_ps(_mm512_loadu_ps(pV + 16), _p, _out1);
                    _out2 = _mm512_fmadd_ps(_mm512_loadu_ps(pV + 32), _p, _out2);
                    _out3 = _mm512_fmadd_ps(_mm512_loadu_ps(pV + 48), _p, _out3);
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
                const float* pV = valueptr;
                const float* pS = score;
                for (int j = 0; j < max_jj; j++)
                {
                    _out = _mm512_fmadd_ps(_mm512_loadu_ps(pV), _mm512_set1_ps(*pS++), _out);
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
                const float* pV = valueptr;
                const float* pS = score;
                for (int j = 0; j < max_jj; j++)
                {
                    __m256 _p = _mm256_set1_ps(*pS++);
                    _out0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pV), _p, _out0);
                    _out1 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pV + 8), _p, _out1);
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
                const float* pV = valueptr;
                const float* pS = score;
                for (int j = 0; j < max_jj; j++)
                {
                    _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pV), _mm256_set1_ps(*pS++), _out);
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
                const float* pV = valueptr;
                const float* pS = score;
                for (int j = 0; j < max_jj; j++)
                {
                    __m128 _p = _mm_set1_ps(*pS++);
                    _out0 = _mm_comp_fmadd_ps(_mm_loadu_ps(pV), _p, _out0);
                    _out1 = _mm_comp_fmadd_ps(_mm_loadu_ps(pV + 4), _p, _out1);
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
                const float* pV = valueptr;
                const float* pS = score;
                for (int j = 0; j < max_jj; j++)
                {
                    _out = _mm_comp_fmadd_ps(_mm_loadu_ps(pV), _mm_set1_ps(*pS++), _out);
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
                const float* pV = valueptr;
                const float* pS = score;
                for (int j = 0; j < max_jj; j++)
                {
                    sum += *pS++ * *pV;
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

static void sdpa_decode_reduce(const Mat& partials, Mat& top_blob, Mat& workspace, int num_tasks, int num_qblocks, int block_q, int num_kv_chunks, int num_query_heads_per_kv_head, int value_dim, const Option& opt)
{
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int task_id = 0; task_id < num_tasks; task_id++)
    {
        const int g = task_id / num_qblocks;
        const int qblock_id = task_id % num_qblocks;
        const int q0 = g * num_query_heads_per_kv_head + qblock_id * block_q;
        const int max_qq = std::min(num_query_heads_per_kv_head - qblock_id * block_q, block_q);
        int qq = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; qq + 15 < max_qq; qq += 16)
        {
            __m512 _m = _mm512_set1_ps(-FLT_MAX);
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state = partials.channel(task_id * num_kv_chunks + chunk_id);
                state += qq * (value_dim + 2);
                _m = _mm512_max_ps(_m, _mm512_loadu_ps(state));
            }

            Mat outT_tile = workspace.channel(get_omp_thread_num());
            float* outT = outT_tile;
            memset(outT, 0, (size_t)value_dim * 16 * sizeof(float));
            __m512 _l = _mm512_setzero_ps();
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state = partials.channel(task_id * num_kv_chunks + chunk_id);
                state += qq * (value_dim + 2);
                const __m512 _partial_l = _mm512_loadu_ps(state + 16);
                const __mmask16 active = _mm512_cmp_ps_mask(_partial_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                const __m512 _partial_scale = _mm512_maskz_mov_ps(active, exp512_ps(_mm512_maskz_sub_ps(active, _mm512_loadu_ps(state), _m)));
                _l = _mm512_fmadd_ps(_partial_l, _partial_scale, _l);
                float* outptr = outT;
                const float* stateptr = state + 32;
                for (int d = 0; d < value_dim; d++)
                {
                    __m512 _out = _mm512_loadu_ps(outptr);
                    _out = _mm512_fmadd_ps(_mm512_loadu_ps(stateptr), _partial_scale, _out);
                    _mm512_storeu_ps(outptr, _out);
                    outptr += 16;
                    stateptr += 16;
                }
            }

            float* output0 = top_blob.channel(q0 + qq + 0);
            float* output1 = top_blob.channel(q0 + qq + 1);
            float* output2 = top_blob.channel(q0 + qq + 2);
            float* output3 = top_blob.channel(q0 + qq + 3);
            float* output4 = top_blob.channel(q0 + qq + 4);
            float* output5 = top_blob.channel(q0 + qq + 5);
            float* output6 = top_blob.channel(q0 + qq + 6);
            float* output7 = top_blob.channel(q0 + qq + 7);
            float* output8 = top_blob.channel(q0 + qq + 8);
            float* output9 = top_blob.channel(q0 + qq + 9);
            float* outputa = top_blob.channel(q0 + qq + 10);
            float* outputb = top_blob.channel(q0 + qq + 11);
            float* outputc = top_blob.channel(q0 + qq + 12);
            float* outputd = top_blob.channel(q0 + qq + 13);
            float* outpute = top_blob.channel(q0 + qq + 14);
            float* outputf = top_blob.channel(q0 + qq + 15);
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
#endif // __AVX512F__
        for (; qq + 7 < max_qq; qq += 8)
        {
            __m256 _m = _mm256_set1_ps(-FLT_MAX);
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state = partials.channel(task_id * num_kv_chunks + chunk_id);
                state += qq * (value_dim + 2);
                _m = _mm256_max_ps(_m, _mm256_loadu_ps(state));
            }

            Mat outT_tile = workspace.channel(get_omp_thread_num());
            float* outT = outT_tile;
            memset(outT, 0, (size_t)value_dim * 8 * sizeof(float));
            __m256 _l = _mm256_setzero_ps();
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state = partials.channel(task_id * num_kv_chunks + chunk_id);
                state += qq * (value_dim + 2);
                const __m256 _partial_l = _mm256_loadu_ps(state + 8);
                const __m256 _active = _mm256_cmp_ps(_partial_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
                const __m256 _partial_scale = _mm256_and_ps(_active, exp256_ps(_mm256_and_ps(_active, _mm256_sub_ps(_mm256_loadu_ps(state), _m))));
                _l = _mm256_comp_fmadd_ps(_partial_l, _partial_scale, _l);
                float* outptr = outT;
                const float* stateptr = state + 16;
                for (int d = 0; d < value_dim; d++)
                {
                    __m256 _out = _mm256_loadu_ps(outptr);
                    _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(stateptr), _partial_scale, _out);
                    _mm256_storeu_ps(outptr, _out);
                    outptr += 8;
                    stateptr += 8;
                }
            }

            float* output0 = top_blob.channel(q0 + qq + 0);
            float* output1 = top_blob.channel(q0 + qq + 1);
            float* output2 = top_blob.channel(q0 + qq + 2);
            float* output3 = top_blob.channel(q0 + qq + 3);
            float* output4 = top_blob.channel(q0 + qq + 4);
            float* output5 = top_blob.channel(q0 + qq + 5);
            float* output6 = top_blob.channel(q0 + qq + 6);
            float* output7 = top_blob.channel(q0 + qq + 7);
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
#endif // __AVX__
        for (; qq + 3 < max_qq; qq += 4)
        {
            __m128 _m = _mm_set1_ps(-FLT_MAX);
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state = partials.channel(task_id * num_kv_chunks + chunk_id);
                state += qq * (value_dim + 2);
                _m = _mm_max_ps(_m, _mm_loadu_ps(state));
            }

            Mat outT_tile = workspace.channel(get_omp_thread_num());
            float* outT = outT_tile;
            memset(outT, 0, (size_t)value_dim * 4 * sizeof(float));
            __m128 _l = _mm_setzero_ps();
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state = partials.channel(task_id * num_kv_chunks + chunk_id);
                state += qq * (value_dim + 2);
                const __m128 _partial_l = _mm_loadu_ps(state + 4);
                const __m128 _active = _mm_cmpneq_ps(_partial_l, _mm_setzero_ps());
                const __m128 _partial_scale = _mm_and_ps(_active, exp_ps(_mm_and_ps(_active, _mm_sub_ps(_mm_loadu_ps(state), _m))));
                _l = _mm_comp_fmadd_ps(_partial_l, _partial_scale, _l);
                float* outptr = outT;
                const float* stateptr = state + 8;
                for (int d = 0; d < value_dim; d++)
                {
                    __m128 _out = _mm_loadu_ps(outptr);
                    _out = _mm_comp_fmadd_ps(_mm_loadu_ps(stateptr), _partial_scale, _out);
                    _mm_storeu_ps(outptr, _out);
                    outptr += 4;
                    stateptr += 4;
                }
            }

            float* output0 = top_blob.channel(q0 + qq);
            float* output1 = top_blob.channel(q0 + qq + 1);
            float* output2 = top_blob.channel(q0 + qq + 2);
            float* output3 = top_blob.channel(q0 + qq + 3);
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
#endif // __SSE2__

        for (; qq < max_qq; qq++)
        {
            float m = -FLT_MAX;
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state_q = partials.channel(task_id * num_kv_chunks + chunk_id);
                state_q += qq * (value_dim + 2);
                m = std::max(m, state_q[0]);
            }

            float* outptr = top_blob.channel(q0 + qq);
            memset(outptr, 0, value_dim * sizeof(float));
            float l = 0.f;
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state_q = partials.channel(task_id * num_kv_chunks + chunk_id);
                state_q += qq * (value_dim + 2);
                float partial_scale = state_q[1] == 0.f ? 0.f : expf(state_q[0] - m);
                l += state_q[1] * partial_scale;
                {
                    const float* partial_out = state_q + 2;
                    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    __m512 _scale_avx512 = _mm512_set1_ps(partial_scale);
                    for (; i + 15 < value_dim; i += 16)
                        _mm512_storeu_ps(outptr + i, _mm512_fmadd_ps(_mm512_loadu_ps(partial_out + i), _scale_avx512, _mm512_loadu_ps(outptr + i)));
#endif // __AVX512F__
                    __m256 _scale_avx = _mm256_set1_ps(partial_scale);
                    for (; i + 7 < value_dim; i += 8)
                        _mm256_storeu_ps(outptr + i, _mm256_comp_fmadd_ps(_mm256_loadu_ps(partial_out + i), _scale_avx, _mm256_loadu_ps(outptr + i)));
#endif // __AVX__
                    __m128 _scale = _mm_set1_ps(partial_scale);
                    for (; i + 3 < value_dim; i += 4)
                        _mm_storeu_ps(outptr + i, _mm_comp_fmadd_ps(_mm_loadu_ps(partial_out + i), _scale, _mm_loadu_ps(outptr + i)));
#endif // __SSE2__
                    for (; i < value_dim; i++)
                        outptr[i] += partial_out[i] * partial_scale;
                }
            }
            if (l != 0.f)
            {
                float inv_sum = 1.f / l;
                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _inv_sum_avx512 = _mm512_set1_ps(inv_sum);
                for (; i + 15 < value_dim; i += 16)
                    _mm512_storeu_ps(outptr + i, _mm512_mul_ps(_mm512_loadu_ps(outptr + i), _inv_sum_avx512));
#endif // __AVX512F__
                __m256 _inv_sum_avx = _mm256_set1_ps(inv_sum);
                for (; i + 7 < value_dim; i += 8)
                    _mm256_storeu_ps(outptr + i, _mm256_mul_ps(_mm256_loadu_ps(outptr + i), _inv_sum_avx));
#endif // __AVX__
                __m128 _inv_sum = _mm_set1_ps(inv_sum);
                for (; i + 3 < value_dim; i += 4)
                    _mm_storeu_ps(outptr + i, _mm_mul_ps(_mm_loadu_ps(outptr + i), _inv_sum));
#endif // __SSE2__
                for (; i < value_dim; i++)
                    outptr[i] *= inv_sum;
            }
        }
    }
}

static int sdpa_decode_fp32(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
{
    const int num_query_heads = query.c;
    const int num_kv_heads = key.c;
    const int key_seqlen = key.h;
    const int value_dim = value.w;
    const int num_query_heads_per_kv_head = num_query_heads / num_kv_heads;
    const int block_q = sdpa_decode_block_q(num_query_heads_per_kv_head);
    const int block_n = sdpa_decode_block_n(query.w, value_dim, key_seqlen, 4, 4, 4, attn_mask_blob.empty() ? 0 : 4, block_q);
    const int num_qblocks = (num_query_heads_per_kv_head + block_q - 1) / block_q;
    const int num_tasks = num_kv_heads * num_qblocks;
    const int num_key_blocks = (key_seqlen + block_n - 1) / block_n;
    const bool use_packed_query = block_q >= 4 && num_query_heads_per_kv_head >= 4;

    const int num_threads = std::max(opt.num_threads, 1);
    int num_kv_chunks = 1;
    if (num_tasks < num_threads && num_key_blocks >= 2)
    {
        num_kv_chunks = std::min((num_threads + num_tasks - 1) / num_tasks, num_key_blocks);
        num_kv_chunks = std::max(num_kv_chunks, 1);
    }

    Mat packed_query;
    if (num_kv_chunks > 1 && use_packed_query)
    {
        packed_query.create(query.w * block_q, 1, num_tasks, 4u, opt.workspace_allocator);
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
            sdpa_decode_pack_query_fp32(query, queryT, scale, q0, max_qq);
        }
    }

    const int query_workspace_size = use_packed_query ? query.w * block_q : 0;
    const int workspace_size = (block_q * (block_n + value_dim) + query_workspace_size + 15) / 16 * 16;
    Mat workspace(workspace_size, 1, num_threads, 4u, opt.workspace_allocator);
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
        sdpa_decode_tile_fp32(query, key, value, attn_mask_blob, top_blob, scale, q0, max_qq, g, n_begin, n_end, block_n, packed_query_tile, workspace_tile, state);
    }

    if (num_kv_chunks > 1)
        sdpa_decode_reduce(partials, top_blob, workspace, num_tasks, num_qblocks, block_q, num_kv_chunks, num_query_heads_per_kv_head, value_dim, opt);

    return 0;
}

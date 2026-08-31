// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static int sdpa_decode_get_optimal_tile_q(int num_query_heads_per_kv_head, int num_kv_heads, int nT)
{
#if __SSE2__
    int TILE_Q = num_query_heads_per_kv_head >= 4 ? 4 : (num_query_heads_per_kv_head >= 2 ? 2 : 1);
#if __AVX__
    if (num_query_heads_per_kv_head >= 8)
        TILE_Q = 8;
#if __AVX512F__
    if (num_query_heads_per_kv_head >= 16)
        TILE_Q = 16;
#endif // __AVX512F__
#endif // __AVX__

    while (TILE_Q > 1)
    {
        const int num_tasks = num_kv_heads * ((num_query_heads_per_kv_head + TILE_Q - 1) / TILE_Q);
        if (num_tasks >= nT)
            break;

        TILE_Q /= 2;
    }

    return TILE_Q;
#else
    (void)num_query_heads_per_kv_head;
    (void)num_kv_heads;
    (void)nT;
    return 1;
#endif // __SSE2__
}

static int sdpa_decode_get_optimal_tile_n(int head_dim, int value_dim, int key_seqlen, int query_storage_size, int key_storage_size, int value_storage_size, int mask_storage_size, int TILE_Q, int num_tasks, int nT)
{
#if __AVX512F__
    const int tile_n_align = 16;
#elif __AVX__
    const int tile_n_align = 8;
#elif __SSE2__
    const int tile_n_align = 4;
#else
    const int tile_n_align = 1;
#endif

    const size_t l2_cache_size = get_cpu_level2_cache_size();
    const size_t fixed_size = (size_t)TILE_Q * ((size_t)head_dim * query_storage_size + (size_t)value_dim * sizeof(float));
    const size_t size_per_token = (size_t)head_dim * key_storage_size + (size_t)value_dim * value_storage_size + (size_t)TILE_Q * (sizeof(float) + mask_storage_size);

    size_t tile_size = l2_cache_size > fixed_size ? (l2_cache_size - fixed_size) / size_per_token : 0;
    tile_size = std::min(tile_size, (size_t)key_seqlen);
    int TILE_N = (int)tile_size;
    TILE_N = std::max(tile_n_align, TILE_N / tile_n_align * tile_n_align);

    const int cache_blocks = (key_seqlen - 1) / TILE_N + 1;
    const int parallel_blocks = num_tasks < nT ? (nT - 1) / num_tasks + 1 : 1;
    const int max_blocks = (key_seqlen - 1) / tile_n_align + 1;
    const int num_blocks = std::min(std::max(cache_blocks, parallel_blocks), max_blocks);

    TILE_N = (key_seqlen - 1) / num_blocks + 1;
    if (parallel_blocks > cache_blocks)
        TILE_N = std::max(tile_n_align, TILE_N / tile_n_align * tile_n_align);
    else
        TILE_N = (TILE_N + tile_n_align - 1) / tile_n_align * tile_n_align;

    return TILE_N;
}

static void sdpa_decode_pack_query_fp32(const Mat& query, Mat& queryT, float scale, int q0, int max_qq)
{
#if __SSE2__
    const int head_dim = query.w;
    const size_t q_cstep = query.cstep * query.elempack;
    float* queryT_ptr = queryT;
    int qq = 0;
#if __AVX__
#if __AVX512F__
    for (; qq + 15 < max_qq; qq += 16)
    {
        const int q = q0 + qq;
        float* pQ = queryT_ptr + (size_t)qq * head_dim;
        const float* qptr = query.channel(q);

        const __m512 _scale = _mm512_set1_ps(scale);
        int d = 0;
        for (; d + 15 < head_dim; d += 16)
        {
            __m512 _r0 = _mm512_loadu_ps(qptr);
            __m512 _r1 = _mm512_loadu_ps(qptr + q_cstep);
            __m512 _r2 = _mm512_loadu_ps(qptr + q_cstep * 2);
            __m512 _r3 = _mm512_loadu_ps(qptr + q_cstep * 3);
            __m512 _r4 = _mm512_loadu_ps(qptr + q_cstep * 4);
            __m512 _r5 = _mm512_loadu_ps(qptr + q_cstep * 5);
            __m512 _r6 = _mm512_loadu_ps(qptr + q_cstep * 6);
            __m512 _r7 = _mm512_loadu_ps(qptr + q_cstep * 7);
            __m512 _r8 = _mm512_loadu_ps(qptr + q_cstep * 8);
            __m512 _r9 = _mm512_loadu_ps(qptr + q_cstep * 9);
            __m512 _ra = _mm512_loadu_ps(qptr + q_cstep * 10);
            __m512 _rb = _mm512_loadu_ps(qptr + q_cstep * 11);
            __m512 _rc = _mm512_loadu_ps(qptr + q_cstep * 12);
            __m512 _rd = _mm512_loadu_ps(qptr + q_cstep * 13);
            __m512 _re = _mm512_loadu_ps(qptr + q_cstep * 14);
            __m512 _rf = _mm512_loadu_ps(qptr + q_cstep * 15);
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
            qptr += 16;
            pQ += 256;
        }
        for (; d < head_dim; d++)
        {
            pQ[0] = qptr[0] * scale;
            pQ[1] = qptr[q_cstep] * scale;
            pQ[2] = qptr[q_cstep * 2] * scale;
            pQ[3] = qptr[q_cstep * 3] * scale;
            pQ[4] = qptr[q_cstep * 4] * scale;
            pQ[5] = qptr[q_cstep * 5] * scale;
            pQ[6] = qptr[q_cstep * 6] * scale;
            pQ[7] = qptr[q_cstep * 7] * scale;
            pQ[8] = qptr[q_cstep * 8] * scale;
            pQ[9] = qptr[q_cstep * 9] * scale;
            pQ[10] = qptr[q_cstep * 10] * scale;
            pQ[11] = qptr[q_cstep * 11] * scale;
            pQ[12] = qptr[q_cstep * 12] * scale;
            pQ[13] = qptr[q_cstep * 13] * scale;
            pQ[14] = qptr[q_cstep * 14] * scale;
            pQ[15] = qptr[q_cstep * 15] * scale;
            qptr++;
            pQ += 16;
        }
    }
#endif // __AVX512F__
    for (; qq + 7 < max_qq; qq += 8)
    {
        const int q = q0 + qq;
        float* pQ = queryT_ptr + (size_t)qq * head_dim;
        const float* qptr = query.channel(q);

        const __m256 _scale = _mm256_set1_ps(scale);
        int d = 0;
        for (; d + 7 < head_dim; d += 8)
        {
            __m256 _r0 = _mm256_loadu_ps(qptr);
            __m256 _r1 = _mm256_loadu_ps(qptr + q_cstep);
            __m256 _r2 = _mm256_loadu_ps(qptr + q_cstep * 2);
            __m256 _r3 = _mm256_loadu_ps(qptr + q_cstep * 3);
            __m256 _r4 = _mm256_loadu_ps(qptr + q_cstep * 4);
            __m256 _r5 = _mm256_loadu_ps(qptr + q_cstep * 5);
            __m256 _r6 = _mm256_loadu_ps(qptr + q_cstep * 6);
            __m256 _r7 = _mm256_loadu_ps(qptr + q_cstep * 7);
            transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            _mm256_storeu_ps(pQ, _mm256_mul_ps(_r0, _scale));
            _mm256_storeu_ps(pQ + 8, _mm256_mul_ps(_r1, _scale));
            _mm256_storeu_ps(pQ + 16, _mm256_mul_ps(_r2, _scale));
            _mm256_storeu_ps(pQ + 24, _mm256_mul_ps(_r3, _scale));
            _mm256_storeu_ps(pQ + 32, _mm256_mul_ps(_r4, _scale));
            _mm256_storeu_ps(pQ + 40, _mm256_mul_ps(_r5, _scale));
            _mm256_storeu_ps(pQ + 48, _mm256_mul_ps(_r6, _scale));
            _mm256_storeu_ps(pQ + 56, _mm256_mul_ps(_r7, _scale));
            qptr += 8;
            pQ += 64;
        }
        for (; d < head_dim; d++)
        {
            pQ[0] = qptr[0] * scale;
            pQ[1] = qptr[q_cstep] * scale;
            pQ[2] = qptr[q_cstep * 2] * scale;
            pQ[3] = qptr[q_cstep * 3] * scale;
            pQ[4] = qptr[q_cstep * 4] * scale;
            pQ[5] = qptr[q_cstep * 5] * scale;
            pQ[6] = qptr[q_cstep * 6] * scale;
            pQ[7] = qptr[q_cstep * 7] * scale;
            qptr++;
            pQ += 8;
        }
    }
#endif // __AVX__
    for (; qq + 3 < max_qq; qq += 4)
    {
        const int q = q0 + qq;
        float* pQ = queryT_ptr + (size_t)qq * head_dim;
        const float* qptr = query.channel(q);

        const __m128 _scale = _mm_set1_ps(scale);
        int d = 0;
        for (; d + 3 < head_dim; d += 4)
        {
            __m128 _r0 = _mm_loadu_ps(qptr);
            __m128 _r1 = _mm_loadu_ps(qptr + q_cstep);
            __m128 _r2 = _mm_loadu_ps(qptr + q_cstep * 2);
            __m128 _r3 = _mm_loadu_ps(qptr + q_cstep * 3);
            _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
            _mm_storeu_ps(pQ, _mm_mul_ps(_r0, _scale));
            _mm_storeu_ps(pQ + 4, _mm_mul_ps(_r1, _scale));
            _mm_storeu_ps(pQ + 8, _mm_mul_ps(_r2, _scale));
            _mm_storeu_ps(pQ + 12, _mm_mul_ps(_r3, _scale));
            qptr += 4;
            pQ += 16;
        }
        for (; d < head_dim; d++)
        {
            pQ[0] = qptr[0] * scale;
            pQ[1] = qptr[q_cstep] * scale;
            pQ[2] = qptr[q_cstep * 2] * scale;
            pQ[3] = qptr[q_cstep * 3] * scale;
            qptr++;
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
        const int mask_hstep = mask_per_head ? attn_mask_blob.cstep : 0;
        const __m512i _mask_index = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(mask_hstep));

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
                    _score = _mm512_add_ps(_score, _mm512_i32gather_ps(_mask_index, pM, sizeof(float)));
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
#endif // defined(__x86_64__) || defined(_M_X64)
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
            float* output = top_blob.channel(q);
            const size_t output_cstep = top_blob.cstep;
            const __mmask16 nonzero = _mm512_cmp_ps_mask(_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
            const __m512 _out_scale = _mm512_maskz_div_ps(nonzero, _mm512_set1_ps(1.f), _l);

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
                const __m512 _r = _mm512_mul_ps(_mm512_loadu_ps(outptr), _out_scale);
                const __m128 _r0 = _mm512_extractf32x4_ps(_r, 0);
                const __m128 _r1 = _mm512_extractf32x4_ps(_r, 1);
                const __m128 _r2 = _mm512_extractf32x4_ps(_r, 2);
                const __m128 _r3 = _mm512_extractf32x4_ps(_r, 3);
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

        Mat state_q;
        if (!state.empty())
            state_q = state.range(qq * (value_dim + 2), (value_dim + 2) * 8);

        const bool mask_per_head = !attn_mask_blob.empty() && attn_mask_blob.dims == 3 && attn_mask_blob.c > 1;
        const float* mask = 0;
        if (!attn_mask_blob.empty())
        {
            if (attn_mask_blob.dims == 3)
                mask = attn_mask_blob.channel(mask_per_head ? q : 0);
            else
                mask = attn_mask_blob;
        }
        const int mask_hstep = mask_per_head ? attn_mask_blob.cstep : 0;
#if __AVX2__
        const __m256i _mask_index = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(mask_hstep));
#endif // __AVX2__

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
            const float* pM = mask ? mask + n : 0;
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
                if (pM)
                {
#if __AVX2__
                    __m256 _mask = _mm256_i32gather_ps(pM, _mask_index, sizeof(float));
#else
                    __m256 _mask = _mm256_set_ps(pM[mask_hstep * 7], pM[mask_hstep * 6], pM[mask_hstep * 5], pM[mask_hstep * 4], pM[mask_hstep * 3], pM[mask_hstep * 2], pM[mask_hstep], pM[0]);
#endif // __AVX2__
                    _score = _mm256_add_ps(_score, _mask);
                    pM++;
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
#if defined(__x86_64__) || defined(_M_X64)
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
#endif // defined(__x86_64__) || defined(_M_X64)
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
            float* output = top_blob.channel(q);
            const size_t output_cstep = top_blob.cstep;
            const __m256 _nonzero = _mm256_cmp_ps(_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
            const __m256 _denom = _mm256_blendv_ps(_mm256_set1_ps(1.f), _l, _nonzero);
            const __m256 _out_scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(1.f), _denom), _nonzero);

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
                const __m256 _r = _mm256_mul_ps(_mm256_loadu_ps(outptr), _out_scale);
                const __m128 _r0 = _mm256_castps256_ps128(_r);
                const __m128 _r1 = _mm256_extractf128_ps(_r, 1);
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

        Mat state_q;
        if (!state.empty())
            state_q = state.range(qq * (value_dim + 2), (value_dim + 2) * 4);

        const bool mask_per_head = !attn_mask_blob.empty() && attn_mask_blob.dims == 3 && attn_mask_blob.c > 1;
        const float* mask = 0;
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
#if __AVX__
        const size_t key_hstep = (size_t)key_head.w * key_head.elempack;
#endif // __AVX__

        __m128 _m = _mm_set1_ps(-FLT_MAX);
        __m128 _l = _mm_setzero_ps();

        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
            __m128 _block_max = _mm_set1_ps(-FLT_MAX);
            {
                float* pS = scoreT;
                const float* pM = mask ? mask + n : 0;
                int j = 0;
#if __AVX__
#if __AVX512F__
                for (; j + 3 < max_jj; j += 4)
                {
                    const float* pK = key_head.row(n + j);
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
                        __m512 _k0 = _mm512_set_ps(pK[key_hstep * 3], pK[key_hstep * 3], pK[key_hstep * 3], pK[key_hstep * 3], pK[key_hstep * 2], pK[key_hstep * 2], pK[key_hstep * 2], pK[key_hstep * 2], pK[key_hstep], pK[key_hstep], pK[key_hstep], pK[key_hstep], pK[0], pK[0], pK[0], pK[0]);
                        __m512 _k1 = _mm512_set_ps(pK[key_hstep * 3 + 1], pK[key_hstep * 3 + 1], pK[key_hstep * 3 + 1], pK[key_hstep * 3 + 1], pK[key_hstep * 2 + 1], pK[key_hstep * 2 + 1], pK[key_hstep * 2 + 1], pK[key_hstep * 2 + 1], pK[key_hstep + 1], pK[key_hstep + 1], pK[key_hstep + 1], pK[key_hstep + 1], pK[1], pK[1], pK[1], pK[1]);
                        __m512 _k2 = _mm512_set_ps(pK[key_hstep * 3 + 2], pK[key_hstep * 3 + 2], pK[key_hstep * 3 + 2], pK[key_hstep * 3 + 2], pK[key_hstep * 2 + 2], pK[key_hstep * 2 + 2], pK[key_hstep * 2 + 2], pK[key_hstep * 2 + 2], pK[key_hstep + 2], pK[key_hstep + 2], pK[key_hstep + 2], pK[key_hstep + 2], pK[2], pK[2], pK[2], pK[2]);
                        __m512 _k3 = _mm512_set_ps(pK[key_hstep * 3 + 3], pK[key_hstep * 3 + 3], pK[key_hstep * 3 + 3], pK[key_hstep * 3 + 3], pK[key_hstep * 2 + 3], pK[key_hstep * 2 + 3], pK[key_hstep * 2 + 3], pK[key_hstep * 2 + 3], pK[key_hstep + 3], pK[key_hstep + 3], pK[key_hstep + 3], pK[key_hstep + 3], pK[3], pK[3], pK[3], pK[3]);
                        _sum0 = _mm512_fmadd_ps(_q0, _k0, _sum0);
                        _sum1 = _mm512_fmadd_ps(_q1, _k1, _sum1);
                        _sum2 = _mm512_fmadd_ps(_q2, _k2, _sum2);
                        _sum3 = _mm512_fmadd_ps(_q3, _k3, _sum3);
                        pQ += 16;
                        pK += 4;
                    }
                    for (; d < head_dim; d++)
                    {
                        __m512 _q = _mm512_broadcast_f32x4(_mm_loadu_ps(pQ));
                        __m512 _k = _mm512_set_ps(pK[key_hstep * 3], pK[key_hstep * 3], pK[key_hstep * 3], pK[key_hstep * 3], pK[key_hstep * 2], pK[key_hstep * 2], pK[key_hstep * 2], pK[key_hstep * 2], pK[key_hstep], pK[key_hstep], pK[key_hstep], pK[key_hstep], pK[0], pK[0], pK[0], pK[0]);
                        _sum0 = _mm512_fmadd_ps(_q, _k, _sum0);
                        pQ += 4;
                        pK++;
                    }
                    __m512 _score = _mm512_add_ps(_mm512_add_ps(_sum0, _sum1), _mm512_add_ps(_sum2, _sum3));
                    if (pM)
                    {
                        __m512 _mask;
                        if (mask_per_head)
                        {
                            __m128 _m0 = _mm_loadu_ps(pM);
                            __m128 _m1 = _mm_loadu_ps(pM + mask_cstep);
                            __m128 _m2 = _mm_loadu_ps(pM + mask_cstep * 2);
                            __m128 _m3 = _mm_loadu_ps(pM + mask_cstep * 3);
                            _MM_TRANSPOSE4_PS(_m0, _m1, _m2, _m3);
                            _mask = combine4x4_ps(_m0, _m1, _m2, _m3);
                        }
                        else
                        {
                            _mask = combine4x4_ps(_mm_set1_ps(pM[0]), _mm_set1_ps(pM[1]), _mm_set1_ps(pM[2]), _mm_set1_ps(pM[3]));
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
                    const float* pK = key_head.row(n + j);
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
                        __m256 _k0 = _mm256_set_ps(pK[key_hstep], pK[key_hstep], pK[key_hstep], pK[key_hstep], pK[0], pK[0], pK[0], pK[0]);
                        __m256 _k1 = _mm256_set_ps(pK[key_hstep + 1], pK[key_hstep + 1], pK[key_hstep + 1], pK[key_hstep + 1], pK[1], pK[1], pK[1], pK[1]);
                        __m256 _k2 = _mm256_set_ps(pK[key_hstep + 2], pK[key_hstep + 2], pK[key_hstep + 2], pK[key_hstep + 2], pK[2], pK[2], pK[2], pK[2]);
                        __m256 _k3 = _mm256_set_ps(pK[key_hstep + 3], pK[key_hstep + 3], pK[key_hstep + 3], pK[key_hstep + 3], pK[3], pK[3], pK[3], pK[3]);
                        _sum0 = _mm256_comp_fmadd_ps(_q0, _k0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_q1, _k1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_q2, _k2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_q3, _k3, _sum3);
                        pQ += 16;
                        pK += 4;
                    }
                    for (; d < head_dim; d++)
                    {
                        __m256 _q = _mm256_broadcast_ps((const __m128*)pQ);
                        __m256 _k = _mm256_set_ps(pK[key_hstep], pK[key_hstep], pK[key_hstep], pK[key_hstep], pK[0], pK[0], pK[0], pK[0]);
                        _sum0 = _mm256_comp_fmadd_ps(_q, _k, _sum0);
                        pQ += 4;
                        pK++;
                    }
                    __m256 _score = _mm256_add_ps(_mm256_add_ps(_sum0, _sum1), _mm256_add_ps(_sum2, _sum3));
                    if (pM)
                    {
                        __m256 _mask;
                        if (mask_per_head)
                        {
                            __m128 _m0 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)pM);
                            __m128 _m1 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pM + mask_cstep));
                            __m128 _m2 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pM + mask_cstep * 2));
                            __m128 _m3 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(pM + mask_cstep * 3));
                            _MM_TRANSPOSE4_PS(_m0, _m1, _m2, _m3);
                            _mask = combine4x2_ps(_m0, _m1);
                        }
                        else
                        {
                            _mask = combine4x2_ps(_mm_set1_ps(pM[0]), _mm_set1_ps(pM[1]));
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
                    if (pM)
                    {
                        _score = _mm_add_ps(_score, _mm_set_ps(pM[mask_cstep * 3], pM[mask_cstep * 2], pM[mask_cstep], pM[0]));
                        pM++;
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
#endif // defined(__x86_64__) || defined(_M_X64)
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
            float* output = top_blob.channel(q);
            const size_t output_cstep = top_blob.cstep;
            const __m128 _nonzero = _mm_cmpneq_ps(_l, _mm_setzero_ps());
            const __m128 _denom = _mm_or_ps(_mm_and_ps(_nonzero, _l), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
            const __m128 _out_scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(1.f), _denom), _nonzero);

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
                const __m128 _r = _mm_mul_ps(_mm_loadu_ps(outptr), _out_scale);
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
#if !__SSE2__
    (void)workspace;
#endif // !__SSE2__

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

            float* output = top_blob.channel(q0 + qq);
            const size_t output_cstep = top_blob.cstep;
            const __mmask16 nonzero = _mm512_cmp_ps_mask(_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
            const __m512 _out_scale = _mm512_maskz_div_ps(nonzero, _mm512_set1_ps(1.f), _l);

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
                const __m512 _r = _mm512_mul_ps(_mm512_loadu_ps(outptr), _out_scale);
                const __m128 _r0 = _mm512_extractf32x4_ps(_r, 0);
                const __m128 _r1 = _mm512_extractf32x4_ps(_r, 1);
                const __m128 _r2 = _mm512_extractf32x4_ps(_r, 2);
                const __m128 _r3 = _mm512_extractf32x4_ps(_r, 3);
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

            float* output = top_blob.channel(q0 + qq);
            const size_t output_cstep = top_blob.cstep;
            const __m256 _nonzero = _mm256_cmp_ps(_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
            const __m256 _denom = _mm256_blendv_ps(_mm256_set1_ps(1.f), _l, _nonzero);
            const __m256 _out_scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(1.f), _denom), _nonzero);

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
                const __m256 _r = _mm256_mul_ps(_mm256_loadu_ps(outptr), _out_scale);
                const __m128 _r0 = _mm256_castps256_ps128(_r);
                const __m128 _r1 = _mm256_extractf128_ps(_r, 1);
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

            float* output = top_blob.channel(q0 + qq);
            const size_t output_cstep = top_blob.cstep;
            const __m128 _nonzero = _mm_cmpneq_ps(_l, _mm_setzero_ps());
            const __m128 _denom = _mm_or_ps(_mm_and_ps(_nonzero, _l), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
            const __m128 _out_scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(1.f), _denom), _nonzero);

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
                const __m128 _r = _mm_mul_ps(_mm_loadu_ps(outptr), _out_scale);
                *p0 = _mm_cvtss_f32(_r);
                p0[output_cstep] = _mm_cvtss_f32(_mm_shuffle_ps(_r, _r, _MM_SHUFFLE(1, 1, 1, 1)));
                p0[output_cstep * 2] = _mm_cvtss_f32(_mm_movehl_ps(_r, _r));
                p0[output_cstep * 3] = _mm_cvtss_f32(_mm_shuffle_ps(_r, _r, _MM_SHUFFLE(3, 3, 3, 3)));
                p0++;
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
    const int nT = std::max(opt.num_threads, 1);
    const int block_q = sdpa_decode_get_optimal_tile_q(num_query_heads_per_kv_head, num_kv_heads, nT);
    const int num_qblocks = (num_query_heads_per_kv_head + block_q - 1) / block_q;
    const int num_tasks = num_kv_heads * num_qblocks;
    const int block_n = sdpa_decode_get_optimal_tile_n(query.w, value_dim, key_seqlen, 4, 4, 4, attn_mask_blob.empty() ? 0 : 4, block_q, num_tasks, nT);
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
        sdpa_decode_tile_fp32(query, key, value, attn_mask_blob, top_blob, scale, q0, max_qq, g, n_begin, n_end, block_n, packed_query_tile, workspace_tile, state);
    }

    if (num_kv_chunks > 1)
        sdpa_decode_reduce(partials, top_blob, workspace, num_tasks, num_qblocks, block_q, num_kv_chunks, num_query_heads_per_kv_head, value_dim, opt);

    return 0;
}

static void sdpa_decode_kvcache_tile_fp32(const Mat& query, const Mat& key_cache, const Mat& value_cache, const Mat& attn_mask_blob, Mat& top_blob, float scale, int q0, int max_qq, int g, int n_begin, int n_end, int block_n, const Mat& packed_query, Mat& workspace, Mat& state)
{
    (void)packed_query;
    const int head_dim = query.w;
    const int value_dim = value_cache.w;
#if __AVX512F__
    const int NR = 16;
#elif __AVX__
    const int NR = 8;
#elif __SSE2__
    const int NR = 4;
#else
    const int NR = 1;
#endif
    const int query_workspace_size = max_qq * head_dim;
    const int score_workspace_size = max_qq * block_n;
    const int out_workspace_size = max_qq * value_dim;
#if __SSE2__
    Mat queryT = packed_query;
    if (max_qq >= 4 && queryT.empty())
    {
        queryT = workspace.range(0, query_workspace_size);
        sdpa_decode_pack_query_fp32(query, queryT, scale, q0, max_qq);
    }
    const float* queryT_ptr = queryT;
#endif // __SSE2__
    Mat scoreT = workspace.range(query_workspace_size, score_workspace_size);
    Mat outT = workspace.range(query_workspace_size + score_workspace_size, out_workspace_size);
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
        const float* mask = 0;
        bool mask_per_head = false;
        if (!attn_mask_blob.empty())
        {
            if (attn_mask_blob.dims == 3)
            {
                mask_per_head = attn_mask_blob.c > 1;
                mask = attn_mask_blob.channel(mask_per_head ? q0 + qq : 0);
            }
            else
            {
                mask = attn_mask_blob;
            }
        }
        const int mask_hstep = mask_per_head ? (int)attn_mask_blob.cstep : 0;
        const __m512i _mask_index = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(mask_hstep));

        const float* queryT_tile = queryT_ptr + (size_t)qq * head_dim;
        float* scoreT_tile = scoreT_ptr + (size_t)qq * block_n;
        float* outT_tile = outT_ptr + (size_t)qq * value_dim;
        __m512 _m = _mm512_set1_ps(-FLT_MAX);
        __m512 _l = _mm512_setzero_ps();
        memset(outT_tile, 0, (size_t)value_dim * 16 * sizeof(float));

        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
            const float* pQ = queryT_tile;
            float* scoreptr = scoreT_tile;
            const float* pM = mask ? mask + n : 0;
            __m512 _block_max = _mm512_set1_ps(-FLT_MAX);
            float* score_panel = scoreptr;

            for (int jj = 0; jj < max_jj; jj += NR)
            {
                const int max_nn = std::min(NR, max_jj - jj);
                const float* key_panel = (const float*)key_cache_head + (size_t)(n + jj) * head_dim;
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
                    if (pM)
                    {
                        if (mask_per_head)
                        {
                            const float* pM0 = pM;
                            _sum0 = _mm512_loadu_ps(pM0);
                            _sum1 = _mm512_loadu_ps(pM0 + mask_hstep);
                            _sum2 = _mm512_loadu_ps(pM0 + mask_hstep * 2);
                            _sum3 = _mm512_loadu_ps(pM0 + mask_hstep * 3);
                            _sum4 = _mm512_loadu_ps(pM0 + mask_hstep * 4);
                            _sum5 = _mm512_loadu_ps(pM0 + mask_hstep * 5);
                            _sum6 = _mm512_loadu_ps(pM0 + mask_hstep * 6);
                            _sum7 = _mm512_loadu_ps(pM0 + mask_hstep * 7);
                            _sum8 = _mm512_loadu_ps(pM0 + mask_hstep * 8);
                            _sum9 = _mm512_loadu_ps(pM0 + mask_hstep * 9);
                            _suma = _mm512_loadu_ps(pM0 + mask_hstep * 10);
                            _sumb = _mm512_loadu_ps(pM0 + mask_hstep * 11);
                            _sumc = _mm512_loadu_ps(pM0 + mask_hstep * 12);
                            _sumd = _mm512_loadu_ps(pM0 + mask_hstep * 13);
                            _sume = _mm512_loadu_ps(pM0 + mask_hstep * 14);
                            _sumf = _mm512_loadu_ps(pM0 + mask_hstep * 15);
                            transpose16x16_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7, _sum8, _sum9, _suma, _sumb, _sumc, _sumd, _sume, _sumf);
                        }
                        else
                        {
                            _sum0 = _mm512_set1_ps(pM[0]);
                            _sum1 = _mm512_set1_ps(pM[1]);
                            _sum2 = _mm512_set1_ps(pM[2]);
                            _sum3 = _mm512_set1_ps(pM[3]);
                            _sum4 = _mm512_set1_ps(pM[4]);
                            _sum5 = _mm512_set1_ps(pM[5]);
                            _sum6 = _mm512_set1_ps(pM[6]);
                            _sum7 = _mm512_set1_ps(pM[7]);
                            _sum8 = _mm512_set1_ps(pM[8]);
                            _sum9 = _mm512_set1_ps(pM[9]);
                            _suma = _mm512_set1_ps(pM[10]);
                            _sumb = _mm512_set1_ps(pM[11]);
                            _sumc = _mm512_set1_ps(pM[12]);
                            _sumd = _mm512_set1_ps(pM[13]);
                            _sume = _mm512_set1_ps(pM[14]);
                            _sumf = _mm512_set1_ps(pM[15]);
                        }
                        pM += 16;
                    }
                    const float* pK = key_panel + j;
                    const float* pA = pQ;
                    for (int d = 0; d < head_dim; d++)
                    {
                        const __m512 _q = _mm512_loadu_ps(pA);
                        _sum0 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[3]), _sum3);
                        _sum4 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[4]), _sum4);
                        _sum5 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[5]), _sum5);
                        _sum6 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[6]), _sum6);
                        _sum7 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[7]), _sum7);
                        _sum8 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[8]), _sum8);
                        _sum9 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[9]), _sum9);
                        _suma = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[10]), _suma);
                        _sumb = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[11]), _sumb);
                        _sumc = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[12]), _sumc);
                        _sumd = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[13]), _sumd);
                        _sume = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[14]), _sume);
                        _sumf = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[15]), _sumf);
                        pA += 16;
                        pK += NR;
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
                    if (pM)
                    {
                        if (mask_per_head)
                        {
                            const float* pM0 = pM;
                            _sum0 = combine8x2_ps(_mm256_loadu_ps(pM0), _mm256_loadu_ps(pM0 + mask_hstep));
                            _sum1 = combine8x2_ps(_mm256_loadu_ps(pM0 + mask_hstep * 2), _mm256_loadu_ps(pM0 + mask_hstep * 3));
                            _sum2 = combine8x2_ps(_mm256_loadu_ps(pM0 + mask_hstep * 4), _mm256_loadu_ps(pM0 + mask_hstep * 5));
                            _sum3 = combine8x2_ps(_mm256_loadu_ps(pM0 + mask_hstep * 6), _mm256_loadu_ps(pM0 + mask_hstep * 7));
                            _sum4 = combine8x2_ps(_mm256_loadu_ps(pM0 + mask_hstep * 8), _mm256_loadu_ps(pM0 + mask_hstep * 9));
                            _sum5 = combine8x2_ps(_mm256_loadu_ps(pM0 + mask_hstep * 10), _mm256_loadu_ps(pM0 + mask_hstep * 11));
                            _sum6 = combine8x2_ps(_mm256_loadu_ps(pM0 + mask_hstep * 12), _mm256_loadu_ps(pM0 + mask_hstep * 13));
                            _sum7 = combine8x2_ps(_mm256_loadu_ps(pM0 + mask_hstep * 14), _mm256_loadu_ps(pM0 + mask_hstep * 15));
                            transpose16x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
                        }
                        else
                        {
                            _sum0 = _mm512_set1_ps(pM[0]);
                            _sum1 = _mm512_set1_ps(pM[1]);
                            _sum2 = _mm512_set1_ps(pM[2]);
                            _sum3 = _mm512_set1_ps(pM[3]);
                            _sum4 = _mm512_set1_ps(pM[4]);
                            _sum5 = _mm512_set1_ps(pM[5]);
                            _sum6 = _mm512_set1_ps(pM[6]);
                            _sum7 = _mm512_set1_ps(pM[7]);
                        }
                        pM += 8;
                    }
                    const float* pK = key_panel + j;
                    const float* pA = pQ;
                    for (int d = 0; d < head_dim; d++)
                    {
                        const __m512 _q = _mm512_loadu_ps(pA);
                        _sum0 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[3]), _sum3);
                        _sum4 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[4]), _sum4);
                        _sum5 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[5]), _sum5);
                        _sum6 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[6]), _sum6);
                        _sum7 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[7]), _sum7);
                        pA += 16;
                        pK += NR;
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
                    const __m512 _max0 = _mm512_max_ps(_sum0, _sum4);
                    const __m512 _max1 = _mm512_max_ps(_sum1, _sum5);
                    const __m512 _max2 = _mm512_max_ps(_sum2, _sum6);
                    const __m512 _max3 = _mm512_max_ps(_sum3, _sum7);
                    _block_max = _mm512_max_ps(_block_max, _mm512_max_ps(_mm512_max_ps(_max0, _max1), _mm512_max_ps(_max2, _max3)));
                }
#endif // defined(__x86_64__) || defined(_M_X64)
                for (; j + 3 < max_nn; j += 4)
                {
                    const float* pK = key_panel + j;
                    __m512 _sum0 = _mm512_setzero_ps();
                    __m512 _sum1 = _mm512_setzero_ps();
                    __m512 _sum2 = _mm512_setzero_ps();
                    __m512 _sum3 = _mm512_setzero_ps();
                    if (pM)
                    {
                        if (mask_per_head)
                        {
                            const float* pM0 = pM;
                            _sum0 = combine4x4_ps(_mm_loadu_ps(pM0), _mm_loadu_ps(pM0 + mask_hstep), _mm_loadu_ps(pM0 + mask_hstep * 2), _mm_loadu_ps(pM0 + mask_hstep * 3));
                            _sum1 = combine4x4_ps(_mm_loadu_ps(pM0 + mask_hstep * 4), _mm_loadu_ps(pM0 + mask_hstep * 5), _mm_loadu_ps(pM0 + mask_hstep * 6), _mm_loadu_ps(pM0 + mask_hstep * 7));
                            _sum2 = combine4x4_ps(_mm_loadu_ps(pM0 + mask_hstep * 8), _mm_loadu_ps(pM0 + mask_hstep * 9), _mm_loadu_ps(pM0 + mask_hstep * 10), _mm_loadu_ps(pM0 + mask_hstep * 11));
                            _sum3 = combine4x4_ps(_mm_loadu_ps(pM0 + mask_hstep * 12), _mm_loadu_ps(pM0 + mask_hstep * 13), _mm_loadu_ps(pM0 + mask_hstep * 14), _mm_loadu_ps(pM0 + mask_hstep * 15));
                            transpose16x4_ps(_sum0, _sum1, _sum2, _sum3);
                        }
                        else
                        {
                            _sum0 = _mm512_set1_ps(pM[0]);
                            _sum1 = _mm512_set1_ps(pM[1]);
                            _sum2 = _mm512_set1_ps(pM[2]);
                            _sum3 = _mm512_set1_ps(pM[3]);
                        }
                        pM += 4;
                    }
                    const float* pA = pQ;
                    for (int d = 0; d < head_dim; d++)
                    {
                        const __m512 _q = _mm512_loadu_ps(pA);
                        _sum0 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[3]), _sum3);
                        pA += 16;
                        pK += NR;
                    }
                    _mm512_storeu_ps(score_panel, _sum0);
                    _mm512_storeu_ps(score_panel + 16, _sum1);
                    _mm512_storeu_ps(score_panel + 32, _sum2);
                    _mm512_storeu_ps(score_panel + 48, _sum3);
                    score_panel += 64;
                    _block_max = _mm512_max_ps(_block_max, _mm512_max_ps(_mm512_max_ps(_sum0, _sum1), _mm512_max_ps(_sum2, _sum3)));
                }
                for (; j < max_nn; j++)
                {
                    const float* pK = key_panel + j;
                    __m512 _sum = pM ? _mm512_i32gather_ps(_mask_index, pM, sizeof(float)) : _mm512_setzero_ps();
                    if (pM)
                        pM++;
                    const float* pA = pQ;
                    for (int d = 0; d < head_dim; d++)
                    {
                        _sum = _mm512_fmadd_ps(_mm512_loadu_ps(pA), _mm512_set1_ps(*pK), _sum);
                        pA += 16;
                        pK += NR;
                    }
                    _mm512_storeu_ps(score_panel, _sum);
                    score_panel += 16;
                    _block_max = _mm512_max_ps(_block_max, _sum);
                }
            }

            const __m512 _m_new = _mm512_max_ps(_m, _block_max);
            const __mmask16 alpha_active = _mm512_cmp_ps_mask(_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
            const __m512 _alpha = _mm512_maskz_mov_ps(alpha_active, exp512_ps(_mm512_maskz_sub_ps(alpha_active, _m, _m_new)));

            __m512 _sum0 = _mm512_setzero_ps();
            __m512 _sum1 = _mm512_setzero_ps();
            __m512 _sum2 = _mm512_setzero_ps();
            __m512 _sum3 = _mm512_setzero_ps();
            int j = 0;
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
            for (; j < max_jj; j++)
            {
                __m512 _p = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(scoreptr), _m_new));
                _mm512_storeu_ps(scoreptr, _p);
                scoreptr += 16;
                _sum0 = _mm512_add_ps(_sum0, _p);
            }
            _l = _mm512_add_ps(_mm512_mul_ps(_l, _alpha), _mm512_add_ps(_mm512_add_ps(_sum0, _sum1), _mm512_add_ps(_sum2, _sum3)));
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
                    const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 16;
                        const float* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            const __m512 _p = _mm512_loadu_ps(pS);
                            _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[0]), _out0);
                            _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[1]), _out1);
                            _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[2]), _out2);
                            _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[3]), _out3);
                            _out4 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[4]), _out4);
                            _out5 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[5]), _out5);
                            _out6 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[6]), _out6);
                            _out7 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[7]), _out7);
                            _out8 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[8]), _out8);
                            _out9 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[9]), _out9);
                            _outa = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[10]), _outa);
                            _outb = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[11]), _outb);
                            _outc = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[12]), _outc);
                            _outd = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[13]), _outd);
                            _oute = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[14]), _oute);
                            _outf = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[15]), _outf);
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
                    const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 16;
                        const float* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            const __m512 _p = _mm512_loadu_ps(pS);
                            _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[0]), _out0);
                            _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[1]), _out1);
                            _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[2]), _out2);
                            _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[3]), _out3);
                            _out4 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[4]), _out4);
                            _out5 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[5]), _out5);
                            _out6 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[6]), _out6);
                            _out7 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[7]), _out7);
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
                    const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 16;
                        const float* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            const __m512 _p = _mm512_loadu_ps(pS);
                            _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[0]), _out0);
                            _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[1]), _out1);
                            _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[2]), _out2);
                            _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[3]), _out3);
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
                for (; lane < value_panel_width; lane++)
                {
                    __m512 _out = _mm512_mul_ps(_mm512_loadu_ps(outptr), _alpha);
                    const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 16;
                        const float* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            _out = _mm512_fmadd_ps(_mm512_loadu_ps(pS), _mm512_set1_ps(*pV), _out);
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

        if (!state.empty())
        {
            float* stateptr = state;
            stateptr += qq * (value_dim + 2);
            _mm512_storeu_ps(stateptr, _m);
            _mm512_storeu_ps(stateptr + 16, _l);
            memcpy(stateptr + 32, outT_tile, (size_t)value_dim * 16 * sizeof(float));
        }
        else
        {
            float* output = top_blob.channel(q0 + qq);
            const size_t output_cstep = top_blob.cstep;
            const __mmask16 nonzero = _mm512_cmp_ps_mask(_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
            const __m512 _out_scale = _mm512_maskz_div_ps(nonzero, _mm512_set1_ps(1.f), _l);
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
                const __m512 _r = _mm512_mul_ps(_mm512_loadu_ps(pO), _out_scale);
                const __m128 _r0 = _mm512_extractf32x4_ps(_r, 0);
                const __m128 _r1 = _mm512_extractf32x4_ps(_r, 1);
                const __m128 _r2 = _mm512_extractf32x4_ps(_r, 2);
                const __m128 _r3 = _mm512_extractf32x4_ps(_r, 3);
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
        const float* mask = 0;
        if (!attn_mask_blob.empty())
        {
            if (attn_mask_blob.dims == 3)
                mask = attn_mask_blob.channel(mask_per_head ? q0 + qq : 0);
            else
                mask = attn_mask_blob;
        }
        const int mask_cstep = mask_per_head ? attn_mask_blob.cstep : 0;
#if __AVX2__
        const __m256i _mask_index = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(mask_cstep));
#endif // __AVX2__

        const float* queryT_tile = queryT_ptr + (size_t)qq * head_dim;
        float* scoreT_tile = scoreT_ptr + (size_t)qq * block_n;
        float* outT_tile = outT_ptr + (size_t)qq * value_dim;
        __m256 _m = _mm256_set1_ps(-FLT_MAX);
        __m256 _l = _mm256_setzero_ps();
        memset(outT_tile, 0, (size_t)value_dim * 8 * sizeof(float));

        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
            const float* pQ = queryT_tile;
            float* scoreptr = scoreT_tile;
            const float* pM = mask ? mask + n : 0;
            __m256 _block_max = _mm256_set1_ps(-FLT_MAX);
            float* score_panel = scoreptr;

            for (int jj = 0; jj < max_jj; jj += NR)
            {
                const int max_nn = std::min(NR, max_jj - jj);
                const float* key_panel = (const float*)key_cache_head + (size_t)(n + jj) * head_dim;
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
                    if (pM)
                    {
                        if (mask_per_head)
                        {
                            _sum0 = _mm256_loadu_ps(pM);
                            _sum1 = _mm256_loadu_ps(pM + mask_cstep);
                            _sum2 = _mm256_loadu_ps(pM + mask_cstep * 2);
                            _sum3 = _mm256_loadu_ps(pM + mask_cstep * 3);
                            _sum4 = _mm256_loadu_ps(pM + mask_cstep * 4);
                            _sum5 = _mm256_loadu_ps(pM + mask_cstep * 5);
                            _sum6 = _mm256_loadu_ps(pM + mask_cstep * 6);
                            _sum7 = _mm256_loadu_ps(pM + mask_cstep * 7);
                            transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
                        }
                        else
                        {
                            _sum0 = _mm256_set1_ps(pM[0]);
                            _sum1 = _mm256_set1_ps(pM[1]);
                            _sum2 = _mm256_set1_ps(pM[2]);
                            _sum3 = _mm256_set1_ps(pM[3]);
                            _sum4 = _mm256_set1_ps(pM[4]);
                            _sum5 = _mm256_set1_ps(pM[5]);
                            _sum6 = _mm256_set1_ps(pM[6]);
                            _sum7 = _mm256_set1_ps(pM[7]);
                        }
                        pM += 8;
                    }
                    const float* pK = key_panel + j;
                    const float* pA = pQ;
                    for (int d = 0; d < head_dim; d++)
                    {
                        const __m256 _q = _mm256_loadu_ps(pA);
                        _sum0 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[1]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[2]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[3]), _sum3);
                        _sum4 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[4]), _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[5]), _sum5);
                        _sum6 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[6]), _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[7]), _sum7);
                        pA += 8;
                        pK += NR;
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
                    const float* pK = key_panel + j;
                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    if (pM)
                    {
                        if (mask_per_head)
                        {
                            _sum0 = combine4x2_ps(_mm_loadu_ps(pM), _mm_loadu_ps(pM + mask_cstep));
                            _sum1 = combine4x2_ps(_mm_loadu_ps(pM + mask_cstep * 2), _mm_loadu_ps(pM + mask_cstep * 3));
                            _sum2 = combine4x2_ps(_mm_loadu_ps(pM + mask_cstep * 4), _mm_loadu_ps(pM + mask_cstep * 5));
                            _sum3 = combine4x2_ps(_mm_loadu_ps(pM + mask_cstep * 6), _mm_loadu_ps(pM + mask_cstep * 7));
                            transpose8x4_ps(_sum0, _sum1, _sum2, _sum3);
                        }
                        else
                        {
                            _sum0 = _mm256_set1_ps(pM[0]);
                            _sum1 = _mm256_set1_ps(pM[1]);
                            _sum2 = _mm256_set1_ps(pM[2]);
                            _sum3 = _mm256_set1_ps(pM[3]);
                        }
                        pM += 4;
                    }
                    const float* pA = pQ;
                    for (int d = 0; d < head_dim; d++)
                    {
                        const __m256 _q = _mm256_loadu_ps(pA);
                        _sum0 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[1]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[2]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[3]), _sum3);
                        pA += 8;
                        pK += NR;
                    }
                    _mm256_storeu_ps(score_panel, _sum0);
                    _mm256_storeu_ps(score_panel + 8, _sum1);
                    _mm256_storeu_ps(score_panel + 16, _sum2);
                    _mm256_storeu_ps(score_panel + 24, _sum3);
                    score_panel += 32;
                    _block_max = _mm256_max_ps(_block_max, _mm256_max_ps(_mm256_max_ps(_sum0, _sum1), _mm256_max_ps(_sum2, _sum3)));
                }
                for (; j < max_nn; j++)
                {
                    const float* pK = key_panel + j;
                    __m256 _sum = _mm256_setzero_ps();
                    if (pM)
                    {
#if __AVX2__
                        _sum = _mm256_i32gather_ps(pM, _mask_index, sizeof(float));
#else
                        _sum = _mm256_set_ps(pM[mask_cstep * 7], pM[mask_cstep * 6], pM[mask_cstep * 5], pM[mask_cstep * 4], pM[mask_cstep * 3], pM[mask_cstep * 2], pM[mask_cstep], pM[0]);
#endif // __AVX2__
                        pM++;
                    }
                    const float* pA = pQ;
                    for (int d = 0; d < head_dim; d++)
                    {
                        _sum = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pA), _mm256_set1_ps(*pK), _sum);
                        pA += 8;
                        pK += NR;
                    }
                    _mm256_storeu_ps(score_panel, _sum);
                    score_panel += 8;
                    _block_max = _mm256_max_ps(_block_max, _sum);
                }
            }

            const __m256 _m_new = _mm256_max_ps(_m, _block_max);
            const __m256 _alpha_active = _mm256_cmp_ps(_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
            __m256 _alpha = exp256_ps(_mm256_and_ps(_alpha_active, _mm256_sub_ps(_m, _m_new)));
            _alpha = _mm256_and_ps(_alpha, _alpha_active);

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
                scoreptr += 32;
                _sum0 = _mm256_add_ps(_sum0, _p0);
                _sum1 = _mm256_add_ps(_sum1, _p1);
                _sum2 = _mm256_add_ps(_sum2, _p2);
                _sum3 = _mm256_add_ps(_sum3, _p3);
            }
            for (; j < max_jj; j++)
            {
                __m256 _p = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr), _m_new));
                _mm256_storeu_ps(scoreptr, _p);
                scoreptr += 8;
                _sum0 = _mm256_add_ps(_sum0, _p);
            }
            _l = _mm256_add_ps(_mm256_mul_ps(_l, _alpha), _mm256_add_ps(_mm256_add_ps(_sum0, _sum1), _mm256_add_ps(_sum2, _sum3)));
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
                    const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 8;
                        const float* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            const __m256 _p = _mm256_loadu_ps(pS);
                            _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[0]), _out0);
                            _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[1]), _out1);
                            _out2 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[2]), _out2);
                            _out3 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[3]), _out3);
                            _out4 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[4]), _out4);
                            _out5 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[5]), _out5);
                            _out6 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[6]), _out6);
                            _out7 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[7]), _out7);
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
                    const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 8;
                        const float* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            const __m256 _p = _mm256_loadu_ps(pS);
                            _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[0]), _out0);
                            _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[1]), _out1);
                            _out2 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[2]), _out2);
                            _out3 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[3]), _out3);
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
                for (; lane < value_panel_width; lane++)
                {
                    __m256 _out = _mm256_mul_ps(_mm256_loadu_ps(outptr), _alpha);
                    const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 8;
                        const float* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pS), _mm256_set1_ps(*pV), _out);
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

        if (!state.empty())
        {
            float* stateptr = state;
            stateptr += qq * (value_dim + 2);
            _mm256_storeu_ps(stateptr, _m);
            _mm256_storeu_ps(stateptr + 8, _l);
            memcpy(stateptr + 16, outT_tile, (size_t)value_dim * 8 * sizeof(float));
        }
        else
        {
            float* output = top_blob.channel(q0 + qq);
            const size_t output_cstep = top_blob.cstep;
            const __m256 _nonzero = _mm256_cmp_ps(_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
            const __m256 _denom = _mm256_blendv_ps(_mm256_set1_ps(1.f), _l, _nonzero);
            const __m256 _out_scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(1.f), _denom), _nonzero);
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
                const __m256 _r = _mm256_mul_ps(_mm256_loadu_ps(pO), _out_scale);
                const __m128 _r0 = _mm256_castps256_ps128(_r);
                const __m128 _r1 = _mm256_extractf128_ps(_r, 1);
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
        const float* mask = 0;
        if (!attn_mask_blob.empty())
        {
            if (attn_mask_blob.dims == 3)
                mask = attn_mask_blob.channel(mask_per_head ? q0 + qq : 0);
            else
                mask = attn_mask_blob;
        }
        const size_t mask_cstep = mask_per_head ? attn_mask_blob.cstep : 0;

        const float* queryT_tile = queryT_ptr + (size_t)qq * head_dim;
        float* scoreT_tile = scoreT_ptr + (size_t)qq * block_n;
        float* outT_tile = outT_ptr + (size_t)qq * value_dim;
        __m128 _m = _mm_set1_ps(-FLT_MAX);
        __m128 _l = _mm_setzero_ps();
        memset(outT_tile, 0, (size_t)value_dim * 4 * sizeof(float));

        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
            const float* pQ = queryT_tile;
            float* scoreptr = scoreT_tile;
            const float* pM = mask ? mask + n : 0;
            __m128 _block_max = _mm_set1_ps(-FLT_MAX);
            float* score_panel = scoreptr;

            for (int jj = 0; jj < max_jj; jj += NR)
            {
                const int max_nn = std::min(NR, max_jj - jj);
                const float* key_panel = (const float*)key_cache_head + (size_t)(n + jj) * head_dim;
                int j = 0;
                for (; j + 3 < max_nn; j += 4)
                {
                    const float* pK = key_panel + j;
                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();
                    if (pM)
                    {
                        if (mask_per_head)
                        {
                            _sum0 = _mm_loadu_ps(pM);
                            _sum1 = _mm_loadu_ps(pM + mask_cstep);
                            _sum2 = _mm_loadu_ps(pM + mask_cstep * 2);
                            _sum3 = _mm_loadu_ps(pM + mask_cstep * 3);
                            _MM_TRANSPOSE4_PS(_sum0, _sum1, _sum2, _sum3);
                        }
                        else
                        {
                            _sum0 = _mm_set1_ps(pM[0]);
                            _sum1 = _mm_set1_ps(pM[1]);
                            _sum2 = _mm_set1_ps(pM[2]);
                            _sum3 = _mm_set1_ps(pM[3]);
                        }
                        pM += 4;
                    }
                    const float* pA = pQ;
                    for (int d = 0; d < head_dim; d++)
                    {
                        const __m128 _q = _mm_loadu_ps(pA);
                        _sum0 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[0]), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[1]), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[2]), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[3]), _sum3);
                        pA += 4;
                        pK += NR;
                    }
                    _mm_storeu_ps(score_panel, _sum0);
                    _mm_storeu_ps(score_panel + 4, _sum1);
                    _mm_storeu_ps(score_panel + 8, _sum2);
                    _mm_storeu_ps(score_panel + 12, _sum3);
                    score_panel += 16;
                    _block_max = _mm_max_ps(_block_max, _mm_max_ps(_mm_max_ps(_sum0, _sum1), _mm_max_ps(_sum2, _sum3)));
                }
                for (; j < max_nn; j++)
                {
                    const float* pK = key_panel + j;
                    __m128 _sum = pM ? (mask_per_head ? _mm_set_ps(pM[mask_cstep * 3], pM[mask_cstep * 2], pM[mask_cstep], pM[0]) : _mm_set1_ps(pM[0])) : _mm_setzero_ps();
                    if (pM)
                        pM++;
                    const float* pA = pQ;
                    for (int d = 0; d < head_dim; d++)
                    {
                        _sum = _mm_comp_fmadd_ps(_mm_loadu_ps(pA), _mm_set1_ps(*pK), _sum);
                        pA += 4;
                        pK += NR;
                    }
                    _mm_storeu_ps(score_panel, _sum);
                    score_panel += 4;
                    _block_max = _mm_max_ps(_block_max, _sum);
                }
            }

            const __m128 _m_new = _mm_max_ps(_m, _block_max);
            const __m128 _alpha_active = _mm_cmpneq_ps(_l, _mm_setzero_ps());
            __m128 _alpha = exp_ps(_mm_and_ps(_alpha_active, _mm_sub_ps(_m, _m_new)));
            _alpha = _mm_and_ps(_alpha, _alpha_active);

            __m128 _sum0 = _mm_setzero_ps();
            __m128 _sum1 = _mm_setzero_ps();
            __m128 _sum2 = _mm_setzero_ps();
            __m128 _sum3 = _mm_setzero_ps();
            int j = 0;
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
            for (; j < max_jj; j++)
            {
                __m128 _p = exp_ps(_mm_sub_ps(_mm_loadu_ps(scoreptr), _m_new));
                _mm_storeu_ps(scoreptr, _p);
                scoreptr += 4;
                _sum0 = _mm_add_ps(_sum0, _p);
            }
            _l = _mm_add_ps(_mm_mul_ps(_l, _alpha), _mm_add_ps(_mm_add_ps(_sum0, _sum1), _mm_add_ps(_sum2, _sum3)));
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
                    const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 4;
                        const float* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            const __m128 _p = _mm_loadu_ps(pS);
                            _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[0]), _out0);
                            _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[1]), _out1);
                            _out2 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[2]), _out2);
                            _out3 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[3]), _out3);
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
                for (; lane < value_panel_width; lane++)
                {
                    __m128 _out = _mm_mul_ps(_mm_loadu_ps(outptr), _alpha);
                    const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = scoreT_tile + (size_t)jj * 4;
                        const float* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            _out = _mm_comp_fmadd_ps(_mm_loadu_ps(pS), _mm_set1_ps(*pV), _out);
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

        if (!state.empty())
        {
            float* stateptr = state;
            stateptr += qq * (value_dim + 2);
            _mm_storeu_ps(stateptr, _m);
            _mm_storeu_ps(stateptr + 4, _l);
            memcpy(stateptr + 8, outT_tile, (size_t)value_dim * 4 * sizeof(float));
        }
        else
        {
            float* output = top_blob.channel(q0 + qq);
            const size_t output_cstep = top_blob.cstep;
            const __m128 _nonzero = _mm_cmpneq_ps(_l, _mm_setzero_ps());
            const __m128 _denom = _mm_or_ps(_mm_and_ps(_nonzero, _l), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
            const __m128 _out_scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(1.f), _denom), _nonzero);
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
                const __m128 _r = _mm_mul_ps(_mm_loadu_ps(pO), _out_scale);
                *p0 = _mm_cvtss_f32(_r);
                p0[output_cstep] = _mm_cvtss_f32(_mm_shuffle_ps(_r, _r, _MM_SHUFFLE(1, 1, 1, 1)));
                p0[output_cstep * 2] = _mm_cvtss_f32(_mm_movehl_ps(_r, _r));
                p0[output_cstep * 3] = _mm_cvtss_f32(_mm_shuffle_ps(_r, _r, _MM_SHUFFLE(3, 3, 3, 3)));
                p0++;
                pO += 4;
            }
        }
    }
    for (; qq + 1 < max_qq; qq += 2)
    {
        const int q = q0 + qq;
        const float* query_ptr = query.channel(q);
        const size_t query_cstep = query.cstep * query.elempack;
        const bool mask_per_head = !attn_mask_blob.empty() && attn_mask_blob.dims == 3 && attn_mask_blob.c > 1;
        const float* mask = 0;
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

        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
            float block_max0 = -FLT_MAX;
            float block_max1 = -FLT_MAX;

            for (int jj = 0; jj < max_jj; jj += NR)
            {
                const int max_nn = std::min(NR, max_jj - jj);
                const float* key_panel = (const float*)key_cache_head + (size_t)(n + jj) * head_dim;
                int k = 0;
#if __AVX__
#if __AVX512F__
                for (; k + 15 < max_nn; k += 16)
                {
                    __m512 _sum0 = _mm512_setzero_ps();
                    __m512 _sum1 = _mm512_setzero_ps();
                    const float* pK = key_panel + k;
                    const float* pA = query_ptr;
                    int d = 0;
                    for (; d < head_dim; d++)
                    {
                        const __m512 _k = _mm512_loadu_ps(pK);
                        _sum0 = _mm512_fmadd_ps(_k, _mm512_set1_ps(pA[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_k, _mm512_set1_ps(pA[query_cstep]), _sum1);
                        pA++;
                        pK += NR;
                    }
                    _sum0 = _mm512_mul_ps(_sum0, _mm512_set1_ps(scale));
                    _sum1 = _mm512_mul_ps(_sum1, _mm512_set1_ps(scale));
                    if (mask)
                    {
                        const float* pM = mask + n + jj + k;
                        _sum0 = _mm512_add_ps(_sum0, _mm512_loadu_ps(pM));
                        _sum1 = _mm512_add_ps(_sum1, _mm512_loadu_ps(pM + mask_cstep));
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
                    const float* pK = key_panel + k;
                    const float* pA = query_ptr;
                    int d = 0;
                    for (; d < head_dim; d++)
                    {
                        const __m256 _k = _mm256_loadu_ps(pK);
                        _sum0 = _mm256_comp_fmadd_ps(_k, _mm256_set1_ps(pA[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_k, _mm256_set1_ps(pA[query_cstep]), _sum1);
                        pA++;
                        pK += NR;
                    }
                    _sum0 = _mm256_mul_ps(_sum0, _mm256_set1_ps(scale));
                    _sum1 = _mm256_mul_ps(_sum1, _mm256_set1_ps(scale));
                    if (mask)
                    {
                        const float* pM = mask + n + jj + k;
                        _sum0 = _mm256_add_ps(_sum0, _mm256_loadu_ps(pM));
                        _sum1 = _mm256_add_ps(_sum1, _mm256_loadu_ps(pM + mask_cstep));
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
                    const float* pK = key_panel + k;
                    const float* pA = query_ptr;
                    int d = 0;
                    for (; d < head_dim; d++)
                    {
                        const __m128 _k = _mm_loadu_ps(pK);
                        _sum0 = _mm_comp_fmadd_ps(_k, _mm_set1_ps(pA[0]), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_k, _mm_set1_ps(pA[query_cstep]), _sum1);
                        pA++;
                        pK += NR;
                    }
                    _sum0 = _mm_mul_ps(_sum0, _mm_set1_ps(scale));
                    _sum1 = _mm_mul_ps(_sum1, _mm_set1_ps(scale));
                    if (mask)
                    {
                        const float* pM = mask + n + jj + k;
                        _sum0 = _mm_add_ps(_sum0, _mm_loadu_ps(pM));
                        _sum1 = _mm_add_ps(_sum1, _mm_loadu_ps(pM + mask_cstep));
                    }
                    _mm_storeu_ps(score0 + jj + k, _sum0);
                    _mm_storeu_ps(score1 + jj + k, _sum1);
                    block_max0 = std::max(block_max0, _mm_reduce_max_ps(_sum0));
                    block_max1 = std::max(block_max1, _mm_reduce_max_ps(_sum1));
                }
                for (; k < max_nn; k++)
                {
                    const float* pK = key_panel + k;
                    float sum0 = 0.f;
                    float sum1 = 0.f;
                    const float* pA = query_ptr;
                    int d = 0;
                    for (; d < head_dim; d++)
                    {
                        const float v = *pK;
                        sum0 += pA[0] * v;
                        sum1 += pA[query_cstep] * v;
                        pA++;
                        pK += NR;
                    }
                    score0[jj + k] = sum0 * scale + (mask ? mask[n + jj + k] : 0.f);
                    score1[jj + k] = sum1 * scale + (mask ? mask[mask_cstep + n + jj + k] : 0.f);
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
#if __AVX__
#if __AVX512F__
            for (; d + 15 < value_dim; d += 16)
            {
                __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(out0 + d), _mm512_set1_ps(alpha0));
                __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(out1 + d), _mm512_set1_ps(alpha1));
                const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pS = score0 + jj;
                    const float* pV = pV_panel;
                    for (int j = 0; j < max_nn; j++)
                    {
                        const __m512 _v = _mm512_loadu_ps(pV);
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
                const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pS = score0 + jj;
                    const float* pV = pV_panel;
                    for (int j = 0; j < max_nn; j++)
                    {
                        const __m256 _v = _mm256_loadu_ps(pV);
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
                const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pS = score0 + jj;
                    const float* pV = pV_panel;
                    for (int j = 0; j < max_nn; j++)
                    {
                        const __m128 _v = _mm_loadu_ps(pV);
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
            for (; d < value_dim; d++)
            {
                float sum0 = out0[d] * alpha0;
                float sum1 = out1[d] * alpha1;
                const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pS = score0 + jj;
                    const float* pV = pV_panel;
                    for (int j = 0; j < max_nn; j++)
                    {
                        sum0 += pS[0] * *pV;
                        sum1 += pS[block_n] * *pV;
                        pS++;
                        pV++;
                    }
                    pV_panel += (size_t)NR * value_dim;
                }
                out0[d] = sum0;
                out1[d] = sum1;
            }
        }

        if (!state.empty())
        {
            float* state0 = (float*)state + qq * (value_dim + 2);
            float* state1 = state0 + value_dim + 2;
            state0[0] = m0;
            state0[1] = l0;
            state1[0] = m1;
            state1[1] = l1;
            memcpy(state0 + 2, out0, (size_t)value_dim * sizeof(float));
            memcpy(state1 + 2, out1, (size_t)value_dim * sizeof(float));
        }
        else
        {
            float* output = top_blob.channel(q);
            const size_t output_cstep = top_blob.cstep;
            float* p0 = output;
            const float inv_sum0 = l0 == 0.f ? 0.f : 1.f / l0;
            const float inv_sum1 = l1 == 0.f ? 0.f : 1.f / l1;
            int d = 0;
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
            for (; d < value_dim; d++)
            {
                *p0 = out0[d] * inv_sum0;
                p0[output_cstep] = out1[d] * inv_sum1;
                p0++;
            }
        }
    }
#endif // __SSE2__

    for (; qq < max_qq; qq++)
    {
        const int q = q0 + qq;
        const float* query_ptr = query.channel(q);
        const float* mask = 0;
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

        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
            float block_max = -FLT_MAX;

            for (int jj = 0; jj < max_jj; jj += NR)
            {
                const int max_nn = std::min(NR, max_jj - jj);
                const float* pK = (const float*)key_cache_head + (size_t)(n + jj) * head_dim;
                int k = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sum_avx512 = _mm512_setzero_ps();
                for (; k + 15 < max_nn; k += 16)
                {
                    const float* pA = query_ptr;
                    const float* pB = pK + k;
                    for (int d = 0; d < head_dim; d++)
                    {
                        _sum_avx512 = _mm512_fmadd_ps(_mm512_set1_ps(*pA++ * scale), _mm512_loadu_ps(pB), _sum_avx512);
                        pB += NR;
                    }
                    if (mask)
                        _sum_avx512 = _mm512_add_ps(_sum_avx512, _mm512_loadu_ps(mask + n + jj + k));
                    _mm512_storeu_ps(score + jj + k, _sum_avx512);
                    block_max = std::max(block_max, _mm512_comp_reduce_max_ps(_sum_avx512));
                }
#endif // __AVX512F__
                __m256 _sum_avx = _mm256_setzero_ps();
                for (; k + 7 < max_nn; k += 8)
                {
                    const float* pA = query_ptr;
                    const float* pB = pK + k;
                    for (int d = 0; d < head_dim; d++)
                    {
                        _sum_avx = _mm256_comp_fmadd_ps(_mm256_set1_ps(*pA++ * scale), _mm256_loadu_ps(pB), _sum_avx);
                        pB += NR;
                    }
                    if (mask)
                        _sum_avx = _mm256_add_ps(_sum_avx, _mm256_loadu_ps(mask + n + jj + k));
                    _mm256_storeu_ps(score + jj + k, _sum_avx);
                    block_max = std::max(block_max, _mm256_reduce_max_ps(_sum_avx));
                }
#endif // __AVX__
                __m128 _sum = _mm_setzero_ps();
                for (; k + 3 < max_nn; k += 4)
                {
                    const float* pA = query_ptr;
                    const float* pB = pK + k;
                    for (int d = 0; d < head_dim; d++)
                    {
                        _sum = _mm_comp_fmadd_ps(_mm_set1_ps(*pA++ * scale), _mm_loadu_ps(pB), _sum);
                        pB += NR;
                    }
                    if (mask)
                        _sum = _mm_add_ps(_sum, _mm_loadu_ps(mask + n + jj + k));
                    _mm_storeu_ps(score + jj + k, _sum);
                    block_max = std::max(block_max, _mm_reduce_max_ps(_sum));
                }
#endif // __SSE2__
                for (; k < max_nn; k++)
                {
                    const float* pA = query_ptr;
                    const float* pB = pK + k;
                    float sum = 0.f;
                    for (int d = 0; d < head_dim; d++)
                    {
                        sum += *pA++ * scale * *pB;
                        pB += NR;
                    }
                    score[jj + k] = sum + (mask ? mask[n + jj + k] : 0.f);
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
            __m512 _sum_exp_avx512 = _mm512_setzero_ps();
            const __m512 _max_avx512 = _mm512_set1_ps(m_new);
            for (; j + 15 < max_jj; j += 16)
            {
                __m512 _p = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(score + j), _max_avx512));
                _mm512_storeu_ps(score + j, _p);
                _sum_exp_avx512 = _mm512_add_ps(_sum_exp_avx512, _p);
            }
            sum += _mm512_comp_reduce_add_ps(_sum_exp_avx512);
#endif // __AVX512F__
            __m256 _sum_exp_avx = _mm256_setzero_ps();
            const __m256 _max_avx = _mm256_set1_ps(m_new);
            for (; j + 7 < max_jj; j += 8)
            {
                __m256 _p = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(score + j), _max_avx));
                _mm256_storeu_ps(score + j, _p);
                _sum_exp_avx = _mm256_add_ps(_sum_exp_avx, _p);
            }
            sum += _mm256_reduce_add_ps(_sum_exp_avx);
#endif // __AVX__
            __m128 _sum_exp_sse = _mm_setzero_ps();
            const __m128 _max_sse = _mm_set1_ps(m_new);
            for (; j + 3 < max_jj; j += 4)
            {
                __m128 _p = exp_ps(_mm_sub_ps(_mm_loadu_ps(score + j), _max_sse));
                _mm_storeu_ps(score + j, _p);
                _sum_exp_sse = _mm_add_ps(_sum_exp_sse, _p);
            }
            sum += _mm_reduce_add_ps(_sum_exp_sse);
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
                const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pS = score + jj;
                    const float* pV = pV_panel;
                    for (int j = 0; j < max_nn; j++)
                    {
                        _out = _mm512_fmadd_ps(_mm512_loadu_ps(pV), _mm512_set1_ps(*pS++), _out);
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
                const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pS = score + jj;
                    const float* pV = pV_panel;
                    for (int j = 0; j < max_nn; j++)
                    {
                        _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pV), _mm256_set1_ps(*pS++), _out);
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
                const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pS = score + jj;
                    const float* pV = pV_panel;
                    for (int j = 0; j < max_nn; j++)
                    {
                        _out = _mm_comp_fmadd_ps(_mm_loadu_ps(pV), _mm_set1_ps(*pS++), _out);
                        pV += 4;
                    }
                    pV_panel += (size_t)NR * value_dim;
                }
                _mm_storeu_ps(out + d, _out);
            }
#endif // __SSE2__
            for (; d < value_dim; d++)
            {
                float sum0 = out[d] * alpha;
                const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pS = score + jj;
                    const float* pV = pV_panel;
                    for (int j = 0; j < max_nn; j++)
                        sum0 += *pS++ * *pV++;
                    pV_panel += (size_t)NR * value_dim;
                }
                out[d] = sum0;
            }
        }

        if (!state.empty())
        {
            float* stateptr = state;
            stateptr += qq * (value_dim + 2);
            stateptr[0] = m;
            stateptr[1] = l;
            memcpy(stateptr + 2, out, (size_t)value_dim * sizeof(float));
        }
        else
        {
            float* output = top_blob.channel(q);
            const float inv_sum = l == 0.f ? 0.f : 1.f / l;
            int d = 0;
#if __SSE2__
            const __m128 _inv_sum = _mm_set1_ps(inv_sum);
            for (; d + 3 < value_dim; d += 4)
                _mm_storeu_ps(output + d, _mm_mul_ps(_mm_loadu_ps(out + d), _inv_sum));
#endif // __SSE2__
            for (; d < value_dim; d++)
                output[d] = out[d] * inv_sum;
        }
    }
}

static int sdpa_decode_kvcache_fp32(const Mat& query, const Mat& key_cache, const Mat& value_cache, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
{
    const int head_dim = query.w;
    const int value_dim = value_cache.w;
    const int key_seqlen = key_cache.h;
    const int num_query_heads = query.c;
    const int num_kv_heads = key_cache.c;
    const int num_query_heads_per_kv_head = num_query_heads / num_kv_heads;
    const int nT = std::max(opt.num_threads, 1);
    const int block_q = sdpa_decode_get_optimal_tile_q(num_query_heads_per_kv_head, num_kv_heads, nT);
#if __AVX512F__
    const int NR = 16;
#elif __AVX__
    const int NR = 8;
#elif __SSE2__
    const int NR = 4;
#else
    const int NR = 1;
#endif
    const int num_qblocks = (num_query_heads_per_kv_head + block_q - 1) / block_q;
    const int num_tasks = num_kv_heads * num_qblocks;
    int block_n = sdpa_decode_get_optimal_tile_n(head_dim, value_dim, key_seqlen, 4, 4, 4, attn_mask_blob.empty() ? 0 : 4, block_q, num_tasks, nT);
    block_n = std::max(NR, (block_n + NR - 1) / NR * NR);
    const int num_key_blocks = (key_seqlen + block_n - 1) / block_n;

    int num_kv_chunks = 1;
    if (num_tasks < nT && num_key_blocks >= 2)
    {
        num_kv_chunks = std::min((nT + num_tasks - 1) / num_tasks, num_key_blocks);
        num_kv_chunks = std::max(num_kv_chunks, 1);
    }

    Mat packed_query;
    if (num_kv_chunks > 1)
    {
        packed_query.create(head_dim * block_q, 1, num_tasks, 4u, opt.workspace_allocator);
        if (packed_query.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int task_id = 0; task_id < num_tasks; task_id++)
        {
            const int g = task_id / num_qblocks;
            const int qblock_id = task_id % num_qblocks;
            const int q0 = g * num_query_heads_per_kv_head + qblock_id * block_q;
            const int max_qq = std::min(num_query_heads_per_kv_head - qblock_id * block_q, block_q);
            if (max_qq >= 4)
            {
                Mat queryT = packed_query.channel(task_id);
                sdpa_decode_pack_query_fp32(query, queryT, scale, q0, max_qq);
            }
        }
    }

    const int workspace_size = block_q * (head_dim + block_n + value_dim);
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
            if (max_qq >= 4)
                packed_query_tile = packed_query.channel(task_id);
        }
        sdpa_decode_kvcache_tile_fp32(query, key_cache, value_cache, attn_mask_blob, top_blob, scale, q0, max_qq, g, n_begin, n_end, block_n, packed_query_tile, workspace_tile, state);
    }

    if (num_kv_chunks > 1)
        sdpa_decode_reduce(partials, top_blob, workspace, num_tasks, num_qblocks, block_q, num_kv_chunks, num_query_heads_per_kv_head, value_dim, opt);

    return 0;
}

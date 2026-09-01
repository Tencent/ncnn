// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static int sdpa_decode_get_optimal_tile_q(int num_query_heads_per_kv_head)
{
#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (num_query_heads_per_kv_head >= 16)
        return 16;
#endif // __AVX512F__
    if (num_query_heads_per_kv_head >= 8)
        return 8;
#endif // __AVX__
    if (num_query_heads_per_kv_head >= 4)
        return 4;
#endif // __SSE2__
    if (num_query_heads_per_kv_head >= 2)
        return 2;

    return 1;
}

static int sdpa_decode_get_optimal_tile_n(int head_dim, int value_dim, int key_seqlen, int query_storage_size, int key_storage_size, int value_storage_size, int mask_storage_size, int TILE_Q)
{
#if __AVX512F__
    const int tile_n_align = 16;
#elif __AVX__
    const int tile_n_align = 8;
#elif __SSE2__
    const int tile_n_align = 4;
#else
    const int tile_n_align = 2;
#endif

    const size_t l2_cache_size = get_cpu_level2_cache_size();
    const size_t fixed_size = (size_t)TILE_Q * ((size_t)head_dim * query_storage_size + (size_t)value_dim * sizeof(float));
    const size_t size_per_token = (size_t)head_dim * key_storage_size + (size_t)value_dim * value_storage_size + (size_t)TILE_Q * (sizeof(float) + mask_storage_size);

    size_t tile_size = l2_cache_size > fixed_size ? (l2_cache_size - fixed_size) / size_per_token : 0;
    tile_size = std::min(tile_size, (size_t)key_seqlen);
    int TILE_N = (int)tile_size;
    TILE_N = std::max(tile_n_align, TILE_N / tile_n_align * tile_n_align);

    const int num_blocks = (key_seqlen - 1) / TILE_N + 1;
    TILE_N = (key_seqlen - 1) / num_blocks + 1;
    TILE_N = (TILE_N + tile_n_align - 1) / tile_n_align * tile_n_align;

    return TILE_N;
}

static void sdpa_decode_attention_tile(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, int q, int max_qq, int g, int block_n, Mat& workspace)
{
    const int head_dim = query.w;
    const int value_dim = value.w;
    const int score_workspace_size = max_qq * block_n;
    const int out_workspace_size = max_qq * value_dim;
    const int l_workspace_size = max_qq;

    float* workspace_ptr = workspace;
    Mat scoreT(score_workspace_size, workspace_ptr, 4u);
    Mat outT(out_workspace_size, workspace_ptr + score_workspace_size, 4u);
    Mat lT(l_workspace_size, workspace_ptr + score_workspace_size + out_workspace_size, 4u);
    Mat queryT(head_dim * max_qq, workspace_ptr + score_workspace_size + out_workspace_size + l_workspace_size, 4u);

    const Mat query_heads = query.channel_range(q, max_qq);
    sdpa_pack_query(query_heads, queryT, 0, max_qq, query.cstep * query.elempack, scale);

    Mat mask;
    size_t mask_hstep = 0;
    if (!attn_mask_blob.empty())
    {
        if (attn_mask_blob.dims == 3)
        {
            if (attn_mask_blob.c > 1)
            {
                mask = attn_mask_blob.channel_range(q, max_qq);
                mask_hstep = attn_mask_blob.cstep;
            }
            else
            {
                mask = attn_mask_blob.channel(0);
            }
        }
        else
        {
            mask = attn_mask_blob;
        }
    }

    const Mat key_head = key.channel(g);
    const Mat value_head = value.channel(g);
    sdpa_attention_tile(queryT, key_head, Mat(), value_head, Mat(), Mat(), mask, mask_hstep, Mat(), scoreT, outT, lT, max_qq);

    Mat top_blob_heads = top_blob.channel_range(q, max_qq);
    sdpa_store_output_tile(outT, lT, top_blob_heads, 0, max_qq, top_blob.cstep * top_blob.elempack);
}

static void sdpa_decode_tile(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, int q0, int max_qq, int g, int block_n, Mat& workspace)
{
    const int key_seqlen = key.h;
    int qq = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; qq + 15 < max_qq; qq += 16)
    {
        sdpa_decode_attention_tile(query, key, value, attn_mask_blob, top_blob, scale, q0 + qq, 16, g, block_n, workspace);
    }
#endif // __AVX512F__
    for (; qq + 7 < max_qq; qq += 8)
    {
        sdpa_decode_attention_tile(query, key, value, attn_mask_blob, top_blob, scale, q0 + qq, 8, g, block_n, workspace);
    }
#endif // __AVX__
    for (; qq + 3 < max_qq; qq += 4)
    {
        sdpa_decode_attention_tile(query, key, value, attn_mask_blob, top_blob, scale, q0 + qq, 4, g, block_n, workspace);
    }
#endif // __SSE2__

    for (; qq + 1 < max_qq; qq += 2)
    {
        const int q = q0 + qq;
        const int head_dim = query.w;
        const int value_dim = value.w;
        const float* query0 = query.channel(q);
        const float* query1 = query.channel(q + 1);
        const float* mask0 = 0;
        const float* mask1 = 0;
        if (!attn_mask_blob.empty())
        {
            if (attn_mask_blob.dims == 3)
            {
                mask0 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q : 0);
                mask1 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q + 1 : 0);
            }
            else
            {
                mask0 = attn_mask_blob;
                mask1 = attn_mask_blob;
            }
        }

        float* score0 = workspace;
        float* score1 = score0 + block_n;
        float* out0 = score1 + block_n;
        float* out1 = out0 + value_dim;
        memset(out0, 0, (size_t)value_dim * 2 * sizeof(float));

        const Mat key_head = key.channel(g);
        const Mat value_head = value.channel(g);
        float m0 = -FLT_MAX;
        float m1 = -FLT_MAX;
        float l0 = 0.f;
        float l1 = 0.f;

        for (int n = 0; n < key_seqlen; n += block_n)
        {
            const int max_jj = std::min(key_seqlen - n, block_n);
            float block_max0 = -FLT_MAX;
            float block_max1 = -FLT_MAX;
            // qk
            {
                const float* pK = key_head.row(n);
                const float* maskptr0 = mask0 ? mask0 + n : 0;
                const float* maskptr1 = mask1 ? mask1 + n : 0;
                float* scoreptr0 = score0;
                float* scoreptr1 = score1;
                for (int j = 0; j < max_jj; j++)
                {
                    const float* pQ0 = query0;
                    const float* pQ1 = query1;
                    float sum0 = 0.f;
                    float sum1 = 0.f;
                    int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    __m512 _sum0_avx512 = _mm512_setzero_ps();
                    __m512 _sum1_avx512 = _mm512_setzero_ps();
                    for (; d + 15 < head_dim; d += 16)
                    {
                        const __m512 _k = _mm512_loadu_ps(pK);
                        _sum0_avx512 = _mm512_fmadd_ps(_mm512_loadu_ps(pQ0), _k, _sum0_avx512);
                        _sum1_avx512 = _mm512_fmadd_ps(_mm512_loadu_ps(pQ1), _k, _sum1_avx512);
                        pQ0 += 16;
                        pQ1 += 16;
                        pK += 16;
                    }
                    sum0 += _mm512_comp_reduce_add_ps(_sum0_avx512);
                    sum1 += _mm512_comp_reduce_add_ps(_sum1_avx512);
#endif // __AVX512F__
                    __m256 _sum0_avx = _mm256_setzero_ps();
                    __m256 _sum1_avx = _mm256_setzero_ps();
                    for (; d + 7 < head_dim; d += 8)
                    {
                        const __m256 _k = _mm256_loadu_ps(pK);
                        _sum0_avx = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pQ0), _k, _sum0_avx);
                        _sum1_avx = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pQ1), _k, _sum1_avx);
                        pQ0 += 8;
                        pQ1 += 8;
                        pK += 8;
                    }
                    sum0 += _mm256_reduce_add_ps(_sum0_avx);
                    sum1 += _mm256_reduce_add_ps(_sum1_avx);
#endif // __AVX__
                    __m128 _sum0_sse = _mm_setzero_ps();
                    __m128 _sum1_sse = _mm_setzero_ps();
                    for (; d + 3 < head_dim; d += 4)
                    {
                        const __m128 _k = _mm_loadu_ps(pK);
                        _sum0_sse = _mm_comp_fmadd_ps(_mm_loadu_ps(pQ0), _k, _sum0_sse);
                        _sum1_sse = _mm_comp_fmadd_ps(_mm_loadu_ps(pQ1), _k, _sum1_sse);
                        pQ0 += 4;
                        pQ1 += 4;
                        pK += 4;
                    }
                    sum0 += _mm_reduce_add_ps(_sum0_sse);
                    sum1 += _mm_reduce_add_ps(_sum1_sse);
#endif // __SSE2__
                    float sum00 = 0.f;
                    float sum01 = 0.f;
                    float sum10 = 0.f;
                    float sum11 = 0.f;
                    for (; d + 1 < head_dim; d += 2)
                    {
                        sum00 += pQ0[0] * pK[0];
                        sum01 += pQ0[1] * pK[1];
                        sum10 += pQ1[0] * pK[0];
                        sum11 += pQ1[1] * pK[1];
                        pQ0 += 2;
                        pQ1 += 2;
                        pK += 2;
                    }
                    sum0 += sum00 + sum01;
                    sum1 += sum10 + sum11;
                    for (; d < head_dim; d++)
                    {
                        sum0 += *pQ0++ * *pK;
                        sum1 += *pQ1++ * *pK++;
                    }

                    const float s0 = sum0 * scale + (maskptr0 ? *maskptr0++ : 0.f);
                    const float s1 = sum1 * scale + (maskptr1 ? *maskptr1++ : 0.f);
                    *scoreptr0++ = s0;
                    *scoreptr1++ = s1;
                    block_max0 = std::max(block_max0, s0);
                    block_max1 = std::max(block_max1, s1);
                }

            }

            float alpha0;
            float alpha1;

            // online softmax
            {
                const float m_new0 = std::max(m0, block_max0);
                const float m_new1 = std::max(m1, block_max1);
                alpha0 = l0 == 0.f ? 0.f : expf(m0 - m_new0);
                alpha1 = l1 == 0.f ? 0.f : expf(m1 - m_new1);
                float sum0 = 0.f;
                float sum1 = 0.f;
                float* scoreptr0 = score0;
                float* scoreptr1 = score1;
                int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sum0_avx512 = _mm512_setzero_ps();
                __m512 _sum1_avx512 = _mm512_setzero_ps();
                for (; j + 15 < max_jj; j += 16)
                {
                    const __m512 _p0 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(scoreptr0), _mm512_set1_ps(m_new0)));
                    const __m512 _p1 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(scoreptr1), _mm512_set1_ps(m_new1)));
                    _mm512_storeu_ps(scoreptr0, _p0);
                    _mm512_storeu_ps(scoreptr1, _p1);
                    scoreptr0 += 16;
                    scoreptr1 += 16;
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
                    const __m256 _p0 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr0), _mm256_set1_ps(m_new0)));
                    const __m256 _p1 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr1), _mm256_set1_ps(m_new1)));
                    _mm256_storeu_ps(scoreptr0, _p0);
                    _mm256_storeu_ps(scoreptr1, _p1);
                    scoreptr0 += 8;
                    scoreptr1 += 8;
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
                    const __m128 _p0 = exp_ps(_mm_sub_ps(_mm_loadu_ps(scoreptr0), _mm_set1_ps(m_new0)));
                    const __m128 _p1 = exp_ps(_mm_sub_ps(_mm_loadu_ps(scoreptr1), _mm_set1_ps(m_new1)));
                    _mm_storeu_ps(scoreptr0, _p0);
                    _mm_storeu_ps(scoreptr1, _p1);
                    scoreptr0 += 4;
                    scoreptr1 += 4;
                    _sum0_sse = _mm_add_ps(_sum0_sse, _p0);
                    _sum1_sse = _mm_add_ps(_sum1_sse, _p1);
                }
                sum0 += _mm_reduce_add_ps(_sum0_sse);
                sum1 += _mm_reduce_add_ps(_sum1_sse);
#endif // __SSE2__
                for (; j + 1 < max_jj; j += 2)
                {
                    scoreptr0[0] = expf(scoreptr0[0] - m_new0);
                    scoreptr0[1] = expf(scoreptr0[1] - m_new0);
                    scoreptr1[0] = expf(scoreptr1[0] - m_new1);
                    scoreptr1[1] = expf(scoreptr1[1] - m_new1);
                    sum0 += scoreptr0[0] + scoreptr0[1];
                    sum1 += scoreptr1[0] + scoreptr1[1];
                    scoreptr0 += 2;
                    scoreptr1 += 2;
                }
                for (; j < max_jj; j++)
                {
                    *scoreptr0 = expf(*scoreptr0 - m_new0);
                    *scoreptr1 = expf(*scoreptr1 - m_new1);
                    sum0 += *scoreptr0++;
                    sum1 += *scoreptr1++;
                }
                l0 = l0 * alpha0 + sum0;
                l1 = l1 * alpha1 + sum1;
                m0 = m_new0;
                m1 = m_new1;

            }

            // pv
            {
                const float* valueptr = value_head.row(n);
                float* outptr0 = out0;
                float* outptr1 = out1;
                int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; d + 15 < value_dim; d += 16)
                {
                    __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr0), _mm512_set1_ps(alpha0));
                    __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(outptr1), _mm512_set1_ps(alpha1));
                    const float* pV = valueptr;
                    const float* pS0 = score0;
                    const float* pS1 = score1;
                    for (int j = 0; j < max_jj; j++)
                    {
                        const __m512 _v = _mm512_loadu_ps(pV);
                        _out0 = _mm512_fmadd_ps(_v, _mm512_set1_ps(*pS0++), _out0);
                        _out1 = _mm512_fmadd_ps(_v, _mm512_set1_ps(*pS1++), _out1);
                        pV += value_dim;
                    }
                    _mm512_storeu_ps(outptr0, _out0);
                    _mm512_storeu_ps(outptr1, _out1);
                    outptr0 += 16;
                    outptr1 += 16;
                    valueptr += 16;
                }
#endif // __AVX512F__
                for (; d + 7 < value_dim; d += 8)
                {
                    __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr0), _mm256_set1_ps(alpha0));
                    __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr1), _mm256_set1_ps(alpha1));
                    const float* pV = valueptr;
                    const float* pS0 = score0;
                    const float* pS1 = score1;
                    for (int j = 0; j < max_jj; j++)
                    {
                        const __m256 _v = _mm256_loadu_ps(pV);
                        _out0 = _mm256_comp_fmadd_ps(_v, _mm256_set1_ps(*pS0++), _out0);
                        _out1 = _mm256_comp_fmadd_ps(_v, _mm256_set1_ps(*pS1++), _out1);
                        pV += value_dim;
                    }
                    _mm256_storeu_ps(outptr0, _out0);
                    _mm256_storeu_ps(outptr1, _out1);
                    outptr0 += 8;
                    outptr1 += 8;
                    valueptr += 8;
                }
#endif // __AVX__
                for (; d + 3 < value_dim; d += 4)
                {
                    __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(outptr0), _mm_set1_ps(alpha0));
                    __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(outptr1), _mm_set1_ps(alpha1));
                    const float* pV = valueptr;
                    const float* pS0 = score0;
                    const float* pS1 = score1;
                    for (int j = 0; j < max_jj; j++)
                    {
                        const __m128 _v = _mm_loadu_ps(pV);
                        _out0 = _mm_comp_fmadd_ps(_v, _mm_set1_ps(*pS0++), _out0);
                        _out1 = _mm_comp_fmadd_ps(_v, _mm_set1_ps(*pS1++), _out1);
                        pV += value_dim;
                    }
                    _mm_storeu_ps(outptr0, _out0);
                    _mm_storeu_ps(outptr1, _out1);
                    outptr0 += 4;
                    outptr1 += 4;
                    valueptr += 4;
                }
#endif // __SSE2__
                for (; d + 1 < value_dim; d += 2)
                {
                    float out00 = outptr0[0] * alpha0;
                    float out01 = outptr0[1] * alpha0;
                    float out10 = outptr1[0] * alpha1;
                    float out11 = outptr1[1] * alpha1;
                    const float* pV = valueptr;
                    const float* pS0 = score0;
                    const float* pS1 = score1;
                    for (int j = 0; j < max_jj; j++)
                    {
                        out00 += *pS0 * pV[0];
                        out01 += *pS0++ * pV[1];
                        out10 += *pS1 * pV[0];
                        out11 += *pS1++ * pV[1];
                        pV += value_dim;
                    }
                    outptr0[0] = out00;
                    outptr0[1] = out01;
                    outptr1[0] = out10;
                    outptr1[1] = out11;
                    outptr0 += 2;
                    outptr1 += 2;
                    valueptr += 2;
                }
                for (; d < value_dim; d++)
                {
                    float out00 = *outptr0 * alpha0;
                    float out10 = *outptr1 * alpha1;
                    const float* pV = valueptr;
                    const float* pS0 = score0;
                    const float* pS1 = score1;
                    for (int j = 0; j < max_jj; j++)
                    {
                        out00 += *pS0++ * *pV;
                        out10 += *pS1++ * *pV;
                        pV += value_dim;
                    }
                    *outptr0++ = out00;
                    *outptr1++ = out10;
                    valueptr++;
                }
            }

        }

        {
            float* output0 = top_blob.channel(q);
            float* output1 = top_blob.channel(q + 1);
            const float* outptr0 = out0;
            const float* outptr1 = out1;
            const float inv_sum0 = l0 == 0.f ? 0.f : 1.f / l0;
            const float inv_sum1 = l1 == 0.f ? 0.f : 1.f / l1;
            int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; d + 15 < value_dim; d += 16)
            {
                _mm512_storeu_ps(output0, _mm512_mul_ps(_mm512_loadu_ps(outptr0), _mm512_set1_ps(inv_sum0)));
                _mm512_storeu_ps(output1, _mm512_mul_ps(_mm512_loadu_ps(outptr1), _mm512_set1_ps(inv_sum1)));
                output0 += 16;
                output1 += 16;
                outptr0 += 16;
                outptr1 += 16;
            }
#endif // __AVX512F__
            for (; d + 7 < value_dim; d += 8)
            {
                _mm256_storeu_ps(output0, _mm256_mul_ps(_mm256_loadu_ps(outptr0), _mm256_set1_ps(inv_sum0)));
                _mm256_storeu_ps(output1, _mm256_mul_ps(_mm256_loadu_ps(outptr1), _mm256_set1_ps(inv_sum1)));
                output0 += 8;
                output1 += 8;
                outptr0 += 8;
                outptr1 += 8;
            }
#endif // __AVX__
            for (; d + 3 < value_dim; d += 4)
            {
                _mm_storeu_ps(output0, _mm_mul_ps(_mm_loadu_ps(outptr0), _mm_set1_ps(inv_sum0)));
                _mm_storeu_ps(output1, _mm_mul_ps(_mm_loadu_ps(outptr1), _mm_set1_ps(inv_sum1)));
                output0 += 4;
                output1 += 4;
                outptr0 += 4;
                outptr1 += 4;
            }
#endif // __SSE2__
            for (; d + 1 < value_dim; d += 2)
            {
                output0[0] = outptr0[0] * inv_sum0;
                output0[1] = outptr0[1] * inv_sum0;
                output1[0] = outptr1[0] * inv_sum1;
                output1[1] = outptr1[1] * inv_sum1;
                output0 += 2;
                output1 += 2;
                outptr0 += 2;
                outptr1 += 2;
            }
            for (; d < value_dim; d++)
            {
                *output0++ = *outptr0++ * inv_sum0;
                *output1++ = *outptr1++ * inv_sum1;
            }
        }
    }

    for (; qq < max_qq; qq++)
    {
        const int q = q0 + qq;
        const int head_dim = query.w;
        const int value_dim = value.w;

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

        for (int n = 0; n < key_seqlen; n += block_n)
        {
            const int max_jj = std::min(key_seqlen - n, block_n);
            float block_max = -FLT_MAX;
            // qk
            {
                const float* pK = key_head.row(n);
                float* pS = score;
                const float* maskptr = mask ? mask + n : 0;
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
                    float sum0 = 0.f;
                    float sum1 = 0.f;
                    for (; i + 1 < head_dim; i += 2)
                    {
                        sum0 += pQ[0] * pK[0];
                        sum1 += pQ[1] * pK[1];
                        pQ += 2;
                        pK += 2;
                    }
                    sum += sum0 + sum1;
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
                    if (maskptr)
                    {
                        s += *maskptr;
                        maskptr++;
                    }
                    *pS++ = s;
                    block_max = std::max(block_max, s);
                }

            }

            float alpha;

            // online softmax
            {
                const float m_new = std::max(m, block_max);
                alpha = l == 0.f ? 0.f : expf(m - m_new);
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
                    for (; i + 1 < max_jj; i += 2)
                    {
                        pS[0] = expf(pS[0] - m_new);
                        pS[1] = expf(pS[1] - m_new);
                        sum += pS[0] + pS[1];
                        pS += 2;
                    }
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

            }

            // pv
            {
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
                for (; d + 1 < value_dim; d += 2)
                {
                    float sum0 = outptr[0] * alpha;
                    float sum1 = outptr[1] * alpha;
                    const float* pV = valueptr;
                    const float* pS = score;
                    for (int j = 0; j < max_jj; j++)
                    {
                        sum0 += *pS * pV[0];
                        sum1 += *pS++ * pV[1];
                        pV += value_dim;
                    }
                    outptr[0] = sum0;
                    outptr[1] = sum1;
                    outptr += 2;
                    valueptr += 2;
                }
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
                for (; i + 1 < value_dim; i += 2)
                {
                    output[i] *= inv_sum;
                    output[i + 1] *= inv_sum;
                }
                for (; i < value_dim; i++)
                    output[i] *= inv_sum;
            }
        }
    }
}

static int sdpa_decode(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
{
    const int num_query_heads = query.c;
    const int num_kv_heads = key.c;
    const int key_seqlen = key.h;
    const int value_dim = value.w;
    const int num_query_heads_per_kv_head = num_query_heads / num_kv_heads;
    const int block_q = sdpa_decode_get_optimal_tile_q(num_query_heads_per_kv_head);
    const int num_qblocks = (num_query_heads_per_kv_head + block_q - 1) / block_q;
    const int num_tasks = num_kv_heads * num_qblocks;
    const int nT = std::min(std::max(opt.num_threads, 1), num_tasks);
    const int block_n = sdpa_decode_get_optimal_tile_n(query.w, value_dim, key_seqlen, 4, 4, 4, attn_mask_blob.empty() ? 0 : 4, block_q);

    const bool pack_query = block_q >= 4;
    const size_t score_workspace_size = (size_t)block_q * block_n * sizeof(float);
    const size_t output_workspace_size = (size_t)block_q * value_dim * sizeof(float);
    const size_t l_workspace_size = pack_query ? (size_t)block_q * sizeof(float) : 0;
    const size_t query_workspace_size = pack_query ? (size_t)block_q * query.w * sizeof(float) : 0;
    const size_t workspace_size = alignSize(score_workspace_size + output_workspace_size + l_workspace_size + query_workspace_size, 64);
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
        sdpa_decode_tile(query, key, value, attn_mask_blob, top_blob, scale, q0, max_qq, g, block_n, workspace_tile);
    }

    return 0;
}

static void sdpa_decode_kvcache_small_tile(const Mat& query, const Mat& key_cache, const Mat& value_cache, const Mat& attn_mask_blob, Mat& top_blob, float scale, int q0, int max_qq, int g, int block_n, Mat& workspace)
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
    float* scoreT_ptr = scoreT;
    float* outT_ptr = outT;
    const Mat key_cache_head = key_cache.channel(g);
    const Mat value_cache_head = value_cache.channel(g);

    int qq = 0;
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

        for (int n = 0; n < key_seqlen; n += block_n)
        {
            const int max_jj = std::min(key_seqlen - n, block_n);
            float block_max0 = -FLT_MAX;
            float block_max1 = -FLT_MAX;

            // qk
            {
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* key_panel = (const float*)key_cache_head + (size_t)(n + jj) * head_dim;
                    float* scoreptr0 = score0 + jj;
                    float* scoreptr1 = score1 + jj;
                    const float* maskptr0 = mask ? mask + n + jj : 0;
                    const float* maskptr1 = maskptr0 ? maskptr0 + mask_cstep : 0;
                    int k = 0;
#if __SSE2__
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
                        if (maskptr0)
                        {
                            _sum0 = _mm512_add_ps(_sum0, _mm512_loadu_ps(maskptr0));
                            _sum1 = _mm512_add_ps(_sum1, _mm512_loadu_ps(maskptr1));
                            maskptr0 += 16;
                            maskptr1 += 16;
                        }
                        _mm512_storeu_ps(scoreptr0, _sum0);
                        _mm512_storeu_ps(scoreptr1, _sum1);
                        scoreptr0 += 16;
                        scoreptr1 += 16;
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
                        if (maskptr0)
                        {
                            _sum0 = _mm256_add_ps(_sum0, _mm256_loadu_ps(maskptr0));
                            _sum1 = _mm256_add_ps(_sum1, _mm256_loadu_ps(maskptr1));
                            maskptr0 += 8;
                            maskptr1 += 8;
                        }
                        _mm256_storeu_ps(scoreptr0, _sum0);
                        _mm256_storeu_ps(scoreptr1, _sum1);
                        scoreptr0 += 8;
                        scoreptr1 += 8;
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
                        if (maskptr0)
                        {
                            _sum0 = _mm_add_ps(_sum0, _mm_loadu_ps(maskptr0));
                            _sum1 = _mm_add_ps(_sum1, _mm_loadu_ps(maskptr1));
                            maskptr0 += 4;
                            maskptr1 += 4;
                        }
                        _mm_storeu_ps(scoreptr0, _sum0);
                        _mm_storeu_ps(scoreptr1, _sum1);
                        scoreptr0 += 4;
                        scoreptr1 += 4;
                        block_max0 = std::max(block_max0, _mm_reduce_max_ps(_sum0));
                        block_max1 = std::max(block_max1, _mm_reduce_max_ps(_sum1));
                    }
#endif // __SSE2__
                    for (; k + 1 < max_nn; k += 2)
                    {
                        const float* pK = key_panel + k;
                        float sum00 = 0.f;
                        float sum01 = 0.f;
                        float sum10 = 0.f;
                        float sum11 = 0.f;
                        const float* pA = query_ptr;
                        for (int d = 0; d < head_dim; d++)
                        {
                            const float k0 = pK[0];
                            const float k1 = pK[1];
                            sum00 += pA[0] * k0;
                            sum01 += pA[0] * k1;
                            sum10 += pA[query_cstep] * k0;
                            sum11 += pA[query_cstep] * k1;
                            pA++;
                            pK += NR;
                        }
                        sum00 = sum00 * scale + (maskptr0 ? maskptr0[0] : 0.f);
                        sum01 = sum01 * scale + (maskptr0 ? maskptr0[1] : 0.f);
                        sum10 = sum10 * scale + (maskptr1 ? maskptr1[0] : 0.f);
                        sum11 = sum11 * scale + (maskptr1 ? maskptr1[1] : 0.f);
                        scoreptr0[0] = sum00;
                        scoreptr0[1] = sum01;
                        scoreptr1[0] = sum10;
                        scoreptr1[1] = sum11;
                        scoreptr0 += 2;
                        scoreptr1 += 2;
                        if (maskptr0)
                        {
                            maskptr0 += 2;
                            maskptr1 += 2;
                        }
                        block_max0 = std::max(block_max0, std::max(sum00, sum01));
                        block_max1 = std::max(block_max1, std::max(sum10, sum11));
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
                        sum0 = sum0 * scale + (maskptr0 ? *maskptr0++ : 0.f);
                        sum1 = sum1 * scale + (maskptr1 ? *maskptr1++ : 0.f);
                        *scoreptr0++ = sum0;
                        *scoreptr1++ = sum1;
                        block_max0 = std::max(block_max0, sum0);
                        block_max1 = std::max(block_max1, sum1);
                    }
                }

            }

            float alpha0;
            float alpha1;

            // online softmax
            {
                const float m_new0 = std::max(m0, block_max0);
                const float m_new1 = std::max(m1, block_max1);
                alpha0 = l0 == 0.f ? 0.f : expf(m0 - m_new0);
                alpha1 = l1 == 0.f ? 0.f : expf(m1 - m_new1);

                float sum0 = 0.f;
                float sum1 = 0.f;
                float* scoreptr0 = score0;
                float* scoreptr1 = score1;
                int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sum0_avx512 = _mm512_setzero_ps();
                __m512 _sum1_avx512 = _mm512_setzero_ps();
                for (; j + 15 < max_jj; j += 16)
                {
                    __m512 _p0 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(scoreptr0), _mm512_set1_ps(m_new0)));
                    __m512 _p1 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(scoreptr1), _mm512_set1_ps(m_new1)));
                    _mm512_storeu_ps(scoreptr0, _p0);
                    _mm512_storeu_ps(scoreptr1, _p1);
                    scoreptr0 += 16;
                    scoreptr1 += 16;
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
                    __m256 _p0 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr0), _mm256_set1_ps(m_new0)));
                    __m256 _p1 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr1), _mm256_set1_ps(m_new1)));
                    _mm256_storeu_ps(scoreptr0, _p0);
                    _mm256_storeu_ps(scoreptr1, _p1);
                    scoreptr0 += 8;
                    scoreptr1 += 8;
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
                    __m128 _p0 = exp_ps(_mm_sub_ps(_mm_loadu_ps(scoreptr0), _mm_set1_ps(m_new0)));
                    __m128 _p1 = exp_ps(_mm_sub_ps(_mm_loadu_ps(scoreptr1), _mm_set1_ps(m_new1)));
                    _mm_storeu_ps(scoreptr0, _p0);
                    _mm_storeu_ps(scoreptr1, _p1);
                    scoreptr0 += 4;
                    scoreptr1 += 4;
                    _sum0_sse = _mm_add_ps(_sum0_sse, _p0);
                    _sum1_sse = _mm_add_ps(_sum1_sse, _p1);
                }
                sum0 += _mm_reduce_add_ps(_sum0_sse);
                sum1 += _mm_reduce_add_ps(_sum1_sse);
#endif // __SSE2__
                for (; j + 1 < max_jj; j += 2)
                {
                    scoreptr0[0] = expf(scoreptr0[0] - m_new0);
                    scoreptr0[1] = expf(scoreptr0[1] - m_new0);
                    scoreptr1[0] = expf(scoreptr1[0] - m_new1);
                    scoreptr1[1] = expf(scoreptr1[1] - m_new1);
                    sum0 += scoreptr0[0] + scoreptr0[1];
                    sum1 += scoreptr1[0] + scoreptr1[1];
                    scoreptr0 += 2;
                    scoreptr1 += 2;
                }
                for (; j < max_jj; j++)
                {
                    *scoreptr0 = expf(*scoreptr0 - m_new0);
                    *scoreptr1 = expf(*scoreptr1 - m_new1);
                    sum0 += *scoreptr0++;
                    sum1 += *scoreptr1++;
                }
                l0 = l0 * alpha0 + sum0;
                l1 = l1 * alpha1 + sum1;
                m0 = m_new0;
                m1 = m_new1;

            }

            // pv
            {
                float* outptr0 = out0;
                float* outptr1 = out1;
                int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; d + 15 < value_dim; d += 16)
                {
                    __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr0), _mm512_set1_ps(alpha0));
                    __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(outptr1), _mm512_set1_ps(alpha1));
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
                    _mm512_storeu_ps(outptr0, _out0);
                    _mm512_storeu_ps(outptr1, _out1);
                    outptr0 += 16;
                    outptr1 += 16;
                }
#endif // __AVX512F__
                for (; d + 7 < value_dim; d += 8)
                {
                    __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr0), _mm256_set1_ps(alpha0));
                    __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr1), _mm256_set1_ps(alpha1));
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
                    _mm256_storeu_ps(outptr0, _out0);
                    _mm256_storeu_ps(outptr1, _out1);
                    outptr0 += 8;
                    outptr1 += 8;
                }
#endif // __AVX__
                for (; d + 3 < value_dim; d += 4)
                {
                    __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(outptr0), _mm_set1_ps(alpha0));
                    __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(outptr1), _mm_set1_ps(alpha1));
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
                    _mm_storeu_ps(outptr0, _out0);
                    _mm_storeu_ps(outptr1, _out1);
                    outptr0 += 4;
                    outptr1 += 4;
                }
#endif // __SSE2__
                for (; d + 1 < value_dim; d += 2)
                {
                    float sum00 = outptr0[0] * alpha0;
                    float sum01 = outptr0[1] * alpha0;
                    float sum10 = outptr1[0] * alpha1;
                    float sum11 = outptr1[1] * alpha1;
                    const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = score0 + jj;
                        const float* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            const float v0 = pV[0];
                            const float v1 = pV[1];
                            sum00 += pS[0] * v0;
                            sum01 += pS[0] * v1;
                            sum10 += pS[block_n] * v0;
                            sum11 += pS[block_n] * v1;
                            pS++;
                            pV += 2;
                        }
                        pV_panel += (size_t)NR * value_dim;
                    }
                    outptr0[0] = sum00;
                    outptr0[1] = sum01;
                    outptr1[0] = sum10;
                    outptr1[1] = sum11;
                    outptr0 += 2;
                    outptr1 += 2;
                }
                for (; d < value_dim; d++)
                {
                    float sum0 = *outptr0 * alpha0;
                    float sum1 = *outptr1 * alpha1;
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
                    *outptr0++ = sum0;
                    *outptr1++ = sum1;
                }
            }

        }

        {
            float* output = top_blob.channel(q);
            const size_t output_cstep = top_blob.cstep;
            float* p0 = output;
            const float* outptr0 = out0;
            const float* outptr1 = out1;
            const float inv_sum0 = l0 == 0.f ? 0.f : 1.f / l0;
            const float inv_sum1 = l1 == 0.f ? 0.f : 1.f / l1;
            int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; d + 15 < value_dim; d += 16)
            {
                _mm512_storeu_ps(p0, _mm512_mul_ps(_mm512_loadu_ps(outptr0), _mm512_set1_ps(inv_sum0)));
                _mm512_storeu_ps(p0 + output_cstep, _mm512_mul_ps(_mm512_loadu_ps(outptr1), _mm512_set1_ps(inv_sum1)));
                p0 += 16;
                outptr0 += 16;
                outptr1 += 16;
            }
#endif // __AVX512F__
            for (; d + 7 < value_dim; d += 8)
            {
                _mm256_storeu_ps(p0, _mm256_mul_ps(_mm256_loadu_ps(outptr0), _mm256_set1_ps(inv_sum0)));
                _mm256_storeu_ps(p0 + output_cstep, _mm256_mul_ps(_mm256_loadu_ps(outptr1), _mm256_set1_ps(inv_sum1)));
                p0 += 8;
                outptr0 += 8;
                outptr1 += 8;
            }
#endif // __AVX__
            for (; d + 3 < value_dim; d += 4)
            {
                _mm_storeu_ps(p0, _mm_mul_ps(_mm_loadu_ps(outptr0), _mm_set1_ps(inv_sum0)));
                _mm_storeu_ps(p0 + output_cstep, _mm_mul_ps(_mm_loadu_ps(outptr1), _mm_set1_ps(inv_sum1)));
                p0 += 4;
                outptr0 += 4;
                outptr1 += 4;
            }
#endif // __SSE2__
            for (; d + 1 < value_dim; d += 2)
            {
                p0[0] = outptr0[0] * inv_sum0;
                p0[1] = outptr0[1] * inv_sum0;
                p0[output_cstep] = outptr1[0] * inv_sum1;
                p0[output_cstep + 1] = outptr1[1] * inv_sum1;
                p0 += 2;
                outptr0 += 2;
                outptr1 += 2;
            }
            for (; d < value_dim; d++)
            {
                *p0 = *outptr0++ * inv_sum0;
                p0[output_cstep] = *outptr1++ * inv_sum1;
                p0++;
            }
        }
    }

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

        for (int n = 0; n < key_seqlen; n += block_n)
        {
            const int max_jj = std::min(key_seqlen - n, block_n);
            float block_max = -FLT_MAX;

            // qk
            {
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    const float* pK = (const float*)key_cache_head + (size_t)(n + jj) * head_dim;
                    float* scoreptr = score + jj;
                    const float* maskptr = mask ? mask + n + jj : 0;
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
                            _sum_avx512 = _mm512_fmadd_ps(_mm512_set1_ps(*pA++), _mm512_loadu_ps(pB), _sum_avx512);
                            pB += NR;
                        }
                        _sum_avx512 = _mm512_mul_ps(_sum_avx512, _mm512_set1_ps(scale));
                        if (maskptr)
                        {
                            _sum_avx512 = _mm512_add_ps(_sum_avx512, _mm512_loadu_ps(maskptr));
                            maskptr += 16;
                        }
                        _mm512_storeu_ps(scoreptr, _sum_avx512);
                        scoreptr += 16;
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
                            _sum_avx = _mm256_comp_fmadd_ps(_mm256_set1_ps(*pA++), _mm256_loadu_ps(pB), _sum_avx);
                            pB += NR;
                        }
                        _sum_avx = _mm256_mul_ps(_sum_avx, _mm256_set1_ps(scale));
                        if (maskptr)
                        {
                            _sum_avx = _mm256_add_ps(_sum_avx, _mm256_loadu_ps(maskptr));
                            maskptr += 8;
                        }
                        _mm256_storeu_ps(scoreptr, _sum_avx);
                        scoreptr += 8;
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
                            _sum = _mm_comp_fmadd_ps(_mm_set1_ps(*pA++), _mm_loadu_ps(pB), _sum);
                            pB += NR;
                        }
                        _sum = _mm_mul_ps(_sum, _mm_set1_ps(scale));
                        if (maskptr)
                        {
                            _sum = _mm_add_ps(_sum, _mm_loadu_ps(maskptr));
                            maskptr += 4;
                        }
                        _mm_storeu_ps(scoreptr, _sum);
                        scoreptr += 4;
                        block_max = std::max(block_max, _mm_reduce_max_ps(_sum));
                    }
#endif // __SSE2__
                    for (; k + 1 < max_nn; k += 2)
                    {
                        const float* pA = query_ptr;
                        const float* pB = pK + k;
                        float sum0 = 0.f;
                        float sum1 = 0.f;
                        for (int d = 0; d < head_dim; d++)
                        {
                            sum0 += *pA * pB[0];
                            sum1 += *pA++ * pB[1];
                            pB += NR;
                        }
                        sum0 = sum0 * scale + (maskptr ? maskptr[0] : 0.f);
                        sum1 = sum1 * scale + (maskptr ? maskptr[1] : 0.f);
                        scoreptr[0] = sum0;
                        scoreptr[1] = sum1;
                        scoreptr += 2;
                        if (maskptr)
                            maskptr += 2;
                        block_max = std::max(block_max, std::max(sum0, sum1));
                    }
                    for (; k < max_nn; k++)
                    {
                        const float* pA = query_ptr;
                        const float* pB = pK + k;
                        float sum = 0.f;
                        for (int d = 0; d < head_dim; d++)
                        {
                            sum += *pA++ * *pB;
                            pB += NR;
                        }
                        sum = sum * scale + (maskptr ? *maskptr++ : 0.f);
                        *scoreptr++ = sum;
                        block_max = std::max(block_max, sum);
                    }
                }

            }

            float alpha;

            // online softmax
            {
                const float m_new = std::max(m, block_max);
                alpha = l == 0.f ? 0.f : expf(m - m_new);

                float sum = 0.f;
                float* scoreptr = score;
                int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sum_exp_avx512 = _mm512_setzero_ps();
                const __m512 _max_avx512 = _mm512_set1_ps(m_new);
                for (; j + 15 < max_jj; j += 16)
                {
                    __m512 _p = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(scoreptr), _max_avx512));
                    _mm512_storeu_ps(scoreptr, _p);
                    scoreptr += 16;
                    _sum_exp_avx512 = _mm512_add_ps(_sum_exp_avx512, _p);
                }
                sum += _mm512_comp_reduce_add_ps(_sum_exp_avx512);
#endif // __AVX512F__
                __m256 _sum_exp_avx = _mm256_setzero_ps();
                const __m256 _max_avx = _mm256_set1_ps(m_new);
                for (; j + 7 < max_jj; j += 8)
                {
                    __m256 _p = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr), _max_avx));
                    _mm256_storeu_ps(scoreptr, _p);
                    scoreptr += 8;
                    _sum_exp_avx = _mm256_add_ps(_sum_exp_avx, _p);
                }
                sum += _mm256_reduce_add_ps(_sum_exp_avx);
#endif // __AVX__
                __m128 _sum_exp_sse = _mm_setzero_ps();
                const __m128 _max_sse = _mm_set1_ps(m_new);
                for (; j + 3 < max_jj; j += 4)
                {
                    __m128 _p = exp_ps(_mm_sub_ps(_mm_loadu_ps(scoreptr), _max_sse));
                    _mm_storeu_ps(scoreptr, _p);
                    scoreptr += 4;
                    _sum_exp_sse = _mm_add_ps(_sum_exp_sse, _p);
                }
                sum += _mm_reduce_add_ps(_sum_exp_sse);
#endif // __SSE2__
                for (; j + 1 < max_jj; j += 2)
                {
                    scoreptr[0] = expf(scoreptr[0] - m_new);
                    scoreptr[1] = expf(scoreptr[1] - m_new);
                    sum += scoreptr[0] + scoreptr[1];
                    scoreptr += 2;
                }
                for (; j < max_jj; j++)
                {
                    *scoreptr = expf(*scoreptr - m_new);
                    sum += *scoreptr++;
                }
                l = l * alpha + sum;
                m = m_new;

            }

            // pv
            {
                float* outptr = out;
                int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; d + 15 < value_dim; d += 16)
                {
                    __m512 _out = _mm512_mul_ps(_mm512_loadu_ps(outptr), _mm512_set1_ps(alpha));
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
                    _mm512_storeu_ps(outptr, _out);
                    outptr += 16;
                }
#endif // __AVX512F__
                for (; d + 7 < value_dim; d += 8)
                {
                    __m256 _out = _mm256_mul_ps(_mm256_loadu_ps(outptr), _mm256_set1_ps(alpha));
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
                    _mm256_storeu_ps(outptr, _out);
                    outptr += 8;
                }
#endif // __AVX__
                for (; d + 3 < value_dim; d += 4)
                {
                    __m128 _out = _mm_mul_ps(_mm_loadu_ps(outptr), _mm_set1_ps(alpha));
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
                    _mm_storeu_ps(outptr, _out);
                    outptr += 4;
                }
#endif // __SSE2__
                for (; d + 1 < value_dim; d += 2)
                {
                    float sum0 = outptr[0] * alpha;
                    float sum1 = outptr[1] * alpha;
                    const float* pV_panel = (const float*)value_cache_head + (size_t)n * value_dim + (size_t)d * NR;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        const float* pS = score + jj;
                        const float* pV = pV_panel;
                        for (int j = 0; j < max_nn; j++)
                        {
                            sum0 += *pS * pV[0];
                            sum1 += *pS++ * pV[1];
                            pV += 2;
                        }
                        pV_panel += (size_t)NR * value_dim;
                    }
                    outptr[0] = sum0;
                    outptr[1] = sum1;
                    outptr += 2;
                }
                for (; d < value_dim; d++)
                {
                    float sum0 = *outptr * alpha;
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
                    *outptr++ = sum0;
                }
            }

        }

        {
            float* output = top_blob.channel(q);
            const float* outptr = out;
            const float inv_sum = l == 0.f ? 0.f : 1.f / l;
            int d = 0;
#if __SSE2__
            const __m128 _inv_sum = _mm_set1_ps(inv_sum);
            for (; d + 3 < value_dim; d += 4)
            {
                _mm_storeu_ps(output, _mm_mul_ps(_mm_loadu_ps(outptr), _inv_sum));
                output += 4;
                outptr += 4;
            }
#endif // __SSE2__
            for (; d + 1 < value_dim; d += 2)
            {
                output[0] = outptr[0] * inv_sum;
                output[1] = outptr[1] * inv_sum;
                output += 2;
                outptr += 2;
            }
            for (; d < value_dim; d++)
                *output++ = *outptr++ * inv_sum;
        }
    }
}

static int sdpa_decode_kvcache(const Mat& query, const Mat& key_cache, const Mat& value_cache, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
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
    int block_n = sdpa_decode_get_optimal_tile_n(head_dim, value_dim, key_seqlen, 4, 4, 4, attn_mask_blob.empty() ? 0 : 4, block_q);
    block_n = std::max(NR, (block_n + NR - 1) / NR * NR);

    const bool use_packed_tile = block_q >= 4;
    const size_t score_workspace_size = (size_t)block_q * block_n * sizeof(float);
    const size_t output_workspace_size = (size_t)block_q * value_dim * sizeof(float);
    const size_t query_workspace_size = use_packed_tile ? (size_t)block_q * head_dim * sizeof(float) : 0;
    const size_t l_workspace_size = use_packed_tile ? (size_t)block_q * sizeof(float) : 0;
    const size_t mask_workspace_size = use_packed_tile && !attn_mask_blob.empty() ? (size_t)block_q * key_seqlen * sizeof(float) : 0;
    const size_t workspace_size = alignSize(score_workspace_size + output_workspace_size + query_workspace_size + l_workspace_size + mask_workspace_size, 64);
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
        if (max_qq >= 4)
        {
            float* workspace_ptr = workspace_tile;
            Mat scoreT(block_q * block_n, workspace_ptr, 4u);
            Mat outT(block_q * value_dim, workspace_ptr + block_q * block_n, 4u);
            Mat lT(block_q, workspace_ptr + block_q * (block_n + value_dim), 4u);
            Mat queryT(block_q * head_dim, workspace_ptr + block_q * (block_n + value_dim + 1), 4u);
            Mat packed_mask_tile;
            if (!attn_mask_blob.empty())
                packed_mask_tile = Mat(block_q * key_seqlen, workspace_ptr + block_q * (block_n + value_dim + 1 + head_dim), 4u);

            const Mat query_heads = query.channel_range(q0, max_qq);
            sdpa_pack_query(query_heads, queryT, 0, max_qq, query.cstep * query.elempack, scale);

            if (!attn_mask_blob.empty())
            {
                const bool mask_per_head = attn_mask_blob.dims == 3 && attn_mask_blob.c > 1;
                const Mat mask_head = attn_mask_blob.dims == 3 ? (mask_per_head ? attn_mask_blob.channel_range(q0, max_qq) : attn_mask_blob.channel(0)) : attn_mask_blob;
                const size_t mask_hstep = mask_per_head ? attn_mask_blob.cstep * attn_mask_blob.elempack : 0;
                sdpa_pack_query(mask_head, packed_mask_tile, 0, max_qq, mask_hstep, 1.f);
            }

            const Mat key_cache_head = key_cache.channel(g);
            const Mat value_cache_head = value_cache.channel(g);
            sdpa_attention_tile(queryT, Mat(), key_cache_head, Mat(), value_cache_head, Mat(), Mat(), 0, packed_mask_tile, scoreT, outT, lT, max_qq);

            Mat top_blob_heads = top_blob.channel_range(q0, max_qq);
            sdpa_store_output_tile(outT, lT, top_blob_heads, 0, max_qq, top_blob.cstep * top_blob.elempack);
        }
        else
        {
            sdpa_decode_kvcache_small_tile(query, key_cache, value_cache, attn_mask_blob, top_blob, scale, q0, max_qq, g, block_n, workspace_tile);
        }
    }

    return 0;
}

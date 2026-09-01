// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static int sdpa_prefill_get_optimal_tile_m()
{
#if __AVX512F__
    return 16;
#elif __AVX__
    return 8;
#elif __SSE2__
    return 4;
#else
    return 2;
#endif
}

static int sdpa_value_panel_width(int remain)
{
#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (remain >= 16)
        return 16;
#endif // __AVX512F__
    if (remain >= 8)
        return 8;
#endif // __AVX__
    if (remain >= 4)
        return 4;
#endif // __SSE2__
    if (remain >= 2)
        return 2;
    return 1;
}

static int sdpa_prefill_get_optimal_tile_n(int head_dim, int value_dim, int key_seqlen, int query_storage_size, int key_storage_size, int value_storage_size, int mask_storage_size, int TILE_M)
{
    int tile_n_align = 2;
#if defined(__x86_64__) || defined(_M_X64)
#if __SSE2__
    tile_n_align = 4;
#if __AVX__
    tile_n_align = 8;
#if __AVX512F__
    tile_n_align = 16;
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
#else
#if __SSE2__
    tile_n_align = 4;
#endif // __SSE2__
#endif // defined(__x86_64__) || defined(_M_X64)

    const size_t l2_cache_size = get_cpu_level2_cache_size();
    const size_t fixed_size = (size_t)TILE_M * ((size_t)head_dim * query_storage_size + (size_t)value_dim * sizeof(float));
    const size_t size_per_token = (size_t)head_dim * key_storage_size + (size_t)value_dim * value_storage_size + (size_t)TILE_M * (sizeof(float) + mask_storage_size);

    size_t tile_size = l2_cache_size > fixed_size ? (l2_cache_size - fixed_size) / size_per_token : 0;
    tile_size = std::min(tile_size, (size_t)key_seqlen);
    int TILE_N = (int)tile_size;
    TILE_N = std::max(tile_n_align, TILE_N / tile_n_align * tile_n_align);

    const int num_blocks = (key_seqlen - 1) / TILE_N + 1;
    TILE_N = (key_seqlen - 1) / num_blocks + 1;
    TILE_N = (TILE_N + tile_n_align - 1) / tile_n_align * tile_n_align;

    return TILE_N;
}

static Mat sdpa_prefill_get_mask_head(const Mat& attn_mask_blob, int q)
{
    if (attn_mask_blob.empty())
        return Mat();

    if (attn_mask_blob.dims == 3)
        return attn_mask_blob.channel(attn_mask_blob.c > 1 ? q : 0);

    return attn_mask_blob;
}

// queryT[head_dim][query_lane]
static void sdpa_pack_query(const Mat& query_head, Mat& queryT, int i, int max_ii, size_t q_hstep, float scale)
{
    const int head_dim = query_head.w;
    float* queryT_ptr = queryT;
    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const int i0 = i + ii;
        float* pQ = queryT_ptr + (size_t)ii * head_dim;
        const float* qptr = (const float*)query_head + (size_t)i0 * q_hstep;

        __m512 _scale = _mm512_set1_ps(scale);
        int d = 0;
        for (; d + 15 < head_dim; d += 16)
        {
            __m512 _r0 = _mm512_loadu_ps(qptr);
            __m512 _r1 = _mm512_loadu_ps(qptr + q_hstep);
            __m512 _r2 = _mm512_loadu_ps(qptr + q_hstep * 2);
            __m512 _r3 = _mm512_loadu_ps(qptr + q_hstep * 3);
            __m512 _r4 = _mm512_loadu_ps(qptr + q_hstep * 4);
            __m512 _r5 = _mm512_loadu_ps(qptr + q_hstep * 5);
            __m512 _r6 = _mm512_loadu_ps(qptr + q_hstep * 6);
            __m512 _r7 = _mm512_loadu_ps(qptr + q_hstep * 7);
            __m512 _r8 = _mm512_loadu_ps(qptr + q_hstep * 8);
            __m512 _r9 = _mm512_loadu_ps(qptr + q_hstep * 9);
            __m512 _ra = _mm512_loadu_ps(qptr + q_hstep * 10);
            __m512 _rb = _mm512_loadu_ps(qptr + q_hstep * 11);
            __m512 _rc = _mm512_loadu_ps(qptr + q_hstep * 12);
            __m512 _rd = _mm512_loadu_ps(qptr + q_hstep * 13);
            __m512 _re = _mm512_loadu_ps(qptr + q_hstep * 14);
            __m512 _rf = _mm512_loadu_ps(qptr + q_hstep * 15);
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
            pQ[1] = qptr[q_hstep] * scale;
            pQ[2] = qptr[q_hstep * 2] * scale;
            pQ[3] = qptr[q_hstep * 3] * scale;
            pQ[4] = qptr[q_hstep * 4] * scale;
            pQ[5] = qptr[q_hstep * 5] * scale;
            pQ[6] = qptr[q_hstep * 6] * scale;
            pQ[7] = qptr[q_hstep * 7] * scale;
            pQ[8] = qptr[q_hstep * 8] * scale;
            pQ[9] = qptr[q_hstep * 9] * scale;
            pQ[10] = qptr[q_hstep * 10] * scale;
            pQ[11] = qptr[q_hstep * 11] * scale;
            pQ[12] = qptr[q_hstep * 12] * scale;
            pQ[13] = qptr[q_hstep * 13] * scale;
            pQ[14] = qptr[q_hstep * 14] * scale;
            pQ[15] = qptr[q_hstep * 15] * scale;
            qptr++;
            pQ += 16;
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const int i0 = i + ii;
        float* pQ = queryT_ptr + (size_t)ii * head_dim;
        const float* qptr = (const float*)query_head + (size_t)i0 * q_hstep;

        __m256 _scale = _mm256_set1_ps(scale);
        int d = 0;
        for (; d + 7 < head_dim; d += 8)
        {
            __m256 _r0 = _mm256_loadu_ps(qptr);
            __m256 _r1 = _mm256_loadu_ps(qptr + q_hstep);
            __m256 _r2 = _mm256_loadu_ps(qptr + q_hstep * 2);
            __m256 _r3 = _mm256_loadu_ps(qptr + q_hstep * 3);
            __m256 _r4 = _mm256_loadu_ps(qptr + q_hstep * 4);
            __m256 _r5 = _mm256_loadu_ps(qptr + q_hstep * 5);
            __m256 _r6 = _mm256_loadu_ps(qptr + q_hstep * 6);
            __m256 _r7 = _mm256_loadu_ps(qptr + q_hstep * 7);
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
            pQ[1] = qptr[q_hstep] * scale;
            pQ[2] = qptr[q_hstep * 2] * scale;
            pQ[3] = qptr[q_hstep * 3] * scale;
            pQ[4] = qptr[q_hstep * 4] * scale;
            pQ[5] = qptr[q_hstep * 5] * scale;
            pQ[6] = qptr[q_hstep * 6] * scale;
            pQ[7] = qptr[q_hstep * 7] * scale;
            qptr++;
            pQ += 8;
        }
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const int i0 = i + ii;
        float* pQ = queryT_ptr + (size_t)ii * head_dim;
        const float* qptr = (const float*)query_head + (size_t)i0 * q_hstep;

        __m128 _scale = _mm_set1_ps(scale);
        int d = 0;
        for (; d + 3 < head_dim; d += 4)
        {
            __m128 _r0 = _mm_loadu_ps(qptr);
            __m128 _r1 = _mm_loadu_ps(qptr + q_hstep);
            __m128 _r2 = _mm_loadu_ps(qptr + q_hstep * 2);
            __m128 _r3 = _mm_loadu_ps(qptr + q_hstep * 3);
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
            pQ[1] = qptr[q_hstep] * scale;
            pQ[2] = qptr[q_hstep * 2] * scale;
            pQ[3] = qptr[q_hstep * 3] * scale;
            qptr++;
            pQ += 4;
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)query_head + (size_t)(i + ii) * q_hstep;
        const float* p1 = p0 + q_hstep;
        float* pp0 = queryT_ptr + (size_t)ii * head_dim;
        float* pp1 = pp0 + head_dim;
        int d = 0;
        for (; d + 1 < head_dim; d += 2)
        {
            pp0[0] = p0[0] * scale;
            pp0[1] = p0[1] * scale;
            pp1[0] = p1[0] * scale;
            pp1[1] = p1[1] * scale;
            p0 += 2;
            p1 += 2;
            pp0 += 2;
            pp1 += 2;
        }
        for (; d < head_dim; d++)
        {
            *pp0++ = *p0++ * scale;
            *pp1++ = *p1++ * scale;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const float* qptr = (const float*)query_head + (size_t)(i + ii) * q_hstep;
        float* pQ = queryT_ptr + (size_t)ii * head_dim;
        int d = 0;
        for (; d + 1 < head_dim; d += 2)
        {
            pQ[0] = qptr[0] * scale;
            pQ[1] = qptr[1] * scale;
            pQ += 2;
            qptr += 2;
        }
        for (; d < head_dim; d++)
            *pQ++ = *qptr++ * scale;
    }
}

// packed_mask[mask_head][query_block][query_panel][key][query_lane] in fp32
static void sdpa_pack_mask(const Mat& attn_mask_blob, Mat& packed_mask, int block_m, const Option& opt)
{
    const int query_seqlen = attn_mask_blob.h;
    const int num_mask_heads = attn_mask_blob.dims == 3 ? attn_mask_blob.c : 1;
    const int num_mblocks = (query_seqlen + block_m - 1) / block_m;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int task_id = 0; task_id < num_mask_heads * num_mblocks; task_id++)
    {
        const int q = task_id / num_mblocks;
        const int mblock_id = task_id % num_mblocks;
        const int i0 = mblock_id * block_m;
        const int max_ii = std::min(query_seqlen - i0, block_m);
        const Mat mask_head = sdpa_prefill_get_mask_head(attn_mask_blob, q);
        Mat packed_mask_head = packed_mask.channel(q);
        Mat packed_mask_tile = packed_mask_head.row_range(mblock_id, 1);
        sdpa_pack_query(mask_head, packed_mask_tile, i0, max_ii, mask_head.w * mask_head.elempack, 1.f);
    }
}

static void sdpa_store_output_tile(const Mat& outT, const Mat& lT, Mat& top_blob_head, int i0, int max_ii, size_t out_hstep)
{
    const int block_m = lT.w;
    const int value_dim = outT.w / block_m;
    float* top_blob_ptr = top_blob_head;
    const float* pp = outT;
    const float* lptr = lT;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        __m512 _l = _mm512_loadu_ps(lptr);
        lptr += 16;
        float* p0 = top_blob_ptr + (size_t)(i0 + ii) * out_hstep;
        __mmask16 nonzero = _mm512_cmp_ps_mask(_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
        __m512 _scale = _mm512_maskz_div_ps(nonzero, _mm512_set1_ps(1.f), _l);
        int d = 0;
        for (; d + 15 < value_dim; d += 16)
        {
            __m512 _r0 = _mm512_mul_ps(_mm512_loadu_ps(pp), _scale);
            __m512 _r1 = _mm512_mul_ps(_mm512_loadu_ps(pp + 16), _scale);
            __m512 _r2 = _mm512_mul_ps(_mm512_loadu_ps(pp + 32), _scale);
            __m512 _r3 = _mm512_mul_ps(_mm512_loadu_ps(pp + 48), _scale);
            __m512 _r4 = _mm512_mul_ps(_mm512_loadu_ps(pp + 64), _scale);
            __m512 _r5 = _mm512_mul_ps(_mm512_loadu_ps(pp + 80), _scale);
            __m512 _r6 = _mm512_mul_ps(_mm512_loadu_ps(pp + 96), _scale);
            __m512 _r7 = _mm512_mul_ps(_mm512_loadu_ps(pp + 112), _scale);
            __m512 _r8 = _mm512_mul_ps(_mm512_loadu_ps(pp + 128), _scale);
            __m512 _r9 = _mm512_mul_ps(_mm512_loadu_ps(pp + 144), _scale);
            __m512 _ra = _mm512_mul_ps(_mm512_loadu_ps(pp + 160), _scale);
            __m512 _rb = _mm512_mul_ps(_mm512_loadu_ps(pp + 176), _scale);
            __m512 _rc = _mm512_mul_ps(_mm512_loadu_ps(pp + 192), _scale);
            __m512 _rd = _mm512_mul_ps(_mm512_loadu_ps(pp + 208), _scale);
            __m512 _re = _mm512_mul_ps(_mm512_loadu_ps(pp + 224), _scale);
            __m512 _rf = _mm512_mul_ps(_mm512_loadu_ps(pp + 240), _scale);
            transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
            _mm512_storeu_ps(p0, _r0);
            _mm512_storeu_ps(p0 + out_hstep, _r1);
            _mm512_storeu_ps(p0 + out_hstep * 2, _r2);
            _mm512_storeu_ps(p0 + out_hstep * 3, _r3);
            _mm512_storeu_ps(p0 + out_hstep * 4, _r4);
            _mm512_storeu_ps(p0 + out_hstep * 5, _r5);
            _mm512_storeu_ps(p0 + out_hstep * 6, _r6);
            _mm512_storeu_ps(p0 + out_hstep * 7, _r7);
            _mm512_storeu_ps(p0 + out_hstep * 8, _r8);
            _mm512_storeu_ps(p0 + out_hstep * 9, _r9);
            _mm512_storeu_ps(p0 + out_hstep * 10, _ra);
            _mm512_storeu_ps(p0 + out_hstep * 11, _rb);
            _mm512_storeu_ps(p0 + out_hstep * 12, _rc);
            _mm512_storeu_ps(p0 + out_hstep * 13, _rd);
            _mm512_storeu_ps(p0 + out_hstep * 14, _re);
            _mm512_storeu_ps(p0 + out_hstep * 15, _rf);
            pp += 256;
            p0 += 16;
        }
        for (; d < value_dim; d++)
        {
            __m512 _r = _mm512_mul_ps(_mm512_loadu_ps(pp), _scale);
            __m128 _r0 = _mm512_extractf32x4_ps(_r, 0);
            __m128 _r1 = _mm512_extractf32x4_ps(_r, 1);
            __m128 _r2 = _mm512_extractf32x4_ps(_r, 2);
            __m128 _r3 = _mm512_extractf32x4_ps(_r, 3);
            p0[0] = _mm_cvtss_f32(_r0);
            p0[out_hstep] = _mm_cvtss_f32(_mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(1, 1, 1, 1)));
            p0[out_hstep * 2] = _mm_cvtss_f32(_mm_movehl_ps(_r0, _r0));
            p0[out_hstep * 3] = _mm_cvtss_f32(_mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(3, 3, 3, 3)));
            p0[out_hstep * 4] = _mm_cvtss_f32(_r1);
            p0[out_hstep * 5] = _mm_cvtss_f32(_mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(1, 1, 1, 1)));
            p0[out_hstep * 6] = _mm_cvtss_f32(_mm_movehl_ps(_r1, _r1));
            p0[out_hstep * 7] = _mm_cvtss_f32(_mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(3, 3, 3, 3)));
            p0[out_hstep * 8] = _mm_cvtss_f32(_r2);
            p0[out_hstep * 9] = _mm_cvtss_f32(_mm_shuffle_ps(_r2, _r2, _MM_SHUFFLE(1, 1, 1, 1)));
            p0[out_hstep * 10] = _mm_cvtss_f32(_mm_movehl_ps(_r2, _r2));
            p0[out_hstep * 11] = _mm_cvtss_f32(_mm_shuffle_ps(_r2, _r2, _MM_SHUFFLE(3, 3, 3, 3)));
            p0[out_hstep * 12] = _mm_cvtss_f32(_r3);
            p0[out_hstep * 13] = _mm_cvtss_f32(_mm_shuffle_ps(_r3, _r3, _MM_SHUFFLE(1, 1, 1, 1)));
            p0[out_hstep * 14] = _mm_cvtss_f32(_mm_movehl_ps(_r3, _r3));
            p0[out_hstep * 15] = _mm_cvtss_f32(_mm_shuffle_ps(_r3, _r3, _MM_SHUFFLE(3, 3, 3, 3)));
            p0++;
            pp += 16;
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        __m256 _l = _mm256_loadu_ps(lptr);
        lptr += 8;
        float* p0 = top_blob_ptr + (size_t)(i0 + ii) * out_hstep;
        __m256 _nonzero = _mm256_cmp_ps(_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
        __m256 _denom = _mm256_blendv_ps(_mm256_set1_ps(1.f), _l, _nonzero);
        __m256 _scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(1.f), _denom), _nonzero);

        int d = 0;
        for (; d + 7 < value_dim; d += 8)
        {
            __m256 _r0 = _mm256_mul_ps(_mm256_loadu_ps(pp), _scale);
            __m256 _r1 = _mm256_mul_ps(_mm256_loadu_ps(pp + 8), _scale);
            __m256 _r2 = _mm256_mul_ps(_mm256_loadu_ps(pp + 16), _scale);
            __m256 _r3 = _mm256_mul_ps(_mm256_loadu_ps(pp + 24), _scale);
            __m256 _r4 = _mm256_mul_ps(_mm256_loadu_ps(pp + 32), _scale);
            __m256 _r5 = _mm256_mul_ps(_mm256_loadu_ps(pp + 40), _scale);
            __m256 _r6 = _mm256_mul_ps(_mm256_loadu_ps(pp + 48), _scale);
            __m256 _r7 = _mm256_mul_ps(_mm256_loadu_ps(pp + 56), _scale);
            transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            _mm256_storeu_ps(p0, _r0);
            _mm256_storeu_ps(p0 + out_hstep, _r1);
            _mm256_storeu_ps(p0 + out_hstep * 2, _r2);
            _mm256_storeu_ps(p0 + out_hstep * 3, _r3);
            _mm256_storeu_ps(p0 + out_hstep * 4, _r4);
            _mm256_storeu_ps(p0 + out_hstep * 5, _r5);
            _mm256_storeu_ps(p0 + out_hstep * 6, _r6);
            _mm256_storeu_ps(p0 + out_hstep * 7, _r7);
            pp += 64;
            p0 += 8;
        }
        for (; d < value_dim; d++)
        {
            __m256 _r = _mm256_mul_ps(_mm256_loadu_ps(pp), _scale);
            __m128 _r0 = _mm256_castps256_ps128(_r);
            __m128 _r1 = _mm256_extractf128_ps(_r, 1);
            p0[0] = _mm_cvtss_f32(_r0);
            p0[out_hstep] = _mm_cvtss_f32(_mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(1, 1, 1, 1)));
            p0[out_hstep * 2] = _mm_cvtss_f32(_mm_movehl_ps(_r0, _r0));
            p0[out_hstep * 3] = _mm_cvtss_f32(_mm_shuffle_ps(_r0, _r0, _MM_SHUFFLE(3, 3, 3, 3)));
            p0[out_hstep * 4] = _mm_cvtss_f32(_r1);
            p0[out_hstep * 5] = _mm_cvtss_f32(_mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(1, 1, 1, 1)));
            p0[out_hstep * 6] = _mm_cvtss_f32(_mm_movehl_ps(_r1, _r1));
            p0[out_hstep * 7] = _mm_cvtss_f32(_mm_shuffle_ps(_r1, _r1, _MM_SHUFFLE(3, 3, 3, 3)));
            p0++;
            pp += 8;
        }
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        __m128 _l = _mm_loadu_ps(lptr);
        lptr += 4;
        float* p0 = top_blob_ptr + (size_t)(i0 + ii) * out_hstep;
        __m128 _nonzero = _mm_cmpneq_ps(_l, _mm_setzero_ps());
        __m128 _denom = _mm_or_ps(_mm_and_ps(_nonzero, _l), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
        __m128 _scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(1.f), _denom), _nonzero);

        int d = 0;
        for (; d + 3 < value_dim; d += 4)
        {
            __m128 _r0 = _mm_mul_ps(_mm_loadu_ps(pp), _scale);
            __m128 _r1 = _mm_mul_ps(_mm_loadu_ps(pp + 4), _scale);
            __m128 _r2 = _mm_mul_ps(_mm_loadu_ps(pp + 8), _scale);
            __m128 _r3 = _mm_mul_ps(_mm_loadu_ps(pp + 12), _scale);
            _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
            _mm_storeu_ps(p0, _r0);
            _mm_storeu_ps(p0 + out_hstep, _r1);
            _mm_storeu_ps(p0 + out_hstep * 2, _r2);
            _mm_storeu_ps(p0 + out_hstep * 3, _r3);
            pp += 16;
            p0 += 4;
        }
        for (; d < value_dim; d++)
        {
            __m128 _r = _mm_mul_ps(_mm_loadu_ps(pp), _scale);
            p0[0] = _mm_cvtss_f32(_r);
            p0[out_hstep] = _mm_cvtss_f32(_mm_shuffle_ps(_r, _r, _MM_SHUFFLE(1, 1, 1, 1)));
            p0[out_hstep * 2] = _mm_cvtss_f32(_mm_movehl_ps(_r, _r));
            p0[out_hstep * 3] = _mm_cvtss_f32(_mm_shuffle_ps(_r, _r, _MM_SHUFFLE(3, 3, 3, 3)));
            p0++;
            pp += 4;
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* p0 = top_blob_ptr + (size_t)(i0 + ii) * out_hstep;
        float* p1 = p0 + out_hstep;
        const float inv_sum0 = lptr[0] == 0.f ? 0.f : 1.f / lptr[0];
        const float inv_sum1 = lptr[1] == 0.f ? 0.f : 1.f / lptr[1];
        lptr += 2;
        const float* pp0 = pp;
        const float* pp1 = pp0 + value_dim;
        int d = 0;
        for (; d + 1 < value_dim; d += 2)
        {
            p0[0] = pp0[0] * inv_sum0;
            p0[1] = pp0[1] * inv_sum0;
            p1[0] = pp1[0] * inv_sum1;
            p1[1] = pp1[1] * inv_sum1;
            p0 += 2;
            p1 += 2;
            pp0 += 2;
            pp1 += 2;
        }
        for (; d < value_dim; d++)
        {
            *p0++ = *pp0++ * inv_sum0;
            *p1++ = *pp1++ * inv_sum1;
        }
        pp += (size_t)value_dim * 2;
    }
    for (; ii < max_ii; ii++)
    {
        float* p0 = top_blob_ptr + (size_t)(i0 + ii) * out_hstep;
        const float inv_sum = *lptr == 0.f ? 0.f : 1.f / *lptr;
        lptr++;
        int d = 0;
        for (; d + 1 < value_dim; d += 2)
        {
            p0[0] = pp[0] * inv_sum;
            p0[1] = pp[1] * inv_sum;
            p0 += 2;
            pp += 2;
        }
        for (; d < value_dim; d++)
            *p0++ = *pp++ * inv_sum;
    }
}

static void sdpa_pack_key_tile(const Mat& key, Mat& packed_key, int src_begin, int dst_begin, int max_seqlen)
{
    const int head_dim = key.w;
    const int hstep = head_dim;
#if __AVX512F__
    const int panel_width = 16;
#elif __AVX__
    const int panel_width = 8;
#elif __SSE2__
    const int panel_width = 4;
#else
    const int panel_width = 2;
#endif
    const int token_lane = dst_begin;
    float* panel = packed_key;
    int j = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX__
#if __AVX512F__
    for (; j + 15 < max_seqlen; j += 16)
    {
        const float* p0 = key.row(src_begin + j);

        float* pp = panel + token_lane + j;
        int d = 0;
        for (; d + 15 < head_dim; d += 16)
        {
            __m512 _r0 = _mm512_loadu_ps(p0);
            __m512 _r1 = _mm512_loadu_ps(p0 + hstep);
            __m512 _r2 = _mm512_loadu_ps(p0 + hstep * 2);
            __m512 _r3 = _mm512_loadu_ps(p0 + hstep * 3);
            __m512 _r4 = _mm512_loadu_ps(p0 + hstep * 4);
            __m512 _r5 = _mm512_loadu_ps(p0 + hstep * 5);
            __m512 _r6 = _mm512_loadu_ps(p0 + hstep * 6);
            __m512 _r7 = _mm512_loadu_ps(p0 + hstep * 7);
            __m512 _r8 = _mm512_loadu_ps(p0 + hstep * 8);
            __m512 _r9 = _mm512_loadu_ps(p0 + hstep * 9);
            __m512 _ra = _mm512_loadu_ps(p0 + hstep * 10);
            __m512 _rb = _mm512_loadu_ps(p0 + hstep * 11);
            __m512 _rc = _mm512_loadu_ps(p0 + hstep * 12);
            __m512 _rd = _mm512_loadu_ps(p0 + hstep * 13);
            __m512 _re = _mm512_loadu_ps(p0 + hstep * 14);
            __m512 _rf = _mm512_loadu_ps(p0 + hstep * 15);
            transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);
            _mm512_storeu_ps(pp, _r0);
            _mm512_storeu_ps(pp + panel_width, _r1);
            _mm512_storeu_ps(pp + panel_width * 2, _r2);
            _mm512_storeu_ps(pp + panel_width * 3, _r3);
            _mm512_storeu_ps(pp + panel_width * 4, _r4);
            _mm512_storeu_ps(pp + panel_width * 5, _r5);
            _mm512_storeu_ps(pp + panel_width * 6, _r6);
            _mm512_storeu_ps(pp + panel_width * 7, _r7);
            _mm512_storeu_ps(pp + panel_width * 8, _r8);
            _mm512_storeu_ps(pp + panel_width * 9, _r9);
            _mm512_storeu_ps(pp + panel_width * 10, _ra);
            _mm512_storeu_ps(pp + panel_width * 11, _rb);
            _mm512_storeu_ps(pp + panel_width * 12, _rc);
            _mm512_storeu_ps(pp + panel_width * 13, _rd);
            _mm512_storeu_ps(pp + panel_width * 14, _re);
            _mm512_storeu_ps(pp + panel_width * 15, _rf);

            p0 += 16;
            pp += panel_width * 16;
        }
        for (; d < head_dim; d++)
        {
            pp[0] = p0[0];
            pp[1] = p0[hstep];
            pp[2] = p0[hstep * 2];
            pp[3] = p0[hstep * 3];
            pp[4] = p0[hstep * 4];
            pp[5] = p0[hstep * 5];
            pp[6] = p0[hstep * 6];
            pp[7] = p0[hstep * 7];
            pp[8] = p0[hstep * 8];
            pp[9] = p0[hstep * 9];
            pp[10] = p0[hstep * 10];
            pp[11] = p0[hstep * 11];
            pp[12] = p0[hstep * 12];
            pp[13] = p0[hstep * 13];
            pp[14] = p0[hstep * 14];
            pp[15] = p0[hstep * 15];
            p0++;
            pp += panel_width;
        }
    }
#endif // __AVX512F__
    for (; j + 7 < max_seqlen; j += 8)
    {
        const float* p0 = key.row(src_begin + j);

        float* pp = panel + token_lane + j;
        int d = 0;
        for (; d + 7 < head_dim; d += 8)
        {
            __m256 _r0 = _mm256_loadu_ps(p0);
            __m256 _r1 = _mm256_loadu_ps(p0 + hstep);
            __m256 _r2 = _mm256_loadu_ps(p0 + hstep * 2);
            __m256 _r3 = _mm256_loadu_ps(p0 + hstep * 3);
            __m256 _r4 = _mm256_loadu_ps(p0 + hstep * 4);
            __m256 _r5 = _mm256_loadu_ps(p0 + hstep * 5);
            __m256 _r6 = _mm256_loadu_ps(p0 + hstep * 6);
            __m256 _r7 = _mm256_loadu_ps(p0 + hstep * 7);
            transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            _mm256_storeu_ps(pp, _r0);
            _mm256_storeu_ps(pp + panel_width, _r1);
            _mm256_storeu_ps(pp + panel_width * 2, _r2);
            _mm256_storeu_ps(pp + panel_width * 3, _r3);
            _mm256_storeu_ps(pp + panel_width * 4, _r4);
            _mm256_storeu_ps(pp + panel_width * 5, _r5);
            _mm256_storeu_ps(pp + panel_width * 6, _r6);
            _mm256_storeu_ps(pp + panel_width * 7, _r7);

            p0 += 8;
            pp += panel_width * 8;
        }
        for (; d < head_dim; d++)
        {
            pp[0] = p0[0];
            pp[1] = p0[hstep];
            pp[2] = p0[hstep * 2];
            pp[3] = p0[hstep * 3];
            pp[4] = p0[hstep * 4];
            pp[5] = p0[hstep * 5];
            pp[6] = p0[hstep * 6];
            pp[7] = p0[hstep * 7];
            p0++;
            pp += panel_width;
        }
    }
#endif // __AVX__
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; j + 3 < max_seqlen; j += 4)
    {
        const float* p0 = key.row(src_begin + j);

        float* pp = panel + token_lane + j;
        int d = 0;
        for (; d + 3 < head_dim; d += 4)
        {
            __m128 _r0 = _mm_loadu_ps(p0);
            __m128 _r1 = _mm_loadu_ps(p0 + hstep);
            __m128 _r2 = _mm_loadu_ps(p0 + hstep * 2);
            __m128 _r3 = _mm_loadu_ps(p0 + hstep * 3);
            _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
            _mm_storeu_ps(pp, _r0);
            _mm_storeu_ps(pp + panel_width, _r1);
            _mm_storeu_ps(pp + panel_width * 2, _r2);
            _mm_storeu_ps(pp + panel_width * 3, _r3);

            p0 += 4;
            pp += panel_width * 4;
        }
        for (; d < head_dim; d++)
        {
            pp[0] = p0[0];
            pp[1] = p0[hstep];
            pp[2] = p0[hstep * 2];
            pp[3] = p0[hstep * 3];
            p0++;
            pp += panel_width;
        }
    }
#endif // __SSE2__
    for (; j + 1 < max_seqlen; j += 2)
    {
        const float* p0 = key.row(src_begin + j);
        float* pp = panel + token_lane + j;
        for (int d = 0; d < head_dim; d++)
        {
            pp[0] = p0[0];
            pp[1] = p0[hstep];
            p0++;
            pp += panel_width;
        }
    }
    for (; j < max_seqlen; j++)
    {
        const float* p0 = key.row(src_begin + j);
        float* pp = panel + token_lane + j;
        for (int d = 0; d < head_dim; d++)
        {
            *pp = *p0++;
            pp += panel_width;
        }
    }
}

// packed_value[token_panel][value_panel][token_lane][value_lane] in fp32
static void sdpa_pack_value_tile(const Mat& value, Mat& packed_value, int src_begin, int dst_begin, int max_seqlen)
{
    const int value_dim = value.w;
#if __AVX512F__
    const int panel_width = 16;
#elif __AVX__
    const int panel_width = 8;
#elif __SSE2__
    const int panel_width = 4;
#else
    const int panel_width = 2;
#endif
    const int token_lane = dst_begin;
    float* panel = packed_value;
    int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; d + 15 < value_dim; d += 16)
    {
        const float* p0 = value.row(src_begin) + d;
        float* pp = panel + (size_t)d * panel_width + token_lane * 16;
        for (int n = 0; n < max_seqlen; n++)
        {
            _mm512_storeu_ps(pp, _mm512_loadu_ps(p0));
            p0 += value_dim;
            pp += 16;
        }
    }
#endif // __AVX512F__
    for (; d + 7 < value_dim; d += 8)
    {
        const float* p0 = value.row(src_begin) + d;
        float* pp = panel + (size_t)d * panel_width + token_lane * 8;
        for (int n = 0; n < max_seqlen; n++)
        {
            _mm256_storeu_ps(pp, _mm256_loadu_ps(p0));
            p0 += value_dim;
            pp += 8;
        }
    }
#endif // __AVX__
    for (; d + 3 < value_dim; d += 4)
    {
        const float* p0 = value.row(src_begin) + d;
        float* pp = panel + (size_t)d * panel_width + token_lane * 4;
        for (int n = 0; n < max_seqlen; n++)
        {
            _mm_storeu_ps(pp, _mm_loadu_ps(p0));
            p0 += value_dim;
            pp += 4;
        }
    }
#endif // __SSE2__
    for (; d + 1 < value_dim; d += 2)
    {
        const float* p0 = value.row(src_begin) + d;
        float* pp = panel + (size_t)d * panel_width + token_lane * 2;
        for (int n = 0; n < max_seqlen; n++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            p0 += value_dim;
            pp += 2;
        }
    }
    for (; d < value_dim; d++)
    {
        const float* p0 = value.row(src_begin) + d;
        float* pp = panel + (size_t)d * panel_width + token_lane;
        for (int n = 0; n < max_seqlen; n++)
        {
            *pp++ = *p0;
            p0 += value_dim;
        }
    }
}

// computation_value[key_block][value_panel][token][value_lane] in fp32
static void sdpa_pack_computation_value_tile(const Mat& packed_value_head, Mat& computation_value_tile, int src_begin, int max_seqlen)
{
    const int value_dim = packed_value_head.w;
#if __AVX512F__
    const int NR = 16;
#elif __AVX__
    const int NR = 8;
#elif __SSE2__
    const int NR = 4;
#else
    const int NR = 2;
#endif

    float* pp = computation_value_tile;
    int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; d + 15 < value_dim; d += 16)
    {
        for (int jj = 0; jj < max_seqlen; jj += NR)
        {
            const int max_nn = std::min(NR, max_seqlen - jj);
            const float* p0 = (const float*)packed_value_head + (size_t)(src_begin + jj) * value_dim + (size_t)d * NR;
            for (int j = 0; j < max_nn; j++)
            {
                _mm512_storeu_ps(pp, _mm512_loadu_ps(p0));
                pp += 16;
                p0 += 16;
            }
        }
    }
#endif // __AVX512F__
    for (; d + 7 < value_dim; d += 8)
    {
        for (int jj = 0; jj < max_seqlen; jj += NR)
        {
            const int max_nn = std::min(NR, max_seqlen - jj);
            const float* p0 = (const float*)packed_value_head + (size_t)(src_begin + jj) * value_dim + (size_t)d * NR;
            for (int j = 0; j < max_nn; j++)
            {
                _mm256_storeu_ps(pp, _mm256_loadu_ps(p0));
                pp += 8;
                p0 += 8;
            }
        }
    }
#endif // __AVX__
    for (; d + 3 < value_dim; d += 4)
    {
        for (int jj = 0; jj < max_seqlen; jj += NR)
        {
            const int max_nn = std::min(NR, max_seqlen - jj);
            const float* p0 = (const float*)packed_value_head + (size_t)(src_begin + jj) * value_dim + (size_t)d * NR;
            for (int j = 0; j < max_nn; j++)
            {
                _mm_storeu_ps(pp, _mm_loadu_ps(p0));
                pp += 4;
                p0 += 4;
            }
        }
    }
#endif // __SSE2__
    for (; d + 1 < value_dim; d += 2)
    {
        for (int jj = 0; jj < max_seqlen; jj += NR)
        {
            const int max_nn = std::min(NR, max_seqlen - jj);
            const float* p0 = (const float*)packed_value_head + (size_t)(src_begin + jj) * value_dim + (size_t)d * NR;
            for (int j = 0; j < max_nn; j++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += 2;
            }
        }
    }
    for (; d < value_dim; d++)
    {
        for (int jj = 0; jj < max_seqlen; jj += NR)
        {
            const int max_nn = std::min(NR, max_seqlen - jj);
            const float* p0 = (const float*)packed_value_head + (size_t)(src_begin + jj) * value_dim + (size_t)d * NR;
            for (int j = 0; j < max_nn; j++)
                *pp++ = *p0++;
        }
    }
}

static void sdpa_pack_computation_value(const Mat& packed_value, Mat& computation_value, int block_n, const Option& opt)
{
    const int key_seqlen = packed_value.h;
    const int num_kv_heads = packed_value.c;
    const int num_key_blocks = (key_seqlen + block_n - 1) / block_n;
    const int nT = std::max(opt.num_threads, 1);
    const int num_pack_chunks = std::min(num_key_blocks, std::max(1, (nT + num_kv_heads - 1) / num_kv_heads));

    #pragma omp parallel for num_threads(nT)
    for (int task_id = 0; task_id < num_kv_heads * num_pack_chunks; task_id++)
    {
        const int g = task_id / num_pack_chunks;
        const int chunk_id = task_id % num_pack_chunks;
        const int block_begin_id = chunk_id * num_key_blocks / num_pack_chunks;
        const int block_end_id = (chunk_id + 1) * num_key_blocks / num_pack_chunks;
        const Mat packed_value_head = packed_value.channel(g);
        Mat computation_value_head = computation_value.channel(g);

        for (int block_id = block_begin_id; block_id < block_end_id; block_id++)
        {
            const int n = block_id * block_n;
            const int max_jj = std::min(block_n, key_seqlen - n);
            Mat computation_value_tile = computation_value_head.row_range(block_id, 1);

            sdpa_pack_computation_value_tile(packed_value_head, computation_value_tile, n, max_jj);
        }
    }
}

static void sdpa_append_kvcache_token(const Mat& key, const Mat& value, Mat& cached_key, Mat& cached_value, int token_index, int panel_width)
{
    const int panel_id = token_index / panel_width;
    const int token_lane = token_index % panel_width;

    for (int g = 0; g < key.c; g++)
    {
        const float* kptr = key.channel(g);
        float* kpp = (float*)cached_key.channel(g) + (size_t)panel_id * key.w * panel_width + token_lane;
        int d = 0;
        for (; d + 1 < key.w; d += 2)
        {
            kpp[0] = kptr[0];
            kpp[panel_width] = kptr[1];
            kptr += 2;
            kpp += panel_width * 2;
        }
        for (; d < key.w; d++)
        {
            *kpp = *kptr++;
            kpp += panel_width;
        }

        const float* vptr = value.channel(g);
        float* vpanel = (float*)cached_value.channel(g) + (size_t)panel_id * value.w * panel_width;
        d = 0;
        float* vpp;
        const float* pV;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        vpp = vpanel + token_lane * 16;
        pV = vptr;
        for (; d + 15 < value.w; d += 16)
        {
            _mm512_storeu_ps(vpp, _mm512_loadu_ps(pV));
            pV += 16;
            vpp += panel_width * 16;
        }
#endif // __AVX512F__
        vpp = vpanel + (size_t)d * panel_width + token_lane * 8;
        pV = vptr + d;
        for (; d + 7 < value.w; d += 8)
        {
            _mm256_storeu_ps(vpp, _mm256_loadu_ps(pV));
            pV += 8;
            vpp += panel_width * 8;
        }
#endif // __AVX__
        vpp = vpanel + (size_t)d * panel_width + token_lane * 4;
        pV = vptr + d;
        for (; d + 3 < value.w; d += 4)
        {
            _mm_storeu_ps(vpp, _mm_loadu_ps(pV));
            pV += 4;
            vpp += panel_width * 4;
        }
#endif // __SSE2__
        vpp = vpanel + (size_t)d * panel_width + token_lane * 2;
        pV = vptr + d;
        for (; d + 1 < value.w; d += 2)
        {
            vpp[0] = pV[0];
            vpp[1] = pV[1];
            pV += 2;
            vpp += panel_width * 2;
        }
        vpp = vpanel + (size_t)d * panel_width + token_lane;
        pV = vptr + d;
        for (; d < value.w; d++)
        {
            *vpp = *pV++;
            vpp += panel_width;
        }
    }
}

static void sdpa_attention_tile(const Mat& queryT, const Mat& key_head, const Mat& packed_key_head, const Mat& value_head, const Mat& packed_value_head, const Mat& computation_value_head, const Mat& mask, size_t mask_hstep, const Mat& packed_mask, Mat& scoreT, Mat& outT, Mat& lT, int max_ii)
{
    const int head_dim = packed_key_head.empty() ? key_head.w : packed_key_head.w;
    const int value_dim = packed_value_head.empty() ? value_head.w : packed_value_head.w;
    const int key_seqlen = packed_key_head.empty() ? key_head.h : packed_key_head.h;
    const int TILE_M = lT.w;
    const int TILE_N = scoreT.w / TILE_M;
#if __AVX512F__
    const int NR = 16;
#elif __AVX__
    const int NR = 8;
#elif __SSE2__
    const int NR = 4;
#else
    const int NR = 2;
#endif

    const float* queryT_ptr = queryT;
    float* scoreT_ptr = scoreT;
    float* outT_ptr = outT;
    float* lptr = lT;
    const float* packed_mask_data = packed_mask;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        __m512 _m = _mm512_set1_ps(-FLT_MAX);
        __m512 _l = _mm512_setzero_ps();
        memset(outT_ptr + (size_t)ii * value_dim, 0, (size_t)value_dim * 16 * sizeof(float));

        for (int n = 0; n < key_seqlen; n += TILE_N)
        {
            const int max_jj = std::min(key_seqlen - n, TILE_N);
            const float* pQ = queryT_ptr + (size_t)ii * head_dim;
            float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N;
            __m512 _block_max = _mm512_set1_ps(-FLT_MAX);

            // qk
            {
                if (!key_head.empty())
                {
                    const float* pK = key_head.row(n);
                    float* pS = scoreptr;
                    const float* maskptr = mask.empty() ? 0 : (const float*)mask + n;
                    const __m512i _mask_index = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32((int)mask_hstep));
                    for (int j = 0; j < max_jj; j++)
                    {
                        __m512 _sum0 = _mm512_setzero_ps();
                        __m512 _sum1 = _mm512_setzero_ps();
                        __m512 _sum2 = _mm512_setzero_ps();
                        __m512 _sum3 = _mm512_setzero_ps();
                        const float* pA = pQ;
                        int d = 0;
                        for (; d + 3 < head_dim; d += 4)
                        {
                            _sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(pA), _mm512_set1_ps(pK[0]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(pA + 16), _mm512_set1_ps(pK[1]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(pA + 32), _mm512_set1_ps(pK[2]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(pA + 48), _mm512_set1_ps(pK[3]), _sum3);
                            pA += 64;
                            pK += 4;
                        }
                        for (; d < head_dim; d++)
                        {
                            _sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(pA), _mm512_set1_ps(*pK), _sum0);
                            pA += 16;
                            pK++;
                        }
                        __m512 _sum = _mm512_add_ps(_mm512_add_ps(_sum0, _sum1), _mm512_add_ps(_sum2, _sum3));
                        if (maskptr)
                        {
                            if (mask_hstep == 0)
                                _sum = _mm512_add_ps(_sum, _mm512_set1_ps(*maskptr));
                            else
                                _sum = _mm512_add_ps(_sum, _mm512_i32gather_ps(_mask_index, maskptr, sizeof(float)));
                            maskptr++;
                        }
                        _mm512_storeu_ps(pS, _sum);
                        pS += 16;
                        _block_max = _mm512_max_ps(_block_max, _sum);
                    }
                }
                else
                {
                    const float* packed_maskptr = packed_mask_data ? packed_mask_data + (size_t)ii * key_seqlen + (size_t)n * 16 : 0;
                    const float* key_panel = (const float*)packed_key_head + (size_t)n * head_dim;
                    float* score_panel = scoreptr;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
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
                            const float* pK = key_panel + j;
                            const float* pA = pQ;
                            for (int d = 0; d < head_dim; d++)
                            {
                                __m512 _q = _mm512_loadu_ps(pA);
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
                            if (packed_maskptr)
                            {
                                _sum0 = _mm512_add_ps(_sum0, _mm512_loadu_ps(packed_maskptr));
                                _sum1 = _mm512_add_ps(_sum1, _mm512_loadu_ps(packed_maskptr + 16));
                                _sum2 = _mm512_add_ps(_sum2, _mm512_loadu_ps(packed_maskptr + 32));
                                _sum3 = _mm512_add_ps(_sum3, _mm512_loadu_ps(packed_maskptr + 48));
                                _sum4 = _mm512_add_ps(_sum4, _mm512_loadu_ps(packed_maskptr + 64));
                                _sum5 = _mm512_add_ps(_sum5, _mm512_loadu_ps(packed_maskptr + 80));
                                _sum6 = _mm512_add_ps(_sum6, _mm512_loadu_ps(packed_maskptr + 96));
                                _sum7 = _mm512_add_ps(_sum7, _mm512_loadu_ps(packed_maskptr + 112));
                                _sum8 = _mm512_add_ps(_sum8, _mm512_loadu_ps(packed_maskptr + 128));
                                _sum9 = _mm512_add_ps(_sum9, _mm512_loadu_ps(packed_maskptr + 144));
                                _suma = _mm512_add_ps(_suma, _mm512_loadu_ps(packed_maskptr + 160));
                                _sumb = _mm512_add_ps(_sumb, _mm512_loadu_ps(packed_maskptr + 176));
                                _sumc = _mm512_add_ps(_sumc, _mm512_loadu_ps(packed_maskptr + 192));
                                _sumd = _mm512_add_ps(_sumd, _mm512_loadu_ps(packed_maskptr + 208));
                                _sume = _mm512_add_ps(_sume, _mm512_loadu_ps(packed_maskptr + 224));
                                _sumf = _mm512_add_ps(_sumf, _mm512_loadu_ps(packed_maskptr + 240));
                                packed_maskptr += 256;
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
                            const float* pK = key_panel + j;
                            const float* pA = pQ;
                            for (int d = 0; d < head_dim; d++)
                            {
                                __m512 _q = _mm512_loadu_ps(pA);
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
                            if (packed_maskptr)
                            {
                                _sum0 = _mm512_add_ps(_sum0, _mm512_loadu_ps(packed_maskptr));
                                _sum1 = _mm512_add_ps(_sum1, _mm512_loadu_ps(packed_maskptr + 16));
                                _sum2 = _mm512_add_ps(_sum2, _mm512_loadu_ps(packed_maskptr + 32));
                                _sum3 = _mm512_add_ps(_sum3, _mm512_loadu_ps(packed_maskptr + 48));
                                _sum4 = _mm512_add_ps(_sum4, _mm512_loadu_ps(packed_maskptr + 64));
                                _sum5 = _mm512_add_ps(_sum5, _mm512_loadu_ps(packed_maskptr + 80));
                                _sum6 = _mm512_add_ps(_sum6, _mm512_loadu_ps(packed_maskptr + 96));
                                _sum7 = _mm512_add_ps(_sum7, _mm512_loadu_ps(packed_maskptr + 112));
                                packed_maskptr += 128;
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
                            __m512 _max0 = _mm512_max_ps(_sum0, _sum4);
                            __m512 _max1 = _mm512_max_ps(_sum1, _sum5);
                            __m512 _max2 = _mm512_max_ps(_sum2, _sum6);
                            __m512 _max3 = _mm512_max_ps(_sum3, _sum7);
                            _block_max = _mm512_max_ps(_block_max, _mm512_max_ps(_mm512_max_ps(_max0, _max1), _mm512_max_ps(_max2, _max3)));
                        }
#endif // defined(__x86_64__) || defined(_M_X64)
                        for (; j + 3 < max_nn; j += 4)
                        {
                            __m512 _sum0 = _mm512_setzero_ps();
                            __m512 _sum1 = _mm512_setzero_ps();
                            __m512 _sum2 = _mm512_setzero_ps();
                            __m512 _sum3 = _mm512_setzero_ps();
                            const float* pK = key_panel + j;
                            const float* pA = pQ;
                            for (int d = 0; d < head_dim; d++)
                            {
                                __m512 _q = _mm512_loadu_ps(pA);
                                _sum0 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[0]), _sum0);
                                _sum1 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[1]), _sum1);
                                _sum2 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[2]), _sum2);
                                _sum3 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[3]), _sum3);
                                pA += 16;
                                pK += NR;
                            }
                            if (packed_maskptr)
                            {
                                _sum0 = _mm512_add_ps(_sum0, _mm512_loadu_ps(packed_maskptr));
                                _sum1 = _mm512_add_ps(_sum1, _mm512_loadu_ps(packed_maskptr + 16));
                                _sum2 = _mm512_add_ps(_sum2, _mm512_loadu_ps(packed_maskptr + 32));
                                _sum3 = _mm512_add_ps(_sum3, _mm512_loadu_ps(packed_maskptr + 48));
                                packed_maskptr += 64;
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
                            const float* pK = key_panel + j;
                            const float* pA = pQ;
                            for (int d = 0; d < head_dim; d++)
                            {
                                __m512 _q = _mm512_loadu_ps(pA);
                                _sum0 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[0]), _sum0);
                                _sum1 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[1]), _sum1);
                                pA += 16;
                                pK += NR;
                            }
                            if (packed_maskptr)
                            {
                                _sum0 = _mm512_add_ps(_sum0, _mm512_loadu_ps(packed_maskptr));
                                _sum1 = _mm512_add_ps(_sum1, _mm512_loadu_ps(packed_maskptr + 16));
                                packed_maskptr += 32;
                            }
                            _mm512_storeu_ps(score_panel, _sum0);
                            _mm512_storeu_ps(score_panel + 16, _sum1);
                            score_panel += 32;
                            _block_max = _mm512_max_ps(_block_max, _mm512_max_ps(_sum0, _sum1));
                        }
                        for (; j < max_nn; j++)
                        {
                            __m512 _sum = _mm512_setzero_ps();
                            const float* pK = key_panel + j;
                            const float* pA = pQ;
                            for (int d = 0; d < head_dim; d++)
                            {
                                _sum = _mm512_fmadd_ps(_mm512_loadu_ps(pA), _mm512_set1_ps(*pK), _sum);
                                pA += 16;
                                pK += NR;
                            }
                            if (packed_maskptr)
                            {
                                _sum = _mm512_add_ps(_sum, _mm512_loadu_ps(packed_maskptr));
                                packed_maskptr += 16;
                            }
                            _mm512_storeu_ps(score_panel, _sum);
                            score_panel += 16;
                            _block_max = _mm512_max_ps(_block_max, _sum);
                        }

                        key_panel += (size_t)head_dim * NR;
                    }
                }
            }

            __m512 _alpha;

            // online softmax
            {
                __m512 _m_new = _mm512_max_ps(_m, _block_max);

                __m512 _sum0 = _mm512_setzero_ps();
                float* pS = scoreptr;
                int j = 0;
#if defined(__x86_64__) || defined(_M_X64)
                __m512 _sum1 = _mm512_setzero_ps();
                __m512 _sum2 = _mm512_setzero_ps();
                __m512 _sum3 = _mm512_setzero_ps();
                for (; j + 3 < max_jj; j += 4)
                {
                    __m512 _p0 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(pS), _m_new));
                    __m512 _p1 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(pS + 16), _m_new));
                    __m512 _p2 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(pS + 32), _m_new));
                    __m512 _p3 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(pS + 48), _m_new));
                    _mm512_storeu_ps(pS, _p0);
                    _mm512_storeu_ps(pS + 16, _p1);
                    _mm512_storeu_ps(pS + 32, _p2);
                    _mm512_storeu_ps(pS + 48, _p3);
                    pS += 64;
                    _sum0 = _mm512_add_ps(_sum0, _p0);
                    _sum1 = _mm512_add_ps(_sum1, _p1);
                    _sum2 = _mm512_add_ps(_sum2, _p2);
                    _sum3 = _mm512_add_ps(_sum3, _p3);
                }
#endif // defined(__x86_64__) || defined(_M_X64)
                for (; j < max_jj; j++)
                {
                    __m512 _p = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(pS), _m_new));
                    _mm512_storeu_ps(pS, _p);
                    pS += 16;
                    _sum0 = _mm512_add_ps(_sum0, _p);
                }
                __m512 _sum = _sum0;
#if defined(__x86_64__) || defined(_M_X64)
                _sum = _mm512_add_ps(_mm512_add_ps(_sum, _sum1), _mm512_add_ps(_sum2, _sum3));
#endif // defined(__x86_64__) || defined(_M_X64)
                __mmask16 alpha_active = _mm512_cmp_ps_mask(_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                _alpha = _mm512_maskz_mov_ps(alpha_active, exp512_ps(_mm512_maskz_sub_ps(alpha_active, _m, _m_new)));
                _m = _m_new;
                _l = _mm512_add_ps(_mm512_mul_ps(_l, _alpha), _sum);
            }

            // pv
            {
                if (!value_head.empty())
                {
                    float* outptr = outT_ptr + (size_t)ii * value_dim;
                    const float* valueptr = value_head.row(n);
                    int d = 0;
#if defined(__x86_64__) || defined(_M_X64)
                    for (; d + 7 < value_dim; d += 8)
                    {
                        __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr), _alpha);
                        __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 16), _alpha);
                        __m512 _out2 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 32), _alpha);
                        __m512 _out3 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 48), _alpha);
                        __m512 _out4 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 64), _alpha);
                        __m512 _out5 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 80), _alpha);
                        __m512 _out6 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 96), _alpha);
                        __m512 _out7 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 112), _alpha);
                        const float* pS = scoreptr;
                        const float* pV = valueptr;
                        for (int j = 0; j < max_jj; j++)
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
                            pV += value_dim;
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
                        valueptr += 8;
                    }
#endif // defined(__x86_64__) || defined(_M_X64)
                    for (; d + 3 < value_dim; d += 4)
                    {
                        __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr), _alpha);
                        __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 16), _alpha);
                        __m512 _out2 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 32), _alpha);
                        __m512 _out3 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 48), _alpha);
                        const float* pS = scoreptr;
                        const float* pV = valueptr;
                        for (int j = 0; j < max_jj; j++)
                        {
                            const __m512 _p = _mm512_loadu_ps(pS);
                            _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[0]), _out0);
                            _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[1]), _out1);
                            _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[2]), _out2);
                            _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[3]), _out3);
                            pS += 16;
                            pV += value_dim;
                        }
                        _mm512_storeu_ps(outptr, _out0);
                        _mm512_storeu_ps(outptr + 16, _out1);
                        _mm512_storeu_ps(outptr + 32, _out2);
                        _mm512_storeu_ps(outptr + 48, _out3);
                        outptr += 64;
                        valueptr += 4;
                    }
                    for (; d < value_dim; d++)
                    {
                        __m512 _out = _mm512_mul_ps(_mm512_loadu_ps(outptr), _alpha);
                        const float* pS = scoreptr;
                        const float* pV = valueptr;
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
                else if (computation_value_head.empty())
                {
                    float* outptr = outT_ptr + (size_t)ii * value_dim;
                    for (int d = 0; d < value_dim;)
                    {
                        const int value_panel_width = sdpa_value_panel_width(value_dim - d);
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
                            const float* pS = scoreptr;
                            const float* value_panel = (const float*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
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
                                    _out8 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[8]), _out8);
                                    _out9 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[9]), _out9);
                                    _outa = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[10]), _outa);
                                    _outb = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[11]), _outb);
                                    _outc = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[12]), _outc);
                                    _outd = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[13]), _outd);
                                    _oute = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[14]), _oute);
                                    _outf = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[15]), _outf);
                                    pS += 16;
                                    pV += 16;
                                }
                                value_panel += (size_t)value_dim * NR;
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
                            const float* pS = scoreptr;
                            const float* value_panel = (const float*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pV = value_panel + lane;
                                for (int j = 0; j < max_nn; j++)
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
                                    pV += value_panel_width;
                                }
                                value_panel += (size_t)value_dim * NR;
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
                            const float* value_panel = (const float*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreptr + (size_t)jj * 16;
                                const float* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    __m512 _p = _mm512_loadu_ps(pS);
                                    _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[0]), _out0);
                                    _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[1]), _out1);
                                    _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[2]), _out2);
                                    _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[3]), _out3);
                                    pS += 16;
                                    pV += value_panel_width;
                                }
                                value_panel += (size_t)NR * value_dim;
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
                            const float* value_panel = (const float*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreptr + (size_t)jj * 16;
                                const float* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    __m512 _p = _mm512_loadu_ps(pS);
                                    _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[0]), _out0);
                                    _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[1]), _out1);
                                    pS += 16;
                                    pV += value_panel_width;
                                }
                                value_panel += (size_t)NR * value_dim;
                            }
                            _mm512_storeu_ps(outptr, _out0);
                            _mm512_storeu_ps(outptr + 16, _out1);
                            outptr += 32;
                        }
                        for (; lane < value_panel_width; lane++)
                        {
                            __m512 _out = _mm512_mul_ps(_mm512_loadu_ps(outptr), _alpha);
                            const float* value_panel = (const float*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreptr + (size_t)jj * 16;
                                const float* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    _out = _mm512_fmadd_ps(_mm512_loadu_ps(pS), _mm512_set1_ps(*pV), _out);
                                    pS += 16;
                                    pV += value_panel_width;
                                }
                                value_panel += (size_t)NR * value_dim;
                            }
                            _mm512_storeu_ps(outptr, _out);
                            outptr += 16;
                        }
                        d += value_panel_width;
                    }
                }
                else
                {
                    const float* computation_value_tile = computation_value_head.row(n / TILE_N);
                    const float* pV = computation_value_tile;
                    float* outptr = outT_ptr + (size_t)ii * value_dim;
                    for (int d = 0; d < value_dim;)
                    {
                        const int value_panel_width = sdpa_value_panel_width(value_dim - d);
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
                            const float* pS = scoreptr;
                            const float* pV0 = pV + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                __m512 _p = _mm512_loadu_ps(pS);
                                _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[0]), _out0);
                                _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[1]), _out1);
                                _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[2]), _out2);
                                _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[3]), _out3);
                                _out4 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[4]), _out4);
                                _out5 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[5]), _out5);
                                _out6 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[6]), _out6);
                                _out7 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[7]), _out7);
                                _out8 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[8]), _out8);
                                _out9 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[9]), _out9);
                                _outa = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[10]), _outa);
                                _outb = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[11]), _outb);
                                _outc = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[12]), _outc);
                                _outd = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[13]), _outd);
                                _oute = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[14]), _oute);
                                _outf = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[15]), _outf);
                                pS += 16;
                                pV0 += 16;
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
                            const float* pS = scoreptr;
                            const float* pV0 = pV + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                __m512 _p = _mm512_loadu_ps(pS);
                                _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[0]), _out0);
                                _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[1]), _out1);
                                _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[2]), _out2);
                                _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[3]), _out3);
                                _out4 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[4]), _out4);
                                _out5 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[5]), _out5);
                                _out6 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[6]), _out6);
                                _out7 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[7]), _out7);
                                pS += 16;
                                pV0 += value_panel_width;
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
                            const float* pS = scoreptr;
                            const float* pV0 = pV + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                __m512 _p = _mm512_loadu_ps(pS);
                                _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[0]), _out0);
                                _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[1]), _out1);
                                _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[2]), _out2);
                                _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[3]), _out3);
                                pS += 16;
                                pV0 += value_panel_width;
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
                            const float* pS = scoreptr;
                            const float* pV0 = pV + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                __m512 _p = _mm512_loadu_ps(pS);
                                _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[0]), _out0);
                                _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV0[1]), _out1);
                                pS += 16;
                                pV0 += value_panel_width;
                            }
                            _mm512_storeu_ps(outptr, _out0);
                            _mm512_storeu_ps(outptr + 16, _out1);
                            outptr += 32;
                        }
                        for (; lane < value_panel_width; lane++)
                        {
                            __m512 _out = _mm512_mul_ps(_mm512_loadu_ps(outptr), _alpha);
                            const float* pS = scoreptr;
                            const float* pV0 = pV + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                _out = _mm512_fmadd_ps(_mm512_loadu_ps(pS), _mm512_set1_ps(*pV0), _out);
                                pS += 16;
                                pV0 += value_panel_width;
                            }
                            _mm512_storeu_ps(outptr, _out);
                            outptr += 16;
                        }
                        pV += (size_t)value_panel_width * max_jj;
                        d += value_panel_width;
                    }
                }
            }
        }

        _mm512_storeu_ps(lptr + ii, _l);
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        __m256 _m = _mm256_set1_ps(-FLT_MAX);
        __m256 _l = _mm256_setzero_ps();
        memset(outT_ptr + (size_t)ii * value_dim, 0, (size_t)value_dim * 8 * sizeof(float));

        for (int n = 0; n < key_seqlen; n += TILE_N)
        {
            const int max_jj = std::min(key_seqlen - n, TILE_N);
            const float* pQ = queryT_ptr + (size_t)ii * head_dim;
            float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N;
            __m256 _block_max = _mm256_set1_ps(-FLT_MAX);

            // qk
            {
                if (!key_head.empty())
                {
                    const float* pK = key_head.row(n);
                    float* pS = scoreptr;
                    const float* maskptr = mask.empty() ? 0 : (const float*)mask + n;
#if __AVX2__
                    const __m256i _mask_index = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32((int)mask_hstep));
#endif // __AVX2__
                    for (int j = 0; j < max_jj; j++)
                    {
                        __m256 _sum0 = _mm256_setzero_ps();
                        __m256 _sum1 = _mm256_setzero_ps();
                        __m256 _sum2 = _mm256_setzero_ps();
                        __m256 _sum3 = _mm256_setzero_ps();
                        const float* pA = pQ;
                        int d = 0;
                        for (; d + 3 < head_dim; d += 4)
                        {
                            _sum0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pA), _mm256_set1_ps(pK[0]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pA + 8), _mm256_set1_ps(pK[1]), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pA + 16), _mm256_set1_ps(pK[2]), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pA + 24), _mm256_set1_ps(pK[3]), _sum3);
                            pA += 32;
                            pK += 4;
                        }
                        for (; d < head_dim; d++)
                        {
                            _sum0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pA), _mm256_set1_ps(*pK), _sum0);
                            pA += 8;
                            pK++;
                        }
                        __m256 _sum = _mm256_add_ps(_mm256_add_ps(_sum0, _sum1), _mm256_add_ps(_sum2, _sum3));
                        if (maskptr)
                        {
                            if (mask_hstep == 0)
                            {
                                _sum = _mm256_add_ps(_sum, _mm256_set1_ps(*maskptr));
                            }
                            else
                            {
#if __AVX2__
                                _sum = _mm256_add_ps(_sum, _mm256_i32gather_ps(maskptr, _mask_index, sizeof(float)));
#else
                                _sum = _mm256_add_ps(_sum, _mm256_set_ps(maskptr[mask_hstep * 7], maskptr[mask_hstep * 6], maskptr[mask_hstep * 5], maskptr[mask_hstep * 4], maskptr[mask_hstep * 3], maskptr[mask_hstep * 2], maskptr[mask_hstep], maskptr[0]));
#endif // __AVX2__
                            }
                            maskptr++;
                        }
                        _mm256_storeu_ps(pS, _sum);
                        pS += 8;
                        _block_max = _mm256_max_ps(_block_max, _sum);
                    }
                }
                else
                {
                    const float* packed_maskptr = packed_mask_data ? packed_mask_data + (size_t)ii * key_seqlen + (size_t)n * 8 : 0;
                    const float* key_panel = (const float*)packed_key_head + (size_t)n * head_dim;
                    float* score_panel = scoreptr;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
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
                            const float* pK = key_panel + j;
                            const float* pA = pQ;
                            for (int d = 0; d < head_dim; d++)
                            {
                                __m256 _q = _mm256_loadu_ps(pA);
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
                            if (packed_maskptr)
                            {
                                _sum0 = _mm256_add_ps(_sum0, _mm256_loadu_ps(packed_maskptr));
                                _sum1 = _mm256_add_ps(_sum1, _mm256_loadu_ps(packed_maskptr + 8));
                                _sum2 = _mm256_add_ps(_sum2, _mm256_loadu_ps(packed_maskptr + 16));
                                _sum3 = _mm256_add_ps(_sum3, _mm256_loadu_ps(packed_maskptr + 24));
                                _sum4 = _mm256_add_ps(_sum4, _mm256_loadu_ps(packed_maskptr + 32));
                                _sum5 = _mm256_add_ps(_sum5, _mm256_loadu_ps(packed_maskptr + 40));
                                _sum6 = _mm256_add_ps(_sum6, _mm256_loadu_ps(packed_maskptr + 48));
                                _sum7 = _mm256_add_ps(_sum7, _mm256_loadu_ps(packed_maskptr + 56));
                                packed_maskptr += 64;
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
                            __m256 _sum0 = _mm256_setzero_ps();
                            __m256 _sum1 = _mm256_setzero_ps();
                            __m256 _sum2 = _mm256_setzero_ps();
                            __m256 _sum3 = _mm256_setzero_ps();
                            const float* pK = key_panel + j;
                            const float* pA = pQ;
                            for (int d = 0; d < head_dim; d++)
                            {
                                __m256 _q = _mm256_loadu_ps(pA);
                                _sum0 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[0]), _sum0);
                                _sum1 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[1]), _sum1);
                                _sum2 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[2]), _sum2);
                                _sum3 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[3]), _sum3);
                                pA += 8;
                                pK += NR;
                            }
                            if (packed_maskptr)
                            {
                                _sum0 = _mm256_add_ps(_sum0, _mm256_loadu_ps(packed_maskptr));
                                _sum1 = _mm256_add_ps(_sum1, _mm256_loadu_ps(packed_maskptr + 8));
                                _sum2 = _mm256_add_ps(_sum2, _mm256_loadu_ps(packed_maskptr + 16));
                                _sum3 = _mm256_add_ps(_sum3, _mm256_loadu_ps(packed_maskptr + 24));
                                packed_maskptr += 32;
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
                            const float* pK = key_panel + j;
                            const float* pA = pQ;
                            for (int d = 0; d < head_dim; d++)
                            {
                                __m256 _q = _mm256_loadu_ps(pA);
                                _sum0 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[0]), _sum0);
                                _sum1 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[1]), _sum1);
                                pA += 8;
                                pK += NR;
                            }
                            if (packed_maskptr)
                            {
                                _sum0 = _mm256_add_ps(_sum0, _mm256_loadu_ps(packed_maskptr));
                                _sum1 = _mm256_add_ps(_sum1, _mm256_loadu_ps(packed_maskptr + 8));
                                packed_maskptr += 16;
                            }
                            _mm256_storeu_ps(score_panel, _sum0);
                            _mm256_storeu_ps(score_panel + 8, _sum1);
                            score_panel += 16;
                            _block_max = _mm256_max_ps(_block_max, _mm256_max_ps(_sum0, _sum1));
                        }
                        for (; j < max_nn; j++)
                        {
                            __m256 _sum = _mm256_setzero_ps();
                            const float* pK = key_panel + j;
                            const float* pA = pQ;
                            for (int d = 0; d < head_dim; d++)
                            {
                                _sum = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pA), _mm256_set1_ps(*pK), _sum);
                                pA += 8;
                                pK += NR;
                            }
                            if (packed_maskptr)
                            {
                                _sum = _mm256_add_ps(_sum, _mm256_loadu_ps(packed_maskptr));
                                packed_maskptr += 8;
                            }
                            _mm256_storeu_ps(score_panel, _sum);
                            score_panel += 8;
                            _block_max = _mm256_max_ps(_block_max, _sum);
                        }

                        key_panel += (size_t)head_dim * NR;
                    }
                }
            }

            __m256 _alpha;

            // online softmax
            {
                __m256 _m_new = _mm256_max_ps(_m, _block_max);

                __m256 _sum0 = _mm256_setzero_ps();
                float* pS = scoreptr;
                int j = 0;
#if defined(__x86_64__) || defined(_M_X64)
                __m256 _sum1 = _mm256_setzero_ps();
                __m256 _sum2 = _mm256_setzero_ps();
                __m256 _sum3 = _mm256_setzero_ps();
                for (; j + 3 < max_jj; j += 4)
                {
                    __m256 _p0 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(pS), _m_new));
                    __m256 _p1 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(pS + 8), _m_new));
                    __m256 _p2 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(pS + 16), _m_new));
                    __m256 _p3 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(pS + 24), _m_new));
                    _mm256_storeu_ps(pS, _p0);
                    _mm256_storeu_ps(pS + 8, _p1);
                    _mm256_storeu_ps(pS + 16, _p2);
                    _mm256_storeu_ps(pS + 24, _p3);
                    pS += 32;
                    _sum0 = _mm256_add_ps(_sum0, _p0);
                    _sum1 = _mm256_add_ps(_sum1, _p1);
                    _sum2 = _mm256_add_ps(_sum2, _p2);
                    _sum3 = _mm256_add_ps(_sum3, _p3);
                }
#endif // defined(__x86_64__) || defined(_M_X64)
                for (; j < max_jj; j++)
                {
                    __m256 _p = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(pS), _m_new));
                    _mm256_storeu_ps(pS, _p);
                    pS += 8;
                    _sum0 = _mm256_add_ps(_sum0, _p);
                }
                __m256 _sum = _sum0;
#if defined(__x86_64__) || defined(_M_X64)
                _sum = _mm256_add_ps(_mm256_add_ps(_sum, _sum1), _mm256_add_ps(_sum2, _sum3));
#endif // defined(__x86_64__) || defined(_M_X64)
                __m256 _alpha_active = _mm256_cmp_ps(_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
                _alpha = _mm256_and_ps(_alpha_active, exp256_ps(_mm256_and_ps(_alpha_active, _mm256_sub_ps(_m, _m_new))));
                _m = _m_new;
                _l = _mm256_add_ps(_mm256_mul_ps(_l, _alpha), _sum);
            }

            // pv
            {
                if (!value_head.empty())
                {
                    float* outptr = outT_ptr + (size_t)ii * value_dim;
                    const float* valueptr = value_head.row(n);
                    int d = 0;
#if defined(__x86_64__) || defined(_M_X64)
                    for (; d + 7 < value_dim; d += 8)
                    {
                        __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr), _alpha);
                        __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 8), _alpha);
                        __m256 _out2 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 16), _alpha);
                        __m256 _out3 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 24), _alpha);
                        __m256 _out4 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 32), _alpha);
                        __m256 _out5 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 40), _alpha);
                        __m256 _out6 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 48), _alpha);
                        __m256 _out7 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 56), _alpha);
                        const float* pS = scoreptr;
                        const float* pV = valueptr;
                        for (int j = 0; j < max_jj; j++)
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
                            pV += value_dim;
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
                        valueptr += 8;
                    }
#endif // defined(__x86_64__) || defined(_M_X64)
                    for (; d + 3 < value_dim; d += 4)
                    {
                        __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr), _alpha);
                        __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 8), _alpha);
                        __m256 _out2 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 16), _alpha);
                        __m256 _out3 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 24), _alpha);
                        const float* pS = scoreptr;
                        const float* pV = valueptr;
                        for (int j = 0; j < max_jj; j++)
                        {
                            const __m256 _p = _mm256_loadu_ps(pS);
                            _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[0]), _out0);
                            _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[1]), _out1);
                            _out2 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[2]), _out2);
                            _out3 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[3]), _out3);
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
                        const float* pS = scoreptr;
                        const float* pV = valueptr;
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
                else if (computation_value_head.empty())
                {
                    float* outptr = outT_ptr + (size_t)ii * value_dim;
                    for (int d = 0; d < value_dim;)
                    {
                        const int value_panel_width = sdpa_value_panel_width(value_dim - d);
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
                            const float* pS = scoreptr;
                            const float* value_panel = (const float*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
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
                                    pV += value_panel_width;
                                }
                                value_panel += (size_t)value_dim * NR;
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
                            const float* value_panel = (const float*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreptr + (size_t)jj * 8;
                                const float* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    __m256 _p = _mm256_loadu_ps(pS);
                                    _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[0]), _out0);
                                    _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[1]), _out1);
                                    _out2 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[2]), _out2);
                                    _out3 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[3]), _out3);
                                    pS += 8;
                                    pV += value_panel_width;
                                }
                                value_panel += (size_t)NR * value_dim;
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
                            const float* value_panel = (const float*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreptr + (size_t)jj * 8;
                                const float* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    __m256 _p = _mm256_loadu_ps(pS);
                                    _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[0]), _out0);
                                    _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[1]), _out1);
                                    pS += 8;
                                    pV += value_panel_width;
                                }
                                value_panel += (size_t)NR * value_dim;
                            }
                            _mm256_storeu_ps(outptr, _out0);
                            _mm256_storeu_ps(outptr + 8, _out1);
                            outptr += 16;
                        }
                        for (; lane < value_panel_width; lane++)
                        {
                            __m256 _out = _mm256_mul_ps(_mm256_loadu_ps(outptr), _alpha);
                            const float* value_panel = (const float*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreptr + (size_t)jj * 8;
                                const float* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pS), _mm256_set1_ps(*pV), _out);
                                    pS += 8;
                                    pV += value_panel_width;
                                }
                                value_panel += (size_t)NR * value_dim;
                            }
                            _mm256_storeu_ps(outptr, _out);
                            outptr += 8;
                        }
                        d += value_panel_width;
                    }
                }
                else
                {
                    const float* computation_value_tile = computation_value_head.row(n / TILE_N);
                    const float* pV = computation_value_tile;
                    float* outptr = outT_ptr + (size_t)ii * value_dim;
                    for (int d = 0; d < value_dim;)
                    {
                        const int value_panel_width = sdpa_value_panel_width(value_dim - d);
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
                            const float* pS = scoreptr;
                            const float* pV0 = pV + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                __m256 _p = _mm256_loadu_ps(pS);
                                _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV0[0]), _out0);
                                _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV0[1]), _out1);
                                _out2 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV0[2]), _out2);
                                _out3 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV0[3]), _out3);
                                _out4 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV0[4]), _out4);
                                _out5 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV0[5]), _out5);
                                _out6 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV0[6]), _out6);
                                _out7 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV0[7]), _out7);
                                pS += 8;
                                pV0 += value_panel_width;
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
                            const float* pS = scoreptr;
                            const float* pV0 = pV + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                __m256 _p = _mm256_loadu_ps(pS);
                                _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV0[0]), _out0);
                                _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV0[1]), _out1);
                                _out2 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV0[2]), _out2);
                                _out3 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV0[3]), _out3);
                                pS += 8;
                                pV0 += value_panel_width;
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
                            const float* pS = scoreptr;
                            const float* pV0 = pV + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                __m256 _p = _mm256_loadu_ps(pS);
                                _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV0[0]), _out0);
                                _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV0[1]), _out1);
                                pS += 8;
                                pV0 += value_panel_width;
                            }
                            _mm256_storeu_ps(outptr, _out0);
                            _mm256_storeu_ps(outptr + 8, _out1);
                            outptr += 16;
                        }
                        for (; lane < value_panel_width; lane++)
                        {
                            __m256 _out = _mm256_mul_ps(_mm256_loadu_ps(outptr), _alpha);
                            const float* pS = scoreptr;
                            const float* pV0 = pV + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pS), _mm256_set1_ps(*pV0), _out);
                                pS += 8;
                                pV0 += value_panel_width;
                            }
                            _mm256_storeu_ps(outptr, _out);
                            outptr += 8;
                        }
                        pV += (size_t)value_panel_width * max_jj;
                        d += value_panel_width;
                    }
                }
            }
        }

        _mm256_storeu_ps(lptr + ii, _l);
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        __m128 _m = _mm_set1_ps(-FLT_MAX);
        __m128 _l = _mm_setzero_ps();
        memset(outT_ptr + (size_t)ii * value_dim, 0, (size_t)value_dim * 4 * sizeof(float));

        for (int n = 0; n < key_seqlen; n += TILE_N)
        {
            const int max_jj = std::min(key_seqlen - n, TILE_N);
            const float* pQ = queryT_ptr + (size_t)ii * head_dim;
            float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N;
            __m128 _block_max = _mm_set1_ps(-FLT_MAX);

            // qk
            {
                if (!key_head.empty())
                {
                    const float* pK = key_head.row(n);
                    float* pS = scoreptr;
                    const float* maskptr = mask.empty() ? 0 : (const float*)mask + n;
                    for (int j = 0; j < max_jj; j++)
                    {
                        __m128 _sum0 = _mm_setzero_ps();
                        __m128 _sum1 = _mm_setzero_ps();
                        __m128 _sum2 = _mm_setzero_ps();
                        __m128 _sum3 = _mm_setzero_ps();
                        const float* pA = pQ;
                        int d = 0;
                        for (; d + 3 < head_dim; d += 4)
                        {
                            _sum0 = _mm_comp_fmadd_ps(_mm_loadu_ps(pA), _mm_set1_ps(pK[0]), _sum0);
                            _sum1 = _mm_comp_fmadd_ps(_mm_loadu_ps(pA + 4), _mm_set1_ps(pK[1]), _sum1);
                            _sum2 = _mm_comp_fmadd_ps(_mm_loadu_ps(pA + 8), _mm_set1_ps(pK[2]), _sum2);
                            _sum3 = _mm_comp_fmadd_ps(_mm_loadu_ps(pA + 12), _mm_set1_ps(pK[3]), _sum3);
                            pA += 16;
                            pK += 4;
                        }
                        for (; d < head_dim; d++)
                        {
                            _sum0 = _mm_comp_fmadd_ps(_mm_loadu_ps(pA), _mm_set1_ps(*pK), _sum0);
                            pA += 4;
                            pK++;
                        }
                        __m128 _sum = _mm_add_ps(_mm_add_ps(_sum0, _sum1), _mm_add_ps(_sum2, _sum3));
                        if (maskptr)
                        {
                            if (mask_hstep == 0)
                                _sum = _mm_add_ps(_sum, _mm_set1_ps(*maskptr));
                            else
                                _sum = _mm_add_ps(_sum, _mm_set_ps(maskptr[mask_hstep * 3], maskptr[mask_hstep * 2], maskptr[mask_hstep], maskptr[0]));
                            maskptr++;
                        }
                        _mm_storeu_ps(pS, _sum);
                        pS += 4;
                        _block_max = _mm_max_ps(_block_max, _sum);
                    }
                }
                else
                {
                    const float* packed_maskptr = packed_mask_data ? packed_mask_data + (size_t)ii * key_seqlen + (size_t)n * 4 : 0;
                    const float* key_panel = (const float*)packed_key_head + (size_t)n * head_dim;
                    float* score_panel = scoreptr;
                    for (int jj = 0; jj < max_jj; jj += NR)
                    {
                        const int max_nn = std::min(NR, max_jj - jj);
                        int j = 0;
                        for (; j + 3 < max_nn; j += 4)
                        {
                            __m128 _sum0 = _mm_setzero_ps();
                            __m128 _sum1 = _mm_setzero_ps();
                            __m128 _sum2 = _mm_setzero_ps();
                            __m128 _sum3 = _mm_setzero_ps();
                            const float* pK = key_panel + j;
                            const float* pA = pQ;
                            for (int d = 0; d < head_dim; d++)
                            {
                                __m128 _q = _mm_loadu_ps(pA);
                                _sum0 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[0]), _sum0);
                                _sum1 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[1]), _sum1);
                                _sum2 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[2]), _sum2);
                                _sum3 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[3]), _sum3);
                                pA += 4;
                                pK += NR;
                            }
                            if (packed_maskptr)
                            {
                                _sum0 = _mm_add_ps(_sum0, _mm_loadu_ps(packed_maskptr));
                                _sum1 = _mm_add_ps(_sum1, _mm_loadu_ps(packed_maskptr + 4));
                                _sum2 = _mm_add_ps(_sum2, _mm_loadu_ps(packed_maskptr + 8));
                                _sum3 = _mm_add_ps(_sum3, _mm_loadu_ps(packed_maskptr + 12));
                                packed_maskptr += 16;
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
                            const float* pK = key_panel + j;
                            const float* pA = pQ;
                            for (int d = 0; d < head_dim; d++)
                            {
                                __m128 _q = _mm_loadu_ps(pA);
                                _sum0 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[0]), _sum0);
                                _sum1 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[1]), _sum1);
                                pA += 4;
                                pK += NR;
                            }
                            if (packed_maskptr)
                            {
                                _sum0 = _mm_add_ps(_sum0, _mm_loadu_ps(packed_maskptr));
                                _sum1 = _mm_add_ps(_sum1, _mm_loadu_ps(packed_maskptr + 4));
                                packed_maskptr += 8;
                            }
                            _mm_storeu_ps(score_panel, _sum0);
                            _mm_storeu_ps(score_panel + 4, _sum1);
                            score_panel += 8;
                            _block_max = _mm_max_ps(_block_max, _mm_max_ps(_sum0, _sum1));
                        }
                        for (; j < max_nn; j++)
                        {
                            __m128 _sum = packed_maskptr ? _mm_loadu_ps(packed_maskptr) : _mm_setzero_ps();
                            const float* pK = key_panel + j;
                            const float* pA = pQ;
                            for (int d = 0; d < head_dim; d++)
                            {
                                _sum = _mm_comp_fmadd_ps(_mm_loadu_ps(pA), _mm_set1_ps(*pK), _sum);
                                pA += 4;
                                pK += NR;
                            }
                            if (packed_maskptr)
                                packed_maskptr += 4;
                            _mm_storeu_ps(score_panel, _sum);
                            score_panel += 4;
                            _block_max = _mm_max_ps(_block_max, _sum);
                        }

                        key_panel += (size_t)head_dim * NR;
                    }
                }
            }

            __m128 _alpha;

            // online softmax
            {
                __m128 _m_new = _mm_max_ps(_m, _block_max);

                __m128 _sum0 = _mm_setzero_ps();
                float* pS = scoreptr;
                int j = 0;
#if defined(__x86_64__) || defined(_M_X64)
                __m128 _sum1 = _mm_setzero_ps();
                __m128 _sum2 = _mm_setzero_ps();
                __m128 _sum3 = _mm_setzero_ps();
                for (; j + 3 < max_jj; j += 4)
                {
                    __m128 _p0 = exp_ps(_mm_sub_ps(_mm_loadu_ps(pS), _m_new));
                    __m128 _p1 = exp_ps(_mm_sub_ps(_mm_loadu_ps(pS + 4), _m_new));
                    __m128 _p2 = exp_ps(_mm_sub_ps(_mm_loadu_ps(pS + 8), _m_new));
                    __m128 _p3 = exp_ps(_mm_sub_ps(_mm_loadu_ps(pS + 12), _m_new));
                    _mm_storeu_ps(pS, _p0);
                    _mm_storeu_ps(pS + 4, _p1);
                    _mm_storeu_ps(pS + 8, _p2);
                    _mm_storeu_ps(pS + 12, _p3);
                    pS += 16;
                    _sum0 = _mm_add_ps(_sum0, _p0);
                    _sum1 = _mm_add_ps(_sum1, _p1);
                    _sum2 = _mm_add_ps(_sum2, _p2);
                    _sum3 = _mm_add_ps(_sum3, _p3);
                }
#endif // defined(__x86_64__) || defined(_M_X64)
                for (; j < max_jj; j++)
                {
                    __m128 _p = exp_ps(_mm_sub_ps(_mm_loadu_ps(pS), _m_new));
                    _mm_storeu_ps(pS, _p);
                    pS += 4;
                    _sum0 = _mm_add_ps(_sum0, _p);
                }
                __m128 _sum = _sum0;
#if defined(__x86_64__) || defined(_M_X64)
                _sum = _mm_add_ps(_mm_add_ps(_sum, _sum1), _mm_add_ps(_sum2, _sum3));
#endif // defined(__x86_64__) || defined(_M_X64)
                __m128 _alpha_active = _mm_cmpneq_ps(_l, _mm_setzero_ps());
                _alpha = exp_ps(_mm_and_ps(_alpha_active, _mm_sub_ps(_m, _m_new)));
                _alpha = _mm_and_ps(_alpha, _alpha_active);
                _m = _m_new;
                _l = _mm_add_ps(_mm_mul_ps(_l, _alpha), _sum);
            }

            // pv
            {
                if (!value_head.empty())
                {
                    float* outptr = outT_ptr + (size_t)ii * value_dim;
                    const float* valueptr = value_head.row(n);
                    int d = 0;
#if defined(__x86_64__) || defined(_M_X64)
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
                        const float* pS = scoreptr;
                        const float* pV = valueptr;
                        for (int j = 0; j < max_jj; j++)
                        {
                            const __m128 _p = _mm_loadu_ps(pS);
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
#endif // defined(__x86_64__) || defined(_M_X64)
                    for (; d + 3 < value_dim; d += 4)
                    {
                        __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(outptr), _alpha);
                        __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(outptr + 4), _alpha);
                        __m128 _out2 = _mm_mul_ps(_mm_loadu_ps(outptr + 8), _alpha);
                        __m128 _out3 = _mm_mul_ps(_mm_loadu_ps(outptr + 12), _alpha);
                        const float* pS = scoreptr;
                        const float* pV = valueptr;
                        for (int j = 0; j < max_jj; j++)
                        {
                            const __m128 _p = _mm_loadu_ps(pS);
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
                        const float* pS = scoreptr;
                        const float* pV = valueptr;
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
                else if (computation_value_head.empty())
                {
                    float* outptr = outT_ptr + (size_t)ii * value_dim;
                    for (int d = 0; d < value_dim;)
                    {
                        const int value_panel_width = sdpa_value_panel_width(value_dim - d);
                        int lane = 0;
                        for (; lane + 3 < value_panel_width; lane += 4)
                        {
                            __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(outptr), _alpha);
                            __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(outptr + 4), _alpha);
                            __m128 _out2 = _mm_mul_ps(_mm_loadu_ps(outptr + 8), _alpha);
                            __m128 _out3 = _mm_mul_ps(_mm_loadu_ps(outptr + 12), _alpha);
                            const float* value_panel = (const float*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreptr + (size_t)jj * 4;
                                const float* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    __m128 _p = _mm_loadu_ps(pS);
                                    _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[0]), _out0);
                                    _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[1]), _out1);
                                    _out2 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[2]), _out2);
                                    _out3 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[3]), _out3);
                                    pS += 4;
                                    pV += value_panel_width;
                                }
                                value_panel += (size_t)NR * value_dim;
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
                            const float* value_panel = (const float*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreptr + (size_t)jj * 4;
                                const float* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    __m128 _p = _mm_loadu_ps(pS);
                                    _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[0]), _out0);
                                    _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[1]), _out1);
                                    pS += 4;
                                    pV += value_panel_width;
                                }
                                value_panel += (size_t)NR * value_dim;
                            }
                            _mm_storeu_ps(outptr, _out0);
                            _mm_storeu_ps(outptr + 4, _out1);
                            outptr += 8;
                        }
                        for (; lane < value_panel_width; lane++)
                        {
                            __m128 _out = _mm_mul_ps(_mm_loadu_ps(outptr), _alpha);
                            const float* value_panel = (const float*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreptr + (size_t)jj * 4;
                                const float* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    _out = _mm_comp_fmadd_ps(_mm_loadu_ps(pS), _mm_set1_ps(*pV), _out);
                                    pS += 4;
                                    pV += value_panel_width;
                                }
                                value_panel += (size_t)NR * value_dim;
                            }
                            _mm_storeu_ps(outptr, _out);
                            outptr += 4;
                        }
                        d += value_panel_width;
                    }
                }
                else
                {
                    const float* computation_value_tile = computation_value_head.row(n / TILE_N);
                    const float* pV = computation_value_tile;
                    float* outptr = outT_ptr + (size_t)ii * value_dim;
                    for (int d = 0; d < value_dim;)
                    {
                        const int value_panel_width = sdpa_value_panel_width(value_dim - d);
                        int lane = 0;
                        for (; lane + 3 < value_panel_width; lane += 4)
                        {
                            __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(outptr), _alpha);
                            __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(outptr + 4), _alpha);
                            __m128 _out2 = _mm_mul_ps(_mm_loadu_ps(outptr + 8), _alpha);
                            __m128 _out3 = _mm_mul_ps(_mm_loadu_ps(outptr + 12), _alpha);
                            const float* pS = scoreptr;
                            const float* pV0 = pV + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                __m128 _p = _mm_loadu_ps(pS);
                                _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV0[0]), _out0);
                                _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV0[1]), _out1);
                                _out2 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV0[2]), _out2);
                                _out3 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV0[3]), _out3);
                                pS += 4;
                                pV0 += value_panel_width;
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
                            const float* pS = scoreptr;
                            const float* pV0 = pV + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                __m128 _p = _mm_loadu_ps(pS);
                                _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV0[0]), _out0);
                                _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV0[1]), _out1);
                                pS += 4;
                                pV0 += value_panel_width;
                            }
                            _mm_storeu_ps(outptr, _out0);
                            _mm_storeu_ps(outptr + 4, _out1);
                            outptr += 8;
                        }
                        for (; lane < value_panel_width; lane++)
                        {
                            __m128 _out = _mm_mul_ps(_mm_loadu_ps(outptr), _alpha);
                            const float* pS = scoreptr;
                            const float* pV0 = pV + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                _out = _mm_comp_fmadd_ps(_mm_loadu_ps(pS), _mm_set1_ps(*pV0), _out);
                                pS += 4;
                                pV0 += value_panel_width;
                            }
                            _mm_storeu_ps(outptr, _out);
                            outptr += 4;
                        }
                        pV += (size_t)value_panel_width * max_jj;
                        d += value_panel_width;
                    }
                }
            }
        }

        _mm_storeu_ps(lptr + ii, _l);
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        float m0 = -FLT_MAX;
        float m1 = -FLT_MAX;
        float l0 = 0.f;
        float l1 = 0.f;
        float* out0 = outT_ptr + (size_t)ii * value_dim;
        float* out1 = out0 + value_dim;
        memset(out0, 0, (size_t)value_dim * 2 * sizeof(float));

        for (int n = 0; n < key_seqlen; n += TILE_N)
        {
            const int max_jj = std::min(key_seqlen - n, TILE_N);
            const float* pQ0 = queryT_ptr + (size_t)ii * head_dim;
            const float* pQ1 = pQ0 + head_dim;
            float* score0 = scoreT_ptr + (size_t)ii * TILE_N;
            float* score1 = score0 + TILE_N;
            const float* packed_mask0 = packed_mask_data ? packed_mask_data + (size_t)ii * key_seqlen + n : 0;
            const float* packed_mask1 = packed_mask0 ? packed_mask0 + key_seqlen : 0;
            float block_max0 = -FLT_MAX;
            float block_max1 = -FLT_MAX;

            // qk
            {
                const float* key_panel = (const float*)packed_key_head + (size_t)n * head_dim;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    float* scoreptr0 = score0 + jj;
                    float* scoreptr1 = score1 + jj;
                    const float* packed_maskptr0 = packed_mask0 ? packed_mask0 + jj : 0;
                    const float* packed_maskptr1 = packed_mask1 ? packed_mask1 + jj : 0;
                    int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    for (; j + 15 < max_nn; j += 16)
                    {
                        __m512 _sum0 = _mm512_setzero_ps();
                        __m512 _sum1 = _mm512_setzero_ps();
                        const float* pK = key_panel + j;
                        const float* pA0 = pQ0;
                        const float* pA1 = pQ1;
                        for (int d = 0; d < head_dim; d++)
                        {
                            __m512 _k = _mm512_loadu_ps(pK);
                            _sum0 = _mm512_fmadd_ps(_k, _mm512_set1_ps(*pA0++), _sum0);
                            _sum1 = _mm512_fmadd_ps(_k, _mm512_set1_ps(*pA1++), _sum1);
                            pK += NR;
                        }
                        if (packed_maskptr0)
                        {
                            _sum0 = _mm512_add_ps(_sum0, _mm512_loadu_ps(packed_maskptr0));
                            _sum1 = _mm512_add_ps(_sum1, _mm512_loadu_ps(packed_maskptr1));
                            packed_maskptr0 += 16;
                            packed_maskptr1 += 16;
                        }
                        _mm512_storeu_ps(scoreptr0, _sum0);
                        _mm512_storeu_ps(scoreptr1, _sum1);
                        scoreptr0 += 16;
                        scoreptr1 += 16;
                        block_max0 = std::max(block_max0, _mm512_comp_reduce_max_ps(_sum0));
                        block_max1 = std::max(block_max1, _mm512_comp_reduce_max_ps(_sum1));
                    }
#endif // __AVX512F__
                    for (; j + 7 < max_nn; j += 8)
                    {
                        __m256 _sum0 = _mm256_setzero_ps();
                        __m256 _sum1 = _mm256_setzero_ps();
                        const float* pK = key_panel + j;
                        const float* pA0 = pQ0;
                        const float* pA1 = pQ1;
                        for (int d = 0; d < head_dim; d++)
                        {
                            __m256 _k = _mm256_loadu_ps(pK);
                            _sum0 = _mm256_comp_fmadd_ps(_k, _mm256_set1_ps(*pA0++), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_k, _mm256_set1_ps(*pA1++), _sum1);
                            pK += NR;
                        }
                        if (packed_maskptr0)
                        {
                            _sum0 = _mm256_add_ps(_sum0, _mm256_loadu_ps(packed_maskptr0));
                            _sum1 = _mm256_add_ps(_sum1, _mm256_loadu_ps(packed_maskptr1));
                            packed_maskptr0 += 8;
                            packed_maskptr1 += 8;
                        }
                        _mm256_storeu_ps(scoreptr0, _sum0);
                        _mm256_storeu_ps(scoreptr1, _sum1);
                        scoreptr0 += 8;
                        scoreptr1 += 8;
                        block_max0 = std::max(block_max0, _mm256_reduce_max_ps(_sum0));
                        block_max1 = std::max(block_max1, _mm256_reduce_max_ps(_sum1));
                    }
#endif // __AVX__
                    for (; j + 3 < max_nn; j += 4)
                    {
                        __m128 _sum0 = _mm_setzero_ps();
                        __m128 _sum1 = _mm_setzero_ps();
                        const float* pK = key_panel + j;
                        const float* pA0 = pQ0;
                        const float* pA1 = pQ1;
                        for (int d = 0; d < head_dim; d++)
                        {
                            __m128 _k = _mm_loadu_ps(pK);
                            _sum0 = _mm_comp_fmadd_ps(_k, _mm_set1_ps(*pA0++), _sum0);
                            _sum1 = _mm_comp_fmadd_ps(_k, _mm_set1_ps(*pA1++), _sum1);
                            pK += NR;
                        }
                        if (packed_maskptr0)
                        {
                            _sum0 = _mm_add_ps(_sum0, _mm_loadu_ps(packed_maskptr0));
                            _sum1 = _mm_add_ps(_sum1, _mm_loadu_ps(packed_maskptr1));
                            packed_maskptr0 += 4;
                            packed_maskptr1 += 4;
                        }
                        _mm_storeu_ps(scoreptr0, _sum0);
                        _mm_storeu_ps(scoreptr1, _sum1);
                        scoreptr0 += 4;
                        scoreptr1 += 4;
                        block_max0 = std::max(block_max0, _mm_reduce_max_ps(_sum0));
                        block_max1 = std::max(block_max1, _mm_reduce_max_ps(_sum1));
                    }
#endif // __SSE2__
                    for (; j + 1 < max_nn; j += 2)
                    {
                        float sum00 = packed_maskptr0 ? packed_maskptr0[0] : 0.f;
                        float sum01 = packed_maskptr0 ? packed_maskptr0[1] : 0.f;
                        float sum10 = packed_maskptr1 ? packed_maskptr1[0] : 0.f;
                        float sum11 = packed_maskptr1 ? packed_maskptr1[1] : 0.f;
                        const float* pK = key_panel + j;
                        const float* pA0 = pQ0;
                        const float* pA1 = pQ1;
                        for (int d = 0; d < head_dim; d++)
                        {
                            const float k0 = pK[0];
                            const float k1 = pK[1];
                            sum00 += pA0[0] * k0;
                            sum01 += pA0[0] * k1;
                            sum10 += pA1[0] * k0;
                            sum11 += pA1[0] * k1;
                            pA0++;
                            pA1++;
                            pK += NR;
                        }
                        scoreptr0[0] = sum00;
                        scoreptr0[1] = sum01;
                        scoreptr1[0] = sum10;
                        scoreptr1[1] = sum11;
                        scoreptr0 += 2;
                        scoreptr1 += 2;
                        if (packed_maskptr0)
                        {
                            packed_maskptr0 += 2;
                            packed_maskptr1 += 2;
                        }
                        block_max0 = std::max(block_max0, std::max(sum00, sum01));
                        block_max1 = std::max(block_max1, std::max(sum10, sum11));
                    }
                    for (; j < max_nn; j++)
                    {
                        float sum0 = packed_maskptr0 ? *packed_maskptr0 : 0.f;
                        float sum1 = packed_maskptr1 ? *packed_maskptr1 : 0.f;
                        const float* pK = key_panel + j;
                        for (int d = 0; d < head_dim; d++)
                        {
                            const float v = *pK;
                            sum0 += pQ0[d] * v;
                            sum1 += pQ1[d] * v;
                            pK += NR;
                        }
                        *scoreptr0++ = sum0;
                        *scoreptr1++ = sum1;
                        if (packed_maskptr0)
                        {
                            packed_maskptr0++;
                            packed_maskptr1++;
                        }
                        block_max0 = std::max(block_max0, sum0);
                        block_max1 = std::max(block_max1, sum1);
                    }
                    key_panel += (size_t)head_dim * NR;
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
                m0 = m_new0;
                m1 = m_new1;
                l0 = l0 * alpha0 + sum0;
                l1 = l1 * alpha1 + sum1;
            }

            // pv
            {
                if (computation_value_head.empty())
                {
                    for (int d = 0; d < value_dim;)
                    {
                        const int value_panel_width = sdpa_value_panel_width(value_dim - d);
                        const float* value_panel = (const float*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR;
                        float* outptr0 = out0 + d;
                        float* outptr1 = out1 + d;
                        int lane = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                        for (; lane + 15 < value_panel_width; lane += 16)
                        {
                            __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr0), _mm512_set1_ps(alpha0));
                            __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(outptr1), _mm512_set1_ps(alpha1));
                            const float* pV_panel = value_panel + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pV = pV_panel;
                                const float* scoreptr0 = score0 + jj;
                                const float* scoreptr1 = score1 + jj;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    __m512 _v = _mm512_loadu_ps(pV);
                                    _out0 = _mm512_fmadd_ps(_v, _mm512_set1_ps(*scoreptr0++), _out0);
                                    _out1 = _mm512_fmadd_ps(_v, _mm512_set1_ps(*scoreptr1++), _out1);
                                    pV += value_panel_width;
                                }
                                pV_panel += (size_t)NR * value_dim;
                            }
                            _mm512_storeu_ps(outptr0, _out0);
                            _mm512_storeu_ps(outptr1, _out1);
                            outptr0 += 16;
                            outptr1 += 16;
                        }
#endif // __AVX512F__
                        for (; lane + 7 < value_panel_width; lane += 8)
                        {
                            __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr0), _mm256_set1_ps(alpha0));
                            __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr1), _mm256_set1_ps(alpha1));
                            const float* pV_panel = value_panel + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pV = pV_panel;
                                const float* scoreptr0 = score0 + jj;
                                const float* scoreptr1 = score1 + jj;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    __m256 _v = _mm256_loadu_ps(pV);
                                    _out0 = _mm256_comp_fmadd_ps(_v, _mm256_set1_ps(*scoreptr0++), _out0);
                                    _out1 = _mm256_comp_fmadd_ps(_v, _mm256_set1_ps(*scoreptr1++), _out1);
                                    pV += value_panel_width;
                                }
                                pV_panel += (size_t)NR * value_dim;
                            }
                            _mm256_storeu_ps(outptr0, _out0);
                            _mm256_storeu_ps(outptr1, _out1);
                            outptr0 += 8;
                            outptr1 += 8;
                        }
#endif // __AVX__
                        for (; lane + 3 < value_panel_width; lane += 4)
                        {
                            __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(outptr0), _mm_set1_ps(alpha0));
                            __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(outptr1), _mm_set1_ps(alpha1));
                            const float* pV_panel = value_panel + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pV = pV_panel;
                                const float* scoreptr0 = score0 + jj;
                                const float* scoreptr1 = score1 + jj;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    __m128 _v = _mm_loadu_ps(pV);
                                    _out0 = _mm_comp_fmadd_ps(_v, _mm_set1_ps(*scoreptr0++), _out0);
                                    _out1 = _mm_comp_fmadd_ps(_v, _mm_set1_ps(*scoreptr1++), _out1);
                                    pV += value_panel_width;
                                }
                                pV_panel += (size_t)NR * value_dim;
                            }
                            _mm_storeu_ps(outptr0, _out0);
                            _mm_storeu_ps(outptr1, _out1);
                            outptr0 += 4;
                            outptr1 += 4;
                        }
#endif // __SSE2__
                        for (; lane + 1 < value_panel_width; lane += 2)
                        {
                            float out00 = outptr0[0] * alpha0;
                            float out01 = outptr0[1] * alpha0;
                            float out10 = outptr1[0] * alpha1;
                            float out11 = outptr1[1] * alpha1;
                            const float* pV_panel = value_panel + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pV = pV_panel;
                                const float* scoreptr0 = score0 + jj;
                                const float* scoreptr1 = score1 + jj;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    const float v0 = pV[0];
                                    const float v1 = pV[1];
                                    out00 += *scoreptr0 * v0;
                                    out01 += *scoreptr0++ * v1;
                                    out10 += *scoreptr1 * v0;
                                    out11 += *scoreptr1++ * v1;
                                    pV += value_panel_width;
                                }
                                pV_panel += (size_t)NR * value_dim;
                            }
                            outptr0[0] = out00;
                            outptr0[1] = out01;
                            outptr1[0] = out10;
                            outptr1[1] = out11;
                            outptr0 += 2;
                            outptr1 += 2;
                        }
                        for (; lane < value_panel_width; lane++)
                        {
                            float out00 = *outptr0 * alpha0;
                            float out10 = *outptr1 * alpha1;
                            const float* pV_panel = value_panel + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pV = pV_panel;
                                const float* scoreptr0 = score0 + jj;
                                const float* scoreptr1 = score1 + jj;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    out00 += *scoreptr0++ * *pV;
                                    out10 += *scoreptr1++ * *pV;
                                    pV += value_panel_width;
                                }
                                pV_panel += (size_t)NR * value_dim;
                            }
                            *outptr0++ = out00;
                            *outptr1++ = out10;
                        }
                        d += value_panel_width;
                    }
                }
                else
                {
                    const float* pV = computation_value_head.row(n / TILE_N);
                    for (int d = 0; d < value_dim;)
                    {
                        const int value_panel_width = sdpa_value_panel_width(value_dim - d);
                        float* outptr0 = out0 + d;
                        float* outptr1 = out1 + d;
                        int lane = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                        for (; lane + 15 < value_panel_width; lane += 16)
                        {
                            __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr0), _mm512_set1_ps(alpha0));
                            __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(outptr1), _mm512_set1_ps(alpha1));
                            const float* pV0 = pV + lane;
                            const float* scoreptr0 = score0;
                            const float* scoreptr1 = score1;
                            for (int j = 0; j < max_jj; j++)
                            {
                                __m512 _v = _mm512_loadu_ps(pV0);
                                _out0 = _mm512_fmadd_ps(_v, _mm512_set1_ps(*scoreptr0++), _out0);
                                _out1 = _mm512_fmadd_ps(_v, _mm512_set1_ps(*scoreptr1++), _out1);
                                pV0 += value_panel_width;
                            }
                            _mm512_storeu_ps(outptr0, _out0);
                            _mm512_storeu_ps(outptr1, _out1);
                            outptr0 += 16;
                            outptr1 += 16;
                        }
#endif // __AVX512F__
                        for (; lane + 7 < value_panel_width; lane += 8)
                        {
                            __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr0), _mm256_set1_ps(alpha0));
                            __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr1), _mm256_set1_ps(alpha1));
                            const float* pV0 = pV + lane;
                            const float* scoreptr0 = score0;
                            const float* scoreptr1 = score1;
                            for (int j = 0; j < max_jj; j++)
                            {
                                __m256 _v = _mm256_loadu_ps(pV0);
                                _out0 = _mm256_comp_fmadd_ps(_v, _mm256_set1_ps(*scoreptr0++), _out0);
                                _out1 = _mm256_comp_fmadd_ps(_v, _mm256_set1_ps(*scoreptr1++), _out1);
                                pV0 += value_panel_width;
                            }
                            _mm256_storeu_ps(outptr0, _out0);
                            _mm256_storeu_ps(outptr1, _out1);
                            outptr0 += 8;
                            outptr1 += 8;
                        }
#endif // __AVX__
                        for (; lane + 3 < value_panel_width; lane += 4)
                        {
                            __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(outptr0), _mm_set1_ps(alpha0));
                            __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(outptr1), _mm_set1_ps(alpha1));
                            const float* pV0 = pV + lane;
                            const float* scoreptr0 = score0;
                            const float* scoreptr1 = score1;
                            for (int j = 0; j < max_jj; j++)
                            {
                                __m128 _v = _mm_loadu_ps(pV0);
                                _out0 = _mm_comp_fmadd_ps(_v, _mm_set1_ps(*scoreptr0++), _out0);
                                _out1 = _mm_comp_fmadd_ps(_v, _mm_set1_ps(*scoreptr1++), _out1);
                                pV0 += value_panel_width;
                            }
                            _mm_storeu_ps(outptr0, _out0);
                            _mm_storeu_ps(outptr1, _out1);
                            outptr0 += 4;
                            outptr1 += 4;
                        }
#endif // __SSE2__
                        for (; lane + 1 < value_panel_width; lane += 2)
                        {
                            float out00 = outptr0[0] * alpha0;
                            float out01 = outptr0[1] * alpha0;
                            float out10 = outptr1[0] * alpha1;
                            float out11 = outptr1[1] * alpha1;
                            const float* pV0 = pV + lane;
                            const float* scoreptr0 = score0;
                            const float* scoreptr1 = score1;
                            for (int j = 0; j < max_jj; j++)
                            {
                                const float v0 = pV0[0];
                                const float v1 = pV0[1];
                                out00 += *scoreptr0 * v0;
                                out01 += *scoreptr0++ * v1;
                                out10 += *scoreptr1 * v0;
                                out11 += *scoreptr1++ * v1;
                                pV0 += value_panel_width;
                            }
                            outptr0[0] = out00;
                            outptr0[1] = out01;
                            outptr1[0] = out10;
                            outptr1[1] = out11;
                            outptr0 += 2;
                            outptr1 += 2;
                        }
                        for (; lane < value_panel_width; lane++)
                        {
                            float out00 = *outptr0 * alpha0;
                            float out10 = *outptr1 * alpha1;
                            const float* pV0 = pV + lane;
                            const float* scoreptr0 = score0;
                            const float* scoreptr1 = score1;
                            for (int j = 0; j < max_jj; j++)
                            {
                                out00 += *scoreptr0++ * *pV0;
                                out10 += *scoreptr1++ * *pV0;
                                pV0 += value_panel_width;
                            }
                            *outptr0++ = out00;
                            *outptr1++ = out10;
                        }
                        pV += (size_t)value_panel_width * max_jj;
                        d += value_panel_width;
                    }
                }
            }
        }

        lptr[ii] = l0;
        lptr[ii + 1] = l1;
    }
    for (; ii < max_ii; ii++)
    {
        float m = -FLT_MAX;
        float l = 0.f;
        memset(outT_ptr + (size_t)ii * value_dim, 0, (size_t)value_dim * sizeof(float));

        for (int n = 0; n < key_seqlen; n += TILE_N)
        {
            const int max_jj = std::min(key_seqlen - n, TILE_N);
            const float* pQ = queryT_ptr + (size_t)ii * head_dim;
            float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N;
            const float* packed_maskptr = packed_mask_data ? packed_mask_data + (size_t)ii * key_seqlen + n : 0;
            float block_max = -FLT_MAX;

            // qk
            {
                const float* key_panel = (const float*)packed_key_head + (size_t)n * head_dim;
                float* score_panel = scoreptr;
                for (int jj = 0; jj < max_jj; jj += NR)
                {
                    const int max_nn = std::min(NR, max_jj - jj);
                    int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    for (; j + 15 < max_nn; j += 16)
                    {
                        __m512 _sum = packed_maskptr ? _mm512_loadu_ps(packed_maskptr) : _mm512_setzero_ps();
                        if (packed_maskptr)
                            packed_maskptr += 16;
                        const float* pK = key_panel + j;
                        for (int d = 0; d < head_dim; d++)
                        {
                            _sum = _mm512_fmadd_ps(_mm512_loadu_ps(pK), _mm512_set1_ps(pQ[d]), _sum);
                            pK += NR;
                        }
                        _mm512_storeu_ps(score_panel, _sum);
                        score_panel += 16;
                        block_max = std::max(block_max, _mm512_comp_reduce_max_ps(_sum));
                    }
#endif // __AVX512F__
                    for (; j + 7 < max_nn; j += 8)
                    {
                        __m256 _sum = packed_maskptr ? _mm256_loadu_ps(packed_maskptr) : _mm256_setzero_ps();
                        if (packed_maskptr)
                            packed_maskptr += 8;
                        const float* pK = key_panel + j;
                        for (int d = 0; d < head_dim; d++)
                        {
                            _sum = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pK), _mm256_set1_ps(pQ[d]), _sum);
                            pK += NR;
                        }
                        _mm256_storeu_ps(score_panel, _sum);
                        score_panel += 8;
                        block_max = std::max(block_max, _mm256_reduce_max_ps(_sum));
                    }
#endif // __AVX__
                    for (; j + 3 < max_nn; j += 4)
                    {
                        __m128 _sum = packed_maskptr ? _mm_loadu_ps(packed_maskptr) : _mm_setzero_ps();
                        if (packed_maskptr)
                            packed_maskptr += 4;
                        const float* pK = key_panel + j;
                        for (int d = 0; d < head_dim; d++)
                        {
                            _sum = _mm_comp_fmadd_ps(_mm_loadu_ps(pK), _mm_set1_ps(pQ[d]), _sum);
                            pK += NR;
                        }
                        _mm_storeu_ps(score_panel, _sum);
                        score_panel += 4;
                        block_max = std::max(block_max, _mm_reduce_max_ps(_sum));
                    }
#endif // __SSE2__
                    for (; j + 1 < max_nn; j += 2)
                    {
                        float sum0 = packed_maskptr ? packed_maskptr[0] : 0.f;
                        float sum1 = packed_maskptr ? packed_maskptr[1] : 0.f;
                        if (packed_maskptr)
                            packed_maskptr += 2;
                        const float* pK = key_panel + j;
                        for (int d = 0; d < head_dim; d++)
                        {
                            const float qv = pQ[d];
                            sum0 += qv * pK[0];
                            sum1 += qv * pK[1];
                            pK += NR;
                        }
                        score_panel[0] = sum0;
                        score_panel[1] = sum1;
                        score_panel += 2;
                        block_max = std::max(block_max, std::max(sum0, sum1));
                    }
                    for (; j < max_nn; j++)
                    {
                        float sum = packed_maskptr ? *packed_maskptr++ : 0.f;
                        const float* pK = key_panel + j;
                        for (int d = 0; d < head_dim; d++)
                        {
                            sum += pQ[d] * *pK;
                            pK += NR;
                        }
                        *score_panel++ = sum;
                        block_max = std::max(block_max, sum);
                    }

                    key_panel += (size_t)head_dim * NR;
                }
            }

            float alpha;

            // online softmax
            {
                const float m_new = std::max(m, block_max);
                alpha = l == 0.f ? 0.f : expf(m - m_new);
                float sum = 0.f;
                float* pS = scoreptr;
                int j = 0;
                for (; j + 1 < max_jj; j += 2)
                {
                    pS[0] = expf(pS[0] - m_new);
                    pS[1] = expf(pS[1] - m_new);
                    sum += pS[0] + pS[1];
                    pS += 2;
                }
                for (; j < max_jj; j++)
                {
                    *pS = expf(*pS - m_new);
                    sum += *pS++;
                }
                m = m_new;
                l = l * alpha + sum;
            }

            // pv
            {
                float* outptr = outT_ptr + (size_t)ii * value_dim;
                if (computation_value_head.empty())
                {
                    for (int d = 0; d < value_dim;)
                    {
                        const int value_panel_width = sdpa_value_panel_width(value_dim - d);
                        int lane = 0;
                        for (; lane + 1 < value_panel_width; lane += 2)
                        {
                            float out0 = outptr[0] * alpha;
                            float out1 = outptr[1] * alpha;
                            const float* value_panel = (const float*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreptr + jj;
                                const float* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    out0 += *pS * pV[0];
                                    out1 += *pS++ * pV[1];
                                    pV += value_panel_width;
                                }
                                value_panel += (size_t)NR * value_dim;
                            }
                            outptr[0] = out0;
                            outptr[1] = out1;
                            outptr += 2;
                        }
                        for (; lane < value_panel_width; lane++)
                        {
                            float out = *outptr * alpha;
                            const float* value_panel = (const float*)packed_value_head + (size_t)n * value_dim + (size_t)d * NR + lane;
                            for (int jj = 0; jj < max_jj; jj += NR)
                            {
                                const int max_nn = std::min(NR, max_jj - jj);
                                const float* pS = scoreptr + jj;
                                const float* pV = value_panel;
                                for (int j = 0; j < max_nn; j++)
                                {
                                    out += *pS++ * *pV;
                                    pV += value_panel_width;
                                }
                                value_panel += (size_t)NR * value_dim;
                            }
                            *outptr++ = out;
                        }
                        d += value_panel_width;
                    }
                }
                else
                {
                    const float* computation_value_tile = computation_value_head.row(n / TILE_N);
                    const float* pV = computation_value_tile;
                    for (int d = 0; d < value_dim;)
                    {
                        const int value_panel_width = sdpa_value_panel_width(value_dim - d);
                        int lane = 0;
                        for (; lane + 1 < value_panel_width; lane += 2)
                        {
                            float out0 = outptr[0] * alpha;
                            float out1 = outptr[1] * alpha;
                            const float* pS = scoreptr;
                            const float* pV0 = pV + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                out0 += *pS * pV0[0];
                                out1 += *pS++ * pV0[1];
                                pV0 += value_panel_width;
                            }
                            outptr[0] = out0;
                            outptr[1] = out1;
                            outptr += 2;
                        }
                        for (; lane < value_panel_width; lane++)
                        {
                            float out = *outptr * alpha;
                            const float* pS = scoreptr;
                            const float* pV0 = pV + lane;
                            for (int j = 0; j < max_jj; j++)
                            {
                                out += *pS++ * *pV0;
                                pV0 += value_panel_width;
                            }
                            *outptr++ = out;
                        }
                        pV += (size_t)value_panel_width * max_jj;
                        d += value_panel_width;
                    }
                }
            }
        }

        lptr[ii] = l;
    }
}

static int sdpa_prefill_packed(const Mat& query, const Mat& packed_key, const Mat& packed_value, const Mat& attn_mask, Mat& top_blob, float scale, const Option& opt)
{
    const int head_dim = query.w;
    const int value_dim = packed_value.w;
    const int query_seqlen = query.h;
    const int key_seqlen = packed_key.h;
    const int num_query_heads = query.c;
    const int num_kv_heads = packed_key.c;
    const int num_query_heads_per_kv_head = num_query_heads / num_kv_heads;
    const int nT = std::max(opt.num_threads, 1);
    const int TILE_M = sdpa_prefill_get_optimal_tile_m();
#if __AVX512F__
    const int NR = 16;
#elif __AVX__
    const int NR = 8;
#elif __SSE2__
    const int NR = 4;
#else
    const int NR = 2;
#endif

    const int num_mblocks = (query_seqlen + TILE_M - 1) / TILE_M;
    const int num_tasks = num_query_heads * num_mblocks;
    int TILE_N = sdpa_prefill_get_optimal_tile_n(head_dim, value_dim, key_seqlen, 4, 4, 4, attn_mask.empty() ? 0 : 4, TILE_M);
    TILE_N = std::max(NR, (TILE_N + NR - 1) / NR * NR);
    const int num_key_blocks = (key_seqlen + TILE_N - 1) / TILE_N;

    const int key_reuse = num_mblocks * num_query_heads_per_kv_head;
    int value_pack_reuse = 12;
#if __SSE2__
#if __AVX__
    value_pack_reuse = 8;
#if __AVX512F__
    value_pack_reuse = 6;
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
    if (value_dim >= 128)
        value_pack_reuse -= 2;
    if (value_dim < 32)
        value_pack_reuse += 4;

    const bool use_computation_value = key_reuse >= value_pack_reuse;
    Mat computation_value;
    if (use_computation_value)
    {
        computation_value.create(value_dim * TILE_N, num_key_blocks, num_kv_heads, 4u, opt.workspace_allocator);
        if (computation_value.empty())
            return -100;

        sdpa_pack_computation_value(packed_value, computation_value, TILE_N, opt);
    }

    const int query_workspace_size = TILE_M * head_dim;
    const int score_workspace_size = TILE_M * TILE_N;
    const int out_workspace_size = TILE_M * value_dim;
    const int l_workspace_size = TILE_M;
    const int workspace_size = query_workspace_size + score_workspace_size + out_workspace_size + l_workspace_size;

    Mat workspace(workspace_size, 1, nT, 4u, opt.workspace_allocator);
    if (workspace.empty())
        return -100;

    Mat packed_mask;
    if (!attn_mask.empty())
    {
        const int num_mask_heads = attn_mask.dims == 3 ? attn_mask.c : 1;
        packed_mask.create(key_seqlen * TILE_M, num_mblocks, num_mask_heads, 4u, opt.workspace_allocator);
        if (packed_mask.empty())
            return -100;

        sdpa_pack_mask(attn_mask, packed_mask, TILE_M, opt);
    }

    #pragma omp parallel for num_threads(nT)
    for (int task_id = 0; task_id < num_tasks; task_id++)
    {
        const int q = task_id / num_mblocks;
        const int i0 = task_id % num_mblocks * TILE_M;
        const int max_ii = std::min(query_seqlen - i0, TILE_M);
        const int g = q / num_query_heads_per_kv_head;

        Mat workspace_tile = workspace.channel(get_omp_thread_num());
        Mat queryT = workspace_tile.range(0, query_workspace_size);
        Mat scoreT = workspace_tile.range(query_workspace_size, score_workspace_size);
        Mat outT = workspace_tile.range(query_workspace_size + score_workspace_size, out_workspace_size);
        Mat lT = workspace_tile.range(query_workspace_size + score_workspace_size + out_workspace_size, l_workspace_size);

        const Mat query_head = query.channel(q);
        const Mat packed_key_head = packed_key.channel(g);
        const Mat packed_value_head = packed_value.channel(g);
        Mat packed_mask_tile;
        if (!packed_mask.empty())
        {
            Mat packed_mask_head = packed_mask.channel(packed_mask.c > 1 ? q : 0);
            packed_mask_tile = packed_mask_head.row_range(task_id % num_mblocks, 1);
        }

        sdpa_pack_query(query_head, queryT, i0, max_ii, query_head.w * query_head.elempack, scale);

        const Mat computation_value_head = use_computation_value ? computation_value.channel(g) : Mat();
        sdpa_attention_tile(queryT, Mat(), packed_key_head, Mat(), packed_value_head, computation_value_head, Mat(), 0, packed_mask_tile, scoreT, outT, lT, max_ii);

        Mat top_blob_head = top_blob.channel(q);
        sdpa_store_output_tile(outT, lT, top_blob_head, i0, max_ii, top_blob_head.w * top_blob_head.elempack);
    }

    return 0;
}

static int sdpa_prefill(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask, Mat& top_blob, float scale, const Option& opt)
{
#if __AVX512F__
    const int panel_width = 16;
#elif __AVX__
    const int panel_width = 8;
#elif __SSE2__
    const int panel_width = 4;
#else
    const int panel_width = 2;
#endif
    const int capacity = (key.h + panel_width - 1) / panel_width * panel_width;

    Mat packed_key(key.w, capacity, key.c, 4u, 1, opt.workspace_allocator);
    if (packed_key.empty())
        return -100;

    Mat packed_value(value.w, capacity, value.c, 4u, 1, opt.workspace_allocator);
    if (packed_value.empty())
        return -100;

    packed_key.h = key.h;
    packed_value.h = value.h;

    const int num_kv_heads = key.c;
    const int num_panels = (key.h + panel_width - 1) / panel_width;
    const int num_panel_tasks = std::min(num_panels, std::max(1, (opt.num_threads + num_kv_heads - 1) / num_kv_heads));
    const int num_tasks = num_kv_heads * num_panel_tasks;
    const int nT = key.h >= panel_width ? std::min(opt.num_threads, num_tasks) : 1;

    #pragma omp parallel for num_threads(nT)
    for (int task_id = 0; task_id < num_tasks; task_id++)
    {
        const int g = task_id / num_panel_tasks;
        const int panel_task_id = task_id % num_panel_tasks;
        const int panel_begin_id = panel_task_id * num_panels / num_panel_tasks;
        const int panel_end_id = (panel_task_id + 1) * num_panels / num_panel_tasks;
        const Mat key_head = key.channel(g);
        const Mat value_head = value.channel(g);
        Mat packed_key_head = packed_key.channel(g);
        Mat packed_value_head = packed_value.channel(g);

        for (int panel_id = panel_begin_id; panel_id < panel_end_id; panel_id++)
        {
            const int n_begin = panel_id * panel_width;
            const int n_end = std::min(key.h, n_begin + panel_width);
            Mat packed_key_tile(key.w * panel_width, (float*)packed_key_head + (size_t)panel_id * key.w * panel_width, 4u);
            Mat packed_value_tile(value.w * panel_width, (float*)packed_value_head + (size_t)panel_id * value.w * panel_width, 4u);

            sdpa_pack_key_tile(key_head, packed_key_tile, n_begin, 0, n_end - n_begin);
            sdpa_pack_value_tile(value_head, packed_value_tile, n_begin, 0, n_end - n_begin);
        }
    }

    return sdpa_prefill_packed(query, packed_key, packed_value, attn_mask, top_blob, scale, opt);
}

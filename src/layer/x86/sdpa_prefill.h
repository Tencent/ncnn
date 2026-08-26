// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static int sdpa_prefill_get_optimal_tile_m(int query_seqlen, int num_query_heads, int nT)
{
    int TILE_M = 1;
#if __SSE2__
    TILE_M = 4;
#if __AVX__
    TILE_M = 8;
#if __AVX512F__
    TILE_M = 16;
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

    while (TILE_M > 4)
    {
        const int num_tasks = num_query_heads * ((query_seqlen + TILE_M - 1) / TILE_M);
        if (num_tasks >= nT)
            break;

        TILE_M /= 2;
    }

    return TILE_M;
}

static int sdpa_prefill_get_optimal_tile_n(int head_dim, int value_dim, int key_seqlen, int query_storage_size, int key_storage_size, int value_storage_size, int mask_storage_size, int TILE_M, int num_tasks, int nT)
{
    int tile_n_align = 1;
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

static Mat sdpa_prefill_get_mask_head(const Mat& attn_mask_blob, int q)
{
    if (attn_mask_blob.empty())
        return Mat();

    if (attn_mask_blob.dims == 3)
        return attn_mask_blob.channel(attn_mask_blob.c > 1 ? q : 0);

    return attn_mask_blob;
}

// queryT[head_dim][query_lane]
static void sdpa_pack_query_fp32(const Mat& query_head, Mat& queryT, int i, int max_ii, float scale)
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
        const float* qptr0 = query_head.row(i0);
        const float* qptr1 = query_head.row(i0 + 1);
        const float* qptr2 = query_head.row(i0 + 2);
        const float* qptr3 = query_head.row(i0 + 3);
        const float* qptr4 = query_head.row(i0 + 4);
        const float* qptr5 = query_head.row(i0 + 5);
        const float* qptr6 = query_head.row(i0 + 6);
        const float* qptr7 = query_head.row(i0 + 7);
        const float* qptr8 = query_head.row(i0 + 8);
        const float* qptr9 = query_head.row(i0 + 9);
        const float* qptra = query_head.row(i0 + 10);
        const float* qptrb = query_head.row(i0 + 11);
        const float* qptrc = query_head.row(i0 + 12);
        const float* qptrd = query_head.row(i0 + 13);
        const float* qptre = query_head.row(i0 + 14);
        const float* qptrf = query_head.row(i0 + 15);

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
    for (; ii + 7 < max_ii; ii += 8)
    {
        const int i0 = i + ii;
        float* pQ = queryT_ptr + (size_t)ii * head_dim;
        const float* qptr0 = query_head.row(i0);
        const float* qptr1 = query_head.row(i0 + 1);
        const float* qptr2 = query_head.row(i0 + 2);
        const float* qptr3 = query_head.row(i0 + 3);
        const float* qptr4 = query_head.row(i0 + 4);
        const float* qptr5 = query_head.row(i0 + 5);
        const float* qptr6 = query_head.row(i0 + 6);
        const float* qptr7 = query_head.row(i0 + 7);

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
    for (; ii + 3 < max_ii; ii += 4)
    {
        const int i0 = i + ii;
        float* pQ = queryT_ptr + (size_t)ii * head_dim;
        const float* qptr0 = query_head.row(i0);
        const float* qptr1 = query_head.row(i0 + 1);
        const float* qptr2 = query_head.row(i0 + 2);
        const float* qptr3 = query_head.row(i0 + 3);

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
#endif // __SSE2__
    for (; ii < max_ii; ii++)
    {
        const float* qptr = query_head.row(i + ii);
        float* pQ = queryT_ptr + (size_t)ii * head_dim;
        for (int d = 0; d < head_dim; d++)
            pQ[d] = qptr[d] * scale;
    }
}

// packed_mask[mask_head][query_block][query_panel][key][query_lane] in fp32
static void sdpa_pack_mask_fp32(const Mat& attn_mask_blob, Mat& packed_mask, int block_m, const Option& opt)
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
        Mat maskT = packed_mask_head.row_range(mblock_id, 1);
        sdpa_pack_query_fp32(mask_head, maskT, i0, max_ii, 1.f);
    }
}

static void sdpa_store_output_tile(const Mat& outT, const Mat& lT, Mat& top_blob_head, int i0, int max_ii)
{
    const int block_m = lT.w;
    const int value_dim = outT.w / block_m;
    const float* pp = outT;
    const float* lptr = lT;

    int ii = 0;
#if __SSE2__
    const size_t out_hstep = top_blob_head.w;
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const __m512 _l = _mm512_loadu_ps(lptr);
        lptr += 16;
        float* p0 = top_blob_head.row(i0 + ii);
        const __mmask16 nonzero = _mm512_cmp_ps_mask(_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
        const __m512 _scale = _mm512_maskz_div_ps(nonzero, _mm512_set1_ps(1.f), _l);
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
            const __m512 _r = _mm512_mul_ps(_mm512_loadu_ps(pp), _scale);
            const __m128 _r0 = _mm512_extractf32x4_ps(_r, 0);
            const __m128 _r1 = _mm512_extractf32x4_ps(_r, 1);
            const __m128 _r2 = _mm512_extractf32x4_ps(_r, 2);
            const __m128 _r3 = _mm512_extractf32x4_ps(_r, 3);
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
        const __m256 _l = _mm256_loadu_ps(lptr);
        lptr += 8;
        float* p0 = top_blob_head.row(i0 + ii);
        const __m256 _nonzero = _mm256_cmp_ps(_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
        const __m256 _denom = _mm256_blendv_ps(_mm256_set1_ps(1.f), _l, _nonzero);
        const __m256 _scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(1.f), _denom), _nonzero);

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
            const __m256 _r = _mm256_mul_ps(_mm256_loadu_ps(pp), _scale);
            const __m128 _r0 = _mm256_castps256_ps128(_r);
            const __m128 _r1 = _mm256_extractf128_ps(_r, 1);
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
        const __m128 _l = _mm_loadu_ps(lptr);
        lptr += 4;
        float* p0 = top_blob_head.row(i0 + ii);
        const __m128 _nonzero = _mm_cmpneq_ps(_l, _mm_setzero_ps());
        const __m128 _denom = _mm_or_ps(_mm_and_ps(_nonzero, _l), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
        const __m128 _scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(1.f), _denom), _nonzero);

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
            const __m128 _r = _mm_mul_ps(_mm_loadu_ps(pp), _scale);
            p0[0] = _mm_cvtss_f32(_r);
            p0[out_hstep] = _mm_cvtss_f32(_mm_shuffle_ps(_r, _r, _MM_SHUFFLE(1, 1, 1, 1)));
            p0[out_hstep * 2] = _mm_cvtss_f32(_mm_movehl_ps(_r, _r));
            p0[out_hstep * 3] = _mm_cvtss_f32(_mm_shuffle_ps(_r, _r, _MM_SHUFFLE(3, 3, 3, 3)));
            p0++;
            pp += 4;
        }
    }
#endif // __SSE2__
    for (; ii < max_ii; ii++)
    {
        float* p0 = top_blob_head.row(i0 + ii);
        const float inv_sum = *lptr == 0.f ? 0.f : 1.f / *lptr;
        lptr++;
        for (int d = 0; d < value_dim; d++)
            *p0++ = *pp++ * inv_sum;
    }
}

static void sdpa_prefill_reduce(const Mat& partials, Mat& top_blob, Mat& workspace, int num_tasks, int num_mblocks, int block_m, int num_kv_chunks, int query_seqlen, int value_dim, const Option& opt)
{
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int task_id = 0; task_id < num_tasks; task_id++)
    {
        const int q = task_id / num_mblocks;
        const int mblock_id = task_id % num_mblocks;
        const int i0 = mblock_id * block_m;
        const int max_ii = std::min(query_seqlen - i0, block_m);
        Mat top_blob_head = top_blob.channel(q);
        Mat workspace_tile = workspace.channel(get_omp_thread_num());
        Mat outT_tile = workspace_tile.range(0, value_dim * block_m);
        Mat lT_tile = workspace_tile.range(value_dim * block_m, block_m);
        float* outT = outT_tile;
        float* lT = lT_tile;

        int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; ii + 15 < max_ii; ii += 16)
        {
            __m512 _m = _mm512_set1_ps(-FLT_MAX);
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state = partials.channel(task_id * num_kv_chunks + chunk_id);
                _m = _mm512_max_ps(_m, _mm512_loadu_ps(state + ii));
            }

            memset(outT + (size_t)ii * value_dim, 0, (size_t)value_dim * 16 * sizeof(float));
            __m512 _l = _mm512_setzero_ps();
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state = partials.channel(task_id * num_kv_chunks + chunk_id);
                const __m512 _partial_l = _mm512_loadu_ps(state + block_m + ii);
                const __mmask16 active = _mm512_cmp_ps_mask(_partial_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                const __m512 _partial_scale = _mm512_maskz_mov_ps(active, exp512_ps(_mm512_maskz_sub_ps(active, _mm512_loadu_ps(state + ii), _m)));
                _l = _mm512_fmadd_ps(_partial_l, _partial_scale, _l);
                float* outptr = outT + (size_t)ii * value_dim;
                const float* stateptr = state + 2 * block_m + ii;
                for (int d = 0; d < value_dim; d++)
                {
                    __m512 _out = _mm512_loadu_ps(outptr);
                    _out = _mm512_fmadd_ps(_mm512_loadu_ps(stateptr), _partial_scale, _out);
                    _mm512_storeu_ps(outptr, _out);
                    outptr += 16;
                    stateptr += block_m;
                }
            }
            _mm512_storeu_ps(lT + ii, _l);
        }
#endif // __AVX512F__
        for (; ii + 7 < max_ii; ii += 8)
        {
            __m256 _m = _mm256_set1_ps(-FLT_MAX);
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state = partials.channel(task_id * num_kv_chunks + chunk_id);
                _m = _mm256_max_ps(_m, _mm256_loadu_ps(state + ii));
            }

            memset(outT + (size_t)ii * value_dim, 0, (size_t)value_dim * 8 * sizeof(float));
            __m256 _l = _mm256_setzero_ps();
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state = partials.channel(task_id * num_kv_chunks + chunk_id);
                const __m256 _partial_l = _mm256_loadu_ps(state + block_m + ii);
                const __m256 _active = _mm256_cmp_ps(_partial_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
                const __m256 _partial_scale = _mm256_and_ps(_active, exp256_ps(_mm256_and_ps(_active, _mm256_sub_ps(_mm256_loadu_ps(state + ii), _m))));
                _l = _mm256_comp_fmadd_ps(_partial_l, _partial_scale, _l);
                float* outptr = outT + (size_t)ii * value_dim;
                const float* stateptr = state + 2 * block_m + ii;
                for (int d = 0; d < value_dim; d++)
                {
                    __m256 _out = _mm256_loadu_ps(outptr);
                    _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(stateptr), _partial_scale, _out);
                    _mm256_storeu_ps(outptr, _out);
                    outptr += 8;
                    stateptr += block_m;
                }
            }
            _mm256_storeu_ps(lT + ii, _l);
        }
#endif // __AVX__
        for (; ii + 3 < max_ii; ii += 4)
        {
            __m128 _m = _mm_set1_ps(-FLT_MAX);
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state = partials.channel(task_id * num_kv_chunks + chunk_id);
                _m = _mm_max_ps(_m, _mm_loadu_ps(state + ii));
            }

            memset(outT + (size_t)ii * value_dim, 0, (size_t)value_dim * 4 * sizeof(float));
            __m128 _l = _mm_setzero_ps();
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state = partials.channel(task_id * num_kv_chunks + chunk_id);
                const __m128 _partial_l = _mm_loadu_ps(state + block_m + ii);
                const __m128 _active = _mm_cmpneq_ps(_partial_l, _mm_setzero_ps());
                const __m128 _partial_scale = _mm_and_ps(_active, exp_ps(_mm_and_ps(_active, _mm_sub_ps(_mm_loadu_ps(state + ii), _m))));
                _l = _mm_comp_fmadd_ps(_partial_l, _partial_scale, _l);
                float* outptr = outT + (size_t)ii * value_dim;
                const float* stateptr = state + 2 * block_m + ii;
                for (int d = 0; d < value_dim; d++)
                {
                    __m128 _out = _mm_loadu_ps(outptr);
                    _out = _mm_comp_fmadd_ps(_mm_loadu_ps(stateptr), _partial_scale, _out);
                    _mm_storeu_ps(outptr, _out);
                    outptr += 4;
                    stateptr += block_m;
                }
            }
            _mm_storeu_ps(lT + ii, _l);
        }
#endif // __SSE2__
        for (; ii < max_ii; ii++)
        {
            float m = -FLT_MAX;
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state = partials.channel(task_id * num_kv_chunks + chunk_id);
                m = std::max(m, state[ii]);
            }

            float* outptr = outT + (size_t)ii * value_dim;
            memset(outptr, 0, value_dim * sizeof(float));
            float l = 0.f;
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state = partials.channel(task_id * num_kv_chunks + chunk_id);
                const float partial_l = state[block_m + ii];
                const float partial_scale = partial_l == 0.f ? 0.f : expf(state[ii] - m);
                l += partial_l * partial_scale;
                for (int d = 0; d < value_dim; d++)
                    outptr[d] += state[(d + 2) * block_m + ii] * partial_scale;
            }
            lT[ii] = l;
        }

        sdpa_store_output_tile(outT_tile, lT_tile, top_blob_head, i0, max_ii);
    }
}

// packed_key[token_panel][head_dim][token_lane] in fp32
static void sdpa_pack_key_tile_fp32(const Mat& key, Mat& packed_key, int src_begin, int dst_begin, int max_seqlen)
{
    const int head_dim = key.w;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    const int panel_width = 16;
#else
    const int panel_width = 8;
#endif // __AVX512F__
#else
    const int panel_width = 4;
#endif // __AVX__
#else
    const int panel_width = 1;
#endif // __SSE2__
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
        const float* p1 = p0 + head_dim;
        const float* p2 = p1 + head_dim;
        const float* p3 = p2 + head_dim;
        const float* p4 = p3 + head_dim;
        const float* p5 = p4 + head_dim;
        const float* p6 = p5 + head_dim;
        const float* p7 = p6 + head_dim;
        const float* p8 = p7 + head_dim;
        const float* p9 = p8 + head_dim;
        const float* pa = p9 + head_dim;
        const float* pb = pa + head_dim;
        const float* pc = pb + head_dim;
        const float* pd = pc + head_dim;
        const float* pe = pd + head_dim;
        const float* pf = pe + head_dim;

        float* pp = panel + token_lane + j;
        int d = 0;
        for (; d + 15 < head_dim; d += 16)
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
            pp += panel_width * 16;
        }
        for (; d < head_dim; d++)
        {
            pp[0] = *p0++;
            pp[1] = *p1++;
            pp[2] = *p2++;
            pp[3] = *p3++;
            pp[4] = *p4++;
            pp[5] = *p5++;
            pp[6] = *p6++;
            pp[7] = *p7++;
            pp[8] = *p8++;
            pp[9] = *p9++;
            pp[10] = *pa++;
            pp[11] = *pb++;
            pp[12] = *pc++;
            pp[13] = *pd++;
            pp[14] = *pe++;
            pp[15] = *pf++;
            pp += panel_width;
        }
    }
#endif // __AVX512F__
    for (; j + 7 < max_seqlen; j += 8)
    {
        const float* p0 = key.row(src_begin + j);
        const float* p1 = p0 + head_dim;
        const float* p2 = p1 + head_dim;
        const float* p3 = p2 + head_dim;
        const float* p4 = p3 + head_dim;
        const float* p5 = p4 + head_dim;
        const float* p6 = p5 + head_dim;
        const float* p7 = p6 + head_dim;

        float* pp = panel + token_lane + j;
        int d = 0;
        for (; d + 7 < head_dim; d += 8)
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
            _mm256_storeu_ps(pp, _r0);
            _mm256_storeu_ps(pp + panel_width, _r1);
            _mm256_storeu_ps(pp + panel_width * 2, _r2);
            _mm256_storeu_ps(pp + panel_width * 3, _r3);
            _mm256_storeu_ps(pp + panel_width * 4, _r4);
            _mm256_storeu_ps(pp + panel_width * 5, _r5);
            _mm256_storeu_ps(pp + panel_width * 6, _r6);
            _mm256_storeu_ps(pp + panel_width * 7, _r7);

            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
            p4 += 8;
            p5 += 8;
            p6 += 8;
            p7 += 8;
            pp += panel_width * 8;
        }
        for (; d < head_dim; d++)
        {
            pp[0] = *p0++;
            pp[1] = *p1++;
            pp[2] = *p2++;
            pp[3] = *p3++;
            pp[4] = *p4++;
            pp[5] = *p5++;
            pp[6] = *p6++;
            pp[7] = *p7++;
            pp += panel_width;
        }
    }
#endif // __AVX__
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; j + 3 < max_seqlen; j += 4)
    {
        const float* p0 = key.row(src_begin + j);
        const float* p1 = p0 + head_dim;
        const float* p2 = p1 + head_dim;
        const float* p3 = p2 + head_dim;

        float* pp = panel + token_lane + j;
        int d = 0;
        for (; d + 3 < head_dim; d += 4)
        {
            __m128 _r0 = _mm_loadu_ps(p0);
            __m128 _r1 = _mm_loadu_ps(p1);
            __m128 _r2 = _mm_loadu_ps(p2);
            __m128 _r3 = _mm_loadu_ps(p3);
            _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
            _mm_storeu_ps(pp, _r0);
            _mm_storeu_ps(pp + panel_width, _r1);
            _mm_storeu_ps(pp + panel_width * 2, _r2);
            _mm_storeu_ps(pp + panel_width * 3, _r3);

            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
            pp += panel_width * 4;
        }
        for (; d < head_dim; d++)
        {
            pp[0] = *p0++;
            pp[1] = *p1++;
            pp[2] = *p2++;
            pp[3] = *p3++;
            pp += panel_width;
        }
    }
#endif // __SSE2__
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
static void sdpa_pack_value_tile_fp32(const Mat& value, Mat& packed_value, int src_begin, int dst_begin, int max_seqlen)
{
    const int value_dim = value.w;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    const int panel_width = 16;
#else
    const int panel_width = 8;
#endif // __AVX512F__
#else
    const int panel_width = 4;
#endif // __AVX__
#else
    const int panel_width = 1;
#endif // __SSE2__
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

static void sdpa_prefill_packed_tile_fp32(const Mat& queryT, const Mat& packed_key_head, const Mat& packed_value_head, const Mat& maskT, Mat& scoreT, Mat& outT, Mat& stateT, int max_ii, int n_begin, int n_end)
{
    const int head_dim = packed_key_head.w;
    const int value_dim = packed_value_head.w;
    const int key_seqlen = packed_key_head.h;
    const int TILE_M = stateT.w / 2;
    const int TILE_N = scoreT.w / TILE_M;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    const int NR = 16;
#else
    const int NR = 8;
#endif // __AVX512F__
#else
    const int NR = 4;
#endif // __AVX__
#else
    const int NR = 1;
#endif // __SSE2__

    const float* queryT_ptr = queryT;
    float* scoreT_ptr = scoreT;
    float* outT_ptr = outT;
    float* mT = stateT;
    float* lT = mT + TILE_M;
    const float* maskT_ptr = maskT;

    for (int i = 0; i < max_ii; i++)
    {
        mT[i] = -FLT_MAX;
        lT[i] = 0.f;
    }
    memset(outT_ptr, 0, (size_t)max_ii * value_dim * sizeof(float));

    for (int n = n_begin; n < n_end; n += TILE_N)
    {
        const int max_jj = std::min(n_end - n, TILE_N);

        int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; ii + 15 < max_ii; ii += 16)
        {
            const float* pQ = queryT_ptr + (size_t)ii * head_dim;
            float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N;
            const float* pM = maskT_ptr ? maskT_ptr + (size_t)ii * key_seqlen : 0;
            __m512 _block_max = _mm512_set1_ps(-FLT_MAX);

            for (int jj = 0; jj < max_jj; jj += NR)
            {
                const int max_nn = std::min(NR, max_jj - jj);
                const float* key_panel = (const float*)packed_key_head + (size_t)(n + jj) * head_dim;
                int j = 0;
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
                    if (pM)
                    {
                        _sum0 = _mm512_add_ps(_sum0, _mm512_loadu_ps(pM + (size_t)(n + jj + j) * 16));
                        _sum1 = _mm512_add_ps(_sum1, _mm512_loadu_ps(pM + (size_t)(n + jj + j + 1) * 16));
                        _sum2 = _mm512_add_ps(_sum2, _mm512_loadu_ps(pM + (size_t)(n + jj + j + 2) * 16));
                        _sum3 = _mm512_add_ps(_sum3, _mm512_loadu_ps(pM + (size_t)(n + jj + j + 3) * 16));
                        _sum4 = _mm512_add_ps(_sum4, _mm512_loadu_ps(pM + (size_t)(n + jj + j + 4) * 16));
                        _sum5 = _mm512_add_ps(_sum5, _mm512_loadu_ps(pM + (size_t)(n + jj + j + 5) * 16));
                        _sum6 = _mm512_add_ps(_sum6, _mm512_loadu_ps(pM + (size_t)(n + jj + j + 6) * 16));
                        _sum7 = _mm512_add_ps(_sum7, _mm512_loadu_ps(pM + (size_t)(n + jj + j + 7) * 16));
                        _sum8 = _mm512_add_ps(_sum8, _mm512_loadu_ps(pM + (size_t)(n + jj + j + 8) * 16));
                        _sum9 = _mm512_add_ps(_sum9, _mm512_loadu_ps(pM + (size_t)(n + jj + j + 9) * 16));
                        _suma = _mm512_add_ps(_suma, _mm512_loadu_ps(pM + (size_t)(n + jj + j + 10) * 16));
                        _sumb = _mm512_add_ps(_sumb, _mm512_loadu_ps(pM + (size_t)(n + jj + j + 11) * 16));
                        _sumc = _mm512_add_ps(_sumc, _mm512_loadu_ps(pM + (size_t)(n + jj + j + 12) * 16));
                        _sumd = _mm512_add_ps(_sumd, _mm512_loadu_ps(pM + (size_t)(n + jj + j + 13) * 16));
                        _sume = _mm512_add_ps(_sume, _mm512_loadu_ps(pM + (size_t)(n + jj + j + 14) * 16));
                        _sumf = _mm512_add_ps(_sumf, _mm512_loadu_ps(pM + (size_t)(n + jj + j + 15) * 16));
                    }
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j) * 16, _sum0);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j + 1) * 16, _sum1);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j + 2) * 16, _sum2);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j + 3) * 16, _sum3);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j + 4) * 16, _sum4);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j + 5) * 16, _sum5);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j + 6) * 16, _sum6);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j + 7) * 16, _sum7);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j + 8) * 16, _sum8);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j + 9) * 16, _sum9);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j + 10) * 16, _suma);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j + 11) * 16, _sumb);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j + 12) * 16, _sumc);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j + 13) * 16, _sumd);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j + 14) * 16, _sume);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j + 15) * 16, _sumf);
                    _block_max = _mm512_max_ps(_block_max, _mm512_max_ps(_mm512_max_ps(_mm512_max_ps(_sum0, _sum1), _mm512_max_ps(_sum2, _sum3)), _mm512_max_ps(_mm512_max_ps(_sum4, _sum5), _mm512_max_ps(_sum6, _sum7))));
                    _block_max = _mm512_max_ps(_block_max, _mm512_max_ps(_mm512_max_ps(_mm512_max_ps(_sum8, _sum9), _mm512_max_ps(_suma, _sumb)), _mm512_max_ps(_mm512_max_ps(_sumc, _sumd), _mm512_max_ps(_sume, _sumf))));
                }
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
                        const __m512 _q = _mm512_loadu_ps(pA);
                        _sum0 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[3]), _sum3);
                        pA += 16;
                        pK += NR;
                    }
                    if (pM)
                    {
                        _sum0 = _mm512_add_ps(_sum0, _mm512_loadu_ps(pM + (size_t)(n + jj + j) * 16));
                        _sum1 = _mm512_add_ps(_sum1, _mm512_loadu_ps(pM + (size_t)(n + jj + j + 1) * 16));
                        _sum2 = _mm512_add_ps(_sum2, _mm512_loadu_ps(pM + (size_t)(n + jj + j + 2) * 16));
                        _sum3 = _mm512_add_ps(_sum3, _mm512_loadu_ps(pM + (size_t)(n + jj + j + 3) * 16));
                    }
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j) * 16, _sum0);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j + 1) * 16, _sum1);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j + 2) * 16, _sum2);
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j + 3) * 16, _sum3);
                    _block_max = _mm512_max_ps(_block_max, _mm512_max_ps(_mm512_max_ps(_sum0, _sum1), _mm512_max_ps(_sum2, _sum3)));
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
                    if (pM)
                        _sum = _mm512_add_ps(_sum, _mm512_loadu_ps(pM + (size_t)(n + jj + j) * 16));
                    _mm512_storeu_ps(scoreptr + (size_t)(jj + j) * 16, _sum);
                    _block_max = _mm512_max_ps(_block_max, _sum);
                }
            }

            const __m512 _m = _mm512_loadu_ps(mT + ii);
            const __m512 _l = _mm512_loadu_ps(lT + ii);
            const __m512 _m_new = _mm512_max_ps(_m, _block_max);
            const __mmask16 alpha_active = _mm512_cmp_ps_mask(_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
            const __m512 _alpha = _mm512_maskz_mov_ps(alpha_active, exp512_ps(_mm512_maskz_sub_ps(alpha_active, _m, _m_new)));

            float* outptr = outT_ptr + (size_t)ii * value_dim;
            for (int d = 0; d < value_dim; d++)
                _mm512_storeu_ps(outptr + (size_t)d * 16, _mm512_mul_ps(_mm512_loadu_ps(outptr + (size_t)d * 16), _alpha));

            __m512 _sum0 = _mm512_setzero_ps();
            __m512 _sum1 = _mm512_setzero_ps();
            __m512 _sum2 = _mm512_setzero_ps();
            __m512 _sum3 = _mm512_setzero_ps();
            int j = 0;
            for (; j + 3 < max_jj; j += 4)
            {
                __m512 _p0 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(scoreptr + (size_t)j * 16), _m_new));
                __m512 _p1 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(scoreptr + (size_t)(j + 1) * 16), _m_new));
                __m512 _p2 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(scoreptr + (size_t)(j + 2) * 16), _m_new));
                __m512 _p3 = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(scoreptr + (size_t)(j + 3) * 16), _m_new));
                _mm512_storeu_ps(scoreptr + (size_t)j * 16, _p0);
                _mm512_storeu_ps(scoreptr + (size_t)(j + 1) * 16, _p1);
                _mm512_storeu_ps(scoreptr + (size_t)(j + 2) * 16, _p2);
                _mm512_storeu_ps(scoreptr + (size_t)(j + 3) * 16, _p3);
                _sum0 = _mm512_add_ps(_sum0, _p0);
                _sum1 = _mm512_add_ps(_sum1, _p1);
                _sum2 = _mm512_add_ps(_sum2, _p2);
                _sum3 = _mm512_add_ps(_sum3, _p3);
            }
            for (; j < max_jj; j++)
            {
                __m512 _p = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(scoreptr + (size_t)j * 16), _m_new));
                _mm512_storeu_ps(scoreptr + (size_t)j * 16, _p);
                _sum0 = _mm512_add_ps(_sum0, _p);
            }
            _mm512_storeu_ps(mT + ii, _m_new);
            _mm512_storeu_ps(lT + ii, _mm512_add_ps(_mm512_mul_ps(_l, _alpha), _mm512_add_ps(_mm512_add_ps(_sum0, _sum1), _mm512_add_ps(_sum2, _sum3))));
        }
#endif // __AVX512F__
        for (; ii + 7 < max_ii; ii += 8)
        {
            const float* pQ = queryT_ptr + (size_t)ii * head_dim;
            float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N;
            const float* pM = maskT_ptr ? maskT_ptr + (size_t)ii * key_seqlen : 0;
            __m256 _block_max = _mm256_set1_ps(-FLT_MAX);

            for (int jj = 0; jj < max_jj; jj += NR)
            {
                const int max_nn = std::min(NR, max_jj - jj);
                const float* key_panel = (const float*)packed_key_head + (size_t)(n + jj) * head_dim;
                int j = 0;
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
                    if (pM)
                    {
                        _sum0 = _mm256_add_ps(_sum0, _mm256_loadu_ps(pM + (size_t)(n + jj + j) * 8));
                        _sum1 = _mm256_add_ps(_sum1, _mm256_loadu_ps(pM + (size_t)(n + jj + j + 1) * 8));
                        _sum2 = _mm256_add_ps(_sum2, _mm256_loadu_ps(pM + (size_t)(n + jj + j + 2) * 8));
                        _sum3 = _mm256_add_ps(_sum3, _mm256_loadu_ps(pM + (size_t)(n + jj + j + 3) * 8));
                        _sum4 = _mm256_add_ps(_sum4, _mm256_loadu_ps(pM + (size_t)(n + jj + j + 4) * 8));
                        _sum5 = _mm256_add_ps(_sum5, _mm256_loadu_ps(pM + (size_t)(n + jj + j + 5) * 8));
                        _sum6 = _mm256_add_ps(_sum6, _mm256_loadu_ps(pM + (size_t)(n + jj + j + 6) * 8));
                        _sum7 = _mm256_add_ps(_sum7, _mm256_loadu_ps(pM + (size_t)(n + jj + j + 7) * 8));
                    }
                    _mm256_storeu_ps(scoreptr + (size_t)(jj + j) * 8, _sum0);
                    _mm256_storeu_ps(scoreptr + (size_t)(jj + j + 1) * 8, _sum1);
                    _mm256_storeu_ps(scoreptr + (size_t)(jj + j + 2) * 8, _sum2);
                    _mm256_storeu_ps(scoreptr + (size_t)(jj + j + 3) * 8, _sum3);
                    _mm256_storeu_ps(scoreptr + (size_t)(jj + j + 4) * 8, _sum4);
                    _mm256_storeu_ps(scoreptr + (size_t)(jj + j + 5) * 8, _sum5);
                    _mm256_storeu_ps(scoreptr + (size_t)(jj + j + 6) * 8, _sum6);
                    _mm256_storeu_ps(scoreptr + (size_t)(jj + j + 7) * 8, _sum7);
                    _block_max = _mm256_max_ps(_block_max, _mm256_max_ps(_mm256_max_ps(_mm256_max_ps(_sum0, _sum1), _mm256_max_ps(_sum2, _sum3)), _mm256_max_ps(_mm256_max_ps(_sum4, _sum5), _mm256_max_ps(_sum6, _sum7))));
                }
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
                        const __m256 _q = _mm256_loadu_ps(pA);
                        _sum0 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[1]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[2]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[3]), _sum3);
                        pA += 8;
                        pK += NR;
                    }
                    if (pM)
                    {
                        _sum0 = _mm256_add_ps(_sum0, _mm256_loadu_ps(pM + (size_t)(n + jj + j) * 8));
                        _sum1 = _mm256_add_ps(_sum1, _mm256_loadu_ps(pM + (size_t)(n + jj + j + 1) * 8));
                        _sum2 = _mm256_add_ps(_sum2, _mm256_loadu_ps(pM + (size_t)(n + jj + j + 2) * 8));
                        _sum3 = _mm256_add_ps(_sum3, _mm256_loadu_ps(pM + (size_t)(n + jj + j + 3) * 8));
                    }
                    _mm256_storeu_ps(scoreptr + (size_t)(jj + j) * 8, _sum0);
                    _mm256_storeu_ps(scoreptr + (size_t)(jj + j + 1) * 8, _sum1);
                    _mm256_storeu_ps(scoreptr + (size_t)(jj + j + 2) * 8, _sum2);
                    _mm256_storeu_ps(scoreptr + (size_t)(jj + j + 3) * 8, _sum3);
                    _block_max = _mm256_max_ps(_block_max, _mm256_max_ps(_mm256_max_ps(_sum0, _sum1), _mm256_max_ps(_sum2, _sum3)));
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
                    if (pM)
                        _sum = _mm256_add_ps(_sum, _mm256_loadu_ps(pM + (size_t)(n + jj + j) * 8));
                    _mm256_storeu_ps(scoreptr + (size_t)(jj + j) * 8, _sum);
                    _block_max = _mm256_max_ps(_block_max, _sum);
                }
            }

            const __m256 _m = _mm256_loadu_ps(mT + ii);
            const __m256 _l = _mm256_loadu_ps(lT + ii);
            const __m256 _m_new = _mm256_max_ps(_m, _block_max);
            const __m256 _alpha_active = _mm256_cmp_ps(_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
            const __m256 _alpha = _mm256_and_ps(_alpha_active, exp256_ps(_mm256_and_ps(_alpha_active, _mm256_sub_ps(_m, _m_new))));

            float* outptr = outT_ptr + (size_t)ii * value_dim;
            for (int d = 0; d < value_dim; d++)
                _mm256_storeu_ps(outptr + (size_t)d * 8, _mm256_mul_ps(_mm256_loadu_ps(outptr + (size_t)d * 8), _alpha));

            __m256 _sum0 = _mm256_setzero_ps();
            __m256 _sum1 = _mm256_setzero_ps();
            __m256 _sum2 = _mm256_setzero_ps();
            __m256 _sum3 = _mm256_setzero_ps();
            int j = 0;
            for (; j + 3 < max_jj; j += 4)
            {
                __m256 _p0 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr + (size_t)j * 8), _m_new));
                __m256 _p1 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr + (size_t)(j + 1) * 8), _m_new));
                __m256 _p2 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr + (size_t)(j + 2) * 8), _m_new));
                __m256 _p3 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr + (size_t)(j + 3) * 8), _m_new));
                _mm256_storeu_ps(scoreptr + (size_t)j * 8, _p0);
                _mm256_storeu_ps(scoreptr + (size_t)(j + 1) * 8, _p1);
                _mm256_storeu_ps(scoreptr + (size_t)(j + 2) * 8, _p2);
                _mm256_storeu_ps(scoreptr + (size_t)(j + 3) * 8, _p3);
                _sum0 = _mm256_add_ps(_sum0, _p0);
                _sum1 = _mm256_add_ps(_sum1, _p1);
                _sum2 = _mm256_add_ps(_sum2, _p2);
                _sum3 = _mm256_add_ps(_sum3, _p3);
            }
            for (; j < max_jj; j++)
            {
                __m256 _p = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(scoreptr + (size_t)j * 8), _m_new));
                _mm256_storeu_ps(scoreptr + (size_t)j * 8, _p);
                _sum0 = _mm256_add_ps(_sum0, _p);
            }
            _mm256_storeu_ps(mT + ii, _m_new);
            _mm256_storeu_ps(lT + ii, _mm256_add_ps(_mm256_mul_ps(_l, _alpha), _mm256_add_ps(_mm256_add_ps(_sum0, _sum1), _mm256_add_ps(_sum2, _sum3))));
        }
#endif // __AVX__
        for (; ii + 3 < max_ii; ii += 4)
        {
            const float* pQ = queryT_ptr + (size_t)ii * head_dim;
            float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N;
            const float* pM = maskT_ptr ? maskT_ptr + (size_t)ii * key_seqlen : 0;
            __m128 _block_max = _mm_set1_ps(-FLT_MAX);

            for (int jj = 0; jj < max_jj; jj += NR)
            {
                const int max_nn = std::min(NR, max_jj - jj);
                const float* key_panel = (const float*)packed_key_head + (size_t)(n + jj) * head_dim;
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
                        const __m128 _q = _mm_loadu_ps(pA);
                        _sum0 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[0]), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[1]), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[2]), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[3]), _sum3);
                        pA += 4;
                        pK += NR;
                    }
                    if (pM)
                    {
                        _sum0 = _mm_add_ps(_sum0, _mm_loadu_ps(pM + (size_t)(n + jj + j) * 4));
                        _sum1 = _mm_add_ps(_sum1, _mm_loadu_ps(pM + (size_t)(n + jj + j + 1) * 4));
                        _sum2 = _mm_add_ps(_sum2, _mm_loadu_ps(pM + (size_t)(n + jj + j + 2) * 4));
                        _sum3 = _mm_add_ps(_sum3, _mm_loadu_ps(pM + (size_t)(n + jj + j + 3) * 4));
                    }
                    _mm_storeu_ps(scoreptr + (size_t)(jj + j) * 4, _sum0);
                    _mm_storeu_ps(scoreptr + (size_t)(jj + j + 1) * 4, _sum1);
                    _mm_storeu_ps(scoreptr + (size_t)(jj + j + 2) * 4, _sum2);
                    _mm_storeu_ps(scoreptr + (size_t)(jj + j + 3) * 4, _sum3);
                    _block_max = _mm_max_ps(_block_max, _mm_max_ps(_mm_max_ps(_sum0, _sum1), _mm_max_ps(_sum2, _sum3)));
                }
                for (; j < max_nn; j++)
                {
                    __m128 _sum = pM ? _mm_loadu_ps(pM + (size_t)(n + jj + j) * 4) : _mm_setzero_ps();
                    const float* pK = key_panel + j;
                    const float* pA = pQ;
                    for (int d = 0; d < head_dim; d++)
                    {
                        _sum = _mm_comp_fmadd_ps(_mm_loadu_ps(pA), _mm_set1_ps(*pK), _sum);
                        pA += 4;
                        pK += NR;
                    }
                    _mm_storeu_ps(scoreptr + (size_t)(jj + j) * 4, _sum);
                    _block_max = _mm_max_ps(_block_max, _sum);
                }
            }

            __m128 _m = _mm_loadu_ps(mT + ii);
            __m128 _l = _mm_loadu_ps(lT + ii);
            __m128 _m_new = _mm_max_ps(_m, _block_max);
            const __m128 _alpha_active = _mm_cmpneq_ps(_l, _mm_setzero_ps());
            __m128 _alpha = exp_ps(_mm_and_ps(_alpha_active, _mm_sub_ps(_m, _m_new)));
            _alpha = _mm_and_ps(_alpha, _alpha_active);

            float* outptr = outT_ptr + (size_t)ii * value_dim;
            for (int d = 0; d < value_dim; d++)
                _mm_storeu_ps(outptr + (size_t)d * 4, _mm_mul_ps(_mm_loadu_ps(outptr + (size_t)d * 4), _alpha));

            __m128 _sum0 = _mm_setzero_ps();
            __m128 _sum1 = _mm_setzero_ps();
            __m128 _sum2 = _mm_setzero_ps();
            __m128 _sum3 = _mm_setzero_ps();
            int j = 0;
            for (; j + 3 < max_jj; j += 4)
            {
                __m128 _p0 = exp_ps(_mm_sub_ps(_mm_loadu_ps(scoreptr + (size_t)j * 4), _m_new));
                __m128 _p1 = exp_ps(_mm_sub_ps(_mm_loadu_ps(scoreptr + (size_t)(j + 1) * 4), _m_new));
                __m128 _p2 = exp_ps(_mm_sub_ps(_mm_loadu_ps(scoreptr + (size_t)(j + 2) * 4), _m_new));
                __m128 _p3 = exp_ps(_mm_sub_ps(_mm_loadu_ps(scoreptr + (size_t)(j + 3) * 4), _m_new));
                _mm_storeu_ps(scoreptr + (size_t)j * 4, _p0);
                _mm_storeu_ps(scoreptr + (size_t)(j + 1) * 4, _p1);
                _mm_storeu_ps(scoreptr + (size_t)(j + 2) * 4, _p2);
                _mm_storeu_ps(scoreptr + (size_t)(j + 3) * 4, _p3);
                _sum0 = _mm_add_ps(_sum0, _p0);
                _sum1 = _mm_add_ps(_sum1, _p1);
                _sum2 = _mm_add_ps(_sum2, _p2);
                _sum3 = _mm_add_ps(_sum3, _p3);
            }
            for (; j < max_jj; j++)
            {
                __m128 _p = exp_ps(_mm_sub_ps(_mm_loadu_ps(scoreptr + (size_t)j * 4), _m_new));
                _mm_storeu_ps(scoreptr + (size_t)j * 4, _p);
                _sum0 = _mm_add_ps(_sum0, _p);
            }
            __m128 _sum = _mm_add_ps(_mm_add_ps(_sum0, _sum1), _mm_add_ps(_sum2, _sum3));
            _mm_storeu_ps(mT + ii, _m_new);
            _mm_storeu_ps(lT + ii, _mm_add_ps(_mm_mul_ps(_l, _alpha), _sum));
        }
#endif // __SSE2__
        for (; ii < max_ii; ii++)
        {
            const float* pQ = queryT_ptr + (size_t)ii * head_dim;
            float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N;
            const float* pM = maskT_ptr ? maskT_ptr + (size_t)ii * key_seqlen : 0;
            float block_max = -FLT_MAX;

            for (int jj = 0; jj < max_jj; jj += NR)
            {
                const int max_nn = std::min(NR, max_jj - jj);
                const float* key_panel = (const float*)packed_key_head + (size_t)(n + jj) * head_dim;
                for (int j = 0; j < max_nn; j++)
                {
                    float sum = pM ? pM[n + jj + j] : 0.f;
                    const float* pK = key_panel + j;
                    for (int d = 0; d < head_dim; d++)
                    {
                        sum += pQ[d] * *pK;
                        pK += NR;
                    }
                    scoreptr[jj + j] = sum;
                    block_max = std::max(block_max, sum);
                }
            }

            const float m_new = std::max(mT[ii], block_max);
            const float alpha = lT[ii] == 0.f ? 0.f : expf(mT[ii] - m_new);
            float* outptr = outT_ptr + (size_t)ii * value_dim;
            for (int d = 0; d < value_dim; d++)
                outptr[d] *= alpha;

            float sum = 0.f;
            for (int j = 0; j < max_jj; j++)
            {
                scoreptr[j] = expf(scoreptr[j] - m_new);
                sum += scoreptr[j];
            }
            mT[ii] = m_new;
            lT[ii] = lT[ii] * alpha + sum;
        }

        for (int jj = 0; jj < max_jj; jj += NR)
        {
            const int max_nn = std::min(NR, max_jj - jj);
            const float* value_panel = (const float*)packed_value_head + (size_t)(n + jj) * value_dim;

            for (int d = 0; d < value_dim;)
            {
                const int value_panel_width = sdpa_kvcache_value_panel_width(value_dim - d);

                ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; ii + 15 < max_ii; ii += 16)
                {
                    const float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N + (size_t)jj * 16;
                    float* outptr = outT_ptr + (size_t)ii * value_dim + (size_t)d * 16;
                    int lane = 0;
                    for (; lane + 15 < value_panel_width; lane += 16)
                    {
                        float* outptr0 = outptr + (size_t)lane * 16;
                        __m512 _out0 = _mm512_loadu_ps(outptr0);
                        __m512 _out1 = _mm512_loadu_ps(outptr0 + 16);
                        __m512 _out2 = _mm512_loadu_ps(outptr0 + 32);
                        __m512 _out3 = _mm512_loadu_ps(outptr0 + 48);
                        __m512 _out4 = _mm512_loadu_ps(outptr0 + 64);
                        __m512 _out5 = _mm512_loadu_ps(outptr0 + 80);
                        __m512 _out6 = _mm512_loadu_ps(outptr0 + 96);
                        __m512 _out7 = _mm512_loadu_ps(outptr0 + 112);
                        __m512 _out8 = _mm512_loadu_ps(outptr0 + 128);
                        __m512 _out9 = _mm512_loadu_ps(outptr0 + 144);
                        __m512 _outa = _mm512_loadu_ps(outptr0 + 160);
                        __m512 _outb = _mm512_loadu_ps(outptr0 + 176);
                        __m512 _outc = _mm512_loadu_ps(outptr0 + 192);
                        __m512 _outd = _mm512_loadu_ps(outptr0 + 208);
                        __m512 _oute = _mm512_loadu_ps(outptr0 + 224);
                        __m512 _outf = _mm512_loadu_ps(outptr0 + 240);
                        const float* pV = value_panel + (size_t)d * NR + lane;
                        const float* pS = scoreptr;
                        for (int j = 0; j < max_nn; j++)
                        {
                            const __m512 _p = _mm512_loadu_ps(pS);
                            const __m512 _v = _mm512_loadu_ps(pV);
                            const __m512 _v0 = _mm512_shuffle_f32x4(_v, _v, _MM_SHUFFLE(0, 0, 0, 0));
                            const __m512 _v1 = _mm512_shuffle_f32x4(_v, _v, _MM_SHUFFLE(1, 1, 1, 1));
                            const __m512 _v2 = _mm512_shuffle_f32x4(_v, _v, _MM_SHUFFLE(2, 2, 2, 2));
                            const __m512 _v3 = _mm512_shuffle_f32x4(_v, _v, _MM_SHUFFLE(3, 3, 3, 3));
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
                        _mm512_storeu_ps(outptr0, _out0);
                        _mm512_storeu_ps(outptr0 + 16, _out1);
                        _mm512_storeu_ps(outptr0 + 32, _out2);
                        _mm512_storeu_ps(outptr0 + 48, _out3);
                        _mm512_storeu_ps(outptr0 + 64, _out4);
                        _mm512_storeu_ps(outptr0 + 80, _out5);
                        _mm512_storeu_ps(outptr0 + 96, _out6);
                        _mm512_storeu_ps(outptr0 + 112, _out7);
                        _mm512_storeu_ps(outptr0 + 128, _out8);
                        _mm512_storeu_ps(outptr0 + 144, _out9);
                        _mm512_storeu_ps(outptr0 + 160, _outa);
                        _mm512_storeu_ps(outptr0 + 176, _outb);
                        _mm512_storeu_ps(outptr0 + 192, _outc);
                        _mm512_storeu_ps(outptr0 + 208, _outd);
                        _mm512_storeu_ps(outptr0 + 224, _oute);
                        _mm512_storeu_ps(outptr0 + 240, _outf);
                    }
                    for (; lane < value_panel_width; lane++)
                    {
                        __m512 _out = _mm512_loadu_ps(outptr + (size_t)lane * 16);
                        const float* pV = value_panel + (size_t)d * NR + lane;
                        const float* pS = scoreptr;
                        for (int j = 0; j < max_nn; j++)
                        {
                            _out = _mm512_fmadd_ps(_mm512_loadu_ps(pS), _mm512_set1_ps(*pV), _out);
                            pS += 16;
                            pV += value_panel_width;
                        }
                        _mm512_storeu_ps(outptr + (size_t)lane * 16, _out);
                    }
                }
#endif // __AVX512F__
                for (; ii + 7 < max_ii; ii += 8)
                {
                    const float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N + (size_t)jj * 8;
                    float* outptr = outT_ptr + (size_t)ii * value_dim + (size_t)d * 8;
                    int lane = 0;
                    for (; lane + 7 < value_panel_width; lane += 8)
                    {
                        float* outptr0 = outptr + (size_t)lane * 8;
                        __m256 _out0 = _mm256_loadu_ps(outptr0);
                        __m256 _out1 = _mm256_loadu_ps(outptr0 + 8);
                        __m256 _out2 = _mm256_loadu_ps(outptr0 + 16);
                        __m256 _out3 = _mm256_loadu_ps(outptr0 + 24);
                        __m256 _out4 = _mm256_loadu_ps(outptr0 + 32);
                        __m256 _out5 = _mm256_loadu_ps(outptr0 + 40);
                        __m256 _out6 = _mm256_loadu_ps(outptr0 + 48);
                        __m256 _out7 = _mm256_loadu_ps(outptr0 + 56);
                        const float* pV = value_panel + (size_t)d * NR + lane;
                        const float* pS = scoreptr;
                        for (int j = 0; j < max_nn; j++)
                        {
                            const __m256 _p = _mm256_loadu_ps(pS);
                            const __m256 _v = _mm256_loadu_ps(pV);
                            const __m256 _v0 = _mm256_permute2f128_ps(_v, _v, 0x00);
                            const __m256 _v1 = _mm256_permute2f128_ps(_v, _v, 0x11);
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
                        _mm256_storeu_ps(outptr0, _out0);
                        _mm256_storeu_ps(outptr0 + 8, _out1);
                        _mm256_storeu_ps(outptr0 + 16, _out2);
                        _mm256_storeu_ps(outptr0 + 24, _out3);
                        _mm256_storeu_ps(outptr0 + 32, _out4);
                        _mm256_storeu_ps(outptr0 + 40, _out5);
                        _mm256_storeu_ps(outptr0 + 48, _out6);
                        _mm256_storeu_ps(outptr0 + 56, _out7);
                    }
                    for (; lane < value_panel_width; lane++)
                    {
                        __m256 _out = _mm256_loadu_ps(outptr + (size_t)lane * 8);
                        const float* pV = value_panel + (size_t)d * NR + lane;
                        const float* pS = scoreptr;
                        for (int j = 0; j < max_nn; j++)
                        {
                            _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pS), _mm256_set1_ps(*pV), _out);
                            pS += 8;
                            pV += value_panel_width;
                        }
                        _mm256_storeu_ps(outptr + (size_t)lane * 8, _out);
                    }
                }
#endif // __AVX__
                for (; ii + 3 < max_ii; ii += 4)
                {
                    const float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N + (size_t)jj * 4;
                    float* outptr = outT_ptr + (size_t)ii * value_dim + (size_t)d * 4;

                    int lane = 0;
                    for (; lane + 3 < value_panel_width; lane += 4)
                    {
                        __m128 _out0 = _mm_loadu_ps(outptr);
                        __m128 _out1 = _mm_loadu_ps(outptr + 4);
                        __m128 _out2 = _mm_loadu_ps(outptr + 8);
                        __m128 _out3 = _mm_loadu_ps(outptr + 12);
                        const float* pV = value_panel + (size_t)d * NR + lane;
                        const float* pS = scoreptr;
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
                        _mm_storeu_ps(outptr, _out0);
                        _mm_storeu_ps(outptr + 4, _out1);
                        _mm_storeu_ps(outptr + 8, _out2);
                        _mm_storeu_ps(outptr + 12, _out3);
                        outptr += 16;
                    }
                    for (; lane < value_panel_width; lane++)
                    {
                        __m128 _out = _mm_loadu_ps(outptr);
                        const float* pV = value_panel + (size_t)d * NR + lane;
                        const float* pS = scoreptr;
                        for (int j = 0; j < max_nn; j++)
                        {
                            _out = _mm_comp_fmadd_ps(_mm_loadu_ps(pS), _mm_set1_ps(*pV), _out);
                            pS += 4;
                            pV += value_panel_width;
                        }
                        _mm_storeu_ps(outptr, _out);
                        outptr += 4;
                    }
                }
#endif // __SSE2__
                for (; ii < max_ii; ii++)
                {
                    const float* scoreptr = scoreT_ptr + (size_t)ii * TILE_N + jj;
                    float* outptr = outT_ptr + (size_t)ii * value_dim + d;
                    for (int lane = 0; lane < value_panel_width; lane++)
                    {
                        float sum = outptr[lane];
                        const float* pV = value_panel + (size_t)d * NR + lane;
                        for (int j = 0; j < max_nn; j++)
                        {
                            sum += scoreptr[j] * *pV;
                            pV += value_panel_width;
                        }
                        outptr[lane] = sum;
                    }
                }

                d += value_panel_width;
            }
        }
    }

}

static int sdpa_prefill_packed_fp32(const Mat& query, const Mat& packed_key, const Mat& packed_value, const Mat& attn_mask, Mat& top_blob, float scale, const Option& opt)
{
    const int head_dim = query.w;
    const int value_dim = packed_value.w;
    const int query_seqlen = query.h;
    const int key_seqlen = packed_key.h;
    const int num_query_heads = query.c;
    const int num_kv_heads = packed_key.c;
    const int num_query_heads_per_kv_head = num_query_heads / num_kv_heads;
    const int nT = std::max(opt.num_threads, 1);
    const int TILE_M = sdpa_prefill_get_optimal_tile_m(query_seqlen, num_query_heads, nT);
#if __SSE2__
#if __AVX__
#if __AVX512F__
    const int NR = 16;
#else
    const int NR = 8;
#endif // __AVX512F__
#else
    const int NR = 4;
#endif // __AVX__
#else
    const int NR = 1;
#endif // __SSE2__

    const int num_mblocks = (query_seqlen + TILE_M - 1) / TILE_M;
    const int num_tasks = num_query_heads * num_mblocks;
    int TILE_N = sdpa_prefill_get_optimal_tile_n(head_dim, value_dim, key_seqlen, 4, 4, 4, attn_mask.empty() ? 0 : 4, TILE_M, num_tasks, nT);
    TILE_N = std::max(NR, (TILE_N + NR - 1) / NR * NR);
    const int num_key_blocks = (key_seqlen + TILE_N - 1) / TILE_N;

    int num_kv_chunks = 1;
    if (num_tasks < nT && num_key_blocks >= 2)
    {
        num_kv_chunks = std::min((nT + num_tasks - 1) / num_tasks, num_key_blocks);
        num_kv_chunks = std::max(num_kv_chunks, 1);
    }

    const int query_workspace_size = TILE_M * head_dim;
    const int score_workspace_size = TILE_M * TILE_N;
    const int out_workspace_size = TILE_M * value_dim;
    const int state_workspace_size = TILE_M * 2;
    const int workspace_size = query_workspace_size + score_workspace_size + out_workspace_size + state_workspace_size;

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

        sdpa_pack_mask_fp32(attn_mask, packed_mask, TILE_M, opt);
    }

    Mat packed_query;
    if (num_kv_chunks > 1)
    {
        packed_query.create(query_workspace_size, 1, num_tasks, 4u, opt.workspace_allocator);
        if (packed_query.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int task_id = 0; task_id < num_tasks; task_id++)
        {
            const int q = task_id / num_mblocks;
            const int i0 = task_id % num_mblocks * TILE_M;
            const int max_ii = std::min(query_seqlen - i0, TILE_M);
            const Mat query_head = query.channel(q);
            Mat queryT = packed_query.channel(task_id);
            sdpa_pack_query_fp32(query_head, queryT, i0, max_ii, scale);
        }
    }

    Mat partials;
    if (num_kv_chunks > 1)
    {
        partials.create((value_dim + 2) * TILE_M, 1, num_tasks * num_kv_chunks, 4u, opt.workspace_allocator);
        if (partials.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ti = 0; ti < num_tasks * num_kv_chunks; ti++)
    {
        const int task_id = ti / num_kv_chunks;
        const int chunk_id = ti % num_kv_chunks;
        const int q = task_id / num_mblocks;
        const int i0 = task_id % num_mblocks * TILE_M;
        const int max_ii = std::min(query_seqlen - i0, TILE_M);
        const int g = q / num_query_heads_per_kv_head;
        const int n_begin = chunk_id * num_key_blocks / num_kv_chunks * TILE_N;
        const int n_end = std::min((chunk_id + 1) * num_key_blocks / num_kv_chunks * TILE_N, key_seqlen);

        Mat workspace_tile = workspace.channel(get_omp_thread_num());
        Mat queryT = workspace_tile.range(0, query_workspace_size);
        Mat scoreT = workspace_tile.range(query_workspace_size, score_workspace_size);
        Mat outT = workspace_tile.range(query_workspace_size + score_workspace_size, out_workspace_size);
        Mat stateT = workspace_tile.range(query_workspace_size + score_workspace_size + out_workspace_size, state_workspace_size);
        float* outT_ptr = outT;
        float* mT = stateT;
        float* lT = mT + TILE_M;

        const Mat query_head = query.channel(q);
        const Mat packed_key_head = packed_key.channel(g);
        const Mat packed_value_head = packed_value.channel(g);
        Mat maskT;
        if (!packed_mask.empty())
        {
            Mat packed_mask_head = packed_mask.channel(packed_mask.c > 1 ? q : 0);
            maskT = packed_mask_head.row_range(task_id % num_mblocks, 1);
        }

        if (!packed_query.empty())
            queryT = packed_query.channel(task_id);
        else
            sdpa_pack_query_fp32(query_head, queryT, i0, max_ii, scale);

        sdpa_prefill_packed_tile_fp32(queryT, packed_key_head, packed_value_head, maskT, scoreT, outT, stateT, max_ii, n_begin, n_end);

        int ii = 0;
        if (num_kv_chunks > 1)
        {
            float* stateptr = partials.channel(ti);
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; ii + 15 < max_ii; ii += 16)
            {
                _mm512_storeu_ps(stateptr + ii, _mm512_loadu_ps(mT + ii));
                _mm512_storeu_ps(stateptr + TILE_M + ii, _mm512_loadu_ps(lT + ii));
                const float* outptr = outT_ptr + (size_t)ii * value_dim;
                for (int d = 0; d < value_dim; d++)
                    _mm512_storeu_ps(stateptr + (size_t)(d + 2) * TILE_M + ii, _mm512_loadu_ps(outptr + (size_t)d * 16));
            }
#endif // __AVX512F__
            for (; ii + 7 < max_ii; ii += 8)
            {
                _mm256_storeu_ps(stateptr + ii, _mm256_loadu_ps(mT + ii));
                _mm256_storeu_ps(stateptr + TILE_M + ii, _mm256_loadu_ps(lT + ii));
                const float* outptr = outT_ptr + (size_t)ii * value_dim;
                for (int d = 0; d < value_dim; d++)
                    _mm256_storeu_ps(stateptr + (size_t)(d + 2) * TILE_M + ii, _mm256_loadu_ps(outptr + (size_t)d * 8));
            }
#endif // __AVX__
            for (; ii + 3 < max_ii; ii += 4)
            {
                _mm_storeu_ps(stateptr + ii, _mm_loadu_ps(mT + ii));
                _mm_storeu_ps(stateptr + TILE_M + ii, _mm_loadu_ps(lT + ii));
                const float* outptr = outT_ptr + (size_t)ii * value_dim;
                for (int d = 0; d < value_dim; d++)
                    _mm_storeu_ps(stateptr + (size_t)(d + 2) * TILE_M + ii, _mm_loadu_ps(outptr + (size_t)d * 4));
            }
#endif // __SSE2__
            for (; ii < max_ii; ii++)
            {
                stateptr[ii] = mT[ii];
                stateptr[TILE_M + ii] = lT[ii];
                const float* outptr = outT_ptr + (size_t)ii * value_dim;
                for (int d = 0; d < value_dim; d++)
                    stateptr[(size_t)(d + 2) * TILE_M + ii] = outptr[d];
            }
        }
        else
        {
            Mat top_blob_head = top_blob.channel(q);
            Mat lT_tile = stateT.range(TILE_M, TILE_M);
            sdpa_store_output_tile(outT, lT_tile, top_blob_head, i0, max_ii);
        }
    }

    if (num_kv_chunks > 1)
        sdpa_prefill_reduce(partials, top_blob, workspace, num_tasks, num_mblocks, TILE_M, num_kv_chunks, query_seqlen, value_dim, opt);

    return 0;
}

static int sdpa_prefill_fp32(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask, Mat& top_blob, float scale, const Option& opt)
{
#if __SSE2__
#if __AVX__
#if __AVX512F__
    const int panel_width = 16;
#else
    const int panel_width = 8;
#endif // __AVX512F__
#else
    const int panel_width = 4;
#endif // __AVX__
#else
    const int panel_width = 1;
#endif // __SSE2__
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
    const int nT = key.h >= panel_width ? opt.num_threads : 1;

    #pragma omp parallel for num_threads(nT)
    for (int task_id = 0; task_id < num_kv_heads * num_panels; task_id++)
    {
        const int g = task_id / num_panels;
        const int panel_id = task_id % num_panels;
        const int panel_begin = panel_id * panel_width;
        const int n_begin = panel_begin;
        const int n_end = std::min(key.h, panel_begin + panel_width);
        const Mat key_head = key.channel(g);
        const Mat value_head = value.channel(g);
        Mat packed_key_head = packed_key.channel(g);
        Mat packed_value_head = packed_value.channel(g);
        Mat packed_key_tile(key.w * panel_width, (float*)packed_key_head + (size_t)panel_id * key.w * panel_width, 4u);
        Mat packed_value_tile(value.w * panel_width, (float*)packed_value_head + (size_t)panel_id * value.w * panel_width, 4u);

        sdpa_pack_key_tile_fp32(key_head, packed_key_tile, n_begin, 0, n_end - n_begin);
        sdpa_pack_value_tile_fp32(value_head, packed_value_tile, n_begin, 0, n_end - n_begin);
    }

    return sdpa_prefill_packed_fp32(query, packed_key, packed_value, attn_mask, top_blob, scale, opt);
}

static int sdpa_kvcache_fp32(const Mat& query, const Mat& past_key, const Mat& past_value, const Mat& cur_key, const Mat& cur_value, Mat& cached_key, Mat& cached_value, const Mat& attn_mask, Mat& top_blob, float scale, const Option& opt)
{
    const int past_seqlen = past_key.empty() ? 0 : past_key.h;
    const int dst_seqlen = past_seqlen + cur_key.h;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    const int panel_width = 16;
#else
    const int panel_width = 8;
#endif // __AVX512F__
#else
    const int panel_width = 4;
#endif // __AVX__
#else
    const int panel_width = 1;
#endif // __SSE2__

    int ret = sdpa_create_or_grow_kvcache(past_key, cached_key, dst_seqlen, cur_key.c, cur_key.w, cur_key.elemsize, panel_width, opt);
    if (ret != 0)
        return ret;

    ret = sdpa_create_or_grow_kvcache(past_value, cached_value, dst_seqlen, cur_value.c, cur_value.w, cur_value.elemsize, panel_width, opt);
    if (ret != 0)
        return ret;

    const int num_kv_heads = cur_key.c;
    const int first_panel = past_seqlen / panel_width;
    const int num_panels = (past_seqlen % panel_width + cur_key.h + panel_width - 1) / panel_width;
    const int nT = cur_key.h >= panel_width ? opt.num_threads : 1;

    #pragma omp parallel for num_threads(nT)
    for (int task_id = 0; task_id < num_kv_heads * num_panels; task_id++)
    {
        const int g = task_id / num_panels;
        const int panel_id = first_panel + task_id % num_panels;
        const int panel_begin = panel_id * panel_width;
        const int n_begin = std::max(past_seqlen, panel_begin);
        const int n_end = std::min(dst_seqlen, panel_begin + panel_width);
        const Mat key_head = cur_key.channel(g);
        const Mat value_head = cur_value.channel(g);
        Mat packed_key_head = cached_key.channel(g);
        Mat packed_value_head = cached_value.channel(g);
        Mat packed_key_tile(cur_key.w * panel_width, (float*)packed_key_head + (size_t)panel_id * cur_key.w * panel_width, 4u);
        Mat packed_value_tile(cur_value.w * panel_width, (float*)packed_value_head + (size_t)panel_id * cur_value.w * panel_width, 4u);

        sdpa_pack_key_tile_fp32(key_head, packed_key_tile, n_begin - past_seqlen, n_begin - panel_begin, n_end - n_begin);
        sdpa_pack_value_tile_fp32(value_head, packed_value_tile, n_begin - past_seqlen, n_begin - panel_begin, n_end - n_begin);
    }

    if (query.h == 1)
        return sdpa_decode_kvcache_fp32(query, cached_key, cached_value, attn_mask, top_blob, scale, opt);

    return sdpa_prefill_packed_fp32(query, cached_key, cached_value, attn_mask, top_blob, scale, opt);
}

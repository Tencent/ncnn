// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static int sdpa_prefill_block_m(int query_seqlen, int num_query_heads, int num_kv_heads, int value_dim, int num_threads)
{
#if __SSE2__
#if __AVX__
#if __AVX512F__
    int block_m = 16;
#else
    int block_m = 8;
#endif // __AVX512F__
#else
    int block_m = 4;
#endif // __AVX__
#else
    int block_m = 1;
#endif // __SSE2__

    const int num_query_heads_per_kv_head = num_query_heads / num_kv_heads;
    const int target_tasks = value_dim >= 192 || num_query_heads_per_kv_head >= 4 ? (num_threads + 1) / 2 : num_threads;
    while (block_m > 4)
    {
        const int num_tasks = num_query_heads * ((query_seqlen + block_m - 1) / block_m);
        const int narrower_block_m = block_m / 2;
        const int narrower_num_tasks = num_query_heads * ((query_seqlen + narrower_block_m - 1) / narrower_block_m);
        if (num_tasks >= target_tasks || narrower_num_tasks == num_tasks)
            break;

        block_m = narrower_block_m;
    }

    return block_m;
}

static int sdpa_prefill_block_n(int head_dim, int value_dim, int key_seqlen, int query_seqlen, int key_storage_size, int value_storage_size, int mask_storage_size, int block_m)
{
    size_t l2_cache_size = get_cpu_level2_cache_size();
    if (l2_cache_size == 0)
        l2_cache_size = 256 * 1024;

    const size_t cache_budget = l2_cache_size * 3 / 4;
    const size_t fixed_size = (size_t)block_m * (head_dim + value_dim) * sizeof(float);
    const size_t size_per_token = (size_t)head_dim * key_storage_size + (size_t)value_dim * value_storage_size + (size_t)block_m * (sizeof(float) + mask_storage_size);

    int block_n = 64;
    if (fixed_size + size_per_token * 256 <= cache_budget)
        block_n = 256;
    else if (fixed_size + size_per_token * 128 <= cache_budget)
        block_n = 128;

    if (query_seqlen >= 256 && (head_dim >= 128 || value_dim >= 128))
        block_n = std::min(block_n, 128);

    const int short_block_n = (key_seqlen + block_m - 1) / block_m * block_m;
    return std::min(block_n, std::max(short_block_n, block_m));
}

// packed_key[block][key_panel][head_dim][key_lane]
static void sdpa_pack_key_fp32(const Mat& key, Mat& packed_key, int block_n, const Option& opt)
{
    const int head_dim = key.w;
    const int key_seqlen = key.h;
    const int num_kv_heads = key.c;

    const int num_key_blocks = (key_seqlen + block_n - 1) / block_n;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int task_id = 0; task_id < num_kv_heads * num_key_blocks; task_id++)
    {
        const int g = task_id / num_key_blocks;
        const int block_id = task_id % num_key_blocks;
        const int n = block_id * block_n;

        const Mat key_head = key.channel(g);
        Mat packed_key_head = packed_key.channel(g);
        const float* key_base = (const float*)key_head + (size_t)n * head_dim;
        float* pp = packed_key_head.row(block_id);

        const int max_jj = std::min(block_n, key_seqlen - n);
        int j = 0;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; j + 15 < max_jj; j += 16)
        {
            const float* p0 = key_base + (size_t)j * head_dim;
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

            int k = 0;
            for (; k + 15 < head_dim; k += 16)
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
                _mm512_storeu_ps(pp + 16, _r1);
                _mm512_storeu_ps(pp + 32, _r2);
                _mm512_storeu_ps(pp + 48, _r3);
                _mm512_storeu_ps(pp + 64, _r4);
                _mm512_storeu_ps(pp + 80, _r5);
                _mm512_storeu_ps(pp + 96, _r6);
                _mm512_storeu_ps(pp + 112, _r7);
                _mm512_storeu_ps(pp + 128, _r8);
                _mm512_storeu_ps(pp + 144, _r9);
                _mm512_storeu_ps(pp + 160, _ra);
                _mm512_storeu_ps(pp + 176, _rb);
                _mm512_storeu_ps(pp + 192, _rc);
                _mm512_storeu_ps(pp + 208, _rd);
                _mm512_storeu_ps(pp + 224, _re);
                _mm512_storeu_ps(pp + 240, _rf);
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
            for (; k < head_dim; k++)
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
                pp += 16;
            }
        }
#endif // __AVX512F__
        for (; j + 7 < max_jj; j += 8)
        {
            const float* p0 = key_base + (size_t)j * head_dim;
            const float* p1 = p0 + head_dim;
            const float* p2 = p1 + head_dim;
            const float* p3 = p2 + head_dim;
            const float* p4 = p3 + head_dim;
            const float* p5 = p4 + head_dim;
            const float* p6 = p5 + head_dim;
            const float* p7 = p6 + head_dim;

            int k = 0;
            for (; k + 7 < head_dim; k += 8)
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
                _mm256_storeu_ps(pp + 8, _r1);
                _mm256_storeu_ps(pp + 16, _r2);
                _mm256_storeu_ps(pp + 24, _r3);
                _mm256_storeu_ps(pp + 32, _r4);
                _mm256_storeu_ps(pp + 40, _r5);
                _mm256_storeu_ps(pp + 48, _r6);
                _mm256_storeu_ps(pp + 56, _r7);
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
            for (; k < head_dim; k++)
            {
                pp[0] = *p0++;
                pp[1] = *p1++;
                pp[2] = *p2++;
                pp[3] = *p3++;
                pp[4] = *p4++;
                pp[5] = *p5++;
                pp[6] = *p6++;
                pp[7] = *p7++;
                pp += 8;
            }
        }
#endif // __AVX__
        for (; j + 3 < max_jj; j += 4)
        {
            const float* p0 = key_base + (size_t)j * head_dim;
            const float* p1 = p0 + head_dim;
            const float* p2 = p1 + head_dim;
            const float* p3 = p2 + head_dim;

            int k = 0;
            for (; k + 3 < head_dim; k += 4)
            {
                __m128 _r0 = _mm_loadu_ps(p0);
                __m128 _r1 = _mm_loadu_ps(p1);
                __m128 _r2 = _mm_loadu_ps(p2);
                __m128 _r3 = _mm_loadu_ps(p3);

                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

                _mm_storeu_ps(pp, _r0);
                _mm_storeu_ps(pp + 4, _r1);
                _mm_storeu_ps(pp + 8, _r2);
                _mm_storeu_ps(pp + 12, _r3);
                pp += 16;

                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
            for (; k < head_dim; k++)
            {
                pp[0] = *p0++;
                pp[1] = *p1++;
                pp[2] = *p2++;
                pp[3] = *p3++;
                pp += 4;
            }
        }
#endif // __SSE2__

        for (; j < max_jj; j++)
        {
            const float* p0 = key_base + (size_t)j * head_dim;
            memcpy(pp, p0, (size_t)head_dim * sizeof(float));
            pp += head_dim;
        }
    }
}

// packed_value block layout:
//   value panels are stored as 16 / 8 / 4 / 1 columns, selected by available ISA.
//   Inside each value panel, data is key-major and value-lane-contiguous.
static void sdpa_pack_value_fp32(const Mat& value, Mat& packed_value, int block_n, const Option& opt)
{
    const int value_dim = value.w;
    const int key_seqlen = value.h;
    const int num_kv_heads = value.c;
    const int num_key_blocks = (key_seqlen + block_n - 1) / block_n;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int task_id = 0; task_id < num_kv_heads * num_key_blocks; task_id++)
    {
        const int g = task_id / num_key_blocks;
        const int block_id = task_id % num_key_blocks;
        const int n = block_id * block_n;
        const int max_jj = std::min(block_n, key_seqlen - n);

        const Mat value_head = value.channel(g);
        const float* value_base = value_head.row(n);
        float* pp = packed_value.channel(g).row(block_id);

        int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; d + 15 < value_dim; d += 16)
        {
            const float* p0 = value_base + d;
            int j = 0;
            for (; j + 3 < max_jj; j += 4)
            {
                _mm512_storeu_ps(pp, _mm512_loadu_ps(p0));
                _mm512_storeu_ps(pp + 16, _mm512_loadu_ps(p0 + value_dim));
                _mm512_storeu_ps(pp + 32, _mm512_loadu_ps(p0 + value_dim * 2));
                _mm512_storeu_ps(pp + 48, _mm512_loadu_ps(p0 + value_dim * 3));
                pp += 64;
                p0 += value_dim * 4;
            }
            for (; j < max_jj; j++)
            {
                _mm512_storeu_ps(pp, _mm512_loadu_ps(p0));
                pp += 16;
                p0 += value_dim;
            }
        }
#endif // __AVX512F__
        for (; d + 7 < value_dim; d += 8)
        {
            const float* p0 = value_base + d;
            int j = 0;
            for (; j + 3 < max_jj; j += 4)
            {
                _mm256_storeu_ps(pp, _mm256_loadu_ps(p0));
                _mm256_storeu_ps(pp + 8, _mm256_loadu_ps(p0 + value_dim));
                _mm256_storeu_ps(pp + 16, _mm256_loadu_ps(p0 + value_dim * 2));
                _mm256_storeu_ps(pp + 24, _mm256_loadu_ps(p0 + value_dim * 3));
                pp += 32;
                p0 += value_dim * 4;
            }
            for (; j < max_jj; j++)
            {
                _mm256_storeu_ps(pp, _mm256_loadu_ps(p0));
                pp += 8;
                p0 += value_dim;
            }
        }
#endif // __AVX__
        for (; d + 3 < value_dim; d += 4)
        {
            const float* p0 = value_base + d;
            int j = 0;
            for (; j + 3 < max_jj; j += 4)
            {
                _mm_storeu_ps(pp, _mm_loadu_ps(p0));
                _mm_storeu_ps(pp + 4, _mm_loadu_ps(p0 + value_dim));
                _mm_storeu_ps(pp + 8, _mm_loadu_ps(p0 + value_dim * 2));
                _mm_storeu_ps(pp + 12, _mm_loadu_ps(p0 + value_dim * 3));
                pp += 16;
                p0 += value_dim * 4;
            }
            for (; j < max_jj; j++)
            {
                _mm_storeu_ps(pp, _mm_loadu_ps(p0));
                pp += 4;
                p0 += value_dim;
            }
        }
#endif // __SSE2__
        for (; d < value_dim; d++)
        {
            const float* p0 = value_base + d;
            int j = 0;
            for (; j + 3 < max_jj; j += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[value_dim];
                pp[2] = p0[value_dim * 2];
                pp[3] = p0[value_dim * 3];
                pp += 4;
                p0 += value_dim * 4;
            }
            for (; j < max_jj; j++)
            {
                *pp++ = *p0;
                p0 += value_dim;
            }
        }
    }
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

#if __SSE2__

static void sdpa_store_output_tile4(const Mat& outT, Mat& top_blob_head, int i, float* state, int state_stride, __m128 _m, __m128 _l, int value_dim)
{
    const float* outptr = outT;
    if (state)
    {
        _mm_storeu_ps(state, _m);
        _mm_storeu_ps(state + state_stride, _l);
        float* state_out = state + state_stride * 2;
        for (int d = 0; d < value_dim; d++)
        {
            _mm_storeu_ps(state_out, _mm_loadu_ps(outptr));
            outptr += 4;
            state_out += state_stride;
        }
        return;
    }

    float* output0 = top_blob_head.row(i);
    float* output1 = top_blob_head.row(i + 1);
    float* output2 = top_blob_head.row(i + 2);
    float* output3 = top_blob_head.row(i + 3);
    const __m128 _nonzero = _mm_cmpneq_ps(_l, _mm_setzero_ps());
    const __m128 _denom = _mm_or_ps(_mm_and_ps(_nonzero, _l), _mm_andnot_ps(_nonzero, _mm_set1_ps(1.f)));
    const __m128 _scale = _mm_and_ps(_mm_div_ps(_mm_set1_ps(1.f), _denom), _nonzero);

    int d = 0;
    for (; d + 3 < value_dim; d += 4)
    {
        __m128 _r0 = _mm_mul_ps(_mm_loadu_ps(outptr), _scale);
        __m128 _r1 = _mm_mul_ps(_mm_loadu_ps(outptr + 4), _scale);
        __m128 _r2 = _mm_mul_ps(_mm_loadu_ps(outptr + 8), _scale);
        __m128 _r3 = _mm_mul_ps(_mm_loadu_ps(outptr + 12), _scale);
        _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
        _mm_storeu_ps(output0 + d, _r0);
        _mm_storeu_ps(output1 + d, _r1);
        _mm_storeu_ps(output2 + d, _r2);
        _mm_storeu_ps(output3 + d, _r3);
        outptr += 16;
    }
    for (; d < value_dim; d++)
    {
        const __m128 _r = _mm_mul_ps(_mm_loadu_ps(outptr), _scale);
        output0[d] = _mm_cvtss_f32(_r);
        output1[d] = _mm_cvtss_f32(_mm_shuffle_ps(_r, _r, _MM_SHUFFLE(1, 1, 1, 1)));
        output2[d] = _mm_cvtss_f32(_mm_movehl_ps(_r, _r));
        output3[d] = _mm_cvtss_f32(_mm_shuffle_ps(_r, _r, _MM_SHUFFLE(3, 3, 3, 3)));
        outptr += 4;
    }
}

#if __AVX__

static void sdpa_store_output_tile8(const Mat& outT, Mat& top_blob_head, int i, float* state, int state_stride, __m256 _m, __m256 _l, int value_dim)
{
    const float* outptr = outT;
    if (state)
    {
        _mm256_storeu_ps(state, _m);
        _mm256_storeu_ps(state + state_stride, _l);
        float* state_out = state + state_stride * 2;
        for (int d = 0; d < value_dim; d++)
        {
            _mm256_storeu_ps(state_out, _mm256_loadu_ps(outptr));
            outptr += 8;
            state_out += state_stride;
        }
        return;
    }

    float* output0 = top_blob_head.row(i);
    float* output1 = top_blob_head.row(i + 1);
    float* output2 = top_blob_head.row(i + 2);
    float* output3 = top_blob_head.row(i + 3);
    float* output4 = top_blob_head.row(i + 4);
    float* output5 = top_blob_head.row(i + 5);
    float* output6 = top_blob_head.row(i + 6);
    float* output7 = top_blob_head.row(i + 7);
    const __m256 _nonzero = _mm256_cmp_ps(_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
    const __m256 _denom = _mm256_blendv_ps(_mm256_set1_ps(1.f), _l, _nonzero);
    const __m256 _scale = _mm256_and_ps(_mm256_div_ps(_mm256_set1_ps(1.f), _denom), _nonzero);

    int d = 0;
    for (; d + 7 < value_dim; d += 8)
    {
        __m256 _r0 = _mm256_mul_ps(_mm256_loadu_ps(outptr), _scale);
        __m256 _r1 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 8), _scale);
        __m256 _r2 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 16), _scale);
        __m256 _r3 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 24), _scale);
        __m256 _r4 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 32), _scale);
        __m256 _r5 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 40), _scale);
        __m256 _r6 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 48), _scale);
        __m256 _r7 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 56), _scale);
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
        const __m256 _r = _mm256_mul_ps(_mm256_loadu_ps(outptr), _scale);
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

#if __AVX512F__

static void sdpa_store_output_tile16(const Mat& outT, Mat& top_blob_head, int i, float* state, int state_stride, __m512 _m, __m512 _l, int value_dim)
{
    const float* outptr = outT;
    if (state)
    {
        _mm512_storeu_ps(state, _m);
        _mm512_storeu_ps(state + state_stride, _l);
        float* state_out = state + state_stride * 2;
        for (int d = 0; d < value_dim; d++)
        {
            _mm512_storeu_ps(state_out, _mm512_loadu_ps(outptr));
            outptr += 16;
            state_out += state_stride;
        }
        return;
    }

    float* output0 = top_blob_head.row(i);
    float* output1 = top_blob_head.row(i + 1);
    float* output2 = top_blob_head.row(i + 2);
    float* output3 = top_blob_head.row(i + 3);
    float* output4 = top_blob_head.row(i + 4);
    float* output5 = top_blob_head.row(i + 5);
    float* output6 = top_blob_head.row(i + 6);
    float* output7 = top_blob_head.row(i + 7);
    float* output8 = top_blob_head.row(i + 8);
    float* output9 = top_blob_head.row(i + 9);
    float* outputa = top_blob_head.row(i + 10);
    float* outputb = top_blob_head.row(i + 11);
    float* outputc = top_blob_head.row(i + 12);
    float* outputd = top_blob_head.row(i + 13);
    float* outpute = top_blob_head.row(i + 14);
    float* outputf = top_blob_head.row(i + 15);
    const __mmask16 nonzero = _mm512_cmp_ps_mask(_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
    const __m512 _scale = _mm512_maskz_div_ps(nonzero, _mm512_set1_ps(1.f), _l);
    int d = 0;
    for (; d + 15 < value_dim; d += 16)
    {
        __m512 _r0 = _mm512_mul_ps(_mm512_loadu_ps(outptr), _scale);
        __m512 _r1 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 16), _scale);
        __m512 _r2 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 32), _scale);
        __m512 _r3 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 48), _scale);
        __m512 _r4 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 64), _scale);
        __m512 _r5 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 80), _scale);
        __m512 _r6 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 96), _scale);
        __m512 _r7 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 112), _scale);
        __m512 _r8 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 128), _scale);
        __m512 _r9 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 144), _scale);
        __m512 _ra = _mm512_mul_ps(_mm512_loadu_ps(outptr + 160), _scale);
        __m512 _rb = _mm512_mul_ps(_mm512_loadu_ps(outptr + 176), _scale);
        __m512 _rc = _mm512_mul_ps(_mm512_loadu_ps(outptr + 192), _scale);
        __m512 _rd = _mm512_mul_ps(_mm512_loadu_ps(outptr + 208), _scale);
        __m512 _re = _mm512_mul_ps(_mm512_loadu_ps(outptr + 224), _scale);
        __m512 _rf = _mm512_mul_ps(_mm512_loadu_ps(outptr + 240), _scale);
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
        const __m512 _r = _mm512_mul_ps(_mm512_loadu_ps(outptr), _scale);
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
#endif // __AVX__
#endif // __SSE2__

static void sdpa_flash_attention_tile_fp32(const Mat& query, const Mat& key, const Mat& packed_key, const Mat& value, const Mat& packed_value, const Mat& attn_mask_blob, const Mat& packed_mask, Mat& top_blob, float scale, int q, int g, int i0, int max_ii, int n_begin, int n_end, int block_n, int state_stride, const Mat& packed_query, Mat& workspace, Mat& state)
{
    Mat top_blob_head = top_blob.channel(q);
    const Mat query_head = query.channel(q);
    const int head_dim = query.w;
    const int key_seqlen = key.h;
    const int value_dim = value.w;
    float* workspace_ptr = workspace;
    float* state_base = state;
    Mat queryT = packed_query;
    if (queryT.empty())
    {
        queryT = workspace.range((block_n + value_dim) * max_ii, head_dim * max_ii);
        sdpa_pack_query_fp32(query_head, queryT, i0, max_ii, scale);
    }
    const float* queryT_base = queryT;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const int i0x = i0 + ii;
        float* state_ptr = state.empty() ? 0 : state_base + ii;
        float* scoreT = workspace_ptr;
        Mat outT_tile = workspace.range(block_n * 16, value_dim * 16);
        float* outT = outT_tile;
        const float* queryT = queryT_base + ii * head_dim;
        const Mat key_head = key.channel(g);
        const Mat packed_key_head = packed_key.empty() ? Mat() : packed_key.channel(g);
        const Mat value_head = value.channel(g);
        const Mat packed_value_head = packed_value.empty() ? Mat() : packed_value.channel(g);
        const float* maskT = packed_mask.empty() ? 0 : (const float*)packed_mask + (size_t)ii * key_seqlen;

        memset(outT, 0, (size_t)value_dim * 16 * sizeof(float));
        __m512 _m = _mm512_set1_ps(-FLT_MAX);
        __m512 _l = _mm512_setzero_ps();
        const float* pM = maskT ? maskT + (size_t)n_begin * 16 : 0;
        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
            __m512 _block_max;
            float* scoreptr = scoreT;
            if (packed_key.empty())
            {
                const float* key = key_head.row(n);
                __m512 _max = _mm512_set1_ps(-FLT_MAX);
                int j = 0;
                for (; j + 7 < max_jj; j += 8)
                {
                    const float* pQ = queryT;
                    const float* pK0 = key + (size_t)j * head_dim;
                    const float* pK1 = pK0 + head_dim;
                    const float* pK2 = pK1 + head_dim;
                    const float* pK3 = pK2 + head_dim;
                    const float* pK4 = pK3 + head_dim;
                    const float* pK5 = pK4 + head_dim;
                    const float* pK6 = pK5 + head_dim;
                    const float* pK7 = pK6 + head_dim;
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
                        _sum0 = _mm512_loadu_ps(pM);
                        _sum1 = _mm512_loadu_ps(pM + 16);
                        _sum2 = _mm512_loadu_ps(pM + 32);
                        _sum3 = _mm512_loadu_ps(pM + 48);
                        _sum4 = _mm512_loadu_ps(pM + 64);
                        _sum5 = _mm512_loadu_ps(pM + 80);
                        _sum6 = _mm512_loadu_ps(pM + 96);
                        _sum7 = _mm512_loadu_ps(pM + 112);
                        pM += 128;
                    }
                    for (int d = 0; d < head_dim; d++)
                    {
                        __m512 _q = _mm512_loadu_ps(pQ);
                        _sum0 = _mm512_fmadd_ps(_q, _mm512_set1_ps(*pK0++), _sum0);
                        _sum1 = _mm512_fmadd_ps(_q, _mm512_set1_ps(*pK1++), _sum1);
                        _sum2 = _mm512_fmadd_ps(_q, _mm512_set1_ps(*pK2++), _sum2);
                        _sum3 = _mm512_fmadd_ps(_q, _mm512_set1_ps(*pK3++), _sum3);
                        _sum4 = _mm512_fmadd_ps(_q, _mm512_set1_ps(*pK4++), _sum4);
                        _sum5 = _mm512_fmadd_ps(_q, _mm512_set1_ps(*pK5++), _sum5);
                        _sum6 = _mm512_fmadd_ps(_q, _mm512_set1_ps(*pK6++), _sum6);
                        _sum7 = _mm512_fmadd_ps(_q, _mm512_set1_ps(*pK7++), _sum7);
                        pQ += 16;
                    }
                    _mm512_storeu_ps(scoreptr, _sum0);
                    _mm512_storeu_ps(scoreptr + 16, _sum1);
                    _mm512_storeu_ps(scoreptr + 32, _sum2);
                    _mm512_storeu_ps(scoreptr + 48, _sum3);
                    _mm512_storeu_ps(scoreptr + 64, _sum4);
                    _mm512_storeu_ps(scoreptr + 80, _sum5);
                    _mm512_storeu_ps(scoreptr + 96, _sum6);
                    _mm512_storeu_ps(scoreptr + 112, _sum7);
                    __m512 _max0 = _mm512_max_ps(_sum0, _sum4);
                    __m512 _max1 = _mm512_max_ps(_sum1, _sum5);
                    __m512 _max2 = _mm512_max_ps(_sum2, _sum6);
                    __m512 _max3 = _mm512_max_ps(_sum3, _sum7);
                    _max = _mm512_max_ps(_max, _mm512_max_ps(_mm512_max_ps(_max0, _max1), _mm512_max_ps(_max2, _max3)));
                    scoreptr += 128;
                }
                for (; j + 3 < max_jj; j += 4)
                {
                    const float* pQ = queryT;
                    const float* pK0 = key + (size_t)j * head_dim;
                    const float* pK1 = pK0 + head_dim;
                    const float* pK2 = pK1 + head_dim;
                    const float* pK3 = pK2 + head_dim;
                    __m512 _sum0 = _mm512_setzero_ps();
                    __m512 _sum1 = _mm512_setzero_ps();
                    __m512 _sum2 = _mm512_setzero_ps();
                    __m512 _sum3 = _mm512_setzero_ps();
                    if (pM)
                    {
                        _sum0 = _mm512_loadu_ps(pM);
                        _sum1 = _mm512_loadu_ps(pM + 16);
                        _sum2 = _mm512_loadu_ps(pM + 32);
                        _sum3 = _mm512_loadu_ps(pM + 48);
                        pM += 64;
                    }
                    for (int d = 0; d < head_dim; d++)
                    {
                        __m512 _q = _mm512_loadu_ps(pQ);
                        _sum0 = _mm512_fmadd_ps(_q, _mm512_set1_ps(*pK0++), _sum0);
                        _sum1 = _mm512_fmadd_ps(_q, _mm512_set1_ps(*pK1++), _sum1);
                        _sum2 = _mm512_fmadd_ps(_q, _mm512_set1_ps(*pK2++), _sum2);
                        _sum3 = _mm512_fmadd_ps(_q, _mm512_set1_ps(*pK3++), _sum3);
                        pQ += 16;
                    }
                    _mm512_storeu_ps(scoreptr, _sum0);
                    _mm512_storeu_ps(scoreptr + 16, _sum1);
                    _mm512_storeu_ps(scoreptr + 32, _sum2);
                    _mm512_storeu_ps(scoreptr + 48, _sum3);
                    __m512 _max01 = _mm512_max_ps(_sum0, _sum1);
                    __m512 _max23 = _mm512_max_ps(_sum2, _sum3);
                    _max = _mm512_max_ps(_max, _mm512_max_ps(_max01, _max23));
                    scoreptr += 64;
                }
                for (; j < max_jj; j++)
                {
                    const float* pQ = queryT;
                    const float* pK = key + (size_t)j * head_dim;
                    __m512 _sum = _mm512_setzero_ps();
                    if (pM)
                    {
                        _sum = _mm512_loadu_ps(pM);
                        pM += 16;
                    }
                    for (int d = 0; d < head_dim; d++)
                    {
                        _sum = _mm512_fmadd_ps(_mm512_loadu_ps(pQ), _mm512_set1_ps(pK[d]), _sum);
                        pQ += 16;
                    }
                    _max = _mm512_max_ps(_max, _sum);
                    _mm512_storeu_ps(scoreptr, _sum);
                    scoreptr += 16;
                }

                _block_max = _max;
            }
            else
            {
                const float* packed_key_tile = packed_key_head.row(n / block_n);
                const float* pK = packed_key_tile;
                __m512 _max = _mm512_set1_ps(-FLT_MAX);
                int j = 0;
                for (; j + 15 < max_jj; j += 16)
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
                        _sum0 = _mm512_loadu_ps(pM);
                        _sum1 = _mm512_loadu_ps(pM + 16);
                        _sum2 = _mm512_loadu_ps(pM + 32);
                        _sum3 = _mm512_loadu_ps(pM + 48);
                        _sum4 = _mm512_loadu_ps(pM + 64);
                        _sum5 = _mm512_loadu_ps(pM + 80);
                        _sum6 = _mm512_loadu_ps(pM + 96);
                        _sum7 = _mm512_loadu_ps(pM + 112);
                        _sum8 = _mm512_loadu_ps(pM + 128);
                        _sum9 = _mm512_loadu_ps(pM + 144);
                        _suma = _mm512_loadu_ps(pM + 160);
                        _sumb = _mm512_loadu_ps(pM + 176);
                        _sumc = _mm512_loadu_ps(pM + 192);
                        _sumd = _mm512_loadu_ps(pM + 208);
                        _sume = _mm512_loadu_ps(pM + 224);
                        _sumf = _mm512_loadu_ps(pM + 240);
                        pM += 256;
                    }
                    const float* pQ = queryT;
                    for (int d = 0; d < head_dim; d++)
                    {
                        __m512 _q = _mm512_loadu_ps(pQ);
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
                        pQ += 16;
                        pK += 16;
                    }
                    _mm512_storeu_ps(scoreptr, _sum0);
                    _mm512_storeu_ps(scoreptr + 16, _sum1);
                    _mm512_storeu_ps(scoreptr + 32, _sum2);
                    _mm512_storeu_ps(scoreptr + 48, _sum3);
                    _mm512_storeu_ps(scoreptr + 64, _sum4);
                    _mm512_storeu_ps(scoreptr + 80, _sum5);
                    _mm512_storeu_ps(scoreptr + 96, _sum6);
                    _mm512_storeu_ps(scoreptr + 112, _sum7);
                    _mm512_storeu_ps(scoreptr + 128, _sum8);
                    _mm512_storeu_ps(scoreptr + 144, _sum9);
                    _mm512_storeu_ps(scoreptr + 160, _suma);
                    _mm512_storeu_ps(scoreptr + 176, _sumb);
                    _mm512_storeu_ps(scoreptr + 192, _sumc);
                    _mm512_storeu_ps(scoreptr + 208, _sumd);
                    _mm512_storeu_ps(scoreptr + 224, _sume);
                    _mm512_storeu_ps(scoreptr + 240, _sumf);
                    __m512 _max0 = _mm512_max_ps(_mm512_max_ps(_sum0, _sum4), _mm512_max_ps(_sum8, _sumc));
                    __m512 _max1 = _mm512_max_ps(_mm512_max_ps(_sum1, _sum5), _mm512_max_ps(_sum9, _sumd));
                    __m512 _max2 = _mm512_max_ps(_mm512_max_ps(_sum2, _sum6), _mm512_max_ps(_suma, _sume));
                    __m512 _max3 = _mm512_max_ps(_mm512_max_ps(_sum3, _sum7), _mm512_max_ps(_sumb, _sumf));
                    _max = _mm512_max_ps(_max, _mm512_max_ps(_mm512_max_ps(_max0, _max1), _mm512_max_ps(_max2, _max3)));
                    scoreptr += 256;
                }
                for (; j + 7 < max_jj; j += 8)
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
                        _sum0 = _mm512_loadu_ps(pM);
                        _sum1 = _mm512_loadu_ps(pM + 16);
                        _sum2 = _mm512_loadu_ps(pM + 32);
                        _sum3 = _mm512_loadu_ps(pM + 48);
                        _sum4 = _mm512_loadu_ps(pM + 64);
                        _sum5 = _mm512_loadu_ps(pM + 80);
                        _sum6 = _mm512_loadu_ps(pM + 96);
                        _sum7 = _mm512_loadu_ps(pM + 112);
                        pM += 128;
                    }
                    const float* pQ = queryT;
                    for (int d = 0; d < head_dim; d++)
                    {
                        __m512 _q = _mm512_loadu_ps(pQ);
                        _sum0 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[3]), _sum3);
                        _sum4 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[4]), _sum4);
                        _sum5 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[5]), _sum5);
                        _sum6 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[6]), _sum6);
                        _sum7 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[7]), _sum7);
                        pQ += 16;
                        pK += 8;
                    }
                    _mm512_storeu_ps(scoreptr, _sum0);
                    _mm512_storeu_ps(scoreptr + 16, _sum1);
                    _mm512_storeu_ps(scoreptr + 32, _sum2);
                    _mm512_storeu_ps(scoreptr + 48, _sum3);
                    _mm512_storeu_ps(scoreptr + 64, _sum4);
                    _mm512_storeu_ps(scoreptr + 80, _sum5);
                    _mm512_storeu_ps(scoreptr + 96, _sum6);
                    _mm512_storeu_ps(scoreptr + 112, _sum7);
                    __m512 _max0 = _mm512_max_ps(_sum0, _sum4);
                    __m512 _max1 = _mm512_max_ps(_sum1, _sum5);
                    __m512 _max2 = _mm512_max_ps(_sum2, _sum6);
                    __m512 _max3 = _mm512_max_ps(_sum3, _sum7);
                    _max = _mm512_max_ps(_max, _mm512_max_ps(_mm512_max_ps(_max0, _max1), _mm512_max_ps(_max2, _max3)));
                    scoreptr += 128;
                }
                for (; j + 3 < max_jj; j += 4)
                {
                    __m512 _sum0 = _mm512_setzero_ps();
                    __m512 _sum1 = _mm512_setzero_ps();
                    __m512 _sum2 = _mm512_setzero_ps();
                    __m512 _sum3 = _mm512_setzero_ps();
                    if (pM)
                    {
                        _sum0 = _mm512_loadu_ps(pM);
                        _sum1 = _mm512_loadu_ps(pM + 16);
                        _sum2 = _mm512_loadu_ps(pM + 32);
                        _sum3 = _mm512_loadu_ps(pM + 48);
                        pM += 64;
                    }
                    const float* pQ = queryT;
                    for (int d = 0; d < head_dim; d++)
                    {
                        __m512 _q = _mm512_loadu_ps(pQ);
                        _sum0 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_q, _mm512_set1_ps(pK[3]), _sum3);
                        pQ += 16;
                        pK += 4;
                    }
                    _mm512_storeu_ps(scoreptr, _sum0);
                    _mm512_storeu_ps(scoreptr + 16, _sum1);
                    _mm512_storeu_ps(scoreptr + 32, _sum2);
                    _mm512_storeu_ps(scoreptr + 48, _sum3);
                    __m512 _max01 = _mm512_max_ps(_sum0, _sum1);
                    __m512 _max23 = _mm512_max_ps(_sum2, _sum3);
                    _max = _mm512_max_ps(_max, _mm512_max_ps(_max01, _max23));
                    scoreptr += 64;
                }
                for (; j < max_jj; j++)
                {
                    const float* pQ = queryT;
                    __m512 _sum = _mm512_setzero_ps();
                    if (pM)
                    {
                        _sum = _mm512_loadu_ps(pM);
                        pM += 16;
                    }
                    for (int d = 0; d < head_dim; d++)
                    {
                        _sum = _mm512_fmadd_ps(_mm512_loadu_ps(pQ), _mm512_set1_ps(pK[d]), _sum);
                        pQ += 16;
                    }
                    pK += head_dim;
                    _max = _mm512_max_ps(_max, _sum);
                    _mm512_storeu_ps(scoreptr, _sum);
                    scoreptr += 16;
                }

                _block_max = _max;
            }
            __m512 _m_new = _mm512_max_ps(_m, _block_max);
            const __mmask16 alpha_active = _mm512_cmp_ps_mask(_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
            __m512 _alpha = exp512_ps(_mm512_maskz_sub_ps(alpha_active, _m, _m_new));
            _alpha = _mm512_maskz_mov_ps(alpha_active, _alpha);

            scoreptr = scoreT;
            __m512 _sum0 = _mm512_setzero_ps();
            __m512 _sum1 = _mm512_setzero_ps();
            __m512 _sum2 = _mm512_setzero_ps();
            __m512 _sum3 = _mm512_setzero_ps();
            int j = 0;
            for (; j + 3 < max_jj; j += 4)
            {
                __m512 _score = _mm512_sub_ps(_mm512_loadu_ps(scoreptr), _m_new);
                __m512 _p = exp512_ps(_score);
                _mm512_storeu_ps(scoreptr, _p);
                _sum0 = _mm512_add_ps(_sum0, _p);
                _score = _mm512_sub_ps(_mm512_loadu_ps(scoreptr + 16), _m_new);
                _p = exp512_ps(_score);
                _mm512_storeu_ps(scoreptr + 16, _p);
                _sum1 = _mm512_add_ps(_sum1, _p);
                _score = _mm512_sub_ps(_mm512_loadu_ps(scoreptr + 32), _m_new);
                _p = exp512_ps(_score);
                _mm512_storeu_ps(scoreptr + 32, _p);
                _sum2 = _mm512_add_ps(_sum2, _p);
                _score = _mm512_sub_ps(_mm512_loadu_ps(scoreptr + 48), _m_new);
                _p = exp512_ps(_score);
                _mm512_storeu_ps(scoreptr + 48, _p);
                _sum3 = _mm512_add_ps(_sum3, _p);
                scoreptr += 64;
            }
            for (; j < max_jj; j++)
            {
                __m512 _score = _mm512_sub_ps(_mm512_loadu_ps(scoreptr), _m_new);
                __m512 _p = exp512_ps(_score);
                _mm512_storeu_ps(scoreptr, _p);
                scoreptr += 16;
                _sum0 = _mm512_add_ps(_sum0, _p);
            }
            __m512 _sum = _mm512_add_ps(_mm512_add_ps(_sum0, _sum1), _mm512_add_ps(_sum2, _sum3));
            _l = _mm512_add_ps(_mm512_mul_ps(_l, _alpha), _sum);
            _m = _m_new;
            float* outptr = outT;
            if (packed_value.empty())
            {
                const float* value = value_head.row(n);
                const float* valueptr = value;
                int d = 0;
                for (; d + 15 < value_dim; d += 16)
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
                        _out8 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[8]), _out8);
                        _out9 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[9]), _out9);
                        _outa = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[10]), _outa);
                        _outb = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[11]), _outb);
                        _outc = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[12]), _outc);
                        _outd = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[13]), _outd);
                        _oute = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[14]), _oute);
                        _outf = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[15]), _outf);
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
                    _mm512_storeu_ps(outptr + 128, _out8);
                    _mm512_storeu_ps(outptr + 144, _out9);
                    _mm512_storeu_ps(outptr + 160, _outa);
                    _mm512_storeu_ps(outptr + 176, _outb);
                    _mm512_storeu_ps(outptr + 192, _outc);
                    _mm512_storeu_ps(outptr + 208, _outd);
                    _mm512_storeu_ps(outptr + 224, _oute);
                    _mm512_storeu_ps(outptr + 240, _outf);
                    outptr += 256;
                    valueptr += 16;
                }
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
                for (; d + 3 < value_dim; d += 4)
                {
                    __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr), _alpha);
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
            else
            {
                const float* packed_value_tile = packed_value_head.row(n / block_n);
                const float* pV = packed_value_tile;
                int d = 0;
                for (; d + 15 < value_dim; d += 16)
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
                        pV += 8;
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
                for (; d + 3 < value_dim; d += 4)
                {
                    __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(outptr), _alpha);
                    __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 16), _alpha);
                    __m512 _out2 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 32), _alpha);
                    __m512 _out3 = _mm512_mul_ps(_mm512_loadu_ps(outptr + 48), _alpha);
                    const float* pS = scoreT;
                    for (int j = 0; j < max_jj; j++)
                    {
                        __m512 _p = _mm512_loadu_ps(pS);
                        _out0 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[0]), _out0);
                        _out1 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[1]), _out1);
                        _out2 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[2]), _out2);
                        _out3 = _mm512_fmadd_ps(_p, _mm512_set1_ps(pV[3]), _out3);
                        pS += 16;
                        pV += 4;
                    }
                    _mm512_storeu_ps(outptr, _out0);
                    _mm512_storeu_ps(outptr + 16, _out1);
                    _mm512_storeu_ps(outptr + 32, _out2);
                    _mm512_storeu_ps(outptr + 48, _out3);
                    outptr += 64;
                }
                for (; d < value_dim; d++)
                {
                    __m512 _out = _mm512_mul_ps(_mm512_loadu_ps(outptr), _alpha);
                    const float* pS = scoreT;
                    for (int j = 0; j < max_jj; j++)
                    {
                        _out = _mm512_fmadd_ps(_mm512_loadu_ps(pS), _mm512_set1_ps(*pV++), _out);
                        pS += 16;
                    }
                    _mm512_storeu_ps(outptr, _out);
                    outptr += 16;
                }
            }
        }

        sdpa_store_output_tile16(outT_tile, top_blob_head, i0x, state_ptr, state_stride, _m, _l, value_dim);
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const int i0x = i0 + ii;
        float* state_ptr = state.empty() ? 0 : state_base + ii;
        float* scoreT = workspace_ptr;
        Mat outT_tile = workspace.range(block_n * 8, value_dim * 8);
        float* outT = outT_tile;
        const float* queryT = queryT_base + ii * head_dim;
        const Mat key_head = key.channel(g);
        const Mat packed_key_head = packed_key.empty() ? Mat() : packed_key.channel(g);
        const Mat value_head = value.channel(g);
        const Mat packed_value_head = packed_value.empty() ? Mat() : packed_value.channel(g);
        const float* maskT = packed_mask.empty() ? 0 : (const float*)packed_mask + (size_t)ii * key_seqlen;

        memset(outT, 0, (size_t)value_dim * 8 * sizeof(float));
        __m256 _m = _mm256_set1_ps(-FLT_MAX);
        __m256 _l = _mm256_setzero_ps();
        const float* pM = maskT ? maskT + (size_t)n_begin * 8 : 0;

        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
            __m256 _block_max;
            float* scoreptr = scoreT;
            if (packed_key.empty())
            {
                const float* key = key_head.row(n);
                __m256 _max = _mm256_set1_ps(-FLT_MAX);
                int j = 0;
                for (; j + 7 < max_jj; j += 8)
                {
                    const float* pQ = queryT;
                    const float* pK0 = key + (size_t)j * head_dim;
                    const float* pK1 = pK0 + head_dim;
                    const float* pK2 = pK1 + head_dim;
                    const float* pK3 = pK2 + head_dim;
                    const float* pK4 = pK3 + head_dim;
                    const float* pK5 = pK4 + head_dim;
                    const float* pK6 = pK5 + head_dim;
                    const float* pK7 = pK6 + head_dim;
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
                        _sum0 = _mm256_loadu_ps(pM);
                        _sum1 = _mm256_loadu_ps(pM + 8);
                        _sum2 = _mm256_loadu_ps(pM + 16);
                        _sum3 = _mm256_loadu_ps(pM + 24);
                        _sum4 = _mm256_loadu_ps(pM + 32);
                        _sum5 = _mm256_loadu_ps(pM + 40);
                        _sum6 = _mm256_loadu_ps(pM + 48);
                        _sum7 = _mm256_loadu_ps(pM + 56);
                        pM += 64;
                    }
                    for (int d = 0; d < head_dim; d++)
                    {
                        __m256 _q = _mm256_loadu_ps(pQ);
                        _sum0 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(*pK0++), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(*pK1++), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(*pK2++), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(*pK3++), _sum3);
                        _sum4 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(*pK4++), _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(*pK5++), _sum5);
                        _sum6 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(*pK6++), _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(*pK7++), _sum7);
                        pQ += 8;
                    }
                    _mm256_storeu_ps(scoreptr, _sum0);
                    _mm256_storeu_ps(scoreptr + 8, _sum1);
                    _mm256_storeu_ps(scoreptr + 16, _sum2);
                    _mm256_storeu_ps(scoreptr + 24, _sum3);
                    _mm256_storeu_ps(scoreptr + 32, _sum4);
                    _mm256_storeu_ps(scoreptr + 40, _sum5);
                    _mm256_storeu_ps(scoreptr + 48, _sum6);
                    _mm256_storeu_ps(scoreptr + 56, _sum7);
                    __m256 _max0 = _mm256_max_ps(_sum0, _sum4);
                    __m256 _max1 = _mm256_max_ps(_sum1, _sum5);
                    __m256 _max2 = _mm256_max_ps(_sum2, _sum6);
                    __m256 _max3 = _mm256_max_ps(_sum3, _sum7);
                    _max = _mm256_max_ps(_max, _mm256_max_ps(_mm256_max_ps(_max0, _max1), _mm256_max_ps(_max2, _max3)));
                    scoreptr += 64;
                }
                for (; j + 3 < max_jj; j += 4)
                {
                    const float* pQ = queryT;
                    const float* pK0 = key + (size_t)j * head_dim;
                    const float* pK1 = pK0 + head_dim;
                    const float* pK2 = pK1 + head_dim;
                    const float* pK3 = pK2 + head_dim;
                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    if (pM)
                    {
                        _sum0 = _mm256_loadu_ps(pM);
                        _sum1 = _mm256_loadu_ps(pM + 8);
                        _sum2 = _mm256_loadu_ps(pM + 16);
                        _sum3 = _mm256_loadu_ps(pM + 24);
                        pM += 32;
                    }
                    for (int d = 0; d < head_dim; d++)
                    {
                        __m256 _q = _mm256_loadu_ps(pQ);
                        _sum0 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(*pK0++), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(*pK1++), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(*pK2++), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(*pK3++), _sum3);
                        pQ += 8;
                    }
                    _mm256_storeu_ps(scoreptr, _sum0);
                    _mm256_storeu_ps(scoreptr + 8, _sum1);
                    _mm256_storeu_ps(scoreptr + 16, _sum2);
                    _mm256_storeu_ps(scoreptr + 24, _sum3);
                    __m256 _max01 = _mm256_max_ps(_sum0, _sum1);
                    __m256 _max23 = _mm256_max_ps(_sum2, _sum3);
                    _max = _mm256_max_ps(_max, _mm256_max_ps(_max01, _max23));
                    scoreptr += 32;
                }
                for (; j < max_jj; j++)
                {
                    const float* pQ = queryT;
                    const float* pK = key + (size_t)j * head_dim;
                    __m256 _sum = _mm256_setzero_ps();
                    if (pM)
                    {
                        _sum = _mm256_loadu_ps(pM);
                        pM += 8;
                    }
                    for (int d = 0; d < head_dim; d++)
                    {
                        _sum = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pQ), _mm256_set1_ps(pK[d]), _sum);
                        pQ += 8;
                    }
                    _max = _mm256_max_ps(_max, _sum);
                    _mm256_storeu_ps(scoreptr, _sum);
                    scoreptr += 8;
                }

                _block_max = _max;
            }
            else
            {
                const float* packed_key_tile = packed_key_head.row(n / block_n);
                const float* pK = packed_key_tile;
                __m256 _max = _mm256_set1_ps(-FLT_MAX);
                int j = 0;
#if __AVX512F__
                for (; j + 15 < max_jj; j += 16)
                {
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
                    if (pM)
                    {
                        _sum0 = _mm256_loadu_ps(pM);
                        _sum1 = _mm256_loadu_ps(pM + 8);
                        _sum2 = _mm256_loadu_ps(pM + 16);
                        _sum3 = _mm256_loadu_ps(pM + 24);
                        _sum4 = _mm256_loadu_ps(pM + 32);
                        _sum5 = _mm256_loadu_ps(pM + 40);
                        _sum6 = _mm256_loadu_ps(pM + 48);
                        _sum7 = _mm256_loadu_ps(pM + 56);
                        _sum8 = _mm256_loadu_ps(pM + 64);
                        _sum9 = _mm256_loadu_ps(pM + 72);
                        _suma = _mm256_loadu_ps(pM + 80);
                        _sumb = _mm256_loadu_ps(pM + 88);
                        _sumc = _mm256_loadu_ps(pM + 96);
                        _sumd = _mm256_loadu_ps(pM + 104);
                        _sume = _mm256_loadu_ps(pM + 112);
                        _sumf = _mm256_loadu_ps(pM + 120);
                        pM += 128;
                    }
                    const float* pQ = queryT;
                    for (int d = 0; d < head_dim; d++)
                    {
                        __m256 _q = _mm256_loadu_ps(pQ);
                        _sum0 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[1]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[2]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[3]), _sum3);
                        _sum4 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[4]), _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[5]), _sum5);
                        _sum6 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[6]), _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[7]), _sum7);
                        _sum8 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[8]), _sum8);
                        _sum9 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[9]), _sum9);
                        _suma = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[10]), _suma);
                        _sumb = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[11]), _sumb);
                        _sumc = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[12]), _sumc);
                        _sumd = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[13]), _sumd);
                        _sume = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[14]), _sume);
                        _sumf = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[15]), _sumf);
                        pQ += 8;
                        pK += 16;
                    }
                    _mm256_storeu_ps(scoreptr, _sum0);
                    _mm256_storeu_ps(scoreptr + 8, _sum1);
                    _mm256_storeu_ps(scoreptr + 16, _sum2);
                    _mm256_storeu_ps(scoreptr + 24, _sum3);
                    _mm256_storeu_ps(scoreptr + 32, _sum4);
                    _mm256_storeu_ps(scoreptr + 40, _sum5);
                    _mm256_storeu_ps(scoreptr + 48, _sum6);
                    _mm256_storeu_ps(scoreptr + 56, _sum7);
                    _mm256_storeu_ps(scoreptr + 64, _sum8);
                    _mm256_storeu_ps(scoreptr + 72, _sum9);
                    _mm256_storeu_ps(scoreptr + 80, _suma);
                    _mm256_storeu_ps(scoreptr + 88, _sumb);
                    _mm256_storeu_ps(scoreptr + 96, _sumc);
                    _mm256_storeu_ps(scoreptr + 104, _sumd);
                    _mm256_storeu_ps(scoreptr + 112, _sume);
                    _mm256_storeu_ps(scoreptr + 120, _sumf);
                    __m256 _max0 = _mm256_max_ps(_mm256_max_ps(_sum0, _sum4), _mm256_max_ps(_sum8, _sumc));
                    __m256 _max1 = _mm256_max_ps(_mm256_max_ps(_sum1, _sum5), _mm256_max_ps(_sum9, _sumd));
                    __m256 _max2 = _mm256_max_ps(_mm256_max_ps(_sum2, _sum6), _mm256_max_ps(_suma, _sume));
                    __m256 _max3 = _mm256_max_ps(_mm256_max_ps(_sum3, _sum7), _mm256_max_ps(_sumb, _sumf));
                    _max = _mm256_max_ps(_max, _mm256_max_ps(_mm256_max_ps(_max0, _max1), _mm256_max_ps(_max2, _max3)));
                    scoreptr += 128;
                }
#endif // __AVX512F__
                for (; j + 7 < max_jj; j += 8)
                {
                    const float* pQ = queryT;
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
                        _sum0 = _mm256_loadu_ps(pM);
                        _sum1 = _mm256_loadu_ps(pM + 8);
                        _sum2 = _mm256_loadu_ps(pM + 16);
                        _sum3 = _mm256_loadu_ps(pM + 24);
                        _sum4 = _mm256_loadu_ps(pM + 32);
                        _sum5 = _mm256_loadu_ps(pM + 40);
                        _sum6 = _mm256_loadu_ps(pM + 48);
                        _sum7 = _mm256_loadu_ps(pM + 56);
                        pM += 64;
                    }
                    for (int d = 0; d < head_dim; d++)
                    {
                        __m256 _q = _mm256_loadu_ps(pQ);
                        _sum0 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[1]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[2]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[3]), _sum3);
                        _sum4 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[4]), _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[5]), _sum5);
                        _sum6 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[6]), _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[7]), _sum7);
                        pQ += 8;
                        pK += 8;
                    }
                    _mm256_storeu_ps(scoreptr, _sum0);
                    _mm256_storeu_ps(scoreptr + 8, _sum1);
                    _mm256_storeu_ps(scoreptr + 16, _sum2);
                    _mm256_storeu_ps(scoreptr + 24, _sum3);
                    _mm256_storeu_ps(scoreptr + 32, _sum4);
                    _mm256_storeu_ps(scoreptr + 40, _sum5);
                    _mm256_storeu_ps(scoreptr + 48, _sum6);
                    _mm256_storeu_ps(scoreptr + 56, _sum7);
                    __m256 _max0 = _mm256_max_ps(_sum0, _sum4);
                    __m256 _max1 = _mm256_max_ps(_sum1, _sum5);
                    __m256 _max2 = _mm256_max_ps(_sum2, _sum6);
                    __m256 _max3 = _mm256_max_ps(_sum3, _sum7);
                    _max = _mm256_max_ps(_max, _mm256_max_ps(_mm256_max_ps(_max0, _max1), _mm256_max_ps(_max2, _max3)));
                    scoreptr += 64;
                }
                for (; j + 3 < max_jj; j += 4)
                {
                    const float* pQ = queryT;
                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    if (pM)
                    {
                        _sum0 = _mm256_loadu_ps(pM);
                        _sum1 = _mm256_loadu_ps(pM + 8);
                        _sum2 = _mm256_loadu_ps(pM + 16);
                        _sum3 = _mm256_loadu_ps(pM + 24);
                        pM += 32;
                    }
                    for (int d = 0; d < head_dim; d++)
                    {
                        __m256 _q = _mm256_loadu_ps(pQ);
                        _sum0 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[1]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[2]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_q, _mm256_set1_ps(pK[3]), _sum3);
                        pQ += 8;
                        pK += 4;
                    }
                    _mm256_storeu_ps(scoreptr, _sum0);
                    _mm256_storeu_ps(scoreptr + 8, _sum1);
                    _mm256_storeu_ps(scoreptr + 16, _sum2);
                    _mm256_storeu_ps(scoreptr + 24, _sum3);
                    __m256 _max01 = _mm256_max_ps(_sum0, _sum1);
                    __m256 _max23 = _mm256_max_ps(_sum2, _sum3);
                    _max = _mm256_max_ps(_max, _mm256_max_ps(_max01, _max23));
                    scoreptr += 32;
                }
                for (; j < max_jj; j++)
                {
                    const float* pQ = queryT;
                    __m256 _sum = _mm256_setzero_ps();
                    if (pM)
                    {
                        _sum = _mm256_loadu_ps(pM);
                        pM += 8;
                    }
                    for (int d = 0; d < head_dim; d++)
                    {
                        _sum = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pQ), _mm256_set1_ps(pK[d]), _sum);
                        pQ += 8;
                    }
                    pK += head_dim;
                    _max = _mm256_max_ps(_max, _sum);
                    _mm256_storeu_ps(scoreptr, _sum);
                    scoreptr += 8;
                }

                _block_max = _max;
            }
            __m256 _m_new = _mm256_max_ps(_m, _block_max);
            const __m256 _alpha_active = _mm256_cmp_ps(_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
            __m256 _alpha = exp256_ps(_mm256_and_ps(_alpha_active, _mm256_sub_ps(_m, _m_new)));
            _alpha = _mm256_and_ps(_alpha, _alpha_active);

            scoreptr = scoreT;
            __m256 _sum0 = _mm256_setzero_ps();
            __m256 _sum1 = _mm256_setzero_ps();
            __m256 _sum2 = _mm256_setzero_ps();
            __m256 _sum3 = _mm256_setzero_ps();
            int j = 0;
            for (; j + 3 < max_jj; j += 4)
            {
                __m256 _score = _mm256_sub_ps(_mm256_loadu_ps(scoreptr), _m_new);
                __m256 _p = exp256_ps(_score);
                _mm256_storeu_ps(scoreptr, _p);
                _sum0 = _mm256_add_ps(_sum0, _p);
                _score = _mm256_sub_ps(_mm256_loadu_ps(scoreptr + 8), _m_new);
                _p = exp256_ps(_score);
                _mm256_storeu_ps(scoreptr + 8, _p);
                _sum1 = _mm256_add_ps(_sum1, _p);
                _score = _mm256_sub_ps(_mm256_loadu_ps(scoreptr + 16), _m_new);
                _p = exp256_ps(_score);
                _mm256_storeu_ps(scoreptr + 16, _p);
                _sum2 = _mm256_add_ps(_sum2, _p);
                _score = _mm256_sub_ps(_mm256_loadu_ps(scoreptr + 24), _m_new);
                _p = exp256_ps(_score);
                _mm256_storeu_ps(scoreptr + 24, _p);
                _sum3 = _mm256_add_ps(_sum3, _p);
                scoreptr += 32;
            }
            for (; j < max_jj; j++)
            {
                __m256 _score = _mm256_sub_ps(_mm256_loadu_ps(scoreptr), _m_new);
                __m256 _p = exp256_ps(_score);
                _mm256_storeu_ps(scoreptr, _p);
                scoreptr += 8;
                _sum0 = _mm256_add_ps(_sum0, _p);
            }
            __m256 _sum = _mm256_add_ps(_mm256_add_ps(_sum0, _sum1), _mm256_add_ps(_sum2, _sum3));
            _l = _mm256_add_ps(_mm256_mul_ps(_l, _alpha), _sum);
            _m = _m_new;
            float* outptr = outT;
            if (packed_value.empty())
            {
                const float* value = value_head.row(n);
                const float* valueptr = value;
                int d = 0;
#if __AVX512F__
                for (; d + 15 < value_dim; d += 16)
                {
                    __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr), _alpha);
                    __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 8), _alpha);
                    __m256 _out2 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 16), _alpha);
                    __m256 _out3 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 24), _alpha);
                    __m256 _out4 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 32), _alpha);
                    __m256 _out5 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 40), _alpha);
                    __m256 _out6 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 48), _alpha);
                    __m256 _out7 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 56), _alpha);
                    __m256 _out8 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 64), _alpha);
                    __m256 _out9 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 72), _alpha);
                    __m256 _outa = _mm256_mul_ps(_mm256_loadu_ps(outptr + 80), _alpha);
                    __m256 _outb = _mm256_mul_ps(_mm256_loadu_ps(outptr + 88), _alpha);
                    __m256 _outc = _mm256_mul_ps(_mm256_loadu_ps(outptr + 96), _alpha);
                    __m256 _outd = _mm256_mul_ps(_mm256_loadu_ps(outptr + 104), _alpha);
                    __m256 _oute = _mm256_mul_ps(_mm256_loadu_ps(outptr + 112), _alpha);
                    __m256 _outf = _mm256_mul_ps(_mm256_loadu_ps(outptr + 120), _alpha);
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
                        _out8 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[8]), _out8);
                        _out9 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[9]), _out9);
                        _outa = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[10]), _outa);
                        _outb = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[11]), _outb);
                        _outc = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[12]), _outc);
                        _outd = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[13]), _outd);
                        _oute = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[14]), _oute);
                        _outf = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[15]), _outf);
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
                    _mm256_storeu_ps(outptr + 64, _out8);
                    _mm256_storeu_ps(outptr + 72, _out9);
                    _mm256_storeu_ps(outptr + 80, _outa);
                    _mm256_storeu_ps(outptr + 88, _outb);
                    _mm256_storeu_ps(outptr + 96, _outc);
                    _mm256_storeu_ps(outptr + 104, _outd);
                    _mm256_storeu_ps(outptr + 112, _oute);
                    _mm256_storeu_ps(outptr + 120, _outf);
                    outptr += 128;
                    valueptr += 16;
                }
#endif // __AVX512F__
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
                for (; d + 3 < value_dim; d += 4)
                {
                    __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr), _alpha);
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
            else
            {
                const float* packed_value_tile = packed_value_head.row(n / block_n);
                const float* pV = packed_value_tile;
                int d = 0;
#if __AVX512F__
                for (; d + 15 < value_dim; d += 16)
                {
                    __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr), _alpha);
                    __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 8), _alpha);
                    __m256 _out2 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 16), _alpha);
                    __m256 _out3 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 24), _alpha);
                    __m256 _out4 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 32), _alpha);
                    __m256 _out5 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 40), _alpha);
                    __m256 _out6 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 48), _alpha);
                    __m256 _out7 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 56), _alpha);
                    __m256 _out8 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 64), _alpha);
                    __m256 _out9 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 72), _alpha);
                    __m256 _outa = _mm256_mul_ps(_mm256_loadu_ps(outptr + 80), _alpha);
                    __m256 _outb = _mm256_mul_ps(_mm256_loadu_ps(outptr + 88), _alpha);
                    __m256 _outc = _mm256_mul_ps(_mm256_loadu_ps(outptr + 96), _alpha);
                    __m256 _outd = _mm256_mul_ps(_mm256_loadu_ps(outptr + 104), _alpha);
                    __m256 _oute = _mm256_mul_ps(_mm256_loadu_ps(outptr + 112), _alpha);
                    __m256 _outf = _mm256_mul_ps(_mm256_loadu_ps(outptr + 120), _alpha);
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
                        _out8 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[8]), _out8);
                        _out9 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[9]), _out9);
                        _outa = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[10]), _outa);
                        _outb = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[11]), _outb);
                        _outc = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[12]), _outc);
                        _outd = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[13]), _outd);
                        _oute = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[14]), _oute);
                        _outf = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[15]), _outf);
                        pS += 8;
                        pV += 16;
                    }
                    _mm256_storeu_ps(outptr, _out0);
                    _mm256_storeu_ps(outptr + 8, _out1);
                    _mm256_storeu_ps(outptr + 16, _out2);
                    _mm256_storeu_ps(outptr + 24, _out3);
                    _mm256_storeu_ps(outptr + 32, _out4);
                    _mm256_storeu_ps(outptr + 40, _out5);
                    _mm256_storeu_ps(outptr + 48, _out6);
                    _mm256_storeu_ps(outptr + 56, _out7);
                    _mm256_storeu_ps(outptr + 64, _out8);
                    _mm256_storeu_ps(outptr + 72, _out9);
                    _mm256_storeu_ps(outptr + 80, _outa);
                    _mm256_storeu_ps(outptr + 88, _outb);
                    _mm256_storeu_ps(outptr + 96, _outc);
                    _mm256_storeu_ps(outptr + 104, _outd);
                    _mm256_storeu_ps(outptr + 112, _oute);
                    _mm256_storeu_ps(outptr + 120, _outf);
                    outptr += 128;
                }
#endif // __AVX512F__
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
                        pV += 8;
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
                for (; d + 3 < value_dim; d += 4)
                {
                    __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(outptr), _alpha);
                    __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 8), _alpha);
                    __m256 _out2 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 16), _alpha);
                    __m256 _out3 = _mm256_mul_ps(_mm256_loadu_ps(outptr + 24), _alpha);
                    const float* pS = scoreT;
                    for (int j = 0; j < max_jj; j++)
                    {
                        __m256 _p = _mm256_loadu_ps(pS);
                        _out0 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[0]), _out0);
                        _out1 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[1]), _out1);
                        _out2 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[2]), _out2);
                        _out3 = _mm256_comp_fmadd_ps(_p, _mm256_set1_ps(pV[3]), _out3);
                        pS += 8;
                        pV += 4;
                    }
                    _mm256_storeu_ps(outptr, _out0);
                    _mm256_storeu_ps(outptr + 8, _out1);
                    _mm256_storeu_ps(outptr + 16, _out2);
                    _mm256_storeu_ps(outptr + 24, _out3);
                    outptr += 32;
                }
                for (; d < value_dim; d++)
                {
                    __m256 _out = _mm256_mul_ps(_mm256_loadu_ps(outptr), _alpha);
                    const float* pS = scoreT;
                    for (int j = 0; j < max_jj; j++)
                    {
                        _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pS), _mm256_set1_ps(*pV++), _out);
                        pS += 8;
                    }
                    _mm256_storeu_ps(outptr, _out);
                    outptr += 8;
                }
            }
        }

        sdpa_store_output_tile8(outT_tile, top_blob_head, i0x, state_ptr, state_stride, _m, _l, value_dim);
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const int i0x = i0 + ii;
        float* state_ptr = state.empty() ? 0 : state_base + ii;
        float* scoreT = workspace_ptr;
        Mat outT_tile = workspace.range(block_n * 4, value_dim * 4);
        float* outT = outT_tile;
        const float* queryT = queryT_base + ii * head_dim;
        const Mat key_head = key.channel(g);
        const Mat packed_key_head = packed_key.empty() ? Mat() : packed_key.channel(g);
        const Mat value_head = value.channel(g);
        const Mat packed_value_head = packed_value.empty() ? Mat() : packed_value.channel(g);
        const float* maskT = packed_mask.empty() ? 0 : (const float*)packed_mask + (size_t)ii * key_seqlen;

        memset(outT, 0, (size_t)value_dim * 4 * sizeof(float));

        __m128 _m = _mm_set1_ps(-FLT_MAX);
        __m128 _l = _mm_setzero_ps();
        const float* pM = maskT ? maskT + (size_t)n_begin * 4 : 0;

        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
            __m128 _block_max;
            float* scoreptr = scoreT;
            if (packed_key.empty())
            {
                const float* key = key_head.row(n);
                __m128 _max = _mm_set1_ps(-FLT_MAX);
                int j = 0;
                for (; j + 3 < max_jj; j += 4)
                {
                    const float* pQ = queryT;
                    const float* pK0 = key + (size_t)j * head_dim;
                    const float* pK1 = pK0 + head_dim;
                    const float* pK2 = pK1 + head_dim;
                    const float* pK3 = pK2 + head_dim;
                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();
                    if (pM)
                    {
                        _sum0 = _mm_loadu_ps(pM);
                        _sum1 = _mm_loadu_ps(pM + 4);
                        _sum2 = _mm_loadu_ps(pM + 8);
                        _sum3 = _mm_loadu_ps(pM + 12);
                        pM += 16;
                    }
                    for (int d = 0; d < head_dim; d++)
                    {
                        __m128 _q = _mm_loadu_ps(pQ);
                        _sum0 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(*pK0++), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(*pK1++), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(*pK2++), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(*pK3++), _sum3);
                        pQ += 4;
                    }
                    _mm_storeu_ps(scoreptr, _sum0);
                    _mm_storeu_ps(scoreptr + 4, _sum1);
                    _mm_storeu_ps(scoreptr + 8, _sum2);
                    _mm_storeu_ps(scoreptr + 12, _sum3);
                    __m128 _max01 = _mm_max_ps(_sum0, _sum1);
                    __m128 _max23 = _mm_max_ps(_sum2, _sum3);
                    _max = _mm_max_ps(_max, _mm_max_ps(_max01, _max23));
                    scoreptr += 16;
                }
                for (; j < max_jj; j++)
                {
                    const float* pQ = queryT;
                    const float* pK = key + (size_t)j * head_dim;
                    __m128 _sum = _mm_setzero_ps();
                    if (pM)
                    {
                        _sum = _mm_loadu_ps(pM);
                        pM += 4;
                    }
                    for (int d = 0; d < head_dim; d++)
                    {
                        _sum = _mm_comp_fmadd_ps(_mm_loadu_ps(pQ), _mm_set1_ps(pK[d]), _sum);
                        pQ += 4;
                    }
                    _max = _mm_max_ps(_max, _sum);
                    _mm_storeu_ps(scoreptr, _sum);
                    scoreptr += 4;
                }

                _block_max = _max;
            }
            else
            {
                const float* packed_key_tile = packed_key_head.row(n / block_n);
                const float* pK = packed_key_tile;
                __m128 _max = _mm_set1_ps(-FLT_MAX);
                int j = 0;
#if __AVX__
#if __AVX512F__
                for (; j + 15 < max_jj; j += 16)
                {
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
                    if (pM)
                    {
                        _sum0 = _mm_loadu_ps(pM);
                        _sum1 = _mm_loadu_ps(pM + 4);
                        _sum2 = _mm_loadu_ps(pM + 8);
                        _sum3 = _mm_loadu_ps(pM + 12);
                        _sum4 = _mm_loadu_ps(pM + 16);
                        _sum5 = _mm_loadu_ps(pM + 20);
                        _sum6 = _mm_loadu_ps(pM + 24);
                        _sum7 = _mm_loadu_ps(pM + 28);
                        _sum8 = _mm_loadu_ps(pM + 32);
                        _sum9 = _mm_loadu_ps(pM + 36);
                        _suma = _mm_loadu_ps(pM + 40);
                        _sumb = _mm_loadu_ps(pM + 44);
                        _sumc = _mm_loadu_ps(pM + 48);
                        _sumd = _mm_loadu_ps(pM + 52);
                        _sume = _mm_loadu_ps(pM + 56);
                        _sumf = _mm_loadu_ps(pM + 60);
                        pM += 64;
                    }
                    const float* pQ = queryT;
                    for (int d = 0; d < head_dim; d++)
                    {
                        __m128 _q = _mm_loadu_ps(pQ);
                        _sum0 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[0]), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[1]), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[2]), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[3]), _sum3);
                        _sum4 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[4]), _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[5]), _sum5);
                        _sum6 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[6]), _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[7]), _sum7);
                        _sum8 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[8]), _sum8);
                        _sum9 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[9]), _sum9);
                        _suma = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[10]), _suma);
                        _sumb = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[11]), _sumb);
                        _sumc = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[12]), _sumc);
                        _sumd = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[13]), _sumd);
                        _sume = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[14]), _sume);
                        _sumf = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[15]), _sumf);
                        pQ += 4;
                        pK += 16;
                    }
                    _mm_storeu_ps(scoreptr, _sum0);
                    _mm_storeu_ps(scoreptr + 4, _sum1);
                    _mm_storeu_ps(scoreptr + 8, _sum2);
                    _mm_storeu_ps(scoreptr + 12, _sum3);
                    _mm_storeu_ps(scoreptr + 16, _sum4);
                    _mm_storeu_ps(scoreptr + 20, _sum5);
                    _mm_storeu_ps(scoreptr + 24, _sum6);
                    _mm_storeu_ps(scoreptr + 28, _sum7);
                    _mm_storeu_ps(scoreptr + 32, _sum8);
                    _mm_storeu_ps(scoreptr + 36, _sum9);
                    _mm_storeu_ps(scoreptr + 40, _suma);
                    _mm_storeu_ps(scoreptr + 44, _sumb);
                    _mm_storeu_ps(scoreptr + 48, _sumc);
                    _mm_storeu_ps(scoreptr + 52, _sumd);
                    _mm_storeu_ps(scoreptr + 56, _sume);
                    _mm_storeu_ps(scoreptr + 60, _sumf);
                    __m128 _max0 = _mm_max_ps(_mm_max_ps(_sum0, _sum4), _mm_max_ps(_sum8, _sumc));
                    __m128 _max1 = _mm_max_ps(_mm_max_ps(_sum1, _sum5), _mm_max_ps(_sum9, _sumd));
                    __m128 _max2 = _mm_max_ps(_mm_max_ps(_sum2, _sum6), _mm_max_ps(_suma, _sume));
                    __m128 _max3 = _mm_max_ps(_mm_max_ps(_sum3, _sum7), _mm_max_ps(_sumb, _sumf));
                    _max = _mm_max_ps(_max, _mm_max_ps(_mm_max_ps(_max0, _max1), _mm_max_ps(_max2, _max3)));
                    scoreptr += 64;
                }
#endif // __AVX512F__
                for (; j + 7 < max_jj; j += 8)
                {
                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();
                    __m128 _sum4 = _mm_setzero_ps();
                    __m128 _sum5 = _mm_setzero_ps();
                    __m128 _sum6 = _mm_setzero_ps();
                    __m128 _sum7 = _mm_setzero_ps();
                    if (pM)
                    {
                        _sum0 = _mm_loadu_ps(pM);
                        _sum1 = _mm_loadu_ps(pM + 4);
                        _sum2 = _mm_loadu_ps(pM + 8);
                        _sum3 = _mm_loadu_ps(pM + 12);
                        _sum4 = _mm_loadu_ps(pM + 16);
                        _sum5 = _mm_loadu_ps(pM + 20);
                        _sum6 = _mm_loadu_ps(pM + 24);
                        _sum7 = _mm_loadu_ps(pM + 28);
                        pM += 32;
                    }
                    const float* pQ = queryT;
                    for (int d = 0; d < head_dim; d++)
                    {
                        __m128 _q = _mm_loadu_ps(pQ);
                        _sum0 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[0]), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[1]), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[2]), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[3]), _sum3);
                        _sum4 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[4]), _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[5]), _sum5);
                        _sum6 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[6]), _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[7]), _sum7);
                        pQ += 4;
                        pK += 8;
                    }
                    _mm_storeu_ps(scoreptr, _sum0);
                    _mm_storeu_ps(scoreptr + 4, _sum1);
                    _mm_storeu_ps(scoreptr + 8, _sum2);
                    _mm_storeu_ps(scoreptr + 12, _sum3);
                    _mm_storeu_ps(scoreptr + 16, _sum4);
                    _mm_storeu_ps(scoreptr + 20, _sum5);
                    _mm_storeu_ps(scoreptr + 24, _sum6);
                    _mm_storeu_ps(scoreptr + 28, _sum7);
                    __m128 _max0 = _mm_max_ps(_sum0, _sum4);
                    __m128 _max1 = _mm_max_ps(_sum1, _sum5);
                    __m128 _max2 = _mm_max_ps(_sum2, _sum6);
                    __m128 _max3 = _mm_max_ps(_sum3, _sum7);
                    _max = _mm_max_ps(_max, _mm_max_ps(_mm_max_ps(_max0, _max1), _mm_max_ps(_max2, _max3)));
                    scoreptr += 32;
                }
#endif // __AVX__
                for (; j + 3 < max_jj; j += 4)
                {
                    const float* pQ = queryT;
                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();
                    if (pM)
                    {
                        _sum0 = _mm_loadu_ps(pM);
                        _sum1 = _mm_loadu_ps(pM + 4);
                        _sum2 = _mm_loadu_ps(pM + 8);
                        _sum3 = _mm_loadu_ps(pM + 12);
                        pM += 16;
                    }
                    for (int d = 0; d < head_dim; d++)
                    {
                        __m128 _q = _mm_loadu_ps(pQ);
                        _sum0 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[0]), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[1]), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[2]), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_q, _mm_set1_ps(pK[3]), _sum3);
                        pQ += 4;
                        pK += 4;
                    }
                    _mm_storeu_ps(scoreptr, _sum0);
                    _mm_storeu_ps(scoreptr + 4, _sum1);
                    _mm_storeu_ps(scoreptr + 8, _sum2);
                    _mm_storeu_ps(scoreptr + 12, _sum3);
                    __m128 _max01 = _mm_max_ps(_sum0, _sum1);
                    __m128 _max23 = _mm_max_ps(_sum2, _sum3);
                    _max = _mm_max_ps(_max, _mm_max_ps(_max01, _max23));
                    scoreptr += 16;
                }
                for (; j < max_jj; j++)
                {
                    const float* pQ = queryT;
                    __m128 _sum = _mm_setzero_ps();
                    if (pM)
                    {
                        _sum = _mm_loadu_ps(pM);
                        pM += 4;
                    }
                    for (int d = 0; d < head_dim; d++)
                    {
                        _sum = _mm_comp_fmadd_ps(_mm_loadu_ps(pQ), _mm_set1_ps(pK[d]), _sum);
                        pQ += 4;
                    }
                    pK += head_dim;
                    _max = _mm_max_ps(_max, _sum);
                    _mm_storeu_ps(scoreptr, _sum);
                    scoreptr += 4;
                }

                _block_max = _max;
            }
            __m128 _m_new = _mm_max_ps(_m, _block_max);
            const __m128 _alpha_active = _mm_cmpneq_ps(_l, _mm_setzero_ps());
            __m128 _alpha = exp_ps(_mm_and_ps(_alpha_active, _mm_sub_ps(_m, _m_new)));
            _alpha = _mm_and_ps(_alpha, _alpha_active);

            scoreptr = scoreT;
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
            if (packed_value.empty())
            {
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
            else
            {
                const float* packed_value_tile = packed_value_head.row(n / block_n);
                const float* pV = packed_value_tile;
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
                        pV += 16;
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
                        pV += 8;
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
                }
#endif // __AVX__
                for (; d + 3 < value_dim; d += 4)
                {
                    __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(outptr), _alpha);
                    __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(outptr + 4), _alpha);
                    __m128 _out2 = _mm_mul_ps(_mm_loadu_ps(outptr + 8), _alpha);
                    __m128 _out3 = _mm_mul_ps(_mm_loadu_ps(outptr + 12), _alpha);
                    const float* pS = scoreT;
                    for (int j = 0; j < max_jj; j++)
                    {
                        __m128 _p = _mm_loadu_ps(pS);
                        _out0 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[0]), _out0);
                        _out1 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[1]), _out1);
                        _out2 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[2]), _out2);
                        _out3 = _mm_comp_fmadd_ps(_p, _mm_set1_ps(pV[3]), _out3);
                        pS += 4;
                        pV += 4;
                    }
                    _mm_storeu_ps(outptr, _out0);
                    _mm_storeu_ps(outptr + 4, _out1);
                    _mm_storeu_ps(outptr + 8, _out2);
                    _mm_storeu_ps(outptr + 12, _out3);
                    outptr += 16;
                }
                for (; d < value_dim; d++)
                {
                    __m128 _out = _mm_mul_ps(_mm_loadu_ps(outptr), _alpha);
                    const float* pS = scoreT;
                    for (int j = 0; j < max_jj; j++)
                    {
                        _out = _mm_comp_fmadd_ps(_mm_loadu_ps(pS), _mm_set1_ps(*pV++), _out);
                        pS += 4;
                    }
                    _mm_storeu_ps(outptr, _out);
                    outptr += 4;
                }
            }
        }

        sdpa_store_output_tile4(outT_tile, top_blob_head, i0x, state_ptr, state_stride, _m, _l, value_dim);
    }
#endif // __SSE2__
    for (; ii + 0 < max_ii; ii += 1)
    {
        const int i0x = i0 + ii;
        float* state_ptr = state.empty() ? 0 : state_base + ii;
        float* output_ptr = top_blob_head.row(i0x);
        float* score = workspace_ptr;
        float* out = score + block_n;

        const Mat key_head = key.channel(g);
        const Mat value_head = value.channel(g);
        const Mat packed_value_head = packed_value.empty() ? Mat() : packed_value.channel(g);
        const Mat mask_head = sdpa_prefill_get_mask_head(attn_mask_blob, q);
        const float* mask = mask_head.empty() ? 0 : mask_head.row(i0x);
        const float* qptr = query_head.row(i0x);

        memset(out, 0, value_dim * sizeof(float));
        float m = -FLT_MAX;
        float l = 0.f;

        for (int n = n_begin; n < n_end; n += block_n)
        {
            const int max_jj = std::min(n_end - n, block_n);
            float block_max = -FLT_MAX;
            for (int j = 0; j < max_jj; j++)
            {
                const float* kptr = key_head.row(n + j);
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sum_avx512 = _mm512_setzero_ps();
#endif // __AVX512F__
                __m256 _sum_avx = _mm256_setzero_ps();
#endif // __AVX__
                __m128 _sum = _mm_setzero_ps();
#endif // __SSE2__
                float sum = 0.f;

                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; i + 15 < head_dim; i += 16)
                    _sum_avx512 = _mm512_fmadd_ps(_mm512_loadu_ps(qptr + i), _mm512_loadu_ps(kptr + i), _sum_avx512);
#endif // __AVX512F__
                for (; i + 7 < head_dim; i += 8)
                    _sum_avx = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr + i), _mm256_loadu_ps(kptr + i), _sum_avx);
#endif // __AVX__
                for (; i + 3 < head_dim; i += 4)
                    _sum = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr + i), _mm_loadu_ps(kptr + i), _sum);
#endif // __SSE2__
                for (; i < head_dim; i++)
                    sum += qptr[i] * kptr[i];

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
                if (mask)
                    s += mask[n + j];
                score[j] = s;
                block_max = std::max(block_max, s);
            }

            float m_new = std::max(m, block_max);
            float alpha = l == 0.f ? 0.f : expf(m - m_new);
            float block_sum;
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

                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; i + 15 < max_jj; i += 16)
                {
                    __m512 _p = _mm512_loadu_ps(score + i);
                    _p = exp512_ps(_mm512_sub_ps(_p, _max_avx512));
                    _mm512_storeu_ps(score + i, _p);
                    _sum_avx512 = _mm512_add_ps(_sum_avx512, _p);
                }
#endif // __AVX512F__
                for (; i + 7 < max_jj; i += 8)
                {
                    __m256 _p = _mm256_loadu_ps(score + i);
                    _p = exp256_ps(_mm256_sub_ps(_p, _max_avx));
                    _mm256_storeu_ps(score + i, _p);
                    _sum_avx = _mm256_add_ps(_sum_avx, _p);
                }
#endif // __AVX__
                for (; i + 3 < max_jj; i += 4)
                {
                    __m128 _p = _mm_loadu_ps(score + i);
                    _p = exp_ps(_mm_sub_ps(_p, _max));
                    _mm_storeu_ps(score + i, _p);
                    _sum = _mm_add_ps(_sum, _p);
                }
#endif // __SSE2__
                for (; i < max_jj; i++)
                {
                    score[i] = expf(score[i] - m_new);
                    sum += score[i];
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

                block_sum = sum;
            }
            l = l * alpha + block_sum;
            m = m_new;
            if (packed_value.empty())
            {
                const float* value = value_head.row(n);
                int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; d + 63 < value_dim; d += 64)
                {
                    __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(out + d), _mm512_set1_ps(alpha));
                    __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(out + d + 16), _mm512_set1_ps(alpha));
                    __m512 _out2 = _mm512_mul_ps(_mm512_loadu_ps(out + d + 32), _mm512_set1_ps(alpha));
                    __m512 _out3 = _mm512_mul_ps(_mm512_loadu_ps(out + d + 48), _mm512_set1_ps(alpha));
                    for (int j = 0; j < max_jj; j++)
                    {
                        const float* vptr = value + (size_t)j * value_dim + d;
                        __m512 _p = _mm512_set1_ps(score[j]);
                        _out0 = _mm512_fmadd_ps(_mm512_loadu_ps(vptr), _p, _out0);
                        _out1 = _mm512_fmadd_ps(_mm512_loadu_ps(vptr + 16), _p, _out1);
                        _out2 = _mm512_fmadd_ps(_mm512_loadu_ps(vptr + 32), _p, _out2);
                        _out3 = _mm512_fmadd_ps(_mm512_loadu_ps(vptr + 48), _p, _out3);
                    }
                    _mm512_storeu_ps(out + d, _out0);
                    _mm512_storeu_ps(out + d + 16, _out1);
                    _mm512_storeu_ps(out + d + 32, _out2);
                    _mm512_storeu_ps(out + d + 48, _out3);
                }

                for (; d + 15 < value_dim; d += 16)
                {
                    __m512 _out = _mm512_mul_ps(_mm512_loadu_ps(out + d), _mm512_set1_ps(alpha));
                    for (int j = 0; j < max_jj; j++)
                        _out = _mm512_fmadd_ps(_mm512_loadu_ps(value + (size_t)j * value_dim + d), _mm512_set1_ps(score[j]), _out);
                    _mm512_storeu_ps(out + d, _out);
                }
#else
                for (; d + 15 < value_dim; d += 16)
                {
                    __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(out + d), _mm256_set1_ps(alpha));
                    __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(out + d + 8), _mm256_set1_ps(alpha));
                    for (int j = 0; j < max_jj; j++)
                    {
                        const float* vptr = value + (size_t)j * value_dim + d;
                        __m256 _p = _mm256_set1_ps(score[j]);
                        _out0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(vptr), _p, _out0);
                        _out1 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(vptr + 8), _p, _out1);
                    }
                    _mm256_storeu_ps(out + d, _out0);
                    _mm256_storeu_ps(out + d + 8, _out1);
                }
#endif // __AVX512F__

                for (; d + 7 < value_dim; d += 8)
                {
                    __m256 _out = _mm256_mul_ps(_mm256_loadu_ps(out + d), _mm256_set1_ps(alpha));
                    for (int j = 0; j < max_jj; j++)
                        _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(value + (size_t)j * value_dim + d), _mm256_set1_ps(score[j]), _out);
                    _mm256_storeu_ps(out + d, _out);
                }
#else
                for (; d + 7 < value_dim; d += 8)
                {
                    __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(out + d), _mm_set1_ps(alpha));
                    __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(out + d + 4), _mm_set1_ps(alpha));
                    for (int j = 0; j < max_jj; j++)
                    {
                        const float* vptr = value + (size_t)j * value_dim + d;
                        __m128 _p = _mm_set1_ps(score[j]);
                        _out0 = _mm_comp_fmadd_ps(_mm_loadu_ps(vptr), _p, _out0);
                        _out1 = _mm_comp_fmadd_ps(_mm_loadu_ps(vptr + 4), _p, _out1);
                    }
                    _mm_storeu_ps(out + d, _out0);
                    _mm_storeu_ps(out + d + 4, _out1);
                }
#endif // __AVX__
                for (; d + 3 < value_dim; d += 4)
                {
                    __m128 _out = _mm_mul_ps(_mm_loadu_ps(out + d), _mm_set1_ps(alpha));
                    for (int j = 0; j < max_jj; j++)
                        _out = _mm_comp_fmadd_ps(_mm_loadu_ps(value + (size_t)j * value_dim + d), _mm_set1_ps(score[j]), _out);
                    _mm_storeu_ps(out + d, _out);
                }
#endif // __SSE2__
                for (; d < value_dim; d++)
                {
                    float sum = out[d] * alpha;
                    for (int j = 0; j < max_jj; j++)
                        sum += score[j] * value[(size_t)j * value_dim + d];
                    out[d] = sum;
                }
            }
            else
            {
                const float* packed_value_tile = packed_value_head.row(n / block_n);
                const float* pV = packed_value_tile;
                int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; d + 15 < value_dim; d += 16)
                {
                    __m512 _out = _mm512_mul_ps(_mm512_loadu_ps(out + d), _mm512_set1_ps(alpha));
                    for (int j = 0; j < max_jj; j++)
                    {
                        _out = _mm512_fmadd_ps(_mm512_loadu_ps(pV), _mm512_set1_ps(score[j]), _out);
                        pV += 16;
                    }
                    _mm512_storeu_ps(out + d, _out);
                }
#endif // __AVX512F__
                for (; d + 7 < value_dim; d += 8)
                {
                    __m256 _out = _mm256_mul_ps(_mm256_loadu_ps(out + d), _mm256_set1_ps(alpha));
                    for (int j = 0; j < max_jj; j++)
                    {
                        _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(pV), _mm256_set1_ps(score[j]), _out);
                        pV += 8;
                    }
                    _mm256_storeu_ps(out + d, _out);
                }
#endif // __AVX__
                for (; d + 3 < value_dim; d += 4)
                {
                    __m128 _out = _mm_mul_ps(_mm_loadu_ps(out + d), _mm_set1_ps(alpha));
                    for (int j = 0; j < max_jj; j++)
                    {
                        _out = _mm_comp_fmadd_ps(_mm_loadu_ps(pV), _mm_set1_ps(score[j]), _out);
                        pV += 4;
                    }
                    _mm_storeu_ps(out + d, _out);
                }
#endif // __SSE2__
                for (; d < value_dim; d++)
                {
                    float sum = out[d] * alpha;
                    for (int j = 0; j < max_jj; j++)
                        sum += *pV++ * score[j];
                    out[d] = sum;
                }
            }
        }

        if (state_ptr)
        {
            state_ptr[0] = m;
            state_ptr[state_stride] = l;
            for (int d = 0; d < value_dim; d++)
                state_ptr[(d + 2) * state_stride] = out[d];
        }
        else
        {
            memcpy(output_ptr, out, value_dim * sizeof(float));
            if (l != 0.f)
            {
                float inv_sum = 1.f / l;
                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _inv_sum_avx512 = _mm512_set1_ps(inv_sum);
                for (; i + 15 < value_dim; i += 16)
                    _mm512_storeu_ps(output_ptr + i, _mm512_mul_ps(_mm512_loadu_ps(output_ptr + i), _inv_sum_avx512));
#endif // __AVX512F__
                __m256 _inv_sum_avx = _mm256_set1_ps(inv_sum);
                for (; i + 7 < value_dim; i += 8)
                    _mm256_storeu_ps(output_ptr + i, _mm256_mul_ps(_mm256_loadu_ps(output_ptr + i), _inv_sum_avx));
#endif // __AVX__
                __m128 _inv_sum = _mm_set1_ps(inv_sum);
                for (; i + 3 < value_dim; i += 4)
                    _mm_storeu_ps(output_ptr + i, _mm_mul_ps(_mm_loadu_ps(output_ptr + i), _inv_sum));
#endif // __SSE2__
                for (; i < value_dim; i++)
                    output_ptr[i] *= inv_sum;
            }
        }
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
        Mat outT_tile = workspace.channel(get_omp_thread_num());
        float* outT = outT_tile;

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

            memset(outT, 0, (size_t)value_dim * 16 * sizeof(float));
            __m512 _l = _mm512_setzero_ps();
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state = partials.channel(task_id * num_kv_chunks + chunk_id);
                const __m512 _partial_l = _mm512_loadu_ps(state + block_m + ii);
                const __mmask16 active = _mm512_cmp_ps_mask(_partial_l, _mm512_setzero_ps(), _CMP_NEQ_OQ);
                const __m512 _partial_scale = _mm512_maskz_mov_ps(active, exp512_ps(_mm512_maskz_sub_ps(active, _mm512_loadu_ps(state + ii), _m)));
                _l = _mm512_fmadd_ps(_partial_l, _partial_scale, _l);
                float* outptr = outT;
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
            sdpa_store_output_tile16(outT_tile, top_blob_head, i0 + ii, 0, block_m, _m, _l, value_dim);
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

            memset(outT, 0, (size_t)value_dim * 8 * sizeof(float));
            __m256 _l = _mm256_setzero_ps();
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state = partials.channel(task_id * num_kv_chunks + chunk_id);
                const __m256 _partial_l = _mm256_loadu_ps(state + block_m + ii);
                const __m256 _active = _mm256_cmp_ps(_partial_l, _mm256_setzero_ps(), _CMP_NEQ_OQ);
                const __m256 _partial_scale = _mm256_and_ps(_active, exp256_ps(_mm256_and_ps(_active, _mm256_sub_ps(_mm256_loadu_ps(state + ii), _m))));
                _l = _mm256_comp_fmadd_ps(_partial_l, _partial_scale, _l);
                float* outptr = outT;
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
            sdpa_store_output_tile8(outT_tile, top_blob_head, i0 + ii, 0, block_m, _m, _l, value_dim);
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

            memset(outT, 0, (size_t)value_dim * 4 * sizeof(float));
            __m128 _l = _mm_setzero_ps();
            for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
            {
                const float* state = partials.channel(task_id * num_kv_chunks + chunk_id);
                const __m128 _partial_l = _mm_loadu_ps(state + block_m + ii);
                const __m128 _active = _mm_cmpneq_ps(_partial_l, _mm_setzero_ps());
                const __m128 _partial_scale = _mm_and_ps(_active, exp_ps(_mm_and_ps(_active, _mm_sub_ps(_mm_loadu_ps(state + ii), _m))));
                _l = _mm_comp_fmadd_ps(_partial_l, _partial_scale, _l);
                float* outptr = outT;
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
            sdpa_store_output_tile4(outT_tile, top_blob_head, i0 + ii, 0, block_m, _m, _l, value_dim);
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

            float* outptr = top_blob_head.row(i0 + ii);
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

static int sdpa_prefill_fp32(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
{
    const int query_seqlen = query.h;
    const int num_query_heads = query.c;
    const int num_kv_heads = key.c;
    const int key_seqlen = key.h;
    const int value_dim = value.w;
    const int num_query_heads_per_kv_head = num_query_heads / num_kv_heads;
    const int num_threads = std::max(opt.num_threads, 1);
    const int block_m = sdpa_prefill_block_m(query_seqlen, num_query_heads, num_kv_heads, value_dim, num_threads);
    const int num_mask_heads = attn_mask_blob.dims == 3 ? attn_mask_blob.c : 1;
    const bool use_packed_mask = !attn_mask_blob.empty() && block_m >= 4;
    const int key_reuse = (query_seqlen + block_m - 1) / block_m * num_query_heads_per_kv_head;
    const bool use_packed_key = query_seqlen >= 4 && key_reuse >= 4;
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
    const bool use_packed_value = key_reuse >= value_pack_reuse;
    const int block_n = sdpa_prefill_block_n(query.w, value_dim, key_seqlen, query_seqlen, 4, 4, use_packed_mask ? 4 : 0, block_m);
    const int state_stride = block_m;
    const int num_mblocks = (query_seqlen + block_m - 1) / block_m;
    const int num_tasks = num_query_heads * num_mblocks;

    const int num_key_blocks = (key_seqlen + block_n - 1) / block_n;

    Mat packed_key;
    if (use_packed_key)
    {
        packed_key.create(key.w * block_n, num_key_blocks, num_kv_heads, 4u, opt.workspace_allocator);
        if (packed_key.empty())
            return -100;

        sdpa_pack_key_fp32(key, packed_key, block_n, opt);
    }

    Mat packed_value;
    if (use_packed_value)
    {
        packed_value.create(value_dim * block_n, num_key_blocks, num_kv_heads, 4u, opt.workspace_allocator);
        if (packed_value.empty())
            return -100;

        sdpa_pack_value_fp32(value, packed_value, block_n, opt);
    }

    Mat packed_mask;
    if (use_packed_mask)
    {
        packed_mask.create(key_seqlen * block_m, num_mblocks, num_mask_heads, 4u, opt.workspace_allocator);
        if (packed_mask.empty())
            return -100;

        sdpa_pack_mask_fp32(attn_mask_blob, packed_mask, block_m, opt);
    }

    int num_kv_chunks = 1;
    if (num_tasks < num_threads && key_seqlen >= 512)
    {
        num_kv_chunks = std::min((num_threads + num_tasks - 1) / num_tasks, num_key_blocks);
        num_kv_chunks = std::max(num_kv_chunks, 1);
    }

    Mat packed_query;
    if (num_kv_chunks > 1)
    {
        packed_query.create(query.w * block_m, 1, num_tasks, 4u, opt.workspace_allocator);
        if (packed_query.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int task_id = 0; task_id < num_tasks; task_id++)
        {
            const int q = task_id / num_mblocks;
            const int i0 = task_id % num_mblocks * block_m;
            const int max_ii = std::min(query_seqlen - i0, block_m);
            const Mat query_head = query.channel(q);
            Mat queryT = packed_query.channel(task_id);
            sdpa_pack_query_fp32(query_head, queryT, i0, max_ii, scale);
        }
    }

    const int workspace_size = (block_m * (block_n + query.w + value_dim) + 15) / 16 * 16;
    Mat workspace(workspace_size, 1, num_threads, 4u, opt.workspace_allocator);
    if (workspace.empty())
        return -100;

    Mat partials;
    if (num_kv_chunks > 1)
    {
        partials.create((value_dim + 2) * block_m, 1, num_tasks * num_kv_chunks, 4u, opt.workspace_allocator);
        if (partials.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ti = 0; ti < num_tasks * num_kv_chunks; ti++)
    {
        const int task_id = ti / num_kv_chunks;
        const int chunk_id = ti % num_kv_chunks;
        const int q = task_id / num_mblocks;
        const int mblock_id = task_id % num_mblocks;
        const int i0 = mblock_id * block_m;
        const int max_ii = std::min(query_seqlen - i0, block_m);
        const int g = q / num_query_heads_per_kv_head;
        const int n_begin = chunk_id * num_key_blocks / num_kv_chunks * block_n;
        const int n_end = std::min((chunk_id + 1) * num_key_blocks / num_kv_chunks * block_n, key_seqlen);

        Mat workspace_tile = workspace.channel(get_omp_thread_num());
        Mat state;
        Mat packed_query_tile;
        Mat packed_mask_tile;
        if (num_kv_chunks > 1)
        {
            state = partials.channel(ti);
            packed_query_tile = packed_query.channel(task_id);
        }
        if (!packed_mask.empty())
        {
            Mat packed_mask_head = packed_mask.channel(packed_mask.c > 1 ? q : 0);
            packed_mask_tile = packed_mask_head.row_range(mblock_id, 1);
        }
        sdpa_flash_attention_tile_fp32(query, key, packed_key, value, packed_value, attn_mask_blob, packed_mask_tile, top_blob, scale, q, g, i0, max_ii, n_begin, n_end, block_n, state_stride, packed_query_tile, workspace_tile, state);
    }

    if (num_kv_chunks > 1)
        sdpa_prefill_reduce(partials, top_blob, workspace, num_tasks, num_mblocks, block_m, num_kv_chunks, query_seqlen, value_dim, opt);

    return 0;
}
